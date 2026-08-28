"""Backward-compatibility shim over the ``ppo`` package.

The PPO pipeline now lives in the ``ppo/`` package (see PPO_PIPELINE.md);
this module keeps the original train_ppo.py API importable so existing
notebooks (e.g. testing_nb.ipynb) keep working:

    from train_ppo import PPOTrainingConfig, PPOTrainer, RewardWeights
    from train_ppo import FlyingBaseStationEnv, run_sinr_evaluation

Semantics preserved:
    - the env is the legacy single-band world (old observation/action layout,
      old reward formula) — saved agents from ppo_runs/ load unchanged;
    - PPOTrainer uses the original SB3 defaults (n_steps=2048, ...) and the
      shared ppo_logs/monitor.csv, exactly like before.

New work should use:

    from ppo import ExperimentConfig, train, evaluate_run

Notes:
    - ``tensorboard_log`` is accepted but ignored (the logging stack is
      CSV-based now — see ppo/run_logging.py).
    - The env internals changed: the MATLAB world lives engine-side, so the
      raw handles (antenna_fbs, mbs_cache, ...) are no longer Python
      attributes. Use ``env.evaluate_state(state)`` for one-off SINR
      evaluations; ``run_sinr_evaluation`` below still accepts the original
      raw-handle signature for any legacy code that kept its own engine.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from ppo.config import (
    BandConfig,
    EnvConfig,
    RewardConfig,
    RewardWeights,  # re-export (same fields as the original dataclass)
    WorldConfig,
)
from ppo.env import FlyingBaseStationEnv as _NewEnv
from ppo.matlab_bridge import MatlabSinrBackend
from ppo.paths import REPO_ROOT

__all__ = [
    "RewardWeights",
    "PPOTrainingConfig",
    "FlyingBaseStationEnv",
    "PPOTrainer",
    "run_sinr_evaluation",
]

warnings.warn(
    "train_ppo is a compatibility shim; new code should use the ppo package "
    "(from ppo import ExperimentConfig, train, evaluate_run).",
    DeprecationWarning,
    stacklevel=2,
)


@dataclass
class PPOTrainingConfig:
    """Original flat config object (legacy single-band world)."""

    # None -> resolved from PPO_QUADRIGA_PATH / secrets.env (see ppo/paths.py)
    quadriga_path: Optional[str] = None
    # None -> the repo's matlab/ folder. Only ever added to the MATLAB path.
    repo_path: Optional[str] = None
    num_fbs: int = 1
    num_users: int = 1000
    sinr_threshold: float = 5.0
    max_episode_steps: int = 25
    total_timesteps: int = 10_000
    verbose: int = 1
    tensorboard_log: Optional[str] = None  # accepted, ignored (CSV stack)
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    ent_coef: float = 0.0
    learning_rate: float = 3e-4
    action_scale: float = 0.05
    max_users: Optional[int] = None
    world_width: Optional[float] = None
    world_height: Optional[float] = None
    num_mbs: Optional[int] = None
    mbs_locations: Optional[Sequence[Tuple[float, float]]] = None


def _legacy_env_config(
    num_fbs: int,
    num_users: int,
    max_users: Optional[int],
    sinr_threshold: float,
    reward_weights: Optional[RewardWeights],
    max_episode_steps: int,
    action_scale: float,
    world_width: Optional[float],
    world_height: Optional[float],
    num_mbs: Optional[int],
    mbs_locations,
) -> EnvConfig:
    if mbs_locations is not None and num_mbs is None:
        num_mbs = len(list(mbs_locations))
    world = WorldConfig(
        width=world_width if world_width is not None else 2000.0,
        height=world_height if world_height is not None else 1500.0,
        num_mbs=num_mbs if num_mbs is not None else 1,
        mbs_locations=mbs_locations,
        num_users=num_users,
        max_users=max_users,
        sinr_threshold=sinr_threshold,
    )
    return EnvConfig(
        num_fbs=num_fbs,
        world=world,
        band=BandConfig(mode="legacy"),
        reward=RewardConfig(mode="legacy_blend",
                            weights=reward_weights or RewardWeights()),
        max_episode_steps=max_episode_steps,
        action_scale=action_scale,
    )


class FlyingBaseStationEnv(_NewEnv):
    """Old-signature wrapper over ppo.env.FlyingBaseStationEnv (legacy mode)."""

    def __init__(
        self,
        quadriga_path: Optional[str] = None,
        repo_path: Optional[str] = None,
        num_fbs: int = 1,
        num_users: int = 1000,
        max_users: Optional[int] = None,
        sinr_threshold: float = 5.0,
        reward_weights: Optional[RewardWeights] = None,
        max_episode_steps: int = 25,
        action_scale: float = 0.05,
        world_width: Optional[float] = None,
        world_height: Optional[float] = None,
        num_mbs: Optional[int] = None,
        mbs_locations: Optional[Sequence[Tuple[float, float]]] = None,
    ):
        config = _legacy_env_config(
            num_fbs, num_users, max_users, sinr_threshold, reward_weights,
            max_episode_steps, action_scale, world_width, world_height,
            num_mbs, mbs_locations,
        )
        backend = MatlabSinrBackend(
            config.world, config.band,
            quadriga_path=quadriga_path, repo_path=repo_path,
        )
        super().__init__(config, backend=backend)
        self._shim_backend = backend

    # ---- legacy attribute surface ------------------------------------ #
    @property
    def W(self) -> float:
        return self.world.width

    @property
    def H(self) -> float:
        return self.world.height

    @property
    def num_users(self) -> int:
        return self.world.num_users

    @property
    def sinr_threshold(self) -> float:
        return self.world.sinr_threshold

    @property
    def area_bounds(self):
        return (0.0, self.world.width, 0.0, self.world.height)

    @property
    def mbs_x(self) -> np.ndarray:
        """True MBS x coordinates. (The old env exposed the swapped-frame
        row here, i.e. the y values — that long-standing transposition is
        fixed; plotting code using these values now gets correct positions.)"""
        return self.backend.mbs_x

    @property
    def mbs_y(self) -> np.ndarray:
        return self.backend.mbs_y

    @property
    def _eng(self):
        return self.backend.session.engine

    def evaluate_state(self, state, collect_user_positions: bool = True) -> dict:
        """One-off SINR evaluation of a flat state vector (replacement for
        the old manual run_sinr_evaluation(env._eng, env.antenna_fbs, ...))."""
        state = np.asarray(state, dtype=float).reshape(self.num_fbs, 5)
        result = self.backend.evaluate(
            state[:, :4], state[:, 4],
            collect_user_positions=collect_user_positions,
        )
        info = result.as_metrics_dict()
        if result.user_positions is not None:
            info["user_positions"] = result.user_positions
        return info

    def close(self):
        super().close()
        if getattr(self, "_shim_backend", None) is not None:
            self._shim_backend.close()
            self._shim_backend = None


class PPOTrainer:
    """Original trainer: SB3 defaults + shared ppo_logs/monitor.csv."""

    def __init__(self, config: PPOTrainingConfig):
        self.config = config
        if config.tensorboard_log:
            print(
                "[train_ppo] tensorboard_log is ignored — the pipeline logs to "
                "CSV now (use `from ppo import train` for full run logging)."
            )
        raw_env = FlyingBaseStationEnv(
            quadriga_path=config.quadriga_path,
            repo_path=config.repo_path,
            num_fbs=config.num_fbs,
            num_users=config.num_users,
            max_users=config.max_users,
            sinr_threshold=config.sinr_threshold,
            reward_weights=config.reward_weights,
            max_episode_steps=config.max_episode_steps,
            action_scale=config.action_scale,
            world_width=config.world_width,
            world_height=config.world_height,
            num_mbs=config.num_mbs,
            mbs_locations=config.mbs_locations,
        )
        monitor_path = REPO_ROOT / "ppo_logs" / "monitor.csv"
        monitor_path.parent.mkdir(parents=True, exist_ok=True)
        self.env = Monitor(raw_env, filename=str(monitor_path), allow_early_resets=True)
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=config.verbose,
            ent_coef=config.ent_coef,
            learning_rate=config.learning_rate,
        )

    def train(self):
        self.model.learn(total_timesteps=self.config.total_timesteps)
        return self

    def save(self, output_path: Optional[str] = None):
        base_path = (
            Path(output_path)
            if output_path is not None
            else REPO_ROOT / "ppo_fbs_agent"
        )
        if self.config.num_fbs > 1:
            base_path = base_path.with_name(f"{base_path.name}_nfbs{self.config.num_fbs}")
        path_str = str(base_path)
        self.model.save(path_str)
        metadata = {"reward_weights": asdict(self.config.reward_weights)}
        metadata_path = base_path.parent / f"{base_path.name}_reward_weights.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return path_str

    def close(self):
        if self.env:
            self.env.close()


# --------------------------------------------------------------------------- #
# Original raw-handle SINR call, kept verbatim for legacy code that manages
# its own engine + antenna/cache handles. New code: backend.evaluate(...).
# --------------------------------------------------------------------------- #
def _matlab_scalar(value: float):
    import matlab

    return matlab.double([float(value)])


def run_sinr_evaluation(
    eng,
    antenna_fbs,
    antenna_mbs,
    mbs_cache,
    tx_vectors: np.ndarray,
    power_status,
    area_bounds: Tuple[float, float, float, float],
    num_users: int,
    sinr_threshold: float,
    contains_mbs: bool,
    mbs_x,
    mbs_y,
    mbs_height,
    mbs_power,
    num_fbs: Optional[int] = None,
) -> Tuple[int, float, float, int, int, np.ndarray]:
    """Call MATLAB SINREvaluation with raw handles (original signature)."""
    import matlab

    tx_vectors = np.asarray(tx_vectors, dtype=float)
    if tx_vectors.ndim == 1:
        tx_vectors = tx_vectors.reshape(1, -1)
    if tx_vectors.shape[1] != 4:
        raise ValueError("tx_vectors must have shape (num_fbs, 4)")
    num_fbs = num_fbs or tx_vectors.shape[0]

    power_status_arr = np.asarray(power_status, dtype=float).reshape(-1)
    if power_status_arr.size == 1:
        power_status_arr = np.repeat(power_status_arr, tx_vectors.shape[0])
    if power_status_arr.size != tx_vectors.shape[0]:
        raise ValueError("power_status must broadcast to num_fbs entries")

    x_min, x_max, y_min, y_max = area_bounds

    result = eng.SINREvaluation(
        antenna_fbs,
        matlab.double([power_status_arr.tolist()]),
        matlab.double([tx_vectors[:, 0].tolist()]),
        matlab.double([tx_vectors[:, 1].tolist()]),
        matlab.double([tx_vectors[:, 2].tolist()]),
        _matlab_scalar(num_fbs),
        matlab.double([tx_vectors[:, 3].tolist()]),
        mbs_x,
        mbs_y,
        mbs_height,
        mbs_power,
        _matlab_scalar(x_min),
        _matlab_scalar(x_max),
        _matlab_scalar(y_min),
        _matlab_scalar(y_max),
        _matlab_scalar(num_users),
        _matlab_scalar(sinr_threshold),
        matlab.logical([contains_mbs]),
        antenna_mbs,
        mbs_cache,
        nargout=7,
    )

    user_positions = np.array(result[0]._data).reshape(result[0].size[::-1]).T
    total_connected = int(result[2])
    total_power = float(result[3])
    avg_rate = float(result[4])
    fbs_connected = int(result[5])
    mbs_connected = int(result[6])

    return total_connected, total_power, avg_rate, fbs_connected, mbs_connected, user_positions


def main():
    config = PPOTrainingConfig()
    trainer = PPOTrainer(config)
    try:
        trainer.train()
        trainer.save()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
