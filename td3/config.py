"""TD3 configuration.

An experiment has two halves. The environment half (world, band, reward and
env settings) is exactly ``ppo.config.EnvConfig``, built through the same
``ExperimentConfig.from_code`` presets, so a TD3 run and a PPO run given the
same code and flags see an identical environment. The algorithm half is
``TD3Params`` -- nothing from ``PPOParams`` is used.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List, Optional, Tuple

from ppo.config import EnvConfig, ExperimentConfig, experiment_from_json_dict

TD3_CONFIG_VERSION = 1


def binary_action_indices(env: EnvConfig) -> List[int]:
    """Action dimensions that drive a *binary* gene, for the layout in
    ``ppo.env.FlyingBaseStationEnv``.

    Each FBS occupies a block of genes: 0-3 are continuous (x, y, z, power),
    4 is the power on/off switch, and 5 is the band flag when the agent
    controls it. Per-MBS capacity toggles, when enabled, trail the FBS
    blocks. Those binary dims are what ``_binary_update`` thresholds, so the
    action-to-reward map is a step function along them and a *continuous*
    dim's exploration noise is the wrong size for them.

    Deliberately duplicated from ``ddpg.config`` rather than imported: the
    two prototypes stay independent of each other, and DDPG is the algorithm
    *below* TD3, so a td3 -> ddpg import would invert the relationship.
    """
    genes = 5 + (1 if env.band.agent_controls_fbs_band else 0)
    idx: List[int] = []
    for f in range(env.num_fbs):
        idx.append(f * genes + 4)
        if env.band.agent_controls_fbs_band:
            idx.append(f * genes + 5)
    n_genes = env.num_fbs * genes
    if env.band.agent_controls_mbs_capacity:
        idx.extend(range(n_genes, n_genes + env.world.num_mbs))
    return idx


@dataclass
class TD3Params:
    """TD3 hyperparameters + run budget.

    Defaults are Stable-Baselines3's TD3 defaults, plus Gaussian exploration
    noise. ``action_noise_std`` is in action units: actions live in [-1, 1],
    so under ``action_mode='delta'`` a std of 0.1 jitters a position gene by
    about 0.1 * action_scale * range per step, and it rarely flips a sticky
    binary gene (a flip needs |a| >= 0.5).

    ``binary_noise_std`` overrides that std on the binary dims (see
    ``binary_action_indices``), which need a different size entirely. TD3's
    actor is deterministic and saturates its tanh, and from an output of
    -1.0 a *sticky* gene only flips when the noise pushes the action to
    >= +0.5, i.e. when the draw exceeds 1.5: at std 0.1 that never happens,
    at 0.3 it is ~3e-7 per step, at 1.0 it is ~6.7%. TD3 shares this blind
    spot with DDPG -- the twin critics and delayed updates do nothing about
    it -- so the knob exists here for the same reason. Left as None, every
    dim shares ``action_noise_std`` (plain SB3 behaviour).

    ``gamma`` is the RL discount factor (not the cost-function gamma in
    ``RewardWeights``).
    """

    total_timesteps: int = 10_000
    learning_rate: float = 1e-3
    buffer_size: int = 1_000_000
    learning_starts: int = 100
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    action_noise_std: float = 0.1
    binary_noise_std: Optional[float] = None
    net_arch: Optional[Tuple[int, ...]] = None   # None -> SB3 default (400, 300)
    # Running reward normalization (SB3 VecNormalize), as in PPOParams. The
    # replay buffer keeps raw rewards; SB3 normalizes them when sampling.
    normalize_reward: bool = False
    seed: Optional[int] = 0
    n_envs: int = 1
    checkpoint_every: int = 0     # timesteps between checkpoints; 0 disables
    eval_every: int = 0           # timesteps between in-training evals; 0 disables
    eval_episodes: int = 3
    log_flush_every: int = 256
    verbose: int = 1

    def __post_init__(self):
        if self.n_envs < 1:
            raise ValueError("n_envs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.policy_delay < 1:
            raise ValueError("policy_delay must be >= 1")
        if min(self.action_noise_std, self.target_policy_noise, self.target_noise_clip) < 0:
            raise ValueError("action_noise_std, target_policy_noise and target_noise_clip must be >= 0")
        if self.binary_noise_std is not None and self.binary_noise_std < 0:
            raise ValueError("binary_noise_std must be >= 0")
        if self.net_arch is not None:
            self.net_arch = tuple(int(n) for n in self.net_arch)
            if not self.net_arch or min(self.net_arch) < 1:
                raise ValueError("net_arch must list positive layer sizes")


@dataclass
class TD3Experiment:
    """The user-facing bundle: X-Y-Z code + env config + TD3 parameters."""

    num_fbs: int = 1          # X
    scenario_id: int = 1      # Y
    config_id: int = 1        # Z
    env: EnvConfig = field(default_factory=EnvConfig)
    td3: TD3Params = field(default_factory=TD3Params)
    tag: Optional[str] = None

    @property
    def code(self) -> str:
        return f"{self.num_fbs}-{self.scenario_id}-{self.config_id}"

    @property
    def label(self) -> str:
        parts = [self.code]
        if self.env.band.is_multi:
            parts.append("mb")
        parts.append("td3")
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)

    @classmethod
    def from_code(
        cls,
        code: str,
        *,
        band=None,
        reward=None,
        world_overrides: Optional[dict] = None,
        env_overrides: Optional[dict] = None,
        td3_overrides: Optional[dict] = None,
        tag: Optional[str] = None,
        total_timesteps: Optional[int] = None,
        max_episode_steps: Optional[int] = None,
        action_scale: Optional[float] = None,
        learning_rate: Optional[float] = None,
        seed: Optional[int] = None,
        n_envs: Optional[int] = None,
    ) -> "TD3Experiment":
        # Build through ppo's presets and keep only the env half, so scenario
        # and cost-config ids resolve exactly as they do for PPO runs.
        base = ExperimentConfig.from_code(
            code,
            band=band,
            reward=reward,
            world_overrides=world_overrides,
            env_overrides=env_overrides,
            max_episode_steps=max_episode_steps,
            action_scale=action_scale,
        )
        td3_kwargs = dict(td3_overrides or {})
        for key, val in (
            ("total_timesteps", total_timesteps),
            ("learning_rate", learning_rate),
            ("seed", seed),
            ("n_envs", n_envs),
        ):
            if val is not None:
                td3_kwargs[key] = val
        return cls(
            num_fbs=base.num_fbs,
            scenario_id=base.scenario_id,
            config_id=base.config_id,
            env=base.env,
            td3=TD3Params(**td3_kwargs),
            tag=tag,
        )

    def to_json_dict(self) -> dict:
        return {
            "td3_config_version": TD3_CONFIG_VERSION,
            "algo": "td3",
            "code": self.code,
            "num_fbs": self.num_fbs,
            "scenario_id": self.scenario_id,
            "config_id": self.config_id,
            "tag": self.tag,
            "env": asdict(self.env),
            "td3": asdict(self.td3),
        }

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_json_dict(), indent=2, default=str))


def td3_experiment_from_json_dict(data: dict) -> TD3Experiment:
    """Load a TD3Experiment from a serialized dict (``to_json_dict`` output,
    or a run's ``experiment_config.json``)."""
    if data.get("algo") != "td3":
        raise ValueError(f"not a TD3 experiment config (algo={data.get('algo')!r})")
    # The env block is parsed by ppo's own loader, so both packages read env
    # settings (defaults for missing fields included) identically.
    shell = experiment_from_json_dict({
        "config_version": 2,
        "num_fbs": data["num_fbs"],
        "scenario_id": data["scenario_id"],
        "config_id": data["config_id"],
        "env": data.get("env", {}),
    })
    known = {f.name for f in fields(TD3Params)}
    td3 = TD3Params(**{k: v for k, v in (data.get("td3") or {}).items() if k in known})
    return TD3Experiment(
        num_fbs=data["num_fbs"],
        scenario_id=data["scenario_id"],
        config_id=data["config_id"],
        env=shell.env,
        td3=td3,
        tag=data.get("tag"),
    )


def load_td3_experiment(path: Path | str) -> TD3Experiment:
    return td3_experiment_from_json_dict(json.loads(Path(path).read_text()))
