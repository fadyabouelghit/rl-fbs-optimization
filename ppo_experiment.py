"""Backward-compatibility shim over the ``ppo`` package.

The old ppo_experiment.py harness (X-Y-Z codes, train/test/list) now
delegates to the ppo package. Kept importable for existing notebooks and
plot_trajectories.py; new code should use:

    from ppo import ExperimentConfig, train, evaluate_run

Differences from the original:
    - train() writes a full v2 run directory (see PPO_PIPELINE.md) instead of
      run_NNN; the returned path is the new run dir.
    - test() delegates to ppo.evaluate.evaluate_run (artifacts land under
      <run_dir>/evals/ and, with log_trajectory=True, are mirrored to the old
      test_logs/<code>/ layout for plot_trajectories.py).
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ppo import evaluate_run as _evaluate_run
from ppo import list_runs as _list_runs
from ppo import load_run, resolve_run as _resolve_run, train as _train
from ppo.config import CONFIGS as _NEW_CONFIGS
from ppo.config import SCENARIOS as _NEW_SCENARIOS
from ppo.config import ExperimentConfig as _NewExperimentConfig
from ppo.paths import LEDGER_PATH as TRAINING_LOG
from ppo.paths import REPO_ROOT, RUNS_DIR, TEST_LOG_DIR

warnings.warn(
    "ppo_experiment is a compatibility shim; new code should use the ppo "
    "package (from ppo import ExperimentConfig, train, evaluate_run).",
    DeprecationWarning,
    stacklevel=2,
)

PPO_LOGS_DIR = REPO_ROOT / "ppo_logs"

# Old-style scenario dicts (world_width/world_height keys) for legacy readers.
SCENARIOS: dict[int, dict] = {
    y: {
        "num_mbs": s["num_mbs"],
        "mbs_locations": s["mbs_locations"],
        "world_width": s["width"],
        "world_height": s["height"],
    }
    for y, s in _NEW_SCENARIOS.items()
}
CONFIGS = _NEW_CONFIGS


@dataclass
class ExperimentConfig:
    """Old flat experiment config (legacy single-band world)."""

    num_fbs: int
    scenario_id: int
    config_id: int
    ent_coef: float = 0.7
    action_scale: float = 0.9
    learning_rate: float = 1e-4
    total_timesteps: int = 3_000
    max_episode_steps: int = 30
    beta: float = 0.8
    fbs_weight: float = 0.4
    fbs_exponent: float = 1.0

    @property
    def code(self) -> str:
        return f"{self.num_fbs}-{self.scenario_id}-{self.config_id}"

    @property
    def scenario(self) -> dict:
        return SCENARIOS[self.scenario_id]

    @property
    def gamma(self) -> float:
        return CONFIGS[self.config_id]["gamma"]

    @classmethod
    def from_code(cls, code: str, **overrides) -> "ExperimentConfig":
        x, y, z = (int(p) for p in code.split("-"))
        if y not in SCENARIOS:
            raise ValueError(f"Unknown scenario id {y}; valid: {sorted(SCENARIOS)}")
        if z not in CONFIGS:
            raise ValueError(f"Unknown config id {z}; valid: {sorted(CONFIGS)}")
        return cls(num_fbs=x, scenario_id=y, config_id=z, **overrides)

    def to_new(self) -> _NewExperimentConfig:
        from ppo.config import RewardConfig, RewardWeights

        return _NewExperimentConfig.from_code(
            self.code,
            total_timesteps=self.total_timesteps,
            max_episode_steps=self.max_episode_steps,
            ent_coef=self.ent_coef,
            action_scale=self.action_scale,
            learning_rate=self.learning_rate,
            reward=RewardConfig(
                mode="legacy_blend",
                weights=RewardWeights(
                    beta=self.beta,
                    gamma=self.gamma,
                    fbs_weight=self.fbs_weight,
                    fbs_exponent=self.fbs_exponent,
                ),
            ),
            # Match the old harness: SB3 default rollout buffer.
            ppo_overrides={"n_steps": 2048},
        )


# --------------------------------------------------------------------------- #
# Run-directory utilities (delegate to ppo.runs)
# --------------------------------------------------------------------------- #
def latest_run_path(root: Path = RUNS_DIR, stem: str = "run") -> Path:
    return _resolve_run("latest", root)


def run_dir_by_name(name: str, root: Path = RUNS_DIR) -> Path:
    return _resolve_run(name, root)


def resolve_run(run, root: Path = RUNS_DIR) -> Path:
    return _resolve_run(run, root)


def list_runs(root: Path = RUNS_DIR) -> pd.DataFrame:
    return _list_runs(root)


def model_basename(num_fbs: int) -> str:
    return "ppo_fbs_agent" if num_fbs <= 1 else f"ppo_fbs_agent_nfbs{num_fbs}"


# --------------------------------------------------------------------------- #
# Train / test (delegate to the new pipeline)
# --------------------------------------------------------------------------- #
def train(exp: ExperimentConfig) -> Path:
    run_dir = _train(exp.to_new())
    print(f"[train] saved run -> {run_dir} (code {exp.code})")
    return run_dir


def flatten_states(state_history, num_fbs: int) -> pd.DataFrame:
    """Original 5-gene flattener (legacy observation layout)."""
    arr = np.asarray(state_history, dtype=float)
    per_fbs_dim = 5 * num_fbs
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < per_fbs_dim:
        raise ValueError(
            f"Expected at least {per_fbs_dim} entries per state, got {arr.shape[1]}"
        )
    base = arr[:, :per_fbs_dim].reshape(-1, num_fbs, 5)
    cols = [
        (f"fbs{idx}", name)
        for idx in range(num_fbs)
        for name in ("x", "y", "height", "power", "power_status")
    ]
    df = pd.DataFrame(base.reshape(len(base), -1),
                      columns=pd.MultiIndex.from_tuples(cols))
    if arr.shape[1] >= per_fbs_dim + 2:
        df[("global", "fbs_delta")] = arr[:, per_fbs_dim]
        df[("global", "total_delta")] = arr[:, per_fbs_dim + 1]
    return df


def test(
    run="latest",
    custom_state: Optional[np.ndarray] = None,
    max_episode_steps: int = 100,
    action_scale_override: Optional[float] = None,
    deterministic: bool = True,
    log_trajectory: bool = True,
    exp_override: Optional[ExperimentConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Old test() contract: one rollout -> (state_df, metrics_df)."""
    new_exp = exp_override.to_new() if exp_override is not None else None
    report = _evaluate_run(
        run,
        episodes=1,
        deterministic=deterministic,
        initial_state=custom_state,
        max_episode_steps=max_episode_steps,
        action_scale=action_scale_override,
        exp_override=new_exp,
        save=log_trajectory,
        make_plots=log_trajectory,
        mirror_test_logs=log_trajectory,
    )
    ep = report.episodes[0]
    return ep.state_df, ep.metrics_df


# --------------------------------------------------------------------------- #
# Saved-test discovery (pure file IO over test_logs/, unchanged)
# --------------------------------------------------------------------------- #
def _split_test_stem(stem: str) -> Tuple[str, str]:
    parts = stem.rsplit("_", 2)
    if len(parts) >= 3:
        return "_".join(parts[:-2]), "_".join(parts[-2:])
    return "", stem


def list_test_results(code: Optional[str] = None) -> pd.DataFrame:
    rows = []
    if not TEST_LOG_DIR.exists():
        return pd.DataFrame(columns=["code", "run_name", "timestamp", "stem", "path"])
    search_dirs = [TEST_LOG_DIR / code] if code else [
        d for d in TEST_LOG_DIR.iterdir() if d.is_dir()
    ]
    for d in search_dirs:
        if not d.exists():
            continue
        for meta_path in sorted(d.glob("*_meta.json")):
            stem = meta_path.name.replace("_meta.json", "")
            run_name, ts = _split_test_stem(stem)
            rows.append({
                "code": d.name, "run_name": run_name, "timestamp": ts,
                "stem": stem, "path": str(d / stem),
            })
    return pd.DataFrame(rows)


def load_test_result(
    code: Optional[str] = None,
    run_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    latest: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = list_test_results(code=code)
    if df.empty:
        raise FileNotFoundError(
            f"No test results under {TEST_LOG_DIR}" + (f"/{code}" if code else "")
        )
    if run_name:
        df = df[df["run_name"] == run_name]
    if timestamp:
        df = df[df["timestamp"] == timestamp]
    if df.empty:
        raise FileNotFoundError(
            f"No test results match: code={code!r} run_name={run_name!r} "
            f"timestamp={timestamp!r}"
        )
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    pick = df.iloc[0] if latest else df.iloc[-1]
    base = Path(pick["path"])
    state_df = pd.read_csv(base.with_name(base.name + "_trajectory.csv"),
                           header=[0, 1], index_col=0)
    metrics_df = pd.read_csv(base.with_name(base.name + "_metrics.csv"))
    with base.with_name(base.name + "_meta.json").open() as f:
        meta = json.load(f)
    return state_df, metrics_df, meta


def load_experiment(run_dir: Path) -> Optional[ExperimentConfig]:
    """Best-effort reconstruction of the old flat config from any run dir."""
    handle = load_run(run_dir)
    if handle.exp is None:
        return None
    e = handle.exp
    return ExperimentConfig(
        num_fbs=e.num_fbs,
        scenario_id=e.scenario_id,
        config_id=e.config_id,
        ent_coef=e.ppo.ent_coef,
        action_scale=e.env.action_scale,
        learning_rate=e.ppo.learning_rate,
        total_timesteps=e.ppo.total_timesteps,
        max_episode_steps=e.env.max_episode_steps,
        beta=e.env.reward.weights.beta,
        fbs_weight=e.env.reward.weights.fbs_weight,
        fbs_exponent=e.env.reward.weights.fbs_exponent,
    )


def write_experiment_metadata(run_dir: Path, exp: ExperimentConfig) -> None:
    payload = {**asdict(exp), "code": exp.code, "gamma": exp.gamma,
               "scenario": exp.scenario}
    with (Path(run_dir) / "experiment_config.json").open("w") as f:
        json.dump(payload, f, indent=2, default=str)


if __name__ == "__main__":
    print(__doc__)
    print("Use the new CLI instead:  python -m ppo --help")
