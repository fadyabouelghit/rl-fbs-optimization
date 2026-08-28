"""Run discovery and loading — new runs, old harness runs, and bare runs.

Three generations of run directories live under ``ppo_runs/``:

- **v2** (this package): ``run_<ts>_<code>[...]/`` with a versioned
  ``experiment_config.json`` and ``model.zip``.
- **v1** (old ppo_experiment.py): ``run_NNN/`` with a flat
  ``experiment_config.json`` and ``ppo_fbs_agent[_nfbsN].zip``.
- **bare** (notebook-era): ``run_*/`` with only ``ppo_fbs_agent*.zip`` and a
  ``*_reward_weights.json``. Env settings are reconstructed from the model's
  saved observation space (num FBS from the layout, world size from the
  position bounds) — the same values the env baked into its Box bounds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    EnvConfig,
    ExperimentConfig,
    RewardConfig,
    RewardWeights,
    WorldConfig,
    experiment_from_json_dict,
)
from .paths import RUNS_DIR


@dataclass
class RunHandle:
    run_dir: Path
    schema: str                       # "v2" | "v1" | "bare"
    model_path: Optional[Path]
    exp: Optional[ExperimentConfig]   # None => must be inferred from the model

    @property
    def name(self) -> str:
        return self.run_dir.name


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def resolve_run(run: str | Path = "latest", root: Path = RUNS_DIR) -> Path:
    if isinstance(run, Path):
        return run
    if run == "latest":
        candidates = sorted(root.glob("run_*"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No runs under {root}")
        return candidates[-1]
    path = root / run
    if not path.exists():
        path = Path(run)
    if not path.exists():
        raise FileNotFoundError(f"Run {run!r} not found under {root}")
    return path


def model_class_for(exp: Optional[ExperimentConfig]):
    """SB3 algorithm class for a run's config. Bare/v1 runs (and any config
    without an ``algo`` field) predate SAC support and are always PPO."""
    from stable_baselines3 import PPO, SAC

    if exp is not None and getattr(exp.ppo, "algo", "ppo") == "sac":
        return SAC
    return PPO


def find_model_path(run_dir: Path) -> Optional[Path]:
    for name in ("model.zip", "best_model.zip"):
        if (run_dir / name).exists():
            return run_dir / name
    legacy = sorted(run_dir.glob("ppo_fbs_agent*.zip"))
    return legacy[0] if legacy else None


def load_run(run: str | Path = "latest", root: Path = RUNS_DIR) -> RunHandle:
    run_dir = resolve_run(run, root)
    model_path = find_model_path(run_dir)
    cfg_path = run_dir / "experiment_config.json"
    if cfg_path.exists():
        data = json.loads(cfg_path.read_text())
        schema = "v2" if data.get("config_version") == 2 else "v1"
        return RunHandle(run_dir, schema, model_path, experiment_from_json_dict(data))
    return RunHandle(run_dir, "bare", model_path, None)


# --------------------------------------------------------------------------- #
# Bare-run inference
# --------------------------------------------------------------------------- #
def infer_experiment_from_model(model, run_dir: Optional[Path] = None) -> ExperimentConfig:
    """Reconstruct a legacy-mode ExperimentConfig from a loaded SB3 model.

    Legacy observations are [5 genes x num_fbs, fbs_delta, total_delta] with
    upper bounds [W, H, 150, 10.5, 1] per FBS — so the world size and FBS
    count are recoverable exactly.
    """
    high = np.asarray(model.observation_space.high, dtype=float).reshape(-1)
    dim = high.size
    if (dim - 2) % 5 != 0:
        raise ValueError(
            f"Cannot infer legacy layout from obs dim {dim}; expected 5*n+2."
        )
    num_fbs = (dim - 2) // 5
    width, height = float(high[0]), float(high[1])

    weights = RewardWeights()
    if run_dir is not None:
        for wpath in sorted(run_dir.glob("*reward_weights.json")):
            try:
                data = json.loads(wpath.read_text()).get("reward_weights", {})
                weights = RewardWeights(**{
                    k: v for k, v in data.items()
                    if k in {"beta", "gamma", "fbs_weight", "fbs_exponent", "epsilon"}
                })
                break
            except Exception:
                continue

    world = WorldConfig(width=width, height=height)
    env = EnvConfig(
        num_fbs=num_fbs,
        world=world,
        reward=RewardConfig(mode="legacy_blend", weights=weights),
    )
    return ExperimentConfig(num_fbs=num_fbs, scenario_id=1, config_id=1, env=env)


# --------------------------------------------------------------------------- #
# Listing
# --------------------------------------------------------------------------- #
def compare_runs(
    runs: Optional[list[str]] = None,
    root: Path = RUNS_DIR,
    only_differing: bool = True,
) -> pd.DataFrame:
    """Full settings table built from each run's ``experiment_config.json``.

    The config files are authoritative (the ledger historically dropped
    columns), so this reads them directly. With ``only_differing`` the output
    keeps just the settings that actually vary across the selected runs —
    the ablation view: what changed, and nothing else.
    """
    from .run_logging import RunLogger

    dirs = (
        [resolve_run(r, root) for r in runs]
        if runs
        else sorted(p for p in root.glob("run_*") if p.is_dir())
    )
    rows = []
    for d in dirs:
        cfg_path = d / "experiment_config.json"
        if not cfg_path.exists():
            continue
        try:
            data = json.loads(cfg_path.read_text())
            exp = experiment_from_json_dict(data)
        except Exception:
            continue
        prov = data.get("provenance", {}) or {}
        logger = RunLogger.__new__(RunLogger)
        logger.exp, logger.run_dir = exp, d
        logger.created_at = prov.get("created_at", "")
        logger.backend_kind = prov.get("backend", "")
        row = logger.ledger_row(prov.get("status", ""))
        row["git_sha"] = prov.get("git_sha", "") or ""
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty or not only_differing:
        return df
    keep = ["run_dir", "tag", "status"]
    varying = [
        c for c in df.columns
        if c not in keep and df[c].astype(str).nunique(dropna=False) > 1
    ]
    return df[keep + varying]


def list_runs(root: Path = RUNS_DIR) -> pd.DataFrame:
    rows = []
    if not root.exists():
        return pd.DataFrame()
    for d in sorted(root.glob("run_*")):
        if not d.is_dir():
            continue
        row = {
            "run": d.name,
            "schema": "bare",
            "code": None,
            "band_mode": None,
            "reward_mode": None,
            "num_fbs": None,
            "timesteps": None,
            "status": None,
            "has_model": find_model_path(d) is not None,
        }
        cfg = d / "experiment_config.json"
        if cfg.exists():
            try:
                data = json.loads(cfg.read_text())
            except Exception:
                data = {}
            if data.get("config_version") == 2:
                row.update(
                    schema="v2",
                    code=data.get("code"),
                    band_mode=(data.get("env", {}).get("band", {}) or {}).get("mode"),
                    reward_mode=(data.get("env", {}).get("reward", {}) or {}).get("mode"),
                    num_fbs=data.get("num_fbs"),
                    timesteps=(data.get("ppo", {}) or {}).get("total_timesteps"),
                    status=(data.get("provenance", {}) or {}).get("status"),
                )
            else:
                row.update(
                    schema="v1",
                    code=data.get("code"),
                    band_mode="legacy",
                    num_fbs=data.get("num_fbs"),
                    timesteps=data.get("total_timesteps"),
                )
        rows.append(row)
    return pd.DataFrame(rows)


def list_evals(run: str | Path = "latest") -> pd.DataFrame:
    run_dir = resolve_run(run)
    rows = []
    for d in sorted((run_dir / "evals").glob("*")):
        if not d.is_dir():
            continue
        summary = d / "summary.json"
        row = {"eval": d.name, "path": str(d)}
        if summary.exists():
            try:
                row.update(json.loads(summary.read_text()).get("aggregate", {}))
            except Exception:
                pass
        rows.append(row)
    return pd.DataFrame(rows)
