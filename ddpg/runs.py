"""Locate and load DDPG run directories under ``ddpg_runs/``."""
from __future__ import annotations

import json
from pathlib import Path

from ppo.runs import resolve_run as _resolve_under

from . import paths
from .config import DDPGExperiment, ddpg_experiment_from_json_dict


def resolve_run(run: str | Path = "latest") -> Path:
    """``"latest"``, a run name under ddpg_runs/, or a path."""
    return _resolve_under(run, root=paths.DDPG_RUNS_DIR)


def load_run(run: str | Path = "latest") -> tuple[Path, DDPGExperiment]:
    run_dir = resolve_run(run)
    cfg = run_dir / "experiment_config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"{run_dir} has no experiment_config.json")
    return run_dir, ddpg_experiment_from_json_dict(json.loads(cfg.read_text()))
