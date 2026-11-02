"""Where letter artifacts live.

    LETTER_RUNS_DIR   all new runs, probe sets, run results (default ./letter_runs)

Nothing here ever points at ppo_runs/, ddpg_runs/, td3_runs/, tuning_runs/ or
training_log.csv. Each run subprocess gets PPO_RUNS_DIR / DDPG_RUNS_DIR /
TD3_RUNS_DIR (and PPO_LEDGER_PATH) redirected under its experiment directory
by ``letterexp.launch``.
"""
from __future__ import annotations

import os
from pathlib import Path

from ppo.paths import REPO_ROOT

LETTER_RUNS_DIR = Path(os.environ.get("LETTER_RUNS_DIR", str(REPO_ROOT / "letter_runs"))).expanduser()
PROBE_SET_DIR = LETTER_RUNS_DIR / "probe_sets"
COMPARISON_DIR = REPO_ROOT / "lab" / "comparison"
FIG_DIR = COMPARISON_DIR / "figures_letter"

FORBIDDEN = ("ppo_runs", "ddpg_runs", "td3_runs", "tuning_runs", "training_log.csv")


def experiment_dir(name: str) -> Path:
    d = LETTER_RUNS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def assert_allowed(path: Path) -> None:
    """Refuse to write anywhere under the protected trees."""
    p = Path(path).resolve()
    for bad in FORBIDDEN:
        target = (REPO_ROOT / bad).resolve()
        if p == target or target in p.parents:
            raise RuntimeError(f"refusing to write into protected path {p}")
