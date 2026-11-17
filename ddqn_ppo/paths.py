"""Where ddqn_ppo runs land.

    DDQN_PPO_RUNS_DIR   run directories + ledger.csv   (default: ./ddqn_ppo_runs)

Read at import, like ``ppo.paths``: launchers (``tuning``, ``letterexp``) set
it per subprocess. A ddqn_ppo run never writes to ``ppo_runs/`` or
``training_log.csv``: its logger takes this directory as its root and writes
its ledger here (see ``ddqn_ppo.run_logging``).
"""
from __future__ import annotations

from pathlib import Path

from ppo.paths import REPO_ROOT, setting

DDQN_PPO_RUNS_DIR = Path(setting("DDQN_PPO_RUNS_DIR", str(REPO_ROOT / "ddqn_ppo_runs"))).expanduser()


def ledger_path() -> Path:
    return DDQN_PPO_RUNS_DIR / "ledger.csv"
