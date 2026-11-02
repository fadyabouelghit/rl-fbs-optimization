"""Where DDPG artifacts live -- deliberately separate from ppo's and td3's.

    DDPG_RUNS_DIR   DDPG run directories + ledger.csv   (default: ./ddpg_runs)

Read like the ``PPO_*`` keys in ``ppo/paths.py``: environment first, then
``secrets.env``, then the default. DDPG runs never land in ``ppo_runs/`` or
``td3_runs/`` and their ledger is never ``training_log.csv``, so PPO tooling
(``python -m ppo list``, ``assess_models.py``, the PPO ledger) never meets a
DDPG model.
"""
from __future__ import annotations

from pathlib import Path

from ppo.paths import REPO_ROOT, setting

DDPG_RUNS_DIR = Path(setting("DDPG_RUNS_DIR", str(REPO_ROOT / "ddpg_runs"))).expanduser()
DDPG_LEDGER_PATH = DDPG_RUNS_DIR / "ledger.csv"
