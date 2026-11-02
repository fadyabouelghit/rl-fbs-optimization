"""Where TD3 artifacts live -- deliberately separate from ppo's.

    TD3_RUNS_DIR   TD3 run directories + ledger.csv   (default: ./td3_runs)

Read like the ``PPO_*`` keys in ``ppo/paths.py``: environment first, then
``secrets.env``, then the default. TD3 runs never land in ``ppo_runs/`` and
their ledger is never ``training_log.csv``, so PPO tooling (``python -m ppo
list``, ``assess_models.py``, the PPO ledger) never meets a TD3 model.
"""
from __future__ import annotations

from pathlib import Path

from ppo.paths import REPO_ROOT, setting

TD3_RUNS_DIR = Path(setting("TD3_RUNS_DIR", str(REPO_ROOT / "td3_runs"))).expanduser()
TD3_LEDGER_PATH = TD3_RUNS_DIR / "ledger.csv"
