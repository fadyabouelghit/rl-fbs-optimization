"""Where ppo_onoff runs land.

    PPO_ONOFF_RUNS_DIR   run directories + ledger.csv   (default: ./ppo_onoff_runs)

Read at import; launchers set it per subprocess. A ppo_onoff run never writes
to ``ppo_runs/`` or ``training_log.csv``.
"""
from __future__ import annotations

from pathlib import Path

from ppo.paths import REPO_ROOT, setting

PPO_ONOFF_RUNS_DIR = Path(setting("PPO_ONOFF_RUNS_DIR", str(REPO_ROOT / "ppo_onoff_runs"))).expanduser()
