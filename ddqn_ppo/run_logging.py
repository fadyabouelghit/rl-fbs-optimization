"""``RunLogger`` for ddqn_ppo: ppo's run directory layout, its own root and ledger.

Every flat file is written exactly as for a PPO run (so ppo's callbacks,
rollouts and plots read the directory unchanged). Two things differ: the root
is ``DDQN_PPO_RUNS_DIR`` instead of ``ppo_runs/``, and the ledger row (algo
``ddqn_ppo`` plus the switch's settings) goes to that root's ``ledger.csv``,
never to ``training_log.csv``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ppo.run_logging import RunLogger, append_ledger_row

from . import paths
from .config import ALGO


class DDQNPPORunLogger(RunLogger):
    def __init__(self, exp, run_dir: Optional[Path] = None, backend_kind: str = "pyqd",
                 switch: Optional[dict] = None, flush_every: int = 256):
        root = paths.DDQN_PPO_RUNS_DIR
        super().__init__(exp, runs_root=root, run_dir=run_dir, backend_kind=backend_kind,
                         flush_every=flush_every)
        self.ledger_file = root / "ledger.csv"
        self.switch = dict(switch or {})

    def ledger_row(self, status: str, metrics: Optional[dict] = None) -> dict:
        row = super().ledger_row(status, metrics)
        finals = {k: row.pop(k) for k in [k for k in row if k.startswith("final_")]}
        for k in ("buffer_size", "learning_starts", "tau", "train_freq", "gradient_steps",
                  "sac_ent_coef"):
            row.pop(k, None)                      # SAC-only fields of PPOParams
        row["algo"] = ALGO
        row.update({f"switch_{k}": v for k, v in self.switch.items()})
        row.update(finals)
        return row

    def append_ledger(self, status: str, metrics: Optional[dict] = None) -> None:
        append_ledger_row(self.ledger_row(status, metrics), self.ledger_file)
