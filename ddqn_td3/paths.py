"""Where ddqn_td3 runs land when nothing redirects them.

    DDQN_TD3_RUNS_DIR   run directories + ledger.csv   (default: ./ddqn_td3_runs)

A switch+pilot run trains through ``td3.train`` / ``ddpg.train``, whose
loggers default to the protected ``td3_runs/`` / ``ddpg_runs/``. Launchers
(``tuning``, ``letterexp``) redirect those through ``TD3_RUNS_DIR`` /
``DDPG_RUNS_DIR`` per subprocess; for anything else (CLI, tests, ad-hoc
scripts) ``routed`` points the pilot package at this directory for the
duration of the run, so a switch+pilot run can never write to the protected
trees or their ledgers.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from ppo.paths import REPO_ROOT, setting

DDQN_TD3_RUNS_DIR = Path(setting("DDQN_TD3_RUNS_DIR", str(REPO_ROOT / "ddqn_td3_runs"))).expanduser()

_PROTECTED = {"td3": REPO_ROOT / "td3_runs", "ddpg": REPO_ROOT / "ddpg_runs"}


def _same(a: Path, b: Path) -> bool:
    return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()


@contextmanager
def routed(pilot: str):
    """Point ``<pilot>.paths`` away from its protected default, if it is there."""
    if pilot == "td3":
        import td3.paths as p
        dir_attr, ledger_attr = "TD3_RUNS_DIR", "TD3_LEDGER_PATH"
    elif pilot == "ddpg":
        import ddpg.paths as p
        dir_attr, ledger_attr = "DDPG_RUNS_DIR", "DDPG_LEDGER_PATH"
    else:
        raise ValueError(f"unknown pilot {pilot!r}")
    old_dir, old_ledger = getattr(p, dir_attr), getattr(p, ledger_attr)
    redirect = _same(old_dir, _PROTECTED[pilot]) or _same(old_ledger.parent, _PROTECTED[pilot])
    if redirect:
        new_dir = DDQN_TD3_RUNS_DIR
        setattr(p, dir_attr, new_dir)
        setattr(p, ledger_attr, new_dir / "ledger.csv")
    try:
        yield
    finally:
        setattr(p, dir_attr, old_dir)
        setattr(p, ledger_attr, old_ledger)
