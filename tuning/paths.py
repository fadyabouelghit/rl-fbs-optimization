"""Where tuning studies live -- deliberately separate from the three algorithms.

    TUNING_DIR   study directories + their optuna storage  (default: ./tuning_runs)

Read like the ``PPO_*`` keys in ``ppo/paths.py``: environment first, then
``secrets.env``, then the default.

A study owns everything it produces. Its trial run directories, its
per-algorithm ledgers and its optuna database all sit under
``tuning_runs/<study>/``, so a 180-run campaign never appends a row to
``training_log.csv`` nor creates a directory in ``ppo_runs/``, ``td3_runs/``
or ``ddpg_runs/``. That isolation is done by pointing each trial subprocess's
``PPO_RUNS_DIR`` / ``TD3_RUNS_DIR`` / ``DDPG_RUNS_DIR`` at the study (see
``tuning.worker``), which has a useful side effect: a trial is an *ordinary*
run of its algorithm, so ``python -m ddpg plot --run ...``,
``lab/ddpg_summarize.py`` and ``python -m ddpg assess`` all work on a study's
output with no special casing.
"""
from __future__ import annotations

from pathlib import Path

from ppo.paths import REPO_ROOT, ensure_dir, setting

TUNING_DIR = Path(setting("TUNING_DIR", str(REPO_ROOT / "tuning_runs"))).expanduser()


def study_dir(name: str) -> Path:
    return TUNING_DIR / name


def storage_url(name: str, algo: str) -> str:
    """SQLite URL for one algorithm's optuna database inside a campaign.

    One file *per algorithm*, not one per campaign. Within a single
    algorithm the orchestrator is the only writer (trials run in
    subprocesses, but only the parent touches the study), so there is no
    concurrency to engineer around. Across algorithms there is: searching
    PPO, TD3 and DDPG at the same time is the sensible way to use 12 cores,
    and optuna's RDB storage on SQLite is explicitly not recommended for
    concurrent access from several processes. Separate files remove the
    question entirely, and cost nothing -- `report` reads the per-algorithm
    trials CSV, not the database.
    """
    return f"sqlite:///{ensure_dir(study_dir(name)) / f'study_{algo}.db'}"


def spec_path(name: str, algo: str, trial_number: int) -> Path:
    return ensure_dir(study_dir(name) / "specs") / f"{algo}_{trial_number:04d}.json"


def result_path(name: str, algo: str, trial_number: int) -> Path:
    return ensure_dir(study_dir(name) / "specs") / f"{algo}_{trial_number:04d}.result.json"


def trial_log_path(name: str, algo: str, trial_number: int) -> Path:
    return ensure_dir(study_dir(name) / "logs") / f"{algo}_{trial_number:04d}.log"
