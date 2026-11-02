"""Common-start-state comparison of DDPG runs against PPO and TD3 runs.

Same protocol and numbers as ``assess_models.py`` (which stays PPO-only):
every policy is rolled deterministically from ONE fixed set of start states
drawn from seed 987654, which no training or model selection ever uses. PPO
runs are evaluated by ``ppo.evaluate``, TD3 runs by ``td3.evaluate``, DDPG
runs by ``ddpg.evaluate`` -- so a DDPG-vs-TD3-vs-PPO table is produced by one
call rather than by joining three separate assessments.
"""
from __future__ import annotations

import sys
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ppo.paths import REPO_ROOT


def _assess_models():
    """Import assess_models.py (repo root) for its start states and objective,
    so every script draws identical states. Imported lazily: the script
    silences all warnings at import time, which should only happen when an
    assessment actually runs."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import assess_models

    return assess_models


def _row(algo: str, name: str, exp, report) -> dict:
    objective = _assess_models().objective
    w = exp.env.reward.weights
    mu = exp.env.world.effective_max_users
    # Scenario-1 reference points from the exhaustive grid sweep, exactly as
    # assess_models.py computes them: do-nothing 694/0, optimum 722/107.
    base = objective(w, 694, 0, mu, 0.0)
    opt = objective(w, 722, 107, mu, 1.0)
    finals = np.array([r.metrics_df["reward_objective"].iloc[-1] for r in report.episodes])
    bests = np.array([r.metrics_df["reward_objective"].max() for r in report.episodes])
    return {
        "algo": algo,
        "run": name,
        "final_J": finals.mean(),
        "final_sd": finals.std(ddof=1) if len(finals) > 1 else 0.0,
        "best_J": bests.mean(),
        "connected": np.mean([r.metrics_df["total_connected"].iloc[-1] for r in report.episodes]),
        "fbs": np.mean([r.metrics_df["fbs_connected"].iloc[-1] for r in report.episodes]),
        "final_gap_closed_%": 100 * (finals.mean() - base) / (opt - base),
        "headroom_%": 100 * (bests.mean() - base) / (opt - base),   # assess_models' column
        "baseline": base,
        "optimum": opt,
    }


def assess(
    ddpg_runs: Sequence[str],
    ppo_runs: Sequence[str] = (),
    td3_runs: Sequence[str] = (),
    *,
    episodes: int = 20,
    backend: str = "pyqd-fast",
    model: Optional[str] = None,
) -> pd.DataFrame:
    """One row per run. ``model`` ('best', a checkpoint name, ...) applies to
    every run; None means each run's final model."""
    from ppo.evaluate import evaluate_run as ppo_evaluate_run
    from ppo.runs import load_run as ppo_load_run

    from .evaluate import evaluate_run as ddpg_evaluate_run
    from .runs import load_run as ddpg_load_run

    start_states = _assess_models().start_states
    rows = []
    for name in ppo_runs:
        handle = ppo_load_run(name)
        states = start_states(handle.exp.env, episodes)
        print(f"[ppo] {handle.name}: {episodes} episodes ...", flush=True)
        report = ppo_evaluate_run(handle.run_dir, episodes=episodes, initial_states=states,
                                  backend=backend, save=False, make_plots=False, model=model)
        rows.append(_row(handle.exp.ppo.algo, handle.name, handle.exp, report))
    if td3_runs:
        # Imported here so ddpg/ works in a tree without the td3 package.
        from td3.evaluate import evaluate_run as td3_evaluate_run
        from td3.runs import load_run as td3_load_run

        for name in td3_runs:
            run_dir, exp = td3_load_run(name)
            states = start_states(exp.env, episodes)
            print(f"[td3] {run_dir.name}: {episodes} episodes ...", flush=True)
            report = td3_evaluate_run(run_dir, episodes=episodes, initial_states=states,
                                      backend=backend, save=False, make_plots=False, model=model)
            rows.append(_row("td3", run_dir.name, exp, report))
    for name in ddpg_runs:
        run_dir, exp = ddpg_load_run(name)
        states = start_states(exp.env, episodes)
        print(f"[ddpg] {run_dir.name}: {episodes} episodes ...", flush=True)
        report = ddpg_evaluate_run(run_dir, episodes=episodes, initial_states=states,
                                   backend=backend, save=False, make_plots=False, model=model)
        rows.append(_row("ddpg", run_dir.name, exp, report))
    return pd.DataFrame(rows)
