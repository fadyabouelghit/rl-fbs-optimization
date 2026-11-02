"""One objective, scored identically for all three algorithms.

    mean final J of the run's selected model, over a fixed set of common
    start states drawn from a seed that training and model selection never
    touch.

This is ``assess_models.py``'s protocol, reused rather than reimplemented:
the start states come from ``assess_models.start_states`` and the
baseline/optimum reference points from ``assess_models.objective``, so a
tuning score and a number from ``python -m ddpg assess`` mean the same thing
and can be put in the same table.

Scoring a *trained run directory* rather than a live model is deliberate. It
keeps the scorer indifferent to which package trained the run, lets a trial
be re-scored later on a different start-state draw without retraining (the
held-out reporting set), and means a study's trials are ordinary runs that
the existing analysis tools already understand.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ppo.paths import REPO_ROOT

from .protocol import Protocol


def _assess_models():
    """Import assess_models.py from the repo root, lazily.

    Lazily because it calls ``warnings.filterwarnings('ignore')`` at import
    time, which should only happen inside a process that is actually
    scoring something.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import assess_models

    return assess_models


def _load(algo: str, run_dir: Path):
    """(experiment, evaluate_run) for the package that owns this run."""
    if algo == "ppo":
        from ppo.evaluate import evaluate_run
        from ppo.runs import load_run

        return load_run(run_dir).exp, evaluate_run
    if algo in ("td3", "td3_hybrid", "td3_hybrid_qd"):
        from td3.evaluate import evaluate_run
        from td3.runs import load_run

        return load_run(run_dir)[1], evaluate_run
    if algo in ("ddpg", "ddpg_hybrid", "ddpg_hybrid_qd"):
        # hybrid runs load through DDPG.load: model.zip names the hybrid
        # policy class, whose deterministic predict is the argmax head
        from ddpg.evaluate import evaluate_run
        from ddpg.runs import load_run

        return load_run(run_dir)[1], evaluate_run
    raise ValueError(f"unknown algo {algo!r}")


def score_run(
    algo: str,
    run_dir: Path | str,
    protocol: Protocol,
    *,
    states_seed: Optional[int] = None,
    episodes: Optional[int] = None,
    model: Optional[str] = None,
) -> Dict[str, float]:
    """Score one trained run. Returns the objective and its context.

    ``states_seed`` defaults to the protocol's tuning seed; pass the held-out
    reporting seed to re-score a finalist on states the search never saw.
    """
    run_dir = Path(run_dir)
    am = _assess_models()
    exp, evaluate_run = _load(algo, run_dir)

    episodes = protocol.objective_episodes if episodes is None else episodes
    states_seed = protocol.objective_seed if states_seed is None else states_seed
    want = protocol.objective_model if model is None else model

    # An interrupted or very short run may have no best_model.zip: the
    # in-training evaluator writes it only once it has run. Fall back to the
    # final model and say so, rather than failing the trial.
    selected = want
    if want == "best" and not (run_dir / "best_model.zip").is_file():
        selected = "final"

    states = am.start_states(exp.env, episodes, seed=states_seed)
    report = evaluate_run(
        run_dir, episodes=episodes, initial_states=states,
        backend=protocol.backend, save=False, make_plots=False, model=selected,
    )

    finals = np.array([r.metrics_df["reward_objective"].iloc[-1] for r in report.episodes])
    bests = np.array([r.metrics_df["reward_objective"].max() for r in report.episodes])

    # Reference points from the exhaustive grid sweep, recomputed per run
    # because they depend on the run's own cost weights.
    #
    # The do-nothing baseline (694 users, 0 on the FBS) is valid for ANY number
    # of FBSs: doing nothing is doing nothing, and the potrec10 reward has
    # gamma = 0, so transmit power never enters the objective. The OPTIMUM is
    # not: 722 / 107 came from a 4-D exhaustive sweep of a ONE-FBS world, and
    # two FBSs can serve more users than that. Reporting gap_closed off
    # scenario 1-1-1 would silently inflate it past 100%, so it is withheld as
    # NaN rather than guessed -- the tuner ranks on raw J and never reads it.
    w = exp.env.reward.weights
    mu = exp.env.world.effective_max_users
    base = am.objective(w, 694, 0, mu, 0.0)
    has_sweep = exp.env.num_fbs == 1 and exp.env.world.num_mbs == 1
    opt = am.objective(w, 722, 107, mu, 1.0) if has_sweep else float("nan")

    return {
        "objective": float(finals.mean()),
        "objective_sd": float(finals.std(ddof=1)) if len(finals) > 1 else 0.0,
        "best_J": float(bests.mean()),
        "gap_closed_pct": (float(100 * (finals.mean() - base) / (opt - base))
                           if has_sweep else float("nan")),
        "connected": float(np.mean(
            [r.metrics_df["total_connected"].iloc[-1] for r in report.episodes])),
        "fbs_connected": float(np.mean(
            [r.metrics_df["fbs_connected"].iloc[-1] for r in report.episodes])),
        "episodes": int(episodes),
        "states_seed": int(states_seed),
        "model": selected,
        "model_requested": want,
        "baseline": float(base),
        "optimum": float(opt),
    }
