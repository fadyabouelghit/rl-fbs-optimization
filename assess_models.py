"""Common-start-state assessment: compare trained policies on identical tests.

Each run's own post-hoc evaluation uses its own seed draw, so cross-run
comparisons inherit that luck, and the peak of a training curve is an
optimistically biased statistic (a max over many noisy 3-episode samples).

This evaluates every model on ONE fixed set of start states, drawn once from a
dedicated seed that is never used in training or model selection. Same states,
same episode length, deterministic policy — differences are the policies.

    python assess_models.py --runs run_A run_B run_C --episodes 20

Reference points for scenario 1 (2000x1500 m, 1 MBS) come from an exhaustive
grid sweep of the world: do-nothing = 694 users / 0 on the FBS, optimum = 722 /
107. Their objective values depend on each run's cost weights, so they are
recomputed per run rather than hardcoded.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ppo.evaluate import evaluate_run
from ppo.matlab_bridge import get_shared_session
from ppo.runs import load_run

ASSESS_SEED = 987_654          # never used for training or model selection


def start_states(env_cfg, n: int, seed: int = ASSESS_SEED) -> list[np.ndarray]:
    """Draw n start states from the env's own reset distribution."""
    rng = np.random.default_rng(seed)
    lo = np.array([0.0, 0.0, env_cfg.fbs_z_bounds[0], env_cfg.fbs_power_bounds[0]])
    hi = np.array([env_cfg.world.width, env_cfg.world.height,
                   env_cfg.fbs_z_bounds[1], env_cfg.fbs_power_bounds[1]])
    n_fbs, genes = env_cfg.num_fbs, env_cfg.genes_per_fbs
    out = []
    for _ in range(n):
        cont = lo + rng.random((n_fbs, 4)) * (hi - lo)
        binaries = rng.integers(0, 2, size=(n_fbs, genes - 4))
        s = np.concatenate([cont, binaries], axis=1).reshape(-1)
        if env_cfg.num_mbs_genes:
            s = np.concatenate([s, rng.integers(0, 2, size=env_cfg.num_mbs_genes)])
        out.append(s.astype(np.float32))
    return out


def objective(weights, users, fbs, max_users, power_frac) -> float:
    eps = weights.epsilon
    norm_users = min(users / max_users, 1.0)
    share = min(fbs / max(users, 1), 1.0)
    base = ((norm_users + eps) ** weights.beta) * (((1 - power_frac) + eps) ** weights.gamma)
    return (1 - weights.fbs_weight) * base + weights.fbs_weight * (share + eps) ** weights.fbs_exponent


class _Report:
    """Minimal stand-in for EvalReport (only .episodes is used here)."""

    def __init__(self, episodes):
        self.episodes = episodes


def rollout_specific_model(handle, exp, states, args, session):
    """Evaluate a checkpoint other than model.zip on the same start states."""
    from dataclasses import replace

    from ppo.env import FlyingBaseStationEnv
    from ppo.evaluate import rollout_episode
    from ppo.runs import model_class_for

    name = {"best": "best_model"}.get(args.model, args.model)
    path = handle.run_dir / f"{name}.zip"
    if not path.exists():
        path = handle.run_dir / "checkpoints" / f"{name}.zip"
    if not path.exists():
        raise SystemExit(f"no such model: {name} in {handle.run_dir}")

    model = model_class_for(exp).load(str(path), device="cpu")
    env_cfg = replace(exp.env, collect_user_positions=False)
    env = FlyingBaseStationEnv(env_cfg, backend_kind=args.backend, session=session)
    try:
        return _Report([
            rollout_episode(model, env, index=k, seed=1000 + k, initial_state=s)
            for k, s in enumerate(states)
        ])
    finally:
        env.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--backend", default="matlab", choices=["matlab", "analytic"])
    p.add_argument("--model", default="final",
                   help="'final' (model.zip), 'best' (best_model.zip), or a "
                        "checkpoint name like model_00025000. Late training can "
                        "degrade a policy, so the final model is not always the "
                        "one worth reporting.")
    args = p.parse_args()

    session = get_shared_session() if args.backend == "matlab" else None
    rows = []
    for name in args.runs:
        handle = load_run(name)
        exp = handle.exp
        states = start_states(exp.env, args.episodes)
        print(f"[{handle.name}] {args.episodes} episodes, model={args.model} ...",
              flush=True)
        if args.model == "final":
            rep = evaluate_run(handle.run_dir, episodes=args.episodes,
                               initial_states=states, backend=args.backend,
                               session=session, save=False, make_plots=False)
        else:
            rep = rollout_specific_model(handle, exp, states, args, session)
        w = exp.env.reward.weights
        mu = exp.env.world.effective_max_users
        base = objective(w, 694, 0, mu, 0.0)
        opt = objective(w, 722, 107, mu, 1.0)
        finals = np.array([r.metrics_df["reward_objective"].iloc[-1] for r in rep.episodes])
        bests = np.array([r.metrics_df["reward_objective"].max() for r in rep.episodes])
        conn = np.array([r.metrics_df["total_connected"].iloc[-1] for r in rep.episodes])
        fbs = np.array([r.metrics_df["fbs_connected"].iloc[-1] for r in rep.episodes])
        rows.append({
            "run": handle.name.split("_1-1-1_")[-1],
            "action_scale": exp.env.action_scale,
            "shaping": exp.env.reward.shaping,
            "n_steps": exp.ppo.n_steps,
            "final_J": finals.mean(), "final_sd": finals.std(ddof=1),
            "best_J": bests.mean(), "best_sd": bests.std(ddof=1),
            "connected": conn.mean(), "fbs": fbs.mean(),
            "headroom_%": 100 * (bests.mean() - base) / (opt - base),
            "baseline": base, "optimum": opt,
        })

    df = pd.DataFrame(rows)
    print()
    print(f"COMMON START STATES (n={args.episodes}, seed {ASSESS_SEED}, deterministic)")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
