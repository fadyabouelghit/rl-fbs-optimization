"""Evaluate a saved DDPG run.

Rollouts, artifacts and plots are ppo's (``ppo.evaluate.rollout_episode``,
``ppo.plotting.eval_gallery``), so a DDPG evaluation directory has exactly the
layout of a PPO one (see ``ppo/evaluate.py``) and is written under
``<ddpg run>/evals/<timestamp>/``. Only model loading is DDPG-specific.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from stable_baselines3 import DDPG

from ppo.env import FlyingBaseStationEnv
from ppo.evaluate import EvalReport, EpisodeRollout, _aggregate, resolve_model_path, rollout_episode
from ppo.matlab_bridge import DEFAULT_BACKEND
from ppo.paths import ensure_dir

from .runs import load_run


def evaluate_run(
    run: str | Path = "latest",
    *,
    episodes: int = 3,
    deterministic: bool = True,
    initial_state: Optional[np.ndarray] = None,
    initial_states: Optional[Sequence[np.ndarray]] = None,
    seed: int = 1000,
    backend: str = DEFAULT_BACKEND,
    max_episode_steps: Optional[int] = None,
    action_scale: Optional[float] = None,
    save: bool = True,
    make_plots: bool = True,
    model: Optional[str] = None,
) -> EvalReport:
    """Roll a saved DDPG policy for ``episodes`` episodes.

    Same semantics as ``ppo.evaluate.evaluate_run``: episode k resets with
    ``seed + k`` unless start states are pinned; ``model`` is 'final'
    (default), 'best', or a checkpoint name such as ``model_00025000``.
    """
    run_dir, exp = load_run(run)
    model_path = resolve_model_path(run_dir, model or "final")
    policy = DDPG.load(str(model_path), device="cpu")

    updates: dict = {"collect_user_positions": True}
    if max_episode_steps is not None:
        updates["max_episode_steps"] = max_episode_steps
    if action_scale is not None:
        updates["action_scale"] = action_scale
    env_config = replace(exp.env, **updates)
    env = FlyingBaseStationEnv(env_config, backend_kind=backend)

    if initial_states is None and initial_state is not None:
        initial_states = [initial_state] * episodes

    rollouts: list[EpisodeRollout] = []
    try:
        for k in range(episodes):
            init = None if initial_states is None else np.asarray(initial_states[k])
            rollouts.append(
                rollout_episode(
                    policy, env, index=k, deterministic=deterministic,
                    seed=seed + k, initial_state=init,
                )
            )
        mbs_x = np.asarray(env.backend.mbs_x, dtype=float)
        mbs_y = np.asarray(env.backend.mbs_y, dtype=float)
    finally:
        env.close()

    summary_df = pd.DataFrame([r.summary for r in rollouts])
    aggregate = _aggregate(summary_df)

    eval_dir = None
    if save:
        eval_dir = ensure_dir(run_dir / "evals" / datetime.now().strftime("%Y%m%d_%H%M%S"))
        for r in rollouts:
            stem = eval_dir / f"ep{r.index:02d}"
            r.state_df.to_csv(f"{stem}_trajectory.csv")
            r.metrics_df.to_csv(f"{stem}_metrics.csv", index=False)
            if r.user_positions is not None:
                pd.DataFrame(r.user_positions, columns=["x", "y"]).to_csv(
                    f"{stem}_users.csv", index=False
                )
        pd.DataFrame({"x": mbs_x, "y": mbs_y}).to_csv(eval_dir / "mbs.csv", index=False)
        summary_df.to_csv(eval_dir / "summary.csv", index=False)
        (eval_dir / "summary.json").write_text(
            json.dumps({"aggregate": aggregate}, indent=2, default=str)
        )
        (eval_dir / "eval_config.json").write_text(
            json.dumps(
                {
                    "run": run_dir.name,
                    "algo": "ddpg",
                    "model": model_path.name,
                    "episodes": episodes,
                    "deterministic": deterministic,
                    "seed": seed,
                    "backend": backend,
                    "max_episode_steps": env_config.max_episode_steps,
                    "action_scale": env_config.action_scale,
                    "initial_state": (
                        np.asarray(initial_states[0]).tolist()
                        if initial_states is not None
                        else None
                    ),
                    "experiment": exp.to_json_dict(),
                },
                indent=2,
                default=str,
            )
        )
        if make_plots:
            from ppo import plotting  # lazy: avoid matplotlib import for pure rollouts

            plotting.eval_gallery(
                eval_dir, rollouts, summary_df, mbs_x=mbs_x, mbs_y=mbs_y,
                world=env_config.world,
                title=f"{run_dir.name} ({exp.code}, {env_config.band.mode})",
            )

    return EvalReport(
        run_dir=run_dir,
        eval_dir=eval_dir,
        exp=exp,
        episodes=rollouts,
        summary_df=summary_df,
        aggregate=aggregate,
        mbs_x=mbs_x,
        mbs_y=mbs_y,
    )
