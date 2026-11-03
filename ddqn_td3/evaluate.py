"""Evaluate a saved switch + pilot run on a plain env.

Same signature and report as ``td3.evaluate.evaluate_run`` (so
``tuning.objective.score_run`` can use it unchanged); only the policy differs:
``SwitchPilotPolicy`` returns the full env action, and its hold clock is
reset at every episode start.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ppo.env import FlyingBaseStationEnv
from ppo.evaluate import EvalReport, EpisodeRollout, _aggregate, rollout_episode
from ppo.matlab_bridge import DEFAULT_BACKEND
from ppo.paths import ensure_dir

from .policy import SwitchPilotPolicy, switch_meta


def load_run(run_dir: Path | str):
    """(run_dir, pilot experiment) for a ddqn_td3 run directory."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "experiment_config.json").read_text())
    pilot = switch_meta(cfg)["pilot"]
    if pilot == "td3":
        from td3.config import td3_experiment_from_json_dict as from_json
    else:
        from ddpg.config import ddpg_experiment_from_json_dict as from_json
    return run_dir, from_json(cfg)


def _which(model: Optional[str]) -> str:
    m = (model or "final").removesuffix(".zip")
    if m in ("best", "best_model"):
        return "best"
    if m in ("final", "model"):
        return "final"
    raise ValueError(f"ddqn_td3 runs save no checkpoints with a paired switch; got model={model!r}")


def evaluate_run(
    run: str | Path,
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
    run_dir, exp = load_run(run)
    which = _which(model)
    policy = SwitchPilotPolicy.load(run_dir, which=which)

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
            policy.reset()
            rollouts.append(rollout_episode(policy, env, index=k, deterministic=deterministic,
                                            seed=seed + k, initial_state=init))
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
        pd.DataFrame({"x": mbs_x, "y": mbs_y}).to_csv(eval_dir / "mbs.csv", index=False)
        summary_df.to_csv(eval_dir / "summary.csv", index=False)
        (eval_dir / "summary.json").write_text(json.dumps({"aggregate": aggregate}, indent=2,
                                                          default=str))
        (eval_dir / "eval_config.json").write_text(json.dumps(
            {"run": run_dir.name, "algo": "ddqn_" + switch_meta(json.loads(
                (run_dir / "experiment_config.json").read_text()))["pilot"],
             "model": which, "episodes": episodes, "deterministic": deterministic,
             "seed": seed, "backend": backend}, indent=2, default=str))
        if make_plots:
            from ppo import plotting
            plotting.eval_gallery(eval_dir, rollouts, summary_df, mbs_x=mbs_x, mbs_y=mbs_y,
                                  world=env_config.world,
                                  title=f"{run_dir.name} ({exp.code}, switch+pilot)")

    return EvalReport(run_dir=run_dir, eval_dir=eval_dir, exp=exp, episodes=rollouts,
                      summary_df=summary_df, aggregate=aggregate, mbs_x=mbs_x, mbs_y=mbs_y)
