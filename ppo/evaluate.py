"""Policy evaluation: multi-episode rollouts with full artifact capture.

``evaluate_run("latest", episodes=5)`` loads any generation of run directory
(v2 / v1 / bare — see runs.py), rebuilds a matching env, rolls the policy,
and writes everything under ``<run_dir>/evals/<timestamp>/``:

    eval_config.json          what was evaluated and how
    ep00_trajectory.csv       MultiIndex per-step state (old tooling format)
    ep00_metrics.csv          per-step env metrics + reward decomposition
    ep00_users.csv            user positions of the final step
    mbs.csv                   MBS coordinates of the world
    summary.csv / summary.json  per-episode rows + aggregate stats
    plots/                    the full plot gallery (unless disabled)

Repeated evaluations in one Python process reuse a single shared MATLAB
engine (see matlab_bridge.get_shared_session), so after the first call a
test cycle costs seconds, not an engine restart.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .config import EnvConfig, ExperimentConfig
from .env import FlyingBaseStationEnv
from .matlab_bridge import MatlabSession, get_shared_session
from .paths import TEST_LOG_DIR, ensure_dir
from .runs import RunHandle, infer_experiment_from_model, load_run, model_class_for

_HEAVY_INFO_KEYS = {"user_positions", "state", "episode", "terminal_observation", "TimeLimit.truncated"}


# --------------------------------------------------------------------------- #
# State flattening
# --------------------------------------------------------------------------- #
def flatten_states(states: Sequence[np.ndarray], env_config: EnvConfig) -> pd.DataFrame:
    """State history (physical units + global blocks) -> MultiIndex DataFrame
    (compatible with the old trajectory CSVs consumed by plot_trajectories.py).

    Rows come from ``env.flat_state_with_globals()`` so values stay in
    physical units even when the *observation* is normalized."""
    arr = np.asarray(states, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    genes = 6 if env_config.band.agent_controls_fbs_band else 5
    n = env_config.num_fbs
    names = ["x", "y", "height", "power", "power_status"]
    if genes == 6:
        names.append("band_flag")

    cols, data = [], []
    fbs_block = arr[:, : genes * n].reshape(len(arr), n, genes)
    for i in range(n):
        for gi, gname in enumerate(names):
            cols.append((f"fbs{i}", gname))
            data.append(fbs_block[:, i, gi])
    offset = genes * n
    if env_config.band.agent_controls_mbs_capacity:
        for j in range(env_config.world.num_mbs):
            cols.append((f"mbs{j}", "capacity"))
            data.append(arr[:, offset + j])
        offset += env_config.world.num_mbs
    if arr.shape[1] >= offset + 2:
        cols.append(("global", "fbs_delta"))
        data.append(arr[:, offset])
        cols.append(("global", "total_delta"))
        data.append(arr[:, offset + 1])
    if env_config.obs_tier_metrics and arr.shape[1] >= offset + 6:
        for k, name in enumerate(("fbs_frac", "mbs_frac", "total_frac", "norm_power")):
            cols.append(("global", name))
            data.append(arr[:, offset + 2 + k])

    return pd.DataFrame(
        np.column_stack(data), columns=pd.MultiIndex.from_tuples(cols)
    )


# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #
def resolve_model_path(run_dir: Path, spec: str) -> Path:
    """Locate a specific saved policy inside a run.

    ``spec`` accepts ``"final"`` / ``"best"``, a checkpoint name such as
    ``model_00025000``, or a path relative to the run directory. Late training
    can degrade a policy, so evaluating a mid-training checkpoint — or
    comparing two runs at a matched step count — is a routine need.
    """
    name = {"final": "model", "best": "best_model"}.get(spec, spec)
    for cand in (
        run_dir / name,
        run_dir / f"{name}.zip",
        run_dir / "checkpoints" / name,
        run_dir / "checkpoints" / f"{name}.zip",
    ):
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"model {spec!r} not found in {run_dir} "
        f"(looked for {name}[.zip] at top level and in checkpoints/)"
    )


# --------------------------------------------------------------------------- #
# Rollouts
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeRollout:
    index: int
    state_df: pd.DataFrame
    metrics_df: pd.DataFrame
    user_positions: Optional[np.ndarray]
    summary: dict


def rollout_episode(
    model,
    env: FlyingBaseStationEnv,
    *,
    index: int = 0,
    deterministic: bool = True,
    seed: Optional[int] = None,
    initial_state: Optional[np.ndarray] = None,
) -> EpisodeRollout:
    obs, _ = env.reset(seed=seed, options={"state": initial_state})
    states = [env.flat_state_with_globals()]
    rewards, infos = [], []
    user_positions = None
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        states.append(env.flat_state_with_globals())
        rewards.append(float(reward))
        if info.get("user_positions") is not None:
            user_positions = info["user_positions"]
        infos.append(
            {k: v for k, v in info.items() if k not in _HEAVY_INFO_KEYS}
        )

    state_df = flatten_states(states, env.config)
    metrics_df = pd.DataFrame(infos)
    metrics_df.insert(0, "reward", rewards)

    last = infos[-1] if infos else {}
    summary = {
        "episode": index,
        "length": len(rewards),
        "reward_sum": float(np.sum(rewards)),
        "reward_mean": float(np.mean(rewards)) if rewards else float("nan"),
        "terminated": bool(terminated),
        **{
            f"final_{k}": last.get(k)
            for k in (
                "total_connected", "fbs_connected", "mbs_connected",
                "mbs_coverage_connected", "mbs_capacity_connected",
                "total_power", "avg_rate", "sum_rate", "num_active_fbs",
            )
        },
        "best_total_connected": (
            int(metrics_df["total_connected"].max())
            if "total_connected" in metrics_df and len(metrics_df)
            else None
        ),
    }
    return EpisodeRollout(index, state_df, metrics_df, user_positions, summary)


# --------------------------------------------------------------------------- #
# Full evaluation of a saved run
# --------------------------------------------------------------------------- #
@dataclass
class EvalReport:
    run_dir: Path
    eval_dir: Optional[Path]
    exp: ExperimentConfig
    episodes: list[EpisodeRollout]
    summary_df: pd.DataFrame
    aggregate: dict
    mbs_x: np.ndarray
    mbs_y: np.ndarray


def _aggregate(summary_df: pd.DataFrame) -> dict:
    agg = {"n_episodes": int(len(summary_df))}
    for col in summary_df.columns:
        if summary_df[col].dtype.kind in "if":
            agg[f"{col}_mean"] = float(summary_df[col].mean())
            agg[f"{col}_std"] = float(summary_df[col].std(ddof=0))
            agg[f"{col}_min"] = float(summary_df[col].min())
            agg[f"{col}_max"] = float(summary_df[col].max())
    return agg


def evaluate_run(
    run: str | Path = "latest",
    *,
    episodes: int = 3,
    deterministic: bool = True,
    initial_state: Optional[np.ndarray] = None,
    initial_states: Optional[Sequence[np.ndarray]] = None,
    seed: int = 1000,
    backend: str = "matlab",
    session: Optional[MatlabSession] = None,
    max_episode_steps: Optional[int] = None,
    action_scale: Optional[float] = None,
    env_config_override: Optional[EnvConfig] = None,
    exp_override: Optional[ExperimentConfig] = None,
    save: bool = True,
    make_plots: bool = True,
    mirror_test_logs: bool = False,
    model: Optional[str] = None,
) -> EvalReport:
    """Roll a saved policy for `episodes` episodes and persist the artifacts.

    ``initial_state`` pins every episode to one start state (the old
    notebook's custom-state workflow); ``initial_states`` gives one per
    episode. Without either, episode k resets with ``seed + k`` — the same
    start states every time you evaluate, for comparable numbers.
    """
    handle: RunHandle = load_run(run)
    model_path = (
        resolve_model_path(handle.run_dir, model) if model else handle.model_path
    )
    if model_path is None:
        raise FileNotFoundError(f"No model zip found in {handle.run_dir}")
    algo_cls = model_class_for(exp_override or handle.exp)
    policy = algo_cls.load(str(model_path), device="cpu")

    exp = exp_override or handle.exp
    if exp is None:
        exp = infer_experiment_from_model(model, handle.run_dir)

    env_config = env_config_override or exp.env
    updates: dict = {"collect_user_positions": True}
    if max_episode_steps is not None:
        updates["max_episode_steps"] = max_episode_steps
    if action_scale is not None:
        updates["action_scale"] = action_scale
    env_config = replace(env_config, **updates)

    if backend == "matlab" and session is None:
        session = get_shared_session()
    env = FlyingBaseStationEnv(env_config, backend_kind=backend, session=session)

    if initial_states is None and initial_state is not None:
        initial_states = [initial_state] * episodes

    rollouts: list[EpisodeRollout] = []
    try:
        for k in range(episodes):
            init = None if initial_states is None else np.asarray(initial_states[k])
            rollouts.append(
                rollout_episode(
                    policy,
                    env,
                    index=k,
                    deterministic=deterministic,
                    seed=seed + k,
                    initial_state=init,
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
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = ensure_dir(handle.run_dir / "evals" / ts)
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
                    "run": handle.name,
                    "model": model_path.name,
                    "schema": handle.schema,
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

        if mirror_test_logs and rollouts:
            # Old tooling layout: test_logs/<code>/<run>_<ts>_trajectory.csv
            mirror_dir = ensure_dir(TEST_LOG_DIR / exp.code)
            stem = mirror_dir / f"{handle.name}_{ts}"
            r0 = rollouts[0]
            r0.state_df.to_csv(f"{stem}_trajectory.csv")
            r0.metrics_df.to_csv(f"{stem}_metrics.csv", index=False)
            Path(f"{stem}_meta.json").write_text(
                json.dumps(
                    {
                        "code": exp.code,
                        "run_dir": handle.name,
                        "timestamp": ts,
                        "num_steps": len(r0.metrics_df),
                        "experiment": exp.to_json_dict(),
                    },
                    indent=2,
                    default=str,
                )
            )

        if make_plots:
            from . import plotting  # lazy: avoid matplotlib import for pure rollouts

            plotting.eval_gallery(
                eval_dir,
                rollouts,
                summary_df,
                mbs_x=mbs_x,
                mbs_y=mbs_y,
                world=env_config.world,
                title=f"{handle.name} ({exp.code}, {env_config.band.mode})",
            )

    return EvalReport(
        run_dir=handle.run_dir,
        eval_dir=eval_dir,
        exp=exp,
        episodes=rollouts,
        summary_df=summary_df,
        aggregate=aggregate,
        mbs_x=mbs_x,
        mbs_y=mbs_y,
    )
