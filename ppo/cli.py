"""Command-line interface: ``python -m ppo <command>``.

    train   train a policy            (real MATLAB physics by default)
    eval    evaluate a saved run      (multi-episode, artifacts + plots)
    plot    render training gallery / multi-run comparisons
    list    list runs (or evals of one run)
    info    show a run's resolved config
    smoke   fast end-to-end pipeline check on the analytic backend

Examples:
    python -m ppo train --code 1-1-1 --timesteps 5000 --seed 0
    python -m ppo train --code 2-2-1 --band multi --fbs-band agent \\
                        --mbs-capacity agent --reward ga_blend --n-envs 4
    python -m ppo eval --run latest --episodes 5
    python -m ppo eval --run run_039 --state 800,800,100,10.5,1
    python -m ppo plot --run latest
    python -m ppo plot --compare run_A run_B --metric reward_sum
    python -m ppo smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _parse_state(s):
    if not s:
        return None
    return np.array([float(v) for v in s.split(",")], dtype=np.float32)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ppo", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---------------- train ----------------
    t = sub.add_parser("train", help="Train an RL agent (PPO or SAC)")
    t.add_argument("--code", default="1-1-1", help="X-Y-Z experiment code")
    t.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    t.add_argument("--band", default="legacy", choices=["legacy", "multi"])
    t.add_argument("--fbs-band", default="coverage", choices=["coverage", "capacity", "agent"])
    t.add_argument("--mbs-capacity", default="off", choices=["off", "on", "agent"])
    t.add_argument("--reward", default=None, choices=["legacy_blend", "ga_blend", "sum_rate"])
    t.add_argument("--reward-shaping", default=None,
                   choices=["none", "potential", "record", "potential_record"],
                   help="'potential': delta-objective; 'record': paid only for new episode "
                        "bests; 'potential_record': both (delta + record_weight * new-best bonus)")
    t.add_argument("--record-weight", type=float, default=None,
                   help="weight on the new-best bonus under --reward-shaping potential_record "
                        "(default 0.5)")
    t.add_argument("--beta", type=float, default=None,
                   help="connectivity exponent in the cost function "
                        "(historical runs and the GA use 0.8; dataclass default is 1.0)")
    t.add_argument("--fbs-weight", type=float, default=None,
                   help="weight on the FBS-share term (historical: 0.4)")
    t.add_argument("--reward-offset", type=float, default=None,
                   help="training signal = (objective - offset) * gain; raw objective still logged")
    t.add_argument("--reward-gain", type=float, default=None,
                   help="fixed scale factor on the training signal (does not move the "
                        "optimum). Prefer --normalize-reward, which achieves the same "
                        "conditioning without reference to a known optimum.")
    t.add_argument("--normalize-reward", action="store_true",
                   help="running reward normalization (VecNormalize): divides rewards by "
                        "a running std estimate from the agent's own experience")
    t.add_argument("--action-mode", default=None, choices=["delta", "absolute"],
                   help="'absolute': actions name gene values directly instead of nudging them")
    t.add_argument("--normalize-obs", action="store_true",
                   help="min-max scale observations to [-1, 1]")
    t.add_argument("--sticky-binaries", action="store_true",
                   help="binary genes flip only when |action| >= 0.5")
    t.add_argument("--obs-tier-metrics", action="store_true",
                   help="append per-tier connectivity fractions to the observation")
    t.add_argument("--no-full-coverage-termination", action="store_true",
                   help="disable the legacy early-stop at full coverage")
    t.add_argument("--discount", type=float, default=None,
                   help="RL discount factor (PPO/SAC gamma; distinct from the reward-weight gamma)")
    t.add_argument("--timesteps", type=int, default=None)
    t.add_argument("--max-episode-steps", type=int, default=None)
    t.add_argument("--ent-coef", type=float, default=None)
    t.add_argument("--lr", type=float, default=None)
    t.add_argument("--action-scale", type=float, default=None)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--n-envs", type=int, default=1)
    t.add_argument("--n-steps", type=int, default=None, help="PPO rollout buffer length")
    t.add_argument("--checkpoint-every", type=int, default=0)
    t.add_argument("--eval-every", type=int, default=0)
    t.add_argument("--tag", default=None)
    t.add_argument("--backend", default="matlab", choices=["matlab", "analytic"])
    t.add_argument("--resume-from", default=None, help="run name/path to warm-start from")

    # ---------------- eval ----------------
    e = sub.add_parser("eval", help="Evaluate a saved run")
    e.add_argument("--run", default="latest")
    e.add_argument("--episodes", type=int, default=3)
    e.add_argument("--model", default=None,
                   help="which saved policy to evaluate: 'final' (default), 'best', "
                        "or a checkpoint name like model_00025000")
    e.add_argument("--stochastic", action="store_true")
    e.add_argument("--state", default=None,
                   help="comma-separated initial state (pins every episode)")
    e.add_argument("--seed", type=int, default=1000)
    e.add_argument("--max-episode-steps", type=int, default=None)
    e.add_argument("--action-scale", type=float, default=None)
    e.add_argument("--backend", default="matlab", choices=["matlab", "analytic"])
    e.add_argument("--no-plots", action="store_true")
    e.add_argument("--no-save", action="store_true")
    e.add_argument("--mirror-test-logs", action="store_true",
                   help="also write old-style test_logs/<code>/ files")

    # ---------------- plot ----------------
    pl = sub.add_parser("plot", help="Render plots from logged CSVs")
    pl.add_argument("--run", default="latest")
    pl.add_argument("--compare", nargs="*", default=None,
                    help="run names to overlay instead of a single-run gallery")
    pl.add_argument("--metric", default="reward_sum")
    pl.add_argument("--window", type=int, default=20)

    # ---------------- list / info ----------------
    ls = sub.add_parser("list", help="List runs")
    ls.add_argument("--evals", default=None, metavar="RUN",
                    help="list evaluations of one run instead")
    ls.add_argument("--compare", nargs="*", default=None, metavar="RUN",
                    help="full settings table (all runs if no names given); "
                         "shows only settings that differ across them")
    ls.add_argument("--all-columns", action="store_true",
                    help="with --compare: show every setting, not just differing ones")

    info = sub.add_parser("info", help="Show a run's resolved config")
    info.add_argument("--run", default="latest")

    # ---------------- smoke ----------------
    s = sub.add_parser("smoke", help="Fast end-to-end check (no MATLAB needed)")
    s.add_argument("--timesteps", type=int, default=1024)
    s.add_argument("--band", default="multi", choices=["legacy", "multi"])
    s.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    s.add_argument("--n-envs", type=int, default=1)

    return p


def cmd_train(args) -> int:
    from .config import BandConfig, ExperimentConfig, RewardConfig, RewardWeights, CONFIGS
    from .train import train

    band = BandConfig(mode=args.band, fbs_band=args.fbs_band, mbs_capacity=args.mbs_capacity)
    reward = None
    if (args.reward or args.reward_shaping or args.beta is not None
            or args.fbs_weight is not None or args.record_weight is not None
            or args.reward_offset is not None or args.reward_gain is not None):
        z = int(args.code.split("-")[2])
        weight_kwargs = {"gamma": CONFIGS[z]["gamma"]}
        if args.beta is not None:
            weight_kwargs["beta"] = args.beta
        if args.fbs_weight is not None:
            weight_kwargs["fbs_weight"] = args.fbs_weight
        reward_kwargs = {}
        if args.record_weight is not None:
            reward_kwargs["record_weight"] = args.record_weight
        reward = RewardConfig(mode=args.reward or "legacy_blend",
                              shaping=args.reward_shaping or "none",
                              **reward_kwargs,
                              reward_offset=args.reward_offset or 0.0,
                              reward_gain=1.0 if args.reward_gain is None else args.reward_gain,
                              weights=RewardWeights(**weight_kwargs))
    ppo_overrides = {"algo": args.algo}
    if args.normalize_reward:
        ppo_overrides["normalize_reward"] = True
    if args.checkpoint_every:
        ppo_overrides["checkpoint_every"] = args.checkpoint_every
    if args.eval_every:
        ppo_overrides["eval_every"] = args.eval_every
    if args.n_steps:
        ppo_overrides["n_steps"] = args.n_steps
    if args.discount is not None:
        ppo_overrides["gamma"] = args.discount

    env_overrides = {}
    if args.action_mode:
        env_overrides["action_mode"] = args.action_mode
    if args.normalize_obs:
        env_overrides["normalize_obs"] = True
    if args.sticky_binaries:
        env_overrides["binary_mode"] = "sticky"
    if args.obs_tier_metrics:
        env_overrides["obs_tier_metrics"] = True
    if args.no_full_coverage_termination:
        env_overrides["terminate_on_full_coverage"] = False

    exp = ExperimentConfig.from_code(
        args.code,
        band=band,
        reward=reward,
        tag=args.tag,
        total_timesteps=args.timesteps,
        max_episode_steps=args.max_episode_steps,
        ent_coef=args.ent_coef,
        learning_rate=args.lr,
        action_scale=args.action_scale,
        seed=args.seed,
        n_envs=args.n_envs,
        env_overrides=env_overrides,
        ppo_overrides=ppo_overrides,
    )
    run_dir = train(exp, backend=args.backend, resume_from=args.resume_from)
    print(f"\nrun complete -> {run_dir}")
    return 0


def cmd_eval(args) -> int:
    from .evaluate import evaluate_run

    report = evaluate_run(
        args.run,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        initial_state=_parse_state(args.state),
        seed=args.seed,
        backend=args.backend,
        max_episode_steps=args.max_episode_steps,
        action_scale=args.action_scale,
        save=not args.no_save,
        make_plots=not args.no_plots,
        mirror_test_logs=args.mirror_test_logs,
        model=args.model,
    )
    print(f"\nevaluated {report.run_dir.name} ({report.exp.code})")
    print(report.summary_df.to_string(index=False))
    print("\naggregate:")
    for k in ("reward_sum_mean", "final_total_connected_mean", "final_total_power_mean"):
        if k in report.aggregate:
            print(f"  {k:32s} {report.aggregate[k]:.3f}")
    if report.eval_dir:
        print(f"\nartifacts -> {report.eval_dir}")
    return 0


def cmd_plot(args) -> int:
    from . import plotting
    from .runs import resolve_run

    if args.compare:
        dirs = [resolve_run(r) for r in args.compare]
        out = dirs[0] / "plots" / f"compare_{args.metric}"
        plotting.plot_learning_curves(dirs, metric=args.metric,
                                      window=args.window, save=out)
        print(f"comparison plot -> {out}.png")
    else:
        run_dir = resolve_run(args.run)
        out = plotting.training_gallery(run_dir, window=args.window)
        print(f"training gallery -> {out}")
    return 0


def cmd_list(args) -> int:
    import pandas as pd

    from .runs import compare_runs, list_evals, list_runs

    if args.compare is not None:
        df = compare_runs(args.compare or None, only_differing=not args.all_columns)
        with pd.option_context("display.max_columns", None, "display.width", 250):
            print("nothing found" if df.empty else df.to_string(index=False))
        return 0
    df = list_evals(args.evals) if args.evals else list_runs()
    if df.empty:
        print("nothing found")
    else:
        print(df.to_string(index=False))
    return 0


def cmd_info(args) -> int:
    from .runs import load_run

    handle = load_run(args.run)
    print(f"run    : {handle.name}")
    print(f"schema : {handle.schema}")
    print(f"model  : {handle.model_path}")
    if handle.exp:
        print(json.dumps(handle.exp.to_json_dict(), indent=2, default=str))
    else:
        print("(bare run — config inferred from the model at eval time)")
    return 0


def cmd_smoke(args) -> int:
    """End-to-end pipeline check on the analytic backend (~30 s, no MATLAB)."""
    from .config import BandConfig, ExperimentConfig
    from .evaluate import evaluate_run
    from .train import train

    band = (
        BandConfig(mode="multi", fbs_band="agent", mbs_capacity="agent")
        if args.band == "multi"
        else BandConfig()
    )
    ppo_overrides = {
        "algo": args.algo, "n_steps": 64, "batch_size": 64,
        "checkpoint_every": 512, "eval_every": 512, "eval_episodes": 2,
        "verbose": 0,
    }
    if args.algo == "sac":
        ppo_overrides.update(buffer_size=5000, learning_starts=64)
    exp = ExperimentConfig.from_code(
        "1-1-1",
        band=band,
        tag="smoke",
        total_timesteps=args.timesteps,
        seed=0,
        n_envs=args.n_envs,
        ppo_overrides=ppo_overrides,
        world_overrides={"num_users": 300},
    )
    print(f"[smoke] training {args.timesteps} steps ({args.algo}) on the analytic backend ...")
    run_dir = train(exp, backend="analytic")

    print("[smoke] evaluating ...")
    report = evaluate_run(run_dir, episodes=2, backend="analytic")

    from .plotting import training_gallery
    training_gallery(run_dir)

    required = [
        run_dir / "model.zip",
        run_dir / "experiment_config.json",
        run_dir / "steps.csv",
        run_dir / "episodes.csv",
        run_dir / "progress.csv",
        run_dir / "plots" / "training_overview.png",
        report.eval_dir / "summary.json",
        report.eval_dir / "plots" / "ep00_trajectory.png",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("[smoke] FAIL — missing artifacts:")
        for m in missing:
            print("   ", m)
        return 1
    print(f"[smoke] PASS — run dir: {run_dir}")
    print(f"[smoke] eval reward_sum_mean = {report.aggregate.get('reward_sum_mean'):.3f}, "
          f"connected_mean = {report.aggregate.get('final_total_connected_mean'):.1f}")
    return 0


def main(argv=None) -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")  # headless-safe plotting
    args = _build_parser().parse_args(argv)
    return {
        "train": cmd_train,
        "eval": cmd_eval,
        "plot": cmd_plot,
        "list": cmd_list,
        "info": cmd_info,
        "smoke": cmd_smoke,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
