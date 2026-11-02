"""Command-line interface: ``python -m ddpg <command>``.

    train    train a DDPG policy         (runs land in ddpg_runs/)
    eval     evaluate a saved DDPG run   (artifacts + plots, same layout as ppo)
    plot     render a DDPG run's training gallery
    assess   DDPG runs vs TD3 and PPO runs on common start states

The environment and reward flags of ``train`` have the same names and
defaults as ``python -m ppo train`` and ``python -m td3 train``, so a launch
command carries over by swapping the module and the algorithm flags. PPO-only
flags (--n-steps, --ent-coef) do not exist here, and neither do the TD3-only
tricks DDPG drops (--policy-delay, --target-policy-noise, --target-noise-clip).

Examples:
    python -m ddpg train --code 1-1-1 --timesteps 25000 --seed 0 --backend pyqd-fast
    python -m ddpg train --code 1-1-1 --noise-type ou --binary-noise-std 1.0
    python -m ddpg eval --run latest --episodes 5 --backend pyqd-fast
    python -m ddpg plot --run latest
    python -m ddpg assess --runs latest --td3-runs <td3 run> --ppo-runs <ppo run>
"""
from __future__ import annotations

import argparse
import os
import sys

from ppo.matlab_bridge import BACKEND_CHOICES, DEFAULT_BACKEND


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ddpg", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---------------- train ----------------
    t = sub.add_parser("train", help="Train a DDPG agent")
    # Environment + reward: same names/defaults as `python -m ppo train`.
    t.add_argument("--code", default="1-1-1", help="X-Y-Z experiment code")
    t.add_argument("--band", default="legacy", choices=["legacy", "multi"])
    t.add_argument("--fbs-band", default="coverage", choices=["coverage", "capacity", "agent"])
    t.add_argument("--mbs-capacity", default="off", choices=["off", "on", "agent"])
    t.add_argument("--reward", default=None,
                   choices=["legacy_blend", "controlled_blend", "ga_blend", "sum_rate"])
    t.add_argument("--reward-shaping", default=None,
                   choices=["none", "potential", "record", "potential_record"])
    t.add_argument("--record-weight", type=float, default=None)
    t.add_argument("--beta", type=float, default=None)
    t.add_argument("--fbs-weight", type=float, default=None)
    t.add_argument("--reward-offset", type=float, default=None)
    t.add_argument("--reward-gain", type=float, default=None)
    t.add_argument("--normalize-reward", action="store_true",
                   help="running reward normalization (SB3 VecNormalize)")
    t.add_argument("--action-mode", default=None, choices=["delta", "absolute"])
    t.add_argument("--normalize-obs", action="store_true")
    t.add_argument("--sticky-binaries", action="store_true")
    t.add_argument("--obs-tier-metrics", action="store_true")
    t.add_argument("--no-full-coverage-termination", action="store_true")
    t.add_argument("--max-episode-steps", type=int, default=None)
    t.add_argument("--action-scale", type=float, default=None)

    t.add_argument("--timesteps", type=int, default=None)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--n-envs", type=int, default=1)
    t.add_argument("--checkpoint-every", type=int, default=0)
    t.add_argument("--eval-every", type=int, default=0)
    t.add_argument("--eval-episodes", type=int, default=None)
    t.add_argument("--tag", default=None)
    t.add_argument("--backend", default=DEFAULT_BACKEND, choices=list(BACKEND_CHOICES))
    t.add_argument("--resume-from", default=None, help="DDPG run name/path to warm-start from")

    g = t.add_argument_group("DDPG hyperparameters (default: SB3 DDPG defaults)")
    g.add_argument("--lr", type=float, default=None, help="learning rate (default 1e-3)")
    g.add_argument("--discount", type=float, default=None, help="RL discount factor (default 0.99)")
    g.add_argument("--batch-size", type=int, default=None, help="default 256")
    g.add_argument("--buffer-size", type=int, default=None, help="default 1000000")
    g.add_argument("--learning-starts", type=int, default=None, help="default 100")
    g.add_argument("--tau", type=float, default=None, help="target-network update rate (default 0.005)")
    g.add_argument("--train-freq", type=int, default=None, help="env steps per update round (default 1)")
    g.add_argument("--gradient-steps", type=int, default=None, help="updates per round (default 1)")
    g.add_argument("--net-arch", default=None,
                   help="comma-separated layer sizes, e.g. 256,256 (default SB3's 400,300)")

    n = t.add_argument_group("exploration noise")
    n.add_argument("--noise-type", default=None, choices=["normal", "ou"],
                   help="Gaussian (default) or Ornstein-Uhlenbeck, the DDPG paper's choice")
    n.add_argument("--action-noise-std", type=float, default=None,
                   help="exploration std in action units (default 0.1)")
    n.add_argument("--binary-noise-std", type=float, default=None,
                   help="separate std for the binary gene dims (default: same as "
                        "--action-noise-std). A sticky gene only flips once the noisy "
                        "action crosses +-0.5, which a continuous dim's std never does.")
    n.add_argument("--ou-theta", type=float, default=None, help="OU mean reversion (default 0.15)")
    n.add_argument("--ou-dt", type=float, default=None,
                   help="OU timestep (default 0.01); the noise decorrelates over "
                        "~1/(theta*dt) steps, so compare it to --max-episode-steps")

    # ---------------- eval ----------------
    e = sub.add_parser("eval", help="Evaluate a saved DDPG run")
    e.add_argument("--run", default="latest")
    e.add_argument("--episodes", type=int, default=3)
    e.add_argument("--model", default=None,
                   help="'final' (default), 'best', or a checkpoint name")
    e.add_argument("--seed", type=int, default=1000)
    e.add_argument("--max-episode-steps", type=int, default=None)
    e.add_argument("--action-scale", type=float, default=None)
    e.add_argument("--backend", default=DEFAULT_BACKEND, choices=list(BACKEND_CHOICES))
    e.add_argument("--no-plots", action="store_true")
    e.add_argument("--no-save", action="store_true")

    # ---------------- plot ----------------
    pl = sub.add_parser("plot", help="Render a DDPG run's training gallery")
    pl.add_argument("--run", default="latest")
    pl.add_argument("--window", type=int, default=20)

    # ---------------- assess ----------------
    a = sub.add_parser("assess", help="Compare runs on common start states")
    a.add_argument("--runs", nargs="+", required=True, help="DDPG run names/paths (ddpg_runs/)")
    a.add_argument("--td3-runs", nargs="*", default=[], help="TD3 run names/paths (td3_runs/)")
    a.add_argument("--ppo-runs", nargs="*", default=[], help="PPO run names/paths (ppo_runs/)")
    a.add_argument("--episodes", type=int, default=20)
    a.add_argument("--model", default=None, help="'best' or a checkpoint name; default: final")
    a.add_argument("--backend", default=DEFAULT_BACKEND, choices=list(BACKEND_CHOICES))

    return p


def cmd_train(args) -> int:
    from ppo.config import CONFIGS, BandConfig, RewardConfig, RewardWeights

    from .config import DDPGExperiment
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

    ddpg_overrides = {}
    for flag, key in (
        ("discount", "gamma"), ("batch_size", "batch_size"), ("buffer_size", "buffer_size"),
        ("learning_starts", "learning_starts"), ("tau", "tau"), ("train_freq", "train_freq"),
        ("gradient_steps", "gradient_steps"), ("noise_type", "noise_type"),
        ("action_noise_std", "action_noise_std"), ("binary_noise_std", "binary_noise_std"),
        ("ou_theta", "ou_theta"), ("ou_dt", "ou_dt"), ("eval_episodes", "eval_episodes"),
    ):
        value = getattr(args, flag)
        if value is not None:
            ddpg_overrides[key] = value
    if args.net_arch:
        ddpg_overrides["net_arch"] = tuple(int(v) for v in args.net_arch.split(","))
    if args.normalize_reward:
        ddpg_overrides["normalize_reward"] = True
    if args.checkpoint_every:
        ddpg_overrides["checkpoint_every"] = args.checkpoint_every
    if args.eval_every:
        ddpg_overrides["eval_every"] = args.eval_every

    exp = DDPGExperiment.from_code(
        args.code,
        band=band,
        reward=reward,
        tag=args.tag,
        total_timesteps=args.timesteps,
        max_episode_steps=args.max_episode_steps,
        learning_rate=args.lr,
        action_scale=args.action_scale,
        seed=args.seed,
        n_envs=args.n_envs,
        env_overrides=env_overrides,
        ddpg_overrides=ddpg_overrides,
    )
    run_dir = train(exp, backend=args.backend, resume_from=args.resume_from)
    print(f"\nrun complete -> {run_dir}")
    return 0


def cmd_eval(args) -> int:
    from .evaluate import evaluate_run

    report = evaluate_run(
        args.run,
        episodes=args.episodes,
        seed=args.seed,
        backend=args.backend,
        max_episode_steps=args.max_episode_steps,
        action_scale=args.action_scale,
        save=not args.no_save,
        make_plots=not args.no_plots,
        model=args.model,
    )
    print(f"\nevaluated {report.run_dir.name} ({report.exp.code}, ddpg)")
    print(report.summary_df.to_string(index=False))
    print("\naggregate:")
    for k in ("reward_sum_mean", "final_total_connected_mean", "final_total_power_mean"):
        if k in report.aggregate:
            print(f"  {k:32s} {report.aggregate[k]:.3f}")
    if report.eval_dir:
        print(f"\nartifacts -> {report.eval_dir}")
    return 0


def cmd_plot(args) -> int:
    from .plotting import training_gallery
    from .runs import resolve_run

    out = training_gallery(resolve_run(args.run), window=args.window)
    print(f"training gallery -> {out}")
    return 0


def cmd_assess(args) -> int:
    from .assess import assess

    df = assess(args.runs, args.ppo_runs, args.td3_runs, episodes=args.episodes,
                backend=args.backend, model=args.model)
    print(f"\nCOMMON START STATES (n={args.episodes}, seed 987654, deterministic)")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


def main(argv=None) -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")  # headless-safe plotting
    args = _build_parser().parse_args(argv)
    return {
        "train": cmd_train,
        "eval": cmd_eval,
        "plot": cmd_plot,
        "assess": cmd_assess,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
