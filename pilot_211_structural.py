"""Pilot: 2-1-1 with the structural RL fixes (SAC + shaped reward + normalized obs).

Motivation: the faithful reruns and the tuned pilot (pilot_extend_211.py) both
converged to the do-nothing policy — the terminal episode sits numerically on
the FBS-off baseline (694 connected, 0 FBS-served, J=0.448), the critic's
explained_variance stayed at ~0 and the policy std never left its init value.
Diagnosis: the dense per-step objective pays the do-nothing plateau, the
observation mixes x~[0,2000] with binary flags unnormalized, binary genes
flicker ~50%/step under an untrained Gaussian policy, and on-policy PPO at
~0.6 MATLAB steps/s gets only ~190 gradient updates in 8.7 h.

This pilot keeps the COST/OBJECTIVE UNCHANGED (legacy_blend, beta 0.8,
gamma 0, fbs_weight 0.4 — the per-step objective is still logged as
``reward_objective`` in steps.csv, directly comparable to the faithful runs,
and the exhaustive sweeps) and changes only the *training machinery*:

    algo            PPO -> SAC        (replay reuse + auto-tuned entropy)
    reward signal   dense J -> delta-J (potential shaping; return telescopes
                                       to J(s_T) - J(s_0))
    observations    raw -> [-1,1] normalized + per-tier connectivity feedback
    binary genes    threshold -> sticky (flip only when |a| >= 0.5)
    termination     full-coverage early-stop disabled
    discount        0.99 -> 0.95      (25-30 step episodes)

Unchanged: action_scale 0.5, seed 0, 30-step episodes, scenario (2 FBS,
2000x1500, 1 MBS), legacy single-band world, static user map.

Two-stage usage (recommended): pretrain cheaply on the analytic backend, then
fine-tune on real MATLAB physics (the model warm-starts via resume_from; the
fine-tune stage runs exactly --timesteps additional steps):

    python pilot_211_structural.py --backend analytic                 # wiring check (~1 min)
    python pilot_211_structural.py --pretrain-steps 100000            # analytic pretrain + MATLAB fine-tune
    python pilot_211_structural.py --algo ppo                         # PPO control with the same structure

The learning signal to watch is ``reward_objective`` (and final_fbs_connected)
against the do-nothing baseline J=0.448 and the 1-FBS brute-force optimum
J=0.522 (exhaustive grid sweep of the world).
"""
from __future__ import annotations

import argparse

from ppo import ExperimentConfig, train
from ppo.config import BandConfig, RewardConfig, RewardWeights

STRUCTURAL_ENV = {
    "normalize_obs": True,
    "binary_mode": "sticky",
    "obs_tier_metrics": True,
    "terminate_on_full_coverage": False,
}


def build_pilot(args, total_timesteps: int, tag: str) -> ExperimentConfig:
    ppo_overrides = {
        "algo": args.algo,
        "gamma": 0.95,                 # discount, not the reward-weight gamma
        "checkpoint_every": 4000,
        "eval_every": 2000,
        "eval_episodes": 3,
    }
    if args.algo == "sac":
        ppo_overrides.update(
            buffer_size=200_000, learning_starts=500, batch_size=256,
            train_freq=1, gradient_steps=1, sac_ent_coef="auto",
        )
    else:  # PPO control: pilot_extend_211 optimizer settings + the structure
        ppo_overrides.update(n_steps=1024, batch_size=64)

    return ExperimentConfig.from_code(
        "2-1-1",
        band=BandConfig(mode="legacy"),
        reward=RewardConfig(
            mode="legacy_blend",
            shaping="potential",
            weights=RewardWeights(beta=0.8, gamma=0.0, fbs_weight=0.4, fbs_exponent=1.0),
        ),
        tag=tag,
        action_scale=args.action_scale,   # historical 0.5 kept for comparability
        max_episode_steps=30,
        ent_coef=0.01,                    # PPO control only; SAC auto-tunes
        learning_rate=3e-4,
        seed=args.seed,
        total_timesteps=total_timesteps,
        env_overrides=dict(STRUCTURAL_ENV),
        ppo_overrides=ppo_overrides,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["matlab", "analytic"], default="matlab")
    p.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    p.add_argument("--timesteps", type=int, default=20000,
                   help="steps on --backend (the fine-tune stage when pretraining)")
    p.add_argument("--pretrain-steps", type=int, default=0,
                   help="analytic-backend steps to pretrain before the main stage")
    p.add_argument("--action-scale", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    resume_from = None
    if args.pretrain_steps > 0:
        exp_pre = build_pilot(args, args.pretrain_steps, tag="pilot211_struct_pre")
        print(f"[stage 1/2] analytic pretrain: {args.pretrain_steps} steps ({args.algo})")
        resume_from = train(exp_pre, backend="analytic")
        print(f"[stage 1/2] pretrain run -> {resume_from}")

    # With resume_from (reset_num_timesteps=False), SB3 runs total_timesteps
    # ADDITIONAL steps on top of the loaded counter — pass the fine-tune
    # budget only.
    exp = build_pilot(args, args.timesteps, tag="pilot211_struct")
    print(f"[stage {'2/2' if resume_from else '1/1'}] {args.backend}: "
          f"{args.timesteps} steps ({args.algo}), shaping=potential, "
          f"normalize_obs + sticky binaries + tier obs, discount 0.95")
    run_dir = train(exp, backend=args.backend, resume_from=resume_from)
    print(f"\npilot complete -> {run_dir}")
    print("watch: steps.csv reward_objective vs the do-nothing baseline (0.448) "
          "and episodes.csv final_fbs_connected")


if __name__ == "__main__":
    main()
