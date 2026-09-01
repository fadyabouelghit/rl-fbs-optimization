"""Pilot: extend config 2-1-1 to a real learning-curve budget.

Motivation: the faithful 4096-step reruns only took 2 gradient updates and
could not show learning. An analytic-backend sweep showed 2-1-1 IS learnable
and that the historical ent_coef=0.5 caps performance (~25% below ent~0.01).
This pilot extends 2-1-1 on real MATLAB physics with tuned optimizer settings
while keeping the COST/REWARD FUNCTION UNCHANGED (beta 0.8, gamma 0,
fbs_weight 0.4), so the result stays comparable to the faithful runs.

Changed vs. historical (optimizer only):
    ent_coef      0.5  -> 0.01   (let the policy actually converge)
    learning_rate 1e-4 -> 3e-4
    n_steps       2048 -> 1024   (more frequent updates)
    total_timesteps 4096 -> 20000 (~19 updates instead of 2)

Unchanged: reward weights, action_scale 0.5, seed 0, 30-step episodes,
scenario (2 FBS, 2000x1500, 1 MBS), legacy single-band world.

The learning signal to watch is FBS-offloaded users over training (steps.csv /
episodes.csv final_fbs_connected), since total reward is macro-dominated.

Usage:
    python pilot_extend_211.py --backend analytic   # wiring check (seconds)
    python pilot_extend_211.py --backend matlab      # real pilot (~8 h)
"""
from __future__ import annotations

import argparse

from ppo import ExperimentConfig, train
from ppo.config import BandConfig, RewardConfig, RewardWeights


def build_pilot(total_timesteps: int) -> ExperimentConfig:
    return ExperimentConfig.from_code(
        "2-1-1",
        band=BandConfig(mode="legacy"),
        reward=RewardConfig(
            mode="legacy_blend",
            weights=RewardWeights(beta=0.8, gamma=0.0, fbs_weight=0.4, fbs_exponent=1.0),
        ),
        tag="pilot211_tuned",
        action_scale=0.5,          # unchanged from historical 2-1-x
        max_episode_steps=30,
        ent_coef=0.01,             # tuned down from 0.5
        learning_rate=3e-4,        # tuned up from 1e-4
        seed=0,
        total_timesteps=total_timesteps,
        ppo_overrides={
            "n_steps": 1024,
            "batch_size": 64,
            "checkpoint_every": 4000,
            "eval_every": 4000,
            "eval_episodes": 3,
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["matlab", "analytic"], default="matlab")
    p.add_argument("--timesteps", type=int, default=20000)
    args = p.parse_args()

    exp = build_pilot(args.timesteps)
    print(f"pilot 2-1-1: ent_coef={exp.ppo.ent_coef} lr={exp.ppo.learning_rate} "
          f"n_steps={exp.ppo.n_steps} total={exp.ppo.total_timesteps} "
          f"backend={args.backend} | reward beta={exp.env.reward.weights.beta} "
          f"gamma={exp.env.reward.weights.gamma} fbs_weight={exp.env.reward.weights.fbs_weight}")
    run_dir = train(exp, backend=args.backend)
    print(f"\npilot complete -> {run_dir}")


if __name__ == "__main__":
    main()
