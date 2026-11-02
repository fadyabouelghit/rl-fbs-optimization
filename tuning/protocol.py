"""The fairness contract, written down once.

A comparison between PPO, TD3 and DDPG is only about the algorithms if
everything else is held equal. The environment already is -- all three build
their env half through ``ppo.config.ExperimentConfig.from_code``. What this
module pins is the rest of it: the budget, the objective, the seed protocol
and the search space rule. Every trial of every algorithm is built from one
``Protocol`` instance, so "same budget, same objective, same seeds" is a
property of the code rather than of a shell script that has to be kept in
sync across three launchers.

The environment settings below are exactly the ones the potrec10 campaign
used (``lab/ddpg_tune_batch*.sh`` and ``lab/launchers/run_200_seeds.sh``), so
a tuned config is directly comparable to every run already on disk.

Why ``objective_model = "best"``: the in-training ``CsvEvalCallback`` selects
``best_model.zip`` on its own seed-12345 evaluation states, which are
disjoint from both the seed-987654 tuning states and the held-out reporting
states. Selection and scoring therefore never share a draw.

Why no pruner: across the 54 DDPG runs that reached 100k, Spearman rank
correlation between J at 25k and J at 100k is +0.26, and the best run overall
ranked 29th of 54 at 25k. Every early-stopping rule would have killed it, so
trials run to full budget.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

from ppo.config import RewardConfig, RewardWeights

ALGOS: Tuple[str, ...] = ("ppo", "td3", "ddpg")

# Opt-in variants with a stochastic Bernoulli head on the on/off dims (the
# letter's Stage 4 remedy). Kept out of ALGOS so the fair1 three-way tooling
# (report, pages) is unchanged; tuned with exactly the same Protocol.
HYBRID_ALGOS: Tuple[str, ...] = ("ddpg_hybrid", "td3_hybrid",
                                 "ddpg_hybrid_qd", "td3_hybrid_qd")

# Never used for training, model selection or the tuning objective -- the
# held-out draw recommendation 5 of the tuning-methods page asks for.
REPORT_SEED = 246_813


@dataclass(frozen=True)
class Protocol:
    """Everything held equal across the three algorithms."""

    # --- environment (identical for all three) ---------------------------
    code: str = "1-1-1"
    backend: str = "pyqd-fast"
    max_episode_steps: int = 40
    action_scale: float = 0.05

    # --- shared hyperparameters: FIXED, not searched ---------------------
    # Held at one common value for all three algorithms, so the search
    # measures only what is algorithm-specific. The cost is real and worth
    # stating: DDPG's own tuned point was lr 1e-4 / gamma 0.9, and it gives
    # up roughly 0.009 J here (seed-0 measurements: gamma 0.95 -> 0.4622 vs
    # 0.9 -> 0.4662; lr 3e-4 -> 0.4608 vs 1e-4 -> 0.4654). The gain is that
    # 60 trials now cover a 5-dimensional space instead of an 8-dimensional
    # one, and that no algorithm carries a learning rate or discount chosen
    # by how much attention it happened to get.
    #
    # gamma 0.95    the potrec10 value PPO and TD3 both ran; DDPG's
    #               second-best, 0.004 behind its own 0.9.
    # lr 3e-4       SB3's PPO default and the best TD3 arm; DDPG's
    #               second-best, 0.005 behind its own 1e-4.
    # batch 256     SB3's TD3/DDPG default, and a valid PPO minibatch at
    #               every n_steps in the search (the smallest is 512).
    # net (64,64)   SB3's *PPO* default, and the only architecture all three
    #               can express: ppo/ passes no policy_kwargs, so PPO cannot
    #               be moved to TD3/DDPG's 400,300. Fixing everyone to PPO's
    #               is what makes the architecture genuinely shared instead
    #               of a knob two of the three get to search.
    fixed_learning_rate: float = 3e-4
    fixed_gamma: float = 0.95
    fixed_batch_size: int = 256
    fixed_net_arch: Tuple[int, ...] = (64, 64)

    # --- budget (identical for all three) --------------------------------
    total_timesteps: int = 100_000
    n_envs: int = 1
    eval_every: int = 4_000
    eval_episodes: int = 10
    # 0: no periodic checkpoints. The objective scores best_model, and the
    # J-over-time curve that a future multi-fidelity decision would need is
    # already in eval_log.csv at 4k resolution. 25 checkpoints x ~4 MB x 240
    # runs is ~23 GB of files nothing reads. Re-train a winner with
    # checkpointing on if you ever want its mid-training policies.
    checkpoint_every: int = 0

    # Per-step env logging (steps.csv) is ~26 MB of every ~27 MB run and
    # nothing in the tuning pipeline reads it -- the objective comes from a
    # post-training rollout and the J curve from eval_log.csv. Dropping it
    # after a trial is scored takes a 720-run campaign from ~19 GB to
    # ~0.7 GB. The file is still written during training (SB3 gives no hook
    # to suppress it without editing ppo/), so peak disk is one file per
    # concurrent worker, not one per run.
    keep_step_log: bool = True

    # --- objective (identical for all three) -----------------------------
    # Mean final J over a fixed set of common start states, scored on the
    # model the run itself selected as best.
    objective_episodes: int = 20
    objective_seed: int = 987_654          # assess_models.ASSESS_SEED
    objective_model: str = "best"

    # --- seed protocol ---------------------------------------------------
    # Stage 1 searches at one seed; stage 2 re-runs the finalists across
    # several and ranks on mean - sd, so reliability sits inside the
    # objective rather than in a footnote.
    search_seed: int = 0
    confirm_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    confirm_top_k: int = 5

    # --- search ----------------------------------------------------------
    n_trials: int = 60
    n_startup_trials: int = 12             # random warm-up before the GP leads

    def reward(self) -> RewardConfig:
        """The potrec10 reward: legacy blend, potential+record shaping."""
        return RewardConfig(
            mode="legacy_blend",
            shaping="potential_record",
            record_weight=1.0,
            reward_offset=0.0,
            reward_gain=1.0,
            weights=RewardWeights(beta=1.0, gamma=0.0, fbs_weight=0.4),
        )

    def env_overrides(self) -> dict:
        return {
            "action_mode": "delta",
            "normalize_obs": True,
            "binary_mode": "sticky",
            "obs_tier_metrics": True,
            "terminate_on_full_coverage": False,
        }

    def shared_algo_overrides(self) -> dict:
        """Budget/logging fields every ``*Params`` dataclass shares."""
        return {
            "total_timesteps": self.total_timesteps,
            "n_envs": self.n_envs,
            "eval_every": self.eval_every,
            "eval_episodes": self.eval_episodes,
            "checkpoint_every": self.checkpoint_every,
            "normalize_reward": True,
            "verbose": 0,
        }


DEFAULT = Protocol()

# A seconds-not-hours protocol for tests and for `--smoke`. Same shape, tiny
# numbers: the point is to exercise train -> select -> score end to end, not
# to learn anything. eval_every must stay below total_timesteps or no
# best_model.zip is ever written and there is nothing to score.
SMOKE = replace(
    DEFAULT,
    backend="analytic",
    total_timesteps=600,
    eval_every=200,
    eval_episodes=1,
    checkpoint_every=0,
    objective_episodes=2,
    n_trials=4,
    n_startup_trials=2,
    confirm_seeds=(0, 1),
    confirm_top_k=2,
)
