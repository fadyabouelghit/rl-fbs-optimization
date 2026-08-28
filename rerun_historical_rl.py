"""Re-run the historical PPO trainings with the new (fully-logged) pipeline.

The original runs were bare notebook-saved models (only model.zip +
reward_weights.json), so their training curves and per-step logs were never
recorded, and their seed / action_scale are unrecoverable. This script
retrains each configuration through the ppo package so every run now gets
steps.csv / episodes.csv / monitor.csv / progress.csv + plots.

Configs recovered exactly from the saved models + reward_weights.json:
    code   FBS  world       ent_coef  (from the model)
    1-1-1   1   2000x1500     0.5      run_044 (gamma 0.0)
    1-1-2   1   2000x1500     0.5      run_039 (gamma 0.1)
    1-2-1   1   4000x3000     0.7      run_062 (gamma 0.0)
    1-2-2   1   4000x3000     0.7      run_061 (gamma 0.1)
    2-1-1   2   2000x1500     0.5      run_045 (gamma 0.0)
    2-1-2   2   2000x1500     0.5      run_046 (gamma 0.1)
    2-2-1   2   4000x3000     0.7      run_057 (gamma 0.0)
    2-2-2   2   4000x3000     0.7      run_056 (gamma 0.1)

Not recoverable (set here, documented in each run's provenance):
    seed              -> fixed to SEED below (originals were single-seed, value lost)
    action_scale      -> per-spec (user: 0.5 for all but the last two)
    max_episode_steps -> 30 (run_068's surviving monitor confirms 30; era default)

Usage:
    python rerun_historical_rl.py --batch 1FBS            # real MATLAB, background-friendly
    python rerun_historical_rl.py --batch 1FBS --backend analytic   # instant wiring check
    python rerun_historical_rl.py --batch 2FBS --action-scale-last2 0.9
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from ppo import ExperimentConfig, train
from ppo.config import CONFIGS, BandConfig, RewardConfig, RewardWeights

SEED = 0                    # fresh fixed seed; the originals' seeds are lost
MAX_EPISODE_STEPS = 30
TOTAL_TIMESTEPS = 4096
N_STEPS = 2048              # SB3 default the old PPOTrainer used
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
BETA = 0.8
FBS_WEIGHT = 0.4
FBS_EXPONENT = 1.0


@dataclass
class Spec:
    code: str
    ent_coef: float
    hist_run: int          # the historical run this reproduces (for the tag)


# ent_coef is scenario-dependent: 0.5 for the small world, 0.7 for the large.
SPECS_1FBS = [
    Spec("1-1-1", 0.5, 44),
    Spec("1-1-2", 0.5, 39),
    Spec("1-2-1", 0.7, 62),
    Spec("1-2-2", 0.7, 61),
]
SPECS_2FBS = [
    Spec("2-1-1", 0.5, 45),
    Spec("2-1-2", 0.5, 46),
    Spec("2-2-1", 0.7, 57),   # "last two" -> action_scale from --action-scale-last2
    Spec("2-2-2", 0.7, 56),
]


def build_config(spec: Spec, action_scale: float) -> ExperimentConfig:
    z = int(spec.code.split("-")[2])
    reward = RewardConfig(
        mode="legacy_blend",
        weights=RewardWeights(
            beta=BETA,
            gamma=CONFIGS[z]["gamma"],
            fbs_weight=FBS_WEIGHT,
            fbs_exponent=FBS_EXPONENT,
        ),
    )
    return ExperimentConfig.from_code(
        spec.code,
        band=BandConfig(mode="legacy"),
        reward=reward,
        tag=f"redo{spec.hist_run}",
        total_timesteps=TOTAL_TIMESTEPS,
        max_episode_steps=MAX_EPISODE_STEPS,
        ent_coef=spec.ent_coef,
        action_scale=action_scale,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        ppo_overrides={"n_steps": N_STEPS, "batch_size": BATCH_SIZE},
    )


def run_batch(specs, action_scale, action_scale_last2, backend):
    n = len(specs)
    print(f"=== rerun batch: {n} configs | backend={backend} | "
          f"action_scale={action_scale} (last2={action_scale_last2}) ===\n")
    results = []
    for i, spec in enumerate(specs):
        # The "last two" of the full 8-run set are 2-2-1 / 2-2-2.
        a_scale = action_scale_last2 if spec.code in ("2-2-1", "2-2-2") else action_scale
        cfg = build_config(spec, a_scale)
        print(f"[{i+1}/{n}] code={spec.code} (redo of run_{spec.hist_run:03d}) "
              f"ent_coef={spec.ent_coef} action_scale={a_scale} gamma={cfg.env.reward.weights.gamma}")
        t0 = time.time()
        run_dir = train(cfg, backend=backend)
        dt = time.time() - t0
        print(f"      -> {run_dir.name}  ({dt/60:.1f} min)\n")
        results.append((spec, run_dir))
    print("=== batch complete ===")
    for spec, run_dir in results:
        print(f"  run_{spec.hist_run:03d} ({spec.code}) -> {run_dir}")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", choices=["1FBS", "2FBS", "all"], default="1FBS")
    p.add_argument("--backend", choices=["matlab", "analytic"], default="matlab")
    p.add_argument("--action-scale", type=float, default=0.5,
                   help="action_scale for all configs except the last two (default 0.5)")
    p.add_argument("--action-scale-last2", type=float, default=None,
                   help="action_scale for codes 2-2-1 / 2-2-2 (required for the 2FBS batch)")
    args = p.parse_args()

    if args.batch == "1FBS":
        specs = SPECS_1FBS
    elif args.batch == "2FBS":
        specs = SPECS_2FBS
    else:
        specs = SPECS_1FBS + SPECS_2FBS

    last2 = args.action_scale_last2
    if last2 is None:
        needs_last2 = any(s.code in ("2-2-1", "2-2-2") for s in specs)
        if needs_last2 and args.backend == "matlab":
            raise SystemExit(
                "Refusing to launch: codes 2-2-1 / 2-2-2 need --action-scale-last2 "
                "(their action_scale is unrecoverable and you flagged them as different)."
            )
        last2 = args.action_scale  # analytic wiring check only

    run_batch(specs, args.action_scale, last2, args.backend)


if __name__ == "__main__":
    main()
