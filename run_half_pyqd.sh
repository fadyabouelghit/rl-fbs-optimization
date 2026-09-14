#!/usr/bin/env bash
# potrec10 at the true halfway point (12500 -> 13 rollouts = 13312 steps),
# reproducing run_2026-08-19_23-46-16_1-1-1_1fbs1mbs_potrec10 exactly except
# for total_timesteps and the physics backend. $1 = backend, $2 = tag.
set -euo pipefail
BACKEND="$1"; TAG="$2"
exec ./venv/bin/python -m ppo train \
  --code 1-1-1 --reward legacy_blend --reward-shaping potential_record \
  --record-weight 1.0 --beta 1.0 --fbs-weight 0.4 --normalize-reward \
  --action-mode delta --action-scale 0.05 --normalize-obs --sticky-binaries \
  --obs-tier-metrics --no-full-coverage-termination --discount 0.95 \
  --lr 3e-4 --ent-coef 0.01 --n-steps 1024 --timesteps 12500 \
  --max-episode-steps 40 --eval-every 2000 --checkpoint-every 4000 --seed 0 \
  --backend "$BACKEND" --tag "$TAG"
