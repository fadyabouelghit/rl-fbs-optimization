#!/usr/bin/env bash
# potrec10 x 10 seeds, STRICTLY SEQUENTIAL (one MATLAB-equivalent world at a time).
#
# Reproduces run_2026-08-19_23-46-16_1-1-1_1fbs1mbs_potrec10 exactly -- full
# 25000 timesteps (13 -> 25 rollouts of n_steps=1024 = 25600 executed steps) --
# varying ONLY --seed. Seed 0 is the archived MATLAB run's seed, so it doubles
# as a ground-truth regression check against MATLAB's logged trajectory.
#
# Backend: pyqd-fast (verified bit-identical to the faithful full-grid path).
# Threads pinned to 1 so the concurrent faithful-path timing run is unaffected.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p comparison/seed_sweep
START=$(date +%s)
for SEED in 0 1 2 3 4 5 6 7 8 9; do
  echo "=== seed ${SEED} started $(date +%H:%M:%S) ==="
  T0=$(date +%s)
  ./venv/bin/python -m ppo train \
    --code 1-1-1 --reward legacy_blend --reward-shaping potential_record \
    --record-weight 1.0 --beta 1.0 --fbs-weight 0.4 --normalize-reward \
    --action-mode delta --action-scale 0.05 --normalize-obs --sticky-binaries \
    --obs-tier-metrics --no-full-coverage-termination --discount 0.95 \
    --lr 3e-4 --ent-coef 0.01 --n-steps 1024 --timesteps 25000 \
    --max-episode-steps 40 --eval-every 2000 --checkpoint-every 4000 \
    --seed "${SEED}" --backend pyqd-fast --tag "potrec10_seed${SEED}" \
    > "comparison/seed_sweep/seed${SEED}.log" 2>&1
  echo "=== seed ${SEED} done in $(( $(date +%s) - T0 ))s ==="
done
echo "ALL 10 SEEDS DONE in $(( $(date +%s) - START ))s"
