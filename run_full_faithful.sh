#!/usr/bin/env bash
# Full-length faithful counterpart to ppo_runs/*_potrec10_seed0 (pyqd-fast, 25600 steps).
# Identical config and seed; ONLY --backend differs. Gives a full-length direct
# faithful-vs-fast comparison, and a full-length wall clock against MATLAB's 6.80 h.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
./venv/bin/python -m ppo train \
  --code 1-1-1 --reward legacy_blend --reward-shaping potential_record \
  --record-weight 1.0 --beta 1.0 --fbs-weight 0.4 --normalize-reward \
  --action-mode delta --action-scale 0.05 --normalize-obs --sticky-binaries \
  --obs-tier-metrics --no-full-coverage-termination --discount 0.95 \
  --lr 3e-4 --ent-coef 0.01 --n-steps 1024 --timesteps 25000 \
  --max-episode-steps 40 --eval-every 2000 --checkpoint-every 4000 --seed 0 \
  --backend pyqd --tag potrec10_seed0_faithful
echo "=== full faithful run complete; comparing against the fast twin ==="
FA=$(ls -td ppo_runs/*_potrec10_seed0_faithful | head -1)
FS=$(ls -td ppo_runs/*_potrec10_seed0 | grep -v faithful | head -1)
./venv/bin/python comparison/compare_paths.py "$FA" "$FS" \
  > comparison/RESULTS_full_faithful_vs_fast.txt 2>&1
cat comparison/RESULTS_full_faithful_vs_fast.txt
echo "=== now running all post-hoc evaluations ==="
./run_all_evals.sh
