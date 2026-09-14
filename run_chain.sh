#!/usr/bin/env bash
# Strictly serial pipeline. Nothing ever runs concurrently:
#   1. wait for the in-flight faithful-path run (PID $1) to exit
#   2. build its MATLAB timing comparison
#   3. run potrec10 x 10 seeds, one after another -- each waits for the last
#   4. aggregate the sweep
set -uo pipefail
WAIT_PID="$1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p comparison/seed_sweep

echo "[chain] waiting for faithful run (PID ${WAIT_PID}) ..."
while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 20; done
echo "[chain] faithful run finished $(date +%H:%M:%S)"
grep -E "training done" full_run_potrec10_half_faithful.log || true

FD=$(ls -td ppo_runs/*_half_faithful | head -1)
./venv/bin/python comparison/compare_vs_matlab.py "$FD" \
  > comparison/RESULTS_faithful_vs_matlab.txt 2>&1
echo "[chain] wrote comparison/RESULTS_faithful_vs_matlab.txt"

echo "[chain] ===== starting 10-seed sweep, sequential ====="
SWEEP_T0=$(date +%s)
for SEED in 0 1 2 3 4 5 6 7 8 9; do
  echo "[chain] seed ${SEED} start $(date +%H:%M:%S)"
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
  RC=$?
  echo "[chain] seed ${SEED} done rc=${RC} in $(( $(date +%s) - T0 ))s"
done
echo "[chain] ALL 10 SEEDS DONE in $(( $(date +%s) - SWEEP_T0 ))s"
echo "[chain] PIPELINE COMPLETE $(date +%H:%M:%S)"
