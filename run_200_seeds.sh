#!/usr/bin/env bash
# 200 additional potrec10 seeds (10..209) on the fast path, STRICTLY SEQUENTIAL,
# then evaluate all 210 (0..209) and run the common-start-state comparison.
#
# Config is byte-identical to the archived MATLAB potrec10 run; only --seed varies.
# Resumable: a seed whose run dir already holds model.zip is skipped.
set -uo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p comparison/seed_sweep comparison/evals
T0=$(date +%s)

have_run () { ls -td ppo_runs/*_potrec10_seed$1 2>/dev/null | grep -v faithful | head -1; }

echo "===== STAGE 1: training seeds 10..209 (sequential) ====="
for SEED in $(seq 10 209); do
  D=$(have_run "$SEED")
  if [ -n "$D" ] && [ -f "$D/model.zip" ]; then echo "[train] seed $SEED already done, skipping"; continue; fi
  S0=$(date +%s)
  ./venv/bin/python -m ppo train \
    --code 1-1-1 --reward legacy_blend --reward-shaping potential_record \
    --record-weight 1.0 --beta 1.0 --fbs-weight 0.4 --normalize-reward \
    --action-mode delta --action-scale 0.05 --normalize-obs --sticky-binaries \
    --obs-tier-metrics --no-full-coverage-termination --discount 0.95 \
    --lr 3e-4 --ent-coef 0.01 --n-steps 1024 --timesteps 25000 \
    --max-episode-steps 40 --eval-every 2000 --checkpoint-every 4000 \
    --seed "$SEED" --backend pyqd-fast --tag "potrec10_seed${SEED}" \
    > "comparison/seed_sweep/seed${SEED}.log" 2>&1
  echo "[train] seed $SEED rc=$? in $(( $(date +%s) - S0 ))s  (elapsed $(( ($(date +%s)-T0)/60 )) min)"
done
echo "===== STAGE 1 DONE in $(( ($(date +%s)-T0)/60 )) min ====="

echo "===== STAGE 2: evaluating all 210 seeds (5 eps, seed 1000, deterministic) ====="
E0=$(date +%s)
for SEED in $(seq 0 209); do
  D=$(have_run "$SEED"); [ -z "$D" ] && { echo "[eval] seed $SEED MISSING"; continue; }
  ./venv/bin/python -m ppo eval --run "$(basename "$D")" --episodes 5 --seed 1000 \
      --backend pyqd-fast --no-plots > "comparison/evals/seed${SEED}_eval.log" 2>&1
  [ $(( SEED % 25 )) -eq 0 ] && echo "[eval] seed $SEED done  (elapsed $(( ($(date +%s)-E0)/60 )) min)"
done
echo "===== STAGE 2 DONE in $(( ($(date +%s)-E0)/60 )) min ====="

echo "===== STAGE 3: common-start-state comparison across all 210 ====="
A0=$(date +%s)
RUNS=$(for SEED in $(seq 0 209); do have_run "$SEED" | xargs -n1 basename; done | tr '\n' ' ')
./venv/bin/python assess_models.py --runs $RUNS --episodes 20 --backend pyqd-fast \
    > comparison/evals/assess_models_210seeds.log 2>&1
echo "[assess] rc=$? in $(( ($(date +%s)-A0)/60 )) min"

echo "===== ALL DONE in $(( ($(date +%s)-T0)/60 )) min ====="
