#!/usr/bin/env bash
# Post-hoc evaluations of every trained agent, sequentially.
# Matches the archived MATLAB eval settings exactly: 5 deterministic episodes,
# seed 1000, model.zip, 40-step episodes -- so summary.csv is directly comparable
# to ppo_runs/run_2026-08-19_23-46-16_1-1-1_1fbs1mbs_potrec10/evals/20260820_063908/.
set -uo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p comparison/evals
ev () {  # ev <run_dir_name> <backend> <label>
  echo "[eval] $3  (run=$1 backend=$2)  $(date +%H:%M:%S)"
  ./venv/bin/python -m ppo eval --run "$1" --episodes 5 --seed 1000 \
      --backend "$2" > "comparison/evals/$3.log" 2>&1
  echo "[eval] $3 rc=$?"
}

# 1. the seed-0 pair, each on its own training backend -- these are the runs that
#    should reproduce the archived MATLAB evaluation numbers exactly.
FA=$(ls -td ppo_runs/*_potrec10_seed0_faithful 2>/dev/null | head -1)
FS=$(ls -td ppo_runs/*_potrec10_seed0 2>/dev/null | grep -v faithful | head -1)
[ -n "$FA" ] && ev "$(basename "$FA")" pyqd      full_seed0_faithful
[ -n "$FS" ] && ev "$(basename "$FS")" pyqd-fast full_seed0_fast
# cross-evaluate: same policy, other backend -- isolates backend from policy.
[ -n "$FA" ] && ev "$(basename "$FA")" pyqd-fast full_seed0_faithful_XevalFast

# 2. the halfway pair
for d in ppo_runs/*_half_faithful; do [ -d "$d" ] && ev "$(basename "$d")" pyqd      half_faithful; done
for d in ppo_runs/*_half_fast;     do [ -d "$d" ] && ev "$(basename "$d")" pyqd-fast half_fast;     done

# 3. the remaining seeds (1..9)
for S in 1 2 3 4 5 6 7 8 9; do
  d=$(ls -td ppo_runs/*_potrec10_seed${S} 2>/dev/null | head -1)
  [ -n "$d" ] && ev "$(basename "$d")" pyqd-fast "seed${S}"
done

# 4. common-start-state comparison across all 10 seeds (the repo's own tool:
#    one fixed set of start states from a seed never used in training/selection)
echo "[eval] assess_models across the 10 seeds  $(date +%H:%M:%S)"
RUNS=$(for S in 0 1 2 3 4 5 6 7 8 9; do ls -d ppo_runs/*_potrec10_seed${S} 2>/dev/null \
        | grep -v faithful | head -1 | xargs -n1 basename; done | tr '\n' ' ')
./venv/bin/python assess_models.py --runs $RUNS --episodes 20 --backend pyqd-fast \
    > comparison/evals/assess_models_10seeds.log 2>&1
echo "[eval] assess_models rc=$?"

# 5. compare our seed-0 evals against the archived MATLAB evaluation
./venv/bin/python comparison/compare_evals.py > comparison/RESULTS_evals_vs_matlab.txt 2>&1
cat comparison/RESULTS_evals_vs_matlab.txt
echo "[eval] ALL EVALUATIONS COMPLETE $(date +%H:%M:%S)"
