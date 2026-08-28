#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 40-step shaping sweep — three new arms to sit alongside 1fbs1mbs_potrec10.
#
# Every flag below reproduces 1fbs1mbs_potrec10's config exactly.
# The ONLY key that differs between these three runs and potrec10 is
#   env.reward.shaping
# (record_weight is pinned to 1.0 everywhere so it does not show up in a diff;
#  it is only read inside the potential_record branch, so it is inert here.)
#
# Verified before launch: each arm diffs from potrec10 in exactly 1 field.
#
# Run from the repo root:
#     bash scripts/run_40step_arms.sh
#
# Sequential on purpose — one MATLAB backend at a time.
# ~6.8 h per run + ~5 min eval, ~21 h total.
# ---------------------------------------------------------------------------
set -euo pipefail

PY=./venv/bin/python

COMMON=(
  --code 1-1-1
  --reward legacy_blend
  --record-weight 1.0
  --beta 1.0
  --fbs-weight 0.4
  --normalize-reward
  --action-mode delta
  --action-scale 0.05
  --normalize-obs
  --sticky-binaries
  --obs-tier-metrics
  --no-full-coverage-termination
  --discount 0.95
  --lr 3e-4
  --ent-coef 0.01
  --n-steps 1024
  --timesteps 25000
  --max-episode-steps 40
  --eval-every 2000
  --checkpoint-every 4000
  --seed 0
)

# train one arm, then evaluate it on 5 seeded episodes
run_arm () {
  local shaping="$1" tag="$2" n="$3"
  echo "=== ARM ${n}/3 : shaping = ${shaping}  -> ${tag} ==="
  $PY -m ppo train "${COMMON[@]}" --reward-shaping "$shaping" --tag "$tag" \
      > "full_run_${tag}.log" 2>&1
  local run
  run=$(ls -td ppo_runs/*_"${tag}" 2>/dev/null | head -1)
  if [ -n "$run" ] && [ -f "$run/model.zip" ]; then
    $PY -m ppo eval --run "$(basename "$run")" --episodes 5 > "eval_${tag}.log" 2>&1
    echo "ARM ${n} EVAL_DONE rc=$? run=$(basename "$run")" >> /tmp/arms40_progress
  else
    echo "ARM ${n} TRAINING_INCOMPLETE tag=${tag}" >> /tmp/arms40_progress
  fi
}

run_arm none      1fbs1mbs_raw_ep40 1
run_arm potential 1fbs1mbs_pot_ep40 2
run_arm record    1fbs1mbs_rec_ep40 3

echo "ALL_ARMS_DONE" >> /tmp/arms40_progress
echo "=== done — compare against 1fbs1mbs_potrec10 (potential_record, already run) ==="
