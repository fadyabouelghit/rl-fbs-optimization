# Next runs for the letter

**For:** Claude Code, in the repo root (`/Users/fadya/Documents/MATLAB/GA_github/rl-fbs-pyqd`), using `./venv/bin/python`.
**Prepared:** 2026-10-09
**Read first:** `HANDOVER_letter.md` (rules, fairness, protocol).

Run the three steps **in order**. Each step uses all 8 workers, so never start one while another is running. Stop and report to Dr. Fady at the end of each step.

---

## Step 1: Failure-prone reruns (about 2 h, most important)

**Why:** the tuned winners (DDPG t27, TD3 t28) almost never fail, so E2 caught only one run that stayed stuck at "off". These three tuner-passed configs fail often in fair1. Rerunning them with the probe on shows whether their failures are the stuck-off trap.

**Task:**

1. Add a launch plan `e3_fail` in `letterexp/launch.py`:
   - arms: DDPG fair1 **t55**, DDPG fair1 **t19**, TD3 fair1 **t45**
   - parameters read from `tuning_runs/fair1/*_confirmation.json`, with **no overrides**
   - seeds 0 to 34, fair1 protocol, instrumented (probe + compact step log)
2. Reproducibility check first: run seed 0 of each arm and confirm `J_best` equals the fair1 value for that trial and seed **exactly**. Stop and report if any differs.
3. Run the plan:
   ```bash
   ./venv/bin/python -m letterexp.launch plan e3_fail
   ./venv/bin/python -m letterexp.launch run  e3_fail --workers 8
   ```

**Report, per config:**

- failures (J_best ≤ 0.4170), with a Wilson 95% interval
- how many failures are **switched off**, meaning J_best is exactly 0.4165, and how many are on but below 0.4165
- for each failed run:
  - whether a trap event lasted to the end of training (`letterexp.pins`)
  - the station's ON share in the last 20k steps, from `steps_compact.csv.gz`
- for each persistent trap: median `frac_prefer_on_q1` and median signed `dq_mu_q1_median` during the trap (critic prefers ON while the slope points toward OFF?)
- the same numbers for the runs that did not fail, for comparison

**Verdict to give:** do switched-off failures coincide with a persistent trap? Answer with counts, not adjectives.

---

## Step 2: Fix the hybrid before the big E4 campaign (about 1 h)

**Why:** the current hybrid (`ddpg_hybrid`, `td3_hybrid`) uses a straight-through Gumbel estimator. That still trains the on/off output through the critic's local slope, the very signal that points the wrong way during traps. The fix trains it on the critic's own on-vs-off comparison instead.

**Task:**

1. In `ddpg_hybrid/hybrid.py`, add an opt-in variant. Leave the existing classes unchanged.
   - **Rollouts:** sample the on/off dim as before (±1). Use argmax at evaluation.
   - **Actor loss, binary dim:** with z the binary logit and a* the actor's action with the binary dim set to its sample,
     `loss_b = - mean( sigmoid(z) * stopgrad[ Q1(s, a*|a_b=+1) - Q1(s, a*|a_b=-1) ] )`
   - **Actor loss, continuous dims:** the usual DPG loss, `- mean( Q1(s, a*) )`, with gradients flowing only into the continuous outputs.
   - The critic and target updates are unchanged from the current hybrid (target smoothing noise stays zero on the binary dim).
2. Register the variant as `ddpg_hybrid_qd` and `td3_hybrid_qd` in `tuning/`, the same way the existing hybrids are registered. Keep the same 5-knob search space (`gumbel_tau` stays as the 5th knob; it now only affects sampling).
3. Add a test in `tests/test_letter.py` showing that when Q(on) > Q(off) everywhere, one gradient step increases sigmoid(z).
4. Run the full test suite, then a 2-seed pilot of each new variant at the parents' other knobs (as `e4_pilot` does).
5. **Stop and report:** the code diff, test results and the pilot numbers.

---

## Step 3: Full E4 campaign with the fixed hybrid (about 10 h, run overnight)

**Only after Dr. Fady approves Step 2.**

```bash
export TUNING_DIR="$PWD/letter_runs/e4_tuning" LETTER_INSTRUMENT=1
for ALGO in ddpg_hybrid_qd td3_hybrid_qd; do
  ./venv/bin/python -m tuning search  --campaign fair1h --algo $ALGO --workers 8 --slim
  ./venv/bin/python -m tuning confirm --campaign fair1h --algo $ALGO --workers 8 --slim --top 8 --seeds 0 1 2 3 4
  ./venv/bin/python -m tuning confirm --campaign fair1h --algo $ALGO --workers 8 --slim --top 8 --seeds $(seq 0 34)
done
unset TUNING_DIR LETTER_INSTRUMENT
./venv/bin/python -m letterexp.analyze e4
```

Before launching, check that `TUNING_DIR` points into `letter_runs/` and **never** `tuning_runs/`. `analyze e4` may need the new algorithm names added. The comparison is the pooled finalists (all 8 × 35 seeds) of each pipeline against fair1's, for failures, plus winner vs winner for mean J.

---

## Not needed now

- **E5 (policy delay):** partly covered by the t45 rerun in Step 1. Revisit after Step 1.
- **E6 (generality):** scope only, as in the handover.

## Rules (unchanged)

- Don't modify `ppo/`.
- New runs go to `letter_runs/` only. Nothing writes to `ppo_runs/`, `ddpg_runs/`, `td3_runs/`, `tuning_runs/` or `training_log.csv`.
- Ask Dr. Fady before any git commit.
