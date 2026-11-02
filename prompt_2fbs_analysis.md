# Prompt: 2-FBS (fair2) analysis and instrumented reruns

**For:** the chat managing runs in `rl-fbs-pyqd` (`/Users/fadya/Documents/MATLAB/GA_github/rl-fbs-pyqd`, Python `./venv/bin/python`).
**From:** the letter experiments chat, 2026-10-13.
**Read first:** `HANDOVER_letter.md`, `NEXT_RUNS_letter.md`, project doc `claude/claims-v2.md`.

---

## Context

`letter_runs_2fbs/run_fair2.sh` repeats the fair1 tuning protocol, unchanged, on scenario 2-1-1 (2 FBS, 1 MBS).
- PPO and TD3 are complete; DDPG is in its 35-seed confirmation.
- The runs were not instrumented (`LETTER_INSTRUMENT` unset). So there is no `probe_log.csv` and no `steps_compact.csv.gz`, but every run kept `best_model.zip`, `model.zip`, `vecnormalize.pkl` and `episodes.csv`.

**Status so far (pooled finalists, 8 x 35 seeds)**

| Algo | Failures | Switched off (J exactly 0.4165) | Mean J |
|---|---|---|---|
| PPO | 0/280 | 0 | 0.523 |
| TD3 | 7/280 | 7 | 0.466 |

All 8 TD3 finalists have policy_delay 1. The do-nothing level is still 0.4165: both FBS off.

**Why this work is needed**
- With 2 FBS, J equals do-nothing only when both are off. One FBS stuck off is invisible to the failure measure.
- Training-episode counts (`final_num_active_fbs`, last 20k steps) suggest exactly that:

| Algo | Both on | One on | None on |
|---|---|---|---|
| PPO | 82% | 18% | 0% |
| TD3 | 19% | 67% | 14% |
| DDPG (partial) | 24% | 55% | 21% |

- These counts include exploration noise, so they are only indicative. The steps below measure the effect properly and test the mechanism.

---

## Rules (unchanged from the letter work)

**Do not touch**
- Do not modify `ppo/`.
- Nothing writes to `tuning_runs/`, `letter_runs/` (the 1-FBS results), `ppo_runs/`, `ddpg_runs/`, `td3_runs/` or `training_log.csv`.

**Where new work goes**
- New outputs go under `letter_runs_2fbs/` and `lab/comparison/`.
- New code is opt-in: add new modules or new functions; do not change existing defaults.

**Compute**
- Never run two 8-worker jobs at once.
- Do not start Step C while `run_fair2.sh` is running.

**Process**
- Ask Dr. Fady before any git commit.
- Stop and report at each STOP. Do not continue until Dr. Fady replies.
- Only tuner-passed configs. Parameters come from `letter_runs_2fbs/tuning/fair2/*_confirmation.json` with no overrides.

**Reporting style:** counts and exact numbers, no italics, no em dashes or double hyphens, plain sentences.

---

## Step 0: wait for fair2 to finish

Wait until `fair2.log` prints `=== FAIR2 ALL DONE`. Then report the DDPG table in the same format as above:
- failures
- switched-off failures
- mean J
- winner
- per-finalist n / mean / sd / failures / params

---

## Step A: re-score every best model per station (no training, about 1 h)

**Goal:** measure how many stations the scored policy actually switches on, and whether a station that stays off is locked off by a saturated actor.

**Script:** new file `letterexp/fbs2_rescore.py`. For every confirmation run of PPO, TD3 and DDPG in fair2 (8 x 35 each), take run dirs from `*_confirmation.json` → `finalists[].run_dirs`. Then:

1. **Evaluate the best model.** Use `best_model.zip`, deterministic, on the same 20 start states the tuner used: `am.start_states(exp.env, protocol.objective_episodes, seed=protocol.objective_seed)`. Call `evaluate_run(..., model="best", save=False, make_plots=False)`, exactly as `tuning/objective.py::score_run` does.
   - **Check first:** the recomputed J must equal the confirmation value for that run. Stop and report on any mismatch.
2. **Per test state, at the final step of the episode:**
   - read `fbs0_power_status` and `fbs1_power_status` from `report.episodes[i].state_df.iloc[-1]`
   - count the active stations (0, 1 or 2)
   - record `fbs_connected` and `reward_objective`
3. **Classify each run by its 20 test states:**
   - `both_on`: at least 15 of 20 end with 2 on
   - `one_on`: at least 15 of 20 end with exactly 1 on
   - `none_on`: at least 15 of 20 end with 0 on
   - `mixed`: anything else

   For `one_on` runs, also record whether it is the same station off in all of those states (`fixed_off_station`), or it varies.
4. **Actor saturation (DDPG and TD3 only).** For each station j, take the actor's deterministic output on that station's on/off dim, `mu_j`, over the final states of the 20 episodes.
   - Report the share of states where station j is off and `mu_j < -0.96`.
   - Call a station locked off when that share is at least 95% of its off states. This is the same 0.96 / 95% rule as `letterexp/probe.py`.
   - **PPO:** use the Gaussian mean of the policy for the same check.
5. **Save** per-run rows to `letter_runs_2fbs/analysis/rescore_best.csv`, and a summary to `lab/comparison/RESULTS_letter_2fbs_A.txt`.

**Report, per algorithm**
- Counts of `both_on` / `one_on` / `none_on` / `mixed` out of 280.
- Mean J (with sd) per class.
- Locked-off stations: how many `one_on` runs have the off station locked off by the actor rule.
- `fixed_off_station` share among `one_on` runs.
- Wilson 95% intervals on the class shares.
- Fisher exact tests: PPO vs TD3 and PPO vs DDPG on the share of runs that are not `both_on`.
- **Is one station a reasonable choice?** Compare J of PPO `both_on` vs PPO `one_on`, and TD3 `one_on` vs PPO `both_on` (Welch). If two stations are not clearly better, say so; the partial-lock-in reading depends on it.

**STOP and report.**

---

## Step B: critic analysis on saved best models (no training, about 1 h)

**Goal:** test the letter's mechanism (R3) per station. In states where a station is off: does the critic rate switching it ON higher, while the slope at the actor's output pushes it OFF?

**Probe set**
1. Build a 2-FBS probe set (256 random reset states), with the existing builder: `./venv/bin/python -m letterexp.probe build --code 2-1-1 --backend pyqd-fast`.
2. **Check first** that `build` honours `--code`; if it does not, add that as an opt-in fix.
3. It must save to `letter_runs/probe_sets/probe_2-1-1_pyqd-fast_n256_s20261009.npz`. That is the only allowed new file under `letter_runs/`; otherwise write it under `letter_runs_2fbs/probe_sets/`.
4. In 2-1-1 the on/off dims are the `power_status` genes of fbs0 and fbs1. Read the indices from `state_labels(env.config)`. Do not hard-code 4.

**Script:** new file `letterexp/fbs2_critic.py`. For every DDPG and TD3 confirmation run (load with `DDPG.load` / `TD3.load` on CPU), on the probe states, for each station j, restricted to the states where station j is off:
- `mu_j`: actor output on dim j
- `gap_j = Q1(s, a | a_j = +1) - Q1(s, a | a_j = -1)`, with the other dims at the actor's output
- `slope_j = dQ1/da_j` at the actor's output (autograd, as in `letterexp/probe.py::_slope`)
- `Q1` only for TD3; also report `Q2` once as a check

**Per run and station, record**
- share with `gap_j > 0`
- share with `slope_j < 0`
- share with both
- medians of `gap_j` and `slope_j`
- the locked-off flag (share of off states with `mu_j < -0.96` at least 95%)

**Save** to `letter_runs_2fbs/analysis/critic_best.csv`, and a summary to `lab/comparison/RESULTS_letter_2fbs_B.txt`.

**Report, grouped by the Step A class**
- (i) the 7 TD3 switched-off failures, and any DDPG ones
- (ii) `one_on` runs: the off station vs the on station
- (iii) `both_on` runs

For each group:
- the number of locked-off stations
- among locked-off stations, how many have a critic that prefers ON in most states while the slope pushes OFF (both shares above one half)
- how many have a critic that prefers OFF (the TD3 t45 pattern in 1-FBS)

Also produce one figure in the style of `letter/figures/fig3b_critic_along_action`, for the clearest TD3 locked-off station: Q along a_j from -1 to +1, median, IQR and 30 sample states. Save PNG and PDF to `lab/comparison/figures_letter/2fbs/`.

**Caveat to state in the report:** these are scored checkpoints, and probe states are random reset states.

**STOP and report.**

---

## Step C: instrumented reruns of failure-prone fair2 finalists (about 3 to 6 h; only after Dr. Fady approves)

**Goal:** the 1-FBS E3_fail evidence (trap timing, escape law inside traps) for 2 FBS.

**Before writing any code, check** that `letterexp/probe.py`, `letterexp/compact_steps.py`, `letterexp/pins.py` and `letterexp/analyze.py` handle two binary genes. The 1-FBS analysis uses the label `fbs0_power_status` in places. Generalise to a list of labels (fbs0 and fbs1), opt-in, keeping 1-FBS outputs byte-identical. Run `tests/test_letter.py` and add a 2-gene test.

**Plan**
1. Add a plan `e3_fail2` in `letterexp/launch.py`. It reads parameters from `letter_runs_2fbs/tuning/fair2/*_confirmation.json` (no overrides), code 2-1-1, seeds 0 to 34, instrumented. Output goes under `letter_runs_2fbs/e3_fail2/`. Arms:
   - TD3 fair2 t35 (3 off failures in 35)
   - TD3 fair2 t04 (2 off failures)
   - the two DDPG finalists with the most failures plus most `one_on` runs after Step A; propose them in the Step A report and let Dr. Fady confirm
2. **Reproducibility check:** run seed 0 of each arm and confirm J equals the fair2 value for that trial and seed exactly. Stop and report on any mismatch.
3. Run with 8 workers.

**Report, per arm and per station**
- traps (rule from `pins.py`, applied per gene)
- trap at the scored checkpoint vs failure and vs `one_on`
- in-trap escape and switch-back rates from `steps_compact` against the general form:
  - escape: `Q((k - mu_off)/s)`
  - switch-back: `Phi((-k - mu_on)/s)`
  - use the median post-flip action for `mu_on`, as done for 1-FBS
- critic preference and slope sign during traps

**STOP and report.**

---

## Step D: scope only (do not run)

The QD head (`ddpg_hybrid_qd`, `td3_hybrid_qd`) has not been through fair2. Without it, the 2-FBS result can support the mechanism and the PPO contrast, but not the remedy.

Estimate the cost of running both QD variants through the full fair2 pipeline. The TD3 fair2 pipeline took about 7.5 h. Then ask Dr. Fady whether to queue them after Step C.

---

## What to send back at each STOP

- The command(s) run and the output file paths.
- The tables above, with counts out of the stated totals.
- Anything that did not reproduce exactly.
- Anything that contradicts the 1-FBS claims in `claims-v2.md` (R1 to R7). Say plainly which claim and by how much.
