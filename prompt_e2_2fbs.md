# Prompt: 2-FBS noise sweep (e2_2fbs)

**For:** the runs chat, repo `rl-fbs-pyqd`, Python `./venv/bin/python`.
**Goal:** test contribution 4 ("exploration noise is not a fix") with 2 FBS. This is the 2-FBS version of E2.

## Already done (by the letter chat)

- I added the plan `e2_2fbs` to `letterexp/launch.py`. It is opt-in, nothing else changed, and it is not committed.
- Arms:
  - fair2 winners DDPG t15 and TD3 t51 (both use Gaussian binary noise)
  - each runs at binary_noise_std {0.1, 0.3, 0.5, 0.8, 1.2}, plus the tuned value as an anchor arm (DDPG 0.412, TD3 0.450)
  - seeds 0 to 19, fair2 protocol, instrumented
  - total 2 x 6 x 20 = 240 runs
- Parameters come from `letter_runs_2fbs/tuning/fair2/*_confirmation.json`. Only `binary_noise_std` moves.

## Rules

- **Protected paths:** do not touch `ppo/`, `tuning_runs/`, `letter_runs/`, `ppo_runs/`, `ddpg_runs/`, `td3_runs/` or `training_log.csv`.
- **Where outputs go:** everything goes under `letter_runs_2fbs/e2_2fbs/`, so always set `LETTER_RUNS_DIR=letter_runs_2fbs`.
- **Compute:** never run two 8-worker jobs at once.
- **Git:** ask Dr. Fady before any git commit, including the `launch.py` change.
- **Reporting style:** counts and exact numbers; no italics, no em dashes.

## Steps

**1. Dry run:**

    LETTER_RUNS_DIR=letter_runs_2fbs ./venv/bin/python -m letterexp.launch plan e2_2fbs

Check that it reports 240 runs (12 arms x 20 seeds), and that the tuned arms carry s_b 0.412 (DDPG) and 0.450 (TD3).

**2. Reproducibility check:** run seed 0 only.

    LETTER_RUNS_DIR=letter_runs_2fbs ./venv/bin/python -m letterexp.launch run e2_2fbs --seeds 0 --workers 8

Both `_bn_tuned_s0` runs must reproduce the fair2 seed-0 J exactly (DDPG t15 0.463324, TD3 t51 0.485205). On any mismatch, STOP and report.

**3. Full run** (about 3.5 h at 8 workers; resumable, so seed 0 is reused):

    LETTER_RUNS_DIR=letter_runs_2fbs nohup ./venv/bin/python -m letterexp.launch run e2_2fbs --workers 8 > letter_runs_2fbs/e2_2fbs_full.log 2>&1 &

**4. Analysis**, saved to `letter_runs_2fbs/analysis/e2_2fbs_by_run.csv` and `lab/comparison/RESULTS_letter_2fbs_E2.txt`.

Use the per-gene trap rule from Step C (`pins.py`, MIN_RUN=5, applied to fbs0 and fbs1). Per run, record:
- J_best, J_final, fail_best
- the Step A class of the best model on the 20 tuner test states (both_on / one_on / none_on / mixed)
- for each station: any trap, trap at the end, on-share in the last 20k steps
- escape and switch-back rates inside traps vs the formulas (special and general form, as in Step C)

Report per algo and per noise level (each out of 20):
- runs with at least one stuck-off period (any station), with Wilson 95% intervals
- runs that are not both_on
- failures
- mean J (sd)

Also test the trend across noise levels for each algo (Cochran-Armitage, or logistic regression of trap on s).

STOP and report.
