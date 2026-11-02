# Handover: experiments for the IEEE letter (actor pinning with on/off genes)

**For:** Claude Code, running on Dr. Fady's Mac in the repo root
(`/Users/fadya/Documents/MATLAB/GA_github/rl-fbs-pyqd`), using the repo's `venv`.
**Prepared:** 2026-10-09. The code below is written and tested. Your job is to run it stage by
stage, stop at every CHECKPOINT, and report to Dr. Fady.

Use `./venv/bin/python` for every command (or run `source venv/bin/activate` first). Run long
launches inside `tmux` or with `nohup ... &` so that closing a terminal does not kill them.
Everything can be resumed: re-running a launch command skips runs that already finished.

---

## 1. The fairness rule (Dr. Fady's decision, not negotiable)

Only configs that passed the tuner get compared as "tuned". In practice:

| Stage | How it stays fair |
|---|---|
| E2 (binary-noise grid) | **Labeled ablation.** Only `binary_noise_std` moves away from the tuner-passed point (DDPG fair1 t27, TD3 fair1 t28). The tuned value runs as its own **anchor arm** (DDPG 0.369 as a 6th arm; TD3's 0.807 replaces the grid's 0.8). |
| E4 (hybrid remedy) | The hybrid goes through **the full fair1 tuner pipeline** with the same `Protocol` and the same 5-knob budget (the `binary_noise_std` slot becomes the Gumbel temperature): 60 trials at seed 0, then the top 8 at seeds 0 to 4, then all 8 at seeds 0 to 34, ranked on mean minus sd. It is compared with fair1's own records. The 2-seed `e4_pilot` only checks the implementation and is never the comparison. |
| E5 (policy delay) | **Labeled ablation** of t28. Delay 1 is t28 itself (anchor). |
| E6b (2-1-1) | No config has been tuned on 2-1-1. Running the 1-1-1 winners there is not a tuned comparison. Scope it at CHECKPOINT 6 (see section 6). |

Base configs are **never typed in by hand**. `letterexp.runner.base_params("ddpg_t27")` reads
them from `tuning_runs/fair1/*_confirmation.json` (read-only).

---

## 2. Facts found while building this. Raise them at the checkpoints named.

1. **The tuned winners almost never fail** (failure = J ≤ 0.4165 + 0.0005). In fair1 at 35 seeds:
   DDPG t27 1/35, t50 0/35; TD3 t28 0/35. The established counts (DDPG 18/280, TD3 15/280) come
   from the *other* finalists. All 15 TD3 failures are in the two `policy_delay=4` configs
   (t41 2/35, t45 13/35). DDPG failures by finalist: t27 1, t50 0, t14 1, t55 4, t39 3, t17 2, t19 5, t3 2.
   Consequences:
   - E4: a winner-vs-winner failure test cannot be significant (1/35 vs 0/35, Fisher p = 1).
     `analyze e4` therefore also reports **pooled finalists** (all 8 × 35 seeds of each
     pipeline, both tuner-passed). Use that as the primary test for the failure claim
     (CHECKPOINT 4a).
   - E2: few failures should be expected around t27/t28, so P2(b) ("failures at s_b = 1.2 > 0")
     may come out zero simply because the base config is good. Raise this at CHECKPOINT 2a.
2. **s_b = 0.1 is not "the continuous-noise case"** for these configs: their continuous noise is
   0.247 (t27) and 0.30 (t28). Keep 0.1 as the low end of the grid but don't describe it as
   "no fix" in the letter. This caveat is already written into the E2 pre-registration.
3. **The four pinned seed-0 DDPG runs** (`t3_bn0.5_lr2e-5`, ...) are pre-fair1 hand-tuned configs
   (lr 2e-5 etc.), not tuner-passed. They are fine evidence for C2 (the flip formula is a
   property of the sticky rule, not of the config). They are **not** evidence that tuned configs pin.
4. **State-dependent saturation shows up early.** In a 10k-step pyqd-fast run of t27 seed 0, the
   actor was saturated on 99% of probe states at 2k steps, but half at +1 and half at -1. The
   actor sees the gene's current value in its observation, so it can learn "keep the current
   value". From an off gene, that policy is just as stuck as an all-states pin at -1. So the
   probe also logs **trapped**: at least 95% of the probe states whose gene is OFF have
   mu_b < -0.96. Pin events now have a direction of off, on or mixed. The prompt's pin
   definition is unchanged and still reported. Trap events are the failure-relevant
   variant used for escapes and "stuck at end". **Confirm these definitions at CHECKPOINT 0,
   before the E2 pre-registration is written.**
5. **The historical flip rate depends on the window chosen.** A container dry run of E1 on
   `t3_bn0.5_lr2e-5` measured off→on 143/96,089 = 1.49e-3 (CI 1.25e-3 to 1.75e-3; predicted
   1.35e-3) and switch-back 120/138 = 0.870 (predicted 0.841). The window was "from the first
   1000-step block after which every block is ≥ 95% off". The earlier figure of 1.24e-3 used a
   different window. Report both and say which window each used.
6. **The optimum computes to 0.4926**, not 0.4925 (rounding; `assess_models.objective(w, 722, 107, ...)`).

---

## 3. What was built (all changes uncommitted; review with `git status` / `git diff`)

**Small opt-in additions to existing files** (defaults unchanged; tests prove it):

- `ddpg/train.py`, `td3/train.py`: keyword-only `extra_callbacks`, `model_factory`, `config_extra` (all default `None`).
- `tuning/`: two new algorithm names, `ddpg_hybrid` and `td3_hybrid`, in `HYBRID_ALGOS`. `ALGOS` is unchanged, so fair1 tooling and pages are unaffected. Their search spaces and seed trials are in `spaces.py`. The seed trials are the same pre-fair1 starting points the parents used, not the fair1 winners. The changes:
  - `study._RUNS_ENV`: run-directory routing for the hybrids
  - `objective._load`: loads hybrid models
  - `worker.py`: hybrid training, plus opt-in `LETTER_INSTRUMENT=1`, which adds the probe to a tuner trial
  - `cli.py`: the hybrids are accepted as `--algo` choices
- `ppo/` is untouched.

**New code:**

| Path | What |
|---|---|
| `letterexp/probe.py` | Fixed 256-state probe set + `ProbeCallback` → `probe_log.csv`, `probe_raw.npz` |
| `letterexp/compact_steps.py` | `steps_compact.csv.gz` (gene before/after, executed action, J) for flip analysis |
| `letterexp/pins.py` | Pin, trap, escape and flip definitions |
| `letterexp/stats.py` | Wilson, Clopper-Pearson, Fisher, Welch, chi-square (Monte-Carlo p for small cells) |
| `letterexp/runner.py` | One run per subprocess: train with instrumentation, then score best_model and model.zip |
| `letterexp/launch.py` | Plans `smoke`, `s0_10k`, `repro`, `e2`, `e4_pilot`, `e5`; resumable pool; reuses identical runs |
| `letterexp/e1_flip.py` | Stage 1 |
| `letterexp/analyze.py` | `prereg`, `repro`, `e2`, `e3`, `e4`, `e5` → `lab/comparison/RESULTS_letter_*.txt`, figures |
| `ddpg_hybrid/`, `td3_hybrid/` | Bernoulli head on the on/off dim (straight-through Gumbel); argmax at evaluation |
| `tests/test_letter.py` | Probe does not perturb training (bitwise); sticky rule; hybrid head; hybrid train/eval |

**Verified in a Linux container** (SB3 2.9.0 = the Mac venv's version, torch 2.14, Python 3.13):

- Full suite: 188 passed, 8 skipped. Before the changes it was 182 passed, 8 skipped; the 6 new tests are in `test_letter.py`.
- Final weights are **bitwise identical** with and without the probe and compact-step callbacks, for DDPG, TD3, DDPG-hybrid and TD3-hybrid.
- The whole pipeline ran end to end on the analytic backend: prereg, e2, e5 (with run reuse), e3, the hybrid tuner search and confirm, e4, and the figures.
- A pyqd-fast 10k run was also checked.

The analytic-backend numbers mean nothing; they only show the code paths work.

**Note on the hybrid:** `ddpg_hybrid/hybrid.py` copies SB3 2.9.0's `TD3.train` and adds one line
(target smoothing noise = 0 on the binary dims). `_check_version()` refuses to run on any other
SB3 version.

All new runs go to `letter_runs/<experiment>/`. Probe sets go to `letter_runs/probe_sets/`. The hybrid tuner campaign
goes to `letter_runs/e4_tuning/fair1h/`. Nothing writes to `ppo_runs/`, `ddpg_runs/`, `td3_runs/`,
`tuning_runs/` or `training_log.csv` (`letterexp.paths.assert_allowed` enforces it for launches).
`letter_runs/_transfer/` (code.tgz, letter_code.tgz) holds leftover transfer archives from the handover and can be deleted.

---

## 4. Rules (from Dr. Fady's brief)

- Do not modify `ppo/`. Run the test suite after any edit to `ddpg/`, `td3/` or `tuning/`.
- fair1 protocol unless told otherwise. The settings are in `tuning/protocol.py DEFAULT`:
  - code 1-1-1, potrec10, 100k steps, pyqd-fast
  - lr 3e-4, gamma 0.95, batch 256, net 64,64
  - score = mean final J of best_model over the 20 start states (seed 987654); model.zip is also scored
- n ≥ 20 seeds per arm for failure-rate claims. Report Wilson 95% intervals, Fisher exact for counts and Welch for mean J. All of these are built into `analyze`.
- **Stop at every CHECKPOINT. Do not start the next stage until Dr. Fady replies.**
- Report results that contradict C1 to C4 plainly. Do not tune the setup until a claim passes.
- Ask Dr. Fady before committing anything to git.

---

## 5. Stage by stage

### Stage 0: instrumentation (no long runs)

```bash
./venv/bin/python -c "import stable_baselines3 as s; print(s.__version__)"     # must print 2.9.0
git status --short && git diff --stat                                          # ppo/ must not appear
./venv/bin/python -m pytest                                                     # all pass (~4 min)
./venv/bin/python -m letterexp.probe build --backend analytic
./venv/bin/python -m letterexp.probe build --backend pyqd-fast
./venv/bin/python -m letterexp.launch run smoke  --workers 2                   # ~1 min, analytic
./venv/bin/python -m letterexp.launch run s0_10k --workers 2                   # ~2-3 min, pyqd-fast 10k
./venv/bin/python -m letterexp.probe show --run "$(ls -d letter_runs/s0_10k/ddpg_runs/run_*)" --rows 8
./venv/bin/python -m letterexp.probe show --run "$(ls -d letter_runs/s0_10k/td3_runs/run_*)"  --rows 8
./venv/bin/python -m letterexp.launch run repro  --workers 2                   # ~10 min: t27 & t28 seed 0, full protocol
./venv/bin/python -m letterexp.analyze repro
```

`repro` must print **IDENTICAL** for both (fair1 seed 0: t27 0.4841142879, t28 0.4884644857).
Identical means two things: the probe does not perturb training on the Mac, and fair1's 35-seed
records are valid baselines for E4. If either run differs, stop and report it, because every
later comparison against fair1 depends on it.

**CHECKPOINT 0.** Report:

- a sample of `probe_log.csv` for DDPG and TD3
- the test result, and `git diff --stat` showing `ppo/` untouched
- the repro verdict
- a request for Dr. Fady to confirm the pin, trap and escape definitions (section 2, item 4) before E2 is pre-registered

### Stage 1 (E1): flip formula (~1 min, no training)

```bash
./venv/bin/python -m letterexp.e1_flip          # reads the 4 pinned runs in ddpg_runs/ read-only
```

Outputs: `lab/comparison/RESULTS_letter_E1.txt` (+ `.json`) and `figures_letter/F1_flip_rates.{png,pdf}`.
Replace the placeholder VERDICT line at the bottom with a verdict for C2 (supported, not supported
or inconclusive) and the reason. With 23 cells at 95%, one or two cells outside their CI is
expected by chance; say so if it happens.

### Stage 2 (E2): does noise cause pinning? 220 runs

Pre-register **first**. The command refuses to run once any E2 result exists:

```bash
./venv/bin/python -m letterexp.analyze prereg e2       # writes P1/P2 + a hash into RESULTS_letter_E2.txt
./venv/bin/python -m letterexp.launch plan e2 --seeds 0 1
./venv/bin/python -m letterexp.launch run  e2 --seeds 0 1 --workers 8           # pilot: 22 runs, ~20 min
./venv/bin/python -m letterexp.analyze e2
./venv/bin/python -m letterexp.launch plan e2                                    # full-run estimate
```

**CHECKPOINT 2a.** Report:

- the pilot tables
- the measured minutes per run (from `launch status e2`)
- the full estimate: 220 runs, about 1,540 core-min, about 3.2 h at 8 workers, about 0.7 GB kept
- section 2, item 1 (few failures expected near t27/t28)

On go-ahead (the pilot is skipped automatically):

```bash
./venv/bin/python -m letterexp.launch run e2 --workers 8
./venv/bin/python -m letterexp.analyze e2              # -> RESULTS_letter_E2.txt (prereg kept, hash checked), F2
```

**CHECKPOINT 2b.** Report the full results, with a plain-language verdict for each of P1 and P2.

### Stage 3 (E3): why the actor cannot unpin (analysis only)

```bash
./venv/bin/python -m letterexp.analyze e3              # -> RESULTS_letter_E3.txt, F3
```

The key line is the share of trapped checkpoints where the critic prefers "on". Write a C3
verdict. If no E2 run ends trapped, say that E3 has no stuck group to analyse; do not invent one.

### Stage 4 (E4): the minimal remedy

Implementation check first (4 runs, ~10 min):

```bash
./venv/bin/python -m letterexp.launch run e4_pilot --workers 4
./venv/bin/python -m letterexp.analyze e4
```

**CHECKPOINT 4a.** Show:

- `ddpg_hybrid/hybrid.py` (HybridActor, HybridMixin) and the two `train.py` files
- the pilot
- the tuner plan and cost below, and the pooled-finalists primary test (section 2, item 1)

On go-ahead, run the hybrid through the fair1 pipeline. Do one variant at a time; do not run
both at once on 8 cores.

```bash
export TUNING_DIR="$PWD/letter_runs/e4_tuning" LETTER_INSTRUMENT=1
for ALGO in ddpg_hybrid td3_hybrid; do
  ./venv/bin/python -m tuning search  --campaign fair1h --algo $ALGO --workers 8 --slim
  ./venv/bin/python -m tuning confirm --campaign fair1h --algo $ALGO --workers 8 --slim --top 8 --seeds 0 1 2 3 4
  ./venv/bin/python -m tuning confirm --campaign fair1h --algo $ALGO --workers 8 --slim --top 8 --seeds $(seq 0 34)
done
unset TUNING_DIR LETTER_INSTRUMENT
./venv/bin/python -m letterexp.analyze e4              # -> RESULTS_letter_E4.txt, F4
```

Cost per variant: 60 + 8×4 + 8×30 = 332 runs, about 39 core-h, about 4.8 h at 8 workers. Both
variants: about 9.7 h and about 2 GB. `--slim` drops steps.csv after scoring, but
`steps_compact.csv.gz` is kept because `LETTER_INSTRUMENT=1` is set. Before launching, check that
`TUNING_DIR` points into `letter_runs/` and **never** `tuning_runs/`.

Report back to Dr. Fady when this finishes (CHECKPOINT 4b).

### Stage 5 (E5): policy delay, controlled

```bash
./venv/bin/python -m letterexp.launch plan e5          # 105 runs; delay-1 seeds 0-19 are reused from E2's anchor
./venv/bin/python -m letterexp.launch run  e5 --workers 8      # ~85 new runs, ~1.2 h
./venv/bin/python -m letterexp.analyze e5              # -> RESULTS_letter_E5.txt, T1_policy_delay.{csv,md,png,pdf}
```

### Stage 6: generality (scope only; ask before running anything)

- **6a, user layouts:** not supported today. `ppo/pyqd_bridge.py` hardcodes `matlab_user_positions(..., seed=0)` (line ~664). Supporting it would take four things:
  - a `user_seed` field on the world config, passed through the pyqd and analytic backends and the map-cache keys. That touches `ppo/`, so Dr. Fady must approve it explicitly.
  - a new exhaustive grid sweep per layout for the do-nothing baseline and the optimum (`assess_models` hardcodes 694 / 722 / 107 for seed 0), so that "failure" and "gap closed" mean something
  - matching changes on the GA side, which also uses seed 0
  - an acknowledgement that the tuned configs were tuned on layout 0 only

  Report this; do not implement it.
- **6b, 2-1-1 / multi-band:** no tuner-passed config exists for these codes. A fair version means a fair1-style campaign per algorithm on the new code: about 300 runs per algorithm, and 2-1-1 steps are slower. The cheap version, running the 1-1-1 winners there, has to be labeled as a transfer test, not a tuned comparison. `letterexp.runner.apply_band` is a stub on purpose.

  Write both options with costs, then stop at **CHECKPOINT 6**.

---

## 6. Deliverables checklist

- `lab/comparison/RESULTS_letter_E1.txt` … `E5.txt`, each with:
  - the run list, seeds and configs. The result JSONs are in `letter_runs/<exp>/results/`; the run list can be printed with `launch status`.
  - the statistics
  - a plain-language verdict per prediction: supported, not supported or inconclusive
- An E6 scoping note: put it in `RESULTS_letter_E6.txt`.
- `lab/comparison/figures_letter/` with F1 to F4 as PNG and PDF, and T1.
- A new dated section in `lab/NOTES.md` (**Letter experiments -- <date>**) covering:
  - what ran and what it cost
  - what held and what did not
  - every result that contradicts C1 to C4, stated plainly
  - section 2 of this handover

## 7. If something goes wrong

- A run crashed: `letter_runs/<exp>/logs/<run_id>.log`, then re-run the same launch command (it resumes).
- `ledger.csv` with a header in the middle: launches run with 1 worker until each experiment's ledger exists. If a ledger still gets corrupted, report it; don't hand-edit fair1 files.
- `probe set ... missing`: run `python -m letterexp.probe build --backend <backend>`.
- A pre-registration hash error: the E2 prereg block was edited. Restore it; never rewrite it after the data exists.
