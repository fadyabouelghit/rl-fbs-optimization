# PPO Pipeline — Architecture & Artifacts

Reference for the Python PPO track. Everything lives in the
[`ppo/`](../ppo/) package; `train_ppo.py` and `ppo_experiment.py` remain as
thin compatibility shims for old notebooks. The MATLAB physics wrappers and
their dependency closure live in [`matlab/`](../matlab/).

---

## 1. The big picture

```
                    ┌────────────────────────────────────────────────┐
                    │                 ppo package                    │
 CLI / notebook ──► │ config.py      ExperimentConfig (X-Y-Z + band) │
 python -m ppo      │ train.py       run dir + logging + callbacks  │
                    │ evaluate.py    rollouts + artifacts + plots   │
                    │ plotting.py    publication-style gallery      │
                    │ runs.py        v2 / v1 / bare run discovery   │
                    └───────┬───────────────────────┬────────────────┘
                            │ env.py                │
                            │ FlyingBaseStationEnv  │
                            ▼                       ▼
      pyqd_bridge.PyqdSinrBackend   matlab_bridge.MatlabSinrBackend   matlab_bridge.AnalyticSinrBackend
      (real physics, in-process)     (real physics, engine-side world) (numpy stand-in, ~10k steps/s)
              │                                   │
   pyqd_channel.antenna.generate           ppo_world_setup.m  ── builds antennas/sites/cache ONCE,
   pyqd_channel.layout.power_map           ppo_sinr_eval.m    ── per-step SINR, scalars-only marshalling
   + a numpy port of SINREvaluation.m                │
              │                            SINREvaluation.m + QuaDRiGa   (the radio physics)
   (same physics, no MATLAB: maps agree to 3.3e-6 dB)
```

Design points:

- **Interchangeable backends.** The env never talks to MATLAB directly; it
  calls a `SinrBackend`. `backend="pyqd"` (the default) is the real QuaDRiGa
  physics computed in-process; `backend="pyqd-fast"` is the same numbers,
  bit-for-bit, ~160× cheaper per FBS map; `backend="matlab"` is the original
  MATLAB/QuaDRiGa stack, kept as the reference oracle; `backend="analytic"`
  runs the entire pipeline (training, logging, eval, plots) in milliseconds
  for tests and iteration, with uncalibrated numbers.
- **The pyqd world stays in the Python process.** There is no marshalling
  boundary at all: antennas, MBS maps and the fixed user map are plain numpy
  living on the backend object. Sites, antennas and the per-band MBS maps are
  built once in `PyqdSinrBackend.__init__` (the analogue of
  `ppo_world_setup.m`), and each step only computes the per-FBS maps.
- **The MATLAB world stays in MATLAB.** `ppo_world_setup.m` persists
  antennas + the MBS map cache in engine appdata; `ppo_sinr_eval.m` reads it
  back per step. Only scalars cross the Python↔MATLAB boundary (the old
  train_ppo.py round-tripped the full map cache every step).
- **Dual band modes.** `band.mode="legacy"` reproduces the original
  single-band world (old agents stay loadable); `band.mode="multi"` is the
  dual-band world (per-FBS band flag, base-MBS coverage/capacity slot
  expansion, same-band interference).

---

## 2. Quick start

```bash
# fast end-to-end pipeline check on the analytic backend (~10 s)
python -m ppo smoke

# real training (legacy world, like the old harness) on the pyqd backend
python -m ppo train --code 1-1-1 --timesteps 5000 --seed 0

# same physics, bit-identical, ~160x cheaper per FBS power map
python -m ppo train --code 1-1-1 --timesteps 5000 --seed 0 --backend pyqd-fast

# multi-band training: agent controls FBS bands + MBS capacity
python -m ppo train --code 2-2-1 --band multi --fbs-band agent \
                    --mbs-capacity agent --reward controlled_blend

# evaluate any run (new or years-old): 5 seeded deterministic episodes
python -m ppo eval --run latest --episodes 5
python -m ppo eval --run run_039 --state 800,800,100,10.5,1

python -m ppo list                 # all runs (v2 + old harness + bare)
python -m ppo plot --run latest    # re-render the training gallery
python -m ppo plot --compare run_A run_B --metric reward_sum
```

Python / notebook:

```python
from ppo import ExperimentConfig, BandConfig, train, evaluate_run

exp = ExperimentConfig.from_code(
    "1-1-1",
    band=BandConfig(mode="multi", fbs_band="agent", mbs_capacity="agent"),
    total_timesteps=5000, seed=0,
    ppo_overrides={"checkpoint_every": 1000, "eval_every": 1000},
)
run_dir = train(exp)                          # backend="analytic" for dry runs
report  = evaluate_run(run_dir, episodes=5)   # artifacts + plots under run_dir/evals/
report.summary_df
```

---

## 3. Configuration model

`ExperimentConfig.from_code("X-Y-Z", ...)` — three axes:

| Axis | Meaning | Presets |
|---|---|---|
| X | FBS count | any int |
| Y | scenario | 1 = 1 MBS, 2000×1500 hex; 2 = 2 MBS, 4000×3000 custom (`ppo/config.py: SCENARIOS`) |
| Z | cost config | 1: γ=0.0; 2: γ=0.1 (`CONFIGS`) |

Nested blocks (all serialized into `experiment_config.json`, version 2):

- **`env.world`** — geometry, users, SINR threshold, QuaDRiGa scenario.
- **`env.band`** — `mode` (`legacy`/`multi`), `fbs_band`
  (`coverage`/`capacity`/`agent`), `mbs_capacity` (`off`/`on`/`agent`).
  Agent-controlled genes extend the state/action vectors (6th gene per FBS
  + one trailing gene per MBS).
- **`env.reward`** — `mode`:
  - `legacy_blend` — the historical train_ppo.py formula (ε-padded);
  - `controlled_blend` — bills only agent-controlled power (active FBS gene
    powers incl. gated macro-capacity carriers), no ε padding. The old name
    `ga_blend` is still accepted;
  - `sum_rate` — sum spectral efficiency over connected users.
- **`ppo`** — SB3 hyperparameters + `seed`, `n_envs`, `checkpoint_every`,
  `eval_every`.

---

## 4. Run directory artifacts (`ppo_runs/run_<ts>_<code>[_mb][_tag]/`)

| File | Written by | Contents |
|---|---|---|
| `experiment_config.json` | run_logging | full v2 config + provenance (git sha, package versions, backend, status) |
| `model.zip` | train | final SB3 model |
| `best_model.zip` | CsvEvalCallback | best in-training eval model (`eval_every > 0`) |
| `checkpoints/model_<steps>.zip` | CheckpointEveryCallback | periodic snapshots (`checkpoint_every > 0`) |
| `steps.csv` | EnvInfoLoggingCallback | **every training step**: reward + decomposition, all connectivity/power/rate metrics, full flat state (fbs0_x, …, band flags, MBS capacity genes) — buffered writes |
| `episodes.csv` | EnvInfoLoggingCallback | per-episode aggregates + episode-end metrics |
| `progress.csv` | SB3 logger | optimizer diagnostics (losses, KL, entropy, fps) |
| `monitor[_i].csv` | SB3 Monitor | episode reward/length/time (one per worker) |
| `eval_log.csv` | CsvEvalCallback | seeded deterministic eval curve during training |
| `run.log` | run_logging | human-readable event log |
| `plots/` | plotting.training_gallery | training dashboards (png + pdf) |
| `evals/<ts>/` | evaluate.evaluate_run | see §5 |

One ledger row per run is appended to `training_log.csv` at the repo root
(status, hyperparameters, final metrics).

## 5. Evaluation artifacts (`<run_dir>/evals/<ts>/`)

| File | Contents |
|---|---|
| `eval_config.json` | run/model/schema, episodes, seeds, initial state, resolved experiment |
| `ep<k>_trajectory.csv` | per-step MultiIndex state (same format as the old test_logs CSVs — plot_trajectories.py-compatible) |
| `ep<k>_metrics.csv` | per-step metrics incl. per-tier splits + reward terms |
| `ep<k>_users.csv` | user positions (final step) |
| `mbs.csv` | true MBS coordinates |
| `summary.csv` / `summary.json` | per-episode rows + aggregate mean/std/min/max |
| `plots/` | trajectory map, altitude/power profile, step metrics, tier stack, cross-episode summary |

`evaluate_run(..., mirror_test_logs=True)` (CLI: `--mirror-test-logs`) also
writes the old `test_logs/<code>/<run>_<ts>_*` files for existing tooling.

---

## 6. Fast testing

Three layers, fastest first:

1. **Unit/integration suite** — `./venv/bin/pytest` (~13 s, 105 tests, no
   MATLAB): env mechanics, reward math vs hand computations, band gene
   plumbing, logging, run discovery, **real PPO training reproducibility**
   on the analytic backend.
2. **pyqd fidelity tests** — included in the suite above
   (`tests/test_pyqd_backend.py`, ~4 s, 25 tests, still no MATLAB): the
   MATLAB user-position stream reproduced bit-exactly, a precomputed MBS map
   compared against the archived MATLAB `.mat` cache, the swap/clamp sampling
   convention, tier invariants, `sum(tx_power)` semantics, and a full replay
   of a completed MATLAB GA run's metrics. The three tests that read the old
   repo's ground-truth artifacts skip themselves when those paths are absent,
   so the suite never hard-depends on them.
3. **Pipeline smoke** — `python -m ppo smoke` (~10 s): full multi-band
   train → checkpoint → in-training eval → evaluation → plot gallery, with
   artifact assertions.
4. **MATLAB bridge tests** — `PPO_MATLAB_TESTS=1 ./venv/bin/pytest
   tests/test_matlab_backend.py` (~20 s with a warm map cache): world build,
   legacy + multi-band evaluation, tier invariants, determinism. These are
   the only 3 tests skipped by default.

Interactive speed-ups:

- `--backend pyqd-fast` is bit-identical to `pyqd` and ~160× cheaper per FBS
  map; it is the right choice for any long run, and the only reason it is not
  the default is that `pyqd`'s cost profile matches the MATLAB baseline's,
  which keeps published timings comparable.
- pyqd has no session concept — nothing to warm up, nothing to attach to.
- MBS power maps are cached on disk: `cache_pyqd_maps/` (`.npz`, keyed by a
  SHA-256 over every map parameter plus the `pyqd-channel` version) and
  `cache_mbs_maps/` (MATLAB's own `.mat` cache). The two caches are separate
  directories on purpose; their key schemes are unrelated.
- `get_shared_session()` reuses one engine across repeated evaluations in a
  process; run `matlab.engine.shareEngine` in a MATLAB console and the
  bridge attaches to it instantly instead of cold-starting (~20 s saved per
  session). MATLAB backend only.

## 7. Reproducibility & parallelism

- `ppo.seed` seeds python/numpy/torch and every env; a fixed
  `(seed, n_envs)` pair reproduces a run bit-for-bit (asserted in
  `tests/test_train_integration.py`). User positions stay pinned to a fixed
  seed (mt19937ar, seed 0) on an isolated stream. The pyqd backend
  reproduces that MATLAB stream bit-for-bit in Python (`ppo/matlab_rng.py`:
  MATLAB maps seed 0 onto MT19937's own default seed 5489, and `randi` is
  `floor(rand*span)+lo` off the 53-bit double stream — `numpy`'s
  `RandomState(0)` is a *different* stream and would silently change the
  demand map), so pyqd and MATLAB runs share the same 1000 users.
- `n_envs > 1` (opt-in) uses SubprocVecEnv; each worker builds its own
  backend and is seeded `seed + worker_index`. On pyqd that is a numpy world
  plus a one-off MBS power-map precompute, which the on-disk cache reduces to
  a disk read after the first ever run; on matlab it is a full engine
  (~1–2 GB each). Runs
  remain reproducible for the same `n_envs`; changing `n_envs` changes the
  rollout interleaving (treat it like any other hyperparameter — that is
  why serial remains the default).
- `train(..., resume_from="run_...")` warm-starts the optimizer from a
  previous run's model into a fresh run dir (provenance recorded).

## 8. Legacy compatibility

| Generation | Layout | Loadable? |
|---|---|---|
| v2 (this package) | `run_<ts>_<code>/` + versioned config | native |
| v1 (old ppo_experiment.py) | `run_NNN/` + flat config json | config auto-migrated |
| bare (notebook era) | `run_*/ppo_fbs_agent*.zip` | env reconstructed from the model's observation-space bounds (num FBS, world size) + `*_reward_weights.json` |

Old imports keep working: `from train_ppo import PPOTrainingConfig,
PPOTrainer, FlyingBaseStationEnv, RewardWeights, run_sinr_evaluation` and
`from ppo_experiment import ExperimentConfig, train, test, SCENARIOS`.

Two legacy defects surfaced during the overhaul (both fixed):

1. **The old MATLAB call no longer ran on this branch.** SINREvaluation's
   no-band-args fallback assigns MBS slots to the *capacity* band, which
   asserts on a single-band cache — so the original train_ppo.py was broken
   after the multi-band migration. The legacy path now passes explicit
   all-coverage band ids (numerically identical to the pre-migration
   physics) via `ppo_sinr_eval.m`.
2. **`env.mbs_x`/`env.mbs_y` were transposed.** The old env exposed the
   swapped-frame row (y values) as `mbs_x`; notebook plots using them drew
   MBS sites transposed. The backend now exposes true coordinates.

## 9. MATLAB-side helpers

| File | Role |
|---|---|
| [ppo_world_setup.m](../matlab/ppo_world_setup.m) | build antennas (band_frequencies-aware), sites (hex or explicit), pack + x↔y swap, precompute/cache MBS maps; persist world in appdata |
| [ppo_sinr_eval.m](../matlab/ppo_sinr_eval.m) | one SINR evaluation against a persisted world; multi mode does per-FBS antenna selection by band flag + base-MBS slot expansion |

Both keep the legacy x↔y row-swap convention, so the on-disk map cache is
keyed by geometry and reused across runs.

## 10. The pyqd port

[`ppo/pyqd_bridge.py`](../ppo/pyqd_bridge.py) reimplements that whole chain in
numpy + `pyqd-channel`. It is a *port*, not a reimplementation: several MATLAB
conventions look like bugs, are load-bearing (every archived run and cached map
was produced under them), and are therefore reproduced exactly, each with a
comment naming the MATLAB line it comes from. The three that matter most:

- **The x↔y row swap is reproduced**, so the MBS map is computed at
  `tx = [true_y, true_x, height]` (`ppo_world_setup.m:59-61`). The
  `mbs_x`/`mbs_y` the backend exposes to Python are the *un*swapped, true
  coordinates, matching `ppo_world_setup.m:88-89`.
- **The transpose asymmetry is reproduced.** An FBS map is transposed to
  `(n_x, n_y)` by `SINREvaluation.m:188`; a cached MBS map is left at
  `(n_y, n_x)` by `precompute_mbs_power_maps.m:87`. Both are read by the same
  `sample_nearest` (`SINREvaluation.m:306-321`), which always indexes
  `P(x_index, y_index)` and clamps each index against the corresponding axis
  of whatever it was handed. For the MBS map that first axis is only
  `height + 1` long, so **every user with `round(x) > height` is evaluated at
  `x == height`** — 226 of the 1000 default-world users. That is reference
  behaviour, not a rounding detail, and `tests/test_pyqd_backend.py` pins it.
- **`total_transmitted_pwr = sum(tx_power)`** with the `power_status` mask
  commented out (`SINREvaluation.m:120-121`): inactive FBSs still count
  towards reported power. Note `AnalyticSinrBackend` does *not* do this (nor
  does it return `NaN` for `avg_rate` when nothing connects) — it is not a
  safe template for a fidelity-critical backend.

Also reproduced: the `single()` cast on MBS maps only, the FBS path's
hard-coded scenario / mode / 1.5 m receiver height, the interleaved
`[cov(site1), cap(site1), cov(site2), …]` slot order, and the strict `>` in the
association loop that resolves ties to the earlier column.
