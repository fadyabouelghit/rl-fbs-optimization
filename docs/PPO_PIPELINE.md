# PPO Pipeline — Architecture & Artifacts

Reference for the Python PPO track. Everything lives in the
[`ppo/`](../ppo/) package; `train_ppo.py` and `ppo_experiment.py` remain as
thin compatibility shims for old notebooks. The MATLAB physics wrappers and
their dependency closure live in [`matlab/`](../matlab/).

The GA counterpart of this pipeline (and its `CODE_MAP.md` /
`OUTPUT_PIPELINE.md` references) lives in the separate
`genetic-algorithm-optimization` repository.

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
              matlab_bridge.MatlabSinrBackend   matlab_bridge.AnalyticSinrBackend
              (real physics, engine-side world) (numpy stand-in, ~10k steps/s)
                            │
                 ppo_world_setup.m  ── builds antennas/sites/cache ONCE,
                 ppo_sinr_eval.m    ── per-step SINR, scalars-only marshalling
                            │
                 SINREvaluation.m + QuaDRiGa   (same physics as the GA)
```

Design points:

- **Two interchangeable backends.** The env never talks to MATLAB directly;
  it calls a `SinrBackend`. `backend="analytic"` runs the entire pipeline
  (training, logging, eval, plots) in milliseconds for tests and iteration;
  `backend="matlab"` is the real QuaDRiGa physics.
- **The MATLAB world stays in MATLAB.** `ppo_world_setup.m` persists
  antennas + the MBS map cache in engine appdata; `ppo_sinr_eval.m` reads it
  back per step. Only scalars cross the Python↔MATLAB boundary (the old
  train_ppo.py round-tripped the full map cache every step).
- **Dual band modes.** `band.mode="legacy"` reproduces the original
  single-band world (old agents stay loadable); `band.mode="multi"` mirrors
  the GA's dual-band world exactly (per-FBS band flag, base-MBS
  coverage/capacity slot expansion, same-band interference).

---

## 2. Quick start

```bash
# fast end-to-end pipeline check, no MATLAB needed (~10 s)
python -m ppo smoke

# real training (legacy world, like the old harness)
python -m ppo train --code 1-1-1 --timesteps 5000 --seed 0

# GA-parity multi-band training: agent controls FBS bands + MBS capacity
python -m ppo train --code 2-2-1 --band multi --fbs-band agent \
                    --mbs-capacity agent --reward ga_blend

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

`ExperimentConfig.from_code("X-Y-Z", ...)` — same axes as the GA harness:

| Axis | Meaning | Presets |
|---|---|---|
| X | FBS count | any int |
| Y | scenario | 1 = 1 MBS, 2000×1500 hex; 2 = 2 MBS, 4000×3000 custom (`ppo/config.py: SCENARIOS`) |
| Z | cost config | 1: γ=0.0; 2: γ=0.1 (`CONFIGS`) |

Nested blocks (all serialized into `experiment_config.json`, version 2):

- **`env.world`** — geometry, users, SINR threshold, QuaDRiGa scenario.
- **`env.band`** — `mode` (`legacy`/`multi`), `fbs_band`
  (`coverage`/`capacity`/`agent`), `mbs_capacity` (`off`/`on`/`agent`).
  Agent-controlled genes extend the state/action vectors exactly like the GA
  chromosome (6th gene per FBS + one trailing gene per MBS).
- **`env.reward`** — `mode`:
  - `legacy_blend` — the historical train_ppo.py formula (ε-padded);
  - `ga_blend` — exact mirror of `evaluatePopulation` targetIdx 1
    (controlled power incl. gated macro-capacity carriers);
  - `sum_rate` — targetIdx 2.
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
(status, hyperparameters, final metrics) — the RL counterpart of the GA
study ledgers.

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

1. **Unit/integration suite** — `./venv/bin/pytest` (~4 s, 40 tests, no
   MATLAB): env mechanics, reward math vs hand computations, band gene
   plumbing, logging, run discovery, **real PPO training reproducibility**
   on the analytic backend.
2. **Pipeline smoke** — `python -m ppo smoke` (~10 s): full multi-band
   train → checkpoint → in-training eval → evaluation → plot gallery, with
   artifact assertions.
3. **MATLAB bridge tests** — `PPO_MATLAB_TESTS=1 ./venv/bin/pytest
   tests/test_matlab_backend.py` (~20 s with a warm map cache): world build,
   legacy + multi-band evaluation, tier invariants, determinism.

Interactive speed-ups:

- `get_shared_session()` reuses one engine across repeated evaluations in a
  process; run `matlab.engine.shareEngine` in a MATLAB console and the
  bridge attaches to it instantly instead of cold-starting (~20 s saved per
  session).
- MBS power maps are cached on disk (`cache_mbs_maps/`, shared with the GA)
  — identical geometries never recompute.

## 7. Reproducibility & parallelism

- `ppo.seed` seeds python/numpy/torch and every env; a fixed
  `(seed, n_envs)` pair reproduces a run bit-for-bit (asserted in
  `tests/test_train_integration.py`). User positions stay pinned to the
  MATLAB-side fixed seed exactly as in the GA.
- `n_envs > 1` (opt-in) uses SubprocVecEnv; each worker starts its own
  MATLAB engine (~1–2 GB each) and is seeded `seed + worker_index`. Runs
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

## 9. MATLAB-side helpers (additive; GA files untouched)

| File | Role |
|---|---|
| [ppo_world_setup.m](../matlab/ppo_world_setup.m) | build antennas (band_frequencies-aware), sites (hex or explicit), pack + x↔y swap, precompute/cache MBS maps; persist world in appdata |
| [ppo_sinr_eval.m](../matlab/ppo_sinr_eval.m) | one SINR evaluation against a persisted world; multi mode reproduces `evaluatePopulation`'s antenna selection + `build_mbs_slots` expansion exactly |

Both mirror `optimize_base_station_ga.m` / `evaluatePopulation.m`
conventions line-for-line (including the x↔y row swap), so cached maps and
physics are shared with the GA for identical geometries.
