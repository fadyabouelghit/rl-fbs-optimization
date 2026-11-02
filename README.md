# rl-fbs-optimization

Reinforcement-learning placement and power control for flying base stations
(FBS) in a multi-band heterogeneous network.

A PPO/SAC agent moves one or more FBSs over a service area and toggles their
power and band, maximising a connectivity/power objective evaluated by real
QuaDRiGa physics. The default backend computes that physics **in pure Python**
(`pyqd-channel`), so the whole pipeline runs with no MATLAB installed; the
original MATLAB/QuaDRiGa stack stays selectable as the reference oracle.

---

## Layout

```
ppo/                 the pipeline (config, env, backends, train, eval, logging, plots)
td3/                 TD3 prototype, separate from ppo/ (reuses its env, backends, plots)
ddpg/                DDPG prototype, same arrangement (TD3 without its three tricks)
tuning/              shared hyperparameter tuner: equal budget for all three
matlab/              MATLAB wrappers + their full dependency closure
tests/               MATLAB-free test suite (pyqd/analytic/stub backends)
notebooks/           interactive workflows (start with ppo_workbench.ipynb)
scripts/             batch sweep drivers
docs/PPO_PIPELINE.md architecture and artifact reference
secrets.env.example  template for machine-local paths (copy to secrets.env)
```

Top-level scripts: `train_ppo.py` / `ppo_experiment.py` (compatibility shims
for pre-package notebooks), `watch_run.py` (live training dashboard),
`assess_models.py` (common-start-state policy comparison),
`plot_trajectories.py` (rollout trajectory overlays), `rerun_historical_rl.py`,
`backfill_ledger.py`, and the `pilot_*.py` studies.

## TD3 prototype

`td3/` trains Stable-Baselines3 TD3 on exactly the same environment without
touching `ppo/`: it imports the env, backends, callbacks, rollouts and plots,
and keeps its own config (`TD3Params`), runs (`td3_runs/`, override with
`TD3_RUNS_DIR`) and ledger (`td3_runs/ledger.csv`). The env and reward flags
match `python -m ppo train`:

`--binary-noise-std` sets the exploration std on the binary gene dims alone.
TD3's actor is deterministic and saturates its tanh on the FBS on/off gene
exactly as DDPG's does — the twin critics and delayed updates do nothing about
it — so under sticky binaries a flip needs the noisy action to cross ±0.5, and
from a saturated −1.0 the draw must exceed 1.5: never at std 0.1, ~3e-7 per
step at 0.3, ~6.7% at 1.0. The continuous dims keep their own small std.

```bash
python -m td3 train --code 1-1-1 --timesteps 25000 --seed 0 --backend pyqd-fast
python -m td3 train --code 1-1-1 --action-noise-std 0.1 --binary-noise-std 0.5
python -m td3 eval --run latest --episodes 5 --backend pyqd-fast
python -m td3 plot --run latest
python -m td3 assess --runs latest --ppo-runs <ppo run> --episodes 20   # common start states
```

## DDPG prototype

`ddpg/` is the same arrangement one algorithm down: SB3's `DDPG` is `TD3` with
its three tricks removed (one critic, `policy_delay=1`, no target policy
smoothing), so `DDPGParams` has no fields for them. Runs land in `ddpg_runs/`
(override with `DDPG_RUNS_DIR`) with their own ledger.

`--binary-noise-std` works here as it does in `td3/` (see above) — the
deterministic-actor blind spot is shared by both, and this is where it was
first diagnosed: it was the single largest effect in the 56-run DDPG tuning
campaign (`bn1.0` → `bn0.5` moved J from 0.4275 to 0.4550).

One knob is genuinely DDPG's own:

- `--noise-type ou` uses the DDPG paper's Ornstein-Uhlenbeck noise, which is
  temporally correlated rather than per-step jitter. Tune `--ou-dt` against
  `--max-episode-steps`: the noise decorrelates over ~`1/(theta*dt)` steps, so
  SB3's default 0.01 is a ~670-step correlation length that a 40-step episode
  never sees the end of.

```bash
python -m ddpg train --code 1-1-1 --timesteps 25000 --seed 0 --backend pyqd-fast
python -m ddpg train --code 1-1-1 --noise-type ou --ou-dt 1.0 --binary-noise-std 1.0
python -m ddpg eval --run latest --episodes 5 --backend pyqd-fast
python -m ddpg plot --run latest
python -m ddpg assess --runs latest --td3-runs <td3 run> --ppo-runs <ppo run> --episodes 20
```

## Shared tuner

The three algorithms have had wildly unequal tuning effort: 56 hand-driven
runs for DDPG, a learning-rate sweep for TD3, none at all for PPO. Comparing
them as they stand compares *search effort* as much as it compares
algorithms. `tuning/` fixes that by giving each one the same budget, the same
objective and the same seed protocol, without touching `ppo/`, `td3/` or
`ddpg/` — it imports them.

```bash
python -m tuning spaces                                      # what is searched, and what cannot be
python -m tuning search  --campaign fair1 --algo ddpg --workers 4
python -m tuning confirm --campaign fair1 --algo ddpg --workers 4
python -m tuning report  --campaign fair1
python -m tuning rescore --campaign fair1 --algo ddpg --run <run dir>   # held-out states
```

The contract, all of it in `tuning/protocol.py`:

- **Same budget** — 60 trials, 100k steps per trial, `pyqd-fast`, identical env
  and reward flags (the potrec10 settings, so results compare to runs already
  on disk).
- **Same objective** — mean final J of the run's `best_model` over 20 common
  start states drawn from seed 987654, via `assess_models.py`'s own start
  states and reference points. Model *selection* uses a third seed (12345),
  so selection and scoring never share a draw.
- **Same seed protocol** — stage 1 searches at seed 0; stage 2 re-runs the top
  5 across seeds 0–4 and ranks on **mean − sd**, putting reliability inside
  the objective rather than in a footnote.
- **Same shared hyperparameters** — learning rate 3e-4, discount 0.95, batch
  256 and architecture (64,64) are *pinned* for all three rather than
  searched, so a trial only ever varies what is specific to its algorithm.
- **Same dimensionality** — each algorithm searches exactly **five** knobs, so
  equal trials buy equal coverage.

| | the five searched knobs |
|---|---|
| **PPO** | `ent_coef` · `n_steps` · `n_epochs` · `gae_lambda` · `clip_range` |
| **TD3** | `action_noise_std` · `binary_noise_std` · `policy_delay` · `target_policy_noise` · `tau` |
| **DDPG** | `action_noise_std` · `binary_noise_std` · `noise_type` · `ou_dt` · `tau` |

`python -m tuning spaces` prints this, the pinned values, and a **NOT
SEARCHABLE** list of knobs a package cannot express. That list is currently
empty — `binary_noise_std` was added to `td3/`, and `net_arch` stopped being
an asymmetry once all three were pinned to PPO's (64,64), which is the only
architecture PPO can express since `ppo/` passes no `policy_kwargs`.

Pinning the shared knobs has a cost worth stating: DDPG's own tuned point was
lr 1e-4 / γ0.9, and at the common values it gives up roughly 0.009 J. The
trade is that no algorithm carries a learning rate or discount chosen by how
much attention it happened to get — PPO's had never been searched at all —
and that 60 trials now cover 5 dimensions instead of 8.

Three further decisions:

- **Search is GP-based Bayesian optimisation** (`optuna.samplers.GPSampler`,
  Matérn-5/2 with ARD, logEI). `deterministic_objective=False`, so the GP fits
  a noise variance — which matters here, where seed noise is the same size as
  the effects being measured.
- **Spaces are flat, never conditional.** `GPSampler` does not support dynamic
  search spaces and silently falls back to `RandomSampler` for a parameter
  that only exists on some trials. So `ou_dt` is suggested on every DDPG trial
  and ignored unless `noise_type='ou'`; ARD learns to discount it.
- **No pruner.** Across the 54 DDPG runs that reached 100k, Spearman rank
  correlation between J at 25k and J at 100k is **+0.26** (+0.53 at 50k), and
  the best run of the campaign ranked **29th of 54** at 25k. Successive
  halving at any early rung throws it away, so every trial runs to full budget.

Each trial is a subprocess with `*_RUNS_DIR` pointed at the campaign, which
keeps 180 runs out of `ppo_runs/`, `td3_runs/`, `ddpg_runs/` and
`training_log.csv` — and has the useful side effect that a trial is an
*ordinary* run of its algorithm, so `python -m ddpg plot --run ...` and
`python -m ddpg assess` work on a campaign's output unchanged.

## Backends

| backend | speed | needs | use for |
|---|---|---|---|
| `pyqd` *(default)* | ~1.0 steps/s | nothing | real physics, reported results |
| `pyqd-fast` | ~160 steps/s | nothing | real physics at scale (bit-identical to `pyqd`) |
| `matlab` | ~1.0 steps/s | MATLAB engine + QuaDRiGa | cross-checking `pyqd` against the original |
| `analytic` | ~10k steps/s | nothing | tests, smoke runs, config/plot iteration |

(steps/s measured with 2 active FBSs on the default 2000×1500 world, i.e.
scenario code `*-1-*`.)

`pyqd` costs one full coverage grid per active FBS per step, so its price scales
with the world *area*, while `pyqd-fast` costs one propagation evaluation per
user and does not. On the 4000×3000 two-MBS world of scenario code `*-2-*` a
single map is ~2.4 s and allocates ~3.4 GB transiently — ~0.2 steps/s, and that
allocation is *per `n_envs` worker*. Prefer `--backend pyqd-fast` there; it is
bit-identical, so nothing is traded away.

`pyqd` is a deliberate port of `SINREvaluation.m` onto
[`pyqd-channel`](https://pypi.org/project/pyqd-channel/), a validated Python
port of QuaDRiGa's coverage-map subset. Its power maps agree with the archived
MATLAB maps to **3.3e-6 dB**, and it reproduces the connection counts, tier
split, transmitted power and average rate of completed MATLAB runs exactly
(`tests/test_pyqd_backend.py`).

`pyqd-fast` exploits the fact that user positions are snapped to integer grid
nodes and the map step is 1 m: it evaluates the propagation model at just those
nodes instead of over the full 3.0-million-point grid. It is **bit-identical**
to `pyqd` (max |Δ| = 0.0 dB), ~160× cheaper per FBS map, and opt-in only so
that default timings stay comparable with the MATLAB baseline.

`analytic` is an uncalibrated log-distance stand-in — same interface and band
semantics, numbers that mean nothing physically.

All of them implement the same interface, so a config runs unchanged on any
of them.

## Setup

```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt
```

That is everything the default `pyqd` backend needs. `matlabengine` is
commented out in `requirements.txt`: uncomment it (matching your installed
MATLAB release — R2024b → `24.2.*`) only if you want `--backend matlab`.

### Machine-local paths

**No absolute path is hardcoded in this repository.** They are read from a
git-ignored `secrets.env` at the repo root:

```bash
cp secrets.env.example secrets.env
```

**There is no required key.** Every setting has a repo-relative default, and
the default `pyqd` backend needs no external installation at all. Each key can
also be given as an environment variable, which takes precedence:

```bash
PPO_PYQD_CACHE_DIR=/scratch/maps python -m ppo train --code 1-1-1
```

`PPO_QUADRIGA_PATH` (your QuaDRiGa `quadriga_src` folder) is needed only by
`--backend matlab`. See `secrets.env.example` for the optional overrides (run
directory, the two map caches, ledger, MATLAB path). Resolution lives in
[`ppo/paths.py`](ppo/paths.py).

## Quick start

```bash
python -m pytest
```

```bash
python -m ppo smoke
```

`smoke` trains and evaluates end to end on the analytic backend in a few
seconds — the fastest check that the pipeline is intact.

```bash
python -m ppo train --code 1-1-1 --timesteps 25000 --max-episode-steps 40
```

Real physics on the `pyqd` backend, no MATLAB. Add `--backend pyqd-fast` for
the same numbers ~160× faster, or `--backend matlab` for the original stack.

```bash
python watch_run.py
```

Follows the newest run from a second terminal; it only reads the run's flat
files, so it is safe to start and stop at any time.

## Where results go

Each run writes a self-describing directory under `ppo_runs/` (model,
checkpoints, `experiment_config.json`, `steps.csv`, `episodes.csv`, plots,
`evals/`), and appends one row to the `training_log.csv` ledger. Full artifact
reference: [docs/PPO_PIPELINE.md](docs/PPO_PIPELINE.md).

TD3, DDPG and tuning campaigns keep their own trees: `td3_runs/`,
`ddpg_runs/` and `tuning_runs/<campaign>/`, each with its own ledger. A tuning
campaign nests the others inside itself (`tuning_runs/fair1/ddpg_runs/...`),
so a campaign is self-contained and deletable in one go.

Run outputs, caches, and `secrets.env` are git-ignored.
