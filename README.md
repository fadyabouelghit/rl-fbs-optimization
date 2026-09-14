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

Run outputs, caches, and `secrets.env` are git-ignored.
