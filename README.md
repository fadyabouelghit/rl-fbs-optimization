# rl-fbs-optimization

Reinforcement-learning placement and power control for flying base stations
(FBS) in a multi-band heterogeneous network.

A PPO/SAC agent moves one or more FBSs over a service area and toggles their
power and band, maximising a connectivity/power objective evaluated by real
QuaDRiGa physics through MATLAB — or by a fast MATLAB-free analytic backend.

This is the RL track. The genetic-algorithm track it shares its physics and
objective with lives in a separate `genetic-algorithm-optimization`
repository; the two agree on the same SINR evaluation so their results are
directly comparable.

---

## Layout

```
ppo/                 the pipeline (config, env, backends, train, eval, logging, plots)
matlab/              MATLAB wrappers + their full dependency closure
tests/               MATLAB-free test suite (analytic/stub backends)
notebooks/           interactive workflows (start with ppo_workbench.ipynb)
scripts/             batch sweep drivers
docs/PPO_PIPELINE.md architecture and artifact reference
secrets.env.example  template for machine-local paths (copy to secrets.env)
```

Top-level scripts: `train_ppo.py` / `ppo_experiment.py` (compatibility shims
for pre-package notebooks), `watch_run.py` (live training dashboard),
`assess_models.py` (common-start-state policy comparison), `plot_trajectories.py`,
`rerun_historical_rl.py`, `backfill_ledger.py`, and the `pilot_*.py` studies.

## Two backends

| backend | speed | needs | use for |
|---|---|---|---|
| `analytic` | ~10k steps/s | nothing | tests, smoke runs, config/plot iteration |
| `matlab` | ~0.6 steps/s | MATLAB engine + QuaDRiGa | real physics, reported results |

Both implement the same interface and band semantics, so a config runs
unchanged on either.

## Setup

```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt
```

`matlabengine` in `requirements.txt` must match your installed MATLAB release
(R2024b → `24.2.*`). Skip it if you only need the analytic backend.

### Machine-local paths

**No absolute path is hardcoded in this repository.** They are read from a
git-ignored `secrets.env` at the repo root:

```bash
cp secrets.env.example secrets.env
```

Then set `PPO_QUADRIGA_PATH` to your QuaDRiGa `quadriga_src` folder. Every key
can also be given as an environment variable, which takes precedence:

```bash
PPO_QUADRIGA_PATH=/opt/quadriga_src python -m ppo train --code 1-1-1
```

QuaDRiGa is the only required key, and only for the MATLAB backend — the test
suite, the analytic backend, and every plotting/analysis script run without
it. See `secrets.env.example` for the optional overrides (run directory,
cache, ledger, MATLAB path). Resolution lives in [`ppo/paths.py`](ppo/paths.py).

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
