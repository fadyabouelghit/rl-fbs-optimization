"""TD3 prototype for the FBS placement problem, kept apart from ``ppo/``.

The ``ppo`` package is not modified by anything here. TD3 reuses it only
through imports: the same environment (``ppo.env``), physics backends
(``ppo.matlab_bridge`` / ``ppo.pyqd_bridge``), scenario presets
(``ppo.config``), logging callbacks, rollouts and plots. Everything
algorithm-specific lives in this package:

    config.py        TD3Params + TD3Experiment (env half = ppo.config.EnvConfig)
    paths.py         td3_runs/ and its ledger (never ppo_runs/ or training_log.csv)
    run_logging.py   TD3RunLogger: ppo's run layout, TD3's ledger columns
    train.py         TD3 training loop
    evaluate.py      rollouts + artifacts + plots for a saved TD3 run
    plotting.py      training gallery with TD3's actor/critic losses
    assess.py        common-start-state comparison of TD3 runs against PPO runs
    cli.py           ``python -m td3 train | eval | plot | assess``

Quick start:

    from td3 import TD3Experiment, train, evaluate_run

    exp = TD3Experiment.from_code("1-1-1", total_timesteps=5000, seed=0)
    run_dir = train(exp, backend="pyqd-fast")
    report = evaluate_run(run_dir, episodes=3, backend="pyqd-fast")
"""
from .config import (
    TD3Experiment,
    TD3Params,
    binary_action_indices,
    load_td3_experiment,
    td3_experiment_from_json_dict,
)
from .evaluate import evaluate_run
from .train import train

__all__ = [
    "TD3Experiment",
    "TD3Params",
    "binary_action_indices",
    "evaluate_run",
    "load_td3_experiment",
    "td3_experiment_from_json_dict",
    "train",
]
