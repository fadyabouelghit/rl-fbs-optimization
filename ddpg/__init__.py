"""DDPG prototype for the FBS placement problem, kept apart from ``ppo/``.

The ``ppo`` package is not modified by anything here, and neither is ``td3/``.
DDPG reuses ppo only through imports: the same environment (``ppo.env``),
physics backends (``ppo.matlab_bridge`` / ``ppo.pyqd_bridge``), scenario
presets (``ppo.config``), logging callbacks, rollouts and plots. Everything
algorithm-specific lives in this package:

    config.py        DDPGParams + DDPGExperiment (env half = ppo.config.EnvConfig)
    paths.py         ddpg_runs/ and its ledger (never ppo_runs/ or td3_runs/)
    run_logging.py   DDPGRunLogger: ppo's run layout, DDPG's ledger columns
    train.py         DDPG training loop
    evaluate.py      rollouts + artifacts + plots for a saved DDPG run
    plotting.py      training gallery with DDPG's actor/critic losses
    assess.py        common-start-state comparison against TD3 and PPO runs
    cli.py           ``python -m ddpg train | eval | plot | assess``

DDPG is TD3 with its three tricks removed (one critic, no delayed policy
updates, no target policy smoothing), so it shares TD3's weakness on this
env: a deterministic actor whose tanh saturates on the binary gene dims stops
exploring them entirely. ``binary_noise_std`` sizes exploration noise on those
dims independently for that reason, and ``noise_type='ou'`` adds the DDPG
paper's temporally correlated noise.

Quick start:

    from ddpg import DDPGExperiment, train, evaluate_run

    exp = DDPGExperiment.from_code("1-1-1", total_timesteps=5000, seed=0)
    run_dir = train(exp, backend="pyqd-fast")
    report = evaluate_run(run_dir, episodes=3, backend="pyqd-fast")
"""
from .config import (
    DDPGExperiment,
    DDPGParams,
    binary_action_indices,
    ddpg_experiment_from_json_dict,
    load_ddpg_experiment,
)
from .evaluate import evaluate_run
from .train import train

__all__ = [
    "DDPGExperiment",
    "DDPGParams",
    "binary_action_indices",
    "ddpg_experiment_from_json_dict",
    "evaluate_run",
    "load_ddpg_experiment",
    "train",
]
