"""PPO pipeline for Flying Base Station placement.

Quick start (see PPO_PIPELINE.md for the full guide):

    from ppo import ExperimentConfig, train, evaluate_run

    exp = ExperimentConfig.from_code("1-1-1", total_timesteps=5000, seed=0)
    run_dir = train(exp)                      # real MATLAB physics
    report = evaluate_run(run_dir, episodes=3)

    # fast, MATLAB-free iteration on pipeline / env mechanics:
    run_dir = train(exp, backend="analytic")

CLI: ``python -m ppo --help``  (train / eval / plot / list / info / smoke).
"""
from .config import (
    BandConfig,
    CONFIGS,
    EnvConfig,
    ExperimentConfig,
    PPOParams,
    RewardConfig,
    RewardWeights,
    SCENARIOS,
    WorldConfig,
    experiment_from_json_dict,
    load_experiment_config,
)
from .env import FlyingBaseStationEnv, state_labels
from .evaluate import EvalReport, evaluate_run, flatten_states, rollout_episode
from .matlab_bridge import (
    AnalyticSinrBackend,
    MatlabSession,
    MatlabSinrBackend,
    SinrResult,
    close_shared_session,
    get_shared_session,
    make_backend,
)
from .runs import list_evals, list_runs, load_run, resolve_run
from .train import train

__all__ = [
    "AnalyticSinrBackend",
    "BandConfig",
    "CONFIGS",
    "EnvConfig",
    "EvalReport",
    "ExperimentConfig",
    "FlyingBaseStationEnv",
    "MatlabSession",
    "MatlabSinrBackend",
    "PPOParams",
    "RewardConfig",
    "RewardWeights",
    "SCENARIOS",
    "SinrResult",
    "WorldConfig",
    "close_shared_session",
    "evaluate_run",
    "experiment_from_json_dict",
    "flatten_states",
    "get_shared_session",
    "list_evals",
    "list_runs",
    "load_experiment_config",
    "load_run",
    "make_backend",
    "resolve_run",
    "rollout_episode",
    "state_labels",
    "train",
]
