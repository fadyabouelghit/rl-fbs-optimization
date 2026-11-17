"""Train switch + pilot: ``td3.train.train`` / ``ddpg.train.train`` with the switch wrapped in.

    train(exp, backend, run_dir, pilot="td3", switch=SwitchParams(hold_k=4))

``exp`` is the pilot's own experiment (``TD3Experiment`` / ``DDPGExperiment``,
e.g. from ``tuning.spaces.build_experiment('ddqn_td3', ...)``). Its
``binary_noise_std`` must be None: the pilot has no binary dims, the switch's
epsilon is the on/off exploration.

Artifacts, on top of the pilot's usual run directory:
    switch.pt / best_switch.pt   final / best-eval switch (pairs with model.zip / best_model.zip)
    switch_summary.json          updates, holds stored, decisions, final eps
    experiment_config.json       extra["ddqn_switch"] = switch params + layout
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ppo.matlab_bridge import DEFAULT_BACKEND

from .callbacks import SwitchCallback
from .config import Layout, SwitchParams
from .paths import routed
from .switch import SwitchAgent
from .wrapper import SwitchEvalWrapper, SwitchVecEnv


def _params(exp, pilot: str):
    return exp.td3 if pilot == "td3" else exp.ddpg


def train(exp, backend: str = DEFAULT_BACKEND, run_dir: Optional[Path] = None, *,
          pilot: str = "td3", switch: SwitchParams = SwitchParams(),
          extra_callbacks: Optional[list] = None, config_extra: Optional[dict] = None) -> Path:
    if pilot == "td3":
        from td3.train import train as pilot_train
    elif pilot == "ddpg":
        from ddpg.train import train as pilot_train
    else:
        raise ValueError(f"unknown pilot {pilot!r}")
    P = _params(exp, pilot)
    if P.n_envs != 1:
        raise ValueError("ddqn_td3 needs n_envs == 1")
    if P.binary_noise_std is not None:
        raise ValueError("the pilot has no binary dims: binary_noise_std must be None")
    layout = Layout(exp.env)
    holder: dict = {}

    def vec_wrap(venv):
        agent = SwitchAgent(
            obs_dim=int(venv.observation_space.shape[0]), n_joint=layout.n_joint, params=switch,
            gamma=P.gamma, learning_rate=P.learning_rate, batch_size=P.batch_size,
            net_arch=P.net_arch or (64, 64), total_timesteps=P.total_timesteps,
            learning_starts=P.learning_starts, seed=P.seed or 0)
        holder["agent"] = agent
        return SwitchVecEnv(venv, agent, layout)

    def eval_wrap(env):
        return SwitchEvalWrapper(env, holder["agent"], layout)

    extra = {**(config_extra or {}),
             "ddqn_switch": {"pilot": pilot, "params": switch.to_dict(),
                             "n_fbs": layout.n_fbs, "n_joint": layout.n_joint,
                             "onoff_idx": layout.onoff_idx, "cont_idx": layout.cont_idx,
                             "net_arch": list(P.net_arch or (64, 64)),
                             "reward": "VecNormalize-normalized at sample time"}}
    callbacks = [SwitchCallback()] + list(extra_callbacks or [])
    with routed(pilot):
        return Path(pilot_train(exp, backend=backend, run_dir=run_dir,
                                extra_callbacks=callbacks, config_extra=extra,
                                vec_env_wrapper=vec_wrap, eval_env_wrapper=eval_wrap))


def switch_params(params: dict, hold_k: int = 1) -> SwitchParams:
    """Tuner params -> SwitchParams: only exploration_fraction is searched; hold_k is the arm."""
    return SwitchParams(hold_k=int(hold_k),
                        exploration_fraction=float(params["exploration_fraction"]))


def train_algo(algo: str, exp, backend: str = DEFAULT_BACKEND, *, params: dict,
               hold_k: int = 1, **kwargs) -> Path:
    """Entry point for ``tuning.worker`` / ``letterexp.runner`` ('ddqn_td3' | 'ddqn_ddpg')."""
    from .config import PILOT_OF
    return train(exp, backend=backend, pilot=PILOT_OF[algo],
                 switch=switch_params(params, hold_k), **kwargs)
