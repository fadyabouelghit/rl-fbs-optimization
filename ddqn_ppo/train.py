"""Train Hybrid A: the ddqn_td3 switch + an SB3 PPO pilot.

    train(exp, backend, run_dir, switch=SwitchParams(...))

``exp`` is a PPO ``ExperimentConfig`` (e.g. from
``tuning.spaces.build_experiment('ddqn_ppo', ...)``). The setup repeats
``ppo.train.train`` for ``n_envs == 1`` (Monitor, DummyVecEnv, eval env on the
shared backend, VecNormalize with norm_obs=False, EnvInfoLoggingCallback,
CsvEvalCallback, model built by ``ppo.train._build_model``) and inserts the
switch below VecNormalize::

    FlyingBaseStationEnv -> Monitor -> DummyVecEnv -> SwitchVecEnv -> VecNormalize -> PPO

PPO's rollout buffer stores the normalized reward VecNormalize returns; the
switch stores raw rewards and normalizes them at sample time with the same
running statistics, so both learn from the same reward (as in ddqn_td3).
PPO bootstraps truncated episodes from ``info["terminal_observation"]``, which
``SwitchVecEnv`` augments with the b that was executed.

Artifacts: ppo's usual run directory plus switch.pt / best_switch.pt /
switch_summary.json, and ``provenance.ddqn_switch`` in experiment_config.json
(pilot 'ppo'), which ``ddqn_td3.policy.SwitchPilotPolicy`` reads to load it.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ddqn_td3.callbacks import SwitchCallback
from ddqn_td3.config import Layout, SwitchParams
from ddqn_td3.switch import SwitchAgent
from ddqn_td3.wrapper import SwitchEvalWrapper, SwitchVecEnv
from ppo.callbacks import CheckpointEveryCallback, CsvEvalCallback, EnvInfoLoggingCallback
from ppo.env import FlyingBaseStationEnv, state_labels
from ppo.matlab_bridge import DEFAULT_BACKEND
from ppo.train import _build_model

from .config import ALGO, SWITCH_LEARNING_STARTS, SWITCH_NET_ARCH
from .run_logging import DDQNPPORunLogger


def _check(exp, switch: SwitchParams) -> None:
    p = exp.ppo
    if p.algo != "ppo":
        raise ValueError(f"ddqn_ppo needs a PPO experiment, got algo={p.algo!r}")
    if p.n_envs != 1:
        raise ValueError("ddqn_ppo needs n_envs == 1: one switch, one stream of holds")
    if switch.learning_starts is None:
        raise ValueError("pass SwitchParams.learning_starts explicitly (see ddqn_ppo.config)")


def train(exp, backend: str = DEFAULT_BACKEND, run_dir: Optional[Path] = None, *,
          switch: SwitchParams = SwitchParams(learning_starts=SWITCH_LEARNING_STARTS),
          extra_callbacks: Optional[list] = None,
          config_extra: Optional[dict] = None) -> Path:
    _check(exp, switch)
    p = exp.ppo
    layout = Layout(exp.env)
    meta = {"pilot": "ppo", "params": switch.to_dict(), "n_fbs": layout.n_fbs,
            "n_joint": layout.n_joint, "onoff_idx": layout.onoff_idx,
            "cont_idx": layout.cont_idx, "net_arch": list(SWITCH_NET_ARCH),
            "reward": "VecNormalize-normalized at sample time"}
    logger = DDQNPPORunLogger(exp, run_dir=run_dir, backend_kind=backend,
                              switch={"hold_k": switch.hold_k,
                                      "exploration_fraction": switch.exploration_fraction,
                                      "learning_starts": switch.learning_starts})
    log = logger.log
    log.info("run dir: %s", logger.run_dir)
    log.info("config: code=%s algo=%s (switch + PPO pilot) backend=%s seed=%s hold_k=%d",
             exp.code, ALGO, backend, p.seed, switch.hold_k)

    if p.seed is not None:
        set_random_seed(p.seed)

    eval_env = None
    vec_env = None
    status = "failed"
    final_metrics: dict = {}
    try:
        t0 = time.time()
        train_env = FlyingBaseStationEnv(exp.env, backend_kind=backend)
        monitored = Monitor(train_env, filename=str(logger.run_dir / "monitor.csv"),
                            allow_early_resets=True)
        vec_env = DummyVecEnv([lambda: monitored])
        # The switch's network is initialised inside torch.random.fork_rng and
        # it samples from its own numpy Generator, so building it here leaves
        # the seeded streams PPO draws from untouched.
        agent = SwitchAgent(
            obs_dim=int(vec_env.observation_space.shape[0]), n_joint=layout.n_joint,
            params=switch, gamma=p.gamma, learning_rate=p.learning_rate,
            batch_size=p.batch_size, net_arch=SWITCH_NET_ARCH,
            total_timesteps=p.total_timesteps, learning_starts=int(switch.learning_starts),
            seed=p.seed or 0)
        vec_env = SwitchVecEnv(vec_env, agent, layout)
        if p.eval_every > 0:
            eval_env = SwitchEvalWrapper(
                FlyingBaseStationEnv(exp.env, backend=train_env.backend), agent, layout)
        if p.normalize_reward:
            vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=p.gamma)
            log.info("running reward normalization enabled (VecNormalize)")
        log.info("env(s) ready in %.1fs", time.time() - t0)

        logger.write_config(extra={
            "resumed_from": None, "env_setup_seconds": round(time.time() - t0, 2),
            "algo": ALGO, **(config_extra or {}), "ddqn_switch": meta})

        model = _build_model(exp, vec_env)
        formats = ["csv"] + (["stdout"] if p.verbose else [])
        model.set_logger(configure_logger(str(logger.run_dir), formats))

        info_cb = EnvInfoLoggingCallback(logger, state_labels(exp.env))
        callbacks = [info_cb]
        if p.checkpoint_every > 0:
            callbacks.append(CheckpointEveryCallback(logger, p.checkpoint_every))
        if eval_env is not None:
            callbacks.append(CsvEvalCallback(logger, eval_env, p.eval_every, p.eval_episodes))
        # after CsvEvalCallback: best_switch.pt pairs with the best_model.zip
        # the same evaluation just wrote (see ddqn_td3.callbacks)
        callbacks.append(SwitchCallback())
        callbacks.extend(extra_callbacks or [])

        t0 = time.time()
        model.learn(total_timesteps=p.total_timesteps, callback=callbacks)
        elapsed = time.time() - t0
        model.save(str(logger.run_dir / "model"))
        if isinstance(vec_env, VecNormalize):
            vec_env.save(str(logger.run_dir / "vecnormalize.pkl"))
        log.info("training done in %.1fs (%.1f steps/s)",
                 elapsed, p.total_timesteps / max(elapsed, 1e-9))

        status = "completed"
        final_metrics = {k: v for k, v in info_cb.last_episode.items()
                         if k.startswith("final_") or k in ("reward_sum", "length")}
        final_metrics["elapsed_sec"] = round(elapsed, 2)
        return logger.run_dir
    finally:
        if eval_env is not None:
            eval_env.close()
        if vec_env is not None:
            vec_env.close()
        logger.finalize(status=status, metrics=final_metrics)


def switch_params(params: dict, hold_k: int = 1) -> SwitchParams:
    """Tuner params -> SwitchParams: exploration_fraction is searched, hold_k is the arm."""
    return SwitchParams(hold_k=int(hold_k), learning_starts=SWITCH_LEARNING_STARTS,
                        exploration_fraction=float(params["exploration_fraction"]))


def train_algo(algo: str, exp, backend: str = DEFAULT_BACKEND, *, params: dict,
               hold_k: int = 1, **kwargs) -> Path:
    """Entry point for ``tuning.worker`` / ``letterexp.runner`` ('ddqn_ppo')."""
    if algo != ALGO:
        raise ValueError(f"ddqn_ppo.train_algo trains {ALGO!r}, not {algo!r}")
    return train(exp, backend=backend, switch=switch_params(params, hold_k), **kwargs)
