"""Train Hybrid B (ppo_onoff).

    train(exp, backend, run_dir, onoff=OnOffParams(...))

``exp`` is a PPO ``ExperimentConfig`` (e.g. from
``tuning.spaces.build_experiment('ppo_onoff', ...)``). The setup repeats
``ppo.train.train`` for ``n_envs == 1``: Monitor, DummyVecEnv, eval env on the
shared backend, VecNormalize(norm_obs=False), EnvInfoLoggingCallback,
CsvEvalCallback. The env is the plain env (no wrapper): the policy takes the
16-dim observation and returns the full 10-dim action.

model.zip is a complete greedy policy: ``PPO.load`` restores OnOffPPOPolicy
(actor, value net, both critics and their targets), so ``ppo.evaluate`` and
``tuning.objective`` score it unchanged.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ddqn_td3.config import Layout
from ppo.callbacks import CheckpointEveryCallback, CsvEvalCallback, EnvInfoLoggingCallback
from ppo.env import FlyingBaseStationEnv, state_labels
from ppo.matlab_bridge import DEFAULT_BACKEND

from .algo import PPOOnOff, check_version
from .config import ALGO, OnOffParams
from .policy import OnOffPPOPolicy
from .run_logging import PPOOnOffRunLogger


def build_model(exp, vec_env, onoff: OnOffParams) -> PPOOnOff:
    """``ppo.train._build_model``'s PPO arguments, with OnOffPPOPolicy and the critic settings."""
    p = exp.ppo
    layout = Layout(exp.env)
    return PPOOnOff(
        OnOffPPOPolicy, vec_env, onoff=onoff,
        learning_rate=p.learning_rate, n_steps=p.n_steps, batch_size=p.batch_size,
        n_epochs=p.n_epochs, gamma=p.gamma, gae_lambda=p.gae_lambda, clip_range=p.clip_range,
        ent_coef=p.ent_coef, vf_coef=p.vf_coef, max_grad_norm=p.max_grad_norm,
        policy_kwargs={"onoff_idx": layout.onoff_idx, "cont_idx": layout.cont_idx,
                       "critic_net_arch": list(onoff.critic_net_arch)},
        seed=p.seed, verbose=p.verbose)


def train(exp, backend: str = DEFAULT_BACKEND, run_dir: Optional[Path] = None, *,
          onoff: OnOffParams = OnOffParams(), extra_callbacks: Optional[list] = None,
          config_extra: Optional[dict] = None) -> Path:
    check_version()
    p = exp.ppo
    if p.algo != "ppo":
        raise ValueError(f"ppo_onoff needs a PPO experiment, got algo={p.algo!r}")
    if p.n_envs != 1:
        raise ValueError("ppo_onoff runs with n_envs == 1 (the fair protocol)")
    layout = Layout(exp.env)
    logger = PPOOnOffRunLogger(exp, run_dir=run_dir, backend_kind=backend,
                               extra_cols={"tau": onoff.tau,
                                           "exploration_fraction": onoff.exploration_fraction})
    log = logger.log
    log.info("run dir: %s", logger.run_dir)
    log.info("config: code=%s algo=%s backend=%s seed=%s tau=%g exploration_fraction=%g",
             exp.code, ALGO, backend, p.seed, onoff.tau, onoff.exploration_fraction)
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
        if p.eval_every > 0:
            eval_env = FlyingBaseStationEnv(exp.env, backend=train_env.backend)
        if p.normalize_reward:
            vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=p.gamma)
            log.info("running reward normalization enabled (VecNormalize)")
        log.info("env(s) ready in %.1fs", time.time() - t0)
        logger.write_config(extra={
            "resumed_from": None, "env_setup_seconds": round(time.time() - t0, 2),
            "algo": ALGO, **(config_extra or {}),
            "ppo_onoff": {"params": onoff.to_dict(), "n_fbs": layout.n_fbs,
                          "onoff_idx": layout.onoff_idx, "cont_idx": layout.cont_idx,
                          "selection": "argmax_b min(Q1,Q2)(s,[clip(mu(s,b)),b]), eps-greedy"}})

        model = build_model(exp, vec_env, onoff)
        formats = ["csv"] + (["stdout"] if p.verbose else [])
        model.set_logger(configure_logger(str(logger.run_dir), formats))

        info_cb = EnvInfoLoggingCallback(logger, state_labels(exp.env))
        callbacks = [info_cb]
        if p.checkpoint_every > 0:
            callbacks.append(CheckpointEveryCallback(logger, p.checkpoint_every))
        if eval_env is not None:
            callbacks.append(CsvEvalCallback(logger, eval_env, p.eval_every, p.eval_episodes))
        callbacks.extend(extra_callbacks or [])

        t0 = time.time()
        model.learn(total_timesteps=p.total_timesteps, callback=callbacks)
        elapsed = time.time() - t0
        model.save(str(logger.run_dir / "model"))
        if isinstance(vec_env, VecNormalize):
            vec_env.save(str(logger.run_dir / "vecnormalize.pkl"))
        log.info("training done in %.1fs (%.1f steps/s); critic updates %d",
                 elapsed, p.total_timesteps / max(elapsed, 1e-9), model.critic_updates)
        logger.update_config(critic_updates=int(model.critic_updates),
                             critic_last_loss=float(model.critic_last_loss))

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


def onoff_params(params: dict) -> OnOffParams:
    """Tuner params -> OnOffParams (tau and exploration_fraction are searched)."""
    return OnOffParams(tau=float(params["tau"]),
                       exploration_fraction=float(params["exploration_fraction"]))


def train_algo(algo: str, exp, backend: str = DEFAULT_BACKEND, *, params: dict, **kwargs) -> Path:
    """Entry point for ``tuning.worker`` / ``letterexp.runner`` ('ppo_onoff')."""
    if algo != ALGO:
        raise ValueError(f"ppo_onoff.train_algo trains {ALGO!r}, not {algo!r}")
    return train(exp, backend=backend, onoff=onoff_params(params), **kwargs)
