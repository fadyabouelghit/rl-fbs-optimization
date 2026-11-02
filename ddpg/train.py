"""DDPG training loop (Stable-Baselines3 DDPG).

Same environment, callbacks and run-directory layout as ``ppo.train`` and
``td3.train``; only the model differs. Key properties:

- **Algorithm**: SB3's ``DDPG`` is ``TD3`` with the three TD3 tricks removed
  (one critic, ``policy_delay=1``, no target policy smoothing), so a DDPG run
  and a TD3 run on the same env differ only in those tricks and the noise.
- **Exploration**: Gaussian or Ornstein-Uhlenbeck noise on the deterministic
  actor's actions during training, with an optional separate std on the
  binary gene dims. Evaluation is always noise-free.
- **Reproducibility**: ``seed`` fixes network init, exploration noise, replay
  sampling and env resets.
- **Parallelism**: ``n_envs > 1`` uses SubprocVecEnv, one backend per worker;
  SB3 gives each worker independent exploration noise.
- **In-training eval**: with ``eval_every > 0`` a dedicated eval env runs
  seeded deterministic episodes and tracks ``best_model.zip`` (shared
  backend when ``n_envs == 1``).
- **Resume**: ``resume_from`` restores the actor/critic and the
  reward-normalization statistics, but not the replay buffer, which starts
  empty in the new run.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from ppo.callbacks import CheckpointEveryCallback, CsvEvalCallback, EnvInfoLoggingCallback
from ppo.config import EnvConfig
from ppo.env import FlyingBaseStationEnv, state_labels
from ppo.matlab_bridge import DEFAULT_BACKEND

from .config import DDPGExperiment, binary_action_indices
from .run_logging import DDPGRunLogger
from .runs import resolve_run


def _make_worker_env(env_config: EnvConfig, backend_kind: str, monitor_path: str):
    """Top-level factory (picklable) for SubprocVecEnv workers."""
    def _init():
        env = FlyingBaseStationEnv(env_config, backend_kind=backend_kind)
        return Monitor(env, filename=monitor_path, allow_early_resets=True)
    return _init


def noise_sigma(exp: DDPGExperiment, n_actions: int) -> np.ndarray:
    """Per-dimension exploration std.

    Flat ``action_noise_std`` everywhere, unless ``binary_noise_std`` is set,
    in which case the binary gene dims get that instead. The two kinds of dim
    want different magnitudes: a continuous gene moves by
    ``noise * action_scale * range`` per step, while a binary gene only
    responds once the noisy action crosses a threshold.
    """
    p = exp.ddpg
    sigma = np.full(n_actions, float(p.action_noise_std), dtype=np.float64)
    if p.binary_noise_std is not None:
        idx = [i for i in binary_action_indices(exp.env) if i < n_actions]
        sigma[idx] = float(p.binary_noise_std)
    return sigma


def build_action_noise(exp: DDPGExperiment, n_actions: int):
    p = exp.ddpg
    sigma = noise_sigma(exp, n_actions)
    if not np.any(sigma > 0):
        return None
    mean = np.zeros(n_actions)
    if p.noise_type == "ou":
        return OrnsteinUhlenbeckActionNoise(
            mean=mean, sigma=sigma, theta=p.ou_theta, dt=p.ou_dt
        )
    return NormalActionNoise(mean=mean, sigma=sigma)


def build_model(exp: DDPGExperiment, vec_env) -> DDPG:
    p = exp.ddpg
    n_actions = int(np.prod(vec_env.action_space.shape))
    return DDPG(
        "MlpPolicy",
        vec_env,
        learning_rate=p.learning_rate,
        buffer_size=p.buffer_size,
        learning_starts=p.learning_starts,
        batch_size=p.batch_size,
        tau=p.tau,
        gamma=p.gamma,
        train_freq=p.train_freq,
        gradient_steps=p.gradient_steps,
        action_noise=build_action_noise(exp, n_actions),
        policy_kwargs={"net_arch": list(p.net_arch)} if p.net_arch else None,
        seed=p.seed,
        verbose=p.verbose,
    )


def train(
    exp: DDPGExperiment,
    backend: str = DEFAULT_BACKEND,
    run_dir: Optional[Path] = None,
    resume_from: Optional[str | Path] = None,
    *,
    extra_callbacks: Optional[list] = None,
    model_factory: Optional[Callable] = None,
    config_extra: Optional[dict] = None,
) -> Path:
    """Train DDPG under ``exp``; returns the run directory (under ddpg_runs/
    unless ``run_dir`` is given).

    backend     : any ppo backend name ('pyqd', 'pyqd-fast', 'matlab', 'analytic').
    resume_from : DDPG run name/path whose model.zip seeds the new run.

    Opt-in hooks (all default to None, which is the unchanged behaviour):
    extra_callbacks : SB3 callbacks appended after the built-in ones.
    model_factory   : ``f(exp, vec_env) -> model`` used instead of
                      ``build_model`` for a fresh (non-resumed) run.
    config_extra    : extra keys merged into experiment_config.json's extra.
    """
    logger = DDPGRunLogger(exp, run_dir=run_dir, backend_kind=backend)
    log = logger.log
    log.info("run dir: %s", logger.run_dir)
    log.info("config: code=%s algo=ddpg band=%s reward=%s shaping=%s backend=%s n_envs=%d seed=%s",
             exp.code, exp.env.band.mode, exp.env.reward.mode,
             exp.env.reward.shaping, backend, exp.ddpg.n_envs, exp.ddpg.seed)

    if exp.ddpg.seed is not None:
        set_random_seed(exp.ddpg.seed)

    n_envs = exp.ddpg.n_envs
    train_env = None
    eval_env = None
    vec_env = None
    status = "failed"
    final_metrics: dict = {}

    try:
        t0 = time.time()
        if n_envs == 1:
            train_env = FlyingBaseStationEnv(exp.env, backend_kind=backend)
            monitored = Monitor(
                train_env,
                filename=str(logger.run_dir / "monitor.csv"),
                allow_early_resets=True,
            )
            vec_env = DummyVecEnv([lambda: monitored])
            if exp.ddpg.eval_every > 0:
                eval_env = FlyingBaseStationEnv(exp.env, backend=train_env.backend)
        else:
            vec_env = SubprocVecEnv(
                [
                    _make_worker_env(exp.env, backend, str(logger.run_dir / f"monitor_{i}.csv"))
                    for i in range(n_envs)
                ],
                start_method="spawn",
            )
            if exp.ddpg.eval_every > 0:
                eval_env = FlyingBaseStationEnv(exp.env, backend_kind=backend)
        if exp.ddpg.normalize_reward:
            # Observations stay unnormalized (the env scales them itself), so
            # the saved policy runs against a plain env at evaluation time.
            vec_env = VecNormalize(
                vec_env, norm_obs=False, norm_reward=True, gamma=exp.ddpg.gamma
            )
            log.info("running reward normalization enabled (VecNormalize)")
        log.info("env(s) ready in %.1fs", time.time() - t0)

        sigma = noise_sigma(exp, int(np.prod(vec_env.action_space.shape)))
        log.info("exploration noise: %s sigma=%s%s", exp.ddpg.noise_type,
                 np.array2string(sigma, precision=3),
                 f" (theta={exp.ddpg.ou_theta}, dt={exp.ddpg.ou_dt})"
                 if exp.ddpg.noise_type == "ou" else "")

        logger.write_config(
            extra={
                "resumed_from": str(resume_from) if resume_from else None,
                "env_setup_seconds": round(time.time() - t0, 2),
                "action_noise_sigma": [round(float(s), 6) for s in sigma],
                "binary_action_dims": binary_action_indices(exp.env),
                **(config_extra or {}),
            }
        )

        if resume_from is not None:
            source = resolve_run(resume_from)
            model_path = source / "model.zip"
            if not model_path.exists():
                raise FileNotFoundError(f"No model.zip to resume from in {source}")
            log.info("resuming DDPG weights from %s (replay buffer starts empty)", model_path)
            stats = source / "vecnormalize.pkl"
            if isinstance(vec_env, VecNormalize) and stats.exists():
                vec_env = VecNormalize.load(str(stats), vec_env.venv)
                log.info("restored reward-normalization statistics from %s", stats.name)
            model = DDPG.load(str(model_path), env=vec_env, seed=exp.ddpg.seed)
            # SB3 pickles the source run's action noise into model.zip, so
            # rebuild it from this run's config: a resume exists to change the
            # schedule (a smaller binary std for a fine-tuning leg, say), and
            # the loaded object would silently override the new flags.
            model.action_noise = build_action_noise(
                exp, int(np.prod(vec_env.action_space.shape))
            )
        elif model_factory is not None:
            model = model_factory(exp, vec_env)
        else:
            model = build_model(exp, vec_env)

        formats = ["csv"] + (["stdout"] if exp.ddpg.verbose else [])
        model.set_logger(configure_logger(str(logger.run_dir), formats))

        info_cb = EnvInfoLoggingCallback(logger, state_labels(exp.env))
        callbacks = [info_cb]
        if exp.ddpg.checkpoint_every > 0:
            callbacks.append(CheckpointEveryCallback(logger, exp.ddpg.checkpoint_every))
        if eval_env is not None:
            callbacks.append(
                CsvEvalCallback(logger, eval_env, exp.ddpg.eval_every, exp.ddpg.eval_episodes)
            )

        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        t0 = time.time()
        model.learn(
            total_timesteps=exp.ddpg.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=resume_from is None,
        )
        elapsed = time.time() - t0
        model.save(str(logger.run_dir / "model"))
        if isinstance(vec_env, VecNormalize):
            vec_env.save(str(logger.run_dir / "vecnormalize.pkl"))
        log.info("training done in %.1fs (%.1f steps/s)",
                 elapsed, exp.ddpg.total_timesteps / max(elapsed, 1e-9))

        status = "completed"
        final_metrics = {
            k: v
            for k, v in info_cb.last_episode.items()
            if k.startswith("final_") or k in ("reward_sum", "length")
        }
        final_metrics["elapsed_sec"] = round(elapsed, 2)
        return logger.run_dir

    finally:
        if eval_env is not None:
            eval_env.close()
        if vec_env is not None:
            vec_env.close()
        logger.finalize(status=status, metrics=final_metrics)
