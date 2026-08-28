"""Training orchestration.

``train(exp)`` runs one PPO training under a fresh run directory with full
logging (see run_logging.py for the artifact layout). Key properties:

- **Reproducibility**: python/numpy/torch and every env are seeded from
  ``exp.ppo.seed``. A fixed (seed, n_envs) pair reproduces a run exactly;
  changing ``n_envs`` changes the rollout interleaving (like any other
  hyperparameter), which is why parallelism is opt-in.
- **Parallelism**: ``exp.ppo.n_envs > 1`` uses SubprocVecEnv; each worker
  process starts its own MATLAB engine (~1-2 GB each). Worker i is seeded
  ``seed + i`` by SB3.
- **In-training eval**: with ``exp.ppo.eval_every > 0`` a dedicated eval env
  runs seeded deterministic episodes and tracks ``best_model.zip``. For
  ``n_envs == 1`` it shares the training env's MATLAB backend (no second
  engine); for parallel training it builds its own.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from .callbacks import CheckpointEveryCallback, CsvEvalCallback, EnvInfoLoggingCallback
from .config import EnvConfig, ExperimentConfig
from .env import FlyingBaseStationEnv, state_labels
from .matlab_bridge import MatlabSession
from .run_logging import RunLogger
from .runs import find_model_path, model_class_for, resolve_run


def _build_model(exp: ExperimentConfig, vec_env):
    """Construct the SB3 model selected by ``exp.ppo.algo``. PPO-only fields
    are ignored under SAC and vice versa (see PPOParams)."""
    p = exp.ppo
    if p.algo == "sac":
        return SAC(
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
            ent_coef=p.sac_ent_coef,
            seed=p.seed,
            verbose=p.verbose,
        )
    return PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=p.learning_rate,
        n_steps=p.n_steps,
        batch_size=p.batch_size,
        n_epochs=p.n_epochs,
        gamma=p.gamma,
        gae_lambda=p.gae_lambda,
        clip_range=p.clip_range,
        ent_coef=p.ent_coef,
        vf_coef=p.vf_coef,
        max_grad_norm=p.max_grad_norm,
        seed=p.seed,
        verbose=p.verbose,
    )


def _make_worker_env(env_config: EnvConfig, backend_kind: str, monitor_path: str):
    """Top-level factory (picklable) for SubprocVecEnv workers. The backend
    (and its MATLAB engine, if any) is created inside the worker process."""
    def _init():
        env = FlyingBaseStationEnv(env_config, backend_kind=backend_kind)
        return Monitor(env, filename=monitor_path, allow_early_resets=True)
    return _init


def train(
    exp: ExperimentConfig,
    backend: str = "matlab",
    session: Optional[MatlabSession] = None,
    run_dir: Optional[Path] = None,
    resume_from: Optional[str | Path] = None,
) -> Path:
    """Train PPO under `exp`; returns the run directory.

    backend      : 'matlab' (real physics) or 'analytic' (fast stand-in).
    session      : optional pre-started MatlabSession to reuse (n_envs == 1).
    resume_from  : run name/path whose model.zip seeds the optimizer state;
                   training continues into a NEW run dir (provenance recorded).
    """
    logger = RunLogger(exp, run_dir=run_dir, backend_kind=backend)
    log = logger.log
    log.info("run dir: %s", logger.run_dir)
    log.info("config: code=%s algo=%s band=%s reward=%s shaping=%s backend=%s n_envs=%d seed=%s",
             exp.code, exp.ppo.algo, exp.env.band.mode, exp.env.reward.mode,
             exp.env.reward.shaping, backend, exp.ppo.n_envs, exp.ppo.seed)

    if exp.ppo.seed is not None:
        set_random_seed(exp.ppo.seed)

    n_envs = exp.ppo.n_envs
    train_env = None
    eval_env = None
    vec_env = None
    status = "failed"
    final_metrics: dict = {}

    try:
        # -------------------------------------------------------------- #
        # Environments
        # -------------------------------------------------------------- #
        t0 = time.time()
        if n_envs == 1:
            train_env = FlyingBaseStationEnv(
                exp.env, backend_kind=backend, session=session
            )
            monitored = Monitor(
                train_env,
                filename=str(logger.run_dir / "monitor.csv"),
                allow_early_resets=True,
            )
            vec_env = DummyVecEnv([lambda: monitored])
            if exp.ppo.eval_every > 0:
                # Shares the training backend — no second MATLAB engine.
                eval_env = FlyingBaseStationEnv(exp.env, backend=train_env.backend)
        else:
            factories = [
                _make_worker_env(
                    exp.env, backend, str(logger.run_dir / f"monitor_{i}.csv")
                )
                for i in range(n_envs)
            ]
            vec_env = SubprocVecEnv(factories, start_method="spawn")
            if exp.ppo.eval_every > 0:
                eval_env = FlyingBaseStationEnv(
                    exp.env, backend_kind=backend, session=session
                )
        if exp.ppo.normalize_reward:
            # norm_obs=False: the env already scales observations itself
            # (EnvConfig.normalize_obs), and leaving obs untouched keeps the
            # saved policy usable against a plain env at evaluation time.
            # Only the training reward is normalized; info["reward_objective"]
            # stays raw, so every logged metric and comparison is unaffected.
            vec_env = VecNormalize(
                vec_env, norm_obs=False, norm_reward=True, gamma=exp.ppo.gamma
            )
            log.info("running reward normalization enabled (VecNormalize)")
        log.info("env(s) ready in %.1fs", time.time() - t0)

        logger.write_config(
            extra={
                "resumed_from": str(resume_from) if resume_from else None,
                "env_setup_seconds": round(time.time() - t0, 2),
            }
        )

        # -------------------------------------------------------------- #
        # Model
        # -------------------------------------------------------------- #
        if resume_from is not None:
            source = resolve_run(resume_from)
            model_path = find_model_path(source)
            if model_path is None:
                raise FileNotFoundError(f"No model to resume from in {source}")
            algo_cls = model_class_for(exp)
            log.info("resuming %s optimizer state from %s", algo_cls.__name__, model_path)
            stats = source / "vecnormalize.pkl"
            if isinstance(vec_env, VecNormalize) and stats.exists():
                vec_env = VecNormalize.load(str(stats), vec_env.venv)
                log.info("restored reward-normalization statistics from %s", stats.name)
            model = algo_cls.load(str(model_path), env=vec_env, seed=exp.ppo.seed)
        else:
            model = _build_model(exp, vec_env)

        # SB3 optimizer diagnostics -> progress.csv (CSV-only stack)
        formats = ["csv"] + (["stdout"] if exp.ppo.verbose else [])
        model.set_logger(configure_logger(str(logger.run_dir), formats))

        info_cb = EnvInfoLoggingCallback(logger, state_labels(exp.env))
        callbacks = [info_cb]
        if exp.ppo.checkpoint_every > 0:
            callbacks.append(CheckpointEveryCallback(logger, exp.ppo.checkpoint_every))
        if eval_env is not None:
            callbacks.append(
                CsvEvalCallback(
                    logger, eval_env, exp.ppo.eval_every, exp.ppo.eval_episodes
                )
            )

        # -------------------------------------------------------------- #
        # Learn + save
        # -------------------------------------------------------------- #
        t0 = time.time()
        model.learn(
            total_timesteps=exp.ppo.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=resume_from is None,
        )
        elapsed = time.time() - t0
        model.save(str(logger.run_dir / "model"))
        if isinstance(vec_env, VecNormalize):
            # Needed to resume training faithfully; not needed to evaluate,
            # since observations are passed through unnormalized.
            vec_env.save(str(logger.run_dir / "vecnormalize.pkl"))
        log.info("training done in %.1fs (%.1f steps/s)",
                 elapsed, exp.ppo.total_timesteps / max(elapsed, 1e-9))

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
