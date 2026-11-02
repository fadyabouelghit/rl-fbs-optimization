"""Train TD3-hybrid: ``td3.train.train`` with a different model factory."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from stable_baselines3 import TD3

from ddpg_hybrid.hybrid import (HybridMixin, HybridTD3Policy, QDiffHybridMixin,
                                _check_version)
from ppo.matlab_bridge import DEFAULT_BACKEND
from td3.config import TD3Experiment, binary_action_indices
from td3.train import build_action_noise
from td3.train import train as _td3_train


class HybridTD3(HybridMixin, TD3):
    pass


def build_model(exp: TD3Experiment, vec_env, gumbel_tau: float) -> HybridTD3:
    import numpy as np

    p = exp.td3
    n_actions = int(np.prod(vec_env.action_space.shape))
    kwargs = {"binary_idx": binary_action_indices(exp.env), "gumbel_tau": float(gumbel_tau)}
    if p.net_arch:
        kwargs["net_arch"] = list(p.net_arch)
    return HybridTD3(
        HybridTD3Policy, vec_env,
        learning_rate=p.learning_rate, buffer_size=p.buffer_size,
        learning_starts=p.learning_starts, batch_size=p.batch_size, tau=p.tau,
        gamma=p.gamma, train_freq=p.train_freq, gradient_steps=p.gradient_steps,
        action_noise=build_action_noise(exp, n_actions),
        policy_delay=p.policy_delay, target_policy_noise=p.target_policy_noise,
        target_noise_clip=p.target_noise_clip,
        policy_kwargs=kwargs, seed=p.seed, verbose=p.verbose,
    )


def train(exp: TD3Experiment, backend: str = DEFAULT_BACKEND, run_dir: Optional[Path] = None,
          *, gumbel_tau: float = 1.0, extra_callbacks: Optional[list] = None,
          config_extra: Optional[dict] = None) -> Path:
    _check_version()
    if exp.td3.binary_noise_std != 0.0:
        raise ValueError("td3_hybrid needs binary_noise_std == 0.0")
    extra = {**(config_extra or {}),
             "hybrid": {"kind": "bernoulli_st_gumbel", "gumbel_tau": float(gumbel_tau),
                        "binary_idx": binary_action_indices(exp.env)}}
    return _td3_train(exp, backend=backend, run_dir=run_dir, extra_callbacks=extra_callbacks,
                      model_factory=lambda e, venv: build_model(e, venv, gumbel_tau),
                      config_extra=extra)


# --------------------------------------------------------------------------- #
# Q-difference variant (letter Step 2)
# --------------------------------------------------------------------------- #
class QDiffHybridTD3(QDiffHybridMixin, TD3):
    pass


def build_model_qd(exp: TD3Experiment, vec_env, gumbel_tau: float) -> QDiffHybridTD3:
    import numpy as np

    p = exp.td3
    n_actions = int(np.prod(vec_env.action_space.shape))
    kwargs = {"binary_idx": binary_action_indices(exp.env), "gumbel_tau": float(gumbel_tau)}
    if p.net_arch:
        kwargs["net_arch"] = list(p.net_arch)
    return QDiffHybridTD3(
        HybridTD3Policy, vec_env,
        learning_rate=p.learning_rate, buffer_size=p.buffer_size,
        learning_starts=p.learning_starts, batch_size=p.batch_size, tau=p.tau,
        gamma=p.gamma, train_freq=p.train_freq, gradient_steps=p.gradient_steps,
        action_noise=build_action_noise(exp, n_actions),
        policy_delay=p.policy_delay, target_policy_noise=p.target_policy_noise,
        target_noise_clip=p.target_noise_clip,
        policy_kwargs=kwargs, seed=p.seed, verbose=p.verbose,
    )


def train_qd(exp: TD3Experiment, backend: str = DEFAULT_BACKEND, run_dir: Optional[Path] = None,
             *, gumbel_tau: float = 1.0, extra_callbacks: Optional[list] = None,
             config_extra: Optional[dict] = None) -> Path:
    _check_version()
    if exp.td3.binary_noise_std != 0.0:
        raise ValueError("td3_hybrid_qd needs binary_noise_std == 0.0")
    extra = {**(config_extra or {}),
             "hybrid": {"kind": "bernoulli_qdiff", "gumbel_tau": float(gumbel_tau),
                        "binary_idx": binary_action_indices(exp.env)}}
    return _td3_train(exp, backend=backend, run_dir=run_dir, extra_callbacks=extra_callbacks,
                      model_factory=lambda e, venv: build_model_qd(e, venv, gumbel_tau),
                      config_extra=extra)
