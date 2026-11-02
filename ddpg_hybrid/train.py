"""Train DDPG-hybrid: ``ddpg.train.train`` with a different model factory."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from stable_baselines3 import DDPG

from ddpg.config import DDPGExperiment, binary_action_indices
from ddpg.train import build_action_noise
from ddpg.train import train as _ddpg_train
from ppo.matlab_bridge import DEFAULT_BACKEND

from .hybrid import HybridMixin, HybridTD3Policy, QDiffHybridMixin, _check_version


class HybridDDPG(HybridMixin, DDPG):
    pass


def build_model(exp: DDPGExperiment, vec_env, gumbel_tau: float) -> HybridDDPG:
    import numpy as np

    p = exp.ddpg
    n_actions = int(np.prod(vec_env.action_space.shape))
    kwargs = {"binary_idx": binary_action_indices(exp.env), "gumbel_tau": float(gumbel_tau)}
    if p.net_arch:
        kwargs["net_arch"] = list(p.net_arch)
    return HybridDDPG(
        HybridTD3Policy, vec_env,
        learning_rate=p.learning_rate, buffer_size=p.buffer_size,
        learning_starts=p.learning_starts, batch_size=p.batch_size, tau=p.tau,
        gamma=p.gamma, train_freq=p.train_freq, gradient_steps=p.gradient_steps,
        action_noise=build_action_noise(exp, n_actions),
        policy_kwargs=kwargs, seed=p.seed, verbose=p.verbose,
    )


def train(exp: DDPGExperiment, backend: str = DEFAULT_BACKEND, run_dir: Optional[Path] = None,
          *, gumbel_tau: float = 1.0, extra_callbacks: Optional[list] = None,
          config_extra: Optional[dict] = None) -> Path:
    _check_version()
    if exp.ddpg.binary_noise_std != 0.0:
        raise ValueError("ddpg_hybrid needs binary_noise_std == 0.0: the Bernoulli sample is "
                         "the binary dims' exploration")
    extra = {**(config_extra or {}),
             "hybrid": {"kind": "bernoulli_st_gumbel", "gumbel_tau": float(gumbel_tau),
                        "binary_idx": binary_action_indices(exp.env)}}
    return _ddpg_train(exp, backend=backend, run_dir=run_dir, extra_callbacks=extra_callbacks,
                       model_factory=lambda e, venv: build_model(e, venv, gumbel_tau),
                       config_extra=extra)


# --------------------------------------------------------------------------- #
# Q-difference variant (letter Step 2)
# --------------------------------------------------------------------------- #
class QDiffHybridDDPG(QDiffHybridMixin, DDPG):
    pass


def build_model_qd(exp: DDPGExperiment, vec_env, gumbel_tau: float) -> QDiffHybridDDPG:
    import numpy as np

    p = exp.ddpg
    n_actions = int(np.prod(vec_env.action_space.shape))
    kwargs = {"binary_idx": binary_action_indices(exp.env), "gumbel_tau": float(gumbel_tau)}
    if p.net_arch:
        kwargs["net_arch"] = list(p.net_arch)
    return QDiffHybridDDPG(
        HybridTD3Policy, vec_env,
        learning_rate=p.learning_rate, buffer_size=p.buffer_size,
        learning_starts=p.learning_starts, batch_size=p.batch_size, tau=p.tau,
        gamma=p.gamma, train_freq=p.train_freq, gradient_steps=p.gradient_steps,
        action_noise=build_action_noise(exp, n_actions),
        policy_kwargs=kwargs, seed=p.seed, verbose=p.verbose,
    )


def train_qd(exp: DDPGExperiment, backend: str = DEFAULT_BACKEND, run_dir: Optional[Path] = None,
             *, gumbel_tau: float = 1.0, extra_callbacks: Optional[list] = None,
             config_extra: Optional[dict] = None) -> Path:
    _check_version()
    if exp.ddpg.binary_noise_std != 0.0:
        raise ValueError("ddpg_hybrid_qd needs binary_noise_std == 0.0")
    extra = {**(config_extra or {}),
             "hybrid": {"kind": "bernoulli_qdiff", "gumbel_tau": float(gumbel_tau),
                        "binary_idx": binary_action_indices(exp.env)}}
    return _ddpg_train(exp, backend=backend, run_dir=run_dir, extra_callbacks=extra_callbacks,
                       model_factory=lambda e, venv: build_model_qd(e, venv, gumbel_tau),
                       config_extra=extra)
