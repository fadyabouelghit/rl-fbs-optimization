"""Train a try-both agent through ``td3.train.train`` / ``ddpg.train.train``.

    train(exp, backend, run_dir, pilot="td3", params=TryBothParams(exploration_fraction=0.2))

``exp`` is the parent package's own experiment (e.g. from
``tuning.spaces.build_experiment('on_off_td3', ...)``) with
``binary_noise_std`` None: exploration noise goes on the 4N continuous dims
only, built here; b's exploration is eps-greedy over patterns.

Artifacts are the parent's usual run directory; ``model.zip`` and
``best_model.zip`` are complete policies (actor + critic), loadable with plain
``TD3.load`` / ``DDPG.load``.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from ppo.matlab_bridge import DEFAULT_BACKEND
from ppo.paths import REPO_ROOT, setting

from .algo import TryBothDDPG, TryBothTD3, check_version
from .config import PILOT_OF, TryBothParams
from .policy import TryBothPolicy

ON_OFF_TD3_RUNS_DIR = Path(setting("ON_OFF_TD3_RUNS_DIR", str(REPO_ROOT / "on_off_td3_runs"))).expanduser()
_PROTECTED = {"td3": REPO_ROOT / "td3_runs", "ddpg": REPO_ROOT / "ddpg_runs"}


@contextmanager
def routed(pilot: str):
    """Point the parent's run dir + ledger away from its protected default, if it is there."""
    if pilot == "td3":
        import td3.paths as p
        d, l = "TD3_RUNS_DIR", "TD3_LEDGER_PATH"
    else:
        import ddpg.paths as p
        d, l = "DDPG_RUNS_DIR", "DDPG_LEDGER_PATH"
    old_d, old_l = getattr(p, d), getattr(p, l)
    if Path(old_d).resolve() == _PROTECTED[pilot].resolve() or \
            Path(old_l).parent.resolve() == _PROTECTED[pilot].resolve():
        setattr(p, d, ON_OFF_TD3_RUNS_DIR)
        setattr(p, l, ON_OFF_TD3_RUNS_DIR / "ledger.csv")
    try:
        yield
    finally:
        setattr(p, d, old_d)
        setattr(p, l, old_l)


def _layout(env_cfg):
    from ddqn_td3.config import Layout       # same index maps; refuses band / capacity genes
    return Layout(env_cfg)


def build_model(exp, vec_env, pilot: str, params: TryBothParams):
    layout = _layout(exp.env)
    if pilot == "td3":
        from td3.train import build_action_noise
        P, cls = exp.td3, TryBothTD3
        extra = dict(policy_delay=P.policy_delay, target_policy_noise=P.target_policy_noise,
                     target_noise_clip=P.target_noise_clip)
    else:
        from ddpg.train import build_action_noise
        P, cls = exp.ddpg, TryBothDDPG
        extra = {}
    kwargs = {"onoff_idx": layout.onoff_idx, "cont_idx": layout.cont_idx}
    if P.net_arch:
        kwargs["net_arch"] = list(P.net_arch)
    model = cls(
        TryBothPolicy, vec_env,
        learning_rate=P.learning_rate, buffer_size=P.buffer_size,
        learning_starts=P.learning_starts, batch_size=P.batch_size, tau=P.tau,
        gamma=P.gamma, train_freq=P.train_freq, gradient_steps=P.gradient_steps,
        action_noise=build_action_noise(exp, layout.n_cont),     # 4N continuous dims only
        policy_kwargs=kwargs, seed=P.seed, verbose=P.verbose, **extra,
    )
    model.set_tryboth(params, P.total_timesteps)
    return model


def train(exp, backend: str = DEFAULT_BACKEND, run_dir: Optional[Path] = None, *,
          pilot: str = "td3", params: TryBothParams = TryBothParams(),
          extra_callbacks: Optional[list] = None, config_extra: Optional[dict] = None) -> Path:
    check_version()
    if pilot == "td3":
        from td3.train import train as parent_train
        P = exp.td3
    elif pilot == "ddpg":
        from ddpg.train import train as parent_train
        P = exp.ddpg
    else:
        raise ValueError(f"unknown pilot {pilot!r}")
    if P.binary_noise_std is not None:
        raise ValueError("on_off_td3: binary_noise_std must be None (b is explored by eps)")
    layout = _layout(exp.env)
    extra = {**(config_extra or {}),
             "on_off": {"pilot": pilot, "params": params.to_dict(), "n_fbs": layout.n_fbs,
                         "onoff_idx": layout.onoff_idx, "cont_idx": layout.cont_idx,
                         "selection": "online critics' min (TD3) / critic (DDPG)"}}
    with routed(pilot):
        return Path(parent_train(exp, backend=backend, run_dir=run_dir,
                                 extra_callbacks=extra_callbacks, config_extra=extra,
                                 model_factory=lambda e, venv: build_model(e, venv, pilot, params)))


def train_algo(algo: str, exp, backend: str = DEFAULT_BACKEND, *, params: dict, **kwargs) -> Path:
    """Entry point for ``tuning.worker`` / ``letterexp.runner`` ('on_off_td3' | 'on_off_ddpg')."""
    return train(exp, backend=backend, pilot=PILOT_OF[algo],
                 params=TryBothParams(exploration_fraction=float(params["exploration_fraction"])),
                 **kwargs)
