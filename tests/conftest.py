"""Shared fixtures. Everything here runs MATLAB-free (analytic/stub backends);
tests marked ``matlab`` are skipped unless PPO_MATLAB_TESTS=1."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppo.config import (  # noqa: E402
    BandConfig,
    EnvConfig,
    ExperimentConfig,
    PPOParams,
    RewardConfig,
    RewardWeights,
    WorldConfig,
)
from ppo.matlab_bridge import SinrResult  # noqa: E402


def small_world(**kwargs) -> WorldConfig:
    defaults = dict(width=1000.0, height=800.0, num_mbs=1, num_users=200)
    defaults.update(kwargs)
    return WorldConfig(**defaults)


@pytest.fixture
def legacy_env_config() -> EnvConfig:
    return EnvConfig(
        num_fbs=2,
        world=small_world(),
        band=BandConfig(mode="legacy"),
        reward=RewardConfig(mode="legacy_blend", weights=RewardWeights(beta=0.8)),
        max_episode_steps=10,
        action_scale=0.2,
    )


@pytest.fixture
def multi_env_config() -> EnvConfig:
    return EnvConfig(
        num_fbs=2,
        world=small_world(num_mbs=2, mbs_locations=[(250.0, 400.0), (750.0, 400.0)]),
        band=BandConfig(mode="multi", fbs_band="agent", mbs_capacity="agent"),
        reward=RewardConfig(mode="controlled_blend", weights=RewardWeights(beta=0.8, gamma=0.2)),
        max_episode_steps=10,
        action_scale=0.2,
    )


class StubBackend:
    """Returns a canned SinrResult; records the arguments it was called with."""

    def __init__(self, world, band, result: SinrResult):
        self.world = world
        self.band = band
        self.result = result
        self.mbs_x = np.array([world.width / 2])
        self.mbs_y = np.array([world.height / 2])
        self.calls: list[dict] = []

    def evaluate(self, cont_params, power_status, fbs_band_flags=None,
                 mbs_capacity_genes=None, collect_user_positions=False):
        self.calls.append({
            "cont_params": np.array(cont_params, copy=True),
            "power_status": np.array(power_status, copy=True),
            "fbs_band_flags": None if fbs_band_flags is None else np.array(fbs_band_flags, copy=True),
            "mbs_capacity_genes": None if mbs_capacity_genes is None else np.array(mbs_capacity_genes, copy=True),
        })
        return self.result

    def close(self):
        pass


@pytest.fixture
def stub_result() -> SinrResult:
    return SinrResult(
        total_connected=120,
        fbs_connected=40,
        mbs_connected=80,
        mbs_coverage_connected=70,
        mbs_capacity_connected=10,
        total_power=12.0,
        avg_rate=3.2,
        sum_rate=384.0,
    )


@pytest.fixture
def tiny_experiment() -> ExperimentConfig:
    """Very small analytic-friendly experiment for integration tests."""
    return ExperimentConfig(
        num_fbs=1,
        scenario_id=1,
        config_id=1,
        env=EnvConfig(
            num_fbs=1,
            world=small_world(num_users=100),
            band=BandConfig(mode="legacy"),
            max_episode_steps=8,
            action_scale=0.3,
        ),
        ppo=PPOParams(
            total_timesteps=128,
            n_steps=32,
            batch_size=32,
            n_epochs=2,
            seed=7,
            verbose=0,
            log_flush_every=16,
        ),
    )


@pytest.fixture(autouse=True)
def _isolate_ledger(tmp_path, monkeypatch):
    """Keep test runs out of the real training_log.csv ledger."""
    import ppo.run_logging as rl

    monkeypatch.setattr(rl, "LEDGER_PATH", tmp_path / "ledger.csv")
    yield


@pytest.fixture(autouse=True)
def _isolate_pyqd_cache(tmp_path, monkeypatch):
    """Keep pyqd power maps out of the repo's cache_pyqd_maps/.

    Any test that builds a pyqd backend without naming a cache dir (e.g. via
    ``make_backend``, whose default backend is now 'pyqd') would otherwise
    write megabyte-scale .npz files into the working tree, and parallel runs
    would race on them. tests/test_pyqd_backend.py additionally passes an
    explicit module-scoped directory, since module-scoped fixtures are built
    before this function-scoped one can patch anything.

    Both halves are needed. The attribute patch covers this process; the
    environment variable covers SubprocVecEnv workers, which are spawned fresh
    and re-import ``ppo.paths`` (where PPO_PYQD_CACHE_DIR is read) rather than
    inheriting anything monkeypatched here.
    """
    import ppo.pyqd_bridge as pb

    cache = tmp_path / "pyqd_maps"
    monkeypatch.setattr(pb, "PYQD_CACHE_DIR", cache)
    monkeypatch.setenv("PPO_PYQD_CACHE_DIR", str(cache))
    yield


def pytest_collection_modifyitems(config, items):
    if os.environ.get("PPO_MATLAB_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="MATLAB tests disabled (set PPO_MATLAB_TESTS=1)")
    for item in items:
        if "matlab" in item.keywords:
            item.add_marker(skip)
