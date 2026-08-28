"""Tests against the real MATLAB/QuaDRiGa stack.

Skipped by default (slow: engine start + power-map computation). Enable with:

    PPO_MATLAB_TESTS=1 ./venv/bin/pytest tests/test_matlab_backend.py -v

Uses one shared engine across all tests in the module.
"""
import numpy as np
import pytest

from ppo.config import BandConfig, WorldConfig

pytestmark = pytest.mark.matlab


@pytest.fixture(scope="module")
def session():
    from ppo.matlab_bridge import MatlabSession

    s = MatlabSession.start()
    yield s
    s.close()


@pytest.fixture(scope="module")
def world():
    return WorldConfig(width=2000.0, height=1500.0, num_mbs=1, num_users=200)


def test_legacy_world_and_eval(session, world):
    from ppo.matlab_bridge import MatlabSinrBackend

    backend = MatlabSinrBackend(world, BandConfig(mode="legacy"), session=session)
    assert backend.num_mbs == 1
    assert backend.n_bands == 1
    tx = np.array([[800.0, 800.0, 100.0, 10.5]])
    r = backend.evaluate(tx, np.array([1.0]), collect_user_positions=True)
    assert r.total_connected == r.fbs_connected + r.mbs_connected
    assert r.user_positions.shape == (world.num_users, 2)
    assert r.total_power == pytest.approx(10.5)
    # determinism: users are seed-0 fixed
    r2 = backend.evaluate(tx, np.array([1.0]))
    assert r2.as_metrics_dict() == r.as_metrics_dict()
    backend.close()


def test_multiband_world_and_slot_gating(session, world):
    from ppo.matlab_bridge import MatlabSinrBackend

    band = BandConfig(mode="multi", fbs_band="agent", mbs_capacity="agent")
    backend = MatlabSinrBackend(world, band, session=session)
    assert backend.n_bands == 2
    tx = np.array([[800.0, 800.0, 100.0, 10.5]])

    no_cap = backend.evaluate(
        tx, np.array([1.0]),
        fbs_band_flags=np.array([0.0]), mbs_capacity_genes=np.array([0.0]),
    )
    with_cap = backend.evaluate(
        tx, np.array([1.0]),
        fbs_band_flags=np.array([0.0]), mbs_capacity_genes=np.array([1.0]),
    )
    assert no_cap.mbs_capacity_connected == 0
    assert with_cap.total_connected == (
        with_cap.fbs_connected
        + with_cap.mbs_coverage_connected
        + with_cap.mbs_capacity_connected
    )
    backend.close()


def test_multiband_all_coverage_matches_tier_invariant(session, world):
    """FBS pinned to coverage with capacity slots off: capacity tier empty."""
    from ppo.matlab_bridge import MatlabSinrBackend

    band = BandConfig(mode="multi", fbs_band="coverage", mbs_capacity="off")
    backend = MatlabSinrBackend(world, band, session=session)
    tx = np.array([[500.0, 400.0, 80.0, 9.0]])
    r = backend.evaluate(
        tx, np.array([1.0]),
        fbs_band_flags=np.array([0.0]), mbs_capacity_genes=np.array([0.0]),
    )
    assert r.mbs_capacity_connected == 0
    backend.close()
