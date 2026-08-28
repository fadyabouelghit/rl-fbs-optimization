import numpy as np

from ppo.config import BandConfig, WorldConfig
from ppo.matlab_bridge import AnalyticSinrBackend

WORLD = WorldConfig(width=1000.0, height=800.0, num_mbs=1, num_users=300)


def _fbs(x=300.0, y=300.0, z=60.0, p=10.5):
    return np.array([[x, y, z, p]])


def test_deterministic():
    b1 = AnalyticSinrBackend(WORLD, BandConfig())
    b2 = AnalyticSinrBackend(WORLD, BandConfig())
    r1 = b1.evaluate(_fbs(), np.array([1.0]))
    r2 = b2.evaluate(_fbs(), np.array([1.0]))
    assert r1.as_metrics_dict() == r2.as_metrics_dict()


def test_tier_invariant_and_off_switch():
    backend = AnalyticSinrBackend(WORLD, BandConfig())
    on = backend.evaluate(_fbs(), np.array([1.0]))
    assert on.total_connected == (
        on.fbs_connected + on.mbs_coverage_connected + on.mbs_capacity_connected
    )
    off = backend.evaluate(_fbs(), np.array([0.0]))
    assert off.fbs_connected == 0
    assert off.total_power == 0.0
    assert on.total_power == 10.5


def test_multiband_capacity_slot_gating():
    band = BandConfig(mode="multi", fbs_band="agent", mbs_capacity="agent")
    backend = AnalyticSinrBackend(WORLD, band)
    no_cap = backend.evaluate(
        _fbs(), np.array([1.0]),
        fbs_band_flags=np.array([0.0]), mbs_capacity_genes=np.array([0.0]),
    )
    with_cap = backend.evaluate(
        _fbs(), np.array([1.0]),
        fbs_band_flags=np.array([0.0]), mbs_capacity_genes=np.array([1.0]),
    )
    assert no_cap.mbs_capacity_connected == 0
    # Capacity slot serves users on an otherwise-empty band.
    assert with_cap.mbs_capacity_connected > 0
    assert with_cap.total_connected >= no_cap.total_connected


def test_band_escape_from_interference():
    """An FBS co-located with the MBS suffers same-band interference; moving
    it to the (empty) capacity band should not reduce its connections."""
    band = BandConfig(mode="multi", fbs_band="agent", mbs_capacity="agent")
    backend = AnalyticSinrBackend(WORLD, band)
    near_mbs = _fbs(x=WORLD.width / 2 + 30, y=WORLD.height / 2 + 30)
    same_band = backend.evaluate(
        near_mbs, np.array([1.0]),
        fbs_band_flags=np.array([0.0]), mbs_capacity_genes=np.array([0.0]),
    )
    other_band = backend.evaluate(
        near_mbs, np.array([1.0]),
        fbs_band_flags=np.array([1.0]), mbs_capacity_genes=np.array([0.0]),
    )
    assert other_band.fbs_connected >= same_band.fbs_connected


def test_user_positions_only_on_request():
    backend = AnalyticSinrBackend(WORLD, BandConfig())
    r = backend.evaluate(_fbs(), np.array([1.0]))
    assert r.user_positions is None
    r = backend.evaluate(_fbs(), np.array([1.0]), collect_user_positions=True)
    assert r.user_positions.shape == (WORLD.num_users, 2)
