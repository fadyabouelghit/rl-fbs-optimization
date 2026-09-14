"""Fidelity tests for the MATLAB-free pyqd physics backend.

Ungated on purpose: unlike ``tests/test_matlab_backend.py`` these need no
MATLAB engine and no QuaDRiGa install, so they belong in the default suite.

Three cost tiers:

- Most tests run on a small 600x400 world so that a full power map is ~0.02 s.
  They pin the *semantics* — sampling convention, band bookkeeping, tier
  invariants, association tie-breaking, the deliberate MATLAB quirks. A
  two-site variant of the same world covers everything that only exists with
  more than one MBS: per-site map indexing and per-site slot expansion.
- A handful use the real 2000x1500 default world because that is where ground
  truth lives: the archived MATLAB user map, the archived MATLAB MBS power
  map, and a completed MATLAB run's metrics. They share one module-scoped
  backend so the expensive map is computed once.
- One reads the archived 4000x3000 scenario-2 maps, the only MATLAB ground
  truth for a *second* MBS site. It samples the model at a few hundred grid
  nodes rather than building four 4001x3001 maps (~10 s and >3 GB).

Some tests compare against MATLAB output that ships with the *old* combined
repo (the power-map caches, a completed GA run's user table). Those paths are
machine-specific, so exactly those tests skip rather than fail when the archive
is absent, and each is guarded on the file its own body opens. Everything else
runs everywhere — including ``test_end_to_end_reproduces_archived_matlab_metrics``,
whose expected values are archived *constants*, not files. That test is the one
that catches a broken row swap, transpose, total-power quirk or noise floor, so
gating it on a machine-specific path would silently retire the entire
no-MATLAB equivalence claim on every other machine.
"""
from __future__ import annotations

import dataclasses
import hashlib
import time
from pathlib import Path

import numpy as np
import pytest

from ppo.config import (
    BAND_CAPACITY,
    BAND_COVERAGE,
    SCENARIOS,
    BandConfig,
    WorldConfig,
)
from ppo.matlab_bridge import make_backend
from ppo.pyqd_bridge import (
    MapSpec,
    PyqdSinrBackend,
    build_slots,
    check_fast_sampling_invariants,
    compute_power_map,
    generate_hex_sites,
    load_or_compute_map,
    matlab_colon,
    sample_nearest,
    sample_power_at,
)

# The world the whole archive was produced under (ppo/config.py defaults).
DEFAULT_WORLD = WorldConfig(width=2000.0, height=1500.0, num_mbs=1, num_users=1000)

# Cheap world for the mechanical tests: one full map is ~0.02 s, not ~0.6 s.
# The MBS sits in one corner and ``_fbs()`` lands in the opposite one, so both
# tiers actually serve users -- a world where the macro wins everywhere would
# make "the FBS contributes nothing when switched off" vacuously true.
SMALL_WORLD = WorldConfig(
    width=600.0, height=400.0, num_mbs=1,
    mbs_locations=[(150.0, 100.0)], num_users=120,
)

# Same cheap footprint, two MBS sites. With one site ``_mbs_columns[b, site]``
# and ``_mbs_columns[b, 0]`` are indistinguishable and "interleaved per site"
# and "grouped by band" are the same list, so nothing about the multi-site path
# is actually exercised by SMALL_WORLD -- yet half the shipped scenario codes
# (SCENARIOS[2], the ``*-2-*`` codes) use two sites. Both x coordinates are
# <= height so that the swapped-frame clamp does not swallow the difference
# between the two sites.
TWO_SITE_WORLD = WorldConfig(
    width=600.0, height=400.0, num_mbs=2,
    mbs_locations=[(150.0, 100.0), (350.0, 300.0)], num_users=120,
)

#: MATLAB-computed reference map for the default world's single MBS site, from
#: the read-only ground-truth repo. v7.3 .mat == HDF5, so h5py reads it — and
#: h5py hands back the storage order, which is the transpose of the MATLAB
#: array. See ``_load_matlab_mbs_map``.
_GROUND_TRUTH_REPO = Path(
    "/Users/fadya/Documents/MATLAB/GA_github/genetic-algorithm-optimization/"
    "genetic-algorithm-optimization"
)
MATLAB_MBS_MAP = (
    _GROUND_TRUTH_REPO / "cache_mbs_maps" / "a5239012dfb605f6bb39001880639153.mat"
)
#: A completed MATLAB GA run on the default world. ``power_FBS_3`` is the
#: MATLAB-sampled MBS coverage column for the same 1000 seed-0 users.
GA_USER_TABLE = (
    _GROUND_TRUTH_REPO / "ga_runs" / "run_2026-06-23_03-39-01_2-1-1" / "user_table.csv"
)

#: The four MATLAB-computed maps for SCENARIOS[2] (4000x3000, sites at
#: (1000, 1000) and (3500, 2200)), keyed by ``(site, band_row)``. Identified by
#: the ``meta`` each file carries: ``x``/``y`` are the SWAPPED transmitter
#: (site 1 is stored as (2200, 3500)) and ``band`` is 1 for coverage, 2 for
#: capacity. This is the only MATLAB ground truth that exists for a second MBS
#: site.
_MBS_CACHE = _GROUND_TRUTH_REPO / "cache_mbs_maps"
SCENARIO_2_MBS_MAPS = {
    (0, 0): _MBS_CACHE / "912b945c467e6e56be1ef93a715cbebe.mat",
    (0, 1): _MBS_CACHE / "b71c0d8c299208e9c573c52e3eeeb757.mat",
    (1, 0): _MBS_CACHE / "f935552ea6e0c7c7803fa20b3b116abf.mat",
    (1, 1): _MBS_CACHE / "7c2ed948760c5b156fafb0f79234ceee.mat",
}

#: First eight (x, y) user positions MATLAB's RandStream('mt19937ar','Seed',0)
#: produces for the default world, straight out of the archived user tables.
EXPECTED_USER_X = [1630, 1812, 254, 1827, 1265, 196, 557, 1094]
EXPECTED_USER_Y = [947, 533, 1496, 337, 979, 908, 581, 214]
#: SHA-256 over the full 1000x2 int64 (x, y) block. Ground truth: this is the
#: digest of ``user_x``/``user_y`` as MATLAB wrote them into the archived GA
#: user table, so it pins all 1000 draws — not just the eight above — on a
#: machine that has no copy of the archive.
EXPECTED_USER_SHA256 = (
    "5faabcecb3c4a0a48a5767b31e8f83cdaf8c5052f168a44dfbaf10bd7a52ffeb"
)


def _fbs(x=480.0, y=320.0, z=60.0, p=10.5):
    return np.array([[x, y, z, p]], dtype=float)


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory) -> Path:
    """Module-scoped map cache. Never the repo's, so tests cannot pollute it."""
    return tmp_path_factory.mktemp("pyqd_maps")


@pytest.fixture(scope="module")
def small_backend(cache_dir) -> PyqdSinrBackend:
    return PyqdSinrBackend(SMALL_WORLD, BandConfig(mode="legacy"), cache_dir=cache_dir)


@pytest.fixture(scope="module")
def small_multi_backend(cache_dir) -> PyqdSinrBackend:
    band = BandConfig(mode="multi", fbs_band="agent", mbs_capacity="agent")
    return PyqdSinrBackend(SMALL_WORLD, band, cache_dir=cache_dir)


@pytest.fixture(scope="module")
def two_site_backend(cache_dir) -> PyqdSinrBackend:
    """Two sites x two bands = four maps, ~0.07 s each on this world."""
    band = BandConfig(mode="multi", fbs_band="agent", mbs_capacity="agent")
    return PyqdSinrBackend(TWO_SITE_WORLD, band, cache_dir=cache_dir)


@pytest.fixture(scope="module")
def default_backend(cache_dir) -> PyqdSinrBackend:
    """The real 2000x1500 world — one 2001x1501 MBS map, computed once."""
    return PyqdSinrBackend(DEFAULT_WORLD, BandConfig(mode="legacy"), cache_dir=cache_dir)


# --------------------------------------------------------------------------- #
# World construction
# --------------------------------------------------------------------------- #
def test_matlab_colon_keeps_the_endpoint_row():
    """A naive arange loses the 4th hex row; MATLAB's tolerance keeps it."""
    rows = matlab_colon(100.0, 500.0 * np.sqrt(3) / 2, 1400.0)
    assert rows.size == 4
    assert rows[2] == pytest.approx(966.0254037844386, abs=1e-12)


def test_hex_sites_match_the_matlab_generator():
    xs, ys = generate_hex_sites(2000.0, 1500.0, 500.0, 100.0, 1)
    # Ground truth: the archived MBS map's meta records the SWAPPED tx as
    # (966.025404, 1100.0), i.e. true (x, y) = (1100.0, 966.0254...).
    assert xs.tolist() == pytest.approx([1100.0])
    assert ys[0] == pytest.approx(966.0254037844386, abs=1e-9)


def test_backend_exposes_true_frame_mbs_coordinates(default_backend):
    """ppo_world_setup.m:88-89 returns UNswapped coordinates to Python."""
    assert default_backend.num_mbs == 1
    assert default_backend.contains_mbs is True
    assert default_backend.n_bands == 1
    assert default_backend.mbs_x[0] == pytest.approx(1100.0)
    assert default_backend.mbs_y[0] == pytest.approx(966.0254037844386, abs=1e-9)


def test_multi_mode_has_two_bands(small_multi_backend):
    assert small_multi_backend.n_bands == 2


def test_backend_conforms_to_the_protocol_surface(small_backend):
    for attr in ("world", "band", "mbs_x", "mbs_y", "num_mbs", "contains_mbs", "n_bands"):
        assert hasattr(small_backend, attr), attr
    assert np.asarray(small_backend.mbs_x, dtype=float).ndim == 1
    assert np.asarray(small_backend.mbs_y, dtype=float).ndim == 1


def test_factory_builds_pyqd_backends():
    assert isinstance(make_backend(SMALL_WORLD, BandConfig(), "pyqd"), PyqdSinrBackend)
    fast = make_backend(SMALL_WORLD, BandConfig(), "pyqd-fast")
    assert isinstance(fast, PyqdSinrBackend) and fast.fast_sampling is True
    # A session is always forwarded by the env; the pyqd branch must drop it.
    assert isinstance(
        make_backend(SMALL_WORLD, BandConfig(), "pyqd", session=object()),
        PyqdSinrBackend,
    )
    with pytest.raises(ValueError, match="unknown backend"):
        make_backend(SMALL_WORLD, BandConfig(), "nope")


# --------------------------------------------------------------------------- #
# User positions: bit-exact against the MATLAB stream
# --------------------------------------------------------------------------- #
def test_user_positions_match_matlab_ground_truth(default_backend):
    users = default_backend.user_positions
    assert users.shape == (1000, 2)
    assert users[:8, 0].astype(int).tolist() == EXPECTED_USER_X
    assert users[:8, 1].astype(int).tolist() == EXPECTED_USER_Y
    # All 1000 draws, not just the first eight: an MT19937 defect that only
    # appears after the first twist boundary (draw index >= 624, i.e. user 312
    # onward) or in the y column -- which starts 2000 words into the same
    # stream -- is invisible to an 8-element check.
    digest = hashlib.sha256(users.astype(np.int64).tobytes()).hexdigest()
    assert digest == EXPECTED_USER_SHA256
    # generate_user_positions clamps the minima up to 1 (SINREvaluation.m:136-137),
    # so nothing is ever drawn at 0 even though the subset starts there.
    assert users[:, 0].min() >= 1 and users[:, 0].max() <= DEFAULT_WORLD.width
    assert users[:, 1].min() >= 1 and users[:, 1].max() <= DEFAULT_WORLD.height


def test_matlab_rng_matches_the_reference_stream():
    """ppo/matlab_rng.py's self-test, promoted out of ``__main__``.

    pytest never runs a module's ``if __name__ == "__main__"`` block, so the
    strongest statement about the RNG was previously unexercised by the suite.
    """
    from ppo.matlab_rng import MatlabMT19937, matlab_user_positions

    # MATLAB `rng(0); rand` and `rng(1); rand`. Seed 0 maps to MT19937's own
    # default seed 5489, which is why numpy's RandomState(0) does not match.
    assert MatlabMT19937(0).rand(1)[0] == pytest.approx(0.8147236863931789, abs=1e-16)
    assert MatlabMT19937(1).rand(1)[0] == pytest.approx(0.4170220047025740, abs=1e-16)
    drawn = matlab_user_positions(0, 2000, 0, 1500, 1000, seed=0)
    assert drawn.shape == (1000, 2)
    assert hashlib.sha256(drawn.tobytes()).hexdigest() == EXPECTED_USER_SHA256


@pytest.mark.skipif(
    not GA_USER_TABLE.is_file(),
    reason="MATLAB ground-truth GA run not present on this machine",
)
def test_every_user_position_matches_the_archived_matlab_table(default_backend):
    """The digest above, spelled out against the MATLAB file it came from."""
    table = pytest.importorskip("pandas").read_csv(GA_USER_TABLE)
    users = default_backend.user_positions
    assert np.array_equal(users[:, 0].astype(np.int64), table["user_x"].to_numpy())
    assert np.array_equal(users[:, 1].astype(np.int64), table["user_y"].to_numpy())


def test_user_positions_only_returned_on_request(small_backend):
    assert small_backend.evaluate(_fbs(), np.array([1.0])).user_positions is None
    r = small_backend.evaluate(_fbs(), np.array([1.0]), collect_user_positions=True)
    assert r.user_positions.shape == (SMALL_WORLD.num_users, 2)


# --------------------------------------------------------------------------- #
# Power maps: the MATLAB oracle
# --------------------------------------------------------------------------- #
def _load_matlab_mbs_map() -> np.ndarray:
    """The archived MATLAB map, in MATLAB's own (n_y, n_x) orientation.

    MATLAB v7.3 files are HDF5 written column-major, so h5py reports the
    transpose of the MATLAB array: the dataset looks (2001, 1501) while
    ``size(map)`` in MATLAB is ``[1501, 2001]``. The ``.T`` puts it back.
    """
    h5py = pytest.importorskip("h5py")
    with h5py.File(MATLAB_MBS_MAP, "r") as f:
        return np.array(f["map"]).T


@pytest.mark.skipif(
    not MATLAB_MBS_MAP.is_file(),
    reason="MATLAB ground-truth map cache not present on this machine",
)
def test_mbs_map_matches_the_matlab_cache(default_backend):
    """The precomputed MBS map must reproduce MATLAB/QuaDRiGa to < 1e-4 dB."""
    reference = _load_matlab_mbs_map()
    spec = default_backend._mbs_spec(0, 2.0e9)
    # Sanity-check the swap before comparing: MATLAB computed this map with
    # tx = [true_y, true_x, height] (ppo_world_setup.m:59-61).
    assert spec.tx_x == pytest.approx(966.0254037844386, abs=1e-9)
    assert spec.tx_y == pytest.approx(1100.0)

    ours = compute_power_map(spec).astype(np.float32)
    assert ours.shape == reference.shape == (1501, 2001)
    max_db = np.abs(
        10.0 * np.log10(ours.astype(float)) - 10.0 * np.log10(reference.astype(float))
    ).max()
    assert max_db < 1e-4, f"max |delta| = {max_db} dB"


@pytest.mark.skipif(
    not GA_USER_TABLE.is_file(),
    reason="MATLAB ground-truth GA run not present on this machine",
)
def test_mbs_column_matches_matlab_sampled_powers(default_backend):
    """End-to-end check of swap + transposed sampling + clamp + float32.

    ``power_FBS_3`` of the archived GA user table is the MATLAB-sampled MBS
    coverage column for exactly this world and exactly these 1000 users. Any
    error in the row swap, the ``P(x_index, y_index)`` indexing, the clamp or
    the ``single()`` cast shows up here immediately -- a naive "correct frame"
    read is off by up to ~60x on the clamped users.
    """
    table = pytest.importorskip("pandas").read_csv(GA_USER_TABLE)
    reference = table["power_FBS_3"].to_numpy(dtype=float)
    ours = default_backend._mbs_columns[0, 0]
    assert np.abs(ours / reference - 1.0).max() < 1e-5


@pytest.mark.skipif(
    not MATLAB_MBS_MAP.is_file(),
    reason="MATLAB ground-truth map cache not present on this machine",
)
def test_mbs_sampling_clamps_beyond_the_swapped_extent(default_backend):
    """``sample_nearest`` clamps ix against ``size(P, 1)``, which for an MBS
    map is ``height + 1``, not ``width + 1``. Every user with
    ``round(x) > height`` is therefore evaluated at ``x == height``."""
    grid = _load_matlab_mbs_map()  # (1501, 2001), MATLAB orientation
    ux = default_backend.user_positions[:, 0]
    uy = default_backend.user_positions[:, 1]
    column = default_backend._mbs_column(grid)

    row = np.floor(ux + 0.5).astype(int)
    col = np.floor(uy + 0.5).astype(int)
    beyond = row > DEFAULT_WORLD.height
    assert beyond.sum() > 0, "expected users past the swapped extent"
    # Clamped users read the last row; unclamped ones read their own row.
    assert np.array_equal(column[beyond], grid[grid.shape[0] - 1, col[beyond]])
    assert np.array_equal(column[~beyond], grid[row[~beyond], col[~beyond]])


def test_end_to_end_reproduces_archived_matlab_metrics(cache_dir):
    """The whole pipeline against a completed MATLAB run.

    ``ga_runs/run_2026-06-23_03-39-01_2-1-1`` is a dual-band GA solution on the
    default world: two FBSs pinned to the coverage band, the MBS capacity
    carrier enabled. Its ``experiment_config.json`` records the metrics MATLAB
    computed. Reproducing all five of them exercises every convention at once
    -- hex placement, the row swap, both map orientations, the clamp, the
    float32 MBS cast, slot interleaving, same-band interference, strict-'>'
    association, the rate definition and ``sum(tx_power)``.

    Deliberately *not* skipped when the archive is missing: the inputs and the
    expected values below are constants transcribed from it, and the body opens
    no file. This is the test that fails if the row swap, the FBS transpose,
    the unmasked ``sum(tx_power)`` or the noise floor is broken, so it has to
    run on every machine, not just the one holding the old repo.
    """
    band = BandConfig(mode="multi", fbs_band="agent", mbs_capacity="agent")
    backend = PyqdSinrBackend(DEFAULT_WORLD, band, cache_dir=cache_dir)
    # The archived best individual (best_individual.csv), two FBS blocks of
    # [x, y, z, power, power_status, band_flag] plus the MBS capacity gene.
    tx = np.array([
        [1161.00599763042, 98.5210527027749, 110.637769250947, 8.59128192067264],
        [363.705411830263, 332.864092659469, 129.7755970318, 9.44866195357879],
    ])
    result = backend.evaluate(
        tx, np.array([1.0, 1.0]),
        fbs_band_flags=np.array([0.0, 0.0]),
        mbs_capacity_genes=np.array([1.0]),
    )
    assert result.total_connected == 787
    assert result.fbs_connected == 99
    assert result.mbs_coverage_connected == 0
    assert result.mbs_capacity_connected == 688
    assert result.total_power == pytest.approx(18.039943874251435, abs=1e-12)
    assert result.avg_rate == pytest.approx(4.3356856897289529, rel=1e-6)


def test_disk_cache_round_trips(cache_dir, monkeypatch):
    """A second construction must reuse the .npz, not recompute the map."""
    backend = PyqdSinrBackend(
        SMALL_WORLD, BandConfig(mode="legacy"), cache_dir=cache_dir
    )
    spec = backend._mbs_spec(0, 2.0e9)
    assert (Path(cache_dir) / f"mbs_{spec.key}.npz").is_file()

    # "The file exists" and "two constructions agree" are both true of a
    # backend that writes the cache and then never reads it, which is exactly
    # the regression worth catching: the disk cache is the whole reason a
    # SubprocVecEnv worker starts cheaply. So make recomputation impossible.
    import ppo.pyqd_bridge as pyqd_bridge

    def _must_not_recompute(_spec):
        raise AssertionError("the .npz cache was not consulted")

    monkeypatch.setattr(pyqd_bridge, "compute_power_map", _must_not_recompute)
    again = PyqdSinrBackend(
        SMALL_WORLD, BandConfig(mode="legacy"), cache_dir=cache_dir
    )
    assert np.array_equal(backend._mbs_columns, again._mbs_columns)


def test_mbs_maps_keep_matlab_single_precision(cache_dir, small_backend):
    """``precompute_mbs_power_maps.m:88`` is ``map = single(map)``.

    The reference MBS physics really is float32 (~6e-8 relative, ~3e-7 dB), and
    every tolerance in this file is looser than that -- so without an explicit
    assertion the cast could be dropped and only a full archived-run replay
    would notice, at the 1e-7 level. The FBS path has no such cast.
    """
    spec = small_backend._mbs_spec(0, 2.0e9)
    assert load_or_compute_map(spec, None).dtype == np.float32
    assert load_or_compute_map(spec, cache_dir).dtype == np.float32
    # The sampled column is widened to float64 for the SINR arithmetic, but
    # every value in it must still be exactly a float32.
    column = small_backend._mbs_columns[0, 0]
    assert column.dtype == np.float64
    assert np.array_equal(column, column.astype(np.float32))


def test_failed_cache_write_leaves_no_orphan_temp_file(tmp_path, monkeypatch):
    """A half-written ``.mbs_<key>.<pid>.tmp.npz`` must not survive.

    It is a dotfile carrying the writer's pid, so under SubprocVecEnv a
    recurring write failure (full disk, quota, EIO) would otherwise accumulate
    one invisible orphan per worker per map, forever.
    """
    import ppo.pyqd_bridge as pyqd_bridge

    def _fail_midway(file, **_arrays):
        Path(file).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(pyqd_bridge.np, "savez_compressed", _fail_midway)
    world = WorldConfig(
        width=120.0, height=90.0, num_mbs=1, mbs_locations=[(60.0, 45.0)], num_users=5
    )
    backend = PyqdSinrBackend(world, BandConfig(mode="legacy"), cache_dir=tmp_path)

    # iterdir, not glob: the temp name starts with a dot.
    assert list(tmp_path.iterdir()) == []  # no entry, and no leftover temp
    # The map itself is still returned, so a broken cache stays a performance
    # problem rather than an error.
    assert np.all(backend._mbs_columns[0, 0] > 0.0)


def test_cache_key_covers_every_map_parameter():
    """Nudging any single field must change the key. A cache that returns a
    map computed for a different transmitter is worse than no cache."""
    base = MapSpec(
        scenario="3GPP_38.901_UMa_LOS", map_mode="quick",
        tx_x=1.0, tx_y=2.0, tx_z=3.0, tx_power=4.0, center_freq=2.0e9,
        x_min=0.0, x_max=100.0, y_min=0.0, y_max=80.0,
        sample_distance=1.0, rx_height=1.5,
    )
    import dataclasses

    for field in dataclasses.fields(base):
        value = getattr(base, field.name)
        bumped = "3GPP_38.901_UMa_NLOS" if field.name == "scenario" else (
            "phase" if field.name == "map_mode" else float(value) + 1.0
        )
        assert dataclasses.replace(base, **{field.name: bumped}).key != base.key, field.name


# --------------------------------------------------------------------------- #
# Evaluation semantics
# --------------------------------------------------------------------------- #
def test_determinism_within_and_across_backends(cache_dir, small_backend):
    """Same state -> identical metrics, including across a fresh world build
    (the user map is a fixed seed-0 draw, not a per-instance RNG)."""
    tx, status = _fbs(), np.array([1.0])
    first = small_backend.evaluate(tx, status)
    second = small_backend.evaluate(tx, status)
    assert first.as_metrics_dict() == second.as_metrics_dict()

    other = PyqdSinrBackend(SMALL_WORLD, BandConfig(mode="legacy"), cache_dir=cache_dir)
    assert other.evaluate(tx, status).as_metrics_dict() == first.as_metrics_dict()


def test_tier_invariant(small_backend, small_multi_backend):
    """fbs + coverage + capacity == total (SINREvaluation.m:114-118)."""
    legacy = small_backend.evaluate(_fbs(), np.array([1.0]))
    assert legacy.total_connected == (
        legacy.fbs_connected
        + legacy.mbs_coverage_connected
        + legacy.mbs_capacity_connected
    )
    assert legacy.mbs_connected == (
        legacy.mbs_coverage_connected + legacy.mbs_capacity_connected
    )
    # Legacy expands to one COVERAGE slot per site (ppo_sinr_eval.m:75), so the
    # capacity tier is structurally empty.
    assert legacy.mbs_capacity_connected == 0

    multi = small_multi_backend.evaluate(
        _fbs(), np.array([1.0]),
        fbs_band_flags=np.array([0.0]), mbs_capacity_genes=np.array([1.0]),
    )
    assert multi.total_connected == (
        multi.fbs_connected
        + multi.mbs_coverage_connected
        + multi.mbs_capacity_connected
    )


def test_inactive_fbs_contributes_nothing(small_backend):
    """power_status 0 -> an all-zero power column, never a served user."""
    on = small_backend.evaluate(_fbs(), np.array([1.0]))
    off = small_backend.evaluate(_fbs(), np.array([0.0]))
    assert off.fbs_connected == 0
    assert on.fbs_connected > 0
    # The MBS is unaffected by the FBS switching off other than losing its
    # interferer, so total connectivity cannot collapse.
    assert off.total_connected == off.mbs_connected


def test_total_power_includes_inactive_fbs(small_backend):
    """SINREvaluation.m:120-121 has the power_status mask COMMENTED OUT.

    This is the single easiest place to accidentally copy the analytic
    backend and silently break MATLAB parity, so it gets its own test.
    """
    tx = np.array([[200.0, 150.0, 60.0, 10.5], [400.0, 250.0, 70.0, 4.25]])
    both_on = small_backend.evaluate(tx, np.array([1.0, 1.0]))
    one_off = small_backend.evaluate(tx, np.array([1.0, 0.0]))
    all_off = small_backend.evaluate(tx, np.array([0.0, 0.0]))
    for result in (both_on, one_off, all_off):
        assert result.total_power == pytest.approx(14.75)


def test_avg_rate_is_nan_when_nothing_connects(cache_dir):
    """MATLAB returns NaN for avg_rate and 0.0 (not NaN) for sum_rate."""
    world = WorldConfig(
        width=400.0, height=300.0, num_mbs=1, mbs_locations=[(200.0, 150.0)],
        num_users=40, sinr_threshold=200.0,  # unreachable: nothing can connect
    )
    backend = PyqdSinrBackend(world, BandConfig(mode="legacy"), cache_dir=cache_dir)
    r = backend.evaluate(np.array([[100.0, 100.0, 50.0, 10.0]]), np.array([0.0]))
    assert r.total_connected == 0
    assert np.isnan(r.avg_rate)
    assert r.sum_rate == 0.0


def test_multiband_capacity_slot_gating(small_multi_backend):
    tx, status = _fbs(), np.array([1.0])
    no_cap = small_multi_backend.evaluate(
        tx, status, fbs_band_flags=np.array([0.0]), mbs_capacity_genes=np.array([0.0])
    )
    with_cap = small_multi_backend.evaluate(
        tx, status, fbs_band_flags=np.array([0.0]), mbs_capacity_genes=np.array([1.0])
    )
    assert no_cap.mbs_capacity_connected == 0
    assert with_cap.mbs_capacity_connected > 0
    assert with_cap.total_connected >= no_cap.total_connected


def test_power_status_uses_matlab_truthiness(small_backend):
    """SINREvaluation.m:208 is `if power_status(i)` — any non-zero activates,
    unlike the >= 0.5 tests used for the slot/band flags."""
    tx = _fbs()
    faint = small_backend.evaluate(tx, np.array([0.25]))
    full = small_backend.evaluate(tx, np.array([1.0]))
    assert faint.as_metrics_dict() == full.as_metrics_dict()


def test_evaluate_rejects_mismatched_power_status(small_backend):
    with pytest.raises(ValueError, match="one entry per FBS"):
        small_backend.evaluate(_fbs(), np.array([1.0, 1.0]))


def test_association_ties_resolve_to_the_earlier_column(cache_dir):
    """SINREvaluation.m:278 compares with a STRICT '>', so an exact SINR tie is
    won by the column visited first -- and FBS columns are visited before MBS
    slots. Physics will not hand us an exact tie, so build one: give the MBS
    capacity slot a bit-for-bit copy of the FBS's own power column and put the
    two on different bands, so neither interferes with the other and both see
    ``P / (0 + noise)`` from identical operands.

    Flipping the comparison to '>=' moves all 120 users from the FBS tier to
    the capacity tier, which is exactly the silent tier-attribution change the
    strictness exists to prevent.
    """
    band = BandConfig(mode="multi", fbs_band="agent", mbs_capacity="agent")
    backend = PyqdSinrBackend(SMALL_WORLD, band, cache_dir=cache_dir)
    x, y, z, power = 300.0, 200.0, 60.0, 10.0
    shared = backend._fbs_column(backend._fbs_spec(x, y, z, power, backend.band_freqs[0]))
    backend._mbs_columns[:] = 0.0
    backend._mbs_columns[1, 0] = shared  # capacity band, site 0

    result = backend.evaluate(
        np.array([[x, y, z, power]]), np.array([1.0]),
        fbs_band_flags=np.array([0.0]),      # FBS on the coverage band
        mbs_capacity_genes=np.array([1.0]),  # capacity slot on
    )
    assert result.total_connected == SMALL_WORLD.num_users
    assert result.fbs_connected == result.total_connected
    assert result.mbs_connected == 0


def test_empty_band_arrays_mean_all_zeros(small_multi_backend):
    """ppo_sinr_eval.m:36/38 substitute zeros for an EMPTY band argument, so
    ``[]`` and ``None`` must agree. A wrong-length vector is a caller bug and
    should say so, not surface as an IndexError from inside the SINR loop."""
    tx, status = _fbs(), np.array([1.0])
    default = small_multi_backend.evaluate(tx, status).as_metrics_dict()
    empty = small_multi_backend.evaluate(
        tx, status, fbs_band_flags=np.array([]), mbs_capacity_genes=np.array([])
    ).as_metrics_dict()
    explicit = small_multi_backend.evaluate(
        tx, status, fbs_band_flags=np.zeros(1), mbs_capacity_genes=np.zeros(1)
    ).as_metrics_dict()
    assert default == empty == explicit

    with pytest.raises(ValueError, match="fbs_band_flags"):
        small_multi_backend.evaluate(tx, status, fbs_band_flags=np.zeros(3))
    with pytest.raises(ValueError, match="mbs_capacity_genes"):
        small_multi_backend.evaluate(tx, status, mbs_capacity_genes=np.zeros(2))


# --------------------------------------------------------------------------- #
# Multiple MBS sites
#
# Half the shipped scenario codes (the ``*-2-*`` family, SCENARIOS[2]) place two
# MBS sites. With a single site nothing below is observable: ``[band_row,
# site]`` and ``[band_row, 0]`` are the same expression, and "interleaved per
# site" and "grouped by band" are the same list.
# --------------------------------------------------------------------------- #
def test_each_site_and_band_gets_its_own_map(two_site_backend):
    """``_mbs_columns[band_row, site]`` is that site's map on that carrier."""
    backend = two_site_backend
    assert backend.num_mbs == 2 and backend.n_bands == 2
    assert backend._mbs_columns.shape == (2, 2, TWO_SITE_WORLD.num_users)

    for band_row, freq in enumerate(backend.band_freqs):
        for site in range(2):
            spec = backend._mbs_spec(site, freq)
            # The row swap is applied per site, not once for the whole world.
            assert spec.tx_x == pytest.approx(backend.mbs_y[site])
            assert spec.tx_y == pytest.approx(backend.mbs_x[site])
            assert spec.center_freq == freq
            expected = sample_nearest(
                compute_power_map(spec).astype(np.float32),
                backend._user_x, backend._user_y,
            ).astype(float)
            assert np.array_equal(backend._mbs_columns[band_row, site], expected)

    # ... and the four are genuinely four different maps, so the assertions
    # above cannot be satisfied by one map stored four times.
    columns = backend._mbs_columns.reshape(4, -1)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.array_equal(columns[i], columns[j]), (i, j)


def test_build_slots_interleaves_per_site():
    """ppo_sinr_eval.m:52-57 emits [cov(1), cap(1), cov(2), cap(2)] -- per
    site, NOT grouped by band. Column order decides ties, so it is physics."""
    slots = build_slots(2, True, np.array([1.0, 0.0]))
    assert [(s.site, s.band_row, s.active, s.band_id) for s in slots] == [
        (0, 0, True, BAND_COVERAGE),
        (0, 1, True, BAND_CAPACITY),   # gene 1.0 -> site 0's capacity slot on
        (1, 0, True, BAND_COVERAGE),
        (1, 1, False, BAND_CAPACITY),  # gene 0.0 -> site 1's capacity slot off
    ]
    # Legacy: one always-on coverage slot per site (ppo_sinr_eval.m:75).
    legacy = build_slots(2, False, np.array([1.0, 1.0]))
    assert [(s.site, s.band_row, s.active, s.band_id) for s in legacy] == [
        (0, 0, True, BAND_COVERAGE),
        (1, 0, True, BAND_COVERAGE),
    ]


def test_evaluate_reads_each_slots_own_site_column(cache_dir):
    """Slot k's power column must come from slot k's OWN site.

    Real physics cannot pin this on its own -- two sites produce two plausible
    columns and any of them "looks right" -- so inject a recognisable, disjoint
    user set per (band, site) and read back which ones connected. Reading site
    0 for every slot gives 10 + 5; reading site 1 for every slot gives 20 + 25.
    """
    band = BandConfig(mode="multi", fbs_band="agent", mbs_capacity="agent")
    backend = PyqdSinrBackend(TWO_SITE_WORLD, band, cache_dir=cache_dir)
    backend._mbs_columns[:] = 0.0            # nothing is audible by default
    backend._mbs_columns[0, 0, 0:10] = 1.0   # coverage band, site 0
    backend._mbs_columns[1, 0, 10:15] = 1.0  # capacity band, site 0
    backend._mbs_columns[0, 1, 15:35] = 1.0  # coverage band, site 1
    backend._mbs_columns[1, 1, 35:60] = 1.0  # capacity band, site 1

    dark_fbs = (np.array([[10.0, 10.0, 50.0, 5.0]]), np.array([0.0]))
    result = backend.evaluate(
        *dark_fbs,
        fbs_band_flags=np.array([0.0]),
        mbs_capacity_genes=np.array([1.0, 1.0]),
    )
    assert result.mbs_coverage_connected == 30   # 10 from site 0 + 20 from site 1
    assert result.mbs_capacity_connected == 30   #  5 from site 0 + 25 from site 1
    assert result.total_connected == 60
    assert result.fbs_connected == 0

    # Legacy expands to one coverage slot per site and no capacity slots.
    legacy = PyqdSinrBackend(TWO_SITE_WORLD, BandConfig(mode="legacy"), cache_dir=cache_dir)
    legacy._mbs_columns[:] = 0.0
    legacy._mbs_columns[0, 0, 0:10] = 1.0
    legacy._mbs_columns[0, 1, 15:35] = 1.0
    legacy_result = legacy.evaluate(*dark_fbs)
    assert legacy_result.mbs_coverage_connected == 30
    assert legacy_result.mbs_capacity_connected == 0


def test_capacity_gating_is_per_site(two_site_backend):
    """Enabling site 0's capacity carrier is not the same as enabling site 1's.

    Pure physics, no injected columns: the two sites cover different users, so
    a backend that sampled one site's map for both slots would return the same
    metrics for both genes.
    """
    tx, status = _fbs(), np.array([1.0])
    flags = np.array([0.0])

    def run(genes):
        return two_site_backend.evaluate(
            tx, status, fbs_band_flags=flags,
            mbs_capacity_genes=np.array(genes, dtype=float),
        )

    neither, first, second, both = run([0, 0]), run([1, 0]), run([0, 1]), run([1, 1])
    assert neither.mbs_capacity_connected == 0
    assert first.mbs_capacity_connected > 0 and second.mbs_capacity_connected > 0
    assert both.mbs_capacity_connected > 0
    assert first.as_metrics_dict() != second.as_metrics_dict()
    for result in (neither, first, second, both):
        assert result.total_connected == (
            result.fbs_connected
            + result.mbs_coverage_connected
            + result.mbs_capacity_connected
        )


def _backend_without_maps(pyqd_bridge, world, band) -> PyqdSinrBackend:
    """A backend whose MBS maps are stubbed out -- for spec-only assertions."""
    real = pyqd_bridge.load_or_compute_map
    pyqd_bridge.load_or_compute_map = lambda spec, cache_dir: np.zeros((2, 2), np.float32)
    try:
        return PyqdSinrBackend(world, band, cache_dir=None)
    finally:
        pyqd_bridge.load_or_compute_map = real


@pytest.mark.skipif(
    not all(path.is_file() for path in SCENARIO_2_MBS_MAPS.values()),
    reason="MATLAB ground-truth scenario-2 map caches not present on this machine",
)
def test_scenario_two_mbs_maps_match_the_matlab_caches():
    """Both sites and both carriers of SCENARIOS[2], against MATLAB.

    The only MATLAB ground truth for a *second* MBS site. A full 4001x3001 pyqd
    map costs ~2.4 s and a ~3.4 GB transient, so instead of four of those this
    evaluates the propagation model at a few hundred grid nodes --
    ``sample_power_at`` is bit-identical to reading the full grid, which
    ``test_fast_sampling_is_bit_identical_to_the_full_grid`` pins -- and
    compares those nodes against MATLAB's.

    Sensitivity, measured out of band on these same nodes: comparing a site
    against the *other* site's map gives 56 dB, the wrong carrier 2.3 dB, and
    an unswapped transmitter 35 dB, versus the ~2e-6 dB asserted here.
    """
    h5py = pytest.importorskip("h5py")
    import ppo.pyqd_bridge as pyqd_bridge

    world = WorldConfig(**SCENARIOS[2])
    assert world.num_mbs == 2  # the whole point of this test
    band = BandConfig(mode="multi", fbs_band="agent", mbs_capacity="agent")

    # Only the four MapSpecs are under test; building their maps for real would
    # cost ~10 s and >3 GB, and the map maths is already covered on the default
    # world by test_mbs_map_matches_the_matlab_cache.
    backend = _backend_without_maps(pyqd_bridge, world, band)

    for (site, band_row), path in SCENARIO_2_MBS_MAPS.items():
        spec = backend._mbs_spec(site, backend.band_freqs[band_row])
        with h5py.File(path, "r") as handle:
            grid = np.array(handle["map"]).T          # MATLAB (n_y, n_x)
            x_coords = np.array(handle["x_coords"]).ravel()
            y_coords = np.array(handle["y_coords"]).ravel()
            meta = {
                name: float(np.array(handle["meta"][name]).ravel()[0])
                for name in ("x", "y", "z", "power", "ue_height", "band")
            }
            scenario = "".join(
                chr(code) for code in np.array(handle["meta"]["scenario"]).ravel()
            )

        # MATLAB's own record of what it computed: the SWAPPED transmitter, the
        # band index (1 = coverage, 2 = capacity) and the scenario.
        assert (spec.tx_x, spec.tx_y) == pytest.approx((meta["x"], meta["y"]))
        assert meta["band"] == band_row + 1
        assert spec.scenario == scenario
        assert spec.tx_z == pytest.approx(meta["z"])
        assert spec.tx_power == pytest.approx(meta["power"])
        assert spec.rx_height == pytest.approx(meta["ue_height"])
        assert grid.shape == (y_coords.size, x_coords.size) == (3001, 4001)

        # Coprime-ish strides so the sample walks the whole map rather than one
        # symmetry line.
        ix = np.arange(0, x_coords.size, 337)
        iy = np.arange(0, y_coords.size, 251)
        mesh_x, mesh_y = np.meshgrid(x_coords[ix], y_coords[iy], indexing="xy")
        ours = sample_power_at(spec, mesh_x.ravel(), mesh_y.ravel())
        reference = grid[np.ix_(iy, ix)].ravel().astype(float)
        max_db = np.abs(10.0 * np.log10(ours) - 10.0 * np.log10(reference)).max()
        assert max_db < 1e-4, f"site {site} band {band_row}: max |delta| = {max_db} dB"


# --------------------------------------------------------------------------- #
# The opt-in fast sampler
# --------------------------------------------------------------------------- #
def test_fast_sampling_is_bit_identical_to_the_full_grid(cache_dir):
    """The fast path evaluates the LOS model at exactly the grid nodes the
    faithful path reads, so it must agree to the last bit — not merely to
    some tolerance. If this ever loosens, the fast path is wrong."""
    faithful = PyqdSinrBackend(
        SMALL_WORLD, BandConfig(mode="legacy"), cache_dir=cache_dir,
        column_cache_size=0,
    )
    fast = PyqdSinrBackend(
        SMALL_WORLD, BandConfig(mode="legacy"), cache_dir=cache_dir,
        fast_sampling=True, column_cache_size=0,
    )
    spec = faithful._fbs_spec(210.0, 160.0, 65.0, 9.0, 2.0e9)

    grid = compute_power_map(spec)
    reference = sample_nearest(grid.T, faithful._user_x, faithful._user_y)
    gx, gy = fast._grid_nodes()
    direct = sample_power_at(spec, gx, gy)
    assert np.array_equal(direct, reference)

    tx, status = _fbs(), np.array([1.0])
    assert (
        fast.evaluate(tx, status).as_metrics_dict()
        == faithful.evaluate(tx, status).as_metrics_dict()
    )


def test_fast_sampling_is_faster(cache_dir):
    """Not a benchmark — just a guard that the fast path is not accidentally
    computing the whole grid anyway. The real margin is ~190x."""
    world = WorldConfig(
        width=1200.0, height=900.0, num_mbs=1, mbs_locations=[(600.0, 450.0)],
        num_users=200,
    )
    faithful = PyqdSinrBackend(world, BandConfig(), cache_dir=cache_dir, column_cache_size=0)
    fast = PyqdSinrBackend(
        world, BandConfig(), cache_dir=cache_dir, fast_sampling=True, column_cache_size=0
    )
    tx, status = np.array([[500.0, 400.0, 60.0, 10.0]]), np.array([1.0])

    t0 = time.perf_counter(); faithful.evaluate(tx, status); slow = time.perf_counter() - t0
    t0 = time.perf_counter(); fast.evaluate(tx, status); quick = time.perf_counter() - t0
    assert quick < slow / 3.0, f"faithful {slow:.4f}s vs fast {quick:.4f}s"


def test_fast_sampling_invariants_hold_on_a_real_spec(small_backend):
    """The guard must be silent on the configuration everything actually uses.

    A guard that fires on the happy path would just get deleted.
    """
    spec = small_backend._fbs_spec(210.0, 160.0, 65.0, 9.0, 2.0e9)
    assert check_fast_sampling_invariants(
        spec, SMALL_WORLD.width, SMALL_WORLD.height
    ) is None


def test_fast_sampling_rejects_a_changed_grid_step(cache_dir, monkeypatch):
    """A future change of grid resolution must fail loudly, not silently.

    ``SAMPLE_DISTANCE`` is documented as load-bearing all over this module but
    nothing used to enforce it, so bumping it would have kept both paths
    running and quietly moved the physics away from every archived map and
    logged metric. Construction and evaluation both have to refuse.
    """
    import ppo.pyqd_bridge as pyqd_bridge

    tx, status = _fbs(), np.array([1.0])
    fast = PyqdSinrBackend(
        SMALL_WORLD, BandConfig(), cache_dir=cache_dir,
        fast_sampling=True, column_cache_size=0,
    )
    fast.evaluate(tx, status)  # fine before the constant moves

    monkeypatch.setattr(pyqd_bridge, "SAMPLE_DISTANCE", 2.0)

    with pytest.raises(ValueError, match="SAMPLE_DISTANCE"):
        fast.evaluate(tx, status)
    with pytest.raises(ValueError, match="SAMPLE_DISTANCE"):
        PyqdSinrBackend(
            SMALL_WORLD, BandConfig(), cache_dir=cache_dir, fast_sampling=True
        )

    # The faithful path reads the map it just computed, so it is not affected
    # by the constant moving -- only the fast path's reconstruction is.
    PyqdSinrBackend(
        SMALL_WORLD, BandConfig(), cache_dir=cache_dir, column_cache_size=0
    ).evaluate(tx, status)


@pytest.mark.parametrize(
    "override, expected",
    [
        ({"x_min": 10.0}, "grid origin"),
        ({"y_min": -1.0}, "grid origin"),
        ({"x_max": SMALL_WORLD.width + 5.0}, "map extent"),
        ({"y_max": SMALL_WORLD.height - 5.0}, "map extent"),
        ({"sample_distance": 0.5}, "sample_distance"),
    ],
)
def test_fast_sampling_rejects_a_grid_it_cannot_reconstruct(
    cache_dir, override, expected
):
    """``_grid_nodes`` ignores the spec's origin, extent and step.

    Each override below leaves the *map* well-formed but makes the node
    coordinates the fast path reconstructs differ from the ones
    ``sample_nearest`` would read, which is exactly the silent-wrong-answer
    case the guard exists for.
    """
    fast = PyqdSinrBackend(
        SMALL_WORLD, BandConfig(), cache_dir=cache_dir,
        fast_sampling=True, column_cache_size=0,
    )
    spec = dataclasses.replace(
        fast._fbs_spec(210.0, 160.0, 65.0, 9.0, 2.0e9), **override
    )
    with pytest.raises(ValueError, match=expected):
        fast._fbs_column(spec)


# --------------------------------------------------------------------------- #
# Environment integration
# --------------------------------------------------------------------------- #
def test_env_runs_on_the_pyqd_backend(small_backend):
    """The env drives the backend positionally for the first two arguments and
    by keyword for the rest; make sure that call shape actually works."""
    from ppo.config import EnvConfig
    from ppo.env import FlyingBaseStationEnv

    cfg = EnvConfig(
        num_fbs=1, world=SMALL_WORLD, band=BandConfig(mode="legacy"),
        max_episode_steps=2, collect_user_positions=True,
    )
    env = FlyingBaseStationEnv(cfg, backend=small_backend)
    obs, _info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    _obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert np.isfinite(reward)
    assert info["total_connected"] == (
        info["fbs_connected"]
        + info["mbs_coverage_connected"]
        + info["mbs_capacity_connected"]
    )
    assert info["user_positions"].shape == (SMALL_WORLD.num_users, 2)
    # gymnasium requires plain bools, and the two flags are mutually exclusive
    # (`terminated` short-circuits the episode before the step budget runs out).
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert not (terminated and truncated)
