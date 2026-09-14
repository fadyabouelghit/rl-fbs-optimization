"""Adversarial hunt for a state where ``pyqd`` (faithful) and ``pyqd-fast`` diverge.

The replay/equivalence work so far samples states that *occurred* during
training. This script does the opposite: it constructs states chosen to break
the equivalence argument, and checks them with BITWISE equality.

Why the FBS power column is the right observable
------------------------------------------------
``PyqdSinrBackend.evaluate`` is shared verbatim by both paths. The MBS columns
are shared (``load_or_compute_map`` -> ``_mbs_column``, no ``fast_sampling``
branch). The association, the rates, the counts, ``total_power`` -- all shared.
The *only* branch on ``self.fast_sampling`` in the whole backend is inside
``_fbs_column``:

    fast      : sample_power_at(spec, *self._grid_nodes())
    faithful  : sample_nearest(compute_power_map(spec).T, user_x, user_y)

So comparing the two 1000-element float64 columns bit-for-bit is *strictly
stronger* than comparing ``SinrResult``: any divergence at all must show up
there first, before the aggregation quantises it away. We do both anyway --
columns for every constructed state, full ``evaluate()`` for a subset -- so the
end-to-end claim is measured and not merely deduced.

The assumption chain under attack (see the sections below):

  A. round(user_coord) lands exactly on a grid node (integrality / the .5 case)
  B. the grid covers every user; the two clamps agree
  C. _grid_nodes' n_x/n_y match the vectors power_map actually returns
  D. the numerics are bit-identical, not merely close (batch size, ordering,
     port collapse, the tx_power scale factor, shared mutable state)
  E. the gene-bound extremes and the degenerate geometries

Run:  venv/bin/python comparison/adversarial_fast_vs_faithful.py
"""

from __future__ import annotations

import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ppo.config import BandConfig, WorldConfig  # noqa: E402
from ppo.matlab_rng import matlab_user_positions  # noqa: E402
from ppo.pyqd_bridge import (  # noqa: E402
    FBS_UE_HEIGHT,
    SAMPLE_DISTANCE,
    PyqdSinrBackend,
    _rx_antenna,
    _scenario_config,
    _tx_antenna,
    compute_power_map,
    sample_nearest,
    sample_power_at,
)

FAILURES: list[str] = []
N_STATES = 0
N_COLUMN_CMP = 0
N_EVAL_CMP = 0


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  *** DIVERGENCE *** {msg}")


def bits(a: np.ndarray) -> np.ndarray:
    """Raw IEEE-754 bit patterns, so -0.0 != 0.0 and NaN payloads compare."""
    return np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)


def cmp_columns(tag: str, fast_r, slow_r) -> bool:
    """Bitwise column comparison of two ``safe()`` results. True when identical."""
    global N_COLUMN_CMP
    N_COLUMN_CMP += 1
    ok_f, fast = fast_r
    ok_s, slow = slow_r
    if not ok_f or not ok_s:
        if ok_f != ok_s:
            f_desc = f"RAISED {fast}" if not ok_f else "returned a column"
            s_desc = f"RAISED {slow}" if not ok_s else "returned a column"
            extra = ""
            if ok_s:
                extra = (f" | faithful column: min={np.min(slow)!r} "
                         f"max={np.max(slow)!r} n_inf={int(np.isinf(slow).sum())}")
            if ok_f:
                extra = (f" | fast column: min={np.min(fast)!r} "
                         f"max={np.max(fast)!r} n_inf={int(np.isinf(fast).sum())}")
            fail(f"{tag}: fast {f_desc} / faithful {s_desc}{extra}")
            return False
        if fast != slow:
            fail(f"{tag}: both raised, different: fast={fast!r} faithful={slow!r}")
            return False
        print(f"  (both paths raise identically for {tag}: {fast})")
        return True
    if fast.shape != slow.shape:
        fail(f"{tag}: shape {fast.shape} vs {slow.shape}")
        return False
    bf, bs = bits(fast), bits(slow)
    if np.array_equal(bf, bs):
        return True
    bad = np.nonzero(bf != bs)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        d_db = np.abs(10 * np.log10(fast[bad]) - 10 * np.log10(slow[bad]))
    i = int(bad[0])
    fail(
        f"{tag}: {bad.size}/{fast.size} entries differ; "
        f"max|dB delta|={np.nanmax(d_db):.6e}; "
        f"first at user {i}: fast={fast[i]!r} slow={slow[i]!r}"
    )
    return False


def _eq(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b  # exact: atol=0, rtol=0
    return a == b


def cmp_results(tag: str, rf_r, rs_r) -> bool:
    global N_EVAL_CMP
    N_EVAL_CMP += 1
    ok_f, rf = rf_r
    ok_s, rs = rs_r
    if not ok_f or not ok_s:
        if ok_f != ok_s:
            f_desc = f"RAISED {rf}" if not ok_f else f"returned {rf}"
            s_desc = f"RAISED {rs}" if not ok_s else f"returned {rs}"
            fail(f"{tag}: fast {f_desc} / faithful {s_desc}")
            return False
        if rf != rs:
            fail(f"{tag}: both raised, different: fast={rf!r} faithful={rs!r}")
            return False
        return True
    fields = (
        "total_connected", "fbs_connected", "mbs_connected",
        "mbs_coverage_connected", "mbs_capacity_connected",
        "total_power", "avg_rate", "sum_rate",
    )
    ok = True
    for f in fields:
        a, b = getattr(rf, f), getattr(rs, f)
        if not _eq(a, b):
            fail(f"{tag}: SinrResult.{f} fast={a!r} faithful={b!r}")
            ok = False
    return ok


def safe(fn, *a, **kw):
    """Run ``fn``, returning ``(ok, value_or_exception_repr)``.

    A path that *raises* where the other returns is a divergence too -- arguably
    the worst kind -- so exceptions are captured and compared, not propagated.
    """
    try:
        return True, fn(*a, **kw)
    except Exception as exc:  # noqa: BLE001 - the exception IS the observable
        return False, f"{type(exc).__name__}: {exc}"


def fbs_columns(backend_fast, backend_slow, x, y, z, p, freq):
    """The two ``_fbs_column`` implementations, on the same MapSpec."""
    spec = backend_fast._fbs_spec(x, y, z, p, freq)
    gx, gy = backend_fast._grid_nodes()

    def _fast():
        return np.asarray(sample_power_at(spec, gx, gy), dtype=float)

    def _slow():
        grid = compute_power_map(spec)
        return np.asarray(
            sample_nearest(grid.T, backend_slow._user_x, backend_slow._user_y),
            dtype=float,
        )

    return safe(_fast), safe(_slow)


# ===========================================================================
# A. "round(user_coord) lands exactly on a grid node"
# ===========================================================================
def section_A() -> None:
    print("\n" + "=" * 78)
    print("A. Can a user coordinate ever be exactly N + 0.5 (round-half tie)?")
    print("=" * 78)
    worlds = [
        (0, 2000, 0, 1500, 1000, 0),
        (0, 2000, 0, 1500, 1000, 7),
        (0, 2000.5, 0, 1500.5, 5000, 0),      # non-integer extent
        (0, 1999.9999999999998, 0, 7.0, 4000, 3),
        (0, 3, 0, 3, 20000, 11),              # tiny span, many draws
        (0, 100000, 0, 99999, 50000, 5),      # wide span
    ]
    worst = 0.0
    for xmin, xmax, ymin, ymax, n, seed in worlds:
        p = matlab_user_positions(xmin, xmax, ymin, ymax, n, seed=seed)
        pf = p.astype(float)
        frac = pf - np.floor(pf)
        worst = max(worst, float(np.abs(frac).max()))
        n_half = int(np.count_nonzero(frac == 0.5))
        integral = bool(np.array_equal(pf, np.floor(pf)))
        # numpy round-half-even vs MATLAB round-half-away: agree iff no .5
        same_rounding = np.array_equal(np.round(pf), np.floor(pf + 0.5))
        print(
            f"  world x<=({xmax}) y<=({ymax}) n={n} seed={seed}: dtype={p.dtype}, "
            f"integral={integral}, exactly-.5 count={n_half}, "
            f"np.round==floor(v+.5): {same_rounding}, "
            f"x in [{pf[:,0].min():g},{pf[:,0].max():g}] "
            f"y in [{pf[:,1].min():g},{pf[:,1].max():g}]"
        )
        if not integral or n_half or not same_rounding:
            fail(f"non-integer user coordinate in world {(xmax, ymax, n, seed)}")
    print(f"  max fractional part over all {sum(w[4] for w in worlds)} draws: {worst}")
    print(
        "  MECHANISM: matlab_randi returns int64 (floor(rand*span)+lo); "
        "matlab_user_positions -> .astype(float) exactly. A half-integer user\n"
        "  coordinate is therefore unrepresentable by construction, so the\n"
        "  numpy-round-half-even / MATLAB-round-half-away split cannot fire."
    )


# ===========================================================================
# C. n_x / n_y arithmetic vs the vectors power_map actually returns
# ===========================================================================
def section_C_grid(world: WorldConfig, band: BandConfig, label: str, backend=None):
    """Compare _grid_nodes() against power_map's real x_coords/y_coords."""
    from pyqd_channel.layout import power_map

    b = backend or PyqdSinrBackend(world, band, fast_sampling=True)
    spec = b._fbs_spec(world.width * 0.37, world.height * 0.61, 40.0, 9.0, 2.0e9)
    maps, x_coords, y_coords = power_map(
        spec.scenario, _tx_antenna(spec.center_freq),
        np.array([spec.tx_x, spec.tx_y, spec.tx_z]), spec.center_freq,
        spec.x_min, spec.x_max, spec.y_min, spec.y_max,
        sample_distance=spec.sample_distance, rx_height=spec.rx_height,
        tx_power=spec.tx_power, usage=spec.map_mode,
    )
    grid = maps[0].sum(axis=(2, 3))                      # (n_y, n_x)
    n_x_claim = int(np.floor(world.width / SAMPLE_DISTANCE)) + 1
    n_y_claim = int(np.floor(world.height / SAMPLE_DISTANCE)) + 1
    ok = (len(x_coords) == n_x_claim) and (len(y_coords) == n_y_claim)
    print(
        f"  [{label}] W={world.width!r} H={world.height!r}: "
        f"_grid_nodes n_x,n_y=({n_x_claim},{n_y_claim}) "
        f"power_map len(x),len(y)=({len(x_coords)},{len(y_coords)}) "
        f"grid.shape={grid.shape} -> {'MATCH' if ok else 'MISMATCH'}"
    )
    if not ok:
        fail(f"[{label}] n_x/n_y mismatch")

    ux, uy = b._user_x, b._user_y
    gx, gy = b._grid_nodes()
    # The exact index sample_nearest would use on the transposed map.
    nx_t, ny_t = grid.T.shape
    ix = np.clip(np.floor(ux + 0.5).astype(np.int64) + 1, 1, nx_t)
    iy = np.clip(np.floor(uy + 0.5).astype(np.int64) + 1, 1, ny_t)
    coord_ok = np.array_equal(bits(x_coords[ix - 1]), bits(gx)) and np.array_equal(
        bits(y_coords[iy - 1]), bits(gy)
    )
    clamp_x = int(np.count_nonzero(np.floor(ux + 0.5).astype(np.int64) + 1 != ix))
    clamp_y = int(np.count_nonzero(np.floor(uy + 0.5).astype(np.int64) + 1 != iy))
    print(
        f"           grid-node coords == power_map coords at every user: {coord_ok}; "
        f"clamp fired on x for {clamp_x} users, on y for {clamp_y} users; "
        f"user extent x[{ux.min():g},{ux.max():g}] y[{uy.min():g},{uy.max():g}] "
        f"vs grid x[{x_coords[0]:g},{x_coords[-1]:g}] y[{y_coords[0]:g},{y_coords[-1]:g}]"
    )
    if not coord_ok:
        fail(f"[{label}] _grid_nodes coordinates differ from power_map's grid")
    # And the actual values.
    col_fast = safe(lambda: np.asarray(sample_power_at(spec, gx, gy), dtype=float))
    col_slow = safe(lambda: np.asarray(sample_nearest(grid.T, ux, uy), dtype=float))
    cmp_columns(f"[{label}] grid-consistency column", col_fast, col_slow)
    return b


# ===========================================================================
# D. numerics: the tx_power scale, batch size, ordering, shared mutable state
# ===========================================================================
def section_D_scale() -> None:
    print("\n" + "=" * 78)
    print("D1. tx_power scale factor: power_map uses np.float64, sample_power_at")
    print("    uses a Python float. 10**(0.1*p) -- same bits?")
    print("=" * 78)
    def py_scale(pv):           # sample_power_at (fast path)
        return 10 ** (0.1 * pv)

    def np_scale(pv):           # power_map (faithful path); tx_power[i] is np.float64
        return float(10 ** (0.1 * np.full(1, pv, dtype=np.float64)[0]))

    vals = np.concatenate([
        np.linspace(-4000.0, 4000.0, 80001),
        np.array([7.0, 10.5, 20.0, 0.0, -0.0, 3.5, 8.123456789012345,
                  1e-300, 1e300, 308.0, 309.0, -308.0, -309.0,
                  3082.0, 3082.5, 3082.54, 3082.547, 3082.5471,
                  3083.0, 3090.0, -3090.0, -3230.0, -3240.0]),
    ])
    n_bit_diff = 0
    n_exc_diff = 0
    first_bit = None
    first_exc = None
    exc_lo = None
    for v in vals:
        pv = float(v)
        ok_a, a = safe(py_scale, pv)
        ok_b, b = safe(np_scale, pv)
        if ok_a != ok_b:
            n_exc_diff += 1
            if first_exc is None:
                first_exc = (pv, (ok_a, a), (ok_b, b))
            if ok_b and not ok_a and (exc_lo is None or pv < exc_lo):
                exc_lo = pv
            continue
        if not ok_a:
            continue
        if bits(np.array([a]))[0] != bits(np.array([b]))[0]:
            n_bit_diff += 1
            if first_bit is None:
                first_bit = (pv, a, b)
    print(f"  scanned {vals.size} tx_power values in [-4000, 4000]")
    print(f"  values where the two produce different BITS  : {n_bit_diff}")
    print(f"  values where one RAISES and the other does not: {n_exc_diff}")
    if n_bit_diff:
        fail(f"D1 tx_power scale bits differ, first at {first_bit}")
    if n_exc_diff:
        fail(
            "D1 tx_power scale: fast path `10 ** (0.1 * python_float)` raises "
            "OverflowError where the faithful path's `10 ** np.float64` returns "
            f"inf. First offender p={first_exc[0]!r}: fast={first_exc[1]!r} "
            f"faithful={first_exc[2]!r}"
        )
        # bisect the exact threshold
        lo, hi = 3080.0, 3086.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if safe(py_scale, mid)[0]:
                lo = mid
            else:
                hi = mid
        print(f"  -> fast path raises for tx_power > ~{lo!r} dBm "
              f"(10**(0.1*p) > DBL_MAX); faithful returns inf there.")
        print(f"     10**(0.1*{lo!r}) = {py_scale(lo)!r};  "
              f"faithful at {hi!r} = {np_scale(hi)!r}")
    if not n_bit_diff and not n_exc_diff:
        print("  -> the Python-float / np.float64 split in the scale factor is a no-op.")


def section_D_batch() -> None:
    print("\n" + "=" * 78)
    print("D2. Is get_los_coeff batch-size / ordering dependent? (SIMD tails,")
    print("    lane alignment, pairwise reductions)")
    print("=" * 78)
    from ppo.pyqd_bridge import MapSpec

    spec = MapSpec(
        scenario="3GPP_38.901_UMa_LOS", map_mode="quick",
        tx_x=997.0, tx_y=613.0, tx_z=23.0, tx_power=9.25,
        center_freq=2.0e9, x_min=0.0, x_max=2000.0, y_min=0.0, y_max=1500.0,
        sample_distance=1.0, rx_height=1.5,
    )
    rng = np.random.default_rng(0)
    probe_x = rng.integers(0, 2001, 64).astype(float)
    probe_y = rng.integers(0, 1501, 64).astype(float)
    ref = sample_power_at(spec, probe_x, probe_y)

    # (a) embed the probe inside padded batches of many different lengths
    for pad in (0, 1, 2, 3, 7, 8, 15, 16, 31, 63, 100, 936, 1000, 4095, 65537, 300000):
        fx = np.concatenate([probe_x, rng.integers(0, 2001, pad).astype(float)])
        fy = np.concatenate([probe_y, rng.integers(0, 1501, pad).astype(float)])
        got = sample_power_at(spec, fx, fy)[: probe_x.size]
        if not np.array_equal(bits(got), bits(ref)):
            fail(f"D2 batch length {probe_x.size + pad}: probe values changed")
    print(f"  batch lengths {probe_x.size}..{probe_x.size + 300000}: probe values stable")

    # (b) offset the probe inside the batch (lane alignment)
    for off in range(0, 17):
        fx = np.concatenate([rng.integers(0, 2001, off).astype(float), probe_x])
        fy = np.concatenate([rng.integers(0, 1501, off).astype(float), probe_y])
        got = sample_power_at(spec, fx, fy)[off:]
        if not np.array_equal(bits(got), bits(ref)):
            fail(f"D2 offset {off}: probe values changed")
    print("  probe offsets 0..16 inside the batch: values stable")

    # (c) permuted ordering (the faithful path lays positions out x-fastest,
    #     the fast path in user order)
    for s in range(5):
        perm = np.random.default_rng(s).permutation(probe_x.size)
        got = sample_power_at(spec, probe_x[perm], probe_y[perm])
        inv = np.empty_like(perm)
        inv[perm] = np.arange(perm.size)
        if not np.array_equal(bits(got[inv]), bits(ref)):
            fail(f"D2 permutation seed {s}: values changed")
    print("  5 random permutations of the same positions: values stable")

    # (d) non-contiguous input (strided views), which can change the ufunc loop
    big_x = np.repeat(probe_x, 3)
    big_y = np.repeat(probe_y, 3)
    got = sample_power_at(spec, big_x[::3], big_y[::3])
    if not np.array_equal(bits(got), bits(ref)):
        fail("D2 strided input: values changed")
    print("  strided (non-contiguous) coordinate views: values stable")


def section_D_mutation(backend_fast, backend_slow) -> None:
    print("\n" + "=" * 78)
    print("D3. Shared mutable state: sample_power_at reuses an lru_cached")
    print("    ScenarioConfig and antenna; power_map builds a fresh scenario and")
    print("    a fresh omni rx array every call. Does either path mutate them?")
    print("=" * 78)
    from pyqd_channel.scenario import load_scenario

    def snap(obj):
        out = {}
        for name in ("Fa", "Fb", "azimuth_grid", "elevation_grid",
                     "element_position", "coupling"):
            v = getattr(obj, name, None)
            if v is not None:
                out[name] = np.array(v, copy=True)
        return out

    def diff(a, b, tag):
        for k in a:
            if not np.array_equal(np.nan_to_num(a[k], nan=-1.0),
                                  np.nan_to_num(b[k], nan=-1.0)):
                fail(f"D3 {tag}: array {k!r} was mutated")

    tx0 = snap(_tx_antenna(2.0e9))
    rx0 = snap(_rx_antenna())
    cfg = _scenario_config("3GPP_38.901_UMa_LOS")
    plpar0 = dict(cfg.plpar)
    scenpar0 = dict(cfg.scenpar)

    for i in range(3):
        cf, cs = fbs_columns(backend_fast, backend_slow, 500.0 + 111 * i,
                             400.0 + 97 * i, 30.0 + 20 * i, 8.0 + 0.5 * i, 2.0e9)
        assert cf[0] and cs[0], (cf, cs)
    diff(tx0, snap(_tx_antenna(2.0e9)), "tx antenna")
    diff(rx0, snap(_rx_antenna()), "rx antenna")
    if dict(cfg.plpar) != plpar0:
        fail("D3: cached scenario plpar mutated")
    if dict(cfg.scenpar) != scenpar0:
        fail("D3: cached scenario scenpar mutated")
    fresh = load_scenario("3GPP_38.901_UMa_LOS", search_cwd=False)
    if dict(fresh.plpar) != dict(cfg.plpar) or dict(fresh.scenpar) != dict(cfg.scenpar):
        fail("D3: cached scenario config differs from a fresh load")
    print("  tx antenna, rx antenna, scenario plpar/scenpar: unmutated after 3 "
          "round trips; cached config == fresh load.")

    # A fresh omni (what power_map builds internally) vs the cached one.
    from pyqd_channel.antenna.generate import generate

    diff(snap(generate("omni")), rx0, "fresh omni vs cached omni")
    print("  fresh generate('omni') == the lru_cached _rx_antenna(), array for array.")


# ===========================================================================
# B/E. The state battery
# ===========================================================================
def build_states(world: WorldConfig, users: np.ndarray) -> list[tuple]:
    """(label, x, y, z, power) states designed to break the equivalence."""
    W, H = float(world.width), float(world.height)
    ux, uy = users[:, 0].astype(float), users[:, 1].astype(float)
    Z_LO, Z_HI = 20.0, 150.0        # fbs_z_bounds
    P_LO, P_HI = 7.0, 10.5          # fbs_power_bounds
    S: list[tuple] = []

    # -- E1: exact corners / edges of the position box, at every gene extreme --
    corners = [(0.0, 0.0), (W, 0.0), (0.0, H), (W, H),
               (-0.0, -0.0), (W, -0.0), (-0.0, H),
               (W / 2, 0.0), (0.0, H / 2), (W, H / 2), (W / 2, H)]
    for (x, y), z, p in itertools.product(corners, (Z_LO, Z_HI), (P_LO, P_HI)):
        S.append((f"corner({x!r},{y!r}) z={z} p={p}", x, y, z, p))

    # -- E2: FBS exactly on top of a user (co-located; d_2d == 0) --
    picks = {
        "user_min_x": int(np.argmin(ux)), "user_max_x": int(np.argmax(ux)),
        "user_min_y": int(np.argmin(uy)), "user_max_y": int(np.argmax(uy)),
        "user_0": 0, "user_center": int(np.argmin((ux - W / 2) ** 2 + (uy - H / 2) ** 2)),
    }
    for name, i in picks.items():
        for z in (FBS_UE_HEIGHT, Z_LO, Z_HI):
            S.append((f"{name}=({ux[i]:g},{uy[i]:g}) z={z}", ux[i], uy[i], z, 9.0))
    # co-located AND at the exact rx height -> 0/0 elevation, d_3d clamp
    i = picks["user_0"]
    S.append((f"colocated-exact z=1.5 p={P_HI}", ux[i], uy[i], FBS_UE_HEIGHT, P_HI))
    S.append((f"colocated z=1.5000000000000002", ux[i], uy[i],
              np.nextafter(FBS_UE_HEIGHT, 2.0), 9.0))
    S.append((f"colocated z=1.4999999999999998", ux[i], uy[i],
              np.nextafter(FBS_UE_HEIGHT, 1.0), 9.0))

    # -- E3: exactly on the MBS site --
    from ppo.pyqd_bridge import generate_hex_sites

    mx, my = generate_hex_sites(W, H, world.isd, world.margin, world.num_mbs)
    for z in (Z_LO, world.mbs_height, Z_HI):
        S.append((f"on-MBS({mx[0]:.6f},{my[0]:.10f}) z={z}", float(mx[0]), float(my[0]), z, 9.0))

    # -- E4: half-integer and irrational-ish coordinates (the FBS is NOT
    #        rounded by either path, but its geometry drives every angle) --
    for x, y in [(1000.5, 750.5), (0.5, 0.5), (W - 0.5, H - 0.5),
                 (1234.5678901234567, 987.6543210987654),
                 (np.nextafter(1000.0, 0.0), np.nextafter(750.0, np.inf)),
                 (1e-323, 1e-323), (5e-324, 0.0)]:
        S.append((f"frac({x!r},{y!r})", x, y, 55.0, 9.0))

    # -- E5: z / power extremes and beyond --
    for z in (Z_LO, Z_HI, 1.0, 1.1, 1.5, 0.1, 0.0, 1e5, 1e-9):
        S.append((f"z={z!r}", 913.0, 641.0, z, 9.0))
    for p in (P_LO, P_HI, 0.0, -0.0, -100.0, 100.0, -308.0, 308.0,
              -3090.0, 3090.0, 8.123456789012345):
        S.append((f"power={p!r}", 913.0, 641.0, 40.0, p))

    # -- E6: dynamic range: weakest and strongest corner of the gene box --
    S.append(("weakest: z=150 p=7 far corner", W, H, Z_HI, P_LO))
    S.append(("strongest: z=20 p=10.5 centre", W / 2, H / 2, Z_LO, P_HI))
    S.append(("very high z=1e4 p=7", W / 2, H / 2, 1e4, P_LO))
    S.append(("very high z=1e6 p=10.5", W / 2, H / 2, 1e6, P_HI))

    # -- E7: outside the world entirely --
    for x, y in [(-500.0, -500.0), (W + 500.0, H + 500.0), (-1e6, 1e6),
                 (-1.0, H / 2), (W + 1.0, H / 2)]:
        S.append((f"outside({x!r},{y!r})", x, y, 40.0, 9.0))

    # -- E8: the dual_slope breakpoint. dBP = 13.34*f_GHz*(hBS-1)*(hMS-1);
    #        land users exactly on it, and sit exactly at hBS == hE.
    for z in (1.0, 1.0000000000000002, 2.0, 21.0):
        S.append((f"breakpoint z={z!r}", 1000.0, 750.0, z, 9.0))

    # -- E9: a 6x6 lattice sweep of the reachable box (coverage in the interior)
    for x in np.linspace(0.0, W, 4):
        for y in np.linspace(0.0, H, 4):
            S.append((f"lattice({x:g},{y:g})", float(x), float(y), 85.0, 8.75))

    return S


def section_F_overflow(world: WorldConfig, band: BandConfig) -> None:
    """End-to-end demonstration of the tx_power overflow asymmetry, if it exists.

    Uses ``make_backend`` with the real backend names, i.e. exactly what a
    caller (env, replay harness, GA) would construct.
    """
    global N_STATES
    print("\n" + "=" * 78)
    print("F. End-to-end: is the tx_power overflow asymmetry visible through")
    print("   make_backend('pyqd-fast') vs make_backend('pyqd')?")
    print("=" * 78)
    from ppo.matlab_bridge import make_backend

    bf = make_backend(world, band, "pyqd-fast")
    bs = make_backend(world, band, "pyqd")
    for p in (10.5, 3082.0, 3082.5, 3083.0, 3090.0, 1e5, -3090.0, -1e5):
        N_STATES += 1
        cont = np.array([[1000.0, 750.0, 40.0, p]], dtype=float)
        with np.errstate(all="ignore"):
            rf = safe(bf.evaluate, cont, np.array([1.0]))
            rs = safe(bs.evaluate, cont, np.array([1.0]))
        same = cmp_results(f"[overflow] power={p!r} (evaluate)", rf, rs)
        print(f"  power={p!r:>10}: fast={'ok' if rf[0] else rf[1]!r:.60} "
              f"faithful={'ok' if rs[0] else rs[1]!r:.60} -> "
              f"{'same' if same else 'DIFFERENT'}")
    print(f"  NOTE: the shipped gene bounds are fbs_power_bounds=(7.0, 10.5) and "
          f"env.py clips to them,\n        so this regime is unreachable from the "
          f"policy; it is reachable by any caller that\n        invokes "
          f"backend.evaluate() directly with an out-of-bounds power gene.")


def section_G_degenerate(band: BandConfig) -> None:
    """Degenerate inputs: NaN/inf genes, 0/1/2 users, several FBSs, live cache."""
    global N_STATES
    print("\n" + "=" * 78)
    print("G. Degenerate inputs: non-finite genes, tiny user counts, multi-FBS,")
    print("   and the live column cache (column_cache_size = default 8)")
    print("=" * 78)
    nan, inf = float("nan"), float("inf")

    # -- G1: non-finite FBS genes, on a small world so the faithful map is cheap
    w = WorldConfig(width=60.0, height=40.0, num_users=200,
                    mbs_locations=[(30.0, 20.0)])
    bf = PyqdSinrBackend(w, band, fast_sampling=True, column_cache_size=0)
    bs = PyqdSinrBackend(w, band, fast_sampling=False, column_cache_size=0)
    nonfinite = [
        ("x=nan", nan, 20.0, 40.0, 9.0),
        ("y=nan", 30.0, nan, 40.0, 9.0),
        ("z=nan", 30.0, 20.0, nan, 9.0),
        ("power=nan", 30.0, 20.0, 40.0, nan),
        ("all nan", nan, nan, nan, nan),
        ("x=+inf", inf, 20.0, 40.0, 9.0),
        ("x=-inf", -inf, 20.0, 40.0, 9.0),
        ("z=+inf", 30.0, 20.0, inf, 9.0),
        ("z=-inf", 30.0, 20.0, -inf, 9.0),
        ("power=+inf", 30.0, 20.0, 40.0, inf),
        ("power=-inf", 30.0, 20.0, 40.0, -inf),
    ]
    for tag, x, y, z, p in nonfinite:
        N_STATES += 1
        with np.errstate(all="ignore"):
            cf, cs = fbs_columns(bf, bs, x, y, z, p, 2.0e9)
            rf = safe(bf.evaluate, np.array([[x, y, z, p]]), np.array([1.0]))
            rs = safe(bs.evaluate, np.array([[x, y, z, p]]), np.array([1.0]))
        same_col = cmp_columns(f"[degenerate] {tag}", cf, cs)
        same_res = cmp_results(f"[degenerate] {tag} (evaluate)", rf, rs)
        if same_col and same_res:
            desc = "raised identically" if not cf[0] else (
                f"nan={int(np.isnan(cf[1]).sum())} inf={int(np.isinf(cf[1]).sum())} "
                f"zero={int((cf[1] == 0).sum())}"
            )
            print(f"  {tag:<12}: identical ({desc})")

    # -- G2: 0, 1 and 2 users
    for n_users in (0, 1, 2):
        w2 = WorldConfig(width=60.0, height=40.0, num_users=n_users,
                         mbs_locations=[(30.0, 20.0)])
        ok, pair = safe(lambda: (
            PyqdSinrBackend(w2, band, fast_sampling=True, column_cache_size=0),
            PyqdSinrBackend(w2, band, fast_sampling=False, column_cache_size=0),
        ))
        if not ok:
            print(f"  num_users={n_users}: backend construction raises for BOTH "
                  f"paths ({pair})")
            continue
        b1, b2 = pair
        N_STATES += 1
        with np.errstate(all="ignore"):
            cf, cs = fbs_columns(b1, b2, 17.0, 11.0, 40.0, 9.0, 2.0e9)
            rf = safe(b1.evaluate, np.array([[17.0, 11.0, 40.0, 9.0]]), np.array([1.0]))
            rs = safe(b2.evaluate, np.array([[17.0, 11.0, 40.0, 9.0]]), np.array([1.0]))
        a = cmp_columns(f"[degenerate] num_users={n_users}", cf, cs)
        b = cmp_results(f"[degenerate] num_users={n_users} (evaluate)", rf, rs)
        print(f"  num_users={n_users}: columns {'identical' if a else 'DIFFER'}, "
              f"SinrResult {'identical' if b else 'DIFFER'}"
              + (f", result={rf[1]}" if rf[0] else ""))

    # -- G3: several FBSs, mixed power_status, duplicate specs, LIVE column cache
    w3 = WorldConfig()
    b1 = PyqdSinrBackend(w3, band, fast_sampling=True)     # default cache = 8
    b2 = PyqdSinrBackend(w3, band, fast_sampling=False)
    cont = np.array([
        [0.0, 0.0, 20.0, 7.0],           # corner, min gene
        [2000.0, 1500.0, 150.0, 10.5],   # opposite corner, max gene
        [0.0, 0.0, 20.0, 7.0],           # exact duplicate -> cache hit
        [1000.0, 750.0, 85.0, 8.75],     # centre
        [-0.0, -0.0, 20.0, 7.0],         # -0.0 vs 0.0: same MapSpec hash/eq
    ], dtype=float)
    for status in ([1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1]):     # last repeat exercises a warm cache
        N_STATES += 1
        st = np.array(status, dtype=float)
        with np.errstate(all="ignore"):
            rf = safe(b1.evaluate, cont, st)
            rs = safe(b2.evaluate, cont, st)
        same = cmp_results(f"[multi-FBS] status={status}", rf, rs)
        print(f"  status={status}: {'identical' if same else 'DIFFERENT'}"
              + (f"  connected={rf[1].total_connected} "
                 f"fbs={rf[1].fbs_connected} sum_rate={rf[1].sum_rate!r}"
                 if rf[0] else ""))


def run_battery(label: str, world: WorldConfig, band: BandConfig,
                states, freq: float, do_eval_every: int = 4):
    global N_STATES
    print("\n" + "=" * 78)
    print(f"B/E. State battery [{label}]: {len(states)} constructed states, "
          f"bitwise FBS-column comparison")
    print("=" * 78)
    t0 = time.time()
    bf = PyqdSinrBackend(world, band, fast_sampling=True, column_cache_size=0)
    bs = PyqdSinrBackend(world, band, fast_sampling=False, column_cache_size=0)
    print(f"  backends built in {time.time() - t0:.1f}s "
          f"(users={world.num_users}, mbs={bf.num_mbs})")

    t0 = time.time()
    n_ok = 0
    for k, (tag, x, y, z, p) in enumerate(states):
        N_STATES += 1
        with np.errstate(all="ignore"):
            cf, cs = fbs_columns(bf, bs, x, y, z, p, freq)
        n_ok += cmp_columns(f"[{label}] {tag}", cf, cs)
        if do_eval_every and k % do_eval_every == 0:
            cont = np.array([[x, y, z, p]], dtype=float)
            with np.errstate(all="ignore"):
                rf = safe(bf.evaluate, cont, np.array([1.0]))
                rs = safe(bs.evaluate, cont, np.array([1.0]))
            cmp_results(f"[{label}] {tag} (evaluate)", rf, rs)
        if (k + 1) % 25 == 0:
            print(f"  ... {k + 1}/{len(states)} states, "
                  f"{time.time() - t0:.0f}s elapsed, {n_ok} columns bit-identical")
    print(f"  [{label}] {n_ok}/{len(states)} columns bit-identical "
          f"in {time.time() - t0:.0f}s")
    return bf, bs


def main() -> int:
    global N_STATES
    t_start = time.time()
    np.seterr(all="ignore")
    band = BandConfig(mode="legacy")

    section_A()

    # ---- default world (the one every run used) --------------------------
    world = WorldConfig()  # 2000 x 1500, 1000 users, 1 MBS
    print("\n" + "=" * 78)
    print("C. n_x/n_y and the clamp, against the grid power_map really returns")
    print("=" * 78)
    section_C_grid(world, band, "default 2000x1500")

    section_D_scale()
    section_D_batch()

    users = matlab_user_positions(0, world.width, 0, world.height,
                                  world.num_users, seed=0)
    states = build_states(world, users)
    bf, bs = run_battery("default", world, band, states, 2.0e9)
    section_D_mutation(bf, bs)
    section_F_overflow(world, band)
    section_G_degenerate(band)

    # ---- adversarial worlds ---------------------------------------------
    print("\n" + "=" * 78)
    print("C/B. Adversarial worlds: non-square, transposed, non-integer extent,")
    print("     extreme aspect ratio, extent not a multiple of sample_distance")
    print("=" * 78)
    adversarial = [
        ("non-integer W/H", WorldConfig(
            width=2000.5, height=1500.5, num_users=400,
            mbs_locations=[(1100.0, 966.0254037844386)])),
        ("W just below int", WorldConfig(
            width=1999.9999999999998, height=1500.0000000000002, num_users=300,
            mbs_locations=[(1100.0, 966.0)])),
        ("transposed 1500x2000", WorldConfig(
            width=1500.0, height=2000.0, num_users=400,
            mbs_locations=[(700.0, 900.0)])),
        ("square 1500x1500", WorldConfig(
            width=1500.0, height=1500.0, num_users=300,
            mbs_locations=[(700.0, 900.0)])),
        ("extreme aspect 3000x7", WorldConfig(
            width=3000.0, height=7.0, num_users=500,
            mbs_locations=[(1500.0, 3.0)])),
        ("extreme aspect 7x3000", WorldConfig(
            width=7.0, height=3000.0, num_users=500,
            mbs_locations=[(3.0, 1500.0)])),
        ("tiny 3x2", WorldConfig(
            width=3.0, height=2.0, num_users=200,
            mbs_locations=[(1.0, 1.0)])),
        ("fractional tiny 3.75x2.25", WorldConfig(
            width=3.75, height=2.25, num_users=200,
            mbs_locations=[(1.0, 1.0)])),
    ]
    for label, w in adversarial:
        try:
            b = section_C_grid(w, band, label)
        except Exception as exc:  # pragma: no cover - diagnostic
            fail(f"[{label}] grid check raised {type(exc).__name__}: {exc}")
            continue
        # a compact but nasty state set for each odd world
        W, H = float(w.width), float(w.height)
        ux, uy = b._user_x, b._user_y
        small = [
            ("corner 0,0 z=20", 0.0, 0.0, 20.0, 7.0),
            (f"corner W,H z=150", W, H, 150.0, 10.5),
            ("centre z=20 p=10.5", W / 2, H / 2, 20.0, 10.5),
            ("outside -W,-H", -W, -H, 40.0, 9.0),
            ("on user 0 z=1.5", float(ux[0]), float(uy[0]), FBS_UE_HEIGHT, 9.0),
            ("on user -1 z=150", float(ux[-1]), float(uy[-1]), 150.0, 10.5),
            ("half-integer", W / 2 + 0.5, H / 2 + 0.5, 33.0, 8.25),
            ("z=1e5 p=-100", W / 3, H / 3, 1e5, -100.0),
        ]
        run_battery(label, w, band, small, 2.0e9, do_eval_every=2)

    # ---- multi-band world: exercises the capacity carrier too ------------
    print("\n" + "=" * 78)
    print("B. Multi-band world (2.6 GHz FBS carrier + MBS capacity slots)")
    print("=" * 78)
    mb = BandConfig(mode="multi", fbs_band="agent", mbs_capacity="on")
    w = WorldConfig(num_users=1000)
    mb_states = [
        ("cap-band corner 0,0 z=20 p=7", 0.0, 0.0, 20.0, 7.0),
        ("cap-band corner W,H z=150 p=10.5", 2000.0, 1500.0, 150.0, 10.5),
        ("cap-band on user 0 z=1.5", float(users[0, 0]), float(users[0, 1]),
         FBS_UE_HEIGHT, 9.0),
        ("cap-band centre", 1000.0, 750.0, 85.0, 8.75),
        ("cap-band z=1e6 p=3090", 1000.0, 750.0, 1e6, 3090.0),
    ]
    bfm = PyqdSinrBackend(w, mb, fast_sampling=True, column_cache_size=0)
    bsm = PyqdSinrBackend(w, mb, fast_sampling=False, column_cache_size=0)
    for tag, x, y, z, p in mb_states:
        for band_flag, fr in ((0.0, 2.0e9), (1.0, 2.6e9)):
            N_STATES += 1
            with np.errstate(all="ignore"):
                cf, cs = fbs_columns(bfm, bsm, x, y, z, p, fr)
            cmp_columns(f"[multi f={fr:.1e}] {tag}", cf, cs)
            cont = np.array([[x, y, z, p]], dtype=float)
            with np.errstate(all="ignore"):
                rf = safe(bfm.evaluate, cont, np.array([1.0]),
                          fbs_band_flags=np.array([band_flag]),
                          mbs_capacity_genes=np.array([1.0]))
                rs = safe(bsm.evaluate, cont, np.array([1.0]),
                          fbs_band_flags=np.array([band_flag]),
                          mbs_capacity_genes=np.array([1.0]))
            cmp_results(f"[multi f={fr:.1e}] {tag} (evaluate)", rf, rs)

    # ---- summary ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  constructed states evaluated through BOTH paths : {N_STATES}")
    print(f"  bitwise FBS-column comparisons                  : {N_COLUMN_CMP}")
    print(f"  full SinrResult comparisons (exact, atol=rtol=0) : {N_EVAL_CMP}")
    print(f"  divergences found                               : {len(FAILURES)}")
    for m in FAILURES:
        print(f"    - {m}")
    print(f"  wall clock: {time.time() - t_start:.0f}s")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
