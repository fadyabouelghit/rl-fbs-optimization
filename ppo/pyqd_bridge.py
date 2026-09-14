"""MATLAB-free physics: QuaDRiGa coverage maps computed in-process.

``PyqdSinrBackend`` is a line-for-line port of the MATLAB stack that
``MatlabSinrBackend`` drives over the engine boundary --
``ppo_world_setup.m`` -> ``precompute_mbs_power_maps.m`` -> ``ppo_sinr_eval.m``
-> ``SINREvaluation.m`` -- onto numpy plus the ``pyqd-channel`` package (a
validated Python port of QuaDRiGa's coverage-map subset). Same inputs, same
``SinrResult``, no MATLAB, no QuaDRiGa install, no ~20 s engine cold start and
no ~1.5 GB engine per SubprocVecEnv worker.

**Why a port and not a rewrite.** The MATLAB reference contains several
conventions that look like bugs and are not: they are load-bearing, because
every historical run, every cached map and every logged metric was produced
under them. Changing any of them would silently move the physics and make new
runs incomparable to the archive. Each one is reproduced here deliberately and
carries a comment pointing at the MATLAB line it comes from. The three that
bite hardest:

1. **The transpose asymmetry.** ``calculate_power`` (SINREvaluation.m:188)
   ends in ``sum(...)'`` -- a trailing transpose -- so an FBS map is MATLAB
   ``(n_x, n_y)``. ``precompute_mbs_power_maps.m:87`` has *no* transpose, so a
   cached MBS map is ``(n_y, n_x)``. Both are then read by the same
   ``sample_nearest`` helper, which always indexes ``P(x_index, y_index)``.
   See :func:`sample_nearest`.
2. **The x<->y row swap.** ``ppo_world_setup.m:59-61`` swaps rows 1 and 2 of
   ``mbs_params`` before precomputing, so the MBS map is computed at
   ``tx = [true_y, true_x, height]``. Sampling it transposed (1) very nearly
   undoes this -- but not exactly, because ``sample_nearest`` clamps against
   the *swapped* extent. See :meth:`PyqdSinrBackend._mbs_column`.
3. **``total_transmitted_pwr = sum(tx_power)``** with the ``power_status``
   mask commented out (SINREvaluation.m:120-121), i.e. inactive FBSs still
   count towards reported power. ``AnalyticSinrBackend`` does *not* do this;
   do not copy it from there.

**Cost model.** The faithful path evaluates a full ``sample_distance = 1 m``
coverage grid per active FBS per step (2001x1501 = 3.0e6 points, ~0.6 s) and
then reads it at ``num_users`` points -- the same work MATLAB does, so a
timing comparison against the MATLAB backend is apples to apples. Because
``sample_nearest`` snaps users to integer grid nodes and the grid step is
exactly 1 m, evaluating the LOS model at just those nodes is *the same
arithmetic on the same inputs*; the opt-in ``fast_sampling=True`` path does
that and is bit-identical (verified: max |delta| = 0.0 dB), turning a 0.57 s
map into a 0.003 s point evaluation -- ~190x on the map, ~160x end to end
(1.0 -> 162 steps/s with two active FBSs on the default world). It is off by
default so that out-of-the-box timings stay directly comparable to the MATLAB
baseline, which measures ~1.0 steps/s on the same problem.

MBS maps are constant for the lifetime of a world, so they are computed once
and memoised on disk (``cache_pyqd_maps/``, keyed by a hash of every parameter
that can move the map) -- this is what keeps ``n_envs > 1`` startup cheap.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from .config import BAND_CAPACITY, BAND_COVERAGE, BandConfig, WorldConfig
from .matlab_bridge import SinrResult
from .matlab_rng import matlab_user_positions
from .paths import PYQD_CACHE_DIR, ensure_dir

__all__ = [
    "MapSpec",
    "MbsSlot",
    "PyqdSinrBackend",
    "band_frequencies",
    "band_vector",
    "build_slots",
    "check_fast_sampling_invariants",
    "compute_power_map",
    "generate_hex_sites",
    "load_or_compute_map",
    "matlab_colon",
    "sample_nearest",
    "sample_power_at",
]


# --------------------------------------------------------------------------- #
# Constants transcribed from the MATLAB
# --------------------------------------------------------------------------- #
#: band_frequencies.m:22-24. Band id 0 -> antenna row 1, band id 1 -> row 2.
F_COVERAGE = 2.0e9
F_CAPACITY = 2.6e9

#: SINREvaluation.m:204. Thermal noise floor in mW, added to the interference
#: sum before the SINR ratio.
NOISE_MW = 1e-11

#: SINREvaluation.m:185-186. The FBS power_map call hard-codes its scenario,
#: usage mode, grid step and receiver height -- it does *not* read the world
#: config, unlike the MBS path (ppo_world_setup.m:64-65). If a world ever sets
#: ``ue_height != 1.5`` or a different scenario, MATLAB silently evaluates FBS
#: and MBS under different assumptions. Reproduced, not "fixed".
FBS_SCENARIO = "3GPP_38.901_UMa_LOS"
FBS_MAP_MODE = "quick"
FBS_UE_HEIGHT = 1.5

#: Both power_map call sites pass sample_distance = 1 (SINREvaluation.m:185,
#: precompute_mbs_power_maps.m:82). The fast-sampling path depends on this
#: being exactly 1 m *and* on the grid origin being an integer, because that is
#: what makes ``round(user_xy)`` land on a grid node. Enforced, not merely
#: documented -- see :func:`check_fast_sampling_invariants`.
SAMPLE_DISTANCE = 1.0

#: Bumped whenever the on-disk .npz layout or its key payload changes, so old
#: cache files are ignored rather than misread.
_CACHE_FORMAT = 1

#: Sentinel for "resolve :data:`~ppo.paths.PYQD_CACHE_DIR` when the backend is
#: constructed", rather than freezing it into the signature's default at import
#: time. Lets tests (and anyone else) redirect the cache by patching the module
#: attribute, the same way ``ppo.run_logging.LEDGER_PATH`` is redirected.
_DEFAULT_CACHE_DIR = object()


def band_frequencies(band: BandConfig) -> tuple[float, ...]:
    """Carrier frequencies by band id, mirroring ppo_world_setup.m:32-41.

    ``multi`` builds coverage + capacity antenna templates; ``legacy`` calls
    ``setup_antenna()`` with no argument, whose default is 2 GHz
    (setup_antenna.m:2-4), giving a single shared band.
    """
    return (F_COVERAGE, F_CAPACITY) if band.is_multi else (F_COVERAGE,)


# --------------------------------------------------------------------------- #
# MATLAB numerics helpers
# --------------------------------------------------------------------------- #
def matlab_colon(base: float, step: float, limit: float) -> np.ndarray:
    """``base:step:limit`` with MATLAB's endpoint tolerance.

    A naive ``np.arange`` drops the final element whenever ``limit`` is a hair
    beyond ``base + n*step`` in floating point, which would silently delete a
    whole row of hex sites. MATLAB admits one extra element when the endpoint
    lands within a few ulps, so we do too.
    """
    if step == 0.0:
        return np.zeros(0, dtype=float)
    n = int(np.floor((limit - base) / step))
    tol = 3.0 * np.finfo(float).eps * max(abs(base), abs(limit), abs(step))
    if abs(base + (n + 1) * step - limit) <= tol:
        n += 1
    if n < 0:
        return np.zeros(0, dtype=float)
    return base + np.arange(n + 1, dtype=float) * step


def generate_hex_sites(
    width: float, height: float, isd: float, margin: float, num_mbs: int
) -> tuple[np.ndarray, np.ndarray]:
    """Port of matlab/generate_hex_sites.m -- MBS sites in the TRUE frame.

    Lays a hex lattice (odd rows flush with the margin, even rows offset by
    half an inter-site distance) and keeps the ``num_mbs`` sites closest to the
    world centre. ``np.argsort(kind="stable")`` matters: MATLAB's ``sort`` is
    stable, so exact distance ties resolve to the earlier lattice point.

    Default world (2000x1500, isd=500, margin=100, num_mbs=1) -> the single
    site (1100.0, 966.0254037844386).
    """
    dx = isd
    dy = isd * np.sqrt(3) / 2
    rows = matlab_colon(margin, dy, height - margin)

    xs_parts: list[np.ndarray] = []
    ys_parts: list[np.ndarray] = []
    for row_index, y_value in enumerate(rows, start=1):  # MATLAB is 1-based
        if row_index % 2 == 1:
            row = matlab_colon(margin, dx, width - margin)
        else:
            row = matlab_colon(margin + dx / 2.0, dx, width - margin)
        xs_parts.append(row)
        ys_parts.append(np.full(row.size, y_value, dtype=float))

    xs = np.concatenate(xs_parts) if xs_parts else np.zeros(0)
    ys = np.concatenate(ys_parts) if ys_parts else np.zeros(0)
    # generate_hex_sites.m:32's clip test is a no-op given the construction
    # above; skipped rather than reproduced, since it cannot change the result.
    if num_mbs > xs.size:
        raise ValueError(
            f"Requested {num_mbs} MBSs but only {xs.size} hex sites fit in area"
        )

    d2 = (xs - width / 2.0) ** 2 + (ys - height / 2.0) ** 2
    order = np.argsort(d2, kind="stable")
    keep = order[:num_mbs]
    return xs[keep].copy(), ys[keep].copy()


def sample_nearest(grid: np.ndarray, user_x: np.ndarray, user_y: np.ndarray) -> np.ndarray:
    """Literal transcription of SINREvaluation.m:306-321.

    ``grid`` must be laid out exactly as the MATLAB variable ``P``, i.e.
    ``grid.shape == size(P)``. The helper always indexes ``P(x_index,
    y_index)`` regardless of which orientation it was handed, and clamps each
    index against the *corresponding* axis of whatever it got:

        ix = round(user_x) + 1  clamped to size(P, 1)
        iy = round(user_y) + 1  clamped to size(P, 2)

    Do not "simplify" this. For an FBS map (transposed to ``(n_x, n_y)`` by
    SINREvaluation.m:188) the clamps never fire and it reduces to natural
    indexing. For an MBS map (left at ``(n_y, n_x)`` by
    precompute_mbs_power_maps.m:87) axis 1 is only ``height + 1`` long, so
    every user with ``round(x) > height`` is evaluated as if its x were exactly
    ``height`` -- 226 of the 1000 default-world users. That clamp is part of
    the reference physics, and the ground-truth power columns only match with
    it in place.

    ``np.floor(v + 0.5)`` rather than ``np.round``: numpy rounds halves to
    even, MATLAB rounds halves away from zero. Users are integers today, so it
    never differs -- but it costs nothing to be right.
    """
    n_x, n_y = grid.shape
    ix = np.clip(np.floor(np.asarray(user_x, dtype=float) + 0.5).astype(np.int64) + 1, 1, n_x)
    iy = np.clip(np.floor(np.asarray(user_y, dtype=float) + 0.5).astype(np.int64) + 1, 1, n_y)
    return grid[ix - 1, iy - 1]


# --------------------------------------------------------------------------- #
# Power-map specification + caching
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MapSpec:
    """Every input that can change a power map -- and therefore the cache key.

    Deliberately exhaustive: a cache that silently returns a map computed for a
    slightly different transmitter is far worse than no cache at all. Floats go
    into the key as ``float.hex()`` so the key is exact rather than
    format-dependent, and the installed ``pyqd-channel`` version is part of the
    payload so a physics fix in the dependency invalidates every stored map.
    """

    scenario: str
    map_mode: str
    tx_x: float
    tx_y: float
    tx_z: float
    tx_power: float
    center_freq: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    sample_distance: float
    rx_height: float

    def payload(self) -> dict:
        import pyqd_channel

        def h(value: float) -> str:
            return float(value).hex()

        return {
            "format": _CACHE_FORMAT,
            "pyqd_channel": getattr(pyqd_channel, "__version__", "unknown"),
            "scenario": self.scenario,
            "map_mode": self.map_mode,
            "tx": [h(self.tx_x), h(self.tx_y), h(self.tx_z)],
            "tx_power": h(self.tx_power),
            "center_freq": h(self.center_freq),
            "bounds": [h(self.x_min), h(self.x_max), h(self.y_min), h(self.y_max)],
            "sample_distance": h(self.sample_distance),
            "rx_height": h(self.rx_height),
        }

    @property
    def key(self) -> str:
        blob = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@lru_cache(maxsize=8)
def _scenario_config(scenario: str):
    """Parsed scenario tables. Cached: re-parsing per FBS per step is pure waste."""
    from pyqd_channel.scenario import load_scenario

    return load_scenario(scenario, search_cwd=False)


@lru_cache(maxsize=8)
def _tx_antenna(center_freq: float):
    """One rotated 3GPP-3D template per carrier, built once and shared.

    Mirrors setup_antenna.m:9-11 (``qd_arrayant('3gpp-3d',1,1,f)`` then
    ``rotate_pattern(-90,'y',1,1)``). ``pyqd_channel.antenna.rotate_pattern``
    mutates its argument in place, so re-rotating a reused template would tilt
    it -180 deg; caching here makes that impossible. MATLAB sidesteps the issue
    by building a fresh ``qd_layout`` per call.
    """
    from pyqd_channel.antenna import generate, rotate_pattern

    ant = generate("3gpp-3d", 1, 1, float(center_freq))
    rotate_pattern(ant, -90, "y", [0], 1)
    return ant


@lru_cache(maxsize=1)
def _rx_antenna():
    """``l.rx_array = qd_arrayant('omni')`` (setup_antenna.m:12).

    ``power_map`` defaults to omni, but the fast path calls ``get_los_coeff``
    directly and has to supply it explicitly.
    """
    from pyqd_channel.antenna import generate

    return generate("omni")


def compute_power_map(spec: MapSpec) -> np.ndarray:
    """Full coverage grid for ``spec``, shaped ``(n_y, n_x)`` in linear mW.

    This is the faithful path: exactly the call MATLAB makes, with the antenna
    ports collapsed the way ``sum(cat(3, map{:}), 3)`` collapses them.

    (MATLAB sums the cell concatenation over dim 3 only, not dim 4. That equals
    ``maps[0].sum(axis=(2, 3))`` here because both arrays are single-element --
    ``qd_arrayant('3gpp-3d',1,1,f)`` has one element and ``no_tx`` is 1 -- so
    the tx-port axis is singleton. It is not a general identity.)
    """
    from pyqd_channel.layout import power_map

    maps, _x, _y = power_map(
        spec.scenario,
        _tx_antenna(spec.center_freq),
        np.array([spec.tx_x, spec.tx_y, spec.tx_z], dtype=float),
        spec.center_freq,
        spec.x_min,
        spec.x_max,
        spec.y_min,
        spec.y_max,
        sample_distance=spec.sample_distance,
        rx_height=spec.rx_height,
        tx_power=spec.tx_power,
        usage=spec.map_mode,
    )
    return maps[0].sum(axis=(2, 3))


def check_fast_sampling_invariants(spec: MapSpec, width: float, height: float) -> None:
    """Fail loudly if the fast sampler's grid assumptions stop holding.

    :meth:`PyqdSinrBackend._grid_nodes` reconstructs the coordinates
    :func:`sample_nearest` would read *without ever looking at the map*: it
    returns ``index * SAMPLE_DISTANCE`` with
    ``index = clip(round(user), 0, floor(world_extent / SAMPLE_DISTANCE))``.
    ``power_map`` builds its own vector as ``x_min + arange(floor((x_max -
    x_min) / sample_distance) + 1) * sample_distance``
    (``pyqd_channel/layout/power_map.py:111-114``). Those two expressions agree
    bit for bit only under the conditions checked here, and every one of them
    fails *silently* if broken -- the fast path would go on returning a
    perfectly plausible power column evaluated at the wrong receiver positions,
    which is the worst possible failure mode for a physics backend.

    Equivalence-critical, i.e. break these and fast != faithful:

    * ``spec.sample_distance == SAMPLE_DISTANCE`` -- ``_grid_nodes`` uses the
      module constant while the map is built from the spec's step.
    * ``spec.x_min == spec.y_min == 0`` -- ``_grid_nodes`` drops the origin
      term, so any non-zero origin shifts every receiver by that offset.
    * ``spec.x_max == width`` and ``spec.y_max == height`` -- ``_grid_nodes``
      takes the node count, and therefore the clamp bound, from the world
      rather than from the spec.

    Contract, i.e. break it and both paths move together but away from the
    archive:

    * ``SAMPLE_DISTANCE == 1.0`` -- this is what makes ``round(user_xy)`` land
      on a node at all, it is what both MATLAB call sites pass, and it is what
      every cached map and every logged metric in the archive was produced
      under. (A step of, say, 2 m would keep the two paths equal to each other,
      because ``_grid_nodes`` scales by the same constant -- but it would move
      the physics for both of them, so it is still a change that must be made
      deliberately rather than inherited.)

    Cost is a handful of float comparisons against a ~3 ms evaluation, so this
    runs on every fast-path call rather than only at construction.

    Raises:
        ValueError: listing every invariant that is violated.
    """
    broken = []
    if SAMPLE_DISTANCE != 1.0:
        broken.append(
            f"SAMPLE_DISTANCE is {SAMPLE_DISTANCE!r}, not 1.0: round(user_xy) no "
            "longer lands on a grid node and the archive was produced at 1 m"
        )
    if spec.sample_distance != SAMPLE_DISTANCE:
        broken.append(
            f"spec.sample_distance ({spec.sample_distance!r}) != SAMPLE_DISTANCE "
            f"({SAMPLE_DISTANCE!r}); _grid_nodes steps by the module constant"
        )
    if spec.x_min != 0.0 or spec.y_min != 0.0:
        broken.append(
            f"grid origin is ({spec.x_min!r}, {spec.y_min!r}), not (0.0, 0.0); "
            "_grid_nodes omits the origin term"
        )
    if spec.x_max != float(width) or spec.y_max != float(height):
        broken.append(
            f"map extent ({spec.x_max!r}, {spec.y_max!r}) != world extent "
            f"({float(width)!r}, {float(height)!r}); _grid_nodes counts nodes "
            "from the world, so the clamp bound would disagree with the map"
        )
    if broken:
        raise ValueError(
            "fast_sampling=True is only bit-identical to the faithful path "
            "while its grid invariants hold, and these do not:\n  - "
            + "\n  - ".join(broken)
            + "\nEither fix the invariant or run with fast_sampling=False "
            "(backend 'pyqd')."
        )


def sample_power_at(spec: MapSpec, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """Received power [mW] at explicit points -- the fast-sampling primitive.

    ``power_map`` builds its receiver positions, calls ``get_los_coeff`` once
    for the whole batch, and scales by ``10**(0.1*tx_power)``. Every step of
    that is elementwise in the receiver index, so evaluating a 1000-point batch
    gives bit-identical values to reading the 3.0e6-point grid at the same
    coordinates. The port-axis collapse is written to mirror ``power_map``'s
    ``transpose(val, (2,0,1)) -> sum`` exactly, so even a hypothetical
    multi-port array would reduce in the same order.

    Callers must pass coordinates that lie *on* the grid (see
    :meth:`PyqdSinrBackend._grid_nodes`); this function does not snap.
    """
    from pyqd_channel.builder.los import get_los_coeff

    cfg = _scenario_config(spec.scenario)
    rx_pos = np.empty((3, np.size(grid_x)), dtype=float)
    rx_pos[0] = grid_x
    rx_pos[1] = grid_y
    rx_pos[2] = spec.rx_height

    coeff = get_los_coeff(
        _tx_antenna(spec.center_freq),
        _rx_antenna(),
        np.array([[spec.tx_x], [spec.tx_y], [spec.tx_z]], dtype=float),
        rx_pos,
        spec.center_freq,
        cfg.plpar,
        cfg.scenpar,
    )
    # power_map: val = |coeff|**2 -> transpose to (n_pos, n_rx, n_tx) -> * scale,
    # and the caller then sums the two port axes.
    val = np.transpose(np.abs(coeff) ** 2, (2, 0, 1)) * (10 ** (0.1 * spec.tx_power))
    return val.sum(axis=(1, 2))


def load_or_compute_map(spec: MapSpec, cache_dir: Optional[Path]) -> np.ndarray:
    """Disk-memoised :func:`compute_power_map`, stored as float32 ``.npz``.

    Only used for MBS maps, which are fixed for the lifetime of a world and are
    the whole of a worker's startup cost under ``SubprocVecEnv``. float32 is
    not a space optimisation -- it is parity: ``precompute_mbs_power_maps.m:88``
    does ``map = single(map)``, so the reference MBS physics really is
    single-precision (~6e-8 relative, ~3e-7 dB on SINR). The FBS path has no
    such cast and stays float64.

    Cache files live in their own directory, never alongside the MATLAB
    ``.mat`` cache: the MATLAB key is an MD5 over a MATLAB ``jsonencode``
    payload that would be fragile to reproduce byte-for-byte from Python, and a
    key collision between the two would be a silent physics swap.
    """
    if cache_dir is None:
        return compute_power_map(spec).astype(np.float32)

    cache_dir = Path(cache_dir)
    path = cache_dir / f"mbs_{spec.key}.npz"
    if path.is_file():
        try:
            with np.load(path) as stored:
                return np.asarray(stored["map"], dtype=np.float32)
        except Exception:
            # A truncated/corrupt cache entry must never be fatal: recompute.
            pass

    grid = compute_power_map(spec).astype(np.float32)
    # The temp name must itself end in .npz: savez_compressed silently appends
    # the extension otherwise, and the rename below would then look for a file
    # that was never written. Bound outside the try so the cleanup path can
    # always see it.
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        ensure_dir(cache_dir)
        np.savez_compressed(
            tmp,
            map=grid,
            meta=np.array(json.dumps(spec.payload(), sort_keys=True)),
        )
        # Atomic publish: concurrent workers must never observe a half-written
        # file (SubprocVecEnv starts n_envs of them at once).
        tmp.replace(path)
    except OSError:
        # A read-only or full cache dir is a performance problem, not an error
        # -- but the partially written temp file must not outlive it. The name
        # carries the pid and starts with a dot, so under SubprocVecEnv a
        # recurring failure would otherwise accumulate one orphan per worker
        # per map, invisible to `ls` and never reclaimed.
        with suppress(OSError):
            tmp.unlink(missing_ok=True)
    return grid


# --------------------------------------------------------------------------- #
# MBS band slots
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MbsSlot:
    """One row of ``mbsSlotMap`` (SINREvaluation.m:66-72), 0-based.

    MATLAB stores ``[mbsIdx, bandIdx, active]`` 1-based; ``band_row`` here is
    ``bandIdx - 1``, i.e. the row index into the per-band map cache, where 0 is
    the coverage band and 1 the capacity band.
    """

    site: int
    band_row: int
    active: bool
    band_id: int


def band_vector(value, expected: int, name: str, unit: str) -> np.ndarray:
    """``reshape(1, [])`` plus the ``isempty -> zeros`` default of ppo_sinr_eval.m:35-38.

    MATLAB substitutes an all-zero vector for an *empty* band argument, so
    ``None`` and ``[]`` must behave identically here -- a caller that mirrors
    the MATLAB signature (a GA replay, say) passes ``[]``, not ``None``.

    A non-empty vector of the wrong length is a caller bug, not a shape to
    broadcast: MATLAB would either index past the end of ``fbsAntennaEval`` or
    build a mis-sized ``bsBandIds``, and this port would raise a bare
    ``IndexError`` from deep inside the association loop. Say what was wrong
    instead.
    """
    if value is None:
        return np.zeros(expected, dtype=float)
    vec = np.asarray(value, dtype=float).reshape(-1)
    if vec.size == 0:
        return np.zeros(expected, dtype=float)
    if vec.size != expected:
        raise ValueError(
            f"{name} must have one entry per {unit} ({expected}), got {vec.size}"
        )
    return vec


def build_slots(
    num_mbs: int, is_multi: bool, capacity_genes: np.ndarray
) -> list[MbsSlot]:
    """Slot expansion, matching ppo_sinr_eval.m:52-57 (multi) / :75 (legacy).

    Multi mode interleaves per site -- ``[cov(mbs1), cap(mbs1), cov(mbs2),
    ...]`` -- *not* grouped by band. Column order is load-bearing because
    association ties resolve to the earlier column (SINREvaluation.m:278), so
    this ordering is part of the physics, not a presentation choice.
    """
    if not is_multi:
        # One always-on coverage slot per site: slotMap = [(1:numMbs)', 1, 1].
        return [MbsSlot(j, 0, True, BAND_COVERAGE) for j in range(num_mbs)]
    slots: list[MbsSlot] = []
    for j in range(num_mbs):
        slots.append(MbsSlot(j, 0, True, BAND_COVERAGE))
        slots.append(MbsSlot(j, 1, bool(capacity_genes[j] >= 0.5), BAND_CAPACITY))
    return slots


# --------------------------------------------------------------------------- #
# The backend
# --------------------------------------------------------------------------- #
class PyqdSinrBackend:
    """Real QuaDRiGa-equivalent physics with no MATLAB anywhere in the loop.

    Constructed like every other backend (``PyqdSinrBackend(world, band)``) and
    returns the same :class:`~ppo.matlab_bridge.SinrResult`. Construction does
    the work of ``ppo_world_setup.m``: resolve the MBS sites, build one antenna
    template per band, apply the legacy x<->y row swap, and precompute (or load
    from disk) one power map per band per site. Per step, ``evaluate`` does the
    work of ``ppo_sinr_eval.m`` + ``SINREvaluation.m``.

    Args:
        world: geometry, MBS parameters, user count and SINR threshold.
        band: ``legacy`` (one shared band) or ``multi`` (coverage + capacity).
        cache_dir: on-disk MBS map cache. Defaults to
            :data:`~ppo.paths.PYQD_CACHE_DIR`, resolved at construction time so
            it stays patchable; ``None`` disables the cache entirely.
        fast_sampling: opt-in. Evaluate the LOS model only at the grid nodes
            that ``sample_nearest`` would read, instead of over the whole
            coverage grid. Bit-identical (validated at 0.0 dB max deviation),
            ~190x cheaper per FBS map and ~160x faster end to end. Default off
            so that out-of-the-box timings stay comparable with the MATLAB
            backend.
        column_cache_size: how many recently-evaluated FBS power columns to
            memoise, keyed by the full :class:`MapSpec`. Only exact repeats
            hit, so this changes no result; it just makes repeated evaluation
            of an unchanged state (tests, absolute-action replays) cheap.
            Columns are ``num_users`` floats, so the cache is kilobytes.
    """

    def __init__(
        self,
        world: WorldConfig,
        band: BandConfig,
        cache_dir=_DEFAULT_CACHE_DIR,
        fast_sampling: bool = False,
        column_cache_size: int = 8,
    ):
        self.world = world
        self.band = band
        if cache_dir is _DEFAULT_CACHE_DIR:
            cache_dir = PYQD_CACHE_DIR
        self.cache_dir: Optional[Path] = None if cache_dir is None else Path(cache_dir)
        self.fast_sampling = bool(fast_sampling)

        # -- bands (ppo_world_setup.m:32-41) ---------------------------- #
        self.band_freqs = band_frequencies(band)
        self.n_bands = len(self.band_freqs)

        # Fail at construction, not on the first active FBS several thousand
        # steps into a run: a fast backend whose grid invariants are broken is
        # not a slower-but-correct backend, it is a wrong one.
        if self.fast_sampling:
            check_fast_sampling_invariants(
                self._fbs_spec(0.0, 0.0, 0.0, 0.0, self.band_freqs[0]),
                world.width,
                world.height,
            )

        # -- MBS sites, TRUE frame (ppo_world_setup.m:44-52) ------------ #
        if world.mbs_locations is not None and len(world.mbs_locations) > 0:
            coords = np.asarray(world.mbs_locations, dtype=float).reshape(-1, 2)
            mbs_x, mbs_y = coords[:, 0].copy(), coords[:, 1].copy()
        else:
            mbs_x, mbs_y = generate_hex_sites(
                world.width, world.height, world.isd, world.margin, world.num_mbs
            )
        # ppo_world_setup.m:88-89 returns the UNswapped coordinates to Python,
        # so this is what plotting/evaluation see. The swap below is confined
        # to the map computation.
        self.mbs_x = np.asarray(mbs_x, dtype=float).ravel()
        self.mbs_y = np.asarray(mbs_y, dtype=float).ravel()
        self.num_mbs = int(self.mbs_x.size)
        self.contains_mbs = self.num_mbs > 0

        # -- fixed user map (SINREvaluation.m:136-144) ------------------ #
        # Drawn once and reused by every evaluate(): the demand map is a
        # property of the world, not of the step. The stream is a bit-exact
        # reproduction of MATLAB's RandStream('mt19937ar','Seed',0) -- see
        # ppo/matlab_rng.py. The subset bounds are the hard-coded 0..W / 0..H
        # of ppo_world_setup.m:56, and generate_user_positions then clamps the
        # minima up to 1, so the draw is randi([1,W]) x randi([1,H]).
        self.user_positions = matlab_user_positions(
            0, world.width, 0, world.height, world.num_users, seed=0
        ).astype(float)
        self._user_x = self.user_positions[:, 0]
        self._user_y = self.user_positions[:, 1]

        # -- MBS power maps + their (constant) sampled columns ---------- #
        self._mbs_columns = self._precompute_mbs_columns()

        self._column_cache: "OrderedDict[MapSpec, np.ndarray]" = OrderedDict()
        self._column_cache_size = max(0, int(column_cache_size))

    # ------------------------------------------------------------------ #
    # World construction
    # ------------------------------------------------------------------ #
    def _precompute_mbs_columns(self) -> np.ndarray:
        """One sampled power column per (band, site), shape ``(n_bands, num_mbs, N)``.

        Mirrors ``pack_mbs_params`` + the row swap + ``precompute_mbs_power_maps``.
        The maps themselves are transient here: the users are fixed, so the only
        thing any step ever needs is the ``num_users``-vector each map is read
        into. The maps stay on disk for the next process; the columns stay in
        RAM for this one.
        """
        columns = np.zeros((self.n_bands, self.num_mbs, self.world.num_users), dtype=float)
        for site in range(self.num_mbs):
            for band_row, freq in enumerate(self.band_freqs):
                spec = self._mbs_spec(site, freq)
                grid = load_or_compute_map(spec, self.cache_dir)
                columns[band_row, site] = self._mbs_column(grid)
        return columns

    def _mbs_spec(self, site: int, center_freq: float) -> MapSpec:
        """The MBS power_map call, in the SWAPPED frame.

        ``ppo_world_setup.m:54-61`` packs ``[x; y; z; power]`` and then swaps
        rows 1 and 2, so ``precompute_mbs_power_maps.m:80`` sets
        ``tx_position = [true_y; true_x; height]``. Unlike the FBS path, this
        one honours the world's scenario / map_mode / ue_height
        (ppo_world_setup.m:64-65).
        """
        return MapSpec(
            scenario=self.world.scenario,
            map_mode=self.world.map_mode,
            tx_x=float(self.mbs_y[site]),  # <-- row swap: map-frame X is TRUE Y
            tx_y=float(self.mbs_x[site]),  # <-- map-frame Y is TRUE X
            tx_z=float(self.world.mbs_height),
            tx_power=float(self.world.mbs_power),
            center_freq=float(center_freq),
            x_min=0.0,
            x_max=float(self.world.width),
            y_min=0.0,
            y_max=float(self.world.height),
            sample_distance=SAMPLE_DISTANCE,
            rx_height=float(self.world.ue_height),
        )

    def _mbs_column(self, grid: np.ndarray) -> np.ndarray:
        """Sample an MBS map the way SINREvaluation.m:253 does.

        ``grid`` arrives as ``(n_y, n_x)`` -- ``precompute_mbs_power_maps.m:87``
        has no transpose, so this *is* the MATLAB layout, and it goes into
        ``sample_nearest`` verbatim. The swapped frame (the map was computed at
        ``[true_y, true_x]``) then cancels against ``sample_nearest``'s
        ``P(x_index, y_index)`` indexing -- except for the clamp, which is
        computed against the swapped extent and therefore folds every user with
        ``round(x) > height`` onto the ``x == height`` line. That is the
        reference behaviour; see :func:`sample_nearest`.
        """
        return sample_nearest(grid, self._user_x, self._user_y).astype(float)

    # ------------------------------------------------------------------ #
    # Per-step FBS physics
    # ------------------------------------------------------------------ #
    def _grid_nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """The exact grid coordinates ``sample_nearest`` reads for an FBS map.

        An FBS map is transposed to ``(n_x, n_y)`` (SINREvaluation.m:188), so
        ``sample_nearest`` returns ``P[iy - 1, ix - 1]`` of the untransposed
        map with ``ix = clip(round(x) + 1, 1, n_x)`` and
        ``iy = clip(round(y) + 1, 1, n_y)``. With ``x_min = y_min = 0`` and a
        1 m step those indices correspond to the coordinates below.
        """
        n_x = int(np.floor(self.world.width / SAMPLE_DISTANCE)) + 1
        n_y = int(np.floor(self.world.height / SAMPLE_DISTANCE)) + 1
        gx = np.clip(np.floor(self._user_x + 0.5).astype(np.int64), 0, n_x - 1)
        gy = np.clip(np.floor(self._user_y + 0.5).astype(np.int64), 0, n_y - 1)
        return gx * SAMPLE_DISTANCE, gy * SAMPLE_DISTANCE

    def _fbs_spec(self, x: float, y: float, z: float, power: float, freq: float) -> MapSpec:
        """The FBS power_map call (SINREvaluation.m:183-186).

        No row swap here: ``calculate_power`` is handed the true ``(x, y)``.
        Scenario, mode and receiver height are the MATLAB literals, *not* the
        world config -- see :data:`FBS_SCENARIO`.
        """
        return MapSpec(
            scenario=FBS_SCENARIO,
            map_mode=FBS_MAP_MODE,
            tx_x=float(x),
            tx_y=float(y),
            tx_z=float(z),
            tx_power=float(power),
            center_freq=float(freq),
            x_min=0.0,
            x_max=float(self.world.width),
            y_min=0.0,
            y_max=float(self.world.height),
            sample_distance=SAMPLE_DISTANCE,
            rx_height=FBS_UE_HEIGHT,
        )

    def _fbs_column(self, spec: MapSpec) -> np.ndarray:
        """Power received by every user from one active FBS, in mW."""
        cached = self._column_cache.get(spec)
        if cached is not None:
            self._column_cache.move_to_end(spec)
            return cached

        if self.fast_sampling:
            # The invariants below are what make this branch equal to the one
            # under it. They are cheap, they are not enforced anywhere else,
            # and breaking them is silent -- so check every call.
            check_fast_sampling_invariants(spec, self.world.width, self.world.height)
            gx, gy = self._grid_nodes()
            column = sample_power_at(spec, gx, gy)
        else:
            # Faithful path: the full grid, then the trailing transpose of
            # SINREvaluation.m:188 that makes the array (n_x, n_y).
            grid = compute_power_map(spec)
            column = sample_nearest(grid.T, self._user_x, self._user_y)
        column = np.asarray(column, dtype=float)

        if self._column_cache_size:
            self._column_cache[spec] = column
            while len(self._column_cache) > self._column_cache_size:
                self._column_cache.popitem(last=False)
        return column

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        cont_params: np.ndarray,
        power_status: np.ndarray,
        fbs_band_flags: Optional[np.ndarray] = None,
        mbs_capacity_genes: Optional[np.ndarray] = None,
        collect_user_positions: bool = False,
    ) -> SinrResult:
        """One SINR evaluation -- ppo_sinr_eval.m + SINREvaluation.m in numpy.

        Args:
            cont_params: ``(num_fbs, 4)`` of ``[x, y, z, power]``.
            power_status: one entry per FBS. MATLAB truthiness: any non-zero
                value switches the FBS on (SINREvaluation.m:208).
            fbs_band_flags: multi-band only. One entry per FBS, expected to be
                exactly ``0.0`` (coverage) or ``1.0`` (capacity); ``None`` or
                an empty array means all-coverage, as in ppo_sinr_eval.m:36.
                Values strictly between the two are *not* MATLAB-equivalent:
                ppo_sinr_eval.m:52 feeds the raw flag into ``bsBandIds`` and
                compares band ids for exact equality, so a flag of 0.7 would
                there put that FBS on a private band that interferes with
                nothing, whereas the ``>= 0.5`` threshold below folds it into
                the capacity band. The env only ever emits 0.0/1.0, and the
                threshold is what selects the antenna and carrier in MATLAB
                too, so this only matters to a caller replaying non-binary
                genes.
            mbs_capacity_genes: multi-band only. One entry per MBS site; the
                capacity slot of site ``j`` is active iff ``genes[j] >= 0.5``
                (ppo_sinr_eval.m:53). ``None``/empty means all-off.
            collect_user_positions: attach the (fixed) user map to the result.
        """
        cont = np.asarray(cont_params, dtype=float).reshape(-1, 4)
        status = np.asarray(power_status, dtype=float).reshape(-1)
        if status.size != cont.shape[0]:
            raise ValueError("power_status must have one entry per FBS")
        n_fbs = cont.shape[0]
        n_users = self.world.num_users

        # -- band bookkeeping (ppo_sinr_eval.m:34-58 / :74-76) ---------- #
        if self.band.is_multi:
            flags = band_vector(fbs_band_flags, n_fbs, "fbs_band_flags", "FBS")
            genes = band_vector(
                mbs_capacity_genes, self.num_mbs, "mbs_capacity_genes", "MBS site"
            )
        else:
            # Legacy passes bsBandIds = zeros(1, nFbs + numMbs): every
            # transmitter shares one band, so everything interferes with
            # everything (ppo_sinr_eval.m:76).
            flags = np.zeros(n_fbs)
            genes = np.zeros(self.num_mbs)

        fbs_is_capacity = flags >= 0.5
        fbs_band_ids = fbs_is_capacity.astype(int)
        slots = build_slots(self.num_mbs, self.band.is_multi, genes)
        band_ids = np.concatenate(
            [fbs_band_ids, np.array([s.band_id for s in slots], dtype=int)]
        ).astype(int)
        n_cols = n_fbs + len(slots)

        # -- power matrix (SINREvaluation.m:203-253) -------------------- #
        power = np.zeros((n_users, n_cols), dtype=float)
        for i in range(n_fbs):
            # SINREvaluation.m:208 is `if power_status(fbs_id)` -- MATLAB
            # truthiness, not a >= 0.5 threshold (which is what the MBS slot
            # flag and the band genes use). Inactive FBSs keep a zero column
            # rather than being dropped, so the column layout is stable.
            if status[i] == 0.0:
                continue
            freq = self.band_freqs[1] if fbs_is_capacity[i] else self.band_freqs[0]
            spec = self._fbs_spec(cont[i, 0], cont[i, 1], cont[i, 2], cont[i, 3], freq)
            power[:, i] = self._fbs_column(spec)

        if self.contains_mbs:
            for k, slot in enumerate(slots):
                if slot.active:
                    power[:, n_fbs + k] = self._mbs_columns[slot.band_row, slot.site]

        # -- association (SINREvaluation.m:259-287) --------------------- #
        # Columns are visited in order and the comparison is a STRICT '>', so
        # an exact SINR tie is won by the earlier column. Preserve both.
        threshold = float(self.world.sinr_threshold)
        best_db = np.full(n_users, -np.inf)
        best_col = np.zeros(n_users, dtype=np.int64)  # 1-based; 0 = unconnected
        sinr_db = np.empty((n_users, n_cols), dtype=float)
        for c in range(n_cols):
            same_band = band_ids == band_ids[c]
            interference = power[:, same_band].sum(axis=1) - power[:, c]
            with np.errstate(divide="ignore", invalid="ignore"):
                # A zero column gives 10*log10(0) = -inf, exactly as in MATLAB.
                # Do NOT floor it the way AnalyticSinrBackend does: -inf never
                # clears the threshold either way, but the exported per-column
                # SINR values would differ.
                db = 10.0 * np.log10(power[:, c] / (interference + NOISE_MW))
            sinr_db[:, c] = db
            better = (db >= threshold) & (db > best_db)
            best_db[better] = db[better]
            best_col[better] = c + 1
        connected = best_db >= threshold

        # -- counts and tier split (SINREvaluation.m:98-118) ------------ #
        total_connected = int(np.count_nonzero(connected))
        fbs_connected = int(
            np.count_nonzero(connected & (best_col >= 1) & (best_col <= n_fbs))
        )
        mbs_mask = connected & (best_col > n_fbs)
        slot_index = best_col[mbs_mask] - n_fbs - 1
        if slot_index.size:
            band_row_of = np.array([s.band_row for s in slots], dtype=int)
            served_rows = band_row_of[slot_index]
            cov_connected = int(np.count_nonzero(served_rows == 0))
            cap_connected = int(np.count_nonzero(served_rows == 1))
        else:
            cov_connected = 0
            cap_connected = 0

        # -- rates (SINREvaluation.m:397-489) --------------------------- #
        if total_connected == 0:
            # MATLAB returns NaN for the average and 0 (not NaN) for the sum.
            # AnalyticSinrBackend returns 0.0 for both; follow the MATLAB.
            # Provenance: this branch is read off SINREvaluation.m's
            # global_avg_rate_connected (sum/0 -> NaN, an empty sum -> 0), not
            # off replay evidence -- an always-on MBS coverage slot means no
            # archived run ever logged a step with nothing connected. Pinned by
            # test_avg_rate_is_nan_when_nothing_connects.
            avg_rate = float("nan")
            sum_rate = 0.0
        else:
            served = sinr_db[np.nonzero(connected)[0], best_col[connected] - 1]
            rates = np.log2(1.0 + 10.0 ** (served / 10.0))
            sum_rate = float(np.nansum(rates))
            denom = int(np.count_nonzero(~np.isnan(rates)))
            avg_rate = float("nan") if denom == 0 else sum_rate / denom

        # SINREvaluation.m:120-121: the power_status mask is COMMENTED OUT, so
        # this sums every FBS power gene including the inactive ones. Verified
        # against the archived GA metrics; do not "fix" it.
        total_power = float(np.sum(cont[:, 3])) if n_fbs else 0.0

        return SinrResult(
            total_connected=total_connected,
            fbs_connected=fbs_connected,
            mbs_connected=cov_connected + cap_connected,
            mbs_coverage_connected=cov_connected,
            mbs_capacity_connected=cap_connected,
            total_power=total_power,
            avg_rate=avg_rate,
            sum_rate=sum_rate,
            user_positions=self.user_positions.copy() if collect_user_positions else None,
        )

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """No engine, no subprocess, no file handle -- just drop the columns."""
        self._column_cache.clear()
