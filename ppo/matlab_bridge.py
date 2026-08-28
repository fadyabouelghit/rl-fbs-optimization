"""SINR evaluation backends.

Two interchangeable backends drive the gym environment:

- ``MatlabSinrBackend``: the real physics. Builds the world once MATLAB-side
  (``ppo_world_setup.m``) and evaluates steps through ``ppo_sinr_eval.m``, so
  each step marshals only scalars across the engine boundary (the legacy
  implementation shipped the full MBS map cache both ways every step).
- ``AnalyticSinrBackend``: a fast, deterministic, MATLAB-free stand-in with
  the same interface and band semantics (log-distance path loss, same-band
  interference, max-SINR association). Powers the test suite, smoke runs,
  and pipeline development at ~10k steps/s.

``MatlabSession`` manages engine lifecycles. For quick interactive testing it
can attach to an already-running shared MATLAB (run ``matlab.engine.shareEngine``
in your MATLAB console once) instead of paying the ~20 s cold start.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from .config import BAND_CAPACITY, BAND_COVERAGE, BandConfig, WorldConfig
from .paths import CACHE_DIR, MATLAB_PATH, require_quadriga_path


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class SinrResult:
    total_connected: int
    fbs_connected: int
    mbs_connected: int
    mbs_coverage_connected: int
    mbs_capacity_connected: int
    total_power: float
    avg_rate: float
    sum_rate: float
    user_positions: Optional[np.ndarray] = None

    def as_metrics_dict(self) -> dict:
        return {
            "total_connected": self.total_connected,
            "fbs_connected": self.fbs_connected,
            "mbs_connected": self.mbs_connected,
            "mbs_coverage_connected": self.mbs_coverage_connected,
            "mbs_capacity_connected": self.mbs_capacity_connected,
            "total_power": self.total_power,
            "avg_rate": self.avg_rate,
            "sum_rate": self.sum_rate,
        }


@runtime_checkable
class SinrBackend(Protocol):
    world: WorldConfig
    band: BandConfig
    mbs_x: np.ndarray
    mbs_y: np.ndarray

    def evaluate(
        self,
        cont_params: np.ndarray,
        power_status: np.ndarray,
        fbs_band_flags: Optional[np.ndarray] = None,
        mbs_capacity_genes: Optional[np.ndarray] = None,
        collect_user_positions: bool = False,
    ) -> SinrResult: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# MATLAB session management
# --------------------------------------------------------------------------- #
class MatlabSession:
    """A MATLAB engine with the QuaDRiGa + repo paths configured."""

    def __init__(self, engine, owns_engine: bool = True):
        self.engine = engine
        self._owns_engine = owns_engine
        self._closed = False

    # -- constructors --------------------------------------------------- #
    @classmethod
    def start(
        cls,
        quadriga_path: str | Path | None = None,
        repo_path: str | Path | None = None,
    ) -> "MatlabSession":
        import matlab.engine  # lazy: analytic-only workflows never need it

        t0 = time.time()
        eng = matlab.engine.start_matlab()
        session = cls(eng, owns_engine=True)
        session._add_paths(quadriga_path, repo_path)
        session.startup_seconds = time.time() - t0
        return session

    @classmethod
    def connect(
        cls,
        name: Optional[str] = None,
        quadriga_path: str | Path | None = None,
        repo_path: str | Path | None = None,
    ) -> "MatlabSession":
        """Attach to a shared MATLAB session (`matlab.engine.shareEngine`).

        Near-instant compared to a cold start; the attached MATLAB survives
        this Python process, so we never quit it on close().
        """
        import matlab.engine

        names = matlab.engine.find_matlab()
        if not names:
            raise RuntimeError(
                "No shared MATLAB session found. In a MATLAB console run:\n"
                "    matlab.engine.shareEngine\n"
                "or use MatlabSession.start() for a private engine."
            )
        eng = matlab.engine.connect_matlab(name or names[0])
        session = cls(eng, owns_engine=False)
        session._add_paths(quadriga_path, repo_path)
        session.startup_seconds = 0.0
        return session

    def _add_paths(self, quadriga_path=None, repo_path=None):
        # Resolved here, not at import: the analytic backend must stay usable
        # on a machine that has neither MATLAB nor QuaDRiGa configured.
        self.engine.addpath(str(require_quadriga_path(quadriga_path)), nargout=0)
        self.engine.addpath(str(Path(repo_path) if repo_path else MATLAB_PATH), nargout=0)

    def close(self) -> None:
        if self._closed or self.engine is None:
            return
        if self._owns_engine:
            try:
                self.engine.quit()
            except Exception:
                pass
        self.engine = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


_SHARED_SESSION: Optional[MatlabSession] = None


def get_shared_session(prefer_connect: bool = True) -> MatlabSession:
    """Module-level session reuse: repeated env/backend constructions in one
    process (e.g. notebook train -> eval -> eval) share a single engine."""
    global _SHARED_SESSION
    if _SHARED_SESSION is not None and not _SHARED_SESSION._closed:
        return _SHARED_SESSION
    if prefer_connect:
        try:
            _SHARED_SESSION = MatlabSession.connect()
            return _SHARED_SESSION
        except Exception:
            pass
    _SHARED_SESSION = MatlabSession.start()
    return _SHARED_SESSION


def close_shared_session() -> None:
    global _SHARED_SESSION
    if _SHARED_SESSION is not None:
        _SHARED_SESSION.close()
        _SHARED_SESSION = None


# --------------------------------------------------------------------------- #
# MATLAB-backed physics
# --------------------------------------------------------------------------- #
class MatlabSinrBackend:
    """Real SINR evaluation through the MATLAB/QuaDRiGa stack."""

    _world_counter = 0

    def __init__(
        self,
        world: WorldConfig,
        band: BandConfig,
        session: Optional[MatlabSession] = None,
        cache_dir: str | Path = CACHE_DIR,
        world_id: Optional[str] = None,
        quadriga_path: str | Path | None = None,
        repo_path: str | Path | None = None,
    ):
        self.world = world
        self.band = band
        if session is None:
            session = MatlabSession.start(quadriga_path, repo_path)
            self._owns_session = True
        else:
            self._owns_session = False
        self.session = session

        MatlabSinrBackend._world_counter += 1
        self.world_id = world_id or f"w{MatlabSinrBackend._world_counter}"

        payload = {
            "width": world.width,
            "height": world.height,
            "num_mbs": world.num_mbs,
            "mbs_locations": (
                [list(p) for p in world.mbs_locations]
                if world.mbs_locations is not None
                else []
            ),
            "isd": world.isd,
            "margin": world.margin,
            "mbs_height": world.mbs_height,
            "mbs_power": world.mbs_power,
            "ue_height": world.ue_height,
            "scenario": world.scenario,
            "map_mode": world.map_mode,
            "cache_dir": str(cache_dir),
            "band_mode": band.mode,
        }
        info = self.session.engine.ppo_world_setup(
            json.dumps(payload), self.world_id, nargout=1
        )
        self.mbs_x = np.asarray(info["mbs_x"], dtype=float).ravel()
        self.mbs_y = np.asarray(info["mbs_y"], dtype=float).ravel()
        self.num_mbs = int(info["num_mbs"])
        self.contains_mbs = bool(info["contains_mbs"])
        self.n_bands = int(info["n_bands"])

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        cont_params: np.ndarray,
        power_status: np.ndarray,
        fbs_band_flags: Optional[np.ndarray] = None,
        mbs_capacity_genes: Optional[np.ndarray] = None,
        collect_user_positions: bool = False,
    ) -> SinrResult:
        import matlab

        cont = np.asarray(cont_params, dtype=float).reshape(-1, 4)
        status = np.asarray(power_status, dtype=float).reshape(-1)
        if status.size != cont.shape[0]:
            raise ValueError("power_status must have one entry per FBS")

        tx_mat = matlab.double(cont.tolist())
        ps_mat = matlab.double([status.tolist()])
        flags_mat = matlab.double(
            [] if fbs_band_flags is None
            else [np.asarray(fbs_band_flags, dtype=float).reshape(-1).tolist()]
        )
        genes_mat = matlab.double(
            [] if mbs_capacity_genes is None
            else [np.asarray(mbs_capacity_genes, dtype=float).reshape(-1).tolist()]
        )

        out = self.session.engine.ppo_sinr_eval(
            tx_mat,
            ps_mat,
            flags_mat,
            genes_mat,
            float(self.world.num_users),
            float(self.world.sinr_threshold),
            bool(collect_user_positions),
            self.world_id,
            nargout=1,
        )

        user_positions = None
        if collect_user_positions and "user_positions" in out:
            user_positions = np.asarray(out["user_positions"], dtype=float)
            user_positions = user_positions.reshape(-1, 2)

        return SinrResult(
            total_connected=int(out["total_connected"]),
            fbs_connected=int(out["fbs_connected"]),
            mbs_connected=int(out["mbs_connected"]),
            mbs_coverage_connected=int(out["mbs_coverage_connected"]),
            mbs_capacity_connected=int(out["mbs_capacity_connected"]),
            total_power=float(out["total_power"]),
            avg_rate=float(out["avg_rate"]),
            sum_rate=float(out["sum_rate"]),
            user_positions=user_positions,
        )

    def close(self) -> None:
        if self._owns_session and self.session is not None:
            self.session.close()
        self.session = None


# --------------------------------------------------------------------------- #
# Analytic (MATLAB-free) physics
# --------------------------------------------------------------------------- #
class AnalyticSinrBackend:
    """Deterministic log-distance path-loss stand-in for SINREvaluation.

    Semantics mirrored from the MATLAB side:
      - users drawn once with a fixed seed (fixed demand map),
      - transmit powers treated as dBm, received power via log-distance decay,
      - interference accumulated over same-band transmitters only,
      - a user connects to its max-SINR BS when SINR >= threshold (dB),
      - MBS sites expand into coverage/capacity slots in 'multi' mode,
      - ``total_power`` is the sum of active-FBS power genes (as in
        SINREvaluation's total_transmitted_pwr output).

    Numbers are not calibrated to QuaDRiGa — this backend exists so env
    mechanics, logging, training loops, and plots can run in milliseconds.
    """

    PATH_LOSS_EXP = 3.0
    NOISE_MW = 1e-11
    MBS_ANTENNA_GAIN = 4.0  # crude macro advantage over an FBS

    def __init__(self, world: WorldConfig, band: BandConfig, user_seed: int = 0):
        self.world = world
        self.band = band
        rng = np.random.default_rng(user_seed)
        self.user_positions = np.column_stack(
            [
                rng.uniform(0.0, world.width, size=world.num_users),
                rng.uniform(0.0, world.height, size=world.num_users),
            ]
        )
        if world.mbs_locations is not None:
            coords = np.asarray(world.mbs_locations, dtype=float)
        else:
            coords = self._hex_like_sites(world)
        self.mbs_x = coords[:, 0].copy()
        self.mbs_y = coords[:, 1].copy()
        self.num_mbs = coords.shape[0]
        self.contains_mbs = self.num_mbs > 0
        self.n_bands = 2 if band.is_multi else 1

    @staticmethod
    def _hex_like_sites(world: WorldConfig) -> np.ndarray:
        """Center-out placement standing in for generate_hex_sites."""
        cx, cy = world.width / 2.0, world.height / 2.0
        if world.num_mbs == 1:
            return np.array([[cx, cy]])
        angles = np.linspace(0.0, 2 * np.pi, world.num_mbs, endpoint=False)
        return np.column_stack(
            [cx + world.isd * np.cos(angles), cy + world.isd * np.sin(angles)]
        )

    # ------------------------------------------------------------------ #
    def _rx_power_mw(self, tx_xy_z: np.ndarray, tx_power_dbm: float, gain: float = 1.0):
        d_sq = (
            (self.user_positions[:, 0] - tx_xy_z[0]) ** 2
            + (self.user_positions[:, 1] - tx_xy_z[1]) ** 2
            + tx_xy_z[2] ** 2
        )
        d = np.sqrt(np.maximum(d_sq, 1.0))
        p_mw = 10.0 ** (tx_power_dbm / 10.0)
        return gain * p_mw / (d ** self.PATH_LOSS_EXP)

    def evaluate(
        self,
        cont_params: np.ndarray,
        power_status: np.ndarray,
        fbs_band_flags: Optional[np.ndarray] = None,
        mbs_capacity_genes: Optional[np.ndarray] = None,
        collect_user_positions: bool = False,
    ) -> SinrResult:
        cont = np.asarray(cont_params, dtype=float).reshape(-1, 4)
        status = np.asarray(power_status, dtype=float).reshape(-1)
        n_fbs = cont.shape[0]

        if self.band.is_multi:
            flags = (
                np.zeros(n_fbs)
                if fbs_band_flags is None
                else np.asarray(fbs_band_flags, dtype=float).reshape(-1)
            )
            genes = (
                np.zeros(self.num_mbs)
                if mbs_capacity_genes is None
                else np.asarray(mbs_capacity_genes, dtype=float).reshape(-1)
            )
        else:
            flags = np.zeros(n_fbs)
            genes = np.zeros(self.num_mbs)

        columns, band_ids, kinds = [], [], []
        # FBS columns
        for i in range(n_fbs):
            if status[i] >= 0.5:
                col = self._rx_power_mw(cont[i, :3], cont[i, 3])
            else:
                col = np.zeros(len(self.user_positions))
            columns.append(col)
            band_ids.append(int(flags[i] >= 0.5))
            kinds.append("fbs")
        # MBS slot columns
        for j in range(self.num_mbs):
            site = np.array([self.mbs_x[j], self.mbs_y[j], self.world.mbs_height])
            cov = self._rx_power_mw(site, self.world.mbs_power, self.MBS_ANTENNA_GAIN)
            if self.band.is_multi:
                columns.append(cov)
                band_ids.append(BAND_COVERAGE)
                kinds.append("mbs_cov")
                cap = (
                    self._rx_power_mw(site, self.world.mbs_power, self.MBS_ANTENNA_GAIN)
                    if genes[j] >= 0.5
                    else np.zeros(len(self.user_positions))
                )
                columns.append(cap)
                band_ids.append(BAND_CAPACITY)
                kinds.append("mbs_cap")
            else:
                columns.append(cov)
                band_ids.append(BAND_COVERAGE)
                kinds.append("mbs_cov")

        power = np.column_stack(columns) if columns else np.zeros((len(self.user_positions), 0))
        band_ids = np.asarray(band_ids)

        best_sinr_db = np.full(len(self.user_positions), -np.inf)
        best_col = np.full(len(self.user_positions), -1)
        best_sinr_lin = np.zeros(len(self.user_positions))
        for c in range(power.shape[1]):
            same_band = band_ids == band_ids[c]
            interference = power[:, same_band].sum(axis=1) - power[:, c]
            sinr = power[:, c] / (interference + self.NOISE_MW)
            sinr_db = 10.0 * np.log10(np.maximum(sinr, 1e-30))
            better = (sinr_db >= self.world.sinr_threshold) & (sinr_db > best_sinr_db)
            best_sinr_db[better] = sinr_db[better]
            best_col[better] = c
            best_sinr_lin[better] = sinr[better]

        connected = best_col >= 0
        kinds_arr = np.asarray(kinds)
        served_kind = np.where(connected, kinds_arr[best_col.clip(min=0)], "none")
        fbs_connected = int(np.sum(served_kind == "fbs"))
        cov_connected = int(np.sum(served_kind == "mbs_cov"))
        cap_connected = int(np.sum(served_kind == "mbs_cap"))
        total_connected = int(np.sum(connected))

        rates = np.log2(1.0 + best_sinr_lin[connected])
        sum_rate = float(rates.sum()) if total_connected else 0.0
        avg_rate = float(rates.mean()) if total_connected else 0.0
        total_power = float(np.sum(cont[status >= 0.5, 3])) if n_fbs else 0.0

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

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def make_backend(
    world: WorldConfig,
    band: BandConfig,
    backend: str = "matlab",
    session: Optional[MatlabSession] = None,
) -> SinrBackend:
    if backend == "matlab":
        return MatlabSinrBackend(world, band, session=session)
    if backend == "analytic":
        return AnalyticSinrBackend(world, band)
    raise ValueError(f"unknown backend {backend!r}; expected 'matlab' or 'analytic'")
