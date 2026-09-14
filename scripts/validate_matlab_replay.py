#!/usr/bin/env python
"""End-to-end validation: replay logged MATLAB training steps through PyqdSinrBackend.

A completed MATLAB-backed PPO run logs, for every training step, BOTH the full
FBS state (``fbs<i>_x/_y/_height/_power/_power_status``) and the physics metrics
MATLAB computed from that state (``total_connected``, ``fbs_connected``,
``mbs_connected``, ``total_power``, ``avg_rate``, ``sum_rate``). Replaying those
states through the MATLAB-free :class:`ppo.pyqd_bridge.PyqdSinrBackend` and
recovering the same metrics is the decisive test of the port: it exercises the
whole stack at once (hex-site placement, the x<->y row swap, the transpose
asymmetry, ``sample_nearest``'s clamp, the MATLAB user RNG, the association
loop's strict ``>``, the un-masked power sum, and the pyqd power maps).

What "pass" means
-----------------
``total_connected`` / ``fbs_connected`` are integer counts produced by
thresholding a continuous SINR at 5 dB. pyqd reproduces MATLAB QuaDRiGa power
maps to ~3e-6 dB, so a user whose SINR sits within a few 1e-6 dB of the
threshold -- or of a tie between two candidate serving cells -- can legitimately
flip. A handful of +-1 count differences is therefore expected. A *transpose* or
*coordinate-frame* error, the failure mode this script really hunts, would show
up as large, systematic deviations instead.

So the script does not just report a match rate. For every mismatching step it
recomputes the per-user, per-column SINR and measures how close the flipped
users actually were to the decision boundary, which distinguishes knife-edge
numerical noise from a real bug.

Usage
-----
    venv/bin/python scripts/validate_matlab_replay.py                  # defaults
    venv/bin/python scripts/validate_matlab_replay.py --steps 80
    venv/bin/python scripts/validate_matlab_replay.py --fast-sampling  # ~160x faster
    venv/bin/python scripts/validate_matlab_replay.py --run /path/to/run_dir
    venv/bin/python scripts/validate_matlab_replay.py --out /tmp/replay.csv

Exit status is 0 when every check passes the tolerances below, 1 otherwise, so
this is usable as a regression gate.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ppo.config import load_experiment_config  # noqa: E402
from ppo.env import state_labels  # noqa: E402
from ppo.pyqd_bridge import NOISE_MW, PyqdSinrBackend, build_slots  # noqa: E402

#: The archived MATLAB run replayed by default. Code 1-1-1: 1 FBS, 1 MBS,
#: 2000x1500 world, 1000 users, 5 dB threshold, legacy (single) band.
DEFAULT_RUN = Path(
    "/Users/fadya/Documents/MATLAB/GA_github/genetic-algorithm-optimization"
    "/genetic-algorithm-optimization/ppo_runs"
    "/run_2026-08-19_23-46-16_1-1-1_1fbs1mbs_potrec10"
)

# -- pass/fail tolerances --------------------------------------------------- #
#: ``total_power`` is a plain sum of the logged power genes -- no physics, no
#: map -- so it must agree to float32 round-trip noise of the CSV.
TOL_TOTAL_POWER = 1e-6
#: Rates are continuous functions of the SINRs of the *connected* users, so
#: their accuracy is bounded by the power-map agreement. pyqd matches MATLAB
#: QuaDRiGa to 3.3e-6 dB, i.e. ~7.6e-7 relative in linear SINR, which
#: propagates to <=1e-6 relative in ``log2(1 + sinr)``. Observed is ~2e-9;
#: the budget is set at the propagated bound so that a genuine physics
#: divergence still trips it.
TOL_RATE_REL = 1e-6
#: Fraction of sampled steps whose integer counts must match exactly.
MIN_EXACT_RATE = 0.90
#: A count difference larger than this is not knife-edge noise.
MAX_ABS_COUNT_DELTA = 2
#: A flipped user is "knife-edge" if its decision margin is under this many dB.
KNIFE_EDGE_DB = 1e-3


# --------------------------------------------------------------------------- #
# Step selection
# --------------------------------------------------------------------------- #
def select_steps(df: pd.DataFrame, n_steps: int, seed: int) -> np.ndarray:
    """Row indices to replay: head, tail, the argmax, and a random middle.

    Deliberately not a uniform random sample. The head of a PPO run is an
    untrained policy parked in a corner of the world, the tail is a converged
    policy, and the ``fbs_connected`` argmax is the single most
    physics-sensitive row in the file (the FBS is winning the most users from
    the MBS, so the most users sit near an association boundary). Both
    ``power_status`` values are forced in because the 0 branch takes a
    completely different code path (an all-zero power column instead of a
    power map).
    """
    n = len(df)
    n_edge = max(3, n_steps // 8)
    picks: list[int] = []
    picks += list(range(min(n_edge, n)))  # first few steps
    picks += list(range(max(0, n - n_edge), n))  # last few steps
    picks.append(int(df["fbs_connected"].idxmax()))  # hardest row in the file
    picks.append(int(df["total_connected"].idxmax()))
    picks.append(int(df["total_connected"].idxmin()))

    rng = np.random.default_rng(seed)
    for status in (0.0, 1.0):
        pool = np.flatnonzero(df["fbs0_power_status"].to_numpy() == status)
        if pool.size:
            take = min(pool.size, max(2, n_steps // 8))
            picks += rng.choice(pool, size=take, replace=False).tolist()

    remaining = n_steps - len(set(picks))
    if remaining > 0:
        middle = np.arange(n_edge, max(n_edge, n - n_edge))
        middle = np.setdiff1d(middle, np.asarray(picks, dtype=int))
        if middle.size:
            picks += rng.choice(
                middle, size=min(remaining, middle.size), replace=False
            ).tolist()
    return np.array(sorted(set(int(i) for i in picks)), dtype=int)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def sinr_matrix(
    backend: PyqdSinrBackend,
    cont: np.ndarray,
    status: np.ndarray,
    flags: np.ndarray | None = None,
    genes: np.ndarray | None = None,
):
    """Recompute the per-user, per-column SINR (dB) for one state.

    Uses the backend's own column builders, so this measures the very numbers
    ``evaluate`` decided on rather than an independent re-derivation. Handles
    both band modes: in ``legacy`` every column shares one band and therefore
    interferes with every other, in ``multi`` interference is accumulated only
    within a band id.
    """
    n_fbs = cont.shape[0]
    if flags is None:
        flags = np.zeros(n_fbs)
    if genes is None:
        genes = np.zeros(backend.num_mbs)
    is_multi = backend.band.is_multi
    if not is_multi:
        flags = np.zeros(n_fbs)
        genes = np.zeros(backend.num_mbs)

    fbs_is_cap = flags >= 0.5
    slots = build_slots(backend.num_mbs, is_multi, genes)
    band_ids = np.concatenate(
        [fbs_is_cap.astype(int), np.array([s.band_id for s in slots], dtype=int)]
    )
    n_cols = n_fbs + len(slots)
    power = np.zeros((backend.world.num_users, n_cols))
    for i in range(n_fbs):
        if status[i] == 0.0:
            continue
        freq = backend.band_freqs[1] if fbs_is_cap[i] else backend.band_freqs[0]
        spec = backend._fbs_spec(cont[i, 0], cont[i, 1], cont[i, 2], cont[i, 3], freq)
        power[:, i] = backend._fbs_column(spec)
    if backend.contains_mbs:
        for k, slot in enumerate(slots):
            if slot.active:
                power[:, n_fbs + k] = backend._mbs_columns[slot.band_row, slot.site]

    db = np.empty_like(power)
    with np.errstate(divide="ignore", invalid="ignore"):
        for c in range(n_cols):
            same = band_ids == band_ids[c]
            interference = power[:, same].sum(axis=1) - power[:, c]
            db[:, c] = 10.0 * np.log10(power[:, c] / (interference + NOISE_MW))
    return db


def unpack_state(row: pd.Series, cfg) -> tuple:
    """Split one logged CSV row back into the arguments ``evaluate`` takes.

    Column names come from :func:`ppo.env.state_labels`, i.e. from the config
    that produced the run, so multi-FBS and multi-band layouts
    (``fbs<i>_band_flag``, ``mbs<j>_capacity``) are handled without guessing.
    """
    n_fbs = cfg.env.num_fbs
    cont = np.array(
        [[row[f"fbs{i}_{n}"] for n in ("x", "y", "height", "power")] for i in range(n_fbs)],
        dtype=float,
    )
    status = np.array([row[f"fbs{i}_power_status"] for i in range(n_fbs)], dtype=float)
    flags = (
        np.array([row[f"fbs{i}_band_flag"] for i in range(n_fbs)], dtype=float)
        if cfg.env.band.agent_controls_fbs_band
        else None
    )
    genes = (
        np.array(
            [row[f"mbs{j}_capacity"] for j in range(cfg.env.world.num_mbs)], dtype=float
        )
        if cfg.env.band.agent_controls_mbs_capacity
        else None
    )
    return cont, status, flags, genes


def knife_edge_counts(db: np.ndarray, threshold: float, eps: float) -> dict:
    """How many users sit within ``eps`` dB of a decision boundary.

    Two boundaries can flip a logged count:

    * the *threshold* -- any column within ``eps`` dB of ``threshold`` can
      gain or lose a user from ``total_connected``;
    * a *tie* between two above-threshold columns -- flips which tier serves
      the user, i.e. moves a count between ``fbs_connected`` and
      ``mbs_connected`` while leaving ``total_connected`` alone.
    """
    finite = np.where(np.isfinite(db), db, -np.inf)
    near_threshold = int(
        np.count_nonzero((np.abs(finite - threshold) < eps).any(axis=1))
    )
    above = np.where(finite >= threshold, finite, -np.inf)
    if above.shape[1] >= 2:
        srt = np.sort(above, axis=1)
        with np.errstate(invalid="ignore"):  # -inf - -inf when nothing connects
            gap = srt[:, -1] - srt[:, -2]
        near_tie = int(np.count_nonzero(np.isfinite(gap) & (gap < eps)))
    else:
        near_tie = 0
    return {"near_threshold": near_threshold, "near_tie": near_tie}


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN, help="MATLAB run directory")
    ap.add_argument("--steps", type=int, default=60, help="how many steps to replay")
    ap.add_argument("--seed", type=int, default=0, help="step-sampling seed")
    ap.add_argument(
        "--fast-sampling",
        action="store_true",
        help="evaluate the LOS model only at the sampled grid nodes (bit-identical)",
    )
    ap.add_argument("--out", type=Path, default=None, help="write per-step CSV here")
    args = ap.parse_args(argv)

    run_dir: Path = args.run
    cfg = load_experiment_config(run_dir / "experiment_config.json")
    df = pd.read_csv(run_dir / "steps.csv")

    world, band = cfg.env.world, cfg.env.band
    n_fbs = cfg.env.num_fbs
    print("=" * 78)
    print(f"REPLAY  {run_dir.name}")
    print(f"  code={cfg.code}  num_fbs={n_fbs}  band={band.mode}  rows={len(df)}")
    print(
        f"  world {world.width:g}x{world.height:g}  num_mbs={world.num_mbs}  "
        f"users={world.num_users}  threshold={world.sinr_threshold} dB"
    )
    print(f"  fast_sampling={args.fast_sampling}")
    print("=" * 78)

    t0 = time.time()
    backend = PyqdSinrBackend(world, band, fast_sampling=args.fast_sampling)
    print(
        f"backend built in {time.time() - t0:.2f}s  "
        f"mbs_x={np.round(backend.mbs_x, 6).tolist()} "
        f"mbs_y={np.round(backend.mbs_y, 6).tolist()}"
    )
    print(
        f"users: x in [{backend.user_positions[:,0].min():.0f},"
        f"{backend.user_positions[:,0].max():.0f}] "
        f"y in [{backend.user_positions[:,1].min():.0f},"
        f"{backend.user_positions[:,1].max():.0f}]"
    )

    missing = [c for c in state_labels(cfg.env) if c not in df.columns]
    if missing:
        raise SystemExit(f"steps.csv is missing state columns {missing}")

    rows = select_steps(df, args.steps, args.seed)
    n_active = df.loc[rows, [f"fbs{i}_power_status" for i in range(n_fbs)]].to_numpy()
    print(
        f"replaying {len(rows)} steps  "
        f"(num_active_fbs histogram: "
        f"{dict(zip(*[a.tolist() for a in np.unique(n_active.sum(1), return_counts=True)]))})\n"
    )

    records = []
    t0 = time.time()
    for k, idx in enumerate(rows):
        row = df.iloc[idx]
        cont, status, flags, genes = unpack_state(row, cfg)
        res = backend.evaluate(cont, status, flags, genes)

        rec = {
            "row": int(idx),
            "timesteps": int(row["timesteps"]),
            "episode": int(row["episode"]),
            "step": int(row["step"]),
            "power_status": float(status[0]),
            "n_active": float(status.sum()),
            "fbs_x": float(cont[0, 0]),
            "fbs_y": float(cont[0, 1]),
            "fbs_z": float(cont[0, 2]),
            "fbs_p": float(cont[0, 3]),
        }
        for name, got in (
            ("total_connected", res.total_connected),
            ("fbs_connected", res.fbs_connected),
            ("mbs_connected", res.mbs_connected),
            ("mbs_coverage_connected", res.mbs_coverage_connected),
            ("mbs_capacity_connected", res.mbs_capacity_connected),
            ("total_power", res.total_power),
            ("avg_rate", res.avg_rate),
            ("sum_rate", res.sum_rate),
        ):
            rec[f"m_{name}"] = float(row[name])
            rec[f"p_{name}"] = float(got)
        records.append(rec)
        if (k + 1) % 10 == 0 or k + 1 == len(rows):
            print(
                f"  ... {k + 1}/{len(rows)} replayed "
                f"({time.time() - t0:.1f}s, {(k + 1) / (time.time() - t0):.2f} steps/s)"
            )
    elapsed = time.time() - t0
    out = pd.DataFrame.from_records(records)

    # ---------------------------------------------------------------- #
    print(f"\nreplayed {len(out)} steps in {elapsed:.1f}s "
          f"({len(out) / elapsed:.2f} steps/s)\n")

    ok = True

    print("-" * 78)
    print("INTEGER COUNTS  (python - matlab)")
    print("-" * 78)
    print(f"{'metric':<26}{'exact':>10}{'rate':>9}{'min':>6}{'max':>6}{'mean':>9}"
          f"{'|d|>0':>7}")
    for name in (
        "total_connected",
        "fbs_connected",
        "mbs_connected",
        "mbs_coverage_connected",
        "mbs_capacity_connected",
    ):
        d = (out[f"p_{name}"] - out[f"m_{name}"]).to_numpy()
        exact = int(np.count_nonzero(d == 0))
        rate = exact / len(d)
        print(
            f"{name:<26}{exact:>7}/{len(d):<3}{rate:>8.1%}{d.min():>6.0f}"
            f"{d.max():>6.0f}{d.mean():>9.3f}{int(np.count_nonzero(d != 0)):>7}"
        )
        if rate < MIN_EXACT_RATE or np.abs(d).max() > MAX_ABS_COUNT_DELTA:
            ok = False

    print("\ndelta histogram (python - matlab):")
    for name in ("total_connected", "fbs_connected", "mbs_connected"):
        d = (out[f"p_{name}"] - out[f"m_{name}"]).to_numpy().astype(int)
        vals, cnts = np.unique(d, return_counts=True)
        hist = "  ".join(f"{v:+d}:{c}" for v, c in zip(vals, cnts))
        print(f"  {name:<24}{hist}")

    # invariant fbs + coverage + capacity == total (SINREvaluation.m:110-118)
    inv = (
        out["p_fbs_connected"]
        + out["p_mbs_coverage_connected"]
        + out["p_mbs_capacity_connected"]
        - out["p_total_connected"]
    ).abs().max()
    print(f"\ntier-split invariant  max|fbs+cov+cap - total| = {inv:.0f}"
          f"   {'OK' if inv == 0 else 'VIOLATED'}")
    if inv != 0:
        ok = False

    print("\n" + "-" * 78)
    print("FLOATS")
    print("-" * 78)
    print(f"{'metric':<16}{'n':>5}{'max abs err':>15}{'max rel err':>15}{'tol(rel)':>12}")
    for name, tol_abs, tol_rel in (
        ("total_power", TOL_TOTAL_POWER, None),
        ("avg_rate", None, TOL_RATE_REL),
        ("sum_rate", None, TOL_RATE_REL),
    ):
        m = out[f"m_{name}"].to_numpy()
        p = out[f"p_{name}"].to_numpy()
        good = np.isfinite(m) & np.isfinite(p)
        # Only judge rates on steps whose connected set matched exactly: a
        # legitimately flipped user changes the rate sum by a real amount, and
        # that is already accounted for in the count statistics above.
        if name in ("avg_rate", "sum_rate"):
            same = (
                (out["p_total_connected"] == out["m_total_connected"])
                & (out["p_fbs_connected"] == out["m_fbs_connected"])
            ).to_numpy()
            good &= same
        abs_err = np.abs(p[good] - m[good])
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err = np.where(m[good] != 0, abs_err / np.abs(m[good]), abs_err)
        mx_a = abs_err.max() if abs_err.size else 0.0
        mx_r = rel_err.max() if rel_err.size else 0.0
        tol = tol_abs if tol_rel is None else tol_rel
        print(f"{name:<16}{int(good.sum()):>5}{mx_a:>15.3e}{mx_r:>15.3e}{tol:>12.1e}")
        if tol_abs is not None and mx_a > tol_abs:
            ok = False
        if tol_rel is not None and mx_r > tol_rel:
            ok = False

    nan_mismatch = int(
        np.count_nonzero(
            np.isnan(out["m_avg_rate"].to_numpy()) != np.isnan(out["p_avg_rate"].to_numpy())
        )
    )
    print(f"avg_rate NaN-pattern mismatches: {nan_mismatch}")
    if nan_mismatch:
        ok = False

    # ---------------------------------------------------------------- #
    # Tier columns are included: a coverage<->capacity flip leaves every
    # headline count intact but is still a real disagreement, and it is the
    # one place the strict-'>' tie-break rule is observable.
    bad = out[
        (out["p_total_connected"] != out["m_total_connected"])
        | (out["p_fbs_connected"] != out["m_fbs_connected"])
        | (out["p_mbs_connected"] != out["m_mbs_connected"])
        | (out["p_mbs_coverage_connected"] != out["m_mbs_coverage_connected"])
        | (out["p_mbs_capacity_connected"] != out["m_mbs_capacity_connected"])
    ]
    print("\n" + "-" * 78)
    print(f"MISMATCHING STEPS: {len(bad)} of {len(out)}")
    print("-" * 78)
    if bad.empty:
        print("  (none)")
    else:
        print(
            f"{'row':>7}{'ts':>8}{'act':>5}{'x0':>10}{'y0':>9}{'z0':>8}{'p0':>7}"
            f"{'tot m/p':>13}{'fbs m/p':>11}{'cov m/p':>11}"
            f"{'near_thr':>10}{'near_tie':>10}"
        )
        unexplained = 0
        for _, r in bad.iterrows():
            src = df.iloc[int(r["row"])]
            db = sinr_matrix(backend, *unpack_state(src, cfg))
            ke = knife_edge_counts(db, world.sinr_threshold, KNIFE_EDGE_DB)
            print(
                f"{int(r['row']):>7}{int(r['timesteps']):>8}{int(r['n_active']):>5}"
                f"{r['fbs_x']:>10.2f}{r['fbs_y']:>9.2f}{r['fbs_z']:>8.2f}{r['fbs_p']:>7.2f}"
                f"{int(r['m_total_connected']):>7}/{int(r['p_total_connected']):<5}"
                f"{int(r['m_fbs_connected']):>6}/{int(r['p_fbs_connected']):<4}"
                f"{int(r['m_mbs_coverage_connected']):>6}/"
                f"{int(r['p_mbs_coverage_connected']):<4}"
                f"{ke['near_threshold']:>10}{ke['near_tie']:>10}"
            )
            d_tot = abs(r["p_total_connected"] - r["m_total_connected"])
            d_fbs = abs(r["p_fbs_connected"] - r["m_fbs_connected"])
            d_cov = abs(r["p_mbs_coverage_connected"] - r["m_mbs_coverage_connected"])
            budget = ke["near_threshold"] + ke["near_tie"]
            if d_tot > ke["near_threshold"] or d_fbs > budget or d_cov > budget:
                unexplained += 1
        print(
            f"\n  near_thr / near_tie = users within {KNIFE_EDGE_DB:g} dB of the "
            f"{world.sinr_threshold:g} dB threshold / of a tie between two\n"
            "  above-threshold columns. A count delta explained by these is "
            "numerical knife-edge, not a bug."
        )
        print(f"  steps whose delta is NOT covered by knife-edge users: {unexplained}")
        if unexplained:
            ok = False

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"\nper-step results -> {args.out}")

    print("\n" + "=" * 78)
    print("VERDICT:", "PASS" if ok else "FAIL")
    print("=" * 78)
    backend.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
