"""Cross-seed fast-vs-faithful equivalence replay for the pyqd backend.

Context. ``PyqdSinrBackend`` has two paths that differ in exactly one branch
(``_fbs_column``): the faithful path (``backend='pyqd'``) builds the full
2001x1501 sample_distance=1 m coverage map via ``pyqd_channel.layout.power_map``,
transposes it and reads it with ``sample_nearest``; the fast path
(``backend='pyqd-fast'``) skips the map and calls ``get_los_coeff`` directly at
the grid nodes the users round onto. They are claimed bit-identical.

That claim was previously verified only on ONE trajectory (seed 0 / potrec10):
51 archived MATLAB states plus a 13312-step training run on each path. This
script closes the gap by replaying states drawn from the seeds 1..9 training
logs (run on the FAST path only), which drive the FBS through regions of state
space seed 0 never visited.

Sampling per seed is deliberate, not a blind sweep (see ``select_states``):
first/last steps, per-gene extremes, both power_status values, the
fbs_connected extremes, world-boundary states (the most plausible divergence
point, since that is where index clamping could bite), plus a uniform random
sample under a fixed numpy seed and an active-only top-up.

Comparison is exact: integers with ``==``, floats bitwise (which is stricter
than atol=0/rtol=0 and additionally treats NaN==NaN as a match). Where a state
has an active FBS, the underlying 1000-user FBS power column of each path is
also compared bitwise -- it is recoverable from each backend's column cache
after the call, so this stronger check is free.

Usage:
    venv/bin/python comparison/cross_seed_fast_vs_faithful.py [--seeds 4 5 6]
                    [--per-seed 72] [--rng-seed 20260903] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ppo.config import load_experiment_config  # noqa: E402
from ppo.matlab_bridge import make_backend  # noqa: E402

GENES = ["fbs0_x", "fbs0_y", "fbs0_height", "fbs0_power"]
INT_FIELDS = [
    "total_connected",
    "fbs_connected",
    "mbs_connected",
    "mbs_coverage_connected",
    "mbs_capacity_connected",
]
FLOAT_FIELDS = ["total_power", "avg_rate", "sum_rate"]
BOUNDARY_M = 2.0


def run_dir(seed: int) -> Path:
    hits = sorted((REPO / "ppo_runs").glob(f"*potrec10_seed{seed}"))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one run dir for seed {seed}, got {hits}")
    return hits[0]


def boundary_distance(df: pd.DataFrame, width: float, height: float) -> np.ndarray:
    """Distance from the FBS to the nearest world edge, in metres."""
    return np.minimum(
        np.minimum(df.fbs0_x.values, width - df.fbs0_x.values),
        np.minimum(df.fbs0_y.values, height - df.fbs0_y.values),
    )


def select_states(df: pd.DataFrame, target: int, rng_seed: int,
                  width: float, height: float) -> list[tuple[int, list[str]]]:
    """Pick a diverse, reproducible set of row indices with their reasons.

    Every draw goes through one ``np.random.default_rng(rng_seed)``, so the
    selection is a pure function of (steps.csv, target, rng_seed).
    """
    rng = np.random.default_rng(rng_seed)
    reasons: dict[int, list[str]] = {}

    def add(idx, why: str) -> None:
        for i in np.atleast_1d(np.asarray(idx, dtype=int)):
            reasons.setdefault(int(i), [])
            if why not in reasons[int(i)]:
                reasons[int(i)].append(why)

    n = len(df)
    status = df.fbs0_power_status.values
    active = np.flatnonzero(status != 0.0)
    inactive = np.flatnonzero(status == 0.0)

    # 1. head and tail of the run
    add(np.arange(min(5, n)), "first-steps")
    add(np.arange(max(0, n - 5), n), "last-steps")

    # 2. per-gene extremes (first occurrence on ties, like MATLAB's min/max).
    #    Also taken restricted to active states: an inactive FBS never reaches
    #    _fbs_column, so a global extreme that lands on an inactive row would
    #    exercise nothing.
    for gene in GENES:
        v = df[gene].values
        add(int(np.argmin(v)), f"min:{gene}")
        add(int(np.argmax(v)), f"max:{gene}")
        if active.size:
            add(int(active[np.argmin(v[active])]), f"min:{gene}|active")
            add(int(active[np.argmax(v[active])]), f"max:{gene}|active")

    # 3. fbs_connected extremes, overall and restricted to active FBS states
    fc = df.fbs_connected.values
    add(int(np.argmax(fc)), "max:fbs_connected")
    add(int(np.argmin(fc)), "min:fbs_connected")
    if active.size:
        add(int(active[np.argmin(fc[active])]), "min:fbs_connected|active")
        add(int(active[np.argmax(fc[active])]), "max:fbs_connected|active")

    # 4. both power_status values explicitly
    if inactive.size:
        add(rng.choice(inactive, size=min(3, inactive.size), replace=False), "status=0")
    if active.size:
        add(rng.choice(active, size=min(3, active.size), replace=False), "status=1")

    # 5. boundary states -- clamping is the most plausible divergence point.
    #    Corners first (two edges at once), then edge-exact rows, then any row
    #    within BOUNDARY_M of an edge.
    bd = boundary_distance(df, width, height)
    on_x = (df.fbs0_x.values <= BOUNDARY_M) | (df.fbs0_x.values >= width - BOUNDARY_M)
    on_y = (df.fbs0_y.values <= BOUNDARY_M) | (df.fbs0_y.values >= height - BOUNDARY_M)
    is_on = status != 0.0
    corners = np.flatnonzero(on_x & on_y)
    if corners.size:
        add(rng.choice(corners, size=min(4, corners.size), replace=False), "corner<=2m")
    corners_on = np.flatnonzero(on_x & on_y & is_on)
    if corners_on.size:
        add(rng.choice(corners_on, size=min(4, corners_on.size), replace=False),
            "corner<=2m|active")
    for name, mask in (
        ("x=0", df.fbs0_x.values == 0.0),
        ("x=W", df.fbs0_x.values == width),
        ("y=0", df.fbs0_y.values == 0.0),
        ("y=H", df.fbs0_y.values == height),
    ):
        idx = np.flatnonzero(mask)
        if idx.size:
            add(rng.choice(idx, size=min(2, idx.size), replace=False), f"edge:{name}")
        idx_on = np.flatnonzero(mask & is_on)
        if idx_on.size:
            add(rng.choice(idx_on, size=min(3, idx_on.size), replace=False),
                f"edge:{name}|active")
    near = np.flatnonzero(bd <= BOUNDARY_M)
    if near.size:
        add(rng.choice(near, size=min(6, near.size), replace=False), "boundary<=2m")
    else:  # no row within 2 m: take the closest ones anyway and say so
        add(np.argsort(bd)[:6], "closest-to-boundary")
    near_on = np.flatnonzero((bd <= BOUNDARY_M) & is_on)
    if near_on.size:
        add(rng.choice(near_on, size=min(6, near_on.size), replace=False),
            "boundary<=2m|active")

    # 6. uniform random sample across the whole run
    k = min(20, n)
    add(rng.choice(n, size=k, replace=False), "uniform-random")

    # 7. top up to target with active-only uniform random draws (an inactive
    #    FBS makes both paths skip _fbs_column entirely, so those states carry
    #    little information about the branch under test)
    pool = np.setdiff1d(active, np.fromiter(reasons, dtype=int))
    if pool.size and len(reasons) < target:
        add(rng.choice(pool, size=min(target - len(reasons), pool.size), replace=False),
            "uniform-random|active")

    return sorted(reasons.items())


def bits(x: float) -> str:
    return np.float64(x).tobytes().hex()


def compare(a, b) -> list[dict]:
    """Every field of two SinrResults, exactly. Returns the mismatches."""
    out = []
    for f in INT_FIELDS:
        va, vb = getattr(a, f), getattr(b, f)
        if not (int(va) == int(vb)):
            out.append({"field": f, "fast": repr(vb), "faithful": repr(va)})
    for f in FLOAT_FIELDS:
        va, vb = float(getattr(a, f)), float(getattr(b, f))
        if bits(va) != bits(vb):  # bitwise: stricter than atol=0/rtol=0
            out.append({
                "field": f,
                "fast": repr(vb),
                "faithful": repr(va),
                "fast_bits": bits(vb),
                "faithful_bits": bits(va),
                "abs_diff": repr(abs(vb - va)),
            })
    return out


def column_of(backend, row) -> np.ndarray | None:
    """The FBS power column the backend just used, straight from its cache."""
    if row.fbs0_power_status == 0.0:
        return None
    spec = backend._fbs_spec(
        row.fbs0_x, row.fbs0_y, row.fbs0_height, row.fbs0_power, backend.band_freqs[0]
    )
    return backend._column_cache.get(spec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[4, 5, 6])
    ap.add_argument("--per-seed", type=int, default=72)
    ap.add_argument("--rng-seed", type=int, default=20260903)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    sink = open(args.out, "w") if args.out else None

    def say(*parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        if sink:
            sink.write(line + "\n")
            sink.flush()

    dirs = {s: run_dir(s) for s in args.seeds}
    say("=" * 78)
    say("cross-seed fast-vs-faithful replay")
    say("=" * 78)
    for s, d in dirs.items():
        say(f"  seed {s}: {d.name}")
    say(f"  numpy rng seed (sampling): {args.rng_seed}   target states/seed: {args.per_seed}")

    # All seeds must share the same world+band, or the "build the backends once"
    # shortcut would silently compare against the wrong physics.
    cfgs = {s: load_experiment_config(str(d / "experiment_config.json")) for s, d in dirs.items()}
    sigs = {s: json.dumps([cfgs[s].env.world.__dict__, cfgs[s].env.band.__dict__],
                          sort_keys=True, default=str) for s in args.seeds}
    shared = len(set(sigs.values())) == 1
    say(f"  world+band identical across seeds {args.seeds}: {shared}")
    if not shared:
        raise SystemExit("seeds do not share a world; per-seed backends would be required")
    exp = cfgs[args.seeds[0]]
    world = exp.env.world
    say(f"  world: {world.width}x{world.height} m, num_users={world.num_users}, "
        f"num_mbs={world.num_mbs}, sinr_threshold={world.sinr_threshold}, "
        f"band={exp.env.band.mode}")

    t0 = time.perf_counter()
    slow = make_backend(world, exp.env.band, "pyqd")
    fast = make_backend(world, exp.env.band, "pyqd-fast")
    say(f"  backends built once for all seeds in {time.perf_counter() - t0:.1f} s "
        f"(fast_sampling: slow={slow.fast_sampling}, fast={fast.fast_sampling})")

    grand = {"n": 0, "match": 0, "mismatch": 0, "active": 0}
    timing = {"slow_active": [], "fast_active": [], "slow_idle": [], "fast_idle": []}
    all_discrepancies: list[dict] = []
    col_max_abs_rel = 0.0
    col_checked = 0
    col_bit_equal = 0

    for seed in args.seeds:
        df = pd.read_csv(dirs[seed] / "steps.csv")
        picks = select_states(df, args.per_seed, args.rng_seed, world.width, world.height)
        say("")
        say("-" * 78)
        say(f"SEED {seed}: {len(df)} logged steps, {len(picks)} states selected")
        bd = boundary_distance(df, world.width, world.height)
        say(f"  gene ranges  x[{df.fbs0_x.min():.3f},{df.fbs0_x.max():.3f}] "
            f"y[{df.fbs0_y.min():.3f},{df.fbs0_y.max():.3f}] "
            f"h[{df.fbs0_height.min():.3f},{df.fbs0_height.max():.3f}] "
            f"p[{df.fbs0_power.min():.3f},{df.fbs0_power.max():.3f}]")
        say(f"  selected: active={sum(1 for i, _ in picks if df.fbs0_power_status.values[i] != 0)}"
            f"  inactive={sum(1 for i, _ in picks if df.fbs0_power_status.values[i] == 0)}"
            f"  min boundary dist in selection={min(bd[i] for i, _ in picks):.3f} m")
        say("-" * 78)

        seed_mismatch_fields = 0
        seed_mismatch_states = 0
        for idx, why in picks:
            row = df.iloc[idx]
            cont = np.array([[row.fbs0_x, row.fbs0_y, row.fbs0_height, row.fbs0_power]])
            status = np.array([row.fbs0_power_status])
            is_active = row.fbs0_power_status != 0.0

            t = time.perf_counter()
            r_slow = slow.evaluate(cont, status)
            dt_slow = time.perf_counter() - t
            t = time.perf_counter()
            r_fast = fast.evaluate(cont, status)
            dt_fast = time.perf_counter() - t

            timing["slow_active" if is_active else "slow_idle"].append(dt_slow)
            timing["fast_active" if is_active else "fast_idle"].append(dt_fast)

            bad = compare(r_slow, r_fast)
            grand["n"] += 1
            grand["active"] += int(is_active)
            grand["match"] += int(not bad)
            grand["mismatch"] += int(bool(bad))

            # Stronger, free check: the 1000-user FBS power column itself.
            col_note = ""
            cs, cf = column_of(slow, row), column_of(fast, row)
            if cs is not None and cf is not None:
                col_checked += 1
                same = cs.tobytes() == cf.tobytes()
                col_bit_equal += int(same)
                if not same:
                    d = np.abs(cf - cs)
                    rel = float(np.max(d / np.maximum(np.abs(cs), 1e-300)))
                    col_max_abs_rel = max(col_max_abs_rel, rel)
                    col_note = f"  COLUMN DIFFERS max|rel|={rel:.3e}"
                else:
                    col_note = "  col=bit-equal"

            tag = "OK  " if not bad else "FAIL"
            say(f"  [{tag}] seed{seed} step {int(row.step):5d} ts {int(row.timesteps):6d} "
                f"x={row.fbs0_x:9.4f} y={row.fbs0_y:9.4f} h={row.fbs0_height:8.4f} "
                f"p={row.fbs0_power:7.4f} st={row.fbs0_power_status:.0f} "
                f"| conn={r_slow.total_connected} fbs={r_slow.fbs_connected} "
                f"sum_rate={r_slow.sum_rate!r} | slow {dt_slow*1000:7.1f} ms "
                f"fast {dt_fast*1000:6.2f} ms | {','.join(why)}{col_note}")
            seed_mismatch_states += int(bool(bad))
            for m in bad:
                seed_mismatch_fields += 1
                rec = {"seed": seed, "step": int(row.step), "timesteps": int(row.timesteps),
                       "state": (float(row.fbs0_x), float(row.fbs0_y), float(row.fbs0_height),
                                 float(row.fbs0_power), float(row.fbs0_power_status)),
                       "reasons": why, **m}
                all_discrepancies.append(rec)
                say(f"        !! {m['field']}: fast={m['fast']} faithful={m['faithful']}")
        say(f"  seed {seed}: {len(picks)} states evaluated, "
            f"{len(picks) - seed_mismatch_states} exact on all 8 fields, "
            f"{seed_mismatch_states} state(s) with {seed_mismatch_fields} mismatching field(s)")

    def stat(key):
        v = np.array(timing[key])
        if v.size == 0:
            return "n/a"
        return (f"n={v.size} mean={v.mean()*1000:.2f} ms median={np.median(v)*1000:.2f} ms "
                f"min={v.min()*1000:.2f} max={v.max()*1000:.2f}")

    say("")
    say("=" * 78)
    say("SUMMARY")
    say("=" * 78)
    say(f"  states tested        : {grand['n']}  (active FBS: {grand['active']}, "
        f"inactive: {grand['n'] - grand['active']})")
    say(f"  all-8-fields exact   : {grand['match']}")
    say(f"  states with any diff : {grand['mismatch']}")
    say(f"  FBS power columns compared bitwise: {col_checked}, bit-equal: {col_bit_equal}, "
        f"max relative deviation: {col_max_abs_rel:.3e}")
    say("  per-evaluation wall clock")
    say(f"    faithful, FBS active   : {stat('slow_active')}")
    say(f"    fast,     FBS active   : {stat('fast_active')}")
    say(f"    faithful, FBS inactive : {stat('slow_idle')}")
    say(f"    fast,     FBS inactive : {stat('fast_idle')}")
    if timing["slow_active"] and timing["fast_active"]:
        ratio = np.mean(timing["slow_active"]) / np.mean(timing["fast_active"])
        say(f"    speed ratio (active, mean): {ratio:.1f}x")
    if all_discrepancies:
        say("  DISCREPANCIES:")
        for d in all_discrepancies:
            say("    " + json.dumps(d))
    else:
        say("  DISCREPANCIES: none -- every field of every state matched exactly.")
    if sink:
        sink.close()
    return 1 if all_discrepancies else 0


if __name__ == "__main__":
    raise SystemExit(main())
