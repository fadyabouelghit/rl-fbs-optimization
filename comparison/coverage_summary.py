"""How much of the visited state space did the equivalence probes actually cover?

Reads the (seed, timestep) pairs out of the probe result logs, joins them back
onto the ten steps.csv files, and reports coverage of the active-FBS state
space at a coarse binning. No backend, no physics -- pure bookkeeping, so it is
cheap to re-run.

The number that matters is the *cell* coverage, not the state count: 891 of
256,000 logged steps sounds thin, but the equivalence branch is a property of
the (x, y, z, power) cell rather than of the individual step.

Usage:
    venv/bin/python comparison/coverage_summary.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

LOGS = [
    REPO / "comparison/RESULTS_cross_seed_fast_vs_faithful_0123.txt",
    REPO / "comparison/RESULTS_cross_seed_fast_vs_faithful_456.txt",
    REPO / "comparison/RESULTS_verify_seeds789.txt",
]
BIN = (100.0, 100.0, 10.0, 0.5)  # x [m], y [m], height [m], power [dB]


def run_dir(seed: int) -> Path:
    hits = sorted((REPO / "ppo_runs").glob(f"*potrec10_seed{seed}"))
    if len(hits) != 1:
        raise SystemExit(f"expected one run dir for seed {seed}, got {hits}")
    return hits[0]


def tested_pairs() -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for path in LOGS:
        if not path.exists():
            print(f"  WARNING: missing {path.name}; its states are not counted")
            continue
        for line in path.read_text().splitlines():
            s = line.strip()
            if not (s.startswith("[OK  ]") or s.startswith("[FAIL]")):
                continue
            try:
                out.add((int(s.split("seed", 1)[1].split()[0]),
                         int(s.split(" ts ", 1)[1].split()[0])))
            except (IndexError, ValueError):
                continue
    return out


def cells(df: pd.DataFrame) -> np.ndarray:
    return np.stack([
        np.floor(df.fbs0_x.values / BIN[0]),
        np.floor(df.fbs0_y.values / BIN[1]),
        np.floor(df.fbs0_height.values / BIN[2]),
        np.floor(df.fbs0_power.values / BIN[3]),
    ], axis=1)


def main() -> int:
    pairs = tested_pairs()
    print(f"tested (seed, timestep) pairs parsed from the probe logs: {len(pairs)}")

    visited, covered, rows_total, rows_active = set(), set(), 0, 0
    per_seed = {}
    for seed in range(10):
        df = pd.read_csv(run_dir(seed) / "steps.csv", float_precision="round_trip")
        rows_total += len(df)
        act = df[df.fbs0_power_status != 0.0]
        rows_active += len(act)
        c = cells(act)
        vis = {tuple(r) for r in c}
        visited |= vis
        ts_here = {t for (s, t) in pairs if s == seed}
        sel = act[act.timesteps.isin(ts_here)]
        cov = {tuple(r) for r in cells(sel)}
        covered |= cov
        per_seed[seed] = (len(act), len(vis), len(sel), len(cov))
        print(f"  seed {seed}: {len(act):6d} active rows in {len(vis):4d} cells | "
              f"{len(sel):3d} tested active states in {len(cov):3d} cells")

    print()
    print(f"binning: x/{BIN[0]:.0f} m  y/{BIN[1]:.0f} m  height/{BIN[2]:.0f} m  "
          f"power/{BIN[3]:.1f} dB")
    print(f"logged steps across the 10 seeds : {rows_total}")
    print(f"  of which the FBS is active     : {rows_active}")
    print(f"distinct active cells VISITED    : {len(visited)}")
    print(f"distinct active cells TESTED     : {len(covered)} "
          f"({100.0 * len(covered) / max(len(visited), 1):.1f}% of visited cells)")
    print(f"untested visited cells           : {len(visited - covered)}")
    print()
    print("Read this as a bound, not a reassurance: cell coverage is 'did any state")
    print("in this region get checked', and the honest claim is a stratified sample,")
    print("not an exhaustive proof. What makes the sample carry further than its size")
    print("suggests is structural: _grid_nodes() takes no arguments -- the receiver")
    print("positions the two paths disagree about are a constant of the world, not a")
    print("function of the FBS state -- so the FBS state can only enter through the")
    print("elementwise LOS arithmetic and the tx_power scale factor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
