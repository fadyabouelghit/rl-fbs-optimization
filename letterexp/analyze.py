"""Stage 2-5 result files and figures.

    python -m letterexp.analyze prereg e2    # BEFORE launching E2: freezes P1/P2 into the results file
    python -m letterexp.analyze e2           # CHECKPOINT 2a/2b tables, F2
    python -m letterexp.analyze e3           # Stage 3 (reads E2 logs), F3
    python -m letterexp.analyze e4           # Stage 4 (hybrid campaign vs fair1), F4
    python -m letterexp.analyze e5           # Stage 5, T1

Every number here is computed from result JSONs and run logs on disk; nothing
is typed in. Verdict lines state the rule they apply, and the rule was written
before the data (prereg) for E2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats as sps

from .paths import COMPARISON_DIR, FIG_DIR, LETTER_RUNS_DIR
from .pins import EARLY, load_compact, summarize_run
from .runner import FAIL_TOL, base_params
from .stats import chi2_counts, fisher, fmt_rate, welch, wilson

LABEL = "fbs0_power_status"
RESULTS_MARK = "=== RESULTS"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_results(exp: str) -> pd.DataFrame:
    rows = []
    for f in sorted((LETTER_RUNS_DIR / exp / "results").glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("status") != "completed":
            rows.append({"run_id": d.get("run_id"), "status": d.get("status"),
                         "error": d.get("error")})
            continue
        spec_f = LETTER_RUNS_DIR / exp / "specs" / f"{d['run_id']}.json"
        meta = json.loads(spec_f.read_text()).get("arm_meta", {}) if spec_f.is_file() else {}
        rows.append({**{k: v for k, v in d.items() if not isinstance(v, (dict, list))},
                     "arm_meta": meta, "status": "completed"})
    return pd.DataFrame(rows)


PIN_KEYS = ("probe", "n_pin_events", "n_off_events", "early_pin", "early_pin_off",
            "first_onset", "first_direction", "first_off_onset", "pinned_off_at_end",
            "n_escapes", "n_trap_events", "early_trap", "first_trap_onset",
            "trapped_at_end", "n_trap_escapes", "n_mixed_events")

MIN_RUN_PRIMARY = 5
MIN_RUN_SECONDARY = 3


def with_pins(df: pd.DataFrame, min_run: int = MIN_RUN_PRIMARY) -> pd.DataFrame:
    pins = [summarize_run(Path(r), min_run=min_run) for r in df["run_dir"]]
    out = df.copy()
    for key in PIN_KEYS:
        out[key] = [p.get(key) for p in pins]
    out["pin_detail"] = pins
    return out


def ref_points() -> tuple[float, float]:
    """(do-nothing baseline J, grid optimum J) under the fair1 reward weights."""
    import sys

    from ppo.paths import REPO_ROOT
    from tuning.protocol import DEFAULT
    from tuning.spaces import build_experiment
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import assess_models as am
    exp = build_experiment("ddpg", base_params("ddpg_t27")["params"], DEFAULT, 0, "ref")
    w, mu = exp.env.reward.weights, exp.env.world.effective_max_users
    return am.objective(w, 694, 0, mu, 0.0), am.objective(w, 722, 107, mu, 1.0)


def _med_iqr(x) -> str:
    x = [v for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not x:
        return "-"
    q = np.percentile(x, [25, 50, 75])
    return f"{q[1]:.0f} [{q[0]:.0f}-{q[2]:.0f}]"


def cochran_armitage(ks, ns, xs) -> tuple[float, float]:
    ks, ns, xs = (np.asarray(v, float) for v in (ks, ns, xs))
    N = ns.sum()
    if N == 0 or ks.sum() in (0, N):
        return float("nan"), float("nan")
    p = ks.sum() / N
    T = np.sum(xs * (ks - ns * p))
    var = p * (1 - p) * (np.sum(ns * xs ** 2) - np.sum(ns * xs) ** 2 / N)
    z = T / np.sqrt(var)
    return float(z), float(2 * sps.norm.sf(abs(z)))


def _write(name: str, text: str, keep_prereg: bool = False) -> Path:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    out = COMPARISON_DIR / name
    head = ""
    if keep_prereg:
        if not out.is_file():
            raise SystemExit(f"{out} has no pre-registration block; run `analyze prereg` first")
        cur = out.read_text()
        head = cur.split(RESULTS_MARK)[0]
        _verify_prereg(head)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    out.write_text(head + (f"{RESULTS_MARK} (generated {stamp}) ===\n" if keep_prereg else "") + text)
    print(text)
    print(f"\n-> {out}")
    return out


# --------------------------------------------------------------------------- #
# pre-registration
# --------------------------------------------------------------------------- #
PREREG_E2 = """\
PRE-REGISTERED PREDICTIONS (written before any E2 run was launched)
-------------------------------------------------------------------
Design: {DDPG fair1 t27, TD3 fair1 t28} x binary noise std s_b, everything else at the
tuner-passed config. This is a LABELED ONE-FACTOR ABLATION: only s_b moves off the tuned
point. Grid: s_b in {0.1, 0.3, 0.5, 0.8, 1.2}; the tuned value is an extra anchor arm
(DDPG 0.369 as a 6th arm; TD3's 0.807 stands in for 0.8). Seeds 0-19 per arm.
Note: s_b = 0.1 is NOT the shared-noise case for these configs (continuous noise is
0.247 for t27 and 0.30 for t28); it is the "almost no binary noise" end of the grid.

Operational definitions (letterexp/pins.py): pinned at a checkpoint = >= 95% of the 256
probe states have |mu_b| > 0.96; direction = off / on if >= 95% of the saturated states are
at -1 / +1, else mixed; pin event = >= MIN_RUN consecutive pinned checkpoints with the same
direction (checkpoints every 1000 steps); early = onset < 10,000 steps. Trapped at a
checkpoint = >= 95% of the probe states whose gene is currently OFF have mu_b < -0.96; trap
event = >= MIN_RUN consecutive trapped checkpoints (an all-states pin at -1 is a trap; so is
a saturated "keep the current value" actor). Escape = a trap event that ends before the last
checkpoint. Failure = J(best_model) within 0.0005 of the do-nothing baseline, or below it.

MIN_RUN is fixed HERE, before any E2 run exists, at TWO declared values: 5 checkpoints
(>= 5000 steps) for the PRIMARY analysis and 3 (>= 3000 steps) for a SECONDARY one. Every
test below is reported at both. Reason, recorded in advance: in Stage 0 the tuned TD3 anchor
(t28, seed 0, 100k) was trapped for exactly 4 consecutive checkpoints -- one short of 5 --
while its training eval curve sat at the do-nothing baseline with zero FBS-connected users.
A sole threshold of 5 would therefore risk (i) confirming P1 on all-zero cells, i.e. a null
prediction "supported" by having measured nothing, and (ii) leaving P2(a)'s denominator
(runs with a trap event) empty, so the trend test could not be computed. Neither threshold
is privileged by the data: both are set now, and both are reported whatever they show.

P1  The share of runs with an early pin event (any direction) does not depend on s_b.
    Test: chi-square of (runs with early pin) across the 5 grid levels, per algorithm
    (Monte-Carlo p if an expected cell < 5). Supported if p >= 0.05. Secondary: the same
    test including the anchor arm, and on early trap events.
    Caveat stated in advance: p >= 0.05 is absence of evidence at n = 20 per level, so the
    result is reported with the per-level Wilson intervals, not as proof of no effect.
P2  Escape probability from a trap rises with s_b, but the failure rate does not reach 0.
    Test (a): Cochran-Armitage trend of (runs that escaped a trap at least once / runs with
    a trap event) on s_b, one-sided in the predicted direction: supported if z > 0 and
    p < 0.05. Secondary: the same on 'off'-direction pin events.
    Test (b): failures at s_b = 1.2 > 0 (reported with the Wilson 95% interval).
    P2 is supported only if (a) and (b) both hold.
"""


def _prereg_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _verify_prereg(head: str) -> None:
    lines = head.rstrip("\n").split("\n")
    sha_line = [ln for ln in lines if ln.startswith("prereg-sha256: ")]
    if not sha_line:
        raise SystemExit("pre-registration block has no hash line")
    body = head.split("prereg-sha256: ")[0]
    if _prereg_hash(body) != sha_line[0].split(": ")[1].strip():
        raise SystemExit("pre-registration block was edited after it was written")


def prereg(exp: str) -> None:
    if exp != "e2":
        raise SystemExit("only E2 has pre-registered predictions")
    out = COMPARISON_DIR / "RESULTS_letter_E2.txt"
    if out.is_file():
        raise SystemExit(f"{out} exists; the pre-registration is written once")
    existing = list((LETTER_RUNS_DIR / "e2" / "results").glob("*.json"))
    if existing:
        raise SystemExit("E2 results already exist; a pre-registration must precede the data")
    body = (f"RESULTS_letter_E2  --  does noise cause pinning, or only set the escape rate? (C1, C4)\n"
            f"{'=' * 78}\nwritten {time.strftime('%Y-%m-%d %H:%M %Z')}\n\n{PREREG_E2}\n")
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(body + f"prereg-sha256: {_prereg_hash(body)}\n\n")
    print(out.read_text())


# --------------------------------------------------------------------------- #
# E2
# --------------------------------------------------------------------------- #
def _s_b(row) -> float:
    return float(row["params"]["binary_noise_std"]) if isinstance(row.get("params"), dict) else np.nan


def _e2_rows(d: pd.DataFrame) -> list[dict]:
    """Per-arm counts the P1/P2 tests need (no formatting)."""
    rows = []
    for s, g in sorted(d.groupby("s_b"), key=lambda x: x[0]):
        trap_runs = g[g["n_trap_events"] > 0]
        off_runs = g[g["n_off_events"] > 0]
        rows.append({"s_b": s, "anchor": bool(g["anchor"].any()), "n": len(g),
                     "early": int(g["early_pin"].sum()),
                     "early_off": int(g["early_trap"].sum()),
                     "off_runs": len(trap_runs),
                     "esc_runs": int((trap_runs["n_trap_escapes"] > 0).sum()),
                     "pin_off_runs": len(off_runs),
                     "esc_off": int((off_runs["n_escapes"] > 0).sum()),
                     "fails": int(g["fail_best"].sum()),
                     "J": g["J_best"].to_numpy(), "stuck": int(g["trapped_at_end"].sum())})
    return rows


def _e2_tests(rows: list[dict], algo: str, min_run: int, tier: str) -> list[str]:
    """P1 and P2 at one event-length threshold. Both thresholds are pre-registered."""
    L = [f"\n  == {tier}: pin/trap event = >= {min_run} consecutive checkpoints "
         f"(>= {min_run * 1000} steps) =="]
    grid = [r for r in rows if not (r["anchor"] and algo == "ddpg")]
    c2, dof, p, mc = chi2_counts([r["early"] for r in grid], [r["n"] for r in grid])
    c2a, dofa, pa, mca = chi2_counts([r["early"] for r in rows], [r["n"] for r in rows])
    c2o, dofo, po, mco = chi2_counts([r["early_off"] for r in grid], [r["n"] for r in grid])
    n_ev = sum(r["early"] for r in grid)
    L.append(f"  P1 early pin vs s_b ({len(grid)} grid levels): chi2={c2:.2f} dof={dof} p={p:.3g}"
             f"{' (Monte-Carlo)' if mc else ''} -> {'SUPPORTED' if p >= 0.05 else 'NOT SUPPORTED'}"
             f" (rule: p >= 0.05)")
    if n_ev == 0:
        L.append("     !! VACUOUS: zero early pin events at any level, so P1 is 'supported' by "
                 "having measured nothing. Do not read this as evidence of no effect.")
    L.append(f"     all {len(rows)} arms incl. anchor: chi2={c2a:.2f} p={pa:.3g}; "
             f"early TRAP events: chi2={c2o:.2f} p={po:.3g}")
    ks = [r["esc_runs"] for r in grid if r["off_runs"]]
    ns = [r["off_runs"] for r in grid if r["off_runs"]]
    xs = [r["s_b"] for r in grid if r["off_runs"]]
    z, pz = cochran_armitage(ks, ns, xs)
    p1s = pz / 2 if (not np.isnan(z) and z > 0) else (1 - pz / 2 if not np.isnan(z) else float("nan"))
    top = max(grid, key=lambda r: r["s_b"])
    a_ok = (not np.isnan(z)) and z > 0 and p1s < 0.05
    b_ok = top["fails"] > 0
    L.append(f"  P2(a) escape trend on s_b (Cochran-Armitage, runs with a trap event): z={z:.2f} "
             f"one-sided p={p1s:.3g} -> {'holds' if a_ok else 'does not hold'}"
             f"{'  [no trap events: not computable]' if sum(ns) == 0 else ''}")
    z2, p2 = cochran_armitage([r["esc_off"] for r in grid if r["pin_off_runs"]],
                              [r["pin_off_runs"] for r in grid if r["pin_off_runs"]],
                              [r["s_b"] for r in grid if r["pin_off_runs"]])
    L.append(f"     secondary, 'off'-direction pin events: z={z2:.2f} two-sided p={p2:.3g}")
    L.append(f"  P2(b) failures at s_b={top['s_b']:g}: {fmt_rate(top['fails'], top['n'])} -> "
             f"{'holds (not zero)' if b_ok else 'does not hold (zero failures)'}")
    L.append(f"  P2 -> {'SUPPORTED' if a_ok and b_ok else 'NOT SUPPORTED' if not (a_ok or b_ok) else 'PARTLY (see a/b)'}")
    return L


def e2(fig: bool = True) -> None:
    df = load_results("e2")
    if df.empty:
        raise SystemExit("no E2 results")
    bad = df[df["status"] != "completed"]
    ok = df[df["status"] == "completed"].reset_index(drop=True)
    df = with_pins(ok, min_run=MIN_RUN_PRIMARY)
    df3 = with_pins(ok, min_run=MIN_RUN_SECONDARY)
    for t in (df, df3):
        t["s_b"] = [m.get("s_b") for m in t["arm_meta"]]
        t["anchor"] = [bool(m.get("anchor")) for m in t["arm_meta"]]
    L = []
    if len(bad):
        L.append(f"!! {len(bad)} failed/crashed runs: {list(bad['run_id'])}\n")
    L.append(f"runs: {len(df)} completed; seeds {sorted(df['seed'].unique().tolist())}\n")
    summary = {}
    for algo in ("ddpg", "td3"):
        d = df[df["algo"] == algo]
        if d.empty:
            continue
        L.append(f"--- {algo.upper()} ({d['base'].iloc[0]}) " + "-" * 50)
        hdr = (f"{'s_b':>6} {'n':>3} | {'early pin':>22} {'dir off/on/mix':>14} {'onset med[IQR]':>18} | "
               f"{'runs w/ trap':>12} {'escaped trap':>22} {'t_esc med':>10} {'flips<esc':>9} {'trap@end':>8} | "
               f"{'failures (best)':>24} {'J_best mean+-sd':>17} {'gap':>6} | {'J_final':>15} {'fail(final)':>11}")
        L.append(hdr)
        rows = []
        for s, g in sorted(d.groupby("s_b"), key=lambda x: x[0]):
            n = len(g)
            ep = int(g["early_pin"].sum())
            first_dirs = [p["events"][0]["direction"] for p in g["pin_detail"] if p.get("events")]
            off_runs = g[g["n_off_events"] > 0]
            esc_off = int((off_runs["n_escapes"] > 0).sum())
            trap_runs = g[g["n_trap_events"] > 0]
            esc_runs = int((trap_runs["n_trap_escapes"] > 0).sum())
            t_esc = sum((p.get("trap_escape_times", []) for p in g["pin_detail"]), [])
            f_esc = sum((p.get("trap_flips_before_escape", []) for p in g["pin_detail"]), [])
            stuck = int(g["trapped_at_end"].sum())
            fails = int(g["fail_best"].sum())
            ffin = int(g["fail_final"].sum())
            J = g["J_best"].to_numpy()
            onsets = [o for o in g["first_onset"] if o is not None and not pd.isna(o)]
            tag = " *" if g["anchor"].any() else ""
            L.append(f"{s:>6.3g}{tag:2s}{n:>2} | {fmt_rate(ep, n):>22} "
                     f"{first_dirs.count('off'):>3}/{first_dirs.count('on')}/{first_dirs.count('mixed'):<3} {_med_iqr(onsets):>18} | "
                     f"{len(trap_runs):>12} {fmt_rate(esc_runs, len(trap_runs)) if len(trap_runs) else '-':>22} "
                     f"{_med_iqr(t_esc):>10} {_med_iqr(f_esc):>9} {stuck:>8} | "
                     f"{fmt_rate(fails, n):>24} {J.mean():>8.4f}+-{J.std(ddof=1) if n > 1 else 0:<7.4f} "
                     f"{np.mean(g['gap_closed_best']):>6.3f} | {g['J_final'].mean():>7.4f}+-{g['J_final'].std(ddof=1) if n > 1 else 0:.4f} "
                     f"{ffin:>5}/{n}")
            rows.append({"s_b": s, "anchor": bool(g["anchor"].any()), "n": n, "early": ep,
                         "early_off": int(g["early_trap"].sum()), "off_runs": len(trap_runs),
                         "esc_runs": esc_runs, "pin_off_runs": len(off_runs), "esc_off": esc_off,
                         "fails": fails, "J": J, "stuck": stuck})
        L.append("  (* = tuned anchor arm; gap = (J - baseline) / (optimum - baseline), best_model)")
        L.extend(_e2_tests(rows, algo, MIN_RUN_PRIMARY, "PRIMARY"))
        # Pre-registered secondary threshold: the same tests on >= 3 consecutive
        # checkpoints. Both MIN_RUN values were fixed before any E2 run existed.
        d3 = df3[df3["algo"] == algo]
        if not d3.empty:
            L.extend(_e2_tests(_e2_rows(d3), algo, MIN_RUN_SECONDARY, "SECONDARY"))
        # link: pinned-off at end vs failure
        ct = pd.crosstab(d["trapped_at_end"].astype(bool), d["fail_best"].astype(bool))
        L.append(f"  trapped-at-end x failure (all arms):\n" +
                 "\n".join("    " + ln for ln in ct.to_string().split("\n")))
        L.append("")
        summary[algo] = rows
    _write("RESULTS_letter_E2.txt", "\n".join(L) + "\n", keep_prereg=True)
    if fig:
        _fig_e2(summary)


def _fig_e2(summary) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    col = {"ddpg": "#264653", "td3": "#e76f51"}
    for algo, rows in summary.items():
        xs = np.array([r["s_b"] for r in rows])
        for ax, key, den in ((axes[0], "early", "n"), (axes[1], "esc_runs", "off_runs"),
                             (axes[2], "fails", "n")):
            ok = [r for r in rows if r[den] > 0]
            x = np.array([r["s_b"] for r in ok])
            w = np.array([wilson(r[key], r[den]) for r in ok])
            if len(w):
                ax.errorbar(x, w[:, 0], yerr=[w[:, 0] - w[:, 1], w[:, 2] - w[:, 0]], fmt="o-",
                            color=col[algo], capsize=3, label=algo.upper())
            for r in ok:
                if r["anchor"]:
                    ax.plot([r["s_b"]], [r[key] / r[den]], "s", mfc="none", mec=col[algo], ms=10)
    for ax, t in zip(axes, ("early pin event (< 10k steps)", "escape | trap event", "failure rate")):
        ax.set_xlabel("binary noise std $s_b$")
        ax.set_title(t)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("share of runs (Wilson 95%)")
    axes[0].legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"F2_noise_vs_pin_escape_failure.{ext}", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# E3
# --------------------------------------------------------------------------- #
E3_METRICS = [("tanhf_median", "tanh factor 1 - mu_b^2"),
              ("absdq_mu_q1_median", "|dQ1/da_b| at mu(s)"),
              ("absdq_p05_q1_median", "|dQ1/da_b| at a_b = +0.5"),
              ("qon_minus_qoff_q1_median", "Q1(a_b=+1) - Q1(a_b=-1)"),
              ("frac_prefer_on_q1", "share of states with Q1(on) > Q1(off)")]


def e3(fig: bool = True) -> None:
    df = load_results("e2")
    df = with_pins(df[df["status"] == "completed"].reset_index(drop=True))
    L = ["RESULTS_letter_E3  --  why the actor cannot unpin (claim C3); analysis of the E2 logs",
         "=" * 78,
         "groups: STUCK = a trap event lasting to the end of training; HEALTHY = no trap event at",
         "all (trap: >= 95% of gene-off probe states have mu_b < -0.96, see pins.py). Per run: median of the per-checkpoint value (itself a median over the 256",
         "probe states) BEFORE pin onset and AFTER it (stuck), or over all checkpoints (healthy);",
         "then median [IQR] across runs.", ""]
    curves = {}
    for algo in ("ddpg", "td3"):
        d = df[df["algo"] == algo]
        if d.empty:
            continue
        metrics = list(E3_METRICS) + ([(m.replace("q1", "q2"), t.replace("Q1", "Q2"))
                                       for m, t in E3_METRICS if "q1" in m] if algo == "td3" else [])
        stuck, healthy = [], []
        for _, r in d.iterrows():
            log = pd.read_csv(Path(r["run_dir"]) / "probe_log.csv")
            p = r["pin_detail"]
            if p.get("trapped_at_end"):
                onset = [e for e in p["trap_events"] if e["at_end"]][0]["onset"]
                stuck.append((log, onset, r))
            elif p.get("n_trap_events", 0) == 0:
                healthy.append((log, None, r))
        L.append(f"--- {algo.upper()}: {len(stuck)} stuck, {len(healthy)} healthy, "
                 f"{len(d) - len(stuck) - len(healthy)} other (trap that escaped) " + "-" * 10)
        L.append(f"{'metric':42s} {'STUCK before onset':>22} {'STUCK after onset':>22} {'HEALTHY':>22}")
        for m, title in metrics:
            col = f"{LABEL}_{m}"
            b = [np.median(lg[col][lg["timesteps"] < on]) for lg, on, _ in stuck if (lg["timesteps"] < on).any()]
            a = [np.median(lg[col][lg["timesteps"] >= on]) for lg, on, _ in stuck]
            h = [np.median(lg[col]) for lg, _, _ in healthy]
            fm = lambda x: (f"{np.median(x):+.3g} [{np.percentile(x, 25):+.2g},{np.percentile(x, 75):+.2g}]"
                            if len(x) else "-")
            L.append(f"{title:42s} {fm(b):>22} {fm(a):>22} {fm(h):>22}")
        if stuck:
            pref = [float(np.mean(lg[f"{LABEL}_frac_prefer_on_q1"][lg["timesteps"] >= on] > 0.5))
                    for lg, on, _ in stuck]
            L.append(f"\nKEY: share of pinned-off checkpoints where most probe states have Q1(on) > Q1(off): "
                     f"median {np.median(pref):.2f} across stuck runs "
                     f"({sum(x > 0.5 for x in pref)}/{len(pref)} runs above one half)")
            L.append("     -> the critic prefers 'on' while the actor stays 'off' in those runs"
                     if np.median(pref) > 0.5 else
                     "     -> the critic does NOT prefer 'on' while pinned (C3's mechanism would then be the"
                     " critic, not only the vanishing gradient)")
        L.append("")
        curves[algo] = (stuck, healthy, metrics)
    _write("RESULTS_letter_E3.txt", "\n".join(L) + "\n")
    if fig and curves:
        _fig_e3(curves)


def _fig_e3(curves) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    show = E3_METRICS[:4]
    fig, axes = plt.subplots(len(curves), len(show), figsize=(15, 3.4 * len(curves)), squeeze=False)
    for i, (algo, (stuck, healthy, _)) in enumerate(curves.items()):
        for j, (m, title) in enumerate(show):
            ax = axes[i, j]
            for grp, color, name in ((stuck, "#e76f51", "stuck off"), (healthy, "#2a9d8f", "healthy")):
                if not grp:
                    continue
                t = grp[0][0]["timesteps"].to_numpy()
                Y = np.array([lg[f"{LABEL}_{m}"].to_numpy()[: len(t)] for lg, _, _ in grp
                              if len(lg) >= len(t)])
                if not len(Y):
                    continue
                q = np.percentile(Y, [25, 50, 75], axis=0)
                ax.plot(t, q[1], color=color, label=f"{name} (n={len(Y)})")
                ax.fill_between(t, q[0], q[2], color=color, alpha=0.2)
            if "absdq" in m:
                ax.set_yscale("symlog", linthresh=1e-4)
            ax.set_title(f"{algo.upper()}: {title}", fontsize=9)
            ax.set_xlabel("timesteps")
            ax.grid(alpha=0.3)
        axes[i, 0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"F3_critic_slope_tanh_factor.{ext}", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# E4
# --------------------------------------------------------------------------- #
def e4(fig: bool = True) -> None:
    from .launch import HYBRID_CAMPAIGN_DIR
    base_j, opt_j = ref_points()
    L = ["RESULTS_letter_E4  --  the minimal remedy (claim C4)", "=" * 78,
         f"hybrid campaign: {HYBRID_CAMPAIGN_DIR} (same Protocol as fair1: 60 trials at seed 0,",
         "top 8 confirmed at seeds 0-4, all 8 extended to seeds 0-34, ranked on mean - sd).",
         f"failure = J(best_model) <= baseline {base_j:.4f} + {FAIL_TOL}; gap = (J - base)/(opt - base), opt {opt_j:.4f}",
         ""]
    panel = {}
    # Both hybrid generations are reported when present: the straight-through
    # original and the Q-difference variant of letter Step 2. A missing
    # confirmation file is skipped with a note, so this works mid-campaign.
    for parent, hyb in (("ddpg_t27", "ddpg_hybrid_qd"), ("td3_t28", "td3_hybrid_qd"),
                        ("ddpg_t27", "ddpg_hybrid"), ("td3_t28", "td3_hybrid")):
        f = HYBRID_CAMPAIGN_DIR / f"{hyb}_confirmation.json"
        if not f.is_file():
            L.append(f"--- {hyb}: no confirmation file yet ({f})\n")
            continue
        conf = json.loads(f.read_text())
        win = conf["winner"]
        par = base_params(parent)
        seeds = sorted(set(int(s) for s in par["seeds"]) & set(int(s) for s in win["seeds"]))
        jb = np.array([par["seeds"][str(s)] for s in seeds], float)
        jh = np.array([win["seeds"][str(s)] for s in seeds], float)
        fb, fh = int(np.sum(jb <= base_j + FAIL_TOL)), int(np.sum(jh <= base_j + FAIL_TOL))
        orr, pf = fisher(fh, len(jh), fb, len(jb))
        dm, t, pw = welch(jh, jb)
        gb, gh = (jb - base_j) / (opt_j - base_j), (jh - base_j) / (opt_j - base_j)
        L.append(f"--- {hyb} winner t{win['trial']} {win['params']} vs fair1 {parent} {par['params']}")
        L.append(f"  common seeds: {len(seeds)} ({seeds[0]}-{seeds[-1]})")
        L.append(f"  failures   hybrid {fmt_rate(fh, len(jh))}   parent {fmt_rate(fb, len(jb))}   "
                 f"Fisher OR={orr:.3g} p={pf:.3g}")
        L.append(f"  mean J     hybrid {jh.mean():.4f}+-{jh.std(ddof=1):.4f}   parent {jb.mean():.4f}+-"
                 f"{jb.std(ddof=1):.4f}   diff {dm:+.4f}  Welch t={t:.2f} p={pw:.3g}")
        L.append(f"  gap closed hybrid {gh.mean():.3f}   parent {gb.mean():.3f}   worst seed hybrid "
                 f"{jh.min():.4f} parent {jb.min():.4f}")
        L.append("  all hybrid finalists (n, mean, sd, failures):")
        for r in conf["finalists"]:
            v = np.array([x for x in r["seeds"].values() if x is not None], float)
            L.append(f"    t{r['trial']:<3} n={len(v):>2} {v.mean():.4f}+-{v.std(ddof=1) if len(v) > 1 else 0:.4f} "
                     f"fail {int(np.sum(v <= base_j + FAIL_TOL))}  {r['params']}")
        L.append(f"  winner vs winner, failures: {_c4_verdict(fh, fb, pf)}")
        # campaign level: every confirmed finalist of each pipeline, pooled. Both sets passed
        # the identical tuner, so this is the like-for-like comparison with the power to see
        # a failure rate (the tuned winners themselves rarely fail).
        pool_b = _pool(json.loads(Path(par["source"]).read_text()))
        pool_h = _pool(conf)
        kb, nb_ = int(np.sum(pool_b <= base_j + FAIL_TOL)), len(pool_b)
        kh, nh = int(np.sum(pool_h <= base_j + FAIL_TOL)), len(pool_h)
        orp, pp = fisher(kh, nh, kb, nb_)
        L.append(f"  POOLED FINALISTS (all 8 x all seeds, each pipeline): hybrid {fmt_rate(kh, nh)}  "
                 f"parent {fmt_rate(kb, nb_)}  Fisher OR={orp:.3g} p={pp:.3g}")
        dmp, tp, pwp = welch(pool_h, pool_b)
        L.append(f"     mean J pooled: hybrid {pool_h.mean():.4f} parent {pool_b.mean():.4f} "
                 f"diff {dmp:+.4f} Welch p={pwp:.3g}")
        L.append(f"  C4 (pooled, primary for the failure claim) -> {_c4_verdict(kh, kb, pp)}")
        # instrumentation of hybrid runs, if present
        rd = [Path(x) for x in win["run_dirs"].values() if x]
        pins = [summarize_run(r) for r in rd if (r / "probe_log.csv").is_file()]
        if pins:
            L.append(f"  probe on {len(pins)} hybrid runs: runs with a trap event (P(on) < 2% on "
                     f">= 95% of gene-off states) {sum(p.get('n_trap_events', 0) > 0 for p in pins)}, "
                     f"trapped at end {sum(bool(p.get('trapped_at_end')) for p in pins)}")
        L.append("")
        panel[hyb] = (jb, jh, fb, fh)
    pilot = load_results("e4_pilot")
    if not pilot.empty and "J_best" in pilot:
        L.append("e4_pilot (implementation check at the parent's knobs, gumbel_tau 1.0; NOT the comparison):")
        for _, r in pilot.iterrows():
            L.append(f"  {r['run_id']:32s} J_best={r['J_best']:.4f} J_final={r['J_final']:.4f} "
                     f"fail={r['fail_best']}")
    _write("RESULTS_letter_E4.txt", "\n".join(L) + "\n")
    if fig and panel:
        _fig_e4(panel, base_j, opt_j)


def _pool(conf: dict) -> np.ndarray:
    return np.array([v for r in conf["finalists"] for v in r["seeds"].values() if v is not None], float)


def _c4_verdict(fh: int, fb: int, p: float) -> str:
    if fb == 0:
        return "INCONCLUSIVE: the parent has no failures here, so removal cannot be shown"
    if fh == 0:
        return (f"SUPPORTED (0 failures)" + (f", Fisher p={p:.3g}" if p < 0.05 else
                f" but not significant (Fisher p={p:.3g}); report as consistent, not proven"))
    if fh < fb:
        return f"PARTLY: reduced, not removed (Fisher p={p:.3g})"
    return f"NOT SUPPORTED: not reduced (Fisher p={p:.3g})"


def _fig_e4(panel, base_j, opt_j) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    labels, x = [], 0
    for hyb, (jb, jh, fb, fh) in panel.items():
        par = hyb.replace("_hybrid", "").upper()
        for name, j, f, c in ((f"{par}\ntuned", jb, fb, "#8d99ae"), (f"{par}-hybrid\ntuned", jh, fh, "#2a9d8f")):
            p, lo, hi = wilson(f, len(j))
            # Wilson contains p analytically, but at k=0 the subtraction lands a
            # few ulp negative and matplotlib rejects a negative yerr.
            lo_err, hi_err = max(0.0, p - lo), max(0.0, hi - p)
            ax[0].bar(x, p, color=c)
            ax[0].errorbar(x, p, yerr=[[lo_err], [hi_err]], color="k", capsize=3)
            g = (j - base_j) / (opt_j - base_j)
            ci = 1.96 * g.std(ddof=1) / np.sqrt(len(g))
            ax[1].bar(x, g.mean(), color=c)
            ax[1].errorbar(x, g.mean(), yerr=ci, color="k", capsize=3)
            labels.append(name)
            x += 1
        x += 0.5
    pos = [i + 0.5 * (i // 2) for i in range(len(labels))]
    for a, t in zip(ax, ("failure rate (Wilson 95%)", "gap closed (mean, 95% CI)")):
        a.set_xticks(pos)
        a.set_xticklabels(labels, fontsize=8)
        a.set_title(t)
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"F4_remedy_vs_baseline.{ext}", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# E5
# --------------------------------------------------------------------------- #
def e5(fig: bool = True) -> None:
    df = load_results("e5")
    df = with_pins(df[df["status"] == "completed"].reset_index(drop=True))
    df["pd"] = [int(m.get("policy_delay")) for m in df["arm_meta"]]
    L = ["RESULTS_letter_E5  --  TD3 policy delay, controlled (LABELED ABLATION of fair1 t28)", "=" * 78,
         "TD3 t28 with policy_delay in {1, 2, 4}, tau 0.02, everything else identical; delay 1 is t28 itself.",
         ""]
    ref = df[df["pd"] == 1]
    rows = []
    for d, g in sorted(df.groupby("pd")):
        n, f = len(g), int(g["fail_best"].sum())
        orr, pf = fisher(f, n, int(ref["fail_best"].sum()), len(ref)) if d != 1 else (np.nan, np.nan)
        dm, t, pw = welch(g["J_best"], ref["J_best"]) if d != 1 else (np.nan, np.nan, np.nan)
        rows.append({"policy_delay": d, "n": n, "failures": fmt_rate(f, n),
                     "J_best": f"{g['J_best'].mean():.4f} +- {g['J_best'].std(ddof=1):.4f}",
                     "gap_closed": f"{g['gap_closed_best'].mean():.3f}",
                     "early pin": fmt_rate(int(g["early_pin"].sum()), n),
                     "trapped at end": fmt_rate(int(g["trapped_at_end"].sum()), n),
                     "Fisher p vs pd1": "-" if d == 1 else f"{pf:.3g}",
                     "Welch dJ vs pd1": "-" if d == 1 else f"{dm:+.4f} (p={pw:.3g})"})
    t1 = pd.DataFrame(rows)
    L.append(t1.to_string(index=False))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t1.to_csv(FIG_DIR / "T1_policy_delay.csv", index=False)
    (FIG_DIR / "T1_policy_delay.md").write_text(t1.to_markdown(index=False) if hasattr(t1, "to_markdown") else t1.to_string())
    _write("RESULTS_letter_E5.txt", "\n".join(L) + "\n")
    if fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(figsize=(12, 1.2 + 0.4 * len(t1)))
        ax.axis("off")
        tb = ax.table(cellText=t1.values, colLabels=t1.columns, loc="center")
        tb.auto_set_font_size(False)
        tb.set_fontsize(8)
        tb.scale(1, 1.4)
        for ext in ("png", "pdf"):
            f.savefig(FIG_DIR / f"T1_policy_delay.{ext}", dpi=200, bbox_inches="tight")
        plt.close(f)


# --------------------------------------------------------------------------- #
# Stage 0 reproducibility
# --------------------------------------------------------------------------- #
def _on_share_tail(run_dir, label: str, last: int = 20_000) -> float:
    """Share of the final ``last`` steps with the station ON (compact log)."""
    c = load_compact(Path(run_dir))
    if c is None or not len(c):
        return float("nan")
    t_end = int(c["timesteps"].max())
    tail = c[c["timesteps"] > t_end - last]
    return float(np.mean(tail[label].to_numpy() > 0.5)) if len(tail) else float("nan")


def _during_trap(run_dir, events: list) -> dict:
    """Median critic signals over the checkpoints inside the trap events."""
    f = Path(run_dir) / "probe_log.csv"
    if not f.is_file() or not events:
        return {}
    log = pd.read_csv(f)
    t = log["timesteps"].to_numpy()
    m = np.zeros(len(t), bool)
    for e in events:
        hi = e["last_pinned"]
        m |= (t >= e["onset"]) & (t <= hi)
    if not m.any():
        return {}
    sub = log[m]
    out = {"n_ckpt": int(m.sum())}
    for col, key in ((f"{LABEL}_frac_prefer_on_q1", "prefer_on_q1"),
                     (f"{LABEL}_dq_mu_q1_median", "dq_mu_q1")):
        if col in sub:
            out[key] = float(np.median(sub[col].to_numpy()))
    return out


def e3_fail(fig: bool = False) -> None:
    """Step 1: do the failures of failure-prone tuner-passed configs coincide
    with a persistent stuck-off trap?"""
    base_j, opt_j = ref_points()
    df = load_results("e3_fail")
    bad = df[df["status"] != "completed"]
    ok = df[df["status"] == "completed"].reset_index(drop=True)
    d5 = with_pins(ok, min_run=MIN_RUN_PRIMARY)
    d3 = with_pins(ok, min_run=MIN_RUN_SECONDARY)
    L = ["RESULTS_letter_E3_fail  --  failure-prone fair1 finalists, rerun with the probe",
         "=" * 78,
         "Configs are the tuner-passed fair1 finalists, UNCHANGED (no overrides); seeds 0-34.",
         f"failure = J_best <= {base_j + FAIL_TOL:.4f};  'switched off' = J_best == the do-nothing",
         f"baseline {base_j:.10f} exactly (the station never on);  'on but worse' = below baseline.",
         f"trap/pin events at the pre-registered MIN_RUN = {MIN_RUN_PRIMARY} (primary) and "
         f"{MIN_RUN_SECONDARY} (secondary).", ""]
    if len(bad):
        L.append(f"!! {len(bad)} failed/crashed runs: {list(bad['run_id'])}\n")

    tally = {}
    for arm, g5 in d5.groupby("arm"):
        g3 = d3[d3["arm"] == arm]
        n = len(g5)
        J = g5["J_best"].to_numpy()
        fail = J <= base_j + FAIL_TOL
        off = np.isclose(J, base_j, rtol=0.0, atol=1e-12)
        on_worse = fail & ~off
        L.append(f"--- {arm}  (fair1 {g5['base'].iloc[0]}, n={n}) " + "-" * 30)
        bp = base_params(g5["base"].iloc[0])
        L.append(f"  params: {bp['params']}")
        L.append(f"  fair1 record for this config: n={bp.get('n_ok')} "
                 f"mean {bp.get('mean')} sd {bp.get('sd')}")
        L.append(f"  failures      {fmt_rate(int(fail.sum()), n)}")
        L.append(f"    switched off (J == baseline exactly) {int(off.sum())}"
                 f"   on but below baseline {int(on_worse.sum())}")
        L.append(f"  mean J_best   {J.mean():.4f} +- {J.std(ddof=1):.4f}   worst {J.min():.4f}")
        L.append(f"  trap events (MIN_RUN={MIN_RUN_PRIMARY}): "
                 f"{int((g5['n_trap_events'] > 0).sum())}/{n} runs, "
                 f"trapped at end {int(g5['trapped_at_end'].sum())}  |  "
                 f"(MIN_RUN={MIN_RUN_SECONDARY}): {int((g3['n_trap_events'] > 0).sum())}/{n} runs, "
                 f"trapped at end {int(g3['trapped_at_end'].sum())}")

        rows = []
        for (_, r5), (_, r3) in zip(g5.iterrows(), g3.iterrows()):
            j = float(r5["J_best"])
            f_ = j <= base_j + FAIL_TOL
            rows.append({
                "run": r5["run_id"].rsplit("_s", 1)[-1], "J": j,
                "class": ("switched-off" if np.isclose(j, base_j, rtol=0, atol=1e-12)
                          else "on-but-worse" if f_ else "ok"),
                "trap_end5": bool(r5["trapped_at_end"]), "trap_end3": bool(r3["trapped_at_end"]),
                "n_trap5": int(r5["n_trap_events"] or 0),
                "on_share_20k": _on_share_tail(r5["run_dir"], LABEL),
                **_during_trap(r5["run_dir"], r5["pin_detail"].get("trap_events", [])),
            })
        t = pd.DataFrame(rows)
        tally[arm] = t

        for cls in ("switched-off", "on-but-worse"):
            sub = t[t["class"] == cls]
            if not len(sub):
                L.append(f"  no {cls} runs")
                continue
            L.append(f"  {cls} runs (seed, J, trap-to-end@5/@3, ON share last 20k, "
                     f"median prefer_on_q1 / dq_mu_q1 during trap):")
            for _, r in sub.iterrows():
                L.append(f"    s{r['run']:<3} J={r['J']:.4f}  trap_end {str(r['trap_end5']):>5}/"
                         f"{str(r['trap_end3']):<5}  ON={r['on_share_20k']:.4f}  "
                         f"prefer_on={r.get('prefer_on_q1', float('nan')):.3f}  "
                         f"dq_mu={r.get('dq_mu_q1', float('nan')):+.4f}")
        okr = t[t["class"] == "ok"]
        L.append(f"  NON-FAILED runs (n={len(okr)}) for comparison: "
                 f"trap-to-end {int(okr['trap_end5'].sum())}@5 / {int(okr['trap_end3'].sum())}@3, "
                 f"ON share last 20k median {okr['on_share_20k'].median():.4f} "
                 f"[{okr['on_share_20k'].quantile(.25):.4f}-{okr['on_share_20k'].quantile(.75):.4f}]")
        if "prefer_on_q1" in okr and okr["prefer_on_q1"].notna().any():
            L.append(f"     among non-failed runs that had a trap: median prefer_on_q1 "
                     f"{okr['prefer_on_q1'].median():.3f}, dq_mu_q1 {okr['dq_mu_q1'].median():+.4f}")
        L.append("")

    # the verdict, in counts
    allt = pd.concat(tally.values()) if tally else pd.DataFrame()
    if len(allt):
        off = allt[allt["class"] == "switched-off"]
        onw = allt[allt["class"] == "on-but-worse"]
        okr = allt[allt["class"] == "ok"]
        L.append("VERDICT -- do switched-off failures coincide with a persistent trap?")
        for nm, sub in (("switched-off", off), ("on-but-worse", onw), ("not failed", okr)):
            if not len(sub):
                L.append(f"  {nm:<13} n=0")
                continue
            L.append(f"  {nm:<13} n={len(sub):<3} trap-to-end {int(sub['trap_end5'].sum()):>3}/{len(sub)} @MIN_RUN=5"
                     f"   {int(sub['trap_end3'].sum()):>3}/{len(sub)} @MIN_RUN=3"
                     f"   ON-share last 20k median {sub['on_share_20k'].median():.4f}")
        if len(off) and len(okr):
            orr, p = fisher(int(off["trap_end5"].sum()), len(off),
                            int(okr["trap_end5"].sum()), len(okr))
            L.append(f"  Fisher, trap-to-end in switched-off vs not-failed (MIN_RUN=5): "
                     f"OR={orr:.3g} p={p:.3g}")
            orr3, p3 = fisher(int(off["trap_end3"].sum()), len(off),
                              int(okr["trap_end3"].sum()), len(okr))
            L.append(f"  Fisher, same at MIN_RUN=3: OR={orr3:.3g} p={p3:.3g}")
    _write("RESULTS_letter_E3_fail.txt", "\n".join(L) + "\n")


def repro(fig: bool = False) -> None:
    """New instrumented runs vs the tuner's own record of the same (config, seed)."""
    df = load_results("repro")
    L = ["STAGE 0 REPRODUCIBILITY  --  instrumented rerun vs fair1's record (same config, same seed)",
         "=" * 78]
    ok_all = True
    for _, r in df.iterrows():
        if r.get("status") != "completed":
            L.append(f"  {r['run_id']}: {r.get('status')} {r.get('error')}")
            ok_all = False
            continue
        ref = base_params(r["base"])["seeds"].get(str(int(r["seed"])))
        diff = r["J_best"] - ref
        same = abs(diff) < 1e-9
        ok_all &= same
        L.append(f"  {r['run_id']:16s} J_best now {r['J_best']:.10f}  fair1 {ref:.10f}  diff {diff:+.2e}  "
                 f"{'IDENTICAL' if same else 'DIFFERENT'}")
    L.append("")
    L.append("-> identical: the probe does not perturb training and today's code/platform reproduce fair1;"
             " fair1's 35-seed records are valid baselines." if ok_all else
             "-> NOT identical: stop. Either the code or the platform differs from fair1, and fair1's"
             " records cannot serve as baselines without rerunning them here.")
    print("\n".join(L))
    (COMPARISON_DIR / "RESULTS_letter_E0_repro.txt").write_text("\n".join(L) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m letterexp.analyze")
    ap.add_argument("what", choices=["prereg", "repro", "e2", "e3", "e3_fail", "e4", "e5"])
    ap.add_argument("exp", nargs="?")
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args(argv)
    if a.what == "prereg":
        prereg(a.exp or "e2")
    else:
        {"repro": repro, "e2": e2, "e3": e3, "e3_fail": e3_fail,
         "e4": e4, "e5": e5}[a.what](fig=not a.no_fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
