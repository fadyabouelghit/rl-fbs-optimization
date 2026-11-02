"""Stage 1 (E1): clamped-actor check of the flip formula (C2). No training.

    python -m letterexp.e1_flip              # numpy grid + env check + historical runs
    python -m letterexp.e1_flip --steps 1e5  # quick look

Model. The actor's on/off output is frozen at mu = -1, Gaussian noise of std s
is added, the result is clipped to [-1, 1] (as SB3 does), and the sticky rule
is applied: the gene changes only if |a| >= k, to 1[a >= 0].

    off -> on per step        P = Q((1 + k) / s)       (a >= k  <=>  noise >= 1 + k)
    on  -> off at next step   P = Phi((1 - k) / s)     (a <= -k <=>  noise <= 1 - k)

Measured three ways:
  chain        one 10^6-step Markov chain per (s, k) starting off; off->on is
               counted over off steps, switch-back over the step right after
               each off->on flip.
  on-trials    10^6 independent steps from the on state (so switch-back is
               measurable even where flips are too rare for the chain).
  env          the real ``FlyingBaseStationEnv._binary_update`` is checked for
               exact agreement with the numpy rule on 10^6 random
               (gene, action) pairs at its fixed k = 0.5, and then drives the
               s = 0.5 chain itself.
Plus the four pinned seed-0 DDPG runs already on disk (steps.csv), within
episodes only (reset re-draws the gene, so cross-episode diffs are not flips).
Intervals are Clopper-Pearson 95%.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from .paths import COMPARISON_DIR, FIG_DIR
from .stats import clopper_pearson

S_GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5)
K_GRID = (0.3, 0.5, 0.7)
HIST_RUNS = ("t3_bn0.5_lr2e-5", "t4_an0.15_lr5e-5", "t4_tau0.01_lr5e-5", "t7_g0.8_s0")
SEED = 4242


def predicted(s: float, k: float) -> tuple[float, float]:
    return float(norm.sf((1 + k) / s)), float(norm.cdf((1 - k) / s))


def sticky(g: np.ndarray, a: np.ndarray, k: float) -> np.ndarray:
    return np.where(np.abs(a) >= k, (a >= 0).astype(g.dtype), g)


def chain(s: float, k: float, n: int, rng, update=None) -> dict:
    a = np.clip(-1.0 + rng.normal(0.0, s, n), -1.0, 1.0)
    big = np.abs(a) >= k
    up = a >= 0
    g = np.empty(n + 1, np.int8)
    g[0] = 0
    if update is None:
        for t in range(n):                       # sticky rule, sequential by nature
            g[t + 1] = up[t] if big[t] else g[t]
    else:
        cur = np.zeros(1, np.float32)
        for t in range(n):
            cur = update(cur, a[t:t + 1].astype(np.float32))
            g[t + 1] = int(cur[0])
    prev, nxt = g[:-1], g[1:]
    off = prev == 0
    flips = np.flatnonzero(off & (nxt == 1))     # step t: off -> on
    nb = flips[flips + 1 < n]                    # the step after each flip
    back = int(np.sum(g[nb + 2] == 0))
    return {"n_off": int(off.sum()), "n_on_flip": int(len(flips)),
            "n_back_trials": int(len(nb)), "n_back": back}


def on_trials(s: float, k: float, n: int, rng) -> dict:
    a = np.clip(-1.0 + rng.normal(0.0, s, n), -1.0, 1.0)
    return {"n": n, "n_off": int(np.sum(a <= -k))}


def env_agreement(n: int, rng) -> tuple[bool, int]:
    from dataclasses import replace

    from ppo.env import FlyingBaseStationEnv
    from tuning.protocol import DEFAULT
    from tuning.spaces import build_experiment

    from .runner import base_params
    exp = build_experiment("ddpg", base_params("ddpg_t27")["params"],
                           replace(DEFAULT, backend="analytic"), 0, "e1")
    env = FlyingBaseStationEnv(exp.env, backend_kind="analytic")
    try:
        assert env.config.binary_mode == "sticky"
        g = rng.integers(0, 2, n).astype(np.float32)
        a = rng.uniform(-1, 1, n).astype(np.float32)
        a[: n // 100] = rng.choice([-0.5, 0.5, -1.0, 1.0, 0.0], n // 100)   # boundaries
        mine = sticky(g, a, 0.5)
        theirs = env._binary_update(g, a)
        return bool(np.array_equal(mine, theirs)), n
    finally:
        env.close()


def env_update_fn():
    from dataclasses import replace

    from ppo.env import FlyingBaseStationEnv
    from tuning.protocol import DEFAULT
    from tuning.spaces import build_experiment

    from .runner import base_params
    exp = build_experiment("ddpg", base_params("ddpg_t27")["params"],
                           replace(DEFAULT, backend="analytic"), 0, "e1")
    env = FlyingBaseStationEnv(exp.env, backend_kind="analytic")
    return env._binary_update, env


# --------------------------------------------------------------------------- #
# historical pinned runs
# --------------------------------------------------------------------------- #
def historical(run_dir: Path, block: int = 1000) -> dict:
    import pandas as pd

    df = pd.read_csv(run_dir / "steps.csv", usecols=["timesteps", "env", "episode", "step",
                                                       "fbs0_power_status"])
    cfg = json.loads((run_dir / "experiment_config.json").read_text())
    s_b = cfg.get("ddpg", {}).get("binary_noise_std")
    df = df.sort_values(["env", "timesteps"]).reset_index(drop=True)
    g = (df["fbs0_power_status"].to_numpy() > 0.5).astype(int)
    same_ep = (df["episode"].to_numpy()[1:] == df["episode"].to_numpy()[:-1]) & \
              (df["env"].to_numpy()[1:] == df["env"].to_numpy()[:-1])
    ts = df["timesteps"].to_numpy()
    # pinned window: from the first 1000-step block after which every block is >= 95% off
    blocks = ts // block
    off_share = pd.Series(1 - g).groupby(blocks).mean()
    ok = (off_share >= 0.95).to_numpy()
    start_block = None
    for i in range(len(ok)):
        if ok[i:].all():
            start_block = int(off_share.index[i])
            break
    if start_block is None:
        return {"run": run_dir.name, "s_b": s_b, "pinned_from": None}
    t0 = start_block * block
    m = same_ep & (ts[1:] > t0)
    prev, nxt = g[:-1][m], g[1:][m]
    n_off = int(np.sum(prev == 0))
    n_flip = int(np.sum((prev == 0) & (nxt == 1)))
    # switch-back: flip at row i -> row i+1 (same episode) is off?
    idx = np.flatnonzero(m)
    flip_rows = idx[(g[:-1][idx] == 0) & (g[1:][idx] == 1)] + 1     # row that is newly on
    nb = [r for r in flip_rows if r + 1 < len(g) and same_ep[r] and ts[r] > t0]
    back = int(sum(g[r + 1] == 0 for r in nb))
    return {"run": run_dir.name, "s_b": s_b, "pinned_from": t0, "n_off": n_off,
            "n_on_flip": n_flip, "n_back_trials": len(nb), "n_back": back}


def find_hist_runs(root: Path) -> list[Path]:
    out = []
    for tag in HIST_RUNS:
        hits = sorted(root.glob(f"run_*_ddpg_{tag}"))
        out.extend(h for h in hits if (h / "steps.csv").is_file())
    return out


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m letterexp.e1_flip")
    ap.add_argument("--steps", type=float, default=1e6)
    ap.add_argument("--hist-root", default=None, help="default <repo>/ddpg_runs (read-only)")
    ap.add_argument("--no-env-chain", action="store_true",
                    help="skip driving the s=0.5 chain through the env (slow, ~1 min)")
    a = ap.parse_args(argv)
    n = int(a.steps)
    rng = np.random.default_rng(SEED)

    from ppo.paths import REPO_ROOT
    rows = []
    for k in K_GRID:
        for s in S_GRID:
            pf, pb = predicted(s, k)
            c = chain(s, k, n, rng)
            o = on_trials(s, k, n, rng)
            rows.append({"s": s, "k": k, "pred_flip": pf, "pred_back": pb, **c,
                         "on_trials": o["n"], "on_trials_off": o["n_off"]})

    agree, n_agree = env_agreement(n, rng)
    env_row = None
    if not a.no_env_chain:
        upd, env = env_update_fn()
        try:
            env_row = {"s": 0.5, "k": 0.5, **chain(0.5, 0.5, n, rng, update=upd)}
        finally:
            env.close()

    hist_root = Path(a.hist_root) if a.hist_root else REPO_ROOT / "ddpg_runs"
    hist = [historical(r) for r in find_hist_runs(hist_root)]

    # ---------------- report ----------------
    L = []
    L.append("RESULTS_letter_E1  --  clamped-actor check of the flip formula (claim C2)")
    L.append("=" * 78)
    L.append(f"numpy chain: {n:,} steps per (s, k), mu = -1 frozen, a = clip(mu + N(0, s^2), -1, 1),")
    L.append("sticky rule g' = 1[a>=0] if |a|>=k else g. Seed %d. Intervals: Clopper-Pearson 95%%." % SEED)
    L.append("")
    L.append(f"{'k':>4} {'s':>4} | {'pred off->on':>12} {'meas':>10} {'95% CI':>23} {'rel.err':>8} | "
             f"{'pred back':>9} {'meas(chain)':>11} {'meas(on-trials)':>15} {'95% CI (on-trials)':>20} {'rel.err':>8}")
    worst = 0.0
    for r in rows:
        p, lo, hi = clopper_pearson(r["n_on_flip"], r["n_off"])
        rel = (p - r["pred_flip"]) / r["pred_flip"] if r["n_on_flip"] else float("nan")
        pb_chain = r["n_back"] / r["n_back_trials"] if r["n_back_trials"] else float("nan")
        q, qlo, qhi = clopper_pearson(r["on_trials_off"], r["on_trials"])
        relb = (q - r["pred_back"]) / r["pred_back"]
        r.update(meas_flip=p, flip_lo=lo, flip_hi=hi, meas_back_chain=pb_chain,
                 meas_back=q, back_lo=qlo, back_hi=qhi)
        inside_f = (lo <= r["pred_flip"] <= hi)
        inside_b = (qlo <= r["pred_back"] <= qhi)
        r.update(inside_flip=inside_f, inside_back=inside_b)
        if r["n_on_flip"] >= 100:
            worst = max(worst, abs(rel))
        L.append(f"{r['k']:>4} {r['s']:>4} | {r['pred_flip']:>12.4e} {p:>10.4e} [{lo:.3e}, {hi:.3e}] "
                 f"{rel:>+8.3f} | {r['pred_back']:>9.4f} {pb_chain:>11.4f} {q:>15.4f} "
                 f"[{qlo:.4f}, {qhi:.4f}] {relb:>+8.4f}"
                 + ("" if inside_f or r["n_on_flip"] == 0 else "  <- pred outside CI (flip)")
                 + ("" if inside_b else "  <- pred outside CI (back)"))
    L.append("")
    nf = sum(1 for r in rows if r["n_on_flip"] > 0)
    L.append(f"off->on: predicted value inside the 95% CI in "
             f"{sum(r['inside_flip'] for r in rows if r['n_on_flip'] > 0)}/{nf} cells with >= 1 flip; "
             f"cells with 0 flips are consistent iff pred < CI upper bound: "
             f"{all(r['pred_flip'] <= r['flip_hi'] for r in rows if r['n_on_flip'] == 0)}")
    L.append(f"switch-back: predicted value inside the 95% CI in {sum(r['inside_back'] for r in rows)}/{len(rows)} cells")
    L.append(f"largest |relative error| of off->on among cells with >= 100 flips: {worst:.3f}")
    L.append("")
    L.append(f"ENV CHECK: ppo.env._binary_update == numpy sticky rule at k = 0.5 on {n_agree:,} random "
             f"(gene, action) pairs (1% at the exact boundaries +-0.5, +-1, 0): {'EXACT MATCH' if agree else 'MISMATCH'}")
    if env_row:
        p, lo, hi = clopper_pearson(env_row["n_on_flip"], env_row["n_off"])
        pf, pb = predicted(0.5, 0.5)
        q = env_row["n_back"] / max(env_row["n_back_trials"], 1)
        L.append(f"  chain driven by the env's own _binary_update, s = 0.5: off->on {p:.4e} "
                 f"[{lo:.3e}, {hi:.3e}] vs pred {pf:.4e}; switch-back {q:.4f} "
                 f"({env_row['n_back']}/{env_row['n_back_trials']}) vs pred {pb:.4f}")
    L.append("")
    L.append("HISTORICAL PINNED RUNS (ddpg_runs/, seed 0, read-only). Window = from the first 1000-step")
    L.append("block after which every block is >= 95% off; within-episode transitions only.")
    L.append("Caveat: steps.csv has no actor output, so mu = -1 during the window is inferred, not measured.")
    if not hist:
        L.append(f"  (no steps.csv found under {hist_root})")
    for h in hist:
        if h.get("pinned_from") is None:
            L.append(f"  {h['run']}: no pinned-off window found")
            continue
        pf, pb = predicted(h["s_b"], 0.5)
        p, lo, hi = clopper_pearson(h["n_on_flip"], h["n_off"])
        q, qlo, qhi = clopper_pearson(h["n_back"], h["n_back_trials"])
        L.append(f"  {h['run']}  s_b={h['s_b']}  from t={h['pinned_from']}: off->on "
                 f"{h['n_on_flip']}/{h['n_off']} = {p:.3e} [{lo:.2e}, {hi:.2e}] (pred {pf:.3e}); "
                 f"switch-back {h['n_back']}/{h['n_back_trials']} = {q:.3f} [{qlo:.3f}, {qhi:.3f}] (pred {pb:.3f})")
    if hist:
        tot = {k: sum(h.get(k, 0) or 0 for h in hist if h.get("pinned_from") is not None)
               for k in ("n_off", "n_on_flip", "n_back_trials", "n_back")}
        if tot["n_off"]:
            p, lo, hi = clopper_pearson(tot["n_on_flip"], tot["n_off"])
            q, qlo, qhi = clopper_pearson(tot["n_back"], tot["n_back_trials"])
            pf, pb = predicted(0.5, 0.5)
            L.append(f"  POOLED: off->on {p:.3e} [{lo:.2e}, {hi:.2e}] vs pred {pf:.3e}; "
                     f"switch-back {q:.3f} [{qlo:.3f}, {qhi:.3f}] vs pred {pb:.3f}")
    L.append("")
    L.append("VERDICT (C2): fill in after reading the table -- supported if the predicted rates sit inside")
    L.append("the CIs across the grid and the env check is an exact match.")

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    out = COMPARISON_DIR / "RESULTS_letter_E1.txt"
    out.write_text("\n".join(L) + "\n")
    (COMPARISON_DIR / "RESULTS_letter_E1.json").write_text(
        json.dumps({"grid": rows, "env_agree": agree, "env_chain": env_row, "historical": hist},
                   indent=1, default=float))
    _figure(rows, hist)
    print("\n".join(L))
    print(f"\n-> {out}")
    return 0


def _figure(rows, hist) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    colors = {0.3: "#2a9d8f", 0.5: "#264653", 0.7: "#e76f51"}
    s_fine = np.linspace(0.18, 1.55, 200)
    for k in K_GRID:
        c = colors[k]
        ax[0].plot(s_fine, norm.sf((1 + k) / s_fine), color=c, lw=1.2, label=f"k = {k} predicted")
        ax[1].plot(s_fine, norm.cdf((1 - k) / s_fine), color=c, lw=1.2, label=f"k = {k} predicted")
        rk = [r for r in rows if r["k"] == k]
        ss = np.array([r["s"] for r in rk if r["n_on_flip"] > 0])
        pm = np.array([r["meas_flip"] for r in rk if r["n_on_flip"] > 0])
        lo = np.array([r["flip_lo"] for r in rk if r["n_on_flip"] > 0])
        hi = np.array([r["flip_hi"] for r in rk if r["n_on_flip"] > 0])
        ax[0].errorbar(ss, pm, yerr=[pm - lo, hi - pm], fmt="o", ms=4, color=c, capsize=2)
        ss = np.array([r["s"] for r in rk])
        q = np.array([r["meas_back"] for r in rk])
        ax[1].errorbar(ss, q, yerr=[q - np.array([r["back_lo"] for r in rk]),
                                    np.array([r["back_hi"] for r in rk]) - q],
                       fmt="o", ms=4, color=c, capsize=2)
    for h in hist:
        if h.get("pinned_from") is None or not h["n_off"]:
            continue
        ax[0].plot([h["s_b"]], [h["n_on_flip"] / h["n_off"]], "x", color="k", ms=7)
        if h["n_back_trials"]:
            ax[1].plot([h["s_b"]], [h["n_back"] / h["n_back_trials"]], "x", color="k", ms=7)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("binary noise std s")
    ax[0].set_ylabel("per-step off -> on probability")
    ax[0].set_title("Escape from mu = -1:  Q((1+k)/s)")
    ax[1].set_xlabel("binary noise std s")
    ax[1].set_ylabel("P(on -> off at next step)")
    ax[1].set_title("Switch-back:  Phi((1-k)/s)")
    ax[0].legend(fontsize=8)
    ax[1].plot([], [], "x", color="k", label="pinned DDPG runs (k = 0.5)")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"F1_flip_rates.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
