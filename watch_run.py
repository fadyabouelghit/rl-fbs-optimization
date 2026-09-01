"""Live progress dashboard for a running (or finished) PPO/SAC training run.

Reads the run's flat files directly — it never touches the training process,
so it is safe to start, stop, and restart at any time, from any number of
terminals.

Usage:
    python watch_run.py                     # follow the newest run, refresh 30 s
    python watch_run.py --once              # print one snapshot and exit
    python watch_run.py --run run_2026-..._1-1-1_ppo_x --every 10
    python watch_run.py --baseline 0.4480 --optimum 0.5216

Reference values default to the scenario-1 world (2000x1500 m, 1 MBS):
do-nothing J = 0.4480 (694 users connected, no FBS traffic) and the 1-FBS
brute-force optimum J = 0.5216 (722 connected, 107 on the FBS), both from an
exhaustive grid sweep of the world. Pass --baseline/--optimum for other
scenarios.

Ctrl-C to quit; quitting the viewer does not affect the training run.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent / "ppo_runs"
DO_NOTHING = 0.4480
EXHAUSTIVE = 0.5216


def resolve(run: str) -> Path:
    if run == "latest":
        runs = sorted(RUNS_DIR.glob("run_*"), key=lambda p: p.stat().st_mtime)
        if not runs:
            raise SystemExit(f"No runs found under {RUNS_DIR}")
        return runs[-1]
    p = RUNS_DIR / run
    return p if p.exists() else Path(run)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open() as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open() as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return 0


def fnum(row: dict, key: str, default=None):
    v = row.get(key)
    if v in (None, "", "None"):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def objective(weights: dict, users: int, fbs: int, max_users: int,
              fbs_power: float, mode: str = "legacy_blend",
              fbs_power_max: float = 10.5, macro_budget: float = 0.0) -> float:
    """Recompute Φ for a reference outcome under a run's own cost settings.

    Reference *outcomes* are scenario facts; the Φ they map to depends on the
    weights AND on the reward mode, because the two modes normalise power
    differently:

      legacy_blend     : norm_power = P / (FBS max)         + epsilon padding
      controlled_blend : norm_power = P / (FBS max + macro) , no padding

    Getting this wrong only shows up when gamma > 0 (at gamma = 0 the power
    term is 1 either way), which is exactly when it matters most.
    """
    beta = weights.get("beta", 1.0)
    gamma = weights.get("gamma", 0.0)
    w_fbs = weights.get("fbs_weight", 0.4)
    eta = weights.get("fbs_exponent", 1.0)
    eps = weights.get("epsilon", 1e-4) if mode != "controlled_blend" else 0.0

    denom = fbs_power_max + (macro_budget if mode == "controlled_blend" else 0.0)
    norm_power = min(fbs_power / denom, 1.0) if denom > 0 else 0.0
    norm_users = min(users / max_users, 1.0)
    fbs_share = min(fbs / max(users, 1), 1.0)
    base = ((norm_users + eps) ** beta) * (((1.0 - norm_power) + eps) ** gamma)
    return (1.0 - w_fbs) * base + w_fbs * ((fbs_share + eps) ** eta)


def parse_started(stamp: str | None) -> float | None:
    """Run-directory timestamp ('2026-08-14_22-43-12') -> epoch seconds."""
    if not stamp:
        return None
    try:
        return datetime.strptime(stamp, "%Y-%m-%d_%H-%M-%S").timestamp()
    except ValueError:
        return None


def bar(frac: float, width: int = 34) -> str:
    frac = min(max(frac, 0.0), 1.0)
    filled = int(round(frac * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def scale_marker(value: float, lo: float, hi: float) -> str:
    """Show where an objective sits between do-nothing and the optimum."""
    span = hi - lo
    pct = 100.0 * (value - lo) / span if span else 0.0
    return f"{pct:+6.1f}% of headroom"


def snapshot(run_dir: Path, baseline, optimum, prev: dict) -> dict:
    cfg_path = run_dir / "experiment_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    ppo = cfg.get("ppo", {}) or {}
    env = cfg.get("env", {}) or {}
    prov = cfg.get("provenance", {}) or {}

    # Derive the reference objectives from THIS run's cost weights unless the
    # user pinned them. Reference outcomes (scenario 1, 2000x1500 m, 1 MBS):
    # do-nothing 694 users / 0 on the FBS; 1-FBS brute-force best 722 / 107.
    reward_cfg = env.get("reward") or {}
    weights = reward_cfg.get("weights") or {}
    world = env.get("world") or {}
    max_users = int(world.get("num_users") or 1000)
    # Runs written before the rename store the old mode name.
    mode = reward_cfg.get("mode", "legacy_blend")
    mode = {"ga_blend": "controlled_blend"}.get(mode, mode)
    n_fbs = int(env.get("num_fbs") or 1)
    kw = dict(
        mode=mode,
        fbs_power_max=10.5 * n_fbs,
        macro_budget=float(world.get("mbs_power") or 0.0) * int(world.get("num_mbs") or 0),
    )
    # do-nothing transmits nothing; the optimum runs one FBS at P = 10.0
    if baseline is None:
        baseline = objective(weights, 694, 0, max_users, 0.0, **kw)
    if optimum is None:
        optimum = objective(weights, 722, 107, max_users, 10.0, **kw)
    total = int(ppo.get("total_timesteps") or 0)
    now = time.time()

    # steps.csv is written in 256-row batches, so it lags by a few minutes at
    # the start. episodes.csv flushes every 8 rows, so early on it is the
    # fresher signal — use whichever implies more progress and flag estimates.
    logged = count_rows(run_dir / "steps.csv")
    episodes_done = count_rows(run_dir / "episodes.csv")
    ep_len = int(env.get("max_episode_steps") or 0)
    steps, approx = logged, False
    if ep_len and episodes_done * ep_len > logged:
        steps, approx = episodes_done * ep_len, True

    lines = []
    A = lines.append
    A("=" * 78)
    A(f" {run_dir.name}")
    A(
        f" code={cfg.get('code','?')}  algo={ppo.get('algo','ppo')}  "
        f"shaping={(env.get('reward') or {}).get('shaping','none')}  "
        f"gain={(env.get('reward') or {}).get('reward_gain',1.0)}  "
        f"actions={env.get('action_mode','delta')}  backend={prov.get('backend','?')}"
    )
    status = prov.get("status")
    A("=" * 78)

    # ---- progress ---------------------------------------------------- #
    frac = steps / total if total else 0.0
    A(
        f" progress {bar(frac)} {'~' if approx else ' '}{steps:,}/{total:,}"
        f"  ({100*frac:.1f}%)"
    )

    # Instantaneous rate needs two samples; fall back to the average since the
    # run started so --once still reports something useful. A finished run has
    # no meaningful "now" to average against, so report its status instead.
    rate, label = None, "recent"
    if status:
        pass
    elif prev.get("steps") is not None and steps > prev["steps"]:
        rate = (steps - prev["steps"]) / max(now - prev["time"], 1e-9)
    elif steps > 0:
        started = parse_started(prov.get("created_at"))
        if started and now > started:
            rate, label = steps / (now - started), "average"
    if rate:
        remaining = (total - steps) / rate if total > steps else 0
        eta = time.strftime("%H:%M", time.localtime(now + remaining))
        A(
            f" rate {rate:.2f} steps/s ({1/rate:.2f} s/step, {label})   "
            f"remaining {remaining/3600:5.1f} h   ETA {eta}"
        )
    elif status:
        A(f" status: {status}   finished_at {prov.get('finished_at','?')}")
    else:
        A(" rate  (starting up — steps.csv flushes in 256-row batches)")

    # ---- evaluation curve --------------------------------------------- #
    evals = read_csv(run_dir / "eval_log.csv")
    A("")
    beta = weights.get("beta", 1.0)
    A(f" DETERMINISTIC EVAL   (beta={beta}, fbs_weight={weights.get('fbs_weight', 0.4)})")
    A(f"   do-nothing={baseline:.4f}   brute-force-optimum={optimum:.4f}")
    if not evals:
        first = int(ppo.get("eval_every") or 0)
        A(f"   (none yet — first eval at {first:,} steps)" if first else "   (eval disabled)")
    else:
        # Under shaping='record' the policy has no incentive to END on its best
        # configuration, so best-in-episode is the number that matters there.
        A(f"   {'steps':>7} {'final J':>9} {'best J':>9} {'connected':>10} {'fbs':>6}   headroom (best)")
        for r in evals[-8:]:
            obj = fnum(r, "mean_objective")
            bst = fnum(r, "mean_best_objective")
            conn = fnum(r, "mean_total_connected", 0.0)
            fbs = fnum(r, "mean_fbs_connected", 0.0)
            if obj is None:
                A(f"   {r.get('timesteps',''):>7} {'n/a':>9} {'n/a':>9} {conn:>10.1f} {fbs:>6.1f}")
                continue
            ref = bst if bst is not None else obj
            A(
                f"   {r.get('timesteps',''):>7} {obj:>9.4f} "
                f"{('%9.4f' % bst) if bst is not None else '      n/a'} "
                f"{conn:>10.1f} {fbs:>6.1f}   {scale_marker(ref, baseline, optimum)}"
            )
        # Rank on the same quantity the table's headroom column uses
        # (best-in-episode when available), so the two never disagree.
        peaks = [
            fnum(r, "mean_best_objective", fnum(r, "mean_objective"))
            for r in evals
        ]
        peaks = [v for v in peaks if v is not None]
        if peaks:
            A(f"   best so far: {max(peaks):.4f}   ({scale_marker(max(peaks), baseline, optimum)})")

    # ---- training episodes -------------------------------------------- #
    eps = read_csv(run_dir / "episodes.csv")
    A("")
    if eps:
        recent = eps[-30:]
        conn = [fnum(r, "final_total_connected", 0.0) for r in recent]
        fbs = [fnum(r, "final_fbs_connected", 0.0) for r in recent]
        zeros = sum(1 for v in fbs if v == 0)
        A(f" TRAINING EPISODES   {len(eps)} total, last {len(recent)}:")
        A(
            f"   connected  mean {sum(conn)/len(conn):7.1f}   "
            f"fbs-served mean {sum(fbs)/len(fbs):6.1f}   "
            f"max {max(fbs):.0f}   do-nothing episodes {zeros}/{len(recent)}"
        )
        A("   fbs per episode: " + " ".join(f"{v:.0f}" for v in fbs[-20:]))
    else:
        A(" TRAINING EPISODES   (none finished yet)")

    A("=" * 78)
    print("\n".join(lines), flush=True)
    return {"steps": steps, "time": now, "done": bool(status) or (total and steps >= total)}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", default="latest", help="run name/path, or 'latest'")
    p.add_argument("--every", type=float, default=30.0, help="refresh seconds")
    p.add_argument("--once", action="store_true", help="print one snapshot and exit")
    p.add_argument("--baseline", type=float, default=None,
                   help="pin the do-nothing reference (default: derived from the run's weights)")
    p.add_argument("--optimum", type=float, default=None,
                   help="pin the optimum reference (default: derived from the run's weights)")
    args = p.parse_args()

    run_dir = resolve(args.run)
    prev: dict = {}
    try:
        while True:
            if not args.once:
                print("\033[2J\033[H", end="")  # clear screen
            prev = snapshot(run_dir, args.baseline, args.optimum, prev)
            if args.once:
                break
            if prev.get("done"):
                print("\n run finished — exiting viewer")
                break
            time.sleep(args.every)
    except KeyboardInterrupt:
        print("\n viewer stopped (training is unaffected)")


if __name__ == "__main__":
    main()
