"""Step B: what the critic and the actor say about each station's on/off dim.

    ./venv/bin/python -m letterexp.fbs2_critic --workers 8

For every DDPG and TD3 confirmation run of fair2, on the 2-FBS probe set, and
for each station j restricted to the probe states where station j is currently
OFF:

    mu_j      the actor's deterministic output on that station's dim
    gap_j     Q1(s, a | a_j = +1) - Q1(s, a | a_j = -1), other dims at mu(s)
    slope_j   dQ1/da_j at a = mu(s)  (autograd)

The pattern the 1-FBS work calls the trap is gap_j > 0 while slope_j < 0: the
critic rates switching the station ON higher, yet the local slope the
deterministic policy gradient follows points the other way.

TD3 has twin critics; Q1 is used throughout and Q2 is recorded once as a check.

Writes letter_runs_2fbs/analysis/critic_best.csv. Nothing trains.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ppo.paths import REPO_ROOT

FAIR2 = REPO_ROOT / "letter_runs_2fbs" / "tuning" / "fair2"
PROBE = REPO_ROOT / "letter_runs_2fbs" / "probe_sets" / "probe_2-1-1_pyqd-fast_n256_s20261009.npz"
OUT_DIR = REPO_ROOT / "letter_runs_2fbs" / "analysis"
SAT, LOCK_FRAC = 0.96, 0.95
ALGOS = ("td3", "ddpg")


def _binary_dims(env_cfg):
    from ppo.env import state_labels
    labels = state_labels(env_cfg)
    idx = [i for i, n in enumerate(labels) if n.endswith("_power_status")]
    return idx, [labels[i] for i in idx]


def _critics(model, obs, a):
    out = model.policy.critic(obs, a)
    return out if isinstance(out, (tuple, list)) else (out,)


def analyse_one(algo: str, run_dir: str, trial: int, seed: int) -> dict:
    import torch as th

    if algo == "td3":
        from stable_baselines3 import TD3 as Algo
        from td3.runs import load_run
        exp = load_run(Path(run_dir))[1]
    else:
        from stable_baselines3 import DDPG as Algo
        from ddpg.runs import load_run
        exp = load_run(Path(run_dir))[1]

    d = np.load(PROBE)
    obs = th.as_tensor(d["obs"].astype(np.float32))
    states = d["states"]
    idx, labels = _binary_dims(exp.env)

    model = Algo.load(str(Path(run_dir) / "best_model.zip"), device="cpu")
    model.policy.set_training_mode(False)
    with th.no_grad():
        a_mu = model.policy.actor(obs)

    row = {"algo": algo, "trial": trial, "seed": seed, "run_dir": str(run_dir)}
    for j, b in enumerate(idx):
        off = states[:, b] <= 0.5                     # station j currently off
        mu_j = a_mu[:, b].cpu().numpy()

        # gap: Q1 with that dim forced to +1 vs -1, everything else at mu(s)
        with th.no_grad():
            a_on, a_off = a_mu.clone(), a_mu.clone()
            a_on[:, b], a_off[:, b] = 1.0, -1.0
            q_on = _critics(model, obs, a_on)
            q_off = _critics(model, obs, a_off)
        gap1 = (q_on[0] - q_off[0]).reshape(-1).cpu().numpy()
        gap2 = ((q_on[1] - q_off[1]).reshape(-1).cpu().numpy()
                if len(q_on) > 1 else None)

        # slope: dQ1/da_j at a = mu(s)
        a_req = a_mu.clone().requires_grad_(True)
        q = _critics(model, obs, a_req)[0]
        g = th.autograd.grad(q.sum(), a_req)[0][:, b].detach().cpu().numpy()

        n_off = int(off.sum())
        row[f"fbs{j}_n_off"] = n_off
        row[f"fbs{j}_mu_median"] = float(np.median(mu_j[off])) if n_off else float("nan")
        row[f"fbs{j}_sat_share"] = float(np.mean(mu_j[off] < -SAT)) if n_off else float("nan")
        row[f"fbs{j}_locked_probe"] = bool(n_off and np.mean(mu_j[off] < -SAT) >= LOCK_FRAC)
        row[f"fbs{j}_gap_median"] = float(np.median(gap1[off])) if n_off else float("nan")
        row[f"fbs{j}_slope_median"] = float(np.median(g[off])) if n_off else float("nan")
        row[f"fbs{j}_share_gap_pos"] = float(np.mean(gap1[off] > 0)) if n_off else float("nan")
        row[f"fbs{j}_share_slope_neg"] = float(np.mean(g[off] < 0)) if n_off else float("nan")
        row[f"fbs{j}_share_both"] = (float(np.mean((gap1[off] > 0) & (g[off] < 0)))
                                     if n_off else float("nan"))
        if gap2 is not None:
            row[f"fbs{j}_share_gap_pos_q2"] = float(np.mean(gap2[off] > 0)) if n_off else float("nan")
    return row


def collect(algos=ALGOS, workers: int = 8) -> pd.DataFrame:
    jobs = []
    for algo in algos:
        f = FAIR2 / f"{algo}_confirmation.json"
        conf = json.loads(f.read_text())
        for r in conf["finalists"]:
            for s, rd in r.get("run_dirs", {}).items():
                if rd:
                    jobs.append((algo, rd, int(r["trial"]), int(s)))
    print(f"{len(jobs)} runs with {workers} workers", flush=True)
    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(analyse_one, *j): j for j in jobs}
        for fu in as_completed(futs):
            j = futs[fu]
            try:
                rows.append(fu.result())
            except Exception as exc:
                rows.append({"algo": j[0], "trial": j[2], "seed": j[3],
                             "run_dir": j[1], "error": f"{type(exc).__name__}: {exc}"})
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m letterexp.fbs2_critic")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    if not PROBE.is_file():
        raise SystemExit(f"probe set missing: {PROBE}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = collect(workers=a.workers)
    out = OUT_DIR / "critic_best.csv"
    df.to_csv(out, index=False)
    print(f"\n-> {out}  ({len(df)} rows)")
    if "error" in df:
        print(f"!! {int(df.error.notna().sum())} errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
