"""Step A: re-score every fair2 best model and count how many stations it uses.

    ./venv/bin/python -m letterexp.fbs2_rescore --workers 8

With 2 FBS, J equals the do-nothing level only when BOTH stations are off, so a
run that leaves one station permanently off is invisible to the failure
measure. This re-scores each confirmation run on the tuner's own 20 start
states (same call path as tuning.objective.score_run, so J must reproduce
exactly) and records, per test state, how many stations the scored policy ends
with switched on.

For a station that ends off, it also asks the actor what it would do there:
a station is called LOCKED OFF when at least 95% of the test states in which it
is off have actor output below -0.96, the same rule letterexp/probe.py uses.

Nothing here trains, and nothing is written outside letter_runs_2fbs/ and
lab/comparison/.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ppo.paths import REPO_ROOT

FAIR2 = REPO_ROOT / "letter_runs_2fbs" / "tuning" / "fair2"
OUT_DIR = REPO_ROOT / "letter_runs_2fbs" / "analysis"
SAT = 0.96          # |mu| above this counts as saturated   (letterexp/probe.py)
LOCK_FRAC = 0.95    # share of off states that must be saturated
CLASS_MIN = 15      # >= 15 of 20 test states to earn a class label
ALGOS = ("ppo", "td3", "ddpg")


def _load_assess():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import assess_models
    return assess_models


def _binary_dims(env_cfg):
    """(indices, labels) of the power_status genes, read from the labels."""
    from ppo.env import state_labels
    labels = state_labels(env_cfg)
    idx = [i for i, n in enumerate(labels) if n.endswith("_power_status")]
    return idx, [labels[i] for i in idx]


def _actor_mu(algo: str, model, obs_batch: np.ndarray) -> np.ndarray:
    """Deterministic actor output (PPO: the Gaussian mean), shape (n, act_dim)."""
    import torch as th
    t = th.as_tensor(np.asarray(obs_batch, dtype=np.float32))
    with th.no_grad():
        if algo == "ppo":
            return model.policy.get_distribution(t).distribution.mean.cpu().numpy()
        return model.policy.actor(t).cpu().numpy()


def _rebuild_obs(env, state_row, globals_row) -> np.ndarray:
    """The observation the policy would see at a logged state.

    Uses the env's own normalisation and observation assembly rather than
    reimplementing it, so it cannot drift from training.
    """
    env._state = np.asarray(state_row, dtype=np.float32)
    env._global_state = np.asarray(globals_row[:2], dtype=np.float32)
    if env.config.obs_tier_metrics:
        env._tier_obs = np.asarray(globals_row[2:6], dtype=np.float32)
    return env._get_obs()


def score_one(algo: str, run_dir: str, trial: int, seed: int, ref_J: float,
              model: str = "best") -> dict:
    from tuning.protocol import DEFAULT
    from tuning.worker import protocol_from_dict
    from dataclasses import replace, asdict

    protocol = replace(DEFAULT, code="2-1-1")
    am = _load_assess()
    run_dir = Path(run_dir)

    # hybrid variants train through their parent package, so they load with it
    fam = {"ddpg_hybrid": "ddpg", "ddpg_hybrid_qd": "ddpg",
           "td3_hybrid": "td3", "td3_hybrid_qd": "td3"}.get(algo, algo)
    if fam == "ppo":
        from ppo.evaluate import evaluate_run
        from ppo.runs import load_run
        from stable_baselines3 import PPO as Algo
        exp = load_run(run_dir).exp
    elif fam == "td3":
        from td3.evaluate import evaluate_run
        from td3.runs import load_run
        from stable_baselines3 import TD3 as Algo
        exp = load_run(run_dir)[1]
    else:
        from ddpg.evaluate import evaluate_run
        from ddpg.runs import load_run
        from stable_baselines3 import DDPG as Algo
        exp = load_run(run_dir)[1]

    states = am.start_states(exp.env, protocol.objective_episodes,
                             seed=protocol.objective_seed)
    rep = evaluate_run(run_dir, episodes=protocol.objective_episodes,
                       initial_states=states, backend=protocol.backend,
                       save=False, make_plots=False, model=model)

    finals = np.array([r.metrics_df["reward_objective"].iloc[-1] for r in rep.episodes])
    J = float(finals.mean())

    idx, labels = _binary_dims(exp.env)
    n_fbs = exp.env.num_fbs

    # per test state: the gene values at the final step
    gene = np.zeros((len(rep.episodes), n_fbs), dtype=int)
    obs_rows, glob_rows, state_rows = [], [], []
    for i, r in enumerate(rep.episodes):
        last = r.state_df.iloc[-1]
        for j in range(n_fbs):
            gene[i, j] = int(last[(f"fbs{j}", "power_status")] > 0.5)
        state_rows.append([last[(f"fbs{j}", g)] for j in range(n_fbs)
                           for g in ("x", "y", "height", "power", "power_status")])
        g = last["global"] if "global" in last.index.get_level_values(0) else None
        glob_rows.append([float(last[("global", k)]) for k in
                          ("fbs_delta", "total_delta", "fbs_frac", "mbs_frac",
                           "total_frac", "norm_power")])

    # actor output at those same final states
    from ppo.env import FlyingBaseStationEnv
    env = FlyingBaseStationEnv(exp.env, backend_kind="analytic")   # obs assembly only
    try:
        for s, gl in zip(state_rows, glob_rows):
            obs_rows.append(_rebuild_obs(env, s, gl))
    finally:
        env.close()
    model = Algo.load(str(run_dir / "best_model.zip"), device="cpu")
    mu = _actor_mu(fam, model, np.stack(obs_rows))

    n_active = gene.sum(axis=1)
    row = {"algo": algo, "trial": trial, "seed": seed, "run_dir": str(run_dir),
           "model": model,
           "J": J, "J_ref": ref_J, "J_match": bool(abs(J - ref_J) < 1e-9),
           "n_states": len(gene),
           "states_2on": int(np.sum(n_active == 2)),
           "states_1on": int(np.sum(n_active == 1)),
           "states_0on": int(np.sum(n_active == 0)),
           "fbs_connected": float(np.mean(
               [r.metrics_df["fbs_connected"].iloc[-1] for r in rep.episodes])),
           }
    if row["states_2on"] >= CLASS_MIN:   row["klass"] = "both_on"
    elif row["states_1on"] >= CLASS_MIN: row["klass"] = "one_on"
    elif row["states_0on"] >= CLASS_MIN: row["klass"] = "none_on"
    else:                                row["klass"] = "mixed"

    # which station is off, and is it locked off
    for j in range(n_fbs):
        off = gene[:, j] == 0
        row[f"fbs{j}_off_states"] = int(off.sum())
        mu_j = mu[:, idx[j]]
        row[f"fbs{j}_mu_median"] = float(np.median(mu_j))
        share = float(np.mean(mu_j[off] < -SAT)) if off.any() else float("nan")
        row[f"fbs{j}_sat_share_when_off"] = share
        row[f"fbs{j}_locked_off"] = bool(off.any() and share >= LOCK_FRAC)
    if row["klass"] == "one_on":
        off_counts = [row[f"fbs{j}_off_states"] for j in range(n_fbs)]
        dominant = int(np.argmax(off_counts))
        row["off_station"] = dominant
        row["fixed_off_station"] = bool(off_counts[dominant] >= CLASS_MIN)
        row["off_station_locked"] = bool(row[f"fbs{dominant}_locked_off"])
    else:
        row["off_station"] = -1
        row["fixed_off_station"] = False
        row["off_station_locked"] = False
    return row


def collect(algos=ALGOS, workers: int = 8) -> pd.DataFrame:
    jobs = []
    for algo in algos:
        f = FAIR2 / f"{algo}_confirmation.json"
        if not f.is_file():
            print(f"[skip] {f} missing")
            continue
        conf = json.loads(f.read_text())
        for r in conf["finalists"]:
            for s, rd in r.get("run_dirs", {}).items():
                ref = r["seeds"].get(str(s))
                if rd and ref is not None:
                    jobs.append((algo, rd, int(r["trial"]), int(s), float(ref)))
    print(f"{len(jobs)} runs to re-score with {workers} workers", flush=True)
    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(score_one, *j): j for j in jobs}
        for fu in as_completed(futs):
            j = futs[fu]
            try:
                rows.append(fu.result())
            except Exception as exc:
                rows.append({"algo": j[0], "trial": j[2], "seed": j[3],
                             "run_dir": j[1], "error": f"{type(exc).__name__}: {exc}"})
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m letterexp.fbs2_rescore")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--algos", nargs="+", default=list(ALGOS))
    a = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = collect(a.algos, a.workers)
    out = OUT_DIR / "rescore_best.csv"
    df.to_csv(out, index=False)
    print(f"\n-> {out}  ({len(df)} rows)")
    bad = df[df.get("error").notna()] if "error" in df else df.iloc[:0]
    if len(bad):
        print(f"!! {len(bad)} runs errored")
    mism = df[df.get("J_match") == False] if "J_match" in df else df.iloc[:0]
    print(f"J mismatches: {len(mism)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
