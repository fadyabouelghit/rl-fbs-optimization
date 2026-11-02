"""Experiment plans and a resumable worker pool.

    python -m letterexp.launch plan   e2 --seeds 0 1          # dry run: runs, time, disk
    python -m letterexp.launch run    e2 --seeds 0 1 --workers 8
    python -m letterexp.launch status e2

Every run is a subprocess (``letterexp.runner``) whose PPO/DDPG/TD3 run
directories and ledgers are redirected to ``letter_runs/<experiment>/``.
A run whose identity key (algo, params, seed, protocol, instrumentation)
already has a completed result anywhere under ``letter_runs/`` is reused, not
retrained: the tuned anchor arms are shared between E2 and E5 this way.

Plans
-----
smoke     analytic backend, 3k steps, DDPG t27 + TD3 t28, seed 0     (Stage 0)
e3_fail2  failure-prone fair2 (2 FBS) finalists rerun instrumented: TD3 t35,
          TD3 t4, DDPG t30, DDPG t40, seeds 0-34
m5_long   review round 1 Run D: 300k steps, top-3 DDPG and TD3 plus PPO t35,
          seeds 0-9, scored at the 100k checkpoint and at 300k
m2_onperiod review round 1 Run C: min-on and forced-on interventions on the
          fair2 winners (training only; evaluation and scoring are unwrapped)
m1_hybrid2 review round 1 Run B: Bernoulli head at M = 2, slope-trained vs
          comparison-trained, from the fair2 winners (transfer test, not tuned)
m1_top3   review round 1 Run A: top-3 fair2 configs of PPO, DDPG and TD3,
          unchanged, seeds 0-19, with the opt-in position log
e2_2fbs   2-FBS noise sweep: fair2 winners {DDPG t15, TD3 t51} x binary_noise_std
          in {0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6, 2.0} plus the tuned anchor
          {0.1, 0.3, 0.5, 0.8, 1.2} plus the tuned value as anchor, seeds 0-19,
          fair2 protocol. LABELED ABLATION. Run with LETTER_RUNS_DIR=letter_runs_2fbs.
e2_ppo    PPO fair1 t43 unchanged, seeds 0-19, fair1 protocol: the on-policy
          check-up arm (no ablation; PPO has no binary noise knob)
s0_10k    pyqd-fast, 10k steps, DDPG t27 + TD3 t28, seed 0          (Stage 0)
repro     pyqd-fast, full fair1 protocol, DDPG t27 + TD3 t28, seed 0: must
          reproduce fair1's seed-0 J, proving the probe does not perturb
          training on this machine                                  (Stage 0)
e3_fail   the failure-prone fair1 finalists (DDPG t55, DDPG t19, TD3 t45)
          rerun unchanged with the probe on, seeds 0-34: do the failures of
          tuner-passed configs coincide with a persistent stuck-off trap?
e2        {DDPG t27, TD3 t28} x binary_noise_std grid, plus the tuned value
          as an anchor arm. LABELED ABLATION: only s_b moves off the tuned point.
e4_pilot  2 seeds of ddpg_hybrid / td3_hybrid at the base config's other knobs
          and gumbel_tau 1.0: an implementation check, NOT the comparison. The
          comparison uses the hybrid's own tuner campaign (fair1h).
e4_pilot_qd  same, for the Q-difference hybrids (ddpg_hybrid_qd / td3_hybrid_qd)
e5        TD3 t28 x policy_delay {1, 2, 4}, tau 0.02. LABELED ABLATION; delay 1
          is t28 itself.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List

from ppo.paths import REPO_ROOT

from .paths import LETTER_RUNS_DIR, assert_allowed, experiment_dir
from .runner import base_params, spec_key

HYBRID_TUNING_DIR = LETTER_RUNS_DIR / "e4_tuning"
HYBRID_CAMPAIGN = "fair1h"
HYBRID_CAMPAIGN_DIR = HYBRID_TUNING_DIR / HYBRID_CAMPAIGN

CORE_MIN_PER_100K = 7.0          # measured on the fair1 machine
KEEP_MB_PER_RUN = 3.0            # run dir after --slim (models, logs, probe, compact steps)
PEAK_MB_PER_WORKER = 30.0        # steps.csv exists until the run is scored

_RUNS_ENV = {"ppo": "PPO_RUNS_DIR", "td3": "TD3_RUNS_DIR", "ddpg": "DDPG_RUNS_DIR",
             "ddpg_hybrid": "DDPG_RUNS_DIR", "td3_hybrid": "TD3_RUNS_DIR",
             "ddpg_hybrid_qd": "DDPG_RUNS_DIR", "td3_hybrid_qd": "TD3_RUNS_DIR"}


# --------------------------------------------------------------------------- #
# protocols
# --------------------------------------------------------------------------- #
def _protocol(kind: str):
    from tuning.protocol import DEFAULT

    if kind == "fair1":
        return DEFAULT
    if kind == "smoke":
        return replace(DEFAULT, backend="analytic", total_timesteps=3000, eval_every=1000,
                       eval_episodes=1, objective_episodes=2)
    if kind == "fair2":
        # the fair1 protocol on scenario 2-1-1 (2 FBS, 1 MBS)
        return replace(DEFAULT, code="2-1-1")
    if kind == "fair2_300k":
        # Run D: 3x the budget, with a checkpoint at 100k so the same run can be
        # scored at both horizons. Everything else is the fair2 protocol.
        return replace(DEFAULT, code="2-1-1", total_timesteps=300_000,
                       checkpoint_every=100_000)
    if kind == "10k":
        return replace(DEFAULT, total_timesteps=10_000)
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# plans
# --------------------------------------------------------------------------- #
def _arms(name: str) -> tuple[List[dict], str, int, List[int]]:
    """(arms, protocol kind, probe_every, default seeds)."""
    if name in ("smoke", "s0_10k", "repro"):
        kind = {"smoke": "smoke", "s0_10k": "10k", "repro": "fair1"}[name]
        arms = [{"arm": "ddpg_t27", "algo": "ddpg", "base": "ddpg_t27", "overrides": {}},
                {"arm": "td3_t28", "algo": "td3", "base": "td3_t28", "overrides": {}}]
        return arms, kind, (500 if name == "smoke" else 1000), [0]

    if name == "e2":
        arms = []
        for base, algo, grid in (("ddpg_t27", "ddpg", (0.1, 0.3, 0.5, 0.8, 1.2)),
                                 ("td3_t28", "td3", (0.1, 0.3, 0.5, 1.2))):
            tuned = float(base_params(base)["params"]["binary_noise_std"])
            arms.append({"arm": f"{base}_bn_tuned", "algo": algo, "base": base,
                         "overrides": {}, "s_b": tuned, "anchor": True})
            for s in grid:
                arms.append({"arm": f"{base}_bn{s:g}", "algo": algo, "base": base,
                             "overrides": {"binary_noise_std": s}, "s_b": s})
        # TD3's tuned 0.807 replaces the grid's 0.8 (the anchor arm above)
        return arms, "fair1", 1000, list(range(20))

    if name == "e5":
        arms = []
        for d in (1, 2, 4):
            ov = {} if d == base_params("td3_t28")["params"]["policy_delay"] else {"policy_delay": d}
            ov_tau = {} if base_params("td3_t28")["params"]["tau"] == 0.02 else {"tau": 0.02}
            arms.append({"arm": f"td3_t28_pd{d}", "algo": "td3", "base": "td3_t28",
                         "overrides": {**ov, **ov_tau}, "policy_delay": d, "anchor": not ov})
        return arms, "fair1", 1000, list(range(35))

    if name == "e3_fail":
        # The failure-prone tuner-passed finalists, rerun with the probe on.
        # NO overrides: these are fair1 configs exactly as the tuner confirmed
        # them, so their fair1 per-seed record is the reproducibility oracle.
        # Chosen because they fail often (fair1 at 35 seeds: ddpg t55 4, ddpg
        # t19 5, td3 t45 13), which is what E2's winners would not do.
        arms = [{"arm": f"{base}_fail", "algo": algo, "base": base, "overrides": {},
                 "anchor": True, "trial": base.rsplit("_t", 1)[1]}
                for base, algo in (("ddpg_t55", "ddpg"), ("ddpg_t19", "ddpg"),
                                   ("td3_t45", "td3"))]
        return arms, "fair1", 1000, list(range(35))

    if name == "e3_fail2":
        # 2-FBS equivalent of e3_fail: failure-prone fair2 finalists, unchanged,
        # instrumented. TD3 t35 and t4 carry 3 and 2 switched-off failures; the
        # two DDPG arms were picked on failures plus one_on count from Step A
        # (t30: 7 failures / 18 one_on, t40: 5 / 20).
        arms = [{"arm": f"{base}_fail2", "algo": algo, "base": base, "overrides": {},
                 "anchor": True, "campaign": "fair2"}
                for base, algo in (("td3_t35", "td3"), ("td3_t4", "td3"),
                                   ("ddpg_t30", "ddpg"), ("ddpg_t40", "ddpg"))]
        return arms, "fair2", 1000, list(range(35))

    if name == "m5_long":
        # Review round 1, Run D: is TD3's both-on vs one-on crossing near 92k a
        # budget effect? Top-3 fair2 configs of DDPG and TD3, plus PPO t35 as the
        # reference, 300k steps, seeds 0-9, scored at 100k and at 300k.
        arms = []
        for algo, trials in (("ddpg", (15, 6, 40)), ("td3", (51, 56, 47)),
                             ("ppo", (35,))):
            for t in trials:
                arms.append({"arm": f"{algo}_t{t}_300k", "algo": algo,
                             "base": f"{algo}_t{t}", "overrides": {},
                             "anchor": True, "campaign": "fair2",
                             "log_positions": True})
        return arms, "fair2_300k", 1000, list(range(10))

    if name == "m2_onperiod":
        # Review round 1, Run C: does the trap end if a station is kept on long
        # enough to learn placement? Bases are the fair2 winners at their tuned
        # s_b; only the training-time on/off intervention differs. The baseline
        # is the e2_2fbs tuned anchor, already run at seeds 0-19.
        arms = []
        for base, algo in (("ddpg_t15", "ddpg"), ("td3_t51", "td3")):
            for L in (5, 10, 20):
                arms.append({"arm": f"{base}_minon{L}", "algo": algo, "base": base,
                             "campaign": "fair2", "overrides": {}, "min_on": L,
                             "log_positions": True})
            for N in (10_000, 20_000):
                arms.append({"arm": f"{base}_forceon{N // 1000}k", "algo": algo,
                             "base": base, "campaign": "fair2", "overrides": {},
                             "force_on_steps": N, "log_positions": True})
        return arms, "fair2", 1000, list(range(20))

    if name == "m1_hybrid2":
        # Review round 1, Run B: the four corners of (policy, update rule) at M = 2.
        # Bases are the fair2 winners; binary_noise_std is removed because the
        # Bernoulli sample IS the exploration, and gumbel_tau is 1.0 as in E4.
        # TRANSFER TEST, not tuned: these variants never went through fair2.
        arms = []
        for base, fam in (("ddpg_t15", "ddpg"), ("td3_t51", "td3")):
            for suffix in ("_hybrid", "_hybrid_qd"):
                algo = fam + suffix
                arms.append({"arm": f"{algo}_from_{base.split('_')[1]}", "algo": algo,
                             "base": base, "campaign": "fair2",
                             "overrides": {"binary_noise_std": None, "gumbel_tau": 1.0},
                             "hybrid_from_base": True, "log_positions": True})
        return arms, "fair2", 1000, list(range(20))

    if name == "m1_top3":
        # Review round 1, Run A: the top 3 fair2 configurations of each algorithm,
        # unchanged, instrumented with the probe, steps_compact and the opt-in
        # position log. Ranked on mean - sd, which is how fair2 picked finalists.
        arms = []
        for algo, trials in (("ppo", (35, 25, 45)), ("ddpg", (15, 6, 40)),
                             ("td3", (51, 56, 47))):
            for t in trials:
                arms.append({"arm": f"{algo}_t{t}_m1", "algo": algo,
                             "base": f"{algo}_t{t}", "overrides": {},
                             "anchor": True, "campaign": "fair2",
                             "log_positions": True})
        return arms, "fair2", 1000, list(range(20))

    if name == "e2_2fbs":
        # 2-FBS equivalent of e2: the fair2 winners (both Gaussian binary noise),
        # only s_b moves off the tuned point. Same grid as e2.
        arms = []
        # td3_t47 added 2026-10-16: t51 is the tuner winner on mean - sd but uses
        # both stations in only 3 of 35 runs, which leaves a noise sweep almost no
        # room to show an effect on abandonment. t47 leads on mean J (0.4740) and
        # on both_on (12 of 35) with 0 failures, so it is the better base for that
        # question. t51 is kept so the tuner-winner comparison is still available.
        for base, algo in (("ddpg_t15", "ddpg"), ("td3_t51", "td3"), ("td3_t47", "td3")):
            from .runner import FAIR2_DIR
            tuned = float(base_params(base, FAIR2_DIR)["params"]["binary_noise_std"])
            arms.append({"arm": f"{base}_bn_tuned", "algo": algo, "base": base, "overrides": {},
                         "s_b": tuned, "anchor": True, "campaign": "fair2"})
            # Grid extended 2026-10-15: 0.0 is the no-switch-noise control (the
            # continuous dims keep their own noise); 1.6 and 2.0 probe above the
            # original 1.2 ceiling, because the trap trend rises to that limit and
            # a trend measured against a boundary needs a probe before it is read
            # as a preference; 0.2 fills the low end where the escape formula is
            # least accurate (predicted rounds to 0, measured 0.0009 to 0.0048).
            for sb in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6, 2.0):
                arms.append({"arm": f"{base}_bn{sb:g}", "algo": algo, "base": base,
                             "overrides": {"binary_noise_std": sb}, "s_b": sb, "campaign": "fair2"})
        return arms, "fair2", 1000, list(range(20))

    if name == "e2_ppo":
        # PPO's tuner-passed winner, unchanged, under the fair1 protocol: a
        # check-up, not an ablation. Nothing moves off the tuned point, so the
        # only thing varying across the 20 runs is the seed.
        arms = [{"arm": "ppo_t43", "algo": "ppo", "base": "ppo_t43",
                 "overrides": {}, "anchor": True}]
        return arms, "fair1", 1000, list(range(20))

    if name in ("e4_pilot", "e4_pilot_qd"):
        variants = (("ddpg_t27", "ddpg_hybrid"), ("td3_t28", "td3_hybrid")) \
            if name == "e4_pilot" else \
            (("ddpg_t27", "ddpg_hybrid_qd"), ("td3_t28", "td3_hybrid_qd"))
        arms = []
        for base, algo in variants:
            arms.append({"arm": f"{algo}_from_{base.split('_')[1]}", "algo": algo, "base": base,
                         "overrides": {"binary_noise_std": None, "gumbel_tau": 1.0},
                         "hybrid_from_base": True})
        return arms, "fair1", 1000, [0, 1]

    raise SystemExit(f"unknown plan {name!r}")


def _campaign_dir(arm: dict):
    """Which tuner campaign an arm's base config is read from (default fair1)."""
    if arm.get("campaign") == "fair2":
        from .runner import FAIR2_DIR
        return FAIR2_DIR
    return None


def _params_for(arm: dict) -> dict:
    p = dict(base_params(arm["base"], _campaign_dir(arm))["params"])
    for k, v in arm["overrides"].items():
        if v is None:
            p.pop(k, None)
        else:
            p[k] = v
    return p


def build_specs(name: str, seeds: List[int] | None = None,
                protocol_override: str | None = None) -> List[dict]:
    """``protocol_override='smoke'`` is for testing the pipeline only (analytic
    backend, 3k steps); never use it for a reported result."""
    arms, kind, probe_every, default_seeds = _arms(name)
    if protocol_override:
        kind, probe_every = protocol_override, (500 if protocol_override == "smoke" else probe_every)
    seeds = default_seeds if seeds is None else seeds
    proto = _protocol(kind)
    pd = asdict(proto)
    exp_dir = experiment_dir(name)
    specs = []
    for seed in seeds:                          # seed-major: partial runs stay balanced
        for arm in arms:
            params = _params_for(arm)
            logpos = bool(arm.get("log_positions", False))
            key = spec_key(arm["algo"], params, seed, pd, True, probe_every,
                           log_positions=logpos,
                           min_on=arm.get("min_on"),
                           force_on_steps=arm.get("force_on_steps"))
            run_id = f"{arm['arm']}_s{seed}"
            specs.append({
                "experiment": name, "run_id": run_id, "arm": arm["arm"], "algo": arm["algo"],
                "base": arm["base"], "overrides": arm["overrides"], "params": params,
                "seed": seed, "tag": f"{arm['arm']}_s{seed}", "protocol": pd,
                "protocol_kind": kind, "instrument": True, "probe_every": probe_every,
                "slim": kind != "smoke", "key": key, "log_positions": logpos,
                "min_on": arm.get("min_on"), "force_on_steps": arm.get("force_on_steps"),
                "arm_meta": {k: v for k, v in arm.items() if k not in ("overrides",)},
                "result_path": str(exp_dir / "results" / f"{run_id}.json"),
            })
    return specs


# --------------------------------------------------------------------------- #
# reuse index
# --------------------------------------------------------------------------- #
def completed_index() -> Dict[str, Path]:
    out = {}
    for f in LETTER_RUNS_DIR.glob("*/results/*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if d.get("status") == "completed" and d.get("key") and not d.get("reused_from"):
            out.setdefault(d["key"], f)
    return out


def _child_env(name: str, algo: str) -> Dict[str, str]:
    env = dict(os.environ)
    runs = experiment_dir(name) / f"{algo}_runs"
    runs.mkdir(parents=True, exist_ok=True)
    assert_allowed(runs)
    env[_RUNS_ENV[algo]] = str(runs)
    if algo == "ppo":
        env["PPO_LEDGER_PATH"] = str(runs / "ledger.csv")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    env["MPLBACKEND"] = "Agg"
    env["LETTER_RUNS_DIR"] = str(LETTER_RUNS_DIR)
    return env


def estimate(specs: List[dict], workers: int) -> str:
    idx = completed_index()
    todo = [s for s in specs if not Path(s["result_path"]).is_file() and s["key"] not in idx]
    reuse = [s for s in specs if not Path(s["result_path"]).is_file() and s["key"] in idx]
    core_min = sum(CORE_MIN_PER_100K * s["protocol"]["total_timesteps"] / 100_000 for s in todo)
    wall_h = core_min / max(workers, 1) / 60
    disk_gb = len(todo) * KEEP_MB_PER_RUN / 1024
    peak_gb = disk_gb + workers * PEAK_MB_PER_WORKER / 1024
    arms = sorted({s["arm"] for s in specs})
    return (f"{len(specs)} runs in plan ({len(arms)} arms x {len({s['seed'] for s in specs})} seeds); "
            f"{len(specs) - len(todo) - len(reuse)} already done, {len(reuse)} reusable, "
            f"{len(todo)} to train\n"
            f"  est. {core_min:.0f} core-min = {wall_h:.1f} h wall at {workers} workers "
            f"(assumes {CORE_MIN_PER_100K} core-min per 100k steps)\n"
            f"  est. disk {disk_gb:.2f} GB kept, {peak_gb:.2f} GB peak "
            f"({KEEP_MB_PER_RUN} MB/run kept, {PEAK_MB_PER_WORKER} MB/worker transient)")


def run(name: str, seeds: List[int] | None, workers: int, poll: float = 2.0,
        protocol_override: str | None = None) -> None:
    specs = build_specs(name, seeds, protocol_override)
    exp_dir = experiment_dir(name)
    for sub in ("specs", "results", "logs"):
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)
    print(estimate(specs, workers), flush=True)

    idx = completed_index()
    pending = []
    for s in specs:
        # every spec is on disk, including reused ones, so analysis can read arm metadata
        (exp_dir / "specs" / f"{s['run_id']}.json").write_text(json.dumps(s, indent=2, default=str))
        rp = Path(s["result_path"])
        if rp.is_file() and json.loads(rp.read_text()).get("status") == "completed":
            continue
        if s["key"] in idx:
            src = json.loads(idx[s["key"]].read_text())
            rp.write_text(json.dumps({**src, "experiment": name, "run_id": s["run_id"],
                                      "arm": s["arm"], "reused_from": str(idx[s["key"]])},
                                     indent=2))
            print(f"  reuse {s['run_id']} <- {idx[s['key']]}", flush=True)
            continue
        pending.append(s)

    jobs = []
    t_start = time.time()
    n_total = len(pending)
    done = 0
    while pending or jobs:
        while pending and len(jobs) < _workers_now(name, pending[0]["algo"], workers):
            s = pending.pop(0)
            sp = exp_dir / "specs" / f"{s['run_id']}.json"
            sp.write_text(json.dumps(s, indent=2, default=str))
            log = (exp_dir / "logs" / f"{s['run_id']}.log").open("w")
            pop = subprocess.Popen([sys.executable, "-m", "letterexp.runner", "--spec", str(sp)],
                                   cwd=str(REPO_ROOT), env=_child_env(name, s["algo"]),
                                   stdout=log, stderr=subprocess.STDOUT)
            jobs.append((s, pop, log, time.time()))
            print(f"-> {s['run_id']} ({len(jobs)} running, {len(pending)} queued)", flush=True)
        time.sleep(poll)
        for job in list(jobs):
            s, pop, log, t0 = job
            if pop.poll() is None:
                continue
            jobs.remove(job)
            log.close()
            done += 1
            rp = Path(s["result_path"])
            r = json.loads(rp.read_text()) if rp.is_file() else {"status": "failed"}
            mark = (f"J_best={r['J_best']:.4f} J_final={r['J_final']:.4f}"
                    f"{' FAIL' if r.get('fail_best') else ''}"
                    if r.get("status") == "completed" else f"FAILED: {r.get('error', 'no result')}")
            el = (time.time() - t_start) / 60
            print(f"<- {s['run_id']} {mark} [{(time.time() - t0) / 60:.1f} min; "
                  f"{done}/{n_total} done, {el:.0f} min elapsed]", flush=True)


def _workers_now(name: str, algo: str, workers: int) -> int:
    """1 until the experiment's ledger for ``algo`` exists (same race as tuning.study)."""
    runs = experiment_dir(name) / f"{algo}_runs"
    return workers if (runs / "ledger.csv").is_file() else 1


def status(name: str) -> None:
    import pandas as pd

    rows = []
    for f in sorted((experiment_dir(name) / "results").glob("*.json")):
        d = json.loads(f.read_text())
        rows.append({k: d.get(k) for k in ("run_id", "arm", "seed", "status", "J_best",
                                            "J_final", "fail_best", "train_seconds")})
    if not rows:
        print("no results yet")
        return
    df = pd.DataFrame(rows)
    print(df.groupby("arm").agg(n=("seed", "size"), ok=("status", lambda s: (s == "completed").sum()),
                                J_best=("J_best", "mean"), fails=("fail_best", "sum"),
                                train_min=("train_seconds", lambda s: s.mean() / 60)).to_string())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m letterexp.launch")
    ap.add_argument("cmd", choices=["plan", "run", "status"])
    ap.add_argument("plan_name")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--protocol", default=None, choices=["smoke", "10k", "fair1"],
                    help="override the plan's protocol (testing only; reported runs use the default)")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        specs = build_specs(a.plan_name, a.seeds, a.protocol)
        print(estimate(specs, a.workers))
        for arm in sorted({s["arm"] for s in specs}):
            s = next(x for x in specs if x["arm"] == arm)
            print(f"  {arm:28s} {s['algo']:12s} overrides={s['overrides']} params={s['params']}")
    elif a.cmd == "run":
        run(a.plan_name, a.seeds, a.workers, protocol_override=a.protocol)
    else:
        status(a.plan_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
