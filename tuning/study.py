"""The orchestrator: ask a GP for configs, run them as subprocesses, tell it back.

One optuna study per algorithm, all inside one campaign directory, so "equal
budget per algorithm" is structural rather than a promise.

Why ask/tell in a single parent instead of N workers sharing a storage: the
GP should see every trial, including the ones still running. Optuna 5's
``GPSampler`` uses a q-batch log expected improvement (qLogEI) with
sequential greedy selection when trials are in flight, which is exactly the
batch case, and it can only do that if one sampler instance is asked for all
of them. Workers sharing storage would each build their own model.

No pruner is installed, and that is a measured decision rather than a
default. Across the 54 DDPG runs that reached 100k, Spearman rank
correlation between J at 25k and J at 100k is +0.26 (+0.53 at 50k), and the
best run of the campaign ranked 29th of 54 at 25k. Successive halving at any
early rung discards the winner.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import optuna

from ppo.paths import REPO_ROOT, ensure_dir

from . import paths
from .protocol import DEFAULT, Protocol
from .spaces import seed_trials, space

# Each algorithm's runs, and its ledger, are redirected into the campaign
# directory by these variables. The ledger path key differs for ppo because
# its ledger is the repo-root training_log.csv rather than a file inside the
# runs directory.
_RUNS_ENV: Dict[str, str] = {"ppo": "PPO_RUNS_DIR", "td3": "TD3_RUNS_DIR",
                             "ddpg": "DDPG_RUNS_DIR",
                             # hybrids train through ddpg/ and td3/ loggers
                             "ddpg_hybrid": "DDPG_RUNS_DIR", "td3_hybrid": "TD3_RUNS_DIR",
                             "ddpg_hybrid_qd": "DDPG_RUNS_DIR",
                             "td3_hybrid_qd": "TD3_RUNS_DIR"}


def trial_env(campaign: str, algo: str) -> Dict[str, str]:
    """Environment for a trial subprocess: artifact routing + thread limits."""
    runs = ensure_dir(paths.study_dir(campaign) / f"{algo}_runs")
    env = dict(os.environ)
    env[_RUNS_ENV[algo]] = str(runs)
    if algo == "ppo":
        env["PPO_LEDGER_PATH"] = str(runs / "ledger.csv")
    # Must be set before torch is imported by the child, which is why this
    # is an env dict for a subprocess and not a runtime call.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    env["MPLBACKEND"] = "Agg"
    return env


def ledger_path(campaign: str, algo: str) -> Path:
    """Where this campaign's ledger for ``algo`` lands (see ``trial_env``)."""
    return paths.study_dir(campaign) / f"{algo}_runs" / "ledger.csv"


def _effective_workers(campaign: str, algo: str, workers: int) -> int:
    """1 until the campaign ledger exists, then ``workers``.

    ``ppo.run_logging.append_ledger_row`` reads the ledger and then writes it,
    with no lock. Two trials finishing at the same instant against a *missing*
    ledger both see no header and both write one, leaving a header row in the
    middle of the file. Letting the first trial of a campaign finish alone
    costs one trial's worth of parallelism out of sixty and removes the race:
    every later append sees a complete header and takes the plain-append
    branch, whose key set is fixed for a given algorithm and protocol.
    """
    return workers if ledger_path(campaign, algo).is_file() else 1


def make_study(campaign: str, algo: str, protocol: Protocol,
               sampler_seed: int = 0) -> optuna.Study:
    sampler = optuna.samplers.GPSampler(
        seed=sampler_seed,
        n_startup_trials=protocol.n_startup_trials,
        # False: the objective is a 20-episode mean of a stochastic training
        # process. Letting the GP fit a noise variance is the whole reason
        # this sampler suits a noisy objective.
        deterministic_objective=False,
        # A warning here means a parameter was sampled independently, i.e.
        # the flat-space invariant in tuning.spaces broke. Worth seeing.
        warn_independent_sampling=True,
    )
    return optuna.create_study(
        study_name=algo,
        storage=paths.storage_url(campaign, algo),
        sampler=sampler,
        # Explicit, because optuna's default is MedianPruner. It would be
        # inert here (ask/tell never calls trial.report), but a pruner on
        # this study would be wrong if anyone later added intermediate
        # reporting -- see the rank-correlation note in the module docstring.
        pruner=optuna.pruners.NopPruner(),
        direction="maximize",
        load_if_exists=True,
    )


class _Job:
    def __init__(self, trial, popen, spec, log):
        self.trial, self.popen, self.spec, self.log = trial, popen, spec, log
        self.started = time.time()


def _launch(campaign: str, algo: str, protocol: Protocol, trial_number: int,
            params: dict, seed: int, tag: str) -> tuple:
    spec = {
        "study": campaign, "algo": algo, "trial": trial_number,
        "params": params, "seed": seed, "tag": tag,
        "protocol": asdict(protocol),
        "result_path": str(paths.result_path(campaign, algo, trial_number)),
    }
    sp = paths.spec_path(campaign, algo, trial_number)
    sp.write_text(json.dumps(spec, indent=2, default=str))
    logf = paths.trial_log_path(campaign, algo, trial_number)
    handle = logf.open("w")
    popen = subprocess.Popen(
        [sys.executable, "-m", "tuning.worker", "--spec", str(sp)],
        cwd=str(REPO_ROOT), env=trial_env(campaign, algo),
        stdout=handle, stderr=subprocess.STDOUT,
    )
    return spec, popen, logf, handle


def search(campaign: str, algo: str, protocol: Protocol = DEFAULT, *,
           workers: int = 4, sampler_seed: int = 0, wide: bool = False,
           poll_seconds: float = 2.0, quiet: bool = False) -> optuna.Study:
    """Stage 1: ``protocol.n_trials`` trials at ``protocol.search_seed``.

    ``wide`` selects the widened boundary-probe space for this algorithm
    (see ``tuning.spaces.SPECIFIC_WIDE``). It is not part of the
    equal-budget comparison and belongs in its own campaign.
    """
    study = make_study(campaign, algo, protocol, sampler_seed)
    knobs = space(algo, wide)

    # Recommendation 3 of the tuning-methods page: start from the best known
    # point. Only on a fresh study -- re-enqueueing on a resume would just
    # repeat work already in the database.
    if not study.trials:
        for seeded in seed_trials(algo, wide):
            study.enqueue_trial(seeded, skip_if_exists=True)

    done = len([t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE])
    issued = done
    jobs: List[_Job] = []
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    say(f"[{algo}] campaign {campaign}: {protocol.n_trials} trials, "
        f"{workers} workers, {protocol.total_timesteps} steps/trial, "
        f"already complete: {done}")

    while issued < protocol.n_trials or jobs:
        while (len(jobs) < _effective_workers(campaign, algo, workers)
               and issued < protocol.n_trials):
            trial = study.ask()
            params = {k.name: k.suggest(trial) for k in knobs}
            tag = f"t{trial.number:04d}_s{protocol.search_seed}"
            spec, popen, logf, handle = _launch(
                campaign, algo, protocol, trial.number, params,
                protocol.search_seed, tag)
            job = _Job(trial, popen, spec, logf)
            job._handle = handle
            jobs.append(job)
            issued += 1
            say(f"[{algo}] -> trial {trial.number:3d} started "
                f"({len(jobs)} running, {issued}/{protocol.n_trials} issued)")

        time.sleep(poll_seconds)

        for job in list(jobs):
            if job.popen.poll() is None:
                continue
            jobs.remove(job)
            job._handle.close()
            result = _read_result(Path(job.spec["result_path"]), job.popen.returncode,
                                  job.log)
            _tell(study, job.trial, result)
            done += 1
            mark = (f"J={result['objective']:.4f}" if result.get("status") == "completed"
                    else f"FAILED ({result.get('error', 'no result')})")
            say(f"[{algo}] <- trial {job.trial.number:3d} {mark}  "
                f"[{time.time() - job.started:.0f}s, {done}/{protocol.n_trials} done]")

    _dump_trials(campaign, algo, study)
    best = study.best_trial
    say(f"[{algo}] best trial {best.number}: J={best.value:.4f}  {best.params}")
    return study


def _read_result(path: Path, returncode: int, log: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {"status": "failed",
            "error": f"worker exited {returncode} without a result; see {log}"}


def _tell(study: optuna.Study, trial, result: dict) -> None:
    if result.get("status") != "completed":
        trial.set_user_attr("error", result.get("error", "unknown"))
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
        return
    for key in ("run_dir", "gap_closed_pct", "objective_sd", "best_J",
                "connected", "fbs_connected", "train_seconds", "score_seconds",
                "model", "seed"):
        if key in result:
            trial.set_user_attr(key, result[key])
    study.tell(trial, result["objective"])


def _dump_trials(campaign: str, algo: str, study: optuna.Study) -> None:
    """A flat CSV of the study, next to the database."""
    import pandas as pd

    rows = []
    for t in study.trials:
        rows.append({
            "trial": t.number,
            "state": t.state.name,
            "objective": t.value,
            **{f"p_{k}": v for k, v in t.params.items()},
            **{k: v for k, v in t.user_attrs.items()},
        })
    if rows:
        pd.DataFrame(rows).to_csv(
            paths.study_dir(campaign) / f"{algo}_trials.csv", index=False)


# --------------------------------------------------------------------------- #
# Stage 2: confirmation across seeds
# --------------------------------------------------------------------------- #
def existing_results(campaign: str, algo: str) -> Dict[tuple, dict]:
    """Every (trial, seed) this campaign has already scored, from its specs.

    Makes ``confirm`` idempotent: re-running it with a wider seed list trains
    only the seeds that are missing, so extending a 5-seed confirmation to 35
    costs 30 runs per finalist rather than 35. It also makes a long
    confirmation resumable after an interruption.
    """
    out: Dict[tuple, dict] = {}
    specs = paths.study_dir(campaign) / "specs"
    for f in sorted(specs.glob(f"{algo}_*.result.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if d.get("status") != "completed":
            continue
        # A confirmation spec's "trial" field is an internal counter, not the
        # finalist's trial number -- that is carried in the tag, which is
        # "c<trial>_s<seed>" for confirmations and "t<trial>_s<seed>" for
        # stage 1. Parse the tag so both kinds index the same way.
        m = re.fullmatch(r"[tc](\d+)_s(\d+)", str(d.get("tag", "")))
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = d
        elif d.get("trial") is not None and d.get("seed") is not None:
            out[(int(d["trial"]), int(d["seed"]))] = d
    return out


def confirm(campaign: str, algo: str, protocol: Protocol = DEFAULT, *,
            workers: int = 4, poll_seconds: float = 2.0,
            quiet: bool = False) -> dict:
    """Re-run the top configs across ``protocol.confirm_seeds``.

    Ranked on mean - sd, so a config that is merely lucky on one seed cannot
    win. The stage-1 run at ``search_seed`` is reused rather than repeated:
    training is seeded and reproducible, so re-running it would produce the
    same number at full cost.
    """
    study = make_study(campaign, algo, protocol)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise SystemExit(f"no completed trials for {algo} in campaign {campaign}")
    finalists = sorted(complete, key=lambda t: t.value, reverse=True)[:protocol.confirm_top_k]
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    say(f"[{algo}] confirming {len(finalists)} finalists over "
        f"{len(protocol.confirm_seeds)} seeds "
        f"({min(protocol.confirm_seeds)}-{max(protocol.confirm_seeds)})")

    have = existing_results(campaign, algo)
    results: Dict[int, Dict[int, dict]] = {t.number: {} for t in finalists}
    pending = []
    reused = 0
    for t in finalists:
        for seed in protocol.confirm_seeds:
            prior = have.get((t.number, seed))
            if seed == protocol.search_seed and prior is None:
                # the stage-1 run, whose result is on the trial itself
                prior = {"status": "completed", "objective": t.value,
                         "run_dir": t.user_attrs.get("run_dir")}
            if prior is not None:
                results[t.number][seed] = {**prior, "reused": True}
                reused += 1
            else:
                pending.append((t, seed))
    say_reuse = f"{reused} seed(s) already on disk, {len(pending)} to train"

    say(f"[{algo}] {say_reuse}")

    jobs: List[_Job] = []
    # Confirmation spec ids must not collide with stage-1 or with specs an
    # earlier confirmation already wrote.
    counter = max([10_000] + [
        int(f.stem.split("_")[-1]) for f in
        (paths.study_dir(campaign) / "specs").glob(f"{algo}_1*.json")
        if f.stem.split("_")[-1].isdigit()])
    while pending or jobs:
        while (len(jobs) < _effective_workers(campaign, algo, workers)
               and pending):
            t, seed = pending.pop(0)
            counter += 1
            tag = f"c{t.number:04d}_s{seed}"
            spec, popen, logf, handle = _launch(
                campaign, algo, protocol, counter, t.params, seed, tag)
            job = _Job(t, popen, spec, logf)
            job._handle, job._seed = handle, seed
            jobs.append(job)
            say(f"[{algo}] -> confirm trial {t.number} seed {seed}")

        time.sleep(poll_seconds)

        for job in list(jobs):
            if job.popen.poll() is None:
                continue
            jobs.remove(job)
            job._handle.close()
            res = _read_result(Path(job.spec["result_path"]), job.popen.returncode, job.log)
            results[job.trial.number][job._seed] = res
            mark = (f"J={res['objective']:.4f}" if res.get("status") == "completed"
                    else "FAILED")
            say(f"[{algo}] <- confirm trial {job.trial.number} seed {job._seed} {mark}")

    summary = _summarise_confirmation(algo, finalists, results, protocol)
    out = paths.study_dir(campaign) / f"{algo}_confirmation.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    say(f"[{algo}] confirmation -> {out}")
    return summary


def _summarise_confirmation(algo, finalists, results, protocol) -> dict:
    import numpy as np

    rows = []
    for t in finalists:
        per_seed = results[t.number]
        vals = [r["objective"] for r in per_seed.values() if r.get("status") == "completed"]
        mean = float(np.mean(vals)) if vals else float("nan")
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append({
            "trial": t.number,
            "params": t.params,
            "stage1_objective": t.value,
            "seeds": {str(s): r.get("objective") for s, r in sorted(per_seed.items())},
            "n_ok": len(vals),
            "mean": mean,
            "sd": sd,
            "score": mean - sd,          # reliability inside the objective
            "run_dirs": {str(s): r.get("run_dir") for s, r in sorted(per_seed.items())},
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return {
        "algo": algo,
        "seeds": list(protocol.confirm_seeds),
        "ranking_rule": "mean - sd over confirm_seeds",
        "winner": rows[0] if rows else None,
        "finalists": rows,
    }
