"""One trial, one process: train a config then score it.

    python -m tuning.worker --spec tuning_runs/<study>/specs/trial_0007.json

Run as a subprocess rather than in the orchestrator for four reasons, all of
which showed up in the 56-run DDPG campaign:

- **Artifact routing.** ``ppo.paths`` and its siblings read their run
  directories at *import* time. A fresh process is the only place
  ``DDPG_RUNS_DIR`` can be redirected at the study, which is what keeps a
  180-run campaign out of ``ddpg_runs/`` and ``training_log.csv``.
- **Thread control.** ``OMP_NUM_THREADS=1`` must be set before torch is
  imported to have any effect, and oversubscribed BLAS threads are what
  makes N concurrent trainings slower than N sequential ones.
- **Crash isolation.** A trial that dies takes its own process down; the
  orchestrator records the failure and carries on.
- **Real parallelism.** No GIL between trials.

The environment for all of that is set by the parent (see
``tuning.study.trial_env``); this module assumes it is already correct.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from .protocol import Protocol


def protocol_from_dict(data: dict) -> Protocol:
    """Rebuild a Protocol from its JSON form (tuples arrive as lists)."""
    data = dict(data)
    for key in ("confirm_seeds",):
        if key in data:
            data[key] = tuple(data[key])
    return Protocol(**data)


def run_spec(spec: dict) -> dict:
    """Train + score one config. Returns the result payload."""
    from .objective import score_run
    from .spaces import build_experiment

    algo = spec["algo"]
    protocol = protocol_from_dict(spec["protocol"])
    seed = int(spec["seed"])
    tag = spec["tag"]

    exp = build_experiment(algo, spec["params"], protocol, seed, tag)

    kwargs = {}
    if algo == "ppo":
        from ppo.train import train
    elif algo == "td3":
        from td3.train import train
    elif algo == "ddpg":
        from ddpg.train import train
    elif algo == "ddpg_hybrid":
        from ddpg_hybrid.train import train
        kwargs["gumbel_tau"] = float(spec["params"]["gumbel_tau"])
    elif algo == "td3_hybrid":
        from td3_hybrid.train import train
    elif algo == "ddpg_hybrid_qd":
        from ddpg_hybrid.train import train_qd as train
    elif algo == "td3_hybrid_qd":
        from td3_hybrid.train import train_qd as train
        kwargs["gumbel_tau"] = float(spec["params"]["gumbel_tau"])
    else:
        raise ValueError(f"unknown algo {algo!r}")

    # Opt-in (letter Stage 4): LETTER_INSTRUMENT=1 adds the probe and the
    # compact step log to an off-policy trial. Unset, nothing changes.
    if os.environ.get("LETTER_INSTRUMENT") == "1" and algo != "ppo":
        from letterexp.runner import instrument_callbacks
        kwargs["extra_callbacks"] = instrument_callbacks(exp, protocol)

    t0 = time.time()
    run_dir = train(exp, backend=protocol.backend, **kwargs)
    train_seconds = time.time() - t0

    t0 = time.time()
    score = score_run(algo, run_dir, protocol)
    score_seconds = time.time() - t0

    freed = 0
    if not protocol.keep_step_log:
        # Only after scoring: score_run needs the model, never the step log.
        f = Path(run_dir) / "steps.csv"
        if f.is_file():
            freed = f.stat().st_size
            f.unlink()

    return {
        "status": "completed",
        "algo": algo,
        "trial": spec["trial"],
        "seed": seed,
        "tag": tag,
        "params": spec["params"],
        "run_dir": str(run_dir),
        "train_seconds": round(train_seconds, 2),
        "freed_bytes": freed,
        "score_seconds": round(score_seconds, 2),
        **score,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m tuning.worker", description=__doc__)
    p.add_argument("--spec", required=True, help="path to the trial spec JSON")
    args = p.parse_args(argv)

    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text())
    out = Path(spec["result_path"])

    try:
        result = run_spec(spec)
        rc = 0
    except BaseException as exc:                      # noqa: BLE001 - reported, not swallowed
        result = {
            "status": "failed",
            "algo": spec.get("algo"),
            "trial": spec.get("trial"),
            "seed": spec.get("seed"),
            "params": spec.get("params"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        rc = 1

    out.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "traceback"}, default=str))
    return rc


if __name__ == "__main__":
    sys.exit(main())
