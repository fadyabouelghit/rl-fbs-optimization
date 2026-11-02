"""One letter run, one process: build -> train (instrumented) -> score best and final.

    python -m letterexp.runner --spec letter_runs/e2/specs/<run_id>.json

Configs are never typed in by hand. ``base_params`` reads each tuned config
from the tuner's own confirmation file (``tuning_runs/fair1/*_confirmation.json``,
read-only), and an experiment may only change named knobs on top of it via
``overrides`` (an ablation). The spec records both, so every result says
exactly how far it is from a tuner-passed config.

Scoring is ``tuning.objective.score_run``, i.e. the fair1 objective: mean
final J over the 20 common start states (seed 987654), on best_model.zip and,
separately, on model.zip.

Like ``tuning.worker`` this runs as a subprocess so that the *_RUNS_DIR
redirection (set by ``letterexp.launch``) is in force before ppo/ddpg/td3 are
imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

from ppo.paths import REPO_ROOT

FAIR1_DIR = REPO_ROOT / "tuning_runs" / "fair1"          # read-only
FAIR2_DIR = REPO_ROOT / "letter_runs_2fbs" / "tuning" / "fair2"   # 2 FBS, read-only
FAIL_TOL = 0.0005                                         # failure: J <= baseline + tol
BASELINE_J = 0.4165
OPTIMUM_J = 0.4925

ALGO_OF = {"ddpg": "ddpg", "td3": "td3", "ppo": "ppo",
           "ddpg_hybrid": "ddpg", "td3_hybrid": "td3",
           "ddpg_hybrid_qd": "ddpg", "td3_hybrid_qd": "td3"}


# --------------------------------------------------------------------------- #
# tuned configs, read from the tuner's files
# --------------------------------------------------------------------------- #
def _confirmation_file(algo: str, campaign_dir: Path) -> Path:
    return campaign_dir / f"{algo}_confirmation.json"


def base_params(name: str, campaign_dir: Path | None = None) -> Dict[str, Any]:
    """``'ddpg_t27'`` -> the finalist's params + its fair1 per-seed record.

    ``'<algo>_winner'`` returns the campaign winner. Hybrid names
    (``ddpg_hybrid_t12``, ``ddpg_hybrid_winner``) are read from the letter's
    own hybrid tuning campaign (see ``letterexp.launch HYBRID_CAMPAIGN_DIR``).
    """
    algo, _, which = name.rpartition("_")
    if campaign_dir is None:
        if algo.endswith("_hybrid"):
            from .launch import HYBRID_CAMPAIGN_DIR
            campaign_dir = HYBRID_CAMPAIGN_DIR
        else:
            campaign_dir = FAIR1_DIR
    data = json.loads(_confirmation_file(algo, campaign_dir).read_text())
    if which == "winner":
        row = data["winner"]
    else:
        trial = int(which.lstrip("t"))
        rows = [r for r in data["finalists"] if int(r["trial"]) == trial]
        if not rows:
            raise KeyError(f"trial {trial} is not a confirmed finalist of {algo} in {campaign_dir}")
        row = rows[0]
    return {"algo": algo, "trial": int(row["trial"]), "params": dict(row["params"]),
            "seeds": row.get("seeds", {}), "run_dirs": row.get("run_dirs", {}),
            "n_ok": row.get("n_ok"), "mean": row.get("mean"), "sd": row.get("sd"),
            "source": str(_confirmation_file(algo, campaign_dir))}


def spec_key(algo: str, params: dict, seed: int, protocol: dict, instrument: bool,
             probe_every: int, log_positions: bool = False,
             min_on=None, force_on_steps=None) -> str:
    """Identity of a run: identical keys are the same training run.

    ``log_positions`` is part of the key: a run logged without positions is not
    a substitute for one logged with them, so reuse must not match across it.
    Omitting it keeps every pre-existing key unchanged (default False).
    """
    payload = {"algo": algo, "params": params, "seed": int(seed),
               "protocol": protocol, "instrument": bool(instrument),
               "probe_every": int(probe_every)}
    if log_positions:
        payload["log_positions"] = True
    if min_on:
        payload["min_on"] = int(min_on)
    if force_on_steps:
        payload["force_on_steps"] = int(force_on_steps)
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def apply_band(exp, band: str):
    """Stage 6b hook (band mode multi / mbs_capacity agent). Scoped, not built."""
    raise NotImplementedError(
        "Stage 6b is scoped at CHECKPOINT 6 and not implemented yet; see HANDOVER_letter.md")


# --------------------------------------------------------------------------- #
# instrumentation
# --------------------------------------------------------------------------- #
def instrument_callbacks(exp, protocol, probe_every: int = 1000, algo: str = "ddpg",
                         positions: bool = False) -> list:
    from .compact_steps import CompactStepCallback
    from .probe import (PPOProbeCallback, ProbeCallback, binary_dims_and_labels,
                        probe_set_path)

    idx, labels = binary_dims_and_labels(exp.env)
    probe = probe_set_path(protocol.code, protocol.backend)
    if not probe.is_file():
        raise FileNotFoundError(
            f"probe set {probe} missing; run: python -m letterexp.probe build "
            f"--code {protocol.code} --backend {protocol.backend}")
    cb = PPOProbeCallback if algo == "ppo" else ProbeCallback
    kw = {}
    if positions:
        from ppo.env import FlyingBaseStationEnv
        e = FlyingBaseStationEnv(exp.env, backend_kind="analytic")   # bounds only
        try:
            kw = {"positions": True, "state_low": e.state_low, "state_high": e.state_high,
                  "n_fbs": exp.env.num_fbs, "genes_per_fbs": e.genes_per_fbs}
        finally:
            e.close()
    return [
        cb(probe, idx, labels, every=probe_every),
        CompactStepCallback(None, idx, labels,
                            obs_normalized=bool(exp.env.normalize_obs), **kw),
    ]


@contextmanager
def ppo_extra_callbacks(extra: list):
    """Attach callbacks to a ``ppo.train`` run without editing ``ppo/``.

    ``ddpg/`` and ``td3/`` grew a keyword-only ``extra_callbacks`` hook for the
    letter; ``ppo/`` is off limits, and ``ppo.train.train`` passes its own list
    straight to ``model.learn(callback=...)``. So wrap ``PPO.learn`` for the
    duration of the call and append there. Each letter run is its own
    subprocess (``letterexp.launch``), so the patch cannot leak into anything
    else, and it is undone in a finally block regardless.
    """
    from stable_baselines3 import PPO

    original = PPO.learn

    def learn(self, *args, callback=None, **kwargs):
        if callback is None:
            base = []
        elif isinstance(callback, list):
            base = list(callback)
        else:
            base = [callback]
        return original(self, *args, callback=base + list(extra), **kwargs)

    PPO.learn = learn
    try:
        yield
    finally:
        PPO.learn = original


def _train_fn(algo: str):
    if algo == "ddpg":
        from ddpg.train import train
    elif algo == "td3":
        from td3.train import train
    elif algo == "ddpg_hybrid":
        from ddpg_hybrid.train import train
    elif algo == "td3_hybrid":
        from td3_hybrid.train import train
    elif algo == "ddpg_hybrid_qd":
        from ddpg_hybrid.train import train_qd as train
    elif algo == "td3_hybrid_qd":
        from td3_hybrid.train import train_qd as train
    elif algo == "ppo":
        from ppo.train import train
    else:
        raise ValueError(f"letter runner does not train {algo!r}")
    return train


def run_spec(spec: dict) -> dict:
    from tuning.objective import score_run
    from tuning.spaces import build_experiment
    from tuning.worker import protocol_from_dict

    algo = spec["algo"]
    protocol = protocol_from_dict(spec["protocol"])
    seed = int(spec["seed"])
    params = dict(spec["params"])
    exp = build_experiment(algo, params, protocol, seed, spec["tag"])

    cbs = instrument_callbacks(exp, protocol, int(spec.get("probe_every", 1000)), algo,
                               positions=bool(spec.get("log_positions", False))) \
        if spec.get("instrument", True) else []
    train = _train_fn(algo)
    letter_meta = {k: spec[k] for k in
                   ("experiment", "run_id", "base", "overrides", "key") if k in spec}

    # Run C: training-only on/off intervention, off unless the spec asks for it
    ov = None
    if spec.get("min_on") or spec.get("force_on_steps"):
        from .onoff_override import OnOffOverride, onoff_override
        from ppo.env import state_labels
        labels = state_labels(exp.env)
        bidx = [i for i, n in enumerate(labels) if n.endswith("_power_status")]
        ov = OnOffOverride(bidx, min_on=spec.get("min_on"),
                           force_on_steps=spec.get("force_on_steps"),
                           obs_normalized=bool(exp.env.normalize_obs))

    t0 = time.time()
    if algo == "ppo":
        # ppo/ has no extra_callbacks or config_extra hook and must not be edited.
        with ppo_extra_callbacks(cbs):
            run_dir = Path(train(exp, backend=protocol.backend))
        (run_dir / "letter_meta.json").write_text(json.dumps(letter_meta, indent=2))
    else:
        kwargs = dict(extra_callbacks=cbs, config_extra={"letter": letter_meta})
        if algo.endswith("_hybrid"):
            kwargs["gumbel_tau"] = float(params["gumbel_tau"])
        if ov is not None:
            from stable_baselines3 import DDPG, TD3
            from .onoff_override import onoff_override
            cls = TD3 if ALGO_OF[algo] == "td3" else DDPG
            with onoff_override(cls, ov):
                run_dir = Path(train(exp, backend=protocol.backend, **kwargs))
        else:
            run_dir = Path(train(exp, backend=protocol.backend, **kwargs))
    train_seconds = time.time() - t0

    t0 = time.time()
    best = score_run(ALGO_OF[algo], run_dir, protocol, model="best")
    final = score_run(ALGO_OF[algo], run_dir, protocol, model="final")
    score_seconds = time.time() - t0

    freed = 0
    if spec.get("slim", True):
        f = run_dir / "steps.csv"
        if f.is_file():
            freed = f.stat().st_size
            f.unlink()

    def _fail(j, base):
        return bool(j <= base + FAIL_TOL)

    return {
        "status": "completed", "experiment": spec["experiment"], "run_id": spec["run_id"],
        "key": spec["key"], "algo": algo, "base": spec.get("base"), "arm": spec.get("arm"),
        "overrides": spec.get("overrides", {}), "params": params, "seed": seed,
        "run_dir": str(run_dir), "train_seconds": round(train_seconds, 1),
        "score_seconds": round(score_seconds, 1), "freed_bytes": freed,
        "J_best": best["objective"], "J_best_sd": best["objective_sd"],
        "J_best_model_used": best["model"],
        "J_final": final["objective"], "J_final_sd": final["objective_sd"],
        "baseline": best["baseline"], "optimum": best["optimum"],
        "gap_closed_best": (best["objective"] - best["baseline"]) / (best["optimum"] - best["baseline"]),
        "gap_closed_final": (final["objective"] - final["baseline"]) / (final["optimum"] - final["baseline"]),
        "fail_best": _fail(best["objective"], best["baseline"]),
        "fail_final": _fail(final["objective"], final["baseline"]),
        "fbs_connected_best": best["fbs_connected"], "connected_best": best["connected"],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m letterexp.runner")
    p.add_argument("--spec", required=True)
    args = p.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    out = Path(spec["result_path"])
    try:
        result = run_spec(spec)
        rc = 0
    except BaseException as exc:                      # noqa: BLE001
        result = {"status": "failed", "run_id": spec.get("run_id"), "key": spec.get("key"),
                  "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        rc = 1
    out.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "traceback"}, default=str))
    return rc


if __name__ == "__main__":
    sys.exit(main())
