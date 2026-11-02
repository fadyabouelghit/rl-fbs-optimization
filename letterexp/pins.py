"""Pin events from probe_log.csv, and flips from steps_compact.csv.gz.

Definitions (fixed before any Stage 2 data, see RESULTS_letter_E2.txt):

  pinned at a checkpoint   >= 95% of probe states have |mu_b| > 0.96
                           (column ``<label>_pinned`` written by ProbeCallback)
  direction                'off' if >= 95% of the saturated probe states sit at -1,
                           'on' if >= 95% sit at +1, else 'mixed' (state-dependent
                           saturation, typically "keep the current value")
  trapped at a checkpoint  >= 95% of probe states whose gene is OFF have mu_b < -0.96
  trap event               >= MIN_RUN consecutive trapped checkpoints (the
                           failure-relevant pin: from an off gene the actor's output
                           is -1). Escapes, stuck-at-end and flip counts are reported
                           for trap events (primary) and for 'off' pin events
                           (secondary).
  pin event                >= MIN_RUN consecutive pinned checkpoints with the same
                           direction; onset = first of them

  MIN_RUN is 5 (>= 5000 steps at the standard probe_every=1000) for the PRIMARY
  analysis and 3 (>= 3000 steps) for the declared SECONDARY one. Both thresholds
  were fixed in the E2 pre-registration before any E2 run existed, because the
  tuned TD3 anchor (t28, seed 0) was measured in Stage 0 to sit trapped for exactly
  4 consecutive checkpoints -- one short of 5 -- while its eval curve sat at the
  do-nothing baseline. A single threshold of 5 therefore risked confirming P1 on
  all-zero cells and leaving P2's escape rate with an empty denominator.
  early pin                onset < 10,000 steps
  escape                   an 'off' pin event that ends before the last checkpoint
  time to escape           end - onset (end = first checkpoint no longer pinned off)
  flips before escape      gene off->on transitions in steps_compact during
                           [onset, end): each is noise-induced, since mu_b ~ -1
  pinned off at end        the run's last checkpoint is inside an 'off' pin event
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

MIN_RUN = 5
EARLY = 10_000


def _direction(log: pd.DataFrame, label: str) -> np.ndarray:
    on = log[f"{label}_frac_on"].to_numpy()
    off = log[f"{label}_frac_off"].to_numpy()
    sat = np.maximum(on + off, 1e-12)
    return np.where(off / sat >= 0.95, "off", np.where(on / sat >= 0.95, "on", "mixed"))


def pin_events(log: pd.DataFrame, label: str, min_run: int = MIN_RUN,
               kind: str = "pin") -> List[dict]:
    """kind='pin': the prompt's definition with direction; kind='trap': trapped off."""
    t = log["timesteps"].to_numpy()
    if kind == "trap":
        pinned = log[f"{label}_trapped"].to_numpy().astype(bool)
        d = np.full(len(t), "off")
    else:
        pinned = log[f"{label}_pinned"].to_numpy().astype(bool)
        d = _direction(log, label)
    events = []
    i, n = 0, len(t)
    while i < n:
        if not pinned[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and pinned[j + 1] and d[j + 1] == d[i]:
            j += 1
        if j - i + 1 >= min_run:
            ended = j + 1 < n
            events.append({"direction": str(d[i]), "onset": int(t[i]),
                           "end": int(t[j + 1]) if ended else None,
                           "last_pinned": int(t[j]), "n_ckpt": int(j - i + 1),
                           "escaped": bool(ended and d[i] == "off"),
                           "at_end": not ended})
        i = j + 1
    return events


def load_compact(run_dir: Path) -> Optional[pd.DataFrame]:
    f = Path(run_dir) / "steps_compact.csv.gz"
    if not f.is_file():
        return None
    with gzip.open(f, "rt") as fh:
        return pd.read_csv(fh)


def flips_between(compact: pd.DataFrame, label: str, t0: int, t1: Optional[int]) -> dict:
    m = compact["timesteps"] > t0
    if t1 is not None:
        m &= compact["timesteps"] <= t1
    c = compact[m]
    prev, now = c[f"{label}_prev"].to_numpy(), c[label].to_numpy()
    off = prev == 0
    return {"off_steps": int(off.sum()), "off_to_on": int(np.sum(off & (now == 1))),
            "on_to_off": int(np.sum((prev == 1) & (now == 0)))}


def summarize_run(run_dir: Path, label: str = "fbs0_power_status",
                  min_run: int = MIN_RUN) -> dict:
    run_dir = Path(run_dir)
    f = run_dir / "probe_log.csv"
    if not f.is_file():
        return {"probe": False}
    log = pd.read_csv(f)
    ev = pin_events(log, label, min_run=min_run)
    compact = load_compact(run_dir)
    off_ev = [e for e in ev if e["direction"] == "off"]
    trap = (pin_events(log, label, min_run=min_run, kind="trap")
            if f"{label}_trapped" in log else [])
    out = {
        "n_trap_events": len(trap),
        "early_trap": any(e["onset"] < EARLY for e in trap),
        "first_trap_onset": min((e["onset"] for e in trap), default=None),
        "trapped_at_end": any(e["at_end"] for e in trap),
        "n_trap_escapes": sum(e["escaped"] for e in trap),
        "trap_events": trap,
        "n_mixed_events": sum(e["direction"] == "mixed" for e in ev),
        "probe": True,
        "n_pin_events": len(ev),
        "n_off_events": len(off_ev),
        "n_on_events": sum(e["direction"] == "on" for e in ev),
        "early_pin": any(e["onset"] < EARLY for e in ev),
        "early_pin_off": any(e["onset"] < EARLY for e in off_ev),
        "first_onset": min((e["onset"] for e in ev), default=None),
        "first_direction": ev[0]["direction"] if ev else None,
        "first_off_onset": min((e["onset"] for e in off_ev), default=None),
        "pinned_off_at_end": any(e["at_end"] for e in off_ev),
        "n_escapes": sum(e["escaped"] for e in off_ev),
        "events": ev,
    }
    if compact is not None and trap:
        for e in trap:
            e.update(flips_between(compact, label, e["onset"], e["end"]))
        out["trap_escape_times"] = [e["end"] - e["onset"] for e in trap if e["escaped"]]
        out["trap_flips_before_escape"] = [e["off_to_on"] for e in trap if e["escaped"]]
    if compact is not None and off_ev:
        for e in off_ev:
            e.update(flips_between(compact, label, e["onset"], e["end"]))
        esc = [e for e in off_ev if e["escaped"]]
        out["escape_times"] = [e["end"] - e["onset"] for e in esc]
        out["flips_before_escape"] = [e["off_to_on"] for e in esc]
        stuck = [e for e in off_ev if not e["escaped"]]
        out["flips_while_stuck"] = sum(e["off_to_on"] for e in stuck)
        out["off_steps_while_stuck"] = sum(e["off_steps"] for e in stuck)
    return out


# --------------------------------------------------------------------------- #
# Multi-gene support (2 FBS and beyond)
# --------------------------------------------------------------------------- #
def gene_labels(run_dir: Path) -> List[str]:
    """The binary gene labels a run's probe_log.csv actually carries."""
    f = Path(run_dir) / "probe_log.csv"
    if not f.is_file():
        return []
    cols = pd.read_csv(f, nrows=0).columns
    return [c[: -len("_pinned")] for c in cols if c.endswith("_pinned")]


def summarize_run_multi(run_dir: Path, labels: Optional[List[str]] = None,
                        min_run: int = MIN_RUN) -> dict:
    """``{label: summarize_run(...)}`` for every binary gene in the run.

    ``summarize_run`` is left exactly as it was, so every 1-FBS number this
    module ever produced is reproduced bit for bit; this is the opt-in wrapper
    for runs with more than one on/off gene.
    """
    run_dir = Path(run_dir)
    labels = labels if labels is not None else gene_labels(run_dir)
    return {lab: summarize_run(run_dir, label=lab, min_run=min_run) for lab in labels}


def any_gene(multi: dict, key: str) -> bool:
    """True if any gene satisfies a boolean summary key (e.g. trapped_at_end)."""
    return any(bool(v.get(key)) for v in multi.values())


def all_genes(multi: dict, key: str) -> bool:
    """True if every gene satisfies it, and there is at least one gene."""
    return bool(multi) and all(bool(v.get(key)) for v in multi.values())
