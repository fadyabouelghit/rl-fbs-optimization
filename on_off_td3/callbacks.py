"""Instrumentation: the critic's on/off preference on the probe set, plus the compact step log.

``TryBothProbeCallback`` writes ``tryboth_probe.csv``. Every ``every`` steps,
on the fixed probe set, with c = mu(s) and scores = the online critics' min
over all 2^N patterns, per station j:

    gap_j        best score with j on - best score with j off
    greedy on    station j's state in the best pattern
    off_given_off / trapped   as letterexp.probe: share of probe states whose
                 gene is OFF where the greedy choice keeps j off, and >= 0.95

Same ``{lab}_trapped`` column names, so ``letterexp.pins.pin_events(...,
kind='trap')`` reads it unchanged. Forward passes under no_grad, no RNG:
bitwise neutral to training.

The compact step log is letterexp's own: the model's action is already the
full 5N executed action, so it needs no adapter.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

TRAP_FRAC = 0.95


class TryBothProbeCallback(BaseCallback):
    def __init__(self, probe_path: Path, binary_idx: Sequence[int], labels: Sequence[str],
                 every: int = 1000, out_dir: Optional[Path] = None, verbose: int = 0):
        super().__init__(verbose)
        self.probe_path = Path(probe_path)
        self.binary_idx = [int(b) for b in binary_idx]
        self.labels = list(labels)
        self.every = int(every)
        self.out_dir = Path(out_dir) if out_dir else None
        self._rows: list = []
        self._last = -1

    def _on_training_start(self) -> None:
        if self.out_dir is None:
            self.out_dir = Path(self.model.logger.get_dir())
        d = np.load(self.probe_path)
        pol = self.model.policy
        self._obs = th.as_tensor(d["obs"].astype(np.float32), device=pol.device)
        self._gene_on = d["states"][:, self.binary_idx] > 0.5
        self._pats = pol.patterns.cpu().numpy() > 0                 # (P, N) station on?
        self._probe()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every == 0 and self.num_timesteps != self._last:
            self._probe()
        return True

    def _probe(self) -> None:
        pol = self.model.policy
        t = int(self.num_timesteps)
        self._last = t
        with th.no_grad():
            c = pol.actor(self._obs)
            sc = pol.pattern_scores(pol.critic, self._obs, c).cpu().numpy()   # (S, P)
        greedy = sc.argmax(axis=1)
        row = {"timesteps": t, "eps": float(self.model.epsilon())}
        for j, lab in enumerate(self.labels):
            on = self._pats[:, j]
            gap = sc[:, on].max(axis=1) - sc[:, ~on].max(axis=1)
            g_on = self._pats[greedy, j]
            off = ~self._gene_on[:, j]
            ogo = float(np.mean(~g_on[off])) if off.any() else float("nan")
            row.update({
                f"{lab}_gap_median": float(np.median(gap)), f"{lab}_gap_mean": float(gap.mean()),
                f"{lab}_frac_greedy_on": float(g_on.mean()),
                f"{lab}_off_given_off": ogo,
                f"{lab}_on_given_on": float(np.mean(g_on[~off])) if (~off).any() else float("nan"),
                f"{lab}_trapped": int(ogo >= TRAP_FRAC) if ogo == ogo else 0,
            })
        self._rows.append(row)

    def _on_training_end(self) -> None:
        if self.num_timesteps != self._last:
            self._probe()
        with (self.out_dir / "tryboth_probe.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self._rows[0]))
            w.writeheader()
            w.writerows(self._rows)
        (self.out_dir / "tryboth_probe_meta.json").write_text(json.dumps({
            "probe_path": str(self.probe_path), "every": self.every, "trap_frac": TRAP_FRAC,
            "labels": self.labels, "binary_idx": self.binary_idx}, indent=2))


def instrument_callbacks(exp, protocol, probe_every: int = 1000, positions: bool = True) -> list:
    """Probe + compact step log (positions on by default, as the spec asks)."""
    from letterexp.compact_steps import CompactStepCallback
    from letterexp.probe import binary_dims_and_labels, probe_set_path

    idx, labels = binary_dims_and_labels(exp.env)
    probe = probe_set_path(protocol.code, protocol.backend)
    if not probe.is_file():
        raise FileNotFoundError(f"probe set {probe} missing; run: python -m letterexp.probe build "
                                f"--code {protocol.code} --backend {protocol.backend}")
    kw = {}
    if positions:
        from ppo.env import FlyingBaseStationEnv
        e = FlyingBaseStationEnv(exp.env, backend_kind="analytic")   # bounds only
        try:
            kw = {"positions": True, "state_low": e.state_low, "state_high": e.state_high,
                  "n_fbs": exp.env.num_fbs, "genes_per_fbs": e.genes_per_fbs}
        finally:
            e.close()
    return [TryBothProbeCallback(probe, idx, labels, every=probe_every),
            CompactStepCallback(None, idx, labels, obs_normalized=bool(exp.env.normalize_obs), **kw)]
