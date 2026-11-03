"""Callbacks: the switch's learning step, its probe, and the compact step log.

Ordering matters for ``SwitchCallback``: the pilot's ``train`` appends extra
callbacks *after* the built-in ``CsvEvalCallback``, so on an eval step the
evaluation (with the switch as it is) has already run when this callback
fires. It first saves ``best_switch.pt`` if the eval just wrote a new
``best_model.zip``, and only then updates the switch. The best pair therefore
always comes from the same evaluation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import unwrap_vec_wrapper

from .wrapper import SwitchVecEnv

SAT_FRAC = 0.95       # same share as letterexp.probe.PIN_FRAC


def switch_env(model) -> SwitchVecEnv:
    env = unwrap_vec_wrapper(model.get_env(), SwitchVecEnv)
    if env is None:
        raise RuntimeError("model's env is not wrapped in SwitchVecEnv")
    return env


class SwitchCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._best_stamp = None

    def _on_training_start(self) -> None:
        self.env = switch_env(self.model)
        self.agent = self.env.agent
        self.agent.normalizer = self.model.get_vec_normalize_env()
        self.run_dir = Path(self.model.logger.get_dir())

    def _on_step(self) -> bool:
        t = int(self.num_timesteps)
        best = self.run_dir / "best_model.zip"
        if best.is_file():
            stamp = best.stat().st_mtime_ns
            if stamp != self._best_stamp:
                self.agent.save(self.run_dir / "best_switch.pt", meta={"timesteps": t})
                self._best_stamp = stamp
        self.agent.on_step(t)
        return True

    def _on_training_end(self) -> None:
        self.agent.save(self.run_dir / "switch.pt", meta={"timesteps": int(self.num_timesteps)})
        (self.run_dir / "switch_summary.json").write_text(json.dumps({
            "n_updates": self.agent.n_updates, "n_stored": self.agent.n_stored,
            "n_decisions": self.env.n_decisions, "final_eps": self.agent.eps,
            "last_loss": self.agent.last_loss}, indent=2))


class SwitchProbeCallback(BaseCallback):
    """Every ``every`` steps, on the fixed probe set, per FBS j:

        gap_j = max Q(s, a | j on) - max Q(s, a | j off)     (joint actions a)
        greedy_j = FBS j's state in argmax_a Q(s, a)

    ``{lab}_off_given_off`` / ``{lab}_trapped`` are the switch's counterpart of
    ``letterexp.probe``'s actor-based trap: share of probe states whose gene is
    OFF where the greedy switch keeps it off, and whether that share is at
    least 0.95. Same column names, so ``letterexp.pins.pin_events(...,
    kind='trap')`` reads ``switch_probe.csv`` unchanged.

    Read-only on the network under no_grad, no RNG: bitwise neutral.
    """

    def __init__(self, probe_path: Path, binary_idx: Sequence[int], labels: Sequence[str],
                 every: int = 1000, out_dir: Optional[Path] = None, verbose: int = 0):
        super().__init__(verbose)
        self.probe_path = Path(probe_path)
        self.binary_idx = [int(b) for b in binary_idx]
        self.labels = list(labels)
        self.every = int(every)
        self.out_dir = Path(out_dir) if out_dir else None
        self._rows: list = []
        self._gaps: list = []
        self._t_seen: list = []
        self._last = -1

    def _on_training_start(self) -> None:
        if self.out_dir is None:
            self.out_dir = Path(self.model.logger.get_dir())
        self.env = switch_env(self.model)
        self.layout = self.env.layout
        d = np.load(self.probe_path)
        self._obs = d["obs"].astype(np.float32)
        self._gene_on = d["states"][:, self.binary_idx] > 0.5
        self._probe()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every == 0 and self.num_timesteps != self._last:
            self._probe()
        return True

    def _probe(self) -> None:
        t = int(self.num_timesteps)
        self._last = t
        q = self.env.agent.q_values(self._obs)                     # (S, 2^N)
        greedy = q.argmax(axis=1)
        row = {"timesteps": t, "eps": float(self.env.agent.eps),
               "n_updates": int(self.env.agent.n_updates)}
        gaps = []
        for j, lab in enumerate(self.labels):
            on = self.layout.on_mask(j)
            gap = q[:, on].max(axis=1) - q[:, ~on].max(axis=1)
            g_on = ((greedy >> j) & 1) == 1
            off = ~self._gene_on[:, j]
            ogo = float(np.mean(~g_on[off])) if off.any() else float("nan")
            ogn = float(np.mean(g_on[~off])) if (~off).any() else float("nan")
            row.update({
                f"{lab}_gap_median": float(np.median(gap)), f"{lab}_gap_mean": float(gap.mean()),
                f"{lab}_frac_greedy_on": float(g_on.mean()),
                f"{lab}_off_given_off": ogo, f"{lab}_on_given_on": ogn,
                f"{lab}_trapped": int(ogo >= SAT_FRAC) if ogo == ogo else 0,
            })
            gaps.append(gap.astype(np.float32))
        self._rows.append(row)
        self._gaps.append(np.stack(gaps))
        self._t_seen.append(t)

    def _on_training_end(self) -> None:
        if self.num_timesteps != self._last:
            self._probe()
        path = self.out_dir / "switch_probe.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self._rows[0]))
            w.writeheader()
            w.writerows(self._rows)
        np.savez_compressed(self.out_dir / "switch_probe_raw.npz",
                            timesteps=np.array(self._t_seen), gap=np.stack(self._gaps),
                            labels=np.array(self.labels))
        (self.out_dir / "switch_probe_meta.json").write_text(json.dumps({
            "probe_path": str(self.probe_path), "every": self.every,
            "n_states": int(self._obs.shape[0]), "trap_frac": SAT_FRAC,
            "labels": self.labels, "binary_idx": self.binary_idx}, indent=2))


def make_compact_step_callback(*args, **kwargs):
    """``letterexp``'s compact step log, fed the *executed* full action.

    The pilot's own action has only the 4N continuous dims, so the parent
    callback's ``actions[i, b]`` would index the wrong gene. The wrapper keeps
    the full [c, b] it sent to the env; substitute it into this callback's own
    copy of the locals (other callbacks keep theirs) and reuse the parent as is.
    """
    from letterexp.compact_steps import _make_class

    base = _make_class()

    class SwitchCompactStepCallback(base):
        def _on_step(self) -> bool:
            env = switch_env(self.model)
            self.locals["actions"] = env.last_full_action[None, :].astype(np.float64)
            return super()._on_step()

    return SwitchCompactStepCallback(*args, **kwargs)
