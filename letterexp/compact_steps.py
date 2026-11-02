"""A per-step log small enough to keep for every run: steps_compact.csv.gz.

Columns: timesteps, env, episode, step, then for every binary gene
``<label>_prev`` (the gene before the step, decoded from the observation the
action was chosen on), ``<label>`` (after the step), ``a_<label>`` (the
executed action on that dim, noise included and clipped), and finally
``reward_objective``.

``<label>_prev`` matters at episode starts: reset re-draws the binary genes,
so a flip analysis that only diffed consecutive rows of steps.csv would count
every reset as a flip. Here a flip is ``prev != after`` within one row.

At ~1 MB gzipped per 100k-step run this replaces the 26 MB steps.csv for the
flip analysis, so runs can drop steps.csv after scoring (``--slim``).
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Sequence

import numpy as np


def _make_class():
    from stable_baselines3.common.callbacks import BaseCallback

    class CompactStepCallback(BaseCallback):
        def __init__(self, out_dir, binary_idx: Sequence[int], labels: Sequence[str],
                     obs_normalized: bool = True, verbose: int = 0,
                     positions: bool = False, state_low=None, state_high=None,
                     n_fbs: int = 0, genes_per_fbs: int = 5):
            super().__init__(verbose)
            self.out_dir = Path(out_dir) if out_dir else None   # None -> run dir
            self.binary_idx = [int(b) for b in binary_idx]
            self.labels = list(labels)
            # normalized obs map a {0,1} gene to {-1,+1}
            self.thr = 0.0 if obs_normalized else 0.5
            # Opt-in (round 1, reviewer M2): the continuous genes BEFORE the step,
            # decoded from the observation the action was chosen on. Off by
            # default, and it only reads values the callback already receives, so
            # it cannot affect training.
            self.positions = bool(positions)
            self.obs_normalized = bool(obs_normalized)
            self.state_low = np.asarray(state_low, float) if state_low is not None else None
            self.state_high = np.asarray(state_high, float) if state_high is not None else None
            self.n_fbs = int(n_fbs)
            self.genes_per_fbs = int(genes_per_fbs)
            self._rows: list = []
            self._ep = None
            self._st = None

        def _decode(self, obs_row):
            """Observation -> physical state values (inverse of env normalisation)."""
            v = np.asarray(obs_row, float)[: self.state_low.size]
            if not self.obs_normalized:
                return v
            rng = np.maximum(self.state_high - self.state_low, 1e-8)
            return (v + 1.0) / 2.0 * rng + self.state_low

        def _on_training_start(self) -> None:
            if self.out_dir is None:
                self.out_dir = Path(self.model.logger.get_dir())
            self.out_path = self.out_dir / "steps_compact.csv.gz"

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)
            actions = np.asarray(self.locals.get("actions"), dtype=np.float64)
            last_obs = getattr(self.model, "_last_obs", None)
            if self._ep is None:
                self._ep = np.zeros(len(infos), dtype=int)
                self._st = np.zeros(len(infos), dtype=int)
            for i, info in enumerate(infos):
                self._st[i] += 1
                state = np.asarray(info.get("state"), dtype=float).reshape(-1)
                row = [self.num_timesteps, i, int(self._ep[i]), int(self._st[i])]
                for b in self.binary_idx:
                    prev = int(np.asarray(last_obs)[i, b] > self.thr) if last_obs is not None else -1
                    row += [prev, int(state[b] > 0.5), float(actions[i, b])]
                if self.positions and last_obs is not None and self.state_low is not None:
                    pre = self._decode(np.asarray(last_obs)[i])
                    for k in range(self.n_fbs):
                        o = k * self.genes_per_fbs
                        row += [float(pre[o]), float(pre[o + 1]),
                                float(pre[o + 2]), float(pre[o + 3])]
                row.append(float(info.get("reward_objective", np.nan)))
                self._rows.append(row)
                if i < len(dones) and dones[i]:
                    self._ep[i] += 1
                    self._st[i] = 0
            return True

        def _on_training_end(self) -> None:
            cols = ["timesteps", "env", "episode", "step"]
            for lab in self.labels:
                cols += [f"{lab}_prev", lab, f"a_{lab}"]
            if self.positions and self.state_low is not None:
                for k in range(self.n_fbs):
                    cols += [f"fbs{k}_x_pre", f"fbs{k}_y_pre",
                             f"fbs{k}_z_pre", f"fbs{k}_p_pre"]
            cols.append("reward_objective")
            with gzip.open(self.out_path, "wt") as f:
                f.write(",".join(cols) + "\n")
                for r in self._rows:
                    f.write(",".join(
                        (f"{v:.5g}" if isinstance(v, float) else str(v)) for v in r) + "\n")

    return CompactStepCallback


_CB = None


def CompactStepCallback(*args, **kwargs):
    global _CB
    if _CB is None:
        _CB = _make_class()
    return _CB(*args, **kwargs)
