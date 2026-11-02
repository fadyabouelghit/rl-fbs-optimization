"""Review round 1, Run C: force an FBS to stay on, during training only.

Two interventions, both off by default:

  MIN_ON(L)      after a switch to on, that FBS's on/off action is forced to +1
                 for the next L - 1 steps of the episode
  FORCE_ON(N)    for the first N training steps, every on/off action is forced
                 to +1; after N the agent's own actions apply

Placement and power actions are never touched.

Why this patches ``_sample_action`` instead of wrapping the env
---------------------------------------------------------------
A gym wrapper that rewrote the action inside ``step`` would leave SB3 storing
the ACTOR's action in the replay buffer while the ENV executed a different one,
so the critic would be trained on transitions that never happened. SB3 decides
both what to execute and what to store inside ``OffPolicyAlgorithm._sample_action``,
so overriding there is the only place where the two stay consistent by
construction. ``tests/test_letter.py`` asserts the buffer holds the forced value.

The override is active only while the model is collecting training rollouts.
Evaluation uses a separate env and ``predict``, and the tuner scores a reloaded
``best_model.zip``, so neither sees the intervention.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Sequence

import numpy as np


class OnOffOverride:
    """Decides, per env and per binary dim, when to emit +1 instead of the actor's value."""

    def __init__(self, binary_idx: Sequence[int], min_on: Optional[int] = None,
                 force_on_steps: Optional[int] = None, obs_normalized: bool = True):
        self.binary_idx = [int(b) for b in binary_idx]
        self.min_on = int(min_on) if min_on else None
        self.force_on_steps = int(force_on_steps) if force_on_steps else None
        self.thr = 0.0 if obs_normalized else 0.5
        self._hold: Optional[np.ndarray] = None       # (n_envs, n_bin) steps left to hold

    def __call__(self, actions: np.ndarray, last_obs, num_timesteps: int) -> np.ndarray:
        a = np.array(actions, dtype=np.float64, copy=True)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        n_env = a.shape[0]
        if self._hold is None or self._hold.shape[0] != n_env:
            self._hold = np.zeros((n_env, len(self.binary_idx)), dtype=int)

        if self.force_on_steps and num_timesteps < self.force_on_steps:
            a[:, self.binary_idx] = 1.0
            return a.reshape(actions.shape)

        if self.min_on and last_obs is not None:
            obs = np.asarray(last_obs)
            for k, b in enumerate(self.binary_idx):
                gene_on = obs[:, b] > self.thr          # gene BEFORE this step
                holding = self._hold[:, k] > 0
                # 1. still inside a hold window: force on, burn one step
                a[holding, b] = 1.0
                self._hold[holding, k] -= 1
                # 2. a fresh switch to on starts a new window of L - 1 further steps
                fresh = (~holding) & (~gene_on) & (a[:, b] >= 0.5)
                self._hold[fresh, k] = max(self.min_on - 1, 0)
                # 3. the gene going off ends any window (it cannot be "held on")
                ended = (~holding) & gene_on & (a[:, b] <= -0.5)
                self._hold[ended, k] = 0
        return a.reshape(actions.shape)


@contextmanager
def onoff_override(model_cls, override: Optional[OnOffOverride]):
    """Patch ``_sample_action`` on an off-policy class for the duration of a run."""
    if override is None:
        yield
        return
    original = model_cls._sample_action

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        action, buffer_action = original(self, learning_starts, action_noise, n_envs)
        buffer_action = override(buffer_action, getattr(self, "_last_obs", None),
                                 int(self.num_timesteps))
        # unscale_action is the inverse SB3 uses to go buffer -> env, so both the
        # executed action and the stored one come from the same array.
        action = self.policy.unscale_action(buffer_action)
        return action, buffer_action

    model_cls._sample_action = _sample_action
    try:
        yield
    finally:
        model_cls._sample_action = original
