"""TryBothPolicy: a TD3Policy whose actor outputs only c and whose predict picks b.

The critic is SB3's ordinary ContinuousCritic on the full 5N env action, so
the replay memory stores the executed action unchanged. ``predict`` (used by
evaluation, scoring and SB3's own rollouts outside warm-up) returns the full
action with b = argmax over all 2^N patterns, so a saved ``model.zip`` loaded
with plain ``TD3.load`` / ``DDPG.load`` is a complete greedy policy.
"""
from __future__ import annotations

import itertools
from typing import Any, Sequence

import torch as th
from gymnasium import spaces
from stable_baselines3.td3.policies import Actor, TD3Policy


def all_patterns(n: int) -> th.Tensor:
    """(2^n, n) tensor of every on/off pattern in {-1, +1}; row 2^n - 1 is all on."""
    return th.tensor(list(itertools.product((-1.0, 1.0), repeat=n)), dtype=th.float32)


class TryBothPolicy(TD3Policy):
    def __init__(self, *args, onoff_idx: Sequence[int] = (), cont_idx: Sequence[int] = (), **kwargs):
        self.onoff_idx = [int(i) for i in onoff_idx]
        self.cont_idx = [int(i) for i in cont_idx]
        if not self.onoff_idx or not self.cont_idx:
            raise ValueError("TryBothPolicy needs onoff_idx and cont_idx")
        super().__init__(*args, **kwargs)
        self.register_buffer("patterns", all_patterns(len(self.onoff_idx)), persistent=False)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(onoff_idx=self.onoff_idx, cont_idx=self.cont_idx)
        return data

    def make_actor(self, features_extractor=None) -> Actor:
        kw = self._update_features_extractor(self.actor_kwargs, features_extractor)
        kw = dict(kw)
        kw["action_space"] = spaces.Box(-1.0, 1.0, shape=(len(self.cont_idx),), dtype=self.action_space.dtype)
        return Actor(**kw).to(self.device)

    # ------------------------------------------------------------------ #
    @property
    def n_patterns(self) -> int:
        return int(self.patterns.shape[0])

    @property
    def all_on(self) -> int:
        return self.n_patterns - 1

    def full_action(self, c: th.Tensor, b: th.Tensor) -> th.Tensor:
        """[c, b] in the env's per-FBS gene layout."""
        a = th.zeros(c.shape[0], len(self.cont_idx) + len(self.onoff_idx), dtype=c.dtype, device=c.device)
        a[:, self.cont_idx] = c
        a[:, self.onoff_idx] = b.to(c.dtype)
        return a

    def pattern_scores(self, critic, obs: th.Tensor, c: th.Tensor) -> th.Tensor:
        """(B, 2^N): min over the critic's heads of Q(s, [c, b]) for every pattern b."""
        B, P = obs.shape[0], self.n_patterns
        pats = self.patterns.to(c.device)
        o = obs.repeat_interleave(P, dim=0)
        a = self.full_action(c.repeat_interleave(P, dim=0), pats.repeat(B, 1))
        qs = th.cat(critic(o, a), dim=1)                    # (B*P, n_critics)
        return qs.min(dim=1).values.reshape(B, P)

    def best_pattern(self, obs: th.Tensor, c: th.Tensor) -> th.Tensor:
        """Index of the best pattern per row, ranked by the ONLINE critic."""
        with th.no_grad():
            return self.pattern_scores(self.critic, obs, c).argmax(dim=1)

    def _predict(self, observation, deterministic: bool = False) -> th.Tensor:
        c = self.actor(observation)
        return self.full_action(c, self.patterns[self.best_pattern(observation, c)])
