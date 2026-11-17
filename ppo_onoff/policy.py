"""OnOffPPOPolicy: a PPO actor-critic over c given b, plus twin Q critics that pick b.

Spaces. The algorithm and its rollout buffer use the env's own spaces
(observation 16, action 10 for 2 FBSs). Inside the policy:

- the PPO actor and value net see [s, b] (16 + N) and the Gaussian is over the
  4N continuous genes c only;
- the twin critics (SB3 ``ContinuousCritic``, 2 heads, ReLU) see s and the full
  5N action [c, b] (16 + 10 = 26 inputs), like on_off_td3's.

The rollout buffer stores the full action [c, b*]. ``evaluate_actions`` reads
b* back out of it as the actor's extra input and scores only c, so PPO's ratio
and entropy are over c, and b* acts as part of the state. (Section 6.6 says
"store b* in obs"; carrying it in the stored action is the same thing without
an env wrapper.)

Selection, for acting, for the value bootstrap and for the critics' target:
    c_b     = clip(mu(s, b), -1, 1)       the PPO mean, clipped as the env clips it
    score_b = min(Q1, Q2)(s, [c_b, b])    online critics
    b*      = argmax_b score_b            (``forward`` only: a random pattern with prob eps)

The critics are built after PPO's own ``_build``, so PPO's optimizer never
holds their parameters; they have their own Adam.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy, ContinuousCritic
from stable_baselines3.common.torch_layers import FlattenExtractor
from torch import nn

from on_off_td3.policy import all_patterns


def _pilot_spaces(obs_space: spaces.Box, n_fbs: int, n_cont: int, dtype):
    lo = np.concatenate([obs_space.low.reshape(-1), -np.ones(n_fbs, np.float32)])
    hi = np.concatenate([obs_space.high.reshape(-1), np.ones(n_fbs, np.float32)])
    return (spaces.Box(lo.astype(np.float32), hi.astype(np.float32), dtype=np.float32),
            spaces.Box(-1.0, 1.0, shape=(n_cont,), dtype=dtype))


class OnOffPPOPolicy(ActorCriticPolicy):
    def __init__(self, observation_space: spaces.Box, action_space: spaces.Box, lr_schedule, *,
                 onoff_idx: Sequence[int] = (), cont_idx: Sequence[int] = (),
                 critic_net_arch: Sequence[int] = (64, 64), **kwargs):
        self.onoff_idx = [int(i) for i in onoff_idx]
        self.cont_idx = [int(i) for i in cont_idx]
        if not self.onoff_idx or not self.cont_idx:
            raise ValueError("OnOffPPOPolicy needs onoff_idx and cont_idx")
        if len(self.onoff_idx) + len(self.cont_idx) != int(np.prod(action_space.shape)):
            raise ValueError("onoff_idx + cont_idx must cover the env action")
        self.critic_net_arch = [int(h) for h in critic_net_arch]
        pilot_obs, pilot_act = _pilot_spaces(observation_space, len(self.onoff_idx),
                                             len(self.cont_idx), action_space.dtype)
        super().__init__(pilot_obs, pilot_act, lr_schedule, **kwargs)
        # From here on the policy faces the env: predict() takes the 16-dim obs and
        # returns (and clips to) the full 10-dim action; save/load rebuilds from these.
        self.pilot_observation_space, self.pilot_action_space = pilot_obs, pilot_act
        self.observation_space, self.action_space = observation_space, action_space
        self.register_buffer("patterns", all_patterns(len(self.onoff_idx)), persistent=False)

        def critic():
            fe = FlattenExtractor(observation_space)
            return ContinuousCritic(observation_space, action_space, net_arch=self.critic_net_arch,
                                    features_extractor=fe, features_dim=fe.features_dim,
                                    activation_fn=nn.ReLU, n_critics=2,
                                    share_features_extractor=False)
        self.critic = critic().to(self.device)
        self.critic_target = critic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.set_training_mode(False)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=lr_schedule(1))
        # exploration over b: set by PPOOnOff during training; a loaded policy is greedy
        self.eps = 0.0
        self.rng = np.random.default_rng(0)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(onoff_idx=self.onoff_idx, cont_idx=self.cont_idx,
                    critic_net_arch=self.critic_net_arch)
        return data

    # ------------------------------------------------------------------ #
    # layout helpers
    # ------------------------------------------------------------------ #
    @property
    def n_patterns(self) -> int:
        return int(self.patterns.shape[0])

    def full_action(self, c: th.Tensor, b: th.Tensor) -> th.Tensor:
        a = th.zeros(c.shape[0], len(self.cont_idx) + len(self.onoff_idx), dtype=c.dtype, device=c.device)
        a[:, self.cont_idx] = c
        a[:, self.onoff_idx] = b.to(c.dtype)
        return a

    @staticmethod
    def augment(obs: th.Tensor, b: th.Tensor) -> th.Tensor:
        return th.cat([obs.float(), b.to(th.float32)], dim=1)

    # ------------------------------------------------------------------ #
    # PPO side: pi(c | s, b), V(s, b)
    # ------------------------------------------------------------------ #
    def _dist_value(self, obs: th.Tensor, b: th.Tensor):
        features = self.extract_features(self.augment(obs, b))
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_f, vf_f = features
            latent_pi = self.mlp_extractor.forward_actor(pi_f)
            latent_vf = self.mlp_extractor.forward_critic(vf_f)
        return self._get_action_dist_from_latent(latent_pi), self.value_net(latent_vf)

    def mean_c(self, obs: th.Tensor, b: th.Tensor) -> th.Tensor:
        """The PPO actor's mean placement for pattern b (unclipped)."""
        features = super(ActorCriticPolicy, self).extract_features(self.augment(obs, b),
                                                                   self.pi_features_extractor)
        return self.action_net(self.mlp_extractor.forward_actor(features))

    # ------------------------------------------------------------------ #
    # critic side: choose b
    # ------------------------------------------------------------------ #
    def pattern_scores(self, critic, obs: th.Tensor) -> th.Tensor:
        """(B, 2^N): min over the critic's heads of Q(s, [clip(mu(s, b)), b]) for every b."""
        B, P = obs.shape[0], self.n_patterns
        o = obs.float().repeat_interleave(P, dim=0)
        pats = self.patterns.to(o.device).repeat(B, 1)
        c = th.clamp(self.mean_c(o, pats), -1.0, 1.0)
        qs = th.cat(critic(o, self.full_action(c, pats)), dim=1)
        return qs.min(dim=1).values.reshape(B, P)

    def greedy_pattern(self, obs: th.Tensor) -> th.Tensor:
        with th.no_grad():
            return self.pattern_scores(self.critic, obs).argmax(dim=1)

    def explore_pattern(self, idx: th.Tensor) -> th.Tensor:
        """eps-greedy over patterns. Draws the coin and the pick on every call, so the
        stream does not depend on eps; uses its own Generator, not torch's."""
        idx = idx.clone()
        for i in range(idx.shape[0]):
            coin, pick = self.rng.random(), int(self.rng.integers(self.n_patterns))
            if coin < self.eps:
                idx[i] = pick
        return idx

    # ------------------------------------------------------------------ #
    # SB3 entry points
    # ------------------------------------------------------------------ #
    def forward(self, obs: th.Tensor, deterministic: bool = False):
        """Rollout collection: b* eps-greedy, c ~ pi(. | s, b*)."""
        idx = self.explore_pattern(self.greedy_pattern(obs))
        b = self.patterns[idx]
        dist, values = self._dist_value(obs, b)
        c = dist.get_actions(deterministic=deterministic)
        return self.full_action(c, b), values, dist.log_prob(c)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        """PPO's update: b* is read from the stored action and is an input; only c is scored."""
        b, c = actions[:, self.onoff_idx], actions[:, self.cont_idx]
        dist, values = self._dist_value(obs, b)
        return values, dist.log_prob(c), dist.entropy()

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """Bootstrap values (end of rollout, time-limit truncation): V(s, greedy b)."""
        _, values = self._dist_value(obs, self.patterns[self.greedy_pattern(obs)])
        return values

    def get_distribution(self, obs: th.Tensor):
        """pi(c | s, greedy b)."""
        dist, _ = self._dist_value(obs, self.patterns[self.greedy_pattern(obs)])
        return dist

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        b = self.patterns[self.greedy_pattern(observation)]
        dist, _ = self._dist_value(observation, b)
        return self.full_action(dist.get_actions(deterministic=deterministic), b)
