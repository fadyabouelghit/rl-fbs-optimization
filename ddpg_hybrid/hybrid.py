"""The minimal remedy: a stochastic Bernoulli head on the on/off dims only.

Everything else is the stock SB3 TD3/DDPG machinery. The actor's network is
unchanged (same MLP, same output width); only how its output is *read* on the
binary dims changes:

    continuous dim j :  a_j = tanh(z_j)                       (as before)
    binary dim b     :  z_b is a Bernoulli logit, P(on) = sigmoid(z_b)
        training     :  binary-concrete / Gumbel-softmax sample with
                        straight-through gradient:
                            y    = sigmoid((z_b + L) / tau),  L ~ Logistic(0, 1)
                            hard = 1[y > 0.5]                 (so P(hard = 1) = sigmoid(z_b))
                            a_b  = 2 * (hard + y - y.detach()) - 1   in {-1, +1}
        evaluation   :  a_b = +1 if z_b > 0 else -1   (argmax)

The env's sticky rule therefore always sees |a_b| = 1 >= 0.5, so the gene is
set to 1[a_b >= 0] at every step. The critic is trained on, and the actor is
updated through, the *sampled* value. The binary dims get zero exploration
noise (the sample is the exploration), and in TD3 the target-policy smoothing
noise is zeroed on the binary dims for the same reason; continuous dims keep
the base config's noise and smoothing exactly.

"Sampling vs argmax" is routed by ``deterministic``: SB3 rollouts call
``predict(deterministic=False)`` and evaluation calls it with True. Inside
``train()`` the actor and actor_target are called directly and sample.

Pinned at SB3 2.9.0: ``HybridMixin.train`` is TD3.train from that release with
one added line (binary target noise = 0). Check ``SB3_VERSION`` before
trusting it on another version.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import stable_baselines3
import torch as th
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.td3.policies import Actor, TD3Policy
from torch.nn import functional as F

SB3_VERSION = "2.9.0"


def _check_version() -> None:
    if stable_baselines3.__version__ != SB3_VERSION:
        raise RuntimeError(
            f"ddpg_hybrid copies TD3.train from SB3 {SB3_VERSION}; installed is "
            f"{stable_baselines3.__version__}. Re-diff HybridMixin.train before running.")


class HybridActor(Actor):
    is_hybrid = True

    def __init__(self, *args, binary_idx: Sequence[int] = (), gumbel_tau: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.binary_idx = [int(b) for b in binary_idx]
        self.gumbel_tau = float(gumbel_tau)
        mask = th.zeros(self.mu[-2].out_features, dtype=th.bool)
        mask[self.binary_idx] = True
        self.register_buffer("binary_mask", mask, persistent=False)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(binary_idx=self.binary_idx, gumbel_tau=self.gumbel_tau)
        return data

    def forward(self, obs: th.Tensor, sample: bool = True) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        z = self.mu[:-1](features)                     # pre-tanh / logits
        cont = th.tanh(z)
        if sample:
            u = th.rand_like(z).clamp_(1e-6, 1 - 1e-6)
            y = th.sigmoid((z + th.log(u) - th.log1p(-u)) / self.gumbel_tau)
            hard = (y > 0.5).to(z.dtype)
            st = hard + (y - y.detach())          # exact 0 or 1 forward, d/dz of y backward
            binv = 2.0 * st - 1.0
        else:
            binv = 2.0 * (z > 0).to(z.dtype) - 1.0
        return th.where(self.binary_mask, binv, cont)

    def _predict(self, observation, deterministic: bool = False) -> th.Tensor:
        return self(observation, sample=not deterministic)


class HybridTD3Policy(TD3Policy):
    """TD3Policy whose actor (and actor_target) is a HybridActor."""

    def __init__(self, *args, binary_idx: Sequence[int] = (), gumbel_tau: float = 1.0, **kwargs):
        self.binary_idx = [int(b) for b in binary_idx]
        self.gumbel_tau = float(gumbel_tau)
        super().__init__(*args, **kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(binary_idx=self.binary_idx, gumbel_tau=self.gumbel_tau)
        return data

    def make_actor(self, features_extractor=None) -> HybridActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return HybridActor(**actor_kwargs, binary_idx=self.binary_idx,
                           gumbel_tau=self.gumbel_tau).to(self.device)

    def _predict(self, observation, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, sample=not deterministic)


class HybridMixin:
    """Put before TD3/DDPG in the MRO: ``class HybridDDPG(HybridMixin, DDPG)``."""

    def _binary_idx(self):
        return list(self.policy.binary_idx)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        action, buffer_action = super()._sample_action(learning_starts, action_noise, n_envs)
        # Warm-up draws are uniform on [-1, 1]; snap the binary dims to {-1, +1}
        # so the buffer only ever holds values the Bernoulli head can emit.
        # After warm-up this is a no-op (zero noise on a +-1 output).
        b = self._binary_idx()
        if b:
            buffer_action = buffer_action.copy()
            buffer_action[:, b] = np.where(buffer_action[:, b] >= 0, 1.0, -1.0)
            action = self.policy.unscale_action(buffer_action)
        return action, buffer_action

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # == stable_baselines3 2.9.0 TD3.train, plus the line marked HYBRID ==
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                noise[:, self._binary_idx()] = 0.0                       # HYBRID
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                actor_loss = -self.critic.q1_forward(replay_data.observations,
                                                     self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))


# --------------------------------------------------------------------------- #
# Q-difference variant (letter Step 2)
# --------------------------------------------------------------------------- #
class QDiffHybridMixin(HybridMixin):
    """Train the on/off logit on Q(on) - Q(off), not on the critic's local slope.

    ``HybridMixin`` backpropagates the actor loss ``-Q1(s, a)`` through the
    straight-through sample, so the binary dimension is updated by dQ1/da_b at
    the sampled action. E3 measured that local slope pointing the wrong way
    (toward off) during traps while the critic's own on-vs-off comparison
    pointed the right way. This variant uses the comparison directly:

        loss_b = - mean( sigmoid(z) * stopgrad[ Q1(s, a*|a_b=+1)
                                                - Q1(s, a*|a_b=-1) ] )

    with ``a*`` the actor's action and the binary dim set to its sample. The
    bracket is a constant w.r.t. the actor, so the only gradient path into z is
    through ``sigmoid(z)``: the probability of "on" rises exactly when the
    critic says on is worth more, by however much it is worth, and no gradient
    flows through the critic's geometry at the sampled point.

    Continuous dims keep the usual DPG loss ``-mean(Q1(s, a*))``, with the
    binary dims detached so that term cannot touch z. Rollout sampling,
    evaluation argmax, the critic update and the zeroed binary target-smoothing
    noise are all inherited unchanged.
    """

    def _actor_loss(self, obs: th.Tensor) -> th.Tensor:
        b = self._binary_idx()
        actor_out = self.actor(obs)                      # straight-through sample
        if not b:
            return -self.critic.q1_forward(obs, actor_out).mean()

        # --- continuous dims: ordinary DPG, binary dims held constant --------
        a_cont = actor_out.clone()
        a_cont[:, b] = actor_out[:, b].detach()
        loss_cont = -self.critic.q1_forward(obs, a_cont).mean()

        # --- binary dims: weight sigmoid(z) by the critic's on-vs-off gap ----
        feats = self.actor.extract_features(obs, self.actor.features_extractor)
        z = self.actor.mu[:-1](feats)
        a_det = actor_out.detach()
        loss_bin = 0.0
        for dim in b:
            with th.no_grad():
                a_on, a_off = a_det.clone(), a_det.clone()
                a_on[:, dim], a_off[:, dim] = 1.0, -1.0
                gap = (self.critic.q1_forward(obs, a_on)
                       - self.critic.q1_forward(obs, a_off)).reshape(-1)
            loss_bin = loss_bin - (th.sigmoid(z[:, dim]) * gap).mean()
        return loss_cont + loss_bin

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # == SB3 2.9.0 TD3.train, with the HYBRID line and the actor loss swap ==
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                noise[:, self._binary_idx()] = 0.0                       # HYBRID
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                actor_loss = self._actor_loss(replay_data.observations)   # QDIFF
                actor_losses.append(actor_loss.item())
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
