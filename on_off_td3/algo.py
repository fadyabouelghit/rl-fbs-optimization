"""TryBothTD3 / TryBothDDPG: SB3 TD3 / DDPG with the on/off pattern chosen by the critic.

``TryBothMixin.train`` is SB3 2.9.0's ``TD3.train`` with three changes, each
marked TRYBOTH:

1. next action: c' = actor_target(s') + clipped smoothing noise (continuous
   dims only), b' = argmax over patterns of the ONLINE critics' min at
   (s', [c', b]); the target critics then score (s', [c', b']).
2. actor loss: -Q1(s, [mu(s), b_train]) with b_train = all on (default) or
   the online best pattern (``actor_pattern='best'``), no gradient through b.
3. nothing else: critic MSE on the stored executed action, Polyak targets,
   delayed actor updates (TD3) are unchanged.

SB3's DDPG is TD3 with one critic, policy_delay 1 and zero target smoothing,
so the same mixin serves both.
"""
from __future__ import annotations

import numpy as np
import stable_baselines3
import torch as th
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F

from .config import TryBothParams

SB3_VERSION = "2.9.0"


def check_version() -> None:
    if stable_baselines3.__version__ != SB3_VERSION:
        raise RuntimeError(
            f"on_off_td3 copies TD3.train from SB3 {SB3_VERSION}; installed is "
            f"{stable_baselines3.__version__}. Re-diff TryBothMixin.train before running.")


class TryBothMixin:
    """Put before TD3/DDPG in the MRO. Configure with ``set_tryboth`` after construction."""

    tb: TryBothParams = TryBothParams()
    tb_total: int = 1

    def set_tryboth(self, params: TryBothParams, total_timesteps: int) -> None:
        self.tb = params
        self.tb_total = max(1, int(total_timesteps))

    # ------------------------------------------------------------------ #
    def epsilon(self) -> float:
        p = self.tb
        frac = min(1.0, self.num_timesteps / max(1.0, p.exploration_fraction * self.tb_total))
        return float(p.eps_start + frac * (p.eps_final - p.eps_start))

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        pol = self.policy
        on = pol.onoff_idx
        if self.num_timesteps < learning_starts:
            # warm-up: uniform action, b snapped to +-1 so the sticky rule applies it
            full = np.array([self.action_space.sample() for _ in range(n_envs)], dtype=np.float32)
            full[:, on] = np.where(full[:, on] >= 0, 1.0, -1.0)
        else:
            obs, _ = pol.obs_to_tensor(self._last_obs)
            with th.no_grad():
                c = pol.actor(obs).cpu().numpy()
            if action_noise is not None:
                c = np.clip(c + action_noise(), -1.0, 1.0)
            c_t = th.as_tensor(c, dtype=th.float32, device=pol.device)
            best = pol.best_pattern(obs, c_t).cpu().numpy()
            eps = self.epsilon()
            for i in range(n_envs):
                if np.random.rand() < eps:
                    best[i] = np.random.randint(pol.n_patterns)
            b = pol.patterns.cpu().numpy()[best]
            full = np.zeros((n_envs, len(pol.cont_idx) + len(on)), dtype=np.float32)
            full[:, pol.cont_idx] = c
            full[:, on] = b
        buffer_action = full
        action = pol.unscale_action(buffer_action)
        return action, buffer_action

    # ------------------------------------------------------------------ #
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # == stable_baselines3 2.9.0 TD3.train, with the TRYBOTH changes ==
        pol = self.policy
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                nxt = replay_data.next_observations
                c_next = self.actor_target(nxt)                                     # TRYBOTH
                noise = c_next.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                c_next = (c_next + noise).clamp(-1, 1)
                b_idx = pol.pattern_scores(self.critic, nxt, c_next).argmax(dim=1)  # TRYBOTH
                next_actions = pol.full_action(c_next, pol.patterns[b_idx])
                next_q_values = th.cat(self.critic_target(nxt, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                obs = replay_data.observations
                c = self.actor(obs)
                if self.tb.actor_pattern == "all_on":                              # TRYBOTH
                    b_idx = th.full((c.shape[0],), pol.all_on, dtype=th.long, device=c.device)
                else:
                    b_idx = pol.best_pattern(obs, c.detach())
                a = pol.full_action(c, pol.patterns[b_idx])
                actor_loss = -self.critic.q1_forward(obs, a).mean()
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
        self.logger.record("train/tryboth_eps", self.epsilon())


class TryBothTD3(TryBothMixin, TD3):
    pass


class TryBothDDPG(TryBothMixin, DDPG):
    pass
