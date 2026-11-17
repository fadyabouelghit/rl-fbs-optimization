"""PPOOnOff: SB3 PPO plus an off-policy twin-critic learner for the on/off choice.

PPO itself is unchanged (collect_rollouts, GAE, clipped update). The policy
(``OnOffPPOPolicy``) picks b* at every rollout step. One extra callback,
appended after the user's callbacks, runs once per env step:

1. store (s, a_executed, r_raw, s', terminated) in the critics' own replay
   memory (raw reward, normalized at sample time with VecNormalize's running
   statistics, as SB3's off-policy buffers do);
2. eps <- linear schedule 1.0 -> 0.05 over exploration_fraction * total steps;
3. after the warm-up, one critic gradient step:
       b'  = argmax_b min(Q1, Q2)(s', [clip(mu(s', b)), b])    online critics
       c'  = clip(mu(s', b'))                                   PPO mean, no noise
       y   = r + gamma (1 - terminated) min(Q1', Q2')(s', [c', b'])
       loss = sum_i MSE(Q_i(s, a), y)
   then Polyak tau on the target critics.

The target is computed under no_grad, so the critics never update the actor;
PPO's loss does not contain the critics, so PPO never updates them. After each
critic step the critics' gradients are set to None, so PPO's clip_grad_norm
over ``policy.parameters()`` sees only its own.

Own RNG for replay sampling and eps; the critics' init consumes the global
torch stream once at construction (seeded), nothing after.
"""
from __future__ import annotations

import numpy as np
import stable_baselines3
import torch as th
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, ConvertCallback
from stable_baselines3.common.utils import polyak_update

from .config import OnOffParams

SB3_VERSION = "2.9.0"
SEED_OFFSET = 104_729


def check_version() -> None:
    if stable_baselines3.__version__ != SB3_VERSION:
        raise RuntimeError(f"ppo_onoff is written against stable-baselines3 {SB3_VERSION} "
                           f"(collect_rollouts locals, ActorCriticPolicy internals); "
                           f"found {stable_baselines3.__version__}")


class _CriticCallback(BaseCallback):
    """Per-step hook into PPO's collect_rollouts (reads its locals, writes nothing back)."""

    def _on_step(self) -> bool:
        self.model._critic_on_step(self.locals)
        return True


class PPOOnOff(PPO):
    def __init__(self, *args, onoff: OnOffParams = OnOffParams(), **kwargs):
        check_version()
        self.onoff = onoff
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def _setup_learn(self, total_timesteps, callback=None, reset_num_timesteps=True,
                     tb_log_name="run", progress_bar=False):
        self._onoff_total = int(total_timesteps)
        seed = int(self.seed or 0)
        self.policy.rng = np.random.default_rng([seed, SEED_OFFSET, 1])
        self._replay_rng = np.random.default_rng([seed, SEED_OFFSET, 2])
        obs_dim = int(np.prod(self.observation_space.shape))
        act_dim = int(np.prod(self.action_space.shape))
        # PPO collects whole rollouts, so up to n_steps past total_timesteps
        cap = max(1, min(int(self.onoff.critic_buffer_size),
                         self._onoff_total + self.n_steps * self.n_envs))
        self._rb = {"obs": np.zeros((cap, obs_dim), np.float32),
                    "next": np.zeros((cap, obs_dim), np.float32),
                    "act": np.zeros((cap, act_dim), np.float32),
                    "rew": np.zeros(cap, np.float32), "term": np.zeros(cap, np.float32)}
        self._rb_cap, self._rb_pos, self._rb_full = cap, 0, False
        self.policy.eps = self.epsilon_at(0)
        self.critic_updates = 0
        self.critic_last_loss = float("nan")
        return super()._setup_learn(total_timesteps, callback, reset_num_timesteps,
                                    tb_log_name, progress_bar)

    def _init_callback(self, callback, progress_bar: bool = False):
        if callback is None:
            cbs = []
        elif isinstance(callback, list):
            cbs = list(callback)
        elif isinstance(callback, BaseCallback):
            cbs = [callback]
        else:
            cbs = [ConvertCallback(callback)]
        return super()._init_callback(cbs + [_CriticCallback()], progress_bar)

    def _excluded_save_params(self) -> list[str]:
        # the replay memory and RNGs stay out of model.zip / best_model.zip (written at
        # every new best): evaluation needs only the policy
        return super()._excluded_save_params() + ["_rb", "_replay_rng"]

    def epsilon_at(self, t: int) -> float:
        p = self.onoff
        horizon = max(1.0, p.exploration_fraction * self._onoff_total)
        return float(p.eps_start + min(1.0, t / horizon) * (p.eps_final - p.eps_start))

    # ------------------------------------------------------------------ #
    # per step
    # ------------------------------------------------------------------ #
    @property
    def rb_size(self) -> int:
        return self._rb_cap if self._rb_full else self._rb_pos

    def _critic_on_step(self, loc: dict) -> None:
        obs = loc["obs_tensor"].cpu().numpy()
        act = np.asarray(loc["clipped_actions"], np.float32)
        new_obs, dones, infos = loc["new_obs"], loc["dones"], loc["infos"]
        vn = self.get_vec_normalize_env()
        raw = vn.get_original_reward() if vn is not None else np.asarray(loc["rewards"])
        for i in range(obs.shape[0]):
            nxt = new_obs[i]
            term = False
            if dones[i]:
                nxt = infos[i]["terminal_observation"]
                term = not bool(infos[i].get("TimeLimit.truncated", False))
            j = self._rb_pos
            self._rb["obs"][j], self._rb["next"][j] = obs[i], np.asarray(nxt, np.float32)
            self._rb["act"][j], self._rb["rew"][j] = act[i], float(raw[i])
            self._rb["term"][j] = float(term)
            self._rb_pos = (j + 1) % self._rb_cap
            self._rb_full = self._rb_full or self._rb_pos == 0
        t = int(self.num_timesteps)
        self.policy.eps = self.epsilon_at(t)
        if t >= self.onoff.critic_learning_starts:
            self.critic_train_step()

    def critic_target_values(self, rew_norm: th.Tensor, nxt: th.Tensor, term: th.Tensor) -> th.Tensor:
        pol = self.policy
        with th.no_grad():
            b_idx = pol.pattern_scores(pol.critic, nxt).argmax(dim=1)       # online picks b'
            b2 = pol.patterns[b_idx]
            c2 = th.clamp(pol.mean_c(nxt, b2), -1.0, 1.0)
            q2 = th.cat(pol.critic_target(nxt, pol.full_action(c2, b2)), dim=1).min(dim=1).values
            return rew_norm + (1.0 - term) * self.gamma * q2

    def critic_train_step(self, idx: np.ndarray | None = None) -> float:
        pol = self.policy
        n = self.rb_size
        if n < 1:
            return float("nan")
        if idx is None:
            idx = self._replay_rng.integers(0, n, size=self.onoff.critic_batch_size)
        rb = self._rb
        rew = rb["rew"][idx].astype(np.float64)
        vn = self.get_vec_normalize_env()
        if vn is not None:
            rew = np.asarray(vn.normalize_reward(rew), np.float64)
        s = th.as_tensor(rb["obs"][idx], device=self.device)
        a = th.as_tensor(rb["act"][idx], device=self.device)
        nxt = th.as_tensor(rb["next"][idx], device=self.device)
        term = th.as_tensor(rb["term"][idx], device=self.device)
        y = self.critic_target_values(th.as_tensor(rew, dtype=th.float32, device=self.device),
                                      nxt, term)
        pol.critic.set_training_mode(True)
        qs = pol.critic(s, a)
        loss = sum(F.mse_loss(q.squeeze(1), y) for q in qs)
        pol.critic_optimizer.zero_grad()
        loss.backward()
        pol.critic_optimizer.step()
        pol.critic_optimizer.zero_grad(set_to_none=True)
        pol.critic.set_training_mode(False)
        polyak_update(pol.critic.parameters(), pol.critic_target.parameters(), self.onoff.tau)
        self.critic_updates += 1
        self.critic_last_loss = float(loss.item())
        self.logger.record("onoff/critic_loss", self.critic_last_loss)
        self.logger.record("onoff/eps", pol.eps)
        return self.critic_last_loss
