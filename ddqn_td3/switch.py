"""The switch: a Double-DQN over the joint on/off decision.

One transition per *hold* (not per env step):

    (s, b, r_0 .. r_{k-1}, s_k, terminated)        k <= hold_k

with the raw per-step rewards kept, so the reward can be normalized at
*sample* time with VecNormalize's current running statistics -- exactly what
SB3 does for the pilot's replay buffer (``ReplayBuffer.sample(env=vec_norm)``).
Both agents therefore learn from the same reward. Target:

    R_k + gamma^k * (1 - terminated) * Q_target(s_k, argmax_a Q(s_k, a))

Own RNG streams only: the network is initialised inside ``torch.random.fork_rng``
and epsilon-greedy / replay sampling use a private ``numpy.random.Generator``,
so adding the switch never shifts the pilot's seeded streams.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .config import SwitchParams

SEED_OFFSET = 7919          # switch streams = run seed + this


def _mlp(n_in: int, hidden: Sequence[int], n_out: int) -> nn.Sequential:
    layers, last = [], n_in
    for h in hidden:
        layers += [nn.Linear(last, int(h)), nn.ReLU()]
        last = int(h)
    layers.append(nn.Linear(last, n_out))
    return nn.Sequential(*layers)


class SwitchAgent:
    def __init__(self, obs_dim: int, n_joint: int, params: SwitchParams, *,
                 gamma: float, learning_rate: float, batch_size: int,
                 net_arch: Sequence[int], total_timesteps: int, learning_starts: int,
                 seed: int = 0):
        self.params = params
        self.obs_dim, self.n_joint = int(obs_dim), int(n_joint)
        self.gamma, self.lr = float(gamma), float(learning_rate)
        self.batch_size = int(batch_size)
        self.net_arch = tuple(int(h) for h in net_arch)
        self.total_timesteps = int(total_timesteps)
        self.learning_starts = int(params.learning_starts if params.learning_starts is not None
                                   else learning_starts)
        self.seed = int(seed)

        with th.random.fork_rng(devices=[]):
            th.manual_seed(self.seed + SEED_OFFSET)
            self.q = _mlp(self.obs_dim, self.net_arch, self.n_joint)
        self.q_target = copy.deepcopy(self.q)
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.opt = th.optim.Adam(self.q.parameters(), lr=self.lr)
        self.rng = np.random.default_rng([self.seed, SEED_OFFSET])

        cap = max(1, min(int(params.buffer_size), self.total_timesteps + 1))
        K = int(params.hold_k)
        self._obs = np.zeros((cap, self.obs_dim), np.float32)
        self._next = np.zeros((cap, self.obs_dim), np.float32)
        self._act = np.zeros(cap, np.int64)
        self._rew = np.zeros((cap, K), np.float32)       # raw rewards, zero-padded
        self._klen = np.zeros(cap, np.int64)
        self._term = np.zeros(cap, np.float32)
        self._cap, self._pos, self._full = cap, 0, False
        self._gpow = self.gamma ** np.arange(K, dtype=np.float64)

        self.eps = float(params.eps_start)
        self.normalizer = None                       # VecNormalize, linked at training start
        self.n_updates = 0
        self.n_stored = 0
        self.last_loss = float("nan")

    # ------------------------------------------------------------------ #
    # acting
    # ------------------------------------------------------------------ #
    def q_values(self, obs: np.ndarray, target: bool = False) -> np.ndarray:
        net = self.q_target if target else self.q
        with th.no_grad():
            x = th.as_tensor(np.asarray(obs, np.float32).reshape(-1, self.obs_dim))
            return net(x).numpy()

    def greedy(self, obs: np.ndarray) -> int:
        return int(np.argmax(self.q_values(obs)[0]))

    def act(self, obs: np.ndarray, explore: bool = True) -> int:
        # Always draw the coin so the stream does not depend on eps.
        coin = self.rng.random()
        pick = int(self.rng.integers(self.n_joint))
        if explore and coin < self.eps:
            return pick
        return self.greedy(obs)

    def epsilon_at(self, t: int) -> float:
        p = self.params
        horizon = max(1.0, p.exploration_fraction * self.total_timesteps)
        frac = min(1.0, t / horizon)
        return float(p.eps_start + frac * (p.eps_final - p.eps_start))

    # ------------------------------------------------------------------ #
    # memory
    # ------------------------------------------------------------------ #
    @property
    def size(self) -> int:
        return self._cap if self._full else self._pos

    def store(self, obs, joint: int, rewards: Sequence[float], next_obs, terminated: bool) -> None:
        k = len(rewards)
        if not 1 <= k <= self.params.hold_k:
            raise ValueError(f"hold of length {k} with hold_k={self.params.hold_k}")
        i = self._pos
        self._obs[i] = np.asarray(obs, np.float32).reshape(-1)
        self._next[i] = np.asarray(next_obs, np.float32).reshape(-1)
        self._act[i] = int(joint)
        self._rew[i] = 0.0
        self._rew[i, :k] = np.asarray(rewards, np.float32)
        self._klen[i] = k
        self._term[i] = float(bool(terminated))
        self._pos = (i + 1) % self._cap
        self._full = self._full or self._pos == 0
        self.n_stored += 1

    def discounted_return(self, raw: np.ndarray, klen: np.ndarray) -> np.ndarray:
        """sum_i gamma^i * norm(r_i) over each hold, norm = VecNormalize's (if linked)."""
        r = np.asarray(raw, np.float64)
        if self.normalizer is not None:
            r = np.asarray(self.normalizer.normalize_reward(r), np.float64)
        mask = np.arange(r.shape[1])[None, :] < np.asarray(klen)[:, None]
        return (r * mask * self._gpow[None, : r.shape[1]]).sum(axis=1)

    # ------------------------------------------------------------------ #
    # learning
    # ------------------------------------------------------------------ #
    def update(self) -> Optional[float]:
        n = self.size
        if n < self.batch_size:
            return None
        idx = self.rng.integers(0, n, size=self.batch_size)
        obs = th.as_tensor(self._obs[idx])
        nxt = th.as_tensor(self._next[idx])
        act = th.as_tensor(self._act[idx])
        klen = self._klen[idx]
        R = th.as_tensor(self.discounted_return(self._rew[idx], klen), dtype=th.float32)
        disc = th.as_tensor(self.gamma ** klen.astype(np.float64), dtype=th.float32)
        term = th.as_tensor(self._term[idx])
        with th.no_grad():
            q_next_t = self.q_target(nxt)
            if self.params.double_q:
                a_star = self.q(nxt).argmax(dim=1, keepdim=True)
                v_next = q_next_t.gather(1, a_star).squeeze(1)
            else:
                v_next = q_next_t.max(dim=1).values
            target = R + (1.0 - term) * disc * v_next
        q = self.q(obs).gather(1, act[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.params.max_grad_norm)
        self.opt.step()
        self.n_updates += 1
        self.last_loss = float(loss.item())
        return self.last_loss

    def on_step(self, t: int) -> None:
        """Called once per env step (after the step): schedule, update, target sync."""
        self.eps = self.epsilon_at(t)
        if t >= self.learning_starts:
            self.update()
        if t % self.params.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Path | str, meta: Optional[dict] = None) -> None:
        th.save({"q": self.q.state_dict(), "q_target": self.q_target.state_dict(),
                 "obs_dim": self.obs_dim, "n_joint": self.n_joint,
                 "net_arch": list(self.net_arch), "params": self.params.to_dict(),
                 "eps": self.eps, "n_updates": self.n_updates, "n_stored": self.n_stored,
                 "meta": dict(meta or {})}, str(path))

    @staticmethod
    def load_net(path: Path | str) -> tuple[nn.Module, dict]:
        """(greedy Q-network, saved blob) -- for evaluation only."""
        blob = th.load(str(path), map_location="cpu", weights_only=False)
        net = _mlp(int(blob["obs_dim"]), blob["net_arch"], int(blob["n_joint"]))
        net.load_state_dict(blob["q"])
        net.eval()
        return net, blob
