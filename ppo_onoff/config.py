"""Hybrid B's own settings on top of the fair protocol.

Tuned (with PPO's ent_coef, n_steps, gae_lambda): ``exploration_fraction``
(eps horizon for b) and the critics' Polyak ``tau``.

Pinned, as section 6.6 says, at the fair2 PPO winner t35's values:
n_epochs 20 and clip_range 0.3 (also the values all 8 fair2 PPO finalists
chose). Critic lr, discount, batch and hidden layers are the protocol's
(3e-4, 0.95, 256, 64-64); critic warm-up 100 steps, the TD3 default that the
other off-policy critics in this repo used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

ALGO = "ppo_onoff"
PINNED_N_EPOCHS = 20
PINNED_CLIP_RANGE = 0.3


@dataclass(frozen=True)
class OnOffParams:
    tau: float = 5e-3
    exploration_fraction: float = 0.2
    eps_start: float = 1.0
    eps_final: float = 0.05
    critic_learning_starts: int = 100
    critic_batch_size: int = 256
    critic_buffer_size: int = 1_000_000
    critic_net_arch: tuple = (64, 64)

    def __post_init__(self):
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if not 0.0 < self.exploration_fraction <= 1.0:
            raise ValueError("exploration_fraction must be in (0, 1]")
        if not 0.0 <= self.eps_final <= self.eps_start <= 1.0:
            raise ValueError("need 0 <= eps_final <= eps_start <= 1")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["critic_net_arch"] = list(self.critic_net_arch)
        return d
