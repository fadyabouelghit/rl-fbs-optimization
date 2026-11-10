"""Hybrid B: twin-critic on/off selection + PPO placement.

For every on/off pattern b, the PPO actor's mean placement c_b = mu(s, b) is
scored by TD3-style twin critics, min(Q1, Q2)(s, [c_b, b]); the best pattern
b* is executed (a random one with probability eps in training) and PPO samples
c ~ pi(. | s, b*). PPO learns on-policy with its own value net and GAE; the
critics learn off-policy from a replay memory of the same steps. The two meet
only at action selection: no PPO gradient reaches the critics and no critic
gradient reaches the actor.

Spec: docs/RL_ARCHITECTURES_2026-11-08.md, section 6.6. ``ppo/`` is imported,
never modified. Pinned to stable-baselines3 2.9.0.
"""
from .config import ALGO, PINNED_CLIP_RANGE, PINNED_N_EPOCHS, OnOffParams

__all__ = ["ALGO", "PINNED_CLIP_RANGE", "PINNED_N_EPOCHS", "OnOffParams"]
