"""The two settings Hybrid A fixes on top of the fair protocol.

PINNED_CLIP_RANGE
    PPO searches five knobs under the protocol; here exploration_fraction
    takes one slot, so one PPO knob is pinned. clip_range is pinned at 0.3,
    the value all 8 fair2 PPO finalists chose (decided 2026-11-09). This uses
    one fact from PPO's own search; it is stated in the report.

SWITCH_LEARNING_STARTS
    ddqn_td3's switch took its learning_starts from the TD3/DDPG pilot (100,
    the TD3Params/DDPGParams default). PPOParams.learning_starts (500) is a
    SAC field, so the value is fixed here to keep the switch identical to the
    one ddqn_td3 tuned.
"""
from __future__ import annotations

ALGO = "ddqn_ppo"
PINNED_CLIP_RANGE = 0.3
SWITCH_LEARNING_STARTS = 100
SWITCH_NET_ARCH = (64, 64)       # PPO's own MlpPolicy default, = Protocol.fixed_net_arch
