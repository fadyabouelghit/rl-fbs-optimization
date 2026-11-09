"""Hybrid A: the Double-DQN switch of ``ddqn_td3/`` with a PPO pilot.

- the **switch** is ``ddqn_td3.switch.SwitchAgent``, unchanged: on/off for every
  FBS jointly (2^N actions), epsilon-greedy, own replay memory, learning from
  the VecNormalize-normalized reward;
- the **pilot** is an unchanged SB3 PPO (``ppo.train._build_model``) that sees
  the switch's decision as N extra inputs and outputs only the 4N continuous
  genes. From PPO's side the switch is part of the environment.

``ppo/`` has no wrapper hook and is not modified: ``ddqn_ppo.train`` repeats
``ppo.train.train``'s setup for ``n_envs == 1`` and inserts ``SwitchVecEnv``
below VecNormalize. Spec: docs/RL_ARCHITECTURES_2026-11-08.md, section 6.5.
"""
from .config import ALGO, PINNED_CLIP_RANGE, SWITCH_LEARNING_STARTS

__all__ = ["ALGO", "PINNED_CLIP_RANGE", "SWITCH_LEARNING_STARTS"]
