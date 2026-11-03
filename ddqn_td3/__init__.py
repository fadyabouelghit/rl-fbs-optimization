"""Two-agent remedy for the on/off decision: a Double-DQN switch + a TD3/DDPG pilot.

- the **switch** chooses on/off for every FBS jointly (2^N actions) by
  comparing Q-values, so no on/off gradient exists at all, and can hold its
  choice for ``hold_k`` steps;
- the **pilot** (TD3 or DDPG, trained by ``td3.train`` / ``ddpg.train``
  unchanged) chooses x, y, height and power given the switch's choice.

Imports ``ppo/``, ``td3/`` and ``ddpg/`` and modifies none of them; the only
contact points are the opt-in ``vec_env_wrapper`` / ``eval_env_wrapper`` /
``extra_callbacks`` / ``config_extra`` hooks of the pilot's ``train``.

See ``prompt_switchpilot.md`` for the brief, and ``ddqn_td3/wrapper.py`` for
where the switch sits relative to VecNormalize and why.
"""
from .config import DDQN_ALGOS, PILOT_OF, SwitchParams

__all__ = ["DDQN_ALGOS", "PILOT_OF", "SwitchParams"]
