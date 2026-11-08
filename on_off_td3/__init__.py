"""Try both, keep the better: one TD3/DDPG agent whose critic picks the on/off pattern.

    actor mu(s) -> c                         (the 4N continuous genes only)
    for every on/off pattern b in {-1,+1}^N:
        score(b) = min_i Q_i(s, [c, b])      (TD3: twin minimum; DDPG: its one critic)
    b* = argmax score  (random pattern with probability eps while training)
    env receives [c, b*]

No network outputs b. One agent, one replay memory, one critic on the full
5N action. Spec: docs/RL_ARCHITECTURES_2026-11-08.md, section 5.1 (where it is
called ``tryboth``; package and algorithm names here are ``on_off_td3`` /
``on_off_ddpg``, at Dr. Fady's request, 2026-11-09).

Deviations from the spec, decided 2026-11-09 with Dr. Fady:
- **Actor loss under "all on".** The spec trains the actor only under the
  currently best pattern b*, so a station judged better off would get almost
  no placement gradient: the circular dependence again. Here the actor loss is
  -Q1(s, [mu(s), all on]), so every station keeps learning where to go while
  it is off. ``actor_pattern="best"`` restores the spec's rule (ablation).
- **One selection rule everywhere.** Acting and the target's choice of b' both
  rank patterns by the online critics' minimum (TD3) / its critic (DDPG).
- **Exploration noise on the 4N continuous dims only**; b's exploration is eps.

Imports ``ppo/``, ``td3/``, ``ddpg/`` and modifies none of them: training goes
through the existing ``model_factory`` hook.
"""
from .config import PILOT_OF, ON_OFF_ALGOS, TryBothParams

__all__ = ["PILOT_OF", "ON_OFF_ALGOS", "TryBothParams"]
