"""Try-both settings. Shared values (lr, gamma, batch, net, budget) stay the pilot's own."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

PILOT_OF = {"on_off_td3": "td3", "on_off_ddpg": "ddpg"}
ON_OFF_ALGOS: Tuple[str, ...] = tuple(PILOT_OF)
ACTOR_PATTERNS = ("all_on", "best")


@dataclass(frozen=True)
class TryBothParams:
    """eps-greedy over on/off patterns, linear eps_start -> eps_final over
    exploration_fraction * total_timesteps (the same schedule as the DQN switch).

    actor_pattern  'all_on' (default): the actor is trained as if every station
                   were on, so an off station keeps learning placement.
                   'best': trained under the critic's best pattern (the spec's rule).
    """

    eps_start: float = 1.0
    eps_final: float = 0.05
    exploration_fraction: float = 0.2
    actor_pattern: str = "all_on"

    def __post_init__(self):
        if not 0.0 < self.exploration_fraction <= 1.0:
            raise ValueError("exploration_fraction must be in (0, 1]")
        if not 0.0 <= self.eps_final <= self.eps_start <= 1.0:
            raise ValueError("need 0 <= eps_final <= eps_start <= 1")
        if self.actor_pattern not in ACTOR_PATTERNS:
            raise ValueError(f"actor_pattern must be one of {ACTOR_PATTERNS}")

    def to_dict(self) -> dict:
        return asdict(self)
