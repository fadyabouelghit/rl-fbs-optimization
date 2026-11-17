"""Switch hyperparameters and the action/observation layout.

Shared values (learning rate, discount, batch size, hidden layers, seed,
budget) are NOT fields here: they are read from the pilot's own params, so the
switch can never drift away from the protocol's pinned values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np

# Tuner / letterexp algorithm names -> pilot package.
PILOT_OF = {"ddqn_td3": "td3", "ddqn_ddpg": "ddpg"}
DDQN_ALGOS: Tuple[str, ...] = tuple(PILOT_OF)


@dataclass(frozen=True)
class SwitchParams:
    """Double-DQN switch settings.

    hold_k               decide every K env steps (a design arm, not a tuned knob)
    eps_start/eps_final  epsilon-greedy schedule, linear over
                         exploration_fraction * total_timesteps (SB3 DQN style)
    target_update        hard copy of the online net every this many env steps
    double_q             Double-DQN target (online argmax, target value)
    buffer_size          capped at the run's total_timesteps (one decision per
                         step at most, so a bigger buffer is never filled)
    learning_starts      None -> the pilot's learning_starts
    max_grad_norm        SB3 DQN default
    """

    hold_k: int = 1
    eps_start: float = 1.0
    eps_final: float = 0.05
    exploration_fraction: float = 0.2
    target_update: int = 1000
    double_q: bool = True
    buffer_size: int = 1_000_000
    learning_starts: Optional[int] = None
    max_grad_norm: float = 10.0

    def __post_init__(self):
        if self.hold_k < 1:
            raise ValueError("hold_k must be >= 1")
        if not 0.0 < self.exploration_fraction <= 1.0:
            raise ValueError("exploration_fraction must be in (0, 1]")
        if not 0.0 <= self.eps_final <= self.eps_start <= 1.0:
            raise ValueError("need 0 <= eps_final <= eps_start <= 1")
        if self.target_update < 1 or self.buffer_size < 1:
            raise ValueError("target_update and buffer_size must be >= 1")

    def to_dict(self) -> dict:
        return asdict(self)


class Layout:
    """Where the continuous and on/off genes sit in the env's action vector.

    ``ppo.env.FlyingBaseStationEnv`` packs 5 genes per FBS: x, y, z, power
    (continuous) then the power on/off switch. The band flag and MBS capacity
    toggles would add more binary dims; the switch does not cover them, so an
    env that enables them is refused rather than half-handled.
    """

    GENES = 5

    def __init__(self, env_cfg):
        if env_cfg.band.agent_controls_fbs_band or env_cfg.band.agent_controls_mbs_capacity:
            raise ValueError("ddqn_td3 supports the on/off gene only "
                             "(no agent-controlled band flag or MBS capacity)")
        self.n_fbs = int(env_cfg.num_fbs)
        if self.n_fbs > 3:
            raise ValueError("joint 2^N switch outputs: N <= 3 (use a branching head beyond)")
        self.n_joint = 2 ** self.n_fbs
        self.n_cont = 4 * self.n_fbs
        self.n_full = self.GENES * self.n_fbs
        self.onoff_idx: List[int] = [f * self.GENES + 4 for f in range(self.n_fbs)]
        self.cont_idx: List[int] = [f * self.GENES + g for f in range(self.n_fbs) for g in range(4)]

    def bits(self, joint: int) -> np.ndarray:
        """Joint action index -> b in {-1, +1}^N (bit j set = FBS j on)."""
        return np.array([1.0 if (int(joint) >> j) & 1 else -1.0 for j in range(self.n_fbs)],
                        dtype=np.float32)

    def on_mask(self, j: int) -> np.ndarray:
        """Boolean mask over joint actions: True where FBS j is on."""
        return np.array([(a >> j) & 1 == 1 for a in range(self.n_joint)])

    def full_action(self, c: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.zeros(self.n_full, dtype=np.float32)
        a[self.cont_idx] = np.asarray(c, dtype=np.float32).reshape(-1)
        a[self.onoff_idx] = b
        return a

    @staticmethod
    def augment(obs: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(obs, np.float32).reshape(-1), b]).astype(np.float32)
