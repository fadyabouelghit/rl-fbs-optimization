"""``SwitchPilotPolicy``: a saved switch + pilot pair acting on a plain env.

    policy = SwitchPilotPolicy.load(run_dir, which="best")
    policy.reset()                       # at every episode start
    action, _ = policy.predict(obs)      # full env action [c, b]

The switch decides greedily at steps 0, K, 2K, ... of the episode; the pilot
acts deterministically. ``reset()`` restarts the hold clock. Callers that
never call it stay aligned anyway as long as the episode length is a multiple
of K (40 steps under the protocol, K in {1, 4, 8}).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch as th

from .config import Layout
from .switch import SwitchAgent

_FILES = {"best": ("best_model.zip", "best_switch.pt"), "final": ("model.zip", "switch.pt")}


def switch_meta(cfg: dict) -> dict:
    """The ``ddqn_switch`` block the pilot's logger wrote into experiment_config.json."""
    try:
        return cfg["provenance"]["ddqn_switch"]
    except KeyError:
        raise ValueError("not a ddqn_td3 run: experiment_config.json has no "
                         "provenance.ddqn_switch block") from None


class SwitchPilotPolicy:
    def __init__(self, pilot_model, q_net, layout: Layout, hold_k: int):
        self.pilot, self.q_net, self.layout = pilot_model, q_net, layout
        self.hold_k = int(hold_k)
        self._t = 0
        self.b = layout.bits(0)

    @classmethod
    def load(cls, run_dir: Path | str, which: str = "best") -> "SwitchPilotPolicy":
        run_dir = Path(run_dir)
        if which not in _FILES:
            raise ValueError(f"which must be 'best' or 'final', got {which!r}")
        cfg = json.loads((run_dir / "experiment_config.json").read_text())
        meta = switch_meta(cfg)
        pilot = meta["pilot"]
        if pilot == "td3":
            from stable_baselines3 import TD3 as Algo
            from td3.config import td3_experiment_from_json_dict as from_json
        else:
            from stable_baselines3 import DDPG as Algo
            from ddpg.config import ddpg_experiment_from_json_dict as from_json
        exp = from_json(cfg)
        model_file, switch_file = _FILES[which]
        model = Algo.load(str(run_dir / model_file), device="cpu")
        q_net, _ = SwitchAgent.load_net(run_dir / switch_file)
        return cls(model, q_net, Layout(exp.env), int(meta["params"]["hold_k"]))

    def reset(self) -> None:
        self._t = 0

    def predict(self, obs, state=None, episode_start=None, deterministic: bool = True):
        if episode_start is not None and bool(np.asarray(episode_start).reshape(-1)[0]):
            self.reset()
        obs = np.asarray(obs, np.float32).reshape(-1)
        if self._t % self.hold_k == 0:
            with th.no_grad():
                joint = int(self.q_net(th.as_tensor(obs[None, :])).argmax(dim=1)[0])
            self.b = self.layout.bits(joint)
        c, _ = self.pilot.predict(self.layout.augment(obs, self.b), deterministic=deterministic)
        self._t += 1
        return self.layout.full_action(c, self.b), None
