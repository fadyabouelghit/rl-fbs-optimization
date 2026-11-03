"""Where the switch sits, and how the pilot sees it.

Training stack, inside out::

    FlyingBaseStationEnv -> Monitor -> DummyVecEnv -> SwitchVecEnv -> VecNormalize -> pilot

``SwitchVecEnv`` is a VecEnv-level wrapper. It must sit *below* VecNormalize,
not above it: SB3's off-policy ``_store_transition`` reads the observations it
puts in the replay buffer from ``VecNormalize.get_original_obs()``, so a wrapper
above VecNormalize would hand the pilot's buffer observations without the
switch's ``b``. Below VecNormalize the switch sees raw rewards, which it
stores raw and normalizes at sample time with VecNormalize's own running
statistics (``SwitchAgent.discounted_return``) -- the same thing SB3 does for
the pilot. Both agents learn from the same normalized reward.

The pilot's spaces: observation = env obs + N dims (b as +-1), action = the
4N continuous genes. The env always receives the full action [c, b]; b is
+-1, so the sticky rule (|a| >= 0.5) always applies it.

``SwitchEvalWrapper`` is the gym-level counterpart for the in-training eval
env: greedy switch, same hold schedule, never stores or learns.
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper

from .config import Layout
from .switch import SwitchAgent


def _pilot_spaces(obs_space: gym.spaces.Box, layout: Layout):
    lo = np.concatenate([obs_space.low.reshape(-1), -np.ones(layout.n_fbs, np.float32)])
    hi = np.concatenate([obs_space.high.reshape(-1), np.ones(layout.n_fbs, np.float32)])
    obs = gym.spaces.Box(low=lo.astype(np.float32), high=hi.astype(np.float32), dtype=np.float32)
    act = gym.spaces.Box(low=-1.0, high=1.0, shape=(layout.n_cont,), dtype=np.float32)
    return obs, act


class SwitchVecEnv(VecEnvWrapper):
    """Holds the switch's decision, applies it, and records one transition per hold."""

    def __init__(self, venv, agent: SwitchAgent, layout: Layout):
        if venv.num_envs != 1:
            raise ValueError("ddqn_td3 needs n_envs == 1: one switch, one stream of holds")
        obs_space, act_space = _pilot_spaces(venv.observation_space, layout)
        super().__init__(venv, observation_space=obs_space, action_space=act_space)
        self.agent, self.layout = agent, layout
        self.hold_k = int(agent.params.hold_k)
        self.joint = 0
        self.b = layout.bits(0)
        self._hold_obs: Optional[np.ndarray] = None
        self._hold_rewards: list = []
        self.last_full_action = np.zeros(layout.n_full, np.float32)
        self.n_decisions = 0

    def _decide(self, obs: np.ndarray) -> None:
        self.joint = self.agent.act(obs, explore=True)
        self.b = self.layout.bits(self.joint)
        self._hold_obs = np.asarray(obs, np.float32).copy()
        self._hold_rewards = []
        self.n_decisions += 1

    def _aug(self, obs: np.ndarray) -> np.ndarray:
        return self.layout.augment(obs[0], self.b)[None, :]

    def reset(self):
        obs = self.venv.reset()
        self._decide(obs[0])
        return self._aug(obs)

    def step_async(self, actions: np.ndarray) -> None:
        full = self.layout.full_action(np.asarray(actions)[0], self.b)
        self.last_full_action = full
        self.venv.step_async(full[None, :])

    def step_wait(self):
        obs, rew, dones, infos = self.venv.step_wait()
        info = infos[0]
        b_exec, joint_exec = self.b.copy(), self.joint
        self._hold_rewards.append(float(rew[0]))
        decided = False
        if dones[0]:
            term_obs = np.asarray(info["terminal_observation"], np.float32)
            terminated = not bool(info.get("TimeLimit.truncated", False))
            self.agent.store(self._hold_obs, self.joint, self._hold_rewards, term_obs, terminated)
            # the pilot's terminal next-obs carries the b it was acting under
            info["terminal_observation"] = self.layout.augment(term_obs, b_exec)
            self._decide(obs[0])           # obs is already the reset observation
            decided = True
        elif len(self._hold_rewards) >= self.hold_k:
            self.agent.store(self._hold_obs, self.joint, self._hold_rewards, obs[0], False)
            self._decide(obs[0])
            decided = True
        info["switch_b_executed"] = b_exec
        info["switch_joint_executed"] = int(joint_exec)
        info["switch_b"] = self.b.copy()
        info["switch_eps"] = float(self.agent.eps)
        info["switch_decided"] = decided
        return self._aug(obs), rew, dones, infos


class SwitchEvalWrapper(gym.Wrapper):
    """Greedy switch on a plain env, for CsvEvalCallback. Never stores, never learns."""

    def __init__(self, env: gym.Env, agent: SwitchAgent, layout: Layout):
        super().__init__(env)
        self.observation_space, self.action_space = _pilot_spaces(env.observation_space, layout)
        self.agent, self.layout = agent, layout
        self.hold_k = int(agent.params.hold_k)
        self._t = 0
        self.b = layout.bits(0)

    def _decide(self, obs) -> None:
        self.b = self.layout.bits(self.agent.greedy(obs))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._t = 0
        self._decide(obs)
        return self.layout.augment(obs, self.b), info

    def step(self, action):
        full = self.layout.full_action(action, self.b)
        obs, reward, term, trunc, info = self.env.step(full)
        self._t += 1
        if self._t % self.hold_k == 0 and not (term or trunc):
            self._decide(obs)
        return self.layout.augment(obs, self.b), reward, term, trunc, info
