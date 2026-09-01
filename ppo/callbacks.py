"""SB3 callbacks wiring env metrics into the run's flat files."""
from __future__ import annotations

from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .run_logging import RunLogger


class EnvInfoLoggingCallback(BaseCallback):
    """Streams every step's info dict into steps.csv and per-episode
    aggregates into episodes.csv (buffered — no per-step disk I/O).

    The env packs its flat state vector under ``info["state"]``; it is
    expanded into one column per gene using ``state_labels`` so full training
    trajectories can be reconstructed from steps.csv alone.
    """

    SKIP_KEYS = {"user_positions", "state", "episode", "terminal_observation", "TimeLimit.truncated"}

    def __init__(self, run_logger: RunLogger, state_labels: list[str], verbose: int = 0):
        super().__init__(verbose)
        self.run_logger = run_logger
        self.state_labels = state_labels
        self.last_episode: dict = {}
        self._episode_idx: Optional[np.ndarray] = None
        self._episode_step: Optional[np.ndarray] = None
        self._episode_reward: Optional[np.ndarray] = None

    def _init_buffers(self, n_envs: int) -> None:
        self._episode_idx = np.zeros(n_envs, dtype=int)
        self._episode_step = np.zeros(n_envs, dtype=int)
        self._episode_reward = np.zeros(n_envs, dtype=float)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = np.asarray(self.locals.get("rewards", []), dtype=float).reshape(-1)
        dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)
        if self._episode_idx is None:
            self._init_buffers(len(infos))

        for i, info in enumerate(infos):
            self._episode_step[i] += 1
            reward = float(rewards[i]) if i < len(rewards) else float("nan")
            self._episode_reward[i] += reward

            row = {
                "timesteps": self.num_timesteps,
                "env": i,
                "episode": int(self._episode_idx[i]),
                "step": int(self._episode_step[i]),
                "reward": reward,
            }
            for key, value in info.items():
                if key in self.SKIP_KEYS:
                    continue
                if isinstance(value, (int, float, np.integer, np.floating, bool)):
                    row[key] = value
            state = info.get("state")
            if state is not None:
                state = np.asarray(state, dtype=float).reshape(-1)
                for label, value in zip(self.state_labels, state):
                    row[label] = value
            self.run_logger.steps.write(row)

            if i < len(dones) and dones[i]:
                ep_row = {
                    "timesteps": self.num_timesteps,
                    "env": i,
                    "episode": int(self._episode_idx[i]),
                    "length": int(self._episode_step[i]),
                    "reward_sum": float(self._episode_reward[i]),
                    "reward_mean": float(self._episode_reward[i] / max(self._episode_step[i], 1)),
                }
                for key in (
                    "total_connected", "fbs_connected", "mbs_connected",
                    "mbs_coverage_connected", "mbs_capacity_connected",
                    "total_power", "avg_rate", "sum_rate", "num_active_fbs",
                ):
                    if key in info:
                        ep_row[f"final_{key}"] = info[key]
                monitor_ep = info.get("episode")
                if monitor_ep:
                    ep_row["monitor_r"] = monitor_ep.get("r")
                    ep_row["monitor_t"] = monitor_ep.get("t")
                self.run_logger.episodes.write(ep_row)
                self.last_episode = ep_row
                self._episode_idx[i] += 1
                self._episode_step[i] = 0
                self._episode_reward[i] = 0.0
        return True

    def _on_training_end(self) -> None:
        self.run_logger.steps.flush()
        self.run_logger.episodes.flush()


class CheckpointEveryCallback(BaseCallback):
    """Saves the model every ``every`` timesteps into ``checkpoints/``."""

    def __init__(self, run_logger: RunLogger, every: int, verbose: int = 0):
        super().__init__(verbose)
        self.run_logger = run_logger
        self.every = int(every)
        self._last_saved = 0

    def _on_step(self) -> bool:
        if self.every > 0 and self.num_timesteps - self._last_saved >= self.every:
            path = self.run_logger.run_dir / "checkpoints" / f"model_{self.num_timesteps:08d}"
            self.model.save(str(path))
            self.run_logger.log.info("checkpoint saved at %s steps -> %s", self.num_timesteps, path.name)
            self._last_saved = self.num_timesteps
        return True


class CsvEvalCallback(BaseCallback):
    """Runs deterministic evaluation episodes every ``every`` timesteps on a
    dedicated eval env, logs one row per evaluation to eval_log.csv, and keeps
    the best-mean-reward model as ``best_model.zip``.

    The eval env may share the training env's MATLAB backend (calls are
    strictly sequential), so no second engine is needed for n_envs == 1.
    """

    def __init__(
        self,
        run_logger: RunLogger,
        eval_env,
        every: int,
        n_episodes: int = 3,
        eval_seed: int = 12345,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.run_logger = run_logger
        self.eval_env = eval_env
        self.every = int(every)
        self.n_episodes = int(n_episodes)
        self.eval_seed = int(eval_seed)
        self._last_eval = 0
        self.best_mean_reward = -np.inf

    def _run_episode(self, episode_idx: int) -> dict:
        # Seeded reset: every evaluation round replays the same start states,
        # so eval curves are comparable across training and across runs.
        obs, _ = self.eval_env.reset(seed=self.eval_seed + episode_idx)
        total_reward, steps, last_info = 0.0, 0, {}
        objectives: list[float] = []
        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = self.eval_env.step(action)
            total_reward += float(reward)
            steps += 1
            last_info = info
            if info.get("reward_objective") is not None:
                objectives.append(float(info["reward_objective"]))
            if term or trunc:
                break
        return {
            "reward_sum": total_reward,
            "length": steps,
            # Absolute quality, independent of reward mode/scaling — this is
            # the cross-run comparable number. Episode returns are NOT comparable
            # across shaping modes, so never rank runs by reward_sum.
            "objective": objectives[-1] if objectives else None,
            "best_objective": max(objectives) if objectives else None,
            **{
                k: last_info.get(k) for k in (
                    "total_connected", "fbs_connected", "mbs_connected",
                    "total_power", "avg_rate",
                )
            },
        }

    def _on_step(self) -> bool:
        if self.every <= 0 or self.num_timesteps - self._last_eval < self.every:
            return True
        self._last_eval = self.num_timesteps

        episodes = [self._run_episode(k) for k in range(self.n_episodes)]
        mean_reward = float(np.mean([e["reward_sum"] for e in episodes]))
        objs = [e["objective"] for e in episodes if e["objective"] is not None]
        best_objs = [e["best_objective"] for e in episodes if e["best_objective"] is not None]
        mean_objective = float(np.mean(objs)) if objs else None
        # Rank by absolute objective, not by the shaped return: under
        # potential/record shaping the return measures *improvement*, so a
        # policy that climbs far from a bad start would outrank a better one.
        score = mean_objective if mean_objective is not None else mean_reward
        row = {
            "timesteps": self.num_timesteps,
            "n_episodes": self.n_episodes,
            "mean_reward": mean_reward,
            "std_reward": float(np.std([e["reward_sum"] for e in episodes])),
            "mean_objective": mean_objective,
            "mean_best_objective": float(np.mean(best_objs)) if best_objs else None,
            "mean_length": float(np.mean([e["length"] for e in episodes])),
            "mean_total_connected": float(np.mean([e.get("total_connected") or 0 for e in episodes])),
            "mean_fbs_connected": float(np.mean([e.get("fbs_connected") or 0 for e in episodes])),
            "mean_total_power": float(np.mean([e.get("total_power") or 0.0 for e in episodes])),
            "best_so_far": score > self.best_mean_reward,
        }
        self.run_logger.eval_log.write(row)
        self.run_logger.log.info(
            "eval @ %s steps: objective=%s reward=%.4f connected=%.1f fbs=%.1f",
            self.num_timesteps,
            "n/a" if mean_objective is None else f"{mean_objective:.4f}",
            mean_reward, row["mean_total_connected"], row["mean_fbs_connected"],
        )
        if score > self.best_mean_reward:
            self.best_mean_reward = score
            self.model.save(str(self.run_logger.run_dir / "best_model"))
        return True
