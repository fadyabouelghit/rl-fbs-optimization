"""Run directory + ledger for TD3 runs.

``TD3RunLogger`` subclasses ``ppo.run_logging.RunLogger`` without changing it:
every flat file (steps.csv, episodes.csv, eval_log.csv, run.log,
experiment_config.json) is written exactly as for a PPO run, so ppo's
callbacks, rollouts and plots read a TD3 run directory unchanged. Two things
are TD3's own: runs land under ``td3_runs/``, and the ledger row carries TD3's
hyperparameters and goes to ``td3_runs/ledger.csv``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ppo.config import ExperimentConfig
from ppo.matlab_bridge import DEFAULT_BACKEND
from ppo.paths import ensure_dir
from ppo.run_logging import RunLogger, append_ledger_row

from . import paths
from .config import TD3Experiment

# Columns of ppo's ledger row that name PPO/SAC hyperparameters. TD3 has no
# such settings, so they are dropped rather than filled with PPO defaults.
_PPO_ONLY_COLUMNS = (
    "ent_coef", "n_steps", "n_epochs", "gae_lambda", "clip_range",
    "vf_coef", "max_grad_norm", "sac_ent_coef",
)


class TD3RunLogger(RunLogger):
    def __init__(
        self,
        exp: TD3Experiment,
        runs_root: Optional[Path] = None,
        run_dir: Optional[Path] = None,
        backend_kind: str = DEFAULT_BACKEND,
        flush_every: int = 256,
    ):
        super().__init__(
            exp,
            runs_root=runs_root or paths.TD3_RUNS_DIR,
            run_dir=run_dir,
            backend_kind=backend_kind,
            flush_every=flush_every,
        )

    def ledger_row(self, status: str, metrics: Optional[dict] = None) -> dict:
        exp = self.exp
        # Scenario / band / cost-function / env columns come from ppo's own
        # ledger row, built on a PPO-shaped view of the same env, so the two
        # ledgers name those columns identically and can be joined.
        view = RunLogger.__new__(RunLogger)
        view.exp = ExperimentConfig(
            num_fbs=exp.num_fbs, scenario_id=exp.scenario_id,
            config_id=exp.config_id, env=exp.env, tag=exp.tag,
        )
        view.run_dir, view.created_at, view.backend_kind = (
            self.run_dir, self.created_at, self.backend_kind,
        )
        row = view.ledger_row(status, metrics)

        finals = {k: row.pop(k) for k in [k for k in row if k.startswith("final_")]}
        for key in _PPO_ONLY_COLUMNS:
            row.pop(key, None)
        t = exp.td3
        row.update({
            "algo": "td3",
            "normalize_reward": t.normalize_reward,
            "total_timesteps": t.total_timesteps,
            "learning_rate": t.learning_rate,
            "discount": t.gamma,                   # RL discount factor
            "batch_size": t.batch_size,
            "buffer_size": t.buffer_size,
            "learning_starts": t.learning_starts,
            "tau": t.tau,
            "train_freq": t.train_freq,
            "gradient_steps": t.gradient_steps,
            "policy_delay": t.policy_delay,
            "target_policy_noise": t.target_policy_noise,
            "target_noise_clip": t.target_noise_clip,
            "action_noise_std": t.action_noise_std,
            "binary_noise_std": "" if t.binary_noise_std is None else t.binary_noise_std,
            "net_arch": "" if t.net_arch is None else "-".join(map(str, t.net_arch)),
            "seed": t.seed,
            "n_envs": t.n_envs,
            "eval_every": t.eval_every,
            "eval_episodes": t.eval_episodes,
            "checkpoint_every": t.checkpoint_every,
        })
        row.update(finals)
        return row

    def append_ledger(self, status: str, metrics: Optional[dict] = None) -> None:
        # Unlike ppo's repo-root ledger, td3_runs/ may not exist yet when the
        # run itself was given an explicit run_dir elsewhere.
        ensure_dir(paths.TD3_LEDGER_PATH.parent)
        append_ledger_row(self.ledger_row(status, metrics), paths.TD3_LEDGER_PATH)
