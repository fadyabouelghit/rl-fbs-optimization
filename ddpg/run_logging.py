"""Run directory + ledger for DDPG runs.

``DDPGRunLogger`` subclasses ``ppo.run_logging.RunLogger`` without changing
it: every flat file (steps.csv, episodes.csv, eval_log.csv, run.log,
experiment_config.json) is written exactly as for a PPO run, so ppo's
callbacks, rollouts and plots read a DDPG run directory unchanged. Two things
are DDPG's own: runs land under ``ddpg_runs/``, and the ledger row carries
DDPG's hyperparameters and goes to ``ddpg_runs/ledger.csv``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ppo.config import ExperimentConfig
from ppo.matlab_bridge import DEFAULT_BACKEND
from ppo.paths import ensure_dir
from ppo.run_logging import RunLogger, append_ledger_row

from . import paths
from .config import DDPGExperiment

# Columns of ppo's ledger row that name PPO/SAC hyperparameters. DDPG has no
# such settings, so they are dropped rather than filled with PPO defaults.
_PPO_ONLY_COLUMNS = (
    "ent_coef", "n_steps", "n_epochs", "gae_lambda", "clip_range",
    "vf_coef", "max_grad_norm", "sac_ent_coef",
)


class DDPGRunLogger(RunLogger):
    def __init__(
        self,
        exp: DDPGExperiment,
        runs_root: Optional[Path] = None,
        run_dir: Optional[Path] = None,
        backend_kind: str = DEFAULT_BACKEND,
        flush_every: int = 256,
    ):
        super().__init__(
            exp,
            runs_root=runs_root or paths.DDPG_RUNS_DIR,
            run_dir=run_dir,
            backend_kind=backend_kind,
            flush_every=flush_every,
        )

    def ledger_row(self, status: str, metrics: Optional[dict] = None) -> dict:
        exp = self.exp
        # Scenario / band / cost-function / env columns come from ppo's own
        # ledger row, built on a PPO-shaped view of the same env, so all three
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
        d = exp.ddpg
        row.update({
            "algo": "ddpg",
            "normalize_reward": d.normalize_reward,
            "total_timesteps": d.total_timesteps,
            "learning_rate": d.learning_rate,
            "discount": d.gamma,                   # RL discount factor
            "batch_size": d.batch_size,
            "buffer_size": d.buffer_size,
            "learning_starts": d.learning_starts,
            "tau": d.tau,
            "train_freq": d.train_freq,
            "gradient_steps": d.gradient_steps,
            "noise_type": d.noise_type,
            "action_noise_std": d.action_noise_std,
            "binary_noise_std": "" if d.binary_noise_std is None else d.binary_noise_std,
            "ou_theta": d.ou_theta,
            "ou_dt": d.ou_dt,
            "net_arch": "" if d.net_arch is None else "-".join(map(str, d.net_arch)),
            "seed": d.seed,
            "n_envs": d.n_envs,
            "eval_every": d.eval_every,
            "eval_episodes": d.eval_episodes,
            "checkpoint_every": d.checkpoint_every,
        })
        row.update(finals)
        return row

    def append_ledger(self, status: str, metrics: Optional[dict] = None) -> None:
        # Unlike ppo's repo-root ledger, ddpg_runs/ may not exist yet when the
        # run itself was given an explicit run_dir elsewhere.
        ensure_dir(paths.DDPG_LEDGER_PATH.parent)
        append_ledger_row(self.ledger_row(status, metrics), paths.DDPG_LEDGER_PATH)
