"""Run-directory management and efficient flat-file logging.

Per-run directory layout (GA-style naming: ``ppo_runs/run_<ts>_<label>/``):

    experiment_config.json   full resolved config + provenance (git sha, ...)
    model.zip                final SB3 model
    best_model.zip           best in-training eval model (when eval enabled)
    checkpoints/             periodic model snapshots
    steps.csv                per-step env metrics + state (buffered writes)
    episodes.csv             per-episode aggregates
    progress.csv             SB3 optimizer diagnostics (losses, kl, fps, ...)
    monitor*.csv             SB3 Monitor episode log (one per worker)
    eval_log.csv             in-training evaluation results
    run.log                  human-readable event log
    evals/<ts>/              post-hoc evaluation artifacts (ppo.evaluate)

A one-row-per-run ledger is appended to ``training_log.csv`` at the repo
root, mirroring the GA study ledgers.
"""
from __future__ import annotations

import csv
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import ExperimentConfig
from .paths import LEDGER_PATH, REPO_ROOT, RUNS_DIR, ensure_dir


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def git_sha() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def package_versions() -> dict:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("numpy", "gymnasium", "stable_baselines3", "torch", "pandas"):
        try:
            versions[pkg] = __import__(pkg).__version__
        except Exception:
            versions[pkg] = None
    return versions


# --------------------------------------------------------------------------- #
# Buffered CSV writer
# --------------------------------------------------------------------------- #
class BufferedCsvWriter:
    """Append-only CSV writer that batches rows before touching the disk.

    The header is fixed by the first row; later rows are aligned to it
    (missing keys -> empty, unknown keys -> dropped) so occasional info-dict
    variations don't corrupt the file.
    """

    def __init__(self, path: Path, flush_every: int = 256):
        self.path = Path(path)
        self.flush_every = max(int(flush_every), 1)
        self._buffer: list[dict] = []
        self._fieldnames: Optional[list[str]] = None
        self._file = None
        self._writer = None

    def write(self, row: dict) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        if self._writer is None:
            self._fieldnames = list(self._buffer[0].keys())
            new_file = not self.path.exists() or self.path.stat().st_size == 0
            self._file = self.path.open("a", newline="")
            self._writer = csv.DictWriter(
                self._file, fieldnames=self._fieldnames, extrasaction="ignore"
            )
            if new_file:
                self._writer.writeheader()
        for row in self._buffer:
            self._writer.writerow({k: row.get(k, "") for k in self._fieldnames})
        self._buffer.clear()
        self._file.flush()

    def close(self) -> None:
        self.flush()
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


# --------------------------------------------------------------------------- #
# Ledger append (header-widening)
# --------------------------------------------------------------------------- #
def append_ledger_row(row: dict, path: Path) -> list[str]:
    """Append one row, widening the header if the row carries new columns.

    The previous implementation pinned the header to whatever the file already
    had and wrote with ``extrasaction='ignore'`` — so any setting added after
    the ledger was first created was silently dropped. That is how ``algo``,
    ``reward_shaping``, ``action_mode`` and the RL ``discount`` went unlogged.

    Here, new keys trigger a rewrite of the file with the union of columns
    (existing rows get blanks for the new ones), preceded by a ``.bak`` copy.
    Returns the field list actually written.
    """
    path = Path(path)
    existing_fields: list[str] = []
    existing_rows: list[dict] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = list(reader)

    new_keys = [k for k in row if k not in existing_fields]
    if existing_fields and new_keys:
        fields = existing_fields + new_keys
        path.with_suffix(path.suffix + ".bak").write_bytes(path.read_bytes())
        with path.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            wr.writeheader()
            for old in existing_rows:
                wr.writerow({k: old.get(k, "") for k in fields})
            wr.writerow({k: row.get(k, "") for k in fields})
        return fields

    fields = existing_fields or list(row.keys())
    with path.open("a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not existing_fields:
            wr.writeheader()
        wr.writerow({k: row.get(k, "") for k in fields})
    return fields


# --------------------------------------------------------------------------- #
# Run logger
# --------------------------------------------------------------------------- #
class RunLogger:
    """Owns one run directory and every flat file written into it."""

    def __init__(
        self,
        exp: ExperimentConfig,
        runs_root: Path = RUNS_DIR,
        run_dir: Optional[Path] = None,
        backend_kind: str = "matlab",
        flush_every: int = 256,
    ):
        self.exp = exp
        self.created_at = timestamp()
        if run_dir is None:
            run_dir = ensure_dir(runs_root) / f"run_{self.created_at}_{exp.label}"
        self.run_dir = ensure_dir(Path(run_dir))
        ensure_dir(self.run_dir / "checkpoints")
        self.backend_kind = backend_kind

        self.steps = BufferedCsvWriter(self.run_dir / "steps.csv", flush_every)
        self.episodes = BufferedCsvWriter(self.run_dir / "episodes.csv", flush_every=8)
        self.eval_log = BufferedCsvWriter(self.run_dir / "eval_log.csv", flush_every=1)

        self.log = logging.getLogger(f"ppo.run.{self.run_dir.name}")
        self.log.setLevel(logging.INFO)
        self.log.propagate = False
        handler = logging.FileHandler(self.run_dir / "run.log")
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s")
        )
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        self.log.handlers = [handler, console]

    # ------------------------------------------------------------------ #
    def write_config(self, extra: Optional[dict] = None) -> None:
        payload = self.exp.to_json_dict()
        payload["provenance"] = {
            "created_at": self.created_at,
            "git_sha": git_sha(),
            "backend": self.backend_kind,
            "versions": package_versions(),
            **(extra or {}),
        }
        (self.run_dir / "experiment_config.json").write_text(
            json.dumps(payload, indent=2, default=str)
        )

    def update_config(self, **fields) -> None:
        path = self.run_dir / "experiment_config.json"
        data = json.loads(path.read_text()) if path.exists() else {}
        data.setdefault("provenance", {}).update(fields)
        path.write_text(json.dumps(data, indent=2, default=str))

    # ------------------------------------------------------------------ #
    def ledger_row(self, status: str, metrics: Optional[dict] = None) -> dict:
        """One flat, complete record of what ran.

        Every setting that can change a result appears here. Column names are
        kept stable for the pre-existing ones (so older analysis scripts keep
        working); new settings are appended and the header is widened on
        write, so a field added later can never be silently dropped.

        Note ``gamma`` is the *cost-function* exponent (reward weight) for
        historical reasons; the RL discount factor is ``discount``.
        """
        metrics = metrics or {}
        exp, env, ppo = self.exp, self.exp.env, self.exp.ppo
        w, world, band, rew = env.reward.weights, env.world, env.band, env.reward
        return {
            # --- identity / provenance ---
            "timestamp": self.created_at,
            "run_dir": self.run_dir.name,
            "code": exp.code,
            "tag": exp.tag or "",
            "status": status,
            "backend": self.backend_kind,
            "git_sha": git_sha(),
            # --- scenario ---
            "num_fbs": exp.num_fbs,
            "scenario_id": exp.scenario_id,
            "config_id": exp.config_id,
            "world_width": world.width,
            "world_height": world.height,
            "num_mbs": world.num_mbs,
            "num_users": world.num_users,
            "sinr_threshold": world.sinr_threshold,
            "mbs_power": world.mbs_power,
            "mbs_height": world.mbs_height,
            "quadriga_scenario": world.scenario,
            "map_mode": world.map_mode,
            # --- band ---
            "band_mode": band.mode,
            "fbs_band": band.fbs_band,
            "mbs_capacity": band.mbs_capacity,
            # --- cost function ---
            "reward_mode": rew.mode,
            "reward_shaping": rew.shaping,
            "record_weight": rew.record_weight,
            "reward_offset": rew.reward_offset,
            "reward_gain": rew.reward_gain,
            "power_includes_macro_capacity": rew.power_includes_macro_capacity,
            "beta": w.beta,
            "gamma": w.gamma,                      # cost-function gamma
            "fbs_weight": w.fbs_weight,
            "fbs_exponent": w.fbs_exponent,
            "epsilon": w.epsilon,
            # --- env / action parameterization ---
            "max_episode_steps": env.max_episode_steps,
            "action_scale": env.action_scale,
            "action_mode": env.action_mode,
            "normalize_obs": env.normalize_obs,
            "binary_mode": env.binary_mode,
            "obs_tier_metrics": env.obs_tier_metrics,
            "terminate_on_full_coverage": env.terminate_on_full_coverage,
            "fbs_z_bounds": f"{env.fbs_z_bounds[0]}-{env.fbs_z_bounds[1]}",
            "fbs_power_bounds": f"{env.fbs_power_bounds[0]}-{env.fbs_power_bounds[1]}",
            # --- optimizer ---
            "algo": ppo.algo,
            "normalize_reward": ppo.normalize_reward,
            "total_timesteps": ppo.total_timesteps,
            "learning_rate": ppo.learning_rate,
            "discount": ppo.gamma,                 # RL discount factor
            "ent_coef": ppo.ent_coef,
            "n_steps": ppo.n_steps,
            "batch_size": ppo.batch_size,
            "n_epochs": ppo.n_epochs,
            "gae_lambda": ppo.gae_lambda,
            "clip_range": ppo.clip_range,
            "vf_coef": ppo.vf_coef,
            "max_grad_norm": ppo.max_grad_norm,
            "buffer_size": ppo.buffer_size,
            "learning_starts": ppo.learning_starts,
            "tau": ppo.tau,
            "train_freq": ppo.train_freq,
            "gradient_steps": ppo.gradient_steps,
            "sac_ent_coef": ppo.sac_ent_coef,
            "seed": ppo.seed,
            "n_envs": ppo.n_envs,
            "eval_every": ppo.eval_every,
            "eval_episodes": ppo.eval_episodes,
            "checkpoint_every": ppo.checkpoint_every,
            # --- outcome ---
            **{f"final_{k}": v for k, v in metrics.items()},
        }

    def append_ledger(self, status: str, metrics: Optional[dict] = None) -> None:
        row = self.ledger_row(status, metrics)
        append_ledger_row(row, LEDGER_PATH)

    # ------------------------------------------------------------------ #
    def finalize(self, status: str = "completed", metrics: Optional[dict] = None) -> None:
        self.steps.close()
        self.episodes.close()
        self.eval_log.close()
        self.update_config(status=status, finished_at=timestamp())
        self.append_ledger(status, metrics)
        for h in list(self.log.handlers):
            h.close()
            self.log.removeHandler(h)
