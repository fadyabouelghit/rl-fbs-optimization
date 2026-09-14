"""Central path resolution for the PPO pipeline.

No machine-specific path is hardcoded anywhere in this repository. Every
absolute path is read at import time from, in order of precedence:

1. the process environment (``PPO_QUADRIGA_PATH=... python -m ppo train``);
2. ``secrets.env`` at the repo root -- a git-ignored ``KEY=value`` file.
   Copy ``secrets.env.example`` to ``secrets.env`` and fill it in;
3. a repo-relative default, for everything that can have one.

Only ``PPO_QUADRIGA_PATH`` has no default: QuaDRiGa lives outside this repo
and its location differs per machine. It is required by the ``matlab`` backend
only -- the default ``pyqd`` backend (which computes the same physics
in-process), the analytic backend, the test suite, and every plotting/analysis
script run without it. ``require_quadriga_path()`` raises a clear, actionable
error at the point of use rather than at import.

Recognised keys (all optional except where noted):

    PPO_QUADRIGA_PATH   QuaDRiGa ``quadriga_src`` folder   ('matlab' backend only)
    PPO_MATLAB_PATH     .m files added to the MATLAB path  (default: ./matlab)
    PPO_RUNS_DIR        training run directories           (default: ./ppo_runs)
    PPO_TEST_LOG_DIR    legacy per-code eval logs          (default: ./test_logs)
    PPO_LEDGER_PATH     cross-run ledger CSV               (default: ./training_log.csv)
    PPO_CACHE_DIR       MATLAB-written MBS map cache       (default: ./cache_mbs_maps)
    PPO_PYQD_CACHE_DIR  pyqd-written MBS map cache         (default: ./cache_pyqd_maps)
    PPO_SECRETS_FILE    location of the secrets file       (default: ./secrets.env)

The two map caches are deliberately separate directories. MATLAB keys its
``.mat`` files by an MD5 over a MATLAB ``jsonencode`` payload; the pyqd backend
keys its ``.npz`` files by a SHA-256 over a Python payload that also pins the
``pyqd-channel`` version. Sharing one namespace could only ever produce a
silent physics swap, never a useful hit.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

SECRETS_FILE = Path(
    os.environ.get("PPO_SECRETS_FILE", REPO_ROOT / "secrets.env")
).expanduser()


def _load_secrets(path: Path) -> dict[str, str]:
    """Parse a ``KEY=value`` file. Blank lines and ``#`` comments ignored.

    Deliberately dependency-free (no python-dotenv) and forgiving: a missing
    file is not an error, since only the MATLAB backend needs any of it.
    """
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        # Strip one layer of surrounding quotes, if present.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


_SECRETS = _load_secrets(SECRETS_FILE)


def setting(key: str, default: Optional[str] = None) -> Optional[str]:
    """Environment first, then ``secrets.env``, then ``default``."""
    value = os.environ.get(key) or _SECRETS.get(key)
    return value if value else default


def _path_setting(key: str, default: Path) -> Path:
    value = setting(key)
    return Path(value).expanduser() if value else default


# --------------------------------------------------------------------------- #
# External dependency: QuaDRiGa (no default -- machine specific)
# --------------------------------------------------------------------------- #
_quadriga = setting("PPO_QUADRIGA_PATH")
QUADRIGA_PATH: Optional[Path] = Path(_quadriga).expanduser() if _quadriga else None


def require_quadriga_path(override: str | Path | None = None) -> Path:
    """Resolve the QuaDRiGa source path or explain exactly how to set it."""
    path = Path(override).expanduser() if override else QUADRIGA_PATH
    if path is None:
        raise RuntimeError(
            "QuaDRiGa path is not configured, so the MATLAB backend cannot "
            "start.\nSet it either way:\n"
            f"    echo 'PPO_QUADRIGA_PATH=/path/to/quadriga_src' >> {SECRETS_FILE}\n"
            "    export PPO_QUADRIGA_PATH=/path/to/quadriga_src\n"
            "(copy secrets.env.example to secrets.env for the full template). "
            "Neither MATLAB-free backend needs any of this: pass "
            "backend='pyqd' for the same physics in-process, or "
            "backend='analytic' for a fast stand-in."
        )
    if not path.is_dir():
        raise RuntimeError(
            f"PPO_QUADRIGA_PATH points at {path}, which is not a directory. "
            f"Fix it in {SECRETS_FILE} or in the environment."
        )
    return path


# --------------------------------------------------------------------------- #
# Repo-relative locations (overridable, but they work out of the box)
# --------------------------------------------------------------------------- #
MATLAB_PATH = _path_setting("PPO_MATLAB_PATH", REPO_ROOT / "matlab")
RUNS_DIR = _path_setting("PPO_RUNS_DIR", REPO_ROOT / "ppo_runs")
TEST_LOG_DIR = _path_setting("PPO_TEST_LOG_DIR", REPO_ROOT / "test_logs")
LEDGER_PATH = _path_setting("PPO_LEDGER_PATH", REPO_ROOT / "training_log.csv")
CACHE_DIR = _path_setting("PPO_CACHE_DIR", REPO_ROOT / "cache_mbs_maps")
PYQD_CACHE_DIR = _path_setting("PPO_PYQD_CACHE_DIR", REPO_ROOT / "cache_pyqd_maps")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
