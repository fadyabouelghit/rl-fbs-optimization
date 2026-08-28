"""Backfill training_log.csv with settings that were never logged.

The ledger writer used to pin its header to whatever the file already had, so
settings introduced later (``algo``, ``reward_shaping``, ``reward_gain``,
``action_mode``, the RL ``discount``, the structural env flags, ...) were
dropped on write. ``run_logging.append_ledger_row`` now widens the header, but
existing rows still lack those columns.

This script reconstructs them from each run's ``experiment_config.json``,
which has always recorded the full configuration. Outcome metrics
(``final_*``) are kept from the ledger — they are not in the config files.

    python backfill_ledger.py --dry-run     # report what would change
    python backfill_ledger.py               # write (creates a .bak first)

Rows whose run directory or config file is missing (deleted runs, notebook-era
runs) are preserved untouched, with the new columns left blank.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from ppo.config import experiment_from_json_dict
from ppo.paths import LEDGER_PATH, RUNS_DIR
from ppo.run_logging import RunLogger

import json


def config_row(run_dir: Path) -> dict | None:
    """Rebuild the ledger fields for one run from its saved config."""
    cfg_path = run_dir / "experiment_config.json"
    if not cfg_path.exists():
        return None
    try:
        data = json.loads(cfg_path.read_text())
        exp = experiment_from_json_dict(data)
    except Exception:
        return None
    prov = data.get("provenance", {}) or {}

    # Reuse the live schema so backfilled and future rows can never drift.
    logger = RunLogger.__new__(RunLogger)
    logger.exp = exp
    logger.run_dir = run_dir
    logger.created_at = prov.get("created_at", "")
    logger.backend_kind = prov.get("backend", "")
    row = logger.ledger_row(prov.get("status", ""))
    row["git_sha"] = prov.get("git_sha", "") or ""
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--ledger", default=str(LEDGER_PATH))
    args = p.parse_args()

    ledger = Path(args.ledger)
    if not ledger.exists():
        print(f"no ledger at {ledger}")
        return 1

    with ledger.open(newline="") as f:
        reader = csv.DictReader(f)
        old_fields = list(reader.fieldnames or [])
        rows = list(reader)

    # Union of the existing header and the current full schema, existing
    # columns first so downstream scripts keep their positions.
    sample = config_row(RUNS_DIR / rows[-1]["run_dir"]) if rows else None
    schema = list(sample.keys()) if sample else []
    fields = old_fields + [k for k in schema if k not in old_fields]

    filled, skipped = 0, []
    out = []
    for r in rows:
        run_dir = RUNS_DIR / (r.get("run_dir") or "")
        cfg = config_row(run_dir) if run_dir.exists() else None
        merged = {k: r.get(k, "") for k in fields}
        if cfg:
            for k, v in cfg.items():
                # config is authoritative for settings; keep ledger's outcomes
                if not k.startswith("final_"):
                    merged[k] = v
            filled += 1
        else:
            skipped.append(r.get("run_dir", "?"))
        out.append(merged)

    added = [k for k in fields if k not in old_fields]
    print(f"rows: {len(rows)}   backfilled from config: {filled}   "
          f"no config found: {len(skipped)}")
    print(f"columns: {len(old_fields)} -> {len(fields)}")
    print(f"added: {', '.join(added) if added else '(none)'}")
    for s in skipped:
        print(f"  left as-is (new columns blank): {s}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    shutil.copy2(ledger, ledger.with_suffix(ledger.suffix + ".bak"))
    with ledger.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(out)
    print(f"\nwritten: {ledger}   (backup: {ledger.name}.bak)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
