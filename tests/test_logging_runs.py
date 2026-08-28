import csv
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from ppo.config import ExperimentConfig
from ppo.run_logging import BufferedCsvWriter, RunLogger
from ppo.runs import list_runs, load_run


def test_buffered_writer_batches_and_aligns(tmp_path):
    path = tmp_path / "out.csv"
    w = BufferedCsvWriter(path, flush_every=3)
    w.write({"a": 1, "b": 2})
    w.write({"a": 3, "b": 4, "c": 99})   # extra key dropped
    assert not path.exists()             # still buffered
    w.write({"a": 5})                    # missing key -> empty
    assert path.exists()                 # flushed at 3
    w.write({"a": 7, "b": 8})
    w.close()                            # flush remainder

    df = pd.read_csv(path)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 4
    assert df["a"].tolist() == [1, 3, 5, 7]
    assert pd.isna(df["b"].iloc[2])


def test_run_logger_writes_config_and_ledger(tmp_path):
    import ppo.run_logging as rl

    exp = ExperimentConfig.from_code("1-1-1", tag="unit")
    logger = RunLogger(exp, runs_root=tmp_path, backend_kind="analytic")
    logger.write_config()
    logger.steps.write({"timesteps": 1, "reward": 0.5})
    logger.episodes.write({"episode": 0, "reward_sum": 1.0})
    logger.finalize(status="completed", metrics={"reward_sum": 1.0})

    run_dir = logger.run_dir
    assert run_dir.name.endswith("_1-1-1_unit")
    cfg = json.loads((run_dir / "experiment_config.json").read_text())
    assert cfg["config_version"] == 2
    assert cfg["provenance"]["backend"] == "analytic"
    assert cfg["provenance"]["status"] == "completed"
    assert (run_dir / "steps.csv").exists()
    assert (run_dir / "episodes.csv").exists()

    ledger = pd.read_csv(rl.LEDGER_PATH)
    assert len(ledger) == 1
    assert ledger.iloc[0]["run_dir"] == run_dir.name
    assert ledger.iloc[0]["status"] == "completed"


def test_ledger_append_survives_schema_change(tmp_path):
    """New columns in later runs must not corrupt an existing ledger."""
    import ppo.run_logging as rl

    exp = ExperimentConfig.from_code("1-1-1")
    l1 = RunLogger(exp, runs_root=tmp_path, backend_kind="analytic")
    l1.append_ledger("completed", {"reward_sum": 1.0})
    l2 = RunLogger(exp, runs_root=tmp_path, backend_kind="analytic")
    l2.append_ledger("completed", {"reward_sum": 2.0, "brand_new_metric": 5})
    ledger = pd.read_csv(rl.LEDGER_PATH)
    assert len(ledger) == 2  # second row aligned to first header


def test_run_discovery_schemas(tmp_path):
    # v2 run
    exp = ExperimentConfig.from_code("2-1-1")
    logger = RunLogger(exp, runs_root=tmp_path, backend_kind="analytic")
    logger.write_config()
    logger.finalize()
    # v1 run
    v1 = tmp_path / "run_039"
    v1.mkdir()
    (v1 / "experiment_config.json").write_text(json.dumps({
        "num_fbs": 1, "scenario_id": 1, "config_id": 2,
        "total_timesteps": 3000, "code": "1-1-2",
    }))
    (v1 / "ppo_fbs_agent.zip").write_bytes(b"zip")
    # bare run
    bare = tmp_path / "run_000_cand1"
    bare.mkdir()
    (bare / "ppo_fbs_agent.zip").write_bytes(b"zip")

    df = list_runs(tmp_path)
    assert set(df["schema"]) == {"v2", "v1", "bare"}

    h_v1 = load_run("run_039", tmp_path)
    assert h_v1.schema == "v1"
    assert h_v1.exp.env.reward.weights.gamma == 0.1
    assert h_v1.model_path.name == "ppo_fbs_agent.zip"

    h_bare = load_run("run_000_cand1", tmp_path)
    assert h_bare.schema == "bare" and h_bare.exp is None

    h_v2 = load_run(logger.run_dir.name, tmp_path)
    assert h_v2.schema == "v2" and h_v2.exp.code == "2-1-1"


# ----------------------------------------------------------------------- #
# Ledger completeness — every setting that can change a result is recorded
# ----------------------------------------------------------------------- #
def test_ledger_row_covers_every_config_field(tiny_experiment):
    """Guard against the class of bug where a new setting is added to the
    config but never reaches the ledger."""
    from dataclasses import fields as dc_fields

    from ppo.run_logging import RunLogger

    logger = RunLogger.__new__(RunLogger)
    logger.exp = tiny_experiment
    logger.run_dir = Path("run_x")
    logger.created_at = "2026-01-01_00-00-00"
    logger.backend_kind = "analytic"
    row = logger.ledger_row("completed", {"reward_sum": 1.0})

    # every EnvConfig scalar setting present (nested blocks checked below)
    skip_env = {"world", "band", "reward", "num_fbs", "collect_user_positions"}
    for f in dc_fields(tiny_experiment.env):
        if f.name in skip_env:
            continue
        assert f.name in row, f"EnvConfig.{f.name} missing from ledger"
    # reward block
    for f in dc_fields(tiny_experiment.env.reward):
        if f.name == "weights":
            continue
        key = f.name if f.name.startswith("reward_") else (
            "reward_mode" if f.name == "mode" else
            "reward_shaping" if f.name == "shaping" else f.name)
        assert key in row, f"RewardConfig.{f.name} missing from ledger"
    for f in dc_fields(tiny_experiment.env.reward.weights):
        assert f.name in row, f"RewardWeights.{f.name} missing from ledger"
    # optimizer: PPO + SAC knobs, with the discount kept distinct from
    # the cost-function gamma
    for f in dc_fields(tiny_experiment.ppo):
        key = "discount" if f.name == "gamma" else (
            "verbose" if f.name == "verbose" else
            "log_flush_every" if f.name == "log_flush_every" else f.name)
        if key in ("verbose", "log_flush_every"):
            continue  # logging-only, cannot change a result
        assert key in row, f"PPOParams.{f.name} missing from ledger"
    assert row["discount"] == tiny_experiment.ppo.gamma
    assert row["gamma"] == tiny_experiment.env.reward.weights.gamma


def test_ledger_widens_header_instead_of_dropping_new_columns(tmp_path):
    """The original defect: a pinned header silently discarded new fields."""
    from ppo.run_logging import append_ledger_row

    path = tmp_path / "ledger.csv"
    append_ledger_row({"run_dir": "a", "seed": 1}, path)
    append_ledger_row({"run_dir": "b", "seed": 2, "algo": "sac"}, path)

    rows = list(csv.DictReader(path.open()))
    assert [r["run_dir"] for r in rows] == ["a", "b"]
    assert "algo" in rows[0], "header was not widened"
    assert rows[0]["algo"] == ""          # old row blank for the new column
    assert rows[1]["algo"] == "sac"       # new row keeps its value
    assert rows[0]["seed"] == "1" and rows[1]["seed"] == "2"
    assert path.with_suffix(".csv.bak").exists(), "no backup before rewrite"


def test_ledger_roundtrip_through_a_real_run(tiny_experiment, tmp_path, monkeypatch):
    import ppo.run_logging as rl

    ledger = tmp_path / "ledger.csv"
    monkeypatch.setattr(rl, "LEDGER_PATH", ledger)
    from ppo.train import train

    exp = replace(
        tiny_experiment,
        env=replace(tiny_experiment.env, action_mode="absolute",
                    reward=replace(tiny_experiment.env.reward, shaping="record",
                                   reward_gain=13.6)),
    )
    train(exp, backend="analytic", run_dir=tmp_path / "r")
    row = list(csv.DictReader(ledger.open()))[-1]
    assert row["action_mode"] == "absolute"
    assert row["reward_shaping"] == "record"
    assert float(row["reward_gain"]) == 13.6
    assert row["algo"] == "ppo"
    assert row["status"] == "completed"


def test_compare_runs_shows_only_differing_settings(tiny_experiment, tmp_path, monkeypatch):
    import ppo.run_logging as rl
    from ppo.runs import compare_runs
    from ppo.train import train

    monkeypatch.setattr(rl, "LEDGER_PATH", tmp_path / "ledger.csv")
    train(tiny_experiment, backend="analytic", run_dir=tmp_path / "run_a")
    exp2 = replace(tiny_experiment,
                   env=replace(tiny_experiment.env, action_scale=0.05))
    train(exp2, backend="analytic", run_dir=tmp_path / "run_b")

    df = compare_runs([str(tmp_path / "run_a"), str(tmp_path / "run_b")])
    assert "action_scale" in df.columns          # the thing that differs
    assert "beta" not in df.columns              # identical -> hidden
    assert len(df) == 2
    full = compare_runs([str(tmp_path / "run_a"), str(tmp_path / "run_b")],
                        only_differing=False)
    assert "beta" in full.columns
