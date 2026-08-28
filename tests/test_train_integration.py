"""End-to-end pipeline tests on the analytic backend (no MATLAB).

These are the slowest tests in the suite (~10-20 s total: real PPO updates on
a tiny MLP) and the highest-value ones: they exercise train -> artifacts ->
evaluate -> plots exactly as a real run does.
"""
import csv
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ppo.config import BandConfig
from ppo.evaluate import evaluate_run
from ppo.train import train


@pytest.fixture
def trained_run(tiny_experiment, tmp_path):
    run_dir = train(
        tiny_experiment, backend="analytic", run_dir=tmp_path / "run_test"
    )
    return run_dir


def test_train_writes_all_artifacts(trained_run, tiny_experiment):
    for name in (
        "model.zip", "experiment_config.json", "steps.csv",
        "episodes.csv", "progress.csv", "monitor.csv", "run.log",
    ):
        assert (trained_run / name).exists(), f"missing {name}"

    steps = pd.read_csv(trained_run / "steps.csv")
    assert len(steps) == tiny_experiment.ppo.total_timesteps
    for col in ("reward", "total_connected", "fbs_connected", "total_power",
                "fbs0_x", "fbs0_power_status", "reward_norm_users"):
        assert col in steps.columns, f"missing steps column {col}"

    episodes = pd.read_csv(trained_run / "episodes.csv")
    assert len(episodes) >= tiny_experiment.ppo.total_timesteps // tiny_experiment.env.max_episode_steps - 1
    assert "final_total_connected" in episodes.columns


def test_evaluate_run_on_trained_artifacts(trained_run):
    report = evaluate_run(
        trained_run, episodes=2, backend="analytic", make_plots=True
    )
    assert len(report.episodes) == 2
    assert report.eval_dir.exists()
    for name in ("summary.csv", "summary.json", "ep00_trajectory.csv",
                 "ep00_metrics.csv", "ep00_users.csv", "mbs.csv"):
        assert (report.eval_dir / name).exists(), f"missing {name}"
    assert (report.eval_dir / "plots" / "ep00_trajectory.png").exists()
    assert (report.eval_dir / "plots" / "ep00_metrics.png").exists()
    # trajectory CSV keeps the old tooling format (MultiIndex, reload-able)
    df = pd.read_csv(report.eval_dir / "ep00_trajectory.csv",
                     header=[0, 1], index_col=0)
    assert ("fbs0", "x") in df.columns


def test_training_is_reproducible(tiny_experiment, tmp_path):
    run_a = train(tiny_experiment, backend="analytic", run_dir=tmp_path / "a")
    run_b = train(tiny_experiment, backend="analytic", run_dir=tmp_path / "b")
    steps_a = pd.read_csv(run_a / "steps.csv")
    steps_b = pd.read_csv(run_b / "steps.csv")
    np.testing.assert_allclose(
        steps_a["reward"].to_numpy(), steps_b["reward"].to_numpy()
    )
    np.testing.assert_allclose(
        steps_a["fbs0_x"].to_numpy(), steps_b["fbs0_x"].to_numpy()
    )


def test_seed_changes_trajectory(tiny_experiment, tmp_path):
    exp2 = replace(tiny_experiment, ppo=replace(tiny_experiment.ppo, seed=99))
    run_a = train(tiny_experiment, backend="analytic", run_dir=tmp_path / "a")
    run_b = train(exp2, backend="analytic", run_dir=tmp_path / "b")
    steps_a = pd.read_csv(run_a / "steps.csv")
    steps_b = pd.read_csv(run_b / "steps.csv")
    assert not np.allclose(steps_a["fbs0_x"], steps_b["fbs0_x"])


def test_multiband_training_smoke(tiny_experiment, tmp_path):
    env_cfg = replace(
        tiny_experiment.env,
        band=BandConfig(mode="multi", fbs_band="agent", mbs_capacity="agent"),
    )
    exp = replace(tiny_experiment, env=env_cfg)
    run_dir = train(exp, backend="analytic", run_dir=tmp_path / "mb")
    steps = pd.read_csv(run_dir / "steps.csv")
    assert "fbs0_band_flag" in steps.columns
    assert "mbs0_capacity" in steps.columns
    assert "num_capacity_fbs" in steps.columns

    report = evaluate_run(run_dir, episodes=1, backend="analytic", make_plots=False)
    assert ("fbs0", "band_flag") in report.episodes[0].state_df.columns


def test_resume_continues_training(tiny_experiment, tmp_path):
    first = train(tiny_experiment, backend="analytic", run_dir=tmp_path / "first")
    resumed = train(
        tiny_experiment,
        backend="analytic",
        run_dir=tmp_path / "resumed",
        resume_from=first,
    )
    assert (resumed / "model.zip").exists()
    steps = pd.read_csv(resumed / "steps.csv")
    assert len(steps) > 0


def test_checkpoints_and_eval_log(tiny_experiment, tmp_path):
    exp = replace(
        tiny_experiment,
        ppo=replace(
            tiny_experiment.ppo,
            checkpoint_every=64, eval_every=64, eval_episodes=2,
        ),
    )
    run_dir = train(exp, backend="analytic", run_dir=tmp_path / "ck")
    assert list((run_dir / "checkpoints").glob("model_*.zip"))
    assert (run_dir / "best_model.zip").exists()
    eval_log = pd.read_csv(run_dir / "eval_log.csv")
    assert {"timesteps", "mean_reward", "mean_total_connected"} <= set(eval_log.columns)


# ----------------------------------------------------------------------- #
# SAC (off-policy) path
# ----------------------------------------------------------------------- #
@pytest.fixture
def tiny_sac_experiment(tiny_experiment):
    from ppo.config import PPOParams

    return replace(
        tiny_experiment,
        ppo=PPOParams(
            algo="sac", total_timesteps=128, batch_size=32, buffer_size=1000,
            learning_starts=32, seed=7, verbose=0, log_flush_every=16,
        ),
    )


def test_sac_train_writes_artifacts_and_evaluates(tiny_sac_experiment, tmp_path):
    run_dir = train(tiny_sac_experiment, backend="analytic", run_dir=tmp_path / "sac")
    for name in ("model.zip", "experiment_config.json", "steps.csv",
                 "episodes.csv", "monitor.csv"):
        assert (run_dir / name).exists(), f"missing {name}"
    steps = pd.read_csv(run_dir / "steps.csv")
    assert len(steps) == tiny_sac_experiment.ppo.total_timesteps
    assert "reward_objective" in steps.columns

    import json
    cfg = json.loads((run_dir / "experiment_config.json").read_text())
    assert cfg["ppo"]["algo"] == "sac"

    # evaluate_run must pick SAC.load from the recorded config
    report = evaluate_run(run_dir, episodes=1, backend="analytic",
                          make_plots=False, save=False)
    assert len(report.episodes) == 1
    assert report.exp.ppo.algo == "sac"


def test_sac_training_is_reproducible(tiny_sac_experiment, tmp_path):
    run_a = train(tiny_sac_experiment, backend="analytic", run_dir=tmp_path / "a")
    run_b = train(tiny_sac_experiment, backend="analytic", run_dir=tmp_path / "b")
    steps_a = pd.read_csv(run_a / "steps.csv")
    steps_b = pd.read_csv(run_b / "steps.csv")
    np.testing.assert_allclose(
        steps_a["reward"].to_numpy(), steps_b["reward"].to_numpy()
    )


def test_sac_resume_continues_training(tiny_sac_experiment, tmp_path):
    first = train(tiny_sac_experiment, backend="analytic", run_dir=tmp_path / "first")
    exp2 = replace(
        tiny_sac_experiment,
        ppo=replace(tiny_sac_experiment.ppo, total_timesteps=256),
    )
    resumed = train(exp2, backend="analytic", run_dir=tmp_path / "resumed",
                    resume_from=first)
    assert (resumed / "model.zip").exists()
    steps = pd.read_csv(resumed / "steps.csv")
    assert len(steps) > 0


# ----------------------------------------------------------------------- #
# Learning check: the structural stack must escape the do-nothing baseline
# ----------------------------------------------------------------------- #
def test_structural_stack_beats_do_nothing_baseline(tmp_path):
    """Regression for the failure mode of the real-physics runs (policies
    converging to the FBS-off plateau): with normalized obs, sticky binaries,
    potential shaping and SAC, a short analytic training must land the
    deterministic eval policy clearly above the do-nothing objective.
    Fully seeded (train seed + eval seeds), wide margins."""
    from ppo.config import (
        BandConfig, EnvConfig, ExperimentConfig, PPOParams,
        RewardConfig, RewardWeights, WorldConfig,
    )
    from ppo.env import FlyingBaseStationEnv
    from ppo.matlab_bridge import AnalyticSinrBackend

    # Weak macro -> coverage holes, the regime of the real 2000x1500 world.
    world = WorldConfig(width=1000.0, height=800.0, num_mbs=1,
                        num_users=200, mbs_power=-30.0)
    env_cfg = EnvConfig(
        num_fbs=1, world=world, band=BandConfig(),
        reward=RewardConfig(mode="legacy_blend", shaping="potential",
                            weights=RewardWeights(beta=0.8, gamma=0.0,
                                                  fbs_weight=0.4)),
        max_episode_steps=10, action_scale=0.3,
        normalize_obs=True, binary_mode="sticky", obs_tier_metrics=True,
        terminate_on_full_coverage=False,
    )

    # Do-nothing baseline objective (FBS off; zero action holds it off
    # under sticky binaries).
    env = FlyingBaseStationEnv(env_cfg, backend=AnalyticSinrBackend(world, env_cfg.band))
    off = np.array([500.0, 400.0, 60.0, 8.0, 0.0], np.float32)
    env.reset(options={"state": off})
    _, _, _, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
    baseline = info["reward_objective"]
    env.close()
    assert baseline < 0.6  # the macro alone must not already be near-optimal

    exp = ExperimentConfig(
        num_fbs=1, scenario_id=1, config_id=1, env=env_cfg,
        ppo=PPOParams(algo="sac", total_timesteps=2000, batch_size=64,
                      buffer_size=5000, learning_starts=200, gamma=0.95,
                      seed=7, verbose=0, log_flush_every=64),
    )
    run_dir = train(exp, backend="analytic", run_dir=tmp_path / "learn")
    report = evaluate_run(run_dir, episodes=3, backend="analytic",
                          make_plots=False, save=False)
    finals = [float(r.metrics_df["reward_objective"].iloc[-1])
              for r in report.episodes]
    assert np.mean(finals) > baseline + 0.10, (finals, baseline)
    assert max(finals) > baseline + 0.30, (finals, baseline)


# ----------------------------------------------------------------------- #
# Running reward normalization (answer-free alternative to reward_gain)
# ----------------------------------------------------------------------- #
def test_normalize_reward_scales_training_signal_not_logged_objective(
    tiny_experiment, tmp_path
):
    """VecNormalize must rescale what the optimizer sees while leaving the
    logged objective raw — every comparison depends on the raw value."""
    from ppo.config import PPOParams

    base_exp = replace(
        tiny_experiment,
        env=replace(tiny_experiment.env,
                    reward=replace(tiny_experiment.env.reward, shaping="record")),
    )
    plain = train(base_exp, backend="analytic", run_dir=tmp_path / "plain")
    norm_exp = replace(
        base_exp, ppo=replace(base_exp.ppo, normalize_reward=True)
    )
    normed = train(norm_exp, backend="analytic", run_dir=tmp_path / "normed")

    a = pd.read_csv(plain / "steps.csv")
    b = pd.read_csv(normed / "steps.csv")

    # Up to the first policy update the two runs are identical: same seed,
    # same policy init, and reward scaling cannot act before a gradient step.
    # Afterwards they legitimately diverge, because rescaling changes the
    # updates -- which is the entire point.
    pre = base_exp.ppo.n_steps
    np.testing.assert_allclose(
        a["reward_objective"].to_numpy()[:pre],
        b["reward_objective"].to_numpy()[:pre],
    )
    # The logged objective stays raw in both (a blend of fractions, so <= 1).
    assert b["reward_objective"].max() <= 1.0
    # ... while the reward handed to the optimizer is rescaled well above the
    # raw per-step improvements it is derived from.
    assert b["reward"].max() > 5 * a["reward"].max()
    # statistics are persisted for faithful resume
    assert (normed / "vecnormalize.pkl").exists()
    assert not (plain / "vecnormalize.pkl").exists()


def test_normalize_reward_recorded_in_config_and_ledger(tiny_experiment, tmp_path, monkeypatch):
    import ppo.run_logging as rl

    ledger = tmp_path / "ledger.csv"
    monkeypatch.setattr(rl, "LEDGER_PATH", ledger)
    exp = replace(tiny_experiment, ppo=replace(tiny_experiment.ppo, normalize_reward=True))
    run_dir = train(exp, backend="analytic", run_dir=tmp_path / "r")

    import json
    assert json.loads((run_dir / "experiment_config.json").read_text())["ppo"]["normalize_reward"]
    row = list(csv.DictReader(ledger.open()))[-1]
    assert row["normalize_reward"] == "True"


def test_normalize_reward_resume_restores_statistics(tiny_experiment, tmp_path):
    exp = replace(tiny_experiment, ppo=replace(tiny_experiment.ppo, normalize_reward=True))
    first = train(exp, backend="analytic", run_dir=tmp_path / "first")
    resumed = train(exp, backend="analytic", run_dir=tmp_path / "resumed",
                    resume_from=first)
    assert (resumed / "model.zip").exists()
    assert (resumed / "vecnormalize.pkl").exists()
