"""TD3 prototype (td3/ package) on the analytic backend (no MATLAB).

The td3 package reuses ppo's environment, callbacks, rollouts and plots
through imports only. These tests pin that it trains, evaluates, resumes and
plots end to end, that its environment is built exactly like a PPO run's,
that its per-dimension exploration noise lands on the binary gene dims, and
that its runs and ledger stay out of ppo_runs/ and training_log.csv.
"""
import csv
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ppo.config import (
    BandConfig, EnvConfig, ExperimentConfig, RewardConfig, RewardWeights, WorldConfig,
)
from td3 import (
    TD3Experiment, TD3Params, binary_action_indices, evaluate_run,
    td3_experiment_from_json_dict, train,
)


@pytest.fixture(autouse=True)
def _isolate_td3_paths(tmp_path, monkeypatch):
    """Keep test runs out of the real td3_runs/ and its ledger."""
    import td3.paths as tp

    monkeypatch.setattr(tp, "TD3_RUNS_DIR", tmp_path / "td3_runs")
    monkeypatch.setattr(tp, "TD3_LEDGER_PATH", tmp_path / "td3_runs" / "ledger.csv")
    yield


@pytest.fixture
def tiny_td3_experiment(tiny_experiment):
    return TD3Experiment(
        num_fbs=1,
        env=tiny_experiment.env,
        td3=TD3Params(
            total_timesteps=128, batch_size=32, buffer_size=1000,
            learning_starts=32, seed=7, verbose=0, log_flush_every=16,
        ),
    )


# ----------------------------------------------------------------------- #
# Per-dimension exploration noise
#
# TD3's actor is deterministic and saturates its tanh on the binary gene
# dims, exactly as DDPG's does -- the twin critics and delayed updates do
# nothing about it. binary_noise_std is the same remedy, and these tests are
# the td3 mirror of the ones in test_ddpg.py.
# ----------------------------------------------------------------------- #
@pytest.mark.parametrize("code,band,expected", [
    ("1-1-1", BandConfig(), [4]),
    ("2-1-1", BandConfig(), [4, 9]),
    ("1-1-1", BandConfig(mode="multi", fbs_band="agent"), [4, 5]),
])
def test_binary_action_indices_match_the_env_layout(code, band, expected):
    exp = TD3Experiment.from_code(code, band=band)
    assert binary_action_indices(exp.env) == expected


def test_noise_sigma_is_flat_unless_binary_std_is_set():
    from td3.train import build_action_noise, noise_sigma

    exp = TD3Experiment.from_code("1-1-1", td3_overrides={"action_noise_std": 0.1})
    np.testing.assert_allclose(noise_sigma(exp, 5), np.full(5, 0.1))

    exp = replace(exp, td3=replace(exp.td3, binary_noise_std=1.0))
    np.testing.assert_allclose(noise_sigma(exp, 5), [0.1, 0.1, 0.1, 0.1, 1.0])

    noise = build_action_noise(exp, 5)
    np.testing.assert_allclose(noise._sigma, [0.1, 0.1, 0.1, 0.1, 1.0])


def test_zero_noise_everywhere_disables_the_noise_object():
    from td3.train import build_action_noise

    exp = TD3Experiment.from_code(
        "1-1-1", td3_overrides={"action_noise_std": 0.0, "binary_noise_std": 0.0})
    assert build_action_noise(exp, 5) is None


def test_binary_noise_is_recorded_in_the_run_config(tiny_td3_experiment, tmp_path):
    exp = replace(tiny_td3_experiment,
                  td3=replace(tiny_td3_experiment.td3, binary_noise_std=0.9))
    run_dir = train(exp, backend="analytic", run_dir=tmp_path / "bn")
    cfg = json.loads((run_dir / "experiment_config.json").read_text())
    assert cfg["td3"]["binary_noise_std"] == 0.9
    sigma = cfg["provenance"]["action_noise_sigma"]
    dims = cfg["provenance"]["binary_action_dims"]
    assert all(sigma[i] == 0.9 for i in dims)
    assert all(sigma[i] != 0.9 for i in range(len(sigma)) if i not in dims)


# ----------------------------------------------------------------------- #
# Config
# ----------------------------------------------------------------------- #
def test_env_half_is_built_exactly_like_ppo():
    kwargs = dict(
        band=BandConfig(mode="multi", fbs_band="agent", mbs_capacity="agent"),
        reward=RewardConfig(mode="controlled_blend", shaping="potential_record",
                            record_weight=1.0, weights=RewardWeights(gamma=0.1)),
        env_overrides={"normalize_obs": True, "binary_mode": "sticky"},
        max_episode_steps=40,
        action_scale=0.05,
    )
    td3_exp = TD3Experiment.from_code("2-2-1", **kwargs)
    ppo_exp = ExperimentConfig.from_code("2-2-1", **kwargs)
    assert td3_exp.env == ppo_exp.env
    assert td3_exp.code == ppo_exp.code == "2-2-1"
    assert td3_exp.label == "2-2-1_mb_td3"


def test_config_roundtrip_and_defaults():
    exp = TD3Experiment.from_code(
        "1-1-1", tag="probe", total_timesteps=5000, seed=3,
        td3_overrides={"policy_delay": 3, "action_noise_std": 0.2, "net_arch": [64, 64]},
    )
    payload = json.loads(json.dumps(exp.to_json_dict()))
    assert payload["algo"] == "td3" and "ppo" not in payload
    loaded = td3_experiment_from_json_dict(payload)
    assert loaded == exp
    assert loaded.td3.net_arch == (64, 64)
    assert loaded.label == "1-1-1_td3_probe"

    # Fields missing from an older payload fall back to TD3Params defaults.
    payload["td3"].pop("target_noise_clip")
    assert td3_experiment_from_json_dict(payload).td3.target_noise_clip == 0.5


def test_config_validation():
    with pytest.raises(ValueError):
        TD3Params(policy_delay=0)
    with pytest.raises(ValueError):
        TD3Params(action_noise_std=-0.1)
    with pytest.raises(ValueError):
        TD3Params(net_arch=())
    with pytest.raises(ValueError):
        td3_experiment_from_json_dict(ExperimentConfig.from_code("1-1-1").to_json_dict())


# ----------------------------------------------------------------------- #
# Train / evaluate / resume / plots
# ----------------------------------------------------------------------- #
def test_train_writes_artifacts_and_its_own_ledger(tiny_td3_experiment, tmp_path):
    import ppo.run_logging as ppo_rl
    import td3.paths as tp

    run_dir = train(tiny_td3_experiment, backend="analytic", run_dir=tmp_path / "td3run")
    for name in ("model.zip", "experiment_config.json", "steps.csv",
                 "episodes.csv", "monitor.csv", "progress.csv"):
        assert (run_dir / name).exists(), f"missing {name}"
    steps = pd.read_csv(run_dir / "steps.csv")
    assert len(steps) == tiny_td3_experiment.td3.total_timesteps
    assert "reward_objective" in steps.columns

    cfg = json.loads((run_dir / "experiment_config.json").read_text())
    assert cfg["algo"] == "td3" and cfg["provenance"]["status"] == "completed"

    rows = list(csv.DictReader(tp.TD3_LEDGER_PATH.open()))
    assert len(rows) == 1
    row = rows[0]
    assert row["algo"] == "td3" and row["policy_delay"] == "2"
    assert row["discount"] == str(tiny_td3_experiment.td3.gamma)
    for ppo_only in ("n_steps", "ent_coef", "clip_range", "sac_ent_coef"):
        assert ppo_only not in row
    assert not ppo_rl.LEDGER_PATH.exists(), "a TD3 run wrote to the PPO ledger"


def test_default_run_dir_is_under_td3_runs(tiny_td3_experiment):
    import td3.paths as tp

    run_dir = train(tiny_td3_experiment, backend="analytic")
    assert run_dir.parent == tp.TD3_RUNS_DIR
    assert run_dir.name.endswith("_1-1-1_td3")


def test_evaluate_and_plot_a_trained_run(tiny_td3_experiment, tmp_path):
    from td3.plotting import training_gallery

    run_dir = train(tiny_td3_experiment, backend="analytic", run_dir=tmp_path / "td3run")
    report = evaluate_run(run_dir, episodes=2, backend="analytic", make_plots=True)
    assert len(report.episodes) == 2
    for name in ("summary.csv", "eval_config.json", "ep00_trajectory.csv", "ep00_metrics.csv"):
        assert (report.eval_dir / name).exists(), f"missing {name}"
    assert (report.eval_dir / "plots" / "ep00_trajectory.png").exists()
    assert json.loads((report.eval_dir / "eval_config.json").read_text())["algo"] == "td3"

    out = training_gallery(run_dir)
    assert (out / "training_overview.png").exists()
    assert (out / "reward_components.png").exists()


def test_training_is_reproducible(tiny_td3_experiment, tmp_path):
    run_a = train(tiny_td3_experiment, backend="analytic", run_dir=tmp_path / "a")
    run_b = train(tiny_td3_experiment, backend="analytic", run_dir=tmp_path / "b")
    np.testing.assert_allclose(
        pd.read_csv(run_a / "steps.csv")["reward"].to_numpy(),
        pd.read_csv(run_b / "steps.csv")["reward"].to_numpy(),
    )


def test_resume_continues_training(tiny_td3_experiment, tmp_path):
    first = train(tiny_td3_experiment, backend="analytic", run_dir=tmp_path / "first")
    exp2 = replace(tiny_td3_experiment, td3=replace(tiny_td3_experiment.td3, total_timesteps=256))
    resumed = train(exp2, backend="analytic", run_dir=tmp_path / "resumed", resume_from=first)
    assert (resumed / "model.zip").exists()
    cfg = json.loads((resumed / "experiment_config.json").read_text())
    assert cfg["provenance"]["resumed_from"] == str(first)
    assert len(pd.read_csv(resumed / "steps.csv")) > 0


def test_cli_train_eval_plot(tmp_path):
    import td3.paths as tp
    from td3.cli import main

    assert main([
        "train", "--backend", "analytic", "--timesteps", "96", "--max-episode-steps", "8",
        "--learning-starts", "16", "--batch-size", "16", "--buffer-size", "500",
        "--policy-delay", "3", "--action-noise-std", "0.2", "--net-arch", "32,32",
        "--normalize-obs", "--sticky-binaries", "--reward-shaping", "potential", "--tag", "cli",
    ]) == 0
    (run_dir,) = sorted(tp.TD3_RUNS_DIR.glob("run_*_td3_cli"))
    cfg = json.loads((run_dir / "experiment_config.json").read_text())
    assert cfg["td3"]["policy_delay"] == 3
    assert cfg["td3"]["net_arch"] == [32, 32]
    assert cfg["env"]["reward"]["shaping"] == "potential"
    assert main(["eval", "--run", run_dir.name, "--episodes", "1",
                 "--backend", "analytic", "--no-plots"]) == 0
    assert main(["plot", "--run", "latest"]) == 0
    assert (run_dir / "plots" / "training_overview.png").exists()


# ----------------------------------------------------------------------- #
# Learning check (mirrors the SAC structural-stack test in ppo's suite)
# ----------------------------------------------------------------------- #
def test_td3_beats_do_nothing_baseline(tmp_path):
    """On a weak-macro analytic world with coverage holes, a short seeded TD3
    run must land its deterministic policy clearly above the FBS-off
    objective. Wide margins (observed: +0.46 on seeds 7, 8 and 9)."""
    from ppo.env import FlyingBaseStationEnv
    from ppo.matlab_bridge import AnalyticSinrBackend

    world = WorldConfig(width=1000.0, height=800.0, num_mbs=1,
                        num_users=200, mbs_power=-30.0)
    env_cfg = EnvConfig(
        num_fbs=1, world=world, band=BandConfig(),
        reward=RewardConfig(mode="legacy_blend", shaping="potential",
                            weights=RewardWeights(beta=0.8, gamma=0.0, fbs_weight=0.4)),
        max_episode_steps=10, action_scale=0.3,
        normalize_obs=True, binary_mode="sticky", obs_tier_metrics=True,
        terminate_on_full_coverage=False,
    )

    env = FlyingBaseStationEnv(env_cfg, backend=AnalyticSinrBackend(world, env_cfg.band))
    env.reset(options={"state": np.array([500.0, 400.0, 60.0, 8.0, 0.0], np.float32)})
    _, _, _, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
    baseline = info["reward_objective"]
    env.close()
    assert baseline < 0.6

    exp = TD3Experiment(
        num_fbs=1, env=env_cfg,
        td3=TD3Params(total_timesteps=2000, batch_size=64, buffer_size=5000,
                      learning_starts=200, gamma=0.95, seed=7, verbose=0,
                      log_flush_every=64),
    )
    run_dir = train(exp, backend="analytic", run_dir=tmp_path / "learn")
    report = evaluate_run(run_dir, episodes=3, backend="analytic", make_plots=False, save=False)
    finals = [float(r.metrics_df["reward_objective"].iloc[-1]) for r in report.episodes]
    assert np.mean(finals) > baseline + 0.10, (finals, baseline)
    assert max(finals) > baseline + 0.30, (finals, baseline)
