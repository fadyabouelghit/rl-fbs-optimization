"""DDPG prototype (ddpg/ package) on the analytic backend (no MATLAB).

The ddpg package reuses ppo's environment, callbacks, rollouts and plots
through imports only. These tests pin that it trains, evaluates, resumes and
plots end to end, that its environment is built exactly like a PPO run's, that
it really is DDPG (one critic, no TD3 tricks), that its per-dimension
exploration noise lands on the binary gene dims, and that its runs and ledger
stay out of ppo_runs/, td3_runs/ and training_log.csv.
"""
import csv
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ddpg import (
    DDPGExperiment, DDPGParams, binary_action_indices, ddpg_experiment_from_json_dict,
    evaluate_run, train,
)
from ppo.config import (
    BandConfig, EnvConfig, ExperimentConfig, RewardConfig, RewardWeights, WorldConfig,
)


@pytest.fixture(autouse=True)
def _isolate_ddpg_paths(tmp_path, monkeypatch):
    """Keep test runs out of the real ddpg_runs/ and its ledger."""
    import ddpg.paths as dp

    monkeypatch.setattr(dp, "DDPG_RUNS_DIR", tmp_path / "ddpg_runs")
    monkeypatch.setattr(dp, "DDPG_LEDGER_PATH", tmp_path / "ddpg_runs" / "ledger.csv")
    yield


@pytest.fixture
def tiny_ddpg_experiment(tiny_experiment):
    return DDPGExperiment(
        num_fbs=1,
        env=tiny_experiment.env,
        ddpg=DDPGParams(
            total_timesteps=128, batch_size=32, buffer_size=1000,
            learning_starts=32, seed=7, verbose=0, log_flush_every=16,
        ),
    )


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
    ddpg_exp = DDPGExperiment.from_code("2-2-1", **kwargs)
    ppo_exp = ExperimentConfig.from_code("2-2-1", **kwargs)
    assert ddpg_exp.env == ppo_exp.env
    assert ddpg_exp.code == ppo_exp.code == "2-2-1"
    assert ddpg_exp.label == "2-2-1_mb_ddpg"


def test_config_roundtrip_and_defaults():
    exp = DDPGExperiment.from_code(
        "1-1-1", tag="probe", total_timesteps=5000, seed=3,
        ddpg_overrides={"noise_type": "ou", "action_noise_std": 0.2,
                        "binary_noise_std": 1.0, "net_arch": [64, 64]},
    )
    payload = json.loads(json.dumps(exp.to_json_dict()))
    assert payload["algo"] == "ddpg" and "ppo" not in payload and "td3" not in payload
    loaded = ddpg_experiment_from_json_dict(payload)
    assert loaded == exp
    assert loaded.ddpg.net_arch == (64, 64)
    assert loaded.label == "1-1-1_ddpg_probe"

    # Fields missing from an older payload fall back to DDPGParams defaults.
    payload["ddpg"].pop("ou_dt")
    assert ddpg_experiment_from_json_dict(payload).ddpg.ou_dt == 1e-2


def test_config_validation():
    with pytest.raises(ValueError):
        DDPGParams(noise_type="gaussian")
    with pytest.raises(ValueError):
        DDPGParams(action_noise_std=-0.1)
    with pytest.raises(ValueError):
        DDPGParams(binary_noise_std=-0.1)
    with pytest.raises(ValueError):
        DDPGParams(ou_dt=0.0)
    with pytest.raises(ValueError):
        DDPGParams(net_arch=())
    with pytest.raises(ValueError):
        ddpg_experiment_from_json_dict(ExperimentConfig.from_code("1-1-1").to_json_dict())


def test_ddpg_config_rejects_a_td3_payload():
    from td3.config import TD3Experiment

    with pytest.raises(ValueError):
        ddpg_experiment_from_json_dict(TD3Experiment.from_code("1-1-1").to_json_dict())


# ----------------------------------------------------------------------- #
# Per-dimension exploration noise
# ----------------------------------------------------------------------- #
@pytest.mark.parametrize("code,band", [
    ("1-1-1", None),
    ("2-2-1", BandConfig(mode="multi", fbs_band="agent", mbs_capacity="agent")),
    ("3-1-1", BandConfig(mode="multi", fbs_band="coverage")),
])
def test_binary_action_indices_match_the_env_layout(code, band):
    """A binary gene is exactly a state dim bounded to [0, 1] by the env."""
    from ppo.env import FlyingBaseStationEnv

    exp = DDPGExperiment.from_code(code, band=band)
    env = FlyingBaseStationEnv(exp.env, backend_kind="analytic")
    try:
        expected = [
            i for i in range(env.state_dim)
            if (env.state_low[i], env.state_high[i]) == (0.0, 1.0)
        ]
    finally:
        env.close()
    assert binary_action_indices(exp.env) == expected
    assert expected, "every scenario here has at least the FBS on/off gene"


def test_noise_sigma_is_flat_unless_binary_std_is_set():
    from ddpg.train import build_action_noise, noise_sigma

    exp = DDPGExperiment.from_code("1-1-1", ddpg_overrides={"action_noise_std": 0.1})
    np.testing.assert_allclose(noise_sigma(exp, 5), np.full(5, 0.1))

    exp = replace(exp, ddpg=replace(exp.ddpg, binary_noise_std=1.0))
    np.testing.assert_allclose(noise_sigma(exp, 5), [0.1, 0.1, 0.1, 0.1, 1.0])

    noise = build_action_noise(exp, 5)
    np.testing.assert_allclose(noise._sigma, [0.1, 0.1, 0.1, 0.1, 1.0])


def test_ou_noise_is_temporally_correlated():
    """Successive OU draws are correlated; successive Gaussian draws are not."""
    from ddpg.train import build_action_noise

    def lag1(kind, **over):
        exp = DDPGExperiment.from_code(
            "1-1-1", ddpg_overrides={"noise_type": kind, "action_noise_std": 0.5, **over})
        noise = build_action_noise(exp, 5)
        np.random.seed(0)
        draws = np.array([noise()[0] for _ in range(4000)])
        return np.corrcoef(draws[:-1], draws[1:])[0, 1]

    assert lag1("ou", ou_dt=1.0) > 0.5
    assert abs(lag1("normal")) < 0.1


def test_zero_noise_everywhere_disables_the_noise_object():
    from ddpg.train import build_action_noise

    exp = DDPGExperiment.from_code(
        "1-1-1", ddpg_overrides={"action_noise_std": 0.0, "binary_noise_std": 0.0})
    assert build_action_noise(exp, 5) is None


# ----------------------------------------------------------------------- #
# The model really is DDPG, not TD3
# ----------------------------------------------------------------------- #
def test_model_has_one_critic_and_no_td3_tricks(tiny_ddpg_experiment):
    from stable_baselines3.common.vec_env import DummyVecEnv

    from ddpg.train import build_model
    from ppo.env import FlyingBaseStationEnv

    env = FlyingBaseStationEnv(tiny_ddpg_experiment.env, backend_kind="analytic")
    try:
        model = build_model(tiny_ddpg_experiment, DummyVecEnv([lambda: env]))
        assert len(model.critic.q_networks) == 1          # twin critics off
        assert model.policy_delay == 1                    # delayed updates off
        assert model.target_noise_clip == 0.0             # target smoothing off
    finally:
        env.close()


# ----------------------------------------------------------------------- #
# Train / evaluate / resume / plots
# ----------------------------------------------------------------------- #
def test_train_writes_artifacts_and_its_own_ledger(tiny_ddpg_experiment, tmp_path):
    import ddpg.paths as dp
    import ppo.run_logging as ppo_rl

    run_dir = train(tiny_ddpg_experiment, backend="analytic", run_dir=tmp_path / "ddpgrun")
    for name in ("model.zip", "experiment_config.json", "steps.csv",
                 "episodes.csv", "monitor.csv", "progress.csv"):
        assert (run_dir / name).exists(), f"missing {name}"
    steps = pd.read_csv(run_dir / "steps.csv")
    assert len(steps) == tiny_ddpg_experiment.ddpg.total_timesteps
    assert "reward_objective" in steps.columns

    cfg = json.loads((run_dir / "experiment_config.json").read_text())
    assert cfg["algo"] == "ddpg" and cfg["provenance"]["status"] == "completed"
    # The resolved per-dimension noise is provenance, not just a flag.
    assert cfg["provenance"]["binary_action_dims"] == [4]
    assert cfg["provenance"]["action_noise_sigma"] == [0.1] * 5

    rows = list(csv.DictReader(dp.DDPG_LEDGER_PATH.open()))
    assert len(rows) == 1
    row = rows[0]
    assert row["algo"] == "ddpg" and row["noise_type"] == "normal"
    assert row["discount"] == str(tiny_ddpg_experiment.ddpg.gamma)
    for absent in ("n_steps", "ent_coef", "clip_range", "sac_ent_coef",
                   "policy_delay", "target_policy_noise", "target_noise_clip"):
        assert absent not in row
    assert not ppo_rl.LEDGER_PATH.exists(), "a DDPG run wrote to the PPO ledger"


def test_default_run_dir_is_under_ddpg_runs(tiny_ddpg_experiment):
    import ddpg.paths as dp

    run_dir = train(tiny_ddpg_experiment, backend="analytic")
    assert run_dir.parent == dp.DDPG_RUNS_DIR
    assert run_dir.name.endswith("_1-1-1_ddpg")


def test_evaluate_and_plot_a_trained_run(tiny_ddpg_experiment, tmp_path):
    from ddpg.plotting import training_gallery

    run_dir = train(tiny_ddpg_experiment, backend="analytic", run_dir=tmp_path / "ddpgrun")
    report = evaluate_run(run_dir, episodes=2, backend="analytic", make_plots=True)
    assert len(report.episodes) == 2
    for name in ("summary.csv", "eval_config.json", "ep00_trajectory.csv", "ep00_metrics.csv"):
        assert (report.eval_dir / name).exists(), f"missing {name}"
    assert (report.eval_dir / "plots" / "ep00_trajectory.png").exists()
    assert json.loads((report.eval_dir / "eval_config.json").read_text())["algo"] == "ddpg"

    out = training_gallery(run_dir)
    assert (out / "training_overview.png").exists()
    assert (out / "reward_components.png").exists()


def test_training_is_reproducible(tiny_ddpg_experiment, tmp_path):
    run_a = train(tiny_ddpg_experiment, backend="analytic", run_dir=tmp_path / "a")
    run_b = train(tiny_ddpg_experiment, backend="analytic", run_dir=tmp_path / "b")
    np.testing.assert_allclose(
        pd.read_csv(run_a / "steps.csv")["reward"].to_numpy(),
        pd.read_csv(run_b / "steps.csv")["reward"].to_numpy(),
    )


def test_resume_continues_training_and_rebuilds_noise(tiny_ddpg_experiment, tmp_path):
    first = train(tiny_ddpg_experiment, backend="analytic", run_dir=tmp_path / "first")
    exp2 = replace(
        tiny_ddpg_experiment,
        ddpg=replace(tiny_ddpg_experiment.ddpg, total_timesteps=256, binary_noise_std=1.0),
    )
    resumed = train(exp2, backend="analytic", run_dir=tmp_path / "resumed", resume_from=first)
    assert (resumed / "model.zip").exists()
    cfg = json.loads((resumed / "experiment_config.json").read_text())
    assert cfg["provenance"]["resumed_from"] == str(first)
    # The resumed leg explores with the new run's noise, not the source run's.
    assert cfg["provenance"]["action_noise_sigma"] == [0.1, 0.1, 0.1, 0.1, 1.0]
    assert len(pd.read_csv(resumed / "steps.csv")) > 0


def test_cli_train_eval_plot(tmp_path):
    import ddpg.paths as dp
    from ddpg.cli import main

    assert main([
        "train", "--backend", "analytic", "--timesteps", "96", "--max-episode-steps", "8",
        "--learning-starts", "16", "--batch-size", "16", "--buffer-size", "500",
        "--noise-type", "ou", "--action-noise-std", "0.2", "--binary-noise-std", "1.0",
        "--ou-dt", "1.0", "--net-arch", "32,32",
        "--normalize-obs", "--sticky-binaries", "--reward-shaping", "potential", "--tag", "cli",
    ]) == 0
    (run_dir,) = sorted(dp.DDPG_RUNS_DIR.glob("run_*_ddpg_cli"))
    cfg = json.loads((run_dir / "experiment_config.json").read_text())
    assert cfg["ddpg"]["noise_type"] == "ou" and cfg["ddpg"]["ou_dt"] == 1.0
    assert cfg["ddpg"]["binary_noise_std"] == 1.0
    assert cfg["ddpg"]["net_arch"] == [32, 32]
    assert cfg["env"]["reward"]["shaping"] == "potential"
    assert main(["eval", "--run", run_dir.name, "--episodes", "1",
                 "--backend", "analytic", "--no-plots"]) == 0
    assert main(["plot", "--run", "latest"]) == 0
    assert (run_dir / "plots" / "training_overview.png").exists()


# ----------------------------------------------------------------------- #
# Learning check (mirrors the TD3 and SAC structural-stack tests)
# ----------------------------------------------------------------------- #
def test_ddpg_beats_do_nothing_baseline(tmp_path):
    """On a weak-macro analytic world with coverage holes, a short seeded DDPG
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

    exp = DDPGExperiment(
        num_fbs=1, env=env_cfg,
        ddpg=DDPGParams(total_timesteps=2000, batch_size=64, buffer_size=5000,
                        learning_starts=200, gamma=0.95, seed=7, verbose=0,
                        log_flush_every=64),
    )
    run_dir = train(exp, backend="analytic", run_dir=tmp_path / "learn")
    report = evaluate_run(run_dir, episodes=3, backend="analytic", make_plots=False, save=False)
    finals = [float(r.metrics_df["reward_objective"].iloc[-1]) for r in report.episodes]
    assert np.mean(finals) > baseline + 0.10, (finals, baseline)
    assert max(finals) > baseline + 0.30, (finals, baseline)
