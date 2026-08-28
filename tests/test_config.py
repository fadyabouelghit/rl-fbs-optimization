import json

import pytest

from ppo.config import (
    BandConfig,
    ExperimentConfig,
    RewardConfig,
    experiment_from_json_dict,
)


def test_from_code_parses_axes():
    exp = ExperimentConfig.from_code("2-2-1", total_timesteps=500, seed=3)
    assert (exp.num_fbs, exp.scenario_id, exp.config_id) == (2, 2, 1)
    assert exp.code == "2-2-1"
    assert exp.env.world.num_mbs == 2
    assert exp.env.world.width == 4000.0
    assert exp.ppo.total_timesteps == 500
    assert exp.ppo.seed == 3
    assert exp.env.reward.weights.gamma == 0.0  # config_id 1


def test_from_code_rejects_unknown_ids():
    with pytest.raises(ValueError):
        ExperimentConfig.from_code("1-9-1")
    with pytest.raises(ValueError):
        ExperimentConfig.from_code("1-1-9")


def test_band_validation():
    with pytest.raises(ValueError):
        BandConfig(mode="legacy", fbs_band="agent")
    with pytest.raises(ValueError):
        BandConfig(mode="nope")
    b = BandConfig(mode="multi", fbs_band="agent", mbs_capacity="on")
    assert b.agent_controls_fbs_band and not b.agent_controls_mbs_capacity


def test_reward_mode_validation():
    with pytest.raises(ValueError):
        RewardConfig(mode="bogus")


def test_env_dims_legacy(legacy_env_config):
    cfg = legacy_env_config
    assert cfg.genes_per_fbs == 5
    assert cfg.state_dim == 10
    assert cfg.obs_dim == 12
    assert cfg.action_dim == 10


def test_env_dims_multi(multi_env_config):
    cfg = multi_env_config
    assert cfg.genes_per_fbs == 6
    assert cfg.state_dim == 6 * 2 + 2  # +1 capacity gene per MBS
    assert cfg.obs_dim == cfg.state_dim + 2


def test_json_roundtrip_v2(multi_env_config):
    exp = ExperimentConfig.from_code("2-2-2", band=multi_env_config.band, tag="rt")
    payload = json.loads(json.dumps(exp.to_json_dict()))
    loaded = experiment_from_json_dict(payload)
    assert loaded.to_json_dict() == exp.to_json_dict()
    assert loaded.env.band.agent_controls_fbs_band


def test_v1_schema_loads():
    v1 = {
        "num_fbs": 2, "scenario_id": 2, "config_id": 2,
        "ent_coef": 0.7, "action_scale": 0.9, "learning_rate": 1e-4,
        "total_timesteps": 3000, "max_episode_steps": 30,
        "beta": 0.8, "fbs_weight": 0.4, "fbs_exponent": 1.0,
        "code": "2-2-2",
    }
    exp = experiment_from_json_dict(v1)
    assert exp.code == "2-2-2"
    assert exp.env.band.mode == "legacy"
    assert exp.env.reward.weights.gamma == 0.1  # config_id 2
    assert exp.env.action_scale == 0.9
    assert exp.ppo.total_timesteps == 3000


# ----------------------------------------------------------------------- #
# Structural fields (config v2 stays backward-compatible)
# ----------------------------------------------------------------------- #
def test_structural_fields_roundtrip():
    from ppo.config import EnvConfig, PPOParams, RewardWeights

    exp = ExperimentConfig.from_code(
        "2-1-1",
        reward=RewardConfig(mode="legacy_blend", shaping="potential"),
        env_overrides={
            "normalize_obs": True,
            "binary_mode": "sticky",
            "obs_tier_metrics": True,
            "terminate_on_full_coverage": False,
        },
        ppo_overrides={"algo": "sac", "buffer_size": 12345, "sac_ent_coef": 0.05},
    )
    loaded = experiment_from_json_dict(json.loads(json.dumps(exp.to_json_dict())))
    assert loaded.env.normalize_obs is True
    assert loaded.env.binary_mode == "sticky"
    assert loaded.env.obs_tier_metrics is True
    assert loaded.env.terminate_on_full_coverage is False
    assert loaded.env.reward.shaping == "potential"
    assert loaded.ppo.algo == "sac"
    assert loaded.ppo.buffer_size == 12345
    assert loaded.ppo.sac_ent_coef == 0.05
    assert loaded.to_json_dict() == exp.to_json_dict()
    assert loaded.label.startswith("2-1-1_sac")


def test_pre_structural_v2_config_loads_with_legacy_defaults():
    """A v2 config written before the structural fields existed must load
    with every new behavior switched off (bit-compatible legacy env)."""
    exp = ExperimentConfig.from_code("2-1-1")
    payload = exp.to_json_dict()
    for key in ("normalize_obs", "binary_mode", "obs_tier_metrics",
                "terminate_on_full_coverage"):
        payload["env"].pop(key, None)
    payload["env"]["reward"].pop("shaping", None)
    for key in ("algo", "buffer_size", "learning_starts", "tau",
                "train_freq", "gradient_steps", "sac_ent_coef"):
        payload["ppo"].pop(key, None)

    loaded = experiment_from_json_dict(json.loads(json.dumps(payload)))
    assert loaded.env.normalize_obs is False
    assert loaded.env.binary_mode == "threshold"
    assert loaded.env.obs_tier_metrics is False
    assert loaded.env.terminate_on_full_coverage is True
    assert loaded.env.reward.shaping == "none"
    assert loaded.env.needs_reset_eval is False
    assert loaded.env.obs_dim == exp.env.state_dim + 2
    assert loaded.ppo.algo == "ppo"
    assert loaded.label == "2-1-1"


def test_algo_and_structural_validation():
    from ppo.config import EnvConfig, PPOParams

    with pytest.raises(ValueError):
        PPOParams(algo="ddpg")
    with pytest.raises(ValueError):
        EnvConfig(binary_mode="fuzzy")
    with pytest.raises(ValueError):
        RewardConfig(shaping="delta")
    # PPO's batch/rollout constraint does not apply to SAC
    with pytest.raises(ValueError):
        PPOParams(algo="ppo", batch_size=512, n_steps=64)
    p = PPOParams(algo="sac", batch_size=512, n_steps=64)
    assert p.algo == "sac"


def test_action_mode_and_reward_scaling_roundtrip():
    from ppo.config import EnvConfig, PPOParams

    exp = ExperimentConfig.from_code(
        "2-1-1",
        reward=RewardConfig(mode="legacy_blend", shaping="record",
                            reward_offset=0.4480, reward_gain=13.6),
        env_overrides={"action_mode": "absolute", "normalize_obs": True},
    )
    loaded = experiment_from_json_dict(json.loads(json.dumps(exp.to_json_dict())))
    assert loaded.env.action_mode == "absolute"
    assert loaded.env.reward.shaping == "record"
    assert loaded.env.reward.reward_offset == 0.4480
    assert loaded.env.reward.reward_gain == 13.6
    assert loaded.env.needs_reset_eval          # record needs phi(s_0)
    assert loaded.to_json_dict() == exp.to_json_dict()


def test_configs_without_new_fields_keep_legacy_defaults():
    exp = ExperimentConfig.from_code("2-1-1")
    payload = exp.to_json_dict()
    payload["env"].pop("action_mode", None)
    for k in ("shaping", "reward_offset", "reward_gain"):
        payload["env"]["reward"].pop(k, None)
    loaded = experiment_from_json_dict(json.loads(json.dumps(payload)))
    assert loaded.env.action_mode == "delta"
    assert loaded.env.reward.shaping == "none"
    assert loaded.env.reward.reward_offset == 0.0
    assert loaded.env.reward.reward_gain == 1.0
    assert loaded.env.needs_reset_eval is False


def test_new_field_validation():
    from ppo.config import EnvConfig

    with pytest.raises(ValueError):
        EnvConfig(action_mode="teleport")
    with pytest.raises(ValueError):
        RewardConfig(shaping="best")
