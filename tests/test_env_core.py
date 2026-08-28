import numpy as np
import pytest

from ppo.env import FlyingBaseStationEnv, state_labels
from tests.conftest import StubBackend


def make_env(cfg, stub_result):
    backend = StubBackend(cfg.world, cfg.band, stub_result)
    return FlyingBaseStationEnv(cfg, backend=backend), backend


# ----------------------------------------------------------------------- #
# Spaces / layout
# ----------------------------------------------------------------------- #
def test_legacy_spaces_match_original_layout(legacy_env_config, stub_result):
    env, _ = make_env(legacy_env_config, stub_result)
    n = legacy_env_config.num_fbs
    assert env.observation_space.shape == (5 * n + 2,)
    assert env.action_space.shape == (5 * n,)
    hi = env.observation_space.high
    assert hi[0] == legacy_env_config.world.width
    assert hi[1] == legacy_env_config.world.height
    assert hi[2] == 150.0 and hi[3] == np.float32(10.5) and hi[4] == 1.0
    lo = env.observation_space.low
    assert lo[2] == 20.0 and lo[3] == 7.0
    assert list(lo[-2:]) == [-1.0, -1.0]


def test_multi_spaces(multi_env_config, stub_result):
    env, _ = make_env(multi_env_config, stub_result)
    dim = 6 * 2 + 2  # 6 genes x 2 FBS + 2 MBS capacity genes
    assert env.action_space.shape == (dim,)
    assert env.observation_space.shape == (dim + 2,)
    labels = state_labels(multi_env_config)
    assert labels[5] == "fbs0_band_flag"
    assert labels[-1] == "mbs1_capacity"
    assert len(labels) == dim


# ----------------------------------------------------------------------- #
# Reset
# ----------------------------------------------------------------------- #
def test_reset_seeding_reproducible(legacy_env_config, stub_result):
    env, _ = make_env(legacy_env_config, stub_result)
    obs_a, _ = env.reset(seed=42)
    obs_b, _ = env.reset(seed=42)
    obs_c, _ = env.reset(seed=43)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert not np.array_equal(obs_a, obs_c)


def test_reset_custom_state(legacy_env_config, stub_result):
    env, _ = make_env(legacy_env_config, stub_result)
    state = np.array([100, 200, 50, 8.0, 1, 300, 400, 90, 9.0, 0], dtype=np.float32)
    obs, _ = env.reset(options={"state": state})
    np.testing.assert_array_equal(obs[:-2], state)
    np.testing.assert_array_equal(obs[-2:], [0.0, 0.0])
    with pytest.raises(ValueError):
        env.reset(options={"state": state[:-1]})


# ----------------------------------------------------------------------- #
# Step mechanics
# ----------------------------------------------------------------------- #
def test_step_clips_to_bounds_and_thresholds_binaries(legacy_env_config, stub_result):
    env, backend = make_env(legacy_env_config, stub_result)
    env.reset(seed=0)
    for _ in range(60):  # push hard toward upper bounds
        obs, *_ = env.step(np.ones(env.action_space.shape, dtype=np.float32))
    state = obs[:-2].reshape(2, 5)
    np.testing.assert_allclose(state[:, 0], legacy_env_config.world.width)
    np.testing.assert_allclose(state[:, 1], legacy_env_config.world.height)
    np.testing.assert_allclose(state[:, 2], 150.0)
    np.testing.assert_allclose(state[:, 3], 10.5)
    np.testing.assert_allclose(state[:, 4], 1.0)  # action >= 0 -> on

    action = np.ones(env.action_space.shape, dtype=np.float32)
    action[4] = -0.01  # fbs0 power_status off
    obs, *_ = env.step(action)
    assert obs[4] == 0.0
    assert backend.calls[-1]["power_status"][0] == 0.0


def test_step_passes_band_and_capacity_to_backend(multi_env_config, stub_result):
    env, backend = make_env(multi_env_config, stub_result)
    env.reset(seed=1)
    action = -np.ones(env.action_space.shape, dtype=np.float32)
    action[5] = 0.5    # fbs0 band -> capacity
    action[-1] = 0.5   # mbs1 capacity -> on
    env.step(action)
    call = backend.calls[-1]
    np.testing.assert_array_equal(call["fbs_band_flags"], [1.0, 0.0])
    np.testing.assert_array_equal(call["mbs_capacity_genes"], [0.0, 1.0])


def test_legacy_backend_gets_no_band_args(legacy_env_config, stub_result):
    env, backend = make_env(legacy_env_config, stub_result)
    env.reset(seed=1)
    env.step(env.action_space.sample())
    call = backend.calls[-1]
    assert call["fbs_band_flags"] is None
    assert call["mbs_capacity_genes"] is None


# ----------------------------------------------------------------------- #
# Reward math
# ----------------------------------------------------------------------- #
def test_legacy_blend_reward_matches_hand_computation(legacy_env_config, stub_result):
    env, _ = make_env(legacy_env_config, stub_result)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.zeros(env.action_space.shape, np.float32))

    w = legacy_env_config.reward.weights
    r = stub_result
    max_users = legacy_env_config.world.num_users
    norm_users = min(r.total_connected / max_users, 1.0)
    norm_power = min(r.total_power / (10.5 * 2), 1.0)
    base = (norm_users + w.epsilon) ** w.beta * ((1 - norm_power) + w.epsilon) ** w.gamma
    fbs_term = (r.fbs_connected / r.total_connected + w.epsilon) ** w.fbs_exponent
    expected = (1 - w.fbs_weight) * base + w.fbs_weight * fbs_term
    assert reward == pytest.approx(expected, rel=1e-6)
    assert info["reward_norm_users"] == pytest.approx(norm_users)


def test_ga_blend_reward_uses_controlled_power(multi_env_config, stub_result):
    env, _ = make_env(multi_env_config, stub_result)
    env.reset(seed=0)
    # All continuous genes drift; binaries: fbs0 on/coverage, fbs1 off,
    # both MBS capacity genes on.
    action = np.zeros(env.action_space.shape, np.float32)
    action[4] = 1.0    # fbs0 on
    action[5] = -1.0   # fbs0 coverage
    action[10] = -1.0  # fbs1 off
    action[12] = 1.0   # mbs0 capacity on
    action[13] = 1.0   # mbs1 capacity on
    _, reward, _, _, info = env.step(action)

    state = info["state"].reshape(-1)
    fbs0_power = state[3]
    w = multi_env_config.reward.weights
    r = stub_result
    controlled = fbs0_power + 2 * multi_env_config.world.mbs_power  # fbs1 off
    max_total = 10.5 * 2 + multi_env_config.world.mbs_power * 2
    norm_users = min(r.total_connected / multi_env_config.world.num_users, 1.0)
    norm_power = min(controlled / max_total, 1.0)
    base = (norm_users ** w.beta) * ((1 - norm_power) ** w.gamma)
    fbs_term = (r.fbs_connected / r.total_connected) ** w.fbs_exponent
    expected = (1 - w.fbs_weight) * base + w.fbs_weight * fbs_term
    assert reward == pytest.approx(expected, rel=1e-6)
    assert info["num_active_capacity_carriers"] == 2


def test_sum_rate_reward(legacy_env_config, stub_result):
    from dataclasses import replace

    cfg = replace(
        legacy_env_config,
        reward=replace(legacy_env_config.reward, mode="sum_rate"),
    )
    env, _ = make_env(cfg, stub_result)
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(np.zeros(env.action_space.shape, np.float32))
    assert reward == pytest.approx(stub_result.sum_rate)


# ----------------------------------------------------------------------- #
# Termination / truncation / deltas
# ----------------------------------------------------------------------- #
def test_truncation_and_termination(legacy_env_config, stub_result):
    env, _ = make_env(legacy_env_config, stub_result)
    env.reset(seed=0)
    for step in range(legacy_env_config.max_episode_steps):
        _, _, terminated, truncated, _ = env.step(
            np.zeros(env.action_space.shape, np.float32)
        )
        assert not terminated
    assert truncated

    full = stub_result
    full.total_connected = legacy_env_config.world.num_users
    env.reset(seed=0)
    _, _, terminated, _, _ = env.step(np.zeros(env.action_space.shape, np.float32))
    assert terminated
    full.total_connected = 120  # restore shared fixture object


def test_delta_observations_track_changes(legacy_env_config, stub_result):
    env, _ = make_env(legacy_env_config, stub_result)
    env.reset(seed=0)
    obs, _, _, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
    max_users = legacy_env_config.world.num_users
    assert info["total_delta"] == pytest.approx(stub_result.total_connected / max_users)
    # second step: same result -> zero delta
    obs, _, _, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
    assert info["total_delta"] == pytest.approx(0.0)
    assert obs[-1] == pytest.approx(0.0)
