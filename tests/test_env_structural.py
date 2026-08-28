"""Structural training options: obs normalization, sticky binaries,
potential-based reward shaping, tier-metric observations, termination flag.

Every option defaults OFF (legacy behavior); each test asserts both the new
semantics and, where relevant, that the default path is unchanged.
"""
from dataclasses import replace

import numpy as np
import pytest

from ppo.config import RewardConfig
from ppo.env import FlyingBaseStationEnv
from ppo.matlab_bridge import SinrResult
from tests.conftest import StubBackend


def make_env(cfg, stub_result):
    backend = StubBackend(cfg.world, cfg.band, stub_result)
    return FlyingBaseStationEnv(cfg, backend=backend), backend


class SequenceBackend:
    """StubBackend variant returning a different result per call."""

    def __init__(self, world, band, results):
        self.world, self.band = world, band
        self.results = list(results)
        self.n_calls = 0
        self.mbs_x = np.array([world.width / 2])
        self.mbs_y = np.array([world.height / 2])

    def evaluate(self, *args, **kwargs):
        r = self.results[min(self.n_calls, len(self.results) - 1)]
        self.n_calls += 1
        return r

    def close(self):
        pass


def result_with(total, fbs, power=12.0):
    return SinrResult(
        total_connected=total, fbs_connected=fbs, mbs_connected=total - fbs,
        mbs_coverage_connected=total - fbs, mbs_capacity_connected=0,
        total_power=power, avg_rate=3.0, sum_rate=3.0 * total,
    )


# ----------------------------------------------------------------------- #
# Observation normalization
# ----------------------------------------------------------------------- #
def test_normalize_obs_space_and_values(legacy_env_config, stub_result):
    cfg = replace(legacy_env_config, normalize_obs=True)
    env, _ = make_env(cfg, stub_result)
    assert env.observation_space.shape == (cfg.obs_dim,)
    np.testing.assert_array_equal(env.observation_space.low, -np.ones(cfg.obs_dim))
    np.testing.assert_array_equal(env.observation_space.high, np.ones(cfg.obs_dim))

    obs, _ = env.reset(seed=3)
    state = env.state  # physical units
    expected = 2.0 * (state - env.state_low) / (env.state_high - env.state_low) - 1.0
    np.testing.assert_allclose(obs[:-2], expected, rtol=1e-5, atol=1e-6)
    assert np.all(obs >= -1.0 - 1e-6) and np.all(obs <= 1.0 + 1e-6)

    obs, *_ = env.step(np.zeros(env.action_space.shape, np.float32))
    assert np.all(np.abs(obs) <= 1.0 + 1e-6)
    # internal state stays physical
    assert env.state[0] > 1.0 or env.state[1] > 1.0


def test_default_obs_is_unnormalized_physical_state(legacy_env_config, stub_result):
    env, _ = make_env(legacy_env_config, stub_result)
    obs, _ = env.reset(seed=3)
    np.testing.assert_array_equal(obs[:-2], env.state)
    np.testing.assert_array_equal(obs, env.flat_state_with_globals())


# ----------------------------------------------------------------------- #
# Sticky binaries
# ----------------------------------------------------------------------- #
def test_sticky_binaries_hold_below_half(legacy_env_config, stub_result):
    cfg = replace(legacy_env_config, binary_mode="sticky")
    env, backend = make_env(cfg, stub_result)
    state = np.array([100, 200, 50, 8.0, 1, 300, 400, 90, 9.0, 0], dtype=np.float32)
    env.reset(options={"state": state})

    action = np.zeros(env.action_space.shape, np.float32)
    action[4] = -0.4   # |a| < 0.5 -> fbs0 stays ON
    action[9] = 0.4    # |a| < 0.5 -> fbs1 stays OFF
    env.step(action)
    np.testing.assert_array_equal(backend.calls[-1]["power_status"], [1.0, 0.0])

    action[4] = -0.9   # decisive -> fbs0 OFF
    action[9] = 0.9    # decisive -> fbs1 ON
    env.step(action)
    np.testing.assert_array_equal(backend.calls[-1]["power_status"], [0.0, 1.0])


def test_sticky_applies_to_band_and_capacity_genes(multi_env_config, stub_result):
    cfg = replace(multi_env_config, binary_mode="sticky")
    env, backend = make_env(cfg, stub_result)
    state = np.zeros(cfg.state_dim, dtype=np.float32)
    state[:4] = [100, 200, 50, 8.0]
    state[6:10] = [300, 400, 90, 9.0]
    state[5] = 1.0    # fbs0 band = capacity
    state[-1] = 1.0   # mbs1 capacity on
    env.reset(options={"state": state})

    action = np.zeros(env.action_space.shape, np.float32)
    action[5] = -0.2   # indecisive -> band flag holds 1
    action[-1] = -0.2  # indecisive -> capacity gene holds 1
    env.step(action)
    call = backend.calls[-1]
    assert call["fbs_band_flags"][0] == 1.0
    assert call["mbs_capacity_genes"][-1] == 1.0

    action[5] = -0.8
    action[-1] = -0.8
    env.step(action)
    call = backend.calls[-1]
    assert call["fbs_band_flags"][0] == 0.0
    assert call["mbs_capacity_genes"][-1] == 0.0


def test_threshold_mode_still_flips_at_zero(legacy_env_config, stub_result):
    env, backend = make_env(legacy_env_config, stub_result)
    state = np.array([100, 200, 50, 8.0, 1, 300, 400, 90, 9.0, 0], dtype=np.float32)
    env.reset(options={"state": state})
    action = np.zeros(env.action_space.shape, np.float32)
    action[4] = -0.01  # legacy: any negative flips off
    env.step(action)
    np.testing.assert_array_equal(backend.calls[-1]["power_status"], [0.0, 1.0])


# ----------------------------------------------------------------------- #
# Potential-based shaping
# ----------------------------------------------------------------------- #
def test_potential_shaping_telescopes(legacy_env_config):
    cfg = replace(
        legacy_env_config,
        reward=replace(legacy_env_config.reward, shaping="potential"),
    )
    results = [result_with(100 + 10 * k, 10 * k) for k in range(8)]
    backend = SequenceBackend(cfg.world, cfg.band, results)
    env = FlyingBaseStationEnv(cfg, backend=backend)

    _, reset_info = env.reset(seed=0)
    assert backend.n_calls == 1          # exactly one reset evaluation
    phi0 = reset_info["objective"]

    rewards, last_phi = [], None
    for _ in range(5):
        _, r, _, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
        rewards.append(r)
        last_phi = info["reward_objective"]
    assert abs(sum(rewards) - (last_phi - phi0)) < 1e-6


def test_potential_shaping_zero_when_static(legacy_env_config, stub_result):
    cfg = replace(
        legacy_env_config,
        reward=replace(legacy_env_config.reward, shaping="potential"),
    )
    env, backend = make_env(cfg, stub_result)  # constant result every call
    _, reset_info = env.reset(seed=0)
    assert "objective" in reset_info
    for _ in range(3):
        _, r, *_ = env.step(np.zeros(env.action_space.shape, np.float32))
        assert abs(r) < 1e-7


def test_no_reset_eval_by_default(legacy_env_config, stub_result):
    env, backend = make_env(legacy_env_config, stub_result)
    _, reset_info = env.reset(seed=0)
    assert backend.calls == []           # legacy reset never touches the backend
    assert reset_info == {}


def test_reset_eval_fixes_first_step_deltas(legacy_env_config, stub_result):
    """With a reset evaluation, an unchanged world yields zero deltas on the
    first step (legacy behavior: delta = frac - 0, a spurious jump)."""
    cfg = replace(
        legacy_env_config,
        reward=replace(legacy_env_config.reward, shaping="potential"),
    )
    env, _ = make_env(cfg, stub_result)
    env.reset(seed=0)
    obs, *_ , info = env.step(np.zeros(env.action_space.shape, np.float32))
    assert info["fbs_delta"] == 0.0 and info["total_delta"] == 0.0

    legacy_env, _ = make_env(legacy_env_config, stub_result)
    legacy_env.reset(seed=0)
    _, _, _, _, legacy_info = legacy_env.step(np.zeros(legacy_env.action_space.shape, np.float32))
    assert legacy_info["total_delta"] > 0.0  # 120/200 - 0


# ----------------------------------------------------------------------- #
# Tier-metric observations
# ----------------------------------------------------------------------- #
def test_tier_metrics_appended_to_obs(legacy_env_config, stub_result):
    cfg = replace(legacy_env_config, obs_tier_metrics=True)
    env, _ = make_env(cfg, stub_result)
    assert env.observation_space.shape == (cfg.obs_dim,)
    assert cfg.obs_dim == cfg.state_dim + 2 + 4

    m = cfg.world.num_users
    expected_norm_power = min(stub_result.total_power / (10.5 * 2), 1.0)
    expected = [40 / m, 80 / m, 120 / m, expected_norm_power]

    obs, _ = env.reset(seed=0)
    # reset eval ran (obs_tier_metrics implies needs_reset_eval)
    np.testing.assert_allclose(obs[-4:], expected, rtol=1e-5)

    obs, *_ = env.step(np.zeros(env.action_space.shape, np.float32))
    np.testing.assert_allclose(obs[-4:], expected, rtol=1e-5)


def test_objective_always_logged(legacy_env_config, stub_result):
    env, _ = make_env(legacy_env_config, stub_result)
    env.reset(seed=0)
    _, r, _, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
    assert info["reward_objective"] == pytest.approx(r)  # shaping='none': reward == Φ


# ----------------------------------------------------------------------- #
# Full-coverage termination flag
# ----------------------------------------------------------------------- #
def test_termination_flag(legacy_env_config):
    full = result_with(legacy_env_config.world.num_users, 30)

    env, _ = make_env(legacy_env_config, full)
    env.reset(seed=0)
    _, _, terminated, _, _ = env.step(np.zeros(env.action_space.shape, np.float32))
    assert terminated  # legacy default early-stops

    cfg = replace(legacy_env_config, terminate_on_full_coverage=False)
    env, _ = make_env(cfg, full)
    env.reset(seed=0)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(
            np.zeros(env.action_space.shape, np.float32)
        )
        steps += 1
    assert not terminated and truncated
    assert steps == cfg.max_episode_steps


# ----------------------------------------------------------------------- #
# Absolute actions
# ----------------------------------------------------------------------- #
def test_absolute_action_maps_range_endpoints(legacy_env_config, stub_result):
    cfg = replace(legacy_env_config, action_mode="absolute")
    env, backend = make_env(cfg, stub_result)
    env.reset(seed=0)
    lo, hi = env.param_lower_single[:4], env.param_upper_single[:4]

    env.step(np.full(env.action_space.shape, -1.0, np.float32))
    np.testing.assert_allclose(backend.calls[-1]["cont_params"][0], lo, rtol=1e-5)

    env.step(np.full(env.action_space.shape, 1.0, np.float32))
    np.testing.assert_allclose(backend.calls[-1]["cont_params"][0], hi, rtol=1e-5)

    env.step(np.zeros(env.action_space.shape, np.float32))
    np.testing.assert_allclose(
        backend.calls[-1]["cont_params"][0], (lo + hi) / 2.0, rtol=1e-5
    )


def test_absolute_is_memoryless_and_ignores_action_scale(legacy_env_config, stub_result):
    """Same action -> same genes regardless of history or action_scale."""
    a = np.array([0.4, -0.6, 0.2, 0.9, 1.0, -0.3, 0.5, -0.8, 0.1, 1.0], np.float32)
    seen = []
    for scale in (0.05, 0.9):
        cfg = replace(legacy_env_config, action_mode="absolute", action_scale=scale)
        env, backend = make_env(cfg, stub_result)
        env.reset(seed=0)
        env.step(np.zeros(env.action_space.shape, np.float32))  # arbitrary history
        env.step(a)
        seen.append(backend.calls[-1]["cont_params"].copy())
    np.testing.assert_allclose(seen[0], seen[1], rtol=1e-6)


def test_delta_mode_unchanged_by_default(legacy_env_config, stub_result):
    assert legacy_env_config.action_mode == "delta"
    env, backend = make_env(legacy_env_config, stub_result)
    state = np.array([100, 200, 50, 8.0, 1, 300, 400, 90, 9.0, 0], dtype=np.float32)
    env.reset(options={"state": state})
    env.step(np.full(env.action_space.shape, 1.0, np.float32))
    scale = legacy_env_config.action_scale
    rng = env.param_upper_single[:4] - env.param_lower_single[:4]
    expected = np.clip(state[:4] + scale * rng,
                       env.param_lower_single[:4], env.param_upper_single[:4])
    np.testing.assert_allclose(backend.calls[-1]["cont_params"][0], expected, rtol=1e-5)


def test_absolute_with_sticky_binaries_warns(legacy_env_config, stub_result):
    cfg = replace(legacy_env_config, action_mode="absolute", binary_mode="sticky")
    with pytest.warns(RuntimeWarning, match="absolute"):
        make_env(cfg, stub_result)


# ----------------------------------------------------------------------- #
# Reward rescaling
# ----------------------------------------------------------------------- #
def test_gain_and_offset_shift_reward_but_not_objective(legacy_env_config, stub_result):
    base_env, _ = make_env(legacy_env_config, stub_result)
    base_env.reset(seed=0)
    _, r_raw, _, _, info_raw = base_env.step(np.zeros(base_env.action_space.shape, np.float32))

    cfg = replace(
        legacy_env_config,
        reward=replace(legacy_env_config.reward, reward_offset=0.4480, reward_gain=13.6),
    )
    env, _ = make_env(cfg, stub_result)
    env.reset(seed=0)
    _, r_scaled, _, _, info = env.step(np.zeros(env.action_space.shape, np.float32))

    assert r_scaled == pytest.approx((r_raw - 0.4480) * 13.6)
    # the logged objective is the RAW value in both cases
    assert info["reward_objective"] == pytest.approx(info_raw["reward_objective"])


def test_offset_cancels_under_potential_shaping(legacy_env_config):
    results = [result_with(100 + 10 * k, 10 * k) for k in range(8)]

    def run(offset, gain):
        cfg = replace(
            legacy_env_config,
            reward=replace(legacy_env_config.reward, shaping="potential",
                           reward_offset=offset, reward_gain=gain),
        )
        env = FlyingBaseStationEnv(cfg, backend=SequenceBackend(cfg.world, cfg.band, results))
        env.reset(seed=0)
        return [env.step(np.zeros(env.action_space.shape, np.float32))[1] for _ in range(4)]

    plain = run(0.0, 1.0)
    offset_only = run(0.4480, 1.0)
    gained = run(0.4480, 10.0)
    np.testing.assert_allclose(plain, offset_only, atol=1e-7)      # offset cancels
    np.testing.assert_allclose(np.array(plain) * 10.0, gained, rtol=1e-5)


# ----------------------------------------------------------------------- #
# Record-based reward
# ----------------------------------------------------------------------- #
def test_record_reward_sums_to_best_minus_start(legacy_env_config):
    # rises, then falls: the fall must contribute nothing
    seq = [100, 140, 200, 160, 120, 150]
    results = [result_with(u, u // 10) for u in seq]
    cfg = replace(legacy_env_config,
                  reward=replace(legacy_env_config.reward, shaping="record"))
    backend = SequenceBackend(cfg.world, cfg.band, results)
    env = FlyingBaseStationEnv(cfg, backend=backend)
    _, reset_info = env.reset(seed=0)
    phi0 = reset_info["objective"]

    rewards, objs = [], []
    for _ in range(5):
        _, r, _, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
        rewards.append(r)
        objs.append(info["reward_objective"])

    assert all(r >= 0.0 for r in rewards)               # never punished
    assert rewards[3] == 0.0 and rewards[4] == 0.0      # declining steps pay nothing
    assert sum(rewards) == pytest.approx(max(objs) - phi0, abs=1e-7)


def test_record_needs_reset_eval(legacy_env_config, stub_result):
    cfg = replace(legacy_env_config,
                  reward=replace(legacy_env_config.reward, shaping="record"))
    assert cfg.needs_reset_eval
    env, backend = make_env(cfg, stub_result)
    _, info = env.reset(seed=0)
    assert len(backend.calls) == 1 and "objective" in info


# --------------------------------------------------------------------------- #
# Hybrid potential + record shaping
# --------------------------------------------------------------------------- #
def test_potential_record_telescopes(legacy_env_config):
    """Return == (Φ_T − Φ_0) + w·(max_t Φ_t − Φ_0), for a path that rises then falls."""
    w = 0.5
    cfg = replace(
        legacy_env_config,
        reward=replace(legacy_env_config.reward, shaping="potential_record",
                       record_weight=w),
    )
    # users rise for 4 steps then fall, so the final state is NOT the best one
    users = [100, 200, 300, 400, 250, 150]
    results = [result_with(u, u // 10) for u in users]
    backend = SequenceBackend(cfg.world, cfg.band, results)
    env = FlyingBaseStationEnv(cfg, backend=backend)

    _, reset_info = env.reset(seed=0)
    phi0 = reset_info["objective"]

    rewards, phis = [], []
    for _ in range(len(users) - 1):
        _, r, _, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
        rewards.append(r)
        phis.append(info["reward_objective"])

    expected = (phis[-1] - phi0) + w * (max(phis) - phi0)
    assert sum(rewards) == pytest.approx(expected, abs=1e-6)
    assert phis[-1] < max(phis)          # the path really did leave its best


def test_potential_record_equals_potential_when_weight_zero(legacy_env_config):
    """record_weight=0 must reproduce plain potential shaping exactly."""
    results = [result_with(100 + 10 * k, 10 * k) for k in range(8)]

    def run(shaping, weight):
        cfg = replace(
            legacy_env_config,
            reward=replace(legacy_env_config.reward, shaping=shaping, record_weight=weight),
        )
        env = FlyingBaseStationEnv(cfg, backend=SequenceBackend(cfg.world, cfg.band, results))
        env.reset(seed=0)
        return [env.step(np.zeros(env.action_space.shape, np.float32))[1] for _ in range(5)]

    np.testing.assert_allclose(run("potential", 0.0), run("potential_record", 0.0), atol=1e-7)


def test_potential_record_rejects_negative_weight():
    with pytest.raises(ValueError, match="record_weight"):
        RewardConfig(shaping="potential_record", record_weight=-0.1)
