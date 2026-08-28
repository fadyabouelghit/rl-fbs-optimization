import numpy as np

from ppo.config import BandConfig, EnvConfig, RewardConfig, RewardWeights
from ppo.env import FlyingBaseStationEnv
from ppo.evaluate import flatten_states, rollout_episode
from ppo.matlab_bridge import AnalyticSinrBackend
from tests.conftest import small_world


class RandomPolicy:
    """Stand-in for an SB3 model in rollout tests."""

    def __init__(self, action_space, seed=0):
        self.action_space = action_space
        self.action_space.seed(seed)

    def predict(self, obs, deterministic=True):
        return self.action_space.sample(), None


def test_flatten_states_legacy(legacy_env_config):
    obs_dim = legacy_env_config.obs_dim
    states = [np.arange(obs_dim, dtype=float) + k for k in range(4)]
    df = flatten_states(states, legacy_env_config)
    assert df.shape == (4, obs_dim)
    assert ("fbs1", "power_status") in df.columns
    assert ("global", "total_delta") in df.columns
    assert df[("fbs0", "x")].iloc[2] == 2.0


def test_flatten_states_multi(multi_env_config):
    obs_dim = multi_env_config.obs_dim
    states = [np.zeros(obs_dim)]
    df = flatten_states(states, multi_env_config)
    assert ("fbs0", "band_flag") in df.columns
    assert ("mbs1", "capacity") in df.columns
    assert df.shape == (1, obs_dim)


def test_rollout_episode_collects_everything(multi_env_config):
    from dataclasses import replace

    cfg = replace(multi_env_config, collect_user_positions=True)
    backend = AnalyticSinrBackend(cfg.world, cfg.band)
    env = FlyingBaseStationEnv(cfg, backend=backend)
    policy = RandomPolicy(env.action_space)

    r = rollout_episode(policy, env, index=3, seed=5)
    assert r.index == 3
    assert len(r.state_df) == len(r.metrics_df) + 1  # includes reset state
    assert 1 <= r.summary["length"] <= cfg.max_episode_steps
    if r.summary["length"] < cfg.max_episode_steps:
        assert r.summary["terminated"]  # early stop == full connectivity
    assert r.user_positions is not None
    assert "reward" in r.metrics_df.columns
    assert "mbs_capacity_connected" in r.metrics_df.columns
    assert r.summary["final_total_connected"] is not None


def test_rollout_pinned_initial_state():
    world = small_world()
    cfg = EnvConfig(
        num_fbs=1,
        world=world,
        band=BandConfig(),
        reward=RewardConfig(weights=RewardWeights()),
        max_episode_steps=3,
    )
    env = FlyingBaseStationEnv(cfg, backend=AnalyticSinrBackend(world, cfg.band))
    policy = RandomPolicy(env.action_space)
    init = np.array([500.0, 400.0, 100.0, 10.5, 1.0], dtype=np.float32)
    r = rollout_episode(policy, env, initial_state=init, seed=0)
    np.testing.assert_array_equal(
        r.state_df.iloc[0][[("fbs0", "x"), ("fbs0", "y"), ("fbs0", "height")]].values,
        [500.0, 400.0, 100.0],
    )


def test_trajectory_stays_physical_with_normalized_obs():
    from dataclasses import replace as _replace

    world = small_world()
    cfg = EnvConfig(
        num_fbs=1,
        world=world,
        band=BandConfig(),
        reward=RewardConfig(weights=RewardWeights(), shaping="potential"),
        max_episode_steps=3,
        normalize_obs=True,
        obs_tier_metrics=True,
        binary_mode="sticky",
    )
    env = FlyingBaseStationEnv(cfg, backend=AnalyticSinrBackend(world, cfg.band))
    policy = RandomPolicy(env.action_space)
    init = np.array([500.0, 400.0, 100.0, 10.5, 1.0], dtype=np.float32)
    r = rollout_episode(policy, env, initial_state=init, seed=0)
    # trajectory CSV values are physical units, not the [-1, 1] observation
    np.testing.assert_array_equal(
        r.state_df.iloc[0][[("fbs0", "x"), ("fbs0", "y"), ("fbs0", "height")]].values,
        [500.0, 400.0, 100.0],
    )
    assert ("global", "fbs_frac") in r.state_df.columns
    assert ("global", "norm_power") in r.state_df.columns
    fracs = r.state_df[("global", "total_frac")].to_numpy()
    assert np.all((fracs >= 0.0) & (fracs <= 1.0))
