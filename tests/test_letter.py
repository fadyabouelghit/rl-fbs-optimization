"""Letter instrumentation and the hybrid remedy (letterexp/, ddpg_hybrid/, td3_hybrid/).

Fast, analytic backend only. The guarantees that matter for the letter:

1. the probe and compact-step callbacks do not change training (bitwise),
2. default ddpg/td3 training is unchanged by the opt-in hooks,
3. the hybrid head emits only +-1 on the on/off dim, samples P(on) = sigmoid(z),
   and is deterministic (argmax) at evaluation.
"""
from __future__ import annotations

import gzip
from dataclasses import replace

import numpy as np
import pytest
import torch

from tuning.protocol import DEFAULT
from tuning.spaces import build_experiment

PROTO = replace(DEFAULT, backend="analytic", total_timesteps=1200, eval_every=600,
                eval_episodes=1, objective_episodes=2)
T27 = {"action_noise_std": 0.2468, "binary_noise_std": 0.3688, "noise_type": "normal",
       "ou_dt": 1.0, "tau": 0.02}
T28 = {"action_noise_std": 0.3, "binary_noise_std": 0.8068, "policy_delay": 1,
       "target_policy_noise": 0.4, "tau": 0.02}


@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    import ddpg.paths
    import letterexp.paths
    import letterexp.probe
    import td3.paths

    monkeypatch.setattr(ddpg.paths, "DDPG_RUNS_DIR", tmp_path / "ddpg")
    monkeypatch.setattr(td3.paths, "TD3_RUNS_DIR", tmp_path / "td3")
    monkeypatch.setattr(letterexp.probe, "PROBE_SET_DIR", tmp_path / "probe")
    return tmp_path


def _probe(exp, tmp_path):
    from letterexp.probe import build_probe_set
    return build_probe_set(exp.env, "analytic", n=32, seed=1, out=tmp_path / "probe.npz")


def _callbacks(exp, probe_path):
    from letterexp.compact_steps import CompactStepCallback
    from letterexp.probe import ProbeCallback, binary_dims_and_labels

    idx, labels = binary_dims_and_labels(exp.env)
    return [ProbeCallback(probe_path, idx, labels, every=300),
            CompactStepCallback(None, idx, labels, obs_normalized=True)]


def _weights(run_dir, cls):
    m = cls.load(str(run_dir / "model.zip"), device="cpu")
    return torch.cat([v.flatten().float() for v in m.policy.state_dict().values()]), m


def _train(algo, params, cbs, tmp_path):
    from stable_baselines3 import DDPG, TD3
    exp = build_experiment(algo, params, PROTO, 3, "t")
    kw = {"extra_callbacks": cbs}
    if algo == "ddpg":
        from ddpg.train import train
    elif algo == "td3":
        from td3.train import train
    elif algo == "ddpg_hybrid":
        from ddpg_hybrid.train import train
        kw["gumbel_tau"] = 1.0
    else:
        from td3_hybrid.train import train
        kw["gumbel_tau"] = 1.0
    rd = train(exp, backend="analytic", **kw)
    return rd, _weights(rd, TD3 if algo.startswith("td3") else DDPG)


@pytest.mark.parametrize("algo,params", [("ddpg", T27), ("td3", T28)])
def test_probe_does_not_perturb_training(dirs, algo, params):
    exp = build_experiment(algo, params, PROTO, 3, "t")
    probe = _probe(exp, dirs)
    _, (w0, _) = _train(algo, params, None, dirs)
    rd, (w1, _) = _train(algo, params, _callbacks(exp, probe), dirs)
    assert torch.equal(w0, w1)

    import pandas as pd
    log = pd.read_csv(rd / "probe_log.csv")
    assert list(log["timesteps"]) == [0, 300, 600, 900, 1200]
    n_q = 2 if algo == "td3" else 1
    assert f"fbs0_power_status_dq_mu_q{n_q}_median" in log
    raw = np.load(rd / "probe_raw.npz")
    assert raw["fbs0_power_status__dq_mu"].shape == (5, n_q, 32)
    with gzip.open(rd / "steps_compact.csv.gz", "rt") as f:
        rows = f.read().strip().split("\n")
    assert len(rows) == 1 + PROTO.total_timesteps


def test_sticky_rule_matches_letter_definition():
    from ppo.env import FlyingBaseStationEnv
    exp = build_experiment("ddpg", T27, PROTO, 0, "t")
    env = FlyingBaseStationEnv(exp.env, backend_kind="analytic")
    cur = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], np.float32)
    a = np.array([0.49, -0.49, 0.5, -0.5, -0.51, 0.51], np.float32)
    out = env._binary_update(cur, a)
    assert out.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    env.close()


def test_hybrid_head():
    from ddpg_hybrid.hybrid import HybridActor
    from stable_baselines3.common.torch_layers import FlattenExtractor
    import gymnasium as gym

    obs_sp = gym.spaces.Box(-1, 1, (6,), np.float32)
    act_sp = gym.spaces.Box(-1, 1, (5,), np.float32)
    actor = HybridActor(obs_sp, act_sp, [16], FlattenExtractor(obs_sp), 6,
                        binary_idx=[4], gumbel_tau=1.0)
    torch.manual_seed(0)
    obs = torch.zeros(20000, 6)
    with torch.no_grad():
        z = actor.mu[:-1](actor.extract_features(obs, actor.features_extractor))[0, 4].item()
        s = actor(obs, sample=True)
        d = actor(obs, sample=False)
    assert set(np.unique(s[:, 4].numpy())) <= {-1.0, 1.0}
    assert abs((s[:, 4] > 0).float().mean().item() - 1 / (1 + np.exp(-z))) < 0.015
    assert torch.all(d[:, 4] == (1.0 if z > 0 else -1.0))
    assert torch.all(s[:, :4].abs() < 1)          # continuous dims untouched
    # straight-through: the binary dim carries a gradient to the logit
    x = torch.zeros(4, 6)
    out = actor(x, sample=True)
    out[:, 4].sum().backward()
    assert actor.mu[-2].weight.grad[4].abs().sum() > 0


@pytest.mark.parametrize("algo,params", [("ddpg_hybrid", T27), ("td3_hybrid", T28)])
def test_hybrid_trains_and_evaluates_deterministically(dirs, algo, params):
    p = {k: v for k, v in params.items() if k != "binary_noise_std"}
    p["gumbel_tau"] = 1.0
    exp = build_experiment(algo, p, PROTO, 3, "t")
    assert (exp.ddpg if algo.startswith("ddpg") else exp.td3).binary_noise_std == 0.0
    rd, (_, model) = _train(algo, p, None, dirs)
    assert type(model.policy).__name__ == "HybridTD3Policy"
    obs = torch.as_tensor(np.random.default_rng(0).uniform(-1, 1, (64, model.observation_space.shape[0])),
                          dtype=torch.float32)
    a1, _ = model.predict(obs.numpy(), deterministic=True)
    a2, _ = model.predict(obs.numpy(), deterministic=True)
    assert np.array_equal(a1, a2)
    assert set(np.unique(a1[:, 4])) <= {-1.0, 1.0}
    import pandas as pd
    steps = pd.read_csv(rd / "steps.csv")
    assert set(steps["fbs0_power_status"].unique()) <= {0.0, 1.0}


def test_qdiff_actor_loss_raises_p_on_when_critic_prefers_on():
    """Letter Step 2: one gradient step must increase sigmoid(z) on the binary
    dim when Q(on) > Q(off) everywhere.

    This is the property the straight-through hybrid does NOT guarantee: it
    follows dQ/da at the sampled action, which E3 measured pointing toward
    'off' while Q(on) - Q(off) pointed toward 'on'.
    """
    import numpy as np
    import torch as th

    from ddpg_hybrid.hybrid import QDiffHybridMixin

    th.manual_seed(0)
    n_obs, n_act, binary = 4, 3, [2]

    class _Critic:
        """Q1 = c * a_b: strictly prefers on (+1) over off (-1) when c > 0."""

        def __init__(self, c):
            self.c = c

        def q1_forward(self, obs, a):
            return (self.c * a[:, binary[0]]).reshape(-1, 1)

    class _Actor(th.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = th.nn.Linear(n_obs, n_act)
            self.mu = th.nn.Sequential(self.lin, th.nn.Identity())
            self.binary_idx = binary
            self.features_extractor = None

        def extract_features(self, obs, _fe):
            return obs

        def forward(self, obs):
            z = self.mu[:-1](obs)
            hard = (th.rand_like(z) < th.sigmoid(z)).to(z.dtype)
            st = hard + (th.sigmoid(z) - th.sigmoid(z).detach())
            return th.where(
                th.tensor([i in binary for i in range(n_act)]), 2.0 * st - 1.0, th.tanh(z))

    class _Model(QDiffHybridMixin):
        def __init__(self, actor, critic):
            self.actor, self.critic = actor, critic
            self.policy = type("P", (), {"binary_idx": binary})()

    actor = _Actor()
    model = _Model(actor, _Critic(c=1.0))      # Q(on) - Q(off) = +2 for every state
    obs = th.randn(64, n_obs)

    with th.no_grad():
        p_before = th.sigmoid(actor.mu[:-1](obs)[:, binary[0]]).mean().item()

    opt = th.optim.SGD(actor.parameters(), lr=0.5)
    loss = model._actor_loss(obs)
    opt.zero_grad()
    loss.backward()
    opt.step()

    with th.no_grad():
        p_after = th.sigmoid(actor.mu[:-1](obs)[:, binary[0]]).mean().item()

    assert p_after > p_before, f"P(on) did not rise: {p_before:.4f} -> {p_after:.4f}"

    # and the reverse: a critic preferring off must push P(on) down
    actor2 = _Actor()
    model2 = _Model(actor2, _Critic(c=-1.0))
    with th.no_grad():
        q_before = th.sigmoid(actor2.mu[:-1](obs)[:, binary[0]]).mean().item()
    opt2 = th.optim.SGD(actor2.parameters(), lr=0.5)
    loss2 = model2._actor_loss(obs)
    opt2.zero_grad()
    loss2.backward()
    opt2.step()
    with th.no_grad():
        q_after = th.sigmoid(actor2.mu[:-1](obs)[:, binary[0]]).mean().item()
    assert q_after < q_before, f"P(on) did not fall: {q_before:.4f} -> {q_after:.4f}"


def test_pins_multi_gene_matches_single_and_handles_two(tmp_path):
    """Letter 2-FBS: summarize_run_multi must reproduce summarize_run per gene.

    The 1-FBS path must stay byte-identical, so the multi wrapper is only
    allowed to be a dict of the very same per-label results.
    """
    import pandas as pd

    from letterexp.pins import (MIN_RUN, gene_labels, summarize_run,
                                summarize_run_multi)

    n = 12
    rows = {"timesteps": [1000 * (i + 1) for i in range(n)]}
    # gene 0: trapped for the first MIN_RUN checkpoints, then free
    # gene 1: never trapped
    for j, trapped_len in ((0, MIN_RUN), (1, 0)):
        lab = f"fbs{j}_power_status"
        tr = [1 if i < trapped_len else 0 for i in range(n)]
        rows[f"{lab}_pinned"] = tr
        rows[f"{lab}_trapped"] = tr
        rows[f"{lab}_frac_on"] = [0.0 if t else 0.5 for t in tr]
        rows[f"{lab}_frac_off"] = [1.0 if t else 0.5 for t in tr]
    pd.DataFrame(rows).to_csv(tmp_path / "probe_log.csv", index=False)

    assert gene_labels(tmp_path) == ["fbs0_power_status", "fbs1_power_status"]

    multi = summarize_run_multi(tmp_path)
    assert set(multi) == {"fbs0_power_status", "fbs1_power_status"}
    for lab in multi:
        assert multi[lab] == summarize_run(tmp_path, label=lab)

    assert multi["fbs0_power_status"]["n_trap_events"] == 1
    assert multi["fbs1_power_status"]["n_trap_events"] == 0

    from letterexp.pins import all_genes, any_gene
    assert any_gene(multi, "early_trap") is True
    assert all_genes(multi, "early_trap") is False


def test_onoff_override_min_on_and_force_on():
    """Run C: the override holds a station on for L steps and forces it early."""
    import numpy as np

    from letterexp.onoff_override import OnOffOverride

    b = [4, 9]
    # FORCE_ON: every binary dim is +1 while num_timesteps < N, untouched after
    ov = OnOffOverride(b, force_on_steps=100)
    a = np.full((2, 10), -1.0)
    out = ov(a, np.zeros((2, 16)), 50)
    assert np.all(out[:, b] == 1.0)
    assert np.all(out[:, 0] == -1.0), "continuous dims must not be touched"
    out = ov(a, np.zeros((2, 16)), 150)
    assert np.all(out[:, b] == -1.0), "override must stop after N steps"

    # MIN_ON(3): a fresh switch to on holds for 2 further steps
    ov = OnOffOverride([4], min_on=3)
    off = np.full((1, 16), -1.0)          # normalized obs: gene off
    a_on = np.array([[0.0, 0, 0, 0, 0.9, 0, 0, 0, 0, 0]])
    assert ov(a_on, off, 0)[0, 4] == 0.9            # the switch itself passes through
    a_off = np.array([[0.0, 0, 0, 0, -0.9, 0, 0, 0, 0, 0]])
    on = np.full((1, 16), 1.0)                       # gene now on
    assert ov(a_off, on, 1)[0, 4] == 1.0             # held
    assert ov(a_off, on, 2)[0, 4] == 1.0             # held
    assert ov(a_off, on, 3)[0, 4] == -0.9            # window over, actor free again


def test_onoff_override_writes_executed_action_to_replay_buffer():
    """The buffer must hold the FORCED action, not the actor's output.

    This is the failure mode a gym wrapper would have: env executes +1 while SB3
    stores whatever the actor said, training the critic on transitions that
    never occurred.
    """
    import numpy as np
    from stable_baselines3 import DDPG

    from letterexp.onoff_override import OnOffOverride, onoff_override

    import gymnasium as gym

    class Tiny(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (4,), np.float32)
        action_space = gym.spaces.Box(-1, 1, (3,), np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros(4, np.float32), {}

        def step(self, a):
            return np.zeros(4, np.float32), 0.0, False, False, {}

    binary = [2]
    model = DDPG("MlpPolicy", Tiny(), learning_starts=0, buffer_size=200,
                 batch_size=8, seed=0, verbose=0)
    ov = OnOffOverride(binary, force_on_steps=10_000)
    with onoff_override(DDPG, ov):
        model.learn(total_timesteps=40, log_interval=None)
    stored = model.replay_buffer.actions[: model.replay_buffer.pos, 0, binary[0]]
    assert len(stored) > 0
    assert np.allclose(stored, 1.0), f"buffer holds {np.unique(stored)}, expected all +1"
