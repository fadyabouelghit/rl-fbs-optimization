"""on_off_td3/: "try both" -- the critic scores every on/off pattern and keeps the best.

Analytic backend, seconds. Covers the spec's test list
(docs/RL_ARCHITECTURES_2026-11-08.md, section 5.1) plus the three agreed
changes: actor trained under "all on", one selection rule, noise on c only.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch as th

from tuning.protocol import SMOKE
from tuning.spaces import SEED_TRIALS, build_experiment


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    import ddpg.paths
    import on_off_td3.train
    import td3.paths
    from tuning import paths as tp

    monkeypatch.setattr(td3.paths, "TD3_RUNS_DIR", tmp_path / "td3")
    monkeypatch.setattr(td3.paths, "TD3_LEDGER_PATH", tmp_path / "td3" / "ledger.csv")
    monkeypatch.setattr(ddpg.paths, "DDPG_RUNS_DIR", tmp_path / "ddpg")
    monkeypatch.setattr(ddpg.paths, "DDPG_LEDGER_PATH", tmp_path / "ddpg" / "ledger.csv")
    monkeypatch.setattr(on_off_td3.train, "ON_OFF_TD3_RUNS_DIR", tmp_path / "on_off")
    monkeypatch.setattr(tp, "TUNING_DIR", tmp_path / "tuning_runs")


PILOT = {"on_off_td3": "td3", "on_off_ddpg": "ddpg"}


def _exp(algo="on_off_td3", code="2-1-1", **kw):
    kw.setdefault("fixed_batch_size", 32)
    proto = replace(SMOKE, code=code, **kw)
    return build_experiment(algo, SEED_TRIALS[algo][0], proto, 0, "t"), proto


def _model(algo="on_off_td3", code="2-1-1", actor_pattern="all_on", **pkw):
    from stable_baselines3.common.vec_env import DummyVecEnv

    from on_off_td3.config import TryBothParams
    from on_off_td3.train import build_model
    from ppo.env import FlyingBaseStationEnv

    exp, _ = _exp(algo, code)
    if pkw:
        P = exp.td3 if PILOT[algo] == "td3" else exp.ddpg
        for k, v in pkw.items():
            setattr(P, k, v)
    env = FlyingBaseStationEnv(exp.env, backend_kind="analytic")
    venv = DummyVecEnv([lambda: env])
    return build_model(exp, venv, PILOT[algo], TryBothParams(actor_pattern=actor_pattern)), env, exp


def _train(algo="on_off_td3", extra_callbacks=None, **proto_kw):
    from on_off_td3.train import train_algo

    exp, proto = _exp(algo, **proto_kw)
    return train_algo(algo, exp, proto.backend, params=SEED_TRIALS[algo][0],
                      extra_callbacks=extra_callbacks), proto


def _load(rd, algo="on_off_td3", file="model.zip"):
    from stable_baselines3 import DDPG, TD3
    return (TD3 if PILOT[algo] == "td3" else DDPG).load(str(Path(rd) / file), device="cpu")


def _same(a, b):
    return a.keys() == b.keys() and all(th.equal(a[k], b[k]) for k in a)


# --------------------------------------------------------------------------- #
# 1-2. pattern choice and the actor's output
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("code,n", [("1-1-1", 1), ("2-1-1", 2)])
def test_argmax_picks_the_best_pattern_from_a_hand_set_critic(code, n):
    model, _, _ = _model(code=code)
    pol = model.policy
    assert pol.n_patterns == 2 ** n
    weights = th.arange(1, n + 1, dtype=th.float32) * th.tensor([1.0, -1.0, 1.0][:n])

    def critic(obs, a):            # score = sum_j w_j * b_j: best is b_j = sign(w_j)
        return (a[:, pol.onoff_idx] @ weights[:, None],)

    obs = th.zeros(3, model.observation_space.shape[0])
    c = th.zeros(3, 4 * n)
    best = pol.pattern_scores(critic, obs, c).argmax(dim=1)
    assert th.equal(pol.patterns[best[0]], th.sign(weights))


@pytest.mark.parametrize("algo", ["on_off_td3", "on_off_ddpg"])
def test_actor_outputs_only_the_continuous_genes(algo):
    model, env, _ = _model(algo)
    assert model.policy.actor.mu[-2].out_features == 8
    assert model.action_noise is not None and model.action_noise().shape == (8,)
    obs, _ = env.reset(seed=1)
    a1, _ = model.predict(obs, deterministic=True)
    a2, _ = model.predict(obs, deterministic=True)
    assert a1.shape == (10,) and np.array_equal(a1, a2)
    assert set(np.unique(a1[[4, 9]])) <= {-1.0, 1.0}
    with th.no_grad():
        o = th.as_tensor(obs[None])
        c = model.policy.actor(o)
        best = model.policy.best_pattern(o, c)
    np.testing.assert_allclose(a1[model.policy.cont_idx], c[0].numpy(), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(a1[[4, 9]], model.policy.patterns[best[0]].numpy())


def test_warm_up_and_exploration_send_b_as_exact_plus_minus_one():
    model, _, _ = _model()
    model.learn(300)
    acts = model.replay_buffer.actions[: model.replay_buffer.pos, 0]
    assert set(np.unique(acts[:, [4, 9]])) <= {-1.0, 1.0}
    assert len(np.unique(acts[:, 0])) > 50                  # continuous dims really vary


# --------------------------------------------------------------------------- #
# 3-4. the learning step against a hand computation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("algo", ["on_off_ddpg", "on_off_td3"])
@pytest.mark.parametrize("actor_pattern", ["all_on", "best"])
def test_critic_and_actor_losses_match_a_hand_computation(algo, actor_pattern):
    from stable_baselines3.common.logger import configure

    kw = {"target_policy_noise": 0.0, "policy_delay": 1} if algo == "on_off_td3" else {}
    model, _, _ = _model(algo, actor_pattern=actor_pattern, **kw)
    if algo == "on_off_td3":
        model.target_policy_noise, model.policy_delay = 0.0, 1
    model.set_logger(configure(None, []))
    model.learn(200)                                    # fills the buffer; trains a little
    rb, pol = model.replay_buffer, model.policy

    # a buffer holding one transition makes the sampled batch deterministic
    one = rb.pos - 1
    for name in ("observations", "next_observations", "actions", "rewards", "dones", "timeouts"):
        arr = getattr(rb, name)
        arr[:] = arr[one]
    with th.no_grad():
        s = th.as_tensor(rb.observations[0]); s2 = th.as_tensor(rb.next_observations[0])
        a = th.as_tensor(rb.actions[0]); r = float(rb.rewards[0, 0])
        d = float(rb.dones[0, 0] * (1 - rb.timeouts[0, 0]))       # SB3: a time-out still bootstraps
        c2 = model.actor_target(s2)
        b2 = pol.pattern_scores(model.critic, s2, c2).argmax(dim=1)      # online picks
        q2 = th.cat(model.critic_target(s2, pol.full_action(c2, pol.patterns[b2])), 1).min(1).values
        y = r + (1 - d) * model.gamma * q2
        crit = sum(float(((q - y[:, None]) ** 2).mean()) for q in model.critic(s, a))
    import copy
    crit_params = [p.detach().clone() for p in model.critic.parameters()]
    actor_before = copy.deepcopy(model.actor)
    model.train(gradient_steps=1, batch_size=8)
    assert model.logger.name_to_value["train/critic_loss"] == pytest.approx(crit, rel=1e-5)

    # actor loss = -Q1(s, [mu(s), b_train]) with the actor as it was before the step
    # and the critic as just updated (SB3 order: critic step, then actor step)
    assert any(not th.equal(p0, p1.detach()) for p0, p1 in zip(crit_params, model.critic.parameters()))
    with th.no_grad():
        c = actor_before(s)
        if actor_pattern == "all_on":
            b = pol.patterns[[pol.all_on]]
        else:
            b = pol.patterns[pol.pattern_scores(model.critic, s, c).argmax(dim=1)]
        act = -float(model.critic.q1_forward(s, pol.full_action(c, b)).mean())
    assert model.logger.name_to_value["train/actor_loss"] == pytest.approx(act, rel=1e-4, abs=1e-8)   # float32: mean of 8 rows vs 1


def test_actor_gradient_does_not_flow_through_the_choice_of_b():
    model, _, _ = _model(actor_pattern="best")
    pol = model.policy
    obs = th.randn(16, model.observation_space.shape[0])
    c = pol.actor(obs)
    b_idx = pol.best_pattern(obs, c.detach())
    assert not b_idx.requires_grad and b_idx.dtype == th.long
    loss = -model.critic.q1_forward(obs, pol.full_action(c, pol.patterns[b_idx])).mean()
    loss.backward()
    assert all(p.grad is not None for p in pol.actor.parameters())


def test_all_on_actor_loss_trains_every_station_placement():
    """Under 'all_on' an off station's genes still get gradient; that is the point."""
    model, _, _ = _model(actor_pattern="all_on")
    pol = model.policy
    obs = th.randn(16, model.observation_space.shape[0])
    c = pol.actor(obs)
    c.retain_grad()
    b = pol.patterns[th.full((16,), pol.all_on, dtype=th.long)]
    assert th.all(b == 1.0)
    (-model.critic.q1_forward(obs, pol.full_action(c, b)).mean()).backward()
    assert th.all(c.grad[:, 4:].abs().sum(dim=1) > 0)       # second FBS's genes move


# --------------------------------------------------------------------------- #
# 5-7. reproducibility, routing, scoring
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("algo", ["on_off_td3", "on_off_ddpg"])
def test_seeded_run_reproduces_bitwise(algo):
    r1, _ = _train(algo)
    r2, _ = _train(algo)
    assert _same(_load(r1, algo).policy.state_dict(), _load(r2, algo).policy.state_dict())


def test_runs_stay_out_of_protected_trees(tmp_path):
    rd, _ = _train()
    assert Path(rd).is_relative_to(tmp_path)


def test_scoring_is_stable_across_loads():
    from tuning.objective import score_run

    rd, proto = _train()
    a = score_run("on_off_td3", rd, proto)
    assert a["objective"] == score_run("on_off_td3", rd, proto)["objective"]
    assert a["model"] == "best"


# --------------------------------------------------------------------------- #
# 8. probe neutrality and readability
# --------------------------------------------------------------------------- #
def test_probe_and_compact_log_are_bitwise_neutral(tmp_path):
    from letterexp.compact_steps import CompactStepCallback
    from letterexp.pins import pin_events
    from letterexp.probe import binary_dims_and_labels, build_probe_set
    from on_off_td3.callbacks import TryBothProbeCallback

    exp, _ = _exp()
    probe = build_probe_set(exp.env, "analytic", n=16, seed=1, out=tmp_path / "probe.npz")
    idx, labels = binary_dims_and_labels(exp.env)
    plain, _ = _train()
    cbs = [TryBothProbeCallback(probe, idx, labels, every=100),
           CompactStepCallback(None, idx, labels, obs_normalized=True)]
    probed, _ = _train(extra_callbacks=cbs)
    assert _same(_load(plain).policy.state_dict(), _load(probed).policy.state_dict())
    log = pd.read_csv(probed / "tryboth_probe.csv")
    assert {f"{lab}_trapped" for lab in labels} <= set(log.columns) and len(log) == 7
    pin_events(log, labels[0], kind="trap")
    steps = pd.read_csv(probed / "steps_compact.csv.gz")
    assert set(steps[f"a_{labels[0]}"].unique()) <= {-1.0, 1.0}
