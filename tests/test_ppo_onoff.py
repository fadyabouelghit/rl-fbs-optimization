"""ppo_onoff/: Hybrid B, twin-critic on/off selection + PPO placement (analytic, seconds).

The spec's test list (docs/RL_ARCHITECTURES_2026-11-08.md section 6.6):
1 argmax over patterns with a hand-set critic; 2 PPO's input contains b*,
only c is scored; 3 critic target vs a hand computation; 4 no gradient path
PPO -> critics or critics -> actor; 5 bitwise reproducibility; 6 stable
scoring; 7 the suite. Plus routing, probe neutrality, the tuner plumbing.
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

ALGO = "ppo_onoff"


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    import ppo.run_logging
    import ppo_onoff.paths
    from tuning import paths as tp

    monkeypatch.setattr(ppo_onoff.paths, "PPO_ONOFF_RUNS_DIR", tmp_path / "ppo_onoff")
    monkeypatch.setattr(ppo.run_logging, "LEDGER_PATH", tmp_path / "ppo_ledger_must_not_exist.csv")
    monkeypatch.setattr(tp, "TUNING_DIR", tmp_path / "tuning_runs")


def _params(**kw):
    p = dict(SEED_TRIALS[ALGO][0])
    p["n_steps"] = 512
    p.update(kw)
    return p


def _exp(code="2-1-1", params=None, **kw):
    kw.setdefault("fixed_batch_size", 32)
    proto = replace(SMOKE, code=code, **kw)
    return build_experiment(ALGO, params or _params(), proto, 0, "t"), proto


def _model(code="2-1-1", **onoff_kw):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from ppo.env import FlyingBaseStationEnv
    from ppo_onoff.config import OnOffParams
    from ppo_onoff.train import build_model

    exp, _ = _exp(code)
    exp.ppo.n_epochs = 2                                     # cheap updates in unit tests
    env = FlyingBaseStationEnv(exp.env, backend_kind="analytic")
    venv = VecNormalize(DummyVecEnv([lambda: env]), norm_obs=False, norm_reward=True, gamma=0.95)
    onoff_kw.setdefault("critic_batch_size", 16)
    return build_model(exp, venv, OnOffParams(**onoff_kw)), env, exp


def _train(extra_callbacks=None, params=None, **proto_kw):
    from ppo_onoff.train import train_algo

    params = params or _params()
    exp, proto = _exp(params=params, **proto_kw)
    return Path(train_algo(ALGO, exp, proto.backend, params=params,
                           extra_callbacks=extra_callbacks)), proto


def _load(rd, file="model.zip"):
    from stable_baselines3 import PPO
    return PPO.load(str(Path(rd) / file), device="cpu")


def _same(a, b):
    return a.keys() == b.keys() and all(th.equal(a[k], b[k]) for k in a)


def _hand_mean(pol, s, b):
    """mu(s, b) by hand: PPO's policy MLP then action_net on [s, b]."""
    return pol.action_net(pol.mlp_extractor.policy_net(th.cat([s, b], dim=1)))


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def test_space_and_pinned_values():
    from tuning.spaces import KNOBS_PER_ALGO, space

    names = [k.name for k in space(ALGO)]
    assert len(names) == KNOBS_PER_ALGO
    assert {"ent_coef", "n_steps", "gae_lambda", "exploration_fraction", "tau"} == set(names)
    exp, _ = _exp(params=_params(n_epochs=5, clip_range=0.1))     # strays are ignored
    assert exp.ppo.n_epochs == 20 and exp.ppo.clip_range == 0.3 and exp.ppo.algo == "ppo"


def test_sb3_version_is_pinned():
    from ppo_onoff.algo import SB3_VERSION
    import stable_baselines3
    assert stable_baselines3.__version__ == SB3_VERSION


# --------------------------------------------------------------------------- #
# 1. argmax over patterns
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("code,n", [("1-1-1", 1), ("2-1-1", 2)])
def test_argmax_picks_the_best_pattern_from_a_hand_set_critic(code, n):
    model, _, _ = _model(code)
    pol = model.policy
    assert pol.n_patterns == 2 ** n
    w = th.arange(1, n + 1, dtype=th.float32) * th.tensor([1.0, -1.0, 1.0][:n])

    def critic(obs, a):                  # score = sum_j w_j b_j - |c|: best b_j = sign(w_j)
        return (a[:, pol.onoff_idx] @ w[:, None] - 1e-3 * a[:, pol.cont_idx].abs().sum(1, keepdim=True),
                a[:, pol.onoff_idx] @ w[:, None] + 5.0)
    obs = th.zeros(3, model.observation_space.shape[0])
    best = pol.pattern_scores(critic, obs).argmax(dim=1)
    assert th.equal(pol.patterns[best[0]], th.sign(w))


def test_scores_use_the_clipped_ppo_mean_for_each_pattern():
    model, env, _ = _model()
    pol = model.policy
    obs = th.as_tensor(env.reset(seed=2)[0][None])
    seen = []

    def critic(o, a):
        seen.append(a.clone())
        return (th.zeros(a.shape[0], 1),)
    pol.pattern_scores(critic, obs)
    a = seen[0]
    for p in range(pol.n_patterns):
        b = pol.patterns[p:p + 1]
        c = th.clamp(_hand_mean(pol, obs, b), -1, 1)
        assert th.allclose(a[p, pol.cont_idx], c[0], atol=1e-6)
        assert th.equal(a[p, pol.onoff_idx], b[0])


# --------------------------------------------------------------------------- #
# 2. PPO sees b*, scores c only
# --------------------------------------------------------------------------- #
def test_rollout_stores_full_action_and_ppo_scores_c_given_b():
    model, _, _ = _model()
    model.learn(512)
    rb, pol = model.rollout_buffer, model.policy
    assert rb.observations.shape[-1] == 16 and rb.actions.shape[-1] == 10
    acts = th.as_tensor(rb.actions.reshape(-1, 10))       # flattened after the update
    obs = th.as_tensor(rb.observations.reshape(-1, 16))
    assert set(np.unique(acts[:, pol.onoff_idx].numpy())) <= {-1.0, 1.0}
    assert len(np.unique(acts[:, 0].numpy())) > 50
    with th.no_grad():
        v, logp, ent = pol.evaluate_actions(obs, acts)
        b, c = acts[:, pol.onoff_idx], acts[:, pol.cont_idx]
        dist = pol.action_dist.proba_distribution(_hand_mean(pol, obs, b), pol.log_std)
        assert th.allclose(logp, dist.log_prob(c), atol=1e-5)
        assert ent.shape == logp.shape and pol.log_std.shape == (8,)
        # b is an input: flipping it changes the value and the log-prob of the same c
        flipped = acts.clone()
        flipped[:, pol.onoff_idx] *= -1
        v2, logp2, _ = pol.evaluate_actions(obs, flipped)
        assert not th.allclose(v, v2) and not th.allclose(logp, logp2)


def test_predict_is_greedy_pattern_plus_ppo_mean():
    model, env, _ = _model()
    model.learn(512)
    pol = model.policy
    obs, _ = env.reset(seed=4)
    a, _ = model.predict(obs, deterministic=True)
    with th.no_grad():
        o = th.as_tensor(obs[None])
        b = pol.patterns[pol.greedy_pattern(o)]
        c = th.clamp(_hand_mean(pol, o, b), -1, 1)
    assert a.shape == (10,)
    np.testing.assert_array_equal(a[pol.onoff_idx], b[0].numpy())
    np.testing.assert_allclose(a[pol.cont_idx], c[0].numpy(), atol=1e-6)


def test_epsilon_schedule_and_exploration_stream():
    model, _, _ = _model(exploration_fraction=0.5)
    model.learn(512)
    assert model.epsilon_at(0) == 1.0
    assert model.epsilon_at(model._onoff_total) == pytest.approx(0.05)
    pol = model.policy
    pol.eps = 1.0
    idx = th.zeros(400, dtype=th.long)
    assert len(set(pol.explore_pattern(idx).tolist())) == pol.n_patterns
    pol.eps = 0.0
    assert th.equal(pol.explore_pattern(idx), idx)


# --------------------------------------------------------------------------- #
# 3. critic target by hand
# --------------------------------------------------------------------------- #
def test_critic_target_and_loss_match_a_hand_computation():
    model, _, _ = _model()
    model.learn(300)
    pol, rb = model.policy, model._rb
    idx = np.arange(8)
    vn = model.get_vec_normalize_env()
    with th.no_grad():
        s = th.as_tensor(rb["obs"][idx]); a = th.as_tensor(rb["act"][idx])
        s2 = th.as_tensor(rb["next"][idx]); term = th.as_tensor(rb["term"][idx])
        r = th.as_tensor(vn.normalize_reward(rb["rew"][idx].astype(np.float64)), dtype=th.float32)
        y = []
        for i in range(len(idx)):
            best, best_score = None, -np.inf
            for p in range(pol.n_patterns):                 # online critics choose b'
                b = pol.patterns[p:p + 1]
                c = th.clamp(_hand_mean(pol, s2[i:i + 1], b), -1, 1)
                full = pol.full_action(c, b)
                sc = min(float(q) for q in pol.critic(s2[i:i + 1], full))
                if sc > best_score:
                    best, best_score = full, sc
            q2 = min(float(q) for q in pol.critic_target(s2[i:i + 1], best))
            y.append(float(r[i]) + (1 - float(term[i])) * model.gamma * q2)
        y = th.tensor(y)
        want = sum(float(((q.squeeze(1) - y) ** 2).mean()) for q in pol.critic(s, a))
    got = model.critic_train_step(idx)
    assert got == pytest.approx(want, rel=1e-5)


def test_replay_holds_executed_actions_raw_rewards_and_truncation():
    model, _, exp = _model()
    model.learn(300)
    rb, n = model._rb, model.rb_size
    assert n == 512                                     # one transition per env step
    assert np.all(np.abs(rb["act"][:n]) <= 1.0)
    assert rb["term"][:n].sum() == 0                    # 40-step time limit: truncation only
    ep = exp.env.max_episode_steps
    assert not np.allclose(rb["next"][ep - 1], rb["obs"][ep])    # terminal obs, not the reset obs


# --------------------------------------------------------------------------- #
# 4. no gradient path either way
# --------------------------------------------------------------------------- #
def test_ppo_update_never_moves_the_critics_and_critic_step_never_moves_ppo():
    model, _, _ = _model()
    model.learn(512)
    pol = model.policy
    critic0 = {k: v.clone() for k, v in pol.critic.state_dict().items()}
    ppo_keys = [k for k in pol.state_dict() if not k.startswith(("critic.", "critic_target."))]
    ppo0 = {k: pol.state_dict()[k].clone() for k in ppo_keys}
    assert all(p.grad is None for p in pol.critic.parameters())
    critic_param_ids = {id(p) for p in pol.critic.parameters()}
    assert not any(id(p) in critic_param_ids for g in pol.optimizer.param_groups for p in g["params"])

    model.train()                                       # one PPO update
    assert _same(critic0, pol.critic.state_dict())
    assert any(not th.equal(ppo0[k], pol.state_dict()[k]) for k in ppo_keys)

    ppo1 = {k: pol.state_dict()[k].clone() for k in ppo_keys}
    model.critic_train_step(np.arange(16))
    assert all(th.equal(ppo1[k], pol.state_dict()[k]) for k in ppo_keys)
    assert not _same(critic0, pol.critic.state_dict())
    assert all(p.grad is None for p in pol.critic.parameters())       # PPO's clip_grad_norm


def test_critic_loss_has_no_path_into_the_actor():
    model, _, _ = _model()
    model.learn(300)
    pol = model.policy
    for p in pol.parameters():
        p.grad = None
    model.critic_train_step(np.arange(8))
    ppo_params = [p for n, p in pol.named_parameters() if not n.startswith(("critic.", "critic_target."))]
    assert all(p.grad is None for p in ppo_params)


# --------------------------------------------------------------------------- #
# 5-7. reproducibility, routing, scoring
# --------------------------------------------------------------------------- #
def test_seeded_run_reproduces_bitwise():
    r1, _ = _train()
    r2, _ = _train()
    assert _same(_load(r1).policy.state_dict(), _load(r2).policy.state_dict())
    import json
    assert json.loads((r1 / "experiment_config.json").read_text())["provenance"]["critic_updates"] > 0


def test_runs_and_ledger_stay_out_of_ppo_trees(tmp_path):
    import ppo.run_logging

    rd, _ = _train()
    assert rd.is_relative_to(tmp_path / "ppo_onoff")
    ledger = pd.read_csv(tmp_path / "ppo_onoff" / "ledger.csv")
    assert list(ledger["algo"]) == [ALGO] and ledger["n_epochs"].iloc[0] == 20
    assert "onoff_tau" in ledger.columns
    assert not Path(ppo.run_logging.LEDGER_PATH).exists()


def test_saved_model_holds_all_four_networks_and_no_replay():
    rd, _ = _train()
    m = _load(rd)
    keys = m.policy.state_dict().keys()
    assert any(k.startswith("critic.") for k in keys) and any(k.startswith("critic_target.") for k in keys)
    assert any(k.startswith("value_net.") for k in keys) and any(k.startswith("action_net.") for k in keys)
    assert not hasattr(m, "_rb")
    assert (rd / "model.zip").stat().st_size < 2_000_000


def test_scoring_is_stable_across_loads():
    from tuning.objective import score_run

    rd, proto = _train()
    a = score_run(ALGO, rd, proto)
    assert a["objective"] == score_run(ALGO, rd, proto)["objective"] and a["model"] == "best"


# --------------------------------------------------------------------------- #
# probe neutrality
# --------------------------------------------------------------------------- #
def test_probe_and_compact_log_are_bitwise_neutral(tmp_path):
    from letterexp.pins import pin_events
    from letterexp.probe import binary_dims_and_labels, build_probe_set
    from ppo_onoff.callbacks import OnOffProbeCallback, make_compact_step_callback

    exp, _ = _exp()
    probe = build_probe_set(exp.env, "analytic", n=16, seed=1, out=tmp_path / "probe.npz")
    idx, labels = binary_dims_and_labels(exp.env)
    plain, _ = _train()
    cbs = [OnOffProbeCallback(probe, idx, labels, every=100),
           make_compact_step_callback(None, idx, labels, obs_normalized=True)]
    probed, _ = _train(extra_callbacks=cbs)
    assert _same(_load(plain).policy.state_dict(), _load(probed).policy.state_dict())
    log = pd.read_csv(probed / "onoff_probe.csv")
    assert {f"{lab}_trapped" for lab in labels} <= set(log.columns)
    pin_events(log, labels[0], kind="trap")
    steps = pd.read_csv(probed / "steps_compact.csv.gz")
    assert set(steps[f"a_{labels[0]}"].unique()) <= {-1.0, 1.0}


# --------------------------------------------------------------------------- #
# tuner plumbing
# --------------------------------------------------------------------------- #
def test_search_runs_a_trial_in_its_campaign_and_refuses_hold_k():
    import json

    import optuna

    from tuning import paths as tp
    from tuning.study import check_hold_k, search

    with pytest.raises(SystemExit, match="applies only"):
        check_hold_k(optuna.create_study(), ALGO, 1)
    proto = replace(SMOKE, code="2-1-1", fixed_batch_size=32, n_trials=1, n_startup_trials=1)
    search("hb", ALGO, proto, workers=1, quiet=True)
    res = [json.loads(f.read_text()) for f in (tp.study_dir("hb") / "specs").glob(f"{ALGO}_*.result.json")]
    assert len(res) == 1 and res[0]["status"] == "completed", res
    assert Path(res[0]["run_dir"]).is_relative_to(tp.study_dir("hb") / f"{ALGO}_runs")
