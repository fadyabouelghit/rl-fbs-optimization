"""ddqn_td3/: Double-DQN switch + TD3/DDPG pilot (analytic backend, seconds).

Covers the brief's Step 1 test list (prompt_switchpilot.md): wrapper layout,
hold bookkeeping, that the switch learns on > off, seeded reproducibility,
best_model / best_switch pairing, scoring, probe neutrality, and that the
td3/ddpg env-wrapper hooks are neutral when given identity wrappers.
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
    """No test here may touch td3_runs/, ddpg_runs/ or their ledgers."""
    import ddpg.paths
    import ddqn_td3.paths
    import td3.paths
    from tuning import paths as tp

    monkeypatch.setattr(td3.paths, "TD3_RUNS_DIR", tmp_path / "td3")
    monkeypatch.setattr(td3.paths, "TD3_LEDGER_PATH", tmp_path / "td3" / "ledger.csv")
    monkeypatch.setattr(ddpg.paths, "DDPG_RUNS_DIR", tmp_path / "ddpg")
    monkeypatch.setattr(ddpg.paths, "DDPG_LEDGER_PATH", tmp_path / "ddpg" / "ledger.csv")
    monkeypatch.setattr(ddqn_td3.paths, "DDQN_TD3_RUNS_DIR", tmp_path / "ddqn")
    monkeypatch.setattr(tp, "TUNING_DIR", tmp_path / "tuning_runs")


def _proto(code="2-1-1", **kw):
    return replace(SMOKE, code=code, **kw)


def _exp(algo="ddqn_td3", code="2-1-1", **kw):
    proto = _proto(code, **kw)
    return build_experiment(algo, SEED_TRIALS[algo][0], proto, 0, "t"), proto


def _wrapped(exp, hold_k=1):
    from stable_baselines3.common.vec_env import DummyVecEnv

    from ddqn_td3.config import Layout, SwitchParams
    from ddqn_td3.switch import SwitchAgent
    from ddqn_td3.wrapper import SwitchVecEnv
    from ppo.env import FlyingBaseStationEnv

    env = FlyingBaseStationEnv(exp.env, backend_kind="analytic")
    venv = DummyVecEnv([lambda: env])
    layout = Layout(exp.env)
    agent = SwitchAgent(int(venv.observation_space.shape[0]), layout.n_joint,
                        SwitchParams(hold_k=hold_k), gamma=0.95, learning_rate=3e-4,
                        batch_size=32, net_arch=(64, 64), total_timesteps=1000,
                        learning_starts=0, seed=0)
    return SwitchVecEnv(venv, agent, layout), env, layout, agent


def _train(algo="ddqn_td3", hold_k=1, extra_callbacks=None, **proto_kw):
    from ddqn_td3.train import train_algo

    proto_kw.setdefault("fixed_batch_size", 32)        # so the switch updates in 600 steps
    exp, proto = _exp(algo, **proto_kw)
    rd = train_algo(algo, exp, proto.backend, params=SEED_TRIALS[algo][0], hold_k=hold_k,
                    extra_callbacks=extra_callbacks)
    return rd, proto


def _pilot_state(run_dir, file="model.zip", algo="td3"):
    from stable_baselines3 import DDPG, TD3
    cls = TD3 if algo == "td3" else DDPG
    return cls.load(str(Path(run_dir) / file), device="cpu").policy.state_dict()


def _switch_state(run_dir, file="switch.pt"):
    return th.load(str(Path(run_dir) / file), weights_only=False)["q"]


def _same(a, b):
    return a.keys() == b.keys() and all(th.equal(a[k], b[k]) for k in a)


# --------------------------------------------------------------------------- #
# 1. layout
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("code,n", [("1-1-1", 1), ("2-1-1", 2)])
def test_wrapper_sends_full_action_with_binary_b(code, n):
    exp, _ = _exp(code=code)
    w, env, layout, _ = _wrapped(exp)
    obs = w.reset()
    assert obs.shape == (1, env.observation_space.shape[0] + n)
    assert w.action_space.shape == (4 * n,)
    rng = np.random.default_rng(0)
    for _ in range(5):
        c = rng.uniform(-1, 1, size=(1, 4 * n)).astype(np.float32)
        b_before = w.b.copy()
        obs, _, _, infos = w.step(c)
        full = w.last_full_action
        assert set(np.unique(full[layout.onoff_idx])) <= {-1.0, 1.0}
        np.testing.assert_array_equal(full[layout.cont_idx], c[0])
        np.testing.assert_array_equal(full[layout.onoff_idx], b_before)
        # sticky rule applied it: the gene now equals b
        assert np.array_equal(env.state[layout.onoff_idx] > 0.5, b_before > 0)
        np.testing.assert_array_equal(obs[0, -n:], w.b)


# --------------------------------------------------------------------------- #
# 2. holds
# --------------------------------------------------------------------------- #
def test_hold_k_bookkeeping_and_truncated_hold():
    exp, _ = _exp(max_episode_steps=10)
    w, _, _, agent = _wrapped(exp, hold_k=4)
    w.reset()
    decided, b_exec, rewards = [], [], []
    for _ in range(14):
        _, r, _, infos = w.step(np.zeros((1, 8), np.float32))
        decided.append(infos[0]["switch_decided"])
        b_exec.append(infos[0]["switch_b_executed"])
        rewards.append(float(r[0]))
    # steps are 1-based here: decisions after steps 4, 8, 10 (episode end), 14
    assert [i + 1 for i, d in enumerate(decided) if d] == [4, 8, 10, 14]
    for a, z in ((0, 4), (4, 8), (8, 10), (10, 14)):            # b constant inside a hold
        assert all(np.array_equal(b_exec[a], b_exec[k]) for k in range(a, z))
    assert agent.n_stored == 4
    np.testing.assert_array_equal(agent._klen[:4], [4, 4, 2, 4])
    np.testing.assert_array_equal(agent._term[:4], 0.0)          # time limit: truncated
    g = 0.95
    for h, (a, z) in enumerate(((0, 4), (4, 8), (8, 10), (10, 14))):
        np.testing.assert_allclose(agent._rew[h, : z - a], rewards[a:z], rtol=1e-6)
        hand = sum(g ** i * rewards[a + i] for i in range(z - a))
        got = agent.discounted_return(agent._rew[h:h + 1], agent._klen[h:h + 1])[0]
        assert got == pytest.approx(hand, rel=1e-6)
    assert agent.gamma ** agent._klen[2] == pytest.approx(g ** 2)


def test_normalizer_is_applied_per_step_before_discounting():
    from ddqn_td3.config import SwitchParams
    from ddqn_td3.switch import SwitchAgent

    class Norm:
        def normalize_reward(self, r):
            return np.clip(np.asarray(r) / 2.0, -1.0, 1.0)

    a = SwitchAgent(3, 2, SwitchParams(hold_k=3), gamma=0.5, learning_rate=1e-3,
                    batch_size=4, net_arch=(8,), total_timesteps=10, learning_starts=0)
    a.normalizer = Norm()
    got = a.discounted_return(np.array([[1.0, 6.0, 4.0]]), np.array([2]))[0]
    assert got == pytest.approx(0.5 + 0.5 * 1.0)           # 1/2 + 0.5 * clip(3)


# --------------------------------------------------------------------------- #
# 3. the switch learns on > off
# --------------------------------------------------------------------------- #
def test_switch_learns_on_beats_off():
    from ddqn_td3.config import Layout, SwitchParams
    from ddqn_td3.switch import SwitchAgent

    exp, _ = _exp()
    layout = Layout(exp.env)
    a = SwitchAgent(6, layout.n_joint, SwitchParams(), gamma=0.95, learning_rate=1e-3,
                    batch_size=64, net_arch=(64, 64), total_timesteps=2000,
                    learning_starts=0, seed=0)
    rng = np.random.default_rng(1)
    for _ in range(800):
        s = rng.uniform(-1, 1, 6).astype(np.float32)
        joint = int(rng.integers(layout.n_joint))
        reward = float(bin(joint).count("1"))                  # each FBS on pays +1
        a.store(s, joint, [reward], s, terminated=True)
    for _ in range(400):
        a.update()
    probe = rng.uniform(-1, 1, (64, 6)).astype(np.float32)
    q = a.q_values(probe)
    for j in range(layout.n_fbs):
        on = layout.on_mask(j)
        assert np.all(q[:, on].max(axis=1) > q[:, ~on].max(axis=1))


# --------------------------------------------------------------------------- #
# 4-6. training: reproducibility, best pairing, scoring
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("algo", ["ddqn_td3", "ddqn_ddpg"])
def test_seeded_run_reproduces_pilot_and_switch_bitwise(algo):
    r1, _ = _train(algo)
    r2, _ = _train(algo)
    pilot = "td3" if algo == "ddqn_td3" else "ddpg"
    assert _same(_pilot_state(r1, algo=pilot), _pilot_state(r2, algo=pilot))
    assert _same(_switch_state(r1), _switch_state(r2))
    summary = pd.read_json(r1 / "switch_summary.json", typ="series")
    assert summary["n_updates"] > 0, "the switch never updated; the test proves nothing"


def test_runs_stay_out_of_protected_trees(tmp_path):
    rd, _ = _train()
    assert Path(rd).is_relative_to(tmp_path)


def test_best_model_and_best_switch_come_from_the_same_eval():
    rd, _ = _train(total_timesteps=1200, eval_every=200)
    log = pd.read_csv(rd / "eval_log.csv")
    best_t = int(log.loc[log["best_so_far"].astype(str) == "True", "timesteps"].iloc[-1])
    meta = th.load(str(rd / "best_switch.pt"), weights_only=False)["meta"]
    assert meta["timesteps"] == best_t
    # and the pair is not just the final model renamed
    if best_t < 1200:
        assert not _same(_switch_state(rd, "best_switch.pt"), _switch_state(rd))


def test_scoring_is_stable_across_loads():
    from tuning.objective import score_run

    rd, proto = _train()
    a = score_run("ddqn_td3", rd, proto)
    b = score_run("ddqn_td3", rd, proto)
    assert a["objective"] == b["objective"] and a["model"] == "best"
    assert score_run("ddqn_td3", rd, proto, model="final")["model"] == "final"


def test_policy_returns_full_action_and_holds():
    from ddqn_td3.policy import SwitchPilotPolicy
    from ppo.env import FlyingBaseStationEnv

    rd, _ = _train(hold_k=4)
    pol = SwitchPilotPolicy.load(rd, "best")
    exp, _ = _exp()
    env = FlyingBaseStationEnv(exp.env, backend_kind="analytic")
    obs, _ = env.reset(seed=3)
    pol.reset()
    bs = []
    for _ in range(8):
        a, _ = pol.predict(obs)
        assert a.shape == env.action_space.shape
        bs.append(a[[4, 9]].copy())
        obs, *_ = env.step(a)
    assert all(np.array_equal(bs[0], bs[k]) for k in range(4))
    assert all(np.array_equal(bs[4], bs[k]) for k in range(4, 8))


# --------------------------------------------------------------------------- #
# 7. probe neutrality
# --------------------------------------------------------------------------- #
def test_probe_and_compact_log_are_bitwise_neutral(tmp_path):
    from ddqn_td3.callbacks import SwitchProbeCallback, make_compact_step_callback
    from letterexp.pins import pin_events
    from letterexp.probe import binary_dims_and_labels, build_probe_set

    exp, _ = _exp()
    probe = build_probe_set(exp.env, "analytic", n=16, seed=1, out=tmp_path / "probe.npz")
    idx, labels = binary_dims_and_labels(exp.env)
    plain, _ = _train()
    cbs = [SwitchProbeCallback(probe, idx, labels, every=100),
           make_compact_step_callback(None, idx, labels, obs_normalized=True)]
    probed, _ = _train(extra_callbacks=cbs)
    assert _same(_pilot_state(plain), _pilot_state(probed))
    assert _same(_switch_state(plain), _switch_state(probed))

    log = pd.read_csv(probed / "switch_probe.csv")
    assert {f"{lab}_trapped" for lab in labels} <= set(log.columns)
    assert len(log) == 7                                # t = 0, 100, ..., 600
    pin_events(log, labels[0], kind="trap")            # readable by the letter's trap rule
    steps = pd.read_csv(probed / "steps_compact.csv.gz")
    assert set(steps[f"a_{labels[0]}"].unique()) <= {-1.0, 1.0}


# --------------------------------------------------------------------------- #
# 8. the td3/ddpg hooks are neutral
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("algo", ["td3", "ddpg"])
def test_identity_env_wrappers_leave_parent_runs_bitwise_unchanged(algo):
    exp, proto = _exp(algo, code="1-1-1")
    if algo == "td3":
        from td3.train import train
    else:
        from ddpg.train import train
    plain = train(exp, backend=proto.backend)
    hooked = train(exp, backend=proto.backend,
                   vec_env_wrapper=lambda v: v, eval_env_wrapper=lambda e: e)
    assert _same(_pilot_state(plain, algo=algo), _pilot_state(hooked, algo=algo))


# --------------------------------------------------------------------------- #
# --hold-k: the tuner writes K into every trial; a campaign cannot mix K
# --------------------------------------------------------------------------- #
def test_check_hold_k_rules():
    import optuna

    from tuning.study import check_hold_k

    study = optuna.create_study()
    assert check_hold_k(study, "td3", None) is None
    with pytest.raises(SystemExit, match="applies only"):
        check_hold_k(study, "td3", 4)
    with pytest.raises(SystemExit, match="needs --hold-k"):
        check_hold_k(study, "ddqn_td3", None)
    assert check_hold_k(study, "ddqn_td3", 4) == 4
    assert study.user_attrs["hold_k"] == 4
    assert check_hold_k(study, "ddqn_td3", 4) == 4
    with pytest.raises(SystemExit, match="hold_k=4"):
        check_hold_k(study, "ddqn_td3", 8)


def test_search_and_confirm_carry_hold_k_into_every_run():
    import json

    from tuning import paths as tp
    from tuning.study import confirm, search

    proto = _proto(n_trials=2, n_startup_trials=2, confirm_seeds=(0, 1), confirm_top_k=1)
    search("hk", "ddqn_td3", proto, workers=1, quiet=True, hold_k=4)
    with pytest.raises(SystemExit, match="hold_k=4"):
        confirm("hk", "ddqn_td3", proto, workers=1, quiet=True, hold_k=8)
    summary = confirm("hk", "ddqn_td3", proto, workers=1, quiet=True, hold_k=4)
    assert summary["hold_k"] == 4

    specs = [json.loads(f.read_text()) for f in (tp.study_dir("hk") / "specs").glob("ddqn_td3_*.json")
             if not f.name.endswith(".result.json")]
    assert len(specs) == 3 and all(s["hold_k"] == 4 for s in specs)     # 2 search + 1 confirm
    results = [json.loads(f.read_text())
               for f in (tp.study_dir("hk") / "specs").glob("ddqn_td3_*.result.json")]
    assert all(r["status"] == "completed" for r in results), results
    for r in results:
        cfg = json.loads((Path(r["run_dir"]) / "experiment_config.json").read_text())
        assert cfg["provenance"]["ddqn_switch"]["params"]["hold_k"] == 4


def test_letter_spec_key_includes_hold_k_only_when_given():
    from letterexp.runner import spec_key

    base = dict(algo="ddqn_td3", params={"a": 1}, seed=0, protocol={"code": "2-1-1"},
                instrument=True, probe_every=1000)
    assert spec_key(**base) == spec_key(**base, hold_k=None)          # old keys unchanged
    assert len({spec_key(**base, hold_k=k) for k in (1, 4, 8)}) == 3
