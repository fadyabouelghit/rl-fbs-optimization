"""ddqn_ppo/: Hybrid A, the ddqn_td3 switch + a PPO pilot (analytic backend, seconds).

Covers: PPO's spaces (b in its input, c only in its output), seeded
reproducibility, routing away from ppo_runs/ and training_log.csv,
best_model / best_switch pairing, scoring through SwitchPilotPolicy, probe
neutrality, the pinned clip_range, and that hold_k travels through the tuner.
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

ALGO = "ddqn_ppo"


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """No test here may touch ppo_runs/ or training_log.csv."""
    import ddqn_ppo.paths
    import ppo.run_logging
    from tuning import paths as tp

    monkeypatch.setattr(ddqn_ppo.paths, "DDQN_PPO_RUNS_DIR", tmp_path / "ddqn_ppo")
    monkeypatch.setattr(ppo.run_logging, "LEDGER_PATH", tmp_path / "ppo_ledger_must_not_exist.csv")
    monkeypatch.setattr(ppo.run_logging, "RUNS_DIR", tmp_path / "ppo_runs_must_not_exist")
    monkeypatch.setattr(tp, "TUNING_DIR", tmp_path / "tuning_runs")


def _proto(code="2-1-1", **kw):
    kw.setdefault("fixed_batch_size", 32)            # so the switch updates in 600 steps
    return replace(SMOKE, code=code, **kw)


def _params(**kw):
    p = dict(SEED_TRIALS[ALGO][0])
    p.update(n_steps=512, n_epochs=2)                # keep PPO's update cheap
    p.update(kw)
    return p


def _exp(code="2-1-1", params=None, **kw):
    proto = _proto(code, **kw)
    return build_experiment(ALGO, params or _params(), proto, 0, "t"), proto


def _train(hold_k=1, extra_callbacks=None, params=None, **proto_kw):
    from ddqn_ppo.train import train_algo

    params = params or _params()
    exp, proto = _exp(params=params, **proto_kw)
    rd = train_algo(ALGO, exp, proto.backend, params=params, hold_k=hold_k,
                    extra_callbacks=extra_callbacks)
    return Path(rd), proto


def _pilot_state(run_dir, file="model.zip"):
    from stable_baselines3 import PPO
    return PPO.load(str(Path(run_dir) / file), device="cpu").policy.state_dict()


def _switch_state(run_dir, file="switch.pt"):
    return th.load(str(Path(run_dir) / file), weights_only=False)["q"]


def _same(a, b):
    return a.keys() == b.keys() and all(th.equal(a[k], b[k]) for k in a)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def test_space_has_five_knobs_and_clip_range_is_pinned():
    from ddqn_ppo.config import PINNED_CLIP_RANGE
    from tuning.spaces import KNOBS_PER_ALGO, space

    names = [k.name for k in space(ALGO)]
    assert len(names) == KNOBS_PER_ALGO and "clip_range" not in names
    assert "exploration_fraction" in names
    exp, _ = _exp(params=_params(clip_range=0.1))      # a stray clip_range is ignored
    assert exp.ppo.clip_range == PINNED_CLIP_RANGE == 0.3
    assert exp.ppo.algo == "ppo" and exp.ppo.n_envs == 1


def test_switch_matches_ddqn_td3s_learning_start():
    from ddqn_ppo.train import switch_params
    from td3.config import TD3Params

    sw = switch_params({"exploration_fraction": 0.2}, hold_k=1)
    assert sw.learning_starts == TD3Params().learning_starts == 100


def test_building_the_switch_leaves_the_global_torch_stream_alone():
    from ddqn_td3.config import SwitchParams
    from ddqn_td3.switch import SwitchAgent

    th.manual_seed(0)
    before = th.random.get_rng_state().clone()
    SwitchAgent(18, 4, SwitchParams(), gamma=0.95, learning_rate=3e-4, batch_size=32,
                net_arch=(64, 64), total_timesteps=100, learning_starts=0, seed=0)
    assert th.equal(before, th.random.get_rng_state())


# --------------------------------------------------------------------------- #
# PPO sees b, outputs c only
# --------------------------------------------------------------------------- #
def test_ppo_sees_b_and_outputs_only_the_continuous_genes():
    from stable_baselines3 import PPO

    rd, _ = _train()
    model = PPO.load(str(rd / "model.zip"), device="cpu")
    assert model.observation_space.shape == (16 + 2,)
    assert model.action_space.shape == (8,)
    cfg = pd.read_json(rd / "experiment_config.json", typ="series")
    meta = cfg["provenance"]["ddqn_switch"]
    assert meta["pilot"] == "ppo" and meta["n_joint"] == 4
    steps = pd.read_csv(rd / "steps.csv")
    status = [c for c in steps.columns if c.endswith("power_status")]
    assert len(status) == 2 and set(np.unique(steps[status].to_numpy())) <= {0.0, 1.0}


def test_terminal_observation_carries_the_executed_b():
    from stable_baselines3.common.vec_env import DummyVecEnv

    from ddqn_td3.config import Layout, SwitchParams
    from ddqn_td3.switch import SwitchAgent
    from ddqn_td3.wrapper import SwitchVecEnv
    from ppo.env import FlyingBaseStationEnv

    exp, _ = _exp()
    env = FlyingBaseStationEnv(exp.env, backend_kind="analytic")
    venv = DummyVecEnv([lambda: env])
    layout = Layout(exp.env)
    agent = SwitchAgent(16, 4, SwitchParams(), gamma=0.95, learning_rate=3e-4, batch_size=32,
                        net_arch=(64, 64), total_timesteps=100, learning_starts=0, seed=0)
    sw = SwitchVecEnv(venv, agent, layout)
    sw.reset()
    for t in range(exp.env.max_episode_steps):
        b = sw.b.copy()
        _, _, dones, infos = sw.step(np.zeros((1, 8), np.float32))
    assert dones[0] and infos[0]["TimeLimit.truncated"]
    term = infos[0]["terminal_observation"]
    assert term.shape == (18,) and np.array_equal(term[-2:], b)


# --------------------------------------------------------------------------- #
# reproducibility, routing, pairing, scoring
# --------------------------------------------------------------------------- #
def test_seeded_run_reproduces_pilot_and_switch_bitwise():
    r1, _ = _train()
    r2, _ = _train()
    assert _same(_pilot_state(r1), _pilot_state(r2))
    assert _same(_switch_state(r1), _switch_state(r2))
    summary = pd.read_json(r1 / "switch_summary.json", typ="series")
    assert summary["n_updates"] > 0, "the switch never updated; the test proves nothing"


def test_runs_and_ledger_stay_out_of_ppo_trees(tmp_path):
    import ppo.run_logging

    rd, _ = _train()
    assert rd.is_relative_to(tmp_path / "ddqn_ppo")
    ledger = pd.read_csv(tmp_path / "ddqn_ppo" / "ledger.csv")
    assert list(ledger["algo"]) == ["ddqn_ppo"] and ledger["clip_range"].iloc[0] == 0.3
    assert ledger["switch_hold_k"].iloc[0] == 1
    assert not Path(ppo.run_logging.LEDGER_PATH).exists()
    assert not (tmp_path / "ppo_runs_must_not_exist").exists()


def test_best_model_and_best_switch_come_from_the_same_eval():
    rd, _ = _train(total_timesteps=1200, eval_every=200)
    log = pd.read_csv(rd / "eval_log.csv")
    best_t = int(log.loc[log["best_so_far"].astype(str) == "True", "timesteps"].iloc[-1])
    meta = th.load(str(rd / "best_switch.pt"), weights_only=False)["meta"]
    assert meta["timesteps"] == best_t
    if best_t < 1200:
        assert not _same(_switch_state(rd, "best_switch.pt"), _switch_state(rd))


def test_scoring_is_stable_across_loads():
    from tuning.objective import score_run

    rd, proto = _train()
    a = score_run(ALGO, rd, proto)
    b = score_run(ALGO, rd, proto)
    assert a["objective"] == b["objective"] and a["model"] == "best"
    assert score_run(ALGO, rd, proto, model="final")["model"] == "final"


def test_policy_returns_full_action_with_greedy_b_and_ppo_mean():
    from stable_baselines3 import PPO

    from ddqn_td3.policy import SwitchPilotPolicy
    from ppo.env import FlyingBaseStationEnv

    rd, _ = _train()
    pol = SwitchPilotPolicy.load(rd, "best")
    ppo_model = PPO.load(str(rd / "best_model.zip"), device="cpu")
    exp, _ = _exp()
    env = FlyingBaseStationEnv(exp.env, backend_kind="analytic")
    obs, _ = env.reset(seed=3)
    pol.reset()
    a, _ = pol.predict(obs)
    assert a.shape == env.action_space.shape
    assert set(np.unique(a[[4, 9]])) <= {-1.0, 1.0}
    c, _ = ppo_model.predict(np.concatenate([obs, a[[4, 9]]]).astype(np.float32),
                             deterministic=True)
    np.testing.assert_array_equal(a[[0, 1, 2, 3, 5, 6, 7, 8]], c)


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
    pin_events(log, labels[0], kind="trap")
    steps = pd.read_csv(probed / "steps_compact.csv.gz")
    assert set(steps[f"a_{labels[0]}"].unique()) <= {-1.0, 1.0}


# --------------------------------------------------------------------------- #
# tuner and letter plumbing
# --------------------------------------------------------------------------- #
def test_hold_k_is_required_and_carried_through_search():
    import json

    import optuna

    from tuning import paths as tp
    from tuning.study import check_hold_k, search

    with pytest.raises(SystemExit, match="needs --hold-k"):
        check_hold_k(optuna.create_study(), ALGO, None)
    proto = _proto(n_trials=1, n_startup_trials=1)
    search("hk", ALGO, proto, workers=1, quiet=True, hold_k=1)
    results = [json.loads(f.read_text())
               for f in (tp.study_dir("hk") / "specs").glob(f"{ALGO}_*.result.json")]
    assert len(results) == 1 and results[0]["status"] == "completed", results
    rd = Path(results[0]["run_dir"])
    assert rd.is_relative_to(tp.study_dir("hk") / f"{ALGO}_runs")
    cfg = json.loads((rd / "experiment_config.json").read_text())
    assert cfg["provenance"]["ddqn_switch"]["params"]["hold_k"] == 1
    assert (tp.study_dir("hk") / f"{ALGO}_runs" / "ledger.csv").is_file()


def test_ha_pilot_plan_is_t35_plus_the_default_switch(tmp_path, monkeypatch):
    import letterexp.paths
    from letterexp.launch import build_specs

    monkeypatch.setattr(letterexp.paths, "LETTER_RUNS_DIR", tmp_path / "letter")   # build_specs mkdirs
    from letterexp.runner import FAIR2_DIR, base_params

    t35 = base_params("ppo_t35", FAIR2_DIR)["params"]
    specs = build_specs("ha_pilot")
    assert len(specs) == 20 and {s["seed"] for s in specs} == set(range(20))
    p = specs[0]["params"]
    assert "clip_range" not in p and p["exploration_fraction"] == 0.2
    assert {k: p[k] for k in ("ent_coef", "n_steps", "n_epochs", "gae_lambda")} == \
        {k: t35[k] for k in ("ent_coef", "n_steps", "n_epochs", "gae_lambda")}
    assert t35["clip_range"] == 0.3 and all(s["hold_k"] == 1 for s in specs)
