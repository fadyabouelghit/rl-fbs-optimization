"""Shared tuner (tuning/ package) on the analytic backend (no MATLAB).

The tuner imports ppo/, td3/ and ddpg/ and modifies none of them. These
tests pin the properties a cross-algorithm comparison actually depends on:
that the three algorithms get the same environment, budget, objective and
shared hyperparameters; that each searches exactly five knobs; that every
search space is flat (GPSampler silently degrades to random sampling on
conditional parameters); that the enqueued starting points specify every
knob; that knobs a package cannot express are declared rather than quietly
dropped; and that a campaign's artifacts stay inside
tuning_runs/ instead of landing in ppo_runs/, td3_runs/, ddpg_runs/ or
training_log.csv.
"""
from dataclasses import replace

import numpy as np
import optuna
import pytest

from tuning import protocol as P
from tuning.spaces import (
    KNOBS_PER_ALGO, SEED_TRIALS, SHARED, SPECIFIC, UNSUPPORTED,
    build_experiment, space, suggest,
)

ALGOS = P.ALGOS

# Small enough to run in a test, real enough to exercise train -> select ->
# score. eval_every must stay under total_timesteps or no best_model is written.
TINY = replace(
    P.SMOKE, total_timesteps=400, eval_every=200, eval_episodes=1,
    objective_episodes=2, n_trials=2, n_startup_trials=2,
)


@pytest.fixture(autouse=True)
def _isolate_tuning_paths(tmp_path, monkeypatch):
    """Keep campaigns out of the real tuning_runs/."""
    from tuning import paths as tp

    monkeypatch.setattr(tp, "TUNING_DIR", tmp_path / "tuning_runs")
    yield


# --------------------------------------------------------------------------- #
# Search spaces
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("algo", ALGOS)
def test_space_is_flat(algo):
    """Every trial suggests exactly the same parameters.

    ``GPSampler`` does not support dynamic search spaces: a parameter present
    on only some trials is handed to ``RandomSampler`` instead of the GP. A
    conditional knob would therefore be searched at random without failing
    anything, which is why this is a test and not a comment.
    """
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    keysets = set()
    for _ in range(12):
        trial = study.ask()
        keysets.add(frozenset(suggest(algo, trial)))
        study.tell(trial, 0.0)
    assert len(keysets) == 1
    assert keysets.pop() == {k.name for k in space(algo)}


@pytest.mark.parametrize("algo", ALGOS)
def test_each_algorithm_searches_exactly_five_knobs(algo):
    """Rule 2: equal budget only buys equal coverage at equal dimensionality."""
    assert len(space(algo)) == KNOBS_PER_ALGO


def test_no_knob_is_both_shared_and_searched():
    """Rule 1: the shared hyperparameters are pinned by the protocol. If one
    is ever re-opened it must appear in SHARED for every algorithm, not in
    one algorithm's specific list."""
    for algo in ALGOS:
        by_name = {k.name: k for k in space(algo)}
        for shared in SHARED:
            assert by_name[shared.name] == shared, f"{algo}.{shared.name} diverged"
        for pinned in ("learning_rate", "gamma", "batch_size", "net_arch"):
            assert pinned not in {k.name for k in SPECIFIC[algo]}


def test_the_pinned_shared_values_reach_all_three_configs():
    for algo, exp in _experiments().items():
        p = getattr(exp, algo)
        assert p.learning_rate == TINY.fixed_learning_rate
        assert p.gamma == TINY.fixed_gamma
        assert p.batch_size == TINY.fixed_batch_size


def test_all_three_run_the_same_architecture():
    """PPO cannot be given a net_arch (ppo/ passes no policy_kwargs), so the
    only architecture the three can share is PPO's own default. TD3 and DDPG
    are pinned to it explicitly; PPO reaches it by leaving SB3 alone."""
    from ppo.config import PPOParams

    assert not hasattr(PPOParams, "net_arch")
    assert TINY.fixed_net_arch == (64, 64)          # SB3's PPO MlpPolicy default
    for algo in ("td3", "ddpg"):
        exp = build_experiment(algo, SEED_TRIALS[algo][0], TINY, 0, "t")
        assert getattr(exp, algo).net_arch == TINY.fixed_net_arch


def test_td3_can_express_binary_noise():
    """Ported from ddpg/ for this campaign: without it TD3 cannot explore the
    binary on/off gene at all, and the comparison would measure which package
    implements the knob rather than which algorithm is better."""
    from dataclasses import fields

    from td3.config import TD3Params

    assert "binary_noise_std" in {f.name for f in fields(TD3Params)}
    assert "binary_noise_std" in {k.name for k in space("td3")}


@pytest.mark.parametrize("algo", ALGOS)
def test_enqueued_start_points_specify_every_knob(algo):
    """A partially specified enqueue_trial leaves the rest to the sampler,
    which silently turns a known-good configuration into a random one."""
    names = {k.name for k in space(algo)}
    for seeded in SEED_TRIALS[algo]:
        assert set(seeded) == names, f"{algo}: {names ^ set(seeded)}"


@pytest.mark.parametrize("algo", ALGOS)
def test_enqueued_values_are_inside_their_ranges(algo):
    for seeded in SEED_TRIALS[algo]:
        for knob in space(algo):
            v = seeded[knob.name]
            if knob.kind == "categorical":
                assert v in knob.choices, f"{algo}.{knob.name}={v!r}"
            else:
                assert knob.low <= v <= knob.high, f"{algo}.{knob.name}={v!r}"


def test_declared_gaps_are_real_and_currently_none():
    """``UNSUPPORTED`` is the printed record of knobs a package cannot
    express. Both original entries are closed, so it should be empty -- and
    anything left in it must genuinely be missing from the dataclass, or the
    printed warning is stale."""
    from dataclasses import fields

    from ddpg.config import DDPGParams
    from ppo.config import PPOParams
    from td3.config import TD3Params

    params = {"ppo": PPOParams, "td3": TD3Params, "ddpg": DDPGParams}
    assert UNSUPPORTED == {}
    for (algo, knob) in UNSUPPORTED:
        assert knob not in {f.name for f in fields(params[algo])}
        assert knob not in {k.name for k in space(algo)}


# --------------------------------------------------------------------------- #
# params -> experiment
# --------------------------------------------------------------------------- #
def _experiments(protocol=TINY, seed=0):
    return {a: build_experiment(a, SEED_TRIALS[a][0], protocol, seed, "t0000")
            for a in ALGOS}


def test_all_three_get_an_identical_environment():
    """The comparison is only about algorithms if the env half matches."""
    envs = [e.env for e in _experiments().values()]
    assert envs[0] == envs[1] == envs[2]


def test_all_three_get_an_identical_budget():
    exps = _experiments()
    budgets = {
        a: (getattr(e, a).total_timesteps, getattr(e, a).n_envs,
            getattr(e, a).eval_every, getattr(e, a).eval_episodes,
            getattr(e, a).seed, getattr(e, a).normalize_reward,
            getattr(e, a).learning_rate, getattr(e, a).gamma,
            getattr(e, a).batch_size)
        for a, e in exps.items()
    }
    assert len(set(budgets.values())) == 1, budgets


def test_sampled_params_reach_the_algorithm_config():
    params = dict(SEED_TRIALS["ddpg"][0], action_noise_std=0.22,
                  binary_noise_std=0.65, noise_type="ou", ou_dt=0.5, tau=0.017)
    d = build_experiment("ddpg", params, TINY, 3, "t0001").ddpg
    assert (d.action_noise_std, d.binary_noise_std) == (0.22, 0.65)
    assert (d.noise_type, d.ou_dt, d.tau, d.seed) == ("ou", 0.5, 0.017, 3)
    # ...and the pinned ones are not taken from the sample even if present.
    assert (d.learning_rate, d.gamma) == (TINY.fixed_learning_rate, TINY.fixed_gamma)


def test_td3_sampled_binary_noise_reaches_the_noise_vector():
    from td3.train import noise_sigma

    params = dict(SEED_TRIALS["td3"][0], action_noise_std=0.1, binary_noise_std=0.8)
    exp = build_experiment("td3", params, TINY, 0, "t")
    np.testing.assert_allclose(noise_sigma(exp, 5), [0.1, 0.1, 0.1, 0.1, 0.8])


def test_ppo_batch_size_is_clamped_to_the_rollout_length():
    """PPOParams rejects batch_size > n_steps * n_envs. The shared batch_size
    grid and the PPO-only n_steps grid can collide there, and a sampled
    config must never raise instead of training."""
    for n_steps in (512, 1024, 2048):
        p = build_experiment("ppo", dict(SEED_TRIALS["ppo"][0], n_steps=n_steps),
                             TINY, 0, "t0002").ppo
        assert p.batch_size <= p.n_steps * TINY.n_envs
        assert p.batch_size == TINY.fixed_batch_size   # 256 never binds at 512+


def test_unknown_algo_is_rejected():
    with pytest.raises(ValueError):
        space("sac")
    with pytest.raises(ValueError):
        build_experiment("sac", {}, TINY, 0, "t")


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #
def test_objective_and_reporting_seeds_are_disjoint():
    """Tune on one draw, report on another (page recommendation 5), and never
    select on either -- CsvEvalCallback picks best_model on its own seed."""
    assert P.DEFAULT.objective_seed != P.REPORT_SEED
    assert P.DEFAULT.objective_seed == 987_654       # assess_models.ASSESS_SEED
    from ppo.callbacks import CsvEvalCallback

    import inspect
    selection_seed = inspect.signature(CsvEvalCallback.__init__).parameters["eval_seed"].default
    assert selection_seed not in (P.DEFAULT.objective_seed, P.REPORT_SEED)


def test_no_pruner_is_configured():
    """Measured on the 56-run DDPG campaign: Spearman(J@25k, J@100k) = +0.26,
    and the best run ranked 29th of 54 at 25k. Early stopping discards it."""
    from tuning.study import make_study

    study = make_study("nopruner", "ddpg", TINY)
    assert isinstance(study.pruner, optuna.pruners.NopPruner)


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("algo", ALGOS)
def test_search_runs_end_to_end_and_keeps_its_artifacts_to_itself(algo, tmp_path):
    from tuning import paths as tp
    from tuning.study import search

    study = search("e2e", algo, TINY, workers=1, poll_seconds=0.2, quiet=True)

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(complete) == TINY.n_trials
    assert all(t.value is not None for t in complete)
    assert all("run_dir" in t.user_attrs for t in complete)

    campaign = tp.study_dir("e2e")
    assert (campaign / f"{algo}_trials.csv").is_file()
    # Every run, and the ledger, live inside the campaign.
    runs = campaign / f"{algo}_runs"
    assert (runs / "ledger.csv").is_file()
    for t in complete:
        assert runs in tp.Path(t.user_attrs["run_dir"]).parents


def test_score_run_falls_back_to_the_final_model(tmp_path):
    """A run too short for the in-training evaluator to have fired has no
    best_model.zip. Scoring it should report the fallback, not crash."""
    from ddpg.config import DDPGExperiment
    from ddpg.train import train
    from tuning.objective import score_run

    proto = replace(TINY, eval_every=0)
    exp = build_experiment("ddpg", SEED_TRIALS["ddpg"][0], proto, 0, "nobest")
    run_dir = train(exp, backend=proto.backend, run_dir=tmp_path / "run")
    assert not (run_dir / "best_model.zip").exists()

    out = score_run("ddpg", run_dir, proto)
    assert out["model_requested"] == "best" and out["model"] == "final"
    assert isinstance(out["objective"], float)
