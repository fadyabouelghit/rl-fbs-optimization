"""Search spaces, and the map from a sampled config to a runnable experiment.

Three rules:

1. Shared knobs are **fixed**, not searched. Learning rate, discount, batch
   size and network architecture are held at one common value for all three
   algorithms (``Protocol.fixed_*``), so a trial only ever varies what is
   specific to its algorithm.
2. Each algorithm searches exactly **five** knobs, so the budget buys the
   same coverage for each.
3. Every space is **flat** -- fixed shape, no conditional parameters.

Rule 3 is not cosmetic. ``optuna.samplers.GPSampler`` does not support
dynamic search spaces: a parameter that only exists on some trials is
silently handed to ``RandomSampler`` instead of the GP (measured: 6 of 20
trials in a conditional toy space). So ``ou_dt`` is suggested on *every*
DDPG trial and simply ignored by the config when ``noise_type == 'normal'``.
The Matern kernel's automatic relevance determination learns a long length
scale for the dimension on the trials where it does nothing, which is the
graceful way to carry an irrelevant knob; dropping to random sampling is not.

``UNSUPPORTED`` records knobs a comparison wants that a package cannot
express, so ``python -m tuning spaces`` prints the asymmetry instead of
hiding it. It is currently empty: ``td3.binary_noise_std`` was added to
``td3/``, and ``ppo.net_arch`` stopped mattering once every algorithm was
pinned to PPO's own (64, 64).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .protocol import Protocol


# --------------------------------------------------------------------------- #
# Knob
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Knob:
    name: str
    kind: str                                   # 'float' | 'int' | 'categorical'
    low: Optional[float] = None
    high: Optional[float] = None
    log: bool = False
    choices: Optional[Tuple[Any, ...]] = None
    note: str = ""

    def suggest(self, trial) -> Any:
        if self.kind == "float":
            return trial.suggest_float(self.name, self.low, self.high, log=self.log)
        if self.kind == "int":
            return trial.suggest_int(self.name, int(self.low), int(self.high), log=self.log)
        if self.kind == "categorical":
            return trial.suggest_categorical(self.name, list(self.choices))
        raise ValueError(f"unknown knob kind {self.kind!r}")

    @property
    def range_str(self) -> str:
        if self.kind == "categorical":
            return " / ".join(str(c) for c in self.choices)
        return f"{self.low:g} - {self.high:g}" + (" (log)" if self.log else "")


# --------------------------------------------------------------------------- #
# The spaces
# --------------------------------------------------------------------------- #
# Nothing is shared *and* searched: the shared hyperparameters are pinned by
# the protocol (see ``Protocol.fixed_*``). Kept as an explicit empty tuple so
# the distinction between "shared" and "specific" survives in the code, and
# so re-opening a shared knob is a one-line change.
SHARED: Tuple[Knob, ...] = ()

SPECIFIC: Dict[str, Tuple[Knob, ...]] = {
    "ppo": (
        Knob("ent_coef", "float", 1e-4, 0.05, log=True),
        Knob("n_steps", "categorical", choices=(512, 1024, 2048)),
        Knob("n_epochs", "categorical", choices=(5, 10, 20)),
        Knob("gae_lambda", "float", 0.90, 0.99),
        Knob("clip_range", "categorical", choices=(0.1, 0.2, 0.3)),
    ),
    "td3": (
        Knob("action_noise_std", "float", 0.05, 0.30),
        Knob("binary_noise_std", "float", 0.20, 1.00,
             note="added to td3/ for this campaign; TD3 has DDPG's blind spot"),
        Knob("policy_delay", "categorical", choices=(1, 2, 4)),
        Knob("target_policy_noise", "float", 0.05, 0.40),
        Knob("tau", "float", 1e-3, 2e-2, log=True),
    ),
    "ddpg": (
        Knob("action_noise_std", "float", 0.05, 0.30),
        Knob("binary_noise_std", "float", 0.20, 1.00,
             note="the biggest single lever in the 56-run campaign"),
        Knob("noise_type", "categorical", choices=("normal", "ou")),
        # Always suggested to keep the space flat (see the module docstring).
        # Ignored unless noise_type == 'ou'. Range chosen against a 40-step
        # episode: OU decorrelates over ~1/(theta*dt) steps, so SB3's default
        # dt=0.01 is a ~670-step correlation length the episode never sees.
        Knob("ou_dt", "float", 1e-2, 1.0, log=True),
        Knob("tau", "float", 1e-3, 2e-2, log=True),
    ),
}

# Hybrid variants (letter Stage 4). Same five-knob budget as their parent:
# ``binary_noise_std`` has no meaning once the on/off dim is a Bernoulli
# sample with zero added noise, so its slot goes to the Gumbel temperature.
SPECIFIC["ddpg_hybrid"] = (
    Knob("action_noise_std", "float", 0.05, 0.30),
    Knob("gumbel_tau", "float", 0.25, 4.0, log=True,
         note="replaces binary_noise_std; straight-through Gumbel temperature"),
    Knob("noise_type", "categorical", choices=("normal", "ou")),
    Knob("ou_dt", "float", 1e-2, 1.0, log=True),
    Knob("tau", "float", 1e-3, 2e-2, log=True),
)
# The Q-difference variants (letter Step 2) search exactly what their
# straight-through counterparts do: only the actor-loss estimator differs,
# and gumbel_tau still governs rollout sampling.
SPECIFIC["td3_hybrid"] = (
    Knob("action_noise_std", "float", 0.05, 0.30),
    Knob("gumbel_tau", "float", 0.25, 4.0, log=True,
         note="replaces binary_noise_std; straight-through Gumbel temperature"),
    Knob("policy_delay", "categorical", choices=(1, 2, 4)),
    Knob("target_policy_noise", "float", 0.05, 0.40),
    Knob("tau", "float", 1e-3, 2e-2, log=True),
)

SPECIFIC["ddpg_hybrid_qd"] = SPECIFIC["ddpg_hybrid"]
SPECIFIC["td3_hybrid_qd"] = SPECIFIC["td3_hybrid"]

# Every algorithm searches the same number of dimensions.
KNOBS_PER_ALGO = 5


# --------------------------------------------------------------------------- #
# Widened TD3 space -- NOT part of the equal-budget comparison
# --------------------------------------------------------------------------- #
# In campaign fair1, TD3's stage-2 finalists sat on their range ceilings:
# target_policy_noise 7/8, tau 6/8, action_noise_std 6/8. An optimum on a
# boundary means the real optimum is probably outside the box, so TD3's
# 0.4740 is a lower bound of unknown tightness.
#
# This variant lifts exactly those three ceilings and changes nothing else,
# so the probe answers one question: how much is the box costing TD3? It is
# deliberately a *separate* space rather than a widened default -- rerunning
# TD3 with more room while PPO and DDPG keep their original boxes would
# break the equal-budget contract that makes fair1 a fair comparison.
#
# Caveat to record with any result: target_noise_clip stays pinned at SB3's
# 0.5, so target_policy_noise above ~0.5 saturates -- at 0.8 roughly half of
# the smoothing draws are clipped, and the noise approaches a two-point mass
# at +-0.5. The probe therefore under-explores that knob's upper range.
SPECIFIC_WIDE: Dict[str, Tuple[Knob, ...]] = {
    "td3": (
        Knob("action_noise_std", "float", 0.05, 0.60, note="ceiling 0.30 -> 0.60"),
        Knob("binary_noise_std", "float", 0.20, 1.00, note="unchanged; was interior"),
        Knob("policy_delay", "categorical", choices=(1, 2, 4), note="unchanged"),
        Knob("target_policy_noise", "float", 0.05, 0.80, note="ceiling 0.40 -> 0.80"),
        Knob("tau", "float", 1e-3, 5e-2, log=True, note="ceiling 0.02 -> 0.05"),
    ),
}

# The fair1 stage-2 winner, so the probe starts from the best known point and
# any gain is measured against it directly.
SEED_TRIALS_WIDE: Dict[str, Tuple[Dict[str, Any], ...]] = {
    "td3": ({"action_noise_std": 0.300, "binary_noise_std": 0.7696,
             "policy_delay": 1, "target_policy_noise": 0.400, "tau": 0.0016},),
}

# Knobs a fair comparison wants that a package cannot express. Declared
# rather than silently omitted, so `python -m tuning spaces` prints them.
#
# Both original entries are closed:
#   td3.binary_noise_std  -- implemented in td3/ (TD3Params.binary_noise_std)
#   ppo.net_arch          -- moot: every algorithm is pinned to PPO's (64,64)
UNSUPPORTED: Dict[Tuple[str, str], str] = {}


def space(algo: str, wide: bool = False) -> List[Knob]:
    if algo not in SPECIFIC:
        raise ValueError(f"unknown algo {algo!r}; expected one of {sorted(SPECIFIC)}")
    if wide:
        if algo not in SPECIFIC_WIDE:
            raise ValueError(f"no widened space defined for {algo!r}")
        return list(SHARED) + list(SPECIFIC_WIDE[algo])
    return list(SHARED) + list(SPECIFIC[algo])


def suggest(algo: str, trial, wide: bool = False) -> Dict[str, Any]:
    return {k.name: k.suggest(trial) for k in space(algo, wide)}


def seed_trials(algo: str, wide: bool = False) -> Tuple[Dict[str, Any], ...]:
    return (SEED_TRIALS_WIDE if wide else SEED_TRIALS).get(algo, ())


# --------------------------------------------------------------------------- #
# Known-good starting points (optuna's enqueue_trial)
# --------------------------------------------------------------------------- #
# Recommendation 3 of the page: start the search from the best known point
# rather than from scratch. Every value here is a config already on disk.
SEED_TRIALS: Dict[str, Tuple[Dict[str, Any], ...]] = {
    # The potrec10 PPO configuration used for all 210 seed runs.
    "ppo": ({
        "ent_coef": 0.01, "n_steps": 1024, "n_epochs": 10,
        "gae_lambda": 0.95, "clip_range": 0.2,
    },),
    # The best TD3 arm from the lr/noise sweep before the line was paused,
    # with DDPG's tuned binary noise as the prior for the knob TD3 has never
    # had -- there is no TD3 measurement of it to start from.
    "td3": ({
        "action_noise_std": 0.30, "binary_noise_std": 0.50,
        "policy_delay": 2, "target_policy_noise": 0.2, "tau": 5e-3,
    },),
    # The tuned DDPG config (5 seeds: final 0.466 +/- 0.004). Its gamma and
    # learning rate are now pinned by the protocol, so the two campaign
    # winners (gamma 0.9 and 0.85) collapse to this single starting point.
    "ddpg": ({
        "action_noise_std": 0.10, "binary_noise_std": 0.50,
        "noise_type": "normal", "ou_dt": 1e-2, "tau": 5e-3,
    },),
    # Hybrids start from the SAME pre-fair1 points their parents started from
    # (not from the fair1 winners, which would hand them 60 trials of the
    # parent's search for free), with binary_noise_std -> gumbel_tau 1.0.
    "ddpg_hybrid": ({
        "action_noise_std": 0.10, "gumbel_tau": 1.0,
        "noise_type": "normal", "ou_dt": 1e-2, "tau": 5e-3,
    },),
    "td3_hybrid": ({
        "action_noise_std": 0.30, "gumbel_tau": 1.0,
        "policy_delay": 2, "target_policy_noise": 0.2, "tau": 5e-3,
    },),
}
SEED_TRIALS["ddpg_hybrid_qd"] = SEED_TRIALS["ddpg_hybrid"]
SEED_TRIALS["td3_hybrid_qd"] = SEED_TRIALS["td3_hybrid"]


# --------------------------------------------------------------------------- #
# params -> experiment
# --------------------------------------------------------------------------- #
def build_experiment(algo: str, params: Dict[str, Any], protocol: Protocol,
                     seed: int, tag: str):
    """Turn a sampled config into the algorithm's own ``*Experiment``.

    Built through each package's ``from_code``, so the env half is identical
    across the three by construction rather than by three shell scripts
    agreeing with each other.
    """
    if algo not in SPECIFIC:
        raise ValueError(f"unknown algo {algo!r}; expected one of {sorted(SPECIFIC)}")

    overrides = dict(protocol.shared_algo_overrides())
    overrides["gamma"] = protocol.fixed_gamma
    overrides["batch_size"] = protocol.fixed_batch_size

    common = dict(
        band=None,
        reward=protocol.reward(),
        env_overrides=protocol.env_overrides(),
        tag=tag,
        max_episode_steps=protocol.max_episode_steps,
        action_scale=protocol.action_scale,
        learning_rate=protocol.fixed_learning_rate,
        seed=seed,
        n_envs=protocol.n_envs,
    )

    if algo == "ppo":
        from ppo.config import ExperimentConfig

        overrides.update(
            algo="ppo",
            ent_coef=params["ent_coef"],
            n_steps=params["n_steps"],
            n_epochs=params["n_epochs"],
            gae_lambda=params["gae_lambda"],
            clip_range=params["clip_range"],
        )
        # PPOParams validates batch_size <= n_steps * n_envs. With the batch
        # size pinned at 256 and the smallest n_steps at 512 this never
        # binds, but it is kept so that widening either one cannot turn a
        # sampled config into a crashed trial.
        overrides["batch_size"] = min(overrides["batch_size"],
                                      params["n_steps"] * protocol.n_envs)
        return ExperimentConfig.from_code(protocol.code, ppo_overrides=overrides, **common)

    if algo == "td3":
        from td3.config import TD3Experiment

        overrides.update(
            net_arch=protocol.fixed_net_arch,
            action_noise_std=params["action_noise_std"],
            binary_noise_std=params["binary_noise_std"],
            policy_delay=params["policy_delay"],
            target_policy_noise=params["target_policy_noise"],
            tau=params["tau"],
        )
        return TD3Experiment.from_code(protocol.code, td3_overrides=overrides, **common)

    if algo == "ddpg":
        from ddpg.config import DDPGExperiment

        overrides.update(
            net_arch=protocol.fixed_net_arch,
            action_noise_std=params["action_noise_std"],
            binary_noise_std=params["binary_noise_std"],
            noise_type=params["noise_type"],
            ou_dt=params["ou_dt"],
            tau=params["tau"],
        )
        return DDPGExperiment.from_code(protocol.code, ddpg_overrides=overrides, **common)

    if algo in ("ddpg_hybrid", "ddpg_hybrid_qd"):
        from ddpg.config import DDPGExperiment

        overrides.update(
            net_arch=protocol.fixed_net_arch,
            action_noise_std=params["action_noise_std"],
            binary_noise_std=0.0,          # the Bernoulli sample is the exploration
            noise_type=params["noise_type"],
            ou_dt=params["ou_dt"],
            tau=params["tau"],
        )
        return DDPGExperiment.from_code(protocol.code, ddpg_overrides=overrides, **common)

    if algo in ("td3_hybrid", "td3_hybrid_qd"):
        from td3.config import TD3Experiment

        overrides.update(
            net_arch=protocol.fixed_net_arch,
            action_noise_std=params["action_noise_std"],
            binary_noise_std=0.0,
            policy_delay=params["policy_delay"],
            target_policy_noise=params["target_policy_noise"],
            tau=params["tau"],
        )
        return TD3Experiment.from_code(protocol.code, td3_overrides=overrides, **common)

    raise ValueError(f"unknown algo {algo!r}")
