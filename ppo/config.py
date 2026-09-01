"""Configuration layer for the PPO pipeline.

Three levels of config:

- ``WorldConfig`` / ``BandConfig`` / ``RewardConfig``: the physics scenario.
- ``PPOParams``: SB3 hyperparameters + run budget.
- ``ExperimentConfig``: the user-facing bundle. Built from an X-Y-Z code
  (X = num FBS, Y = scenario id, Z = cost config id) plus overrides, with a
  ``band`` block for the multi-band world.

Serialization is versioned: ``to_json_dict`` writes ``config_version: 2``.
``experiment_from_json_dict`` also accepts the v1 schema written by the old
``ppo_experiment.py`` so historical runs stay loadable.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Optional, Sequence, Tuple

CONFIG_VERSION = 2

# Per-FBS gene bounds (continuous part of the placement vector).
FBS_Z_BOUNDS = (20.0, 150.0)
FBS_POWER_BOUNDS = (7.0, 10.5)

# Band ids follow band_frequencies.m: 0 = coverage, 1 = capacity.
BAND_COVERAGE = 0
BAND_CAPACITY = 1

# Reward modes renamed in the standalone RL repo. Old names are still accepted
# so ``experiment_config.json`` files written before the rename keep loading.
REWARD_MODE_ALIASES = {"ga_blend": "controlled_blend"}


# --------------------------------------------------------------------------- #
# Reward
# --------------------------------------------------------------------------- #
@dataclass
class RewardWeights:
    """Weights of the connectivity-blend reward."""

    beta: float = 1.0
    gamma: float = 0.0
    fbs_weight: float = 0.4
    fbs_exponent: float = 1.0
    epsilon: float = 1e-4


@dataclass
class RewardConfig:
    """Reward shaping options.

    mode (the objective Φ):
      - ``legacy_blend``: the historical train_ppo.py formula (epsilon-padded
        terms, power normalized against SINREvaluation's total power output).
      - ``controlled_blend``: bills only the power the agent actually
        controls (active FBS gene powers + gated macro-capacity power) and
        drops the epsilon padding, so the objective is a clean function of
        the decision variables. Accepts the old name ``ga_blend``.
      - ``sum_rate``: sum spectral efficiency over connected users.

    shaping (the training signal built from Φ):
      - ``none``: reward = Φ(s_t) each step (the historical dense signal —
        rewards *staying* in good states, so the do-nothing plateau pays).
      - ``potential``: reward = Φ(s_t) − Φ(s_{t−1}); the return telescopes to
        Φ(s_T) − Φ(s_0), a pure improvement signal.
      - ``record``: reward = max(0, Φ(s_t) − best Φ so far this episode); the
        return telescopes to max_t Φ(s_t) − Φ(s_0), so the agent is paid for
        *discovering* a good configuration and never punished for wandering
        afterwards. Targets the observed failure mode where good states are
        found and then lost.
      - ``potential_record``: the sum of the two, reward = (Φ(s_t) − Φ(s_{t−1}))
        + ``record_weight`` · max(0, Φ(s_t) − best Φ so far). The return
        telescopes to (Φ(s_T) − Φ(s_0)) + w·(max_t Φ(s_t) − Φ(s_0)), so the
        agent is paid both for *ending* well and for *having discovered* a good
        configuration. Pure ``potential`` charges back every step that leaves a
        good state, which under a static user map discourages exploring away
        from a local optimum; the record term refunds part of that.

    ``potential`` and ``record`` need one extra backend evaluation at reset
    (to establish Φ(s_0)). Φ itself is always logged per step as
    ``reward_objective`` so curves stay comparable across modes and runs.

    ``record_weight`` applies only to ``potential_record`` (ignored otherwise).

    reward_offset / reward_gain — training-signal conditioning only:
    the reward handed to the optimizer is ``(Φ − offset) · gain``, applied
    *before* shaping. Under ``potential``/``record`` the offset cancels and
    only the gain bites. Φ is a narrow band (~0.42–0.52 on the 2-1-1 world),
    so per-step signals under shaping are ~1e-3 — the same order as critic
    noise. Mapping do-nothing to 0 and the known optimum to 1 (offset=0.4480,
    gain≈13.6 there) is a monotone rescaling: it cannot move the argmax, it
    only makes the differences resolvable. The RAW Φ is what gets logged.
    """

    mode: str = "legacy_blend"
    weights: RewardWeights = field(default_factory=RewardWeights)
    power_includes_macro_capacity: bool = True
    shaping: str = "none"
    record_weight: float = 0.5
    reward_offset: float = 0.0
    reward_gain: float = 1.0

    def __post_init__(self):
        self.mode = REWARD_MODE_ALIASES.get(self.mode, self.mode)
        allowed = {"legacy_blend", "controlled_blend", "sum_rate"}
        if self.mode not in allowed:
            raise ValueError(f"reward mode {self.mode!r} not in {sorted(allowed)}")
        allowed_shaping = {"none", "potential", "record", "potential_record"}
        if self.shaping not in allowed_shaping:
            raise ValueError(
                f"reward shaping {self.shaping!r} not in {sorted(allowed_shaping)}"
            )
        if self.record_weight < 0:
            raise ValueError("record_weight must be non-negative")


# --------------------------------------------------------------------------- #
# World / band
# --------------------------------------------------------------------------- #
@dataclass
class WorldConfig:
    """Static world geometry + radio parameters (the 'scenario')."""

    width: float = 2000.0
    height: float = 1500.0
    num_mbs: int = 1
    mbs_locations: Optional[Sequence[Tuple[float, float]]] = None
    isd: float = 500.0
    margin: float = 100.0
    mbs_height: float = 25.0
    mbs_power: float = 20.0
    ue_height: float = 1.5
    scenario: str = "3GPP_38.901_UMa_LOS"
    map_mode: str = "quick"
    num_users: int = 1000
    max_users: Optional[int] = None
    sinr_threshold: float = 5.0

    def __post_init__(self):
        if self.num_mbs <= 0:
            raise ValueError("num_mbs must be positive")
        if self.mbs_locations is not None:
            locs = [tuple(map(float, p)) for p in self.mbs_locations]
            if len(locs) == 0:
                raise ValueError("mbs_locations must contain at least one pair")
            if len(locs) != self.num_mbs:
                raise ValueError(
                    f"num_mbs={self.num_mbs} does not match "
                    f"{len(locs)} mbs_locations entries"
                )
            self.mbs_locations = locs

    @property
    def effective_max_users(self) -> int:
        return int(self.max_users if self.max_users is not None else self.num_users)


@dataclass
class BandConfig:
    """Dual-band controls, mirroring the harness' controlsFbsBand /
    gaControlsMbsCapacity switches.

    mode:
      - ``legacy``: single-band world, byte-compatible with the original
        train_ppo.py MATLAB call (no bsBandIds/mbsSlotMap passed).
      - ``multi``: dual-band world (coverage + capacity antennas, MBS slot
        expansion, same-band interference accounting).

    fbs_band (multi mode only):
      - ``coverage`` / ``capacity``: all FBSs pinned to one band.
      - ``agent``: the policy controls a per-FBS band flag (6th action dim).

    mbs_capacity (multi mode only):
      - ``off`` / ``on``: capacity slots gated statically.
      - ``agent``: the policy controls one capacity toggle per MBS site.
    """

    mode: str = "legacy"
    fbs_band: str = "coverage"
    mbs_capacity: str = "off"

    def __post_init__(self):
        if self.mode not in {"legacy", "multi"}:
            raise ValueError(f"band mode {self.mode!r} must be 'legacy' or 'multi'")
        if self.fbs_band not in {"coverage", "capacity", "agent"}:
            raise ValueError(f"fbs_band {self.fbs_band!r} invalid")
        if self.mbs_capacity not in {"off", "on", "agent"}:
            raise ValueError(f"mbs_capacity {self.mbs_capacity!r} invalid")
        if self.mode == "legacy" and (
            self.fbs_band == "agent" or self.mbs_capacity == "agent"
        ):
            raise ValueError("agent band/capacity control requires band mode 'multi'")

    @property
    def is_multi(self) -> bool:
        return self.mode == "multi"

    @property
    def agent_controls_fbs_band(self) -> bool:
        return self.is_multi and self.fbs_band == "agent"

    @property
    def agent_controls_mbs_capacity(self) -> bool:
        return self.is_multi and self.mbs_capacity == "agent"

    @property
    def fixed_fbs_band_id(self) -> int:
        return BAND_CAPACITY if self.fbs_band == "capacity" else BAND_COVERAGE


# --------------------------------------------------------------------------- #
# Env-level config
# --------------------------------------------------------------------------- #
@dataclass
class EnvConfig:
    """Env-level config. The structural-training fields (``normalize_obs``,
    ``binary_mode``, ``obs_tier_metrics``, ``terminate_on_full_coverage``)
    default to the exact legacy behavior so historical runs stay loadable
    and bit-reproducible; new runs opt in per flag."""

    num_fbs: int = 1
    world: WorldConfig = field(default_factory=WorldConfig)
    band: BandConfig = field(default_factory=BandConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    max_episode_steps: int = 25
    action_scale: float = 0.05
    fbs_z_bounds: Tuple[float, float] = FBS_Z_BOUNDS
    fbs_power_bounds: Tuple[float, float] = FBS_POWER_BOUNDS
    collect_user_positions: bool = False
    # Structural training options (all legacy-off by default)
    normalize_obs: bool = False        # min-max scale the state part of obs to [-1, 1]
    binary_mode: str = "threshold"     # 'threshold': bit = (a >= 0) every step;
                                       # 'sticky': flip only when |a| >= 0.5, else hold
    obs_tier_metrics: bool = False     # append [fbs_frac, mbs_frac, total_frac, norm_power]
    terminate_on_full_coverage: bool = True  # legacy early-stop (truncates return)
    action_mode: str = "delta"         # 'delta'   : v += a * action_scale * range (legacy)
                                       # 'absolute': v  = lower + (a+1)/2 * range

    def __post_init__(self):
        if self.binary_mode not in {"threshold", "sticky"}:
            raise ValueError(
                f"binary_mode {self.binary_mode!r} must be 'threshold' or 'sticky'"
            )
        if self.action_mode not in {"delta", "absolute"}:
            raise ValueError(
                f"action_mode {self.action_mode!r} must be 'delta' or 'absolute'"
            )

    @property
    def genes_per_fbs(self) -> int:
        """State genes per FBS: [x, y, z, power, power_status(, band_flag)]."""
        return 6 if self.band.agent_controls_fbs_band else 5

    @property
    def num_mbs_genes(self) -> int:
        return self.world.num_mbs if self.band.agent_controls_mbs_capacity else 0

    @property
    def state_dim(self) -> int:
        return self.genes_per_fbs * self.num_fbs + self.num_mbs_genes

    @property
    def num_tier_obs(self) -> int:
        return 4 if self.obs_tier_metrics else 0

    @property
    def obs_dim(self) -> int:
        # + [fbs_delta, total_delta] (+ tier metrics when enabled)
        return self.state_dim + 2 + self.num_tier_obs

    @property
    def action_dim(self) -> int:
        return self.state_dim

    @property
    def needs_reset_eval(self) -> bool:
        """Whether reset() must evaluate the backend at the initial state
        (potential/record shaping need Φ(s_0); tier metrics need initial
        values)."""
        return self.obs_tier_metrics or self.reward.shaping in (
            "potential", "record", "potential_record"
        )


# --------------------------------------------------------------------------- #
# PPO hyperparameters
# --------------------------------------------------------------------------- #
@dataclass
class PPOParams:
    """Algorithm hyperparameters + run budget.

    ``algo`` selects the SB3 algorithm: ``"ppo"`` (on-policy, historical
    default) or ``"sac"`` (off-policy replay + auto-tuned entropy — far more
    sample-efficient at MATLAB step rates). PPO-only fields (n_steps,
    clip_range, gae_lambda, ...) are ignored under SAC and vice versa; the
    shared fields (learning_rate, gamma, batch_size, seed, ...) apply to both.
    The dataclass keeps its historical name for config compatibility.
    """

    total_timesteps: int = 10_000
    learning_rate: float = 3e-4
    ent_coef: float = 0.0
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: Optional[int] = 0
    n_envs: int = 1
    checkpoint_every: int = 0     # timesteps between checkpoints; 0 disables
    eval_every: int = 0           # timesteps between in-training evals; 0 disables
    eval_episodes: int = 3
    log_flush_every: int = 256    # step-log buffer size before flushing to disk
    verbose: int = 1
    # Running reward normalization (SB3 VecNormalize). Divides rewards by a
    # running std estimate of the discounted return, computed from the agent's
    # own experience — the same numerical conditioning as a hand-set
    # reward_gain, but derived from data rather than from a known optimum.
    # Prefer this over reward_gain for anything reported.
    normalize_reward: bool = False
    # Algorithm selection + SAC-only knobs (ignored under PPO)
    algo: str = "ppo"
    buffer_size: int = 50_000
    learning_starts: int = 500
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    sac_ent_coef: "float | str" = "auto"

    def __post_init__(self):
        if self.algo not in {"ppo", "sac"}:
            raise ValueError(f"algo {self.algo!r} must be 'ppo' or 'sac'")
        if self.n_envs < 1:
            raise ValueError("n_envs must be >= 1")
        if self.algo == "ppo" and self.batch_size > self.n_steps * self.n_envs:
            raise ValueError(
                f"batch_size ({self.batch_size}) cannot exceed "
                f"n_steps*n_envs ({self.n_steps * self.n_envs})"
            )


# --------------------------------------------------------------------------- #
# Scenario / cost-config presets (the Y and Z axes of the code)
# --------------------------------------------------------------------------- #
SCENARIOS: dict[int, dict] = {
    1: {  # single MBS, default 2000 x 1500 world (auto hex placement)
        "num_mbs": 1,
        "mbs_locations": None,
        "width": 2000.0,
        "height": 1500.0,
    },
    2: {  # two MBSs, 4000 x 3000 world
        "num_mbs": 2,
        "mbs_locations": [(1000.0, 1000.0), (3500.0, 2200.0)],
        "width": 4000.0,
        "height": 3000.0,
    },
}

CONFIGS: dict[int, dict] = {
    1: {"gamma": 0.0},
    2: {"gamma": 0.1},
}


# --------------------------------------------------------------------------- #
# ExperimentConfig — the user-facing bundle
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    num_fbs: int = 1          # X
    scenario_id: int = 1      # Y
    config_id: int = 1        # Z

    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOParams = field(default_factory=PPOParams)
    tag: Optional[str] = None

    @property
    def code(self) -> str:
        return f"{self.num_fbs}-{self.scenario_id}-{self.config_id}"

    @property
    def label(self) -> str:
        parts = [self.code]
        if self.env.band.is_multi:
            parts.append("mb")
        if self.ppo.algo != "ppo":
            parts.append(self.ppo.algo)
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_code(
        cls,
        code: str,
        *,
        band: Optional[BandConfig | dict | str] = None,
        reward: Optional[RewardConfig | dict] = None,
        world_overrides: Optional[dict] = None,
        env_overrides: Optional[dict] = None,
        ppo_overrides: Optional[dict] = None,
        tag: Optional[str] = None,
        # Frequently-tuned shortcuts (kept for parity with the old harness)
        total_timesteps: Optional[int] = None,
        max_episode_steps: Optional[int] = None,
        ent_coef: Optional[float] = None,
        action_scale: Optional[float] = None,
        learning_rate: Optional[float] = None,
        seed: Optional[int] = None,
        n_envs: Optional[int] = None,
    ) -> "ExperimentConfig":
        x, y, z = (int(p) for p in code.split("-"))
        if y not in SCENARIOS:
            raise ValueError(f"Unknown scenario id {y}; valid: {sorted(SCENARIOS)}")
        if z not in CONFIGS:
            raise ValueError(f"Unknown config id {z}; valid: {sorted(CONFIGS)}")

        world = WorldConfig(**{**SCENARIOS[y], **(world_overrides or {})})

        if band is None:
            band_cfg = BandConfig()
        elif isinstance(band, BandConfig):
            band_cfg = band
        elif isinstance(band, str):
            band_cfg = BandConfig(mode=band)
        else:
            band_cfg = BandConfig(**band)

        if reward is None:
            reward_cfg = RewardConfig(
                weights=RewardWeights(gamma=CONFIGS[z]["gamma"])
            )
        elif isinstance(reward, RewardConfig):
            reward_cfg = reward
        else:
            rw = reward.pop("weights", None)
            weights = RewardWeights(**rw) if isinstance(rw, dict) else (
                rw or RewardWeights(gamma=CONFIGS[z]["gamma"])
            )
            reward_cfg = RewardConfig(weights=weights, **reward)

        env_kwargs = dict(env_overrides or {})
        if max_episode_steps is not None:
            env_kwargs["max_episode_steps"] = max_episode_steps
        if action_scale is not None:
            env_kwargs["action_scale"] = action_scale
        env = EnvConfig(
            num_fbs=x, world=world, band=band_cfg, reward=reward_cfg, **env_kwargs
        )

        ppo_kwargs = dict(ppo_overrides or {})
        for key, val in (
            ("total_timesteps", total_timesteps),
            ("ent_coef", ent_coef),
            ("learning_rate", learning_rate),
            ("seed", seed),
            ("n_envs", n_envs),
        ):
            if val is not None:
                ppo_kwargs[key] = val
        ppo = PPOParams(**ppo_kwargs)

        return cls(num_fbs=x, scenario_id=y, config_id=z, env=env, ppo=ppo, tag=tag)

    def with_overrides(self, **kwargs) -> "ExperimentConfig":
        return replace(self, **kwargs)

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_json_dict(self) -> dict:
        return {
            "config_version": CONFIG_VERSION,
            "code": self.code,
            "num_fbs": self.num_fbs,
            "scenario_id": self.scenario_id,
            "config_id": self.config_id,
            "tag": self.tag,
            "env": asdict(self.env),
            "ppo": asdict(self.ppo),
        }

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_json_dict(), indent=2, default=str))


def _dataclass_from_dict(cls, data: dict):
    """Build a dataclass instance, ignoring unknown keys (forward compat)."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


def _env_from_dict(data: dict) -> EnvConfig:
    world = _dataclass_from_dict(WorldConfig, data.get("world", {}))
    band = _dataclass_from_dict(BandConfig, data.get("band", {}))
    reward_raw = dict(data.get("reward", {}))
    weights = _dataclass_from_dict(RewardWeights, reward_raw.pop("weights", {}) or {})
    reward = _dataclass_from_dict(RewardConfig, {**reward_raw, "weights": weights})
    reward.weights = weights
    env_kwargs = {
        k: v
        for k, v in data.items()
        if k in {f.name for f in fields(EnvConfig)}
        and k not in {"world", "band", "reward"}
    }
    if "fbs_z_bounds" in env_kwargs:
        env_kwargs["fbs_z_bounds"] = tuple(env_kwargs["fbs_z_bounds"])
    if "fbs_power_bounds" in env_kwargs:
        env_kwargs["fbs_power_bounds"] = tuple(env_kwargs["fbs_power_bounds"])
    return EnvConfig(world=world, band=band, reward=reward, **env_kwargs)


def experiment_from_json_dict(data: dict) -> ExperimentConfig:
    """Load an ExperimentConfig from a serialized dict (v2 or legacy v1)."""
    version = data.get("config_version")
    if version == CONFIG_VERSION:
        return ExperimentConfig(
            num_fbs=data["num_fbs"],
            scenario_id=data["scenario_id"],
            config_id=data["config_id"],
            tag=data.get("tag"),
            env=_env_from_dict(data.get("env", {})),
            ppo=_dataclass_from_dict(PPOParams, data.get("ppo", {})),
        )

    # v1: the flat schema written by the old ppo_experiment.py
    return ExperimentConfig.from_code(
        f"{data['num_fbs']}-{data['scenario_id']}-{data['config_id']}",
        total_timesteps=data.get("total_timesteps"),
        max_episode_steps=data.get("max_episode_steps"),
        ent_coef=data.get("ent_coef"),
        action_scale=data.get("action_scale"),
        learning_rate=data.get("learning_rate"),
        reward=RewardConfig(
            weights=RewardWeights(
                beta=data.get("beta", 0.8),
                gamma=data.get("gamma", CONFIGS[data["config_id"]]["gamma"]),
                fbs_weight=data.get("fbs_weight", 0.4),
                fbs_exponent=data.get("fbs_exponent", 1.0),
            )
        ),
    )


def load_experiment_config(path: Path | str) -> ExperimentConfig:
    with Path(path).open() as f:
        return experiment_from_json_dict(json.load(f))
