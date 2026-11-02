"""DDPG configuration.

An experiment has two halves. The environment half (world, band, reward and
env settings) is exactly ``ppo.config.EnvConfig``, built through the same
``ExperimentConfig.from_code`` presets, so a DDPG run, a TD3 run and a PPO
run given the same code and flags see an identical environment. The algorithm
half is ``DDPGParams`` -- nothing from ``PPOParams`` or ``TD3Params`` is used.

DDPG is TD3 without its three tricks: one critic instead of twin critics, no
delayed policy updates, and no target policy smoothing. Stable-Baselines3
implements it as a ``TD3`` subclass that pins those off, so ``DDPGParams``
simply has no ``policy_delay`` / ``target_policy_noise`` /
``target_noise_clip`` fields to set.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List, Optional, Tuple

from ppo.config import EnvConfig, ExperimentConfig, experiment_from_json_dict

DDPG_CONFIG_VERSION = 1

NOISE_TYPES = ("normal", "ou")


def binary_action_indices(env: EnvConfig) -> List[int]:
    """Action dimensions that drive a *binary* gene, for the layout in
    ``ppo.env.FlyingBaseStationEnv``.

    Each FBS occupies a block of genes: 0-3 are continuous (x, y, z, power),
    4 is the power on/off switch, and 5 is the band flag when the agent
    controls it. Per-MBS capacity toggles, when enabled, trail the FBS
    blocks. Those binary dims are what ``_binary_update`` thresholds, so the
    action-to-reward map is a step function along them and a *continuous*
    dim's exploration noise is the wrong size for them.
    """
    genes = 5 + (1 if env.band.agent_controls_fbs_band else 0)
    idx: List[int] = []
    for f in range(env.num_fbs):
        idx.append(f * genes + 4)
        if env.band.agent_controls_fbs_band:
            idx.append(f * genes + 5)
    n_genes = env.num_fbs * genes
    if env.band.agent_controls_mbs_capacity:
        idx.extend(range(n_genes, n_genes + env.world.num_mbs))
    return idx


@dataclass
class DDPGParams:
    """DDPG hyperparameters + run budget.

    Defaults are Stable-Baselines3's DDPG defaults, plus exploration noise.
    ``action_noise_std`` is in action units: actions live in [-1, 1], so under
    ``action_mode='delta'`` a std of 0.1 jitters a position gene by about
    0.1 * action_scale * range per step.

    ``binary_noise_std`` overrides that std on the binary dims (see
    ``binary_action_indices``), which need a different size entirely. A
    deterministic actor saturates its tanh, and from an output of -1.0 a
    *sticky* gene only flips when the noise pushes the action to >= +0.5,
    i.e. when the draw exceeds 1.5: at std 0.1 that never happens, at 0.3 it
    is ~3e-7 per step, at 1.0 it is ~6.7%. Left as None, every dim shares
    ``action_noise_std`` (plain SB3 behaviour).

    ``noise_type`` picks Gaussian ('normal') or Ornstein-Uhlenbeck ('ou')
    exploration noise. OU is what the DDPG paper used: temporally correlated,
    so it pushes a gene in one direction for a stretch of steps instead of
    jittering it, which matters when reaching a good position takes many
    small ``delta`` steps. SB3 resets the OU state at every episode boundary.

    ``ou_theta``/``ou_dt`` set the mean reversion. Their product is what
    matters: the noise decorrelates over roughly ``1 / (ou_theta * ou_dt)``
    steps and the per-step innovation is ``sigma * sqrt(ou_dt)``. SB3's
    defaults (0.15, 1e-2) give a ~670-step correlation length, so on a
    40-step episode OU is a slow random walk that never reaches its
    stationary spread; ``ou_dt=1.0`` gives a ~7-step correlation length,
    which is the regime the DDPG paper's noise actually operates in.

    ``gamma`` is the RL discount factor (not the cost-function gamma in
    ``RewardWeights``).
    """

    total_timesteps: int = 10_000
    learning_rate: float = 1e-3
    buffer_size: int = 1_000_000
    learning_starts: int = 100
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    action_noise_std: float = 0.1
    binary_noise_std: Optional[float] = None
    noise_type: str = "normal"
    ou_theta: float = 0.15
    ou_dt: float = 1e-2
    net_arch: Optional[Tuple[int, ...]] = None   # None -> SB3 default (400, 300)
    # Running reward normalization (SB3 VecNormalize), as in TD3Params. The
    # replay buffer keeps raw rewards; SB3 normalizes them when sampling.
    normalize_reward: bool = False
    seed: Optional[int] = 0
    n_envs: int = 1
    checkpoint_every: int = 0     # timesteps between checkpoints; 0 disables
    eval_every: int = 0           # timesteps between in-training evals; 0 disables
    eval_episodes: int = 3
    log_flush_every: int = 256
    verbose: int = 1

    def __post_init__(self):
        if self.n_envs < 1:
            raise ValueError("n_envs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.action_noise_std < 0:
            raise ValueError("action_noise_std must be >= 0")
        if self.binary_noise_std is not None and self.binary_noise_std < 0:
            raise ValueError("binary_noise_std must be >= 0")
        if self.noise_type not in NOISE_TYPES:
            raise ValueError(f"noise_type {self.noise_type!r} not in {list(NOISE_TYPES)}")
        if self.ou_theta <= 0 or self.ou_dt <= 0:
            raise ValueError("ou_theta and ou_dt must be > 0")
        if self.net_arch is not None:
            self.net_arch = tuple(int(n) for n in self.net_arch)
            if not self.net_arch or min(self.net_arch) < 1:
                raise ValueError("net_arch must list positive layer sizes")


@dataclass
class DDPGExperiment:
    """The user-facing bundle: X-Y-Z code + env config + DDPG parameters."""

    num_fbs: int = 1          # X
    scenario_id: int = 1      # Y
    config_id: int = 1        # Z
    env: EnvConfig = field(default_factory=EnvConfig)
    ddpg: DDPGParams = field(default_factory=DDPGParams)
    tag: Optional[str] = None

    @property
    def code(self) -> str:
        return f"{self.num_fbs}-{self.scenario_id}-{self.config_id}"

    @property
    def label(self) -> str:
        parts = [self.code]
        if self.env.band.is_multi:
            parts.append("mb")
        parts.append("ddpg")
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)

    @classmethod
    def from_code(
        cls,
        code: str,
        *,
        band=None,
        reward=None,
        world_overrides: Optional[dict] = None,
        env_overrides: Optional[dict] = None,
        ddpg_overrides: Optional[dict] = None,
        tag: Optional[str] = None,
        total_timesteps: Optional[int] = None,
        max_episode_steps: Optional[int] = None,
        action_scale: Optional[float] = None,
        learning_rate: Optional[float] = None,
        seed: Optional[int] = None,
        n_envs: Optional[int] = None,
    ) -> "DDPGExperiment":
        # Build through ppo's presets and keep only the env half, so scenario
        # and cost-config ids resolve exactly as they do for PPO and TD3 runs.
        base = ExperimentConfig.from_code(
            code,
            band=band,
            reward=reward,
            world_overrides=world_overrides,
            env_overrides=env_overrides,
            max_episode_steps=max_episode_steps,
            action_scale=action_scale,
        )
        ddpg_kwargs = dict(ddpg_overrides or {})
        for key, val in (
            ("total_timesteps", total_timesteps),
            ("learning_rate", learning_rate),
            ("seed", seed),
            ("n_envs", n_envs),
        ):
            if val is not None:
                ddpg_kwargs[key] = val
        return cls(
            num_fbs=base.num_fbs,
            scenario_id=base.scenario_id,
            config_id=base.config_id,
            env=base.env,
            ddpg=DDPGParams(**ddpg_kwargs),
            tag=tag,
        )

    def to_json_dict(self) -> dict:
        return {
            "ddpg_config_version": DDPG_CONFIG_VERSION,
            "algo": "ddpg",
            "code": self.code,
            "num_fbs": self.num_fbs,
            "scenario_id": self.scenario_id,
            "config_id": self.config_id,
            "tag": self.tag,
            "env": asdict(self.env),
            "ddpg": asdict(self.ddpg),
        }

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_json_dict(), indent=2, default=str))


def ddpg_experiment_from_json_dict(data: dict) -> DDPGExperiment:
    """Load a DDPGExperiment from a serialized dict (``to_json_dict`` output,
    or a run's ``experiment_config.json``)."""
    if data.get("algo") != "ddpg":
        raise ValueError(f"not a DDPG experiment config (algo={data.get('algo')!r})")
    # The env block is parsed by ppo's own loader, so all packages read env
    # settings (defaults for missing fields included) identically.
    shell = experiment_from_json_dict({
        "config_version": 2,
        "num_fbs": data["num_fbs"],
        "scenario_id": data["scenario_id"],
        "config_id": data["config_id"],
        "env": data.get("env", {}),
    })
    known = {f.name for f in fields(DDPGParams)}
    ddpg = DDPGParams(**{k: v for k, v in (data.get("ddpg") or {}).items() if k in known})
    return DDPGExperiment(
        num_fbs=data["num_fbs"],
        scenario_id=data["scenario_id"],
        config_id=data["config_id"],
        env=shell.env,
        ddpg=ddpg,
        tag=data.get("tag"),
    )


def load_ddpg_experiment(path: Path | str) -> DDPGExperiment:
    return ddpg_experiment_from_json_dict(json.loads(Path(path).read_text()))
