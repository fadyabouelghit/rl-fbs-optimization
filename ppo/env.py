"""Gymnasium environment for FBS placement driven by a SINR backend.

Layout of the internal state vector (one block per FBS):

    per FBS : [x, y, z, power, power_status(, band_flag)]   band flag only
                                                            when the agent
                                                            controls bands
    trailing: one capacity gene per MBS site                only when the
                                                            agent controls
                                                            MBS capacity

The observation appends two global deltas ``[fbs_delta, total_delta]``
(step-over-step change in connected-user fractions), matching the original
train_ppo.py observation so legacy agents remain loadable.

Actions live in [-1, 1]^state_dim: the four continuous genes move by
``action * action_scale * gene_range`` and are clipped to bounds; binary
genes (power status, band flag, capacity genes) are set to ``action >= 0``.

Structural training options (all opt-in via EnvConfig, legacy-off defaults):

- ``normalize_obs``       state part of the observation min-max scaled to
                          [-1, 1]; the internal state (and everything logged
                          from it) stays in physical units.
- ``binary_mode="sticky"``  binary genes flip only when |action| >= 0.5,
                          else hold — stops the ~50%/step flicker an
                          untrained Gaussian policy produces at threshold 0.
- ``obs_tier_metrics``    append [fbs_frac, mbs_frac, total_frac, norm_power]
                          so the policy sees connectivity feedback.
- ``reward.shaping="potential"``  reward = Φ(s_t) − Φ(s_{t−1}) instead of
                          the dense Φ(s_t); Φ is always logged as
                          ``reward_objective``. Needs one backend evaluation
                          at reset (also fixes the spurious first-step delta).
- ``reward.shaping="record"``  reward = max(0, Φ(s_t) − best Φ this episode):
                          paid for *discovering* a better configuration, never
                          punished for leaving it.
- ``reward.shaping="potential_record"``  the sum of the two: the dense
                          step-over-step delta plus ``record_weight`` times the
                          new-best bonus. Keeps potential's incentive to end in
                          a good state while refunding part of the charge for
                          exploring away from one.
- ``reward_offset/gain``  monotone rescaling of the training signal only
                          (raw Φ still logged); fixes the ~1e-3 signal scale.
- ``action_mode="absolute"``  actions name gene values directly instead of
                          nudging them. With a static user map the answer is a
                          fixed point, and delta control at action_scale=0.5
                          needs |a| <= 0.02 to place within 20 m — absolute
                          control removes that precision ceiling.
- ``terminate_on_full_coverage=False``  drop the legacy early-stop whose
                          truncated return penalizes reaching full coverage.
"""
from __future__ import annotations

import warnings
from typing import Optional

import gymnasium as gym
import numpy as np

from .config import EnvConfig
from .matlab_bridge import SinrBackend, SinrResult, make_backend


def state_labels(config: EnvConfig) -> list[str]:
    """Column labels of the flat state vector for a given env config."""
    names = ["x", "y", "height", "power", "power_status"]
    if config.band.agent_controls_fbs_band:
        names.append("band_flag")
    labels = [f"fbs{i}_{n}" for i in range(config.num_fbs) for n in names]
    if config.band.agent_controls_mbs_capacity:
        labels += [f"mbs{j}_capacity" for j in range(config.world.num_mbs)]
    return labels


class FlyingBaseStationEnv(gym.Env):
    """N-FBS placement environment over a pluggable SINR backend."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvConfig,
        backend: Optional[SinrBackend] = None,
        backend_kind: str = "matlab",
        session=None,
    ):
        super().__init__()
        self.config = config
        self.num_fbs = config.num_fbs
        self.world = config.world
        self.band = config.band
        self.reward_config = config.reward

        if backend is None:
            backend = make_backend(self.world, self.band, backend_kind, session=session)
            self._owns_backend = True
        else:
            self._owns_backend = False
        self.backend = backend

        # ------------------------------------------------------------ #
        # Bounds / spaces
        # ------------------------------------------------------------ #
        z_lo, z_hi = config.fbs_z_bounds
        p_lo, p_hi = config.fbs_power_bounds
        lower = [0.0, 0.0, z_lo, p_lo, 0.0]
        upper = [self.world.width, self.world.height, z_hi, p_hi, 1.0]
        if self.band.agent_controls_fbs_band:
            lower.append(0.0)
            upper.append(1.0)
        self.param_lower_single = np.asarray(lower, dtype=np.float32)
        self.param_upper_single = np.asarray(upper, dtype=np.float32)
        self.genes_per_fbs = len(lower)

        state_low = np.tile(self.param_lower_single, self.num_fbs)
        state_high = np.tile(self.param_upper_single, self.num_fbs)
        if self.band.agent_controls_mbs_capacity:
            state_low = np.concatenate([state_low, np.zeros(self.world.num_mbs, np.float32)])
            state_high = np.concatenate([state_high, np.ones(self.world.num_mbs, np.float32)])
        self.state_low = state_low.astype(np.float32)
        self.state_high = state_high.astype(np.float32)
        self.state_dim = self.state_low.size
        self._state_range = np.maximum(self.state_high - self.state_low, 1e-8)

        if config.normalize_obs:
            obs_state_low = -np.ones_like(self.state_low)
            obs_state_high = np.ones_like(self.state_high)
        else:
            obs_state_low, obs_state_high = self.state_low, self.state_high
        extra_low, extra_high = [-1.0, -1.0], [1.0, 1.0]   # [fbs_delta, total_delta]
        if config.obs_tier_metrics:
            extra_low += [0.0] * 4                          # fractions stay in [0, 1]
            extra_high += [1.0] * 4
        obs_low = np.concatenate([obs_state_low, np.array(extra_low, np.float32)])
        obs_high = np.concatenate([obs_state_high, np.array(extra_high, np.float32)])
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )

        single_scale = self.param_upper_single[:4] - self.param_lower_single[:4]
        self._action_scale = (
            config.action_scale * np.tile(single_scale, (self.num_fbs, 1))
        ).astype(np.float32)

        # Reward normalization constants
        self.min_power = 0.0
        self.max_power = float(p_hi) * self.num_fbs
        self._max_users_norm = float(max(self.world.effective_max_users, 1))
        self._macro_power_budget = (
            self.world.mbs_power * self.world.num_mbs
            if (self.reward_config.power_includes_macro_capacity and self.world.num_mbs > 0)
            else 0.0
        )

        if config.action_mode == "absolute" and config.binary_mode == "sticky":
            # Sticky holds a binary gene unless the action is decisive, so a
            # step that names an excellent position can still be scored with
            # the FBS stuck off — breaking the link between action and reward
            # that sticky exists to protect under delta control.
            warnings.warn(
                "action_mode='absolute' with binary_mode='sticky': continuous "
                "genes are named directly while binary genes only hold/flip, "
                "so rewards may not reflect the action just taken. "
                "Use binary_mode='threshold' with absolute actions.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._state: Optional[np.ndarray] = None
        self._global_state = np.zeros(2, dtype=np.float32)
        self._tier_obs = np.zeros(4, dtype=np.float32)
        self._last_total_connected_frac = 0.0
        self._last_fbs_connected_frac = 0.0
        self._prev_potential = 0.0
        self._best_potential = 0.0
        self._steps = 0

    # ------------------------------------------------------------------ #
    # Introspection helpers (used by logging / evaluation / plotting)
    # ------------------------------------------------------------------ #
    def state_labels(self) -> list[str]:
        return state_labels(self.config)

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    def _fbs_blocks(self, state: np.ndarray) -> np.ndarray:
        n_genes = self.genes_per_fbs * self.num_fbs
        return state[:n_genes].reshape(self.num_fbs, self.genes_per_fbs)

    def _mbs_genes(self, state: np.ndarray) -> Optional[np.ndarray]:
        if not self.band.agent_controls_mbs_capacity:
            return None
        return state[self.genes_per_fbs * self.num_fbs:]

    def _effective_band_flags(self, blocks: np.ndarray) -> Optional[np.ndarray]:
        if not self.band.is_multi:
            return None
        if self.band.agent_controls_fbs_band:
            return blocks[:, 5]
        return np.full(self.num_fbs, float(self.band.fixed_fbs_band_id))

    def _effective_mbs_genes(self, state: np.ndarray) -> Optional[np.ndarray]:
        if not self.band.is_multi:
            return None
        if self.band.agent_controls_mbs_capacity:
            return self._mbs_genes(state)
        value = 1.0 if self.band.mbs_capacity == "on" else 0.0
        return np.full(self.world.num_mbs, value)

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (2.0 * (state - self.state_low) / self._state_range - 1.0).astype(np.float32)

    def _obs_parts(self, state: np.ndarray) -> list[np.ndarray]:
        parts = [state, self._global_state]
        if self.config.obs_tier_metrics:
            parts.append(self._tier_obs)
        return parts

    def _get_obs(self) -> np.ndarray:
        state = np.asarray(self._state, np.float32).reshape(-1)
        if self.config.normalize_obs:
            state = self._normalize_state(state)
        return np.concatenate(self._obs_parts(state)).astype(np.float32)

    def flat_state_with_globals(self) -> np.ndarray:
        """Physical-units state + global deltas (+ tier metrics) — the
        trajectory-logging view, independent of observation normalization."""
        state = np.asarray(self._state, np.float32).reshape(-1)
        return np.concatenate(self._obs_parts(state)).astype(np.float32)

    def _update_connectivity_trackers(self, result: SinrResult, terms: dict) -> None:
        """Refresh the frac trackers + tier-metric obs block from a backend result."""
        self._last_total_connected_frac = float(result.total_connected) / self._max_users_norm
        self._last_fbs_connected_frac = float(result.fbs_connected) / self._max_users_norm
        if self.config.obs_tier_metrics:
            norm_power = float(
                terms.get(
                    "norm_power",
                    np.clip(result.total_power / max(self.max_power, 1e-8), 0.0, 1.0),
                )
            )
            self._tier_obs = np.clip(
                np.array(
                    [
                        result.fbs_connected / self._max_users_norm,
                        result.mbs_connected / self._max_users_norm,
                        result.total_connected / self._max_users_norm,
                        norm_power,
                    ],
                    dtype=np.float32,
                ),
                0.0,
                1.0,
            )

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._steps = 0
        options = options or {}

        if "state" in options and options["state"] is not None:
            state = np.asarray(options["state"], dtype=np.float32).reshape(-1)
            if state.size != self.state_dim:
                raise ValueError(
                    f"options['state'] has {state.size} entries; expected "
                    f"{self.state_dim} ({self.genes_per_fbs} per FBS x {self.num_fbs}"
                    + (f" + {self.world.num_mbs} MBS genes" if self._mbs_genes(self.state_low) is not None else "")
                    + ")"
                )
            self._state = state.copy()
        else:
            rand_cont = self.np_random.random((self.num_fbs, 4))
            cont_range = self.param_upper_single[:4] - self.param_lower_single[:4]
            cont = self.param_lower_single[:4] + rand_cont * cont_range
            binaries = self.np_random.integers(
                0, 2, size=(self.num_fbs, self.genes_per_fbs - 4)
            )
            state = np.concatenate([cont, binaries], axis=1).astype(np.float32)
            state = state.reshape(-1)
            if self.band.agent_controls_mbs_capacity:
                genes = self.np_random.integers(0, 2, size=self.world.num_mbs)
                state = np.concatenate([state, genes.astype(np.float32)])
            self._state = state

        self._global_state = np.zeros(2, dtype=np.float32)
        self._tier_obs = np.zeros(4, dtype=np.float32)
        self._last_total_connected_frac = 0.0
        self._last_fbs_connected_frac = 0.0
        self._prev_potential = 0.0
        self._best_potential = 0.0

        info: dict = {}
        if self.config.needs_reset_eval:
            # One extra evaluation per episode: initializes Φ(s_0) for
            # potential shaping, the tier-metric obs block, and the frac
            # trackers (so the first step's deltas are real deltas, not
            # "current minus zero").
            blocks = self._fbs_blocks(self._state)
            band_flags = self._effective_band_flags(blocks)
            mbs_genes = self._effective_mbs_genes(self._state)
            result = self.backend.evaluate(
                blocks[:, :4],
                blocks[:, 4],
                fbs_band_flags=band_flags,
                mbs_capacity_genes=mbs_genes,
            )
            potential, terms = self._compute_objective(result, blocks, mbs_genes)
            scaled = self._scaled(potential)
            self._prev_potential = scaled
            self._best_potential = scaled
            self._update_connectivity_trackers(result, terms)
            info = {**result.as_metrics_dict(), "objective": potential}
        return self._get_obs(), info

    def step(self, action: np.ndarray):
        self._steps += 1
        action = np.asarray(np.clip(action, -1.0, 1.0), dtype=np.float32).reshape(-1)
        if action.size != self.state_dim:
            raise ValueError(f"action has {action.size} entries; expected {self.state_dim}")

        n_genes = self.genes_per_fbs * self.num_fbs
        fbs_actions = action[:n_genes].reshape(self.num_fbs, self.genes_per_fbs)
        state = self._state.copy()
        blocks = self._fbs_blocks(state)

        lo, hi = self.param_lower_single[:4], self.param_upper_single[:4]
        if self.config.action_mode == "absolute":
            # Name the gene values directly; in-bounds by construction.
            blocks[:, :4] = lo + 0.5 * (fbs_actions[:, :4] + 1.0) * (hi - lo)
        else:
            blocks[:, :4] = np.clip(
                blocks[:, :4] + fbs_actions[:, :4] * self._action_scale, lo, hi
            )
        blocks[:, 4:] = self._binary_update(blocks[:, 4:], fbs_actions[:, 4:])
        state[:n_genes] = blocks.reshape(-1)
        if self.band.agent_controls_mbs_capacity:
            state[n_genes:] = self._binary_update(state[n_genes:], action[n_genes:])
        self._state = state.astype(np.float32)

        blocks = self._fbs_blocks(self._state)
        band_flags = self._effective_band_flags(blocks)
        mbs_genes = self._effective_mbs_genes(self._state)

        result = self.backend.evaluate(
            blocks[:, :4],
            blocks[:, 4],
            fbs_band_flags=band_flags,
            mbs_capacity_genes=mbs_genes,
            collect_user_positions=self.config.collect_user_positions,
        )

        potential, terms = self._compute_objective(result, blocks, mbs_genes)
        scaled = self._scaled(potential)
        shaping = self.reward_config.shaping
        if shaping == "potential":
            reward = scaled - self._prev_potential
            self._prev_potential = scaled
        elif shaping == "record":
            reward = max(0.0, scaled - self._best_potential)
            self._best_potential = max(self._best_potential, scaled)
        elif shaping == "potential_record":
            # Dense improvement signal + a bonus for each new episode best.
            # Return telescopes to (Φ_T − Φ_0) + w·(max_t Φ_t − Φ_0).
            reward = (scaled - self._prev_potential) + (
                self.reward_config.record_weight
                * max(0.0, scaled - self._best_potential)
            )
            self._prev_potential = scaled
            self._best_potential = max(self._best_potential, scaled)
        else:
            reward = scaled
        # NOTE: the RAW objective is logged, never the rescaled training
        # signal — every cross-run and baseline comparison depends on it.
        reward_terms = {**terms, "objective": float(potential)}

        total_frac = float(result.total_connected) / self._max_users_norm
        fbs_frac = float(result.fbs_connected) / self._max_users_norm
        total_delta = total_frac - self._last_total_connected_frac
        fbs_delta = fbs_frac - self._last_fbs_connected_frac
        self._global_state = np.array([fbs_delta, total_delta], dtype=np.float32)
        self._update_connectivity_trackers(result, terms)

        terminated = (
            self.config.terminate_on_full_coverage
            and result.total_connected >= self.world.num_users
        )
        truncated = self._steps >= self.config.max_episode_steps

        info = {
            **result.as_metrics_dict(),
            "num_active_fbs": int(np.sum(blocks[:, 4] >= 0.5)),
            "fbs_delta": fbs_delta,
            "total_delta": total_delta,
            "state": self._state.copy(),
            **{f"reward_{k}": v for k, v in reward_terms.items()},
        }
        if band_flags is not None:
            info["num_capacity_fbs"] = int(np.sum(band_flags >= 0.5))
        if mbs_genes is not None:
            info["num_active_capacity_carriers"] = int(np.sum(mbs_genes >= 0.5))
        if result.user_positions is not None:
            info["user_positions"] = result.user_positions

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def _binary_update(self, current: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Map raw actions onto binary genes per ``binary_mode``."""
        proposed = (actions >= 0.0).astype(np.float32)
        if self.config.binary_mode == "sticky":
            proposed = np.where(np.abs(actions) >= 0.5, proposed, current)
        return proposed.astype(np.float32)

    def _scaled(self, potential: float) -> float:
        """Condition the training signal: (Φ − offset) · gain.

        Monotone, so the optimum is unchanged; only the numeric range the
        optimizer sees changes. Applied *before* shaping, so under
        potential/record the offset cancels and only the gain survives.
        """
        return (
            potential - self.reward_config.reward_offset
        ) * self.reward_config.reward_gain

    # ------------------------------------------------------------------ #
    # Objective Φ (the reward under shaping='none'; the potential otherwise)
    # ------------------------------------------------------------------ #
    def _compute_objective(
        self,
        result: SinrResult,
        blocks: np.ndarray,
        mbs_genes: Optional[np.ndarray],
    ) -> tuple[float, dict]:
        w = self.reward_config.weights
        mode = self.reward_config.mode

        norm_users = float(np.clip(result.total_connected / self._max_users_norm, 0.0, 1.0))
        fbs_share = float(
            np.clip(result.fbs_connected / max(result.total_connected, 1), 0.0, 1.0)
        )

        if mode == "sum_rate":
            terms = {"sum_rate": result.sum_rate}
            return result.sum_rate, terms

        if mode == "legacy_blend":
            power_range = max(self.max_power - self.min_power, w.epsilon)
            norm_power = float(
                np.clip((result.total_power - self.min_power) / power_range, 0.0, 1.0)
            )
            base = (norm_users + w.epsilon) ** w.beta
            base *= ((1.0 - norm_power) + w.epsilon) ** w.gamma
            fbs_term = (fbs_share + w.epsilon) ** w.fbs_exponent
        else:  # controlled_blend — bill only agent-controlled power
            active = blocks[:, 4] >= 0.5
            controlled_power = float(np.sum(blocks[active, 3]))
            if (
                self.reward_config.power_includes_macro_capacity
                and mbs_genes is not None
            ):
                controlled_power += float(
                    np.sum(mbs_genes >= 0.5) * self.world.mbs_power
                )
            max_power_total = self.max_power + self._macro_power_budget
            if max_power_total <= 0:
                max_power_total = 1.0
            norm_power = float(np.clip(controlled_power / max_power_total, 0.0, 1.0))
            base = (norm_users ** w.beta) * ((1.0 - norm_power) ** w.gamma)
            fbs_term = fbs_share ** w.fbs_exponent

        reward = (1.0 - w.fbs_weight) * base + w.fbs_weight * fbs_term
        terms = {
            "base": float(base),
            "fbs_term": float(fbs_term),
            "norm_users": norm_users,
            "norm_power": norm_power,
            "fbs_share": fbs_share,
        }
        return float(reward), terms

    # ------------------------------------------------------------------ #
    def render(self):
        return None

    def close(self):
        if self._owns_backend and self.backend is not None:
            self.backend.close()
            self.backend = None
