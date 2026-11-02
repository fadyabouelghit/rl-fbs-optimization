"""Stage 0 instrumentation: what the actor and critic say about the on/off dim.

Probe set
---------
256 states drawn once from the env's own reset distribution
(``assess_models.start_states`` with ``PROBE_SEED``), each turned into an
observation by resetting the env to it and taking one zero action (a zero
delta leaves the continuous genes in place, and |0| < 0.5 leaves every sticky
binary gene in place), so the tier metrics in the observation are real.
Saved once to ``letter_runs/probe_sets/`` and only *loaded* inside training,
so the probe never builds an env, never touches an RNG and cannot perturb a
seeded run.

    python -m letterexp.probe build --backend pyqd-fast        # once
    python -m letterexp.probe show  --run <run dir>            # sample of probe_log.csv

ProbeCallback
-------------
Every ``every`` steps (and once before the first update), for every binary
action dim b:

    mu_b          actor's on/off output (tanh), and its pre-tanh value
    tanhf         1 - mu_b^2
    dq_mu_q{i}    dQ_i/da_b at a = mu(s)                    (TD3: i = 1, 2)
    dq_p05_q{i}   dQ_i/da_b at a_b = +0.5, other dims at mu(s)
    dq_m05_q{i}   dQ_i/da_b at a_b = -0.5, other dims at mu(s)
    q_on_q{i}     Q_i at a_b = +1, other dims at mu(s)
    q_off_q{i}    Q_i at a_b = -1, other dims at mu(s)

For the hybrid (Bernoulli) actor, ``mu_b`` is the *expected* action
2*sigmoid(logit) - 1 and ``pre`` is the logit, so the same pin definition
applies; the critic slope is taken at the deterministic (argmax) action.

probe_log.csv holds per-checkpoint summaries across the 256 states;
probe_raw.npz holds the per-state arrays for Stage 3.

Pinned at a checkpoint: >= 95% of probe states have |mu_b| > 0.96.
Trapped at a checkpoint: >= 95% of the probe states whose gene is currently OFF
have mu_b < -0.96 (the actor cannot switch the gene on from there). A
"keep the current value" saturated actor is pinned in both directions at once
and trapped; an all-states pin at -1 is both pinned 'off' and trapped.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .paths import PROBE_SET_DIR

PROBE_SEED = 20_261_009
N_PROBE = 256
SAT = 0.96          # |mu_b| above this counts as saturated
PIN_FRAC = 0.95     # share of probe states that must be saturated


# --------------------------------------------------------------------------- #
# probe set
# --------------------------------------------------------------------------- #
def probe_set_path(code: str = "1-1-1", backend: str = "pyqd-fast",
                   n: int = N_PROBE, seed: int = PROBE_SEED, tag: str = "") -> Path:
    t = f"_{tag}" if tag else ""
    return PROBE_SET_DIR / f"probe_{code}_{backend}_n{n}_s{seed}{t}.npz"


def build_probe_set(env_cfg, backend: str, n: int = N_PROBE, seed: int = PROBE_SEED,
                    out: Optional[Path] = None) -> Path:
    import sys

    from ppo.env import FlyingBaseStationEnv
    from ppo.paths import REPO_ROOT

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import assess_models

    states = assess_models.start_states(env_cfg, n, seed=seed)
    env = FlyingBaseStationEnv(env_cfg, backend_kind=backend)
    obs_list, st_list = [], []
    try:
        zero = np.zeros(env.action_space.shape, dtype=np.float32)
        for k, s in enumerate(states):
            env.reset(seed=seed + k, options={"state": s})
            obs, *_ = env.step(zero)
            obs_list.append(np.asarray(obs, np.float32))
            st_list.append(env.state.astype(np.float32))
    finally:
        env.close()
    out = Path(out) if out else probe_set_path(backend=backend, n=n, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, obs=np.stack(obs_list), states=np.stack(st_list),
                        seed=seed, backend=backend)
    return out


# --------------------------------------------------------------------------- #
# callback
# --------------------------------------------------------------------------- #
def _make_callback_class():
    import torch as th
    from stable_baselines3.common.callbacks import BaseCallback

    class ProbeCallback(BaseCallback):
        def __init__(self, probe_path: Path, binary_idx: Sequence[int],
                     labels: Sequence[str], out_dir: Optional[Path] = None, every: int = 1000,
                     verbose: int = 0):
            super().__init__(verbose)
            self.probe_path = Path(probe_path)
            self.binary_idx = [int(b) for b in binary_idx]
            self.labels = list(labels)
            # None -> the run directory, read from the SB3 logger at start
            self.out_dir = Path(out_dir) if out_dir else None
            self.every = int(every)
            self._raw: dict = {}
            self._header_written = False
            self._last_probe_t = -1

        # -- lifecycle ------------------------------------------------------
        def _on_training_start(self) -> None:
            if self.out_dir is None:
                self.out_dir = Path(self.model.logger.get_dir())
            self._csv_path = self.out_dir / "probe_log.csv"
            d = np.load(self.probe_path)
            self._obs_np = d["obs"].astype(np.float32)
            self._obs = th.as_tensor(self._obs_np, device=self.model.device)
            # the gene's current value in each probe state (the actor sees it in obs)
            self._gene = d["states"][:, self.binary_idx] > 0.5
            self._probe()

        def _on_step(self) -> bool:
            if self.num_timesteps % self.every == 0 and self.num_timesteps != self._last_probe_t:
                self._probe()
            return True

        def _on_training_end(self) -> None:
            if self.num_timesteps != self._last_probe_t:
                self._probe()
            arrays = {k: np.stack(v) for k, v in self._raw.items()}
            np.savez_compressed(self.out_dir / "probe_raw.npz",
                                binary_idx=np.array(self.binary_idx),
                                labels=np.array(self.labels), **arrays)
            meta = {"probe_path": str(self.probe_path), "every": self.every,
                    "n_states": int(self._obs_np.shape[0]), "sat": SAT, "pin_frac": PIN_FRAC,
                    "binary_idx": self.binary_idx, "labels": self.labels}
            (self.out_dir / "probe_meta.json").write_text(json.dumps(meta, indent=2))

        # -- the measurement ------------------------------------------------
        def _actor_outputs(self):
            actor = self.model.actor
            with th.no_grad():
                feats = actor.extract_features(self._obs, actor.features_extractor)
                pre = actor.mu[:-1](feats)
                if getattr(actor, "is_hybrid", False):
                    a_det = actor(self._obs, sample=False)
                    mu = th.tanh(pre)
                    for b in actor.binary_idx:
                        mu[:, b] = 2.0 * th.sigmoid(pre[:, b]) - 1.0
                else:
                    a_det = actor(self._obs)
                    mu = a_det
            return pre, mu, a_det

        def _critics(self, obs, a):
            out = self.model.critic(obs, a)
            return out if isinstance(out, (tuple, list)) else (out,)

        def _slope(self, a_base, b, value=None):
            a = a_base.clone()
            if value is not None:
                a[:, b] = value
            a.requires_grad_(True)
            qs = self._critics(self._obs, a)
            grads = []
            for i, q in enumerate(qs):
                g = th.autograd.grad(q.sum(), a, retain_graph=i < len(qs) - 1)[0]
                grads.append(g[:, b].detach().cpu().numpy())
            return np.stack(grads)

        def _q_at(self, a_base, b, value):
            a = a_base.clone()
            a[:, b] = value
            with th.no_grad():
                return np.stack([q.reshape(-1).cpu().numpy() for q in self._critics(self._obs, a)])

        def _probe(self) -> None:
            self._last_probe_t = self.num_timesteps
            pre, mu, a_det = self._actor_outputs()
            row = {"timesteps": int(self.num_timesteps)}
            raw = {"timesteps": np.array(self.num_timesteps)}
            for b, lab in zip(self.binary_idx, self.labels):
                m = mu[:, b].cpu().numpy()
                p = pre[:, b].cpu().numpy()
                tf = 1.0 - m ** 2
                dq_mu = self._slope(a_det, b)
                dq_p = self._slope(a_det, b, 0.5)
                dq_m = self._slope(a_det, b, -0.5)
                q_on = self._q_at(a_det, b, 1.0)
                q_off = self._q_at(a_det, b, -1.0)
                dqo = q_on - q_off
                frac_sat = float(np.mean(np.abs(m) > SAT))
                frac_on = float(np.mean(m > SAT))
                frac_off = float(np.mean(m < -SAT))
                g_on = self._gene[:, self.binary_idx.index(b)]
                g_off = ~g_on
                trap = float(np.mean(m[g_off] < -SAT)) if g_off.any() else float("nan")
                hold = float(np.mean(m[g_on] > SAT)) if g_on.any() else float("nan")
                row.update({
                    f"{lab}_mu_mean": float(m.mean()), f"{lab}_mu_median": float(np.median(m)),
                    f"{lab}_pre_median": float(np.median(p)),
                    f"{lab}_frac_sat": frac_sat, f"{lab}_frac_on": frac_on,
                    f"{lab}_frac_off": frac_off,
                    # conditional on the gene's current value: an actor that is saturated at
                    # -1 wherever the gene is off cannot turn it on, whatever it does when on
                    f"{lab}_off_given_off": trap, f"{lab}_on_given_on": hold,
                    f"{lab}_trapped": int(trap >= PIN_FRAC),
                    f"{lab}_pinned": int(frac_sat >= PIN_FRAC),
                    f"{lab}_tanhf_median": float(np.median(tf)), f"{lab}_tanhf_mean": float(tf.mean()),
                })
                for i in range(dq_mu.shape[0]):
                    q = f"q{i + 1}"
                    row.update({
                        f"{lab}_dq_mu_{q}_median": float(np.median(dq_mu[i])),
                        f"{lab}_absdq_mu_{q}_median": float(np.median(np.abs(dq_mu[i]))),
                        f"{lab}_dq_p05_{q}_median": float(np.median(dq_p[i])),
                        f"{lab}_absdq_p05_{q}_median": float(np.median(np.abs(dq_p[i]))),
                        f"{lab}_dq_m05_{q}_median": float(np.median(dq_m[i])),
                        f"{lab}_absdq_m05_{q}_median": float(np.median(np.abs(dq_m[i]))),
                        f"{lab}_qon_minus_qoff_{q}_mean": float(dqo[i].mean()),
                        f"{lab}_qon_minus_qoff_{q}_median": float(np.median(dqo[i])),
                        f"{lab}_frac_prefer_on_{q}": float(np.mean(dqo[i] > 0)),
                    })
                raw.update({
                    f"{lab}__mu": m.astype(np.float32), f"{lab}__pre": p.astype(np.float32),
                    f"{lab}__dq_mu": dq_mu.astype(np.float32), f"{lab}__dq_p05": dq_p.astype(np.float32),
                    f"{lab}__dq_m05": dq_m.astype(np.float32), f"{lab}__q_on": q_on.astype(np.float32),
                    f"{lab}__q_off": q_off.astype(np.float32),
                })
            for k, v in raw.items():
                self._raw.setdefault(k, []).append(v)
            self._write_row(row)

        def _write_row(self, row: dict) -> None:
            mode = "a" if self._header_written else "w"
            with self._csv_path.open(mode, newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not self._header_written:
                    w.writeheader()
                    self._header_written = True
                w.writerow(row)

    return ProbeCallback


def _make_ppo_callback_class():
    import torch as th
    from scipy.special import ndtr
    from stable_baselines3.common.callbacks import BaseCallback

    class PPOProbeCallback(BaseCallback):
        """The on/off dim of PPO's Gaussian policy, on the same 256-state probe set.

        PPO has no Q(s, a), so none of the critic-slope columns exist; what it
        has instead is an explicit action distribution, so the flip
        probabilities the off-policy probe has to infer from a noise std are
        read straight off the policy:

            mu_b, sigma_b            mean and std of the pre-clip Gaussian on dim b
            P(on | gene off)         P(a_b >= +0.5) = 1 - Phi((0.5 - mu)/sigma)
            P(off | gene on)         P(a_b <= -0.5) = Phi((-0.5 - mu)/sigma)

        Both are invariant to the [-1, 1] clip: the clip maps a > 1 to 1, which
        is still >= 0.5, and a < -1 to -1, which is still <= -0.5, so no
        probability mass crosses either threshold.

        frac_sat / pinned / trapped use mu_b under exactly the DDPG and TD3
        definitions (SAT = 0.96, PIN_FRAC = 0.95), so pins.py reads this file
        unchanged. One caveat worth stating in any comparison: PPO's mean is
        NOT tanh-squashed, so |mu_b| > 1 is reachable and 'saturated' is a
        weaker statement for PPO than for a tanh actor that has to be driven
        to its asymptote.
        """

        def __init__(self, probe_path: Path, binary_idx: Sequence[int],
                     labels: Sequence[str], out_dir: Optional[Path] = None,
                     every: int = 1000, verbose: int = 0):
            super().__init__(verbose)
            self.probe_path = Path(probe_path)
            self.binary_idx = [int(b) for b in binary_idx]
            self.labels = list(labels)
            self.out_dir = Path(out_dir) if out_dir else None
            self.every = int(every)
            self._raw: dict = {}
            self._header_written = False
            self._last_probe_t = -1

        # -- lifecycle ------------------------------------------------------
        def _on_training_start(self) -> None:
            if self.out_dir is None:
                self.out_dir = Path(self.model.logger.get_dir())
            self._csv_path = self.out_dir / "probe_log.csv"
            d = np.load(self.probe_path)
            self._obs_np = d["obs"].astype(np.float32)
            self._obs = th.as_tensor(self._obs_np, device=self.model.device)
            self._gene = d["states"][:, self.binary_idx] > 0.5
            self._probe()

        def _on_step(self) -> bool:
            if self.num_timesteps % self.every == 0 and self.num_timesteps != self._last_probe_t:
                self._probe()
            return True

        def _on_training_end(self) -> None:
            if self.num_timesteps != self._last_probe_t:
                self._probe()
            arrays = {k: np.stack(v) for k, v in self._raw.items()}
            np.savez_compressed(self.out_dir / "probe_raw.npz",
                                binary_idx=np.array(self.binary_idx),
                                labels=np.array(self.labels), **arrays)
            meta = {"probe_path": str(self.probe_path), "every": self.every,
                    "n_states": int(self._obs_np.shape[0]), "sat": SAT, "pin_frac": PIN_FRAC,
                    "binary_idx": self.binary_idx, "labels": self.labels, "algo": "ppo"}
            (self.out_dir / "probe_meta.json").write_text(json.dumps(meta, indent=2))

        # -- the measurement ------------------------------------------------
        def _mu_sigma(self):
            """(mu, sigma) of the policy's Gaussian, per probe state, no grad."""
            was_training = self.model.policy.training
            self.model.policy.set_training_mode(False)
            try:
                with th.no_grad():
                    d = self.model.policy.get_distribution(self._obs).distribution
                    mu = d.mean.cpu().numpy()
                    sigma = d.stddev.cpu().numpy()
            finally:
                self.model.policy.set_training_mode(was_training)
            if sigma.shape != mu.shape:          # state-independent log_std
                sigma = np.broadcast_to(sigma, mu.shape)
            return mu, sigma

        def _probe(self) -> None:
            self._last_probe_t = self.num_timesteps
            mu_all, sig_all = self._mu_sigma()
            row = {"timesteps": int(self.num_timesteps)}
            raw = {"timesteps": np.array(self.num_timesteps)}
            for b, lab in zip(self.binary_idx, self.labels):
                m = mu_all[:, b]
                s = np.maximum(sig_all[:, b], 1e-12)
                # clip-invariant: 0.5 and -0.5 are strictly inside [-1, 1]
                p_on = ndtr((m - 0.5) / s)        # P(a_b >= +0.5)
                p_off = ndtr((-0.5 - m) / s)      # P(a_b <= -0.5)
                frac_sat = float(np.mean(np.abs(m) > SAT))
                frac_on = float(np.mean(m > SAT))
                frac_off = float(np.mean(m < -SAT))
                g_on = self._gene[:, self.binary_idx.index(b)]
                g_off = ~g_on
                trap = float(np.mean(m[g_off] < -SAT)) if g_off.any() else float("nan")
                hold = float(np.mean(m[g_on] > SAT)) if g_on.any() else float("nan")
                row.update({
                    f"{lab}_mu_mean": float(m.mean()), f"{lab}_mu_median": float(np.median(m)),
                    f"{lab}_sigma_mean": float(s.mean()),
                    f"{lab}_sigma_median": float(np.median(s)),
                    f"{lab}_frac_sat": frac_sat, f"{lab}_frac_on": frac_on,
                    f"{lab}_frac_off": frac_off,
                    f"{lab}_off_given_off": trap, f"{lab}_on_given_on": hold,
                    f"{lab}_trapped": int(trap >= PIN_FRAC),
                    f"{lab}_pinned": int(frac_sat >= PIN_FRAC),
                    # flip probabilities per step, conditional on the gene's value
                    f"{lab}_p_on_given_off_median":
                        float(np.median(p_on[g_off])) if g_off.any() else float("nan"),
                    f"{lab}_p_on_given_off_mean":
                        float(p_on[g_off].mean()) if g_off.any() else float("nan"),
                    f"{lab}_p_off_given_on_median":
                        float(np.median(p_off[g_on])) if g_on.any() else float("nan"),
                    f"{lab}_p_off_given_on_mean":
                        float(p_off[g_on].mean()) if g_on.any() else float("nan"),
                    f"{lab}_p_on_median_all": float(np.median(p_on)),
                    f"{lab}_p_off_median_all": float(np.median(p_off)),
                })
                raw.update({
                    f"{lab}__mu": m.astype(np.float32), f"{lab}__sigma": s.astype(np.float32),
                    f"{lab}__p_on": p_on.astype(np.float32),
                    f"{lab}__p_off": p_off.astype(np.float32),
                })
            for k, v in raw.items():
                self._raw.setdefault(k, []).append(v)
            self._write_row(row)

        def _write_row(self, row: dict) -> None:
            mode = "a" if self._header_written else "w"
            with self._csv_path.open(mode, newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not self._header_written:
                    w.writeheader()
                    self._header_written = True
                w.writerow(row)

    return PPOProbeCallback


_CB = None
_PPO_CB = None


def ProbeCallback(*args, **kwargs):
    """Factory so that importing this module does not import torch/SB3."""
    global _CB
    if _CB is None:
        _CB = _make_callback_class()
    return _CB(*args, **kwargs)


def PPOProbeCallback(*args, **kwargs):
    """Same, for the on-policy (PPO) probe."""
    global _PPO_CB
    if _PPO_CB is None:
        _PPO_CB = _make_ppo_callback_class()
    return _PPO_CB(*args, **kwargs)


def binary_dims_and_labels(env_cfg) -> tuple[List[int], List[str]]:
    from ddpg.config import binary_action_indices
    from ppo.env import state_labels

    idx = binary_action_indices(env_cfg)
    labels = state_labels(env_cfg)
    return idx, [labels[i] for i in idx]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m letterexp.probe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="sample and save the fixed probe set")
    b.add_argument("--code", default="1-1-1")
    b.add_argument("--backend", default="pyqd-fast")
    b.add_argument("--n", type=int, default=N_PROBE)
    b.add_argument("--seed", type=int, default=PROBE_SEED)
    b.add_argument("--band", default=None, help="band mode (Stage 6b), default protocol's")
    s = sub.add_parser("show", help="print a sample of a run's probe_log.csv")
    s.add_argument("--run", required=True)
    s.add_argument("--rows", type=int, default=8)
    args = ap.parse_args(argv)

    if args.cmd == "build":
        from dataclasses import replace

        from tuning.protocol import DEFAULT
        from tuning.spaces import build_experiment

        from .runner import base_params
        proto = replace(DEFAULT, code=args.code, backend=args.backend)
        exp = build_experiment("ddpg", base_params("ddpg_t27")["params"], proto, 0, "probe")
        tag = ""
        if args.band:
            from .runner import apply_band
            exp = apply_band(exp, args.band)
            tag = args.band
        out = build_probe_set(exp.env, args.backend, args.n, args.seed,
                              out=probe_set_path(args.code, args.backend, args.n, args.seed, tag))
        d = np.load(out)
        print(f"probe set -> {out}  obs {d['obs'].shape}  "
              f"power_on share {np.mean(d['states'][:, 4] > 0.5):.3f}")
        return 0

    import pandas as pd
    df = pd.read_csv(Path(args.run) / "probe_log.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    idx = np.unique(np.linspace(0, len(df) - 1, args.rows).astype(int))
    print(f"{len(df)} checkpoints x {df.shape[1]} columns")
    print(df.iloc[idx].T.to_string(float_format=lambda v: f"{v:.4g}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
