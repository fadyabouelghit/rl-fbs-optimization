"""Letter instrumentation for switch + pilot runs: switch probe + compact step log.

The actor-based ``letterexp.probe.ProbeCallback`` does not apply (the pilot has
no on/off output); ``SwitchProbeCallback`` is its counterpart.
"""
from __future__ import annotations

from .callbacks import SwitchProbeCallback, make_compact_step_callback


def instrument_callbacks(exp, protocol, probe_every: int = 1000, positions: bool = False) -> list:
    from letterexp.probe import binary_dims_and_labels, probe_set_path

    idx, labels = binary_dims_and_labels(exp.env)
    probe = probe_set_path(protocol.code, protocol.backend)
    if not probe.is_file():
        raise FileNotFoundError(
            f"probe set {probe} missing; run: python -m letterexp.probe build "
            f"--code {protocol.code} --backend {protocol.backend}")
    kw = {}
    if positions:
        from ppo.env import FlyingBaseStationEnv
        e = FlyingBaseStationEnv(exp.env, backend_kind="analytic")   # bounds only
        try:
            kw = {"positions": True, "state_low": e.state_low, "state_high": e.state_high,
                  "n_fbs": exp.env.num_fbs, "genes_per_fbs": e.genes_per_fbs}
        finally:
            e.close()
    return [SwitchProbeCallback(probe, idx, labels, every=probe_every),
            make_compact_step_callback(None, idx, labels,
                                       obs_normalized=bool(exp.env.normalize_obs), **kw)]
