"""Experiments for the IEEE letter on actor pinning with thresholded on/off genes.

Nothing in here changes how ppo/, ddpg/ or td3/ train by default. Every piece
is opt-in and new:

    probe.py          fixed probe-state set + ProbeCallback (actor/critic at the
                      on/off dim, every 1000 steps) -> probe_log.csv, probe_raw.npz
    compact_steps.py  CompactStepCallback -> steps_compact.csv.gz (flip analysis)
    pins.py           pin-event detection from probe_log.csv / probe_raw.npz
    stats.py          Wilson, Fisher exact, Welch, chi-square helpers
    runner.py         one run = one subprocess: train + score best and final
    launch.py         experiment plans (E2, E5, smoke, repro) + worker pool
    e1_flip.py        Stage 1: clamped-actor check of the flip formula
    analyze.py        Stage 2/3/4/5 result files and figures

All new run artifacts go under ``letter_runs/<experiment>/``.
"""
