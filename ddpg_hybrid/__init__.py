"""DDPG with a stochastic Bernoulli (straight-through Gumbel) head on the
on/off dims only. Opt-in; imports ddpg/ and ppo/ and modifies neither.

    from ddpg_hybrid.train import train
    train(exp, backend="pyqd-fast", gumbel_tau=1.0)   # exp.ddpg.binary_noise_std must be 0.0

See ``ddpg_hybrid.hybrid`` for the mechanism.
"""
