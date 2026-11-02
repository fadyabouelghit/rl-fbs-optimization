"""Training plots for TD3 runs.

ppo's training overview plots only PPO's optimizer keys, which a TD3 run does
not log. This gallery reuses ppo's figures as they are and fills the
optimizer panel with TD3's actor and critic losses instead.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ppo.plotting import load_training_frames, plot_reward_components, plot_training_overview, save_fig


def training_gallery(run_dir: Path | str, window: int = 20) -> Path:
    """Render the training plots into <run_dir>/plots/."""
    run_dir = Path(run_dir)
    out = run_dir / "plots"
    fig = plot_training_overview(run_dir, window=window,
                                 title=f"TD3 training overview — {run_dir.name}")
    ax = fig.axes[5]                      # "Optimizer diagnostics" (row 3, col 2)
    prog = load_training_frames(run_dir)["progress"]
    plotted = False
    if prog is not None and "time/total_timesteps" in prog:
        for col, label in (("train/actor_loss", "actor loss (−Q)"),
                           ("train/critic_loss", "critic loss")):
            if col in prog:
                series = prog[["time/total_timesteps", col]].dropna()
                ax.plot(series["time/total_timesteps"], series[col], lw=1.2, label=label)
                plotted = True
    if plotted:
        ax.legend()
    save_fig(fig, out / "training_overview")
    plt.close(fig)
    fig = plot_reward_components(run_dir, save=out / "reward_components")
    plt.close(fig)
    return out
