"""Plotting for the PPO pipeline — everything renders from the flat CSV
artifacts (no live env or TensorBoard required).

Training-side (consume a run directory):
    plot_training_overview   6-panel training dashboard
    plot_reward_components   reward-term decomposition over training
    plot_learning_curves     multi-run overlay of an episode metric

Rollout-side (consume evaluation artifacts / DataFrames):
    plot_trajectory_map      2-D world map with time-colored FBS paths
    plot_altitude_power      altitude + tx power profiles with off shading
    plot_step_metrics        per-step reward/connectivity/power/rate grid
    plot_connectivity_stack  per-tier stacked connectivity
    plot_eval_summary        distribution of final metrics across episodes
    eval_gallery             renders the full gallery into an eval dir

Every function returns the figure; pass ``save=<path stem>`` to write
``<stem>.png`` and ``<stem>.pdf`` at publication settings (serif, 300 dpi).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from .config import WorldConfig

STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8.5,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

FBS_COLORS = plt.cm.tab10.colors
TIER_COLORS = {"fbs": "#1f77b4", "mbs_coverage": "#ff7f0e", "mbs_capacity": "#2ca02c"}


def apply_style():
    mpl.rcParams.update(STYLE)


def save_fig(fig, stem: str | Path, formats: Sequence[str] = ("png", "pdf")) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), bbox_inches="tight")


def _rolling(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window=max(int(window), 1), min_periods=1).mean()


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path, comment="#")
    except Exception:
        pass
    return None


def load_training_frames(run_dir: Path | str) -> dict:
    run_dir = Path(run_dir)
    return {
        "steps": _read_csv(run_dir / "steps.csv"),
        "episodes": _read_csv(run_dir / "episodes.csv"),
        "progress": _read_csv(run_dir / "progress.csv"),
        "eval": _read_csv(run_dir / "eval_log.csv"),
    }


# --------------------------------------------------------------------------- #
# Training plots
# --------------------------------------------------------------------------- #
def plot_training_overview(
    run_dir: Path | str,
    window: int = 20,
    save: Optional[str | Path] = None,
    title: Optional[str] = None,
):
    apply_style()
    run_dir = Path(run_dir)
    frames = load_training_frames(run_dir)
    eps, steps, prog = frames["episodes"], frames["steps"], frames["progress"]

    fig, axes = plt.subplots(3, 2, figsize=(9.5, 9), sharex=False)
    fig.suptitle(title or f"Training overview — {run_dir.name}", y=0.995)

    ax = axes[0, 0]
    if eps is not None and "reward_sum" in eps:
        ax.plot(eps.index, eps["reward_sum"], alpha=0.25, lw=0.8, label="episode")
        ax.plot(eps.index, _rolling(eps["reward_sum"], window), lw=1.6,
                label=f"rolling({window})")
        ax.legend()
    ax.set_title("Episode reward")
    ax.set_xlabel("Episode")

    ax = axes[0, 1]
    if eps is not None and "final_total_connected" in eps:
        ax.plot(eps.index, _rolling(eps["final_total_connected"], window),
                color=TIER_COLORS["mbs_coverage"], lw=1.6, label="total")
        if "final_fbs_connected" in eps:
            ax.plot(eps.index, _rolling(eps["final_fbs_connected"], window),
                    color=TIER_COLORS["fbs"], lw=1.4, label="FBS")
        if "final_mbs_connected" in eps:
            ax.plot(eps.index, _rolling(eps["final_mbs_connected"], window),
                    color=TIER_COLORS["mbs_capacity"], lw=1.4, label="MBS")
        ax.legend()
    ax.set_title("Connected users (episode end)")
    ax.set_xlabel("Episode")

    ax = axes[1, 0]
    if steps is not None and "reward" in steps:
        ax.plot(steps["timesteps"], steps["reward"], alpha=0.15, lw=0.5)
        ax.plot(steps["timesteps"], _rolling(steps["reward"], window * 10), lw=1.5)
    ax.set_title("Step reward")
    ax.set_xlabel("Timesteps")

    ax = axes[1, 1]
    if eps is not None and "final_total_power" in eps:
        ax.plot(eps.index, _rolling(eps["final_total_power"], window), lw=1.5)
    ax.set_title("Total tx power (episode end)")
    ax.set_xlabel("Episode")

    ax = axes[2, 0]
    if eps is not None and "final_avg_rate" in eps:
        ax.plot(eps.index, _rolling(eps["final_avg_rate"], window), lw=1.5)
    ax.set_title("Avg connected rate (bps/Hz)")
    ax.set_xlabel("Episode")

    ax = axes[2, 1]
    plotted = False
    if prog is not None:
        for col, label in (
            ("train/policy_gradient_loss", "policy loss"),
            ("train/value_loss", "value loss"),
            ("train/entropy_loss", "entropy loss"),
            ("train/approx_kl", "approx KL"),
        ):
            if col in prog:
                series = prog[["time/total_timesteps", col]].dropna() \
                    if "time/total_timesteps" in prog else prog[[col]].dropna()
                xs = series["time/total_timesteps"] if "time/total_timesteps" in series else series.index
                ax.plot(xs, series[col], lw=1.2, label=label)
                plotted = True
    if plotted:
        ax.legend()
    ax.set_title("Optimizer diagnostics")
    ax.set_xlabel("Timesteps")

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    if save:
        save_fig(fig, save)
    return fig


def plot_reward_components(
    run_dir: Path | str,
    window: int = 200,
    save: Optional[str | Path] = None,
):
    apply_style()
    run_dir = Path(run_dir)
    steps = load_training_frames(run_dir)["steps"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    fig.suptitle(f"Reward decomposition — {run_dir.name}")

    if steps is not None:
        ax = axes[0]
        for col, label in (
            ("reward", "reward"),
            ("reward_base", "base term"),
            ("reward_fbs_term", "FBS term"),
        ):
            if col in steps:
                ax.plot(steps["timesteps"], _rolling(steps[col], window), lw=1.4, label=label)
        ax.set_xlabel("Timesteps")
        ax.legend()
        ax.set_title("Blend terms")

        ax = axes[1]
        for col, label in (
            ("reward_norm_users", "norm users"),
            ("reward_norm_power", "norm power"),
            ("reward_fbs_share", "FBS share"),
        ):
            if col in steps:
                ax.plot(steps["timesteps"], _rolling(steps[col], window), lw=1.4, label=label)
        ax.set_xlabel("Timesteps")
        ax.legend()
        ax.set_title("Normalized components")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if save:
        save_fig(fig, save)
    return fig


def plot_learning_curves(
    run_dirs: Sequence[Path | str],
    metric: str = "reward_sum",
    window: int = 20,
    labels: Optional[Sequence[str]] = None,
    save: Optional[str | Path] = None,
):
    """Overlay one episode-level metric across multiple runs."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for k, rd in enumerate(run_dirs):
        rd = Path(rd)
        eps = load_training_frames(rd)["episodes"]
        if eps is None or metric not in eps:
            continue
        label = labels[k] if labels else rd.name.replace("run_", "")
        ax.plot(eps.index, _rolling(eps[metric], window), lw=1.5, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} (rolling {window})")
    ax.legend()
    fig.tight_layout()
    if save:
        save_fig(fig, save)
    return fig


def plot_reward_curve(
    run_dir: Path | str,
    window: int = 15,
    ax=None,
    title: Optional[str] = None,
    color: str = "#16324f",
    save: Optional[str | Path] = None,
):
    """Single-config learning curve: episode return vs cumulative TIMESTEPS.

    Raw per-episode return (light) plus a rolling mean (dark). The x-axis is
    environment timesteps (RL's common currency across papers), read from the
    ``timesteps`` column of episodes.csv (cumulative step count at each episode
    end). One figure per configuration — no overlays.
    """
    apply_style()
    run_dir = Path(run_dir)
    eps = _read_csv(run_dir / "episodes.csv")
    if eps is None or "reward_sum" not in eps:
        raise FileNotFoundError(f"no episodes.csv with reward_sum in {run_dir}")
    x = eps["timesteps"] if "timesteps" in eps else (eps.index + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.4, 3.0))
    else:
        fig = ax.figure
    ax.plot(x, eps["reward_sum"], color="#b6bec7", lw=0.7, label="episode return")
    ax.plot(x, _rolling(eps["reward_sum"], window), color=color, lw=1.7,
            label=f"rolling mean ({window} ep)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Episode return")
    ax.set_title(title or run_dir.name, fontsize=9.5)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, handlelength=1.6)
    fig.tight_layout()
    if save:
        save_fig(fig, save)
    return fig


# --------------------------------------------------------------------------- #
# Rollout / trajectory plots
# --------------------------------------------------------------------------- #
def _fbs_ids(state_df: pd.DataFrame) -> list[str]:
    return sorted(
        {c[0] for c in state_df.columns if str(c[0]).startswith("fbs")},
        key=lambda s: int(s[3:]),
    )


def plot_trajectory_map(
    state_df: pd.DataFrame,
    mbs_x: Optional[np.ndarray] = None,
    mbs_y: Optional[np.ndarray] = None,
    world: Optional[WorldConfig] = None,
    users: Optional[np.ndarray] = None,
    title: str = "FBS trajectory",
    ax=None,
    save: Optional[str | Path] = None,
):
    """2-D world map: time-colored FBS paths, on/off + band markers, MBS sites."""
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
    else:
        fig = ax.figure

    if users is not None and len(users):
        ax.scatter(users[:, 0], users[:, 1], s=3, c="0.82", zorder=1, label="users")

    if mbs_x is not None and len(np.atleast_1d(mbs_x)):
        ax.scatter(
            np.atleast_1d(mbs_x), np.atleast_1d(mbs_y),
            marker="X", s=130, c="black", zorder=5, label="MBS",
        )

    steps = np.arange(len(state_df))
    for k, fbs in enumerate(_fbs_ids(state_df)):
        x = state_df[(fbs, "x")].to_numpy(float)
        y = state_df[(fbs, "y")].to_numpy(float)
        on = state_df[(fbs, "power_status")].to_numpy(float) >= 0.5
        band = (
            state_df[(fbs, "band_flag")].to_numpy(float)
            if (fbs, "band_flag") in state_df.columns
            else None
        )

        pts = np.column_stack([x, y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(
            segs, cmap="viridis",
            norm=plt.Normalize(0, max(len(steps) - 1, 1)),
            linewidths=1.8, zorder=3, alpha=0.95,
        )
        lc.set_array(steps[:-1])
        ax.add_collection(lc)

        base_color = FBS_COLORS[k % len(FBS_COLORS)]
        # ON steps: filled markers; OFF steps: open markers.
        if band is None:
            ax.scatter(x[on], y[on], s=16, marker="o",
                       facecolors=base_color, edgecolors="none", zorder=4)
        else:
            cov, cap = on & (band < 0.5), on & (band >= 0.5)
            ax.scatter(x[cov], y[cov], s=16, marker="o",
                       facecolors=base_color, edgecolors="none", zorder=4)
            ax.scatter(x[cap], y[cap], s=20, marker="s",
                       facecolors=base_color, edgecolors="none", zorder=4)
        ax.scatter(x[~on], y[~on], s=18, marker="o",
                   facecolors="none", edgecolors=base_color,
                   linewidths=0.9, zorder=4)

        ax.scatter(x[0], y[0], marker="^", s=90, c="forestgreen",
                   edgecolors="black", linewidths=0.5, zorder=6)
        ax.scatter(x[-1], y[-1], marker="*", s=160, c="crimson",
                   edgecolors="black", linewidths=0.5, zorder=6)
        ax.annotate(fbs.upper(), (x[-1], y[-1]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)

    if world is not None:
        ax.set_xlim(0, world.width)
        ax.set_ylim(0, world.height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title, fontsize=9.5, wrap=True)

    handles = [
        Line2D([], [], marker="^", ls="none", c="forestgreen",
               markeredgecolor="black", label="start"),
        Line2D([], [], marker="*", ls="none", c="crimson",
               markeredgecolor="black", label="end"),
        Line2D([], [], marker="o", ls="none", markerfacecolor="none",
               markeredgecolor="0.3", label="FBS off"),
    ]
    if any((f, "band_flag") in state_df.columns for f in _fbs_ids(state_df)):
        handles += [
            Line2D([], [], marker="o", ls="none", c="0.3", label="coverage band"),
            Line2D([], [], marker="s", ls="none", c="0.3", label="capacity band"),
        ]
    if mbs_x is not None and len(np.atleast_1d(mbs_x)):
        handles.append(Line2D([], [], marker="X", ls="none", c="black", label="MBS"))
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(
            cmap="viridis", norm=plt.Normalize(0, max(len(steps) - 1, 1))
        ),
        ax=ax, fraction=0.04, pad=0.02,
    )
    cbar.set_label("step")
    fig.tight_layout()
    if save:
        save_fig(fig, save)
    return fig


def plot_altitude_power(
    state_df: pd.DataFrame,
    title: str = "Altitude / power profile",
    save: Optional[str | Path] = None,
):
    apply_style()
    fbs_list = _fbs_ids(state_df)
    fig, axes = plt.subplots(
        len(fbs_list), 1, figsize=(6.4, 2.6 * len(fbs_list)), sharex=True, squeeze=False
    )
    steps = state_df.index

    for k, fbs in enumerate(fbs_list):
        ax = axes[k, 0]
        color = FBS_COLORS[k % len(FBS_COLORS)]
        ax.plot(steps, state_df[(fbs, "height")], lw=1.6, color=color, label="altitude")
        ax.set_ylabel(f"{fbs} altitude (m)")

        ax2 = ax.twinx()
        ax2.plot(steps, state_df[(fbs, "power")], lw=1.4, ls=":", color="0.25",
                 label="tx power")
        ax2.set_ylabel("tx power")
        ax2.grid(False)

        off = state_df[(fbs, "power_status")].to_numpy(float) < 0.5
        if off.any():
            ax.fill_between(steps, 0, 1, where=off, transform=ax.get_xaxis_transform(),
                            color="red", alpha=0.08, label="off")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    axes[-1, 0].set_xlabel("Step")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save:
        save_fig(fig, save)
    return fig


def plot_step_metrics(
    metrics_df: pd.DataFrame,
    title: str = "Rollout metrics",
    save: Optional[str | Path] = None,
):
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), sharex=True)
    fig.suptitle(title)

    ax = axes[0, 0]
    if "reward" in metrics_df:
        ax.plot(metrics_df.index, metrics_df["reward"], lw=1.5)
    ax.set_title("Reward")

    ax = axes[0, 1]
    for col, label, color in (
        ("total_connected", "total", "0.2"),
        ("fbs_connected", "FBS", TIER_COLORS["fbs"]),
        ("mbs_connected", "MBS", TIER_COLORS["mbs_coverage"]),
    ):
        if col in metrics_df:
            ax.plot(metrics_df.index, metrics_df[col], lw=1.4, label=label, color=color)
    ax.legend()
    ax.set_title("Connected users")

    ax = axes[1, 0]
    if "total_power" in metrics_df:
        ax.plot(metrics_df.index, metrics_df["total_power"], lw=1.5)
    if "num_active_fbs" in metrics_df:
        ax2 = ax.twinx()
        ax2.step(metrics_df.index, metrics_df["num_active_fbs"], where="post",
                 color="0.4", lw=1.0, ls="--")
        ax2.set_ylabel("# active FBS")
        ax2.grid(False)
    ax.set_title("Tx power")
    ax.set_xlabel("Step")

    ax = axes[1, 1]
    if "avg_rate" in metrics_df:
        ax.plot(metrics_df.index, metrics_df["avg_rate"], lw=1.5)
    ax.set_title("Avg rate (bps/Hz)")
    ax.set_xlabel("Step")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save:
        save_fig(fig, save)
    return fig


def plot_connectivity_stack(
    metrics_df: pd.DataFrame,
    title: str = "Per-tier connectivity",
    save: Optional[str | Path] = None,
):
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    tiers = [
        ("fbs_connected", "FBS", TIER_COLORS["fbs"]),
        ("mbs_coverage_connected", "MBS coverage", TIER_COLORS["mbs_coverage"]),
        ("mbs_capacity_connected", "MBS capacity", TIER_COLORS["mbs_capacity"]),
    ]
    present = [(c, l, col) for c, l, col in tiers if c in metrics_df]
    if present:
        ax.stackplot(
            metrics_df.index,
            [metrics_df[c] for c, _, _ in present],
            labels=[l for _, l, _ in present],
            colors=[col for _, _, col in present],
            alpha=0.85,
        )
        ax.legend(loc="upper left")
    ax.set_xlabel("Step")
    ax.set_ylabel("Connected users")
    ax.set_title(title)
    fig.tight_layout()
    if save:
        save_fig(fig, save)
    return fig


def plot_eval_summary(
    summary_df: pd.DataFrame,
    title: str = "Evaluation summary",
    save: Optional[str | Path] = None,
):
    apply_style()
    metrics = [
        ("reward_sum", "Episode reward"),
        ("final_total_connected", "Final connected"),
        ("final_total_power", "Final tx power"),
        ("length", "Episode length"),
    ]
    present = [(c, l) for c, l in metrics if c in summary_df]
    fig, axes = plt.subplots(1, len(present), figsize=(2.6 * len(present), 3.2))
    axes = np.atleast_1d(axes)
    fig.suptitle(title)
    for ax, (col, label) in zip(axes, present):
        vals = summary_df[col].astype(float)
        ax.boxplot([vals.dropna()], widths=0.5, showmeans=True)
        jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.16
        ax.scatter(1 + jitter, vals, s=18, alpha=0.8, zorder=3)
        ax.set_xticks([])
        ax.set_title(label, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if save:
        save_fig(fig, save)
    return fig


# --------------------------------------------------------------------------- #
# Galleries
# --------------------------------------------------------------------------- #
def training_gallery(run_dir: Path | str, window: int = 20) -> Path:
    """Render every training plot into <run_dir>/plots/."""
    run_dir = Path(run_dir)
    out = run_dir / "plots"
    fig = plot_training_overview(run_dir, window=window, save=out / "training_overview")
    plt.close(fig)
    fig = plot_reward_components(run_dir, save=out / "reward_components")
    plt.close(fig)
    return out


def eval_gallery(
    eval_dir: Path,
    rollouts,
    summary_df: pd.DataFrame,
    mbs_x: np.ndarray,
    mbs_y: np.ndarray,
    world: WorldConfig,
    title: str = "",
) -> Path:
    """Render the rollout gallery into <eval_dir>/plots/."""
    out = Path(eval_dir) / "plots"
    for r in rollouts:
        prefix = f"ep{r.index:02d}"
        fig = plot_trajectory_map(
            r.state_df, mbs_x, mbs_y, world,
            users=r.user_positions,
            title=f"{title} — episode {r.index}",
            save=out / f"{prefix}_trajectory",
        )
        plt.close(fig)
        fig = plot_altitude_power(
            r.state_df, title=f"{title} — episode {r.index}",
            save=out / f"{prefix}_altitude_power",
        )
        plt.close(fig)
        fig = plot_step_metrics(
            r.metrics_df, title=f"{title} — episode {r.index}",
            save=out / f"{prefix}_metrics",
        )
        plt.close(fig)
        if "mbs_coverage_connected" in r.metrics_df:
            fig = plot_connectivity_stack(
                r.metrics_df, title=f"{title} — episode {r.index}",
                save=out / f"{prefix}_tiers",
            )
            plt.close(fig)
    if len(summary_df) > 1:
        fig = plot_eval_summary(summary_df, title=title, save=out / "summary")
        plt.close(fig)
    return out
