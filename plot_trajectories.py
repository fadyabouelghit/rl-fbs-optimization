"""
plot_trajectories.py — overlay GA + PPO FBS trajectories on common axes.

Supported sources, auto-detected from path:
  - ga_runs/run_NNN/                              -> uses trajectory.csv or result.mat
  - ga_runs/run_NNN/trajectory.csv                -> flat per-generation best individuals
  - ga_runs/run_NNN/result.mat                    -> reads history.bestIndividuals
  - test_logs/<X-Y-Z>/<stem>                      -> appends "_trajectory.csv"
  - test_logs/<X-Y-Z>/<stem>_trajectory.csv       -> PPO rollout (MultiIndex CSV)

Constraint: do not mix 1-FBS and 2-FBS trajectories in one figure.
The plotter will raise if num_fbs differs across the inputs.

Python usage:

    from plot_trajectories import plot_trajectories
    plot_trajectories([
        ("ga_runs/run_001",                                 "GA gen-best"),
        ("test_logs/1-1-1/run_039_20260501_122954",         "PPO rollout"),
    ], title="GA vs PPO  (1-1-1)")

CLI:

    python plot_trajectories.py \\
        ga_runs/run_001 \\
        test_logs/1-1-1/run_039_20260501_122954_trajectory.csv \\
        --label "GA" --label "PPO" --title "GA vs PPO" --save out.png
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

try:
    from ppo_experiment import SCENARIOS as RL_SCENARIOS
except ImportError:
    RL_SCENARIOS = {}


REPO_ROOT = Path(__file__).resolve().parent
GA_RUNS = REPO_ROOT / "ga_runs"
RL_TEST_LOGS = REPO_ROOT / "test_logs"


# --------------------------------------------------------------------------- #
@dataclass
class Trajectory:
    name: str
    source: str                      # "ga" or "rl"
    num_fbs: int
    df: pd.DataFrame                 # flat columns: fbs<i>_<field> + 'step'
    mbs_x: Optional[np.ndarray] = None
    mbs_y: Optional[np.ndarray] = None
    code: Optional[str] = None       # X-Y-Z if known
    path: Optional[Path] = None


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _flatten_rl_csv(path: Path) -> pd.DataFrame:
    """Read a PPO trajectory CSV (MultiIndex headers) -> flat-named DataFrame."""
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    keep = []
    new_cols = []
    for lvl0, lvl1 in df.columns:
        if str(lvl0).startswith("fbs"):
            keep.append((lvl0, lvl1))
            new_cols.append(f"{lvl0}_{lvl1}")
    flat = df.loc[:, keep].copy()
    flat.columns = new_cols
    flat = flat.reset_index(drop=True)
    flat["step"] = np.arange(len(flat))
    return flat


def _read_ga_mat(path: Path) -> Tuple[pd.DataFrame, Optional[np.ndarray],
                                      Optional[np.ndarray], int, Optional[str]]:
    """Load a GA result.mat -> (flat_df, mbs_x, mbs_y, num_fbs, code)."""
    from scipy.io import loadmat

    data = loadmat(path, struct_as_record=False, squeeze_me=True)
    history = data["history"]
    best = np.asarray(history.bestIndividuals, dtype=float)
    if best.ndim == 1:
        best = best.reshape(1, -1)

    # num_fbs: prefer the saved exp struct, fall back to reasonable guess
    num_fbs = None
    code = None
    if "exp" in data:
        exp_obj = data["exp"]
        num_fbs = int(getattr(exp_obj, "numBS", 0)) or None
        code = getattr(exp_obj, "code", None)
        if isinstance(code, np.ndarray):
            code = str(code)
    total_cols = best.shape[1]
    if num_fbs is None:
        # Layouts seen so far:
        #   5 * num_fbs              (legacy)
        #   6 * num_fbs              (post multi-band commit, no MBS suffix)
        #   6 * num_fbs + num_mbs    (current: trailing MBS band_flag rows)
        candidates = []
        for genes in (5, 6):
            if total_cols % genes == 0:
                candidates.append((genes, total_cols // genes, 0))
        # Try 6 genes/FBS plus a small trailing MBS block
        for trailing in range(1, min(total_cols // 6, 8) + 1):
            rem = total_cols - trailing
            if rem > 0 and rem % 6 == 0:
                candidates.append((6, rem // 6, trailing))
        if not candidates:
            raise ValueError(f"Cannot infer num_fbs from {total_cols} columns in {path}")
        # Prefer the layout with trailing MBS columns (matches current GA output)
        candidates.sort(key=lambda c: (c[0] != 6, -c[2]))
        genes_per_fbs, num_fbs, num_mbs_trailing = candidates[0]
    else:
        # num_fbs known from saved exp struct; trailing cols (if any) are MBS.
        if total_cols >= 6 * num_fbs:
            genes_per_fbs = 6
            num_mbs_trailing = total_cols - 6 * num_fbs
        else:
            genes_per_fbs = total_cols // num_fbs
            num_mbs_trailing = 0

    base_names = ["x", "y", "height", "power", "power_status", "band_flag"][:genes_per_fbs]
    cols = [f"fbs{i}_{n}" for i in range(num_fbs) for n in base_names]
    cols += [f"mbs{i}_band_flag" for i in range(num_mbs_trailing)]
    df = pd.DataFrame(best, columns=cols)
    df["step"] = np.arange(len(df))

    mbs_x = mbs_y = None
    if "mbs_params" in data:
        mp = np.atleast_2d(np.asarray(data["mbs_params"], dtype=float))
        # squeeze_me=True can flatten (4, 1) -> (4,); after atleast_2d it's
        # either (4, K) for K MBSs or (1, 4) when there's just one. Detect:
        if mp.shape[0] != 4 and mp.shape[1] == 4:
            mp = mp.T
        # ga_experiment.m row-swaps to (x, y, h, p)
        mbs_x = mp[0, :].ravel()
        mbs_y = mp[1, :].ravel()

    return df, mbs_x, mbs_y, num_fbs, code


def _resolve_path(spec: Union[str, Path]) -> Path:
    """Map a user-friendly spec to a concrete file."""
    p = Path(spec)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve() if not p.exists() else p.resolve()
    if p.is_dir():
        for cand in ("trajectory.csv", "result.mat"):
            if (p / cand).exists():
                return p / cand
        raise FileNotFoundError(f"No trajectory.csv or result.mat in {p}")
    if not p.exists():
        sibling = p.with_name(p.name + "_trajectory.csv")
        if sibling.exists():
            return sibling
        raise FileNotFoundError(p)
    return p


def load_trajectory(spec: Union[str, Path], name: Optional[str] = None) -> Trajectory:
    """Auto-detect format and return a normalized `Trajectory`."""
    path = _resolve_path(spec)
    parts = path.parts

    # ----- GA result.mat
    if path.suffix == ".mat":
        df, mbs_x, mbs_y, num_fbs, code = _read_ga_mat(path)
        cfg = path.parent / "experiment_config.json"
        if cfg.exists() and code is None:
            with cfg.open() as f:
                code = json.load(f).get("code")
        return Trajectory(
            name=name or f"GA {path.parent.name}",
            source="ga",
            num_fbs=num_fbs,
            df=df,
            mbs_x=mbs_x,
            mbs_y=mbs_y,
            code=code,
            path=path,
        )

    # ----- GA trajectory.csv
    if "ga_runs" in parts and path.name == "trajectory.csv":
        df = pd.read_csv(path)
        num_fbs = sum(1 for c in df.columns if c.startswith("fbs") and c.endswith("_x"))
        mbs_x = mbs_y = None
        rmat = path.parent / "result.mat"
        if rmat.exists():
            try:
                _, mbs_x, mbs_y, _, _ = _read_ga_mat(rmat)
            except Exception:
                pass
        code = None
        cfg = path.parent / "experiment_config.json"
        if cfg.exists():
            with cfg.open() as f:
                code = json.load(f).get("code")
        return Trajectory(
            name=name or f"GA {path.parent.name}",
            source="ga",
            num_fbs=num_fbs,
            df=df,
            mbs_x=mbs_x,
            mbs_y=mbs_y,
            code=code,
            path=path,
        )

    # ----- PPO test rollout
    if "test_logs" in parts:
        df = _flatten_rl_csv(path)
        num_fbs = sum(1 for c in df.columns if c.startswith("fbs") and c.endswith("_x"))
        code = None
        try:
            idx = parts.index("test_logs")
            code = parts[idx + 1]
        except (ValueError, IndexError):
            pass

        mbs_x = mbs_y = None
        if code:
            try:
                _, scenario_id, _ = (int(x) for x in code.split("-"))
                scen = RL_SCENARIOS.get(scenario_id, {})
                if scen.get("mbs_locations"):
                    locs = np.asarray(scen["mbs_locations"], dtype=float)
                    mbs_x, mbs_y = locs[:, 0], locs[:, 1]
            except (ValueError, KeyError):
                pass

        stem = path.name.replace("_trajectory.csv", "")
        return Trajectory(
            name=name or f"PPO {stem}",
            source="rl",
            num_fbs=num_fbs,
            df=df,
            mbs_x=mbs_x,
            mbs_y=mbs_y,
            code=code,
            path=path,
        )

    # ----- Fallback: try both readers
    try:
        df = _flatten_rl_csv(path)
        num_fbs = sum(1 for c in df.columns if c.endswith("_x"))
        return Trajectory(name=name or path.stem, source="rl", num_fbs=num_fbs,
                          df=df, path=path)
    except Exception:
        pass
    df = pd.read_csv(path)
    num_fbs = sum(1 for c in df.columns if c.startswith("fbs") and c.endswith("_x"))
    if num_fbs == 0:
        raise ValueError(f"Could not parse {path} as a trajectory.")
    return Trajectory(name=name or path.stem, source="ga", num_fbs=num_fbs,
                      df=df, path=path)


# --------------------------------------------------------------------------- #
# Plotter
# --------------------------------------------------------------------------- #
def _normalize_specs(specs) -> Sequence[Tuple[Union[str, Path], Optional[str]]]:
    norm = []
    for s in specs:
        if isinstance(s, (tuple, list)) and len(s) == 2:
            norm.append((s[0], s[1]))
        else:
            norm.append((s, None))
    return norm


# IEEE-leaning rcParams. Applied via mpl.rc_context so they don't pollute
# the caller's session.
IEEE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "0.25",
    "grid.linewidth": 0.4,
    "grid.color": "0.7",
    "lines.linewidth": 1.4,
    "patch.linewidth": 0.6,
    "figure.dpi": 200,
    "savefig.dpi": 400,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.fancybox": False,
    "legend.edgecolor": "0.5",
}

# Curated qualitative palette: print-friendly, distinguishable in greyscale.
IEEE_PALETTE = [
    "#1f4e79",   # deep blue
    "#c0392b",   # crimson
    "#2e7d32",   # forest green
    "#6a1b9a",   # violet
    "#d4801f",   # ochre
    "#5d4037",   # brown
    "#0097a7",   # teal
    "#7f7f7f",   # neutral gray
]

# Default per-source sequential colormaps. Each (trajectory, FBS) pair gets a
# distinct shade so overlapping paths remain traceable.
SOURCE_CMAPS = {
    "ga": "Blues",
    "rl": "Reds",
}


def _assign_track_colors(trajs, num_fbs, source_cmaps):
    """Map (traj_idx, fbs_idx) -> RGBA. Trajectories sharing a source split
    that source's colormap across a 'safe' middle-to-dark band so all shades
    remain visible on a white page."""
    from collections import defaultdict

    tracks_per_source = defaultdict(int)
    for t in trajs:
        tracks_per_source[t.source] += num_fbs

    counters = defaultdict(int)
    colors = {}
    LO, HI = 0.40, 0.90
    for ti, t in enumerate(trajs):
        cmap = plt.get_cmap(source_cmaps.get(t.source, "Greys"))
        total = tracks_per_source[t.source]
        for fi in range(num_fbs):
            idx = counters[t.source]
            frac = (LO + (HI - LO) * idx / (total - 1)) if total > 1 else 0.72
            colors[(ti, fi)] = cmap(frac)
            counters[t.source] += 1
    return colors


def plot_trajectories(
    specs: Iterable,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (5.0, 4.5),
    show_mbs: bool = True,
    color_by: str = "identity",
    mark_power_status: bool = False,
    annotate_endpoints: bool = True,
    style_preset: str = "ieee",
    palette: Optional[Sequence[str]] = None,
    source_cmaps: Optional[dict] = None,
    power_cmap: str = "viridis",
    legend_loc: str = "bottom",
    legend_ncol: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    color_by_step: Optional[bool] = None,   # deprecated, maps to color_by='step'
):
    """Plot one or more FBS trajectories on common axes (IEEE-styled by default).

    Each trajectory is drawn as a line in its own color (GA solid, RL dashed),
    with a clearly differentiated start/end marker:
      * Start = filled circle in the trajectory's color
      * End   = filled diamond in the trajectory's color (slightly larger)
    The legend explicitly labels Start / End / MBS so meaning is unambiguous.

    Parameters
    ----------
    specs : iterable of (path, label) tuples or bare paths
    title : optional figure title
    figsize : default (5.0, 4.0) — fits an IEEE single column comfortably
    show_mbs : draw MBS markers when known
    color_by : {'identity', 'power', 'step'}
        - 'identity' (default): one color per trajectory from `palette`
        - 'power': segments colored by transmit power; shared global colormap
          and colorbar in W. Endpoint glyphs adopt the power-coded color.
        - 'step': segments colored by step / generation index.
    mark_power_status : when True, overlay sparse markers along each path
        showing ON (filled) / OFF (hollow) status, plus matching legend entries.
    annotate_endpoints : draw the Start/End markers
    style_preset : 'ieee' applies the journal-style rcParams; None leaves
        matplotlib defaults alone
    palette : optional flat color list. When provided, each trajectory gets
        the next color in the list (legacy behavior). When None (default),
        colors are auto-shaded per source via `source_cmaps` instead.
    source_cmaps : dict mapping source ('ga', 'rl') to a sequential matplotlib
        colormap name. Each (trajectory, FBS) pair of that source draws a
        distinct shade from the colormap so overlapping paths stay
        traceable. Defaults to {'ga': 'Blues', 'rl': 'Reds'}. Ignored when
        `palette` is set.
    power_cmap : colormap name when color_by='power' (default 'viridis')
    legend_loc : 'bottom' (default; placed under the axes, frameless),
        'best' (auto-place inside the axes), or any matplotlib loc string.
    legend_ncol : number of legend columns; auto-computed if None.
    """
    # Back-compat shim
    if color_by_step is not None:
        color_by = "step" if color_by_step else "identity"
    if color_by not in ("identity", "power", "step"):
        raise ValueError(
            f"color_by must be 'identity', 'power', or 'step' (got {color_by!r})"
        )
    trajs = [load_trajectory(p, n) for p, n in _normalize_specs(specs)]
    if not trajs:
        raise ValueError("No trajectories provided.")

    nfbs_set = {t.num_fbs for t in trajs}
    if len(nfbs_set) > 1:
        raise ValueError(
            f"Refusing to plot mixed FBS counts in one figure: {sorted(nfbs_set)}. "
            "Group trajectories by num_fbs first."
        )
    num_fbs = nfbs_set.pop()

    rc_ctx = IEEE_RC if style_preset == "ieee" else {}

    with mpl.rc_context(rc_ctx):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # ---- MBS markers (deduplicated) -----------------------------------
        mbs_drawn = False
        if show_mbs:
            seen = set()
            for t in trajs:
                if t.mbs_x is None or t.mbs_y is None:
                    continue
                xs = np.atleast_1d(t.mbs_x).ravel()
                ys = np.atleast_1d(t.mbs_y).ravel()
                for i, (x, y) in enumerate(zip(xs, ys)):
                    key = (round(float(x), 2), round(float(y), 2))
                    if key in seen:
                        continue
                    seen.add(key)
                    ax.scatter(x, y, marker="s", s=70, color="black",
                               edgecolors="white", linewidths=0.9, zorder=6)
                    ax.annotate(f"MBS{i+1}" if len(xs) > 1 else "MBS",
                                (x, y), textcoords="offset points",
                                xytext=(7, 6), fontsize=7.5,
                                fontweight="bold", zorder=7)
                    mbs_drawn = True

        # ---- Trajectories -------------------------------------------------
        source_ls = {"ga": "-", "rl": "--"}
        step_cmap = plt.cm.viridis
        max_steps = max(len(t.df) for t in trajs)
        step_norm = mpl.colors.Normalize(vmin=0, vmax=max(max_steps - 1, 1))

        # Global power normalization — shared across every trajectory so the
        # colorbar is comparable across runs.
        power_norm = None
        power_cmap_obj = plt.get_cmap(power_cmap)
        if color_by == "power" or mark_power_status:
            all_p = []
            for t in trajs:
                for fi in range(num_fbs):
                    col = f"fbs{fi}_power"
                    if col in t.df.columns:
                        all_p.append(t.df[col].to_numpy(dtype=float))
            if all_p:
                cat = np.concatenate(all_p)
                cat = cat[np.isfinite(cat)]
                if cat.size:
                    pmin, pmax = float(cat.min()), float(cat.max())
                    if pmax - pmin < 1e-6:
                        pmax = pmin + 1.0
                    power_norm = mpl.colors.Normalize(vmin=pmin, vmax=pmax)

        # Resolve a color per (trajectory, FBS). When `palette` is supplied,
        # fall back to the legacy flat-list assignment (one color per traj,
        # shared across its FBSs).
        if palette is not None:
            track_colors = None
        else:
            scs = source_cmaps if source_cmaps is not None else SOURCE_CMAPS
            track_colors = _assign_track_colors(trajs, num_fbs, scs)

        traj_handles = []
        any_status_overlay = False

        for ti, t in enumerate(trajs):
            ls = source_ls.get(t.source, "-")

            for fi in range(num_fbs):
                if track_colors is not None:
                    traj_color = track_colors[(ti, fi)]
                else:
                    traj_color = palette[ti % len(palette)]

                x = t.df[f"fbs{fi}_x"].to_numpy()
                y = t.df[f"fbs{fi}_y"].to_numpy()
                label = t.name if num_fbs == 1 else f"{t.name} — FBS{fi+1}"

                pcol = f"fbs{fi}_power"
                scol = f"fbs{fi}_power_status"
                power = (t.df[pcol].to_numpy(dtype=float)
                         if pcol in t.df.columns else np.full(len(x), np.nan))
                status = (t.df[scol].to_numpy(dtype=float)
                          if scol in t.df.columns else np.ones(len(x)))

                # ---- main path
                # Line color always tracks trajectory identity so an FBS is
                # easy to follow. Power encoding lives on the nodes / endpoints
                # via the same colormap as the colorbar.
                if color_by == "step" and len(x) > 1:
                    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
                    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                    lc = LineCollection(segs, cmap=step_cmap, norm=step_norm,
                                        linewidths=1.5, alpha=0.9, linestyle=ls)
                    lc.set_array(np.arange(len(segs)))
                    ax.add_collection(lc)
                    h, = ax.plot([], [], ls, color=traj_color,
                                 linewidth=1.5, label=label)
                    traj_handles.append(h)
                else:  # 'identity' or 'power' — both draw a flat-color line
                    h, = ax.plot(x, y, ls, color=traj_color, linewidth=1.4,
                                 alpha=0.95, label=label, zorder=2,
                                 solid_capstyle="round", dash_capstyle="round")
                    traj_handles.append(h)

                # ---- endpoints (filled when ON, hollow when OFF)
                if annotate_endpoints and len(x) > 0:
                    if color_by == "power" and power_norm is not None and np.isfinite(power[0]):
                        start_color = power_cmap_obj(power_norm(power[0]))
                    else:
                        start_color = traj_color
                    if color_by == "power" and power_norm is not None and np.isfinite(power[-1]):
                        end_color = power_cmap_obj(power_norm(power[-1]))
                    else:
                        end_color = traj_color

                    start_on = bool(status[0] >= 0.5) if len(status) else True
                    end_on   = bool(status[-1] >= 0.5) if len(status) else True

                    # Start marker
                    if start_on:
                        ax.scatter(x[0], y[0], marker="o", s=55,
                                   facecolors=start_color, edgecolors="black",
                                   linewidths=0.8, zorder=4)
                    else:
                        ax.scatter(x[0], y[0], marker="o", s=60,
                                   facecolors="none", edgecolors=start_color,
                                   linewidths=1.7, zorder=4)
                        any_status_overlay = True

                    # End marker
                    if end_on:
                        ax.scatter(x[-1], y[-1], marker="D", s=80,
                                   facecolors=end_color, edgecolors="black",
                                   linewidths=0.9, zorder=4)
                    else:
                        ax.scatter(x[-1], y[-1], marker="D", s=88,
                                   facecolors="none", edgecolors=end_color,
                                   linewidths=1.8, zorder=4)
                        any_status_overlay = True

                # ---- ON/OFF status overlay (sparse along path)
                if mark_power_status and len(x) > 2:
                    stride = max(1, (len(x) - 2) // 12)  # ~12 markers per traj
                    idx = np.arange(1, len(x) - 1, stride)
                    if len(idx) == 0:
                        continue

                    if color_by == "power" and power_norm is not None:
                        marker_face = [power_cmap_obj(power_norm(p))
                                       if np.isfinite(p) else traj_color
                                       for p in power[idx]]
                    else:
                        marker_face = [traj_color] * len(idx)

                    on_mask = status[idx] >= 0.5
                    off_mask = ~on_mask

                    if on_mask.any():
                        ax.scatter(x[idx][on_mask], y[idx][on_mask],
                                   marker="o", s=18,
                                   facecolors=[marker_face[i] for i in np.where(on_mask)[0]],
                                   edgecolors="black", linewidths=0.45,
                                   alpha=0.9, zorder=3.5)
                        any_status_overlay = True
                    if off_mask.any():
                        ax.scatter(x[idx][off_mask], y[idx][off_mask],
                                   marker="o", s=18,
                                   facecolors="none",
                                   edgecolors="black", linewidths=0.7,
                                   alpha=0.9, zorder=3.5)
                        any_status_overlay = True

        # ---- Axes formatting ---------------------------------------------
        ax.set_xlabel(r"$x$ (m)")
        ax.set_ylabel(r"$y$ (m)")
        ax.grid(True, which="both", linestyle=":", alpha=0.55)
        ax.set_aspect("equal", adjustable="datalim")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(direction="in", length=3, width=0.7, top=False, right=False)

        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)
        if title:
            ax.set_title(title, pad=6)

        # ---- Optional colorbar (off when color_by='identity') ------------
        if color_by == "power" and power_norm is not None:
            sm = plt.cm.ScalarMappable(cmap=power_cmap_obj, norm=power_norm)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, fraction=0.038, pad=0.025, aspect=28)
            cb.set_label("Transmit Power (W)", fontsize=8)
            cb.ax.tick_params(labelsize=7)
            cb.outline.set_linewidth(0.6)
        elif color_by == "step":
            sources = {t.source for t in trajs}
            if sources == {"ga"}:
                cb_label = "Generation"
            elif sources == {"rl"}:
                cb_label = "Step"
            else:
                cb_label = "Time index"
            sm = plt.cm.ScalarMappable(cmap=step_cmap, norm=step_norm)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, fraction=0.038, pad=0.025, aspect=28)
            cb.set_label(cb_label, fontsize=8)
            cb.ax.tick_params(labelsize=7)
            cb.outline.set_linewidth(0.6)

        # ---- Legend: trajectories + status entries -----------------------
        from matplotlib.lines import Line2D

        status_handles = [
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor="0.4", markeredgecolor="black",
                   markersize=6.0, markeredgewidth=0.7,
                   label="Start"),
            Line2D([0], [0], marker="D", linestyle="None",
                   markerfacecolor="0.4", markeredgecolor="black",
                   markersize=6.5, markeredgewidth=0.7,
                   label="End"),
        ]
        if mbs_drawn:
            status_handles.append(
                Line2D([0], [0], marker="s", linestyle="None",
                       markerfacecolor="black", markeredgecolor="white",
                       markersize=6.5, markeredgewidth=0.7,
                       label="MBS")
            )
        # Show ON/OFF entries whenever any hollow status marker was drawn —
        # either mid-path (mark_power_status) or at an endpoint that ended OFF.
        if any_status_overlay:
            status_handles.extend([
                Line2D([0], [0], marker="o", linestyle="None",
                       markerfacecolor="0.4", markeredgecolor="black",
                       markersize=4.5, markeredgewidth=0.5,
                       label="ON (filled)"),
                Line2D([0], [0], marker="o", linestyle="None",
                       markerfacecolor="none", markeredgecolor="black",
                       markersize=4.5, markeredgewidth=0.7,
                       label="OFF (hollow)"),
            ])

        all_handles = traj_handles + status_handles
        all_labels = [h.get_label() for h in all_handles]
        n = len(all_handles)
        ncol = legend_ncol if legend_ncol is not None else (n if n <= 4 else 4)

        if legend_loc == "bottom":
            ax.legend(
                all_handles, all_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.13),
                ncol=ncol,
                frameon=False,
                fontsize=7.5,
                handlelength=2.2, handletextpad=0.6,
                columnspacing=1.4, borderpad=0.4,
            )
            # Reserve space below the axes so the legend isn't clipped.
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.18 + 0.04 * (max(ncol, 1) and (n // ncol)))
        else:
            ax.legend(
                all_handles, all_labels,
                loc=legend_loc, ncol=ncol,
                fontsize=7.5, handlelength=2.2, handletextpad=0.6,
                columnspacing=0.9, borderpad=0.4, borderaxespad=0.5,
            )
            fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+",
                   help="Trajectory paths or shortcuts (see module docstring).")
    p.add_argument("--label", action="append", default=None,
                   help="Display label; repeat once per path if given.")
    p.add_argument("--title", default=None)
    p.add_argument("--no-mbs", action="store_true")
    p.add_argument("--color-by", choices=("identity", "power", "step"),
                   default="identity",
                   help="Line color encoding (default: identity).")
    p.add_argument("--mark-status", action="store_true",
                   help="Overlay sparse ON/OFF markers along each path.")
    p.add_argument("--power-cmap", default="viridis",
                   help="Colormap when --color-by=power (default viridis).")
    p.add_argument("--no-ieee", action="store_true",
                   help="Skip the IEEE rcParams preset.")
    p.add_argument("--save", default=None, help="Save figure to this path.")
    args = p.parse_args()

    if args.label is not None and len(args.label) != len(args.paths):
        p.error("--label must be repeated once per path if provided.")

    specs = list(zip(args.paths, args.label or [None] * len(args.paths)))
    fig, _ = plot_trajectories(
        specs,
        title=args.title,
        show_mbs=not args.no_mbs,
        color_by=args.color_by,
        mark_power_status=args.mark_status,
        power_cmap=args.power_cmap,
        style_preset=None if args.no_ieee else "ieee",
    )
    if args.save:
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"Saved -> {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
