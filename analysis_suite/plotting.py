"""Shared plotting utilities for the analysis suite.

Provides publication-quality matplotlib defaults, figure saving,
and common multi-panel helpers.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from config import FIGURE_DIR, STAGE_ORDER, STAGE_COLORS


def setup_style():
    """Apply publication-quality matplotlib rcParams."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# Apply on import
setup_style()


def save_figure(fig, name, module_name, formats=("png",)):
    """Save figure to figures/{module_name}/{name}.{fmt}."""
    out_dir = os.path.join(FIGURE_DIR, module_name)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for fmt in formats:
        path = os.path.join(out_dir, f"{name}.{fmt}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def add_stage_background(ax, manifest_df, alpha=0.08):
    """Add colored vertical spans for Naive/Learning/Expert stages.

    manifest_df must have 'session_idx' and 'stage' columns.
    """
    for stage in STAGE_ORDER:
        rows = manifest_df[manifest_df["stage"] == stage]
        if rows.empty:
            continue
        xmin = rows["session_idx"].min() - 0.5
        xmax = rows["session_idx"].max() + 0.5
        ax.axvspan(xmin, xmax, alpha=alpha, color=STAGE_COLORS[stage],
                   label=f"_{stage}")  # underscore hides from legend


def plot_significance_stars(ax, x, y, pval, height_offset=0.05):
    """Add significance stars above a point."""
    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
    ypos = y + height_offset * yrange
    if pval < 0.001:
        text = "***"
    elif pval < 0.01:
        text = "**"
    elif pval < 0.05:
        text = "*"
    else:
        return
    ax.text(x, ypos, text, ha="center", va="bottom", fontsize=12, fontweight="bold")


def multi_panel_figure(nrows, ncols, figsize=None, hspace=0.4, wspace=0.35):
    """Create a multi-panel figure with GridSpec."""
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows, ncols, hspace=hspace, wspace=wspace)
    return fig, gs


def stage_legend(ax, loc="upper left"):
    """Add a stage color legend to an axis."""
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=STAGE_COLORS[s], label=s) for s in STAGE_ORDER]
    ax.legend(handles=handles, loc=loc, framealpha=0.8, fontsize=8)
