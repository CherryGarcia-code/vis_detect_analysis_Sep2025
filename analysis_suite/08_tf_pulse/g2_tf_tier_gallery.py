"""Fig41g: Gallery of ALL TF-responsive units per tier.

Produces paginated grid figures (up to 50 cells per page) so every
classified unit can be visually inspected.  Each subplot shows the
fast (blue) and slow (red) z-scored traces ± SEM for one unit.

Figures are named:
  fig41g_tier1_splitter_page1.png  (page2, page3, ...)
  fig41g_tier2_unilateral_page1.png
  fig41g_tier3_omni_page1.png

Reads from:
  cache/tf_cell_classification.csv   (tier assignments)
  data/cache/tf_traces/BG_046/*.npz  (per-unit z-scored traces)
"""

import math
import os
import sys


import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import CACHE_DIR
from visdetect.suite.loader import load_staging_manifest, load_tf_traces_npz
from visdetect.suite.plotting import setup_style, save_figure

setup_style()

MODULE_NAME = "08_tf_pulse"

# Tier labels (must match g_tf_cell_classifier.py)
TIER_SPLITTER = "Tier 1 (Splitter)"
TIER_UNILATERAL = "Tier 2 (Unilateral)"
TIER_OMNI = "Tier 3 (Omni)"

TIER_META = {
    TIER_SPLITTER:   {"short": "tier1_splitter",   "color": "#8E24AA"},
    TIER_UNILATERAL: {"short": "tier2_unilateral", "color": "#FB8C00"},
    TIER_OMNI:       {"short": "tier3_omni",       "color": "#43A047"},
}

# Grid layout
COLS = 10
ROWS = 5
UNITS_PER_PAGE = COLS * ROWS  # 50

# Smoothing
SIGMA_SMOOTH = 5


def _load_all_npz(manifest):
    """Load all NPZ trace caches into a dict keyed by session_name (int)."""
    npz_traces = {}
    for _, row in manifest.iterrows():
        sname = int(row["session_name"])
        npz = load_tf_traces_npz(sname)
        if npz is not None:
            npz_traces[sname] = npz
    return npz_traces


def _gather_tier_units(df, tier_name, npz_traces):
    """Return sorted list of (session_name, cluster_id, npz_idx, effect_size, sub_type, trend_ratio)
    for all units in the given tier, sorted by descending post-pulse |z|."""
    tier_df = df[df["tier"] == tier_name]
    units = []
    for _, row in tier_df.iterrows():
        sname = int(row["session_name"])
        cid = int(row["cluster_id"])
        npz = npz_traces.get(sname)
        if npz is None:
            continue
        cids = list(npz["cluster_ids"].astype(int))
        if cid not in cids:
            continue
        idx = cids.index(cid)
        fz = npz["fast_z"][idx]
        sz = npz["slow_z"][idx]
        tv = npz["t_vec"]
        post = tv >= 0
        eff = max(np.nanmax(np.abs(fz[post])), np.nanmax(np.abs(sz[post])))
        tr = row.get("trend_ratio", np.nan)
        units.append((sname, cid, idx, float(eff), row.get("sub_type", ""), float(tr) if pd.notna(tr) else 0.0))
    # Sort by effect size descending
    units.sort(key=lambda x: x[3], reverse=True)
    return units


def _plot_unit(ax, npz, unit_idx, sname, cid, sub_type, trend_ratio=0.0):
    """Draw fast/slow traces ± SEM for a single unit, baseline-corrected."""
    tv_ms = npz["t_vec"] * 1000
    fz = gaussian_filter1d(npz["fast_z"][unit_idx], sigma=SIGMA_SMOOTH, mode="nearest")
    sz = gaussian_filter1d(npz["slow_z"][unit_idx], sigma=SIGMA_SMOOTH, mode="nearest")

    # Baseline correction: subtract pre-window mean (safe interior -300 to -50 ms)
    safe_mask = (tv_ms >= -300) & (tv_ms < -50)
    fz = fz - np.nanmean(fz[safe_mask])
    sz = sz - np.nanmean(sz[safe_mask])

    # SEM bands
    if "fast_z_sem" in npz and "slow_z_sem" in npz:
        f_sem = gaussian_filter1d(npz["fast_z_sem"][unit_idx], sigma=SIGMA_SMOOTH, mode="nearest")
        s_sem = gaussian_filter1d(npz["slow_z_sem"][unit_idx], sigma=SIGMA_SMOOTH, mode="nearest")
        ax.fill_between(tv_ms, fz - f_sem, fz + f_sem,
                        color="#1565C0", alpha=0.15, linewidth=0)
        ax.fill_between(tv_ms, sz - s_sem, sz + s_sem,
                        color="#E53935", alpha=0.15, linewidth=0)

    ax.plot(tv_ms, fz, color="#1565C0", linewidth=1.0, label="Fast")
    ax.plot(tv_ms, sz, color="#E53935", linewidth=1.0, label="Slow")
    ax.axvline(0, color="k", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axhline(0, color="grey", linewidth=0.3, alpha=0.3)
    ax.set_xlim(-300, tv_ms[-1])
    title_str = f"s{sname} c{cid}\n{sub_type}"
    ax.set_title(title_str, fontsize=6, pad=2)
    ax.tick_params(labelsize=5)


def _make_gallery_pages(units, tier_name, npz_traces):
    """Create paginated gallery figures for one tier.

    Returns list of (fig, figure_name) tuples.
    """
    meta = TIER_META[tier_name]
    n_units = len(units)
    n_pages = max(1, math.ceil(n_units / UNITS_PER_PAGE))

    figures = []
    for page in range(n_pages):
        start = page * UNITS_PER_PAGE
        end = min(start + UNITS_PER_PAGE, n_units)
        page_units = units[start:end]
        n_on_page = len(page_units)

        # Compute actual rows needed for this page
        n_rows = max(1, math.ceil(n_on_page / COLS))

        fig = plt.figure(figsize=(COLS * 3, n_rows * 2.5))
        gs = gridspec.GridSpec(n_rows, COLS, figure=fig,
                               hspace=0.55, wspace=0.30,
                               top=0.92, bottom=0.04, left=0.03, right=0.98)

        for i, (sname, cid, npz_idx, eff, sub_type, trend_ratio) in enumerate(page_units):
            r, c = divmod(i, COLS)
            ax = fig.add_subplot(gs[r, c])
            npz = npz_traces[sname]
            _plot_unit(ax, npz, npz_idx, sname, cid, sub_type, trend_ratio)

            # Only add axis labels on edges
            if r == n_rows - 1:
                ax.set_xlabel("ms", fontsize=5)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel("z", fontsize=5)
            else:
                ax.set_yticklabels([])

        page_label = f"Page {page + 1}/{n_pages}" if n_pages > 1 else ""
        fig.suptitle(
            f"{tier_name}  —  All units (n={n_units})  "
            f"sorted by effect size (descending)  {page_label}",
            fontsize=12, fontweight="bold", color=meta["color"],
        )

        fig_name = f"fig41g_{meta['short']}_page{page + 1}"
        figures.append((fig, fig_name))

    return figures


def main():
    print("=" * 70)
    print("[08g2] TF Tier Gallery — All responsive units per tier")
    print("=" * 70)

    # Load classification CSV
    csv_path = os.path.join(CACHE_DIR, "tf_cell_classification.csv")
    if not os.path.exists(csv_path):
        print(f"  ERROR: {csv_path} not found. Run g_tf_cell_classifier.py first.")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows from classification CSV")

    # Load NPZ caches
    manifest = load_staging_manifest(qc_only=True)
    print(f"  Loading NPZ caches for {len(manifest)} sessions ...")
    npz_traces = _load_all_npz(manifest)
    print(f"  Loaded {len(npz_traces)} NPZ caches")

    # Generate gallery for each tier
    for tier_name in [TIER_SPLITTER, TIER_UNILATERAL, TIER_OMNI]:
        units = _gather_tier_units(df, tier_name, npz_traces)
        n_pages = max(1, math.ceil(len(units) / UNITS_PER_PAGE))
        print(f"\n  {tier_name}: {len(units)} units → {n_pages} page(s)")

        pages = _make_gallery_pages(units, tier_name, npz_traces)
        for fig, fig_name in pages:
            save_figure(fig, fig_name, MODULE_NAME)
            print(f"    Saved {fig_name}.png")
            plt.close(fig)

    print("\n[08g2] Done.")


if __name__ == "__main__":
    main()
