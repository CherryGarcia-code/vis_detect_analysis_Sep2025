"""Fig 35c: TF detrended classification — re-classify TF responsiveness with detrending.

Applies linear detrending to existing NPZ z-scored traces (post-hoc, no
re-extraction needed) and produces a classification CSV with both standard
and detrended z-scores for all units across all sessions.

Outputs:
  - cache/tf_responsiveness_detrended.csv
  - figures/08_tf_pulse/fig35c_tf_detrended_classification.png
"""
import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import CACHE_DIR, STAGE_ORDER, STAGE_COLORS
from loader import load_staging_manifest, load_tf_traces_npz
from plotting import setup_style, save_figure
from visdetect.analysis.tf_pulse import detrend_tf_traces
from visdetect.analysis.constants import DEFAULT_Z_THRESH_TF

setup_style()

CACHE_FILE = os.path.join(CACHE_DIR, "tf_responsiveness_detrended.csv")

# Detrending parameters
DETREND_BASELINE = (-0.4, -0.01)  # seconds
DETREND_POST_WINDOW = (0.0, 0.3)  # seconds (conservative extrapolation)


def compute_or_load(force=False):
    if os.path.exists(CACHE_FILE) and not force:
        return pd.read_csv(CACHE_FILE)

    manifest = load_staging_manifest(qc_only=True)
    rows = []

    for _, mrow in manifest.iterrows():
        sname = str(mrow["session_name"])
        stage = mrow["stage"]
        npz = load_tf_traces_npz(sname)
        if npz is None:
            print(f"  {sname}: no NPZ, skip")
            continue

        cluster_ids = npz["cluster_ids"]
        n_units = len(cluster_ids)

        # Standard z-scores (from NPZ, no detrending)
        z_abs_max_std = np.maximum(
            np.maximum(np.abs(npz["z_max_fast"]), np.abs(npz["z_min_fast"])),
            np.maximum(np.abs(npz["z_max_slow"]), np.abs(npz["z_min_slow"])),
        )

        # Detrended z-scores
        _, z_max_fast_dt, z_min_fast_dt = detrend_tf_traces(
            npz["t_vec"], npz["fast_z"],
            baseline_window=DETREND_BASELINE,
            post_window=DETREND_POST_WINDOW,
        )
        _, z_max_slow_dt, z_min_slow_dt = detrend_tf_traces(
            npz["t_vec"], npz["slow_z"],
            baseline_window=DETREND_BASELINE,
            post_window=DETREND_POST_WINDOW,
        )
        z_abs_max_dt = np.maximum(
            np.maximum(np.abs(z_max_fast_dt), np.abs(z_min_fast_dt)),
            np.maximum(np.abs(z_max_slow_dt), np.abs(z_min_slow_dt)),
        )

        for u in range(n_units):
            rows.append({
                "session_name": int(sname),
                "cluster_id": int(cluster_ids[u]),
                "stage": stage,
                "z_abs_max_standard": float(z_abs_max_std[u]),
                "z_abs_max_detrended": float(z_abs_max_dt[u]),
                "is_tf_responsive_standard": bool(z_abs_max_std[u] >= DEFAULT_Z_THRESH_TF),
                "is_tf_responsive_detrended": bool(z_abs_max_dt[u] >= DEFAULT_Z_THRESH_TF),
                "z_max_fast_dt": float(z_max_fast_dt[u]),
                "z_min_fast_dt": float(z_min_fast_dt[u]),
                "z_max_slow_dt": float(z_max_slow_dt[u]),
                "z_min_slow_dt": float(z_min_slow_dt[u]),
            })

        n_std = (z_abs_max_std >= DEFAULT_Z_THRESH_TF).sum() if n_units else 0
        n_dt = (z_abs_max_dt >= DEFAULT_Z_THRESH_TF).sum() if n_units else 0
        print(f"  {sname} ({stage}): {n_units} units, "
              f"standard={n_std} ({100*n_std/n_units:.0f}%), "
              f"detrended={n_dt} ({100*n_dt/n_units:.0f}%)")

    df = pd.DataFrame(rows)
    df.to_csv(CACHE_FILE, index=False)
    print(f"\n  Saved {len(df)} rows to {CACHE_FILE}")
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print("[08j] TF detrended classification...")
    df = compute_or_load(force=args.force)

    n_total = len(df)
    n_std = df["is_tf_responsive_standard"].sum()
    n_dt = df["is_tf_responsive_detrended"].sum()
    print(f"\n  Overall: {n_total} units")
    print(f"  Standard responsive: {n_std} ({100*n_std/n_total:.1f}%)")
    print(f"  Detrended responsive: {n_dt} ({100*n_dt/n_total:.1f}%)")

    # ── Figure: 4 panels ─────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Scatter — standard vs detrended z_abs_max
    ax_a = fig.add_subplot(gs[0, 0])
    from scipy.stats import spearmanr
    rho, pval = spearmanr(df["z_abs_max_standard"], df["z_abs_max_detrended"])
    ax_a.scatter(df["z_abs_max_standard"], df["z_abs_max_detrended"],
                 s=3, alpha=0.3, color="#4e79a7", rasterized=True)
    ax_a.axhline(DEFAULT_Z_THRESH_TF, color="red", ls="--", lw=0.8, alpha=0.6)
    ax_a.axvline(DEFAULT_Z_THRESH_TF, color="red", ls="--", lw=0.8, alpha=0.6)
    lim = max(df["z_abs_max_standard"].max(), df["z_abs_max_detrended"].max()) * 1.05
    ax_a.plot([0, lim], [0, lim], "k--", lw=0.5, alpha=0.4)
    ax_a.set_xlabel("Standard z_abs_max")
    ax_a.set_ylabel("Detrended z_abs_max")
    ax_a.set_title(f"A. Standard vs Detrended (rho={rho:.3f}, p={pval:.1e})")

    # Panel B: Histogram overlay
    ax_b = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, 15, 60)
    ax_b.hist(df["z_abs_max_standard"], bins=bins, alpha=0.5,
              color="#4e79a7", label=f"Standard ({n_std}/{n_total})")
    ax_b.hist(df["z_abs_max_detrended"], bins=bins, alpha=0.5,
              color="#e15759", label=f"Detrended ({n_dt}/{n_total})")
    ax_b.axvline(DEFAULT_Z_THRESH_TF, color="k", ls="--", lw=1)
    ax_b.set_xlabel("|z| max")
    ax_b.set_ylabel("Count")
    ax_b.set_title("B. Distribution of peak |z|")
    ax_b.legend(fontsize=9)

    # Panel C: Per-stage responsive fractions
    ax_c = fig.add_subplot(gs[1, 0])
    stage_data = []
    for stage in STAGE_ORDER:
        sub = df[df["stage"] == stage]
        if len(sub) == 0:
            continue
        stage_data.append({
            "stage": stage,
            "n": len(sub),
            "frac_std": sub["is_tf_responsive_standard"].mean(),
            "frac_dt": sub["is_tf_responsive_detrended"].mean(),
        })
    sd = pd.DataFrame(stage_data)
    x = np.arange(len(sd))
    w = 0.35
    ax_c.bar(x - w/2, sd["frac_std"] * 100, w, label="Standard",
             color=[STAGE_COLORS.get(s, "gray") for s in sd["stage"]], alpha=0.6)
    ax_c.bar(x + w/2, sd["frac_dt"] * 100, w, label="Detrended",
             color=[STAGE_COLORS.get(s, "gray") for s in sd["stage"]], alpha=1.0)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([f"{r['stage']}\n(n={r['n']})" for _, r in sd.iterrows()])
    ax_c.set_ylabel("% TF-responsive")
    ax_c.set_title("C. Responsive fraction by stage")
    ax_c.legend(fontsize=9)

    # Panel D: Contingency — neurons gained by detrending
    ax_d = fig.add_subplot(gs[1, 1])
    both = (df["is_tf_responsive_standard"] & df["is_tf_responsive_detrended"]).sum()
    std_only = (df["is_tf_responsive_standard"] & ~df["is_tf_responsive_detrended"]).sum()
    dt_only = (~df["is_tf_responsive_standard"] & df["is_tf_responsive_detrended"]).sum()
    neither = (~df["is_tf_responsive_standard"] & ~df["is_tf_responsive_detrended"]).sum()

    categories = ["Both", "Standard only", "Detrended only", "Neither"]
    counts = [both, std_only, dt_only, neither]
    colors = ["#59a14f", "#4e79a7", "#e15759", "#bab0ac"]
    bars = ax_d.bar(categories, counts, color=colors)
    for bar, c in zip(bars, counts):
        ax_d.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                  str(c), ha="center", va="bottom", fontsize=9)
    ax_d.set_ylabel("Unit count")
    ax_d.set_title("D. Classification agreement")

    # ── Save ─────────────────────────────────────────────────────────
    save_figure(fig, "fig35c_tf_detrended_classification", "08_tf_pulse")

    # Stats CSV
    stats = pd.DataFrame([{
        "n_total": n_total,
        "n_standard_responsive": int(n_std),
        "pct_standard": 100 * n_std / n_total,
        "n_detrended_responsive": int(n_dt),
        "pct_detrended": 100 * n_dt / n_total,
        "spearman_rho": rho,
        "spearman_p": pval,
        "n_both": both,
        "n_std_only": std_only,
        "n_dt_only": dt_only,
        "n_neither": neither,
        "detrend_baseline_window": str(DETREND_BASELINE),
        "detrend_post_window": str(DETREND_POST_WINDOW),
    }])
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "08_tf_pulse", "tf_detrended_classification_stats.csv"
    )
    stats.to_csv(stats_path, index=False)
    print(f"\n  Saved figure and stats")


if __name__ == "__main__":
    main()
