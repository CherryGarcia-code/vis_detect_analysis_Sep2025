"""Fig36: TF pulse response properties — latency, width, amplitude.

Loads pre-computed TF trace caches (NPZ). For every TF-responsive unit,
measures peak latency, half-width, and amplitude from the *fast_z* trace.

Produces fig36_tf_response_properties.png:
  - Panel A: Distribution of peak latency
  - Panel B: Distribution of half-width (FWHM)
  - Panel C: Latency vs. amplitude scatter by cell type
  - Panel D: Response property comparisons across learning stages
"""

import argparse
import os
import sys


import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import (
    STAGE_ORDER, STAGE_COLORS, CELLTYPE_COLORS, CACHE_DIR,
    DEFAULT_Z_THRESH_TF,
)
from visdetect.suite.loader import load_staging_manifest, load_waveform_labels, load_tf_traces_npz
from visdetect.suite.plotting import setup_style, save_figure

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

setup_style()

Z_THRESH = DEFAULT_Z_THRESH_TF  # 3.0


# ── Response property measurement ────────────────────────────────────────
def measure_response_properties(fast_z, t_vec):
    """Return peak_latency_ms, half_width_ms, amplitude for a single unit.

    Searches only the post-pulse window (t >= 0).  Returns NaNs if trace
    is flat or all-NaN.
    """
    dt_ms = (t_vec[1] - t_vec[0]) * 1000
    post_mask = t_vec >= 0
    tv_post = t_vec[post_mask] * 1000  # ms
    fz_post = fast_z[post_mask]

    if len(fz_post) == 0 or np.all(np.isnan(fz_post)):
        return np.nan, np.nan, np.nan

    pk_idx = np.nanargmax(np.abs(fz_post))
    amplitude = fz_post[pk_idx]
    peak_latency = tv_post[pk_idx]

    # Half-width (FWHM) — find where |trace| first exceeds half-peak, going both ways
    half_amp = np.abs(amplitude) / 2.0
    above = np.abs(fz_post) >= half_amp
    if np.any(above):
        first = np.argmax(above)
        last = len(above) - 1 - np.argmax(above[::-1])
        half_width = (last - first) * dt_ms
        half_width = max(half_width, dt_ms)
    else:
        half_width = np.nan

    return peak_latency, half_width, amplitude


def main():
    parser = argparse.ArgumentParser(description="TF pulse response properties")
    parser.add_argument("--n-workers", type=int, default=1, help="(unused)")
    args = parser.parse_args()

    print("=" * 70)
    print("[08b] TF Pulse Response Properties  [from NPZ cache]")
    print("=" * 70)

    manifest = load_staging_manifest(qc_only=True)
    print(f"  Sessions: {len(manifest)}")

    ct_lookup = {}
    try:
        wf = load_waveform_labels()
        for _, row in wf.iterrows():
            ct_lookup[(int(row["session_name"]), int(row["cluster_id"]))] = row["cell_type"]
    except (FileNotFoundError, KeyError):
        print("  Warning: cell-type labels not found")

    records = []
    session_args = [
        (int(row["session_name"]), row["stage"], row["session_idx"])
        for _, row in manifest.iterrows()
    ]
    iterator = tqdm(session_args, desc="Sessions") if tqdm else session_args

    for sname, stage, sidx in iterator:
        npz = load_tf_traces_npz(sname)
        if npz is None:
            continue

        t_vec = npz["t_vec"]
        cluster_ids = npz["cluster_ids"]
        for i, cid in enumerate(cluster_ids):
            cid = int(cid)
            z_abs = max(
                abs(npz["z_max_fast"][i]), abs(npz["z_min_fast"][i]),
                abs(npz["z_max_slow"][i]), abs(npz["z_min_slow"][i]),
            )
            if z_abs < Z_THRESH:
                continue

            lat, hw, amp = measure_response_properties(npz["fast_z"][i], t_vec)
            records.append({
                "session_name": sname, "cluster_id": cid,
                "stage": stage, "session_idx": sidx,
                "cell_type": ct_lookup.get((sname, cid), "Unknown"),
                "peak_latency_ms": lat,
                "half_width_ms": hw,
                "amplitude": amp,
                "z_abs_max": z_abs,
            })

    df = pd.DataFrame(records)
    print(f"  TF-responsive units with properties: {len(df)}")
    if len(df) == 0:
        print("  No responsive units found. Exiting.")
        return

    # Cache
    cache_path = os.path.join(CACHE_DIR, "tf_response_properties.csv")
    df.to_csv(cache_path, index=False)
    print(f"  Cached: {cache_path}")

    # ── Create figure ────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ── Panel A: Latency distribution ────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    lat_vals = df["peak_latency_ms"].dropna().values
    if len(lat_vals):
        ax_a.hist(lat_vals, bins=40, color="#5C6BC0", edgecolor="white",
                  linewidth=0.5, alpha=0.8)
        med = np.median(lat_vals)
        ax_a.axvline(med, color="#E53935", linewidth=1.5, linestyle="--",
                     label=f"Median={med:.0f} ms")
    ax_a.set_xlabel("Peak latency (ms)")
    ax_a.set_ylabel("Count")
    ax_a.set_title(f"A – Peak latency distribution (n={len(lat_vals)})")
    ax_a.legend(fontsize=8)

    # ── Panel B: Half-width distribution ─────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    hw_vals = df["half_width_ms"].dropna().values
    hw_vals = hw_vals[hw_vals < np.percentile(hw_vals, 99)] if len(hw_vals) > 20 else hw_vals
    if len(hw_vals):
        ax_b.hist(hw_vals, bins=40, color="#26A69A", edgecolor="white",
                  linewidth=0.5, alpha=0.8)
        med_hw = np.median(hw_vals)
        ax_b.axvline(med_hw, color="#E53935", linewidth=1.5, linestyle="--",
                     label=f"Median={med_hw:.0f} ms")
    ax_b.set_xlabel("Half-width (ms)")
    ax_b.set_ylabel("Count")
    ax_b.set_title(f"B – Response half-width (FWHM) (n={len(hw_vals)})")
    ax_b.legend(fontsize=8)

    # ── Panel C: Scatter – latency vs amplitude by cell type ─────
    ax_c = fig.add_subplot(gs[1, 0])
    filtered = df.dropna(subset=["peak_latency_ms", "amplitude"])
    cell_types = sorted([c for c in filtered["cell_type"].unique() if c != "Unknown"])
    for ct in cell_types:
        sub = filtered[filtered["cell_type"] == ct]
        ax_c.scatter(sub["peak_latency_ms"], sub["amplitude"],
                     s=12, alpha=0.45,
                     color=CELLTYPE_COLORS.get(ct, "#999"),
                     label=f"{ct} (n={len(sub)})")
    ax_c.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax_c.set_xlabel("Peak latency (ms)")
    ax_c.set_ylabel("Amplitude (z-score)")
    ax_c.set_title("C – Latency vs. amplitude by cell type")
    if cell_types:
        ax_c.legend(fontsize=7, loc="best")

    # ── Panel D: Properties across learning stages ───────────────
    ax_d = fig.add_subplot(gs[1, 1])
    stages = [s for s in STAGE_ORDER if s in df["stage"].values]
    width = 0.25
    x = np.arange(len(stages))
    metrics = [
        ("peak_latency_ms", "Median latency (ms)"),
        ("half_width_ms", "Median width (ms)"),
    ]
    for mi, (col, label) in enumerate(metrics):
        vals = [df[df["stage"] == s][col].median() for s in stages]
        q25 = [df[df["stage"] == s][col].quantile(0.25) for s in stages]
        q75 = [df[df["stage"] == s][col].quantile(0.75) for s in stages]
        err_lo = [v - q for v, q in zip(vals, q25)]
        err_hi = [q - v for v, q in zip(vals, q75)]
        ax_d.bar(x + mi * width, vals, width, yerr=[err_lo, err_hi],
                 label=label, alpha=0.7, capsize=3,
                 color=["#5C6BC0", "#26A69A"][mi])
    ax_d.set_xticks(x + width / 2)
    ax_d.set_xticklabels(stages)
    ax_d.set_ylabel("Value (ms)")
    ax_d.set_title("D – Response properties across learning stages")
    ax_d.legend(fontsize=8)
    if len(stages) >= 2:
        groups_lat = [df[df["stage"]==s]["peak_latency_ms"].dropna().values for s in stages]
        groups_lat = [g for g in groups_lat if len(g) > 0]
        if len(groups_lat) >= 2:
            try:
                _, p = kruskal(*groups_lat)
                ax_d.text(0.5, 0.95, f"Kruskal–Wallis (latency): p={p:.2e}",
                         transform=ax_d.transAxes, fontsize=8, ha="center", va="top")
            except Exception:
                pass

    fig.suptitle(
        "TF Pulse Response Properties\n(Measured from cached fast-pulse z-scored PSTHs)",
        fontsize=13, fontweight="bold", y=0.98)
    save_figure(fig, "fig36_tf_response_properties", "08_tf_pulse")
    print("\n  Saved fig36_tf_response_properties.png")

    # Summary statistics
    print("\n  Summary by stage:")
    for s in stages:
        sub = df[df["stage"] == s]
        print(f"    {s}: n={len(sub)}, "
              f"latency={sub['peak_latency_ms'].median():.0f}ms (IQR {sub['peak_latency_ms'].quantile(0.25):.0f}–{sub['peak_latency_ms'].quantile(0.75):.0f}), "
              f"width={sub['half_width_ms'].median():.0f}ms")


if __name__ == "__main__":
    main()
