"""08c – TF pulse temporal integration: IPI-conditioned PSTHs.

This script REQUIRES session pickles because it needs raw spike times
and raw baseline TF pulse event times to build inter-pulse-interval
(IPI) conditioned PSTHs.  The NPZ cache alone is insufficient.

Uses `_collect_pulses()` from `visdetect.analysis.tf_pulse` to extract
fast/slow pulse times per session, then aligns spikes to those pulses
conditioned on IPI (short vs. long).

Produces fig26_tf_pulse_integration.png:
  - Panel A: Example IPI-conditioned PSTHs (short vs. long IPI)
  - Panel B: Population mean PSTHs by IPI tertile
  - Panel C: Integration index (short-IPI amplitude / long-IPI amplitude) distribution
  - Panel D: Integration index by cell type and learning stage
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu, kruskal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    STAGE_ORDER, STAGE_COLORS, CELLTYPE_COLORS, CACHE_DIR,
    DEFAULT_Z_THRESH_TF,
)
from loader import (
    load_staging_manifest, load_waveform_labels, load_tf_traces_npz,
    session_iterator,
)
from plotting import setup_style, save_figure

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

setup_style()

Z_THRESH = DEFAULT_Z_THRESH_TF

# ── Helper: fast pulse-aligned traces (simplified) ───────────────────
DT = 0.001          # 1 ms bins
SIGMA_MS = 17.0     # Gaussian smoothing
PRE_WIN = (-0.4, 0.0)
POST_WIN = (0.0, 0.5)


def _smooth(rel_times, t_vec, sigma_bins):
    """Bin relative spike times and smooth."""
    train = np.zeros_like(t_vec)
    if rel_times.size == 0:
        return train
    idx = np.searchsorted(t_vec, rel_times)
    idx = idx[(idx >= 0) & (idx < train.size)]
    train[idx] = 1.0
    return gaussian_filter1d(train, sigma=sigma_bins)


def _zscore(trace, t_vec, pre_win):
    pre_mask = (t_vec >= pre_win[0]) & (t_vec < pre_win[1])
    mu = np.nanmean(trace[pre_mask]) if np.any(pre_mask) else 0.0
    sd = np.nanstd(trace[pre_mask]) if np.any(pre_mask) else 0.0
    if not np.isfinite(sd) or sd <= 0:
        return trace * 0.0
    return (trace - mu) / sd


def _vectorized_psth(spike_times, pulses, t_vec, sigma_bins):
    """Fast vectorized pulse-triggered histogram (no per-pulse loop)."""
    if pulses.size == 0:
        return np.zeros_like(t_vec)
    full0, full1 = t_vec[0], t_vec[-1] + (t_vec[1] - t_vec[0])
    # For each pulse, find spikes in window using searchsorted
    # Stack all relative spike times
    all_rel = []
    for tp in pulses:
        lo = np.searchsorted(spike_times, tp + full0)
        hi = np.searchsorted(spike_times, tp + full1)
        if hi > lo:
            all_rel.append(spike_times[lo:hi] - tp)
    if not all_rel:
        return np.zeros_like(t_vec)
    all_rel = np.concatenate(all_rel)
    # Histogram into t_vec bins
    dt = t_vec[1] - t_vec[0]
    counts, _ = np.histogram(all_rel, bins=np.append(t_vec, t_vec[-1] + dt))
    # Average per pulse: counts / n_pulses, then smooth
    rate = counts.astype(float) / pulses.size
    return gaussian_filter1d(rate, sigma=sigma_bins)


def _compute_ipi_traces(spike_times, fast_times, t_vec, sigma_bins):
    """Compute z-scored PSTHs for short-IPI and long-IPI pulses (vectorized)."""
    if fast_times.size < 3:
        return None, None
    # Ensure sorted for searchsorted
    spike_times = np.sort(spike_times)
    fast_times = np.sort(fast_times)

    ipis = np.diff(fast_times)
    med_ipi = np.median(ipis)
    short_mask = np.concatenate([[False], ipis <= med_ipi])
    long_mask = np.concatenate([[False], ipis > med_ipi])

    short_pulses = fast_times[short_mask]
    long_pulses = fast_times[long_mask]

    if short_pulses.size < 3 or long_pulses.size < 3:
        return None, None

    mean_short = _zscore(_vectorized_psth(spike_times, short_pulses, t_vec, sigma_bins), t_vec, PRE_WIN)
    mean_long = _zscore(_vectorized_psth(spike_times, long_pulses, t_vec, sigma_bins), t_vec, PRE_WIN)
    return mean_short, mean_long


def main():
    parser = argparse.ArgumentParser(description="TF pulse temporal integration")
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()

    print("=" * 70)
    print("[08c] TF Pulse Temporal Integration (IPI)  [requires session pkls]")
    print("=" * 70)

    manifest = load_staging_manifest(qc_only=True)
    print(f"  Sessions: {len(manifest)}")

    # Get TF-responsive unit set from NPZ cache
    responsive_set = set()
    for _, row in manifest.iterrows():
        sname = int(row["session_name"])
        npz = load_tf_traces_npz(sname)
        if npz is None:
            continue
        for i, cid in enumerate(npz["cluster_ids"]):
            z_abs = max(
                abs(npz["z_max_fast"][i]), abs(npz["z_min_fast"][i]),
                abs(npz["z_max_slow"][i]), abs(npz["z_min_slow"][i]),
            )
            if z_abs >= Z_THRESH:
                responsive_set.add((sname, int(cid)))
    print(f"  TF-responsive units (from NPZ): {len(responsive_set)}")

    ct_lookup = {}
    try:
        wf = load_waveform_labels()
        for _, r in wf.iterrows():
            ct_lookup[(int(r["session_name"]), int(r["cluster_id"]))] = r["cell_type"]
    except Exception:
        pass

    # ── Session loop (needs pkl) ──────────────────────────────────────
    from visdetect.analysis.tf_pulse import _collect_pulses, TFRespPulseConfig

    t_vec = np.arange(PRE_WIN[0], POST_WIN[1], DT, dtype=float)
    sigma_bins = (SIGMA_MS / 1000.0) / DT

    records = []
    example_traces = []  # (z_abs, short_z, long_z, t_vec, sname, cid)
    pop_short_all, pop_long_all = [], []  # for population average

    cfg = TFRespPulseConfig()

    # Build session_idx lookup from manifest
    sidx_lookup = {int(r["session_name"]): r["session_idx"] for _, r in manifest.iterrows()}
    stage_lookup = {int(r["session_name"]): r["stage"] for _, r in manifest.iterrows()}

    for sname_int, stage, session in session_iterator():
        sidx = sidx_lookup.get(sname_int, -1)
        fast_times, slow_times = _collect_pulses(session, cfg, show_progress=False)
        if fast_times.size < 10:
            print(f"    {sname_int}: only {fast_times.size} fast pulses – skip")
            continue
        print(f"    {sname_int}: {fast_times.size} fast pulses, scanning units…")

        # Only process TF-responsive units
        for c in session.clusters:
            cid = int(c.cluster_id)
            if (sname_int, cid) not in responsive_set:
                continue
            st = np.asarray(c.spike_times, dtype=float).flatten()
            if st.size == 0:
                continue

            short_z, long_z = _compute_ipi_traces(st, fast_times, t_vec, sigma_bins)
            if short_z is None:
                continue

            # peak amplitude in post window
            post_mask = (t_vec >= POST_WIN[0]) & (t_vec < POST_WIN[1])
            amp_short = float(np.nanmax(np.abs(short_z[post_mask])))
            amp_long = float(np.nanmax(np.abs(long_z[post_mask])))
            integ_idx = amp_short / amp_long if amp_long > 0 else np.nan

            # Get z_abs from NPZ for sorting examples
            npz = load_tf_traces_npz(sname_int)
            z_abs = 0.0
            if npz is not None:
                idx_match = np.where(npz["cluster_ids"] == cid)[0]
                if len(idx_match):
                    j = idx_match[0]
                    z_abs = max(abs(npz["z_max_fast"][j]), abs(npz["z_min_fast"][j]),
                                abs(npz["z_max_slow"][j]), abs(npz["z_min_slow"][j]))

            records.append({
                "session_name": sname_int, "cluster_id": cid,
                "stage": stage, "session_idx": sidx,
                "cell_type": ct_lookup.get((sname_int, cid), "Unknown"),
                "amp_short_ipi": amp_short,
                "amp_long_ipi": amp_long,
                "integration_index": integ_idx,
            })
            example_traces.append((z_abs, short_z.copy(), long_z.copy(), cid, sname_int, stage))
            pop_short_all.append(short_z.copy())
            pop_long_all.append(long_z.copy())

    df = pd.DataFrame(records)
    print(f"\n  Units with IPI data: {len(df)}")

    if len(df) == 0:
        print("  No data. Exiting.")
        return

    cache_path = os.path.join(CACHE_DIR, "tf_ipi_integration.csv")
    df.to_csv(cache_path, index=False)
    print(f"  Cached: {cache_path}")

    # Sort examples by z_abs
    example_traces.sort(key=lambda x: x[0], reverse=True)

    # ── Create figure ────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ── Panel A: Example IPI-conditioned PSTHs ───────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    n_ex = min(3, len(example_traces))
    pal_short = ["#E53935", "#C62828", "#880E4F"]
    pal_long = ["#1565C0", "#0D47A1", "#01579B"]
    for i in range(n_ex):
        zv, sz, lz, ci, sn, st = example_traces[i]
        ax_a.plot(t_vec * 1000, sz, color=pal_short[i], linewidth=1.5,
                  label=f"Short IPI #{ci}" if i == 0 else None)
        ax_a.plot(t_vec * 1000, lz, color=pal_long[i], linewidth=1.2,
                  linestyle="--",
                  label=f"Long IPI #{ci}" if i == 0 else None)
    ax_a.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_a.set_xlabel("Time from fast TF pulse (ms)")
    ax_a.set_ylabel("Z-score")
    ax_a.set_title("A – Example IPI-conditioned PSTHs")
    ax_a.legend(fontsize=7)

    # ── Panel B: Population mean ─────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    if pop_short_all:
        mn_s = np.nanmean(np.stack(pop_short_all), axis=0)
        se_s = np.nanstd(np.stack(pop_short_all), axis=0) / np.sqrt(len(pop_short_all))
        mn_l = np.nanmean(np.stack(pop_long_all), axis=0)
        se_l = np.nanstd(np.stack(pop_long_all), axis=0) / np.sqrt(len(pop_long_all))
        ax_b.plot(t_vec * 1000, mn_s, color="#E53935", linewidth=1.5, label="Short IPI")
        ax_b.fill_between(t_vec * 1000, mn_s - se_s, mn_s + se_s,
                          color="#E53935", alpha=0.15)
        ax_b.plot(t_vec * 1000, mn_l, color="#1565C0", linewidth=1.5, label="Long IPI")
        ax_b.fill_between(t_vec * 1000, mn_l - se_l, mn_l + se_l,
                          color="#1565C0", alpha=0.15)
    ax_b.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_b.set_xlabel("Time from fast TF pulse (ms)")
    ax_b.set_ylabel("Mean z-score")
    ax_b.set_title(f"B – Population mean (n={len(pop_short_all)})")
    ax_b.legend(fontsize=8)

    # ── Panel C: Integration index histogram ─────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    ii_vals = df["integration_index"].dropna().values
    ii_vals = ii_vals[np.isfinite(ii_vals) & (ii_vals < 10)]  # remove extreme
    if len(ii_vals):
        ax_c.hist(ii_vals, bins=40, color="#7E57C2", edgecolor="white",
                  linewidth=0.5, alpha=0.8)
        med = np.median(ii_vals)
        ax_c.axvline(1.0, color="grey", linewidth=1, linestyle=":", label="No integration")
        ax_c.axvline(med, color="#E53935", linewidth=1.5, linestyle="--",
                     label=f"Median={med:.2f}")
    ax_c.set_xlabel("Integration index (short/long IPI amplitude)")
    ax_c.set_ylabel("Count")
    ax_c.set_title("C – Integration index distribution")
    ax_c.legend(fontsize=8)

    # ── Panel D: By stage and cell type ──────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    stages = [s for s in STAGE_ORDER if s in df["stage"].values]
    x = np.arange(len(stages))
    vals = [df[df["stage"]==s]["integration_index"].median() for s in stages]
    q25 = [df[df["stage"]==s]["integration_index"].quantile(0.25) for s in stages]
    q75 = [df[df["stage"]==s]["integration_index"].quantile(0.75) for s in stages]
    err_lo = [v - q for v, q in zip(vals, q25)]
    err_hi = [q - v for v, q in zip(vals, q75)]
    ax_d.bar(x - 0.15, vals, 0.3, yerr=[err_lo, err_hi],
             color=[STAGE_COLORS[s] for s in stages], edgecolor="black",
             linewidth=0.5, capsize=3)
    cell_types = sorted([c for c in df["cell_type"].unique() if c != "Unknown"])
    if len(cell_types) >= 2:
        for ci, ct in enumerate(cell_types):
            ct_vals = [df[(df["stage"]==s)&(df["cell_type"]==ct)]["integration_index"].median()
                       for s in stages]
            ax_d.scatter(x + 0.2 + ci*0.1, ct_vals, marker="^" if ci else "o",
                        s=60, color=CELLTYPE_COLORS.get(ct,"#999"),
                        edgecolors="black", linewidth=0.5, label=ct, zorder=5)
    ax_d.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(stages)
    ax_d.set_ylabel("Integration index (median)")
    ax_d.set_title("D – Integration by stage & cell type")
    if cell_types:
        ax_d.legend(fontsize=7)

    fig.suptitle(
        "TF Pulse Temporal Integration\n"
        "(IPI = inter-pulse interval; fast pulses only)",
        fontsize=13, fontweight="bold", y=0.98)
    save_figure(fig, "fig26_tf_pulse_integration", "08_tf_pulse")
    print("\n  ✓ Saved fig26_tf_pulse_integration.png")


if __name__ == "__main__":
    main()
