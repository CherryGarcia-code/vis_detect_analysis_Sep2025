"""08f – TF sensory vs. motor: compare TF-responsive units to change/lick signals.

Uses NPZ cache for TF classification (eliminates collect_tf_pulse_traces()),
but REQUIRES session pickles for change-aligned and lick-aligned PSTHs.

Cross-references TF responsiveness with change-detection signals and
pre-lick ramping to test whether TF-responsive neurons carry sensory
and/or motor information.

Produces fig29_tf_sensory_motor.png:
  - Panel A: Change PSTH of TF-responsive vs. non-responsive units
  - Panel B: Lick PSTH of TF-responsive vs. non-responsive units
  - Panel C: TF responsiveness vs. change selectivity (scatter)
  - Panel D: Venn / overlap summary: TF × change × lick responsive
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu

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

# ── PSTH parameters ──────────────────────────────────────────────────
DT = 0.001
SIGMA_MS = 20.0
CHANGE_WIN = (-0.3, 0.5)
LICK_WIN = (-0.5, 0.3)


def _smooth(rel_times, t_vec, sigma_bins):
    train = np.zeros_like(t_vec)
    if rel_times.size == 0:
        return train
    idx = np.searchsorted(t_vec, rel_times)
    idx = idx[(idx >= 0) & (idx < train.size)]
    train[idx] = 1.0
    return gaussian_filter1d(train, sigma=sigma_bins)


def _zscore(trace, t_vec, pre_end=0.0):
    pre_mask = t_vec < pre_end
    mu = np.nanmean(trace[pre_mask]) if np.any(pre_mask) else 0.0
    sd = np.nanstd(trace[pre_mask]) if np.any(pre_mask) else 0.0
    if not np.isfinite(sd) or sd <= 0:
        return trace * 0.0
    return (trace - mu) / sd


def _event_psth(spike_times, event_times, window, dt, sigma_ms):
    """Compute z-scored PSTH aligned to event times (vectorized)."""
    t_vec = np.arange(window[0], window[1], dt, dtype=float)
    sigma_bins = (sigma_ms / 1000.0) / dt
    event_times = event_times[np.isfinite(event_times)]
    if event_times.size == 0:
        return np.full_like(t_vec, np.nan), t_vec
    # Sort spike times for efficient searchsorted
    spike_sorted = np.sort(spike_times)
    full0, full1 = t_vec[0], t_vec[-1] + dt
    all_rel = []
    for ev in event_times:
        lo = np.searchsorted(spike_sorted, ev + full0)
        hi = np.searchsorted(spike_sorted, ev + full1)
        if hi > lo:
            all_rel.append(spike_sorted[lo:hi] - ev)
    if not all_rel:
        return np.full_like(t_vec, np.nan), t_vec
    all_rel = np.concatenate(all_rel)
    counts, _ = np.histogram(all_rel, bins=np.append(t_vec, t_vec[-1] + dt))
    rate = counts.astype(float) / event_times.size
    mean_tr = gaussian_filter1d(rate, sigma=sigma_bins)
    z = _zscore(mean_tr, t_vec, pre_end=0.0)
    return z, t_vec


def main():
    parser = argparse.ArgumentParser(description="TF sensory vs. motor")
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()

    print("=" * 70)
    print("[08f] TF Sensory vs. Motor  [NPZ + session pkls]")
    print("=" * 70)

    manifest = load_staging_manifest(qc_only=True)
    print(f"  Sessions: {len(manifest)}")

    # Build TF responsiveness from NPZ
    tf_status = {}  # (sname, cid) → bool
    for _, row in manifest.iterrows():
        sname = int(row["session_name"])
        npz = load_tf_traces_npz(sname)
        if npz is None:
            continue
        for i, cid in enumerate(npz["cluster_ids"]):
            z_abs = max(abs(npz["z_max_fast"][i]), abs(npz["z_min_fast"][i]),
                        abs(npz["z_max_slow"][i]), abs(npz["z_min_slow"][i]))
            tf_status[(sname, int(cid))] = z_abs >= Z_THRESH
    print(f"  Units with TF status: {len(tf_status)}")

    ct_lookup = {}
    try:
        wf = load_waveform_labels()
        for _, r in wf.iterrows():
            ct_lookup[(int(r["session_name"]), int(r["cluster_id"]))] = r["cell_type"]
    except Exception:
        pass

    # Also try loading existing change-detection / lick responsiveness caches
    cd_cache = os.path.join(CACHE_DIR, "responsiveness_all_sessions.csv")
    lick_cache = os.path.join(CACHE_DIR, "pre_lick_ramping.csv")
    cd_resp_set = set()
    lick_resp_set = set()
    try:
        cdf = pd.read_csv(cd_cache)
        for _, r in cdf.iterrows():
            if r.get("is_responsive", False) or r.get("significant", False):
                cd_resp_set.add((int(r["session_name"]), int(r["cluster_id"])))
    except Exception:
        pass
    try:
        ldf = pd.read_csv(lick_cache)
        for _, r in ldf.iterrows():
            if r.get("ramping", False) or r.get("is_lick_responsive", False):
                lick_resp_set.add((int(r["session_name"]), int(r["cluster_id"])))
    except Exception:
        pass
    print(f"  Change-responsive (from cache): {len(cd_resp_set)}")
    print(f"  Lick-responsive (from cache): {len(lick_resp_set)}")

    # ── Session loop ──────────────────────────────────────────────────
    from visdetect.analysis.align import get_event_times_by_trial

    change_t_vec = np.arange(CHANGE_WIN[0], CHANGE_WIN[1], DT, dtype=float)
    lick_t_vec = np.arange(LICK_WIN[0], LICK_WIN[1], DT, dtype=float)

    pop_change_tf = []  # (z_psth, is_tf_resp)
    pop_lick_tf = []
    records = []

    # Build session_idx lookup from manifest
    sidx_lookup = {int(r["session_name"]): r["session_idx"] for _, r in manifest.iterrows()}

    for sname_int, stage, session in session_iterator():
        sidx = sidx_lookup.get(sname_int, -1)

        # Get event times
        try:
            change_times = np.array(get_event_times_by_trial(session, "Change_ON"), dtype=float)
            change_times = change_times[np.isfinite(change_times)]
        except Exception:
            change_times = np.array([])

        # Lick times: try first-lick per trial
        try:
            lick_times = np.array(get_event_times_by_trial(session, "FirstLick"), dtype=float)
            lick_times = lick_times[np.isfinite(lick_times)]
        except Exception:
            lick_times = np.array([])

        if change_times.size < 5 and lick_times.size < 5:
            continue
        print(f"    {sname_int}: {change_times.size} changes, {lick_times.size} licks")

        sigma_bins = (SIGMA_MS / 1000.0) / DT
        for c in session.clusters:
            cid = int(c.cluster_id)
            is_tf = tf_status.get((sname_int, cid), None)
            if is_tf is None:
                continue

            st = np.asarray(c.spike_times, dtype=float).flatten()
            if st.size == 0:
                continue

            # Change PSTH
            change_z, _ = _event_psth(st, change_times, CHANGE_WIN, DT, SIGMA_MS) if change_times.size >= 5 else (None, None)
            # Lick PSTH
            lick_z, _ = _event_psth(st, lick_times, LICK_WIN, DT, SIGMA_MS) if lick_times.size >= 5 else (None, None)

            # Peak amplitudes
            change_amp = np.nan
            if change_z is not None:
                post_c = change_t_vec >= 0
                change_amp = float(np.nanmax(np.abs(change_z[post_c]))) if np.any(post_c) else np.nan
                pop_change_tf.append((change_z.copy(), is_tf))

            lick_amp = np.nan
            if lick_z is not None:
                pre_l = lick_t_vec < 0
                lick_amp = float(np.nanmax(np.abs(lick_z[pre_l]))) if np.any(pre_l) else np.nan
                pop_lick_tf.append((lick_z.copy(), is_tf))

            is_cd = (sname_int, cid) in cd_resp_set
            is_lick = (sname_int, cid) in lick_resp_set

            records.append({
                "session_name": sname_int, "cluster_id": cid,
                "stage": stage, "session_idx": sidx,
                "cell_type": ct_lookup.get((sname_int, cid), "Unknown"),
                "is_tf_responsive": is_tf,
                "is_change_responsive": is_cd,
                "is_lick_responsive": is_lick,
                "change_amp": change_amp,
                "lick_amp": lick_amp,
            })

    df = pd.DataFrame(records)
    print(f"\n  Units with event data: {len(df)}")

    if len(df) == 0:
        print("  No data. Exiting.")
        return

    cache_path = os.path.join(CACHE_DIR, "tf_sensory_motor.csv")
    df.to_csv(cache_path, index=False)
    print(f"  Cached: {cache_path}")

    # ── Create figure ─────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ── Panel A: Change PSTH (TF resp vs. non) ───────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    for tf_flag, color, label in [(True, "#E53935", "TF-responsive"),
                                   (False, "#78909C", "Non-responsive")]:
        traces = [t for t, f in pop_change_tf if f == tf_flag]
        if not traces:
            continue
        mn = np.nanmean(np.stack(traces), axis=0)
        se = np.nanstd(np.stack(traces), axis=0) / np.sqrt(len(traces))
        ax_a.plot(change_t_vec * 1000, mn, color=color, linewidth=1.5,
                  label=f"{label} (n={len(traces)})")
        ax_a.fill_between(change_t_vec * 1000, mn - se, mn + se, color=color, alpha=0.15)
    ax_a.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_a.set_xlabel("Time from change onset (ms)")
    ax_a.set_ylabel("Z-score")
    ax_a.set_title("A – Change PSTH: TF-resp. vs. non-resp.")
    ax_a.legend(fontsize=8)

    # ── Panel B: Lick PSTH (TF resp vs. non) ─────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    for tf_flag, color, label in [(True, "#E53935", "TF-responsive"),
                                   (False, "#78909C", "Non-responsive")]:
        traces = [t for t, f in pop_lick_tf if f == tf_flag]
        if not traces:
            continue
        mn = np.nanmean(np.stack(traces), axis=0)
        se = np.nanstd(np.stack(traces), axis=0) / np.sqrt(len(traces))
        ax_b.plot(lick_t_vec * 1000, mn, color=color, linewidth=1.5,
                  label=f"{label} (n={len(traces)})")
        ax_b.fill_between(lick_t_vec * 1000, mn - se, mn + se, color=color, alpha=0.15)
    ax_b.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_b.set_xlabel("Time from first lick (ms)")
    ax_b.set_ylabel("Z-score")
    ax_b.set_title("B – Lick PSTH: TF-resp. vs. non-resp.")
    ax_b.legend(fontsize=8)

    # ── Panel C: Scatter – TF z vs. change amplitude ─────────────
    ax_c = fig.add_subplot(gs[1, 0])
    resp = df[df["is_tf_responsive"]]
    non = df[~df["is_tf_responsive"]]
    if len(non):
        ax_c.scatter(non["change_amp"], non["lick_amp"],
                    s=8, alpha=0.3, color="#BDBDBD", label="Non-TF-resp")
    if len(resp):
        ax_c.scatter(resp["change_amp"], resp["lick_amp"],
                    s=12, alpha=0.5, color="#E53935", label="TF-resp")
    ax_c.set_xlabel("Change response amplitude (|z|)")
    ax_c.set_ylabel("Lick response amplitude (|z|)")
    ax_c.set_title("C – TF-responsive units: change vs. lick response")
    ax_c.legend(fontsize=8)

    # ── Panel D: Overlap summary ──────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    tf_set = set(df[df["is_tf_responsive"]][["session_name","cluster_id"]].apply(tuple, axis=1))
    cd_set = set(df[df["is_change_responsive"]][["session_name","cluster_id"]].apply(tuple, axis=1))
    lk_set = set(df[df["is_lick_responsive"]][["session_name","cluster_id"]].apply(tuple, axis=1))
    all_ids = set(df[["session_name","cluster_id"]].apply(tuple, axis=1))

    # Simple bar chart of overlaps
    categories = [
        ("TF only", len(tf_set - cd_set - lk_set)),
        ("TF ∩ Change", len(tf_set & cd_set - lk_set)),
        ("TF ∩ Lick", len(tf_set & lk_set - cd_set)),
        ("TF ∩ Both", len(tf_set & cd_set & lk_set)),
        ("Change only", len(cd_set - tf_set - lk_set)),
        ("Lick only", len(lk_set - tf_set - cd_set)),
        ("None", len(all_ids - tf_set - cd_set - lk_set)),
    ]
    cats, vals = zip(*categories)
    colors_bar = ["#E53935", "#AB47BC", "#29B6F6", "#7E57C2",
                   "#FF7043", "#66BB6A", "#BDBDBD"]
    bars = ax_d.bar(np.arange(len(cats)), vals, color=colors_bar[:len(cats)],
                    edgecolor="black", linewidth=0.3)
    ax_d.set_xticks(np.arange(len(cats)))
    ax_d.set_xticklabels(cats, rotation=30, ha="right", fontsize=7)
    ax_d.set_ylabel("Number of units")
    ax_d.set_title("D – Unit category overlaps")
    for b, v in zip(bars, vals):
        if v > 0:
            ax_d.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                     str(v), ha="center", va="bottom", fontsize=7)

    total_tf = len(tf_set)
    total_cd = len(cd_set)
    total_lk = len(lk_set)
    ax_d.text(0.02, 0.95,
             f"TF: {total_tf} | Change: {total_cd} | Lick: {total_lk}",
             transform=ax_d.transAxes, fontsize=8, va="top")

    fig.suptitle(
        "TF Pulse Responsive Units: Sensory & Motor Overlap\n"
        "(Change-detection, lick-responsiveness, TF modulation)",
        fontsize=13, fontweight="bold", y=0.98)
    save_figure(fig, "fig29_tf_sensory_motor", "08_tf_pulse")
    print("\n  ✓ Saved fig29_tf_sensory_motor.png")

    # Summary
    print("\n  Overlap summary:")
    for cat, val in categories:
        print(f"    {cat}: {val}")


if __name__ == "__main__":
    main()
