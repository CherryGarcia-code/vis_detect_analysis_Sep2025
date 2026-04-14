"""Fig34: Second-to-last TF pulse before FA lick — neural divergence by FA subtype.

Scientific question:
  For each FA trial, we identify fast TF pulses that occurred before the
  lick.  We then align neural activity to the SECOND-TO-LAST fast pulse
  and ask: does peri-pulse neural activity differ between TF-triggered
  and Impulsive FAs?  If TF-triggered FAs are truly driven by a sensory
  event, we expect stronger post-pulse modulation in TF-triggered trials
  at the penultimate pulse (the one that may set the stage for the
  final lick-triggering pulse).

Approach:
  1. Load per-session fast pulse times from the newly cached
     data/cache/tf_pulse_times/BG_046/<session>_tf_pulses.csv.
  2. For each FA trial, find all cached fast pulse times that fall in
     [Baseline_ON + 1.0, lick_time - 0.200].  The lower bound matches
     the standard min_after_baseline constraint; the upper bound is the
     200ms-shifted lick time.
  3. If >= 2 fast pulses exist in that window, take the second-to-last
     one as the alignment event.
  4. Align each unit's spikes to that pulse, build a
     (n_trials, n_bins, n_units) tensor per session, compute z-scored
     PSTHs, and compare TF-triggered vs Impulsive via time-resolved AUC
     and cluster-based permutation test.
  5. Facet by TF-responsiveness tier (All, TF-resp, Non-TF).

Produces:
  - Fig 34A: Grand-average PSTH at 2nd-to-last pulse (All units)
  - Fig 34B: Same for TF-responsive units
  - Fig 34C: Same for Non-TF units
  - Fig 34D: Time-resolved AUC by unit group
  - Fig 34E: Distribution of n_pulses_before_lick and pulse-to-lick lag
  - Fig 34F: Summary text

Saves:
  figures/07_advanced/fig34_second_pulse_divergence.png
  figures/07_advanced/second_pulse_divergence_stats.csv
  cache/second_pulse_divergence.csv
"""

import os
import sys
import gc
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import CACHE_DIR, ROOT, SUBJECT, FA_SUBTYPE_COLORS, TF_PULSE_WINDOW
from loader import load_staging_manifest, load_session
from visdetect.analysis.utils import get_good_cluster_ids, compute_zscore_normalized, smooth_psth
from plotting import setup_style, save_figure
from _fa_helpers import compute_timeresolved_auc, _find_clusters, grand_auc_cluster_test

from visdetect.analysis.align import (
    align_spikes_to_events,
    get_event_times_by_trial,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

setup_style()
warnings.filterwarnings("ignore")

# ── Parameters ────────────────────────────────────────────────────────
PULSE_WINDOW = TF_PULSE_WINDOW   # combines TF_PULSE_PRE_WINDOW + TF_PULSE_POST_WINDOW
BIN_SIZE = 0.010                  # 10 ms bins (finer for pulse-locked)
SMOOTH_SIGMA_MS = 30.0            # Gaussian smoothing sigma (ms)
ZSCORE_BASELINE = (-0.4, -0.05)  # pre-pulse baseline for z-score
MIN_TRIALS_PER_CLASS = 10        # minimum FA trials of each subtype
MIN_RT_FOR_FA = 0.6              # match Fig 24/25 exclusion
MIN_AFTER_BASELINE = 1.0         # pulse must be >= 1.0 s after Baseline_ON
N_PERM = 1000                    # permutations for cluster-based test
CLUSTER_P_THRESH = 0.05          # cluster significance threshold

# Pulse cache
PULSE_CACHE_DIR = os.path.join(ROOT, "data", "cache", "tf_pulse_times", SUBJECT)

# TF classification
TF_CLASS_FILE = os.path.join(CACHE_DIR, "tf_cell_classification.csv")


# =====================================================================
# Helpers
# =====================================================================
def load_tf_classification():
    """Load TF cell classification CSV -> dict: (session_name, cluster_id) -> tier."""
    if not os.path.exists(TF_CLASS_FILE):
        print(f"  WARNING: {TF_CLASS_FILE} not found. Tier faceting disabled.")
        return {}
    df = pd.read_csv(TF_CLASS_FILE)
    tier_map = {
        "Tier 1 (Splitter)": "Splitter",
        "Tier 2 (Unilateral)": "Unilateral",
        "Tier 3 (Omni)": "Omni",
        "Non-responsive": "Non-responsive",
    }
    lookup = {}
    for _, r in df.iterrows():
        raw_tier = r["tier"]
        lookup[(int(r["session_name"]), int(r["cluster_id"]))] = tier_map.get(
            raw_tier, "Non-responsive")
    return lookup


def load_pulse_times(session_name):
    """Load fast pulse times from cache CSV. Returns 1D float array."""
    sname = str(session_name).zfill(8)
    csv_path = os.path.join(PULSE_CACHE_DIR, f"{sname}_tf_pulses.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    fast = df["fast_times"].dropna().values.astype(float)
    return np.sort(fast)


def find_second_to_last_pulse(fast_times, baseline_on, lick_time):
    """Find the 2nd-to-last fast pulse in [baseline_on + 1.0, lick_time].

    Returns (pulse_time, n_pulses_in_window) or (None, n_pulses).
    """
    lo = baseline_on + MIN_AFTER_BASELINE
    hi = lick_time  # already 200ms-shifted

    mask = (fast_times >= lo) & (fast_times <= hi)
    pulses_in_window = fast_times[mask]
    n = len(pulses_in_window)

    if n >= 2:
        return float(pulses_in_window[-2]), n
    return None, n


# =====================================================================
# Per-session worker
# =====================================================================
def _process_session_worker(args):
    """Load one session, find 2nd-to-last pulses, align neural activity."""
    sname, stage, sidx, fa_sub_dict, tf_lookup = args
    fa_sub = pd.DataFrame(fa_sub_dict)

    # Load cached pulse times
    fast_times = load_pulse_times(sname)
    if fast_times is None or len(fast_times) == 0:
        return sname, stage, sidx, None, "no cached pulse times"

    try:
        sess = load_session(sname)
    except FileNotFoundError:
        return sname, stage, sidx, None, "pkl not found"

    baseline_on = get_event_times_by_trial(sess, "Baseline_ON")
    trials = getattr(sess, "trials", []) or []
    good_ids = get_good_cluster_ids(sess)

    if len(good_ids) == 0:
        del sess; gc.collect()
        return sname, stage, sidx, None, "no good units"

    # Build unit group membership
    sname_int = int(sname)
    unit_tier = {}
    for cid in good_ids:
        unit_tier[cid] = tf_lookup.get((sname_int, cid), "Non-responsive")

    group_indices = {
        "All": list(range(len(good_ids))),
        "TF-resp": [],
        "Non-TF": [],
        "Splitter": [],
        "Unilateral": [],
        "Omni": [],
    }
    for i, cid in enumerate(good_ids):
        tier = unit_tier[cid]
        if tier in ("Splitter", "Unilateral", "Omni"):
            group_indices["TF-resp"].append(i)
            group_indices[tier].append(i)
        else:
            group_indices["Non-TF"].append(i)

    # Find 2nd-to-last pulse for each FA trial
    pulse_events_tf = []
    pulse_events_imp = []
    pulse_stats = []  # for diagnostics

    for _, row in fa_sub.iterrows():
        tidx = int(row["trial_idx"])
        subtype = row["fa_subtype"]

        if tidx >= len(trials):
            continue

        trial = trials[tidx]
        rt_dict = getattr(trial, "reactiontimes", {}) or {}
        rt = rt_dict.get("FA", rt_dict.get("fa", np.nan))
        if np.isnan(rt) or rt < MIN_RT_FOR_FA:
            continue

        if tidx >= len(baseline_on) or np.isnan(baseline_on[tidx]):
            continue

        bo = float(baseline_on[tidx])
        lick_time = bo + rt - 0.200  # 200ms latency shift

        pulse_t, n_pulses = find_second_to_last_pulse(fast_times, bo, lick_time)
        if pulse_t is None:
            continue

        pulse_stats.append({
            "subtype": subtype,
            "n_pulses": n_pulses,
            "pulse_to_lick_lag": lick_time - pulse_t,
        })

        if subtype == "TF-triggered":
            pulse_events_tf.append(pulse_t)
        elif subtype == "Impulsive":
            pulse_events_imp.append(pulse_t)

    n_tf = len(pulse_events_tf)
    n_imp = len(pulse_events_imp)

    if n_tf < MIN_TRIALS_PER_CLASS or n_imp < MIN_TRIALS_PER_CLASS:
        del sess; gc.collect()
        return sname, stage, sidx, None, (
            f"insufficient trials with 2nd-to-last pulse "
            f"(TF={n_tf}, Imp={n_imp})"
        )

    # Align each unit to 2nd-to-last pulse events
    cluster_map = {int(c.cluster_id): c for c in sess.clusters}
    all_tf_mats = []
    all_imp_mats = []
    bin_centers = None

    for cid in good_ids:
        c = cluster_map.get(int(cid))
        if c is None:
            continue
        mat_tf, bin_centers = align_spikes_to_events(
            c.spike_times, pulse_events_tf,
            window=PULSE_WINDOW, bin_size=BIN_SIZE,
        )
        mat_imp, _ = align_spikes_to_events(
            c.spike_times, pulse_events_imp,
            window=PULSE_WINDOW, bin_size=BIN_SIZE,
        )
        all_tf_mats.append(mat_tf)
        all_imp_mats.append(mat_imp)

    del sess; gc.collect()

    if len(all_tf_mats) == 0 or bin_centers is None:
        return sname, stage, sidx, None, "alignment failed"

    full_tensor_tf = np.stack(all_tf_mats, axis=2)
    full_tensor_imp = np.stack(all_imp_mats, axis=2)

    # Build results per unit group
    summary = {}
    for group_name, idx_list in group_indices.items():
        if len(idx_list) == 0:
            continue

        tensor_tf = full_tensor_tf[:, :, idx_list]
        tensor_imp = full_tensor_imp[:, :, idx_list]

        # Z-score each unit
        ztf = compute_zscore_normalized(tensor_tf, bin_centers, ZSCORE_BASELINE)
        zimp = compute_zscore_normalized(tensor_imp, bin_centers, ZSCORE_BASELINE)

        psth_tf = np.nanmean(np.nanmean(ztf, axis=0), axis=1)
        psth_imp = np.nanmean(np.nanmean(zimp, axis=0), axis=1)
        auc = compute_timeresolved_auc(tensor_tf, tensor_imp)

        summary[group_name] = {
            "bin_centers": bin_centers,
            "psth_tf": psth_tf,
            "psth_imp": psth_imp,
            "auc": auc,
            "n_units": len(idx_list),
            "n_tf": n_tf,
            "n_imp": n_imp,
        }

    # Pulse diagnostics
    pulse_stats_summary = {}
    if pulse_stats:
        ps_df = pd.DataFrame(pulse_stats)
        pulse_stats_summary = {
            "mean_n_pulses": float(ps_df["n_pulses"].mean()),
            "median_pulse_to_lick_lag": float(ps_df["pulse_to_lick_lag"].median()),
            "all_n_pulses": ps_df["n_pulses"].values.tolist(),
            "all_lags": ps_df["pulse_to_lick_lag"].values.tolist(),
        }

    msg = f"n_tf={n_tf}, n_imp={n_imp}, units={len(good_ids)}"
    return sname, stage, sidx, (summary, pulse_stats_summary), msg


# =====================================================================
# Grand-average computation
# =====================================================================
def compute_grand_average(all_summaries, group_name):
    psth_tf_list = []
    psth_imp_list = []
    auc_list = []
    n_units_list = []

    for sname, stage, sidx, data_tuple in all_summaries:
        if data_tuple is None:
            continue
        summary, _ = data_tuple
        if group_name not in summary:
            continue
        data = summary[group_name]
        bin_centers = data["bin_centers"]
        psth_tf_list.append(data["psth_tf"])
        psth_imp_list.append(data["psth_imp"])
        auc_list.append(data["auc"])
        n_units_list.append(data["n_units"])

    if len(psth_tf_list) == 0:
        return None

    psth_tf_arr = np.array(psth_tf_list)
    psth_imp_arr = np.array(psth_imp_list)
    auc_arr = np.array(auc_list)
    n_sess = len(psth_tf_list)
    sqrt_n = np.sqrt(n_sess)

    return {
        "bin_centers": bin_centers,
        "mean_psth_tf": np.nanmean(psth_tf_arr, axis=0),
        "sem_psth_tf": np.nanstd(psth_tf_arr, axis=0) / sqrt_n,
        "mean_psth_imp": np.nanmean(psth_imp_arr, axis=0),
        "sem_psth_imp": np.nanstd(psth_imp_arr, axis=0) / sqrt_n,
        "mean_auc": np.nanmean(auc_arr, axis=0),
        "sem_auc": np.nanstd(auc_arr, axis=0) / sqrt_n,
        "n_sessions": n_sess,
        "psth_tf_arr": psth_tf_arr,
        "psth_imp_arr": psth_imp_arr,
        "auc_arr": auc_arr,
        "mean_n_units": np.mean(n_units_list),
    }


def smooth(arr, sigma_ms=SMOOTH_SIGMA_MS):
    """Gaussian-smooth a 1D array (delegates to shared smooth_psth)."""
    return smooth_psth(arr, BIN_SIZE, sigma_ms)


# =====================================================================
# Main
# =====================================================================
CLASSIFICATION_FILES = {
    "original": "fa_subtype_classification.csv",
    "circular_shuffle": "fa_classification_circular_shuffle.csv",
    "matched_null": "fa_classification_matched_null.csv",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Parallel workers (default: 1 = sequential)")
    parser.add_argument("--classification", type=str, default="original",
                        choices=list(CLASSIFICATION_FILES.keys()),
                        help="Which FA classification to use (default: original)")
    args = parser.parse_args()

    cls_name = args.classification
    cls_suffix = "" if cls_name == "original" else f"_{cls_name}"

    print("=" * 60)
    print("[07h] 2nd-to-last TF pulse before FA lick")
    print(f"       Classification: {cls_name}")
    print("=" * 60)

    # Load FA subtype classification
    fa_path = os.path.join(CACHE_DIR, CLASSIFICATION_FILES[cls_name])
    if not os.path.exists(fa_path):
        print(f"  ERROR: {CLASSIFICATION_FILES[cls_name]} not found.")
        print("  Run the corresponding classification script first.")
        return
    fa_df = pd.read_csv(fa_path)
    # Remap labels so downstream logic (which uses "TF-triggered") works unchanged
    fa_df["fa_subtype"] = fa_df["fa_subtype"].replace("Stimulus-driven", "TF-triggered")
    print(f"  {len(fa_df)} classified FA trials loaded ({cls_name})")

    # Check pulse cache
    if not os.path.isdir(PULSE_CACHE_DIR):
        print(f"  ERROR: Pulse cache not found: {PULSE_CACHE_DIR}")
        print("  Run scripts/extract_tf_pulse_times.py first.")
        return

    manifest = load_staging_manifest(qc_only=True)
    tf_lookup = load_tf_classification()
    print(f"  {len(manifest)} QC-passed sessions")
    print(f"  {len(tf_lookup)} units with TF tier classification")

    # ── Build task list ───────────────────────────────────────────────
    tasks = []
    for _, row in manifest.iterrows():
        sname = int(row["session_name"])
        stage = row["stage"]
        sidx = row["session_idx"]

        fa_sub = fa_df[fa_df["session_name"] == sname].copy()
        n_tf = (fa_sub["fa_subtype"] == "TF-triggered").sum()
        n_imp = (fa_sub["fa_subtype"] == "Impulsive").sum()
        if n_tf < MIN_TRIALS_PER_CLASS or n_imp < MIN_TRIALS_PER_CLASS:
            continue

        tasks.append((
            sname, stage, sidx,
            fa_sub.to_dict("list"),
            tf_lookup,
        ))

    print(f"\n  {len(tasks)} sessions have enough FA trials of both subtypes")

    # ── Per-session processing ────────────────────────────────────────
    print(f"\n[Step 1] Extracting pulse-aligned neural activity...")
    all_summaries = []

    if args.n_workers > 1:
        actual_workers = min(args.n_workers, len(tasks))
        print(f"  Using {actual_workers} parallel workers")
        with ProcessPoolExecutor(max_workers=actual_workers) as ex:
            futures = {
                ex.submit(_process_session_worker, task): task[0]
                for task in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Sessions"):
                sname, stage, sidx, data_tuple, msg = future.result()
                print(f"  Session {sname} ({stage}, idx={sidx})... {msg}")
                all_summaries.append((sname, stage, sidx, data_tuple))
    else:
        for task_args in tqdm(tasks, desc="Sessions"):
            sname, stage, sidx = task_args[0], task_args[1], task_args[2]
            print(f"  Session {sname} ({stage}, idx={sidx})...", end=" ",
                  flush=True)
            _, _, _, data_tuple, msg = _process_session_worker(task_args)
            print(msg)
            all_summaries.append((sname, stage, sidx, data_tuple))

    valid = [(s, st, si, dt) for s, st, si, dt in all_summaries
             if dt is not None]
    print(f"\n  {len(valid)} sessions with valid data")

    if len(valid) == 0:
        print("  ERROR: No valid sessions. Exiting.")
        return

    # ── Save per-session results to CSV ───────────────────────────────
    rows = []
    for sname, stage, sidx, (summary, pulse_stats) in valid:
        for group_name, data in summary.items():
            for bi, bc in enumerate(data["bin_centers"]):
                rows.append({
                    "session_name": sname,
                    "stage": stage,
                    "session_idx": sidx,
                    "unit_group": group_name,
                    "bin_center": round(bc, 4),
                    "psth_tf": data["psth_tf"][bi],
                    "psth_imp": data["psth_imp"][bi],
                    "auc": data["auc"][bi],
                    "n_units": data["n_units"],
                    "n_tf": data["n_tf"],
                    "n_imp": data["n_imp"],
                })
    results_df = pd.DataFrame(rows)
    cache_path = os.path.join(CACHE_DIR, f"second_pulse_divergence{cls_suffix}.csv")
    results_df.to_csv(cache_path, index=False)
    print(f"  Saved: {cache_path}")

    # ══════════════════════════════════════════════════════════════════
    # Compute grand averages
    # ══════════════════════════════════════════════════════════════════
    print("\n[Step 2] Computing grand averages and cluster tests...")
    grand = {}
    for gname in ["All", "TF-resp", "Non-TF", "Splitter", "Unilateral", "Omni"]:
        ga = compute_grand_average(valid, gname)
        if ga is not None:
            grand[gname] = ga
            print(f"  {gname}: {ga['n_sessions']} sessions, "
                  f"~{ga['mean_n_units']:.0f} units/session")

    # Cluster permutation tests
    print("\n  Running cluster permutation tests (1000 permutations)...")
    cluster_results = {}
    for gname in ["All", "TF-resp", "Non-TF"]:
        if gname not in grand:
            continue
        auc_arr = grand[gname]["auc_arr"]
        mean_auc, sig_mask, p_clusters = grand_auc_cluster_test(auc_arr)
        cluster_results[gname] = {
            "mean_auc": mean_auc,
            "sig_mask": sig_mask,
            "p_clusters": p_clusters,
        }
        n_sig_bins = sig_mask.sum()
        sig_str = (f"{n_sig_bins} sig bins" if n_sig_bins > 0
                   else "no significant clusters")
        print(f"  Cluster test [{gname}]: {sig_str}")
        for s, e, stat, pv in p_clusters:
            bc = grand[gname]["bin_centers"]
            print(f"    cluster [{bc[s]:.3f}, {bc[min(e-1, len(bc)-1)]:.3f}] s: "
                  f"stat={stat:.2f}, p={pv:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # Aggregate pulse diagnostics
    # ══════════════════════════════════════════════════════════════════
    all_n_pulses = []
    all_lags = []
    for _, _, _, (_, ps) in valid:
        if ps:
            all_n_pulses.extend(ps.get("all_n_pulses", []))
            all_lags.extend(ps.get("all_lags", []))

    # ══════════════════════════════════════════════════════════════════
    # FIGURE 34
    # ══════════════════════════════════════════════════════════════════
    print("\n[Step 3] Generating Figure 34...")
    fig = plt.figure(figsize=(22, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.32)
    stats_rows = []

    # ── PSTH panel helper ─────────────────────────────────────────────
    def plot_psth_panel(ax, gname, title_letter, title_suffix):
        if gname not in grand:
            ax.text(0.5, 0.5, f"No data for {gname}", ha="center",
                    va="center", transform=ax.transAxes)
            ax.set_title(f"{title_letter}. {title_suffix}")
            return

        ga = grand[gname]
        bc = ga["bin_centers"]
        psth_tf_sm = smooth(ga["mean_psth_tf"])
        psth_imp_sm = smooth(ga["mean_psth_imp"])
        sem_tf_sm = smooth(ga["sem_psth_tf"])
        sem_imp_sm = smooth(ga["sem_psth_imp"])

        ax.plot(bc, psth_tf_sm, color=FA_SUBTYPE_COLORS["Stimulus-driven"],
                lw=2, label="TF-triggered")
        ax.fill_between(bc, psth_tf_sm - sem_tf_sm, psth_tf_sm + sem_tf_sm,
                        color=FA_SUBTYPE_COLORS["Stimulus-driven"], alpha=0.2)
        ax.plot(bc, psth_imp_sm, color=FA_SUBTYPE_COLORS["Impulsive"],
                lw=2, label="Impulsive")
        ax.fill_between(bc, psth_imp_sm - sem_imp_sm, psth_imp_sm + sem_imp_sm,
                        color=FA_SUBTYPE_COLORS["Impulsive"], alpha=0.2)

        ax.axvline(0, color="k", ls="--", lw=1, alpha=0.5, label="Pulse onset")

        # Shade significant clusters
        if gname in cluster_results:
            for s, e, _, pv in cluster_results[gname]["p_clusters"]:
                if pv < CLUSTER_P_THRESH:
                    ax.axvspan(bc[s], bc[min(e-1, len(bc)-1)],
                               color="gold", alpha=0.25, zorder=0)

        ax.set_xlabel("Time relative to pulse onset (s)")
        ax.set_ylabel("z-scored firing rate")
        ax.legend(fontsize=8, loc="upper left")
        n_units_str = f"{ga['mean_n_units']:.0f}"
        ax.set_title(
            f"{title_letter}. {title_suffix}\n"
            f"(n={ga['n_sessions']} sessions, ~{n_units_str} units/sess)",
            fontweight="bold",
        )

    # ── Panel A: All units ────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    plot_psth_panel(ax_a, "All", "A",
                    "All units: 2nd-to-last pulse")

    # ── Panel B: TF-responsive ────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    plot_psth_panel(ax_b, "TF-resp", "B", "TF-responsive units")

    # ── Panel C: Non-TF ──────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    plot_psth_panel(ax_c, "Non-TF", "C", "Non-TF-responsive units")

    # ── Panel D: Time-resolved AUC ────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    group_colors = {
        "All": "#2c3e50", "TF-resp": "#e74c3c", "Non-TF": "#7f8c8d",
    }
    for gname in ["All", "TF-resp", "Non-TF"]:
        if gname not in grand:
            continue
        ga = grand[gname]
        bc = ga["bin_centers"]
        auc_sm = smooth(ga["mean_auc"])
        sem_sm = smooth(ga["sem_auc"])
        ax_d.plot(bc, auc_sm, color=group_colors[gname], lw=2, label=gname)
        ax_d.fill_between(bc, auc_sm - sem_sm, auc_sm + sem_sm,
                          color=group_colors[gname], alpha=0.15)

        if gname in cluster_results:
            for s, e, _, pv in cluster_results[gname]["p_clusters"]:
                if pv < CLUSTER_P_THRESH:
                    ax_d.axvspan(bc[s], bc[min(e-1, len(bc)-1)],
                                 color=group_colors[gname], alpha=0.08, zorder=0)

    ax_d.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.5)
    ax_d.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
    ax_d.set_xlabel("Time relative to pulse onset (s)")
    ax_d.set_ylabel("AUC (TF-triggered vs Impulsive)")
    ax_d.legend(fontsize=8)
    ax_d.set_title("D. Time-resolved discriminability\n"
                   "(grand-average AUC)", fontweight="bold")
    ax_d.set_ylim(0.35, 0.65)

    # ── Panel E: Pulse diagnostics ────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    if all_n_pulses:
        ax_e2 = ax_e.twinx()

        # Histogram of n_pulses before lick
        n_arr = np.array(all_n_pulses)
        bins_n = np.arange(2, max(n_arr) + 2) - 0.5
        ax_e.hist(n_arr, bins=bins_n, color="#3498db", alpha=0.6,
                  edgecolor="white", label="# fast pulses")
        ax_e.set_xlabel("Count / lag (s)")
        ax_e.set_ylabel("# FA trials (pulse count)", color="#3498db")

        # Histogram of pulse-to-lick lag
        lag_arr = np.array(all_lags)
        bins_lag = np.linspace(0, max(lag_arr), 30)
        ax_e2.hist(lag_arr, bins=bins_lag, color="#e74c3c", alpha=0.4,
                   edgecolor="white", label="Pulse-to-lick lag")
        ax_e2.set_ylabel("# FA trials (lag)", color="#e74c3c")

        ax_e.set_title(
            f"E. Pulse diagnostics\n"
            f"(median lag = {np.median(lag_arr):.2f} s, "
            f"mean pulses = {np.mean(n_arr):.1f})",
            fontweight="bold",
        )
        # Combine legends
        h1, l1 = ax_e.get_legend_handles_labels()
        h2, l2 = ax_e2.get_legend_handles_labels()
        ax_e.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    else:
        ax_e.text(0.5, 0.5, "No pulse diagnostics", ha="center",
                  va="center", transform=ax_e.transAxes)
        ax_e.set_title("E. Pulse diagnostics", fontweight="bold")

    # ── Panel F: Summary ──────────────────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")

    n_sess_total = len(valid)
    total_tf = sum(dt[0].get("All", {}).get("n_tf", 0) for _, _, _, dt in valid)
    total_imp = sum(dt[0].get("All", {}).get("n_imp", 0) for _, _, _, dt in valid)
    total_units = sum(dt[0].get("All", {}).get("n_units", 0)
                      for _, _, _, dt in valid)
    tf_units = sum(dt[0].get("TF-resp", {}).get("n_units", 0)
                   for _, _, _, dt in valid if "TF-resp" in dt[0])

    lines = [
        "F. Summary",
        "",
        f"Sessions analyzed: {n_sess_total}",
        f"Total FA trials (with 2nd pulse): {total_tf + total_imp}",
        f"  TF-triggered: {total_tf}",
        f"  Impulsive: {total_imp}",
        f"Total unit-sessions: {total_units}",
        f"  TF-responsive: {tf_units}",
        "",
        f"Alignment: 2nd-to-last fast pulse",
        f"Pulse window: [{PULSE_WINDOW[0]}, {PULSE_WINDOW[1]}] s",
        f"Bin size: {BIN_SIZE*1000:.0f} ms",
        f"Smoothing: {SMOOTH_SIGMA_MS:.0f} ms Gaussian",
        f"Z-score baseline: [{ZSCORE_BASELINE[0]}, {ZSCORE_BASELINE[1]}] s",
        f"Min pulses before lick: 2",
        f"Min RT: {MIN_RT_FOR_FA} s",
        f"Lick time shift: -200 ms",
        f"Cluster test: {N_PERM} permutations",
    ]

    if all_lags:
        lines.extend([
            "",
            f"Median pulse-to-lick lag: {np.median(all_lags):.2f} s",
            f"Mean fast pulses/trial: {np.mean(all_n_pulses):.1f}",
        ])

    lines.append("")
    lines.append("Cluster-based permutation tests:")
    for gname in ["All", "TF-resp", "Non-TF"]:
        if gname in cluster_results:
            cr = cluster_results[gname]
            n_sig = cr["sig_mask"].sum()
            if n_sig > 0:
                lines.append(f"  {gname}: {n_sig} sig bins")
                for s, e, stat, pv in cr["p_clusters"]:
                    if pv < CLUSTER_P_THRESH:
                        bc = grand[gname]["bin_centers"]
                        t0 = bc[s]
                        t1 = bc[min(e-1, len(bc)-1)]
                        lines.append(f"    [{t0:.3f}, {t1:.3f}] s (p={pv:.4f})")
            else:
                lines.append(f"  {gname}: no significant clusters")

    # Record stats
    for gname in ["All", "TF-resp", "Non-TF"]:
        if gname in cluster_results:
            for s, e, stat, pv in cluster_results[gname]["p_clusters"]:
                bc = grand[gname]["bin_centers"]
                stats_rows.append({
                    "test": f"cluster_{gname}",
                    "t_start": round(bc[s], 4),
                    "t_end": round(bc[min(e-1, len(bc)-1)], 4),
                    "cluster_stat": round(stat, 4),
                    "p": round(pv, 4),
                    "significant": pv < CLUSTER_P_THRESH,
                    "n_sessions": grand[gname]["n_sessions"],
                })

    ax_f.text(0.05, 0.95, "\n".join(lines), transform=ax_f.transAxes,
              fontsize=9, va="top", ha="left", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                        edgecolor="grey", alpha=0.8))

    # Save figure
    save_figure(fig, f"fig34_second_pulse_divergence{cls_suffix}", "07_advanced")
    plt.close(fig)

    # Save stats
    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        stats_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "figures", "07_advanced", f"second_pulse_divergence{cls_suffix}_stats.csv",
        )
        stats_df.to_csv(stats_path, index=False)
        print(f"  Saved stats: {stats_path}")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nSessions: {n_sess_total}")
    print(f"FA trials with >=2 fast pulses: {total_tf} TF-triggered, "
          f"{total_imp} Impulsive")
    print(f"Alignment: 2nd-to-last fast pulse, "
          f"window [{PULSE_WINDOW[0]}, {PULSE_WINDOW[1]}] s")

    if all_lags:
        print(f"Median pulse-to-lick lag: {np.median(all_lags):.2f} s")
        print(f"Mean fast pulses before lick: {np.mean(all_n_pulses):.1f}")

    print("\nGrand-average AUC (peak deviation from 0.5):")
    for gname in ["All", "TF-resp", "Non-TF", "Splitter", "Unilateral", "Omni"]:
        if gname in grand:
            ga = grand[gname]
            peak_dev = np.max(np.abs(ga["mean_auc"] - 0.5))
            peak_bin = ga["bin_centers"][np.argmax(np.abs(ga["mean_auc"] - 0.5))]
            print(f"  {gname:15s}: peak |AUC-0.5| = {peak_dev:.4f} "
                  f"at {peak_bin:.3f} s (n={ga['n_sessions']} sess)")

    print("\nSignificant clusters:")
    any_sig = False
    for gname in ["All", "TF-resp", "Non-TF"]:
        if gname in cluster_results:
            for s, e, stat, pv in cluster_results[gname]["p_clusters"]:
                if pv < CLUSTER_P_THRESH:
                    bc = grand[gname]["bin_centers"]
                    print(f"  {gname}: [{bc[s]:.3f}, "
                          f"{bc[min(e-1,len(bc)-1)]:.3f}] s "
                          f"(stat={stat:.2f}, p={pv:.4f})")
                    any_sig = True
    if not any_sig:
        print("  None")

    print("\nDone.")


if __name__ == "__main__":
    main()
