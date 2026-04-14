"""Fig33: FA lick-aligned neural divergence — TF-triggered vs Impulsive, by TF tier.

Scientific question:
  Given an FA lick aligned to time zero, how does peri-lick neural activity
  differ between TF-triggered (preceded by a fast TF fluctuation) and
  Impulsive (no preceding TF fluctuation) FAs?  Where in time does
  the divergence emerge, and is it specific to TF-responsive neurons?

Alignment:
  All neural activity aligned to the FA lick itself (Baseline_ON + RT_FA).
  Window: [-1.0, +0.2] s relative to lick, 25 ms bins.

Analysis per time bin:
  - AUC (TF-triggered=1 vs Impulsive=0) from per-trial population firing
    rate, computed per unit then averaged across units within each group.
  - Cluster-based permutation test for temporal significance.
  - Faceted by unit group: All, TF-responsive, Non-TF, and each tier
    (Splitter, Unilateral, Omni).

Produces:
  - Fig 33A: Grand-average peri-lick PSTH, TF-triggered vs Impulsive (all units)
  - Fig 33B: Same for TF-responsive units only
  - Fig 33C: Same for Non-TF units
  - Fig 33D: Time-resolved AUC by unit group (All, TF-resp, Non-TF)
  - Fig 33E: Time-resolved AUC for each TF tier (Splitter, Unilateral, Omni)
  - Fig 33F: Population-level stats (n sessions, n trials, significance)

Saves:
  figures/07_advanced/fig33_fa_lick_aligned_divergence.png
  figures/07_advanced/fa_lick_aligned_divergence_stats.csv
  cache/fa_lick_aligned_divergence.csv
"""

import os
import sys
import gc
import argparse
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import CACHE_DIR, HMM_STATE_COLORS, DEFAULT_BIN_SIZE, FA_SUBTYPE_COLORS
from loader import load_staging_manifest, load_session
from visdetect.analysis.utils import get_good_cluster_ids, compute_zscore_normalized, smooth_psth
from plotting import setup_style, save_figure
from _fa_helpers import compute_timeresolved_auc, _find_clusters, grand_auc_cluster_test

from visdetect.analysis.align import (
    align_spikes_to_events,
    get_event_times_by_trial,
)

setup_style()
warnings.filterwarnings("ignore")

# ── Parameters ────────────────────────────────────────────────────────
LICK_WINDOW = (-1.0, 0.2)        # seconds relative to lick
BIN_SIZE = DEFAULT_BIN_SIZE
SMOOTH_SIGMA_MS = 50.0            # Gaussian smoothing sigma (ms)
ZSCORE_BASELINE = (-1.0, -0.5)   # pre-lick baseline for z-score normalization
MIN_TRIALS_PER_CLASS = 15         # minimum FA trials of each subtype per session
MIN_RT_FOR_LTA = 0.6              # match Fig 24 exclusion
N_PERM = 1000                     # permutations for cluster-based test
CLUSTER_P_THRESH = 0.05           # cluster significance threshold

# TF classification file
TF_CLASS_FILE = os.path.join(CACHE_DIR, "tf_cell_classification.csv")


# =====================================================================
# Helper: load TF classification lookup
# =====================================================================
def load_tf_classification():
    """Load TF cell classification CSV and return dict: (session_name, cluster_id) -> tier."""
    if not os.path.exists(TF_CLASS_FILE):
        print(f"  WARNING: {TF_CLASS_FILE} not found. Tier faceting disabled.")
        return {}
    df = pd.read_csv(TF_CLASS_FILE)
    # Map tier labels to short names
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


# =====================================================================
# Core: extract lick-aligned firing rates for one session
# =====================================================================
def process_session(sess, sname, fa_sub, tf_lookup):
    """For one session, align all unit spikes to FA lick times.

    Aligns each unit ONCE, then builds group tensors by indexing.

    Returns:
      dict of group_name -> {tensor_tf, tensor_imp, bin_centers, n_units, n_tf, n_imp}
      or None if insufficient data.
    """
    baseline_on = get_event_times_by_trial(sess, "Baseline_ON")
    trials = getattr(sess, "trials", []) or []

    good_ids = get_good_cluster_ids(sess)
    if len(good_ids) == 0:
        return None

    # Build unit group membership (indices into good_ids)
    sname_int = int(sname)
    unit_tier = {}  # cid -> tier string
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

    # Compute absolute lick times for each FA trial
    lick_times_tf = []
    lick_times_imp = []
    # HMM state breakdown (FA subtype × state)
    HMM_STATES = ["Disengaged", "Engaged", "Impulsive"]
    lick_times_by_state = {
        (subtype, state): []
        for subtype in ("TF-triggered", "Impulsive")
        for state in HMM_STATES
    }

    for _, row in fa_sub.iterrows():
        tidx = int(row["trial_idx"])
        subtype = row["fa_subtype"]
        hmm_state = row.get("hmm_state", "Unknown")

        if tidx >= len(trials):
            continue

        trial = trials[tidx]
        rt_dict = getattr(trial, "reactiontimes", {}) or {}
        rt = rt_dict.get("FA", rt_dict.get("fa", np.nan))
        if np.isnan(rt) or rt < MIN_RT_FOR_LTA:
            continue

        if tidx >= len(baseline_on) or np.isnan(baseline_on[tidx]):
            continue

        # Subtract 200 ms latency shift (matching compute_true_reaction_time convention)
        abs_lick_time = baseline_on[tidx] + rt - 0.200

        if subtype == "TF-triggered":
            lick_times_tf.append(abs_lick_time)
        elif subtype == "Impulsive":
            lick_times_imp.append(abs_lick_time)

        if hmm_state in HMM_STATES:
            lick_times_by_state[(subtype, hmm_state)].append(abs_lick_time)

    n_tf = len(lick_times_tf)
    n_imp = len(lick_times_imp)

    if n_tf < MIN_TRIALS_PER_CLASS or n_imp < MIN_TRIALS_PER_CLASS:
        return None

    # Align each unit ONCE (both subtypes)
    cluster_map = {int(c.cluster_id): c for c in sess.clusters}
    all_tf_mats = []
    all_imp_mats = []
    # Per HMM state: align each unit to each state's lick subset
    hmm_state_mats = {key: [] for key in lick_times_by_state}
    bin_centers = None

    for cid in good_ids:
        c = cluster_map.get(int(cid))
        if c is None:
            continue

        mat_tf, bin_centers = align_spikes_to_events(
            c.spike_times, lick_times_tf,
            window=LICK_WINDOW, bin_size=BIN_SIZE,
        )
        mat_imp, _ = align_spikes_to_events(
            c.spike_times, lick_times_imp,
            window=LICK_WINDOW, bin_size=BIN_SIZE,
        )
        all_tf_mats.append(mat_tf)
        all_imp_mats.append(mat_imp)

        # Align to each HMM state subset
        for key, times in lick_times_by_state.items():
            if len(times) >= 5:
                mat_s, _ = align_spikes_to_events(
                    c.spike_times, times,
                    window=LICK_WINDOW, bin_size=BIN_SIZE,
                )
                hmm_state_mats[key].append(mat_s)
            else:
                hmm_state_mats[key].append(None)

    if len(all_tf_mats) == 0 or bin_centers is None:
        return None

    # Full tensors: (n_trials, n_bins, n_units)
    full_tensor_tf = np.stack(all_tf_mats, axis=2)
    full_tensor_imp = np.stack(all_imp_mats, axis=2)

    # Build group tensors by indexing into the full tensor (unit-axis faceting)
    results = {}
    for group_name, idx_list in group_indices.items():
        if len(idx_list) == 0:
            continue
        results[group_name] = {
            "tensor_tf": full_tensor_tf[:, :, idx_list],
            "tensor_imp": full_tensor_imp[:, :, idx_list],
            "bin_centers": bin_centers,
            "n_units": len(idx_list),
            "n_tf": n_tf,
            "n_imp": n_imp,
        }

    # Build HMM state results (All units, trial-axis faceting)
    # Group key: "TF-Engaged", "TF-Disengaged", "TF-Impulsive", "Imp-Engaged" etc.
    n_all_units = len(all_tf_mats)
    for (subtype, state), mats in hmm_state_mats.items():
        valid_mats = [m for m in mats if m is not None]
        if len(valid_mats) == 0 or len(lick_times_by_state[(subtype, state)]) < 5:
            continue
        n_trials_state = len(lick_times_by_state[(subtype, state)])
        tensor_state = np.stack(valid_mats, axis=2)  # (n_trials, n_bins, n_units)
        # Tag: "TF-Triggered|Engaged", etc. (shortened for group name)
        short_sub = "TF" if subtype == "TF-triggered" else "Imp"
        group_key = f"{short_sub}|{state}"
        results[group_key] = {
            "tensor_state": tensor_state,   # single-subtype tensor for this state
            "bin_centers": bin_centers,
            "n_units": len(valid_mats),
            "n_trials": n_trials_state,
            "subtype": subtype,
            "hmm_state": state,
        }

    return results if results else None


# =====================================================================
# Module-level worker for parallel execution
# =====================================================================
def _process_session_worker(args):
    """Load one session, extract lick-aligned neural data, return summary."""
    sname, stage, sidx, fa_sub_dict, tf_lookup = args
    fa_sub = pd.DataFrame(fa_sub_dict)

    try:
        sess = load_session(sname)
    except FileNotFoundError:
        return sname, stage, sidx, None, "pkl not found"

    result = process_session(sess, sname, fa_sub, tf_lookup)

    del sess
    gc.collect()

    if result is None:
        return sname, stage, sidx, None, "skipped (insufficient trials)"

    # Summarise for transport: per-group mean PSTH and AUC
    summary = {}
    for group_name, data in result.items():
        bin_centers = data["bin_centers"]

        # Standard unit-facet groups (TF vs Imp comparison)
        if "tensor_tf" in data:
            tensor_tf = data["tensor_tf"]
            tensor_imp = data["tensor_imp"]

            # Z-score each unit using pre-lick baseline before computing PSTH
            ztf = compute_zscore_normalized(tensor_tf, bin_centers, ZSCORE_BASELINE)
            zimp = compute_zscore_normalized(tensor_imp, bin_centers, ZSCORE_BASELINE)

            # Mean across trials, then across units -> (n_bins,)
            psth_tf = np.nanmean(np.nanmean(ztf, axis=0), axis=1)
            psth_imp = np.nanmean(np.nanmean(zimp, axis=0), axis=1)

            auc = compute_timeresolved_auc(tensor_tf, tensor_imp)

            summary[group_name] = {
                "bin_centers": bin_centers,
                "psth_tf": psth_tf,
                "psth_imp": psth_imp,
                "auc": auc,
                "n_units": data["n_units"],
                "n_tf": data["n_tf"],
                "n_imp": data["n_imp"],
                "is_hmm_group": False,
            }

        # HMM state groups (single-subtype PSTH only)
        elif "tensor_state" in data:
            tensor_state = data["tensor_state"]
            z_state = compute_zscore_normalized(tensor_state, bin_centers, ZSCORE_BASELINE)
            psth = np.nanmean(np.nanmean(z_state, axis=0), axis=1)
            summary[group_name] = {
                "bin_centers": bin_centers,
                "psth": psth,
                "n_units": data["n_units"],
                "n_trials": data["n_trials"],
                "subtype": data["subtype"],
                "hmm_state": data["hmm_state"],
                "is_hmm_group": True,
            }

    # Get n_tf/n_imp from the All group
    all_data = result.get("All", {})
    n_tf = all_data.get("n_tf", 0)
    n_imp = all_data.get("n_imp", 0)
    all_nu = summary.get("All", {}).get("n_units", 0)
    tf_nu = summary.get("TF-resp", {}).get("n_units", 0)
    msg = f"n_tf={n_tf}, n_imp={n_imp}, units={all_nu}(TF={tf_nu})"

    # Count HMM state groups found
    hmm_groups = [k for k, v in summary.items() if v.get("is_hmm_group")]
    if hmm_groups:
        msg += f", HMM groups: {len(hmm_groups)}"

    return sname, stage, sidx, summary, msg


# =====================================================================
# Grand-average computation
# =====================================================================
def compute_grand_average(all_summaries, group_name):
    """Pool per-session PSTHs and AUCs for a given unit group."""
    psth_tf_list = []
    psth_imp_list = []
    auc_list = []
    n_units_list = []

    for sname, stage, sidx, summary in all_summaries:
        if summary is None or group_name not in summary:
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


def compute_hmm_grand_average(all_summaries, group_key):
    """Pool per-session PSTHs for an HMM state group (single-subtype)."""
    psth_list = []
    n_trials_list = []
    bin_centers = None

    for sname, stage, sidx, summary in all_summaries:
        if summary is None or group_key not in summary:
            continue
        data = summary[group_key]
        if not data.get("is_hmm_group", False):
            continue
        psth_list.append(data["psth"])
        n_trials_list.append(data["n_trials"])
        if bin_centers is None:
            bin_centers = data["bin_centers"]

    if len(psth_list) == 0:
        return None

    psth_arr = np.array(psth_list)
    n_sess = len(psth_list)
    sqrt_n = np.sqrt(n_sess)

    return {
        "bin_centers": bin_centers,
        "mean_psth": np.nanmean(psth_arr, axis=0),
        "sem_psth": np.nanstd(psth_arr, axis=0) / sqrt_n,
        "n_sessions": n_sess,
        "mean_n_trials": np.mean(n_trials_list),
    }


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
    print("[07g] FA lick-aligned neural divergence")
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

    # ── Per-session processing ────────────────────────────────────────
    print(f"\n[Step 1] Extracting lick-aligned neural activity ({len(tasks)} sessions)...")
    all_summaries = []

    if args.n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        actual_workers = min(args.n_workers, len(tasks))
        print(f"  Using {actual_workers} parallel workers")
        with ProcessPoolExecutor(max_workers=actual_workers) as ex:
            for sname, stage, sidx, summary, msg in ex.map(
                    _process_session_worker, tasks):
                print(f"  Session {sname} ({stage}, idx={sidx})... {msg}")
                all_summaries.append((sname, stage, sidx, summary))
    else:
        for task_args in tasks:
            sname, stage, sidx = task_args[0], task_args[1], task_args[2]
            print(f"  Session {sname} ({stage}, idx={sidx})...", end=" ", flush=True)
            _, _, _, summary, msg = _process_session_worker(task_args)
            print(msg)
            all_summaries.append((sname, stage, sidx, summary))

    valid = [(s, st, si, sm) for s, st, si, sm in all_summaries if sm is not None]
    print(f"\n  {len(valid)} sessions with valid data")

    if len(valid) == 0:
        print("  ERROR: No valid sessions. Exiting.")
        return

    # ── Save per-session results to CSV ───────────────────────────────
    rows = []
    for sname, stage, sidx, summary in valid:
        for group_name, data in summary.items():
            if data.get("is_hmm_group", False):
                # HMM state group: single PSTH (no TF vs Imp comparison here)
                for bi, bc in enumerate(data["bin_centers"]):
                    rows.append({
                        "session_name": sname,
                        "stage": stage,
                        "session_idx": sidx,
                        "unit_group": group_name,
                        "bin_center": round(bc, 4),
                        "psth_tf": np.nan,
                        "psth_imp": np.nan,
                        "psth": data["psth"][bi],
                        "auc": np.nan,
                        "n_units": data["n_units"],
                        "n_tf": np.nan,
                        "n_imp": np.nan,
                        "n_trials": data["n_trials"],
                        "subtype": data["subtype"],
                        "hmm_state": data["hmm_state"],
                    })
            else:
                for bi, bc in enumerate(data["bin_centers"]):
                    rows.append({
                        "session_name": sname,
                        "stage": stage,
                        "session_idx": sidx,
                        "unit_group": group_name,
                        "bin_center": round(bc, 4),
                        "psth_tf": data["psth_tf"][bi],
                        "psth_imp": data["psth_imp"][bi],
                        "psth": np.nan,
                        "auc": data["auc"][bi],
                        "n_units": data["n_units"],
                        "n_tf": data["n_tf"],
                        "n_imp": data["n_imp"],
                        "n_trials": np.nan,
                        "subtype": "",
                        "hmm_state": "",
                    })
    results_df = pd.DataFrame(rows)
    cache_path = os.path.join(CACHE_DIR, f"fa_lick_aligned_divergence{cls_suffix}.csv")
    results_df.to_csv(cache_path, index=False)
    print(f"  Saved: {cache_path}")

    # ══════════════════════════════════════════════════════════════════
    # Compute grand averages for key groups
    # ══════════════════════════════════════════════════════════════════
    print("\n[Step 2] Computing grand averages and cluster tests...")
    grand = {}
    for gname in ["All", "TF-resp", "Non-TF", "Splitter", "Unilateral", "Omni"]:
        ga = compute_grand_average(valid, gname)
        if ga is not None:
            grand[gname] = ga
            print(f"  {gname}: {ga['n_sessions']} sessions, "
                  f"~{ga['mean_n_units']:.0f} units/session")

    # HMM state grand averages
    HMM_STATES = ["Disengaged", "Engaged", "Impulsive"]
    hmm_grand = {}
    for subtype_short, subtype_full in [("TF", "TF-triggered"), ("Imp", "Impulsive")]:
        for state in HMM_STATES:
            key = f"{subtype_short}|{state}"
            ga = compute_hmm_grand_average(valid, key)
            if ga is not None:
                hmm_grand[key] = ga
                print(f"  HMM [{key}]: {ga['n_sessions']} sessions, "
                      f"~{ga['mean_n_trials']:.0f} trials/session")

    # Cluster permutation tests on grand-average AUC
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
        sig_str = f"{n_sig_bins} sig bins" if n_sig_bins > 0 else "no significant clusters"
        print(f"  Cluster test [{gname}]: {sig_str}")
        for s, e, stat, pv in p_clusters:
            bc = grand[gname]["bin_centers"]
            print(f"    cluster [{bc[s]:.3f}, {bc[min(e-1, len(bc)-1)]:.3f}] s: "
                  f"stat={stat:.2f}, p={pv:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # FIGURE 33
    # ══════════════════════════════════════════════════════════════════
    print("\n[Step 3] Generating Figure 33...")
    fig = plt.figure(figsize=(22, 18))
    gs = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.32)
    stats_rows = []

    # ── Helper for PSTH panel ─────────────────────────────────────────
    def plot_psth_panel(ax, gname, title_letter, title_suffix):
        if gname not in grand:
            ax.text(0.5, 0.5, f"No data for {gname}", ha="center", va="center",
                    transform=ax.transAxes)
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

        ax.axvline(0, color="k", ls="--", lw=1, alpha=0.5, label="Lick")

        # Shade significant clusters
        if gname in cluster_results:
            for s, e, _, pv in cluster_results[gname]["p_clusters"]:
                if pv < CLUSTER_P_THRESH:
                    ax.axvspan(bc[s], bc[min(e-1, len(bc)-1)],
                               color="gold", alpha=0.25, zorder=0)

        ax.set_xlabel("Time relative to lick (s)")
        ax.set_ylabel("z-scored firing rate")
        ax.legend(fontsize=8, loc="upper left")
        n_units_str = f"{ga['mean_n_units']:.0f}"
        ax.set_title(f"{title_letter}. {title_suffix}\n"
                     f"(n={ga['n_sessions']} sessions, ~{n_units_str} units/sess)",
                     fontweight="bold")

    # ── Panel A: All units PSTH ───────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    plot_psth_panel(ax_a, "All", "A", "All units: TF-triggered vs Impulsive FA")

    # ── Panel B: TF-responsive PSTH ──────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    plot_psth_panel(ax_b, "TF-resp", "B", "TF-responsive units only")

    # ── Panel C: Non-TF PSTH ─────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    plot_psth_panel(ax_c, "Non-TF", "C", "Non-TF-responsive units")

    # ── Panel D: Time-resolved AUC by main groups ─────────────────────
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

        # Shade significant clusters
        if gname in cluster_results:
            for s, e, _, pv in cluster_results[gname]["p_clusters"]:
                if pv < CLUSTER_P_THRESH:
                    ax_d.axvspan(bc[s], bc[min(e-1, len(bc)-1)],
                                 color=group_colors[gname], alpha=0.08, zorder=0)

    ax_d.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.5)
    ax_d.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
    ax_d.set_xlabel("Time relative to lick (s)")
    ax_d.set_ylabel("AUC (TF-triggered vs Impulsive)")
    ax_d.legend(fontsize=8)
    ax_d.set_title("D. Time-resolved discriminability\n"
                   "(grand-average AUC across sessions)",
                   fontweight="bold")
    ax_d.set_ylim(0.42, 0.58)

    # ── Panel E: Per-tier AUC ─────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    tier_colors = {
        "Splitter": "#e67e22", "Unilateral": "#27ae60", "Omni": "#8e44ad",
    }
    for tier in ["Splitter", "Unilateral", "Omni"]:
        if tier not in grand:
            continue
        ga = grand[tier]
        bc = ga["bin_centers"]
        auc_sm = smooth(ga["mean_auc"])
        sem_sm = smooth(ga["sem_auc"])
        ax_e.plot(bc, auc_sm, color=tier_colors[tier], lw=2,
                  label=f"{tier} (n={ga['n_sessions']})")
        ax_e.fill_between(bc, auc_sm - sem_sm, auc_sm + sem_sm,
                          color=tier_colors[tier], alpha=0.15)

    ax_e.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.5)
    ax_e.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
    ax_e.set_xlabel("Time relative to lick (s)")
    ax_e.set_ylabel("AUC (TF-triggered vs Impulsive)")
    ax_e.legend(fontsize=8)
    ax_e.set_title("E. Per-tier discriminability\n"
                   "(Splitter / Unilateral / Omni)",
                   fontweight="bold")
    ax_e.set_ylim(0.42, 0.58)

    # ── Panel F: Summary text ─────────────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")

    n_sess_total = len(valid)
    total_tf = sum(sm.get("All", {}).get("n_tf", 0) for _, _, _, sm in valid)
    total_imp = sum(sm.get("All", {}).get("n_imp", 0) for _, _, _, sm in valid)
    total_units = sum(sm.get("All", {}).get("n_units", 0) for _, _, _, sm in valid)
    tf_units = sum(sm.get("TF-resp", {}).get("n_units", 0)
                   for _, _, _, sm in valid if "TF-resp" in sm)

    lines = [
        "F. Summary",
        "",
        f"Sessions analyzed: {n_sess_total}",
        f"Total FA trials: {total_tf + total_imp}",
        f"  TF-triggered: {total_tf}",
        f"  Impulsive: {total_imp}",
        f"Total unit-sessions: {total_units}",
        f"  TF-responsive: {tf_units}",
        "",
        f"Alignment: FA lick time (200 ms shift)",
        f"Window: [{LICK_WINDOW[0]}, {LICK_WINDOW[1]}] s",
        f"Bin size: {BIN_SIZE*1000:.0f} ms",
        f"Smoothing: {SMOOTH_SIGMA_MS:.0f} ms Gaussian",
        f"Normalization: z-score (baseline [{ZSCORE_BASELINE[0]}, {ZSCORE_BASELINE[1]}] s)",
        f"Cluster test: {N_PERM} permutations",
        "",
        "Cluster-based permutation tests:",
    ]

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

    # Record stats for CSV
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

    # ── Row 3: HMM state faceting ─────────────────────────────────────
    # Panels G, H, I: peri-lick PSTH for each FA subtype, faceted by HMM state
    HMM_STATES = ["Disengaged", "Engaged", "Impulsive"]
    # Colors per HMM state (for lines), with subtype distinguished by linestyle
    hmm_state_colors = HMM_STATE_COLORS
    subtype_ls = {"TF": "-", "Imp": "--"}
    subtype_labels = {"TF": "TF-triggered", "Imp": "Impulsive"}

    def plot_hmm_psth_panel(ax, title_letter, title_suffix):
        """Plot peri-lick PSTH for both FA subtypes, all HMM states, on one axis."""
        has_data = False
        for subtype_short in ("TF", "Imp"):
            for state in HMM_STATES:
                key = f"{subtype_short}|{state}"
                if key not in hmm_grand:
                    continue
                ga = hmm_grand[key]
                bc = ga["bin_centers"]
                m = smooth(ga["mean_psth"])
                se = smooth(ga["sem_psth"])
                ls = subtype_ls[subtype_short]
                color = hmm_state_colors[state]
                label = f"{subtype_labels[subtype_short]} / {state} (n={ga['n_sessions']})"
                ax.plot(bc, m, color=color, ls=ls, lw=1.5, label=label)
                ax.fill_between(bc, m - se, m + se, color=color, alpha=0.12)
                has_data = True

        if not has_data:
            ax.text(0.5, 0.5, "No HMM data", ha="center", va="center",
                    transform=ax.transAxes)
        ax.axvline(0, color="k", ls="--", lw=1, alpha=0.5, label="Lick")
        ax.set_xlabel("Time relative to lick (s)")
        ax.set_ylabel("z-scored firing rate")
        ax.legend(fontsize=6, loc="upper left", ncol=1)
        ax.set_title(f"{title_letter}. {title_suffix}", fontweight="bold")

    ax_g = fig.add_subplot(gs[2, 0])
    plot_hmm_psth_panel(ax_g, "G", "HMM-faceted PSTH: All FA licks\n"
                        "(solid=TF-triggered, dashed=Impulsive)")

    # Panel H: TF-triggered only, faceted by HMM state (individual state lines)
    ax_h = fig.add_subplot(gs[2, 1])
    has_data = False
    for state in HMM_STATES:
        for (short, full) in [("TF", "TF-triggered"), ("Imp", "Impulsive")]:
            key = f"{short}|{state}"
            if key not in hmm_grand:
                continue
            ga = hmm_grand[key]
            bc = ga["bin_centers"]
            m = smooth(ga["mean_psth"])
            se = smooth(ga["sem_psth"])
            ls = subtype_ls[short]
            color = hmm_state_colors[state]
            n_t = ga["mean_n_trials"]
            ax_h.plot(bc, m, color=color, ls=ls, lw=1.5,
                      label=f"{full[:3]}/{state[:3]} (~{n_t:.0f}/sess)")
            ax_h.fill_between(bc, m - se, m + se, color=color, alpha=0.10)
            has_data = True
    if not has_data:
        ax_h.text(0.5, 0.5, "No HMM data", ha="center", va="center",
                  transform=ax_h.transAxes)
    ax_h.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
    ax_h.set_xlabel("Time relative to lick (s)")
    ax_h.set_ylabel("z-scored firing rate")
    ax_h.legend(fontsize=6, loc="upper left", ncol=2)
    ax_h.set_title("H. HMM state × FA subtype (mean FR)\n"
                   "(solid=TF-trig, dashed=Impulsive)", fontweight="bold")

    # Panel I: Trial counts per HMM state per subtype across sessions
    ax_i = fig.add_subplot(gs[2, 2])
    ax_i.axis("off")
    hmm_lines = ["I. HMM state breakdown", ""]
    hmm_lines.append(f"{'Subtype':12s} {'State':14s} {'Sessions':>8s} {'~Trials/sess':>12s}")
    hmm_lines.append("-" * 50)
    for subtype_short, subtype_full in [("TF", "TF-triggered"), ("Imp", "Impulsive")]:
        for state in HMM_STATES:
            key = f"{subtype_short}|{state}"
            if key in hmm_grand:
                ga = hmm_grand[key]
                hmm_lines.append(
                    f"{subtype_full[:12]:12s} {state:14s} {ga['n_sessions']:>8d} "
                    f"{ga['mean_n_trials']:>12.1f}"
                )
            else:
                hmm_lines.append(
                    f"{subtype_full[:12]:12s} {state:14s} {'—':>8s} {'—':>12s}"
                )
    ax_i.text(0.05, 0.95, "\n".join(hmm_lines), transform=ax_i.transAxes,
              fontsize=8, va="top", ha="left", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan",
                        edgecolor="grey", alpha=0.8))

    # ── Save figure ───────────────────────────────────────────────────
    save_figure(fig, f"fig33_fa_lick_aligned_divergence{cls_suffix}", "07_advanced")
    plt.close(fig)

    # Save stats
    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        stats_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "figures", "07_advanced", f"fa_lick_aligned_divergence{cls_suffix}_stats.csv"
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
    print(f"FA trials: {total_tf} TF-triggered, {total_imp} Impulsive")
    print(f"Alignment: lick time, window [{LICK_WINDOW[0]}, {LICK_WINDOW[1]}] s")

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
