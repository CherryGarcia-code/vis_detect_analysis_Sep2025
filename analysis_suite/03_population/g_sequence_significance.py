"""Fig 14b: Sequence significance — RT-controlled tests for temporal structure.

Tests whether the sequential activation pattern observed in Fig14 (population
heatmap sorted by peak latency) represents genuine temporal tiling or is
driven by sorting artifacts / reaction-time jitter.

Four complementary tests:
  1. Split-half peak-order stability (Spearman ρ with circular-shift null)
  2. Cross-validated time decoding (Ridge regression, R²)
  3. RT-controlled time decoding (within narrow RT bins)
  4. Lick-aligned comparison (Change_ON vs Lick alignment)

Statistical framework:
  - Normalization: Per-unit z-score to shared baseline (-0.5, -0.05 s)
  - Time decoding: Ridge regression at trial level (avoids within-trial autocorrelation)
  - Null models: Circular-shift (split-half), circular time-shift (time decoding)
  - RT control: Within-bin analysis to isolate sequence structure from RT variability
  - Sequence metric: Cross-validated R² (variance of elapsed time explained)

Saves:
  figures/03_population/fig14b_sequence_significance.png
  figures/03_population/sequence_significance_stats.csv
"""

import os
import sys
import gc
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import STAGE_ORDER, STAGE_COLORS, CACHE_DIR
from loader import load_staging_manifest, load_session
from utils import (
    get_good_cluster_ids, build_population_tensor, smooth_psth,
    compute_zscore_normalized, bootstrap_ci,
)
from plotting import setup_style, save_figure

from visdetect.analysis.constants import DEFAULT_BIN_SIZE

setup_style()

# ── Parameters ──────────────────────────────────────────────────────
WINDOW = (-0.5, 1.0)
BIN_SIZE = 0.025          # 25 ms for sequence analysis (matches default)
BASELINE_WIN = (-0.5, -0.05)
RESPONSE_WIN = (0.0, 0.5)  # post-change window for peak finding
MIN_UNITS = 15
MIN_TRIALS_PER_CONDITION = 8
RT_BINS = [(0.15, 0.25), (0.25, 0.35), (0.35, 0.50), (0.50, 0.80)]
SEED = 42

# Default permutation counts (overridden by --fast / --full)
N_PERM_SPLIT = 100
N_PERM_DECODE = 50

CACHE_FILE = os.path.join(CACHE_DIR, "sequence_significance.csv")


# ── Helper: time decoding (Ridge regression) ────────────────────────

def time_decode_r2(tensor_z, bin_centers, response_mask, n_perm=0, seed=42):
    """Cross-validated time decoding from population activity.

    Predicts elapsed time (bin index) from the population vector at each bin.
    CV is at the trial level to avoid within-trial autocorrelation leakage.

    Parameters
    ----------
    tensor_z : ndarray, shape (n_trials, n_bins, n_units)
        Z-scored population tensor.
    bin_centers : ndarray
        Time bin centers.
    response_mask : ndarray (bool)
        Which bins to use for decoding (response window).
    n_perm : int
        Number of circular-shift permutations for null distribution.
    seed : int
        Random seed.

    Returns
    -------
    r2 : float
        Cross-validated R² (mean across folds).
    null_r2s : ndarray
        Null R² values from permutations (empty if n_perm=0).
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(seed)

    n_trials, n_bins, n_units = tensor_z.shape
    resp_bins = np.where(response_mask)[0]
    time_labels = bin_centers[resp_bins]  # continuous time values

    # Build design matrix: each row = one (trial, time_bin) observation
    # But CV at trial level: all time bins from a trial go together
    X_all = tensor_z[:, resp_bins, :]  # (n_trials, n_resp_bins, n_units)

    def _cv_r2(X_trials, y_time):
        """Compute 5-fold CV R² with trial-level splitting."""
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        r2s = []
        for train_idx, test_idx in kf.split(range(n_trials)):
            # Flatten: (trials × time_bins) rows, n_units features
            X_train = X_trials[train_idx].reshape(-1, n_units)
            X_test = X_trials[test_idx].reshape(-1, n_units)
            y_train = np.tile(y_time, len(train_idx))
            y_test = np.tile(y_time, len(test_idx))

            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Ridge regression
            model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # R²
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - y_test.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r2s.append(r2)

        return np.mean(r2s)

    # Observed R²
    r2 = _cv_r2(X_all, time_labels)

    # Null distribution: circular-shift each trial's time axis independently.
    # This preserves per-neuron temporal autocorrelation but destroys
    # coordinated population timing. A trial-permutation null is ineffective
    # here because time_labels is tiled identically for every trial.
    n_resp_bins = len(resp_bins)
    null_r2s = np.zeros(n_perm) if n_perm > 0 else np.array([])
    for i in range(n_perm):
        X_null = X_all.copy()
        for t in range(n_trials):
            shift = rng.randint(1, n_resp_bins)  # avoid shift=0
            X_null[t] = np.roll(X_null[t], shift, axis=0)
        null_r2s[i] = _cv_r2(X_null, time_labels)

    return r2, null_r2s


def split_half_peak_stability(tensor_z, bin_centers, response_mask, n_perm=500, seed=42):
    """Split-half peak-order stability with circular-shift null.

    Parameters
    ----------
    tensor_z : ndarray, shape (n_trials, n_bins, n_units)
    bin_centers : ndarray
    response_mask : ndarray (bool)
    n_perm : int
        Circular-shift permutations for null.
    seed : int

    Returns
    -------
    rho : float
        Spearman ρ of peak orders between halves.
    null_rhos : ndarray
        Null ρ values from circular-shift permutations.
    """
    rng = np.random.RandomState(seed)
    n_trials = tensor_z.shape[0]
    n_units = tensor_z.shape[2]
    resp_idx = np.where(response_mask)[0]

    # Split trials
    idx = rng.permutation(n_trials)
    half = n_trials // 2
    half1, half2 = idx[:half], idx[half:]

    # Mean PSTH per unit for each half
    psth1 = np.nanmean(tensor_z[half1], axis=0)  # (n_bins, n_units)
    psth2 = np.nanmean(tensor_z[half2], axis=0)

    # Peak latency in response window
    peak1 = np.argmax(psth1[resp_idx, :], axis=0)
    peak2 = np.argmax(psth2[resp_idx, :], axis=0)

    rho, _ = spearmanr(peak1, peak2)

    # Null: circular-shift each unit's PSTH independently, recompute
    null_rhos = np.zeros(n_perm)
    n_resp = len(resp_idx)
    for i in range(n_perm):
        shifted_psth = psth1[resp_idx, :].copy()
        for u in range(n_units):
            shift = rng.randint(0, n_resp)
            shifted_psth[:, u] = np.roll(shifted_psth[:, u], shift)
        shifted_peak = np.argmax(shifted_psth, axis=0)
        null_rhos[i] = spearmanr(shifted_peak, peak2)[0]

    return rho, null_rhos


# ── Per-session worker (supports parallel execution) ──────────────

def process_one_session(sname, stage, n_perm_split, n_perm_decode):
    """Process a single session. Returns dict or None if skipped."""
    try:
        sess = load_session(sname)
    except FileNotFoundError:
        return None

    good_ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
    if len(good_ids) < MIN_UNITS:
        del sess; gc.collect()
        return None

    # --- Identify Hit trials (go only) with valid RTs ---
    trials = sess.trials
    go_hit_idx = []
    hit_rts = []
    for i, t in enumerate(trials):
        oc = getattr(t, "trialoutcome", None)
        cs = getattr(t, "change_size", None) or 1.0
        if oc == "Hit" and cs > 1.01:
            rt_dict = getattr(t, "reactiontimes", None)
            if isinstance(rt_dict, dict):
                rt = rt_dict.get("RT", float("nan"))
                try:
                    rt = float(rt)
                except (TypeError, ValueError):
                    rt = float("nan")
            else:
                rt = float("nan")
            if np.isfinite(rt) and 0.1 < rt < 1.5:
                go_hit_idx.append(i)
                hit_rts.append(rt)

    go_miss_idx = [
        i for i, t in enumerate(trials)
        if getattr(t, "trialoutcome", None) == "Miss"
        and (getattr(t, "change_size", None) or 1.0) > 1.01
    ]

    if len(go_hit_idx) < MIN_TRIALS_PER_CONDITION:
        del sess; gc.collect()
        return None

    hit_rts = np.array(hit_rts)

    # --- Build population tensor (Hit trials) ---
    tensor_hit, bc, _ = build_population_tensor(
        sess, good_ids, event_name="Change_ON",
        window=WINDOW, bin_size=BIN_SIZE,
        trial_indices=go_hit_idx,
    )

    # Z-score normalize using shared baseline
    tensor_z = compute_zscore_normalized(tensor_hit, bc, BASELINE_WIN)

    response_mask = (bc >= RESPONSE_WIN[0]) & (bc < RESPONSE_WIN[1])
    n_units = tensor_z.shape[2]
    n_trials = tensor_z.shape[0]

    # ── Test 1: Split-half peak-order stability ─────────────────
    rho_split, null_rhos = split_half_peak_stability(
        tensor_z, bc, response_mask, n_perm=n_perm_split, seed=SEED,
    )
    p_split = (np.sum(np.abs(null_rhos) >= np.abs(rho_split)) + 1) / (n_perm_split + 1)

    # ── Test 2: Cross-validated time decoding ───────────────────
    r2_hit, null_r2s = time_decode_r2(
        tensor_z, bc, response_mask, n_perm=n_perm_decode, seed=SEED,
    )
    p_decode = (np.sum(null_r2s >= r2_hit) + 1) / (n_perm_decode + 1)
    null_mean = np.mean(null_r2s) if len(null_r2s) > 0 else 0.0
    null_sd = np.std(null_r2s) if len(null_r2s) > 0 else 0.0

    # ── Test 2b: Time decoding on Miss trials (control) ────────
    r2_miss = float("nan")
    if len(go_miss_idx) >= MIN_TRIALS_PER_CONDITION:
        try:
            tensor_miss, bc_m, _ = build_population_tensor(
                sess, good_ids, event_name="Change_ON",
                window=WINDOW, bin_size=BIN_SIZE,
                trial_indices=go_miss_idx,
            )
            tensor_miss_z = compute_zscore_normalized(tensor_miss, bc_m, BASELINE_WIN)
            r2_miss, _ = time_decode_r2(tensor_miss_z, bc_m, response_mask, n_perm=0)
        except Exception:
            pass

    # ── Test 3: RT-controlled time decoding ─────────────────────
    rt_bin_results = {}
    for rt_lo, rt_hi in RT_BINS:
        bin_mask = (hit_rts >= rt_lo) & (hit_rts < rt_hi)
        n_in_bin = int(bin_mask.sum())
        if n_in_bin < MIN_TRIALS_PER_CONDITION:
            rt_bin_results[f"{rt_lo:.2f}-{rt_hi:.2f}"] = (float("nan"), n_in_bin)
            continue
        tensor_rt = tensor_z[bin_mask]
        r2_rt, _ = time_decode_r2(tensor_rt, bc, response_mask, n_perm=0)
        rt_bin_results[f"{rt_lo:.2f}-{rt_hi:.2f}"] = (r2_rt, n_in_bin)

    # ── Test 4: Lick-aligned time decoding ──────────────────────
    r2_lick = float("nan")
    try:
        from visdetect.analysis.align import get_event_times_by_trial, align_spikes_to_events
        change_times = get_event_times_by_trial(sess, "Change_ON", enforce_valid_outcomes=True)
        lick_times = []
        valid_lick_idx = []
        for j, ti in enumerate(go_hit_idx):
            ct = change_times[ti] if ti < len(change_times) else float("nan")
            if np.isfinite(ct) and np.isfinite(hit_rts[j]):
                lick_times.append(ct + hit_rts[j])
                valid_lick_idx.append(j)

        if len(lick_times) >= MIN_TRIALS_PER_CONDITION:
            cluster_map = {c.cluster_id: c for c in sess.clusters}
            lick_window = (-0.5, 0.5)
            n_bins_lick = int((lick_window[1] - lick_window[0]) / BIN_SIZE)
            lick_tensor = np.zeros((len(lick_times), n_bins_lick, n_units))
            bc_lick = None

            for u_idx, cid in enumerate(good_ids):
                cluster = cluster_map.get(cid)
                if cluster is None:
                    continue
                mat, bc_l = align_spikes_to_events(
                    cluster.spike_times, lick_times,
                    window=lick_window, bin_size=BIN_SIZE,
                )
                if bc_lick is None:
                    bc_lick = bc_l
                lick_tensor[:, :, u_idx] = mat

            lick_baseline = (-0.5, -0.2)
            lick_z = compute_zscore_normalized(lick_tensor, bc_lick, lick_baseline)
            lick_resp_mask = (bc_lick >= -0.3) & (bc_lick < 0.1)

            r2_lick, _ = time_decode_r2(lick_z, bc_lick, lick_resp_mask, n_perm=0)
    except Exception:
        pass

    # ── Collect results ─────────────────────────────────────────
    row = {
        "session_name": sname,
        "stage": stage,
        "n_units": n_units,
        "n_hit_trials": n_trials,
        "n_miss_trials": len(go_miss_idx),
        "median_rt": float(np.median(hit_rts)),
        # Test 1: split-half
        "split_half_rho": rho_split,
        "split_half_p": p_split,
        "split_half_null_mean": float(np.mean(null_rhos)),
        "split_half_null_sd": float(np.std(null_rhos)),
        # Test 2: time decoding
        "time_decode_r2_hit": r2_hit,
        "time_decode_r2_miss": r2_miss,
        "time_decode_p": p_decode,
        "time_decode_null_mean": null_mean,
        "time_decode_null_sd": null_sd,
        "time_decode_null_97p": float(np.percentile(null_r2s, 97.5)) if len(null_r2s) > 0 else float("nan"),
        # Test 4: lick-aligned
        "lick_aligned_r2": r2_lick,
    }

    # Test 3: RT bins
    for bin_label, (r2_val, n_val) in rt_bin_results.items():
        row[f"rt_bin_{bin_label}_r2"] = r2_val
        row[f"rt_bin_{bin_label}_n"] = n_val

    del sess; gc.collect()
    return row


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sequence significance analysis")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Number of parallel workers (default: 1 = sequential)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: 20 split / 10 decode permutations")
    parser.add_argument("--full", action="store_true",
                        help="Full mode: 500 split / 200 decode permutations")
    parser.add_argument("--force", action="store_true",
                        help="Force recompute even if cache exists")
    args = parser.parse_args()

    # Set permutation counts based on mode
    if args.fast:
        n_perm_split, n_perm_decode = 20, 10
    elif args.full:
        n_perm_split, n_perm_decode = 500, 200
    else:
        n_perm_split, n_perm_decode = N_PERM_SPLIT, N_PERM_DECODE

    print(f"[03g] Sequence significance analysis "
          f"(split={n_perm_split}, decode={n_perm_decode}, workers={args.n_workers})...")

    # Check cache
    if os.path.exists(CACHE_FILE) and not args.force:
        print(f"  Loading cached results from {CACHE_FILE}")
        df = pd.read_csv(CACHE_FILE)
    else:
        manifest = load_staging_manifest(qc_only=True)
        session_list = [
            (int(mrow["session_name"]), mrow["stage"])
            for _, mrow in manifest.iterrows()
        ]

        rows = []
        if args.n_workers > 1:
            # Parallel execution
            print(f"  Processing {len(session_list)} sessions with {args.n_workers} workers...")
            with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
                futures = {
                    executor.submit(
                        process_one_session, sname, stage, n_perm_split, n_perm_decode
                    ): (sname, stage)
                    for sname, stage in session_list
                }
                for future in as_completed(futures):
                    sname, stage = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            rows.append(result)
                            print(f"  {sname} ({stage}): "
                                  f"{result['n_units']}u, {result['n_hit_trials']}t | "
                                  f"split rho={result['split_half_rho']:.3f} "
                                  f"(p={result['split_half_p']:.3f}), "
                                  f"R2={result['time_decode_r2_hit']:.3f} "
                                  f"(p={result['time_decode_p']:.3f})")
                        else:
                            print(f"  {sname} ({stage}): skipped")
                    except Exception as e:
                        print(f"  {sname} ({stage}): ERROR - {e}")
        else:
            # Sequential execution (original behavior)
            for sname, stage in session_list:
                print(f"  Session {sname} ({stage})...", end=" ")
                result = process_one_session(sname, stage, n_perm_split, n_perm_decode)
                if result is not None:
                    rows.append(result)
                    print(f"{result['n_units']}u, {result['n_hit_trials']}t | "
                          f"split rho={result['split_half_rho']:.3f} "
                          f"(p={result['split_half_p']:.3f}), "
                          f"R2={result['time_decode_r2_hit']:.3f} "
                          f"(p={result['time_decode_p']:.3f})")
                else:
                    print("skipped")

        # ── Save cache ──────────────────────────────────────────────
        df = pd.DataFrame(rows)
    df.to_csv(CACHE_FILE, index=False)
    print(f"\n  Saved {len(df)} sessions to {CACHE_FILE}")

    if len(df) == 0:
        print("  No data. Exiting.")
        return

    # Sort by stage (Learning first, then Expert) then chronologically
    from visdetect.analysis.config import parse_session_date
    _stage_rank = {s: i for i, s in enumerate(STAGE_ORDER)}
    df["_stage_rank"] = df["stage"].map(_stage_rank).fillna(99).astype(int)
    df["_date_sort"] = df["session_name"].apply(
        lambda x: parse_session_date(int(x))
    )
    df = df.sort_values(["_stage_rank", "_date_sort"]).reset_index(drop=True)
    df.drop(columns=["_stage_rank", "_date_sort"], inplace=True)

    # ── Figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

    # --- Panel A: Split-half ρ by stage ---
    ax_a = fig.add_subplot(gs[0, 0])
    for stage in STAGE_ORDER:
        sdf = df[df["stage"] == stage]
        if len(sdf) > 0:
            ax_a.scatter(
                [stage] * len(sdf), sdf["split_half_rho"],
                color=STAGE_COLORS[stage], alpha=0.7, s=50, zorder=3,
            )
            median_val = sdf["split_half_rho"].median()
            ax_a.hlines(median_val, -0.3, 0.3, transform=ax_a.get_yaxis_transform(),
                        colors=STAGE_COLORS[stage], linewidths=2, zorder=4)
    ax_a.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax_a.set_ylabel("Split-half Spearman ρ")
    ax_a.set_title("A. Peak-order stability")

    # --- Panel B: Time decoding R² (Hit vs Miss) ---
    ax_b = fig.add_subplot(gs[0, 1])
    for stage in STAGE_ORDER:
        sdf = df[df["stage"] == stage]
        if len(sdf) == 0:
            continue
        ax_b.scatter(sdf["time_decode_r2_hit"], sdf["time_decode_r2_miss"],
                     color=STAGE_COLORS[stage], alpha=0.7, s=50, label=stage)
    lims = [
        min(df["time_decode_r2_hit"].min(), df["time_decode_r2_miss"].min()) - 0.02,
        max(df["time_decode_r2_hit"].max(), df["time_decode_r2_miss"].max()) + 0.02,
    ]
    ax_b.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
    ax_b.set_xlabel("R² (Hit trials)")
    ax_b.set_ylabel("R² (Miss trials)")
    ax_b.set_title("B. Time decoding: Hit vs Miss")
    ax_b.legend(fontsize=8)

    # --- Panel C: Time decoding R² across sessions ---
    ax_c = fig.add_subplot(gs[0, 2])
    session_idx = range(len(df))
    for i, (_, row_data) in enumerate(df.iterrows()):
        color = STAGE_COLORS.get(row_data["stage"], "gray")
        ax_c.bar(i, row_data["time_decode_r2_hit"], color=color, alpha=0.7)
        # Null threshold: use stored 97.5th percentile if available, else mean+2SD
        if "time_decode_null_97p" in row_data and np.isfinite(row_data["time_decode_null_97p"]):
            null_upper = row_data["time_decode_null_97p"]
        else:
            null_upper = row_data["time_decode_null_mean"] + 2 * row_data["time_decode_null_sd"]
        ax_c.hlines(null_upper, i - 0.4, i + 0.4, colors="red", linewidths=0.8,
                     linestyles="--", alpha=0.6)
    # Add stage boundary marker
    n_learning = (df["stage"] == "Learning").sum()
    if 0 < n_learning < len(df):
        ax_c.axvline(n_learning - 0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax_c.text(n_learning / 2, ax_c.get_ylim()[1] * 0.95, "Learning",
                  ha="center", va="top", fontsize=7, color=STAGE_COLORS.get("Learning", "gray"))
        ax_c.text(n_learning + (len(df) - n_learning) / 2, ax_c.get_ylim()[1] * 0.95, "Expert",
                  ha="center", va="top", fontsize=7, color=STAGE_COLORS.get("Expert", "gray"))
    ax_c.set_xlabel("Session (chronological)")
    ax_c.set_ylabel("Time decoding R²")
    ax_c.set_title("C. Per-session R² (red = null 97.5th)")

    # --- Panel D: RT-bin analysis ---
    ax_d = fig.add_subplot(gs[1, 0])
    rt_bin_labels = [f"{lo:.2f}-{hi:.2f}" for lo, hi in RT_BINS]
    for stage in STAGE_ORDER:
        sdf = df[df["stage"] == stage]
        if len(sdf) == 0:
            continue
        means = []
        sems = []
        for bl in rt_bin_labels:
            col = f"rt_bin_{bl}_r2"
            vals = sdf[col].dropna()
            means.append(vals.mean() if len(vals) > 0 else float("nan"))
            sems.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        x = np.arange(len(rt_bin_labels))
        ax_d.errorbar(x, means, yerr=sems, marker="o", color=STAGE_COLORS[stage],
                      label=stage, capsize=3, linewidth=1.5)
    ax_d.set_xticks(range(len(rt_bin_labels)))
    ax_d.set_xticklabels([f"{lo*1000:.0f}-{hi*1000:.0f}" for lo, hi in RT_BINS], fontsize=8)
    ax_d.set_xlabel("RT bin (ms)")
    ax_d.set_ylabel("Time decoding R²")
    ax_d.set_title("D. Sequence within RT bins")
    ax_d.legend(fontsize=8)
    ax_d.axhline(0, color="gray", linestyle=":", linewidth=0.8)

    # --- Panel E: Change_ON vs Lick-aligned R² ---
    ax_e = fig.add_subplot(gs[1, 1])
    valid = df.dropna(subset=["lick_aligned_r2", "time_decode_r2_hit"])
    for stage in STAGE_ORDER:
        sdf = valid[valid["stage"] == stage]
        if len(sdf) > 0:
            ax_e.scatter(sdf["time_decode_r2_hit"], sdf["lick_aligned_r2"],
                         color=STAGE_COLORS[stage], alpha=0.7, s=50, label=stage)
    if len(valid) > 0:
        lims_e = [
            min(valid["time_decode_r2_hit"].min(), valid["lick_aligned_r2"].min()) - 0.02,
            max(valid["time_decode_r2_hit"].max(), valid["lick_aligned_r2"].max()) + 0.02,
        ]
        ax_e.plot(lims_e, lims_e, "k--", linewidth=0.8, alpha=0.5)
    ax_e.set_xlabel("R² (Change_ON aligned)")
    ax_e.set_ylabel("R² (Lick aligned)")
    ax_e.set_title("E. Stimulus vs motor alignment")
    ax_e.legend(fontsize=8)

    # --- Panel F: Effect size summary ---
    ax_f = fig.add_subplot(gs[1, 2])
    # Show per-stage summary statistics
    summary_data = []
    for stage in STAGE_ORDER:
        sdf = df[df["stage"] == stage]
        if len(sdf) == 0:
            continue
        summary_data.append({
            "stage": stage,
            "n_sess": len(sdf),
            "median_split_rho": sdf["split_half_rho"].median(),
            "median_r2_hit": sdf["time_decode_r2_hit"].median(),
            "median_r2_miss": sdf["time_decode_r2_miss"].median(),
            "frac_sig_split": (sdf["split_half_p"] < 0.05).mean(),
            "frac_sig_decode": (sdf["time_decode_p"] < 0.05).mean(),
        })
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        cell_text = []
        for _, r in summary_df.iterrows():
            cell_text.append([
                r["stage"], f"{r['n_sess']:.0f}",
                f"{r['median_split_rho']:.3f}", f"{r['frac_sig_split']:.0%}",
                f"{r['median_r2_hit']:.3f}", f"{r['median_r2_miss']:.3f}",
                f"{r['frac_sig_decode']:.0%}",
            ])
        cols = ["Stage", "N", "Med ρ", "% sig ρ", "Med R²\nHit", "Med R²\nMiss", "% sig R²"]
        table = ax_f.table(cellText=cell_text, colLabels=cols, loc="center",
                          cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.5)
    ax_f.axis("off")
    ax_f.set_title("F. Summary statistics")

    # --- Panel G: R² by learning stage (box plot) ---
    ax_g = fig.add_subplot(gs[2, 0])
    stage_data = []
    stage_labels_plot = []
    stage_colors_plot = []
    for stage in STAGE_ORDER:
        vals = df.loc[df["stage"] == stage, "time_decode_r2_hit"].dropna().values
        if len(vals) > 0:
            stage_data.append(vals)
            stage_labels_plot.append(stage)
            stage_colors_plot.append(STAGE_COLORS[stage])
    if len(stage_data) >= 2:
        bp = ax_g.boxplot(stage_data, labels=stage_labels_plot, patch_artist=True)
        for patch, color in zip(bp["boxes"], stage_colors_plot):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        # Mann-Whitney if 2 stages
        from scipy.stats import mannwhitneyu
        if len(stage_data) == 2:
            U, p = mannwhitneyu(stage_data[0], stage_data[1], alternative="two-sided")
            r_rb = 1 - 2 * U / (len(stage_data[0]) * len(stage_data[1]))
            ax_g.set_title(f"G. R² by stage (U={U:.0f}, p={p:.3f}, r={r_rb:.2f})")
        else:
            ax_g.set_title("G. R² by stage")
    ax_g.set_ylabel("Time decoding R²")

    # --- Panel H: Split-half ρ distribution ---
    ax_h = fig.add_subplot(gs[2, 1])
    ax_h.hist(df["split_half_rho"].dropna(), bins=15, color="steelblue",
              edgecolor="white", alpha=0.7)
    ax_h.axvline(0, color="red", linestyle="--", linewidth=1)
    med_rho = df["split_half_rho"].median()
    ax_h.axvline(med_rho, color="navy", linestyle="-", linewidth=1.5,
                 label=f"Median ρ = {med_rho:.3f}")
    ax_h.set_xlabel("Split-half Spearman ρ")
    ax_h.set_ylabel("Sessions")
    ax_h.set_title("H. Peak-order stability distribution")
    ax_h.legend(fontsize=9)

    # --- Panel I: R² vs number of units ---
    ax_i = fig.add_subplot(gs[2, 2])
    ax_i.scatter(df["n_units"], df["time_decode_r2_hit"],
                 c=[STAGE_COLORS.get(s, "gray") for s in df["stage"]],
                 alpha=0.7, s=50)
    rho_nu, p_nu = spearmanr(df["n_units"], df["time_decode_r2_hit"])
    ax_i.set_xlabel("Number of units")
    ax_i.set_ylabel("Time decoding R²")
    ax_i.set_title(f"I. R² vs population size (ρ={rho_nu:.2f}, p={p_nu:.3f})")

    # ── Statistics CSV ──────────────────────────────────────────────
    stats = []

    # Overall split-half
    from scipy.stats import wilcoxon
    all_rho = df["split_half_rho"].dropna().values
    if len(all_rho) >= 5:
        W, p_w = wilcoxon(all_rho)
        z_w = (W - len(all_rho) * (len(all_rho) + 1) / 4) / np.sqrt(
            len(all_rho) * (len(all_rho) + 1) * (2 * len(all_rho) + 1) / 24)
        r_eff = abs(z_w) / np.sqrt(len(all_rho))
        stats.append({
            "test": "split_half_rho_vs_zero",
            "statistic_name": "W", "statistic_value": W,
            "p_value": p_w, "effect_size_name": "r",
            "effect_size_value": r_eff,
            "n": len(all_rho),
            "interpretation": "Strong" if r_eff > 0.5 else "Medium" if r_eff > 0.3 else "Small",
        })

    # Overall time decoding vs null
    all_r2 = df["time_decode_r2_hit"].dropna().values
    all_null = df["time_decode_null_mean"].dropna().values
    if len(all_r2) >= 5 and len(all_null) >= 5:
        W2, p_w2 = wilcoxon(all_r2, all_null)
        stats.append({
            "test": "time_decode_r2_vs_null",
            "statistic_name": "W", "statistic_value": W2,
            "p_value": p_w2, "effect_size_name": "median_r2",
            "effect_size_value": np.median(all_r2),
            "n": len(all_r2),
            "interpretation": f"Median R²={np.median(all_r2):.3f} vs null={np.median(all_null):.3f}",
        })

    # Hit vs Miss time decoding
    paired_hm = df.dropna(subset=["time_decode_r2_hit", "time_decode_r2_miss"])
    hit_r2 = paired_hm["time_decode_r2_hit"].values
    miss_r2 = paired_hm["time_decode_r2_miss"].values
    if len(hit_r2) >= 5:
        W3, p_w3 = wilcoxon(hit_r2, miss_r2)
        stats.append({
            "test": "time_decode_hit_vs_miss",
            "statistic_name": "W", "statistic_value": W3,
            "p_value": p_w3, "effect_size_name": "median_diff",
            "effect_size_value": np.median(hit_r2) - np.median(miss_r2),
            "n": len(hit_r2),
            "interpretation": "Hit > Miss" if np.median(hit_r2) > np.median(miss_r2) else "Miss >= Hit",
        })

    # Change_ON vs Lick alignment
    valid_both = df.dropna(subset=["time_decode_r2_hit", "lick_aligned_r2"])
    if len(valid_both) >= 5:
        W4, p_w4 = wilcoxon(valid_both["time_decode_r2_hit"], valid_both["lick_aligned_r2"])
        stats.append({
            "test": "change_vs_lick_alignment",
            "statistic_name": "W", "statistic_value": W4,
            "p_value": p_w4, "effect_size_name": "median_diff",
            "effect_size_value": float(
                valid_both["time_decode_r2_hit"].median() - valid_both["lick_aligned_r2"].median()
            ),
            "n": len(valid_both),
            "interpretation": "Stimulus-locked" if valid_both["time_decode_r2_hit"].median() > valid_both["lick_aligned_r2"].median() else "Lick-locked",
        })

    stats_df = pd.DataFrame(stats)

    # ── Save ────────────────────────────────────────────────────────
    save_figure(fig, "fig14b_sequence_significance", "03_population")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "03_population", "sequence_significance_stats.csv",
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats ({len(stats_df)} tests)")
    for _, row_s in stats_df.iterrows():
        print(f"    {row_s['test']}: {row_s['statistic_name']}="
              f"{row_s['statistic_value']:.2f}, p={row_s['p_value']:.4f}, "
              f"{row_s['effect_size_name']}={row_s['effect_size_value']:.3f}")


if __name__ == "__main__":
    main()
