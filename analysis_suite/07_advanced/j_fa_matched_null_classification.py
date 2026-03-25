"""Fig32j: FA classification via between-trial matched null.

For each session, builds a null distribution of pre-lick TF features
from the baseline periods of non-FA (hit/miss) trials. Each FA trial
is then tested against this session-specific null.

Two test statistics:
  1. max(log2 TF) in [-1.0, -0.4]s before lick  (magnitude)
  2. max(dTF/dt) in the same window                (onset sharpness)

If either p < 0.05 the lick is classified as 'Stimulus-driven';
otherwise 'Impulsive'.

Saves:
  cache/fa_classification_matched_null.csv
  figures/07_advanced/fig32j_fa_matched_null.png
  figures/07_advanced/fa_matched_null_stats.csv
"""

import os
import sys
import gc
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency, spearmanr
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    STAGE_ORDER, STAGE_COLORS, CACHE_DIR,
    HMM_STATE_ORDER, HMM_STATE_COLORS,
    FA_SUBTYPE_COLORS,
)
from loader import (
    load_staging_manifest, load_session,
    load_hmm_assignments,
)
from plotting import setup_style, save_figure, add_stage_background
from _fa_helpers import (
    extract_baseline_tf_trace,
    extract_lta_segment,
    original_threshold_classify,
)

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_root, "src"))
from visdetect.analysis.constants import TF_FAST_THRESH_LOG2

setup_style()
warnings.filterwarnings("ignore")

# ── Parameters ────────────────────────────────────────────────────────
BASELINE_STRIDE = 3
SAMPLE_PERIOD = 0.05
CLASSIFY_WINDOW = (-1.0, -0.4)  # window relative to lick for classification features
MIN_RT_FOR_LTA = 2.0
ALPHA = 0.05
N_NULL = 1000                 # null windows to sample per session
MIN_SOURCE_TRIALS = 10        # minimum non-FA trials for valid null

# LTA parameters (match MATLAB LickTrigAvg.m)
LTA_HISTORY = 60
LTA_POST = 20
LTA_N_SAMPLES = LTA_HISTORY + LTA_POST
LTA_T_AXIS = (np.arange(LTA_N_SAMPLES) - (LTA_HISTORY - 1)) * SAMPLE_PERIOD
LTA_SMOOTH_SIGMA = 1.0

# Classification window in samples
CLS_WIN_SAMPLES = int(round((CLASSIFY_WINDOW[1] - CLASSIFY_WINDOW[0]) / SAMPLE_PERIOD))


# =====================================================================
# Core functions
# =====================================================================


def compute_window_stats(log2_segment):
    """Compute max(log2 TF) and max(dTF/dt) on a fixed-length segment.

    Input: log2 TF values of length CLS_WIN_SAMPLES.
    Returns (max_log2tf, max_dtf_dt).
    """
    max_log2tf = float(np.max(log2_segment))
    if len(log2_segment) >= 2:
        dtf = np.diff(log2_segment) / SAMPLE_PERIOD
        max_dtf_dt = float(np.max(dtf))
    else:
        max_dtf_dt = 0.0
    return max_log2tf, max_dtf_dt


def build_session_null(session, n_null=N_NULL, rng=None):
    """Build null distribution from non-FA trial baselines.

    Samples random windows from the pre-change baseline of hit/miss trials.
    Returns (null_max_tf, null_max_dtf, n_source_trials).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Collect valid baseline source segments
    sources = []  # list of (log2_tf_array, valid_start_idx, valid_end_idx)
    min_start_idx = int(round(MIN_RT_FOR_LTA / SAMPLE_PERIOD))

    for t in session.trials:
        outcome = getattr(t, "trialoutcome", None)
        if outcome not in ("Hit", "Miss"):
            continue

        bv = getattr(t, "baseline_values", None)
        if bv is None:
            continue

        arr = np.array(bv).flatten()
        if BASELINE_STRIDE > 1:
            arr = arr[::BASELINE_STRIDE]

        n_seen = getattr(t, "n_seen", None)
        if isinstance(n_seen, (int, np.integer)) and n_seen is not None and n_seen > 0:
            arr = arr[: int(n_seen)]

        if len(arr) < min_start_idx + CLS_WIN_SAMPLES + 1:
            continue

        log2_tf = np.log2(np.clip(arr.astype(float), 0.01, None))

        # Valid range: from MIN_RT_FOR_LTA to change_time (or end of trace)
        change_time = getattr(t, "change_time", None)
        if change_time is not None and not np.isnan(change_time):
            max_end_idx = int(round(change_time / SAMPLE_PERIOD))
        else:
            max_end_idx = len(log2_tf)

        valid_end = max_end_idx - CLS_WIN_SAMPLES
        if valid_end <= min_start_idx:
            continue

        sources.append((log2_tf, min_start_idx, valid_end))

    n_sources = len(sources)
    if n_sources < MIN_SOURCE_TRIALS:
        return None, None, n_sources

    # Weighted sampling by available range length
    ranges = np.array([end - start for _, start, end in sources], dtype=float)
    weights = ranges / ranges.sum()

    null_max_tf = np.empty(n_null)
    null_max_dtf = np.empty(n_null)

    trial_choices = rng.choice(n_sources, size=n_null, p=weights)
    for ni in range(n_null):
        log2_tf, start, end = sources[trial_choices[ni]]
        idx = rng.integers(start, end)
        seg = log2_tf[idx: idx + CLS_WIN_SAMPLES]
        null_max_tf[ni], null_max_dtf[ni] = compute_window_stats(seg)

    return null_max_tf, null_max_dtf, n_sources


def compute_fa_pre_lick_stats(log2_tf, rt):
    """Compute test statistics in the classification window for one FA trial.

    Returns (max_log2tf, max_dtf_dt) or (None, None).
    """
    sample_times = np.arange(len(log2_tf)) * SAMPLE_PERIOD
    sample_times_rel = sample_times - rt

    mask = (sample_times_rel >= CLASSIFY_WINDOW[0]) & (sample_times_rel < CLASSIFY_WINDOW[1])
    if mask.sum() < 2:
        return None, None

    seg = log2_tf[mask]
    return compute_window_stats(seg)


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 60)
    print("[07j] FA classification: between-trial matched null")
    print("=" * 60)

    manifest = load_staging_manifest(qc_only=True)
    hmm_assign = load_hmm_assignments()
    print(f"  {len(manifest)} QC-passed sessions")

    rng = np.random.default_rng(42)
    all_records = []
    lta_by_subtype = {"Stimulus-driven": [], "Impulsive": []}

    print(f"\n[Step 1] Classifying FA trials (session null: {N_NULL} windows)...")

    for _, row in manifest.iterrows():
        sname = int(row["session_name"])
        stage = row["stage"]
        sidx = row["session_idx"]

        print(f"  Session {sname} ({stage})...", end=" ", flush=True)
        try:
            sess = load_session(sname)
        except FileNotFoundError:
            print("pkl not found")
            continue

        # Build session-specific null
        null_max_tf, null_max_dtf, n_src = build_session_null(sess, N_NULL, rng)
        if null_max_tf is None:
            print(f"skipped (only {n_src} source trials)")
            del sess
            gc.collect()
            continue

        # HMM states
        sess_hmm = hmm_assign[hmm_assign["session_name"] == sname]
        trial_to_hmm = {}
        if len(sess_hmm) > 0:
            for _, hr in sess_hmm.iterrows():
                trial_to_hmm[int(hr["trial_idx"])] = hr["hmm_state_label"]

        n_stim = 0
        n_imp = 0

        for i, t in enumerate(sess.trials):
            outcome = getattr(t, "trialoutcome", None)
            if outcome != "FA":
                continue

            log2_tf, n_valid, rt = extract_baseline_tf_trace(
                t, BASELINE_STRIDE, SAMPLE_PERIOD, MIN_RT_FOR_LTA
            )
            if log2_tf is None or rt is None:
                continue

            obs_tf, obs_dtf = compute_fa_pre_lick_stats(log2_tf, rt)
            if obs_tf is None:
                continue

            # Test against null
            p_tf = (np.sum(null_max_tf >= obs_tf) + 1) / (N_NULL + 1)
            p_dtf = (np.sum(null_max_dtf >= obs_dtf) + 1) / (N_NULL + 1)

            if p_tf < ALPHA or p_dtf < ALPHA:
                subtype = "Stimulus-driven"
                n_stim += 1
            else:
                subtype = "Impulsive"
                n_imp += 1

            orig = original_threshold_classify(
                log2_tf, rt, CLASSIFY_WINDOW, SAMPLE_PERIOD, TF_FAST_THRESH_LOG2
            )

            cs = getattr(t, "change_size", 1.0) or 1.0
            hmm_state = trial_to_hmm.get(i, "Unknown")

            all_records.append({
                "session_name": sname,
                "stage": stage,
                "session_idx": sidx,
                "trial_idx": i,
                "fa_subtype": subtype,
                "p_value_max_tf": float(p_tf),
                "p_value_dtf": float(p_dtf),
                "max_log2tf": obs_tf,
                "max_dtf_dt": obs_dtf,
                "rt": rt,
                "hmm_state": hmm_state,
                "original_subtype": orig,
                "change_size": cs,
                "is_go": cs > 1.01,
                "n_null_source_trials": n_src,
            })

            lta_seg = extract_lta_segment(
                t, BASELINE_STRIDE, SAMPLE_PERIOD, MIN_RT_FOR_LTA,
                LTA_HISTORY, LTA_POST
            )
            if lta_seg is not None:
                lta_by_subtype[subtype].append(lta_seg)

        print(f"Stim-driven={n_stim}, Impulsive={n_imp} (null from {n_src} trials)")
        del sess
        gc.collect()

    fa_df = pd.DataFrame(all_records)
    print(f"\n  Total classified: {len(fa_df)}")
    if len(fa_df) == 0:
        print("  ERROR: No FA trials classified. Check trial outcome labels.")
        return
    print(f"  Stimulus-driven: {(fa_df['fa_subtype'] == 'Stimulus-driven').sum()}")
    print(f"  Impulsive: {(fa_df['fa_subtype'] == 'Impulsive').sum()}")

    cache_path = os.path.join(CACHE_DIR, "fa_classification_matched_null.csv")
    fa_df.to_csv(cache_path, index=False)
    print(f"  Saved: {cache_path}")

    # ══════════════════════════════════════════════════════════════════
    # FIGURE
    # ══════════════════════════════════════════════════════════════════
    print("\n[Step 2] Generating Figure 32j...")
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.35)
    stats = []

    # ── Panel A: LTA by subtype ──────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    for subtype in ["Stimulus-driven", "Impulsive"]:
        traces = lta_by_subtype.get(subtype, [])
        if len(traces) < 5:
            continue
        arr = np.array(traces)
        mean_tr = np.nanmean(arr, axis=0)
        sem_tr = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
        mean_sm = gaussian_filter1d(mean_tr, sigma=LTA_SMOOTH_SIGMA)
        ci95_sm = gaussian_filter1d(sem_tr * 1.96, sigma=LTA_SMOOTH_SIGMA)
        ax_a.plot(LTA_T_AXIS, mean_sm, color=FA_SUBTYPE_COLORS[subtype],
                  linewidth=2, label=f"{subtype} (n={len(traces)})")
        ax_a.fill_between(LTA_T_AXIS, mean_sm - ci95_sm, mean_sm + ci95_sm,
                          color=FA_SUBTYPE_COLORS[subtype], alpha=0.2)
    ax_a.axvline(0, color="k", ls="--", lw=1, alpha=0.7, label="Lick")
    ax_a.axhline(0, color="grey", ls=":", lw=0.5)
    ax_a.axhline(TF_FAST_THRESH_LOG2, color="#e74c3c", ls="--", lw=0.8, alpha=0.3)
    ax_a.set_xlabel("Time relative to lick (s)")
    ax_a.set_ylabel("log2(TF ratio)")
    ax_a.legend(fontsize=7, loc="upper left")
    ax_a.set_xlim(LTA_T_AXIS[0], LTA_T_AXIS[-1])
    ax_a.set_title("A. LTA by subtype (matched null)", fontweight="bold")

    # ── Panel B: P-value distribution ────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    p_combined = fa_df[["p_value_max_tf", "p_value_dtf"]].min(axis=1).values
    bins = np.linspace(0, 1, 40)
    ax_b.hist(p_combined, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
    ax_b.axvline(ALPHA, color="red", ls="--", lw=1.5, label=f"alpha={ALPHA}")
    n_sig = np.sum(p_combined < ALPHA)
    n_tot = len(p_combined)
    ax_b.text(0.98, 0.98,
              f"Significant: {n_sig}/{n_tot} ({100 * n_sig / n_tot:.1f}%)",
              transform=ax_b.transAxes, fontsize=8, ha="right", va="top")
    ax_b.set_xlabel("min(p_max_tf, p_dtf)")
    ax_b.set_ylabel("Count")
    ax_b.legend(fontsize=7)
    ax_b.set_title("B. P-value distribution", fontweight="bold")

    # ── Panel C: Scatter max(log2 TF) vs max(dTF/dt) ────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    for subtype in ["Stimulus-driven", "Impulsive"]:
        sub = fa_df[fa_df["fa_subtype"] == subtype]
        ax_c.scatter(sub["max_log2tf"], sub["max_dtf_dt"],
                     c=FA_SUBTYPE_COLORS[subtype], s=8, alpha=0.4,
                     label=f"{subtype} (n={len(sub)})", edgecolors="none")
    ax_c.axvline(TF_FAST_THRESH_LOG2, color="k", ls="--", lw=1, alpha=0.5,
                 label=f"Old thresh ({TF_FAST_THRESH_LOG2})")
    ax_c.set_xlabel("max log2(TF) in [-1.0, -0.4]s")
    ax_c.set_ylabel("max dTF/dt (log2/s)")
    ax_c.legend(fontsize=7, loc="upper left")
    ax_c.set_title("C. Feature scatter by classification", fontweight="bold")

    # ── Panel D: Agreement with original threshold ───────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    valid = fa_df.dropna(subset=["original_subtype"])
    if len(valid) > 10:
        new_pos = (valid["fa_subtype"] == "Stimulus-driven").values
        old_pos = (valid["original_subtype"] == "TF-triggered").values
        tp = np.sum(new_pos & old_pos)
        fp = np.sum(new_pos & ~old_pos)
        fn = np.sum(~new_pos & old_pos)
        tn = np.sum(~new_pos & ~old_pos)
        conf = np.array([[tp, fp], [fn, tn]])

        agree = (tp + tn) / len(valid)
        p_o = agree
        p_e = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (len(valid) ** 2)
        kappa = (p_o - p_e) / (1 - p_e) if p_e < 1.0 else 1.0

        stats.append({"test": "agreement_with_threshold",
                       "agreement_pct": agree * 100, "cohen_kappa": kappa,
                       "n": len(valid)})

        ax_d.imshow(conf, cmap="Blues", interpolation="nearest")
        labels_new = ["Stim-driven", "Impulsive"]
        labels_old = ["TF-triggered", "Impulsive"]
        for ii in range(2):
            for jj in range(2):
                ax_d.text(jj, ii, str(conf[ii, jj]),
                          ha="center", va="center", fontsize=14, fontweight="bold")
        ax_d.set_xticks([0, 1])
        ax_d.set_xticklabels(labels_old, fontsize=9)
        ax_d.set_yticks([0, 1])
        ax_d.set_yticklabels(labels_new, fontsize=9)
        ax_d.set_xlabel("Original threshold")
        ax_d.set_ylabel("Matched null")
        ax_d.set_title(f"D. Agreement: {agree:.1%}, kappa={kappa:.2f}", fontweight="bold")
    else:
        ax_d.text(0.5, 0.5, "Insufficient data", transform=ax_d.transAxes,
                  ha="center", va="center")
        ax_d.set_title("D. Agreement with threshold", fontweight="bold")

    # ── Panel E: Subtype fractions across sessions ───────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    sess_counts = fa_df.groupby(
        ["session_name", "session_idx", "stage", "fa_subtype"]
    ).size().unstack(fill_value=0)
    sess_fracs = sess_counts.div(sess_counts.sum(axis=1), axis=0)
    sess_fracs = sess_fracs.reset_index().sort_values("session_idx")

    for subtype in ["Stimulus-driven", "Impulsive"]:
        if subtype in sess_fracs.columns:
            ax_e.scatter(sess_fracs["session_idx"], sess_fracs[subtype],
                         c=FA_SUBTYPE_COLORS[subtype], s=40, label=subtype,
                         edgecolors="k", linewidth=0.3, zorder=3)
            if len(sess_fracs) >= 5:
                z = np.polyfit(sess_fracs["session_idx"], sess_fracs[subtype], 1)
                x_fit = np.linspace(sess_fracs["session_idx"].min(),
                                    sess_fracs["session_idx"].max(), 50)
                ax_e.plot(x_fit, np.polyval(z, x_fit),
                          color=FA_SUBTYPE_COLORS[subtype], ls="--", lw=1, alpha=0.5)

    add_stage_background(ax_e, manifest, alpha=0.06)
    ax_e.set_xlabel("Session index")
    ax_e.set_ylabel("Fraction of FA trials")
    ax_e.set_ylim(-0.05, 1.05)
    ax_e.legend(fontsize=7, loc="center right")

    if "Stimulus-driven" in sess_fracs.columns and len(sess_fracs) >= 5:
        r, p = spearmanr(sess_fracs["session_idx"], sess_fracs["Stimulus-driven"])
        stats.append({"test": "stim_driven_frac_vs_session", "rho": r, "p": p,
                       "n": len(sess_fracs)})
        ax_e.text(0.02, 0.98, f"Stim-driven frac: rho={r:.3f}, p={p:.3f}",
                  transform=ax_e.transAxes, fontsize=7, va="top")

    ax_e.set_title("E. FA subtype fractions across learning", fontweight="bold")

    # ── Panel F: HMM state composition by subtype ────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    hmm_valid = fa_df[fa_df["hmm_state"].isin(HMM_STATE_ORDER)]
    if len(hmm_valid) > 20:
        ct = pd.crosstab(hmm_valid["fa_subtype"], hmm_valid["hmm_state"])
        ct_frac = ct.div(ct.sum(axis=1), axis=0)
        subtypes_present = [s for s in ["Impulsive", "Stimulus-driven"] if s in ct_frac.index]
        x_pos = np.arange(len(subtypes_present))
        bar_w = 0.25
        for si, state in enumerate(HMM_STATE_ORDER):
            if state in ct_frac.columns:
                vals = [ct_frac.loc[sub, state] if sub in ct_frac.index else 0
                        for sub in subtypes_present]
                ax_f.bar(x_pos + si * bar_w, vals, bar_w,
                         color=HMM_STATE_COLORS[state], label=state,
                         edgecolor="white", linewidth=0.5)
        ax_f.set_xticks(x_pos + bar_w)
        ax_f.set_xticklabels(subtypes_present, fontsize=9)
        ax_f.set_ylabel("Fraction of FA trials")
        ax_f.legend(fontsize=7, loc="upper right")

        ct_vals = ct.reindex(index=subtypes_present,
                             columns=[s for s in HMM_STATE_ORDER if s in ct.columns])
        if ct_vals.shape[0] >= 2 and ct_vals.shape[1] >= 2:
            chi2, p_chi, _, _ = chi2_contingency(ct_vals.values)
            n_chi = ct_vals.values.sum()
            min_dim = min(ct_vals.shape) - 1
            v = np.sqrt(chi2 / (n_chi * min_dim)) if min_dim > 0 else 0
            stats.append({"test": "hmm_state_by_fa_subtype_chi2",
                           "chi2": chi2, "p": p_chi, "n": n_chi, "cramers_v": v})
            ax_f.text(0.5, 0.98, f"chi2={chi2:.1f}, p={p_chi:.2e}, V={v:.2f}",
                      transform=ax_f.transAxes, fontsize=7, ha="center", va="top")

    ax_f.set_title("F. HMM state by FA subtype", fontweight="bold")

    # ── Save ─────────────────────────────────────────────────────────
    save_figure(fig, "fig32j_fa_matched_null", "07_advanced")
    print("  Saved fig32j_fa_matched_null")

    if stats:
        stats_df = pd.DataFrame(stats)
        stats_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "figures", "07_advanced", "fa_matched_null_stats.csv",
        )
        stats_df.to_csv(stats_path, index=False)
        print(f"  Saved stats: {stats_path}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_total = len(fa_df)
    n_sd = (fa_df["fa_subtype"] == "Stimulus-driven").sum()
    n_imp = (fa_df["fa_subtype"] == "Impulsive").sum()
    print(f"  Total classified: {n_total}")
    print(f"  Stimulus-driven: {n_sd} ({100 * n_sd / n_total:.1f}%)")
    print(f"  Impulsive: {n_imp} ({100 * n_imp / n_total:.1f}%)")

    for stage in STAGE_ORDER:
        sub = fa_df[fa_df["stage"] == stage]
        n_s = len(sub)
        if n_s > 0:
            frac = (sub["fa_subtype"] == "Stimulus-driven").sum() / n_s
            print(f"  {stage}: {n_s} FAs, {100 * frac:.1f}% Stimulus-driven")

    print("\nDone.")


if __name__ == "__main__":
    main()
