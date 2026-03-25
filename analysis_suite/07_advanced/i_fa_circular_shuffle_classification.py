"""Fig32i: FA classification via within-trial circular shuffle.

For each FA trial, tests whether the pre-lick TF fluctuation was
statistically unusual compared to the rest of that trial's baseline.
Uses circular permutation to build a within-trial null distribution,
preserving autocorrelation structure.

Two test statistics:
  1. max(log2 TF) in [-1.0, -0.4]s before lick  (magnitude)
  2. max(dTF/dt) in the same window                (onset sharpness)

If either p < 0.05 the lick is classified as 'Stimulus-driven';
otherwise 'Impulsive'.

Saves:
  cache/fa_classification_circular_shuffle.csv
  figures/07_advanced/fig32i_fa_circular_shuffle.png
  figures/07_advanced/fa_circular_shuffle_stats.csv
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
SAMPLE_PERIOD = 0.05          # 50 ms per sample after stride
CLASSIFY_WINDOW = (-1.0, -0.4)  # window relative to lick for classification features
MIN_RT_FOR_LTA = 2.0
ALPHA = 0.05
N_PERM = 500
MIN_SHIFT_SAMPLES = 20        # 1.0 s minimum circular shift

# LTA parameters (match MATLAB LickTrigAvg.m)
LTA_HISTORY = 60              # 60 samples = 3.0 s before lick
LTA_POST = 20                 # 20 samples = 1.0 s after lick
LTA_N_SAMPLES = LTA_HISTORY + LTA_POST
LTA_T_AXIS = (np.arange(LTA_N_SAMPLES) - (LTA_HISTORY - 1)) * SAMPLE_PERIOD
LTA_SMOOTH_SIGMA = 1.0

# Number of samples in the classification window
_CLS_N_SAMPLES = int(round((CLASSIFY_WINDOW[1] - CLASSIFY_WINDOW[0]) / SAMPLE_PERIOD))


# =====================================================================
# Core functions
# =====================================================================


def compute_pre_lick_stats(log2_tf, rt, window=CLASSIFY_WINDOW):
    """Compute max(log2 TF) and max(dTF/dt) in the pre-lick window.

    Returns (max_log2tf, max_dtf_dt, window_mask) or (None, None, None).
    """
    sample_times = np.arange(len(log2_tf)) * SAMPLE_PERIOD
    sample_times_rel = sample_times - rt

    mask = (sample_times_rel >= window[0]) & (sample_times_rel < window[1])
    if mask.sum() < 2:
        return None, None, None

    max_log2tf = float(np.max(log2_tf[mask]))

    # Rate-of-change: use diff of log2_tf, mask shifted by half sample
    dtf = np.diff(log2_tf) / SAMPLE_PERIOD
    dmask = mask[:-1] & mask[1:]  # both endpoints in window
    if dmask.sum() < 1:
        max_dtf_dt = 0.0
    else:
        max_dtf_dt = float(np.max(dtf[dmask]))

    return max_log2tf, max_dtf_dt, mask


def circular_shuffle_test(log2_tf, n_valid, rt, n_perm=N_PERM, rng=None):
    """Circular permutation test for pre-lick TF being unusual.

    Returns (p_max_tf, p_dtf, obs_max_tf, obs_max_dtf).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    obs_max_tf, obs_max_dtf, obs_mask = compute_pre_lick_stats(log2_tf, rt)
    if obs_max_tf is None:
        return None, None, None, None

    # Window indices relative to lick for the classification window
    lick_idx = int(round(rt / SAMPLE_PERIOD))
    cls_start = lick_idx + int(round(CLASSIFY_WINDOW[0] / SAMPLE_PERIOD))
    cls_end = lick_idx + int(round(CLASSIFY_WINDOW[1] / SAMPLE_PERIOD))
    cls_len = cls_end - cls_start

    if cls_len < 2:
        return None, None, None, None

    # Need enough room for circular shifts
    if n_valid < 2 * MIN_SHIFT_SAMPLES + cls_len:
        return None, None, None, None

    # Generate shift offsets (avoid small offsets that preserve the window)
    shifts = rng.integers(MIN_SHIFT_SAMPLES, n_valid - MIN_SHIFT_SAMPLES, size=n_perm)

    trace = log2_tf[:n_valid]
    null_max_tf = np.empty(n_perm)
    null_max_dtf = np.empty(n_perm)

    for pi in range(n_perm):
        shifted = np.roll(trace, int(shifts[pi]))
        # Extract same window position (relative to lick index)
        seg = shifted[cls_start:cls_end]
        if len(seg) < 2:
            null_max_tf[pi] = -np.inf
            null_max_dtf[pi] = -np.inf
            continue
        null_max_tf[pi] = np.max(seg)
        d = np.diff(seg) / SAMPLE_PERIOD
        null_max_dtf[pi] = np.max(d) if len(d) > 0 else -np.inf

    p_max_tf = (np.sum(null_max_tf >= obs_max_tf) + 1) / (n_perm + 1)
    p_dtf = (np.sum(null_max_dtf >= obs_max_dtf) + 1) / (n_perm + 1)

    return float(p_max_tf), float(p_dtf), obs_max_tf, obs_max_dtf


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 60)
    print("[07i] FA classification: within-trial circular shuffle")
    print("=" * 60)

    manifest = load_staging_manifest(qc_only=True)
    hmm_assign = load_hmm_assignments()
    print(f"  {len(manifest)} QC-passed sessions")

    rng = np.random.default_rng(42)
    all_records = []
    lta_by_subtype = {"Stimulus-driven": [], "Impulsive": []}
    lta_by_stage = {"Learning": [], "Expert": []}

    print(f"\n[Step 1] Classifying FA trials ({N_PERM} permutations each)...")

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
            if log2_tf is None:
                continue

            # Circular shuffle test
            p_tf, p_dtf, obs_tf, obs_dtf = circular_shuffle_test(
                log2_tf, n_valid, rt, n_perm=N_PERM, rng=rng
            )
            if p_tf is None:
                continue

            # Classification
            if p_tf < ALPHA or p_dtf < ALPHA:
                subtype = "Stimulus-driven"
                n_stim += 1
            else:
                subtype = "Impulsive"
                n_imp += 1

            # Original threshold for comparison
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
                "p_value_max_tf": p_tf,
                "p_value_dtf": p_dtf,
                "max_log2tf": obs_tf,
                "max_dtf_dt": obs_dtf,
                "rt": rt,
                "hmm_state": hmm_state,
                "original_subtype": orig,
                "change_size": cs,
                "is_go": cs > 1.01,
            })

            # LTA trace
            lta_seg = extract_lta_segment(
                t, BASELINE_STRIDE, SAMPLE_PERIOD, MIN_RT_FOR_LTA,
                LTA_HISTORY, LTA_POST
            )
            if lta_seg is not None:
                lta_by_subtype[subtype].append(lta_seg)
                if stage in lta_by_stage:
                    lta_by_stage[stage].append(lta_seg)

        print(f"Stim-driven={n_stim}, Impulsive={n_imp}")
        del sess
        gc.collect()

    fa_df = pd.DataFrame(all_records)
    print(f"\n  Total classified: {len(fa_df)}")
    if len(fa_df) == 0:
        print("  ERROR: No FA trials classified.")
        return
    print(f"  Stimulus-driven: {(fa_df['fa_subtype'] == 'Stimulus-driven').sum()}")
    print(f"  Impulsive: {(fa_df['fa_subtype'] == 'Impulsive').sum()}")

    cache_path = os.path.join(CACHE_DIR, "fa_classification_circular_shuffle.csv")
    fa_df.to_csv(cache_path, index=False)
    print(f"  Saved: {cache_path}")

    # ══════════════════════════════════════════════════════════════════
    # FIGURE
    # ══════════════════════════════════════════════════════════════════
    print("\n[Step 2] Generating Figure 32i...")
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
    ax_a.set_title("A. LTA by subtype (circular shuffle)", fontweight="bold")

    # ── Panel B: P-value distribution ────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    # Use min(p_tf, p_dtf) as the combined p-value
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
        # Map to binary for confusion matrix
        new_pos = (valid["fa_subtype"] == "Stimulus-driven").values
        old_pos = (valid["original_subtype"] == "TF-triggered").values
        # 2x2 contingency: rows=new, cols=old
        tp = np.sum(new_pos & old_pos)
        fp = np.sum(new_pos & ~old_pos)
        fn = np.sum(~new_pos & old_pos)
        tn = np.sum(~new_pos & ~old_pos)
        conf = np.array([[tp, fp], [fn, tn]])

        agree = (tp + tn) / len(valid)
        # Cohen's kappa
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
        ax_d.set_ylabel("Circular shuffle")
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
    save_figure(fig, "fig32i_fa_circular_shuffle", "07_advanced")
    print("  Saved fig32i_fa_circular_shuffle")

    if stats:
        stats_df = pd.DataFrame(stats)
        stats_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "figures", "07_advanced", "fa_circular_shuffle_stats.csv",
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
