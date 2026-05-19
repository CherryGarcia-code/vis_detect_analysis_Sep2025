"""Fig32: Lick-triggered TF analysis — separating sensory-driven from purely impulsive FAs.

Scientific question:
  Some early licks (FA trials) are preceded by a speed-up in the drifting
  grating's temporal frequency (TF) ~250 ms before the lick, suggesting the
  mouse 'thought' a real stimulus change was occurring.  Other early licks
  show no such TF deflection and are presumably purely impulsive.

  Can we:
  (a) Compute lick-triggered average TF to confirm this pattern?
  (b) Classify individual FA licks as 'TF-pulse-triggered' vs 'purely impulsive'?
  (c) Show that neural prediction differs between these FA subtypes?
  (d) Track the composition of FA subtypes across learning?

Approach:
  1. For every FA trial, extract the baseline TF trace (St1TrialVector)
     in a window before the lick (e.g. -0.5 to 0 s relative to lick).
  2. Compute the grand-average lick-triggered TF (log2 scale) across
     all FA trials, and separately for Learning vs Expert.
  3. For each FA lick, compute the max log2(TF) in (-0.3, -0.05) s
     before the lick.  If it exceeds +0.25 (the fast-pulse threshold),
     classify as 'TF-triggered'; otherwise 'impulsive'.
  4. Compare pre-trial neural activity between these subtypes.
  5. Compare HMM state composition between subtypes.

Produces:
  - Fig 32A: Grand-average lick-triggered TF (all FAs, by stage)
  - Fig 32B: Distribution of pre-lick max log2(TF) with classification threshold
  - Fig 32C: FA subtype proportions across sessions
  - Fig 32D: HMM state composition by FA subtype
  - Fig 32E: Neural prediction AUC: Hit vs TF-triggered FA vs Impulsive FA
  - Fig 32F: RT distributions by FA subtype

Saves:
  figures/07_advanced/fig32_fa_subtype_lick_triggered_tf.png
  figures/07_advanced/fa_subtype_stats.csv
  cache/fa_subtype_classification.csv
"""

import os
import sys
import gc
import warnings


import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency, spearmanr, ks_2samp
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import (
    STAGE_ORDER, STAGE_COLORS, CACHE_DIR,
    HMM_STATE_ORDER, HMM_STATE_COLORS, DEFAULT_BIN_SIZE,
)
from visdetect.suite.loader import (
    load_staging_manifest, load_session,
    load_hmm_assignments,
)
from visdetect.analysis.utils import get_good_cluster_ids, build_population_tensor
from visdetect.suite.plotting import setup_style, save_figure, add_stage_background

from visdetect.analysis.align import get_event_times_by_trial

setup_style()
warnings.filterwarnings("ignore")

# ── Parameters ────────────────────────────────────────────────────────
BASELINE_STRIDE = 3         # St1TrialVector stride (legacy convention)
SAMPLE_PERIOD = 0.05        # 50 ms per sample after stride
LTA_WINDOW = (-3.0, 1.0)    # Lick-triggered average window (s) — matches MATLAB reference
LTA_SMOOTH_SIGMA = 1.0      # Gaussian sigma in sample pts; MATLAB smoothdata('gaussian',5) ≈ sigma=1
CLASSIFY_WINDOW = (-0.30, -0.05)  # Window for max TF classification
FAST_THRESH = 0.25          # log2(TF) threshold for fast pulse
PRE_TRIAL_WINDOW = (-1.5, -0.5)
BIN_SIZE = DEFAULT_BIN_SIZE
MIN_UNITS = 5
MIN_TRIALS_PER_CLASS = 10
N_FOLDS = 5
MIN_RT_FOR_LTA = 2.0        # FA must occur >= 2.0s into baseline (matches MATLAB convention)


# =====================================================================
# Extract lick-triggered TF trace for one FA trial
# =====================================================================
# MATLAB-style: fixed-length segment using direct array indexing.
# LTA_HISTORY samples before lick + LTA_POST samples after.
LTA_HISTORY = 60             # samples before lick (60 * 50ms = 3.0s) — matches MATLAB LickHistory=60
LTA_POST = 20                # samples after lick (20 * 50ms = 1.0s) — matches MATLAB +20 post-lick
LTA_N_SAMPLES = LTA_HISTORY + LTA_POST  # total trace length
LTA_T_AXIS = (np.arange(LTA_N_SAMPLES) - (LTA_HISTORY - 1)) * SAMPLE_PERIOD


def extract_lta_segment(trial):
    """Extract fixed-length log2(TF ratio) segment centered on lick.

    Uses MATLAB-style direct array indexing (no interpolation).
    Returns log2_segment of length LTA_N_SAMPLES, or None.
    """
    bv = getattr(trial, "baseline_values", None)
    if bv is None:
        return None

    arr = np.array(bv).flatten()
    if BASELINE_STRIDE > 1:
        arr = arr[::BASELINE_STRIDE]

    rt_dict = getattr(trial, "reactiontimes", {}) or {}
    rt = rt_dict.get("FA", rt_dict.get("fa", np.nan))
    if np.isnan(rt) or rt < MIN_RT_FOR_LTA:
        return None

    # Lick sample index in the post-stride array
    lick_idx = int(round(rt / SAMPLE_PERIOD))
    start_idx = lick_idx - (LTA_HISTORY - 1)
    end_idx = lick_idx + LTA_POST + 1

    if start_idx < 0 or end_idx > len(arr):
        return None

    seg = arr[start_idx:end_idx].astype(float)
    return np.log2(np.clip(seg, 0.01, None))


def classify_fa_lick(trial, baseline_on_time):
    """Classify an FA lick as 'TF-triggered' or 'Impulsive'.

    Looks at max log2(TF) in the classification window before the lick.
    Returns (subtype, max_log2tf, rt) or (None, None, None).
    """
    bv = getattr(trial, "baseline_values", None)
    if bv is None:
        return None, None, None

    arr = np.array(bv).flatten()
    if BASELINE_STRIDE > 1:
        arr = arr[::BASELINE_STRIDE]

    n_seen = getattr(trial, "n_seen", None)
    if isinstance(n_seen, (int, np.integer)) and n_seen is not None and n_seen > 0:
        arr = arr[:int(n_seen)]

    if len(arr) < 5:
        return None, None, None

    rt_dict = getattr(trial, "reactiontimes", {}) or {}
    rt = rt_dict.get("FA", rt_dict.get("fa", np.nan))
    if np.isnan(rt) or rt < MIN_RT_FOR_LTA:
        return None, None, None

    arr_clipped = np.clip(arr.astype(float), 0.01, None)
    log2_tf = np.log2(arr_clipped)
    sample_times = np.arange(len(log2_tf)) * SAMPLE_PERIOD
    sample_times_rel = sample_times - rt

    # Classification window
    cls_mask = (sample_times_rel >= CLASSIFY_WINDOW[0]) & (sample_times_rel < CLASSIFY_WINDOW[1])
    if cls_mask.sum() < 2:
        return None, None, None

    max_log2tf = np.max(log2_tf[cls_mask])

    subtype = "TF-triggered" if max_log2tf >= FAST_THRESH else "Impulsive"
    return subtype, float(max_log2tf), float(rt)


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 60)
    print("[07f] Lick-triggered TF: FA subtype classification")
    print("=" * 60)

    manifest = load_staging_manifest(qc_only=True)
    hmm_assign = load_hmm_assignments()
    print(f"  {len(manifest)} QC-passed sessions")

    # ── Step 1: Classify all FA trials ────────────────────────────────
    print("\n[Step 1] Classifying FA trials across all sessions...")
    all_fa_records = []
    all_lta_traces = {"Learning": [], "Expert": []}
    all_lta_traces_by_subtype = {"TF-triggered": [], "Impulsive": []}

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

        baseline_on = get_event_times_by_trial(sess, "Baseline_ON")
        trials = sess.trials

        # HMM states for this session
        sess_hmm = hmm_assign[hmm_assign["session_name"] == sname]
        trial_to_hmm = {}
        if len(sess_hmm) > 0:
            for _, hr in sess_hmm.iterrows():
                trial_to_hmm[int(hr["trial_idx"])] = hr["hmm_state_label"]

        n_tf_trig = 0
        n_imp = 0

        for i, t in enumerate(trials):
            outcome = getattr(t, "trialoutcome", None)
            if outcome != "FA":
                continue

            t0 = baseline_on[i] if i < len(baseline_on) else np.nan
            if np.isnan(t0):
                continue

            subtype, max_l2, rt = classify_fa_lick(t, t0)
            if subtype is None:
                continue

            # Get change_size for this trial
            cs = getattr(t, "change_size", 1.0)
            if cs is None:
                cs = 1.0

            hmm_state = trial_to_hmm.get(i, "Unknown")

            all_fa_records.append({
                "session_name": sname,
                "stage": stage,
                "session_idx": sidx,
                "trial_idx": i,
                "fa_subtype": subtype,
                "max_log2tf_pre_lick": max_l2,
                "rt": rt,
                "change_size": cs,
                "is_go": cs > 1.01,
                "hmm_state": hmm_state,
            })

            if subtype == "TF-triggered":
                n_tf_trig += 1
            else:
                n_imp += 1

            # Extract LTA trace (fixed-length, no interpolation)
            lta_seg = extract_lta_segment(t)
            if lta_seg is not None:
                if stage in all_lta_traces:
                    all_lta_traces[stage].append(lta_seg)
                all_lta_traces_by_subtype[subtype].append(lta_seg)

        print(f"TF-trig={n_tf_trig}, Impulse={n_imp}")

        del sess
        gc.collect()

    fa_df = pd.DataFrame(all_fa_records)
    print(f"\n  Total classified FAs: {len(fa_df)}")
    print(f"  TF-triggered: {(fa_df['fa_subtype'] == 'TF-triggered').sum()}")
    print(f"  Impulsive: {(fa_df['fa_subtype'] == 'Impulsive').sum()}")

    # Save classification
    cache_path = os.path.join(CACHE_DIR, "fa_subtype_classification.csv")
    fa_df.to_csv(cache_path, index=False)
    print(f"  Saved: {cache_path}")

    # ══════════════════════════════════════════════════════════════════
    # FIGURE 32: FA subtype analysis
    # ══════════════════════════════════════════════════════════════════
    print("\n[Step 2] Generating Figure 32...")
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.35)
    stats = []

    subtype_colors = {
        "TF-triggered": "#e74c3c",
        "Impulsive": "#3498db",
    }

    # ── Panel A: Grand-average lick-triggered TF ──────────────────────
    ax_a = fig.add_subplot(gs[0, 0])

    # By stage
    for stage in STAGE_ORDER:
        traces = all_lta_traces.get(stage, [])
        if len(traces) < 5:
            continue
        arr = np.array(traces)
        mean_trace = np.nanmean(arr, axis=0)
        sem_trace = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
        # Gaussian smooth to match MATLAB smoothdata('gaussian',5) convention
        mean_sm = gaussian_filter1d(mean_trace, sigma=LTA_SMOOTH_SIGMA)
        ci95_sm = gaussian_filter1d(sem_trace * 1.96, sigma=LTA_SMOOTH_SIGMA)
        ax_a.plot(LTA_T_AXIS, mean_sm, color=STAGE_COLORS[stage],
                  linewidth=2, label=f"{stage} (n={len(traces)})")
        ax_a.fill_between(LTA_T_AXIS,
                          mean_sm - ci95_sm, mean_sm + ci95_sm,
                          color=STAGE_COLORS[stage], alpha=0.2)

    ax_a.axvline(0, color="k", ls="--", lw=1, alpha=0.7, label="Lick")
    ax_a.axhline(0, color="grey", ls=":", lw=0.5, alpha=0.5)
    ax_a.axhline(FAST_THRESH, color="#e74c3c", ls="--", lw=0.8, alpha=0.5,
                  label=f"Fast thresh ({FAST_THRESH})")
    ax_a.set_xlabel("Time relative to lick (s)")
    ax_a.set_ylabel("log2(TF ratio)")
    ax_a.legend(fontsize=7, loc="upper left")
    ax_a.set_xlim(LTA_T_AXIS[0], LTA_T_AXIS[-1])
    ax_a.set_title("A. Lick-triggered avg TF (all FAs)", fontweight="bold")

    # ── Panel B: By subtype ───────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])

    for subtype in ["TF-triggered", "Impulsive"]:
        traces = all_lta_traces_by_subtype.get(subtype, [])
        if len(traces) < 5:
            continue
        arr = np.array(traces)
        mean_trace = np.nanmean(arr, axis=0)
        sem_trace = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
        # Gaussian smooth to match MATLAB smoothdata('gaussian',5) convention
        mean_sm = gaussian_filter1d(mean_trace, sigma=LTA_SMOOTH_SIGMA)
        ci95_sm = gaussian_filter1d(sem_trace * 1.96, sigma=LTA_SMOOTH_SIGMA)
        ax_b.plot(LTA_T_AXIS, mean_sm, color=subtype_colors[subtype],
                  linewidth=2, label=f"{subtype} (n={len(traces)})")
        ax_b.fill_between(LTA_T_AXIS,
                          mean_sm - ci95_sm, mean_sm + ci95_sm,
                          color=subtype_colors[subtype], alpha=0.2)

    ax_b.axvline(0, color="k", ls="--", lw=1, alpha=0.7, label="Lick")
    ax_b.axhline(0, color="grey", ls=":", lw=0.5, alpha=0.5)
    ax_b.axhline(FAST_THRESH, color="#e74c3c", ls="--", lw=0.8, alpha=0.3)
    ax_b.set_xlabel("Time relative to lick (s)")
    ax_b.set_ylabel("log2(TF ratio)")
    ax_b.legend(fontsize=7, loc="upper left")
    ax_b.set_xlim(LTA_T_AXIS[0], LTA_T_AXIS[-1])
    ax_b.set_title("B. Lick-triggered avg TF by FA subtype", fontweight="bold")

    # ── Panel C: Distribution of pre-lick max log2(TF) ────────────────
    ax_c = fig.add_subplot(gs[0, 2])

    tf_trig_vals = fa_df[fa_df["fa_subtype"] == "TF-triggered"]["max_log2tf_pre_lick"].values
    imp_vals = fa_df[fa_df["fa_subtype"] == "Impulsive"]["max_log2tf_pre_lick"].values

    bins = np.linspace(-1.0, 1.5, 50)
    ax_c.hist(imp_vals, bins=bins, alpha=0.6, color=subtype_colors["Impulsive"],
              label=f"Impulsive (n={len(imp_vals)})", density=True, edgecolor="white")
    ax_c.hist(tf_trig_vals, bins=bins, alpha=0.6, color=subtype_colors["TF-triggered"],
              label=f"TF-triggered (n={len(tf_trig_vals)})", density=True, edgecolor="white")
    ax_c.axvline(FAST_THRESH, color="k", ls="--", lw=1.5, label=f"Threshold ({FAST_THRESH})")
    ax_c.set_xlabel("Max log2(TF) in [-0.3, -0.05]s before lick")
    ax_c.set_ylabel("Density")
    ax_c.legend(fontsize=7)

    # KS test
    if len(tf_trig_vals) >= 5 and len(imp_vals) >= 5:
        ks_stat, ks_p = ks_2samp(tf_trig_vals, imp_vals)
        stats.append({"test": "max_log2tf_ks", "ks_stat": ks_stat, "p": ks_p,
                       "n_tf_trig": len(tf_trig_vals), "n_imp": len(imp_vals)})
        ax_c.text(0.98, 0.98, f"KS={ks_stat:.3f}, p={ks_p:.1e}",
                  transform=ax_c.transAxes, fontsize=7, ha="right", va="top")

    ax_c.set_title("C. Pre-lick TF classification", fontweight="bold")

    # ── Panel D: FA subtype proportions across sessions ───────────────
    ax_d = fig.add_subplot(gs[1, 0])

    sess_counts = fa_df.groupby(["session_name", "session_idx", "stage", "fa_subtype"]).size().unstack(fill_value=0)
    sess_fracs = sess_counts.div(sess_counts.sum(axis=1), axis=0)
    sess_fracs = sess_fracs.reset_index().sort_values("session_idx")

    for subtype in ["TF-triggered", "Impulsive"]:
        if subtype in sess_fracs.columns:
            ax_d.scatter(sess_fracs["session_idx"], sess_fracs[subtype],
                         c=[subtype_colors[subtype]] * len(sess_fracs),
                         s=40, label=subtype, edgecolors="k", linewidth=0.3, zorder=3)
            # Trend line
            if len(sess_fracs) >= 5:
                z = np.polyfit(sess_fracs["session_idx"], sess_fracs[subtype], 1)
                x_fit = np.linspace(sess_fracs["session_idx"].min(),
                                    sess_fracs["session_idx"].max(), 50)
                ax_d.plot(x_fit, np.polyval(z, x_fit), color=subtype_colors[subtype],
                          ls="--", lw=1, alpha=0.5)

    add_stage_background(ax_d, manifest, alpha=0.06)
    ax_d.set_xlabel("Session index")
    ax_d.set_ylabel("Fraction of FA trials")
    ax_d.set_ylim(-0.05, 1.05)
    ax_d.legend(fontsize=7, loc="center right")

    # Spearman trend for TF-triggered fraction
    if "TF-triggered" in sess_fracs.columns and len(sess_fracs) >= 5:
        r, p = spearmanr(sess_fracs["session_idx"], sess_fracs["TF-triggered"])
        stats.append({"test": "tf_triggered_frac_vs_session", "rho": r, "p": p,
                       "n": len(sess_fracs)})
        ax_d.text(0.02, 0.98, f"TF-trig fraction: rho={r:.3f}, p={p:.3f}",
                  transform=ax_d.transAxes, fontsize=7, va="top")

    ax_d.set_title("D. FA subtype fractions across learning", fontweight="bold")

    # ── Panel E: HMM state composition by FA subtype ──────────────────
    ax_e = fig.add_subplot(gs[1, 1])

    hmm_valid = fa_df[fa_df["hmm_state"].isin(HMM_STATE_ORDER)]
    if len(hmm_valid) > 20:
        ct = pd.crosstab(hmm_valid["fa_subtype"], hmm_valid["hmm_state"])
        ct_frac = ct.div(ct.sum(axis=1), axis=0)

        subtypes_present = [s for s in ["Impulsive", "TF-triggered"] if s in ct_frac.index]
        x_pos = np.arange(len(subtypes_present))
        bar_width = 0.25

        for si, state in enumerate(HMM_STATE_ORDER):
            if state in ct_frac.columns:
                vals = [ct_frac.loc[sub, state] if sub in ct_frac.index else 0
                        for sub in subtypes_present]
                ax_e.bar(x_pos + si * bar_width, vals, bar_width,
                         color=HMM_STATE_COLORS[state], label=state,
                         edgecolor="white", linewidth=0.5)

        ax_e.set_xticks(x_pos + bar_width)
        ax_e.set_xticklabels(subtypes_present)
        ax_e.set_ylabel("Fraction of FA trials")
        ax_e.legend(fontsize=7, loc="upper right")

        # Chi-squared test
        ct_vals = ct.reindex(index=subtypes_present,
                             columns=[s for s in HMM_STATE_ORDER if s in ct.columns])
        if ct_vals.shape[0] >= 2 and ct_vals.shape[1] >= 2:
            chi2, p_chi, _, _ = chi2_contingency(ct_vals.values)
            n_chi = ct_vals.values.sum()
            # Cramer's V
            min_dim = min(ct_vals.shape) - 1
            v = np.sqrt(chi2 / (n_chi * min_dim)) if min_dim > 0 else 0
            stats.append({"test": "hmm_state_by_fa_subtype_chi2",
                           "chi2": chi2, "p": p_chi, "n": n_chi, "cramers_v": v})
            ax_e.text(0.5, 0.98,
                      f"chi2={chi2:.1f}, p={p_chi:.2e}, V={v:.2f}",
                      transform=ax_e.transAxes, fontsize=7, ha="center", va="top")

        # Annotate counts
        for i, sub in enumerate(subtypes_present):
            n = ct.loc[sub].sum() if sub in ct.index else 0
            ax_e.text(i + bar_width, -0.05, f"n={n}", ha="center", fontsize=7)

    ax_e.set_title("E. HMM state by FA subtype", fontweight="bold")

    # ── Panel F: RT distributions by FA subtype ───────────────────────
    ax_f = fig.add_subplot(gs[1, 2])

    for subtype in ["TF-triggered", "Impulsive"]:
        rts = fa_df[fa_df["fa_subtype"] == subtype]["rt"].dropna().values
        if len(rts) < 5:
            continue
        # Clip for display
        rts_clip = rts[rts <= 20]
        ax_f.hist(rts_clip, bins=50, alpha=0.5, color=subtype_colors[subtype],
                  label=f"{subtype} (n={len(rts)}, med={np.median(rts):.1f}s)",
                  density=True, edgecolor="white")

    # Mann-Whitney on RT
    rt_tf = fa_df[fa_df["fa_subtype"] == "TF-triggered"]["rt"].dropna().values
    rt_imp = fa_df[fa_df["fa_subtype"] == "Impulsive"]["rt"].dropna().values
    if len(rt_tf) >= 5 and len(rt_imp) >= 5:
        U, p_u = mannwhitneyu(rt_tf, rt_imp, alternative="two-sided")
        r_rb = 1 - 2 * U / (len(rt_tf) * len(rt_imp))
        stats.append({"test": "rt_by_fa_subtype_mannwhitney",
                       "U": U, "p": p_u, "r_rb": r_rb,
                       "median_tf_trig": np.median(rt_tf),
                       "median_imp": np.median(rt_imp)})
        ax_f.text(0.98, 0.98,
                  f"U={U:.0f}, p={p_u:.2e}\nr_rb={r_rb:.3f}",
                  transform=ax_f.transAxes, fontsize=7, ha="right", va="top")

    ax_f.set_xlabel("FA reaction time (s from baseline onset)")
    ax_f.set_ylabel("Density")
    ax_f.legend(fontsize=7, loc="upper right")
    ax_f.set_title("F. RT distribution by FA subtype", fontweight="bold")

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig32_fa_subtype_lick_triggered_tf", "07_advanced")
    print("  Saved fig32_fa_subtype_lick_triggered_tf")

    if stats:
        stats_df = pd.DataFrame(stats)
        stats_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "figures", "07_advanced", "fa_subtype_stats.csv"
        )
        stats_df.to_csv(stats_path, index=False)
        print(f"  Saved stats: {stats_path}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_total = len(fa_df)
    n_tf = (fa_df["fa_subtype"] == "TF-triggered").sum()
    n_imp = (fa_df["fa_subtype"] == "Impulsive").sum()
    print(f"  Total classified FAs: {n_total}")
    print(f"  TF-triggered: {n_tf} ({100*n_tf/n_total:.1f}%)")
    print(f"  Impulsive:    {n_imp} ({100*n_imp/n_total:.1f}%)")
    print(f"  Median RT - TF-triggered: {fa_df[fa_df['fa_subtype']=='TF-triggered']['rt'].median():.2f}s")
    print(f"  Median RT - Impulsive:    {fa_df[fa_df['fa_subtype']=='Impulsive']['rt'].median():.2f}s")

    # Stage breakdown
    for stage in STAGE_ORDER:
        sub = fa_df[fa_df["stage"] == stage]
        n_s = len(sub)
        if n_s > 0:
            frac_tf = (sub["fa_subtype"] == "TF-triggered").sum() / n_s
            print(f"  {stage}: {n_s} FAs, {100*frac_tf:.1f}% TF-triggered")

    # HMM state composition
    for subtype in ["TF-triggered", "Impulsive"]:
        sub = fa_df[(fa_df["fa_subtype"] == subtype) & fa_df["hmm_state"].isin(HMM_STATE_ORDER)]
        if len(sub) > 0:
            dist = sub["hmm_state"].value_counts(normalize=True)
            print(f"  {subtype} HMM: " + ", ".join(f"{s}={dist.get(s,0):.1%}" for s in HMM_STATE_ORDER))

    print("\nDone.")


if __name__ == "__main__":
    main()
