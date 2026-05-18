"""Fig 06: Post-error controls — selection bias & controlled analyses.

Builds on the 01d finding (apparent post-error accuracy boost) by
demonstrating that the effect is a selection/survivorship bias and
providing controlled analyses.

The key insight:
  After errors, FA/abort rate on go trials rises (~62% vs ~49%).
  When only hit+miss trials are retained (standard psychometric),
  the surviving subset is biased toward high-lick-readiness → inflated
  hit rate.  When FA/abort are included, performance is WORSE after
  errors, consistent with post-error impulsivity.

Panels (4 × 3 = 12):
  Row 1 — The bias & its demonstration
    A. FA/abort rate on go trials: post-correct vs post-error
    B. Hit rate two ways: biased (hit/(hit+miss)) vs inclusive (hit/all_go)
    C. Psychometric curves: biased vs inclusive metric, post-correct vs post-error

  Row 2 — Expert-stage only
    D. Expert-only: FA/abort rate post-correct vs post-error
    E. Expert-only: psychometric (inclusive metric) by error-streak
    F. Expert-only: RT distributions post-correct vs post-error

  Row 3 — Change-timing control
    G. Change-time distribution (early vs mid vs late bins)
    H. Inclusive hit rate by change-time bin, post-correct vs post-error
    I. FA/abort rate by change-time bin (do errors cause earlier licking?)

  Row 4 — Lick-rate & HMM-state controls
    J. Per-session lick rate vs post-error FA/abort excess
    K. Post-error inclusive hit rate, sessions split by median lick rate
    L. HMM-state breakdown (if available): post-error effect per state

Saves:
  figures/01_behavior/fig06_post_error_controls.png
  figures/01_behavior/post_error_controls_stats.csv
"""

import os
import sys
import gc
import warnings


import numpy as np
import pandas as pd
from scipy.stats import (
    chi2_contingency, mannwhitneyu, kruskal, spearmanr, wilcoxon,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import (
    STAGE_ORDER, STAGE_COLORS,
    CHANGE_SIZES, CHANGE_SIZE_LABELS, CHANGE_SIZE_POSITIONS,
    CACHE_DIR,
)
from visdetect.suite.loader import load_staging_manifest, load_session
from visdetect.suite.plotting import setup_style, save_figure

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visdetect.analysis.behavior import get_trial_dataframe

setup_style()
warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────
MIN_TRIALS = 5
PKL_DIR = os.path.join(_root, "data", "pkls", "BG_046")
TRIAL_CACHE = os.path.join(CACHE_DIR, "all_trials_behavior.csv")
BOOL_COLS = ["is_hit", "is_miss", "is_fa", "is_abort", "is_ref",
             "is_go", "is_catch"]

# Change-time quantile bins
N_CT_BINS = 3
CT_BIN_LABELS = ["Early", "Mid", "Late"]
CT_BIN_COLORS = ["#1B5E20", "#FFA000", "#B71C1C"]

# Colors for post-correct vs post-error
COL_CORRECT = "#4CAF50"
COL_ERROR = "#F44336"

# HMM state config (imported here so script doesn't crash if unavailable)
try:
    from visdetect.suite.config import HMM_STATE_ORDER, HMM_STATE_COLORS
    from visdetect.suite.loader import load_hmm_assignments
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False
    HMM_STATE_ORDER = []
    HMM_STATE_COLORS = {}


# ══════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════
def load_trials(manifest):
    """Load trial data from cache or build from session pickles."""
    if os.path.exists(TRIAL_CACHE):
        print(f"  Loading cached trial data from {os.path.basename(TRIAL_CACHE)}")
        trials = pd.read_csv(TRIAL_CACHE)
        for c in BOOL_COLS:
            if c in trials.columns:
                trials[c] = trials[c].astype(bool)
        return trials

    print("  Loading session pickles (will cache for future runs)...")
    all_dfs = []
    for i, (_, row) in enumerate(manifest.iterrows()):
        sname = int(row["session_name"])
        stage = row["stage"]
        sidx = row["session_idx"]
        date_str = str(sname).zfill(8)
        pkl_path = os.path.join(PKL_DIR, f"BG_046_{date_str}.pkl")
        if not os.path.exists(pkl_path):
            continue
        sess = load_session(sname)
        df = get_trial_dataframe(sess)
        if df.empty:
            del sess
            gc.collect()
            continue
        df["session_name"] = sname
        df["stage"] = stage
        df["session_idx"] = sidx
        all_dfs.append(df)
        del sess
        gc.collect()
        print(f"    [{i+1}/{len(manifest)}] {date_str}: {len(df)} trials")

    trials = pd.concat(all_dfs, ignore_index=True)
    trials.to_csv(TRIAL_CACHE, index=False)
    print(f"  Cached {len(trials)} trials → {os.path.basename(TRIAL_CACHE)}")
    return trials


def prepare_trials(trials):
    """Add error history, lick-rate, and HMM columns to trial DataFrame."""
    # Error flag and previous-trial info
    trials["is_error"] = trials["outcome"].isin(["fa", "abort"])
    trials["prev_outcome"] = trials.groupby("session_name")["outcome"].shift(1)
    trials["prev_error"] = trials.groupby("session_name")["is_error"].shift(1)
    trials["prev_error"] = trials["prev_error"].fillna(False).astype(bool)

    # Error streak
    def _streak(group):
        is_err = group["is_error"].values
        s = np.zeros(len(is_err), dtype=int)
        for i in range(1, len(is_err)):
            s[i] = s[i - 1] + 1 if is_err[i - 1] else 0
        group = group.copy()
        group["n_preceding_errors"] = s
        return group

    trials = trials.groupby("session_name", group_keys=False).apply(_streak)
    trials["error_streak_bin"] = trials["n_preceding_errors"].clip(upper=3)

    # Post-error vs post-correct label
    trials["post_cond"] = np.where(trials["prev_error"], "error", "correct")

    # Per-session lick rate (licks / trial as proxy — FA + abort + hit fraction)
    sess_lick = trials.groupby("session_name").apply(
        lambda g: g["outcome"].isin(["fa", "abort", "hit"]).mean()
    ).rename("sess_lick_rate")
    trials = trials.merge(sess_lick, on="session_name", how="left")

    # Change-time quantile bins (within go trials)
    go_mask = trials["is_go"] == True
    if "change_time" in trials.columns:
        ct = trials.loc[go_mask, "change_time"]
        ct_valid = ct.dropna()
        if len(ct_valid) > 30:
            bin_edges = ct_valid.quantile([0, 1/3, 2/3, 1.0]).values
            bin_edges[0] -= 1e-6
            bin_edges[-1] += 1e-6
            trials["ct_bin"] = pd.cut(
                trials["change_time"], bins=bin_edges,
                labels=CT_BIN_LABELS, include_lowest=True,
            )
        else:
            trials["ct_bin"] = np.nan
    else:
        trials["ct_bin"] = np.nan

    # HMM state merge
    trials["hmm_state"] = np.nan
    if _HMM_AVAILABLE:
        try:
            hmm = load_hmm_assignments()
            hmm_sub = hmm[["session_name", "trial_idx", "hmm_state_label"]].copy()
            hmm_sub["session_name"] = hmm_sub["session_name"].astype(int)
            trials["session_name"] = trials["session_name"].astype(int)
            trials = trials.merge(hmm_sub, on=["session_name", "trial_idx"],
                                  how="left")
            trials["hmm_state"] = trials["hmm_state_label"]
            trials.drop(columns=["hmm_state_label"], inplace=True, errors="ignore")
            n_hmm = trials["hmm_state"].notna().sum()
            print(f"  HMM states merged: {n_hmm}/{len(trials)} trials labelled")
        except Exception as e:
            print(f"  HMM states not available ({e})")

    return trials


# ══════════════════════════════════════════════════════════════════════
# Plotting Helpers
# ══════════════════════════════════════════════════════════════════════
def _inclusive_hit_rate(sub):
    """Hit rate = hits / ALL go trials (including FA/abort)."""
    n = len(sub)
    if n < MIN_TRIALS:
        return np.nan, 0, n
    hr = sub["is_hit"].mean()
    se = np.sqrt(hr * (1 - hr) / n)
    return hr, se, n


def _biased_hit_rate(sub):
    """Hit rate = hits / (hits + misses) — the standard but biased metric."""
    hm = sub[sub["outcome"].isin(["hit", "miss"])]
    n = len(hm)
    if n < MIN_TRIALS:
        return np.nan, 0, n
    hr = hm["is_hit"].mean()
    se = np.sqrt(hr * (1 - hr) / n)
    return hr, se, n


def _fa_abort_rate(sub):
    """FA+abort rate on go trials."""
    n = len(sub)
    if n < MIN_TRIALS:
        return np.nan, 0, n
    r = sub["outcome"].isin(["fa", "abort"]).mean()
    se = np.sqrt(r * (1 - r) / n)
    return r, se, n


def _plot_psychometric(ax, data, condition_col, cond_values, colors, labels,
                       hr_func, title=""):
    """Plot psychometric curves with a given hit-rate function."""
    for cval in cond_values:
        sub = data[data[condition_col] == cval]
        hrs, errs, ns = [], [], []
        for cs in CHANGE_SIZES:
            cs_sub = sub[sub["change_size"].between(cs - 0.01, cs + 0.01)]
            hr, se, n = hr_func(cs_sub)
            hrs.append(hr)
            errs.append(se)
            ns.append(n)
        total_n = sum(ns)
        ax.errorbar(CHANGE_SIZE_POSITIONS, hrs, yerr=errs,
                     fmt="o-", color=colors[cval], lw=2, ms=6, capsize=4,
                     label=f"{labels[cval]} (n={total_n})")
    ax.set_xticks(CHANGE_SIZE_POSITIONS)
    ax.set_xticklabels(CHANGE_SIZE_LABELS, fontsize=8)
    ax.set_xlabel("Change size (TF ratio)")
    ax.set_ylabel("Hit rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")


def _sig_str(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("[01f] Post-error performance:  selection bias & controlled analyses")
    print("=" * 70)

    manifest = load_staging_manifest(qc_only=True)
    print(f"  {len(manifest)} QC-passed sessions")

    trials = load_trials(manifest)
    trials = prepare_trials(trials)

    go = trials[trials["is_go"] == True].copy()
    go_first_valid = go[go["prev_outcome"].notna()]  # exclude trial 1 per session

    print(f"  Total trials: {len(trials)}")
    print(f"  Go trials with prior context: {len(go_first_valid)}")
    print(f"  Post-correct go: {(~go_first_valid['prev_error']).sum()}")
    print(f"  Post-error go:   {go_first_valid['prev_error'].sum()}")

    fig = plt.figure(figsize=(24, 28))
    gs = gridspec.GridSpec(4, 3, hspace=0.42, wspace=0.35)
    stats = []

    cond_colors = {"correct": COL_CORRECT, "error": COL_ERROR}
    cond_labels = {"correct": "After correct", "error": "After error"}

    # ==================================================================
    # ROW 1 — The selection bias & its demonstration
    # ==================================================================

    # ── Panel A: FA/abort rate on go trials ────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])

    after_c = go_first_valid[go_first_valid["post_cond"] == "correct"]
    after_e = go_first_valid[go_first_valid["post_cond"] == "error"]

    far_c, far_c_se, n_c = _fa_abort_rate(after_c)
    far_e, far_e_se, n_e = _fa_abort_rate(after_e)

    ax_a.bar([0, 1], [far_c, far_e], yerr=[far_c_se, far_e_se],
             color=[COL_CORRECT, COL_ERROR], edgecolor="k", lw=0.5,
             capsize=6, alpha=0.75, width=0.5)
    ax_a.set_xticks([0, 1])
    ax_a.set_xticklabels([f"After correct\n(n={n_c})",
                           f"After error\n(n={n_e})"])
    ax_a.set_ylabel("FA + abort rate on go trials")
    ax_a.set_ylim(0, 1)
    ax_a.set_title("A. Post-error impulsivity on go trials", fontweight="bold")

    # Chi-squared
    ct_a = pd.crosstab(go_first_valid["post_cond"],
                        go_first_valid["outcome"].isin(["fa", "abort"]))
    if ct_a.shape == (2, 2):
        chi2, p_a, _, _ = chi2_contingency(ct_a)
        stats.append({"test": "chi2_fa_abort_rate_correct_vs_error",
                       "chi2": chi2, "p": p_a,
                       "far_correct": far_c, "far_error": far_e})
        ax_a.text(0.5, 0.95,
                  f"χ²={chi2:.1f}, p={p_a:.2e} {_sig_str(p_a)}",
                  transform=ax_a.transAxes, fontsize=9, ha="center", va="top")

    # ── Panel B: Hit rate — biased vs inclusive ────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])

    hr_biased_c, _, _ = _biased_hit_rate(after_c)
    hr_biased_e, _, _ = _biased_hit_rate(after_e)
    hr_incl_c, _, _ = _inclusive_hit_rate(after_c)
    hr_incl_e, _, _ = _inclusive_hit_rate(after_e)

    x_pos = np.array([0, 1, 3, 4])
    heights = [hr_biased_c, hr_biased_e, hr_incl_c, hr_incl_e]
    bar_colors = [COL_CORRECT, COL_ERROR, COL_CORRECT, COL_ERROR]

    ax_b.bar(x_pos, heights, color=bar_colors, edgecolor="k", lw=0.5,
             alpha=0.75, width=0.7)
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(["Correct", "Error", "Correct", "Error"],
                          fontsize=8)
    ax_b.set_ylabel("Hit rate")
    ax_b.set_ylim(0, 1.05)

    # Group labels
    ax_b.text(0.5, -0.15, "hit/(hit+miss)\n[BIASED]", fontsize=8,
              transform=ax_b.transAxes, ha="center", va="top",
              color="grey", style="italic")
    ax_b.text(0.5, -0.08, "|", fontsize=8,
              transform=ax_b.transAxes, ha="center", va="top", color="grey")

    # Annotate deltas
    ax_b.annotate(f"+{(hr_biased_e - hr_biased_c)*100:.1f} pp",
                  xy=(0.5, max(hr_biased_c, hr_biased_e) + 0.02),
                  fontsize=9, ha="center", color="#E65100", fontweight="bold")
    ax_b.annotate(f"{(hr_incl_e - hr_incl_c)*100:.1f} pp",
                  xy=(3.5, max(hr_incl_c, hr_incl_e) + 0.02),
                  fontsize=9, ha="center", color="#1565C0", fontweight="bold")

    # Bracket labels above bar groups
    ax_b.plot([0, 1], [0.88, 0.88], "k-", lw=0.8, transform=ax_b.get_xaxis_transform())
    ax_b.text(0.5, 0.89, "hit/(hit+miss)", fontsize=7, ha="center", color="grey",
              transform=ax_b.get_xaxis_transform())
    ax_b.plot([3, 4], [0.45, 0.45], "k-", lw=0.8, transform=ax_b.get_xaxis_transform())
    ax_b.text(3.5, 0.46, "hit/all go", fontsize=7, ha="center", color="grey",
              transform=ax_b.get_xaxis_transform())

    ax_b.set_title("B. Selection bias: biased vs inclusive hit rate",
                    fontweight="bold")
    stats.append({
        "test": "hit_rate_comparison",
        "hr_biased_correct": hr_biased_c, "hr_biased_error": hr_biased_e,
        "hr_inclusive_correct": hr_incl_c, "hr_inclusive_error": hr_incl_e,
        "biased_boost_pp": (hr_biased_e - hr_biased_c) * 100,
        "inclusive_boost_pp": (hr_incl_e - hr_incl_c) * 100,
    })

    # ── Panel C: Psychometric — biased vs inclusive ────────────────────
    ax_c = fig.add_subplot(gs[0, 2])

    # Plot inclusive (solid) and biased (dashed) for both conditions
    for cval, color in [("correct", COL_CORRECT), ("error", COL_ERROR)]:
        sub = go_first_valid[go_first_valid["post_cond"] == cval]
        # Inclusive
        hrs_i, errs_i = [], []
        for cs in CHANGE_SIZES:
            cs_sub = sub[sub["change_size"].between(cs - 0.01, cs + 0.01)]
            hr, se, _ = _inclusive_hit_rate(cs_sub)
            hrs_i.append(hr)
            errs_i.append(se)
        ax_c.errorbar(CHANGE_SIZE_POSITIONS, hrs_i, yerr=errs_i,
                       fmt="o-", color=color, lw=2, ms=6, capsize=3,
                       label=f"{cond_labels[cval]} (inclusive)")
        # Biased (dashed)
        hrs_b, errs_b = [], []
        for cs in CHANGE_SIZES:
            cs_sub = sub[sub["change_size"].between(cs - 0.01, cs + 0.01)]
            hr, se, _ = _biased_hit_rate(cs_sub)
            hrs_b.append(hr)
            errs_b.append(se)
        ax_c.errorbar(CHANGE_SIZE_POSITIONS, hrs_b, yerr=errs_b,
                       fmt="s--", color=color, lw=1.5, ms=5, capsize=3,
                       alpha=0.5, label=f"{cond_labels[cval]} (biased)")

    ax_c.set_xticks(CHANGE_SIZE_POSITIONS)
    ax_c.set_xticklabels(CHANGE_SIZE_LABELS, fontsize=8)
    ax_c.set_xlabel("Change size")
    ax_c.set_ylabel("Hit rate")
    ax_c.set_ylim(-0.05, 1.05)
    ax_c.legend(fontsize=6, loc="lower right")
    ax_c.set_title("C. Psychometric: solid=inclusive, dashed=biased",
                    fontweight="bold")

    # ==================================================================
    # ROW 2 — Expert stage only
    # ==================================================================
    go_expert = go_first_valid[go_first_valid["stage"] == "Expert"].copy()
    after_c_exp = go_expert[go_expert["post_cond"] == "correct"]
    after_e_exp = go_expert[go_expert["post_cond"] == "error"]

    # ── Panel D: Expert FA/abort rate ──────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    far_c_exp, far_c_se_exp, n_c_exp = _fa_abort_rate(after_c_exp)
    far_e_exp, far_e_se_exp, n_e_exp = _fa_abort_rate(after_e_exp)

    ax_d.bar([0, 1], [far_c_exp, far_e_exp],
             yerr=[far_c_se_exp, far_e_se_exp],
             color=[COL_CORRECT, COL_ERROR], edgecolor="k", lw=0.5,
             capsize=6, alpha=0.75, width=0.5)
    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels([f"After correct\n(n={n_c_exp})",
                           f"After error\n(n={n_e_exp})"])
    ax_d.set_ylabel("FA + abort rate on go trials")
    ax_d.set_ylim(0, 1)
    ax_d.set_title("D. Expert only: post-error impulsivity", fontweight="bold")

    ct_d = pd.crosstab(go_expert["post_cond"],
                        go_expert["outcome"].isin(["fa", "abort"]))
    if ct_d.shape == (2, 2):
        chi2_d, p_d, _, _ = chi2_contingency(ct_d)
        stats.append({"test": "chi2_fa_abort_expert_only",
                       "chi2": chi2_d, "p": p_d,
                       "far_correct": far_c_exp, "far_error": far_e_exp})
        ax_d.text(0.5, 0.95, f"χ²={chi2_d:.1f}, p={p_d:.2e} {_sig_str(p_d)}",
                  transform=ax_d.transAxes, fontsize=9, ha="center", va="top")

    # ── Panel E: Expert psychometric (inclusive metric) ─────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    _plot_psychometric(ax_e, go_expert, "post_cond",
                       ["correct", "error"], cond_colors, cond_labels,
                       hr_func=_inclusive_hit_rate,
                       title="E. Expert: psychometric (inclusive metric)")

    # ── Panel F: Expert RT distributions ───────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    hits_exp = go_expert[(go_expert["outcome"] == "hit")
                         & go_expert["rt"].notna()
                         & (go_expert["rt"] > 0)
                         & (go_expert["rt"] < 5)]
    rt_c = hits_exp[hits_exp["post_cond"] == "correct"]["rt"]
    rt_e = hits_exp[hits_exp["post_cond"] == "error"]["rt"]

    if len(rt_c) >= 5 and len(rt_e) >= 5:
        bp = ax_f.boxplot([rt_c.values, rt_e.values],
                          labels=[f"After correct\n(n={len(rt_c)})",
                                  f"After error\n(n={len(rt_e)})"],
                          patch_artist=True, widths=0.5, showfliers=False,
                          medianprops=dict(color="black", linewidth=2))
        for patch, c in zip(bp["boxes"], [COL_CORRECT, COL_ERROR]):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        u, p_f = mannwhitneyu(rt_c, rt_e, alternative="two-sided")
        delta_ms = (rt_e.median() - rt_c.median()) * 1000
        stats.append({"test": "mwu_hit_rt_expert_correct_vs_error",
                       "U": u, "p": p_f,
                       "median_correct_ms": rt_c.median() * 1000,
                       "median_error_ms": rt_e.median() * 1000,
                       "delta_ms": delta_ms})
        ax_f.text(0.5, 0.95,
                  f"ΔRT = {delta_ms:.0f} ms, p={p_f:.2e} {_sig_str(p_f)}",
                  transform=ax_f.transAxes, fontsize=9, ha="center", va="top")
    ax_f.set_ylabel("Hit RT (s)")
    ax_f.set_title("F. Expert: post-error RT", fontweight="bold")

    # ==================================================================
    # ROW 3 — Change-timing control
    # ==================================================================

    # ── Panel G: Change-time distributions ─────────────────────────────
    ax_g = fig.add_subplot(gs[2, 0])
    if "ct_bin" in go_first_valid.columns:
        for cval, color in cond_colors.items():
            sub = go_first_valid[(go_first_valid["post_cond"] == cval)
                                 & go_first_valid["change_time"].notna()]
            if not sub.empty:
                ax_g.hist(sub["change_time"], bins=40, color=color, alpha=0.5,
                          density=True, label=cond_labels[cval])
        # Show bin boundaries
        ct_valid = go_first_valid["change_time"].dropna()
        if len(ct_valid) > 30:
            q33 = ct_valid.quantile(1/3)
            q66 = ct_valid.quantile(2/3)
            ax_g.axvline(q33, color="grey", ls="--", lw=1, alpha=0.7)
            ax_g.axvline(q66, color="grey", ls="--", lw=1, alpha=0.7)
            ax_g.text(q33, ax_g.get_ylim()[1] * 0.95, "  33%ile",
                      fontsize=7, color="grey")
            ax_g.text(q66, ax_g.get_ylim()[1] * 0.95, "  66%ile",
                      fontsize=7, color="grey")
    ax_g.set_xlabel("Change time (s from trial start)")
    ax_g.set_ylabel("Density")
    ax_g.legend(fontsize=8)
    ax_g.set_title("G. Change-time distributions", fontweight="bold")

    # ── Panel H: Hit rate by change-time bin (inclusive + biased) ──────
    ax_h = fig.add_subplot(gs[2, 1])
    if go_first_valid["ct_bin"].notna().any():
        # 4 bar groups per bin: correct-incl, error-incl, correct-biased, error-biased
        x_off_incl = [-0.30, -0.10]
        x_off_bias = [0.10, 0.30]
        bar_w = 0.18
        for ci, (cval, color) in enumerate(cond_colors.items()):
            hrs_incl, errs_incl = [], []
            hrs_bias, errs_bias = [], []
            for bl in CT_BIN_LABELS:
                sub = go_first_valid[(go_first_valid["post_cond"] == cval)
                                     & (go_first_valid["ct_bin"] == bl)]
                hr_i, se_i, _ = _inclusive_hit_rate(sub)
                hr_b, se_b, _ = _biased_hit_rate(sub)
                hrs_incl.append(hr_i); errs_incl.append(se_i)
                hrs_bias.append(hr_b); errs_bias.append(se_b)
            lbl_i = f"{cond_labels[cval]} (inclusive)" if ci == 0 else f"{cond_labels[cval]} (inclusive)"
            lbl_b = f"{cond_labels[cval]} (biased)" if ci == 0 else f"{cond_labels[cval]} (biased)"
            ax_h.bar(np.arange(len(CT_BIN_LABELS)) + x_off_incl[ci],
                     hrs_incl, yerr=errs_incl, width=bar_w, color=color,
                     edgecolor="k", lw=0.5, capsize=3, alpha=0.75,
                     label=lbl_i)
            ax_h.bar(np.arange(len(CT_BIN_LABELS)) + x_off_bias[ci],
                     hrs_bias, yerr=errs_bias, width=bar_w, color=color,
                     edgecolor="k", lw=0.5, capsize=3, alpha=0.30,
                     hatch="//", label=lbl_b)
        ax_h.set_xticks(range(len(CT_BIN_LABELS)))
        ax_h.set_xticklabels(CT_BIN_LABELS)
        ax_h.set_ylabel("Hit rate")
        ax_h.set_ylim(0, 1.05)
        ax_h.legend(fontsize=6, loc="upper left", ncol=1)

        # Stats: is post-error deficit consistent across bins?
        for bl in CT_BIN_LABELS:
            sub_c = go_first_valid[(go_first_valid["post_cond"] == "correct")
                                   & (go_first_valid["ct_bin"] == bl)]
            sub_e = go_first_valid[(go_first_valid["post_cond"] == "error")
                                   & (go_first_valid["ct_bin"] == bl)]
            if len(sub_c) >= MIN_TRIALS and len(sub_e) >= MIN_TRIALS:
                ct_tab = pd.crosstab(
                    go_first_valid[(go_first_valid["ct_bin"] == bl)
                                   & go_first_valid["prev_outcome"].notna()]["post_cond"],
                    go_first_valid[(go_first_valid["ct_bin"] == bl)
                                   & go_first_valid["prev_outcome"].notna()]["is_hit"],
                )
                if ct_tab.shape == (2, 2):
                    chi2_h, p_h, _, _ = chi2_contingency(ct_tab)
                    hr_c_bin, _, _ = _inclusive_hit_rate(sub_c)
                    hr_e_bin, _, _ = _inclusive_hit_rate(sub_e)
                    stats.append({
                        "test": f"chi2_inclusive_hr_ctbin_{bl}",
                        "chi2": chi2_h, "p": p_h,
                        "hr_correct": hr_c_bin, "hr_error": hr_e_bin,
                    })
    ax_h.set_title("H. Hit rate by change timing\n(solid=inclusive, hatched=biased)", fontweight="bold")

    # ── Panel I: FA/abort rate by change-time bin ──────────────────────
    ax_i = fig.add_subplot(gs[2, 2])
    if go_first_valid["ct_bin"].notna().any():
        x_off_i = [-0.15, 0.15]
        for ci, (cval, color) in enumerate(cond_colors.items()):
            fars_ct, ferrs_ct = [], []
            for bl in CT_BIN_LABELS:
                sub = go_first_valid[(go_first_valid["post_cond"] == cval)
                                     & (go_first_valid["ct_bin"] == bl)]
                fr, se, _ = _fa_abort_rate(sub)
                fars_ct.append(fr)
                ferrs_ct.append(se)
            ax_i.bar(np.arange(len(CT_BIN_LABELS)) + x_off_i[ci], fars_ct,
                     yerr=ferrs_ct, width=0.28, color=color, edgecolor="k",
                     lw=0.5, capsize=4, alpha=0.75,
                     label=cond_labels[cval])
        ax_i.set_xticks(range(len(CT_BIN_LABELS)))
        ax_i.set_xticklabels(CT_BIN_LABELS)
        ax_i.set_ylabel("FA + abort rate")
        ax_i.set_ylim(0, 1)
        ax_i.legend(fontsize=8)
    ax_i.set_title("I. FA/abort rate by change timing", fontweight="bold")

    # ==================================================================
    # ROW 4 — Lick-rate & HMM-state controls
    # ==================================================================

    # ── Panel J: Per-session lick rate vs post-error FA excess ─────────
    ax_j = fig.add_subplot(gs[3, 0])
    sess_records = []
    for sn, grp in go_first_valid.groupby("session_name"):
        stage = grp["stage"].iloc[0]
        sidx = grp["session_idx"].iloc[0]
        lr = grp["sess_lick_rate"].iloc[0]
        ac = grp[grp["post_cond"] == "correct"]
        ae = grp[grp["post_cond"] == "error"]
        if len(ac) >= MIN_TRIALS and len(ae) >= MIN_TRIALS:
            far_c_s = ac["outcome"].isin(["fa", "abort"]).mean()
            far_e_s = ae["outcome"].isin(["fa", "abort"]).mean()
            hr_c_s = ac["is_hit"].mean()  # inclusive
            hr_e_s = ae["is_hit"].mean()
            sess_records.append({
                "session_name": sn, "stage": stage, "session_idx": sidx,
                "sess_lick_rate": lr,
                "fa_excess": far_e_s - far_c_s,
                "hr_inclusive_correct": hr_c_s, "hr_inclusive_error": hr_e_s,
                "delta_hr_inclusive": hr_e_s - hr_c_s,
            })
    sdf = pd.DataFrame(sess_records)

    if len(sdf) >= 5:
        for stage in STAGE_ORDER:
            sub = sdf[sdf["stage"] == stage]
            if sub.empty:
                continue
            ax_j.scatter(sub["sess_lick_rate"], sub["fa_excess"],
                         c=STAGE_COLORS[stage], s=50, edgecolors="white",
                         linewidths=0.5, label=stage, zorder=3)
        ax_j.axhline(0, color="grey", ls="--", lw=0.8, alpha=0.5)
        ax_j.set_xlabel("Session lick rate (fraction of trials with lick)")
        ax_j.set_ylabel("Post-error FA excess (ΔFA rate)")
        ax_j.legend(fontsize=8)

        rho_j, p_j = spearmanr(sdf["sess_lick_rate"], sdf["fa_excess"])
        stats.append({"test": "spearman_lick_rate_vs_fa_excess",
                       "rho": rho_j, "p": p_j, "n": len(sdf)})
        ax_j.text(0.05, 0.95, f"ρ={rho_j:.3f}, p={p_j:.3f}",
                  transform=ax_j.transAxes, fontsize=9, va="top")
    ax_j.set_title("J. Session lick rate vs post-error FA excess",
                    fontweight="bold")

    # ── Panel K: Inclusive hit rate split by median lick rate ───────────
    ax_k = fig.add_subplot(gs[3, 1])
    if len(sdf) >= 6:
        median_lr = sdf["sess_lick_rate"].median()
        low_sn = set(sdf[sdf["sess_lick_rate"] <= median_lr]["session_name"])
        high_sn = set(sdf[sdf["sess_lick_rate"] > median_lr]["session_name"])

        x_off_k = [-0.15, 0.15]
        for ci, (cval, color) in enumerate(cond_colors.items()):
            hrs_k, errs_k = [], []
            for g_label, g_set in [("Low lick", low_sn),
                                    ("High lick", high_sn)]:
                sub = go_first_valid[(go_first_valid["post_cond"] == cval)
                                     & go_first_valid["session_name"].isin(g_set)]
                hr, se, _ = _inclusive_hit_rate(sub)
                hrs_k.append(hr)
                errs_k.append(se)
            ax_k.bar(np.arange(2) + x_off_k[ci], hrs_k, yerr=errs_k,
                     width=0.28, color=color, edgecolor="k", lw=0.5,
                     capsize=4, alpha=0.75, label=cond_labels[cval])
        ax_k.set_xticks([0, 1])
        ax_k.set_xticklabels([f"Low lick rate\n(≤{median_lr:.2f})",
                               f"High lick rate\n(>{median_lr:.2f})"])
        ax_k.set_ylabel("Inclusive hit rate")
        ax_k.set_ylim(0, 0.6)
        ax_k.legend(fontsize=8)
    ax_k.set_title("K. Inclusive hit rate by session lick rate",
                    fontweight="bold")

    # ── Panel L: HMM-state breakdown ──────────────────────────────────
    ax_l = fig.add_subplot(gs[3, 2])
    hmm_col = "hmm_state"
    has_hmm = (hmm_col in go_first_valid.columns
               and go_first_valid[hmm_col].notna().sum() > 50)

    if has_hmm:
        available_states = [s for s in HMM_STATE_ORDER
                            if s in go_first_valid[hmm_col].values]
        x_off_l = [-0.15, 0.15]
        for ci, (cval, color) in enumerate(cond_colors.items()):
            hrs_l, errs_l = [], []
            for st in available_states:
                sub = go_first_valid[(go_first_valid["post_cond"] == cval)
                                     & (go_first_valid[hmm_col] == st)]
                hr, se, _ = _inclusive_hit_rate(sub)
                hrs_l.append(hr)
                errs_l.append(se)
            ax_l.bar(np.arange(len(available_states)) + x_off_l[ci],
                     hrs_l, yerr=errs_l, width=0.28, color=color,
                     edgecolor="k", lw=0.5, capsize=4, alpha=0.75,
                     label=cond_labels[cval])
        ax_l.set_xticks(range(len(available_states)))
        ax_l.set_xticklabels(available_states)
        ax_l.set_ylabel("Inclusive hit rate")
        ax_l.set_ylim(0, 0.6)
        ax_l.legend(fontsize=8)
        ax_l.set_title("L. Post-error effect by HMM-GLM state",
                        fontweight="bold")

        # Per-state chi-squared
        for st in available_states:
            st_sub = go_first_valid[go_first_valid[hmm_col] == st]
            if len(st_sub) >= 20:
                ct_l = pd.crosstab(st_sub["post_cond"], st_sub["is_hit"])
                if ct_l.shape == (2, 2):
                    chi2_l, p_l, _, _ = chi2_contingency(ct_l)
                    st_c = st_sub[st_sub["post_cond"] == "correct"]
                    st_e = st_sub[st_sub["post_cond"] == "error"]
                    stats.append({
                        "test": f"chi2_inclusive_hr_hmm_{st}",
                        "chi2": chi2_l, "p": p_l,
                        "hr_correct": st_c["is_hit"].mean(),
                        "hr_error": st_e["is_hit"].mean(),
                    })
    else:
        ax_l.text(0.5, 0.5,
                  "HMM-GLM states not yet available\n\n"
                  "Run HMM-GLM pipeline and\n"
                  "re-run this script to populate.",
                  ha="center", va="center", fontsize=10,
                  transform=ax_l.transAxes,
                  bbox=dict(boxstyle="round", facecolor="#FFF9C4",
                             edgecolor="#F9A825", alpha=0.9))
        ax_l.set_title("L. Post-error effect by HMM-GLM state",
                        fontweight="bold")

    # ==================================================================
    # Finalize
    # ==================================================================
    fig.suptitle(
        "Post-Error Performance: Selection Bias & Controlled Analyses (BG_046)",
        fontsize=15, fontweight="bold", y=0.995,
    )

    paths = save_figure(fig, "fig06_post_error_controls", "01_behavior")
    print(f"\n  Saved figure: {paths}")

    if stats:
        stats_df = pd.DataFrame(stats)
        stats_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "figures", "01_behavior", "post_error_controls_stats.csv",
        )
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        stats_df.to_csv(stats_path, index=False)
        print(f"  Saved statistics: {stats_path}")
        print("\n  Statistics summary:")
        for _, r in stats_df.iterrows():
            cols = {k: v for k, v in r.items() if pd.notna(v)}
            print(f"    {cols['test']}: p={cols.get('p', 'N/A')}")

    print("\n[01f] Done.")


if __name__ == "__main__":
    main()
