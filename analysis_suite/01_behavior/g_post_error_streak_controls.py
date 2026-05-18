"""Fig 07: Post-error streak controls — error-streak analysis with selection-bias corrections.

Mirrors the error-streak structure of 01d (psychometric performance
after 0, 1, 2, 3+ consecutive early-lick errors) while incorporating
the selection-bias corrections and controls from 01f.

The critical insight: after errors the FA/abort rate on go trials
escalates with streak length, so the standard hit/(hit+miss)
psychometric is progressively biased.  This script shows both the
biased and inclusive (hit/all_go) metrics side by side, broken down
by consecutive error streak.

Panels (4 × 3 = 12):
  Row 1 — Error-streak psychometric: biased vs inclusive
    A. Psychometric by streak (solid=inclusive, dashed=biased)
    B. FA/abort rate on go trials by streak length
    C. Hit rate by streak: inclusive vs biased bar comparison

  Row 2 — Previous-FA timing (mirrors 01d Panels C–D)
    D. Inclusive psychometric by prev-FA timing (early/late/correct)
    E. Inclusive hit rate vs prev-FA RT (binned, small vs big Δ)
    F. FA/abort rate by prev-FA timing category

  Row 3 — Stage & longitudinal (mirrors 01d Panels E–F)
    G. Per-session post-error shift (inclusive metric)
    H. Psychometric by stage × error streak (inclusive)
    I. Expert-only: inclusive psychometric by streak

  Row 4 — Additional controls
    J. Inclusive hit rate by change-time bin × error history
    K. HMM-state × error history (inclusive)
    L. Session lick rate vs post-error inclusive HR deficit

Saves:
  figures/01_behavior/fig07_post_error_streak_controls.png
  figures/01_behavior/post_error_streak_controls_stats.csv
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
    FA_RT_SPLIT, CACHE_DIR,
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

ERROR_STREAK_COLORS = {
    0: "#4CAF50",   # After correct (green)
    1: "#FFC107",   # After 1 error (amber)
    2: "#FF9800",   # After 2 errors (orange)
    3: "#F44336",   # After 3+ errors (red)
}
ERROR_STREAK_LABELS = {
    0: "After correct",
    1: "After 1 error",
    2: "After 2 consec.",
    3: "After 3+ consec.",
}

FA_RT_EARLY_LABEL = f"Early FA (RT < {FA_RT_SPLIT}s)"
FA_RT_LATE_LABEL  = f"Late FA (RT ≥ {FA_RT_SPLIT}s)"

# Change-time quantile bins
N_CT_BINS = 3
CT_BIN_LABELS = ["Early", "Mid", "Late"]

# HMM state config
try:
    from visdetect.suite.config import HMM_STATE_ORDER, HMM_STATE_COLORS
    from visdetect.suite.loader import load_hmm_assignments
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False
    HMM_STATE_ORDER = []
    HMM_STATE_COLORS = {}


# ══════════════════════════════════════════════════════════════════════
# Data Loading & Preparation
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
            del sess; gc.collect()
            continue
        df["session_name"] = sname
        df["stage"] = stage
        df["session_idx"] = sidx
        all_dfs.append(df)
        del sess; gc.collect()
        print(f"    [{i+1}/{len(manifest)}] {date_str}: {len(df)} trials")

    trials = pd.concat(all_dfs, ignore_index=True)
    trials.to_csv(TRIAL_CACHE, index=False)
    print(f"  Cached {len(trials)} trials → {os.path.basename(TRIAL_CACHE)}")
    return trials


def prepare_trials(trials):
    """Add error-streak, prev-FA timing, change-time bins, HMM states."""
    # ── Error flag ────────────────────────────────────────────────────
    trials["is_error"] = trials["outcome"].isin(["fa", "abort"])

    # ── Error streak (consecutive FA/abort before each trial) ─────────
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

    # ── Previous-trial info ───────────────────────────────────────────
    trials["prev_outcome"] = trials.groupby("session_name")["outcome"].shift(1)
    trials["prev_rt"] = trials.groupby("session_name")["rt"].shift(1)
    trials["prev_is_fa"] = trials["prev_outcome"] == "fa"
    trials["prev_error"] = trials.groupby("session_name")["is_error"].shift(1)
    trials["prev_error"] = trials["prev_error"].fillna(False).astype(bool)

    # Classify previous FA RT as early/late
    trials["prev_fa_rt_class"] = np.nan
    mask_prev_fa = trials["prev_is_fa"] & trials["prev_rt"].notna()
    trials.loc[mask_prev_fa & (trials["prev_rt"] < FA_RT_SPLIT),
               "prev_fa_rt_class"] = "early"
    trials.loc[mask_prev_fa & (trials["prev_rt"] >= FA_RT_SPLIT),
               "prev_fa_rt_class"] = "late"

    # ── Per-session lick rate ─────────────────────────────────────────
    sess_lick = trials.groupby("session_name").apply(
        lambda g: g["outcome"].isin(["fa", "abort", "hit"]).mean()
    ).rename("sess_lick_rate")
    trials = trials.merge(sess_lick, on="session_name", how="left")

    # ── Change-time quantile bins ─────────────────────────────────────
    go_mask = trials["is_go"] == True
    if "change_time" in trials.columns:
        ct_valid = trials.loc[go_mask, "change_time"].dropna()
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

    # ── HMM state merge ──────────────────────────────────────────────
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
            trials.drop(columns=["hmm_state_label"], inplace=True,
                        errors="ignore")
            n_hmm = trials["hmm_state"].notna().sum()
            print(f"  HMM states merged: {n_hmm}/{len(trials)} trials labelled")
        except Exception as e:
            print(f"  HMM states not available ({e})")

    return trials


# ══════════════════════════════════════════════════════════════════════
# Hit-rate helpers
# ══════════════════════════════════════════════════════════════════════
def _inclusive_hr(sub):
    """hit / all_go (unbiased)."""
    n = len(sub)
    if n < MIN_TRIALS:
        return np.nan, 0, n
    hr = sub["is_hit"].mean()
    se = np.sqrt(hr * (1 - hr) / n)
    return hr, se, n


def _biased_hr(sub):
    """hit / (hit + miss) (standard but biased by FA/abort filtering)."""
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
    print("[01g] Post-error streak analysis with selection-bias controls")
    print("=" * 70)

    manifest = load_staging_manifest(qc_only=True)
    print(f"  {len(manifest)} QC-passed sessions")

    trials = load_trials(manifest)
    trials = prepare_trials(trials)

    # Go trials with prior-trial context (exclude trial 1 per session)
    go = trials[(trials["is_go"] == True)
                & trials["prev_outcome"].notna()].copy()
    print(f"  Total trials: {len(trials)}")
    print(f"  Go trials with prior context: {len(go)}")
    for sb in [0, 1, 2, 3]:
        n = (go["error_streak_bin"] == sb).sum()
        print(f"    streak={sb}: {n} go trials")

    fig = plt.figure(figsize=(24, 28))
    gs = gridspec.GridSpec(4, 3, hspace=0.42, wspace=0.35)
    stats = []

    streak_bins = [0, 1, 2, 3]

    # ==================================================================
    # ROW 1 — Error-streak psychometric: biased vs inclusive
    # ==================================================================

    # ── Panel A: Psychometric by streak — solid=inclusive, dashed=biased
    ax_a = fig.add_subplot(gs[0, 0])
    for sb in streak_bins:
        sub = go[go["error_streak_bin"] == sb]
        color = ERROR_STREAK_COLORS[sb]
        label = ERROR_STREAK_LABELS[sb]

        # Inclusive (solid)
        hrs_i, errs_i, ns_i = [], [], []
        for cs in CHANGE_SIZES:
            cs_sub = sub[sub["change_size"].between(cs - 0.01, cs + 0.01)]
            hr, se, n = _inclusive_hr(cs_sub)
            hrs_i.append(hr); errs_i.append(se); ns_i.append(n)
        ax_a.errorbar(CHANGE_SIZE_POSITIONS, hrs_i, yerr=errs_i,
                      fmt="o-", color=color, lw=2, ms=6, capsize=3,
                      label=f"{label} (n={sum(ns_i)})")

        # Biased (dashed, faded)
        hrs_b, errs_b = [], []
        for cs in CHANGE_SIZES:
            cs_sub = sub[sub["change_size"].between(cs - 0.01, cs + 0.01)]
            hr, se, _ = _biased_hr(cs_sub)
            hrs_b.append(hr); errs_b.append(se)
        ax_a.errorbar(CHANGE_SIZE_POSITIONS, hrs_b, yerr=errs_b,
                      fmt="s--", color=color, lw=1.2, ms=4, capsize=2,
                      alpha=0.4)

    ax_a.set_xticks(CHANGE_SIZE_POSITIONS)
    ax_a.set_xticklabels(CHANGE_SIZE_LABELS, fontsize=8)
    ax_a.set_xlabel("Change size (TF ratio)")
    ax_a.set_ylabel("Hit rate")
    ax_a.set_ylim(-0.05, 1.05)
    ax_a.legend(fontsize=6, loc="lower right")
    ax_a.set_title("A. Psychometric by streak\n(solid=inclusive, dashed=biased)",
                    fontweight="bold", fontsize=10)

    # ── Panel B: FA/abort rate by streak length ───────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    streak_vals = sorted(go["n_preceding_errors"].unique())
    streak_vals = [s for s in streak_vals if s <= 6]

    fars_b, ferrs_b, ns_b = [], [], []
    for sv in streak_vals:
        sub = go[go["n_preceding_errors"] == sv]
        fr, se, n = _fa_abort_rate(sub)
        fars_b.append(fr); ferrs_b.append(se); ns_b.append(n)

    bar_colors = [ERROR_STREAK_COLORS.get(min(s, 3), "#F44336")
                  for s in streak_vals]
    ax_b.bar(streak_vals, fars_b, yerr=ferrs_b, color=bar_colors,
             edgecolor="k", linewidth=0.5, capsize=4, alpha=0.85)
    for sv, n in zip(streak_vals, ns_b):
        ax_b.text(sv, -0.05, f"n={n}", ha="center", fontsize=7, color="grey")

    ax_b.set_xlabel("Number of preceding consecutive errors")
    ax_b.set_ylabel("FA + abort rate on go trials")
    ax_b.set_ylim(-0.08, 1.0)
    ax_b.set_title("B. Selection mechanism:\nFA/abort rate escalates with streak",
                    fontweight="bold", fontsize=10)

    # Spearman: streak vs FA/abort (trial-level)
    valid_b = go[go["n_preceding_errors"] <= 6].copy()
    valid_b["is_fa_abort"] = valid_b["outcome"].isin(["fa", "abort"])
    if len(valid_b) >= 20:
        rho_b, p_b = spearmanr(valid_b["n_preceding_errors"],
                                valid_b["is_fa_abort"].astype(float))
        stats.append({"test": "spearman_streak_vs_fa_abort_rate",
                       "rho": rho_b, "p": p_b, "n": len(valid_b)})
        ax_b.text(0.95, 0.95, f"ρ={rho_b:.3f}, p={p_b:.2e}",
                  transform=ax_b.transAxes, fontsize=9, va="top", ha="right")

    # ── Panel C: Hit rate by streak — inclusive vs biased bars ─────────
    ax_c = fig.add_subplot(gs[0, 2])
    x_pos = np.arange(len(streak_bins))
    width = 0.35

    hrs_incl, errs_incl = [], []
    hrs_bias, errs_bias = [], []
    for sb in streak_bins:
        sub = go[go["error_streak_bin"] == sb]
        hi, sei, _ = _inclusive_hr(sub)
        hb, seb, _ = _biased_hr(sub)
        hrs_incl.append(hi); errs_incl.append(sei)
        hrs_bias.append(hb); errs_bias.append(seb)

    bars1 = ax_c.bar(x_pos - width/2, hrs_incl, width, yerr=errs_incl,
                     color=[ERROR_STREAK_COLORS[s] for s in streak_bins],
                     edgecolor="k", linewidth=0.5, capsize=3, alpha=0.85,
                     label="Inclusive (hit/all go)")
    bars2 = ax_c.bar(x_pos + width/2, hrs_bias, width, yerr=errs_bias,
                     color=[ERROR_STREAK_COLORS[s] for s in streak_bins],
                     edgecolor="k", linewidth=0.5, capsize=3, alpha=0.30,
                     hatch="//", label="Biased (hit/(hit+miss))")

    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels([ERROR_STREAK_LABELS[s] for s in streak_bins],
                          fontsize=7, rotation=15, ha="right")
    ax_c.set_ylabel("Hit rate")
    ax_c.set_ylim(0, 1.05)
    ax_c.legend(fontsize=7, loc="upper left")
    ax_c.set_title("C. Inclusive vs biased hit rate by streak",
                    fontweight="bold", fontsize=10)

    # Annotate the bias gap
    for i, sb in enumerate(streak_bins):
        gap = (hrs_bias[i] - hrs_incl[i]) * 100 if not (
            np.isnan(hrs_bias[i]) or np.isnan(hrs_incl[i])) else 0
        if gap > 0:
            y_top = max(hrs_bias[i], hrs_incl[i]) + errs_bias[i] + 0.02
            ax_c.text(i, min(y_top, 1.0), f"+{gap:.0f}pp\nbias",
                      ha="center", fontsize=6, color="#E65100")

    # Stats: Kruskal on per-session inclusive HR across streak bins
    sess_hr_incl = []
    for sn, grp in go.groupby("session_name"):
        for sb in streak_bins:
            sub = grp[grp["error_streak_bin"] == sb]
            if len(sub) >= MIN_TRIALS:
                sess_hr_incl.append({"session_name": sn, "streak_bin": sb,
                                      "hit_rate_incl": sub["is_hit"].mean()})
    sess_df = pd.DataFrame(sess_hr_incl)
    groups_kw = [g["hit_rate_incl"].values
                 for _, g in sess_df.groupby("streak_bin") if len(g) >= 3]
    if len(groups_kw) >= 2:
        H, p_kw = kruskal(*groups_kw)
        stats.append({"test": "kruskal_inclusive_hr_vs_streak_bin",
                       "H": H, "p": p_kw, "n_groups": len(groups_kw)})
        ax_c.text(0.95, 0.95, f"KW H={H:.1f}, p={p_kw:.2e}",
                  transform=ax_c.transAxes, fontsize=8, va="top", ha="right")

    # ==================================================================
    # ROW 2 — Previous-FA timing (mirrors 01d Panels C–D)
    # ==================================================================

    # ── Panel D: Inclusive psychometric by prev-FA timing ──────────────
    ax_d = fig.add_subplot(gs[1, 0])
    go_after_fa = go[go["prev_fa_rt_class"].isin(["early", "late"])].copy()
    go_after_correct = go[go["error_streak_bin"] == 0].copy()
    go_after_correct["prev_fa_rt_class"] = "correct"
    merged_d = pd.concat([go_after_fa, go_after_correct], ignore_index=True)

    fa_rt_colors = {"correct": "#4CAF50", "early": "#FF5722", "late": "#2196F3"}
    fa_rt_labels_map = {"correct": "After correct (ref)",
                        "early": FA_RT_EARLY_LABEL,
                        "late": FA_RT_LATE_LABEL}

    for cval in ["correct", "early", "late"]:
        sub = merged_d[merged_d["prev_fa_rt_class"] == cval]
        hrs, errs, ns = [], [], []
        for cs in CHANGE_SIZES:
            cs_sub = sub[sub["change_size"].between(cs - 0.01, cs + 0.01)]
            hr, se, n = _inclusive_hr(cs_sub)
            hrs.append(hr); errs.append(se); ns.append(n)
        ax_d.errorbar(CHANGE_SIZE_POSITIONS, hrs, yerr=errs,
                      fmt="o-", color=fa_rt_colors[cval], lw=2, ms=6,
                      capsize=4,
                      label=f"{fa_rt_labels_map[cval]} (n={sum(ns)})")

    ax_d.set_xticks(CHANGE_SIZE_POSITIONS)
    ax_d.set_xticklabels(CHANGE_SIZE_LABELS, fontsize=8)
    ax_d.set_xlabel("Change size (TF ratio)")
    ax_d.set_ylabel("Inclusive hit rate")
    ax_d.set_ylim(-0.05, 1.05)
    ax_d.legend(fontsize=6, loc="lower right")
    ax_d.set_title("D. Inclusive psychometric by prev-FA timing",
                    fontweight="bold", fontsize=10)

    print(f"  Go trials after FA with RT class: {len(go_after_fa)} "
          f"(early={sum(go_after_fa['prev_fa_rt_class']=='early')}, "
          f"late={sum(go_after_fa['prev_fa_rt_class']=='late')})")

    # Chi-squared: early vs late FA (inclusive — on all go trials)
    if len(go_after_fa) >= 20:
        ct_d = pd.crosstab(go_after_fa["prev_fa_rt_class"],
                            go_after_fa["is_hit"])
        if ct_d.shape == (2, 2):
            chi2_d, p_d, _, _ = chi2_contingency(ct_d)
            hr_early = go_after_fa[
                go_after_fa["prev_fa_rt_class"] == "early"]["is_hit"].mean()
            hr_late = go_after_fa[
                go_after_fa["prev_fa_rt_class"] == "late"]["is_hit"].mean()
            stats.append({
                "test": "chi2_inclusive_early_vs_late_fa",
                "chi2": chi2_d, "p": p_d,
                "hr_early_incl": hr_early, "hr_late_incl": hr_late,
            })

    # ── Panel E: Inclusive hit rate vs prev-FA RT (binned) ─────────────
    ax_e = fig.add_subplot(gs[1, 1])
    go_after_fa_valid = go[go["prev_is_fa"]
                           & go["prev_rt"].notna()].copy()

    if len(go_after_fa_valid) >= 20:
        n_bins = 5
        go_after_fa_valid["fa_rt_bin"] = pd.qcut(
            go_after_fa_valid["prev_rt"], q=n_bins, duplicates="drop"
        )
        bin_order = sorted(go_after_fa_valid["fa_rt_bin"].dropna().unique(),
                           key=lambda x: x.mid)

        small_mask = go_after_fa_valid["change_size"].isin([1.25, 1.35, 1.5])
        big_mask = go_after_fa_valid["change_size"].isin([2.0, 4.0])

        for mask, label, color, marker in [
            (small_mask, "Small Δ (1.25–1.5)", "#e74c3c", "o"),
            (big_mask,   "Big Δ (2.0–4.0)",    "#3498db", "s"),
        ]:
            sub = go_after_fa_valid[mask]
            bin_hrs, bin_sems, bin_mids = [], [], []
            for b in bin_order:
                bsub = sub[sub["fa_rt_bin"] == b]
                hr, se, _ = _inclusive_hr(bsub)
                bin_hrs.append(hr); bin_sems.append(se)
                bin_mids.append(b.mid)
            ax_e.errorbar(bin_mids, bin_hrs, yerr=bin_sems,
                          fmt=f"{marker}-", color=color, lw=2, ms=7,
                          capsize=4, label=label)

        ax_e.axvline(FA_RT_SPLIT, color="grey", ls="--", lw=1, alpha=0.6,
                     label=f"FA split ({FA_RT_SPLIT}s)")
        ax_e.set_xlabel("Previous FA reaction time (s)")
        ax_e.set_ylabel("Inclusive hit rate on next go trial")
        ax_e.set_ylim(-0.05, 0.75)
        ax_e.legend(fontsize=7, loc="lower right")
        ax_e.set_title("E. Inclusive HR vs prev-FA timing (binned)",
                        fontweight="bold", fontsize=10)

        rho_e, p_e = spearmanr(go_after_fa_valid["prev_rt"],
                                go_after_fa_valid["is_hit"].astype(float))
        stats.append({"test": "spearman_prev_fa_rt_vs_inclusive_hit",
                       "rho": rho_e, "p": p_e, "n": len(go_after_fa_valid)})
        ax_e.text(0.05, 0.05, f"ρ={rho_e:.3f}, p={p_e:.2e}",
                  transform=ax_e.transAxes, fontsize=8, va="bottom")
    else:
        ax_e.text(0.5, 0.5, "Insufficient data",
                  ha="center", va="center", transform=ax_e.transAxes)
        ax_e.set_title("E. Inclusive HR vs prev-FA timing (binned)",
                        fontweight="bold", fontsize=10)

    # ── Panel F: FA/abort rate by prev-FA timing category ─────────────
    ax_f = fig.add_subplot(gs[1, 2])
    fa_cats = ["correct", "early", "late"]
    fa_cat_colors = [fa_rt_colors[c] for c in fa_cats]
    fa_cat_labels = [fa_rt_labels_map[c] for c in fa_cats]

    fars_f, ferrs_f, ns_f = [], [], []
    for cat in fa_cats:
        if cat == "correct":
            sub = go[go["error_streak_bin"] == 0]
        else:
            sub = go[go["prev_fa_rt_class"] == cat]
        fr, se, n = _fa_abort_rate(sub)
        fars_f.append(fr); ferrs_f.append(se); ns_f.append(n)

    ax_f.bar(range(len(fa_cats)), fars_f, yerr=ferrs_f,
             color=fa_cat_colors, edgecolor="k", linewidth=0.5,
             capsize=5, alpha=0.85, width=0.55)
    for i, n in enumerate(ns_f):
        ax_f.text(i, -0.05, f"n={n}", ha="center", fontsize=7, color="grey")

    ax_f.set_xticks(range(len(fa_cats)))
    ax_f.set_xticklabels(["After\ncorrect", f"After early FA\n(RT<{FA_RT_SPLIT}s)",
                           f"After late FA\n(RT≥{FA_RT_SPLIT}s)"], fontsize=7)
    ax_f.set_ylabel("FA + abort rate on go trials")
    ax_f.set_ylim(-0.08, 1.0)
    ax_f.set_title("F. FA/abort rate by prev-trial category",
                    fontweight="bold", fontsize=10)

    # Chi-squared across all three
    fa_for_chi = go[go["prev_fa_rt_class"].isin(["early", "late"])
                    | (go["error_streak_bin"] == 0)].copy()
    fa_for_chi["cat"] = np.where(
        fa_for_chi["error_streak_bin"] == 0, "correct",
        fa_for_chi["prev_fa_rt_class"])
    fa_for_chi = fa_for_chi[fa_for_chi["cat"].isin(fa_cats)]
    fa_for_chi["is_fa_abort"] = fa_for_chi["outcome"].isin(["fa", "abort"])
    ct_f = pd.crosstab(fa_for_chi["cat"], fa_for_chi["is_fa_abort"])
    if ct_f.shape[0] >= 2 and ct_f.shape[1] == 2:
        chi2_f, p_f, _, _ = chi2_contingency(ct_f)
        stats.append({"test": "chi2_fa_abort_by_prev_category",
                       "chi2": chi2_f, "p": p_f})
        ax_f.text(0.95, 0.95, f"χ²={chi2_f:.1f}, p={p_f:.2e}",
                  transform=ax_f.transAxes, fontsize=8, va="top", ha="right")

    # ==================================================================
    # ROW 3 — Stage & longitudinal (mirrors 01d Panels E–F)
    # ==================================================================

    # ── Panel G: Per-session post-error shift (inclusive metric) ───────
    ax_g = fig.add_subplot(gs[2, 0])
    sess_shift = []
    for sn, grp in go.groupby("session_name"):
        stage = grp["stage"].iloc[0]
        sidx = grp["session_idx"].iloc[0]
        after0 = grp[grp["error_streak_bin"] == 0]
        after1 = grp[grp["error_streak_bin"] >= 1]
        if len(after0) >= MIN_TRIALS and len(after1) >= MIN_TRIALS:
            hr0 = after0["is_hit"].mean()   # inclusive
            hr1 = after1["is_hit"].mean()   # inclusive
            sess_shift.append({
                "session_name": sn, "stage": stage, "session_idx": sidx,
                "hr_incl_correct": hr0, "hr_incl_error": hr1,
                "delta_hr_incl": hr0 - hr1,
            })
    shift_df = pd.DataFrame(sess_shift)

    if not shift_df.empty:
        for stage in STAGE_ORDER:
            sub = shift_df[shift_df["stage"] == stage]
            if sub.empty:
                continue
            ax_g.scatter(sub["session_idx"], sub["delta_hr_incl"],
                         c=STAGE_COLORS[stage], s=50, edgecolors="white",
                         linewidths=0.5, label=f"{stage} (n={len(sub)})",
                         zorder=3)

        ax_g.axhline(0, color="grey", ls="--", lw=1, alpha=0.5)
        ax_g.set_xlabel("Session index")
        ax_g.set_ylabel("ΔInclusive HR\n(after correct − after error)")
        ax_g.set_title("G. Post-error inclusive-HR shift across learning",
                        fontweight="bold", fontsize=10)
        ax_g.legend(fontsize=7, loc="upper left")

        # Wilcoxon per stage
        for stage in STAGE_ORDER:
            sub = shift_df[shift_df["stage"] == stage]
            if len(sub) >= 3:
                try:
                    w, p_w = wilcoxon(sub["delta_hr_incl"])
                    stats.append({
                        "test": f"wilcoxon_delta_incl_hr_{stage}",
                        "W": w, "p": p_w,
                        "median_delta": sub["delta_hr_incl"].median(),
                        "n": len(sub),
                    })
                except ValueError:
                    pass

        # Kruskal across stages
        stage_groups = [g["delta_hr_incl"].values
                        for _, g in shift_df.groupby("stage") if len(g) >= 3]
        if len(stage_groups) >= 2:
            H_g, p_g = kruskal(*stage_groups)
            stats.append({"test": "kruskal_delta_incl_hr_by_stage",
                           "H": H_g, "p": p_g})
            ax_g.text(0.95, 0.05, f"KW H={H_g:.1f}, p={p_g:.2e}",
                      transform=ax_g.transAxes, fontsize=8,
                      va="bottom", ha="right")

    # ── Panel H: Psychometric by stage × error streak (inclusive) ─────
    ax_h = fig.add_subplot(gs[2, 1])
    for si, stage in enumerate(STAGE_ORDER):
        stage_go = go[go["stage"] == stage]
        for ebin, elabel, alpha_val in [(0, "correct", 1.0),
                                         (1, "1+ errors", 0.55)]:
            sub = (stage_go[stage_go["error_streak_bin"] == ebin] if ebin == 0
                   else stage_go[stage_go["error_streak_bin"] >= 1])
            hrs, errs = [], []
            for cs in CHANGE_SIZES:
                cs_sub = sub[sub["change_size"].between(cs - 0.01, cs + 0.01)]
                hr, se, _ = _inclusive_hr(cs_sub)
                hrs.append(hr); errs.append(se)

            ls = "-" if ebin == 0 else "--"
            ax_h.errorbar(CHANGE_SIZE_POSITIONS, hrs, yerr=errs,
                          fmt=f"o{ls}", color=STAGE_COLORS[stage], lw=2,
                          ms=5, capsize=3, alpha=alpha_val,
                          label=f"{stage} – {elabel}")

    ax_h.set_xticks(CHANGE_SIZE_POSITIONS)
    ax_h.set_xticklabels(CHANGE_SIZE_LABELS, fontsize=8)
    ax_h.set_xlabel("Change size (TF ratio)")
    ax_h.set_ylabel("Inclusive hit rate")
    ax_h.set_ylim(-0.05, 0.75)
    ax_h.legend(fontsize=6, loc="lower right", ncol=1)
    ax_h.set_title("H. Stage × error streak (inclusive)",
                    fontweight="bold", fontsize=10)

    # ── Panel I: Expert-only: inclusive psychometric by streak ─────────
    ax_i = fig.add_subplot(gs[2, 2])
    go_expert = go[go["stage"] == "Expert"]

    for sb in streak_bins:
        sub = go_expert[go_expert["error_streak_bin"] == sb]
        hrs, errs, ns = [], [], []
        for cs in CHANGE_SIZES:
            cs_sub = sub[sub["change_size"].between(cs - 0.01, cs + 0.01)]
            hr, se, n = _inclusive_hr(cs_sub)
            hrs.append(hr); errs.append(se); ns.append(n)
        total_n = sum(ns)
        if total_n >= MIN_TRIALS:
            ax_i.errorbar(CHANGE_SIZE_POSITIONS, hrs, yerr=errs,
                          fmt="o-", color=ERROR_STREAK_COLORS[sb], lw=2,
                          ms=6, capsize=4,
                          label=f"{ERROR_STREAK_LABELS[sb]} (n={total_n})")

    ax_i.set_xticks(CHANGE_SIZE_POSITIONS)
    ax_i.set_xticklabels(CHANGE_SIZE_LABELS, fontsize=8)
    ax_i.set_xlabel("Change size (TF ratio)")
    ax_i.set_ylabel("Inclusive hit rate")
    ax_i.set_ylim(-0.05, 0.75)
    ax_i.legend(fontsize=6, loc="lower right")
    ax_i.set_title("I. Expert only: inclusive psychometric by streak",
                    fontweight="bold", fontsize=10)

    # Expert-level chi-squared: streak 0 vs 1+
    exp_after0 = go_expert[go_expert["error_streak_bin"] == 0]
    exp_after1 = go_expert[go_expert["error_streak_bin"] >= 1]
    if len(exp_after0) >= MIN_TRIALS and len(exp_after1) >= MIN_TRIALS:
        ct_i = pd.crosstab(
            go_expert["error_streak_bin"].apply(
                lambda x: "correct" if x == 0 else "error"),
            go_expert["is_hit"],
        )
        if ct_i.shape == (2, 2):
            chi2_i, p_i, _, _ = chi2_contingency(ct_i)
            stats.append({
                "test": "chi2_inclusive_hr_expert_streak0_vs1plus",
                "chi2": chi2_i, "p": p_i,
                "hr_correct": exp_after0["is_hit"].mean(),
                "hr_error": exp_after1["is_hit"].mean(),
            })
            ax_i.text(0.95, 0.95, f"χ²={chi2_i:.1f}, p={p_i:.2e}",
                      transform=ax_i.transAxes, fontsize=8,
                      va="top", ha="right")

    # ==================================================================
    # ROW 4 — Additional controls
    # ==================================================================

    # ── Panel J: HR by change-time bin × error history (inclusive + biased)
    ax_j = fig.add_subplot(gs[3, 0])
    if go["ct_bin"].notna().any():
        x_off_incl = [-0.30, -0.10]
        x_off_bias = [0.10, 0.30]
        bar_w = 0.18
        cond_vals = [("correct", 0, "#4CAF50", "After correct"),
                     ("error", None, "#F44336", "After 1+ errors")]
        for ci, (_, sb_val, color, label) in enumerate(cond_vals):
            hrs_incl, errs_incl = [], []
            hrs_bias, errs_bias = [], []
            for bl in CT_BIN_LABELS:
                if sb_val is not None:
                    sub = go[(go["error_streak_bin"] == sb_val)
                             & (go["ct_bin"] == bl)]
                else:
                    sub = go[(go["error_streak_bin"] >= 1)
                             & (go["ct_bin"] == bl)]
                hr_i, se_i, _ = _inclusive_hr(sub)
                hr_b, se_b, _ = _biased_hr(sub)
                hrs_incl.append(hr_i); errs_incl.append(se_i)
                hrs_bias.append(hr_b); errs_bias.append(se_b)
            ax_j.bar(np.arange(len(CT_BIN_LABELS)) + x_off_incl[ci],
                     hrs_incl, yerr=errs_incl, width=bar_w, color=color,
                     edgecolor="k", lw=0.5, capsize=3, alpha=0.75,
                     label=f"{label} (inclusive)")
            ax_j.bar(np.arange(len(CT_BIN_LABELS)) + x_off_bias[ci],
                     hrs_bias, yerr=errs_bias, width=bar_w, color=color,
                     edgecolor="k", lw=0.5, capsize=3, alpha=0.30,
                     hatch="//", label=f"{label} (biased)")

        ax_j.set_xticks(range(len(CT_BIN_LABELS)))
        ax_j.set_xticklabels(CT_BIN_LABELS)
        ax_j.set_ylabel("Hit rate")
        ax_j.set_ylim(0, 1.05)
        ax_j.legend(fontsize=5.5, loc="upper left", ncol=1)

        # Stats per bin
        for bl in CT_BIN_LABELS:
            sub_c = go[(go["error_streak_bin"] == 0) & (go["ct_bin"] == bl)]
            sub_e = go[(go["error_streak_bin"] >= 1) & (go["ct_bin"] == bl)]
            if len(sub_c) >= MIN_TRIALS and len(sub_e) >= MIN_TRIALS:
                combined = pd.concat([sub_c, sub_e])
                ct_j = pd.crosstab(
                    combined["error_streak_bin"].apply(
                        lambda x: "correct" if x == 0 else "error"),
                    combined["is_hit"],
                )
                if ct_j.shape == (2, 2):
                    chi2_j, p_j, _, _ = chi2_contingency(ct_j)
                    stats.append({
                        "test": f"chi2_inclusive_hr_ctbin_{bl}",
                        "chi2": chi2_j, "p": p_j,
                        "hr_correct": sub_c["is_hit"].mean(),
                        "hr_error": sub_e["is_hit"].mean(),
                    })
    ax_j.set_title("J. HR by change-time bin\n(solid=inclusive, hatched=biased)",
                    fontweight="bold", fontsize=10)

    # ── Panel K: HMM-state × error streak (inclusive) ─────────────────
    ax_k = fig.add_subplot(gs[3, 1])
    hmm_col = "hmm_state"
    has_hmm = (hmm_col in go.columns and go[hmm_col].notna().sum() > 50)

    if has_hmm:
        available_states = [s for s in HMM_STATE_ORDER
                            if s in go[hmm_col].values]
        x_off_k = [-0.17, 0.17]
        cond_vals_k = [(0, "#4CAF50", "After correct"),
                       (None, "#F44336", "After 1+ errors")]
        for ci, (sb_val, color, label) in enumerate(cond_vals_k):
            hrs_k, errs_k = [], []
            for st in available_states:
                if sb_val is not None:
                    sub = go[(go["error_streak_bin"] == sb_val)
                             & (go[hmm_col] == st)]
                else:
                    sub = go[(go["error_streak_bin"] >= 1)
                             & (go[hmm_col] == st)]
                hr, se, _ = _inclusive_hr(sub)
                hrs_k.append(hr); errs_k.append(se)
            ax_k.bar(np.arange(len(available_states)) + x_off_k[ci],
                     hrs_k, yerr=errs_k, width=0.32, color=color,
                     edgecolor="k", lw=0.5, capsize=4, alpha=0.75,
                     label=label)

        ax_k.set_xticks(range(len(available_states)))
        ax_k.set_xticklabels(available_states)
        ax_k.set_ylabel("Inclusive hit rate")
        ax_k.set_ylim(0, 0.60)
        ax_k.legend(fontsize=7)

        # Per-state chi-squared
        for st in available_states:
            st_sub = go[go[hmm_col] == st]
            st_c = st_sub[st_sub["error_streak_bin"] == 0]
            st_e = st_sub[st_sub["error_streak_bin"] >= 1]
            if len(st_c) >= MIN_TRIALS and len(st_e) >= MIN_TRIALS:
                ct_k = pd.crosstab(
                    st_sub["error_streak_bin"].apply(
                        lambda x: "correct" if x == 0 else "error"),
                    st_sub["is_hit"],
                )
                if ct_k.shape == (2, 2):
                    chi2_k, p_k, _, _ = chi2_contingency(ct_k)
                    stats.append({
                        "test": f"chi2_inclusive_hr_hmm_{st}",
                        "chi2": chi2_k, "p": p_k,
                        "hr_correct": st_c["is_hit"].mean(),
                        "hr_error": st_e["is_hit"].mean(),
                    })
        ax_k.set_title("K. HMM state × error streak (inclusive)",
                        fontweight="bold", fontsize=10)
    else:
        ax_k.text(0.5, 0.5,
                  "HMM states not available\n\n"
                  "Run HMM pipeline and\nre-run to populate.",
                  ha="center", va="center", fontsize=10,
                  transform=ax_k.transAxes,
                  bbox=dict(boxstyle="round", facecolor="#FFF9C4",
                            edgecolor="#F9A825", alpha=0.9))
        ax_k.set_title("K. HMM state × error streak (inclusive)",
                        fontweight="bold", fontsize=10)

    # ── Panel L: Session lick rate vs post-error inclusive HR deficit ──
    ax_l = fig.add_subplot(gs[3, 2])
    if not shift_df.empty:
        shift_df["sess_lick_rate"] = shift_df["session_name"].map(
            trials.drop_duplicates("session_name").set_index(
                "session_name")["sess_lick_rate"])

        for stage in STAGE_ORDER:
            sub = shift_df[shift_df["stage"] == stage]
            if sub.empty:
                continue
            ax_l.scatter(sub["sess_lick_rate"], sub["delta_hr_incl"],
                         c=STAGE_COLORS[stage], s=50, edgecolors="white",
                         linewidths=0.5, label=stage, zorder=3)

        ax_l.axhline(0, color="grey", ls="--", lw=0.8, alpha=0.5)
        ax_l.set_xlabel("Session lick rate")
        ax_l.set_ylabel("ΔInclusive HR\n(correct − error)")
        ax_l.legend(fontsize=7)

        valid_l = shift_df.dropna(subset=["sess_lick_rate", "delta_hr_incl"])
        if len(valid_l) >= 5:
            rho_l, p_l = spearmanr(valid_l["sess_lick_rate"],
                                    valid_l["delta_hr_incl"])
            stats.append({"test": "spearman_lick_rate_vs_delta_incl_hr",
                           "rho": rho_l, "p": p_l, "n": len(valid_l)})
            ax_l.text(0.05, 0.95, f"ρ={rho_l:.3f}, p={p_l:.3f}",
                      transform=ax_l.transAxes, fontsize=9, va="top")
    ax_l.set_title("L. Session lick rate vs post-error\ninclusive HR deficit",
                    fontweight="bold", fontsize=10)

    # ==================================================================
    # Finalize
    # ==================================================================
    fig.suptitle(
        "Post-Error Streak Analysis with Selection-Bias Controls (BG_046)",
        fontsize=15, fontweight="bold", y=0.995,
    )

    paths = save_figure(fig, "fig07_post_error_streak_controls", "01_behavior")
    print(f"\n  Saved figure: {paths}")

    if stats:
        stats_df = pd.DataFrame(stats)
        stats_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "figures", "01_behavior",
            "post_error_streak_controls_stats.csv",
        )
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        stats_df.to_csv(stats_path, index=False)
        print(f"  Saved statistics: {stats_path}")
        print("\n  Statistics summary:")
        for _, r in stats_df.iterrows():
            cols = {k: v for k, v in r.items() if pd.notna(v)}
            print(f"    {cols['test']}: p={cols.get('p', 'N/A')}")

    print("\n[01g] Done.")


if __name__ == "__main__":
    main()
