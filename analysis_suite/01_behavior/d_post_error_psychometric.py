"""Fig 04: Post-error psychometric — psychometric performance after error streaks.

Asks: Do mice perform better (psychometric curve) after a single
early-lick error (FA), after 2 consecutive errors, or after 3+
consecutive errors?  Does this depend on the timing (RT) of the
early lick on the previous trial?

Panels:
  A. Psychometric curves conditioned on n_preceding_errors (0, 1, 2, 3+)
  B. Hit rate (collapsed across change sizes) vs error-streak length
  C. Psychometric curves split by previous-FA RT (early vs late)
  D. Hit rate vs previous-FA RT (binned), for small vs big change sizes
  E. Per-session post-error psychometric shift (scatter: Learning vs Expert)
  F. Summary statistics table

Saves:
  figures/01_behavior/fig04_post_error_psychometric.png
  figures/01_behavior/post_error_psychometric_stats.csv
"""

import os
import sys
import gc
import warnings


import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal, sem, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import (
    STAGE_ORDER, STAGE_COLORS, OUTCOME_COLORS,
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
MIN_TRIALS = 5            # Minimum go trials per change-size bin
PKL_DIR = os.path.join(_root, "data", "pkls", "BG_046")
TRIAL_CACHE = os.path.join(CACHE_DIR, "all_trials_behavior.csv")
BOOL_COLS = ["is_hit", "is_miss", "is_fa", "is_abort", "is_ref", "is_go", "is_catch"]

ERROR_STREAK_COLORS = {
    0: "#4CAF50",   # After correct (green)
    1: "#FFC107",   # After 1 error (amber)
    2: "#FF9800",   # After 2 errors (orange)
    3: "#F44336",   # After 3+ errors (red)
}
ERROR_STREAK_LABELS = {
    0: "After correct",
    1: "After 1 error (FA)",
    2: "After 2 consec. errors",
    3: "After 3+ consec. errors",
}

FA_RT_EARLY_LABEL = f"Early FA (RT < {FA_RT_SPLIT}s)"
FA_RT_LATE_LABEL  = f"Late FA (RT ≥ {FA_RT_SPLIT}s)"


# ══════════════════════════════════════════════════════════════════════
# Section 1 — Data Loading
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("[01d] Post-error psychometric analysis")
    print("=" * 60)

    manifest = load_staging_manifest(qc_only=True)
    print(f"  {len(manifest)} QC-passed sessions")

    # ── Load all trials (with CSV cache) ──────────────────────────────
    if os.path.exists(TRIAL_CACHE):
        print(f"  Loading cached trial data from {os.path.basename(TRIAL_CACHE)}")
        trials = pd.read_csv(TRIAL_CACHE)
        for c in BOOL_COLS:
            if c in trials.columns:
                trials[c] = trials[c].astype(bool)
    else:
        print("  Loading session pickles (will cache for future runs)...")
        all_dfs = []
        for i, (_, row) in enumerate(manifest.iterrows()):
            sname = int(row["session_name"])
            stage = row["stage"]
            sidx  = row["session_idx"]
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

    print(f"  Total trials: {len(trials)} across {trials['session_name'].nunique()} sessions")

    # ── Compute preceding-error streak length for each trial ──────────
    # An "error" here = FA (early/anticipatory lick) or abort
    trials["is_error"] = trials["outcome"].isin(["fa", "abort"])

    def compute_error_streak(group):
        """For each trial, count how many consecutive errors preceded it."""
        is_err = group["is_error"].values
        streak = np.zeros(len(is_err), dtype=int)
        for i in range(1, len(is_err)):
            if is_err[i - 1]:
                streak[i] = streak[i - 1] + 1
            else:
                streak[i] = 0
        group = group.copy()
        group["n_preceding_errors"] = streak
        return group

    trials = trials.groupby("session_name", group_keys=False).apply(compute_error_streak)

    # Cap at 3+ for grouping
    trials["error_streak_bin"] = trials["n_preceding_errors"].clip(upper=3)

    # ── Add previous-trial FA RT ──────────────────────────────────────
    trials["prev_outcome"] = trials.groupby("session_name")["outcome"].shift(1)
    trials["prev_rt"] = trials.groupby("session_name")["rt"].shift(1)
    trials["prev_is_fa"] = trials["prev_outcome"] == "fa"

    # Classify previous FA RT as early/late
    trials["prev_fa_rt_class"] = np.nan
    mask_prev_fa = trials["prev_is_fa"] & trials["prev_rt"].notna()
    trials.loc[mask_prev_fa & (trials["prev_rt"] < FA_RT_SPLIT), "prev_fa_rt_class"] = "early"
    trials.loc[mask_prev_fa & (trials["prev_rt"] >= FA_RT_SPLIT), "prev_fa_rt_class"] = "late"

    # ── Change-time quantile bins ─────────────────────────────────────
    CT_BIN_LABELS = ["Early", "Mid", "Late"]
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

    print(f"  Error-streak distribution:")
    for k in sorted(ERROR_STREAK_LABELS.keys()):
        n = (trials["error_streak_bin"] == k).sum()
        print(f"    streak={k}: {n} trials")

    # ══════════════════════════════════════════════════════════════════
    # Section 2 — Analysis Helpers
    # ══════════════════════════════════════════════════════════════════

    def psychometric_by_condition(data, ax, condition_col, cond_values,
                                  colors, labels, title=""):
        """Plot psychometric curves split by a condition column.

        Hit rate = hits / (hits + misses) on go trials that reached the
        response window.  FA/abort trials are excluded from the denominator
        because the animal never saw the stimulus change.
        """
        for cval in cond_values:
            sub = data[data[condition_col] == cval]
            hrs, errs, ns_per_cs = [], [], []
            for i, cs in enumerate(CHANGE_SIZES):
                go = sub[(sub["change_size"].between(cs - 0.01, cs + 0.01))
                         & (sub["is_go"] == True)
                         & (sub["outcome"].isin(["hit", "miss"]))]
                n = len(go)
                if n >= MIN_TRIALS:
                    hr = go["is_hit"].mean()
                    se = np.sqrt(hr * (1 - hr) / n)
                else:
                    hr, se = np.nan, 0
                hrs.append(hr)
                errs.append(se)
                ns_per_cs.append(n)

            ax.errorbar(CHANGE_SIZE_POSITIONS, hrs, yerr=errs,
                        fmt="o-", color=colors[cval], lw=2, ms=6,
                        capsize=4, label=f"{labels[cval]} (n={sum(ns_per_cs)})")

        ax.set_xticks(CHANGE_SIZE_POSITIONS)
        ax.set_xticklabels(CHANGE_SIZE_LABELS, fontsize=8)
        ax.set_xlabel("Change size (TF ratio)")
        ax.set_ylabel("Hit rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")

    # ══════════════════════════════════════════════════════════════════
    # Section 3 — Figure
    # ══════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(22, 23))
    gs = gridspec.GridSpec(3, 3, hspace=0.40, wspace=0.35)
    stat_results = []

    # ── Go trials that reached the response window (hit or miss) ─────
    # FA/abort on go trials are excluded: the animal licked before the
    # stimulus change, so these trials never tested detection.
    go_trials = trials[(trials["is_go"] == True)
                       & (trials["outcome"].isin(["hit", "miss"]))].copy()
    print(f"  Go trials (hit+miss) for psychometric analysis: {len(go_trials)}")

    # ── Panel A: Psychometric curves by error-streak length ───────────
    ax_a = fig.add_subplot(gs[0, 0])
    psychometric_by_condition(
        go_trials, ax_a,
        condition_col="error_streak_bin",
        cond_values=[0, 1, 2, 3],
        colors=ERROR_STREAK_COLORS,
        labels=ERROR_STREAK_LABELS,
        title="A. Psychometric by preceding error streak",
    )

    # Statistics: chi-squared test per change size (streak 0 vs 1+)
    for cs in CHANGE_SIZES:
        cs_go = go_trials[go_trials["change_size"].between(cs - 0.01, cs + 0.01)
                          & go_trials["outcome"].isin(["hit", "miss"])]
        after_correct = cs_go[cs_go["error_streak_bin"] == 0]
        after_error   = cs_go[cs_go["error_streak_bin"] >= 1]
        if len(after_correct) >= MIN_TRIALS and len(after_error) >= MIN_TRIALS:
            ct = pd.crosstab(
                cs_go["error_streak_bin"].apply(lambda x: "correct" if x == 0 else "error"),
                cs_go["is_hit"],
            )
            if ct.shape == (2, 2):
                chi2, p, _, _ = chi2_contingency(ct)
                stat_results.append({
                    "test": f"chi2_correct_vs_error_cs{cs}",
                    "chi2": chi2, "p": p,
                    "n_correct": len(after_correct),
                    "n_error": len(after_error),
                    "hr_correct": after_correct["is_hit"].mean(),
                    "hr_error": after_error["is_hit"].mean(),
                })

    # ── Panel B: Hit rate vs error-streak length (all go trials) ──────
    ax_b = fig.add_subplot(gs[0, 1])
    streak_vals = sorted(go_trials["n_preceding_errors"].unique())
    streak_vals = [s for s in streak_vals if s <= 6]  # cap for display
    means_b, sems_b, ns_b = [], [], []
    for sv in streak_vals:
        sub = go_trials[go_trials["n_preceding_errors"] == sv]
        if len(sub) >= MIN_TRIALS:
            means_b.append(sub["is_hit"].mean())
            sems_b.append(np.sqrt(sub["is_hit"].mean() * (1 - sub["is_hit"].mean()) / len(sub)))
        else:
            means_b.append(np.nan)
            sems_b.append(0)
        ns_b.append(len(sub))

    bar_colors = [ERROR_STREAK_COLORS.get(min(s, 3), "#F44336") for s in streak_vals]
    bars = ax_b.bar(streak_vals, means_b, yerr=sems_b, color=bar_colors,
                    edgecolor="k", linewidth=0.5, capsize=4, alpha=0.85)
    for sv, n in zip(streak_vals, ns_b):
        ax_b.text(sv, -0.06, f"n={n}", ha="center", fontsize=7, color="grey")
    ax_b.set_xlabel("Number of preceding consecutive errors")
    ax_b.set_ylabel("Hit rate (go trials, hit+miss only)")
    ax_b.set_ylim(-0.10, 1.05)
    ax_b.set_title("B. Hit rate vs error-streak length", fontweight="bold")

    # Kruskal-Wallis across streak groups on per-session hit rates
    sess_hr_by_streak = []
    for sn, grp in go_trials.groupby("session_name"):
        for sb in [0, 1, 2, 3]:
            sub = grp[grp["error_streak_bin"] == sb]
            if len(sub) >= MIN_TRIALS:
                sess_hr_by_streak.append({"session_name": sn, "streak_bin": sb,
                                           "hit_rate": sub["is_hit"].mean()})
    sess_hr_df = pd.DataFrame(sess_hr_by_streak)
    groups_kw = [g["hit_rate"].values for _, g in sess_hr_df.groupby("streak_bin")
                 if len(g) >= 3]
    if len(groups_kw) >= 2:
        H, p_kw = kruskal(*groups_kw)
        stat_results.append({"test": "kruskal_hitrate_vs_streak_bin",
                              "H": H, "p": p_kw, "n_groups": len(groups_kw)})
        ax_b.text(0.95, 0.95, f"Kruskal H={H:.1f}, p={p_kw:.2e}",
                  transform=ax_b.transAxes, fontsize=8, va="top", ha="right")

    # Spearman correlation: streak length vs hit rate (trial-level)
    valid_b = go_trials[go_trials["n_preceding_errors"] <= 6].dropna(subset=["is_hit"])
    if len(valid_b) >= 20:
        rho_b, p_b = spearmanr(valid_b["n_preceding_errors"], valid_b["is_hit"].astype(float))
        stat_results.append({"test": "spearman_streak_vs_hit", "rho": rho_b,
                              "p": p_b, "n": len(valid_b)})
        ax_b.text(0.95, 0.87, f"ρ={rho_b:.3f}, p={p_b:.2e}",
                  transform=ax_b.transAxes, fontsize=8, va="top", ha="right")

    # ── Panel C: Psychometric split by previous FA RT (early vs late) ─
    ax_c = fig.add_subplot(gs[0, 2])
    # Only trials where previous trial was FA
    go_after_fa = go_trials[go_trials["prev_fa_rt_class"].isin(["early", "late"])].copy()
    print(f"  Go trials after FA with RT class: {len(go_after_fa)} "
          f"(early={sum(go_after_fa['prev_fa_rt_class']=='early')}, "
          f"late={sum(go_after_fa['prev_fa_rt_class']=='late')})")

    fa_rt_colors = {"early": "#FF5722", "late": "#2196F3"}
    fa_rt_labels = {"early": FA_RT_EARLY_LABEL, "late": FA_RT_LATE_LABEL}

    # Also plot "after correct" as reference
    go_after_correct = go_trials[go_trials["error_streak_bin"] == 0].copy()
    go_after_correct["prev_fa_rt_class"] = "correct"
    merged_c = pd.concat([go_after_fa, go_after_correct], ignore_index=True)
    fa_rt_colors["correct"] = "#4CAF50"
    fa_rt_labels["correct"] = "After correct (ref)"

    psychometric_by_condition(
        merged_c, ax_c,
        condition_col="prev_fa_rt_class",
        cond_values=["correct", "early", "late"],
        colors=fa_rt_colors,
        labels=fa_rt_labels,
        title="C. Psychometric by prev-FA timing",
    )

    # Chi-squared: early vs late FA effect on hit rate (all change sizes pooled)
    early_go = go_after_fa[go_after_fa["prev_fa_rt_class"] == "early"]
    late_go  = go_after_fa[go_after_fa["prev_fa_rt_class"] == "late"]
    if len(early_go) >= MIN_TRIALS and len(late_go) >= MIN_TRIALS:
        ct_el = pd.crosstab(go_after_fa["prev_fa_rt_class"], go_after_fa["is_hit"])
        if ct_el.shape == (2, 2):
            chi2_el, p_el, _, _ = chi2_contingency(ct_el)
            stat_results.append({
                "test": "chi2_early_vs_late_fa_hitrate",
                "chi2": chi2_el, "p": p_el,
                "hr_early": early_go["is_hit"].mean(),
                "hr_late": late_go["is_hit"].mean(),
                "n_early": len(early_go), "n_late": len(late_go),
            })

    # ── Panel D: Hit rate vs previous FA RT (binned) ──────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    go_after_fa_valid = go_trials[go_trials["prev_is_fa"] & go_trials["prev_rt"].notna()
                                    & go_trials["outcome"].isin(["hit", "miss"])].copy()

    if len(go_after_fa_valid) >= 20:
        # Bin previous FA RT into quantiles
        n_bins = 5
        go_after_fa_valid["fa_rt_bin"] = pd.qcut(
            go_after_fa_valid["prev_rt"], q=n_bins, duplicates="drop"
        )
        bin_order = sorted(go_after_fa_valid["fa_rt_bin"].dropna().unique(),
                           key=lambda x: x.mid)

        # Small vs big change sizes
        small_mask = go_after_fa_valid["change_size"].isin([1.25, 1.35, 1.5])
        big_mask   = go_after_fa_valid["change_size"].isin([2.0, 4.0])

        for mask, label, color, marker in [
            (small_mask, "Small Δ (1.25–1.5)", "#e74c3c", "o"),
            (big_mask,   "Big Δ (2.0–4.0)",     "#3498db", "s"),
        ]:
            sub = go_after_fa_valid[mask]
            bin_hrs, bin_sems, bin_mids = [], [], []
            for b in bin_order:
                bsub = sub[sub["fa_rt_bin"] == b]
                if len(bsub) >= MIN_TRIALS:
                    hr = bsub["is_hit"].mean()
                    bin_hrs.append(hr)
                    bin_sems.append(np.sqrt(hr * (1 - hr) / len(bsub)))
                else:
                    bin_hrs.append(np.nan)
                    bin_sems.append(0)
                bin_mids.append(b.mid)

            ax_d.errorbar(bin_mids, bin_hrs, yerr=bin_sems,
                          fmt=f"{marker}-", color=color, lw=2, ms=7,
                          capsize=4, label=label)

        ax_d.axvline(FA_RT_SPLIT, color="grey", ls="--", lw=1, alpha=0.6,
                     label=f"FA split ({FA_RT_SPLIT}s)")
        ax_d.set_xlabel("Previous FA reaction time (s)")
        ax_d.set_ylabel("Hit rate on next go trial")
        ax_d.set_ylim(-0.05, 1.05)
        ax_d.legend(fontsize=7, loc="lower right")
        ax_d.set_title("D. Hit rate vs prev-FA timing (binned)", fontweight="bold")

        # Spearman: prev_rt vs is_hit for trials after FA
        rho_d, p_d = spearmanr(go_after_fa_valid["prev_rt"],
                                go_after_fa_valid["is_hit"].astype(float))
        stat_results.append({"test": "spearman_prev_fa_rt_vs_hit",
                              "rho": rho_d, "p": p_d, "n": len(go_after_fa_valid)})
        ax_d.text(0.05, 0.05, f"ρ={rho_d:.3f}, p={p_d:.2e}",
                  transform=ax_d.transAxes, fontsize=8, va="bottom")
    else:
        ax_d.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                  transform=ax_d.transAxes, fontsize=12)
        ax_d.set_title("D. Hit rate vs prev-FA timing (binned)", fontweight="bold")

    # ── Panel E: Per-session post-error shift by stage ────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    # For each session, compute: HR(after 0 errors) - HR(after 1+ errors)
    sess_shift = []
    for sn, grp in go_trials.groupby("session_name"):
        stage = grp["stage"].iloc[0]
        sidx  = grp["session_idx"].iloc[0]
        after0 = grp[grp["error_streak_bin"] == 0]
        after1 = grp[grp["error_streak_bin"] >= 1]
        if len(after0) >= MIN_TRIALS and len(after1) >= MIN_TRIALS:
            hr0 = after0["is_hit"].mean()
            hr1 = after1["is_hit"].mean()
            sess_shift.append({
                "session_name": sn, "stage": stage, "session_idx": sidx,
                "hr_after_correct": hr0, "hr_after_error": hr1,
                "delta_hr": hr0 - hr1,
            })
    shift_df = pd.DataFrame(sess_shift)

    if not shift_df.empty:
        for stage in STAGE_ORDER:
            sub = shift_df[shift_df["stage"] == stage]
            if sub.empty:
                continue
            ax_e.scatter(sub["session_idx"], sub["delta_hr"],
                         c=STAGE_COLORS[stage], s=50, edgecolors="white",
                         linewidths=0.5, label=f"{stage} (n={len(sub)})", zorder=3)

        ax_e.axhline(0, color="grey", ls="--", lw=1, alpha=0.5)
        ax_e.set_xlabel("Session index")
        ax_e.set_ylabel("ΔHit rate (after correct − after error)")
        ax_e.set_title("E. Post-error hit-rate shift across learning", fontweight="bold")
        ax_e.legend(fontsize=7, loc="upper left")

        # Test if shift differs by stage
        for stage in STAGE_ORDER:
            sub = shift_df[shift_df["stage"] == stage]
            if len(sub) >= 3:
                from scipy.stats import wilcoxon
                try:
                    w, p_w = wilcoxon(sub["delta_hr"])
                    stat_results.append({
                        "test": f"wilcoxon_delta_hr_{stage}",
                        "W": w, "p": p_w,
                        "median_delta": sub["delta_hr"].median(),
                        "n": len(sub),
                    })
                except ValueError:
                    pass  # all zeros

        stage_groups = [g["delta_hr"].values for _, g in shift_df.groupby("stage")
                        if len(g) >= 3]
        if len(stage_groups) >= 2:
            H_e, p_e = kruskal(*stage_groups)
            stat_results.append({"test": "kruskal_delta_hr_by_stage",
                                  "H": H_e, "p": p_e})
            ax_e.text(0.95, 0.05, f"KW H={H_e:.1f}, p={p_e:.2e}",
                      transform=ax_e.transAxes, fontsize=8, va="bottom", ha="right")

    # ── Panel F: Psychometric by streak, split by stage ───────────────
    ax_f = fig.add_subplot(gs[1, 2])
    # Two sub-conditions: after correct vs after 1+ error, per stage
    # Use hatched vs solid to distinguish correct/error within each stage
    x_offset = 0
    legend_handles = []
    for si, stage in enumerate(STAGE_ORDER):
        stage_go = go_trials[go_trials["stage"] == stage]
        for ei, (ebin, elabel, alpha_val) in enumerate([
            (0, "correct", 1.0),
            (1, "1+ error", 0.55),
        ]):
            sub = stage_go[stage_go["error_streak_bin"] == ebin] if ebin == 0 else \
                  stage_go[stage_go["error_streak_bin"] >= 1]
            hrs, errs = [], []
            for cs in CHANGE_SIZES:
                cs_sub = sub[sub["change_size"].between(cs - 0.01, cs + 0.01)
                             & sub["outcome"].isin(["hit", "miss"])]
                if len(cs_sub) >= MIN_TRIALS:
                    hr = cs_sub["is_hit"].mean()
                    hrs.append(hr)
                    errs.append(np.sqrt(hr * (1 - hr) / len(cs_sub)))
                else:
                    hrs.append(np.nan)
                    errs.append(0)

            ls = "-" if ebin == 0 else "--"
            line = ax_f.errorbar(
                CHANGE_SIZE_POSITIONS, hrs, yerr=errs,
                fmt=f"o{ls}", color=STAGE_COLORS[stage], lw=2, ms=5,
                capsize=3, alpha=alpha_val,
                label=f"{stage} – after {elabel}",
            )
            legend_handles.append(line)

    ax_f.set_xticks(CHANGE_SIZE_POSITIONS)
    ax_f.set_xticklabels(CHANGE_SIZE_LABELS, fontsize=8)
    ax_f.set_xlabel("Change size (TF ratio)")
    ax_f.set_ylabel("Hit rate")
    ax_f.set_ylim(-0.05, 1.05)
    ax_f.set_title("F. Psychometric by stage × error history", fontweight="bold")
    ax_f.legend(fontsize=6, loc="lower right", ncol=1)

    # ── Panel G: Hit rate by change-time bin (post-correct vs post-error)
    ax_g = fig.add_subplot(gs[2, 0])
    if trials["ct_bin"].notna().any():
        x_off = [-0.15, 0.15]
        cond_colors_g = {0: "#4CAF50", 1: "#F44336"}
        cond_labels_g = {0: "After correct", 1: "After 1+ errors"}
        for ci, (ebin, color) in enumerate(cond_colors_g.items()):
            hrs_g, errs_g = [], []
            for bl in CT_BIN_LABELS:
                if ebin == 0:
                    sub = go_trials[(go_trials["error_streak_bin"] == 0)
                                    & (go_trials["ct_bin"] == bl)]
                else:
                    sub = go_trials[(go_trials["error_streak_bin"] >= 1)
                                    & (go_trials["ct_bin"] == bl)]
                n = len(sub)
                if n >= MIN_TRIALS:
                    hr = sub["is_hit"].mean()
                    se = np.sqrt(hr * (1 - hr) / n)
                else:
                    hr, se = np.nan, 0
                hrs_g.append(hr); errs_g.append(se)
            ax_g.bar(np.arange(len(CT_BIN_LABELS)) + x_off[ci],
                     hrs_g, yerr=errs_g, width=0.28, color=color,
                     edgecolor="k", lw=0.5, capsize=4, alpha=0.75,
                     label=cond_labels_g[ebin])
        ax_g.set_xticks(range(len(CT_BIN_LABELS)))
        ax_g.set_xticklabels(CT_BIN_LABELS)
        ax_g.set_ylabel("Hit rate (hit/(hit+miss))")
        ax_g.set_ylim(0, 1.05)
        ax_g.legend(fontsize=8)
    ax_g.set_title("G. Hit rate by change timing\n(standard/biased metric)",
                    fontweight="bold")

    # ── Finalize ──────────────────────────────────────────────────────
    fig.suptitle("Post-Error Psychometric Performance (BG_046)",
                 fontsize=14, fontweight="bold", y=0.98)

    paths = save_figure(fig, "fig04_post_error_psychometric", "01_behavior")
    print(f"  Saved figure: {paths}")

    # Save statistics
    if stat_results:
        stats_df = pd.DataFrame(stat_results)
        stats_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "figures", "01_behavior", "post_error_psychometric_stats.csv",
        )
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        stats_df.to_csv(stats_path, index=False)
        print(f"  Saved statistics: {stats_path}")
        print("\n  Statistics summary:")
        for _, r in stats_df.iterrows():
            print(f"    {r['test']}: p={r.get('p', 'N/A')}")

    print("\n[01d] Done.")


if __name__ == "__main__":
    main()
