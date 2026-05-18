"""Fig 05: Post-error dynamics — speed-accuracy tradeoff and criterion shift.

Follow-up to 01d (post-error psychometric), which found a surprising post-error
accuracy BOOST.  This script asks: *why* does accuracy improve after errors?

Three competing (non-exclusive) explanations:
  1. Speed-accuracy tradeoff: the mouse slows down after errors → more time →
     better hit rate.
  2. Criterion shift: the mouse becomes more conservative → fewer FAs on catch
     trials → higher d'.
  3. Attentional reorienting: errors prime increased vigilance on the next trial.

Panels:
  A. Post-error RT slowing (hit RT distributions: after correct vs after error)
  B. Speed-accuracy tradeoff (per-session scatter: ΔRT vs ΔHR)
  C. Post-error criterion shift (FA rate on catch trials: after correct vs error)
  D. Temporal decay of the post-error boost (hit rate at lag 1, 2, 3, …)
  E. Trial-outcome transition matrix (heatmap)
  F. Stage emergence: post-error slowing trajectory across learning

Saves:
  figures/01_behavior/fig05_post_error_dynamics.png
  figures/01_behavior/post_error_dynamics_stats.csv

Reuses the cached trial table from 01d (all_trials_behavior.csv).
"""

import os
import sys
import gc
import warnings


import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon, chi2_contingency

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import (
    STAGE_ORDER, STAGE_COLORS, OUTCOME_COLORS,
    CHANGE_SIZES, CACHE_DIR,
)
from visdetect.suite.loader import load_staging_manifest, load_session
from visdetect.suite.plotting import setup_style, save_figure, add_stage_background

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visdetect.analysis.behavior import get_trial_dataframe

setup_style()
warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────
MIN_TRIALS = 5
PKL_DIR = os.path.join(_root, "data", "pkls", "BG_046")
TRIAL_CACHE = os.path.join(CACHE_DIR, "all_trials_behavior.csv")
BOOL_COLS = ["is_hit", "is_miss", "is_fa", "is_abort", "is_ref", "is_go", "is_catch"]

MAX_LAG = 5  # For decay analysis: how many trials after error to track

OUTCOME_ORDER = ["hit", "miss", "fa", "abort", "ref"]
OUTCOME_DISPLAY = {"hit": "Hit", "miss": "Miss", "fa": "FA",
                   "abort": "Abort", "ref": "Ref"}


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


def add_trial_history(trials):
    """Add previous-trial outcome, error flags, and error-streak column."""
    trials["is_error"] = trials["outcome"].isin(["fa", "abort"])

    def compute_error_streak(group):
        is_err = group["is_error"].values
        streak = np.zeros(len(is_err), dtype=int)
        for i in range(1, len(is_err)):
            streak[i] = streak[i - 1] + 1 if is_err[i - 1] else 0
        group = group.copy()
        group["n_preceding_errors"] = streak
        return group

    trials = trials.groupby("session_name", group_keys=False).apply(compute_error_streak)
    trials["error_streak_bin"] = trials["n_preceding_errors"].clip(upper=3)

    # Previous-trial info
    trials["prev_outcome"] = trials.groupby("session_name")["outcome"].shift(1)
    trials["prev_rt"] = trials.groupby("session_name")["rt"].shift(1)
    trials["prev_is_error"] = trials["prev_outcome"].isin(["fa", "abort"])

    # Tag "after correct" vs "after error"
    trials["post_error"] = trials["prev_is_error"].fillna(False)

    return trials


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("[01e] Post-error dynamics: speed-accuracy & criterion shift")
    print("=" * 60)

    manifest = load_staging_manifest(qc_only=True)
    print(f"  {len(manifest)} QC-passed sessions")

    trials = load_trials(manifest)
    trials = add_trial_history(trials)

    # Go trials that reached the response window (hit or miss)
    go = trials[(trials["is_go"] == True)
                & (trials["outcome"].isin(["hit", "miss"]))].copy()

    # Hit trials only (for RT analysis)
    hits = go[go["outcome"] == "hit"].copy()

    print(f"  Total trials: {len(trials)}, Go (hit+miss): {len(go)}, "
          f"Hits: {len(hits)}")
    print(f"  Post-correct hits: {(~hits['post_error']).sum()}, "
          f"Post-error hits: {hits['post_error'].sum()}")

    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.35)
    stat_results = []

    # ══════════════════════════════════════════════════════════════════
    # Panel A: Post-error RT slowing (hit RT distributions)
    # ══════════════════════════════════════════════════════════════════
    ax_a = fig.add_subplot(gs[0, 0])

    # Exclude NaN RTs and filter to reasonable range
    hits_with_rt = hits[hits["rt"].notna() & (hits["rt"] > 0) & (hits["rt"] < 5)].copy()

    rt_correct = hits_with_rt[~hits_with_rt["post_error"]]["rt"]
    rt_error = hits_with_rt[hits_with_rt["post_error"]]["rt"]

    box_data = [rt_correct.values, rt_error.values]
    box_labels = [f"After correct\n(n={len(rt_correct)})",
                  f"After error\n(n={len(rt_error)})"]
    box_colors = ["#4CAF50", "#F44336"]

    bp = ax_a.boxplot(box_data, labels=box_labels, patch_artist=True,
                      widths=0.5, showfliers=False,
                      medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual points (jittered)
    rng = np.random.default_rng(42)
    for i, (data, color) in enumerate(zip(box_data, box_colors)):
        jitter = rng.uniform(-0.12, 0.12, len(data))
        ax_a.scatter(np.full(len(data), i + 1) + jitter, data,
                     c=color, s=3, alpha=0.15, zorder=2)

    ax_a.set_ylabel("Hit RT (s)")
    ax_a.set_title("A. Post-error reaction time slowing", fontweight="bold")

    # Mann-Whitney U test
    if len(rt_correct) >= 10 and len(rt_error) >= 10:
        u, p_a = mannwhitneyu(rt_correct, rt_error, alternative="two-sided")
        median_diff = rt_error.median() - rt_correct.median()
        stat_results.append({
            "test": "mwu_hit_rt_post_correct_vs_error",
            "U": u, "p": p_a,
            "median_correct": rt_correct.median(),
            "median_error": rt_error.median(),
            "delta_median": median_diff,
            "n_correct": len(rt_correct), "n_error": len(rt_error),
        })
        sig = "***" if p_a < 0.001 else "**" if p_a < 0.01 else "*" if p_a < 0.05 else "ns"
        ax_a.text(0.5, 0.95,
                  f"Δmedian = {median_diff*1000:.0f} ms, p={p_a:.2e} {sig}",
                  transform=ax_a.transAxes, fontsize=9, ha="center", va="top")

    # ══════════════════════════════════════════════════════════════════
    # Panel B: Speed-accuracy tradeoff (per-session scatter)
    # ══════════════════════════════════════════════════════════════════
    ax_b = fig.add_subplot(gs[0, 1])

    sess_records = []
    for sn, grp in go.groupby("session_name"):
        stage = grp["stage"].iloc[0]
        sidx = grp["session_idx"].iloc[0]

        # Hit rate: after correct vs after error
        after_c = grp[~grp["post_error"]]
        after_e = grp[grp["post_error"]]
        if len(after_c) < MIN_TRIALS or len(after_e) < MIN_TRIALS:
            continue

        hr_c = after_c["is_hit"].mean()
        hr_e = after_e["is_hit"].mean()
        delta_hr = hr_e - hr_c  # positive = post-error improvement

        # Hit RT: after correct vs after error
        hits_c = after_c[(after_c["outcome"] == "hit") & after_c["rt"].notna()
                         & (after_c["rt"] > 0)]
        hits_e = after_e[(after_e["outcome"] == "hit") & after_e["rt"].notna()
                         & (after_e["rt"] > 0)]
        if len(hits_c) < 3 or len(hits_e) < 3:
            continue

        delta_rt = hits_e["rt"].median() - hits_c["rt"].median()  # positive = slowing

        sess_records.append({
            "session_name": sn, "stage": stage, "session_idx": sidx,
            "hr_correct": hr_c, "hr_error": hr_e, "delta_hr": delta_hr,
            "rt_correct": hits_c["rt"].median(),
            "rt_error": hits_e["rt"].median(),
            "delta_rt": delta_rt,
        })

    sess_df = pd.DataFrame(sess_records)

    if len(sess_df) >= 5:
        for stage in STAGE_ORDER:
            sub = sess_df[sess_df["stage"] == stage]
            if sub.empty:
                continue
            ax_b.scatter(sub["delta_rt"] * 1000, sub["delta_hr"],
                         c=STAGE_COLORS[stage], s=60, edgecolors="white",
                         linewidths=0.5, label=stage, zorder=3)

        ax_b.axhline(0, color="grey", ls="--", lw=0.8, alpha=0.5)
        ax_b.axvline(0, color="grey", ls="--", lw=0.8, alpha=0.5)
        ax_b.set_xlabel("ΔRT after error (ms, + = slower)")
        ax_b.set_ylabel("ΔHit rate after error (+ = better)")
        ax_b.set_title("B. Speed-accuracy tradeoff", fontweight="bold")
        ax_b.legend(fontsize=8)

        # Annotate quadrants
        ax_b.text(0.02, 0.98, "Slower &\nbetter", fontsize=7, color="grey",
                  transform=ax_b.transAxes, va="top", ha="left", alpha=0.6)
        ax_b.text(0.98, 0.98, "Faster &\nbetter", fontsize=7, color="grey",
                  transform=ax_b.transAxes, va="top", ha="right", alpha=0.6)
        ax_b.text(0.02, 0.02, "Slower &\nworse", fontsize=7, color="grey",
                  transform=ax_b.transAxes, va="bottom", ha="left", alpha=0.6)
        ax_b.text(0.98, 0.02, "Faster &\nworse", fontsize=7, color="grey",
                  transform=ax_b.transAxes, va="bottom", ha="right", alpha=0.6)

        # Correlation
        rho_b, p_b = spearmanr(sess_df["delta_rt"], sess_df["delta_hr"])
        stat_results.append({
            "test": "spearman_delta_rt_vs_delta_hr",
            "rho": rho_b, "p": p_b, "n": len(sess_df),
        })
        ax_b.text(0.5, 0.02, f"ρ = {rho_b:.3f}, p = {p_b:.3f}",
                  transform=ax_b.transAxes, fontsize=9, ha="center", va="bottom")

        # How many sessions show post-error slowing?
        n_slower = (sess_df["delta_rt"] > 0).sum()
        n_faster = (sess_df["delta_rt"] < 0).sum()
        stat_results.append({
            "test": "sessions_post_error_slowing",
            "n_slower": n_slower, "n_faster": n_faster,
            "frac_slower": n_slower / len(sess_df),
        })

    # ══════════════════════════════════════════════════════════════════
    # Panel C: Post-error criterion shift (FA rate on catch trials)
    # ══════════════════════════════════════════════════════════════════
    ax_c = fig.add_subplot(gs[0, 2])

    # SDT on catch trials: FA = licked (outcome "hit"), CR = withheld (outcome "miss").
    # NOTE: behavioral "fa" means early lick before change, NOT an SDT false alarm.
    catch = trials[(trials["is_catch"] == True)].copy()
    catch_evaluable = catch[catch["outcome"].isin(["hit", "miss"])].copy()
    catch_evaluable["is_catch_fa"] = (catch_evaluable["outcome"] == "hit")

    catch_after_c = catch_evaluable[~catch_evaluable["post_error"]]
    catch_after_e = catch_evaluable[catch_evaluable["post_error"]]

    fa_rate_c = catch_after_c["is_catch_fa"].mean() if len(catch_after_c) > 0 else np.nan
    fa_rate_e = catch_after_e["is_catch_fa"].mean() if len(catch_after_e) > 0 else np.nan

    # Per-session FA rates for paired test
    sess_fa = []
    for sn, grp in catch_evaluable.groupby("session_name"):
        stage = grp["stage"].iloc[0]
        ac = grp[~grp["post_error"]]
        ae = grp[grp["post_error"]]
        if len(ac) >= 3 and len(ae) >= 3:
            sess_fa.append({
                "session_name": sn, "stage": stage,
                "fa_after_correct": ac["is_catch_fa"].mean(),
                "fa_after_error": ae["is_catch_fa"].mean(),
            })
    sess_fa_df = pd.DataFrame(sess_fa)

    # Bar chart: pooled FA rates
    bar_x = [0, 1]
    bar_heights = [fa_rate_c, fa_rate_e]
    bar_ns = [len(catch_after_c), len(catch_after_e)]
    bar_errs = [np.sqrt(fa_rate_c * (1 - fa_rate_c) / max(bar_ns[0], 1)),
                np.sqrt(fa_rate_e * (1 - fa_rate_e) / max(bar_ns[1], 1))]
    bar_cols = ["#4CAF50", "#F44336"]

    ax_c.bar(bar_x, bar_heights, yerr=bar_errs, color=bar_cols,
             edgecolor="k", linewidth=0.5, capsize=6, alpha=0.7, width=0.5)

    # Overlay per-session FA rates
    if not sess_fa_df.empty:
        for _, row in sess_fa_df.iterrows():
            ax_c.plot([0, 1], [row["fa_after_correct"], row["fa_after_error"]],
                      "o-", color="grey", alpha=0.3, ms=4, lw=0.8)

    ax_c.set_xticks(bar_x)
    ax_c.set_xticklabels([f"After correct\n(n={bar_ns[0]})",
                           f"After error\n(n={bar_ns[1]})"])
    ax_c.set_ylabel("FA rate on catch trials")
    ax_c.set_ylim(0, 1.05)
    ax_c.set_title("C. Post-error criterion shift (catch FA)", fontweight="bold")

    # Chi-squared on pooled counts
    if bar_ns[0] >= 10 and bar_ns[1] >= 10:
        ct = pd.crosstab(catch_evaluable["post_error"], catch_evaluable["is_catch_fa"])
        if ct.shape == (2, 2):
            chi2, p_c, _, _ = chi2_contingency(ct)
            stat_results.append({
                "test": "chi2_catch_fa_post_correct_vs_error",
                "chi2": chi2, "p": p_c,
                "fa_after_correct": fa_rate_c,
                "fa_after_error": fa_rate_e,
                "n_after_correct": bar_ns[0], "n_after_error": bar_ns[1],
            })
            sig = "***" if p_c < 0.001 else "**" if p_c < 0.01 else "*" if p_c < 0.05 else "ns"
            ax_c.text(0.5, 0.95, f"χ²={chi2:.1f}, p={p_c:.2e} {sig}",
                      transform=ax_c.transAxes, fontsize=9, ha="center", va="top")

    # Wilcoxon paired on per-session FA rates
    if len(sess_fa_df) >= 5:
        try:
            w, p_w = wilcoxon(sess_fa_df["fa_after_correct"],
                              sess_fa_df["fa_after_error"])
            stat_results.append({
                "test": "wilcoxon_paired_catch_fa",
                "W": w, "p": p_w, "n_sessions": len(sess_fa_df),
                "median_delta_fa": (sess_fa_df["fa_after_error"]
                                    - sess_fa_df["fa_after_correct"]).median(),
            })
        except ValueError:
            pass  # all zeros

    # ══════════════════════════════════════════════════════════════════
    # Panel D: Temporal decay of post-error boost
    # ══════════════════════════════════════════════════════════════════
    ax_d = fig.add_subplot(gs[1, 0])

    # For each error trial, track hit rate on the N+1, N+2, … N+MAX_LAG
    # go trials that follow it.
    def compute_lag_hitrate(trials_df, max_lag):
        """Compute hit rate at each lag after an error."""
        lag_hits = {lag: [] for lag in range(1, max_lag + 1)}

        for _, sess_grp in trials_df.groupby("session_name"):
            sess_grp = sess_grp.reset_index(drop=True)
            is_err = sess_grp["is_error"].values
            is_go_arr = sess_grp["is_go"].values
            outcomes = sess_grp["outcome"].values

            # Find error trial indices
            err_idx = np.where(is_err)[0]
            for eidx in err_idx:
                go_count = 0
                for j in range(eidx + 1, len(sess_grp)):
                    if is_go_arr[j] and outcomes[j] in ("hit", "miss"):
                        go_count += 1
                        if go_count <= max_lag:
                            lag_hits[go_count].append(1.0 if outcomes[j] == "hit" else 0.0)
                        else:
                            break
        return lag_hits

    lag_hits = compute_lag_hitrate(trials, MAX_LAG)

    # Baseline: hit rate on go trials NOT preceded by any error (streak=0)
    go_after_correct = go[go["n_preceding_errors"] == 0]
    baseline_hr = go_after_correct["is_hit"].mean()

    lags = list(range(1, MAX_LAG + 1))
    lag_means = [np.mean(lag_hits[l]) if lag_hits[l] else np.nan for l in lags]
    lag_sems = [np.std(lag_hits[l]) / np.sqrt(len(lag_hits[l]))
                if len(lag_hits[l]) > 1 else 0 for l in lags]
    lag_ns = [len(lag_hits[l]) for l in lags]

    ax_d.errorbar(lags, lag_means, yerr=lag_sems,
                  fmt="o-", color="#F44336", lw=2, ms=8, capsize=5,
                  label="After error")
    ax_d.axhline(baseline_hr, color="#4CAF50", ls="--", lw=2, alpha=0.7,
                 label=f"Baseline (after correct): {baseline_hr:.3f}")

    for l, n in zip(lags, lag_ns):
        ax_d.text(l, -0.08, f"n={n}", ha="center", fontsize=7, color="grey")

    ax_d.set_xlabel("Go-trial lag after error")
    ax_d.set_ylabel("Hit rate")
    ax_d.set_ylim(-0.12, 1.05)
    ax_d.set_xticks(lags)
    ax_d.set_title("D. Decay of post-error accuracy boost", fontweight="bold")
    ax_d.legend(fontsize=8, loc="upper right")

    # Is lag-1 significantly above baseline? (chi-squared)
    if lag_hits[1]:
        lag1_hr = np.mean(lag_hits[1])
        stat_results.append({
            "test": "lag1_vs_baseline_hr",
            "lag1_hr": lag1_hr, "baseline_hr": baseline_hr,
            "n_lag1": len(lag_hits[1]),
            "boost_pp": (lag1_hr - baseline_hr) * 100,
        })

    # ══════════════════════════════════════════════════════════════════
    # Panel E: Trial-outcome transition matrix
    # ══════════════════════════════════════════════════════════════════
    ax_e = fig.add_subplot(gs[1, 1])

    trials["next_outcome"] = trials.groupby("session_name")["outcome"].shift(-1)
    valid_trans = trials.dropna(subset=["outcome", "next_outcome"])
    valid_trans = valid_trans[valid_trans["outcome"].isin(OUTCOME_ORDER)
                             & valid_trans["next_outcome"].isin(OUTCOME_ORDER)]

    trans_counts = pd.crosstab(valid_trans["outcome"], valid_trans["next_outcome"],
                               normalize="index")
    # Reindex to standard order
    trans_counts = trans_counts.reindex(index=OUTCOME_ORDER, columns=OUTCOME_ORDER,
                                        fill_value=0)

    im = ax_e.imshow(trans_counts.values, cmap="YlOrRd", vmin=0, vmax=0.6,
                     aspect="auto")

    ax_e.set_xticks(range(len(OUTCOME_ORDER)))
    ax_e.set_xticklabels([OUTCOME_DISPLAY[o] for o in OUTCOME_ORDER], fontsize=9)
    ax_e.set_yticks(range(len(OUTCOME_ORDER)))
    ax_e.set_yticklabels([OUTCOME_DISPLAY[o] for o in OUTCOME_ORDER], fontsize=9)
    ax_e.set_xlabel("Next trial outcome")
    ax_e.set_ylabel("Current trial outcome")
    ax_e.set_title("E. Trial-outcome transition probabilities", fontweight="bold")

    # Annotate cells
    for i in range(len(OUTCOME_ORDER)):
        for j in range(len(OUTCOME_ORDER)):
            val = trans_counts.values[i, j]
            color = "white" if val > 0.35 else "black"
            ax_e.text(j, i, f"{val:.2f}", ha="center", va="center",
                      fontsize=8, color=color)

    plt.colorbar(im, ax=ax_e, shrink=0.8, label="P(next | current)")

    # Key transition stat: P(hit | prev=FA) vs P(hit | prev=Hit)
    after_fa_outcomes = valid_trans[valid_trans["outcome"] == "fa"]["next_outcome"]
    after_hit_outcomes = valid_trans[valid_trans["outcome"] == "hit"]["next_outcome"]
    if len(after_fa_outcomes) > 10 and len(after_hit_outcomes) > 10:
        p_hit_after_fa = (after_fa_outcomes == "hit").mean()
        p_hit_after_hit = (after_hit_outcomes == "hit").mean()
        stat_results.append({
            "test": "p_hit_after_fa_vs_hit",
            "p_hit_after_fa": p_hit_after_fa,
            "p_hit_after_hit": p_hit_after_hit,
            "n_after_fa": len(after_fa_outcomes),
            "n_after_hit": len(after_hit_outcomes),
        })

    # ══════════════════════════════════════════════════════════════════
    # Panel F: Post-error slowing trajectory across learning
    # ══════════════════════════════════════════════════════════════════
    ax_f = fig.add_subplot(gs[1, 2])

    if not sess_df.empty:
        add_stage_background(ax_f, manifest)

        # Plot ΔRT (post-error slowing)
        color_rt = "#1976D2"
        ax_f.scatter(sess_df["session_idx"], sess_df["delta_rt"] * 1000,
                     c=color_rt, s=50, edgecolors="white", linewidths=0.5,
                     zorder=3, label="ΔRT (ms)")
        ax_f.plot(sess_df["session_idx"], sess_df["delta_rt"] * 1000,
                  c=color_rt, alpha=0.3, lw=1, zorder=2)
        ax_f.axhline(0, color="grey", ls="--", lw=0.8, alpha=0.5)

        ax_f.set_xlabel("Session index")
        ax_f.set_ylabel("Post-error ΔRT (ms, + = slower)", color=color_rt)
        ax_f.tick_params(axis="y", labelcolor=color_rt)
        ax_f.set_title("F. Post-error slowing across learning", fontweight="bold")

        # Secondary axis: ΔHR
        ax_f2 = ax_f.twinx()
        color_hr = "#E65100"
        ax_f2.scatter(sess_df["session_idx"], sess_df["delta_hr"],
                      c=color_hr, s=50, marker="D", edgecolors="white",
                      linewidths=0.5, zorder=3, label="ΔHR")
        ax_f2.plot(sess_df["session_idx"], sess_df["delta_hr"],
                   c=color_hr, alpha=0.3, lw=1, zorder=2)
        ax_f2.set_ylabel("Post-error ΔHit rate (+ = better)", color=color_hr)
        ax_f2.tick_params(axis="y", labelcolor=color_hr)

        # Combined legend
        lines_a, labels_a = ax_f.get_legend_handles_labels()
        lines_b, labels_b = ax_f2.get_legend_handles_labels()
        ax_f.legend(lines_a + lines_b, labels_a + labels_b,
                    fontsize=7, loc="upper left")

        # Spearman: session_idx vs delta_rt
        rho_f, p_f = spearmanr(sess_df["session_idx"], sess_df["delta_rt"])
        stat_results.append({
            "test": "spearman_session_vs_delta_rt",
            "rho": rho_f, "p": p_f, "n": len(sess_df),
        })
        ax_f.text(0.95, 0.05, f"ΔRT vs session: ρ={rho_f:.3f}, p={p_f:.3f}",
                  transform=ax_f.transAxes, fontsize=8, va="bottom", ha="right")

    # ══════════════════════════════════════════════════════════════════
    # Finalize
    # ══════════════════════════════════════════════════════════════════
    fig.suptitle("Post-Error Behavioral Dynamics (BG_046)",
                 fontsize=14, fontweight="bold", y=0.98)

    paths = save_figure(fig, "fig05_post_error_dynamics", "01_behavior")
    print(f"\n  Saved figure: {paths}")

    # Save statistics
    if stat_results:
        stats_df = pd.DataFrame(stat_results)
        stats_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "figures", "01_behavior", "post_error_dynamics_stats.csv",
        )
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        stats_df.to_csv(stats_path, index=False)
        print(f"  Saved statistics: {stats_path}")
        print("\n  Statistics summary:")
        for _, r in stats_df.iterrows():
            cols = {k: v for k, v in r.items() if pd.notna(v)}
            print(f"    {cols.get('test', '?')}: {cols}")

    print("\n[01e] Done.")


if __name__ == "__main__":
    main()
