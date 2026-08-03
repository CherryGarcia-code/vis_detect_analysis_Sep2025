"""Cross-subject replication of the BG_046 early-lick-across-learning result.

Gate 3/4 of the harden-result battery: a single-mouse trajectory is not a claim.
Re-run the SAME methodology (compute_session_performance -> fraction_fa; block-
permutation trend p; Silverman multimodality; KDE-antimode split) on all three
striatal mice and ask whether the decline reproduces:

  BG_046 = DMS  (the original)
  BG_039 = DMS  (independent DMS replication)
  BG_031 = VMS  (memory-flagged IMPULSIVE NON-LEARNER = negative control:
                 if the decline is learning-specific it should be weak/absent here)

Sessions + stage labels come from evidence_learning_io.subject_sessions (reads each
subject's own staging manifest). Metric = anticipatory 'fa' label rate, NOT SDT FA.

Run: py scripts/analysis/behavior/early_lick_replication.py [--force]
Out: FIGURES/behavior/replication/early_lick_replication_3mice.png
     data/cache/behavior/early_lick_repl_<subject>{,_rts}.csv
     data/cache/behavior/early_lick_replication_summary.csv
"""
import os
import sys
import gc
import argparse

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))   # reuse the main script's helpers

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from visdetect.analysis.behavior import compute_session_performance, get_trial_dataframe
from visdetect.analysis.evidence_learning_io import subject_sessions, SUBJECTS
from visdetect.suite.plotting import save_figure

import early_lick_learning_trajectory as ell   # data_driven_split, block_perm_p, silverman_p, kde_logboot, palettes

SUBJECTS_ORDER = ["BG_046", "BG_039", "BG_031"]
STAGES = ("Naive", "Learning", "Expert")


def compute_subject(subject, force=False):
    cache = os.path.join(ell.CACHE_DIR, f"early_lick_repl_{subject}.csv")
    rtcache = os.path.join(ell.CACHE_DIR, f"early_lick_repl_{subject}_rts.csv")
    if os.path.exists(cache) and os.path.exists(rtcache) and not force:
        return (pd.read_csv(cache, dtype={"session_name": str}),
                pd.read_csv(rtcache, dtype={"session_name": str}))

    os.makedirs(ell.CACHE_DIR, exist_ok=True)
    rows, rt_rows, sidx = [], [], 0
    for skey, sname, stage, sess in subject_sessions(subject, stages=STAGES):
        try:
            perf = compute_session_performance(sess)
            tdf = get_trial_dataframe(sess)
            fa = tdf.loc[tdf["is_fa"], "rt"].values
            for rt in fa:
                if np.isfinite(rt):
                    rt_rows.append({"subject": subject, "session_idx": sidx,
                                    "stage": stage, "rt": float(rt)})
            rows.append({"subject": subject, "session_name": str(sname),
                         "session_idx": sidx, "stage": stage,
                         "early_lick_rate": perf["fraction_fa"], "n_fa": perf["n_fa"],
                         "n_trials": perf["n_trials"], "d_prime": perf["d_prime"]})
            sidx += 1
            print(f"  {subject} [{sidx}] {sname} ({stage})")
        finally:
            del sess
            gc.collect()
    df = pd.DataFrame(rows)
    rt_df = pd.DataFrame(rt_rows)
    df.to_csv(cache, index=False)
    rt_df.to_csv(rtcache, index=False)
    return df, rt_df


def summarise(subject, df, rt_df):
    df = df.sort_values("session_idx").reset_index(drop=True)
    x = df["session_idx"].values
    rate = df["early_lick_rate"].values
    rho, p_iid = spearmanr(x, rate)
    _, p_bp = ell.block_perm_p(x, rate)
    naive = df.loc[df["stage"] == "Naive", "early_lick_rate"].dropna().values
    expert = df.loc[df["stage"] == "Expert", "early_lick_rate"].dropna().values
    mwu_p = mannwhitneyu(naive, expert).pvalue if len(naive) and len(expert) else np.nan
    er = rt_df.loc[rt_df["stage"] == "Expert", "rt"].values
    er = er[np.isfinite(er) & (er > 0)]
    sil = ell.silverman_p(er) if len(er) >= 40 else np.nan   # linear-space multimodality
    thr, meth = ell.data_driven_split(er) if len(er) >= 40 else (np.nan, "insufficient")
    return {
        "subject": subject, "region": SUBJECTS.get(subject, "?"), "n_sessions": len(df),
        "n_naive": len(naive), "n_expert": len(expert),
        "rho_rate_vs_session": rho, "p_iid": p_iid, "p_blockperm": p_bp,
        "naive_mean_rate": naive.mean() if len(naive) else np.nan,
        "expert_mean_rate": expert.mean() if len(expert) else np.nan,
        "naive_vs_expert_mwu_p": mwu_p,
        "expert_rt_silverman_p": sil, "expert_rt_antimode_s": thr, "antimode_method": meth,
        "n_expert_fa_licks": len(er),
    }


def make_figure(data):
    grid = np.linspace(ell.RT_LO, 10.0, 400)   # linear seconds (area = mass)
    fig = plt.figure(figsize=(16.5, 8.5))
    gs = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.28)

    for j, subject in enumerate(SUBJECTS_ORDER):
        df, rt_df, summ = data[subject]
        df = df.sort_values("session_idx").reset_index(drop=True)
        x = df["session_idx"].values
        rate = df["early_lick_rate"].values
        lo, hi = ell.wilson_ci(df["n_fa"].values, df["n_trials"].values)
        cols = df["stage"].map(ell.STAGE_COLORS_FULL).fillna("#999999").values

        # Row 1: trajectory
        ax = fig.add_subplot(gs[0, j])
        ell.shade_stages(ax, df)
        ax.errorbar(x, rate, yerr=[rate - lo, hi - rate], fmt="none", ecolor="0.6",
                    elinewidth=0.7, alpha=0.6, zorder=1)
        ax.plot(x, rate, "-", color="0.45", lw=1.1, zorder=2)
        ax.scatter(x, rate, c=cols, s=34, edgecolors="white", linewidths=0.4, zorder=3)
        ell.annotate_stat(ax, f"ρ = {summ['rho_rate_vs_session']:.2f}\n"
                              f"p = {summ['p_iid']:.2g} (iid)\n"
                              f"p = {summ['p_blockperm']:.2g} (block-perm)")
        note = ""
        if np.isfinite(summ["naive_mean_rate"]) and np.isfinite(summ["expert_mean_rate"]):
            note = (f"Naive {summ['naive_mean_rate']:.2f} → Expert {summ['expert_mean_rate']:.2f}"
                    f"  (MWU p={summ['naive_vs_expert_mwu_p']:.2g})")
        ax.text(0.5, -0.30, note, transform=ax.transAxes, ha="center", fontsize=8, color="0.25")
        ax.set_title(f"{subject} ({summ['region']}) — early-lick rate",
                     fontweight="bold", loc="left", fontsize=11)
        ax.set_xlabel("Session (chronological)")
        if j == 0:
            ax.set_ylabel("Early-lick rate\nP(anticipatory lick)")
        ax.set_ylim(bottom=0)
        if j == 0:
            handles = [Patch(facecolor=ell.STAGE_COLORS_FULL[s], label=s, alpha=0.9)
                       for s in ell.STAGE_ORDER_FULL]
            ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)

        # Row 2: RT distribution by stage (linear seconds, area = mass)
        ax2 = fig.add_subplot(gs[1, j])
        for s in ell.STAGE_ORDER_FULL:
            srt = rt_df.loc[rt_df["stage"] == s, "rt"].values
            base, blo, bhi = ell.kde_lin_boot(srt, grid)
            if base is None:
                continue
            ax2.fill_between(grid, blo, bhi, color=ell.STAGE_COLORS_FULL[s], alpha=0.18, lw=0)
            ax2.plot(grid, base, color=ell.STAGE_COLORS_FULL[s], lw=1.6,
                     label=f"{s} (n={int(np.isfinite(srt).sum())})")
        ax2.axvline(ell.FA_RT_SPLIT, color="k", ls="--", lw=0.9, alpha=0.6)
        if np.isfinite(summ["expert_rt_antimode_s"]):
            ax2.axvline(summ["expert_rt_antimode_s"], color=ell.EARLYLICK_COLOR, lw=1.5)
            ell.annotate_stat(ax2, f"Expert antimode {summ['expert_rt_antimode_s']:.2f} s\n"
                                   f"Silverman p={summ['expert_rt_silverman_p']:.3f}", loc="upper left")
        ax2.set_xlim(0, 10.0)
        ax2.set_xlabel("Early-lick RT from baseline onset (s)")
        if j == 0:
            ax2.set_ylabel("Density of log RT (area = mass)")
        ax2.set_title(f"{subject} — RT distribution by stage", fontweight="bold",
                      loc="left", fontsize=10.5)
        ax2.legend(loc="upper right", frameon=False, fontsize=7.5)

    fig.suptitle("Cross-subject replication — anticipatory early-lick behaviour across learning "
                 "(DMS: BG_046, BG_039 | VMS: BG_031)", fontsize=13, fontweight="bold", y=0.995)
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cross-subject early-lick replication.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data, summaries = {}, []
    for subject in SUBJECTS_ORDER:
        print(f"=== {subject} ===")
        df, rt_df = compute_subject(subject, force=args.force)
        summ = summarise(subject, df, rt_df)
        data[subject] = (df, rt_df, summ)
        summaries.append(summ)

    summ_df = pd.DataFrame(summaries)
    summ_path = os.path.join(ell.CACHE_DIR, "early_lick_replication_summary.csv")
    summ_df.to_csv(summ_path, index=False)

    fig = make_figure(data)
    paths = save_figure(fig, "early_lick_replication_3mice", "behavior/replication")
    print("\nSaved figure:", paths[0])
    print("Saved summary:", summ_path)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== REPLICATION SUMMARY ===")
    print(summ_df.to_string(index=False))
