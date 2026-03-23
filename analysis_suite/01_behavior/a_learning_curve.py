"""Fig01: Learning curve and behavioral performance overview.

Produces the foundational behavioral summary:
  - Fig 01A: d' trajectory across sessions with stage bands
  - Fig 01B: Hit rate and FA rate across sessions
  - Fig 01C: Psychometric curves (hit rate vs change size) per stage
  - Fig 01D: Reaction time distributions by outcome

Saves statistics to figures/01_behavior/learning_curve_stats.csv.
"""

import os
import sys

# Add analysis_suite to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    STAGE_ORDER, STAGE_COLORS, OUTCOME_COLORS,
    CHANGE_SIZES, CHANGE_SIZE_LABELS, CHANGE_SIZE_POSITIONS,
)
from loader import load_staging_manifest, load_session, session_iterator
from plotting import setup_style, save_figure, add_stage_background, stage_legend

setup_style()


def main():
    print("[01a] Loading staging manifest...")
    manifest = load_staging_manifest(qc_only=True)
    print(f"  {len(manifest)} QC-passed sessions: "
          f"Naive={sum(manifest['stage']=='Naive')}, "
          f"Learning={sum(manifest['stage']=='Learning')}, "
          f"Expert={sum(manifest['stage']=='Expert')}")

    # ── Collect per-session behavioral metrics ────────────────────────
    records = []
    psychometric_data = {stage: {cs: [] for cs in CHANGE_SIZES} for stage in STAGE_ORDER}

    for _, row in manifest.iterrows():
        sname = int(row["session_name"])
        stage = row["stage"]
        sidx = row["session_idx"]

        try:
            sess = load_session(sname)
        except FileNotFoundError:
            print(f"  Skipping {sname}: pkl not found")
            continue

        trials = sess.trials
        n_trials = len(trials)

        # Compute metrics from trials
        outcomes = [getattr(t, "trialoutcome", None) for t in trials]
        change_sizes = [getattr(t, "change_size", None) for t in trials]

        # Classify trials (outcomes are capitalized: Hit, Miss, FA, abort, Ref)
        n_hit = sum(1 for o in outcomes if o == "Hit")
        n_miss = sum(1 for o in outcomes if o == "Miss")
        n_fa = sum(1 for o in outcomes if o == "FA")

        # Go trials: change_size > 1 (not catch)
        go_trials = [(o, cs) for o, cs in zip(outcomes, change_sizes)
                     if cs is not None and cs > 1.0 and o in ("Hit", "Miss")]
        n_go = len(go_trials)
        hits_on_go = sum(1 for o, cs in go_trials if o == "Hit")

        # Catch trials: change_size == 1
        catch_trials = [(o, cs) for o, cs in zip(outcomes, change_sizes)
                        if cs is not None and cs <= 1.01 and o in ("Hit", "Miss")]
        n_catch = len(catch_trials)
        # FA on catch = licking on catch (outcome is "Hit" on catch = false alarm in SDT)
        fa_on_catch = sum(1 for o, cs in catch_trials if o == "Hit")

        # Hit rate, FA rate, d'
        hit_rate = hits_on_go / max(n_go, 1)
        fa_rate = fa_on_catch / max(n_catch, 1)

        # d': clip rates for z-transform
        from scipy.stats import norm
        hr_clipped = np.clip(hit_rate, 0.01, 0.99)
        far_clipped = np.clip(fa_rate, 0.01, 0.99)
        dprime = float(norm.ppf(hr_clipped) - norm.ppf(far_clipped))

        # Reaction times
        hit_rts = []
        fa_rts = []
        for t in trials:
            rt_dict = getattr(t, "reactiontimes", {}) or {}
            outcome = getattr(t, "trialoutcome", None)
            if outcome == "Hit" and "RT" in rt_dict:
                val = rt_dict["RT"]
                if np.isfinite(val):
                    hit_rts.append(val)
            elif outcome == "FA":
                val = rt_dict.get("FA", rt_dict.get("RT", np.nan))
                if np.isfinite(val):
                    fa_rts.append(val)

        records.append({
            "session_name": sname,
            "session_idx": sidx,
            "stage": stage,
            "n_trials": n_trials,
            "n_go": n_go,
            "n_catch": n_catch,
            "hit_rate": hit_rate,
            "fa_rate": fa_rate,
            "d_prime": dprime,
            "n_hit": n_hit,
            "n_miss": n_miss,
            "n_fa": n_fa,
            "mean_rt_hit": np.nanmean(hit_rts) if hit_rts else np.nan,
            "median_rt_hit": np.nanmedian(hit_rts) if hit_rts else np.nan,
            "mean_rt_fa": np.nanmean(fa_rts) if fa_rts else np.nan,
        })

        # Psychometric data: per change-size hit rate
        for cs in CHANGE_SIZES:
            cs_trials = [(o, c) for o, c in zip(outcomes, change_sizes)
                         if c is not None and abs(c - cs) < 0.01 and o in ("Hit", "Miss")]
            if len(cs_trials) >= 3:
                cs_hr = sum(1 for o, _ in cs_trials if o == "Hit") / len(cs_trials)
                psychometric_data[stage][cs].append(cs_hr)

        del sess

    df = pd.DataFrame(records)
    print(f"  Collected metrics for {len(df)} sessions")

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: d' trajectory
    ax_a = fig.add_subplot(gs[0, 0])
    add_stage_background(ax_a, manifest)
    for stage in STAGE_ORDER:
        m = df[df["stage"] == stage]
        ax_a.scatter(m["session_idx"], m["d_prime"],
                     c=STAGE_COLORS[stage], s=60, zorder=3, edgecolors="white",
                     linewidths=0.5, label=stage)
    ax_a.plot(df["session_idx"], df["d_prime"], c="gray", alpha=0.4, linewidth=1, zorder=2)
    ax_a.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_a.axhline(1.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_a.set_xlabel("Session index (chronological)")
    ax_a.set_ylabel("d'")
    ax_a.set_title("A. Sensitivity (d') across learning")
    ax_a.legend(loc="upper left", fontsize=8)

    # Panel B: Hit rate and FA rate
    ax_b = fig.add_subplot(gs[0, 1])
    add_stage_background(ax_b, manifest)
    ax_b.plot(df["session_idx"], df["hit_rate"], "o-",
              color=OUTCOME_COLORS["Hit"], markersize=5, linewidth=1.5, label="Hit rate")
    ax_b.plot(df["session_idx"], df["fa_rate"], "s-",
              color=OUTCOME_COLORS["FA"], markersize=5, linewidth=1.5, label="FA rate")
    ax_b.set_xlabel("Session index")
    ax_b.set_ylabel("Rate")
    ax_b.set_ylim(-0.05, 1.05)
    ax_b.set_title("B. Hit and FA rates across learning")
    ax_b.legend(loc="center right", fontsize=8)

    # Panel C: Psychometric curves per stage
    ax_c = fig.add_subplot(gs[1, 0])
    for stage in STAGE_ORDER:
        means = []
        sems = []
        for cs in CHANGE_SIZES:
            vals = psychometric_data[stage][cs]
            if vals:
                means.append(np.mean(vals))
                sems.append(np.std(vals) / np.sqrt(len(vals)))
            else:
                means.append(np.nan)
                sems.append(0)
        ax_c.errorbar(CHANGE_SIZE_POSITIONS, means, yerr=sems,
                      fmt="o-", color=STAGE_COLORS[stage], label=stage,
                      linewidth=2, markersize=6, capsize=3)
    ax_c.set_xticks(CHANGE_SIZE_POSITIONS)
    ax_c.set_xticklabels(CHANGE_SIZE_LABELS)
    ax_c.set_xlabel("Change size (TF ratio)")
    ax_c.set_ylabel("Hit rate")
    ax_c.set_ylim(-0.05, 1.05)
    ax_c.set_title("C. Psychometric curves by stage")
    ax_c.legend(loc="lower right", fontsize=8)

    # Panel D: Reaction time by stage (violin)
    ax_d = fig.add_subplot(gs[1, 1])
    rt_data_by_stage = {s: df[df["stage"] == s]["median_rt_hit"].dropna().values
                        for s in STAGE_ORDER}
    positions = []
    data_list = []
    colors_list = []
    for i, stage in enumerate(STAGE_ORDER):
        vals = rt_data_by_stage[stage]
        if len(vals) > 0:
            positions.append(i)
            data_list.append(vals)
            colors_list.append(STAGE_COLORS[stage])

    if data_list and any(len(d) >= 2 for d in data_list):
        # Filter to groups with enough data for violin
        valid = [(p, d, c) for p, d, c in zip(positions, data_list, colors_list) if len(d) >= 2]
        if valid:
            vpos, vdata, vcols = zip(*valid)
            parts = ax_d.violinplot(list(vdata), positions=list(vpos), showmeans=True, showmedians=True)
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(vcols[i])
                pc.set_alpha(0.6)
        # Overlay individual points
        for i, (pos, vals) in enumerate(zip(positions, data_list)):
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
            ax_d.scatter(pos + jitter, vals, c=colors_list[i], s=30,
                         edgecolors="white", linewidths=0.5, zorder=3)

    ax_d.set_xticks(range(len(STAGE_ORDER)))
    ax_d.set_xticklabels(STAGE_ORDER)
    ax_d.set_ylabel("Median RT (s)")
    ax_d.set_title("D. Hit reaction times by stage")

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # d' trend across sessions
    rho, p = spearmanr(df["session_idx"], df["d_prime"])
    stats.append({"test": "d_prime_vs_session_spearman", "rho": rho, "p": p,
                  "n": len(df)})

    # d' by stage (Kruskal-Wallis)
    stage_groups = [df[df["stage"] == s]["d_prime"].values for s in STAGE_ORDER]
    stage_groups = [g for g in stage_groups if len(g) >= 2 and np.std(g) > 0]
    if len(stage_groups) >= 2:
        try:
            h, p = kruskal(*stage_groups)
            stats.append({"test": "d_prime_kruskal_by_stage", "H": h, "p": p})
        except ValueError:
            pass

    # FA rate trend
    fa_finite = df[["session_idx", "fa_rate"]].dropna()
    if len(fa_finite) >= 3 and fa_finite["fa_rate"].std() > 0:
        rho, p = spearmanr(fa_finite["session_idx"], fa_finite["fa_rate"])
        stats.append({"test": "fa_rate_vs_session_spearman", "rho": rho, "p": p,
                      "n": len(fa_finite)})

    # RT by stage
    rt_groups = [df[df["stage"] == s]["median_rt_hit"].dropna().values for s in STAGE_ORDER]
    rt_groups = [g for g in rt_groups if len(g) >= 2 and np.std(g) > 0]
    if len(rt_groups) >= 2:
        try:
            h, p = kruskal(*rt_groups)
            stats.append({"test": "rt_kruskal_by_stage", "H": h, "p": p})
        except ValueError:
            pass

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig01_learning_curve", "01_behavior")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "01_behavior", "learning_curve_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)
    print(f"  Saved figure and stats to figures/01_behavior/")
    print(f"\n  Key results:")
    print(f"    d' range: {df['d_prime'].min():.2f} to {df['d_prime'].max():.2f}")
    print(f"    d' vs session: rho={stats[0]['rho']:.3f}, p={stats[0]['p']:.1e}")
    for row in stats:
        print(f"    {row['test']}: {row}")


if __name__ == "__main__":
    main()
