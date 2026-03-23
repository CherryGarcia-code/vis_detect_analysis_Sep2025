"""Fig 10: Change-size tuning — tuning curves per unit.

For each unit, computes firing rates in the response window for each
change size, tests for tuning (Kruskal-Wallis across change sizes),
and computes monotonicity (Spearman correlation with change magnitude).

Produces:
  - Fig 8A: Population-average tuning curve by stage
  - Fig 8B: Distribution of tuning strength (Spearman rho)
  - Fig 8C: Fraction significantly tuned per stage
  - Fig 8D: Tuning slope (rho) vs session index

Saves: figures/02_single_unit/change_size_tuning_stats.csv
       cache/tuning_all_sessions.csv
"""

import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    STAGE_ORDER, STAGE_COLORS,
    CHANGE_SIZES, CHANGE_SIZE_LABELS, CHANGE_SIZE_POSITIONS,
    CACHE_DIR,
)
from loader import load_staging_manifest, load_session
from utils import get_good_cluster_ids, fdr_correct
from plotting import setup_style, save_figure, add_stage_background

from visdetect.analysis.align import align_spikes_to_events, get_event_times_by_trial

setup_style()

RESP_WIN = (0.0, 0.25)     # response window
BASE_WIN = (-0.4, -0.05)   # baseline
MIN_TRIALS_PER_SIZE = 3


def compute_tuning_for_session(sess, sname, stage, sidx):
    """Compute change-size tuning for each unit in a session."""
    good_ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
    if len(good_ids) < 3:
        return []

    trials = sess.trials
    event_times = get_event_times_by_trial(sess, "Change_ON")

    # Group trials by change size (Hit + Miss only)
    size_trial_map = {cs: [] for cs in CHANGE_SIZES}
    for i, t in enumerate(trials):
        outcome = getattr(t, "trialoutcome", None)
        cs = getattr(t, "change_size", None)
        if outcome not in ("Hit", "Miss") or cs is None:
            continue
        if i >= len(event_times) or not np.isfinite(event_times[i]):
            continue
        # Match to canonical change size
        for canonical in CHANGE_SIZES:
            if abs(cs - canonical) < 0.01:
                size_trial_map[canonical].append(i)
                break

    # Need at least some change sizes with enough trials
    usable_sizes = [cs for cs in CHANGE_SIZES if len(size_trial_map[cs]) >= MIN_TRIALS_PER_SIZE]
    if len(usable_sizes) < 2:
        return []

    # Get cluster lookup
    cluster_map = {int(c.cluster_id): c for c in sess.clusters}

    results = []
    for cid in good_ids:
        c = cluster_map.get(int(cid))
        if c is None:
            continue

        # Compute mean FR in response window per change size
        size_frs = {}
        size_frs_all = {}  # keep individual trial FRs for Kruskal

        for cs in usable_sizes:
            trial_idxs = size_trial_map[cs]
            trial_event_times = [float(event_times[i]) for i in trial_idxs]

            mat, bc = align_spikes_to_events(
                c.spike_times, trial_event_times,
                window=(-0.5, 0.5), bin_size=0.025,
            )
            resp_mask = (bc >= RESP_WIN[0]) & (bc < RESP_WIN[1])
            base_mask = (bc >= BASE_WIN[0]) & (bc < BASE_WIN[1])

            # Per-trial mean FR in response window
            resp_frs = np.nanmean(mat[:, resp_mask], axis=1)
            base_frs = np.nanmean(mat[:, base_mask], axis=1)

            size_frs[cs] = float(np.nanmean(resp_frs) - np.nanmean(base_frs))
            size_frs_all[cs] = resp_frs

        # Kruskal-Wallis across change sizes
        kw_groups = [size_frs_all[cs] for cs in usable_sizes if len(size_frs_all[cs]) >= 2]
        if len(kw_groups) >= 2:
            try:
                h, p_kw = kruskal(*kw_groups)
            except ValueError:
                h, p_kw = 0, 1.0
        else:
            h, p_kw = 0, 1.0

        # Spearman correlation with change size magnitude
        sizes_for_corr = []
        frs_for_corr = []
        for cs in usable_sizes:
            for fr_val in size_frs_all[cs]:
                sizes_for_corr.append(cs)
                frs_for_corr.append(fr_val)

        if len(sizes_for_corr) >= 5:
            rho, p_sp = spearmanr(sizes_for_corr, frs_for_corr)
        else:
            rho, p_sp = np.nan, 1.0

        # Mean tuning curve values
        tuning_curve = [size_frs.get(cs, np.nan) for cs in CHANGE_SIZES]

        results.append({
            "session_name": sname,
            "cluster_id": cid,
            "stage": stage,
            "session_idx": sidx,
            "kw_H": h,
            "kw_p": p_kw,
            "spearman_rho": rho,
            "spearman_p": p_sp,
            **{f"fr_{cs}": size_frs.get(cs, np.nan) for cs in CHANGE_SIZES},
        })

    return results


def main():
    print("[02c] Change-size tuning analysis...")
    manifest = load_staging_manifest(qc_only=True)

    cache_path = os.path.join(CACHE_DIR, "tuning_all_sessions.csv")
    if os.path.exists(cache_path):
        print(f"  Loading cached results from {cache_path}")
        tune_df = pd.read_csv(cache_path)
    else:
        all_results = []
        for _, row in manifest.iterrows():
            sname = int(row["session_name"])
            stage = row["stage"]
            sidx = row["session_idx"]

            print(f"  Session {sname} ({stage})...", end=" ")
            try:
                sess = load_session(sname)
            except FileNotFoundError:
                print("not found")
                continue

            res = compute_tuning_for_session(sess, sname, stage, sidx)
            all_results.extend(res)
            print(f"{len(res)} units" if res else "insufficient data")

            del sess
            gc.collect()

        tune_df = pd.DataFrame(all_results)
        if len(tune_df) > 0:
            tune_df.to_csv(cache_path, index=False)
            print(f"\n  Cached {len(tune_df)} units")

    if len(tune_df) == 0:
        print("  No tuning data. Exiting.")
        return

    # FDR correct Kruskal-Wallis p-values
    tune_df["kw_significant"] = fdr_correct(tune_df["kw_p"].values, alpha=0.05)
    n_tuned = tune_df["kw_significant"].sum()
    print(f"\n  {n_tuned}/{len(tune_df)} units significantly tuned (FDR q<0.05)")

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Population-average tuning curves by stage
    ax_a = fig.add_subplot(gs[0, 0])
    for stage in STAGE_ORDER:
        sub = tune_df[tune_df["stage"] == stage]
        means, sems = [], []
        for cs in CHANGE_SIZES:
            col = f"fr_{cs}"
            if col in sub.columns:
                vals = sub[col].dropna().values
                if len(vals) >= 2:
                    means.append(np.mean(vals))
                    sems.append(np.std(vals) / np.sqrt(len(vals)))
                else:
                    means.append(np.nan)
                    sems.append(0)
            else:
                means.append(np.nan)
                sems.append(0)

        ax_a.errorbar(CHANGE_SIZE_POSITIONS, means, yerr=sems,
                      fmt="o-", color=STAGE_COLORS[stage], label=stage,
                      linewidth=2, markersize=6, capsize=3)

    ax_a.set_xticks(CHANGE_SIZE_POSITIONS)
    ax_a.set_xticklabels(CHANGE_SIZE_LABELS)
    ax_a.set_xlabel("Change size (TF ratio)")
    ax_a.set_ylabel("ΔFR (resp - base, Hz)")
    ax_a.set_title("A. Population tuning curves by stage")
    ax_a.legend(fontsize=8)

    # Panel B: Spearman rho distribution
    ax_b = fig.add_subplot(gs[0, 1])
    rho_vals = tune_df["spearman_rho"].dropna().values
    if len(rho_vals) > 0:
        ax_b.hist(rho_vals, bins=40, color="#7986CB", edgecolor="white",
                  linewidth=0.5, alpha=0.8)
        ax_b.axvline(0, color="k", linestyle="--", linewidth=0.8)
        ax_b.axvline(np.median(rho_vals), color="#E53935", linewidth=1.5,
                     label=f"Median={np.median(rho_vals):.3f}")
    ax_b.set_xlabel("Spearman ρ (FR vs change size)")
    ax_b.set_ylabel("Number of units")
    ax_b.set_title(f"B. Tuning strength distribution (n={len(rho_vals)})")
    ax_b.legend(fontsize=8)

    # Panel C: Fraction tuned per stage
    ax_c = fig.add_subplot(gs[1, 0])
    stage_frac = {}
    for stage in STAGE_ORDER:
        sub = tune_df[tune_df["stage"] == stage]
        if len(sub) > 0:
            stage_frac[stage] = {"frac": sub["kw_significant"].mean(),
                                  "n_sig": sub["kw_significant"].sum(),
                                  "n_total": len(sub)}
        else:
            stage_frac[stage] = {"frac": 0, "n_sig": 0, "n_total": 0}

    bar_x = range(len(STAGE_ORDER))
    bar_vals = [stage_frac[s]["frac"] for s in STAGE_ORDER]
    bar_colors = [STAGE_COLORS[s] for s in STAGE_ORDER]
    ax_c.bar(bar_x, bar_vals, color=bar_colors, edgecolor="white")
    for i, stage in enumerate(STAGE_ORDER):
        info = stage_frac[stage]
        ax_c.text(i, info["frac"] + 0.01, f"{info['n_sig']}/{info['n_total']}",
                  ha="center", fontsize=8)

    ax_c.set_xticks(bar_x)
    ax_c.set_xticklabels(STAGE_ORDER)
    ax_c.set_ylabel("Fraction tuned (FDR q<0.05)")
    ax_c.set_ylim(0, max(bar_vals) * 1.3 if bar_vals and max(bar_vals) > 0 else 0.5)
    ax_c.set_title("C. Fraction tuned by stage")

    # Panel D: Mean Spearman rho across sessions
    ax_d = fig.add_subplot(gs[1, 1])
    add_stage_background(ax_d, manifest)

    sess_stats = tune_df.groupby(["session_name", "session_idx", "stage"]).agg(
        mean_rho=("spearman_rho", "mean"),
        frac_tuned=("kw_significant", "mean"),
    ).reset_index().sort_values("session_idx")

    for stage in STAGE_ORDER:
        sub = sess_stats[sess_stats["stage"] == stage]
        if len(sub) > 0:
            ax_d.scatter(sub["session_idx"], sub["mean_rho"],
                         c=STAGE_COLORS[stage], s=60, edgecolors="white",
                         linewidths=0.5, zorder=3, label=stage)
    if len(sess_stats) > 0:
        ax_d.plot(sess_stats["session_idx"], sess_stats["mean_rho"],
                  color="gray", alpha=0.3, linewidth=1, zorder=2)
    ax_d.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_d.set_xlabel("Session index")
    ax_d.set_ylabel("Mean Spearman ρ")
    ax_d.set_title("D. Tuning strength across learning")
    ax_d.legend(fontsize=8)

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # Population bias toward positive tuning
    from scipy.stats import wilcoxon
    if len(rho_vals) >= 10:
        try:
            w, p = wilcoxon(rho_vals)
            stats.append({"test": "rho_vs_zero_wilcoxon", "W": w, "p": p,
                          "median_rho": float(np.median(rho_vals))})
        except ValueError:
            pass

    # Tuning trend across sessions
    if len(sess_stats) >= 3:
        rho_s, p_s = spearmanr(sess_stats["session_idx"], sess_stats["mean_rho"])
        stats.append({"test": "mean_rho_vs_session_spearman", "rho": rho_s, "p": p_s})

    # Fraction tuned by stage (chi-square)
    from scipy.stats import chi2_contingency
    ct_data = []
    for stage in STAGE_ORDER:
        sub = tune_df[tune_df["stage"] == stage]
        if len(sub) > 0:
            ct_data.append([sub["kw_significant"].sum(), len(sub) - sub["kw_significant"].sum()])
    if len(ct_data) >= 2:
        ct_arr = np.array(ct_data)
        if ct_arr.sum() > 0 and ct_arr.min(axis=0).sum() > 0:
            try:
                chi2, p, dof, _ = chi2_contingency(ct_arr)
                stats.append({"test": "tuned_fraction_chi2_by_stage", "chi2": chi2, "p": p})
            except ValueError:
                pass

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig10_change_size_tuning", "02_single_unit")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "02_single_unit", "change_size_tuning_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: p={row.get('p', 'N/A')}")


if __name__ == "__main__":
    main()
