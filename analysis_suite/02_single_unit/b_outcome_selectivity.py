"""Fig 09: Outcome selectivity — Hit vs Miss auROC per unit.

For each unit, computes auROC comparing firing rates on Hit vs Miss trials
in a response window (0–250ms post-Change_ON).  This quantifies whether
individual neurons carry information about trial outcome.

Produces:
  - Fig 6A: auROC distribution (histogram) with significance threshold
  - Fig 6B: Fraction of selective units per stage
  - Fig 6C: Selectivity heatmap (Expert sessions, sorted by auROC)
  - Fig 6D: Mean PSTH for top Hit-preferring vs Miss-preferring units

Saves: figures/02_single_unit/outcome_selectivity_stats.csv
       cache/selectivity_all_sessions.csv
"""

import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    STAGE_ORDER, STAGE_COLORS, OUTCOME_COLORS, CACHE_DIR, DEFAULT_BIN_SIZE,
)
from loader import load_staging_manifest, load_session, load_waveform_labels
from utils import (
    get_good_cluster_ids, build_population_tensor, smooth_psth,
    compute_zscore_normalized, compute_auroc, fdr_correct,
)
from plotting import setup_style, save_figure, add_stage_background

setup_style()

# Parameters
WINDOW = (-0.5, 1.0)
BIN_SIZE = DEFAULT_BIN_SIZE
RESP_WIN = (0.0, 0.25)       # response window for auROC
BASE_WIN = (-0.4, -0.05)     # baseline window
MIN_TRIALS_PER_CLASS = 5
N_PERM = 500                  # permutations for significance


def compute_unit_selectivity(session, sname, stage, sidx):
    """Compute Hit vs Miss auROC for each unit in a session."""
    good_ids = get_good_cluster_ids(session, min_rate_hz=1.0)
    if len(good_ids) < 3:
        return []

    trials = session.trials
    go_hit_idx = [
        i for i, t in enumerate(trials)
        if getattr(t, "trialoutcome", None) == "Hit"
        and (getattr(t, "change_size", None) or 1.0) > 1.01
    ]
    go_miss_idx = [
        i for i, t in enumerate(trials)
        if getattr(t, "trialoutcome", None) == "Miss"
        and (getattr(t, "change_size", None) or 1.0) > 1.01
    ]
    fa_idx = [
        i for i, t in enumerate(trials)
        if getattr(t, "trialoutcome", None) == "Hit"
        and (getattr(t, "change_size", None) or 1.0) <= 1.01
    ]

    if len(go_hit_idx) < MIN_TRIALS_PER_CLASS or len(go_miss_idx) < MIN_TRIALS_PER_CLASS:
        return []

    tensor, bin_centers, used = build_population_tensor(
        session, good_ids, event_name="Change_ON",
        window=WINDOW, bin_size=BIN_SIZE,
        trial_indices=go_hit_idx + go_miss_idx,
    )

    if tensor.shape[0] < 2 * MIN_TRIALS_PER_CLASS:
        return []

    # Build FA tensor for Hit vs FA selectivity
    fa_tensor = None
    has_fa = len(fa_idx) >= 3
    if has_fa:
        fa_tensor, _, _ = build_population_tensor(
            session, good_ids, event_name="Change_ON",
            window=WINDOW, bin_size=BIN_SIZE,
            trial_indices=fa_idx,
        )
        if fa_tensor.shape[0] < 3:
            has_fa = False
            fa_tensor = None

    # Build labels
    labels = np.array([
        1 if getattr(trials[i], "trialoutcome", None) == "Hit" else 0
        for i in used
    ])

    # Response window mask
    resp_mask = (bin_centers >= RESP_WIN[0]) & (bin_centers < RESP_WIN[1])
    base_mask = (bin_centers >= BASE_WIN[0]) & (bin_centers < BASE_WIN[1])

    results = []
    rng = np.random.default_rng(42)

    for u_i, cid in enumerate(good_ids):
        if u_i >= tensor.shape[2]:
            break

        # Mean FR in response window per trial
        resp_fr = np.nanmean(tensor[:, resp_mask, u_i], axis=1)
        base_fr = np.nanmean(tensor[:, base_mask, u_i], axis=1)

        hit_resp = resp_fr[labels == 1]
        miss_resp = resp_fr[labels == 0]

        # auROC
        auroc = compute_auroc(hit_resp, miss_resp)

        # Permutation test for significance
        n_extreme = 0
        combined = resp_fr.copy()
        for _ in range(N_PERM):
            rng.shuffle(combined)
            perm_auroc = compute_auroc(
                combined[: len(hit_resp)],
                combined[len(hit_resp):],
            )
            if np.isfinite(perm_auroc) and np.isfinite(auroc):
                if abs(perm_auroc - 0.5) >= abs(auroc - 0.5):
                    n_extreme += 1
        p_val = (n_extreme + 1) / (N_PERM + 1)

        # Mean response and baseline FR
        mean_resp = float(np.nanmean(resp_fr))
        mean_base = float(np.nanmean(base_fr))

        # Hit vs FA auROC
        auroc_fa = np.nan
        p_value_fa = np.nan
        n_fa_val = 0
        if has_fa and u_i < fa_tensor.shape[2]:
            fa_resp_fr = np.nanmean(fa_tensor[:, resp_mask, u_i], axis=1)
            auroc_fa = compute_auroc(hit_resp, fa_resp_fr)
            n_fa_val = len(fa_resp_fr)

            # Permutation test for Hit vs FA significance
            n_extreme_fa = 0
            combined_fa = np.concatenate([hit_resp, fa_resp_fr])
            for _ in range(N_PERM):
                rng.shuffle(combined_fa)
                perm_auroc_fa = compute_auroc(
                    combined_fa[:len(hit_resp)],
                    combined_fa[len(hit_resp):],
                )
                if np.isfinite(perm_auroc_fa) and np.isfinite(auroc_fa):
                    if abs(perm_auroc_fa - 0.5) >= abs(auroc_fa - 0.5):
                        n_extreme_fa += 1
            p_value_fa = (n_extreme_fa + 1) / (N_PERM + 1)

        results.append({
            "session_name": sname,
            "cluster_id": cid,
            "stage": stage,
            "session_idx": sidx,
            "auroc": auroc,
            "p_value": p_val,
            "auroc_fa": auroc_fa,
            "p_value_fa": p_value_fa,
            "n_fa": n_fa_val,
            "mean_resp_fr": mean_resp,
            "mean_base_fr": mean_base,
            "n_hit": int(labels.sum()),
            "n_miss": int((~labels.astype(bool)).sum()),
        })

    return results


def main():
    print("[02b] Outcome selectivity analysis (Hit vs Miss auROC)...")
    manifest = load_staging_manifest(qc_only=True)

    # Check for cached results
    cache_path = os.path.join(CACHE_DIR, "selectivity_all_sessions.csv")
    recompute = True
    if os.path.exists(cache_path):
        sel_df = pd.read_csv(cache_path)
        if "auroc_fa" in sel_df.columns:
            print(f"  Loading cached results from {cache_path}")
            recompute = False
        else:
            print("  Stale cache (missing FA columns), recomputing...")
    if recompute:
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

            res = compute_unit_selectivity(sess, sname, stage, sidx)
            all_results.extend(res)
            print(f"{len(res)} units" if res else "insufficient data")

            del sess
            gc.collect()

        sel_df = pd.DataFrame(all_results)
        if len(sel_df) > 0:
            sel_df.to_csv(cache_path, index=False)
            print(f"\n  Cached {len(sel_df)} units to {cache_path}")

    if len(sel_df) == 0:
        print("  No selectivity data. Exiting.")
        return

    # FDR correction
    sel_df["significant"] = fdr_correct(sel_df["p_value"].values, alpha=0.05)
    n_sig = sel_df["significant"].sum()
    n_total = len(sel_df)
    print(f"\n  {n_sig}/{n_total} units significant Hit vs Miss (FDR q<0.05)")

    # FDR correction for Hit vs FA
    sel_df["significant_fa"] = False
    if "p_value_fa" in sel_df.columns:
        fa_pvals = sel_df["p_value_fa"].values
        finite_fa = np.isfinite(fa_pvals)
        if finite_fa.sum() > 0:
            sig_fa = np.zeros(len(sel_df), dtype=bool)
            sig_fa[finite_fa] = fdr_correct(fa_pvals[finite_fa], alpha=0.05)
            sel_df["significant_fa"] = sig_fa
        n_sig_fa = sel_df["significant_fa"].sum()
        n_with_fa = int(finite_fa.sum())
        print(f"  {n_sig_fa}/{n_with_fa} units significant Hit vs FA (FDR q<0.05)")

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

    # Panel A: auROC distribution
    ax_a = fig.add_subplot(gs[0, 0])
    auroc_vals = sel_df["auroc"].dropna().values

    ax_a.hist(auroc_vals, bins=40, color="#7986CB", edgecolor="white",
              linewidth=0.5, alpha=0.8, density=True)

    # Highlight significant
    sig_auroc = sel_df[sel_df["significant"]]["auroc"].dropna().values
    if len(sig_auroc) > 0:
        ax_a.hist(sig_auroc, bins=40, color="#E53935", edgecolor="white",
                  linewidth=0.5, alpha=0.6, density=True, label="Significant (FDR)")

    ax_a.axvline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_a.set_xlabel("auROC (Hit vs Miss)")
    ax_a.set_ylabel("Density")
    ax_a.set_title(f"A. Outcome selectivity distribution (n={len(auroc_vals)} units)")
    ax_a.legend(fontsize=8)

    # Panel B: Fraction selective per stage
    ax_b = fig.add_subplot(gs[0, 1])
    stage_frac = {}
    for stage in STAGE_ORDER:
        sub = sel_df[sel_df["stage"] == stage]
        if len(sub) > 0:
            stage_frac[stage] = {
                "frac": sub["significant"].mean(),
                "n_sig": sub["significant"].sum(),
                "n_total": len(sub),
            }
        else:
            stage_frac[stage] = {"frac": 0, "n_sig": 0, "n_total": 0}

    bar_x = range(len(STAGE_ORDER))
    bar_vals = [stage_frac[s]["frac"] for s in STAGE_ORDER]
    bar_colors = [STAGE_COLORS[s] for s in STAGE_ORDER]
    bars = ax_b.bar(bar_x, bar_vals, color=bar_colors, edgecolor="white", linewidth=1)

    for i, (s, stage) in enumerate(zip(bar_x, STAGE_ORDER)):
        info = stage_frac[stage]
        ax_b.text(s, info["frac"] + 0.01, f"{info['n_sig']}/{info['n_total']}",
                  ha="center", fontsize=8)

    ax_b.set_xticks(bar_x)
    ax_b.set_xticklabels(STAGE_ORDER)
    ax_b.set_ylabel("Fraction selective (FDR q<0.05)")
    ax_b.set_ylim(0, max(bar_vals) * 1.3 if bar_vals and max(bar_vals) > 0 else 0.5)
    ax_b.set_title("B. Fraction outcome-selective by stage")

    # Panel C: Selectivity heatmap (Expert sessions)
    ax_c = fig.add_subplot(gs[1, 0])
    expert = sel_df[sel_df["stage"] == "Expert"].copy()

    if len(expert) > 0:
        expert_sorted = expert.sort_values("auroc", ascending=False)
        # Create a horizontal bar chart showing auROC per unit
        y_pos = np.arange(len(expert_sorted))
        auroc_centered = expert_sorted["auroc"].values - 0.5

        colors = ["#4CAF50" if a > 0 else "#F44336" for a in auroc_centered]
        sig_mask = expert_sorted["significant"].values
        edge_colors = ["#B71C1C" if s else "none" for s in sig_mask]

        ax_c.barh(y_pos, auroc_centered, color=colors, edgecolor=edge_colors,
                  linewidth=0.3, height=1.0)
        ax_c.axvline(0, color="k", linewidth=0.5)
        ax_c.set_xlabel("auROC - 0.5 (Hit preference →)")
        ax_c.set_ylabel(f"Units (n={len(expert_sorted)})")
        ax_c.set_yticks([])
        n_hit_pref = (auroc_centered > 0).sum()
        n_miss_pref = (auroc_centered < 0).sum()
        ax_c.set_title(f"C. Expert selectivity (Hit-pref: {n_hit_pref}, Miss-pref: {n_miss_pref})")
    else:
        ax_c.text(0.5, 0.5, "No Expert data", transform=ax_c.transAxes, ha="center")
        ax_c.set_title("C. Expert selectivity")

    # Panel D: auROC vs session index (learning trajectory)
    ax_d = fig.add_subplot(gs[1, 1])
    add_stage_background(ax_d, manifest)

    # Compute per-session mean auROC
    sess_stats = sel_df.groupby(["session_name", "session_idx", "stage"]).agg(
        mean_auroc=("auroc", "mean"),
        frac_sig=("significant", "mean"),
        n_units=("cluster_id", "count"),
    ).reset_index()
    sess_stats = sess_stats.sort_values("session_idx")

    for stage in STAGE_ORDER:
        sub = sess_stats[sess_stats["stage"] == stage]
        if len(sub) > 0:
            ax_d.scatter(sub["session_idx"], sub["mean_auroc"],
                         c=STAGE_COLORS[stage], s=60, edgecolors="white",
                         linewidths=0.5, zorder=3, label=stage)
    if len(sess_stats) > 0:
        ax_d.plot(sess_stats["session_idx"], sess_stats["mean_auroc"],
                  color="gray", alpha=0.3, linewidth=1, zorder=2)
    ax_d.axhline(0.5, color="gray", linestyle=":", linewidth=0.5)
    ax_d.set_xlabel("Session index")
    ax_d.set_ylabel("Mean auROC")
    ax_d.set_title("D. Mean outcome selectivity across learning")
    ax_d.legend(fontsize=8)

    # Panel E: Hit vs FA auROC distribution
    ax_e = fig.add_subplot(gs[2, 0])
    if "auroc_fa" in sel_df.columns:
        auroc_fa_vals = sel_df["auroc_fa"].dropna().values
        if len(auroc_fa_vals) > 0:
            ax_e.hist(auroc_fa_vals, bins=40, color=OUTCOME_COLORS["FA"],
                      edgecolor="white", linewidth=0.5, alpha=0.8, density=True,
                      label="All units")
            sig_fa_auroc = sel_df[sel_df["significant_fa"]]["auroc_fa"].dropna().values
            if len(sig_fa_auroc) > 0:
                ax_e.hist(sig_fa_auroc, bins=40, color="#E53935", edgecolor="white",
                          linewidth=0.5, alpha=0.6, density=True,
                          label="Significant (FDR)")
            ax_e.axvline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
            ax_e.set_xlabel("auROC (Hit vs FA)")
            ax_e.set_ylabel("Density")
            ax_e.set_title(f"E. Hit vs FA selectivity (n={len(auroc_fa_vals)} units)")
            ax_e.legend(fontsize=8)
        else:
            ax_e.text(0.5, 0.5, "No FA data", transform=ax_e.transAxes, ha="center")
            ax_e.set_title("E. Hit vs FA selectivity")
    else:
        ax_e.text(0.5, 0.5, "No FA data", transform=ax_e.transAxes, ha="center")
        ax_e.set_title("E. Hit vs FA selectivity")

    # Panel F: Hit-Miss auROC vs Hit-FA auROC scatter
    ax_f = fig.add_subplot(gs[2, 1])
    if "auroc_fa" in sel_df.columns:
        both_valid = sel_df.dropna(subset=["auroc", "auroc_fa"])
        if len(both_valid) > 0:
            ax_f.scatter(both_valid["auroc"], both_valid["auroc_fa"],
                         c="#7986CB", s=15, alpha=0.4, edgecolors="none")
            # Highlight doubly-significant
            doubly_sig = both_valid[
                both_valid["significant"] & both_valid["significant_fa"]
            ]
            if len(doubly_sig) > 0:
                ax_f.scatter(doubly_sig["auroc"], doubly_sig["auroc_fa"],
                             c="#E53935", s=20, alpha=0.6, edgecolors="none",
                             label=f"Both sig (n={len(doubly_sig)})")
            ax_f.axhline(0.5, color="gray", linewidth=0.5, linestyle=":")
            ax_f.axvline(0.5, color="gray", linewidth=0.5, linestyle=":")
            ax_f.plot([0, 1], [0, 1], color="gray", linestyle="--",
                      linewidth=0.5, alpha=0.5)
            rho_fa, p_fa = spearmanr(both_valid["auroc"], both_valid["auroc_fa"])
            ax_f.set_xlabel("auROC (Hit vs Miss)")
            ax_f.set_ylabel("auROC (Hit vs FA)")
            ax_f.set_title(
                f"F. Outcome vs catch-trial selectivity "
                f"(\u03C1={rho_fa:.2f}, p={p_fa:.1e})"
            )
            ax_f.legend(fontsize=8)
        else:
            ax_f.text(0.5, 0.5, "No overlapping data",
                      transform=ax_f.transAxes, ha="center")
            ax_f.set_title("F. Outcome vs catch-trial selectivity")
    else:
        ax_f.text(0.5, 0.5, "No FA data", transform=ax_f.transAxes, ha="center")
        ax_f.set_title("F. Outcome vs catch-trial selectivity")

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # Selectivity trend
    if len(sess_stats) >= 3:
        rho, p = spearmanr(sess_stats["session_idx"], sess_stats["mean_auroc"])
        stats.append({"test": "mean_auroc_vs_session_spearman", "rho": rho, "p": p,
                      "n": len(sess_stats)})

    # Fraction selective trend
    if len(sess_stats) >= 3:
        rho, p = spearmanr(sess_stats["session_idx"], sess_stats["frac_sig"])
        stats.append({"test": "frac_selective_vs_session_spearman", "rho": rho, "p": p,
                      "n": len(sess_stats)})

    # Stage comparison using Kruskal-Wallis on per-unit auROC
    from scipy.stats import kruskal
    stage_groups = [sel_df[sel_df["stage"] == s]["auroc"].dropna().values for s in STAGE_ORDER]
    stage_groups = [g for g in stage_groups if len(g) >= 2 and np.std(g) > 0]
    if len(stage_groups) >= 2:
        try:
            h, p = kruskal(*stage_groups)
            stats.append({"test": "auroc_kruskal_by_stage", "H": h, "p": p})
        except ValueError:
            pass

    # Overall: is population biased toward Hit preference?
    from scipy.stats import wilcoxon
    finite_auroc = sel_df["auroc"].dropna().values
    if len(finite_auroc) >= 10:
        try:
            stat_w, p_w = wilcoxon(finite_auroc - 0.5)
            stats.append({"test": "auroc_vs_0.5_wilcoxon", "W": stat_w, "p": p_w,
                          "median_auroc": float(np.median(finite_auroc))})
        except ValueError:
            pass

    # Hit vs FA selectivity stats
    if "auroc_fa" in sel_df.columns:
        finite_auroc_fa = sel_df["auroc_fa"].dropna().values
        if len(finite_auroc_fa) >= 10:
            try:
                stat_w_fa, p_w_fa = wilcoxon(finite_auroc_fa - 0.5)
                stats.append({"test": "auroc_fa_vs_0.5_wilcoxon", "W": stat_w_fa,
                              "p": p_w_fa,
                              "median_auroc_fa": float(np.median(finite_auroc_fa))})
            except ValueError:
                pass
        # Correlation between Hit-Miss and Hit-FA auROC
        both = sel_df.dropna(subset=["auroc", "auroc_fa"])
        if len(both) >= 5:
            rho_both, p_both = spearmanr(both["auroc"], both["auroc_fa"])
            stats.append({"test": "auroc_vs_auroc_fa_spearman", "rho": rho_both,
                          "p": p_both, "n": len(both)})

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig09_outcome_selectivity", "02_single_unit")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "02_single_unit", "outcome_selectivity_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: p={row.get('p', 'N/A')}")


if __name__ == "__main__":
    main()
