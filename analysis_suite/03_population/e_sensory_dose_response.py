"""Fig 17: Sensory dose-response — coding direction dose-response (all go trials).

Companion to 03d (Hit-only dose-response).  Instead of conditioning on
detection at each change size, this script uses ALL go trials (Hit + Miss)
so the dose-response reflects the population's average sensory
representation of change magnitude, free of selection bias.

At low change sizes where d' is small, 03d's Hit-only selection keeps only
the rare detected trials — those with unusually strong neural responses —
inflating the small-dose projection and flattening the curve.  This script
removes that bias by pooling all go trials at each dose.

Produces (5 × 2 figure):
  Rows 1–4: One row per HMM state (Disengaged, Engaged, Impulsive) +
            All trials (pooled across states).
    Left column:  Grand-average time-resolved CD (Expert sessions, z-scored)
                  FA / all go-small / all go-big
    Right column: All-go-trials dose-response curve (FA → 1.25 → … → 4.0)
  Row 5 (summary):
    I. Dose-response slope (ρ) by learning stage (all go trials)
    J. Sensory fraction by stage

Saves: figures/03_population/sensory_dose_response_stats.csv
"""

import os
import sys
import gc
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    STAGE_ORDER, STAGE_COLORS, OUTCOME_COLORS,
    HMM_STATE_ORDER, HMM_STATE_COLORS,
)
from loader import load_staging_manifest, load_session, load_hmm_assignments
from utils import smooth_psth
from plotting import setup_style, save_figure

# Import heavy analysis function from script d (avoids duplication)
from d_state_matched_cd import (
    analyse_session, DOSE_LEVELS, DOSE_LABELS, BIN_SIZE, RESP_WIN,
)

setup_style()


# ── Helpers ───────────────────────────────────────────────────────────

CHANGE_BL = (-0.5, -0.1)


def _zscore_baseline(trace, bin_centers, bl_window):
    bl_mask = (bin_centers >= bl_window[0]) & (bin_centers < bl_window[1])
    if bl_mask.sum() < 2:
        return trace
    bl = trace[bl_mask]
    mu, sd = bl.mean(), bl.std()
    if sd < 1e-12:
        return trace - mu
    return (trace - mu) / sd


def _zscore_resp_scalar(d, bc):
    """Z-score a category's mean trace to baseline, return resp-window mean."""
    if d is None:
        return np.nan
    pm = d.get("proj_mean")
    if pm is None or len(pm) != len(bc):
        return np.nan
    sm = smooth_psth(pm, BIN_SIZE, 15.0)
    z = _zscore_baseline(sm, bc, CHANGE_BL)
    resp_m = (bc >= RESP_WIN[0]) & (bc < RESP_WIN[1])
    return float(np.mean(z[resp_m])) if resp_m.sum() > 0 else np.nan


def _get(r, state, cat):
    """Lookup (state, cat) with fallback to pooled."""
    d = r.get((state, cat))
    if d is not None:
        return d
    if state == "_pooled":
        return None
    return r.get(("_pooled", cat))


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, default=1)
    args = parser.parse_args()

    print("[03e] Sensory dose-response (all go trials)...")
    manifest = load_staging_manifest(qc_only=True)
    hmm_df = load_hmm_assignments()

    tasks = [
        (int(row["session_name"]), row["stage"], row["session_idx"])
        for _, row in manifest.iterrows()
    ]

    results = {}
    for sname, stage, sidx in tasks:
        print(f"  Session {sname} ({stage})...", end=" ", flush=True)
        try:
            sess = load_session(sname)
        except FileNotFoundError:
            print("not found")
            continue
        r = analyse_session(sess, sname, hmm_df, stage, sidx)
        if r is not None:
            results[sname] = r
            n_groups = sum(1 for k in r if isinstance(k, tuple))
            print(f"{n_groups} state\u00d7category groups")
        else:
            print("insufficient data")
        del sess
        gc.collect()

    print(f"\n  Analysed {len(results)} sessions")
    if not results:
        print("  No results. Exiting.")
        return

    # ── Figure layout ─────────────────────────────────────────────────
    STATE_ROWS = list(HMM_STATE_ORDER) + ["_pooled"]
    STATE_LABELS = {s: s for s in HMM_STATE_ORDER}
    STATE_LABELS["_pooled"] = "All trials"
    STATE_ROW_COLORS = dict(HMM_STATE_COLORS)
    STATE_ROW_COLORS["_pooled"] = "#555555"

    n_rows = len(STATE_ROWS) + 1   # +1 for summary row
    fig = plt.figure(figsize=(20, 5 * n_rows + 2))
    gs = gridspec.GridSpec(n_rows, 2, hspace=0.45, wspace=0.3,
                           top=0.95, bottom=0.10)

    fig.suptitle("Sensory dose\u2013response on the CD (all go trials at each change size)",
                 fontsize=14, fontweight="bold", y=0.98)

    expert = {k: v for k, v in results.items() if v["stage"] == "Expert"}

    # Colour scheme — use "go" variants instead of hit-only
    CAT_COLORS = {
        "fa":        OUTCOME_COLORS["FA"],
        "go_small":  "#81C784",
        "go_big":    "#2E7D32",
        "go_all":    OUTCOME_COLORS["Hit"],
    }
    CAT_LABELS = {
        "fa":        "True FA",
        "go_small":  "Small-\u0394 (all go)",
        "go_big":    "Big-\u0394 (all go)",
        "go_all":    "All go",
    }

    # Dose categories: FA then go_dose at each level
    dose_cats = ["fa"] + [f"go_{d}" for d in DOSE_LEVELS[1:]]
    dose_x = list(range(len(DOSE_LEVELS)))

    panel_letter = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    # ── State rows ────────────────────────────────────────────────────
    for row_i, state in enumerate(STATE_ROWS):
        state_label = STATE_LABELS[state]
        state_color = STATE_ROW_COLORS[state]
        ltr_left = next(panel_letter)
        ltr_right = next(panel_letter)

        # ── Left panel: Grand-average time-resolved (Expert sessions) ─
        ax_left = fig.add_subplot(gs[row_i, 0])
        if expert:
            ref_bc = list(expert.values())[0]["bin_centers"]
            grand = {}
            for cat in ["fa", "go_small", "go_big"]:
                traces = []
                for r in expert.values():
                    d = _get(r, state, cat)
                    if d is not None and len(d["proj_mean"]) == len(ref_bc):
                        sm = smooth_psth(d["proj_mean"], BIN_SIZE, 15.0)
                        traces.append(_zscore_baseline(sm, ref_bc, CHANGE_BL))
                if traces:
                    grand[cat] = (np.mean(traces, axis=0),
                                  np.std(traces, axis=0) / np.sqrt(len(traces)),
                                  len(traces))

            plotted = False
            for cat in ["fa", "go_small", "go_big"]:
                if cat in grand:
                    m, s, n = grand[cat]
                    ax_left.plot(ref_bc, m, color=CAT_COLORS[cat], linewidth=2,
                                 label=f"{CAT_LABELS[cat]} (n={n} sess)")
                    ax_left.fill_between(ref_bc, m - s, m + s,
                                         color=CAT_COLORS[cat], alpha=0.2)
                    plotted = True
            if plotted:
                ax_left.axvline(0, color="k", linestyle="--",
                                linewidth=0.8, alpha=0.5)
            ax_left.set_title(f"{ltr_left}. Grand-average \u2014 {state_label} "
                              f"Expert sessions", color=state_color,
                              fontweight="bold")
        else:
            ax_left.text(0.5, 0.5, "No Expert sessions",
                         transform=ax_left.transAxes, ha="center")
            ax_left.set_title(f"{ltr_left}. Grand-average \u2014 {state_label}",
                              color=state_color, fontweight="bold")
        ax_left.set_xlabel("Time from Change_ON (s)")
        ax_left.set_ylabel("CD projection (z-score vs baseline)")
        ax_left.legend(fontsize=7, loc="upper left")

        # ── Right panel: All-go dose-response (Expert sessions) ───────
        ax_right = fig.add_subplot(gs[row_i, 1])
        if expert:
            ref_bc = list(expert.values())[0]["bin_centers"]
            dose_per_session = []
            for r in expert.values():
                row = []
                for cat in dose_cats:
                    d = _get(r, state, cat)
                    row.append(_zscore_resp_scalar(d, ref_bc))
                dose_per_session.append(row)

            dose_arr = np.array(dose_per_session)
            for sess_row in dose_arr:
                valid = np.isfinite(sess_row)
                if valid.sum() >= 2:
                    ax_right.plot(np.array(dose_x)[valid], sess_row[valid],
                                  color="gray", alpha=0.15, linewidth=0.8,
                                  zorder=1)

            dose_mean = np.nanmean(dose_arr, axis=0)
            dose_sem = np.nanstd(dose_arr, axis=0) / np.sqrt(
                np.sum(np.isfinite(dose_arr), axis=0).clip(1))
            finite = np.isfinite(dose_mean)
            if finite.any():
                x_f = np.array(dose_x)[finite]
                ax_right.errorbar(x_f, dose_mean[finite],
                                  yerr=dose_sem[finite],
                                  color=state_color, linewidth=2,
                                  marker="o", markersize=6, capsize=4,
                                  zorder=3, label="Grand mean")
                # Spearman per session
                rho_list = []
                for sess_row in dose_arr:
                    v = np.isfinite(sess_row)
                    if v.sum() >= 3:
                        r_s, _ = spearmanr(np.array(DOSE_LEVELS)[v],
                                           sess_row[v])
                        if np.isfinite(r_s):
                            rho_list.append(r_s)
                if rho_list:
                    med_rho = np.median(rho_list)
                    ax_right.set_title(
                        f"{ltr_right}. All-go dose-response ({state_label}) "
                        f"\u2014 median \u03c1={med_rho:.2f}",
                        color=state_color, fontweight="bold")
                else:
                    ax_right.set_title(
                        f"{ltr_right}. All-go dose-response ({state_label})",
                        color=state_color, fontweight="bold")
                ax_right.legend(fontsize=7)
            else:
                ax_right.set_title(
                    f"{ltr_right}. All-go dose-response ({state_label})",
                    color=state_color, fontweight="bold")
        else:
            ax_right.set_title(
                f"{ltr_right}. All-go dose-response ({state_label})",
                color=state_color, fontweight="bold")

        ax_right.set_xticks(dose_x)
        ax_right.set_xticklabels(DOSE_LABELS, rotation=30)
        ax_right.set_xlabel("Change size (0 = catch)")
        ax_right.set_ylabel("CD projection (z-score vs baseline)")

    # ── Summary row ───────────────────────────────────────────────────
    ltr_e = next(panel_letter)
    ltr_f = next(panel_letter)
    SUMMARY_STATE = "_pooled"

    # ── Panel I: dose-slope ρ by stage ────────────────────────────────
    ax_e = fig.add_subplot(gs[len(STATE_ROWS), 0])

    stage_slopes = {s: [] for s in STAGE_ORDER}
    for r in results.values():
        _bc = r["bin_centers"]
        row = []
        for cat in dose_cats:
            d = _get(r, SUMMARY_STATE, cat)
            row.append(_zscore_resp_scalar(d, _bc))
        finite_vals = [(DOSE_LEVELS[j], row[j])
                       for j in range(len(row)) if np.isfinite(row[j])]
        if len(finite_vals) >= 3:
            xs, ys = zip(*finite_vals)
            rho, _ = spearmanr(xs, ys)
            if np.isfinite(rho):
                stage_slopes[r["stage"]].append(rho)

    box_data, box_pos, box_colors = [], [], []
    for i, stage in enumerate(STAGE_ORDER):
        if stage_slopes[stage]:
            box_pos.append(i)
            box_data.append(stage_slopes[stage])
            box_colors.append(STAGE_COLORS[stage])

    if box_data:
        bp = ax_e.boxplot(box_data, positions=box_pos, widths=0.5,
                          patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for pos, vals, color in zip(box_pos, box_data, box_colors):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax_e.scatter(pos + jitter, vals, c=color, s=40,
                         edgecolors="white", linewidths=0.5, zorder=3)

    ax_e.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_e.set_xticks(range(len(STAGE_ORDER)))
    ax_e.set_xticklabels(STAGE_ORDER)
    ax_e.set_ylabel("Dose-response slope (\u03c1)")
    ax_e.set_title(f"{ltr_e}. All-go dose slope across learning")

    # ── Panel J: sensory fraction by stage ────────────────────────────
    ax_f = fig.add_subplot(gs[len(STATE_ROWS), 1])

    stage_fracs = {s: [] for s in STAGE_ORDER}
    for r in results.values():
        _bc = r["bin_centers"]
        d_big = _get(r, SUMMARY_STATE, "go_big")
        d_fa = _get(r, SUMMARY_STATE, "fa")
        d_miss = _get(r, SUMMARY_STATE, "miss")
        if d_big is None or d_fa is None or d_miss is None:
            continue
        h = _zscore_resp_scalar(d_big, _bc)
        f = _zscore_resp_scalar(d_fa, _bc)
        m = _zscore_resp_scalar(d_miss, _bc)
        if not (np.isfinite(h) and np.isfinite(f) and np.isfinite(m)):
            continue
        denom = h - m
        if abs(denom) < 1e-6:
            continue
        sensory_frac = (h - f) / denom
        stage_fracs[r["stage"]].append(sensory_frac)

    box2_data, box2_pos, box2_colors = [], [], []
    for i, stage in enumerate(STAGE_ORDER):
        if stage_fracs[stage]:
            box2_pos.append(i)
            box2_data.append(stage_fracs[stage])
            box2_colors.append(STAGE_COLORS[stage])

    if box2_data:
        bp2 = ax_f.boxplot(box2_data, positions=box2_pos, widths=0.5,
                           patch_artist=True, showfliers=False)
        for patch, color in zip(bp2["boxes"], box2_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for pos, vals, color in zip(box2_pos, box2_data, box2_colors):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax_f.scatter(pos + jitter, vals, c=color, s=40,
                         edgecolors="white", linewidths=0.5, zorder=3)

    ax_f.axhline(0, color="gray", linewidth=0.5, linestyle=":",
                 label="0 = purely motor")
    ax_f.axhline(1, color="gray", linewidth=0.5, linestyle="--",
                 label="1 = purely sensory")
    ax_f.set_xticks(range(len(STAGE_ORDER)))
    ax_f.set_xticklabels(STAGE_ORDER)
    ax_f.set_ylabel("Sensory fraction\n(goBig\u2212FA) / (goBig\u2212Miss)")
    ax_f.set_title(f"{ltr_f}. Sensory vs motor contribution to CD by stage")
    ax_f.legend(fontsize=8, loc="lower right")

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    all_slopes, all_sidxs = [], []
    for r in results.values():
        _bc = r["bin_centers"]
        row = []
        for cat in dose_cats:
            d = _get(r, SUMMARY_STATE, cat)
            row.append(_zscore_resp_scalar(d, _bc))
        finite_vals = [(DOSE_LEVELS[j], row[j])
                       for j in range(len(row)) if np.isfinite(row[j])]
        if len(finite_vals) >= 3:
            xs, ys = zip(*finite_vals)
            rho, _ = spearmanr(xs, ys)
            if np.isfinite(rho):
                all_slopes.append(rho)
                all_sidxs.append(r["session_idx"])
    if len(all_slopes) >= 3:
        rho_trend, p_trend = spearmanr(all_sidxs, all_slopes)
        stats.append({"test": "go_dose_slope_vs_session_spearman",
                      "rho": rho_trend, "p": p_trend, "n": len(all_slopes)})

    from scipy.stats import kruskal as _kruskal
    valid_groups = [np.array(stage_slopes[s]) for s in STAGE_ORDER
                    if len(stage_slopes[s]) >= 2]
    if len(valid_groups) >= 2:
        try:
            h_val, p_val = _kruskal(*valid_groups)
            stats.append({"test": "go_dose_slope_kruskal_by_stage",
                          "H": h_val, "p": p_val})
        except ValueError:
            pass

    valid_frac = [np.array(stage_fracs[s]) for s in STAGE_ORDER
                  if len(stage_fracs[s]) >= 2]
    if len(valid_frac) >= 2:
        try:
            h_val, p_val = _kruskal(*valid_frac)
            stats.append({"test": "go_sensory_frac_kruskal_by_stage",
                          "H": h_val, "p": p_val})
        except ValueError:
            pass

    expert_slopes = stage_slopes.get("Expert", [])
    if len(expert_slopes) >= 3:
        from scipy.stats import wilcoxon as _wilcoxon
        try:
            w, p = _wilcoxon(expert_slopes)
            stats.append({"test": "expert_go_dose_slope_vs_0_wilcoxon",
                          "W": w, "p": p,
                          "median_rho": float(np.median(expert_slopes)),
                          "n": len(expert_slopes)})
        except ValueError:
            pass

    stats_df = pd.DataFrame(stats)

    # ── Explanation text ──────────────────────────────────────────────
    explanation = (
        "ANALYSIS  Same coding direction as fig\u200910 (Hit vs Miss, cross-"
        "validated), but the dose\u2013response uses ALL go trials (Hit + Miss "
        "combined) at each change size, removing the selection bias that "
        "inflates small-dose projections when conditioning on detection.  "
        "Left column shows grand-average time-resolved CD for FA / Small-\u0394 "
        "(all go) / Big-\u0394 (all go).  Right column shows the unbiased "
        "sensory dose\u2013response.  Bottom row: summary across learning stages."
    )
    fig.text(
        0.5, 0.005, explanation,
        ha="center", va="bottom", fontsize=8,
        fontstyle="italic", color="#444444",
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5",
                  ec="#cccccc", alpha=0.9),
        transform=fig.transFigure,
    )

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig17_sensory_dose_response", "03_population")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "03_population", "sensory_dose_response_stats.csv",
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: p={row.get('p', 'N/A')}")


if __name__ == "__main__":
    main()
