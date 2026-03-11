"""05c - Population geometry shifts across learning.

Tracks how the coding direction (CD) angle and neural state-space
geometry change across learning stages.

Produces:
  - Fig 16A: CD magnitude (Hit vs Miss separation) across sessions
  - Fig 16B: CD angle stability (cosine similarity between consecutive sessions)
  - Fig 16C: CD angle relative to first Expert session
  - Fig 16D: Variance along vs orthogonal to CD across stages

Saves: figures/05_longitudinal/geometry_shift_stats.csv
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

from config import STAGE_ORDER, STAGE_COLORS, OUTCOME_COLORS, CACHE_DIR
from loader import load_staging_manifest, load_session
from utils import (
    get_good_cluster_ids, build_population_tensor,
    compute_zscore_normalized,
)
from plotting import setup_style, save_figure, add_stage_background

setup_style()

WINDOW = (-0.5, 1.0)
BIN_SIZE = 0.025
BASELINE_WIN = (-0.5, -0.05)
RESP_WIN = (0.0, 0.25)
MIN_UNITS = 10
MIN_TRIALS_PER_CLASS = 5


def compute_cd_vector(tensor, labels, bc, resp_win):
    """Compute coding direction as normalized difference of class means in response window."""
    resp_mask = (bc >= resp_win[0]) & (bc < resp_win[1])
    # Mean FR in response window per trial per unit: (n_trials, n_units)
    resp = np.nanmean(tensor[:, resp_mask, :], axis=1)

    hit_mean = np.nanmean(resp[labels == 1], axis=0)
    miss_mean = np.nanmean(resp[labels == 0], axis=0)

    cd = hit_mean - miss_mean
    cd_norm = np.linalg.norm(cd)
    if cd_norm > 0:
        cd_unit = cd / cd_norm
    else:
        cd_unit = cd
    return cd, cd_unit, cd_norm


def project_onto_cd(tensor, bc, cd_unit):
    """Project population activity onto coding direction at each time bin."""
    n_trials, n_bins, n_units = tensor.shape
    projections = np.full((n_trials, n_bins), np.nan)
    for b in range(n_bins):
        activity = tensor[:, b, :]
        valid = ~np.isnan(activity).any(axis=1)
        projections[valid, b] = activity[valid] @ cd_unit
    return projections


def main():
    print("[05c] Population geometry shift analysis...")
    manifest = load_staging_manifest(qc_only=True)

    results = {}
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

        good_ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
        if len(good_ids) < MIN_UNITS:
            print("too few units")
            del sess; gc.collect()
            continue

        # Separate go-trial hits/misses from catch-trial FAs
        trials = sess.trials
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
        cr_idx = [
            i for i, t in enumerate(trials)
            if getattr(t, "trialoutcome", None) == "Miss"
            and (getattr(t, "change_size", None) or 1.0) <= 1.01
        ]

        tensor, bc, used = build_population_tensor(
            sess, good_ids, event_name="Change_ON",
            window=WINDOW, bin_size=BIN_SIZE,
            trial_indices=go_hit_idx + go_miss_idx,
        )

        labels = np.array([
            1 if getattr(trials[i], "trialoutcome", None) == "Hit" else 0
            for i in used
        ])

        n_hit = labels.sum()
        n_miss = (~labels.astype(bool)).sum()
        if n_hit < MIN_TRIALS_PER_CLASS or n_miss < MIN_TRIALS_PER_CLASS:
            print("too few per class")
            del sess; gc.collect()
            continue

        # Z-score
        z_tensor = compute_zscore_normalized(tensor, bc, BASELINE_WIN)

        # Compute CD
        cd_vec, cd_unit, cd_mag = compute_cd_vector(z_tensor, labels, bc, RESP_WIN)

        # Variance along CD vs orthogonal
        resp_mask = (bc >= RESP_WIN[0]) & (bc < RESP_WIN[1])
        resp_activity = np.nanmean(z_tensor[:, resp_mask, :], axis=1)  # (n_trials, n_units)
        valid = ~np.isnan(resp_activity).any(axis=1)
        resp_clean = resp_activity[valid]

        if len(resp_clean) > 2 and np.linalg.norm(cd_unit) > 0:
            proj_cd = resp_clean @ cd_unit
            var_along = float(np.var(proj_cd))

            # Orthogonal variance: total variance minus CD variance
            total_var = float(np.sum(np.var(resp_clean, axis=0)))
            var_ortho = total_var - var_along
        else:
            var_along = np.nan
            var_ortho = np.nan

        # Project FA trials onto CD
        fa_cd_projection = np.nan
        n_fa = 0
        hit_cd_projection = np.nan
        miss_cd_projection = np.nan
        if len(resp_clean) > 2 and np.linalg.norm(cd_unit) > 0:
            hit_proj_vals = resp_clean[labels[valid] == 1] @ cd_unit
            miss_proj_vals = resp_clean[labels[valid] == 0] @ cd_unit
            if len(hit_proj_vals) > 0:
                hit_cd_projection = float(np.mean(hit_proj_vals))
            if len(miss_proj_vals) > 0:
                miss_cd_projection = float(np.mean(miss_proj_vals))

        if len(fa_idx) >= 3 and np.linalg.norm(cd_unit) > 0:
            fa_tensor, _, _ = build_population_tensor(
                sess, good_ids, event_name="Change_ON",
                window=WINDOW, bin_size=BIN_SIZE,
                trial_indices=fa_idx,
            )
            if fa_tensor.shape[0] >= 3:
                fa_z = compute_zscore_normalized(fa_tensor, bc, BASELINE_WIN)
                resp_mask_fa = (bc >= RESP_WIN[0]) & (bc < RESP_WIN[1])
                fa_resp = np.nanmean(fa_z[:, resp_mask_fa, :], axis=1)
                valid_fa = ~np.isnan(fa_resp).any(axis=1)
                if valid_fa.sum() > 0:
                    fa_cd_projection = float(np.mean(fa_resp[valid_fa] @ cd_unit))
                    n_fa = int(valid_fa.sum())

        cr_cd_projection = np.nan
        n_cr = 0
        if len(cr_idx) >= 3 and np.linalg.norm(cd_unit) > 0:
            cr_tensor, _, _ = build_population_tensor(
                sess, good_ids, event_name="Change_ON",
                window=WINDOW, bin_size=BIN_SIZE,
                trial_indices=cr_idx,
            )
            if cr_tensor.shape[0] >= 3:
                cr_z = compute_zscore_normalized(cr_tensor, bc, BASELINE_WIN)
                cr_resp = np.nanmean(cr_z[:, resp_mask, :], axis=1)
                valid_cr = ~np.isnan(cr_resp).any(axis=1)
                if valid_cr.sum() > 0:
                    cr_cd_projection = float(np.mean(cr_resp[valid_cr] @ cd_unit))
                    n_cr = int(valid_cr.sum())

        results[sname] = {
            "stage": stage,
            "session_idx": sidx,
            "cd_magnitude": cd_mag,
            "cd_unit": cd_unit,
            "n_units": len(good_ids),
            "var_along_cd": var_along,
            "var_orthogonal": var_ortho,
            "fa_cd_projection": fa_cd_projection,
            "hit_cd_projection": hit_cd_projection,
            "miss_cd_projection": miss_cd_projection,
            "n_fa": n_fa,
            "cr_cd_projection": cr_cd_projection,
            "n_cr": n_cr,
        }

        print(f"CD_mag={cd_mag:.3f}, {len(good_ids)} units")
        del sess; gc.collect()

    print(f"\n  Computed geometry for {len(results)} sessions")

    if not results:
        print("  No data. Exiting.")
        return

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

    sess_list = sorted(results.keys(), key=lambda k: results[k]["session_idx"])
    idxs = [results[k]["session_idx"] for k in sess_list]
    cd_mags = [results[k]["cd_magnitude"] for k in sess_list]
    stages = [results[k]["stage"] for k in sess_list]

    # Panel A: CD magnitude across sessions
    ax_a = fig.add_subplot(gs[0, 0])
    add_stage_background(ax_a, manifest)
    colors = [STAGE_COLORS[s] for s in stages]
    ax_a.scatter(idxs, cd_mags, c=colors, s=60, edgecolors="white",
                 linewidths=0.5, zorder=3)
    ax_a.plot(idxs, cd_mags, color="gray", alpha=0.3, linewidth=1, zorder=2)
    ax_a.set_xlabel("Session index")
    ax_a.set_ylabel("CD magnitude (z-scored)")
    ax_a.set_title("A. Coding direction magnitude across learning")

    # Panel B: CD angle stability (cosine similarity between consecutive sessions)
    ax_b = fig.add_subplot(gs[0, 1])
    add_stage_background(ax_b, manifest)

    cos_sims = []
    cos_idxs = []
    cos_stages = []
    for i in range(1, len(sess_list)):
        cd_prev = results[sess_list[i-1]]["cd_unit"]
        cd_curr = results[sess_list[i]]["cd_unit"]
        # Need same dimensionality
        if len(cd_prev) == len(cd_curr) and np.linalg.norm(cd_prev) > 0 and np.linalg.norm(cd_curr) > 0:
            cos_sim = float(np.dot(cd_prev, cd_curr) / (np.linalg.norm(cd_prev) * np.linalg.norm(cd_curr)))
            cos_sims.append(cos_sim)
            cos_idxs.append(results[sess_list[i]]["session_idx"])
            cos_stages.append(results[sess_list[i]]["stage"])

    if cos_sims:
        cos_colors = [STAGE_COLORS[s] for s in cos_stages]
        ax_b.scatter(cos_idxs, cos_sims, c=cos_colors, s=60, edgecolors="white",
                     linewidths=0.5, zorder=3)
        ax_b.plot(cos_idxs, cos_sims, color="gray", alpha=0.3, linewidth=1, zorder=2)
    ax_b.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_b.set_xlabel("Session index")
    ax_b.set_ylabel("Cosine similarity (consecutive)")
    ax_b.set_ylim(-1.05, 1.05)
    ax_b.set_title("B. CD angle stability")

    # Panel C: CD angle relative to first Expert session
    ax_c = fig.add_subplot(gs[1, 0])
    expert_sessions = [k for k in sess_list if results[k]["stage"] == "Expert"]
    if expert_sessions:
        ref_cd = results[expert_sessions[0]]["cd_unit"]
        ref_n = len(ref_cd)

        angle_sims = []
        angle_idxs = []
        angle_stages = []
        for k in sess_list:
            cd = results[k]["cd_unit"]
            if len(cd) == ref_n and np.linalg.norm(cd) > 0:
                cos_val = float(np.dot(ref_cd, cd) / (np.linalg.norm(ref_cd) * np.linalg.norm(cd)))
                angle_sims.append(cos_val)
                angle_idxs.append(results[k]["session_idx"])
                angle_stages.append(results[k]["stage"])

        if angle_sims:
            add_stage_background(ax_c, manifest)
            ax_c.scatter(angle_idxs, angle_sims,
                         c=[STAGE_COLORS[s] for s in angle_stages],
                         s=60, edgecolors="white", linewidths=0.5, zorder=3)
            ax_c.plot(angle_idxs, angle_sims, color="gray", alpha=0.3,
                      linewidth=1, zorder=2)
            ax_c.axhline(0, color="gray", linewidth=0.5, linestyle=":")

    ax_c.set_xlabel("Session index")
    ax_c.set_ylabel("Cosine sim. to first Expert CD")
    ax_c.set_ylim(-1.05, 1.05)
    ax_c.set_title("C. CD alignment to Expert reference")

    # Panel D: Variance along vs orthogonal to CD by stage
    ax_d = fig.add_subplot(gs[1, 1])
    var_data = {"stage": [], "var_along": [], "var_ortho": [], "ratio": []}
    for k in sess_list:
        r = results[k]
        if np.isfinite(r["var_along_cd"]) and np.isfinite(r["var_orthogonal"]):
            va = r["var_along_cd"]
            vo = r["var_orthogonal"]
            var_data["stage"].append(r["stage"])
            var_data["var_along"].append(va)
            var_data["var_ortho"].append(vo)
            var_data["ratio"].append(va / max(vo, 1e-6))

    var_df = pd.DataFrame(var_data)
    bar_width = 0.35
    for i, stage in enumerate(STAGE_ORDER):
        sub = var_df[var_df["stage"] == stage]
        if len(sub) > 0:
            ax_d.bar(i - bar_width/2, sub["var_along"].mean(), bar_width,
                     color=STAGE_COLORS[stage], alpha=0.8, label="Along CD" if i == 0 else "")
            ax_d.bar(i + bar_width/2, sub["var_ortho"].mean(), bar_width,
                     color=STAGE_COLORS[stage], alpha=0.3, label="Orthogonal" if i == 0 else "")

    ax_d.set_xticks(range(len(STAGE_ORDER)))
    ax_d.set_xticklabels(STAGE_ORDER)
    ax_d.set_ylabel("Variance")
    ax_d.set_title("D. Variance along vs orthogonal to CD")
    ax_d.legend(fontsize=8)

    # Panel E: FA projection onto CD across sessions
    ax_e = fig.add_subplot(gs[2, 0])
    add_stage_background(ax_e, manifest)

    fa_projs = [results[k].get("fa_cd_projection", np.nan) for k in sess_list]
    hit_projs = [results[k].get("hit_cd_projection", np.nan) for k in sess_list]
    miss_projs = [results[k].get("miss_cd_projection", np.nan) for k in sess_list]
    cr_projs = [results[k].get("cr_cd_projection", np.nan) for k in sess_list]

    # Plot Hit, Miss, FA, CR projections onto CD
    for proj_vals, color, label in [
        (hit_projs, OUTCOME_COLORS["Hit"], "True Hit"),
        (miss_projs, OUTCOME_COLORS["Miss"], "Miss"),
        (fa_projs, OUTCOME_COLORS["FA"], "True FA"),
        (cr_projs, OUTCOME_COLORS["CR"], "True CR"),
    ]:
        finite = [(i, p) for i, p in zip(idxs, proj_vals) if np.isfinite(p)]
        if finite:
            x, y = zip(*finite)
            ax_e.scatter(x, y, c=color, s=50, edgecolors="white",
                         linewidths=0.5, zorder=3, label=label)
            ax_e.plot(x, y, color=color, alpha=0.3, linewidth=1, zorder=2)

    ax_e.set_xlabel("Session index")
    ax_e.set_ylabel("Mean projection onto CD")
    ax_e.set_title("E. All SDT outcomes on the decision axis")
    ax_e.legend(fontsize=8)

    # Panel F: Normalized FA and CR position on decision axis by stage
    # (X - Miss) / (Hit - Miss): 0 = Miss-like, 1 = Hit-like
    ax_f = fig.add_subplot(gs[2, 1])
    norm_fa = {s: [] for s in STAGE_ORDER}
    norm_cr = {s: [] for s in STAGE_ORDER}
    for k in sess_list:
        r = results[k]
        h = r.get("hit_cd_projection", np.nan)
        m = r.get("miss_cd_projection", np.nan)
        f = r.get("fa_cd_projection", np.nan)
        c = r.get("cr_cd_projection", np.nan)
        if np.isfinite(h) and np.isfinite(m) and abs(h - m) > 1e-6:
            if np.isfinite(f):
                norm_fa[r["stage"]].append((f - m) / (h - m))
            if np.isfinite(c):
                norm_cr[r["stage"]].append((c - m) / (h - m))

    # Paired box/scatter for FA and CR at each stage
    box_width = 0.3
    for i, stage in enumerate(STAGE_ORDER):
        fa_vals = norm_fa[stage]
        cr_vals = norm_cr[stage]
        if fa_vals:
            bp_fa = ax_f.boxplot([fa_vals], positions=[i - box_width/2],
                                widths=box_width, patch_artist=True,
                                showfliers=False)
            bp_fa["boxes"][0].set_facecolor(OUTCOME_COLORS["FA"])
            bp_fa["boxes"][0].set_alpha(0.5)
            jitter = np.random.default_rng(42).uniform(-0.06, 0.06, len(fa_vals))
            ax_f.scatter(i - box_width/2 + jitter, fa_vals,
                         c=OUTCOME_COLORS["FA"], s=30,
                         edgecolors="white", linewidths=0.5, zorder=3)
        if cr_vals:
            bp_cr = ax_f.boxplot([cr_vals], positions=[i + box_width/2],
                                widths=box_width, patch_artist=True,
                                showfliers=False)
            bp_cr["boxes"][0].set_facecolor(OUTCOME_COLORS["CR"])
            bp_cr["boxes"][0].set_alpha(0.5)
            jitter = np.random.default_rng(99).uniform(-0.06, 0.06, len(cr_vals))
            ax_f.scatter(i + box_width/2 + jitter, cr_vals,
                         c=OUTCOME_COLORS["CR"], s=30,
                         edgecolors="white", linewidths=0.5, zorder=3)

    ax_f.axhline(0, color=OUTCOME_COLORS["Miss"], linewidth=0.8, linestyle=":",
                 alpha=0.5, label="Miss level")
    ax_f.axhline(1, color=OUTCOME_COLORS["Hit"], linewidth=0.8, linestyle=":",
                 alpha=0.5, label="Hit level")
    ax_f.set_xticks(range(len(STAGE_ORDER)))
    ax_f.set_xticklabels(STAGE_ORDER)
    ax_f.set_ylabel("Normalized position\n(0=Miss, 1=Hit)")
    ax_f.set_title("F. FA & CR position on decision axis by stage")
    # Manual legend for FA / CR
    from matplotlib.patches import Patch
    ax_f.legend(handles=[
        Patch(facecolor=OUTCOME_COLORS["FA"], alpha=0.6, label="True FA"),
        Patch(facecolor=OUTCOME_COLORS["CR"], alpha=0.6, label="True CR"),
    ], fontsize=8)

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # CD magnitude trend
    finite = [(i, m) for i, m in zip(idxs, cd_mags) if np.isfinite(m)]
    if len(finite) >= 3:
        rho, p = spearmanr([x[0] for x in finite], [x[1] for x in finite])
        stats.append({"test": "cd_mag_vs_session_spearman", "rho": rho, "p": p})

    # Cosine similarity trend
    if len(cos_sims) >= 3:
        rho, p = spearmanr(cos_idxs, cos_sims)
        stats.append({"test": "cd_stability_vs_session_spearman", "rho": rho, "p": p})

    # Variance ratio by stage
    from scipy.stats import kruskal as kruskal_test
    ratio_groups = [var_df[var_df["stage"] == s]["ratio"].values for s in STAGE_ORDER]
    ratio_groups = [g for g in ratio_groups if len(g) >= 2]
    if len(ratio_groups) >= 2:
        try:
            h, p = kruskal_test(*ratio_groups)
            stats.append({"test": "var_ratio_kruskal_by_stage", "H": h, "p": p})
        except ValueError:
            pass

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig17_geometry_shift", "05_longitudinal")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "05_longitudinal", "geometry_shift_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: p={row.get('p', 'N/A')}")


if __name__ == "__main__":
    main()
