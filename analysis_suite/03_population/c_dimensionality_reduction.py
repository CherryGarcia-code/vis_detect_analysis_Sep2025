"""03c - Population dimensionality reduction (PCA).

Applies PCA to population activity tensors to visualize neural
state-space trajectories for Hit vs Miss trials and quantify
effective dimensionality changes across learning stages.

Produces:
  - Fig 12A: PC1 vs PC2 trajectories for Hit vs Miss (Expert session)
  - Fig 12B: Variance explained (scree plot) by stage
  - Fig 12C: Effective dimensionality across sessions
  - Fig 12D: PC1 temporal profile by outcome (Expert grand-average)

Saves: figures/03_population/dimensionality_stats.csv
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
from sklearn.decomposition import PCA

from config import (
    STAGE_ORDER, STAGE_COLORS, OUTCOME_COLORS, CACHE_DIR,
)
from loader import load_staging_manifest, load_session
from utils import (
    get_good_cluster_ids, build_population_tensor, smooth_psth,
    compute_zscore_normalized,
)
from plotting import setup_style, save_figure, add_stage_background

setup_style()

WINDOW = (-0.5, 1.0)
BIN_SIZE = 0.025
BASELINE_WIN = (-0.5, -0.05)
MIN_UNITS = 10
MIN_TRIALS_PER_CLASS = 5


def compute_effective_dim(eigenvalues):
    """Participation ratio: (sum(lambda))^2 / sum(lambda^2)."""
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) == 0:
        return 0
    return float((np.sum(eigenvalues)) ** 2 / np.sum(eigenvalues ** 2))


def run_pca_session(sess, sname, stage, sidx):
    """Run PCA on population tensor for a single session."""
    good_ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
    if len(good_ids) < MIN_UNITS:
        return None

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

    tensor, bc, used = build_population_tensor(
        sess, good_ids, event_name="Change_ON",
        window=WINDOW, bin_size=BIN_SIZE,
        trial_indices=go_hit_idx + go_miss_idx,
    )
    if tensor.shape[0] < 2 * MIN_TRIALS_PER_CLASS or tensor.shape[2] < MIN_UNITS:
        return None

    labels = np.array([
        1 if getattr(trials[i], "trialoutcome", None) == "Hit" else 0
        for i in used
    ])

    n_hit = labels.sum()
    n_miss = (~labels.astype(bool)).sum()
    if n_hit < MIN_TRIALS_PER_CLASS or n_miss < MIN_TRIALS_PER_CLASS:
        return None

    # Z-score
    z_tensor = compute_zscore_normalized(tensor, bc, BASELINE_WIN)

    # Condition-averaged PSTHs: (n_bins, n_units)
    hit_mean = np.nanmean(z_tensor[labels == 1], axis=0)  # (n_bins, n_units)
    miss_mean = np.nanmean(z_tensor[labels == 0], axis=0)

    # PCA on concatenated condition means
    data = np.vstack([hit_mean, miss_mean])  # (2*n_bins, n_units)
    n_components = min(10, data.shape[1], data.shape[0])
    pca = PCA(n_components=n_components)
    pca.fit(data)

    # Project condition means
    hit_pc = pca.transform(hit_mean)
    miss_pc = pca.transform(miss_mean)

    # FA trajectory (project catch-trial FA mean onto same PCA)
    fa_pc = None
    n_fa = 0
    if len(fa_idx) >= 3:
        fa_tensor, _, _ = build_population_tensor(
            sess, good_ids, event_name="Change_ON",
            window=WINDOW, bin_size=BIN_SIZE,
            trial_indices=fa_idx,
        )
        if fa_tensor.shape[0] >= 3:
            fa_z = compute_zscore_normalized(fa_tensor, bc, BASELINE_WIN)
            fa_mean = np.nanmean(fa_z, axis=0)  # (n_bins, n_units)
            fa_pc = pca.transform(fa_mean)
            n_fa = fa_tensor.shape[0]

    # Effective dimensionality from all trials
    # Reshape tensor: (n_trials * n_bins, n_units)
    n_trials, n_bins, n_units = z_tensor.shape
    flat = z_tensor.reshape(n_trials * n_bins, n_units)
    # Remove NaN rows
    valid_mask = ~np.isnan(flat).any(axis=1)
    flat_clean = flat[valid_mask]

    if len(flat_clean) > n_units:
        pca_full = PCA(n_components=min(n_units, len(flat_clean)))
        pca_full.fit(flat_clean)
        eff_dim = compute_effective_dim(pca_full.explained_variance_)
        var_explained = pca_full.explained_variance_ratio_
    else:
        eff_dim = np.nan
        var_explained = pca.explained_variance_ratio_

    return {
        "bin_centers": bc,
        "hit_pc": hit_pc,
        "miss_pc": miss_pc,
        "fa_pc": fa_pc,
        "var_explained": var_explained,
        "effective_dim": eff_dim,
        "n_units": n_units,
        "n_hit": int(n_hit),
        "n_miss": int(n_miss),
        "n_fa": n_fa,
        "stage": stage,
        "session_idx": sidx,
    }


def main():
    print("[03c] Population dimensionality reduction (PCA)...")
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

        r = run_pca_session(sess, sname, stage, sidx)
        if r is not None:
            results[sname] = r
            print(f"eff_dim={r['effective_dim']:.1f}, {r['n_units']} units")
        else:
            print("insufficient data")

        del sess
        gc.collect()

    print(f"\n  PCA computed for {len(results)} sessions")

    if not results:
        print("  No data. Exiting.")
        return

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    expert = {k: v for k, v in results.items() if v["stage"] == "Expert"}

    # Panel A: PC1 vs PC2 trajectory (best Expert session)
    ax_a = fig.add_subplot(gs[0, 0])
    if expert:
        best = max(expert.keys(), key=lambda k: expert[k]["n_units"])
        r = expert[best]
        bc = r["bin_centers"]

        hit_pc = r["hit_pc"]
        miss_pc = r["miss_pc"]

        # Color-code by time
        t_colors = plt.cm.viridis(np.linspace(0, 1, len(bc)))

        ax_a.plot(hit_pc[:, 0], hit_pc[:, 1],
                  color=OUTCOME_COLORS["Hit"], linewidth=2, label="Hit")
        ax_a.plot(miss_pc[:, 0], miss_pc[:, 1],
                  color=OUTCOME_COLORS["Miss"], linewidth=2, label="Miss")

        # Mark stimulus onset
        onset_idx = np.argmin(np.abs(bc))
        ax_a.scatter(hit_pc[onset_idx, 0], hit_pc[onset_idx, 1],
                     c="k", s=100, marker="*", zorder=5, label="Change_ON")
        ax_a.scatter(miss_pc[onset_idx, 0], miss_pc[onset_idx, 1],
                     c="k", s=100, marker="*", zorder=5)

        # Mark start
        ax_a.scatter(hit_pc[0, 0], hit_pc[0, 1], c=OUTCOME_COLORS["Hit"],
                     s=80, marker="o", edgecolors="k", zorder=5)
        ax_a.scatter(miss_pc[0, 0], miss_pc[0, 1], c=OUTCOME_COLORS["Miss"],
                     s=80, marker="o", edgecolors="k", zorder=5)

        # FA trajectory
        if r.get("fa_pc") is not None:
            fa_pc = r["fa_pc"]
            ax_a.plot(fa_pc[:, 0], fa_pc[:, 1],
                      color=OUTCOME_COLORS["FA"], linewidth=2,
                      label=f"FA (n={r['n_fa']})")
            ax_a.scatter(fa_pc[onset_idx, 0], fa_pc[onset_idx, 1],
                         c="k", s=100, marker="*", zorder=5)
            ax_a.scatter(fa_pc[0, 0], fa_pc[0, 1], c=OUTCOME_COLORS["FA"],
                         s=80, marker="o", edgecolors="k", zorder=5)

        ax_a.set_xlabel("PC1")
        ax_a.set_ylabel("PC2")
        ax_a.set_title(f"A. Neural trajectory - Expert {best} (n={r['n_units']})")
        ax_a.legend(fontsize=8)

    # Panel B: Variance explained by stage
    ax_b = fig.add_subplot(gs[0, 1])
    for stage in STAGE_ORDER:
        stage_results = [v for v in results.values() if v["stage"] == stage]
        if stage_results:
            all_var = [r["var_explained"] for r in stage_results]
            # Pad to same length
            max_len = max(len(v) for v in all_var)
            padded = [np.pad(v, (0, max_len - len(v)), constant_values=0)
                      for v in all_var]
            mean_var = np.mean(padded, axis=0)
            sem_var = np.std(padded, axis=0) / np.sqrt(len(padded))

            pcs = np.arange(1, max_len + 1)
            ax_b.plot(pcs, np.cumsum(mean_var), "o-",
                      color=STAGE_COLORS[stage], label=stage,
                      linewidth=2, markersize=4)

    ax_b.axhline(0.9, color="gray", linestyle=":", linewidth=0.5)
    ax_b.set_xlabel("Number of PCs")
    ax_b.set_ylabel("Cumulative variance explained")
    ax_b.set_title("B. Variance explained by stage")
    ax_b.legend(fontsize=8)
    ax_b.set_ylim(0, 1.05)

    # Panel C: Effective dimensionality across sessions
    ax_c = fig.add_subplot(gs[1, 0])
    add_stage_background(ax_c, manifest)

    sess_list = sorted(results.keys(), key=lambda k: results[k]["session_idx"])
    idxs = [results[k]["session_idx"] for k in sess_list]
    eff_dims = [results[k]["effective_dim"] for k in sess_list]
    stages = [results[k]["stage"] for k in sess_list]
    colors = [STAGE_COLORS[s] for s in stages]

    ax_c.scatter(idxs, eff_dims, c=colors, s=60, edgecolors="white",
                 linewidths=0.5, zorder=3)
    ax_c.plot(idxs, eff_dims, color="gray", alpha=0.3, linewidth=1, zorder=2)
    ax_c.set_xlabel("Session index")
    ax_c.set_ylabel("Effective dimensionality")
    ax_c.set_title("C. Neural dimensionality across learning")

    # Panel D: PC1 temporal profiles (Expert grand-average)
    ax_d = fig.add_subplot(gs[1, 1])
    if expert:
        ref_bc = list(expert.values())[0]["bin_centers"]
        all_hit_pc1 = []
        all_miss_pc1 = []
        for r in expert.values():
            if len(r["hit_pc"]) == len(ref_bc):
                all_hit_pc1.append(r["hit_pc"][:, 0])
                all_miss_pc1.append(r["miss_pc"][:, 0])

        if all_hit_pc1:
            hit_mean = np.mean(all_hit_pc1, axis=0)
            hit_sem = np.std(all_hit_pc1, axis=0) / np.sqrt(len(all_hit_pc1))
            miss_mean = np.mean(all_miss_pc1, axis=0)
            miss_sem = np.std(all_miss_pc1, axis=0) / np.sqrt(len(all_miss_pc1))

            ax_d.plot(ref_bc, smooth_psth(hit_mean, BIN_SIZE, 15.0),
                      color=OUTCOME_COLORS["Hit"], linewidth=2, label="Hit")
            ax_d.fill_between(ref_bc,
                              smooth_psth(hit_mean - hit_sem, BIN_SIZE, 15.0),
                              smooth_psth(hit_mean + hit_sem, BIN_SIZE, 15.0),
                              color=OUTCOME_COLORS["Hit"], alpha=0.2)
            ax_d.plot(ref_bc, smooth_psth(miss_mean, BIN_SIZE, 15.0),
                      color=OUTCOME_COLORS["Miss"], linewidth=2, label="Miss")
            ax_d.fill_between(ref_bc,
                              smooth_psth(miss_mean - miss_sem, BIN_SIZE, 15.0),
                              smooth_psth(miss_mean + miss_sem, BIN_SIZE, 15.0),
                              color=OUTCOME_COLORS["Miss"], alpha=0.2)

            # FA PC1 trajectory
            all_fa_pc1 = []
            for r in expert.values():
                if r.get("fa_pc") is not None and len(r["fa_pc"]) == len(ref_bc):
                    all_fa_pc1.append(r["fa_pc"][:, 0])
            if all_fa_pc1:
                fa_mean_pc1 = np.mean(all_fa_pc1, axis=0)
                fa_sem_pc1 = np.std(all_fa_pc1, axis=0) / np.sqrt(len(all_fa_pc1))
                ax_d.plot(ref_bc, smooth_psth(fa_mean_pc1, BIN_SIZE, 15.0),
                          color=OUTCOME_COLORS["FA"], linewidth=2,
                          label=f"FA (n={len(all_fa_pc1)})")
                ax_d.fill_between(ref_bc,
                                  smooth_psth(fa_mean_pc1 - fa_sem_pc1, BIN_SIZE, 15.0),
                                  smooth_psth(fa_mean_pc1 + fa_sem_pc1, BIN_SIZE, 15.0),
                                  color=OUTCOME_COLORS["FA"], alpha=0.2)

            ax_d.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
            ax_d.set_title(f"D. PC1 over time (n={len(all_hit_pc1)} Expert)")
        else:
            ax_d.set_title("D. PC1 over time")

    ax_d.set_xlabel("Time from Change_ON (s)")
    ax_d.set_ylabel("PC1 projection")
    ax_d.legend(fontsize=8)

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # Effective dimensionality trend
    finite_mask = [np.isfinite(d) for d in eff_dims]
    if sum(finite_mask) >= 3:
        x = [i for i, m in zip(idxs, finite_mask) if m]
        y = [d for d, m in zip(eff_dims, finite_mask) if m]
        rho, p = spearmanr(x, y)
        stats.append({"test": "eff_dim_vs_session_spearman", "rho": rho, "p": p})

    # By stage
    from scipy.stats import kruskal as kruskal_test
    stage_dims = {s: [] for s in STAGE_ORDER}
    for k in sess_list:
        if np.isfinite(results[k]["effective_dim"]):
            stage_dims[results[k]["stage"]].append(results[k]["effective_dim"])

    valid_groups = [np.array(stage_dims[s]) for s in STAGE_ORDER
                    if len(stage_dims[s]) >= 2 and np.std(stage_dims[s]) > 0]
    if len(valid_groups) >= 2:
        try:
            h, p = kruskal_test(*valid_groups)
            stats.append({"test": "eff_dim_kruskal_by_stage", "H": h, "p": p})
        except ValueError:
            pass

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig11_pca_dimensionality", "03_population")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "03_population", "dimensionality_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: p={row.get('p', 'N/A')}")


if __name__ == "__main__":
    main()
