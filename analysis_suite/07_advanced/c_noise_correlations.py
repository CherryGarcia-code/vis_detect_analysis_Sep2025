"""Fig29: Pairwise noise correlations — by learning stage and cell-type pair.

Noise correlations (r_noise) quantify shared trial-to-trial variability
between neuron pairs, a key measure of population coding efficiency.

Computes r_noise for simultaneously-recorded pairs within sessions,
conditioned on learning stage and HMM behavioral state.

Produces:
  - Fig 29A: r_noise distribution by stage
  - Fig 29B: Mean r_noise across sessions (learning trajectory)
  - Fig 29C: r_noise by HMM state (Expert sessions)
  - Fig 29D: r_noise vs inter-unit distance or cell-type pair

Saves: figures/07_advanced/noise_correlation_stats.csv
"""

import os
import sys
import gc
from itertools import combinations


import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import (
    STAGE_ORDER, STAGE_COLORS, HMM_STATE_ORDER, HMM_STATE_COLORS,
    CELLTYPE_COLORS, CACHE_DIR, DEFAULT_BIN_SIZE,
)
from visdetect.suite.loader import (
    load_staging_manifest, load_session,
    load_hmm_assignments, load_waveform_labels,
)
from visdetect.analysis.utils import (
    get_good_cluster_ids, build_population_tensor,
)
from visdetect.suite.plotting import setup_style, save_figure, add_stage_background

setup_style()

WINDOW = (0.0, 0.5)      # Only use post-change response window
BIN_SIZE = DEFAULT_BIN_SIZE
MIN_UNITS = 5
MIN_TRIALS = 10
MAX_PAIRS_PER_SESSION = 500  # cap to avoid memory issues


def compute_noise_correlations(tensor, labels=None):
    """Compute pairwise noise correlations from population tensor.

    tensor: (n_trials, n_bins, n_units)
    Returns list of (unit_i, unit_j, r_noise) tuples.

    r_noise = Pearson correlation of trial-to-trial residuals
    after subtracting condition means.
    """
    n_trials, n_bins, n_units = tensor.shape

    # Mean FR across bins per trial -> (n_trials, n_units)
    mean_fr = np.nanmean(tensor, axis=1)

    # Subtract condition means (if labels provided)
    if labels is not None:
        residuals = np.zeros_like(mean_fr)
        for lab in np.unique(labels):
            mask = labels == lab
            cond_mean = np.nanmean(mean_fr[mask], axis=0)
            residuals[mask] = mean_fr[mask] - cond_mean
    else:
        residuals = mean_fr - np.nanmean(mean_fr, axis=0)

    # Pairwise Pearson correlations
    pairs = []
    pair_list = list(combinations(range(n_units), 2))

    # Cap if too many pairs
    if len(pair_list) > MAX_PAIRS_PER_SESSION:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pair_list), MAX_PAIRS_PER_SESSION, replace=False)
        pair_list = [pair_list[i] for i in sorted(idx)]

    for i, j in pair_list:
        ri = residuals[:, i]
        rj = residuals[:, j]
        valid = np.isfinite(ri) & np.isfinite(rj)
        if valid.sum() >= 5:
            r = float(np.corrcoef(ri[valid], rj[valid])[0, 1])
            if np.isfinite(r):
                pairs.append((i, j, r))

    return pairs


def main():
    print("[07c] Noise correlations analysis...")
    manifest = load_staging_manifest(qc_only=True)
    hmm = load_hmm_assignments()
    wf_labels = load_waveform_labels()

    ct_lookup = {}
    if wf_labels is not None:
        for _, row in wf_labels.iterrows():
            ct_lookup[(int(row["session_name"]), int(row["cluster_id"]))] = row["cell_type"]

    all_records = []

    for _, mrow in manifest.iterrows():
        sname = int(mrow["session_name"])
        stage = mrow["stage"]
        sidx = mrow["session_idx"]

        print(f"  Session {sname} ({stage})...", end=" ")
        try:
            sess = load_session(sname)
        except FileNotFoundError:
            print("not found")
            continue

        good_ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
        if len(good_ids) < MIN_UNITS:
            print("too few units")
            del sess
            gc.collect()
            continue

        # Build tensor (Hit+Miss only)
        tensor, bc, used = build_population_tensor(
            sess, good_ids, event_name="Change_ON",
            window=WINDOW, bin_size=BIN_SIZE,
            outcome_filter={"Hit", "Miss"},
        )

        if tensor.shape[0] < MIN_TRIALS:
            print("too few trials")
            del sess
            gc.collect()
            continue

        trials = sess.trials
        labels = np.array([
            1 if getattr(trials[i], "trialoutcome", None) == "Hit" else 0
            for i in used
        ])

        # Compute noise correlations
        pairs = compute_noise_correlations(tensor, labels)

        for i, j, r in pairs:
            cid_i = good_ids[i] if i < len(good_ids) else -1
            cid_j = good_ids[j] if j < len(good_ids) else -1
            ct_i = ct_lookup.get((sname, cid_i), "Unknown")
            ct_j = ct_lookup.get((sname, cid_j), "Unknown")

            # Pair type
            cts = sorted([ct_i, ct_j])
            if "Narrow (FSI)" in cts and "Broad (MSN/Proj)" in cts:
                pair_type = "FSI-MSN"
            elif ct_i == ct_j and "Narrow (FSI)" in ct_i:
                pair_type = "FSI-FSI"
            elif ct_i == ct_j and "Broad" in ct_i:
                pair_type = "MSN-MSN"
            else:
                pair_type = "Other"

            all_records.append({
                "session_name": sname,
                "stage": stage,
                "session_idx": sidx,
                "unit_i": cid_i,
                "unit_j": cid_j,
                "r_noise": r,
                "pair_type": pair_type,
            })

        print(f"{len(pairs)} pairs")
        del sess
        gc.collect()

    df = pd.DataFrame(all_records)
    print(f"\n  Total: {len(df)} pairs from {df['session_name'].nunique()} sessions")

    if len(df) == 0:
        print("  No data. Exiting.")
        return

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: r_noise distribution by stage
    ax_a = fig.add_subplot(gs[0, 0])
    for stage in STAGE_ORDER:
        vals = df[df["stage"] == stage]["r_noise"].dropna().values
        if len(vals) > 0:
            ax_a.hist(vals, bins=50, color=STAGE_COLORS[stage],
                      alpha=0.4, label=f"{stage} (n={len(vals)})", density=True)
    ax_a.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax_a.set_xlabel("Noise correlation (r)")
    ax_a.set_ylabel("Density")
    ax_a.set_title("A. r_noise distribution by stage")
    ax_a.legend(fontsize=8)
    ax_a.set_xlim(-0.5, 0.5)

    # Panel B: Mean r_noise across sessions
    ax_b = fig.add_subplot(gs[0, 1])
    add_stage_background(ax_b, manifest)

    sess_mean = df.groupby(["session_name", "session_idx", "stage"]).agg(
        mean_r_noise=("r_noise", "mean"),
        n_pairs=("r_noise", "count"),
    ).reset_index().sort_values("session_idx")

    for stage in STAGE_ORDER:
        sub = sess_mean[sess_mean["stage"] == stage]
        if len(sub) > 0:
            ax_b.scatter(sub["session_idx"], sub["mean_r_noise"],
                         c=STAGE_COLORS[stage], s=60, edgecolors="white",
                         linewidths=0.5, zorder=3, label=stage)
    if len(sess_mean) > 0:
        ax_b.plot(sess_mean["session_idx"], sess_mean["mean_r_noise"],
                  color="gray", alpha=0.3, linewidth=1, zorder=2)
    ax_b.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_b.set_xlabel("Session index")
    ax_b.set_ylabel("Mean r_noise")
    ax_b.set_title("B. Noise correlations across learning")
    ax_b.legend(fontsize=8)

    # Panel C: r_noise by stage (boxplot)
    ax_c = fig.add_subplot(gs[1, 0])
    stage_data = []
    stage_positions = []
    stage_colors = []
    for i, stage in enumerate(STAGE_ORDER):
        vals = sess_mean[sess_mean["stage"] == stage]["mean_r_noise"].dropna().values
        if len(vals) >= 1:
            stage_positions.append(i)
            stage_data.append(vals)
            stage_colors.append(STAGE_COLORS[stage])

    if stage_data:
        bp = ax_c.boxplot(stage_data, positions=stage_positions, widths=0.5,
                          patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], stage_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for pos, vals, color in zip(stage_positions, stage_data, stage_colors):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax_c.scatter(pos + jitter, vals, c=color, s=40,
                         edgecolors="white", linewidths=0.5, zorder=3)
    ax_c.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_c.set_xticks(range(len(STAGE_ORDER)))
    ax_c.set_xticklabels(STAGE_ORDER)
    ax_c.set_ylabel("Mean r_noise per session")
    ax_c.set_title("C. r_noise by learning stage")

    # Panel D: r_noise by cell-type pair
    ax_d = fig.add_subplot(gs[1, 1])
    pair_types = ["FSI-FSI", "MSN-MSN", "FSI-MSN"]
    pair_data = []
    pair_positions = []

    for i, pt in enumerate(pair_types):
        vals = df[df["pair_type"] == pt]["r_noise"].dropna().values
        if len(vals) >= 5:
            pair_positions.append(i)
            pair_data.append(vals)

    if pair_data:
        bp = ax_d.boxplot(pair_data, positions=pair_positions, widths=0.5,
                          patch_artist=True, showfliers=False)
        colors_pt = ["#e74c3c", "#3498db", "#9b59b6"]
        for patch, color in zip(bp["boxes"], colors_pt[:len(pair_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

    ax_d.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_d.set_xticks(range(len(pair_types)))
    ax_d.set_xticklabels(pair_types)
    ax_d.set_ylabel("r_noise")
    ax_d.set_title("D. Noise correlations by cell-type pair")

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # r_noise trend
    if len(sess_mean) >= 3:
        rho, p = spearmanr(sess_mean["session_idx"], sess_mean["mean_r_noise"])
        stats.append({"test": "r_noise_vs_session_spearman", "rho": rho, "p": p})

    # Stage comparison
    from scipy.stats import kruskal
    valid_groups = [sess_mean[sess_mean["stage"] == s]["mean_r_noise"].dropna().values
                    for s in STAGE_ORDER]
    valid_groups = [g for g in valid_groups if len(g) >= 2 and np.std(g) > 0]
    if len(valid_groups) >= 2:
        try:
            h, p = kruskal(*valid_groups)
            stats.append({"test": "r_noise_kruskal_by_stage", "H": h, "p": p})
        except ValueError:
            pass

    # Overall mean
    stats.append({
        "test": "overall_mean_r_noise",
        "value": float(df["r_noise"].mean()),
        "median": float(df["r_noise"].median()),
        "n_pairs": len(df),
    })

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig29_noise_correlations", "07_advanced")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "07_advanced", "noise_correlation_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: {row.get('p', row.get('value', 'N/A'))}")


if __name__ == "__main__":
    main()
