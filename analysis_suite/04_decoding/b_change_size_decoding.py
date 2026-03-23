"""Fig19: Change-Size Decoding — Big vs Small from population activity.

Cross-validated logistic regression to decode whether the change is
Big (2.0, 4.0 Hz) vs Small (1.25, 1.35, 1.5 Hz) from population
activity in the response window.

Produces:
  - Fig 19A: Time-resolved decoding accuracy (Expert grand-average)
  - Fig 19B: Peak decoding accuracy across sessions
  - Fig 19C: Decoding by stage (boxplot)
  - Fig 19D: Confusion matrix (Expert pooled)

Saves: figures/04_decoding/change_size_decoding_stats.csv
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from config import (
    STAGE_ORDER, STAGE_COLORS,
    SMALL_CHANGE_SIZES, BIG_CHANGE_SIZES,
    CACHE_DIR,
)
from loader import load_staging_manifest, load_session
from utils import get_good_cluster_ids, build_population_tensor
from plotting import setup_style, save_figure, add_stage_background

setup_style()

WINDOW = (-0.5, 1.0)
BIN_SIZE = 0.05
MIN_UNITS = 5
MIN_TRIALS_PER_CLASS = 8
N_FOLDS = 5


def decode_change_size_session(sess, sname, stage, sidx):
    """Run change-size decoding for a single session."""
    good_ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
    if len(good_ids) < MIN_UNITS:
        return None

    # Get all Hit+Miss trials
    tensor, bc, used = build_population_tensor(
        sess, good_ids, event_name="Change_ON",
        window=WINDOW, bin_size=BIN_SIZE,
        outcome_filter={"Hit", "Miss"},
    )

    if tensor.shape[0] < 2 * MIN_TRIALS_PER_CLASS:
        return None

    # Label: Big (1) vs Small (0)
    trials = sess.trials
    labels = np.full(len(used), -1, dtype=int)
    for ti, idx in enumerate(used):
        cs = getattr(trials[idx], "change_size", None)
        if cs is None:
            continue
        if any(abs(cs - s) < 0.01 for s in BIG_CHANGE_SIZES):
            labels[ti] = 1
        elif any(abs(cs - s) < 0.01 for s in SMALL_CHANGE_SIZES):
            labels[ti] = 0

    # Filter out unknown
    valid = labels >= 0
    tensor = tensor[valid]
    labels = labels[valid]

    n_big = (labels == 1).sum()
    n_small = (labels == 0).sum()
    if n_big < MIN_TRIALS_PER_CLASS or n_small < MIN_TRIALS_PER_CLASS:
        return None

    n_trials, n_bins, n_units = tensor.shape

    # Time-resolved decoding
    accs = np.full(n_bins, np.nan)
    all_y_true = []
    all_y_pred = []

    for b in range(n_bins):
        X = tensor[:, b, :]
        nan_mask = np.isnan(X).any(axis=1)
        if nan_mask.sum() > len(X) * 0.3:
            continue
        X = np.nan_to_num(X, nan=0.0)

        if len(np.unique(labels)) < 2:
            continue

        n_splits = min(N_FOLDS, n_big, n_small)
        if n_splits < 2:
            continue

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_accs = []
        for train_idx, test_idx in cv.split(X, labels):
            clf = LogisticRegression(
                C=1.0, solver="liblinear", max_iter=500, random_state=42
            )
            clf.fit(X[train_idx], labels[train_idx])
            fold_accs.append(clf.score(X[test_idx], labels[test_idx]))

            # Collect for confusion matrix (use response window only)
            if bc[b] >= 0.0 and bc[b] < 0.3:
                all_y_true.extend(labels[test_idx])
                all_y_pred.extend(clf.predict(X[test_idx]))

        accs[b] = np.mean(fold_accs)

    return {
        "bin_centers": bc,
        "accuracy": accs,
        "peak_acc": float(np.nanmax(accs)) if np.any(np.isfinite(accs)) else np.nan,
        "stage": stage,
        "session_idx": sidx,
        "n_units": n_units,
        "n_big": int(n_big),
        "n_small": int(n_small),
        "y_true": np.array(all_y_true),
        "y_pred": np.array(all_y_pred),
    }


def main():
    print("[04b] Change-size decoding...")
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

        r = decode_change_size_session(sess, sname, stage, sidx)
        if r is not None:
            results[sname] = r
            print(f"peak={r['peak_acc']:.3f}, {r['n_units']} units")
        else:
            print("insufficient data")

        del sess
        gc.collect()

    print(f"\n  Decoded {len(results)} sessions")

    if not results:
        print("  No data. Exiting.")
        return

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Time-resolved accuracy by stage
    ax_a = fig.add_subplot(gs[0, 0])
    for stage in STAGE_ORDER:
        stage_res = [v for v in results.values() if v["stage"] == stage]
        if stage_res:
            ref_bc = stage_res[0]["bin_centers"]
            all_accs = [r["accuracy"] for r in stage_res if len(r["accuracy"]) == len(ref_bc)]
            if all_accs:
                mean_acc = np.nanmean(all_accs, axis=0)
                sem_acc = np.nanstd(all_accs, axis=0) / np.sqrt(len(all_accs))
                ax_a.plot(ref_bc, mean_acc, color=STAGE_COLORS[stage],
                          linewidth=2, label=f"{stage} (n={len(all_accs)})")
                ax_a.fill_between(ref_bc, mean_acc - sem_acc, mean_acc + sem_acc,
                                  color=STAGE_COLORS[stage], alpha=0.2)
    ax_a.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax_a.axvline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_a.set_xlabel("Time from Change_ON (s)")
    ax_a.set_ylabel("Decoding accuracy")
    ax_a.set_title("A. Big vs Small change decoding")
    ax_a.legend(fontsize=8)
    ax_a.set_ylim(0.3, 1.0)

    # Panel B: Peak accuracy across sessions
    ax_b = fig.add_subplot(gs[0, 1])
    add_stage_background(ax_b, manifest)
    sess_list = sorted(results.keys(), key=lambda k: results[k]["session_idx"])
    idxs = [results[k]["session_idx"] for k in sess_list]
    peaks = [results[k]["peak_acc"] for k in sess_list]
    stages = [results[k]["stage"] for k in sess_list]
    colors = [STAGE_COLORS[s] for s in stages]

    ax_b.scatter(idxs, peaks, c=colors, s=60, edgecolors="white",
                 linewidths=0.5, zorder=3)
    ax_b.plot(idxs, peaks, color="gray", alpha=0.3, linewidth=1, zorder=2)
    ax_b.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax_b.set_xlabel("Session index")
    ax_b.set_ylabel("Peak decoding accuracy")
    ax_b.set_title("B. Change-size decoding across learning")

    # Panel C: Peak accuracy by stage (boxplot)
    ax_c = fig.add_subplot(gs[1, 0])
    stage_data = []
    stage_positions = []
    stage_colors = []
    for i, stage in enumerate(STAGE_ORDER):
        vals = [results[k]["peak_acc"] for k in results
                if results[k]["stage"] == stage and np.isfinite(results[k]["peak_acc"])]
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
    ax_c.axhline(0.5, color="gray", linewidth=0.5, linestyle=":")
    ax_c.set_xticks(range(len(STAGE_ORDER)))
    ax_c.set_xticklabels(STAGE_ORDER)
    ax_c.set_ylabel("Peak decoding accuracy")
    ax_c.set_title("C. Decoding by stage")

    # Panel D: Confusion matrix (Expert pooled)
    ax_d = fig.add_subplot(gs[1, 1])
    expert_yt = np.concatenate([r["y_true"] for r in results.values()
                                 if r["stage"] == "Expert" and len(r["y_true"]) > 0])
    expert_yp = np.concatenate([r["y_pred"] for r in results.values()
                                 if r["stage"] == "Expert" and len(r["y_pred"]) > 0])
    if len(expert_yt) > 0:
        cm = confusion_matrix(expert_yt, expert_yp, labels=[0, 1])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax_d.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for r in range(2):
            for c in range(2):
                val = cm_norm[r, c]
                color = "white" if val > 0.5 else "black"
                ax_d.text(c, r, f"{val:.2f}\n({cm[r,c]})",
                          ha="center", va="center", fontsize=10, color=color)
        ax_d.set_xticks([0, 1])
        ax_d.set_yticks([0, 1])
        ax_d.set_xticklabels(["Small", "Big"])
        ax_d.set_yticklabels(["Small", "Big"])
        ax_d.set_xlabel("Predicted")
        ax_d.set_ylabel("True")
        plt.colorbar(im, ax=ax_d, fraction=0.046)
    ax_d.set_title("D. Confusion matrix (Expert)")

    # ── Statistics ────────────────────────────────────────────────────
    stats = []
    finite_peaks = [(i, p) for i, p in zip(idxs, peaks) if np.isfinite(p)]
    if len(finite_peaks) >= 3:
        rho, p = spearmanr([x[0] for x in finite_peaks], [x[1] for x in finite_peaks])
        stats.append({"test": "peak_acc_vs_session_spearman", "rho": rho, "p": p})

    from scipy.stats import kruskal as kruskal_test
    valid_groups = []
    for stage in STAGE_ORDER:
        vals = [results[k]["peak_acc"] for k in results
                if results[k]["stage"] == stage and np.isfinite(results[k]["peak_acc"])]
        if len(vals) >= 2:
            valid_groups.append(vals)
    if len(valid_groups) >= 2:
        try:
            h, p = kruskal_test(*valid_groups)
            stats.append({"test": "peak_acc_kruskal_by_stage", "H": h, "p": p})
        except ValueError:
            pass

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig19_change_size_decoding", "04_decoding")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "04_decoding", "change_size_decoding_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: p={row.get('p', 'N/A')}")


if __name__ == "__main__":
    main()
