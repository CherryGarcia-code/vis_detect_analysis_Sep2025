"""Fig20: State Decoding — HMM behavioral state from pre-trial activity.

Decode HMM behavioral state (Disengaged / Engaged / Impulsive) from
pre-trial population firing rates using multinomial logistic regression
with stratified 5-fold CV.

Pre-trial window: mean firing rate per unit in [-1.5, -0.5) relative to
Change_ON captures tonic activity before the visual change.

Produces:
  - Fig 20A: Decoding accuracy across sessions (scatter + stage background)
  - Fig 20B: Accuracy by learning stage (boxplot)
  - Fig 20C: Confusion matrix (Expert sessions, pooled across folds)
  - Fig 20D: Feature importance (histogram of mean |LR coefficients|)

Saves: figures/04_decoding/state_decoding_stats.csv
"""

import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from config import (
    STAGE_ORDER, STAGE_COLORS, HMM_STATE_ORDER, HMM_STATE_COLORS, CACHE_DIR, DEFAULT_BIN_SIZE,
)
from loader import load_staging_manifest, load_session, load_hmm_assignments
from utils import (
    get_good_cluster_ids, build_population_tensor,
    compute_baseline_subtracted
)
from plotting import setup_style, save_figure, add_stage_background

setup_style()

# ── Parameters ────────────────────────────────────────────────────────
TENSOR_WINDOW = (-1.5, 0.0)   # full pre-trial window for tensor
BASELINE_WINDOW = (-1.5, -1.0)  # Baseline for normalization (early pre-trial)
FEATURE_WIN = (-1.5, -0.5)    # sub-window for mean FR feature extraction
BIN_SIZE = DEFAULT_BIN_SIZE
MIN_UNITS = 5
MIN_TRIALS_PER_STATE = 5
N_FOLDS = 5
CHANCE_LEVEL = 1.0 / len(HMM_STATE_ORDER)  # 0.333...

# Label mapping: HMM_STATE_ORDER index -> integer label
STATE_TO_LABEL = {s: i for i, s in enumerate(HMM_STATE_ORDER)}


def decode_state_session(sess, sname, trial_states, stage, sidx):
    """Run state decoding for a single session.

    Parameters
    ----------
    sess : Session
        Loaded session object.
    sname : int
        Session name (DDMMYYYY).
    trial_states : dict
        {trial_idx: hmm_state_label} for this session.
    stage : str
        Learning stage.
    sidx : int
        Session index.

    Returns
    -------
    dict or None
        Decoding results for this session.
    """
    good_ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
    if len(good_ids) < MIN_UNITS:
        return None

    # Trial indices with HMM assignments
    hmm_trial_list = sorted(trial_states.keys())
    if len(hmm_trial_list) < len(HMM_STATE_ORDER) * MIN_TRIALS_PER_STATE:
        return None

    # Build population tensor aligned to Baseline_ON over pre-trial window.
    # NOTE: We use Baseline_ON (not Change_ON) because HMM state labels
    # are assigned to ALL trial types, including FA/abort where the change
    # stimulus was never presented.
    tensor, bin_centers, used = build_population_tensor(
        sess, good_ids, event_name="Baseline_ON",
        window=TENSOR_WINDOW, bin_size=BIN_SIZE,
        trial_indices=hmm_trial_list,
    )

    if tensor.shape[0] == 0 or tensor.shape[2] < MIN_UNITS:
        return None

    # Normalize to early pre-trial baseline (removes baseline rate confounds)
    tensor = compute_baseline_subtracted(tensor, bin_centers, BASELINE_WINDOW)

    # Map used trial indices to HMM state labels -> integer labels
    labels = np.full(len(used), -1, dtype=int)
    for ti, trial_idx in enumerate(used):
        state_label = trial_states.get(trial_idx)
        if state_label is not None and state_label in STATE_TO_LABEL:
            labels[ti] = STATE_TO_LABEL[state_label]

    # Remove trials without valid labels
    valid_mask = labels >= 0
    tensor = tensor[valid_mask]
    labels = labels[valid_mask]

    if len(labels) == 0:
        return None

    # Check minimum trials per state
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        return None
    for lbl in unique_labels:
        if counts[unique_labels == lbl][0] < MIN_TRIALS_PER_STATE:
            return None

    # ── Feature extraction: mean FR in FEATURE_WIN ────────────────────
    feat_mask = (bin_centers >= FEATURE_WIN[0]) & (bin_centers < FEATURE_WIN[1])
    if feat_mask.sum() == 0:
        return None

    # X: (n_trials, n_units) -- mean firing rate per unit in feature window
    X = np.nanmean(tensor[:, feat_mask, :], axis=1)
    X = np.nan_to_num(X, nan=0.0)

    n_trials, n_units = X.shape

    # ── Cross-validated classification ────────────────────────────────
    n_splits = min(N_FOLDS, int(np.min(counts)))
    if n_splits < 2:
        return None

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accs = []
    all_y_true = []
    all_y_pred = []
    coef_abs_list = []

    for train_idx, test_idx in cv.split(X, labels):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs",
            multi_class="multinomial", max_iter=500, random_state=42,
        )
        clf.fit(X_train, labels[train_idx])

        fold_accs.append(clf.score(X_test, labels[test_idx]))
        all_y_true.extend(labels[test_idx])
        all_y_pred.extend(clf.predict(X_test))

        # Feature importance: mean |coefficient| across classes
        coef_abs_list.append(np.mean(np.abs(clf.coef_), axis=0))  # (n_units,)

    accuracy = float(np.mean(fold_accs))
    accuracy_sem = float(np.std(fold_accs) / np.sqrt(len(fold_accs)))

    # Average absolute coefficients across folds -> (n_units,)
    mean_abs_coef = np.mean(np.array(coef_abs_list), axis=0)

    return {
        "accuracy": accuracy,
        "accuracy_sem": accuracy_sem,
        "stage": stage,
        "session_idx": sidx,
        "n_units": n_units,
        "n_trials": n_trials,
        "n_per_state": {HMM_STATE_ORDER[lbl]: int(c)
                        for lbl, c in zip(unique_labels, counts)},
        "y_true": np.array(all_y_true),
        "y_pred": np.array(all_y_pred),
        "mean_abs_coef": mean_abs_coef,
    }


def main():
    print("[04c] HMM state decoding from pre-trial activity...")
    manifest = load_staging_manifest(qc_only=True)

    # Load HMM trial-level assignments
    hmm = load_hmm_assignments()
    if hmm is None or len(hmm) == 0:
        print("  No HMM assignments. Exiting.")
        return

    # ── Decode each session ───────────────────────────────────────────
    results = {}
    for _, row in manifest.iterrows():
        sname = int(row["session_name"])
        stage = row["stage"]
        sidx = row["session_idx"]

        # Get trial-level HMM assignments for this session
        sess_hmm = hmm[hmm["session_name"] == sname]
        if len(sess_hmm) == 0:
            continue

        # Build trial-to-state lookup
        trial_states = {}
        if "trial_idx" in sess_hmm.columns and "hmm_state_label" in sess_hmm.columns:
            for _, hr in sess_hmm.iterrows():
                trial_states[int(hr["trial_idx"])] = hr["hmm_state_label"]

        if not trial_states:
            continue

        print(f"  Session {sname} ({stage})...", end=" ")
        try:
            sess = load_session(sname)
        except FileNotFoundError:
            print("not found")
            continue

        r = decode_state_session(sess, sname, trial_states, stage, sidx)
        if r is not None:
            results[sname] = r
            print(f"acc={r['accuracy']:.3f}, {r['n_units']} units, "
                  f"{r['n_trials']} trials")
        else:
            print("insufficient data")

        del sess
        gc.collect()

    print(f"\n  Decoded {len(results)} sessions")

    if not results:
        print("  No results. Exiting.")
        return

    # ── Collect data for plotting ─────────────────────────────────────
    sess_list = sorted(results.keys(), key=lambda k: results[k]["session_idx"])
    idxs = [results[k]["session_idx"] for k in sess_list]
    accuracies = [results[k]["accuracy"] for k in sess_list]
    stages = [results[k]["stage"] for k in sess_list]
    colors = [STAGE_COLORS[s] for s in stages]

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Accuracy across sessions
    ax_a = fig.add_subplot(gs[0, 0])
    add_stage_background(ax_a, manifest)
    ax_a.scatter(idxs, accuracies, c=colors, s=60, edgecolors="white",
                 linewidths=0.5, zorder=3)
    ax_a.plot(idxs, accuracies, color="gray", alpha=0.3, linewidth=1, zorder=2)
    ax_a.axhline(CHANCE_LEVEL, color="gray", linestyle="--", linewidth=0.8,
                 label=f"Chance ({CHANCE_LEVEL:.2f})")
    ax_a.set_xlabel("Session index")
    ax_a.set_ylabel("Decoding accuracy")
    ax_a.set_title("A. State decoding accuracy across learning")
    ax_a.legend(fontsize=8)
    ax_a.set_ylim(0, 1.05)

    # Panel B: Accuracy by stage (boxplot)
    ax_b = fig.add_subplot(gs[0, 1])
    stage_accs = {s: [] for s in STAGE_ORDER}
    for k in sess_list:
        stage_accs[results[k]["stage"]].append(results[k]["accuracy"])

    box_data = []
    box_pos = []
    box_colors = []
    for i, stage in enumerate(STAGE_ORDER):
        if stage_accs[stage]:
            box_pos.append(i)
            box_data.append(stage_accs[stage])
            box_colors.append(STAGE_COLORS[stage])

    if box_data:
        bp = ax_b.boxplot(box_data, positions=box_pos, widths=0.5,
                          patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for pos, vals, color in zip(box_pos, box_data, box_colors):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax_b.scatter(pos + jitter, vals, c=color, s=40,
                         edgecolors="white", linewidths=0.5, zorder=3)
    ax_b.axhline(CHANCE_LEVEL, color="gray", linewidth=0.8, linestyle="--",
                 label=f"Chance ({CHANCE_LEVEL:.2f})")
    ax_b.set_xticks(range(len(STAGE_ORDER)))
    ax_b.set_xticklabels(STAGE_ORDER)
    ax_b.set_ylabel("Decoding accuracy")
    ax_b.set_title("B. Decoding accuracy by stage")
    ax_b.legend(fontsize=8)
    ax_b.set_ylim(0, 1.05)

    # Panel C: Confusion matrix (Expert sessions, pooled)
    ax_c = fig.add_subplot(gs[1, 0])
    expert_yt = [r["y_true"] for r in results.values()
                 if r["stage"] == "Expert" and len(r["y_true"]) > 0]
    expert_yp = [r["y_pred"] for r in results.values()
                 if r["stage"] == "Expert" and len(r["y_pred"]) > 0]

    if expert_yt:
        all_yt = np.concatenate(expert_yt)
        all_yp = np.concatenate(expert_yp)
        n_states = len(HMM_STATE_ORDER)
        cm = confusion_matrix(all_yt, all_yp, labels=list(range(n_states)))
        row_sums = cm.sum(axis=1, keepdims=True)
        # Avoid division by zero for states with no true labels
        row_sums[row_sums == 0] = 1
        cm_norm = cm.astype(float) / row_sums

        im = ax_c.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for r in range(n_states):
            for c in range(n_states):
                val = cm_norm[r, c]
                text_color = "white" if val > 0.5 else "black"
                ax_c.text(c, r, f"{val:.2f}\n({cm[r, c]})",
                          ha="center", va="center", fontsize=10, color=text_color)
        ax_c.set_xticks(range(n_states))
        ax_c.set_yticks(range(n_states))
        ax_c.set_xticklabels(HMM_STATE_ORDER, rotation=30, ha="right")
        ax_c.set_yticklabels(HMM_STATE_ORDER)
        ax_c.set_xlabel("Predicted")
        ax_c.set_ylabel("True")
        plt.colorbar(im, ax=ax_c, fraction=0.046)
    ax_c.set_title("C. Confusion matrix (Expert)")

    # Panel D: Feature importance histogram
    ax_d = fig.add_subplot(gs[1, 1])
    all_coefs = np.concatenate([r["mean_abs_coef"] for r in results.values()])
    if len(all_coefs) > 0:
        ax_d.hist(all_coefs, bins=40, color="#78909C", edgecolor="white",
                  linewidth=0.5, alpha=0.8)
        median_coef = float(np.median(all_coefs))
        ax_d.axvline(median_coef, color="#E53935", linestyle="-", linewidth=1.5,
                     label=f"Median={median_coef:.4f}")
        ax_d.legend(fontsize=8)
    ax_d.set_xlabel("Mean |LR coefficient|")
    ax_d.set_ylabel("Number of units")
    ax_d.set_title(f"D. Feature importance (n={len(all_coefs)} units)")

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # 1. Spearman: accuracy vs session_idx
    finite_pairs = [(i, a) for i, a in zip(idxs, accuracies) if np.isfinite(a)]
    if len(finite_pairs) >= 3:
        x_vals, y_vals = zip(*finite_pairs)
        rho, p = spearmanr(x_vals, y_vals)
        stats.append({"test": "acc_vs_session_spearman", "rho": rho, "p": p,
                       "n": len(finite_pairs)})

    # 2. Kruskal-Wallis: accuracy by stage
    valid_groups = [np.array(stage_accs[s]) for s in STAGE_ORDER if stage_accs[s]]
    valid_groups = [g for g in valid_groups if len(g) >= 2]
    if len(valid_groups) >= 2:
        try:
            h, p = kruskal(*valid_groups)
            stats.append({"test": "acc_kruskal_by_stage", "H": h, "p": p})
        except ValueError:
            pass

    # 3. Wilcoxon signed-rank: accuracy > chance (1/3)
    acc_arr = np.array(accuracies)
    shifted = acc_arr - CHANCE_LEVEL
    if len(shifted) >= 10:
        try:
            w, p = wilcoxon(shifted, alternative="greater")
            stats.append({"test": "acc_above_chance_wilcoxon", "W": w, "p": p,
                           "median_acc": float(np.median(acc_arr)),
                           "chance": CHANCE_LEVEL})
        except ValueError:
            pass

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig20_state_decoding", "04_decoding")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "04_decoding", "state_decoding_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: p={row.get('p', 'N/A')}")


if __name__ == "__main__":
    main()
