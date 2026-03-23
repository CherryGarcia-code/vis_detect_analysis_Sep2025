"""Fig31: Comprehensive single-trial outcome prediction — targets, feature sets, and permutation null.

Scientific question:
  Can we reliably predict on a trial-by-trial basis what the behavioral
  outcome will be?  Which feature sets (behavioral state, sensory context,
  pre-trial neural activity, or their combination) yield the best
  single-trial prediction?

Prediction targets:
  A. Lick vs No-lick (all trials) -- already addressed in 07d
  B. Hit vs Miss (go trials only) -- can neural state predict detection?
  C. Hit vs FA (lick trials only) -- sensory vs impulsive lick?
  D. Three-way: Hit vs Miss vs FA

Feature sets tested per target:
  1. Behavioral state only:   P(Impulsive), P(Engaged), P(Disengaged)
  2. Sensory only:            log2(change_size)
  3. Trial history:           FA lags 1-3, rolling FA rate, prev_licked
  4. Neural only:             Pre-trial population FR per unit
  5. Behavioral + sensory:    1 + 2
  6. Full behavioral:         1 + 2 + 3
  7. Neural + behavioral:     4 + 1
  8. All features:            1 + 2 + 3 + 4

Includes permutation null (200 label shuffles) for each session/target.

Produces:
  - Fig 31A: Summary heatmap (target x feature-set, mean AUC across sessions)
  - Fig 31B: Best-model AUC per session across learning (Hit vs Miss)
  - Fig 31C: Best-model AUC per session across learning (Hit vs FA)
  - Fig 31D: Feature importance (top 20 units + behavioral) for best model
  - Fig 31E: Permutation null distributions for key comparisons
  - Fig 31F: Stage comparison -- does prediction improve with learning?

Saves:
  figures/07_advanced/fig31_trial_outcome_prediction.png
  figures/07_advanced/trial_outcome_prediction_stats.csv
  cache/trial_outcome_prediction.csv
"""

import os
import sys
import gc
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu, sem as sp_sem

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from config import (
    STAGE_ORDER, STAGE_COLORS, CACHE_DIR,
    HMM_STATE_COLORS, DEFAULT_BIN_SIZE,
)
from loader import (
    load_staging_manifest, load_session,
    load_hmm_assignments,
)
from utils import get_good_cluster_ids, build_population_tensor
from plotting import setup_style, save_figure, add_stage_background

setup_style()
warnings.filterwarnings("ignore")

# ── Parameters ────────────────────────────────────────────────────────
PRE_TRIAL_WINDOW = (-1.5, -0.5)
BIN_SIZE = DEFAULT_BIN_SIZE
MIN_UNITS = 5
MIN_TRIALS_PER_CLASS = 10
N_FOLDS = 5
N_PERM = 200
FA_HISTORY_LAGS = 3
ROLLING_FA_WINDOW = 15
RANDOM_STATE = 42


# =====================================================================
# Helper: run cross-validated prediction for one feature set
# =====================================================================
def cv_predict(X, y, n_folds=N_FOLDS, seed=RANDOM_STATE):
    """Stratified CV logistic regression. Returns mean AUC (OVR for >2 classes)."""
    if X.shape[0] < 2 * MIN_TRIALS_PER_CLASS:
        return np.nan, np.nan, None

    n_classes = len(np.unique(y))
    min_class = min(np.bincount(y))
    if min_class < MIN_TRIALS_PER_CLASS:
        return np.nan, np.nan, None

    n_splits = min(n_folds, min_class)
    if n_splits < 2:
        return np.nan, np.nan, None

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aucs = []

    for train_idx, test_idx in cv.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])

        if n_classes == 2:
            clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                     max_iter=500, random_state=seed)
            clf.fit(X_tr, y[train_idx])
            prob = clf.predict_proba(X_te)[:, 1]
            try:
                fold_aucs.append(roc_auc_score(y[test_idx], prob))
            except ValueError:
                pass
        else:
            clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                     max_iter=500, random_state=seed,
                                     multi_class="multinomial")
            clf.fit(X_tr, y[train_idx])
            prob = clf.predict_proba(X_te)
            try:
                fold_aucs.append(roc_auc_score(y[test_idx], prob,
                                                multi_class="ovr", average="macro"))
            except ValueError:
                pass

    if not fold_aucs:
        return np.nan, np.nan, None

    return np.mean(fold_aucs), np.std(fold_aucs) / np.sqrt(len(fold_aucs)), fold_aucs


def permutation_auc(X, y, n_perm=N_PERM, seed=RANDOM_STATE):
    """Label-shuffle null distribution for AUC."""
    rng = np.random.default_rng(seed)
    null_aucs = []
    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        auc, _, _ = cv_predict(X, y_shuf, seed=seed + i + 1)
        if not np.isnan(auc):
            null_aucs.append(auc)
    return np.array(null_aucs)


# =====================================================================
# Build enriched trial table from cache + HMM
# =====================================================================
def load_trial_table():
    """Load the cached impulsivity trial table and add HMM state posteriors."""
    cache_path = os.path.join(CACHE_DIR, "impulsivity_trial_table.csv")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Cached trial table not found at {cache_path}. "
            "Run d_impulsivity_regression.py first."
        )
    df = pd.read_csv(cache_path)

    # Also load full HMM assignments for all 3 state posteriors
    hmm = load_hmm_assignments()
    # Merge p_state_0, p_state_1, p_state_2
    state_cols = [c for c in hmm.columns if c.startswith("p_state_")]
    merge_cols = ["session_name", "trial_idx"] + state_cols
    hmm_sub = hmm[merge_cols].drop_duplicates()
    df = df.merge(hmm_sub, on=["session_name", "trial_idx"], how="left")

    return df


# =====================================================================
# Per-session prediction across targets and feature sets
# =====================================================================
def predict_session(sess, session_name, trial_sub, stage, session_idx):
    """Run all prediction targets x feature sets for one session.

    Returns list of result dicts.
    """
    good_ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
    n_units = len(good_ids)
    if n_units < MIN_UNITS:
        return []

    # Get trial indices we have data for
    trial_indices = sorted(trial_sub["trial_idx"].unique())

    # Build pre-trial population tensor aligned to Baseline_ON.
    # Using Baseline_ON as anchor for ALL trial types (Hit, Miss, FA) ensures
    # FA trials are included — FA trials never reach Change_ON and would be
    # silently excluded if aligned to that event.
    tensor, bin_centers, used = build_population_tensor(
        sess, good_ids, event_name="Baseline_ON",
        window=PRE_TRIAL_WINDOW, bin_size=BIN_SIZE,
        trial_indices=trial_indices,
    )
    if tensor.shape[0] < 2 * MIN_TRIALS_PER_CLASS or tensor.shape[2] < MIN_UNITS:
        return []

    # Mean FR per unit across pre-trial window -> (n_trials, n_units)
    X_neural_full = np.nanmean(tensor, axis=1)
    X_neural_full = np.nan_to_num(X_neural_full, nan=0.0)

    # Match trial_sub to used indices
    used_set = set(used)
    trial_matched = trial_sub[trial_sub["trial_idx"].isin(used_set)].copy()
    idx_map = {idx: i for i, idx in enumerate(used)}
    trial_matched["_row"] = trial_matched["trial_idx"].map(idx_map)
    trial_matched = trial_matched.dropna(subset=["_row"]).sort_values("_row")

    tensor_rows = trial_matched["_row"].astype(int).values
    X_neural = X_neural_full[tensor_rows]

    # Behavioral feature arrays
    hmm_cols = [c for c in trial_matched.columns if c.startswith("p_state_")]
    X_behav_state = trial_matched[hmm_cols].values if hmm_cols else None

    X_sensory = trial_matched[["change_size_log2"]].values

    hist_cols = (["rolling_fa_rate"] +
                 [f"prev_fa_{k}" for k in range(1, FA_HISTORY_LAGS + 1)])
    hist_cols = [c for c in hist_cols if c in trial_matched.columns]
    X_history = trial_matched[hist_cols].values if hist_cols else None

    # Fill NaN for behavioral features
    for arr in [X_behav_state, X_sensory, X_history]:
        if arr is not None:
            arr[~np.isfinite(arr)] = 0.0

    # ── Define prediction targets ─────────────────────────────────────
    outcomes = trial_matched["outcome"].values
    is_hit = outcomes == "hit"
    is_miss = outcomes == "miss"
    is_fa = outcomes == "fa"

    targets = {}

    # B. Hit vs Miss (go trials only)
    go_mask = is_hit | is_miss
    if go_mask.sum() >= 2 * MIN_TRIALS_PER_CLASS:
        y_hm = is_hit[go_mask].astype(int)
        if y_hm.sum() >= MIN_TRIALS_PER_CLASS and (len(y_hm) - y_hm.sum()) >= MIN_TRIALS_PER_CLASS:
            targets["Hit_vs_Miss"] = (go_mask, y_hm)

    # C. Hit vs FA (lick trials only)
    lick_mask = is_hit | is_fa
    if lick_mask.sum() >= 2 * MIN_TRIALS_PER_CLASS:
        y_hf = is_hit[lick_mask].astype(int)
        if y_hf.sum() >= MIN_TRIALS_PER_CLASS and (len(y_hf) - y_hf.sum()) >= MIN_TRIALS_PER_CLASS:
            targets["Hit_vs_FA"] = (lick_mask, y_hf)

    # D. Three-way: Hit=0, Miss=1, FA=2
    tri_mask = is_hit | is_miss | is_fa
    if tri_mask.sum() >= 3 * MIN_TRIALS_PER_CLASS:
        y_tri = np.zeros(tri_mask.sum(), dtype=int)
        y_tri[is_miss[tri_mask]] = 1
        y_tri[is_fa[tri_mask]] = 2
        counts = np.bincount(y_tri, minlength=3)
        if all(c >= MIN_TRIALS_PER_CLASS for c in counts[:3]):
            targets["HitMissFa"] = (tri_mask, y_tri)

    if not targets:
        return []

    # ── Define feature sets ───────────────────────────────────────────
    feature_sets = {}
    if X_behav_state is not None:
        feature_sets["Behav_state"] = X_behav_state
    feature_sets["Sensory"] = X_sensory
    if X_history is not None:
        feature_sets["Trial_history"] = X_history
    feature_sets["Neural"] = X_neural

    # Combinations
    if X_behav_state is not None:
        feature_sets["Behav+Sensory"] = np.hstack([X_behav_state, X_sensory])
        combo = [X_behav_state, X_sensory]
        if X_history is not None:
            combo.append(X_history)
            feature_sets["Full_behavioral"] = np.hstack(combo)
        feature_sets["Neural+Behav"] = np.hstack([X_neural, X_behav_state])
        all_parts = [X_neural, X_behav_state, X_sensory]
        if X_history is not None:
            all_parts.append(X_history)
        feature_sets["All_features"] = np.hstack(all_parts)
    else:
        feature_sets["All_features"] = np.hstack([X_neural, X_sensory])

    # ── Run prediction for each target x feature set ──────────────────
    results = []
    for target_name, (mask, y) in targets.items():
        best_auc = -1
        best_fset = None
        best_X = None

        for fset_name, X_full in feature_sets.items():
            X = X_full[mask]
            auc_mean, auc_sem, folds = cv_predict(X, y)

            row = {
                "session_name": session_name,
                "stage": stage,
                "session_idx": session_idx,
                "target": target_name,
                "feature_set": fset_name,
                "auc": auc_mean,
                "auc_sem": auc_sem,
                "n_trials": len(y),
                "n_units": n_units,
            }

            # Track counts per class
            if target_name == "Hit_vs_Miss":
                row["n_class0"] = int((y == 0).sum())  # miss
                row["n_class1"] = int((y == 1).sum())  # hit
            elif target_name == "Hit_vs_FA":
                row["n_class0"] = int((y == 0).sum())  # fa
                row["n_class1"] = int((y == 1).sum())  # hit
            elif target_name == "HitMissFa":
                row["n_class0"] = int((y == 0).sum())
                row["n_class1"] = int((y == 1).sum())
                row["n_class2"] = int((y == 2).sum())

            results.append(row)

            if not np.isnan(auc_mean) and auc_mean > best_auc:
                best_auc = auc_mean
                best_fset = fset_name
                best_X = X

        # Permutation null for the best feature set on this target
        if best_X is not None and not np.isnan(best_auc):
            null_dist = permutation_auc(best_X, y, n_perm=N_PERM)
            p_perm = (np.sum(null_dist >= best_auc) + 1) / (len(null_dist) + 1) if len(null_dist) > 0 else np.nan
            # Update best row
            for r in results:
                if (r["session_name"] == session_name and
                    r["target"] == target_name and
                    r["feature_set"] == best_fset):
                    r["p_perm"] = p_perm
                    r["null_mean"] = np.mean(null_dist) if len(null_dist) > 0 else np.nan
                    r["null_std"] = np.std(null_dist) if len(null_dist) > 0 else np.nan
                    r["is_best"] = True
                    break

    return results


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 60)
    print("[07e] Comprehensive single-trial outcome prediction")
    print("=" * 60)

    manifest = load_staging_manifest(qc_only=True)
    trial_df = load_trial_table()

    print(f"  {len(manifest)} QC-passed sessions")
    print(f"  {len(trial_df)} trials in cached table")

    # ── Per-session prediction loop ───────────────────────────────────
    print("\n[Step 1] Per-session prediction (all targets x feature sets)...")
    all_results = []

    for _, row in manifest.iterrows():
        sname = int(row["session_name"])
        stage = row["stage"]
        sidx = row["session_idx"]

        sess_trials = trial_df[trial_df["session_name"] == sname].copy()
        if len(sess_trials) < 30:
            continue

        print(f"  Session {sname} ({stage}, idx={sidx})...", end=" ", flush=True)
        try:
            sess = load_session(sname)
        except FileNotFoundError:
            print("pkl not found")
            continue

        sess_results = predict_session(sess, sname, sess_trials, stage, sidx)
        all_results.extend(sess_results)

        n_targets = len(set(r["target"] for r in sess_results))
        n_fsets = len(set(r["feature_set"] for r in sess_results))
        best_rows = [r for r in sess_results if r.get("is_best")]
        best_info = "; ".join(
            f"{r['target']}={r['auc']:.3f}({r['feature_set']})"
            for r in best_rows
        )
        print(f"{n_targets} targets, {n_fsets} fsets | {best_info}")

        del sess
        gc.collect()

    results_df = pd.DataFrame(all_results)

    if len(results_df) == 0:
        print("  ERROR: No prediction results. Exiting.")
        return

    print(f"\n  Total results: {len(results_df)} (target x feature_set x session)")

    # ── Save results ──────────────────────────────────────────────────
    cache_path = os.path.join(CACHE_DIR, "trial_outcome_prediction.csv")
    results_df.to_csv(cache_path, index=False)
    print(f"  Saved: {cache_path}")

    # ══════════════════════════════════════════════════════════════════
    # FIGURE 31: Comprehensive trial outcome prediction
    # ══════════════════════════════════════════════════════════════════
    print("\n[Step 2] Generating Figure 31...")
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.35)
    stats = []

    # ── Panel A: Summary heatmap (target x feature set) ───────────────
    ax_a = fig.add_subplot(gs[0, 0])

    # Compute mean AUC per target x feature_set
    pivot = results_df.groupby(["target", "feature_set"])["auc"].mean().unstack()

    # Order feature sets and targets
    fset_order = ["Behav_state", "Sensory", "Trial_history", "Neural",
                  "Behav+Sensory", "Full_behavioral", "Neural+Behav", "All_features"]
    target_order = ["Hit_vs_Miss", "Hit_vs_FA", "HitMissFa"]

    fset_present = [f for f in fset_order if f in pivot.columns]
    target_present = [t for t in target_order if t in pivot.index]
    pivot = pivot.reindex(index=target_present, columns=fset_present)

    im = ax_a.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                     vmin=0.45, vmax=0.85)
    ax_a.set_xticks(range(len(fset_present)))
    ax_a.set_xticklabels([f.replace("_", "\n") for f in fset_present],
                         fontsize=6, rotation=45, ha="right")
    ax_a.set_yticks(range(len(target_present)))
    target_display = {"Hit_vs_Miss": "Hit vs Miss",
                      "Hit_vs_FA": "Hit vs FA",
                      "HitMissFa": "3-way"}
    ax_a.set_yticklabels([target_display.get(t, t) for t in target_present], fontsize=8)

    # Annotate cells
    for i in range(len(target_present)):
        for j in range(len(fset_present)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.72 or val < 0.52 else "black"
                ax_a.text(j, i, f"{val:.3f}", ha="center", va="center",
                          fontsize=7, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax_a, shrink=0.7, label="Mean AUC")
    ax_a.set_title("A. Prediction summary\n(mean AUC across sessions)", fontweight="bold")

    # ── Panel B: Hit vs Miss across sessions ──────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])

    hm_df = results_df[results_df["target"] == "Hit_vs_Miss"].copy()
    if len(hm_df) > 0:
        # Find best feature set per session
        best_hm = hm_df.loc[hm_df.groupby("session_name")["auc"].idxmax()].copy()

        for stage in STAGE_ORDER:
            sub = best_hm[best_hm["stage"] == stage]
            if len(sub) == 0:
                continue
            ax_b.scatter(sub["session_idx"], sub["auc"],
                         c=STAGE_COLORS[stage], s=60, label=stage,
                         edgecolors="k", linewidth=0.5, zorder=3)

        # Also plot Neural-only for comparison
        neural_hm = hm_df[hm_df["feature_set"] == "Neural"]
        if len(neural_hm) > 0:
            ax_b.scatter(neural_hm["session_idx"], neural_hm["auc"],
                         marker="^", c="steelblue", s=35, alpha=0.5,
                         edgecolors="k", linewidth=0.3, label="Neural only", zorder=2)

        ax_b.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.5)
        add_stage_background(ax_b, manifest, alpha=0.06)
        ax_b.set_xlabel("Session index")
        ax_b.set_ylabel("AUC")
        ax_b.legend(fontsize=7, loc="lower right")

        # Spearman trend
        if len(best_hm) >= 5:
            r, p = spearmanr(best_hm["session_idx"], best_hm["auc"])
            stats.append({"test": "hit_miss_auc_vs_session", "rho": r, "p": p,
                           "n": len(best_hm)})
            ax_b.text(0.02, 0.98, f"rho={r:.3f}, p={p:.3f}", transform=ax_b.transAxes,
                      fontsize=7, va="top")

        # Mean AUC
        mean_auc = best_hm["auc"].mean()
        ax_b.text(0.98, 0.02, f"Mean best AUC={mean_auc:.3f}",
                  transform=ax_b.transAxes, fontsize=7, ha="right", va="bottom")

    ax_b.set_title("B. Hit vs Miss prediction\n(best model per session)", fontweight="bold")

    # ── Panel C: Hit vs FA across sessions ────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])

    hf_df = results_df[results_df["target"] == "Hit_vs_FA"].copy()
    if len(hf_df) > 0:
        best_hf = hf_df.loc[hf_df.groupby("session_name")["auc"].idxmax()].copy()

        for stage in STAGE_ORDER:
            sub = best_hf[best_hf["stage"] == stage]
            if len(sub) == 0:
                continue
            ax_c.scatter(sub["session_idx"], sub["auc"],
                         c=STAGE_COLORS[stage], s=60, label=stage,
                         edgecolors="k", linewidth=0.5, zorder=3)

        neural_hf = hf_df[hf_df["feature_set"] == "Neural"]
        if len(neural_hf) > 0:
            ax_c.scatter(neural_hf["session_idx"], neural_hf["auc"],
                         marker="^", c="steelblue", s=35, alpha=0.5,
                         edgecolors="k", linewidth=0.3, label="Neural only", zorder=2)

        ax_c.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.5)
        add_stage_background(ax_c, manifest, alpha=0.06)
        ax_c.set_xlabel("Session index")
        ax_c.set_ylabel("AUC")
        ax_c.legend(fontsize=7, loc="lower right")

        if len(best_hf) >= 5:
            r, p = spearmanr(best_hf["session_idx"], best_hf["auc"])
            stats.append({"test": "hit_fa_auc_vs_session", "rho": r, "p": p,
                           "n": len(best_hf)})
            ax_c.text(0.02, 0.98, f"rho={r:.3f}, p={p:.3f}", transform=ax_c.transAxes,
                      fontsize=7, va="top")

        mean_auc = best_hf["auc"].mean()
        ax_c.text(0.98, 0.02, f"Mean best AUC={mean_auc:.3f}",
                  transform=ax_c.transAxes, fontsize=7, ha="right", va="bottom")

    ax_c.set_title("C. Hit vs FA prediction\n(best model per session)", fontweight="bold")

    # ── Panel D: Feature set comparison (grouped bar) ─────────────────
    ax_d = fig.add_subplot(gs[1, 0])

    # For each target, show mean AUC by feature set (key ones only)
    key_fsets = ["Behav_state", "Neural", "Sensory",
                 "Full_behavioral", "Neural+Behav", "All_features"]
    key_fsets_present = [f for f in key_fsets if f in results_df["feature_set"].unique()]

    if len(key_fsets_present) > 0 and len(target_present) > 0:
        bar_width = 0.8 / len(target_present)
        target_colors = {"Hit_vs_Miss": "#e74c3c", "Hit_vs_FA": "#3498db",
                         "HitMissFa": "#9b59b6"}

        for ti, target in enumerate(target_present):
            sub = results_df[results_df["target"] == target]
            means = []
            sems = []
            for fset in key_fsets_present:
                vals = sub[sub["feature_set"] == fset]["auc"].dropna()
                means.append(vals.mean() if len(vals) > 0 else 0)
                sems.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0)

            x_pos = np.arange(len(key_fsets_present)) + ti * bar_width
            ax_d.bar(x_pos, means, bar_width * 0.9, yerr=sems,
                     color=target_colors.get(target, "#95a5a6"),
                     alpha=0.8, label=target_display.get(target, target),
                     capsize=2, edgecolor="white", linewidth=0.5)

        ax_d.set_xticks(np.arange(len(key_fsets_present)) + bar_width * (len(target_present) - 1) / 2)
        ax_d.set_xticklabels([f.replace("_", "\n") for f in key_fsets_present],
                             fontsize=6, rotation=45, ha="right")
        ax_d.set_ylabel("Mean AUC")
        ax_d.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.5)
        ax_d.legend(fontsize=7, loc="upper left")

    ax_d.set_title("D. Feature set comparison", fontweight="bold")

    # ── Panel E: Permutation null for best models ─────────────────────
    ax_e = fig.add_subplot(gs[1, 1])

    perm_rows = results_df.dropna(subset=["p_perm"])
    if len(perm_rows) > 0:
        for target in target_present:
            sub = perm_rows[perm_rows["target"] == target]
            if len(sub) == 0:
                continue
            observed = sub["auc"].values
            null_means = sub["null_mean"].values
            null_stds = sub["null_std"].values

            color = {"Hit_vs_Miss": "#e74c3c", "Hit_vs_FA": "#3498db",
                     "HitMissFa": "#9b59b6"}.get(target, "#95a5a6")

            # Plot observed vs null mean
            ax_e.scatter(null_means, observed, c=color, s=40,
                         edgecolors="k", linewidth=0.3,
                         label=f"{target_display.get(target, target)} (n={len(sub)})",
                         alpha=0.7, zorder=3)

        # Unity line (obs = null -> chance)
        lims = [0.3, 1.0]
        ax_e.plot(lims, lims, "k--", lw=1, alpha=0.3)
        ax_e.set_xlabel("Null AUC (permutation mean)")
        ax_e.set_ylabel("Observed AUC (best model)")
        ax_e.legend(fontsize=7)

        # How many sessions significantly above chance?
        n_sig = (perm_rows["p_perm"] < 0.05).sum()
        n_total = len(perm_rows)
        stats.append({"test": "perm_significant_sessions",
                       "n_sig": n_sig, "n_total": n_total,
                       "frac": n_sig / n_total if n_total > 0 else 0})
        ax_e.text(0.02, 0.98, f"p<0.05: {n_sig}/{n_total} session-targets",
                  transform=ax_e.transAxes, fontsize=7, va="top")

    ax_e.set_title("E. Observed vs permutation null", fontweight="bold")

    # ── Panel F: Stage comparison ─────────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])

    # Box plots: best AUC per session, by stage, for each target
    box_data = {}
    for target in target_present:
        sub = results_df[results_df["target"] == target]
        best_sub = sub.loc[sub.groupby("session_name")["auc"].idxmax()]
        for stage in STAGE_ORDER:
            key = (target, stage)
            vals = best_sub[best_sub["stage"] == stage]["auc"].dropna().values
            if len(vals) >= 2:
                box_data[key] = vals

    if box_data:
        # Plot grouped box plots
        positions = []
        tick_labels = []
        colors_list = []
        data_list = []
        pos = 0
        group_centers = {}

        for target in target_present:
            group_start = pos
            for si, stage in enumerate(STAGE_ORDER):
                key = (target, stage)
                if key in box_data:
                    data_list.append(box_data[key])
                    positions.append(pos)
                    tick_labels.append(f"{stage[:3]}")
                    colors_list.append(STAGE_COLORS[stage])
                    pos += 1
            group_centers[target] = (group_start + pos - 1) / 2
            pos += 0.5  # gap between targets

        if data_list:
            bp = ax_f.boxplot(data_list, positions=positions, widths=0.6,
                              patch_artist=True, showfliers=False)
            for patch, color in zip(bp["boxes"], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            # Strip plot
            rng = np.random.default_rng(42)
            for i, (vals, p) in enumerate(zip(data_list, positions)):
                jitter = rng.normal(0, 0.08, len(vals))
                ax_f.scatter(np.full(len(vals), p) + jitter, vals,
                             c=colors_list[i], s=30, edgecolors="k",
                             linewidth=0.3, zorder=3)

            ax_f.set_xticks(list(group_centers.values()))
            ax_f.set_xticklabels([target_display.get(t, t) for t in group_centers.keys()],
                                 fontsize=8)

            # Stage legend on tick labels at bottom
            ax_f.set_xlabel("")

            # Mann-Whitney per target (Learning vs Expert)
            for target in target_present:
                k_l = (target, "Learning")
                k_e = (target, "Expert")
                if k_l in box_data and k_e in box_data:
                    if len(box_data[k_l]) >= 3 and len(box_data[k_e]) >= 3:
                        U, p_u = mannwhitneyu(box_data[k_l], box_data[k_e],
                                              alternative="two-sided")
                        r_rb = 1 - 2 * U / (len(box_data[k_l]) * len(box_data[k_e]))
                        stats.append({"test": f"{target}_stage_mannwhitney",
                                       "U": U, "p": p_u, "r_rb": r_rb,
                                       "n_learning": len(box_data[k_l]),
                                       "n_expert": len(box_data[k_e]),
                                       "mean_learning": np.mean(box_data[k_l]),
                                       "mean_expert": np.mean(box_data[k_e])})

        ax_f.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.5)
        ax_f.set_ylabel("Best model AUC")

        # Custom legend for stages
        from matplotlib.patches import Patch
        stage_patches = [Patch(facecolor=STAGE_COLORS[s], label=s, alpha=0.6)
                         for s in STAGE_ORDER]
        ax_f.legend(handles=stage_patches, fontsize=7, loc="lower right")

    ax_f.set_title("F. Prediction by learning stage", fontweight="bold")

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig31_trial_outcome_prediction", "07_advanced")
    print("  Saved fig31_trial_outcome_prediction")

    if stats:
        stats_df = pd.DataFrame(stats)
        stats_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "figures", "07_advanced", "trial_outcome_prediction_stats.csv"
        )
        stats_df.to_csv(stats_path, index=False)
        print(f"  Saved statistics: {stats_path}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for target in target_present:
        sub = results_df[results_df["target"] == target]
        print(f"\n  {target_display.get(target, target)}:")
        for fset in fset_present:
            vals = sub[sub["feature_set"] == fset]["auc"].dropna()
            if len(vals) > 0:
                print(f"    {fset:20s}: AUC = {vals.mean():.3f} +/- {vals.std() / np.sqrt(len(vals)):.3f} "
                      f"(n={len(vals)} sessions)")

    # Best model summary
    print("\n  Best model per target:")
    for target in target_present:
        sub = results_df[results_df["target"] == target]
        best = sub.groupby("feature_set")["auc"].mean()
        if len(best) > 0:
            top = best.idxmax()
            print(f"    {target_display.get(target, target):15s}: {top} (AUC={best[top]:.3f})")

    # Permutation significance
    if len(perm_rows) > 0:
        n_sig = (perm_rows["p_perm"] < 0.05).sum()
        print(f"\n  Permutation significance: {n_sig}/{len(perm_rows)} session-targets at p<0.05")

    print("\nDone.")


if __name__ == "__main__":
    main()
