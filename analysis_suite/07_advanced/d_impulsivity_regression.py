"""Fig30: Impulsivity baseline regression — nested logistic models and neural prediction.

Scientific question:
  Is there a slow-varying 'impulsivity undercurrent' that modulates the
  baseline probability of early licking, on top of which stimulus-driven
  responses ride?  Can we separate early licks driven by actual sensory
  information from those driven by impulsivity, and can pre-trial neural
  activity predict single-trial outcomes beyond behavioral state?

Approach:
  1. Quantify the impulsivity baseline using HMM P(Impulsive) posteriors
     as a continuous, trial-by-trial impulsivity index.
  2. Build a hierarchical logistic regression:
       - Model 1 (Impulsivity only): P(lick) ~ P(Impulsive) + FA_history
       - Model 2 (+ Sensory):        + change_size + TF_responsive_fraction
       - Model 3 (+ Neural):         + pre-trial population firing rate
     Compare deviance explained to quantify each component's contribution.
  3. Classify FA trials as 'impulsivity-driven' vs 'sensory-residual'
     based on model residuals.
  4. Test whether pre-trial neural activity predicts next-trial Hit vs FA
     above and beyond the behavioral impulsivity index.

Produces:
  - Fig 30A: Impulsivity index (P(Impulsive)) trajectory with FA rate overlay
  - Fig 30B: Logistic regression component contributions (nested model comparison)
  - Fig 30C: ROC curves for Hit vs FA prediction at each model level
  - Fig 30D: Pre-trial neural activity predicts outcome beyond impulsivity
  - Fig 30E: Impulsivity-residual FA classification across learning stages
  - Fig 30F: Neural prediction accuracy across sessions

Saves:
  figures/07_advanced/fig30_impulsivity_regression.png
  figures/07_advanced/impulsivity_regression_stats.csv
  cache/impulsivity_trial_table.csv
"""

import os
import sys
import gc
import warnings


import numpy as np
import pandas as pd
from scipy.stats import (
    mannwhitneyu, spearmanr, pearsonr, sem, wilcoxon,
)
from scipy.special import expit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss

from visdetect.suite.config import (
    STAGE_ORDER, STAGE_COLORS, CACHE_DIR,
    HMM_STATE_ORDER, HMM_STATE_COLORS,
    CHANGE_SIZES, DEFAULT_BIN_SIZE,
)
from visdetect.suite.loader import (
    load_staging_manifest, load_session,
    load_hmm_assignments,
)
from visdetect.analysis.utils import get_good_cluster_ids, build_population_tensor
from visdetect.suite.plotting import setup_style, save_figure, add_stage_background

from visdetect.analysis.behavior import get_trial_dataframe
from visdetect.analysis.align import get_event_times_by_trial

setup_style()
warnings.filterwarnings("ignore")

# ── Parameters ────────────────────────────────────────────────────────
PRE_TRIAL_WINDOW = (-1.5, -0.5)   # Pre-change baseline for neural features
BIN_SIZE = DEFAULT_BIN_SIZE
MIN_UNITS = 5
MIN_TRIALS_PER_CLASS = 10
N_FOLDS = 5
FA_HISTORY_LAGS = 3                # Number of preceding trials for FA history
ROLLING_FA_WINDOW = 15             # Trials for rolling FA rate (impulsivity proxy)
IMPULSIVE_STATE_LABEL = "Impulsive"


# =====================================================================
# Section 1: Build trial-level table with impulsivity features
# =====================================================================

def build_impulsivity_trial_table(manifest, hmm_assign):
    """Build a comprehensive trial table with impulsivity features.

    For each trial, computes:
      - p_impulsive: HMM posterior probability of Impulsive state
      - fa_history_k: whether each of the k preceding trials was an FA
      - rolling_fa_rate: rolling FA rate over the preceding N trials
      - change_size_log2: log2(change_size), sensory evidence strength
      - licked: binary outcome (1 = licked: hit or fa; 0 = withheld)
    """
    # HMM state index for Impulsive (Biased in raw CSV, state 2)
    # After rename: Impulsive maps to p_state_2
    # Find the correct p_state column for Impulsive
    impulsive_p_col = None
    for state_idx in [0, 1, 2]:
        labels = hmm_assign[hmm_assign["hmm_state"] == state_idx]["hmm_state_label"].unique()
        if IMPULSIVE_STATE_LABEL in labels:
            impulsive_p_col = f"p_state_{state_idx}"
            break

    if impulsive_p_col is None:
        # Fallback: try raw label "Biased"
        for state_idx in [0, 1, 2]:
            labels = hmm_assign[hmm_assign["hmm_state"] == state_idx]["hmm_state_label"].unique()
            if "Biased" in labels:
                impulsive_p_col = f"p_state_{state_idx}"
                break

    if impulsive_p_col is None:
        raise ValueError("Cannot find Impulsive/Biased state in HMM assignments")

    print(f"  Impulsive state posterior column: {impulsive_p_col}")

    # Merge HMM posteriors with behavioral trial data
    # hmm_assign already has trial-level data with outcomes
    df = hmm_assign.copy()

    # Add stage and session_idx from manifest
    date_to_stage = dict(zip(manifest["session_name"].astype(int), manifest["stage"]))
    date_to_idx = dict(zip(manifest["session_name"].astype(int), manifest["session_idx"]))
    df["stage"] = df["session_name"].map(date_to_stage)
    df["session_idx"] = df["session_name"].map(date_to_idx)
    df = df.dropna(subset=["stage"])

    # Core features
    df["p_impulsive"] = df[impulsive_p_col].astype(float)
    df["licked"] = (df["is_hit"] | df["is_fa"]).astype(int)
    df["change_size_log2"] = np.log2(np.clip(df["change_size"].astype(float), 1.0, None))

    # FA history features (per-session)
    for lag in range(1, FA_HISTORY_LAGS + 1):
        df[f"prev_fa_{lag}"] = df.groupby("session_name")["is_fa"].shift(lag).astype(float)

    # Rolling FA rate (trailing window, per-session)
    df["rolling_fa_rate"] = (
        df.groupby("session_name")["is_fa"]
        .transform(lambda x: x.rolling(ROLLING_FA_WINDOW, min_periods=3).mean().shift(1))
    )

    # Previous trial licked
    df["prev_licked"] = df.groupby("session_name")["licked"].shift(1).astype(float)

    # Previous trial outcome
    df["prev_outcome"] = df.groupby("session_name")["outcome"].shift(1)

    return df


# =====================================================================
# Section 2: Nested logistic regression models
# =====================================================================

def fit_nested_models(df, include_neural=False):
    """Fit nested logistic regression models with cross-validation.

    Model hierarchy:
      M0: Intercept only (baseline)
      M1: Impulsivity (P(Impulsive) + FA history + rolling FA rate)
      M2: M1 + Sensory (change_size_log2)
      M3: M2 + Neural (pre-trial mean FR, if available)

    Returns dict with per-model AUC, log-loss, and coefficient summaries.
    """
    # Filter to trials with all behavioral features
    required_cols = ["p_impulsive", "rolling_fa_rate", "change_size_log2", "licked"]
    for lag in range(1, FA_HISTORY_LAGS + 1):
        required_cols.append(f"prev_fa_{lag}")
    mask = df[required_cols].notna().all(axis=1)
    clean = df[mask].copy()

    if len(clean) < 2 * MIN_TRIALS_PER_CLASS:
        return None

    y = clean["licked"].values

    # Check class balance
    if y.sum() < MIN_TRIALS_PER_CLASS or (len(y) - y.sum()) < MIN_TRIALS_PER_CLASS:
        return None

    # Feature sets for each model level
    impulsivity_cols = ["p_impulsive", "rolling_fa_rate"] + [
        f"prev_fa_{lag}" for lag in range(1, FA_HISTORY_LAGS + 1)
    ]
    sensory_cols = ["change_size_log2"]
    neural_cols = ["pre_trial_mean_fr"] if include_neural and "pre_trial_mean_fr" in clean.columns else []

    model_specs = {
        "M1_impulsivity": impulsivity_cols,
        "M2_imp_sensory": impulsivity_cols + sensory_cols,
    }
    if neural_cols:
        model_specs["M3_imp_sens_neural"] = impulsivity_cols + sensory_cols + neural_cols

    results = {}

    for model_name, feature_cols in model_specs.items():
        X = clean[feature_cols].values
        if np.any(~np.isfinite(X)):
            X = np.nan_to_num(X, nan=0.0)

        # Stratified CV
        n_splits = min(N_FOLDS, int(min(y.sum(), len(y) - y.sum())))
        if n_splits < 2:
            continue

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_aucs = []
        fold_logloss = []
        all_y_true = []
        all_y_prob = []
        coef_list = []

        for train_idx, test_idx in cv.split(X, y):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])

            clf = LogisticRegression(
                C=1.0, penalty="l2", solver="lbfgs", max_iter=500, random_state=42,
            )
            clf.fit(X_train, y[train_idx])
            y_prob = clf.predict_proba(X_test)[:, 1]

            fold_aucs.append(roc_auc_score(y[test_idx], y_prob))
            fold_logloss.append(log_loss(y[test_idx], y_prob))
            all_y_true.extend(y[test_idx])
            all_y_prob.extend(y_prob)
            coef_list.append(clf.coef_[0])

        results[model_name] = {
            "auc_mean": np.mean(fold_aucs),
            "auc_sem": np.std(fold_aucs) / np.sqrt(len(fold_aucs)),
            "auc_folds": fold_aucs,
            "logloss_mean": np.mean(fold_logloss),
            "logloss_sem": np.std(fold_logloss) / np.sqrt(len(fold_logloss)),
            "y_true": np.array(all_y_true),
            "y_prob": np.array(all_y_prob),
            "coef_mean": np.mean(coef_list, axis=0),
            "coef_sem": np.std(coef_list, axis=0) / np.sqrt(len(coef_list)),
            "feature_names": feature_cols,
            "n_trials": len(y),
            "n_lick": int(y.sum()),
            "n_no_lick": int(len(y) - y.sum()),
        }

    return results


# =====================================================================
# Section 3: Per-session neural prediction
# =====================================================================

def compute_pretrial_neural_features(sess, session_name, hmm_row_df):
    """Compute pre-trial mean firing rate for each trial in a session.

    Returns a DataFrame with trial_idx and pre_trial_mean_fr columns.
    """
    good_ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
    if len(good_ids) < MIN_UNITS:
        return None

    # Get trial indices we have HMM data for
    trial_indices = sorted(hmm_row_df["trial_idx"].unique())

    # Build population tensor aligned to Baseline_ON in pre-trial window.
    # NOTE: We use Baseline_ON (not Change_ON) because this analysis includes
    # FA trials where the change stimulus was never presented, making
    # Change_ON alignment invalid for those trials.
    tensor, bin_centers, used = build_population_tensor(
        sess, good_ids, event_name="Baseline_ON",
        window=PRE_TRIAL_WINDOW, bin_size=BIN_SIZE,
        trial_indices=trial_indices,
    )

    if tensor.shape[0] == 0 or tensor.shape[2] < MIN_UNITS:
        return None

    # Mean FR across time bins and units for each trial
    mean_fr = np.nanmean(tensor, axis=(1, 2))  # shape (n_trials,)

    result = pd.DataFrame({
        "trial_idx": used,
        "pre_trial_mean_fr": mean_fr,
        "n_units": tensor.shape[2],
    })

    return result


def neural_prediction_session(sess, session_name, hmm_row_df, stage):
    """Test whether pre-trial neural activity predicts Hit vs FA above
    impulsivity baseline, for a single session.

    Uses only go-trials where either Hit or FA occurred (trials where
    the animal licked), comparing whether neural state distinguishes
    sensory-driven hits from impulsive FAs.

    Returns dict with prediction accuracy or None.
    """
    good_ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
    if len(good_ids) < MIN_UNITS:
        return None

    # Merge neural features
    neural_df = compute_pretrial_neural_features(sess, session_name, hmm_row_df)
    if neural_df is None:
        return None

    merged = hmm_row_df.merge(neural_df, on="trial_idx", how="inner")

    # Focus on Hit vs FA trials only (both involve a lick)
    hit_fa = merged[merged["outcome"].isin(["hit", "fa"])].copy()
    hit_fa["is_hit_binary"] = (hit_fa["outcome"] == "hit").astype(int)

    n_hit = hit_fa["is_hit_binary"].sum()
    n_fa = len(hit_fa) - n_hit

    if n_hit < MIN_TRIALS_PER_CLASS or n_fa < MIN_TRIALS_PER_CLASS:
        return None

    # Build per-unit pre-trial features (not just mean FR)
    trial_indices = sorted(hit_fa["trial_idx"].values)
    tensor, bin_centers, used = build_population_tensor(
        sess, good_ids, event_name="Baseline_ON",
        window=PRE_TRIAL_WINDOW, bin_size=BIN_SIZE,
        trial_indices=trial_indices,
    )

    if tensor.shape[0] < 2 * MIN_TRIALS_PER_CLASS or tensor.shape[2] < MIN_UNITS:
        return None

    # Mean FR per unit across pre-trial window -> (n_trials, n_units)
    X_neural = np.nanmean(tensor, axis=1)
    X_neural = np.nan_to_num(X_neural, nan=0.0)

    # Match labels to used trial indices
    used_set = set(used)
    hit_fa_filtered = hit_fa[hit_fa["trial_idx"].isin(used_set)].copy()
    # Align order
    idx_to_row = {idx: i for i, idx in enumerate(used)}
    hit_fa_filtered["tensor_row"] = hit_fa_filtered["trial_idx"].map(idx_to_row)
    hit_fa_filtered = hit_fa_filtered.dropna(subset=["tensor_row"])
    hit_fa_filtered = hit_fa_filtered.sort_values("tensor_row")

    tensor_rows = hit_fa_filtered["tensor_row"].astype(int).values
    X_neural = X_neural[tensor_rows]
    y = hit_fa_filtered["is_hit_binary"].values
    p_imp = hit_fa_filtered["p_impulsive"].values.reshape(-1, 1)

    if y.sum() < MIN_TRIALS_PER_CLASS or (len(y) - y.sum()) < MIN_TRIALS_PER_CLASS:
        return None

    n_splits = min(N_FOLDS, int(min(y.sum(), len(y) - y.sum())))
    if n_splits < 2:
        return None

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Model A: Impulsivity only (P(Impulsive))
    aucs_imp = []
    for train_idx, test_idx in cv.split(p_imp, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(p_imp[train_idx])
        X_te = scaler.transform(p_imp[test_idx])
        clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=500)
        clf.fit(X_tr, y[train_idx])
        prob = clf.predict_proba(X_te)[:, 1]
        try:
            aucs_imp.append(roc_auc_score(y[test_idx], prob))
        except ValueError:
            pass

    # Model B: Neural only (pre-trial population FR)
    aucs_neural = []
    for train_idx, test_idx in cv.split(X_neural, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_neural[train_idx])
        X_te = scaler.transform(X_neural[test_idx])
        clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=500)
        clf.fit(X_tr, y[train_idx])
        prob = clf.predict_proba(X_te)[:, 1]
        try:
            aucs_neural.append(roc_auc_score(y[test_idx], prob))
        except ValueError:
            pass

    # Model C: Neural + Impulsivity (combined)
    X_combined = np.hstack([X_neural, p_imp])
    aucs_combined = []
    for train_idx, test_idx in cv.split(X_combined, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_combined[train_idx])
        X_te = scaler.transform(X_combined[test_idx])
        clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=500)
        clf.fit(X_tr, y[train_idx])
        prob = clf.predict_proba(X_te)[:, 1]
        try:
            aucs_combined.append(roc_auc_score(y[test_idx], prob))
        except ValueError:
            pass

    if not aucs_imp or not aucs_neural or not aucs_combined:
        return None

    return {
        "session_name": session_name,
        "stage": stage,
        "n_hit": int(n_hit),
        "n_fa": int(n_fa),
        "n_units": int(X_neural.shape[1]),
        "auc_impulsivity": np.mean(aucs_imp),
        "auc_neural": np.mean(aucs_neural),
        "auc_combined": np.mean(aucs_combined),
        "auc_neural_gain": np.mean(aucs_combined) - np.mean(aucs_imp),
    }


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 60)
    print("[07d] Impulsivity baseline regression & neural prediction")
    print("=" * 60)

    manifest = load_staging_manifest(qc_only=True)
    hmm_assign = load_hmm_assignments()

    print(f"  {len(manifest)} QC-passed sessions")
    print(f"  {len(hmm_assign)} HMM trial assignments")

    # ── Step 1: Build trial table with impulsivity features ───────────
    print("\n[Step 1] Building impulsivity trial table...")
    trial_df = build_impulsivity_trial_table(manifest, hmm_assign)
    print(f"  {len(trial_df)} trials with impulsivity features")
    print(f"  P(Impulsive) range: {trial_df['p_impulsive'].min():.4f} – "
          f"{trial_df['p_impulsive'].max():.4f}")
    print(f"  Outcomes: {trial_df['outcome'].value_counts().to_dict()}")

    # ── Step 2: Nested logistic regression (behavioral only) ──────────
    print("\n[Step 2] Fitting nested logistic models (all trials pooled)...")
    model_results = fit_nested_models(trial_df, include_neural=False)

    if model_results:
        for name, res in model_results.items():
            print(f"  {name}: AUC={res['auc_mean']:.3f}±{res['auc_sem']:.3f}, "
                  f"LogLoss={res['logloss_mean']:.3f}")
    else:
        print("  WARNING: Could not fit models (insufficient data)")

    # ── Step 3: Per-session neural prediction (Hit vs FA) ─────────────
    print("\n[Step 3] Per-session neural prediction (Hit vs FA)...")
    neural_results = []
    trial_df["pre_trial_mean_fr"] = np.nan  # Initialize column

    for _, row in manifest.iterrows():
        sname = int(row["session_name"])
        stage = row["stage"]
        sidx = row["session_idx"]

        sess_hmm = hmm_assign[hmm_assign["session_name"] == sname].copy()
        if len(sess_hmm) < 20:
            continue

        print(f"  Session {sname} ({stage})...", end=" ")
        try:
            sess = load_session(sname)
        except FileNotFoundError:
            print("pkl not found")
            continue

        # Add pre-trial neural features to trial table
        neural_feats = compute_pretrial_neural_features(sess, sname, sess_hmm)
        if neural_feats is not None:
            # Map trial_idx -> pre_trial_mean_fr for this session
            fr_map = dict(zip(neural_feats["trial_idx"], neural_feats["pre_trial_mean_fr"]))
            sess_mask = trial_df["session_name"] == sname
            trial_df.loc[sess_mask, "pre_trial_mean_fr"] = (
                trial_df.loc[sess_mask, "trial_idx"].map(fr_map)
            )

        # Neural prediction — use enriched trial_df (has p_impulsive)
        sess_trials_enriched = trial_df[trial_df["session_name"] == sname].copy()
        result = neural_prediction_session(sess, sname, sess_trials_enriched, stage)
        if result is not None:
            result["session_idx"] = sidx
            neural_results.append(result)
            print(f"AUC imp={result['auc_impulsivity']:.3f}, "
                  f"neural={result['auc_neural']:.3f}, "
                  f"combined={result['auc_combined']:.3f}")
        else:
            print("too few Hit/FA trials")

        del sess
        gc.collect()

    neural_df = pd.DataFrame(neural_results)
    print(f"\n  Neural prediction completed for {len(neural_df)} sessions")

    # ── Step 4: Re-fit nested models with neural features ─────────────
    if "pre_trial_mean_fr" in trial_df.columns and trial_df["pre_trial_mean_fr"].notna().sum() > 100:
        print("\n[Step 4] Re-fitting nested models with neural features...")
        model_results_neural = fit_nested_models(trial_df, include_neural=True)
        if model_results_neural:
            for name, res in model_results_neural.items():
                print(f"  {name}: AUC={res['auc_mean']:.3f}±{res['auc_sem']:.3f}")
    else:
        model_results_neural = model_results

    # Use best available model results for plotting
    plot_models = model_results_neural if model_results_neural else model_results

    # ── Step 5: Cache trial table ─────────────────────────────────────
    cache_path = os.path.join(CACHE_DIR, "impulsivity_trial_table.csv")
    save_cols = ["session_name", "trial_idx", "stage", "session_idx", "outcome",
                 "change_size", "change_size_log2", "is_hit", "is_fa", "is_go",
                 "p_impulsive", "rolling_fa_rate", "licked", "prev_outcome"]
    for lag in range(1, FA_HISTORY_LAGS + 1):
        save_cols.append(f"prev_fa_{lag}")
    if "pre_trial_mean_fr" in trial_df.columns:
        save_cols.append("pre_trial_mean_fr")
    available_cols = [c for c in save_cols if c in trial_df.columns]
    trial_df[available_cols].to_csv(cache_path, index=False)
    print(f"\n  Cached trial table: {cache_path}")

    # ══════════════════════════════════════════════════════════════════
    # FIGURE 30: Impulsivity regression & neural prediction
    # ══════════════════════════════════════════════════════════════════
    print("\n[Step 5] Generating Figure 30...")
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.35)

    stats = []

    # ── Panel A: Impulsivity index trajectory with FA rate overlay ────
    ax_a = fig.add_subplot(gs[0, 0])

    sess_summary = trial_df.groupby(["session_name", "session_idx", "stage"]).agg(
        mean_p_imp=("p_impulsive", "mean"),
        fa_rate=("is_fa", "mean"),
        n_trials=("trial_idx", "count"),
    ).reset_index().sort_values("session_idx")

    # P(Impulsive) trajectory
    ax_a.plot(sess_summary["session_idx"], sess_summary["mean_p_imp"],
              "o-", color=HMM_STATE_COLORS.get("Impulsive", "#fb6a4a"),
              markersize=6, linewidth=1.5, label="Mean P(Impulsive)", zorder=3)

    # FA rate on secondary axis
    ax_a2 = ax_a.twinx()
    ax_a2.plot(sess_summary["session_idx"], sess_summary["fa_rate"],
               "s--", color="#FF9800", markersize=5, linewidth=1, alpha=0.7,
               label="FA rate")
    ax_a2.set_ylabel("FA rate", color="#FF9800")
    ax_a2.tick_params(axis="y", labelcolor="#FF9800")

    # Correlation
    valid_corr = sess_summary.dropna(subset=["mean_p_imp", "fa_rate"])
    if len(valid_corr) >= 5:
        r, p = spearmanr(valid_corr["mean_p_imp"], valid_corr["fa_rate"])
        stats.append({"test": "p_impulsive_vs_fa_rate_spearman",
                       "rho": r, "p": p, "n": len(valid_corr)})
        ax_a.text(0.02, 0.98, f"ρ={r:.3f}, p={p:.3f}",
                  transform=ax_a.transAxes, fontsize=8, va="top")

    add_stage_background(ax_a, manifest, alpha=0.06)
    ax_a.set_xlabel("Session index")
    ax_a.set_ylabel("Mean P(Impulsive)", color=HMM_STATE_COLORS.get("Impulsive", "#fb6a4a"))
    ax_a.set_title("A. Impulsivity index & FA rate across learning", fontweight="bold")
    ax_a.legend(loc="upper left", fontsize=7)
    ax_a2.legend(loc="upper right", fontsize=7)

    # ── Panel B: Nested model comparison (AUC bar chart) ──────────────
    ax_b = fig.add_subplot(gs[0, 1])

    if plot_models:
        model_names = list(plot_models.keys())
        display_names = {
            "M1_impulsivity": "Impulsivity\nonly",
            "M2_imp_sensory": "Impulsivity\n+ Sensory",
            "M3_imp_sens_neural": "Impulsivity\n+ Sensory\n+ Neural",
        }
        model_colors = {
            "M1_impulsivity": HMM_STATE_COLORS.get("Impulsive", "#fb6a4a"),
            "M2_imp_sensory": "#2ecc71",
            "M3_imp_sens_neural": "#3498db",
        }

        x_pos = np.arange(len(model_names))
        aucs = [plot_models[m]["auc_mean"] for m in model_names]
        auc_errs = [plot_models[m]["auc_sem"] for m in model_names]
        colors = [model_colors.get(m, "#95a5a6") for m in model_names]

        bars = ax_b.bar(x_pos, aucs, yerr=auc_errs, color=colors,
                        edgecolor="white", linewidth=1.5, capsize=5, alpha=0.85)
        ax_b.set_xticks(x_pos)
        ax_b.set_xticklabels([display_names.get(m, m) for m in model_names], fontsize=8)
        ax_b.set_ylabel("Cross-validated AUC")
        ax_b.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.5, label="Chance")
        ax_b.set_ylim(0.45, min(1.0, max(aucs) + 0.1))

        # Annotate AUC values on bars
        for i, (auc, err) in enumerate(zip(aucs, auc_errs)):
            ax_b.text(i, auc + err + 0.01, f"{auc:.3f}", ha="center", fontsize=8)

        # Delta AUC statistics
        if len(model_names) >= 2:
            m1_folds = plot_models[model_names[0]]["auc_folds"]
            m2_folds = plot_models[model_names[1]]["auc_folds"]
            if len(m1_folds) == len(m2_folds) and len(m1_folds) >= 3:
                try:
                    w_stat, p_w = wilcoxon(m2_folds, m1_folds, alternative="greater")
                    stats.append({"test": "M2_vs_M1_auc_wilcoxon",
                                   "W": w_stat, "p": p_w,
                                   "delta_auc": np.mean(m2_folds) - np.mean(m1_folds)})
                except ValueError:
                    pass

        n_info = plot_models[model_names[0]]
        ax_b.text(0.98, 0.02,
                  f"n={n_info['n_trials']} trials\n({n_info['n_lick']} lick, {n_info['n_no_lick']} no-lick)",
                  transform=ax_b.transAxes, fontsize=7, ha="right", va="bottom")

    ax_b.set_title("B. Nested model comparison: P(lick)", fontweight="bold")

    # ── Panel C: Coefficient contributions from M2 ────────────────────
    ax_c = fig.add_subplot(gs[0, 2])

    if plot_models and "M2_imp_sensory" in plot_models:
        m2 = plot_models["M2_imp_sensory"]
        feat_names = m2["feature_names"]
        coefs = m2["coef_mean"]
        coef_errs = m2["coef_sem"]

        # Display names for features
        feat_display = {
            "p_impulsive": "P(Impulsive)",
            "rolling_fa_rate": "Rolling FA rate",
            "prev_fa_1": "Prev FA (lag 1)",
            "prev_fa_2": "Prev FA (lag 2)",
            "prev_fa_3": "Prev FA (lag 3)",
            "change_size_log2": "log₂(change size)",
            "pre_trial_mean_fr": "Pre-trial FR",
        }

        y_pos = np.arange(len(feat_names))
        colors_coef = ["#fb6a4a" if "fa" in f or "impulsive" in f
                       else "#2ecc71" if "change" in f
                       else "#3498db" for f in feat_names]

        ax_c.barh(y_pos, coefs, xerr=coef_errs, color=colors_coef,
                  edgecolor="white", linewidth=0.5, capsize=3, alpha=0.85)
        ax_c.set_yticks(y_pos)
        ax_c.set_yticklabels([feat_display.get(f, f) for f in feat_names], fontsize=8)
        ax_c.axvline(0, color="grey", ls="--", lw=1)
        ax_c.set_xlabel("Logistic regression coefficient (standardized)")

        # Record coefficients
        for f, c, e in zip(feat_names, coefs, coef_errs):
            stats.append({"test": f"M2_coef_{f}", "coef": c, "sem": e})

    ax_c.set_title("C. Feature contributions to P(lick)", fontweight="bold")

    # ── Panel D: Per-session neural prediction (Hit vs FA) ────────────
    ax_d = fig.add_subplot(gs[1, 0])

    if len(neural_df) > 0:
        for stage in STAGE_ORDER:
            sub = neural_df[neural_df["stage"] == stage]
            if len(sub) == 0:
                continue
            ax_d.scatter(sub["session_idx"], sub["auc_combined"],
                         c=STAGE_COLORS[stage], s=60, label=f"{stage} (combined)",
                         edgecolors="k", linewidth=0.5, zorder=3, marker="o")
            ax_d.scatter(sub["session_idx"], sub["auc_impulsivity"],
                         c=STAGE_COLORS[stage], s=40,
                         edgecolors="k", linewidth=0.5, zorder=2, marker="^", alpha=0.5)

        ax_d.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.5, label="Chance")
        add_stage_background(ax_d, manifest, alpha=0.06)
        ax_d.set_xlabel("Session index")
        ax_d.set_ylabel("AUC (Hit vs FA)")
        ax_d.legend(fontsize=7, loc="lower right")

        # Summary stats
        if len(neural_df) >= 3:
            combined_above_chance = neural_df["auc_combined"] > 0.5
            stats.append({"test": "neural_combined_above_chance",
                           "n_above": int(combined_above_chance.sum()),
                           "n_total": len(neural_df),
                           "mean_auc": neural_df["auc_combined"].mean(),
                           "mean_gain": neural_df["auc_neural_gain"].mean()})

            # Spearman: does neural prediction improve with learning?
            r_n, p_n = spearmanr(neural_df["session_idx"], neural_df["auc_combined"])
            stats.append({"test": "neural_auc_vs_session_spearman",
                           "rho": r_n, "p": p_n})
            ax_d.text(0.02, 0.98,
                      f"ρ={r_n:.3f}, p={p_n:.3f}\n"
                      f"○ combined, △ impulsivity only",
                      transform=ax_d.transAxes, fontsize=7, va="top")

    ax_d.set_title("D. Neural prediction: Hit vs FA per session", fontweight="bold")

    # ── Panel E: Neural gain over impulsivity (box by stage) ──────────
    ax_e = fig.add_subplot(gs[1, 1])

    if len(neural_df) > 0:
        stage_data = []
        stage_labels = []
        for stage in STAGE_ORDER:
            vals = neural_df[neural_df["stage"] == stage]["auc_neural_gain"].dropna().values
            if len(vals) >= 2:
                stage_data.append(vals)
                stage_labels.append(stage)

        if stage_data:
            bp = ax_e.boxplot(stage_data, labels=stage_labels, patch_artist=True, widths=0.5)
            for i, (box, stg) in enumerate(zip(bp["boxes"], stage_labels)):
                box.set_facecolor(STAGE_COLORS[stg])
                box.set_alpha(0.6)
                # Strip plot
                vals = stage_data[i]
                jitter = np.random.default_rng(42).normal(0, 0.05, len(vals))
                ax_e.scatter(np.full(len(vals), i + 1) + jitter, vals,
                             c=STAGE_COLORS[stg], s=35, edgecolors="k",
                             linewidth=0.5, zorder=3)

            ax_e.axhline(0, color="grey", ls="--", lw=1, alpha=0.5, label="No gain")
            ax_e.set_ylabel("ΔAUC (combined − impulsivity only)")

            # Test: is neural gain > 0 across all sessions?
            all_gains = neural_df["auc_neural_gain"].dropna().values
            if len(all_gains) >= 5:
                try:
                    w_g, p_g = wilcoxon(all_gains, alternative="greater")
                    stats.append({"test": "neural_gain_gt_zero_wilcoxon",
                                   "W": w_g, "p": p_g,
                                   "median_gain": float(np.median(all_gains)),
                                   "mean_gain": float(np.mean(all_gains))})
                    ax_e.text(0.5, 0.98, f"Gain > 0: p={p_g:.3f}",
                              transform=ax_e.transAxes, fontsize=8, ha="center", va="top")
                except ValueError:
                    pass

    ax_e.set_title("E. Neural gain over impulsivity by stage", fontweight="bold")

    # ── Panel F: Impulsivity-residual FA classification ───────────────
    ax_f = fig.add_subplot(gs[1, 2])

    # For each FA trial, compute residual P(lick) from impulsivity-only model
    # High-residual FAs = licking beyond what impulsivity predicts -> potentially sensory
    # Low-residual FAs = licking well-explained by impulsivity -> impulsive
    if plot_models and "M1_impulsivity" in plot_models and "M2_imp_sensory" in plot_models:
        fa_trials = trial_df[trial_df["is_fa"] == True].copy()
        fa_trials = fa_trials.dropna(subset=["p_impulsive", "rolling_fa_rate", "change_size_log2"])

        if len(fa_trials) >= 20:
            # Fit full model on all data to get residuals
            imp_cols = ["p_impulsive", "rolling_fa_rate"] + [
                f"prev_fa_{lag}" for lag in range(1, FA_HISTORY_LAGS + 1)
            ]
            all_clean = trial_df.dropna(subset=imp_cols + ["change_size_log2", "licked"]).copy()
            X_imp = all_clean[imp_cols].values
            X_imp = np.nan_to_num(X_imp, nan=0.0)
            y_all = all_clean["licked"].values

            scaler = StandardScaler()
            X_imp_s = scaler.fit_transform(X_imp)
            clf_imp = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=500)
            clf_imp.fit(X_imp_s, y_all)

            # Predicted P(lick) from impulsivity model for FA trials
            fa_in_clean = all_clean[all_clean["is_fa"] == True].copy()
            X_fa_imp = fa_in_clean[imp_cols].values
            X_fa_imp = np.nan_to_num(X_fa_imp, nan=0.0)
            fa_p_imp = clf_imp.predict_proba(scaler.transform(X_fa_imp))[:, 1]

            fa_in_clean = fa_in_clean.copy()
            fa_in_clean["p_lick_impulsivity"] = fa_p_imp
            fa_in_clean["residual"] = 1.0 - fa_p_imp  # How much licking is unexplained

            # Classify: high vs low impulsivity-explained FAs
            median_p = np.median(fa_p_imp)
            fa_in_clean["fa_type"] = np.where(
                fa_p_imp >= median_p,
                "Impulsivity-driven",
                "Sensory-residual"
            )

            # Plot proportions by stage
            fa_by_stage = fa_in_clean.groupby(["stage", "fa_type"]).size().unstack(fill_value=0)
            fa_fracs = fa_by_stage.div(fa_by_stage.sum(axis=1), axis=0)

            stage_order_present = [s for s in STAGE_ORDER if s in fa_fracs.index]
            if stage_order_present:
                fa_fracs = fa_fracs.reindex(stage_order_present)
                x_pos = np.arange(len(stage_order_present))

                bot = np.zeros(len(stage_order_present))
                fa_type_colors = {
                    "Impulsivity-driven": HMM_STATE_COLORS.get("Impulsive", "#fb6a4a"),
                    "Sensory-residual": "#2ecc71",
                }
                for fa_type in ["Impulsivity-driven", "Sensory-residual"]:
                    if fa_type in fa_fracs.columns:
                        vals = fa_fracs[fa_type].values
                        ax_f.bar(x_pos, vals, bottom=bot, width=0.6,
                                 color=fa_type_colors.get(fa_type, "#95a5a6"),
                                 label=fa_type, edgecolor="white", linewidth=0.5)
                        bot += vals

                ax_f.set_xticks(x_pos)
                ax_f.set_xticklabels(stage_order_present)
                ax_f.set_ylabel("Fraction of FA trials")
                ax_f.set_ylim(0, 1.05)
                ax_f.legend(fontsize=8, loc="upper right")

                # Count annotations
                for i, stg in enumerate(stage_order_present):
                    n = fa_by_stage.loc[stg].sum()
                    ax_f.text(i, 1.01, f"n={n}", ha="center", fontsize=7)

                # Stats: does the impulsivity/sensory split change with learning?
                if len(stage_order_present) >= 2:
                    ct = fa_by_stage.reindex(stage_order_present).values
                    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                        from scipy.stats import chi2_contingency
                        chi2, p_chi2, _, _ = chi2_contingency(ct)
                        stats.append({"test": "fa_type_by_stage_chi2",
                                       "chi2": chi2, "p": p_chi2})
                        ax_f.text(0.5, 0.02,
                                  f"χ²={chi2:.2f}, p={p_chi2:.3f}",
                                  transform=ax_f.transAxes, ha="center", fontsize=7)

    ax_f.set_title("F. FA classification: impulsive vs residual", fontweight="bold")

    # ── Save figure ───────────────────────────────────────────────────
    save_figure(fig, "fig30_impulsivity_regression", "07_advanced")
    print("  Saved fig30_impulsivity_regression")

    # ── Save statistics ───────────────────────────────────────────────
    if stats:
        stats_df = pd.DataFrame(stats)
        stats_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "figures", "07_advanced", "impulsivity_regression_stats.csv"
        )
        stats_df.to_csv(stats_path, index=False)
        print(f"  Saved statistics: {stats_path}")

    # ── Save neural results ───────────────────────────────────────────
    if len(neural_df) > 0:
        neural_path = os.path.join(CACHE_DIR, "neural_impulsivity_prediction.csv")
        neural_df.to_csv(neural_path, index=False)
        print(f"  Saved neural results: {neural_path}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if plot_models:
        for name, res in plot_models.items():
            print(f"  {name}: AUC = {res['auc_mean']:.3f} ± {res['auc_sem']:.3f}")
    if len(neural_df) > 0:
        print(f"\n  Neural prediction (Hit vs FA):")
        print(f"    Sessions analyzed: {len(neural_df)}")
        print(f"    Mean AUC (impulsivity only): {neural_df['auc_impulsivity'].mean():.3f}")
        print(f"    Mean AUC (neural only):      {neural_df['auc_neural'].mean():.3f}")
        print(f"    Mean AUC (combined):         {neural_df['auc_combined'].mean():.3f}")
        print(f"    Mean neural gain (dAUC):     {neural_df['auc_neural_gain'].mean():.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
