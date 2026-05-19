"""Fit a Bernoulli GLM-HMM to behavioural data across sessions.

End-to-end script: loads sessions from a manifest, fits GLM-HMMs with
model selection (BIC over K), saves the fitted model + per-trial state
assignments, and generates diagnostic / interpretation plots.

Reference
---------
Ashwood, Roy, Stone et al. (2022). "Mice alternate between discrete
strategies during perceptual decision-making."
Nature Neuroscience 25, 201-212.

Usage
-----
    python scripts/analysis/behavior/fit_behavioral_hmm.py \
        --manifest data/BG_046_staging_manifest.csv \
        --pkl-dir  data/pkls/BG_046 \
        --out      FIGURES/behavior/BG_046/hmm \
        --data-out data/hmm/BG_046

    # With parallel processing (recommended for faster fitting):
    python scripts/analysis/behavior/fit_behavioral_hmm.py \
        --manifest data/BG_046_staging_manifest.csv \
        --pkl-dir  data/pkls/BG_046 \
        --out      FIGURES/behavior/BG_046/hmm \
        --data-out data/hmm/BG_046 \
        --n-workers 12

    # Replot only (no refitting):
    python scripts/analysis/behavior/fit_behavioral_hmm.py \
        --manifest data/BG_046_staging_manifest.csv \
        --out      FIGURES/behavior/BG_046/hmm \
        --data-out data/hmm/BG_046 \
        --replot-only
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns


from visdetect.analysis.config import load_staging_manifest, HMM_STATE_COLORS
from visdetect.core.session import load_session
from visdetect.analysis.hmm import (
    GLMHMM,
    GLMHMMConfig,
    auto_label_states,
    decode_session,
    fit_best_model,
    prepare_session_data,
)
from visdetect.analysis.hmm_downstream import loso_cross_validation
from visdetect.viz.plotting import set_style, despine


# =====================================================================
# Helpers
# =====================================================================

def _parse_date(session_name: str, subject: str = "") -> datetime:
    """Extract date from session name (format Subject_DDMMYYYY)."""
    try:
        date_str = session_name.split("_")[-1]
        return datetime.strptime(date_str, "%d%m%Y")
    except Exception:
        return datetime.min


# =====================================================================
# Plotting functions
# =====================================================================

def plot_model_selection(selection_df: pd.DataFrame, out_dir: Path):
    """BIC and AIC vs K."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(selection_df["K"], selection_df["bic"], "o-", color="tab:blue",
            label="BIC", linewidth=2, markersize=8)
    ax.plot(selection_df["K"], selection_df["aic"], "s--", color="tab:orange",
            label="AIC", linewidth=2, markersize=8)
    best_k = int(selection_df.loc[selection_df["bic"].idxmin(), "K"])
    best_bic = selection_df.loc[selection_df["bic"].idxmin(), "bic"]
    ax.annotate(f"K={best_k}", (best_k, best_bic),
                textcoords="offset points", xytext=(10, -15),
                fontsize=12, fontweight="bold", color="tab:blue")
    ax.set_xlabel("Number of states (K)")
    ax.set_ylabel("Information criterion")
    ax.set_xticks(selection_df["K"].values)
    ax.legend()
    ax.set_title("Model Selection")
    despine(ax)
    plt.tight_layout()
    fig.savefig(out_dir / "model_selection.png", dpi=150)
    plt.close(fig)


def plot_state_psychometrics(model: GLMHMM, state_labels: list, out_dir: Path):
    """P(lick) vs stimulus for each state — the signature plot."""
    psy = model.state_psychometrics()
    change_sizes = [1.0, 1.25, 1.35, 1.5, 2.0, 4.0]
    stim_vals = np.log2(change_sizes)

    fig, ax = plt.subplots(figsize=(7, 5))
    palette = _state_palette(state_labels)
    for k in range(model.n_states):
        sub = psy[psy["state"] == k]
        ax.plot(sub["stimulus"], sub["p_lick"], "o-",
                color=palette[k], label=state_labels[k], linewidth=2, markersize=8)
    ax.set_xticks(stim_vals)
    ax.set_xticklabels([str(c) for c in change_sizes])
    ax.set_xlabel("Change Size")
    ax.set_ylabel("P(lick)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("State Psychometric Curves")
    ax.legend(title="State")
    despine(ax)
    plt.tight_layout()
    fig.savefig(out_dir / "state_psychometrics.png", dpi=150)
    plt.close(fig)


def plot_transition_matrix(model: GLMHMM, state_labels: list, out_dir: Path):
    """Annotated heatmap of A."""
    A = model.transition_matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(A, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=state_labels, yticklabels=state_labels, ax=ax,
                cbar_kws={"label": "P(transition)"})
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_title("State Transition Matrix")
    plt.tight_layout()
    fig.savefig(out_dir / "transition_matrix.png", dpi=150)
    plt.close(fig)


def plot_glm_weights(model: GLMHMM, state_labels: list, out_dir: Path):
    """Bar chart of GLM weights per state."""
    K = model.n_states
    D = model.n_features
    feat_names = model.feature_names if model.feature_names else [f"x{i}" for i in range(D)]
    palette = _state_palette(state_labels)

    fig, axes = plt.subplots(1, K, figsize=(4 * K, 4), sharey=True)
    if K == 1:
        axes = [axes]
    for k in range(K):
        ax = axes[k]
        bars = ax.barh(range(D), model.weights[k], color=palette[k], edgecolor="k")
        ax.set_yticks(range(D))
        ax.set_yticklabels(feat_names if k == 0 else [])
        ax.axvline(0, color="k", linewidth=0.5)
        ax.set_title(state_labels[k])
        ax.set_xlabel("Weight")
    fig.suptitle("GLM Weights per State", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "glm_weights.png", dpi=150)
    plt.close(fig)


def plot_session_states(posteriors: np.ndarray, df: pd.DataFrame,
                        session_name: str, state_labels: list,
                        out_dir: Path):
    """Trial-by-trial state posteriors for one session (line plot style).

    Parameters
    ----------
    posteriors : (T, K) array of state posterior probabilities.
    df : DataFrame with trial-level columns (is_hit, is_go, is_fa, …).
    session_name : Used for title and filename.
    state_labels : List of K label strings.
    out_dir : Directory to save into (a session_states/ sub-dir is created).
    """
    T, K = posteriors.shape
    if T == 0:
        return
    palette = _state_palette(state_labels)
    trial_idx = np.arange(T)
    ml_states = posteriors.argmax(axis=1)

    fig, (ax_top, ax_vit, ax_bot) = plt.subplots(
        3, 1, figsize=(12, 5.5),
        gridspec_kw={"height_ratios": [1, 0.25, 3]},
        sharex=True)

    # Top: outcome raster
    for i, row in df.iterrows():
        if i >= T:
            break
        if row["is_hit"] and row["is_go"]:
            c = "green"
        elif row["is_hit"] and row["is_catch"]:
            c = "darkorange"
        elif row["is_fa"]:
            c = "red"
        elif row["is_miss"]:
            c = "mediumpurple"
        else:
            c = "gray"
        ax_top.axvline(i, color=c, linewidth=0.8, alpha=0.6)
    ax_top.set_ylabel("Outcome")
    ax_top.set_yticks([])

    # Middle: Viterbi (MAP) state strip
    for i in range(T):
        ax_vit.axvline(i, color=palette[ml_states[i]], linewidth=0.8, alpha=0.9)
    ax_vit.set_yticks([])
    ax_vit.set_ylabel("MAP", fontsize=7, labelpad=2)

    # Bottom: individual state posterior lines
    for k in range(K):
        ax_bot.plot(trial_idx, posteriors[:, k],
                    color=palette[k], linewidth=1.5, label=state_labels[k])

    # Mark MAP state transitions with dashed vertical lines
    for t in range(1, T):
        if ml_states[t] != ml_states[t - 1]:
            ax_bot.axvline(t, color="k", linestyle="--", linewidth=0.6, alpha=0.5)

    ax_bot.set_ylim(-0.02, 1.02)
    ax_bot.set_xlabel("Trial #")
    ax_bot.set_ylabel("p(state)")
    ax_bot.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"{session_name}", fontsize=12)
    despine(ax_top)
    despine(ax_vit)
    despine(ax_bot)
    plt.tight_layout()
    sess_dir = out_dir / "session_states"
    sess_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(sess_dir / f"{session_name}_states.png", dpi=120)
    plt.close(fig)


def plot_learning_state_fractions(
    assignments_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    state_labels: list,
    n_states: int,
    out_dir: Path,
):
    """Stacked bar of state fractions across sessions (learning dynamics)."""
    palette = _state_palette(state_labels)

    # Compute per-session fractions
    frac_rows = []
    for sname, grp in assignments_df.groupby("session_name", sort=False):
        total = len(grp)
        for k in range(n_states):
            frac_rows.append({
                "session_name": sname,
                "state": k,
                "label": state_labels[k],
                "fraction": (grp["hmm_state"] == k).sum() / max(total, 1),
            })
    frac_df = pd.DataFrame(frac_rows)

    # Order by date
    session_order = manifest_df.sort_values("parsed_date")["session_name"].values
    # Keep only sessions present in assignments
    session_order = [s for s in session_order if s in frac_df["session_name"].values]

    fig, ax = plt.subplots(figsize=(max(10, len(session_order) * 0.35), 5))
    x = np.arange(len(session_order))
    bottoms = np.zeros(len(session_order))

    for k in range(n_states):
        vals = []
        for sname in session_order:
            sub = frac_df[(frac_df["session_name"] == sname) & (frac_df["state"] == k)]
            vals.append(sub["fraction"].values[0] if len(sub) > 0 else 0.0)
        vals = np.array(vals)
        ax.bar(x, vals, bottom=bottoms, color=palette[k], label=state_labels[k],
               edgecolor="white", linewidth=0.5)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(session_order, rotation=90, fontsize=7)
    ax.set_ylabel("Fraction of trials")
    ax.set_ylim(0, 1)
    ax.legend(title="State", loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_title("Behavioral State Fractions Across Learning")
    despine(ax)
    plt.tight_layout()
    fig.savefig(out_dir / "learning_state_fractions.png", dpi=150)
    plt.close(fig)


def _state_palette(state_labels: list) -> list:
    """Color palette for states using canonical HMM_STATE_COLORS."""
    fallback = ["#7570b3", "#1b9e77", "#d95f02", "#e7298a", "#66a61e", "#e6ab02"]
    return [HMM_STATE_COLORS.get(lab, fallback[i % len(fallback)])
            for i, lab in enumerate(state_labels)]


def _posteriors_from_assign_df(
    assign_df: pd.DataFrame, K: int, session_name: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract (T, K) posteriors array + df for one session from assignments CSV."""
    sub = assign_df[assign_df["session_name"] == session_name].reset_index(drop=True)
    p_cols = [f"p_state_{k}" for k in range(K)]
    posteriors = sub[p_cols].values  # (T, K)
    return posteriors, sub


# =====================================================================
# Replot from saved artefacts
# =====================================================================

def replot_from_saved(args):
    """Regenerate all plots from previously saved models + CSVs.

    Requires --data-out to point to a directory containing:
      model_selection.csv, model_K{n}.pkl, state_labels_K{n}.json,
      state_assignments_K{n}.csv for each K value.
    """
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_out = Path(args.data_out)

    set_style(context="talk")

    # Load manifest for session ordering (centralized SESSION_FILTER)
    manifest = load_staging_manifest(manifest_path=args.manifest)
    subject = manifest.iloc[0].get("session_name", "").split("_")[0] if len(manifest) > 0 else ""
    manifest["parsed_date"] = manifest["session_name"].apply(
        lambda x: _parse_date(x, subject)
    )
    manifest = manifest.sort_values("parsed_date").reset_index(drop=True)

    # Discover saved K values
    selection_path = data_out / "model_selection.csv"
    if not selection_path.exists():
        print(f"ERROR: {selection_path} not found. Run fitting first.")
        sys.exit(1)
    selection_df = pd.read_csv(selection_path)
    best_K = int(selection_df.loc[selection_df["bic"].idxmin(), "K"])

    # Collect all K values that have saved artefacts
    all_K = sorted([int(r["K"]) for _, r in selection_df.iterrows()
                    if (data_out / f"model_K{int(r['K'])}.pkl").exists()])
    if not all_K:
        print("ERROR: No saved model pkl files found. Run fitting first.")
        sys.exit(1)

    print(f"Replotting from saved data: K values = {all_K}, best K = {best_K}")
    print(f"  Data dir:   {data_out}")
    print(f"  Figure dir: {out_dir}")

    # 1. Model selection plot
    plot_model_selection(selection_df, out_dir)

    # 2. Per-K plots
    for K_val in all_K:
        kmodel = GLMHMM.load(data_out / f"model_K{K_val}.pkl")
        # Re-derive labels from the loaded model (not from stale JSON)
        k_labels = auto_label_states(kmodel)
        # Refresh JSON with corrected labels
        with open(data_out / f"state_labels_K{K_val}.json", "w") as f:
            json.dump({"K": K_val, "labels": k_labels}, f, indent=2)
        k_assign_df = pd.read_csv(
            data_out / f"state_assignments_K{K_val}.csv",
            dtype={"session_name": str},
        )
        # Refresh hmm_state_label column using corrected labels
        if "hmm_state" in k_assign_df.columns:
            k_assign_df["hmm_state_label"] = k_assign_df["hmm_state"].map(
                lambda s: k_labels[int(s)]
            )
            k_assign_df.to_csv(data_out / f"state_assignments_K{K_val}.csv", index=False)

        k_dir = out_dir / f"K{K_val}"
        k_dir.mkdir(parents=True, exist_ok=True)
        tag = " (best)" if K_val == best_K else ""
        print(f"  Plotting K={K_val}{tag} -> {k_dir}")

        plot_state_psychometrics(kmodel, k_labels, k_dir)
        plot_transition_matrix(kmodel, k_labels, k_dir)
        plot_glm_weights(kmodel, k_labels, k_dir)

        # Per-session state timecourses from saved posteriors
        for sname in k_assign_df["session_name"].unique():
            post_k, sub_df = _posteriors_from_assign_df(k_assign_df, K_val, sname)
            if len(post_k) > 0:
                plot_session_states(post_k, sub_df, sname, k_labels, k_dir)

        plot_learning_state_fractions(
            k_assign_df, manifest, k_labels, K_val, k_dir
        )

        # Also put best-K plots in root out_dir
        if K_val == best_K:
            plot_state_psychometrics(kmodel, k_labels, out_dir)
            plot_transition_matrix(kmodel, k_labels, out_dir)
            plot_glm_weights(kmodel, k_labels, out_dir)
            for sname in k_assign_df["session_name"].unique():
                post_k, sub_df = _posteriors_from_assign_df(k_assign_df, K_val, sname)
                if len(post_k) > 0:
                    plot_session_states(post_k, sub_df, sname, k_labels, out_dir)
            plot_learning_state_fractions(
                k_assign_df, manifest, k_labels, K_val, out_dir
            )

    print(f"\nReplot complete. Figures saved to: {out_dir}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fit Bernoulli GLM-HMM for behavioral state identification."
    )
    parser.add_argument("--manifest", default=None,
                        help="Path to staging manifest CSV (default: canonical path).")
    parser.add_argument("--pkl-dir", required=True,
                        help="Directory containing session .pkl files.")
    parser.add_argument("--out", default="FIGURES/behavior/hmm",
                        help="Output directory for figures.")
    parser.add_argument("--data-out", default="data/hmm",
                        help="Output directory for model / CSV artefacts.")
    parser.add_argument("--K-min", type=int, default=2,
                        help="Minimum number of states.")
    parser.add_argument("--K-max", type=int, default=5,
                        help="Maximum number of states.")
    parser.add_argument("--n-restarts", type=int, default=20,
                        help="Random restarts per K.")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="Max EM iterations per restart.")
    parser.add_argument("--exclude-qc-fail", action="store_true",
                        help="DEPRECATED: SESSION_FILTER handles QC. Kept for backward compat.")
    parser.add_argument("--no-filter", action="store_true",
                        help="Bypass SESSION_FILTER and use the full manifest.")
    parser.add_argument("--replot-only", action="store_true",
                        help="Regenerate plots from saved artefacts (no refitting).")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Number of parallel workers for K fitting (default: 1).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed for reproducibility (default: 0).")
    parser.add_argument("--cv", action="store_true",
                        help="Run LOSO cross-validation on the best-K model after fitting.")
    parser.add_argument("--cv-restarts", type=int, default=5,
                        help="Random restarts per LOSO fold (default: 5).")
    args = parser.parse_args()

    # ---- Replot-only mode: skip fitting, load saved data ----
    if args.replot_only:
        replot_from_saved(args)
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_out = Path(args.data_out)
    data_out.mkdir(parents=True, exist_ok=True)

    set_style(context="talk")

    # ------------------------------------------------------------------
    # 1. Load manifest & sessions
    # ------------------------------------------------------------------
    manifest = load_staging_manifest(
        manifest_path=args.manifest,
        apply_filter=not getattr(args, 'no_filter', False),
    )

    # Parse dates for ordering
    subject = manifest.iloc[0].get("session_name", "").split("_")[0] if len(manifest) > 0 else ""
    manifest["parsed_date"] = manifest["session_name"].apply(
        lambda x: _parse_date(x, subject)
    )
    manifest = manifest.sort_values("parsed_date").reset_index(drop=True)

    pkl_dir = Path(args.pkl_dir)
    sessions_data = []
    session_names_loaded = []

    print(f"Loading sessions from manifest ({len(manifest)} entries)...")
    for _, row in manifest.iterrows():
        sname = str(row["session_name"])
        # Resolve pkl path
        if "path" in row and pd.notna(row["path"]):
            pkl_path = Path(row["path"])
            if not pkl_path.exists():
                pkl_path = pkl_dir / pkl_path.name
        else:
            candidates = list(pkl_dir.glob(f"*{sname}*.pkl"))
            pkl_path = candidates[0] if candidates else None

        if pkl_path is None or not pkl_path.exists():
            print(f"  SKIP {sname}: pkl not found")
            continue

        try:
            session = load_session(str(pkl_path))
            sd = prepare_session_data(session)
            if len(sd["y"]) < 10:
                print(f"  SKIP {sname}: only {len(sd['y'])} valid trials")
                continue
            sd["session_name"] = sname
            sessions_data.append(sd)
            session_names_loaded.append(sname)
        except Exception as exc:
            print(f"  SKIP {sname}: {exc}")
            continue

    total_trials = sum(len(s["y"]) for s in sessions_data)
    print(f"\nLoaded {len(sessions_data)} sessions  ({total_trials} total trials)")

    if len(sessions_data) == 0:
        print("ERROR: No valid sessions. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Fit models (model selection)
    # ------------------------------------------------------------------
    K_range = list(range(args.K_min, args.K_max + 1))

    config = GLMHMMConfig(
        max_iter=args.max_iter,
        n_restarts=args.n_restarts,
        verbose=True,
    )

    best_model, selection_df, all_models = fit_best_model(
        sessions_data, K_range=K_range, config=config, verbose=True,
        n_workers=args.n_workers, seed=args.seed,
    )
    state_labels = auto_label_states(best_model)
    best_K = best_model.n_states

    # ------------------------------------------------------------------
    # 3. Save artefacts
    # ------------------------------------------------------------------

    # Model selection table
    selection_df.to_csv(data_out / "model_selection.csv", index=False)

    # Save ALL fitted models (pkl + assignments + labels) so downstream
    # scripts can load any K with  --K <n>.
    for K_val, kmodel in sorted(all_models.items()):
        k_labels = auto_label_states(kmodel)
        kmodel.save(data_out / f"model_K{K_val}.pkl")

        # Per-K state labels
        with open(data_out / f"state_labels_K{K_val}.json", "w") as f:
            json.dump({"K": K_val, "labels": k_labels}, f, indent=2)

        # Per-K trial state assignments
        k_assign_rows = []
        for sd in sessions_data:
            states_k = kmodel.most_likely_states(sd)
            posteriors_k = kmodel.state_posteriors(sd)
            df_k = sd["df"].copy()
            df_k.insert(0, "session_name", sd["session_name"])
            df_k["hmm_state"] = states_k
            df_k["hmm_state_label"] = [k_labels[s] for s in states_k]
            for ki in range(K_val):
                df_k[f"p_state_{ki}"] = posteriors_k[:, ki]
            k_assign_rows.append(df_k)
        k_assign_df = pd.concat(k_assign_rows, ignore_index=True)
        k_assign_df.to_csv(data_out / f"state_assignments_K{K_val}.csv", index=False)
        tag = " (best)" if K_val == best_K else ""
        print(f"  Saved K={K_val}{tag}: model pkl + assignments + labels")

    # Also write convenience aliases for the best K (backwards compat)
    state_labels = auto_label_states(best_model)
    labels_path = data_out / "state_labels.json"
    with open(labels_path, "w") as f:
        json.dump({"K": best_K, "labels": state_labels}, f, indent=2)

    all_assignments = []
    for sd in sessions_data:
        sname = sd["session_name"]
        states = best_model.most_likely_states(sd)
        posteriors = best_model.state_posteriors(sd)
        df = sd["df"].copy()
        df.insert(0, "session_name", sname)
        df["hmm_state"] = states
        df["hmm_state_label"] = [state_labels[s] for s in states]
        for k in range(best_K):
            df[f"p_state_{k}"] = posteriors[:, k]
        all_assignments.append(df)

    assignments_df = pd.concat(all_assignments, ignore_index=True)
    assign_path = data_out / "state_assignments.csv"
    assignments_df.to_csv(assign_path, index=False)
    print(f"\nBest-K state assignments saved: {assign_path}  ({len(assignments_df)} trials)")

    # ------------------------------------------------------------------
    # 3b. LOSO cross-validation (optional)
    # ------------------------------------------------------------------
    if args.cv:
        print(f"\n{'=' * 60}")
        print(f"Running LOSO cross-validation  K={best_K}  "
              f"({args.cv_restarts} restarts/fold)")
        print("=" * 60)
        cv_df = loso_cross_validation(
            sessions_data, best_K,
            n_restarts=args.cv_restarts,
            max_iter=args.max_iter,
            seed=args.seed,
            verbose=True,
        )
        cv_path = data_out / f"cv_results_K{best_K}.csv"
        cv_df.to_csv(cv_path, index=False)
        print(f"\nCV results saved: {cv_path}")

        # Per-stage summary
        manifest_cv = manifest[["session_name", "stage"]].copy()
        manifest_cv["session_name"] = manifest_cv["session_name"].astype(str)
        cv_df["held_out_session"] = cv_df["held_out_session"].astype(str)
        cv_merged = cv_df.merge(
            manifest_cv, left_on="held_out_session", right_on="session_name", how="left"
        )
        print("\nPer-stage CV summary:")
        for stage in cv_merged["stage"].dropna().unique():
            sub = cv_merged[cv_merged["stage"] == stage]
            mean_ll = sub["test_ll_per_trial"].mean()
            sem_ll = sub["test_ll_per_trial"].sem()
            mean_acc = sub["test_accuracy"].mean()
            print(f"  {stage:12s}: LL/trial = {mean_ll:.4f} ± {sem_ll:.4f}  "
                  f"accuracy = {mean_acc:.3f}  (n={len(sub)} sessions)")

    # ------------------------------------------------------------------
    # 4. Generate plots
    # ------------------------------------------------------------------
    print("\nGenerating plots...")

    # 4a. Model selection (shared — goes into root out_dir)
    plot_model_selection(selection_df, out_dir)

    # 4b. Per-K plots: psychometrics, transition matrix, GLM weights,
    #     session timecourses, and learning fractions in K{n}/ sub-dirs.
    for K_val, kmodel in sorted(all_models.items()):
        k_labels = auto_label_states(kmodel)
        k_dir = out_dir / f"K{K_val}"
        k_dir.mkdir(parents=True, exist_ok=True)
        tag = " (best)" if K_val == best_K else ""
        print(f"  Plotting K={K_val}{tag} -> {k_dir}")

        plot_state_psychometrics(kmodel, k_labels, k_dir)
        plot_transition_matrix(kmodel, k_labels, k_dir)
        plot_glm_weights(kmodel, k_labels, k_dir)

        # Per-session state timecourses
        for sd in sessions_data:
            post_k = kmodel.state_posteriors(sd)
            plot_session_states(post_k, sd["df"], sd["session_name"], k_labels, k_dir)

        # Learning state fractions — load per-K assignments already saved
        k_assign_df = pd.read_csv(
            data_out / f"state_assignments_K{K_val}.csv",
            dtype={"session_name": str},
        )
        plot_learning_state_fractions(
            k_assign_df, manifest, k_labels, K_val, k_dir
        )

    # Also put best-model plots in root out_dir for convenience
    plot_state_psychometrics(best_model, state_labels, out_dir)
    plot_transition_matrix(best_model, state_labels, out_dir)
    plot_glm_weights(best_model, state_labels, out_dir)
    for sd in sessions_data:
        post_best = best_model.state_posteriors(sd)
        plot_session_states(post_best, sd["df"], sd["session_name"], state_labels, out_dir)
    plot_learning_state_fractions(
        assignments_df, manifest, state_labels, best_K, out_dir
    )

    # 4g. Summary statistics
    print("\n" + "=" * 60)
    print("HMM FIT SUMMARY")
    print("=" * 60)
    print(best_model.summary())
    print(f"\nState labels: {state_labels}")
    print(f"\nPer-session state fractions:")
    for sname in session_names_loaded:
        sub = assignments_df[assignments_df["session_name"] == sname]
        fracs = []
        for k in range(best_K):
            f_k = (sub["hmm_state"] == k).mean()
            fracs.append(f"{state_labels[k]}={f_k:.2f}")
        print(f"  {sname}: {', '.join(fracs)}")

    print(f"\nAll outputs saved to:\n  Figures: {out_dir}\n  Data:    {data_out}")
    print("Done.")


if __name__ == "__main__":
    main()
