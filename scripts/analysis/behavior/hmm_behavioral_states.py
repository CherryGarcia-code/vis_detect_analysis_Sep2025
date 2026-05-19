"""Per-state behavioral metrics and across-learning dynamics.

Loads the fitted HMM results and generates:
  1. aggregate_state_metrics.csv  – SDT metrics per state (d', criterion, etc.)
  2. per_session_state_metrics.csv – metrics broken out by session × state
  3. learning_trajectory.csv – state fractions + d' across sessions
  4. Plots: state_behavioral_summary.png, learning_dprime_by_state.png,
     learning_state_fractions_dprime.png

Usage
-----
    python scripts/analysis/behavior/hmm_behavioral_states.py \
        --data-dir data/hmm/BG_046 \
        --manifest data/BG_046_staging_manifest_v2.csv \
        --out      FIGURES/behavior/BG_046/hmm/behavioral_states \
        --data-out data/hmm/BG_046 \
        --exclude-qc-fail
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


from visdetect.analysis.config import load_staging_manifest
from visdetect.analysis.hmm_downstream import (
    compute_learning_trajectory,
    compute_per_session_state_metrics,
    compute_state_behavioral_metrics,
    load_hmm_results,
)
from visdetect.viz.plotting import set_style, despine


def _parse_date(session_name: str) -> datetime:
    try:
        return datetime.strptime(session_name.split("_")[-1], "%d%m%Y")
    except Exception:
        return datetime.min


def _state_palette(K):
    base = ["#7570b3", "#1b9e77", "#d95f02", "#e7298a", "#66a61e", "#e6ab02"]
    return base[:K]


def plot_aggregate_metrics(metrics_df: pd.DataFrame, out_dir: Path):
    """Bar chart of d', hit rate, FA rate per state."""
    K = len(metrics_df)
    palette = _state_palette(K)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    # d'
    ax = axes[0]
    ax.bar(range(K), metrics_df["dprime"].fillna(0), color=palette, edgecolor="k")
    ax.set_xticks(range(K))
    ax.set_xticklabels(metrics_df["label"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("d'")
    ax.set_title("Sensitivity (d')")
    ax.axhline(0, color="k", linewidth=0.5)
    despine(ax)

    # Hit rate
    ax = axes[1]
    ax.bar(range(K), metrics_df["hit_rate_go"].fillna(0), color=palette, edgecolor="k")
    ax.set_xticks(range(K))
    ax.set_xticklabels(metrics_df["label"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Hit rate (go trials)")
    ax.set_title("Hit Rate")
    ax.set_ylim(0, 1)
    despine(ax)

    # Catch lick rate
    ax = axes[2]
    ax.bar(range(K), metrics_df["catch_lick_rate"].fillna(0), color=palette, edgecolor="k")
    ax.set_xticks(range(K))
    ax.set_xticklabels(metrics_df["label"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Catch lick rate")
    ax.set_title("False Alarm Rate (catch)")
    ax.set_ylim(0, 1)
    despine(ax)

    # Early lick rate
    ax = axes[3]
    ax.bar(range(K), metrics_df["early_lick_rate"].fillna(0), color=palette, edgecolor="k")
    ax.set_xticks(range(K))
    ax.set_xticklabels(metrics_df["label"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Early lick rate")
    ax.set_title("Early/Anticipatory Lick Rate")
    ax.set_ylim(0, 1)
    despine(ax)

    fig.suptitle("Aggregate Behavioral Metrics per HMM State", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "state_behavioral_summary.png", dpi=150)
    plt.close(fig)


def plot_learning_dprime(
    trajectory_df: pd.DataFrame,
    state_labels: list,
    n_states: int,
    out_dir: Path,
):
    """d' per state across sessions (learning curve)."""
    palette = _state_palette(n_states)
    sessions = trajectory_df["session_name"].values

    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(sessions) * 0.4), 9),
                             sharex=True, gridspec_kw={"height_ratios": [1, 2]})

    # Top panel: overall d'
    ax = axes[0]
    ax.plot(range(len(sessions)), trajectory_df["overall_dprime"], "k-o",
            linewidth=2, markersize=5, label="Overall")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("d'")
    ax.set_title("Overall Sensitivity Across Learning")
    ax.legend(fontsize=8)
    despine(ax)

    # Bottom panel: per-state d'
    ax = axes[1]
    for k in range(n_states):
        lbl = state_labels[k]
        col = f"dprime_{lbl}"
        if col in trajectory_df.columns:
            vals = trajectory_df[col].values
            ax.plot(range(len(sessions)), vals, "o-", color=palette[k],
                    linewidth=1.5, markersize=4, label=lbl, alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xticks(range(len(sessions)))
    ax.set_xticklabels(sessions, rotation=90, fontsize=7)
    ax.set_ylabel("d' (within state)")
    ax.set_xlabel("Session")
    ax.set_title("Per-State Sensitivity Across Learning")
    ax.legend(title="State", fontsize=8, ncol=min(n_states, 3))
    despine(ax)

    plt.tight_layout()
    fig.savefig(out_dir / "learning_dprime_by_state.png", dpi=150)
    plt.close(fig)


def plot_combined_learning(
    trajectory_df: pd.DataFrame,
    state_labels: list,
    n_states: int,
    out_dir: Path,
):
    """Combined plot: state fractions (top) + overall d' (bottom)."""
    palette = _state_palette(n_states)
    sessions = trajectory_df["session_name"].values
    n_sess = len(sessions)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, n_sess * 0.4), 8),
                                    sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # State fractions stacked bar
    x = np.arange(n_sess)
    bottoms = np.zeros(n_sess)
    for k in range(n_states):
        lbl = state_labels[k]
        vals = trajectory_df[f"frac_{lbl}"].fillna(0).values
        ax1.bar(x, vals, bottom=bottoms, color=palette[k], label=lbl,
                edgecolor="white", linewidth=0.5)
        bottoms += vals
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Fraction of trials")
    ax1.set_title("HMM State Fractions & Overall Sensitivity Across Learning")
    ax1.legend(title="State", loc="upper right", fontsize=8, ncol=min(n_states, 3))
    despine(ax1)

    # Overall d'
    ax2.plot(x, trajectory_df["overall_dprime"], "ko-", linewidth=2, markersize=5)
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sessions, rotation=90, fontsize=7)
    ax2.set_ylabel("Overall d'")
    ax2.set_xlabel("Session")
    despine(ax2)

    plt.tight_layout()
    fig.savefig(out_dir / "learning_state_fractions_dprime.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Behavioral metrics and learning dynamics per HMM state."
    )
    parser.add_argument("--data-dir", required=True,
                        help="Directory with model pkl, state_assignments.csv, labels.")
    parser.add_argument("--K", type=int, default=None,
                        help="Number of states to load (e.g. 3). "
                             "Default: highest-K model found on disk.")
    parser.add_argument("--manifest", default=None,
                        help="Staging manifest CSV (for session ordering).")
    parser.add_argument("--out", default="FIGURES/behavior/hmm",
                        help="Output directory for plots.")
    parser.add_argument("--data-out", default=None,
                        help="Output directory for CSVs (defaults to --data-dir).")
    parser.add_argument("--exclude-qc-fail", action="store_true",
                        help="DEPRECATED: SESSION_FILTER handles QC.")
    parser.add_argument("--no-filter", action="store_true",
                        help="Bypass SESSION_FILTER and use the full manifest.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_out = Path(args.data_out) if args.data_out else data_dir
    data_out.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style(context="talk")

    # Load
    model, assignments_df, state_labels = load_hmm_results(data_dir, K=args.K)
    K = model.n_states
    print(f"Loaded K={K} model, {len(assignments_df)} trials, labels={state_labels}")

    # Session ordering from manifest if available
    session_order = None
    if args.manifest or True:  # Always load manifest for filtering
        manifest = load_staging_manifest(
            manifest_path=args.manifest,
            apply_filter=not getattr(args, 'no_filter', False),
        )
        manifest["parsed_date"] = manifest["session_name"].apply(_parse_date)
        manifest = manifest.sort_values("parsed_date")
        session_order = manifest["session_name"].tolist()
        # keep only sessions present in assignments
        present = set(assignments_df["session_name"].unique())
        session_order = [s for s in session_order if s in present]

    # ------------------------------------------------------------------
    # 1. Aggregate state metrics
    # ------------------------------------------------------------------
    agg = compute_state_behavioral_metrics(assignments_df, state_labels, K)
    agg_path = data_out / "aggregate_state_metrics.csv"
    agg.to_csv(agg_path, index=False)
    print(f"\nAggregate metrics saved: {agg_path}")
    print(agg.to_string(index=False))

    # ------------------------------------------------------------------
    # 2. Per-session × state metrics
    # ------------------------------------------------------------------
    per_sess = compute_per_session_state_metrics(assignments_df, state_labels, K)
    per_sess_path = data_out / "per_session_state_metrics.csv"
    per_sess.to_csv(per_sess_path, index=False)
    print(f"\nPer-session metrics saved: {per_sess_path}")

    # ------------------------------------------------------------------
    # 3. Learning trajectory
    # ------------------------------------------------------------------
    trajectory = compute_learning_trajectory(
        assignments_df, state_labels, K, session_order=session_order
    )
    traj_path = data_out / "learning_trajectory.csv"
    trajectory.to_csv(traj_path, index=False)
    print(f"Learning trajectory saved: {traj_path}")

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    print("\nGenerating plots...")
    plot_aggregate_metrics(agg, out_dir)
    plot_learning_dprime(trajectory, state_labels, K, out_dir)
    plot_combined_learning(trajectory, state_labels, K, out_dir)
    print(f"Plots saved to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
