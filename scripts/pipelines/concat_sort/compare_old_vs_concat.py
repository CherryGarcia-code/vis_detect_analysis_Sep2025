#!/usr/bin/env python3
"""Compare original vs concat-sort pkls: unit counts, QC metrics, and selection.

Runs run_qc() and run_unit_selection() on both the original and concat-sort pkl
for a session and produces a side-by-side summary figure + CSV tables.

Output goes to FIGURES/concat_sort_qc/<session_name>/{original,concat_sort}/.

Usage:
    python scripts/pipelines/concat_sort/compare_old_vs_concat.py BG_046_01072025
    python scripts/pipelines/concat_sort/compare_old_vs_concat.py --all
"""

import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from visdetect.core.session import load_session
from visdetect.core.qc import run_qc, run_unit_selection

# ── Paths ─────────────────────────────────────────────────────────────
SUBJECT = "BG_046"
OLD_PKL_DIR = REPO_ROOT / "data" / "pkls" / SUBJECT
NEW_PKL_DIR = REPO_ROOT / "data" / "pkls" / f"{SUBJECT}_concat_sort"
OUT_BASE = REPO_ROOT / "FIGURES" / "concat_sort_qc"


def compare_session(session_name: str):
    """Run QC + unit selection on both pkls and produce comparison output."""
    old_pkl = OLD_PKL_DIR / f"{session_name}.pkl"
    new_pkl = NEW_PKL_DIR / f"{session_name}.pkl"

    if not old_pkl.exists():
        print(f"  SKIP: original pkl not found: {old_pkl.name}")
        return None
    if not new_pkl.exists():
        print(f"  SKIP: concat-sort pkl not found: {new_pkl.name}")
        return None

    out_dir = OUT_BASE / session_name
    old_dir = out_dir / "original"
    new_dir = out_dir / "concat_sort"

    # ── Load sessions ─────────────────────────────────────────────────
    print(f"  Loading original pkl ...")
    old_sess = load_session(str(old_pkl))
    print(f"  Loading concat-sort pkl ...")
    new_sess = load_session(str(new_pkl))

    # ── Run QC ────────────────────────────────────────────────────────
    print(f"  Running QC (original) ...")
    run_qc(old_sess, str(old_dir))
    print(f"  Running QC (concat-sort) ...")
    run_qc(new_sess, str(new_dir))

    # ── Run unit selection (same params for fair comparison) ──────────
    sel_params = dict(
        event_name="Change_ON",
        window=(-0.5, 1.0),
        bin_size=0.02,
        require_good_cluster=True,
        min_total_spikes=500,
        min_mean_rate_hz=0.1,
        max_isi_viol_frac=0.2,
        min_median_spikes_per_trial=0.1,
    )
    print(f"  Running unit selection (original) ...")
    old_res = run_unit_selection(old_sess, str(old_dir), **sel_params)
    print(f"  Running unit selection (concat-sort) ...")
    new_res = run_unit_selection(new_sess, str(new_dir), **sel_params)

    # ── Build summary comparison ──────────────────────────────────────
    old_df = pd.read_csv(old_dir / "unit_selection.csv")
    new_df = pd.read_csv(new_dir / "unit_selection.csv")

    summary = {
        "session": session_name,
        "original": _summarise(old_sess, old_df, old_res),
        "concat_sort": _summarise(new_sess, new_df, new_res),
    }
    with (out_dir / "comparison_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # ── Side-by-side figure ───────────────────────────────────────────
    _plot_comparison(old_df, new_df, session_name, out_dir)

    return summary


def _summarise(session, filt_df, result):
    """Extract key numbers for one pipeline variant."""
    good_stable = getattr(session, "good_and_stable_ids", None) or []
    return {
        "n_clusters": len(session.clusters),
        "n_good_and_stable": len(good_stable),
        "n_good_cluster_ids": len(session.good_cluster_ids) if session.good_cluster_ids else 0,
        "n_kept": result["n_kept"],
        "n_total": result["n_total"],
        "median_rate_hz": float(filt_df["mean_rate_hz"].median()),
        "median_isi_viol": float(filt_df["isi_violations_frac"].median()),
    }


def _plot_comparison(old_df, new_df, session_name, out_dir):
    """Side-by-side histograms for key metrics."""
    metrics = [
        ("mean_rate_hz", "Firing rate (Hz)", True),
        ("isi_violations_frac", "ISI violation frac", False),
        ("n_spikes", "Total spikes", True),
        ("median_spikes_per_trial", "Median spikes / trial", False),
    ]

    fig, axes = plt.subplots(2, len(metrics), figsize=(4 * len(metrics), 7),
                             gridspec_kw={"hspace": 0.4, "wspace": 0.35})

    for col, (metric, label, use_log) in enumerate(metrics):
        for row, (df, tag, color) in enumerate([
            (old_df, "Original", "C0"),
            (new_df, "Concat-sort", "C1"),
        ]):
            ax = axes[row, col]
            kept = df.loc[df["keep"], metric].dropna()
            dropped = df.loc[~df["keep"], metric].dropna()
            bins = 40
            if use_log and kept.size + dropped.size > 0:
                all_vals = pd.concat([kept, dropped])
                pos = all_vals[all_vals > 0]
                if len(pos) > 0:
                    bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 40)
            ax.hist(dropped, bins=bins, alpha=0.5, color="C3", label="dropped")
            ax.hist(kept, bins=bins, alpha=0.5, color=color, label="kept")
            if use_log:
                ax.set_xscale("log")
            ax.set_xlabel(label)
            ax.set_ylabel("Units")
            ax.set_title(f"{tag} (kept={len(kept)})")
            ax.legend(fontsize="x-small")

    fig.suptitle(f"Unit comparison: {session_name}", fontsize=13, y=1.0)
    fig.savefig(out_dir / "comparison_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Scatter: rate vs ISI for both ─────────────────────────────────
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, df, tag in [(ax1, old_df, "Original"), (ax2, new_df, "Concat-sort")]:
        c = np.where(df["keep"], "C0", "C3")
        ax.scatter(df["mean_rate_hz"], df["isi_violations_frac"],
                   c=c, s=8, alpha=0.6)
        ax.set_xlabel("Firing rate (Hz)")
        ax.set_ylabel("ISI violation frac")
        n_kept = df["keep"].sum()
        ax.set_title(f"{tag}  (kept={n_kept} / {len(df)})")
        ax.set_xscale("log")
        ax.axhline(0.2, ls="--", c="gray", lw=0.7, alpha=0.5)
        ax.axvline(0.1, ls="--", c="gray", lw=0.7, alpha=0.5)
    fig2.suptitle(f"Rate vs ISI: {session_name}", y=1.02)
    fig2.tight_layout()
    fig2.savefig(out_dir / "comparison_rate_vs_isi.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"  Figures saved to {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare original vs concat-sort pkls")
    parser.add_argument("sessions", nargs="*", help="Session names (e.g. BG_046_01072025)")
    parser.add_argument("--all", action="store_true", help="Compare all sessions that have both pkls")
    args = parser.parse_args()

    if args.all:
        sessions = sorted(
            p.stem for p in NEW_PKL_DIR.glob("*.pkl")
            if (OLD_PKL_DIR / p.name).exists()
        )
    elif args.sessions:
        sessions = args.sessions
    else:
        parser.print_help()
        return

    print(f"Comparing {len(sessions)} session(s)\n")
    results = []
    for sess in sessions:
        print(f"[{sess}]")
        r = compare_session(sess)
        if r:
            results.append(r)
            o = r["original"]
            n = r["concat_sort"]
            print(f"  Original:    {o['n_clusters']} clusters, {o['n_kept']} kept")
            print(f"  Concat-sort: {n['n_clusters']} clusters, {n['n_kept']} kept")
        print()

    # Print overall summary table
    if results:
        print("=" * 70)
        print(f"{'Session':<22} {'Orig kept':>10} {'Concat kept':>12} {'Orig clust':>11} {'Concat clust':>13}")
        print("-" * 70)
        for r in results:
            print(f"{r['session']:<22} {r['original']['n_kept']:>10} "
                  f"{r['concat_sort']['n_kept']:>12} "
                  f"{r['original']['n_clusters']:>11} "
                  f"{r['concat_sort']['n_clusters']:>13}")


if __name__ == "__main__":
    main()
