#!/usr/bin/env python
"""Quick overlap demo on a small real-session subset.

Loads 2+ session pickle files, trims to a limited number of good clusters
(or first N clusters if good list absent), computes lick responsiveness and
TF pulse responsiveness in-memory (no large grids) and prints fraction stats:

  - lick_only
  - tf_only (fast OR slow responsive at |z| >= z_thresh)
  - both
  - neither

Also saves a summary CSV for downstream reference.

Usage (example):
    python scripts/demo_overlap_subset.py \
        --sessions pkls/BG_046_01072025.pkl pkls/BG_046_16092025.pkl \
        --max-clusters 40 --z-thresh 3.0 --out-dir table_output/demo_overlap_subset

This is intentionally lightweight so you can validate logic before running
full-session batch screens.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from dataclasses import replace
from datetime import datetime
import pandas as pd
import numpy as np

# Ensure src/ on path for direct module imports
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from session_io import load_session, Session, Cluster  # type: ignore
from visdetect.analysis.su_analysis import load_kept_ids, selection_csv_default_path

# Import via package path
from visdetect.analysis.lick import MatlabLickConfig, MatlabLickAnalyzer
from visdetect.analysis.tf_pulse import TFRespPulseConfig, collect_tf_pulse_traces


def _subset_session(sess: Session, max_clusters: int) -> Session:
    """Return a shallow-copied Session with clusters/good_cluster_ids trimmed."""
    if max_clusters <= 0:
        return sess
    # Determine candidate cluster IDs (prefer good_cluster_ids ordering)
    if sess.good_cluster_ids:
        keep_ids = sorted(sess.good_cluster_ids)[: max_clusters]
        clusters = [c for c in sess.clusters if int(c.cluster_id) in keep_ids]
    else:
        clusters_sorted = sorted(sess.clusters, key=lambda c: int(c.cluster_id))
        clusters = clusters_sorted[: max_clusters]
        keep_ids = [int(c.cluster_id) for c in clusters]
    # Shallow copy; trials unchanged
    new = Session(
        trials=list(sess.trials),
        clusters=[Cluster(cluster_id=int(c.cluster_id), spike_times=np.array(c.spike_times, dtype=float)) for c in clusters],
        subject=sess.subject,
        session_name=sess.session_name,
        good_cluster_ids=keep_ids,
        ni_events=sess.ni_events,
    )
    return new


def _compute_tf_responsiveness(sess: Session, z_thresh: float) -> pd.DataFrame:
    """Minimal TF responsiveness table using existing trace collection logic.

    Classifies fast/slow responsiveness via bidirectional checks:
      - Fast: |z_max| >= thresh OR |z_min| >= thresh
      - Slow: |z_max| >= thresh OR |z_min| >= thresh
    Returns DataFrame with columns: cluster_id, fast_responsive, slow_responsive, tf_responsive.
    """
    cfg = TFRespPulseConfig(kept_only=True)
    _, entries = collect_tf_pulse_traces(sess, cfg=cfg, selection_csv=None, show_progress=True)
    rows = []
    for e in entries:
        # Bidirectional checks
        fast_exc = bool(np.isfinite(e.z_max_fast) and (e.z_max_fast >= z_thresh))
        fast_inh = bool(np.isfinite(e.z_min_fast) and (e.z_min_fast <= -z_thresh))
        
        slow_exc = bool(np.isfinite(e.z_max_slow) and (e.z_max_slow >= z_thresh))
        slow_inh = bool(np.isfinite(e.z_min_slow) and (e.z_min_slow <= -z_thresh))
        
        fast_resp = fast_exc or fast_inh
        slow_resp = slow_exc or slow_inh
        
        rows.append({
            "cluster_id": int(e.cluster_id),
            "fast_responsive": fast_resp,
            "slow_responsive": slow_resp,
            "tf_responsive": bool(fast_resp or slow_resp),
            "z_max_fast": float(e.z_max_fast) if np.isfinite(e.z_max_fast) else np.nan,
            "z_min_fast": float(e.z_min_fast) if np.isfinite(e.z_min_fast) else np.nan,
            "z_max_slow": float(e.z_max_slow) if np.isfinite(e.z_max_slow) else np.nan,
            "z_min_slow": float(e.z_min_slow) if np.isfinite(e.z_min_slow) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def _compute_lick_responsiveness(sess: Session) -> pd.DataFrame:
    """Compute lick responsiveness using MATLAB-faithful analyzer."""
    cfg = MatlabLickConfig()
    # Use good_cluster_ids from session if available to filter
    analyzer = MatlabLickAnalyzer(cfg=cfg, good_ids=sess.good_cluster_ids)
    res = analyzer.run_session(sess)
    
    df = res.table.copy()
    # Map columns to match expected output
    # MatlabLickResult table has: cluster_id, n_events, baseline_mean, post_mean, delta_mean, p_value, is_significant
    df.rename(columns={
        "delta_mean": "delta_fr",
        "is_significant": "lick_responsive"
    }, inplace=True)
    
    # Ensure required columns exist
    required = ["cluster_id", "delta_fr", "p_value", "lick_responsive"]
    for c in required:
        if c not in df.columns:
            df[c] = np.nan
            
    return df[required].sort_values("cluster_id").reset_index(drop=True)


def analyze_sessions(sessions: list[str], max_clusters: int, z_thresh: float, out_dir: Path, profiles_root: Path | None, profile_name: str | None) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for path in sessions:
        p = Path(path)
        if not p.exists():
            print(f"[WARN] Session missing: {path}")
            continue
        print(f"Loading session: {path}")
        sess = load_session(str(p))
        # If profile QC exists, override subset using kept IDs from selection CSV
        if profiles_root is not None:
            try:
                sel_csv = selection_csv_default_path(sess, root=profiles_root)
                kept_ids = load_kept_ids(sess, selection_csv=str(sel_csv))
            except Exception as e:
                print(f"[WARN] Could not load kept IDs for {path}: {e}")
                kept_ids = []
        else:
            kept_ids = []

        subset = _subset_session(sess, max_clusters=max_clusters if not kept_ids else min(max_clusters, len(kept_ids)))
        if kept_ids:
            subset.good_cluster_ids = kept_ids
        print(f"Subset clusters: {len(subset.clusters)} (max={max_clusters})")

        lick_df = _compute_lick_responsiveness(subset)
        tf_df = _compute_tf_responsiveness(subset, z_thresh=z_thresh)

        merged = pd.merge(lick_df, tf_df, on="cluster_id", how="outer")
        merged["lick_responsive"].fillna(False, inplace=True)
        merged["tf_responsive"].fillna(False, inplace=True)

        n_total = merged.shape[0]
        n_lick = int(merged["lick_responsive"].sum())
        n_tf = int(merged["tf_responsive"].sum())
        n_both = int((merged["lick_responsive"] & merged["tf_responsive"]).sum())
        n_neither = n_total - (n_lick + n_tf - n_both)

        summary_rows.append({
            "session": f"{subset.subject}_{subset.session_name}",
            "n_total": n_total,
            "n_lick_only": n_lick - n_both,
            "n_tf_only": n_tf - n_both,
            "n_both": n_both,
            "n_neither": n_neither,
            "frac_lick_only": (n_lick - n_both) / n_total if n_total else 0.0,
            "frac_tf_only": (n_tf - n_both) / n_total if n_total else 0.0,
            "frac_both": n_both / n_total if n_total else 0.0,
            "frac_neither": n_neither / n_total if n_total else 0.0,
            "z_thresh": z_thresh,
            "max_clusters": max_clusters,
        })

        # Write per-session merged CSV for inspection
        merged_csv = out_dir / f"{subset.subject}_{subset.session_name}_overlap_subset.csv"
        merged.to_csv(merged_csv, index=False)
        print(f"Wrote merged overlap table: {merged_csv}")

    summary = pd.DataFrame(summary_rows).sort_values("session").reset_index(drop=True)
    summary_csv = out_dir / "overlap_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Wrote summary: {summary_csv}")
    return summary


def parse_session_date(session_name: str) -> datetime:
    """Extract date from session name (e.g. BG_046_01072025 -> 2025-07-01)."""
    # Assume format ends with DDMMYYYY
    try:
        parts = session_name.split("_")
        date_str = parts[-1]
        if len(date_str) == 8 and date_str.isdigit():
            return datetime.strptime(date_str, "%d%m%Y")
    except Exception:
        pass
    return datetime.min


def main(argv=None):
    ap = argparse.ArgumentParser(description="Overlap demo on subset of real sessions")
    ap.add_argument("--sessions", nargs="+", help="Paths to session .pkl files")
    ap.add_argument("--max-clusters", type=int, default=40, help="Maximum clusters per session to include")
    ap.add_argument("--z-thresh", type=float, default=3.0, help="|z| threshold for TF responsiveness")
    ap.add_argument("--out-dir", default="table_output/demo_overlap_subset", help="Output directory for summary + merged tables")
    ap.add_argument("--profiles-root", default=None, help="Root directory containing unit_qc outputs (for unit_selection.csv)")
    ap.add_argument("--profile-name", default=None, help="QC profile name (e.g. striatal_strict)")
    ap.add_argument("--plot", action="store_true", help="Generate stacked bar plot of fraction categories")
    ap.add_argument("--plot-only", action="store_true", help="Skip analysis and plot existing summary CSV")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    
    if args.plot_only:
        summary_csv = out_dir / "overlap_summary.csv"
        if not summary_csv.exists():
            print(f"[ERROR] Summary CSV not found at {summary_csv}. Cannot run --plot-only.")
            return 1
        print(f"Loading existing summary from {summary_csv}")
        summary = pd.read_csv(summary_csv)
    else:
        if not args.sessions:
            ap.error("the following arguments are required: --sessions (unless --plot-only is used)")
        
        sessions = args.sessions
        profiles_root = Path(args.profiles_root) if args.profiles_root else None
        summary = analyze_sessions(sessions, max_clusters=int(args.max_clusters), z_thresh=float(args.z_thresh), out_dir=out_dir, profiles_root=profiles_root, profile_name=args.profile_name)

    if (args.plot or args.plot_only) and not summary.empty:
        try:
            # Sort by date
            summary["date"] = summary["session"].apply(parse_session_date)
            summary.sort_values("date", inplace=True)
            
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            cats = ["frac_lick_only", "frac_tf_only", "frac_both", "frac_neither"]
            colors = {
                "frac_lick_only": "#FF8C00",  # orange
                "frac_tf_only": "#1f77b4",    # blue
                "frac_both": "#2ca02c",       # green
                "frac_neither": "#AAAAAA",     # gray
            }
            fig, ax = plt.subplots(figsize=(8, 4.5))
            bottom = np.zeros(len(summary))
            x = np.arange(len(summary))
            for cat in cats:
                vals = summary[cat].to_numpy()
                ax.bar(x, vals, bottom=bottom, color=colors[cat], label=cat.replace("frac_", ""))
                bottom += vals
            ax.set_xticks(x)
            ax.set_xticklabels(summary["session"], rotation=45, ha="right")
            ax.set_ylabel("Fraction of units")
            ax.set_title("Lick vs TF responsiveness categories (strict QC, full units)")
            ax.set_ylim(0, 1.0)
            ax.legend(frameon=False, ncol=2)
            fig.tight_layout()
            png_path = out_dir / "overlap_fraction_stacked.png"
            fig.savefig(png_path, dpi=140, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote plot: {png_path}")
        except Exception as e:
            print(f"[WARN] Plot generation failed: {e}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
