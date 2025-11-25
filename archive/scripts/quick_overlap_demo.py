#!/usr/bin/env python
"""Quick overlap demo: lick-responsive vs TF pulse-responsive units on limited subset.

Purpose:
    Load one early and one late session pickle, optionally limit to first N clusters,
    compute lick responsiveness (pooled 'All' outcome) and TF pulse responsiveness,
    then summarize overlap categories (lick-only, TF-only, both, neither).

Why:
    Fast sanity check before running full ~50-session analysis; avoids long runtimes.

Usage:
    python scripts/quick_overlap_demo.py \
        --early pkls/BG_046_30062025.pkl \
        --late pkls/BG_046_16092025.pkl \
        --limit-clusters 25 --tf-z 3.0 --out table_output/quick_overlap_demo

Outputs:
    - summary CSV: <out>/quick_overlap_summary.csv
    - printed table of fractions early vs late

Notes:
    - Lick responsiveness uses `compute_lick_responsiveness_table` (delta_fr, p_value, is_responsive).
    - TF responsiveness runs a minimal screening via `run_tf_pulse_screening` with kept_only enforced.
    - Cluster limiting: we filter `session.clusters` and `session.good_cluster_ids` (if present).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.session_io import load_session  # noqa: E402
from src.lick_responsiveness import compute_lick_responsiveness_table, LickRespConfig  # noqa: E402
from src.tf_pulse import TFRespPulseConfig, run_tf_pulse_screening  # noqa: E402


def _limit_clusters(session, limit: int | None):
    if limit is None:
        return session
    clusters = sorted(session.clusters, key=lambda c: int(c.cluster_id))[: limit]
    setattr(session, 'clusters', clusters)
    good_ids = getattr(session, 'good_cluster_ids', None)
    if good_ids is not None:
        good_set = set(int(c.cluster_id) for c in clusters)
        setattr(session, 'good_cluster_ids', [gid for gid in good_ids if int(gid) in good_set])
    return session


def _lick_table(session, event: str, show_progress: bool = False) -> pd.DataFrame:
    cfg = LickRespConfig(event_name=event, kept_only=True)
    df = compute_lick_responsiveness_table(session, cfg, selection_csv=None, show_progress=show_progress)
    # pooled rows outcome == 'All'
    pooled = df[df['outcome'] == 'All'].copy()
    return pooled[['cluster_id', 'delta_fr', 'p_value', 'is_responsive']]


def _tf_table(session, fast_log2: float, slow_log2: float, z_thresh: float, tmp_root: Path) -> pd.DataFrame:
    # Run screening into a temp directory (kept_only True)
    cfg = TFRespPulseConfig(
        fast_thresh_log2=fast_log2,
        slow_thresh_log2=slow_log2,
        z_thresh=z_thresh,
        kept_only=True,
    )
    out_root = tmp_root / 'tf_tmp'
    png_root = tmp_root / 'tf_tmp_png'
    out_root.mkdir(parents=True, exist_ok=True)
    png_root.mkdir(parents=True, exist_ok=True)
    paths = run_tf_pulse_screening(session, out_root=str(out_root), png_root=str(png_root), cfg=cfg, selection_csv=None, generate_grid=False)
    csv_path = Path(paths.get('csv', out_root / 'tf_pulse_units.csv'))
    if not csv_path.exists():
        return pd.DataFrame(columns=['cluster_id', 'fast_responsive', 'slow_responsive'])
    df = pd.read_csv(csv_path)
    keep_cols = [c for c in ['cluster_id', 'fast_responsive', 'slow_responsive'] if c in df.columns]
    return df[keep_cols]


def _overlap_df(lick_df: pd.DataFrame, tf_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(lick_df, tf_df, on='cluster_id', how='outer')
    merged['lick_resp'] = merged['is_responsive'].fillna(False).astype(bool)
    # TF responsive if fast OR slow responsive True
    fr = merged.get('fast_responsive')
    sr = merged.get('slow_responsive')
    merged['tf_resp'] = ((fr.fillna(False)) | (sr.fillna(False))).astype(bool)
    merged['category'] = np.select(
        [merged['lick_resp'] & merged['tf_resp'], merged['lick_resp'], merged['tf_resp']],
        ['both', 'lick_only', 'tf_only'],
        default='neither'
    )
    return merged[['cluster_id', 'lick_resp', 'tf_resp', 'category']]


def _category_counts(df: pd.DataFrame) -> dict:
    total = len(df)
    counts = df['category'].value_counts().to_dict()
    return {
        'total_units': total,
        'lick_only': counts.get('lick_only', 0),
        'tf_only': counts.get('tf_only', 0),
        'both': counts.get('both', 0),
        'neither': counts.get('neither', 0),
    }


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(description="Quick limited overlap demo (early vs late session)")
    ap.add_argument('--early', required=True, help='Path to early session .pkl')
    ap.add_argument('--late', required=True, help='Path to late session .pkl')
    ap.add_argument('--limit-clusters', type=int, default=25, help='Limit to first N clusters by cluster_id')
    ap.add_argument('--lick-event', default='Lick_L', help='Lick event name')
    ap.add_argument('--fast-thresh-log2', type=float, default=0.25, help='Fast TF log2 threshold')
    ap.add_argument('--slow-thresh-log2', type=float, default=-0.25, help='Slow TF log2 threshold')
    ap.add_argument('--tf-z', type=float, default=3.0, help='Z threshold for TF responsiveness')
    ap.add_argument('--out', default='table_output/quick_overlap_demo', help='Output directory for summary CSV')
    ap.add_argument('--no-progress', action='store_true', help='Disable lick responsiveness progress')
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = out_dir / 'tmp'
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Early session
    early_sess = load_session(args.early)
    early_sess = _limit_clusters(early_sess, args.limit_clusters)
    early_lick = _lick_table(early_sess, args.lick_event, show_progress=(not args.no_progress))
    early_tf = _tf_table(early_sess, args.fast_thresh_log2, args.slow_thresh_log2, args.tf_z, tmp_root)
    early_overlap = _overlap_df(early_lick, early_tf)
    early_counts = _category_counts(early_overlap)

    # Late session
    late_sess = load_session(args.late)
    late_sess = _limit_clusters(late_sess, args.limit_clusters)
    late_lick = _lick_table(late_sess, args.lick_event, show_progress=(not args.no_progress))
    late_tf = _tf_table(late_sess, args.fast_thresh_log2, args.slow_thresh_log2, args.tf_z, tmp_root)
    late_overlap = _overlap_df(late_lick, late_tf)
    late_counts = _category_counts(late_overlap)

    # Build summary dataframe
    summary_rows = []
    early_name = f"{getattr(early_sess,'subject','unk')}_{getattr(early_sess,'session_name','early')}"
    late_name = f"{getattr(late_sess,'subject','unk')}_{getattr(late_sess,'session_name','late')}"
    summary_rows.append({'session': early_name, **early_counts})
    summary_rows.append({'session': late_name, **late_counts})
    summary_df = pd.DataFrame(summary_rows)
    csv_path = out_dir / 'quick_overlap_summary.csv'
    summary_df.to_csv(csv_path, index=False)

    # Print concise summary
    def fmt(row):
        tot = row['total_units'] or 1
        return (
            f"{row['session']}: total={row['total_units']} "
            f"lick_only={row['lick_only']} ({row['lick_only']/tot:.2%}) "
            f"tf_only={row['tf_only']} ({row['tf_only']/tot:.2%}) both={row['both']} ({row['both']/tot:.2%}) "
            f"neither={row['neither']} ({row['neither']/tot:.2%})"
        )
    print(fmt(summary_df.iloc[0]))
    print(fmt(summary_df.iloc[1]))
    print(f"Wrote summary CSV: {csv_path}")
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
