#!/usr/bin/env python
"""Demonstration script for progress bars using synthetic data.

Runs three quick analyses with visible progress:
  1. TF pulse trace collection
  2. Lick responsiveness (paired pre/post)
  3. Generic Change_ON responsiveness

This avoids waiting on large real sessions; useful for verifying tqdm integration.

Usage:
    python scripts/demo_progress_synthetic.py --trials 60 --clusters 30

Set PROGRESS_SIMPLE=1 to force simple printing instead of tqdm.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from visdetect.utils.synthetic import make_synthetic_session
from visdetect.analysis.tf_pulse import collect_tf_pulse_traces, TFRespPulseConfig
from visdetect.analysis.lick_responsiveness import compute_lick_responsiveness_table, LickRespConfig
from visdetect.analysis.responsiveness import compute_responsiveness_table, RespConfig


def main(argv=None):
    ap = argparse.ArgumentParser(description="Synthetic progress demo")
    ap.add_argument("--trials", type=int, default=50, help="Number of synthetic trials")
    ap.add_argument("--clusters", type=int, default=25, help="Number of synthetic clusters")
    ap.add_argument("--no-tqdm", action="store_true", help="Force simple printing (sets PROGRESS_SIMPLE)")
    args = ap.parse_args(argv)

    if args.no_tqdm:
        import os
        os.environ["PROGRESS_SIMPLE"] = "1"

    sess = make_synthetic_session(n_trials=args.trials, n_clusters=args.clusters)
    print(f"Synthetic session: trials={len(sess.trials)} clusters={len(sess.clusters)}")

    # 1. TF pulse traces (progress internal)
    print("[1] Collecting TF pulse traces ...")
    cfg_tf = TFRespPulseConfig(kept_only=True)
    t_vec, entries = collect_tf_pulse_traces(sess, cfg=cfg_tf, selection_csv=None, show_progress=True)
    print(f"Collected TF pulse traces for {len(entries)} clusters, time bins={t_vec.size}")

    # 2. Lick responsiveness with progress
    print("[2] Computing lick responsiveness ...")
    cfg_lick = LickRespConfig()
    df_lick = compute_lick_responsiveness_table(sess, cfg_lick, selection_csv=None, show_progress=True)
    print(f"Lick responsiveness rows: {len(df_lick)} responsive={df_lick['is_responsive'].sum()} / {len(df_lick)}")

    # 3. Change_ON responsiveness (no built-in progress; show manual progress using wrapper)
    print("[3] Computing Change_ON responsiveness ...")
    cfg_resp = RespConfig(event_name="Change_ON", per_outcome=True)
    df_resp = compute_responsiveness_table(sess, cfg_resp, selection_csv=None)
    print(f"Change_ON responsiveness rows: {len(df_resp)} responsive={df_resp['is_responsive'].sum()} / {len(df_resp)}")

    # Simple sanity outputs
    print("Top 5 lick ΔFR:")
    print(df_lick.sort_values('delta_fr', ascending=False).head(5)[['cluster_id','outcome','delta_fr','p_value']])
    print("Top 5 Change_ON d':")
    print(df_resp.sort_values('dprime', ascending=False).head(5)[['cluster_id','outcome','dprime','p_value']])

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())