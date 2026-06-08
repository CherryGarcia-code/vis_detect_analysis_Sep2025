#!/usr/bin/env python3
"""Curate the liberal UnitMatch registry into precision tracks.

Outputs (FIGURES/tracking_qc/curation/):
    curated_links.csv    per-link audit trail
    curated_tracks.csv   per-track kept/skipped/dropped + confidence tier

Usage:
    py scripts/pipelines/tracking/curate_tracks.py [--min-span 2] [--rebuild-cache]

Note on registry path: the liberal cohort (with the ``batch_uid_liberal``
column) and its row-aligned ``output_prob_matrix.npy`` both live under
``batch0/`` for this single-batch run. The top-level ``all42/unit_index.csv``
is a reconciled registry that only carries ``global_uid`` — do not use it here.
"""
from __future__ import annotations

import argparse
import gc
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from visdetect.analysis import state_provider as sp                 # noqa: E402
from visdetect.analysis import track_curation as tc                 # noqa: E402
from visdetect.analysis.tracking_qc import (                        # noqa: E402
    load_channel_positions, estimate_session_drift)
from visdetect.core.session import load_session                     # noqa: E402
from visdetect.suite.loader import load_filtered_manifest           # noqa: E402

UM_ROOT = Path("X:/public/projects/BeJG_20230130_VisDetect/wEPhys/"
               "BG_046/unit_match/output/all42")
DEFAULT_REGISTRY = UM_ROOT / "batch0" / "unit_index.csv"
DEFAULT_PROB_MATRIX = UM_ROOT / "batch0" / "output_prob_matrix.npy"
DEFAULT_RAW_WF_ROOT = REPO_ROOT / "data" / "unit_match" / "input" / "BG_046"
DEFAULT_PKL_DIR = REPO_ROOT / "data" / "pkls" / "BG_046"
DEFAULT_STATES_DIR = REPO_ROOT / "data" / "cache" / "states" / "BG_046"
DEFAULT_OUT_DIR = REPO_ROOT / "FIGURES" / "tracking_qc" / "curation"
DEFAULT_CACHE = REPO_ROOT / "data" / "cache" / "curation_features.pkl"


def _date_key(s: str) -> Tuple[int, int, int]:
    p = str(s).zfill(8)
    return (int(p[4:8]), int(p[2:4]), int(p[0:2]))


def _session_pkl(pkl_dir: Path, sess: str):
    for s in (sess, str(sess).zfill(8)):
        p = pkl_dir / f"BG_046_{s}.pkl"
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--liberal-col", default="batch_uid_liberal")
    ap.add_argument("--prob-matrix", type=Path, default=DEFAULT_PROB_MATRIX)
    ap.add_argument("--raw-wf-root", type=Path, default=DEFAULT_RAW_WF_ROOT)
    ap.add_argument("--pkl-dir", type=Path, default=DEFAULT_PKL_DIR)
    ap.add_argument("--states-dir", type=Path, default=DEFAULT_STATES_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--min-span", type=int, default=2)
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--max-bridge-gap", type=int, default=tc.MAX_BRIDGE_GAP)
    ap.add_argument("--min-inzone-trials", type=int, default=tc.MIN_INZONE_TRIALS)
    ap.add_argument("--min-trusted-span", type=int, default=tc.MIN_TRUSTED_SPAN)
    ap.add_argument("--corroborator-ref", choices=["rolling", "expert"], default="rolling")
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Registry re-keyed on the liberal column ──────────────────────────
    reg = pd.read_csv(args.registry)
    reg["session"] = reg["session"].astype(str)
    reg["uid"] = reg[args.liberal_col].astype(int)
    span = reg.groupby("uid")["session"].nunique()
    keep_uids = set(span[span >= args.min_span].index.tolist())
    reg = reg[reg["uid"].isin(keep_uids)].copy()
    uid_to_ks: Dict[int, Dict[str, int]] = {}
    for _, r in reg.iterrows():
        uid_to_ks.setdefault(int(r["uid"]), {})[str(r["session"])] = int(r["ks_unit_id"])
    print(f"liberal cohort: {len(keep_uids)} uids span>={args.min_span}", flush=True)

    manifest = load_filtered_manifest(
        include_stages=["Naive", "Learning", "Expert"],
        merge_naive_learning=True, min_trials=150, min_dprime=None)
    stage_by_sess = {str(r["session_name"]).zfill(8): str(r["stage"])
                     for _, r in manifest.iterrows()}

    # ── Drift offsets across all registry sessions ───────────────────────
    all_sess = sorted(reg["session"].unique().tolist(), key=_date_key)
    drift_offsets = {}
    if args.prob_matrix.exists():
        prob = np.load(args.prob_matrix)
        drift_offsets = estimate_session_drift(reg, prob, args.raw_wf_root, all_sess)
    else:
        print("prob matrix missing — depth uses raw (offset 0)", flush=True)

    # ── Build / load feature cache (outer loop by session) ───────────────
    if args.rebuild_cache or not args.cache_path.exists():
        features: Dict[Tuple[int, str], tc.CurationFeature] = {}
        for sess in all_sess:
            pkl = _session_pkl(args.pkl_dir, sess)
            if pkl is None:
                print(f"  skip {sess}: no pkl", flush=True); continue
            S = load_session(str(pkl))
            cp = load_channel_positions(args.raw_wf_root, sess)
            in_zone = sp.in_zone_trial_indices(sess, args.states_dir,
                                               min_confidence=args.min_confidence)
            off = float(drift_offsets.get(str(sess).zfill(8),
                        drift_offsets.get(sess, 0.0)))
            stage = stage_by_sess.get(str(sess).zfill(8), "Unknown")
            for uid, ksmap in uid_to_ks.items():
                if sess not in ksmap:
                    continue
                feat = tc.extract_curation_feature(
                    S, ksmap[sess], session_name=sess, stage=stage,
                    raw_wf_root=args.raw_wf_root, channel_positions=cp,
                    in_zone_idx=in_zone, drift_offset=off)
                if feat is not None:
                    features[(uid, sess)] = feat
            del S; gc.collect()
            print(f"  {sess}: features cached", flush=True)
        args.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.cache_path, "wb") as f:
            pickle.dump(features, f)
        print(f"  saved {len(features)} features -> {args.cache_path}", flush=True)
    else:
        with open(args.cache_path, "rb") as f:
            features = pickle.load(f)
        print(f"loaded {len(features)} cached features", flush=True)

    # ── Sweep + write ────────────────────────────────────────────────────
    uid_to_sessions = {uid: sorted(ks.keys(), key=_date_key)
                       for uid, ks in uid_to_ks.items()}
    params = tc.CurationParams(
        max_bridge_gap=args.max_bridge_gap, min_inzone_trials=args.min_inzone_trials,
        min_trusted_span=args.min_trusted_span, corroborator_ref=args.corroborator_ref)
    links_df, tracks_df = tc.curate_registry(uid_to_sessions, features, params)
    links_df.to_csv(args.out_dir / "curated_links.csv", index=False)
    tracks_df.to_csv(args.out_dir / "curated_tracks.csv", index=False)
    n_tier = tracks_df["confidence_tier"].value_counts().to_dict()
    print(f"Wrote curated_links.csv + curated_tracks.csv -> {args.out_dir}", flush=True)
    print(f"  tiers: {n_tier}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
