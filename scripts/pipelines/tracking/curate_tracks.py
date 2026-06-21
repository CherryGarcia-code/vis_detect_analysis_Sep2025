#!/usr/bin/env python3
"""Curate the UnitMatch registry into precision tracks.

Outputs (FIGURES/tracking_qc/<SUBJECT>/curation/):
    curated_links.csv    per-link audit trail
    curated_tracks.csv   per-track kept/skipped/dropped + confidence tier

Usage:
    py scripts/pipelines/tracking/curate_tracks.py --subject BG_049 [--rebuild-cache]

Registry note: curate the ``global_uid`` registry (``unit_index.csv``), NOT the
``batch_uid_liberal`` column. The "liberal" assignment over-merges into a few
heterogeneous mega-blobs that the backward sweep cannot recover and truncates to
span 1; the intermediate/``global_uid`` registry has the clean long tracks. Override
with ``--registry``/``--liberal-col`` to curate a different column.

Multi-subject: --subject selects the UM output dir (all42 for BG_046, all_sessions
otherwise) and all local paths; subjects without a staging manifest render as
'Unknown' stage and (with no behavioural trials) skip the functional corroborator.
"""
from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))      # for _subject_paths


def _early_subject(default: str = "BG_046") -> str:
    for i, a in enumerate(sys.argv):
        if a == "--subject" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--subject="):
            return a.split("=", 1)[1]
    return default


os.environ["VISDETECT_SUBJECT"] = _early_subject()

import _subject_paths as sjp                                    # noqa: E402
from visdetect.analysis import state_provider as sp             # noqa: E402
from visdetect.analysis import track_curation as tc            # noqa: E402
from visdetect.analysis.tracking_qc import (                    # noqa: E402
    load_channel_positions, estimate_session_drift)
from visdetect.core.session import load_session                 # noqa: E402
from visdetect.suite.loader import load_filtered_manifest       # noqa: E402


def _load_fingerprint_offsets(csv_path: Path) -> Dict[str, float]:
    """Per-session drift offset (um) from the fingerprint diagnostic CSV, keyed by
    the raw session token (matches the registry)."""
    df = pd.read_csv(csv_path)
    return {str(r["session"]): float(r["drift_vs_ref0_um"]) for _, r in df.iterrows()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="BG_046")
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--liberal-col", default="global_uid",
                    help="UID column to curate. Default global_uid (clean long "
                         "tracks); batch_uid_liberal over-merges into blobs.")
    ap.add_argument("--prob-matrix", type=Path, default=None)
    ap.add_argument("--raw-wf-root", type=Path, default=None)
    ap.add_argument("--pkl-dir", type=Path, default=None)
    ap.add_argument("--states-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--cache-path", type=Path, default=None)
    ap.add_argument("--min-span", type=int, default=2)
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--max-bridge-gap", type=int, default=tc.MAX_BRIDGE_GAP)
    ap.add_argument("--min-inzone-trials", type=int, default=tc.MIN_INZONE_TRIALS)
    ap.add_argument("--min-trusted-span", type=int, default=tc.MIN_TRUSTED_SPAN)
    ap.add_argument("--corroborator-ref", choices=["rolling", "expert"], default="rolling")
    ap.add_argument("--drift-source", choices=["none", "fingerprint", "match"],
                    default="none",
                    help="Depth drift correction source. 'none' (default) = RAW depth; "
                         "'fingerprint' = offsets from --drift-csv; 'match' = legacy "
                         "UnitMatch-anchor estimator (starves on low-anchor sessions).")
    ap.add_argument("--drift-csv", type=Path, default=None,
                    help="intersession_drift.csv for --drift-source fingerprint")
    ap.add_argument("--drop-sessions", nargs="*", default=None,
                    help="Session tokens to EXCLUDE from the registry before curation "
                         "(e.g. a duplicate re-sort like BG_039_23042025_v2). Restart "
                         "chunks of the same day with DIFFERENT content should NOT be "
                         "dropped — they are genuine separate sessions.")
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()
    subj = args.subject
    if args.registry is None: args.registry = sjp.um_registry(subj)
    if args.prob_matrix is None: args.prob_matrix = sjp.um_prob_matrix(subj)
    if args.raw_wf_root is None: args.raw_wf_root = sjp.raw_wf_root(subj)
    if args.pkl_dir is None: args.pkl_dir = sjp.pkl_dir(subj)
    if args.states_dir is None: args.states_dir = sjp.states_dir(subj)
    if args.out_dir is None: args.out_dir = sjp.curation_out_dir(subj)
    if args.cache_path is None: args.cache_path = sjp.features_cache(subj)
    if args.drift_csv is None: args.drift_csv = sjp.drift_csv(subj)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Registry re-keyed on the chosen uid column ───────────────────────
    reg = pd.read_csv(args.registry)
    reg["session"] = reg["session"].astype(str)
    if args.drop_sessions:
        drop = set(args.drop_sessions)
        n0 = len(reg)
        reg = reg[~reg["session"].isin(drop)].copy()
        print(f"dropped sessions {sorted(drop)}: {n0 - len(reg)} registry rows removed",
              flush=True)
    reg["uid"] = reg[args.liberal_col].astype(int)
    span = reg.groupby("uid")["session"].nunique()
    keep_uids = set(span[span >= args.min_span].index.tolist())
    reg = reg[reg["uid"].isin(keep_uids)].copy()
    uid_to_ks: Dict[int, Dict[str, int]] = {}
    for _, r in reg.iterrows():
        uid_to_ks.setdefault(int(r["uid"]), {})[str(r["session"])] = int(r["ks_unit_id"])
    print(f"{subj} cohort: {len(keep_uids)} uids span>={args.min_span} "
          f"(col {args.liberal_col})", flush=True)

    manifest = load_filtered_manifest(
        include_stages=["Naive", "Learning", "Expert"],
        merge_naive_learning=True, min_trials=150, min_dprime=None)
    stage_by_sess = {str(r["session_name"]): str(r["stage"])
                     for _, r in manifest.iterrows()}

    # ── Drift offsets (default 'none' = raw depth) ───────────────────────
    all_sess = sorted(reg["session"].unique().tolist(), key=sjp.session_date_key)
    drift_offsets: Dict[str, float] = {}
    if args.drift_source == "match":
        if args.prob_matrix.exists():
            prob = np.load(args.prob_matrix)
            drift_offsets = estimate_session_drift(reg, prob, args.raw_wf_root, all_sess)
        else:
            print("prob matrix missing — depth uses raw (offset 0)", flush=True)
    elif args.drift_source == "fingerprint":
        if args.drift_csv.exists():
            drift_offsets = _load_fingerprint_offsets(args.drift_csv)
            print(f"fingerprint drift: {len(drift_offsets)} offsets", flush=True)
        else:
            print(f"drift csv {args.drift_csv} missing — raw depth (offset 0)", flush=True)
    else:
        print("drift-source=none — depth gate uses RAW depth (offset 0)", flush=True)

    # ── Build / load feature cache (outer loop by session) ───────────────
    if args.rebuild_cache or not args.cache_path.exists():
        features: Dict[Tuple[int, str], tc.CurationFeature] = {}
        for sess in all_sess:
            pkl = sjp.session_pkl(subj, sess, args.pkl_dir)
            if pkl is None:
                print(f"  skip {sess}: no pkl", flush=True); continue
            S = load_session(str(pkl))
            cp = load_channel_positions(args.raw_wf_root, sess)
            in_zone = sp.in_zone_trial_indices(sess, args.states_dir,
                                               min_confidence=args.min_confidence)
            off = float(drift_offsets.get(sess, 0.0))
            stage = stage_by_sess.get(sess, "Unknown")
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
    uid_to_sessions = {uid: sorted(ks.keys(), key=sjp.session_date_key)
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
