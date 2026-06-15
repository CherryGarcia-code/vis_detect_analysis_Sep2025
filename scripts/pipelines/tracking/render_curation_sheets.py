#!/usr/bin/env python3
"""Render per-candidate QC sheets for a curated-track tier.

Thin driver that reuses the existing per-UID renderer (qc_sheet_figures.write_uid_pdf)
on the *curated* cohort. build_qc_sheets.py can't do this directly: it's hardwired to
the >=10-session global_uid cohort (select_long_tracks min_span=10), whereas curated
trusted tracks span 3-7 sessions and are keyed on the liberal batch_uid_liberal column.

For each curated track of the chosen --tier, builds a UIDIntermediate from its KEPT
sessions (the track the curation asserts), computes the standard QC metrics over those
sessions, and writes the same 2-page PDF (footprints, waveform overlay, depth trajectory,
ISI hist, FR, PSTHs + badges), stamped with the curation tier via `trimmed_verdict`.

Usage:
    py scripts/pipelines/tracking/render_curation_sheets.py [--tier trusted] [--uids 842 ...]
"""
from __future__ import annotations

import argparse
import gc
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))           # sibling scripts (qc_sheet_figures, build_qc_sheets)
sys.path.insert(0, str(REPO_ROOT / "src"))

from visdetect.analysis.tracking_qc import (                       # noqa: E402
    UIDIntermediate, extract_session_records, load_channel_positions)
from visdetect.core.session import load_session                    # noqa: E402
from visdetect.suite.loader import load_filtered_manifest          # noqa: E402

from qc_sheet_figures import write_uid_pdf                         # noqa: E402
from build_qc_sheets import compute_uid_metrics, _pair_scores_from_paths  # noqa: E402

UM_ROOT = Path("X:/public/projects/BeJG_20230130_VisDetect/wEPhys/"
               "BG_046/unit_match/output/all42")
DEFAULT_REGISTRY = UM_ROOT / "batch0" / "unit_index.csv"
DEFAULT_PROB_MATRIX = UM_ROOT / "batch0" / "output_prob_matrix.npy"
DEFAULT_PROB_INDEX = UM_ROOT / "batch0" / "unit_index.csv"
DEFAULT_TRACKS = REPO_ROOT / "FIGURES" / "tracking_qc" / "curation" / "curated_tracks.csv"
DEFAULT_RAW_WF_ROOT = REPO_ROOT / "data" / "unit_match" / "input" / "BG_046"
DEFAULT_PKL_DIR = REPO_ROOT / "data" / "pkls" / "BG_046"
DEFAULT_OUT_DIR = REPO_ROOT / "FIGURES" / "tracking_qc" / "curation" / "sheets"


def _norm(s) -> str:
    return str(s).zfill(8)


def _session_date(s) -> datetime:
    return datetime.strptime(_norm(s), "%d%m%Y")


def _session_pkl(pkl_dir: Path, sess: str):
    for s in (sess, _norm(sess)):
        p = pkl_dir / f"BG_046_{s}.pkl"
        if p.exists():
            return p
    return None


def cohort_from_tracks(tracks_df: pd.DataFrame, tier: str) -> Dict[int, List[str]]:
    """{curated_uid -> kept-session list} for the rows in `tier` with span >= 2."""
    out: Dict[int, List[str]] = {}
    sub = tracks_df[tracks_df["confidence_tier"] == tier]
    for _, row in sub.iterrows():
        kept = [s for s in str(row["kept_sessions"]).split(";") if s]
        if len(kept) >= 2:
            out[int(row["curated_uid"])] = kept
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    ap.add_argument("--tier", default="trusted",
                    choices=["trusted", "review", "suspect"])
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--liberal-col", default="batch_uid_liberal")
    ap.add_argument("--prob-matrix", type=Path, default=DEFAULT_PROB_MATRIX)
    ap.add_argument("--prob-index", type=Path, default=DEFAULT_PROB_INDEX)
    ap.add_argument("--raw-wf-root", type=Path, default=DEFAULT_RAW_WF_ROOT)
    ap.add_argument("--pkl-dir", type=Path, default=DEFAULT_PKL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--uids", type=int, nargs="*", default=None,
                    help="render only these curated UIDs (within the tier)")
    ap.add_argument("--max-uids", type=int, default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tracks = pd.read_csv(args.tracks)
    cohort = cohort_from_tracks(tracks, args.tier)
    if args.uids:
        cohort = {u: s for u, s in cohort.items() if u in set(args.uids)}
    if args.max_uids:
        cohort = dict(list(cohort.items())[:args.max_uids])
    if not cohort:
        print(f"no {args.tier} tracks to render", flush=True)
        return 0
    print(f"{args.tier}: {len(cohort)} tracks to render", flush=True)

    # Liberal registry -> ks_unit_id per (uid, kept session)
    reg = pd.read_csv(args.registry)
    reg["session"] = reg["session"].astype(str)
    reg["uid"] = reg[args.liberal_col].astype(int)
    uid_to_ks: Dict[int, Dict[str, int]] = {}
    for _, r in reg.iterrows():
        u = int(r["uid"])
        if u in cohort and str(r["session"]) in cohort[u]:
            uid_to_ks.setdefault(u, {})[str(r["session"])] = int(r["ks_unit_id"])

    manifest = load_filtered_manifest(
        include_stages=["Naive", "Learning", "Expert"],
        merge_naive_learning=True, min_trials=150, min_dprime=None)
    stage_by_sess = {_norm(r["session_name"]): str(r["stage"])
                     for _, r in manifest.iterrows()}

    # Build intermediates, outer loop by session (load each pkl once).
    intermediates: Dict[int, UIDIntermediate] = {}
    for u, kept in cohort.items():
        stages = {stage_by_sess.get(_norm(s), "Unknown") for s in kept}
        has_n2e = ("Expert" in stages) and bool(stages & {"Naive", "Learning"})
        intermediates[u] = UIDIntermediate(global_uid=u, span=len(kept),
                                           has_naive_to_expert=has_n2e,
                                           suspect_known=False, sessions=[])

    sess_set = sorted({s for ks in uid_to_ks.values() for s in ks}, key=_session_date)
    for sess in sess_set:
        pkl = _session_pkl(args.pkl_dir, sess)
        if pkl is None:
            print(f"  skip {sess}: no pkl", flush=True); continue
        S = load_session(str(pkl))
        cp = load_channel_positions(args.raw_wf_root, sess)
        uids_here = [u for u, ks in uid_to_ks.items() if sess in ks]
        ks_here = [uid_to_ks[u][sess] for u in uids_here]
        records = extract_session_records(
            S, ks_here, session_name=sess,
            stage=stage_by_sess.get(_norm(sess), "Unknown"),
            raw_wf_root=args.raw_wf_root, channel_positions=cp)
        for u in uids_here:
            rec = records.get(int(uid_to_ks[u][sess]))
            if rec is not None:
                intermediates[u].sessions.append(rec)
        del S; gc.collect()
        print(f"  {sess}: {len(records)}/{len(uids_here)} records", flush=True)

    # Pair-score (consecutive UM match prob) traces from the liberal prob matrix.
    uid_to_sessions = {u: sorted(ks.keys(), key=_session_date)
                       for u, ks in uid_to_ks.items()}
    pair_scores = _pair_scores_from_paths(args.prob_matrix, args.prob_index,
                                          uid_to_sessions, uid_to_ks)

    n = 0
    for u, iv in intermediates.items():
        iv.sessions.sort(key=lambda r: _session_date(r.session_name))
        if len(iv.sessions) < 2:
            print(f"  uid {u}: <2 records, skipped", flush=True); continue
        m = compute_uid_metrics(iv)
        out = args.out_dir / f"{args.tier}_uid{u}_span{len(iv.sessions)}.pdf"
        write_uid_pdf(out, iv, pair_scores.get(u),
                      isi_score=m["isi_hist_corr"], depth_std=m["depth_std_um"],
                      wave_corr=m["wave_corr"], fr_cv_val=m["fr_cv"],
                      n_kept=len(iv.sessions), trimmed_verdict=args.tier)
        n += 1
        print(f"  wrote {out.name}", flush=True)
    print(f"Done: {n} sheets -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
