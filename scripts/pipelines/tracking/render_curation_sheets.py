#!/usr/bin/env python3
"""Render per-candidate QC sheets for a curated-track tier.

Thin driver that reuses the existing per-UID renderer (qc_sheet_figures.write_uid_pdf)
on the *curated* cohort. For each curated track it renders the FULL track and marks the
curation's DROPPED sessions (dimmed + red heatmap label); badge metrics are computed on
the KEPT subset. Stages come from the staging manifest (Naive->Learning); sessions absent
there (all of them, for manifest-less subjects) render light-grey "Unknown" — NOT dropped.

Multi-subject: --subject selects the UM output dir + local paths. Session tokens are
matched on the raw registry string (curated_tracks.kept_sessions and unit_index.session
share the same source) and sorted via the subject-aware session_date_key.

Usage:
    py scripts/pipelines/tracking/render_curation_sheets.py --subject BG_049 [--tier trusted]
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np  # noqa: F401
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))           # sibling scripts (qc_sheet_figures, build_qc_sheets)
sys.path.insert(0, str(REPO_ROOT / "src"))


def _early_subject(default: str = "BG_046") -> str:
    for i, a in enumerate(sys.argv):
        if a == "--subject" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--subject="):
            return a.split("=", 1)[1]
    return default


os.environ["VISDETECT_SUBJECT"] = _early_subject()

import _subject_paths as sjp                                       # noqa: E402
from visdetect.analysis.tracking_qc import (                       # noqa: E402
    UIDIntermediate, extract_session_records, load_channel_positions)
from visdetect.core.session import load_session                    # noqa: E402
from visdetect.suite.loader import load_staging_manifest           # noqa: E402

from qc_sheet_figures import write_uid_pdf                         # noqa: E402
from build_qc_sheets import compute_uid_metrics, _pair_scores_from_paths  # noqa: E402


def cohort_from_tracks(tracks_df: pd.DataFrame, tier: str) -> Dict[int, set]:
    """{curated_uid -> set of KEPT session tokens} for the rows in `tier` (span>=2).

    Tokens are kept RAW (curated_tracks.kept_sessions == unit_index.session strings).
    """
    out: Dict[int, set] = {}
    sub = tracks_df[tracks_df["confidence_tier"] == tier]
    for _, row in sub.iterrows():
        kept = {s for s in str(row["kept_sessions"]).split(";") if s}
        if len(kept) >= 2:
            out[int(row["curated_uid"])] = kept
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="BG_046")
    ap.add_argument("--tracks", type=Path, default=None)
    ap.add_argument("--tier", default="trusted",
                    choices=["trusted", "review", "suspect"])
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--liberal-col", default="global_uid")
    ap.add_argument("--prob-matrix", type=Path, default=None)
    ap.add_argument("--prob-index", type=Path, default=None)
    ap.add_argument("--raw-wf-root", type=Path, default=None)
    ap.add_argument("--pkl-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--uids", type=int, nargs="*", default=None,
                    help="render only these curated UIDs (within the tier)")
    ap.add_argument("--max-uids", type=int, default=None)
    ap.add_argument("--no-pair-scores", action="store_true",
                    help="skip the UM match-probability bar (avoids loading the large "
                         "prob_matrix; use to keep heavy I/O off the X: mount)")
    ap.add_argument("--kept-only", action="store_true",
                    help="render ONLY the kept sessions (drop trimmed/removed rows from "
                         "all panels). Cleaner page-2 heatmap colors for heavily-trimmed "
                         "tracks (dropped rows otherwise pin the diverging colormap "
                         "extremes). Default keeps dropped rows (dimmed/red).")
    args = ap.parse_args()
    subj = args.subject
    cur_dir = sjp.curation_out_dir(subj)
    if args.tracks is None: args.tracks = cur_dir / "curated_tracks.csv"
    if args.registry is None: args.registry = sjp.um_registry(subj)
    if args.prob_matrix is None: args.prob_matrix = sjp.um_prob_matrix(subj)
    if args.prob_index is None: args.prob_index = sjp.um_prob_index(subj)
    if args.raw_wf_root is None: args.raw_wf_root = sjp.raw_wf_root(subj)
    if args.pkl_dir is None: args.pkl_dir = sjp.pkl_dir(subj)
    if args.out_dir is None: args.out_dir = sjp.sheets_dir(subj)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tracks = pd.read_csv(args.tracks)
    kept_by_uid = cohort_from_tracks(tracks, args.tier)
    if args.uids:
        kept_by_uid = {u: s for u, s in kept_by_uid.items() if u in set(args.uids)}
    if args.max_uids:
        kept_by_uid = dict(list(kept_by_uid.items())[:args.max_uids])
    if not kept_by_uid:
        print(f"no {args.tier} tracks to render", flush=True)
        return 0
    print(f"{subj} {args.tier}: {len(kept_by_uid)} tracks to render", flush=True)

    # Registry: FULL session list + ks per (uid, session); dropped = full minus kept.
    reg = pd.read_csv(args.registry)
    reg["session"] = reg["session"].astype(str)
    reg["uid"] = reg[args.liberal_col].astype(int)
    uid_to_ks: Dict[int, Dict[str, int]] = {}
    for _, r in reg.iterrows():
        u = int(r["uid"])
        if u in kept_by_uid:
            uid_to_ks.setdefault(u, {})[str(r["session"])] = int(r["ks_unit_id"])

    # Stage source keyed by date-tuple (robust to 7-vs-8-digit + prefix); empty for
    # manifest-less subjects -> all 'Unknown'.
    manifest = load_staging_manifest(qc_only=False, apply_filter=False)
    stage_by_date: Dict[tuple, str] = {}
    for _, r in manifest.iterrows():
        st = str(r["stage"])
        stage_by_date[sjp.session_date_key(r["session_name"])] = \
            "Learning" if st == "Naive" else st

    def _stage(sess):
        return stage_by_date.get(sjp.session_date_key(sess), "Unknown")

    intermediates: Dict[int, UIDIntermediate] = {
        u: UIDIntermediate(global_uid=u, span=len(ks), has_naive_to_expert=False,
                           suspect_known=False, sessions=[])
        for u, ks in uid_to_ks.items()
    }

    sess_set = sorted({s for ks in uid_to_ks.values() for s in ks},
                      key=sjp.session_date_key)
    for sess in sess_set:
        pkl = sjp.session_pkl(subj, sess, args.pkl_dir)
        if pkl is None:
            print(f"  skip {sess}: no pkl", flush=True); continue
        S = load_session(str(pkl))
        cp = load_channel_positions(args.raw_wf_root, sess)
        uids_here = [u for u, ks in uid_to_ks.items() if sess in ks]
        ks_here = [uid_to_ks[u][sess] for u in uids_here]
        records = extract_session_records(
            S, ks_here, session_name=sess, stage=_stage(sess),
            raw_wf_root=args.raw_wf_root, channel_positions=cp)
        for u in uids_here:
            rec = records.get(int(uid_to_ks[u][sess]))
            if rec is not None:
                intermediates[u].sessions.append(rec)
        del S; gc.collect()
        print(f"  {sess}: {len(records)}/{len(uids_here)} records", flush=True)

    uid_to_sessions = {u: sorted(ks.keys(), key=sjp.session_date_key)
                       for u, ks in uid_to_ks.items()}
    if args.no_pair_scores:
        print("  --no-pair-scores: skipping prob-matrix (no X: heavy read)", flush=True)
        pair_scores = {}
    else:
        pair_scores = _pair_scores_from_paths(args.prob_matrix, args.prob_index,
                                              uid_to_sessions, uid_to_ks)

    n = 0
    for u, iv in intermediates.items():
        iv.sessions.sort(key=lambda r: sjp.session_date_key(r.session_name))
        iv.span = len(iv.sessions)
        kept = kept_by_uid[u]
        kept_recs = [r for r in iv.sessions if r.session_name in kept]
        dropped_idx = [i for i, r in enumerate(iv.sessions)
                       if r.session_name not in kept]
        iv.has_naive_to_expert = (
            "Expert" in {r.stage for r in kept_recs}
            and "Learning" in {r.stage for r in kept_recs})
        if len(kept_recs) < 2:
            print(f"  uid {u}: <2 kept records, skipped", flush=True); continue
        kept_iv = UIDIntermediate(global_uid=u, span=len(kept_recs),
                                  has_naive_to_expert=iv.has_naive_to_expert,
                                  suspect_known=False, sessions=kept_recs)
        m = compute_uid_metrics(kept_iv)
        out = args.out_dir / f"{args.tier}_uid{u}_span{len(kept_recs)}.pdf"
        # --kept-only: feed the kept-only intermediate so dropped sessions vanish from
        # every panel (no red dropped-rows compressing the page-2 colormap).
        render_iv = kept_iv if args.kept_only else iv
        render_dropped = None if args.kept_only else (dropped_idx or None)
        write_uid_pdf(out, render_iv, pair_scores.get(u),
                      isi_score=m["isi_hist_corr"], depth_std=m["depth_std_um"],
                      wave_corr=m["wave_corr"], fr_cv_val=m["fr_cv"],
                      dropped_indices=render_dropped,
                      n_kept=len(kept_recs), trimmed_verdict=args.tier)
        n += 1
        print(f"  wrote {out.name} (kept {len(kept_recs)}/{len(iv.sessions)})", flush=True)
    print(f"Done: {n} sheets -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
