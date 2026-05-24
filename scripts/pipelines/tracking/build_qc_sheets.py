#!/usr/bin/env python3
"""Build per-UID QC sheets for the UnitMatch long-track cohort.

See docs/superpowers/specs/2026-05-21-tracking-qc-sheets-design.md and
docs/superpowers/plans/2026-05-22-tracking-qc-sheets-plan.md.

Usage:
    py scripts/pipelines/tracking/build_qc_sheets.py \
        [--rebuild-cache] [--uids 334 1294 600] [--max-uids N]
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from visdetect.analysis.tracking_qc import (        # noqa: E402
    UIDIntermediate, SessionRecord,
    select_long_tracks, annotate_naive_to_expert,
    extract_session_records, load_channel_positions,
    load_isi_scores, load_um_pair_scores,
    depth_std_um, waveform_corr, fr_cv,
    badge_isi, badge_depth, badge_waveform, badge_fr, composite_verdict,
    save_cache, load_cache,
)
from visdetect.core.session import load_session                 # noqa: E402
from visdetect.suite.loader import load_staging_manifest        # noqa: E402

from qc_sheet_figures import write_uid_pdf                       # noqa: E402

UM_ROOT       = Path("X:/public/projects/BeJG_20230130_VisDetect/wEPhys/"
                     "BG_046/unit_match/output/all42")
UNIT_INDEX    = UM_ROOT / "unit_index.csv"
ISI_STATS     = REPO_ROOT / "FIGURES" / "tracking_qc" / "track_validation_stats.csv"
RAW_WF_ROOT   = REPO_ROOT / "data" / "unit_match" / "input" / "BG_046"
PKL_DIR       = REPO_ROOT / "data" / "pkls" / "BG_046"

OUT_DIR       = REPO_ROOT / "FIGURES" / "tracking_qc" / "per_uid_sheets"
VERDICTS_CSV  = REPO_ROOT / "FIGURES" / "tracking_qc" / "verdicts.csv"
CACHE_PATH    = REPO_ROOT / "data" / "cache" / "tracking_qc_intermediates.pkl"


def _session_pkl(session_name: str) -> Optional[Path]:
    for s in (session_name, session_name.zfill(8)):
        p = PKL_DIR / f"BG_046_{s}.pkl"
        if p.exists():
            return p
    return None


def build_cache(unit_index_df: pd.DataFrame, cohort: pd.DataFrame,
                manifest: pd.DataFrame) -> Dict[int, UIDIntermediate]:
    """Outer loop by session.  Returns dict[uid -> UIDIntermediate]."""
    stage_by_session = {str(r["session_name"]): str(r["stage"])
                        for _, r in manifest.iterrows()}

    cohort_uids = set(cohort["global_uid"].astype(int).tolist())
    in_cohort = unit_index_df[unit_index_df["global_uid"].astype(int).isin(cohort_uids)].copy()
    in_cohort["session"] = in_cohort["session"].astype(str)
    uid_to_ks: Dict[int, Dict[str, int]] = {}
    for _, row in in_cohort.iterrows():
        uid = int(row["global_uid"])
        uid_to_ks.setdefault(uid, {})[str(row["session"])] = int(row["ks_unit_id"])

    cohort = cohort.set_index("global_uid")
    intermediates: Dict[int, UIDIntermediate] = {}
    for uid in cohort_uids:
        row = cohort.loc[uid]
        intermediates[uid] = UIDIntermediate(
            global_uid=uid,
            span=int(row["span"]),
            has_naive_to_expert=bool(row["has_naive_to_expert"]),
            suspect_known=bool(row["suspect_known"]),
            sessions=[],
        )

    sessions_chrono = manifest["session_name"].astype(str).tolist()
    sess_set = sorted({s for ksmap in uid_to_ks.values() for s in ksmap.keys()},
                      key=lambda s: sessions_chrono.index(s) if s in sessions_chrono else 1e9)

    for sess in sess_set:
        pkl = _session_pkl(sess)
        if pkl is None:
            print(f"  skip {sess}: no pkl", flush=True); continue
        t0 = time.time()
        S = load_session(str(pkl))
        chan_pos = load_channel_positions(RAW_WF_ROOT, sess)
        uids_here = [u for u, ks in uid_to_ks.items() if sess in ks]
        ks_ids_here = [uid_to_ks[u][sess] for u in uids_here]
        records = extract_session_records(
            S, ks_ids_here, session_name=sess,
            stage=stage_by_session.get(sess, "Learning"),
            raw_wf_root=RAW_WF_ROOT, channel_positions=chan_pos,
        )
        for u in uids_here:
            rec = records.get(int(uid_to_ks[u][sess]))
            if rec is not None:
                intermediates[u].sessions.append(rec)
        del S
        gc.collect()
        print(f"  {sess}: {len(records)}/{len(uids_here)} cached "
              f"in {time.time() - t0:.1f}s", flush=True)

    order_idx = {s: i for i, s in enumerate(sessions_chrono)}
    for uid in intermediates:
        intermediates[uid].sessions.sort(
            key=lambda r: order_idx.get(r.session_name, 1e9)
        )
    return intermediates


def compute_uid_metrics(uid: UIDIntermediate) -> Dict[str, float]:
    """Depth std, waveform corr, FR CV for one UID across its sessions."""
    depths = np.array([r.peak_depth_um for r in uid.sessions], dtype=float)
    rates  = np.array([r.baseline_fr_hz for r in uid.sessions], dtype=float)
    waves = [r.waveform_peak for r in uid.sessions if r.waveform_peak is not None]
    if waves:
        min_len = min(w.size for w in waves)
        wf_stack = np.stack([w[:min_len] for w in waves])
    else:
        wf_stack = np.zeros((0, 0), dtype=np.float32)
    return {
        "depth_std_um": depth_std_um(depths),
        "wave_corr":    waveform_corr(wf_stack),
        "fr_cv":        fr_cv(rates),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--uids", type=int, nargs="*", default=None,
                        help="Only render these UIDs (cohort filter still applies)")
    parser.add_argument("--max-uids", type=int, default=None,
                        help="Render at most N UIDs (debug)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading manifest + cohort ...", flush=True)
    manifest = load_staging_manifest(qc_only=True, apply_filter=True)
    unit_index_df = pd.read_csv(UNIT_INDEX)
    cohort = select_long_tracks(UNIT_INDEX, ISI_STATS, min_span=10)
    cohort = annotate_naive_to_expert(cohort, manifest)
    print(f"  cohort size: {len(cohort)}", flush=True)

    if args.rebuild_cache or not CACHE_PATH.exists():
        print("Building cache (this is slow — outer loop by session) ...", flush=True)
        intermediates = build_cache(unit_index_df, cohort, manifest)
        save_cache(intermediates, CACHE_PATH)
        print(f"  saved cache to {CACHE_PATH}", flush=True)
    else:
        print(f"Loading cached intermediates from {CACHE_PATH}", flush=True)
        intermediates = load_cache(CACHE_PATH)

    uid_to_sessions = {u: [r.session_name for r in iv.sessions]
                       for u, iv in intermediates.items()}
    uid_to_ks = {}
    for _, row in unit_index_df.iterrows():
        uid = int(row["global_uid"])
        uid_to_ks.setdefault(uid, {})[str(row["session"])] = int(row["ks_unit_id"])
    pair_scores = load_um_pair_scores(UM_ROOT, uid_to_sessions, uid_to_ks)

    isi_scores = load_isi_scores(ISI_STATS)

    rows = []
    uids_to_render = sorted(intermediates)
    if args.uids:
        uids_to_render = [u for u in uids_to_render if u in set(args.uids)]
    if args.max_uids:
        uids_to_render = uids_to_render[: args.max_uids]
    print(f"Rendering {len(uids_to_render)} UIDs ...", flush=True)

    for uid in uids_to_render:
        iv = intermediates[uid]
        if not iv.sessions:
            print(f"  uid {uid}: no sessions extracted, skipping"); continue
        metrics = compute_uid_metrics(iv)
        isi = isi_scores[uid]
        out_path = OUT_DIR / f"uid_{uid:04d}.pdf"
        verdict = write_uid_pdf(
            out_path, iv, pair_scores.get(uid),
            isi_score=isi,
            depth_std=metrics["depth_std_um"],
            wave_corr=metrics["wave_corr"],
            fr_cv_val=metrics["fr_cv"],
        )
        rows.append({
            "global_uid": uid,
            "span": iv.span,
            "sessions": ";".join(r.session_name for r in iv.sessions),
            "has_naive_to_expert": iv.has_naive_to_expert,
            "suspect_known": iv.suspect_known,
            "isi_median": isi,
            "depth_std_um": metrics["depth_std_um"],
            "wave_corr": metrics["wave_corr"],
            "fr_cv": metrics["fr_cv"],
            "badge_isi":   badge_isi(isi),
            "badge_depth": badge_depth(metrics["depth_std_um"]),
            "badge_wave":  badge_waveform(metrics["wave_corr"]),
            "badge_fr":    badge_fr(metrics["fr_cv"]),
            "verdict": verdict,
        })
        print(f"  uid {uid}: {verdict}", flush=True)

    pd.DataFrame(rows).to_csv(VERDICTS_CSV, index=False)
    print(f"Wrote {VERDICTS_CSV}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
