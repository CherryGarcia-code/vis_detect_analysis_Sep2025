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
    depth_std_um, waveform_corr, fr_cv, isi_peak_agreement,
    baseline_psth_corr,
    badge_isi, badge_depth, badge_waveform, badge_fr, badge_isi_peak,
    badge_func_resp,
    composite_verdict,
    save_cache, load_cache,
    find_stable_subset,
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
    """Depth std, waveform corr, FR CV, ISI peak agreement, functional-response corr."""
    depths = np.array([r.peak_depth_um for r in uid.sessions], dtype=float)
    rates  = np.array([r.baseline_fr_hz for r in uid.sessions], dtype=float)
    waves = [r.waveform_peak for r in uid.sessions if r.waveform_peak is not None]
    if waves:
        min_len = min(w.size for w in waves)
        wf_stack = np.stack([w[:min_len] for w in waves])
    else:
        wf_stack = np.zeros((0, 0), dtype=np.float32)
    isi_hists = [r.isi_hist for r in uid.sessions]
    baseline_psths = [r.psths.get("baseline_on", (None, None, 0))[0] for r in uid.sessions]
    return {
        "depth_std_um":     depth_std_um(depths),
        "wave_corr":        waveform_corr(wf_stack),
        "fr_cv":            fr_cv(rates),
        "isi_peak_agree":   isi_peak_agreement(isi_hists),
        "func_resp_corr":   baseline_psth_corr(baseline_psths),
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

    # Precompute the per-UID stable subset (used by both the PDF renderer and
    # the trimmed-CSV loop).  Built once here so we don't recompute inside the
    # trimmed loop or duplicate metric work; both downstream consumers read
    # from `uid_trim_info`.
    uid_trim_info: Dict[int, Dict[str, object]] = {}
    for uid in uids_to_render:
        iv = intermediates[uid]
        if not iv.sessions:
            continue
        stable = find_stable_subset(iv)
        kept = stable["kept_indices"]
        dropped = stable["dropped_indices"]
        if kept:
            kept_sessions = [iv.sessions[i] for i in kept]
            trimmed_iv = UIDIntermediate(
                global_uid=iv.global_uid, span=len(kept_sessions),
                has_naive_to_expert=iv.has_naive_to_expert,
                suspect_known=iv.suspect_known, sessions=kept_sessions,
            )
            tm = compute_uid_metrics(trimmed_iv)
            tv = composite_verdict([
                badge_isi(isi_scores[uid]),
                badge_depth(tm["depth_std_um"]),
                badge_waveform(tm["wave_corr"]),
                badge_fr(tm["fr_cv"]),
                badge_isi_peak(tm["isi_peak_agree"]),
                badge_func_resp(tm["func_resp_corr"]),
            ])
        else:
            tm = None
            tv = "suspect"
        uid_trim_info[uid] = {
            "stable": stable,
            "kept_indices": kept,
            "dropped_indices": dropped,
            "trimmed_metrics": tm,
            "trimmed_verdict": tv,
        }

    for uid in uids_to_render:
        iv = intermediates[uid]
        if not iv.sessions:
            print(f"  uid {uid}: no sessions extracted, skipping"); continue
        metrics = compute_uid_metrics(iv)
        isi = isi_scores[uid]
        trim = uid_trim_info[uid]
        out_path = OUT_DIR / f"uid_{uid:04d}.pdf"
        # PDF stays at 4-badge layout (visual unchanged); write_uid_pdf returns
        # the 4-badge composite verdict it renders in the header.  Trim info
        # (dropped sessions, kept-count, trimmed verdict) is forwarded so the
        # renderer can visually mark dropped sessions throughout the PDF.
        verdict_pdf = write_uid_pdf(
            out_path, iv, pair_scores.get(uid),
            isi_score=isi,
            depth_std=metrics["depth_std_um"],
            wave_corr=metrics["wave_corr"],
            fr_cv_val=metrics["fr_cv"],
            dropped_indices=list(trim["dropped_indices"]),
            n_kept=len(trim["kept_indices"]),
            trimmed_verdict=str(trim["trimmed_verdict"]),
        )
        # CSV verdict incorporates the 5th badge (ISI peak-agreement) and the
        # 6th badge (Baseline_ON PSTH shape correlation) so the cross-session
        # bimodality and functional-tuning detectors are auditable.
        # Intentionally may differ from verdict_pdf — pdf_csv_disagree flags those rows.
        b_isi   = badge_isi(isi)
        b_depth = badge_depth(metrics["depth_std_um"])
        b_wave  = badge_waveform(metrics["wave_corr"])
        b_fr    = badge_fr(metrics["fr_cv"])
        b_peak  = badge_isi_peak(metrics["isi_peak_agree"])
        b_func  = badge_func_resp(metrics["func_resp_corr"])
        verdict_csv = composite_verdict([b_isi, b_depth, b_wave, b_fr, b_peak, b_func])
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
            "isi_peak_agree": metrics["isi_peak_agree"],
            "func_resp_corr": metrics["func_resp_corr"],
            "badge_isi":       b_isi,
            "badge_depth":     b_depth,
            "badge_wave":      b_wave,
            "badge_fr":        b_fr,
            "badge_isi_peak":  b_peak,
            "badge_func_resp": b_func,
            "verdict": verdict_csv,
            "verdict_pdf": verdict_pdf,
            "pdf_csv_disagree": verdict_csv != verdict_pdf,
        })
        print(f"  uid {uid}: csv={verdict_csv} pdf={verdict_pdf}", flush=True)

    pd.DataFrame(rows).to_csv(VERDICTS_CSV, index=False)
    print(f"Wrote {VERDICTS_CSV}", flush=True)

    # Per-UID stable-subset (Tier-2 rescue) analysis — reuses the precomputed
    # trim info to avoid recomputing find_stable_subset or trimmed metrics.
    trimmed_rows = []
    for uid in uids_to_render:
        iv = intermediates[uid]
        if not iv.sessions:
            continue
        trim = uid_trim_info[uid]
        kept = trim["kept_indices"]
        dropped = trim["dropped_indices"]
        tm = trim["trimmed_metrics"]
        tv = trim["trimmed_verdict"]
        if not kept:
            trimmed_rows.append({
                "global_uid": uid, "original_span": iv.span,
                "trimmed_span": 0,
                "dropped_sessions": ";".join(r.session_name for r in iv.sessions),
                "kept_sessions": "", "trimmed_verdict": "suspect",
                "rescued": False,
            })
            continue
        kept_sessions = [iv.sessions[i] for i in kept]
        dropped_sessions = [iv.sessions[i] for i in dropped]
        # Look up the original CSV verdict for comparison
        original_verdict = next((r["verdict"] for r in rows if r["global_uid"] == uid), "")
        rescued = (original_verdict == "suspect" and tv in ("trusted", "review")
                   and len(kept) >= 5)
        trimmed_rows.append({
            "global_uid": uid,
            "original_span": iv.span,
            "trimmed_span": len(kept),
            "n_dropped": len(dropped_sessions),
            "dropped_sessions": ";".join(r.session_name for r in dropped_sessions),
            "kept_sessions": ";".join(r.session_name for r in kept_sessions),
            "trimmed_depth_std_um":   tm["depth_std_um"],
            "trimmed_wave_corr":      tm["wave_corr"],
            "trimmed_fr_cv":          tm["fr_cv"],
            "trimmed_isi_peak_agree": tm["isi_peak_agree"],
            "trimmed_func_resp_corr": tm["func_resp_corr"],
            "original_verdict": original_verdict,
            "trimmed_verdict": tv,
            "rescued": rescued,
        })

    trimmed_csv = REPO_ROOT / "FIGURES" / "tracking_qc" / "verdicts_trimmed.csv"
    pd.DataFrame(trimmed_rows).to_csv(trimmed_csv, index=False)
    print(f"Wrote {trimmed_csv}  ({sum(1 for r in trimmed_rows if r['rescued'])} rescued)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
