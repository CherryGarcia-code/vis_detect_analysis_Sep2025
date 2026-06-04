#!/usr/bin/env python3
"""Build per-UID QC sheets for a cross-session tracking long-track cohort.

Defaults reproduce the UnitMatch ``output/all42`` cohort exactly. Every input
(registry, prob-matrix + its row-index, ISI stats) and every output path is
overridable, so the same sheets can be built for an alternative tracker (e.g. a
DeepUnitMatch fine-tune run) whose output uses a flat layout (``unit_index.csv``
+ ``prob_matrix.npy`` with no ``batch0/`` subdir). Raw waveforms and pkls are
keyed by (session, ks_unit_id) and therefore shared across trackers.

See docs/superpowers/specs/2026-05-21-tracking-qc-sheets-design.md and
docs/superpowers/plans/2026-05-22-tracking-qc-sheets-plan.md.

Usage:
    # default UM cohort (unchanged behaviour)
    py scripts/pipelines/tracking/build_qc_sheets.py [--rebuild-cache] [--uids 334 ...]

    # alternative tracker (e.g. DeepUM rung3-ep0), flat output layout
    py scripts/pipelines/tracking/build_qc_sheets.py --rebuild-cache \
        --registry   <run>/unit_index.csv \
        --prob-matrix <run>/prob_matrix.npy --prob-index <run>/unit_index.csv \
        --isi-stats  FIGURES/tracking_qc/track_validation_stats_rung3ep0.csv \
        --out-dir    FIGURES/tracking_qc/per_uid_sheets_rung3ep0 \
        --verdicts-csv FIGURES/tracking_qc/verdicts_rung3ep0.csv \
        --cache-path data/cache/tracking_qc_intermediates_rung3ep0.pkl
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from datetime import datetime
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
    load_isi_scores,
    depth_std_um, waveform_corr, fr_cv,
    isi_peak_agreement, baseline_psth_corr, baseline_isi_hist_corr,
    badge_isi, badge_depth, badge_waveform, badge_fr,
    badge_isi_peak, badge_func_resp, badge_isi_hist_corr, composite_verdict,
    apply_isi_autopass,
    estimate_session_drift, depth_std_um_corrected,
    save_cache, load_cache,
    find_stable_subset,
)
from visdetect.core.session import load_session                 # noqa: E402
from visdetect.suite.loader import load_staging_manifest, load_filtered_manifest  # noqa: E402

from qc_sheet_figures import write_uid_pdf                       # noqa: E402

# Defaults reproduce the UnitMatch output/all42 cohort. All are overridable on
# the CLI (see main()) so the same sheets can be built for any tracker.
DEFAULT_UM_ROOT      = Path("X:/public/projects/BeJG_20230130_VisDetect/wEPhys/"
                            "BG_046/unit_match/output/all42")
DEFAULT_REGISTRY     = DEFAULT_UM_ROOT / "unit_index.csv"
# Prob-matrix + its row-index. UM stores these under batch0/ (matrix named
# output_prob_matrix.npy); a flat fine-tune run stores prob_matrix.npy +
# unit_index.csv directly. Both are row-aligned with their own registry.
DEFAULT_PROB_MATRIX  = DEFAULT_UM_ROOT / "batch0" / "output_prob_matrix.npy"
DEFAULT_PROB_INDEX   = DEFAULT_UM_ROOT / "batch0" / "unit_index.csv"
DEFAULT_ISI_STATS    = REPO_ROOT / "FIGURES" / "tracking_qc" / "track_validation_stats.csv"
DEFAULT_RAW_WF_ROOT  = REPO_ROOT / "data" / "unit_match" / "input" / "BG_046"
DEFAULT_PKL_DIR      = REPO_ROOT / "data" / "pkls" / "BG_046"

DEFAULT_OUT_DIR      = REPO_ROOT / "FIGURES" / "tracking_qc" / "per_uid_sheets"
DEFAULT_VERDICTS_CSV = REPO_ROOT / "FIGURES" / "tracking_qc" / "verdicts.csv"
DEFAULT_CACHE_PATH   = REPO_ROOT / "data" / "cache" / "tracking_qc_intermediates.pkl"


def _session_pkl(session_name: str, pkl_dir: Path) -> Optional[Path]:
    for s in (session_name, session_name.zfill(8)):
        p = pkl_dir / f"BG_046_{s}.pkl"
        if p.exists():
            return p
    return None


def build_cache(unit_index_df: pd.DataFrame, cohort: pd.DataFrame,
                manifest: pd.DataFrame, raw_wf_root: Path, pkl_dir: Path,
                ) -> Dict[int, UIDIntermediate]:
    """Outer loop by session.  Returns dict[uid -> UIDIntermediate]."""

    def _norm_session(name) -> str:
        """Normalize session name to 8-char zero-padded DDMMYYYY string.

        unit_index.csv stores 'session' as int64, so July sessions like
        1072025 become the 7-char string '1072025' after astype(str). The
        staging manifest stores session_name as an 8-char string already
        ('01072025'). Both sides must be zfill(8) before comparison or
        chronological-sort lookups silently miss.
        """
        return str(name).zfill(8)

    def _parse_session_date(name) -> datetime:
        """Parse session_name as a DDMMYYYY date.

        Works for ALL sessions (manifest-present and Unknown alike), so
        chronological sort doesn't depend on the manifest containing every
        session. Critical for the looser tracking-QC manifest (spec §3.4),
        which deliberately excludes <150-trial sessions that nonetheless
        appear in the UnitMatch cache.
        """
        return datetime.strptime(_norm_session(name), '%d%m%Y')

    stage_by_session = {_norm_session(r["session_name"]): str(r["stage"])
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

    sess_set = sorted(
        {s for ksmap in uid_to_ks.values() for s in ksmap.keys()},
        key=_parse_session_date,
    )

    for sess in sess_set:
        pkl = _session_pkl(sess, pkl_dir)
        if pkl is None:
            print(f"  skip {sess}: no pkl", flush=True); continue
        t0 = time.time()
        S = load_session(str(pkl))
        chan_pos = load_channel_positions(raw_wf_root, sess)
        uids_here = [u for u, ks in uid_to_ks.items() if sess in ks]
        ks_ids_here = [uid_to_ks[u][sess] for u in uids_here]
        records = extract_session_records(
            S, ks_ids_here, session_name=sess,
            stage=stage_by_session.get(_norm_session(sess), "Unknown"),
            raw_wf_root=raw_wf_root, channel_positions=chan_pos,
        )
        for u in uids_here:
            rec = records.get(int(uid_to_ks[u][sess]))
            if rec is not None:
                intermediates[u].sessions.append(rec)
        del S
        gc.collect()
        print(f"  {sess}: {len(records)}/{len(uids_here)} cached "
              f"in {time.time() - t0:.1f}s", flush=True)

    for uid in intermediates:
        intermediates[uid].sessions.sort(
            key=lambda r: _parse_session_date(r.session_name)
        )
    return intermediates


def _depth_for_badge(metrics: Dict[str, float]) -> float:
    """Prefer drift-corrected depth_std for the badge; fall back to raw if NaN.

    Drift correction (via UM high-confidence anchor matches) removes apparent
    depth instability that is really just probe drift. UIDs whose sessions
    fall outside the drift-correction chain (NaN corrected) get judged by the
    raw value as before.
    """
    corrected = metrics.get("depth_std_corrected_um", float("nan"))
    if np.isfinite(corrected):
        return float(corrected)
    return float(metrics.get("depth_std_um", float("nan")))


def compute_uid_metrics(uid: UIDIntermediate,
                         drift_offsets: Optional[Dict[str, float]] = None,
                         ) -> Dict[str, float]:
    """Depth std, waveform corr, FR CV, ISI peak agreement, functional-response corr, ISI hist corr.

    If `drift_offsets` is provided, also compute depth_std_corrected_um (informational
    only — not used by badge logic yet).
    """
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
    out = {
        "depth_std_um":     depth_std_um(depths),
        "wave_corr":        waveform_corr(wf_stack),
        "fr_cv":            fr_cv(rates),
        "isi_peak_agree":   isi_peak_agreement(isi_hists),
        "func_resp_corr":   baseline_psth_corr(baseline_psths),
        "isi_hist_corr":    baseline_isi_hist_corr(isi_hists),
    }
    if drift_offsets:
        out["depth_std_corrected_um"] = depth_std_um_corrected(uid, drift_offsets)
    else:
        out["depth_std_corrected_um"] = float("nan")
    return out


def _pair_scores_from_paths(matrix_path, index_path,
                            uid_to_sessions: Dict[int, List[str]],
                            uid_to_ks: Dict[int, Dict[str, int]],
                            ) -> Dict[int, np.ndarray]:
    """Per-UID consecutive-session match probabilities from an explicit
    prob-matrix + its row-index unit_index.csv.

    Faithful re-implementation of tracking_qc.load_um_pair_scores but taking
    explicit file paths instead of a UM root, so it also handles the flat
    fine-tune output layout (prob_matrix.npy + unit_index.csv, no batch0/).
    For the default UM paths it reads the same files load_um_pair_scores would
    and returns identical arrays. Returns {uid: ndarray(n_sess-1,)}; all-empty
    if either file is missing.
    """
    matrix_path, index_path = Path(matrix_path), Path(index_path)
    if not matrix_path.exists() or not index_path.exists():
        print(f"  pair-score matrix/index missing ({matrix_path.name}) — "
              f"sheets will show no match-probability trace", flush=True)
        return {uid: np.array([]) for uid in uid_to_sessions}
    mat = np.load(matrix_path)
    idx = pd.read_csv(index_path)
    idx["session"] = idx["session"].astype(str)
    lookup: Dict[Tuple[str, int], int] = {}
    for i, row in idx.iterrows():
        lookup[(str(row["session"]), int(row["ks_unit_id"]))] = i

    out: Dict[int, np.ndarray] = {}
    for uid, sess_list in uid_to_sessions.items():
        ks_map = uid_to_ks.get(uid, {})
        rows = [lookup.get((s, int(ks_map[s]))) if s in ks_map else None
                for s in sess_list]
        scores = [np.nan if (a is None or b is None) else float(mat[a, b])
                  for a, b in zip(rows[:-1], rows[1:])]
        out[uid] = np.array(scores, dtype=float)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--uids", type=int, nargs="*", default=None,
                        help="Only render these UIDs (cohort filter still applies)")
    parser.add_argument("--max-uids", type=int, default=None,
                        help="Render at most N UIDs (debug)")
    parser.add_argument(
        "--shared-baseline", action="store_true",
        help="Use one Baseline_ON-derived baseline scalar for ALL heatmaps "
             "in each UID's page 2 (cross-event comparison mode). Default: "
             "per-event baseline from EVENT_RESPONSIVENESS_WINDOWS.",
    )
    # ── Input/output paths (defaults reproduce the UM all42 cohort) ──────────
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY,
                        help="unit_index.csv (session, ks_unit_id, global_uid)")
    parser.add_argument("--prob-matrix", type=Path, default=DEFAULT_PROB_MATRIX,
                        help="Match-probability matrix; used for pair-score "
                             "traces AND probe-drift estimation")
    parser.add_argument("--prob-index", type=Path, default=DEFAULT_PROB_INDEX,
                        help="unit_index.csv giving the row order of --prob-matrix")
    parser.add_argument("--isi-stats", type=Path, default=DEFAULT_ISI_STATS,
                        help="track_validation_stats.csv (span + median ISI corr)")
    parser.add_argument("--raw-wf-root", type=Path, default=DEFAULT_RAW_WF_ROOT,
                        help="UnitMatch raw-waveform input root (shared across trackers)")
    parser.add_argument("--pkl-dir", type=Path, default=DEFAULT_PKL_DIR,
                        help="Session pkl directory (shared across trackers)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Directory for uid_XXXX.pdf sheets")
    parser.add_argument("--verdicts-csv", type=Path, default=DEFAULT_VERDICTS_CSV,
                        help="Per-UID verdict CSV (trimmed CSV derived alongside)")
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH,
                        help="Intermediate cache pkl (use a distinct path per tracker)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading manifest + cohort ...", flush=True)
    # Tracking-QC uses a looser filter than behavioral analyses: keeps engaged
    # sessions (>=150 trials) regardless of d', so early-Naive/Learning sessions
    # with poor performance but enough trials are still tracked across stages.
    # min_dprime=0.8 (the SDT default) wrongly excludes the very sessions needed
    # for cross-stage tracking studies. See spec §3.4.
    #
    # include_stages mirrors the prior qc_only=True behavior (no Disengaged);
    # merge_naive_learning=True mirrors SESSION_FILTER so Naive sessions are
    # relabeled "Learning" in the stage column (downstream STAGE_ORDER is
    # ["Learning", "Expert"] only — Naive-as-Naive would be silently dropped).
    manifest = load_filtered_manifest(
        include_stages=["Naive", "Learning", "Expert"],
        merge_naive_learning=True,
        min_trials=150,
        min_dprime=None,
    )
    unit_index_df = pd.read_csv(args.registry)
    cohort = select_long_tracks(args.registry, args.isi_stats, min_span=10)
    cohort = annotate_naive_to_expert(cohort, manifest)
    print(f"  cohort size: {len(cohort)}", flush=True)

    if args.rebuild_cache or not args.cache_path.exists():
        print("Building cache (this is slow — outer loop by session) ...", flush=True)
        intermediates = build_cache(unit_index_df, cohort, manifest,
                                    args.raw_wf_root, args.pkl_dir)
        save_cache(intermediates, args.cache_path)
        print(f"  saved cache to {args.cache_path}", flush=True)
    else:
        print(f"Loading cached intermediates from {args.cache_path}", flush=True)
        intermediates = load_cache(args.cache_path)

    uid_to_sessions = {u: [r.session_name for r in iv.sessions]
                       for u, iv in intermediates.items()}
    uid_to_ks = {}
    for _, row in unit_index_df.iterrows():
        uid = int(row["global_uid"])
        uid_to_ks.setdefault(uid, {})[str(row["session"])] = int(row["ks_unit_id"])
    pair_scores = _pair_scores_from_paths(args.prob_matrix, args.prob_index,
                                          uid_to_sessions, uid_to_ks)

    isi_scores = load_isi_scores(args.isi_stats)

    # Estimate per-session probe drift from high-confidence matches.
    # --prob-matrix must be row-aligned with --registry (same run): true for the
    # UM batch0/output_prob_matrix.npy + all42/unit_index.csv pair (verified) and
    # for a flat fine-tune run's prob_matrix.npy + its own unit_index.csv.
    #
    # NOTE: we walk ALL sessions in the registry (not just the QC-filtered
    # manifest sessions). The manifest skips QC-failing sessions, which can leave
    # multi-week gaps between consecutive pairs — too long for reliable high-prob
    # anchors. Using all registry sessions gives a dense chain that is then
    # sampled by SessionRecord.session_name at lookup time.
    prob_matrix_path = args.prob_matrix
    if prob_matrix_path.exists():
        prob_matrix = np.load(prob_matrix_path)
        # Chronological order over all sessions. Names are DDMMYYYY (sometimes
        # D-MMYYYY with leading zero stripped); pad to 8 chars then sort by
        # (year, month, day) for a true date order.
        def _date_key(s: str) -> Tuple[int, int, int]:
            p = str(s).zfill(8)
            return (int(p[4:8]), int(p[2:4]), int(p[0:2]))
        um_sessions_all = sorted(
            unit_index_df["session"].astype(str).unique().tolist(),
            key=_date_key,
        )
        print(f"Estimating cross-session probe drift across {len(um_sessions_all)} "
              f"sessions ...", flush=True)
        drift_offsets = estimate_session_drift(
            unit_index_df, prob_matrix, args.raw_wf_root, um_sessions_all,
        )
        n_finite = sum(1 for v in drift_offsets.values() if np.isfinite(v))
        # drift_offsets has 2 keys per session (raw + zfill(8)); count uniques.
        unique_sess = set(str(s).zfill(8) for s in drift_offsets.keys())
        n_finite_unique = sum(
            1 for s in unique_sess
            if np.isfinite(drift_offsets.get(s, drift_offsets.get(s.lstrip("0"), float("nan"))))
        )
        print(f"  drift offsets computed for {n_finite_unique}/{len(um_sessions_all)} "
              f"sessions", flush=True)
    else:
        drift_offsets = {}
        print(f"prob matrix missing ({prob_matrix_path}) — drift correction "
              f"disabled", flush=True)

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
        skipped = stable["skipped_indices"]
        dropped = stable["dropped_indices"]
        if kept:
            kept_sessions = [iv.sessions[i] for i in kept]
            trimmed_iv = UIDIntermediate(
                global_uid=iv.global_uid, span=len(kept_sessions),
                has_naive_to_expert=iv.has_naive_to_expert,
                suspect_known=iv.suspect_known, sessions=kept_sessions,
            )
            tm = compute_uid_metrics(trimmed_iv, drift_offsets=drift_offsets)
            b_isi_trim   = badge_isi(isi_scores[uid])
            b_depth_trim = badge_depth(_depth_for_badge(tm))
            b_wave_trim  = badge_waveform(tm["wave_corr"])
            b_fr_trim    = badge_fr(tm["fr_cv"])
            b_hist_trim  = badge_isi_hist_corr(tm["isi_hist_corr"])
            b_func_trim  = badge_func_resp(tm["func_resp_corr"])
            tv_pre_autopass = composite_verdict([
                b_isi_trim, b_depth_trim, b_wave_trim, b_fr_trim, b_hist_trim, b_func_trim,
            ])
            tv = apply_isi_autopass(
                tv_pre_autopass, tm["isi_hist_corr"], b_wave_trim, b_depth_trim,
            )
            trimmed_autopass_applied = (tv != tv_pre_autopass)
        else:
            tm = None
            tv = "suspect"
            trimmed_autopass_applied = False
        uid_trim_info[uid] = {
            "stable": stable,
            "kept_indices": kept,
            "skipped_indices": skipped,
            "dropped_indices": dropped,
            "trimmed_metrics": tm,
            "trimmed_verdict": tv,
            "trimmed_autopass_applied": trimmed_autopass_applied,
        }

    for uid in uids_to_render:
        iv = intermediates[uid]
        if not iv.sessions:
            print(f"  uid {uid}: no sessions extracted, skipping"); continue
        metrics = compute_uid_metrics(iv, drift_offsets=drift_offsets)
        isi = isi_scores[uid]
        trim = uid_trim_info[uid]
        out_path = args.out_dir / f"uid_{uid:04d}.pdf"
        # PDF stays at 4-badge layout (visual unchanged); write_uid_pdf returns
        # the 4-badge composite verdict it renders in the header.  Trim info
        # (dropped sessions, kept-count, trimmed verdict) is forwarded so the
        # renderer can visually mark dropped sessions throughout the PDF.
        # Skipped sessions render identically to dropped per spec §3.5;
        # union both into the single "visually dim" set the renderer expects.
        visually_dropped = sorted(set(trim["dropped_indices"]) | set(trim["skipped_indices"]))
        verdict_pdf = write_uid_pdf(
            out_path, iv, pair_scores.get(uid),
            isi_score=isi,
            depth_std=metrics["depth_std_um"],
            wave_corr=metrics["wave_corr"],
            fr_cv_val=metrics["fr_cv"],
            dropped_indices=visually_dropped,
            n_kept=len(trim["kept_indices"]),
            trimmed_verdict=str(trim["trimmed_verdict"]),
            shared_baseline=args.shared_baseline,
        )
        # CSV verdict incorporates the 5th badge (ISI histogram cross-session
        # correlation) and the 6th badge (Baseline_ON PSTH shape correlation)
        # so the full-shape bimodality and functional-tuning detectors are
        # auditable. badge_isi_peak is retained in the CSV for diagnostic
        # transparency but is NOT part of the composite.
        # Intentionally may differ from verdict_pdf — pdf_csv_disagree flags those rows.
        b_isi   = badge_isi(isi)
        b_depth = badge_depth(_depth_for_badge(metrics))
        b_wave  = badge_waveform(metrics["wave_corr"])
        b_fr    = badge_fr(metrics["fr_cv"])
        b_peak  = badge_isi_peak(metrics["isi_peak_agree"])     # kept for CSV transparency only
        b_func  = badge_func_resp(metrics["func_resp_corr"])
        b_hist  = badge_isi_hist_corr(metrics["isi_hist_corr"])
        verdict_csv_pre_autopass = composite_verdict([b_isi, b_depth, b_wave, b_fr, b_hist, b_func])
        verdict_csv = apply_isi_autopass(verdict_csv_pre_autopass, metrics["isi_hist_corr"], b_wave, b_depth)
        autopass_applied = (verdict_csv != verdict_csv_pre_autopass)
        rows.append({
            "global_uid": uid,
            "span": iv.span,
            "sessions": ";".join(r.session_name for r in iv.sessions),
            "has_naive_to_expert": iv.has_naive_to_expert,
            "suspect_known": iv.suspect_known,
            "isi_median": isi,
            "depth_std_um": metrics["depth_std_um"],
            "depth_std_corrected_um": metrics["depth_std_corrected_um"],
            "wave_corr": metrics["wave_corr"],
            "fr_cv": metrics["fr_cv"],
            "isi_peak_agree":      metrics["isi_peak_agree"],
            "isi_hist_corr":       metrics["isi_hist_corr"],
            "func_resp_corr":      metrics["func_resp_corr"],
            "badge_isi":           b_isi,
            "badge_depth":         b_depth,
            "badge_wave":          b_wave,
            "badge_fr":            b_fr,
            "badge_isi_peak":      b_peak,
            "badge_isi_hist_corr": b_hist,
            "badge_func_resp":     b_func,
            "autopass_applied":    autopass_applied,
            "verdict": verdict_csv,
            "verdict_pdf": verdict_pdf,
            "pdf_csv_disagree": verdict_csv != verdict_pdf,
        })
        print(f"  uid {uid}: csv={verdict_csv} pdf={verdict_pdf}", flush=True)

    pd.DataFrame(rows).to_csv(args.verdicts_csv, index=False)
    print(f"Wrote {args.verdicts_csv}", flush=True)

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
                "skipped_sessions": "",
                "kept_sessions": "", "trimmed_verdict": "suspect",
                "trimmed_autopass_applied": False,
                "rescued": False,
            })
            continue
        kept_sessions = [iv.sessions[i] for i in kept]
        skipped_sessions = [iv.sessions[i] for i in trim["skipped_indices"]]
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
            "skipped_sessions": ";".join(r.session_name for r in skipped_sessions),
            "kept_sessions": ";".join(r.session_name for r in kept_sessions),
            "trimmed_depth_std_um":            tm["depth_std_um"],
            "trimmed_depth_std_corrected_um": tm["depth_std_corrected_um"],
            "trimmed_wave_corr":               tm["wave_corr"],
            "trimmed_fr_cv":                   tm["fr_cv"],
            "trimmed_isi_peak_agree":          tm["isi_peak_agree"],
            "trimmed_isi_hist_corr":           tm["isi_hist_corr"],
            "trimmed_func_resp_corr":          tm["func_resp_corr"],
            "original_verdict": original_verdict,
            "trimmed_verdict": tv,
            "trimmed_autopass_applied": trim["trimmed_autopass_applied"],
            "rescued": rescued,
        })

    trimmed_csv = args.verdicts_csv.with_name(
        args.verdicts_csv.stem + "_trimmed.csv")
    pd.DataFrame(trimmed_rows).to_csv(trimmed_csv, index=False)
    print(f"Wrote {trimmed_csv}  ({sum(1 for r in trimmed_rows if r['rescued'])} rescued)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
