"""fit_sync.py — Phase 2 of video sync: linear clock model from manual anchors.

Reads the v2 anchor JSON for a session, fits a linear clock model
(video_time_s = slope * nidaq_baseline_on_s + offset) from the anchors,
writes the canonical {session}_video_sync.json via save_video_sync, and
renders a slope-fitted barcode montage so the user can visually confirm
the fit holds across the session.

Requires at least 2 anchors in the anchor JSON (run click_anchor twice:
once for trial 0, once with --anchor-last).

Run:  py scripts/video/fit_sync.py --session 09092025
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import sys

import numpy as np

# matplotlib backend setup (script imports cv2 indirectly via shared helpers).
import matplotlib
matplotlib.use("Agg", force=True)

from visdetect.suite.loader import load_session
from visdetect.core.video_sync import (
    find_camera_files,
    load_camera_metadata,
    load_anchor,
    load_video_sync,
    save_video_sync,
    fit_2anchor_clock,
    fit_multianchor_clock,
    archive_sync_artifacts,
)
from visdetect.analysis.config import subject_video_sync_dir, canonical_camera_session

# Reuse the barcode-montage renderer from click_anchor.py
import importlib.util
_CA_SPEC = importlib.util.spec_from_file_location(
    "click_anchor",
    os.path.join(os.path.dirname(__file__), "click_anchor.py"),
)
_CA = importlib.util.module_from_spec(_CA_SPEC)
_CA_SPEC.loader.exec_module(_CA)
# click_anchor.py forces the interactive TkAgg backend at module scope when
# exec'd above; restore Agg so the montage renders headlessly (no GUI window).
matplotlib.use("Agg", force=True)
render_barcode_montage = _CA.render_barcode_montage
FIGS_DIR = _CA.FIGS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("fit_sync")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2: fit linear clock model from manual anchors.",
    )
    parser.add_argument("--session", required=True, help="Session name (e.g. 09092025).")
    parser.add_argument("--subject", default=None, help="Subject (default: config.SUBJECT).")
    args = parser.parse_args()

    try:
        session_name = canonical_camera_session(args.session)
    except (ValueError, TypeError):
        logger.error("Session name '%s' could not be parsed to a date.", args.session)
        return 2
    sync_dir = subject_video_sync_dir(args.subject)

    anchor_file = load_anchor(session_name, sync_dir=sync_dir)
    if anchor_file is None:
        logger.error("No anchor JSON for %s in %s. Run click_anchor first.",
                     session_name, sync_dir)
        return 2
    anchors = anchor_file["anchors"]
    if len(anchors) < 2:
        logger.error("Anchor JSON has %d anchor(s); need >=2.", len(anchors))
        return 2

    sess = load_session(session_name)
    baseline_on = np.asarray(sess.ni_events.get("Baseline_ON", []), dtype=float)
    baseline_on = baseline_on[baseline_on > 0]
    n_task_trials = len(sess.trials)
    if n_task_trials > 0 and len(baseline_on) > n_task_trials:
        baseline_on = baseline_on[:n_task_trials]
    n_baseline_on = int(len(baseline_on))
    del sess
    gc.collect()
    fps = float(anchor_file["frame_rate_fps"])

    try:
        if len(anchors) >= 3:
            sync_result = fit_multianchor_clock(anchors, n_baseline_on=n_baseline_on)
        else:
            sync_result = fit_2anchor_clock(
                anchors=anchors, fps=fps, n_baseline_on=n_baseline_on)
    except ValueError as exc:
        logger.error("Clock fit failed: %s", exc)
        return 2

    existing_sync = load_video_sync(session_name, sync_dir=sync_dir)
    if existing_sync is not None:
        prior = (existing_sync.get("eye_cam") or {}).get("per_trial_overrides") or {}
        if prior:
            sync_result.per_trial_overrides = {int(k): int(v) for k, v in prior.items()}

    # Never silent-overwrite: archive the prior sync JSON before writing.
    # include_anchor=False leaves the live anchor in place so a re-fit is
    # repeatable (fit_sync re-reads that anchor on the next run).
    archive_sync_artifacts(session_name, sync_dir=sync_dir, include_anchor=False)

    out_path = save_video_sync(
        session_name=session_name, eye_cam=sync_result, sync_dir=sync_dir)
    logger.info("Fit: slope=%.6f (%.2f ppm), offset=%.4f s, n=%d, cv_rmse=%.2f ms, quality=%s",
                sync_result.slope, sync_result.slope_ppm, sync_result.offset,
                sync_result.n_anchors, sync_result.cv_rmse_ms, sync_result.quality)
    print(f"Sync JSON: {out_path}")

    # Render the slope-fitted barcode montage. Use the new slope+offset_s
    # kwargs on render_barcode_montage (added in Step 4.0) so each row is
    # centred on the slope-fitted prediction for that trial.
    try:
        cam = find_camera_files(session_name, subject=args.subject)
    except Exception as exc:
        logger.error("Could not locate camera files for %s: %s", session_name, exc)
        return 2
    if "eye_cam" not in cam:
        logger.error(
            "No eye camera found for session %s. Cannot render montage.", session_name,
        )
        return 2
    video_path = cam["eye_cam"]["video"]
    ts_ms, _, _ = load_camera_metadata(cam["eye_cam"]["metadata"])

    # render_barcode_montage still requires the `anchor` arg for backwards
    # compat (it uses it for title metadata in the slope=1 path). For the
    # slope-fit path the anchor dict's fields are unused by the renderer,
    # but we pass a sentinel that makes the dict accesses safe.
    sentinel_anchor = {
        "session": session_name,
        "anchor_trial_index": -1,
        "nidaq_baseline_on_s": 0.0,
        "video_frame_idx": 0,
        "video_time_s": 0.0,
        "implied_offset_s": 0.0,
        "frame_rate_fps": fps,
        "n_trials": n_baseline_on,
        "clicked_at": "slope_fit",
    }
    montage_path = os.path.join(
        FIGS_DIR, f"{session_name}_barcode_montage_slopefit.png",
    )
    # render_barcode_montage expects the fit_2anchor orientation
    # (video_time_s = slope*nidaq + offset_s). fit_multianchor_clock stores the
    # INVERSE (nidaq = slope*cam + offset), so invert before handing it over.
    if sync_result.detection_method == "manual_multianchor":
        montage_slope = 1.0 / sync_result.slope
        montage_offset = -sync_result.offset / sync_result.slope
    else:
        montage_slope = sync_result.slope
        montage_offset = sync_result.offset
    render_barcode_montage(
        session_name=session_name,
        anchor=sentinel_anchor,
        baseline_on=baseline_on,
        video_path=video_path,
        ts_ms=ts_ms,
        fps=fps,
        out_path=montage_path,
        slope=montage_slope,
        offset_s=montage_offset,
    )
    print(f"Montage:   {montage_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
