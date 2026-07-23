"""Reconstruct a camera metadata CSV for sessions whose timestamp log is empty.

Some sessions saved a header-only camera ``*_metadata.csv`` (just the column
header, no per-frame rows), so the per-frame timestamps the video-sync pipeline
needs were never written. Known affected BG_046 sessions: 12082025, 260725.

The camera directory on ``CAMERA_ROOT`` (X:/ceph) is treated as STRICTLY
READ-ONLY: nothing is backed up, overwritten, or written there. The
reconstructed CSV + provenance land under ``subject_video_sync_dir`` (local
cache), and ``find_camera_files`` transparently prefers that local CSV.

For each requested camera this script:
  1. Probes the camera's own .mp4 for frame count + container fps (OpenCV).
  2. Refuses to reconstruct a metadata file that already has real data (unless
     --force).
  3. Writes a reconstructed ``<session>_<cam>_metadata.reconstructed.csv`` with
     steady-fps timestamps (ts[i] = i * 1000/fps) under ``subject_video_sync_dir``,
     consumed unchanged by load_camera_metadata / find_camera_files.
  4. Writes a ``*.reconstructed.json`` provenance sidecar alongside it (local).
  5. Round-trips the written LOCAL CSV through load_camera_metadata to verify it.

Why linear is good enough: the BG_046 eye/front cameras run a metronomic
~50 fps with no frame drops (verified on reference session 140825), and
fit_sync fits a slope (NI-DAQ -> video time) that absorbs any constant fps
error. Accuracy depends on linearity, not the exact fps value.

Usage:
    py scripts/video/reconstruct_camera_metadata.py --session 12082025 --dry-run
    py scripts/video/reconstruct_camera_metadata.py --session 12082025
    py scripts/video/reconstruct_camera_metadata.py --session 260725
    py scripts/video/reconstruct_camera_metadata.py --session 12082025 --cameras eye_cam
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Optional

import numpy as np

from visdetect.core.video_sync import (
    find_camera_files,
    load_camera_metadata,
    metadata_is_header_only,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("reconstruct_camera_metadata")


def _probe_video(video_path: str):
    """Return (frame_count, fps) from the video container via OpenCV."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if frame_count <= 0 or fps <= 0:
        raise ValueError(
            f"Invalid frame_count={frame_count} / fps={fps} from {video_path}"
        )
    return frame_count, fps


def reconstruct_camera(
    session_name: str, cam_label: str, video_path: str, meta_path: str,
    force: bool = False, dry_run: bool = False, subject: Optional[str] = None,
) -> bool:
    """Reconstruct one camera's metadata to LOCAL cache. Returns True if written.

    The camera directory on ``CAMERA_ROOT`` (X:) is strictly read-only: nothing
    is backed up, overwritten, or written there. The reconstructed CSV and its
    provenance sidecar land under ``subject_video_sync_dir`` instead.
    """
    if not force and not metadata_is_header_only(meta_path):
        logger.warning(
            "[%s/%s] metadata already has per-frame data; skipping "
            "(use --force to overwrite). %s",
            session_name, cam_label, meta_path,
        )
        return False

    frame_count, fps = _probe_video(video_path)
    logger.info(
        "[%s/%s] %d frames @ %.4f fps (~%.1f min). video=%s",
        session_name, cam_label, frame_count, fps, frame_count / fps / 60.0,
        video_path,
    )

    if dry_run:
        logger.info("[%s/%s] --dry-run: no files written.", session_name, cam_label)
        return False

    from visdetect.core.video_sync import write_local_reconstructed_metadata
    local_csv = write_local_reconstructed_metadata(
        session_name, cam_label, frame_count, fps, subject=subject)
    logger.info("[%s/%s] reconstructed (LOCAL, X: untouched) -> %s",
                session_name, cam_label, local_csv)
    prov = {
        "session": session_name, "camera": cam_label, "source": "RECONSTRUCTED_LOCAL",
        "method": "linear steady-fps (ts[i] = i*1000/fps) from video container",
        "frame_count": frame_count, "fps": fps, "duration_s": frame_count / fps,
        "video": video_path, "reconstructed_at": datetime.now().isoformat(timespec="seconds"),
        "tool": "scripts/video/reconstruct_camera_metadata.py",
    }
    prov_path = local_csv[: -len(".csv")] + ".json"
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2)

    # Round-trip verification through the real loader (against the LOCAL CSV).
    ts_ms, _, _ = load_camera_metadata(local_csv)
    if len(ts_ms) != frame_count:
        raise RuntimeError(
            f"[{session_name}/{cam_label}] round-trip failed: loader returned "
            f"{len(ts_ms)} timestamps, expected {frame_count}."
        )
    if not np.all(np.diff(ts_ms) > 0):
        raise RuntimeError(
            f"[{session_name}/{cam_label}] round-trip failed: timestamps not "
            f"strictly increasing."
        )
    logger.info(
        "[%s/%s] wrote %s (%d timestamps verified) + %s",
        session_name, cam_label, local_csv, len(ts_ms), prov_path,
    )
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct header-only camera metadata CSVs from the video."
    )
    parser.add_argument("--session", required=True,
                        help="Session name (e.g. 12082025 or 260725).")
    parser.add_argument("--subject", default=None,
                        help="Subject id (default: config.SUBJECT, e.g. BG_046).")
    parser.add_argument("--cameras", default="eye_cam,front_cam",
                        help="Comma-separated camera labels (default: eye_cam,front_cam).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite metadata even if it already has data.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report frame count / fps without writing anything.")
    args = parser.parse_args(argv)

    from visdetect.analysis.config import canonical_camera_session
    try:
        session_name = canonical_camera_session(args.session)
    except (TypeError, ValueError):
        logger.error("Session name '%s' could not be parsed to a date.", args.session)
        return 2

    try:
        cam_files = find_camera_files(session_name, subject=args.subject)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    selected = [c.strip() for c in args.cameras.split(",") if c.strip()]
    n_written = 0
    for cam in selected:
        if cam not in cam_files:
            logger.warning(
                "[%s/%s] no video+metadata pair found; skipping.", session_name, cam
            )
            continue
        wrote = reconstruct_camera(
            session_name, cam,
            cam_files[cam]["video"], cam_files[cam]["metadata"],
            force=args.force, dry_run=args.dry_run, subject=args.subject,
        )
        n_written += int(wrote)

    logger.info("Done: %d camera metadata file(s) written for %s.",
                n_written, session_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
