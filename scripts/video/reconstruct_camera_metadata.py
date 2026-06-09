"""Reconstruct a camera metadata CSV for sessions whose timestamp log is empty.

Some sessions saved a header-only camera ``*_metadata.csv`` (just the column
header, no per-frame rows), so the per-frame timestamps the video-sync pipeline
needs were never written. Known affected BG_046 sessions: 12082025, 260725.

For each requested camera this script:
  1. Probes the camera's own .mp4 for frame count + container fps (OpenCV).
  2. Refuses to touch a metadata file that already has real data (unless --force).
  3. Backs up the original header-only CSV to ``*_metadata.header_only.bak``
     (an existing backup is preserved, never overwritten).
  4. Writes a reconstructed ``*_metadata.csv`` with steady-fps timestamps
     (ts[i] = i * 1000/fps), consumed unchanged by load_camera_metadata /
     find_camera_files.
  5. Writes a ``*_metadata.reconstructed.json`` provenance sidecar.
  6. Round-trips the written CSV through load_camera_metadata to verify it.

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

import numpy as np

from visdetect.core.video_sync import (
    find_camera_files,
    load_camera_metadata,
    metadata_is_header_only,
    backup_header_only_metadata,
    write_reconstructed_metadata,
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


def _provenance_path(meta_path: str) -> str:
    suffix = "_metadata.csv"
    if meta_path.endswith(suffix):
        return meta_path[: -len(suffix)] + "_metadata.reconstructed.json"
    return meta_path + ".reconstructed.json"


def reconstruct_camera(
    session_name: str, cam_label: str, video_path: str, meta_path: str,
    force: bool = False, dry_run: bool = False,
) -> bool:
    """Reconstruct one camera's metadata. Returns True if a CSV was written."""
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

    bak_path = backup_header_only_metadata(meta_path)
    logger.info("[%s/%s] original backed up -> %s", session_name, cam_label, bak_path)

    write_reconstructed_metadata(meta_path, frame_count, fps)

    prov = {
        "session": session_name,
        "camera": cam_label,
        "source": "RECONSTRUCTED",
        "method": "linear steady-fps (ts[i] = i * 1000/fps) from video container",
        "reason": "original camera metadata CSV was header-only (no per-frame rows)",
        "frame_count": frame_count,
        "fps": fps,
        "duration_s": frame_count / fps,
        "video": video_path,
        "original_backup": bak_path,
        "reconstructed_at": datetime.now().isoformat(timespec="seconds"),
        "tool": "scripts/video/reconstruct_camera_metadata.py",
    }
    prov_path = _provenance_path(meta_path)
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2)

    # Round-trip verification through the real loader.
    ts_ms, _, _ = load_camera_metadata(meta_path)
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
        session_name, cam_label, meta_path, len(ts_ms), prov_path,
    )
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct header-only camera metadata CSVs from the video."
    )
    parser.add_argument("--session", required=True,
                        help="Session name (e.g. 12082025).")
    parser.add_argument("--cameras", default="eye_cam,front_cam",
                        help="Comma-separated camera labels (default: eye_cam,front_cam).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite metadata even if it already has data.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report frame count / fps without writing anything.")
    args = parser.parse_args(argv)

    try:
        session_name = str(int(args.session)).zfill(8)
    except (TypeError, ValueError):
        logger.error("Session name '%s' is not numeric.", args.session)
        return 2

    try:
        cam_files = find_camera_files(session_name)
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
            force=args.force, dry_run=args.dry_run,
        )
        n_written += int(wrote)

    logger.info("Done: %d camera metadata file(s) written for %s.",
                n_written, session_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
