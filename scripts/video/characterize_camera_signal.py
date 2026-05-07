"""Characterize camera signal quality for video-neural synchronization.

Systematically evaluates what visual features are present in the eye camera
video around known trial events (Baseline_ON). Extracts multiple candidate
features across multiple ROIs, computes quality metrics (SNR, detection rate,
timing jitter), and compares across sessions spanning learning stages.

Camera geometry: Eye camera captures a profile view of the mouse head.
The computer screen is BEHIND the mouse, so the background IS the screen.
The mouse head + lick spout occlude the central portion.

Outputs:
  - Per-session diagnostic figure  → figures/video_sync/characterize/
  - Cross-session comparison figure → figures/video_sync/characterize/
  - Feature quality summary CSV    → data/cache/video_sync/feature_characterization.csv
  - Cached NPZ per session         → data/cache/video_sync/characterize/

Usage:
    py scripts/video/characterize_camera_signal.py
    py scripts/video/characterize_camera_signal.py --sessions 27062025 --n-trials 20
    py scripts/video/characterize_camera_signal.py --sessions 27062025 03072025 --force
"""

import argparse
import gc
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ── Project paths ────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "analysis_suite"))
sys.path.insert(0, _PROJECT_ROOT)

from src.visdetect.core.session import load_session
from src.visdetect.core.video_sync import (
    find_camera_files,
    load_camera_metadata,
    coarse_align,
)
from src.visdetect.analysis.config import (
    CAMERA_ROOT,
    VIDEO_SYNC_DIR,
    VIDEO_SYNC_FIG_DIR,
    load_staging_manifest,
    parse_session_date,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
DEFAULT_SESSIONS = ["27062025", "03072025", "14082025", "29082025", "09092025"]
DEFAULT_N_TRIALS = 50
WINDOW_S = 2.0  # ±2s around Baseline_ON
FPS_NOMINAL = 50  # expected eye cam frame rate

# Pre/post windows for SNR computation (seconds relative to Baseline_ON)
PRE_WINDOW = (-2.0, -0.1)
POST_WINDOW = (0.1, 1.0)

# Detection threshold (MAD multiplier)
DETECT_MAD_MULT = 3.0
DETECT_SEARCH_S = 0.5  # search ±0.5s around expected onset

# Local video directory (faster than network X: drive)
LOCAL_VIDEO_DIR = os.path.join(_PROJECT_ROOT, "data", "videos")

# ── ROI definitions (y0:y1, x0:x1) ──────────────────────────────
# Frame is ~1024 (H) x 976 (W) based on existing code
ROI_DEFS = {
    "full_frame": (0, 1024, 0, 976),
    "above_head": (0, 200, 0, 976),
    "below_head": (800, 1024, 0, 976),
    "top_right": (0, 200, 600, 976),
    "top_left": (0, 200, 0, 400),
    "mouse_head": (200, 750, 200, 800),
    # "background_combined" is handled specially (above_head + below_head)
}
# background_combined uses the union of above_head and below_head
_BG_COMBINED_ROIS = ["above_head", "below_head"]

FEATURE_NAMES = ["mean_luminance", "spatial_variance", "motion_energy", "temporal_gradient"]
ROI_NAMES = list(ROI_DEFS.keys()) + ["background_combined"]

# Output dirs
CHAR_FIG_DIR = os.path.join(VIDEO_SYNC_FIG_DIR, "characterize")
CHAR_CACHE_DIR = os.path.join(VIDEO_SYNC_DIR, "characterize")


def _find_video_and_metadata(session_name: str) -> tuple[str, str]:
    """Find eye camera video and metadata for a session.

    Checks local data/videos/ first (flat MP4 files), falls back to X: drive.
    Metadata always comes from X: drive (small files, fast to read).
    """
    sn = str(session_name).zfill(8)
    dd, mm, yyyy = sn[:2], sn[2:4], sn[4:]
    yy = yyyy[2:]
    subject = "BG_046"

    # Try local video first
    local_pattern = f"{subject}_{dd}{mm}{yy}_Eye_cam.mp4"
    local_path = os.path.join(LOCAL_VIDEO_DIR, local_pattern)
    if os.path.isfile(local_path):
        video_path = local_path
        logger.info(f"  Using local video: {local_path}")
    else:
        # Fall back to network
        cam_files = find_camera_files(session_name)
        if "eye_cam" not in cam_files:
            raise FileNotFoundError(f"No eye camera files for session {session_name}")
        video_path = cam_files["eye_cam"]["video"]
        logger.info(f"  Using network video: {video_path}")

    # Metadata always from X: drive (small file)
    cam_files = find_camera_files(session_name)
    if "eye_cam" not in cam_files:
        raise FileNotFoundError(f"No eye camera metadata for session {session_name}")
    meta_path = cam_files["eye_cam"]["metadata"]

    return video_path, meta_path


# ── Feature extraction ───────────────────────────────────────────


def _extract_roi(gray: np.ndarray, roi_name: str) -> np.ndarray:
    """Extract pixel values for a named ROI from a (possibly downsampled) frame."""
    ds = SPATIAL_DOWNSAMPLE
    if roi_name == "background_combined":
        parts = []
        for sub in _BG_COMBINED_ROIS:
            y0, y1, x0, x1 = ROI_DEFS[sub]
            parts.append(gray[y0 // ds : y1 // ds, x0 // ds : x1 // ds].ravel())
        return np.concatenate(parts)
    y0, y1, x0, x1 = ROI_DEFS[roi_name]
    return gray[y0 // ds : y1 // ds, x0 // ds : x1 // ds].ravel()


SPATIAL_DOWNSAMPLE = 2  # Downsample frames by this factor before feature extraction

# Coarse offset cache to avoid redundant computation
_COARSE_OFFSET_CACHE_FILE = os.path.join(VIDEO_SYNC_DIR, "coarse_offsets.json")


def _fast_coarse_offset(
    video_path: str,
    metadata_path: str,
    nidaq_baseline_on_s: np.ndarray,
    target_fps: float = 2.0,
) -> float:
    """Estimate rough offset via ffmpeg continuous low-fps luminance extraction.

    Uses ffmpeg to decode the video at a low frame rate (default 2fps),
    producing a continuous luminance trace. This is efficient because ffmpeg
    reads the video once start-to-finish. Detects transitions and cross-
    correlates with Baseline_ON events.
    """
    import subprocess

    ts_ms, _, _ = load_camera_metadata(metadata_path)
    total_duration_s = ts_ms[-1] / 1000.0

    # Extract low-fps, low-res grayscale stream
    out_w, out_h = 64, 64  # tiny frames, we only need mean luminance
    frame_bytes = out_w * out_h
    expected_frames = int(total_duration_s * target_fps) + 10

    logger.info(
        f"  Extracting ~{expected_frames} frames at {target_fps}fps via ffmpeg "
        f"(duration={total_duration_s:.0f}s)..."
    )

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps={target_fps},scale={out_w}:{out_h}",
        "-pix_fmt", "gray",
        "-f", "rawvideo",
        "pipe:1",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=int(total_duration_s * 0.5 + 120)
        )
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg timed out during coarse scan")
        return 0.0

    if result.returncode != 0:
        logger.warning(f"ffmpeg failed: {result.stderr[:200]}")
        return 0.0

    raw = result.stdout
    n_frames = len(raw) // frame_bytes
    if n_frames < 20:
        logger.warning(f"Only {n_frames} frames extracted")
        return 0.0

    # Compute mean luminance per frame
    pixels = np.frombuffer(raw[: n_frames * frame_bytes], dtype=np.uint8)
    pixels = pixels.reshape(n_frames, out_h, out_w)
    lum = pixels.astype(np.float32).mean(axis=(1, 2))
    sample_times_s = np.arange(n_frames) / target_fps

    logger.info(f"  Got {n_frames} frames ({n_frames / target_fps:.0f}s covered)")

    # Detect transitions via large luminance derivatives
    deriv = np.abs(np.diff(lum))
    med_d = np.median(deriv)
    mad_d = np.median(np.abs(deriv - med_d))
    if mad_d < 1e-6:
        thresh = np.percentile(deriv, 97)
    else:
        thresh = med_d + 5.0 * 1.4826 * mad_d

    peaks = np.where(deriv > thresh)[0]
    if len(peaks) == 0:
        logger.warning("Coarse scan: no transitions detected")
        return 0.0

    # Cluster nearby transitions (within 2s)
    peak_times = sample_times_s[peaks]
    transition_times = [peak_times[0]]
    for t in peak_times[1:]:
        if t - transition_times[-1] > 2.0:
            transition_times.append(t)
    transition_times = np.array(transition_times)

    logger.info(f"  {len(transition_times)} transitions detected")
    return coarse_align(transition_times, nidaq_baseline_on_s)


def _load_or_compute_coarse_offset(
    session_name: str,
    video_path: str,
    metadata_path: str,
    nidaq_baseline_on_s: np.ndarray,
) -> float:
    """Load cached coarse offset or compute it."""
    # Check cache
    if os.path.exists(_COARSE_OFFSET_CACHE_FILE):
        with open(_COARSE_OFFSET_CACHE_FILE) as f:
            cache = json.load(f)
        if session_name in cache:
            offset = cache[session_name]
            logger.info(f"[{session_name}] Using cached coarse offset = {offset:.2f}s")
            return offset

    # Compute
    offset = _fast_coarse_offset(video_path, metadata_path, nidaq_baseline_on_s)

    # Save to cache
    cache = {}
    if os.path.exists(_COARSE_OFFSET_CACHE_FILE):
        with open(_COARSE_OFFSET_CACHE_FILE) as f:
            cache = json.load(f)
    cache[session_name] = offset
    os.makedirs(os.path.dirname(_COARSE_OFFSET_CACHE_FILE), exist_ok=True)
    with open(_COARSE_OFFSET_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

    return offset


def _compute_frame_features(
    gray: np.ndarray, prev_gray: np.ndarray | None, prev_lum: np.ndarray | None
) -> np.ndarray:
    """Compute all features for a single frame across all ROIs.

    Returns (n_features * n_rois,) array.
    """
    n_rois = len(ROI_NAMES)
    n_feats = len(FEATURE_NAMES)
    out = np.full(n_feats * n_rois, np.nan, dtype=np.float32)

    for j, roi_name in enumerate(ROI_NAMES):
        pixels = _extract_roi(gray, roi_name)

        # Feature 0: mean luminance
        lum = np.mean(pixels)
        out[0 * n_rois + j] = lum
        # Feature 1: spatial variance
        out[1 * n_rois + j] = np.var(pixels)

        # Feature 2: motion energy (frame diff)
        if prev_gray is not None:
            prev_pixels = _extract_roi(prev_gray, roi_name)
            out[2 * n_rois + j] = np.mean(np.abs(pixels - prev_pixels))

        # Feature 3: temporal gradient (luminance derivative)
        if prev_lum is not None:
            out[3 * n_rois + j] = lum - prev_lum[j]

    return out


def extract_trial_features_sequential(
    cap, trial_frame_ranges: list[tuple[int, int]], ts_ms: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract features for multiple trials by seeking once per window.

    For each trial window, seeks to the start frame and then reads
    sequentially (no per-frame seeking). This is critical for H.264
    performance, especially over network drives.
    """
    import cv2

    n_rois = len(ROI_NAMES)
    n_feats = len(FEATURE_NAMES)

    # Sort trials by start frame for forward-only access
    order = sorted(range(len(trial_frame_ranges)), key=lambda k: trial_frame_ranges[k][0])

    all_features = [None] * len(trial_frame_ranges)
    all_timestamps = [None] * len(trial_frame_ranges)
    n_total = len(ts_ms)

    for count, idx in enumerate(order):
        f_start, f_end = trial_frame_ranges[idx]
        f_start = max(0, f_start)
        f_end = min(f_end, n_total - 1)
        n_frames = f_end - f_start + 1
        if n_frames <= 0:
            continue

        features = np.full((n_frames, n_feats * n_rois), np.nan, dtype=np.float32)

        # Seek to one frame before the window for motion energy
        seek_to = max(0, f_start - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)

        prev_gray = None
        if seek_to < f_start:
            ret, frame = cap.read()
            if ret:
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                if SPATIAL_DOWNSAMPLE > 1:
                    prev_gray = prev_gray[::SPATIAL_DOWNSAMPLE, ::SPATIAL_DOWNSAMPLE]

        # Read window frames sequentially (no per-frame seeking)
        prev_lum = None
        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if SPATIAL_DOWNSAMPLE > 1:
                gray = gray[::SPATIAL_DOWNSAMPLE, ::SPATIAL_DOWNSAMPLE]
            features[i] = _compute_frame_features(gray, prev_gray, prev_lum)

            # Store luminance values for next frame's gradient computation
            prev_lum = features[i, 0 * n_rois : 1 * n_rois].copy()
            prev_gray = gray

        all_features[idx] = features
        all_timestamps[idx] = ts_ms[f_start : f_end + 1]

        if (count + 1) % 10 == 0:
            logger.info(f"  ... {count + 1}/{len(order)} trials extracted")

    return all_features, all_timestamps


# ── Quality metrics ──────────────────────────────────────────────


def compute_feature_quality(
    features_list: list[np.ndarray],
    timestamps_list: list[np.ndarray],
    onset_times_ms: np.ndarray,
) -> pd.DataFrame:
    """Compute SNR, detection rate, jitter for each feature×ROI channel.

    Parameters
    ----------
    features_list : list of (n_frames, n_channels) arrays, one per trial
    timestamps_list : list of (n_frames,) arrays (camera ms), one per trial
    onset_times_ms : (n_trials,) array of expected onset times in camera ms

    Returns
    -------
    DataFrame with columns: feature, roi, snr, detection_rate, jitter_ms,
        consistency, mean_pre, mean_post, std_pre
    """
    n_rois = len(ROI_NAMES)
    n_feats = len(FEATURE_NAMES)
    n_channels = n_feats * n_rois
    n_trials = len(features_list)

    rows = []

    for ch_idx in range(n_channels):
        feat_idx = ch_idx // n_rois
        roi_idx = ch_idx % n_rois
        feat_name = FEATURE_NAMES[feat_idx]
        roi_name = ROI_NAMES[roi_idx]

        pre_vals = []
        post_vals = []
        detected_offsets_ms = []

        for t in range(n_trials):
            feats = features_list[t]
            ts = timestamps_list[t]
            if feats is None or ts is None:
                continue

            onset_ms = onset_times_ms[t]
            rel_s = (ts - onset_ms) / 1000.0  # relative time in seconds

            # Pre window
            pre_mask = (rel_s >= PRE_WINDOW[0]) & (rel_s < PRE_WINDOW[1])
            post_mask = (rel_s >= POST_WINDOW[0]) & (rel_s < POST_WINDOW[1])

            channel_vals = feats[:, ch_idx]

            if np.sum(pre_mask) > 2:
                pre_vals.append(np.nanmean(channel_vals[pre_mask]))
            if np.sum(post_mask) > 2:
                post_vals.append(np.nanmean(channel_vals[post_mask]))

            # Detection: find threshold crossing in derivative near onset
            search_mask = (rel_s >= -DETECT_SEARCH_S) & (rel_s <= DETECT_SEARCH_S)
            search_vals = channel_vals[search_mask]
            search_ts = rel_s[search_mask]
            if len(search_vals) > 3:
                deriv = np.abs(np.diff(search_vals))
                med_d = np.median(deriv)
                mad_d = np.median(np.abs(deriv - med_d))
                if mad_d > 1e-6:
                    thresh = med_d + DETECT_MAD_MULT * 1.4826 * mad_d
                    peaks = np.where(deriv > thresh)[0]
                    if len(peaks) > 0:
                        # Take first crossing
                        detected_offsets_ms.append(search_ts[peaks[0]] * 1000.0)

        pre_arr = np.array(pre_vals)
        post_arr = np.array(post_vals)

        mean_pre = np.nanmean(pre_arr) if len(pre_arr) > 0 else np.nan
        std_pre = np.nanstd(pre_arr) if len(pre_arr) > 0 else np.nan
        mean_post = np.nanmean(post_arr) if len(post_arr) > 0 else np.nan

        # SNR
        if std_pre > 1e-6:
            snr = (mean_post - mean_pre) / std_pre
        else:
            snr = 0.0

        # Detection rate
        detection_rate = len(detected_offsets_ms) / max(n_trials, 1)

        # Timing jitter
        if len(detected_offsets_ms) >= 3:
            jitter_ms = np.std(detected_offsets_ms)
        else:
            jitter_ms = np.nan

        # Split-half consistency
        consistency = np.nan
        if n_trials >= 10:
            half = n_trials // 2
            traces_first = []
            traces_second = []
            for t in range(n_trials):
                feats = features_list[t]
                if feats is None:
                    continue
                trace = feats[:, ch_idx]
                if t < half:
                    traces_first.append(trace)
                else:
                    traces_second.append(trace)
            if traces_first and traces_second:
                min_len = min(
                    min(len(tr) for tr in traces_first),
                    min(len(tr) for tr in traces_second),
                )
                if min_len > 5:
                    avg1 = np.nanmean(
                        np.array([tr[:min_len] for tr in traces_first]), axis=0
                    )
                    avg2 = np.nanmean(
                        np.array([tr[:min_len] for tr in traces_second]), axis=0
                    )
                    valid = ~(np.isnan(avg1) | np.isnan(avg2))
                    if np.sum(valid) > 5:
                        consistency = np.corrcoef(avg1[valid], avg2[valid])[0, 1]

        rows.append(
            {
                "feature": feat_name,
                "roi": roi_name,
                "snr": snr,
                "detection_rate": detection_rate,
                "jitter_ms": jitter_ms,
                "consistency": consistency,
                "mean_pre": mean_pre,
                "mean_post": mean_post,
                "std_pre": std_pre,
            }
        )

    return pd.DataFrame(rows)


# ── Characterize one session ─────────────────────────────────────


def characterize_session(
    session_name: str,
    n_trials: int = DEFAULT_N_TRIALS,
    force: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run full feature characterization for one session.

    Returns (quality_df, metadata_dict).
    """
    import cv2

    cache_path = os.path.join(CHAR_CACHE_DIR, f"{session_name}_features.npz")
    if os.path.exists(cache_path) and not force:
        logger.info(f"[{session_name}] Loading cached features from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        features_list = list(data["features_list"])
        timestamps_list = list(data["timestamps_list"])
        onset_times_ms = data["onset_times_ms"]
        coarse_offset = float(data["coarse_offset"])
        frame_shape = tuple(data["frame_shape"])
        sample_frames = data.get("sample_frames", None)
    else:
        t0 = time.time()
        # Load session
        manifest = load_staging_manifest(qc_only=True, apply_filter=False)
        stage = "unknown"
        pkl_path = None
        for _, row in manifest.iterrows():
            if str(row["session_name"]) == str(session_name):
                stage = row.get("stage", "unknown")
                pkl_path = os.path.join(_PROJECT_ROOT, row["path"])
                break

        if pkl_path is None:
            # Try direct path
            pkl_path = os.path.join(
                _PROJECT_ROOT, "data", "pkls", "BG_046", f"BG_046_{session_name}.pkl"
            )

        logger.info(f"[{session_name}] Loading session from {pkl_path}")
        sess = load_session(pkl_path)
        baseline_on = np.asarray(
            sess.ni_events.get("Baseline_ON", []), dtype=float
        ).flatten()
        n_total_trials = len(baseline_on)
        logger.info(f"[{session_name}] {n_total_trials} Baseline_ON events")

        # Find camera files (local first, then network)
        video_path, meta_path = _find_video_and_metadata(session_name)

        # Load metadata
        ts_ms, _, _ = load_camera_metadata(meta_path)

        # Coarse alignment (fast — sequential reads with caching)
        logger.info(f"[{session_name}] Computing coarse offset...")
        coarse_offset = _load_or_compute_coarse_offset(
            session_name, video_path, meta_path, baseline_on
        )
        logger.info(f"[{session_name}] Coarse offset = {coarse_offset:.2f}s")

        # Select trials evenly spaced
        if n_trials >= n_total_trials:
            trial_indices = np.arange(n_total_trials)
        else:
            trial_indices = np.linspace(0, n_total_trials - 1, n_trials, dtype=int)

        # Convert Baseline_ON times to camera ms
        # camera_ms = (nidaq_s - offset) * 1000  (inverse of t_nidaq = t_cam/1000 + offset)
        onset_nidaq_s = baseline_on[trial_indices]
        onset_cam_ms = (onset_nidaq_s - coarse_offset) * 1000.0

        # Find frame ranges for each trial
        trial_frame_ranges = []
        for onset_ms in onset_cam_ms:
            f_center = np.searchsorted(ts_ms, onset_ms)
            f_start = f_center - int(WINDOW_S * FPS_NOMINAL)
            f_end = f_center + int(WINDOW_S * FPS_NOMINAL)
            trial_frame_ranges.append((f_start, f_end))

        # Open video and get frame shape
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_shape = (frame_h, frame_w)
            logger.info(f"[{session_name}] Frame size: {frame_w}x{frame_h}")

            # Extract sample frames (pre, onset, post) from first valid trial
            sample_frames = _extract_sample_frames(
                cap, ts_ms, onset_cam_ms[0], frame_shape
            )

            # Extract features for all trials
            logger.info(
                f"[{session_name}] Extracting features for "
                f"{len(trial_indices)} trials..."
            )
            features_list, timestamps_list = extract_trial_features_sequential(
                cap, trial_frame_ranges, ts_ms
            )
        finally:
            cap.release()

        elapsed = time.time() - t0
        logger.info(f"[{session_name}] Feature extraction done in {elapsed:.1f}s")

        # Free session memory
        del sess
        gc.collect()

        # Cache
        os.makedirs(CHAR_CACHE_DIR, exist_ok=True)
        np.savez_compressed(
            cache_path,
            features_list=np.array(features_list, dtype=object),
            timestamps_list=np.array(timestamps_list, dtype=object),
            onset_times_ms=onset_cam_ms,
            coarse_offset=coarse_offset,
            frame_shape=np.array(frame_shape),
            sample_frames=sample_frames,
        )
        onset_times_ms = onset_cam_ms

    # Compute quality metrics
    quality_df = compute_feature_quality(features_list, timestamps_list, onset_times_ms)
    quality_df["session"] = session_name

    # Look up stage
    try:
        manifest = load_staging_manifest(qc_only=True, apply_filter=False)
        stage_row = manifest[manifest["session_name"].astype(str) == str(session_name)]
        stage = stage_row.iloc[0]["stage"] if len(stage_row) > 0 else "unknown"
    except Exception:
        stage = "unknown"
    quality_df["stage"] = stage

    metadata = {
        "session_name": session_name,
        "stage": stage,
        "coarse_offset": coarse_offset,
        "frame_shape": frame_shape if isinstance(frame_shape, tuple) else tuple(frame_shape),
        "n_trials_sampled": len(features_list),
        "features_list": features_list,
        "timestamps_list": timestamps_list,
        "onset_times_ms": onset_times_ms,
        "sample_frames": sample_frames,
    }
    return quality_df, metadata


def _extract_sample_frames(cap, ts_ms, onset_cam_ms, frame_shape):
    """Extract 3 sample frames: pre-onset (-1s), onset (0s), post-onset (+1s)."""
    import cv2

    offsets_ms = [-1000, 0, 1000]
    sample_frames = []
    for off in offsets_ms:
        target_ms = onset_cam_ms + off
        fidx = np.searchsorted(ts_ms, target_ms)
        fidx = np.clip(fidx, 0, len(ts_ms) - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sample_frames.append(gray)
        else:
            sample_frames.append(np.zeros(frame_shape, dtype=np.uint8))
    return np.array(sample_frames)


# ── Plotting ─────────────────────────────────────────────────────


def plot_session_diagnostic(
    quality_df: pd.DataFrame, metadata: dict, save_dir: str | None = None
):
    """Generate per-session multi-panel diagnostic figure."""
    session_name = metadata["session_name"]
    stage = metadata["stage"]
    features_list = metadata["features_list"]
    timestamps_list = metadata["timestamps_list"]
    onset_times_ms = metadata["onset_times_ms"]
    sample_frames = metadata.get("sample_frames", None)
    n_rois = len(ROI_NAMES)

    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(
        f"Camera Signal Characterization — {session_name} ({stage})",
        fontsize=14,
        fontweight="bold",
    )
    gs = gridspec.GridSpec(4, 4, hspace=0.4, wspace=0.35, top=0.93)

    # ── Row 1: Trial-averaged traces for top 4 feature×ROI combos ─
    top4 = quality_df.reindex(quality_df["snr"].abs().nlargest(4).index)
    for panel_idx, (_, row) in enumerate(top4.iterrows()):
        ax = fig.add_subplot(gs[0, panel_idx])
        feat_name = row["feature"]
        roi_name = row["roi"]
        feat_idx = FEATURE_NAMES.index(feat_name)
        roi_idx = ROI_NAMES.index(roi_name)
        ch_idx = feat_idx * n_rois + roi_idx

        # Collect aligned traces
        traces = []
        for t in range(len(features_list)):
            feats = features_list[t]
            ts = timestamps_list[t]
            if feats is None or ts is None:
                continue
            rel_s = (ts - onset_times_ms[t]) / 1000.0
            traces.append((rel_s, feats[:, ch_idx]))

        if traces:
            # Interpolate to common time grid
            t_grid = np.linspace(-WINDOW_S, WINDOW_S, 200)
            interp_traces = []
            for rel_s, vals in traces:
                valid = ~np.isnan(vals)
                if np.sum(valid) > 10:
                    interp = np.interp(t_grid, rel_s[valid], vals[valid])
                    interp_traces.append(interp)

            if interp_traces:
                arr = np.array(interp_traces)
                mean_trace = np.nanmean(arr, axis=0)
                sem_trace = np.nanstd(arr, axis=0) / np.sqrt(len(interp_traces))

                ax.plot(t_grid, mean_trace, "k-", linewidth=1.5)
                ax.fill_between(
                    t_grid,
                    mean_trace - sem_trace,
                    mean_trace + sem_trace,
                    alpha=0.3,
                    color="steelblue",
                )

        ax.axvline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_title(f"{feat_name}\n{roi_name}\nSNR={row['snr']:.2f}", fontsize=8)
        ax.set_xlabel("Time from Baseline_ON (s)", fontsize=7)
        if panel_idx == 0:
            ax.set_ylabel("Feature value", fontsize=7)
        ax.tick_params(labelsize=6)

    # ── Row 2: Single-trial heatmap for best feature ─
    if len(top4) > 0:
        best = quality_df.loc[quality_df["snr"].abs().idxmax()]
        feat_idx = FEATURE_NAMES.index(best["feature"])
        roi_idx = ROI_NAMES.index(best["roi"])
        ch_idx = feat_idx * n_rois + roi_idx

        t_grid = np.linspace(-WINDOW_S, WINDOW_S, 200)
        heatmap_data = []
        for t in range(len(features_list)):
            feats = features_list[t]
            ts = timestamps_list[t]
            if feats is None or ts is None:
                continue
            rel_s = (ts - onset_times_ms[t]) / 1000.0
            vals = feats[:, ch_idx]
            valid = ~np.isnan(vals)
            if np.sum(valid) > 10:
                interp = np.interp(t_grid, rel_s[valid], vals[valid])
                heatmap_data.append(interp)

        if heatmap_data:
            ax = fig.add_subplot(gs[1, :3])
            hm = np.array(heatmap_data)
            # Z-score per trial for visualization
            trial_means = np.nanmean(hm[:, :50], axis=1, keepdims=True)
            trial_stds = np.nanstd(hm[:, :50], axis=1, keepdims=True)
            trial_stds[trial_stds < 1e-6] = 1.0
            hm_z = (hm - trial_means) / trial_stds

            im = ax.imshow(
                hm_z,
                aspect="auto",
                extent=[t_grid[0], t_grid[-1], len(heatmap_data), 0],
                cmap="RdBu_r",
                vmin=-3,
                vmax=3,
            )
            ax.axvline(0, color="red", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Time from Baseline_ON (s)", fontsize=8)
            ax.set_ylabel("Trial", fontsize=8)
            ax.set_title(
                f"Best: {best['feature']} / {best['roi']} (z-scored per trial)",
                fontsize=9,
            )
            plt.colorbar(im, ax=ax, shrink=0.6, label="z-score")

    # ── Row 3: SNR bar chart across all channels ─
    ax = fig.add_subplot(gs[2, :])
    snr_vals = quality_df["snr"].values
    labels = [
        f"{r['feature'][:4]}_{r['roi'][:4]}" for _, r in quality_df.iterrows()
    ]
    colors = []
    feat_colors = {"mean": "#1f77b4", "spat": "#ff7f0e", "moti": "#2ca02c", "temp": "#d62728"}
    for _, r in quality_df.iterrows():
        colors.append(feat_colors.get(r["feature"][:4], "gray"))
    ax.bar(range(len(snr_vals)), snr_vals, color=colors, alpha=0.8, width=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_ylabel("SNR", fontsize=8)
    ax.set_title("SNR by Feature × ROI", fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.5)

    # ── Row 4: Sample frames with ROI overlays ─
    if sample_frames is not None and len(sample_frames) >= 3:
        frame_labels = ["-1.0s (pre)", "0.0s (onset)", "+1.0s (post)"]
        for fi in range(3):
            ax = fig.add_subplot(gs[3, fi])
            frame = sample_frames[fi] if isinstance(sample_frames, np.ndarray) else sample_frames
            if isinstance(frame, np.ndarray) and frame.ndim >= 2:
                if frame.ndim == 3 and fi < frame.shape[0]:
                    frame = frame[fi]
                elif frame.ndim == 3:
                    frame = frame[0]
                ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
                # Overlay ROI rectangles
                roi_colors_plot = {
                    "above_head": "cyan",
                    "below_head": "lime",
                    "top_right": "magenta",
                    "top_left": "yellow",
                    "mouse_head": "red",
                }
                for rname, rcolor in roi_colors_plot.items():
                    y0, y1, x0, x1 = ROI_DEFS[rname]
                    rect = plt.Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        linewidth=1,
                        edgecolor=rcolor,
                        facecolor="none",
                        linestyle="--",
                    )
                    ax.add_patch(rect)
            ax.set_title(frame_labels[fi], fontsize=8)
            ax.axis("off")

        # Legend in 4th column
        ax_leg = fig.add_subplot(gs[3, 3])
        ax_leg.axis("off")
        roi_colors_plot = {
            "above_head": "cyan",
            "below_head": "lime",
            "top_right": "magenta",
            "top_left": "yellow",
            "mouse_head": "red",
        }
        for i, (rname, rcolor) in enumerate(roi_colors_plot.items()):
            ax_leg.plot([], [], color=rcolor, linewidth=2, label=rname)
        ax_leg.legend(loc="center", fontsize=8, frameon=False)
        ax_leg.set_title("ROI Legend", fontsize=9)

    # Save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"characterize_{session_name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {out_path}")
    plt.close(fig)


def plot_cross_session(all_quality: pd.DataFrame, save_dir: str | None = None):
    """Generate cross-session comparison figure."""
    from visdetect.analysis.config import STAGE_COLORS

    sessions = all_quality["session"].unique()
    n_sessions = len(sessions)

    if n_sessions < 2:
        logger.info("Skipping cross-session plot (only 1 session)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Cross-Session Feature Quality Comparison", fontsize=14, fontweight="bold"
    )

    # Get best feature per session (by SNR)
    best_per_session = (
        all_quality.groupby("session")
        .apply(lambda g: g.loc[g["snr"].abs().idxmax()])
        .reset_index(drop=True)
    )

    stage_color_map = {
        "Naive": STAGE_COLORS.get("Naive", "#999999"),
        "Learning": STAGE_COLORS.get("Learning", "#4daf4a"),
        "Expert": STAGE_COLORS.get("Expert", "#e41a1c"),
    }

    # Panel A: Best SNR per session
    ax = axes[0, 0]
    colors = [stage_color_map.get(s, "gray") for s in best_per_session["stage"]]
    ax.bar(range(n_sessions), best_per_session["snr"].values, color=colors, alpha=0.8)
    ax.set_xticks(range(n_sessions))
    ax.set_xticklabels(best_per_session["session"].values, rotation=45, fontsize=7)
    ax.set_ylabel("Best SNR")
    ax.set_title("A) Best Feature SNR per Session")

    # Panel B: Detection rate for best feature
    ax = axes[0, 1]
    ax.bar(
        range(n_sessions),
        best_per_session["detection_rate"].values,
        color=colors,
        alpha=0.8,
    )
    ax.set_xticks(range(n_sessions))
    ax.set_xticklabels(best_per_session["session"].values, rotation=45, fontsize=7)
    ax.set_ylabel("Detection Rate")
    ax.set_title("B) Detection Rate (Best Feature)")
    ax.set_ylim([0, 1])

    # Panel C: Timing jitter for best feature
    ax = axes[1, 0]
    jitter_vals = best_per_session["jitter_ms"].values
    valid_jitter = ~np.isnan(jitter_vals)
    ax.bar(
        np.arange(n_sessions)[valid_jitter],
        jitter_vals[valid_jitter],
        color=[c for c, v in zip(colors, valid_jitter) if v],
        alpha=0.8,
    )
    ax.set_xticks(range(n_sessions))
    ax.set_xticklabels(best_per_session["session"].values, rotation=45, fontsize=7)
    ax.set_ylabel("Jitter (ms)")
    ax.set_title("C) Timing Jitter (Best Feature)")

    # Panel D: Feature × ROI heatmap (sessions × features, top 10)
    ax = axes[1, 1]
    # Find globally best 10 feature×ROI combos
    global_ranking = (
        all_quality.groupby(["feature", "roi"])["snr"]
        .apply(lambda x: np.abs(x).mean())
        .nlargest(10)
        .index.tolist()
    )
    hm_data = np.full((n_sessions, len(global_ranking)), np.nan)
    for i, sess in enumerate(sessions):
        sess_df = all_quality[all_quality["session"] == sess]
        for j, (feat, roi) in enumerate(global_ranking):
            match = sess_df[(sess_df["feature"] == feat) & (sess_df["roi"] == roi)]
            if len(match) > 0:
                hm_data[i, j] = match.iloc[0]["snr"]

    im = ax.imshow(hm_data, aspect="auto", cmap="RdBu_r")
    ax.set_yticks(range(n_sessions))
    ax.set_yticklabels(
        [f"{s} ({st})" for s, st in zip(sessions, best_per_session["stage"])],
        fontsize=7,
    )
    ax.set_xticks(range(len(global_ranking)))
    ax.set_xticklabels(
        [f"{f[:4]}_{r[:4]}" for f, r in global_ranking], rotation=90, fontsize=6
    )
    ax.set_title("D) Top 10 Feature×ROI (SNR)")
    plt.colorbar(im, ax=ax, shrink=0.6, label="SNR")

    # Stage legend
    for stage, color in stage_color_map.items():
        fig.patches.append(
            plt.Rectangle((0, 0), 0, 0, facecolor=color, label=stage)
        )
    fig.legend(
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "cross_session_comparison.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {out_path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Characterize camera signal for video sync"
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=DEFAULT_SESSIONS,
        help="Session names to process (default: 5 spanning stages)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help=f"Trials to sample per session (default: {DEFAULT_N_TRIALS})",
    )
    parser.add_argument("--force", action="store_true", help="Force recompute (ignore cache)")
    args = parser.parse_args()

    logger.info(f"Sessions: {args.sessions}")
    logger.info(f"Trials per session: {args.n_trials}")

    all_quality = []
    all_metadata = []

    for session_name in args.sessions:
        logger.info(f"\n{'='*60}\nProcessing session {session_name}\n{'='*60}")
        try:
            quality_df, metadata = characterize_session(
                session_name, n_trials=args.n_trials, force=args.force
            )
            all_quality.append(quality_df)
            all_metadata.append(metadata)

            # Per-session diagnostic
            plot_session_diagnostic(quality_df, metadata, save_dir=CHAR_FIG_DIR)

        except (FileNotFoundError, RuntimeError) as e:
            logger.error(f"[{session_name}] Failed: {e}")
            continue

        gc.collect()

    if not all_quality:
        logger.error("No sessions processed successfully")
        return

    # Combine and save CSV
    combined_df = pd.concat(all_quality, ignore_index=True)
    csv_path = os.path.join(VIDEO_SYNC_DIR, "feature_characterization.csv")
    combined_df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved feature summary: {csv_path}")

    # Cross-session comparison
    if len(all_quality) > 1:
        plot_cross_session(combined_df, save_dir=CHAR_FIG_DIR)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY — Top 5 Feature×ROI by mean |SNR| across sessions:")
    logger.info("=" * 60)
    top5 = (
        combined_df.groupby(["feature", "roi"])
        .agg(
            mean_snr=("snr", lambda x: np.abs(x).mean()),
            mean_detect=("detection_rate", "mean"),
            mean_jitter=("jitter_ms", "mean"),
            mean_consistency=("consistency", "mean"),
        )
        .nlargest(5, "mean_snr")
    )
    print(top5.to_string())


if __name__ == "__main__":
    main()
