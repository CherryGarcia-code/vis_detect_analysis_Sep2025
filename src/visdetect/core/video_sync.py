"""Video-to-NI-DAQ temporal synchronization via luminance-based multi-anchor fitting.

Camera timestamps (USB clock) drift relative to the NI-DAQ master clock by
~1750 ppm (~16 s over a 2.6 h session).  This module detects stimulus
transitions (gray -> grating) in the per-frame luminance trace, matches them to
``Baseline_ON`` events from the NI-DAQ, and fits a linear clock model::

    t_nidaq = slope * (t_camera_ms / 1000) + offset

Detection method
----------------
The eye camera is IR-illuminated; visible-light stimulus changes are faint on
the mouse itself.  The strongest signal is the **background screen glow**
visible in the upper-right region of the frame (behind/above the mouse head).

When the gray screen transitions to a drifting grating (Baseline_ON), the
luminance in this ROI goes from flat to oscillating (TF modulation at ~1 Hz).
We detect the **first large absolute derivative** as the onset timestamp, with
sub-frame linear interpolation on the derivative threshold crossing.

Precision: ~17 ms RMSE with 50 fps eye camera (limited by frame-rate
quantization at ~6 ms and 50 ms TF update binning at ~14 ms, combined in
quadrature).  Front cam sync is derived from the eye cam clock model.

Functions
---------
detect_onsets_derivative  Per-trial derivative onset detection (primary method)
extract_luminance         Full-session luminance extraction (for diagnostics)
coarse_align              Brute-force offset search for initial alignment
fit_clock_model           Theil-Sen robust regression + outlier rejection + CV
camera_to_nidaq           Convert camera timestamp(s) to NI-DAQ seconds
nidaq_to_camera           Inverse conversion
load_camera_metadata      Load camera metadata CSV
find_camera_files         Locate video + metadata files for a session
camera_dir_to_session     Map ``DDMMYY`` camera dirname -> ``DDMMYYYY`` session name
save_video_sync           Save sync parameters to JSON sidecar
load_video_sync           Load sync parameters from JSON sidecar
sync_session              End-to-end pipeline for one session
plot_sync_diagnostic      4-panel diagnostic figure
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# MAD-to-sigma conversion factor: 1 / Phi^{-1}(3/4)
_MAD_SCALE = 1.4826

# Minimum anchor count for fitting / CV
_MIN_ANCHORS_FIT = 10
_MIN_ANCHORS_CV = 20

# ---------------------------------------------------------------------------
# Constants (imported from canonical source)
# ---------------------------------------------------------------------------
from visdetect.analysis.constants import (
    VIDEO_SYNC_COARSE_SEARCH_S,
    VIDEO_SYNC_COARSE_STEP_S,
    VIDEO_SYNC_DEFAULT_EYE_ROI,
    VIDEO_SYNC_DERIV_MIN_THRESH,
    VIDEO_SYNC_DERIV_PRE_FRAMES,
    VIDEO_SYNC_DERIV_SEARCH_FRAMES,
    VIDEO_SYNC_DERIV_SIGMA_MULT,
    VIDEO_SYNC_MAX_DRIFT_PPM,
    VIDEO_SYNC_MAX_RESIDUAL_MS,
    VIDEO_SYNC_MIN_COVERAGE,
    VIDEO_SYNC_OUTLIER_N_ITER,
    VIDEO_SYNC_OUTLIER_SIGMA,
)

from visdetect.analysis.config import CAMERA_ROOT, VIDEO_SYNC_DIR

# Quality-tier thresholds for SyncResult.quality
_GOOD_RMSE_MS = 20.0
_GOOD_DW_RANGE = (1.5, 2.5)
_REVIEW_RMSE_MS = 40.0
_REVIEW_COVERAGE = 0.60


# =====================================================================
# Data classes
# =====================================================================


@dataclass
class SyncResult:
    """Result of a single-camera clock synchronization fit."""

    slope: float
    offset: float
    n_anchors: int
    n_baseline_on: int
    rmse_ms: float
    max_residual_ms: float
    cv_rmse_ms: float
    slope_ppm: float
    durbin_watson: float
    roi: Optional[List[int]] = None
    n_frames: int = 0
    n_dropped: int = 0
    detection_method: str = "derivative"
    inlier_mask: Optional[np.ndarray] = field(default=None, repr=False)
    residuals_ms: Optional[np.ndarray] = field(default=None, repr=False)
    matched_cam_ms: Optional[np.ndarray] = field(default=None, repr=False)
    matched_nidaq_s: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def coverage(self) -> float:
        return self.n_anchors / max(self.n_baseline_on, 1)

    @property
    def quality(self) -> str:
        """Composite quality tier: good / review / failed."""
        if (
            self.rmse_ms < _GOOD_RMSE_MS
            and self.coverage >= VIDEO_SYNC_MIN_COVERAGE
            and self.max_residual_ms < VIDEO_SYNC_MAX_RESIDUAL_MS
            and _GOOD_DW_RANGE[0] <= self.durbin_watson <= _GOOD_DW_RANGE[1]
        ):
            return "good"
        elif (
            self.rmse_ms < _REVIEW_RMSE_MS
            and self.coverage >= _REVIEW_COVERAGE
            and abs(self.slope_ppm) < VIDEO_SYNC_MAX_DRIFT_PPM
        ):
            return "review"
        else:
            return "failed"

    def to_dict(self) -> dict:
        return {
            "slope": self.slope,
            "offset": self.offset,
            "n_anchors": self.n_anchors,
            "n_baseline_on": self.n_baseline_on,
            "coverage": round(self.coverage, 4),
            "rmse_ms": round(self.rmse_ms, 2),
            "max_residual_ms": round(self.max_residual_ms, 2),
            "cv_rmse_ms": round(self.cv_rmse_ms, 2),
            "slope_ppm": round(self.slope_ppm, 2),
            "durbin_watson": round(self.durbin_watson, 4),
            "quality": self.quality,
            "roi": self.roi,
            "n_frames": self.n_frames,
            "n_dropped": self.n_dropped,
            "detection_method": self.detection_method,
        }


# =====================================================================
# Metadata parsing
# =====================================================================


def load_camera_metadata(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load camera metadata CSV.

    Returns (timestamps_ms, acquired, saved) arrays.  The terminal
    zero-row is dropped automatically.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    ts = df.iloc[:, 0].values.astype(np.float64)
    acq = df.iloc[:, 1].values.astype(np.int64)
    saved = df.iloc[:, 2].values.astype(np.int64)

    # Drop terminal zero-row
    if len(ts) > 1 and ts[-1] == 0.0:
        ts, acq, saved = ts[:-1], acq[:-1], saved[:-1]

    return ts, acq, saved


def camera_dir_to_session(dirname: str, subject: str = "BG_046") -> str:
    """Convert camera directory name ``BG_046_DDMMYY`` -> session ``DDMMYYYY``."""
    parts = dirname.split("_")
    date6 = parts[-1]  # e.g. "010725"
    if len(date6) != 6:
        raise ValueError(f"Cannot parse 6-digit date from '{dirname}'")
    dd, mm, yy = date6[:2], date6[2:4], date6[4:6]
    return f"{dd}{mm}20{yy}"


def find_camera_files(
    session_name: str,
    camera_root: Optional[str] = None,
    subject: str = "BG_046",
) -> Dict[str, Dict[str, str]]:
    """Locate video + metadata files for a session.

    Returns dict like::

        {"eye_cam": {"video": "path.mp4", "metadata": "path.csv"},
         "front_cam": {"video": "path.mp4", "metadata": "path.csv"}}

    Note: keys are only present if both video and metadata files are found.
    Callers should check ``"eye_cam" in result`` before accessing.
    """
    root = camera_root or CAMERA_ROOT

    sn = str(session_name).zfill(8)
    dd, mm, yyyy = sn[:2], sn[2:4], sn[4:]
    yy = yyyy[2:]
    cam_dir = os.path.join(root, f"{subject}_{dd}{mm}{yy}")

    if not os.path.isdir(cam_dir):
        raise FileNotFoundError(f"Camera directory not found: {cam_dir}")

    result = {}
    for cam_label, prefix in [("eye_cam", "Eye_cam"), ("front_cam", "Front_cam")]:
        video = None
        meta = None
        for f in os.listdir(cam_dir):
            if prefix in f and f.endswith(".mp4"):
                video = os.path.join(cam_dir, f)
            elif prefix in f and f.endswith("_metadata.csv"):
                meta = os.path.join(cam_dir, f)
        if video and meta:
            result[cam_label] = {"video": video, "metadata": meta}

    return result


# =====================================================================
# Luminance extraction (diagnostic / full-session)
# =====================================================================


def extract_luminance(
    video_path: str,
    metadata_path: str,
    roi: Optional[Tuple[int, int, int, int]] = None,
    spatial_downsample: int = 4,
    progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-frame mean luminance and spatial variance from video.

    This is a **diagnostic utility** for inspecting the full luminance trace.
    The main sync pipeline uses ``detect_onsets_derivative`` instead.

    Returns (timestamps_ms, mean_lum, spatial_var).
    """
    import cv2

    ts_ms, _, _ = load_camera_metadata(metadata_path)
    n_frames = len(ts_ms)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total != n_frames:
            logger.warning(
                f"Frame count mismatch: video={total}, metadata={n_frames}. "
                f"Using min({total}, {n_frames})."
            )
            n_frames = min(total, n_frames)
            ts_ms = ts_ms[:n_frames]

        mean_lum = np.empty(n_frames, dtype=np.float32)
        spatial_var = np.empty(n_frames, dtype=np.float32)

        iterator = range(n_frames)
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Extracting luminance", unit="frame")
            except ImportError:
                pass

        ds = spatial_downsample
        for i in iterator:
            ret, frame = cap.read()
            if not ret:
                mean_lum[i] = mean_lum[i - 1] if i > 0 else 0.0
                spatial_var[i] = spatial_var[i - 1] if i > 0 else 0.0
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if roi is not None:
                y0, y1, x0, x1 = roi
                patch = gray[y0:y1:ds, x0:x1:ds].astype(np.float32)
            else:
                patch = gray[::ds, ::ds].astype(np.float32)

            mean_lum[i] = patch.mean()
            spatial_var[i] = patch.var()
    finally:
        cap.release()

    return ts_ms, mean_lum, spatial_var


# =====================================================================
# Derivative-based onset detection (primary method)
# =====================================================================


def detect_onsets_derivative(
    video_path: str,
    metadata_path: str,
    baseline_on_s: np.ndarray,
    rough_offset_s: float,
    roi: Tuple[int, int, int, int] = VIDEO_SYNC_DEFAULT_EYE_ROI,
    search_frames: int = VIDEO_SYNC_DERIV_SEARCH_FRAMES,
    pre_frames: int = VIDEO_SYNC_DERIV_PRE_FRAMES,
    sigma_mult: float = VIDEO_SYNC_DERIV_SIGMA_MULT,
    min_thresh: float = VIDEO_SYNC_DERIV_MIN_THRESH,
    progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect grating onset per trial via first large luminance derivative.

    For each Baseline_ON event, reads a window of frames around the expected
    camera time, computes the ROI mean luminance, and finds the first frame
    where |d(luminance)/dt| exceeds a threshold derived from the pre-onset
    baseline noise.  Sub-frame linear interpolation is applied at the
    threshold crossing.

    Returns (detected_cam_s, detected_nidaq_s).
    """
    import cv2

    ts_ms, _, _ = load_camera_metadata(metadata_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        y0, y1, x0, x1 = roi
        n_trials = len(baseline_on_s)
        detected_cam_s = []
        detected_nidaq_s = []
        n_miss = 0

        for trial_idx in range(n_trials):
            nidaq_t = baseline_on_s[trial_idx]
            cam_ms_est = (nidaq_t - rough_offset_s) * 1000.0
            center_frame = int(np.searchsorted(ts_ms, cam_ms_est))

            start_f = max(0, center_frame - search_frames)
            end_f = min(len(ts_ms) - 1, center_frame + search_frames)

            if end_f - start_f < pre_frames + 5:
                n_miss += 1
                continue

            # Read frames in the search window
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            lum = []
            ftimes = []
            for fi in range(start_f, end_f):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                lum.append(gray[y0:y1, x0:x1].mean())
                ftimes.append(ts_ms[fi] / 1000.0)

            lum_arr = np.array(lum)
            ftimes_arr = np.array(ftimes)

            if len(lum_arr) < pre_frames + 5:
                n_miss += 1
                continue

            # Absolute derivative
            deriv = np.abs(np.diff(lum_arr))

            # Baseline noise from pre-onset frames
            pre_deriv = deriv[:pre_frames]
            noise_median = np.median(pre_deriv)
            noise_mad = np.median(np.abs(pre_deriv - noise_median))
            thresh = noise_median + max(
                sigma_mult * _MAD_SCALE * noise_mad,
                min_thresh,
            )

            # Find first exceedance after the pre baseline
            exceed_idx = np.where(deriv[pre_frames:] > thresh)[0]
            if len(exceed_idx) == 0:
                n_miss += 1
                continue

            onset_idx = exceed_idx[0] + pre_frames

            # Sub-frame interpolation of the threshold crossing
            if onset_idx > 0:
                d_before = deriv[onset_idx - 1]
                d_after = deriv[onset_idx]
                if d_after > d_before and (d_after - d_before) > 1e-6:
                    frac = np.clip(
                        (thresh - d_before) / (d_after - d_before), 0.0, 1.0
                    )
                    t_interp = (
                        ftimes_arr[onset_idx - 1]
                        + frac * (ftimes_arr[onset_idx] - ftimes_arr[onset_idx - 1])
                    )
                else:
                    t_interp = ftimes_arr[onset_idx]
            else:
                t_interp = ftimes_arr[onset_idx]

            detected_cam_s.append(t_interp)
            detected_nidaq_s.append(nidaq_t)

            if progress and (trial_idx + 1) % 100 == 0:
                logger.info(
                    f"  Onset detection: {trial_idx + 1}/{n_trials} trials processed"
                )
    finally:
        cap.release()

    logger.info(
        f"Onset detection complete: {len(detected_cam_s)}/{n_trials} detected, "
        f"{n_miss} missed"
    )

    return np.array(detected_cam_s), np.array(detected_nidaq_s)


# =====================================================================
# Coarse alignment (brute-force offset search)
# =====================================================================


def coarse_align(
    cam_transitions_s: np.ndarray,
    nidaq_events_s: np.ndarray,
    search_range_s: float = VIDEO_SYNC_COARSE_SEARCH_S,
    step_s: float = VIDEO_SYNC_COARSE_STEP_S,
    match_tolerance_s: float = 1.0,
) -> float:
    """Find the approximate time offset between camera and NI-DAQ clocks.

    Brute-force: for each candidate offset, count nearest-neighbor matches.
    Returns the offset (seconds) to add to camera times.
    """
    offsets = np.arange(-search_range_s, search_range_s + step_s, step_s)
    best_offset = 0.0
    best_count = 0

    for off in offsets:
        shifted = cam_transitions_s + off
        idx = np.searchsorted(nidaq_events_s, shifted)
        idx = np.clip(idx, 0, len(nidaq_events_s) - 1)
        dists = np.abs(shifted - nidaq_events_s[idx])
        idx_prev = np.clip(idx - 1, 0, len(nidaq_events_s) - 1)
        dists_prev = np.abs(shifted - nidaq_events_s[idx_prev])
        min_dists = np.minimum(dists, dists_prev)
        count = np.sum(min_dists < match_tolerance_s)
        if count > best_count:
            best_count = count
            best_offset = off

    logger.info(
        f"Coarse alignment: offset={best_offset:.2f}s, "
        f"matched {best_count}/{len(cam_transitions_s)} transitions"
    )
    return best_offset


def _coarse_offset_from_metadata(
    metadata_path: str,
    video_path: str,
    nidaq_baseline_on_s: np.ndarray,
    n_sample: int = 5000,
) -> float:
    """Estimate rough offset by sparse full-frame luminance + brute-force search.

    Uses **full-frame** luminance (not ROI) for maximum signal strength.
    """
    import cv2

    ts_ms, _, _ = load_camera_metadata(metadata_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(
            f"Cannot open video for coarse offset: {video_path}. "
            "Returning offset=0.0 — downstream detection may fail."
        )
        return 0.0

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // n_sample)
        indices = np.arange(0, total, step)

        lum = np.empty(len(indices), dtype=np.float32)
        sample_ts_s = np.empty(len(indices), dtype=np.float64)

        for i, fi in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lum[i] = gray[::8, ::8].astype(np.float32).mean()  # 8x downsample for coarse scan speed
            else:
                lum[i] = lum[i - 1] if i > 0 else 128.0
            sample_ts_s[i] = ts_ms[min(fi, len(ts_ms) - 1)] / 1000.0
    finally:
        cap.release()

    # Detect transitions via large luminance derivatives
    deriv = np.abs(np.diff(lum))
    med_d = np.median(deriv)
    mad_d = np.median(np.abs(deriv - med_d))
    if mad_d < 1e-6:
        thresh = np.percentile(deriv, 97)
    else:
        thresh = med_d + VIDEO_SYNC_DERIV_SIGMA_MULT * _MAD_SCALE * mad_d

    peaks = np.where(deriv > thresh)[0]
    if len(peaks) == 0:
        logger.warning("Coarse scan: no transitions detected")
        return 0.0

    transition_times_s = _cluster_times(sample_ts_s[peaks], cluster_s=2.0)
    logger.info(
        f"Coarse scan: {len(transition_times_s)} transitions from "
        f"{len(indices)} sampled frames"
    )
    return coarse_align(transition_times_s, nidaq_baseline_on_s)


def _cluster_times(times_s: np.ndarray, cluster_s: float) -> np.ndarray:
    """Cluster nearby timestamps and return the first in each cluster."""
    if len(times_s) == 0:
        return np.array([])
    times_sorted = np.sort(times_s)
    clusters = [times_sorted[0]]
    for i in range(1, len(times_sorted)):
        if times_sorted[i] - times_sorted[i - 1] > cluster_s:
            clusters.append(times_sorted[i])
    return np.array(clusters)


# =====================================================================
# Clock model fitting
# =====================================================================


def fit_clock_model(
    detected_cam_s: np.ndarray,
    detected_nidaq_s: np.ndarray,
    n_baseline_on: int,
    n_cv_folds: int = 5,
    outlier_n_iter: int = VIDEO_SYNC_OUTLIER_N_ITER,
    outlier_sigma: float = VIDEO_SYNC_OUTLIER_SIGMA,
) -> SyncResult:
    """Fit linear clock model with Theil-Sen regression and outlier rejection.

    Steps:
    1. Theil-Sen robust regression (deterministic, breakdown ~29%)
    2. Iterative MAD outlier rejection (with minimum-inlier guard)
    3. OLS refit on inliers for minimum-variance estimates
    4. Durbin-Watson autocorrelation test
    5. Temporal-block cross-validation (5 contiguous folds)
    """
    n_original = len(detected_cam_s)

    if n_original < _MIN_ANCHORS_FIT:
        logger.error(f"Only {n_original} matches found - insufficient for fitting")
        return SyncResult(
            slope=1.0, offset=0.0, n_anchors=n_original,
            n_baseline_on=n_baseline_on,
            rmse_ms=999.0, max_residual_ms=999.0, cv_rmse_ms=999.0,
            slope_ppm=0.0, durbin_watson=0.0,
        )

    # Track original indices for outlier visualization
    cam_s = detected_cam_s.copy()
    nidaq_s = detected_nidaq_s.copy()
    original_indices = np.arange(n_original)

    # Step 1: Theil-Sen robust regression (explicit 'separate' = classic method)
    from scipy.stats import theilslopes

    result = theilslopes(nidaq_s, cam_s, method="separate")
    slope = float(result[0])
    offset = float(result[1])

    # Step 2: Iterative MAD outlier rejection with minimum-inlier guard
    min_inliers = max(_MIN_ANCHORS_CV, int(0.5 * n_original))

    for iteration in range(outlier_n_iter):
        predicted = slope * cam_s + offset
        residuals_ms = (nidaq_s - predicted) * 1000.0

        med_res = np.median(residuals_ms)
        mad = np.median(np.abs(residuals_ms - med_res))
        thresh = outlier_sigma * _MAD_SCALE * mad

        inliers = np.abs(residuals_ms - med_res) < thresh
        n_inliers = int(inliers.sum())

        logger.info(
            f"  Outlier rejection iter {iteration}: "
            f"{n_inliers}/{len(inliers)} inliers, MAD={mad:.1f} ms"
        )

        if inliers.all():
            break

        # Guard: don't reject below minimum
        if n_inliers < min_inliers:
            logger.warning(
                f"  Stopping: would drop to {n_inliers} < {min_inliers} minimum"
            )
            break

        cam_s = cam_s[inliers]
        nidaq_s = nidaq_s[inliers]
        original_indices = original_indices[inliers]

        result = theilslopes(nidaq_s, cam_s, method="separate")
        slope = float(result[0])
        offset = float(result[1])

    # Step 3: Final OLS refit on inliers
    A = np.column_stack([cam_s, np.ones(len(cam_s))])
    params, _, _, _ = np.linalg.lstsq(A, nidaq_s, rcond=None)
    slope = float(params[0])
    offset = float(params[1])

    # Recompute residuals with final OLS parameters
    predicted = slope * cam_s + offset
    residuals_ms = (nidaq_s - predicted) * 1000.0

    n_anchors = len(cam_s)
    rmse_ms = float(np.sqrt(np.mean(residuals_ms ** 2)))
    max_residual_ms = float(np.max(np.abs(residuals_ms)))
    slope_ppm = (slope - 1.0) * 1e6

    # Step 4: Durbin-Watson statistic
    if len(residuals_ms) > 2:
        dw = float(np.sum(np.diff(residuals_ms) ** 2) / np.sum(residuals_ms ** 2))
    else:
        dw = 0.0

    # Step 5: Temporal-block cross-validation
    cv_rmse = _temporal_cv(cam_s, nidaq_s, n_folds=n_cv_folds)

    # Build inlier mask relative to original input length (for outlier viz)
    inlier_mask = np.zeros(n_original, dtype=bool)
    inlier_mask[original_indices] = True

    # Build full residual array (NaN for outliers, computed residual for inliers)
    all_residuals_ms = np.full(n_original, np.nan)
    all_predicted = slope * detected_cam_s + offset
    all_res = (detected_nidaq_s - all_predicted) * 1000.0
    all_residuals_ms[inlier_mask] = all_res[inlier_mask]

    return SyncResult(
        slope=slope,
        offset=offset,
        n_anchors=n_anchors,
        n_baseline_on=n_baseline_on,
        rmse_ms=rmse_ms,
        max_residual_ms=max_residual_ms,
        cv_rmse_ms=cv_rmse,
        slope_ppm=slope_ppm,
        durbin_watson=dw,
        detection_method="derivative",
        inlier_mask=inlier_mask,
        residuals_ms=all_residuals_ms,
        matched_cam_ms=detected_cam_s * 1000.0,
        matched_nidaq_s=detected_nidaq_s,
    )


def _temporal_cv(
    cam_s: np.ndarray,
    nidaq_s: np.ndarray,
    n_folds: int = 5,
) -> float:
    """Temporal-block cross-validation of the linear clock model.

    Uses contiguous blocks (not random splits) to test both interpolation
    and extrapolation accuracy.
    """
    n = len(cam_s)
    if n < _MIN_ANCHORS_CV:
        return 999.0

    fold_size = n // n_folds
    cv_errors = []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = min(test_start + fold_size, n)
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_start:test_end] = True
        train_mask = ~test_mask

        X_train = cam_s[train_mask]
        y_train = nidaq_s[train_mask]
        X_test = cam_s[test_mask]
        y_test = nidaq_s[test_mask]

        A = np.column_stack([X_train, np.ones(len(X_train))])
        params, _, _, _ = np.linalg.lstsq(A, y_train, rcond=None)
        pred = params[0] * X_test + params[1]
        errors_ms = (y_test - pred) * 1000.0
        cv_errors.append(np.sqrt(np.mean(errors_ms ** 2)))

    return float(np.mean(cv_errors))


# =====================================================================
# Conversion functions
# =====================================================================


def camera_to_nidaq(t_camera_ms, slope: float, offset: float):
    """Convert camera timestamp(s) (ms) to NI-DAQ time (seconds)."""
    t = np.asarray(t_camera_ms, dtype=np.float64)
    return slope * (t / 1000.0) + offset


def nidaq_to_camera(t_nidaq_s, slope: float, offset: float):
    """Convert NI-DAQ time (seconds) to camera timestamp(s) (ms)."""
    t = np.asarray(t_nidaq_s, dtype=np.float64)
    return ((t - offset) / slope) * 1000.0


# =====================================================================
# Persistence (JSON sidecar)
# =====================================================================


def save_video_sync(
    session_name: str,
    eye_cam: Optional[SyncResult] = None,
    front_cam: Optional[SyncResult] = None,
    sync_dir: Optional[str] = None,
) -> str:
    """Save sync parameters to JSON sidecar file."""
    out_dir = sync_dir or VIDEO_SYNC_DIR
    os.makedirs(out_dir, exist_ok=True)

    data = {"session_name": str(session_name)}
    quality_tiers = []

    if eye_cam is not None:
        data["eye_cam"] = eye_cam.to_dict()
        quality_tiers.append(eye_cam.quality)
    if front_cam is not None:
        data["front_cam"] = front_cam.to_dict()
        quality_tiers.append(front_cam.quality)

    if "failed" in quality_tiers:
        data["quality"] = "failed"
    elif "review" in quality_tiers:
        data["quality"] = "review"
    elif quality_tiers:
        data["quality"] = "good"
    else:
        data["quality"] = "failed"

    path = os.path.join(out_dir, f"{session_name}_video_sync.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved video sync: {path} (quality={data['quality']})")
    return path


def load_video_sync(
    session_name: str,
    sync_dir: Optional[str] = None,
) -> Optional[dict]:
    """Load sync parameters from JSON sidecar."""
    out_dir = sync_dir or VIDEO_SYNC_DIR
    path = os.path.join(out_dir, f"{session_name}_video_sync.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# =====================================================================
# Diagnostic figure
# =====================================================================


def plot_sync_diagnostic(
    sync: SyncResult,
    camera_label: str = "Eye cam",
    session_name: str = "",
    save_path: Optional[str] = None,
):
    """Generate 4-panel diagnostic figure for sync quality assessment."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Video Sync Diagnostic - {session_name} ({camera_label})",
        fontsize=14, fontweight="bold",
    )

    if sync.matched_cam_ms is None or sync.residuals_ms is None:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    cam_s = sync.matched_cam_ms / 1000.0
    inlier = (
        sync.inlier_mask
        if sync.inlier_mask is not None
        else np.ones(len(cam_s), dtype=bool)
    )
    res = sync.residuals_ms

    # Panel A: Residuals vs time (outliers shown in red)
    ax = axes[0, 0]
    ax.scatter(
        cam_s[inlier] / 3600, res[inlier], s=2, alpha=0.5, label="Inlier"
    )
    if (~inlier).any():
        ax.scatter(
            cam_s[~inlier] / 3600, res[~inlier],
            s=8, c="red", alpha=0.7, label="Outlier",
        )
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Camera time (hours)")
    ax.set_ylabel("Residual (ms)")
    ax.set_title("A. Residuals vs Time")
    ax.legend(fontsize=8)

    # Panel B: Matched anchors
    ax = axes[0, 1]
    ax.scatter(cam_s[inlier], sync.matched_nidaq_s[inlier], s=2, alpha=0.5)
    t_range = np.array([cam_s.min(), cam_s.max()])
    ax.plot(t_range, sync.slope * t_range + sync.offset, "r-", lw=1.5, label="Fit")
    ax.set_xlabel("Camera time (s)")
    ax.set_ylabel("NI-DAQ time (s)")
    ax.set_title("B. Clock Model Fit")
    ax.legend(fontsize=8)

    # Panel C: Residual histogram (inliers only)
    ax = axes[1, 0]
    ax.hist(res[inlier], bins=50, alpha=0.7, edgecolor="k", lw=0.5)
    ax.axvline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Residual (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"C. Residual Distribution (RMSE={sync.rmse_ms:.1f} ms)")

    # Panel D: Summary metrics
    ax = axes[1, 1]
    ax.axis("off")
    metrics = [
        f"Quality: {sync.quality.upper()}",
        f"Detection: {sync.detection_method}",
        f"Slope: {sync.slope:.8f} ({sync.slope_ppm:+.1f} ppm)",
        f"Offset: {sync.offset:.4f} s",
        f"Anchors: {sync.n_anchors} / {sync.n_baseline_on} ({sync.coverage:.1%})",
        f"RMSE: {sync.rmse_ms:.2f} ms",
        f"Max residual: {sync.max_residual_ms:.2f} ms",
        f"CV RMSE: {sync.cv_rmse_ms:.2f} ms",
        f"Durbin-Watson: {sync.durbin_watson:.3f}",
        f"ROI: {sync.roi}",
    ]
    text = "\n".join(metrics)
    color = {"good": "#2e7d32", "review": "#f57f17", "failed": "#c62828"}
    ax.text(
        0.1, 0.9, text, transform=ax.transAxes,
        fontsize=12, fontfamily="monospace", verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=color.get(sync.quality, "#bbb"),
            alpha=0.15,
        ),
    )
    ax.set_title("D. Summary")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved diagnostic figure: {save_path}")

    return fig


# =====================================================================
# High-level session sync pipeline
# =====================================================================


def sync_session(
    session_name: str,
    nidaq_baseline_on_s: np.ndarray,
    camera_root: Optional[str] = None,
    subject: str = "BG_046",
    roi: Optional[Tuple[int, int, int, int]] = None,
    sync_dir: Optional[str] = None,
    fig_dir: Optional[str] = None,
    force: bool = False,
    progress: bool = True,
) -> dict:
    """End-to-end sync pipeline for one session.

    1. Find camera files
    2. Estimate rough offset via sparse full-frame luminance scan
    3. Per-trial derivative onset detection with sub-frame interpolation
    4. Theil-Sen fit with iterative outlier rejection
    5. Temporal-block cross-validation
    6. Save sync JSON sidecar + diagnostic figure

    Returns the sync dict (same as ``load_video_sync()`` output).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visdetect.analysis.config import VIDEO_SYNC_FIG_DIR

    _sync_dir = sync_dir or VIDEO_SYNC_DIR
    _fig_dir = fig_dir or VIDEO_SYNC_FIG_DIR

    # Check if already done
    if not force:
        existing = load_video_sync(session_name, sync_dir=_sync_dir)
        if existing is not None:
            logger.info(
                f"Session {session_name}: sync already exists "
                f"(quality={existing['quality']})"
            )
            return existing

    cam_files = find_camera_files(
        session_name, camera_root=camera_root, subject=subject
    )
    if not cam_files:
        raise FileNotFoundError(f"No camera files found for session {session_name}")

    results = {}

    for cam_key in ["eye_cam", "front_cam"]:
        if cam_key not in cam_files:
            logger.warning(f"Session {session_name}: {cam_key} not found, skipping")
            continue

        video_path = cam_files[cam_key]["video"]
        meta_path = cam_files[cam_key]["metadata"]

        logger.info(f"Processing {session_name} / {cam_key}")

        # Determine ROI (only eye cam has a validated default ROI)
        used_roi = roi
        if used_roi is None:
            if cam_key == "eye_cam":
                used_roi = VIDEO_SYNC_DEFAULT_EYE_ROI
            else:
                logger.info(
                    f"  {cam_key}: skipping (sync derived from eye cam model)"
                )
                continue

        # Step 1: Rough offset via sparse full-frame luminance
        rough_offset = _coarse_offset_from_metadata(
            meta_path, video_path, nidaq_baseline_on_s,
        )

        # Step 2: Per-trial derivative onset detection
        detected_cam_s, detected_nidaq_s = detect_onsets_derivative(
            video_path, meta_path, nidaq_baseline_on_s,
            rough_offset_s=rough_offset,
            roi=used_roi,
            progress=progress,
        )

        # Step 3: Fit clock model
        sync_result = fit_clock_model(
            detected_cam_s, detected_nidaq_s,
            n_baseline_on=len(nidaq_baseline_on_s),
        )
        sync_result.roi = list(used_roi) if used_roi else None

        # Record frame count
        ts_ms, _, _ = load_camera_metadata(meta_path)
        sync_result.n_frames = len(ts_ms)

        results[cam_key] = sync_result

        # Diagnostic figure
        fig_path = os.path.join(_fig_dir, f"{session_name}_{cam_key}_sync.png")
        fig = plot_sync_diagnostic(
            sync_result,
            camera_label=cam_key.replace("_", " ").title(),
            session_name=str(session_name),
            save_path=fig_path,
        )
        plt.close(fig)

    # Save JSON sidecar
    save_video_sync(
        session_name,
        eye_cam=results.get("eye_cam"),
        front_cam=results.get("front_cam"),
        sync_dir=_sync_dir,
    )

    return load_video_sync(session_name, sync_dir=_sync_dir)
