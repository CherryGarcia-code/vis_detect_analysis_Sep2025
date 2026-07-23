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
visible to the right of the frame (behind/above/below the mouse head).

At Baseline_ON, the uniform bright ITI screen transitions to a drifting
grating, which has lower mean luminance — producing a luminance **drop** in
the background glow region.  The grating can appear anywhere to the right of
x~600 but the mouse head (y~250-750) blocks the signal, so we use two strips
above and below the head.

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

import datetime as _dt
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

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
    VIDEO_SYNC_DERIV_MAX_THRESH,
    VIDEO_SYNC_DERIV_MIN_THRESH,
    VIDEO_SYNC_DERIV_PRE_FRAMES,
    VIDEO_SYNC_DERIV_SEARCH_FRAMES,
    VIDEO_SYNC_DERIV_SIGMA_MULT,
    VIDEO_SYNC_MASK_MIN_COMPONENT,
    VIDEO_SYNC_MASK_MORPH_OPEN,
    VIDEO_SYNC_MASK_N_TRANSITIONS,
    VIDEO_SYNC_MASK_POST_FRAMES,
    VIDEO_SYNC_MASK_PRE_FRAMES,
    VIDEO_SYNC_MASK_X_MIN,
    VIDEO_SYNC_MAX_DRIFT_PPM,
    VIDEO_SYNC_MAX_RESIDUAL_MS,
    VIDEO_SYNC_MIN_COVERAGE,
    VIDEO_SYNC_OUTLIER_N_ITER,
    VIDEO_SYNC_OUTLIER_SIGMA,
    CORNEAL_CAL_N_TRANSITIONS,
    CORNEAL_CAL_PRE_FRAMES,
    CORNEAL_CAL_POST_FRAMES,
    CORNEAL_CAL_SEARCH_MARGIN_PX,
    CORNEAL_CAL_PUPIL_EXCLUSION_FACTOR,
    CORNEAL_CAL_ANGLE_MIN_DEG,
    CORNEAL_CAL_ANGLE_MAX_DEG,
    CORNEAL_CAL_MAX_DIST_PX,
    CORNEAL_CAL_MASK_MIN_AREA_PX,
    CORNEAL_CAL_MASK_MAX_AREA_PX,
    CORNEAL_CAL_MASK_THRESHOLD_PCT,
    CORNEAL_CAL_PUPIL_MIN_AREA_PX,
    CORNEAL_CAL_PUPIL_MAX_AREA_PX,
    CORNEAL_CAL_PUPIL_MIN_CIRCULARITY,
)

from visdetect.analysis.config import CAMERA_ROOT, VIDEO_SYNC_DIR

# Quality-tier thresholds for SyncResult.quality
_GOOD_RMSE_MS = 20.0
_GOOD_DW_RANGE = (1.5, 2.5)
_REVIEW_RMSE_MS = 40.0
_REVIEW_COVERAGE = 0.60

# ROI type: 4-tuple rectangle (y0, y1, x0, x1),
#   single polygon as list of (x, y) vertices,
#   multiple polygons as list of list of (x, y) vertices,
#   OR a raw 2-D boolean numpy mask.
RoiSpec = Union[
    Tuple[int, int, int, int],
    Sequence[Tuple[int, int]],
    Sequence[Sequence[Tuple[int, int]]],
    np.ndarray,
]


def _is_rectangle(roi) -> bool:
    """Check if roi is a 4-tuple rectangle spec."""
    if isinstance(roi, np.ndarray):
        return False
    return (
        isinstance(roi, (tuple, list))
        and len(roi) == 4
        and all(isinstance(v, (int, np.integer)) for v in roi)
    )


def _is_multi_polygon(roi) -> bool:
    """Check if roi is a list of polygons (list of list of [x,y])."""
    if isinstance(roi, np.ndarray):
        return False
    if not isinstance(roi, (list, tuple)) or len(roi) == 0:
        return False
    first = roi[0]
    if not isinstance(first, (list, tuple)) or len(first) == 0:
        return False
    # Single polygon: first element is [x, y] (2 ints)
    # Multi polygon: first element is [[x, y], ...] (list of pairs)
    inner = first[0]
    return isinstance(inner, (list, tuple)) and len(inner) == 2


def _build_roi_mask(
    roi: RoiSpec,
    frame_h: int,
    frame_w: int,
) -> np.ndarray:
    """Build a 2-D boolean mask from a rectangle, polygon, multi-polygon, or raw mask.

    Parameters
    ----------
    roi : RoiSpec
        - ``np.ndarray`` 2-D boolean mask (passed through with shape validation)
        - 4-tuple ``(y0, y1, x0, x1)`` for a rectangle
        - list of ``(x, y)`` vertices for a single polygon
        - list of polygons (each a list of ``(x, y)`` vertices)
    frame_h, frame_w : int
        Frame dimensions.

    Returns
    -------
    mask : np.ndarray, shape (frame_h, frame_w), dtype bool
    """
    import cv2

    # Raw numpy mask — validate and pass through
    if isinstance(roi, np.ndarray):
        if roi.ndim != 2 or roi.shape != (frame_h, frame_w):
            raise ValueError(
                f"ROI mask shape {roi.shape} != frame ({frame_h}, {frame_w})"
            )
        return roi.astype(bool)

    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)

    if _is_rectangle(roi):
        y0, y1, x0, x1 = roi
        mask[y0:y1, x0:x1] = 1
        return mask.astype(bool)

    if _is_multi_polygon(roi):
        # Multiple polygons — union them
        for poly_verts in roi:
            pts = np.array(poly_verts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 1)
        return mask.astype(bool)

    # Single polygon: list of (x, y) vertices
    pts = np.array(roi, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def _roi_to_json(roi: RoiSpec):
    """Convert an RoiSpec to a JSON-serialisable value."""
    if isinstance(roi, np.ndarray):
        return {"type": "screen_mask", "n_pixels": int(roi.sum()),
                "shape": list(roi.shape)}
    if _is_rectangle(roi):
        return [int(v) for v in roi]
    if _is_multi_polygon(roi):
        return [[[int(x), int(y)] for x, y in poly] for poly in roi]
    # Single polygon: list of (x, y) pairs
    return [[int(x), int(y)] for x, y in roi]


# =====================================================================
# Data-driven screen mask
# =====================================================================


def build_screen_mask(
    video_path: str,
    metadata_path: str,
    transition_times_s: np.ndarray,
    rough_offset_s: float,
    n_transitions: int = VIDEO_SYNC_MASK_N_TRANSITIONS,
    pre_frames: int = VIDEO_SYNC_MASK_PRE_FRAMES,
    post_frames: int = VIDEO_SYNC_MASK_POST_FRAMES,
    morph_open_size: int = VIDEO_SYNC_MASK_MORPH_OPEN,
    min_component_area: int = VIDEO_SYNC_MASK_MIN_COMPONENT,
    x_min: int = VIDEO_SYNC_MASK_X_MIN,
    cache_dir: Optional[str] = None,
    session_name: Optional[str] = None,
    force: bool = False,
) -> Tuple[np.ndarray, dict]:
    """Build a data-driven binary screen mask from transition difference images.

    Computes the **trial-to-trial standard deviation** of (post - pre) pixel
    values across known Baseline_ON transitions.  This separates three classes:

    - **Direct screen pixels** (high std): grating phase varies randomly
      trial-to-trial, so the same pixel sees bright-bar on one trial and
      dark-bar on another → signed diff flips → high std.
    - **Indirect illumination** (low std): screen glow on mouse fur/whiskers
      always gets dimmer (gray → grating has lower mean luminance) → signed
      diff is consistently negative → low std despite nonzero mean.
    - **Mouse head** (near-zero std): no change → zero diff → zero std.

    This is superior to ``mean(|post - pre|)`` which conflates direct and
    indirect illumination changes.

    Parameters
    ----------
    video_path : str
        Path to the eye camera MP4.
    metadata_path : str
        Path to camera metadata CSV.
    transition_times_s : np.ndarray
        NI-DAQ Baseline_ON times in seconds.
    rough_offset_s : float
        Coarse camera-to-NI-DAQ offset (seconds).
    n_transitions : int
        Number of transitions to sample (evenly spaced).
    pre_frames, post_frames : int
        Frames to average before/after each transition.
    morph_open_size : int
        Morphological opening kernel size.
    min_component_area : int
        Minimum connected component area (pixels) to keep.
    x_min : int
        Spatial prior: zero out mask for all columns x < x_min.  The screen
        glow is always on the right side of the frame; this eliminates
        spurious whisker/body-motion regions on the left.
    cache_dir : str, optional
        Directory for caching.  Defaults to ``VIDEO_SYNC_DIR / screen_masks``.
    session_name : str, optional
        Session name for cache file naming.
    force : bool
        If True, recompute even if cached.

    Returns
    -------
    mask : np.ndarray, shape (H, W), dtype bool
        Binary screen mask.
    info : dict
        Diagnostic metadata (threshold, n_used, screen_fraction, avg_diff).
    """
    import cv2
    from scipy import ndimage as ndi

    # ── Cache check ─────────────────────────────────────────────────
    if cache_dir is None:
        cache_dir = os.path.join(VIDEO_SYNC_DIR, "screen_masks")
    if session_name and not force:
        cache_path = os.path.join(cache_dir, f"{session_name}_screen_mask.npz")
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            mask = data["mask"]
            info = data["info"].item()
            logger.info(f"  Loaded cached screen mask: {mask.sum()} pixels "
                        f"({mask.mean():.1%} of frame)")
            return mask, info

    # ── Load metadata and open video ────────────────────────────────
    ts_ms, _, _ = load_camera_metadata(metadata_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Select evenly-spaced transitions
        n_avail = len(transition_times_s)
        n_use = min(n_transitions, n_avail)
        if n_use < 5:
            raise ValueError(
                f"Too few transitions ({n_avail}) to build screen mask"
            )
        indices = np.round(np.linspace(0, n_avail - 1, n_use)).astype(int)
        selected_times = transition_times_s[indices]

        # ── Collect per-transition (post - pre) difference images ─────
        # We store the SIGNED diff for each transition.  Direct screen
        # pixels flip sign trial-to-trial (grating phase varies) → high
        # std.  Indirect illumination pixels change consistently (always
        # dimmer) → low std.  Mouse head pixels ≈ 0 → low std.
        # Using std(post-pre) instead of mean(|post-pre|) separates
        # direct screen glow from indirect illumination changes.
        diff_images = []

        for nidaq_t in selected_times:
            cam_ms = (nidaq_t - rough_offset_s) * 1000.0
            center_frame = int(np.searchsorted(ts_ms, cam_ms))

            start_f = center_frame - pre_frames
            end_f = center_frame + post_frames
            if start_f < 0 or end_f >= len(ts_ms):
                continue

            # Read pre-transition frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            pre_sum = np.zeros((frame_h, frame_w), dtype=np.float64)
            pre_count = 0
            for _ in range(pre_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(
                    np.float64
                )
                pre_sum += gray
                pre_count += 1

            # Read post-transition frames
            post_sum = np.zeros((frame_h, frame_w), dtype=np.float64)
            post_count = 0
            for _ in range(post_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(
                    np.float64
                )
                post_sum += gray
                post_count += 1

            if pre_count == 0 or post_count == 0:
                continue

            pre_img = pre_sum / pre_count
            post_img = post_sum / post_count
            diff_images.append(post_img - pre_img)

        n_used = len(diff_images)
        if n_used < 5:
            raise RuntimeError(
                f"Only {n_used} transitions produced valid frames "
                f"(need >= 5 for std computation)"
            )

        diff_stack = np.stack(diff_images, axis=0)  # (n_used, H, W)
        # Trial-to-trial std: high where grating phase varies (screen),
        # low where illumination change is consistent (indirect) or absent
        avg_diff = np.std(diff_stack, axis=0)

    finally:
        cap.release()

    # ── Threshold: Otsu on the averaged difference image ────────────
    # Normalize to uint8 for cv2.threshold
    diff_max = avg_diff.max()
    if diff_max < 1e-6:
        raise RuntimeError("Averaged difference image is all zeros")
    diff_u8 = (avg_diff / diff_max * 255).astype(np.uint8)

    otsu_thresh, mask_u8 = cv2.threshold(
        diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Convert Otsu threshold back to original scale
    otsu_orig = otsu_thresh / 255.0 * diff_max

    # Compute between-class variance ratio (Otsu quality diagnostic)
    fg_vals = avg_diff[mask_u8 > 0]
    bg_vals = avg_diff[mask_u8 == 0]
    total_var = np.var(avg_diff)
    if total_var > 1e-12 and len(fg_vals) > 0 and len(bg_vals) > 0:
        w_fg = len(fg_vals) / avg_diff.size
        w_bg = len(bg_vals) / avg_diff.size
        between_var = w_fg * w_bg * (fg_vals.mean() - bg_vals.mean()) ** 2
        otsu_quality = between_var / total_var
    else:
        otsu_quality = 0.0

    mask = mask_u8.astype(bool)
    screen_frac = mask.mean()

    # Sanity check — fall back to percentile if Otsu is degenerate
    if screen_frac < 0.02 or screen_frac > 0.60:
        logger.warning(
            f"  Otsu mask covers {screen_frac:.1%} of frame (outside 2-60%); "
            f"falling back to 75th percentile threshold"
        )
        p75 = np.percentile(avg_diff, 75)
        mask = avg_diff > p75
        screen_frac = mask.mean()
        otsu_orig = p75

    # ── Morphological cleanup ───────────────────────────────────────
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_open_size, morph_open_size)
    )
    mask_u8_clean = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
    )

    # ── Spatial prior: screen is always on the right ──────────────
    if x_min > 0:
        mask_u8_clean[:, :x_min] = 0

    # Connected component filtering
    labeled, n_components = ndi.label(mask_u8_clean)
    for comp_id in range(1, n_components + 1):
        comp_area = (labeled == comp_id).sum()
        if comp_area < min_component_area:
            mask_u8_clean[labeled == comp_id] = 0

    mask = mask_u8_clean.astype(bool)
    screen_frac = mask.mean()

    logger.info(
        f"  Screen mask: {mask.sum()} pixels ({screen_frac:.1%} of frame), "
        f"Otsu quality={otsu_quality:.3f}, n_transitions={n_used}"
    )

    # ── Build info dict ─────────────────────────────────────────────
    info = {
        "threshold": float(otsu_orig),
        "otsu_quality": float(otsu_quality),
        "n_transitions_used": int(n_used),
        "n_screen_pixels": int(mask.sum()),
        "screen_fraction": float(screen_frac),
        "x_min": int(x_min),
        "min_component_area": int(min_component_area),
        "avg_diff": avg_diff.astype(np.float32),
    }

    # ── Cache ───────────────────────────────────────────────────────
    if session_name:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{session_name}_screen_mask.npz")
        np.savez_compressed(
            cache_path, mask=mask, info=np.array(info, dtype=object)
        )
        logger.info(f"  Cached screen mask to {cache_path}")

    return mask, info


# =====================================================================
# Data classes
# =====================================================================


@dataclass
class OnsetDetectionResult:
    """Per-trial onset detection output with confidence scores."""

    detected_cam_s: np.ndarray = field(repr=False)
    detected_nidaq_s: np.ndarray = field(repr=False)
    confidence: np.ndarray = field(repr=False)
    n_trials: int = 0

    @property
    def n_detected(self) -> int:
        return len(self.detected_cam_s)

    @property
    def n_missed(self) -> int:
        return self.n_trials - self.n_detected

    @property
    def detection_rate(self) -> float:
        return self.n_detected / max(self.n_trials, 1)


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
    per_trial_overrides: Optional[Dict[int, int]] = field(default=None, repr=False)

    @property
    def coverage(self) -> float:
        return self.n_anchors / max(self.n_baseline_on, 1)

    @property
    def quality(self) -> str:
        """Composite quality tier: good / review / failed."""
        # Manual 2-anchor fits don't have the regression-style metrics
        # the rest of this logic checks. A manual fit is "good" iff the
        # slope is physically sensible and there are >=2 anchors.
        if self.detection_method == "manual_slope_fit":
            return "good" if (self.slope > 0 and self.n_anchors >= 2) else "failed"

        if self.detection_method == "manual_multianchor":
            from visdetect.analysis.constants import (
                VIDEO_SYNC_MANUAL_GOOD_CV_MS, VIDEO_SYNC_MANUAL_REVIEW_CV_MS,
                VIDEO_SYNC_MANUAL_MIN_ANCHORS, VIDEO_SYNC_MAX_DRIFT_PPM,
            )
            if self.slope <= 0 or self.n_anchors < VIDEO_SYNC_MANUAL_MIN_ANCHORS:
                return "failed"
            low_drift = abs(self.slope_ppm) < VIDEO_SYNC_MAX_DRIFT_PPM
            if self.cv_rmse_ms < VIDEO_SYNC_MANUAL_GOOD_CV_MS and low_drift:
                return "good"
            if self.cv_rmse_ms < VIDEO_SYNC_MANUAL_REVIEW_CV_MS and low_drift:
                return "review"
            return "failed"

        good_rmse = self.rmse_ms < _GOOD_RMSE_MS
        good_maxres = self.max_residual_ms < VIDEO_SYNC_MAX_RESIDUAL_MS
        good_dw = _GOOD_DW_RANGE[0] <= self.durbin_watson <= _GOOD_DW_RANGE[1]
        low_drift = abs(self.slope_ppm) < VIDEO_SYNC_MAX_DRIFT_PPM

        if (
            good_rmse and good_maxres and good_dw
            and (self.coverage >= VIDEO_SYNC_MIN_COVERAGE
                 or self.n_anchors >= 100)
        ):
            return "good"
        elif (
            self.rmse_ms < _REVIEW_RMSE_MS
            and low_drift
            and (self.coverage >= _REVIEW_COVERAGE
                 or (self.n_anchors >= 30 and good_dw))
        ):
            return "review"
        elif (
            # Low-coverage fallback: few detections but clock model is stable.
            self.rmse_ms < _REVIEW_RMSE_MS * 1.5  # 60 ms
            and self.n_anchors >= 30
            and good_dw and low_drift
        ):
            return "review"
        else:
            return "failed"

    def to_dict(self) -> dict:
        # ORIENTATION CONTRACT: the persisted slope/offset orientation is
        # detection_method-dependent. ``manual_multianchor`` / ``derivative``
        # store  nidaq = slope*cam + offset  (the orientation camera_to_nidaq /
        # nidaq_to_camera assume); legacy ``manual_slope_fit`` stores the
        # INVERSE  video = slope*nidaq + offset. Downstream consumers must
        # branch on detection_method before applying those converters.
        d = {
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
        if self.per_trial_overrides is not None:
            d["per_trial_overrides"] = self.per_trial_overrides
        return d


def fit_2anchor_clock(
    anchors: List[dict],
    fps: float,
    n_baseline_on: int,
) -> "SyncResult":
    """Fit a linear clock model from 2+ v2 anchor entries.

    Model: ``video_time_s = slope * nidaq_baseline_on_s + offset``.

    For exactly 2 anchors: closed-form linear fit (rmse_ms = 0, max_residual_ms = 0).
    For >=3 anchors: least-squares fit; rmse_ms and max_residual_ms from residuals.

    Parameters
    ----------
    anchors : list of dict
        Each dict must have ``nidaq_baseline_on_s`` and ``video_time_s`` keys.
        The fit uses ``video_time_s`` directly; ``fps`` is currently unused and
        reserved for forward compatibility.
    fps : float
        Camera frame rate (reserved; not used in the fit calculation).
    n_baseline_on : int
        Total number of Baseline_ON events in the session (for coverage reporting).

    Returns a SyncResult with detection_method = "manual_slope_fit".
    Raises ValueError on fewer than 2 anchors, duplicate nidaq times, or
    non-positive slope.
    """
    if len(anchors) < 2:
        raise ValueError(
            f"fit_2anchor_clock needs at least 2 anchors; got {len(anchors)}"
        )

    x = np.array(
        [float(a["nidaq_baseline_on_s"]) for a in anchors], dtype=np.float64
    )
    y = np.array(
        [float(a["video_time_s"]) for a in anchors], dtype=np.float64
    )

    if len(anchors) == 2 and x[0] == x[1]:
        raise ValueError(
            f"Both anchors have the same nidaq baseline_on time ({x[0]:.6f}s); "
            f"cannot fit a slope. Check for duplicate/erroneous anchor entries."
        )

    if len(anchors) == 2:
        slope = float((y[1] - y[0]) / (x[1] - x[0]))
        offset = float(y[0] - slope * x[0])
        rmse_ms = 0.0
        max_residual_ms = 0.0
    else:
        A = np.vstack([x, np.ones_like(x)]).T
        soln, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        slope = float(soln[0])
        offset = float(soln[1])
        residuals_s = y - (slope * x + offset)
        rmse_ms = float(np.sqrt(np.mean(residuals_s ** 2)) * 1000.0)
        max_residual_ms = float(np.max(np.abs(residuals_s)) * 1000.0)

    if slope <= 0:
        raise ValueError(
            f"Computed slope {slope} is non-positive; anchors are likely "
            f"out of order or one is wrong. Re-verify via --scrub."
        )

    return SyncResult(
        slope=slope,
        offset=offset,
        n_anchors=int(len(anchors)),
        n_baseline_on=int(n_baseline_on),
        rmse_ms=rmse_ms,
        max_residual_ms=max_residual_ms,
        cv_rmse_ms=0.0,
        slope_ppm=float((slope - 1.0) * 1e6),
        durbin_watson=2.0,  # N/A for this fit type; report the neutral value
        detection_method="manual_slope_fit",
    )


def _loo_cv(cam_s: np.ndarray, nidaq_s: np.ndarray) -> float:
    """Leave-one-out CV RMSE (ms) for the linear clock. Requires n >= 3.

    For sparse manual anchors the dense 5-fold ``_temporal_cv`` leaves ~1
    anchor/fold (and returns its 999 sentinel below 20 anchors), so we use
    LOO: fit on all-but-one, predict the held-out anchor, RMS the errors.
    """
    n = len(cam_s)
    if n < 3:
        return float("nan")
    errs = []
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        A = np.column_stack([cam_s[m], np.ones(m.sum())])
        params, _, _, _ = np.linalg.lstsq(A, nidaq_s[m], rcond=None)
        pred = params[0] * cam_s[i] + params[1]
        errs.append(((nidaq_s[i] - pred) * 1000.0) ** 2)
    return float(np.sqrt(np.mean(errs)))


def fit_multianchor_clock(
    anchors: List[dict],
    n_baseline_on: int,
    outlier_sigma: float = VIDEO_SYNC_OUTLIER_SIGMA,
) -> SyncResult:
    """Fit a validated linear clock from >=3 manual anchors (any event type).

    Orientation matches ``camera_to_nidaq``: ``nidaq_s = slope*cam_s + offset``
    where ``cam_s = anchor['video_time_s']`` and ``nidaq_s`` is the anchor's
    ``nidaq_event_s`` (falling back to ``nidaq_baseline_on_s`` for legacy).
    Theil-Sen fit -> MAD outlier rejection -> LOO CV. detection_method =
    "manual_multianchor". Raises ValueError on <3 anchors or non-positive slope.
    """
    from scipy.stats import theilslopes
    if len(anchors) < 3:
        raise ValueError(
            f"fit_multianchor_clock needs >=3 anchors; got {len(anchors)}")

    cam_s = np.array([float(a["video_time_s"]) for a in anchors], dtype=np.float64)
    nidaq_s = np.array(
        [float(a.get("nidaq_event_s", a.get("nidaq_baseline_on_s"))) for a in anchors],
        dtype=np.float64)
    order = np.argsort(cam_s)
    cam_s, nidaq_s = cam_s[order], nidaq_s[order]

    slope, intercept, _, _ = theilslopes(nidaq_s, cam_s)
    resid_ms = (nidaq_s - (slope * cam_s + intercept)) * 1000.0
    mad = np.median(np.abs(resid_ms - np.median(resid_ms))) or 1.0
    keep = np.abs(resid_ms - np.median(resid_ms)) <= outlier_sigma * 1.4826 * mad
    if keep.sum() >= 3 and keep.sum() < len(keep):
        cam_s, nidaq_s = cam_s[keep], nidaq_s[keep]
        slope, intercept, _, _ = theilslopes(nidaq_s, cam_s)
        resid_ms = (nidaq_s - (slope * cam_s + intercept)) * 1000.0

    if slope <= 0:
        raise ValueError(f"Computed slope {slope} is non-positive; check anchors.")

    return SyncResult(
        slope=float(slope),
        offset=float(intercept),
        n_anchors=int(len(cam_s)),
        n_baseline_on=int(n_baseline_on),
        rmse_ms=float(np.sqrt(np.mean(resid_ms ** 2))),
        max_residual_ms=float(np.max(np.abs(resid_ms))),
        cv_rmse_ms=_loo_cv(cam_s, nidaq_s),
        slope_ppm=float((slope - 1.0) * 1e6),
        durbin_watson=2.0,
        detection_method="manual_multianchor",
        residuals_ms=resid_ms,
        matched_cam_ms=cam_s * 1000.0,
        matched_nidaq_s=nidaq_s,
    )


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


# =====================================================================
# Camera-metadata reconstruction (header-only / empty timestamp logs)
# =====================================================================


def build_reconstructed_timestamps(frame_count: int, fps: float) -> np.ndarray:
    """Reconstruct steady-fps per-frame timestamps (ms) for a header-only session.

    Returns ``ts[i] = i * (1000 / fps)`` for ``i`` in ``0 .. frame_count - 1``.

    Use when a session's camera metadata CSV was saved header-only, so the
    per-frame timestamps were never written. Valid only when the camera ran at a
    steady fps with negligible frame drops (true of the BG_046 eye/front cameras:
    metronomic ~50 fps, zero drops in reference sessions). ``fit_sync`` fits a
    slope mapping NI-DAQ time to video time, so a constant fps error is absorbed;
    accuracy therefore depends on *linearity*, not on the exact fps value.
    """
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    dt_ms = 1000.0 / float(fps)
    return np.arange(frame_count, dtype=np.float64) * dt_ms


def metadata_is_header_only(csv_path: str) -> bool:
    """True if a camera metadata CSV has no usable per-frame timestamps.

    Header-only/empty files (and degenerate single-row files) return True; a
    normal multi-row log returns False. Delegates to :func:`load_camera_metadata`
    so the terminal zero-row convention is handled exactly as elsewhere.
    """
    ts, _, _ = load_camera_metadata(csv_path)
    return len(ts) <= 1


def backup_header_only_metadata(csv_path: str) -> str:
    """Move a header-only metadata CSV aside to ``*_metadata.header_only.bak``.

    Returns the backup path. If a backup already exists it is preserved (assumed
    to be the first/true original) and the current file is left in place for the
    caller to overwrite — so re-runs never clobber the genuine original with a
    previously-reconstructed file.
    """
    suffix = "_metadata.csv"
    if csv_path.endswith(suffix):
        bak_path = csv_path[: -len(suffix)] + "_metadata.header_only.bak"
    else:
        bak_path = csv_path + ".header_only.bak"
    if os.path.exists(bak_path):
        return bak_path
    os.rename(csv_path, bak_path)
    return bak_path


def write_reconstructed_metadata(csv_path: str, frame_count: int, fps: float) -> None:
    """Write a reconstructed camera metadata CSV with steady-fps timestamps.

    Columns match the acquisition format (``Timestamp (ms), Acquired frames,
    Saved frames``) so :func:`load_camera_metadata` and :func:`find_camera_files`
    consume it unchanged. ``Acquired``/``Saved`` carry the 1-based frame index
    (the sync only uses the timestamp column).
    """
    import pandas as pd

    ts = build_reconstructed_timestamps(frame_count, fps)
    idx = np.arange(1, frame_count + 1, dtype=np.int64)
    df = pd.DataFrame(
        {
            "Timestamp (ms)": ts,
            "Acquired frames": idx,
            "Saved frames": idx,
        }
    )
    df.to_csv(csv_path, index=False)


def local_reconstructed_metadata_path(
    session_name: str, cam_label: str, subject: Optional[str] = None) -> str:
    """LOCAL path for a reconstructed metadata CSV (never on X:/CAMERA_ROOT).

    ``<subject_video_sync_dir>/<DDMMYYYY>_<cam_label>_metadata.reconstructed.csv``.
    """
    from visdetect.analysis.config import subject_video_sync_dir, canonical_camera_session
    sn = canonical_camera_session(session_name)
    return os.path.join(
        subject_video_sync_dir(subject), f"{sn}_{cam_label}_metadata.reconstructed.csv")


def write_local_reconstructed_metadata(
    session_name: str, cam_label: str, frame_count: int, fps: float,
    subject: Optional[str] = None) -> str:
    """Write reconstructed steady-fps metadata to LOCAL cache (never X:)."""
    out = local_reconstructed_metadata_path(session_name, cam_label, subject)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_reconstructed_metadata(out, frame_count, fps)  # existing writer, local path
    return out


def camera_dir_to_session(dirname: str, subject: str = None) -> str:
    """Convert a camera directory name (``BG_046_DDMMYY``, possibly with a
    re-record suffix like ``BG_039_010425_b``) to session ``DDMMYYYY``."""
    from visdetect.analysis.config import canonical_camera_session
    return canonical_camera_session(dirname)


def find_camera_files(
    session_name: str,
    camera_root: Optional[str] = None,
    subject: str = None,
) -> Dict[str, Dict[str, str]]:
    """Locate video + metadata files for a session.

    Returns dict like::

        {"eye_cam": {"video": "path.mp4", "metadata": "path.csv"},
         "front_cam": {"video": "path.mp4", "metadata": "path.csv"}}

    Keys are present only if both video and metadata files are found.
    """
    from visdetect.analysis.config import SUBJECT, camera_dir_token
    root = camera_root or CAMERA_ROOT
    subject = subject or SUBJECT
    token = camera_dir_token(session_name)
    cam_dir = os.path.join(root, f"{subject}_{token}")

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

    # Prefer a LOCAL reconstructed metadata CSV (X: stays read-only). The video
    # path always stays on CAMERA_ROOT; only the metadata is redirected local.
    from visdetect.analysis.config import canonical_camera_session
    sn = canonical_camera_session(session_name)
    for cam_label in list(result.keys()):
        local_meta = local_reconstructed_metadata_path(sn, cam_label, subject)
        if os.path.exists(local_meta):
            result[cam_label]["metadata"] = local_meta

    return result


# =====================================================================
# Local video staging (read-only X: source -> local scratch)
# =====================================================================


def _staging_dir(session_name: str, subject: Optional[str], staging_dir: Optional[str]) -> str:
    from visdetect.analysis.config import VIDEO_STAGING_DIR, SUBJECT, canonical_camera_session
    base = staging_dir or VIDEO_STAGING_DIR
    return os.path.join(base, subject or SUBJECT, canonical_camera_session(session_name))


def stage_session_video(
    session_name: str,
    subject: Optional[str] = None,
    cams=("eye_cam",),
    camera_root: Optional[str] = None,
    staging_dir: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Dict[str, str]]:
    """Copy a session's camera video+metadata from X: (read-only) to local scratch.

    Bulk sequential read only; never writes to CAMERA_ROOT. Returns the same
    dict shape as find_camera_files but with LOCAL paths.
    """
    import shutil
    src = find_camera_files(session_name, camera_root=camera_root, subject=subject)
    dst_dir = _staging_dir(session_name, subject, staging_dir)
    os.makedirs(dst_dir, exist_ok=True)
    out: Dict[str, Dict[str, str]] = {}
    for cam in cams:
        if cam not in src:
            continue
        out[cam] = {}
        for kind, spath in src[cam].items():
            dpath = os.path.join(dst_dir, os.path.basename(spath))
            if force or not os.path.exists(dpath):
                shutil.copy2(spath, dpath)  # copy2, never move -> source intact
            out[cam][kind] = dpath
    return out


def unstage_session_video(
    session_name: str, subject: Optional[str] = None,
    staging_dir: Optional[str] = None) -> None:
    """Delete the local staged copy for a session (frees disk)."""
    import shutil
    dst_dir = _staging_dir(session_name, subject, staging_dir)
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)


# =====================================================================
# Luminance extraction (diagnostic / full-session)
# =====================================================================


def extract_luminance(
    video_path: str,
    metadata_path: str,
    roi: Optional[RoiSpec] = None,
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
        roi_mask = None
        if roi is not None:
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            roi_mask = _build_roi_mask(roi, frame_h, frame_w)
            # Downsample the mask consistently
            roi_mask_ds = roi_mask[::ds, ::ds]

        for i in iterator:
            ret, frame = cap.read()
            if not ret:
                mean_lum[i] = mean_lum[i - 1] if i > 0 else 0.0
                spatial_var[i] = spatial_var[i - 1] if i > 0 else 0.0
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if roi_mask is not None:
                patch = gray[::ds, ::ds].astype(np.float32)[roi_mask_ds]
            else:
                patch = gray[::ds, ::ds].astype(np.float32).ravel()

            mean_lum[i] = patch.mean()
            spatial_var[i] = patch.var()
    finally:
        cap.release()

    return ts_ms, mean_lum, spatial_var


# =====================================================================
# Spatial-variance-based onset detection (primary method)
# =====================================================================


def detect_onsets_variance(
    video_path: str,
    metadata_path: str,
    baseline_on_s: np.ndarray,
    rough_offset_s: float,
    roi: RoiSpec = VIDEO_SYNC_DEFAULT_EYE_ROI,
    search_frames: int = VIDEO_SYNC_DERIV_SEARCH_FRAMES,
    pre_frames: int = VIDEO_SYNC_DERIV_PRE_FRAMES,
    sigma_mult: float = VIDEO_SYNC_DERIV_SIGMA_MULT,
    min_thresh: float = VIDEO_SYNC_DERIV_MIN_THRESH,
    envelope_frames: int = 25,
    progress: bool = True,
) -> OnsetDetectionResult:
    """Detect grating onset per trial via spatial-variance envelope step-up.

    The drifting grating creates *oscillating* spatial variance in the ROI
    (peaks when bars cross, troughs between bars).  During ITI the variance
    is flat and low.  A raw derivative detector misses onsets because the
    first variance peak depends on bar phase.

    **Solution**: apply a running max-filter (``envelope_frames`` ≈ 0.5 s)
    to the per-frame spatial variance.  This converts the oscillating grating
    signal into a sustained plateau (~4000) vs. the flat ITI (~1500–1800).
    Then detect the step-up in the envelope derivative, which is a clean
    single transition per trial.

    Returns an :class:`OnsetDetectionResult` with per-detection confidence.
    """
    import cv2
    from scipy.ndimage import maximum_filter1d

    ts_ms, _, _ = load_camera_metadata(metadata_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        roi_mask = _build_roi_mask(roi, frame_h, frame_w)
        n_trials = len(baseline_on_s)
        detected_cam_s = []
        detected_nidaq_s = []
        confidences = []
        n_miss = 0

        for trial_idx in range(n_trials):
            nidaq_t = baseline_on_s[trial_idx]
            cam_ms_est = (nidaq_t - rough_offset_s) * 1000.0
            center_frame = int(np.searchsorted(ts_ms, cam_ms_est))

            start_f = max(0, center_frame - search_frames)
            end_f = min(len(ts_ms) - 1, center_frame + search_frames)

            min_needed = pre_frames + envelope_frames + 5
            if end_f - start_f < min_needed:
                n_miss += 1
                continue

            # Read frames in the search window — compute spatial variance
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            svar = []
            ftimes = []
            for fi in range(start_f, end_f):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                svar.append(gray[roi_mask].var())
                ftimes.append(ts_ms[fi] / 1000.0)

            svar_arr = np.array(svar)
            ftimes_arr = np.array(ftimes)

            if len(svar_arr) < min_needed:
                n_miss += 1
                continue

            # Max-filter envelope: smooths out grating oscillations into
            # a sustained plateau while ITI stays flat
            envelope = maximum_filter1d(svar_arr, size=envelope_frames)

            # Derivative of the envelope
            deriv = np.diff(envelope)

            # Baseline noise from pre-onset envelope frames
            pre_deriv = deriv[:pre_frames]
            noise_median = np.median(pre_deriv)
            noise_mad = np.median(np.abs(pre_deriv - noise_median))
            thresh = noise_median + max(
                sigma_mult * _MAD_SCALE * noise_mad,
                min_thresh,
            )

            # Find first positive exceedance after the pre baseline
            # (envelope step-UP = grating onset)
            exceed_idx = np.where(deriv[pre_frames:] > thresh)[0]
            if len(exceed_idx) == 0:
                n_miss += 1
                continue

            onset_idx = exceed_idx[0] + pre_frames

            # Stage 2: refine using raw variance.  The max-filter smears
            # timing by up to half its window, so look back in the raw
            # signal to find the actual first variance rise.
            refine_start = max(0, onset_idx - envelope_frames)
            refine_end = min(len(svar_arr), onset_idx + 5)
            raw_deriv = np.diff(svar_arr[refine_start:refine_end])
            if len(raw_deriv) > 0:
                raw_med = np.median(raw_deriv)
                raw_mad = np.median(np.abs(raw_deriv - raw_med))
                raw_thresh = raw_med + max(
                    sigma_mult * _MAD_SCALE * raw_mad, 50.0
                )
                raw_exceed = np.where(raw_deriv > raw_thresh)[0]
                if len(raw_exceed) > 0:
                    refined_idx = raw_exceed[0] + refine_start
                else:
                    refined_idx = onset_idx
            else:
                refined_idx = onset_idx

            # Sub-frame interpolation at the refined position
            raw_d = np.diff(svar_arr)
            if refined_idx > 0 and refined_idx < len(raw_d):
                d_before = raw_d[refined_idx - 1]
                d_after = raw_d[refined_idx]
                r_thresh = raw_thresh if len(raw_deriv) > 0 else thresh
                if d_after > d_before and (d_after - d_before) > 1e-6:
                    frac = np.clip(
                        (r_thresh - d_before) / (d_after - d_before),
                        0.0, 1.0,
                    )
                    t_interp = (
                        ftimes_arr[refined_idx - 1]
                        + frac * (ftimes_arr[refined_idx]
                                  - ftimes_arr[refined_idx - 1])
                    )
                else:
                    t_interp = ftimes_arr[refined_idx]
            else:
                t_interp = ftimes_arr[min(refined_idx, len(ftimes_arr) - 1)]

            # Per-detection confidence: envelope step size / baseline SD
            baseline_env = envelope[:pre_frames]
            peak_env = np.max(
                envelope[refined_idx:min(refined_idx + envelope_frames,
                                         len(envelope))]
            )
            step = peak_env - np.median(baseline_env)
            baseline_sd = np.std(baseline_env)
            conf = step / max(baseline_sd, 1e-6)

            detected_cam_s.append(t_interp)
            detected_nidaq_s.append(nidaq_t)
            confidences.append(conf)

            if progress and (trial_idx + 1) % 100 == 0:
                logger.info(
                    f"  Onset detection: {trial_idx + 1}/{n_trials} trials processed"
                )
    finally:
        cap.release()

    logger.info(
        f"Onset detection (variance): {len(detected_cam_s)}/{n_trials} detected, "
        f"{n_miss} missed"
    )

    return OnsetDetectionResult(
        detected_cam_s=np.array(detected_cam_s),
        detected_nidaq_s=np.array(detected_nidaq_s),
        confidence=np.array(confidences),
        n_trials=n_trials,
    )


# =====================================================================
# Derivative-based onset detection (legacy method)
# =====================================================================


def detect_onsets_derivative(
    video_path: str,
    metadata_path: str,
    baseline_on_s: np.ndarray,
    rough_offset_s: float,
    roi: RoiSpec = VIDEO_SYNC_DEFAULT_EYE_ROI,
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
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        roi_mask = _build_roi_mask(roi, frame_h, frame_w)
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
                lum.append(gray[roi_mask].mean())
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


def fast_coarse_offset(
    video_path: str,
    metadata_path: str,
    nidaq_baseline_on_s: np.ndarray,
    target_fps: float = 2.0,
) -> float:
    """Estimate rough offset via ffmpeg continuous low-fps luminance extraction.

    Much faster than :func:`_coarse_offset_from_metadata` (which seeks per
    frame) because ffmpeg decodes the video once start-to-finish at a low
    frame rate.  Detects luminance transitions and cross-correlates with
    Baseline_ON events via :func:`coarse_align`.

    Parameters
    ----------
    video_path : str
        Path to the camera video file (MP4).
    metadata_path : str
        Path to camera metadata CSV.
    nidaq_baseline_on_s : np.ndarray
        NI-DAQ Baseline_ON times in seconds.
    target_fps : float
        Frame rate for ffmpeg extraction (default 2 fps).

    Returns
    -------
    float
        Estimated camera-to-NI-DAQ offset in seconds.
    """
    import subprocess

    ts_ms, _, _ = load_camera_metadata(metadata_path)
    total_duration_s = ts_ms[-1] / 1000.0

    out_w, out_h = 64, 64
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

    pixels = np.frombuffer(raw[: n_frames * frame_bytes], dtype=np.uint8)
    pixels = pixels.reshape(n_frames, out_h, out_w)
    lum = pixels.astype(np.float32).mean(axis=(1, 2))
    sample_times_s = np.arange(n_frames) / target_fps

    logger.info(f"  Got {n_frames} frames ({n_frames / target_fps:.0f}s covered)")

    deriv = np.abs(np.diff(lum))
    med_d = np.median(deriv)
    mad_d = np.median(np.abs(deriv - med_d))
    if mad_d < 1e-6:
        thresh = np.percentile(deriv, 97)
    else:
        thresh = med_d + 5.0 * _MAD_SCALE * mad_d

    peaks = np.where(deriv > thresh)[0]
    if len(peaks) == 0:
        logger.warning("Coarse scan: no transitions detected")
        return 0.0

    transition_times = _cluster_times(sample_times_s[peaks], cluster_s=2.0)
    logger.info(f"  {len(transition_times)} transitions detected")
    return coarse_align(transition_times, nidaq_baseline_on_s)


def load_or_compute_coarse_offset(
    session_name: str,
    video_path: str,
    metadata_path: str,
    nidaq_baseline_on_s: np.ndarray,
    cache_file: Optional[str] = None,
    use_ffmpeg: bool = True,
) -> float:
    """Load cached coarse offset or compute and cache it.

    Parameters
    ----------
    session_name : str
        Session identifier for cache key.
    video_path, metadata_path : str
        Paths to video and metadata files.
    nidaq_baseline_on_s : np.ndarray
        NI-DAQ Baseline_ON times in seconds.
    cache_file : str, optional
        JSON cache file path.  Defaults to ``VIDEO_SYNC_DIR/coarse_offsets.json``.
    use_ffmpeg : bool
        If True (default), use :func:`fast_coarse_offset` (ffmpeg-based).
        If False, fall back to :func:`_coarse_offset_from_metadata`.

    Returns
    -------
    float
        Coarse camera-to-NI-DAQ offset in seconds.
    """
    if cache_file is None:
        cache_file = os.path.join(VIDEO_SYNC_DIR, "coarse_offsets.json")

    sname = str(session_name)
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)
        if sname in cache:
            offset = cache[sname]
            logger.info(f"[{sname}] Using cached coarse offset = {offset:.2f}s")
            return offset

    if use_ffmpeg:
        offset = fast_coarse_offset(video_path, metadata_path, nidaq_baseline_on_s)
    else:
        offset = _coarse_offset_from_metadata(
            metadata_path, video_path, nidaq_baseline_on_s
        )

    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)
    cache[sname] = offset
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)

    return offset


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
    min_inliers = max(_MIN_ANCHORS_CV, int(0.3 * n_original))

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
# Anchor selection (confidence-based pre-filter)
# =====================================================================


def select_anchors(
    detection: OnsetDetectionResult,
    min_anchors: int = 30,
    confidence_percentile: float = 50.0,
    n_temporal_bins: int = 10,
) -> OnsetDetectionResult:
    """Pre-filter detections by confidence before clock model fitting.

    Strategy: threshold by confidence percentile, then ensure temporal
    spread by dividing the session into bins and keeping the top-K per bin.
    Progressive fallback: lower the percentile in steps of 10 until
    ``min_anchors`` are met.

    Complementary to ``fit_clock_model()``'s existing Theil-Sen + MAD
    outlier rejection.  This step removes obvious bad detections; the
    clock fitting handles the rest.

    Parameters
    ----------
    detection : OnsetDetectionResult
        Full detection output with confidence scores.
    min_anchors : int
        Minimum number of anchors to retain.
    confidence_percentile : float
        Initial percentile threshold (keep detections above this).
    n_temporal_bins : int
        Number of temporal bins for ensuring spread.

    Returns
    -------
    OnsetDetectionResult
        Filtered detection result with only selected anchors.
    """
    n = detection.n_detected
    if n <= min_anchors:
        return detection

    cam = detection.detected_cam_s
    nidaq = detection.detected_nidaq_s
    conf = detection.confidence

    # Progressive fallback: lower threshold until min_anchors met
    pct = confidence_percentile
    while pct > 0:
        thresh = np.percentile(conf, pct)
        keep = conf >= thresh

        if keep.sum() >= min_anchors:
            break
        pct -= 10.0

    if keep.sum() < min_anchors:
        # Not enough even at pct=0 — return all
        return detection

    # Ensure temporal spread: divide session into bins, keep top detections
    # per bin so clock model has support across the full session
    t_min, t_max = cam[keep].min(), cam[keep].max()
    if t_max - t_min < 1e-6:
        # All at same time — just return the threshold-filtered set
        return OnsetDetectionResult(
            detected_cam_s=cam[keep],
            detected_nidaq_s=nidaq[keep],
            confidence=conf[keep],
            n_trials=detection.n_trials,
        )

    bin_edges = np.linspace(t_min, t_max + 1e-6, n_temporal_bins + 1)
    selected = np.zeros(n, dtype=bool)

    for i in range(n_temporal_bins):
        in_bin = keep & (cam >= bin_edges[i]) & (cam < bin_edges[i + 1])
        if in_bin.sum() == 0:
            continue
        # Keep all above-threshold detections in this bin
        selected |= in_bin

    # If temporal binning somehow reduced below min_anchors, use the
    # simple threshold result instead
    if selected.sum() < min_anchors:
        selected = keep

    logger.info(
        f"  Anchor selection: {selected.sum()}/{n} detections retained "
        f"(confidence >= {thresh:.1f}, {n_temporal_bins} temporal bins)"
    )

    return OnsetDetectionResult(
        detected_cam_s=cam[selected],
        detected_nidaq_s=nidaq[selected],
        confidence=conf[selected],
        n_trials=detection.n_trials,
    )


# =====================================================================
# Conversion functions
# =====================================================================


# ---------------------------------------------------------------------------
# ORIENTATION CONTRACT (read before applying these converters)
# ---------------------------------------------------------------------------
# The ``slope``/``offset`` stored on a SyncResult are NOT a single fixed
# orientation — they depend on ``detection_method``:
#   * ``derivative`` / ``manual_multianchor``  -> store  nidaq = slope*cam + offset
#       (the orientation the two converters below ASSUME).
#   * legacy ``manual_slope_fit``               -> store the INVERSE
#       video = slope*nidaq + offset.
# Downstream consumers MUST branch on ``detection_method`` and invert
# (slope' = 1/slope, offset' = -offset/slope) for a ``manual_slope_fit`` result
# BEFORE passing slope/offset to camera_to_nidaq / nidaq_to_camera.
def camera_to_nidaq(t_camera_ms, slope: float, offset: float):
    """Convert camera timestamp(s) (ms) to NI-DAQ time (seconds).

    Assumes the ``nidaq = slope*cam + offset`` orientation (see ORIENTATION
    CONTRACT above): valid for ``derivative`` / ``manual_multianchor`` results.
    For a legacy ``manual_slope_fit`` result invert slope/offset first.
    """
    t = np.asarray(t_camera_ms, dtype=np.float64)
    return slope * (t / 1000.0) + offset


def nidaq_to_camera(t_nidaq_s, slope: float, offset: float):
    """Convert NI-DAQ time (seconds) to camera timestamp(s) (ms).

    Assumes the ``nidaq = slope*cam + offset`` orientation (see ORIENTATION
    CONTRACT above): valid for ``derivative`` / ``manual_multianchor`` results.
    For a legacy ``manual_slope_fit`` result invert slope/offset first.
    """
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
    session_name = str(int(session_name)).zfill(8)

    data = {"session_name": session_name}
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
    session_name = str(int(session_name)).zfill(8)
    path = os.path.join(out_dir, f"{session_name}_video_sync.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def archive_sync_artifacts(
    session_name: str,
    subject: Optional[str] = None,
    sync_dir: Optional[str] = None,
    when: Optional[str] = None,
    include_anchor: bool = True,
) -> Optional[str]:
    """Move existing sync (+ optionally anchor) JSONs into ``<sync_dir>/_archive/<when>/``.

    Called before a re-fit so a re-tag never silently clobbers a prior fit
    (spec migration policy). Returns the archive dir, or None if nothing moved.

    Parameters
    ----------
    include_anchor : bool
        When True (default), archive BOTH ``_video_sync.json`` and
        ``_anchor.json`` (the §3.14 migration semantics used by Plan 2's future
        tagger). When False, archive ONLY ``_video_sync.json`` and leave the
        live anchor in place — required so a re-fit (fit_sync) is repeatable
        without stranding the anchor it reads.
    """
    import shutil
    from visdetect.analysis.config import subject_video_sync_dir, canonical_camera_session
    out_dir = sync_dir or subject_video_sync_dir(subject)
    sn = canonical_camera_session(session_name)
    when = when or _dt.date.today().isoformat()
    moved = False
    arch = os.path.join(out_dir, "_archive", when)
    suffixes = ("_video_sync.json", "_anchor.json") if include_anchor else ("_video_sync.json",)
    for suffix in suffixes:
        src = os.path.join(out_dir, f"{sn}{suffix}")
        if os.path.exists(src):
            os.makedirs(arch, exist_ok=True)
            shutil.move(src, os.path.join(arch, f"{sn}{suffix}"))
            moved = True
    return arch if moved else None


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
# Corneal auto-calibration: pupil detection + data-driven mask
# =====================================================================


def detect_pupil_in_frame(
    gray_frame: np.ndarray,
    search_roi: Optional[Tuple[int, int, int, int]] = None,
    min_area: int = CORNEAL_CAL_PUPIL_MIN_AREA_PX,
    max_area: int = CORNEAL_CAL_PUPIL_MAX_AREA_PX,
    min_circularity: float = CORNEAL_CAL_PUPIL_MIN_CIRCULARITY,
    blur_sigma: int = 5,
    dark_percentile: float = 8.0,
) -> Optional[Dict]:
    """Detect the pupil in a grayscale eye-camera frame.

    The pupil is the darkest large circular blob in the frame.  Detection
    uses a dark-pixel threshold derived from the frame's intensity histogram,
    followed by morphological closing and contour filtering.

    Parameters
    ----------
    gray_frame : np.ndarray, shape (H, W), dtype uint8 or float32
        Full-resolution grayscale frame.
    search_roi : (y0, y1, x0, x1), optional
        Restrict the search to this rectangular region.  Coordinates are in
        full-frame pixels.  If None, the whole frame is searched.
    min_area, max_area : int
        Contour area bounds for valid pupil candidates (px²).
    min_circularity : float
        Minimum 4π·area/perimeter² ratio (0–1).  Circles = 1.
    blur_sigma : int
        Gaussian blur radius before thresholding.
    dark_percentile : float
        Threshold = this percentile of pixel intensities in the ROI.

    Returns
    -------
    dict with keys ``center_y``, ``center_x``, ``radius``, ``area``,
    ``circularity``, ``bbox`` (y0, y1, x0, x1 of contour bounding box)
    in **full-frame** coordinates, or ``None`` if no valid pupil found.
    """
    import cv2

    frame = np.asarray(gray_frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    # Crop to search region
    offset_y, offset_x = 0, 0
    if search_roi is not None:
        y0, y1, x0, x1 = search_roi
        frame = frame[y0:y1, x0:x1]
        offset_y, offset_x = y0, x0

    # Gaussian blur
    ksize = blur_sigma * 2 + 1
    blurred = cv2.GaussianBlur(frame, (ksize, ksize), blur_sigma)

    # Threshold: darkest pixels (pupil is dark)
    thresh_val = np.percentile(blurred, dark_percentile)
    thresh_val = max(1.0, min(thresh_val, 200.0))  # sanity bounds
    _, binary = cv2.threshold(
        blurred, thresh_val, 255, cv2.THRESH_BINARY_INV
    )

    # Morphological closing to fill the pupil interior
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    # Filter and score candidates
    best = None
    best_score = -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1e-3:
            continue
        circularity = 4.0 * np.pi * area / (perimeter ** 2)
        if circularity < min_circularity:
            continue

        # Score: prefer large and circular
        score = circularity * np.sqrt(area)
        if score > best_score:
            best_score = score
            best = (cnt, area, circularity)

    if best is None:
        return None

    cnt, area, circularity = best

    # Fit ellipse for precise center + axes
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        cx_local = float(ellipse[0][0])
        cy_local = float(ellipse[0][1])
        axes = ellipse[1]
        radius = float(max(axes)) / 2.0
    else:
        M = cv2.moments(cnt)
        if M["m00"] < 1e-6:
            return None
        cx_local = float(M["m10"] / M["m00"])
        cy_local = float(M["m01"] / M["m00"])
        radius = float(np.sqrt(area / np.pi))

    # Bounding box
    bx, by, bw, bh = cv2.boundingRect(cnt)

    return {
        "center_y": cy_local + offset_y,
        "center_x": cx_local + offset_x,
        "radius": radius,
        "area": float(area),
        "circularity": float(circularity),
        "bbox": (by + offset_y, by + bh + offset_y,
                 bx + offset_x, bx + bw + offset_x),
    }


def build_corneal_mask(
    video_path: str,
    metadata_path: str,
    baseline_on_s: np.ndarray,
    rough_offset_s: float,
    pupil_cy: float,
    pupil_cx: float,
    pupil_radius: float,
    frame_h: int,
    frame_w: int,
    search_margin: int = CORNEAL_CAL_SEARCH_MARGIN_PX,
    pupil_exclusion_factor: float = CORNEAL_CAL_PUPIL_EXCLUSION_FACTOR,
    angle_min_deg: float = CORNEAL_CAL_ANGLE_MIN_DEG,
    angle_max_deg: float = CORNEAL_CAL_ANGLE_MAX_DEG,
    max_dist_px: int = CORNEAL_CAL_MAX_DIST_PX,
    n_transitions: int = CORNEAL_CAL_N_TRANSITIONS,
    pre_frames: int = CORNEAL_CAL_PRE_FRAMES,
    post_frames: int = CORNEAL_CAL_POST_FRAMES,
    min_area: int = CORNEAL_CAL_MASK_MIN_AREA_PX,
    max_area: int = CORNEAL_CAL_MASK_MAX_AREA_PX,
    threshold_pct: int = CORNEAL_CAL_MASK_THRESHOLD_PCT,
    cache_dir: Optional[str] = None,
    session_name: Optional[str] = None,
    force: bool = False,
) -> Tuple[np.ndarray, dict]:
    """Build a data-driven binary mask for the corneal grating reflection.

    Uses the same std(post-pre diff) logic as :func:`build_screen_mask` but
    applied to the eye region, with two biological constraints:

    1. **Pupil exclusion** — pixels within ``pupil_radius × exclusion_factor``
       of the pupil centre are zeroed out (pupil shows no grating signal).

    2. **Angular wedge (lower-left prior)** — the grating reflection always
       appears LOWER and SLIGHTLY LEFT of the pupil centre in the eye camera.
       This is a rig-geometry constraint (screen is below-left of the animal)
       and holds across sessions and subjects.  The wedge is defined by
       ``angle_min_deg``–``angle_max_deg`` in image coordinates where 0°=right
       and 90°=downward (y-axis increases toward the bottom of the frame).
       Default [90°, 185°] restricts to the lower-left quadrant.

    3. **Maximum distance cap** — pixels further than ``max_dist_px`` from the
       pupil centre are excluded.  The corneal reflection is always ON the
       cornea, typically within 20–50 px of the pupil.  This prevents picking
       up distant fur/skin movement artefacts.

    This approach is **eye-cam specific**.  Do NOT apply to the front cam.

    Parameters
    ----------
    video_path, metadata_path : str
        Paths to the eye-camera video and metadata CSV.
    baseline_on_s : np.ndarray
        NI-DAQ Baseline_ON times in seconds.
    rough_offset_s : float
        Coarse camera-to-NI-DAQ offset estimate (seconds).
    pupil_cy, pupil_cx : float
        Pupil centre in full-frame pixel coordinates.
    pupil_radius : float
        Effective pupil radius (px).
    frame_h, frame_w : int
        Full-frame dimensions.
    search_margin : int
        Number of pixels to extend the bounding box around the pupil centre.
    pupil_exclusion_factor : float
        Exclude pixels within ``pupil_radius × this`` of the pupil centre.
    angle_min_deg, angle_max_deg : float
        Angular wedge bounds (degrees).  0°=right, 90°=down (image coords).
    max_dist_px : int
        Maximum distance (px) from pupil centre for candidate pixels.
        The corneal reflection is always on the cornea, not distant fur/skin.
    n_transitions : int
        Number of Baseline_ON transitions to sample.
    pre_frames, post_frames : int
        Frames averaged before/after each transition.
    min_area, max_area : int
        Connected-component area bounds for the corneal reflection (px²).
    threshold_pct : int
        Percentile cutoff within the search region (default 80 = top 20%).
        More robust than Otsu for diffuse, non-bimodal corneal signals.
    cache_dir : str, optional
        Cache directory.  Defaults to ``VIDEO_SYNC_DIR/corneal_masks``.
    session_name : str, optional
        Used for cache file naming.
    force : bool
        Recompute even if cache exists.

    Returns
    -------
    mask : np.ndarray, shape (frame_h, frame_w), dtype bool
        Full-frame binary mask of the corneal reflection.
    info : dict
        Diagnostic metadata (threshold, n_transitions_used, mask_area,
        best_component_mean_std, avg_diff image cropped to search box).
    """
    import cv2
    from scipy import ndimage as ndi

    # ── Cache check ────────────────────────────────────────────────────
    if cache_dir is None:
        cache_dir = os.path.join(VIDEO_SYNC_DIR, "corneal_masks")
    if session_name and not force:
        cache_path = os.path.join(cache_dir, f"{session_name}_corneal_mask.npz")
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            mask = data["mask"]
            info = data["info"].item()
            logger.info(
                f"  Loaded cached corneal mask: {mask.sum()} px "
                f"({mask.mean():.3%} of frame)"
            )
            return mask, info

    ts_ms, _, _ = load_camera_metadata(metadata_path)

    # ── Bounding box: generous window around pupil ─────────────────────
    bb_y0 = max(0, int(pupil_cy) - search_margin)
    bb_y1 = min(frame_h, int(pupil_cy) + search_margin)
    bb_x0 = max(0, int(pupil_cx) - search_margin)
    bb_x1 = min(frame_w, int(pupil_cx) + search_margin)
    bb_h = bb_y1 - bb_y0
    bb_w = bb_x1 - bb_x0

    # ── Constraint masks (computed in full-frame coords, then cropped) ──
    yy, xx = np.mgrid[bb_y0:bb_y1, bb_x0:bb_x1].astype(np.float32)

    dy = yy - pupil_cy   # positive = below pupil (y increases downward)
    dx = xx - pupil_cx   # positive = right of pupil

    # Pupil exclusion: circular region around pupil centre.
    # Cap at half of max_dist_px so the exclusion zone can never consume the
    # entire search annulus (protects against over-estimated pupil radius when
    # the iris blob is mistakenly detected as the pupil).
    dist2 = dy ** 2 + dx ** 2
    pupil_excl_radius = min(
        pupil_radius * pupil_exclusion_factor,
        max_dist_px * 0.5,
    )
    pupil_mask = dist2 > (pupil_excl_radius ** 2)

    # Angular wedge: lower-left of pupil
    # atan2(dy, dx): 0°=right, 90°=down, 180°=left, 270°=up
    angle_deg = np.degrees(np.arctan2(dy, dx))           # [-180, 180]
    angle_deg = (angle_deg + 360.0) % 360.0              # [0, 360]
    if angle_min_deg <= angle_max_deg:
        angle_mask = (angle_deg >= angle_min_deg) & (angle_deg <= angle_max_deg)
    else:
        # Wraps around 360° (e.g. [300, 60])
        angle_mask = (angle_deg >= angle_min_deg) | (angle_deg <= angle_max_deg)

    # Maximum distance: corneal reflection is ON the cornea, never far from pupil
    dist_mask = dist2 <= (float(max_dist_px) ** 2)

    search_mask = pupil_mask & angle_mask & dist_mask  # (bb_h, bb_w), bool

    if search_mask.sum() < 10:
        logger.warning(
            "build_corneal_mask: search mask has fewer than 10 pixels after "
            "pupil exclusion + angle wedge + distance cap — widening constraints"
        )
        # Fallback: skip angle and distance constraints, keep only pupil exclusion
        search_mask = pupil_mask

    # ── Sample transitions ─────────────────────────────────────────────
    n_avail = len(baseline_on_s)
    n_use = min(n_transitions, n_avail)
    if n_use < 5:
        raise ValueError(
            f"Too few Baseline_ON events ({n_avail}) to build corneal mask"
        )
    indices = np.round(np.linspace(0, n_avail - 1, n_use)).astype(int)
    selected_nidaq = baseline_on_s[indices]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    diff_images = []
    try:
        for nidaq_t in selected_nidaq:
            cam_ms = (nidaq_t - rough_offset_s) * 1000.0
            center_frame = int(np.searchsorted(ts_ms, cam_ms))

            start_f = center_frame - pre_frames
            end_f = center_frame + post_frames
            if start_f < 0 or end_f >= len(ts_ms):
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            pre_sum = np.zeros((bb_h, bb_w), dtype=np.float64)
            pre_count = 0
            for _ in range(pre_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
                pre_sum += gray[bb_y0:bb_y1, bb_x0:bb_x1]
                pre_count += 1

            post_sum = np.zeros((bb_h, bb_w), dtype=np.float64)
            post_count = 0
            for _ in range(post_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
                post_sum += gray[bb_y0:bb_y1, bb_x0:bb_x1]
                post_count += 1

            if pre_count == 0 or post_count == 0:
                continue

            diff_images.append(post_sum / post_count - pre_sum / pre_count)
    finally:
        cap.release()

    n_used = len(diff_images)
    if n_used < 5:
        raise RuntimeError(
            f"Only {n_used} valid transitions — need ≥ 5 for std computation"
        )

    diff_stack = np.stack(diff_images, axis=0)   # (n_used, bb_h, bb_w)
    avg_diff = np.std(diff_stack, axis=0)         # high where phase varies

    # ── Apply search mask and threshold ───────────────────────────────
    std_masked = avg_diff.copy()
    std_masked[~search_mask] = 0.0

    # Top-N% percentile threshold — more robust than Otsu for the diffuse
    # corneal reflection signal (moderate, spread std values; not bimodal).
    valid_vals = std_masked[search_mask]
    if valid_vals.max() < 1e-6:
        raise RuntimeError("All std values in search region are zero")

    thresh = np.percentile(valid_vals, threshold_pct)
    local_mask = (std_masked >= thresh) & search_mask

    # ── Connected component filtering ─────────────────────────────────
    labeled, n_comp = ndi.label(local_mask)

    best_comp_id = None
    best_comp_score = -1.0
    for comp_id in range(1, n_comp + 1):
        comp = labeled == comp_id
        comp_area = int(comp.sum())
        if comp_area < min_area or comp_area > max_area:
            continue
        # Score: mean std value within component (strongest grating signal)
        comp_score = float(avg_diff[comp].mean())
        if comp_score > best_comp_score:
            best_comp_score = comp_score
            best_comp_id = comp_id

    if best_comp_id is None:
        logger.warning(
            "build_corneal_mask: no component with area in "
            f"[{min_area}, {max_area}] px² — returning empty mask. "
            "Consider widening area bounds or adjusting angle range."
        )
        local_mask = np.zeros((bb_h, bb_w), dtype=bool)
        best_comp_score = 0.0
    else:
        local_mask = labeled == best_comp_id

    # ── Place local mask into full-frame coordinates ───────────────────
    full_mask = np.zeros((frame_h, frame_w), dtype=bool)
    full_mask[bb_y0:bb_y1, bb_x0:bb_x1] = local_mask

    mask_area = int(full_mask.sum())
    logger.info(
        f"  Corneal mask: {mask_area} px, best_std={best_comp_score:.2f}, "
        f"n_transitions={n_used}, search_mask={search_mask.sum()} px"
    )

    info = {
        "threshold_pct": int(threshold_pct),
        "threshold_value": float(thresh),
        "n_transitions_used": int(n_used),
        "mask_area_px": mask_area,
        "best_component_mean_std": float(best_comp_score),
        "search_mask_area_px": int(search_mask.sum()),
        "pupil_cy": float(pupil_cy),
        "pupil_cx": float(pupil_cx),
        "pupil_radius": float(pupil_radius),
        "bb": (int(bb_y0), int(bb_y1), int(bb_x0), int(bb_x1)),
        "avg_diff": avg_diff.astype(np.float32),
    }

    # ── Cache ──────────────────────────────────────────────────────────
    if session_name:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{session_name}_corneal_mask.npz")
        np.savez_compressed(
            cache_path, mask=full_mask, info=np.array(info, dtype=object)
        )
        logger.info(f"  Cached corneal mask → {cache_path}")

    return full_mask, info


def auto_calibrate_corneal_roi(
    session_name: str,
    video_path: str,
    metadata_path: str,
    baseline_on_s: np.ndarray,
    rough_offset_s: float,
    broad_eye_roi: Optional[Tuple[int, int, int, int]] = None,
    cache_dir: Optional[str] = None,
    force: bool = False,
) -> Optional[Dict]:
    """Auto-calibrate the corneal grating reflection mask for one session.

    Orchestrates:
    1. Pupil detection on a representative frame (:func:`detect_pupil_in_frame`)
    2. Data-driven corneal mask (:func:`build_corneal_mask`)

    The result is cached as JSON (parameters) + NPZ (binary mask) and can
    be reloaded via :func:`load_corneal_cal`.

    Parameters
    ----------
    session_name : str
        Session identifier (used for cache naming).
    video_path, metadata_path : str
        Eye camera video + metadata CSV paths.
    baseline_on_s : np.ndarray
        NI-DAQ Baseline_ON times in seconds.
    rough_offset_s : float
        Coarse camera-to-NI-DAQ offset (seconds).
    broad_eye_roi : (y0, y1, x0, x1), optional
        Loose bounding box restricting pupil search.  If None, the whole
        frame is searched.  A generous per-subject box (e.g. 300×300 px)
        is all that is needed — much coarser than the per-session tight ROIs
        it replaces.
    cache_dir : str, optional
        Directory for JSON + NPZ caches.  Defaults to
        ``VIDEO_SYNC_DIR/corneal_cal``.
    force : bool
        Recompute even if cached result exists.

    Returns
    -------
    dict with keys:
        ``pupil_center`` (cy, cx), ``pupil_radius``,
        ``mask_area_px``, ``mask_quality`` ("good"/"marginal"/"failed"),
        ``best_component_mean_std``, ``corneal_bbox`` (y0, y1, x0, x1),
        ``session_name``.
    Returns ``None`` if pupil detection fails.
    """
    import cv2

    sname = str(session_name)
    _cache_dir = cache_dir or os.path.join(VIDEO_SYNC_DIR, "corneal_cal")
    os.makedirs(_cache_dir, exist_ok=True)
    json_path = os.path.join(_cache_dir, f"{sname}_corneal_roi_cal.json")

    if not force and os.path.exists(json_path):
        with open(json_path) as f:
            cal = json.load(f)
        logger.info(
            f"[{sname}] Loaded cached corneal calibration: "
            f"pupil=({cal['pupil_center'][0]:.1f},{cal['pupil_center'][1]:.1f}) "
            f"r={cal['pupil_radius']:.1f}, mask={cal['mask_area_px']}px "
            f"({cal['mask_quality']})"
        )
        return cal

    # ── Step 1: extract a representative frame ─────────────────────────
    # Use a trial ~30% into the session (avoiding warmup / end-of-session drift)
    ts_ms, _, _ = load_camera_metadata(metadata_path)
    target_nidaq = float(np.percentile(baseline_on_s, 30))
    target_cam_ms = (target_nidaq - rough_offset_s) * 1000.0
    frame_idx = int(np.searchsorted(ts_ms, target_cam_ms))
    frame_idx = max(0, min(frame_idx, len(ts_ms) - 1))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, bgr = cap.read()
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    if not ret:
        logger.error(f"[{sname}] Could not read representative frame")
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # ── Step 2: detect pupil ───────────────────────────────────────────
    pupil = detect_pupil_in_frame(gray, search_roi=broad_eye_roi)
    if pupil is None:
        logger.warning(
            f"[{sname}] Pupil not detected in representative frame. "
            "Try specifying broad_eye_roi or check that the frame is valid."
        )
        return None

    pupil_cy = pupil["center_y"]
    pupil_cx = pupil["center_x"]
    pupil_radius = pupil["radius"]
    logger.info(
        f"[{sname}] Pupil detected: centre=({pupil_cy:.1f},{pupil_cx:.1f}) "
        f"r={pupil_radius:.1f}px area={pupil['area']:.0f}px² "
        f"circ={pupil['circularity']:.2f}"
    )

    # ── Step 3: build corneal reflection mask ──────────────────────────
    try:
        mask, mask_info = build_corneal_mask(
            video_path, metadata_path, baseline_on_s, rough_offset_s,
            pupil_cy=pupil_cy,
            pupil_cx=pupil_cx,
            pupil_radius=pupil_radius,
            frame_h=frame_h,
            frame_w=frame_w,
            session_name=sname,
            cache_dir=_cache_dir,
            force=force,
        )
    except Exception as exc:
        logger.error(f"[{sname}] build_corneal_mask failed: {exc}")
        return None

    # ── Step 4: assess mask quality ────────────────────────────────────
    mask_area = mask_info["mask_area_px"]
    mean_std = mask_info["best_component_mean_std"]

    if mask_area == 0:
        quality = "failed"
    elif mask_area < 10 or mean_std < 1.0:
        quality = "marginal"
    else:
        quality = "good"

    # Tight bounding box of the mask
    ys, xs = np.where(mask)
    if len(ys) > 0:
        corneal_bbox = (int(ys.min()), int(ys.max() + 1),
                        int(xs.min()), int(xs.max() + 1))
    else:
        corneal_bbox = None

    cal = {
        "session_name": sname,
        "pupil_center": [float(pupil_cy), float(pupil_cx)],
        "pupil_radius": float(pupil_radius),
        "pupil_area_px": float(pupil["area"]),
        "pupil_circularity": float(pupil["circularity"]),
        "mask_area_px": mask_area,
        "mask_quality": quality,
        "best_component_mean_std": float(mean_std),
        "corneal_bbox": list(corneal_bbox) if corneal_bbox else None,
        "broad_eye_roi": list(broad_eye_roi) if broad_eye_roi else None,
        "n_transitions_used": mask_info["n_transitions_used"],
        "search_mask_area_px": mask_info["search_mask_area_px"],
    }

    with open(json_path, "w") as f:
        json.dump(cal, f, indent=2)
    logger.info(
        f"[{sname}] Corneal calibration saved → {json_path} "
        f"(quality={quality}, mask={mask_area}px, mean_std={mean_std:.2f})"
    )
    return cal


def load_corneal_cal(
    session_name: str,
    cache_dir: Optional[str] = None,
) -> Optional[Dict]:
    """Load cached corneal calibration result for a session.

    Returns the JSON dict from :func:`auto_calibrate_corneal_roi`, or
    ``None`` if no cache exists.
    """
    _cache_dir = cache_dir or os.path.join(VIDEO_SYNC_DIR, "corneal_cal")
    path = os.path.join(_cache_dir, f"{session_name}_corneal_roi_cal.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_corneal_mask(
    session_name: str,
    cache_dir: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Load the cached binary corneal mask for a session.

    Returns a (H, W) bool array, or ``None`` if not cached.
    """
    _cache_dir = cache_dir or os.path.join(VIDEO_SYNC_DIR, "corneal_cal")
    path = os.path.join(_cache_dir, f"{session_name}_corneal_mask.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    return data["mask"]


# =====================================================================
# High-level session sync pipeline
# =====================================================================


def sync_session(
    session_name: str,
    nidaq_baseline_on_s: np.ndarray,
    camera_root: Optional[str] = None,
    subject: str = "BG_046",
    roi: Optional[RoiSpec] = None,
    sync_dir: Optional[str] = None,
    fig_dir: Optional[str] = None,
    force: bool = False,
    progress: bool = True,
    detection_mode: str = "fixed",
) -> dict:
    """End-to-end sync pipeline for one session.

    1. Find camera files
    2. Estimate rough offset via sparse full-frame luminance scan
    3. Per-trial spatial-variance onset detection with sub-frame interpolation
    4. Theil-Sen fit with iterative outlier rejection
    5. Temporal-block cross-validation
    6. Save sync JSON sidecar + diagnostic figure

    Parameters
    ----------
    detection_mode : str
        ROI selection strategy for onset detection:
        - ``"fixed"`` (default) — use explicit ``roi`` or
          :data:`VIDEO_SYNC_DEFAULT_EYE_ROI`.  Backward compatible.
        - ``"mask"`` — build a data-driven screen mask via
          :func:`build_screen_mask`.  Raises on failure.
        - ``"auto"`` — try mask-based detection first, fall back to
          fixed ROI if mask construction fails.

    Returns the sync dict (same as ``load_video_sync()`` output).
    """
    session_name = str(int(session_name)).zfill(8)

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

        # Step 2: Optionally build data-driven mask
        detection_method_used = "variance"
        if detection_mode in ("auto", "mask") and roi is None:
            try:
                mask, mask_info = build_screen_mask(
                    video_path, meta_path,
                    nidaq_baseline_on_s, rough_offset,
                    session_name=session_name,
                )
                used_roi = mask
                detection_method_used = "variance_mask"
                logger.info(
                    f"  Using data-driven mask: {mask_info['n_screen_pixels']} "
                    f"pixels ({mask_info['screen_fraction']:.1%})"
                )
            except Exception as exc:
                if detection_mode == "mask":
                    raise
                logger.warning(
                    f"  Mask construction failed ({exc}), "
                    f"falling back to fixed ROI"
                )

        # Step 3: Per-trial spatial variance onset detection
        detection = detect_onsets_variance(
            video_path, meta_path, nidaq_baseline_on_s,
            rough_offset_s=rough_offset,
            roi=used_roi,
            progress=progress,
        )

        # Optional anchor selection for degraded sessions
        if detection.detection_rate < 0.70 and detection.n_detected > _MIN_ANCHORS_FIT:
            logger.info(
                f"  Detection rate {detection.detection_rate:.0%} < 70%, "
                f"applying confidence-based anchor selection"
            )
            detection = select_anchors(detection)

        # Step 4: Fit clock model
        sync_result = fit_clock_model(
            detection.detected_cam_s, detection.detected_nidaq_s,
            n_baseline_on=len(nidaq_baseline_on_s),
        )
        sync_result.roi = _roi_to_json(used_roi) if used_roi else None
        sync_result.detection_method = detection_method_used

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


# =====================================================================
# Anchor JSON helpers (Phase 2: list-of-anchors schema, v1 read compat)
# =====================================================================


def _anchor_path(session_name: str, sync_dir: Optional[str] = None) -> str:
    """Path to the anchor JSON for *session_name*."""
    out_dir = sync_dir or VIDEO_SYNC_DIR
    session_name = str(int(session_name)).zfill(8)
    return os.path.join(out_dir, f"{session_name}_anchor.json")


def _migrate_anchor_v1_to_v2(d: dict) -> dict:
    """Convert a Phase 1 (single-anchor) JSON dict to the Phase 2 (list) shape.

    Idempotent: passing a v2 dict returns it unchanged.
    """
    if d.get("schema_version") == 2 or "anchors" in d:
        return d
    entry = {
        "trial_index": int(d["anchor_trial_index"]),
        "nidaq_baseline_on_s": float(d["nidaq_baseline_on_s"]),
        "video_frame_idx": int(d["video_frame_idx"]),
        "video_time_s": float(d["video_time_s"]),
        "clicked_at": str(d["clicked_at"]),
    }
    return {
        "session": str(d["session"]),
        "schema_version": 2,
        "frame_rate_fps": float(d["frame_rate_fps"]),
        "n_trials": int(d["n_trials"]),
        "anchors": [entry],
    }


def _migrate_anchor_to_v3(d: dict) -> dict:
    """Add event_type/nidaq_event_s to v2 baseline-only anchors (idempotent)."""
    d = _migrate_anchor_v1_to_v2(d)
    if d.get("schema_version") == 3:
        return d
    # Copy each entry so a passed-in v2 dict's caller entries are not mutated
    # in place. Only derive nidaq_event_s from nidaq_baseline_on_s when that
    # key is present (a change-type entry may lack it -> avoid float(None)).
    new_anchors = []
    for a in d["anchors"]:
        a = dict(a)
        a.setdefault("event_type", "baseline_on")
        if "nidaq_event_s" not in a and "nidaq_baseline_on_s" in a:
            a["nidaq_event_s"] = float(a["nidaq_baseline_on_s"])
        new_anchors.append(a)
    d = dict(d)
    d["anchors"] = new_anchors
    d["schema_version"] = 3
    return d


def compute_implied_offset(anchor: dict) -> float:
    """Return ``video_time_s - nidaq_baseline_on_s`` for a single anchor entry.

    Used by HUDs and reports that want to display "the camera started this
    many seconds after NI-DAQ" in a human-readable form.
    """
    return float(anchor["video_time_s"]) - float(anchor["nidaq_baseline_on_s"])


def _build_anchor_entry(
    baseline_on: np.ndarray,
    ts_ms: np.ndarray,
    trial_index: int,
    frame_idx: int,
) -> dict:
    """Build a single v2 anchor entry from a clicked frame index."""
    fi = int(frame_idx)
    return {
        "trial_index": int(trial_index),
        "nidaq_baseline_on_s": float(baseline_on[int(trial_index)]),
        "video_frame_idx": fi,
        "video_time_s": float(ts_ms[fi] / 1000.0),
        "clicked_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }


def _build_change_anchor_entry(
    change_on_s: float,
    ts_ms: np.ndarray,
    trial_index: int,
    frame_idx: int,
    change_size: float,
    outcome: str,
) -> dict:
    """Build a v3 change-onset anchor entry from a clicked frame index."""
    fi = int(frame_idx)
    return {
        "trial_index": int(trial_index),
        "event_type": "change_on",
        "nidaq_event_s": float(change_on_s),
        "change_size": float(change_size),
        "outcome": str(outcome),
        "video_frame_idx": fi,
        "video_time_s": float(ts_ms[fi] / 1000.0),
        "clicked_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }


def _build_v2_anchor_file(
    session_name: str,
    fps: float,
    n_trials: int,
    anchor_entries: list,
) -> dict:
    """Construct the top-level v2 anchor JSON dict."""
    return {
        "session": str(session_name),
        "schema_version": 2,
        "frame_rate_fps": float(fps),
        "n_trials": int(n_trials),
        "anchors": list(anchor_entries),
    }


def _merge_anchor_into_file(base: dict, new_entry: dict) -> dict:
    """Return a copy of *base* with *new_entry* merged into its anchors list.

    Replaces an existing anchor with the same ``(trial_index, event_type)``
    (default event_type ``baseline_on`` for legacy entries). Sorted by
    ``(trial_index, event_type)``.
    """
    def key(a):
        return (int(a["trial_index"]), a.get("event_type", "baseline_on"))
    nk = key(new_entry)
    kept = [a for a in base["anchors"] if key(a) != nk]
    kept.append(new_entry)
    kept.sort(key=key)
    out = dict(base)
    out["anchors"] = kept
    return out


def save_anchor(
    session_name: str,
    anchor: dict,
    sync_dir: Optional[str] = None,
) -> None:
    """Write *anchor* (v2 schema) to ``{sync_dir}/{session_name}_anchor.json``.

    Callers must pass a v2 dict; building one is the responsibility of
    :func:`_build_v2_anchor_file` (top-level) plus :func:`_build_anchor_entry`
    (per-anchor) plus :func:`_merge_anchor_into_file` (composition).
    Overwrites any existing file. Creates the directory if needed.
    """
    out_dir = sync_dir or VIDEO_SYNC_DIR
    os.makedirs(out_dir, exist_ok=True)
    with open(_anchor_path(session_name, sync_dir=sync_dir), "w") as f:
        json.dump(anchor, f, indent=2)


def load_anchor(
    session_name: str,
    sync_dir: Optional[str] = None,
) -> Optional[dict]:
    """Read the anchor JSON for *session_name* and return it in v3 form.

    Legacy v1/v2 JSONs are migrated in memory (the on-disk file is NOT
    rewritten by this read; it gets rewritten next time :func:`save_anchor`
    is called). Returns ``None`` if no file exists.
    """
    path = _anchor_path(session_name, sync_dir=sync_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        raw = json.load(f)
    return _migrate_anchor_to_v3(raw)


def compute_predicted_frame_idx(
    baseline_on_s: float,
    coarse_offset_s: float,
    ts_ms: np.ndarray,
) -> int:
    """Map a NI-DAQ Baseline_ON time to the nearest video frame index.

    Parameters
    ----------
    baseline_on_s
        NI-DAQ time of the event, in seconds.
    coarse_offset_s
        Seconds elapsed in NI-DAQ clock before video recording started.
    ts_ms
        Camera-frame timestamps in milliseconds, relative to video start.
        Typically returned by :func:`load_camera_metadata`.

    Returns
    -------
    int
        Index of the closest frame in ``ts_ms``. Clamped to ``[0, len(ts_ms) - 1]``.
    """
    video_ms = (baseline_on_s - coarse_offset_s) * 1000.0
    if video_ms <= ts_ms[0]:
        return 0
    if video_ms >= ts_ms[-1]:
        return int(len(ts_ms) - 1)
    return int(np.argmin(np.abs(ts_ms - video_ms)))
