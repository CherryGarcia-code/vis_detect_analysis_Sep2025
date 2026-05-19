"""POC: Multi-Anchor Video Sync — Lick Barcode + Corneal Reflection.

Validates two complementary anchor sources for camera↔NI-DAQ synchronization
that work independently of screen-glow signal strength:
  Phase A — Lick motion energy detection on a known-good session (09092025)
  Phase B — Lick-barcode sync rescue on a failed session (27062025)
  Phase C — Corneal reflection onset detection on the good session

Produces:
  - ROI preview frames: figures/video_sync/multianchor_poc/{session}_roi_preview.png
  - 3×3 diagnostic figure: figures/video_sync/multianchor_poc/multianchor_poc_*.png

Usage:
    py scripts/video/poc_multianchor_sync.py
    py scripts/video/poc_multianchor_sync.py --session-a 09092025 --session-b 27062025
"""

import os
import sys
import gc
import argparse
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import median_abs_deviation
from sklearn.mixture import GaussianMixture

# ── Project paths ────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

from visdetect.suite.loader import load_session
from src.visdetect.core.video_sync import (
    find_camera_files,
    load_camera_metadata,
    detect_onsets_variance,
    fit_clock_model,
    nidaq_to_camera,
    load_or_compute_coarse_offset,
    build_screen_mask,
)
from src.visdetect.analysis.constants import LICK_HARDWARE_DELAY_MS
from src.visdetect.analysis.config import VIDEO_SYNC_DIR, VIDEO_SYNC_FIG_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Script constants ─────────────────────────────────────────────
LICK_WINDOW_S = 1.0                # ±1s window around lick time (paper: 2s centred)
PASS1_WINDOW_S = 2.0               # wider window for Pass 1 (absorbs coarse offset error)
PRELIM_MODEL_MAX_RMSE_MS = 800.0   # guard: skip Pass 2 if preliminary model RMSE exceeds this
MIN_PASS1_ANCHORS = 10             # minimum anchors for fit_clock_model (matches library _MIN_ANCHORS_FIT)
GMM_N_COMPONENTS = 3               # Gaussian mixture components (paper)
GMM_THRESHOLD_N_SD = 2.0           # threshold = mean + N*SD of noise Gaussian (paper)
GMM_N_SAMPLE_LICKS = 100           # max licks used to fit the session GMM
SCREEN_FAIL_RMSE_MS = 40.0         # screen-glow RMSE above which lick-only is reported as primary
CORNEAL_WINDOW_S = 0.5             # ±0.5s around Baseline_ON
CORNEAL_SLIDING_WINDOW = 5         # frames for temporal variance
CORNEAL_EYE_ROI = {                # (y0, y1, x0, x1) — per-session, verified from frames
    "09092025": (305, 335, 425, 460),  # tight: corneal reflection region below pupil
    "27062025": (170, 320, 325, 495),
}
MOUTH_ROI = (660, 960, 400, 750)   # (y0, y1, x0, x1) — corrected mouth/jaw region

LOCAL_VIDEO_DIR = os.path.join(_PROJECT_ROOT, "data", "videos")
FIG_DIR = os.path.join(_PROJECT_ROOT, "figures", "video_sync", "multianchor_poc")

# Session name → local video filename mapping (DDMMYYYY → DDMMYY)
SESSION_VIDEO_MAP = {
    "09092025": "BG_046_090925_Eye_cam.mp4",
    "27062025": "BG_046_270625_Eye_cam.mp4",
}


# ═══════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════

def _get_lick_times(sess, shift_ms):
    """Get absolute lick times for Hit and FA trials.

    Reuses pattern from batch_sync_sessions.py:70-102.

    Returns list of (trial_index, outcome, lick_time_s) sorted chronologically.
    """
    baseline_on = np.asarray(
        sess.ni_events.get("Baseline_ON", []), dtype=float
    ).flatten()
    change_on = np.asarray(
        sess.ni_events.get("Change_ON", []), dtype=float
    ).flatten()
    shift_s = shift_ms / 1000.0

    entries = []
    for i, trial in enumerate(sess.trials):
        outcome = (trial.trialoutcome or "").lower()
        if outcome not in ("hit", "fa"):
            continue
        rt_dict = trial.reactiontimes or {}

        if outcome == "hit":
            rt = rt_dict.get("RT", rt_dict.get("Hit", rt_dict.get("hit", np.nan)))
            if i >= len(change_on) or np.isnan(rt):
                continue
            t_change = change_on[i]
            if t_change == 0 or np.isnan(t_change):
                continue
            lick_time = t_change + rt - shift_s
        elif outcome == "fa":
            rt = rt_dict.get("FA", rt_dict.get("fa", np.nan))
            if i >= len(baseline_on) or np.isnan(rt):
                continue
            lick_time = baseline_on[i] + rt - shift_s

        entries.append((i, outcome, lick_time))

    # Sort chronologically for sequential H.264 seeking (audit D6)
    entries.sort(key=lambda x: x[2])
    return entries


def _find_local_video(session_name):
    """Find local video file and its metadata CSV."""
    fname = SESSION_VIDEO_MAP.get(session_name)
    if fname is None:
        raise FileNotFoundError(
            f"No local video mapping for session {session_name}"
        )
    video_path = os.path.join(LOCAL_VIDEO_DIR, fname)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    # Metadata CSV: same name with .csv extension
    meta_path = video_path.rsplit(".", 1)[0] + ".csv"
    if not os.path.isfile(meta_path):
        # Try finding metadata from camera root
        cam_files = find_camera_files(session_name)
        meta_path = cam_files["eye_cam"]["metadata"]
    return video_path, meta_path


def extract_motion_energy_window(cap, ts_ms, center_ms, window_s, roi):
    """Extract motion energy in a time window around center_ms.

    Parameters
    ----------
    cap : cv2.VideoCapture
        Open video capture (position will be modified).
    ts_ms : np.ndarray
        Camera timestamps in ms (from metadata CSV).
    center_ms : float
        Center of the extraction window in camera ms.
    window_s : float
        Half-window size in seconds.
    roi : tuple
        (y0, y1, x0, x1) for the mouth region.

    Returns
    -------
    me_array : np.ndarray
        Motion energy values (N-1 frames).
    ts_array : np.ndarray
        Timestamps in ms corresponding to each ME value (midpoint of pair).
    """
    import cv2

    window_ms = window_s * 1000.0
    t_start = center_ms - window_ms
    t_end = center_ms + window_ms

    # Find frame indices in window
    mask = (ts_ms >= t_start) & (ts_ms <= t_end)
    indices = np.where(mask)[0]
    if len(indices) < 3:
        return np.array([]), np.array([])

    y0, y1, x0, x1 = roi

    # Seek once to start, then read sequentially
    cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])

    frames_gray = []
    frame_ts = []
    expected_idx = indices[0]
    for idx in indices:
        # If there's a gap, seek to it
        if idx != expected_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        expected_idx = idx + 1
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        frames_gray.append(gray[y0:y1, x0:x1])
        frame_ts.append(ts_ms[idx])

    if len(frames_gray) < 2:
        return np.array([]), np.array([])

    # Compute motion energy: mean absolute difference between consecutive frames
    me_values = []
    me_ts = []
    for i in range(1, len(frames_gray)):
        diff = np.abs(frames_gray[i] - frames_gray[i - 1])
        me_values.append(np.mean(diff))
        me_ts.append((frame_ts[i] + frame_ts[i - 1]) / 2.0)

    return np.array(me_values), np.array(me_ts)


def build_lick_me_threshold(cap, ts_ms, lick_cam_ms_all, roi):
    """Compute session-level ME threshold via Gaussian mixture model.

    Implements Khilkevich & Lohse (Nature 2024) movement onset method:
      1. Pool ME values from ±1s windows around a sample of lick events.
      2. Fit a 3-Gaussian mixture to capture lick-ME variance + noise.
      3. Threshold = mean + 2*SD of the lowest-mean (noise) Gaussian.

    Falls back to MAD-based threshold if GMM fitting fails.

    Returns
    -------
    threshold : float
    gmm_params : dict or None
        Fitted GMM parameters {means, stds, weights, threshold}, or None.
    """
    # Sample licks in chronological order for efficient H.264 seeking
    rng = np.random.RandomState(42)
    n_sample = min(GMM_N_SAMPLE_LICKS, len(lick_cam_ms_all))
    chosen = np.sort(rng.choice(len(lick_cam_ms_all), size=n_sample, replace=False))

    all_me = []
    for idx in chosen:
        center_ms = lick_cam_ms_all[idx]
        if center_ms < ts_ms[0] + LICK_WINDOW_S * 1000 or \
           center_ms > ts_ms[-1] - LICK_WINDOW_S * 1000:
            continue
        me_array, _ = extract_motion_energy_window(
            cap, ts_ms, center_ms, LICK_WINDOW_S, roi
        )
        if len(me_array) > 0:
            all_me.extend(me_array.tolist())

    if len(all_me) < 50:
        logger.warning("  Too few ME samples for GMM — using MAD fallback threshold")
        med = np.median(all_me) if all_me else 0.0
        mad = median_abs_deviation(all_me, scale="normal") if all_me else 1.0
        return max(med + 3.0 * mad, 1.0), None

    me_pool = np.array(all_me).reshape(-1, 1)

    try:
        gmm = GaussianMixture(
            n_components=GMM_N_COMPONENTS, covariance_type="full",
            random_state=42, max_iter=300, n_init=3,
        )
        gmm.fit(me_pool)

        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.reshape(GMM_N_COMPONENTS, -1)[:, 0])
        weights = gmm.weights_

        order = np.argsort(means)
        noise_mean = means[order[0]]
        noise_std = stds[order[0]]
        threshold = noise_mean + GMM_THRESHOLD_N_SD * noise_std

        logger.info(
            f"  GMM threshold: {threshold:.3f} "
            f"(noise Gaussian: mean={noise_mean:.3f}, std={noise_std:.3f})"
        )
        return threshold, {
            "means": means[order].tolist(),
            "stds": stds[order].tolist(),
            "weights": weights[order].tolist(),
            "threshold": threshold,
        }

    except Exception as exc:
        logger.warning(f"  GMM fitting failed ({exc}) — using MAD fallback threshold")
        med = np.median(all_me)
        mad = median_abs_deviation(all_me, scale="normal")
        return max(med + 3.0 * mad, 1.0), None


def detect_motion_onset_backward(me_array, ts_array, lick_cam_ms, threshold):
    """Detect movement onset by scanning backward from lick registration time.

    Implements Khilkevich & Lohse (Nature 2024):
      Scan backward in time from the lick registration timestamp.
      The time point preceding the first instance of ME dropping below
      the noise threshold is the movement onset time.

    Parameters
    ----------
    me_array : np.ndarray
        Motion energy values.
    ts_array : np.ndarray
        Timestamps in ms for each ME value.
    lick_cam_ms : float
        Camera-time equivalent of the software lick registration timestamp.
    threshold : float
        Session-level ME threshold from build_lick_me_threshold.

    Returns
    -------
    onset_ms : float or None
        Camera time of movement onset, or None if not found.
    """
    if len(me_array) < 3:
        return None

    # Start from the frame closest to lick registration time
    lick_idx = np.argmin(np.abs(ts_array - lick_cam_ms))

    # Scan backward; find the last above-threshold point
    for i in range(lick_idx, 0, -1):
        if me_array[i] < threshold:
            # First below-threshold point going backward →
            # the preceding index (i+1) is the onset
            onset_idx = i + 1
            if onset_idx <= lick_idx:
                return float(ts_array[onset_idx])
            return None  # onset would be at or after lick — reject

    return None  # ME never dropped below threshold (noisy baseline)


def _detect_me_peak(me_array, ts_array, threshold):
    """Find the timestamp of the highest ME peak above a permissive threshold.

    Used in Pass 1 of _iterative_lick_detection, where the lick camera-time
    estimate may be off by hundreds of ms (coarse offset only).  Peak-finding
    is more robust than backward-scan under high temporal uncertainty because
    the peak is the strongest feature in the window regardless of where exactly
    the window is centred.

    The returned timestamp approximates the lick time with a systematic lead of
    ~300 ms (jaw velocity peak precedes lick contact).  This bias is corrected
    in Pass 2 when the backward-scan finds the true movement onset.

    Parameters
    ----------
    me_array : np.ndarray
    ts_array : np.ndarray
    threshold : float
        Session-level GMM threshold.  Pass 1 uses threshold * 0.5 to be
        permissive (coarse detection; Pass 2 will refine).

    Returns
    -------
    peak_ms : float or None
    """
    if len(me_array) < 3:
        return None
    peak_idx = int(np.argmax(me_array))
    if me_array[peak_idx] < threshold * 0.5:
        return None
    return float(ts_array[peak_idx])


def extract_temporal_variance_window(cap, ts_ms, center_ms, window_s, roi,
                                     sliding_n=CORNEAL_SLIDING_WINDOW):
    """Extract sliding temporal variance of intensity in an eye ROI.

    For each frame, computes mean pixel intensity in the ROI, then applies
    a sliding-window variance (captures onset of flickering grating reflection).

    Returns
    -------
    tvar_array : np.ndarray
        Temporal variance values.
    tvar_ts : np.ndarray
        Timestamps in ms for each variance value (center of sliding window).
    """
    import cv2

    window_ms = window_s * 1000.0
    t_start = center_ms - window_ms
    t_end = center_ms + window_ms

    mask = (ts_ms >= t_start) & (ts_ms <= t_end)
    indices = np.where(mask)[0]
    if len(indices) < sliding_n + 2:
        return np.array([]), np.array([])

    y0, y1, x0, x1 = roi

    # Read frames sequentially (seek once, then read)
    intensities = []
    frame_ts = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
    expected_idx = indices[0]
    for idx in indices:
        if idx != expected_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        expected_idx = idx + 1
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        intensities.append(np.mean(gray[y0:y1, x0:x1]))
        frame_ts.append(ts_ms[idx])

    intensities = np.array(intensities)
    frame_ts = np.array(frame_ts)

    if len(intensities) < sliding_n + 1:
        return np.array([]), np.array([])

    # Sliding-window temporal variance
    tvar = []
    tvar_ts = []
    half = sliding_n // 2
    for i in range(half, len(intensities) - half):
        window = intensities[i - half: i + half + 1]
        tvar.append(np.var(window))
        tvar_ts.append(frame_ts[i])

    return np.array(tvar), np.array(tvar_ts)


def detect_variance_onset(tvar_array, ts_array):
    """Detect grating-onset variance increase via first sustained threshold crossing.

    Searches for the first 3-consecutive-frame crossing of baseline_mean + 3*SD,
    starting from the first third of the window (n_base) to capture reflections
    that become visible slightly before the nominal Baseline_ON time (grating drift
    phase + ~30ms sync uncertainty).

    Returns
    -------
    onset_ms : float or None
        Camera time of detected onset, or None if no clean crossing found.
    snr : float
        peak_post / baseline_mean.
    """
    if len(tvar_array) < 10:
        return None, 0.0

    n_baseline = len(tvar_array) // 3
    baseline = tvar_array[:n_baseline]
    bl_mean = np.mean(baseline)
    bl_std = np.std(baseline)

    if bl_std < 1e-6:
        bl_std = 1.0

    threshold = bl_mean + 3.0 * bl_std

    n_sustained = 3
    for i in range(n_baseline, len(tvar_array) - n_sustained + 1):
        if all(tvar_array[i + j] > threshold for j in range(n_sustained)):
            peak_var = np.mean(tvar_array[i: i + n_sustained])
            snr = peak_var / bl_mean if bl_mean > 1e-6 else peak_var
            return ts_array[i], snr

    return None, 0.0


# ═══════════════════════════════════════════════════════════════════
# ROI Preview
# ═══════════════════════════════════════════════════════════════════

def preview_rois(session_name, video_path, ts_ms):
    """Save annotated sample frame with both ROIs drawn."""
    import cv2

    os.makedirs(FIG_DIR, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    # Seek to ~5 minutes in for a representative frame
    target_ms = 5 * 60 * 1000
    target_idx = np.argmin(np.abs(ts_ms - target_ms))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        logger.warning(f"  Could not read preview frame for {session_name}")
        return

    annotated = frame.copy()

    # Mouth ROI (green)
    y0, y1, x0, x1 = MOUTH_ROI
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(annotated, "Mouth ROI", (x0, y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Eye ROI (cyan) — per-session
    eye_roi = CORNEAL_EYE_ROI.get(session_name, list(CORNEAL_EYE_ROI.values())[0])
    y0, y1, x0, x1 = eye_roi
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 255, 0), 2)
    cv2.putText(annotated, "Eye ROI", (x0, y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    out_path = os.path.join(FIG_DIR, f"{session_name}_roi_preview.png")
    cv2.imwrite(out_path, annotated)
    logger.info(f"  ROI preview saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════
# Phase A — Lick detection validation on good session
# ═══════════════════════════════════════════════════════════════════

def run_phase_a(session_name):
    """Phase A: Validate lick detection on a session with known-good screen sync.

    Returns dict with all results needed for the diagnostic figure.
    """
    import cv2

    logger.info(f"=== Phase A: Lick detection on {session_name} ===")

    # 1. Load session
    sess = load_session(session_name)
    baseline_on = np.asarray(
        sess.ni_events.get("Baseline_ON", []), dtype=float
    ).flatten()
    baseline_on = baseline_on[baseline_on > 0]

    # 2. Find video files
    video_path, meta_path = _find_local_video(session_name)
    ts_ms, _, _ = load_camera_metadata(meta_path)
    logger.info(f"  Video: {os.path.basename(video_path)}, "
                f"{len(ts_ms)} frames, {ts_ms[-1]/1000:.0f}s")

    # ROI preview
    preview_rois(session_name, video_path, ts_ms)

    # 3. Generate screen-glow sync model
    logger.info("  Building screen-glow sync model...")
    coarse_offset = load_or_compute_coarse_offset(
        session_name, video_path, meta_path, baseline_on
    )
    logger.info(f"  Coarse offset: {coarse_offset:.1f}s")

    onset_result = detect_onsets_variance(
        video_path, meta_path, baseline_on, coarse_offset, progress=True
    )
    sync_result = fit_clock_model(
        onset_result.detected_cam_s,
        onset_result.detected_nidaq_s,
        n_baseline_on=len(baseline_on),
    )
    logger.info(f"  Screen sync: RMSE={sync_result.rmse_ms:.1f}ms, "
                f"anchors={sync_result.n_anchors}/{len(baseline_on)}")

    if sync_result.rmse_ms > 50:
        logger.warning(f"  Screen sync RMSE too high ({sync_result.rmse_ms:.1f}ms). "
                       "Phase A results may be unreliable.")

    # 4. Extract lick times and convert to camera ms
    lick_entries = _get_lick_times(sess, LICK_HARDWARE_DELAY_MS)
    logger.info(f"  {len(lick_entries)} lick events (Hit + FA)")

    lick_nidaq_s = np.array([e[2] for e in lick_entries])
    lick_cam_ms = nidaq_to_camera(lick_nidaq_s, sync_result.slope, sync_result.offset)

    # 5. Compute session-level GMM threshold (paper method)
    logger.info("  Fitting GMM threshold from lick ME distribution...")
    cap = cv2.VideoCapture(video_path)
    lick_threshold, gmm_params = build_lick_me_threshold(
        cap, ts_ms, lick_cam_ms, MOUTH_ROI
    )

    # 6. Detect movement onset (backward scan) for each lick
    detections = []   # (trial_idx, outcome, expected_ms, onset_ms)
    misses = []       # (trial_idx, outcome, expected_ms)

    n_total = len(lick_entries)
    for k, (trial_idx, outcome, nidaq_t) in enumerate(lick_entries):
        if k % 50 == 0:
            logger.info(f"  Processing lick {k+1}/{n_total}...")

        expected_ms = lick_cam_ms[k]

        # Skip if outside video range
        if expected_ms < ts_ms[0] + LICK_WINDOW_S * 1000 or \
           expected_ms > ts_ms[-1] - LICK_WINDOW_S * 1000:
            continue

        me_array, me_ts = extract_motion_energy_window(
            cap, ts_ms, expected_ms, LICK_WINDOW_S, MOUTH_ROI
        )
        onset_ms = detect_motion_onset_backward(
            me_array, me_ts, expected_ms, lick_threshold
        )

        if onset_ms is not None:
            detections.append((trial_idx, outcome, expected_ms, onset_ms))
        else:
            misses.append((trial_idx, outcome, expected_ms))

    cap.release()

    # 7. Compute offsets (onset_ms − lick_cam_ms, expected to be negative)
    n_detected = len(detections)
    n_missed = len(misses)
    n_valid = n_detected + n_missed
    det_rate = n_detected / max(n_valid, 1)

    offsets_ms = np.array([d[3] - d[2] for d in detections])

    logger.info(f"  Detection rate: {n_detected}/{n_valid} = {det_rate:.1%}")
    if len(offsets_ms) > 0:
        logger.info(f"  Onset offset (onset − lick reg.): median={np.median(offsets_ms):.1f}ms, "
                    f"IQR=[{np.percentile(offsets_ms, 25):.1f}, "
                    f"{np.percentile(offsets_ms, 75):.1f}]ms")

    # 8. Temporal cross-validation (audit S2)
    cv_results = {}
    if n_detected >= 20:
        det_arr = np.array(detections)
        # Split by chronological order (already sorted)
        even_mask = np.arange(n_detected) % 2 == 0
        odd_mask = ~even_mask
        for label, mask in [("even", even_mask), ("odd", odd_mask)]:
            sub_offsets = offsets_ms[mask]
            cv_results[label] = {
                "n": int(mask.sum()),
                "median_offset": float(np.median(sub_offsets)),
                "iqr": (float(np.percentile(sub_offsets, 25)),
                        float(np.percentile(sub_offsets, 75))),
            }
        logger.info(f"  CV (even): n={cv_results['even']['n']}, "
                    f"median={cv_results['even']['median_offset']:.1f}ms")
        logger.info(f"  CV (odd):  n={cv_results['odd']['n']}, "
                    f"median={cv_results['odd']['median_offset']:.1f}ms")

    # Save an example ME trace
    example_trace = None
    if n_detected > 0:
        median_offset = np.median(offsets_ms)
        closest_to_median = np.argmin(np.abs(offsets_ms - median_offset))
        example_det = detections[closest_to_median]
        # Re-extract the ME trace for this trial
        cap = cv2.VideoCapture(video_path)
        me_ex, ts_ex = extract_motion_energy_window(
            cap, ts_ms, example_det[2], LICK_WINDOW_S, MOUTH_ROI
        )
        cap.release()
        example_trace = {
            "me": me_ex, "ts": ts_ex,
            "lick_cam_ms": example_det[2],   # software lick time in camera ms
            "onset_ms": example_det[3],       # detected movement onset
            "threshold": lick_threshold,
        }

    del sess
    gc.collect()

    return {
        "session": session_name,
        "n_detected": n_detected,
        "n_missed": n_missed,
        "n_valid": n_valid,
        "det_rate": det_rate,
        "offsets_ms": offsets_ms,
        "cv_results": cv_results,
        "example_trace": example_trace,
        "sync_rmse_ms": sync_result.rmse_ms,
        "sync_slope": sync_result.slope,
        "sync_offset": sync_result.offset,
    }


# ═══════════════════════════════════════════════════════════════════
# Phase B — Lick-barcode sync on failed session
# ═══════════════════════════════════════════════════════════════════

def _iterative_lick_detection(cap, ts_ms, lick_entries, initial_offset_s,
                               lick_threshold):
    """Two-pass iterative lick detection with clock refinement.

    Pass 1 — robust peak-finding with wide window:
        Uses ``_detect_me_peak`` (highest ME value in a ±PASS1_WINDOW_S window).
        Peak-finding tolerates coarse offset errors of ±(PASS1_WINDOW_S − 0.5) s
        because it finds the strongest feature regardless of window centring.
        The resulting anchors have a systematic ~300 ms lead bias (jaw velocity
        peak precedes lick contact), but that is fine for fitting a preliminary
        clock model with the correct slope and approximate offset.

    Pass 2 — precise backward-scan with tighter window:
        Uses ``detect_motion_onset_backward`` (paper method) with the refined
        lick-time estimates from the preliminary model.  The Pass 1 bias
        (anchors ~300 ms before true lick) causes Pass 2 windows to be centred
        ~300 ms before the true lick, but the onset at ~−692 ms is still within
        the ±LICK_WINDOW_S search window and the backward scan finds it correctly.

    Guard:
        If Pass 1 finds fewer than MIN_PASS1_ANCHORS events, or the preliminary
        model RMSE exceeds PRELIM_MODEL_MAX_RMSE_MS, Pass 2 is skipped and the
        Pass 1 anchors are returned as-is (better than a degenerate model).

    Returns
    -------
    list of (nidaq_s, cam_s) anchor pairs.
    """
    # ── Pass 1: robust peak-finding with wide window ──────────────────────
    n_pass1 = min(50, len(lick_entries))
    logger.info(f"  Pass 1: {n_pass1} events with coarse offset={initial_offset_s:.1f}s "
                f"(±{PASS1_WINDOW_S:.0f}s window, peak-finding)")

    pass1_anchors = []
    for k in range(n_pass1):
        trial_idx, outcome, nidaq_t = lick_entries[k]
        est_cam_ms = (nidaq_t - initial_offset_s) * 1000.0

        if est_cam_ms < ts_ms[0] + PASS1_WINDOW_S * 1000 or \
           est_cam_ms > ts_ms[-1] - PASS1_WINDOW_S * 1000:
            continue

        me_array, me_ts = extract_motion_energy_window(
            cap, ts_ms, est_cam_ms, PASS1_WINDOW_S, MOUTH_ROI
        )
        peak_ms = _detect_me_peak(me_array, me_ts, lick_threshold)

        if peak_ms is not None:
            pass1_anchors.append((nidaq_t, peak_ms / 1000.0))

    n_detected_p1 = len(pass1_anchors)
    logger.info(f"  Pass 1 detected: {n_detected_p1}/{n_pass1}")

    if n_detected_p1 < MIN_PASS1_ANCHORS:
        logger.warning(
            f"  Pass 1: only {n_detected_p1} anchors < {MIN_PASS1_ANCHORS} minimum. "
            "Skipping Pass 2 — returning Pass 1 anchors."
        )
        return pass1_anchors

    # ── Preliminary clock model ───────────────────────────────────────────
    cam_s_p1 = np.array([a[1] for a in pass1_anchors])
    nidaq_s_p1 = np.array([a[0] for a in pass1_anchors])
    prelim_result = fit_clock_model(cam_s_p1, nidaq_s_p1, n_baseline_on=len(lick_entries))
    logger.info(f"  Preliminary model: slope={prelim_result.slope:.8f}, "
                f"offset={prelim_result.offset:.3f}s, "
                f"RMSE={prelim_result.rmse_ms:.1f}ms")

    if prelim_result.rmse_ms > PRELIM_MODEL_MAX_RMSE_MS:
        logger.warning(
            f"  Preliminary model RMSE={prelim_result.rmse_ms:.0f}ms "
            f"> {PRELIM_MODEL_MAX_RMSE_MS:.0f}ms guard. "
            "Skipping Pass 2 — returning Pass 1 anchors."
        )
        return pass1_anchors

    # ── Pass 2: backward-scan onset detection with refined lick times ─────
    logger.info(f"  Pass 2: {len(lick_entries)} events with preliminary model "
                f"(±{LICK_WINDOW_S:.1f}s window, backward-scan)")
    all_nidaq_s = np.array([e[2] for e in lick_entries])
    refined_cam_ms = nidaq_to_camera(
        all_nidaq_s, prelim_result.slope, prelim_result.offset
    )

    pass2_anchors = []
    n_total = len(lick_entries)
    for k, (trial_idx, outcome, nidaq_t) in enumerate(lick_entries):
        if k % 50 == 0 and k > 0:
            logger.info(f"    Processing {k}/{n_total}...")

        est_cam_ms = refined_cam_ms[k]
        if est_cam_ms < ts_ms[0] + LICK_WINDOW_S * 1000 or \
           est_cam_ms > ts_ms[-1] - LICK_WINDOW_S * 1000:
            continue

        me_array, me_ts = extract_motion_energy_window(
            cap, ts_ms, est_cam_ms, LICK_WINDOW_S, MOUTH_ROI
        )
        onset_ms = detect_motion_onset_backward(
            me_array, me_ts, est_cam_ms, lick_threshold
        )

        if onset_ms is not None:
            pass2_anchors.append((nidaq_t, onset_ms / 1000.0))

    logger.info(f"  Pass 2 detected: {len(pass2_anchors)}/{n_total}")
    return pass2_anchors


def run_phase_b(session_name):
    """Phase B: Test lick-barcode sync on a failed (screen-glow) session.

    Returns dict with results for the diagnostic figure.
    """
    import cv2

    logger.info(f"=== Phase B: Barcode sync on {session_name} ===")

    # 1. Load session
    sess = load_session(session_name)
    baseline_on = np.asarray(
        sess.ni_events.get("Baseline_ON", []), dtype=float
    ).flatten()
    baseline_on = baseline_on[baseline_on > 0]

    # 2. Find video files
    video_path, meta_path = _find_local_video(session_name)
    ts_ms, _, _ = load_camera_metadata(meta_path)
    logger.info(f"  Video: {os.path.basename(video_path)}, "
                f"{len(ts_ms)} frames, {ts_ms[-1]/1000:.0f}s")

    # ROI preview
    preview_rois(session_name, video_path, ts_ms)

    # 3. Get coarse offset
    coarse_offset = load_or_compute_coarse_offset(
        session_name, video_path, meta_path, baseline_on
    )
    logger.info(f"  Coarse offset: {coarse_offset:.1f}s")

    # 4. Extract lick times
    lick_entries = _get_lick_times(sess, LICK_HARDWARE_DELAY_MS)
    logger.info(f"  {len(lick_entries)} lick events")

    # 5. Compute session-level GMM threshold using coarse offset
    lick_nidaq_s_all = np.array([e[2] for e in lick_entries])
    lick_cam_ms_coarse = (lick_nidaq_s_all - coarse_offset) * 1000.0

    cap = cv2.VideoCapture(video_path)
    logger.info("  Fitting GMM threshold from lick ME distribution...")
    lick_threshold, _ = build_lick_me_threshold(
        cap, ts_ms, lick_cam_ms_coarse, MOUTH_ROI
    )

    # 6. Iterative lick detection (paper backward-scan method)
    lick_anchors = _iterative_lick_detection(
        cap, ts_ms, lick_entries, coarse_offset, lick_threshold
    )
    cap.release()

    # 6. Screen glow detection
    logger.info("  Running screen-glow onset detection...")
    onset_result = detect_onsets_variance(
        video_path, meta_path, baseline_on, coarse_offset, progress=True
    )

    # 7. Fit three clock models
    lick_cam_s = np.array([a[1] for a in lick_anchors])
    lick_nidaq_s = np.array([a[0] for a in lick_anchors])

    screen_cam_s = onset_result.detected_cam_s
    screen_nidaq_s = onset_result.detected_nidaq_s

    results = {}

    # (a) Screen-only
    if len(screen_cam_s) >= 5:
        sr_screen = fit_clock_model(screen_cam_s, screen_nidaq_s,
                                    n_baseline_on=len(baseline_on))
        results["screen"] = {
            "rmse_ms": sr_screen.rmse_ms,
            "n_anchors": sr_screen.n_anchors,
            "residuals_ms": sr_screen.residuals_ms,
            "slope": sr_screen.slope,
            "offset": sr_screen.offset,
        }
        logger.info(f"  Screen-only: RMSE={sr_screen.rmse_ms:.1f}ms, "
                    f"n={sr_screen.n_anchors}")
    else:
        results["screen"] = {"rmse_ms": np.nan, "n_anchors": 0}
        logger.info(f"  Screen-only: too few anchors ({len(screen_cam_s)})")

    # (b) Lick-only
    if len(lick_cam_s) >= 5:
        sr_lick = fit_clock_model(lick_cam_s, lick_nidaq_s,
                                  n_baseline_on=len(lick_entries))
        results["lick"] = {
            "rmse_ms": sr_lick.rmse_ms,
            "n_anchors": sr_lick.n_anchors,
            "residuals_ms": sr_lick.residuals_ms,
            "slope": sr_lick.slope,
            "offset": sr_lick.offset,
        }
        logger.info(f"  Lick-only:   RMSE={sr_lick.rmse_ms:.1f}ms, "
                    f"n={sr_lick.n_anchors}")
    else:
        results["lick"] = {"rmse_ms": np.nan, "n_anchors": 0}
        logger.info(f"  Lick-only: too few anchors ({len(lick_cam_s)})")

    # (c) Combined
    if len(screen_cam_s) >= 1 and len(lick_cam_s) >= 1:
        comb_cam_s = np.concatenate([screen_cam_s, lick_cam_s])
        comb_nidaq_s = np.concatenate([screen_nidaq_s, lick_nidaq_s])
        comb_type = np.array(
            ["screen"] * len(screen_cam_s) + ["lick"] * len(lick_cam_s)
        )
        sr_comb = fit_clock_model(comb_cam_s, comb_nidaq_s,
                                  n_baseline_on=len(baseline_on) + len(lick_entries))
        # Per-type residuals (audit S4)
        if sr_comb.residuals_ms is not None and sr_comb.inlier_mask is not None:
            inlier_types = comb_type[sr_comb.inlier_mask]
            inlier_resids = sr_comb.residuals_ms[sr_comb.inlier_mask]  # filter to inliers only
            for atype in ["screen", "lick"]:
                type_mask = inlier_types == atype
                if type_mask.sum() > 0:
                    type_resid = inlier_resids[type_mask]
                    logger.info(f"    {atype} residuals: median={np.median(type_resid):.1f}ms, "
                                f"MAD={median_abs_deviation(type_resid):.1f}ms, n={type_mask.sum()}")

        inlier_mask = sr_comb.inlier_mask
        results["combined"] = {
            "rmse_ms": sr_comb.rmse_ms,
            "n_anchors": sr_comb.n_anchors,
            "residuals_ms": sr_comb.residuals_ms[inlier_mask] if inlier_mask is not None else sr_comb.residuals_ms,
            "matched_nidaq_s": sr_comb.matched_nidaq_s[inlier_mask] if inlier_mask is not None else sr_comb.matched_nidaq_s,
            "anchor_types": comb_type[inlier_mask] if inlier_mask is not None else None,
            "slope": sr_comb.slope,
            "offset": sr_comb.offset,
        }
        logger.info(f"  Combined:    RMSE={sr_comb.rmse_ms:.1f}ms, "
                    f"n={sr_comb.n_anchors}")
    else:
        results["combined"] = {"rmse_ms": np.nan, "n_anchors": 0}

    del sess
    gc.collect()

    # Determine primary sync result: prefer lick-only when screen-glow fails
    screen_rmse = results.get("screen", {}).get("rmse_ms", np.nan)
    lick_rmse = results.get("lick", {}).get("rmse_ms", np.nan)
    if not np.isnan(screen_rmse) and screen_rmse > SCREEN_FAIL_RMSE_MS \
            and not np.isnan(lick_rmse):
        primary = "lick"
        logger.info(
            f"  Screen-glow FAILED (RMSE={screen_rmse:.0f}ms > {SCREEN_FAIL_RMSE_MS:.0f}ms). "
            f"Primary sync: lick-only (RMSE={lick_rmse:.1f}ms, "
            f"n={results['lick']['n_anchors']})"
        )
    elif not np.isnan(lick_rmse) and not np.isnan(screen_rmse) \
            and lick_rmse < screen_rmse:
        primary = "lick"
        logger.info(f"  Primary sync: lick-only (RMSE={lick_rmse:.1f}ms < "
                    f"screen {screen_rmse:.1f}ms)")
    else:
        primary = "screen"
        logger.info(f"  Primary sync: screen (RMSE={screen_rmse:.1f}ms)")

    return {
        "session": session_name,
        "models": results,
        "primary": primary,
        "n_lick_anchors": len(lick_anchors),
        "n_screen_anchors": len(screen_cam_s),
        "n_lick_events": len(lick_entries),
        "n_baseline_on": len(baseline_on),
    }


# ═══════════════════════════════════════════════════════════════════
# Phase C — Corneal reflection test
# ═══════════════════════════════════════════════════════════════════

def run_phase_c(session_name, sync_slope, sync_offset):
    """Phase C: Test corneal reflection (grating flicker) on good session.

    Uses the sync model from Phase A to align camera time to Baseline_ON.
    """
    import cv2

    logger.info(f"=== Phase C: Corneal reflection on {session_name} ===")

    sess = load_session(session_name)
    baseline_on = np.asarray(
        sess.ni_events.get("Baseline_ON", []), dtype=float
    ).flatten()
    baseline_on = baseline_on[baseline_on > 0]

    video_path, meta_path = _find_local_video(session_name)
    ts_ms, _, _ = load_camera_metadata(meta_path)

    # Convert Baseline_ON to camera ms using the sync model
    baseline_cam_ms = nidaq_to_camera(baseline_on, sync_slope, sync_offset)

    # Limit to first 100 trials
    n_trials = min(100, len(baseline_cam_ms))
    logger.info(f"  Processing {n_trials} Baseline_ON trials")

    cap = cv2.VideoCapture(video_path)
    detections = []  # (trial_idx, expected_ms, onset_ms, snr)
    misses = []
    example_trace = None

    for k in range(n_trials):
        if k % 20 == 0:
            logger.info(f"  Trial {k+1}/{n_trials}...")

        center_ms = baseline_cam_ms[k]
        if center_ms < ts_ms[0] + CORNEAL_WINDOW_S * 1000 or \
           center_ms > ts_ms[-1] - CORNEAL_WINDOW_S * 1000:
            continue

        tvar, tvar_ts = extract_temporal_variance_window(
            cap, ts_ms, center_ms, CORNEAL_WINDOW_S,
            CORNEAL_EYE_ROI.get(session_name, list(CORNEAL_EYE_ROI.values())[0])
        )
        onset_ms, snr = detect_variance_onset(tvar, tvar_ts)

        if onset_ms is not None:
            detections.append((k, center_ms, onset_ms, snr))
            # Save first good example for figure
            if example_trace is None and snr > 2.0:
                example_trace = {
                    "tvar": tvar, "ts": tvar_ts,
                    "expected_ms": center_ms, "onset_ms": onset_ms,
                    "snr": snr,
                }
        else:
            misses.append((k, center_ms))

    cap.release()

    n_detected = len(detections)
    n_valid = n_detected + len(misses)
    det_rate = n_detected / max(n_valid, 1)

    snr_values = np.array([d[3] for d in detections]) if detections else np.array([])
    onset_offsets = np.array([d[2] - d[1] for d in detections]) if detections else np.array([])

    logger.info(f"  Detection rate: {n_detected}/{n_valid} = {det_rate:.1%}")
    if len(snr_values) > 0:
        logger.info(f"  SNR: median={np.median(snr_values):.2f}, "
                    f"IQR=[{np.percentile(snr_values, 25):.2f}, "
                    f"{np.percentile(snr_values, 75):.2f}]")
        logger.info(f"  Onset offset: median={np.median(onset_offsets):.1f}ms")

    del sess
    gc.collect()

    return {
        "session": session_name,
        "n_detected": n_detected,
        "n_valid": n_valid,
        "det_rate": det_rate,
        "snr_values": snr_values,
        "onset_offsets": onset_offsets,
        "example_trace": example_trace,
    }


# ═══════════════════════════════════════════════════════════════════
# Diagnostic figure (3×3 grid)
# ═══════════════════════════════════════════════════════════════════

def make_diagnostic_figure(phase_a, phase_b, phase_c, session_a, session_b):
    """Create the 3×3 diagnostic figure from all phase results."""
    os.makedirs(FIG_DIR, exist_ok=True)

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35,
                           left=0.07, right=0.97, top=0.95, bottom=0.05)

    # ── Row 0: Phase A ──────────────────────────────────────────
    # Panel (0,0): Example ME trace
    ax00 = fig.add_subplot(gs[0, 0])
    if phase_a["example_trace"] is not None:
        ex = phase_a["example_trace"]
        t_rel = ex["ts"] - ex["lick_cam_ms"]
        ax00.plot(t_rel, ex["me"], "k-", lw=0.8)
        ax00.axhline(ex["threshold"], color="gray", ls=":", lw=1, label="GMM threshold")
        ax00.axvline(0, color="blue", ls="--", lw=1, label="Lick registration")
        onset_rel = ex["onset_ms"] - ex["lick_cam_ms"]
        ax00.axvline(onset_rel, color="red", ls="-", lw=1.5, label="Movement onset")
        ax00.legend(fontsize=7)
    ax00.set_xlabel("Time from lick registration (ms)")
    ax00.set_ylabel("Motion energy")
    ax00.set_title("A: Example ME trace (backward scan)")

    # Panel (0,1): Offset histogram
    ax01 = fig.add_subplot(gs[0, 1])
    if len(phase_a["offsets_ms"]) > 0:
        offsets = phase_a["offsets_ms"]
        ax01.hist(offsets, bins=40, color="#4393c3", edgecolor="white", lw=0.5)
        ax01.axvline(np.median(offsets), color="red", ls="--", lw=1.5,
                     label=f"Median: {np.median(offsets):.1f}ms")
        ax01.legend(fontsize=8)
    ax01.set_xlabel("Offset (detected − expected, ms)")
    ax01.set_ylabel("Count")
    ax01.set_title("A: Offset distribution")

    # Panel (0,2): Summary text
    ax02 = fig.add_subplot(gs[0, 2])
    ax02.axis("off")
    lines = [
        f"Phase A: Lick detection ({session_a})",
        f"Screen sync RMSE: {phase_a['sync_rmse_ms']:.1f} ms",
        "",
        f"Lick events: {phase_a['n_valid']}",
        f"Detected: {phase_a['n_detected']} ({phase_a['det_rate']:.0%})",
        f"Missed: {phase_a['n_missed']}",
    ]
    if len(phase_a["offsets_ms"]) > 0:
        off = phase_a["offsets_ms"]
        lines += [
            "",
            f"Onset offset median: {np.median(off):.1f} ms",
            f"Onset offset IQR: [{np.percentile(off, 25):.1f}, {np.percentile(off, 75):.1f}]",
            "(negative = onset precedes lick reg.)",
        ]
    if phase_a["cv_results"]:
        cv = phase_a["cv_results"]
        lines += [
            "",
            "Temporal CV (audit S2):",
            f"  Even: med={cv['even']['median_offset']:.1f}ms (n={cv['even']['n']})",
            f"  Odd:  med={cv['odd']['median_offset']:.1f}ms (n={cv['odd']['n']})",
        ]
    ax02.text(0.05, 0.95, "\n".join(lines), transform=ax02.transAxes,
              fontsize=8, va="top", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0"))

    # ── Row 1: Phase B ──────────────────────────────────────────
    # Panel (1,0): Residuals scatter
    ax10 = fig.add_subplot(gs[1, 0])
    models = phase_b["models"]
    if "combined" in models and models["combined"].get("residuals_ms") is not None:
        resid = models["combined"]["residuals_ms"]
        nidaq_t = models["combined"]["matched_nidaq_s"]
        atypes = models["combined"].get("anchor_types")
        if atypes is not None and nidaq_t is not None:
            screen_mask = atypes == "screen"
            lick_mask = atypes == "lick"
            if screen_mask.sum() > 0:
                ax10.scatter(nidaq_t[screen_mask], resid[screen_mask],
                            s=10, alpha=0.5, c="#4393c3", label="Screen")
            if lick_mask.sum() > 0:
                ax10.scatter(nidaq_t[lick_mask], resid[lick_mask],
                            s=10, alpha=0.5, c="#d6604d", label="Lick")
            ax10.legend(fontsize=7)
        elif nidaq_t is not None:
            ax10.scatter(nidaq_t, resid, s=10, alpha=0.4, c="gray")
    ax10.axhline(0, color="k", ls="-", lw=0.5)
    ax10.set_xlabel("NI-DAQ time (s)")
    ax10.set_ylabel("Residual (ms)")
    ax10.set_title("B: Combined model residuals")

    # Panel (1,1): RMSE bar chart
    ax11 = fig.add_subplot(gs[1, 1])
    model_names = ["Screen", "Lick", "Combined"]
    model_keys = ["screen", "lick", "combined"]
    rmses = [models.get(k, {}).get("rmse_ms", np.nan) for k in model_keys]
    colors = ["#4393c3", "#d6604d", "#762a83"]
    bars = ax11.bar(model_names, rmses, color=colors, edgecolor="white", lw=0.5)
    for bar, val in zip(bars, rmses):
        if not np.isnan(val):
            ax11.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax11.set_ylabel("RMSE (ms)")
    ax11.set_title("B: Model comparison")
    # Add quality tier line
    ax11.axhline(20, color="green", ls="--", lw=1, alpha=0.5)
    ax11.text(2.6, 21, "Good", fontsize=7, color="green")
    ax11.axhline(40, color="orange", ls="--", lw=1, alpha=0.5)
    ax11.text(2.6, 41, "Review", fontsize=7, color="orange")

    # Panel (1,2): Summary text
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.axis("off")
    lines = [
        f"Phase B: Barcode sync ({session_b})",
        f"Baseline_ON trials: {phase_b['n_baseline_on']}",
        f"Lick events: {phase_b['n_lick_events']}",
        "",
    ]
    for key, name in [("screen", "Screen"), ("lick", "Lick"), ("combined", "Combined")]:
        m = models.get(key, {})
        rmse = m.get("rmse_ms", np.nan)
        n = m.get("n_anchors", 0)
        lines.append(f"{name:>8}: RMSE={rmse:.1f}ms, n={n}")
    lines += [
        "",
        f"Lick anchors: {phase_b['n_lick_anchors']}",
        f"Screen anchors: {phase_b['n_screen_anchors']}",
    ]
    # Verdict — show primary sync and RMSE tier
    primary = phase_b.get("primary", "screen")
    lick_rmse = models.get("lick", {}).get("rmse_ms", np.nan)
    screen_rmse = models.get("screen", {}).get("rmse_ms", np.nan)
    if primary == "lick" and not np.isnan(screen_rmse) and screen_rmse > SCREEN_FAIL_RMSE_MS:
        tier = "GOOD" if lick_rmse <= 20 else ("REVIEW" if lick_rmse <= 40 else "POOR")
        lines += ["", f"Screen FAILED ({screen_rmse:.0f}ms)",
                  f"PRIMARY: lick-only {lick_rmse:.1f}ms [{tier}]"]
    elif primary == "lick":
        lines += ["", f"VERDICT: Lick better ({lick_rmse:.0f} vs {screen_rmse:.0f}ms)"]
    else:
        lines += ["", f"VERDICT: Screen ({screen_rmse:.0f}ms)"]

    ax12.text(0.05, 0.95, "\n".join(lines), transform=ax12.transAxes,
              fontsize=8, va="top", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0"))

    # ── Row 2: Phase C ──────────────────────────────────────────
    # Panel (2,0): Example temporal variance trace
    ax20 = fig.add_subplot(gs[2, 0])
    if phase_c["example_trace"] is not None:
        ex = phase_c["example_trace"]
        t_rel = ex["ts"] - ex["expected_ms"]
        ax20.plot(t_rel, ex["tvar"], "k-", lw=0.8)
        ax20.axvline(0, color="blue", ls="--", lw=1, label="Baseline_ON")
        onset_rel = ex["onset_ms"] - ex["expected_ms"]
        ax20.axvline(onset_rel, color="red", ls="-", lw=1.5,
                     label=f"Onset (SNR={ex['snr']:.1f})")
        ax20.legend(fontsize=7)
    ax20.set_xlabel("Time from Baseline_ON (ms)")
    ax20.set_ylabel("Temporal variance")
    ax20.set_title("C: Example corneal variance")

    # Panel (2,1): SNR histogram
    ax21 = fig.add_subplot(gs[2, 1])
    if len(phase_c["snr_values"]) > 0:
        ax21.hist(phase_c["snr_values"], bins=30, color="#66c2a5",
                  edgecolor="white", lw=0.5)
        med_snr = np.median(phase_c["snr_values"])
        ax21.axvline(med_snr, color="red", ls="--", lw=1.5,
                     label=f"Median: {med_snr:.2f}")
        ax21.axvline(2.0, color="orange", ls=":", lw=1.5, label="Viable threshold")
        ax21.legend(fontsize=8)
    ax21.set_xlabel("SNR")
    ax21.set_ylabel("Count")
    ax21.set_title("C: SNR distribution")

    # Panel (2,2): Summary text
    ax22 = fig.add_subplot(gs[2, 2])
    ax22.axis("off")
    lines = [
        f"Phase C: Corneal reflection ({phase_c['session']})",
        f"Trials tested: {phase_c['n_valid']}",
        f"Detected: {phase_c['n_detected']} ({phase_c['det_rate']:.0%})",
        "",
    ]
    if len(phase_c["snr_values"]) > 0:
        snr = phase_c["snr_values"]
        lines += [
            f"SNR median: {np.median(snr):.2f}",
            f"SNR IQR: [{np.percentile(snr, 25):.2f}, {np.percentile(snr, 75):.2f}]",
        ]
    if len(phase_c["onset_offsets"]) > 0:
        off = phase_c["onset_offsets"]
        lines += [
            f"Onset offset: {np.median(off):.1f} ms (median)",
            f"Onset jitter: {np.percentile(off, 75) - np.percentile(off, 25):.1f} ms (IQR)",
        ]
    # Verdict
    if len(phase_c["snr_values"]) > 0:
        med_snr = np.median(phase_c["snr_values"])
        if med_snr > 2.0:
            lines += ["", f"VERDICT: Corneal reflection VIABLE (SNR={med_snr:.1f})"]
        else:
            lines += ["", f"VERDICT: Weak signal (SNR={med_snr:.1f} < 2.0)"]

    ax22.text(0.05, 0.95, "\n".join(lines), transform=ax22.transAxes,
              fontsize=8, va="top", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0"))

    out_path = os.path.join(
        FIG_DIR, f"multianchor_poc_{session_a}_{session_b}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Diagnostic figure saved: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="POC: Multi-anchor video sync")
    parser.add_argument("--session-a", default="09092025",
                        help="Good-sync session for Phase A+C (default: 09092025)")
    parser.add_argument("--session-b", default="27062025",
                        help="Failed-sync session for Phase B (default: 27062025)")
    args = parser.parse_args()

    session_a = args.session_a
    session_b = args.session_b

    logger.info(f"Multi-anchor sync POC")
    logger.info(f"  Session A (good): {session_a}")
    logger.info(f"  Session B (fail): {session_b}")

    # Phase A
    phase_a = run_phase_a(session_a)

    # Phase B
    phase_b = run_phase_b(session_b)

    # Phase C — reuse sync model from Phase A (no recomputation needed)
    phase_c = run_phase_c(session_a, phase_a["sync_slope"], phase_a["sync_offset"])

    # Diagnostic figure
    fig_path = make_diagnostic_figure(phase_a, phase_b, phase_c, session_a, session_b)

    logger.info("=== POC complete ===")
    logger.info(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
