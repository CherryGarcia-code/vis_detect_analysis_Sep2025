"""Centralized analysis constants for the visdetect pipeline.

All event windows, outcome filters, threshold values, and change-size
groupings used across neural and behavioral analyses are defined here
as the **single source of truth**.

Importing from this module ensures consistency across scripts (e.g.
``hmm_neural_states.py``, ``hmm_neural_TF_event_comparison.py``,
``compare_early_late_fa.py``, ``quantify_fa_suppression.py``, etc.).

Usage
-----
    from visdetect.analysis.constants import (
        EVENT_RESPONSIVENESS_WINDOWS,
        EVENT_VALID_OUTCOMES,
        SMALL_CHANGE_SIZES,
        BIG_CHANGE_SIZES,
        FA_RT_SPLIT,
    )
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

# =====================================================================
# Event responsiveness windows
# =====================================================================
# Maps event name → (baseline_window, response_window).
# These define the time intervals relative to event onset used to
# compute a responsiveness index (baseline-subtracted or z-scored).

EVENT_RESPONSIVENESS_WINDOWS: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {
    # Sensory-evoked: short latency after stimulus change
    "Change_ON": ((-0.4, -0.05), (0, 0.25)),
    # Preparatory motor (lick-aligned): activity ramp before movement
    "FA":        ((-1.75, -1.25), (-0.3, -0.15)),
    "Hit":       ((-1.75, -1.25), (-0.3, -0.15)),
    # Baseline onset: early vs late baseline
    "Baseline_ON": ((-1.75, -1.25), (0.0, 1.0)),
}

# =====================================================================
# Trial-type / outcome filters
# =====================================================================
# For each event alignment, which ``trialoutcome`` values are valid.
# ``None`` means no filtering (all trials used).

EVENT_VALID_OUTCOMES: Dict[str, Optional[Set[str]]] = {
    "Change_ON": {"hit", "miss"},   # exclude FA/abort — change never happened
    "FA":        {"fa"},
    "Hit":       {"hit"},
    "Baseline_ON": None,            # all trial types see baseline
}

# =====================================================================
# Change-size grouping
# =====================================================================
# Go-trial change sizes: [1.25, 1.35, 1.5, 2.0, 4.0]
# Catch trials have change_size ≈ 1.0 (no change).

SMALL_CHANGE_SIZES: Set[float] = {1.25, 1.35, 1.5}
BIG_CHANGE_SIZES: Set[float] = {2.0, 4.0}
ALL_GO_CHANGE_SIZES: Set[float] = SMALL_CHANGE_SIZES | BIG_CHANGE_SIZES

# =====================================================================
# FA early / late split
# =====================================================================
# Reaction time threshold (seconds) to split false alarms.
# RT ≤ FA_RT_SPLIT → early;  RT > FA_RT_SPLIT → late.

FA_RT_SPLIT: float = 3.0

# =====================================================================
# Lick hardware delay
# =====================================================================
# The lick spout sensor reports contact ~200 ms after the actual tongue
# contact.  Subtracted from software-recorded RT to estimate true lick time.
# Used in ``align.compute_true_reaction_time`` and video sync validation.

LICK_HARDWARE_DELAY_MS: float = 200.0

# =====================================================================
# TF pulse alignment windows
# =====================================================================
# Pre- and post-pulse windows used for z-scoring TF pulse responses.
# These mirror the defaults in ``TFRespPulseConfig``.

TF_PULSE_PRE_WINDOW: Tuple[float, float] = (-0.4, 0.0)
TF_PULSE_POST_WINDOW: Tuple[float, float] = (0.0, 0.5)

# Wider trace extraction window (for detrending in labeling GUI).
# Z-score baseline still uses TF_PULSE_PRE_WINDOW; this only controls how far
# back the extracted trace extends.
TF_PULSE_TRACE_PRE: float = -1.0

# Convenience: full TF pulse analysis window
TF_PULSE_WINDOW: Tuple[float, float] = (TF_PULSE_PRE_WINDOW[0], TF_PULSE_POST_WINDOW[1])

# Linear detrending parameters for TF pulse traces.
# Baseline window for fitting the trend line; post-window for conservative
# peak measurement after extrapolation (narrower than TF_PULSE_POST_WINDOW
# to avoid unreliable linear extrapolation far beyond the fit region).
TF_DETREND_BASELINE: Tuple[float, float] = (-0.4, -0.01)
TF_DETREND_POST_WINDOW: Tuple[float, float] = (0.0, 0.3)

# =====================================================================
# TF pulse classification thresholds (log2 scale)
# =====================================================================

TF_FAST_THRESH_LOG2: float = 0.25
TF_SLOW_THRESH_LOG2: float = -0.25
TF_SAMPLE_PERIOD: float = 0.25   # seconds per baseline TF bin (4 Hz base)

# =====================================================================
# Default PSTH / smoothing parameters
# =====================================================================

DEFAULT_BIN_SIZE: float = 0.025        # 25 ms
DEFAULT_SIGMA_MS: float = 25.0         # Gaussian smoothing sigma
DEFAULT_MIN_FR: float = 1.0            # minimum firing rate (Hz)
DEFAULT_Z_THRESH_TF: float = 3.0       # TF responsiveness z-score threshold

# =====================================================================
# Lohse et al. (2025) parameters
# =====================================================================
# From "Frontal cortex gates striatal dynamics to enable flexible control
# of behaviour", Methods lines 698-876.

LOHSE_PULSE_SIGMA_MS: float = 17.0              # 40 ms FWHM for pulse-aligned activity
LOHSE_TRIAL_SIGMA_MS: float = 42.5              # 100 ms FWHM for trial-aligned activity
LOHSE_SENSORY_CD_WINDOW: Tuple[float, float] = (0.122, 0.167)  # Post-pulse window for sensory CD
LOHSE_EVIDENCE_THRESH: float = 7.5              # Evidence selectivity threshold (peak z)
LOHSE_TRIAL_NORM_BASELINE: Tuple[float, float] = (-1.3, -0.3)  # Pre-trial-start normalization

# =====================================================================
# Video synchronization parameters
# =====================================================================
# Used by ``visdetect.core.video_sync`` to align camera timestamps to
# the NI-DAQ master clock via luminance-based multi-anchor fitting.

VIDEO_SYNC_SMOOTH_FRAMES: int = 5           # Median filter window for luminance derivative
VIDEO_SYNC_DETECT_THRESH: float = 5.0       # MAD multiplier for transition detection
VIDEO_SYNC_DETECT_THRESH_LOW: float = 3.0   # Adaptive fallback if <80% transitions found
VIDEO_SYNC_CLUSTER_MS: float = 200.0        # Cluster nearby detections within this window (ms)
VIDEO_SYNC_MATCH_REJECT_S: float = 1.0      # Reject anchor matches with residual > this (s)
VIDEO_SYNC_RANSAC_THRESH_S: float = 0.1     # RANSAC inlier threshold (seconds)
VIDEO_SYNC_MAX_DRIFT_PPM: float = 5000.0    # Flag sessions with clock drift > this (ppm)
VIDEO_SYNC_MAX_RESIDUAL_MS: float = 60.0    # Flag if max residual exceeds this (ms)
VIDEO_SYNC_MIN_COVERAGE: float = 0.85       # Minimum anchor fraction for "good" quality
VIDEO_SYNC_COARSE_SEARCH_S: float = 60.0    # Brute-force offset search range (±seconds)
VIDEO_SYNC_COARSE_STEP_S: float = 0.5       # Brute-force offset search step (seconds)

# --- Derivative-based onset detection (preferred method) ---
# The eye camera is IR-illuminated; the strongest Baseline_ON signal is the
# background screen glow visible in the upper-right region of the frame.
# When the gray screen transitions to a drifting grating, the luminance trace
# in this ROI goes from flat to oscillating (TF modulation).  We detect the
# first large absolute derivative as the onset timestamp.
VIDEO_SYNC_DERIV_SIGMA_MULT: float = 5.0    # MAD multiplier for derivative threshold
VIDEO_SYNC_DERIV_MIN_THRESH: float = 2.0    # Minimum abs-derivative for onset (pixel units)
VIDEO_SYNC_DERIV_PRE_FRAMES: int = 20       # Frames before expected onset for baseline noise
VIDEO_SYNC_DERIV_SEARCH_FRAMES: int = 30    # Frames before/after expected onset to search
VIDEO_SYNC_OUTLIER_N_ITER: int = 3          # Iterative MAD outlier rejection passes
VIDEO_SYNC_OUTLIER_SIGMA: float = 3.0       # MAD multiplier for outlier rejection
# Default ROI for BG_046 eye camera: background glow upper-right of frame
# (y0, y1, x0, x1) — screen illumination visible behind the mouse head
VIDEO_SYNC_DEFAULT_EYE_ROI: tuple = (50, 300, 600, 900)
