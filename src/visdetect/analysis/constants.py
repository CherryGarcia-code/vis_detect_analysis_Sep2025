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
# TF pulse alignment windows
# =====================================================================
# Pre- and post-pulse windows used for z-scoring TF pulse responses.
# These mirror the defaults in ``TFRespPulseConfig``.

TF_PULSE_PRE_WINDOW: Tuple[float, float] = (-0.4, 0.0)
TF_PULSE_POST_WINDOW: Tuple[float, float] = (0.0, 0.5)

# Convenience: full TF pulse analysis window
TF_PULSE_WINDOW: Tuple[float, float] = (TF_PULSE_PRE_WINDOW[0], TF_PULSE_POST_WINDOW[1])

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
