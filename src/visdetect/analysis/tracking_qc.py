"""Per-UID tracking QC: metrics, badge logic, and extraction primitives.

This module is library code (no I/O orchestration). The
`scripts/pipelines/tracking/build_qc_sheets.py` driver wires it up.

See docs/superpowers/specs/2026-05-21-tracking-qc-sheets-design.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# ─── Badge thresholds (tweakable; documented in spec §7) ──────────────
ISI_PASS: float = 0.75
ISI_WARN: float = 0.65

DEPTH_PASS_UM: float = 15.0
DEPTH_WARN_UM: float = 30.0

WAVE_PASS_R: float = 0.95
WAVE_WARN_R: float = 0.90

FR_CV_PASS: float = 0.35
FR_CV_WARN: float = 0.60

# ─── Change-size pools for Change_ON heatmaps ─────────────────────────
# Spec excludes 1.5× from heatmaps (ambiguous mid).
BIG_POOL: Set[float] = {2.0, 4.0}
SMALL_POOL: Set[float] = {1.25, 1.35}

# ─── Footprint extraction ─────────────────────────────────────────────
# How many channels above/below the peak to include in the footprint snippet.
FOOTPRINT_HALFWIDTH_CHANS: int = 8


# ─── Cross-session metric functions ───────────────────────────────────

def depth_std_um(depths_um: np.ndarray) -> float:
    """Std of peak-channel depth across sessions, in microns.

    NaN values are ignored. Returns NaN if fewer than 2 finite values.
    """
    arr = np.asarray(depths_um, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    return float(np.std(arr, ddof=0))


def waveform_corr(waveforms: np.ndarray) -> float:
    """Mean pairwise Pearson r of L2-normalized peak-channel waveforms.

    Parameters
    ----------
    waveforms : ndarray, shape (n_sessions, n_samples)
        Per-session mean waveform on the peak channel.

    Returns
    -------
    float
        Mean over the (n*(n-1)/2) cross-session pairwise correlations.
        NaN if fewer than 2 sessions or if normalization fails.
    """
    arr = np.asarray(waveforms, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return float("nan")

    # L2-normalize per row; drop rows that are all-zero
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    keep = norms.flatten() > 1e-12
    if keep.sum() < 2:
        return float("nan")
    normed = arr[keep] / norms[keep]

    # Pearson r of normalized vectors == cosine == dot product after mean removal
    # We want Pearson, not cosine — subtract row mean first
    normed = normed - normed.mean(axis=1, keepdims=True)
    # Renormalize after mean-subtraction
    norms2 = np.linalg.norm(normed, axis=1, keepdims=True)
    norms2[norms2 < 1e-12] = 1.0
    normed = normed / norms2

    n = normed.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(float(np.dot(normed[i], normed[j])))
    return float(np.mean(pairs))


def fr_cv(rates_hz: np.ndarray) -> float:
    """Coefficient of variation (std/mean) of baseline firing rate.

    NaNs are dropped. Returns NaN for empty / zero-mean / single-session inputs.
    """
    arr = np.asarray(rates_hz, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    mean = float(np.mean(arr))
    if abs(mean) < 1e-9:
        return float("nan")
    return float(np.std(arr, ddof=0) / mean)


# ─── Badge / verdict logic ────────────────────────────────────────────

def _badge_threshold(value: float, pass_thr: float, warn_thr: float,
                     direction: str) -> str:
    """Apply pass/warn/fail thresholds.

    direction='high' : pass if value >= pass_thr, warn between, fail below.
    direction='low'  : pass if value <= pass_thr, warn between, fail above.
    NaN always returns 'fail'.
    """
    if not np.isfinite(value):
        return "fail"
    if direction == "high":
        if value >= pass_thr:
            return "pass"
        if value >= warn_thr:
            return "warn"
        return "fail"
    elif direction == "low":
        if value <= pass_thr:
            return "pass"
        if value <= warn_thr:
            return "warn"
        return "fail"
    raise ValueError(f"direction must be 'high' or 'low', got {direction!r}")


def badge_isi(median_corr: float) -> str:
    return _badge_threshold(median_corr, ISI_PASS, ISI_WARN, direction="high")


def badge_depth(std_um: float) -> str:
    return _badge_threshold(std_um, DEPTH_PASS_UM, DEPTH_WARN_UM, direction="low")


def badge_waveform(mean_pairwise_r: float) -> str:
    return _badge_threshold(mean_pairwise_r, WAVE_PASS_R, WAVE_WARN_R, direction="high")


def badge_fr(cv: float) -> str:
    return _badge_threshold(cv, FR_CV_PASS, FR_CV_WARN, direction="low")


def composite_verdict(badges: Sequence[str]) -> str:
    """Spec §7 composite logic.

    trusted = all pass
    review  = ≤1 warn AND no fails
    suspect = any fail OR ≥2 warns
    """
    n_fail = sum(1 for b in badges if b == "fail")
    n_warn = sum(1 for b in badges if b == "warn")
    if n_fail >= 1 or n_warn >= 2:
        return "suspect"
    if n_warn == 1:
        return "review"
    return "trusted"
