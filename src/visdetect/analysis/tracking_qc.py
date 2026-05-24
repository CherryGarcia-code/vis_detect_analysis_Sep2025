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


import pandas as pd
from pathlib import Path
from collections import defaultdict


def load_isi_scores(csv_path) -> Dict[int, float]:
    """Read the median ISI corr per global_uid from validate_long_tracks output.

    Missing UIDs are returned as NaN via a defaultdict.
    """
    df = pd.read_csv(csv_path)
    scores = defaultdict(lambda: float("nan"))
    for _, row in df.iterrows():
        scores[int(row["global_uid"])] = float(row["median"])
    return scores


# ─── ISI histogram ────────────────────────────────────────────────────
# Matches the binning used by validate_long_tracks.py (1 ms .. 10 s, log).
_ISI_BIN_EDGES = np.logspace(-3, 1, 51)
_ISI_CENTERS = 0.5 * (_ISI_BIN_EDGES[:-1] + _ISI_BIN_EDGES[1:])


def isi_log_histogram(spike_times: np.ndarray, n_bins: int = 50
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Normalised log-ISI histogram, 1 ms .. 10 s, 50 bins by default.

    Returns
    -------
    h : ndarray, shape (n_bins,)
        Probability mass per bin (sums to 1).  All-NaN if too few spikes.
    centers : ndarray, shape (n_bins,)
        Bin centres (s).
    """
    if n_bins != 50:
        edges = np.logspace(-3, 1, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
    else:
        edges = _ISI_BIN_EDGES
        centers = _ISI_CENTERS

    if spike_times is None or len(spike_times) < 20:
        return np.full(n_bins, np.nan), centers
    isis = np.diff(np.sort(spike_times))
    isis = isis[(isis > 0) & (isis < 10)]
    if len(isis) < 10:
        return np.full(n_bins, np.nan), centers
    h, _ = np.histogram(isis, bins=edges)
    if h.sum() == 0:
        return np.full(n_bins, np.nan), centers
    return h.astype(float) / h.sum(), centers


# ─── Waveform / footprint extraction ──────────────────────────────────

def extract_peak_channel(mean_waveform: np.ndarray) -> int:
    """Index of the channel with the largest peak-to-peak amplitude.

    Parameters
    ----------
    mean_waveform : ndarray, shape (n_samples, n_channels)
    """
    ptp = mean_waveform.max(axis=0) - mean_waveform.min(axis=0)
    return int(np.argmax(ptp))


def extract_footprint(mean_waveform: np.ndarray, peak_chan: int,
                      halfwidth: int = FOOTPRINT_HALFWIDTH_CHANS
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Footprint snippet: (n_samples, 2*halfwidth+1) clipped at probe edges.

    Returns
    -------
    snippet : ndarray, shape (n_samples, n_channels_kept)
    channel_indices : ndarray, shape (n_channels_kept,)
    """
    n_ch = mean_waveform.shape[1]
    lo = max(0, peak_chan - halfwidth)
    hi = min(n_ch, peak_chan + halfwidth + 1)
    channels = np.arange(lo, hi)
    snippet = mean_waveform[:, lo:hi]
    return snippet, channels


import os


def load_raw_mean_waveform(raw_wf_root, session_name: str, ks_unit_id: int
                            ) -> Optional[np.ndarray]:
    """Load Unit{kid}_RawSpikes.npy and return mean across CV halves.

    Parameters
    ----------
    raw_wf_root : str or Path
        e.g. ``data/unit_match/input/BG_046``
    session_name : str
        DDMMYYYY (8-digit) — matches the unit-match input layout.
    ks_unit_id : int

    Returns
    -------
    mean_waveform : ndarray, shape (n_samples, n_channels), or None if file missing.
    """
    candidates = [session_name, session_name.zfill(8)]
    for cand in candidates:
        path = os.path.join(str(raw_wf_root), cand, "RawWaveforms",
                            f"Unit{ks_unit_id}_RawSpikes.npy")
        if os.path.exists(path):
            raw = np.load(path)   # (n_samples, n_channels, n_cv)
            if raw.ndim == 3:
                return raw.mean(axis=-1).astype(np.float32)
            elif raw.ndim == 2:
                return raw.astype(np.float32)
            return None
    return None


def load_channel_positions(raw_wf_root, session_name: str) -> Optional[np.ndarray]:
    """Load channel_positions.npy for a session.  Shape (n_channels, 2) [x_um, y_um]."""
    for cand in (session_name, session_name.zfill(8)):
        path = os.path.join(str(raw_wf_root), cand, "channel_positions.npy")
        if os.path.exists(path):
            return np.load(path).astype(np.float32)
    return None
