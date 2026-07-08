"""Tracking-free anatomical population field — instrument primitives.

Cross-session correspondence comes from fixed anatomy on a MATCH-FREE
registered depth axis (the amplitude-depth activity landscape), never from
single-unit tracking. See docs/superpowers/specs/2026-07-07-tracking-free-
population-field-design.md.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# New constants (flagged for user confirmation — Global Constraints).
DEPTH_BIN_UM: float = 60.0
REG_MAX_LAG_UM: float = 300.0


def depth_bin_edges(channel_positions: np.ndarray,
                    depth_bin_um: float = DEPTH_BIN_UM) -> np.ndarray:
    """Monotonic y-edges (µm) covering the active depth band at ``depth_bin_um``."""
    y = np.asarray(channel_positions, float)[:, 1]
    lo = np.floor(y.min() / depth_bin_um) * depth_bin_um
    hi = np.ceil(y.max() / depth_bin_um) * depth_bin_um
    return np.arange(lo, hi + depth_bin_um, depth_bin_um)


def robust_unit_depth(mean_waveform: np.ndarray,
                      channel_positions: np.ndarray) -> float:
    """Amplitude(ptp)-weighted centroid of channel depth. NaN if no amplitude."""
    ptp = mean_waveform.max(axis=0) - mean_waveform.min(axis=0)   # (n_chan,)
    y = np.asarray(channel_positions, float)[:, 1]
    w = np.asarray(ptp, float)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    return float((w * y).sum() / total)


def amplitude_depth_fingerprint(unit_waveforms: List[np.ndarray],
                                channel_positions: np.ndarray,
                                y_edges: np.ndarray) -> np.ndarray:
    """Pool every channel's ptp of every unit into its depth bin (whole-probe)."""
    y = np.asarray(channel_positions, float)[:, 1]
    n_bins = len(y_edges) - 1
    chan_bin = np.clip(np.searchsorted(y_edges, y) - 1, 0, n_bins - 1)
    profile = np.zeros(n_bins, float)
    for mw in unit_waveforms:
        ptp = mw.max(axis=0) - mw.min(axis=0)       # (n_chan,)
        np.add.at(profile, chan_bin, ptp)
    return profile


def estimate_shift_bins(ref: np.ndarray, mov: np.ndarray,
                        max_lag_bins: int) -> Tuple[int, float]:
    """Rigid bin shift aligning ``mov`` onto ``ref`` + peak normalized corr.

    Lifted from scripts/pipelines/tracking/diagnose_intersession_drift.py::estimate_shift.
    """
    ref = ref - ref.mean()
    mov = mov - mov.mean()
    denom = np.sqrt((ref ** 2).sum() * (mov ** 2).sum())
    if denom < 1e-9:
        return 0, 0.0
    best_lag, best_c = 0, -np.inf
    for lag in range(-max_lag_bins, max_lag_bins + 1):
        shifted = np.roll(mov, lag)
        if lag > 0:
            shifted[:lag] = 0
        elif lag < 0:
            shifted[lag:] = 0
        c = float((ref * shifted).sum() / denom)
        if c > best_c:
            best_c, best_lag = c, lag
    return best_lag, best_c


from visdetect.analysis.tracking_qc import (          # noqa: E402
    load_raw_mean_waveform, load_channel_positions,
)


def session_fingerprint_from_root(raw_wf_root, session_name: str,
                                  unit_ids: List[int],
                                  y_edges: np.ndarray) -> np.ndarray:
    """Whole-probe amplitude-depth fingerprint for one session's good+stable units."""
    pos = load_channel_positions(raw_wf_root, session_name)
    if pos is None:
        return np.zeros(len(y_edges) - 1, float)
    wfs = []
    for uid in unit_ids:
        mw = load_raw_mean_waveform(raw_wf_root, session_name, int(uid))
        if mw is not None:
            wfs.append(mw)
    return amplitude_depth_fingerprint(wfs, pos, y_edges)


def session_shift_um(fingerprints: Dict[str, np.ndarray], ref_session: str,
                     depth_bin_um: float = DEPTH_BIN_UM,
                     max_lag_um: float = REG_MAX_LAG_UM
                     ) -> Dict[str, Tuple[float, float]]:
    """Per-session rigid registration shift (µm) + corr vs the reference session.

    Positive shift_um ⇒ that session's landscape sits deeper than the reference.
    """
    ref = fingerprints[ref_session]
    max_lag_bins = int(round(max_lag_um / depth_bin_um))
    out: Dict[str, Tuple[float, float]] = {}
    for sess, mov in fingerprints.items():
        lag, corr = estimate_shift_bins(ref, mov, max_lag_bins)
        out[sess] = (-lag * depth_bin_um, corr)   # deeper session -> positive shift
    return out
