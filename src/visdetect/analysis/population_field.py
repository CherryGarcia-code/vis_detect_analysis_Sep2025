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
