"""FSI/SPN cell-type from extracellular waveform shape (M2).

Features and the 2-component-GMM-on-T2P classification are ported from
AI_exploration/analysis_3_waveform_celltype.py. Pure (no I/O): the producer
script wires these to RawWaveforms via visdetect.analysis.tracking_qc.

See docs/superpowers/specs/2026-06-03-presentation-prep-roadmap-design.md (§9).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

SR_HZ: float = 30000.0          # Neuropixels sample rate
T2P_MIN_MS: float = 0.02        # GMM fit window lower bound
T2P_MAX_MS: float = 1.5         # GMM fit window upper bound

_NAN_FEATURES = {"t2p_ms": np.nan, "half_width_ms": np.nan, "pt_ratio": np.nan}


def compute_waveform_features(peak_waveform: np.ndarray) -> Dict[str, float]:
    """Trough-to-peak, half-width, and peak/trough ratio from a 1-D peak-channel waveform.

    Returns NaN features for degenerate inputs (too short / flat).
    """
    w = np.asarray(peak_waveform, dtype=float)
    if w.size < 10:
        return dict(_NAN_FEATURES)
    denom = np.abs(w).max()
    if denom < 1e-12:
        return dict(_NAN_FEATURES)
    w_norm = w / (denom + 1e-12)

    trough_idx = int(np.argmin(w_norm))
    after = w_norm[trough_idx:]
    if after.size < 2:
        return dict(_NAN_FEATURES)
    peak_after_idx = trough_idx + int(np.argmax(after))
    t2p_ms = (peak_after_idx - trough_idx) / SR_HZ * 1000.0

    half_min = w_norm[trough_idx] / 2.0
    below_half = np.where(w_norm < half_min)[0]
    hw_ms = ((below_half[-1] - below_half[0]) / SR_HZ * 1000.0
             if below_half.size >= 2 else np.nan)

    pt_ratio = float(w_norm[peak_after_idx] / (-w_norm[trough_idx] + 1e-12))
    return {"t2p_ms": float(t2p_ms), "half_width_ms": float(hw_ms), "pt_ratio": pt_ratio}


def classify_celltype(
    t2p_ms_array: np.ndarray,
    *,
    random_state: int = 42,
) -> Tuple[np.ndarray, object]:
    """Classify FSI vs SPN using a 2-component GMM fit on trough-to-peak duration.

    Parameters
    ----------
    t2p_ms_array:
        1-D array of per-unit T2P values (ms). NaN entries are excluded from
        fitting and returned as label ``-1`` (unclassified).
    random_state:
        Seed for GMM initialisation.

    Returns
    -------
    labels : np.ndarray of int (same length as input)
        0 = FSI (narrow spike, short T2P), 1 = SPN (broad spike, long T2P),
        -1 = NaN input (unclassified).
    gmm : fitted sklearn GaussianMixture instance (for inspection / plotting).

    Notes
    -----
    Implemented in Task 2. Raises NotImplementedError until then.
    """
    raise NotImplementedError("classify_celltype will be implemented in Task 2 (GMM classifier).")
