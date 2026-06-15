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
    t2p_ms: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Classify units FSI/SPN from trough-to-peak via a 2-component GMM.

    A single global GMM is fit on T2P values within (T2P_MIN_MS, T2P_MAX_MS).
    The decision threshold is the mean of the two component means; units with
    T2P below it are FSI (narrow), at/above it SPN (broad). NaN T2P → Unclassified.

    Returns
    -------
    labels : ndarray of str, same shape as t2p_ms (values in {FSI, SPN, Unclassified}).
    info : dict with threshold_ms, narrow_mean_ms, broad_mean_ms, delta_bic, n.
    """
    from sklearn.mixture import GaussianMixture

    arr = np.asarray(t2p_ms, dtype=float)
    finite = np.isfinite(arr)
    in_window = finite & (arr > T2P_MIN_MS) & (arr < T2P_MAX_MS)
    X = arr[in_window].reshape(-1, 1)
    if X.shape[0] < 2:
        labels = np.full(arr.shape, "Unclassified", dtype=object)
        return labels, {"threshold_ms": np.nan, "narrow_mean_ms": np.nan,
                        "broad_mean_ms": np.nan, "delta_bic": np.nan, "n": int(X.shape[0])}

    gmm2 = GaussianMixture(n_components=2, random_state=random_state).fit(X)
    gmm1 = GaussianMixture(n_components=1, random_state=random_state).fit(X)
    means = np.sort(gmm2.means_.flatten())
    threshold = float(means.mean())

    labels = np.full(arr.shape, "Unclassified", dtype=object)
    labels[finite & (arr < threshold)] = "FSI"
    labels[finite & (arr >= threshold)] = "SPN"

    info = {
        "threshold_ms": threshold,
        "narrow_mean_ms": float(means[0]),
        "broad_mean_ms": float(means[1]),
        "delta_bic": float(gmm1.bic(X) - gmm2.bic(X)),
        "n": int(X.shape[0]),
    }
    return labels, info
