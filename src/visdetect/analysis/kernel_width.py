"""Continuous width estimators for a 1-D GLM TF kernel (or any deflection trace).

All estimators operate on the ABSOLUTE deflection |K| so suppression-type cells
(~half of TF-responsive units fire *less* to fast pulses) are treated the same as
excitatory cells. `grid_fwhm` reproduces the pipeline's coarse walk-out exactly
(for the registry validation gate); `interpolated_fwhm` and `temporal_spread` are
the continuous measures the 50 ms lag grid cannot resolve.
"""
from __future__ import annotations

import numpy as np


def _abs(K: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(K, dtype=float))


def peak_lag(K: np.ndarray, lags: np.ndarray) -> float:
    a = _abs(K)
    if a.size == 0 or not np.any(a > 0):
        return float("nan")
    return float(np.asarray(lags, float)[int(np.argmax(a))])


def grid_fwhm(K: np.ndarray, lags: np.ndarray) -> float:
    """Pipeline-identical FWHM: walk out from the peak while |K| >= half-max,
    return lags[hi] - lags[lo] (quantized to the lag grid)."""
    a = _abs(K)
    lags = np.asarray(lags, float)
    if a.size == 0 or not np.any(a > 0):
        return float("nan")
    ip = int(np.argmax(a))
    half = a[ip] / 2.0
    lo = ip
    while lo > 0 and a[lo - 1] >= half:
        lo -= 1
    hi = ip
    while hi < a.size - 1 and a[hi + 1] >= half:
        hi += 1
    return float(lags[hi] - lags[lo])


def _half_cross(a: np.ndarray, lags: np.ndarray, ip: int, half: float, direction: int) -> float:
    """Linear-interpolated lag where |K| crosses `half` moving `direction` (+1 right,
    -1 left) from the peak. Clamps to the boundary lag if no crossing (censored)."""
    i = ip
    while 0 <= i + direction < a.size and a[i + direction] >= half:
        i += direction
    j = i + direction  # first index strictly below half (or out of range)
    if j < 0 or j >= a.size:
        return float(lags[0] if direction < 0 else lags[-1])
    # a[j] < half <= a[i]; interpolate the crossing between lags[j] and lags[i]
    denom = a[i] - a[j]
    frac = 0.0 if denom == 0 else (half - a[j]) / denom
    return float(lags[j] + frac * (lags[i] - lags[j]))


def interpolated_fwhm(K: np.ndarray, lags: np.ndarray) -> float:
    """Sub-bin FWHM of |K| via linear half-max crossing interpolation."""
    a = _abs(K)
    lags = np.asarray(lags, float)
    if a.size < 2 or not np.any(a > 0):
        return float("nan")
    ip = int(np.argmax(a))
    half = a[ip] / 2.0
    left = _half_cross(a, lags, ip, half, -1)
    right = _half_cross(a, lags, ip, half, +1)
    return float(right - left)


def temporal_spread(K: np.ndarray, lags: np.ndarray) -> float:
    """sqrt second-moment (temporal SD, s) of the |K| mass about its centroid."""
    a = _abs(K)
    lags = np.asarray(lags, float)
    tot = a.sum()
    if a.size == 0 or tot <= 0:
        return float("nan")
    w = a / tot
    tbar = float(np.sum(w * lags))
    return float(np.sqrt(np.sum(w * (lags - tbar) ** 2)))
