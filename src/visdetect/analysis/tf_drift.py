"""Source-level drift correction for TF pulse-triggered analysis.

See docs/superpowers/specs/2026-06-15-tf-responsiveness-labeler-design.md (§5).
Estimate each unit's slow firing drift over the whole session and subtract it
*before* pulse-triggered averaging, so the pre-pulse baseline (and its z-score
SD) is genuinely flat. Pure functions; reuses the KDE/z-score helpers in
tf_pulse.py rather than reimplementing them.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d


def estimate_drift(
    spike_times: np.ndarray,
    t_start: float,
    t_end: float,
    bin_s: float = 0.5,
    kernel_s: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Slow firing-rate estimate (Hz) over [t_start, t_end).

    Bins spikes at ``bin_s`` and smooths with a Gaussian of width
    ``kernel_s`` (seconds). Wide ``kernel_s`` captures session-scale drift;
    narrower also removes faster within-window structure — Phase 0 tunes it.

    Returns
    -------
    grid_t : (n_bins,) bin-centre times (s)
    drift  : (n_bins,) smoothed rate (Hz)
    mean_rate : scalar mean rate (Hz) over the window
    """
    spike_times = np.asarray(spike_times, dtype=float)
    spike_times = spike_times[(spike_times >= t_start) & (spike_times < t_end)]
    dur = max(t_end - t_start, 1e-9)
    n_bins = max(int(np.ceil(dur / bin_s)), 1)
    edges = t_start + np.arange(n_bins + 1) * bin_s
    counts, _ = np.histogram(spike_times, bins=edges)
    rate = counts.astype(float) / bin_s
    sigma_bins = max(kernel_s / bin_s, 1e-6)
    drift = gaussian_filter1d(rate, sigma=sigma_bins, mode="nearest")
    grid_t = 0.5 * (edges[:-1] + edges[1:])
    mean_rate = float(spike_times.size / dur)
    return grid_t, drift, mean_rate


# add to src/visdetect/analysis/tf_drift.py
def detrended_pulse_average(
    spike_times: np.ndarray,
    pulse_times: np.ndarray,
    pre_window: Tuple[float, float],
    post_window: Tuple[float, float],
    dt: float,
    sigma_ms: float,
    drift_grid_t: np.ndarray,
    drift_rate: np.ndarray,
    mean_rate: float,
    trace_start=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drift-corrected pulse-triggered average in Hz.

    detrended(t) = PTA(fine_rate)/dt - PTA(slow_drift) + mean_rate
    """
    from visdetect.analysis.tf_pulse import _mean_activity_per_unit

    mean_fine, sem, t_vec = _mean_activity_per_unit(
        spike_times, pulse_times, pre_window, post_window, dt, sigma_ms,
        trace_start=trace_start)
    if mean_fine.size == 0:
        return mean_fine, sem, t_vec

    mean_fine_hz = mean_fine / dt
    sem_hz = sem / dt

    pulses = np.asarray(pulse_times, dtype=float)
    pulses = pulses[np.isfinite(pulses)]
    drift_grid_t = np.asarray(drift_grid_t, dtype=float)
    drift_rate = np.asarray(drift_rate, dtype=float)

    drift_pta = np.zeros_like(t_vec)
    for tp in pulses:
        drift_pta += np.interp(
            tp + t_vec, drift_grid_t, drift_rate,
            left=drift_rate[0], right=drift_rate[-1])
    drift_pta /= max(pulses.size, 1)

    detrended = mean_fine_hz - drift_pta + float(mean_rate)
    return detrended, sem_hz, t_vec
