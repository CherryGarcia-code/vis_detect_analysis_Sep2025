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


def prepulse_slope(trace, t_vec, pre_window: Tuple[float, float]) -> float:
    """Linear slope (units/s) of ``trace`` within the pre-pulse window.

    The Phase-0 success metric: after detrend, the population distribution of
    this should collapse toward 0. NaN if < 2 samples fall in the window.
    """
    t_vec = np.asarray(t_vec, dtype=float)
    trace = np.asarray(trace, dtype=float)
    mask = (t_vec >= pre_window[0]) & (t_vec < pre_window[1])
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(t_vec[mask], trace[mask], 1)[0])


def circular_shift_null(
    spike_times: np.ndarray,
    pulse_times: np.ndarray,
    pre_window: Tuple[float, float],
    post_window: Tuple[float, float],
    dt: float,
    sigma_ms: float,
    bin_s: float,
    kernel_s: float,
    session_dur: float,
    n_shuffles: int = 200,
    seed: int = 0,
    trace_start=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Null bank of z-scored detrended pulse traces under circular shifts.

    Returns (null_z, t_vec): null_z is (n_shuffles, n_time).
    """
    from visdetect.analysis.tf_pulse import _zscore_trace

    rng = np.random.default_rng(seed)
    spike_times = np.sort(np.asarray(spike_times, dtype=float))
    min_shift = max(30.0, session_dur * 0.05)
    hi = session_dur - min_shift
    if hi <= min_shift:
        hi = session_dur * 0.95
    rows = []
    t_vec = None
    for _ in range(int(n_shuffles)):
        shift = rng.uniform(min_shift, hi)
        shifted = np.sort((spike_times + shift) % session_dur)
        gt, dr, mr = estimate_drift(shifted, 0.0, session_dur, bin_s, kernel_s)
        det, _, t_vec = detrended_pulse_average(
            shifted, pulse_times, pre_window, post_window, dt, sigma_ms,
            gt, dr, mr, trace_start=trace_start)
        rows.append(_zscore_trace(det, t_vec, pre_window))
    return np.asarray(rows), t_vec
