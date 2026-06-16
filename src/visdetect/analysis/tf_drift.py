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
