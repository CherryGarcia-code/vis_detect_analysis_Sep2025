"""Tests for the fast-minus-slow selectivity detection core."""
import numpy as np

from visdetect.core.session import Session, Trial, Cluster
from visdetect.analysis.tf_selectivity import (
    TFSelectivityConfig,
    _time_vector,
    _per_pulse_rate_matrix,
)


def test_per_pulse_rate_matrix_recovers_hz():
    # A unit firing at a regular 100 Hz over the whole window; the per-pulse
    # mean rate in a flat interior region should be ~100 Hz.
    cfg = TFSelectivityConfig()
    t_vec = _time_vector(cfg)
    rate = 100.0
    spikes = np.arange(0.0, 1000.0, 1.0 / rate)
    pulses = np.array([100.0, 200.0, 300.0, 400.0])
    mat = _per_pulse_rate_matrix(spikes, pulses, t_vec, cfg.pulse.dt, cfg.pulse.sigma_ms)
    assert mat.shape == (4, t_vec.size)
    interior = (t_vec >= -0.5) & (t_vec < -0.1)
    mean_hz = np.nanmean(mat[:, interior])
    assert np.isclose(mean_hz, rate, rtol=0.05), mean_hz


def test_per_pulse_rate_matrix_empty_pulses():
    cfg = TFSelectivityConfig()
    t_vec = _time_vector(cfg)
    mat = _per_pulse_rate_matrix(np.arange(0, 10, 0.01), np.array([]), t_vec,
                                 cfg.pulse.dt, cfg.pulse.sigma_ms)
    assert mat.shape == (0, t_vec.size)
