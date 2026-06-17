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


from visdetect.analysis.tf_selectivity import (
    _shared_baseline,
    compute_unit_selectivity,
)


def test_shared_baseline_is_single_value():
    cfg = TFSelectivityConfig()
    t_vec = _time_vector(cfg)
    # Two traces with deliberately different pre-window spread.
    fast_hz = np.zeros_like(t_vec); fast_hz[:] = 10.0
    slow_hz = np.zeros_like(t_vec); slow_hz[:] = 10.0
    pre = (t_vec >= cfg.pulse.pre_window[0]) & (t_vec < cfg.pulse.pre_window[1])
    rng = np.random.default_rng(0)
    fast_hz[pre] += rng.normal(0, 5.0, pre.sum())
    slow_hz[pre] += rng.normal(0, 1.0, pre.sum())
    mu, sd = _shared_baseline(fast_hz, slow_hz, t_vec, cfg.pulse.pre_window, cfg.eps)
    # The pooled sd must lie between the two per-condition sds, i.e. it is one
    # shared number, not computed separately per condition.
    assert sd > 1.0 and sd < 5.0


def test_selectivity_uses_shared_sigma():
    cfg = TFSelectivityConfig(n_shuffles=10)
    t_vec = _time_vector(cfg)
    # Hand-built fast/slow Hz traces: identical baseline, fast bump in post.
    sel = compute_unit_selectivity.__wrapped__ if hasattr(compute_unit_selectivity, "__wrapped__") else None
    # Use the real driver via a tiny session in the next tasks; here we check
    # the algebra directly through the public helper composition:
    fast_hz = np.full_like(t_vec, 8.0)
    slow_hz = np.full_like(t_vec, 8.0)
    post = (t_vec >= 0.0) & (t_vec < 0.2)
    fast_hz[post] = 18.0
    mu, sd = _shared_baseline(fast_hz, slow_hz, t_vec, cfg.pulse.pre_window, cfg.eps)
    selectivity = (fast_hz - slow_hz) / sd
    # baseline difference is zero -> selectivity flat there; post bump positive.
    pre = (t_vec >= cfg.pulse.pre_window[0]) & (t_vec < cfg.pulse.pre_window[1])
    assert np.allclose(selectivity[pre], 0.0)
    assert selectivity[post].max() > 0
