"""Tests for FSI/SPN waveform cell-type features + classification (M2)."""
import numpy as np
import pytest

from visdetect.analysis.waveform_celltype import (
    compute_waveform_features, classify_celltype, SR_HZ,
)


def _synthetic_spike(trough_lo=28, trough_hi=33, peak_idx=40, n=82):
    """Broad trough (depth -1) over [trough_lo, trough_hi), positive peak after."""
    w = np.zeros(n, dtype=float)
    w[trough_lo:trough_hi] = -1.0
    w[peak_idx] = 0.5
    return w


def test_features_t2p_and_halfwidth_known_values():
    w = _synthetic_spike(28, 33, 40)
    f = compute_waveform_features(w)
    # trough argmin = 28; peak after = 40 -> t2p = 12 samples
    assert f["t2p_ms"] == pytest.approx((40 - 28) / SR_HZ * 1000, rel=1e-6)
    # w < -0.5 at indices 28..32 -> half width = 4 samples
    assert f["half_width_ms"] == pytest.approx((32 - 28) / SR_HZ * 1000, rel=1e-6)
    assert f["pt_ratio"] == pytest.approx(0.5, rel=1e-6)


def test_features_short_input_returns_nans():
    f = compute_waveform_features(np.array([0.0, -1.0, 0.5]))
    assert np.isnan(f["t2p_ms"])


def test_features_flat_input_safe():
    f = compute_waveform_features(np.zeros(82))
    assert set(f) == {"t2p_ms", "half_width_ms", "pt_ratio"}
