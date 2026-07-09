import numpy as np
import pytest
from visdetect.analysis.kernel_width import (
    grid_fwhm, interpolated_fwhm, temporal_spread, peak_lag,
)

LAGS = np.arange(0.0, 1.5, 0.05)  # 30 lags, 0..1.45 s (mirrors kern['tf'])

def _triangular(peak_idx, half_bins, amp=1.0, n=30):
    """Symmetric triangular kernel peaking at peak_idx, reaching 0 half_bins out."""
    k = np.zeros(n)
    for i in range(n):
        d = abs(i - peak_idx)
        k[i] = amp * max(0.0, 1 - d / half_bins) if half_bins > 0 else (amp if d == 0 else 0.0)
    return k

def test_grid_fwhm_matches_pipeline_walkout():
    # peak at bin 6; |K|>=half for bins 5..7 (3 bins) => grid FWHM = lags[7]-lags[5] = 0.10
    K = _triangular(6, 3, amp=1.0)
    assert grid_fwhm(K, LAGS) == pytest.approx(LAGS[7] - LAGS[5])

def test_grid_fwhm_is_sign_agnostic():
    # suppression kernel (negative peak) must give the same width as its positive mirror
    K = _triangular(6, 3, amp=1.0)
    assert grid_fwhm(-K, LAGS) == pytest.approx(grid_fwhm(K, LAGS))

def test_interpolated_fwhm_subbin_between_grid_points():
    # triangular half-width 3 bins: half-max crossings fall exactly halfway between bins
    # -> interpolated FWHM = 3 bins * 0.05 = 0.15 (wider than the 0.10 grid value)
    K = _triangular(6, 3, amp=1.0)
    got = interpolated_fwhm(K, LAGS)
    assert got == pytest.approx(0.15, abs=1e-6)
    assert got > grid_fwhm(K, LAGS)  # sub-bin interpolation widens the coarse grid value

def test_interpolated_fwhm_left_censored_peak_at_zero_lag():
    # monotonic-decaying kernel peaking at bin 0: no left crossing -> clamp to lags[0]
    K = np.maximum(0.0, 1.0 - np.arange(30) / 4.0)
    got = interpolated_fwhm(K, LAGS)
    assert np.isfinite(got) and got > 0

def test_temporal_spread_wider_kernel_larger():
    narrow = _triangular(10, 2)
    broad = _triangular(10, 8)
    assert temporal_spread(broad, LAGS) > temporal_spread(narrow, LAGS)

def test_temporal_spread_sign_agnostic():
    K = _triangular(10, 5)
    assert temporal_spread(-K, LAGS) == pytest.approx(temporal_spread(K, LAGS))

def test_peak_lag_picks_abs_max():
    K = _triangular(9, 3, amp=-2.0)  # strongest deflection is negative at bin 9
    assert peak_lag(K, LAGS) == pytest.approx(LAGS[9])

def test_degenerate_flat_kernel_returns_nan():
    K = np.zeros(30)
    assert np.isnan(interpolated_fwhm(K, LAGS))
    assert np.isnan(temporal_spread(K, LAGS))
