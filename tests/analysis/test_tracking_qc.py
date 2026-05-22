import pytest
from visdetect.analysis import tracking_qc as qc


def test_thresholds_present():
    assert qc.ISI_PASS == 0.75
    assert qc.ISI_WARN == 0.65
    assert qc.DEPTH_PASS_UM == 15.0
    assert qc.DEPTH_WARN_UM == 30.0
    assert qc.WAVE_PASS_R == 0.95
    assert qc.WAVE_WARN_R == 0.90
    assert qc.FR_CV_PASS == 0.35
    assert qc.FR_CV_WARN == 0.60


def test_change_size_pools():
    assert qc.BIG_POOL == {2.0, 4.0}
    assert qc.SMALL_POOL == {1.25, 1.35}


import numpy as np


def test_depth_std_um_basic():
    depths = np.array([100.0, 105.0, 95.0, 100.0])
    assert qc.depth_std_um(depths) == pytest.approx(3.5355, rel=1e-3)


def test_depth_std_um_handles_nans():
    depths = np.array([100.0, np.nan, 110.0, np.nan])
    assert qc.depth_std_um(depths) == pytest.approx(5.0, rel=1e-3)


def test_depth_std_um_empty_returns_nan():
    assert np.isnan(qc.depth_std_um(np.array([])))


def test_waveform_corr_identical_returns_one():
    waves = np.tile(np.array([0.0, 1.0, 0.0, -1.0, 0.0]), (4, 1)).astype(float)
    assert qc.waveform_corr(waves) == pytest.approx(1.0, rel=1e-6)


def test_waveform_corr_normalizes_then_correlates():
    w1 = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
    w2 = w1 * 10.0          # same shape, larger amplitude
    w3 = -w1                # flipped polarity
    waves = np.stack([w1, w2, w3])
    # pairs: (1,2)=+1, (1,3)=-1, (2,3)=-1 → mean = -1/3
    assert qc.waveform_corr(waves) == pytest.approx(-1.0 / 3.0, abs=1e-6)


def test_waveform_corr_too_few_returns_nan():
    waves = np.array([[1.0, 2.0, 3.0]])     # only one session
    assert np.isnan(qc.waveform_corr(waves))


def test_fr_cv_basic():
    rates = np.array([10.0, 12.0, 8.0, 11.0, 9.0])
    expected = np.std(rates, ddof=0) / np.mean(rates)
    assert qc.fr_cv(rates) == pytest.approx(expected, rel=1e-6)


def test_fr_cv_zero_mean_returns_nan():
    assert np.isnan(qc.fr_cv(np.array([0.0, 0.0, 0.0])))


def test_fr_cv_handles_nans():
    rates = np.array([10.0, np.nan, 12.0])
    assert qc.fr_cv(rates) == pytest.approx(np.std([10.0, 12.0]) / 11.0, rel=1e-3)
