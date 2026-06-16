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


def test_features_trough_at_last_sample_returns_nans():
    # Global min at the final sample -> no post-trough samples to find a peak.
    w = np.zeros(82)
    w[-1] = -1.0
    f = compute_waveform_features(w)
    assert np.isnan(f["t2p_ms"])


def test_classify_bimodal_splits_fsi_spn():
    rng = np.random.default_rng(0)
    narrow = rng.normal(0.20, 0.02, 60)      # FSI-like short T2P
    broad = rng.normal(0.65, 0.05, 60)       # SPN-like long T2P
    t2p = np.concatenate([narrow, broad])
    labels, info = classify_celltype(t2p)
    assert set(np.unique(labels)) <= {"FSI", "SPN", "Unclassified"}
    # threshold falls between the two modes; counts roughly balanced
    assert 0.20 < info["threshold_ms"] < 0.65
    assert (labels == "FSI").sum() == pytest.approx(60, abs=8)
    assert (labels == "SPN").sum() == pytest.approx(60, abs=8)
    assert info["delta_bic"] > 0            # 2 comps beat 1 on bimodal data


def test_classify_nan_is_unclassified():
    labels, _ = classify_celltype(np.array([0.2, np.nan, 0.7]))
    assert labels[1] == "Unclassified"


def test_classify_labels_align_with_input_length():
    t2p = np.array([0.2, 0.65, np.nan, 0.25, 0.6])
    labels, info = classify_celltype(t2p)
    assert labels.shape == t2p.shape
    assert info["n"] >= 1


def test_classify_too_few_in_window_all_unclassified():
    # Fewer than 2 in-window values -> no GMM fit; all Unclassified, NaN info.
    labels, info = classify_celltype(np.array([np.nan, np.nan]))
    assert list(labels) == ["Unclassified", "Unclassified"]
    assert info["n"] == 0
    assert np.isnan(info["threshold_ms"])
    assert np.isnan(info["delta_bic"])
