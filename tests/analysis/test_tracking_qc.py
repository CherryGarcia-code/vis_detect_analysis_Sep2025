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


def test_badge_isi():
    assert qc.badge_isi(0.91) == "pass"
    assert qc.badge_isi(0.70) == "warn"
    assert qc.badge_isi(0.28) == "fail"
    assert qc.badge_isi(float("nan")) == "fail"


def test_badge_depth():
    assert qc.badge_depth(8.0) == "pass"
    assert qc.badge_depth(20.0) == "warn"
    assert qc.badge_depth(45.0) == "fail"
    assert qc.badge_depth(float("nan")) == "fail"


def test_badge_waveform():
    assert qc.badge_waveform(0.97) == "pass"
    assert qc.badge_waveform(0.92) == "warn"
    assert qc.badge_waveform(0.50) == "fail"


def test_badge_fr():
    assert qc.badge_fr(0.20) == "pass"
    assert qc.badge_fr(0.45) == "warn"
    assert qc.badge_fr(0.80) == "fail"


def test_composite_all_pass_is_trusted():
    assert qc.composite_verdict(["pass", "pass", "pass", "pass"]) == "trusted"


def test_composite_one_warn_is_review():
    assert qc.composite_verdict(["pass", "warn", "pass", "pass"]) == "review"


def test_composite_two_warns_is_suspect():
    assert qc.composite_verdict(["pass", "warn", "warn", "pass"]) == "suspect"


def test_composite_any_fail_is_suspect():
    assert qc.composite_verdict(["pass", "pass", "pass", "fail"]) == "suspect"
    assert qc.composite_verdict(["pass", "warn", "fail", "pass"]) == "suspect"


def test_badges_at_pass_boundary():
    # value == pass_thr should be "pass" (>= for high, <= for low)
    assert qc.badge_isi(qc.ISI_PASS) == "pass"            # 0.75
    assert qc.badge_depth(qc.DEPTH_PASS_UM) == "pass"     # 15.0
    assert qc.badge_waveform(qc.WAVE_PASS_R) == "pass"    # 0.95
    assert qc.badge_fr(qc.FR_CV_PASS) == "pass"           # 0.35


def test_badges_at_warn_boundary():
    # value == warn_thr should be "warn"
    assert qc.badge_isi(qc.ISI_WARN) == "warn"            # 0.65
    assert qc.badge_depth(qc.DEPTH_WARN_UM) == "warn"     # 30.0
    assert qc.badge_waveform(qc.WAVE_WARN_R) == "warn"    # 0.90
    assert qc.badge_fr(qc.FR_CV_WARN) == "warn"           # 0.60


def test_badge_waveform_nan_is_fail():
    assert qc.badge_waveform(float("nan")) == "fail"


def test_badge_fr_nan_is_fail():
    assert qc.badge_fr(float("nan")) == "fail"


import pandas as pd
import tempfile
from pathlib import Path


def test_load_isi_scores(tmp_path):
    csv = tmp_path / "track_validation_stats.csv"
    csv.write_text(
        "global_uid,mean,median,min,count,span,nonmatched_rank_pct\n"
        "334,0.73,0.91,-0.39,725,27,82.9\n"
        "779,0.30,0.28,-0.40,500,15,5.0\n"
    )
    scores = qc.load_isi_scores(csv)
    assert scores[334] == pytest.approx(0.91)
    assert scores[779] == pytest.approx(0.28)
    assert scores.get(9999, float("nan")) != scores.get(9999, float("nan"))  # NaN sentinel for missing


def test_isi_log_histogram():
    rng = np.random.default_rng(0)
    spike_times = np.sort(rng.exponential(0.1, size=1000).cumsum())
    h, centers = qc.isi_log_histogram(spike_times, n_bins=50)
    assert h.shape == (50,)
    assert centers.shape == (50,)
    assert h.sum() == pytest.approx(1.0, rel=1e-6)


def test_isi_log_histogram_too_few_spikes_returns_nans():
    h, centers = qc.isi_log_histogram(np.array([0.1, 0.2]), n_bins=50)
    assert np.all(np.isnan(h))
    assert centers.shape == (50,)


def test_extract_peak_channel_picks_max_amplitude():
    # raw waveform shape: (n_samples, n_channels, n_cv_halves)
    n_samp, n_ch, n_cv = 82, 384, 2
    waveforms = np.zeros((n_samp, n_ch, n_cv), dtype=np.float32)
    # channel 17 has a clean spike
    waveforms[30:40, 17, :] = -1.5
    waveforms[40, 17, :] = 0.5
    mean_wave = waveforms.mean(axis=-1)  # (n_samp, n_ch)
    peak_chan = qc.extract_peak_channel(mean_wave)
    assert peak_chan == 17


def test_extract_footprint_centered_on_peak():
    n_samp, n_ch = 82, 384
    mean_wave = np.zeros((n_samp, n_ch), dtype=np.float32)
    mean_wave[:, 100] = np.linspace(-1.0, 1.0, n_samp)
    fp, channels = qc.extract_footprint(mean_wave, peak_chan=100, halfwidth=8)
    assert fp.shape == (n_samp, 17)        # 2*8 + 1
    assert channels.tolist() == list(range(92, 109))


def test_extract_footprint_clips_at_probe_edge():
    n_samp, n_ch = 82, 384
    mean_wave = np.zeros((n_samp, n_ch), dtype=np.float32)
    fp, channels = qc.extract_footprint(mean_wave, peak_chan=2, halfwidth=8)
    assert fp.shape[1] == 11               # 0..10 inclusive
    assert channels.tolist() == list(range(0, 11))


def test_extract_footprint_clips_at_top_edge():
    n_samp, n_ch = 82, 384
    mean_wave = np.zeros((n_samp, n_ch), dtype=np.float32)
    fp, channels = qc.extract_footprint(mean_wave, peak_chan=n_ch - 1, halfwidth=8)
    # peak at 383, halfwidth 8 → channels 375..383 inclusive (9 channels)
    assert fp.shape[1] == 9
    assert channels.tolist() == list(range(n_ch - 9, n_ch))


def test_extract_peak_channel_all_zero_returns_first_channel():
    # Document current behaviour: argmax of all-zero ptp array → 0.
    # Callers should treat a peak_chan of 0 with zero amplitude as a dead-unit sentinel.
    mean_wave = np.zeros((82, 384), dtype=np.float32)
    assert qc.extract_peak_channel(mean_wave) == 0
