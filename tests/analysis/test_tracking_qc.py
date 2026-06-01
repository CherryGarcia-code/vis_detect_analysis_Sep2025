import pytest
from visdetect.analysis import tracking_qc as qc


def test_thresholds_present():
    assert qc.ISI_PASS == 0.75
    assert qc.ISI_WARN == 0.65
    assert qc.DEPTH_PASS_UM == 25.0
    assert qc.DEPTH_WARN_UM == 40.0
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
    assert qc.badge_depth(32.0) == "warn"
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
    assert qc.badge_depth(qc.DEPTH_PASS_UM) == "pass"     # 25.0
    assert qc.badge_waveform(qc.WAVE_PASS_R) == "pass"    # 0.95
    assert qc.badge_fr(qc.FR_CV_PASS) == "pass"           # 0.35


def test_badges_at_warn_boundary():
    # value == warn_thr should be "warn"
    assert qc.badge_isi(qc.ISI_WARN) == "warn"            # 0.65
    assert qc.badge_depth(qc.DEPTH_WARN_UM) == "warn"     # 40.0
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


def test_isi_peak_agreement_all_same():
    # All 5 sessions peak at bin 15 → perfect agreement
    h = np.zeros(50); h[15] = 0.5; h[14] = 0.25; h[16] = 0.25
    hists = [h.copy() for _ in range(5)]
    assert qc.isi_peak_agreement(hists) == pytest.approx(1.0)


def test_isi_peak_agreement_bimodal():
    # 3 sessions peak at bin 15, 2 sessions peak at bin 35 → 3/5 = 0.6 agreement
    h_a = np.zeros(50); h_a[15] = 0.5; h_a[14] = 0.25; h_a[16] = 0.25
    h_b = np.zeros(50); h_b[35] = 0.5; h_b[34] = 0.25; h_b[36] = 0.25
    hists = [h_a, h_a, h_a, h_b, h_b]
    assert qc.isi_peak_agreement(hists) == pytest.approx(0.6)


def test_isi_peak_agreement_within_tolerance():
    # Peaks at bins 14, 15, 16 all count as agreeing with mode bin 15 (±2 tolerance)
    hists = []
    for peak in [14, 15, 16, 15, 17]:
        h = np.zeros(50); h[peak] = 1.0
        hists.append(h)
    assert qc.isi_peak_agreement(hists) == pytest.approx(1.0)


def test_isi_peak_agreement_drops_nan():
    h = np.zeros(50); h[15] = 1.0
    h_nan = np.full(50, np.nan)
    hists = [h, h, h_nan, h, h_nan]
    # 3 valid sessions, all peak at 15 → 1.0
    assert qc.isi_peak_agreement(hists) == pytest.approx(1.0)


def test_isi_peak_agreement_too_few_valid_returns_nan():
    h = np.zeros(50); h[15] = 1.0
    hists = [h, np.full(50, np.nan)]
    assert np.isnan(qc.isi_peak_agreement(hists))


def test_isi_peak_agreement_empty_returns_nan():
    assert np.isnan(qc.isi_peak_agreement([]))


def test_badge_isi_peak_thresholds():
    assert qc.badge_isi_peak(0.95) == "pass"
    assert qc.badge_isi_peak(qc.ISI_PEAK_AGREE_PASS) == "pass"
    assert qc.badge_isi_peak(0.75) == "warn"
    assert qc.badge_isi_peak(qc.ISI_PEAK_AGREE_WARN) == "warn"
    assert qc.badge_isi_peak(0.30) == "fail"
    assert qc.badge_isi_peak(float("nan")) == "fail"


def test_composite_still_works_with_5_badges():
    # 5 pass → trusted
    assert qc.composite_verdict(["pass"]*5) == "trusted"
    # 4 pass + 1 warn → review
    assert qc.composite_verdict(["pass","pass","pass","pass","warn"]) == "review"
    # 3 pass + 2 warns → suspect
    assert qc.composite_verdict(["pass","pass","pass","warn","warn"]) == "suspect"
    # any fail → suspect
    assert qc.composite_verdict(["pass","pass","pass","pass","fail"]) == "suspect"


def _make_synthetic_uid(specs):
    """Build a UIDIntermediate from a list of (peak_bin, fr, wave_scale, depth) tuples."""
    recs = []
    for i, (peak_bin, fr, wave_scale, depth) in enumerate(specs):
        h = np.zeros(50, dtype=np.float32); h[int(peak_bin)] = 1.0
        wave = (np.array([0.0, 1.0, 0.0, -1.0, 0.0] * 16 + [0.0, 1.0])
                * wave_scale).astype(np.float32)
        rec = qc.SessionRecord(
            session_name=f"s{i:02d}", ks_unit_id=0, stage="Learning",
            peak_chan=0, peak_depth_um=float(depth), amplitude=1.0,
            baseline_fr_hz=float(fr),
            waveform_peak=wave,
            footprint=np.zeros((82, 17), dtype=np.float32),
            footprint_channels=np.arange(17),
            isi_hist=h, isi_centers=np.zeros(50, dtype=np.float32),
        )
        recs.append(rec)
    return qc.UIDIntermediate(
        global_uid=1, span=len(specs), has_naive_to_expert=False,
        suspect_known=False, sessions=recs,
    )


def test_session_outlier_flags_clean_uid():
    # 5 nearly-identical sessions
    specs = [(15, 5.0, 1.0, 1000.0)] * 5
    uid = _make_synthetic_uid(specs)
    f = qc.session_outlier_flags(uid)
    assert not any(f["is_outlier"])


def test_session_outlier_flags_one_bimodal_session():
    # 4 good sessions + 1 with very different ISI peak bin
    specs = [(15, 5.0, 1.0, 1000.0)] * 4 + [(35, 5.0, 1.0, 1000.0)]
    uid = _make_synthetic_uid(specs)
    f = qc.session_outlier_flags(uid)
    assert f["is_outlier"] == [False, False, False, False, True]


def test_longest_good_run_contiguous_basic():
    # 0..2 good, 3 bad, 4..7 good (length 4) → best (4,8)
    flags = [False, False, False, True, False, False, False, False]
    assert qc._longest_good_run_contiguous(flags) == (4, 8)


def test_longest_good_run_contiguous_all_bad_returns_zero():
    assert qc._longest_good_run_contiguous([True, True, True]) == (0, 0)


def test_longest_good_run_contiguous_all_good_returns_full():
    assert qc._longest_good_run_contiguous([False, False, False, False]) == (0, 4)


def test_find_stable_subset_trims_outlier_at_end():
    # Index 4 has a different ISI peak (bin 35 vs bin 15) — soft outlier.
    # Skip-able algorithm: no hard outliers → one span [0..4]; kept=[0,1,2,3],
    # skipped=[4]; ISI corr of kept (all bin-15) = 1.0 → qualifies.
    specs = [(15, 5.0, 1.0, 1000.0)] * 4 + [(35, 5.0, 1.0, 1000.0)]
    uid = _make_synthetic_uid(specs)
    out = qc.find_stable_subset(uid)
    assert out["kept_indices"] == [0, 1, 2, 3]
    assert out["skipped_indices"] == [4]
    assert out["dropped_indices"] == []
    assert out["trimmed_span"] == 4


def test_find_stable_subset_picks_longer_run():
    # Soft outliers at indices 0 and 3 (different ISI). No hard outliers →
    # one span [0..6]. kept=[1,2,4,5,6], skipped=[0,3]; ISI corr of kept
    # (all bin-15) = 1.0 → qualifies. All 5 good sessions are kept.
    specs = ([(35, 5.0, 1.0, 1000.0)] + [(15, 5.0, 1.0, 1000.0)] * 2 +
             [(35, 5.0, 1.0, 1000.0)] + [(15, 5.0, 1.0, 1000.0)] * 3)
    uid = _make_synthetic_uid(specs)
    out = qc.find_stable_subset(uid)
    assert out["kept_indices"] == [1, 2, 4, 5, 6]
    assert out["skipped_indices"] == [0, 3]
    assert out["dropped_indices"] == []
    assert out["trimmed_span"] == 5


# ─── Functional-response stability (6th badge) ────────────────────────

def test_baseline_psth_corr_identical_returns_one():
    psth = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    psths = [psth.copy() for _ in range(5)]
    assert qc.baseline_psth_corr(psths) == pytest.approx(1.0, abs=1e-6)


def test_baseline_psth_corr_handles_magnitude_scaling():
    # Same shape, different magnitudes — Pearson r should still be 1
    base = np.array([1.0, 3.0, 5.0, 3.0, 1.0])
    psths = [base, base * 2.0, base * 0.5]
    assert qc.baseline_psth_corr(psths) == pytest.approx(1.0, abs=1e-6)


def test_baseline_psth_corr_flipped_polarity():
    base = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    psths = [base, -base, base]
    # pairs: (a,-a)=-1, (a,a)=+1, (-a,a)=-1 → median = -1
    assert qc.baseline_psth_corr(psths) == pytest.approx(-1.0, abs=1e-6)


def test_baseline_psth_corr_drops_none_sessions():
    base = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    psths = [base, None, base, None, base]
    # 3 valid sessions, all identical → r = 1
    assert qc.baseline_psth_corr(psths) == pytest.approx(1.0, abs=1e-6)


def test_baseline_psth_corr_too_few_returns_nan():
    base = np.array([1.0, 2.0, 3.0])
    assert np.isnan(qc.baseline_psth_corr([base]))
    assert np.isnan(qc.baseline_psth_corr([base, None]))


def test_baseline_psth_corr_zero_variance_drops_session():
    # Session with zero-variance PSTH (flat all-zeros) cannot be correlated → drop it
    flat = np.zeros(5)
    base = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    # 2 valid (after dropping flat), both identical → r = 1
    assert qc.baseline_psth_corr([base, flat, base]) == pytest.approx(1.0, abs=1e-6)


def test_badge_func_resp_thresholds():
    assert qc.badge_func_resp(0.50) == "pass"
    assert qc.badge_func_resp(qc.FUNC_RESP_PASS) == "pass"
    assert qc.badge_func_resp(0.25) == "warn"
    assert qc.badge_func_resp(qc.FUNC_RESP_WARN) == "warn"
    assert qc.badge_func_resp(0.05) == "fail"


def test_badge_func_resp_nan_is_pass():
    # NaN means "no measurable baseline modulation" — not evidence of matching failure.
    assert qc.badge_func_resp(float("nan")) == "pass"


def test_baseline_psth_corr_flat_returns_nan():
    # All PSTHs have std < FUNC_RESP_MIN_PSTH_STD → modulation gate returns NaN
    flat_a = np.ones(40) * 5.0 + np.random.default_rng(0).standard_normal(40) * 0.1
    flat_b = np.ones(40) * 5.0 + np.random.default_rng(1).standard_normal(40) * 0.1
    flat_c = np.ones(40) * 5.0 + np.random.default_rng(2).standard_normal(40) * 0.1
    assert np.isnan(qc.baseline_psth_corr([flat_a, flat_b, flat_c]))


def test_composite_with_6_badges():
    assert qc.composite_verdict(["pass"] * 6) == "trusted"
    assert qc.composite_verdict(["pass"]*5 + ["warn"]) == "review"
    assert qc.composite_verdict(["pass"]*4 + ["warn"]*2) == "suspect"
    assert qc.composite_verdict(["pass"]*5 + ["fail"]) == "suspect"


# ─── Cross-session probe drift correction ─────────────────────────────

def test_depth_std_um_corrected_basic():
    # 3 sessions with depths 100, 110, 120 and matching drift offsets 0, 10, 20
    # Corrected depths: 100, 100, 100 → std = 0
    from visdetect.analysis import tracking_qc as qc_local
    recs = []
    for i, (sess, depth) in enumerate([("a", 100.0), ("b", 110.0), ("c", 120.0)]):
        recs.append(qc_local.SessionRecord(
            session_name=sess, ks_unit_id=0, stage="Learning", peak_chan=0,
            peak_depth_um=depth, amplitude=1.0, baseline_fr_hz=5.0,
            waveform_peak=np.zeros(82, np.float32),
            footprint=np.zeros((82, 17), np.float32),
            footprint_channels=np.arange(17),
            isi_hist=np.zeros(50, np.float32),
            isi_centers=np.zeros(50, np.float32),
        ))
    uid = qc_local.UIDIntermediate(
        global_uid=1, span=3, has_naive_to_expert=False, suspect_known=False,
        sessions=recs,
    )
    offsets = {"a": 0.0, "b": 10.0, "c": 20.0}
    assert qc.depth_std_um_corrected(uid, offsets) == pytest.approx(0.0, abs=1e-9)


def test_depth_std_um_corrected_skips_nan_offset_sessions():
    from visdetect.analysis import tracking_qc as qc_local
    recs = []
    for sess, depth in [("a", 100.0), ("b", 110.0), ("c", 120.0)]:
        recs.append(qc_local.SessionRecord(
            session_name=sess, ks_unit_id=0, stage="Learning", peak_chan=0,
            peak_depth_um=depth, amplitude=1.0, baseline_fr_hz=5.0,
            waveform_peak=np.zeros(82, np.float32),
            footprint=np.zeros((82, 17), np.float32),
            footprint_channels=np.arange(17),
            isi_hist=np.zeros(50, np.float32),
            isi_centers=np.zeros(50, np.float32),
        ))
    uid = qc_local.UIDIntermediate(
        global_uid=1, span=3, has_naive_to_expert=False, suspect_known=False,
        sessions=recs,
    )
    # Session 'b' has NaN offset → dropped. Remaining sessions a (corrected=100) and c (corrected=100) → std=0.
    offsets = {"a": 0.0, "b": float("nan"), "c": 20.0}
    assert qc.depth_std_um_corrected(uid, offsets) == pytest.approx(0.0, abs=1e-9)


def test_depth_std_um_corrected_too_few_returns_nan():
    from visdetect.analysis import tracking_qc as qc_local
    recs = [qc_local.SessionRecord(
        session_name="a", ks_unit_id=0, stage="Learning", peak_chan=0,
        peak_depth_um=100.0, amplitude=1.0, baseline_fr_hz=5.0,
        waveform_peak=np.zeros(82, np.float32),
        footprint=np.zeros((82, 17), np.float32),
        footprint_channels=np.arange(17),
        isi_hist=np.zeros(50, np.float32),
        isi_centers=np.zeros(50, np.float32),
    )]
    uid = qc_local.UIDIntermediate(
        global_uid=1, span=1, has_naive_to_expert=False, suspect_known=False,
        sessions=recs,
    )
    assert np.isnan(qc.depth_std_um_corrected(uid, {"a": 0.0}))


# ─── ISI histogram cross-session correlation (5th badge in composite) ─

def test_baseline_isi_hist_corr_identical_returns_one():
    h = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
    hists = [h.copy() for _ in range(5)]
    assert qc.baseline_isi_hist_corr(hists) == pytest.approx(1.0, abs=1e-6)


def test_baseline_isi_hist_corr_handles_magnitude_scaling():
    # Same shape, different magnitudes — Pearson r should still be 1.
    base = np.array([1.0, 3.0, 5.0, 3.0, 1.0])
    hists = [base, base * 2.0, base * 0.5]
    assert qc.baseline_isi_hist_corr(hists) == pytest.approx(1.0, abs=1e-6)


def test_baseline_isi_hist_corr_flipped_polarity_median():
    base = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    # pairs: (a, -a) = -1, (a, a) = +1, (-a, a) = -1 → median = -1
    hists = [base, -base, base]
    assert qc.baseline_isi_hist_corr(hists) == pytest.approx(-1.0, abs=1e-6)


def test_baseline_isi_hist_corr_drops_none_sessions():
    base = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    hists = [base, None, base, None, base]
    # 3 valid sessions, all identical → r = 1
    assert qc.baseline_isi_hist_corr(hists) == pytest.approx(1.0, abs=1e-6)


def test_baseline_isi_hist_corr_drops_flat_sessions():
    flat = np.zeros(5)
    base = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    # 2 valid (after dropping flat), both identical → r = 1
    assert qc.baseline_isi_hist_corr([base, flat, base]) == pytest.approx(1.0, abs=1e-6)


def test_baseline_isi_hist_corr_too_few_returns_nan():
    base = np.array([1.0, 2.0, 3.0])
    assert np.isnan(qc.baseline_isi_hist_corr([base]))
    assert np.isnan(qc.baseline_isi_hist_corr([base, None]))


def test_badge_isi_hist_corr_thresholds():
    assert qc.badge_isi_hist_corr(0.95) == "pass"
    assert qc.badge_isi_hist_corr(qc.ISI_HIST_CORR_PASS) == "pass"     # 0.85 boundary
    assert qc.badge_isi_hist_corr(0.75) == "warn"
    assert qc.badge_isi_hist_corr(qc.ISI_HIST_CORR_WARN) == "warn"     # 0.65 boundary
    assert qc.badge_isi_hist_corr(0.40) == "fail"
    assert qc.badge_isi_hist_corr(float("nan")) == "fail"


def test_session_outlier_flags_unknown_stage_is_outlier():
    """A session with stage='Unknown' is unconditionally flagged outlier."""
    rec_good = qc.SessionRecord(
        session_name="s00", ks_unit_id=0, stage="Learning", peak_chan=10,
        peak_depth_um=100.0, amplitude=50.0, baseline_fr_hz=5.0,
        waveform_peak=np.array([0.0, -1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        footprint=np.zeros((5, 17), dtype=np.float32),
        footprint_channels=np.arange(17),
        isi_hist=np.array([0.1, 0.5, 0.3, 0.1] + [0.0] * 46, dtype=np.float32),
        isi_centers=np.zeros(50, dtype=np.float32),
    )
    rec_unknown = qc.SessionRecord(
        session_name="s01", ks_unit_id=0, stage="Unknown", peak_chan=10,
        peak_depth_um=100.0, amplitude=50.0, baseline_fr_hz=5.0,
        waveform_peak=np.array([0.0, -1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        footprint=np.zeros((5, 17), dtype=np.float32),
        footprint_channels=np.arange(17),
        isi_hist=np.array([0.1, 0.5, 0.3, 0.1] + [0.0] * 46, dtype=np.float32),
        isi_centers=np.zeros(50, dtype=np.float32),
    )
    uid = qc.UIDIntermediate(
        global_uid=1, span=2, has_naive_to_expert=False, suspect_known=False,
        sessions=[rec_good, rec_good, rec_unknown, rec_good],
    )
    flags = qc.session_outlier_flags(uid)
    assert "unknown_stage" in flags
    assert flags["unknown_stage"] == [False, False, True, False]
    assert flags["is_outlier"] == [False, False, True, False]


def test_find_stable_subset_drops_unknown_sessions():
    """Unknown-stage sessions are soft outliers; skip-able algorithm skips them
    rather than breaking the run, provided ISI consistency holds."""
    rec_good = qc.SessionRecord(
        session_name="s00", ks_unit_id=0, stage="Learning", peak_chan=10,
        peak_depth_um=100.0, amplitude=50.0, baseline_fr_hz=5.0,
        waveform_peak=np.array([0.0, -1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        footprint=np.zeros((5, 17), dtype=np.float32),
        footprint_channels=np.arange(17),
        isi_hist=np.array([0.1, 0.5, 0.3, 0.1] + [0.0] * 46, dtype=np.float32),
        isi_centers=np.zeros(50, dtype=np.float32),
    )
    rec_unknown = qc.SessionRecord(
        session_name="s01", ks_unit_id=0, stage="Unknown", peak_chan=10,
        peak_depth_um=100.0, amplitude=50.0, baseline_fr_hz=5.0,
        waveform_peak=np.array([0.0, -1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        footprint=np.zeros((5, 17), dtype=np.float32),
        footprint_channels=np.arange(17),
        isi_hist=np.array([0.1, 0.5, 0.3, 0.1] + [0.0] * 46, dtype=np.float32),
        isi_centers=np.zeros(50, dtype=np.float32),
    )
    # Sequence: good, good, unknown, good, good, good.
    # All have identical ISI hists; no hard outliers → one span [0..5].
    # kept=[0,1,3,4,5], skipped=[2] (unknown is a soft outlier).
    uid = qc.UIDIntermediate(
        global_uid=1, span=6, has_naive_to_expert=False, suspect_known=False,
        sessions=[rec_good, rec_good, rec_unknown, rec_good, rec_good, rec_good],
    )
    stable = qc.find_stable_subset(uid)
    assert stable["kept_indices"] == [0, 1, 3, 4, 5]
    assert stable["skipped_indices"] == [2]
    assert stable["dropped_indices"] == []


def test_session_outlier_flags_classifies_hard_vs_soft():
    """is_hard_outlier = wave OR depth (independent of composite is_outlier rule).
    is_soft_outlier = is_outlier AND NOT is_hard_outlier.

    Note: is_hard_outlier and is_outlier are independent signals — a session
    with ONLY a wave flag (strikes=1) is is_hard_outlier=True but NOT
    is_outlier (the existing composite rule requires isi_peak OR strikes>=2
    OR unknown_stage). The new algorithm uses both flags separately."""
    h_clean = np.zeros(50, dtype=np.float32); h_clean[15] = 1.0
    h_bimodal = np.zeros(50, dtype=np.float32); h_bimodal[35] = 1.0
    wave_clean = np.array([0.0, 1.0, 0.0, -1.0, 0.0] * 16 + [0.0, 1.0], dtype=np.float32)
    wave_flipped = -wave_clean
    def mk_rec(name, stage, peak_hist, fr, wave, depth):
        return qc.SessionRecord(
            session_name=name, ks_unit_id=0, stage=stage,
            peak_chan=0, peak_depth_um=float(depth), amplitude=1.0,
            baseline_fr_hz=float(fr), waveform_peak=wave,
            footprint=np.zeros((82, 17), dtype=np.float32),
            footprint_channels=np.arange(17),
            isi_hist=peak_hist, isi_centers=np.zeros(50, dtype=np.float32),
        )
    sessions = [
        mk_rec("s00", "Learning", h_clean,    5.0, wave_clean,   1000.0),  # clean
        mk_rec("s01", "Learning", h_clean,    5.0, wave_flipped, 1000.0),  # wave-only: HARD, NOT is_outlier
        mk_rec("s02", "Learning", h_bimodal,  5.0, wave_clean,   1000.0),  # isi_peak alone: SOFT, is_outlier
        mk_rec("s03", "Learning", h_clean,    5.0, wave_clean,   1000.0),  # clean
        mk_rec("s04", "Unknown",  h_clean,    5.0, wave_clean,   1000.0),  # unknown_stage: SOFT, is_outlier
        mk_rec("s05", "Learning", h_clean,    5.0, wave_clean,   2000.0),  # depth-only: HARD, NOT is_outlier
    ]
    uid = qc.UIDIntermediate(
        global_uid=1, span=6, has_naive_to_expert=False,
        suspect_known=False, sessions=sessions,
    )
    f = qc.session_outlier_flags(uid)
    assert "is_hard_outlier" in f
    assert "is_soft_outlier" in f
    # s01: wave-only triggers is_hard_outlier but NOT is_outlier (strikes=1)
    # s02: isi_peak alone triggers is_outlier directly; soft (no wave/depth)
    # s04: unknown_stage triggers is_outlier directly; soft (no wave/depth)
    # s05: depth-only triggers is_hard_outlier but NOT is_outlier (strikes=1)
    assert f["is_hard_outlier"] == [False, True,  False, False, False, True]
    assert f["is_soft_outlier"] == [False, False, True,  False, True,  False]
    assert f["is_outlier"]      == [False, False, True,  False, True,  False]


# ─── Skip-able longest_good_run ───────────────────────────────────────

def _identical_isi_hists(n: int) -> list:
    """Helper: n copies of a fixed peak-15 log-ISI histogram. Guarantees
    set-wide isi_hist_corr == 1.0 (passes gate trivially)."""
    h = np.zeros(50, dtype=np.float32); h[15] = 0.5; h[14] = 0.25; h[16] = 0.25
    return [h.copy() for _ in range(n)]


def test_longest_good_run_skips_soft_with_high_consistency():
    """Sequence [G, G, S, G, G] with identical ISI hists → all 4 good kept,
    soft outlier at index 2 is skipped, NO sessions dropped (no hard
    outliers). Set-wide isi_hist_corr = 1.0 passes 0.85 gate trivially."""
    is_outlier      = [False, False, True,  False, False]
    is_hard_outlier = [False, False, False, False, False]
    hists = _identical_isi_hists(5)
    out = qc.longest_good_run(is_outlier, is_hard_outlier, hists)
    assert out["kept_indices"] == [0, 1, 3, 4]
    assert out["skipped_indices"] == [2]


def test_longest_good_run_falls_back_when_consistency_fails():
    """Sequence [G, G, S, G, G] where the two good halves have DIFFERENT
    ISI shapes → set-wide isi_hist_corr fails 0.85 gate → falls back to
    longest contiguous all-good run = [0,1] (length 2; ties broken by
    first-encountered)."""
    is_outlier      = [False, False, True,  False, False]
    is_hard_outlier = [False, False, False, False, False]
    h_a = np.zeros(50, dtype=np.float32); h_a[10] = 1.0
    h_b = np.zeros(50, dtype=np.float32); h_b[40] = 1.0
    # Soft outlier at index 2 (any shape — will be skipped); halves divergent
    hists = [h_a.copy(), h_a.copy(), h_a.copy(), h_b.copy(), h_b.copy()]
    out = qc.longest_good_run(is_outlier, is_hard_outlier, hists)
    # Set [0,1,3,4]: pairs (0,1)=1, (0,3)=-1, (0,4)=-1, (1,3)=-1, (1,4)=-1, (3,4)=1
    # median = -1 < 0.85 → fallback
    # Fallback picks first longest contiguous good run = [0,1]
    assert out["kept_indices"] == [0, 1]
    assert out["skipped_indices"] == []


def test_longest_good_run_never_skips_hard_outliers():
    """Sequence [G, G, H, G, G] where H is a HARD outlier → must NEVER appear
    in kept or skipped. Result is one of [0,1] or [3,4] (both length 2)."""
    is_outlier      = [False, False, True, False, False]
    is_hard_outlier = [False, False, True, False, False]
    hists = _identical_isi_hists(5)
    out = qc.longest_good_run(is_outlier, is_hard_outlier, hists)
    assert 2 not in out["kept_indices"]
    assert 2 not in out["skipped_indices"]
    # Tie-break: largest kept_set; ties → longest span (kept+skipped); ties
    # → earliest start. [0,1] starts earlier, so it wins on the last tie-break.
    assert out["kept_indices"] == [0, 1]
    assert out["skipped_indices"] == []


def test_find_stable_subset_returns_skipped_indices():
    """find_stable_subset exposes skipped_indices and redefines dropped_indices
    to exclude skipped. Set: [Learning, Learning, Unknown, Learning, Learning]
    with identical ISI → kept=[0,1,3,4], skipped=[2], dropped=[]."""
    h = np.zeros(50, dtype=np.float32); h[15] = 0.5; h[14] = 0.25; h[16] = 0.25
    wave = np.array([0.0, 1.0, 0.0, -1.0, 0.0] * 16 + [0.0, 1.0], dtype=np.float32)
    def mk_rec(name, stage):
        return qc.SessionRecord(
            session_name=name, ks_unit_id=0, stage=stage,
            peak_chan=0, peak_depth_um=1000.0, amplitude=1.0,
            baseline_fr_hz=5.0, waveform_peak=wave,
            footprint=np.zeros((82, 17), dtype=np.float32),
            footprint_channels=np.arange(17),
            isi_hist=h.copy(), isi_centers=np.zeros(50, dtype=np.float32),
        )
    sessions = [mk_rec(f"s{i:02d}", "Unknown" if i == 2 else "Learning")
                for i in range(5)]
    uid = qc.UIDIntermediate(
        global_uid=1, span=5, has_naive_to_expert=False,
        suspect_known=False, sessions=sessions,
    )
    out = qc.find_stable_subset(uid)
    assert "skipped_indices" in out
    assert out["kept_indices"]    == [0, 1, 3, 4]
    assert out["skipped_indices"] == [2]
    assert out["dropped_indices"] == []
    # Sanity: union covers all sessions and the three sets are disjoint
    union = set(out["kept_indices"]) | set(out["skipped_indices"]) | set(out["dropped_indices"])
    assert union == set(range(5))
    assert (set(out["kept_indices"]) & set(out["skipped_indices"])) == set()
    assert (set(out["kept_indices"]) & set(out["dropped_indices"])) == set()
    assert (set(out["skipped_indices"]) & set(out["dropped_indices"])) == set()


def test_longest_good_run_all_hard_outliers_returns_empty():
    """All sessions are hard outliers → spans list is empty → fallback called
    on is_outlier=[T,T,T] → contiguous-good-run returns (0,0) → empty result."""
    is_outlier      = [True, True, True]
    is_hard_outlier = [True, True, True]
    hists = _identical_isi_hists(3)
    out = qc.longest_good_run(is_outlier, is_hard_outlier, hists)
    assert out["kept_indices"] == []
    assert out["skipped_indices"] == []


def test_longest_good_run_fallback_excludes_hard_outliers():
    """Fallback path must NOT include hard outliers in kept_indices, even if
    is_outlier=False for them (wave-only/depth-only has strikes=1, fails the
    composite rule so is_outlier stays False).

    Setup: [G_a, G_b, H_wave] — 3 sessions where index 2 is a wave-only hard
    outlier with is_hard_outlier=True but is_outlier=False.  G_a and G_b have
    divergent ISI histograms so the 2-session span [0,1] fails the consistency
    gate, meaning no span produces a valid best_kept and the code falls through
    to the Step-6 fallback.

    Without the fix, the fallback calls _longest_good_run_contiguous on
    is_outlier=[F, F, F] and returns (0, 3) — INCLUDING the hard outlier at
    index 2.  With the fix, the fallback unions is_outlier with is_hard_outlier
    giving effective_outlier=[F, F, T], so the contiguous-good run ends at
    index 2 and kept_indices = [0, 1], correctly excluding index 2."""
    h_a = np.zeros(50, dtype=np.float32); h_a[10] = 1.0   # peak at bin 10
    h_b = np.zeros(50, dtype=np.float32); h_b[40] = 1.0   # peak at bin 40 — divergent

    is_outlier      = [False, False, False]   # composite False everywhere
    is_hard_outlier = [False, False, True]    # index 2 is a wave-only hard outlier

    hists = [h_a.copy(), h_b.copy(), h_a.copy()]

    out = qc.longest_good_run(is_outlier, is_hard_outlier, hists)

    assert 2 not in out["kept_indices"], (
        f"Hard outlier at index 2 leaked into kept_indices: {out}"
    )
    assert 2 not in out["skipped_indices"], (
        f"Hard outlier at index 2 leaked into skipped_indices: {out}"
    )


# ─── Option B: isi_hist_corr auto-pass ────────────────────────────────

def test_apply_isi_autopass_promotes_when_threshold_met():
    """ISI 0.97 + no wave/depth fail + suspect verdict → trusted."""
    assert qc.apply_isi_autopass("suspect", 0.97, "pass", "pass") == "trusted"
    assert qc.apply_isi_autopass("review",  0.97, "warn", "warn") == "trusted"


def test_apply_isi_autopass_already_trusted_is_noop():
    """Already-trusted verdicts pass through unchanged when conditions hold (function does not demote)."""
    assert qc.apply_isi_autopass("trusted", 0.97, "pass", "pass") == "trusted"


def test_apply_isi_autopass_blocks_on_wave_fail():
    """High ISI + wave FAIL → verdict unchanged (hard biophysical block)."""
    assert qc.apply_isi_autopass("suspect", 0.99, "fail", "pass") == "suspect"
    assert qc.apply_isi_autopass("review",  0.99, "fail", "warn") == "review"


def test_apply_isi_autopass_blocks_on_depth_fail():
    """High ISI + depth FAIL → verdict unchanged."""
    assert qc.apply_isi_autopass("suspect", 0.99, "pass", "fail") == "suspect"
    assert qc.apply_isi_autopass("review",  0.99, "warn", "fail") == "review"


def test_apply_isi_autopass_below_threshold_no_change():
    """ISI 0.94 (just below 0.95 threshold) → no promotion."""
    assert qc.apply_isi_autopass("suspect", 0.94, "pass", "pass") == "suspect"
    assert qc.apply_isi_autopass("review",  0.85, "pass", "pass") == "review"


def test_apply_isi_autopass_nan_no_change():
    """NaN ISI → no promotion (the threshold check fails)."""
    assert qc.apply_isi_autopass("suspect", float("nan"), "pass", "pass") == "suspect"
    assert qc.apply_isi_autopass("review",  float("nan"), "pass", "pass") == "review"
