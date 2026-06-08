import numpy as np
import pytest
from visdetect.analysis import track_curation as tc


def test_partitioned_isi_hists_disjoint_and_valid():
    rng = np.random.default_rng(0)
    spikes = np.cumsum(rng.exponential(0.05, size=4000))   # stationary-ish train
    cur, hold = tc.partitioned_isi_hists(spikes)
    assert cur.shape == (50,) and hold.shape == (50,)
    assert np.isfinite(cur).all() and np.isfinite(hold).all()
    # Same underlying distribution -> the two partitions correlate strongly
    r = np.corrcoef(cur, hold)[0, 1]
    assert r > 0.8


def test_partitioned_isi_hists_too_few_spikes_returns_nan():
    cur, hold = tc.partitioned_isi_hists(np.array([0.1, 0.2, 0.3]))
    assert np.isnan(cur).all() and np.isnan(hold).all()


from visdetect.analysis import tracking_qc as qc
from visdetect.utils.synthetic import make_synthetic_session


def test_extract_unit_psths_restrict_trials_subsets():
    sess = make_synthetic_session(n_trials=40, n_clusters=3, seed=1)
    full = qc.extract_unit_psths(sess, ks_unit_id=0)
    restricted = qc.extract_unit_psths(sess, ks_unit_id=0, restrict_trials={0, 1, 2, 3, 4})
    # baseline_on uses all trials when unrestricted; restricting lowers n_trials
    assert restricted["baseline_on"][2] <= 5
    assert full["baseline_on"][2] >= restricted["baseline_on"][2]


def test_extract_unit_psths_empty_restrict_returns_none():
    sess = make_synthetic_session(n_trials=40, n_clusters=3, seed=1)
    out = qc.extract_unit_psths(sess, ks_unit_id=0, restrict_trials=set())
    assert out["baseline_on"] == (None, None, 0)


import os


def _write_toy_waveforms(root, session, kid, n_samples=82, n_ch=20, seed=0):
    """Write a UM-style RawWaveforms npy + channel_positions for one unit."""
    rng = np.random.default_rng(seed)
    sess_dir = os.path.join(str(root), session, "RawWaveforms")
    os.makedirs(sess_dir, exist_ok=True)
    wf = rng.standard_normal((n_samples, n_ch, 2)).astype(np.float32)
    # give channel 5 a clear peak so peak-channel detection is deterministic
    wf[40, 5, :] += 30.0
    wf[20, 5, :] -= 30.0
    np.save(os.path.join(sess_dir, f"Unit{kid}_RawSpikes.npy"), wf)
    pos = np.zeros((n_ch, 2), dtype=np.float32)
    pos[:, 1] = np.arange(n_ch) * 20.0      # y-depth 20 um spacing
    np.save(os.path.join(str(root), session, "channel_positions.npy"), pos)


def test_extract_curation_feature_assembles_record(tmp_path):
    sess = make_synthetic_session(n_trials=40, n_clusters=3, seed=2)
    _write_toy_waveforms(tmp_path, "07072025", kid=0)
    cp = qc.load_channel_positions(tmp_path, "07072025")
    feat = tc.extract_curation_feature(
        sess, ks_unit_id=0, session_name="07072025", stage="Expert",
        raw_wf_root=tmp_path, channel_positions=cp,
        in_zone_idx=list(range(40)), drift_offset=0.0,
    )
    assert feat.session_name == "07072025"
    assert feat.peak_depth_um == pytest.approx(5 * 20.0)        # channel 5 * 20um
    assert feat.peak_depth_corrected_um == pytest.approx(5 * 20.0)
    assert feat.waveform_peak.shape[0] == 82
    assert feat.isi_hist_curation.shape == (50,)
    assert feat.isi_hist_holdout.shape == (50,)
    assert "baseline_on" in feat.inzone_psths
    assert feat.n_inzone_trials == 40


def test_extract_curation_feature_missing_waveform_returns_none(tmp_path):
    sess = make_synthetic_session(n_trials=20, n_clusters=1, seed=3)
    feat = tc.extract_curation_feature(
        sess, ks_unit_id=0, session_name="07072025", stage="Expert",
        raw_wf_root=tmp_path, channel_positions=None,
        in_zone_idx=list(range(20)), drift_offset=0.0,
    )
    assert feat is None
