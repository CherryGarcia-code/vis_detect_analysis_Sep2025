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


def _feat(session_name, *, wave, depth, isi, psth_val, n_inzone, n_bins=40,
          corr_depth="same"):
    """Build a minimal CurationFeature for score_link tests.

    corr_depth: drift-corrected depth. Default "same" => equals raw `depth`.
    Pass a float (incl. nan) to set corrected independently of raw.
    """
    wave = np.asarray(wave, dtype=float)
    isi = np.asarray(isi, dtype=float)
    psth = None if psth_val is None else np.full(n_bins, 0.0)
    psths = {}
    if psth_val is not None:
        # a modulated ramp scaled by psth_val so two features correlate or not
        ramp = np.linspace(0, 1, n_bins) * 10.0
        psths["baseline_on"] = ramp * psth_val
    corrected = depth if corr_depth == "same" else corr_depth
    return tc.CurationFeature(
        session_name=session_name, ks_unit_id=0, stage="Expert",
        waveform_peak=wave, footprint=np.zeros((1, 1)), footprint_channels=np.array([0]),
        peak_chan=0, peak_depth_um=depth, peak_depth_corrected_um=corrected,
        baseline_fr_hz=5.0, isi_hist_curation=isi, isi_hist_holdout=isi,
        inzone_psths=psths, n_inzone_trials=n_inzone,
    )


_W = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.5, -0.5, 0.2])
_ISI = np.linspace(0, 1, 50)


def test_score_link_clean_pair_keeps_no_flag():
    p = tc.CurationParams()
    a = _feat("S2", wave=_W, depth=100.0, isi=_ISI, psth_val=1.0, n_inzone=50)
    b = _feat("S1", wave=_W, depth=102.0, isi=_ISI, psth_val=1.0, n_inzone=50)
    lr = tc.score_link(a, b, a, p, gap_sessions=1)
    assert lr.decision == "KEEP"
    assert lr.review_flag is False
    assert lr.func_evaluable is True


def test_score_link_hard_contradiction_stops():
    p = tc.CurationParams()
    a = _feat("S2", wave=_W, depth=100.0, isi=_ISI, psth_val=1.0, n_inzone=50)
    b = _feat("S1", wave=-_W, depth=200.0, isi=_ISI, psth_val=1.0, n_inzone=50)  # flipped wf + 100um jump
    lr = tc.score_link(a, b, a, p, gap_sessions=1)
    assert lr.decision == "STOP"
    assert lr.stop_reason == "hard_contradiction"


def test_score_link_warn_keeps_with_review():
    # A 'warn'-level link (depth 30um here; wave 0.90-0.95 is symmetric) is
    # plausibly the same neuron degraded by drift/slow shape change -> KEEP but
    # flag review, NOT truncate. (Long tracks routinely have warn-level links.)
    p = tc.CurationParams()
    a = _feat("S2", wave=_W, depth=100.0, isi=_ISI, psth_val=1.0, n_inzone=50)
    b = _feat("S1", wave=_W, depth=130.0, isi=_ISI, psth_val=1.0, n_inzone=50)  # 30um = depth warn
    lr = tc.score_link(a, b, a, p, gap_sessions=1)
    assert lr.decision == "KEEP"
    assert lr.review_flag is True


def test_score_link_single_hard_fail_skips():
    # Exactly one metric HARD-fails (depth 50um jump, waveform fine) -> SKIP
    # (bridgeable). STOP needs BOTH to fail; a single fail is not a hard
    # contradiction.
    p = tc.CurationParams()
    a = _feat("S2", wave=_W, depth=100.0, isi=_ISI, psth_val=1.0, n_inzone=50)
    b = _feat("S1", wave=_W, depth=150.0, isi=_ISI, psth_val=1.0, n_inzone=50)  # 50um = depth fail
    lr = tc.score_link(a, b, a, p, gap_sessions=1)
    assert lr.decision == "SKIP"
    assert lr.stop_reason == ""


def test_score_link_func_conflict_flags_review_but_keeps():
    p = tc.CurationParams()
    a = _feat("S2", wave=_W, depth=100.0, isi=_ISI, psth_val=1.0, n_inzone=50)
    b = _feat("S1", wave=_W, depth=101.0, isi=_ISI, psth_val=-1.0, n_inzone=50)  # anti-correlated PSTH
    lr = tc.score_link(a, b, a, p, gap_sessions=1)
    assert lr.decision == "KEEP"
    assert lr.review_flag is True


def test_score_link_func_not_evaluable_when_few_inzone():
    p = tc.CurationParams()      # min_inzone_trials default 20
    a = _feat("S2", wave=_W, depth=100.0, isi=_ISI, psth_val=1.0, n_inzone=50)
    b = _feat("S1", wave=_W, depth=101.0, isi=_ISI, psth_val=-1.0, n_inzone=5)  # too few in-zone
    lr = tc.score_link(a, b, a, p, gap_sessions=1)
    assert lr.decision == "KEEP"
    assert lr.func_evaluable is False
    assert lr.review_flag is False


_NAN = float("nan")


def test_score_link_uses_raw_depth_when_corrected_missing():
    # Corrected depth unavailable (broken drift chain) -> fall back to RAW depth.
    p = tc.CurationParams()
    a = _feat("S2", wave=_W, depth=100.0, isi=_ISI, psth_val=1.0, n_inzone=50, corr_depth=_NAN)
    b = _feat("S1", wave=_W, depth=102.0, isi=_ISI, psth_val=1.0, n_inzone=50, corr_depth=_NAN)
    lr = tc.score_link(a, b, a, p, gap_sessions=1)
    assert lr.decision == "KEEP"
    assert lr.depth_evaluable is True
    assert lr.depth_jump_um == pytest.approx(2.0)       # raw 100 vs 102


def test_score_link_raw_depth_contradiction_still_stops():
    # Even via raw fallback, a flipped waveform + big depth jump is a hard stop.
    p = tc.CurationParams()
    a = _feat("S2", wave=-_W, depth=100.0, isi=_ISI, psth_val=1.0, n_inzone=50, corr_depth=_NAN)
    b = _feat("S1", wave=_W, depth=300.0, isi=_ISI, psth_val=1.0, n_inzone=50, corr_depth=_NAN)
    lr = tc.score_link(a, b, a, p, gap_sessions=1)
    assert lr.decision == "STOP"
    assert lr.stop_reason == "hard_contradiction"


def test_score_link_abstains_when_all_depth_missing():
    # Neither corrected nor raw depth available -> depth ABSTAINS (not a veto).
    p = tc.CurationParams()
    a = _feat("S2", wave=_W, depth=_NAN, isi=_ISI, psth_val=1.0, n_inzone=50, corr_depth=_NAN)
    b = _feat("S1", wave=_W, depth=_NAN, isi=_ISI, psth_val=1.0, n_inzone=50, corr_depth=_NAN)
    lr = tc.score_link(a, b, a, p, gap_sessions=1)
    assert lr.decision == "KEEP"            # waveform passes; depth abstains
    assert lr.depth_evaluable is False
    assert lr.review_flag is True           # kept without depth corroboration -> review


def test_score_link_abstain_does_not_stop_on_waveform_fail():
    # Missing depth must never manufacture a hard_contradiction STOP.
    p = tc.CurationParams()
    a = _feat("S2", wave=_W, depth=_NAN, isi=_ISI, psth_val=1.0, n_inzone=50, corr_depth=_NAN)
    b = _feat("S1", wave=-_W, depth=_NAN, isi=_ISI, psth_val=1.0, n_inzone=50, corr_depth=_NAN)
    lr = tc.score_link(a, b, a, p, gap_sessions=1)
    assert lr.decision == "SKIP"
    assert lr.stop_reason == ""


def test_score_link_prefers_corrected_over_raw_when_both_present():
    # When corrected depth exists for both, use it (raw difference would mislead).
    p = tc.CurationParams()
    a = _feat("S2", wave=_W, depth=100.0, isi=_ISI, psth_val=1.0, n_inzone=50, corr_depth=100.0)
    b = _feat("S1", wave=_W, depth=300.0, isi=_ISI, psth_val=1.0, n_inzone=50, corr_depth=102.0)
    lr = tc.score_link(a, b, a, p, gap_sessions=1)
    assert lr.decision == "KEEP"
    assert lr.depth_jump_um == pytest.approx(2.0)       # corrected 100 vs 102, NOT raw 200


def _chain_feats(session_names, *, swap_at=None, dropout_at=None):
    """Build a per-session feature dict for a clean chain, with optional defects.

    swap_at: session name whose unit is a different neuron (flipped wf + depth jump).
    dropout_at: session name whose unit is garbled (flipped wf only -> soft skip).
    """
    feats = {}
    for s in session_names:
        wave, depth = _W.copy(), 100.0
        if s == swap_at:
            wave, depth = -_W.copy(), 220.0      # hard contradiction
        elif s == dropout_at:
            depth = 150.0                         # 50um = hard depth fail -> SKIP (bridgeable)
        feats[s] = _feat(s, wave=wave, depth=depth, isi=_ISI, psth_val=1.0, n_inzone=50)
    return feats


def test_sweep_clean_chain_is_one_trusted_track():
    p = tc.CurationParams()
    order = ["S1", "S2", "S3", "S4"]            # chronological ascending
    feats = _chain_feats(order)
    res = tc.sweep_uid(feats, order, p)
    assert res.anchor_session == "S4"
    assert set(res.kept_sessions) == {"S1", "S2", "S3", "S4"}
    assert res.confidence_tier == "trusted"


def test_sweep_mid_chain_swap_stops_and_truncates():
    p = tc.CurationParams()
    order = ["S1", "S2", "S3", "S4"]
    feats = _chain_feats(order, swap_at="S2")    # walking back S4->S3->S2 hits swap
    res = tc.sweep_uid(feats, order, p)
    assert "S2" in res.dropped_sessions and "S1" in res.dropped_sessions
    assert set(res.kept_sessions) == {"S3", "S4"}


def test_sweep_single_dropout_is_bridged():
    p = tc.CurationParams()                       # max_bridge_gap default 2
    order = ["S1", "S2", "S3", "S4"]
    feats = _chain_feats(order, dropout_at="S3")  # S3 soft-fails, S2/S1 clean -> resurface
    res = tc.sweep_uid(feats, order, p)
    assert "S3" in res.skipped_sessions
    assert set(res.kept_sessions) == {"S1", "S2", "S4"}
    assert res.confidence_tier == "review"        # a bridge present


def test_sweep_skips_exhausted_drops_trailing():
    p = tc.CurationParams(max_bridge_gap=1)
    order = ["S1", "S2", "S3", "S4"]
    # S3 and S2 both hard-fail on depth -> 2 consecutive skips > max_bridge_gap=1 -> STOP
    feats = _chain_feats(order)
    feats["S3"] = _feat("S3", wave=_W, depth=150.0, isi=_ISI, psth_val=1.0, n_inzone=50)
    feats["S2"] = _feat("S2", wave=_W, depth=150.0, isi=_ISI, psth_val=1.0, n_inzone=50)
    res = tc.sweep_uid(feats, order, p)
    assert res.kept_sessions == ["S4"]
    assert "S3" in res.dropped_sessions and "S2" in res.dropped_sessions
    assert res.confidence_tier == "suspect"       # span 1


def test_compute_tier_short_is_suspect():
    p = tc.CurationParams()
    assert tc.compute_tier(["S4"], [], [], p) == "suspect"


def test_curate_registry_builds_links_and_tracks():
    p = tc.CurationParams()
    order = ["S1", "S2", "S3", "S4"]
    uid_to_sessions = {10: order, 11: ["S1", "S2"]}
    feats = {}
    for s in order:
        feats[(10, s)] = _feat(s, wave=_W, depth=100.0, isi=_ISI, psth_val=1.0, n_inzone=50)
    # uid 11: a swap between its two sessions -> short/suspect
    feats[(11, "S2")] = _feat("S2", wave=_W, depth=100.0, isi=_ISI, psth_val=1.0, n_inzone=50)
    feats[(11, "S1")] = _feat("S1", wave=-_W, depth=220.0, isi=_ISI, psth_val=1.0, n_inzone=50)

    links_df, tracks_df = tc.curate_registry(uid_to_sessions, feats, p)

    t10 = tracks_df[tracks_df.liberal_uid == 10].iloc[0]
    assert t10.confidence_tier == "trusted"
    assert t10.trimmed_span == 4
    t11 = tracks_df[tracks_df.liberal_uid == 11].iloc[0]
    assert t11.confidence_tier == "suspect"
    assert set(links_df.columns) >= {
        "liberal_uid", "anchor_session", "candidate_session", "wave_corr",
        "depth_jump_um", "isi_shape_corr", "func_corr", "func_evaluable",
        "link_decision", "review_flag", "stop_reason"}


import pandas as pd


def test_held_out_isi_auc_separates_matched_from_nonmatched():
    rng = np.random.default_rng(0)
    # Two distinct unit "shapes": A peaks early, B peaks late.
    shapeA = np.exp(-((np.arange(50) - 10) ** 2) / 20.0)
    shapeB = np.exp(-((np.arange(50) - 40) ** 2) / 20.0)

    def noisy(shape):
        h = shape + rng.normal(0, 0.02, size=50)
        h = np.clip(h, 0, None)
        return h / h.sum()

    # uid 1 = unit A across S1,S2,S3 ; uid 2 = unit B across S1,S2,S3
    holdout = {}
    for s in ["S1", "S2", "S3"]:
        holdout[(1, s)] = noisy(shapeA)
        holdout[(2, s)] = noisy(shapeB)
    tracks = pd.DataFrame([
        {"curated_uid": 1, "kept_sessions": "S1;S2;S3", "confidence_tier": "trusted"},
        {"curated_uid": 2, "kept_sessions": "S1;S2;S3", "confidence_tier": "trusted"},
    ])
    out = tc.held_out_isi_auc_by_tier(tracks, holdout)
    assert out["trusted"]["auc"] > 0.9
    assert out["trusted"]["n_matched"] == 6     # 2 uids * C(3,2)=3
