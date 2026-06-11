import numpy as np
import pandas as pd
import pytest

from visdetect.analysis.state_calibration import extract_state_features


def _raster(lick_valences, is_go=None, change_size=None):
    n = len(lick_valences)
    return pd.DataFrame({
        "trial_idx": range(n),
        "lick_valence": lick_valences,
        "is_go": [True] * n if is_go is None else is_go,
        "change_size": [1.5] * n if change_size is None else change_size,
    })


def test_features_center_window_fractions():
    lv = ["appropriate_lick", "appropriate_lick", "inappropriate_lick",
          "inappropriate_lick", "inappropriate_lick", "nolick", "nolick"]
    feats = extract_state_features(_raster(lv), W=3)
    # center index 3 window {2,3,4} are all inappropriate_lick
    assert feats.loc[3, "f_inapplick"] == pytest.approx(1.0)
    # index 0 window {0,1} are all appropriate_lick
    assert feats.loc[0, "f_applick"] == pytest.approx(1.0)
    # four primary fractions sum to 1 everywhere (no ref/abort here)
    s = feats[["f_applick", "f_inapplick", "f_nolick", "f_abort"]].sum(axis=1)
    assert np.allclose(s.values, 1.0)


def test_features_difficulty_aware():
    lv = ["inappropriate_lick", "nolick", "nolick"]
    feats = extract_state_features(
        _raster(lv, is_go=[True, True, True], change_size=[1.5, 4.0, 4.0]), W=3,
    )
    # index 1 window {0,1,2}: miss_easy at idx1,2 (nolick & go & cs>=2) -> 2/3
    assert feats.loc[1, "f_miss_easy"] == pytest.approx(2.0 / 3.0)


def test_features_hit_hard():
    # go-trial hits: idx0 hard (cs<2 -> hit_hard), idx1 easy (cs>=2 -> NOT hit_hard)
    lv = ["appropriate_lick", "appropriate_lick"]
    feats = extract_state_features(
        _raster(lv, is_go=[True, True], change_size=[1.25, 4.0]), W=3,
    )
    # window {0,1}: only idx0 is a hard hit -> 1/2
    assert feats.loc[0, "f_hit_hard"] == pytest.approx(0.5)


def test_features_all_ref_window_is_zero_not_nan():
    # an all-'ref' window has denom 0; the guard must yield 0.0, never NaN/inf
    feats = extract_state_features(_raster(["ref", "ref", "ref"]), W=3)
    cols = ["f_applick", "f_inapplick", "f_nolick", "f_abort", "f_miss_easy", "f_hit_hard"]
    assert (feats[cols].fillna(-1).values == 0.0).all()


from visdetect.analysis.state_labeling import StateEpisode
from visdetect.analysis.state_calibration import attach_episode_labels, fit_state_tree


def test_attach_episode_labels_by_trial_idx():
    feats = extract_state_features(_raster(["nolick"] * 6), W=3)
    eps = [StateEpisode("S1", 1, 3, "Disengaged", "ben", "t")]
    labeled = attach_episode_labels(feats, eps, "S1")
    assert labeled.loc[0, "state"] is None
    assert list(labeled.loc[1:3, "state"]) == ["Disengaged"] * 3
    assert labeled.loc[4, "state"] is None


def _separable_training_frame():
    # clean, linearly separable 3-class table over the feature columns
    from visdetect.analysis.constants import STATE_FEATURE_COLS
    data = []
    for _ in range(8):
        data.append({**{c: 0.0 for c in STATE_FEATURE_COLS}, "f_inapplick": 0.9, "state": "Impulsive"})
        data.append({**{c: 0.0 for c in STATE_FEATURE_COLS}, "f_nolick": 0.9, "state": "Disengaged"})
        data.append({**{c: 0.0 for c in STATE_FEATURE_COLS}, "f_applick": 0.9, "state": "StimSens"})
    return pd.DataFrame(data)


def test_fit_state_tree_separates_classes_and_is_deterministic():
    df = _separable_training_frame()
    t1 = fit_state_tree(df, seed=42)
    t2 = fit_state_tree(df, seed=42)
    from visdetect.analysis.constants import STATE_FEATURE_COLS
    pred = t1.predict(df[STATE_FEATURE_COLS].values)
    assert (pred == df["state"].values).mean() == 1.0           # separable -> perfect train fit
    assert list(t1.feature_importances_) == list(t2.feature_importances_)  # deterministic


def test_fit_state_tree_raises_on_no_labeled_rows():
    feats = extract_state_features(_raster(["nolick"] * 4), W=3)
    feats["state"] = None   # nothing labeled
    with pytest.raises(ValueError, match="no labeled rows"):
        fit_state_tree(feats, seed=42)


from visdetect.analysis.state_calibration import calibrate_states, CalibrationResult


def _planted_raster(session_name):
    # trials 0-9 impulsive (inappropriate_lick), 10-19 stimsens (appropriate_lick, easy),
    # 20-29 disengaged (nolick, go, easy)
    lv = (["inappropriate_lick"] * 10 + ["appropriate_lick"] * 10 + ["nolick"] * 10)
    cs = ([1.5] * 10 + [4.0] * 10 + [4.0] * 10)
    return pd.DataFrame({
        "trial_idx": range(30), "lick_valence": lv,
        "is_go": [True] * 30, "change_size": cs,
    })


def test_calibrate_states_returns_result_and_fits():
    rasters = {"A": _planted_raster("A"), "B": _planted_raster("B")}
    eps = []
    for s in ("A", "B"):
        eps += [
            StateEpisode(s, 2, 7, "Impulsive", "ben", "t"),
            StateEpisode(s, 12, 17, "StimSens", "ben", "t"),
            StateEpisode(s, 22, 27, "Disengaged", "ben", "t"),
        ]
    result = calibrate_states(rasters, eps, w_grid=[3, 5], seed=42)
    assert isinstance(result, CalibrationResult)
    assert result.window in (3, 5)
    assert set(result.state_labels) == {"Impulsive", "StimSens", "Disengaged"}
    assert result.loso_kappa > 0.5
    assert "f_" in result.rules_text


def test_calibration_result_save_load(tmp_path):
    rasters = {"A": _planted_raster("A"), "B": _planted_raster("B")}
    eps = []
    for s in ("A", "B"):
        eps += [
            StateEpisode(s, 2, 7, "Impulsive", "ben", "t"),
            StateEpisode(s, 12, 17, "StimSens", "ben", "t"),
            StateEpisode(s, 22, 27, "Disengaged", "ben", "t"),
        ]
    result = calibrate_states(rasters, eps, w_grid=[3, 5], seed=42)
    p = tmp_path / "model.pkl"
    result.save(p)
    loaded = CalibrationResult.load(p)
    assert loaded.window == result.window
    assert loaded.state_labels == result.state_labels


def test_calibrate_states_single_session_warns_nan_kappa_but_fits():
    # With only one labeled session every LOSO fold is degenerate: the model must
    # still fit (on that session) but report loso_kappa=NaN and warn — never crash.
    rasters = {"A": _planted_raster("A")}
    eps = [
        StateEpisode("A", 2, 7, "Impulsive", "ben", "t"),
        StateEpisode("A", 12, 17, "StimSens", "ben", "t"),
        StateEpisode("A", 22, 27, "Disengaged", "ben", "t"),
    ]
    with pytest.warns(UserWarning, match="degenerate"):
        result = calibrate_states(rasters, eps, w_grid=[3, 5], seed=42)
    assert isinstance(result, CalibrationResult)
    assert np.isnan(result.loso_kappa)
    assert set(result.state_labels) == {"Impulsive", "StimSens", "Disengaged"}


from visdetect.analysis.state_calibration import tag_features, decode_session_states


def test_tag_features_columns_and_confidence_gating():
    df = _separable_training_frame()
    tree = fit_state_tree(df, seed=42)
    from visdetect.analysis.constants import STATE_FEATURE_COLS
    feats = df[STATE_FEATURE_COLS].copy()
    feats["trial_idx"] = range(len(feats))

    tagged = tag_features(tree, feats, confidence_threshold=0.8)
    K = len(tree.classes_)
    for k in range(K):
        assert f"p_state_{k}" in tagged.columns
    assert {"state", "state_label", "state_confidence", "state_gated"}.issubset(tagged.columns)
    # separable data -> pure leaves -> confidence 1.0 -> nothing gated at 0.8
    assert (tagged["state_gated"] == -1).sum() == 0
    # threshold above the max confidence gates everything
    tagged_hi = tag_features(tree, feats, confidence_threshold=1.0)
    assert (tagged_hi["state_gated"] == -1).all()


def test_decode_session_states_runs_on_synthetic_session():
    from visdetect.utils.synthetic import make_synthetic_session
    df = _separable_training_frame()
    tree = fit_state_tree(df, seed=42)
    result = CalibrationResult(tree, 5, list(tree.classes_), list(df.columns[:-1]), 1.0, "")
    sess = make_synthetic_session(n_trials=30, n_clusters=2, seed=1)
    tagged = decode_session_states(result, sess)
    assert len(tagged) == 30
    assert {"state", "state_label", "state_confidence", "state_gated"}.issubset(tagged.columns)


def test_tag_features_emits_hmm_downstream_aliases():
    # hmm_downstream consumes an `hmm_state` column; the tagged frame must carry the
    # hmm_*-prefixed aliases (== the unprefixed columns) so it is drop-in there.
    df = _separable_training_frame()
    tree = fit_state_tree(df, seed=42)
    from visdetect.analysis.constants import STATE_FEATURE_COLS
    feats = df[STATE_FEATURE_COLS].copy()
    feats["trial_idx"] = range(len(feats))
    tagged = tag_features(tree, feats, confidence_threshold=0.8)
    assert {"hmm_state", "hmm_state_label", "hmm_state_gated"}.issubset(tagged.columns)
    assert list(tagged["hmm_state"]) == list(tagged["state"])
    assert list(tagged["hmm_state_label"]) == list(tagged["state_label"])
    assert list(tagged["hmm_state_gated"]) == list(tagged["state_gated"])


import subprocess, sys, os

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "state_labeling")


def test_calibrate_cli_help():
    r = subprocess.run([sys.executable, os.path.join(_SCRIPTS, "calibrate_states.py"), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "usage" in (r.stdout + r.stderr).lower()


def test_tag_cli_help():
    r = subprocess.run([sys.executable, os.path.join(_SCRIPTS, "tag_sessions.py"), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "usage" in (r.stdout + r.stderr).lower()
