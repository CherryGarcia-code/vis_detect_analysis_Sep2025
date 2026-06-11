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
