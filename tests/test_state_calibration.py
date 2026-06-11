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
