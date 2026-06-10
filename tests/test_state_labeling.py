from visdetect.analysis import constants as C
from visdetect.analysis import config as CFG


def test_state_constants_exist():
    assert C.STATE_LABELS == ["Impulsive", "StimSens", "Disengaged"]
    assert C.STATE_EASY_CHANGE_THRESH == 2.0
    assert C.STATE_CONFIDENCE_THRESHOLD == 0.8
    assert C.STATE_LABEL_W_DEFAULT in C.STATE_LABEL_W_GRID
    assert C.STATE_FEATURE_COLS == [
        "f_applick", "f_inapplick", "f_nolick", "f_abort", "f_miss_easy", "f_hit_hard",
    ]


def test_lick_valence_colors():
    for k in ["appropriate_lick", "inappropriate_lick", "nolick", "abort", "ref"]:
        assert k in CFG.LICK_VALENCE_COLORS


import pytest
from visdetect.analysis.state_labeling import classify_lick_valence


@pytest.mark.parametrize("outcome,is_go,is_catch,expected", [
    ("hit",  True,  False, "appropriate_lick"),    # go hit
    ("Hit",  True,  False, "appropriate_lick"),    # case-insensitive
    ("hit",  False, True,  "inappropriate_lick"),  # catch SDT false alarm
    ("miss", True,  False, "nolick"),              # go miss
    ("miss", False, True,  "nolick"),              # correct rejection
    ("fa",   True,  False, "inappropriate_lick"),  # early lick on go
    ("fa",   False, True,  "inappropriate_lick"),  # early lick on catch
    ("abort", True, False, "abort"),
    ("ref",  True,  False, "ref"),
])
def test_classify_lick_valence(outcome, is_go, is_catch, expected):
    assert classify_lick_valence(outcome, is_go, is_catch) == expected
