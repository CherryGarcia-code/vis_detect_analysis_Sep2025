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
