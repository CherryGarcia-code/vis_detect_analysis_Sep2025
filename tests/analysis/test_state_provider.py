import numpy as np
import pandas as pd
import pytest
from visdetect.analysis import state_provider as sp


def test_canonical_states_are_three():
    assert sp.CANONICAL_STATES == ("disengaged", "impulsive", "in_zone")


def test_write_then_load_state_table_roundtrip(tmp_path):
    rows = [(0, "in_zone", 0.91), (2, "impulsive", 0.7), (3, "disengaged", 0.99)]
    sp.write_state_table("07072025", rows, tmp_path)
    loaded = sp.load_state_table("07072025", tmp_path)
    assert loaded[0] == ("in_zone", pytest.approx(0.91))
    assert loaded[2] == ("impulsive", pytest.approx(0.7))
    assert set(loaded.keys()) == {0, 2, 3}


def test_write_state_table_rejects_unknown_label(tmp_path):
    with pytest.raises(ValueError):
        sp.write_state_table("07072025", [(0, "engaged", 1.0)], tmp_path)


def test_in_zone_trial_indices_filters_by_label(tmp_path):
    rows = [(0, "in_zone", 0.9), (1, "disengaged", 0.9),
            (2, "in_zone", 0.5), (5, "in_zone", 0.95)]
    sp.write_state_table("07072025", rows, tmp_path)
    idx = sp.in_zone_trial_indices("07072025", tmp_path)
    assert idx == [0, 2, 5]


def test_in_zone_trial_indices_confidence_floor(tmp_path):
    rows = [(0, "in_zone", 0.9), (2, "in_zone", 0.5), (5, "in_zone", 0.95)]
    sp.write_state_table("07072025", rows, tmp_path)
    idx = sp.in_zone_trial_indices("07072025", tmp_path, min_confidence=0.8)
    assert idx == [0, 5]


def test_in_zone_trial_indices_missing_table_returns_empty(tmp_path):
    assert sp.in_zone_trial_indices("99999999", tmp_path) == []
