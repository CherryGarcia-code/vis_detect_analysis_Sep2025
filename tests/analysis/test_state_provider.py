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


from visdetect.core.session import Session, Trial, Cluster


def test_canonical_from_hmm_label_maps_three():
    assert sp.canonical_from_hmm_label("Stimulus_sensitive") == "in_zone"
    assert sp.canonical_from_hmm_label("Impulsive") == "impulsive"
    assert sp.canonical_from_hmm_label("Disengaged") == "disengaged"


def test_canonical_from_hmm_label_strips_rank_suffix():
    assert sp.canonical_from_hmm_label("Impulsive_2") == "impulsive"


def test_canonical_from_hmm_label_unknown_returns_none():
    assert sp.canonical_from_hmm_label("Intermediate_1") is None


def test_rows_from_decoded_df_uses_trial_idx_column():
    df = pd.DataFrame({
        "trial_idx": [0, 3, 7],                         # raw session.trials index
        "hmm_state_label": ["Stimulus_sensitive", "Impulsive", "Intermediate_0"],
        "p_state_max": [0.9, 0.8, 0.55],
    })
    rows = sp.rows_from_decoded_df(df)
    # Intermediate_0 -> None canonical -> dropped
    assert rows == [(0, "in_zone", pytest.approx(0.9)),
                    (3, "impulsive", pytest.approx(0.8))]


def _toy_session():
    trials = [Trial(trialoutcome="Hit", reactiontimes={"Hit": 0.4},
                    change_size=2.0, orientation=None, ITI=1.0,
                    change_time=2.3, baseline_values=None) for _ in range(4)]
    clusters = [Cluster(cluster_id=0, spike_times=__import__("numpy").array([0.1, 0.2]),
                        quality=None)]
    return Session(trials=trials, clusters=clusters, subject="S",
                   session_name="07072025",
                   good_cluster_ids=[0],
                   ni_events={"Baseline_ON": __import__("numpy").arange(4) * 3.0,
                              "Change_ON": __import__("numpy").arange(4) * 3.0 + 2.3})


def test_uniform_inzone_provider_labels_all_valid(tmp_path):
    sess = _toy_session()
    prov = sp.UniformInZoneStateProvider()
    prov.write(sess, "07072025", tmp_path)
    idx = sp.in_zone_trial_indices("07072025", tmp_path)
    assert idx == [0, 1, 2, 3]
