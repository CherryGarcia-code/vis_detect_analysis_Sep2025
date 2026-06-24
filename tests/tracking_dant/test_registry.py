import numpy as np
import pandas as pd
import pytest

import registry


def _lookup():
    return pd.DataFrame({
        "pooled_index": [0, 1, 2, 3],
        "session": ["01072025", "02072025", "01072025", "02072025"],
        "ks_unit_id": [10, 11, 12, 13],
    })


def test_idxcluster_to_registry_basic():
    idx = np.array([1, 1, -1, 2])   # units 0&1 = neuron 1; unit 2 untracked; unit 3 = neuron 2
    out = registry.idxcluster_to_registry(idx, _lookup())
    assert list(out.columns) == ["session", "ks_unit_id", "dant_uid"]
    assert out.loc[out.ks_unit_id == 10, "dant_uid"].item() == 1
    assert out.loc[out.ks_unit_id == 12, "dant_uid"].item() == -1


def test_idxcluster_to_registry_rejects_length_mismatch():
    with pytest.raises(ValueError):
        registry.idxcluster_to_registry(np.array([1, 2]), _lookup())


def test_tracked_lengths_counts_distinct_sessions():
    reg = pd.DataFrame({
        "session": ["a", "b", "a", "b", "a"],
        "ks_unit_id": [1, 2, 3, 4, 5],
        "dant_uid": [1, 1, 2, -1, 1],   # uid 1 spans sessions a,b (unit 5 also session a)
    })
    lengths = registry.tracked_lengths(reg)
    assert lengths[1] == 2     # sessions a, b
    assert lengths[2] == 1
    assert -1 not in lengths.index


def test_survival_function():
    lengths = pd.Series([1, 2, 2, 3])
    ks, frac = registry.survival_function(lengths, n_sessions=3)
    assert ks.tolist() == [1, 2, 3]
    assert np.allclose(frac, [1.0, 0.75, 0.25])


def test_comembership_agreement_identical_is_one():
    reg_a = pd.DataFrame({"session": ["a", "b", "a"], "ks_unit_id": [1, 2, 3], "dant_uid": [1, 1, 2]})
    reg_b = reg_a.rename(columns={"dant_uid": "um_uid"})
    res = registry.comembership_agreement(reg_a, reg_b, "dant_uid", "um_uid")
    assert res["n_shared"] == 3
    assert res["ari"] == pytest.approx(1.0)
    assert res["pairwise_precision"] == pytest.approx(1.0)
    assert res["pairwise_recall"] == pytest.approx(1.0)


def test_melt_cellregistry():
    wide = pd.DataFrame({"UID": [7, 8], "01072025": [10, 0], "02072025": [11, 99]})
    # 0/NaN/empty cells mean "absent in this session"
    long = registry.melt_cellregistry(wide)
    row = long[(long.um_uid == 7) & (long.session == "02072025")]
    assert row.ks_unit_id.item() == 11
    assert ((long.um_uid == 8) & (long.session == "01072025")).sum() == 0  # 0 dropped


def test_melt_cellregistry_splits_merged_cell():
    # A ';'-joined merged cell (two ks ids in one session) becomes TWO long rows.
    wide = pd.DataFrame({"UID": [5], "01072025": ["10;12"], "02072025": [0]})
    long = registry.melt_cellregistry(wide)
    sess_rows = long[(long.um_uid == 5) & (long.session == "01072025")]
    assert len(sess_rows) == 2
    assert sorted(sess_rows.ks_unit_id.tolist()) == [10, 12]
    # the absent (0) session yields no rows
    assert ((long.um_uid == 5) & (long.session == "02072025")).sum() == 0
