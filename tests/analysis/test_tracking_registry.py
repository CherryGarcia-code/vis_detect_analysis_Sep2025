"""Tests for the canonical tracking registry adapter (M1)."""
import pandas as pd
import pytest

from visdetect.analysis.tracking_registry import (
    load_canonical_long, find_cluster_collisions,
)


def _write_long(tmp_path, rows, cols=("session", "ks_unit_id", "global_uid")):
    csv = tmp_path / "reg.csv"
    pd.DataFrame(rows, columns=list(cols)).to_csv(csv, index=False)
    return csv


def test_load_zero_pads_session_to_8(tmp_path):
    csv = _write_long(tmp_path, [["1072025", 3, 0], ["23062025", 4, 1]])
    df = load_canonical_long(csv)
    assert list(df["session"]) == ["01072025", "23062025"]
    assert df["ks_unit_id"].dtype.kind in ("i", "u")
    assert df["global_uid"].dtype.kind in ("i", "u")


def test_load_accepts_ks_id_alias(tmp_path):
    csv = _write_long(tmp_path, [["1072025", 3, 0]], cols=("session", "ks_id", "global_uid"))
    df = load_canonical_long(csv)
    assert "ks_unit_id" in df.columns
    assert int(df.loc[0, "ks_unit_id"]) == 3


def test_load_missing_columns_raises(tmp_path):
    csv = _write_long(tmp_path, [["1072025", 3]], cols=("session", "ks_unit_id"))
    with pytest.raises(ValueError, match="global_uid"):
        load_canonical_long(csv)


def test_find_collisions_flags_cluster_claimed_by_two_uids(tmp_path):
    # (01072025, ks 3) appears under uid 0 AND uid 9 -> collision (bimodal-ISI failure)
    csv = _write_long(tmp_path, [
        ["1072025", 3, 0], ["1072025", 3, 9], ["1072025", 4, 1],
    ])
    df = load_canonical_long(csv)
    coll = find_cluster_collisions(df)
    assert set(coll["global_uid"]) == {0, 9}
    assert (coll["session"] == "01072025").all()
    assert (coll["ks_unit_id"] == 3).all()


def test_find_collisions_empty_when_clean(tmp_path):
    csv = _write_long(tmp_path, [["1072025", 3, 0], ["1072025", 4, 1]])
    df = load_canonical_long(csv)
    assert find_cluster_collisions(df).empty
