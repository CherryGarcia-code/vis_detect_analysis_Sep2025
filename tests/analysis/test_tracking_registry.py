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


from visdetect.analysis.tracking_registry import (
    resolve_collisions, long_to_cellregistry,
)


def test_resolve_collisions_keeps_supported_uid(tmp_path):
    csv = _write_long(tmp_path, [
        ["1072025", 3, 0],   # uid 0 keeps this session (supported)
        ["1072025", 3, 9],   # uid 9 does NOT keep this session
        ["2072025", 4, 1],   # uncontested
    ])
    df = load_canonical_long(csv)
    kept = {0: {1072025}, 9: {2072025}}           # int-session kept-sets
    out = resolve_collisions(df, kept)
    # Only uid 0 retains (01072025, 3); uid 9's contested row dropped.
    held = out[(out["session"] == "01072025") & (out["ks_unit_id"] == 3)]
    assert list(held["global_uid"]) == [0]
    assert len(out) == 2                           # uncontested row survives


def test_resolve_collisions_drops_when_ambiguous(tmp_path):
    csv = _write_long(tmp_path, [["1072025", 3, 0], ["1072025", 3, 9]])
    df = load_canonical_long(csv)
    kept = {0: {1072025}, 9: {1072025}}            # BOTH keep it -> ambiguous
    out = resolve_collisions(df, kept)
    assert out[(out["session"] == "01072025") & (out["ks_unit_id"] == 3)].empty


def test_long_to_cellregistry_pivots_and_zero_pads(tmp_path):
    csv = _write_long(tmp_path, [
        ["1072025", 3, 0], ["2072025", 5, 0], ["1072025", 4, 1],
    ])
    df = load_canonical_long(csv)
    reg = long_to_cellregistry(df)
    # index = global_uid; columns = 8-digit sessions; cells = ks_unit_id
    assert list(reg.index) == [0, 1]
    assert "01072025" in reg.columns and "02072025" in reg.columns
    assert str(reg.loc[0, "01072025"]) == "3"
    assert pd.isna(reg.loc[1, "02072025"])


def test_long_to_cellregistry_joins_oversplit_with_semicolon(tmp_path):
    # uid 0 has TWO clusters (3 and 7) in the same session -> "3;7"
    csv = _write_long(tmp_path, [["1072025", 3, 0], ["1072025", 7, 0]])
    df = load_canonical_long(csv)
    reg = long_to_cellregistry(df)
    assert set(str(reg.loc[0, "01072025"]).split(";")) == {"3", "7"}
