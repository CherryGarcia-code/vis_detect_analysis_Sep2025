"""Tests for the frozen unit-label-table schema contract."""
import pandas as pd
import pytest

from visdetect.suite.unit_table_schema import (
    CONTRACT_COLUMNS,
    KEY_COLUMNS,
    LABEL_DEFAULTS,
    UnitTableContractError,
    add_label_defaults,
    validate_unit_table,
)


def _valid_df():
    return pd.DataFrame({
        "Session_Date": [7072025, 7072025],
        "Cluster_ID": [1, 2],
        "Global_UID": pd.array([10, 11], dtype="Int64"),
        "stage": ["Learning", "Learning"],
        "session_idx": [0, 0],
        "track_verdict": ["trusted", "unknown"],
        "celltype": ["FSI", "SPN"],
        "opto_tag": ["D1", "none"],
        "tf_class": ["Splitter", "unclassified"],
        "is_lick_responsive": [True, False],
        # anatomy columns (Task 7)
        "peak_channel": [-1, -1],
        "shank": [-1, -1],
        "depth_um": [float("nan"), float("nan")],
        "ccf_ap": [float("nan"), float("nan")],
        "ccf_ml": [float("nan"), float("nan")],
        "ccf_dv": [float("nan"), float("nan")],
        "region_acronym": ["unknown", "unknown"],
        "region_name": ["unknown", "unknown"],
        "region_coarse": ["unknown", "unknown"],
        "region_confidence": [float("nan"), float("nan")],
        "loc_method": ["none", "none"],
    })


def test_contract_columns_cover_all_label_defaults():
    for col in LABEL_DEFAULTS:
        assert col in CONTRACT_COLUMNS


def test_validate_passes_on_valid_df():
    validate_unit_table(_valid_df())  # must not raise


def test_validate_raises_on_missing_column():
    df = _valid_df().drop(columns=["opto_tag"])
    with pytest.raises(UnitTableContractError, match="opto_tag"):
        validate_unit_table(df)


def test_validate_raises_on_noninteger_key():
    df = _valid_df()
    df["Cluster_ID"] = df["Cluster_ID"].astype(float)
    with pytest.raises(UnitTableContractError, match="Cluster_ID"):
        validate_unit_table(df)


def test_validate_raises_on_duplicate_keys():
    df = _valid_df()
    df.loc[1, ["Session_Date", "Cluster_ID"]] = [7072025, 1]
    with pytest.raises(UnitTableContractError, match="duplicate"):
        validate_unit_table(df)


def test_validate_raises_on_bad_categorical():
    df = _valid_df()
    df.loc[0, "opto_tag"] = "D9"
    with pytest.raises(UnitTableContractError, match="opto_tag"):
        validate_unit_table(df)


def test_add_label_defaults_fills_missing_columns():
    import math
    df = pd.DataFrame({"Session_Date": [1], "Cluster_ID": [1]})
    out = add_label_defaults(df)
    for col, default in LABEL_DEFAULTS.items():
        assert col in out.columns
        # NaN defaults cannot use == (NaN != NaN); use isna() instead.
        if isinstance(default, float) and math.isnan(default):
            assert out[col].isna().all(), f"Expected NaN default for {col!r}"
        else:
            assert (out[col] == default).all(), f"Expected default {default!r} for {col!r}"


def test_add_label_defaults_preserves_existing():
    df = pd.DataFrame({
        "Session_Date": [1], "Cluster_ID": [1], "celltype": ["FSI"],
    })
    out = add_label_defaults(df)
    assert out.loc[0, "celltype"] == "FSI"          # not overwritten
    assert out.loc[0, "opto_tag"] == LABEL_DEFAULTS["opto_tag"]  # filled
