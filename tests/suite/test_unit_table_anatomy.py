"""Tests for anatomy column registration in the unit-table contract + loader merge."""
import pandas as pd
import pytest
from visdetect.suite.unit_table_schema import (
    LABEL_DEFAULTS, CONTRACT_COLUMNS, ALLOWED_VALUES,
    add_label_defaults, validate_unit_table, UnitTableContractError,
)


def _minimal_table():
    return pd.DataFrame({
        "Session_Date": [7072025], "Cluster_ID": [3],
        "Global_UID": [1], "stage": ["Expert"], "session_idx": [0],
    })


def test_anatomy_defaults_present_after_add():
    df = add_label_defaults(_minimal_table())
    for c in ("region_coarse", "ccf_ap", "ccf_ml", "ccf_dv", "region_confidence", "loc_method"):
        assert c in df.columns


def test_contract_includes_anatomy():
    assert "region_coarse" in CONTRACT_COLUMNS


def test_region_coarse_value_check():
    df = add_label_defaults(_minimal_table())
    df["region_coarse"] = "Mars"
    with pytest.raises(UnitTableContractError, match="region_coarse"):
        validate_unit_table(df)


def test_build_unit_table_has_anatomy_columns(monkeypatch):
    import pandas as pd
    from visdetect.suite import loader as L
    minimal = pd.DataFrame({
        "Session_Date": [7072025], "Cluster_ID": [3],
        "Global_UID": [1], "stage": ["Expert"], "session_idx": [0],
    })
    monkeypatch.setattr(L, "load_glt", lambda qc_only=True: minimal.copy())
    monkeypatch.setattr(L, "load_all_lick_responsiveness", lambda: pd.DataFrame())
    monkeypatch.setattr(L, "load_waveform_labels",
                        lambda path=None: (_ for _ in ()).throw(FileNotFoundError("none")))
    monkeypatch.setattr(L, "load_tf_responsiveness_detrended", lambda: pd.DataFrame())
    monkeypatch.setattr(L, "load_tf_classification_detrended", lambda: pd.DataFrame())
    df = L.build_unit_table(qc_only=True, validate=True)
    for c in ("region_coarse", "ccf_ap", "ccf_dv", "loc_method"):
        assert c in df.columns
    # no anatomy file present -> defaults
    assert df.loc[0, "region_coarse"] == "unknown"
    assert df.loc[0, "loc_method"] == "none"
