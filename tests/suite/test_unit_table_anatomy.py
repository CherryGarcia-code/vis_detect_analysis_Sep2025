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


def test_build_unit_table_deduplicates_anatomy(monkeypatch):
    """Duplicate (session_name, cluster_id) rows in anatomy CSV must not multiply GLT rows."""
    import pandas as pd
    from visdetect.suite import loader as L
    minimal = pd.DataFrame({
        "Session_Date": [7072025], "Cluster_ID": [3],
        "Global_UID": [1], "stage": ["Expert"], "session_idx": [0],
    })
    # Anatomy CSV has TWO rows for the same unit (simulates a bad CSV).
    anat_dup = pd.DataFrame({
        "session_name": [7072025, 7072025],
        "cluster_id": [3, 3],
        "peak_channel": [10, 11],   # different values; keep="last" should keep 11
        "shank": [0, 0],
        "depth_um": [100.0, 100.0],
        "ccf_ap": [1.0, 1.0], "ccf_ml": [2.0, 2.0], "ccf_dv": [3.0, 3.0],
        "region_acronym": ["CP", "CP"],
        "region_name": ["Caudoputamen", "Caudoputamen"],
        "region_coarse": ["CP", "CP"],
        "region_confidence": [0.9, 0.9],
        "loc_method": ["ccf", "ccf"],
    })
    monkeypatch.setattr(L, "load_glt", lambda qc_only=True: minimal.copy())
    monkeypatch.setattr(L, "load_all_lick_responsiveness", lambda: pd.DataFrame())
    monkeypatch.setattr(L, "load_waveform_labels",
                        lambda path=None: (_ for _ in ()).throw(FileNotFoundError("none")))
    monkeypatch.setattr(L, "load_tf_responsiveness_detrended", lambda: pd.DataFrame())
    monkeypatch.setattr(L, "load_tf_classification_detrended", lambda: pd.DataFrame())
    monkeypatch.setattr(L, "load_unit_anatomy", lambda path=None: anat_dup)
    # Must not raise UnitTableContractError (which fires on duplicate key rows).
    df = L.build_unit_table(qc_only=True, validate=True)
    assert len(df) == 1, f"Expected 1 row, got {len(df)}"
