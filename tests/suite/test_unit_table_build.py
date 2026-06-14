"""Tests for the unit-table build path (synthetic CSVs via path overrides)."""
import pandas as pd
import pytest

from visdetect.suite import loader


def test_load_glt_missing_raises_actionable(tmp_path):
    missing = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError, match="build_longitudinal_table"):
        loader.load_glt(qc_only=False, glt_path=str(missing))


def test_load_waveform_labels_override(tmp_path):
    csv = tmp_path / "wf.csv"
    csv.write_text("session_date,cluster_id,celltype\n7072025,1,FSI\n", encoding="utf-8")
    df = loader.load_waveform_labels(path=str(csv))
    assert "cell_type" in df.columns          # normalized from celltype
    assert df.loc[0, "cell_type"] == "FSI"


def test_load_waveform_labels_missing_override_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Waveform labels not found"):
        loader.load_waveform_labels(path=str(tmp_path / "absent.csv"))


def _write_min_glt(tmp_path):
    """Minimal GLT CSV with the identity columns build_unit_table needs."""
    csv = tmp_path / "glt.csv"
    pd.DataFrame({
        "Session_Date": [7072025, 7072025],
        "Cluster_ID": [1, 2],
        "Global_UID": [10, 11],
    }).to_csv(csv, index=False)
    return csv


def test_build_unit_table_adds_contract_columns(tmp_path, monkeypatch):
    from visdetect.suite import loader as L
    from visdetect.suite.unit_table_schema import CONTRACT_COLUMNS, validate_unit_table

    glt_csv = _write_min_glt(tmp_path)

    # Stub the auxiliary loaders so only the GLT drives the row set.
    monkeypatch.setattr(L, "load_glt",
                        lambda qc_only=True: pd.read_csv(glt_csv).assign(stage="Learning", session_idx=0))
    monkeypatch.setattr(L, "load_all_lick_responsiveness", lambda: pd.DataFrame())
    monkeypatch.setattr(L, "load_waveform_labels",
                        lambda path=None: (_ for _ in ()).throw(FileNotFoundError("none")))
    monkeypatch.setattr(L, "load_tf_responsiveness_detrended", lambda: pd.DataFrame())
    monkeypatch.setattr(L, "load_tf_classification_detrended", lambda: pd.DataFrame())

    df = L.build_unit_table(qc_only=True)
    for col in CONTRACT_COLUMNS:
        assert col in df.columns, col
    assert (df["track_verdict"] == "unknown").all()
    assert (df["opto_tag"] == "none").all()
    validate_unit_table(df)            # must not raise


def test_build_unit_table_raises_on_duplicate_keys(tmp_path, monkeypatch):
    from visdetect.suite import loader as L
    from visdetect.suite.unit_table_schema import UnitTableContractError

    # GLT with a duplicate (Session_Date, Cluster_ID) — simulates a bad merge upstream.
    bad = pd.DataFrame({
        "Session_Date": [7072025, 7072025],
        "Cluster_ID": [1, 1],
        "Global_UID": [10, 11],
        "stage": ["Learning", "Learning"],
        "session_idx": [0, 0],
    })
    monkeypatch.setattr(L, "load_glt", lambda qc_only=True: bad)
    monkeypatch.setattr(L, "load_all_lick_responsiveness", lambda: pd.DataFrame())
    monkeypatch.setattr(L, "load_waveform_labels",
                        lambda path=None: (_ for _ in ()).throw(FileNotFoundError("none")))
    monkeypatch.setattr(L, "load_tf_responsiveness_detrended", lambda: pd.DataFrame())
    monkeypatch.setattr(L, "load_tf_classification_detrended", lambda: pd.DataFrame())

    with pytest.raises(UnitTableContractError, match="duplicate"):
        L.build_unit_table(qc_only=True)


import os
from visdetect.analysis.config import GLT_PATH


@pytest.mark.skipif(not os.path.exists(GLT_PATH),
                    reason="GLT not regenerated yet (run build_longitudinal_table.py)")
def test_build_unit_table_real_data_contract():
    from visdetect.suite.loader import build_unit_table
    from visdetect.suite.unit_table_schema import validate_unit_table, KEY_COLUMNS

    df = build_unit_table(qc_only=True)
    validate_unit_table(df)                       # contract holds on real data
    assert df.duplicated(subset=KEY_COLUMNS).sum() == 0
    assert len(df) > 0
    # Global_UID must be present and at least partially populated (tracking output).
    assert df["Global_UID"].notna().any()


def test_default_verdicts_path_is_repo_figures_not_analysis_suite():
    """Regression: build_unit_table's default verdicts path must be the repo-root,
    SUBJECT-scoped FIGURES/tracking_qc/<SUBJECT> dir (where the QC-sheets pipeline
    writes verdicts), NOT analysis_suite/figures (FIGURE_DIR). The latter silently
    yields all-unknown track_verdict on real data."""
    import os as _os
    from visdetect.suite import loader as L
    from visdetect.suite.config import ROOT, SUBJECT
    assert L.DEFAULT_VERDICTS_PATH == _os.path.join(
        ROOT, "FIGURES", "tracking_qc", SUBJECT, "verdicts_trimmed.csv")
    assert "analysis_suite" not in L.DEFAULT_VERDICTS_PATH


def test_build_unit_table_fills_track_verdict(tmp_path, monkeypatch):
    from visdetect.suite import loader as L

    glt = pd.DataFrame({
        "Session_Date": [2092025, 1072025, 1072025],
        "Cluster_ID": [3, 4, 5],
        "Global_UID": [177, 177, 9999],
        "stage": ["Expert", "Learning", "Learning"],
        "session_idx": [20, 0, 0],
    })
    trimmed = tmp_path / "verdicts_trimmed.csv"
    pd.DataFrame({
        "global_uid": [177],
        "kept_sessions": ["2092025"],
        "trimmed_verdict": ["trusted"],
    }).to_csv(trimmed, index=False)

    monkeypatch.setattr(L, "load_glt", lambda qc_only=True: glt.copy())
    monkeypatch.setattr(L, "load_all_lick_responsiveness", lambda: pd.DataFrame())
    monkeypatch.setattr(L, "load_waveform_labels",
                        lambda path=None: (_ for _ in ()).throw(FileNotFoundError("none")))
    monkeypatch.setattr(L, "load_tf_responsiveness_detrended", lambda: pd.DataFrame())
    monkeypatch.setattr(L, "load_tf_classification_detrended", lambda: pd.DataFrame())

    df = L.build_unit_table(qc_only=True, verdicts_path=str(trimmed))
    by_key = df.set_index(["Session_Date", "Cluster_ID"])["track_verdict"]
    assert by_key[(2092025, 3)] == "trusted"     # 177 kept this session
    assert by_key[(1072025, 4)] == "suspect"     # 177, session dropped from subset
    assert by_key[(1072025, 5)] == "unknown"     # 9999 not in trimmed cohort
