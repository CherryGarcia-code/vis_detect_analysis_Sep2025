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
