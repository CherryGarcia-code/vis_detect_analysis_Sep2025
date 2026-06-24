# tests/suite/test_load_unit_anatomy.py
import pandas as pd
from visdetect.suite import loader as L


def _write(d, session_name):
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"session_name": [session_name], "cluster_id": [1],
                  "region_coarse": ["CP"]}).to_csv(d / "unit_anatomy.csv", index=False)


def test_per_subject_layout_concatenates(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "ROOT", str(tmp_path))
    _write(tmp_path / "data" / "anatomy" / "BG_046", 17092025)
    _write(tmp_path / "data" / "anatomy" / "BG_031", 1042025)
    df = L.load_unit_anatomy()
    assert len(df) == 2 and set(df.session_name) == {17092025, 1042025}


def test_legacy_flat_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "ROOT", str(tmp_path))
    base = tmp_path / "data" / "anatomy"; base.mkdir(parents=True)
    pd.DataFrame({"session_name": [7072025], "cluster_id": [3],
                  "region_coarse": ["CP"]}).to_csv(base / "unit_anatomy.csv", index=False)
    df = L.load_unit_anatomy()
    assert len(df) == 1 and int(df.session_name.iloc[0]) == 7072025


def test_empty_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "ROOT", str(tmp_path))
    assert L.load_unit_anatomy().empty


def test_explicit_path_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "ROOT", str(tmp_path / "nowhere"))
    p = tmp_path / "ua.csv"
    pd.DataFrame({"session_name": [1], "cluster_id": [2], "region_coarse": ["CP"]}).to_csv(p, index=False)
    assert len(L.load_unit_anatomy(path=str(p))) == 1
