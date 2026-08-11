# tests/audit/test_audit_lib.py
import csv, importlib, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from audit._audit_lib import classify_token, record, MEASUREMENTS_CSV


def test_classify_token_trio_and_edges():
    assert classify_token("01072025") == "8digit"
    assert classify_token("1072025") == "7digit-stripped"
    assert classify_token("1072025.0") == "float-string"
    assert classify_token("050325") == "6digit"
    assert classify_token("00050325") == "00-padded"
    assert classify_token("23042025_v2") == "suffixed"
    assert classify_token("BG_046_01072025") == "subject-prefixed"
    assert classify_token("garbage") == "other"


def test_record_appends_and_overwrites(tmp_path, monkeypatch):
    target = tmp_path / "m.csv"
    monkeypatch.setattr("audit._audit_lib.MEASUREMENTS_CSV", target)
    record("t.one", "D1", "demo", 42, "count", "py x.py", "x.py")
    record("t.one", "D1", "demo", 43, "count", "py x.py", "x.py")  # overwrite same id
    rows = list(csv.DictReader(target.open()))
    assert len(rows) == 1 and rows[0]["value"] == "43"
    assert rows[0]["measurement_id"] == "t.one"
