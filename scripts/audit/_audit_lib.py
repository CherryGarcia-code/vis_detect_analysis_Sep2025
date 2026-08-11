# scripts/audit/_audit_lib.py
"""Shared audit harness: measurement recorder + session-token classifier.

Audit rule: every measurement lands in docs/audit/measurements.csv through
record(), never by hand, so the executive summary can cite ids (spec A6).
"""
import csv
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MEASUREMENTS_CSV = REPO / "docs" / "audit" / "measurements.csv"
_FIELDS = ["measurement_id", "domain", "metric", "value", "unit",
           "command", "script", "evidence", "notes"]


def record(measurement_id, domain, metric, value, unit, command, script,
           evidence="", notes=""):
    MEASUREMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if MEASUREMENTS_CSV.exists():
        with MEASUREMENTS_CSV.open(newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f)
                    if r["measurement_id"] != measurement_id]
    rows.append({"measurement_id": measurement_id, "domain": domain,
                 "metric": metric, "value": str(value), "unit": unit,
                 "command": command, "script": script,
                 "evidence": evidence, "notes": notes})
    rows.sort(key=lambda r: (r["domain"], r["measurement_id"]))
    with MEASUREMENTS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def classify_token(s):
    s = str(s)
    if re.fullmatch(r"BG_\d{3}_\d{6,8}(_[A-Za-z0-9]+)*", s):
        return "subject-prefixed"
    if re.fullmatch(r"\d{6,8}_[A-Za-z0-9_]+", s):
        return "suffixed"
    if re.fullmatch(r"\d+\.0", s):
        return "float-string"
    if re.fullmatch(r"00\d{6}", s):
        return "00-padded"
    if re.fullmatch(r"\d{8}", s):
        return "8digit"
    if re.fullmatch(r"\d{7}", s):
        return "7digit-stripped"
    if re.fullmatch(r"\d{6}", s):
        return "6digit"
    return "other"


def canonical(s):
    from visdetect.analysis.config import canonical_session_id
    return canonical_session_id(s)
