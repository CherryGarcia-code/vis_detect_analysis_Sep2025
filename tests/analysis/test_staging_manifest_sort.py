"""load_staging_manifest sorts chronologically for every session-token format,
including 6-digit DDMMYY tokens (BG_031/039) whose leading-zero day an int() cast
drops. Guards change #2 (sort via session_date_key, not parse_session_date(int(x)))."""
from visdetect.analysis.config import load_staging_manifest


def _write(tmp_path, rows):
    p = tmp_path / "m.csv"
    p.write_text("session_name,date,stage\n" + "\n".join(rows) + "\n")
    return str(p)


def test_sort_6digit_ddmmyy(tmp_path):
    # CROSS-YEAR DDMMYY tokens: 23 Nov 2024, 1 Dec 2024, 5 Mar 2025 (out of order)
    # -> chronological [231124, 011224, 050325]. The old parse_session_date(int(x))
    # path zfills to '00DDMMYY' and reads MMYY as the year, so it sorts by month-of-
    # a-fake-year and places Mar-2025 first ([050325, 231124, 011224]) -- WRONG across
    # a year boundary. session_date_key parses DDMMYY -> 20YY and orders correctly.
    path = _write(tmp_path, ["011224,011224,Learning",
                             "050325,050325,Learning",
                             "231124,231124,Learning"])
    df = load_staging_manifest(manifest_path=path, qc_only=False, apply_filter=False)
    assert list(df["session_name"]) == ["231124", "011224", "050325"]


def test_sort_8digit_day1to9(tmp_path):
    # BG_046 form: 23 Jun, 9 Jul, 1 Jul -> chronological 23 Jun, 1 Jul, 9 Jul.
    path = _write(tmp_path, ["23062025,23062025,Expert",
                             "09072025,09072025,Expert",
                             "01072025,01072025,Expert"])
    df = load_staging_manifest(manifest_path=path, qc_only=False, apply_filter=False)
    assert list(df["session_name"]) == ["23062025", "01072025", "09072025"]
