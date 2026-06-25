"""Tests for the canonical session-id helper (config.canonical_session_id).

This pins the project-wide fix for a recurring bug: session ids are DDMMYYYY
8-digit strings, but the leading-zero DAY of days 1-9 (e.g. 01072025 = 1 Jul 2025)
drops to 7 digits (1072025) when stored int64 / cast via int(), silently breaking
key lookups and chronological ordering. `canonical_session_id` collapses every
representation to the same zfill8 string; `parse_session_date` must be robust to the
same forms.
"""
import numpy as np
import pandas as pd

from visdetect.analysis.config import (
    canonical_session_id,
    parse_session_date,
    chronological_sort,
)


def test_canonical_session_id_collapses_all_representations():
    """int64, int-form string, already-padded, int-valued float, and the
    float-as-string CSV round-trip ALL map to the same zfill8 key."""
    expected = "01072025"  # 1 Jul 2025
    assert canonical_session_id(1072025) == expected          # int64
    assert canonical_session_id("1072025") == expected        # int-form string
    assert canonical_session_id("01072025") == expected       # already canonical
    assert canonical_session_id(1072025.0) == expected        # int-valued float
    assert canonical_session_id("1072025.0") == expected      # float-as-string
    assert canonical_session_id(np.int64(1072025)) == expected  # numpy int


def test_canonical_session_id_day_10_plus_unchanged():
    """Days 10-31 are already 8 digits and must be left exactly as-is."""
    assert canonical_session_id(23062025) == "23062025"
    assert canonical_session_id("23062025") == "23062025"
    assert canonical_session_id(15092025) == "15092025"


def test_canonical_session_id_non_numeric_passthrough():
    """Non-numeric ids (test mocks, subject-prefixed names) pass through unchanged
    -- the helper must NOT mangle them."""
    assert canonical_session_id("S_expert") == "S_expert"
    assert canonical_session_id("BG_046_01072025") == "BG_046_01072025"
    assert canonical_session_id("  01072025  ".strip()) == "01072025"


def test_canonical_session_id_matches_csv_int64_roundtrip():
    """The exact failure mode: a numeric session written to CSV reads back int64
    (leading zero dropped); canonicalizing both the column and a zfill8 key makes
    them compare equal."""
    df = pd.DataFrame({"session_name": [1072025, 23062025]})
    # the deliverable stores int64 -> astype(str) drops the leading zero
    assert df["session_name"].dtype == np.int64
    assert df["session_name"].astype(str).iloc[0] == "1072025"
    # canonicalizing both sides recovers the match
    col = df["session_name"].map(canonical_session_id)
    assert list(col) == ["01072025", "23062025"]
    assert (col == canonical_session_id("01072025")).iloc[0]


def test_parse_session_date_robust_to_all_forms():
    """parse_session_date returns the correct (y, m, d) for int, int-form string,
    canonical string, AND the float-as-string form (previously a crash/NaT)."""
    target = (2025, 7, 1)
    assert parse_session_date(1072025) == target
    assert parse_session_date("1072025") == target
    assert parse_session_date("01072025") == target
    assert parse_session_date(1072025.0) == target
    assert parse_session_date("1072025.0") == target


def test_chronological_sort_not_fooled_by_leading_zero():
    """The misordering case: 1 Jul (01072025) is AFTER 23 Jun (23062025), but a
    naive lexical/numeric sort of the int forms puts '1072025' first. chronological
    _sort (via parse_session_date) orders them correctly regardless of form."""
    sessions = [1072025, 23062025, 30062025, 2092025]  # mixed leading-zero days
    out = chronological_sort(sessions)
    # correct chronology: 23 Jun, 30 Jun, 1 Jul, 2 Sep
    assert [parse_session_date(s) for s in out] == [
        (2025, 6, 23), (2025, 6, 30), (2025, 7, 1), (2025, 9, 2)]
    # a naive lexical sort of the int-form strings would be WRONG (sanity contrast)
    naive = sorted(str(s) for s in sessions)
    assert naive[0] == "1072025"  # 1 Jul wrongly first under lexical int-form sort
