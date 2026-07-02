"""Robust, subject-aware session-date parsing for the multi-subject curation pipeline.

The legacy config.parse_session_date does str(int(session)).zfill(8) and so only
accepts BARE NUMERIC date tokens (BG_046 style). The new subjects' registries use
fully-prefixed tokens (BG_049_01092025), 6-digit DDMMYY (BG_031_050325), and
suffixed re-recordings (BG_031_19052025_b, BG_039_01042025_v2). session_date_key
must yield a correct chronological (year, month, day) key for all of these.
"""
import pytest
from visdetect.analysis.config import session_date_key


def test_bare_8digit_ddmmyyyy():
    assert session_date_key("01072025") == (2025, 7, 1)


def test_bare_7digit_leading_zero_stripped():
    # BG_046 registry stores DDMMYYYY with a stripped leading zero (1072025)
    assert session_date_key("1072025") == (2025, 7, 1)


def test_subject_prefixed_8digit():
    assert session_date_key("BG_049_01092025") == (2025, 9, 1)


def test_subject_prefixed_6digit_ddmmyy():
    assert session_date_key("BG_031_050325") == (2025, 3, 5)


def test_bare_6digit_ddmmyy():
    assert session_date_key("050325") == (2025, 3, 5)


def test_prefixed_with_letter_suffix():
    # a re-recording suffix must be ignored for the date key
    assert session_date_key("BG_031_19052025_b") == (2025, 5, 19)


def test_prefixed_with_v2_suffix():
    assert session_date_key("BG_039_01042025_v2") == (2025, 4, 1)


def test_bare_5digit_dmmyy_leading_zero_stripped():
    # BG_031 manifest stores DDMMYY as an int64 -> the leading-zero DAY is dropped
    # (5 Mar 2025 -> "50325"). Must parse to 2025-03-05, NOT the (325, 5, 0) garbage
    # the old zfill(8)-as-DDMMYYYY path produced (silently dropped these sessions).
    assert session_date_key("50325") == (2025, 3, 5)
    assert session_date_key("70325") == (2025, 3, 7)
    assert session_date_key(50325) == (2025, 3, 5)          # int form too


def test_5digit_matches_6digit_and_prefixed():
    # the manifest 5-digit form and the pkl/registry 6-digit + prefixed forms of the
    # SAME session must yield the SAME key (so the join succeeds).
    assert (session_date_key("50325")
            == session_date_key("050325")
            == session_date_key("BG_031_050325") == (2025, 3, 5))


def test_chronological_ordering_mixed_formats():
    sessions = ["BG_031_19052025_b", "BG_031_050325", "BG_031_01042025"]
    assert sorted(sessions, key=session_date_key) == [
        "BG_031_050325",      # 2025-03-05
        "BG_031_01042025",    # 2025-04-01
        "BG_031_19052025_b",  # 2025-05-19
    ]
