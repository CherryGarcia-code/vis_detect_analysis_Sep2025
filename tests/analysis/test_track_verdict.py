"""Tests for row-level track_verdict resolution (M1)."""
import pandas as pd

from visdetect.analysis.track_verdict import (
    load_kept_map, load_trimmed_verdicts, resolve_row_verdict,
)


def _write_trimmed(tmp_path):
    csv = tmp_path / "verdicts_trimmed.csv"
    pd.DataFrame({
        "global_uid": [177, 262],
        "kept_sessions": ["2092025;16092025;17092025", "25072025"],
        "trimmed_verdict": ["trusted", "suspect"],
    }).to_csv(csv, index=False)
    return csv


def test_load_kept_map_int_sessions(tmp_path):
    kept = load_kept_map(_write_trimmed(tmp_path))
    assert kept[177] == {2092025, 16092025, 17092025}
    assert kept[262] == {25072025}


def test_kept_session_gets_trimmed_verdict(tmp_path):
    csv = _write_trimmed(tmp_path)
    kept, verds = load_kept_map(csv), load_trimmed_verdicts(csv)
    # session present in 8-digit form must still match (int compare)
    assert resolve_row_verdict(177, "02092025", kept, verds) == "trusted"
    assert resolve_row_verdict(177, 16092025, kept, verds) == "trusted"


def test_dropped_session_is_suspect(tmp_path):
    csv = _write_trimmed(tmp_path)
    kept, verds = load_kept_map(csv), load_trimmed_verdicts(csv)
    # 177 is trimmed-trusted but this session was dropped from its stable subset
    assert resolve_row_verdict(177, 1072025, kept, verds) == "suspect"


def test_non_cohort_uid_is_unknown(tmp_path):
    csv = _write_trimmed(tmp_path)
    kept, verds = load_kept_map(csv), load_trimmed_verdicts(csv)
    assert resolve_row_verdict(9999, 1072025, kept, verds) == "unknown"


def test_output_in_contract_vocabulary(tmp_path):
    from visdetect.suite.unit_table_schema import ALLOWED_VALUES
    csv = _write_trimmed(tmp_path)
    kept, verds = load_kept_map(csv), load_trimmed_verdicts(csv)
    allowed = ALLOWED_VALUES["track_verdict"]
    for uid, sess in [(177, 2092025), (177, 1072025), (262, 25072025), (1, 1072025)]:
        assert resolve_row_verdict(uid, sess, kept, verds) in allowed


def test_empty_kept_sessions_is_suspect(tmp_path):
    """A UID in the trimmed cohort but with no kept sessions → 'suspect'."""
    csv = tmp_path / "verdicts_trimmed_empty.csv"
    pd.DataFrame({
        "global_uid": [500],
        "kept_sessions": [""],          # empty -> no kept sessions
        "trimmed_verdict": ["review"],
    }).to_csv(csv, index=False)
    kept, verds = load_kept_map(csv), load_trimmed_verdicts(csv)
    assert kept[500] == set()            # NaN/empty parsed to empty set
    assert resolve_row_verdict(500, 1072025, kept, verds) == "suspect"


def test_nan_global_uid_is_unknown(tmp_path):
    import numpy as np
    csv = _write_trimmed(tmp_path)
    kept, verds = load_kept_map(csv), load_trimmed_verdicts(csv)
    assert resolve_row_verdict(np.nan, 2092025, kept, verds) == "unknown"
    assert resolve_row_verdict(float("nan"), 1072025, kept, verds) == "unknown"
