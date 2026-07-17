"""Guard: no CSV deliverable may contain a leading-zero-stripped session id.

WHY THIS TEST EXISTS
--------------------
Session ids are ``DDMMYYYY``. Written to CSV from an int -- or read back into an
all-numeric column, which pandas types as int64 -- the leading-zero DAY of days 1-9
is silently dropped::

    01072025  --int-->  1072025   (7 digits)

On 2026-07-13 an audit found **38,730 corrupted rows across 19 CSVs**, including the
flagship B8 deliverable ``decision_latents_by_state.csv`` (5,187 rows), all four
``data/anatomy/BG_0*/unit_anatomy.csv`` tables, and ``waveform_celltype_labels.csv``.
Any join keyed on those columns silently DROPPED every day-1-9 session.

This is a DATA-correctness bug, so it is guarded by asserting on DATA -- not by
pattern-matching source text. A lint for ``int(...)`` / ``.zfill(8)`` was evaluated
and rejected: it is evadable by renaming a variable, and it flags
``canonical_session_id`` itself (whose body legitimately *is* ``str(int(x)).zfill(8)``).
This test cannot be evaded by any spelling of the bug, because it checks the output.

A 7-digit id is an UNAMBIGUOUS signature: the repo's other token width is 6-digit
``DDMMYY`` (BG_031/BG_039), which can never be 7 digits. So 7 digits always means a
stripped ``DDMMYYYY``.

Repair with::

    py scripts/qc/repair_session_ids.py --execute
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

from visdetect.analysis.config import canonical_session_id

_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "data"

SESSION_COLS = {"session_name", "session_date", "session", "session_id"}
SKIP_PARTS = {".git", ".venv", "archive", "_DeepUnitMatch_repo", ".claude", "node_modules"}
CORRUPT = re.compile(r"^\d{7}$")


def _csvs_with_session_col():
    if not _DATA.exists():
        return []
    out = []
    for p in sorted(_DATA.rglob("*.csv")):
        if any(part in SKIP_PARTS for part in p.parts):
            continue
        out.append(p)
    return out


@pytest.mark.skipif(not _DATA.exists(), reason="no local data/ dir (CI checkout)")
def test_no_stripped_session_ids_in_csv_deliverables():
    """No CSV under data/ may contain a bare 7-digit session id."""
    offenders = []
    for p in _csvs_with_session_col():
        try:
            with p.open("r", newline="", encoding="utf-8-sig") as fh:
                rdr = csv.reader(fh)
                try:
                    header = next(rdr)
                except StopIteration:
                    continue
                idxs = [i for i, h in enumerate(header)
                        if h.strip().lower() in SESSION_COLS]
                if not idxs:
                    continue
                bad = 0
                for rec in rdr:
                    for i in idxs:
                        if i < len(rec) and CORRUPT.match(rec[i].strip()):
                            bad += 1
                if bad:
                    offenders.append((p.relative_to(_ROOT).as_posix(), bad))
        except (UnicodeDecodeError, OSError):
            continue

    if offenders:
        lines = "\n".join(f"    {n:>6} rows | {f}" for f, n in
                          sorted(offenders, key=lambda t: -t[1]))
        pytest.fail(
            f"{len(offenders)} CSV deliverable(s) contain leading-zero-stripped "
            f"(7-digit) session ids:\n{lines}\n\n"
            f"  A 7-digit id is always a corrupted 8-digit DDMMYYYY id.\n"
            f"  Any join on this column silently drops every day-1-9 session.\n"
            f"  Repair:  py scripts/qc/repair_session_ids.py --execute\n"
            f"  Prevent: canonicalize the column via "
            f"visdetect.analysis.config.canonicalize_session_column(df) BEFORE to_csv()."
        )


# ── The invariant the guard relies on ────────────────────────────────────────

def test_seven_digits_is_an_unambiguous_corruption_signature():
    """zfill(8) is always the right repair for a 7-digit id."""
    assert canonical_session_id(1072025) == "01072025"
    assert canonical_session_id("1072025") == "01072025"
    assert canonical_session_id("1072025.0") == "01072025"   # CSV float round-trip
    assert canonical_session_id(1072025.0) == "01072025"
    assert canonical_session_id("01072025") == "01072025"    # idempotent


def test_canonicalize_session_column_fixes_a_corrupt_frame():
    """The write-boundary helper repairs an int64 column."""
    pd = pytest.importorskip("pandas")
    from visdetect.analysis.config import canonicalize_session_column

    # int64 column -- exactly what read_csv/to_csv produces for an all-numeric col
    df = pd.DataFrame({"session_name": [1072025, 23062025, 2072025], "x": [1, 2, 3]})
    assert df["session_name"].dtype.kind == "i"

    out = canonicalize_session_column(df.copy())
    assert out["session_name"].tolist() == ["01072025", "23062025", "02072025"]
    assert out["x"].tolist() == [1, 2, 3]          # other columns untouched
    # idempotent
    assert canonicalize_session_column(out.copy())["session_name"].tolist() == \
        ["01072025", "23062025", "02072025"]


def test_canonicalize_session_column_leaves_non_ddmmyyyy_tokens_alone():
    """6-digit DDMMYY and subject-prefixed/suffixed tokens must pass through."""
    pd = pytest.importorskip("pandas")
    from visdetect.analysis.config import canonicalize_session_column

    df = pd.DataFrame({"session_name": ["BG_012_01112023_pr", "01042025_v2", "19052025_b"]})
    out = canonicalize_session_column(df.copy())
    assert out["session_name"].tolist() == ["BG_012_01112023_pr", "01042025_v2", "19052025_b"]


def test_restore_session_token_is_width_preserving():
    """restore_session_token must NOT promote a 6-digit DDMMYY to 8 digits.

    This is the entire reason the helper exists alongside canonical_session_id.
    BG_031/BG_039 carry 6-digit DDMMYY tokens; an int cast strips their leading-zero
    day to 5 digits. The restore must return them to 6 -- not to the 8-digit DDMMYYYY
    form, which would be a different (wrong) date.
    """
    from visdetect.analysis.config import restore_session_token

    # 8-digit DDMMYYYY family: stripped 7 -> restored 8
    assert restore_session_token(1072025) == "01072025"
    assert restore_session_token("1072025") == "01072025"
    assert restore_session_token("1072025.0") == "01072025"   # CSV float round-trip
    assert restore_session_token(23062025) == "23062025"      # already correct
    assert restore_session_token("01072025") == "01072025"    # idempotent

    # 6-digit DDMMYY family (BG_031 / BG_039): stripped 5 -> restored 6, NOT 8
    assert restore_session_token(50325) == "050325"
    assert restore_session_token("50325") == "050325"
    assert restore_session_token("050325") == "050325"        # idempotent
    assert restore_session_token(230625) == "230625"          # already correct

    # Non-numeric tokens pass through untouched
    assert restore_session_token("BG_012_01112023_pr") == "BG_012_01112023_pr"
    assert restore_session_token("01042025_v2") == "01042025_v2"


def test_restore_session_token_vs_canonical_session_id_on_ddmmyy():
    """Documents WHY both helpers exist: canonical_session_id corrupts DDMMYY."""
    from visdetect.analysis.config import restore_session_token

    # canonical_session_id forces the 8-digit DDMMYYYY form -- correct for a
    # single DDMMYYYY subject, WRONG for a 6-digit DDMMYY token:
    assert canonical_session_id("050325") == "00050325"       # day 00, month 05 -- nonsense
    # restore_session_token preserves the token family:
    assert restore_session_token("050325") == "050325"

    # ...and they agree on the 8-digit family, which is the common case.
    assert canonical_session_id(1072025) == restore_session_token(1072025) == "01072025"
