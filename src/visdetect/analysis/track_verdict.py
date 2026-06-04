"""Row-level track_verdict resolution from the trimmed-verdict cohort (M1).

Trust rule (2026-06-03): a (Global_UID, Session) row is the UID's trimmed
verdict iff the session is in that UID's stable kept-subset; otherwise suspect;
UIDs absent from the trimmed cohort are unknown. Sessions are compared as int
to bridge 7/8-digit DDMMYYYY forms.
"""
from __future__ import annotations

from typing import Dict, Set

import pandas as pd


def _to_int_session(s) -> int:
    """Convert session label to int.  Handles int, str, np.int64, and
    float-string forms (e.g. "2092025.0") that arise when a CSV column is
    inferred as float64 by pandas."""
    return int(float(str(s).strip()))


def load_kept_map(trimmed_path) -> Dict[int, Set[int]]:
    """{global_uid -> set of kept sessions as int} from verdicts_trimmed.csv."""
    df = pd.read_csv(trimmed_path)
    out: Dict[int, Set[int]] = {}
    for _, row in df.iterrows():
        raw = row.get("kept_sessions")
        sessions: Set[int] = set()
        if isinstance(raw, str) and raw.strip():
            sessions = {_to_int_session(s) for s in raw.split(";") if s.strip()}
        out[int(row["global_uid"])] = sessions
    return out


def load_trimmed_verdicts(trimmed_path) -> Dict[int, str]:
    """{global_uid -> trimmed_verdict} from verdicts_trimmed.csv."""
    df = pd.read_csv(trimmed_path)
    return {int(r["global_uid"]): str(r["trimmed_verdict"]) for _, r in df.iterrows()}


def resolve_row_verdict(
    global_uid,
    session,
    kept_map: Dict[int, Set[int]],
    trimmed_verdict_map: Dict[int, str],
) -> str:
    """Return track_verdict for one (global_uid, session) row."""
    uid = int(global_uid)
    if uid not in trimmed_verdict_map:
        return "unknown"
    if _to_int_session(session) in kept_map.get(uid, set()):
        return trimmed_verdict_map[uid]
    return "suspect"
