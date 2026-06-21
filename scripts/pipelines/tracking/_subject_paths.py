"""Per-subject path resolution for the multi-subject track-curation CLIs.

BG_046 is the historical special case: its UnitMatch output dir is ``all42`` and
its registry/session tokens are BARE dates (``1072025``), so pkls are
``BG_046_{date}.pkl`` and raw-wf subdirs are the zero-padded date. ALL other
subjects use ``all_sessions`` and FULLY-PREFIXED session tokens
(``BG_049_01092025``) that are themselves the pkl stem AND the raw-wf subdir name.

Every curation CLI builds its paths through these helpers so the only
subject-specific knowledge lives here. ``session_date_key`` (re-exported from
visdetect.analysis.config) is the one parser that handles all token formats.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from visdetect.analysis.config import session_date_key  # noqa: F401 (re-export)

UM_BASE = Path("X:/public/projects/BeJG_20230130_VisDetect/wEPhys")


def _repo_root() -> Path:
    # this file: scripts/pipelines/tracking/_subject_paths.py -> parents[3] = repo root
    return Path(__file__).resolve().parents[3]


# ── UnitMatch output (ceph / X:) ──────────────────────────────────────
def um_output_root(subject: str) -> Path:
    umdir = "all42" if subject == "BG_046" else "all_sessions"
    return UM_BASE / subject / "unit_match" / "output" / umdir


def um_registry(subject: str) -> Path:
    return um_output_root(subject) / "unit_index.csv"


def um_prob_matrix(subject: str) -> Path:
    return um_output_root(subject) / "batch0" / "output_prob_matrix.npy"


def um_prob_index(subject: str) -> Path:
    return um_output_root(subject) / "batch0" / "unit_index.csv"


# ── Local repo data/output (subject-scoped) ───────────────────────────
def pkl_dir(subject: str) -> Path:
    return _repo_root() / "data" / "pkls" / subject


def raw_wf_root(subject: str) -> Path:
    return _repo_root() / "data" / "unit_match" / "input" / subject


def states_dir(subject: str) -> Path:
    return _repo_root() / "data" / "cache" / "states" / subject


def tags_dir(subject: str) -> Path:
    return _repo_root() / "data" / "cache" / "state_tags" / subject


def curation_out_dir(subject: str) -> Path:
    return _repo_root() / "FIGURES" / "tracking_qc" / subject / "curation"


def sheets_dir(subject: str) -> Path:
    return curation_out_dir(subject) / "sheets"


def drift_csv(subject: str) -> Path:
    return _repo_root() / "FIGURES" / "tracking_qc" / subject / "intersession_drift.csv"


def features_cache(subject: str) -> Path:
    return _repo_root() / "data" / "cache" / f"curation_features_{subject}.pkl"


# ── Session token -> pkl / raw-wf resolution ──────────────────────────
def _bare_date(session) -> str:
    """The numeric date part of a session token (strips a BG_xxx_ prefix)."""
    return re.sub(r"^BG_\d+_", "", str(session))


def session_pkl(subject: str, session, pkl_directory) -> Optional[Path]:
    """On-disk pkl Path for a session token, or None. Tolerant of both naming
    schemes: prefixed token IS the stem (new subjects); BARE date needs the
    ``{subject}_`` prefix + zfill(8) (BG_046)."""
    s = str(session)
    bare = _bare_date(s)
    cands = [f"{s}.pkl", f"{subject}_{s}.pkl"]
    if bare.isdigit():
        cands += [f"{subject}_{bare.zfill(8)}.pkl", f"{bare.zfill(8)}.pkl"]
    seen = set()
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        p = Path(pkl_directory) / c
        if p.exists():
            return p
    return None
