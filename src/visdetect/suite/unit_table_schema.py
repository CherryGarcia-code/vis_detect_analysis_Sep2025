"""Frozen schema contract for the per-unit label table (the roadmap 'spine').

Every workstream in the presentation-prep roadmap writes exactly ONE column
into this table; findings group by these columns and never re-derive labels.
See docs/superpowers/specs/2026-06-03-presentation-prep-roadmap-design.md (§3).

This module is pure (no I/O). It is the single source of truth for what a
valid unit-label table looks like.
"""
from __future__ import annotations

from typing import Dict, List, Set

import pandas as pd

# Key columns — together they uniquely identify one unit-session row.
KEY_COLUMNS: List[str] = ["Session_Date", "Cluster_ID"]

# Identity / metadata columns expected from the GLT producer.
IDENTITY_COLUMNS: List[str] = ["Global_UID", "stage", "session_idx"]

# One or more label columns per workstream + the default for not-yet-produced rows.
LABEL_DEFAULTS: Dict[str, object] = {
    "track_verdict": "unknown",        # tracking      -> trusted/review/suspect/unknown
    "celltype": "unknown",             # FSI/SPN waveform
    "opto_tag": "none",                # optotagging   -> D1/D2/none
    "tf_class": "unclassified",        # TF-responsive
    "is_lick_responsive": False,       # lick
}

# Anatomy localization columns (one workstream, several columns: a region label
# plus its CCF coordinates / confidence / method). Defaults mark not-yet-localized rows.
ANATOMY_DEFAULTS: Dict[str, object] = {
    "peak_channel": -1,
    "shank": -1,
    "depth_um": float("nan"),
    "ccf_ap": float("nan"),
    "ccf_ml": float("nan"),
    "ccf_dv": float("nan"),
    "region_acronym": "unknown",
    "region_name": "unknown",
    "region_coarse": "unknown",
    "region_confidence": float("nan"),
    "loc_method": "none",
}
LABEL_DEFAULTS.update(ANATOMY_DEFAULTS)

# Allowed categorical values for columns whose vocabulary is fixed.
# Columns not listed here are not value-checked (free/derived).
ALLOWED_VALUES: Dict[str, Set[object]] = {
    "track_verdict": {"trusted", "review", "suspect", "unknown"},
    "opto_tag": {"D1", "D2", "none"},
}
ALLOWED_VALUES["region_coarse"] = {
    "CP", "GPe", "CTX", "WM", "VS", "out", "other", "unknown",
}

CONTRACT_COLUMNS: List[str] = KEY_COLUMNS + IDENTITY_COLUMNS + list(LABEL_DEFAULTS)


class UnitTableContractError(ValueError):
    """Raised when a unit table violates the frozen schema contract."""


def add_label_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with every LABEL_DEFAULTS column present.

    Existing columns are preserved; missing ones are filled with their default.
    """
    out = df.copy()
    for col, default in LABEL_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
    return out


def validate_unit_table(df: pd.DataFrame) -> None:
    """Raise UnitTableContractError if df violates the contract; else return None."""
    missing = [c for c in CONTRACT_COLUMNS if c not in df.columns]
    if missing:
        raise UnitTableContractError(f"Missing contract columns: {missing}")

    for k in KEY_COLUMNS:
        if not pd.api.types.is_integer_dtype(df[k]):
            raise UnitTableContractError(
                f"Key column {k!r} must be integer dtype, got {df[k].dtype}"
            )

    n_dup = int(df.duplicated(subset=KEY_COLUMNS).sum())
    if n_dup:
        raise UnitTableContractError(
            f"{n_dup} duplicate (Session_Date, Cluster_ID) rows — a merge multiplied rows"
        )

    for col, allowed in ALLOWED_VALUES.items():
        bad = set(pd.Series(df[col]).dropna().unique()) - allowed
        if bad:
            raise UnitTableContractError(
                f"Column {col!r} has values outside {sorted(allowed)}: {sorted(bad, key=str)}"
            )
