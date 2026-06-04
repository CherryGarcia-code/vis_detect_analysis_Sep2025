"""Canonical tracking-registry adapters (M1).

Bridges the canonical LONG registry (session, ks_unit_id, global_uid) and the
WIDE CellRegistry (UID-indexed; session-date columns; ks_unit_id cells) that
``scripts/analysis/build_longitudinal_table.py`` consumes. Registry-agnostic:
any method that emits the canonical long form (UM 3.2.9 now, DeepUM later) can
drive the same pipeline, so Global_UID and track_verdict share one ID space.

See docs/superpowers/specs/2026-06-03-presentation-prep-roadmap-design.md (§9).
"""
from __future__ import annotations

from typing import Dict, Set

import pandas as pd

CANONICAL_COLS = ["session", "ks_unit_id", "global_uid"]


def load_canonical_long(path) -> pd.DataFrame:
    """Load a canonical long registry.

    Columns (after normalization): session (str, 8-digit DDMMYYYY), ks_unit_id
    (int), global_uid (int). Accepts ``ks_id`` as an alias for ``ks_unit_id``.
    """
    df = pd.read_csv(path, dtype={"session": str})
    if "ks_unit_id" not in df.columns and "ks_id" in df.columns:
        df = df.rename(columns={"ks_id": "ks_unit_id"})
    missing = [c for c in CANONICAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"registry missing columns {missing}; has {list(df.columns)}"
        )
    df = df[CANONICAL_COLS].copy()
    # Zero-pad to 8-digit DDMMYYYY so wide columns survive build_grand_table's
    # `len(c) == 8` session filter (it silently drops 7-digit single-digit-day dates).
    df["session"] = df["session"].astype(str).str.strip().str.zfill(8)
    df["ks_unit_id"] = df["ks_unit_id"].astype(int)
    df["global_uid"] = df["global_uid"].astype(int)
    return df


def find_cluster_collisions(long_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where one (session, ks_unit_id) is claimed by >1 global_uid.

    This is the bimodal-ISI matching failure (two different units fused under
    distinct tracked IDs) that P0's dedupe guard catches downstream. Empty frame
    if the registry is clean.
    """
    n_uid = long_df.groupby(["session", "ks_unit_id"])["global_uid"].transform("nunique")
    return long_df[n_uid > 1].copy()


def resolve_collisions(
    long_df: pd.DataFrame,
    kept_sessions_by_uid: Dict[int, Set[int]],
) -> pd.DataFrame:
    """Resolve (session, ks_unit_id)-claimed-by->1-UID collisions.

    Policy (trust rule, 2026-06-03): for a contested cluster, keep the UID whose
    stable kept-subset includes that session; if exactly one UID qualifies, keep
    it and drop the others; if zero or more than one qualify, drop ALL claims on
    that cluster (ambiguous -> excluded from the registry). Uncontested rows pass
    through untouched.

    Parameters
    ----------
    long_df : canonical long registry (from load_canonical_long).
    kept_sessions_by_uid : {global_uid -> set of kept sessions as int}.
    """
    collisions = find_cluster_collisions(long_df)
    if collisions.empty:
        return long_df.copy()

    contested_keys = set(map(tuple, collisions[["session", "ks_unit_id"]].values))
    keep_rows = []
    for idx, row in long_df.iterrows():
        key = (row["session"], row["ks_unit_id"])
        if key not in contested_keys:
            keep_rows.append(idx)
            continue
        # Among UIDs claiming this cluster, which keep this session in their subset?
        sess_int = int(row["session"])
        claimants = long_df[(long_df["session"] == row["session"]) &
                            (long_df["ks_unit_id"] == row["ks_unit_id"])]["global_uid"]
        supported = [u for u in claimants
                     if sess_int in kept_sessions_by_uid.get(int(u), set())]
        if len(supported) == 1 and int(row["global_uid"]) == supported[0]:
            keep_rows.append(idx)
        # else: drop (ambiguous or unsupported)
    return long_df.loc[keep_rows].copy()


def long_to_cellregistry(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot canonical long -> wide CellRegistry consumed by build_grand_table.

    index = global_uid; columns = 8-digit session strings; cells = ks_unit_id
    (``;``-joined string when a UID has multiple clusters in one session).
    """
    def _join(s):
        vals = sorted(int(v) for v in s)
        return ";".join(str(v) for v in vals) if len(vals) > 1 else str(vals[0])

    wide = (long_df
            .groupby(["global_uid", "session"])["ks_unit_id"]
            .apply(_join)
            .unstack("session"))
    wide.index.name = "UID"
    return wide
