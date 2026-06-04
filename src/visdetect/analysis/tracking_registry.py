"""Canonical tracking-registry adapters (M1).

Bridges the canonical LONG registry (session, ks_unit_id, global_uid) and the
WIDE CellRegistry (UID-indexed; session-date columns; ks_unit_id cells) that
``scripts/analysis/build_longitudinal_table.py`` consumes. Registry-agnostic:
any method that emits the canonical long form (UM 3.2.9 now, DeepUM later) can
drive the same pipeline, so Global_UID and track_verdict share one ID space.

See docs/superpowers/specs/2026-06-03-presentation-prep-roadmap-design.md (§9).
"""
from __future__ import annotations

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
