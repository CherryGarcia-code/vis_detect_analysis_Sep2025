"""Unified data loading for the analysis suite.

All data access goes through this module: session pkl files, the Grand
Longitudinal Table, staging manifest, HMM state assignments, lick
responsiveness CSVs, and waveform cell-type labels.

Manifest loading (load_staging_manifest, load_filtered_manifest,
load_valid_sessions) is delegated to the canonical
:mod:`visdetect.analysis.config` module.  This file adds suite-specific
loaders (load_session, session_iterator, load_glt, etc.).
"""

import gc
import os
import sys
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Ensure visdetect is importable
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from config import (
    CACHE_DIR,
    GLT_PATH,
    HMM_LABEL_RENAME,
    HMM_PER_SESSION_PATH,
    HMM_STATE_ASSIGN_PATH,
    HMM_TRAJECTORY_PATH,
    LICK_DIR,
    PKL_DIR,
    STAGING_MANIFEST_PATH,
    VALID_STAGES,
    WAVEFORM_LABELS_PATH,
    chronological_sort,
    parse_session_date,
)

# Re-export canonical manifest loaders from visdetect.analysis.config
# so callers that do `from loader import load_staging_manifest` still work.
from visdetect.analysis.config import (   # noqa: F401
    load_staging_manifest,
    load_filtered_manifest,
    load_valid_sessions,
)

from visdetect.core.session import load_session as _load_session_raw


# ── Session loading ───────────────────────────────────────────────────

def load_session(session_name) -> "Session":
    """Load a single session by name (DDMMYYYY int or string).

    Wraps visdetect.core.session.load_session with path resolution.
    """
    name_str = str(int(session_name)).zfill(8)
    pkl_path = os.path.join(PKL_DIR, f"BG_046_{name_str}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Session pkl not found: {pkl_path}")
    return _load_session_raw(pkl_path)


def session_iterator(
    stages: Tuple[str, ...] = ("Naive", "Learning", "Expert"),
) -> Iterator[Tuple[int, str, "Session"]]:
    """Yield (session_name_int, stage, Session) in chronological order.

    Processes one session at a time and garbage-collects after each yield
    to keep memory manageable.
    """
    manifest = load_staging_manifest(qc_only=True)
    manifest = manifest[manifest["stage"].isin(set(stages))]
    for _, row in manifest.iterrows():
        session_name = int(row["session_name"])
        stage = row["stage"]
        sess = load_session(session_name)
        yield session_name, stage, sess
        del sess
        gc.collect()


# ── Staging manifest ──────────────────────────────────────────────────
# load_staging_manifest, load_filtered_manifest, load_valid_sessions are
# re-exported from visdetect.analysis.config (see imports above).


def get_session_stage(session_name) -> Optional[str]:
    """Return the learning stage for a given session name."""
    manifest = load_staging_manifest(qc_only=False, apply_filter=False)
    row = manifest[manifest["session_name"].astype(int) == int(session_name)]
    if row.empty:
        return None
    return row.iloc[0]["stage"]


# ── Grand Longitudinal Table ─────────────────────────────────────────

def load_glt(qc_only: bool = True) -> pd.DataFrame:
    """Load Grand Longitudinal Table, optionally filtered to valid sessions.

    Merges stage and session_idx from the staging manifest.
    """
    glt = pd.read_csv(GLT_PATH)
    manifest = load_staging_manifest(qc_only=qc_only)
    # Build lookup maps
    date_to_stage = dict(
        zip(manifest["session_name"].astype(int), manifest["stage"])
    )
    date_to_idx = dict(
        zip(manifest["session_name"].astype(int), manifest["session_idx"])
    )
    if qc_only:
        valid = set(manifest["session_name"].astype(int))
        glt = glt[glt["Session_Date"].isin(valid)].copy()
    glt["stage"] = glt["Session_Date"].map(date_to_stage)
    glt["session_idx"] = glt["Session_Date"].map(date_to_idx)
    return glt


# ── HMM state assignments ────────────────────────────────────────────

def _rename_hmm_labels(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Rename HMM labels from CSV originals to display names."""
    if col in df.columns:
        df[col] = df[col].map(lambda x: HMM_LABEL_RENAME.get(x, x))
    return df


def _rename_hmm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename HMM state columns (frac_*, dprime_*) to display names."""
    rename_map = {}
    for old, new in HMM_LABEL_RENAME.items():
        for prefix in ("frac_", "dprime_"):
            old_col = f"{prefix}{old}"
            new_col = f"{prefix}{new}"
            if old_col in df.columns:
                rename_map[old_col] = new_col
    return df.rename(columns=rename_map)


def load_hmm_assignments(K: int = 3) -> pd.DataFrame:
    """Load per-trial HMM state assignments with renamed labels."""
    df = pd.read_csv(HMM_STATE_ASSIGN_PATH)
    df["session_name"] = df["session_name"].astype(int)
    df = _rename_hmm_labels(df, "hmm_state_label")
    return df


def load_hmm_per_session(K: int = 3) -> pd.DataFrame:
    """Load per-session HMM metrics with renamed labels."""
    df = pd.read_csv(HMM_PER_SESSION_PATH)
    df["session_name"] = df["session_name"].astype(int)
    df = _rename_hmm_labels(df, "label")
    return df


def load_hmm_trajectory(K: int = 3) -> pd.DataFrame:
    """Load HMM learning trajectory with renamed columns."""
    df = pd.read_csv(HMM_TRAJECTORY_PATH)
    df["session_name"] = df["session_name"].astype(int)
    df = _rename_hmm_columns(df)
    return df


# ── Lick responsiveness ──────────────────────────────────────────────

def load_lick_responsiveness(session_name) -> Optional[pd.DataFrame]:
    """Load pre-computed lick responsiveness CSV for a session.

    Returns None if the file doesn't exist.
    """
    name_str = str(int(session_name)).zfill(8)
    csv_path = os.path.join(LICK_DIR, name_str, "lick_responsiveness.csv")
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def load_all_lick_responsiveness() -> pd.DataFrame:
    """Load lick responsiveness for all available sessions, merged."""
    manifest = load_staging_manifest(qc_only=True)
    frames = []
    for _, row in manifest.iterrows():
        session_name = int(row["session_name"])
        df = load_lick_responsiveness(session_name)
        if df is not None:
            df["session_name"] = session_name
            df["stage"] = row["stage"]
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Pre-computed TF trace cache ───────────────────────────────────────

def load_tf_traces_npz(session_name):
    """Load cached TF pulse traces (NPZ) for a session.

    Returns (t_vec, cluster_ids, fast_z, slow_z, fast_z_sem, slow_z_sem,
            z_max_fast, z_min_fast, z_max_slow, z_min_slow) or None.
    NPZ files use zero-padded session names (e.g. 01072025).
    """
    from config import TF_TRACES_DIR
    sname_padded = str(int(session_name)).zfill(8)
    npz_path = os.path.join(TF_TRACES_DIR, f"BG_046_{sname_padded}_traces.npz")
    if not os.path.exists(npz_path):
        return None
    try:
        npz = np.load(npz_path, allow_pickle=False)
        return {
            "t_vec": npz["t_vec"],
            "cluster_ids": npz["cluster_ids"].astype(int),
            "fast_z": npz["fast_z"],
            "slow_z": npz["slow_z"],
            "fast_z_sem": npz["fast_z_sem"],
            "slow_z_sem": npz["slow_z_sem"],
            "z_max_fast": npz["z_max_fast"],
            "z_min_fast": npz["z_min_fast"],
            "z_max_slow": npz["z_max_slow"],
            "z_min_slow": npz["z_min_slow"],
        }
    except Exception as e:
        print(f"  Warning: failed to load {npz_path}: {e}")
        return None


# ── Waveform cell-type labels ────────────────────────────────────────

def load_waveform_labels() -> pd.DataFrame:
    """Load pre-computed waveform cell-type labels (Narrow/Broad).

    Normalizes column names so downstream code can use:
      session_name, cluster_id, cell_type
    """
    if os.path.exists(WAVEFORM_LABELS_PATH):
        df = pd.read_csv(WAVEFORM_LABELS_PATH)
        # Normalize column names from CSV (session_date -> session_name, celltype -> cell_type)
        rename = {}
        if "session_date" in df.columns and "session_name" not in df.columns:
            rename["session_date"] = "session_name"
        if "celltype" in df.columns and "cell_type" not in df.columns:
            rename["celltype"] = "cell_type"
        if rename:
            df = df.rename(columns=rename)
        return df
    raise FileNotFoundError(
        f"Waveform labels not found at {WAVEFORM_LABELS_PATH}. "
        "Run analysis_3_waveform_celltype.py in AI_exploration/ first."
    )


# ── Convenience: merged unit table ───────────────────────────────────

def build_unit_table(qc_only: bool = True) -> pd.DataFrame:
    """Build a comprehensive per-unit table merging GLT, lick, and cell-type data.

    Columns include: Global_UID, Session_Date, Cluster_ID, stage, session_idx,
    all GLT neural/behavioral metrics, is_lick_responsive, celltype, plus
    derived TF responsiveness columns.
    """
    glt = load_glt(qc_only=qc_only)

    # Derive TF responsiveness
    tf_cols = ["tf_z_max_fast", "tf_z_min_fast", "tf_z_max_slow", "tf_z_min_slow"]
    if all(c in glt.columns for c in tf_cols):
        glt["tf_z_abs_max"] = glt[tf_cols].abs().max(axis=1)
        glt["is_tf_responsive"] = glt["tf_z_abs_max"] >= 3.0
    else:
        glt["tf_z_abs_max"] = np.nan
        glt["is_tf_responsive"] = False

    # Merge lick responsiveness
    lick_all = load_all_lick_responsiveness()
    if not lick_all.empty and "cluster_id" in lick_all.columns:
        lick_summary = lick_all.groupby(["session_name", "cluster_id"]).agg(
            is_lick_responsive=("is_significant", "any"),
            lick_p_value=("p_value", "min"),
        ).reset_index()
        glt = glt.merge(
            lick_summary,
            left_on=["Session_Date", "Cluster_ID"],
            right_on=["session_name", "cluster_id"],
            how="left",
        )
        glt["is_lick_responsive"] = glt["is_lick_responsive"].fillna(False)
        glt.drop(columns=["session_name", "cluster_id"], errors="ignore", inplace=True)
    else:
        glt["is_lick_responsive"] = False
        glt["lick_p_value"] = np.nan

    # Merge waveform cell-type labels
    try:
        wf = load_waveform_labels()
        if "session_date" in wf.columns and "cluster_id" in wf.columns:
            wf_sub = wf[["session_date", "cluster_id", "celltype"]].copy()
            wf_sub["session_date"] = wf_sub["session_date"].astype(int)
            glt = glt.merge(
                wf_sub,
                left_on=["Session_Date", "Cluster_ID"],
                right_on=["session_date", "cluster_id"],
                how="left",
            )
            glt.drop(columns=["session_date", "cluster_id"], errors="ignore", inplace=True)
        else:
            glt["celltype"] = np.nan
    except FileNotFoundError:
        glt["celltype"] = np.nan

    return glt
