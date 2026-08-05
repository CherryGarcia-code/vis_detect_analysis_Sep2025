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
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .config import (
    CACHE_DIR,
    GLT_PATH,
    HMM_DIR,
    HMM_LABEL_RENAME,
    HMM_PER_SESSION_PATH,
    HMM_STATE_ASSIGN_PATH,
    HMM_TRAJECTORY_PATH,
    LICK_DIR,
    PKL_DIR,
    ROOT,
    STAGING_MANIFEST_PATH,
    SUBJECT,
    VALID_STAGES,
    WAVEFORM_LABELS_PATH,
    canonical_session_id,
    chronological_sort,
    parse_session_date,
)

# Re-export canonical manifest loaders from visdetect.analysis.config
# so callers that do `from loader import load_staging_manifest` still work.
from visdetect.analysis.config import (   # noqa: F401
    load_staging_manifest,
    load_filtered_manifest,
    load_valid_sessions,
    canonical_camera_session,
)

from visdetect.analysis.constants import DEFAULT_Z_THRESH_TF
from visdetect.core.session import load_session as _load_session_raw

# Canonical trimmed-verdict cohort location: repo-root FIGURES/tracking_qc/<SUBJECT>
# (where the QC-sheets pipeline writes it) — NOT analysis_suite/figures.
DEFAULT_VERDICTS_PATH = os.path.join(ROOT, "FIGURES", "tracking_qc", SUBJECT, "verdicts_trimmed.csv")


# ── Session loading ───────────────────────────────────────────────────

def _session_pkl_candidates(session_name) -> List[str]:
    """Ordered, de-duplicated date-format variants to try for ``session_name``.

    Handles subjects with mixed naming:
      - 8-digit DDMMYYYY (BG_046, BG_039, newer sessions)
      - 6-digit DDMMYY   (BG_031, BG_038, older sessions)
    Also converts between the two formats when needed.
    """
    digits = str(int(session_name))

    candidates = []
    # Standard zero-padding to 8 and 6 digits
    candidates.append(digits.zfill(8))
    candidates.append(digits.zfill(6))
    # If given 8-digit DDMMYYYY, also try 6-digit DDMMYY (strip century)
    if len(digits) == 8:
        candidates.append(digits[:4] + digits[6:])  # e.g. 25042025 -> 250425
    # If given 6-digit DDMMYY, also try 8-digit DDMMYYYY (expand century to 20xx)
    if len(digits.zfill(6)) == 6:
        padded6 = digits.zfill(6)
        candidates.append(padded6[:4] + "20" + padded6[4:])  # e.g. 250425 -> 25042025

    seen, ordered = set(), []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def list_session_recordings(session_name, subject: Optional[str] = None) -> List[str]:
    """EVERY pkl on disk for ``session_name``'s date — plain AND suffixed.

    Some recording days were RESTARTED after a problem, producing a second (third…)
    file distinguished by a suffix: ``BG_031_19052025.pkl`` + ``BG_031_19052025_b.pkl``.
    On other days the restart is the ONLY surviving file (``BG_046_05092025_b.pkl``,
    ``BG_038_17062025_c.pkl``, ``BG_039_{01,25}042025_v2.pkl``).

    Use this whenever a date could legitimately have more than one recording; it is the
    only way to see the twin that :func:`resolve_session_pkl` hides. Sorted, plain first.
    """
    import glob
    subj = subject or SUBJECT
    pkl_dir = os.path.join(ROOT, "data", "pkls", subj)
    out = []
    for candidate in _session_pkl_candidates(session_name):
        out.append(os.path.join(pkl_dir, f"{subj}_{candidate}.pkl"))
        out.extend(sorted(glob.glob(os.path.join(pkl_dir, f"{subj}_{candidate}_*.pkl"))))
    seen, ordered = set(), []
    for p in out:
        if p not in seen and os.path.exists(p):
            seen.add(p); ordered.append(p)
    return ordered


def resolve_session_pkl(session_name) -> Optional[str]:
    """Return the on-disk pkl path for ``session_name``, or None if none exists.

    Cheap (only ``os.path.exists`` checks, no unpickling), so it is safe to call
    to pre-filter a queue of sessions before loading any of them.

    Resolution order (2026-08-03): the PLAIN numeric filename always wins, exactly as
    before — this function's answer for any session that already resolved is unchanged.
    Only when NO plain file exists does it fall back to a uniquely-suffixed restart
    (e.g. ``BG_046_05092025_b.pkl``), which previously returned None despite the data
    being on disk. If several suffixed files exist and none is plain, it returns None
    ON PURPOSE — that date is genuinely ambiguous (BG_012 stores different task
    protocols this way: ``_prot4_lickEndsTrial``, ``_airpuff``, …), so the caller must
    disambiguate via :func:`list_session_recordings` rather than get a silent guess.
    """
    for candidate in _session_pkl_candidates(session_name):
        pkl_path = os.path.join(PKL_DIR, f"{SUBJECT}_{candidate}.pkl")
        if os.path.exists(pkl_path):
            return pkl_path
    suffixed = [p for p in list_session_recordings(session_name)
                if not os.path.basename(p)[:-4].split("_")[-1].isdigit()]
    return suffixed[0] if len(suffixed) == 1 else None


def session_exists(session_name) -> bool:
    """True if a pkl for ``session_name`` exists on disk (no load)."""
    return resolve_session_pkl(session_name) is not None


def list_pkl_sessions(subject: Optional[str] = None) -> List[str]:
    """Session-name strings (the date token) for every pkl on disk for ``subject``.

    A manifest-free session source: subjects other than BG_046 have pkls but no
    staging manifest, so callers that just need "every session this subject has"
    (e.g. cross-subject tagging) can enumerate them directly. Chronologically
    sorted. ``subject`` defaults to the active SUBJECT (VISDETECT_SUBJECT).
    """
    import glob
    subj = subject or SUBJECT
    pkl_dir = os.path.join(ROOT, "data", "pkls", subj)
    prefix = f"{subj}_"
    names = []
    for path in glob.glob(os.path.join(pkl_dir, f"{prefix}*.pkl")):
        base = os.path.basename(path)[:-4]      # strip ".pkl"
        token = base[len(prefix):] if base.startswith(prefix) else ""
        if token.isdigit():
            names.append(token)
        else:
            # Suffixed RESTART (e.g. "05092025_b", "17062025_c", "01042025_v2") — the
            # old `isdigit()` gate dropped these silently, hiding 4 dates whose ONLY
            # recording is a restart, and all 40 dates of BG_012 (whose files carry
            # protocol descriptors). Keep the DATE once; callers wanting every physical
            # recording for a date use list_session_recordings().
            head = token.split("_")[0]
            if head.isdigit():
                names.append(head)
    return chronological_sort(sorted(set(names)))


def load_session(session_name) -> "Session":
    """Load a single session by name (DDMMYYYY or DDMMYY int or string).

    Tries multiple date-format variants to handle subjects with mixed naming;
    see :func:`_session_pkl_candidates`.

    NOTE: frozen to the import-time ``config.SUBJECT`` (the ``VISDETECT_SUBJECT``
    env). For an explicit, subject-aware load (cross-subject tools that take a
    ``--subject`` flag) use :func:`load_session_for_subject` instead — this
    signature is kept unchanged for the many existing single-subject callers.
    """
    pkl_path = resolve_session_pkl(session_name)
    if pkl_path is not None:
        return _load_session_raw(pkl_path)

    raise FileNotFoundError(
        f"pkl not found for session '{session_name}' in {PKL_DIR} "
        f"(tried {SUBJECT}_<date>.pkl with candidates: "
        f"{_session_pkl_candidates(session_name)})"
    )


def resolve_subject_pkl(session, subject: Optional[str] = None) -> Optional[str]:
    """On-disk pkl path for *subject*'s *session*, or None if none exists.

    Subject-aware sibling of :func:`resolve_session_pkl`. ``resolve_session_pkl``
    is frozen to the import-time ``config.SUBJECT`` (the ``VISDETECT_SUBJECT``
    env), so a cross-subject tool that passes ``--subject`` but resolves through it
    silently loads BG_046's behaviour instead. This honours an explicit ``subject``:
    it enumerates that subject's pkls (:func:`list_pkl_sessions`, the on-disk
    convention ``data/pkls/<subject>/<subject>_<token>.pkl``) and matches each token
    to ``session`` via :func:`canonical_camera_session`, so a 6-digit ``DDMMYY``
    token matches an 8-digit ``DDMMYYYY`` request (and leading-zero days resolve).
    Cheap: only lists directory names, no unpickling — safe to pre-filter with.
    """
    subj = subject or SUBJECT
    try:
        want = canonical_camera_session(session)
    except (TypeError, ValueError):
        return None
    pkl_dir = os.path.join(ROOT, "data", "pkls", subj)
    for token in list_pkl_sessions(subj):
        if canonical_camera_session(token) == want:
            return os.path.join(pkl_dir, f"{subj}_{token}.pkl")
    return None


def load_session_for_subject(session, subject: Optional[str] = None) -> "Session":
    """Load *subject*'s :class:`Session`, honouring an explicit ``subject``.

    Unlike :func:`load_session` (frozen to ``config.SUBJECT``), this resolves the
    pkl for ``subject`` via :func:`resolve_subject_pkl` and loads it through the
    PATH-based core loader. Raises :class:`FileNotFoundError` with a clear message
    (not a bare traceback) when no pkl matches, so callers can surface a friendly
    error.
    """
    pkl_path = resolve_subject_pkl(session, subject)
    if pkl_path is None:
        subj = subject or SUBJECT
        raise FileNotFoundError(
            f"pkl not found for subject '{subj}' session '{session}' under "
            f"{os.path.join(ROOT, 'data', 'pkls', subj)} "
            f"(expected {subj}_<token>.pkl matching that date)."
        )
    return _load_session_raw(pkl_path)


def session_iterator(
    stages: Tuple[str, ...] = ("Naive", "Learning", "Expert"),
) -> Iterator[Tuple[str, str, "Session"]]:
    """Yield (session_id, stage, Session) in CHRONOLOGICAL order.

    ``session_id`` is the CANONICAL 8-digit ``DDMMYYYY`` STRING (e.g. ``'01072025'``),
    not an int -- an int cast silently drops the leading-zero DAY of days 1-9, and any
    downstream join/dict-key on that value then misses those sessions. Use the yielded
    id directly as a key; do not re-cast it.

    Processes one session at a time and frees it after each yield, so a loop over all
    ~45 sessions (~100+ MB each) stays flat in memory. Prefer this over a hand-rolled
    ``for ... load_session(...)`` loop: it gets the id form, the ordering, and the
    memory hygiene right by construction.
    """
    manifest = load_staging_manifest(qc_only=True)
    manifest = manifest[manifest["stage"].isin(set(stages))]
    stage_by_id = {
        canonical_session_id(row["session_name"]): row["stage"]
        for _, row in manifest.iterrows()
    }
    for session_id in chronological_sort(stage_by_id):
        sess = load_session(session_id)
        yield session_id, stage_by_id[session_id], sess
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

def load_glt(qc_only: bool = True, glt_path: Optional[str] = None) -> pd.DataFrame:
    """Load Grand Longitudinal Table, optionally filtered to valid sessions.

    Merges stage and session_idx from the staging manifest.

    Parameters
    ----------
    qc_only : bool
        Filter to QC-passed sessions from the staging manifest.
    glt_path : str, optional
        Override the default GLT_PATH (used for testing).

    Raises
    ------
    FileNotFoundError
        If the GLT CSV is absent, with a message pointing at the regenerator.
    """
    path = glt_path if glt_path is not None else GLT_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Grand Longitudinal Table not found at {path}. "
            "Regenerate it with: "
            "py scripts/analysis/build_longitudinal_table.py --workers 6  "
            "(requires a UnitMatch registry + unitmatch_env)."
        )
    glt = pd.read_csv(path)
    manifest = load_staging_manifest(qc_only=qc_only)
    date_to_stage = dict(zip(manifest["session_name"].astype(int), manifest["stage"]))
    date_to_idx = dict(zip(manifest["session_name"].astype(int), manifest["session_idx"]))
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
    path = os.path.join(HMM_DIR, f"state_assignments_K{K}.csv")
    df = pd.read_csv(path)
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
    from .config import TF_TRACES_DIR, SUBJECT
    sname_padded = str(int(session_name)).zfill(8)
    npz_path = os.path.join(TF_TRACES_DIR, f"{SUBJECT}_{sname_padded}_traces.npz")
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


# ── Detrended TF classification ─────────────────────────────────────

def load_tf_responsiveness_detrended() -> pd.DataFrame:
    """Load the detrended TF responsiveness CSV (from j_tf_detrended_classification.py).

    Returns a DataFrame with columns: session_name, cluster_id, stage,
    z_abs_max_standard, z_abs_max_detrended, is_tf_responsive_standard,
    is_tf_responsive_detrended, z_max_fast_dt, z_min_fast_dt,
    z_max_slow_dt, z_min_slow_dt.

    Returns empty DataFrame if the cache file does not exist yet.
    """
    from .config import CACHE_DIR
    path = os.path.join(CACHE_DIR, "tf_responsiveness_detrended.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def load_tf_classification_detrended() -> pd.DataFrame:
    """Load the detrended tier classification CSV (from g_tf_cell_classifier.py --detrend).

    Returns a DataFrame with columns: session_name, cluster_id, stage,
    tier, sub_type, and all per-unit metrics computed with linear detrending.

    Returns empty DataFrame if the cache file does not exist yet.
    """
    from .config import CACHE_DIR
    path = os.path.join(CACHE_DIR, "tf_cell_classification_detrended.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


# ── Waveform cell-type labels ────────────────────────────────────────

def load_waveform_labels(path: Optional[str] = None) -> pd.DataFrame:
    """Load pre-computed waveform cell-type labels.

    Normalizes column names so downstream code can use:
      session_name, cluster_id, cell_type

    Parameters
    ----------
    path : str, optional
        Override WAVEFORM_LABELS_PATH (used for testing / regenerated labels).

    Raises
    ------
    FileNotFoundError
        If the labels CSV is absent (callers may catch this and fall back).
    """
    p = path if path is not None else WAVEFORM_LABELS_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Waveform labels not found at {p}. Regenerate cell-type labels from "
            "the CURRENT per-session KS4 output (the FSI/SPN waveform workstream); "
            "the legacy AI_exploration/preTprime CSV is stale and must not be used."
        )
    df = pd.read_csv(p)
    rename = {}
    if "session_date" in df.columns and "session_name" not in df.columns:
        rename["session_date"] = "session_name"
    if "celltype" in df.columns and "cell_type" not in df.columns:
        rename["celltype"] = "cell_type"
    if rename:
        df = df.rename(columns=rename)
    return df


# ── Per-unit anatomical localization ────────────────────────────────

def load_unit_anatomy(path: Optional[str] = None) -> pd.DataFrame:
    """Per-unit anatomical localization (produced by scripts/anatomy/localize_units.py).

    Layout is per-subject: ``data/anatomy/<subject>/unit_anatomy.csv`` (one file each).
    All are concatenated (rows are keyed by Session_Date/Cluster_ID, so subjects don't
    collide). The legacy flat ``data/anatomy/unit_anatomy.csv`` is still honoured for
    back-compat. An explicit ``path`` overrides discovery. Empty DataFrame if none.
    """
    import glob
    if path is not None:
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    base = os.path.join(ROOT, "data", "anatomy")
    files = sorted(glob.glob(os.path.join(base, "*", "unit_anatomy.csv")))  # per-subject
    flat = os.path.join(base, "unit_anatomy.csv")                            # legacy flat
    if os.path.exists(flat):
        files.append(flat)
    frames = [pd.read_csv(f) for f in files]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


# ── Convenience: merged unit table ───────────────────────────────────

def build_unit_table(qc_only: bool = True, validate: bool = True,
                     verdicts_path: Optional[str] = None) -> pd.DataFrame:
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
        glt["is_tf_responsive"] = glt["tf_z_abs_max"] >= DEFAULT_Z_THRESH_TF
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

    # Merge waveform cell-type labels.
    # load_waveform_labels normalizes to session_name/cell_type, so accept either
    # naming and always populate `celltype` (fillna so unlabeled units are "unknown").
    try:
        wf = load_waveform_labels()
        sess_col = next((c for c in ("session_name", "session_date") if c in wf.columns), None)
        type_col = next((c for c in ("cell_type", "celltype") if c in wf.columns), None)
        if sess_col and "cluster_id" in wf.columns and type_col:
            wf_sub = wf[[sess_col, "cluster_id", type_col]].copy()
            wf_sub.columns = ["_wf_session", "_wf_cluster", "celltype"]
            wf_sub["_wf_session"] = wf_sub["_wf_session"].astype(int)
            wf_sub["_wf_cluster"] = wf_sub["_wf_cluster"].astype(int)
            # The waveform workstream owns `celltype`; drop any pre-existing column
            # so the merge can't suffix it to celltype_x/celltype_y.
            glt.drop(columns=["celltype"], errors="ignore", inplace=True)
            glt = glt.merge(
                wf_sub,
                left_on=["Session_Date", "Cluster_ID"],
                right_on=["_wf_session", "_wf_cluster"],
                how="left",
            )
            glt.drop(columns=["_wf_session", "_wf_cluster"], errors="ignore", inplace=True)
            glt["celltype"] = glt["celltype"].fillna("unknown")
        else:
            glt["celltype"] = "unknown"
    except FileNotFoundError:
        glt["celltype"] = "unknown"

    # Merge detrended TF responsiveness (if available)
    dt_df = load_tf_responsiveness_detrended()
    if not dt_df.empty and "cluster_id" in dt_df.columns:
        dt_sub = dt_df[["session_name", "cluster_id",
                        "z_abs_max_detrended", "is_tf_responsive_detrended"]].copy()
        dt_sub["session_name"] = dt_sub["session_name"].astype(int)
        glt = glt.merge(
            dt_sub,
            left_on=["Session_Date", "Cluster_ID"],
            right_on=["session_name", "cluster_id"],
            how="left",
        )
        glt["is_tf_responsive_detrended"] = glt["is_tf_responsive_detrended"].fillna(False)
        glt.drop(columns=["session_name", "cluster_id"], errors="ignore", inplace=True)
    else:
        glt["z_abs_max_detrended"] = np.nan
        glt["is_tf_responsive_detrended"] = False

    # Merge detrended tier classification (if available)
    tier_dt = load_tf_classification_detrended()
    if not tier_dt.empty and "cluster_id" in tier_dt.columns:
        tier_sub = tier_dt[["session_name", "cluster_id", "tier"]].copy()
        tier_sub = tier_sub.rename(columns={"tier": "tier_detrended"})
        tier_sub["session_name"] = tier_sub["session_name"].astype(int)
        glt = glt.merge(
            tier_sub,
            left_on=["Session_Date", "Cluster_ID"],
            right_on=["session_name", "cluster_id"],
            how="left",
        )
        glt["tier_detrended"] = glt["tier_detrended"].fillna("Non-responsive")
        glt.drop(columns=["session_name", "cluster_id"], errors="ignore", inplace=True)
    else:
        glt["tier_detrended"] = "Non-responsive"

    # ── Standardize tf_class from the detrended tier (TF-responsive workstream
    #    will overwrite this column with its by-eye-matched classifier output) ──
    if "tier_detrended" in glt.columns:
        glt["tf_class"] = glt["tier_detrended"].fillna("unclassified")
    else:
        glt["tf_class"] = "unclassified"

    # ── Resolve track_verdict per (Global_UID, Session_Date) (M1) ──
    vpath = verdicts_path or DEFAULT_VERDICTS_PATH
    if "Global_UID" in glt.columns and os.path.exists(vpath):
        from visdetect.analysis.track_verdict import (
            load_kept_map, load_trimmed_verdicts, resolve_row_verdict,
        )
        kept_map = load_kept_map(vpath)
        verd_map = load_trimmed_verdicts(vpath)
        # Session_Date is a GLT key (never NaN); it is cast to int just below /
        # enforced by validate_unit_table, so resolve_row_verdict's int parse is safe.
        glt["track_verdict"] = [
            resolve_row_verdict(u, s, kept_map, verd_map)
            for u, s in zip(glt["Global_UID"], glt["Session_Date"])
        ]

    # Merge anatomical localization (peak channel -> CCF + region).
    anat = load_unit_anatomy()
    anat_cols = ["peak_channel", "shank", "depth_um", "ccf_ap", "ccf_ml", "ccf_dv",
                 "region_acronym", "region_name", "region_coarse",
                 "region_confidence", "loc_method"]
    if not anat.empty and {"session_name", "cluster_id"}.issubset(anat.columns):
        anat_sub = anat[["session_name", "cluster_id"] + anat_cols].copy()
        anat_sub["session_name"] = anat_sub["session_name"].astype(int)
        anat_sub["cluster_id"] = anat_sub["cluster_id"].astype(int)
        anat_sub = anat_sub.drop_duplicates(subset=["session_name", "cluster_id"], keep="last")
        glt = glt.drop(columns=anat_cols, errors="ignore")
        glt = glt.merge(
            anat_sub, left_on=["Session_Date", "Cluster_ID"],
            right_on=["session_name", "cluster_id"], how="left",
        )
        glt.drop(columns=["session_name", "cluster_id"], errors="ignore", inplace=True)
        # Unmatched rows (sessions without a track artifact yet) get clean defaults,
        # mirroring the celltype merge. CCF coords / confidence stay NaN.
        for c, dflt in (("region_acronym", "unknown"), ("region_name", "unknown"),
                        ("region_coarse", "unknown"), ("loc_method", "none"),
                        ("peak_channel", -1), ("shank", -1)):
            glt[c] = glt[c].fillna(dflt)

    # ── Add not-yet-produced contract columns with their defaults ──
    from .unit_table_schema import add_label_defaults, validate_unit_table
    glt = add_label_defaults(glt)

    # ── Guard against row-multiplying merges: keys must be unique integers ──
    for key in ("Session_Date", "Cluster_ID"):
        if key in glt.columns:
            glt[key] = glt[key].astype(int)

    if validate:
        validate_unit_table(glt)

    return glt
