"""Unified project configuration for the vis_detect_analysis pipeline.

**Single source of truth** for paths, color palettes, learning-stage
definitions, session filtering, HMM state labels, date utilities, and
manifest loading.  All analysis constants (event windows, change sizes,
PSTH defaults) are re-exported from :mod:`visdetect.analysis.constants`
so callers need only one import target.

Both ``analysis_suite`` and ``AI_exploration`` thin-wrapper configs
re-export everything from this module, ensuring consistency across the
entire project.

Usage
-----
    from visdetect.analysis.config import (
        STAGE_ORDER, STAGE_COLORS, SESSION_FILTER,
        load_staging_manifest, load_filtered_manifest,
        SMALL_CHANGE_SIZES, BIG_CHANGE_SIZES,
    )
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# ── Re-export ALL constants from the canonical constants module ──────
from visdetect.analysis.constants import (          # noqa: F401
    EVENT_RESPONSIVENESS_WINDOWS,
    EVENT_VALID_OUTCOMES,
    SMALL_CHANGE_SIZES,
    BIG_CHANGE_SIZES,
    ALL_GO_CHANGE_SIZES,
    FA_RT_SPLIT,
    TF_PULSE_PRE_WINDOW,
    TF_PULSE_POST_WINDOW,
    TF_PULSE_WINDOW,
    TF_FAST_THRESH_LOG2,
    TF_SLOW_THRESH_LOG2,
    TF_SAMPLE_PERIOD,
    DEFAULT_BIN_SIZE,
    DEFAULT_SIGMA_MS,
    DEFAULT_MIN_FR,
    DEFAULT_Z_THRESH_TF,
    VIDEO_SYNC_SMOOTH_FRAMES,
    VIDEO_SYNC_DETECT_THRESH,
    VIDEO_SYNC_DETECT_THRESH_LOW,
    VIDEO_SYNC_CLUSTER_MS,
    VIDEO_SYNC_MATCH_REJECT_S,
    VIDEO_SYNC_RANSAC_THRESH_S,
    VIDEO_SYNC_MAX_DRIFT_PPM,
    VIDEO_SYNC_MAX_RESIDUAL_MS,
    VIDEO_SYNC_MIN_COVERAGE,
    VIDEO_SYNC_COARSE_SEARCH_S,
    VIDEO_SYNC_COARSE_STEP_S,
    VIDEO_SYNC_DERIV_SIGMA_MULT,
    VIDEO_SYNC_DERIV_MIN_THRESH,
    VIDEO_SYNC_DERIV_MAX_THRESH,
    VIDEO_SYNC_DERIV_PRE_FRAMES,
    VIDEO_SYNC_DERIV_SEARCH_FRAMES,
    VIDEO_SYNC_OUTLIER_N_ITER,
    VIDEO_SYNC_OUTLIER_SIGMA,
    VIDEO_SYNC_DEFAULT_EYE_ROI,
    VIDEO_SYNC_MASK_N_TRANSITIONS,
    VIDEO_SYNC_MASK_PRE_FRAMES,
    VIDEO_SYNC_MASK_POST_FRAMES,
    VIDEO_SYNC_MASK_MORPH_OPEN,
    VIDEO_SYNC_MASK_MIN_COMPONENT,
)

# =====================================================================
# Project root (two levels up from this file: src/visdetect/analysis/)
# =====================================================================
ROOT = str(Path(__file__).resolve().parents[3])

# =====================================================================
# Subject Configuration
# =====================================================================
# Allow subject to be configured via environment variable or default to BG_046
SUBJECT: str = os.getenv("VISDETECT_SUBJECT", "BG_046")

# =====================================================================
# Paths
# =====================================================================
PKL_DIR                = os.path.join(ROOT, "data", "pkls", SUBJECT) # f"{SUBJECT}_concat_sort")
GLT_PATH               = os.path.join(ROOT, "table_output", SUBJECT, "Grand_Longitudinal_Table.csv")
STAGING_MANIFEST_PATH  = os.path.join(ROOT, "data", f"{SUBJECT}_staging_manifest.csv")

# HMM K = 3 (QC-passed, self_init_0p8)
HMM_DIR                = os.path.join(ROOT, "data", "hmm", SUBJECT)
HMM_STATE_ASSIGN_PATH  = os.path.join(HMM_DIR, "state_assignments_K3.csv")
HMM_PER_SESSION_PATH   = os.path.join(HMM_DIR, "per_session_state_metrics.csv")
HMM_TRAJECTORY_PATH    = os.path.join(HMM_DIR, "learning_trajectory.csv")
HMM_MODEL_SEL_PATH     = os.path.join(HMM_DIR, "model_selection.csv")

# Lick responsiveness CSVs
LICK_DIR               = os.path.join(ROOT, "FIGURES", "lick", SUBJECT)

# Mat files
MAT_DIR                = os.path.join(ROOT, "data", "mat", SUBJECT)

# Raw waveforms (UnitMatch input)
RAW_WF_DIR             = os.path.join(ROOT, "data", "unit_match", "input", SUBJECT)

# Waveform cell-type labels (regenerated per-subject by
# scripts/analysis/build_waveform_celltype_labels.py; the legacy
# AI_exploration/preTprime CSV is stale and must not be used)
WAVEFORM_LABELS_PATH   = os.path.join(ROOT, "data", SUBJECT, "waveform_celltype_labels.csv")

# Per-subject tracking-QC output dir (track_validation, verdicts, per-UID sheets,
# regenerated GLT verdicts). Subject-scoped so multiple subjects never collide.
TRACKING_QC_DIR        = os.path.join(ROOT, "FIGURES", "tracking_qc", SUBJECT)

# Pre-computed TF pulse traces (per-session NPZ files)
TF_TRACES_DIR          = os.path.join(ROOT, "data", "cache", "tf_traces", SUBJECT)

# Legacy QC sessions manifest (deprecated — use staging manifest)
QC_SESSIONS_PATH       = os.path.join(ROOT, "data", "pkls", SUBJECT,
                                      f"{SUBJECT}_sessions_manifest.csv")

# Camera data root (may live on a different drive from main data)
CAMERA_ROOT            = os.getenv("VISDETECT_CAMERA_ROOT",
                         os.path.join("X:/public/projects/BeJG_20230130_VisDetect",
                                      "wEPhys", "Cameras_sortIntoSubjects"))
VIDEO_SYNC_DIR         = os.path.join(ROOT, "data", "cache", "video_sync")
VIDEO_SYNC_FIG_DIR     = os.path.join(ROOT, "figures", "video_sync")

# Camera feature extraction cache directories
MOTION_ENERGY_DIR      = os.path.join(ROOT, "data", "cache", "motion_energy")
PUPIL_DIR              = os.path.join(ROOT, "data", "cache", "pupil")

# =====================================================================
# Learning stages (full set — used internally for manifest validation)
# =====================================================================
_ALL_STAGE_ORDER: List[str] = ["Naive", "Learning", "Expert"]
_ALL_STAGE_COLORS: Dict[str, str] = {
    "Naive":    "#c7e9c0",
    "Learning": "#74c476",
    "Expert":   "#238b45",
}
VALID_STAGES: Set[str] = set(_ALL_STAGE_ORDER)

# =====================================================================
# Session filtering preset
# =====================================================================
# Central definition for which sessions to include in analyses.
# Change these values here to affect ALL scripts project-wide.
# load_staging_manifest() applies these automatically.
#
# Set any key to None to disable that filter.
SESSION_FILTER: Dict = {
    "include_stages":        ["Naive", "Learning", "Expert"],
    "exclude_stages":        None,
    "merge_naive_learning":  True,           # Naive → 'Learning' in stage column
    "stage_specific_dprime": None,           # No per-stage d' gate
    "min_trials":            150,            # Exclude sessions with < 150 trials
    "min_dprime":            0.8,            # Global d' floor
}

# =====================================================================
# Active stage order & colors (post-filter)
# =====================================================================
# When merge_naive_learning is True, Naive sessions are relabeled as
# Learning in the 'stage' column.  STAGE_ORDER and STAGE_COLORS therefore
# only contain the stages that exist after filtering — every script that
# iterates `for stage in STAGE_ORDER` or colours with STAGE_COLORS[stage]
# works without modification.
if SESSION_FILTER.get("merge_naive_learning", False):
    STAGE_ORDER:  List[str]      = ["Learning", "Expert"]
    STAGE_COLORS: Dict[str, str] = {s: _ALL_STAGE_COLORS[s] for s in STAGE_ORDER}
else:
    STAGE_ORDER  = list(_ALL_STAGE_ORDER)
    STAGE_COLORS = dict(_ALL_STAGE_COLORS)

# =====================================================================
# HMM state labels (K = 3)
# =====================================================================
HMM_LABEL_RENAME: Dict[str, str] = {
    "Engaged_1": "Disengaged",
    "Engaged_2": "Engaged",
    "Engaged_3": "Impulsive",
    "Biased":    "Impulsive",
}
HMM_STATE_ORDER: List[str] = ["Disengaged", "Engaged", "Impulsive"]
HMM_STATE_COLORS: Dict[str, str] = {
    "Disengaged": "#bdbdbd",
    "Engaged":    "#6baed6",
    "Impulsive":  "#fb6a4a",
}

# =====================================================================
# Cell-type colors
# =====================================================================
CELLTYPE_COLORS: Dict[str, str] = {
    "Narrow (FSI)":       "#e74c3c",
    "Broad (MSN/Proj)":   "#3498db",
}

# =====================================================================
# Outcome colors
# =====================================================================
OUTCOME_COLORS: Dict[str, str] = {
    "Hit":  "#4CAF50",
    "Miss": "#F44336",
    "FA":   "#FF9800",
    "CR":   "#7986CB",
}

# =====================================================================
# FA subtype colors (Stimulus-driven vs Impulsive false alarms)
# =====================================================================
FA_SUBTYPE_COLORS: Dict[str, str] = {
    "Stimulus-driven": "#e74c3c",
    "Impulsive":       "#3498db",
}

# =====================================================================
# Lick-valence colors (behavioral state labeler raster)
# =====================================================================
# Softened (desaturated) palette so the dense raster is easy on the eye; each hue
# is still legible down to the lowest change-size opacity (alpha 0.30). The abort
# grey is shared with the Abort state below so the two tracks echo each other.
LICK_VALENCE_COLORS: Dict[str, str] = {
    "appropriate_lick":   "#6fb58f",   # soft green   — hit on a real change
    "inappropriate_lick": "#e3897c",   # soft coral   — early lick or catch SDT-FA
    "nolick":             "#9488bf",   # soft lavender — miss or correct rejection
    "abort":              "#bdbdbd",   # grey (= Abort state)
    "ref":                "#ddd0b3",   # soft tan     — reflex lick (excluded from fractions)
}

# =====================================================================
# Behavioral-state colors (labeler state strips). Warm -> cool "arousal" ramp:
# Impulsive (warm red) -> StimSens (light blue, engaged) -> Disengaged (darker
# blue, withdrawn); Abort = neutral grey (shares the abort-outcome grey above).
# =====================================================================
STATE_LABEL_COLORS: Dict[str, str] = {
    "Impulsive":  "#ef6548",   # soft red — over-aroused / inappropriate-lick driven
    "StimSens":   "#6baed6",   # light blue — stimulus-sensitive engaged
    "Disengaged": "#3474ae",   # darker soft blue — withdrawn
    "Abort":      "#bdbdbd",   # grey — neutral 4th state (= abort outcome colour)
}

# =====================================================================
# Change-size groupings (derived from constants)
# =====================================================================
CHANGE_SIZES: List[float]  = sorted(ALL_GO_CHANGE_SIZES)   # [1.25, 1.35, 1.5, 2.0, 4.0]
CHANGE_SIZE_LABELS: List[str] = [str(cs) for cs in CHANGE_SIZES]
CHANGE_SIZE_POSITIONS: List[int] = list(range(len(CHANGE_SIZES)))  # equidistant

# Small (1.25-1.5x) vs big (2-4x) change-size split colours — the SMALL/BIG orange ramp
# used across change-detection figures (previously hardcoded per-script; canonical here).
CHANGE_SIZE_COLORS: Dict[str, str] = {
    "small": "#fdae6b",   # light orange — small change (1.25-1.5x)
    "big":   "#d94801",   # dark orange  — big change (2-4x)
}

# Up- vs down-modulation split colours (held-out modulation sign in PSTHs).
MODULATION_SIGN_COLORS: Dict[str, str] = {
    "up":   "#d73027",   # red  — up-modulated
    "down": "#4575b4",   # blue — down-modulated
}

# =====================================================================
# Event alignment windows (alias to constants module name)
# =====================================================================
EVENT_WINDOWS = EVENT_RESPONSIVENESS_WINDOWS

# =====================================================================
# PSTH / analysis defaults (re-exported from constants for convenience)
# =====================================================================
DEFAULT_ANALYSIS_WINDOW: Tuple[float, float] = (-1.0, 1.5)

# =====================================================================
# Date parsing utilities
# =====================================================================

def canonical_session_id(session) -> str:
    """Return the canonical 8-digit ``DDMMYYYY`` session-id STRING.

    THE single source of truth for session-id keys project-wide. Use this anywhere
    a session id is compared, joined, dict-keyed, or written/read across a CSV
    boundary -- never raw ``int(...)``, ``str(...)``, or an ad-hoc ``.zfill(8)``.

    Why this exists (a recurring, cross-project bug): session ids are ``DDMMYYYY``
    (e.g. ``01072025`` = 1 Jul 2025). Stored as int64 (the default when a CSV column
    is all-numeric) or cast via ``int()``, the leading-zero DAY of days 1-9 drops ->
    ``1072025`` (7 digits). There is NO ``1072025`` session; it is just
    ``int('01072025')``. Mixing the two forms silently breaks key lookups/joins
    (day-1-9 sessions miss) and ordering (``'1072025'`` sorts before ``'23062025'``
    lexically though 1 Jul is AFTER 23 Jun). This helper collapses every
    representation -- int64, numeric string, int-valued float (``1072025.0`` from a
    NaN-bearing column), and the float-as-string ``'1072025.0'`` (CSV float
    round-trip) -- to the same zfill8 string. Non-numeric ids (test mocks,
    subject-prefixed names) pass through unchanged. For chronological ordering use
    :func:`chronological_sort` / :func:`parse_session_date` (NOT a raw sort).

    Examples
    --------
    >>> canonical_session_id(1072025)        # int64
    '01072025'
    >>> canonical_session_id('1072025')      # int-form string
    '01072025'
    >>> canonical_session_id('01072025')     # already canonical
    '01072025'
    >>> canonical_session_id(1072025.0)      # int-valued float
    '01072025'
    >>> canonical_session_id('1072025.0')    # float-as-string (CSV round-trip)
    '01072025'
    """
    try:
        return str(int(session)).zfill(8)
    except (TypeError, ValueError):
        pass
    try:
        f = float(session)
        if f.is_integer():
            return str(int(f)).zfill(8)
    except (TypeError, ValueError):
        pass
    return str(session).strip()


SESSION_ID_COLUMNS = ("session_name", "session_date", "session", "session_id")


def restore_session_token(value) -> str:
    """Return `value` as a string with any int-stripped leading-zero DAY restored.

    WIDTH-PRESERVING, unlike :func:`canonical_session_id`. The repo carries two
    session-token families:

      * 8-digit ``DDMMYYYY``  (BG_046, BG_038, anatomy, most caches)
      * 6-digit ``DDMMYY``    (BG_031 / BG_039 raw tokens)

    An ``int()`` cast strips the leading-zero day of days 1-9 from either family,
    yielding a 7- or 5-digit token respectively. Both widths are unambiguous (a
    ``DDMMYY`` can never be 7 digits), so the stripped zero can be restored without
    knowing the subject::

        1072025  -> '01072025'    (7 -> 8, DDMMYYYY)
          50325  -> '050325'      (5 -> 6, DDMMYY)

    Anything else -- already-correct 8/6-digit ids, subject-prefixed tokens
    (``BG_012_01112023_pr``), re-recording suffixes (``01042025_v2``, ``19052025_b``)
    -- passes through unchanged.

    Use :func:`canonical_session_id` to force the 8-digit ``DDMMYYYY`` form for
    joins on a single DDMMYYYY subject; use THIS when normalizing a column that may
    span subjects (it will not mangle a 6-digit ``DDMMYY`` into ``00050325``).
    """
    s = str(value).strip()
    if s.endswith(".0") and s[:-2].isdigit():   # CSV float round-trip: '1072025.0'
        s = s[:-2]
    if s.isdigit() and len(s) in (5, 7):        # stripped leading-zero DAY
        return s.zfill(len(s) + 1)
    return s


def canonicalize_session_column(df, col: Optional[str] = None):
    """Make a DataFrame's session-id column safe to write to CSV. Returns `df`.

    Casts the column to width-preserving STRINGS via :func:`restore_session_token`,
    so that ``to_csv`` cannot emit an int64 that has silently dropped the leading-zero
    day of a day-1-9 session.

    ALWAYS call this immediately before ``to_csv()`` on any table carrying a session
    id. An all-numeric session column round-trips through pandas as int64, which is
    how 38,730 rows across 19 deliverables were corrupted (see
    ``tests/test_session_id_csv_integrity.py``).

    Returns a NEW frame; the input is never modified in place (mutating a caller's
    frame is a footgun -- e.g. an incremental upsert that aliases its argument would
    see the session column silently change dtype mid-call).

    Parameters
    ----------
    df : pandas.DataFrame
    col : str, optional
        Column to fix. If None, every column named in :data:`SESSION_ID_COLUMNS`
        that is present is fixed.

    Examples
    --------
    >>> df = canonicalize_session_column(df)
    >>> df.to_csv(path, index=False)     # now safe
    """
    df = df.copy()
    cols = [col] if col else [c for c in SESSION_ID_COLUMNS if c in df.columns]
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(restore_session_token)
    return df


def parse_session_date(session_int) -> Tuple[int, int, int]:
    """Convert a DDMMYYYY session id to a sortable (year, month, day) tuple.

    Robust to the int64 / leading-zero forms via :func:`canonical_session_id`.
    """
    s = canonical_session_id(session_int)
    dd, mm, yyyy = int(s[:2]), int(s[2:4]), int(s[4:])
    return (yyyy, mm, dd)


def session_date_key(session) -> Tuple[int, int, int]:
    """Sortable (year, month, day) for ANY session-token format used across subjects.

    Handles, in addition to the bare DDMMYYYY ints `parse_session_date` accepts:
      - subject-prefixed tokens   BG_049_01092025  (the new subjects' registry form)
      - 6-digit DDMMYY            050325 / BG_031_050325  ->  20YY
      - 7-digit DMMYYYY           1072025 (BG_046, leading zero stripped)
      - re-recording suffixes     BG_031_19052025_b, BG_039_01042025_v2 (date taken,
                                  suffix ignored)

    Unlike parse_session_date this is string-safe (no int() on a prefixed token).
    """
    s = re.sub(r"^BG_\d+_", "", str(session))      # drop subject prefix if present
    m = re.match(r"(\d+)", s)                        # leading digit run (drops _b/_v2)
    if not m:
        raise ValueError(f"no date digits in session token {session!r}")
    d = m.group(1)
    if len(d) in (5, 7):                             # DMMYY / DMMYYYY: int64 dropped
        d = d.zfill(len(d) + 1)                      # the leading-zero DAY -> restore it
    if len(d) == 6:                                  # DDMMYY -> 20YY
        return (2000 + int(d[4:6]), int(d[2:4]), int(d[:2]))
    d = d.zfill(8)                                   # 8-digit -> DDMMYYYY
    return (int(d[4:]), int(d[2:4]), int(d[:2]))


def session_int_to_iso(session_int) -> str:
    """Convert a DDMMYYYY session id to a 'YYYY-MM-DD' string."""
    s = canonical_session_id(session_int)
    return f"{s[4:]}-{s[2:4]}-{s[:2]}"


def iso_to_session_int(iso_str) -> Optional[int]:
    """Convert 'YYYY-MM-DD' to DDMMYYYY int."""
    parts = iso_str.split("-")
    if len(parts) == 3:
        return int(parts[2] + parts[1] + parts[0])
    return None


def chronological_sort(session_ints):
    """Return session ints sorted chronologically."""
    return sorted(session_ints, key=parse_session_date)


# Alias used by AI_exploration scripts
chronological_session_order = chronological_sort


# =====================================================================
# Manifest loading
# =====================================================================

def load_staging_manifest(
    qc_only: bool = True,
    apply_filter: bool = True,
    manifest_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load staging manifest, chronologically sorted, with session_idx.

    Parameters
    ----------
    qc_only : bool
        If True (default), return only Naive/Learning/Expert sessions.
        If False, also include Disengaged (but never Excluded).
    apply_filter : bool
        If True (default), apply :data:`SESSION_FILTER` so all scripts
        share the same session selection.  Set to False for the
        unfiltered manifest (e.g. full learning trajectory plots).
    manifest_path : str or Path, optional
        Override the default :data:`STAGING_MANIFEST_PATH`.  Useful for
        CLI scripts that accept ``--manifest`` arguments.  When None,
        uses the canonical path.
    """
    from visdetect.analysis.behavior import filter_manifest_by_stage

    path = str(manifest_path) if manifest_path else STAGING_MANIFEST_PATH
    if not os.path.exists(path):
        # Subjects other than BG_046 (BG_031/038/039/049) have pkls + UM output but
        # NO staging manifest. Degrade gracefully to an EMPTY manifest so callers
        # fall back to 'Unknown' stage (cosmetic) instead of crashing on read_csv.
        # Session lists for these subjects must come from the registry or
        # list_pkl_sessions(), not the manifest.
        return pd.DataFrame({c: [] for c in
                             ["session_name", "date", "stage",
                              "n_go", "n_catch", "dprime", "session_idx"]})
    manifest = pd.read_csv(path, dtype={"session_name": str, "date": str})
    if qc_only:
        manifest = manifest[manifest["stage"].isin(VALID_STAGES)].copy()
    else:
        manifest = manifest[manifest["stage"] != "Excluded"].copy()

    # Chronological sort
    manifest["_sort"] = manifest["session_name"].apply(
        lambda x: parse_session_date(int(x))
    )
    manifest = manifest.sort_values("_sort").reset_index(drop=True)
    manifest.drop(columns=["_sort"], inplace=True)

    # Apply centralized filter
    if apply_filter and SESSION_FILTER:
        manifest = filter_manifest_by_stage(
            manifest,
            include_stages=SESSION_FILTER.get("include_stages"),
            exclude_stages=SESSION_FILTER.get("exclude_stages"),
            merge_naive_learning=SESSION_FILTER.get("merge_naive_learning", False),
            min_trials=SESSION_FILTER.get("min_trials"),
            min_dprime=SESSION_FILTER.get("min_dprime"),
            stage_specific_dprime=SESSION_FILTER.get("stage_specific_dprime"),
        )

    manifest["session_idx"] = range(len(manifest))
    return manifest


def load_valid_sessions() -> set:
    """Return the set of CANONICAL session-id STRINGS for all filtered valid sessions.

    Ids are the 8-digit ``DDMMYYYY`` string form (``'01072025'``), NOT ints: an int
    cast drops the leading-zero DAY of days 1-9, so a membership test against an
    int-domain set silently excludes those sessions. Compare against
    :func:`canonical_session_id` of whatever id you hold.
    """
    manifest = load_staging_manifest(qc_only=True, apply_filter=True)
    return {canonical_session_id(s) for s in manifest["session_name"]}


def load_filtered_manifest(
    include_stages: Optional[List[str]] = None,
    exclude_stages: Optional[List[str]] = None,
    merge_naive_learning: bool = False,
    min_trials: Optional[int] = None,
    min_dprime: Optional[float] = None,
    stage_specific_dprime: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Load staging manifest with custom one-off filtering.

    Bypasses :data:`SESSION_FILTER` and applies the specified parameters
    instead.  See :func:`visdetect.analysis.behavior.filter_manifest_by_stage`
    for detailed parameter docs.

    Parameters
    ----------
    include_stages : list of str, optional
        Stages to include.
    exclude_stages : list of str, optional
        Stages to exclude.
    merge_naive_learning : bool
        Create ``stage_group`` column mapping Naive → Learning.
    min_trials : int, optional
        Minimum n_go + n_catch.
    min_dprime : float, optional
        Global d' threshold.
    stage_specific_dprime : dict, optional
        Per-stage d' thresholds.

    Returns
    -------
    pd.DataFrame
        Filtered manifest with ``session_idx`` and chronological ordering.
    """
    from visdetect.analysis.behavior import filter_manifest_by_stage

    manifest = load_staging_manifest(qc_only=False, apply_filter=False)
    if manifest.empty:                       # no staging manifest for this subject
        manifest["session_idx"] = range(len(manifest))
        return manifest
    manifest = filter_manifest_by_stage(
        manifest,
        include_stages=include_stages,
        exclude_stages=exclude_stages,
        merge_naive_learning=merge_naive_learning,
        min_trials=min_trials,
        min_dprime=min_dprime,
        stage_specific_dprime=stage_specific_dprime,
    )
    manifest["session_idx"] = range(len(manifest))
    return manifest
