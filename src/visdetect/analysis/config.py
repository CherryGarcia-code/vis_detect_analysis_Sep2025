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
GLT_PATH               = os.path.join(ROOT, "table_output", "Grand_Longitudinal_Table.csv")
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

# Waveform cell-type labels (pre-computed)
WAVEFORM_LABELS_PATH   = os.path.join(ROOT, "AI_exploration", "figures",
                                      "waveform_celltype_labels.csv")

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
LICK_VALENCE_COLORS: Dict[str, str] = {
    "appropriate_lick":   "#2e8b57",   # green  — hit on a real change
    "inappropriate_lick": "#d6453a",   # red    — early lick or catch SDT-FA
    "nolick":             "#7b5cb8",   # purple — miss or correct rejection
    "abort":              "#9aa0a6",   # grey
    "ref":                "#d9c7a0",   # muted  — reflex lick (excluded from fractions)
}

# =====================================================================
# Change-size groupings (derived from constants)
# =====================================================================
CHANGE_SIZES: List[float]  = sorted(ALL_GO_CHANGE_SIZES)   # [1.25, 1.35, 1.5, 2.0, 4.0]
CHANGE_SIZE_LABELS: List[str] = [str(cs) for cs in CHANGE_SIZES]
CHANGE_SIZE_POSITIONS: List[int] = list(range(len(CHANGE_SIZES)))  # equidistant

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

def parse_session_date(session_int) -> Tuple[int, int, int]:
    """Convert DDMMYYYY int to sortable (year, month, day) tuple."""
    s = str(int(session_int)).zfill(8)
    dd, mm, yyyy = int(s[:2]), int(s[2:4]), int(s[4:])
    return (yyyy, mm, dd)


def session_int_to_iso(session_int) -> str:
    """Convert DDMMYYYY int to 'YYYY-MM-DD' string."""
    s = str(int(session_int)).zfill(8)
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
    """Return set of session-name ints for all filtered valid sessions."""
    manifest = load_staging_manifest(qc_only=True, apply_filter=True)
    return set(manifest["session_name"].astype(int).tolist())


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
