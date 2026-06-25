#!/usr/bin/env python3
"""Curate + QC-render DANT's BG_046 cross-session tracks (spec 2026-06-25).

Thin orchestration runner. Writes a curation-ready registry (dant_uid>0), then
drives the EXISTING registry-agnostic curation pipeline (curate_tracks.py /
render_curation_sheets.py) via subprocess with --liberal-col dant_uid, biophysical
-only (empty states dir -> corroborator abstains), into a DANT-specific out-dir so
the UnitMatch curation outputs are never touched. Held-out ISI AUC is computed
IN-PROCESS (validate_curation.py hardcodes the UM dir and would clobber it).

Run from the worktree root with the analysis interpreter:
    <PRIMARY>/.venv/Scripts/python.exe scripts/tracking_dant/curate_dant.py \
        [--steps registry,curate,validate,render,summary]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

WORKTREE_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_DEFAULT = Path("E:/python_analysis/git_repos/vis_detect_analysis_Sep2025")

# UnitMatch curation yardstick (project records, memory neuron_tracking_may2026);
# referenced for the summary, NOT re-run here.
UM_YARDSTICK: Dict[str, dict] = {
    "trusted": {"n": 22, "auc": 0.96},
    "review": {"n": 567},
    "suspect": {"n": 160},
}


def write_curation_registry(in_csv, out_csv) -> Tuple[int, int]:
    """Keep only tracked rows (dant_uid > 0); write session, ks_unit_id, dant_uid.

    Drops the untracked (dant_uid <= 0) rows so they cannot collapse into one bogus
    mega-track (the pipeline filters only on --min-span, not on uid value).
    Returns (n_rows_kept, n_distinct_uids).
    """
    df = pd.read_csv(in_csv, dtype={"session": str})
    kept = df[df["dant_uid"].astype(int) > 0][["session", "ks_unit_id", "dant_uid"]].copy()
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_csv, index=False)
    return len(kept), int(kept["dant_uid"].nunique())
