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


@dataclass(frozen=True)
class DantCurationPaths:
    """All paths the runner needs. Worktree-local outputs; PRIMARY data inputs."""
    worktree_root: Path
    primary_root: Path
    registry_in: Path          # data/cache/dant/BG_046/dant_registry.csv
    registry_curation: Path    # data/cache/dant/BG_046/dant_registry_curation.csv
    raw_wf_root: Path          # <PRIMARY>/data/unit_match/input/BG_046
    pkl_dir: Path              # <PRIMARY>/data/pkls/BG_046
    states_empty: Path         # empty -> corroborator abstains
    out_dir: Path              # FIGURES/tracking_dant/BG_046/curation
    cache_path: Path           # curation_features_dant.pkl
    sheets_dir: Path           # out_dir/sheets
    curate_script: Path        # scripts/pipelines/tracking/curate_tracks.py
    render_script: Path        # scripts/pipelines/tracking/render_curation_sheets.py

    @classmethod
    def default(cls, worktree_root, primary_root) -> "DantCurationPaths":
        wt = Path(worktree_root)
        pr = Path(primary_root)
        cache = wt / "data" / "cache" / "dant" / "BG_046"
        out = wt / "FIGURES" / "tracking_dant" / "BG_046" / "curation"
        tracking = wt / "scripts" / "pipelines" / "tracking"
        return cls(
            worktree_root=wt,
            primary_root=pr,
            registry_in=cache / "dant_registry.csv",
            registry_curation=cache / "dant_registry_curation.csv",
            raw_wf_root=pr / "data" / "unit_match" / "input" / "BG_046",
            pkl_dir=pr / "data" / "pkls" / "BG_046",
            states_empty=cache / "states_empty",
            out_dir=out,
            cache_path=cache / "curation_features_dant.pkl",
            sheets_dir=out / "sheets",
            curate_script=tracking / "curate_tracks.py",
            render_script=tracking / "render_curation_sheets.py",
        )


def build_curate_cmd(python_exe, paths: DantCurationPaths,
                     rebuild_cache: bool = True) -> List[str]:
    """argv for curate_tracks.py: biophysical-only, DANT out-dir, dant_uid column."""
    cmd = [
        str(python_exe), str(paths.curate_script),
        "--subject", "BG_046",
        "--registry", str(paths.registry_curation),
        "--liberal-col", "dant_uid",
        "--raw-wf-root", str(paths.raw_wf_root),
        "--pkl-dir", str(paths.pkl_dir),
        "--states-dir", str(paths.states_empty),
        "--out-dir", str(paths.out_dir),
        "--cache-path", str(paths.cache_path),
        "--drift-source", "none",
        "--min-span", "2",
    ]
    if rebuild_cache:
        cmd.append("--rebuild-cache")
    return cmd


def build_render_cmd(python_exe, paths: DantCurationPaths, tier: str,
                     max_uids: Optional[int] = None,
                     uids: Optional[List[int]] = None) -> List[str]:
    """argv for render_curation_sheets.py: one tier, DANT sheets dir, no pair scores."""
    cmd = [
        str(python_exe), str(paths.render_script),
        "--subject", "BG_046",
        "--tracks", str(paths.out_dir / "curated_tracks.csv"),
        "--registry", str(paths.registry_curation),
        "--liberal-col", "dant_uid",
        "--raw-wf-root", str(paths.raw_wf_root),
        "--pkl-dir", str(paths.pkl_dir),
        "--out-dir", str(paths.sheets_dir),
        "--tier", tier,
        "--no-pair-scores",
    ]
    if max_uids is not None:
        cmd += ["--max-uids", str(max_uids)]
    if uids:
        cmd += ["--uids", *[str(u) for u in uids]]
    return cmd
