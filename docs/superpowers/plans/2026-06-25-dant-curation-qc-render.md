# DANT Track Curation + QC-Sheet Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run DANT's BG_046 cross-session tracks through the project's existing, registry-agnostic curation + QC-sheet pipeline so we can visually inspect what DANT matched, tiered trusted/review/suspect with a held-out ISI AUC per tier — without touching the UnitMatch curation outputs.

**Architecture:** A single thin runner `scripts/tracking_dant/curate_dant.py` writes a curation-ready registry (`dant_uid > 0`), then drives the **existing** `curate_tracks.py` and `render_curation_sheets.py` CLIs via `subprocess` with `--liberal-col dant_uid`, biophysical-only curation (empty states dir → corroborator abstains), into a DANT-specific out-dir. The held-out ISI AUC is computed **in-process** (the `validate_curation.py` CLI hardcodes the UM dir and would clobber it). No edits to the shared pipeline (`scripts/pipelines/tracking/*`, `visdetect/*`).

**Tech Stack:** Python 3.11 (analysis venv at `E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv`), pandas, numpy, matplotlib, pytest 9.0.2. Reuses `visdetect.analysis.track_curation` (`partitioned_isi_hists`, `held_out_isi_auc_by_tier`) and the existing curation/render CLIs unchanged.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-06-25-dant-curation-qc-render-design.md` (commit 981aa31). This plan implements it; the spec governs on any conflict.
- **Worktree / branch:** `feature/dant-tracking` @ `E:/python_analysis/git_repos/vd_dant`. All paths below are relative to this worktree root unless prefixed `<PRIMARY>`.
- **`<PRIMARY>` = `E:/python_analysis/git_repos/vis_detect_analysis_Sep2025`** — raw waveforms (`<PRIMARY>/data/unit_match/input/BG_046`) and pkls (`<PRIMARY>/data/pkls/BG_046`) live here (the worktree's `data/` is gitignored/empty). Never use `X:`/Samba; never create junctions.
- **Interpreter:** run everything with `<PRIMARY>/.venv/Scripts/python.exe`. Subprocesses inherit it via `sys.executable`.
- **No edits to the shared pipeline.** `scripts/pipelines/tracking/*` and `visdetect/*` are read-only here. Do NOT use `build_qc_sheets.py`/`validate_long_tracks.py` (they hardcode `global_uid`).
- **`--liberal-col dant_uid` on every CLI call** (curate, render). It must be identical across calls — the curated uid keys back to the registry by it.
- **Clobber-safety:** curate + render write only to `--out-dir` under `FIGURES/tracking_dant/BG_046/curation/`. Validation JSON is written there in-process. Never write to `FIGURES/tracking_qc/BG_046/curation/` (the UM dir).
- **Biophysical-only:** `--states-dir` points at a fresh **empty** dir so the functional corroborator abstains; `--drift-source none`; `--no-pair-scores`.
- **Read every registry CSV with `dtype={"session": str}`** to preserve 8-digit DDMMYYYY tokens (leading zeros).
- **Opus 4.8 for every subagent** (implementer, reviewer, fixer). Never downgrade.
- **Presentation-ready output:** the rendered sheets + the tier/AUC summary figure save under `FIGURES/tracking_dant/BG_046/`.

## File Structure

| File | Responsibility |
|------|----------------|
| `scripts/tracking_dant/curate_dant.py` (create) | The whole runner: pure helpers (`write_curation_registry`, `DantCurationPaths`, `build_curate_cmd`, `build_render_cmd`, `write_validation_json`, `build_summary_table`) + step glue (`step_registry/curate/validate/render/summary`) + `main()`. |
| `tests/tracking_dant/test_curate_dant.py` (create) | Unit tests for the pure helpers (visdetect-free, fast). |
| `scripts/tracking_dant/README.md` (modify) | Add the curation + render command recipes. |

**Heavy imports are lazy.** Module top-level imports only `argparse, dataclasses, json, os, subprocess, sys, pathlib, typing, pandas`. `numpy`/`matplotlib`/`visdetect` are imported **inside** the functions that need them, so `import curate_dant` in tests stays cheap and visdetect-free.

---

## Task 1: Curation-ready registry filter + module skeleton

**Files:**
- Create: `scripts/tracking_dant/curate_dant.py`
- Test: `tests/tracking_dant/test_curate_dant.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `write_curation_registry(in_csv, out_csv) -> tuple[int, int]` — reads the DANT registry, keeps rows with `dant_uid > 0`, writes `session, ks_unit_id, dant_uid` to `out_csv` (creating parent dirs), returns `(n_rows_kept, n_distinct_uids)`. Module constants `WORKTREE_ROOT`, `PRIMARY_DEFAULT`, `UM_YARDSTICK`.

- [ ] **Step 1: Write the failing test**

Create `tests/tracking_dant/test_curate_dant.py`:

```python
import pandas as pd

import curate_dant


def test_write_curation_registry_keeps_positive_uids(tmp_path):
    src = tmp_path / "dant_registry.csv"
    pd.DataFrame({
        "session": ["01072025", "01072025", "02072025", "02072025"],
        "ks_unit_id": [3, 4, 5, 6],
        "dant_uid": [-1, 37, 0, 37],     # -1 untracked, 0 untracked, 37 tracked x2
    }).to_csv(src, index=False)
    out = tmp_path / "sub" / "dant_registry_curation.csv"

    n_rows, n_uids = curate_dant.write_curation_registry(src, out)

    assert out.exists()                       # parent dir created
    got = pd.read_csv(out, dtype={"session": str})
    assert list(got.columns) == ["session", "ks_unit_id", "dant_uid"]
    assert n_rows == 2 and n_uids == 1        # two rows of uid 37
    assert set(got["dant_uid"]) == {37}       # -1 and 0 dropped


def test_write_curation_registry_preserves_session_leading_zero(tmp_path):
    src = tmp_path / "dant_registry.csv"
    pd.DataFrame({
        "session": ["01072025"], "ks_unit_id": [3], "dant_uid": [37],
    }).to_csv(src, index=False)
    out = tmp_path / "dant_registry_curation.csv"

    curate_dant.write_curation_registry(src, out)

    got = pd.read_csv(out, dtype={"session": str})
    assert got["session"].iloc[0] == "01072025"   # not 1072025
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe" -m pytest tests/tracking_dant/test_curate_dant.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'curate_dant'` (file not created yet). The bare `import curate_dant` resolves via `tests/tracking_dant/conftest.py`, which already inserts `scripts/tracking_dant` on `sys.path`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/tracking_dant/curate_dant.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe" -m pytest tests/tracking_dant/test_curate_dant.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/tracking_dant/curate_dant.py tests/tracking_dant/test_curate_dant.py
git commit -m "feat(dant-curation): curation-ready registry filter (dant_uid>0)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Path resolver + command builders

**Files:**
- Modify: `scripts/tracking_dant/curate_dant.py` (append the dataclass + two builders)
- Test: `tests/tracking_dant/test_curate_dant.py` (append)

**Interfaces:**
- Consumes: `WORKTREE_ROOT`, `PRIMARY_DEFAULT` (Task 1).
- Produces:
  - `DantCurationPaths` frozen dataclass with fields `worktree_root, primary_root, registry_in, registry_curation, raw_wf_root, pkl_dir, states_empty, out_dir, cache_path, sheets_dir, curate_script, render_script` (all `Path`), plus classmethod `default(worktree_root, primary_root) -> DantCurationPaths`.
  - `build_curate_cmd(python_exe, paths: DantCurationPaths, rebuild_cache: bool = True) -> list[str]`
  - `build_render_cmd(python_exe, paths: DantCurationPaths, tier: str, max_uids: Optional[int] = None, uids: Optional[List[int]] = None) -> list[str]`

- [ ] **Step 1: Write the failing test**

Append to `tests/tracking_dant/test_curate_dant.py`:

```python
def _paths(tmp_path):
    return curate_dant.DantCurationPaths.default(
        worktree_root=tmp_path / "wt", primary_root=tmp_path / "primary")


def test_default_paths_target_dant_dir_not_um(tmp_path):
    p = _paths(tmp_path)
    # out-dir under tracking_dant (NOT tracking_qc), so UM curation is untouched
    assert "tracking_dant" in str(p.out_dir).replace("\\", "/")
    assert "tracking_qc" not in str(p.out_dir).replace("\\", "/")
    assert p.out_dir.name == "curation"
    assert p.sheets_dir == p.out_dir / "sheets"
    # raw waveforms + pkls live under PRIMARY
    assert str(p.raw_wf_root).replace("\\", "/").endswith(
        "primary/data/unit_match/input/BG_046")
    assert str(p.pkl_dir).replace("\\", "/").endswith("primary/data/pkls/BG_046")
    # the existing CLIs we drive
    assert p.curate_script.name == "curate_tracks.py"
    assert p.render_script.name == "render_curation_sheets.py"


def test_build_curate_cmd_has_critical_flags(tmp_path):
    p = _paths(tmp_path)
    cmd = curate_dant.build_curate_cmd("py.exe", p, rebuild_cache=True)
    assert cmd[:2] == ["py.exe", str(p.curate_script)]
    # flag/value pairs must be present and correct
    assert _pair(cmd, "--liberal-col") == "dant_uid"
    assert _pair(cmd, "--drift-source") == "none"
    assert _pair(cmd, "--min-span") == "2"
    assert _pair(cmd, "--registry") == str(p.registry_curation)
    assert _pair(cmd, "--states-dir") == str(p.states_empty)
    assert _pair(cmd, "--out-dir") == str(p.out_dir)
    assert _pair(cmd, "--cache-path") == str(p.cache_path)
    assert _pair(cmd, "--raw-wf-root") == str(p.raw_wf_root)
    assert _pair(cmd, "--pkl-dir") == str(p.pkl_dir)
    assert "--rebuild-cache" in cmd


def test_build_curate_cmd_omits_rebuild_when_false(tmp_path):
    cmd = curate_dant.build_curate_cmd("py.exe", _paths(tmp_path), rebuild_cache=False)
    assert "--rebuild-cache" not in cmd


def test_build_render_cmd_has_critical_flags(tmp_path):
    p = _paths(tmp_path)
    cmd = curate_dant.build_render_cmd("py.exe", p, tier="trusted", max_uids=25)
    assert _pair(cmd, "--liberal-col") == "dant_uid"
    assert _pair(cmd, "--tier") == "trusted"
    assert _pair(cmd, "--registry") == str(p.registry_curation)
    assert _pair(cmd, "--tracks") == str(p.out_dir / "curated_tracks.csv")
    assert _pair(cmd, "--out-dir") == str(p.sheets_dir)
    assert _pair(cmd, "--max-uids") == "25"
    assert "--no-pair-scores" in cmd


def test_build_render_cmd_uids_and_no_max(tmp_path):
    cmd = curate_dant.build_render_cmd(
        "py.exe", _paths(tmp_path), tier="review", uids=[1, 2, 3])
    assert "--max-uids" not in cmd
    i = cmd.index("--uids")
    assert cmd[i + 1:i + 4] == ["1", "2", "3"]
```

Add this helper near the top of the test file (after the imports):

```python
def _pair(cmd, flag):
    """Return the single value following `flag` in an argv list."""
    return cmd[cmd.index(flag) + 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe" -m pytest tests/tracking_dant/test_curate_dant.py -v`
Expected: FAIL — `AttributeError: module 'curate_dant' has no attribute 'DantCurationPaths'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/tracking_dant/curate_dant.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe" -m pytest tests/tracking_dant/test_curate_dant.py -v`
Expected: PASS (7 passed total).

- [ ] **Step 5: Commit**

```bash
git add scripts/tracking_dant/curate_dant.py tests/tracking_dant/test_curate_dant.py
git commit -m "feat(dant-curation): path resolver + curate/render command builders

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: In-process held-out ISI validation (clobber-safe)

**Files:**
- Modify: `scripts/tracking_dant/curate_dant.py` (append validation functions)
- Test: `tests/tracking_dant/test_curate_dant.py` (append)

**Why in-process:** `validate_curation.py` has no `--out-dir`; it hardcodes `sjp.curation_out_dir(subj)` (the UM dir) and would overwrite the UnitMatch `curation_validation.json`. We replicate its small loop (it is a faithful transcription of `validate_curation.py` lines 54-87) but write to the DANT out-dir.

**Interfaces:**
- Consumes: `DantCurationPaths` (Task 2); `visdetect.analysis.track_curation.partitioned_isi_hists` and `held_out_isi_auc_by_tier` (existing); `_subject_paths.session_pkl`; `visdetect.core.session.load_session`.
- Produces:
  - `write_validation_json(result: dict, out_dir) -> Path` — pure; writes `curation_validation.json` to the **given** out_dir, returns the path.
  - `collect_holdout_isi(kept_pairs: dict, subj: str, pkl_dir) -> dict` — heavy; `{(uid, session) -> holdout ISI hist}`. Lazy-imports visdetect.
  - `step_validate(paths: DantCurationPaths, subj: str = "BG_046") -> dict` — reads `curated_tracks.csv` + the curation registry, builds `kept_pairs`, collects holdout, calls `held_out_isi_auc_by_tier`, writes JSON to `paths.out_dir`, returns the result dict.

- [ ] **Step 1: Write the failing test**

The heavy functions need real pkls; the unit test covers only the pure, clobber-safety-critical part. Append to `tests/tracking_dant/test_curate_dant.py`:

```python
import json


def test_write_validation_json_writes_to_given_dir(tmp_path):
    result = {"trusted": {"auc": 0.9, "n_matched": 5, "n_nonmatched": 7}}
    out_dir = tmp_path / "FIGURES" / "tracking_dant" / "BG_046" / "curation"

    p = curate_dant.write_validation_json(result, out_dir)

    assert p == out_dir / "curation_validation.json"
    assert p.exists()                                  # parent dirs created
    assert json.loads(p.read_text())["trusted"]["auc"] == 0.9
    # clobber-safety: nothing written outside the given dir
    assert "tracking_qc" not in str(p).replace("\\", "/")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe" -m pytest tests/tracking_dant/test_curate_dant.py::test_write_validation_json_writes_to_given_dir -v`
Expected: FAIL — `AttributeError: module 'curate_dant' has no attribute 'write_validation_json'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/tracking_dant/curate_dant.py`:

```python
def write_validation_json(result: dict, out_dir) -> Path:
    """Write the per-tier AUC result to the GIVEN out_dir (never the UM dir)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "curation_validation.json"
    with open(p, "w") as f:
        json.dump(result, f, indent=2)
    return p


def _import_pipeline(subj: str):
    """Lazy-import the worktree pipeline modules (visdetect + _subject_paths).

    VISDETECT_SUBJECT must be set before _subject_paths is imported. We prepend the
    worktree src + tracking dir so we get THIS worktree's code, not the editable
    install pinned to PRIMARY (memory worktree_editable_install_pythonpath).
    """
    os.environ["VISDETECT_SUBJECT"] = subj
    sys.path.insert(0, str(WORKTREE_ROOT / "src"))
    sys.path.insert(0, str(WORKTREE_ROOT / "scripts" / "pipelines" / "tracking"))
    import _subject_paths as sjp
    from visdetect.analysis import track_curation as tc
    from visdetect.core.session import load_session
    return sjp, tc, load_session


def collect_holdout_isi(kept_pairs: Dict[Tuple[int, str], int], subj: str,
                        pkl_dir) -> Dict[Tuple[int, str], "object"]:
    """Holdout (odd-partition) log-ISI hist per kept (uid, session). Loads each
    session pkl once. Faithful to validate_curation.py lines 67-80."""
    import numpy as np
    sjp, tc, load_session = _import_pipeline(subj)
    holdout: Dict[Tuple[int, str], object] = {}
    for sess in sorted({s for (_, s) in kept_pairs}):
        pkl = sjp.session_pkl(subj, sess, pkl_dir)
        if pkl is None:
            print(f"  [validate] skip {sess}: no pkl", flush=True)
            continue
        S = load_session(str(pkl))
        cmap = {c.cluster_id: c for c in S.clusters}
        for (uid, s), kid in kept_pairs.items():
            if s != sess or kid not in cmap:
                continue
            _, hold = tc.partitioned_isi_hists(np.asarray(cmap[kid].spike_times))
            holdout[(uid, s)] = hold
        del S
    return holdout


def step_validate(paths: DantCurationPaths, subj: str = "BG_046") -> dict:
    """Held-out ISI AUC by tier, written IN-PROCESS to the DANT out-dir."""
    sjp, tc, load_session = _import_pipeline(subj)
    tracks = pd.read_csv(paths.out_dir / "curated_tracks.csv")
    reg = pd.read_csv(paths.registry_curation, dtype={"session": str})
    reg["uid"] = reg["dant_uid"].astype(int)
    # (uid, session) -> ks_unit_id, restricted to each track's kept sessions.
    kept_pairs: Dict[Tuple[int, str], int] = {}
    for _, row in tracks.iterrows():
        uid = int(row["curated_uid"])
        for s in [s for s in str(row["kept_sessions"]).split(";") if s]:
            m = reg[(reg["uid"] == uid) & (reg["session"] == s)]
            if len(m):
                kept_pairs[(uid, s)] = int(m.iloc[0]["ks_unit_id"])
    holdout = collect_holdout_isi(kept_pairs, subj, paths.pkl_dir)
    result = tc.held_out_isi_auc_by_tier(tracks, holdout)
    write_validation_json(result, paths.out_dir)
    print(f"[validate] held-out ISI AUC by tier: {result}", flush=True)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe" -m pytest tests/tracking_dant/test_curate_dant.py -v`
Expected: PASS (8 passed total).

- [ ] **Step 5: Commit**

```bash
git add scripts/tracking_dant/curate_dant.py tests/tracking_dant/test_curate_dant.py
git commit -m "feat(dant-curation): in-process held-out ISI AUC (clobber-safe)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Summary table + figure

**Files:**
- Modify: `scripts/tracking_dant/curate_dant.py` (append summary functions)
- Test: `tests/tracking_dant/test_curate_dant.py` (append)

**Interfaces:**
- Consumes: `UM_YARDSTICK` (Task 1); `DantCurationPaths` (Task 2); the `curated_tracks.csv` (`confidence_tier` column) + `curation_validation.json` produced earlier.
- Produces:
  - `build_summary_table(tier_counts: dict, auc_by_tier: dict, yardstick: dict = UM_YARDSTICK) -> pd.DataFrame` — one row per tier (`trusted, review, suspect`) with DANT counts/AUC and the UM yardstick columns. Pure.
  - `plot_summary(table: pd.DataFrame, out_png) -> None` — saves a 2-panel figure (tier-count bars + AUC bars vs UM yardstick line).
  - `step_summary(paths: DantCurationPaths) -> pd.DataFrame` — reads `curated_tracks.csv` + `curation_validation.json`, writes `dant_curation_summary.{csv,png}` to `paths.out_dir`, returns the table.

- [ ] **Step 1: Write the failing test**

Append to `tests/tracking_dant/test_curate_dant.py`:

```python
def test_build_summary_table_rows_and_yardstick():
    tier_counts = {"trusted": 40, "review": 300, "suspect": 80}
    auc_by_tier = {
        "trusted": {"auc": 0.81, "n_matched": 120, "n_nonmatched": 200},
        "review": {"auc": 0.70, "n_matched": 90, "n_nonmatched": 150},
    }
    df = curate_dant.build_summary_table(tier_counts, auc_by_tier)

    assert list(df["tier"]) == ["trusted", "review", "suspect"]
    trusted = df[df.tier == "trusted"].iloc[0]
    assert trusted["dant_n_tracks"] == 40
    assert trusted["dant_auc"] == 0.81
    assert trusted["um_n_tracks"] == 22           # yardstick wired in
    assert trusted["um_auc"] == 0.96
    # a tier with no AUC entry still produces a row (suspect)
    suspect = df[df.tier == "suspect"].iloc[0]
    assert suspect["dant_n_tracks"] == 80
    assert suspect["dant_n_matched"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe" -m pytest tests/tracking_dant/test_curate_dant.py::test_build_summary_table_rows_and_yardstick -v`
Expected: FAIL — `AttributeError: module 'curate_dant' has no attribute 'build_summary_table'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/tracking_dant/curate_dant.py`:

```python
def build_summary_table(tier_counts: Dict[str, int], auc_by_tier: Dict[str, dict],
                        yardstick: Dict[str, dict] = UM_YARDSTICK) -> pd.DataFrame:
    """One row per tier: DANT track count + held-out ISI AUC, with the UM yardstick."""
    rows = []
    for tier in ["trusted", "review", "suspect"]:
        a = auc_by_tier.get(tier, {})
        y = yardstick.get(tier, {})
        rows.append({
            "tier": tier,
            "dant_n_tracks": int(tier_counts.get(tier, 0)),
            "dant_auc": a.get("auc", float("nan")),
            "dant_n_matched": int(a.get("n_matched", 0)),
            "dant_n_nonmatched": int(a.get("n_nonmatched", 0)),
            "um_n_tracks": y.get("n", float("nan")),
            "um_auc": y.get("auc", float("nan")),
        })
    return pd.DataFrame(rows)


def plot_summary(table: pd.DataFrame, out_png) -> None:
    """2-panel summary: tier counts (DANT vs UM) + held-out ISI AUC vs UM yardstick."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tiers = list(table["tier"])
    x = np.arange(len(tiers))
    fig, (axc, axa) = plt.subplots(1, 2, figsize=(11, 4.2))

    axc.bar(x - 0.2, table["dant_n_tracks"], width=0.4, label="DANT", color="#3474ae")
    axc.bar(x + 0.2, table["um_n_tracks"], width=0.4, label="UnitMatch", color="#9e9e9e")
    axc.set_xticks(x); axc.set_xticklabels(tiers)
    axc.set_ylabel("tracks (span>=2)"); axc.set_title("Tier counts")
    axc.legend(frameon=False)

    axa.bar(x, table["dant_auc"], width=0.5, color="#6baed6", label="DANT")
    for xi, v in zip(x, table["um_auc"]):
        if np.isfinite(v):
            axa.hlines(v, xi - 0.25, xi + 0.25, color="#ef6548", lw=2,
                       label="UM yardstick" if xi == 0 else None)
    axa.axhline(0.5, color="k", lw=0.8, ls=":", label="chance")
    axa.set_xticks(x); axa.set_xticklabels(tiers)
    axa.set_ylim(0.4, 1.0); axa.set_ylabel("held-out ISI AUC")
    axa.set_title("Independent quality (held-out ISI)")
    axa.legend(frameon=False, fontsize=8)

    fig.suptitle("DANT BG_046 track curation vs UnitMatch yardstick")
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def step_summary(paths: DantCurationPaths) -> pd.DataFrame:
    tracks = pd.read_csv(paths.out_dir / "curated_tracks.csv")
    tier_counts = tracks["confidence_tier"].value_counts().to_dict()
    val_path = paths.out_dir / "curation_validation.json"
    auc_by_tier = json.loads(val_path.read_text()) if val_path.exists() else {}
    table = build_summary_table(tier_counts, auc_by_tier)
    table.to_csv(paths.out_dir / "dant_curation_summary.csv", index=False)
    plot_summary(table, paths.out_dir / "dant_curation_summary.png")
    print(f"[summary] tiers={tier_counts}", flush=True)
    print(table.to_string(index=False), flush=True)
    return table
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe" -m pytest tests/tracking_dant/test_curate_dant.py -v`
Expected: PASS (9 passed total).

- [ ] **Step 5: Commit**

```bash
git add scripts/tracking_dant/curate_dant.py tests/tracking_dant/test_curate_dant.py
git commit -m "feat(dant-curation): tier/AUC summary table + figure vs UM yardstick

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Step glue + `main()` + README

**Files:**
- Modify: `scripts/tracking_dant/curate_dant.py` (append step wrappers + `main`)
- Modify: `scripts/tracking_dant/README.md`
- Test: `tests/tracking_dant/test_curate_dant.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces:
  - `step_registry(paths) -> tuple[int, int]`, `step_curate(paths, rebuild_cache=True) -> None`, `step_render(paths, tier, max_uids=None, uids=None) -> None` (subprocess wrappers using `sys.executable`).
  - `parse_steps(s: str) -> list[str]` — split a comma list, validate against the known steps, preserve canonical order.
  - `main(argv=None) -> int` — argparse (`--steps`, `--primary`, `--review-max-uids`, `--trusted-max-uids`, `--no-rebuild-cache`), runs the selected steps in order.

- [ ] **Step 1: Write the failing test**

Append to `tests/tracking_dant/test_curate_dant.py`:

```python
import pytest


def test_parse_steps_default_order():
    assert curate_dant.parse_steps("registry,curate,validate,render,summary") == [
        "registry", "curate", "validate", "render", "summary"]


def test_parse_steps_subset_canonical_order():
    # given out of order, returns canonical order; whitespace tolerated
    assert curate_dant.parse_steps("summary, registry") == ["registry", "summary"]


def test_parse_steps_rejects_unknown():
    with pytest.raises(ValueError):
        curate_dant.parse_steps("registry,bogus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe" -m pytest tests/tracking_dant/test_curate_dant.py::test_parse_steps_default_order -v`
Expected: FAIL — `AttributeError: module 'curate_dant' has no attribute 'parse_steps'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/tracking_dant/curate_dant.py`:

```python
STEPS = ["registry", "curate", "validate", "render", "summary"]


def parse_steps(s: str) -> List[str]:
    """Comma list -> validated steps in canonical order."""
    want = {tok.strip() for tok in s.split(",") if tok.strip()}
    bad = want - set(STEPS)
    if bad:
        raise ValueError(f"unknown step(s): {sorted(bad)}; valid: {STEPS}")
    return [s for s in STEPS if s in want]


def step_registry(paths: DantCurationPaths) -> Tuple[int, int]:
    paths.states_empty.mkdir(parents=True, exist_ok=True)
    n_rows, n_uids = write_curation_registry(paths.registry_in, paths.registry_curation)
    print(f"[registry] kept {n_rows} rows / {n_uids} dant_uids (dant_uid>0) "
          f"-> {paths.registry_curation}", flush=True)
    return n_rows, n_uids


def step_curate(paths: DantCurationPaths, rebuild_cache: bool = True) -> None:
    paths.states_empty.mkdir(parents=True, exist_ok=True)
    cmd = build_curate_cmd(sys.executable, paths, rebuild_cache)
    print("[curate]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def step_render(paths: DantCurationPaths, tier: str,
                max_uids: Optional[int] = None,
                uids: Optional[List[int]] = None) -> None:
    cmd = build_render_cmd(sys.executable, paths, tier, max_uids=max_uids, uids=uids)
    print(f"[render:{tier}]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", default=",".join(STEPS),
                    help="comma list of steps to run; default all")
    ap.add_argument("--primary", type=Path, default=PRIMARY_DEFAULT,
                    help="PRIMARY repo root (raw waveforms + pkls live there)")
    ap.add_argument("--review-max-uids", type=int, default=25,
                    help="cap on review-tier sheets (spot-check sample)")
    ap.add_argument("--trusted-max-uids", type=int, default=None,
                    help="cap on trusted-tier sheets (None = render all)")
    ap.add_argument("--no-rebuild-cache", action="store_true",
                    help="reuse an existing feature cache instead of rebuilding")
    args = ap.parse_args(argv)
    steps = parse_steps(args.steps)
    paths = DantCurationPaths.default(WORKTREE_ROOT, args.primary)
    print(f"DANT curation runner — steps={steps}\n  out_dir={paths.out_dir}", flush=True)

    if "registry" in steps:
        step_registry(paths)
    if "curate" in steps:
        step_curate(paths, rebuild_cache=not args.no_rebuild_cache)
    if "validate" in steps:
        step_validate(paths)
    if "render" in steps:
        step_render(paths, "trusted", max_uids=args.trusted_max_uids)
        step_render(paths, "review", max_uids=args.review_max_uids)
    if "summary" in steps:
        step_summary(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe" -m pytest tests/tracking_dant/test_curate_dant.py -v`
Expected: PASS (12 passed total).

- [ ] **Step 5: Update the README**

Append this section to `scripts/tracking_dant/README.md`:

```markdown
## Curation + QC-sheet rendering (`curate_dant.py`)

Runs DANT's tracks through the project's existing curation + QC-sheet pipeline,
biophysical-only, into a DANT-specific output dir (the UnitMatch curation outputs
are never touched). Run from the worktree root with the analysis interpreter:

    <PRIMARY>/.venv/Scripts/python.exe scripts/tracking_dant/curate_dant.py

Steps (default: all, in order): `registry,curate,validate,render,summary`.
- `registry`  filter `dant_registry.csv` to `dant_uid > 0` -> `dant_registry_curation.csv`
- `curate`    drive `curate_tracks.py` (`--liberal-col dant_uid`, empty states dir
              -> corroborator off, `--drift-source none`) -> `curated_tracks.csv`
- `validate`  held-out ISI AUC by tier, computed IN-PROCESS (the `validate_curation.py`
              CLI hardcodes the UM dir) -> `curation_validation.json`
- `render`    `render_curation_sheets.py --no-pair-scores`: all trusted sheets +
              a capped review sample (`--review-max-uids`, default 25)
- `summary`   tier counts + AUC vs the UM yardstick -> `dant_curation_summary.{csv,png}`

Outputs land under `FIGURES/tracking_dant/BG_046/curation/` (+ `/sheets`).

Pilot a few sheets before the full render:

    ... curate_dant.py --steps render --trusted-max-uids 5

Re-render only (reuse the curate cache):

    ... curate_dant.py --steps render,summary
```

- [ ] **Step 6: Commit**

```bash
git add scripts/tracking_dant/curate_dant.py tests/tracking_dant/test_curate_dant.py scripts/tracking_dant/README.md
git commit -m "feat(dant-curation): step glue + main(--steps) runner + README

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Post-build: controller runs the real pipeline (not a subagent task)

After all five tasks pass review (mirrors how the DANT build itself was run — subagent-driven build, then controller executes on real data):

1. **Pilot** — `curate_dant.py --steps registry,curate` (rebuilds the feature cache over the ~41 DANT sessions), then `--steps render --trusted-max-uids 5`. Open one trusted sheet; confirm the cross-session waveform footprint, ISI panels, and page-2 task PSTHs populate (BG_046 pkls have real trials).
2. **Full run** — `curate_dant.py --steps validate,render,summary` (validate, render all trusted + 25 review, summary). If the cache is fresh from step 1, this reuses it.
3. **Report** — tier distribution, held-out ISI AUC by tier vs the UM yardstick (22 trusted / 567 review / 160 suspect, trusted AUC ≈ 0.96), and where the sheets landed. Report honestly if trusted is small or any sessions were skipped (memory note: `--rebuild-cache` is mandatory for the DANT run since the `(uid, session)` cache key's uid meaning differs from any UM cache).

---

## Self-Review

**1. Spec coverage:**
- Curation-ready registry (drop `dant_uid<=0`) → Task 1. ✓
- Biophysical-only curate into DANT out-dir + cache → Tasks 2 (cmd) + 5 (run). ✓
- Held-out ISI AUC per tier, in-process, clobber-safe → Task 3. ✓
- Render trusted in full + capped review sample → Tasks 2 (cmd) + 5 (run). ✓
- Summary referencing UM yardstick → Task 4. ✓
- `--liberal-col dant_uid` everywhere, `--no-pair-scores`, `--drift-source none`, empty states dir, PRIMARY raw-wf/pkls, no shared-pipeline edits → encoded in the builders (Task 2) + Global Constraints. ✓
- Do NOT use `build_qc_sheets.py`/`validate_long_tracks.py` → Global Constraints; we drive `render_curation_sheets.py`. ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code; every command shows the exact interpreter + expected output. ✓

**3. Type consistency:** `DantCurationPaths` field names (`registry_curation`, `states_empty`, `out_dir`, `cache_path`, `sheets_dir`, `curate_script`, `render_script`) are used identically in the builders (Task 2), validation (Task 3), summary (Task 4), and glue (Task 5). `build_curate_cmd`/`build_render_cmd`/`write_validation_json`/`build_summary_table`/`parse_steps` signatures match their call sites. `step_validate` returns the dict that `held_out_isi_auc_by_tier` produces, which `build_summary_table` consumes via the JSON. ✓
