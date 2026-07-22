# Camera Baseline Sync — Backbone (Plan 1 of Sub-project A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the non-GUI sync backbone — validated multi-anchor clock fitting, multi-subject namespacing, robust session-id/camera-dir handling, ceph-safe metadata reconstruction + video staging, and subject-aware status — so the existing manual anchors (and, after Plan 2, mixed baseline+change anchors) produce a cross-validated video↔neural clock for any subject with zero writes to X:.

**Architecture:** Extend the mature `src/visdetect/core/video_sync.py` in place (follow its existing patterns). Add a new sparse-anchor fit path (`fit_multianchor_clock` + leave-one-out CV + a `manual_multianchor` quality tier) alongside the existing `fit_2anchor_clock`; generalize the anchor schema to carry event type (baseline vs change); namespace all caches by subject; redirect header-only metadata reconstruction to local storage; add a read-only-X: staging helper. Callers (`fit_sync.py`, `sync_status.py`, `reconstruct_camera_metadata.py`) pass a subject-namespaced `sync_dir` rather than changing library defaults.

**Tech Stack:** Python 3.10 (venv `.venv`, invoke via `py`), numpy, scipy (`theilslopes`), pandas, pytest. Windows + git-bash.

## Global Constraints

- **Zero writes to X: (ceph).** `CAMERA_ROOT` (`X:/public/.../Cameras_sortIntoSubjects`) is READ-ONLY. Every artifact writes under the local repo (`data/cache/...`, `figures/...`). Verified by a guard test (Task 6).
- **No heavy compute over X:.** Whole-video decode happens on locally-staged copies only (Task 7). The backbone itself reads only metadata CSVs + (in Plan 2) frames.
- **Canonical session ids.** Any session id used as a key/join/path MUST go through `config.session_date_key()` / a `canonical_camera_session()` helper — never raw `str(int(x)).zfill(8)`. Handles int64, 6-digit `DDMMYY` (BG_031/039), 7-digit, subject-prefixed, and `_b`/`_v2`/`_laser` suffix tokens.
- **Subject-namespaced caches.** `data/cache/video_sync/<SUBJECT>/…`, `data/cache/video_labels/<SUBJECT>/…`. Never write bare-date-keyed files into a flat dir (cross-subject date collisions).
- **Never silent-overwrite.** `fit_sync` archives the prior `*_video_sync.json` / `*_anchor.json` before writing (Task 5, migration policy §3.14 of the spec).
- **Subagent model:** every subagent = Opus 4.8.
- **Clock orientation:** the new fit and all downstream feature code use `camera_to_nidaq`'s convention `nidaq_s = slope * cam_s + offset` (x = camera seconds, y = NI-DAQ seconds). The legacy `fit_2anchor_clock` stores the *inverse* (`video = slope * nidaq + offset`) — do not mix.

**Spec:** `docs/superpowers/specs/2026-07-21-camera-baseline-sync-multisubject-design.md` (sub-project A). Plan 2 (tagger GUI + label capture) is a separate document.

---

## File Structure

| File | Change | Responsibility |
|------|--------|----------------|
| `src/visdetect/analysis/config.py` | modify | Add `VIDEO_LABELS_DIR`, `VIDEO_STAGING_DIR`, `subject_video_sync_dir()`, `subject_video_labels_dir()`, `canonical_camera_session()`, `camera_dir_token()` |
| `src/visdetect/analysis/constants.py` | modify | Add `VIDEO_SYNC_MANUAL_GOOD_CV_MS`, `VIDEO_SYNC_MANUAL_REVIEW_CV_MS`, `VIDEO_SYNC_MANUAL_MIN_ANCHORS` |
| `src/visdetect/core/video_sync.py` | modify | `find_camera_files`/`camera_dir_to_session` (subject+suffix+canonical); generalized anchor schema (v3 event_type); `fit_multianchor_clock` + `_loo_cv`; `manual_multianchor` quality tier; `local_reconstructed_metadata_path` + reconstruction redirect; `stage_session_video`/`unstage_session_video`; `archive_sync_artifacts` |
| `scripts/video/fit_sync.py` | modify | Route ≥3 anchors → `fit_multianchor_clock`; archive-before-write; subject-namespaced `sync_dir` |
| `scripts/video/reconstruct_camera_metadata.py` | modify | Write reconstructed CSV + provenance to LOCAL (`subject_video_sync_dir`), never X: |
| `scripts/video/sync_status.py` | modify | `--subject`; report `cv_rmse_ms` + label coverage; namespaced `sync_dir` |
| `tests/video/test_camera_session_ids.py` | create | canonical_camera_session / camera_dir_token / find_camera_files |
| `tests/video/test_anchor_schema_v3.py` | create | event_type schema + migration + merge |
| `tests/video/test_multianchor_fit.py` | create | fit_multianchor_clock + _loo_cv + quality tier |
| `tests/video/test_sync_archive_and_reconstruct.py` | create | archive-before-write + local reconstruction + read-only-X: guard |
| `tests/video/test_stage_video.py` | create | staging copy + read-only source + idempotency |

---

## Task 1: Config — namespaced dirs + canonical camera-session helpers

**Files:**
- Modify: `src/visdetect/analysis/config.py` (after line 134, the `PUPIL_DIR` block; helpers after `session_int_to_iso` ~line 437)
- Test: `tests/video/test_camera_session_ids.py`

**Interfaces:**
- Consumes: `ROOT`, `SUBJECT`, `VIDEO_SYNC_DIR` (existing, config.py:78/84/129); `session_date_key` (existing, config.py:409, returns `(year, month, day)`).
- Produces:
  - `VIDEO_LABELS_DIR: str`, `VIDEO_STAGING_DIR: str`
  - `subject_video_sync_dir(subject: Optional[str] = None) -> str`
  - `subject_video_labels_dir(subject: Optional[str] = None) -> str`
  - `canonical_camera_session(session) -> str` → 8-digit `DDMMYYYY`
  - `camera_dir_token(session) -> str` → 6-digit `DDMMYY`

- [ ] **Step 1: Write the failing test**

```python
# tests/video/test_camera_session_ids.py
import os
import pytest
from visdetect.analysis import config


@pytest.mark.parametrize("raw,expected", [
    ("05032025", "05032025"),   # already 8-digit DDMMYYYY
    ("050325", "05032025"),     # 6-digit DDMMYY (BG_031/039 early sessions)
    (5032025, "05032025"),      # int, leading-zero day dropped -> 7 digits
    ("BG_031_050325", "05032025"),   # subject-prefixed
    ("BG_039_01042025_b", "01042025"),  # subject-prefixed + re-record suffix
])
def test_canonical_camera_session(raw, expected):
    assert config.canonical_camera_session(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("05032025", "050325"),
    ("050325", "050325"),
    ("BG_039_01042025_b", "010425"),
])
def test_camera_dir_token(raw, expected):
    assert config.camera_dir_token(raw) == expected


def test_subject_dirs_are_namespaced(monkeypatch):
    d046 = config.subject_video_sync_dir("BG_046")
    d031 = config.subject_video_sync_dir("BG_031")
    assert d046.endswith(os.path.join("video_sync", "BG_046"))
    assert d031.endswith(os.path.join("video_sync", "BG_031"))
    assert d046 != d031
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/video/test_camera_session_ids.py -v`
Expected: FAIL — `AttributeError: module 'visdetect.analysis.config' has no attribute 'canonical_camera_session'`

- [ ] **Step 3: Add the config code**

Insert after the `PUPIL_DIR` line (config.py:134):

```python
VIDEO_LABELS_DIR       = os.path.join(ROOT, "data", "cache", "video_labels")
# Local scratch for staged camera videos (gitignored). NEVER on X:.
VIDEO_STAGING_DIR      = os.getenv(
    "VISDETECT_VIDEO_STAGING", os.path.join(ROOT, "data", "_staging", "video"))


def subject_video_sync_dir(subject: Optional[str] = None) -> str:
    """Per-subject sync cache dir: data/cache/video_sync/<SUBJECT>/."""
    return os.path.join(VIDEO_SYNC_DIR, subject or SUBJECT)


def subject_video_labels_dir(subject: Optional[str] = None) -> str:
    """Per-subject label sidecar dir: data/cache/video_labels/<SUBJECT>/."""
    return os.path.join(VIDEO_LABELS_DIR, subject or SUBJECT)
```

Insert after `session_int_to_iso` (config.py:437):

```python
def canonical_camera_session(session) -> str:
    """8-digit ``DDMMYYYY`` for ANY session token (camera-path safe).

    Unlike :func:`canonical_session_id` (which zero-pads and would turn the
    6-digit ``DDMMYY`` ids BG_031/039 carry into ``00050325``), this routes
    through :func:`session_date_key`, which parses 6-/7-/8-digit, subject-
    prefixed (``BG_031_050325``) and re-record-suffixed (``..._b``) forms.
    """
    y, m, d = session_date_key(session)
    return f"{d:02d}{m:02d}{y:04d}"


def camera_dir_token(session) -> str:
    """6-digit ``DDMMYY`` token used in camera directory names."""
    y, m, d = session_date_key(session)
    return f"{d:02d}{m:02d}{y % 100:02d}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/video/test_camera_session_ids.py -v`
Expected: PASS (all `test_canonical_camera_session`, `test_camera_dir_token`, `test_subject_dirs_are_namespaced`)

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/config.py tests/video/test_camera_session_ids.py
git commit -m "feat(video-sync): subject-namespaced cache dirs + canonical camera-session helpers"
```

---

## Task 2: Subject/suffix-tolerant camera-file discovery

**Files:**
- Modify: `src/visdetect/core/video_sync.py:778-825` (`camera_dir_to_session`, `find_camera_files`)
- Test: `tests/video/test_camera_session_ids.py` (extend)

**Interfaces:**
- Consumes: `config.canonical_camera_session`, `config.camera_dir_token`, `config.SUBJECT`, `CAMERA_ROOT` (existing import).
- Produces: `find_camera_files(session, camera_root=None, subject=None)` returning `{"eye_cam": {"video","metadata"}, "front_cam": {...}}`, subject defaulting to `config.SUBJECT`; `camera_dir_to_session(dirname)` tolerant of suffix tokens.

- [ ] **Step 1: Write the failing test**

```python
# tests/video/test_camera_session_ids.py  (append)
from visdetect.core import video_sync


def _make_cam_dir(tmp_path, subject, token, cams=("Eye_cam", "Front_cam")):
    d = tmp_path / f"{subject}_{token}"
    d.mkdir()
    for c in cams:
        (d / f"{subject}_{token}_{c}.mp4").write_bytes(b"x")
        (d / f"{subject}_{token}_{c}_metadata.csv").write_text("Timestamp (ms)\n")
    return d


def test_find_camera_files_6digit_subject(tmp_path):
    _make_cam_dir(tmp_path, "BG_031", "050325")
    files = video_sync.find_camera_files(
        "050325", camera_root=str(tmp_path), subject="BG_031")
    assert "eye_cam" in files and "front_cam" in files
    assert files["eye_cam"]["video"].endswith("Eye_cam.mp4")


def test_camera_dir_to_session_tolerates_suffix():
    assert video_sync.camera_dir_to_session("BG_039_010425_b") == "01042025"
    assert video_sync.camera_dir_to_session("BG_046_010725") == "01072025"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/video/test_camera_session_ids.py -k "camera_files or tolerates_suffix" -v`
Expected: FAIL — `find_camera_files` mis-builds the dir (`00050325` path) / `camera_dir_to_session` raises `ValueError` on the `_b` suffix.

- [ ] **Step 3: Rewrite the two functions**

Replace `camera_dir_to_session` (video_sync.py:778-785):

```python
def camera_dir_to_session(dirname: str, subject: str = None) -> str:
    """Convert a camera directory name (``BG_046_DDMMYY``, possibly with a
    re-record suffix like ``BG_039_010425_b``) to session ``DDMMYYYY``."""
    from visdetect.analysis.config import canonical_camera_session
    return canonical_camera_session(dirname)
```

Replace the path-building head of `find_camera_files` (video_sync.py:788-808) — keep the glob loop (lines 813-825) unchanged:

```python
def find_camera_files(
    session_name: str,
    camera_root: Optional[str] = None,
    subject: str = None,
) -> Dict[str, Dict[str, str]]:
    """Locate video + metadata files for a session.

    Returns dict like::

        {"eye_cam": {"video": "path.mp4", "metadata": "path.csv"},
         "front_cam": {"video": "path.mp4", "metadata": "path.csv"}}

    Keys are present only if both video and metadata files are found.
    """
    from visdetect.analysis.config import SUBJECT, camera_dir_token
    root = camera_root or CAMERA_ROOT
    subject = subject or SUBJECT
    token = camera_dir_token(session_name)
    cam_dir = os.path.join(root, f"{subject}_{token}")

    if not os.path.isdir(cam_dir):
        raise FileNotFoundError(f"Camera directory not found: {cam_dir}")

    result = {}
    for cam_label, prefix in [("eye_cam", "Eye_cam"), ("front_cam", "Front_cam")]:
        video = None
        meta = None
        for f in os.listdir(cam_dir):
            if prefix in f and f.endswith(".mp4"):
                video = os.path.join(cam_dir, f)
            elif prefix in f and f.endswith("_metadata.csv"):
                meta = os.path.join(cam_dir, f)
        if video and meta:
            result[cam_label] = {"video": video, "metadata": meta}

    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/video/test_camera_session_ids.py -v`
Expected: PASS (all tests, incl. the two new ones)

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/core/video_sync.py tests/video/test_camera_session_ids.py
git commit -m "feat(video-sync): subject- and suffix-tolerant camera-file discovery via canonical session ids"
```

---

## Task 3: Generalized anchor schema (v3: baseline + change events)

**Files:**
- Modify: `src/visdetect/core/video_sync.py` (`_migrate_anchor_v1_to_v2` → add v3 step at ~2761; add `_build_change_anchor_entry`, generalize `_merge_anchor_into_file` at ~2826)
- Test: `tests/video/test_anchor_schema_v3.py`

**Interfaces:**
- Consumes: existing `_build_anchor_entry` (baseline), `save_anchor`/`load_anchor`.
- Produces:
  - `_migrate_anchor_to_v3(d: dict) -> dict` — adds `event_type="baseline_on"` + `nidaq_event_s` to legacy entries, sets `schema_version=3`.
  - `_build_change_anchor_entry(change_on_s, ts_ms, trial_index, frame_idx, change_size, outcome) -> dict`
  - `_merge_anchor_into_file(base, new_entry)` keyed on `(trial_index, event_type)`.
  - Canonical per-anchor field: `nidaq_event_s` (baseline anchors also keep `nidaq_baseline_on_s` for legacy `fit_2anchor_clock`).

- [ ] **Step 1: Write the failing test**

```python
# tests/video/test_anchor_schema_v3.py
import numpy as np
from visdetect.core import video_sync as vs


def test_migrate_v2_to_v3_adds_event_type():
    v2 = {
        "session": "01072025", "schema_version": 2,
        "frame_rate_fps": 50.0, "n_trials": 3,
        "anchors": [{
            "trial_index": 0, "nidaq_baseline_on_s": 12.5,
            "video_frame_idx": 600, "video_time_s": 12.0,
            "clicked_at": "2026-07-21T10:00:00",
        }],
    }
    out = vs._migrate_anchor_to_v3(v2)
    assert out["schema_version"] == 3
    a = out["anchors"][0]
    assert a["event_type"] == "baseline_on"
    assert a["nidaq_event_s"] == 12.5


def test_build_change_anchor_entry():
    ts_ms = np.arange(1000, dtype=float) * 20.0  # 50 fps
    e = vs._build_change_anchor_entry(
        change_on_s=30.0, ts_ms=ts_ms, trial_index=5,
        frame_idx=100, change_size=4.0, outcome="hit")
    assert e["event_type"] == "change_on"
    assert e["nidaq_event_s"] == 30.0
    assert e["change_size"] == 4.0
    assert e["outcome"] == "hit"
    assert e["video_time_s"] == 2.0  # ts_ms[100]/1000


def test_merge_keys_on_trial_and_event_type():
    base = {"anchors": [
        {"trial_index": 5, "event_type": "baseline_on", "nidaq_event_s": 1.0},
    ]}
    change = {"trial_index": 5, "event_type": "change_on", "nidaq_event_s": 3.0}
    out = vs._merge_anchor_into_file(base, change)
    # same trial, different event type -> BOTH kept (2 anchors), not replaced
    assert len(out["anchors"]) == 2
    # replacing the baseline on the same trial -> still 2
    repl = {"trial_index": 5, "event_type": "baseline_on", "nidaq_event_s": 1.5}
    out2 = vs._merge_anchor_into_file(out, repl)
    assert len(out2["anchors"]) == 2
    b = [a for a in out2["anchors"] if a["event_type"] == "baseline_on"][0]
    assert b["nidaq_event_s"] == 1.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/video/test_anchor_schema_v3.py -v`
Expected: FAIL — `_migrate_anchor_to_v3` / `_build_change_anchor_entry` undefined; merge collapses the two same-trial anchors to 1.

- [ ] **Step 3: Add / modify the schema code**

Add after `_migrate_anchor_v1_to_v2` (video_sync.py:2781):

```python
def _migrate_anchor_to_v3(d: dict) -> dict:
    """Add event_type/nidaq_event_s to v2 baseline-only anchors (idempotent)."""
    d = _migrate_anchor_v1_to_v2(d)
    if d.get("schema_version") == 3:
        return d
    for a in d["anchors"]:
        a.setdefault("event_type", "baseline_on")
        a.setdefault("nidaq_event_s", float(a.get("nidaq_baseline_on_s")))
    d = dict(d)
    d["schema_version"] = 3
    return d


def _build_change_anchor_entry(
    change_on_s: float,
    ts_ms: np.ndarray,
    trial_index: int,
    frame_idx: int,
    change_size: float,
    outcome: str,
) -> dict:
    """Build a v3 change-onset anchor entry from a clicked frame index."""
    fi = int(frame_idx)
    return {
        "trial_index": int(trial_index),
        "event_type": "change_on",
        "nidaq_event_s": float(change_on_s),
        "change_size": float(change_size),
        "outcome": str(outcome),
        "video_frame_idx": fi,
        "video_time_s": float(ts_ms[fi] / 1000.0),
        "clicked_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
```

Replace `_merge_anchor_into_file` (video_sync.py:2826-2838) to key on `(trial_index, event_type)`:

```python
def _merge_anchor_into_file(base: dict, new_entry: dict) -> dict:
    """Return a copy of *base* with *new_entry* merged into its anchors list.

    Replaces an existing anchor with the same ``(trial_index, event_type)``
    (default event_type ``baseline_on`` for legacy entries). Sorted by
    ``(trial_index, event_type)``.
    """
    def key(a):
        return (int(a["trial_index"]), a.get("event_type", "baseline_on"))
    nk = key(new_entry)
    kept = [a for a in base["anchors"] if key(a) != nk]
    kept.append(new_entry)
    kept.sort(key=key)
    out = dict(base)
    out["anchors"] = kept
    return out
```

Make `load_anchor` return v3 — change its last line (video_sync.py:2874) from `return _migrate_anchor_v1_to_v2(raw)` to:

```python
    return _migrate_anchor_to_v3(raw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/video/test_anchor_schema_v3.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/core/video_sync.py tests/video/test_anchor_schema_v3.py
git commit -m "feat(video-sync): v3 anchor schema with event_type (baseline + change anchors)"
```

---

## Task 4: `fit_multianchor_clock` + leave-one-out CV + quality tier

**Files:**
- Modify: `src/visdetect/analysis/constants.py` (after line 169); `src/visdetect/core/video_sync.py` (add `_loo_cv`, `fit_multianchor_clock` near `fit_2anchor_clock` ~672; add `manual_multianchor` branch to `SyncResult.quality` ~537)
- Test: `tests/video/test_multianchor_fit.py`

**Interfaces:**
- Consumes: `SyncResult`, `theilslopes` (scipy), `_GOOD_RMSE_MS`/`_REVIEW_RMSE_MS` (video_sync.py:108/110), `VIDEO_SYNC_MAX_DRIFT_PPM`.
- Produces:
  - `_loo_cv(cam_s: np.ndarray, nidaq_s: np.ndarray) -> float` (RMS ms; requires n≥3).
  - `fit_multianchor_clock(anchors: List[dict], n_baseline_on: int, outlier_sigma: float = VIDEO_SYNC_OUTLIER_SIGMA) -> SyncResult` with `detection_method="manual_multianchor"`, orientation `nidaq = slope*cam_s + offset`.
  - Constants `VIDEO_SYNC_MANUAL_GOOD_CV_MS=20.0`, `VIDEO_SYNC_MANUAL_REVIEW_CV_MS=40.0`, `VIDEO_SYNC_MANUAL_MIN_ANCHORS=3`.

- [ ] **Step 1: Write the failing test**

```python
# tests/video/test_multianchor_fit.py
import numpy as np
import pytest
from visdetect.core import video_sync as vs


def _anchors(nidaq, video, event="baseline_on"):
    out = []
    for i, (n, v) in enumerate(zip(nidaq, video)):
        out.append({"trial_index": i, "event_type": event,
                    "nidaq_event_s": float(n), "video_time_s": float(v)})
    return out


def test_loo_cv_small_n_returns_finite():
    x = np.array([0., 1., 2., 3., 4.])
    y = 1.0 * x + 0.5
    assert vs._loo_cv(x, y) < 1.0  # near-perfect line -> tiny cv (ms)


def test_fit_multianchor_recovers_slope_offset():
    nidaq = np.linspace(10, 600, 8)
    slope_true, off_true = 1.00002, -3.2
    video = slope_true * nidaq + off_true          # cam(video) = fn(nidaq)... invert below
    # anchors store video_time_s and nidaq_event_s; fit models nidaq = slope*cam + off
    anchors = _anchors(nidaq, video)
    res = vs.fit_multianchor_clock(anchors, n_baseline_on=8)
    # cam_s = video, nidaq = nidaq -> slope ~ 1/slope_true, offset ~ -off/slope_true
    pred = res.slope * np.asarray(video) + res.offset
    assert np.allclose(pred, nidaq, atol=1e-3)
    assert res.detection_method == "manual_multianchor"
    assert res.cv_rmse_ms < 5.0
    assert res.quality == "good"


def test_fit_multianchor_mad_rejects_outlier():
    nidaq = np.linspace(10, 600, 8)
    video = 1.0 * nidaq + 0.0
    video[3] += 0.5  # 500 ms bad anchor
    anchors = _anchors(nidaq, video)
    res = vs.fit_multianchor_clock(anchors, n_baseline_on=8)
    assert res.n_anchors == 7  # one rejected


def test_quality_manual_multianchor_review_and_failed():
    good = vs.SyncResult(slope=1.0, offset=0.0, n_anchors=6, n_baseline_on=6,
                         rmse_ms=10, max_residual_ms=15, cv_rmse_ms=15,
                         slope_ppm=5, durbin_watson=2.0,
                         detection_method="manual_multianchor")
    review = vs.SyncResult(slope=1.0, offset=0.0, n_anchors=6, n_baseline_on=6,
                           rmse_ms=30, max_residual_ms=35, cv_rmse_ms=30,
                           slope_ppm=5, durbin_watson=2.0,
                           detection_method="manual_multianchor")
    failed = vs.SyncResult(slope=1.0, offset=0.0, n_anchors=6, n_baseline_on=6,
                           rmse_ms=80, max_residual_ms=90, cv_rmse_ms=80,
                           slope_ppm=5, durbin_watson=2.0,
                           detection_method="manual_multianchor")
    assert good.quality == "good"
    assert review.quality == "review"
    assert failed.quality == "failed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/video/test_multianchor_fit.py -v`
Expected: FAIL — `_loo_cv` / `fit_multianchor_clock` undefined; quality has no `manual_multianchor` branch.

- [ ] **Step 3a: Add constants**

Append to `src/visdetect/analysis/constants.py` after line 169:

```python
# Sparse manual multi-anchor sync (video_sync.fit_multianchor_clock)
VIDEO_SYNC_MANUAL_GOOD_CV_MS: float = 20.0    # cv_rmse below this = "good"
VIDEO_SYNC_MANUAL_REVIEW_CV_MS: float = 40.0  # cv_rmse below this = "review"
VIDEO_SYNC_MANUAL_MIN_ANCHORS: int = 3        # need >=3 for a LOO-validated fit
```

- [ ] **Step 3b: Add the quality-tier branch**

In `SyncResult.quality` (video_sync.py:537), insert immediately after the `manual_slope_fit` branch (after line 543):

```python
        if self.detection_method == "manual_multianchor":
            from visdetect.analysis.constants import (
                VIDEO_SYNC_MANUAL_GOOD_CV_MS, VIDEO_SYNC_MANUAL_REVIEW_CV_MS,
                VIDEO_SYNC_MANUAL_MIN_ANCHORS, VIDEO_SYNC_MAX_DRIFT_PPM,
            )
            if self.slope <= 0 or self.n_anchors < VIDEO_SYNC_MANUAL_MIN_ANCHORS:
                return "failed"
            low_drift = abs(self.slope_ppm) < VIDEO_SYNC_MAX_DRIFT_PPM
            if self.cv_rmse_ms < VIDEO_SYNC_MANUAL_GOOD_CV_MS and low_drift:
                return "good"
            if self.cv_rmse_ms < VIDEO_SYNC_MANUAL_REVIEW_CV_MS and low_drift:
                return "review"
            return "failed"
```

- [ ] **Step 3c: Add `_loo_cv` and `fit_multianchor_clock`**

Add after `fit_2anchor_clock` (video_sync.py:672):

```python
def _loo_cv(cam_s: np.ndarray, nidaq_s: np.ndarray) -> float:
    """Leave-one-out CV RMSE (ms) for the linear clock. Requires n >= 3.

    For sparse manual anchors the dense 5-fold ``_temporal_cv`` leaves ~1
    anchor/fold (and returns its 999 sentinel below 20 anchors), so we use
    LOO: fit on all-but-one, predict the held-out anchor, RMS the errors.
    """
    n = len(cam_s)
    if n < 3:
        return float("nan")
    errs = []
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        A = np.column_stack([cam_s[m], np.ones(m.sum())])
        params, _, _, _ = np.linalg.lstsq(A, nidaq_s[m], rcond=None)
        pred = params[0] * cam_s[i] + params[1]
        errs.append(((nidaq_s[i] - pred) * 1000.0) ** 2)
    return float(np.sqrt(np.mean(errs)))


def fit_multianchor_clock(
    anchors: List[dict],
    n_baseline_on: int,
    outlier_sigma: float = VIDEO_SYNC_OUTLIER_SIGMA,
) -> SyncResult:
    """Fit a validated linear clock from >=3 manual anchors (any event type).

    Orientation matches ``camera_to_nidaq``: ``nidaq_s = slope*cam_s + offset``
    where ``cam_s = anchor['video_time_s']`` and ``nidaq_s`` is the anchor's
    ``nidaq_event_s`` (falling back to ``nidaq_baseline_on_s`` for legacy).
    Theil-Sen fit -> MAD outlier rejection -> LOO CV. detection_method =
    "manual_multianchor". Raises ValueError on <3 anchors or non-positive slope.
    """
    from scipy.stats import theilslopes
    if len(anchors) < 3:
        raise ValueError(
            f"fit_multianchor_clock needs >=3 anchors; got {len(anchors)}")

    cam_s = np.array([float(a["video_time_s"]) for a in anchors], dtype=np.float64)
    nidaq_s = np.array(
        [float(a.get("nidaq_event_s", a.get("nidaq_baseline_on_s"))) for a in anchors],
        dtype=np.float64)
    order = np.argsort(cam_s)
    cam_s, nidaq_s = cam_s[order], nidaq_s[order]

    slope, intercept, _, _ = theilslopes(nidaq_s, cam_s)
    resid_ms = (nidaq_s - (slope * cam_s + intercept)) * 1000.0
    mad = np.median(np.abs(resid_ms - np.median(resid_ms))) or 1.0
    keep = np.abs(resid_ms - np.median(resid_ms)) <= outlier_sigma * 1.4826 * mad
    if keep.sum() >= 3 and keep.sum() < len(keep):
        cam_s, nidaq_s = cam_s[keep], nidaq_s[keep]
        slope, intercept, _, _ = theilslopes(nidaq_s, cam_s)
        resid_ms = (nidaq_s - (slope * cam_s + intercept)) * 1000.0

    if slope <= 0:
        raise ValueError(f"Computed slope {slope} is non-positive; check anchors.")

    return SyncResult(
        slope=float(slope),
        offset=float(intercept),
        n_anchors=int(len(cam_s)),
        n_baseline_on=int(n_baseline_on),
        rmse_ms=float(np.sqrt(np.mean(resid_ms ** 2))),
        max_residual_ms=float(np.max(np.abs(resid_ms))),
        cv_rmse_ms=_loo_cv(cam_s, nidaq_s),
        slope_ppm=float((slope - 1.0) * 1e6),
        durbin_watson=2.0,
        detection_method="manual_multianchor",
        residuals_ms=resid_ms,
        matched_cam_ms=cam_s * 1000.0,
        matched_nidaq_s=nidaq_s,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/video/test_multianchor_fit.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/constants.py src/visdetect/core/video_sync.py tests/video/test_multianchor_fit.py
git commit -m "feat(video-sync): fit_multianchor_clock with LOO-CV and manual_multianchor quality tier"
```

---

## Task 5: fit_sync routing + archive-before-write

**Files:**
- Modify: `src/visdetect/core/video_sync.py` (add `archive_sync_artifacts`); `scripts/video/fit_sync.py:56-136`
- Test: `tests/video/test_sync_archive_and_reconstruct.py`

**Interfaces:**
- Consumes: `fit_multianchor_clock`, `fit_2anchor_clock`, `save_video_sync`, `subject_video_sync_dir`.
- Produces: `archive_sync_artifacts(session, subject=None, sync_dir=None, when=None) -> Optional[str]` — moves existing `*_video_sync.json` + `*_anchor.json` to `<sync_dir>/_archive/<when>/`, returns the archive dir or None.

- [ ] **Step 1: Write the failing test**

```python
# tests/video/test_sync_archive_and_reconstruct.py
import json, os
from visdetect.core import video_sync as vs


def test_archive_moves_existing_sync_and_anchor(tmp_path):
    d = tmp_path
    (d / "01072025_video_sync.json").write_text(json.dumps({"session_name": "01072025"}))
    (d / "01072025_anchor.json").write_text(json.dumps({"anchors": []}))
    arch = vs.archive_sync_artifacts("01072025", sync_dir=str(d), when="2026-07-21")
    assert arch is not None
    assert not (d / "01072025_video_sync.json").exists()
    assert os.path.exists(os.path.join(arch, "01072025_video_sync.json"))
    assert os.path.exists(os.path.join(arch, "01072025_anchor.json"))


def test_archive_noop_when_nothing_exists(tmp_path):
    assert vs.archive_sync_artifacts("01072025", sync_dir=str(tmp_path)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/video/test_sync_archive_and_reconstruct.py -k archive -v`
Expected: FAIL — `archive_sync_artifacts` undefined.

- [ ] **Step 3a: Add `archive_sync_artifacts`**

Add near `save_video_sync` in video_sync.py:

```python
def archive_sync_artifacts(
    session_name: str,
    subject: Optional[str] = None,
    sync_dir: Optional[str] = None,
    when: Optional[str] = None,
) -> Optional[str]:
    """Move existing sync + anchor JSONs into ``<sync_dir>/_archive/<when>/``.

    Called before a re-fit so a re-tag never silently clobbers a prior fit
    (spec migration policy). Returns the archive dir, or None if nothing moved.
    """
    import shutil
    from visdetect.analysis.config import subject_video_sync_dir, canonical_camera_session
    out_dir = sync_dir or subject_video_sync_dir(subject)
    sn = canonical_camera_session(session_name)
    when = when or _dt.date.today().isoformat()
    moved = False
    arch = os.path.join(out_dir, "_archive", when)
    for suffix in ("_video_sync.json", "_anchor.json"):
        src = os.path.join(out_dir, f"{sn}{suffix}")
        if os.path.exists(src):
            os.makedirs(arch, exist_ok=True)
            shutil.move(src, os.path.join(arch, f"{sn}{suffix}"))
            moved = True
    return arch if moved else None
```

- [ ] **Step 4a: Run archive test — passes**

Run: `py -m pytest tests/video/test_sync_archive_and_reconstruct.py -k archive -v`
Expected: PASS

- [ ] **Step 3b: Rewire fit_sync.py**

Add `--subject` and route by anchor count. Replace the import block (fit_sync.py:29-36) to add the new names:

```python
from visdetect.core.video_sync import (
    find_camera_files,
    load_camera_metadata,
    load_anchor,
    load_video_sync,
    save_video_sync,
    fit_2anchor_clock,
    fit_multianchor_clock,
    archive_sync_artifacts,
)
from visdetect.analysis.config import subject_video_sync_dir, canonical_camera_session
```

Replace the arg parse + fit + save region (fit_sync.py:56-129). Add `--subject`; use `canonical_camera_session`; route on anchor count; archive before save; pass the namespaced `sync_dir`:

```python
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2: fit linear clock model from manual anchors.",
    )
    parser.add_argument("--session", required=True, help="Session name (e.g. 09092025).")
    parser.add_argument("--subject", default=None, help="Subject (default: config.SUBJECT).")
    args = parser.parse_args()

    session_name = canonical_camera_session(args.session)
    sync_dir = subject_video_sync_dir(args.subject)

    anchor_file = load_anchor(session_name, sync_dir=sync_dir)
    if anchor_file is None:
        logger.error("No anchor JSON for %s in %s. Run click_anchor first.",
                     session_name, sync_dir)
        return 2
    anchors = anchor_file["anchors"]
    if len(anchors) < 2:
        logger.error("Anchor JSON has %d anchor(s); need >=2.", len(anchors))
        return 2

    sess = load_session(session_name)
    baseline_on = np.asarray(sess.ni_events.get("Baseline_ON", []), dtype=float)
    baseline_on = baseline_on[baseline_on > 0]
    n_task_trials = len(sess.trials)
    if n_task_trials > 0 and len(baseline_on) > n_task_trials:
        baseline_on = baseline_on[:n_task_trials]
    n_baseline_on = int(len(baseline_on))
    del sess
    gc.collect()
    fps = float(anchor_file["frame_rate_fps"])

    try:
        if len(anchors) >= 3:
            sync_result = fit_multianchor_clock(anchors, n_baseline_on=n_baseline_on)
        else:
            sync_result = fit_2anchor_clock(
                anchors=anchors, fps=fps, n_baseline_on=n_baseline_on)
    except ValueError as exc:
        logger.error("Clock fit failed: %s", exc)
        return 2

    existing_sync = load_video_sync(session_name, sync_dir=sync_dir)
    if existing_sync is not None:
        prior = (existing_sync.get("eye_cam") or {}).get("per_trial_overrides") or {}
        if prior:
            sync_result.per_trial_overrides = {int(k): int(v) for k, v in prior.items()}

    # Never silent-overwrite: archive the prior sync+anchor before writing.
    archive_sync_artifacts(session_name, sync_dir=sync_dir)

    out_path = save_video_sync(
        session_name=session_name, eye_cam=sync_result, sync_dir=sync_dir)
    logger.info("Fit: slope=%.6f (%.2f ppm), offset=%.4f s, n=%d, cv_rmse=%.2f ms, quality=%s",
                sync_result.slope, sync_result.slope_ppm, sync_result.offset,
                sync_result.n_anchors, sync_result.cv_rmse_ms, sync_result.quality)
    print(f"Sync JSON: {out_path}")
```

Keep the montage-render block (fit_sync.py:138-183) as-is but pass `sync_dir` where camera files are resolved (the montage uses `find_camera_files(session_name, subject=args.subject)`; update that call).

> **Note for implementer:** `archive_sync_artifacts` runs *before* `save_video_sync`, so the archive captures the prior fit; the freshly-fitted JSON is then written to the live `sync_dir`. The per-trial-override preservation still reads the prior JSON *before* archiving (order above is correct: load → archive → save).

- [ ] **Step 4b: Manual smoke (no automated integration test — needs a real session)**

Run: `py scripts/video/fit_sync.py --session 09092025 --subject BG_046`
Expected: writes `data/cache/video_sync/BG_046/09092025_video_sync.json`; logs `cv_rmse` and `quality`; the prior flat/namespaced JSON (if any) is under `.../BG_046/_archive/<date>/`. (With only 2 legacy anchors it uses `fit_2anchor_clock` → `manual_slope_fit`; after Plan 2 tagging adds ≥3 anchors it routes to `fit_multianchor_clock`.)

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/core/video_sync.py scripts/video/fit_sync.py tests/video/test_sync_archive_and_reconstruct.py
git commit -m "feat(video-sync): fit_sync routes >=3 anchors to validated multianchor fit + archives before overwrite"
```

---

## Task 6: Ceph-safe local metadata reconstruction + read-only-X: guard

**Files:**
- Modify: `src/visdetect/core/video_sync.py` (add `local_reconstructed_metadata_path`; make `find_camera_files` prefer a local reconstructed CSV); `scripts/video/reconstruct_camera_metadata.py:70-140`
- Test: `tests/video/test_sync_archive_and_reconstruct.py` (extend)

**Interfaces:**
- Produces: `local_reconstructed_metadata_path(session, cam_label, subject=None) -> str` → `<subject_video_sync_dir>/<session>_<cam>_metadata.reconstructed.csv`.
- `find_camera_files` returns the local reconstructed CSV as `metadata` when it exists (video still from X:).

- [ ] **Step 1: Write the failing test**

```python
# tests/video/test_sync_archive_and_reconstruct.py  (append)
import numpy as np
from visdetect.core import video_sync as vs


def test_reconstruction_writes_local_not_camera_root(tmp_path, monkeypatch):
    # Camera root is treated as READ-ONLY: a reconstructed CSV must NOT appear there.
    cam_root = tmp_path / "X"
    cam_dir = cam_root / "BG_046_010725"
    cam_dir.mkdir(parents=True)
    (cam_dir / "BG_046_010725_Eye_cam.mp4").write_bytes(b"x")
    (cam_dir / "BG_046_010725_Eye_cam_metadata.csv").write_text("Timestamp (ms)\n")  # header-only
    local = tmp_path / "sync"
    monkeypatch.setattr(vs, "CAMERA_ROOT", str(cam_root))
    monkeypatch.setattr(
        "visdetect.analysis.config.VIDEO_SYNC_DIR", str(local), raising=False)

    out = vs.write_local_reconstructed_metadata(
        "01072025", "eye_cam", frame_count=100, fps=50.0, subject="BG_046")
    assert str(local) in out and out.endswith("reconstructed.csv")
    # camera dir untouched apart from the original header-only file
    names = sorted(p.name for p in cam_dir.iterdir())
    assert names == ["BG_046_010725_Eye_cam.mp4", "BG_046_010725_Eye_cam_metadata.csv"]
    ts, _, _ = vs.load_camera_metadata(out)
    assert len(ts) == 100 and abs(ts[1] - 20.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/video/test_sync_archive_and_reconstruct.py -k reconstruction -v`
Expected: FAIL — `write_local_reconstructed_metadata` undefined.

- [ ] **Step 3a: Add local reconstruction to video_sync.py**

```python
def local_reconstructed_metadata_path(
    session_name: str, cam_label: str, subject: Optional[str] = None) -> str:
    from visdetect.analysis.config import subject_video_sync_dir, canonical_camera_session
    sn = canonical_camera_session(session_name)
    return os.path.join(
        subject_video_sync_dir(subject), f"{sn}_{cam_label}_metadata.reconstructed.csv")


def write_local_reconstructed_metadata(
    session_name: str, cam_label: str, frame_count: int, fps: float,
    subject: Optional[str] = None) -> str:
    """Write reconstructed steady-fps metadata to LOCAL cache (never X:)."""
    out = local_reconstructed_metadata_path(session_name, cam_label, subject)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_reconstructed_metadata(out, frame_count, fps)  # existing writer, local path
    return out
```

- [ ] **Step 3b: Make `find_camera_files` prefer the local reconstructed CSV**

In `find_camera_files`, after building `result` and before `return result`, add:

```python
    from visdetect.analysis.config import canonical_camera_session
    sn = canonical_camera_session(session_name)
    for cam_label in list(result.keys()):
        local_meta = local_reconstructed_metadata_path(sn, cam_label, subject)
        if os.path.exists(local_meta):
            result[cam_label]["metadata"] = local_meta
```

- [ ] **Step 3c: Rewire reconstruct_camera_metadata.py to write local**

Replace the three-write body of `reconstruct_camera` (reconstruct_camera_metadata.py:101-122) so it (a) does NOT touch X:, (b) writes the reconstructed CSV + provenance under `subject_video_sync_dir`:

```python
    from visdetect.core.video_sync import (
        write_local_reconstructed_metadata, local_reconstructed_metadata_path)
    from visdetect.analysis.config import subject_video_sync_dir
    local_csv = write_local_reconstructed_metadata(
        session_name, cam_label, frame_count, fps, subject=subject)
    logger.info("[%s/%s] reconstructed (LOCAL, X: untouched) -> %s",
                session_name, cam_label, local_csv)
    prov = {
        "session": session_name, "camera": cam_label, "source": "RECONSTRUCTED_LOCAL",
        "method": "linear steady-fps (ts[i] = i*1000/fps) from video container",
        "frame_count": frame_count, "fps": fps, "duration_s": frame_count / fps,
        "video": video_path, "reconstructed_at": datetime.now().isoformat(timespec="seconds"),
        "tool": "scripts/video/reconstruct_camera_metadata.py",
    }
    prov_path = local_csv[: -len(".csv")] + ".json"
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2)
```

Add a `--subject` arg to `main()` (reconstruct_camera_metadata.py:147) and thread `subject=args.subject` into `reconstruct_camera(...)` (add the `subject` param to its signature).

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/video/test_sync_archive_and_reconstruct.py -v`
Expected: PASS (archive + reconstruction-local)

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/core/video_sync.py scripts/video/reconstruct_camera_metadata.py tests/video/test_sync_archive_and_reconstruct.py
git commit -m "feat(video-sync): reconstruct camera metadata to local cache (X: strictly read-only)"
```

---

## Task 7: Local video staging helper (read-only source)

**Files:**
- Modify: `src/visdetect/core/video_sync.py` (add `stage_session_video`, `unstage_session_video`)
- Test: `tests/video/test_stage_video.py`

**Interfaces:**
- Produces:
  - `stage_session_video(session, subject=None, cams=("eye_cam",), camera_root=None, staging_dir=None, force=False) -> Dict[str, Dict[str, str]]` — copies video+metadata from `find_camera_files` into `<VIDEO_STAGING_DIR>/<subject>/<session>/`; returns the same dict shape with local paths. Source read-only (copy, never move).
  - `unstage_session_video(session, subject=None, staging_dir=None) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/video/test_stage_video.py
import os
from visdetect.core import video_sync as vs


def test_stage_copies_and_leaves_source_intact(tmp_path, monkeypatch):
    cam_root = tmp_path / "X"
    cam_dir = cam_root / "BG_046_010725"
    cam_dir.mkdir(parents=True)
    (cam_dir / "BG_046_010725_Eye_cam.mp4").write_bytes(b"video")
    (cam_dir / "BG_046_010725_Eye_cam_metadata.csv").write_text("Timestamp (ms)\n0\n")
    staging = tmp_path / "stage"

    out = vs.stage_session_video(
        "01072025", subject="BG_046", cams=("eye_cam",),
        camera_root=str(cam_root), staging_dir=str(staging))
    assert os.path.exists(out["eye_cam"]["video"])
    assert str(staging) in out["eye_cam"]["video"]
    # source untouched
    assert (cam_dir / "BG_046_010725_Eye_cam.mp4").read_bytes() == b"video"
    # idempotent (force=False -> no error, returns same paths)
    out2 = vs.stage_session_video(
        "01072025", subject="BG_046", cams=("eye_cam",),
        camera_root=str(cam_root), staging_dir=str(staging))
    assert out2["eye_cam"]["video"] == out["eye_cam"]["video"]

    vs.unstage_session_video("01072025", subject="BG_046", staging_dir=str(staging))
    assert not os.path.exists(out["eye_cam"]["video"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/video/test_stage_video.py -v`
Expected: FAIL — `stage_session_video` undefined.

- [ ] **Step 3: Add the staging helpers**

```python
def _staging_dir(session_name: str, subject: Optional[str], staging_dir: Optional[str]) -> str:
    from visdetect.analysis.config import VIDEO_STAGING_DIR, SUBJECT, canonical_camera_session
    base = staging_dir or VIDEO_STAGING_DIR
    return os.path.join(base, subject or SUBJECT, canonical_camera_session(session_name))


def stage_session_video(
    session_name: str,
    subject: Optional[str] = None,
    cams=("eye_cam",),
    camera_root: Optional[str] = None,
    staging_dir: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Dict[str, str]]:
    """Copy a session's camera video+metadata from X: (read-only) to local scratch.

    Bulk sequential read only; never writes to CAMERA_ROOT. Returns the same
    dict shape as find_camera_files but with LOCAL paths.
    """
    import shutil
    src = find_camera_files(session_name, camera_root=camera_root, subject=subject)
    dst_dir = _staging_dir(session_name, subject, staging_dir)
    os.makedirs(dst_dir, exist_ok=True)
    out: Dict[str, Dict[str, str]] = {}
    for cam in cams:
        if cam not in src:
            continue
        out[cam] = {}
        for kind, spath in src[cam].items():
            dpath = os.path.join(dst_dir, os.path.basename(spath))
            if force or not os.path.exists(dpath):
                shutil.copy2(spath, dpath)  # copy2, never move -> source intact
            out[cam][kind] = dpath
    return out


def unstage_session_video(
    session_name: str, subject: Optional[str] = None,
    staging_dir: Optional[str] = None) -> None:
    """Delete the local staged copy for a session (frees disk)."""
    import shutil
    dst_dir = _staging_dir(session_name, subject, staging_dir)
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/video/test_stage_video.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/core/video_sync.py tests/video/test_stage_video.py
git commit -m "feat(video-sync): local video staging helper (read-only X: source)"
```

---

## Task 8: Subject-aware sync_status

**Files:**
- Modify: `scripts/video/sync_status.py`
- Test: none automated (print tool over live manifest); manual verify.

**Interfaces:**
- Consumes: `subject_video_sync_dir`, `canonical_camera_session`, `load_anchor`/`load_video_sync` (with `sync_dir`).
- Produces: `--subject` flag; per-session `cv_rmse_ms` column; namespaced reads.

- [ ] **Step 1: Add `--subject` + namespaced reads**

In `_session_status` (sync_status.py:31), accept a `sync_dir` and report cv_rmse:

```python
def _session_status(session_int, sync_dir) -> dict:
    from visdetect.analysis.config import canonical_camera_session, session_int_to_iso
    name = canonical_camera_session(session_int)
    anchor = load_anchor(name, sync_dir=sync_dir)
    n_anchors = len(anchor["anchors"]) if anchor else 0
    sync = load_video_sync(name, sync_dir=sync_dir)
    if sync is not None:
        eye = sync.get("eye_cam") or {}
        slope_ppm = eye.get("slope_ppm")
        cv_rmse = eye.get("cv_rmse_ms")
        quality = sync.get("quality") or eye.get("quality")
        n_over = len(eye.get("per_trial_overrides") or {})
        status = "DONE"
    else:
        slope_ppm = quality = cv_rmse = None
        n_over = 0
        status = "PARTIAL" if n_anchors >= 1 else "TODO"
    return {"session": name, "iso": session_int_to_iso(session_int), "status": status,
            "n_anchors": n_anchors, "slope_ppm": slope_ppm, "cv_rmse": cv_rmse,
            "quality": quality, "n_over": n_over}
```

In `main()` add the arg and thread the namespaced dir:

```python
    p.add_argument("--subject", default=None, help="Subject (default: config.SUBJECT).")
    # ... after parsing:
    from visdetect.analysis.config import subject_video_sync_dir
    sync_dir = subject_video_sync_dir(args.subject)
    # manifest selection: load_staging_manifest keys off config.SUBJECT; for a
    # non-default subject set VISDETECT_SUBJECT or pass manifest_path. Document:
    #   VISDETECT_SUBJECT=BG_031 py scripts/video/sync_status.py --subject BG_031
    rows = [_session_status(s, sync_dir) for s in sessions]
```

Add a `cv_ms` column to the header/format strings (mirror the existing `ppm` column formatting).

- [ ] **Step 2: Manual verify**

Run: `py scripts/video/sync_status.py --subject BG_046`
Expected: table lists BG_046 roster with a `cv_ms` column; DONE rows read from `data/cache/video_sync/BG_046/`.

- [ ] **Step 3: Commit**

```bash
git add scripts/video/sync_status.py
git commit -m "feat(video-sync): subject-aware sync_status with cv_rmse column"
```

---

## Final: full backbone test run

- [ ] **Run the whole new suite + the existing sync tests**

Run: `py -m pytest tests/video/ tests/test_reconstruct_camera_metadata.py -v`
Expected: all PASS (new backbone tests + the pre-existing ~52 sync tests unaffected).

- [ ] **Verify no regressions in the library import**

Run: `py -c "import visdetect.core.video_sync, visdetect.analysis.config, visdetect.analysis.constants; print('ok')"`
Expected: `ok`

---

## Self-Review (against the spec)

- **§3.5 namespacing** → Task 1 (dirs) + Tasks 5/6/8 (callers pass namespaced `sync_dir`). ✓
- **§3.5 camera-dir fixes (subject/suffix/6-digit)** → Tasks 1–2. ✓
- **§3.8 generalized anchor schema (event_type, change)** → Task 3. ✓
- **§3.1/§3.4 validated fit + CV-RMSE + tiers** → Task 4 (`fit_multianchor_clock`, `_loo_cv`, `manual_multianchor` tier). ✓
- **§3.3 fit routing (≥3 → multianchor)** → Task 5. ✓
- **§3.14 archive-before-write / never silent-overwrite** → Task 5 (`archive_sync_artifacts`). ✓
- **§3.6 ceph-safe reconstruction + read-only-X:** → Task 6 (local reconstruction + guard test). ✓
- **§3.6 local staging helper** → Task 7. ✓
- **§3.1 subject-aware status + cv_rmse** → Task 8. ✓
- **Deferred to Plan 2 (not in this plan):** the tagger GUI (playback, full-frame view, mid-session/change anchoring, ROI + live pupil overlay, per-frame labels), the label-sidecar IO + `data/cache/video_labels/`, the neural sharpening validation figure generalization, and front-cam clock derivation. Flagged here so coverage gaps are intentional, not omissions.

**Placeholder scan:** none — every code step carries real code. **Type consistency:** `fit_multianchor_clock`/`_loo_cv`/`archive_sync_artifacts`/`stage_session_video`/`local_reconstructed_metadata_path`/`canonical_camera_session`/`camera_dir_token` names are used identically across tasks and tests.
