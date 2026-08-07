# Amortized ROI + per-frame pupil label capture (Plan 2b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Plan 2a unified tagger so the same single manual pass that produces the video↔neural sync also captures a per-session mouth/eye ROI and per-frame pupil labels (confirm / correct / blink) into a decoupled JSON sidecar that sub-projects B (motion energy) and C (pupillometry) consume without re-watching the video.

**Architecture:** All non-GUI logic lands in a new, fully unit-tested library module `src/visdetect/analysis/video_labels.py` (schema v1 + atomic IO + upsert + cross-session ROI seeding + crop/ellipse geometry). The GUI wires it up: `scripts/video/_tagger_ui.py` gains two *optional* seams (a `RectangleSelector` mouse hook and a per-refresh overlay hook) that leave `click_anchor` byte-identical, and `scripts/video/tag_session.py` composes those seams into ROI capture, a live pupil overlay, a clamped eye-zoom view, and label capture.

**Tech Stack:** Python 3.10 (`.venv` python), matplotlib TkAgg + `matplotlib.widgets.RectangleSelector`, cv2, numpy, pytest. Windows + git-bash. Detector reused: `visdetect.core.video_sync.detect_pupil_in_frame`.

## Global Constraints

- **venv python:** `E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe` (`py` is NOT available to subagents).
- **PYTHONPATH:** `E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.claude/worktrees/camera-tagger-2b/src`
- **Worktree root (all absolute paths live under this dir):** `E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.claude/worktrees/camera-tagger-2b`
- **Suite command:** `<venv> -m pytest tests/video/ tests/test_video_sync_anchor.py tests/test_video_sync_tag_trials.py -q` → currently **97 passed**, must stay green (plus the new `tests/video/test_video_labels.py`). Run with `PYTHONPATH` set as above, `cwd` = worktree root.
- **`MPLBACKEND=Agg` ONLY for one-off `--help`/import checks**, never globally — several tests subprocess-import GUI scripts and assert the interactive backend survives.
- **Clamp before indexing.** `tagging.eye_zoom_crop` returns UNCLAMPED `(y0,y1,x0,x1)`: padding an ROI near a frame edge yields negative or out-of-frame coords, and numpy slicing with a negative index does NOT error — it silently wraps from the far edge and yields the WRONG crop. Clamp to `0 <= y0 < y1 <= H`, `0 <= x0 < x1 <= W` via `video_labels.clamp_crop` before ANY frame indexing.
- **ROIs stored in FULL-FRAME pixel coordinates always**, never view-relative — a box drawn while (hypothetically) zoomed must still mean the same thing. ROIs are only drawable in the full-frame view.
- **Chronology via the project's canonical id helpers** (`config.session_date_key` / `config.canonical_camera_session`), never a raw `sorted()`/`max()` on id strings: `'1072025'` sorts before `'23062025'` lexically though 1 Jul is after 23 Jun, and BG_031/BG_039 use 6-digit `DDMMYY` ids. "Most recent prior session" is a **date** comparison.
- **Atomic writes** (temp file + `os.replace`, mirroring `tag_trials._persist_overrides`) on every ROI/label change, plus a final flush on quit. A crash mid-session must not corrupt or truncate prior work.
- **`frames` keyed on `frame_idx`** — `upsert_frame_label` replaces the existing entry, never appends a duplicate.
- **`click_anchor` must remain behaviorally unchanged.** Its `ScrubberConfig` never sets the new `on_selector`/`on_refresh` fields (they default `None`); the new selector-armed navigation guard only fires when a tool has armed the selector, which `click_anchor` never does.
- **Matplotlib keymap collisions:** `p` = pan and `f` = fullscreen are matplotlib defaults; `_tagger_ui.run_scrubber` already clears `keymap.{save,fullscreen,xscale,yscale,back,forward,home,pan,zoom,grid}` (~line 187), which covers both — this is **load-bearing**. The new keys `m`, `e`, `u`, `x` are free of matplotlib defaults. Any future key must be checked against that list.
- **Labels are `confirm`/`correct`/`blink`** (`verdict` values `"confirmed"`/`"corrected"`/`"blink"`). A **correction stores BOTH** the detector's proposed ellipse and the human's corrected ellipse (a bare "wrong" tag cannot quantify the eyelid-occlusion diameter bias).
- **Subject model for any subagent: Opus 4.8** (`claude-opus-4-8`).

**Design (contract — do not redesign):** `docs/superpowers/specs/2026-07-30-camera-tagger-roi-labels-2b-design.md`. Sibling executed plan: `docs/superpowers/plans/2026-07-24-camera-tagger-core-2a.md`.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/visdetect/analysis/video_labels.py` | create | Pure, no-GUI: schema v1 constants; `label_sidecar_path`, `new_sidecar`, `load_sidecar`, `save_sidecar` (atomic); `set_roi`; `upsert_frame_label` (frame-keyed upsert); `seed_rois_from_previous` (date-chronological, frame-size-guarded, provenance); `clamp_crop`; `ellipse_from_box`; `ellipse_from_detection` |
| `tests/video/test_video_labels.py` | create | Tasks 1–3 pure-logic tests (schema round-trip, atomic-write failure, upsert idempotence, date-based seeding incl. leading-zero-day, frame-size mismatch, clamp, inscribed ellipse, detector→ellipse mapping) |
| `scripts/video/_tagger_ui.py` | modify | Add two OPTIONAL `ScrubberConfig` seams — `on_selector` (RectangleSelector arm/disarm + drag→full-frame box) and `on_refresh` (per-frame overlay hook) — plus a selector-armed navigation guard; `click_anchor` path unaffected |
| `scripts/video/tag_session.py` | modify | ROI capture (`m`/`e`), live pupil overlay via `detect_pupil_in_frame`, `f` clamped eye-zoom, label capture (`u`/`p`/`x`), atomic sidecar persistence, seeding on open, HUD extensions |

---

## Task 1: `video_labels` schema v1 + atomic IO + upsert

**Files:** Create `src/visdetect/analysis/video_labels.py`; Create `tests/video/test_video_labels.py`.

**Interfaces:**
- Consumes (existing, verified): `config.subject_video_labels_dir(subject=None) -> str` (`data/cache/video_labels/<subject>/`), `config.canonical_camera_session(session) -> str` (8-digit `DDMMYYYY` for any token, incl. 6-digit `DDMMYY`), `config.session_date_key(session) -> (yyyy,mm,dd)`.
- Produces:
  - `SCHEMA_VERSION = 1`; `VERDICT_CONFIRMED = "confirmed"`, `VERDICT_CORRECTED = "corrected"`, `VERDICT_BLINK = "blink"`.
  - `label_sidecar_path(session, subject=None, labels_dir=None) -> str`
  - `new_sidecar(subject: str, session, frame_size, camera="eye_cam") -> dict`
  - `load_sidecar(session, subject=None, labels_dir=None) -> Optional[dict]`
  - `save_sidecar(sidecar: dict, session, subject=None, labels_dir=None) -> None` (atomic)
  - `set_roi(sidecar: dict, name: str, box, source: str) -> dict`
  - `upsert_frame_label(sidecar, frame_idx, verdict, proposed_ellipse=None, corrected_ellipse=None, labeled_at=None) -> dict`

- [ ] **Step 1: Write the failing test** — create `tests/video/test_video_labels.py`:

```python
# tests/video/test_video_labels.py
import json
import os

import pytest

from visdetect.analysis import video_labels as vl


# ---------------------------------------------------------------------------
# Task 1: schema v1 + atomic IO + upsert
# ---------------------------------------------------------------------------


def test_new_sidecar_schema_v1_shape():
    sc = vl.new_sidecar("BG_031", "09042025", [976, 1024], camera="eye_cam")
    assert sc["schema_version"] == 1
    assert sc["subject"] == "BG_031"
    assert sc["session"] == "09042025"
    assert sc["camera"] == "eye_cam"
    assert sc["frame_size"] == [976, 1024]
    assert sc["rois"] == {}
    assert sc["frames"] == []


def test_sidecar_round_trip(tmp_path):
    sc = vl.new_sidecar("BG_031", "09042025", [976, 1024])
    vl.set_roi(sc, "eye", [300, 400, 500, 600], source="drawn")
    vl.save_sidecar(sc, "09042025", "BG_031", labels_dir=str(tmp_path))
    loaded = vl.load_sidecar("09042025", "BG_031", labels_dir=str(tmp_path))
    assert loaded["schema_version"] == vl.SCHEMA_VERSION
    assert loaded["rois"]["eye"] == {"box": [300, 400, 500, 600], "source": "drawn"}


def test_load_sidecar_missing_returns_none(tmp_path):
    assert vl.load_sidecar("09042025", "BG_031", labels_dir=str(tmp_path)) is None


def test_save_sidecar_atomic_leaves_no_partial_on_failure(tmp_path, monkeypatch):
    sc = vl.new_sidecar("BG_TEST", "01072025", [10, 10])
    # Pre-existing good file must survive a failed rewrite.
    vl.save_sidecar(sc, "01072025", "BG_TEST", labels_dir=str(tmp_path))

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(vl.json, "dump", boom)
    with pytest.raises(RuntimeError):
        vl.save_sidecar(sc, "01072025", "BG_TEST", labels_dir=str(tmp_path))
    # Original file intact, no leftover temp file.
    assert (tmp_path / "01072025.json").exists()
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())


def test_upsert_frame_label_replaces_not_duplicates():
    sc = vl.new_sidecar("BG_TEST", "01072025", [10, 10])
    vl.upsert_frame_label(sc, 42, vl.VERDICT_CONFIRMED,
                          proposed_ellipse={"cx": 1.0, "cy": 2.0, "major": 3.0,
                                            "minor": 3.0, "angle": 0.0})
    vl.upsert_frame_label(sc, 7, vl.VERDICT_BLINK)
    vl.upsert_frame_label(sc, 42, vl.VERDICT_BLINK)  # re-label -> replace, not append
    frames = sc["frames"]
    assert len(frames) == 2
    e42 = [f for f in frames if f["frame_idx"] == 42][0]
    assert e42["verdict"] == vl.VERDICT_BLINK
    assert e42["proposed_ellipse"] is None       # replacement cleared the old proposal
    assert e42["corrected_ellipse"] is None
    assert isinstance(e42["labeled_at"], str) and e42["labeled_at"]


def test_upsert_frame_label_correction_stores_both_ellipses():
    sc = vl.new_sidecar("BG_TEST", "01072025", [10, 10])
    proposed = {"cx": 10.0, "cy": 20.0, "major": 8.0, "minor": 8.0, "angle": 0.0}
    corrected = {"cx": 11.0, "cy": 21.0, "major": 12.0, "minor": 9.0, "angle": 0.0}
    vl.upsert_frame_label(sc, 99, vl.VERDICT_CORRECTED,
                          proposed_ellipse=proposed, corrected_ellipse=corrected)
    e = sc["frames"][0]
    assert e["verdict"] == "corrected"
    assert e["proposed_ellipse"] == proposed
    assert e["corrected_ellipse"] == corrected
```

- [ ] **Step 2: Run — verify fail**

Run (cwd = worktree root, `PYTHONPATH` set): `<venv> -m pytest tests/video/test_video_labels.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'visdetect.analysis.video_labels'`.

- [ ] **Step 3: Implement** — create `src/visdetect/analysis/video_labels.py`:

```python
"""Pure sidecar schema + IO for per-session ROIs and per-frame pupil labels.

Plan 2b: everything the unified tagger's ROI/label capture needs that is NOT
GUI — the JSON schema (v1), atomic load/save, frame-label upsert, ROI setters,
cross-session ROI seeding, and the crop/ellipse geometry helpers. No cv2, no
matplotlib: fully unit-testable in isolation.

Sidecar location: ``data/cache/video_labels/<subject>/<session>.json`` (see
``config.subject_video_labels_dir``), decoupled from the anchor/sync JSON so the
label schema can evolve with sub-projects B/C independently of the sync contract.

Schema (v1)::

    {
      "schema_version": 1,
      "subject": "BG_031",
      "session": "09042025",
      "camera": "eye_cam",
      "frame_size": [H, W],
      "rois": {"eye": {"box": [y0,y1,x0,x1], "source": "drawn|inherited:<sess>"}, ...},
      "frames": [{"frame_idx": int, "verdict": "confirmed|corrected|blink",
                  "proposed_ellipse": {..}|null, "corrected_ellipse": {..}|null,
                  "labeled_at": "<iso8601>"}]
    }
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Optional

from visdetect.analysis.config import (
    subject_video_labels_dir,
    canonical_camera_session,
    session_date_key,
)

SCHEMA_VERSION = 1

VERDICT_CONFIRMED = "confirmed"
VERDICT_CORRECTED = "corrected"
VERDICT_BLINK = "blink"


def label_sidecar_path(session, subject: Optional[str] = None,
                       labels_dir: Optional[str] = None) -> str:
    """Absolute path of the label sidecar for *session* / *subject*.

    Filename is the canonical 8-digit ``DDMMYYYY`` session id so 6-digit
    ``DDMMYY`` subjects (BG_031/039) and leading-zero-day ids never collide.
    """
    d = labels_dir or subject_video_labels_dir(subject)
    return os.path.join(d, f"{canonical_camera_session(session)}.json")


def new_sidecar(subject: str, session, frame_size,
                camera: str = "eye_cam") -> dict:
    """Fresh schema-v1 sidecar dict (empty rois + frames). ``frame_size`` is
    ``(H, W)`` and is stored as ``[H, W]``."""
    return {
        "schema_version": SCHEMA_VERSION,
        "subject": str(subject),
        "session": canonical_camera_session(session),
        "camera": str(camera),
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
        "rois": {},
        "frames": [],
    }


def load_sidecar(session, subject: Optional[str] = None,
                 labels_dir: Optional[str] = None) -> Optional[dict]:
    """Read the sidecar JSON, or ``None`` if it does not exist."""
    path = label_sidecar_path(session, subject, labels_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def save_sidecar(sidecar: dict, session, subject: Optional[str] = None,
                 labels_dir: Optional[str] = None) -> None:
    """Atomically write *sidecar* (temp file + ``os.replace``), mirroring
    ``tag_trials._persist_overrides`` — a crash mid-write never corrupts the
    prior file and never leaves a partial one in place."""
    path = label_sidecar_path(session, subject, labels_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(sidecar, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def set_roi(sidecar: dict, name: str, box, source: str) -> dict:
    """Set ``sidecar['rois'][name] = {'box': [y0,y1,x0,x1], 'source': source}``.

    ``name`` is ``'eye'`` or ``'mouth'``; ``box`` is a FULL-FRAME
    ``(y0,y1,x0,x1)``; ``source`` is ``'drawn'`` or ``'inherited:<session>'``.
    Re-drawing an inherited ROI is the caller's cue to pass ``source='drawn'``.
    Returns *sidecar* (mutated in place).
    """
    sidecar.setdefault("rois", {})[name] = {
        "box": [int(v) for v in box],
        "source": str(source),
    }
    return sidecar


def upsert_frame_label(sidecar: dict, frame_idx: int, verdict: str,
                       proposed_ellipse: Optional[dict] = None,
                       corrected_ellipse: Optional[dict] = None,
                       labeled_at: Optional[str] = None) -> dict:
    """Insert or REPLACE the label for *frame_idx* (keyed on ``frame_idx``;
    never duplicates). A ``corrected`` verdict is expected to carry BOTH
    ``proposed_ellipse`` and ``corrected_ellipse``. Returns *sidecar*."""
    entry = {
        "frame_idx": int(frame_idx),
        "verdict": str(verdict),
        "proposed_ellipse": proposed_ellipse,
        "corrected_ellipse": corrected_ellipse,
        "labeled_at": labeled_at or datetime.now(timezone.utc).isoformat(),
    }
    frames = sidecar.setdefault("frames", [])
    for i, fr in enumerate(frames):
        if int(fr.get("frame_idx", -1)) == int(frame_idx):
            frames[i] = entry
            return sidecar
    frames.append(entry)
    return sidecar
```

- [ ] **Step 4: Run — verify pass**

Run: `<venv> -m pytest tests/video/test_video_labels.py -q` → all Task-1 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/video_labels.py tests/video/test_video_labels.py
git commit -m "feat(tagger): video_labels sidecar schema v1 + atomic IO + frame-keyed upsert"
```

---

## Task 2: `seed_rois_from_previous` — date-chronological seeding + frame-size guard + provenance

**Files:** Modify `src/visdetect/analysis/video_labels.py`; Modify `tests/video/test_video_labels.py` (append).

**Interfaces:**
- Consumes: `config.session_date_key`, `config.canonical_camera_session`, `config.subject_video_labels_dir`, and the sidecars written by Task 1 (`frame_size`, `rois`).
- Produces: `seed_rois_from_previous(session, subject, current_frame_size, labels_dir=None) -> Optional[dict]` returning
  `{"source_session": "<prior DDMMYYYY>", "rois": {name: {"box": [...], "source": "inherited:<prior>"}}, "frame_size": [H,W], "applied": bool}`,
  where `applied = (prior frame_size == list(current_frame_size))`, or `None` when no strictly-earlier prior session sidecar exists.

- [ ] **Step 1: Write the failing test** — append to `tests/video/test_video_labels.py`:

```python
# ---------------------------------------------------------------------------
# Task 2: cross-session ROI seeding (most-recent PRIOR by DATE)
# ---------------------------------------------------------------------------


def _write_prior(labels_dir, session, frame_size, eye_box):
    sc = vl.new_sidecar("BG_TEST", session, frame_size)
    vl.set_roi(sc, "eye", eye_box, source="drawn")
    vl.save_sidecar(sc, session, "BG_TEST", labels_dir=str(labels_dir))


def test_seed_picks_most_recent_prior_by_date_not_lexical(tmp_path):
    # Lexical max of the 8-digit strings would be '28062025' (28 Jun); the correct
    # most-recent PRIOR to 05072025 (5 Jul) is '01072025' (1 Jul, leading-zero day).
    _write_prior(tmp_path, "28062025", [976, 1024], [1, 1, 1, 1])
    _write_prior(tmp_path, "01072025", [976, 1024], [7, 7, 7, 7])
    res = vl.seed_rois_from_previous("05072025", "BG_TEST", (976, 1024),
                                     labels_dir=str(tmp_path))
    assert res is not None
    assert res["source_session"] == "01072025"     # date-based, not lexical
    assert res["applied"] is True
    assert res["rois"]["eye"] == {"box": [7, 7, 7, 7], "source": "inherited:01072025"}


def test_seed_never_picks_a_later_session(tmp_path):
    _write_prior(tmp_path, "09072025", [976, 1024], [1, 1, 1, 1])  # 9 Jul (later)
    assert vl.seed_rois_from_previous("05072025", "BG_TEST", (976, 1024),
                                      labels_dir=str(tmp_path)) is None


def test_seed_none_when_no_prior(tmp_path):
    assert vl.seed_rois_from_previous("05072025", "BG_TEST", (976, 1024),
                                      labels_dir=str(tmp_path)) is None


def test_seed_ddmmyy_six_digit_ids(tmp_path):
    # 6-digit DDMMYY (BG_031/039 form). canonical_camera_session maps both the
    # prior filename and the query to 8-digit; seeding compares by DATE.
    _write_prior(tmp_path, "080425", [976, 1024], [5, 5, 5, 5])   # 8 Apr 2025
    res = vl.seed_rois_from_previous("090425", "BG_TEST", (976, 1024),
                                     labels_dir=str(tmp_path))     # 9 Apr 2025
    assert res is not None
    assert res["source_session"] == "08042025"
    assert res["rois"]["eye"]["source"] == "inherited:08042025"


def test_seed_frame_size_mismatch_offers_but_does_not_apply(tmp_path):
    _write_prior(tmp_path, "01072025", [976, 1024], [7, 7, 7, 7])
    res = vl.seed_rois_from_previous("05072025", "BG_TEST", (500, 500),
                                     labels_dir=str(tmp_path))
    assert res is not None
    assert res["source_session"] == "01072025"
    assert res["applied"] is False                 # different resolution -> not applied
    assert res["frame_size"] == [976, 1024]
```

- [ ] **Step 2: Run — verify fail**

Run: `<venv> -m pytest tests/video/test_video_labels.py -k seed -q`
Expected: FAIL — `AttributeError: module 'visdetect.analysis.video_labels' has no attribute 'seed_rois_from_previous'`.

- [ ] **Step 3: Implement** — append to `src/visdetect/analysis/video_labels.py`:

```python
def seed_rois_from_previous(session, subject, current_frame_size,
                            labels_dir: Optional[str] = None) -> Optional[dict]:
    """Return the most-recent PRIOR session's ROIs, or ``None``.

    Camera geometry is usually fixed within a subject, so a new session inherits
    the last session's ROIs as editable seeds instead of drawing from scratch.

    "Most recent prior" is a **date** comparison via
    :func:`config.session_date_key`, never a lexical/int sort — ``'1072025'``
    sorts before ``'23062025'`` lexically though 1 Jul is after 23 Jun, and
    6-digit ``DDMMYY`` ids exist. Only sidecars strictly EARLIER than *session*
    are eligible; a later session is never chosen.

    Provenance: every returned ROI is marked ``source='inherited:<prior>'`` so a
    silently-copied-forward ROI is distinguishable from one a human drew. The
    caller flips it to ``'drawn'`` (via :func:`set_roi`) the moment it is re-dragged.

    Frame-size guard: an absolute-pixel box is meaningless at a different
    resolution, so ``applied`` is ``True`` only when the prior sidecar's
    ``frame_size`` equals ``current_frame_size`` (as ``(H, W)``). On a mismatch the
    ROIs are still returned (``applied=False``) so the caller can warn/offer them.

    Returns ``{"source_session", "rois", "frame_size", "applied"}`` or ``None``.
    """
    d = labels_dir or subject_video_labels_dir(subject)
    if not os.path.isdir(d):
        return None
    cur = canonical_camera_session(session)
    cur_key = session_date_key(cur)
    best = None  # (date_key, stem)
    for fn in os.listdir(d):
        if not fn.endswith(".json"):
            continue
        stem = fn[:-len(".json")]
        if stem == cur:
            continue
        try:
            k = session_date_key(stem)
        except ValueError:
            continue
        if k >= cur_key:            # not strictly earlier -> ineligible
            continue
        if best is None or k > best[0]:
            best = (k, stem)
    if best is None:
        return None
    with open(os.path.join(d, best[1] + ".json"), "r") as f:
        prior = json.load(f)
    prior_fs = list(prior.get("frame_size") or [])
    rois = {}
    for name, r in (prior.get("rois") or {}).items():
        rois[name] = {"box": [int(v) for v in r["box"]],
                      "source": f"inherited:{best[1]}"}
    applied = prior_fs == [int(v) for v in current_frame_size]
    return {"source_session": best[1], "rois": rois,
            "frame_size": prior_fs, "applied": applied}
```

- [ ] **Step 4: Run — verify pass**

Run: `<venv> -m pytest tests/video/test_video_labels.py -q` → all Task-1 + Task-2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/video_labels.py tests/video/test_video_labels.py
git commit -m "feat(tagger): seed ROIs from most-recent prior session (date-chronological, frame-size guarded, provenance)"
```

---

## Task 3: `clamp_crop` + `ellipse_from_box` + `ellipse_from_detection` (crop/ellipse geometry)

**Files:** Modify `src/visdetect/analysis/video_labels.py`; Modify `tests/video/test_video_labels.py` (append).

**Interfaces:**
- Consumes: the raw dict returned by `video_sync.detect_pupil_in_frame`, whose FULL-FRAME keys are `center_y`, `center_x`, `radius`, `area`, `circularity`, `bbox` — it does NOT expose the minor axis or rotation (only `radius = max(axes)/2`). See the report note in the plan's tail.
- Produces:
  - `clamp_crop(crop, H, W) -> Optional[Tuple[int,int,int,int]]` — clamp `(y0,y1,x0,x1)` into the frame, returning a guaranteed-NON-EMPTY `(y0,y1,x0,x1)` (`0<=y0<y1<=H`, `0<=x0<x1<=W`) when the box intersects, or `None` when it does not — the clamped width/height would be zero (box entirely off-frame) OR the box is inverted/malformed (inverted boxes are NOT coordinate-swapped, which would invent an ROI the user never drew). `None` means "no valid crop — the caller MUST fall back" (for the GUI: stay on / revert to the full frame).
  - `ellipse_from_box(box) -> dict` — inscribed axis-aligned ellipse `{cx,cy,major,minor,angle}` from a drag box `(y0,y1,x0,x1)`; `major=max(width,height)`, `minor=min(width,height)`, `angle=0.0` when wider-than-tall else `90.0`.
  - `ellipse_from_detection(det) -> Optional[dict]` — map the detector dict to `{cx,cy,major,minor,angle}` (a circle of diameter `2*radius`, `angle=0.0`, preserving the true major diameter); `None` when `det` is `None`.

- [ ] **Step 1: Write the failing test** — append to `tests/video/test_video_labels.py`:

```python
# ---------------------------------------------------------------------------
# Task 3: crop clamp + ellipse geometry
# ---------------------------------------------------------------------------


def test_clamp_crop_negative_oversize():
    # partially outside but still intersecting: negatives -> 0, oversize -> H/W
    assert vl.clamp_crop((-30, 500, -20, 700), 480, 640) == (0, 480, 0, 640)
    # already valid -> unchanged
    assert vl.clamp_crop((100, 200, 150, 250), 480, 640) == (100, 200, 150, 250)


def test_clamp_crop_non_intersecting_returns_none():
    # box entirely below/right of the frame -> no intersection -> None
    assert vl.clamp_crop((500, 600, 700, 800), 480, 640) is None
    # box entirely in negative space (above/left of the frame) -> None
    assert vl.clamp_crop((-50, -10, -30, -5), 480, 640) is None
    # inverted (y1<y0 / x1<x0) is malformed -> None (NOT silently swapped: a swap
    # would invent an ROI the user never drew)
    assert vl.clamp_crop((300, 100, 400, 200), 480, 640) is None


def test_clamp_crop_partial_still_clamps_to_valid_nonempty():
    # partially past the bottom-right edge -> clamps to a valid, NON-EMPTY crop
    # (regression guard that the None path did not break the normal clamp path).
    out = vl.clamp_crop((400, 999, 500, 999), 480, 640)
    assert out == (400, 480, 500, 640)
    y0, y1, x0, x1 = out
    assert 0 <= y0 < y1 <= 480
    assert 0 <= x0 < x1 <= 640


def test_ellipse_from_box_axis_aligned():
    # y:100-200 (h=100), x:300-500 (w=200) -> wider than tall -> major=w, angle 0
    assert vl.ellipse_from_box((100, 200, 300, 500)) == {
        "cx": 400.0, "cy": 150.0, "major": 200.0, "minor": 100.0, "angle": 0.0}
    # y:100-400 (h=300), x:300-500 (w=200) -> taller than wide -> major=h, angle 90
    assert vl.ellipse_from_box((100, 400, 300, 500)) == {
        "cx": 400.0, "cy": 250.0, "major": 300.0, "minor": 200.0, "angle": 90.0}


def test_ellipse_from_detection_maps_radius_to_circle():
    det = {"center_x": 512.0, "center_y": 480.0, "radius": 20.0,
           "area": 1200.0, "circularity": 0.9, "bbox": (460, 500, 492, 532)}
    assert vl.ellipse_from_detection(det) == {
        "cx": 512.0, "cy": 480.0, "major": 40.0, "minor": 40.0, "angle": 0.0}
    assert vl.ellipse_from_detection(None) is None
```

- [ ] **Step 2: Run — verify fail**

Run: `<venv> -m pytest tests/video/test_video_labels.py -k "clamp or ellipse" -q`
Expected: FAIL — `AttributeError: ... has no attribute 'clamp_crop'`.

- [ ] **Step 3: Implement** — append to `src/visdetect/analysis/video_labels.py` (add `from typing import Tuple` to the existing typing import at the top: change `from typing import Optional` to `from typing import Optional, Tuple`):

```python
def clamp_crop(crop, H: int, W: int) -> Optional[Tuple[int, int, int, int]]:
    """Clamp a ``(y0,y1,x0,x1)`` crop into the frame, or ``None`` if it misses it.

    HARD REQUIREMENT (design §7): ``tagging.eye_zoom_crop`` returns UNCLAMPED
    coords — padding an ROI near a frame edge can yield negative or out-of-frame
    values, and numpy slicing with a negative index does NOT error: it silently
    WRAPS from the far edge and returns the WRONG crop. Always clamp here before
    indexing a frame.

    Contract:
      * When the box intersects the frame, return the clamped
        ``(y0,y1,x0,x1)`` with ``0 <= y0 < y1 <= H`` and ``0 <= x0 < x1 <= W``
        (guaranteed non-empty — slicing a frame with it yields a real sub-image).
      * Return ``None`` when there is NO intersection: the clamped width or
        height would be zero (box entirely off-frame), OR the box is malformed.
        ``None`` means "no valid crop — the caller MUST fall back". For the GUI
        that means staying on / reverting to the full frame, never a zoom onto an
        empty array.

    Inverted (malformed) inputs — ``y1 < y0`` or ``x1 < x0`` — are NOT
    order-normalized: silently swapping the coordinates would invent an ROI the
    user never drew, so a malformed box is treated as non-intersecting → ``None``.
    """
    y0, y1, x0, x1 = (int(v) for v in crop)
    # Malformed / degenerate box (inverted or zero-area before clamping): reject
    # outright rather than swapping — a swap would fabricate an unintended ROI.
    if y1 <= y0 or x1 <= x0:
        return None
    y0 = max(0, min(y0, H))
    y1 = max(0, min(y1, H))
    x0 = max(0, min(x0, W))
    x1 = max(0, min(x1, W))
    # After clamping the box may collapse (it lay wholly outside the frame): a
    # zero-width/height slice is empty, so there is no valid crop.
    if y1 <= y0 or x1 <= x0:
        return None
    return (y0, y1, x0, x1)


def ellipse_from_box(box) -> dict:
    """Inscribed axis-aligned ellipse ``{cx,cy,major,minor,angle}`` from a drag
    box ``(y0,y1,x0,x1)``.

    ``major`` = larger of (width, height), ``minor`` = smaller, ``angle`` = 0.0
    when wider-than-tall else 90.0. Rotation beyond 0/90 is intentionally lost
    (design §5): negligible for a near-circular rodent pupil, and the two-drag
    major/minor variant can follow if it ever matters.
    """
    y0, y1, x0, x1 = box
    w = float(x1 - x0)
    h = float(y1 - y0)
    cx = (float(x0) + float(x1)) / 2.0
    cy = (float(y0) + float(y1)) / 2.0
    if w >= h:
        return {"cx": cx, "cy": cy, "major": w, "minor": h, "angle": 0.0}
    return {"cx": cx, "cy": cy, "major": h, "minor": w, "angle": 90.0}


def ellipse_from_detection(det: Optional[dict]) -> Optional[dict]:
    """Map a :func:`video_sync.detect_pupil_in_frame` result to the sidecar
    ellipse schema ``{cx,cy,major,minor,angle}``, or ``None`` if ``det`` is None.

    The detector surfaces only ``center_x``, ``center_y`` and ``radius``
    (``radius = max(axes)/2`` from the internal ``cv2.fitEllipse``); the minor
    axis and rotation are not exposed. The proposed ellipse is therefore stored
    as a CIRCLE of diameter ``2*radius`` (``angle=0``). This preserves the
    scientifically-critical quantity — the pupil's major diameter, which the
    too-small-diameter eyelid-occlusion bias is measured against — while
    collapsing the (unavailable) minor axis. A human ``correct`` supplies a true
    two-axis ellipse via :func:`ellipse_from_box` when the shape matters.
    """
    if det is None:
        return None
    diameter = 2.0 * float(det["radius"])
    return {"cx": float(det["center_x"]), "cy": float(det["center_y"]),
            "major": diameter, "minor": diameter, "angle": 0.0}
```

- [ ] **Step 4: Run — verify pass**

Run: `<venv> -m pytest tests/video/test_video_labels.py -q` → ALL Task-1/2/3 tests PASS.
Then run the full suite to confirm no regression: `<venv> -m pytest tests/video/ tests/test_video_sync_anchor.py tests/test_video_sync_tag_trials.py -q` → **97 + new** passed.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/video_labels.py tests/video/test_video_labels.py
git commit -m "feat(tagger): clamp_crop + inscribed-ellipse geometry + detector->ellipse mapping"
```

---

## Task 4: `_tagger_ui` — OPTIONAL RectangleSelector + per-refresh overlay seams

**Files:** Modify `scripts/video/_tagger_ui.py`. No automated GUI test (backend/selector behavior can't run headless); verified by the full suite staying green (the `click_anchor` path must be untouched) + a headless import/attr check.

**Interfaces:**
- Consumes: existing `ScrubberConfig` (fields `video_path, ts_ms, fps, n_frames, start_frame, crop, hud_fn, on_key_extra, on_save`) and `run_scrubber(cfg)` internals (`state` dict `{"frame_idx","result"}`, `_read_frame`, `_refresh`, `on_key`).
- Produces (additive, all defaulted so `click_anchor` is unaffected):
  - `ScrubberConfig.on_selector: Optional[Callable[[Tuple[int,int,int,int], dict], None]] = None` — called with a FULL-FRAME `(y0,y1,x0,x1)` box when a drag completes.
  - `ScrubberConfig.on_refresh: Optional[Callable[[int, Any], None]] = None` — called at the end of every `_refresh` with `(frame_idx, fig)` (lets a tool redraw an overlay after the frame image updates).
  - In `run_scrubber`'s `state` dict, when `on_selector` is set: `state["arm_selector"]` (a zero-arg callable that activates the selector) and `state["selector_armed"]` (bool). A tool arms the drag from its `on_key_extra`; the selector auto-disarms when the drag completes or is cancelled.

**Why two seams (justification for adjusting the suggested decomposition):** the spec (§7.1) names only the mouse hook, but the live pupil overlay must also refresh when the frame changes via the scrubber's *internal* arrow-step/jump path — which `tag_session` cannot otherwise intercept (arrows are consumed before `on_key_extra`). `on_refresh` is the minimal seam for that; both are optional and inert for `click_anchor`.

- [ ] **Step 1: Add the two optional fields** to `ScrubberConfig` (after `on_save`; both LAST so the no-default fields still precede them):

```python
    on_selector: Optional[Callable[[Tuple[int, int, int, int], dict], None]] = None
    on_refresh: Optional[Callable[[int, Any], None]] = None
```

Update the class docstring's Attributes list to mention `on_selector` (drag→full-frame box, armed on demand) and `on_refresh` (post-frame redraw hook). `Any`, `Callable`, `Optional`, `Tuple` are already imported at the top of the module.

- [ ] **Step 2: Call `on_refresh` inside `_refresh`** — extend the existing `_refresh` (currently sets frame + HUD then `draw_idle`) so it invokes the hook before the canvas redraw:

```python
    def _refresh():
        fi = state["frame_idx"]
        im.set_data(_read_frame(fi))
        hud_text.set_text(cfg.hud_fn(fi))
        if cfg.on_refresh is not None:
            cfg.on_refresh(fi, fig)
        fig.canvas.draw_idle()
```

- [ ] **Step 3: Wire the RectangleSelector** — insert this block AFTER `_refresh` is defined and BEFORE the `keymap` neutralization loop / `mpl_connect`. It is a no-op unless `cfg.on_selector` is set:

```python
    # Optional ROI/correction drag seam. Inert (and click_anchor-safe) unless the
    # tool supplies cfg.on_selector. The selector is armed only on demand (the tool
    # calls state["arm_selector"] from its on_key_extra) and auto-disarms as soon
    # as a drag completes or is cancelled, so arrow-stepping/playback/save/quit
    # keep working while no drag is in progress. Boxes are reported in FULL-FRAME
    # pixel coords: the tool only arms the selector in the full-frame view
    # (cfg.crop is None), where the imshow data coords equal frame pixels.
    selector = {"obj": None}

    def _disarm_selector():
        if selector["obj"] is not None:
            selector["obj"].set_active(False)
        state["selector_armed"] = False

    if cfg.on_selector is not None:
        from matplotlib.widgets import RectangleSelector

        def _on_select(eclick, erelease):
            if eclick.xdata is None or erelease.xdata is None:
                _disarm_selector()
                return
            x0, x1 = sorted((eclick.xdata, erelease.xdata))
            y0, y1 = sorted((eclick.ydata, erelease.ydata))
            box = (int(round(y0)), int(round(y1)), int(round(x0)), int(round(x1)))
            _disarm_selector()
            cfg.on_selector(box, state)
            _refresh()

        selector["obj"] = RectangleSelector(
            ax_frame, _on_select, useblit=False, button=[1],
            minspanx=3, minspany=3, spancoords="pixels", interactive=False)
        selector["obj"].set_active(False)
        state["selector_armed"] = False

        def _arm_selector():
            if selector["obj"] is not None:
                selector["obj"].set_active(True)
                state["selector_armed"] = True

        state["arm_selector"] = _arm_selector
```

- [ ] **Step 4: Guard navigation while a drag is armed** — replace the top of `on_key` (the `q/escape` branch) with a version that (a) lets `escape` cancel an armed drag instead of quitting and (b) suspends built-in stepping/playback/save while armed, so the frame the ROI/ellipse is recorded against cannot change mid-drag:

```python
    def on_key(event):
        key = event.key
        if key in ("q", "escape"):
            if state.get("selector_armed"):
                _disarm_selector()
                _refresh()
                return
            plt.close(fig); return

        if state.get("selector_armed"):
            # A drag is being set up: do NOT navigate/advance the frame. Route the
            # key to the tool (e.g. to re-arm a different ROI) then redraw.
            if cfg.on_key_extra(event, state):
                _refresh()
            return

        step = 0
        ...  # (rest of on_key unchanged: stepping / enter / space / on_key_extra)
```

`state.get("selector_armed")` is `None` (falsy) for `click_anchor` — which never sets `on_selector` — so both new branches are dead code on that path.

- [ ] **Step 5: Verify (headless + suite)**

Run the full suite (the strongest regression check — `click_anchor` and `tag_trials` anchor logic must be byte-identical):
`<venv> -m pytest tests/video/ tests/test_video_sync_anchor.py tests/test_video_sync_tag_trials.py -q` → **97 + new** passed.

Headless import + new-field check (from worktree root):
`MPLBACKEND=Agg <venv> -c "import sys; sys.path.insert(0,'scripts/video'); import _tagger_ui as u; c=u.ScrubberConfig(video_path='x', ts_ms=[], fps=1.0, n_frames=1, start_frame=0, crop=None, hud_fn=lambda i:'', on_key_extra=lambda e,s:False, on_save=lambda i:None); print('on_selector', c.on_selector, 'on_refresh', c.on_refresh)"`
Expected: `on_selector None on_refresh None` (both default; construction with the 2a positional/keyword shape still works).

- [ ] **Step 6: Commit**

```bash
git add scripts/video/_tagger_ui.py
git commit -m "feat(tagger): optional RectangleSelector + per-refresh overlay seams in run_scrubber (click_anchor unaffected)"
```

---

## Task 5: `tag_session` — ROI capture (`m`/`e`) + live pupil overlay + `f` clamped zoom + HUD

**Files:** Modify `scripts/video/tag_session.py`. No automated GUI test; verified by import/parse + `--help` + the A2 pilot (below). This task adds the ROI/overlay/zoom scaffolding; Task 6 adds the per-frame label keys on top of the same `on_key_extra`/`on_selector` hooks.

**Interfaces:**
- Consumes: Task-1/2/3 library (`video_labels.{new_sidecar,load_sidecar,save_sidecar,set_roi,seed_rois_from_previous,clamp_crop,ellipse_from_detection}`), Task-4 seams (`ScrubberConfig.on_selector`, `.on_refresh`, `state["arm_selector"]`), `tagging.eye_zoom_crop` (Plan 2a, UNCLAMPED), `video_sync.detect_pupil_in_frame(gray, search_roi=(y0,y1,x0,x1)) -> Optional[dict]`, the existing `tag_session` spine (`session`, `subject`, `frame_h`, `frame_w`, `_read_play_frame`, `tag`, `cfg`, `_hud_fn`, `_on_key_extra`, `_draw_current`).
- Produces: ROI state + a redraw of the proposed pupil ellipse whenever the eye ROI is set/updated and on every manual frame change; the eye/mouth ROIs persisted to the sidecar; the `f` zoom view derived from the eye ROI and CLAMPED before indexing.

- [ ] **Step 1: Imports + state.** Add to the `visdetect.analysis.tagging` import list `eye_zoom_crop`; add `detect_pupil_in_frame` to the `visdetect.core.video_sync` import list; add `from visdetect.analysis import video_labels as vl` next to the other `visdetect` imports. After the backend-selection block (so it follows `matplotlib.use`), add `from matplotlib.patches import Ellipse`. Extend `TagSessionState` with:

```python
    mouth_roi: Optional[tuple] = None      # full-frame (y0,y1,x0,x1) or None
    sidecar: Optional[dict] = None         # video_labels sidecar (schema v1)
    zoomed: bool = False                   # f-toggle: eye-zoom vs full frame
    arming: Optional[str] = None           # active drag intent: "eye"|"mouth"|"correct"
    last_proposed: Optional[dict] = None   # last detect_pupil ellipse on the shown frame
    overlay: object = None                 # matplotlib Ellipse artist on ax_frame
```

`tag.eye_roi` already exists (default `None`) and is reused as the full-frame eye box.

- [ ] **Step 2: Build/seed the sidecar in the setup spine.** After `frame_h`/`frame_w` are determined and validated (right before `tag = TagSessionState(...)`), construct the sidecar and adopt any ROIs into state:

```python
    # --- Per-frame label + ROI sidecar (Plan 2b). Decoupled from the anchor JSON.
    sidecar = vl.load_sidecar(session, subject)
    if sidecar is None:
        sidecar = vl.new_sidecar(subj_display, session, (frame_h, frame_w))
        seeded = vl.seed_rois_from_previous(session, subject, (frame_h, frame_w))
        if seeded is not None and seeded["applied"]:
            for name, r in seeded["rois"].items():
                sidecar["rois"][name] = r
            logger.info("Seeded %d ROI(s) from prior session %s (inherited).",
                        len(seeded["rois"]), seeded["source_session"])
        elif seeded is not None:
            logger.warning(
                "Prior session %s frame_size %s != current %s; ROIs offered but "
                "NOT applied (draw fresh with e/m).",
                seeded["source_session"], seeded["frame_size"], [frame_h, frame_w])
        vl.save_sidecar(sidecar, session, subject)
```

Then set `tag = TagSessionState(queue=queue, anchors=seed, sidecar=sidecar)` and adopt ROIs after construction:

```python
    _eye = sidecar["rois"].get("eye")
    tag.eye_roi = tuple(_eye["box"]) if _eye else None
    _mouth = sidecar["rois"].get("mouth")
    tag.mouth_roi = tuple(_mouth["box"]) if _mouth else None
```

- [ ] **Step 3: Detection + overlay helpers.** Add near the other closures in `main()` (they use the existing `_read_play_frame(fi)` full-frame reader and `tag.fig`):

```python
    def _run_detect(fi: int):
        """Detect the pupil in the eye ROI on frame *fi*; cache the proposal."""
        if tag.eye_roi is None:
            tag.last_proposed = None
            return
        gray = _read_play_frame(fi)                       # full-frame grayscale
        det = detect_pupil_in_frame(gray, search_roi=tag.eye_roi)
        tag.last_proposed = vl.ellipse_from_detection(det)  # {cx,cy,major,minor,angle}|None

    def _update_overlay():
        """Draw/refresh the proposed-ellipse patch on the scrubber's frame axis.
        Shown only in the FULL-FRAME view (ellipse coords are full-frame pixels)."""
        fig = tag.fig
        if fig is None or not fig.axes:
            return
        ax = fig.axes[0]
        ell = tag.last_proposed
        show = (ell is not None) and (not tag.zoomed)
        if tag.overlay is None:
            tag.overlay = Ellipse((0.0, 0.0), 1.0, 1.0, angle=0.0, fill=False,
                                  edgecolor="#00ff00", linewidth=1.5)
            ax.add_patch(tag.overlay)
        if show:
            tag.overlay.set_center((ell["cx"], ell["cy"]))
            tag.overlay.width = ell["major"]
            tag.overlay.height = ell["minor"]
            tag.overlay.angle = ell["angle"]
        tag.overlay.set_visible(show)

    def _on_frame_shown(fi: int, fig) -> None:
        """cfg.on_refresh hook: re-detect + redraw the overlay on every manual
        frame change (arrow step / jump / mode toggle / ROI draw)."""
        tag.fig = fig
        _run_detect(fi)
        _update_overlay()
```

Update the existing playback `_draw_current(fi)` so the overlay is hidden during playback (no per-frame detection while streaming): after its `hud.set_text(...)` line, add `if tag.overlay is not None: tag.overlay.set_visible(False)`. The overlay reappears on the next manual step via `_on_frame_shown`.

- [ ] **Step 4: `f` clamped zoom.** Add a zoom toggle that mutates the (mutable-dataclass) `cfg.crop` the scrubber reads each `_read_frame`. CLAMP the unclamped `eye_zoom_crop` before it can index a frame:

```python
    def _toggle_zoom() -> bool:
        if tag.eye_roi is None:
            logger.warning("No eye ROI yet; draw one with 'e' before zooming.")
            return True
        if not tag.zoomed:
            raw = eye_zoom_crop(tag.eye_roi)              # UNCLAMPED (y0,y1,x0,x1)
            crop = vl.clamp_crop(raw, frame_h, frame_w)  # None if it misses the frame
            if crop is None:                             # no valid crop -> stay full-frame
                logger.warning("Eye ROI does not intersect the frame; staying on "
                               "the full view (no zoom).")
                return True                              # do NOT toggle into a broken zoom
            tag.zoomed = True
            cfg.crop = crop                              # guaranteed non-empty crop
        else:
            tag.zoomed = False
            cfg.crop = None                              # back to full frame
        return True
```

`clamp_crop` returns `None` (not a degenerate crop) when the padded eye box does
not intersect the frame; treat that as "stay on the full frame" so `f` never
zooms onto an empty array.

(`cfg` is assigned later in `main()` but only referenced when `_toggle_zoom` runs during `run_scrubber`, so the closure resolves it at call time — the same late-binding pattern the file already uses for `_hud_fn`/`cfg`.)

- [ ] **Step 5: ROI drag callback.** Add the `on_selector` handler. Re-drawn ROIs are `source="drawn"` (design §4.1); it persists atomically on every change and refreshes the overlay for the eye:

```python
    def _on_roi_drawn(box, state) -> None:
        """cfg.on_selector: a completed drag sets the armed ROI (eye/mouth) or,
        in Task 6, a pupil correction. Boxes are full-frame (y0,y1,x0,x1)."""
        if tag.arming == "eye":
            tag.eye_roi = tuple(box)
            vl.set_roi(tag.sidecar, "eye", box, source="drawn")
            vl.save_sidecar(tag.sidecar, session, subject)
            _run_detect(state["frame_idx"])              # immediate live feedback
            _update_overlay()
        elif tag.arming == "mouth":
            tag.mouth_roi = tuple(box)
            vl.set_roi(tag.sidecar, "mouth", box, source="drawn")
            vl.save_sidecar(tag.sidecar, session, subject)
        # (Task 6 adds the tag.arming == "correct" branch here.)
        tag.arming = None
```

- [ ] **Step 6: Key handlers.** In `_on_key_extra`, add these branches (before the final `logger.debug("unhandled key", ...)` fallthrough). ROIs are only drawable in the full-frame view:

```python
        if key == "e":
            if tag.zoomed:
                logger.warning("Return to full frame (press f) before drawing an ROI.")
                return True
            tag.arming = "eye"
            state["arm_selector"]()
            return True
        if key == "m":
            if tag.zoomed:
                logger.warning("Return to full frame (press f) before drawing an ROI.")
                return True
            tag.arming = "mouth"
            state["arm_selector"]()
            return True
        if key == "f":
            return _toggle_zoom()
```

- [ ] **Step 7: HUD ROI line + wire the hooks into `cfg`.** In `_hud_fn`, add an ROI-status line (before the legend), and extend the legend with the new keys:

```python
        eye_state = "set" if tag.eye_roi is not None else "none"
        mouth_state = "set" if tag.mouth_roi is not None else "none"
        view = "ZOOM" if tag.zoomed else "full"
        roi_line = f"ROI: eye[{eye_state}] mouth[{mouth_state}]   view: {view}"
```

Add `roi_line` to the returned `"\n".join([...])` (between the anchors line and the legend), and change the legend string to include the new keys, e.g.:
`"[space]play [-/+]spd {:g}x [j/k]jump [c]base<->chg [e/m]roi [f]zoom [enter]save [d]del [q]quit"`.

Finally set the two new `ScrubberConfig` args where `cfg` is built:

```python
    cfg = ScrubberConfig(
        video_path=video_path, ts_ms=ts_ms, fps=fps, n_frames=n_frames,
        start_frame=start_frame, crop=None,
        hud_fn=_hud_fn, on_key_extra=_on_key_extra, on_save=_do_save,
        on_selector=_on_roi_drawn,          # Task 5
        on_refresh=_on_frame_shown,         # Task 5
    )
```

- [ ] **Step 8: Verify (headless + manual)**

- `MPLBACKEND=Agg <venv> scripts/video/tag_session.py --help` → exit 0, shows `--session`/`--subject`/`--no-stage`.
- `MPLBACKEND=Agg <venv> -c "import ast; ast.parse(open('scripts/video/tag_session.py').read()); print('parse ok')"` → `parse ok`.
- Full suite green: `<venv> -m pytest tests/video/ tests/test_video_sync_anchor.py tests/test_video_sync_tag_trials.py -q` → **97 + new** passed (the backend-invariant test in `tests/video/test_tagger_backend.py` still passes — the new imports stay AFTER the TkAgg selection).
- **Interactive acceptance = the A2 pilot** (final section): draw eye + mouth ROI, watch the live green pupil ellipse appear/track, toggle `f` zoom, re-open and confirm ROIs persisted + inherited on a 2nd session.

- [ ] **Step 9: Commit**

```bash
git add scripts/video/tag_session.py
git commit -m "feat(tagger): ROI capture (m/e) + live pupil overlay + clamped f-zoom + HUD (Plan 2b)"
```

---

## Task 6: `tag_session` — per-frame label capture (`u`/`p`/`x`) + atomic persistence + HUD counts

**Files:** Modify `scripts/video/tag_session.py`. No automated GUI test; verified by import/parse + the A2 pilot.

**Interfaces:**
- Consumes: Task-5 scaffolding (`tag.sidecar`, `tag.last_proposed`, `tag.arming`, `_on_roi_drawn`, `_on_key_extra`, `_hud_fn`), `video_labels.{upsert_frame_label,ellipse_from_box,save_sidecar,VERDICT_CONFIRMED,VERDICT_CORRECTED,VERDICT_BLINK}`, Task-4 `state["arm_selector"]`.
- Produces: `confirm`/`correct`/`blink` labels written to the sidecar (frame-keyed upsert, atomic), a correction storing BOTH proposed and human ellipse, a final flush on quit, and HUD label counts.

- [ ] **Step 1: Label keys.** In `_on_key_extra`, add these branches alongside the `e`/`m`/`f` branches from Task 5 (before the `logger.debug` fallthrough):

```python
        if key == "u":  # proposed ellipse is CORRECT
            if tag.last_proposed is None:
                logger.warning("No proposed ellipse to confirm (set the eye ROI with 'e').")
                return True
            vl.upsert_frame_label(tag.sidecar, state["frame_idx"], vl.VERDICT_CONFIRMED,
                                  proposed_ellipse=tag.last_proposed)
            vl.save_sidecar(tag.sidecar, session, subject)
            logger.info("confirmed frame %d", state["frame_idx"])
            return True
        if key == "p":  # proposed is WRONG -> drag the true pupil
            if tag.zoomed:
                logger.warning("Return to full frame (press f) before correcting.")
                return True
            tag.arming = "correct"
            state["arm_selector"]()
            return True
        if key == "x":  # blink / occluded -> no valid pupil
            vl.upsert_frame_label(tag.sidecar, state["frame_idx"], vl.VERDICT_BLINK,
                                  proposed_ellipse=tag.last_proposed)
            vl.save_sidecar(tag.sidecar, session, subject)
            logger.info("blink frame %d", state["frame_idx"])
            return True
```

- [ ] **Step 2: Correction branch in the drag callback.** Extend `_on_roi_drawn` (Task 5, Step 5) with the `"correct"` case — it stores BOTH the detector's proposal and the human's inscribed ellipse (design §5):

```python
        elif tag.arming == "correct":
            corrected = vl.ellipse_from_box(box)
            vl.upsert_frame_label(tag.sidecar, state["frame_idx"], vl.VERDICT_CORRECTED,
                                  proposed_ellipse=tag.last_proposed,   # may be None
                                  corrected_ellipse=corrected)
            vl.save_sidecar(tag.sidecar, session, subject)
            logger.info("corrected frame %d", state["frame_idx"])
```

(insert immediately after the `elif tag.arming == "mouth":` block and before `tag.arming = None`.)

- [ ] **Step 3: HUD label counts.** In `_hud_fn`, count verdicts from the sidecar and extend the ROI line (or add a labels line):

```python
        _frames = tag.sidecar["frames"] if tag.sidecar else []
        n_conf = sum(1 for f in _frames if f["verdict"] == vl.VERDICT_CONFIRMED)
        n_corr = sum(1 for f in _frames if f["verdict"] == vl.VERDICT_CORRECTED)
        n_blink = sum(1 for f in _frames if f["verdict"] == vl.VERDICT_BLINK)
        label_line = f"labels: {n_conf} ok / {n_corr} fix / {n_blink} blink"
```

Append `label_line` to the returned `"\n".join([...])` (after `roi_line`), and extend the legend with the label keys:
`"... [e/m]roi [f]zoom [u]ok [p]fix [x]blink [enter]save [d]del [q]quit"`.

- [ ] **Step 4: Final flush on quit.** After `run_scrubber(cfg)` returns (in the `finally`/post-run block, alongside `play_cap.release()`), add a belt-and-suspenders flush so the last state is durable even though every change already saved:

```python
        if tag.sidecar is not None:
            vl.save_sidecar(tag.sidecar, session, subject)
```

- [ ] **Step 5: Verify (headless + manual)**

- `MPLBACKEND=Agg <venv> scripts/video/tag_session.py --help` → exit 0.
- `MPLBACKEND=Agg <venv> -c "import ast; ast.parse(open('scripts/video/tag_session.py').read()); print('parse ok')"` → `parse ok`.
- Full suite green: `<venv> -m pytest tests/video/ tests/test_video_sync_anchor.py tests/test_video_sync_tag_trials.py -q` → **97 + new** passed.
- **Interactive acceptance = the A2 pilot.**

- [ ] **Step 6: Commit**

```bash
git add scripts/video/tag_session.py
git commit -m "feat(tagger): per-frame pupil labels (confirm/correct/blink) with atomic sidecar persistence + HUD counts"
```

---

## Final: A2 pilot (interactive acceptance — with the user)

Headless checks cannot validate interactive backend selection or real GUI behavior (the 2a pilot caught a backend bug, a wrong crop, and a misleading legend that every headless check passed over). This layer is unproven until a human drives it.

- [ ] Full suite green: `<venv> -m pytest tests/video/ tests/test_video_sync_anchor.py tests/test_video_sync_tag_trials.py -q`.
- [ ] **Manual, real session (design §9):** `tag_session --subject BG_031 --session 09042025` —
  1. Draw a mouth ROI (`m` then drag) and an eye ROI (`e` then drag); confirm the eye drag immediately overlays a green proposed pupil ellipse; a bad box shows an absent/wrong ellipse → re-drag.
  2. Step to a few baseline/change frames; confirm a few (`u`), correct at least one (`p` + drag the true pupil), flag a blink (`x`).
  3. Toggle `f` eye-zoom (must NOT wrap/garble — the `clamp_crop` guard); return to full frame.
  4. Quit (`q`); re-open the SAME session and confirm the ROIs + labels persisted (sidecar `data/cache/video_labels/BG_031/09042025.json`).
- [ ] **Cross-session seeding proven:** open a SECOND BG_031 session recorded later; confirm its ROIs are pre-loaded from 09042025 and marked `inherited:09042025` in the sidecar; re-drag one and confirm it flips to `source:"drawn"`.

---

## Self-Review (against the design §3–§9)

- **§3 architecture — pure logic in a new library module, GUI only wires it:** `video_labels.py` (Tasks 1–3) is cv2/matplotlib-free and fully unit-tested; `tag_session`/`_tagger_ui` only compose it. ✓
- **§4 ROI capture — `m`/`e` arm a RectangleSelector on the full-frame view; boxes stored in full-frame pixels; setting eye box runs `detect_pupil_in_frame` and overlays the proposal; free at any time:** Task 4 seam + Task 5 `_on_roi_drawn`/`_run_detect`/`_update_overlay`; ROI keys refuse while zoomed. ✓
- **§4.1 seeding + provenance + `frame_size` guard + canonical chronology:** Task 2 `seed_rois_from_previous` (date-based, `inherited:<session>`, `applied` flag on frame-size match) + Task 5 setup-spine wiring. ✓
- **§5 labels confirm/correct/blink; correction stores BOTH ellipses; `p` reuses the ROI drag; axis-aligned inscribed ellipse:** Task 6 `u`/`p`/`x` + Task 3 `ellipse_from_box` + `upsert_frame_label`. ✓
- **§6 sidecar schema (v1, keys, frames keyed on frame_idx, atomic write, decoupled path):** Task 1 `new_sidecar`/`save_sidecar`/`upsert_frame_label` + `data/cache/video_labels/<subject>/<session>.json`. ✓
- **§7 zoom restored, derived from eye ROI, CLAMPED before indexing:** Task 3 `clamp_crop` + Task 5 `_toggle_zoom` (`clamp_crop(eye_zoom_crop(...))`); `clamp_crop` returns `None` when the padded box misses the frame, and `_toggle_zoom` treats `None` as "stay on the full frame" so it never zooms onto an empty array. ✓
- **§7.1 risks — matplotlib keymap collisions (`p`/`f` already cleared, load-bearing); selector must not fight the scrubber; scrubber gains an OPTIONAL mouse hook that leaves `click_anchor` unaffected:** Global Constraints note + Task 4 selector-armed guard + defaulted `on_selector`/`on_refresh`. ✓
- **§8 testing — pure logic unit-tested; GUI = import/parse + `--help` + human pilot:** Tasks 1–3 TDD; Tasks 4–6 headless checks + A2 pilot. ✓
- **§9 acceptance — suite green + interactive pass + cross-session seeding:** Final A2 pilot section. ✓
- **§10 out of scope (ME/pupil extraction, front-cam ROIs, PNG export, time-ranged ROIs, reject tag, Plan 2c) — intentionally not implemented:** flagged so the coverage gaps are deliberate. ✓

**Placeholder scan:** Tasks 1–3 carry complete test + implementation code; Tasks 4–6 carry complete code for every added block plus exact slot-in locations and headless verification commands, with interactive acceptance deferred to the A2 pilot (GUI cannot be unit-tested — per §8). No "TBD"/"add error handling"/"similar to Task N" placeholders.

**Type/name consistency:** `SCHEMA_VERSION`, `VERDICT_CONFIRMED/CORRECTED/BLINK`, `new_sidecar`, `load_sidecar`, `save_sidecar`, `set_roi`, `upsert_frame_label`, `seed_rois_from_previous`, `clamp_crop`, `ellipse_from_box`, `ellipse_from_detection`, `on_selector`, `on_refresh`, `state["arm_selector"]`, `tag.eye_roi`/`mouth_roi`/`sidecar`/`zoomed`/`arming`/`last_proposed`/`overlay` are used identically across every task and match the §6 JSON keys (`schema_version`, `subject`, `session`, `camera`, `frame_size`, `rois.{name}.{box,source}`, `frames[].{frame_idx,verdict,proposed_ellipse,corrected_ellipse,labeled_at}`).

> **Known divergence from the spec's assumed detector shape (carried into Task 3):** the design's `proposed_ellipse: {cx,cy,major,minor,angle}` assumed `detect_pupil_in_frame` returns a full ellipse. It actually returns `{center_y, center_x, radius, area, circularity, bbox}` — `radius = max(axes)/2`, with the minor axis and rotation NOT exposed. `ellipse_from_detection` therefore stores the proposal as a CIRCLE of diameter `2*radius` (`angle=0`), which preserves the scientifically-critical major diameter (the eyelid-occlusion too-small-diameter bias is measured against it) while collapsing the unavailable minor axis. Human `correct` ellipses (from `ellipse_from_box`) still carry two distinct axes. No change to the reused detector (design §3: "reused, not rebuilt").
