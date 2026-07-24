# Unified Tagger — Sync-Critical Core (Plan 2a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the sync-critical core of the unified single-pass tagger — the testable logic (big-change queue, v3 emission, migration/seed, orientation-aware frame mapping, per-session eye-zoom crop) plus the GUI shell (`tag_session.py`) that composes a refactored shared scrubber — so a user can place validated multi-anchor syncs (baseline **and** change) on any subject through one window. ROI/pupil/label capture is Plan 2b; the validation-figure generalization + front-cam derivation is Plan 2c.

**Architecture:** Extract the pure logic into a new testable module `src/visdetect/analysis/tagging.py`; add a v3 top-level anchor builder to `video_sync.py` (the v3 entry builders + migration already exist from Plan 1); extract the shared scrubber/HUD from `click_anchor.py` into `scripts/video/_tagger_ui.py`; build `scripts/video/tag_session.py` as the thin unified entry point that composes them. The legacy `click_anchor --scrub` and `tag_trials` keep working throughout.

**Tech stack:** Python 3.10 (`.venv`, run via the venv python), matplotlib TkAgg, cv2, numpy, pytest. Windows + git-bash.

## Global Constraints

- **Clock orientation is `detection_method`-dependent.** `manual_multianchor`/`derivative` store `nidaq = slope*cam + offset` (what `camera_to_nidaq`/`nidaq_to_camera` assume). Legacy `manual_slope_fit` (2-anchor) stores the inverse `video = slope*nidaq + offset`. Any nidaq→frame mapping MUST branch on `detection_method`.
- **Canonical session ids** via `config.canonical_camera_session()` (handles 6-digit `DDMMYY` for BG_031/039) — never raw `str(int(x)).zfill(8)`.
- **Subject-namespaced** sync dir via `config.subject_video_sync_dir(subject)`; pass `sync_dir=` to library IO.
- **Zero writes to X:.** Read camera frames from a locally-staged copy (`stage_session_video`) for responsive playback; never write CAMERA_ROOT.
- **Alignment safety:** change queue uses `get_event_times_by_trial(sess, "Change_ON")` (enforces `EVENT_VALID_OUTCOMES["Change_ON"] = {hit,miss}`); never `get_event_times` (drops NaN, destroys trial alignment). Big changes = `BIG_CHANGE_SIZES = {2.0, 4.0}`.
- **Never silent-overwrite:** the tagger archives prior anchor+sync (`archive_sync_artifacts(..., include_anchor=True)`) and seeds legacy anchors as editable.
- **Keybindings:** `space`=play/pause, `enter`=save, `j`/`k`=jump, `c`=baseline⇄change, `f`=full/zoom, `d`=delete (per the design). Subagent model: Opus 4.8.

**Design:** `docs/superpowers/specs/2026-07-23-camera-tagger-ux-design.md`. Backbone spec: `docs/superpowers/specs/2026-07-21-camera-baseline-sync-multisubject-design.md`.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/visdetect/analysis/tagging.py` | create | Pure logic: `ChangeTarget`, `build_change_queue`, `seed_from_archive`, `eye_zoom_crop`, `nidaq_to_frame_oriented` |
| `src/visdetect/core/video_sync.py` | modify | `_build_v3_anchor_file` (schema 3 + baseline `event_type`); `_build_or_merge_anchor_file` writes v3 |
| `scripts/video/tag_trials.py` | modify | guard accepts `manual_multianchor`; `_slope_fit_frame` orientation-aware |
| `scripts/video/_tagger_ui.py` | create | Shared scrubber primitive (frame I/O + 2-row figure + key loop + hooks) |
| `scripts/video/click_anchor.py` | modify | `_run_scrub` delegates to `_tagger_ui.run_scrubber`; full-frame/per-session crop |
| `scripts/video/tag_session.py` | create | Unified entry point: setup spine + change queue + modes + migration/seed |
| `tests/video/test_tagging_logic.py` | create | Tasks 1–5 logic |

---

## Task 1: Big-change queue

**Files:** Create `src/visdetect/analysis/tagging.py`; Test `tests/video/test_tagging_logic.py`

**Interfaces:**
- Consumes: `visdetect.analysis.align.get_event_times_by_trial(session, event_name, enforce_valid_outcomes=True) -> List[float]` (len n_trials, NaN off hit/miss); `constants.BIG_CHANGE_SIZES = {2.0, 4.0}`; `sess.trials[i].change_size` / `.trialoutcome`.
- Produces: `@dataclass ChangeTarget(trial_index:int, change_on_s:float, change_size:float, outcome:str)`; `build_change_queue(sess) -> List[ChangeTarget]` (size-4 first, then size-2, trial order within).

- [ ] **Step 1: Write the failing test**

```python
# tests/video/test_tagging_logic.py
import numpy as np
import pytest
from visdetect.analysis import tagging


class _Trial:
    def __init__(self, change_size, outcome):
        self.change_size = change_size
        self.trialoutcome = outcome


class _Sess:
    def __init__(self, trials):
        self.trials = trials


def test_build_change_queue_orders_size4_then_size2_hitmiss_only(monkeypatch):
    trials = [
        _Trial(4.0, "hit"),    # 0 -> keep, size4
        _Trial(2.0, "miss"),   # 1 -> keep, size2
        _Trial(1.25, "hit"),   # 2 -> drop (small change)
        _Trial(4.0, "miss"),   # 3 -> keep, size4
        _Trial(4.0, "fa"),     # 4 -> dropped by getter (NaN, not hit/miss)
        _Trial(1.0, "miss"),   # 5 -> catch, drop
    ]
    sess = _Sess(trials)
    # getter returns absolute Change_ON s per trial, NaN off hit/miss (idx 4 fa, idx 5 catch treated hit/miss but small)
    fake = [10.0, 20.0, 30.0, 40.0, float("nan"), 60.0]
    monkeypatch.setattr(tagging, "get_event_times_by_trial", lambda s, e: fake)
    q = tagging.build_change_queue(sess)
    assert [t.trial_index for t in q] == [0, 3, 1]      # size4 (0,3) then size2 (1)
    assert [t.change_size for t in q] == [4.0, 4.0, 2.0]
    assert q[0].change_on_s == 10.0 and q[0].outcome == "hit"
```

- [ ] **Step 2: Run — verify fail**

Run: `PYTHONPATH="<WT>/src" "<venv>/python.exe" -m pytest tests/video/test_tagging_logic.py::test_build_change_queue_orders_size4_then_size2_hitmiss_only -v`
Expected: FAIL — `ModuleNotFoundError: visdetect.analysis.tagging`.

- [ ] **Step 3: Implement**

```python
# src/visdetect/analysis/tagging.py
"""Pure logic for the unified video tagger (no GUI). Testable in isolation."""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from visdetect.analysis.align import get_event_times_by_trial
from visdetect.analysis.constants import BIG_CHANGE_SIZES


@dataclass
class ChangeTarget:
    trial_index: int
    change_on_s: float
    change_size: float
    outcome: str


def build_change_queue(sess) -> List[ChangeTarget]:
    """Ordered change-anchor targets: big changes (size-4 first, then size-2),
    hit/miss go-trials only, trial order within a size. Uses the trial-indexed,
    outcome-safe Change_ON getter."""
    change_on = get_event_times_by_trial(sess, "Change_ON")
    out: List[ChangeTarget] = []
    for idx, t_on in enumerate(change_on):
        if t_on is None or np.isnan(float(t_on)):
            continue
        tr = sess.trials[idx]
        cs = tr.change_size
        if cs is None or float(cs) not in BIG_CHANGE_SIZES:
            continue
        out.append(ChangeTarget(int(idx), float(t_on), float(cs),
                                str(tr.trialoutcome).lower()))
    out.sort(key=lambda t: (0 if t.change_size == 4.0 else 1, t.trial_index))
    return out
```

- [ ] **Step 4: Run — verify pass**

Run: same command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/tagging.py tests/video/test_tagging_logic.py
git commit -m "feat(tagger): big-change queue (size-4 then size-2, hit/miss) for change anchoring"
```

---

## Task 2: v3 anchor emission in the tagger

**Files:** Modify `src/visdetect/core/video_sync.py` (add `_build_v3_anchor_file`, route `_build_or_merge_anchor_file`); the tagger writer will call it. Test `tests/video/test_tagging_logic.py`.

**Interfaces:**
- Consumes existing (Plan 1): `_build_anchor_entry` (baseline, no event_type), `_build_change_anchor_entry`, `_merge_anchor_into_file` (keyed on `(trial_index, event_type)`), `_migrate_anchor_to_v3`.
- Produces: `_build_v3_anchor_file(session, fps, n_trials, anchor_entries) -> dict` (top-level `schema_version=3`; every entry carries `event_type`, baseline entries get `event_type="baseline_on"` + `nidaq_event_s`).

- [ ] **Step 1: Write the failing test**

```python
# tests/video/test_tagging_logic.py  (append)
import numpy as np
from visdetect.core import video_sync as vs


def test_build_v3_anchor_file_stamps_v3_and_baseline_event_type():
    ts_ms = np.arange(1000, dtype=float) * 20.0
    base = vs._build_anchor_entry(np.array([5.0, 9.0]), ts_ms, trial_index=0, frame_idx=100)
    f = vs._build_v3_anchor_file("01072025", fps=50.0, n_trials=2, anchor_entries=[base])
    assert f["schema_version"] == 3
    a = f["anchors"][0]
    assert a["event_type"] == "baseline_on"
    assert a["nidaq_event_s"] == a["nidaq_baseline_on_s"]
```

- [ ] **Step 2: Run — verify fail** (`AttributeError: _build_v3_anchor_file`).

- [ ] **Step 3: Implement** — add after `_build_v2_anchor_file` (video_sync.py ~2823):

```python
def _build_v3_anchor_file(session_name, fps, n_trials, anchor_entries) -> dict:
    """Top-level v3 anchor dict: schema_version 3; every entry carries event_type.
    Baseline entries (lacking event_type) get event_type='baseline_on' + nidaq_event_s."""
    entries = []
    for a in anchor_entries:
        a = dict(a)
        a.setdefault("event_type", "baseline_on")
        if "nidaq_event_s" not in a and "nidaq_baseline_on_s" in a:
            a["nidaq_event_s"] = float(a["nidaq_baseline_on_s"])
        entries.append(a)
    return {
        "session": str(session_name),
        "schema_version": 3,
        "frame_rate_fps": float(fps),
        "n_trials": int(n_trials),
        "anchors": list(entries),
    }
```

Then route `_build_or_merge_anchor_file` in `scripts/video/click_anchor.py:113-141`: change the `existing is None` branch to call `_build_v3_anchor_file` (import it), and import it at the top with the other `video_sync` names. (The merge branch already produces v3-compatible dicts via `_merge_anchor_into_file`.) Verify the existing anchor tests (`tests/test_video_sync_anchor.py`) still pass.

- [ ] **Step 4: Run — verify pass** (new test + `tests/test_video_sync_anchor.py` green).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/core/video_sync.py scripts/video/click_anchor.py tests/video/test_tagging_logic.py
git commit -m "feat(tagger): emit v3 anchor schema (baseline event_type) from the tagger writer"
```

---

## Task 3: Migration / seed-from-archive

**Files:** Modify `src/visdetect/analysis/tagging.py`; Test `tests/video/test_tagging_logic.py`.

**Interfaces:**
- Consumes: `video_sync.archive_sync_artifacts(session, subject, sync_dir, include_anchor=True)`, `video_sync.load_anchor(session, sync_dir=)`, `config.subject_video_sync_dir`.
- Produces: `seed_from_archive(session, subject=None, sync_dir=None) -> Optional[dict]` — archives prior artifacts, returns the archived anchor file with each entry marked `source="legacy"` (or None if nothing to seed).

- [ ] **Step 1: Write the failing test**

```python
# tests/video/test_tagging_logic.py  (append)
import json, os
from visdetect.analysis import tagging as tg
from visdetect.core import video_sync as vs


def test_seed_from_archive_archives_and_marks_legacy(tmp_path):
    d = tmp_path
    anchor = {"session": "01072025", "schema_version": 3, "frame_rate_fps": 50.0,
              "n_trials": 2, "anchors": [
                  {"trial_index": 0, "event_type": "baseline_on", "nidaq_event_s": 5.0,
                   "nidaq_baseline_on_s": 5.0, "video_frame_idx": 100, "video_time_s": 2.0,
                   "clicked_at": "x"}]}
    (d / "01072025_anchor.json").write_text(json.dumps(anchor))
    (d / "01072025_video_sync.json").write_text(json.dumps({"session_name": "01072025"}))
    seeded = tg.seed_from_archive("01072025", sync_dir=str(d))
    # prior files archived (not in live dir)
    assert not (d / "01072025_anchor.json").exists()
    # seed returned, legacy-marked
    assert seeded is not None
    assert seeded["anchors"][0]["source"] == "legacy"


def test_seed_from_archive_none_when_empty(tmp_path):
    assert tg.seed_from_archive("01072025", sync_dir=str(tmp_path)) is None
```

- [ ] **Step 2: Run — verify fail** (`AttributeError: seed_from_archive`).

- [ ] **Step 3: Implement** — append to `tagging.py`:

```python
import os
from typing import Optional
from visdetect.core import video_sync as _vs
from visdetect.analysis.config import subject_video_sync_dir, canonical_camera_session


def seed_from_archive(session_name, subject: Optional[str] = None,
                      sync_dir: Optional[str] = None) -> Optional[dict]:
    """Archive any prior anchor+sync (§3.14 migration), then return the archived
    anchor file with every entry marked source='legacy' as editable seeds. None if
    there was nothing to seed."""
    out_dir = sync_dir or subject_video_sync_dir(subject)
    sn = canonical_camera_session(session_name)
    arch = _vs.archive_sync_artifacts(session_name, subject=subject,
                                      sync_dir=out_dir, include_anchor=True)
    if arch is None:
        return None
    archived_anchor = os.path.join(arch, f"{sn}_anchor.json")
    if not os.path.exists(archived_anchor):
        return None
    seeded = _vs.load_anchor(session_name, sync_dir=arch)  # migrates to v3 in memory
    if seeded is None:
        return None
    for a in seeded["anchors"]:
        a["source"] = "legacy"
    return seeded
```

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/tagging.py tests/video/test_tagging_logic.py
git commit -m "feat(tagger): seed-from-archive migration (archive prior, pre-load legacy anchors editable)"
```

---

## Task 4: Orientation-aware nidaq→frame + tag_trials guard

**Files:** Modify `src/visdetect/analysis/tagging.py` (add `nidaq_to_frame_oriented`); `scripts/video/tag_trials.py` (guard + `_slope_fit_frame`). Test `tests/video/test_tagging_logic.py`.

**Interfaces:**
- Produces: `nidaq_to_frame_oriented(nidaq_s, slope, offset, fps, detection_method) -> int` — `manual_slope_fit` → `video = slope*nidaq + offset`; else (`manual_multianchor`/`derivative`) → `video = (nidaq - offset)/slope`; returns `round(video*fps)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/video/test_tagging_logic.py  (append)
from visdetect.analysis import tagging as tg


def test_nidaq_to_frame_orientation_branches():
    # manual_slope_fit: video = slope*nidaq + offset
    f1 = tg.nidaq_to_frame_oriented(10.0, slope=1.0, offset=2.0, fps=50.0,
                                    detection_method="manual_slope_fit")
    assert f1 == round((1.0 * 10.0 + 2.0) * 50.0)   # 600
    # manual_multianchor: video = (nidaq - offset)/slope
    f2 = tg.nidaq_to_frame_oriented(12.0, slope=1.0, offset=2.0, fps=50.0,
                                    detection_method="manual_multianchor")
    assert f2 == round(((12.0 - 2.0) / 1.0) * 50.0)  # 500
```

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Implement** — in `tagging.py`:

```python
def nidaq_to_frame_oriented(nidaq_s: float, slope: float, offset: float,
                            fps: float, detection_method: str) -> int:
    """NI time -> video frame, respecting the detection_method-dependent clock
    orientation (see Global Constraints)."""
    if detection_method == "manual_slope_fit":
        video_time_s = slope * float(nidaq_s) + offset          # inverse-orientation legacy
    else:
        video_time_s = (float(nidaq_s) - offset) / slope        # camera_to_nidaq orientation
    return int(round(video_time_s * fps))
```

Then fix `scripts/video/tag_trials.py`:
- Guard (tag_trials.py:373-381): change `if method != "manual_slope_fit":` to `if method not in ("manual_slope_fit", "manual_multianchor"):` (and update the message).
- `_slope_fit_frame` (tag_trials.py:174-181): replace its body to delegate to the oriented mapper —

```python
def _slope_fit_frame(sync_json: dict, nidaq_s: float, fps: float) -> int:
    from visdetect.analysis.tagging import nidaq_to_frame_oriented
    eye = sync_json["eye_cam"]
    return nidaq_to_frame_oriented(
        nidaq_s, float(eye["slope"]), float(eye["offset"]), fps,
        eye.get("detection_method", "manual_slope_fit"))
```

Verify `tests/test_video_sync_tag_trials.py` still passes (its fixtures use `manual_slope_fit` → same result as before).

- [ ] **Step 4: Run — verify pass** (new test + `tests/test_video_sync_tag_trials.py`).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/tagging.py scripts/video/tag_trials.py tests/video/test_tagging_logic.py
git commit -m "fix(tagger): orientation-aware nidaq->frame + tag_trials accepts manual_multianchor"
```

---

## Task 5: Per-session eye-zoom crop

**Files:** Modify `src/visdetect/analysis/tagging.py`; Test `tests/video/test_tagging_logic.py`.

**Interfaces:**
- Produces: `eye_zoom_crop(eye_roi, pad=0.15, fallback=(200,420,320,540)) -> Tuple[int,int,int,int]` — `(y0,y1,x0,x1)` from an eye ROI `[y0,y1,x0,x1]` box padded by `pad` fraction; returns `fallback` when `eye_roi` is None.

- [ ] **Step 1: Write the failing test**

```python
# tests/video/test_tagging_logic.py  (append)
from visdetect.analysis import tagging as tg


def test_eye_zoom_crop_from_roi_and_fallback():
    assert tg.eye_zoom_crop(None) == (200, 420, 320, 540)
    y0, y1, x0, x1 = tg.eye_zoom_crop([300, 400, 500, 600], pad=0.0)
    assert (y0, y1, x0, x1) == (300, 400, 500, 600)
    yy0, yy1, xx0, xx1 = tg.eye_zoom_crop([300, 400, 500, 600], pad=0.10)
    assert yy0 == 290 and yy1 == 410 and xx0 == 490 and xx1 == 610  # 10% of 100 each side
```

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Implement** — in `tagging.py`:

```python
def eye_zoom_crop(eye_roi, pad: float = 0.15,
                  fallback: Tuple[int, int, int, int] = (200, 420, 320, 540)
                  ) -> Tuple[int, int, int, int]:
    """(y0,y1,x0,x1) eye-zoom crop from an eye ROI box (padded), else the fallback."""
    if eye_roi is None:
        return fallback
    y0, y1, x0, x1 = [int(v) for v in eye_roi]
    dy, dx = int(round((y1 - y0) * pad)), int(round((x1 - x0) * pad))
    return (y0 - dy, y1 + dy, x0 - dx, x1 + dx)
```

- [ ] **Step 4: Run — verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/tagging.py tests/video/test_tagging_logic.py
git commit -m "feat(tagger): per-session eye-zoom crop derived from the eye ROI"
```

---

## Task 6: Extract the shared scrubber UI primitive

**Files:** Create `scripts/video/_tagger_ui.py`; Modify `scripts/video/click_anchor.py` (`_run_scrub` delegates to it). No new automated test (GUI); verified by existing tests + a headless import/parse check + manual scrub.

**Interfaces:**
- Produces: `run_scrubber(cfg)` where `cfg` supplies: `video_path, ts_ms, fps, n_frames, start_frame`, a `crop` `(y0,y1,x0,x1)` **or None** (None = full frame), a `hud_fn(frame_idx) -> str` (what to print), an `on_key_extra(event, state) -> bool` hook (handles tool-specific keys; returns True if it consumed the event), and `on_save(frame_idx) -> Optional[dict]` (what `enter` does). It owns: the 2-row figure (`ax_frame` imshow + `ax_hud` text), the `state = {"frame_idx", "result"}` dict, the cv2 capture + `_read_frame` (applies `crop` if not None), the arrow-step keys (±1/±10/±100), and `q`/`esc`. Returns `state["result"]`.

**Design note:** this is a refactor of `click_anchor._run_scrub` (:354-586) whose figure/frame-I/O/`state` are already byte-identical to `tag_trials._run_tag_ui`. Move that common core into `run_scrubber`; the divergent bits (`_refresh` text, `j`/`k`/`space`/`enter` semantics) become the `hud_fn`/`on_key_extra`/`on_save` hooks. **Do not change behavior** of the existing `--scrub` flow — after the refactor, `click_anchor --scrub` must produce identical anchors.

- [ ] **Step 1: Build `run_scrubber`** — lift the frame-I/O + figure + arrow/step/quit loop from `_run_scrub` verbatim into `scripts/video/_tagger_ui.py:run_scrubber(cfg)`, replacing the hardcoded `EYE_REGION_CROP_BG046` unpack with `cfg.crop` (skip the slice when `cfg.crop is None`), and replacing `_refresh`'s body with `hud_text.set_text(cfg.hud_fn(state["frame_idx"]))`. Route unmatched keys to `cfg.on_key_extra`; route `enter` to `cfg.on_save`.

- [ ] **Step 2: Rewire `click_anchor._run_scrub`** to build a `cfg` (crop=`EYE_REGION_CROP_BG046`, `hud_fn` = the current `_refresh` text builder, `on_key_extra` handling `j`/`k`/`home`/`end`/`r`, `on_save` = the current space/enter save) and `return run_scrubber(cfg)`. Delete the now-migrated inline loop.

- [ ] **Step 3: Verify no regression**

Run: `PYTHONPATH="<WT>/src" "<venv>/python.exe" -m pytest tests/video/ tests/test_video_sync_anchor.py tests/test_video_sync_tag_trials.py -v` (all green — the anchor-building logic is unchanged).
Run headless import: `... "<venv>/python.exe" -c "import importlib.util as u; s=u.spec_from_file_location('ca','scripts/video/click_anchor.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print('import ok', hasattr(m,'_run_scrub'))"` → `import ok True`.
**Interactive (manual, deferred to pilot):** `click_anchor --scrub --session 09092025` still scrubs + saves identically.

- [ ] **Step 4: Commit**

```bash
git add scripts/video/_tagger_ui.py scripts/video/click_anchor.py
git commit -m "refactor(tagger): extract shared scrubber primitive (_tagger_ui.run_scrubber); click_anchor delegates"
```

---

## Task 7: `tag_session.py` unified entry point

**Files:** Create `scripts/video/tag_session.py`. No automated GUI test; verified by import/parse + `--help` + the A1 pilot.

**Interfaces:**
- Consumes: the Task-1–5 logic (`build_change_queue`, `seed_from_archive`, `eye_zoom_crop`, `nidaq_to_frame_oriented`), `_tagger_ui.run_scrubber` (Task 6), `stage_session_video`, `find_camera_files`, `load_camera_metadata`, `load_session`, the v3 anchor writers, `fit_multianchor_clock` (for live cv_rmse), `config.canonical_camera_session`/`subject_video_sync_dir`.

- [ ] **Step 1: Build the setup spine + session state** — replicate `click_anchor.main`'s common setup (:721-770) but canonical + subject-namespaced: `session = canonical_camera_session(args.session)`; `sync_dir = subject_video_sync_dir(args.subject)`; `seed = seed_from_archive(session, args.subject)` (pre-load legacy); `sess = load_session(session)`; `baseline_on` (filtered/truncated); `stage_session_video(session, args.subject, cams=("eye_cam",))` → local `video_path`; `ts_ms, _, _ = load_camera_metadata(...)`; `fps`, `n_frames`; `queue = build_change_queue(sess)`; a `TagSessionState` holding `mode ∈ {baseline, change}`, `queue_pos`, `anchors` (seeded), and the current best offset model.

- [ ] **Step 2: Compose `run_scrubber`** with a `cfg` whose:
  - `crop`: toggled by `f` between `None` (full frame) and `eye_zoom_crop(state.eye_roi)` — held in state, default fallback.
  - `hud_fn`: renders the design's HUD (subject/session, MODE + change_size/outcome in change mode, trial/frame/Δ, anchor counts, **live `cv_rmse` via `fit_multianchor_clock(state.anchors...)` once ≥3**, key legend).
  - `on_key_extra`: `space`→toggle a playback timer (`fig.canvas.new_timer`, advances frame at `fps*speed`; `[`/`]` change speed); `f`→toggle crop; `c`→toggle `mode` (baseline⇄change) and reseed the jump target; `j`/`k`→advance/retreat (baseline: `jump_to_predicted_frame`; change: step `queue_pos` and jump to `nidaq_to_frame_oriented(queue[pos].change_on_s, <provisional slope/offset>, fps, "manual_multianchor")` — provisional model = coarse offset → first baseline-anchor implied offset → `fit_multianchor_clock` once ≥3); `home`/`end`; `d`→delete current trial's anchor of the current mode.
  - `on_save(frame_idx)`: in baseline mode build a v3 baseline entry (`_build_anchor_entry` + event_type via `_build_v3_anchor_file` path); in change mode build `_build_change_anchor_entry(queue[pos].change_on_s, ts_ms, queue[pos].trial_index, frame_idx, queue[pos].change_size, queue[pos].outcome)`; merge via `_merge_anchor_into_file`; `save_anchor(session, merged, sync_dir=sync_dir)`; update `state.anchors`. Do NOT close the figure on save (multi-anchor: keep tagging).
- Arg parse: `--session` (required), `--subject` (default None→config.SUBJECT), `--no-stage` (skip local staging, read over X:).

- [ ] **Step 3: Verify (headless + manual)**

Run: `... "<venv>/python.exe" scripts/video/tag_session.py --help` → exit 0, shows `--session`/`--subject`.
Run import-parse: `... -c "import ast; ast.parse(open('scripts/video/tag_session.py').read()); print('parse ok')"` → `parse ok`.
**Interactive verification = the A1 pilot** (below): tag a real session end-to-end.

- [ ] **Step 4: Commit**

```bash
git add scripts/video/tag_session.py
git commit -m "feat(tagger): tag_session unified single-pass entry point (baseline + change anchoring, playback, full-frame view)"
```

---

## Final: A1 pilot (interactive acceptance)

- [ ] Full video suite green: `... -m pytest tests/video/ tests/test_video_sync_anchor.py tests/test_video_sync_tag_trials.py tests/test_reconstruct_camera_metadata.py -v`.
- [ ] **Manual (with the user):** `tag_session --subject BG_031 --session 09042025` — place ~3 baseline + ~3 change anchors, watch live `cv_rmse` reach `good`/`review`; then `fit_sync --subject BG_031 --session 09042025` writes a `manual_multianchor` sync JSON. Repeat on one BG_039 + one BG_038 session. This is the sync half of the A1 pilot; the label half arrives with Plan 2b.

---

## Self-Review (against the design)

- **Unified single-pass tool** → Task 7 (`tag_session`) composing Task 6 (shared scrubber). ✓
- **space=play/pause, enter=save; j/k; c baseline⇄change; f full/zoom; d delete** → Task 7 `on_key_extra`/`on_save`. ✓
- **Auto-cycle big changes (size-4 then size-2, hit/miss)** → Task 1 queue + Task 7 change-mode jump. ✓
- **Change-jump seeding (coarse → baseline-implied → multianchor)** → Task 7 provisional model. ✓
- **v3 emission (baseline event_type + change entries)** → Task 2 + Task 7 on_save. ✓
- **Migration/seed-from-archive** → Task 3 + Task 7 setup. ✓
- **Full-frame view + per-session eye-zoom crop** → Task 5 + Task 6 `crop=None` + Task 7 `f` toggle. ✓
- **Orientation-aware frame + tag_trials guard** → Task 4. ✓
- **Live cv_rmse QC** → Task 7 `hud_fn`. ✓
- **Deferred to later plans (intentional):** ROI/pupil/per-frame-label capture (Plan 2b); `sync_validation_figure` generalization + front-cam derivation (Plan 2c). Flagged so coverage gaps are deliberate.

**Placeholder scan:** logic tasks (1–5) carry full code; GUI tasks (6–7) carry precise composition specs + import/parse verification with interactive acceptance in the pilot (GUI can't be unit-tested — per the design's testing section). **Type consistency:** `ChangeTarget`, `build_change_queue`, `seed_from_archive`, `nidaq_to_frame_oriented`, `eye_zoom_crop`, `_build_v3_anchor_file`, `run_scrubber` names are used identically across tasks.
