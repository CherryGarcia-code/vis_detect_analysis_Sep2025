# Video Sync — Anchor-and-Barcode (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-script tool (`scripts/video/click_anchor.py`) that lets the user manually anchor trial 1 Baseline_ON in a session's eye-cam video via a two-stage click UI, then renders a static PNG barcode-montage so the user can verify by eye that the implied clock offset aligns subsequent trials.

**Architecture:** One CLI script + two tiny library helpers in `visdetect.core.video_sync`. The script reads NI-DAQ trial events from the session, picks a predicted trial-1 frame from a cached coarse offset, shows a coarse 5×10 grid (1s sampling, 50s span) followed by a fine 5×10 grid (1 frame/cell, ±500ms span) for the user to click through, saves an anchor JSON, and renders a 5-row × 7-column barcode montage PNG. No automated detection, no thresholds, no regression — pure visual anchor + visual verification.

**Tech Stack:** Python 3, matplotlib TkAgg, OpenCV (`cv2.VideoCapture`), NumPy, `visdetect.suite.loader`, `visdetect.core.video_sync`. Tests use pytest. Run scripts with `py` (Windows + Git Bash).

**Spec:** [`docs/superpowers/specs/2026-05-27-video-sync-anchor-barcode-design.md`](../specs/2026-05-27-video-sync-anchor-barcode-design.md)

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `src/visdetect/core/video_sync.py` | Modify | Add `load_anchor`, `save_anchor`, `compute_predicted_frame_idx` |
| `tests/test_video_sync_anchor.py` | Create | Unit tests for the three new library helpers and the grid-math helpers |
| `scripts/video/click_anchor.py` | Create | CLI tool: two-stage click UI + barcode montage rendering |

**Testing scope (per spec):** Library helpers (`load_anchor`/`save_anchor`/`compute_predicted_frame_idx`) and pure-logic grid helpers (`stage1_frame_indices`, `stage2_frame_indices`, `pixel_to_cell`) **are unit-tested**. Interactive matplotlib code and montage visual quality are **manual smoke-tested** on the 3 anchor sessions.

---

## Task 1 — Library helpers: `load_anchor`, `save_anchor`, `compute_predicted_frame_idx`

**Files:**
- Modify: `src/visdetect/core/video_sync.py` (append helpers near the existing `find_camera_files` / `load_camera_metadata` block; pick a stable insertion point at end of file before the orientation-selective feature block, or at end-of-file — either is fine)
- Create: `tests/test_video_sync_anchor.py`

- [ ] **Step 1.1: Write the failing tests for `save_anchor` + `load_anchor` round-trip and missing file**

Create file `tests/test_video_sync_anchor.py`:

```python
"""Tests for video sync anchor helpers (Phase 1 of corneal-barcode redesign)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from visdetect.core import video_sync as vs


# ---------------------------------------------------------------------------
# load_anchor / save_anchor round trip
# ---------------------------------------------------------------------------


def _make_anchor_dict() -> dict:
    return {
        "session": "TESTSESSION",
        "anchor_trial_index": 0,
        "nidaq_baseline_on_s": 12.3456,
        "video_frame_idx": 1047,
        "video_time_s": 20.94,
        "implied_offset_s": 8.5944,
        "frame_rate_fps": 50.0,
        "n_trials": 350,
        "clicked_at": "2026-05-27T14:32:10",
    }


def test_save_anchor_creates_json_at_expected_path(tmp_path, monkeypatch):
    monkeypatch.setattr(vs, "VIDEO_SYNC_DIR", str(tmp_path))
    anchor = _make_anchor_dict()

    vs.save_anchor("TESTSESSION", anchor)

    expected = tmp_path / "TESTSESSION_anchor.json"
    assert expected.exists()
    payload = json.loads(expected.read_text())
    assert payload == anchor


def test_load_anchor_returns_saved_dict(tmp_path, monkeypatch):
    monkeypatch.setattr(vs, "VIDEO_SYNC_DIR", str(tmp_path))
    anchor = _make_anchor_dict()
    vs.save_anchor("TESTSESSION", anchor)

    loaded = vs.load_anchor("TESTSESSION")

    assert loaded == anchor


def test_load_anchor_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(vs, "VIDEO_SYNC_DIR", str(tmp_path))

    loaded = vs.load_anchor("NONEXISTENT_SESSION")

    assert loaded is None
```

- [ ] **Step 1.2: Run the tests; expect failure**

Run: `py -m pytest tests/test_video_sync_anchor.py -v`
Expected: All three tests FAIL with `AttributeError: module 'visdetect.core.video_sync' has no attribute 'save_anchor'` (or `load_anchor`).

- [ ] **Step 1.3: Implement `save_anchor` and `load_anchor`**

Open `src/visdetect/core/video_sync.py`. Find the existing `from analysis.config import ...` block (it should already include `VIDEO_SYNC_DIR`). If `VIDEO_SYNC_DIR` is not yet imported in this module, locate the existing path constants near the top of the file and add it. Then append at the end of the file (or directly after the orientation-selective feature block added earlier in this branch):

```python
# =====================================================================
# Anchor JSON helpers (Phase 1 of corneal-barcode redesign)
# =====================================================================


def _anchor_path(session_name: str) -> str:
    """Path to the anchor JSON for *session_name*."""
    import os
    return os.path.join(VIDEO_SYNC_DIR, f"{session_name}_anchor.json")


def save_anchor(session_name: str, anchor: dict) -> None:
    """Write *anchor* to ``{VIDEO_SYNC_DIR}/{session_name}_anchor.json``.

    Overwrites any existing file. Creates the directory if needed.
    """
    import json
    import os
    os.makedirs(VIDEO_SYNC_DIR, exist_ok=True)
    with open(_anchor_path(session_name), "w") as f:
        json.dump(anchor, f, indent=2)


def load_anchor(session_name: str) -> dict | None:
    """Read the anchor JSON for *session_name*, or return ``None`` if absent."""
    import json
    import os
    path = _anchor_path(session_name)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)
```

- [ ] **Step 1.4: Run the tests; expect pass**

Run: `py -m pytest tests/test_video_sync_anchor.py -v`
Expected: All three tests PASS.

- [ ] **Step 1.5: Write failing tests for `compute_predicted_frame_idx`**

Append to `tests/test_video_sync_anchor.py`:

```python
# ---------------------------------------------------------------------------
# compute_predicted_frame_idx
# ---------------------------------------------------------------------------


def test_predicted_frame_idx_exact_match():
    ts_ms = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])  # 50 fps
    # baseline_on_s = 0.04 s, coarse_offset_s = 0.0 → predicted = 40 ms → frame 2
    frame_idx = vs.compute_predicted_frame_idx(
        baseline_on_s=0.04, coarse_offset_s=0.0, ts_ms=ts_ms
    )
    assert frame_idx == 2


def test_predicted_frame_idx_with_coarse_offset():
    # 50 fps, 100 frames covering 2 s
    ts_ms = np.arange(0.0, 2000.0, 20.0)
    # baseline_on_s = 5.0 s in NI-DAQ; camera started 4.0 s after NI-DAQ → video time = 1.0 s
    frame_idx = vs.compute_predicted_frame_idx(
        baseline_on_s=5.0, coarse_offset_s=4.0, ts_ms=ts_ms
    )
    # 1.0 s → 1000 ms → frame 50 (since ts_ms[50] == 1000.0)
    assert frame_idx == 50


def test_predicted_frame_idx_chooses_nearest():
    # ts_ms not uniformly spaced; target falls between samples
    ts_ms = np.array([0.0, 100.0, 250.0, 500.0])
    # baseline=0.27, offset=0 → video_ms = 270 → nearest is index 2 (250 ms)
    frame_idx = vs.compute_predicted_frame_idx(
        baseline_on_s=0.27, coarse_offset_s=0.0, ts_ms=ts_ms
    )
    assert frame_idx == 2


def test_predicted_frame_idx_clamps_to_zero_when_negative():
    # baseline before camera start → negative video time → clamp to 0
    ts_ms = np.arange(0.0, 1000.0, 20.0)
    frame_idx = vs.compute_predicted_frame_idx(
        baseline_on_s=0.5, coarse_offset_s=10.0, ts_ms=ts_ms
    )
    assert frame_idx == 0


def test_predicted_frame_idx_clamps_to_last_frame_when_beyond():
    ts_ms = np.arange(0.0, 1000.0, 20.0)  # 50 frames total
    frame_idx = vs.compute_predicted_frame_idx(
        baseline_on_s=100.0, coarse_offset_s=0.0, ts_ms=ts_ms
    )
    assert frame_idx == len(ts_ms) - 1
```

- [ ] **Step 1.6: Run new tests; expect failure**

Run: `py -m pytest tests/test_video_sync_anchor.py -v`
Expected: 5 new tests FAIL with `AttributeError: module 'visdetect.core.video_sync' has no attribute 'compute_predicted_frame_idx'`.

- [ ] **Step 1.7: Implement `compute_predicted_frame_idx`**

Append to `src/visdetect/core/video_sync.py` (right after `load_anchor`):

```python
def compute_predicted_frame_idx(
    baseline_on_s: float,
    coarse_offset_s: float,
    ts_ms: np.ndarray,
) -> int:
    """Map a NI-DAQ Baseline_ON time to the nearest video frame index.

    Parameters
    ----------
    baseline_on_s
        NI-DAQ time of the event, in seconds.
    coarse_offset_s
        Seconds elapsed in NI-DAQ clock before video recording started.
    ts_ms
        Camera-frame timestamps in milliseconds, relative to video start.
        Typically returned by :func:`load_camera_metadata`.

    Returns
    -------
    int
        Index of the closest frame in ``ts_ms``. Clamped to ``[0, len(ts_ms) - 1]``.
    """
    video_ms = (baseline_on_s - coarse_offset_s) * 1000.0
    if video_ms <= ts_ms[0]:
        return 0
    if video_ms >= ts_ms[-1]:
        return int(len(ts_ms) - 1)
    return int(np.argmin(np.abs(ts_ms - video_ms)))
```

- [ ] **Step 1.8: Run all tests; expect pass**

Run: `py -m pytest tests/test_video_sync_anchor.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 1.9: Commit**

```bash
git add src/visdetect/core/video_sync.py tests/test_video_sync_anchor.py
git commit -m "Add anchor JSON helpers + predicted-frame mapper for Phase 1 sync"
```

---

## Task 2 — Script skeleton: grid-math helpers + frame I/O

**Files:**
- Create: `scripts/video/click_anchor.py`
- Modify: `tests/test_video_sync_anchor.py`

- [ ] **Step 2.1: Write failing tests for grid-math helpers**

Append to `tests/test_video_sync_anchor.py`:

```python
# ---------------------------------------------------------------------------
# Grid-math helpers in scripts/video/click_anchor.py
# ---------------------------------------------------------------------------


def _import_click_anchor():
    """Import the script module by file path (it lives outside the import-path)."""
    import importlib.util
    import os
    # tests/test_video_sync_anchor.py → project_root = parent of tests/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec_path = os.path.join(project_root, "scripts", "video", "click_anchor.py")
    spec = importlib.util.spec_from_file_location("click_anchor", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage1_frame_indices_basic():
    ca = _import_click_anchor()
    # 50 fps, video has 5000 frames (100 s of video)
    n_frames = 5000
    # predicted at frame 1000 (20 s into video)
    # stage 1 covers [predicted - 15 s, predicted + 35 s]
    # = frames [250, 2750], 50 cells at 50-frame step → cells span 2500 frames
    idx = ca.stage1_frame_indices(predicted=1000, fps=50.0, n_frames=n_frames)
    assert len(idx) == 50
    assert idx[0] == 250
    assert idx[-1] == 250 + 49 * 50  # 49 steps from start
    # spacing is exactly fps frames (1 s)
    assert all(idx[i + 1] - idx[i] == 50 for i in range(len(idx) - 1))


def test_stage1_frame_indices_clamps_at_start():
    ca = _import_click_anchor()
    # predicted very early → cannot go 15 s back; start clamped to 0
    idx = ca.stage1_frame_indices(predicted=100, fps=50.0, n_frames=5000)
    assert idx[0] == 0


def test_stage1_frame_indices_clamps_at_end():
    ca = _import_click_anchor()
    # predicted very late → last cell clamped to last frame
    idx = ca.stage1_frame_indices(predicted=4990, fps=50.0, n_frames=5000)
    assert idx[-1] <= 4999
    assert len(idx) == 50  # always 50 cells even when clamped


def test_stage2_frame_indices_centered():
    ca = _import_click_anchor()
    # 50 fps, ±25 frames around clicked frame → 50 cells
    idx = ca.stage2_frame_indices(stage1_click=1000, fps=50.0, n_frames=5000)
    assert len(idx) == 50
    assert idx[0] == 1000 - 25
    assert idx[-1] == 1000 + 24


def test_stage2_frame_indices_clamps():
    ca = _import_click_anchor()
    idx = ca.stage2_frame_indices(stage1_click=10, fps=50.0, n_frames=5000)
    assert idx[0] == 0
    assert len(idx) == 50


```

Note: the interactive UI identifies the clicked cell by comparing `event.inaxes` against each subplot axes — no manual pixel→cell math needed.

- [ ] **Step 2.2: Run tests; expect failure**

Run: `py -m pytest tests/test_video_sync_anchor.py -v`
Expected: All grid-math tests FAIL with `ImportError` or `AttributeError`.

- [ ] **Step 2.3: Create the script skeleton with grid-math + frame I/O**

Create `scripts/video/click_anchor.py`:

```python
"""click_anchor.py — Phase 1 of the video-sync anchor-barcode redesign.

Workflow:
  1. Load a session and pick a predicted trial-1 frame from cached coarse offset.
  2. Show a coarse 5×10 grid (1s sampling, 50s span); user clicks the cell where
     the grating first appears in the eye.
  3. Show a fine 5×10 grid (1 frame/cell, ±500ms span) around the stage-1 click;
     user clicks the exact frame.
  4. Save the anchor JSON.
  5. Render a 5-row × 7-column barcode montage PNG so the user can verify the
     implied clock offset visually.

Run:  py scripts/video/click_anchor.py --session 09092025
"""
from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import sys
from typing import Optional, Sequence

import cv2
import numpy as np

# Visdetect imports
from visdetect.suite.loader import load_session
from visdetect.core.video_sync import (
    find_camera_files,
    load_camera_metadata,
    compute_predicted_frame_idx,
    load_anchor,
    save_anchor,
)
from visdetect.analysis.config import VIDEO_SYNC_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("click_anchor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EYE_REGION_CROP_BG046 = (200, 420, 320, 540)  # y0, y1, x0, x1

# Both stages use the same 5×10 grid; only the frame indices differ.
GRID_ROWS = 5
GRID_COLS = 10
N_CELLS = GRID_ROWS * GRID_COLS  # 50

# Stage 1: coarse window, 1 s sampling, 50 s span, biased forward.
STAGE1_PRE_S = 15.0   # seconds before predicted onset
STAGE1_POST_S = 35.0  # seconds after predicted onset
STAGE1_SAMPLING_S = 1.0

# Stage 2: ±25 frames around stage-1 click.
STAGE2_HALF_WIDTH_FRAMES = 25

# Default coarse offset if missing from cache.
DEFAULT_COARSE_OFFSET_S = 15.0

# Barcode montage layout.
MONTAGE_ROWS = 5
MONTAGE_COLS = 7  # frames around predicted onset: -3, -2, -1, 0, +1, +2, +3

# Output directories (created on demand).
FIGS_DIR = os.path.join("figs", "video_sync")


# ---------------------------------------------------------------------------
# Grid-math helpers (pure-logic, unit-tested)
# ---------------------------------------------------------------------------


def stage1_frame_indices(predicted: int, fps: float, n_frames: int) -> list[int]:
    """Return 50 frame indices for the coarse stage.

    Sampling interval is 1 s of video (= ``fps`` frames). Window covers
    ``[predicted − STAGE1_PRE_S, predicted + STAGE1_POST_S]`` in seconds,
    clamped to ``[0, n_frames − 1]``. If clamping shortens the window, the
    sampling step is preserved; the array may end earlier than predicted+post.
    """
    step = max(1, int(round(fps * STAGE1_SAMPLING_S)))
    start = predicted - int(round(fps * STAGE1_PRE_S))
    start = max(0, start)
    # End is start + (N_CELLS - 1) * step, but clamped to n_frames - 1.
    indices = [min(n_frames - 1, start + i * step) for i in range(N_CELLS)]
    return indices


def stage2_frame_indices(stage1_click: int, fps: float, n_frames: int) -> list[int]:
    """Return 50 frame indices for the fine stage (1 frame per cell)."""
    # Note: fps not used here (one frame per cell), but kept for signature symmetry.
    start = stage1_click - STAGE2_HALF_WIDTH_FRAMES
    start = max(0, min(n_frames - N_CELLS, start))
    return [min(n_frames - 1, start + i) for i in range(N_CELLS)]


# ---------------------------------------------------------------------------
# Frame I/O + eye-region crop
# ---------------------------------------------------------------------------


def load_cropped_frames(
    video_path: str,
    frame_indices: Sequence[int],
    crop: tuple[int, int, int, int] = EYE_REGION_CROP_BG046,
) -> list[np.ndarray]:
    """Load specified frames from *video_path*, crop to *crop*, return grayscale arrays.

    Reads via ``cv2.VideoCapture``. Frames are returned in the same order as
    ``frame_indices``. If a frame cannot be read, a zero-filled placeholder
    of the crop shape is inserted (so the grid layout never breaks).
    """
    y0, y1, x0, x1 = crop
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    out: list[np.ndarray] = []
    try:
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                out.append(np.zeros((y1 - y0, x1 - x0), dtype=np.uint8))
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out.append(gray[y0:y1, x0:x1])
    finally:
        cap.release()
    return out


# ---------------------------------------------------------------------------
# Placeholders for later tasks (filled in by Tasks 3–5).
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point. Implemented in Task 5."""
    raise NotImplementedError("Wired up in Task 5.")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2.4: Run tests; expect pass**

Run: `py -m pytest tests/test_video_sync_anchor.py -v`
Expected: All 12 tests PASS (8 from Task 1 + 4 grid-math).

- [ ] **Step 2.5: Smoke-test the frame I/O against a real video**

Run a one-liner to verify `load_cropped_frames` works on session 09092025 (this confirms the eye-region crop is reasonable):

```bash
py -c "
from scripts.video.click_anchor import load_cropped_frames
from visdetect.core.video_sync import find_camera_files
cam = find_camera_files('09092025')
v = cam['eye_cam']['video']
frames = load_cropped_frames(v, [100, 200, 300])
print('Loaded', len(frames), 'frames; shape:', frames[0].shape)
"
```

Expected output: `Loaded 3 frames; shape: (220, 220)`. (`find_camera_files` returns `{"eye_cam": {"video": ..., "metadata": ...}, "front_cam": {...}}` — see `src/visdetect/core/video_sync.py:621`.)

- [ ] **Step 2.6: Commit**

```bash
git add scripts/video/click_anchor.py tests/test_video_sync_anchor.py
git commit -m "Add click_anchor.py skeleton with grid-math helpers and frame I/O"
```

---

## Task 3 — Interactive two-stage click UI

**Files:**
- Modify: `scripts/video/click_anchor.py`

No automated tests (matplotlib interactive). Verify manually after implementation.

- [ ] **Step 3.1: Implement the grid renderer + click handler**

In `scripts/video/click_anchor.py`, replace the placeholder section with these functions (insert above `def main()`):

```python
# ---------------------------------------------------------------------------
# Interactive two-stage click UI
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("TkAgg")  # interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _show_grid_and_get_click(
    frames: list[np.ndarray],
    frame_indices: Sequence[int],
    title: str,
    centre_frame: int,
    fps: float,
) -> Optional[int]:
    """Show a 5×10 grid of *frames* and wait for one click.

    Returns the absolute video frame index of the clicked cell, or ``None``
    if the user pressed ESC.

    Cell labels show "fr <abs_idx>\n<offset>ms" where offset is relative to
    *centre_frame*. Centre cell gets a yellow border.
    """
    assert len(frames) == N_CELLS == len(frame_indices)

    fig, axes = plt.subplots(
        GRID_ROWS, GRID_COLS, figsize=(15, 8.5),
        gridspec_kw=dict(wspace=0.05, hspace=0.25),
    )
    fig.suptitle(title, fontsize=10)

    centre_idx = int(np.argmin(np.abs(np.asarray(frame_indices) - centre_frame)))

    for i, (ax, frame, fidx) in enumerate(zip(axes.flat, frames, frame_indices)):
        ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
        offset_ms = (int(fidx) - int(centre_frame)) / fps * 1000.0
        ax.set_title(f"fr {fidx}\n{offset_ms:+.0f}ms", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if i == centre_idx:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("gold")
                spine.set_linewidth(2.0)

    # State captured by handlers via mutable list (closures in Python).
    result: list[Optional[int]] = [None]
    done: list[bool] = [False]

    def on_click(event):
        if event.inaxes is None:
            return
        # Identify which axes was clicked.
        for i, ax in enumerate(axes.flat):
            if event.inaxes is ax:
                # Confirm visually: draw a red box around the clicked axes.
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor("red")
                    spine.set_linewidth(3.0)
                fig.canvas.draw_idle()
                result[0] = int(frame_indices[i])
                # Schedule close after a short pause so the user sees the confirmation.
                fig.canvas.start_event_loop(0.5)
                done[0] = True
                plt.close(fig)
                return

    def on_key(event):
        if event.key == "escape":
            result[0] = None
            done[0] = True
            plt.close(fig)

    cid_click = fig.canvas.mpl_connect("button_press_event", on_click)
    cid_key = fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()  # blocks until plt.close(fig)

    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)

    return result[0]


def run_stage1(
    video_path: str,
    predicted: int,
    fps: float,
    n_frames: int,
) -> Optional[int]:
    """Stage 1 — coarse 50-second window, 1 frame per second."""
    indices = stage1_frame_indices(predicted, fps, n_frames)
    frames = load_cropped_frames(video_path, indices)
    return _show_grid_and_get_click(
        frames=frames,
        frame_indices=indices,
        title=(
            "Stage 1 — Coarse scan. Click the cell where the grating first "
            "appears in the eye. ESC to cancel.\n"
            f"(predicted onset = frame {predicted}; 1 s between cells; gold = predicted)"
        ),
        centre_frame=predicted,
        fps=fps,
    )


def run_stage2(
    video_path: str,
    stage1_click: int,
    fps: float,
    n_frames: int,
) -> Optional[int]:
    """Stage 2 — fine ±500ms window, 1 frame per cell."""
    indices = stage2_frame_indices(stage1_click, fps, n_frames)
    frames = load_cropped_frames(video_path, indices)
    return _show_grid_and_get_click(
        frames=frames,
        frame_indices=indices,
        title=(
            "Stage 2 — Fine pick. Click the exact frame where the grating "
            "appears. ESC to cancel.\n"
            f"(stage-1 click = frame {stage1_click}; 1 frame between cells; gold = stage-1 click)"
        ),
        centre_frame=stage1_click,
        fps=fps,
    )
```

- [ ] **Step 3.2: Smoke-test stage 1 manually**

Add a tiny temporary block at the bottom of the script for the smoke test, **DO NOT COMMIT**:

```python
# TEMPORARY smoke test — remove before commit
if __name__ == "__main__" and "--smoke" in sys.argv:
    from visdetect.core.video_sync import find_camera_files, load_camera_metadata
    cam = find_camera_files("09092025")
    video_path = cam["eye_cam"]["video"]
    meta_path = cam["eye_cam"]["metadata"]
    ts_ms, _, _ = load_camera_metadata(meta_path)
    fps = 1000.0 / float(np.median(np.diff(ts_ms)))
    n_frames = len(ts_ms)
    print(f"fps={fps:.2f}, n_frames={n_frames}")
    click = run_stage1(video_path, predicted=200, fps=fps, n_frames=n_frames)
    print("stage 1 click:", click)
    if click is not None:
        click2 = run_stage2(video_path, stage1_click=click, fps=fps, n_frames=n_frames)
        print("stage 2 click:", click2)
    sys.exit(0)
```

Run: `py scripts/video/click_anchor.py --smoke`

Expected: A 5×10 grid window opens with 50 cropped eye frames. You can click a cell; you should see a brief red border before the window closes; stdout prints "stage 1 click: <frame_idx>". Then a second 5×10 grid opens; click → window closes → stdout prints stage 2 click. ESC at either stage cancels.

If the window does not open or the click is not registered, the most likely cause is matplotlib backend selection — verify `matplotlib.use("TkAgg")` is taking effect. On Windows you may also need `pip install pyqt5` if Tk is unavailable.

- [ ] **Step 3.3: Remove the smoke block and commit**

Delete the temporary `--smoke` block from the bottom of the script. Then:

```bash
git add scripts/video/click_anchor.py
git commit -m "Add interactive two-stage click UI for anchor selection"
```

---

## Task 4 — Barcode montage renderer

**Files:**
- Modify: `scripts/video/click_anchor.py`

- [ ] **Step 4.1: Implement the montage renderer**

Insert above `def main()` (and below the stage-2 function from Task 3):

```python
# ---------------------------------------------------------------------------
# Barcode montage renderer
# ---------------------------------------------------------------------------


def _pick_sampled_trials(n_trials: int, n_rows: int = MONTAGE_ROWS) -> list[int]:
    """Return ``n_rows`` evenly spaced trial indices including 0 and n_trials-1."""
    if n_trials <= n_rows:
        return list(range(n_trials))
    return [int(round(i * (n_trials - 1) / (n_rows - 1))) for i in range(n_rows)]


def render_barcode_montage(
    session_name: str,
    anchor: dict,
    baseline_on: np.ndarray,
    video_path: str,
    ts_ms: np.ndarray,
    fps: float,
    out_path: str,
) -> None:
    """Render a 5-row × 7-column montage of predicted-onset frames per sampled trial.

    Each row corresponds to a sampled trial; columns show frames at predicted
    onset ± 3 frames. Centre column gets a red border (the predicted-onset
    frame); the user inspects whether the grating appears in the centre cells.
    """
    n_trials = len(baseline_on)
    trial_indices = _pick_sampled_trials(n_trials, MONTAGE_ROWS)
    implied_offset_s = float(anchor["implied_offset_s"])

    col_offsets = list(range(-(MONTAGE_COLS // 2), MONTAGE_COLS // 2 + 1))  # [-3..3]

    fig, axes = plt.subplots(
        MONTAGE_ROWS, MONTAGE_COLS,
        figsize=(MONTAGE_COLS * 1.8, MONTAGE_ROWS * 2.0),
        gridspec_kw=dict(wspace=0.05, hspace=0.25),
    )
    title = (
        f"Anchor-barcode montage — {session_name}\n"
        f"anchor trial 0 @ frame {anchor['video_frame_idx']} "
        f"(NI-DAQ {anchor['nidaq_baseline_on_s']:.3f}s, "
        f"implied offset {implied_offset_s:.3f}s) — {n_trials} trials"
    )
    fig.suptitle(title, fontsize=10)

    n_frames = len(ts_ms)

    for r, ti in enumerate(trial_indices):
        # Predicted video frame for this trial: where ts_ms is closest to
        # (baseline_on[ti] + implied_offset_s) * 1000.
        target_ms = (float(baseline_on[ti]) + implied_offset_s) * 1000.0
        target_ms_clamped = float(np.clip(target_ms, ts_ms[0], ts_ms[-1]))
        centre_frame = int(np.argmin(np.abs(ts_ms - target_ms_clamped)))
        indices = [
            int(np.clip(centre_frame + off, 0, n_frames - 1)) for off in col_offsets
        ]
        frames = load_cropped_frames(video_path, indices)

        for c, (ax, frame, fidx, off) in enumerate(zip(axes[r], frames, indices, col_offsets)):
            ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if off == 0:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor("red")
                    spine.set_linewidth(2.0)
            if r == 0:
                offset_ms = off / fps * 1000.0
                ax.set_title(f"{offset_ms:+.0f}ms", fontsize=8)

        # Row label (left side).
        axes[r, 0].set_ylabel(
            f"trial {ti}\nNI {float(baseline_on[ti]):.2f}s",
            fontsize=8, rotation=0, ha="right", va="center", labelpad=30,
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 4.2: Smoke-test by mocking an anchor dict and rendering on 09092025**

Add a temporary smoke block, **DO NOT COMMIT**:

```python
if __name__ == "__main__" and "--smoke-montage" in sys.argv:
    from visdetect.suite.loader import load_session
    from visdetect.core.video_sync import find_camera_files, load_camera_metadata
    sess = load_session("09092025")
    baseline_on = np.asarray(sess.ni_events.get("Baseline_ON", []), dtype=float)
    baseline_on = baseline_on[baseline_on > 0][:len(sess.trials)]
    cam = find_camera_files("09092025")
    video_path = cam["eye_cam"]["video"]
    meta_path = cam["eye_cam"]["metadata"]
    ts_ms, _, _ = load_camera_metadata(meta_path)
    fps = 1000.0 / float(np.median(np.diff(ts_ms)))
    # Use the cached coarse offset 4.0 s as a fake "implied offset" for the smoke test.
    anchor = {
        "session": "09092025",
        "anchor_trial_index": 0,
        "nidaq_baseline_on_s": float(baseline_on[0]),
        "video_frame_idx": compute_predicted_frame_idx(
            float(baseline_on[0]), 4.0, ts_ms
        ),
        "video_time_s": float(baseline_on[0] - 4.0),
        "implied_offset_s": -4.0,  # video_time_s - nidaq_baseline_on_s
        "frame_rate_fps": fps,
        "n_trials": int(len(baseline_on)),
        "clicked_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    out = os.path.join(FIGS_DIR, "09092025_barcode_montage_SMOKE.png")
    render_barcode_montage("09092025", anchor, baseline_on, video_path, ts_ms, fps, out)
    print("Wrote:", out)
    sys.exit(0)
```

Run: `py scripts/video/click_anchor.py --smoke-montage`

Expected: Writes `figs/video_sync/09092025_barcode_montage_SMOKE.png`. Open it and check:
- 5 rows × 7 cols of grayscale eye frames
- Centre column has red borders
- Row labels show trial index + NI-DAQ time
- Top row has column-offset labels in ms
- Title shows session, anchor info, implied offset, trial count

Visual sanity check — at the cached coarse offset 4.0s, the centre columns should show the grating on most rows (this is the known-good session). If they're systematically empty or systematically off, something is wrong in the time-to-frame mapping.

- [ ] **Step 4.3: Remove the smoke block + smoke PNG, commit**

```bash
rm figs/video_sync/09092025_barcode_montage_SMOKE.png
# Delete the temporary --smoke-montage block from the script.
git add scripts/video/click_anchor.py
git commit -m "Add barcode-montage renderer for Phase 1 anchor verification"
```

---

## Task 5 — CLI wiring + end-to-end run

**Files:**
- Modify: `scripts/video/click_anchor.py`

- [ ] **Step 5.1: Implement `main()`**

Replace the placeholder `main()` with the full CLI:

```python
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manually anchor trial 1 Baseline_ON for a session's eye-cam video.",
    )
    parser.add_argument("--session", required=True, help="Session name (e.g. 09092025).")
    parser.add_argument(
        "--reuse-existing-anchor", action="store_true",
        help="Skip the click UI and just render the montage from a saved anchor.",
    )
    args = parser.parse_args()

    session_name = args.session

    # ── Load session + camera + coarse offset ────────────────────────────
    sess = load_session(session_name)
    baseline_on = np.asarray(
        sess.ni_events.get("Baseline_ON", []), dtype=float
    )
    baseline_on = baseline_on[baseline_on > 0]
    n_task_trials = len(sess.trials)
    if n_task_trials > 0 and len(baseline_on) > n_task_trials:
        baseline_on = baseline_on[:n_task_trials]
    if len(baseline_on) == 0:
        logger.error("No Baseline_ON events for session %s — aborting.", session_name)
        return 2

    try:
        cam_files = find_camera_files(session_name)
    except Exception as exc:
        logger.error("Could not locate camera files for %s: %s", session_name, exc)
        return 2
    if "eye_cam" not in cam_files:
        logger.error("No eye-cam video/metadata pair found for %s.", session_name)
        return 2
    video_path = cam_files["eye_cam"]["video"]
    meta_path = cam_files["eye_cam"]["metadata"]

    ts_ms, _, _ = load_camera_metadata(meta_path)
    fps = 1000.0 / float(np.median(np.diff(ts_ms)))
    n_frames = len(ts_ms)

    coarse_offset_s = _read_coarse_offset(session_name)
    if coarse_offset_s is None:
        logger.warning(
            "No cached coarse offset for %s; falling back to %.1fs default.",
            session_name, DEFAULT_COARSE_OFFSET_S,
        )
        coarse_offset_s = DEFAULT_COARSE_OFFSET_S

    predicted = compute_predicted_frame_idx(
        baseline_on_s=float(baseline_on[0]),
        coarse_offset_s=float(coarse_offset_s),
        ts_ms=ts_ms,
    )

    # ── Anchor: load existing or run two-stage click ─────────────────────
    anchor: Optional[dict] = None
    if args.reuse_existing_anchor:
        anchor = load_anchor(session_name)
        if anchor is None:
            logger.error(
                "--reuse-existing-anchor passed but no anchor JSON found for %s.",
                session_name,
            )
            return 2
    else:
        existing = load_anchor(session_name)
        if existing is not None:
            resp = input(
                f"Anchor JSON for {session_name} already exists; overwrite? [y/N] "
            ).strip().lower()
            if resp not in ("y", "yes"):
                logger.info("Aborting; existing anchor preserved.")
                return 0

        click1 = run_stage1(video_path, predicted, fps, n_frames)
        if click1 is None:
            logger.info("Stage 1 cancelled by user.")
            return 1
        click2 = run_stage2(video_path, click1, fps, n_frames)
        if click2 is None:
            logger.info("Stage 2 cancelled by user.")
            return 1

        anchor = {
            "session": session_name,
            "anchor_trial_index": 0,
            "nidaq_baseline_on_s": float(baseline_on[0]),
            "video_frame_idx": int(click2),
            "video_time_s": float(ts_ms[int(click2)] / 1000.0),
            "implied_offset_s": float(ts_ms[int(click2)] / 1000.0 - float(baseline_on[0])),
            "frame_rate_fps": float(fps),
            "n_trials": int(len(baseline_on)),
            "clicked_at": _dt.datetime.now().isoformat(timespec="seconds"),
        }
        save_anchor(session_name, anchor)
        logger.info(
            "Anchor saved: trial 0 @ frame %d (video time %.3fs); implied offset = %.3fs",
            anchor["video_frame_idx"],
            anchor["video_time_s"],
            anchor["implied_offset_s"],
        )

    # ── Render montage ──────────────────────────────────────────────────
    montage_path = os.path.join(FIGS_DIR, f"{session_name}_barcode_montage.png")
    render_barcode_montage(
        session_name=session_name,
        anchor=anchor,
        baseline_on=baseline_on,
        video_path=video_path,
        ts_ms=ts_ms,
        fps=fps,
        out_path=montage_path,
    )

    anchor_path = os.path.join(VIDEO_SYNC_DIR, f"{session_name}_anchor.json")
    print(f"Anchor:   {anchor_path}")
    print(f"Montage:  {montage_path}")
    return 0


def _read_coarse_offset(session_name: str) -> Optional[float]:
    """Return the cached coarse offset for *session_name*, or ``None`` if absent."""
    import json
    path = os.path.join(VIDEO_SYNC_DIR, "coarse_offsets.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    val = data.get(session_name)
    return float(val) if val is not None else None
```

- [ ] **Step 5.2: Run end-to-end on session 09092025**

Run: `py scripts/video/click_anchor.py --session 09092025`

Expected flow:
1. Stage 1 grid opens — click the cell where you see the grating appear in the eye reflection.
2. Stage 2 grid opens — click the exact frame.
3. Stdout prints anchor + montage paths.
4. Files exist:
   - `data/cache/video_sync/09092025_anchor.json`
   - `figs/video_sync/09092025_barcode_montage.png`
5. Open the montage. Centre column should show the grating on every (or nearly every) row — that confirms the barcode holds for 09092025.

- [ ] **Step 5.3: Commit**

```bash
git add scripts/video/click_anchor.py
git commit -m "Wire up click_anchor CLI with end-to-end anchor + montage flow"
```

- [ ] **Step 5.4: Run on the remaining two anchor sessions**

```bash
py scripts/video/click_anchor.py --session 14082025
py scripts/video/click_anchor.py --session 03072025
```

For each session, click trial 1 in both stages and inspect the resulting montage. Record observations (which centre cells show the grating, drift patterns, anything weird) in a short summary you bring back to the conversation — these inform Phase 2 design.

- [ ] **Step 5.5: Commit the generated artifacts**

```bash
git add data/cache/video_sync/09092025_anchor.json \
        data/cache/video_sync/14082025_anchor.json \
        data/cache/video_sync/03072025_anchor.json \
        figs/video_sync/09092025_barcode_montage.png \
        figs/video_sync/14082025_barcode_montage.png \
        figs/video_sync/03072025_barcode_montage.png
git commit -m "Phase 1 anchor + montage outputs for 3 anchor sessions"
```

**(Skip the commit of figures if `figs/` is gitignored in this repo — check `.gitignore` first. If figures are not tracked, just keep the anchor JSONs committed and bring the PNGs to the conversation as attachments.)**

---

## Self-review notes

- **Spec coverage:** Each Phase 1 deliverable in the spec maps to a task:
  - Two-stage click UI → Task 3
  - Anchor JSON schema + persistence → Task 1 (helpers) + Task 5 (writing)
  - Barcode montage 5×7 grid → Task 4
  - CLI invocation and `--reuse-existing-anchor` flag → Task 5
  - Run on 3 anchor sessions → Task 5 steps 5.2 + 5.4
  - Error handling: missing coarse offset (default 15s), missing video (abort), ESC at either stage (return non-zero, no writes), existing anchor (prompt) → Task 5 main()
- **Placeholder scan:** No TBDs. Every code step contains the actual code. Smoke-test blocks are explicitly marked as DO NOT COMMIT.
- **Type consistency:** `compute_predicted_frame_idx` signature `(baseline_on_s, coarse_offset_s, ts_ms) -> int` is the same in Task 1 and Task 5. `stage1_frame_indices(predicted, fps, n_frames)` and `stage2_frame_indices(stage1_click, fps, n_frames)` signatures match between Task 2 tests and Task 5 usage. `_show_grid_and_get_click(frames, frame_indices, title, centre_frame, fps)` signature matches between Task 3 definition and Task 3 usage by `run_stage1`/`run_stage2`.
- **Out of scope (per spec):** front-cam, multi-anchor, auto-detection, Theil-Sen — none of these appear in the plan.
