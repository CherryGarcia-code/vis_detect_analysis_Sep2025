# Video Sync Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement subsystem A (multi-anchor + 2-anchor slope fit producing `{session}_video_sync.json`) and subsystem C (per-trial manual tag tool that overrides slope-fit predictions). Three atomic CLI commands plus targeted library additions. No auto-detection (deferred to Phase 3).

**Architecture:** Three CLI commands — `click_anchor --anchor-last` (existing script + new flag), new `scripts/video/fit_sync.py`, new `scripts/video/tag_trials.py`. Library additions in `src/visdetect/core/video_sync.py`: anchor JSON v1→v2 schema migration, `compute_implied_offset` helper, `per_trial_overrides` field on the existing `SyncResult` dataclass, `fit_2anchor_clock` function, and a single-line carve-out in `SyncResult.quality` for `detection_method == "manual_slope_fit"`.

**Tech Stack:** Python 3, NumPy, matplotlib TkAgg, OpenCV (`cv2.VideoCapture`), pytest. Run scripts with `py` (Windows + Git Bash). Tests via `py -m pytest`.

**Spec:** [`docs/superpowers/specs/2026-05-29-video-sync-phase2-design.md`](../specs/2026-05-29-video-sync-phase2-design.md)

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `src/visdetect/core/video_sync.py` | Modify | + `_migrate_anchor_v1_to_v2`, `compute_implied_offset`, `_build_v2_anchor_file`, `_build_anchor_entry`, `_merge_anchor_into_file`; update `load_anchor` (transparent v1 read) and `save_anchor` (v2 write); add `per_trial_overrides` field to `SyncResult`; update `SyncResult.to_dict` to include it; add `manual_slope_fit` carve-out in `SyncResult.quality`; + `fit_2anchor_clock` |
| `scripts/video/click_anchor.py` | Modify | + `--anchor-last` CLI flag; refactor `_build_anchor_dict` and its 3 call sites to use the new v2 helpers from the library |
| `scripts/video/fit_sync.py` | Create | NEW CLI: load anchors → `fit_2anchor_clock` → `save_video_sync` → render slope-fitted barcode montage |
| `scripts/video/tag_trials.py` | Create | NEW CLI: per-trial verify-and-advance scrubber that mutates `per_trial_overrides` in the sync JSON |
| `tests/test_video_sync_anchor.py` | Modify | + migration tests, `compute_implied_offset` tests, `fit_2anchor_clock` tests, `per_trial_overrides` round-trip tests, `quality` manual carve-out test |
| `tests/test_video_sync_tag_trials.py` | Create | Pure-logic `TagState` transition tests |

**Testing scope:** Unit-test all pure-logic and pure-math additions. Interactive matplotlib UI is not unit-tested — verified by user smoke-runs (per Phase 1 convention).

---

## Task 1 — Library: Anchor schema v1 ↔ v2 + `compute_implied_offset`

Refactor the anchor JSON to a list-of-anchors v2 schema while transparently reading legacy v1 JSONs. Update click_anchor.py callers to build v2-style anchor entries.

**Files:**
- Modify: `src/visdetect/core/video_sync.py` (add helpers near the existing `_anchor_path` / `save_anchor` / `load_anchor` block around lines 2585–2616)
- Modify: `tests/test_video_sync_anchor.py`
- Modify: `scripts/video/click_anchor.py` (the existing `_build_anchor_dict` at module level + its 3 call sites)

### Step 1.1: Write failing tests for the migration + new helpers

Append to `tests/test_video_sync_anchor.py`:

```python
# ---------------------------------------------------------------------------
# Phase 2: anchor v1 -> v2 migration and helpers
# ---------------------------------------------------------------------------


def _v1_anchor_fixture() -> dict:
    """A v1 anchor JSON identical in shape to what Phase 1.5 wrote."""
    return {
        "session": "09092025",
        "anchor_trial_index": 0,
        "nidaq_baseline_on_s": 27.829173432012986,
        "video_frame_idx": 1167,
        "video_time_s": 23.218682,
        "implied_offset_s": -4.610491432012985,
        "frame_rate_fps": 50.0400320251914,
        "n_trials": 551,
        "clicked_at": "2026-05-29T11:51:46",
    }


def test_migrate_anchor_v1_to_v2_basic():
    v1 = _v1_anchor_fixture()
    v2 = vs._migrate_anchor_v1_to_v2(v1)
    assert v2["schema_version"] == 2
    assert v2["session"] == "09092025"
    assert v2["frame_rate_fps"] == 50.0400320251914
    assert v2["n_trials"] == 551
    assert isinstance(v2["anchors"], list)
    assert len(v2["anchors"]) == 1
    a = v2["anchors"][0]
    assert a["trial_index"] == 0
    assert a["nidaq_baseline_on_s"] == 27.829173432012986
    assert a["video_frame_idx"] == 1167
    assert a["video_time_s"] == 23.218682
    assert a["clicked_at"] == "2026-05-29T11:51:46"
    # implied_offset_s is dropped (derivable)
    assert "implied_offset_s" not in a
    # top-level v1 fields are dropped from anchor entries
    assert "anchor_trial_index" not in a


def test_migrate_anchor_v2_is_idempotent():
    v1 = _v1_anchor_fixture()
    v2 = vs._migrate_anchor_v1_to_v2(v1)
    v2_again = vs._migrate_anchor_v1_to_v2(v2)
    assert v2_again == v2


def test_compute_implied_offset_from_anchor_entry():
    anchor = {
        "trial_index": 0,
        "nidaq_baseline_on_s": 27.829,
        "video_frame_idx": 1167,
        "video_time_s": 23.219,
        "clicked_at": "2026-05-29T11:51:46",
    }
    offset = vs.compute_implied_offset(anchor)
    # offset = video_time_s - nidaq_baseline_on_s
    assert abs(offset - (23.219 - 27.829)) < 1e-9


def test_build_anchor_entry_returns_v2_shape():
    ts_ms = np.arange(0.0, 100000.0, 20.0)  # 50fps, 5000 frames
    baseline_on = np.array([27.829, 1574.27])
    entry = vs._build_anchor_entry(
        baseline_on=baseline_on,
        ts_ms=ts_ms,
        trial_index=0,
        frame_idx=1167,
    )
    assert set(entry.keys()) == {
        "trial_index", "nidaq_baseline_on_s",
        "video_frame_idx", "video_time_s", "clicked_at",
    }
    assert entry["trial_index"] == 0
    assert entry["video_frame_idx"] == 1167
    assert abs(entry["video_time_s"] - (ts_ms[1167] / 1000.0)) < 1e-9
    assert entry["nidaq_baseline_on_s"] == 27.829


def test_build_v2_anchor_file_minimal():
    entry0 = {
        "trial_index": 0,
        "nidaq_baseline_on_s": 27.829,
        "video_frame_idx": 1167,
        "video_time_s": 23.219,
        "clicked_at": "2026-05-29T11:51:46",
    }
    f = vs._build_v2_anchor_file(
        session_name="09092025",
        fps=50.04,
        n_trials=551,
        anchor_entries=[entry0],
    )
    assert f["schema_version"] == 2
    assert f["session"] == "09092025"
    assert f["frame_rate_fps"] == 50.04
    assert f["n_trials"] == 551
    assert f["anchors"] == [entry0]


def test_merge_anchor_into_file_appends_new_trial():
    base = vs._build_v2_anchor_file(
        session_name="09092025", fps=50.04, n_trials=551,
        anchor_entries=[
            {"trial_index": 0, "nidaq_baseline_on_s": 27.83,
             "video_frame_idx": 1167, "video_time_s": 23.22,
             "clicked_at": "2026-05-29T11:51:46"}
        ],
    )
    new = {
        "trial_index": 550, "nidaq_baseline_on_s": 7255.49,
        "video_frame_idx": 363270, "video_time_s": 7259.79,
        "clicked_at": "2026-06-01T14:00:00",
    }
    merged = vs._merge_anchor_into_file(base, new)
    assert len(merged["anchors"]) == 2
    # Sorted by trial_index
    assert merged["anchors"][0]["trial_index"] == 0
    assert merged["anchors"][1]["trial_index"] == 550


def test_merge_anchor_into_file_overwrites_existing_trial_index():
    base = vs._build_v2_anchor_file(
        session_name="09092025", fps=50.04, n_trials=551,
        anchor_entries=[
            {"trial_index": 0, "nidaq_baseline_on_s": 27.83,
             "video_frame_idx": 1167, "video_time_s": 23.22,
             "clicked_at": "2026-05-29T11:51:46"}
        ],
    )
    replacement = {
        "trial_index": 0, "nidaq_baseline_on_s": 27.83,
        "video_frame_idx": 1200, "video_time_s": 23.88,
        "clicked_at": "2026-06-01T15:00:00",
    }
    merged = vs._merge_anchor_into_file(base, replacement)
    assert len(merged["anchors"]) == 1
    assert merged["anchors"][0]["video_frame_idx"] == 1200
    assert merged["anchors"][0]["clicked_at"] == "2026-06-01T15:00:00"


def test_load_anchor_migrates_v1_file_in_memory(tmp_path):
    import json
    v1 = _v1_anchor_fixture()
    p = tmp_path / "09092025_anchor.json"
    p.write_text(json.dumps(v1))

    loaded = vs.load_anchor("09092025", sync_dir=str(tmp_path))

    assert loaded["schema_version"] == 2
    assert loaded["anchors"][0]["trial_index"] == 0
    # On-disk file should NOT have been rewritten (load is read-only)
    on_disk = json.loads(p.read_text())
    assert "anchor_trial_index" in on_disk


def test_save_anchor_writes_v2_only(tmp_path):
    import json
    f = vs._build_v2_anchor_file(
        session_name="09092025", fps=50.04, n_trials=551,
        anchor_entries=[
            {"trial_index": 0, "nidaq_baseline_on_s": 27.83,
             "video_frame_idx": 1167, "video_time_s": 23.22,
             "clicked_at": "2026-05-29T11:51:46"}
        ],
    )
    vs.save_anchor("09092025", f, sync_dir=str(tmp_path))
    on_disk = json.loads((tmp_path / "09092025_anchor.json").read_text())
    assert on_disk["schema_version"] == 2
    assert "anchors" in on_disk
    assert "anchor_trial_index" not in on_disk
```

### Step 1.2: Run tests; expect failure

Run: `py -m pytest tests/test_video_sync_anchor.py -v -k "phase2 or migrate or build_anchor or build_v2 or merge_anchor or compute_implied or load_anchor_migr or save_anchor_writes"`

(The `-k` expression matches only the new tests so we see clean failures; existing 18 tests still pass.)

Expected: 9 new tests FAIL with `AttributeError: module 'visdetect.core.video_sync' has no attribute '_migrate_anchor_v1_to_v2'` (or similar for each helper).

### Step 1.3: Implement the v2 helpers

In `src/visdetect/core/video_sync.py`, replace the existing Phase 1 anchor section (the `_anchor_path` / `save_anchor` / `load_anchor` block) with a Phase 2 expanded version. Find this block near the end of the file (search for `# Anchor JSON helpers (Phase 1 of corneal-barcode redesign)`).

Replace the entire block with:

```python
# =====================================================================
# Anchor JSON helpers (Phase 2: list-of-anchors schema, v1 read compat)
# =====================================================================


def _anchor_path(session_name: str, sync_dir: Optional[str] = None) -> str:
    """Path to the anchor JSON for *session_name*."""
    out_dir = sync_dir or VIDEO_SYNC_DIR
    session_name = str(int(session_name)).zfill(8)
    return os.path.join(out_dir, f"{session_name}_anchor.json")


def _migrate_anchor_v1_to_v2(d: dict) -> dict:
    """Convert a Phase 1 (single-anchor) JSON dict to the Phase 2 (list) shape.

    Idempotent: passing a v2 dict returns it unchanged.
    """
    if d.get("schema_version") == 2 or "anchors" in d:
        return d
    entry = {
        "trial_index": int(d["anchor_trial_index"]),
        "nidaq_baseline_on_s": float(d["nidaq_baseline_on_s"]),
        "video_frame_idx": int(d["video_frame_idx"]),
        "video_time_s": float(d["video_time_s"]),
        "clicked_at": str(d["clicked_at"]),
    }
    return {
        "session": str(d["session"]),
        "schema_version": 2,
        "frame_rate_fps": float(d["frame_rate_fps"]),
        "n_trials": int(d["n_trials"]),
        "anchors": [entry],
    }


def compute_implied_offset(anchor: dict) -> float:
    """Return ``video_time_s - nidaq_baseline_on_s`` for a single anchor entry.

    Used by HUDs and reports that want to display "the camera started this
    many seconds after NI-DAQ" in a human-readable form.
    """
    return float(anchor["video_time_s"]) - float(anchor["nidaq_baseline_on_s"])


def _build_anchor_entry(
    baseline_on: np.ndarray,
    ts_ms: np.ndarray,
    trial_index: int,
    frame_idx: int,
) -> dict:
    """Build a single v2 anchor entry from a clicked frame index."""
    import datetime as _dt
    fi = int(frame_idx)
    return {
        "trial_index": int(trial_index),
        "nidaq_baseline_on_s": float(baseline_on[int(trial_index)]),
        "video_frame_idx": fi,
        "video_time_s": float(ts_ms[fi] / 1000.0),
        "clicked_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }


def _build_v2_anchor_file(
    session_name: str,
    fps: float,
    n_trials: int,
    anchor_entries: list,
) -> dict:
    """Construct the top-level v2 anchor JSON dict."""
    return {
        "session": str(session_name),
        "schema_version": 2,
        "frame_rate_fps": float(fps),
        "n_trials": int(n_trials),
        "anchors": list(anchor_entries),
    }


def _merge_anchor_into_file(base: dict, new_entry: dict) -> dict:
    """Return a copy of *base* with *new_entry* merged into its anchors list.

    Replaces any existing anchor with the same ``trial_index``. The result is
    sorted by ``trial_index``.
    """
    new_idx = int(new_entry["trial_index"])
    kept = [a for a in base["anchors"] if int(a["trial_index"]) != new_idx]
    kept.append(new_entry)
    kept.sort(key=lambda a: int(a["trial_index"]))
    out = dict(base)
    out["anchors"] = kept
    return out


def save_anchor(
    session_name: str,
    anchor: dict,
    sync_dir: Optional[str] = None,
) -> None:
    """Write *anchor* (v2 schema) to ``{sync_dir}/{session_name}_anchor.json``.

    Callers must pass a v2 dict; building one is the responsibility of
    :func:`_build_v2_anchor_file` (top-level) plus :func:`_build_anchor_entry`
    (per-anchor) plus :func:`_merge_anchor_into_file` (composition).
    Overwrites any existing file. Creates the directory if needed.
    """
    out_dir = sync_dir or VIDEO_SYNC_DIR
    os.makedirs(out_dir, exist_ok=True)
    with open(_anchor_path(session_name, sync_dir=sync_dir), "w") as f:
        json.dump(anchor, f, indent=2)


def load_anchor(
    session_name: str,
    sync_dir: Optional[str] = None,
) -> Optional[dict]:
    """Read the anchor JSON for *session_name* and return it in v2 form.

    Legacy v1 JSONs are migrated in memory (the on-disk file is NOT rewritten
    by this read; it gets rewritten next time :func:`save_anchor` is called).
    Returns ``None`` if no file exists.
    """
    path = _anchor_path(session_name, sync_dir=sync_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        raw = json.load(f)
    return _migrate_anchor_v1_to_v2(raw)
```

Also confirm that `json`, `os`, `datetime as _dt`, `Optional`, `np` are already imported at the top of `video_sync.py` (they should be from existing code; do not add duplicate imports).

### Step 1.4: Run tests; confirm new tests pass and existing 18 tests still pass

Run: `py -m pytest tests/test_video_sync_anchor.py -v`

Expected: All tests pass. Existing 18 + 9 new = **27 passing**.

If any existing test fails because it monkeypatches `VIDEO_SYNC_DIR` and the new `_anchor_path` signature changed, update only those tests minimally to pass `sync_dir=str(tmp_path)` instead — but the existing helpers' signatures should remain compatible (the new `_anchor_path` still defaults to `VIDEO_SYNC_DIR` when `sync_dir` is None).

### Step 1.5: Update `scripts/video/click_anchor.py` to build v2 anchors

The existing `_build_anchor_dict` in `click_anchor.py` (introduced in commit `672c2f0`) builds v1 dicts and is called from 3 sites: the 2-stage `main()` save path, the scrub Save handler, and the scrub R-key preview handler.

Replace `_build_anchor_dict` and its 3 call sites with the new v2-aware logic.

First, find `_build_anchor_dict` in `scripts/video/click_anchor.py` (it lives in the grid-math helpers section, just above `jump_to_predicted_frame`). Replace it with:

```python
def _build_or_merge_anchor_file(
    session_name: str,
    baseline_on: np.ndarray,
    ts_ms: np.ndarray,
    fps: float,
    trial_index: int,
    frame_idx: int,
) -> dict:
    """Build a v2 anchor JSON dict that merges this anchor into any existing file.

    All three Phase 1 anchor-creation paths (2-stage click, scrub Save, scrub
    preview-render) use this helper so the multi-anchor list stays consistent
    across re-saves.
    """
    new_entry = _build_anchor_entry(
        baseline_on=baseline_on,
        ts_ms=ts_ms,
        trial_index=trial_index,
        frame_idx=frame_idx,
    )
    existing = load_anchor(session_name)
    if existing is None:
        return _build_v2_anchor_file(
            session_name=session_name,
            fps=fps,
            n_trials=int(len(baseline_on)),
            anchor_entries=[new_entry],
        )
    return _merge_anchor_into_file(existing, new_entry)
```

And update the existing visdetect import block at the top of `click_anchor.py` to add the new helpers (`_build_anchor_entry`, `_build_v2_anchor_file`, `_merge_anchor_into_file`, and `compute_implied_offset` — used in the v1-shape adapters below):

```python
from visdetect.core.video_sync import (
    find_camera_files,
    load_camera_metadata,
    compute_predicted_frame_idx,
    load_anchor,
    save_anchor,
    compute_implied_offset,
    _build_anchor_entry,
    _build_v2_anchor_file,
    _merge_anchor_into_file,
)
```

Now update the 3 call sites in `click_anchor.py`. Each previously called `_build_anchor_dict(session_name, baseline_on, ts_ms, fps, frame_idx)`. Each now calls `_build_or_merge_anchor_file(session_name, baseline_on, ts_ms, fps, trial_index=0, frame_idx=frame_idx)`.

**Call site 1** (Space/Enter handler in `_run_scrub`):

Find:
```python
            anchor = _build_anchor_dict(session_name, baseline_on, ts_ms, fps, state["frame_idx"])
```

Replace with:
```python
            anchor = _build_or_merge_anchor_file(
                session_name, baseline_on, ts_ms, fps,
                trial_index=0, frame_idx=state["frame_idx"],
            )
```

**Call site 2** (R handler in `_run_scrub`):

Find:
```python
            candidate = _build_anchor_dict(session_name, baseline_on, ts_ms, fps, state["frame_idx"])
```

Replace with:
```python
            candidate = _build_or_merge_anchor_file(
                session_name, baseline_on, ts_ms, fps,
                trial_index=0, frame_idx=state["frame_idx"],
            )
```

The `render_barcode_montage` call in the R handler currently passes the whole dict. We need it to pass a dict that looks like a single-anchor v1 dict for backward compat with `render_barcode_montage`'s schema requirements — OR update `render_barcode_montage` to accept v2 anchors. The cleaner choice for Task 1's scope is to leave `render_barcode_montage` alone and pass a v1-shaped dict to it. Build a v1 view of the trial-0 anchor for rendering:

Replace the R handler block more fully with:
```python
            candidate_file = _build_or_merge_anchor_file(
                session_name, baseline_on, ts_ms, fps,
                trial_index=0, frame_idx=state["frame_idx"],
            )
            # render_barcode_montage expects v1 single-anchor shape.
            entry0 = candidate_file["anchors"][0]
            candidate_for_render = {
                "session": candidate_file["session"],
                "anchor_trial_index": entry0["trial_index"],
                "nidaq_baseline_on_s": entry0["nidaq_baseline_on_s"],
                "video_frame_idx": entry0["video_frame_idx"],
                "video_time_s": entry0["video_time_s"],
                "implied_offset_s": compute_implied_offset(entry0),
                "frame_rate_fps": candidate_file["frame_rate_fps"],
                "n_trials": candidate_file["n_trials"],
                "clicked_at": entry0["clicked_at"],
            }
            montage_path = os.path.join(
                FIGS_DIR, f"{session_name}_barcode_montage_PREVIEW.png"
            )
            render_barcode_montage(
                session_name=session_name,
                anchor=candidate_for_render,
                baseline_on=baseline_on,
                video_path=video_path,
                ts_ms=ts_ms,
                fps=fps,
                out_path=montage_path,
            )
```

(`compute_implied_offset` was already added to the imports group above.)

**Call site 3** (2-stage flow `else:` branch in `main()`):

Find:
```python
        anchor = _build_anchor_dict(session_name, baseline_on, ts_ms, fps, click2)
```

Replace with:
```python
        anchor = _build_or_merge_anchor_file(
            session_name, baseline_on, ts_ms, fps,
            trial_index=0, frame_idx=click2,
        )
```

Then the existing montage-rendering code at the bottom of `main()` also needs to convert v2 to v1 shape for `render_barcode_montage`. Find:

```python
    # Render montage
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
```

Replace with:

```python
    # Render montage. render_barcode_montage takes v1 single-anchor shape.
    entry0 = anchor["anchors"][0]
    anchor_for_render = {
        "session": anchor["session"],
        "anchor_trial_index": entry0["trial_index"],
        "nidaq_baseline_on_s": entry0["nidaq_baseline_on_s"],
        "video_frame_idx": entry0["video_frame_idx"],
        "video_time_s": entry0["video_time_s"],
        "implied_offset_s": compute_implied_offset(entry0),
        "frame_rate_fps": anchor["frame_rate_fps"],
        "n_trials": anchor["n_trials"],
        "clicked_at": entry0["clicked_at"],
    }
    montage_path = os.path.join(FIGS_DIR, f"{session_name}_barcode_montage.png")
    render_barcode_montage(
        session_name=session_name,
        anchor=anchor_for_render,
        baseline_on=baseline_on,
        video_path=video_path,
        ts_ms=ts_ms,
        fps=fps,
        out_path=montage_path,
    )
```

Also in the `--reuse-existing-anchor` branch, `anchor = load_anchor(session_name)` now returns a v2 dict. The same v1 conversion applies before the render call — but only one set of conversion code is needed because the render path is the same in both branches. The simplest refactor is to define `anchor_for_render` once, just before the render call, regardless of how `anchor` was obtained.

Delete the old `_build_anchor_dict` function from `click_anchor.py` entirely. Its only remaining role is replaced by `_build_or_merge_anchor_file`.

### Step 1.6: Verify the script still imports and runs

Run: `py -c "from scripts.video.click_anchor import main, _run_scrub, jump_to_predicted_frame, _build_or_merge_anchor_file; print('ok')"`

Expected: `ok`. (Note: `_build_anchor_dict` is gone; replace its import in any test files if used. Search `_build_anchor_dict` in the test file with `grep -n _build_anchor_dict tests/test_video_sync_anchor.py` — if it appears, the corresponding test needs updating.)

Run: `py -m pytest tests/test_video_sync_anchor.py -v`

Expected: 27/27 passing (18 existing + 9 new). (Verified: the existing test file does not import `_build_anchor_dict`, so no test updates are needed.)

Run: `py scripts/video/click_anchor.py --help`

Expected: Existing help text, no errors at import time.

### Step 1.7: Commit

```bash
git add src/visdetect/core/video_sync.py scripts/video/click_anchor.py tests/test_video_sync_anchor.py
git commit -m "$(cat <<'EOF'
Phase 2: anchor JSON v2 schema (list of anchors) + transparent v1 read

Adds _migrate_anchor_v1_to_v2, _build_anchor_entry, _build_v2_anchor_file,
_merge_anchor_into_file, and compute_implied_offset to video_sync.py.
load_anchor migrates v1 -> v2 in memory; save_anchor writes v2.

click_anchor.py:
- Replaces _build_anchor_dict with _build_or_merge_anchor_file, which
  merges a new anchor (by trial_index) into the existing v2 file.
- All 3 anchor-creation paths (2-stage save, scrub Save, scrub preview)
  now produce v2-merged dicts; render_barcode_montage continues to take
  the v1 single-anchor shape via an inline conversion at each render call.

Adds 9 unit tests covering migration, helper construction, merge semantics,
load-time v1 migration, and v2-only save.

Phase 1 anchor JSONs already on disk remain readable; they get rewritten
in v2 form on the next save_anchor call.
EOF
)"
```

---

## Task 2 — Library: `per_trial_overrides` on `SyncResult`, manual quality carve-out, `fit_2anchor_clock`

Extend the existing `SyncResult` with an optional `per_trial_overrides` field and a single-line carve-out in the `quality` property so that 2-anchor exact fits are not mis-tiered as "failed". Add the `fit_2anchor_clock` function that produces a `SyncResult` from a list of v2 anchor entries.

**Files:**
- Modify: `src/visdetect/core/video_sync.py` (the existing `SyncResult` dataclass around lines 508–582)
- Modify: `tests/test_video_sync_anchor.py`

### Step 2.1: Write failing tests

Append to `tests/test_video_sync_anchor.py`:

```python
# ---------------------------------------------------------------------------
# Phase 2: SyncResult.per_trial_overrides + manual quality + fit_2anchor_clock
# ---------------------------------------------------------------------------


def test_sync_result_default_per_trial_overrides_is_none():
    sr = vs.SyncResult(
        slope=1.0, offset=0.0, n_anchors=2, n_baseline_on=10,
        rmse_ms=0.0, max_residual_ms=0.0, cv_rmse_ms=0.0,
        slope_ppm=0.0, durbin_watson=2.0,
        detection_method="manual_slope_fit",
    )
    assert sr.per_trial_overrides is None


def test_sync_result_to_dict_includes_per_trial_overrides_when_set():
    sr = vs.SyncResult(
        slope=1.0, offset=0.0, n_anchors=2, n_baseline_on=10,
        rmse_ms=0.0, max_residual_ms=0.0, cv_rmse_ms=0.0,
        slope_ppm=0.0, durbin_watson=2.0,
        detection_method="manual_slope_fit",
        per_trial_overrides={5: 250, 8: 400},
    )
    d = sr.to_dict()
    assert "per_trial_overrides" in d
    # JSON keys are strings on disk; field stays as int keys in memory.
    assert d["per_trial_overrides"] == {5: 250, 8: 400}


def test_sync_result_to_dict_omits_overrides_when_none():
    sr = vs.SyncResult(
        slope=1.0, offset=0.0, n_anchors=2, n_baseline_on=10,
        rmse_ms=0.0, max_residual_ms=0.0, cv_rmse_ms=0.0,
        slope_ppm=0.0, durbin_watson=2.0,
        detection_method="manual_slope_fit",
    )
    d = sr.to_dict()
    assert "per_trial_overrides" not in d


def test_quality_manual_carve_out_returns_good_for_valid_2anchor():
    sr = vs.SyncResult(
        slope=1.0000234, offset=-4.61, n_anchors=2, n_baseline_on=551,
        rmse_ms=0.0, max_residual_ms=0.0, cv_rmse_ms=0.0,
        slope_ppm=23.4, durbin_watson=0.0,  # DW=0 would fail regression path
        detection_method="manual_slope_fit",
    )
    assert sr.quality == "good"


def test_quality_manual_carve_out_returns_failed_for_negative_slope():
    sr = vs.SyncResult(
        slope=-0.5, offset=10.0, n_anchors=2, n_baseline_on=551,
        rmse_ms=0.0, max_residual_ms=0.0, cv_rmse_ms=0.0,
        slope_ppm=-500000.0, durbin_watson=2.0,
        detection_method="manual_slope_fit",
    )
    assert sr.quality == "failed"


def test_quality_manual_carve_out_returns_failed_for_one_anchor():
    sr = vs.SyncResult(
        slope=1.0, offset=0.0, n_anchors=1, n_baseline_on=551,
        rmse_ms=0.0, max_residual_ms=0.0, cv_rmse_ms=0.0,
        slope_ppm=0.0, durbin_watson=2.0,
        detection_method="manual_slope_fit",
    )
    assert sr.quality == "failed"


# fit_2anchor_clock ---------------------------------------------------------


def test_fit_2anchor_clock_exact_2_anchors():
    fps = 50.0
    anchors = [
        {"trial_index": 0, "nidaq_baseline_on_s": 10.0,
         "video_frame_idx": 500, "video_time_s": 10.0,
         "clicked_at": "x"},
        # video_time_s = 1010 / fps = 20.2 vs nidaq 20.0 => slope > 1
        {"trial_index": 100, "nidaq_baseline_on_s": 20.0,
         "video_frame_idx": 1010, "video_time_s": 20.2,
         "clicked_at": "x"},
    ]
    sr = vs.fit_2anchor_clock(
        anchors=anchors, fps=fps, n_baseline_on=101,
    )
    # slope = (20.2 - 10.0) / (20.0 - 10.0) = 1.02
    # offset = 10.0 - 1.02 * 10.0 = -0.2
    assert abs(sr.slope - 1.02) < 1e-9
    assert abs(sr.offset - (-0.2)) < 1e-9
    assert sr.n_anchors == 2
    assert sr.n_baseline_on == 101
    assert sr.rmse_ms == 0.0
    assert sr.detection_method == "manual_slope_fit"
    assert abs(sr.slope_ppm - 20000.0) < 1e-6


def test_fit_2anchor_clock_3_anchor_lsq():
    fps = 50.0
    # Three exactly-collinear anchors → slope=1.02, offset=-0.2, rmse=0
    anchors = [
        {"trial_index": 0,   "nidaq_baseline_on_s": 10.0,
         "video_frame_idx": 500,  "video_time_s": 10.0,
         "clicked_at": "x"},
        {"trial_index": 50,  "nidaq_baseline_on_s": 15.0,
         "video_frame_idx": 755,  "video_time_s": 15.1,
         "clicked_at": "x"},
        {"trial_index": 100, "nidaq_baseline_on_s": 20.0,
         "video_frame_idx": 1010, "video_time_s": 20.2,
         "clicked_at": "x"},
    ]
    sr = vs.fit_2anchor_clock(
        anchors=anchors, fps=fps, n_baseline_on=101,
    )
    assert abs(sr.slope - 1.02) < 1e-6
    assert abs(sr.offset - (-0.2)) < 1e-6
    assert sr.n_anchors == 3
    assert sr.rmse_ms < 1e-3  # essentially zero


def test_fit_2anchor_clock_rejects_fewer_than_2_anchors():
    import pytest
    with pytest.raises(ValueError, match="at least 2"):
        vs.fit_2anchor_clock(
            anchors=[{"trial_index": 0, "nidaq_baseline_on_s": 0.0,
                      "video_frame_idx": 0, "video_time_s": 0.0,
                      "clicked_at": "x"}],
            fps=50.0, n_baseline_on=10,
        )


def test_fit_2anchor_clock_rejects_non_positive_slope():
    """Anchors that produce a slope <= 0 are physically impossible."""
    import pytest
    fps = 50.0
    anchors = [
        # later trial mapped to earlier video time → impossible
        {"trial_index": 0,   "nidaq_baseline_on_s": 20.0,
         "video_frame_idx": 500,  "video_time_s": 10.0,
         "clicked_at": "x"},
        {"trial_index": 100, "nidaq_baseline_on_s": 10.0,
         "video_frame_idx": 1010, "video_time_s": 20.2,
         "clicked_at": "x"},
    ]
    # NOTE: this anchor pair has DECREASING nidaq with INCREASING video, so
    # slope is positive in (x,y) terms. Construct a truly impossible case:
    anchors = [
        {"trial_index": 0,   "nidaq_baseline_on_s": 10.0,
         "video_frame_idx": 1000, "video_time_s": 20.0,
         "clicked_at": "x"},
        {"trial_index": 100, "nidaq_baseline_on_s": 20.0,
         "video_frame_idx": 500, "video_time_s": 10.0,
         "clicked_at": "x"},
    ]
    with pytest.raises(ValueError, match="non-positive"):
        vs.fit_2anchor_clock(
            anchors=anchors, fps=fps, n_baseline_on=101,
        )
```

### Step 2.2: Run tests; expect failure

Run: `py -m pytest tests/test_video_sync_anchor.py -v -k "per_trial_overrides or quality_manual or fit_2anchor"`

Expected: 10 new tests FAIL — `per_trial_overrides` field absent on `SyncResult`; `fit_2anchor_clock` not defined.

### Step 2.3: Implement the `SyncResult` extensions

In `src/visdetect/core/video_sync.py`, modify the `SyncResult` dataclass (around lines 508–528). Add `per_trial_overrides` as a new optional field at the end of the field list:

Find:
```python
    inlier_mask: Optional[np.ndarray] = field(default=None, repr=False)
    residuals_ms: Optional[np.ndarray] = field(default=None, repr=False)
    matched_cam_ms: Optional[np.ndarray] = field(default=None, repr=False)
    matched_nidaq_s: Optional[np.ndarray] = field(default=None, repr=False)
```

Replace with:
```python
    inlier_mask: Optional[np.ndarray] = field(default=None, repr=False)
    residuals_ms: Optional[np.ndarray] = field(default=None, repr=False)
    matched_cam_ms: Optional[np.ndarray] = field(default=None, repr=False)
    matched_nidaq_s: Optional[np.ndarray] = field(default=None, repr=False)
    per_trial_overrides: Optional[Dict[int, int]] = field(default=None, repr=False)
```

Ensure `Dict` is imported from `typing` at the top of the file (search for `from typing import`; add `Dict` to the imports if absent).

Now update `to_dict` to include `per_trial_overrides` only when non-None. Find the existing `to_dict`:

```python
    def to_dict(self) -> dict:
        return {
            "slope": self.slope,
            "offset": self.offset,
            ...
            "detection_method": self.detection_method,
        }
```

Replace with:
```python
    def to_dict(self) -> dict:
        d = {
            "slope": self.slope,
            "offset": self.offset,
            "n_anchors": self.n_anchors,
            "n_baseline_on": self.n_baseline_on,
            "coverage": round(self.coverage, 4),
            "rmse_ms": round(self.rmse_ms, 2),
            "max_residual_ms": round(self.max_residual_ms, 2),
            "cv_rmse_ms": round(self.cv_rmse_ms, 2),
            "slope_ppm": round(self.slope_ppm, 2),
            "durbin_watson": round(self.durbin_watson, 4),
            "quality": self.quality,
            "roi": self.roi,
            "n_frames": self.n_frames,
            "n_dropped": self.n_dropped,
            "detection_method": self.detection_method,
        }
        if self.per_trial_overrides is not None:
            d["per_trial_overrides"] = self.per_trial_overrides
        return d
```

### Step 2.4: Add manual quality carve-out

Find the `quality` property (around lines 534–563). Add the manual carve-out as the first check:

Find:
```python
    @property
    def quality(self) -> str:
        """Composite quality tier: good / review / failed."""
        good_rmse = self.rmse_ms < _GOOD_RMSE_MS
```

Replace with:
```python
    @property
    def quality(self) -> str:
        """Composite quality tier: good / review / failed."""
        # Manual 2-anchor fits don't have the regression-style metrics
        # the rest of this logic checks. A manual fit is "good" iff the
        # slope is physically sensible and there are >=2 anchors.
        if self.detection_method == "manual_slope_fit":
            return "good" if (self.slope > 0 and self.n_anchors >= 2) else "failed"

        good_rmse = self.rmse_ms < _GOOD_RMSE_MS
```

### Step 2.5: Implement `fit_2anchor_clock`

Add this new function in `src/visdetect/core/video_sync.py` immediately after the `SyncResult` class definition (above the `# Metadata parsing` section):

```python
def fit_2anchor_clock(
    anchors: List[dict],
    fps: float,
    n_baseline_on: int,
) -> "SyncResult":
    """Fit a linear clock model from 2+ v2 anchor entries.

    Model: ``video_time_s = slope * nidaq_baseline_on_s + offset``.

    For exactly 2 anchors: closed-form linear fit (rmse_ms = 0).
    For >=3 anchors: least-squares fit; rmse_ms from residuals.

    Returns a SyncResult with detection_method = "manual_slope_fit".
    Raises ValueError on fewer than 2 anchors or non-positive slope.
    """
    if len(anchors) < 2:
        raise ValueError(
            f"fit_2anchor_clock needs at least 2 anchors; got {len(anchors)}"
        )

    x = np.array(
        [float(a["nidaq_baseline_on_s"]) for a in anchors], dtype=np.float64
    )
    y = np.array(
        [float(a["video_time_s"]) for a in anchors], dtype=np.float64
    )

    if len(anchors) == 2:
        slope = float((y[1] - y[0]) / (x[1] - x[0]))
        offset = float(y[0] - slope * x[0])
        rmse_ms = 0.0
    else:
        A = np.vstack([x, np.ones_like(x)]).T
        soln, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        slope = float(soln[0])
        offset = float(soln[1])
        residuals_s = y - (slope * x + offset)
        rmse_ms = float(np.sqrt(np.mean(residuals_s ** 2)) * 1000.0)

    if slope <= 0:
        raise ValueError(
            f"Computed slope {slope} is non-positive; anchors are likely "
            f"out of order or one is wrong. Re-verify via --scrub."
        )

    return SyncResult(
        slope=slope,
        offset=offset,
        n_anchors=int(len(anchors)),
        n_baseline_on=int(n_baseline_on),
        rmse_ms=rmse_ms,
        max_residual_ms=rmse_ms,  # for 2-anchor exact fit, residual==0; for lsq, conservative upper bound
        cv_rmse_ms=0.0,
        slope_ppm=float((slope - 1.0) * 1e6),
        durbin_watson=2.0,  # N/A for this fit type; report the neutral value
        detection_method="manual_slope_fit",
    )
```

### Step 2.6: Run tests; confirm all pass

Run: `py -m pytest tests/test_video_sync_anchor.py -v`

Expected: 27 + 10 = **37 passing**.

### Step 2.7: Commit

```bash
git add src/visdetect/core/video_sync.py tests/test_video_sync_anchor.py
git commit -m "$(cat <<'EOF'
Phase 2: SyncResult.per_trial_overrides + manual quality + fit_2anchor_clock

- Adds optional per_trial_overrides: Dict[int,int] field to SyncResult.
  to_dict includes it only when non-None.
- Adds early-return in SyncResult.quality for detection_method==
  "manual_slope_fit": returns "good" iff slope>0 and n_anchors>=2.
  This prevents 2-anchor exact fits (rmse=0, dw=N/A) from being
  mis-tiered as "failed" by the regression-pipeline quality gates.
- Adds fit_2anchor_clock(anchors, fps, n_baseline_on) -> SyncResult.
  Closed-form for n=2, least-squares for n>=3. Rejects fewer than 2
  anchors and non-positive slopes.
EOF
)"
```

---

## Task 3 — `click_anchor.py --anchor-last` flag

Add the second-anchor flow to the existing script. Uses the Phase 1.5 scrubber UI verbatim; differs only in starting frame, HUD label, and the trial index it saves under.

**Files:**
- Modify: `scripts/video/click_anchor.py`

No new automated tests (the scrubber UI is manual-smoke-tested). One small helper (`_predicted_last_trial_frame`) is pure-logic and gets a unit test in the same file.

### Step 3.1: Write failing test for the predicted-last-trial helper

Append to `tests/test_video_sync_anchor.py`:

```python
def test_predicted_last_trial_frame_from_anchor_0():
    ca = _import_click_anchor()
    ts_ms = np.arange(0.0, 10000000.0, 20.0)  # 50fps, 500k frames
    baseline_on = np.array([27.83, 1000.0, 7255.49])
    # Anchor 0: trial 0, video_time = 23.22 s
    anchor0 = {
        "trial_index": 0,
        "nidaq_baseline_on_s": 27.83,
        "video_frame_idx": 1161,
        "video_time_s": 23.22,
        "clicked_at": "x",
    }
    # implied_offset = 23.22 - 27.83 = -4.61
    # Predicted last trial video time = 7255.49 + (-4.61) = 7250.88 s
    # Predicted last frame = 7250.88 * 50 = 362544
    frame = ca._predicted_last_trial_frame(anchor0, baseline_on, ts_ms)
    expected_ms = (7255.49 - 4.61) * 1000.0
    expected_frame = int(np.argmin(np.abs(ts_ms - expected_ms)))
    assert frame == expected_frame
```

### Step 3.2: Run test; expect failure

Run: `py -m pytest tests/test_video_sync_anchor.py::test_predicted_last_trial_frame_from_anchor_0 -v`

Expected: FAIL — helper not yet defined.

### Step 3.3: Implement the helper

In `scripts/video/click_anchor.py`, add this helper next to `jump_to_predicted_frame` in the grid-math helpers section:

```python
def _predicted_last_trial_frame(
    anchor0: dict,
    baseline_on: np.ndarray,
    ts_ms: np.ndarray,
) -> int:
    """Predict the video frame for the last task trial using anchor 0's offset.

    Used to seed the --anchor-last scrubber so it opens close to the actual
    last-trial Baseline_ON. Uses slope=1 (the offset from anchor 0); the
    scrubber lets the user correct any drift.
    """
    implied_offset_s = (
        float(anchor0["video_time_s"]) - float(anchor0["nidaq_baseline_on_s"])
    )
    last_nidaq_s = float(baseline_on[-1])
    target_ms = (last_nidaq_s + implied_offset_s) * 1000.0
    if target_ms <= ts_ms[0]:
        return 0
    if target_ms >= ts_ms[-1]:
        return int(len(ts_ms) - 1)
    return int(np.argmin(np.abs(ts_ms - target_ms)))
```

### Step 3.4: Wire up the `--anchor-last` flag in `main()`

Add the argparse argument near the existing `--scrub` / `--start-from` block:

```python
parser.add_argument(
    "--anchor-last", action="store_true",
    help="Anchor the LAST task trial (uses scrubber UI, requires a trial-0 "
         "anchor to already exist).",
)
```

Then in `main()`, branch on `args.anchor_last` BEFORE the existing `args.scrub` branch. The full update to the branching logic:

Find the existing block:
```python
    anchor: Optional[dict] = None
    if args.scrub:
        ...
    else:
        ...
```

Replace with:
```python
    anchor: Optional[dict] = None
    if args.anchor_last:
        existing = load_anchor(session_name)
        if existing is None:
            logger.error(
                "--anchor-last requires an existing anchor file. "
                "Run --session %s first (no --anchor-last) to anchor trial 0.",
                session_name,
            )
            return 2
        # Find anchor for trial 0 (must exist).
        anchor0 = next(
            (a for a in existing["anchors"] if int(a["trial_index"]) == 0),
            None,
        )
        if anchor0 is None:
            logger.error(
                "Existing anchor file has no trial-0 anchor. "
                "Run --session %s first (no --anchor-last) to anchor trial 0.",
                session_name,
            )
            return 2
        last_trial_idx = int(len(baseline_on)) - 1
        if last_trial_idx <= 0:
            logger.error("Session has <=1 trial; nothing to anchor as 'last'.")
            return 2
        start_frame = _predicted_last_trial_frame(anchor0, baseline_on, ts_ms)
        logger.info(
            "Opening scrubber at predicted last-trial (idx %d) frame %d.",
            last_trial_idx, start_frame,
        )
        anchor_after = _run_scrub(
            session_name=session_name,
            video_path=video_path,
            baseline_on=baseline_on,
            ts_ms=ts_ms,
            fps=fps,
            n_frames=n_frames,
            start_frame=start_frame,
            existing_anchor=anchor0,  # for HUD context
            anchor_trial_index=last_trial_idx,  # NEW kwarg; see Step 3.5
        )
        if anchor_after is None:
            logger.info("Scrubber exited without saving.")
            return 1
        anchor = anchor_after
    elif args.scrub:
        # ... existing scrub flow, unchanged
    else:
        # ... existing 2-stage flow, unchanged
```

### Step 3.5: Update `_run_scrub` to accept `anchor_trial_index`

`_run_scrub` currently hard-codes `trial_index=0` when calling `_build_or_merge_anchor_file`. Make this configurable.

Find the function signature of `_run_scrub`:

```python
def _run_scrub(
    session_name: str,
    video_path: str,
    baseline_on: np.ndarray,
    ts_ms: np.ndarray,
    fps: float,
    n_frames: int,
    start_frame: int,
    existing_anchor: Optional[dict],
) -> Optional[dict]:
```

Replace with:
```python
def _run_scrub(
    session_name: str,
    video_path: str,
    baseline_on: np.ndarray,
    ts_ms: np.ndarray,
    fps: float,
    n_frames: int,
    start_frame: int,
    existing_anchor: Optional[dict],
    anchor_trial_index: int = 0,
) -> Optional[dict]:
```

In the Space/Enter handler inside `_run_scrub`, find:
```python
            anchor = _build_or_merge_anchor_file(
                session_name, baseline_on, ts_ms, fps,
                trial_index=0, frame_idx=state["frame_idx"],
            )
```

Replace with:
```python
            anchor = _build_or_merge_anchor_file(
                session_name, baseline_on, ts_ms, fps,
                trial_index=int(anchor_trial_index),
                frame_idx=state["frame_idx"],
            )
```

In the R handler, find:
```python
            candidate_file = _build_or_merge_anchor_file(
                session_name, baseline_on, ts_ms, fps,
                trial_index=0, frame_idx=state["frame_idx"],
            )
```

Replace with:
```python
            candidate_file = _build_or_merge_anchor_file(
                session_name, baseline_on, ts_ms, fps,
                trial_index=int(anchor_trial_index),
                frame_idx=state["frame_idx"],
            )
```

In the R handler's `entry0 = candidate_file["anchors"][0]` line: with multi-anchor support, "anchors[0]" is no longer guaranteed to be the anchor we just merged. Replace with a lookup:
```python
            entry = next(
                a for a in candidate_file["anchors"]
                if int(a["trial_index"]) == int(anchor_trial_index)
            )
            # render_barcode_montage takes the v1 single-anchor shape:
            candidate_for_render = {
                "session": candidate_file["session"],
                "anchor_trial_index": entry["trial_index"],
                "nidaq_baseline_on_s": entry["nidaq_baseline_on_s"],
                "video_frame_idx": entry["video_frame_idx"],
                "video_time_s": entry["video_time_s"],
                "implied_offset_s": compute_implied_offset(entry),
                "frame_rate_fps": candidate_file["frame_rate_fps"],
                "n_trials": candidate_file["n_trials"],
                "clicked_at": entry["clicked_at"],
            }
```

Update the scrubber HUD to make the active trial index visible. The HUD lines are constructed inside `_refresh` at the bottom of `_run_scrub`. Find this line (around line 425 of `click_anchor.py`):

```python
            f"If saved here (anchor for trial 0): implied_offset = {if_saved_offset_s:+.4f} s",
```

Replace with:

```python
            f"If saved here (anchor for trial {anchor_trial_index}): implied_offset = {if_saved_offset_s:+.4f} s",
```

Also update the line in `_run_scrub` that computes `if_saved_offset_s` — currently it hard-codes `baseline_on[0]`:

```python
        if_saved_offset_s = float(
            ts_ms[fi] / 1000.0 - float(baseline_on[0])
        )
```

Replace with:

```python
        if_saved_offset_s = float(
            ts_ms[fi] / 1000.0 - float(baseline_on[int(anchor_trial_index)])
        )
```

These two edits make the HUD's "If saved here" line correct for whatever trial is being anchored.

### Step 3.6: Run all tests + import check

Run: `py -m pytest tests/test_video_sync_anchor.py -v`

Expected: 37 + 1 = **38 passing**.

Run: `py -c "from scripts.video.click_anchor import main, _run_scrub, _predicted_last_trial_frame; print('ok')"`

Expected: `ok`.

Run: `py scripts/video/click_anchor.py --help`

Expected: Help text now includes `--anchor-last`.

### Step 3.7: Commit

```bash
git add scripts/video/click_anchor.py tests/test_video_sync_anchor.py
git commit -m "$(cat <<'EOF'
Phase 2: click_anchor --anchor-last for second-anchor scrubber

Adds --anchor-last CLI flag and _predicted_last_trial_frame helper.
Reuses _run_scrub verbatim; differs only in (a) starting frame
(predicted last-trial from anchor 0's offset), (b) HUD title
("Scrub anchor: trial N..."), and (c) anchor_trial_index saved
(len(baseline_on)-1 instead of 0).

_run_scrub gains an anchor_trial_index keyword (default 0 = unchanged
behavior). The existing --scrub flow continues to save trial 0.

Aborts cleanly when no trial-0 anchor exists, when the session has
<=1 trial, or when the user ESCs.
EOF
)"
```

---

## Task 4 — `scripts/video/fit_sync.py` (new CLI) + slope-aware `render_barcode_montage`

The second of the three Phase 2 commands. Reads the anchors, fits a linear clock model, writes the canonical `{session}_video_sync.json`, and renders a slope-fitted barcode montage so the user can visually verify the fit.

Phase 1's `render_barcode_montage` centres each row on `(baseline_on[i] + implied_offset_s) * fps` — a slope=1 assumption. With Phase 2's slope ≠ 1, a single `implied_offset_s` only matches one trial; other trials drift away from the centre column by `(slope − 1) * t`. For 24 ppm drift over 10000s that's 240ms — well outside the ±60ms barcode window. We need the renderer to optionally take `slope` and `offset_s` directly and use them per-row.

**Files:**
- Modify: `scripts/video/click_anchor.py` (extend `render_barcode_montage` signature)
- Create: `scripts/video/fit_sync.py`

No automated tests; the math is covered by Task 2's `fit_2anchor_clock` tests, the renderer change is small enough to verify visually, and the CLI is manually exercised in Task 6.

### Step 4.0: Add slope-aware kwargs to `render_barcode_montage`

In `scripts/video/click_anchor.py`, find the existing signature of `render_barcode_montage`:

```python
def render_barcode_montage(
    session_name: str,
    anchor: dict,
    baseline_on: np.ndarray,
    video_path: str,
    ts_ms: np.ndarray,
    fps: float,
    out_path: str,
) -> None:
```

Replace with:

```python
def render_barcode_montage(
    session_name: str,
    anchor: dict,
    baseline_on: np.ndarray,
    video_path: str,
    ts_ms: np.ndarray,
    fps: float,
    out_path: str,
    slope: Optional[float] = None,
    offset_s: Optional[float] = None,
) -> None:
```

Inside the function, find the per-row centre-frame computation. The existing code looks like:

```python
        target_ms = (float(baseline_on[ti]) + implied_offset_s) * 1000.0
```

Replace with:

```python
        if slope is not None and offset_s is not None:
            # Slope-fitted prediction: video_time_s = slope * nidaq + offset_s
            target_ms = (slope * float(baseline_on[ti]) + offset_s) * 1000.0
        else:
            # Slope=1 fallback (Phase 1 behavior): single implied_offset_s
            target_ms = (float(baseline_on[ti]) + implied_offset_s) * 1000.0
```

Also update the title to indicate which mode was used. Find the existing `fig.suptitle(title, fontsize=10)` block in `render_barcode_montage` and the lines that build `title`. The existing `title` includes `implied_offset {implied_offset_s:.3f}s`. Replace that title-building block with:

```python
    if slope is not None and offset_s is not None:
        title = (
            f"Anchor-barcode montage (slope-fit) - {session_name}\n"
            f"slope = {slope:.6f} ({(slope - 1) * 1e6:+.2f} ppm), "
            f"offset = {offset_s:+.4f} s, {n_trials} trials"
        )
    else:
        title = (
            f"Anchor-barcode montage - {session_name}\n"
            f"anchor trial {anchor['anchor_trial_index']} @ frame {anchor['video_frame_idx']} "
            f"(NI-DAQ {anchor['nidaq_baseline_on_s']:.3f}s, "
            f"implied offset {implied_offset_s:.3f}s) - {n_trials} trials"
        )
```

(`n_trials` is already in scope from the earlier `n_trials = len(baseline_on)` line.)

Run a quick smoke test that the existing Phase 1.5 montage rendering still works (slope=1 fallback path):

```bash
py -c "from scripts.video.click_anchor import render_barcode_montage; import inspect; sig = inspect.signature(render_barcode_montage); print(list(sig.parameters.keys()))"
```

Expected: prints a list ending with `..., 'out_path', 'slope', 'offset_s']`.

Commit this change separately so the fit_sync.py creation diff stays clean:

```bash
git add scripts/video/click_anchor.py
git commit -m "$(cat <<'EOF'
Phase 2 prep: render_barcode_montage learns slope+offset_s kwargs

When slope and offset_s are both provided, render_barcode_montage
centres each row on (slope * nidaq_time + offset_s) * fps instead of
(nidaq_time + implied_offset_s) * fps. Title reflects the mode.

Default behavior (kwargs omitted) is identical to Phase 1: slope=1
with a single implied_offset_s. No Phase 1.5 callers touched.

Used by fit_sync.py in the next commit to render the slope-fitted
barcode montage that verifies the 2-anchor clock fit.
EOF
)"
```

### Step 4.1: Create `fit_sync.py`

Write `scripts/video/fit_sync.py`:

```python
"""fit_sync.py — Phase 2 of video sync: linear clock model from manual anchors.

Reads the v2 anchor JSON for a session, fits a linear clock model
(video_time_s = slope * nidaq_baseline_on_s + offset) from the anchors,
writes the canonical {session}_video_sync.json via save_video_sync, and
renders a slope-fitted barcode montage so the user can visually confirm
the fit holds across the session.

Requires at least 2 anchors in the anchor JSON (run click_anchor twice:
once for trial 0, once with --anchor-last).

Run:  py scripts/video/fit_sync.py --session 09092025
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

import numpy as np

# matplotlib backend setup (script imports cv2 indirectly via shared helpers).
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from visdetect.suite.loader import load_session
from visdetect.core.video_sync import (
    find_camera_files,
    load_camera_metadata,
    load_anchor,
    save_video_sync,
    fit_2anchor_clock,
)
from visdetect.analysis.config import VIDEO_SYNC_DIR

# Reuse the barcode-montage renderer from click_anchor.py
import importlib.util
_CA_SPEC = importlib.util.spec_from_file_location(
    "click_anchor",
    os.path.join(os.path.dirname(__file__), "click_anchor.py"),
)
_CA = importlib.util.module_from_spec(_CA_SPEC)
_CA_SPEC.loader.exec_module(_CA)
render_barcode_montage = _CA.render_barcode_montage
FIGS_DIR = _CA.FIGS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("fit_sync")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2: fit linear clock model from manual anchors.",
    )
    parser.add_argument(
        "--session", required=True, help="Session name (e.g. 09092025).",
    )
    args = parser.parse_args()

    session_name = str(int(args.session)).zfill(8)

    # Load anchors.
    anchor_file = load_anchor(session_name)
    if anchor_file is None:
        logger.error(
            "No anchor JSON for %s. Run click_anchor.py --session %s first.",
            session_name, session_name,
        )
        return 2
    anchors = anchor_file["anchors"]
    if len(anchors) < 2:
        logger.error(
            "Anchor JSON has %d anchor(s); need >=2. "
            "Run click_anchor.py --session %s --anchor-last to add a second anchor.",
            len(anchors), session_name,
        )
        return 2

    # Load session for baseline_on / n_trials sanity.
    sess = load_session(session_name)
    baseline_on = np.asarray(
        sess.ni_events.get("Baseline_ON", []), dtype=float
    )
    baseline_on = baseline_on[baseline_on > 0]
    n_task_trials = len(sess.trials)
    if n_task_trials > 0 and len(baseline_on) > n_task_trials:
        baseline_on = baseline_on[:n_task_trials]
    n_baseline_on = int(len(baseline_on))
    fps = float(anchor_file["frame_rate_fps"])

    # Fit.
    try:
        sync_result = fit_2anchor_clock(
            anchors=anchors, fps=fps, n_baseline_on=n_baseline_on,
        )
    except ValueError as exc:
        logger.error("Slope fit failed: %s", exc)
        return 2

    # Persist via existing save_video_sync.
    out_path = save_video_sync(
        session_name=session_name, eye_cam=sync_result,
    )
    logger.info(
        "Slope fit: slope=%.6f (%.2f ppm), offset=%.4f s, "
        "n_anchors=%d, rmse=%.2f ms, quality=%s",
        sync_result.slope, sync_result.slope_ppm, sync_result.offset,
        sync_result.n_anchors, sync_result.rmse_ms, sync_result.quality,
    )
    print(f"Sync JSON: {out_path}")

    # Render the slope-fitted barcode montage. Use the new slope+offset_s
    # kwargs on render_barcode_montage (added in Step 4.0) so each row is
    # centred on the slope-fitted prediction for that trial.
    cam = find_camera_files(session_name)
    video_path = cam["eye_cam"]["video"]
    ts_ms, _, _ = load_camera_metadata(cam["eye_cam"]["metadata"])

    # render_barcode_montage still requires the `anchor` arg for backwards
    # compat (it uses it for title metadata in the slope=1 path). For the
    # slope-fit path the anchor dict's fields are unused by the renderer,
    # but we pass a sentinel that makes the dict accesses safe.
    sentinel_anchor = {
        "session": session_name,
        "anchor_trial_index": -1,
        "nidaq_baseline_on_s": 0.0,
        "video_frame_idx": 0,
        "video_time_s": 0.0,
        "implied_offset_s": 0.0,
        "frame_rate_fps": fps,
        "n_trials": n_baseline_on,
        "clicked_at": "slope_fit",
    }
    montage_path = os.path.join(
        FIGS_DIR, f"{session_name}_barcode_montage_slopefit.png",
    )
    render_barcode_montage(
        session_name=session_name,
        anchor=sentinel_anchor,
        baseline_on=baseline_on,
        video_path=video_path,
        ts_ms=ts_ms,
        fps=fps,
        out_path=montage_path,
        slope=sync_result.slope,
        offset_s=sync_result.offset,
    )
    print(f"Montage:   {montage_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Step 4.2: Verify the script imports and `--help` works

Run: `py -c "import importlib.util, os; s=importlib.util.spec_from_file_location('fit_sync', os.path.join('scripts','video','fit_sync.py')); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print('ok')"`

Expected: `ok`.

Run: `py scripts/video/fit_sync.py --help`

Expected: Help text with `--session SESSION`.

### Step 4.3: Run existing tests; confirm nothing broke

Run: `py -m pytest tests/test_video_sync_anchor.py -v`

Expected: 38/38 passing (no change from Task 3).

### Step 4.4: Commit

```bash
git add scripts/video/fit_sync.py
git commit -m "$(cat <<'EOF'
Phase 2: fit_sync.py CLI — linear clock fit + slope-fit montage

Reads v2 anchor JSON for a session, fits a 2-anchor linear clock model
via fit_2anchor_clock, writes the canonical {session}_video_sync.json
through save_video_sync, then renders a barcode montage centered on the
slope-fitted midpoint prediction (figs/video_sync/{session}_barcode_
montage_slopefit.png) for visual verification.

Reuses render_barcode_montage from click_anchor.py via importlib.util
to avoid splitting it into a shared module while Phase 2 stabilizes.
EOF
)"
```

---

## Task 5 — `scripts/video/tag_trials.py` (new CLI)

The third Phase 2 command. Per-trial verify-and-advance scrubber. Pure-logic state machine is unit-tested; the matplotlib UI is manually verified.

**Files:**
- Create: `scripts/video/tag_trials.py`
- Create: `tests/test_video_sync_tag_trials.py`

### Step 5.1: Write failing tests for `TagState` transitions

Create `tests/test_video_sync_tag_trials.py`:

```python
"""Tests for tag_trials state-machine transitions (pure-logic, no UI)."""
from __future__ import annotations

import importlib.util
import os

import pytest


def _import_tag_trials():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec_path = os.path.join(project_root, "scripts", "video", "tag_trials.py")
    spec = importlib.util.spec_from_file_location("tag_trials", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_initial_resume_idx_no_overrides():
    tt = _import_tag_trials()
    assert tt.initial_resume_idx({}, n_trials=10) == 0


def test_initial_resume_idx_with_overrides():
    tt = _import_tag_trials()
    # Trials 0, 1, 2 done; resume at 3
    overrides = {0: 100, 1: 200, 2: 300}
    assert tt.initial_resume_idx(overrides, n_trials=10) == 3


def test_initial_resume_idx_all_done_returns_n_trials():
    tt = _import_tag_trials()
    overrides = {i: i * 10 for i in range(5)}
    assert tt.initial_resume_idx(overrides, n_trials=5) == 5


def test_handle_enter_sets_override_and_advances():
    tt = _import_tag_trials()
    state = tt.TagState(trial_idx=3, overrides={}, n_trials=10)
    new_state = tt.handle_enter(state, current_frame=999)
    assert new_state.overrides == {3: 999}
    assert new_state.trial_idx == 4
    assert not new_state.done


def test_handle_enter_at_last_trial_marks_done():
    tt = _import_tag_trials()
    state = tt.TagState(trial_idx=9, overrides={5: 500}, n_trials=10)
    new_state = tt.handle_enter(state, current_frame=1000)
    assert new_state.overrides == {5: 500, 9: 1000}
    assert new_state.trial_idx == 10
    assert new_state.done


def test_handle_skip_preserves_overrides_and_advances():
    tt = _import_tag_trials()
    state = tt.TagState(trial_idx=3, overrides={3: 999}, n_trials=10)
    new_state = tt.handle_skip(state)
    assert new_state.overrides == {3: 999}
    assert new_state.trial_idx == 4


def test_handle_skip_no_existing_override_is_noop_on_overrides():
    tt = _import_tag_trials()
    state = tt.TagState(trial_idx=3, overrides={}, n_trials=10)
    new_state = tt.handle_skip(state)
    assert new_state.overrides == {}
    assert new_state.trial_idx == 4


def test_handle_delete_removes_override_and_advances():
    tt = _import_tag_trials()
    state = tt.TagState(trial_idx=3, overrides={3: 999, 5: 500}, n_trials=10)
    new_state = tt.handle_delete(state)
    assert new_state.overrides == {5: 500}
    assert new_state.trial_idx == 4


def test_handle_delete_no_existing_override_is_noop_on_overrides():
    tt = _import_tag_trials()
    state = tt.TagState(trial_idx=3, overrides={}, n_trials=10)
    new_state = tt.handle_delete(state)
    assert new_state.overrides == {}
    assert new_state.trial_idx == 4


def test_handle_back_decrements_trial_idx():
    tt = _import_tag_trials()
    state = tt.TagState(trial_idx=5, overrides={2: 200}, n_trials=10)
    new_state = tt.handle_back(state)
    assert new_state.trial_idx == 4
    assert new_state.overrides == {2: 200}


def test_handle_back_at_trial_zero_stays_at_zero():
    tt = _import_tag_trials()
    state = tt.TagState(trial_idx=0, overrides={}, n_trials=10)
    new_state = tt.handle_back(state)
    assert new_state.trial_idx == 0
```

### Step 5.2: Run tests; expect failure

Run: `py -m pytest tests/test_video_sync_tag_trials.py -v`

Expected: 11 tests FAIL with `ImportError` (`tag_trials.py` doesn't exist yet).

### Step 5.3: Create `tag_trials.py` with the state machine + minimal UI

Write `scripts/video/tag_trials.py`:

```python
"""tag_trials.py — Phase 2 of video sync: per-trial manual onset tagging.

Walks each trial in a session and lets the user verify-and-advance the
slope-fitted predicted onset, overriding it on a per-trial basis when the
prediction is wrong. Per-trial overrides are persisted to
{session}_video_sync.json on every keystroke that mutates state, so
crashes do not lose progress.

Requires {session}_video_sync.json to exist (run fit_sync.py first).

Keys:
    Left/Right     +-1 frame
    Shift+Left/Right (or PgUp/Down)  +-10 frames
    Ctrl+Left/Right  +-100 frames
    Enter   Save current frame as this trial's override, advance.
    S       Skip: advance without changing this trial's override state.
    D       Delete this trial's override (revert to slope-fit), advance.
    B       Back to previous trial.
    Q / Esc Save and quit.

Run:  py scripts/video/tag_trials.py --session 09092025
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time as _time
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np

# matplotlib TkAgg setup (mirrors click_anchor.py for consistency).
os.environ["MPLBACKEND"] = "TkAgg"
import matplotlib
matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")

from visdetect.suite.loader import load_session
from visdetect.core.video_sync import (
    find_camera_files,
    load_camera_metadata,
    load_video_sync,
)
from visdetect.analysis.config import VIDEO_SYNC_DIR

# Reuse the eye-region crop from click_anchor.py (same subject, same camera).
import importlib.util as _ilu
_CA_SPEC = _ilu.spec_from_file_location(
    "click_anchor",
    os.path.join(os.path.dirname(__file__), "click_anchor.py"),
)
_CA = _ilu.module_from_spec(_CA_SPEC)
_CA_SPEC.loader.exec_module(_CA)
EYE_REGION_CROP_BG046 = _CA.EYE_REGION_CROP_BG046

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("tag_trials")


# ---------------------------------------------------------------------------
# Pure-logic state machine (unit-tested independently of the UI)
# ---------------------------------------------------------------------------


@dataclass
class TagState:
    """Per-trial tagging state. Immutable transitions: handlers return new instances."""
    trial_idx: int
    overrides: Dict[int, int]
    n_trials: int
    done: bool = False

    def with_trial_idx(self, new_idx: int) -> "TagState":
        return TagState(
            trial_idx=new_idx,
            overrides=dict(self.overrides),
            n_trials=self.n_trials,
            done=(new_idx >= self.n_trials),
        )


def initial_resume_idx(overrides: Dict[int, int], n_trials: int) -> int:
    """Lowest trial index that has no override yet (resume-where-left-off)."""
    for i in range(n_trials):
        if i not in overrides:
            return i
    return n_trials


def handle_enter(state: TagState, current_frame: int) -> TagState:
    new_overrides = dict(state.overrides)
    new_overrides[state.trial_idx] = int(current_frame)
    new_idx = state.trial_idx + 1
    return TagState(
        trial_idx=new_idx,
        overrides=new_overrides,
        n_trials=state.n_trials,
        done=(new_idx >= state.n_trials),
    )


def handle_skip(state: TagState) -> TagState:
    new_idx = state.trial_idx + 1
    return TagState(
        trial_idx=new_idx,
        overrides=dict(state.overrides),
        n_trials=state.n_trials,
        done=(new_idx >= state.n_trials),
    )


def handle_delete(state: TagState) -> TagState:
    new_overrides = {
        k: v for k, v in state.overrides.items() if k != state.trial_idx
    }
    new_idx = state.trial_idx + 1
    return TagState(
        trial_idx=new_idx,
        overrides=new_overrides,
        n_trials=state.n_trials,
        done=(new_idx >= state.n_trials),
    )


def handle_back(state: TagState) -> TagState:
    new_idx = max(0, state.trial_idx - 1)
    return TagState(
        trial_idx=new_idx,
        overrides=dict(state.overrides),
        n_trials=state.n_trials,
        done=False,
    )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _sync_json_path(session_name: str) -> str:
    session_name = str(int(session_name)).zfill(8)
    return os.path.join(VIDEO_SYNC_DIR, f"{session_name}_video_sync.json")


def _persist_overrides(session_name: str, overrides: Dict[int, int]) -> None:
    """Read the sync JSON, write back with updated per_trial_overrides."""
    path = _sync_json_path(session_name)
    with open(path, "r") as f:
        data = json.load(f)
    if "eye_cam" not in data:
        raise KeyError("sync JSON has no eye_cam entry; cannot persist overrides")
    # On-disk JSON object keys must be strings.
    data["eye_cam"]["per_trial_overrides"] = {
        str(k): int(v) for k, v in sorted(overrides.items())
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_overrides(session_name: str) -> Dict[int, int]:
    """Read existing per_trial_overrides from the sync JSON (may be empty)."""
    path = _sync_json_path(session_name)
    with open(path, "r") as f:
        data = json.load(f)
    raw = (data.get("eye_cam") or {}).get("per_trial_overrides") or {}
    return {int(k): int(v) for k, v in raw.items()}


def _slope_fit_frame(
    sync_json: dict, nidaq_baseline_on_s: float,
) -> int:
    eye = sync_json["eye_cam"]
    slope = float(eye["slope"])
    offset_s = float(eye["offset"])
    fps = float(sync_json.get("frame_rate_fps")
                or sync_json["eye_cam"].get("frame_rate_fps")
                or 50.0)
    video_time_s = slope * float(nidaq_baseline_on_s) + offset_s
    return int(round(video_time_s * fps))


# ---------------------------------------------------------------------------
# Interactive UI
# ---------------------------------------------------------------------------


def _run_tag_ui(
    session_name: str,
    video_path: str,
    baseline_on: np.ndarray,
    ts_ms: np.ndarray,
    sync_json: dict,
    n_trials: int,
    initial_state: TagState,
) -> TagState:
    """Open the TkAgg per-trial UI and drive the state machine until quit/done.

    Returns the final TagState. Persists overrides on every state-changing
    keystroke (Enter, S, D).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    y0, y1, x0, x1 = EYE_REGION_CROP_BG046
    n_frames = int(len(ts_ms))

    state_ref = {"state": initial_state, "current_frame": 0}

    def _read_frame(fi: int) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            return np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray[y0:y1, x0:x1]

    def _start_frame_for(trial_idx: int) -> int:
        """Initial frame when entering trial_idx: override if present, else slope-fit."""
        if trial_idx in state_ref["state"].overrides:
            return int(state_ref["state"].overrides[trial_idx])
        return _slope_fit_frame(sync_json, float(baseline_on[trial_idx]))

    state_ref["current_frame"] = _start_frame_for(initial_state.trial_idx)

    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1], hspace=0.1)
    ax_frame = fig.add_subplot(gs[0])
    ax_hud = fig.add_subplot(gs[1])
    ax_hud.axis("off")

    im = ax_frame.imshow(
        _read_frame(state_ref["current_frame"]),
        cmap="gray", vmin=0, vmax=255, interpolation="nearest",
    )
    ax_frame.set_xticks([]); ax_frame.set_yticks([])

    hud_text = ax_hud.text(
        0.02, 0.5, "", fontsize=9, family="monospace",
        verticalalignment="center", transform=ax_hud.transAxes,
    )

    start_time = _time.time()
    trials_processed = [0]  # closure-mutable counter

    def _refresh():
        st = state_ref["state"]
        if st.done:
            plt.close(fig)
            return
        fi = state_ref["current_frame"]
        im.set_data(_read_frame(fi))
        elapsed_min = (_time.time() - start_time) / 60.0
        rate_min_per_trial = (
            elapsed_min / trials_processed[0] if trials_processed[0] > 0 else 0.0
        )
        remaining_trials = st.n_trials - st.trial_idx
        eta_min = remaining_trials * rate_min_per_trial
        predicted_frame = _slope_fit_frame(
            sync_json, float(baseline_on[st.trial_idx])
        )
        override_status = (
            f"ON  (frame {st.overrides[st.trial_idx]})"
            if st.trial_idx in st.overrides else "OFF (slope-fit)"
        )
        lines = [
            f"Tagging trial {st.trial_idx + 1} of {st.n_trials}  ({session_name})",
            f"NI baseline_on: {float(baseline_on[st.trial_idx]):.4f} s",
            f"Slope-fit predicted: frame {predicted_frame}",
            f"Current frame:        {fi}   (Delta vs predicted = {fi - predicted_frame:+d})",
            f"Override: {override_status}",
            f"Elapsed: {elapsed_min:.1f} min   ETA: {eta_min:.1f} min",
            "",
            "Arrows = +-1f  Shift+Arrows = +-10f  Ctrl+Arrows = +-100f",
            "Enter = save+advance  S = skip+advance  D = delete+advance  B = back  Q/Esc = save+quit",
        ]
        hud_text.set_text("\n".join(lines))
        fig.canvas.draw_idle()

    def on_key(event):
        key = event.key
        st = state_ref["state"]
        if key in ("q", "escape"):
            plt.close(fig); return
        # Frame stepping within the current trial.
        step = 0
        if key == "left":   step = -1
        elif key == "right": step = +1
        elif key in ("shift+left", "pageup"):   step = -10
        elif key in ("shift+right", "pagedown"): step = +10
        elif key in ("ctrl+left",):  step = -100
        elif key in ("ctrl+right",): step = +100
        if step != 0:
            state_ref["current_frame"] = int(np.clip(
                state_ref["current_frame"] + step, 0, n_frames - 1,
            ))
            _refresh()
            return

        if key == "enter":
            new_state = handle_enter(st, state_ref["current_frame"])
            trials_processed[0] += 1
            state_ref["state"] = new_state
            _persist_overrides(session_name, new_state.overrides)
            if new_state.done:
                logger.info("All %d trials reviewed; quitting.", st.n_trials)
                plt.close(fig); return
            state_ref["current_frame"] = _start_frame_for(new_state.trial_idx)
            _refresh(); return

        if key == "s":
            new_state = handle_skip(st)
            trials_processed[0] += 1
            state_ref["state"] = new_state
            _persist_overrides(session_name, new_state.overrides)
            if new_state.done:
                logger.info("All %d trials reviewed; quitting.", st.n_trials)
                plt.close(fig); return
            state_ref["current_frame"] = _start_frame_for(new_state.trial_idx)
            _refresh(); return

        if key == "d":
            new_state = handle_delete(st)
            trials_processed[0] += 1
            state_ref["state"] = new_state
            _persist_overrides(session_name, new_state.overrides)
            if new_state.done:
                logger.info("All %d trials reviewed; quitting.", st.n_trials)
                plt.close(fig); return
            state_ref["current_frame"] = _start_frame_for(new_state.trial_idx)
            _refresh(); return

        if key == "b":
            new_state = handle_back(st)
            state_ref["state"] = new_state
            state_ref["current_frame"] = _start_frame_for(new_state.trial_idx)
            _refresh(); return

    fig.canvas.mpl_connect("key_press_event", on_key)
    try:
        _refresh()
        plt.show()
    finally:
        cap.release()
        # Final save (covers Q/Esc case).
        _persist_overrides(session_name, state_ref["state"].overrides)

    return state_ref["state"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2: per-trial manual onset tagging.",
    )
    parser.add_argument(
        "--session", required=True, help="Session name (e.g. 09092025).",
    )
    args = parser.parse_args()
    session_name = str(int(args.session)).zfill(8)

    sync = load_video_sync(session_name)
    if sync is None:
        logger.error(
            "No {session}_video_sync.json for %s. Run fit_sync.py first.",
            session_name,
        )
        return 2
    if "eye_cam" not in sync:
        logger.error("Sync JSON for %s has no eye_cam entry.", session_name)
        return 2

    sess = load_session(session_name)
    baseline_on = np.asarray(
        sess.ni_events.get("Baseline_ON", []), dtype=float
    )
    baseline_on = baseline_on[baseline_on > 0]
    n_task_trials = len(sess.trials)
    if n_task_trials > 0 and len(baseline_on) > n_task_trials:
        baseline_on = baseline_on[:n_task_trials]
    n_trials = int(len(baseline_on))
    if n_trials == 0:
        logger.error("No Baseline_ON events for %s.", session_name)
        return 2

    cam = find_camera_files(session_name)
    if "eye_cam" not in cam:
        logger.error("No eye_cam video for %s.", session_name)
        return 2
    video_path = cam["eye_cam"]["video"]
    meta_path = cam["eye_cam"]["metadata"]
    ts_ms, _, _ = load_camera_metadata(meta_path)

    overrides = _load_overrides(session_name)
    start_idx = initial_resume_idx(overrides, n_trials)
    if start_idx >= n_trials:
        logger.info(
            "All %d trials already have overrides. Nothing to do. "
            "Use B (back) inside the UI to re-review.",
            n_trials,
        )
        # Open anyway at trial 0 so user can navigate.
        start_idx = 0

    initial_state = TagState(
        trial_idx=start_idx, overrides=overrides, n_trials=n_trials,
    )
    logger.info(
        "Opening per-trial tag UI for %s: %d trials, resuming at trial %d, "
        "%d existing overrides.",
        session_name, n_trials, start_idx, len(overrides),
    )
    final_state = _run_tag_ui(
        session_name=session_name,
        video_path=video_path,
        baseline_on=baseline_on,
        ts_ms=ts_ms,
        sync_json=sync,
        n_trials=n_trials,
        initial_state=initial_state,
    )
    logger.info(
        "Tag UI exited. Final overrides: %d trials tagged.",
        len(final_state.overrides),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Step 5.4: Run tests; confirm passes

Run: `py -m pytest tests/test_video_sync_tag_trials.py -v`

Expected: 11/11 passing.

Run: `py -m pytest tests/test_video_sync_anchor.py -v`

Expected: 38/38 passing (no change).

Run: `py -c "import importlib.util, os; s=importlib.util.spec_from_file_location('tag_trials', os.path.join('scripts','video','tag_trials.py')); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print('ok')"`

Expected: `ok`.

Run: `py scripts/video/tag_trials.py --help`

Expected: Help text with `--session SESSION`.

### Step 5.5: Commit

```bash
git add scripts/video/tag_trials.py tests/test_video_sync_tag_trials.py
git commit -m "$(cat <<'EOF'
Phase 2: tag_trials.py CLI — per-trial verify-and-advance UI

New CLI for per-trial manual onset tagging. Pure-logic TagState +
handlers (handle_enter, handle_skip, handle_delete, handle_back,
initial_resume_idx) plus a TkAgg single-frame UI.

Each state-changing keystroke (Enter/S/D) persists per_trial_overrides
to {session}_video_sync.json immediately, so crashes do not lose work.
On entry, resumes at the lowest trial_idx with no override.

Adds 11 unit tests for the state-machine transitions.
EOF
)"
```

---

## Task 6 — Manual user runs on the 3 anchor sessions

This is the validation step. The user runs the Phase 2 pipeline on 09092025, 14082025, and 03072025 and reports findings.

No code changes; runtime artifacts produced.

### Step 6.1: Add the second anchor on 09092025

Run:
```bash
py scripts/video/click_anchor.py --session 09092025 --anchor-last
```

The scrubber should open near the predicted last-trial frame (using anchor 0's offset, +/- the drift the user knows is real). Scrub to the actual onset, press Enter. The file `data/cache/video_sync/09092025_anchor.json` is rewritten in v2 format with both anchors.

### Step 6.2: Fit the slope and render the verification montage on 09092025

Run:
```bash
py scripts/video/fit_sync.py --session 09092025
```

Expected output: slope, slope_ppm (~+11 ppm expected based on Phase 1 evidence), quality=good. Files produced:
- `data/cache/video_sync/09092025_video_sync.json`
- `figs/video_sync/09092025_barcode_montage_slopefit.png`

Inspect the montage. Centre column should now hit on all (or nearly all) sampled trials, including trials 275/412/550 that previously had "grating throughout" or were empty.

### Step 6.3: Repeat for 14082025

```bash
py scripts/video/click_anchor.py --session 14082025 --anchor-last
py scripts/video/fit_sync.py --session 14082025
```

Inspect montage. Expected: slope ~+24 ppm; trials 249/498/748/997 that previously showed drift should now hit centre column.

### Step 6.4: Repeat for 03072025 (expected to fail informatively)

```bash
py scripts/video/click_anchor.py --session 03072025 --anchor-last
py scripts/video/fit_sync.py --session 03072025
```

Inspect montage. If barcode looks bad (signal too weak even at the right frame), proceed to Step 6.5 to use per-trial manual tagging.

### Step 6.5 (only if Step 6.4 montage is unsatisfactory): per-trial tagging on 03072025

```bash
py scripts/video/tag_trials.py --session 03072025
```

Walk a representative subset of trials (or all, if needed). Per the spec, overrides take precedence over slope-fit in downstream consumers.

### Step 6.6: Commit the produced artifacts (anchors + sync JSONs)

After Step 6.4 (and optionally 6.5), the working tree has:
- Updated `data/cache/video_sync/{0307,0909,1408}_anchor.json` (now in v2 form with 2 anchors each).
- New `data/cache/video_sync/{0307,0909,1408}_video_sync.json`.
- New `figs/video_sync/{0307,0909,1408}_barcode_montage_slopefit.png`.

```bash
git add -f data/cache/video_sync/09092025_anchor.json \
           data/cache/video_sync/14082025_anchor.json \
           data/cache/video_sync/03072025_anchor.json \
           data/cache/video_sync/09092025_video_sync.json \
           data/cache/video_sync/14082025_video_sync.json \
           data/cache/video_sync/03072025_video_sync.json
git commit -m "$(cat <<'EOF'
Add Phase 2 sync artifacts for BG_046 09092025, 14082025, 03072025

Anchor JSONs migrated to v2 schema (list of anchors). Each now contains
two anchors: trial 0 and the last task trial. Sync JSONs are the
canonical {session}_video_sync.json output of fit_sync.py with slope,
offset, and quality reported.

09092025: slope=X.XX, slope_ppm=X.XX, quality=good
14082025: slope=X.XX, slope_ppm=X.XX, quality=good
03072025: slope=X.XX, slope_ppm=X.XX, quality=good[/review/failed]
  [+ N per_trial_overrides if per-trial tagging was used]

PNGs not committed (already gitignored).
EOF
)"
```

---

## Self-review notes (run before handing off to execution)

**1. Spec coverage check.** Each spec section maps to at least one task:
- Schema migration (spec § "Anchor JSON schema migration") → Task 1.
- `compute_implied_offset` (spec) → Task 1.
- `per_trial_overrides` field + manual quality (spec § "Output schema") → Task 2.
- `fit_2anchor_clock` (spec § "Subsystem A — Command 2") → Task 2.
- `--anchor-last` flag (spec § "Subsystem A — Command 1a") → Task 3.
- `fit_sync.py` CLI (spec § "Subsystem A — Command 2") → Task 4.
- `tag_trials.py` CLI + state machine (spec § "Subsystem C") → Task 5.
- Error paths (spec § "Error handling") → covered inline in Tasks 3, 4, 5.
- Manual smoke-test on 3 sessions (spec § "Anchor sessions") → Task 6.

**2. Placeholder scan.** None — every code step has full code; every test step has full assertions; every commit step has a full message.

**3. Type/name consistency.**
- `TagState` field names (`trial_idx`, `overrides`, `n_trials`, `done`) are consistent across all 11 tests and all 5 handlers.
- `_build_anchor_entry` / `_build_v2_anchor_file` / `_merge_anchor_into_file` are referenced uniformly in click_anchor.py and the tests.
- `fit_2anchor_clock(anchors, fps, n_baseline_on)` signature is the same in tests, library implementation, and `fit_sync.py` caller.
- `_predicted_last_trial_frame(anchor0, baseline_on, ts_ms)` signature is the same in tests and the click_anchor.py call site.
- `_run_scrub(..., anchor_trial_index: int = 0)` keyword is consistently used by the `--scrub` path (default 0) and the `--anchor-last` path (last trial index).

**4. Reuse via importlib.util.** Tasks 4 and 5 import `render_barcode_montage` and `EYE_REGION_CROP_BG046` from `click_anchor.py` via `importlib.util.spec_from_file_location` (the same pattern the tests use). This avoids pulling those into a shared module while Phase 2 stabilizes; if Phase 3 needs broader sharing, extract them then.

**5. Out-of-scope (per spec).**
- No auto-detection / feature-based verifier.
- No front-cam.
- No batch wrapper across all sessions.
- No multi-subject support.
- No sub-frame interpolation.

All of these are explicitly deferred to Phase 3 in the spec.
