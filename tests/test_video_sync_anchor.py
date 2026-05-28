"""Tests for video sync anchor helpers (Phase 1 of corneal-barcode redesign)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from visdetect.core import video_sync as vs


# ---------------------------------------------------------------------------
# load_anchor / save_anchor round trip
# ---------------------------------------------------------------------------


def _make_anchor_dict() -> dict:
    return {
        "session": "00000123",
        "anchor_trial_index": 0,
        "nidaq_baseline_on_s": 12.3456,
        "video_frame_idx": 1047,
        "video_time_s": 20.94,
        "implied_offset_s": 8.5944,
        "frame_rate_fps": 50.0,
        "n_trials": 350,
        "clicked_at": "2026-05-27T14:32:10",
    }


def test_save_anchor_creates_json_at_expected_path(tmp_path):
    anchor = _make_anchor_dict()

    vs.save_anchor("123", anchor, sync_dir=str(tmp_path))

    expected = tmp_path / "00000123_anchor.json"
    assert expected.exists()
    payload = json.loads(expected.read_text())
    assert payload == anchor


def test_load_anchor_returns_saved_dict(tmp_path):
    anchor = _make_anchor_dict()
    vs.save_anchor("123", anchor, sync_dir=str(tmp_path))

    loaded = vs.load_anchor("123", sync_dir=str(tmp_path))

    assert loaded == anchor


def test_load_anchor_returns_none_when_missing(tmp_path):
    loaded = vs.load_anchor("99999999", sync_dir=str(tmp_path))

    assert loaded is None


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


def test_stage2_frame_indices_biased_backward():
    ca = _import_click_anchor()
    # Window is [click - 49, click], 50 cells, biased entirely backward.
    idx = ca.stage2_frame_indices(stage1_click=1000, fps=50.0, n_frames=5000)
    assert len(idx) == 50
    assert idx[0] == 1000 - 49
    assert idx[-1] == 1000


def test_stage2_frame_indices_clamps():
    ca = _import_click_anchor()
    idx = ca.stage2_frame_indices(stage1_click=10, fps=50.0, n_frames=5000)
    assert idx[0] == 0
    assert len(idx) == 50


def test_pick_sampled_trials_includes_first_and_last():
    ca = _import_click_anchor()
    picks = ca._pick_sampled_trials(100, 5)
    assert len(picks) == 5
    assert picks[0] == 0
    assert picks[-1] == 99
    # Evenly spaced (within rounding tolerance)
    diffs = [picks[i + 1] - picks[i] for i in range(len(picks) - 1)]
    assert max(diffs) - min(diffs) <= 1


def test_pick_sampled_trials_short_session():
    ca = _import_click_anchor()
    # n_trials < n_rows → returns range(n_trials)
    assert ca._pick_sampled_trials(3, 5) == [0, 1, 2]
    # n_trials == n_rows → also returns range(n_trials)
    assert ca._pick_sampled_trials(5, 5) == [0, 1, 2, 3, 4]


def test_jump_to_predicted_frame_basic():
    ca = _import_click_anchor()
    # 50 fps, 5000 frames covering 100 s
    ts_ms = np.arange(0.0, 100000.0, 20.0)
    baseline_on = np.array([5.0, 10.0, 15.0])  # NI-DAQ times
    # implied_offset = -4.0  → video_time = NI_time - 4.0
    # trial 0 predicted at video 1.0 s = frame 50
    frame = ca.jump_to_predicted_frame(0, baseline_on, -4.0, ts_ms)
    assert frame == 50
    # trial 1 predicted at video 6.0 s = frame 300
    frame = ca.jump_to_predicted_frame(1, baseline_on, -4.0, ts_ms)
    assert frame == 300


def test_jump_to_predicted_frame_clamps():
    ca = _import_click_anchor()
    ts_ms = np.arange(0.0, 1000.0, 20.0)  # 50 frames
    baseline_on = np.array([0.5, 100.0])
    # trial 0: video time = 0.5 - 10 = -9.5 s → clamp to 0
    assert ca.jump_to_predicted_frame(0, baseline_on, -10.0, ts_ms) == 0
    # trial 1: video time = 100 - 0 = 100 s → clamp to last
    assert ca.jump_to_predicted_frame(1, baseline_on, 0.0, ts_ms) == len(ts_ms) - 1


def test_jump_to_predicted_frame_out_of_range_raises():
    ca = _import_click_anchor()
    ts_ms = np.arange(0.0, 1000.0, 20.0)
    baseline_on = np.array([1.0, 2.0])
    with pytest.raises(IndexError):
        ca.jump_to_predicted_frame(2, baseline_on, 0.0, ts_ms)
    with pytest.raises(IndexError):
        ca.jump_to_predicted_frame(-1, baseline_on, 0.0, ts_ms)
