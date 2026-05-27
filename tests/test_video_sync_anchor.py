"""Tests for video sync anchor helpers (Phase 1 of corneal-barcode redesign)."""
from __future__ import annotations

import json

import numpy as np

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
