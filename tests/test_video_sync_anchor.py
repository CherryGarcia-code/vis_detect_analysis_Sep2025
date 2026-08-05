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

    # load_anchor now migrates legacy anchors up to v3 in memory; compare form.
    assert loaded == vs._migrate_anchor_to_v3(anchor)


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

    assert loaded["schema_version"] == 3
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
    with pytest.raises(ValueError, match="at least 2"):
        vs.fit_2anchor_clock(
            anchors=[{"trial_index": 0, "nidaq_baseline_on_s": 0.0,
                      "video_frame_idx": 0, "video_time_s": 0.0,
                      "clicked_at": "x"}],
            fps=50.0, n_baseline_on=10,
        )


def test_fit_2anchor_clock_rejects_non_positive_slope():
    """Anchors that produce a slope <= 0 are physically impossible."""
    fps = 50.0
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


def test_fit_2anchor_clock_rejects_duplicate_nidaq_times():
    """Two anchors with identical nidaq_baseline_on_s must raise ValueError."""
    anchors = [
        {"trial_index": 0,  "nidaq_baseline_on_s": 10.0,
         "video_frame_idx": 1000, "video_time_s": 20.0,
         "clicked_at": "x"},
        {"trial_index": 50, "nidaq_baseline_on_s": 10.0,
         "video_frame_idx": 1010, "video_time_s": 20.2,
         "clicked_at": "x"},
    ]
    with pytest.raises(ValueError, match="same nidaq"):
        vs.fit_2anchor_clock(anchors=anchors, fps=50.0, n_baseline_on=100)


def test_fit_2anchor_clock_max_residual_exceeds_rmse_for_noncollinear():
    """For non-collinear anchors the max_residual_ms must be strictly > rmse_ms."""
    # Middle point is pulled off the line: nidaq=[10,15,20], video=[10.0,15.5,20.2]
    # The best-fit line through these three points has non-zero, unequal residuals.
    anchors = [
        {"trial_index": 0,   "nidaq_baseline_on_s": 10.0,
         "video_frame_idx": 500,  "video_time_s": 10.0,
         "clicked_at": "x"},
        {"trial_index": 50,  "nidaq_baseline_on_s": 15.0,
         "video_frame_idx": 775,  "video_time_s": 15.5,
         "clicked_at": "x"},
        {"trial_index": 100, "nidaq_baseline_on_s": 20.0,
         "video_frame_idx": 1010, "video_time_s": 20.2,
         "clicked_at": "x"},
    ]
    sr = vs.fit_2anchor_clock(anchors=anchors, fps=50.0, n_baseline_on=101)
    assert sr.rmse_ms > 0
    assert sr.max_residual_ms >= sr.rmse_ms
    assert sr.max_residual_ms > sr.rmse_ms  # strictly greater for non-collinear data


def test_fit_2anchor_clock_accepts_change_entry_via_nidaq_event_s():
    """A 2-anchor set including a CHANGE entry (``nidaq_event_s`` only, no
    ``nidaq_baseline_on_s``) must fit, not KeyError. The event-time read falls
    back to ``nidaq_event_s``, so the fit equals the all-baseline twin with the
    same (nidaq, video) pairs."""
    fps = 50.0
    mixed = [
        {"trial_index": 0, "event_type": "baseline_on", "nidaq_baseline_on_s": 10.0,
         "nidaq_event_s": 10.0, "video_frame_idx": 500, "video_time_s": 10.0,
         "clicked_at": "x"},
        {"trial_index": 3, "event_type": "change_on", "nidaq_event_s": 20.0,
         "video_frame_idx": 1010, "video_time_s": 20.2, "clicked_at": "x"},  # no baseline key
    ]
    sr = vs.fit_2anchor_clock(anchors=mixed, fps=fps, n_baseline_on=101)
    # slope = (20.2 - 10.0)/(20.0 - 10.0) = 1.02 ; offset = 10 - 1.02*10 = -0.2
    assert abs(sr.slope - 1.02) < 1e-9
    assert abs(sr.offset - (-0.2)) < 1e-9

    # Baseline-only twin with identical (nidaq, video) pairs fits byte-identically.
    baseline_only = [
        {"trial_index": 0, "nidaq_baseline_on_s": 10.0, "video_frame_idx": 500,
         "video_time_s": 10.0, "clicked_at": "x"},
        {"trial_index": 3, "nidaq_baseline_on_s": 20.0, "video_frame_idx": 1010,
         "video_time_s": 20.2, "clicked_at": "x"},
    ]
    sr_base = vs.fit_2anchor_clock(anchors=baseline_only, fps=fps, n_baseline_on=101)
    assert sr.slope == sr_base.slope
    assert sr.offset == sr_base.offset


def test_fit_2anchor_clock_rejects_anchor_missing_both_nidaq_keys():
    """An anchor with neither nidaq time key raises ValueError (which fit_sync
    catches) rather than a KeyError/TypeError that would escape uncaught."""
    anchors = [
        {"trial_index": 0, "nidaq_baseline_on_s": 10.0, "video_time_s": 10.0,
         "clicked_at": "x"},
        {"trial_index": 1, "video_time_s": 20.2, "clicked_at": "x"},  # no nidaq key
    ]
    with pytest.raises(ValueError, match="neither"):
        vs.fit_2anchor_clock(anchors=anchors, fps=50.0, n_baseline_on=101)


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
