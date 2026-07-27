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


# ---------------------------------------------------------------------------
# Task 3: seed-from-archive migration
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Task 4: orientation-aware nidaq->frame
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Task 5: per-session eye-zoom crop
# ---------------------------------------------------------------------------
from visdetect.analysis import tagging as tg


def test_eye_zoom_crop_from_roi_and_fallback():
    assert tg.eye_zoom_crop(None) == (200, 420, 320, 540)
    y0, y1, x0, x1 = tg.eye_zoom_crop([300, 400, 500, 600], pad=0.0)
    assert (y0, y1, x0, x1) == (300, 400, 500, 600)
    yy0, yy1, xx0, xx1 = tg.eye_zoom_crop([300, 400, 500, 600], pad=0.10)
    assert yy0 == 290 and yy1 == 410 and xx0 == 490 and xx1 == 610  # 10% of 100 each side


# ---------------------------------------------------------------------------
# Task 2: v3 anchor-file builder
# ---------------------------------------------------------------------------


def test_build_v3_anchor_file_stamps_v3_and_baseline_event_type():
    ts_ms = np.arange(1000, dtype=float) * 20.0
    base = vs._build_anchor_entry(np.array([5.0, 9.0]), ts_ms, trial_index=0, frame_idx=100)
    f = vs._build_v3_anchor_file("01072025", fps=50.0, n_trials=2, anchor_entries=[base])
    assert f["schema_version"] == 3
    a = f["anchors"][0]
    assert a["event_type"] == "baseline_on"
    assert a["nidaq_event_s"] == a["nidaq_baseline_on_s"]


# ---------------------------------------------------------------------------
# Provisional change-clock seeding math (extracted from tag_session; UX design §8)
# ---------------------------------------------------------------------------


def test_provisional_change_clock_zero_anchors_uses_coarse_offset():
    slope, offset = tagging.provisional_change_clock([], coarse_offset_s=15.0)
    assert slope == 1.0
    assert offset == 15.0
    # nidaq = 1.0*cam + 15.0  ->  cam(nidaq=15.0) = 0.0 -> frame 0
    assert tagging.nidaq_to_frame_oriented(15.0, slope, offset, 50.0,
                                           "manual_multianchor") == 0


def test_provisional_change_clock_one_anchor_implied_offset_slope1():
    # change entry (nidaq_event_s only): offset = nidaq - video = 22 - 4 = 18
    a = {"event_type": "change_on", "nidaq_event_s": 22.0, "video_time_s": 4.0}
    slope, offset = tagging.provisional_change_clock([a], coarse_offset_s=15.0)
    assert slope == 1.0
    assert offset == 18.0
    # baseline entry (nidaq_baseline_on_s fallback) gives identical math
    b = {"event_type": "baseline_on", "nidaq_baseline_on_s": 22.0, "video_time_s": 4.0}
    assert tagging.provisional_change_clock([b], coarse_offset_s=15.0) == (1.0, 18.0)
    # feeding nidaq=22 back recovers the anchor's own frame: cam=4.0 -> frame 200 @50fps
    assert tagging.nidaq_to_frame_oriented(22.0, slope, offset, 50.0,
                                           "manual_multianchor") == 200


def test_provisional_change_clock_three_anchors_uses_multianchor_fit():
    # Exactly-collinear nidaq = 1.02*cam - 0.2  -> Theil-Sen recovers it exactly,
    # so the >=3-anchor fit path (not the slope-1.0 fallback) must be taken.
    anchors = [
        {"video_time_s": 10.0, "nidaq_event_s": 10.0},   # 1.02*10 - 0.2
        {"video_time_s": 15.0, "nidaq_event_s": 15.1},   # 1.02*15 - 0.2
        {"video_time_s": 20.0, "nidaq_event_s": 20.2},   # 1.02*20 - 0.2
    ]
    slope, offset = tagging.provisional_change_clock(anchors, coarse_offset_s=15.0)
    assert abs(slope - 1.02) < 1e-6
    assert abs(offset - (-0.2)) < 1e-6
    assert slope != 1.0  # proves the multianchor fit ran, not the offset fallback
    # nidaq=15.1 -> cam=(15.1+0.2)/1.02 = 15.0 -> frame 750 @50fps
    assert tagging.nidaq_to_frame_oriented(15.1, slope, offset, 50.0,
                                           "manual_multianchor") == 750
