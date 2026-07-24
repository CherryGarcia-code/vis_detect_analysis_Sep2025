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
