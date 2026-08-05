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
