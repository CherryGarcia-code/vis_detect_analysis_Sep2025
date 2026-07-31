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
