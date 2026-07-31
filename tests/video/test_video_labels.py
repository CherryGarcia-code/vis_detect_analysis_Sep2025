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


# ---------------------------------------------------------------------------
# Task 2: cross-session ROI seeding (most-recent PRIOR by DATE)
# ---------------------------------------------------------------------------


def _write_prior(labels_dir, session, frame_size, eye_box):
    sc = vl.new_sidecar("BG_TEST", session, frame_size)
    vl.set_roi(sc, "eye", eye_box, source="drawn")
    vl.save_sidecar(sc, session, "BG_TEST", labels_dir=str(labels_dir))


def test_seed_picks_most_recent_prior_by_date_not_lexical(tmp_path):
    # Lexical max of the 8-digit strings would be '28062025' (28 Jun); the correct
    # most-recent PRIOR to 05072025 (5 Jul) is '01072025' (1 Jul, leading-zero day).
    _write_prior(tmp_path, "28062025", [976, 1024], [1, 1, 1, 1])
    _write_prior(tmp_path, "01072025", [976, 1024], [7, 7, 7, 7])
    res = vl.seed_rois_from_previous("05072025", "BG_TEST", (976, 1024),
                                     labels_dir=str(tmp_path))
    assert res is not None
    assert res["source_session"] == "01072025"     # date-based, not lexical
    assert res["applied"] is True
    assert res["rois"]["eye"] == {"box": [7, 7, 7, 7], "source": "inherited:01072025"}


def test_seed_never_picks_a_later_session(tmp_path):
    _write_prior(tmp_path, "09072025", [976, 1024], [1, 1, 1, 1])  # 9 Jul (later)
    assert vl.seed_rois_from_previous("05072025", "BG_TEST", (976, 1024),
                                      labels_dir=str(tmp_path)) is None


def test_seed_none_when_no_prior(tmp_path):
    assert vl.seed_rois_from_previous("05072025", "BG_TEST", (976, 1024),
                                      labels_dir=str(tmp_path)) is None


def test_seed_ddmmyy_six_digit_ids(tmp_path):
    # 6-digit DDMMYY (BG_031/039 form). canonical_camera_session maps both the
    # prior filename and the query to 8-digit; seeding compares by DATE.
    _write_prior(tmp_path, "080425", [976, 1024], [5, 5, 5, 5])   # 8 Apr 2025
    res = vl.seed_rois_from_previous("090425", "BG_TEST", (976, 1024),
                                     labels_dir=str(tmp_path))     # 9 Apr 2025
    assert res is not None
    assert res["source_session"] == "08042025"
    assert res["rois"]["eye"]["source"] == "inherited:08042025"


def test_seed_frame_size_mismatch_offers_but_does_not_apply(tmp_path):
    _write_prior(tmp_path, "01072025", [976, 1024], [7, 7, 7, 7])
    res = vl.seed_rois_from_previous("05072025", "BG_TEST", (500, 500),
                                     labels_dir=str(tmp_path))
    assert res is not None
    assert res["source_session"] == "01072025"
    assert res["applied"] is False                 # different resolution -> not applied
    assert res["frame_size"] == [976, 1024]
