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


def test_save_sidecar_overwrites_stale_temp_and_leaves_none(tmp_path):
    # Fix 2: a pre-existing stale '<session>.json.tmp' (left by a hard kill) must
    # NOT prevent a successful save, and after a good save no temp remains.
    stale = tmp_path / "01072025.json.tmp"
    stale.write_text("leftover partial from a crashed save")
    sc = vl.new_sidecar("BG_TEST", "01072025", [10, 10])
    vl.set_roi(sc, "eye", [1, 2, 3, 4], source="drawn")
    vl.save_sidecar(sc, "01072025", "BG_TEST", labels_dir=str(tmp_path))
    loaded = vl.load_sidecar("01072025", "BG_TEST", labels_dir=str(tmp_path))
    assert loaded["rois"]["eye"] == {"box": [1, 2, 3, 4], "source": "drawn"}
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


# ---------------------------------------------------------------------------
# Fix 1 (MEDIUM): a corrupt/unreadable prior sidecar must NOT abort tagger
# startup. The most-recent prior is chosen by DATE, then loaded; a malformed
# winner must be SKIPPED and the next-most-recent VALID prior tried, never a
# raw JSONDecodeError propagating up through the (unguarded) GUI call.
# ---------------------------------------------------------------------------


def test_seed_skips_malformed_most_recent_and_falls_back_to_next(tmp_path):
    # 01 Jul is the most-recent prior but is CORRUPT; 28 Jun is the next-most-
    # recent VALID prior -> its ROIs are returned (no crash), tagged to 28 Jun.
    _write_prior(tmp_path, "28062025", [976, 1024], [3, 3, 3, 3])   # valid, older
    (tmp_path / "01072025.json").write_text("{ this is not valid json ]")
    res = vl.seed_rois_from_previous("05072025", "BG_TEST", (976, 1024),
                                     labels_dir=str(tmp_path))
    assert res is not None
    assert res["source_session"] == "28062025"      # fell back past the corrupt one
    assert res["applied"] is True
    assert res["rois"]["eye"] == {"box": [3, 3, 3, 3], "source": "inherited:28062025"}


def test_seed_malformed_only_prior_returns_none_no_crash(tmp_path):
    # The ONLY eligible prior is corrupt -> None, not a JSONDecodeError.
    (tmp_path / "01072025.json").write_text("{ nope")
    assert vl.seed_rois_from_previous("05072025", "BG_TEST", (976, 1024),
                                      labels_dir=str(tmp_path)) is None


def test_seed_garbage_prior_ignored_when_valid_more_recent_exists(tmp_path):
    # A garbage file that is OLDER than a valid winner must not perturb selection:
    # the valid more-recent prior is still chosen normally.
    (tmp_path / "20062025.json").write_text("\x00\x01 not json at all")  # older garbage
    _write_prior(tmp_path, "01072025", [976, 1024], [9, 9, 9, 9])        # valid winner
    res = vl.seed_rois_from_previous("05072025", "BG_TEST", (976, 1024),
                                     labels_dir=str(tmp_path))
    assert res is not None
    assert res["source_session"] == "01072025"
    assert res["rois"]["eye"] == {"box": [9, 9, 9, 9], "source": "inherited:01072025"}


# ---------------------------------------------------------------------------
# Task 3: crop clamp + ellipse geometry
# ---------------------------------------------------------------------------


def test_clamp_crop_negative_oversize():
    # partially outside but still intersecting: negatives -> 0, oversize -> H/W
    assert vl.clamp_crop((-30, 500, -20, 700), 480, 640) == (0, 480, 0, 640)
    # already valid -> unchanged
    assert vl.clamp_crop((100, 200, 150, 250), 480, 640) == (100, 200, 150, 250)


def test_clamp_crop_non_intersecting_returns_none():
    # box entirely below/right of the frame -> no intersection -> None
    assert vl.clamp_crop((500, 600, 700, 800), 480, 640) is None
    # box entirely in negative space (above/left of the frame) -> None
    assert vl.clamp_crop((-50, -10, -30, -5), 480, 640) is None
    # inverted (y1<y0 / x1<x0) is malformed -> None (NOT silently swapped: a swap
    # would invent an ROI the user never drew)
    assert vl.clamp_crop((300, 100, 400, 200), 480, 640) is None


def test_clamp_crop_partial_still_clamps_to_valid_nonempty():
    # partially past the bottom-right edge -> clamps to a valid, NON-EMPTY crop
    # (regression guard that the None path did not break the normal clamp path).
    out = vl.clamp_crop((400, 999, 500, 999), 480, 640)
    assert out == (400, 480, 500, 640)
    y0, y1, x0, x1 = out
    assert 0 <= y0 < y1 <= 480
    assert 0 <= x0 < x1 <= 640


def test_ellipse_from_box_axis_aligned():
    # y:100-200 (h=100), x:300-500 (w=200) -> wider than tall -> major=w, angle 0
    assert vl.ellipse_from_box((100, 200, 300, 500)) == {
        "cx": 400.0, "cy": 150.0, "major": 200.0, "minor": 100.0, "angle": 0.0}
    # y:100-400 (h=300), x:300-500 (w=200) -> taller than wide -> major=h, angle 90
    assert vl.ellipse_from_box((100, 400, 300, 500)) == {
        "cx": 400.0, "cy": 250.0, "major": 300.0, "minor": 200.0, "angle": 90.0}


def test_ellipse_from_box_inverted_normalizes_to_same_ellipse():
    # Fix 3: an inverted drag box yields the SAME ellipse as its normalized form
    # (an ellipse box is symmetric in intent -> safe to order-normalize).
    normalized = vl.ellipse_from_box((100, 200, 300, 500))
    assert vl.ellipse_from_box((200, 100, 500, 300)) == normalized
    assert normalized == {"cx": 400.0, "cy": 150.0, "major": 200.0,
                          "minor": 100.0, "angle": 0.0}


def test_ellipse_from_box_degenerate_raises():
    # Fix 3: zero-area (zero width or zero height) box is meaningless -> ValueError.
    with pytest.raises(ValueError):
        vl.ellipse_from_box((100, 100, 300, 500))   # zero height
    with pytest.raises(ValueError):
        vl.ellipse_from_box((100, 200, 300, 300))   # zero width
    with pytest.raises(ValueError):
        vl.ellipse_from_box((100, 100, 300, 300))   # zero area (point)


def test_ellipse_from_detection_maps_radius_to_circle():
    det = {"center_x": 512.0, "center_y": 480.0, "radius": 20.0,
           "area": 1200.0, "circularity": 0.9, "bbox": (460, 500, 492, 532)}
    assert vl.ellipse_from_detection(det) == {
        "cx": 512.0, "cy": 480.0, "major": 40.0, "minor": 40.0, "angle": 0.0}
    assert vl.ellipse_from_detection(None) is None


# ---------------------------------------------------------------------------
# Task (coordinate fix): image_extent_for_crop keeps the displayed image in
# FULL-FRAME data coords in BOTH views.
#
# This is the transform that produced THREE coordinate defects (drag rebasing,
# ellipse rebasing, and the underlying frozen-extent stretch) with zero test
# coverage. The scrubber shows a cropped array in the SAME imshow artist as the
# full frame; if the extent stays frozen at full-frame size, matplotlib
# STRETCHES the (crop_h, crop_w) array across the whole frame, so every ROI /
# pupil coordinate read off the axes is silently rescaled. These tests would
# FAIL if the extent were left full-frame (or otherwise stretched) for a crop.
# ---------------------------------------------------------------------------


def test_image_extent_full_frame_when_crop_none():
    # crop=None -> the default imshow extent for an H x W image.
    assert vl.image_extent_for_crop(None, 480, 640) == (-0.5, 639.5, 479.5, -0.5)


def test_image_extent_for_crop_is_full_frame_box_with_inverted_y():
    # crop (y0,y1,x0,x1) -> (left, right, bottom, top) = (x0-.5, x1-.5, y1-.5, y0-.5).
    # A crop OFFSET from the origin (the reviewer's eye_roi): the box, not (0,0).
    assert vl.image_extent_for_crop((200, 420, 320, 540), 480, 640) == (
        319.5, 539.5, 419.5, 199.5)
    # A genuinely NON-SQUARE crop, also offset (the two conditions that made the
    # old frozen-extent stretch measurable — a square crop at the origin hides it).
    assert vl.image_extent_for_crop((100, 300, 50, 500), 480, 640) == (
        49.5, 499.5, 299.5, 99.5)


def test_image_extent_round_trip_spans_exactly_crop_no_scale_or_offset():
    # Round-trip invariant: the extent must span EXACTLY x0..x1 / y0..y1 (in
    # full-frame pixel edges), so it introduces NO scale and NO offset. Under the
    # OLD frozen full-frame extent the span would be (640, 480) != crop dims.
    frame_h, frame_w = 480, 640
    for crop in [(200, 420, 320, 540),      # square, offset
                 (100, 300, 50, 500)]:       # non-square, offset
        y0, y1, x0, x1 = crop
        left, right, bottom, top = vl.image_extent_for_crop(crop, frame_h, frame_w)
        # Edges land on the crop's full-frame pixel borders.
        assert (left, right) == (x0 - 0.5, x1 - 0.5)
        assert (top, bottom) == (y0 - 0.5, y1 - 0.5)   # inverted-y (origin top)
        # Span equals the crop's full-frame width/height -> no stretch.
        assert (right - left) == pytest.approx(x1 - x0)
        assert (bottom - top) == pytest.approx(y1 - y0)


def test_image_extent_maps_cropped_pixel_to_true_fullframe_coord():
    # The crux regression, expressed as the imshow pixel->data map: the array
    # pixel that SHOWS a given full-frame point must read back that point's TRUE
    # full-frame data coord. The old frozen full-frame extent stretched the
    # (crop_h, crop_w) array across the whole frame, so the same pixel read back
    # a scaled+offset (wrong) coord -- e.g. the reviewer's true pupil (430, 310)
    # came out ~(607, 407).
    frame_h, frame_w = 480, 640
    crop = (200, 420, 320, 540)
    y0, y1, x0, x1 = crop
    crop_h, crop_w = y1 - y0, x1 - x0                    # 220 x 220
    left, right, bottom, top = vl.image_extent_for_crop(crop, frame_h, frame_w)

    px, py = 430.0, 310.0                                # true pupil, inside crop
    col = px - x0                                        # crop-local column (110)
    row = py - y0                                        # crop-local row (110)
    # imshow maps array pixel-centre col -> data x = left + (col+0.5)/ncols*(right-left)
    data_x = left + (col + 0.5) / crop_w * (right - left)
    data_y = top + (row + 0.5) / crop_h * (bottom - top)
    assert data_x == pytest.approx(px)                  # not ~607 (the old stretch)
    assert data_y == pytest.approx(py)                  # not ~407
