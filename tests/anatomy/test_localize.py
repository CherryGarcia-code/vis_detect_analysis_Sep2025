# tests/anatomy/test_localize.py
import numpy as np
import pytest
from visdetect.anatomy.tracks import ShankTrack
from visdetect.anatomy.localize import (
    place_channel_on_track, region_confidence, build_channel_atlas,
)


def _straight_shank(idx=0, ml=1600.0):
    # deepest at DV=3500 (tip, y=0), top at DV=2500 (y=1000); straight in DV
    return ShankTrack(
        probe_shank_index=idx,
        ccf_polyline=np.array([[5000., ml, 3500.], [5000., ml, 2500.]]),
        tip_y_um=0.0, method="extended_from_tip",
        sigma_along_um=20., sigma_across_um=20., sigma_growth_k=0.1,
        planned_entry=None, planned_vector=np.array([0., 0., -1.0]),
    )

def test_place_within_polyline():
    sh = _straight_shank()
    xyz, sig = place_channel_on_track(sh, 500.0)  # halfway
    np.testing.assert_allclose(xyz, [5000., 1600., 3000.], atol=1e-6)
    assert sig == pytest.approx(20.0)

def test_place_extrapolates_above_with_growing_sigma():
    sh = _straight_shank()
    xyz, sig = place_channel_on_track(sh, 1200.0)  # 200 um above the top (y=1000)
    np.testing.assert_allclose(xyz, [5000., 1600., 2300.], atol=1e-6)
    assert sig == pytest.approx(40.0)  # 20 + 0.1 * 200

def test_region_confidence_monotonic():
    assert region_confidence(30., 5.) < region_confidence(30., 200.)
    assert 0.0 <= region_confidence(30., 5.) <= 1.0

def test_build_channel_atlas_columns_and_rows():
    from test_channel_geometry import _np2_positions  # bare: tests/anatomy on sys.path (prepend mode)
    from visdetect.anatomy.tracks import TrackArtifact
    from visdetect.anatomy.atlas import AllenAtlas
    pos = _np2_positions()
    art = TrackArtifact("BG_046", "allen_mouse_25um", "right", "forward",
                        "test", "2026-06-17",
                        [_straight_shank(i, ml=1350. + 250. * i) for i in range(4)])
    ann = np.ones((400, 200, 200), dtype=int)  # all region id 1
    atlas = AllenAtlas(annotation=ann, resolution_um=25.0,
                       id_to_acronym={1: "CP"}, id_to_name={1: "Caudoputamen"})
    df = build_channel_atlas("BG_046", art, pos, "sigABC", atlas)
    assert len(df) == len(pos)
    for c in ("ccf_ap", "ccf_ml", "ccf_dv", "region_acronym", "region_confidence",
              "shank", "loc_method", "chanmap_signature"):
        assert c in df.columns
    assert (df["chanmap_signature"] == "sigABC").all()
    assert set(df["shank"].unique()) == {0, 1, 2, 3}


def test_place_at_tip():
    sh = _straight_shank()
    xyz, sig = place_channel_on_track(sh, 0.0)
    np.testing.assert_allclose(xyz, [5000., 1600., 3500.], atol=1e-6)
    assert sig == pytest.approx(20.0)


def test_place_extrapolates_with_segment_direction_fallback():
    sh = _straight_shank()
    sh.planned_vector = None
    xyz, sig = place_channel_on_track(sh, 1200.0)
    np.testing.assert_allclose(xyz, [5000., 1600., 2300.], atol=1e-6)
    assert sig == pytest.approx(40.0)


def test_build_channel_atlas_raises_on_missing_shank():
    from test_channel_geometry import _np2_positions
    from visdetect.anatomy.tracks import TrackArtifact
    from visdetect.anatomy.atlas import AllenAtlas
    pos = _np2_positions()
    art = TrackArtifact("BG_046", "allen_mouse_25um", "right", "forward",
                        "test", "2026-06-17",
                        [_straight_shank(i, ml=1350. + 250. * i) for i in range(3)])  # only 3 shanks
    ann = np.ones((400, 200, 200), dtype=int)
    atlas = AllenAtlas(annotation=ann, resolution_um=25.0,
                       id_to_acronym={1: "CP"}, id_to_name={1: "Caudoputamen"})
    with pytest.raises(KeyError, match="shank index 3"):
        build_channel_atlas("BG_046", art, pos, "sig", atlas)
