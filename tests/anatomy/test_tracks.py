import numpy as np
import pytest
from visdetect.anatomy.tracks import (
    ShankTrack, TrackArtifact, TrackArtifactError,
    load_track_artifact, save_track_artifact, validate_track_artifact,
)

def _shank(idx=0):
    return ShankTrack(
        probe_shank_index=idx,
        ccf_polyline=np.array([[5000., 1600., 3500.], [5000., 1600., 2500.]]),
        tip_y_um=0.0, method="brainreg_traced",
        sigma_along_um=30.0, sigma_across_um=30.0, sigma_growth_k=0.0,
        planned_entry=None, planned_vector=None,
    )

def _artifact():
    return TrackArtifact(
        subject="BG_046", atlas="allen_mouse_25um", hemisphere="right",
        barcode_orientation="forward", source_tool="brainglobe-segmentation",
        created="2026-06-17", shanks=[_shank(i) for i in range(4)],
    )

def test_roundtrip(tmp_path):
    art = _artifact()
    p = tmp_path / "BG_046_shank_tracks.json"
    save_track_artifact(art, p)
    loaded = load_track_artifact(p)
    assert loaded.subject == "BG_046"
    assert len(loaded.shanks) == 4
    np.testing.assert_allclose(loaded.shanks[0].ccf_polyline, art.shanks[0].ccf_polyline)

def test_validate_rejects_bad_method():
    art = _artifact()
    art.shanks[0].method = "guesswork"
    with pytest.raises(TrackArtifactError, match="method"):
        validate_track_artifact(art)

def test_validate_rejects_bad_orientation():
    art = _artifact()
    art.barcode_orientation = "sideways"
    with pytest.raises(TrackArtifactError, match="orientation"):
        validate_track_artifact(art)

def test_validate_rejects_wrong_polyline_shape():
    art = _artifact()
    art.shanks[0].ccf_polyline = np.zeros((3, 2))  # not (N,3)
    with pytest.raises(TrackArtifactError, match="polyline"):
        validate_track_artifact(art)

def test_load_validates(tmp_path):
    art = _artifact(); art.hemisphere = "middle"
    p = tmp_path / "bad.json"
    save_track_artifact(art, p)  # save does not validate
    with pytest.raises(TrackArtifactError):
        load_track_artifact(p)   # load does
