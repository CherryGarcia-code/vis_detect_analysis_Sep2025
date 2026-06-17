# tests/anatomy/test_orientation.py
import numpy as np
import pytest
from visdetect.anatomy.tracks import ShankTrack, TrackArtifact, TrackArtifactError
from visdetect.anatomy.orientation import assign_probe_shank_indices, validate_shank_order

def _shank_at_ml(ml):
    return ShankTrack(
        probe_shank_index=-1,
        ccf_polyline=np.array([[5000., ml, 3500.], [5000., ml, 2500.]]),
        tip_y_um=0.0, method="brainreg_traced",
        sigma_along_um=30., sigma_across_um=30., sigma_growth_k=0.,
    )

def _art(shanks, orientation="forward", hemi="right"):
    return TrackArtifact("BG_046", "allen_mouse_25um", hemi, orientation,
                         "test", "2026-06-17", shanks)

def test_forward_right_assigns_increasing_ml_to_increasing_index():
    shanks = [_shank_at_ml(ml) for ml in (1850, 1600, 2100, 1350)]  # unsorted
    out = assign_probe_shank_indices(shanks, "forward", "right")
    mls = [s.ccf_polyline[0, 1] for s in out]
    assert [s.probe_shank_index for s in out] == [0, 1, 2, 3]
    assert mls == sorted(mls)  # forward+right -> index 0 is most-medial (smallest ML)

def test_backward_reverses():
    shanks = [_shank_at_ml(ml) for ml in (1350, 1600, 1850, 2100)]
    out = assign_probe_shank_indices(shanks, "backward", "right")
    mls = [s.ccf_polyline[0, 1] for s in out]
    assert mls == sorted(mls, reverse=True)  # backward -> index 0 is most-lateral

def test_validate_passes_on_good_order():
    shanks = [_shank_at_ml(ml) for ml in (1350, 1600, 1850, 2100)]
    out = assign_probe_shank_indices(shanks, "forward", "right")
    validate_shank_order(_art(out))  # no raise

def test_validate_raises_on_nonmonotonic():
    shanks = [_shank_at_ml(ml) for ml in (1350, 1600, 1850, 2100)]
    out = assign_probe_shank_indices(shanks, "forward", "right")
    out[2].ccf_polyline[0, 1] = 1000.0  # break monotonicity
    with pytest.raises(TrackArtifactError, match="monoton|spacing"):
        validate_shank_order(_art(out))
