# tests/anatomy/test_stereotaxic.py
import numpy as np

from visdetect.anatomy.stereotaxic import (
    BREGMA_AP_UM, MIDLINE_ML_UM, ap_to_bregma_mm, ml_to_lateral_mm,
    dv_to_depth_mm, pia_dv_um, CoordMap)
from visdetect.anatomy.tracks import TrackArtifact, ShankTrack


def test_ap_conversion_matches_paxinos_crosscheck():
    # project cross-check: BG_046 CCF-AP 5150um -> +0.25mm; BG_039 4921um -> ~+0.48mm
    assert abs(ap_to_bregma_mm(5150) - 0.25) < 1e-6
    assert abs(ap_to_bregma_mm(4921) - 0.479) < 0.01
    assert ap_to_bregma_mm(BREGMA_AP_UM) == 0.0


def test_ml_lateral_magnitude_and_sign():
    assert abs(ml_to_lateral_mm(7800)) == 2.1            # 2.1 mm from midline
    assert ml_to_lateral_mm(MIDLINE_ML_UM) == 0.0
    assert ml_to_lateral_mm(7800) > 0                    # CCF ML increases toward LEFT hemi


def test_dv_depth_below_pia():
    assert abs(dv_to_depth_mm(4000, 1000) - 3.0) < 1e-6


def test_pia_from_artifact_is_mean_surface():
    # polylines are deepest-first, so row -1 = surface; surfaces 1000/1100/1200/1300 -> 1150
    sh = [ShankTrack(i, np.array([[2500., 7000. + i, 5100.], [2500., 7000. + i, 1000. + i * 100]]),
                     0., "brainreg_traced", 30., 30., 0.) for i in range(4)]
    art = TrackArtifact("X", "allen_mouse_25um", "left", "forward", "t", "2026-06-25", sh)
    assert abs(pia_dv_um(art) - 1150.0) < 1e-6


def test_coordmap_ccf_is_identity():
    cm = CoordMap("ccf")
    assert cm.x(7800) == 7800 and cm.y(3000) == 3000 and cm.length(500) == 500
    img = np.zeros((4, 6, 3)); ext = [0, 1000, 800, 0]
    i2, e2 = cm.image(img, ext)
    assert np.array_equal(i2, img) and list(e2) == ext
    assert "µm" in cm.xlabel and "AP" in cm.ap_title(4921)


def test_coordmap_stereotaxic_flip_scale_and_labels():
    cm = CoordMap("stereotaxic", pia_dv_um=1000.0)
    assert cm.x(7800) < 0                       # left hemi -> negative -> drawn on the left
    assert abs(cm.x(7800) - (-2.1)) < 1e-6
    assert cm.x(MIDLINE_ML_UM) == 0.0
    assert abs(cm.y(4000) - 3.0) < 1e-6         # depth below pia (mm)
    assert cm.length(500) == 0.5
    img = np.arange(4 * 6 * 3).reshape(4, 6, 3); ext = [0, 11400, 8000, 0]
    i2, e2 = cm.image(img, ext)
    assert np.array_equal(i2, np.fliplr(img))   # mirrored L-R over the ML axis
    assert e2[0] < e2[1]                         # extent x increasing
    assert "mm" in cm.xlabel and "Bregma" in cm.ap_title(4921)


def test_coordmap_rejects_bad_mode():
    import pytest
    with pytest.raises(ValueError):
        CoordMap("paxinos")
