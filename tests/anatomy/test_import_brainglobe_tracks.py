# tests/anatomy/test_import_brainglobe_tracks.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "anatomy"))
from import_brainglobe_tracks import brainglobe_npy_to_polyline, import_brainglobe_tracks
from visdetect.anatomy.tracks import load_track_artifact


def _bg_track(ml, dv_deep=4500.0, dv_shallow=1500.0, ap=5150.0, n=20, reversed_order=False):
    """A fake brainglobe-segmentation atlas_space spline: (n,3) in um, axis order
    (AP, DV, ML) — DV runs deep->shallow unless reversed_order."""
    dv = np.linspace(dv_deep, dv_shallow, n)
    if reversed_order:
        dv = dv[::-1]
    return np.stack([np.full(n, ap), dv, np.full(n, ml)], axis=1)


def test_npy_to_polyline_transforms_and_orders_deepest_first(tmp_path):
    p = tmp_path / "shankX.npy"
    np.save(p, _bg_track(ml=7400.0, reversed_order=True))  # shallow-first on disk
    poly = brainglobe_npy_to_polyline(p)
    assert poly.shape[1] == 3
    # our order is (AP, ML, DV); ML column should be the constant 7400
    np.testing.assert_allclose(poly[:, 1], 7400.0)
    # deepest-first: row 0 has the largest DV (most ventral)
    assert poly[0, 2] > poly[-1, 2]
    assert poly[0, 2] == 4500.0 and poly[-1, 2] == 1500.0


def test_import_assigns_indices_by_explicit_order_and_validates(tmp_path):
    # 4 shanks, medial->lateral ML, 250 um apart
    stems = ["s_med", "s2", "s3", "s_lat"]
    mls = [7167.0, 7417.0, 7667.0, 7917.0]
    for stem, ml in zip(stems, mls):
        np.save(tmp_path / f"{stem}.npy", _bg_track(ml=ml))
    out = tmp_path / "BG_046_shank_tracks.json"
    art = import_brainglobe_tracks(
        tmp_path, subject="BG_046", hemisphere="left",
        shank_order=stems, out_json=out, tip_y_um=0.0, sigma_um=50.0,
    )
    assert out.exists()
    loaded = load_track_artifact(out)  # re-validates schema + shank order
    assert loaded.hemisphere == "left"
    assert [s.probe_shank_index for s in loaded.shanks] == [0, 1, 2, 3]
    # index 0 is the medial (smallest-ML) track we listed first
    assert loaded.shanks[0].ccf_polyline[0, 1] == 7167.0
    assert loaded.shanks[3].ccf_polyline[0, 1] == 7917.0
    # all deepest-first, source recorded
    for s in loaded.shanks:
        assert s.ccf_polyline[0, 2] > s.ccf_polyline[-1, 2]
        assert s.method == "brainreg_traced"
    assert loaded.source_tool == "brainglobe-segmentation"


def test_import_reverse_order_list_flips_indices(tmp_path):
    # listing lateral-first must put the lateral track at index 0
    stems = ["s_med", "s2", "s3", "s_lat"]
    mls = [7167.0, 7417.0, 7667.0, 7917.0]
    for stem, ml in zip(stems, mls):
        np.save(tmp_path / f"{stem}.npy", _bg_track(ml=ml))
    out = tmp_path / "art.json"
    art = import_brainglobe_tracks(
        tmp_path, subject="BG_046", hemisphere="left",
        shank_order=list(reversed(stems)), out_json=out,
    )
    assert art.shanks[0].ccf_polyline[0, 1] == 7917.0  # lateral now index 0
    assert art.shanks[3].ccf_polyline[0, 1] == 7167.0
