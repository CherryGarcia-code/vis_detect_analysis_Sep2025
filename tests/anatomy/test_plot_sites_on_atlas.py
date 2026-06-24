# tests/anatomy/test_plot_sites_on_atlas.py
import matplotlib
matplotlib.use("Agg")
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "anatomy"))
from plot_sites_on_atlas import plot_sites_on_atlas, coronal_coarse_image
from visdetect.anatomy.atlas import AllenAtlas
from visdetect.anatomy.tracks import TrackArtifact, ShankTrack, save_track_artifact


def _atlas():
    ann = np.ones((200, 160, 160), dtype=int)  # AP5000 x ML4000 x DV4000, all CP
    return AllenAtlas(annotation=ann, resolution_um=25.0, id_to_acronym={1: "CP"},
                      id_to_name={1: "Caudoputamen"}, id_to_coarse={1: "CP"})


def _inputs(tmp_path):
    mls = [1600.0, 1850.0, 2100.0, 2350.0]  # 250 um spacing -> passes validate_shank_order
    ch = pd.DataFrame({
        "subject": "BG_046", "chanmap_signature": "sig", "channel": range(8),
        "shank": [0, 0, 1, 1, 2, 2, 3, 3], "x_um": [27.0] * 8, "y_um": [1600.0] * 8,
        "ccf_ap": [2500.0] * 8,
        "ccf_ml": [1600, 1610, 1850, 1860, 2100, 2110, 2350, 2360],
        "ccf_dv": [2600.0] * 8, "sigma_um": [20.0] * 8,
        "region_acronym": ["CP"] * 8, "region_name": ["Caudoputamen"] * 8,
        "region_coarse": ["CP"] * 8, "region_confidence": [0.9] * 8,
        "loc_method": ["brainreg_traced"] * 8})
    acsv = tmp_path / "BG_046_channel_atlas.csv"; ch.to_csv(acsv, index=False)
    shanks = [ShankTrack(i, np.array([[2500., mls[i], 2900.], [2500., mls[i], 2000.]]),
                         0.0, "brainreg_traced", 30., 30., 0.0) for i in range(4)]
    art = TrackArtifact("BG_046", "allen_mouse_25um", "left", "forward", "t", "2026-06-24", shanks)
    tj = tmp_path / "BG_046_shank_tracks.json"; save_track_artifact(art, tj)
    return acsv, tj


def test_coronal_image_shape():
    img, ext = coronal_coarse_image(_atlas(), 2500.0)
    assert img.ndim == 3 and img.shape[2] == 3
    assert ext[1] > 0 and ext[2] > 0


def test_sites_figure_png(tmp_path):
    acsv, tj = _inputs(tmp_path)
    out = tmp_path / "f.png"
    assert os.path.exists(plot_sites_on_atlas("BG_046", acsv, tj, out, atlas=_atlas()))


def test_sites_figure_with_values_colorbar(tmp_path):
    acsv, tj = _inputs(tmp_path)
    out = tmp_path / "f2.png"
    p = plot_sites_on_atlas("BG_046", acsv, tj, out, atlas=_atlas(),
                            values=np.arange(8), value_label="rate (Hz)")
    assert os.path.exists(p)
