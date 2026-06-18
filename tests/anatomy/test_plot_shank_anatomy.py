# tests/anatomy/test_plot_shank_anatomy.py
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "anatomy"))
from plot_shank_anatomy import plot_subject_anatomy

def test_plot_produces_png(tmp_path):
    atlas = pd.DataFrame({
        "subject": "BG_046", "chanmap_signature": "sigA",
        "channel": range(8), "shank": [0]*4 + [1]*4,
        "x_um": [27.]*8, "y_um": [100., 200., 300., 400.]*2,
        "ccf_ap": [5000.]*8, "ccf_ml": [1600.]*8, "ccf_dv": [3400., 3300., 3200., 3100.]*2,
        "sigma_um": [20.]*8,
        "region_acronym": ["CP"]*8, "region_name": ["Caudoputamen"]*8,
        "region_coarse": ["CTX", "CTX", "CP", "CP"]*2,
        "region_confidence": [0.9]*8, "loc_method": ["brainreg_traced"]*8,
    })
    csv = tmp_path / "BG_046_channel_atlas.csv"; atlas.to_csv(csv, index=False)
    out = tmp_path / "fig.png"
    path = plot_subject_anatomy("BG_046", csv, out)
    assert os.path.exists(path)
