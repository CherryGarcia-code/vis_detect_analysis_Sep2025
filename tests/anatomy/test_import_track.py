# tests/anatomy/test_import_track.py
import json
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "anatomy"))
from import_track import import_track
from visdetect.anatomy.tracks import load_track_artifact

def test_import_track_builds_valid_artifact(tmp_path):
    pts = []
    for i in range(4):
        ml = 1350. + 250. * i
        pts += [{"probe_shank_index": i, "point_order": 0, "ap_um": 5000., "ml_um": ml, "dv_um": 3500.},
                {"probe_shank_index": i, "point_order": 1, "ap_um": 5000., "ml_um": ml, "dv_um": 2500.}]
    pcsv = tmp_path / "BG_046_track_points.csv"; pd.DataFrame(pts).to_csv(pcsv, index=False)
    meta = {"subject": "BG_046", "hemisphere": "right", "barcode_orientation": "forward",
            "atlas": "allen_mouse_25um", "source_tool": "brainglobe-segmentation",
            "created": "2026-06-17",
            "shanks": {str(i): {"tip_y_um": 0.0, "method": "brainreg_traced",
                                "sigma_along_um": 25., "sigma_across_um": 25.,
                                "sigma_growth_k": 0.0,
                                "planned_entry": None, "planned_vector": [0, 0, -1]}
                       for i in range(4)}}
    mjson = tmp_path / "BG_046_track_meta.json"; mjson.write_text(json.dumps(meta))
    out = tmp_path / "BG_046_shank_tracks.json"
    art = import_track(pcsv, mjson, out)
    assert out.exists()
    loaded = load_track_artifact(out)   # re-validates
    assert len(loaded.shanks) == 4
    assert loaded.subject == "BG_046"
