# tests/anatomy/test_build_channel_atlas_cli.py
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "anatomy"))
from build_channel_atlas import build_subject_atlas
from visdetect.anatomy.tracks import TrackArtifact, ShankTrack, save_track_artifact
from visdetect.anatomy.atlas import AllenAtlas
from test_channel_geometry import _np2_positions  # bare: tests/anatomy on sys.path (prepend mode)

def _artifact():
    shanks = [ShankTrack(i, np.array([[5000., 1350.+250.*i, 3500.],
                                      [5000., 1350.+250.*i, 2500.]]),
                         0.0, "brainreg_traced", 20., 20., 0.0,
                         None, np.array([0., 0., -1.])) for i in range(4)]
    return TrackArtifact("BG_046", "allen_mouse_25um", "right", "forward",
                         "test", "2026-06-17", shanks)

def test_build_subject_atlas_writes_files(tmp_path):
    # two sessions, same geometry -> one signature
    raw = tmp_path / "raw"
    for s in ("01072025", "02072025"):
        d = raw / s; d.mkdir(parents=True)
        np.save(d / "channel_positions.npy", _np2_positions())
    art_p = tmp_path / "BG_046_shank_tracks.json"
    save_track_artifact(_artifact(), art_p)
    ann = np.ones((400, 200, 200), dtype=int)
    atlas = AllenAtlas(annotation=ann, resolution_um=25.0,
                       id_to_acronym={1: "CP"}, id_to_name={1: "Caudoputamen"})
    out = tmp_path / "anatomy"
    df = build_subject_atlas("BG_046", art_p, raw, ["01072025", "02072025"], atlas, out)
    assert (out / "BG_046_channel_atlas.csv").exists()
    sig_map = pd.read_csv(out / "BG_046_session_signatures.csv")
    assert sig_map["chanmap_signature"].nunique() == 1  # shared bank
    assert df["region_coarse"].eq("CP").all()

def test_session_token_and_resolve(tmp_path):
    from build_channel_atlas import session_token, resolve_session_dir
    assert session_token("BG_031_01042025") == "01042025"
    assert session_token("01072025") == "01072025"
    (tmp_path / "BG_031_01042025").mkdir()
    assert resolve_session_dir(tmp_path, "01042025") == "BG_031_01042025"
    assert resolve_session_dir(tmp_path, "99999999") is None
