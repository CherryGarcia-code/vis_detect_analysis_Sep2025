# tests/anatomy/test_localize_units_cli.py
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "anatomy"))
from localize_units import localize_subject_units

def test_localize_units_joins_via_peak_channel(tmp_path):
    # channel atlas: 4 channels, channel 2 is CP at known coords
    atlas = pd.DataFrame({
        "subject": "BG_046", "chanmap_signature": "sigA",
        "channel": [0, 1, 2, 3], "shank": [0, 0, 0, 0],
        "x_um": [27., 27., 27., 27.], "y_um": [100., 200., 300., 400.],
        "ccf_ap": [5000.]*4, "ccf_ml": [1600.]*4, "ccf_dv": [3400., 3300., 3200., 3100.],
        "sigma_um": [20.]*4,
        "region_acronym": ["CP"]*4, "region_name": ["Caudoputamen"]*4,
        "region_coarse": ["CP"]*4, "region_confidence": [0.9]*4, "loc_method": ["brainreg_traced"]*4,
    })
    atlas_csv = tmp_path / "BG_046_channel_atlas.csv"; atlas.to_csv(atlas_csv, index=False)
    sig = pd.DataFrame({"session_name": ["01072025"], "chanmap_signature": ["sigA"]})
    sig_csv = tmp_path / "BG_046_session_signatures.csv"; sig.to_csv(sig_csv, index=False)
    # raw waveform for unit 42 peaking on channel 2
    rw = tmp_path / "01072025" / "RawWaveforms"; rw.mkdir(parents=True)
    raw = np.zeros((82, 4, 2)); raw[40, 2, :] = -5.; raw[50, 2, :] = 4.
    np.save(rw / "Unit42_RawSpikes.npy", raw)

    df = localize_subject_units("BG_046", atlas_csv, sig_csv, tmp_path,
                                {"01072025": [42]})
    assert len(df) == 1
    row = df.iloc[0]
    assert row["peak_channel"] == 2
    assert row["region_coarse"] == "CP"
    assert row["depth_um"] == 300.0
    assert row["ccf_dv"] == 3200.0
