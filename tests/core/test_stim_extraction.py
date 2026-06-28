import numpy as np
from visdetect.core.ingest import extract_stim_timeseries
from visdetect.core.session import Trial

def test_extract_stim_timeseries_parses_arrays():
    raw = {
        "vbl": [100.0, 100.0166, 100.0333],
        "TF": [0.0, 1.2, 0.8],
        "phase": [[0, 0], [10, 0], [25, 0]],
    }
    out = extract_stim_timeseries(raw)
    assert out["stim_vbl"].shape == (3,)
    assert out["stim_tf_disp"].shape == (3,)
    assert out["stim_phase"].shape == (3, 2)
    np.testing.assert_allclose(out["stim_vbl"][0], 100.0)

def test_extract_stim_timeseries_missing_keys_returns_none():
    out = extract_stim_timeseries({"trialoutcome": "Hit"})
    assert out["stim_phase"] is None and out["stim_vbl"] is None

def test_trial_has_new_fields_default_none():
    t = Trial()
    assert t.stim_phase is None and t.stim_tf_disp is None and t.stim_vbl is None
