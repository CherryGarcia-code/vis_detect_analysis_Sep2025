import pandas as pd, pytest
from visdetect.analysis import decision_latents as dl

def test_load_state_labels_reads_trial_indexed_moods(tmp_path):
    d = tmp_path / "BG_046"; d.mkdir()
    pd.DataFrame({"trial_idx": [0, 1, 2],
                  "state_label": ["Impulsive", "StimSens", "Disengaged"],
                  "state_confidence": [0.9, 0.8, 0.95]}).to_csv(d / "01072025.csv", index=False)
    out = dl.load_state_labels("01072025", tag_dir=str(tmp_path))
    assert list(out.index) == [0, 1, 2]
    assert out.loc[1, "state_label"] == "StimSens"
    assert dl.MAIN_MOODS == ("Impulsive", "StimSens")

def test_load_state_labels_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        dl.load_state_labels("99999999", tag_dir=str(tmp_path))
