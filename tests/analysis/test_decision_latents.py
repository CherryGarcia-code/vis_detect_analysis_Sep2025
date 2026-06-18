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

def test_enumerate_valid_sessions_sorted_and_filtered(tmp_path):
    d = tmp_path / "BG_046"; d.mkdir()
    for s in ["30062025", "01072025"]:
        (d / f"{s}.csv").write_text("trial_idx,state_label,state_confidence\n0,Impulsive,0.9\n")
    out = dl.enumerate_valid_sessions(tag_dir=str(tmp_path), min_total_trials=0)
    assert out == ["30062025", "01072025"]  # chronological (30 Jun before 01 Jul)

def test_assign_comprehension_flags_marks_boundary():
    dprime = {"30062025": 0.1, "01072025": 0.2, "02072025": 0.7, "03072025": 0.9}
    flags = dl.assign_comprehension_flags(dprime, threshold=0.5)
    assert flags["30062025"] == "pre" and flags["01072025"] == "pre"
    assert flags["02072025"] == "post" and flags["03072025"] == "post"
