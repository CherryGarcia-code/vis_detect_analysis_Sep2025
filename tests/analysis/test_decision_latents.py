import numpy as np
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


def test_build_trial_table_filters_and_columns(synth_session, synth_state_labels):
    from visdetect.analysis import decision_latents as dl
    tab = dl.build_trial_table(synth_session, synth_state_labels, "07072025", dt=0.05)
    assert "Abort" not in tab["state_label"].values          # mood Abort dropped
    assert {"sharpness", "itchiness"}.isdisjoint(tab.columns) # not here yet
    for col in ["session_name", "trial_idx", "outcome", "change_size",
                "change_time_planned", "change_reached", "decision_time",
                "lick", "censored", "state_label", "trial_in_session"]:
        assert col in tab.columns
    # change_reached True only for hit/miss
    assert (tab.loc[tab["change_reached"], "outcome"].isin(["hit", "miss"])).all()
    assert tab["trial_in_session"].is_monotonic_increasing


def test_censored_hazard_counts_events_and_censoring():
    from visdetect.analysis import decision_latents as dl
    # 3 trials: event at 0.10s, event at 0.10s, CENSORED at 0.05s (no event)
    dur = np.array([0.10, 0.10, 0.05]); ev = np.array([True, True, False])
    centers, hz, surv = dl.censored_hazard(dur, ev, dt=0.05, t_max=0.15)
    # bin0 [0,0.05): risk=3, events=0 -> hz=0 ; the censored trial leaves after bin0
    assert hz[0] == 0.0
    # bin1 [0.05,0.10): risk=2 (censored gone), events=2 -> hz=1.0
    assert np.isclose(hz[1], 1.0)
    assert np.all(surv <= 1.0) and np.all(np.diff(surv) <= 1e-9)

def test_censored_hazard_survival_is_one_minus_prod():
    from visdetect.analysis import decision_latents as dl
    dur = np.array([0.10, 0.15]); ev = np.array([True, True])
    _, hz, surv = dl.censored_hazard(dur, ev, dt=0.05, t_max=0.20)
    assert np.isclose(surv[-1], np.prod(1 - hz))

def test_censored_hazard_bins_mid_bin_duration_correctly():
    import numpy as np
    from visdetect.analysis import decision_latents as dl
    # one event at 0.07s -> belongs in bin1 (0.05,0.10], NOT bin0
    centers, hz, surv = dl.censored_hazard(np.array([0.07]), np.array([True]), dt=0.05, t_max=0.15)
    assert hz[0] == 0.0          # nothing happens in bin0 (0,0.05]
    assert np.isclose(hz[1], 1.0)  # the event lands in bin1


def test_sharpness_scores_keys_and_dprime_direction():
    import numpy as np, pandas as pd
    from visdetect.analysis import decision_latents as dl
    rng = np.random.default_rng(0)
    rows = []
    for cs, p in [(1.0, 0.1), (1.25, 0.4), (2.0, 0.8), (4.0, 0.95)]:
        for _ in range(50):
            lick = rng.random() < p
            outcome = "hit" if (cs > 1.0 and lick) else ("fa" if (cs == 1.0 and lick) else "miss")
            ct = 5.0
            rows.append({"change_size": cs, "lick": int(lick), "outcome": outcome,
                         "change_time_planned": ct,
                         "decision_time": ct + rng.uniform(0.2, 0.6) if outcome == "hit" else ct + 2.0})
    sc = dl.sharpness_scores(pd.DataFrame(rows))
    assert "psy_slope" in sc and "dprime" in sc
    assert any(k.startswith("rt_cv_cs") for k in sc)
    assert sc["dprime"] > 0          # more hits on big changes than FAs on catch
