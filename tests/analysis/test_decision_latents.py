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


def test_itchiness_scores_more_fa_higher_criterion_shift():
    import numpy as np, pandas as pd
    from visdetect.analysis import decision_latents as dl
    def make(fa_frac):
        rng = np.random.default_rng(0)
        rows = []
        for _ in range(200):
            if rng.random() < fa_frac:
                rows.append({"change_size": 2.0, "lick": 1, "outcome": "fa",
                             "decision_time": 1.0, "change_time_planned": 5.0, "censored": False})
            else:
                rows.append({"change_size": 2.0, "lick": 1, "outcome": "hit",
                             "decision_time": 5.3, "change_time_planned": 5.0, "censored": False})
        return pd.DataFrame(rows)
    hi = dl.itchiness_scores(make(0.6)); lo = dl.itchiness_scores(make(0.1))
    assert hi["fa_rate"] > lo["fa_rate"]
    assert "criterion_c" in hi and "baseline_hazard" in hi


def test_timing_scores_peak_and_offset():
    import numpy as np, pandas as pd
    from visdetect.analysis import decision_latents as dl
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(300):
        # changes cluster near 5s; licks (hits) shortly after
        ct = 5.0 + rng.normal(0, 0.2)
        rows.append({"change_reached": True, "change_time_planned": ct,
                     "lick": 1, "outcome": "hit", "decision_time": ct + 0.3})
    sc = dl.timing_scores(pd.DataFrame(rows), dt=0.05)
    assert 4.0 < sc["change_hazard_peak_time"] < 6.0
    assert sc["lick_hazard_peak_time"] >= sc["change_hazard_peak_time"]  # licks after change
    assert "peak_offset" in sc and "lick_hazard_spread" in sc


def test_fa_lick_hazard_nonzero_in_fa_bins_and_reuses_censored_hazard():
    import numpy as np, pandas as pd
    from visdetect.analysis import decision_latents as dl
    # 4 FA trials whose anticipatory licks land at ~4.5s, plus some hits at ~7s.
    rows = []
    for _ in range(4):
        rows.append({"outcome": "fa", "decision_time": 4.5,
                     "change_time_planned": 7.0, "lick": 1})
    for _ in range(6):
        rows.append({"outcome": "hit", "decision_time": 7.3,
                     "change_time_planned": 7.0, "lick": 1})
    df = pd.DataFrame(rows)
    centers, hazard, survival = dl.fa_lick_hazard(df, dt=0.05)
    # reuses censored_hazard: same shape contract (centers/hazard/survival aligned).
    # FIX (round 2): non-FA trials are censored at the CHANGE (min(change_time,
    # decision_time)), not at decision_time — an FA can only occur before the change.
    censor_t = np.where(df["outcome"].values == "fa",
                        df["decision_time"].values.astype(float),
                        np.minimum(df["change_time_planned"].values.astype(float),
                                   df["decision_time"].values.astype(float)))
    ref_c, ref_h, ref_s = dl.censored_hazard(
        censor_t, (df["outcome"] == "fa").values, dt=0.05)
    assert centers.shape == hazard.shape == survival.shape == ref_h.shape
    assert np.allclose(centers, ref_c) and np.allclose(hazard, ref_h)
    # hazard is non-zero in the bin containing the FA licks (~4.5s)
    fa_bin = np.argmin(np.abs(centers - 4.5))
    assert hazard[fa_bin] > 0.0
    # nothing happens before any lick (e.g. ~2s bin is zero)
    early_bin = np.argmin(np.abs(centers - 2.0))
    assert hazard[early_bin] == 0.0


def test_fa_lick_hazard_censors_non_fa_at_change_not_decision():
    """FIX (round 2): an anticipatory (FA) lick can only happen BEFORE the change.
    A hit/miss trial must therefore leave the FA at-risk set at its change time,
    NOT at its (later) decision_time. This test FAILS under the old behaviour
    (censor at decision_time) and PASSES once non-FA trials censor at the change.

    Construction: one FA trial whose early lick lands at 5.2 s, and 9 hit trials
    whose change is at 5.0 s with a (later) decision_time of 5.5 s. At the FA bin
    (~5.2 s) the at-risk denominator differs sharply:
      * FIX  → 9 hits already censored at 5.0 s ⇒ at_risk = 1 ⇒ FA hazard = 1.0
      * OLD  → 9 hits still at risk at 5.2 s    ⇒ at_risk = 10 ⇒ FA hazard ≈ 0.1
    """
    import numpy as np, pandas as pd
    from visdetect.analysis import decision_latents as dl
    rows = [{"outcome": "fa", "decision_time": 5.2, "change_time_planned": 8.0, "lick": 1}]
    for _ in range(9):
        rows.append({"outcome": "hit", "decision_time": 5.5,
                     "change_time_planned": 5.0, "lick": 1})
    df = pd.DataFrame(rows)
    centers, hazard, survival = dl.fa_lick_hazard(df, dt=0.05)
    fa_bin = np.argmin(np.abs(centers - 5.2))
    # FIX: the 9 hits are gone by 5.0 s, leaving only the FA trial at risk ⇒ haz=1.0
    assert np.isclose(hazard[fa_bin], 1.0)
    # the OLD (decision_time) censoring would have left the 9 hits at risk ⇒ haz≈0.1,
    # so the behaviour genuinely changed (the at-risk set now depletes at the change).
    bad_c, bad_h, bad_s = dl.censored_hazard(
        df["decision_time"].values.astype(float),
        (df["outcome"] == "fa").values, dt=0.05)
    bad_fa_bin = np.argmin(np.abs(bad_c - 5.2))
    assert np.isclose(bad_h[bad_fa_bin], 0.1, atol=1e-6)   # old buggy value
    assert hazard[fa_bin] > bad_h[bad_fa_bin] + 1e-6        # fix raised the hazard


def test_sharpness_scores_psy_threshold_present_and_finite():
    import numpy as np, pandas as pd
    from visdetect.analysis import decision_latents as dl
    rng = np.random.default_rng(0)
    rows = []
    # well-separated psychometric: low detect at small Δ, near-ceiling at big Δ
    for cs, p in [(1.0, 0.05), (1.25, 0.3), (1.5, 0.55), (2.0, 0.85), (4.0, 0.98)]:
        for _ in range(60):
            lick = rng.random() < p
            outcome = "hit" if (cs > 1.0 and lick) else ("fa" if (cs == 1.0 and lick) else "miss")
            ct = 7.0
            rows.append({"change_size": cs, "lick": int(lick), "outcome": outcome,
                         "change_time_planned": ct,
                         "decision_time": ct + rng.uniform(0.2, 0.6) if outcome == "hit" else ct + 2.0})
    sc = dl.sharpness_scores(pd.DataFrame(rows))
    assert "psy_threshold" in sc
    assert np.isfinite(sc["psy_threshold"])
    assert 1.0 <= sc["psy_threshold"] <= 8.0   # clamped to plausible change-size range


def test_cell_and_latent_tables(synth_session, synth_state_labels):
    from visdetect.analysis import decision_latents as dl
    tab = dl.build_trial_table(synth_session, synth_state_labels, "07072025")
    tab["session_dprime"] = 0.9; tab["comprehension_flag"] = "post"
    cells = dl.descriptive_cell_table(tab, min_cell_trials=1)
    assert set(cells["state_label"]).issubset(set(dl.MAIN_MOODS + dl.SEPARATE_MOODS))
    assert {"criterion_c", "psy_slope", "lick_hazard_peak_time", "n_trials"}.issubset(cells.columns)
    lat = dl.descriptive_latent_table(tab, cells)
    assert len(lat) == len(tab)
    assert {"criterion_c", "sharpness_psy_slope", "trial_in_session"}.issubset(lat.columns)
    assert "rt_cv_by_cs" in lat.columns   # spec §5 cross-check column propagated
