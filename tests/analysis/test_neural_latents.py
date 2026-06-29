import types, numpy as np, pandas as pd, pytest
from visdetect.analysis import neural_latents as nl

def _fake_session(outcomes, change_times, change_sizes, baseline_on, n_clusters=4, seed=0):
    """Minimal Session stand-in for build_population_tensor: trials with
    trialoutcome/change_time/change_size, ni_events Baseline_ON/Change_ON
    (absolute s), clusters with cluster_id/spike_times, and good id lists."""
    rng = np.random.default_rng(seed)
    trials = [types.SimpleNamespace(trialoutcome=o, change_time=ct, change_size=cs)
              for o, ct, cs in zip(outcomes, change_times, change_sizes)]
    change_on = [b + ct if o in ("hit", "miss") else np.nan
                 for b, ct, o in zip(baseline_on, change_times, outcomes)]
    ni = {"Baseline_ON": np.array(baseline_on, float),
          "Change_ON": np.array(change_on, float)}
    clusters = [types.SimpleNamespace(cluster_id=cid,
                spike_times=np.sort(rng.uniform(0, max(baseline_on) + 12, 4000)))
                for cid in range(10, 10 + n_clusters)]
    return types.SimpleNamespace(trials=trials, ni_events=ni, clusters=clusters,
        good_and_stable_ids=[c.cluster_id for c in clusters], good_cluster_ids=[])

def test_join_keys_by_literal_trial_idx_with_gaps():
    # 6 session trials; latent table covers a NON-contiguous subset (gaps),
    # mimicking the real deliverable (abort/ref dropped, not renumbered).
    outcomes = ["abort", "hit", "fa", "miss", "ref", "hit"]
    cts      = [7.0,      6.9,   7.1,  7.2,    7.0,   6.8]
    css      = [1.0,      2.0,   1.0,  1.5,    1.0,   4.0]
    base     = [10.0, 30.0, 55.0, 80.0, 105.0, 130.0]
    sess = _fake_session(outcomes, cts, css, base)
    # latent rows reference trial_idx 1,2,3,5 (skip aborts/ref 0,4) — note the GAP at 4
    latent_rows = pd.DataFrame([
        dict(trial_idx=1, outcome="hit",  change_size=2.0, decision_time=7.4,
             change_time_planned=6.9, change_reached=True,  state_label="StimSens",
             timing_urgency_at_decision=0.3, itchiness_caution=-5.0, sharpness_drift=1.0,
             evidence_integral_at_decision=0.5, expected_change_time=7.0),
        dict(trial_idx=2, outcome="fa",   change_size=1.0, decision_time=3.0,
             change_time_planned=7.1, change_reached=False, state_label="Impulsive",
             timing_urgency_at_decision=0.1, itchiness_caution=-4.5, sharpness_drift=1.0,
             evidence_integral_at_decision=0.0, expected_change_time=7.0),
        dict(trial_idx=3, outcome="miss", change_size=1.5, decision_time=9.355,
             change_time_planned=7.2, change_reached=True,  state_label="StimSens",
             timing_urgency_at_decision=0.05, itchiness_caution=-5.0, sharpness_drift=1.0,
             evidence_integral_at_decision=0.2, expected_change_time=7.0),
        dict(trial_idx=5, outcome="hit",  change_size=4.0, decision_time=7.1,
             change_time_planned=6.8, change_reached=True,  state_label="StimSens",
             timing_urgency_at_decision=0.4, itchiness_caution=-5.0, sharpness_drift=1.0,
             evidence_integral_at_decision=0.8, expected_change_time=7.0),
    ])
    res = nl.join_session(sess, latent_rows, window=(-1.3, 6.0), bin_size=0.05,
                          baseline_window=(-1.3, -0.3), min_rate_hz=0.0, verify=True)
    # rows align to literal trial_idx, gap-trial 4 absent, decoy ordering not used
    assert list(res.y["trial_idx"]) == [1, 2, 3, 5]
    assert res.z.shape[0] == 4 and res.z.shape[2] == 4   # 4 trials, 4 units
    assert list(res.kept_trials) == [1, 2, 3, 5]
    # verification triple-check passes silently (outcome/size/time match the session)

def test_join_verification_catches_misalignment():
    outcomes = ["hit", "fa"]; cts = [6.9, 7.1]; css = [2.0, 1.0]; base = [10.0, 30.0]
    sess = _fake_session(outcomes, cts, css, base)
    bad = pd.DataFrame([dict(trial_idx=0, outcome="fa",  # WRONG outcome for trial 0 (really hit)
        change_size=2.0, decision_time=7.4, change_time_planned=6.9, change_reached=True,
        state_label="StimSens", timing_urgency_at_decision=0.3, itchiness_caution=-5.0,
        sharpness_drift=1.0, evidence_integral_at_decision=0.5, expected_change_time=7.0)])
    with pytest.raises(AssertionError):
        nl.join_session(sess, bad, window=(-1.3, 6.0), bin_size=0.05,
                        baseline_window=(-1.3, -0.3), min_rate_hz=0.0, verify=True)

def test_window_feature_matrix_means_over_fixed_window():
    n_tr, n_units = 5, 3
    bin_centers = np.arange(0.0, 6.0, 0.05)
    z = np.zeros((n_tr, bin_centers.size, n_units))
    early = (bin_centers >= 0.5) & (bin_centers < 2.5)
    z[:, early, :] = 2.0                       # constant 2.0 inside the early window
    X = nl.window_feature_matrix(z, bin_centers, nl.WINDOWS["early"])
    assert X.shape == (n_tr, n_units)
    assert np.allclose(X, 2.0)
    with pytest.raises(ValueError):
        nl.window_feature_matrix(z, bin_centers, (100.0, 200.0))  # no bins

def test_project_out_axis_removes_component():
    rng = np.random.default_rng(0)
    axis = np.array([1.0, 0.0, 0.0])
    X = rng.normal(size=(20, 3))
    Xp = nl.project_out_axis(X, axis)
    assert np.allclose(Xp @ axis, 0.0, atol=1e-10)          # component along axis removed
    # variance orthogonal to axis preserved
    assert np.allclose(Xp[:, 1:], X[:, 1:], atol=1e-10)

def test_motor_axis_signal_tracks_projection():
    axis = np.array([0.0, 1.0, 0.0])
    X = np.array([[0, 3.0, 0], [0, -1.0, 0]])
    assert np.allclose(nl.motor_axis_signal(X, axis), [3.0, -1.0])

# ── Task 4: per-session decoder + cohort aggregation + nulls ───────────────
def _sessions_with_within_signal(K=6, n=80, p=6, seed=1):
    rng = np.random.default_rng(seed); out = []
    for s in range(K):
        w = rng.normal(size=p); y = rng.normal(size=n)
        X = y[:, None] * w[None, :] + rng.normal(scale=0.5, size=(n, p))  # within-session signal
        tt = np.where(rng.random(n) < 0.5, "hit", "fa")
        out.append((s, X, y, tt))
    return out

def _sessions_simpson_only(K=6, n=80, p=6, seed=2):
    """NO within-session signal; only between-session y-offsets + disjoint feature
    blocks encoding the offset. A pooled global Spearman is spuriously high; the
    correct per-session aggregate is near chance."""
    rng = np.random.default_rng(seed); out = []
    for s in range(K):
        offset = 5.0 * s
        y = offset + rng.normal(scale=0.3, size=n)                 # ~constant within session
        X = np.zeros((n, p * K))
        X[:, s*p:(s+1)*p] = offset + rng.normal(scale=0.3, size=(n, p))  # disjoint block ~ offset
        tt = np.where(rng.random(n) < 0.5, "hit", "fa")
        out.append((s, X, y, tt))
    return out

def test_decode_cohort_recovers_within_session_signal():
    res = nl.decode_cohort(_sessions_with_within_signal(), n_null=100, seed=42)
    assert len(res["per_session"]) == 6                       # ONE r per session
    assert res["mean_r"] > 0.5                                # within-session signal recovered
    assert res["mean_r"] > res["null_mean"] + 2 * res["null_sd"]
    assert res["within_type"]["hit"] > 0.3                    # NOTE B: graded within hits
    # pin reported mean_r to the per-session aggregate by construction
    assert res["mean_r"] == pytest.approx(np.mean([p["r"] for p in res["per_session"]]))

def test_per_session_scheme_defeats_simpson_inflation():
    sessions = _sessions_simpson_only()
    res = nl.decode_cohort(sessions, n_null=50, seed=42)
    assert len(res["per_session"]) == 6
    assert abs(res["mean_r"]) < 0.2                           # CORRECT aggregate ~ chance
    # REGRESSION GUARD: the REJECTED pooled-global-Spearman scheme is spuriously high
    from scipy.stats import spearmanr
    yp = [nl.decode_session(X, y, seed=42)["y_pred"] for _, X, y, _ in sessions]
    yt = [y for _, _, y, _ in sessions]
    r_global = spearmanr(np.concatenate(yp), np.concatenate(yt)).correlation
    assert r_global > 0.5                                     # between-session inflation, as feared

def test_within_type_graded_separates_types():
    y_true = np.array([1., 2., 3., 10., 11., 12.])
    y_pred = np.array([1.1, 2.0, 2.9, 10.2, 10.9, 12.1])
    tt = np.array(["fa", "fa", "fa", "hit", "hit", "hit"])
    g = nl.within_type_graded(y_pred, y_true, tt)
    assert g["hit"] > 0.8 and g["fa"] > 0.8                   # graded WITHIN each type

def test_decode_cohort_null_byte_identical_across_workers():
    # The parallel null must be BYTE-IDENTICAL to the serial null regardless of
    # worker count (CI lock, mirrors the B8 ladder-determinism guarantee). Seeds
    # are pre-generated parent-side, one per shuffle, so null[i] never depends on
    # how the work is sharded across the ProcessPoolExecutor.
    sessions = _sessions_with_within_signal()
    null_1 = nl._cohort_null(sessions, n_null=24, seed=42, n_workers=1)
    null_2 = nl._cohort_null(sessions, n_null=24, seed=42, n_workers=2)
    assert np.array_equal(null_1, null_2)                     # byte-identical

# ── Task 5: phi/ramp bases for the NOTE-A discriminability prerequisite ────
def test_phi_ramp_bases_shapes():
    # window = nl.WINDOWS["late"] = (4.0, 6.0): the latest pre-mu readout window
    # (PRIMARY candidate per the §4 selection rule). NOTE-A collinearity is
    # window-dependent: negligible far below mu (early/mid windows are near-flat
    # phi tails) and strong only NEAR mu; over [0,6] the 7-sigma tail kills it.
    t = np.linspace(4.0, 6.0, 120)
    b = nl.phi_ramp_bases(t, mu=7.0, sigma=0.8)
    assert b["phi"].argmax() == len(t) - 1          # rising flank: phi peaks toward mu>6
    assert np.all(np.diff(b["ramp"]) > 0)           # ramp strictly monotonic
    # over the latest pre-mu readout window ([4,6], the PRIMARY candidate), phi's
    # rising flank is highly collinear with a monotonic ramp (NOTE A) ->
    # phi-specificity is expected to be underpowered there.
    assert np.corrcoef(b["phi"], b["ramp"])[0, 1] > 0.85   # true value ~0.89 over [4,6]

# ── Task 6: C1 real-data gate pure core (per-session decode + motor-CD survival) ──
def test_evaluate_window_per_session_gate():
    from scripts.neural_latents import n1_c1_gate as gate
    sessions = _sessions_with_within_signal()         # Task-4 generator: per-session signal
    # benign per-session motor axis: remove only 1 of the p=6 signal dims, so the
    # within-session decode survives projection (signal is spread across all dims).
    motor_axes = {s: np.eye(X.shape[1])[0] for s, X, y, tt in sessions}
    res = gate.evaluate_window(sessions, motor_axes, n_null=50, seed=42)
    assert res["mean_r"] > res["null_mean"] + 2 * res["null_sd"]         # prong 1
    assert isinstance(res["survives_projection"], bool)                 # prong 2 computed
    assert np.isfinite(res["mean_r_after_projection"])                  # finite (real test)
    assert res["mean_r_after_projection"] > 0.3                         # signal spread > 1 dim survives
    assert res["within_type"]["hit"] > 0.0                              # NOTE B
