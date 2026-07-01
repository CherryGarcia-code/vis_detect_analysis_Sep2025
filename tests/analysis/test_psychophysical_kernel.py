"""B10 — psychophysical / neural impulsivity-kernel library tests.

Pure unit tests (no real pkls). The synthetic-recovery test
(``test_reverse_correlation_recovers_planted_kernel``) is the rigor backbone:
plant a known kernel, recover it.
"""
import numpy as np
from types import SimpleNamespace

import visdetect.analysis.psychophysical_kernel as pk


# ── helpers ──────────────────────────────────────────────────────────────
def _trial(bv, outcome="fa", fa=None, ct=np.nan, cs=1.0):
    rt = {} if fa is None else {"FA": fa}
    return SimpleNamespace(baseline_values=np.asarray(bv, float),
                           reactiontimes=rt, trialoutcome=outcome,
                           change_time=ct, change_size=cs)


def _session(trials):
    return SimpleNamespace(trials=list(trials))


def _cluster(cid, spikes):
    return SimpleNamespace(cluster_id=cid, spike_times=np.asarray(spikes, float))


_L = round((pk.KERNEL_PRE_S - pk.KERNEL_REFRACTORY_S) / pk.DT)   # 27


# ── Task 1 ───────────────────────────────────────────────────────────────
def test_baseline_log2tf_stride3_and_log2():
    bv = np.array([1, 1, 1, 2, 2, 2, 0.5, 0.5, 0.5], float)     # 3 TF values held x3
    t, y = pk.baseline_log2tf(_trial(bv), tf_base=1.0)
    assert y.shape == (3,)
    np.testing.assert_allclose(y, [0.0, 1.0, -1.0], atol=1e-9)
    np.testing.assert_allclose(t, [0.0, 0.05, 0.10], atol=1e-9)


# ── Task 2 ───────────────────────────────────────────────────────────────
def test_fa_kernel_epochs_window_and_guards():
    rng = np.random.default_rng(0)
    bv = np.repeat(np.exp2(rng.normal(0, 0.25, 800)), 3)         # 40 s of white baseline
    good = _trial(bv, "fa", fa=5.0, ct=np.nan)
    early = _trial(bv, "fa", fa=0.3, ct=np.nan)                  # < MIN_BASELINE_S
    near_change = _trial(bv, "fa", fa=5.0, ct=5.2)              # within CHANGE_GUARD
    miss = _trial(bv, "miss", fa=None, ct=6.0)                  # not fa
    eps = pk.fa_kernel_epochs(_session([good, early, near_change, miss]))
    assert len(eps) == 1
    assert eps[0]["window"].shape == (_L,)
    assert eps[0]["trial_idx"] == 0


# ── Task 3 ───────────────────────────────────────────────────────────────
def test_withhold_epochs_time_matched_and_prechange():
    rng = np.random.default_rng(1)
    bv = np.repeat(np.exp2(rng.normal(0, 0.25, 800)), 3)
    fa = _trial(bv, "fa", fa=5.0, ct=np.nan)
    hit = _trial(bv, "hit", fa=None, ct=8.0)                    # baseline covers 5 s
    sess = _session([fa, hit])
    eps = pk.fa_kernel_epochs(sess)
    wh = pk.withhold_epochs(sess, eps, rng=np.random.default_rng(2))
    assert len(wh) == 1 and wh[0] is not None and wh[0].shape == (_L,)


def test_withhold_epochs_none_when_no_prechange_coverage():
    rng = np.random.default_rng(3)
    bv = np.repeat(np.exp2(rng.normal(0, 0.25, 800)), 3)
    fa = _trial(bv, "fa", fa=5.0, ct=np.nan)
    hit = _trial(bv, "hit", fa=None, ct=1.0)                    # change at 1 s
    sess = _session([fa, hit])
    eps = pk.fa_kernel_epochs(sess)
    wh = pk.withhold_epochs(sess, eps, rng=np.random.default_rng(4))
    assert wh[0] is None


# ── Task 4 (rigor backbone) ──────────────────────────────────────────────
def test_reverse_correlation_recovers_planted_kernel():
    rng = np.random.default_rng(7)
    planted = np.zeros(_L)
    planted[-10:] = np.linspace(0, 0.6, 10)                     # rising ramp toward lick
    fa_windows = [planted + rng.normal(0, 0.25, _L) for _ in range(400)]
    withhold_windows = [rng.normal(0, 0.25, _L) for _ in range(400)]
    k = pk.reverse_correlation_kernel(fa_windows, withhold_windows)
    assert k.shape == (_L,)
    assert np.corrcoef(k, planted)[0, 1] > 0.9
    assert k[-1] > k[0]
    assert pk.kernel_lags().shape == (_L,)
    assert pk.kernel_lags()[-1] < 0


# ── Task 5 ───────────────────────────────────────────────────────────────
def test_bootstrap_ci_deterministic_and_bounds():
    rng = np.random.default_rng(11)
    fa = [np.full(_L, 0.3) + rng.normal(0, 0.1, _L) for _ in range(200)]
    wh = [rng.normal(0, 0.1, _L) for _ in range(200)]
    k1, lo1, hi1 = pk.bootstrap_kernel_ci(fa, wh, n_boot=200, seed=42)
    k2, lo2, hi2 = pk.bootstrap_kernel_ci(fa, wh, n_boot=200, seed=42)
    np.testing.assert_array_equal(lo1, lo2)
    np.testing.assert_array_equal(hi1, hi2)
    assert np.all(lo1 <= k1) and np.all(k1 <= hi1)
    assert np.all(lo1 > -1) and np.all(hi1 < 1)


# ── Task 6 ───────────────────────────────────────────────────────────────
def test_kernel_shape_metrics():
    k = np.zeros(_L)
    k[-6:-2] = [0.2, 0.4, 0.4, 0.2]                             # peak 0.4 near the lick
    m = pk.kernel_shape_metrics(k)
    assert abs(m["peak_amp"] - 0.4) < 1e-9
    assert m["peak_lag_s"] < 0
    assert m["half_width_s"] >= 2 * pk.DT


# ── Task 7 ───────────────────────────────────────────────────────────────
def test_signed_population_signal_tracks_stimulus():
    bon = [100.0, 200.0]
    tr0 = _trial(np.ones(2400), "fa", fa=3.0, ct=np.nan)
    tr1 = _trial(np.ones(2400), "hit", fa=None, ct=4.0)
    dense = 100.0 + np.r_[np.linspace(0, 1, 20), np.linspace(1, 2, 200),
                          np.linspace(2, 3, 20)]
    sparse = 200.0 + np.linspace(0, 4, 60)
    sess = SimpleNamespace(trials=[tr0, tr1],
                           ni_events={"Baseline_ON": np.array(bon)},
                           clusters=[_cluster(10, np.r_[dense, sparse])])
    out = pk.signed_population_signal(sess, {10: +1})
    assert set(out) == {0, 1}
    t, S = out[0]
    assert 0.8 < t[np.argmax(S)] < 2.2


# ── Task 8 ───────────────────────────────────────────────────────────────
def test_stimulus_matched_control_decomposes_gain():
    stim_fa = [np.linspace(0, 0.5, _L) for _ in range(100)]
    stim_wh = [np.linspace(0, 0.5, _L) for _ in range(100)]
    pop_wh = [s.copy() for s in stim_wh]
    pop_fa = [s + 0.4 for s in stim_fa]
    d = pk.stimulus_matched_control(stim_fa, stim_wh, pop_fa, pop_wh)
    np.testing.assert_allclose(d["sensory"], np.mean(pop_wh, 0), atol=1e-9)
    np.testing.assert_allclose(d["gain"], np.full(_L, 0.4), atol=1e-9)
    np.testing.assert_allclose(d["total"], d["sensory"] + d["gain"], atol=1e-9)
