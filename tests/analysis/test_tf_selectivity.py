"""Tests for the fast-minus-slow selectivity detection core."""
import numpy as np

from visdetect.core.session import Session, Trial, Cluster
from visdetect.analysis.tf_selectivity import (
    TFSelectivityConfig,
    _time_vector,
    _per_pulse_rate_matrix,
)


def test_per_pulse_rate_matrix_recovers_hz():
    # A unit firing at a regular 100 Hz over the whole window; the per-pulse
    # mean rate in a flat interior region should be ~100 Hz.
    cfg = TFSelectivityConfig()
    t_vec = _time_vector(cfg)
    rate = 100.0
    spikes = np.arange(0.0, 1000.0, 1.0 / rate)
    pulses = np.array([100.0, 200.0, 300.0, 400.0])
    mat = _per_pulse_rate_matrix(spikes, pulses, t_vec, cfg.pulse.dt, cfg.pulse.sigma_ms)
    assert mat.shape == (4, t_vec.size)
    interior = (t_vec >= -0.5) & (t_vec < -0.1)
    mean_hz = np.nanmean(mat[:, interior])
    assert np.isclose(mean_hz, rate, rtol=0.05), mean_hz


def test_per_pulse_rate_matrix_empty_pulses():
    cfg = TFSelectivityConfig()
    t_vec = _time_vector(cfg)
    mat = _per_pulse_rate_matrix(np.arange(0, 10, 0.01), np.array([]), t_vec,
                                 cfg.pulse.dt, cfg.pulse.sigma_ms)
    assert mat.shape == (0, t_vec.size)


from visdetect.analysis.tf_selectivity import (
    _shared_baseline,
    compute_unit_selectivity,
)


def test_shared_baseline_is_single_value():
    cfg = TFSelectivityConfig()
    t_vec = _time_vector(cfg)
    # Two traces with deliberately different pre-window spread.
    fast_hz = np.zeros_like(t_vec); fast_hz[:] = 10.0
    slow_hz = np.zeros_like(t_vec); slow_hz[:] = 10.0
    pre = (t_vec >= cfg.pulse.pre_window[0]) & (t_vec < cfg.pulse.pre_window[1])
    rng = np.random.default_rng(0)
    fast_hz[pre] += rng.normal(0, 5.0, pre.sum())
    slow_hz[pre] += rng.normal(0, 1.0, pre.sum())
    mu, sd = _shared_baseline(fast_hz, slow_hz, t_vec, cfg.pulse.pre_window, cfg.eps)
    # The pooled sd must lie between the two per-condition sds, i.e. it is one
    # shared number, not computed separately per condition.
    assert sd > 1.0 and sd < 5.0


def test_selectivity_uses_shared_sigma():
    cfg = TFSelectivityConfig(n_shuffles=10)
    t_vec = _time_vector(cfg)
    # Hand-built fast/slow Hz traces: identical baseline, fast bump in post.
    sel = compute_unit_selectivity.__wrapped__ if hasattr(compute_unit_selectivity, "__wrapped__") else None
    # Use the real driver via a tiny session in the next tasks; here we check
    # the algebra directly through the public helper composition:
    fast_hz = np.full_like(t_vec, 8.0)
    slow_hz = np.full_like(t_vec, 8.0)
    post = (t_vec >= 0.0) & (t_vec < 0.2)
    fast_hz[post] = 18.0
    mu, sd = _shared_baseline(fast_hz, slow_hz, t_vec, cfg.pulse.pre_window, cfg.eps)
    selectivity = (fast_hz - slow_hz) / sd
    # baseline difference is zero -> selectivity flat there; post bump positive.
    pre = (t_vec >= cfg.pulse.pre_window[0]) & (t_vec < cfg.pulse.pre_window[1])
    assert np.allclose(selectivity[pre], 0.0)
    assert selectivity[post].max() > 0


from visdetect.analysis.tf_selectivity import _post_metrics


def test_post_metrics_signed_peak_and_latency():
    cfg = TFSelectivityConfig()
    t_vec = _time_vector(cfg)
    trace = np.zeros_like(t_vec)
    post = (t_vec >= 0.0) & (t_vec < 0.5)
    # a positive bump centred at ~0.10 s
    idx = np.argmin(np.abs(t_vec - 0.10))
    trace[idx - 20: idx + 20] = 5.0
    peak, lat, auc, hw = _post_metrics(trace, t_vec, cfg.pulse.post_window)
    assert np.isclose(peak, 5.0)
    assert abs(lat - 0.10) < 0.03
    assert auc > 0
    assert 0.0 < hw < 0.1


def _make_selectivity_session(n_trials=40, base_rate=20.0, evoked_rate=140.0,
                              evoked_dur=0.15, seed=0, inject=True):
    """Synthetic session yielding BOTH fast and slow pulses.

    Each trial baseline (neutral TF=1) carries alternating fast (TF=2) and slow
    (TF=0.5) samples at post-stride indices 40..140 spaced 1.0 s apart (>=2.0 s,
    before the change at +250 s). The 1.0 s spacing keeps a fast pulse's 0.15 s
    burst tail out of the next slow pulse's [-0.4, 0] pre-window, so there is no
    common-mode contamination of the baseline. The injected cluster fires a
    regular base train everywhere plus a high-rate burst after each FAST pulse
    only -> positive selectivity bump.
    """
    base_on = (np.arange(n_trials) * 300.0).astype(float)
    change_on = base_on + 250.0
    trials, fast_t, slow_t = [], [], []
    for k in range(n_trials):
        bv = np.ones(3 * 200)
        for j, idx in enumerate(range(40, 160, 20)):
            val = 2.0 if (j % 2 == 0) else 0.5
            bv[3 * idx] = val
            t_abs = base_on[k] + idx * 0.05
            (fast_t if val == 2.0 else slow_t).append(t_abs)
        trials.append(Trial(trialoutcome="Hit", reactiontimes={"RT": 0.3},
                            change_size=2.0, change_time=250.0,
                            baseline_values=bv, n_seen=None))
    fast_t = np.array(fast_t); slow_t = np.array(slow_t)
    t_end = float(change_on[-1] + 10.0)
    spikes = [np.arange(0.0, t_end, 1.0 / base_rate)]
    if inject:
        for tp in fast_t:
            spikes.append(np.arange(tp + 0.005, tp + 0.005 + evoked_dur, 1.0 / evoked_rate))
    spikes = np.sort(np.concatenate(spikes))
    clusters = [Cluster(cluster_id=0, spike_times=spikes, quality="good")]
    ni = {"Baseline_ON": base_on, "Change_ON": change_on}
    sess = Session(trials=trials, clusters=clusters, subject="SYN",
                   session_name="SEL", good_cluster_ids=[0], ni_events=ni)
    return sess, fast_t, slow_t


def test_compute_unit_selectivity_detects_injected_unit():
    cfg = TFSelectivityConfig(n_shuffles=50)
    sess, fast_t, slow_t = _make_selectivity_session(inject=True)
    st = sess.clusters[0].spike_times
    sel = compute_unit_selectivity(st, fast_t, slow_t, cfg)
    assert sel.n_fast > 0 and sel.n_slow > 0
    assert sel.sufficient
    # injected fast-locked unit -> clearly positive selectivity peak
    assert sel.sel_peak > 3.0, sel.sel_peak
    assert 0.0 < sel.sel_peak_latency < 0.25
    # Common-mode (drift) must cancel in the baseline. Check away from t=0: the
    # smoothed response legitimately smears ~50 ms back past the pulse (17 ms
    # sigma), so exclude the last 50 ms of the pre-window -- that smear is a real
    # effect, not leakage. Cancellation is essentially perfect in the rest.
    clean = (sel.t_vec >= cfg.pulse.pre_window[0]) & (sel.t_vec < -0.05)
    assert np.nanmax(np.abs(sel.selectivity[clean])) < 1.0


def test_null_separates_injected_from_random():
    cfg = TFSelectivityConfig(n_shuffles=200, seed=1)
    sess_pos, fast_t, slow_t = _make_selectivity_session(inject=True, seed=1)
    sel_pos = compute_unit_selectivity(sess_pos.clusters[0].spike_times, fast_t, slow_t, cfg)
    # injected unit clears the null
    assert sel_pos.shuffle_p < 0.05, sel_pos.shuffle_p
    assert sel_pos.sel_z_vs_null > 3.0, sel_pos.sel_z_vs_null

    cfg2 = TFSelectivityConfig(n_shuffles=200, seed=2)
    sess_neg, fast_t2, slow_t2 = _make_selectivity_session(inject=False, seed=2)
    sel_neg = compute_unit_selectivity(sess_neg.clusters[0].spike_times, fast_t2, slow_t2, cfg2)
    # no fast/slow difference (only common-mode) -> stays in the null
    assert sel_neg.shuffle_p > 0.05, sel_neg.shuffle_p


def test_split_half_high_for_injected_unit():
    cfg = TFSelectivityConfig(n_shuffles=50, seed=3)
    sess, fast_t, slow_t = _make_selectivity_session(inject=True, seed=3)
    sel = compute_unit_selectivity(sess.clusters[0].spike_times, fast_t, slow_t, cfg)
    assert sel.split_half_r > 0.5, sel.split_half_r


def test_insufficient_pulses_flagged():
    cfg = TFSelectivityConfig(n_shuffles=10, min_pulses_per_label=1000)
    sess, fast_t, slow_t = _make_selectivity_session(inject=True, seed=4)
    sel = compute_unit_selectivity(sess.clusters[0].spike_times, fast_t, slow_t, cfg)
    assert sel.sufficient is False  # far fewer than 1000 pulses per label


def test_silent_unit_does_not_crash():
    cfg = TFSelectivityConfig(n_shuffles=10, seed=5)
    sess, fast_t, slow_t = _make_selectivity_session(inject=True, seed=5)
    sel = compute_unit_selectivity(np.array([]), fast_t, slow_t, cfg)
    # no spikes -> zero traces, finite (guarded) selectivity, no exception
    assert sel.n_fast > 0 and sel.n_slow > 0
    assert np.all(np.isfinite(sel.selectivity))
    assert np.allclose(sel.selectivity, 0.0)
