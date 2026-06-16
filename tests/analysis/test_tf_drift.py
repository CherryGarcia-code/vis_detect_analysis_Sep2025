# tests/analysis/test_tf_drift.py
import numpy as np
from visdetect.analysis.tf_drift import estimate_drift


def _ramp_spikes(t_end, r0, r1, seed):
    """Inhomogeneous Poisson via thinning: rate ramps linearly r0 -> r1."""
    rng = np.random.default_rng(seed)
    r_max = max(r0, r1)
    cand = np.sort(rng.uniform(0, t_end, size=int(r_max * t_end * 1.5)))
    lam = r0 + (r1 - r0) * (cand / t_end)
    keep = rng.random(cand.size) < (lam / r_max)
    return cand[keep]


def test_estimate_drift_recovers_ramp():
    t_end = 600.0
    spikes = _ramp_spikes(t_end, 2.0, 10.0, seed=0)
    grid_t, drift, mean_rate = estimate_drift(spikes, 0.0, t_end, bin_s=1.0, kernel_s=20.0)
    assert grid_t.shape == drift.shape
    early = drift[grid_t < 100].mean()
    late = drift[grid_t > 500].mean()
    assert late > early + 3.0            # rising drift recovered
    assert abs(mean_rate - spikes.size / t_end) < 1e-6


def test_estimate_drift_flat_is_flat():
    rng = np.random.default_rng(1)
    t_end = 600.0
    spikes = np.sort(rng.uniform(0, t_end, size=int(5.0 * t_end)))
    grid_t, drift, mean_rate = estimate_drift(spikes, 0.0, t_end, bin_s=1.0, kernel_s=20.0)
    assert abs(mean_rate - 5.0) < 0.5
    assert np.std(drift) < 1.0           # ~flat, no spurious trend


from visdetect.analysis.tf_drift import detrended_pulse_average

PRE = (-0.4, 0.0)
POST = (0.0, 0.5)


def _flat_spikes(t_end, rate, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0, t_end, size=rng.poisson(rate * t_end)))


def _flat_plus_bump(t_end, rate, pulses, bump_hz, bump_dur, seed):
    rng = np.random.default_rng(seed)
    base = np.sort(rng.uniform(0, t_end, size=rng.poisson(rate * t_end)))
    parts = [base]
    for p in pulses:
        n = rng.poisson(bump_hz * bump_dur)
        if n:
            parts.append(rng.uniform(p, p + bump_dur, size=n))
    return np.sort(np.concatenate(parts))


def test_detrended_baseline_is_in_hz_near_mean_rate():
    t_end = 400.0
    spikes = _flat_spikes(t_end, 5.0, seed=2)
    pulses = np.arange(10.0, t_end - 10.0, 2.0)
    gt, dr, mr = estimate_drift(spikes, 0.0, t_end, bin_s=1.0, kernel_s=20.0)
    det, sem, t_vec = detrended_pulse_average(
        spikes, pulses, PRE, POST, 0.005, 20.0, gt, dr, mr)
    pre = det[t_vec < 0.0]
    assert abs(pre.mean() - 5.0) < 1.5            # baseline ~ true rate in Hz
    assert abs(np.polyfit(t_vec[t_vec < 0.0], pre, 1)[0]) < 5.0  # ~flat


def test_detrended_preserves_pulse_response():
    t_end = 400.0
    pulses = np.arange(10.0, t_end - 10.0, 2.0)
    spikes = _flat_plus_bump(t_end, 5.0, pulses, bump_hz=30.0, bump_dur=0.1, seed=4)
    gt, dr, mr = estimate_drift(spikes, 0.0, t_end, bin_s=1.0, kernel_s=20.0)
    det, sem, t_vec = detrended_pulse_average(
        spikes, pulses, PRE, POST, 0.005, 20.0, gt, dr, mr)
    baseline = det[t_vec < 0.0].mean()
    peak = det[(t_vec >= 0.0) & (t_vec < 0.15)].max()
    assert peak > baseline + 8.0                  # injected bump survives detrend


def test_detrended_empty_pulses_returns_empty():
    spikes = _flat_spikes(100.0, 5.0, seed=5)
    det, sem, t_vec = detrended_pulse_average(
        spikes, np.array([]), PRE, POST, 0.005, 20.0,
        *estimate_drift(spikes, 0.0, 100.0))
    assert det.size == 0


from visdetect.analysis.tf_drift import prepulse_slope


def test_prepulse_slope_of_flat_is_zero():
    t_vec = np.linspace(-0.4, 0.5, 180)
    trace = np.full_like(t_vec, 7.0)
    assert abs(prepulse_slope(trace, t_vec, PRE)) < 1e-6


def test_prepulse_slope_of_ramp_matches_slope():
    t_vec = np.linspace(-0.4, 0.5, 180)
    trace = 3.0 * t_vec + 2.0           # slope 3.0 over the pre-window
    assert abs(prepulse_slope(trace, t_vec, PRE) - 3.0) < 1e-6


def test_prepulse_slope_too_few_bins_is_nan():
    t_vec = np.array([0.1, 0.2])        # nothing in the pre-window
    assert np.isnan(prepulse_slope(np.array([1.0, 2.0]), t_vec, PRE))


def test_prepulse_slope_exactly_one_bin_is_nan():
    t_vec = np.array([-0.2, 0.1])   # one sample in PRE, one outside
    assert np.isnan(prepulse_slope(np.array([5.0, 5.0]), t_vec, PRE))


from visdetect.analysis.tf_drift import circular_shift_null


def test_null_envelope_flags_real_bump():
    t_end = 200.0
    pulses = np.arange(10.0, t_end - 10.0, 3.0)
    spikes = _flat_plus_bump(t_end, 5.0, pulses, bump_hz=30.0, bump_dur=0.1, seed=7)
    gt, dr, mr = estimate_drift(spikes, 0.0, t_end, bin_s=1.0, kernel_s=20.0)
    det, _, t_vec = detrended_pulse_average(
        spikes, pulses, PRE, POST, 0.005, 20.0, gt, dr, mr)
    from visdetect.analysis.tf_pulse import _zscore_trace
    obs_z = _zscore_trace(det, t_vec, PRE)

    null_z, t_null = circular_shift_null(
        spikes, pulses, PRE, POST, 0.005, 20.0, bin_s=1.0, kernel_s=20.0,
        session_dur=t_end, n_shuffles=30, seed=0)
    hi = np.percentile(null_z, 95, axis=0)
    post = (t_vec >= 0.0) & (t_vec < 0.15)
    assert obs_z[post].max() > hi[post].max()      # real response exits the null


def test_null_envelope_contains_flat_unit():
    t_end = 200.0
    pulses = np.arange(10.0, t_end - 10.0, 3.0)
    spikes = _flat_spikes(t_end, 5.0, seed=8)
    gt, dr, mr = estimate_drift(spikes, 0.0, t_end, bin_s=1.0, kernel_s=20.0)
    det, _, t_vec = detrended_pulse_average(
        spikes, pulses, PRE, POST, 0.005, 20.0, gt, dr, mr)
    from visdetect.analysis.tf_pulse import _zscore_trace
    obs_z = _zscore_trace(det, t_vec, PRE)
    null_z, _ = circular_shift_null(
        spikes, pulses, PRE, POST, 0.005, 20.0, bin_s=1.0, kernel_s=20.0,
        session_dur=t_end, n_shuffles=30, seed=0)
    lo = np.percentile(null_z, 2.5, axis=0)
    hi = np.percentile(null_z, 97.5, axis=0)
    post = (t_vec >= 0.0) & (t_vec < 0.5)
    frac_inside = np.mean((obs_z[post] >= lo[post]) & (obs_z[post] <= hi[post]))
    assert frac_inside > 0.8                       # flat unit mostly within null


def test_phase0_run_on_session_smoke(tmp_path):
    from visdetect.utils.synthetic import make_synthetic_session
    import sys, importlib.util, pathlib
    mod_path = pathlib.Path(__file__).resolve().parents[2] / \
        "scripts" / "tf_responsiveness" / "validate_drift_phase0.py"
    spec = importlib.util.spec_from_file_location("validate_drift_phase0", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    sess = make_synthetic_session(n_trials=60, n_clusters=4, seed=11)
    rows = mod.run_units(
        sess, cluster_ids=[0, 1], kernel_s=20.0, n_shuffles=10,
        use_constraints=False,
        out_png=str(tmp_path / "phase0.png"))
    assert (tmp_path / "phase0.png").exists()
    assert len(rows) == 2
    for r in rows:
        assert "slope_raw" in r and "slope_detrended" in r
