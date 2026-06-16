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
