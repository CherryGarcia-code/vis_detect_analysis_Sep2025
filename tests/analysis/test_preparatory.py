import numpy as np
import pytest
from visdetect.analysis import preparatory as P


def test_baseline_mean_sd_guards_tiny_sd():
    mu, sd = P.baseline_mean_sd(np.full((5, 8), 3.0))  # zero variance
    assert mu == pytest.approx(3.0)
    assert sd == pytest.approx(3.0)  # sd<1e-6 -> max(mu,1)=3.0


def test_zscore_and_active_mask():
    z = P.zscore_trace(np.array([0.0, 5.0, -5.0]), mu=0.0, sd=1.0)
    m = P.active_mask(z)  # |z|>2.576
    assert list(m) == [False, True, True]


def test_fraction_active_baseline_subtraction():
    A = np.array([[1, 1, 0, 0], [1, 0, 0, 0]], float)  # 2 units x 4 bins
    frac = P.fraction_active(A, baseline_bins=slice(2, 4))  # baseline frac = 0
    assert frac[0] == pytest.approx(1.0)
    assert frac[1] == pytest.approx(0.5)


def test_bootstrap_fraction_ci_brackets_mean():
    rng = np.random.default_rng(0)
    A = (rng.random((60, 10)) < 0.5).astype(float)
    mean, lo, hi = P.bootstrap_fraction_ci(A, n=500, seed=1)
    assert np.all(lo <= mean + 1e-9) and np.all(mean - 1e-9 <= hi)


def test_population_onset_detects_sustained_rise():
    t = np.arange(-2, 1, 0.025)
    frac = np.zeros_like(t)
    lo = np.zeros_like(t)
    on_idx = np.argmin(np.abs(t - (-0.5)))
    frac[on_idx:] = 0.4
    lo[on_idx:] = 0.1  # sustained, >0.1, ci>0
    onset = P.population_onset(t, frac, lo)
    assert onset == pytest.approx(-0.5, abs=0.03)


def test_population_onset_returns_nan_when_flat():
    t = np.arange(-2, 1, 0.025)
    assert np.isnan(P.population_onset(t, np.zeros_like(t), np.zeros_like(t)))


def test_cell_onset_single_bin_blip_rejected():
    t = np.arange(-1, 1, 0.025)
    z = np.zeros_like(t)
    z[10] = 5.0  # one bin only -> not sustained 80ms
    assert np.isnan(P.cell_onset(t, z))


def test_width_deciles_equal_count():
    w = np.arange(100.0)
    idx, edges = P.width_deciles(w, n=10)
    counts = np.bincount(idx[idx >= 0])
    assert counts.min() >= 9 and len(edges) == 11


def test_pulse_half_peak_width_triangle():
    t = np.linspace(0, 1, 201)
    resp = np.maximum(0, 1 - np.abs(t - 0.2) / 0.1)  # peak at 0.2, half-width 0.1
    w, pk = P.pulse_half_peak_width(resp, t)
    assert pk == pytest.approx(0.2, abs=0.02)
    assert w == pytest.approx(0.1, abs=0.02)
