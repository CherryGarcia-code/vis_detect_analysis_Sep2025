# tests/analysis/test_continuum_common.py
import sys, os
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                       "scripts/tf_responsiveness/state_conditioned"))
from continuum_common import decile_stats, width_bin_assign  # noqa: E402

def test_decile_stats_monotone_positive_relationship():
    rng = np.random.default_rng(0)
    x = rng.random(500)
    y = 3.0 * x + rng.normal(0, 0.1, 500)          # strong positive
    d = decile_stats(x, y, n_bins=10)
    assert d["rho"] > 0.9 and d["p"] < 1e-20
    assert len(d["centers"]) == 10 and len(d["mean"]) == 10
    # bin means increase with width
    assert d["mean"][-1] > d["mean"][0]
    # CI brackets the mean
    assert np.all(d["ci_lo"] <= d["mean"] + 1e-9) and np.all(d["ci_hi"] >= d["mean"] - 1e-9)

def test_decile_stats_deterministic():
    rng = np.random.default_rng(1); x = rng.random(300); y = rng.random(300)
    a = decile_stats(x, y, seed=42); b = decile_stats(x, y, seed=42)
    assert np.allclose(a["ci_lo"], b["ci_lo"]) and np.allclose(a["ci_hi"], b["ci_hi"])

def test_decile_stats_handles_nan():
    x = np.array([1., 2., np.nan, 4., 5., 6., 7., 8., 9., 10., 11., 12.])
    y = np.array([1., np.nan, 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.])
    d = decile_stats(x, y, n_bins=3)
    assert np.isfinite(d["rho"])

def test_width_bin_assign_equal_count():
    w = np.arange(100.0)
    idx, edges = width_bin_assign(w, n=5)
    # 5 bins, each ~20 cells, monotone assignment
    counts = np.bincount(idx, minlength=5)
    assert counts.min() >= 18 and counts.max() <= 22
    assert idx[0] == 0 and idx[-1] == 4
