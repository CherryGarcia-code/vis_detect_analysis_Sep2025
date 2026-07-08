import numpy as np
import pytest
from visdetect.analysis import population_field as pf


def _np2_positions():
    """4-shank NP2.0-like geometry: x in {0,32,250,282,500,532,750,782}, y 1500..2200."""
    xs = [0, 32, 250, 282, 500, 532, 750, 782]
    ys = np.arange(1500, 2205, 15.0)  # 15 um row pitch
    pos = np.array([[x, y] for y in ys for x in xs], dtype=float)
    return pos


def test_depth_bin_edges_cover_active_band():
    pos = _np2_positions()
    edges = pf.depth_bin_edges(pos, depth_bin_um=60.0)
    assert edges[0] <= pos[:, 1].min()
    assert edges[-1] >= pos[:, 1].max()
    assert np.allclose(np.diff(edges), 60.0)


def test_robust_unit_depth_weighted_centroid():
    # 3 channels at y = 0, 100, 200; ptp concentrated as 1:2:1 -> centroid = 100
    n_samp = 82
    mw = np.zeros((n_samp, 3))
    mw[:, 0] = np.linspace(-0.5, 0.5, n_samp)   # ptp 1
    mw[:, 1] = np.linspace(-1.0, 1.0, n_samp)   # ptp 2
    mw[:, 2] = np.linspace(-0.5, 0.5, n_samp)   # ptp 1
    pos = np.array([[0.0, 0.0], [0.0, 100.0], [0.0, 200.0]])
    assert pf.robust_unit_depth(mw, pos) == pytest.approx(100.0)


def test_robust_unit_depth_zero_amplitude_is_nan():
    mw = np.zeros((82, 3))
    pos = np.array([[0.0, 0.0], [0.0, 100.0], [0.0, 200.0]])
    assert np.isnan(pf.robust_unit_depth(mw, pos))
