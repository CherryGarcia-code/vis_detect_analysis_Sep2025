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
