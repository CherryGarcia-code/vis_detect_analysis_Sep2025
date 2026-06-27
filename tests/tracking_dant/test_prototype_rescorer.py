import numpy as np

import prototype_rescorer as pr


def test_pearson_basic_and_degenerate():
    assert pr._pearson([1, 2, 3], [1, 2, 3]) == 1.0
    assert pr._pearson([1, 2, 3], [3, 2, 1]) == -1.0
    assert np.isnan(pr._pearson([1, 1, 1], [1, 2, 3]))   # zero variance -> NaN
    assert np.isnan(pr._pearson([1.0], [1.0]))           # too few points


def test_anatomy_factor_shank_gate_and_depth_penalty():
    a = {"shank": 0, "depth": 1000.0}
    b_same = {"shank": 0, "depth": 1000.0}
    b_far = {"shank": 0, "depth": 1000.0 + pr.DEPTH_TAU_UM}   # one tau away
    b_shank = {"shank": 1, "depth": 1000.0}
    assert pr.anatomy_factor(a, b_same) == 1.0               # same place -> 1
    assert pr.anatomy_factor(a, b_shank) == 0.0              # cross-shank -> hard 0
    assert abs(pr.anatomy_factor(a, b_far) - np.exp(-1.0)) < 1e-6   # one-tau depth penalty


def test_shape_features_returns_two_correlations():
    f = {}
    wf_a = np.array([0.0, 1.0, 2.0, 1.0]); wf_b = wf_a.copy()
    isi_a = np.array([0.1, 0.5, 0.3, 0.1]); isi_b = isi_a.copy()
    sf = pr.shape_features(f, f, wf_a, wf_b, isi_a, isi_b)
    assert len(sf) == 2
    assert sf[0] == 1.0 and sf[1] == 1.0      # identical -> corr 1
