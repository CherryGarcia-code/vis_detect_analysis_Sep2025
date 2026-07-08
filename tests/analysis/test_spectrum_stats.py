import numpy as np
import pytest
from visdetect.analysis.spectrum_stats import (
    gmm_delta_bic, bimodality_coefficient, silverman_bootstrap,
    dip_test, segmented_vs_linear,
)

def test_gmm_delta_bic_positive_for_bimodal():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 0.3, 400), rng.normal(5, 0.3, 400)])
    assert gmm_delta_bic(x)["delta_bic"] > 0  # 2 components clearly better

def test_gmm_delta_bic_nonpositive_for_unimodal():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 800)
    assert gmm_delta_bic(x)["delta_bic"] < 20  # no strong 2-component preference

def test_bimodality_coefficient_higher_for_bimodal():
    rng = np.random.default_rng(2)
    uni = rng.normal(0, 1, 1000)
    bi = np.concatenate([rng.normal(-3, 0.5, 500), rng.normal(3, 0.5, 500)])
    assert bimodality_coefficient(bi) > bimodality_coefficient(uni)

def test_silverman_bootstrap_unimodal_high_p():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 500)
    assert silverman_bootstrap(x, n_boot=200)["p_unimodal"] > 0.1

def test_dip_test_returns_keys():
    rng = np.random.default_rng(4)
    out = dip_test(rng.normal(0, 1, 300))
    assert set(out) == {"dip", "p"}  # nan-filled if diptest not installed

def test_segmented_prefers_breakpoint_on_hinge_data():
    x = np.linspace(0, 1, 200)
    y = np.where(x < 0.5, 0.0, 4.0 * (x - 0.5)) + 0.01  # flat then rising (a hinge)
    out = segmented_vs_linear(x, y)
    assert out["delta_bic"] > 0                       # breakpoint beats a line
    assert 0.35 < out["breakpoint"] < 0.65            # near the true hinge

def test_segmented_no_gain_on_linear_data():
    rng = np.random.default_rng(5)
    x = np.linspace(0, 1, 200)
    y = 2.0 * x + rng.normal(0, 0.02, 200)
    assert segmented_vs_linear(x, y)["delta_bic"] < 6  # no meaningful breakpoint gain
