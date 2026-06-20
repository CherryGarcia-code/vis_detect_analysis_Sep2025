import numpy as np
import pytest
from visdetect.analysis.tf_glm import (TFGLMConfig, tf_pulse_peth,
                                        pulse_times_from_tf, identify_tf_responsive,
                                        DesignMatrix, FitResult)

# ── pulse helpers (still used for exemplar visualisation) ────────────────────
def test_tf_pulse_peth_triggers():
    edges = np.arange(0.0, 1.0, 0.05)
    sig = np.zeros(edges.size); sig[10] = 5.0       # impulse at t=0.5
    pulses = np.array([0.5])
    t, peth = tf_pulse_peth(sig, edges, pulses, (-0.15, 0.20), 0.05)
    assert peth[np.argmin(np.abs(t - 0.0))] == 5.0

def test_pulse_times_split_by_sd():
    # Real ±0.5-SD split test with enough valid bins to clear the >=10-valid guard.
    # tf_bins are log2-encoded (negatives present -> taken as log2 directly).
    edges = np.arange(0.0, 2.0, 0.05)            # 40 bins
    tf = np.zeros(edges.size)
    tf[2:8] = 1.0                                 # 6 fast bins (early)
    tf[20:26] = -1.0                              # 6 slow bins (late)
    tf[10:16] = 0.05                              # filler nonzero (within +/-0.5SD)
    d = DesignMatrix(X=np.zeros((edges.size, 0)), col_groups={}, bin_edges=edges,
                     trial_index=np.zeros(edges.size, int), tf_bins=tf)
    cfg = TFGLMConfig()
    fast, slow = pulse_times_from_tf(d, cfg)
    assert fast.size > 0 and slow.size > 0
    assert fast.size == 6 and slow.size == 6
    assert fast.max() < slow.min()


# ── dense paired full-vs-reduced identification ──────────────────────────────
def _design_stub(n_bins, n_tf_cols=3):
    """Minimal DesignMatrix so identify_tf_responsive's kernel block runs."""
    bin_edges = np.arange(n_bins, dtype=float) * 0.05
    return DesignMatrix(
        X=np.zeros((n_bins, n_tf_cols)),
        col_groups={"tf": slice(0, n_tf_cols)},
        bin_edges=bin_edges,
        trial_index=np.zeros(n_bins, int),
        tf_bins=np.ones(n_bins),
    )


def _fit_stub(pred, fold_ids, n_tf_cols=3):
    coef = [np.zeros(n_tf_cols) for _ in np.unique(fold_ids)]
    return FitResult(pred=np.asarray(pred, float), fold_ids=np.asarray(fold_ids),
                     coef_by_fold=coef, best_lambdas=[1.0] * len(coef))


def test_dense_identify_responsive_when_full_beats_reduced():
    # FULL prediction tracks the actual counts; REDUCED is near-constant noise.
    rng = np.random.default_rng(0)
    n_folds, per = 5, 40
    n = n_folds * per
    fold_ids = np.repeat(np.arange(n_folds), per)
    y = rng.poisson(2.0, n).astype(float)
    # full pred strongly correlated with y on every fold; reduced ~ flat
    pred_full = 0.5 * y + rng.normal(0, 0.3, n)
    pred_red = np.full(n, y.mean()) + rng.normal(0, 0.3, n)
    cfg = TFGLMConfig()
    out = identify_tf_responsive(_design_stub(n), y,
                                 _fit_stub(pred_full, fold_ids),
                                 _fit_stub(pred_red, fold_ids), cfg)
    assert out["c1_r"] > 0.2
    assert out["c2_p"] < 0.01
    assert out["is_responsive"]
    assert out["r_full_mean"] >= out["r_red_mean"]


def test_dense_identify_not_responsive_when_full_equals_reduced():
    # FULL and REDUCED predict identically (TF kernel adds nothing) -> diff ~ 0,
    # C2 non-significant -> not responsive even though c1_r may be high.
    rng = np.random.default_rng(1)
    n_folds, per = 5, 40
    n = n_folds * per
    fold_ids = np.repeat(np.arange(n_folds), per)
    y = rng.poisson(2.0, n).astype(float)
    shared = 0.5 * y + rng.normal(0, 0.3, n)
    cfg = TFGLMConfig()
    out = identify_tf_responsive(_design_stub(n), y,
                                 _fit_stub(shared.copy(), fold_ids),
                                 _fit_stub(shared.copy(), fold_ids), cfg)
    assert not out["is_responsive"]
    assert out["c2_p"] >= 0.01 or not np.isfinite(out["c2_p"])


def test_criterion_c2_vs_c1_and_c2_when_c1_low():
    # Construct a case where C2 is significant (FULL beats REDUCED on every fold)
    # but the FULL model's overall predictive correlation c1_r is BELOW 0.2:
    #   - pred_full is only WEAKLY (but consistently) correlated with y -> low c1_r
    #   - pred_red is ANTI-correlated with y -> r_full - r_red > 0 on every fold
    # Default "c2": responsive (TF kernel adds paired held-out power).
    # "c1_and_c2": NOT responsive (c1_r fails the raw-bin 0.2 floor).
    rng = np.random.default_rng(7)
    n_folds, per = 6, 60
    n = n_folds * per
    fold_ids = np.repeat(np.arange(n_folds), per)
    y = rng.poisson(2.0, n).astype(float)
    # weak positive signal -> small but reliably positive r_full (< 0.2)
    pred_full = 0.05 * y + rng.normal(0, 1.0, n)
    # reduced is reliably anti-correlated -> r_red < 0 on every fold
    pred_red = -0.20 * y + rng.normal(0, 0.3, n)

    out_c2 = identify_tf_responsive(_design_stub(n), y,
                                    _fit_stub(pred_full, fold_ids),
                                    _fit_stub(pred_red, fold_ids),
                                    TFGLMConfig(responsive_criterion="c2"))
    # sanity: this is exactly the c1-low, c2-significant regime we want to test
    assert out_c2["c1_r"] < 0.2
    assert out_c2["c2_p"] < 0.01
    assert out_c2["is_responsive"]                 # C2 alone -> responsive

    out_both = identify_tf_responsive(_design_stub(n), y,
                                      _fit_stub(pred_full, fold_ids),
                                      _fit_stub(pred_red, fold_ids),
                                      TFGLMConfig(responsive_criterion="c1_and_c2"))
    # same c1_r / c2_p, but the conjunction fails on the raw-bin C1 floor
    assert out_both["c1_r"] < 0.2
    assert out_both["c2_p"] < 0.01
    assert not out_both["is_responsive"]           # c1_and_c2 -> NOT responsive


def test_unknown_responsive_criterion_raises():
    n_folds, per = 5, 40
    n = n_folds * per
    fold_ids = np.repeat(np.arange(n_folds), per)
    rng = np.random.default_rng(3)
    y = rng.poisson(2.0, n).astype(float)
    pred = 0.5 * y + rng.normal(0, 0.3, n)
    cfg = TFGLMConfig(responsive_criterion="bogus")
    with pytest.raises(ValueError):
        identify_tf_responsive(_design_stub(n), y,
                               _fit_stub(pred, fold_ids),
                               _fit_stub(pred, fold_ids), cfg)


def test_dense_identify_skips_low_variance_folds():
    # A fold with <10 finite bins or zero-variance prediction must be skipped,
    # not crash; here one fold is all-NaN pred.
    n_folds, per = 4, 20
    n = n_folds * per
    fold_ids = np.repeat(np.arange(n_folds), per)
    rng = np.random.default_rng(2)
    y = rng.poisson(2.0, n).astype(float)
    pred_full = 0.5 * y + rng.normal(0, 0.3, n)
    pred_red = y.mean() + rng.normal(0, 0.3, n)   # near-flat but nonzero variance
    pred_full[fold_ids == 0] = np.nan             # one bad fold
    cfg = TFGLMConfig()
    out = identify_tf_responsive(_design_stub(n), y,
                                 _fit_stub(pred_full, fold_ids),
                                 _fit_stub(pred_red, fold_ids), cfg)
    assert out["n_folds_used"] == n_folds - 1
    assert np.isfinite(out["c1_r"])
