import numpy as np
from visdetect.analysis.tf_glm import (TFGLMConfig, fit_poisson_cv, make_trial_folds)

def test_make_trial_folds_keeps_trials_intact():
    trial_index = np.array([0,0,0, 1,1, 2,2,2,2, 3])
    folds = make_trial_folds(trial_index, n_folds=2, seed=0)
    # all rows of a trial share one fold
    for t in np.unique(trial_index):
        assert len(set(folds[trial_index == t])) == 1

def test_fit_recovers_known_rate():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(0, 1, n)
    X = x.reshape(-1, 1)
    rate = np.exp(-1.0 + 0.8 * x)         # true log-linear rate
    y = rng.poisson(rate).astype(float)
    cfg = TFGLMConfig(n_folds=5, lambdas=(1e-3, 1e-2, 1e-1))
    fold_ids = np.repeat(np.arange(5), n // 5)
    res = fit_poisson_cv(X, y, cfg, fold_ids=fold_ids)
    # held-out prediction correlates with true rate
    assert np.corrcoef(res.pred, rate)[0, 1] > 0.5
    # recovered slope (mean across folds) near 0.8
    slope = np.mean([c[0] for c in res.coef_by_fold])
    assert 0.5 < slope < 1.1


def test_fast_fit_recovers_and_is_cheaper():
    # Same synthetic as test_fit_recovers_known_rate, but with fast_fit=True:
    # λ is selected ONCE per unit (one train/val split) instead of nested per
    # outer fold. Must still recover the slope and held-out prediction, and must
    # do strictly fewer PoissonRegressor fits.
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(0, 1, n)
    X = x.reshape(-1, 1)
    rate = np.exp(-1.0 + 0.8 * x)
    y = rng.poisson(rate).astype(float)
    fold_ids = np.repeat(np.arange(10), n // 10)

    cfg_fast = TFGLMConfig(n_folds=10, lambdas=(1e-3, 1e-2, 1e-1), fast_fit=True)
    cfg_nest = TFGLMConfig(n_folds=10, lambdas=(1e-3, 1e-2, 1e-1), fast_fit=False)

    # Count PoissonRegressor.fit calls to prove fast mode is cheaper.
    import visdetect.analysis.tf_glm as tg
    calls = {"n": 0}
    orig = tg._fit_one

    def _counting_fit_one(Xtr, ytr, lam):
        calls["n"] += 1
        return orig(Xtr, ytr, lam)

    tg._fit_one = _counting_fit_one
    try:
        calls["n"] = 0
        res = fit_poisson_cv(X, y, cfg_fast, fold_ids=fold_ids)
        fast_fits = calls["n"]
        calls["n"] = 0
        fit_poisson_cv(X, y, cfg_nest, fold_ids=fold_ids)
        nested_fits = calls["n"]
    finally:
        tg._fit_one = orig

    # held-out prediction correlates with true rate
    assert np.corrcoef(res.pred, rate)[0, 1] > 0.5
    # recovered slope (mean across folds) near 0.8 (same tolerance)
    slope = np.mean([c[0] for c in res.coef_by_fold])
    assert 0.5 < slope < 1.1
    # held-out pred structure preserved: one coef array per outer fold
    assert len(res.coef_by_fold) == cfg_fast.n_folds
    # fast mode does strictly fewer fits (~len(lambdas)+n_folds vs nested)
    assert fast_fits < nested_fits
    assert fast_fits <= len(cfg_fast.lambdas) + cfg_fast.n_folds
