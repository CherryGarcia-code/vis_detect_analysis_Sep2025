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
