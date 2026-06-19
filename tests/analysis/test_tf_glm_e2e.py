import numpy as np
from visdetect.analysis.tf_glm import (TFGLMConfig, TrialRegressors, assemble_design,
    count_vector, fit_poisson_cv, make_trial_folds, identify_tf_responsive)

def _session(n_trials=40, dur=6.0, bin_s=0.05, tf_gain=0.0, lick_gain=0.0, seed=0):
    rng = np.random.default_rng(seed)
    trials, spikes = [], []
    for i in range(n_trials):
        t0 = i * (dur + 1.0)
        n = int(dur / bin_s)
        tf = np.zeros(n); nb = n // 2
        tf[:nb] = 2 ** rng.normal(0, 0.25, nb)       # linear TF, log2 N(0,0.25)
        licks = np.array([t0 + dur - 0.5])
        tr = TrialRegressors(t_start=t0, t_end=t0 + dur, change_time=t0 + dur/2,
            change_size=2.0, tf_bins=tf, lick_times=licks, reward_time=np.nan,
            abort_time=np.nan, wheel_bins=np.zeros(n), phase_bins=None)
        trials.append(tr)
    cfg = TFGLMConfig(n_folds=5, lambdas=(1e-2, 1e-1, 1.0))
    design = assemble_design(trials, cfg)
    # synth rate: baseline + tf_gain * (log2 tf at lag 0) + lick bump
    log2tf = np.where(design.tf_bins > 0, np.log2(np.clip(design.tf_bins, 1e-9, None)), 0.0)
    lograte = -1.5 + tf_gain * log2tf
    rate = np.exp(lograte)
    y = rng.poisson(rate).astype(float)
    return trials, design, y, cfg

def test_tf_neuron_is_responsive():
    trials, design, y, cfg = _session(tf_gain=1.5, seed=1)
    folds = make_trial_folds(design.trial_index, cfg.n_folds, cfg.seed)
    full = fit_poisson_cv(design.X, y, cfg, folds)
    Xr = design.X.copy(); Xr[:, design.col_groups["tf"]] = 0.0
    red = fit_poisson_cv(Xr, y, cfg, folds)
    out = identify_tf_responsive(design, y, full, red, cfg)
    assert out["c1_r"] > 0.2 and out["c2_p"] < 0.01 and out["is_responsive"]

def test_flat_neuron_not_responsive():
    trials, design, y, cfg = _session(tf_gain=0.0, seed=2)
    folds = make_trial_folds(design.trial_index, cfg.n_folds, cfg.seed)
    full = fit_poisson_cv(design.X, y, cfg, folds)
    Xr = design.X.copy(); Xr[:, design.col_groups["tf"]] = 0.0
    red = fit_poisson_cv(Xr, y, cfg, folds)
    out = identify_tf_responsive(design, y, full, red, cfg)
    assert not out["is_responsive"]
    assert out["c1_r"] < 0.2
