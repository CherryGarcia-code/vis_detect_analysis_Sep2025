import numpy as np
import pytest
from visdetect.analysis.tf_glm import (TFGLMConfig, TrialRegressors, assemble_design,
    count_vector, fit_poisson_cv, make_trial_folds, identify_tf_responsive)

def _session(n_trials=60, dur=6.0, bin_s=0.05, tf_gain=0.0, base=0.0, lick_gain=0.0,
             seed=0, fast_fit=False):
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
    cfg = TFGLMConfig(n_folds=5, lambdas=(1e-2, 1e-1, 1.0), fast_fit=fast_fit)
    design = assemble_design(trials, cfg)
    # synth rate: baseline + tf_gain * (log2 tf at lag 0) + lick bump. The dense
    # held-out correlation is computed on z-scored single-bin counts, which are
    # Poisson-noisy; a clearly TF-driven neuron needs a firing rate of ~>=1 Hz
    # and a strong TF gain for corr(zscore(y), pred) to clear 0.2 (matching the
    # paper's regime, where TF-responsive units have a robust dense correlation).
    log2tf = np.where(design.tf_bins > 0, np.log2(np.clip(design.tf_bins, 1e-9, None)), 0.0)
    lograte = base + tf_gain * log2tf
    rate = np.exp(lograte)
    y = rng.poisson(rate).astype(float)
    return trials, design, y, cfg


def _run(tf_gain, seed, base=0.0, fast_fit=False):
    trials, design, y, cfg = _session(tf_gain=tf_gain, base=base, seed=seed,
                                      fast_fit=fast_fit)
    folds = make_trial_folds(design.trial_index, cfg.n_folds, cfg.seed)
    full = fit_poisson_cv(design.X, y, cfg, folds)
    Xr = design.X.copy(); Xr[:, design.col_groups["tf"]] = 0.0
    red = fit_poisson_cv(Xr, y, cfg, folds)
    return identify_tf_responsive(design, y, full, red, cfg)


@pytest.mark.parametrize("fast_fit", [False, True])
def test_tf_neuron_is_responsive(fast_fit):
    # Dense criterion: a neuron whose rate genuinely depends on TF has the FULL
    # model (with TF kernel) predict held-out activity better than the REDUCED
    # model -> r_full >> r_red (paired positive, significant) AND c1_r > 0.2.
    out = _run(tf_gain=3.0, base=0.0, seed=1, fast_fit=fast_fit)
    assert out["c1_r"] > 0.2
    assert out["c2_p"] < 0.01
    assert out["is_responsive"]
    # the TF kernel adds held-out predictive power
    assert out["r_full_mean"] >= out["r_red_mean"]


def test_flat_neuron_not_responsive():
    # A TF-independent neuron (same baseline rate, tf_gain=0): FULL and REDUCED
    # predict held-out activity equally poorly, and the FULL model has no real
    # TF signal to lock onto -> c1_r stays below 0.2 -> not responsive.
    out = _run(tf_gain=0.0, base=0.0, seed=2)
    assert not out["is_responsive"]
    assert out["c1_r"] < 0.2
