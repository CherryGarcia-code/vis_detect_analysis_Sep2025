# tests/analysis/test_tf_glm_assemble.py
import numpy as np
from visdetect.analysis.tf_glm import (TFGLMConfig, TrialRegressors,
                                        assemble_design, count_vector)

def _toy_trial(t0, dur=2.0, bin_s=0.05, change_size=2.0, seed=0):
    n = int(dur / bin_s)
    rng = np.random.default_rng(seed)
    tf = np.zeros(n); tf[: n // 2] = rng.normal(0, 0.25, n // 2)  # baseline only
    return TrialRegressors(
        t_start=t0, t_end=t0 + dur, change_time=t0 + dur / 2, change_size=change_size,
        tf_bins=tf, lick_times=np.array([t0 + 1.6]), reward_time=t0 + 1.7,
        abort_time=np.nan, wheel_bins=np.zeros(n), phase_bins=None)

def test_assemble_shapes_and_groups():
    cfg = TFGLMConfig()
    trials = [_toy_trial(10.0), _toy_trial(20.0, seed=1)]
    d = assemble_design(trials, cfg)
    assert d.X.shape[0] == d.bin_edges.size == d.trial_index.size
    # six change-size columns groups present
    assert "tf" in d.col_groups and "lick_prep" in d.col_groups
    # TF group width == number of tf lags (1.5/0.05 = 30)
    assert d.col_groups["tf"].stop - d.col_groups["tf"].start == 30
    assert d.tf_bins.size == d.X.shape[0]

def test_count_vector_matches_rows():
    cfg = TFGLMConfig()
    trials = [_toy_trial(10.0)]
    d = assemble_design(trials, cfg)
    y = count_vector(trials, np.array([10.3, 10.32, 11.0]), d)
    assert y.size == d.X.shape[0] and y.sum() == 3

def test_wheel_columns_standardized_tf_not():
    """Nuisance continuous regressors (wheel) are standardized to unit variance;
    the TF block is NOT (it is already in z-scored SD-octave units); event
    indicators (e.g. lick_prep) left as 0/1."""
    cfg = TFGLMConfig()
    # nonzero wheel so the wheel columns have variance to standardize
    trials = [_toy_trial(10.0), _toy_trial(20.0, seed=1)]
    rng = np.random.default_rng(7)
    for tr in trials:
        tr.wheel_bins = rng.normal(0.0, 1.0, tr.wheel_bins.size)
    d = assemble_design(trials, cfg)
    # wheel: standardized
    sl = d.col_groups["wheel"]
    cols = d.X[:, sl]
    for j in range(cols.shape[1]):
        c = cols[:, j]
        if np.std(c) < 1e-8:
            continue
        assert abs(np.mean(c)) < 1e-6, f"wheel col {j} mean {np.mean(c)}"
        assert abs(np.std(c) - 1.0) < 1e-6, f"wheel col {j} std {np.std(c)}"
    # tf: NOT standardized — lag-0 column equals log2(TF)/0.25 of the baseline
    # bins, which is NOT unit-variance (its SD is ~1 only because the baseline
    # noise is ~0.25 octave; the point is the code does NOT force std==1).
    tf_sl = d.col_groups["tf"]
    tf_lag0 = d.X[:, tf_sl.start]            # off=0 column
    # reconstruct expected octaves directly from the trials' tf_bins
    import numpy as _np
    expected = _np.concatenate([
        _np.where(tr.tf_bins > 0, _np.log2(_np.clip(tr.tf_bins, 1e-12, None)) / 0.25, 0.0)
        for tr in trials])
    _np.testing.assert_allclose(tf_lag0, expected, atol=1e-9)
    # event indicator column untouched: still 0/1
    ev = d.X[:, d.col_groups["lick_prep"]]
    assert set(np.unique(ev)).issubset({0.0, 1.0})


def test_tf_columns_are_symmetric_octaves():
    """The lag-0 'tf' column is log2(TF)/0.25: a 2x-up bin (TF=2) -> +4 SD and a
    2x-down bin (TF=0.5) -> -4 SD (symmetric), NOT the asymmetric linear values
    (+1 Hz vs -0.5 Hz). Post-change / masked bins (TF<=0) stay 0."""
    cfg = TFGLMConfig()
    n = 40            # >= tf kernel width (1.5/0.05 = 30 lags) so fir_continuous fits
    tf = np.zeros(n)
    tf[2] = 2.0      # 2x up  -> log2(2)/0.25  = +4
    tf[3] = 0.5      # 2x down -> log2(0.5)/0.25 = -4
    tf[4] = 1.0      # geomean -> log2(1)/0.25  =  0
    # tf[5..] stay 0 (post-change / masked)
    tr = TrialRegressors(
        t_start=0.0, t_end=n * cfg.bin_s, change_time=20 * cfg.bin_s,
        change_size=2.0, tf_bins=tf, lick_times=np.zeros(0), reward_time=np.nan,
        abort_time=np.nan, wheel_bins=np.zeros(n), phase_bins=None)
    d = assemble_design([tr], cfg)
    tf_lag0 = d.X[:, d.col_groups["tf"].start]   # off=0 column
    assert abs(tf_lag0[2] - 4.0) < 1e-9, tf_lag0[2]      # 2x up  = +4 SD
    assert abs(tf_lag0[3] - (-4.0)) < 1e-9, tf_lag0[3]   # 2x down = -4 SD
    assert abs(tf_lag0[4]) < 1e-9, tf_lag0[4]            # geomean = 0
    # symmetric: equal-and-opposite (NOT the asymmetric linear +1 / -0.5)
    assert abs(tf_lag0[2] + tf_lag0[3]) < 1e-9
    # masked / post-change bins are 0
    assert abs(tf_lag0[0]) < 1e-9 and abs(tf_lag0[5]) < 1e-9


def test_tf_block_not_variance_standardized_away():
    """Constant-injection check: if the tf regressor were per-column z-scored,
    scaling the input octaves by any constant would leave the columns unchanged
    (std forced to 1). Because the tf block is NOT standardized, doubling the
    baseline-TF excursion doubles the tf column magnitude — proving the
    z-score step does not touch it. (wheel, by contrast, IS standardized and is
    invariant to such scaling.)"""
    cfg = TFGLMConfig()
    rng = np.random.default_rng(3)
    n = 40
    tf = np.zeros(n); tf[: n // 2] = 2.0 ** rng.normal(0, 0.25, n // 2)
    wheel = rng.normal(0, 1.0, n)
    def _mk(scale):
        tr = TrialRegressors(
            t_start=0.0, t_end=n * cfg.bin_s, change_time=(n // 2) * cfg.bin_s,
            change_size=2.0, tf_bins=tf ** scale, lick_times=np.zeros(0),
            reward_time=np.nan, abort_time=np.nan,
            wheel_bins=wheel * scale, phase_bins=None)
        return assemble_design([tr], cfg)
    d1 = _mk(1.0)
    d2 = _mk(2.0)   # tf**2 doubles octaves (log2(tf**2)=2 log2 tf); wheel*2
    tf1 = d1.X[:, d1.col_groups["tf"].start]
    tf2 = d2.X[:, d2.col_groups["tf"].start]
    # tf NOT standardized: doubling octaves doubles the column (within nonzero bins)
    nz = np.abs(tf1) > 1e-9
    assert np.allclose(tf2[nz], 2.0 * tf1[nz], atol=1e-9), "tf was standardized away!"
    # wheel IS standardized: scaling the input leaves the standardized column unchanged
    w1 = d1.X[:, d1.col_groups["wheel"].start]
    w2 = d2.X[:, d2.col_groups["wheel"].start]
    assert np.allclose(w1, w2, atol=1e-9), "wheel should be standardized (scale-invariant)"
