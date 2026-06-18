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
