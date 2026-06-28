import numpy as np
from visdetect.analysis.tf_glm import TFGLMConfig, trial_bin_edges, bin_spike_counts

def test_bin_edges_50ms():
    cfg = TFGLMConfig()
    assert cfg.bin_s == 0.05
    e = trial_bin_edges(10.0, 10.2, cfg.bin_s)
    np.testing.assert_allclose(e, [10.0, 10.05, 10.10, 10.15])

def test_bin_spike_counts():
    e = np.array([0.0, 0.05, 0.10, 0.15])
    st = np.array([0.01, 0.02, 0.12, 0.99])  # two in bin0, one in bin2, one past end
    c = bin_spike_counts(st, e)
    np.testing.assert_array_equal(c, [2, 0, 1, 0])
