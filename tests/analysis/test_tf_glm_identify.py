import numpy as np
from visdetect.analysis.tf_glm import (TFGLMConfig, tf_pulse_peth,
                                        pulse_times_from_tf, identify_tf_responsive,
                                        DesignMatrix)

def test_tf_pulse_peth_triggers():
    edges = np.arange(0.0, 1.0, 0.05)
    sig = np.zeros(edges.size); sig[10] = 5.0       # impulse at t=0.5
    pulses = np.array([0.5])
    t, peth = tf_pulse_peth(sig, edges, pulses, (-0.15, 0.20), 0.05)
    assert peth[np.argmin(np.abs(t - 0.0))] == 5.0

def test_pulse_times_split_by_sd():
    # fabricate a design with tf_bins having a clear +/- excursion
    edges = np.arange(0.0, 1.0, 0.05)
    tf = np.zeros(edges.size); tf[5] = 1.0; tf[15] = -1.0   # +4SD, -4SD if SD~0.25
    d = DesignMatrix(X=np.zeros((edges.size, 0)), col_groups={}, bin_edges=edges,
                     trial_index=np.zeros(edges.size, int), tf_bins=tf)
    cfg = TFGLMConfig()
    fast, slow = pulse_times_from_tf(d, cfg)
    assert fast.size == 1 and slow.size == 1
    assert fast[0] < slow[0]
