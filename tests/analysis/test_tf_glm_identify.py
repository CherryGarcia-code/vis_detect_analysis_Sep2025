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
    # Real ±0.5-SD split test with enough valid bins to clear the >=10-valid guard.
    # tf_bins are log2-encoded (negatives present -> taken as log2 directly).
    # 6 clearly-positive bins (early) and 6 clearly-negative bins (late), plus
    # filler nonzero bins so >=12 are valid. With ~symmetric +/-1.0 excursions the
    # baseline log2 SD ~1.0 and threshold ~0.5 -> all +1.0 are fast, all -1.0 slow.
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
    # all fast pulses precede all slow pulses
    assert fast.max() < slow.min()
