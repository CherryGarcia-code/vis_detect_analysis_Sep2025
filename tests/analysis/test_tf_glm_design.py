"""Tests for FIR lagged-column builders (Task 4)."""
import numpy as np
from visdetect.analysis.tf_glm import fir_event, fir_continuous


def test_fir_event_places_unit_at_each_lag():
    edges = np.arange(0.0, 0.5, 0.05)            # 10 bins
    ev = np.array([0.20])                        # event at bin 4
    X = fir_event(ev, edges, (0.0, 0.15), 0.05)  # lags 0,0.05,0.10 -> 3 cols
    assert X.shape == (10, 3)
    assert X[4, 0] == 1 and X[5, 1] == 1 and X[6, 2] == 1
    assert X[:, 0].sum() == 1


def test_fir_event_negative_lags():
    edges = np.arange(0.0, 0.5, 0.05)
    ev = np.array([0.20])
    X = fir_event(ev, edges, (-0.10, 0.05), 0.05)  # lags -0.10,-0.05,0.0
    assert X[2, 0] == 1 and X[3, 1] == 1 and X[4, 2] == 1


def test_fir_continuous_shifts():
    sig = np.array([1.0, 2.0, 3.0, 4.0])
    X = fir_continuous(sig, (0.0, 0.10), 0.05)   # lags 0, 0.05 -> 2 cols
    assert X.shape == (4, 2)
    np.testing.assert_array_equal(X[:, 0], [1, 2, 3, 4])   # lag 0
    np.testing.assert_array_equal(X[:, 1], [0, 1, 2, 3])   # lag +1 bin (causal)
