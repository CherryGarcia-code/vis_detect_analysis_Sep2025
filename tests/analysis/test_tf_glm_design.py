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


def test_fir_continuous_negative_window_past_signal_no_crash():
    """Fix 1: offsets reaching -8 on a 4-bin signal must not crash.

    win=(-0.40, 0.05), bin_s=0.05 -> offsets [-8,-7,-6,-5,-4,-3,-2,-1, 0] (9 cols).
    The most-negative-offset column (off=-8) is fully shifted out -> all zeros.
    The off=0 column equals the signal.
    """
    sig = np.array([1.0, 2.0, 3.0, 4.0])
    X = fir_continuous(sig, (-0.40, 0.05), 0.05)
    assert X.shape == (4, 9), f"Expected (4, 9), got {X.shape}"
    # most-negative offset column (j=0, off=-8) is fully shifted out
    np.testing.assert_array_equal(X[:, 0], [0, 0, 0, 0])
    # off=0 column (j=8, the last column) equals the signal
    np.testing.assert_array_equal(X[:, 8], [1, 2, 3, 4])


def test_fir_event_boundary_event_contributes_in_window():
    """Fix 2: an event just before the window should still contribute via positive lags.

    edges = 10 bins [0.0, 0.05, ..., 0.45].
    event at t=-0.03 -> bin index -1 (outside window).
    win=(0.0, 0.15) -> offsets [0, 1, 2] (3 cols).
    At lag offset +2: b = -1 + 2 = 1, which is in-window -> X[1, 2] == 1.
    Before Fix 2 the event was dropped by the outer idx filter, so column 2 was all-zero.
    """
    edges = np.arange(0.0, 0.5, 0.05)   # 10 bins
    ev = np.array([-0.03])               # event at bin index -1
    X = fir_event(ev, edges, (0.0, 0.15), 0.05)  # offsets [0,1,2] -> 3 cols
    assert X.shape == (10, 3)
    assert X[1, 2] == 1, f"Expected X[1,2]=1 but got {X[1,2]}"
