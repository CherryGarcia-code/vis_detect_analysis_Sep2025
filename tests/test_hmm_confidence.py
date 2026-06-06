"""Tests for F14: posterior-confidence gating helper."""

import numpy as np
import pytest

from visdetect.analysis.hmm import assign_states_with_confidence


def test_assign_high_confidence_keeps_argmax():
    """When every trial has γ_max > threshold, return argmax."""
    # T=3, K=3.  Each row has a clear winner above 0.8.
    posteriors = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.85, 0.05],
        [0.05, 0.1, 0.85],
    ])
    out = assign_states_with_confidence(posteriors, threshold=0.8)
    np.testing.assert_array_equal(out, np.array([0, 1, 2]))


def test_assign_low_confidence_returns_minus_one():
    """When γ_max <= threshold, the trial gets -1 (unassigned)."""
    posteriors = np.array([
        [0.5, 0.4, 0.1],   # max 0.5 < 0.8 → -1
        [0.1, 0.85, 0.05], # max 0.85 → state 1
        [0.45, 0.45, 0.10],# max 0.45 < 0.8 → -1
    ])
    out = assign_states_with_confidence(posteriors, threshold=0.8)
    np.testing.assert_array_equal(out, np.array([-1, 1, -1]))


def test_assign_threshold_zero_passes_everything():
    """Threshold = 0 always returns argmax (no -1)."""
    posteriors = np.array([
        [0.34, 0.33, 0.33],
        [0.4, 0.4, 0.2],
    ])
    out = assign_states_with_confidence(posteriors, threshold=0.0)
    # argmax breaks ties at the first equal index — that's fine for our test.
    assert (out >= 0).all()
    np.testing.assert_array_equal(out, np.argmax(posteriors, axis=1))


def test_assign_empty_input():
    """Empty input returns empty array (shape preserved)."""
    posteriors = np.empty((0, 3))
    out = assign_states_with_confidence(posteriors, threshold=0.5)
    assert out.shape == (0,)


def test_assign_dtype_is_int():
    """Output is always integer (for indexing downstream)."""
    posteriors = np.array([[0.9, 0.1], [0.5, 0.5]])
    out = assign_states_with_confidence(posteriors, threshold=0.6)
    assert np.issubdtype(out.dtype, np.integer)


def test_decode_session_gated_column_only_when_threshold_set(monkeypatch):
    """`decode_session` adds hmm_state_gated only when confidence_threshold is given."""
    import pandas as pd
    from visdetect.analysis.hmm import GLMHMM, decode_session

    # Build a minimal fake session via monkey-patching prepare_session_data
    # — we just need the data dict shape; the model needs to fit it.
    fake_df = pd.DataFrame({
        "outcome": ["hit", "miss", "fa"],
        "is_hit": [True, False, False],
        "is_fa":  [False, False, True],
        "is_go":  [True, True, False],
        "is_catch": [False, False, True],
        "change_size": [2.0, 2.0, 1.0],
    })
    fake_data = {
        "y": np.array([1.0, 0.0, 1.0]),
        "X": np.array([
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ]),
        "df": fake_df,
        "session_name": "fake",
        "feature_names": ["bias","stim","pc","pr","pel"],
    }

    from visdetect.analysis import hmm as hmm_mod
    monkeypatch.setattr(hmm_mod, "prepare_session_data", lambda s: fake_data)

    model = GLMHMM(n_states=2, n_features=5)
    model._init_params(seed=0)

    # without threshold -> no gated column
    out = decode_session(model, session=None)
    assert "hmm_state_gated" not in out.columns

    # with threshold -> gated column present
    out = decode_session(model, session=None, confidence_threshold=0.8)
    assert "hmm_state_gated" in out.columns
    assert out["hmm_state_gated"].dtype.kind in ("i", "u")
