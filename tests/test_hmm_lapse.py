"""Tests for F3: lapse-model baseline."""

import numpy as np
import pytest
from scipy.special import expit

from visdetect.analysis.hmm import GLMHMMConfig, fit_lapse_model


def _generate_lapse_data(T=400, lapse_rate=0.1, seed=0):
    """Synthetic data: clean engaged GLM with a lapse rate of `lapse_rate`."""
    rng = np.random.default_rng(seed)
    w_engaged = np.array([-0.5, 2.0, 0.0, 0.0, 0.0])
    X = np.column_stack([
        np.ones(T),
        rng.uniform(-0.5, 2.0, T),
        rng.binomial(1, 0.5, T).astype(float),
        rng.binomial(1, 0.3, T).astype(float),
        rng.binomial(1, 0.2, T).astype(float),
    ])
    is_lapse = rng.binomial(1, lapse_rate, T).astype(bool)
    p_engaged = expit(X @ w_engaged)
    p_lapse = 0.5
    y = np.empty(T, dtype=float)
    y[~is_lapse] = rng.binomial(1, p_engaged[~is_lapse])
    y[is_lapse]  = rng.binomial(1, p_lapse, is_lapse.sum())
    return [{
        "y": y, "X": X, "df": None,
        "session_name": "lapse_sess",
        "feature_names": ["bias","stim","pc","pr","pel"],
    }]


def test_lapse_model_returns_glmhmm():
    sessions = _generate_lapse_data(T=300, lapse_rate=0.1, seed=1)
    cfg = GLMHMMConfig(max_iter=100, n_restarts=3, verbose=False)
    model = fit_lapse_model(sessions, n_features=5, config=cfg)
    assert model.n_states == 2
    # Lapse state's stimulus weight must be ~0 (constrained).
    np.testing.assert_allclose(model.weights[1, 1:], 0.0, atol=1e-6)


def test_lapse_model_transition_rows_identical():
    """Lapse model has identical transition rows (stimulus-independent lapse)."""
    sessions = _generate_lapse_data(T=300, lapse_rate=0.1, seed=2)
    cfg = GLMHMMConfig(max_iter=100, n_restarts=3, verbose=False)
    model = fit_lapse_model(sessions, n_features=5, config=cfg)
    A = model.transition_matrix
    np.testing.assert_allclose(A[0], A[1], atol=1e-3)


def test_lapse_model_recovers_engaged_weights():
    """On data with low lapse rate, engaged GLM should recover stimulus sensitivity."""
    sessions = _generate_lapse_data(T=600, lapse_rate=0.05, seed=3)
    cfg = GLMHMMConfig(max_iter=150, n_restarts=5, verbose=False)
    model = fit_lapse_model(sessions, n_features=5, config=cfg)
    # Engaged state has positive stim weight (~2.0); lapse has 0.
    stim_weights = model.weights[:, 1]
    assert stim_weights.max() > 1.0
    assert abs(stim_weights.min()) < 0.1


def test_fit_best_model_includes_lapse_row():
    """selection_df now has an 'L' row alongside the K-state rows."""
    from visdetect.analysis.hmm import fit_best_model
    # LOSO CV requires ≥2 sessions; generate 3 short ones.
    sessions = (
        _generate_lapse_data(T=200, lapse_rate=0.1, seed=10)
        + _generate_lapse_data(T=200, lapse_rate=0.1, seed=11)
        + _generate_lapse_data(T=200, lapse_rate=0.1, seed=12)
    )
    cfg = GLMHMMConfig(max_iter=80, n_restarts=3, verbose=False)
    best_model, selection_df, all_models = fit_best_model(
        sessions, K_range=(1, 2), config=cfg, verbose=False,
        use_cross_validation=True, n_workers=1, cv_n_restarts=2,
    )
    # Lapse row present, identifiable by K == "L"
    assert (selection_df["K"] == "L").any()
    assert "L" in all_models
    # Selected best_K must be an integer (the lapse row is excluded from selection)
    assert isinstance(best_model.n_states, int)
