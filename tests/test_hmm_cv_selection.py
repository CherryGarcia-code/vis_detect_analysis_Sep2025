"""Tests for F1: CV-based K selection in bits/trial."""

import numpy as np
import pytest
from scipy.special import expit

from visdetect.analysis.hmm import GLMHMMConfig, fit_best_model


def _generate_synthetic_sessions(
    n_sessions: int = 4,
    T: int = 120,
    K_true: int = 2,
    seed: int = 0,
):
    """Synthetic data: K_true states, identifiable enough that K=2 beats K=1."""
    rng = np.random.default_rng(seed)
    true_w = np.array([[-2.0, 1.5, 0.0, 0.0, 0.0],
                       [ 1.5, 0.3, 0.0, 0.0, 0.0]])
    true_A = np.array([[0.95, 0.05],
                       [0.06, 0.94]])

    sessions = []
    for s in range(n_sessions):
        z = np.empty(T, dtype=int)
        z[0] = rng.choice(K_true)
        for t in range(1, T):
            z[t] = rng.choice(K_true, p=true_A[z[t-1]])

        X = np.column_stack([
            np.ones(T),
            rng.uniform(-1, 2, T),
            rng.binomial(1, 0.5, T).astype(float),
            rng.binomial(1, 0.3, T).astype(float),
            rng.binomial(1, 0.2, T).astype(float),
        ])
        y = np.array([rng.binomial(1, expit(true_w[z[t]] @ X[t])) for t in range(T)],
                     dtype=float)
        sessions.append({
            "y": y, "X": X, "df": None,
            "session_name": f"sess{s}",
            "feature_names": ["bias","stim","pc","pr","pel"],
        })
    return sessions


def test_fit_best_model_returns_cv_column():
    sessions = _generate_synthetic_sessions(n_sessions=4, T=120, seed=1)
    cfg = GLMHMMConfig(max_iter=80, n_restarts=3, verbose=False)
    _, selection_df, _ = fit_best_model(
        sessions, K_range=(1, 2, 3), config=cfg, verbose=False,
        use_cross_validation=True, n_workers=1,
    )
    assert "cv_ll_bits_per_trial" in selection_df.columns
    assert "cv_ll_std" in selection_df.columns


def test_fit_best_model_cv_selects_higher_ll():
    """Best K maximises cv_ll_bits_per_trial (not minimises BIC)."""
    sessions = _generate_synthetic_sessions(n_sessions=4, T=120, seed=2)
    cfg = GLMHMMConfig(max_iter=80, n_restarts=3, verbose=False)
    best_model, selection_df, _ = fit_best_model(
        sessions, K_range=(1, 2, 3), config=cfg, verbose=False,
        use_cross_validation=True, n_workers=1,
    )
    best_K = best_model.n_states
    # The best K's CV LL must be the maximum in the table.
    assert selection_df.loc[selection_df["K"] == best_K, "cv_ll_bits_per_trial"].iloc[0] \
        == selection_df["cv_ll_bits_per_trial"].max()


def test_fit_best_model_legacy_bic_path_still_works():
    """use_cross_validation=False keeps the old BIC-based selection."""
    sessions = _generate_synthetic_sessions(n_sessions=3, T=120, seed=3)
    cfg = GLMHMMConfig(max_iter=60, n_restarts=2, verbose=False)
    best_model, selection_df, _ = fit_best_model(
        sessions, K_range=(1, 2), config=cfg, verbose=False,
        use_cross_validation=False, n_workers=1,
    )
    # Legacy path: selected K minimises BIC.
    best_K = best_model.n_states
    assert selection_df.loc[selection_df["K"] == best_K, "bic"].iloc[0] \
        == selection_df["bic"].min()
    # No CV column in legacy path.
    assert "cv_ll_bits_per_trial" not in selection_df.columns
