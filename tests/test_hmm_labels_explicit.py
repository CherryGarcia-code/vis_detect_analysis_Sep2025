"""Tests for F25: explicit a priori auto-labeling."""

import numpy as np
import pytest

from visdetect.analysis.hmm import GLMHMM, auto_label_states_explicit


def _make_model_with_weights(weights: np.ndarray) -> GLMHMM:
    """Build a GLMHMM with the given weight matrix; no fitting."""
    K, D = weights.shape
    model = GLMHMM(n_states=K, n_features=D)
    model._init_params(seed=0)
    model._weights = weights.copy()
    return model


def test_three_states_canonical():
    """K=3 with weights placing states cleanly in Impulsive/Stim/Disengaged regions."""
    # D = 5: [bias, stim, prev_choice, prev_reward, prev_early_lick]
    # Impulsive: bias high (large positive) → P(lick | catch)≈1 AND P(lick | go)≈1
    # Stim-sensitive: bias very negative, stim weight large positive
    #                 → P(lick | catch)≈0, P(lick | log2=2)≈1
    # Disengaged: bias very negative, stim weight ~0
    #             → P(lick | catch)≈0, P(lick | go)≈0
    weights = np.array([
        [ 3.0, 0.0, 0.0, 0.0, 0.0],   # Impulsive
        [-3.0, 2.5, 0.0, 0.0, 0.0],   # Stim-sensitive
        [-3.0, 0.0, 0.0, 0.0, 0.0],   # Disengaged
    ])
    model = _make_model_with_weights(weights)
    labels = auto_label_states_explicit(model, tau_low=0.2, tau_high=0.5)
    assert labels == ["Impulsive", "Stimulus_sensitive", "Disengaged"]


def test_intermediate_states_marked():
    """A state that falls into none of the three regions is marked Intermediate."""
    weights = np.array([
        [ 3.0, 0.0, 0.0, 0.0, 0.0],   # Impulsive
        [ 0.0, 0.5, 0.0, 0.0, 0.0],   # ambiguous (mid-bias, low stim)
        [-3.0, 0.0, 0.0, 0.0, 0.0],   # Disengaged
    ])
    model = _make_model_with_weights(weights)
    labels = auto_label_states_explicit(model, tau_low=0.2, tau_high=0.5)
    assert labels[0] == "Impulsive"
    assert labels[2] == "Disengaged"
    assert labels[1].startswith("Intermediate")


def test_label_count_matches_state_count():
    """Output has one label per state, always."""
    for K in (2, 3, 4, 5):
        weights = np.random.RandomState(K).normal(0, 1, (K, 5))
        model = _make_model_with_weights(weights)
        labels = auto_label_states_explicit(model)
        assert len(labels) == K


def test_threshold_tuning_changes_labels():
    """Lowering tau_high lets more states qualify as Impulsive."""
    weights = np.array([
        [ 0.8, 0.0, 0.0, 0.0, 0.0],   # P(catch) ≈ 0.69
        [-3.0, 0.0, 0.0, 0.0, 0.0],
    ])
    model = _make_model_with_weights(weights)

    strict = auto_label_states_explicit(model, tau_low=0.2, tau_high=0.7)
    loose  = auto_label_states_explicit(model, tau_low=0.2, tau_high=0.5)
    # With strict τ_high=0.7, state 0's P(catch)=0.69 falls *just* below;
    # with loose τ_high=0.5 it's above.
    assert "Impulsive" not in strict[0]
    assert loose[0] == "Impulsive"
