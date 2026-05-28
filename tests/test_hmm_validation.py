"""Tests for F22: external behavioral validation observables."""

import numpy as np
import pandas as pd
import pytest

from visdetect.analysis.hmm_validation import (
    per_state_lick_latency,
    per_state_response_time_quantiles,
    per_state_psychometric_slope,
)


def _make_assignments_df(n_per_state=20, seed=0):
    """Synthetic assignments DataFrame with 3 states, RTs that differ by state."""
    rng = np.random.default_rng(seed)
    rows = []
    # State 0 = Impulsive: short RT (early-anticipatory licks)
    # State 1 = Stim-sensitive: short tight RT
    # State 2 = Disengaged: long, variable RT
    for state, mean, scale in [(0, 0.25, 0.10), (1, 0.30, 0.06), (2, 0.70, 0.30)]:
        for _ in range(n_per_state):
            rt = max(0.0, rng.normal(mean, scale))
            rows.append({
                "hmm_state": state,
                "rt": rt,                # change-relative reaction time
                "change_time": 1.0,      # absolute (arbitrary)
                "response_time": rt + 1.0,
                "is_hit": True,
                "is_go": True,
                "is_catch": False,
                "is_fa": False,
                "change_size": rng.choice([1.25, 1.5, 2.0, 4.0]),
            })
    # A handful of catch / fa trials
    for state in (0, 1, 2):
        for _ in range(5):
            rows.append({
                "hmm_state": state,
                "rt": np.nan,
                "change_time": np.nan,
                "response_time": np.nan,
                "is_hit": False,
                "is_go": False,
                "is_catch": True,
                "is_fa": False,
                "change_size": 1.0,
            })
    return pd.DataFrame(rows)


def test_lick_latency_distinguishes_states():
    df = _make_assignments_df(seed=1)
    out = per_state_lick_latency(df, n_states=3)
    # Disengaged (state 2) must have higher median latency than Stim-sensitive (state 1).
    med = out.set_index("state")["median_latency_s"]
    assert med[2] > med[1]
    # All three states present.
    assert set(out["state"]) == {0, 1, 2}


def test_response_time_quantiles_shape():
    df = _make_assignments_df(seed=2)
    out = per_state_response_time_quantiles(df, n_states=3, quantiles=(0.25, 0.5, 0.75, 0.9))
    assert set(out.columns) >= {"state", "q25", "q50", "q75", "q90", "n"}
    assert len(out) == 3


def test_psychometric_slope_higher_in_stim_sensitive():
    """Stim-sensitive state should have a steeper P(lick) vs change_size slope."""
    # Build a dataset where state 1 (stim-sensitive) has hits scaling with change_size,
    # while state 2 has uniform low hit rate.
    rng = np.random.default_rng(3)
    rows = []
    sizes = [1.25, 1.5, 2.0, 4.0]
    for cs in sizes:
        # State 1: hit rate scales (0.3, 0.5, 0.75, 0.95)
        p1 = {1.25: 0.3, 1.5: 0.5, 2.0: 0.75, 4.0: 0.95}[cs]
        for _ in range(30):
            rows.append({"hmm_state": 1, "is_hit": rng.binomial(1, p1) == 1,
                          "is_go": True, "is_catch": False, "change_size": cs})
        # State 2: flat low ~0.2
        for _ in range(30):
            rows.append({"hmm_state": 2, "is_hit": rng.binomial(1, 0.2) == 1,
                          "is_go": True, "is_catch": False, "change_size": cs})
    df = pd.DataFrame(rows)
    out = per_state_psychometric_slope(df, n_states=3)
    slope = out.set_index("state")["slope"]
    assert slope.get(1, 0) > slope.get(2, 0)
