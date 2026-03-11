"""
Tests for visdetect.analysis.behavior — the canonical SDT implementation.

Uses synthetic Session objects so no real data files are needed.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

# Ensure src/ is importable
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from visdetect.core.session import Session, Trial, Cluster
from visdetect.analysis.behavior import (
    calculate_dprime,
    get_trial_dataframe,
    compute_session_performance,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic sessions
# ---------------------------------------------------------------------------

def _make_trial(outcome: str, change_size: float = 2.0) -> Trial:
    """Create a minimal Trial with the given outcome and change_size."""
    return Trial(trialoutcome=outcome, change_size=change_size)


def _make_session(trials, name="test_session"):
    """Create a Session with the given trial list."""
    return Session(
        trials=trials,
        clusters=[],
        subject="TEST",
        session_name=name,
    )


# ---------------------------------------------------------------------------
# Tests: calculate_dprime
# ---------------------------------------------------------------------------

class TestCalculateDprime:
    def test_perfect_performance(self):
        """d' should be high (capped by clip) when hit_rate~1 and fa_rate~0."""
        d = calculate_dprime(1.0, 0.0)
        assert d > 3.0  # clipped to 0.99/0.01 → ~4.65

    def test_chance_performance(self):
        """d' should be ~0 when hit_rate == fa_rate."""
        d = calculate_dprime(0.5, 0.5)
        assert abs(d) < 0.01

    def test_negative_dprime(self):
        """d' should be negative when fa_rate > hit_rate."""
        d = calculate_dprime(0.3, 0.7)
        assert d < 0


# ---------------------------------------------------------------------------
# Tests: get_trial_dataframe
# ---------------------------------------------------------------------------

class TestGetTrialDataframe:
    def test_go_catch_classification(self):
        """Trials with change_size>1 should be 'go', change_size==1 should be 'catch'."""
        trials = [
            _make_trial("Hit", change_size=2.0),
            _make_trial("Miss", change_size=1.0),
        ]
        session = _make_session(trials)
        df = get_trial_dataframe(session)
        assert df.iloc[0]["is_go"] == True
        assert df.iloc[0]["is_catch"] == False
        assert df.iloc[1]["is_go"] == False
        assert df.iloc[1]["is_catch"] == True

    def test_none_change_size_defaults_to_catch(self):
        """Trials with change_size=None should default to catch."""
        trials = [_make_trial("Hit", change_size=None)]
        session = _make_session(trials)
        df = get_trial_dataframe(session)
        assert df.iloc[0]["is_catch"] == True

    def test_empty_session(self):
        """Empty session should produce empty DataFrame."""
        session = _make_session([])
        df = get_trial_dataframe(session)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Tests: compute_session_performance
# ---------------------------------------------------------------------------

class TestComputeSessionPerformance:
    def test_all_hits_on_go(self):
        """
        100% hit rate on go trials, 0% FA on catch → high d'.
        10 go hits + 5 catch CRs.
        """
        trials = (
            [_make_trial("Hit", change_size=2.0) for _ in range(10)]
            + [_make_trial("Miss", change_size=1.0) for _ in range(5)]
        )
        session = _make_session(trials)
        perf = compute_session_performance(session)

        assert perf["n_sdt_hits"] == 10
        assert perf["n_sdt_misses"] == 0
        assert perf["n_sdt_fas"] == 0
        assert perf["n_sdt_crs"] == 5
        assert perf["n_go"] == 10
        assert perf["n_catch"] == 5
        assert perf["hit_rate"] == 1.0
        assert perf["fa_rate_total"] == 0.0
        assert perf["d_prime"] > 3.0  # very high (clipped)

    def test_all_misses_on_go(self):
        """0% hit rate on go trials → low/negative d'."""
        trials = (
            [_make_trial("Miss", change_size=2.0) for _ in range(10)]
            + [_make_trial("Miss", change_size=1.0) for _ in range(5)]
        )
        session = _make_session(trials)
        perf = compute_session_performance(session)

        assert perf["hit_rate"] == 0.0
        assert perf["fa_rate_total"] == 0.0
        assert perf["d_prime"] == 0.0  # both clipped to 0.01 → equal → d'=0

    def test_early_licks_excluded_from_sdt(self):
        """
        Trials with outcome='FA' (early/anticipatory lick) should NOT
        count as SDT false alarms. They should be tallied separately.
        """
        trials = [
            _make_trial("Hit", change_size=2.0),  # SDT hit
            _make_trial("FA", change_size=2.0),    # early lick on go trial — NOT SDT
            _make_trial("FA", change_size=1.0),    # early lick on catch — NOT SDT FA
            _make_trial("Miss", change_size=1.0),  # SDT CR
        ]
        session = _make_session(trials)
        perf = compute_session_performance(session)

        assert perf["n_sdt_hits"] == 1
        assert perf["n_sdt_fas"] == 0   # early licks don't count as SDT FA
        assert perf["n_sdt_crs"] == 1
        assert perf["n_go"] == 1
        assert perf["n_catch"] == 1
        assert perf["n_fa"] == 2        # both early licks counted here

    def test_reflex_and_abort_excluded(self):
        """'Ref' and 'abort' trials should be excluded from SDT entirely."""
        trials = [
            _make_trial("Hit", change_size=2.0),
            _make_trial("Ref", change_size=2.0),
            _make_trial("abort", change_size=1.0),
            _make_trial("Miss", change_size=1.0),
        ]
        session = _make_session(trials)
        perf = compute_session_performance(session)

        assert perf["n_go"] == 1
        assert perf["n_catch"] == 1
        assert perf["n_trials"] == 4  # all trials counted in total

    def test_empty_session_returns_empty(self):
        session = _make_session([])
        perf = compute_session_performance(session)
        assert perf == {}

    def test_mixed_realistic_session(self):
        """
        Realistic mix: 80% hit rate on go, 20% FA on catch.
        d' should be moderate-to-high.
        """
        trials = (
            [_make_trial("Hit", change_size=2.0) for _ in range(80)]
            + [_make_trial("Miss", change_size=2.0) for _ in range(20)]
            + [_make_trial("Hit", change_size=1.0) for _ in range(10)]   # SDT FA (lick on catch)
            + [_make_trial("Miss", change_size=1.0) for _ in range(40)]  # SDT CR
        )
        session = _make_session(trials)
        perf = compute_session_performance(session)

        assert perf["n_go"] == 100
        assert perf["n_catch"] == 50
        assert abs(perf["hit_rate"] - 0.80) < 0.01
        assert abs(perf["fa_rate_total"] - 0.20) < 0.01
        # d' for 0.8/0.2 ≈ 1.68
        assert 1.5 < perf["d_prime"] < 2.0


# ---------------------------------------------------------------------------
# Tests: staging-compatible SDT (verify stage_sessions would get same result)
# ---------------------------------------------------------------------------

class TestStagingCompatibility:
    def test_staging_keys_present(self):
        """compute_session_performance should return all keys needed by stage_sessions.py."""
        trials = [_make_trial("Hit", change_size=2.0) for _ in range(5)]
        session = _make_session(trials)
        perf = compute_session_performance(session)

        required_keys = [
            "n_sdt_hits", "n_sdt_misses", "n_sdt_fas", "n_sdt_crs",
            "n_go", "n_catch", "hit_rate", "fa_rate_total", "d_prime",
            "n_fa", "n_abort",
        ]
        for key in required_keys:
            assert key in perf, f"Missing key: {key}"
