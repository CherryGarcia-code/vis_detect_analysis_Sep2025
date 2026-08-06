"""Behavioural tests for the measured-alignment audit.

No pkls are needed: `audit.load_session` is monkeypatched to return a stub built
from `make_case()`, the same synthetic generator the solver tests use.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "QC_technical"))
import audit_trial_baselineon_alignment as audit  # noqa: E402

from test_run_alignment import make_case  # noqa: E402

PREEXISTING_COLUMNS = ("n_trials", "n_baseline_on", "diff", "ephys_s", "bon_last")


class StubSession:
    """Minimal stand-in for SessionData: only what audit_pkl touches."""

    def __init__(self, trials, ni_events):
        self.trials = trials
        self.ni_events = ni_events
        self.clusters = []


@pytest.fixture
def patched(monkeypatch):
    """Install a stub session; returns a setter for the case under test."""

    def _use(trials, ni):
        monkeypatch.setattr(audit, "load_session", lambda path: StubSession(trials, ni))

    return _use


def test_audit_exposes_measured_columns():
    assert hasattr(audit, "audit_pkl")
    assert hasattr(audit, "TOL_BENIGN")
    # the measured columns the repair depends on
    for col in ("agreement", "median_resid_s", "resid_n", "aligned"):
        assert col in audit.MEASURED_COLUMNS


def test_audit_pkl_returns_every_measured_and_preexisting_key(patched):
    trials, ni = make_case(n=60, n_pad=25)
    patched(trials, ni)
    row = audit.audit_pkl("ignored.pkl")
    for col in audit.MEASURED_COLUMNS:
        assert col in row, f"missing measured column {col}"
    for col in PREEXISTING_COLUMNS:          # the pre-existing schema must survive
        assert col in row, f"lost pre-existing column {col}"


def test_aligned_case_reports_offsets_and_no_reject_reason(patched):
    """Sign B: the true pairing sits at event_offset=25, trial_start=0."""
    trials, ni = make_case(n=60, n_pad=25)
    patched(trials, ni)
    row = audit.audit_pkl("ignored.pkl")
    assert row["aligned"] is True
    assert row["trial_start"] == 0
    assert row["event_offset"] == 25
    assert row["n_trials_matched"] == 60
    assert row["agreement"] == pytest.approx(1.0)
    assert row["median_resid_s"] < 0.05
    assert row["reject_reason"] == ""


def test_failed_case_still_carries_evidence(patched):
    """A rejected pkl must report WHY -- populated agreement, not a bare NaN."""
    trials, ni = make_case(n=60)
    ni["Change_ON"] = np.full(len(ni["Change_ON"]), np.nan)   # no usable evidence
    patched(trials, ni)
    row = audit.audit_pkl("ignored.pkl")
    assert row["aligned"] is False
    # the whole point of the fix: evidence, not a bare False
    assert np.isfinite(row["agreement"]), "failed row lost its agreement evidence"
    assert row["agreement"] < 1.0
    assert row["reject_reason"] == "agreement_below_1"


def test_reject_reason_distinguishes_check2_from_check1(patched):
    """Check 1 passes but too few testable trials -> a DIFFERENT exit than above."""
    trials, ni = make_case(n=10)             # only 5 Hit trials, < MIN_RESID_N
    patched(trials, ni)
    row = audit.audit_pkl("ignored.pkl")
    assert row["aligned"] is False
    assert row["agreement"] == pytest.approx(1.0)     # Check 1 was fine
    assert row["reject_reason"] == "resid_n_below_min"


def test_empty_trial_table_reports_no_trials(patched):
    _, ni = make_case(n=60)
    patched([], ni)
    row = audit.audit_pkl("ignored.pkl")
    assert row["aligned"] is False
    assert row["reject_reason"] == "no_trials"
    assert row["n_trials"] == 0


def test_n_trials_matched_is_recorded_and_can_be_less_than_n_trials(patched):
    """The BG_038_08082025 hazard: verified pairing, but most trials have no event."""
    trials_run1, _ = make_case(n=40, seed=1)
    trials_run2, ni = make_case(n=60)
    patched(trials_run1 + trials_run2, ni)
    row = audit.audit_pkl("ignored.pkl")
    assert row["aligned"] is True
    assert row["n_trials"] == 100
    assert row["n_trials_matched"] == 60      # 40 trials have NO ephys event
    assert row["n_trials_matched"] < row["n_trials"]
    assert row["trial_start"] == 40


# ── the verdict derivation ──────────────────────────────────────────────────
def test_measured_verdict_overrides_the_count_proxy():
    """diff==0 but aligned=False -> NOT neural_safe. This is the distinguishing case:
    the count proxy would have called this session perfectly safe."""
    df = audit.derive_verdicts(pd.DataFrame([
        {"diff": 0, "aligned": False},     # counts agree, measurement disagrees
        {"diff": 0, "aligned": True},
        {"diff": 300, "aligned": True},    # counts scream, measurement rescues it
    ]))
    assert df["match"].tolist() == [True, True, False]
    assert df["count_safe"].tolist() == [True, True, False]
    assert df["neural_safe"].tolist() == [False, True, True]


def test_derive_verdicts_survives_all_rows_failing_to_load():
    """If EVERY pkl fails, the error dict's measured defaults must keep this working."""
    rec = {"n_trials": -1, "n_baseline_on": -1, "diff": np.nan,
           **audit.MEASURED_DEFAULTS, "reject_reason": "load_error",
           "error": "OSError: boom"}
    df = audit.derive_verdicts(pd.DataFrame([rec]))   # must not KeyError
    assert df["neural_safe"].tolist() == [False]
