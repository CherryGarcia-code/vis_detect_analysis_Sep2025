"""Task 4.3 — Engine-C pyddm spot-check (GLM-vs-DDM construct validity).

These tests use a TINY SYNTHETIC pyddm-friendly session (fast, deterministic):
no real-session loading is required. They lock the contract of
``engine_c_spotcheck``:

  * On a fittable session it returns one row per session with finite ``v``,``u``
    (the DDM drift / urgency the B0 model recovers), ``failed=False``.
  * A genuine pyddm failure (here: a session that yields NO usable trials, so the
    Sample is empty and pyddm cannot fit) is CAUGHT + LOGGED and the row marked
    ``failed=True`` with a non-empty ``reason`` and NaN params -- never silently
    skipped (this is the honest "long-baseline intractability" outcome).
"""
import numpy as np
import pandas as pd
import pytest

from types import SimpleNamespace

from visdetect.analysis.decision_latents_enginec import engine_c_spotcheck

# Fast, seeded differential-evolution config: keeps these real-pyddm-fit tests to
# seconds while preserving the contract they assert (finite v,u; failure handling).
# Same pattern the ddm.py slow tests use (bounded + seeded DE, no polish).
_FAST_FITPARAMS = {"seed": 0, "maxiter": 5, "popsize": 4, "polish": False}


def _fittable_session(name, seed=0):
    """A tiny pyddm-friendly synthetic session.

    Short change_times (~0.3-0.5 s) keep the decision window small => the
    Fokker-Planck fit is fast and tractable (the whole point of the spot-check is
    that this is NOT true for real expert sessions with ~6 s baselines).
    A mix of Hit (lick after a strong change) and Miss (no lick) trials gives the
    fitter both bound crossings, so v and u are identifiable.
    """
    rng = np.random.default_rng(seed)
    # base TF stream: 1.0 (e=log2=0) so the change pulse drives evidence cleanly.
    base = np.ones(64)
    trials = []
    for k in range(24):
        ct = float(rng.uniform(0.30, 0.50))            # short baseline -> tractable
        if k % 2 == 0:                                  # Hit: strong change, fast lick
            trials.append(SimpleNamespace(
                trialoutcome="Hit", change_size=4.0, change_time=ct,
                reactiontimes={"RT": float(rng.uniform(0.10, 0.30))},
                baseline_values=base, n_seen=None))
        else:                                           # Miss: change present, no lick
            trials.append(SimpleNamespace(
                trialoutcome="Miss", change_size=4.0, change_time=ct,
                reactiontimes={}, baseline_values=base, n_seen=None))
    return SimpleNamespace(session_name=name, trials=trials,
                           ni_events={"Baseline_ON": np.zeros(40)})


def _empty_session(name):
    """A session whose trials are ALL abort/ref -> build_trial_evidence drops them
    all -> empty Sample -> pyddm cannot fit. Stands in for a genuine failure."""
    trials = [SimpleNamespace(trialoutcome="abort", change_size=1.0, change_time=1.0,
                              reactiontimes={}, baseline_values=np.ones(10), n_seen=None)
              for _ in range(5)]
    return SimpleNamespace(session_name=name, trials=trials,
                           ni_events={"Baseline_ON": np.zeros(5)})


def test_spotcheck_returns_finite_row_for_fittable_session():
    sess = _fittable_session("synthA", seed=1)
    df = engine_c_spotcheck([sess], dt=0.02, fitparams=_FAST_FITPARAMS)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    row = df.iloc[0]
    for col in ("session", "v", "u", "a", "z", "ll", "failed"):
        assert col in df.columns, f"missing column {col}"
    assert str(row["session"]) == "synthA"
    assert not bool(row["failed"])
    assert np.isfinite(row["v"]), "drift v must be finite on a fittable session"
    assert np.isfinite(row["u"]), "urgency u must be finite on a fittable session"
    assert np.isfinite(row["ll"]), "log-likelihood must be finite"


def test_spotcheck_marks_failure_with_logged_reason():
    sess = _empty_session("synthFail")
    df = engine_c_spotcheck([sess], dt=0.02, fitparams=_FAST_FITPARAMS)
    assert len(df) == 1
    row = df.iloc[0]
    assert bool(row["failed"]) is True, "an unfittable session must be failed=True"
    assert "reason" in df.columns
    assert isinstance(row["reason"], str) and len(row["reason"]) > 0, \
        "the failure reason must be logged, not blank"
    assert not np.isfinite(row["v"]), "failed row must have NaN params"


def test_spotcheck_one_row_per_session_mixed():
    good = _fittable_session("ok", seed=2)
    bad = _empty_session("bad")
    df = engine_c_spotcheck([good, bad], dt=0.02, fitparams=_FAST_FITPARAMS)
    assert len(df) == 2
    by_sess = {str(s): bool(f) for s, f in zip(df["session"], df["failed"])}
    assert by_sess["ok"] is False
    assert by_sess["bad"] is True
