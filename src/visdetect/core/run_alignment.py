"""Trial <-> ni_events alignment scoring and solving (QC1).

A recording's per-trial NI arrays (Baseline_ON, Change_ON, Valve_L) must be
index-aligned to the trial table. They are not, on 17 sessions, because the
converter loads whatever *trials.json files sit in Session/ without checking
they belong to that recording. See
docs/superpowers/specs/2026-08-03-QC1-trial-event-alignment-repair-design.md

Two checks score a candidate pairing:
  Check 1 (primary, 100% trial coverage): isfinite(Change_ON) must agree with
          "was a change presented", i.e. outcome in {Hit, Miss, Ref}.
  Check 2 (secondary, precision): (Change_ON - Baseline_ON) must equal the
          trial's scheduled change_time -- only on trials where the change was
          actually presented (~45%).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

# CASE-SENSITIVE. Real pkl labels are capitalised: Hit/Miss/FA/abort/Ref.
# Do NOT refactor onto EVENT_VALID_OUTCOMES -- that is lowercase and omits Ref.
CHANGE_PRESENTED_OUTCOMES = frozenset({"Hit", "Miss", "Ref"})

ACCEPT_AGREEMENT = 1.0      # Check 1: exact, no tolerance
ACCEPT_RESID_S = 0.05       # Check 2: 10x above the observed 0.0051 s aligned value
MIN_RESID_N = 20            # below this Check 2 is not evaluable -> REJECT

_REQUIRED_KEYS = ("Baseline_ON", "Change_ON")


def _arr(x: Any) -> np.ndarray:
    if isinstance(x, dict) and "rise_t" in x:
        return np.asarray(x["rise_t"], dtype=float).ravel()
    if x is None:
        return np.zeros(0, dtype=float)
    return np.asarray(x, dtype=float).ravel()


def per_trial_event_keys(ni_events: Dict[str, Any]) -> List[str]:
    """Event keys whose arrays have one entry per recorded trial.

    Defined as: same length as Baseline_ON (which is per-trial by construction).
    """
    ni_events = ni_events or {}
    n = len(_arr(ni_events.get("Baseline_ON")))
    if n == 0:
        return []
    keys = []
    for k, v in ni_events.items():
        if k == "session_name":
            continue
        try:
            if len(_arr(v)) == n:
                keys.append(k)
        except Exception:
            continue
    return sorted(keys)


def _trial_fields(trials: Sequence[Any], trial_slice: slice):
    sub = list(trials)[trial_slice]
    outcomes = np.array([str(getattr(t, "trialoutcome", "") or "") for t in sub])
    ct = np.array(
        [
            float(getattr(t, "change_time", np.nan))
            if getattr(t, "change_time", None) is not None
            else np.nan
            for t in sub
        ],
        dtype=float,
    )
    return outcomes, ct


def outcome_change_agreement(
    trials: Sequence[Any], ni_events: Dict[str, Any], trial_slice: slice, event_offset: int
) -> Tuple[float, int]:
    """Check 1. Fraction of trials where change-presence agrees with the outcome label.

    Returns (agreement, n_compared). Returns (nan, 0) if the candidate does not fit.
    """
    outcomes, _ = _trial_fields(trials, trial_slice)
    n = len(outcomes)
    con = _arr((ni_events or {}).get("Change_ON"))
    if n == 0 or event_offset < 0 or event_offset + n > len(con):
        return float("nan"), 0
    observed = np.isfinite(con[event_offset : event_offset + n])
    expected = np.isin(outcomes, list(CHANGE_PRESENTED_OUTCOMES))
    return float(np.mean(observed == expected)), int(n)


def alignment_residual(
    trials: Sequence[Any], ni_events: Dict[str, Any], trial_slice: slice, event_offset: int
) -> Tuple[float, int]:
    """Check 2. Median |(Change_ON - Baseline_ON) - change_time| in seconds.

    Scored ONLY over trials whose scheduled change was actually presented.
    Returns (nan, n) when n < MIN_RESID_N -- an empty/thin residual set is a
    REJECT, never a pass.
    """
    _, ct = _trial_fields(trials, trial_slice)
    n = len(ct)
    ni = ni_events or {}
    bon = _arr(ni.get("Baseline_ON"))
    con = _arr(ni.get("Change_ON"))
    if n == 0 or event_offset < 0 or event_offset + n > min(len(bon), len(con)):
        return float("nan"), 0
    sl = slice(event_offset, event_offset + n)
    resid = (con[sl] - bon[sl]) - ct
    finite = np.isfinite(resid)
    n_fin = int(finite.sum())
    if n_fin < MIN_RESID_N:
        return float("nan"), n_fin
    return float(np.median(np.abs(resid[finite]))), n_fin
