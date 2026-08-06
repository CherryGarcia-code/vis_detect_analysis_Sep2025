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


from dataclasses import dataclass
from typing import Optional


@dataclass
class Alignment:
    """A verified pairing of a contiguous trial block to a contiguous event block."""

    trial_start: int
    n_trials_matched: int
    event_offset: int
    agreement: float
    resid_s: float
    resid_n: int
    runner_up_agreement: float = float("nan")
    runner_up_resid_s: float = float("nan")


def _passes(agreement: float, resid_s: float) -> bool:
    return (
        np.isfinite(agreement)
        and agreement >= ACCEPT_AGREEMENT
        and np.isfinite(resid_s)
        and resid_s < ACCEPT_RESID_S
    )


def best_candidate(trials: Sequence[Any], ni_events: Dict[str, Any]) -> Optional[Alignment]:
    """Top-ranked candidate pairing WITHOUT applying the acceptance test.

    Same enumeration and ranking as `solve_alignment`; it just does not judge.
    Returns None only when no candidate could be scored at all (no trials, no
    events, per-trial key length mismatch, or nothing scoreable).

    The returned Alignment is therefore a *hypothesis*, not a verified pairing --
    it is what a rejected session's evidence looks like (see `rejection_reason`).
    Never treat this as usable alignment; call `solve_alignment` for that.

    Search space:
      sign B  -> trial_start = 0, event_offset varies   (events outnumber trials)
      sign A  -> event_offset = 0, trial_start varies   (trials outnumber events)
    Both reduce to matching a contiguous trial block against a contiguous event
    block; we scan whichever dimension has slack.
    """
    trials = list(trials or [])
    n_tr = len(trials)
    ni = ni_events or {}
    n_ev = len(_arr(ni.get("Baseline_ON")))
    if n_tr == 0 or n_ev == 0:
        return None
    for key in _REQUIRED_KEYS:
        if len(_arr(ni.get(key))) != n_ev:
            return None

    candidates = []
    if n_ev >= n_tr:
        # sign B: whole trial table fits; slide it along the event arrays
        for off in range(0, n_ev - n_tr + 1):
            candidates.append((0, n_tr, off))
    else:
        # sign A: whole event array is covered; slide the trial window
        for start in range(0, n_tr - n_ev + 1):
            candidates.append((start, n_ev, 0))

    scored = []
    for start, n_match, off in candidates:
        sl = slice(start, start + n_match)
        agr, _ = outcome_change_agreement(trials, ni, sl, off)
        if not np.isfinite(agr):
            continue
        res, res_n = alignment_residual(trials, ni, sl, off)
        scored.append((agr, res, res_n, start, n_match, off))

    if not scored:
        return None

    # rank on Check 1 (full coverage) first, then Check 2 (precision)
    scored.sort(key=lambda r: (-r[0], r[1] if np.isfinite(r[1]) else np.inf))
    best = scored[0]
    runner = scored[1] if len(scored) > 1 else None

    return Alignment(
        trial_start=best[3],
        n_trials_matched=best[4],
        event_offset=best[5],
        agreement=best[0],
        resid_s=best[1],
        resid_n=best[2],
        runner_up_agreement=runner[0] if runner else float("nan"),
        runner_up_resid_s=runner[1] if runner else float("nan"),
    )


def solve_alignment(trials: Sequence[Any], ni_events: Dict[str, Any]) -> Optional[Alignment]:
    """Brute-force search for the unique (trial_start, event_offset) pairing.

    `best_candidate`, then accept-or-None -- the two share one enumeration so the
    ranking cannot drift apart. Returns None unless the top candidate passes BOTH
    Check 1 (agreement == 1.0) and Check 2 (median residual < ACCEPT_RESID_S over
    at least MIN_RESID_N trials).

    NOTE: this operates on a built pkl, where the per-run JSON boundaries are no
    longer available -- so the search is exhaustive by construction. The
    converter has a different, JSON-informed path (see ingest.py).
    """
    best = best_candidate(trials, ni_events)
    if best is None or not _passes(best.agreement, best.resid_s):
        return None
    return best


_UNSET = object()


def rejection_reason(
    trials: Sequence[Any], ni_events: Dict[str, Any], best: Any = _UNSET
) -> str:
    """Why `solve_alignment` rejected this pkl. Empty string when it accepted.

    Distinguishes the exits that all otherwise collapse to a bare None:
      no_trials / no_events / key_len_mismatch  -- structurally unscoreable
      no_candidates                             -- nothing could be scored
      agreement_below_1                         -- Check 1 failed (primary)
      resid_n_below_min                         -- Check 2 not evaluable
      resid_above_tol                           -- Check 2 failed on precision

    Pass `best` (from `best_candidate`) to avoid re-running the search.
    """
    trials = list(trials or [])
    ni = ni_events or {}
    n_ev = len(_arr(ni.get("Baseline_ON")))
    if len(trials) == 0:
        return "no_trials"
    if n_ev == 0:
        return "no_events"
    for key in _REQUIRED_KEYS:
        if len(_arr(ni.get(key))) != n_ev:
            return "key_len_mismatch"

    if best is _UNSET:
        best = best_candidate(trials, ni)
    if best is None:
        return "no_candidates"

    # Check 1 is primary and is judged first, exactly as _passes does.
    if not (np.isfinite(best.agreement) and best.agreement >= ACCEPT_AGREEMENT):
        return "agreement_below_1"
    if not np.isfinite(best.resid_s):
        return "resid_n_below_min"      # alignment_residual returns nan below MIN_RESID_N
    if best.resid_s >= ACCEPT_RESID_S:
        return "resid_above_tol"
    return ""


def build_trial_event_index(n_trials: int, alignment: Optional[Alignment]) -> np.ndarray:
    """Per-trial map into the per-trial ni_events arrays. -1 = no ephys event."""
    idx = np.full(int(n_trials), -1, dtype=int)
    if alignment is None:
        return idx
    a = alignment
    idx[a.trial_start : a.trial_start + a.n_trials_matched] = np.arange(
        a.event_offset, a.event_offset + a.n_trials_matched, dtype=int
    )
    return idx
