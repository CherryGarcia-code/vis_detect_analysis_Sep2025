"""Canonical resolver for NI lick-contact times.

Why this module exists (audit 2026-07-30/31, BG_046 46 pkls + BG_031/BG_012):

The same physical lick sensor (SpikeGLX analog lines 4-5) is present in EVERY
session -- the raw ``nidq.meta`` channel map is byte-identical across all 50
BG_046 raw sessions and always names those lines ``Piezo_1``/``Piezo_2``. But two
MATLAB extraction conventions coexist downstream:

* the 2025 acquisition-time extraction wrote the lick lines as ``Lick_L``/``Lick_R``;
* a 2026-03-06 re-extraction (run to add the opto ``Laser`` channel) rewrote 33
  BG_046 sessions using the raw names ``Piezo_1``/``Piezo_2``.

Never both in one session. Consequences this module defends against:

1. **Silent zeros.** Reading a single hard-coded name returns an EMPTY array on
   every session of the other convention. Four scripts had this bug. We raise
   :class:`NoLickChannelError` instead, so a mismatch fails LOUD.
2. **Double counting.** The previous shared helper pooled all four channels. But
   ``Lick_R`` is a lower-fidelity second detector on the SAME single spout
   (``Valve_R`` is always 0) whose events sit 2-3 ms from ``Lick_L`` events, and
   ``Piezo_2`` is a sparse ~11 ms-shifted subset of ``Piezo_1``. Pooling counted
   most licks twice. We select exactly ONE channel.
3. **Untrustworthy names.** ``Lick_L`` is not always the clean line: BG_031's
   ``Lick_L`` is a contaminated ~63 Hz signal (751793 events) while ``Lick_R`` is
   real; BG_012 is the mirror image. So candidates are screened by a
   physiologically-implausible sustained-rate gate, not trusted by name.

``Piezo_2`` is deliberately NOT a candidate: a circular-shift null showed it is
not lick-locked at all (FA z=0.9, p=0.25), unlike ``Piezo_1`` (FA z=12.7).

.. warning::
   Selecting the right channel does NOT make lick trains comparable across the
   two extraction conventions. The 2026 re-extraction under-detects licks by
   ~10-40x (``Piezo_1`` catches only ~20-45% of hit-trial licks and lags
   300-640 ms, vs ~100% for ``Lick_L``). For lick TIMING prefer
   :func:`visdetect.analysis.align.compute_true_reaction_time`; for cross-session
   lick RATES the raw NI must first be re-extracted consistently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

#: Candidate lick channels in preference order. ``Piezo_2`` is excluded on
#: purpose (verified not lick-locked). Preference puts each convention's primary
#: detector first, with the secondary optical line as a fallback for subjects
#: whose primary is contaminated (BG_031).
LICK_CHANNEL_CANDIDATES: tuple = ("Piezo_1", "Lick_L", "Lick_R")

#: Session-mean event rate above which a channel is rejected as contaminated.
#: Real lick lines run ~0.2-5 Hz session-mean (densest observed: 36582 events /
#: 7320 s = 5.0 Hz). Known-bad lines run 63-83 Hz. 20 Hz sits safely between.
MAX_PLAUSIBLE_LICK_RATE_HZ: float = 20.0

#: The rate gate is only meaningful with enough events; below this a channel is
#: accepted without screening (you cannot judge a sustained rate from a handful).
MIN_EVENTS_FOR_RATE_GATE: int = 200


class NoLickChannelError(RuntimeError):
    """No usable NI lick channel could be resolved for a session.

    Raised instead of silently returning an empty array -- the failure mode that
    made four scripts drop whole eras of sessions without any visible error.
    """


@dataclass
class LickChannelResult:
    """Outcome of resolving a session's lick channel."""

    channel: str
    times: np.ndarray
    rate_hz: float
    rejected: Dict[str, str] = field(default_factory=dict)

    @property
    def n_events(self) -> int:
        return int(self.times.size)


def _finite_sorted(values) -> np.ndarray:
    """Flatten to a sorted 1-D float array of finite values."""
    if values is None:
        return np.empty(0, float)
    arr = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
    return np.sort(arr[np.isfinite(arr)])


def _ni_events(session) -> dict:
    ni = getattr(session, "ni_events", session)
    return ni if isinstance(ni, dict) else {}


def _mean_rate_hz(times: np.ndarray) -> float:
    """Session-mean event rate over the channel's own span."""
    if times.size < 2:
        return 0.0
    span = float(times[-1] - times[0])
    if span <= 0:
        return float("inf")
    return float(times.size) / span


def resolve_lick_channel(
    session,
    candidates: Sequence[str] = LICK_CHANNEL_CANDIDATES,
    max_rate_hz: float = MAX_PLAUSIBLE_LICK_RATE_HZ,
    min_events_for_rate_gate: int = MIN_EVENTS_FOR_RATE_GATE,
) -> LickChannelResult:
    """Pick the single NI channel carrying this session's lick contacts.

    Walks ``candidates`` in preference order, skipping absent/empty channels and
    rejecting any whose session-mean rate is physiologically implausible.

    Raises
    ------
    NoLickChannelError
        If no candidate is present, or every present candidate is contaminated.
    """
    ni = _ni_events(session)
    rejected: Dict[str, str] = {}
    usable: list = []

    # Screen EVERY candidate before choosing, so contamination is reported even
    # when a higher-preference channel is fine (a QC signal worth surfacing --
    # e.g. BG_012's Lick_R is contaminated while Lick_L is clean).
    for name in candidates:
        if name not in ni:
            continue
        times = _finite_sorted(ni[name])
        if times.size == 0:
            continue                      # present-but-empty key == absent
        rate = _mean_rate_hz(times)
        if times.size >= min_events_for_rate_gate and rate > max_rate_hz:
            rejected[name] = (
                f"implausible sustained rate {rate:.1f} Hz "
                f"(> {max_rate_hz:.1f} Hz) over {times.size} events"
            )
            continue
        usable.append((name, times, rate))

    if usable:
        name, times, rate = usable[0]     # first in preference order
        return LickChannelResult(channel=name, times=times, rate_hz=rate,
                                 rejected=rejected)

    sess_id = getattr(session, "session_name", "<unknown session>")
    if rejected:
        detail = "; ".join(f"{k}: {v}" for k, v in rejected.items())
        raise NoLickChannelError(
            f"{sess_id}: every candidate lick channel was rejected as "
            f"contaminated ({detail}). Candidates tried: {list(candidates)}."
        )
    raise NoLickChannelError(
        f"{sess_id}: no NI lick channel found. Looked for {list(candidates)}; "
        f"ni_events has {sorted(ni)}."
    )


def get_lick_times(session, **kwargs) -> np.ndarray:
    """Convenience wrapper returning just the resolved lick times (sorted, s)."""
    return resolve_lick_channel(session, **kwargs).times


def debounce(times: np.ndarray, refractory_s: float) -> np.ndarray:
    """Collapse threshold crossings into bout onsets.

    Keeps an event only if it is at least ``refractory_s`` after the last KEPT
    event. Opt-in: ``resolve_lick_channel`` returns raw crossings, because the
    two extraction conventions differ in event density by 10-40x and de-bouncing
    alone does NOT make them comparable.
    """
    times = _finite_sorted(times)
    if times.size == 0 or refractory_s <= 0:
        return times
    kept = [times[0]]
    for t in times[1:]:
        if t - kept[-1] >= refractory_s:
            kept.append(t)
    return np.asarray(kept, dtype=float)
