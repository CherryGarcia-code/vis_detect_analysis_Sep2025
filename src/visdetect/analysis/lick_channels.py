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
#: purpose (verified not lick-locked, FA z=0.9 p=0.25).
#:
#: The HIGH-FIDELITY 2025-extraction channels come FIRST: ``Piezo_1`` detects
#: only ~20-45% of licks and lags 300-640 ms, so preferring it would hand back
#: the known-inferior train. No session currently carries both conventions
#: (verified 0/253 pkls), so on today's data this order is equivalent -- but if a
#: future re-extraction ever produces one, we pick the better detector rather
#: than relying on that coincidence.
LICK_CHANNEL_CANDIDATES: tuple = ("Lick_L", "Lick_R", "Piezo_1")

#: Mean event rate (over a channel's own event span) above which it is rejected
#: as contaminated. Empirically measured over all 253 pkls / 8 subjects:
#: real lick channels span ~0.01-5.46 Hz (densest real = BG_031_130325 Lick_R,
#: 63610 events / 11650 s); contaminated lines start at 22.7 Hz (BG_031 Lick_L
#: runs ~63 Hz, BG_012 Lick_R ~83 Hz). 10 Hz sits near the middle of that gap --
#: ~2x above the densest real channel and ~2x below the sparsest contaminated
#: one -- and a mouse cannot sustain a >10 Hz SESSION-MEAN lick rate.
MAX_PLAUSIBLE_LICK_RATE_HZ: float = 10.0

#: The rate gate is only meaningful with enough events; below this a channel is
#: accepted without screening (you cannot judge a sustained rate from a handful).
MIN_EVENTS_FOR_RATE_GATE: int = 200


class NoLickChannelError(RuntimeError):
    """No usable NI lick channel could be resolved for a session.

    Raised instead of silently returning an empty array -- the failure mode that
    made four scripts drop whole eras of sessions without any visible error.
    """


#: Which MATLAB extraction convention a resolved channel belongs to.
CONVENTION_BY_CHANNEL = {
    "Lick_L": "lick_2025",     # acquisition-time extraction, ~100% detection
    "Lick_R": "lick_2025",     # second detector, same spout, ~87-95%
    "Piezo_1": "piezo_2026",   # 2026-03-06 re-extraction, ~20-45%, lags 300-640 ms
}

#: Channels known to under-detect licks. Cross-session lick RATE comparisons that
#: span both conventions are confounded (the convention aliases with the BG_046
#: learning timeline, MWU on session rank p=0.014).
UNDER_DETECTING_CHANNELS = frozenset({"Piezo_1"})


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

    @property
    def convention(self) -> str:
        """``'lick_2025'`` or ``'piezo_2026'`` -- which extraction wrote it."""
        return CONVENTION_BY_CHANNEL.get(self.channel, "unknown")

    @property
    def under_detects(self) -> bool:
        """True if this channel is known to miss most licks (see module docs)."""
        return self.channel in UNDER_DETECTING_CHANNELS


def assert_single_convention(results, context: str = "") -> str:
    """Raise if a set of per-session results spans BOTH extraction conventions.

    Guards the real hazard this module cannot fix: lick trains from the 2026
    re-extraction detect ~20-45% of licks and lag 300-640 ms versus ~100% for the
    2025 extraction, and the convention is partly confounded with the BG_046
    learning timeline. Pooling or ordering them in one cross-session figure makes
    an extraction artifact look like a behavioural change.

    Parameters
    ----------
    results : iterable of LickChannelResult (or convention strings)

    Returns
    -------
    str
        The single shared convention.
    """
    conventions = {
        r if isinstance(r, str) else r.convention for r in results
    }
    conventions.discard("unknown")
    if len(conventions) > 1:
        where = f" ({context})" if context else ""
        raise ValueError(
            f"lick data spans multiple extraction conventions {sorted(conventions)}"
            f"{where}. Their detection rates differ ~6-16x and the convention is "
            "confounded with session date, so comparing them across sessions is "
            "invalid. Split the analysis by convention, or re-extract the raw NI "
            "consistently first."
        )
    return conventions.pop() if conventions else "unknown"


def _finite_sorted(values) -> np.ndarray:
    """Flatten to a sorted 1-D float array of finite values.

    Malformed shapes (object/ragged arrays -- some ni_events channels such as
    ``Valve_L`` are stored per-trial) degrade to "absent" rather than raising an
    uncaught ``ValueError``, so the module's contract holds: absent/empty ->
    skip, contaminated -> reject, nothing usable -> NoLickChannelError.
    """
    if values is None:
        return np.empty(0, float)
    try:
        arr = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
    except (ValueError, TypeError):
        try:                                  # ragged / object array of arrays
            arr = np.concatenate([np.ravel(np.asarray(v, dtype=float))
                                  for v in np.atleast_1d(values)])
        except (ValueError, TypeError):
            return np.empty(0, float)
    return np.sort(arr[np.isfinite(arr)])


def _ni_events(session) -> dict:
    ni = getattr(session, "ni_events", session)
    return ni if isinstance(ni, dict) else {}


def _mean_rate_hz(times: np.ndarray) -> float:
    """Mean event rate over the channel's OWN event span (not session duration).

    Own-span is the conservative choice for a contamination gate: a channel whose
    events are concentrated in part of the session gets a HIGHER rate, so the
    gate errs toward flagging. The most temporally-concentrated real channel
    observed (BG_046_30062025 Lick_L, spanning 26% of the session) still reads
    only 0.91 Hz, far under the threshold.
    """
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
    # 1e-9 tolerance: without it, exact-grid inputs are decided by float noise
    # (0.15 - 0.10 == 0.049999999999999996 < 0.05 while 0.10 - 0.05 == 0.05).
    kept = [times[0]]
    for t in times[1:]:
        if t - kept[-1] >= refractory_s - 1e-9:
            kept.append(t)
    return np.asarray(kept, dtype=float)
