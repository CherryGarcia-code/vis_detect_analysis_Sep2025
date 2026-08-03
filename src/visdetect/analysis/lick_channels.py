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
2. **Pooling inflated the train.** The previous shared helper pooled all four
   channels, inflating lick counts 1.12x-4.99x on BG_046 (far more on subjects
   with a contaminated line). ``Lick_L`` is the channel the behavioural software
   derives RT from -- its first post-change event equals ``change+RT`` to the
   millisecond, and it fires on ~100% of FA and Hit trials. ``Lick_R`` is a
   distinct, DENSER train on the same single spout (``Valve_R`` is always 0):
   ~2-4x MORE events than ``Lick_L``, median nearest-neighbour distance ~8 ms and
   only ~22% within 3 ms, so it is not a near-duplicate subset but a noisier /
   multi-bounce detector. ``Piezo_2`` is a sparse ~11 ms-shifted subset of
   ``Piezo_1``. Either way, unioning them is not "more licks" -- it is one lick
   counted several times. We select exactly ONE channel.
3. **Untrustworthy names.** ``Lick_L`` is not always the usable line, in two
   independent ways, so candidates are SCREENED rather than trusted by name:

   * *Contamination* -- BG_031_170325's ``Lick_L`` is a ~63 Hz signal (751793
     events) while ``Lick_R`` is real; BG_012 is the mirror image (``Lick_R``
     ~67-85 Hz). Caught by the sustained-rate gate (7/43 BG_031 sessions,
     46/49 BG_012 sessions).
   * *Truncation* -- BG_046_30062025's ``Lick_L`` stops recording at 1943 s,
     missing 61% of trials, while ``Lick_R`` covers the session. The rate gate
     CANNOT catch this (the truncated channel reads a plausible 0.91 Hz); it is
     caught by the trial-coverage check.

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

#: Minimum fraction of the session's trials (``Baseline_ON`` onsets) that must
#: fall inside a channel's event span for it to be preferred on NAME alone.
#:
#: Guards against a TRUNCATED recording: on BG_046_30062025 ``Lick_L`` stops at
#: 1943 s while 239/390 trials (61%) occur after it, and ``Lick_R`` covers the
#: whole session -- so preferring ``Lick_L`` by name returns a train that is
#: empty for most of the session. The rate gate cannot catch this (the truncated
#: channel's own-span rate reads a perfectly plausible 0.91 Hz); only coverage
#: can. Also fires on BG_041_30062025 (88% vs 100%).
MIN_TRIAL_COVERAGE: float = 0.90


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
    #: Fraction of the session's trials falling inside this channel's event span
    #: (1.0 when trial times are unavailable).
    trial_coverage: float = 1.0

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


def _trial_coverage(times: np.ndarray, trial_times: np.ndarray) -> float:
    """Fraction of trial onsets falling inside a channel's event span.

    Returns 1.0 when trial times are unavailable (nothing to check against).
    """
    if trial_times.size == 0 or times.size == 0:
        return 1.0
    return float(np.mean((trial_times >= times[0]) & (trial_times <= times[-1])))


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
    trial_times = _finite_sorted(ni.get("Baseline_ON"))

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
        usable.append((name, times, rate, _trial_coverage(times, trial_times)))

    if usable:
        # Preference order applies only among channels that actually COVER the
        # session. A truncated recording (real case: BG_046_30062025 Lick_L ends
        # at 1943 s, missing 61% of trials) must lose to one that does not, even
        # if the truncated channel is higher preference by name.
        covering = [u for u in usable if u[3] >= MIN_TRIAL_COVERAGE]
        if covering:
            name, times, rate, coverage = covering[0]
        else:
            name, times, rate, coverage = max(usable, key=lambda u: u[3])
        for other_name, _t, _r, other_cov in usable:
            if other_name != name and other_cov < MIN_TRIAL_COVERAGE:
                rejected[other_name] = (
                    f"covers only {other_cov:.0%} of trials (truncated "
                    f"recording); {name} covers {coverage:.0%}"
                )
        return LickChannelResult(channel=name, times=times, rate_hz=rate,
                                 rejected=rejected, trial_coverage=coverage)

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
