"""Tests for the canonical NI lick-channel resolver.

Grounded in the 2026-07-30/31 audit of BG_046 (46 pkls) + BG_031/BG_012:

* Two extraction conventions coexist. The 2025 acquisition-time MATLAB extraction
  names the lick lines ``Lick_L``/``Lick_R``; the 2026-03-06 re-extraction (which
  added the opto ``Laser`` channel) names the SAME physical lines ``Piezo_1``/
  ``Piezo_2``. Never both in one session.
* ``Piezo_2`` is NOT lick-locked (circular-shift null z=0.9, p=0.25) and must be
  excluded. ``Lick_R`` is a lower-fidelity second detector on the same single
  spout (Valve_R is always 0) whose events sit 2-3 ms from Lick_L events, so
  pooling it double-counts.
* Channel NAME alone is not trustworthy across subjects: BG_031's ``Lick_L`` is a
  contaminated ~63 Hz line (751793 events) while ``Lick_R`` is the real one;
  BG_012 is the mirror image (``Lick_R`` ~83 Hz). A resolver must therefore
  reject channels by implausible sustained rate, not by name.
"""
import numpy as np
import pytest

from visdetect.analysis import lick_channels as lc


# ── helpers ──────────────────────────────────────────────────────────
def _session(ni, duration_s=1000.0):
    """Minimal stand-in exposing the only attribute the resolver reads."""
    class _S:
        pass
    s = _S()
    s.ni_events = ni
    s.session_name = "TEST_0101202X"
    return s


def _poisson_times(rate_hz, duration_s, seed=0):
    rng = np.random.default_rng(seed)
    n = rng.poisson(rate_hz * duration_s)
    return np.sort(rng.uniform(0, duration_s, n))


# ── channel selection ────────────────────────────────────────────────
def test_prefers_piezo1_when_piezo_config():
    """Re-extraction-era session: Piezo_1 is the lick line; Piezo_2 is excluded."""
    p1 = _poisson_times(1.0, 1000.0, seed=1)
    p2 = _poisson_times(0.3, 1000.0, seed=2)
    res = lc.resolve_lick_channel(_session({"Piezo_1": p1, "Piezo_2": p2}))
    assert res.channel == "Piezo_1"
    np.testing.assert_allclose(res.times, p1)


def test_prefers_lick_l_when_lick_config():
    """2025-extraction session: Lick_L is the lick line; Lick_R is excluded."""
    ll = _poisson_times(1.0, 1000.0, seed=3)
    lr = _poisson_times(0.8, 1000.0, seed=4)
    res = lc.resolve_lick_channel(_session({"Lick_L": ll, "Lick_R": lr}))
    assert res.channel == "Lick_L"
    np.testing.assert_allclose(res.times, ll)


def test_never_pools_multiple_channels():
    """The old bug: unioning all four channels double-counted every lick."""
    p1 = _poisson_times(1.0, 1000.0, seed=5)
    p2 = _poisson_times(0.5, 1000.0, seed=6)
    res = lc.resolve_lick_channel(_session({"Piezo_1": p1, "Piezo_2": p2}))
    assert len(res.times) == len(p1)          # not len(p1) + len(p2)


# ── contamination rejection (name is not trustworthy) ────────────────
def test_rejects_contaminated_lick_l_and_falls_back_to_lick_r():
    """BG_031: Lick_L is a ~63 Hz contaminated line; Lick_R is the real one."""
    bad = _poisson_times(63.0, 1000.0, seed=7)     # physiologically impossible
    good = _poisson_times(1.1, 1000.0, seed=8)
    res = lc.resolve_lick_channel(_session({"Lick_L": bad, "Lick_R": good}))
    assert res.channel == "Lick_R"
    np.testing.assert_allclose(res.times, good)
    assert "Lick_L" in res.rejected


def test_rejects_contaminated_lick_r_and_keeps_lick_l():
    """BG_012: mirror image - Lick_R is the ~83 Hz contaminated line."""
    good = _poisson_times(1.2, 1000.0, seed=9)
    bad = _poisson_times(83.0, 1000.0, seed=10)
    res = lc.resolve_lick_channel(_session({"Lick_L": good, "Lick_R": bad}))
    assert res.channel == "Lick_L"
    assert "Lick_R" in res.rejected


def test_plausible_lick_rate_is_not_rejected():
    """A real lick channel (~1 Hz session-mean) must survive the rate gate."""
    ll = _poisson_times(1.0, 1000.0, seed=11)
    res = lc.resolve_lick_channel(_session({"Lick_L": ll}))
    assert res.channel == "Lick_L"
    assert res.rejected == {}


# ── failing loud instead of silently returning zero ──────────────────
def test_raises_when_no_lick_channel_present():
    """The old bug-A failure mode returned an EMPTY array and looked fine."""
    with pytest.raises(lc.NoLickChannelError):
        lc.resolve_lick_channel(_session({"Baseline_ON": np.array([1.0, 2.0])}))


def test_raises_when_every_candidate_is_contaminated():
    both_bad = {"Lick_L": _poisson_times(70.0, 1000.0, seed=12),
                "Lick_R": _poisson_times(80.0, 1000.0, seed=13)}
    with pytest.raises(lc.NoLickChannelError):
        lc.resolve_lick_channel(_session(both_bad))


def test_empty_channel_is_treated_as_absent():
    """An empty array is a PRESENT key in the pkl - it must not be selected."""
    ll = _poisson_times(1.0, 1000.0, seed=14)
    res = lc.resolve_lick_channel(_session({"Piezo_1": np.array([]), "Lick_L": ll}))
    assert res.channel == "Lick_L"


def test_nan_values_are_dropped():
    times = np.array([1.0, np.nan, 3.0, 2.0])
    res = lc.resolve_lick_channel(_session({"Lick_L": times}))
    np.testing.assert_allclose(res.times, [1.0, 2.0, 3.0])   # finite + sorted


# ── convenience wrapper ──────────────────────────────────────────────
def test_get_lick_times_returns_bare_array():
    ll = _poisson_times(1.0, 1000.0, seed=15)
    out = lc.get_lick_times(_session({"Lick_L": ll}))
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, ll)


# ── optional de-bounce (NOT default) ─────────────────────────────────
def test_debounce_merges_events_inside_refractory():
    """Lick_L is raw threshold crossings; callers needing bout onsets opt in."""
    times = np.array([0.0, 0.010, 0.020, 1.0, 1.005, 2.0])
    out = lc.debounce(times, refractory_s=0.05)
    np.testing.assert_allclose(out, [0.0, 1.0, 2.0])


def test_debounce_is_not_applied_by_default():
    times = np.array([0.0, 0.010, 0.020])
    res = lc.resolve_lick_channel(_session({"Lick_L": times}))
    assert len(res.times) == 3
