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


def test_debounce_boundary_is_not_decided_by_float_noise():
    """Events exactly one refractory apart must all survive."""
    times = np.arange(0.0, 0.25, 0.05)          # 0.00 .05 .10 .15 .20
    out = lc.debounce(times, refractory_s=0.05)
    np.testing.assert_allclose(out, times)


# ── the rate-gate threshold must be pinned by tests ──────────────────
def test_densest_real_channel_is_accepted():
    """BG_031_130325 Lick_R: 63610 events / 11650 s = 5.46 Hz -- the real max."""
    times = np.sort(np.random.default_rng(0).uniform(0, 11650.0, 63610))
    res = lc.resolve_lick_channel(_session({"Lick_R": times}))
    assert res.channel == "Lick_R"


def test_sparsest_contaminated_channel_is_rejected():
    """The lowest observed contaminated rate is 22.7 Hz -- must not pass."""
    times = _poisson_times(22.7, 1000.0, seed=20)
    with pytest.raises(lc.NoLickChannelError):
        lc.resolve_lick_channel(_session({"Lick_L": times}))


def test_threshold_sits_below_the_physiological_ceiling():
    """Pins the gate near 10 Hz, not 20.

    A mouse cannot sustain a >10 Hz SESSION-MEAN lick rate (densest real channel
    observed: 5.46 Hz), so a 15 Hz line is noise and must be rejected. Without
    this, the constant could drift back to 20 Hz with every other test green.
    """
    times = _poisson_times(15.0, 1000.0, seed=29)
    with pytest.raises(lc.NoLickChannelError):
        lc.resolve_lick_channel(_session({"Lick_L": times}))


# ── convention / under-detection must be machine-readable ────────────
def test_result_exposes_convention_and_under_detection():
    lick = lc.resolve_lick_channel(_session({"Lick_L": _poisson_times(1.0, 1000.0, 21)}))
    piezo = lc.resolve_lick_channel(_session({"Piezo_1": _poisson_times(0.2, 1000.0, 22)}))
    assert lick.convention == "lick_2025" and not lick.under_detects
    assert piezo.convention == "piezo_2026" and piezo.under_detects


def test_assert_single_convention_rejects_mixed_sets():
    lick = lc.resolve_lick_channel(_session({"Lick_L": _poisson_times(1.0, 1000.0, 23)}))
    piezo = lc.resolve_lick_channel(_session({"Piezo_1": _poisson_times(0.2, 1000.0, 24)}))
    assert lc.assert_single_convention([lick, lick]) == "lick_2025"
    with pytest.raises(ValueError, match="multiple extraction conventions"):
        lc.assert_single_convention([lick, piezo], context="FA PETH")


def test_prefers_high_fidelity_lick_over_under_detecting_piezo():
    """If a session ever carried BOTH, we must not hand back the worse channel."""
    ll = _poisson_times(1.0, 1000.0, seed=25)
    p1 = _poisson_times(0.2, 1000.0, seed=26)
    res = lc.resolve_lick_channel(_session({"Piezo_1": p1, "Lick_L": ll}))
    assert res.channel == "Lick_L"
    assert not res.under_detects


# ── malformed channels degrade to "absent", never an opaque crash ────
def test_ragged_object_channel_does_not_raise_valueerror():
    ni = {"Lick_L": np.array([np.array([1.0, 2.0]), np.array([3.0])], dtype=object),
          "Lick_R": _poisson_times(1.0, 1000.0, seed=27)}
    res = lc.resolve_lick_channel(_session(ni))          # must not ValueError
    assert res.channel in ("Lick_L", "Lick_R")


def test_unparseable_channel_is_treated_as_absent():
    ni = {"Lick_L": [{"not": "numeric"}], "Lick_R": _poisson_times(1.0, 1000.0, seed=28)}
    res = lc.resolve_lick_channel(_session(ni))
    assert res.channel == "Lick_R"
