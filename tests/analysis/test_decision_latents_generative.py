"""Tests for the Engine-A generative module (`decision_latents_generative`).

Task 0.9: the expert-anchor contingency GATE — `select_expert_anchors` returns
one of three modes (expert / pooled / fallback) given the Task-0.8 inventory.
"""
import numpy as np
import pandas as pd
import pytest

from visdetect.analysis import decision_latents_generative as dlg


def _inv(rows):
    """Build an inventory DataFrame from (session, dprime, n_impu, n_stim) tuples."""
    return pd.DataFrame(rows, columns=["session", "dprime", "n_impu", "n_stim"])


# ── Mode: expert ────────────────────────────────────────────────────────────
def test_expert_mode_three_qualifying():
    """>=3 qualifying sessions -> mode 'expert', anchors = the qualifying ids."""
    df = _inv([
        ("01072025", 1.5, 100, 100),   # qualifies
        ("02072025", 1.2, 50, 40),     # qualifies
        ("03072025", 0.9, 30, 25),     # qualifies
        ("04072025", 0.2, 5, 5),       # fails dprime + n
    ])
    out = dlg.select_expert_anchors(df, min_d=0.7, min_mood_n=20, min_anchors=3)
    assert out["mode"] == "expert"
    assert out["anchors"] == ["01072025", "02072025", "03072025"]


def test_expert_mode_preserves_inventory_order():
    """When the inventory is ordered (chronological), anchors keep that order."""
    df = _inv([
        ("05072025", 1.0, 50, 50),
        ("01072025", 1.0, 50, 50),
        ("03072025", 1.0, 50, 50),
    ])
    out = dlg.select_expert_anchors(df)
    assert out["mode"] == "expert"
    assert out["anchors"] == ["05072025", "01072025", "03072025"]  # inventory order kept


def test_qualification_requires_both_moods_above_min():
    """A high-d' session with one mood below min_mood_n does NOT qualify."""
    df = _inv([
        ("01072025", 1.5, 100, 100),
        ("02072025", 1.5, 100, 100),
        ("03072025", 1.5, 100, 5),     # n_stim too low -> NOT qualifying
        ("04072025", 1.5, 5, 100),     # n_impu too low -> NOT qualifying
    ])
    out = dlg.select_expert_anchors(df, min_d=0.7, min_mood_n=20, min_anchors=3)
    # only 2 qualify -> cannot be expert
    assert out["mode"] != "expert"


# ── Mode: pooled ────────────────────────────────────────────────────────────
def test_pooled_mode_tops_up_to_min_anchors():
    """1 qualifying + extra sessions -> 'pooled' with exactly min_anchors anchors,
    topped up by the best remaining sessions (dprime desc, then recency)."""
    df = _inv([
        ("01072025", 1.5, 100, 100),   # qualifies
        ("02072025", 0.9, 100, 5),     # fails (n_stim) but high d'
        ("03072025", 0.8, 5, 100),     # fails (n_impu)
        ("04072025", 0.3, 5, 5),       # weak
    ])
    out = dlg.select_expert_anchors(df, min_d=0.7, min_mood_n=20, min_anchors=3)
    assert out["mode"] == "pooled"
    assert len(out["anchors"]) == 3
    # the sole qualifier must be in the pool
    assert "01072025" in out["anchors"]
    # top-ups are the two best remaining by dprime-desc -> 02072025 (0.9), 03072025 (0.8)
    assert "02072025" in out["anchors"]
    assert "03072025" in out["anchors"]
    assert "04072025" not in out["anchors"]


def test_pooled_mode_zero_qualifying_but_enough_sessions():
    """No qualifier but >= min_anchors sessions -> pooled, length min_anchors."""
    df = _inv([
        ("01072025", 0.6, 100, 5),
        ("02072025", 0.5, 5, 100),
        ("03072025", 0.4, 10, 10),
        ("04072025", 0.1, 1, 1),
    ])
    out = dlg.select_expert_anchors(df, min_d=0.7, min_mood_n=20, min_anchors=3)
    assert out["mode"] == "pooled"
    assert len(out["anchors"]) == 3
    # best three by dprime-desc
    assert out["anchors"] == ["01072025", "02072025", "03072025"]


# ── Mode: fallback ──────────────────────────────────────────────────────────
def test_fallback_mode_too_few_sessions():
    """Fewer than min_anchors sessions total and none qualify -> fallback."""
    df = _inv([
        ("01072025", 0.5, 10, 5),
        ("02072025", 0.3, 5, 5),
    ])
    out = dlg.select_expert_anchors(df, min_d=0.7, min_mood_n=20, min_anchors=3)
    assert out["mode"] == "fallback"
    assert len(out["anchors"]) < 3  # whatever qualifies (here, none)


def test_fallback_mode_empty_inventory():
    """Empty inventory -> fallback, empty anchors."""
    df = _inv([])
    out = dlg.select_expert_anchors(df)
    assert out["mode"] == "fallback"
    assert out["anchors"] == []


def test_fallback_returns_qualifying_when_present_but_below_min():
    """2 qualify, no other sessions to top up, min_anchors=3 -> fallback with the
    2 qualifiers (downstream ships proxies for the rest)."""
    df = _inv([
        ("01072025", 1.5, 100, 100),
        ("02072025", 1.2, 50, 50),
    ])
    out = dlg.select_expert_anchors(df, min_d=0.7, min_mood_n=20, min_anchors=3)
    assert out["mode"] == "fallback"
    assert out["anchors"] == ["01072025", "02072025"]


# ── Column-name robustness (real inventory uses `session_name`) ─────────────
def test_accepts_session_name_column():
    """Real Task-0.8 inventory uses `session_name`; the gate must accept it."""
    df = pd.DataFrame({
        "session_name": ["01072025", "02072025", "03072025"],
        "dprime": [1.5, 1.2, 0.9],
        "n_impu": [100, 50, 30],
        "n_stim": [100, 40, 25],
    })
    out = dlg.select_expert_anchors(df)
    assert out["mode"] == "expert"
    assert out["anchors"] == ["01072025", "02072025", "03072025"]


def test_int_session_ids_are_zfilled_to_8_digits():
    """CSV-read session ids come back as int64; the leading zero must be
    restored (`1072025` -> `01072025`) so downstream loaders find the session."""
    df = pd.DataFrame({
        "session_name": [1072025, 2072025, 3072025],   # int64, leading zero lost
        "dprime": [1.5, 1.2, 0.9],
        "n_impu": [100, 50, 30],
        "n_stim": [100, 40, 25],
    })
    out = dlg.select_expert_anchors(df)
    assert out["mode"] == "expert"
    assert out["anchors"] == ["01072025", "02072025", "03072025"]


# ── Real inventory smoke test ───────────────────────────────────────────────
def test_real_inventory_is_expert_with_30(real_inventory_csv):
    """On the real Task-0.8 inventory, the gate yields expert mode with 30 anchors."""
    df = pd.read_csv(real_inventory_csv)
    out = dlg.select_expert_anchors(df, min_d=0.7, min_mood_n=20, min_anchors=3)
    assert out["mode"] == "expert"
    assert len(out["anchors"]) == 30


@pytest.fixture
def real_inventory_csv():
    import os
    p = os.path.join("data", "cache", "decision_latents",
                     "b8p2_expert_anchor_inventory.csv")
    if not os.path.exists(p):
        pytest.skip(f"real inventory CSV not found at {p}")
    return p


# ════════════════════════════════════════════════════════════════════════════
# Task 1.1: leaky accumulator + rectification  (contract §A.3)
# ════════════════════════════════════════════════════════════════════════════
# Ground-truth tests (not structural): the accumulator must reach the analytic
# steady state on constant evidence, rectification must gate negative evidence as
# specified, and the exponential decay must always live in (0,1).
#
# Discrete-vs-continuous note: the recurrence A[k] = decay*A[k-1] + R*dt has the
# exact fixed point  A* = R*dt/(1-decay)  with  decay = exp(-dt/tau).  This is the
# genuine steady state of the *discrete* integrator and equals the continuous
# R*tau only in the limit dt << tau (here  A* = R*tau * (dt/tau)/(1-e^{-dt/tau}),
# a factor that is 1.0 as dt->0 but ~1.18 at dt/tau=1/3).  The brief's "approaches
# R*tau within 5% after 5 tau" is the continuous-limit intuition; the precise,
# implementation-faithful ground truth these tests assert is the discrete A*.


def _discrete_fixed_point(R, dt, tau):
    """Exact steady state of A[k] = exp(-dt/tau)*A[k-1] + R*dt."""
    decay = np.exp(-dt / tau)
    return R * dt / (1.0 - decay)


def test_leaky_accumulate_steady_state_and_rectification():
    """Brief's canonical test: approaches steady state on constant +evidence and
    rectification gates negative evidence (halfwave zeros it, signed goes neg).

    Asserts the exact discrete fixed point A* = R*dt/(1-decay) (the true steady
    state of the recurrence) AND that A* is close to the continuous R*tau."""
    R, dt, tau = 1.0, 0.05, 0.27
    e = np.ones(int(5 * tau / dt))                          # ~5 tau of constant evidence
    A = dlg.leaky_accumulate(e, dt=dt, leak_tau=tau, rectification="signed")
    A_star = _discrete_fixed_point(R, dt, tau)
    assert abs(A[-1] - A_star) < 0.01 * A_star             # within 1% of discrete steady state
    assert abs(A_star - R * tau) < 0.2 * R * tau           # discrete A* ~ continuous R*tau

    neg = -np.ones(20)
    assert np.all(dlg.leaky_accumulate(neg, rectification="halfwave") == 0.0)
    assert dlg.leaky_accumulate(neg, rectification="signed")[-1] < 0


def test_steady_state_matches_discrete_fixed_point_for_several_taus():
    """For constant positive evidence, A[-1] -> the discrete fixed point within 1%
    after ~5 tau, for every swept leak constant (ground-truth, not just one tau)."""
    dt, R = 0.05, 2.0
    for tau in (0.15, 0.27, 0.40):
        n = int(round(5 * tau / dt))
        e = np.full(n, R)                                   # constant positive evidence
        A = dlg.leaky_accumulate(e, dt=dt, leak_tau=tau, rectification="signed")
        A_star = _discrete_fixed_point(R, dt, tau)
        assert abs(A[-1] - A_star) < 0.01 * A_star


def test_halfwave_zeros_negative_signed_drives_negative():
    """Halfwave rectification ignores a purely-negative trace (A stays 0);
    signed/symmetric rectification lets it accumulate negative."""
    neg = -np.ones(20)
    A_half = dlg.leaky_accumulate(neg, rectification="halfwave")
    assert np.all(A_half == 0.0)
    A_signed = dlg.leaky_accumulate(neg, rectification="signed")
    assert A_signed[-1] < 0
    assert np.all(A_signed <= 0.0)                          # monotone-negative under leak


def test_decay_in_open_unit_interval():
    """decay = exp(-dt/leak_tau) is strictly in (0,1) for every swept tau.
    Probe it behaviourally: with constant unit evidence, A[1]/A[0] - and the
    increment ratios - equal `decay`, so a value in (0,1) means each new bin
    keeps a fraction of the old accumulator (no blow-up, no full reset)."""
    dt = 0.05
    for tau in (0.15, 0.27, 0.40):
        decay = np.exp(-dt / tau)
        assert 0.0 < decay < 1.0
        # A[0] = R*dt ; A[1] = decay*A[0] + R*dt  -> (A[1]-R*dt)/A[0] == decay
        e = np.ones(3)
        A = dlg.leaky_accumulate(e, dt=dt, leak_tau=tau, rectification="signed")
        r_dt = 1.0 * dt
        recovered_decay = (A[1] - r_dt) / A[0]
        assert abs(recovered_decay - decay) < 1e-9


def test_signed_maps_to_ddm_symmetric_passthrough():
    """rectification='signed' is the contract's alias for ddm's 'symmetric'
    (identity R), so the first bin equals e[0]*dt exactly."""
    dt = 0.05
    e = np.array([1.7, -0.3, 0.5])
    A = dlg.leaky_accumulate(e, dt=dt, leak_tau=0.27, rectification="signed")
    assert abs(A[0] - 1.7 * dt) < 1e-12


def test_output_shape_and_dtype():
    """Output is a float ndarray with one value per evidence bin."""
    e = np.linspace(-1.0, 1.0, 17)
    A = dlg.leaky_accumulate(e)
    assert isinstance(A, np.ndarray)
    assert A.shape == (17,)
    assert np.issubdtype(A.dtype, np.floating)


# ════════════════════════════════════════════════════════════════════════════
# Task 1.2: cloglog hazard link + temporal-expectation (urgency) bump
# (contract §A.1 + §A.3)
# ════════════════════════════════════════════════════════════════════════════
# Ground-truth tests (not structural): the link's forward/inverse must round-trip,
# the inverse must stay a valid probability (0,1) across an extreme linear-predictor
# range without overflow, and the urgency bump must be a Gaussian peaked at 1.0 at
# mu and symmetric about it.


def test_hazard_lp_roundtrip_over_hazard_grid():
    """hazard_from_lp(lp_from_hazard(h)) ~= h for h in {0.01, ..., 0.99}."""
    h = np.linspace(0.01, 0.99, 99)
    recovered = dlg.hazard_from_lp(dlg.lp_from_hazard(h))
    assert np.allclose(recovered, h, atol=1e-9)


def test_hazard_lp_roundtrip_scalar():
    """Round-trip works for scalar inputs too (not just arrays)."""
    for h in (0.01, 0.1, 0.5, 0.9, 0.99):
        assert abs(float(dlg.hazard_from_lp(dlg.lp_from_hazard(h))) - h) < 1e-9


def test_hazard_from_lp_stays_in_open_unit_interval_no_overflow():
    """For lp in [-50, 50] the hazard is a valid probability with NO overflow.

    The brief's load-bearing guarantee is "no overflow": every value is finite
    (the clip to [-30,30] inside exp prevents inf/nan) and a lower bound > 0.
    For lp above ~3.7 the inverse cloglog 1 - exp(-exp(lp)) rounds to exactly
    1.0 in float64 (a genuine floating-point limit, NOT overflow), so the upper
    bound is the closed (0,1] rather than the open interval at the extreme tail.
    Strict (0,1) is asserted over the non-saturating range below."""
    lp = np.linspace(-50.0, 50.0, 1001)
    h = dlg.hazard_from_lp(lp)
    assert np.all(np.isfinite(h))           # no overflow -> no inf/nan
    assert np.all(h > 0.0)                  # lower tail never underflows to 0
    assert np.all(h <= 1.0)                 # valid probability (saturates at 1.0)


def test_hazard_from_lp_strictly_open_in_nonsaturating_range():
    """Within the range where float64 does not saturate, the hazard is strictly
    inside the open interval (0,1)."""
    lp = np.linspace(-50.0, 3.0, 1001)      # 1 - exp(-exp(3)) ~ 1 - 2e-9 < 1.0
    h = dlg.hazard_from_lp(lp)
    assert np.all(h > 0.0)
    assert np.all(h < 1.0)


def test_hazard_from_lp_monotone_nondecreasing():
    """The inverse-cloglog link is monotone non-decreasing in the linear
    predictor (strictly increasing until it saturates at 1.0 in float64)."""
    lp = np.linspace(-10.0, 10.0, 200)
    h = dlg.hazard_from_lp(lp)
    assert np.all(np.diff(h) >= 0)          # never decreases
    # strictly increasing over the pre-saturation range
    lp2 = np.linspace(-5.0, 3.0, 200)
    assert np.all(np.diff(dlg.hazard_from_lp(lp2)) > 0)


def test_hazard_from_lp_matches_closed_form_in_safe_range():
    """In a numerically safe lp range, hazard equals the textbook 1-exp(-exp(lp))."""
    lp = np.linspace(-5.0, 2.0, 50)
    expected = 1.0 - np.exp(-np.exp(lp))
    assert np.allclose(dlg.hazard_from_lp(lp), expected, atol=1e-12)


def test_expectation_bump_peaks_one_at_mu():
    """The urgency bump is exactly 1.0 at t == mu (Gaussian peak)."""
    t_grid = np.arange(0, 40) * 0.05
    mu, sigma = 1.0, 0.8
    phi = dlg.expectation_bump(t_grid, mu, sigma)
    k = int(np.argmin(np.abs(t_grid - mu)))     # bin nearest mu (here exactly t=1.0)
    assert abs(phi[k] - 1.0) < 1e-12
    assert np.max(phi) <= 1.0 + 1e-12           # peak never exceeds 1.0


def test_expectation_bump_is_symmetric_about_mu():
    """phi(mu - d) == phi(mu + d): the bump is symmetric about its peak."""
    mu, sigma = 1.2, 0.5
    offsets = np.array([0.1, 0.3, 0.7, 1.5])
    left = dlg.expectation_bump(mu - offsets, mu, sigma)
    right = dlg.expectation_bump(mu + offsets, mu, sigma)
    assert np.allclose(left, right, atol=1e-12)


def test_expectation_bump_decays_with_distance():
    """The bump decreases monotonically as |t - mu| grows."""
    mu, sigma = 1.0, 0.8
    dist = np.linspace(0.0, 4.0, 50)
    phi = dlg.expectation_bump(mu + dist, mu, sigma)
    assert np.all(np.diff(phi) < 0)             # strictly decreasing away from mu
    assert abs(phi[0] - 1.0) < 1e-12            # peak at distance 0
