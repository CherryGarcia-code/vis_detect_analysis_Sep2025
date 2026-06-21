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


# ════════════════════════════════════════════════════════════════════════════
# Task 1.3: ragged Design + build_design  (contract §A.5)
# ════════════════════════════════════════════════════════════════════════════
# Ground-truth tests (per brief): a 2-trial evidence frame (one ~3-bin short
# trial, one ~200-bin long trial) plus state labels builds a ragged Design whose
# A/phi lengths match each trial's n_bins, whose event_bin == n_bins-1, whose
# mood_code indexes MAIN_MOODS, and which is sliceable via .subset(...). A trial
# whose mood is not in MAIN_MOODS is dropped.

from visdetect.analysis.decision_latents import MAIN_MOODS  # noqa: E402


def _evidence_frame(specs):
    """Build a trial_evidence_df from (trial_idx, n_bins, lick, censored) specs.

    Each trial's evidence is a deterministic ramp of length n_bins so the leaky
    accumulator output is non-trivial and its length is checkable.
    """
    rows = []
    for trial_idx, n_bins, lick, censored in specs:
        ev = np.linspace(-0.5, 0.5, n_bins)
        rows.append({
            "trial_idx": trial_idx,
            "outcome": "hit" if lick else "miss",
            "change_size": 2.0,
            "change_time": 0.5,
            "decision_time": n_bins * 0.05,
            "lick": int(lick),
            "censored": bool(censored),
            "evidence": ev,
            "n_bins": n_bins,
        })
    return pd.DataFrame(rows)


def _state_labels(mapping):
    """DataFrame indexed by trial_idx with a state_label column (load_state_labels form)."""
    df = pd.DataFrame(
        {"trial_idx": list(mapping.keys()), "state_label": list(mapping.values())}
    )
    return df.set_index("trial_idx")


def test_build_design_two_trials_short_and_long():
    """A 3-bin Impulsive trial + a 200-bin StimSens trial -> Design with len==2,
    ragged A/phi of matching lengths, event_bin == [2, 199], mood_code indexing
    MAIN_MOODS."""
    ev_df = _evidence_frame([
        (10, 3, 1, False),     # short, ~3 bins, Impulsive, lick
        (20, 200, 0, True),    # long, ~200 bins, StimSens, censored miss
    ])
    labels = _state_labels({10: "Impulsive", 20: "StimSens"})
    design = dlg.build_design(ev_df, labels, mu=0.5, sigma=0.8, dt=0.05)

    assert len(design) == 2
    # ragged A/phi lengths match each trial's n_bins
    assert len(design.A[0]) == 3 and len(design.phi[0]) == 3
    assert len(design.A[1]) == 200 and len(design.phi[1]) == 200
    # event_bin == n_bins - 1
    assert list(design.event_bin) == [2, 199]
    # mood_code indexes MAIN_MOODS
    assert MAIN_MOODS[design.mood_code[0]] == "Impulsive"
    assert MAIN_MOODS[design.mood_code[1]] == "StimSens"
    # lick / censored / trial_idx carried through
    assert list(design.lick) == [1, 0]
    assert list(design.censored.astype(bool)) == [False, True]
    assert list(design.trial_idx) == [10, 20]
    assert design.dt == 0.05


def test_build_design_phi_matches_expectation_bump():
    """phi[i] == expectation_bump(arange(n_bins)*dt, mu, sigma) for each trial."""
    ev_df = _evidence_frame([(10, 3, 1, False), (20, 200, 0, True)])
    labels = _state_labels({10: "Impulsive", 20: "StimSens"})
    mu, sigma, dt = 0.7, 0.8, 0.05
    design = dlg.build_design(ev_df, labels, mu=mu, sigma=sigma, dt=dt)
    for i, n_bins in enumerate((3, 200)):
        expected_phi = dlg.expectation_bump(np.arange(n_bins) * dt, mu, sigma)
        assert np.allclose(design.phi[i], expected_phi)


def test_build_design_A_matches_leaky_accumulate():
    """A[i] == leaky_accumulate(evidence, dt, leak_tau, rectification)."""
    ev_df = _evidence_frame([(10, 3, 1, False), (20, 200, 0, True)])
    labels = _state_labels({10: "Impulsive", 20: "StimSens"})
    design = dlg.build_design(ev_df, labels, mu=0.5, sigma=0.8, dt=0.05,
                              leak_tau=0.27, rectification="signed")
    for i, row in ev_df.reset_index(drop=True).iterrows():
        expected_A = dlg.leaky_accumulate(row["evidence"], dt=0.05,
                                          leak_tau=0.27, rectification="signed")
        assert np.allclose(design.A[i], expected_A)


def test_build_design_subset_keeps_only_long_trial():
    """design.subset([1]) -> a 1-trial Design containing only the long trial."""
    ev_df = _evidence_frame([(10, 3, 1, False), (20, 200, 0, True)])
    labels = _state_labels({10: "Impulsive", 20: "StimSens"})
    design = dlg.build_design(ev_df, labels, mu=0.5, sigma=0.8, dt=0.05)
    sub = design.subset([1])
    assert len(sub) == 1
    assert len(sub.A[0]) == 200 and len(sub.phi[0]) == 200
    assert list(sub.event_bin) == [199]
    assert list(sub.trial_idx) == [20]
    assert MAIN_MOODS[sub.mood_code[0]] == "StimSens"


def test_build_design_drops_non_main_mood_trial():
    """A trial whose mood is not in MAIN_MOODS (e.g. Disengaged) is dropped."""
    ev_df = _evidence_frame([
        (10, 3, 1, False),     # Impulsive -> kept
        (20, 200, 0, True),    # StimSens  -> kept
        (30, 5, 1, False),     # Disengaged -> dropped
    ])
    labels = _state_labels({10: "Impulsive", 20: "StimSens", 30: "Disengaged"})
    design = dlg.build_design(ev_df, labels, mu=0.5, sigma=0.8, dt=0.05)
    assert len(design) == 2
    assert list(design.trial_idx) == [10, 20]


def test_build_design_drops_untagged_trial():
    """A trial with no state label (missing from state_labels) is dropped."""
    ev_df = _evidence_frame([
        (10, 3, 1, False),     # Impulsive -> kept
        (20, 200, 0, True),    # not in labels -> dropped
    ])
    labels = _state_labels({10: "Impulsive"})
    design = dlg.build_design(ev_df, labels, mu=0.5, sigma=0.8, dt=0.05)
    assert len(design) == 1
    assert list(design.trial_idx) == [10]
    assert MAIN_MOODS[design.mood_code[0]] == "Impulsive"


def test_build_design_array_dtypes():
    """event_bin/mood_code/lick/trial_idx are int arrays; censored is bool."""
    ev_df = _evidence_frame([(10, 3, 1, False), (20, 200, 0, True)])
    labels = _state_labels({10: "Impulsive", 20: "StimSens"})
    design = dlg.build_design(ev_df, labels, mu=0.5, sigma=0.8, dt=0.05)
    assert np.issubdtype(design.event_bin.dtype, np.integer)
    assert np.issubdtype(design.mood_code.dtype, np.integer)
    assert np.issubdtype(design.lick.dtype, np.integer)
    assert np.issubdtype(design.trial_idx.dtype, np.integer)
    assert design.censored.dtype == bool


# ════════════════════════════════════════════════════════════════════════════
# Task 1.4: ParamSpec layout + closed-form censored hazard NLL
# (contract §A.4 + §A.6)
# ════════════════════════════════════════════════════════════════════════════
# Ground-truth tests (load-bearing, per brief):
#   (a) LAYOUT-INVARIANCE — ParamSpec maps theta<->dial/mood by *name*, never by a
#       hardcoded index. For two different state_terms orderings, `value(...)`
#       must read the slot that genuinely holds that dial/mood's parameter.
#   (b) RAGGED-SAFETY — hazard_nll over a Design mixing a 3-bin and a 200-bin trial
#       must EQUAL the sum of the two trials' NLLs computed singly (via subset),
#       and be finite.
#   (c) L2 — l2>0 with seed_theta==theta adds exactly 0; seed_theta!=theta adds
#       exactly l2*sum((theta-seed)**2).


def _design_one_trial(A, phi, event_bin, lick, censored, mood_code):
    """A 1-trial Design with explicit A/phi (so the NLL is hand-checkable)."""
    return dlg.Design(
        A=[np.asarray(A, float)],
        phi=[np.asarray(phi, float)],
        event_bin=np.asarray([event_bin], int),
        lick=np.asarray([lick], int),
        censored=np.asarray([censored], bool),
        mood_code=np.asarray([mood_code], int),
        trial_idx=np.asarray([0], int),
        dt=0.05,
    )


def _ragged_two_trial_design():
    """A Design with a 3-bin lick trial (Impulsive) and a 200-bin censored trial
    (StimSens). Returns (design, theta, param_spec)."""
    ps = dlg.ParamSpec(moods=("Impulsive", "StimSens"),
                       dials=("v", "z", "u"),
                       state_terms=("v", "z", "u"))
    # theta laid out as [v_Imp, v_Stim, z_Imp, z_Stim, u_Imp, u_Stim]
    theta = np.array([1.2, 0.8, -2.0, -2.5, 0.5, 0.3])
    rng = np.random.default_rng(0)
    A_short = rng.standard_normal(3)
    phi_short = rng.random(3)
    A_long = rng.standard_normal(200)
    phi_long = rng.random(200)
    design = dlg.Design(
        A=[A_short, A_long],
        phi=[phi_short, phi_long],
        event_bin=np.asarray([2, 199], int),       # n_bins - 1
        lick=np.asarray([1, 0], int),
        censored=np.asarray([False, True], bool),
        mood_code=np.asarray([0, 1], int),          # Impulsive, StimSens
        trial_idx=np.asarray([10, 20], int),
        dt=0.05,
    )
    return design, theta, ps


# ── (a) Layout-invariance ───────────────────────────────────────────────────
def test_paramspec_n_params_two_moods_three_dials():
    """2 moods x 3 per-mood dials -> n_params == 6."""
    ps = dlg.ParamSpec(moods=("Impulsive", "StimSens"),
                       dials=("v", "z", "u"),
                       state_terms=("v", "z", "u"))
    assert ps.n_params() == 6


def test_paramspec_value_layout_invariant_to_state_terms_order():
    """`value(theta, dial, mood)` reads the correct slot regardless of the order
    in which dials appear in `state_terms`/`dials`. We build a theta for each
    ordering carrying *named* per-mood values and assert the read-back matches."""
    moods = ("Impulsive", "StimSens")
    # canonical per-(dial,mood) values we want value(...) to return
    truth = {
        ("v", "Impulsive"): 1.1, ("v", "StimSens"): 1.9,
        ("z", "Impulsive"): -2.0, ("z", "StimSens"): -2.7,
        ("u", "Impulsive"): 0.4, ("u", "StimSens"): 0.6,
    }

    def build_theta(ps):
        theta = np.empty(ps.n_params())
        for dial in ps.dials:
            off = ps._offset(dial)
            for j, mood in enumerate(ps.moods):
                theta[off + j] = truth[(dial, mood)]
        return theta

    for order in (("v", "z", "u"), ("u", "z", "v")):
        ps = dlg.ParamSpec(moods=moods, dials=order, state_terms=order)
        theta = build_theta(ps)
        for (dial, mood), want in truth.items():
            got = ps.value(theta, dial, mood)
            assert abs(float(got) - want) < 1e-12, (order, dial, mood)


def test_paramspec_value_reordered_reads_correct_slot_directly():
    """Concretely: with state_terms=('u','z','v'), the v-block lives LAST. A theta
    whose last two entries are (v_Imp, v_Stim) must be read by value(...,'v',...)."""
    moods = ("Impulsive", "StimSens")
    ps = dlg.ParamSpec(moods=moods, dials=("u", "z", "v"), state_terms=("u", "z", "v"))
    # layout: [u_Imp, u_Stim, z_Imp, z_Stim, v_Imp, v_Stim]
    theta = np.array([0.4, 0.6, -2.0, -2.7, 1.1, 1.9])
    assert abs(ps.value(theta, "v", "Impulsive") - 1.1) < 1e-12
    assert abs(ps.value(theta, "v", "StimSens") - 1.9) < 1e-12
    assert abs(ps.value(theta, "u", "Impulsive") - 0.4) < 1e-12
    assert abs(ps.value(theta, "z", "StimSens") - (-2.7)) < 1e-12


def test_paramspec_shared_dial_when_not_in_state_terms():
    """A dial absent from state_terms is shared across moods -> 1 slot, both moods
    read the same value, and n_params shrinks accordingly."""
    moods = ("Impulsive", "StimSens")
    # only z is per-mood; v and u are shared -> 1 + 2 + 1 = 4 params
    ps = dlg.ParamSpec(moods=moods, dials=("v", "z", "u"), state_terms=("z",))
    assert ps.n_params() == 4
    # layout: [v_shared, z_Imp, z_Stim, u_shared]
    theta = np.array([1.3, -2.0, -2.7, 0.5])
    assert abs(ps.value(theta, "v", "Impulsive") - 1.3) < 1e-12
    assert abs(ps.value(theta, "v", "StimSens") - 1.3) < 1e-12   # shared
    assert abs(ps.value(theta, "u", "Impulsive") - 0.5) < 1e-12
    assert abs(ps.value(theta, "z", "Impulsive") - (-2.0)) < 1e-12
    assert abs(ps.value(theta, "z", "StimSens") - (-2.7)) < 1e-12


def test_paramspec_per_trial_matches_value():
    """per_trial(theta, mood_code) returns (v,z,u) arrays whose entries match
    value(theta, dial, mood) for the mood each trial belongs to."""
    moods = ("Impulsive", "StimSens")
    ps = dlg.ParamSpec(moods=moods, dials=("v", "z", "u"), state_terms=("v", "z", "u"))
    theta = np.array([1.2, 0.8, -2.0, -2.5, 0.5, 0.3])
    mood_code = np.array([0, 1, 1, 0])
    v, z, u = ps.per_trial(theta, mood_code)
    for i, m in enumerate(mood_code):
        mood = moods[m]
        assert abs(v[i] - ps.value(theta, "v", mood)) < 1e-12
        assert abs(z[i] - ps.value(theta, "z", mood)) < 1e-12
        assert abs(u[i] - ps.value(theta, "u", mood)) < 1e-12


# ── (b) Ragged-safety ───────────────────────────────────────────────────────
def test_hazard_nll_ragged_equals_sum_of_singletons():
    """NLL of a mixed 3-bin/200-bin Design == sum of the two single-trial NLLs
    (built via design.subset), and is finite."""
    design, theta, ps = _ragged_two_trial_design()
    total = dlg.hazard_nll(theta, design, ps)
    nll0 = dlg.hazard_nll(theta, design.subset([0]), ps)
    nll1 = dlg.hazard_nll(theta, design.subset([1]), ps)
    assert np.isfinite(total)
    assert np.isfinite(nll0) and np.isfinite(nll1)
    assert abs(total - (nll0 + nll1)) < 1e-9


def test_hazard_nll_matches_hand_computed_lick_trial():
    """For a single 3-bin lick trial, hazard_nll equals the closed-form
    -(sum log(1-h[:K]) + log(h[K])) with lp = z + v*A + u*phi."""
    ps = dlg.ParamSpec(moods=("Impulsive", "StimSens"),
                       dials=("v", "z", "u"), state_terms=("v", "z", "u"))
    theta = np.array([1.2, 0.8, -2.0, -2.5, 0.5, 0.3])
    A = np.array([0.1, 0.2, 0.4])
    phi = np.array([0.3, 0.6, 0.9])
    design = _design_one_trial(A, phi, event_bin=2, lick=1, censored=False, mood_code=0)
    # mood 0 = Impulsive -> v=1.2, z=-2.0, u=0.5
    lp = -2.0 + 1.2 * A + 0.5 * phi
    h = np.clip(dlg.hazard_from_lp(lp), 1e-12, 1 - 1e-12)
    K = 2
    expected = -(np.sum(np.log1p(-h[:K])) + np.log(h[K]))
    assert abs(dlg.hazard_nll(theta, design, ps) - expected) < 1e-9


def test_hazard_nll_matches_hand_computed_censored_trial():
    """For a single censored trial, hazard_nll equals
    -(sum log(1-h[:K]) + log(1-h[K]))."""
    ps = dlg.ParamSpec(moods=("Impulsive", "StimSens"),
                       dials=("v", "z", "u"), state_terms=("v", "z", "u"))
    theta = np.array([1.2, 0.8, -2.0, -2.5, 0.5, 0.3])
    A = np.array([0.1, 0.2, 0.4, 0.5])
    phi = np.array([0.3, 0.6, 0.9, 1.0])
    design = _design_one_trial(A, phi, event_bin=3, lick=0, censored=True, mood_code=1)
    # mood 1 = StimSens -> v=0.8, z=-2.5, u=0.3
    lp = -2.5 + 0.8 * A + 0.3 * phi
    h = np.clip(dlg.hazard_from_lp(lp), 1e-12, 1 - 1e-12)
    K = 3
    expected = -(np.sum(np.log1p(-h[:K])) + np.log1p(-h[K]))
    assert abs(dlg.hazard_nll(theta, design, ps) - expected) < 1e-9


def test_hazard_nll_is_finite_and_nonnegative_for_reasonable_theta():
    """A well-conditioned theta yields a finite, non-negative NLL."""
    design, theta, ps = _ragged_two_trial_design()
    val = dlg.hazard_nll(theta, design, ps)
    assert np.isfinite(val)
    assert val >= 0.0


# ── (c) L2 regularisation ───────────────────────────────────────────────────
def test_l2_zero_penalty_when_seed_equals_theta():
    """l2>0 with seed_theta == theta adds exactly 0 to the NLL."""
    design, theta, ps = _ragged_two_trial_design()
    base = dlg.hazard_nll(theta, design, ps)
    with_l2 = dlg.hazard_nll(theta, design, ps, l2=10.0, seed_theta=theta.copy())
    assert abs(with_l2 - base) < 1e-9


def test_l2_adds_exact_penalty_when_seed_differs():
    """l2>0 with seed_theta != theta increases the NLL by exactly
    l2 * sum((theta - seed)**2)."""
    design, theta, ps = _ragged_two_trial_design()
    base = dlg.hazard_nll(theta, design, ps)
    seed = theta + np.array([0.1, -0.2, 0.3, 0.0, -0.1, 0.05])
    l2 = 2.5
    with_l2 = dlg.hazard_nll(theta, design, ps, l2=l2, seed_theta=seed)
    expected_penalty = l2 * np.sum((theta - seed) ** 2)
    assert with_l2 > base
    assert abs((with_l2 - base) - expected_penalty) < 1e-9


def test_l2_no_penalty_without_seed():
    """l2>0 but seed_theta None -> no penalty (penalty needs a reference point)."""
    design, theta, ps = _ragged_two_trial_design()
    base = dlg.hazard_nll(theta, design, ps)
    with_l2 = dlg.hazard_nll(theta, design, ps, l2=5.0, seed_theta=None)
    assert abs(with_l2 - base) < 1e-9


def test_hazard_nll_returns_python_float():
    """hazard_nll returns a plain Python float (scipy.minimize requires it)."""
    design, theta, ps = _ragged_two_trial_design()
    val = dlg.hazard_nll(theta, design, ps)
    assert isinstance(val, float)


# ════════════════════════════════════════════════════════════════════════════
# Task 3.1: simulate_licks + design_with_outcomes  (contract §A.8)
# ════════════════════════════════════════════════════════════════════════════
# Ground-truth, SURVIVAL-AWARE test (primary recovery-infrastructure test).
#
# The generative draw must reproduce the discrete-time survival law exactly:
# for a per-bin hazard h_k, the probability the FIRST lick lands in bin k is
#     P(lick at k) = h_k * Prod_{j<k}(1 - h_j),
# and the probability of NO lick (censored at the last bin) is
#     P(censor) = Prod_k (1 - h_k).
# We build ONE single-bin-grid trial with a known A/phi/mood and a chosen
# true_theta so the per-bin hazards are moderate (~0.11-0.22, a non-degenerate
# lick/censor mix), repeat it N=5000x, and compare the empirical first-lick-bin
# histogram (plus the censor bucket) to the THEORETICAL survival pmf with a
# chi-square goodness-of-fit test. This is NOT tautological: the expected
# frequencies come from the closed-form survival law, computed independently of
# how simulate_licks walks the bins.


def _single_trial_design(A, phi, mood_code, n_rep):
    """A Design of `n_rep` IDENTICAL single trials (same A/phi/mood).

    event_bin = len(A)-1 and lick/censored are placeholders (overwritten by the
    simulator). Repeating the same trial lets us treat the draws as i.i.d.
    samples from one trial's survival law.
    """
    n_bins = len(A)
    return dlg.Design(
        A=[np.asarray(A, float)] * n_rep,
        phi=[np.asarray(phi, float)] * n_rep,
        event_bin=np.full(n_rep, n_bins - 1, int),
        lick=np.zeros(n_rep, int),
        censored=np.zeros(n_rep, bool),
        mood_code=np.full(n_rep, mood_code, int),
        trial_idx=np.arange(n_rep, dtype=int),
        dt=0.05,
    )


def test_simulate_licks_matches_theoretical_survival_law():
    """N=5000 identical trials: empirical first-lick-bin distribution (+ censor)
    matches the theoretical h_k*Prod_{j<k}(1-h_j) survival pmf (chi-square p>0.05),
    and the empirical censor rate matches Prod_k(1-h_k)."""
    from scipy import stats

    ps = dlg.ParamSpec(moods=("Impulsive", "StimSens"),
                       dials=("v", "z", "u"), state_terms=("v", "z", "u"))
    # theta = [v_Imp, v_Stim, z_Imp, z_Stim, u_Imp, u_Stim]; mood 0 = Impulsive
    # chosen so the per-bin hazards land ~0.11-0.22 (moderate) with ~32% censor.
    theta = np.array([0.5, 0.5, -2.2, -2.2, 0.3, 0.3])
    A = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1])
    phi = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 0.9])
    n_bins = len(A)
    N = 5000

    # ── THEORETICAL survival pmf (computed independently of the simulator) ──
    lp = -2.2 + 0.5 * A + 0.3 * phi          # mood-0 dials: v=0.5, z=-2.2, u=0.3
    h = np.clip(dlg.hazard_from_lp(lp), 1e-12, 1 - 1e-12)
    surv = np.concatenate([[1.0], np.cumprod(1.0 - h)[:-1]])  # Prod_{j<k}(1-h_j)
    pmf_lick = h * surv                      # P(first lick in bin k)
    p_censor = float(np.prod(1.0 - h))       # P(no lick at all)
    expected_probs = np.concatenate([pmf_lick, [p_censor]])
    assert abs(expected_probs.sum() - 1.0) < 1e-12        # proper distribution

    # ── EMPIRICAL outcome of the generative draw ──
    design = _single_trial_design(A, phi, mood_code=0, n_rep=N)
    event_bin, lick, censored = dlg.simulate_licks(design, theta, ps, seed=12345)

    # bin counts: index 0..n_bins-1 = first lick in that bin; index n_bins = censor
    observed = np.zeros(n_bins + 1, float)
    for eb, lk, cs in zip(event_bin, lick, censored):
        if cs:
            observed[n_bins] += 1
        else:
            assert lk == 1                   # a non-censored trial must have licked
            observed[int(eb)] += 1
    assert observed.sum() == N

    # ── chi-square goodness-of-fit: empirical vs theoretical survival pmf ──
    expected_counts = expected_probs * N
    assert np.all(expected_counts > 5)       # chi-square validity (min ~464 here)
    chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected_counts)
    assert p_value > 0.05, (
        f"empirical first-lick distribution departs from the survival law: "
        f"chi2={chi2:.3f}, p={p_value:.4f}"
    )

    # ── censor rate matches Prod_k(1-h_k) within Monte-Carlo error ──
    emp_censor = observed[n_bins] / N
    se = np.sqrt(p_censor * (1.0 - p_censor) / N)
    assert abs(emp_censor - p_censor) < 4.0 * se, (
        f"censor rate {emp_censor:.4f} != theoretical {p_censor:.4f} (4 SE band)"
    )


def test_simulate_licks_asserts_theta_length():
    """simulate_licks asserts len(true_theta) == param_spec.n_params()."""
    ps = dlg.ParamSpec(moods=("Impulsive", "StimSens"),
                       dials=("v", "z", "u"), state_terms=("v", "z", "u"))
    design = _single_trial_design(np.array([0.1, 0.3, 0.5]),
                                  np.array([0.2, 0.4, 0.6]), mood_code=0, n_rep=4)
    with pytest.raises(AssertionError):
        dlg.simulate_licks(design, np.zeros(ps.n_params() - 1), ps, seed=0)


def test_simulate_licks_is_seed_reproducible():
    """Same seed -> identical (event_bin, lick, censored); a different seed differs."""
    ps = dlg.ParamSpec(moods=("Impulsive", "StimSens"),
                       dials=("v", "z", "u"), state_terms=("v", "z", "u"))
    theta = np.array([0.5, 0.5, -2.2, -2.2, 0.3, 0.3])
    A = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1])
    phi = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 0.9])
    design = _single_trial_design(A, phi, mood_code=0, n_rep=200)

    eb1, lk1, cs1 = dlg.simulate_licks(design, theta, ps, seed=7)
    eb2, lk2, cs2 = dlg.simulate_licks(design, theta, ps, seed=7)
    assert np.array_equal(eb1, eb2) and np.array_equal(lk1, lk2) and np.array_equal(cs1, cs2)

    eb3, _, _ = dlg.simulate_licks(design, theta, ps, seed=8)
    assert not np.array_equal(eb1, eb3)          # different seed -> different draws


def test_simulate_licks_outcome_invariants():
    """Every trial is either a lick (lick==1, censored False, event_bin in range)
    or a censor (lick==0, censored True, event_bin == n_bins-1)."""
    ps = dlg.ParamSpec(moods=("Impulsive", "StimSens"),
                       dials=("v", "z", "u"), state_terms=("v", "z", "u"))
    theta = np.array([0.5, 0.5, -2.2, -2.2, 0.3, 0.3])
    A = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1])
    phi = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 0.9])
    n_bins = len(A)
    design = _single_trial_design(A, phi, mood_code=0, n_rep=500)
    event_bin, lick, censored = dlg.simulate_licks(design, theta, ps, seed=3)

    assert event_bin.dtype.kind in "iu" and lick.dtype.kind in "iu"
    assert censored.dtype == bool
    # licks: censored False, event_bin within [0, n_bins-1]
    lick_mask = lick == 1
    assert np.all(~censored[lick_mask])
    assert np.all((event_bin[lick_mask] >= 0) & (event_bin[lick_mask] <= n_bins - 1))
    # censors: lick 0, event_bin pinned to the last bin
    assert np.all(censored[~lick_mask])
    assert np.all(event_bin[~lick_mask] == n_bins - 1)


def test_design_with_outcomes_swaps_outcomes_keeps_A_phi():
    """design_with_outcomes returns a Design with the simulated outcomes but the
    SAME A/phi/mood/dt (so the simulated Design can be refit)."""
    ps = dlg.ParamSpec(moods=("Impulsive", "StimSens"),
                       dials=("v", "z", "u"), state_terms=("v", "z", "u"))
    theta = np.array([0.5, 0.5, -2.2, -2.2, 0.3, 0.3])
    A = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1])
    phi = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 0.9])
    design = _single_trial_design(A, phi, mood_code=0, n_rep=50)
    event_bin, lick, censored = dlg.simulate_licks(design, theta, ps, seed=1)

    sim = dlg.design_with_outcomes(design, event_bin, lick, censored)
    # outcomes swapped in
    assert np.array_equal(sim.event_bin, event_bin)
    assert np.array_equal(sim.lick, lick)
    assert np.array_equal(sim.censored, censored)
    # A/phi/mood/dt unchanged (same objects from the source design)
    assert sim.A is design.A and sim.phi is design.phi
    assert np.array_equal(sim.mood_code, design.mood_code)
    assert sim.dt == design.dt
    # the simulated Design must be a valid input to the likelihood
    assert np.isfinite(dlg.hazard_nll(theta, sim, ps))


def test_design_with_outcomes_does_not_mutate_source():
    """design_with_outcomes returns a copy; the source design's outcomes are
    untouched."""
    ps = dlg.ParamSpec(moods=("Impulsive", "StimSens"),
                       dials=("v", "z", "u"), state_terms=("v", "z", "u"))
    theta = np.array([0.5, 0.5, -2.2, -2.2, 0.3, 0.3])
    A = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1])
    phi = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 0.9])
    design = _single_trial_design(A, phi, mood_code=0, n_rep=50)
    orig_lick = design.lick.copy()
    orig_censored = design.censored.copy()

    event_bin, lick, censored = dlg.simulate_licks(design, theta, ps, seed=2)
    _ = dlg.design_with_outcomes(design, event_bin, lick, censored)

    assert np.array_equal(design.lick, orig_lick)
    assert np.array_equal(design.censored, orig_censored)
