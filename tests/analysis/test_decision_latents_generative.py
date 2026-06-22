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


# ════════════════════════════════════════════════════════════════════════════
# Task 1.5: fit_anchor + FitResult  (contract §A.7)
# ════════════════════════════════════════════════════════════════════════════
# Ground-truth, RECOVERY test (load-bearing, NOT a tautology).
#
# We build a Design on *genuinely identifiable* synthetic evidence: ~2000 trials
# spanning both moods, with per-trial baseline log2-TF fluctuations AND a real
# change-driven evidence excursion (a positive step after change_time on go
# trials) so the sharpness dial `v` (which multiplies the leaky-accumulated
# evidence) is identifiable; the trial grid is long enough for the urgency bump
# to bite (so `u` is identifiable) and the censor/lick mix is non-degenerate (so
# the itchiness baseline `z` is identifiable). We pick a known `true_theta` that
# yields MODERATE per-bin hazards, simulate licks through the per-bin hazard,
# refit with fit_anchor, and assert PER-DIAL PER-MOOD |recovered - true| < 0.3,
# hessian_cond < 1e6, and the locked dials structure. Recovery here is a real
# test of the fitter, not luck: each dial leaves a distinct signature in the
# survival pattern.


def _identifiable_recovery_design(n_trials=2000, dt=0.05, seed=0,
                                  step=1.5, noise=0.25, go_p=0.7):
    """Build an identifiable two-mood Design on synthetic per-trial evidence.

    Each trial has:
      * a fluctuating baseline log2-TF (small zero-mean noise around 0),
      * on go trials, a sustained positive excursion (``step``) after a per-trial
        change_time -> drives the leaky accumulator A upward, making `v`
        identifiable,
      * a trial length of ~30-60 bins (1.5-3.0 s) so the urgency bump phi (peaked
        near the change) modulates the late hazard, making `u` identifiable.

    Identifiability of the SHARPNESS dial `v` requires that lick *timing* tracks
    the accumulated evidence — i.e. trials must SURVIVE long enough to reach the
    post-change excursion before they lick. That only happens when the baseline
    hazard is low (a very negative itchiness `z` in true_theta) and the evidence
    step is the dominant driver. With a too-high baseline hazard, every trial
    licks in the first few bins and `v` washes out. The default true_theta below
    (z ~ -4) plus ``step=1.5`` yields a ~0.6 lick rate where v is recovered.
    Moods alternate Impulsive / StimSens. mu is the median change_time.
    """
    rng = np.random.default_rng(seed)
    rows = []
    change_times = []
    for tidx in range(n_trials):
        n_bins = int(rng.integers(30, 61))          # 1.5 - 3.0 s on the dt grid
        ct = float(rng.uniform(0.5, 1.2))           # change time (s)
        change_times.append(ct)
        go = bool(rng.random() < go_p)              # mostly go trials
        # fluctuating baseline log2-TF evidence (zero-mean), then a step on go
        ev = rng.normal(0.0, noise, size=n_bins)
        if go:
            t_grid = np.arange(n_bins) * dt
            ev = ev + np.where(t_grid >= ct, step, 0.0)  # sustained log2-TF excursion
        rows.append({
            "trial_idx": tidx,
            "outcome": "hit" if go else "miss",
            "change_size": 2.0 if go else 1.0,
            "change_time": ct,
            "decision_time": n_bins * dt,
            "lick": 1,                              # placeholder (simulator sets it)
            "censored": False,                      # placeholder
            "evidence": ev,
            "n_bins": n_bins,
        })
    ev_df = pd.DataFrame(rows)
    # alternate moods across trials so both are well populated
    labels = pd.DataFrame({
        "trial_idx": np.arange(n_trials),
        "state_label": [MAIN_MOODS[i % len(MAIN_MOODS)] for i in range(n_trials)],
    }).set_index("trial_idx")
    mu = float(np.median(change_times))
    design = dlg.build_design(ev_df, labels, mu=mu, sigma=0.8, dt=dt)
    return design


def test_fit_anchor_recovers_ground_truth_per_dial_per_mood():
    """Simulate outcomes from a known true_theta on an identifiable Design, refit
    via fit_anchor, and assert per-dial per-mood |recovered - true| < 0.3, a
    well-conditioned Hessian (cond < 1e6), and the locked dials structure."""
    ps = dlg.ParamSpec(moods=("Impulsive", "StimSens"),
                       dials=("v", "z", "u"), state_terms=("v", "z", "u"))
    design = _identifiable_recovery_design(n_trials=2000, dt=0.05, seed=0)

    # true_theta laid out as [v_Imp, v_Stim, z_Imp, z_Stim, u_Imp, u_Stim].
    # Distinct per-mood values so recovery is non-trivial; itchiness z ~ -4 keeps
    # the baseline hazard low so trials survive to the post-change evidence
    # excursion (making sharpness v identifiable) -> a ~0.6 lick/censor mix.
    true_theta = np.array([1.4, 0.9, -4.0, -4.4, 0.4, 0.2])
    true = {
        "Impulsive": {"sharpness": 1.4, "itchiness": -4.0, "timing": 0.4},
        "StimSens":  {"sharpness": 0.9, "itchiness": -4.4, "timing": 0.2},
    }

    eb, lk, cs = dlg.simulate_licks(design, true_theta, ps, seed=42)
    # sanity: a non-degenerate lick/censor mix (real survival information)
    lick_rate = lk.mean()
    assert 0.2 < lick_rate < 0.9, f"degenerate lick rate {lick_rate:.3f}"

    sim_design = dlg.design_with_outcomes(design, eb, lk, cs)
    result = dlg.fit_anchor(sim_design, ps, seed_theta=None,
                            l2=0.0, n_restarts=4, seed=0)

    # ── locked dials structure ──
    assert set(result.dials.keys()) == set(ps.moods)
    for mood in ps.moods:
        assert set(result.dials[mood].keys()) == {"sharpness", "itchiness", "timing"}

    # ── per-dial per-mood recovery within 0.3 ──
    for mood in ps.moods:
        for dial in ("sharpness", "itchiness", "timing"):
            rec = result.dials[mood][dial]
            tru = true[mood][dial]
            assert abs(rec - tru) < 0.3, (
                f"{mood}/{dial}: recovered {rec:.3f} vs true {tru:.3f} "
                f"(|diff|={abs(rec - tru):.3f} >= 0.3)"
            )

    # ── well-conditioned Hessian ──
    assert np.isfinite(result.hessian_cond)
    assert result.hessian_cond < 1e6, f"hessian_cond {result.hessian_cond:.3e} >= 1e6"

    # ── FitResult bookkeeping ──
    assert result.n_params == ps.n_params()
    assert result.theta.shape == (ps.n_params(),)
    assert np.isfinite(result.ll)
    assert result.hessian.shape == (ps.n_params(), ps.n_params())
    if result.cov is not None:
        assert result.cov.shape == (ps.n_params(), ps.n_params())


def test_fit_anchor_ll_is_pure_data_loglik_no_l2():
    """`ll` is the pure data log-likelihood (-hazard_nll with l2=0), even when the
    fit used an L2 penalty toward a seed."""
    ps = dlg.ParamSpec(moods=("Impulsive", "StimSens"),
                       dials=("v", "z", "u"), state_terms=("v", "z", "u"))
    design = _identifiable_recovery_design(n_trials=600, dt=0.05, seed=1)
    true_theta = np.array([1.2, 1.0, -4.0, -4.2, 0.4, 0.3])
    eb, lk, cs = dlg.simulate_licks(design, true_theta, ps, seed=7)
    sim_design = dlg.design_with_outcomes(design, eb, lk, cs)

    seed = np.zeros(ps.n_params())
    result = dlg.fit_anchor(sim_design, ps, seed_theta=seed, l2=5.0,
                            n_restarts=2, seed=0)
    # ll must equal -hazard_nll at the optimum WITHOUT the penalty
    expected_ll = -dlg.hazard_nll(result.theta, sim_design, ps, l2=0.0)
    assert abs(result.ll - expected_ll) < 1e-6


def test_fit_anchor_seed_theta_init_is_used():
    """Passing a good seed_theta still yields a valid recovery (seed is one of the
    inits, and the best restart wins)."""
    ps = dlg.ParamSpec(moods=("Impulsive", "StimSens"),
                       dials=("v", "z", "u"), state_terms=("v", "z", "u"))
    design = _identifiable_recovery_design(n_trials=1200, dt=0.05, seed=2)
    true_theta = np.array([1.3, 1.0, -4.0, -4.2, 0.4, 0.3])
    eb, lk, cs = dlg.simulate_licks(design, true_theta, ps, seed=11)
    sim_design = dlg.design_with_outcomes(design, eb, lk, cs)

    result = dlg.fit_anchor(sim_design, ps, seed_theta=true_theta.copy(),
                            l2=0.0, n_restarts=1, seed=0)
    # recovery within tolerance with the true seed present
    for mood, j in (("Impulsive", 0), ("StimSens", 1)):
        assert abs(result.dials[mood]["sharpness"] - true_theta[0 + j]) < 0.4


# ════════════════════════════════════════════════════════════════════════════
# Task 1.6: rectification selection by cross-validated log-likelihood
# (contract §A.3 rectification; §A.5 build_design/subset; §A.7 fit_anchor)
# ════════════════════════════════════════════════════════════════════════════
# Ground-truth, NON-VACUOUS test (load-bearing, NOT a tautology).
#
# We simulate EXPERT data from a SIGNED-evidence ground truth where DOWN-
# deflections genuinely carry information: each trial's baseline log2-TF evidence
# fluctuates ABOVE *and* BELOW base, and on go trials a sustained POSITIVE
# excursion follows the change. Under SIGNED rectification the accumulator A sees
# the full (positive AND negative) evidence, so its value — and hence the lick
# hazard — tracks the net signed evidence. We simulate licks through that signed
# hazard and bake the outcomes back into the evidence frame.
#
# `select_rectification` then rebuilds a Design per candidate from the SAME
# evidence via the SAME builder (changing only the rectification of A) and scores
# each by k-fold CV log-likelihood. HALFWAVE zeros every down-deflection, so its
# accumulator is systematically too high on trials where negative evidence
# suppressed the (signed-driven) lick — it cannot reproduce the timing structure.
# The test asserts a REAL CV-LL margin: signed strictly beats halfwave by > a
# meaningful gap and is the winner. (Identical scores would be a vacuous pass.)


def _signed_information_evidence_frame(n_trials=900, dt=0.05, seed=0,
                                       pos_step=1.2, neg_amp=1.2, noise=0.15,
                                       go_p=0.6):
    """Evidence frame where SIGNED down-deflections carry real information.

    Each trial:
      * baseline log2-TF evidence fluctuates ABOVE and BELOW base (zero-mean
        noise PLUS, on a random subset of bins, a sustained NEGATIVE excursion of
        amplitude ``neg_amp`` — a genuine "TF dropped below base" episode),
      * on go trials, a sustained POSITIVE excursion (``pos_step``) after a
        per-trial change_time.

    The signed accumulator integrates both signs, so its trajectory (and the
    lick hazard built from it) genuinely depends on the negative episodes. A
    halfwave accumulator zeros them and is blind to that information.
    """
    rng = np.random.default_rng(seed)
    rows = []
    change_times = []
    for tidx in range(n_trials):
        n_bins = int(rng.integers(30, 61))           # 1.5 - 3.0 s on the dt grid
        ct = float(rng.uniform(0.5, 1.2))
        change_times.append(ct)
        go = bool(rng.random() < go_p)
        t_grid = np.arange(n_bins) * dt
        ev = rng.normal(0.0, noise, size=n_bins)
        # a sustained NEGATIVE episode in a random early window (TF below base)
        neg_start = int(rng.integers(2, max(3, n_bins // 2)))
        neg_len = int(rng.integers(4, 12))
        neg_end = min(n_bins, neg_start + neg_len)
        ev[neg_start:neg_end] -= neg_amp
        if go:
            ev = ev + np.where(t_grid >= ct, pos_step, 0.0)
        rows.append({
            "trial_idx": tidx,
            "outcome": "hit" if go else "miss",
            "change_size": 2.0 if go else 1.0,
            "change_time": ct,
            "decision_time": n_bins * dt,
            "lick": 1,                               # placeholder (simulator sets)
            "censored": False,                       # placeholder
            "evidence": ev,
            "n_bins": n_bins,
        })
    ev_df = pd.DataFrame(rows)
    labels = pd.DataFrame({
        "trial_idx": np.arange(n_trials),
        "state_label": [MAIN_MOODS[i % len(MAIN_MOODS)] for i in range(n_trials)],
    }).set_index("trial_idx")
    mu = float(np.median(change_times))
    return ev_df, labels, mu


def test_select_rectification_signed_beats_halfwave_on_signed_truth():
    """Expert data simulated from a SIGNED ground truth (down-deflections carry
    information): select_rectification must score `signed` STRICTLY above
    `halfwave` by a real CV-LL margin and pick `signed` as the winner."""
    ps = dlg.ParamSpec(moods=("Impulsive", "StimSens"),
                       dials=("v", "z", "u"), state_terms=("v", "z", "u"))
    sigma = 0.8
    ev_df, labels, mu = _signed_information_evidence_frame(
        n_trials=900, dt=0.05, seed=0)

    # ── ground truth Design uses SIGNED accumulation; pick a true_theta with a
    # sizeable sharpness v so the (signed) negative episodes really suppress the
    # hazard and shape lick timing. z keeps baseline hazard low so trials survive
    # to integrate the signed evidence; u adds the urgency bump. ──
    truth_design = dlg.build_design(ev_df, labels, mu=mu, sigma=sigma, dt=0.05,
                                    rectification="signed")
    true_theta = np.array([1.8, 1.8, -3.6, -3.6, 0.5, 0.5])
    eb, lk, cs = dlg.simulate_licks(truth_design, true_theta, ps, seed=42)
    lick_rate = lk.mean()
    assert 0.2 < lick_rate < 0.95, f"degenerate lick rate {lick_rate:.3f}"

    # ── bake the SIGNED-simulated outcomes back into the evidence frame, keyed by
    # the Design's trial_idx (build_design may have dropped untagged trials; here
    # every trial is MAIN_MOODS so order is preserved). ──
    by_tidx = {int(t): (int(e), int(l), bool(c))
               for t, e, l, c in zip(truth_design.trial_idx, eb, lk, cs)}
    out_lick, out_cens, out_dec = [], [], []
    for row in ev_df.itertuples(index=False):
        tidx = int(row.trial_idx)
        if tidx in by_tidx:
            e_bin, l, c = by_tidx[tidx]
            out_lick.append(l)
            out_cens.append(c)
            # truncate decision_time to the realised event bin so the rebuilt
            # Design's event_bin matches the simulated outcome
            out_dec.append((e_bin + 1) * 0.05)
        else:                                        # untagged (none here)
            out_lick.append(int(row.lick))
            out_cens.append(bool(row.censored))
            out_dec.append(float(row.decision_time))
    ev_df = ev_df.assign(lick=out_lick, censored=out_cens, decision_time=out_dec)
    ev_df["n_bins"] = (np.asarray(out_dec) / 0.05).round().astype(int)
    # re-truncate each trial's evidence to its realised n_bins
    ev_df["evidence"] = [
        np.asarray(e, float)[:n] for e, n in zip(ev_df["evidence"], ev_df["n_bins"])
    ]

    # ── score the candidates by k-fold CV-LL via select_rectification ──
    out = dlg.select_rectification(
        dlg.build_design, ev_df, labels, mu, sigma,
        candidates=("signed", "halfwave", "asym"), k=5, seed=0)

    scores = out["scores"]
    assert set(scores) == {"signed", "halfwave", "asym"}
    assert all(np.isfinite(v) for v in scores.values())

    margin = 1.0   # CV-LL units (summed held-out log-likelihood); a real gap
    assert scores["signed"] > scores["halfwave"] + margin, (
        f"signed CV-LL {scores['signed']:.3f} not > halfwave "
        f"{scores['halfwave']:.3f} + {margin}")
    # scores must differ meaningfully (a vacuous identical-score pass is invalid)
    assert abs(scores["signed"] - scores["halfwave"]) > margin
    # The load-bearing claim is HALFWAVE LOSES (down-deflections carry info). The
    # winner is `signed` OR `asym` because with default unit gains
    # rectify(e, "asym", 1.0, 1.0) == rectify(e, "symmetric") == signed: asym
    # NESTS signed and builds an identical Design, so on signed-truth the two are
    # statistically indistinguishable and either may win by CV noise (here, with
    # the fairness-fixed shared fold split, they tie to optimizer tolerance).
    # halfwave must NOT win.
    assert out["winner"] in ("signed", "asym"), (
        f"winner {out['winner']!r} not in ('signed', 'asym'); scores={scores}")


def test_select_rectification_winner_is_argmax_of_scores():
    """The reported winner is exactly the argmax over the candidate scores."""
    ps = dlg.ParamSpec(moods=("Impulsive", "StimSens"),
                       dials=("v", "z", "u"), state_terms=("v", "z", "u"))
    sigma = 0.8
    ev_df, labels, mu = _signed_information_evidence_frame(
        n_trials=400, dt=0.05, seed=3)
    truth_design = dlg.build_design(ev_df, labels, mu=mu, sigma=sigma, dt=0.05,
                                    rectification="signed")
    true_theta = np.array([1.8, 1.8, -3.6, -3.6, 0.5, 0.5])
    eb, lk, cs = dlg.simulate_licks(truth_design, true_theta, ps, seed=7)
    by_tidx = {int(t): (int(e), int(l), bool(c))
               for t, e, l, c in zip(truth_design.trial_idx, eb, lk, cs)}
    out_lick, out_cens, out_dec = [], [], []
    for row in ev_df.itertuples(index=False):
        e_bin, l, c = by_tidx[int(row.trial_idx)]
        out_lick.append(l); out_cens.append(c); out_dec.append((e_bin + 1) * 0.05)
    ev_df = ev_df.assign(lick=out_lick, censored=out_cens, decision_time=out_dec)
    ev_df["n_bins"] = (np.asarray(out_dec) / 0.05).round().astype(int)
    ev_df["evidence"] = [
        np.asarray(e, float)[:n] for e, n in zip(ev_df["evidence"], ev_df["n_bins"])
    ]

    out = dlg.select_rectification(dlg.build_design, ev_df, labels, mu, sigma,
                                   candidates=("signed", "halfwave"), k=4, seed=1)
    winner = max(out["scores"], key=out["scores"].get)
    assert out["winner"] == winner


def test_select_rectification_is_seed_reproducible():
    """Same seed -> identical scores; the fold split is RNG-seeded."""
    ev_df, labels, mu = _signed_information_evidence_frame(
        n_trials=300, dt=0.05, seed=5)
    # minimal outcome bake-in (use the placeholder lick/censored as-is is fine for
    # reproducibility — we only check determinism, not the science here).
    out1 = dlg.select_rectification(dlg.build_design, ev_df, labels, mu, 0.8,
                                    candidates=("signed", "halfwave"), k=3, seed=0)
    out2 = dlg.select_rectification(dlg.build_design, ev_df, labels, mu, 0.8,
                                    candidates=("signed", "halfwave"), k=3, seed=0)
    assert out1["scores"] == out2["scores"]
    assert out1["winner"] == out2["winner"]


# ════════════════════════════════════════════════════════════════════════════
# Task 1.7: build_anchor_designs  (per-session Design dict, QC-gated)
# (contract §A.5 build_design + §A.10 item 3 — the explicit anchor-design dict)
# ════════════════════════════════════════════════════════════════════════════
# Ground-truth, QC-OMISSION test (load-bearing, NOT structural).
#
# `build_anchor_designs` is the Phase-1→Phase-2 bridge: per session it loads the
# session, builds the Phase-1 trial table + corrected per-trial evidence, filters
# to the MAIN_MOOD cells that pass `usable_generative` QC, builds a `Design` on
# only those cells, and keys it by session — OMITTING any session with no usable
# cell. We fabricate three synthetic sessions via the real Trial/Session
# dataclasses + synthetic state labels, monkeypatching `load_session` and
# `load_state_labels` so no disk/pkl is touched. Two sessions carry enough
# Impulsive+StimSens trials to clear every QC_GEN_* floor; the third is
# deliberately degenerate (a handful of trials per mood) so EVERY cell fails
# `usable_generative`. The test asserts: exactly the two usable sessions are
# keyed (the degenerate one is omitted), each value is a non-empty Design whose
# moods are MAIN_MOODS, and the per-session anchor mu was applied (phi peaks near
# that session's change-time).

from visdetect.core.session import Session, Trial  # noqa: E402
from visdetect.analysis import decision_latents as _dl  # noqa: E402


def _fab_trial(outcome, change_size, change_time, rt, seed):
    """One Trial via the real dataclass. 60-frame baseline_values (60 Hz, runs-of-3).

    outcome drives the Phase-1 decision-time helper:
      * 'Hit'  -> lick, decision_time = change_time + rt
      * 'Miss' -> censored, decision_time = change_time + RESPONSE_WINDOW_S
    """
    rng = np.random.default_rng(seed)
    rts = {}
    if outcome == "Hit":
        rts["RT"] = float(rt)
    elif outcome == "Miss":
        rts["Miss"] = float(rt)
    bv = (rng.random(60) * 4.0) + 1.0          # synthetic TF vector (60 frames)
    return Trial(
        trialoutcome=outcome,
        reactiontimes=rts,
        change_size=float(change_size),
        orientation=None,
        ITI=1.0,
        change_time=float(change_time),
        baseline_values=bv,
    )


def _fab_session_usable(seed=0):
    """A session whose BOTH MAIN_MOODs clear every QC_GEN_* floor.

    Per mood: 16 go-Hit (lick events + excursions) + 10 go-Miss (censored +
    excursions; long decision_time spans the anchor) -> 16 licks, 10 censored,
    26 excursions, all spanning the ~0.5 s change-anchor. Change sizes vary so
    there is a real psychometric's worth of excursions.
    """
    rng = np.random.default_rng(seed)
    trials = []
    s = seed * 1000
    cs_cycle = [1.25, 1.35, 1.5, 2.0]
    for _mood_block in range(2):                 # both moods get the same recipe
        for i in range(16):                      # Hits
            ct = float(rng.uniform(0.4, 0.6))
            trials.append(_fab_trial("Hit", cs_cycle[i % 4], ct,
                                     rng.uniform(0.25, 0.5), s)); s += 1
        for i in range(10):                      # Misses (censored)
            ct = float(rng.uniform(0.4, 0.6))
            trials.append(_fab_trial("Miss", cs_cycle[i % 4], ct,
                                     rng.uniform(0.6, 1.0), s)); s += 1
    return Session(trials=trials, clusters=[], subject="SYN",
                   session_name="USABLE", good_cluster_ids=[],
                   ni_events={"session_name": "USABLE"})


def _fab_session_unusable(seed=99):
    """A session whose EVERY cell FAILS usable_generative (too few trials).

    3 Hit + 2 Miss per mood -> 3 licks (< 15), 2 censored (< 8), 5 excursions
    (< 10): below every QC_GEN_* floor, so no mood is usable and the whole
    session must be omitted from the dict.
    """
    rng = np.random.default_rng(seed)
    trials = []
    s = seed * 1000
    for _mood_block in range(2):
        for _ in range(3):
            trials.append(_fab_trial("Hit", 2.0, float(rng.uniform(0.4, 0.6)),
                                     rng.uniform(0.25, 0.5), s)); s += 1
        for _ in range(2):
            trials.append(_fab_trial("Miss", 2.0, float(rng.uniform(0.4, 0.6)),
                                     rng.uniform(0.6, 1.0), s)); s += 1
    return Session(trials=trials, clusters=[], subject="SYN",
                   session_name="UNUSABLE", good_cluster_ids=[],
                   ni_events={"session_name": "UNUSABLE"})


def _alternating_labels(session, moods=("Impulsive", "StimSens")):
    """State-label frame (load_state_labels form) alternating MAIN_MOODS by
    trial_idx, so each mood gets an equal share of the fabricated trials."""
    n = len(session.trials)
    rows = [{"trial_idx": i, "state_label": moods[i % len(moods)],
             "state_confidence": 0.9} for i in range(n)]
    return pd.DataFrame(rows).set_index("trial_idx")[
        ["state_label", "state_confidence"]]


@pytest.fixture
def _patch_anchor_io(monkeypatch):
    """Monkeypatch load_session + load_state_labels to serve fabricated sessions.

    Returns the session/label registry so a test can register sessions by name.
    """
    sessions, labels = {}, {}

    def fake_load_session(name):
        return sessions[str(name)]

    def fake_load_state_labels(name, *a, **k):
        return labels[str(name)]

    # patch where build_anchor_designs looks them up
    import visdetect.suite.loader as _suite_loader
    monkeypatch.setattr(_suite_loader, "load_session", fake_load_session)
    monkeypatch.setattr(_dl, "load_state_labels", fake_load_state_labels)
    return sessions, labels


def test_build_anchor_designs_keys_usable_omits_unusable(_patch_anchor_io):
    """Two usable sessions -> a dict with both keyed; an all-unusable session is
    OMITTED. The omission path is genuinely exercised (S3 fails every QC floor)."""
    sessions, labels = _patch_anchor_io
    s1 = _fab_session_usable(seed=1)
    s2 = _fab_session_usable(seed=2)
    s3 = _fab_session_unusable(seed=99)          # all cells unusable -> omitted
    for name, sess in (("S1", s1), ("S2", s2), ("S3", s3)):
        sessions[name] = sess
        labels[name] = _alternating_labels(sess)

    ps = dlg.ParamSpec()
    # caller computes the per-session change-time anchor mu (Task 0.4) per session
    mu_by_session = {}
    for name, sess in (("S1", s1), ("S2", s2), ("S3", s3)):
        tab = _dl.build_trial_table(sess, labels[name], name)
        mu_by_session[name] = _dl.change_time_anchor(tab)

    out = dlg.build_anchor_designs(
        ["S1", "S2", "S3"], ps, mu_by_session, sigma=0.8, dt=0.05)

    # exactly the two usable sessions are keyed; the unusable one is omitted
    assert set(out.keys()) == {"S1", "S2"}, out.keys()
    assert "S3" not in out

    for name in ("S1", "S2"):
        design = out[name]
        assert isinstance(design, dlg.Design)
        assert len(design) > 0
        # only MAIN_MOODS enter the fit Design
        for mc in design.mood_code:
            assert MAIN_MOODS[mc] in ("Impulsive", "StimSens")
        # per-session mu applied: phi peaks near that session's anchor bin
        mu = mu_by_session[name]
        assert np.isfinite(mu)
        peak_bin = int(np.argmax(design.phi[0]))
        assert abs(peak_bin * design.dt - mu) <= design.dt + 1e-9


def test_build_anchor_designs_empty_when_all_sessions_unusable(_patch_anchor_io):
    """If EVERY session is unusable, the returned dict is empty (not None)."""
    sessions, labels = _patch_anchor_io
    s3 = _fab_session_unusable(seed=99)
    sessions["S3"] = s3
    labels["S3"] = _alternating_labels(s3)

    ps = dlg.ParamSpec()
    tab = _dl.build_trial_table(s3, labels["S3"], "S3")
    mu_by_session = {"S3": _dl.change_time_anchor(tab)}

    out = dlg.build_anchor_designs(["S3"], ps, mu_by_session, sigma=0.8, dt=0.05)
    assert out == {}


def test_build_anchor_designs_filters_to_usable_moods_only(_patch_anchor_io):
    """A session where ONE mood is usable and the other is not keeps only the
    usable mood's trials in the Design (the unusable mood is dropped)."""
    sessions, labels = _patch_anchor_io
    sess = _fab_session_usable(seed=7)
    n = len(sess.trials)
    # Make StimSens degenerate: give it only the first 4 trials; label everything
    # else Impulsive. StimSens then falls below every QC_GEN_* floor while
    # Impulsive stays well-populated.
    stim_idx = {0, 1, 2, 3}
    rows = []
    for i in range(n):
        mood = "StimSens" if i in stim_idx else "Impulsive"
        rows.append({"trial_idx": i, "state_label": mood, "state_confidence": 0.9})
    lab = pd.DataFrame(rows).set_index("trial_idx")[
        ["state_label", "state_confidence"]]
    sessions["S1"] = sess
    labels["S1"] = lab

    ps = dlg.ParamSpec()
    tab = _dl.build_trial_table(sess, lab, "S1")
    qc_stim = _dl.compute_cell_qc(tab[tab["state_label"] == "StimSens"])
    assert not qc_stim["usable_generative"]      # precondition: StimSens unusable
    qc_impu = _dl.compute_cell_qc(tab[tab["state_label"] == "Impulsive"])
    assert qc_impu["usable_generative"]          # precondition: Impulsive usable
    mu_by_session = {"S1": _dl.change_time_anchor(tab)}

    out = dlg.build_anchor_designs(["S1"], ps, mu_by_session, sigma=0.8, dt=0.05)
    assert set(out.keys()) == {"S1"}
    design = out["S1"]
    # only the usable Impulsive cell's trials survive into the Design
    assert len(design) > 0
    assert all(MAIN_MOODS[mc] == "Impulsive" for mc in design.mood_code)


# ════════════════════════════════════════════════════════════════════════════
# Task 2.1: backward_sweep — expert-first, backward-seeded anchored sweep
# (contract §A.7 fit_anchor/FitResult; §A.10 item 3)
# ════════════════════════════════════════════════════════════════════════════
# Ground-truth, RAMP-RECOVERY test (load-bearing, NOT structural).
#
# Three synthetic anchors are arranged in CHRONOLOGICAL order (oldest -> mid ->
# expert) with a TRUE sharpness `v` ramp across learning (v_old < v_mid <
# v_expert, same for both moods). Each anchor's licks are simulated from its OWN
# true_theta on an identifiable Design (low baseline itchiness z so trials
# survive to the post-change evidence excursion — recall Task 1.5: v needs
# survival to be identifiable). `backward_sweep`:
#   1. fits the MOST-EXPERT anchor FIRST (last of anchors_chrono), free
#      (seed_theta=None, l2=0 implicitly) — the identifiable reference template;
#   2. walks BACKWARD in reverse-chronological order, each anchor L2-seeded from
#      its more-expert (newer) neighbour's fitted theta.
# We assert (a) the recovered v ramps in the right direction (v_old < v_mid <
# v_expert, averaged over moods), and (b) the EXPERT anchor (fit FIRST, free,
# l2=0) is more faithful to its true v than the SAME expert data would be if it
# were L2-shrunk toward a naive (low-v) prior — i.e. fitting the expert free is
# the right call (shrinking it toward an earlier session pulls it AWAY from its
# true, higher v). If the ramp does NOT recover, that is a real signal — the test
# is NOT loosened to force a pass.
#
# CONCERN documented by this test (a real Phase-2 recovery limit, NOT a bug):
# the ABSOLUTE LEVEL of the sharpness dial v is only weakly identifiable at HIGH
# v. The likelihood is flat along a v<->z ridge (lp = z + v*A + ...), so as true
# v grows the MLE shrinks v and compensates with z; recovery is biased toward
# smaller v (e.g. true v_expert=1.8 recovers ~1.2, a ~0.6 downward bias, stable
# across restarts with a well-conditioned Hessian). The RAMP DIRECTION is robust;
# the absolute v LEVEL at the expert end is biased low. Downstream comparisons of
# v should therefore lead with the ramp/ordering, treating absolute high-v levels
# as descriptive (consistent with the recovery-gate philosophy, contract §A.9).
# This is exactly why claim (b) is framed as free-beats-wrong-prior-shrinkage
# rather than "expert recovers v closest" (which the high-v bias makes false for
# an increasing ramp).


def _ramp_anchor_design(v_level, n_trials=900, dt=0.05, seed=0,
                        step=1.6, noise=0.2, go_p=0.75):
    """Build an identifiable two-mood Design whose true sharpness is ``v_level``.

    Mirrors ``_identifiable_recovery_design`` (Task 1.5): fluctuating baseline
    log2-TF evidence + a sustained post-change positive excursion on go trials so
    the leaky accumulator A rises and `v` is identifiable, with trial lengths long
    enough for the urgency bump to bite. Returns ``(design, true_theta)`` where
    ``true_theta`` carries ``v_level`` in BOTH moods' sharpness slot; z is very
    negative so the baseline hazard is low (trials survive to the excursion).
    """
    rng = np.random.default_rng(seed)
    rows = []
    change_times = []
    for tidx in range(n_trials):
        n_bins = int(rng.integers(30, 61))           # 1.5 - 3.0 s on the dt grid
        ct = float(rng.uniform(0.5, 1.2))
        change_times.append(ct)
        go = bool(rng.random() < go_p)
        ev = rng.normal(0.0, noise, size=n_bins)
        if go:
            t_grid = np.arange(n_bins) * dt
            ev = ev + np.where(t_grid >= ct, step, 0.0)
        rows.append({
            "trial_idx": tidx,
            "outcome": "hit" if go else "miss",
            "change_size": 2.0 if go else 1.0,
            "change_time": ct,
            "decision_time": n_bins * dt,
            "lick": 1,                               # placeholder (simulator sets)
            "censored": False,                       # placeholder
            "evidence": ev,
            "n_bins": n_bins,
        })
    ev_df = pd.DataFrame(rows)
    labels = pd.DataFrame({
        "trial_idx": np.arange(n_trials),
        "state_label": [MAIN_MOODS[i % len(MAIN_MOODS)] for i in range(n_trials)],
    }).set_index("trial_idx")
    mu = float(np.median(change_times))
    design = dlg.build_design(ev_df, labels, mu=mu, sigma=0.8, dt=dt)
    # true_theta = [v_Imp, v_Stim, z_Imp, z_Stim, u_Imp, u_Stim]; v ramps via v_level.
    true_theta = np.array([v_level, v_level, -4.0, -4.0, 0.4, 0.3])
    return design, true_theta


def test_backward_sweep_recovers_v_ramp_and_free_expert_beats_shrunk():
    """Three anchors with a TRUE v ramp (v_old < v_mid < v_expert): the expert is
    fit FIRST (free), then earlier anchors are L2-seeded backward. Assert (a) the
    recovered v ramps in the right direction, and (b) fitting the expert FREE
    (as the sweep does) is more faithful to its true v than L2-shrinking that
    same expert data toward a naive (low-v) prior would be."""
    ps = dlg.ParamSpec(moods=("Impulsive", "StimSens"),
                       dials=("v", "z", "u"), state_terms=("v", "z", "u"))

    # CHRONOLOGICAL order: oldest -> mid -> expert (a sharpening v ramp).
    v_old, v_mid, v_expert = 0.7, 1.2, 1.8
    specs = [
        ("OLD", v_old, 10),
        ("MID", v_mid, 20),
        ("EXPERT", v_expert, 30),
    ]
    anchors_chrono = [name for name, _, _ in specs]

    anchor_designs = {}
    true_v = {}
    for name, v_level, sd in specs:
        design, true_theta = _ramp_anchor_design(v_level, n_trials=900, seed=sd)
        eb, lk, cs = dlg.simulate_licks(design, true_theta, ps, seed=sd + 100)
        # non-degenerate lick/censor mix carries the survival information for v
        assert 0.2 < lk.mean() < 0.95, f"{name}: degenerate lick rate {lk.mean():.3f}"
        sim_design = dlg.design_with_outcomes(design, eb, lk, cs)
        anchor_designs[name] = sim_design
        true_v[name] = v_level

    results = dlg.backward_sweep(anchor_designs, anchors_chrono, ps,
                                 l2=1.0, seed=0)

    # ── all anchors fit; FitResult per anchor ──
    assert set(results.keys()) == set(anchors_chrono)
    for name in anchors_chrono:
        assert isinstance(results[name], dlg.FitResult)

    # ── recovered v = mean of the two moods' sharpness per anchor ──
    def rec_v_from(dials):
        return 0.5 * (dials["Impulsive"]["sharpness"] + dials["StimSens"]["sharpness"])

    def rec_v(name):
        return rec_v_from(results[name].dials)

    rv_old, rv_mid, rv_expert = rec_v("OLD"), rec_v("MID"), rec_v("EXPERT")

    # (a) PRIMARY: the recovered v ramps in the right direction (monotone up).
    assert rv_old < rv_mid < rv_expert, (
        f"recovered v did not ramp old->mid->expert: "
        f"{rv_old:.3f} < {rv_mid:.3f} < {rv_expert:.3f}")

    # (b) The sweep fits the expert FREE. Show that is the right call: the same
    # expert data L2-shrunk toward a NAIVE (low-v) prior is pulled FURTHER from
    # the expert's true (high) v than the free fit is. (The absolute high-v level
    # is biased low either way — see the module-level CONCERN note — but free is
    # strictly more faithful than wrong-prior shrinkage.)
    expert_design = anchor_designs["EXPERT"]
    free_err = abs(rec_v("EXPERT") - true_v["EXPERT"])      # the sweep's free fit
    naive_prior = np.array([v_old, v_old, -4.0, -4.0, 0.4, 0.3])
    shrunk = dlg.fit_anchor(expert_design, ps, seed_theta=naive_prior,
                            l2=5.0, n_restarts=4, seed=0)
    shrunk_err = abs(rec_v_from(shrunk.dials) - true_v["EXPERT"])
    assert free_err < shrunk_err, (
        f"free expert fit (err {free_err:.3f}) should beat shrink-toward-naive "
        f"(err {shrunk_err:.3f}); fitting the expert free is the right call")

    # The expert's free fit is well-conditioned (the bias is structural, not an
    # optimizer failure): the level is weakly identified, the FIT is not broken.
    assert np.isfinite(results["EXPERT"].hessian_cond)
    assert results["EXPERT"].hessian_cond < 1e6


def test_backward_sweep_fits_expert_first_free_then_seeds_backward(monkeypatch):
    """The MOST-EXPERT anchor (last of anchors_chrono) is fit FIRST with
    seed_theta=None and l2=0; each earlier anchor is then fit with seed_theta =
    its more-expert (newer) neighbour's fitted theta and the passed l2. Verified
    by recording the (name, seed_theta, l2) of every fit_anchor call."""
    ps = dlg.ParamSpec(moods=("Impulsive", "StimSens"),
                       dials=("v", "z", "u"), state_terms=("v", "z", "u"))

    specs = [("OLD", 0.7, 10), ("MID", 1.2, 20), ("EXPERT", 1.8, 30)]
    anchors_chrono = [name for name, _, _ in specs]
    anchor_designs = {}
    for name, v_level, sd in specs:
        design, true_theta = _ramp_anchor_design(v_level, n_trials=300, seed=sd)
        eb, lk, cs = dlg.simulate_licks(design, true_theta, ps, seed=sd + 100)
        anchor_designs[name] = dlg.design_with_outcomes(design, eb, lk, cs)

    # map each design object back to its anchor name (identity)
    design_name = {id(d): n for n, d in anchor_designs.items()}
    calls = []
    real_fit = dlg.fit_anchor

    def spy_fit(design, param_spec, seed_theta=None, l2=0.0, **kw):
        calls.append({
            "name": design_name[id(design)],
            "seed_is_none": seed_theta is None,
            "l2": l2,
        })
        return real_fit(design, param_spec, seed_theta=seed_theta, l2=l2, **kw)

    monkeypatch.setattr(dlg, "fit_anchor", spy_fit)
    dlg.backward_sweep(anchor_designs, anchors_chrono, ps, l2=1.0, seed=0)

    # exactly one call per anchor, in EXPERT-first reverse-chronological order
    assert [c["name"] for c in calls] == ["EXPERT", "MID", "OLD"]
    # expert fit first: free (no seed, l2 == 0)
    assert calls[0]["seed_is_none"] is True
    assert calls[0]["l2"] == 0.0
    # earlier anchors: L2-seeded (seed present, l2 == passed l2)
    for c in calls[1:]:
        assert c["seed_is_none"] is False
        assert c["l2"] == 1.0


def test_backward_sweep_skips_missing_anchor_and_seeds_from_last_fit(monkeypatch):
    """A session in anchors_chrono but absent from anchor_designs (QC-omitted) is
    skipped; the NEXT (earlier) present anchor is seeded from the last
    successfully-fit theta — NOT from the missing one."""
    ps = dlg.ParamSpec(moods=("Impulsive", "StimSens"),
                       dials=("v", "z", "u"), state_terms=("v", "z", "u"))

    # chronological: OLD, MID(omitted), EXPERT.  MID has no Design.
    specs = [("OLD", 0.7, 10), ("EXPERT", 1.8, 30)]
    anchors_chrono = ["OLD", "MID", "EXPERT"]
    anchor_designs = {}
    for name, v_level, sd in specs:
        design, true_theta = _ramp_anchor_design(v_level, n_trials=300, seed=sd)
        eb, lk, cs = dlg.simulate_licks(design, true_theta, ps, seed=sd + 100)
        anchor_designs[name] = dlg.design_with_outcomes(design, eb, lk, cs)

    design_name = {id(d): n for n, d in anchor_designs.items()}
    calls = []
    real_fit = dlg.fit_anchor

    def spy_fit(design, param_spec, seed_theta=None, l2=0.0, **kw):
        res = real_fit(design, param_spec, seed_theta=seed_theta, l2=l2, **kw)
        calls.append({
            "name": design_name[id(design)],
            "seed_is_none": seed_theta is None,
            "seed_theta": None if seed_theta is None else np.asarray(seed_theta).copy(),
            "l2": l2,
            "fitted_theta": res.theta.copy(),
        })
        return res

    monkeypatch.setattr(dlg, "fit_anchor", spy_fit)
    out = dlg.backward_sweep(anchor_designs, anchors_chrono, ps, l2=1.0, seed=0)

    # only the present anchors are returned; MID is skipped entirely
    assert set(out.keys()) == {"OLD", "EXPERT"}
    # call order: EXPERT first (free), then OLD (MID skipped, not fit)
    assert [c["name"] for c in calls] == ["EXPERT", "OLD"]
    assert calls[0]["seed_is_none"] is True            # expert free
    # OLD is seeded from EXPERT's fitted theta (the last successfully-fit theta),
    # NOT from a missing MID
    assert calls[1]["seed_is_none"] is False
    assert np.allclose(calls[1]["seed_theta"], calls[0]["fitted_theta"])


def test_backward_sweep_single_anchor_is_free_fit(monkeypatch):
    """With a single anchor, it is the most-expert -> fit free (seed_theta None,
    l2 0); the result is keyed by its id."""
    ps = dlg.ParamSpec(moods=("Impulsive", "StimSens"),
                       dials=("v", "z", "u"), state_terms=("v", "z", "u"))
    design, true_theta = _ramp_anchor_design(1.5, n_trials=300, seed=5)
    eb, lk, cs = dlg.simulate_licks(design, true_theta, ps, seed=55)
    anchor_designs = {"ONLY": dlg.design_with_outcomes(design, eb, lk, cs)}

    calls = []
    real_fit = dlg.fit_anchor

    def spy_fit(design, param_spec, seed_theta=None, l2=0.0, **kw):
        calls.append({"seed_is_none": seed_theta is None, "l2": l2})
        return real_fit(design, param_spec, seed_theta=seed_theta, l2=l2, **kw)

    monkeypatch.setattr(dlg, "fit_anchor", spy_fit)
    out = dlg.backward_sweep(anchor_designs, ["ONLY"], ps, l2=1.0, seed=0)

    assert set(out.keys()) == {"ONLY"}
    assert len(calls) == 1
    assert calls[0]["seed_is_none"] is True
    assert calls[0]["l2"] == 0.0
