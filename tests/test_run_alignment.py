import numpy as np
import pytest

from visdetect.core.run_alignment import (
    CHANGE_PRESENTED_OUTCOMES,
    MIN_RESID_N,
    alignment_residual,
    outcome_change_agreement,
    per_trial_event_keys,
)


class FakeTrial:
    def __init__(self, outcome, change_time):
        self.trialoutcome = outcome
        self.change_time = change_time


def make_case(n=60, offset=0, n_pad=0, seed=0):
    """Build trials + ni_events that are aligned at `offset`.

    Every 2nd trial is a Hit (change presented); the rest alternate FA/abort.
    `n_pad` prepends unrelated events so the true alignment is at index n_pad.
    `seed` controls the change_time draws: distinct seeds produce genuinely
    different change_times, so concatenated runs don't accidentally align on a
    shared-seed prefix (which would let a wrong trial_start pass Check 2).
    """
    rng = np.random.default_rng(seed)
    trials, bon, con = [], [], []
    for _ in range(n_pad):                      # orphan events from earlier runs
        t0 = len(bon) * 10.0
        bon.append(t0)
        con.append(t0 + 5.0)
    for i in range(n):
        t0 = (n_pad + i) * 10.0
        ct = round(float(rng.uniform(6.0, 11.0)), 3)
        if i % 2 == 0:
            outcome = "Hit"
            con.append(t0 + ct)                 # change WAS presented
        else:
            outcome = "FA" if i % 4 == 1 else "abort"
            con.append(np.nan)                  # change never presented
        bon.append(t0)
        trials.append(FakeTrial(outcome, ct))
    ni = {
        "Baseline_ON": np.array(bon, float),
        "Change_ON": np.array(con, float),
        "Valve_L": np.zeros(len(bon), float),
        "Rot_enc_A": np.zeros(9999, float),     # NOT per-trial
    }
    return trials, ni


def test_per_trial_event_keys_finds_equal_length_arrays():
    trials, ni = make_case(n=40)
    keys = per_trial_event_keys(ni)
    assert set(keys) == {"Baseline_ON", "Change_ON", "Valve_L"}
    assert "Rot_enc_A" not in keys


def test_agreement_is_one_at_correct_offset_and_chance_when_shifted():
    trials, ni = make_case(n=60, n_pad=25)
    good, n_cmp = outcome_change_agreement(trials, ni, slice(None), 25)
    assert good == pytest.approx(1.0)
    assert n_cmp == 60                      # 100% trial coverage
    bad, _ = outcome_change_agreement(trials, ni, slice(None), 24)
    assert bad < 0.95                       # a single-trial shift breaks it


def test_residual_is_zero_at_correct_offset_and_large_when_shifted():
    trials, ni = make_case(n=60, n_pad=25)
    med, n = alignment_residual(trials, ni, slice(None), 25)
    assert med == pytest.approx(0.0, abs=1e-9)
    assert n == 30                          # only the Hit trials
    med_bad, _ = alignment_residual(trials, ni, slice(None), 24)
    assert med_bad > 0.5


def test_residual_rejects_when_too_few_finite_trials():
    """n < MIN_RESID_N must yield nan (reject), never a vacuous pass."""
    trials, ni = make_case(n=10)            # only 5 Hit trials
    med, n = alignment_residual(trials, ni, slice(None), 0)
    assert n < MIN_RESID_N
    assert np.isnan(med)


def test_outcome_set_is_case_sensitive_and_includes_ref():
    assert CHANGE_PRESENTED_OUTCOMES == frozenset({"Hit", "Miss", "Ref"})
    assert "hit" not in CHANGE_PRESENTED_OUTCOMES


from visdetect.core.run_alignment import Alignment, build_trial_event_index, solve_alignment


def test_solver_finds_sign_b_offset():
    """Sign B: trials are correct, events are untrimmed -> a pure offset."""
    trials, ni = make_case(n=60, n_pad=25)
    a = solve_alignment(trials, ni)
    assert a is not None
    assert a.event_offset == 25
    assert a.trial_start == 0
    assert a.agreement == pytest.approx(1.0)
    assert a.resid_s < 0.05


def test_solver_finds_sign_a_trial_slice():
    """Sign A: trial table spans runs; only a suffix belongs to this recording."""
    trials_run2, ni = make_case(n=60)
    trials_run1, _ = make_case(n=40, seed=1)    # DISTINCT change_times from run 2
    trials = trials_run1 + trials_run2          # concatenated, as the converter does
    a = solve_alignment(trials, ni)
    assert a is not None
    assert a.trial_start == 40
    assert a.n_trials_matched == 60
    assert a.event_offset == 0
    # Uniqueness: the runner-up must NOT also pass both checks. This is the same
    # gate Task 3's real-data repair uses -- without distinct seeds, trial_start=0
    # would spuriously pass Check 2 on the shared-seed prefix and this would fail.
    assert not (a.runner_up_agreement >= 1.0 and a.runner_up_resid_s < 0.05)


def test_solver_returns_none_when_unsolvable():
    trials, ni = make_case(n=60)
    ni["Change_ON"] = np.full(len(ni["Change_ON"]), np.nan)   # no usable evidence
    assert solve_alignment(trials, ni) is None


def test_solver_returns_none_for_empty_trial_table():
    _, ni = make_case(n=60)
    assert solve_alignment([], ni) is None


def test_build_trial_event_index_maps_and_marks_missing():
    trials_run1, _ = make_case(n=40, seed=1)    # DISTINCT change_times from run 2
    trials_run2, ni = make_case(n=60)
    trials = trials_run1 + trials_run2
    a = solve_alignment(trials, ni)
    idx = build_trial_event_index(len(trials), a)
    assert idx.shape == (100,)
    assert (idx[:40] == -1).all()               # run-1 trials have no ephys here
    assert (idx[40:] == np.arange(60)).all()


def test_build_trial_event_index_all_minus_one_when_unsolved():
    idx = build_trial_event_index(7, None)
    assert (idx == -1).all()
