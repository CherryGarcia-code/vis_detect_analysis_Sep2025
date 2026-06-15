# tests/analysis/test_tf_pulse_guards.py
import numpy as np
from visdetect.core.session import Trial
from visdetect.analysis.tf_pulse import _outcome_time_for_trial, _collect_pulses, TFRespPulseConfig
from visdetect.utils.synthetic import make_synthetic_session


def _trial(outcome, rts):
    return Trial(trialoutcome=outcome, reactiontimes=rts)


def test_outcome_time_uppercase_fa():
    # Existing behaviour must still work: FA lick at baseline_t + rt
    t = _trial("FA", {"FA": 3.5})
    assert _outcome_time_for_trial(t, 10.0) == 10.0 + 3.5


def test_outcome_time_lowercase_fa_is_now_caught():
    # Was silently None before the fix (lowercase 'fa' != 'FA')
    t = _trial("fa", {"fa": 3.5})
    assert _outcome_time_for_trial(t, 10.0) == 13.5


def test_outcome_time_capitalised_abort_is_now_caught():
    # Was silently None before the fix ('Abort' != 'abort')
    t = _trial("Abort", {"Abort": 1.2})
    assert _outcome_time_for_trial(t, 10.0) == 11.2


def test_outcome_time_ref_is_covered():
    t = _trial("ref", {"ref": 0.2})
    assert _outcome_time_for_trial(t, 10.0) == 10.2


def test_outcome_time_hit_returns_none():
    # Hit is not a baseline lick -> no early-reaction time
    t = _trial("Hit", {"RT": 0.3})
    assert _outcome_time_for_trial(t, 10.0) is None


def test_outcome_time_none_baseline_returns_none():
    t = _trial("FA", {"FA": 3.5})
    assert _outcome_time_for_trial(t, None) is None


def test_outcome_time_missing_rt_key_returns_none():
    # Outcome is a baseline lick but no matching reaction-time key -> None
    t = _trial("FA", {"RT": 0.3})
    assert _outcome_time_for_trial(t, 10.0) is None


def test_collect_pulses_constraints_reduce_count():
    sess = make_synthetic_session(n_trials=30, n_clusters=2, seed=3)
    cfg_on = TFRespPulseConfig(use_constraints=True)
    cfg_off = TFRespPulseConfig(use_constraints=False)
    fast_on, slow_on = _collect_pulses(sess, cfg_on)
    fast_off, slow_off = _collect_pulses(sess, cfg_off)
    # Guards can only remove pulses, never add them.
    assert fast_on.size <= fast_off.size
    assert slow_on.size <= slow_off.size
