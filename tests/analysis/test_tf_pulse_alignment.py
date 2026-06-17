"""Regression tests for _collect_pulses trial-alignment.

Guards against the off-by-one where `enumerate(trials, 1)` indexed the
0-indexed per-trial onset arrays, pairing each trial's TF values with the
NEXT trial's baseline onset (and dropping the last trial). Because per-trial
TF sequences are independent, that scrambles the fast/slow -> spike alignment.
"""
import numpy as np

from visdetect.core.session import Session, Trial
from visdetect.analysis.tf_pulse import _collect_pulses, TFRespPulseConfig


def _make_session():
    """3 trials with long baselines so pulses clear the guards.

    Each trial's TF vector is neutral (TF=1, log2=0) except a single fast
    sample (TF=2) at post-stride index 100 -> 5.0 s after that trial's onset.
    Only trials 0 and 2 carry the fast sample, so we can check that the fast
    pulse lands at the CORRECT trial's onset and that the last trial survives.
    """
    base_on = [0.0, 100.0, 200.0]
    change_on = [50.0, 150.0, 250.0]
    trials = []
    for k in range(3):
        bv = np.ones(1800)              # neutral TF=1 (3x-upsampled length)
        if k in (0, 2):
            bv[300:303] = 2.0           # post-stride index 100 -> +5.0 s
        trials.append(Trial(
            trialoutcome="Hit", reactiontimes={"RT": 0.3},
            change_size=2.0, change_time=change_on[k] - base_on[k],
            baseline_values=bv, n_seen=None))
    ni = {"Baseline_ON": np.array(base_on), "Change_ON": np.array(change_on)}
    return Session(trials=trials, clusters=[], ni_events=ni, session_name="ALIGN")


def test_pulse_times_use_correct_trial_onset():
    fast, _ = _collect_pulses(_make_session(), TFRespPulseConfig(use_constraints=True))
    # Trial 0's fast sample -> onset(0) + 100*0.05 = 5.0 s
    assert np.any(np.isclose(fast, 5.0, atol=0.051)), \
        f"expected fast pulse ~5.0 s (trial 0 onset), got {np.sort(fast)[:10]}"
    # Off-by-one would place trial 0's value at trial 1's onset (100) -> 105.0 s
    assert not np.any(np.isclose(fast, 105.0, atol=0.051)), \
        "off-by-one: trial 0's pulse leaked onto trial 1's onset"


def test_last_trial_not_dropped():
    fast, _ = _collect_pulses(_make_session(), TFRespPulseConfig(use_constraints=True))
    # Trial 2's fast sample -> onset(200) + 5.0 = 205.0 s
    assert np.any(np.isclose(fast, 205.0, atol=0.051)), \
        f"last trial dropped: no pulse ~205.0 s; got {np.sort(fast)[-5:]}"
