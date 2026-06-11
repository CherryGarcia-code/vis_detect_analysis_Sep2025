import numpy as np
import pandas as pd
import pytest
from visdetect.analysis.ddm import build_trial_evidence, DT


def _toy_session():
    # Two trials: a Hit (lick after change) and an FA (early lick, no change reached).
    from types import SimpleNamespace
    base = np.r_[np.ones(20), np.ones(20) * 4.0]   # log2-able TF: 1 (e=0) then 4 (e=2)
    t_hit = SimpleNamespace(trialoutcome="Hit", change_size=4.0, change_time=1.0,
                            reactiontimes={"RT": 0.3}, baseline_values=base, n_seen=None)
    t_fa = SimpleNamespace(trialoutcome="FA", change_size=1.0, change_time=2.0,
                           reactiontimes={"FA": 0.5}, baseline_values=base, n_seen=None)
    return SimpleNamespace(trials=[t_hit, t_fa],
                           ni_events={"Baseline_ON": np.array([0.0, 10.0]),
                                      "Change_ON": np.array([1.0, 12.0])})


def test_build_trial_evidence_truncates_at_decision():
    sess = _toy_session()
    df = build_trial_evidence(sess, tf_base=1.0)
    assert len(df) == 2
    hit = df.iloc[0]
    # Hit decision_time = change_time + RT = 1.3 s -> evidence length = round(1.3/DT)
    assert hit["decision_time"] == pytest.approx(1.3, abs=DT)
    assert len(hit["evidence"]) == pytest.approx(1.3 / DT, abs=1)
    # FA decision_time = FA latency 0.5 s (truncated well before its change_time 2.0)
    fa = df.iloc[1]
    assert fa["decision_time"] == pytest.approx(0.5, abs=DT)
    assert len(fa["evidence"]) < len(hit["evidence"])
    assert fa["lick"] == 1 and hit["lick"] == 1
    # evidence is log2(TF/base): 0 in the first second, ~2 after the change (Hit only)
    assert abs(hit["evidence"][0]) < 1e-6
