# tests/video/test_tagging_logic.py
import numpy as np
import pytest
from visdetect.analysis import tagging


class _Trial:
    def __init__(self, change_size, outcome):
        self.change_size = change_size
        self.trialoutcome = outcome


class _Sess:
    def __init__(self, trials):
        self.trials = trials


def test_build_change_queue_orders_size4_then_size2_hitmiss_only(monkeypatch):
    trials = [
        _Trial(4.0, "hit"),    # 0 -> keep, size4
        _Trial(2.0, "miss"),   # 1 -> keep, size2
        _Trial(1.25, "hit"),   # 2 -> drop (small change)
        _Trial(4.0, "miss"),   # 3 -> keep, size4
        _Trial(4.0, "fa"),     # 4 -> dropped by getter (NaN, not hit/miss)
        _Trial(1.0, "miss"),   # 5 -> catch, drop
    ]
    sess = _Sess(trials)
    # getter returns absolute Change_ON s per trial, NaN off hit/miss (idx 4 fa, idx 5 catch treated hit/miss but small)
    fake = [10.0, 20.0, 30.0, 40.0, float("nan"), 60.0]
    monkeypatch.setattr(tagging, "get_event_times_by_trial", lambda s, e: fake)
    q = tagging.build_change_queue(sess)
    assert [t.trial_index for t in q] == [0, 3, 1]      # size4 (0,3) then size2 (1)
    assert [t.change_size for t in q] == [4.0, 4.0, 2.0]
    assert q[0].change_on_s == 10.0 and q[0].outcome == "hit"
