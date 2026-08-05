"""Real-pkl validation of the QC1 alignment solver.

Skips when the pkls are absent (data/ is gitignored), so CI stays green.
Expected values come from the spec's measured table (§2).
"""
import gc
import os

import numpy as np
import pytest

from visdetect.core.run_alignment import solve_alignment
from visdetect.core.session import load_session

PKL_DIR = os.path.join("data", "pkls", "BG_046")

CASES = [
    # file,                    trial_start, event_offset, n_matched
    ("BG_046_19082025.pkl",    0,   0,   587),   # known good -> identity
    ("BG_046_20082025.pkl",    0,   228, 486),   # sign B: untrimmed ephys
    ("BG_046_05092025_b.pkl",  281, 0,   248),   # sign A: concatenated runs
]


@pytest.mark.parametrize("fname,exp_start,exp_off,exp_n", CASES)
def test_solver_recovers_known_alignment(fname, exp_start, exp_off, exp_n):
    path = os.path.join(PKL_DIR, fname)
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    s = load_session(path)
    try:
        a = solve_alignment(s.trials, s.ni_events)
        assert a is not None, f"{fname}: solver failed to find an alignment"
        assert a.trial_start == exp_start
        assert a.event_offset == exp_off
        assert a.n_trials_matched == exp_n
        assert a.agreement == pytest.approx(1.0)
        assert a.resid_s < 0.05
        # uniqueness: the runner-up must NOT also pass
        assert not (a.runner_up_agreement >= 1.0 and a.runner_up_resid_s < 0.05)
    finally:
        del s
        gc.collect()
