"""Smoke test for the visdetect-Session TF-GLM regressor builder (Task 10).

Runs against a real BG_046 pkl in the PRIMARY repo; skipped when absent.
"""
import os
import numpy as np
import pytest

from visdetect.analysis.tf_glm import TFGLMConfig
from visdetect.analysis.tf_glm_data import session_trial_regressors

PKL = (r"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/"
       r"data/pkls/BG_046/BG_046_01072025.pkl")


@pytest.mark.skipif(not os.path.isfile(PKL), reason="BG_046 pkl not available")
def test_session_trial_regressors_smoke():
    from visdetect.core.session import load_session
    session = load_session(PKL)
    cfg = TFGLMConfig(include_phase=False, fast_fit=True)
    trials, units = session_trial_regressors(session, cfg)

    # One regressor block per Session trial (indexing preserved).
    assert len(trials) == len(session.trials)

    # Some change_time finite (hit/miss/ref) and some NaN (FA/abort).
    cts = np.array([tr.change_time for tr in trials], float)
    assert np.isfinite(cts).any(), "expected some finite change_time"
    assert np.isnan(cts).any(), "expected some NaN change_time (FA/abort)"

    # tf_bins / wheel_bins are 1-D per trial.
    for tr in trials[:20]:
        assert np.ndim(tr.tf_bins) == 1
        assert np.ndim(tr.wheel_bins) == 1

    # Units non-empty, mapped to 1-D spike trains.
    assert len(units) > 0
    for uid, st in list(units.items())[:5]:
        assert np.ndim(st) == 1


@pytest.mark.skipif(not os.path.isfile(PKL), reason="BG_046 pkl not available")
def test_session_tf_bins_zeroed_after_change():
    """TF signal must be zero at/after change onset on change-reached trials."""
    from visdetect.core.session import load_session
    session = load_session(PKL)
    cfg = TFGLMConfig(include_phase=False, fast_fit=True)
    trials, _ = session_trial_regressors(session, cfg)
    from visdetect.analysis.tf_glm import trial_bin_edges
    checked = 0
    for tr in trials:
        if not np.isfinite(tr.change_time):
            continue
        edges = trial_bin_edges(tr.t_start, tr.t_end, cfg.bin_s)
        if edges.size != tr.tf_bins.size:
            continue
        post = tr.tf_bins[edges >= tr.change_time]
        if post.size:
            assert np.allclose(post, 0.0)
            checked += 1
        if checked >= 10:
            break
    assert checked > 0
