"""Tests for tag_trials state-machine transitions (pure-logic, no UI)."""
from __future__ import annotations

import importlib.util
import os

import pytest


def _import_tag_trials():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec_path = os.path.join(project_root, "scripts", "video", "tag_trials.py")
    spec = importlib.util.spec_from_file_location("tag_trials", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_initial_resume_idx_no_overrides():
    tt = _import_tag_trials()
    assert tt.initial_resume_idx({}, n_trials=10) == 0


def test_initial_resume_idx_with_overrides():
    tt = _import_tag_trials()
    # Trials 0, 1, 2 done; resume at 3
    overrides = {0: 100, 1: 200, 2: 300}
    assert tt.initial_resume_idx(overrides, n_trials=10) == 3


def test_initial_resume_idx_all_done_returns_n_trials():
    tt = _import_tag_trials()
    overrides = {i: i * 10 for i in range(5)}
    assert tt.initial_resume_idx(overrides, n_trials=5) == 5


def test_handle_enter_sets_override_and_advances():
    tt = _import_tag_trials()
    state = tt.TagState(trial_idx=3, overrides={}, n_trials=10)
    new_state = tt.handle_enter(state, current_frame=999)
    assert new_state.overrides == {3: 999}
    assert new_state.trial_idx == 4
    assert not new_state.done


def test_handle_enter_at_last_trial_marks_done():
    tt = _import_tag_trials()
    state = tt.TagState(trial_idx=9, overrides={5: 500}, n_trials=10)
    new_state = tt.handle_enter(state, current_frame=1000)
    assert new_state.overrides == {5: 500, 9: 1000}
    assert new_state.trial_idx == 10
    assert new_state.done


def test_handle_skip_preserves_overrides_and_advances():
    tt = _import_tag_trials()
    state = tt.TagState(trial_idx=3, overrides={3: 999}, n_trials=10)
    new_state = tt.handle_skip(state)
    assert new_state.overrides == {3: 999}
    assert new_state.trial_idx == 4


def test_handle_skip_no_existing_override_is_noop_on_overrides():
    tt = _import_tag_trials()
    state = tt.TagState(trial_idx=3, overrides={}, n_trials=10)
    new_state = tt.handle_skip(state)
    assert new_state.overrides == {}
    assert new_state.trial_idx == 4


def test_handle_delete_removes_override_and_advances():
    tt = _import_tag_trials()
    state = tt.TagState(trial_idx=3, overrides={3: 999, 5: 500}, n_trials=10)
    new_state = tt.handle_delete(state)
    assert new_state.overrides == {5: 500}
    assert new_state.trial_idx == 4


def test_handle_delete_no_existing_override_is_noop_on_overrides():
    tt = _import_tag_trials()
    state = tt.TagState(trial_idx=3, overrides={}, n_trials=10)
    new_state = tt.handle_delete(state)
    assert new_state.overrides == {}
    assert new_state.trial_idx == 4


def test_handle_back_decrements_trial_idx():
    tt = _import_tag_trials()
    state = tt.TagState(trial_idx=5, overrides={2: 200}, n_trials=10)
    new_state = tt.handle_back(state)
    assert new_state.trial_idx == 4
    assert new_state.overrides == {2: 200}


def test_handle_back_at_trial_zero_stays_at_zero():
    tt = _import_tag_trials()
    state = tt.TagState(trial_idx=0, overrides={}, n_trials=10)
    new_state = tt.handle_back(state)
    assert new_state.trial_idx == 0


def test_slope_fit_frame_basic():
    tt = _import_tag_trials()
    # slope=1.0, offset=0.0, nidaq=10.0 s, fps=50.0 → video_time=10.0 s → frame=500
    sync_json = {"eye_cam": {"slope": 1.0, "offset": 0.0}}
    assert tt._slope_fit_frame(sync_json, nidaq_baseline_on_s=10.0, fps=50.0) == 500

    # slope=1.02, offset=-0.2, nidaq=20.0 s, fps=50.0
    # video_time = 1.02*20.0 + (-0.2) = 20.4 - 0.2 = 20.2 s → frame = round(20.2*50) = 1010
    sync_json2 = {"eye_cam": {"slope": 1.02, "offset": -0.2}}
    assert tt._slope_fit_frame(sync_json2, nidaq_baseline_on_s=20.0, fps=50.0) == 1010
