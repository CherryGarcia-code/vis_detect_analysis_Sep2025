import matplotlib
matplotlib.use("Agg")  # headless backend; must precede the pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from visdetect.analysis import config as CFG
from visdetect.analysis import constants as C
from visdetect.core.session import Session, Trial
from visdetect.analysis.state_labeling import (
    build_outcome_raster,
    classify_lick_valence,
    episodes_to_trial_labels,
    get_labeling_queue,
    load_episodes,
    render_raster,
    save_episode,
    StateEpisode,
)


def test_state_constants_exist():
    assert C.STATE_LABELS == ["Impulsive", "StimSens", "Disengaged"]
    assert C.STATE_EASY_CHANGE_THRESH == 2.0
    assert C.STATE_CONFIDENCE_THRESHOLD == 0.8
    assert C.STATE_LABEL_W_DEFAULT in C.STATE_LABEL_W_GRID
    assert C.STATE_FEATURE_COLS == [
        "f_applick", "f_inapplick", "f_nolick", "f_abort", "f_miss_easy", "f_hit_hard",
    ]


def test_lick_valence_colors():
    for k in ["appropriate_lick", "inappropriate_lick", "nolick", "abort", "ref"]:
        assert k in CFG.LICK_VALENCE_COLORS


@pytest.mark.parametrize("outcome,is_go,is_catch,expected", [
    ("hit",  True,  False, "appropriate_lick"),    # go hit
    ("Hit",  True,  False, "appropriate_lick"),    # case-insensitive
    ("hit",  False, True,  "inappropriate_lick"),  # catch SDT false alarm
    ("miss", True,  False, "nolick"),              # go miss
    ("miss", False, True,  "nolick"),              # correct rejection
    ("fa",   True,  False, "inappropriate_lick"),  # early lick on go
    ("fa",   False, True,  "inappropriate_lick"),  # early lick on catch
    ("abort", True, False, "abort"),
    ("ref",  True,  False, "ref"),
])
def test_classify_lick_valence(outcome, is_go, is_catch, expected):
    assert classify_lick_valence(outcome, is_go, is_catch) == expected


def test_episode_save_load_roundtrip(tmp_path):
    path = tmp_path / "episodes.csv"
    e1 = StateEpisode("07072025", 10, 25, "Impulsive", "ben", "2026-06-10T00:00:00")
    e2 = StateEpisode("07072025", 40, 55, "Disengaged", "ben", "2026-06-10T00:01:00", notes="zoned out")
    save_episode(e1, path)
    save_episode(e2, path)
    loaded = load_episodes(path)
    assert len(loaded) == 2
    assert loaded[0].session_name == "07072025"
    assert loaded[0].start_trial == 10 and loaded[0].end_trial == 25
    assert loaded[1].state_label == "Disengaged"
    assert loaded[1].notes == "zoned out"


def test_save_episode_writes_header_into_empty_file(tmp_path):
    # A zero-byte file (e.g. left by a crash mid-write) must still get a header,
    # otherwise the first appended row would be misparsed as column names.
    path = tmp_path / "episodes.csv"
    path.touch()
    save_episode(StateEpisode("07072025", 3, 9, "StimSens", "ben", "t"), path)
    loaded = load_episodes(path)
    assert len(loaded) == 1
    assert loaded[0].session_name == "07072025"
    assert loaded[0].start_trial == 3 and loaded[0].end_trial == 9
    assert loaded[0].state_label == "StimSens"


def test_episodes_to_trial_labels():
    eps = [
        StateEpisode("S1", 2, 4, "Impulsive", "ben", "t"),
        StateEpisode("S1", 7, 8, "Disengaged", "ben", "t"),
        StateEpisode("S2", 0, 1, "StimSens", "ben", "t"),  # different session ignored
    ]
    labels = episodes_to_trial_labels(eps, "S1", n_trials=10)
    assert labels[0] is None and labels[1] is None
    assert list(labels[2:5]) == ["Impulsive", "Impulsive", "Impulsive"]
    assert labels[5] is None and labels[6] is None
    assert list(labels[7:9]) == ["Disengaged", "Disengaged"]
    assert labels[9] is None


def _trial(outcome, change_size):
    return Trial(
        trialoutcome=outcome, reactiontimes={}, change_size=change_size,
        orientation=None, ITI=1.0, change_time=2.0, baseline_values=np.zeros(5),
    )


def _session(trials):
    return Session(
        trials=trials, clusters=[], subject="T", session_name="T1",
        good_cluster_ids=[], ni_events={},
    )


def test_build_outcome_raster_lick_valence():
    trials = [
        _trial("Hit", 2.0),    # go hit          -> appropriate_lick
        _trial("Hit", 1.0),    # catch SDT FA    -> inappropriate_lick
        _trial("Miss", 4.0),   # go miss         -> nolick
        _trial("Miss", 1.0),   # correct reject  -> nolick
        _trial("FA", 1.5),     # early lick      -> inappropriate_lick
        _trial("abort", 1.5),  # abort           -> abort
    ]
    raster = build_outcome_raster(_session(trials))
    assert list(raster["lick_valence"]) == [
        "appropriate_lick", "inappropriate_lick", "nolick",
        "nolick", "inappropriate_lick", "abort",
    ]
    # color column is populated from LICK_VALENCE_COLORS
    assert raster.loc[0, "color"] == "#2e8b57"
    assert set(["trial_idx", "is_go", "is_catch", "change_size"]).issubset(raster.columns)
    # SDT behavioral-label booleans are carried for hmm_downstream drop-in
    assert set(["is_hit", "is_fa", "is_miss"]).issubset(raster.columns)
    assert bool(raster.loc[0, "is_hit"]) is True          # go hit
    assert bool(raster.loc[4, "is_fa"]) is True           # early lick
    assert bool(raster.loc[2, "is_miss"]) is True         # go miss


def test_build_outcome_raster_empty_session_keeps_schema():
    raster = build_outcome_raster(_session([]))
    assert len(raster) == 0
    # an empty session still returns the raster schema (not get_trial_dataframe's columns)
    assert set(["trial_idx", "outcome", "is_go", "is_catch",
                "change_size", "lick_valence", "color"]).issubset(raster.columns)


def test_get_labeling_queue_expert_first_then_recent():
    manifest = pd.DataFrame({
        "session_name": ["01012025", "01032025", "01062025", "15062025"],
        "stage": ["Learning", "Learning", "Expert", "Expert"],
    })
    queue = get_labeling_queue(manifest=manifest)
    # Expert sessions first (most-recent first), then Learning (most-recent first)
    assert queue == ["15062025", "01062025", "01032025", "01012025"]


def test_render_raster_draws_one_patch_per_trial():
    raster = build_outcome_raster(_session([
        _trial("Hit", 2.0), _trial("Miss", 1.0), _trial("FA", 1.5),
    ]))
    fig, ax = plt.subplots()
    render_raster(ax, raster)
    # one colored bar per trial
    assert len(ax.patches) >= 3
    plt.close(fig)


def test_render_raster_change_size_shading_runs():
    raster = build_outcome_raster(_session([_trial("Hit", 1.25), _trial("Hit", 4.0)]))
    fig, ax = plt.subplots()
    render_raster(ax, raster, change_size_shading=True)
    assert len(ax.patches) >= 2
    plt.close(fig)


def test_render_raster_outlines_only_catch_trials():
    # trial 0 is a go trial (cs>1, edge "none"); trial 1 is a catch trial (cs==1, outlined)
    raster = build_outcome_raster(_session([_trial("Hit", 2.0), _trial("Hit", 1.0)]))
    fig, ax = plt.subplots()
    render_raster(ax, raster)  # no episodes -> only the two trial rectangles
    edge_alphas = sorted(p.get_edgecolor()[3] for p in ax.patches)
    assert edge_alphas[0] == 0.0    # the go trial has no outline
    assert edge_alphas[-1] > 0.0    # the catch trial is outlined
    plt.close(fig)


def test_render_raster_episode_spans_add_patches():
    raster = build_outcome_raster(_session([_trial("Hit", 2.0), _trial("Hit", 1.0)]))
    fig, ax = plt.subplots()
    render_raster(ax, raster)
    base = len(ax.patches)
    plt.close(fig)
    fig2, ax2 = plt.subplots()
    render_raster(ax2, raster, episodes=[StateEpisode("T1", 0, 1, "Impulsive", "ben", "t")])
    assert len(ax2.patches) == base + 1   # one extra span patch behind the trial bars
    plt.close(fig2)
