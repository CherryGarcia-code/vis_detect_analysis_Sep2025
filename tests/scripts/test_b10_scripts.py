"""B10 evidence_learning script smoke tests — pure seams on synthetic sessions.

Scripts live under scripts/ and are loaded BY PATH (repo convention; see
tests/scripts/test_validate_selectivity_phase0.py). No real pkls required.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from visdetect.analysis import psychophysical_kernel as pk

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "evidence_learning"
_L = len(pk.kernel_lags())


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mk_session(n_fa=30, n_hit=20, with_clusters=False, seed=0):
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(n_fa):
        bv = np.repeat(np.exp2(rng.normal(0, 0.25, 800)), 3)
        trials.append(SimpleNamespace(baseline_values=bv, reactiontimes={"FA": 5.0},
                                      trialoutcome="fa", change_time=np.nan,
                                      change_size=1.0))
    for _ in range(n_hit):
        bv = np.repeat(np.exp2(rng.normal(0, 0.25, 800)), 3)
        trials.append(SimpleNamespace(baseline_values=bv, reactiontimes={},
                                      trialoutcome="hit", change_time=8.0,
                                      change_size=2.0))
    n = len(trials)
    clusters = ([SimpleNamespace(cluster_id=7,
                 spike_times=np.sort(rng.uniform(0, 10.0 * n, 6000)))]
                if with_clusters else [])
    return SimpleNamespace(trials=trials,
                           ni_events={"Baseline_ON": 10.0 * np.arange(n)},
                           clusters=clusters)


def test_coverage_row_counts():
    cov = _load("b10_phase0_coverage")
    row = cov.coverage_row("BG_046", "Expert", "01072025", _mk_session())
    assert row["n_fa_usable"] == 30
    assert row["n_withhold_ok"] == 30
    assert row["subject"] == "BG_046" and row["stage"] == "Expert"


def test_session_kernel_pairs_and_length():
    b = _load("b10_phase1_behavioral")
    k, npairs = b.session_kernel(_mk_session(40), np.random.default_rng(0))
    assert k.shape[0] == _L and npairs == 40


def test_neural_fa_withhold_shapes():
    nmod = _load("b10_phase1_neural")
    sess = _mk_session(30, 20, with_clusters=True)
    fp, wp, fs, ws = nmod.neural_fa_withhold(sess, {7: +1}, np.random.default_rng(0))
    assert len(fp) == len(wp) == len(fs) == len(ws) > 0
    assert all(len(w) == _L for w in fp)
    assert all(len(w) == _L for w in wp)


def test_fa_epochs_by_state_splits(monkeypatch):
    s = _load("b10_phase2_state")
    sess = _mk_session(10, 0)
    labels = pd.DataFrame({"state_label": ["StimSens"] * 6 + ["Impulsive"] * 4,
                           "state_confidence": [0.9] * 10}, index=range(10))
    monkeypatch.setattr(s, "load_state_labels_by_key", lambda *a, **k: labels)
    by = s.fa_epochs_by_state(sess, "BG_046", skey="01072025")
    assert len(by["StimSens"]) == 6 and len(by["Impulsive"]) == 4


def test_session_tracking_shapes():
    t = _load("b10_tf_tracking")
    sess = _mk_session(10, 10, with_clusters=True)
    real, shuf, peakr = t.session_tracking(sess, {7: +1}, np.random.default_rng(0))
    max_lag = int(round(t.MAX_LAG_S / pk.DT))
    assert real is not None and real.shape[0] == max_lag + 1
    assert isinstance(peakr, dict) and len(peakr) > 0
