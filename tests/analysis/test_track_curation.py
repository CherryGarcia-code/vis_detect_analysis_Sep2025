import numpy as np
import pytest
from visdetect.analysis import track_curation as tc


def test_partitioned_isi_hists_disjoint_and_valid():
    rng = np.random.default_rng(0)
    spikes = np.cumsum(rng.exponential(0.05, size=4000))   # stationary-ish train
    cur, hold = tc.partitioned_isi_hists(spikes)
    assert cur.shape == (50,) and hold.shape == (50,)
    assert np.isfinite(cur).all() and np.isfinite(hold).all()
    # Same underlying distribution -> the two partitions correlate strongly
    r = np.corrcoef(cur, hold)[0, 1]
    assert r > 0.8


def test_partitioned_isi_hists_too_few_spikes_returns_nan():
    cur, hold = tc.partitioned_isi_hists(np.array([0.1, 0.2, 0.3]))
    assert np.isnan(cur).all() and np.isnan(hold).all()


from visdetect.analysis import tracking_qc as qc
from visdetect.utils.synthetic import make_synthetic_session


def test_extract_unit_psths_restrict_trials_subsets():
    sess = make_synthetic_session(n_trials=40, n_clusters=3, seed=1)
    full = qc.extract_unit_psths(sess, ks_unit_id=0)
    restricted = qc.extract_unit_psths(sess, ks_unit_id=0, restrict_trials={0, 1, 2, 3, 4})
    # baseline_on uses all trials when unrestricted; restricting lowers n_trials
    assert restricted["baseline_on"][2] <= 5
    assert full["baseline_on"][2] >= restricted["baseline_on"][2]


def test_extract_unit_psths_empty_restrict_returns_none():
    sess = make_synthetic_session(n_trials=40, n_clusters=3, seed=1)
    out = qc.extract_unit_psths(sess, ks_unit_id=0, restrict_trials=set())
    assert out["baseline_on"] == (None, None, 0)
