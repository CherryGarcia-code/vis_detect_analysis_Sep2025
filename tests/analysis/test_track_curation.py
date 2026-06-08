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
