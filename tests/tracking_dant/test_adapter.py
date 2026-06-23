import numpy as np
import pytest

import adapter


def test_collapse_cv_shape_and_mean():
    # (n_samp=4, n_ch=3, n_cv=2)
    raw = np.zeros((4, 3, 2), dtype=np.float32)
    raw[..., 0] = 1.0
    raw[..., 1] = 3.0  # mean over cv -> 2.0 everywhere
    out = adapter.collapse_cv(raw)
    assert out.shape == (3, 4)            # (n_ch, n_samp)
    assert np.allclose(out, 2.0)


def test_collapse_cv_rejects_bad_shape():
    with pytest.raises(ValueError):
        adapter.collapse_cv(np.zeros((4, 3)))         # not 3D
    with pytest.raises(ValueError):
        adapter.collapse_cv(np.zeros((4, 3, 5)))      # cv axis != 2


def test_derive_channel_shanks_four_shanks():
    # BG_046 x-layout: 4 shanks x 2 columns, ~250 um apart
    xs = np.array([27, 59, 277, 309, 527, 559, 777, 809], dtype=float)
    pos = np.column_stack([xs, np.zeros_like(xs)])
    shanks = adapter.derive_channel_shanks(pos)
    assert shanks.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert shanks.dtype == np.int64


def test_seconds_to_ms():
    out = adapter.seconds_to_ms(np.array([0.0, 1.0, 2.5]))
    assert np.allclose(out, [0.0, 1000.0, 2500.0])


def test_is_positive_going():
    # negative-going: trough deeper than peak on the peak channel
    neg = np.zeros((2, 10)); neg[0] = -5.0; neg[0, 5] = -10.0; neg[0, 0] = 2.0
    assert adapter.is_positive_going(neg) is False
    # positive-going: peak taller than trough on the peak channel
    pos = np.zeros((2, 10)); pos[0, 5] = 10.0; pos[0, 0] = -2.0
    assert adapter.is_positive_going(pos) is True
