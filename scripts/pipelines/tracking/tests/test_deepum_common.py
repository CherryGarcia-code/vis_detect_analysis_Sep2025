import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_deepum_common import AverageMeter, augment_channel_roll


def test_average_meter():
    m = AverageMeter()
    m.update(2.0, n=1); m.update(4.0, n=3)
    assert m.count == 4
    assert abs(m.avg - (2.0 + 4.0 * 3) / 4) < 1e-9
    print("test_average_meter PASS")


def test_augment_none_is_identity():
    data = np.arange(60 * 30, dtype=float).reshape(60, 30)
    out = augment_channel_roll(data, choice="none")
    assert out.shape == data.shape
    assert np.array_equal(out, data)
    assert out is not data  # must not mutate caller's array
    print("test_augment_none_is_identity PASS")


def test_augment_roll_preserves_shape_and_changes_values():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((60, 30))
    for choice in ("roll_up", "roll_down"):
        out = augment_channel_roll(data, choice=choice)
        assert out.shape == (60, 30)
        assert not np.array_equal(out, data)  # a non-constant input must change
    # original is untouched
    assert np.array_equal(data, data)
    print("test_augment_roll_preserves_shape_and_changes_values PASS")


if __name__ == "__main__":
    test_average_meter()
    test_augment_none_is_identity()
    test_augment_roll_preserves_shape_and_changes_values()
    print("ALL COMMON-BASIC TESTS PASSED")
