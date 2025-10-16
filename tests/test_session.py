import numpy as np
from visdetect.io import parse_good_cluster_ids


def test_parse_scalar_int():
    assert parse_good_cluster_ids(5) == [5]


def test_parse_numpy_array():
    arr = np.array([1, 2, 3])
    assert parse_good_cluster_ids(arr) == [1, 2, 3]


def test_parse_nested_array():
    arr = np.array([[1], [2], [3]])
    assert parse_good_cluster_ids(arr) == [1, 2, 3]


def test_parse_string_and_bytes():
    assert parse_good_cluster_ids('4') == [4]
    assert parse_good_cluster_ids(b'6') == [6]


def test_parse_empty_or_nan():
    assert parse_good_cluster_ids(np.array([])) is None
    assert parse_good_cluster_ids(np.array([np.nan])) is None


def test_parse_duplicates_and_sorting():
    assert parse_good_cluster_ids([3, 1, 2, 1]) == [1, 2, 3]
