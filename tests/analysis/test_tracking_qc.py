import pytest
from visdetect.analysis import tracking_qc as qc


def test_thresholds_present():
    assert qc.ISI_PASS == 0.75
    assert qc.ISI_WARN == 0.65
    assert qc.DEPTH_PASS_UM == 15.0
    assert qc.DEPTH_WARN_UM == 30.0
    assert qc.WAVE_PASS_R == 0.95
    assert qc.WAVE_WARN_R == 0.90
    assert qc.FR_CV_PASS == 0.35
    assert qc.FR_CV_WARN == 0.60


def test_change_size_pools():
    assert qc.BIG_POOL == {2.0, 4.0}
    assert qc.SMALL_POOL == {1.25, 1.35}
