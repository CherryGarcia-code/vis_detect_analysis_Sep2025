# tests/anatomy/test_peak_channel.py
import numpy as np
from pathlib import Path
from visdetect.anatomy.peak_channel import peak_channel_from_mean, unit_peak_channel

def test_peak_channel_from_mean_known():
    mw = np.zeros((82, 10))             # (samples, channels)
    mw[40, 7] = -5.0; mw[50, 7] = 4.0   # biggest peak-to-peak on channel 7
    assert peak_channel_from_mean(mw) == 7

def test_unit_peak_channel_reads_rawwaveform(tmp_path):
    sess = tmp_path / "01072025" / "RawWaveforms"
    sess.mkdir(parents=True)
    raw = np.zeros((82, 10, 2))
    raw[40, 3, :] = -6.0; raw[50, 3, :] = 5.0   # channel 3 dominant
    np.save(sess / "Unit42_RawSpikes.npy", raw)
    pc = unit_peak_channel(tmp_path, "01072025", 42)
    assert pc == 3

def test_unit_peak_channel_missing_returns_none(tmp_path):
    assert unit_peak_channel(tmp_path, "01072025", 999) is None
