"""Pure helpers converting visdetect-extracted unit data into DANT's input conventions.

No file I/O here — these operate on arrays so they are unit-testable. Orchestration
(reading RawWaveforms/pkls, writing the DANT input folder) lives in build_dant_inputs.py.
"""
import numpy as np


def collapse_cv(raw_spikes):
    """RawWaveforms (n_samp, n_ch, n_cv=2) -> DANT waveform (n_ch, n_samp).

    Averages the two cross-validation halves, then transposes so the channel axis
    is first (DANT's waveform_all is (n_unit, n_channel, n_sample)).
    """
    arr = np.asarray(raw_spikes, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(f"expected (n_samp, n_ch, 2), got shape {arr.shape}")
    mean_wave = arr.mean(axis=2)        # (n_samp, n_ch)
    return mean_wave.T                  # (n_ch, n_samp)


def derive_channel_shanks(channel_positions, gap_um=150.0):
    """(n_ch, 2) x/y positions -> (n_ch,) 0-based shank id, grouping x by gaps."""
    pos = np.asarray(channel_positions, dtype=float)
    x = pos[:, 0]
    ux = np.unique(x)
    shank_of_ux = np.zeros(len(ux), dtype=np.int64)
    cur = 0
    for i in range(1, len(ux)):
        if ux[i] - ux[i - 1] > gap_um:
            cur += 1
        shank_of_ux[i] = cur
    mapping = {val: s for val, s in zip(ux, shank_of_ux)}
    return np.array([mapping[v] for v in x], dtype=np.int64)


def seconds_to_ms(spike_times):
    """Spike times in seconds -> milliseconds (DANT ACG/ISI bins are in ms)."""
    return np.asarray(spike_times, dtype=np.float64) * 1000.0


def is_positive_going(waveform):
    """True if the peak channel's waveform is positive-going (|max| > |min|).

    waveform: (n_ch, n_samp). DANT's trough-centering assumes negative-going spikes,
    so positive-going units should be excluded before centering.
    """
    w = np.asarray(waveform, dtype=float)
    ptp = w.max(axis=1) - w.min(axis=1)
    peak = int(np.argmax(ptp))
    return abs(float(w[peak].max())) > abs(float(w[peak].min()))
