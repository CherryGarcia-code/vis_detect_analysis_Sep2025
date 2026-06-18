# src/visdetect/anatomy/peak_channel.py
"""Per-unit peak channel. RawWaveforms primary (reuses tracking_qc), KS templates
fallback. See spec §3/§6."""
from __future__ import annotations

from typing import Optional

import numpy as np

from visdetect.analysis.tracking_qc import extract_peak_channel, load_raw_mean_waveform


def peak_channel_from_mean(mean_waveform: np.ndarray) -> int:
    return int(extract_peak_channel(np.asarray(mean_waveform)))


def unit_peak_channel(raw_wf_root, session_name: str, cluster_id: int) -> Optional[int]:
    mw = load_raw_mean_waveform(raw_wf_root, session_name, cluster_id)
    if mw is None:
        return None
    return peak_channel_from_mean(mw)
