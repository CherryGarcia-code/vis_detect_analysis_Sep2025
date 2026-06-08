"""Precision curation of UnitMatch cross-session tracks.

Expert->Naive backward sweep over the liberal UM registry: biophysical gate +
availability-gated in-zone functional corroborator, rolling anchor with
gap-bridge tolerance. Never alters the original registry. See
docs/superpowers/specs/2026-06-07-track-curation-design.md.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from visdetect.analysis.tracking_qc import isi_log_histogram


def partitioned_isi_hists(spike_times: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Two log-ISI histograms from disjoint spike partitions (even/odd index).

    The curation ISI feature uses one partition; validation uses the other, so
    ISI validation is statistically independent of the ISI curation feature
    (spec sec 8.1). Both estimate the same stationary fingerprint.

    Returns (curation_hist, holdout_hist), each shape (50,); all-NaN if a
    partition has too few spikes.
    """
    st = np.asarray(spike_times, dtype=float)
    st = np.sort(st)
    cur = st[0::2]
    hold = st[1::2]
    cur_h, _ = isi_log_histogram(cur)
    hold_h, _ = isi_log_histogram(hold)
    return cur_h, hold_h
