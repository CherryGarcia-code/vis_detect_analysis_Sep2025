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


from dataclasses import dataclass, field
from typing import Dict, List, Optional

from visdetect.analysis.tracking_qc import (
    load_raw_mean_waveform, extract_peak_channel, extract_footprint,
    extract_unit_psths,
)


@dataclass
class CurationFeature:
    session_name: str
    ks_unit_id: int
    stage: str
    waveform_peak: np.ndarray
    footprint: np.ndarray
    footprint_channels: np.ndarray
    peak_chan: int
    peak_depth_um: float
    peak_depth_corrected_um: float
    baseline_fr_hz: float
    isi_hist_curation: np.ndarray
    isi_hist_holdout: np.ndarray
    inzone_psths: Dict[str, Optional[np.ndarray]]
    n_inzone_trials: int


def _baseline_fr(cluster, session) -> float:
    st = np.asarray(cluster.spike_times, dtype=float)
    if st.size == 0:
        return 0.0
    dur = float(st.max() - st.min())
    return float(st.size / dur) if dur > 0 else 0.0


def extract_curation_feature(session, ks_unit_id: int, session_name: str,
                             stage: str, raw_wf_root,
                             channel_positions: Optional[np.ndarray],
                             in_zone_idx: List[int],
                             drift_offset: float = 0.0,
                             ) -> Optional[CurationFeature]:
    """Assemble a CurationFeature for one (session, uid). None if no waveform."""
    cluster_map = {c.cluster_id: c for c in session.clusters}
    cluster = cluster_map.get(int(ks_unit_id))
    if cluster is None:
        return None

    mean_wf = load_raw_mean_waveform(raw_wf_root, session_name, int(ks_unit_id))
    if mean_wf is None:
        return None
    peak_chan = extract_peak_channel(mean_wf)
    peak_wave = mean_wf[:, peak_chan]
    footprint, fp_chans = extract_footprint(mean_wf, peak_chan)

    if channel_positions is not None and peak_chan < channel_positions.shape[0]:
        depth_um = float(channel_positions[peak_chan, 1])
    else:
        depth_um = float("nan")
    depth_corr = depth_um - float(drift_offset) if np.isfinite(depth_um) else float("nan")

    cur_h, hold_h = partitioned_isi_hists(np.asarray(cluster.spike_times))

    in_zone_set = set(int(i) for i in in_zone_idx)
    psth_dict = extract_unit_psths(session, int(ks_unit_id),
                                   restrict_trials=in_zone_set)
    inzone_psths = {k: v[0] for k, v in psth_dict.items()}

    return CurationFeature(
        session_name=session_name, ks_unit_id=int(ks_unit_id), stage=stage,
        waveform_peak=peak_wave.astype(np.float32),
        footprint=footprint.astype(np.float32), footprint_channels=fp_chans,
        peak_chan=peak_chan, peak_depth_um=depth_um,
        peak_depth_corrected_um=depth_corr,
        baseline_fr_hz=_baseline_fr(cluster, session),
        isi_hist_curation=cur_h, isi_hist_holdout=hold_h,
        inzone_psths=inzone_psths, n_inzone_trials=len(in_zone_set),
    )
