"""Adapter to attach Kilosort / Phy outputs to our Session dataclass.

Functions here read standard Kilosort/Phy files (templates.npy, spike_templates.npy,
spike_clusters.npy, spike_times.npy) and compute a mean waveform per cluster
which is attached as `mean_waveform` attribute on each `Cluster` object.

This is intentionally conservative and will not attempt to read raw continuous
data. It handles common variations in file names and is robust to missing
files.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd


def attach_kilosort_waveforms(session: Any, kilosort_dir: str, sample_rate: Optional[float] = None) -> pd.DataFrame:
    """Attach mean waveforms to clusters in-place and return a summary DataFrame.

    Args:
        session: Session dataclass instance (from src.session_io.load_session).
        kilosort_dir: Path to the Kilosort/Phy result folder (where templates.npy etc live).
        sample_rate: optional sampling rate (Hz) if you want to convert spike times.

    Returns:
        DataFrame with one row per cluster containing columns: cluster_id, n_spikes, mean_amp, waveform_shape, kilosort_present

    Notes:
    - The function looks for files in the folder: templates.npy, spike_templates.npy,
      spike_clusters.npy or spike_times.npy. Many installations use variations of
      these names; the logic below tries common variants.
    - If templates are not available, no waveforms will be attached.
    """
    d = Path(kilosort_dir)
    if not d.exists():
        raise FileNotFoundError(f'Kilosort folder not found: {kilosort_dir}')

    # Common filenames
    templates_paths = [d / 'templates.npy', d / 'templates.waveforms.npy']
    spike_templates_paths = [d / 'spike_templates.npy', d / 'spike_template.npy']
    spike_clusters_paths = [d / 'spike_clusters.npy', d / 'spike_cluster.npy', d / 'spike_clusters_ks.npy']
    spike_times_paths = [d / 'spike_times.npy', d / 'spike_time.npy']

    templates_file = next((p for p in templates_paths if p.exists()), None)
    spike_templates_file = next((p for p in spike_templates_paths if p.exists()), None)
    spike_clusters_file = next((p for p in spike_clusters_paths if p.exists()), None)
    spike_times_file = next((p for p in spike_times_paths if p.exists()), None)

    templates = None
    spike_templates = None
    spike_clusters = None
    spike_times = None

    if templates_file is not None:
        try:
            templates = np.load(str(templates_file), allow_pickle=False)
        except Exception:
            templates = np.load(str(templates_file), allow_pickle=True)

    if spike_templates_file is not None:
        spike_templates = np.load(str(spike_templates_file), allow_pickle=False)
    if spike_clusters_file is not None:
        spike_clusters = np.load(str(spike_clusters_file), allow_pickle=False)
    if spike_times_file is not None:
        spike_times = np.load(str(spike_times_file), allow_pickle=False)

    # Map cluster_id -> summary
    rows = []
    # Make an index: cluster_id -> Cluster object in session
    cluster_map = {int(c.cluster_id): c for c in session.clusters}

    # If we have spike_clusters, iterate over unique cluster ids
    if spike_clusters is not None:
        unique_clusters = np.unique(spike_clusters)
    else:
        # Fall back: use session.good_cluster_ids or session.clusters
        unique_clusters = [int(c.cluster_id) for c in session.clusters]

    # If we have templates and spike_templates, compute per-cluster mean waveform
    if templates is not None and spike_templates is not None:
        # templates: n_templates x n_channels x n_samples
        # spike_templates: n_spikes array of template idx per spike
        # spike_clusters: n_spikes array of cluster id per spike (optional)
        for cid in unique_clusters:
            mask = None
            if spike_clusters is not None:
                mask = (spike_clusters == cid)
            else:
                # If no spike_clusters, try to find templates that map to a cluster id equal to cid
                mask = np.zeros(spike_templates.shape, dtype=bool)
            n_spikes = int(np.sum(mask))
            mean_wf = None
            if n_spikes > 0:
                tmpl_idx = np.unique(spike_templates[mask])
                # average the corresponding template waveforms
                wf_list = []
                for t in tmpl_idx:
                    if 0 <= int(t) < templates.shape[0]:
                        wf_list.append(templates[int(t)])
                if len(wf_list) > 0:
                    mean_wf = np.stack(wf_list, axis=0).mean(axis=0)
            # attach to cluster object if present
            if int(cid) in cluster_map:
                cluster = cluster_map[int(cid)]
                if mean_wf is not None:
                    setattr(cluster, 'mean_waveform', mean_wf)
                setattr(cluster, 'kilosort_present', True)
            rows.append({'cluster_id': int(cid), 'n_spikes': n_spikes, 'mean_amp': float(np.nan if mean_wf is None else np.abs(mean_wf).max()), 'waveform_shape': None if mean_wf is None else mean_wf.shape, 'kilosort_present': True})
    elif templates is not None:
        # If we only have templates, attempt to assign them by index ordering
        for idx in range(templates.shape[0]):
            mean_wf = templates[idx]
            # no clear cluster id mapping; attempt to map by order if counts match
            mapped_cid = None
            if idx < len(session.clusters):
                mapped_cid = int(session.clusters[idx].cluster_id)
                setattr(session.clusters[idx], 'mean_waveform', mean_wf)
                setattr(session.clusters[idx], 'kilosort_present', True)
                rows.append({'cluster_id': mapped_cid, 'n_spikes': 0, 'mean_amp': float(np.abs(mean_wf).max()), 'waveform_shape': mean_wf.shape, 'kilosort_present': True})
            else:
                rows.append({'cluster_id': -1, 'n_spikes': 0, 'mean_amp': float(np.abs(mean_wf).max()), 'waveform_shape': mean_wf.shape, 'kilosort_present': True})
    else:
        # No template info - just report counts if spike_clusters is present
        if spike_clusters is not None:
            for cid in unique_clusters:
                n_spikes = int(np.sum(spike_clusters == cid))
                if int(cid) in cluster_map:
                    setattr(cluster_map[int(cid)], 'kilosort_present', True)
                rows.append({'cluster_id': int(cid), 'n_spikes': n_spikes, 'mean_amp': np.nan, 'waveform_shape': None, 'kilosort_present': True})
        else:
            # nothing found
            for c in session.clusters:
                rows.append({'cluster_id': int(c.cluster_id), 'n_spikes': 0, 'mean_amp': np.nan, 'waveform_shape': None, 'kilosort_present': False})

    df = pd.DataFrame(rows).set_index('cluster_id')
    return df
