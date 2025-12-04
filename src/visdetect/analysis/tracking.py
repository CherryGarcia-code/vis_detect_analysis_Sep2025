"""Utilities to attempt tracking the same neurons across sessions.

This module provides helper functions and data structures to compute pairwise
unit similarity (waveforms, firing statistics) and to propose matches across
sessions. Implementations are stubs and will need project-specific thresholds
and gating logic.
"""

from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path
import pandas as pd
import numpy as np
import logging

try:  # Progress meter is optional; degrade gracefully if tqdm unavailable
    from tqdm import tqdm
except ImportError:  # pragma: no cover - runtime optional
    tqdm = None

logger = logging.getLogger(__name__)


def _read_spikeglx_meta(meta_path: Path) -> Dict[str, str]:
    """Parse a SpikeGLX .meta file into a dictionary of string keys/values."""
    meta: Dict[str, str] = {}
    try:
        with open(meta_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                meta[key.strip()] = value.strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"SpikeGLX meta file not found: {meta_path}")
    return meta


def _meta_get_float(meta: Dict[str, str], *keys: str, default: Optional[float] = None) -> Optional[float]:
    for key in keys:
        if key in meta:
            try:
                return float(meta[key])
            except Exception:
                continue
    return default


def _infer_bit_volts(meta: Dict[str, str]) -> float:
    # Prefer explicit bit-volts entries if present
    bv = _meta_get_float(meta, "imBitVolts", default=None)
    if bv is not None:
        return bv
    # Fallback: derive from input range and max integer described in meta
    vmax = _meta_get_float(meta, "imAiRangeMax", default=0.5)
    vmin = _meta_get_float(meta, "imAiRangeMin", default=-0.5)
    span = (vmax or 0.5) - (vmin or -0.5)
    max_int = _meta_get_float(meta, "imMaxInt", default=2 ** 15)
    if max_int is None or max_int <= 0:
        max_int = 2 ** 15
    return span / (2 * max_int)


def compute_iti_windows(
    session,
    min_iti_duration: float = 0.5,
) -> np.ndarray:
    """Return an array of [start, end] ITI windows using Trial.ITI and Baseline_ON."""
    trials = getattr(session, "trials", []) or []
    ni_events = getattr(session, "ni_events", {}) or {}
    baseline = ni_events.get("Baseline_ON")
    if baseline is None:
        logger.warning("Session lacks Baseline_ON event times; cannot build ITI windows")
        return np.zeros((0, 2))
    if isinstance(baseline, dict):
        for key in ("rise_t", "times", "time", "t"):
            if key in baseline:
                baseline = baseline[key]
                break
        else:
            # Take the first value in the dict if no known key found
            baseline = next(iter(baseline.values()))
    baseline = np.array(baseline, dtype=object).flatten()
    n = min(len(trials), len(baseline))
    windows: List[Tuple[float, float]] = []
    for idx in range(n):
        trial = trials[idx]
        iti = getattr(trial, "ITI", None)
        if iti is None:
            continue
        try:
            iti_val = float(iti)
        except Exception:
            continue
        if not np.isfinite(iti_val) or iti_val < min_iti_duration:
            continue
        baseline_time = baseline[idx]
        try:
            baseline_time = float(baseline_time)
        except Exception:
            continue
        if not np.isfinite(baseline_time):
            continue
        start = baseline_time - iti_val
        end = baseline_time
        if start < 0 or end <= start:
            continue
        windows.append((start, end))
    if not windows:
        logger.warning("No valid ITI windows detected (min duration %.2fs)", min_iti_duration)
        return np.zeros((0, 2))
    windows_arr = np.array(sorted(windows, key=lambda w: w[0]), dtype=float)
    return windows_arr


def _mask_spikes_in_windows(spike_times_sec: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """Return boolean mask marking spikes that fall into any ITI window."""
    if spike_times_sec.size == 0 or windows.size == 0:
        return np.zeros(spike_times_sec.shape, dtype=bool)
    starts = windows[:, 0]
    ends = windows[:, 1]
    idx = np.searchsorted(starts, spike_times_sec, side="right") - 1
    valid = idx >= 0
    mask = np.zeros_like(spike_times_sec, dtype=bool)
    if np.any(valid):
        sel_idx = idx[valid]
        within = spike_times_sec[valid] < ends[sel_idx]
        mask[valid] = within
    return mask


def extract_iti_waveforms_from_raw(
    session,
    ks_dir: Path,
    raw_ap_path: Path,
    cluster_ids: Optional[List[int]] = None,
    max_spikes_per_unit: int = 500,
    min_spikes_per_unit: int = 80,
    min_spikes_per_half: int = 20,
    rng_seed: int = 0,
    session_label: Optional[str] = None,
    window_sampling: str = "all",
    max_windows: Optional[int] = None,
    show_progress: bool = True,
) -> Tuple[np.ndarray, List[int], Dict[int, Dict[str, float]]]:
    """Build (n_units, spike_w, n_ch, 2) ITI-only waveforms directly from the AP binary."""

    ks_dir = Path(ks_dir)
    raw_ap_path = Path(raw_ap_path)
    if not raw_ap_path.exists():
        raise FileNotFoundError(f"Raw AP file not found: {raw_ap_path}")
    meta_path = raw_ap_path.with_suffix(".meta")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found for raw AP data: {meta_path}")

    meta = _read_spikeglx_meta(meta_path)
    sample_rate = _meta_get_float(meta, "imSampRate", "sampleRate", default=30000.0) or 30000.0
    n_saved_chans = int(_meta_get_float(meta, "nSavedChans", "nChannels", default=384) or 384)
    bit_volts = _infer_bit_volts(meta)

    spike_times = np.load(ks_dir / "spike_times.npy").astype(np.int64)
    spike_clusters = np.load(ks_dir / "spike_clusters.npy").astype(int)
    templates = np.load(ks_dir / "templates.npy")
    channel_map = np.load(ks_dir / "channel_map.npy").astype(int)

    spike_w = templates.shape[1]
    n_channels = templates.shape[2]
    channel_map = channel_map[:n_channels]
    pre_samples = spike_w // 2
    post_samples = spike_w - pre_samples

    windows = compute_iti_windows(session)
    if window_sampling not in {"all", "uniform"}:
        raise ValueError(f"Unknown window_sampling mode: {window_sampling}")

    rng = np.random.default_rng(rng_seed)
    if window_sampling == "uniform" and max_windows is not None and len(windows) > max_windows:
        sel = rng.choice(len(windows), size=max_windows, replace=False)
        windows = np.sort(windows[sel], axis=0)

    spike_times_sec = spike_times / float(sample_rate)
    iti_mask = _mask_spikes_in_windows(spike_times_sec, windows)

    available_clusters = np.unique(spike_clusters)
    if cluster_ids is None:
        target_clusters = available_clusters.tolist()
    else:
        present = np.intersect1d(cluster_ids, available_clusters, assume_unique=False)
        target_clusters = present.tolist()

    raw = np.memmap(raw_ap_path, dtype=np.int16, mode="r")
    total_samples = raw.size // n_saved_chans
    raw = raw.reshape(total_samples, n_saved_chans)

    scale = bit_volts * 1e6  # convert to microvolts

    waveforms: List[np.ndarray] = []
    kept_clusters: List[int] = []
    diagnostics: Dict[int, Dict[str, float]] = {}

    iterator: Any = target_clusters
    use_progress = show_progress and tqdm is not None and len(target_clusters) > 1
    if use_progress:
        desc = f"{session_label or 'session'} ITI"
        iterator = tqdm(target_clusters, desc=desc, leave=False)

    for cid in iterator:
        idx = np.where((spike_clusters == cid) & iti_mask)[0]
        if idx.size == 0:
            continue
        samples = spike_times[idx]
        samples = samples[(samples >= pre_samples) & (samples < total_samples - post_samples)]
        if samples.size < min_spikes_per_unit:
            continue
        if samples.size > max_spikes_per_unit:
            samples = rng.choice(samples, size=max_spikes_per_unit, replace=False)
        rng.shuffle(samples)
        half = samples.size // 2
        if half < min_spikes_per_half:
            continue
        samples_first = np.sort(samples[:half])
        samples_second = np.sort(samples[half: 2 * half])
        if samples_second.size < min_spikes_per_half:
            continue

        def _mean_waveform(sample_list: np.ndarray) -> np.ndarray:
            acc: List[np.ndarray] = []
            for s in sample_list:
                start = int(s) - pre_samples
                end = start + spike_w
                snippet = raw[start:end, :][:, channel_map]
                acc.append(snippet.astype(np.float32))
            stacked = np.stack(acc, axis=0)
            return stacked.mean(axis=0) * scale

        wf_first = _mean_waveform(samples_first)
        wf_second = _mean_waveform(samples_second)
        stacked = np.stack([wf_first, wf_second], axis=-1)

        waveforms.append(stacked)
        kept_clusters.append(int(cid))
        diagnostics[int(cid)] = {
            "n_spikes_total": float(samples_first.size + samples_second.size),
            "n_windows": float(len(windows)),
        }

    if use_progress and hasattr(iterator, "close"):
        iterator.close()

    if not waveforms:
        raise RuntimeError("No ITI waveforms could be constructed from raw data")

    return np.stack(waveforms, axis=0), kept_clusters, diagnostics


def extract_iti_spikes(
    session,
    method: str = 'trial_field',
    fallback_window: Tuple[float, float] = (1.0, 3.0),
    min_iti_duration: float = 0.5
) -> Dict[int, np.ndarray]:
    """Extract spike masks for ITI (inter-trial interval) periods.
    
    The ITI period is defined as the window BEFORE each trial starts:
        ITI_start = Baseline_ON[i] - Trial.ITI[i]
        ITI_end = Baseline_ON[i]
    
    Args:
        session: Session object with trials, clusters, and ni_events
        method: 'trial_field' (use Trial.ITI before Baseline_ON), 'trial_boundaries' (compute from events),
                or 'fallback' (use fixed window before trial)
        fallback_window: (start, end) in seconds before Baseline_ON for fallback method
        min_iti_duration: Minimum ITI duration to consider valid (seconds)
    
    Returns:
        Dict mapping cluster_id -> boolean mask (True for spikes in ITI periods)
    """
    try:
        from .align import compute_true_reaction_time, get_event_times
    except ImportError:
        from align import compute_true_reaction_time, get_event_times
    
    trials = getattr(session, 'trials', [])
    clusters = getattr(session, 'clusters', [])
    ni_events = getattr(session, 'ni_events', {}) or {}
    
    if not trials:
        logger.warning("No trials found in session")
        return {}
    
    # Get Baseline_ON times - these mark the START of each trial
    baseline_times = []
    if 'Baseline_ON' in ni_events:
        baseline_on = ni_events['Baseline_ON']
        if isinstance(baseline_on, dict) and 'rise_t' in baseline_on:
            baseline_times = np.array(baseline_on['rise_t']).flatten()
        else:
            baseline_times = np.array(baseline_on).flatten()
    
    if len(baseline_times) == 0:
        logger.warning("No Baseline_ON times found, cannot extract ITI periods")
        return {}
    
    # There can be a mismatch between number of trials and number of NI events.
    # We need to find the closest Baseline_ON time for each trial.
    iti_windows = []
    
    # If there are no baseline times, we can't proceed.
    if len(baseline_times) == 0:
        logger.warning("No Baseline_ON times found, cannot extract ITI periods.")
        return {}

    for trial_idx, trial in enumerate(trials):
        # Anchor time for this trial. Since 'start_time' doesn't exist,
        # 'change_time' is the best available timestamp within the trial.
        trial_anchor_time = getattr(trial, 'change_time', None)
        if trial_anchor_time is None:
            logger.warning(f"Trial {trial_idx} is missing 'change_time' attribute. Skipping.")
            continue

        # Find the index of the minimum absolute difference to the anchor time
        closest_event_idx = np.argmin(np.abs(baseline_times - trial_anchor_time))
        baseline_on_time = baseline_times[closest_event_idx]
        
        time_diff = np.abs(baseline_on_time - trial_anchor_time)
        # A baseline event should occur *before* the change time.
        # We add a small tolerance (e.g., 10s) in case of edge cases.
        if (baseline_on_time > trial_anchor_time) and (time_diff > 10.0):
            logger.warning(f"Trial {trial_idx}: Closest Baseline_ON event is {time_diff:.2f}s *after* change_time. Skipping.")
            continue

        if np.isnan(baseline_on_time):
            continue
        
        # Determine ITI window based on method
        if method == 'trial_field':
            # Use Trial.ITI field - ITI is the period BEFORE Baseline_ON
            # ITI_start = Baseline_ON - Trial.ITI
            # ITI_end = Baseline_ON
            iti_dur = getattr(trial, 'ITI', None) if not isinstance(trial, dict) else trial.get('ITI', None)
            if iti_dur is not None and not np.isnan(iti_dur) and iti_dur >= min_iti_duration:
                iti_start = baseline_on_time - iti_dur
                iti_end = baseline_on_time
                if iti_start >= 0:  # Make sure we don't go negative
                    iti_windows.append((iti_start, iti_end))
        
        elif method == 'trial_boundaries':
            # This logic is complex and likely also needs re-evaluation
            logger.warning("Method 'trial_boundaries' is not robust to event count mismatches and is not recommended.")
            continue
        
        elif method == 'fallback':
            # Use fixed window before Baseline_ON
            iti_start = baseline_on_time - fallback_window[1]
            iti_end = baseline_on_time - fallback_window[0]
            if iti_start >= 0:
                iti_windows.append((iti_start, iti_end))
    
    logger.info(f"Extracted {len(iti_windows)} ITI windows using method '{method}'")
    
    # Create spike masks for each cluster
    iti_masks = {}
    for cluster in clusters:
        cluster_id = getattr(cluster, 'cluster_id', -1) if not isinstance(cluster, dict) else cluster.get('cluster_id', -1)
        spike_times = getattr(cluster, 'spike_times', np.array([])) if not isinstance(cluster, dict) else cluster.get('spike_times', np.array([]))
        
        if len(spike_times) == 0:
            iti_masks[cluster_id] = np.zeros(len(spike_times), dtype=bool)
            continue
        
        # Mark spikes within any ITI window
        mask = np.zeros(len(spike_times), dtype=bool)
        for iti_start, iti_end in iti_windows:
            in_window = (spike_times >= iti_start) & (spike_times < iti_end)
            mask |= in_window
        
        iti_masks[cluster_id] = mask
        n_iti_spikes = np.sum(mask)
        logger.debug(f"Cluster {cluster_id}: {n_iti_spikes}/{len(spike_times)} spikes in ITI ({100*n_iti_spikes/len(spike_times):.1f}%)")
    
    return iti_masks


def extract_waveforms_from_kilosort(
    session,
    ks_dir: Path,
    bc_dir: Optional[Path] = None,
    source: str = 'kilosort',
    use_iti_only: bool = False,
    iti_masks: Optional[Dict[int, np.ndarray]] = None,
    iti_method: str = 'trial_boundaries',
    fallback_window: Tuple[float, float] = (1.0, 3.0)
) -> Dict[str, np.ndarray]:
    """Extract waveforms from Kilosort templates or Bombcell outputs.
    
    Args:
        session: Session object with cluster data
        ks_dir: Path to Kilosort output directory
        bc_dir: Path to Bombcell output directory (required if source='bombcell' or 'both')
        source: 'kilosort', 'bombcell', or 'both'
        use_iti_only: If True, compute waveforms using only ITI spikes
        iti_masks: Pre-computed ITI masks (if None, will compute if use_iti_only=True)
        iti_method: Method for ITI extraction if iti_masks not provided
        fallback_window: Fallback window for ITI extraction
    
    Returns:
        Dict with keys 'kilosort' and/or 'bombcell', each containing
        (n_units, spike_w, n_channels, 2) array where last dim is [first_half, second_half]
        for cross-validation
    """
    ks_dir = Path(ks_dir)
    if bc_dir is not None:
        bc_dir = Path(bc_dir)
    
    if source not in ['kilosort', 'bombcell', 'both']:
        raise ValueError(f"Invalid source: {source}. Must be 'kilosort', 'bombcell', or 'both'")
    
    if source in ['bombcell', 'both'] and bc_dir is None:
        raise ValueError(f"bc_dir required when source='{source}'")
    
    # Get ITI masks if needed
    if use_iti_only and iti_masks is None:
        logger.info(f"Computing ITI masks using method '{iti_method}'")
        iti_masks = extract_iti_spikes(session, method=iti_method, fallback_window=fallback_window)
    
    result = {}
    
    # Extract from Kilosort
    if source in ['kilosort', 'both']:
        logger.info(f"Extracting waveforms from Kilosort: {ks_dir}")
        result['kilosort'] = _extract_from_kilosort_templates(
            session, ks_dir, use_iti_only, iti_masks
        )
    
    # Extract from Bombcell
    if source in ['bombcell', 'both']:
        logger.info(f"Extracting waveforms from Bombcell: {bc_dir}")
        result['bombcell'] = _extract_from_bombcell(
            session, bc_dir, use_iti_only, iti_masks
        )
    
    return result


def _extract_from_kilosort_templates(
    session,
    ks_dir: Path,
    use_iti_only: bool,
    iti_masks: Optional[Dict[int, np.ndarray]]
) -> np.ndarray:
    """Extract waveforms directly from Kilosort templates.
    
    Returns:
        (n_units, spike_w, n_channels, 2) array
    """
    # Load Kilosort files
    templates_file = ks_dir / "templates.npy"
    spike_templates_file = ks_dir / "spike_templates.npy"
    spike_clusters_file = ks_dir / "spike_clusters.npy"
    spike_times_file = ks_dir / "spike_times.npy"
    
    if not templates_file.exists():
        raise FileNotFoundError(f"Kilosort templates not found: {templates_file}")
    
    templates = np.load(templates_file)  # (n_templates, n_samples, n_channels)
    spike_templates = np.load(spike_templates_file).flatten()  # (n_spikes,)
    spike_clusters = np.load(spike_clusters_file).flatten()  # (n_spikes,)
    
    # Load spike times if ITI filtering needed
    ks_spike_times = None
    if use_iti_only and spike_times_file.exists():
        ks_spike_times = np.load(spike_times_file).flatten()  # (n_spikes,) in samples
        # Convert to seconds (assuming 30kHz sampling rate)
        ks_spike_times = ks_spike_times / 30000.0
        logger.info(f"Loaded Kilosort spike times: {len(ks_spike_times)} spikes")
    
    n_samples = templates.shape[1]
    n_channels = templates.shape[2]
    
    logger.info(f"Loaded Kilosort templates: {templates.shape}, spike_templates: {spike_templates.shape}")
    
    clusters = getattr(session, 'clusters', [])
    cluster_ids = [getattr(c, 'cluster_id', -1) if not isinstance(c, dict) else c.get('cluster_id', -1) for c in clusters]
    
    waveforms_list = []
    valid_cluster_ids = []
    
    logger.info(f"Processing {len(clusters)} clusters...")
    
    for idx, cluster in enumerate(clusters):
        if (idx + 1) % 50 == 0 or idx == 0:
            logger.info(f"  Progress: {idx + 1}/{len(clusters)} clusters processed")
        
        cluster_id = getattr(cluster, 'cluster_id', -1) if not isinstance(cluster, dict) else cluster.get('cluster_id', -1)
        quality = getattr(cluster, 'quality', None) if not isinstance(cluster, dict) else cluster.get('quality', None)
        
        # Filter to only 'good' or 'good_and_stable' quality clusters
        if quality not in ['good', 'good_and_stable']:
            continue
        
        # Get spike indices for this cluster  
        spike_idx = np.where(spike_clusters == cluster_id)[0]
        
        if len(spike_idx) == 0:
            logger.warning(f"No spikes found for cluster {cluster_id}")
            continue
        
        # Filter by ITI if requested
        if use_iti_only and iti_masks is not None and ks_spike_times is not None:
            if cluster_id in iti_masks:
                iti_mask = iti_masks[cluster_id]
                # Get session spike times for this cluster
                session_spike_times = getattr(cluster, 'spike_times', np.array([])) if not isinstance(cluster, dict) else cluster.get('spike_times', np.array([]))
                
                if len(session_spike_times) > 0 and len(iti_mask) == len(session_spike_times):
                    # Get ITI spike times from session
                    iti_spike_times = session_spike_times[iti_mask]
                    
                    if len(iti_spike_times) > 0:
                        # Find which Kilosort spike indices correspond to ITI times
                        # Get Kilosort spike times for this cluster
                        cluster_ks_times = ks_spike_times[spike_idx]
                        
                        # Match ITI spike times to Kilosort spike indices
                        # Use vectorized approach for efficiency
                        iti_spike_idx = []
                        n_iti = len(iti_spike_times)
                        n_matched = 0
                        max_diff = 0.0
                        
                        for iti_time in iti_spike_times:
                            time_diffs = np.abs(cluster_ks_times - iti_time)
                            min_diff = time_diffs.min()
                            max_diff = max(max_diff, min_diff)
                            
                            if min_diff < 0.01:  # Within 10ms tolerance (increased from 1ms)
                                min_diff_idx = np.argmin(time_diffs)
                                iti_spike_idx.append(spike_idx[min_diff_idx])
                                n_matched += 1
                        
                        if len(iti_spike_idx) > 0:
                            spike_idx = np.array(iti_spike_idx)
                            match_rate = 100 * n_matched / n_iti
                            logger.info(f"Cluster {cluster_id}: Using {len(spike_idx)}/{n_iti} ITI spikes ({match_rate:.1f}% matched, max_diff={max_diff*1000:.3f}ms)")
                        else:
                            logger.warning(f"Cluster {cluster_id}: No matching ITI spikes found (max_diff={max_diff*1000:.3f}ms), using all spikes")
                    else:
                        logger.warning(f"Cluster {cluster_id}: No ITI spikes in session, using all spikes")
                else:
                    logger.warning(f"Cluster {cluster_id}: ITI mask size mismatch, using all spikes")
            else:
                logger.warning(f"No ITI mask for cluster {cluster_id}, using all spikes")
        
        if len(spike_idx) < 10:
            logger.warning(f"Cluster {cluster_id}: only {len(spike_idx)} spikes, skipping")
            continue
        
        # Get templates for these spikes
        cluster_template_ids = spike_templates[spike_idx]
        
        # Split into two halves for cross-validation
        n_half = len(spike_idx) // 2
        first_half_templates = cluster_template_ids[:n_half]
        second_half_templates = cluster_template_ids[n_half:2*n_half]
        
        # Compute mean waveform for each half
        wf_first = templates[first_half_templates].mean(axis=0)  # (n_samples, n_channels)
        wf_second = templates[second_half_templates].mean(axis=0)
        
        # Stack as (n_samples, n_channels, 2)
        wf_combined = np.stack([wf_first, wf_second], axis=-1)
        
        waveforms_list.append(wf_combined)
        valid_cluster_ids.append(cluster_id)
    
    if len(waveforms_list) == 0:
        logger.error("No valid waveforms extracted from Kilosort")
        logger.error(f"Total clusters: {len(clusters)}, Valid (good quality): 0")
        return np.zeros((0, n_samples, n_channels, 2))
    
    waveforms = np.stack(waveforms_list, axis=0)  # (n_units, n_samples, n_channels, 2)
    logger.info(f"Extracted waveforms from Kilosort: {waveforms.shape}")
    logger.info(f"  Total clusters: {len(clusters)}, Valid (good quality): {len(valid_cluster_ids)} ({100*len(valid_cluster_ids)/len(clusters):.1f}%)")
    
    return waveforms


def _extract_from_bombcell(
    session,
    bc_dir: Path,
    use_iti_only: bool,
    iti_masks: Optional[Dict[int, np.ndarray]]
) -> np.ndarray:
    """Extract waveforms from Bombcell pre-computed waveforms.
    
    Returns:
        (n_units, spike_w, n_channels, 2) array
    """
    templates_file = bc_dir / "templates._bc_rawWaveforms.npy"
    
    if not templates_file.exists():
        raise FileNotFoundError(f"Bombcell waveforms not found: {templates_file}")
    
    # Load Bombcell waveforms
    # Expected shape from run_unitmatch_pair.py: loaded then reshaped to (n_units, spike_w, n_ch, 2)
    w = np.load(templates_file, allow_pickle=True)
    
    logger.info(f"Loaded Bombcell waveforms: {w.shape}")
    
    # Bombcell already provides cross-validation splits in the format we need
    # The existing code in run_unitmatch_pair.py line 140 shows this is the expected format
    if w.ndim == 4 and w.shape[-1] == 2:
        # Already in correct format: (n_units, spike_w, n_channels, 2)
        waveforms = w
    else:
        # May need reshaping - follow existing code pattern
        logger.warning(f"Bombcell waveforms have unexpected shape {w.shape}, attempting reshape")
        # This would need project-specific logic based on actual Bombcell output format
        waveforms = w
    
    logger.info(f"Extracted waveforms from Bombcell: {waveforms.shape}")
    
    # Note: ITI filtering for Bombcell waveforms would require re-computing from raw data
    # since Bombcell pre-computes waveforms. For now, log a warning if use_iti_only is requested.
    if use_iti_only:
        logger.warning("ITI filtering not supported for Bombcell pre-computed waveforms. Using all spikes.")
    
    return waveforms


def compute_unit_similarity(
    sess_a: Any, sess_b: Any, cluster_id_a: int, cluster_id_b: int
) -> Dict[str, float]:
    """Compute similarity metrics between two clusters across sessions.

    Returns a dict with waveform_corr, isi_ks, firing_rate_ratio, and a
    composite score.
    """
    # Placeholder: extract waveforms and compute correlation; compute ISI stats
    return {
        "waveform_corr": 0.0,
        "isi_ks": 1.0,
        "firing_rate_ratio": 1.0,
        "composite_score": 0.0,
    }


def propose_matches(sess_a: Any, sess_b: Any, top_k: int = 5) -> pd.DataFrame:
    """Return a DataFrame listing top-k candidate matches between sessions.

    Columns: cluster_a, cluster_b, composite_score, waveform_corr, isi_ks, firing_rate_ratio
    """
    # Placeholder: iterate over possible cluster pairs and compute similarity
    rows = []
    # Prefer good_and_stable_ids, then good_cluster_ids, else all clusters
    if getattr(sess_a, "good_and_stable_ids", None):
        clusters_a = list(sess_a.good_and_stable_ids)
    elif getattr(sess_a, "good_cluster_ids", None):
        clusters_a = list(sess_a.good_cluster_ids)
    else:
        clusters_a = [c.cluster_id for c in sess_a.clusters]
    if getattr(sess_b, "good_and_stable_ids", None):
        clusters_b = list(sess_b.good_and_stable_ids)
    elif getattr(sess_b, "good_cluster_ids", None):
        clusters_b = list(sess_b.good_cluster_ids)
    else:
        clusters_b = [c.cluster_id for c in sess_b.clusters]
    for a in clusters_a:
        for b in clusters_b:
            rows.append(
                {
                    "cluster_a": int(a),
                    "cluster_b": int(b),
                    "composite_score": 0.0,
                    "waveform_corr": 0.0,
                    "isi_ks": 1.0,
                    "firing_rate_ratio": 1.0,
                }
            )
    df = pd.DataFrame(rows)
    return df.sort_values("composite_score", ascending=False).head(top_k)
