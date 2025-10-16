import numpy as np
from typing import Dict, Any, Tuple


def align_spikes_to_onsets(spike_times: np.ndarray, onsets_s: np.ndarray, window: Tuple[float, float]):
    """Align spike times (in seconds) to a set of onsets.

    Returns a 2D ragged list: list of arrays of spike times relative to each onset.
    """
    aligned = []
    start, end = window
    for t0 in onsets_s:
        lo = t0 + start
        hi = t0 + end
        mask = (spike_times >= lo) & (spike_times <= hi)
        rel = spike_times[mask] - t0
        aligned.append(rel)
    return aligned


def compute_unit_metrics(aligned_spike_lists, spike_count_baseline_window=(-0.1, 0.0), evoked_window=(0.0, 0.01)):
    """Compute simple optotagging metrics per unit.

    aligned_spike_lists: list of arrays (per-pulse relative spike times)
    Returns dict: n_pulses, mean_latency_ms, reliability, evoked_count_mean, evoked_count_std
    """
    n_pulses = len(aligned_spike_lists)
    # latency: for pulses with at least one spike in evoked window, take first spike time
    latencies = []
    evoked_counts = []
    baseline_counts = []
    ev_lo, ev_hi = evoked_window
    b_lo, b_hi = spike_count_baseline_window
    for rel in aligned_spike_lists:
        # count spikes in windows
        ev_mask = (rel >= ev_lo) & (rel <= ev_hi)
        b_mask = (rel >= b_lo) & (rel < b_hi)
        evoked_counts.append(ev_mask.sum())
        baseline_counts.append(b_mask.sum())
        if ev_mask.any():
            latencies.append(rel[ev_mask].min())
    reliability = np.mean(np.array(evoked_counts) > 0) if n_pulses > 0 else 0.0
    mean_latency_ms = np.mean(latencies) * 1000.0 if len(latencies) > 0 else np.nan
    return {
        'n_pulses': n_pulses,
        'mean_latency_ms': mean_latency_ms,
        'reliability': reliability,
        'evoked_count_mean': float(np.mean(evoked_counts)) if n_pulses>0 else 0.0,
        'evoked_count_std': float(np.std(evoked_counts)) if n_pulses>0 else 0.0,
        'baseline_count_mean': float(np.mean(baseline_counts)) if n_pulses>0 else 0.0,
    }


def compute_psth(aligned_spike_lists, bin_width=0.001, window=(-0.1, 0.05)):
    """Compute PSTH (mean rate across trials) from aligned spike lists.
    Returns bin_centers (s) and rate (Hz).
    """
    start, end = window
    bins = np.arange(start, end + 1e-12, bin_width)
    counts = np.zeros(len(bins)-1, dtype=float)
    n_trials = len(aligned_spike_lists)
    for rel in aligned_spike_lists:
        # histogram of rel spikes
        h, _ = np.histogram(rel, bins=bins)
        counts += h
    # average per trial and convert to Hz
    rates = counts / float(max(1, n_trials)) / bin_width
    centers = (bins[:-1] + bins[1:]) / 2.0
    return centers, rates


def compute_waveform_metrics(waveform: np.ndarray, wf_sample_rate: float = 30000.0) -> Dict[str, float]:
    """Compute waveform-derived metrics.

    waveform: 1D (n_timepoints,) or 2D (n_channels, n_timepoints).
    wf_sample_rate: samples per second.

    Returns dict with keys: peak_width_us, peak_valley_us, peak_idx, valley_idx
    """
    # normalize waveform shape
    w = np.asarray(waveform)
    if w.ndim == 2:
        # assume shape (n_channels, n_timepoints) -> pick channel with largest peak-to-valley range
        ranges = w.max(axis=1) - w.min(axis=1)
        ch = int(np.argmax(ranges))
        w1 = w[ch, :]
    elif w.ndim == 1:
        w1 = w
    else:
        raise ValueError('waveform must be 1D or 2D array')

    # find peak (max) and valley (min)
    peak_idx = int(np.argmax(w1))
    valley_idx = int(np.argmin(w1))
    peak_val = float(w1[peak_idx])
    half_amp = peak_val / 2.0

    # find left and right indices where waveform crosses half amplitude (for FWHM)
    left_idx = peak_idx
    while left_idx > 0 and w1[left_idx] > half_amp:
        left_idx -= 1
    right_idx = peak_idx
    while right_idx < len(w1) - 1 and w1[right_idx] > half_amp:
        right_idx += 1

    fwhm_samples = max(0, right_idx - left_idx)
    peak_width_us = (fwhm_samples / float(wf_sample_rate)) * 1e6

    peak_valley_us = (abs(valley_idx - peak_idx) / float(wf_sample_rate)) * 1e6

    return {
        'peak_width_us': float(peak_width_us),
        'peak_valley_us': float(peak_valley_us),
        'peak_idx': peak_idx,
        'valley_idx': valley_idx,
    }


def poisson_p_value(total_evoked_count, baseline_rate_hz, n_pulses, evoked_window_s):
    """One-sided Poisson test p-value for observing >= total_evoked_count given expected lambda.
    lambda = baseline_rate_hz * n_pulses * evoked_window_s
    Uses survival function.
    """
    try:
        from scipy.stats import poisson
        lam = baseline_rate_hz * n_pulses * evoked_window_s
        # survival function gives P[X > k-1] = P[X >= k]
        p = poisson.sf(total_evoked_count - 1, lam)
        return float(p)
    except Exception:
        # fallback using numpy sums (approx)
        lam = baseline_rate_hz * n_pulses * evoked_window_s
        # compute p = sum_{k>=total} e^-lam lam^k / k!
        # approximate with tail using Poisson cdf via numpy (may be slow)
        # return 1.0 if lam is nan or zero
        if lam <= 0:
            return 1.0
        # use naive summation up to total_evoked_count
        from math import exp, factorial
        cdf = 0.0
        for k in range(0, max(0, int(total_evoked_count))):
            cdf += exp(-lam) * (lam ** k) / factorial(k)
        return float(max(0.0, 1.0 - cdf))


def permutation_test_total_evoked(spike_times, n_pulses, evoked_window, recording_start, recording_end, n_iters=500):
    """Permutation test: sample n_pulses random onsets and compute total evoked spike counts repeatedly.
    Returns p-value (proportion >= observed) given observed total evoked count computed outside.
    """
    ev_lo, ev_hi = evoked_window
    # observed total must be computed by caller
    rng = np.random.default_rng(0)
    counts = np.zeros(n_iters, dtype=int)
    margin = max(0.1, ev_hi - ev_lo)
    start = recording_start + margin
    end = recording_end - margin
    if end <= start:
        return 1.0, counts
    # pre-sort spikes for fast counting
    spikes = np.sort(spike_times)
    for i in range(n_iters):
        rand_onsets = rng.uniform(start, end, size=n_pulses)
        total = 0
        # vectorized counting using searchsorted
        left = np.searchsorted(spikes, rand_onsets + ev_lo, side='left')
        right = np.searchsorted(spikes, rand_onsets + ev_hi, side='right')
        total = int(np.sum(right - left))
        counts[i] = total
    return counts


def classify_unit(metrics: Dict[str, Any], latency_thresh_ms=6.0, reliability_thresh=0.5):
    """Simple classification rules:
    - 'opto' if mean_latency_ms <= latency_thresh_ms and reliability >= reliability_thresh
    - otherwise 'none'
    """
    lat = metrics.get('mean_latency_ms', np.nan)
    rel = metrics.get('reliability', 0.0)
    if not np.isnan(lat) and (lat <= latency_thresh_ms) and (rel >= reliability_thresh):
        return 'opto'
    return 'none'


def build_units_table(spike_times_list, spike_clusters, pulses_onset_s, window=(-0.1, 0.05), latency_thresh_ms=6.0, reliability_thresh=0.5, waveforms: Dict[int, np.ndarray] = None, wf_sample_rate: float = 30000.0):
    """Given spike times (s) and cluster ids, compute metrics for each cluster.

    Returns a list of dicts (one per cluster) to avoid importing pandas.
    """
    unique_clusters = np.unique(spike_clusters)
    rows = []
    # recording extents from spike times
    rec_start = float(np.min(spike_times_list)) if spike_times_list.size>0 else 0.0
    rec_end = float(np.max(spike_times_list)) if spike_times_list.size>0 else np.max(pulses_onset_s) + 1.0
    for cid in unique_clusters:
        mask = spike_clusters == cid
        st = spike_times_list[mask]
        aligned = align_spikes_to_onsets(st, pulses_onset_s, window)
        metrics = compute_unit_metrics(aligned)
        # waveform metrics if provided
        if waveforms is not None and int(cid) in waveforms:
            wf = waveforms[int(cid)]
            wf_metrics = compute_waveform_metrics(wf, wf_sample_rate)
            # mean firing rate over recording
            rec_dur = max(1.0, rec_end - rec_start)
            mean_fr = float(len(st) / rec_dur)
            # waveform-based classification
            # SPN: peak_width >150us, peak-valley >500us, mean_fr <=10Hz
            # FSI: peak_width <=150us, peak-valley <=500us, mean_fr >=0.1Hz
            pw = wf_metrics['peak_width_us']
            pv = wf_metrics['peak_valley_us']
            if (pw > 150.0) and (pv > 500.0) and (mean_fr <= 10.0):
                wf_class = 'SPN'
            elif (pw <= 150.0) and (pv <= 500.0) and (mean_fr >= 0.1):
                wf_class = 'FSI'
            else:
                wf_class = 'Other'
            metrics.update({'wf_peak_width_us': pw, 'wf_peak_valley_us': pv, 'wf_mean_fr_hz': mean_fr, 'wf_class': wf_class})
        # latency percentiles
        latencies = []
        ev_lo, ev_hi = (0.0, 0.01)
        n_pulses = metrics['n_pulses']
        for rel in aligned:
            ev_mask = (rel >= ev_lo) & (rel <= ev_hi)
            if ev_mask.any():
                latencies.append(float(rel[ev_mask].min()))
        if len(latencies) > 0:
            p10 = float(np.percentile(latencies, 10) * 1000.0)
            p50 = float(np.percentile(latencies, 50) * 1000.0)
        else:
            p10 = float('nan')
            p50 = float('nan')
        # Poisson test
        baseline_rate = metrics['baseline_count_mean'] / (abs(window[0]) if abs(window[0])>0 else 0.1)
        total_evoked = int(sum(((rel >= ev_lo) & (rel <= ev_hi)).sum() for rel in aligned))
        evoked_window_len = ev_hi - ev_lo
        p_poiss = poisson_p_value(total_evoked, baseline_rate, n_pulses, evoked_window_len) if n_pulses>0 else 1.0
        # permutation test (sampled distribution)
        perm_counts = permutation_test_total_evoked(st, n_pulses, (ev_lo, ev_hi), rec_start, rec_end, n_iters=300)
        if isinstance(perm_counts, tuple):
            perm_counts = perm_counts[0]
        if len(perm_counts) > 0:
            p_perm = float((perm_counts >= total_evoked).sum() / float(len(perm_counts)))
        else:
            p_perm = 1.0
        metrics.update({'latency_p10_ms': p10, 'latency_p50_ms': p50, 'p_poisson': p_poiss, 'p_permutation': p_perm, 'baseline_rate_hz': baseline_rate})
        cls = classify_unit(metrics, latency_thresh_ms, reliability_thresh)
        row = {'cluster_id': int(cid), **metrics, 'classification': cls}
        rows.append(row)
    # sort by cluster_id
    rows = sorted(rows, key=lambda r: r['cluster_id'])
    return rows
