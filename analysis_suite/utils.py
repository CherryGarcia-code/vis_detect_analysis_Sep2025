"""Shared computation utilities for the analysis suite.

Population tensor builders, smoothing, statistical helpers.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu

# Ensure visdetect is importable
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from visdetect.analysis.align import align_spikes_to_events, get_event_times_by_trial

from config import DEFAULT_BIN_SIZE, DEFAULT_ANALYSIS_WINDOW


# ── Population tensor builder ─────────────────────────────────────────

def build_population_tensor(
    session,
    cluster_ids: List[int],
    event_name: str = "Change_ON",
    window: Tuple[float, float] = DEFAULT_ANALYSIS_WINDOW,
    bin_size: float = DEFAULT_BIN_SIZE,
    outcome_filter: Optional[set] = None,
    trial_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Build a (n_trials x n_bins x n_units) population activity tensor.

    Parameters
    ----------
    session : Session
        The session object.
    cluster_ids : list of int
        Which clusters to include.
    event_name : str
        Event to align to (e.g. 'Change_ON', 'Baseline_ON').
    window : tuple of float
        (start, end) in seconds relative to event.
    bin_size : float
        Bin width in seconds.
    outcome_filter : set of str, optional
        If provided, only include trials with trialoutcome in this set.
    trial_indices : list of int, optional
        If provided, only include these trial indices.

    Returns
    -------
    tensor : np.ndarray, shape (n_trials, n_bins, n_units)
        Firing rates in Hz.
    bin_centers : np.ndarray, shape (n_bins,)
    used_trial_indices : list of int
        Which trial indices were actually used.
    """
    trials = getattr(session, "trials", []) or []
    n_trials = len(trials)

    # Get per-trial event times
    event_times = get_event_times_by_trial(session, event_name)

    # Determine which trials to use
    valid_indices = []
    for i in range(n_trials):
        if trial_indices is not None and i not in trial_indices:
            continue
        if outcome_filter is not None:
            outcome = getattr(trials[i], "trialoutcome", None)
            if outcome not in outcome_filter:
                continue
        if i < len(event_times) and np.isfinite(event_times[i]):
            valid_indices.append(i)

    if len(valid_indices) == 0 or len(cluster_ids) == 0:
        n_bins = int(np.round((window[1] - window[0]) / bin_size))
        return (
            np.empty((0, n_bins, len(cluster_ids))),
            np.linspace(window[0] + bin_size / 2, window[1] - bin_size / 2, n_bins),
            [],
        )

    # Collect event times for valid trials
    valid_event_times = [float(event_times[i]) for i in valid_indices]

    # Build cluster lookup
    cluster_map = {int(c.cluster_id): c for c in session.clusters}

    # Build tensor: align each cluster's spikes
    unit_matrices = []
    for cid in cluster_ids:
        c = cluster_map.get(int(cid))
        if c is None:
            # Missing cluster: fill with zeros
            n_bins = int(np.round((window[1] - window[0]) / bin_size))
            unit_matrices.append(np.zeros((len(valid_indices), n_bins)))
            continue
        mat, bin_centers = align_spikes_to_events(
            c.spike_times, valid_event_times, window=window, bin_size=bin_size
        )
        unit_matrices.append(mat)

    # Stack: each mat is (n_trials, n_bins) -> tensor is (n_trials, n_bins, n_units)
    tensor = np.stack(unit_matrices, axis=2)

    return tensor, bin_centers, valid_indices


# ── Smoothing ─────────────────────────────────────────────────────────

def smooth_psth(psth: np.ndarray, bin_size: float, sigma_ms: float = 25.0) -> np.ndarray:
    """Gaussian-smooth a PSTH array (1D or 2D along axis=1)."""
    sigma_bins = (sigma_ms / 1000.0) / bin_size
    if psth.ndim == 1:
        return gaussian_filter1d(psth, sigma=sigma_bins)
    elif psth.ndim == 2:
        return gaussian_filter1d(psth, sigma=sigma_bins, axis=1)
    return psth


def compute_zscore_normalized(
    tensor: np.ndarray, bin_centers: np.ndarray, baseline_window: Tuple[float, float]
) -> np.ndarray:
    """Z-score normalize each unit's activity against baseline period.

    tensor: (n_trials, n_bins, n_units)
    Returns same shape, z-scored per unit.
    """
    mask = (bin_centers >= baseline_window[0]) & (bin_centers < baseline_window[1])
    if mask.sum() == 0:
        return tensor
    baseline = tensor[:, mask, :]  # (n_trials, n_baseline_bins, n_units)
    # Compute per-unit baseline stats across trials and baseline bins
    mu = np.nanmean(baseline, axis=(0, 1), keepdims=True)  # (1, 1, n_units)
    sd = np.nanstd(baseline, axis=(0, 1), keepdims=True)
    sd[sd == 0] = 1.0  # avoid division by zero
    return (tensor - mu) / sd


def compute_baseline_subtracted(
    tensor: np.ndarray, bin_centers: np.ndarray, baseline_window: Tuple[float, float]
) -> np.ndarray:
    """Subtract per-unit baseline mean from population tensor."""
    mask = (bin_centers >= baseline_window[0]) & (bin_centers < baseline_window[1])
    if mask.sum() == 0:
        return tensor
    baseline = tensor[:, mask, :]
    mu = np.nanmean(baseline, axis=(0, 1), keepdims=True)
    return tensor - mu


# ── Good cluster selection ────────────────────────────────────────────

def get_good_cluster_ids(session, min_rate_hz: float = 1.0) -> List[int]:
    """Return cluster IDs passing basic QC: in good_cluster_ids and firing rate >= min_rate.

    Falls back to good_and_stable_ids > good_cluster_ids > all clusters.
    """
    # Get candidate list
    if getattr(session, "good_and_stable_ids", None):
        candidates = set(int(x) for x in session.good_and_stable_ids)
    elif getattr(session, "good_cluster_ids", None):
        candidates = set(int(x) for x in session.good_cluster_ids)
    else:
        candidates = {int(c.cluster_id) for c in session.clusters}

    # Estimate session duration from spike times
    all_spike_times = []
    for c in session.clusters:
        if len(c.spike_times) > 0:
            all_spike_times.extend([c.spike_times[0], c.spike_times[-1]])
    if all_spike_times:
        duration = max(all_spike_times) - min(all_spike_times)
    else:
        duration = 1.0  # fallback

    good_ids = []
    for c in session.clusters:
        cid = int(c.cluster_id)
        if cid not in candidates:
            continue
        n_spikes = len(c.spike_times)
        rate = n_spikes / max(duration, 1.0)
        if rate >= min_rate_hz:
            good_ids.append(cid)

    return sorted(good_ids)


# ── Statistical helpers ───────────────────────────────────────────────

def bootstrap_ci(
    data: np.ndarray,
    statistic=np.mean,
    n_boot: int = 1000,
    ci: float = 0.95,
    rng=None,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval.

    Returns (point_estimate, ci_low, ci_high).
    """
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return (np.nan, np.nan, np.nan)
    if rng is None:
        rng = np.random.default_rng(42)
    point = float(statistic(data))
    boot_stats = np.array(
        [statistic(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    )
    alpha = (1 - ci) / 2
    ci_low = float(np.percentile(boot_stats, 100 * alpha))
    ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha)))
    return (point, ci_low, ci_high)


def permutation_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_perm: int = 1000,
    statistic: str = "mean_diff",
    rng=None,
) -> Tuple[float, float]:
    """Two-sided permutation test.

    Returns (observed_stat, p_value).
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return (np.nan, 1.0)
    if rng is None:
        rng = np.random.default_rng(42)

    if statistic == "mean_diff":
        obs = float(np.mean(a) - np.mean(b))
        combined = np.concatenate([a, b])
        n_a = len(a)
        null_stats = np.empty(n_perm)
        for i in range(n_perm):
            rng.shuffle(combined)
            null_stats[i] = np.mean(combined[:n_a]) - np.mean(combined[n_a:])
        p = float((np.sum(np.abs(null_stats) >= abs(obs)) + 1) / (n_perm + 1))
        return (obs, p)

    raise ValueError(f"Unknown statistic: {statistic}")


def fdr_correct(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Returns boolean array of which hypotheses are rejected.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=bool)
    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    thresholds = alpha * np.arange(1, n + 1) / n
    reject_sorted = sorted_p <= thresholds
    # Make monotone: if reject[i] is True, all j < i should also be True
    if reject_sorted.any():
        max_reject = np.max(np.where(reject_sorted)[0])
        reject_sorted[: max_reject + 1] = True
    reject = np.zeros(n, dtype=bool)
    reject[sorted_idx] = reject_sorted
    return reject


def compute_auroc(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute area under ROC curve (Mann-Whitney U / n1*n2).

    Values > 0.5 mean group_a > group_b on average.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    try:
        u, _ = mannwhitneyu(a, b, alternative="two-sided")
        return float(u / (len(a) * len(b)))
    except Exception:
        return np.nan
