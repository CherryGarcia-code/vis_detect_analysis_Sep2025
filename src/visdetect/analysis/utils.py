"""Shared analysis utilities for the visdetect package.

This module contains reusable functions for population analysis, statistical
testing, and data processing used across the analysis suite.

Moved from analysis_suite/utils.py to centralize shared code in the library.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu

from visdetect.analysis.align import align_spikes_to_events, get_event_times_by_trial
from visdetect.analysis.constants import EVENT_VALID_OUTCOMES, DEFAULT_BIN_SIZE, DEFAULT_SIGMA_MS
from visdetect.analysis.config import DEFAULT_ANALYSIS_WINDOW


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
    outcome_filter : set, optional
        Only include trials with these outcomes.
    trial_indices : list of int, optional
        Only include these trial indices.

    Returns
    -------
    tensor : ndarray, shape (n_trials, n_bins, n_units)
        Population activity tensor.
    bin_centers : ndarray, shape (n_bins,)
        Time bin centers relative to event (seconds).
    valid_trials : list of int
        Trial indices that were included.
    """
    # Get per-trial event times (handles NI events, outcome filtering, NaN fill)
    per_trial_events = get_event_times_by_trial(
        session, event_name, enforce_valid_outcomes=True,
    )

    # Determine valid trial indices
    valid_outcomes_raw = EVENT_VALID_OUTCOMES.get(event_name, None)
    # Normalize to lowercase for case-insensitive comparison
    valid_outcomes = {o.lower() for o in valid_outcomes_raw} if valid_outcomes_raw else None
    valid_trials = []
    for i, trial in enumerate(session.trials):
        # Skip if event time is missing/NaN for this trial
        if i >= len(per_trial_events) or np.isnan(per_trial_events[i]):
            continue
        # Check outcome validity (case-insensitive)
        oc = (getattr(trial, "trialoutcome", None) or "").lower()
        if valid_outcomes and oc not in valid_outcomes:
            continue
        # Check caller's outcome filter (case-insensitive)
        if outcome_filter and oc not in {o.lower() for o in outcome_filter}:
            continue
        # Check trial index filter
        if trial_indices is not None and i not in trial_indices:
            continue
        valid_trials.append(i)

    if not valid_trials:
        raise ValueError(f"No valid trials found for event '{event_name}'")

    # Collect event times for valid trials only
    valid_event_times = [per_trial_events[i] for i in valid_trials]

    # Build cluster lookup
    cluster_map = {c.cluster_id: c for c in session.clusters}

    # Initialize outputs
    n_units = len(cluster_ids)
    bin_centers = None
    tensor = None

    # Fill tensor one unit at a time
    for unit_idx, cluster_id in enumerate(cluster_ids):
        cluster = cluster_map.get(cluster_id)
        if cluster is None:
            continue

        # align_spikes_to_events returns (trials_matrix, bin_centers)
        # trials_matrix shape: (n_trials, n_bins) — already in Hz
        trials_matrix, bc = align_spikes_to_events(
            cluster.spike_times, valid_event_times,
            window=window, bin_size=bin_size,
        )

        if tensor is None:
            bin_centers = bc
            tensor = np.zeros((len(valid_trials), len(bc), n_units))

        tensor[:, :, unit_idx] = trials_matrix

    # Fallback if no units produced output
    if tensor is None:
        n_bins = int((window[1] - window[0]) / bin_size)
        bin_centers = np.linspace(window[0] + bin_size / 2, window[1] - bin_size / 2, n_bins)
        tensor = np.zeros((len(valid_trials), n_bins, n_units))

    return tensor, bin_centers, valid_trials


# ── Signal processing ─────────────────────────────────────────────────

def smooth_psth(psth: np.ndarray, bin_size: float, sigma_ms: float = DEFAULT_SIGMA_MS) -> np.ndarray:
    """Apply Gaussian smoothing to PSTH data.

    Parameters
    ----------
    psth : ndarray
        PSTH data (any shape, smoothed along last axis).
    bin_size : float
        Bin size in seconds.
    sigma_ms : float
        Gaussian sigma in milliseconds.

    Returns
    -------
    ndarray
        Smoothed PSTH.
    """
    sigma_bins = (sigma_ms / 1000.0) / bin_size
    return gaussian_filter1d(psth, sigma=sigma_bins, axis=-1)


def compute_zscore_normalized(
    tensor: np.ndarray,
    bin_centers: np.ndarray,
    baseline_window: Tuple[float, float],
) -> np.ndarray:
    """Z-score normalize tensor using shared baseline across all conditions.

    Parameters
    ----------
    tensor : ndarray, shape (..., n_bins, n_units)
        Activity tensor to normalize.
    bin_centers : ndarray, shape (n_bins,)
        Time bin centers.
    baseline_window : tuple of float
        (start, end) baseline window in seconds.

    Returns
    -------
    ndarray
        Z-score normalized tensor.
    """
    baseline_mask = (bin_centers >= baseline_window[0]) & (bin_centers <= baseline_window[1])

    # Compute shared baseline stats across all conditions
    baseline_data = tensor[..., baseline_mask, :].reshape(-1, tensor.shape[-1])
    mu = np.mean(baseline_data, axis=0, keepdims=True)
    sigma = np.std(baseline_data, axis=0, keepdims=True)

    # Prevent division by zero
    sigma = np.maximum(sigma, 1e-6)

    # Normalize
    return (tensor - mu) / sigma


def compute_baseline_subtracted(
    tensor: np.ndarray,
    bin_centers: np.ndarray,
    baseline_window: Tuple[float, float],
) -> np.ndarray:
    """Baseline-subtract tensor (preserves Hz units).

    Parameters
    ----------
    tensor : ndarray, shape (..., n_bins, n_units)
        Activity tensor.
    bin_centers : ndarray, shape (n_bins,)
        Time bin centers.
    baseline_window : tuple of float
        (start, end) baseline window in seconds.

    Returns
    -------
    ndarray
        Baseline-subtracted tensor.
    """
    baseline_mask = (bin_centers >= baseline_window[0]) & (bin_centers <= baseline_window[1])
    baseline_data = tensor[..., baseline_mask, :].reshape(-1, tensor.shape[-1])
    mu = np.mean(baseline_data, axis=0, keepdims=True)
    return tensor - mu


# ── Unit selection ────────────────────────────────────────────────────

def get_good_cluster_ids(session, min_rate_hz: float = 1.0) -> List[int]:
    """Get cluster IDs that pass quality criteria.

    Uses the priority order:
    1. good_and_stable_ids (UnitMatch tracked)
    2. good_cluster_ids (Kilosort "good")
    3. All clusters (fallback)

    Then applies firing rate filter.

    Parameters
    ----------
    session : Session
        Session object.
    min_rate_hz : float
        Minimum firing rate threshold.

    Returns
    -------
    list of int
        Cluster IDs that pass criteria.
    """
    # Priority order for cluster selection
    if hasattr(session, 'good_and_stable_ids') and session.good_and_stable_ids:
        candidates = set(session.good_and_stable_ids)
    elif session.good_cluster_ids:
        candidates = set(session.good_cluster_ids)
    else:
        candidates = {c.cluster_id for c in session.clusters}

    # Apply rate filter
    max_time = max((c.spike_times[-1] if len(c.spike_times) > 0 else 0) for c in session.clusters)
    recording_duration = max_time if max_time > 0 else 1.0

    good_ids = []
    for cluster in session.clusters:
        if cluster.cluster_id in candidates:
            rate = len(cluster.spike_times) / recording_duration
            if rate >= min_rate_hz:
                good_ids.append(cluster.cluster_id)

    return sorted(good_ids)


# ── Statistical utilities ─────────────────────────────────────────────

def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    axis: int = 0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bootstrap confidence intervals.

    Parameters
    ----------
    data : ndarray
        Data to bootstrap.
    n_bootstrap : int
        Number of bootstrap samples.
    ci_level : float
        Confidence level (e.g., 0.95 for 95% CI).
    axis : int
        Axis to bootstrap along.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ci_lower : ndarray
        Lower bound of confidence interval.
    ci_upper : ndarray
        Upper bound of confidence interval.
    """
    np.random.seed(seed)

    n_samples = data.shape[axis]
    bootstrap_means = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample = np.take(data, indices, axis=axis)
        bootstrap_means.append(np.mean(bootstrap_sample, axis=axis))

    bootstrap_means = np.array(bootstrap_means)

    # Compute percentiles
    alpha = 1 - ci_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(bootstrap_means, lower_percentile, axis=0)
    ci_upper = np.percentile(bootstrap_means, upper_percentile, axis=0)

    return ci_lower, ci_upper


def permutation_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> float:
    """Two-sample permutation test.

    Parameters
    ----------
    group_a, group_b : ndarray
        Data groups to compare.
    n_permutations : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    float
        P-value (two-tailed).
    """
    np.random.seed(seed)

    # Observed difference
    observed_diff = np.mean(group_a) - np.mean(group_b)

    # Combine groups
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)

    # Permutation distribution
    null_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        null_diffs.append(np.mean(perm_a) - np.mean(perm_b))

    null_diffs = np.array(null_diffs)

    # Two-tailed p-value
    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
    return p_value


def fdr_correct(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvals : ndarray
        Array of p-values.
    alpha : float
        FDR level.

    Returns
    -------
    ndarray
        Boolean array of significant tests.
    """
    pvals_flat = pvals.ravel()
    n_tests = len(pvals_flat)

    # Sort p-values
    sorted_indices = np.argsort(pvals_flat)
    sorted_pvals = pvals_flat[sorted_indices]

    # BH procedure
    threshold_line = alpha * np.arange(1, n_tests + 1) / n_tests
    significant_sorted = sorted_pvals <= threshold_line

    # Find largest k where p(k) <= alpha * k / m
    if np.any(significant_sorted):
        k_max = np.where(significant_sorted)[0][-1]
        threshold = sorted_pvals[k_max]
    else:
        threshold = 0.0

    # Apply threshold to original order
    significant = pvals_flat <= threshold
    return significant.reshape(pvals.shape)


def compute_auroc(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute area under ROC curve (AUROC).

    Parameters
    ----------
    group_a, group_b : ndarray
        Data for two groups.

    Returns
    -------
    float
        AUROC value (0.5 = chance, 1.0 = perfect separation).
    """
    # Use Mann-Whitney U test statistic
    try:
        u_stat, _ = mannwhitneyu(group_a, group_b, alternative='two-sided')
        auroc = u_stat / (len(group_a) * len(group_b))
        return auroc
    except ValueError:
        # Handle case where groups are identical
        return 0.5


# ── LDA coding direction ─────────────────────────────────────────────

def compute_lda_cd(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "sklearn",
    reg: float = 1.0,
    reg_style: str = "flat",
) -> np.ndarray:
    """Compute a coding direction via Fisher LDA with shrinkage.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix (e.g., per-trial baseline-averaged firing rates).
    y : ndarray, shape (n_samples,)
        Binary labels (0 or 1).
    method : str
        "sklearn" uses LinearDiscriminantAnalysis with Ledoit-Wolf shrinkage.
        "manual" uses Tikhonov-regularized inv(Cov + reg*I) * (mu1 - mu0).
    reg : float
        Regularization strength for method="manual" only.
    reg_style : str
        "flat" adds ``reg * I`` (default, matches Lohse et al. 2025).
        "trace_scaled" adds ``reg * trace(Cov)/p * I`` (proportional to
        average eigenvalue; used by 06_lick_motor scripts).

    Returns
    -------
    cd : ndarray, shape (n_features,)
        Unit-length coding direction vector (class 1 > class 0).
    """
    y = np.asarray(y).ravel()
    if len(np.unique(y)) < 2:
        raise ValueError("Need both classes present in y")

    if method == "sklearn":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(X, y)
        # LDA coefficients define the discriminant direction
        w = lda.coef_.ravel()
    else:
        # Manual Tikhonov-regularized Fisher discriminant
        mu1 = X[y == 1].mean(axis=0)
        mu0 = X[y == 0].mean(axis=0)
        diff = mu1 - mu0
        Xc = X - X.mean(axis=0)
        cov = np.cov(Xc, rowvar=False)
        if cov.ndim < 2:
            cov = np.atleast_2d(cov)
        if reg_style == "trace_scaled":
            reg_strength = reg * np.trace(cov) / cov.shape[0]
        else:  # "flat"
            reg_strength = reg
        cov_reg = cov + reg_strength * np.eye(cov.shape[0])
        try:
            w = np.linalg.solve(cov_reg, diff)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(cov_reg) @ diff

    # Ensure sign convention: class 1 (Hit) projects higher than class 0 (Miss)
    mu1_proj = X[y == 1].mean(axis=0) @ w
    mu0_proj = X[y == 0].mean(axis=0) @ w
    if mu1_proj < mu0_proj:
        w = -w

    norm = np.linalg.norm(w)
    return w / norm if norm > 0 else w