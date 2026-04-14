"""Shared helper functions for FA (false alarm) classification and analysis.

Functions extracted from scripts g, h, i, j in 07_advanced/ to avoid
code duplication across FA subtype analyses.

Functions
---------
Baseline TF trace helpers (from i, j):
    extract_baseline_tf_trace, extract_lta_segment, original_threshold_classify

Neural divergence helpers (from g, h):
    compute_timeresolved_auc, _find_clusters, grand_auc_cluster_test
"""

import numpy as np

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visdetect.analysis.utils import compute_auroc


# =====================================================================
# Baseline TF trace extraction (from i, j)
# =====================================================================

def extract_baseline_tf_trace(trial, baseline_stride, sample_period, min_rt):
    """Extract full baseline log2(TF) trace for one FA trial.

    Parameters
    ----------
    trial : Trial
        A Trial object with baseline_values, n_seen, reactiontimes attrs.
    baseline_stride : int
        Stride for downsampling baseline values.
    sample_period : float
        Sampling period in seconds after stride (for consistency with
        callers; not used directly in this function).
    min_rt : float
        Minimum reaction time to include the trial.

    Returns
    -------
    log2_tf : np.ndarray or None
        Log2-transformed TF trace, or None if invalid.
    n_valid : int or None
        Number of valid samples.
    rt : float or None
        Reaction time in seconds.
    """
    bv = getattr(trial, "baseline_values", None)
    if bv is None:
        return None, None, None

    arr = np.array(bv).flatten()
    if baseline_stride > 1:
        arr = arr[::baseline_stride]

    n_seen = getattr(trial, "n_seen", None)
    if isinstance(n_seen, (int, np.integer)) and n_seen is not None and n_seen > 0:
        arr = arr[: int(n_seen)]

    if len(arr) < 10:
        return None, None, None

    rt_dict = getattr(trial, "reactiontimes", {}) or {}
    rt = rt_dict.get("FA", rt_dict.get("fa", np.nan))
    if np.isnan(rt) or rt < min_rt:
        return None, None, None

    log2_tf = np.log2(np.clip(arr.astype(float), 0.01, None))
    return log2_tf, len(log2_tf), float(rt)


def extract_lta_segment(trial, baseline_stride, sample_period, min_rt,
                        lta_history, lta_post):
    """Extract fixed-length log2(TF ratio) segment centered on lick.

    Uses MATLAB-style direct array indexing (no interpolation).

    Parameters
    ----------
    trial : Trial
        A Trial object.
    baseline_stride : int
        Stride for downsampling.
    sample_period : float
        Sampling period in seconds.
    min_rt : float
        Minimum reaction time.
    lta_history : int
        Number of samples before lick.
    lta_post : int
        Number of samples after lick.

    Returns
    -------
    log2_segment : np.ndarray or None
        Log2-transformed segment of length (lta_history + lta_post),
        or None if the trial is invalid or the segment is out of bounds.
    """
    bv = getattr(trial, "baseline_values", None)
    if bv is None:
        return None

    arr = np.array(bv).flatten()
    if baseline_stride > 1:
        arr = arr[::baseline_stride]

    rt_dict = getattr(trial, "reactiontimes", {}) or {}
    rt = rt_dict.get("FA", rt_dict.get("fa", np.nan))
    if np.isnan(rt) or rt < min_rt:
        return None

    lick_idx = int(round(rt / sample_period))
    start_idx = lick_idx - (lta_history - 1)
    end_idx = lick_idx + lta_post + 1

    if start_idx < 0 or end_idx > len(arr):
        return None

    seg = arr[start_idx:end_idx].astype(float)
    return np.log2(np.clip(seg, 0.01, None))


def original_threshold_classify(log2_tf, rt, classify_window, sample_period,
                                threshold):
    """Reproduce the original threshold-based FA classification.

    Parameters
    ----------
    log2_tf : np.ndarray
        Log2-transformed TF trace.
    rt : float
        Reaction time in seconds.
    classify_window : tuple of float
        (start, end) relative to lick for classification features.
    sample_period : float
        Sampling period in seconds.
    threshold : float
        Log2 TF threshold for "TF-triggered" classification.

    Returns
    -------
    str or None
        "TF-triggered" or "Impulsive", or None if insufficient data.
    """
    sample_times = np.arange(len(log2_tf)) * sample_period
    sample_times_rel = sample_times - rt
    mask = ((sample_times_rel >= classify_window[0])
            & (sample_times_rel < classify_window[1]))
    if mask.sum() < 2:
        return None
    max_l2 = np.max(log2_tf[mask])
    return "TF-triggered" if max_l2 >= threshold else "Impulsive"


# =====================================================================
# Time-resolved AUC (from g, h)
# =====================================================================

def compute_timeresolved_auc(tensor_a, tensor_b):
    """Compute AUC at each time bin using population-mean FR as the score.

    Parameters
    ----------
    tensor_a : np.ndarray, shape (n_a, n_bins, n_units)
        Firing rate tensor for group A.
    tensor_b : np.ndarray, shape (n_b, n_bins, n_units)
        Firing rate tensor for group B.

    Returns
    -------
    auc : np.ndarray, shape (n_bins,)
        AUC per time bin.
    """
    pop_a = np.nanmean(tensor_a, axis=2)    # (n_a, n_bins)
    pop_b = np.nanmean(tensor_b, axis=2)    # (n_b, n_bins)

    n_bins = pop_a.shape[1]
    auc = np.full(n_bins, np.nan)

    for b in range(n_bins):
        vals_a = pop_a[:, b]
        vals_b = pop_b[:, b]
        vals_a = vals_a[~np.isnan(vals_a)]
        vals_b = vals_b[~np.isnan(vals_b)]

        if len(vals_a) < 5 or len(vals_b) < 5:
            continue

        auc[b] = compute_auroc(vals_a, vals_b)

    return auc


# =====================================================================
# Cluster-finding helper (from g, h)
# =====================================================================

def _find_clusters(vals, thresh):
    """Find contiguous runs where |vals| > thresh.

    Parameters
    ----------
    vals : np.ndarray
        1-D array of test statistics.
    thresh : float
        Absolute threshold.

    Returns
    -------
    list of (int, int)
        List of (start, end) index pairs for each cluster.
    """
    above = np.abs(vals) > thresh
    clusters = []
    in_cluster = False
    start = 0
    for i in range(len(above)):
        if above[i] and not in_cluster:
            in_cluster = True
            start = i
        elif not above[i] and in_cluster:
            clusters.append((start, i))
            in_cluster = False
    if in_cluster:
        clusters.append((start, len(above)))
    return clusters


# =====================================================================
# Grand-average cluster permutation test (from g, h)
# =====================================================================

def grand_auc_cluster_test(auc_arr, n_perm=1000, cluster_alpha=0.05,
                           rng=None):
    """Cluster-based permutation test on grand-average AUC across sessions.

    H0: AUC = 0.5 at each bin. Tests via sign-flipping each session's
    deviation from 0.5.

    Parameters
    ----------
    auc_arr : np.ndarray, shape (n_sessions, n_bins)
        AUC values per session per bin.
    n_perm : int
        Number of sign-flip permutations.
    cluster_alpha : float
        Significance threshold for clusters.
    rng : np.random.Generator or None
        Random number generator. If None, uses default_rng(42).

    Returns
    -------
    mean_auc : np.ndarray, shape (n_bins,)
    sig_mask : np.ndarray, shape (n_bins,) boolean
    p_clusters : list of (start, end, stat, p_val)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_sess, n_bins = auc_arr.shape
    dev = auc_arr - 0.5

    mean_dev = np.nanmean(dev, axis=0)
    se = np.nanstd(dev, axis=0) / np.sqrt(n_sess)
    se[se == 0] = 1e-10
    t_obs = mean_dev / se

    thresh = 2.0  # ~p < 0.05 uncorrected
    obs_clusters = _find_clusters(t_obs, thresh)
    obs_cluster_stats = [np.sum(np.abs(t_obs[s:e])) for s, e in obs_clusters]

    max_cluster_null = np.zeros(n_perm)

    for p in range(n_perm):
        signs = rng.choice([-1, 1], size=n_sess)
        perm_dev = dev * signs[:, None]
        perm_mean = np.nanmean(perm_dev, axis=0)
        perm_se = np.nanstd(perm_dev, axis=0) / np.sqrt(n_sess)
        perm_se[perm_se == 0] = 1e-10
        perm_t = perm_mean / perm_se

        perm_clusters = _find_clusters(perm_t, thresh)
        if perm_clusters:
            max_cluster_null[p] = max(np.sum(np.abs(perm_t[s:e]))
                                       for s, e in perm_clusters)

    sig_mask = np.zeros(n_bins, dtype=bool)
    p_clusters = []

    for i, (s, e) in enumerate(obs_clusters):
        stat = obs_cluster_stats[i]
        p_val = (np.sum(max_cluster_null >= stat) + 1) / (n_perm + 1)
        p_clusters.append((s, e, stat, p_val))
        if p_val < cluster_alpha:
            sig_mask[s:e] = True

    mean_auc = np.nanmean(auc_arr, axis=0)
    return mean_auc, sig_mask, p_clusters
