"""Analysis helpers to compute TF-responsive and lick-responsive neurons.

Provides utilities to compute per-cluster responsiveness using simple statistical
tests on PETH windows. These are intentionally minimal and should be adapted to
your experimental design (baseline windows, multiple comparisons, etc.).
"""
from typing import Any, Dict, Tuple, List
import numpy as np
import pandas as pd
from scipy import stats


def is_tf_responsive(peth_trials: np.ndarray, baseline_bins: List[int], response_bins: List[int], alpha: float = 0.01) -> Dict[str, float]:
    """Simple per-unit test for TF-responsivity.

    peth_trials: trials x bins array of spike counts/rates for a unit aligned to TF event.
    baseline_bins: list or slice indices defining the baseline period.
    response_bins: list or slice indices defining the response period.

    Returns a dict with p-value, mean_baseline, mean_response, and is_responsive boolean.
    """
    baseline = peth_trials[:, baseline_bins].mean(axis=1)
    response = peth_trials[:, response_bins].mean(axis=1)
    try:
        stat, p = stats.ttest_rel(response, baseline, nan_policy='omit')
    except Exception:
        stat, p = np.nan, np.nan
    res = {
        'p_value': float(p) if np.isfinite(p) else np.nan,
        'mean_baseline': float(np.nanmean(baseline)),
        'mean_response': float(np.nanmean(response)),
        'is_responsive': bool(p < alpha) if np.isfinite(p) else False
    }
    return res


def summarize_responsivity(peths: Dict[int, Dict], baseline_idx: List[int], response_idx: List[int], alpha: float = 0.01) -> pd.DataFrame:
    """Run is_tf_responsive across a dict of peths (cluster_id -> {'trials_matrix': ...})."""
    rows = []
    for cid, info in peths.items():
        trials = info['trials_matrix']
        r = is_tf_responsive(trials, baseline_idx, response_idx, alpha=alpha)
        r['cluster_id'] = int(cid)
        rows.append(r)
    df = pd.DataFrame(rows).set_index('cluster_id')
    return df
