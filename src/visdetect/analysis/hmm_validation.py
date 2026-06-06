"""External state validation observables for the GLM-HMM (F22).

These compute per-state distributions of *observables not used in fitting*,
to test whether the inferred behavioral states correspond to genuinely
different regimes (per audit spec §4.7).

Observables provided here:
  - per-state lick latency on hits (change-relative)
  - per-state response-time quantiles (Ashwood-style Q-Q analog)
  - per-state psychometric slope (logistic fit P(lick) vs log2(change_size))

A separate script (``scripts/analysis/behavior/hmm_external_validation.py``)
computes TF-pulse responsiveness per state by integrating with the existing
``visdetect.analysis.tf_pulse`` module.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize


# =====================================================================
# 1.  Lick latency
# =====================================================================

def per_state_lick_latency(
    assignments_df: pd.DataFrame,
    n_states: int,
    state_col: str = "hmm_state",
    latency_col: str = "rt",
) -> pd.DataFrame:
    """Median, IQR, and n_hits of lick latency per HMM state.

    Latency here = change-relative reaction time (``rt`` column in the
    trial dataframe), restricted to hits on go trials.

    Returns DataFrame: state, median_latency_s, iqr_s, n_hits.
    """
    required = {"is_hit", "is_go", state_col, latency_col}
    missing = required - set(assignments_df.columns)
    if missing:
        raise KeyError(f"assignments_df is missing required columns: {sorted(missing)}")
    hits = assignments_df[assignments_df["is_hit"] & assignments_df["is_go"]]
    rows = []
    for k in range(n_states):
        sub = hits[hits[state_col] == k][latency_col].dropna()
        if len(sub):
            rows.append({
                "state": k,
                "median_latency_s": float(np.median(sub)),
                "iqr_s": float(np.percentile(sub, 75) - np.percentile(sub, 25)),
                "n_hits": int(len(sub)),
            })
        else:
            rows.append({
                "state": k,
                "median_latency_s": np.nan,
                "iqr_s": np.nan,
                "n_hits": 0,
            })
    return pd.DataFrame(rows)


# =====================================================================
# 2.  Response-time quantiles (Ashwood Fig 6 Q-Q analog)
# =====================================================================

def per_state_response_time_quantiles(
    assignments_df: pd.DataFrame,
    n_states: int,
    quantiles: Iterable[float] = (0.25, 0.5, 0.75, 0.9),
    state_col: str = "hmm_state",
    latency_col: str = "rt",
) -> pd.DataFrame:
    """Per-state RT quantiles on hits."""
    required = {"is_hit", "is_go", state_col, latency_col}
    missing = required - set(assignments_df.columns)
    if missing:
        raise KeyError(f"assignments_df is missing required columns: {sorted(missing)}")
    hits = assignments_df[assignments_df["is_hit"] & assignments_df["is_go"]]
    rows = []
    quantiles = list(quantiles)
    for k in range(n_states):
        sub = hits[hits[state_col] == k][latency_col].dropna().values
        row = {"state": k, "n": int(len(sub))}
        for q in quantiles:
            label = f"q{int(round(q * 100)):02d}"
            row[label] = float(np.quantile(sub, q)) if len(sub) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# =====================================================================
# 3.  Per-state psychometric slope
# =====================================================================

def _logistic_slope(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """MLE fit of P(y=1) = sigmoid(beta0 + beta1 * x); return (intercept, slope).

    Uses scipy.optimize.minimize on negative log-likelihood. Returns
    (np.nan, np.nan) if fitting fails or the data is degenerate.
    """
    if len(x) == 0 or y.std() == 0 or x.std() == 0:
        return (np.nan, np.nan)

    def nll(params):
        b0, b1 = params
        z = b0 + b1 * x
        return -np.sum(y * z - np.logaddexp(0, z))

    try:
        res = minimize(nll, x0=[0.0, 0.0], method="L-BFGS-B")
        if res.success:
            return (float(res.x[0]), float(res.x[1]))
    except Exception:
        pass
    return (np.nan, np.nan)


def per_state_psychometric_slope(
    assignments_df: pd.DataFrame,
    n_states: int,
    state_col: str = "hmm_state",
) -> pd.DataFrame:
    """Per-state psychometric: logistic fit of P(lick) vs log2(change_size) on go trials.

    Returns DataFrame: state, intercept, slope, n_go.
    """
    required = {"is_go", "is_hit", "change_size", state_col}
    missing = required - set(assignments_df.columns)
    if missing:
        raise KeyError(f"assignments_df is missing required columns: {sorted(missing)}")
    go = assignments_df[assignments_df["is_go"]]
    rows = []
    for k in range(n_states):
        sub = go[go[state_col] == k]
        if len(sub) < 5:
            rows.append({"state": k, "intercept": np.nan, "slope": np.nan,
                         "n_go": int(len(sub))})
            continue
        x = np.log2(np.clip(sub["change_size"].values.astype(float), 1.0, None))
        y = sub["is_hit"].astype(float).values
        b0, b1 = _logistic_slope(x, y)
        rows.append({"state": k, "intercept": b0, "slope": b1, "n_go": int(len(sub))})
    return pd.DataFrame(rows)


# =====================================================================
# 4.  TF-pulse responsiveness per state (key F22 discriminator)
# =====================================================================

def per_state_tf_pulse_lick_rate(
    session,
    assignments_df: pd.DataFrame,
    n_states: int,
    *,
    pulse_log2_threshold: float = 0.10,
    response_window_s: float = 0.40,
    state_col: str = "hmm_state",
) -> pd.DataFrame:
    """Per-state probability of lick within ``response_window_s`` of a sub-threshold
    TF pulse during baseline.

    A pulse is any TF excursion with |log2(tf/baseline_tf)| > pulse_log2_threshold
    that is NOT itself a scheduled change event (i.e., happens during baseline).

    Strategy:
      1. For each trial in ``assignments_df``, identify TF pulses during the
         baseline period (before change_time).
      2. For each pulse, check whether any lick (any outcome) occurred within
         response_window_s.
      3. Aggregate per state.

    Returns DataFrame: state, n_pulses, n_pulse_locked_licks, p_lick_pulse_locked.

    Notes
    -----
    .. warning::
       The DataFrame index of ``assignments_df`` is used to look up
       ``session.trials[trial_idx]``. Callers must ensure the index is
       a 0..N-1 sequential range matching the session's local trial order
       (call ``df.reset_index(drop=True)`` before passing in if needed).

    Requires session.trials to expose ``tf_trace``, ``tf_times``, and the
    trial's ``lick_times`` / ``firstlick`` fields. If the per-trial TF trace
    isn't available, returns NaN rows (per-state lengths preserved).
    """
    rows = []
    pulses_per_state = {k: [] for k in range(n_states)}
    locked_per_state = {k: 0 for k in range(n_states)}

    for trial_idx, row in assignments_df.iterrows():
        k = int(row[state_col])
        if k < 0 or k >= n_states:
            continue
        # Try to fetch trial-level TF trace.
        try:
            trial = session.trials[trial_idx]
        except Exception:
            continue
        tf_trace = getattr(trial, "tf_trace", None)
        tf_times = getattr(trial, "tf_times", None)
        if tf_trace is None or tf_times is None or len(tf_trace) == 0:
            continue
        tf_trace = np.asarray(tf_trace, dtype=float)
        tf_times = np.asarray(tf_times, dtype=float)

        baseline_tf = float(np.median(tf_trace))
        if baseline_tf <= 0:
            continue
        log2_dev = np.log2(np.maximum(tf_trace, 1e-6) / baseline_tf)
        pulse_mask = np.abs(log2_dev) > pulse_log2_threshold

        # Limit to baseline period (before change_time if available).
        ct = getattr(trial, "change_time", None)
        if ct is not None and ct > 0:
            pulse_mask &= tf_times < ct
        pulse_times = tf_times[pulse_mask]
        if len(pulse_times) == 0:
            continue
        pulses_per_state[k].extend(pulse_times.tolist())

        # Lick times on this trial.
        lick_times = getattr(trial, "lick_times", None)
        if lick_times is None or len(lick_times) == 0:
            continue
        lick_times = np.asarray(lick_times, dtype=float)
        for pt in pulse_times:
            if np.any((lick_times > pt) & (lick_times <= pt + response_window_s)):
                locked_per_state[k] += 1

    for k in range(n_states):
        n_p = len(pulses_per_state[k])
        n_l = locked_per_state[k]
        rows.append({
            "state": k,
            "n_pulses": n_p,
            "n_pulse_locked_licks": n_l,
            "p_lick_pulse_locked": (n_l / n_p) if n_p > 0 else np.nan,
        })
    return pd.DataFrame(rows)
