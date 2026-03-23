"""Downstream analysis utilities for GLM-HMM behavioral states.

Provides reusable functions for:
  - Cross-validation (leave-one-session-out)
  - Per-state behavioral metrics (d', criterion, hit/FA rates)
  - State-conditioned neural analysis (PSTHs, modulation indices)
  - Online (causal) single-trial state prediction
  - State-transition neural signatures

All functions accept the fitted GLMHMM model and/or the
``state_assignments.csv`` DataFrame produced by the fitting pipeline.

Reference
---------
Ashwood, Roy, Stone et al. (2022). Mice alternate between discrete
strategies during perceptual decision-making. Nat. Neurosci. 25, 201-212.
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from scipy.stats import norm


# =====================================================================
# 1.  Cross-Validation  (Leave-One-Session-Out)
# =====================================================================

def loso_cross_validation(
    sessions_data: List[Dict[str, Any]],
    K: int,
    *,
    config=None,
    n_restarts: int = 10,
    max_iter: int = 200,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Leave-One-Session-Out cross-validation for a GLM-HMM at a given K.

    For each fold, one session is held out.  The model is fit on the
    remaining sessions and evaluated on the held-out session via
    log-likelihood.

    Parameters
    ----------
    sessions_data : list of session dicts (from ``prepare_session_data``).
    K : int  –  Number of states.
    config : GLMHMMConfig, optional.
    n_restarts : int  –  Random restarts per fold.
    max_iter : int  –  Max EM iterations.
    seed : int  –  Base random seed.
    verbose : bool  –  Print progress.

    Returns
    -------
    DataFrame with columns:
        fold, held_out_session, n_trials_test,
        train_ll, test_ll, test_ll_per_trial, test_accuracy
    """
    from visdetect.analysis.hmm import GLMHMM, GLMHMMConfig

    cfg = config or GLMHMMConfig(
        max_iter=max_iter, n_restarts=n_restarts, verbose=False
    )
    # Ensure per-restart verbosity is off
    cfg_fold = GLMHMMConfig(**{
        k: getattr(cfg, k) for k in cfg.__dataclass_fields__
    })
    cfg_fold.verbose = False

    n_features = sessions_data[0]["X"].shape[1]
    n_sessions = len(sessions_data)
    records = []

    for fold_idx in range(n_sessions):
        held_out = sessions_data[fold_idx]
        train = [s for i, s in enumerate(sessions_data) if i != fold_idx]
        sname = held_out.get("session_name", f"session_{fold_idx}")

        if verbose:
            print(f"  Fold {fold_idx + 1}/{n_sessions}  "
                  f"(held-out: {sname}, {len(held_out['y'])} trials)")

        best_ll = -np.inf
        best_model = None
        for r in range(cfg.n_restarts):
            model = GLMHMM(K, n_features, config=cfg_fold)
            try:
                ll = model.fit(train, seed=seed + r * 137 + fold_idx * 7)
            except Exception:
                continue
            if ll > best_ll:
                best_ll = ll
                best_model = model

        if best_model is None:
            warnings.warn(f"All restarts failed for fold {fold_idx}")
            continue

        # Evaluate on held-out session
        test_ll = best_model.log_likelihood([held_out])
        n_test = len(held_out["y"])

        # Prediction accuracy: most-likely state → predicted P(lick),
        # then threshold at 0.5 to get predicted choice
        states = best_model.most_likely_states(held_out)
        X_test = held_out["X"]
        y_test = held_out["y"]
        p_lick = np.array([
            expit(best_model.weights[states[t]] @ X_test[t])
            for t in range(n_test)
        ])
        pred_choice = (p_lick >= 0.5).astype(float)
        accuracy = np.mean(pred_choice == y_test)

        records.append({
            "fold": fold_idx,
            "held_out_session": sname,
            "n_trials_test": n_test,
            "train_ll": best_ll,
            "test_ll": test_ll,
            "test_ll_per_trial": test_ll / max(n_test, 1),
            "test_accuracy": accuracy,
        })

    return pd.DataFrame(records)


# =====================================================================
# 2.  Per-State Behavioral Metrics
# =====================================================================

def compute_state_behavioral_metrics(
    assignments_df: pd.DataFrame,
    state_labels: List[str],
    n_states: int,
) -> pd.DataFrame:
    """Compute SDT and behavioral metrics per HMM state.

    Parameters
    ----------
    assignments_df : DataFrame with columns ``hmm_state``, ``is_hit``,
        ``is_miss``, ``is_fa``, ``is_go``, ``is_catch``.
    state_labels : list of str  –  Human-readable state names.
    n_states : int

    Returns
    -------
    DataFrame with one row per state: hit_rate, fa_rate, early_lick_rate,
    catch_lick_rate, dprime, criterion, n_trials, fraction.
    """
    rows = []
    N = len(assignments_df)
    for k in range(n_states):
        sub = assignments_df[assignments_df["hmm_state"] == k]
        n = len(sub)
        if n == 0:
            rows.append({"state": k, "label": state_labels[k], "n_trials": 0,
                          "fraction": 0.0})
            continue

        # SDT-style tallies: identify go/catch trials, but include only
        # genuine SDT outcomes ('hit' or 'miss') in denominators so that
        # anticipatory 'fa', 'ref', 'abort' are excluded (match staging
        # logic in `compute_session_performance`).
        go_all = sub[sub["is_go"] == True]
        catch_all = sub[sub["is_catch"] == True]

        # Keep only trials whose outcome is 'hit' or 'miss' for SDT counts
        out_series_go = go_all.get("outcome", pd.Series(dtype=object))
        out_series_catch = catch_all.get("outcome", pd.Series(dtype=object))
        go = go_all[out_series_go.isin(["hit", "miss"]) ]
        catch = catch_all[out_series_catch.isin(["hit", "miss"]) ]

        # Hit rate (on go trials excluding anticipatory FAs). Default 0 if no trials.
        hit_rate = go["is_hit"].mean() if len(go) > 0 else 0.0
        early_lick_rate = sub["is_fa"].mean()

        # Catch-trial lick rate for SDT: only count hits on catch trials.
        catch_lick = catch["is_hit"].mean() if len(catch) > 0 else 0.0

        # d-prime: compute only when both go and catch SDT denominators exist
        if len(go) > 0 and len(catch) > 0:
            hr_c = np.clip(hit_rate, 0.01, 0.99)
            far_c = np.clip(catch_lick, 0.01, 0.99)
            dprime = float(norm.ppf(hr_c) - norm.ppf(far_c))
            criterion = float(-0.5 * (norm.ppf(hr_c) + norm.ppf(far_c)))
        else:
            dprime = np.nan
            criterion = np.nan

        rows.append({
            "state": k,
            "label": state_labels[k],
            "n_trials": n,
            "fraction": n / N,
            "hit_rate_go": hit_rate,
            "catch_lick_rate": catch_lick,
            "early_lick_rate": early_lick_rate,
            "dprime": dprime,
            "criterion": criterion,
        })
    return pd.DataFrame(rows)


def compute_per_session_state_metrics(
    assignments_df: pd.DataFrame,
    state_labels: List[str],
    n_states: int,
) -> pd.DataFrame:
    """Compute behavioral metrics per session x state.

    Returns DataFrame with columns: session_name, state, label,
    n_trials, fraction, hit_rate_go, catch_lick_rate, early_lick_rate,
    dprime, criterion.
    """
    all_rows = []
    for sname, sdf in assignments_df.groupby("session_name", sort=False):
        per_state = compute_state_behavioral_metrics(sdf, state_labels, n_states)
        per_state.insert(0, "session_name", sname)
        all_rows.append(per_state)
    return pd.concat(all_rows, ignore_index=True)


# =====================================================================
# 3.  Across-Learning State Dynamics
# =====================================================================

def compute_learning_trajectory(
    assignments_df: pd.DataFrame,
    state_labels: List[str],
    n_states: int,
    session_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """State fractions and d' per session, ordered chronologically.

    Parameters
    ----------
    assignments_df : full state_assignments DataFrame.
    state_labels : list of human-readable labels.
    n_states : number of states.
    session_order : explicit chronological ordering (optional).

    Returns
    -------
    DataFrame with one row per session, columns:
        session_name, session_idx, overall_dprime,
        frac_State0 … frac_StateK,  dprime_State0 … dprime_StateK
    """
    metrics = compute_per_session_state_metrics(
        assignments_df, state_labels, n_states
    )
    sessions = (
        session_order
        if session_order
        else list(assignments_df["session_name"].unique())
    )

    rows = []
    for idx, sname in enumerate(sessions):
        sdf = assignments_df[assignments_df["session_name"] == sname]
        if len(sdf) == 0:
            continue
        row: Dict[str, Any] = {"session_name": sname, "session_idx": idx}

        # Overall d': use only go/catch trials whose outcome is 'hit' or
        # 'miss' so anticipatory 'fa', 'ref', and 'abort' are excluded
        # from the SDT denominators (consistent with staging/manifest).
        go = sdf[(sdf["is_go"] == True) & (sdf.get("outcome", pd.Series(dtype=object)).isin(["hit", "miss"]))]
        catch = sdf[(sdf["is_catch"] == True) & (sdf.get("outcome", pd.Series(dtype=object)).isin(["hit", "miss"]))]
        if len(go) > 0 and len(catch) > 0:
            hr = np.clip(go["is_hit"].mean(), 0.01, 0.99)
            far = np.clip(catch["is_hit"].mean(), 0.01, 0.99)
            row["overall_dprime"] = float(norm.ppf(hr) - norm.ppf(far))
        else:
            row["overall_dprime"] = np.nan

        # Per-state fractions and d'
        s_metrics = metrics[metrics["session_name"] == sname]
        for k in range(n_states):
            lbl = state_labels[k]
            s_row = s_metrics[s_metrics["state"] == k]
            row[f"frac_{lbl}"] = float(s_row["fraction"].values[0]) if len(s_row) > 0 else 0.0
            row[f"dprime_{lbl}"] = float(s_row["dprime"].values[0]) if len(s_row) > 0 else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# =====================================================================
# 4.  State-Conditioned Neural Analysis
# =====================================================================


def get_state_trial_indices(
    assignments_df: pd.DataFrame,
    session_name: str,
    state: int,
) -> np.ndarray:
    """Return 0-based trial indices for a given session and state."""
    mask = (
        (assignments_df["session_name"] == session_name)
        & (assignments_df["hmm_state"] == state)
    )
    return assignments_df.loc[mask, "trial_idx"].values.astype(int)


def compute_state_conditioned_psth(
    session,
    assignments_df: pd.DataFrame,
    state: int,
    cluster_id: int,
    event_name: str = "Change_ON",
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.01,
    valid_outcomes: Optional[set] = None,
) -> Dict[str, Any]:
    """Compute a PSTH for one cluster, restricted to trials in a given HMM state.

    Parameters
    ----------
    session : Session object (with clusters and ni_events).
    assignments_df : full state_assignments DataFrame.
    state : int  –  HMM state index.
    cluster_id : int  –  Which cluster to analyze.
    event_name : str  –  NI event to align to (default "Change_ON").
    window : (pre, post) seconds.
    bin_size : float  –  Bin width in seconds.
    valid_outcomes : optional set of lowercase trialoutcome strings.
        When given, only trials whose ``trialoutcome`` (lower-cased)
        belongs to this set are included.  Useful e.g. for excluding
        FA/abort trials when aligning to ``Change_ON``.

    Returns
    -------
    dict with keys:
        psth (1D array, Hz), sem (1D array), bin_centers (1D),
        trials_matrix (n_trials x n_bins, Hz), trial_indices, n_trials.
    """
    from visdetect.analysis.align import (
        get_event_times_by_trial,
        align_spikes_to_events,
    )

    sname = session.session_name or ""
    trial_idx = get_state_trial_indices(assignments_df, sname, state)
    if len(trial_idx) == 0:
        bins = np.arange(window[0], window[1] + bin_size, bin_size)
        bc = (bins[:-1] + bins[1:]) / 2.0
        return {
            "psth": np.zeros(len(bc)),
            "sem": np.zeros(len(bc)),
            "bin_centers": bc,
            "trials_matrix": np.empty((0, len(bc))),
            "trial_indices": trial_idx,
            "n_trials": 0,
        }

    # Get per-trial event times (NaN for missing)
    all_event_times = get_event_times_by_trial(session, event_name)

    # Select event times for state trials only
    valid_events = []
    valid_idx = []
    trials = getattr(session, "trials", []) or []
    for ti in trial_idx:
        if ti < len(all_event_times) and not np.isnan(all_event_times[ti]):
            # Outcome filter
            if valid_outcomes is not None and ti < len(trials):
                oc = getattr(trials[ti], "trialoutcome", "").lower()
                if oc not in valid_outcomes:
                    continue
            valid_events.append(all_event_times[ti])
            valid_idx.append(ti)

    if len(valid_events) == 0:
        bins = np.arange(window[0], window[1] + bin_size, bin_size)
        bc = (bins[:-1] + bins[1:]) / 2.0
        return {
            "psth": np.zeros(len(bc)),
            "sem": np.zeros(len(bc)),
            "bin_centers": bc,
            "trials_matrix": np.empty((0, len(bc))),
            "trial_indices": np.array(valid_idx),
            "n_trials": 0,
        }

    # Find cluster spike times
    spike_times = None
    for c in session.clusters:
        if c.cluster_id == cluster_id:
            spike_times = c.spike_times
            break
    if spike_times is None:
        raise ValueError(f"Cluster {cluster_id} not found in session {sname}")

    # Align
    fr_matrix, bin_centers = align_spikes_to_events(
        spike_times, valid_events, window=window, bin_size=bin_size
    )
    psth = fr_matrix.mean(axis=0) if fr_matrix.shape[0] > 0 else np.zeros(len(bin_centers))
    sem = (
        fr_matrix.std(axis=0) / np.sqrt(fr_matrix.shape[0])
        if fr_matrix.shape[0] > 1
        else np.zeros(len(bin_centers))
    )

    return {
        "psth": psth,
        "sem": sem,
        "bin_centers": bin_centers,
        "trials_matrix": fr_matrix,
        "trial_indices": np.array(valid_idx),
        "n_trials": fr_matrix.shape[0],
    }


def compute_state_modulation_index(
    session,
    assignments_df: pd.DataFrame,
    cluster_id: int,
    state_a: int,
    state_b: int,
    event_name: str = "Change_ON",
    response_window: Tuple[float, float] = (0.0, 0.3),
    bin_size: float = 0.01,
) -> Dict[str, float]:
    """Compute a modulation index between two HMM states for one cluster.

    MI = (r_A - r_B) / (r_A + r_B)

    where r_A, r_B are mean firing rates in the response_window.

    Returns dict with: MI, rate_A, rate_B, n_trials_A, n_trials_B.
    """
    results_a = compute_state_conditioned_psth(
        session, assignments_df, state_a, cluster_id,
        event_name=event_name,
        window=(response_window[0] - 0.1, response_window[1] + 0.1),
        bin_size=bin_size,
    )
    results_b = compute_state_conditioned_psth(
        session, assignments_df, state_b, cluster_id,
        event_name=event_name,
        window=(response_window[0] - 0.1, response_window[1] + 0.1),
        bin_size=bin_size,
    )

    bc = results_a["bin_centers"]
    mask = (bc >= response_window[0]) & (bc <= response_window[1])

    rate_a = float(results_a["psth"][mask].mean()) if results_a["n_trials"] > 0 else 0.0
    rate_b = float(results_b["psth"][mask].mean()) if results_b["n_trials"] > 0 else 0.0

    denom = rate_a + rate_b
    mi = (rate_a - rate_b) / denom if denom > 0 else 0.0

    return {
        "modulation_index": mi,
        "rate_state_a": rate_a,
        "rate_state_b": rate_b,
        "n_trials_a": results_a["n_trials"],
        "n_trials_b": results_b["n_trials"],
    }


def compute_population_state_modulation(
    session,
    assignments_df: pd.DataFrame,
    state_a: int,
    state_b: int,
    cluster_ids: Optional[List[int]] = None,
    event_name: str = "Change_ON",
    response_window: Tuple[float, float] = (0.0, 0.3),
    bin_size: float = 0.01,
) -> pd.DataFrame:
    """Modulation index for all (or selected) clusters in a session.

    Returns DataFrame: cluster_id, modulation_index, rate_state_a,
    rate_state_b, n_trials_a, n_trials_b.
    """
    if cluster_ids is None:
        # Use good_and_stable_ids > good_cluster_ids > all clusters
        cluster_ids = (
            session.good_and_stable_ids
            or session.good_cluster_ids
            or [c.cluster_id for c in session.clusters]
        )

    rows = []
    for cid in cluster_ids:
        try:
            mi = compute_state_modulation_index(
                session, assignments_df, cid, state_a, state_b,
                event_name=event_name,
                response_window=response_window,
                bin_size=bin_size,
            )
            mi["cluster_id"] = cid
            rows.append(mi)
        except Exception as exc:
            warnings.warn(f"Cluster {cid}: {exc}")
    return pd.DataFrame(rows)


# =====================================================================
# 5.  State-Transition Neural Signatures
# =====================================================================

def find_state_transitions(
    assignments_df: pd.DataFrame,
    session_name: str,
    from_state: int,
    to_state: int,
    context_trials: int = 5,
) -> List[Dict[str, Any]]:
    """Find trials where the HMM state switches from `from_state` to `to_state`.

    Returns a list of dicts, each with:
      - transition_trial: the first trial in the new state
      - pre_trials: trial indices of preceding `context_trials`
      - post_trials: trial indices of following `context_trials`
    """
    sdf = assignments_df[assignments_df["session_name"] == session_name].copy()
    sdf = sdf.sort_values("trial_idx").reset_index(drop=True)

    states = sdf["hmm_state"].values
    trial_indices = sdf["trial_idx"].values

    transitions = []
    for i in range(1, len(states)):
        if states[i - 1] == from_state and states[i] == to_state:
            t_idx = trial_indices[i]
            pre_start = max(0, i - context_trials)
            post_end = min(len(trial_indices), i + context_trials)
            transitions.append({
                "transition_trial": int(t_idx),
                "transition_pos": i,
                "pre_trials": trial_indices[pre_start:i].tolist(),
                "post_trials": trial_indices[i:post_end].tolist(),
            })
    return transitions


def compute_transition_triggered_psth(
    session,
    assignments_df: pd.DataFrame,
    cluster_id: int,
    from_state: int,
    to_state: int,
    event_name: str = "Change_ON",
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.01,
    context_trials: int = 5,
) -> Dict[str, Any]:
    """Compute average PSTHs before and after state transitions.

    Averages firing across all transition events in the session.

    Returns dict with:
      pre_psth, post_psth, bin_centers, n_transitions,
      pre_sem, post_sem
    """
    from visdetect.analysis.align import (
        get_event_times_by_trial,
        align_spikes_to_events,
    )

    sname = session.session_name or ""
    transitions = find_state_transitions(
        assignments_df, sname, from_state, to_state,
        context_trials=context_trials,
    )
    if not transitions:
        bins = np.arange(window[0], window[1] + bin_size, bin_size)
        bc = (bins[:-1] + bins[1:]) / 2.0
        return {
            "pre_psth": np.zeros(len(bc)),
            "post_psth": np.zeros(len(bc)),
            "bin_centers": bc,
            "n_transitions": 0,
            "pre_sem": np.zeros(len(bc)),
            "post_sem": np.zeros(len(bc)),
        }

    all_event_times = get_event_times_by_trial(session, event_name)

    # Find spike times for cluster
    spike_times = None
    for c in session.clusters:
        if c.cluster_id == cluster_id:
            spike_times = c.spike_times
            break
    if spike_times is None:
        raise ValueError(f"Cluster {cluster_id} not found")

    # Gather pre and post event times across all transitions
    pre_events, post_events = [], []
    for tr in transitions:
        for ti in tr["pre_trials"]:
            if ti < len(all_event_times) and not np.isnan(all_event_times[ti]):
                pre_events.append(all_event_times[ti])
        for ti in tr["post_trials"]:
            if ti < len(all_event_times) and not np.isnan(all_event_times[ti]):
                post_events.append(all_event_times[ti])

    def _psth(events):
        if not events:
            bins = np.arange(window[0], window[1] + bin_size, bin_size)
            bc = (bins[:-1] + bins[1:]) / 2.0
            return np.zeros(len(bc)), np.zeros(len(bc)), bc
        fr, bc = align_spikes_to_events(spike_times, events, window=window, bin_size=bin_size)
        return fr.mean(axis=0), fr.std(axis=0) / np.sqrt(max(fr.shape[0], 1)), bc

    pre_psth, pre_sem, bc = _psth(pre_events)
    post_psth, post_sem, _ = _psth(post_events)

    return {
        "pre_psth": pre_psth,
        "post_psth": post_psth,
        "bin_centers": bc,
        "n_transitions": len(transitions),
        "pre_sem": pre_sem,
        "post_sem": post_sem,
    }


# =====================================================================
# 6.  Online (Causal) State Prediction
# =====================================================================

def forward_only_state_posteriors(model, session_data: Dict[str, Any]) -> np.ndarray:
    """Compute causal (forward-only) state posteriors: P(z_t | y_{1:t}).

    Unlike the full forward-backward posteriors used during fitting, these
    only use *past and present* observations — suitable for real-time or
    held-out prediction where the future is unknown.

    Parameters
    ----------
    model : GLMHMM  –  Fitted model.
    session_data : dict with 'y' and 'X' arrays.

    Returns
    -------
    filtered : (T, K) array of causal state posteriors.
    """
    y = session_data["y"]
    X = session_data["X"]
    T = len(y)
    K = model.n_states

    if T == 0:
        return np.empty((0, K))

    log_likes = model._emission_log_likes(y, X)
    log_alpha, _ = model._forward(log_likes)

    # Normalize each row to get P(z_t | y_{1:t})
    log_filtered = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_filtered)


def predict_trial_by_trial(
    model, session_data: Dict[str, Any], causal: bool = True
) -> pd.DataFrame:
    """Predict P(lick) and most-likely state for each trial.

    Parameters
    ----------
    model : GLMHMM
    session_data : session dict
    causal : if True, use forward-only posteriors (no future info).

    Returns
    -------
    DataFrame with columns: trial_idx, y_true, p_lick, pred_choice,
        most_likely_state, and p_state_0 … p_state_K.
    """
    y = session_data["y"]
    X = session_data["X"]
    T = len(y)
    K = model.n_states

    if causal:
        posteriors = forward_only_state_posteriors(model, session_data)
    else:
        posteriors = model.state_posteriors(session_data)

    # Weighted P(lick) across states
    logits = X @ model.weights.T  # (T, K)
    p_per_state = expit(logits)   # (T, K)
    p_lick = (posteriors * p_per_state).sum(axis=1)  # (T,)
    pred_choice = (p_lick >= 0.5).astype(float)
    most_likely = posteriors.argmax(axis=1)

    rows = {
        "trial_idx": np.arange(T),
        "y_true": y,
        "p_lick": p_lick,
        "pred_choice": pred_choice,
        "most_likely_state": most_likely,
    }
    for k in range(K):
        rows[f"p_state_{k}"] = posteriors[:, k]

    return pd.DataFrame(rows)


# =====================================================================
# 7.  Utility: load model + assignments
# =====================================================================

def load_hmm_results(
    data_dir: str | Path,
    K: int | None = None,
) -> Tuple[Any, pd.DataFrame, List[str]]:
    """Load fitted model, state assignments, and labels from disk.

    Parameters
    ----------
    data_dir : directory containing model_K*.pkl, state_assignments.csv,
        and state_labels.json.
    K : optional number of states.  When given, loads
        ``model_K{K}.pkl`` / ``state_assignments_K{K}.csv`` /
        ``state_labels_K{K}.json`` instead of the defaults (highest-K model).

    Returns
    -------
    model : GLMHMM
    assignments_df : DataFrame
    state_labels : list of str
    """
    data_dir = Path(data_dir)

    if K is not None:
        # Explicit K requested — load per-K artefacts
        model_path = data_dir / f"model_K{K}.pkl"
        if not model_path.exists():
            avail = sorted(data_dir.glob("model_K*.pkl"))
            avail_ks = [p.stem.replace("model_K", "") for p in avail]
            raise FileNotFoundError(
                f"model_K{K}.pkl not found in {data_dir}.  "
                f"Available: {avail_ks}"
            )
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Per-K assignments (fall back to default if missing)
        k_assign = data_dir / f"state_assignments_K{K}.csv"
        if k_assign.exists():
            assignments_df = pd.read_csv(k_assign, dtype={"session_name": str})
        else:
            assignments_df = pd.read_csv(
                data_dir / "state_assignments.csv", dtype={"session_name": str}
            )

        # Per-K labels (fall back to default / generic)
        k_labels = data_dir / f"state_labels_K{K}.json"
        if k_labels.exists():
            with open(k_labels) as f:
                info = json.load(f)
                state_labels = info.get("labels", [f"State_{i}" for i in range(K)])
        else:
            labels_path = data_dir / "state_labels.json"
            if labels_path.exists():
                with open(labels_path) as f:
                    info = json.load(f)
                    if info.get("K") == K:
                        state_labels = info["labels"]
                    else:
                        state_labels = [f"State_{i}" for i in range(K)]
            else:
                state_labels = [f"State_{i}" for i in range(K)]

        return model, assignments_df, state_labels

    # Default behaviour: highest-K model
    pkls = sorted(data_dir.glob("model_K*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No model pkl found in {data_dir}")
    model_path = pkls[-1]  # highest K if multiple exist
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    assignments_df = pd.read_csv(
        data_dir / "state_assignments.csv", dtype={"session_name": str}
    )

    labels_path = data_dir / "state_labels.json"
    if labels_path.exists():
        with open(labels_path) as f:
            info = json.load(f)
            state_labels = info.get("labels", [f"State_{k}" for k in range(model.n_states)])
    else:
        state_labels = [f"State_{k}" for k in range(model.n_states)]

    return model, assignments_df, state_labels
