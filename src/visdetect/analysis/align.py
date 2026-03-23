"""Event alignment and PETH utilities.

Functions:
- compute_true_reaction_time(trial, ni_events, trial_idx, shift_fa_hit_ms)
- get_event_times(session, event_name, outcomes=None, enforce_valid_outcomes=True)
- get_event_times_by_trial(session, event_name, enforce_valid_outcomes=True)
- align_spikes_to_events(spike_times, event_times, window, bin_size)
- compute_peth_for_session(session, event_name, window, bin_size, good_cluster_ids=None)

Safety: By default, ``get_event_times`` and ``get_event_times_by_trial`` auto-apply
the outcome filters defined in ``EVENT_VALID_OUTCOMES`` (from constants.py).
For example, Change_ON alignment automatically excludes FA/abort trials because the
change stimulus was never presented on those trials. Pass
``enforce_valid_outcomes=False`` to override (with care).
"""

from typing import List, Tuple, Optional, Dict, Any, Set
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import h5py
import warnings

from visdetect.analysis.constants import EVENT_VALID_OUTCOMES


def compute_true_reaction_time(
    trial, ni_events: Dict[str, Any], trial_idx: int, shift_fa_hit_ms: float = 200.0
) -> Optional[float]:
    """Replicates the notebook logic to compute a trial's true reaction time (absolute time).

    Returns None if it cannot be computed.
    """
    outcome = (
        getattr(trial, "trialoutcome", None)
        if not isinstance(trial, dict)
        else trial.get("trialoutcome", None)
    )
    reactiontimes = (
        getattr(trial, "reactiontimes", None)
        if not isinstance(trial, dict)
        else trial.get("reactiontimes", {})
    )
    if reactiontimes is None:
        reactiontimes = {}

    # Baseline_ON reference
    if "Baseline_ON" in ni_events:
        baseline_on = ni_events["Baseline_ON"]
        baseline_on_times = (
            np.array(baseline_on).flatten()
            if not (isinstance(baseline_on, dict) and "rise_t" in baseline_on)
            else np.array(baseline_on["rise_t"]).flatten()
        )
        if trial_idx < len(baseline_on_times):
            t0 = baseline_on_times[trial_idx]
        else:
            return None
    else:
        return None

    # Change_ON
    if "Change_ON" in ni_events:
        change_on = ni_events["Change_ON"]
        change_on_times = (
            np.array(change_on).flatten()
            if not (isinstance(change_on, dict) and "rise_t" in change_on)
            else np.array(change_on["rise_t"]).flatten()
        )
        t_change = (
            change_on_times[trial_idx] if trial_idx < len(change_on_times) else None
        )
        # If Change_ON entry is NaN or None, try to fallback to trial.change_time.
        # If baseline ON is available, prefer baseline_on_times[trial_idx] + trial.change_time
        try:
            if t_change is not None and np.isnan(t_change):
                t_change = None
        except Exception:
            pass
        if t_change is None:
            # try trial.change_time if available
            if isinstance(trial, dict):
                ct = trial.get("change_time", None)
            else:
                ct = getattr(trial, "change_time", None)
            try:
                if ct is not None and not np.isnan(float(ct)):
                    ct_val = float(ct)
                    # If baseline_on_times available, use baseline + ct
                    try:
                        if "Baseline_ON" in ni_events:
                            baseline_on = ni_events["Baseline_ON"]
                            baseline_on_times = (
                                np.array(baseline_on).flatten()
                                if not (isinstance(baseline_on, dict) and "rise_t" in baseline_on)
                                else np.array(baseline_on["rise_t"]).flatten()
                            )
                            if trial_idx < len(baseline_on_times) and not np.isnan(baseline_on_times[trial_idx]):
                                t_change = float(baseline_on_times[trial_idx] + ct_val)
                                # done
                    except Exception:
                        pass
                    if t_change is None:
                        t_change = ct_val
            except Exception:
                pass
    else:
        t_change = None

    shift = 0.0
    if outcome in ["FA", "Hit"]:
        shift = shift_fa_hit_ms / 1000.0

    if outcome == "Hit":
        rt = reactiontimes.get("RT", np.nan)
        if not np.isnan(rt) and t_change is not None:
            return float(t_change + rt - shift)
        else:
            return None
    elif outcome == "Miss":
        rt = reactiontimes.get("Miss", np.nan)
        if not np.isnan(rt) and t_change is not None:
            return float(t_change + rt)
        else:
            return None
    elif outcome in ["FA", "abort"]:
        rt = reactiontimes.get(outcome, np.nan)
        if not np.isnan(rt):
            return float(t0 + rt - shift) if outcome == "FA" else float(t0 + rt)
        else:
            return None
    else:
        return None


def get_event_times(
    session, event_name: str, outcomes: Optional[List[str]] = None,
    enforce_valid_outcomes: bool = True,
) -> List[float]:
    """Get event times for alignment.

    For event_name in ['Baseline_ON', 'Change_ON'] this returns the per-trial event times
    from session.ni_events. For behavioral outcomes like 'Hit', 'FA', it computes reaction times
    using compute_true_reaction_time across trials.

    Safety
    ------
    When ``enforce_valid_outcomes=True`` (default), the function automatically
    applies the outcome filter from ``EVENT_VALID_OUTCOMES`` for known events
    (e.g. Change_ON → hit/miss only).  An explicit ``outcomes`` parameter
    overrides the automatic filter.  Pass ``enforce_valid_outcomes=False``
    to disable automatic filtering entirely.
    """
    # Resolve outcome filter: explicit > auto > none
    if outcomes is not None:
        # Caller explicitly provided outcomes — use them
        _outcome_filter: Optional[Set[str]] = set(o.lower() for o in outcomes)
    elif enforce_valid_outcomes and event_name in EVENT_VALID_OUTCOMES:
        _outcome_filter = EVENT_VALID_OUTCOMES[event_name]  # may be None (e.g. Baseline_ON)
    else:
        _outcome_filter = None

    ni_events = getattr(session, "ni_events", {}) or {}
    trials = getattr(session, "trials", []) or []

    if event_name in ["Baseline_ON", "Change_ON"]:
        ev = ni_events.get(event_name, [])
        if isinstance(ev, dict) and "rise_t" in ev:
            arr = np.array(ev["rise_t"]).flatten()
        else:
            arr = np.array(ev).flatten()
        try:
            nan_mask = np.isnan(arr)
        except Exception:
            arr = arr.astype(float)
            nan_mask = np.isnan(arr)

        if event_name == "Change_ON":
            # attempt to obtain baseline times to convert per-trial change_time
            baseline = ni_events.get("Baseline_ON", None)
            if isinstance(baseline, dict) and "rise_t" in baseline:
                baseline_times = np.array(baseline["rise_t"]).flatten()
            else:
                baseline_times = np.array(baseline).flatten() if baseline is not None else np.array([])

            n_fill = min(len(arr), len(trials))
            for idx in range(n_fill):
                if nan_mask[idx]:
                    # SAFETY: Only fill NaN Change_ON times for valid outcomes.
                    # On FA/abort trials the change was never presented, so filling
                    # from trial.change_time would create a scientifically invalid
                    # alignment reference.
                    if _outcome_filter is not None:
                        t_obj = trials[idx]
                        oc = getattr(t_obj, "trialoutcome", "").lower() if not isinstance(t_obj, dict) else t_obj.get("trialoutcome", "").lower()
                        if oc not in _outcome_filter:
                            continue
                    t = trials[idx]
                    ct = None
                    if isinstance(t, dict):
                        ct = t.get("change_time", None)
                    else:
                        ct = getattr(t, "change_time", None)
                    if ct is not None:
                        try:
                            ct_val = float(ct)
                        except Exception:
                            continue
                        # If we have a baseline time for this trial, assume ct is relative
                        if idx < len(baseline_times):
                            try:
                                if not np.isnan(baseline_times[idx]):
                                    arr[idx] = float(baseline_times[idx] + ct_val)
                                    continue
                            except Exception:
                                pass
                        # Otherwise, use ct as-is (best effort)
                        arr[idx] = ct_val

        # Apply outcome filter: NaN-out entries for invalid trial outcomes
        if _outcome_filter is not None:
            n_check = min(len(arr), len(trials))
            for idx in range(n_check):
                t_obj = trials[idx]
                oc = getattr(t_obj, "trialoutcome", "").lower() if not isinstance(t_obj, dict) else t_obj.get("trialoutcome", "").lower()
                if oc not in _outcome_filter:
                    arr[idx] = np.nan

        # Drop NaNs
        arr = arr[~np.isnan(arr)]

        return list(map(float, arr))

    # Generic NI event: if present as a key in ni_events, return its times (drop NaNs)
    if event_name in ni_events:
        ev = ni_events.get(event_name, [])
        if isinstance(ev, dict) and "rise_t" in ev:
            arr = np.array(ev["rise_t"]).flatten()
        else:
            arr = np.array(ev).flatten()
        try:
            arr = arr.astype(float)
        except Exception:
            arr = np.array(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        return list(map(float, arr))

    # Otherwise treat as behavioral outcome
    event_times = []
    for idx, t in enumerate(session.trials):
        if getattr(t, "trialoutcome", None) == event_name:
            et = compute_true_reaction_time(t, ni_events, idx)
            if et is not None and not np.isnan(et):
                event_times.append(float(et))
    return event_times


def get_event_times_by_trial(
    session,
    event_name: str,
    enforce_valid_outcomes: bool = True,
) -> List[float]:
    """Return per-trial event times aligned to trial indices.

    For NI events ('Baseline_ON', 'Change_ON') returns a list of length n_trials,
    where missing entries are NaN. For 'Change_ON', missing NI entries are
    filled from per-trial Trial.change_time, preferably converted to absolute
    seconds using Baseline_ON for that trial when available.

    For behavioral outcomes like 'Hit', 'Miss', 'FA', returns per-trial reaction
    times computed via compute_true_reaction_time for trials whose outcome matches;
    non-matching trials are NaN.

    Safety
    ------
    When ``enforce_valid_outcomes=True`` (default), entries for trials whose
    outcome is not in ``EVENT_VALID_OUTCOMES[event_name]`` are set to NaN.
    For Change_ON this means FA/abort trials get NaN (the change was never
    presented).  Pass ``enforce_valid_outcomes=False`` to disable.
    """
    import numpy as _np

    # Resolve outcome filter
    if enforce_valid_outcomes and event_name in EVENT_VALID_OUTCOMES:
        _outcome_filter: Optional[Set[str]] = EVENT_VALID_OUTCOMES[event_name]  # may be None
    else:
        _outcome_filter = None

    n_trials = len(getattr(session, "trials", []) or [])
    if n_trials == 0:
        return []

    ni_events = getattr(session, "ni_events", {}) or {}

    def _to_array(x):
        if isinstance(x, dict) and "rise_t" in x:
            return _np.array(x["rise_t"]).flatten()
        return _np.array(x).flatten() if x is not None else _np.array([])

    out = _np.full((n_trials,), _np.nan, dtype=float)

    if event_name in ["Baseline_ON", "Change_ON"]:
        ev = ni_events.get(event_name, None)
        arr = _to_array(ev)
        # prefill direct values
        m = min(len(arr), n_trials)
        if m > 0:
            out[:m] = arr[:m]
        if event_name == "Change_ON":
            # Fill NaNs using per-trial change_time, converting to absolute with Baseline_ON when possible
            baseline = ni_events.get("Baseline_ON", None)
            base_arr = _to_array(baseline)
            trials = getattr(session, "trials", []) or []
            for idx in range(n_trials):
                if _np.isnan(out[idx]):
                    # SAFETY: Only fill NaN Change_ON times for valid outcomes.
                    # On FA/abort trials the change was never presented.
                    if _outcome_filter is not None:
                        t_obj = trials[idx]
                        oc = getattr(t_obj, "trialoutcome", "").lower() if not isinstance(t_obj, dict) else t_obj.get("trialoutcome", "").lower()
                        if oc not in _outcome_filter:
                            continue
                    t = trials[idx]
                    ct = getattr(t, "change_time", None) if not isinstance(t, dict) else t.get("change_time", None)
                    if ct is None:
                        continue
                    try:
                        ct_val = float(ct)
                    except Exception:
                        continue
                    if idx < len(base_arr):
                        try:
                            if not _np.isnan(base_arr[idx]):
                                out[idx] = float(base_arr[idx] + ct_val)
                                continue
                        except Exception:
                            pass
                    # fallback to absolute ct if baseline not available
                    out[idx] = ct_val

        # Apply outcome filter: NaN-out entries for invalid trial outcomes
        if _outcome_filter is not None:
            trials = getattr(session, "trials", []) or []
            for idx in range(min(len(out), len(trials))):
                t_obj = trials[idx]
                oc = getattr(t_obj, "trialoutcome", "").lower() if not isinstance(t_obj, dict) else t_obj.get("trialoutcome", "").lower()
                if oc not in _outcome_filter:
                    out[idx] = _np.nan

        # return as list of floats (with NaNs retained)
        return out.tolist()

    # Generic NI event by trial: if key exists in ni_events, map to per-trial array
    if event_name in ni_events:
        ev = ni_events.get(event_name, None)
        arr = _to_array(ev)
        m = min(len(arr), n_trials)
        if m > 0:
            out[:m] = arr[:m]
        # retain NaNs for missing trials
        return out.tolist()

    # Behavioral outcomes: per-trial reaction time or NaN for non-matching trials
    for idx, t in enumerate(session.trials):
        if getattr(t, "trialoutcome", None) == event_name:
            et = compute_true_reaction_time(t, ni_events, idx)
            if et is not None:
                try:
                    val = float(et)
                    if not _np.isnan(val):
                        out[idx] = val
                except Exception:
                    pass
    return out.tolist()


def align_spikes_to_events(
    spike_times: np.ndarray,
    event_times: List[float],
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align spike times to event times and return a trials x bins count matrix and bin centers."""
    spike_times = np.array(spike_times).flatten()
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    trials_counts = []
    for et in event_times:
        aligned = spike_times - float(et)
        mask = (aligned >= window[0]) & (aligned <= window[1])
        counts, _ = np.histogram(aligned[mask], bins=bins)
        trials_counts.append(counts)
    if len(trials_counts) == 0:
        arr = np.empty((0, len(bins) - 1), dtype=float)
    else:
        arr = np.atleast_2d(np.array(trials_counts, dtype=float))
    # Convert to firing rate (Hz)
    arr = arr / float(bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    return arr, bin_centers


def compute_peth_for_session(
    session,
    event_name: str,
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.01,
    cluster_id_list: Optional[List[int]] = None,
    restrict_to_good: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """Compute PETHs for clusters in session aligned to event_name.

    By default (restrict_to_good=True), this function will use the session's canonical
    `good_and_stable_ids` if present, then fall back to `good_cluster_ids` if needed.
    If `restrict_to_good` is False, or the session has neither, all clusters in `session.clusters` are used.
    You can override the cluster list by passing `cluster_id_list`.

    Returns a dict keyed by cluster_id with {'peth': mean_psth (1D), 'trials_matrix': 2D, 'n_trials': int, 'bin_centers': array}
    """

    if cluster_id_list is None:
        if restrict_to_good:
            # Prefer good_and_stable_ids, then good_cluster_ids
            if getattr(session, "good_and_stable_ids", None):
                cluster_id_list = list(session.good_and_stable_ids)
                print("[align] Using good_and_stable_ids for cluster selection.")
            elif getattr(session, "good_cluster_ids", None):
                cluster_id_list = list(session.good_cluster_ids)
                print("[align] Using good_cluster_ids for cluster selection.")
            else:
                cluster_id_list = [c.cluster_id for c in session.clusters]
                print("[align] No good cluster list found; using all clusters.")
        else:
            cluster_id_list = [c.cluster_id for c in session.clusters]

    event_times = get_event_times(session, event_name, enforce_valid_outcomes=True)
    out = {}
    for c in session.clusters:
        if c.cluster_id not in cluster_id_list:
            continue
        trials_mat, bin_centers = align_spikes_to_events(
            c.spike_times, event_times, window=window, bin_size=bin_size
        )
        mean_psth = (
            np.mean(trials_mat, axis=0)
            if trials_mat.shape[0] > 0
            else np.zeros(len(bin_centers))
        )
        out[int(c.cluster_id)] = {
            "peth": mean_psth,
            "trials_matrix": trials_mat,
            "n_trials": int(trials_mat.shape[0]),
            "bin_centers": bin_centers,
        }
    return out


def compute_and_cache_peth(
    session,
    event_name: str,
    out_h5_path: str,
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.01,
    sigma: Optional[float] = None,
    cluster_id_list: Optional[List[int]] = None,
    restrict_to_good: bool = True,
):
    """Compute PETHs and save to HDF5 cache. If sigma is provided, smooth mean PSTH with gaussian filter.
    Returns the HDF5 path written.

    By default, prefers session.good_and_stable_ids, then good_cluster_ids, for cluster selection.
    """
    path = Path(out_h5_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    peths = compute_peth_for_session(
        session,
        event_name,
        window=window,
        bin_size=bin_size,
        cluster_id_list=cluster_id_list,
        restrict_to_good=restrict_to_good,
    )
    with h5py.File(str(path), "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["event_name"] = event_name
        meta.attrs["window"] = json_serialize(window)
        meta.attrs["bin_size"] = float(bin_size)
        data_grp = h5.create_group("data")
        for cid, info in peths.items():
            grp = data_grp.create_group(str(cid))
            grp.create_dataset("peth", data=info["peth"])
            grp.create_dataset("bin_centers", data=info["bin_centers"])
            grp.attrs["n_trials"] = int(info["n_trials"])
            if sigma is not None and info["peth"].size > 0:
                smooth = gaussian_filter1d(info["peth"], sigma=sigma)
                grp.create_dataset("peth_smoothed", data=smooth)
    return str(path)


def json_serialize(x):
    try:
        import json as _json

        return _json.dumps(x)
    except Exception:
        return str(x)


if __name__ == "__main__":
    print(
        "align module: import and use compute_peth_for_session(session, event_name, window, bin_size)"
    )
