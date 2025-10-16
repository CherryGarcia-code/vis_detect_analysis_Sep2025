"""Event alignment and PETH utilities.

Functions:
- compute_true_reaction_time(trial, ni_events, trial_idx, shift_fa_hit_ms)
- get_event_times(session, event_name, outcomes=None)
- align_spikes_to_events(spike_times, event_times, window, bin_size)
- compute_peth_for_session(session, event_name, window, bin_size, good_cluster_ids=None)
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import h5py


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
    session, event_name: str, outcomes: Optional[List[str]] = None
) -> List[float]:
    """Get event times for alignment.

    For event_name in ['Baseline_ON', 'Change_ON'] this returns the per-trial event times
    from session.ni_events. For behavioral outcomes like 'Hit', 'FA', it computes reaction times
    using compute_true_reaction_time across trials.
    """
    ni_events = getattr(session, "ni_events", {}) or {}
    if event_name in ["Baseline_ON", "Change_ON"]:
        ev = ni_events.get(event_name, [])
        if isinstance(ev, dict) and "rise_t" in ev:
            arr = np.array(ev["rise_t"]).flatten()
        else:
            arr = np.array(ev).flatten()
        return list(map(float, arr))

    # Otherwise treat as behavioral outcome
    event_times = []
    for idx, t in enumerate(session.trials):
        if getattr(t, "trialoutcome", None) == event_name:
            et = compute_true_reaction_time(t, ni_events, idx)
            if et is not None and not np.isnan(et):
                event_times.append(float(et))
    return event_times


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
    good_cluster_ids: Optional[List[int]] = None,
    use_good_only: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """Compute PETHs for clusters in session aligned to event_name.

    By default (use_good_only=True) this function will prefer the session's canonical
    `good_cluster_ids` when present. If `use_good_only` is False, or the session
    has no `good_cluster_ids`, all clusters in `session.clusters` are used.

    Returns a dict keyed by cluster_id with {'peth': mean_psth (1D), 'trials_matrix': 2D, 'n_trials': int, 'bin_centers': array}
    """
    if good_cluster_ids is None:
        if use_good_only and getattr(session, "good_cluster_ids", None):
            good_cluster_ids = list(session.good_cluster_ids)
        else:
            good_cluster_ids = [c.cluster_id for c in session.clusters]

    event_times = get_event_times(session, event_name)
    out = {}
    for c in session.clusters:
        if c.cluster_id not in good_cluster_ids:
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
    good_cluster_ids: Optional[List[int]] = None,
    use_good_only: bool = True,
):
    """Compute PETHs and save to HDF5 cache. If sigma is provided, smooth mean PSTH with gaussian filter.
    Returns the HDF5 path written.
    """
    path = Path(out_h5_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    peths = compute_peth_for_session(
        session,
        event_name,
        window=window,
        bin_size=bin_size,
        good_cluster_ids=good_cluster_ids,
        use_good_only=use_good_only,
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
