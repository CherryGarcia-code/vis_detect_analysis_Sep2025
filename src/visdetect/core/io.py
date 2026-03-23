"""I/O helpers for loading session MAT files and normalizing fields.

This module provides a safe loader for MATLAB session files (using
scipy.io.loadmat for v5-v7.2 and h5py for v7.3 HDF5 files) and small
helper utilities like `parse_good_cluster_ids` to normalize different
MAT -> Python representations.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import scipy.io
from .session import Session, Trial, Cluster


def mat_struct_to_dict(obj: Any):
    """Recursively convert MATLAB structs (loaded with scipy) into Python types.

    This preserves numpy arrays and numeric types, but converts object-dtype
    arrays and MATLAB structs into Python lists/dicts for easier consumption.
    """
    if isinstance(obj, np.ndarray):
        if obj.dtype == "O":
            return [mat_struct_to_dict(o) for o in obj]
        else:
            return obj
    elif hasattr(obj, "_fieldnames"):
        return {
            field: mat_struct_to_dict(getattr(obj, field)) for field in obj._fieldnames
        }
    else:
        return obj


def parse_good_cluster_ids(raw: Any) -> Optional[List[int]]:
    """Normalize `cluster_id_KS_good` field to a list[int] or None.

    Accepts numpy arrays, lists, scalars, bytes/strings (e.g. from MATLAB),
    and returns a sorted, deduplicated list of ints or None if none found.
    """
    if raw is None:
        return None

    # Try to convert to numpy array for uniform handling
    try:
        arr = np.asarray(raw)
    except Exception:
        # Fallback: single value attempt
        try:
            return [int(raw)]
        except Exception:
            return None

    # Flatten and handle empty
    try:
        arr = arr.flatten()
    except Exception:
        arr = np.array([arr])

    if arr.size == 0:
        return None

    out = []
    for v in arr:
        if v is None:
            continue
        # bytes/strings
        if isinstance(v, (bytes, str)):
            s = v.decode() if isinstance(v, bytes) else v
            s = s.strip()
            if s == "":
                continue
            try:
                out.append(int(float(s)))
            except Exception:
                continue
        else:
            # numeric-like
            try:
                if np.isnan(v):
                    continue
            except Exception:
                pass
            try:
                out.append(int(v))
            except Exception:
                continue

    if len(out) == 0:
        return None

    return sorted(set(out))


def load_mat_file_to_session(mat_path: str) -> Session:
    """Load a MATLAB session file and convert to a `Session` dataclass.

    Supports both older MATLAB formats (v5-v7.2 via scipy) and v7.3
    HDF5 files (via h5py). Tries scipy first; falls back to h5py.
    """
    try:
        return _load_mat_scipy(mat_path)
    except NotImplementedError:
        return _load_mat_h5py(mat_path)
    except Exception as exc:
        if "HDF" in str(exc) or "hdf" in str(exc):
            return _load_mat_h5py(mat_path)
        raise


def _load_mat_scipy(mat_path: str) -> Session:
    """Load a MATLAB v5-v7.2 session file via scipy.io.loadmat."""
    data = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    data = mat_struct_to_dict(data)
    if "data" in data:
        data_dict = mat_struct_to_dict(data["data"])
    elif "ans" in data:
        data_dict = mat_struct_to_dict(data["ans"])
    else:
        data_dict = data

    session_keys = list(data_dict.keys())
    session_key = session_keys[0] if session_keys else None
    subkey = list(data_dict[session_key].keys())[0] if session_key else None
    session_data = (
        data_dict[session_key][subkey] if session_key and subkey else data_dict
    )

    behav_data = session_data["behav_data"]
    trials_raw = behav_data["trials_data_exp"]
    trials = []
    for t in trials_raw:
        trial = Trial(
            trialoutcome=t.get("trialoutcome", ""),
            reactiontimes=t.get("reactiontimes", {}),
            change_size=t.get("Stim2TF", None),
            orientation=t.get("Stim2Ori", None),
            ITI=t.get("stimD", None),
            change_time=t.get("stimT", None),
            baseline_values=t.get("St1TrialVector", None),
        )
        trials.append(trial)

    npx_probes = session_data["NPX_probes"]
    cluster_ids = np.unique(npx_probes["clu"])
    clusters = []
    for clu in cluster_ids:
        spike_times = np.array(npx_probes["st"])[np.array(npx_probes["clu"]) == clu]
        cluster = Cluster(cluster_id=int(clu), spike_times=spike_times, quality=None)
        clusters.append(cluster)

    good_cluster_ids = None
    if "cluster_id_KS_good" in npx_probes:
        good_cluster_ids = parse_good_cluster_ids(npx_probes["cluster_id_KS_good"])

    good_and_stable_ids = None
    # Check for both potential naming conventions
    if "cluster_id_good_and_stable" in npx_probes:
        good_and_stable_ids = parse_good_cluster_ids(npx_probes["cluster_id_good_and_stable"])
    elif "cluster_id_KS_good_and_stable" in npx_probes:
        good_and_stable_ids = parse_good_cluster_ids(npx_probes["cluster_id_KS_good_and_stable"])

    ni_events_raw = session_data.get("NI_events", None)
    ni_events = mat_struct_to_dict(ni_events_raw) if ni_events_raw is not None else None
    session_name_str = (
        ni_events.get("session_name", "unknown") if ni_events else "unknown"
    )
    parts = session_name_str.split("_")
    subject = "_".join(parts[:2]) if len(parts) >= 3 else session_name_str
    session_name = parts[2] if len(parts) >= 3 else "unknown"

    return Session(
        trials=trials,
        clusters=clusters,
        subject=subject,
        session_name=session_name,
        good_cluster_ids=good_cluster_ids,
        good_and_stable_ids=good_and_stable_ids,
        ni_events=ni_events,
    )


# ---------------------------------------------------------------------------
# HDF5 (MATLAB v7.3) loader
# ---------------------------------------------------------------------------

def _h5_deref_scalar(f, ref):
    """Dereference an HDF5 object reference and return a Python scalar."""
    obj = f[ref]
    if isinstance(obj, h5py.Group):
        # Struct → return as dict of scalars
        out = {}
        for k in obj.keys():
            v = obj[k]
            if isinstance(v, h5py.Dataset):
                arr = v[()].flatten()
                out[k] = float(arr[0]) if arr.size == 1 else arr
        return out
    data = obj[()]
    if obj.dtype == np.uint16:
        return "".join(chr(c) for c in data.flatten())
    arr = data.flatten()
    if arr.size == 1:
        return float(arr[0])
    return arr


def _h5_read_ni_events(ni_group, f) -> Dict[str, Any]:
    """Read NI_events from an HDF5 group into a dict."""
    ni_events: Dict[str, Any] = {}
    for key in ni_group.keys():
        obj = ni_group[key]
        if isinstance(obj, h5py.Group):
            if "rise_t" in obj:
                ni_events[key] = obj["rise_t"][()].flatten()
            else:
                # Sub-struct (e.g. frame_times_tr) → read each field
                sub = {}
                for k2 in obj.keys():
                    v = obj[k2]
                    if isinstance(v, h5py.Dataset):
                        if v.dtype == object:
                            # Array of object references → dereference each
                            sub[k2] = np.array([
                                f[v[i, 0]][()].flatten()
                                for i in range(v.shape[0])
                            ], dtype=object)
                        else:
                            sub[k2] = v[()].flatten()
                ni_events[key] = sub
        elif isinstance(obj, h5py.Dataset):
            if obj.dtype == np.uint16:
                ni_events[key] = "".join(chr(c) for c in obj[()].flatten())
            else:
                ni_events[key] = obj[()].flatten()
    return ni_events


def _load_mat_h5py(mat_path: str) -> Session:
    """Load a MATLAB v7.3 (HDF5) session file via h5py."""
    import h5py as _h5py
    global h5py
    h5py = _h5py

    with h5py.File(mat_path, "r") as f:
        # Navigate: data / <subject> / <session_name> /
        root = f["data"] if "data" in f else f["ans"] if "ans" in f else f
        subject_key = [k for k in root.keys() if k != "#refs#"][0]
        session_key = list(root[subject_key].keys())[0]
        sd = root[subject_key][session_key]

        # --- Trials ---
        tde = sd["behav_data"]["trials_data_exp"]
        n_trials = tde["trialoutcome"].shape[0]

        trials = []
        for i in range(n_trials):
            trialoutcome = _h5_deref_scalar(f, tde["trialoutcome"][i, 0])
            reactiontimes = _h5_deref_scalar(f, tde["reactiontimes"][i, 0])
            if not isinstance(reactiontimes, dict):
                reactiontimes = {}
            change_size = _h5_deref_scalar(f, tde["Stim2TF"][i, 0])
            orientation = _h5_deref_scalar(f, tde["Stim2Ori"][i, 0])
            ITI = _h5_deref_scalar(f, tde["stimD"][i, 0])
            change_time = _h5_deref_scalar(f, tde["stimT"][i, 0])
            baseline_raw = _h5_deref_scalar(f, tde["St1TrialVector"][i, 0])
            baseline_values = baseline_raw if isinstance(baseline_raw, np.ndarray) else None

            trials.append(Trial(
                trialoutcome=str(trialoutcome) if trialoutcome else "",
                reactiontimes=reactiontimes,
                change_size=change_size,
                orientation=orientation,
                ITI=ITI,
                change_time=change_time,
                baseline_values=baseline_values,
            ))

        # --- Clusters ---
        npx = sd["NPX_probes"]
        all_clu = npx["clu"][()].flatten()
        all_st = npx["st"][()].flatten()
        cluster_ids = np.unique(all_clu)
        clusters = []
        for cid in cluster_ids:
            mask = all_clu == cid
            clusters.append(Cluster(
                cluster_id=int(cid),
                spike_times=all_st[mask],
                quality=None,
            ))

        # --- Good cluster IDs ---
        good_cluster_ids = None
        if "cluster_id_KS_good" in npx:
            good_cluster_ids = parse_good_cluster_ids(npx["cluster_id_KS_good"][()].flatten())

        good_and_stable_ids = None
        if "cluster_id_good_and_stable" in npx:
            good_and_stable_ids = parse_good_cluster_ids(npx["cluster_id_good_and_stable"][()].flatten())
        elif "cluster_id_KS_good_and_stable" in npx:
            good_and_stable_ids = parse_good_cluster_ids(npx["cluster_id_KS_good_and_stable"][()].flatten())

        # --- NI events ---
        ni_events = _h5_read_ni_events(sd["NI_events"], f)

        # --- Subject / session name ---
        session_name_str = ni_events.get("session_name", "unknown")
        if not isinstance(session_name_str, str):
            session_name_str = "unknown"
        parts = session_name_str.split("_")
        subject = "_".join(parts[:2]) if len(parts) >= 3 else session_name_str
        session_name = parts[2] if len(parts) >= 3 else "unknown"

    return Session(
        trials=trials,
        clusters=clusters,
        subject=subject,
        session_name=session_name,
        good_cluster_ids=good_cluster_ids,
        good_and_stable_ids=good_and_stable_ids,
        ni_events=ni_events,
    )
