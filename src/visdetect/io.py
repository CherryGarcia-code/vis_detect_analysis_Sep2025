"""I/O helpers for loading session MAT files and normalizing fields.

This module provides a safe loader for MATLAB session files (using
scipy.io.loadmat) and small helper utilities like `parse_good_cluster_ids`
to normalize different MAT -> Python representations.
"""
from typing import Any, List, Optional
import numpy as np
import scipy.io
from .session import Session, Trial, Cluster


def mat_struct_to_dict(obj: Any):
    """Recursively convert MATLAB structs (loaded with scipy) into Python types.

    This preserves numpy arrays and numeric types, but converts object-dtype
    arrays and MATLAB structs into Python lists/dicts for easier consumption.
    """
    if isinstance(obj, np.ndarray):
        if obj.dtype == 'O':
            return [mat_struct_to_dict(o) for o in obj]
        else:
            return obj
    elif hasattr(obj, '_fieldnames'):
        return {field: mat_struct_to_dict(getattr(obj, field)) for field in obj._fieldnames}
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
            if s == '':
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

    This function implements the same loading logic that existed in the
    legacy helper script, but returns the typed dataclasses in `session.py`.
    It intentionally leaves aggressive error handling to callers so issues
    are visible during development.
    """
    data = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    data = mat_struct_to_dict(data)
    if 'data' in data:
        data_dict = mat_struct_to_dict(data['data'])
    elif 'ans' in data:
        data_dict = mat_struct_to_dict(data['ans'])
    else:
        data_dict = data

    session_keys = list(data_dict.keys())
    session_key = session_keys[0] if session_keys else None
    subkey = list(data_dict[session_key].keys())[0] if session_key else None
    session_data = data_dict[session_key][subkey] if session_key and subkey else data_dict

    behav_data = session_data['behav_data']
    trials_raw = behav_data['trials_data_exp']
    trials = []
    for t in trials_raw:
        trial = Trial(
            trialoutcome=t.get('trialoutcome', ''),
            reactiontimes=t.get('reactiontimes', {}),
            change_size=t.get('Stim2TF', None),
            orientation=t.get('Stim2Ori', None),
            ITI=t.get('stimD', None),
            change_time=t.get('stimT', None),
            baseline_values=t.get('St1TrialVector', None)
        )
        trials.append(trial)

    npx_probes = session_data['NPX_probes']
    cluster_ids = np.unique(npx_probes['clu'])
    clusters = []
    for clu in cluster_ids:
        spike_times = np.array(npx_probes['st'])[np.array(npx_probes['clu']) == clu]
        cluster = Cluster(
            cluster_id=int(clu),
            spike_times=spike_times,
            quality=None
        )
        clusters.append(cluster)

    good_cluster_ids = None
    if 'cluster_id_KS_good' in npx_probes:
        good_cluster_ids = parse_good_cluster_ids(npx_probes['cluster_id_KS_good'])

    ni_events_raw = session_data.get('NI_events', None)
    ni_events = mat_struct_to_dict(ni_events_raw) if ni_events_raw is not None else None
    session_name_str = ni_events.get('session_name', 'unknown') if ni_events else 'unknown'
    parts = session_name_str.split('_')
    subject = '_'.join(parts[:2]) if len(parts) >= 3 else session_name_str
    session_name = parts[2] if len(parts) >= 3 else 'unknown'

    return Session(
        trials=trials,
        clusters=clusters,
        subject=subject,
        session_name=session_name,
        good_cluster_ids=good_cluster_ids,
        ni_events=ni_events
    )
