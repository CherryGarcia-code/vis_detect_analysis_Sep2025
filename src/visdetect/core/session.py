
"""
Unified session dataclasses and I/O for vis_detect_analysis.

This module provides canonical dataclasses (Trial, Cluster, Session) and unified
load/save functions for both .pkl and .mat files, with backward compatibility for legacy pickles.

History:
  Originally session.py handled only basic remapping. The legacy_io.py module had
  more elaborate handling for very old pickle formats. As of the project tidy (v5.1),
  all legacy handling has been consolidated here and legacy_io.py is archived.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from collections.abc import Mapping
import numpy as np
import pickle
from pathlib import Path

@dataclass
class Trial:
    trialoutcome: Optional[str] = None
    reactiontimes: Dict[str, float] = field(default_factory=dict)
    change_size: Optional[float] = None
    orientation: Optional[float] = None
    ITI: Optional[float] = None
    change_time: Optional[float] = None
    baseline_values: Optional[Any] = None
    n_seen: Optional[int] = None

@dataclass
class Cluster:
    cluster_id: int = -1
    spike_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    quality: Optional[str] = None

@dataclass
class Session:
    trials: List[Trial] = field(default_factory=list)
    clusters: List[Cluster] = field(default_factory=list)
    subject: Optional[str] = None
    session_name: Optional[str] = None
    good_cluster_ids: Optional[List[int]] = None
    good_and_stable_ids: Optional[List[int]] = None
    ni_events: Optional[Dict[str, Any]] = None

def _normalize_event_array(x):
    """Normalize an NI event entry to a 1D numpy array of times.

    The original data sometimes stores a dict with 'rise_t' or a MATLAB-like
    nested object. This function handles common shapes.
    """
    if x is None:
        return np.array([])
    if isinstance(x, dict) and "rise_t" in x:
        return np.array(x["rise_t"]).flatten()
    try:
        return np.array(x).flatten()
    except Exception:
        return np.array([])


def _convert_external_session(obj):
    """Convert a legacy Session-like object (dict or old-style object) into
    the canonical Session dataclass.

    Supports several shapes: a plain dict, or an object with attributes matching
    the older helper's Session fields. Handles legacy field names such as Stim2TF,
    Stim2Ori, stimD, stimT, St1TrialVector from MATLAB-era exports.
    """
    if isinstance(obj, Session):
        return obj

    if isinstance(obj, Mapping):
        data = dict(obj)
    else:
        data = {}
        for name in [
            "trials", "clusters", "subject", "session_name",
            "good_cluster_ids", "good_and_stable_ids", "ni_events",
        ]:
            if hasattr(obj, name):
                data[name] = getattr(obj, name)

    # --- Convert trials ---
    trials_out = []
    for t in data.get("trials", []):
        if isinstance(t, dict):
            _get = t.get
        else:
            _get = lambda k, d=None: getattr(t, k, d)

        trialoutcome = _get("trialoutcome")
        reactiontimes = _get("reactiontimes", {}) or {}
        change_size = _get("change_size") if _get("change_size") is not None else _get("Stim2TF")
        orientation = _get("orientation") if _get("orientation") is not None else _get("Stim2Ori")
        ITI = _get("ITI") if _get("ITI") is not None else _get("stimD")
        change_time = _get("change_time") if _get("change_time") is not None else _get("stimT")
        baseline_values = _get("baseline_values") if _get("baseline_values") is not None else _get("St1TrialVector")
        n_seen = _get("n_seen")

        trials_out.append(Trial(
            trialoutcome=trialoutcome,
            reactiontimes=reactiontimes or {},
            change_size=change_size,
            orientation=orientation,
            ITI=ITI,
            change_time=change_time,
            baseline_values=baseline_values,
            n_seen=n_seen,
        ))

    # --- Convert clusters ---
    clusters_out = []
    for c in data.get("clusters", []):
        if isinstance(c, dict):
            cid = int(c.get("cluster_id", -1))
            st = np.array(c.get("spike_times", []), dtype=float).flatten()
            quality = c.get("quality", None)
        else:
            cid = int(getattr(c, "cluster_id", -1))
            st = np.array(getattr(c, "spike_times", [])).flatten()
            quality = getattr(c, "quality", None)
        clusters_out.append(Cluster(cluster_id=cid, spike_times=st, quality=quality))

    good_ids = data.get("good_cluster_ids")
    if isinstance(good_ids, (list, tuple, np.ndarray)):
        good_ids = [int(x) for x in np.array(good_ids).flatten()]

    good_and_stable_ids = data.get("good_and_stable_ids")
    if good_and_stable_ids is None:
        good_and_stable_ids = data.get("cluster_id_good_and_stable")
    if isinstance(good_and_stable_ids, (list, tuple, np.ndarray)):
        good_and_stable_ids = [int(x) for x in np.array(good_and_stable_ids).flatten()]

    ni_events_raw = data.get("ni_events")
    ni_events = {}
    if ni_events_raw is not None:
        if isinstance(ni_events_raw, dict):
            items = ni_events_raw.items()
        else:
            items = getattr(ni_events_raw, "__dict__", {}).items()
        for k, v in items:
            ni_events[k] = _normalize_event_array(v)

    return Session(
        trials=trials_out,
        clusters=clusters_out,
        subject=data.get("subject"),
        session_name=data.get("session_name"),
        good_cluster_ids=good_ids,
        good_and_stable_ids=good_and_stable_ids,
        ni_events=ni_events,
    )


def _post_load_validate(session: Session) -> Session:
    """Normalize common fields after loading from any format."""
    if session.ni_events is None:
        session.ni_events = {}
    for k, v in list(session.ni_events.items()):
        session.ni_events[k] = _normalize_event_array(v)
    for c in session.clusters:
        if not isinstance(c.spike_times, np.ndarray):
            c.spike_times = np.array(c.spike_times).flatten()
    return session

def load_session(path: str) -> Session:
    """Load a session from a pickle (.pkl) or MATLAB (.mat) file.

    Handles legacy pickles with class remapping for all known historical
    module paths used in this project.
    """
    path = str(path)
    if path.endswith('.mat'):
        from .io import load_mat_file_to_session
        return load_mat_file_to_session(path)
    elif path.endswith('.pkl') or path.endswith('.pickle'):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Session file not found: {path}")

        with open(path, 'rb') as f:
            class RenamingUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # Map deprecated numpy module path (numpy >= 2.0)
                    if module.startswith('numpy._core'):
                        module = module.replace('numpy._core', 'numpy.core')

                    # Map legacy 'src.visdetect...' to 'visdetect...'
                    if module.startswith('src.visdetect'):
                        module = module.replace('src.visdetect', 'visdetect', 1)

                    # Bare 'src' module used in some very old pickles
                    if module == 'src' and name in ('Trial', 'Cluster', 'Session'):
                        return globals()[name]

                    # Strip 'src.' prefix on any remaining module
                    if module.startswith('src.'):
                        module = module[len('src.'):]

                    # Map legacy helper module
                    if 'vis_detect_helpers_EPHYS_August2025' in module:
                        if name in ('Trial', 'Cluster', 'Session'):
                            return globals()[name]
                        module = 'visdetect.core.session'

                    # Map legacy visdetect.session, bare visdetect, and legacy_io
                    if module in (
                        'visdetect.session',
                        'visdetect',
                        'visdetect.core.legacy_io',
                    ):
                        if name in ('Trial', 'Cluster', 'Session'):
                            return globals()[name]

                    # Handle __main__ or core.session references
                    if name in ('Trial', 'Cluster', 'Session') and (
                        module.endswith('core.session') or module == '__main__'
                    ):
                        return globals()[name]

                    return super().find_class(module, name)

            try:
                obj = RenamingUnpickler(f).load()
            except Exception:
                f.seek(0)
                obj = pickle.load(f)

        if isinstance(obj, Session):
            return _post_load_validate(obj)

        # Convert from legacy dict / old-style object
        session = _convert_external_session(obj)
        return _post_load_validate(session)
    else:
        raise ValueError(f"Unsupported file extension for {path}")

def save_session(session: Session, path: str):
    """Save a Session object to a pickle (.pkl) file."""
    with open(path, 'wb') as f:
        pickle.dump(session, f)

def session_summary(session: Session) -> Dict[str, Any]:
    """Return a JSON-serializable summary dict for a session."""
    return {
        "subject": session.subject,
        "session_name": session.session_name,
        "n_trials": len(session.trials),
        "n_clusters": len(session.clusters),
        "n_good_clusters": len(session.good_cluster_ids)
        if session.good_cluster_ids
        else None,
        "n_good_and_stable_clusters": len(session.good_and_stable_ids)
        if session.good_and_stable_ids
        else None,
        "ni_event_keys": list(session.ni_events.keys()) if session.ni_events else [],
    }
