
"""
Unified session dataclasses and I/O for vis_detect_analysis.

This module provides canonical dataclasses (Trial, Cluster, Session) and unified
load/save functions for both .pkl and .mat files, with backward compatibility for legacy pickles.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np
import pickle
from pathlib import Path
import scipy.io
import sys

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
    """Normalize an NI event entry to a 1D numpy array of times."""
    if x is None:
        return np.array([])
    if isinstance(x, dict) and "rise_t" in x:
        return np.array(x["rise_t"]).flatten()
    if hasattr(x, 'flatten'):
        return np.array(x).flatten()
    return np.array(x)

def load_session(path: str) -> Session:
    """Load a session from a pickle (.pkl) or MATLAB (.mat) file.
    Handles legacy pickles with class remapping if needed.
    """
    path = str(path)
    if path.endswith('.mat'):
        return load_mat_file_to_session(path)
    elif path.endswith('.pkl') or path.endswith('.pickle'):
        with open(path, 'rb') as f:
            # Define a custom unpickler that handles module renaming
            class RenamingUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # Map legacy 'src.visdetect...' to 'visdetect...'
                    if module.startswith('src.visdetect'):
                        module = module.replace('src.visdetect', 'visdetect')
                    
                    # Map legacy 'vis_detect_helpers_EPHYS_August2025' to 'visdetect.core.session'
                    if 'vis_detect_helpers_EPHYS_August2025' in module:
                        # If the class is one of our core dataclasses, redirect to this module
                        if name in {'Trial', 'Cluster', 'Session'}:
                            return globals()[name]
                        # Otherwise, try to map the module to visdetect.core.session and hope the class is there
                        module = 'visdetect.core.session'

                    # Also handle cases where classes might be top-level in the pickle but local here
                    if name in {'Trial', 'Cluster', 'Session'} and (module.endswith('core.session') or module == '__main__'):
                        return globals()[name]
                        
                    return super().find_class(module, name)

            try:
                obj = RenamingUnpickler(f).load()
            except Exception:
                # Fallback to standard load if custom fails (though custom should handle standard too)
                f.seek(0)
                obj = pickle.load(f)

        if isinstance(obj, Session):
            return obj
        # Optionally, add conversion logic if obj is dict-like
        raise ValueError(f"Unrecognized pickle format in {path}. Loaded type: {type(obj)}")
    else:
        raise ValueError(f"Unsupported file extension for {path}")

def save_session(session: Session, path: str):
    """Save a Session object to a pickle (.pkl) file."""
    with open(path, 'wb') as f:
        pickle.dump(session, f)

def load_mat_file_to_session(mat_path: str) -> Session:
    """Load a MATLAB session file and convert to a Session dataclass."""
    data = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    # Conversion logic here (placeholder)
    # You should implement conversion from MATLAB dict to Session
    raise NotImplementedError("MATLAB to Session conversion not implemented.")


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
