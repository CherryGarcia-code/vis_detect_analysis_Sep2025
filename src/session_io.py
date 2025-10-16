"""Session I/O and small validation utilities.

Provides a stable Session dataclass, a loader that can read pickles produced
by the existing helper script (by making the scripts/ folder importable),
and light validation/normalization of common fields.

This is intentionally lightweight: downstream modules can import these
dataclasses and rely on numpy arrays / standard Python types.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
import pickle
import sys
import numpy as np


@dataclass
class Trial:
    trialoutcome: Optional[str] = None
    reactiontimes: Dict[str, float] = field(default_factory=dict)
    change_size: Optional[float] = None
    orientation: Optional[float] = None
    ITI: Optional[float] = None
    change_time: Optional[float] = None
    baseline_values: Optional[Any] = None


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
    ni_events: Optional[Dict[str, Any]] = None


def _normalize_event_array(x):
    """Normalize an NI event entry to a 1D numpy array of times.

    The original data sometimes stores a dict with 'rise_t' or a MATLAB-like
    nested object. This function handles common shapes.
    """
    if x is None:
        return np.array([])
    # If it's a dict containing rise_t, prefer that
    if isinstance(x, dict) and "rise_t" in x:
        return np.array(x["rise_t"]).flatten()
    # If already an array-like
    try:
        return np.array(x).flatten()
    except Exception:
        return np.array([])


def _convert_external_session(obj):
    """Convert the external Session-like object (from legacy helper) into
    the local Session dataclass.
    """
    # Support several shapes: either our local Session, a dict, or an object
    # with attributes named similarly to the older helper's Session.
    if isinstance(obj, Session):
        return obj
    # If it's a mapping (plain dict)
    from collections.abc import Mapping

    if isinstance(obj, Mapping):
        data = obj
    else:
        # Try to extract attributes
        data = {}
        for name in [
            "trials",
            "clusters",
            "subject",
            "session_name",
            "good_cluster_ids",
            "ni_events",
        ]:
            if hasattr(obj, name):
                data[name] = getattr(obj, name)

    # Convert trials
    trials_out = []
    for t in data.get("trials", []):
        # t may be a dict-like or object
        if isinstance(t, dict):
            trialoutcome = t.get("trialoutcome")
            reactiontimes = t.get("reactiontimes", {}) or {}
            change_size = (
                t.get("change_size", None)
                if t.get("change_size", None) is not None
                else t.get("Stim2TF", None)
            )
            orientation = (
                t.get("orientation", None)
                if t.get("orientation", None) is not None
                else t.get("Stim2Ori", None)
            )
            ITI = (
                t.get("ITI", None)
                if t.get("ITI", None) is not None
                else t.get("stimD", None)
            )
            change_time = (
                t.get("change_time", None)
                if t.get("change_time", None) is not None
                else t.get("stimT", None)
            )
            baseline_values = (
                t.get("baseline_values", None)
                if t.get("baseline_values", None) is not None
                else t.get("St1TrialVector", None)
            )
        else:
            trialoutcome = getattr(t, "trialoutcome", None)
            reactiontimes = getattr(t, "reactiontimes", {}) or {}
            # legacy keys: prefer explicit None checks to avoid numpy truthiness errors
            change_size = (
                getattr(t, "change_size", None)
                if getattr(t, "change_size", None) is not None
                else getattr(t, "Stim2TF", None)
            )
            orientation = (
                getattr(t, "orientation", None)
                if getattr(t, "orientation", None) is not None
                else getattr(t, "Stim2Ori", None)
            )
            ITI = (
                getattr(t, "ITI", None)
                if getattr(t, "ITI", None) is not None
                else getattr(t, "stimD", None)
            )
            change_time = (
                getattr(t, "change_time", None)
                if getattr(t, "change_time", None) is not None
                else getattr(t, "stimT", None)
            )
            baseline_values = (
                getattr(t, "baseline_values", None)
                if getattr(t, "baseline_values", None) is not None
                else getattr(t, "St1TrialVector", None)
            )
        trials_out.append(
            Trial(
                trialoutcome=trialoutcome,
                reactiontimes=reactiontimes or {},
                change_size=change_size,
                orientation=orientation,
                ITI=ITI,
                change_time=change_time,
                baseline_values=baseline_values,
            )
        )

    # Convert clusters
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

    good_ids = data.get("good_cluster_ids", None)
    if isinstance(good_ids, (list, tuple, np.ndarray)):
        good_ids = [int(x) for x in np.array(good_ids).flatten()]

    ni_events_raw = data.get("ni_events", None)
    ni_events = {}
    if ni_events_raw is not None:
        # Normalize common events to arrays
        # Handle either a plain dict (from our helper) or a MATLAB mat_struct-like
        if isinstance(ni_events_raw, dict):
            items = ni_events_raw.items()
        else:
            items = getattr(ni_events_raw, "__dict__", {}).items()
        for k, v in items:
            ni_events[k] = _normalize_event_array(v)

    subject = data.get("subject")
    session_name = data.get("session_name")

    return Session(
        trials=trials_out,
        clusters=clusters_out,
        subject=subject,
        session_name=session_name,
        good_cluster_ids=good_ids,
        ni_events=ni_events,
    )


def load_session(path: str) -> Session:
    """Load a session from a pickle file and normalize to the local Session dataclass.

    This function attempts to import the repository's `scripts/` folder so that
    pickles created with the legacy helper can be unpickled. It then converts
    the resulting object into the local dataclass representation.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Session file not found: {path}")

    # Make scripts/ importable if present
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if scripts_dir.exists():
        sys.path.insert(0, str(scripts_dir))

        with p.open("rb") as f:
            try:
                obj = pickle.load(f)
            except ModuleNotFoundError:
                # Fallback: some legacy pickles reference compiled submodules like
                # 'numpy._core' which may not exist in newer numpy builds. Try a
                # resilient unpickler that remaps that module path to 'numpy.core'.
                f.seek(0)

                class RenamingUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module.startswith("numpy._core"):
                            module = module.replace("numpy._core", "numpy.core")
                        return super().find_class(module, name)

                obj = RenamingUnpickler(f).load()

    # If the loaded object is already our Session, return
    try:
        if isinstance(obj, Session):
            return obj
    except Exception:
        pass

    # Convert external object into our Session dataclass
    session = _convert_external_session(obj)

    # Basic validation
    if session.ni_events is None:
        session.ni_events = {}
    # Ensure event arrays are numpy arrays
    for k, v in list(session.ni_events.items()):
        session.ni_events[k] = _normalize_event_array(v)

    # Ensure spike times are numpy arrays
    for c in session.clusters:
        if not isinstance(c.spike_times, np.ndarray):
            c.spike_times = np.array(c.spike_times).flatten()

    return session


def save_session(session: Session, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
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
        "ni_event_keys": list(session.ni_events.keys()) if session.ni_events else [],
    }


if __name__ == "__main__":
    print("session_io module - import this file and call load_session(path)")
