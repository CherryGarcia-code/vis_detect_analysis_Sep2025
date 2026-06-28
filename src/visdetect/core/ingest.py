"""Raw-data ingest: build Session objects directly from raw/processed files.

Replaces the MATLAB loadSessionNPX_main.m pipeline by reading behavioral
JSON, NI-DAQ .mat events, and Kilosort .npy outputs directly in Python.

Functions
---------
- load_behavioral_trials: Parse behavioral JSON files into Trial objects.
- load_kilosort_spikes: Load spike data from Kilosort output directory.
- load_ni_events: Load NI-DAQ events from processed .mat file.
- build_session_from_raw: Orchestrate all loaders into a Session.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io

from .session import Cluster, Session, Trial
from .qc import find_good_stable_units

logger = logging.getLogger(__name__)


# ── Behavioral data ──────────────────────────────────────────────────

def extract_stim_timeseries(raw_trial: dict) -> dict:
    """Pull the per-frame stimulus log (phase, displayed TF, vbl flip times)
    from a raw trials.json trial dict. Returns None for any absent key."""
    def _arr(key, ncol=None):
        v = raw_trial.get(key)
        if v is None:
            return None
        a = np.asarray(v, dtype=np.float64)
        if ncol is not None and (a.ndim != 2 or a.shape[1] != ncol):
            a = a.reshape(-1, ncol)
        return a
    return {
        "stim_phase": _arr("phase", ncol=2),
        "stim_tf_disp": _arr("TF"),
        "stim_vbl": _arr("vbl"),
    }


def load_behavioral_trials(
    raw_session_dir: Path,
) -> Tuple[List[Trial], Optional[dict], Optional[dict]]:
    """Load behavioral trials from JSON files in the Session subdirectory.

    Handles multiple runs per session (sorted by filename timestamp).

    Parameters
    ----------
    raw_session_dir : Path
        Path to the raw session folder (e.g., ``Raw data/BG_046_01072025``).

    Returns
    -------
    trials : list of Trial
    session_settings : dict or None
    computer_settings : dict or None
    """
    session_dir = raw_session_dir / "Session"
    if not session_dir.exists():
        logger.warning("No Session directory in %s", raw_session_dir)
        return [], None, None

    # Find and sort trial files by filename (contains timestamp)
    trial_files = sorted(session_dir.glob("*trials.json"))
    if not trial_files:
        logger.warning("No *trials.json files in %s", session_dir)
        return [], None, None

    # Read session/computer settings from first run
    session_settings = None
    computer_settings = None
    settings_files = sorted(session_dir.glob("*session_settings.json"))
    if settings_files:
        with open(settings_files[0], "r", encoding="utf-8") as f:
            session_settings = json.load(f)
    comp_files = sorted(session_dir.glob("*computer_settings.json"))
    if comp_files:
        with open(comp_files[0], "r", encoding="utf-8") as f:
            computer_settings = json.load(f)

    # Concatenate trials across all runs
    all_trials_raw = []
    for tf in trial_files:
        with open(tf, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            all_trials_raw.extend(data)
        elif isinstance(data, dict):
            # Single trial wrapped in dict
            all_trials_raw.append(data)

    # Convert to Trial dataclass instances
    trials = []
    for t in all_trials_raw:
        # Outcome string — preserve as-is from JSON (Hit, Miss, FA, Ref, abort)
        outcome = t.get("trialoutcome", "")

        # Reaction times dict
        rt = t.get("reactiontimes", {})
        if not isinstance(rt, dict):
            rt = {}

        # Baseline TF fluctuation vector
        bv_raw = t.get("St1TrialVector")
        baseline_values = np.array(bv_raw, dtype=np.float64) if bv_raw is not None else None

        # Extract per-frame stimulus log (phase, TF, vbl)
        stim = extract_stim_timeseries(t)

        trials.append(Trial(
            trialoutcome=outcome,
            reactiontimes=rt,
            change_size=t.get("Stim2TF"),
            orientation=t.get("Stim2Ori"),
            ITI=t.get("stimD"),
            change_time=t.get("stimT"),
            baseline_values=baseline_values,
            stim_phase=stim["stim_phase"],
            stim_tf_disp=stim["stim_tf_disp"],
            stim_vbl=stim["stim_vbl"],
        ))

    logger.info(
        "Loaded %d trials from %d JSON file(s) in %s",
        len(trials), len(trial_files), raw_session_dir.name,
    )
    return trials, session_settings, computer_settings


# ── Kilosort spike data ─────────────────────────────────────────────

def _parse_params_py(params_path: Path) -> dict:
    """Parse a Kilosort params.py file into a dict."""
    result = {}
    with open(params_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" not in line or line.startswith("#"):
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            # Try numeric
            try:
                result[key] = int(val)
                continue
            except ValueError:
                pass
            try:
                result[key] = float(val)
                continue
            except ValueError:
                pass
            # Boolean
            if val in ("True", "true"):
                result[key] = True
                continue
            if val in ("False", "false"):
                result[key] = False
                continue
            # Strip quotes for strings
            if val.startswith("'") and val.endswith("'"):
                val = val[1:-1]
            elif val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            result[key] = val
    return result


def _read_cluster_labels(ks_dir: Path) -> Tuple[List[int], set]:
    """Read cluster quality labels from KSLabel or group TSV files.

    Returns
    -------
    good_ids : list of int
        Cluster IDs labeled "good".
    noise_ids : set of int
        Cluster IDs labeled "noise" (to be excluded).
    """
    good_ids = []
    noise_ids = set()

    # Prefer cluster_KSLabel.tsv (Kilosort auto labels)
    for tsv_name in ("cluster_KSLabel.tsv", "cluster_group.tsv"):
        tsv_path = ks_dir / tsv_name
        if not tsv_path.exists():
            continue
        with open(tsv_path, "r", encoding="utf-8") as f:
            header = f.readline()  # skip header
            label_is_ks = "KSLabel" in header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                try:
                    cid = int(parts[0])
                except ValueError:
                    continue
                label = parts[1].strip()
                if label == "good":
                    good_ids.append(cid)
                elif label == "noise":
                    noise_ids.add(cid)
        break  # use only the first found file

    return sorted(good_ids), noise_ids


def load_kilosort_spikes(ks_dir: Path) -> dict:
    """Load spike data from a Kilosort output directory.

    Prefers TPrime-corrected spike times (spike_times_sec_adj.npy).

    Parameters
    ----------
    ks_dir : Path
        Kilosort/Phy output directory for one probe.

    Returns
    -------
    dict with keys:
        spike_times : ndarray float64, seconds
        spike_clusters : ndarray int
        good_cluster_ids : list of int
        sample_rate : float
    """
    # Parse params.py for sample_rate
    params = {}
    params_path = ks_dir / "params.py"
    if params_path.exists():
        params = _parse_params_py(params_path)
    sample_rate = float(params.get("sample_rate", 30000))

    # Load spike times — prefer TPrime-corrected
    adj_path = ks_dir / "spike_times_sec_adj.npy"
    sec_path = ks_dir / "spike_times_sec.npy"
    raw_path = ks_dir / "spike_times.npy"

    if adj_path.exists():
        spike_times = np.load(adj_path).flatten().astype(np.float64)
        logger.info("Using TPrime-corrected spike times (spike_times_sec_adj.npy)")
    elif sec_path.exists():
        spike_times = np.load(sec_path).flatten().astype(np.float64)
        logger.info("Using spike_times_sec.npy (no TPrime correction)")
    elif raw_path.exists():
        raw_samples = np.load(raw_path).flatten()
        spike_times = raw_samples.astype(np.float64) / sample_rate
        logger.info("Using spike_times.npy / sample_rate (no TPrime correction)")
    else:
        raise FileNotFoundError(f"No spike times file found in {ks_dir}")

    # Load spike cluster assignments
    for sc_name in ("spike_clusters.npy", "spike_clusters_ks.npy"):
        sc_path = ks_dir / sc_name
        if sc_path.exists():
            spike_clusters = np.load(sc_path).flatten()
            break
    else:
        raise FileNotFoundError(f"No spike_clusters.npy found in {ks_dir}")

    # Read quality labels
    good_cluster_ids, noise_ids = _read_cluster_labels(ks_dir)

    # Exclude noise clusters
    if noise_ids:
        mask = np.isin(spike_clusters, list(noise_ids), invert=True)
        spike_times = spike_times[mask]
        spike_clusters = spike_clusters[mask]

    return {
        "spike_times": spike_times,
        "spike_clusters": spike_clusters,
        "good_cluster_ids": good_cluster_ids,
        "sample_rate": sample_rate,
    }


# ── NI-DAQ events ───────────────────────────────────────────────────

def load_ni_events(processed_session_dir: Path) -> dict:
    """Load NI-DAQ event data from the Nidaq subdirectory.

    Parameters
    ----------
    processed_session_dir : Path
        Path to the processed session folder (e.g., ``Processed data/BG_046_01072025``).

    Returns
    -------
    dict
        Event name -> event data (numpy arrays or nested structures).
    """
    nidaq_dir = processed_session_dir / "Nidaq"
    if not nidaq_dir.exists():
        raise FileNotFoundError(f"No Nidaq directory in {processed_session_dir}")

    mat_files = list(nidaq_dir.glob("*NIdaq_events.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No *NIdaq_events.mat file in {nidaq_dir}")

    mat_data = scipy.io.loadmat(
        str(mat_files[0]), squeeze_me=True, struct_as_record=False
    )
    ni_struct = mat_data["NIdaq_events"]

    ni_events: Dict[str, Any] = {}
    for field in ni_struct._fieldnames:
        val = getattr(ni_struct, field)

        if isinstance(val, str):
            # session_name: store as 1-element object array to match existing .pkl
            ni_events[field] = np.array([val], dtype=object)

        elif hasattr(val, "_fieldnames"):
            # Sub-struct (e.g., Synch with rise_t, fall_t, duration)
            if "rise_t" in val._fieldnames:
                rise_t = np.asarray(getattr(val, "rise_t")).flatten()
                ni_events[field] = rise_t
            elif field == "frame_times_tr":
                # Nested struct with 'time' and 'delayed_frames_numb'
                ni_events[field] = _convert_frame_times_tr(val)
            else:
                # Generic sub-struct — extract rise_t if available, else flatten
                rise_t = getattr(val, "rise_t", None)
                if rise_t is not None:
                    ni_events[field] = np.asarray(rise_t).flatten()
                else:
                    ni_events[field] = np.asarray(val).flatten()
        else:
            ni_events[field] = np.asarray(val).flatten()

    return ni_events


def _convert_frame_times_tr(ft_struct) -> np.ndarray:
    """Convert frame_times_tr MATLAB struct to match existing .pkl format.

    Returns a 1-element object array containing a dict with:
      - 'time': object array of variable-length float64 arrays (per trial)
      - 'delayed_frames_numb': float64 array
    """
    result = {}

    if hasattr(ft_struct, "time"):
        time_raw = getattr(ft_struct, "time")
        time_arr = np.asarray(time_raw, dtype=object)
        if time_arr.ndim == 0:
            time_arr = np.array([time_arr.item()], dtype=object)
        # Ensure each element is a float64 array
        for i in range(len(time_arr)):
            time_arr[i] = np.asarray(time_arr[i], dtype=np.float64).flatten()
        result["time"] = time_arr

    if hasattr(ft_struct, "delayed_frames_numb"):
        dfn = getattr(ft_struct, "delayed_frames_numb")
        result["delayed_frames_numb"] = np.asarray(dfn, dtype=np.float64).flatten()

    # Wrap in 1-element object array to match existing format
    out = np.empty(1, dtype=object)
    out[0] = result
    return out


# ── Session builder ──────────────────────────────────────────────────

def _find_probe_folder(ks_phy_dir: Path) -> Optional[Path]:
    """Find the probe folder inside Kilosort&Phy, matching MATLAB logic.

    Excludes entries starting with '.' and folders named 'Sorted'.
    Also excludes files (non-directories) like ct_offsets.txt.
    """
    if not ks_phy_dir.exists():
        return None
    candidates = [
        d for d in ks_phy_dir.iterdir()
        if d.is_dir()
        and not d.name.startswith(".")
        and "Sorted" not in d.name
        # Match the MATLAB pattern: contains 'imec' or similar probe identifier
    ]
    if not candidates:
        return None
    # Return first match (typically only one probe for this project)
    return candidates[0]


def _parse_session_identity(session_folder_name: str) -> Tuple[str, str]:
    """Parse subject and session date from folder name.

    E.g., 'BG_046_01072025' -> ('BG_046', '01072025')
    """
    parts = session_folder_name.split("_")
    if len(parts) >= 3:
        subject = f"{parts[0]}_{parts[1]}"
        session_name = parts[2]
    else:
        subject = session_folder_name
        session_name = "unknown"
    return subject, session_name


def build_session_from_raw(
    raw_root: Path,
    processed_root: Path,
    session_folder_name: str,
    *,
    keep_all_good: bool = False,
) -> Session:
    """Build a Session object directly from raw and processed data files.

    This replaces the MATLAB loadSessionNPX_main.m pipeline.

    Parameters
    ----------
    raw_root : Path
        Root directory for raw data (contains session subdirectories).
    processed_root : Path
        Root directory for processed data (Kilosort, NI-DAQ).
    session_folder_name : str
        Name of the session folder (e.g., 'BG_046_01072025').
    keep_all_good : bool
        If False (default), only good_and_stable clusters are stored (matching
        existing .pkl format). If True, all KS-good clusters are stored.

    Returns
    -------
    Session
    """
    raw_dir = raw_root / session_folder_name
    processed_dir = processed_root / session_folder_name

    # 1. Behavioral trials
    trials, session_settings, computer_settings = load_behavioral_trials(raw_dir)

    # 2. NI-DAQ events
    ni_events = load_ni_events(processed_dir)

    # 3. Kilosort spike data
    ks_phy_dir = processed_dir / "Kilosort&Phy"
    probe_dir = _find_probe_folder(ks_phy_dir)
    if probe_dir is None:
        raise FileNotFoundError(
            f"No probe folder found in {ks_phy_dir}"
        )

    # KS4 sometimes writes outputs into a kilosort4/ subfolder; on this dataset
    # the spike files live directly in the probe folder. Use kilosort4/ only if
    # it actually contains a spike-times file, otherwise fall back to probe_dir
    # (a kilosort4/ folder can exist yet be empty of spike data).
    _spike_names = ("spike_times_sec_adj.npy", "spike_times_sec.npy", "spike_times.npy")
    ks_output_dir = probe_dir / "kilosort4"
    if not any((ks_output_dir / n).exists() for n in _spike_names):
        ks_output_dir = probe_dir

    ks_data = load_kilosort_spikes(ks_output_dir)
    spike_times = ks_data["spike_times"]
    spike_clusters = ks_data["spike_clusters"]
    good_cluster_ids = ks_data["good_cluster_ids"]

    # 4. Build Cluster objects for ALL non-noise clusters
    unique_ids = np.unique(spike_clusters)
    # Pre-sort for fast per-cluster slicing
    order = np.argsort(spike_clusters)
    sorted_sc = spike_clusters[order]
    left = np.searchsorted(sorted_sc, unique_ids, side="left")
    right = np.searchsorted(sorted_sc, unique_ids, side="right")

    all_clusters = []
    for i, cid in enumerate(unique_ids):
        idx = order[left[i]:right[i]]
        times = np.sort(spike_times[idx])
        all_clusters.append(
            Cluster(cluster_id=int(cid), spike_times=times, quality=None)
        )

    # 5. Stability filter
    good_and_stable_ids = find_good_stable_units(all_clusters, good_cluster_ids)
    logger.info(
        "Stability filter: %d / %d good clusters are stable",
        len(good_and_stable_ids), len(good_cluster_ids),
    )

    # 6. Trim clusters based on keep_all_good flag
    if keep_all_good:
        keep_set = set(good_cluster_ids)
    else:
        keep_set = set(good_and_stable_ids)
    clusters = [c for c in all_clusters if c.cluster_id in keep_set]

    # 7. Parse identity
    subject, session_name = _parse_session_identity(session_folder_name)

    return Session(
        trials=trials,
        clusters=clusters,
        subject=subject,
        session_name=session_name,
        good_cluster_ids=sorted(good_cluster_ids),
        good_and_stable_ids=sorted(good_and_stable_ids),
        ni_events=ni_events,
    )
