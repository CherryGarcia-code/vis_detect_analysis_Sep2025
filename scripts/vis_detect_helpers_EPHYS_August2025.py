# session_utils.py

import pickle
import numpy as np

# Import the canonical dataclasses and IO helpers from the new package.
from visdetect.session import Trial, Cluster, Session
from visdetect.io import mat_struct_to_dict, parse_good_cluster_ids
import scipy.io


def save_session_to_file(session: Session, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(session, f)


def load_session_from_file(filename: str) -> Session:
    with open(filename, "rb") as f:
        session = pickle.load(f)
    return session


def load_mat_file_to_session(mat_path: str) -> Session:
    # This wrapper preserves the original behavior but delegates parsing to
    # shared helpers where possible.
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
        ni_events=ni_events,
    )
