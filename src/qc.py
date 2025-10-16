"""QC utilities for sessions: trial-level and cluster-level checks.

Functions
---------
- run_qc(session, outdir): runs QC and writes JSON summary and a couple of PNG plots.
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _session_duration_from_spikes(session):
    starts = []
    ends = []
    for c in session.clusters:
        st = np.array(c.spike_times).flatten()
        if st.size > 0:
            starts.append(float(np.min(st)))
            ends.append(float(np.max(st)))
    if not starts:
        return 1.0
    start = min(starts)
    end = max(ends)
    dur = float(end - start) if end > start else 1.0
    return max(dur, 1e-3)


def cluster_qc_stats(cluster, session_duration: float) -> Dict[str, Any]:
    st = np.array(cluster.spike_times).flatten()
    n_spikes = int(st.size)
    mean_rate = float(n_spikes / session_duration) if session_duration > 0 else 0.0
    isi = np.diff(np.sort(st)) if st.size > 1 else np.array([])
    isi_violations = int(np.sum(isi < 0.002)) if isi.size > 0 else 0
    isi_frac = float(isi_violations / isi.size) if isi.size > 0 else 0.0
    return {
        "cluster_id": int(cluster.cluster_id),
        "n_spikes": n_spikes,
        "mean_rate_hz": mean_rate,
        "isi_violations_count": isi_violations,
        "isi_violations_frac": isi_frac,
    }


def trial_qc_stats(session) -> Dict[str, Any]:
    outcomes = {}
    missing_change_time = 0
    missing_rt = 0
    for t in session.trials:
        o = getattr(t, "trialoutcome", None) or (
            t.get("trialoutcome") if isinstance(t, dict) else None
        )
        outcomes[o] = outcomes.get(o, 0) + 1
        ct = (
            getattr(t, "change_time", None)
            if not isinstance(t, dict)
            else t.get("change_time", None)
        )
        if ct is None:
            missing_change_time += 1
        rt_dict = (
            getattr(t, "reactiontimes", None)
            if not isinstance(t, dict)
            else t.get("reactiontimes", None)
        )
        if not rt_dict:
            missing_rt += 1
    return {
        "n_trials": len(session.trials),
        "outcome_counts": outcomes,
        "missing_change_time": missing_change_time,
        "trials_missing_reactiontimes": missing_rt,
    }


def run_qc(session, outdir: str):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    sdur = _session_duration_from_spikes(session)

    # Cluster QC
    clusters_stats = [cluster_qc_stats(c, sdur) for c in session.clusters]

    # Trial QC
    trial_stats = trial_qc_stats(session)

    # Summary
    summary = {
        "subject": session.subject,
        "session_name": session.session_name,
        "n_clusters": len(session.clusters),
        "n_trials": len(session.trials),
        "n_good_clusters": len(session.good_cluster_ids)
        if session.good_cluster_ids
        else None,
        "session_duration_s": sdur,
    }

    # Write JSON files
    with (out / "qc_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (out / "clusters_qc.json").open("w") as f:
        json.dump(clusters_stats, f, indent=2)
    with (out / "trials_qc.json").open("w") as f:
        json.dump(trial_stats, f, indent=2)

    # Plots: firing rate histogram and ISI violation fraction scatter
    rates = np.array([c["mean_rate_hz"] for c in clusters_stats])
    plt.figure(figsize=(6, 4))
    plt.hist(rates, bins=50)
    plt.xlabel("Mean firing rate (Hz)")
    plt.ylabel("Number of clusters")
    plt.title("Cluster mean rate distribution")
    plt.tight_layout()
    plt.savefig(out / "cluster_mean_rate_hist.png", dpi=120)
    plt.close()

    isi_fracs = np.array([c["isi_violations_frac"] for c in clusters_stats])
    cluster_ids = np.array([c["cluster_id"] for c in clusters_stats])
    plt.figure(figsize=(6, 4))
    plt.scatter(cluster_ids, isi_fracs, s=6)
    plt.xlabel("Cluster ID")
    plt.ylabel("ISI violation fraction (<2 ms)")
    plt.title("ISI violation fraction per cluster")
    plt.tight_layout()
    plt.savefig(out / "isi_violation_scatter.png", dpi=120)
    plt.close()

    return {
        "summary_path": str(out / "qc_summary.json"),
        "clusters_qc_path": str(out / "clusters_qc.json"),
        "trials_qc_path": str(out / "trials_qc.json"),
        "plots": [
            str(out / "cluster_mean_rate_hist.png"),
            str(out / "isi_violation_scatter.png"),
        ],
    }


if __name__ == "__main__":
    print("qc module: import and call run_qc(session, outdir)")
