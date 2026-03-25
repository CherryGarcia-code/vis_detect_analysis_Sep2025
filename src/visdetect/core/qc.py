"""QC utilities for sessions: trial-level and cluster-level checks.

Functions
---------
- cluster_qc_stats(cluster, session_duration): QC statistics for a single cluster.
- trial_qc_stats(session): QC statistics for all trials in a session.
- run_qc(session, outdir): runs QC and writes JSON summary and a couple of PNG plots.
- apply_unit_filters(df, ...): apply threshold rules; returns a DataFrame with pass flags and 'keep'.
- load_qc_profile(name, ...): load a QC profile dict from YAML.
- read_kept_cluster_ids(path): return kept cluster IDs from a unit_selection.csv.
- find_good_stable_units(clusters, good_cluster_ids): stability filter (port of MATLAB).

Note
----
Higher-level functions that depend on spike-event alignment (``compute_unit_selection_table``,
``run_unit_selection``, ``plot_kept_vs_dropped_distributions``) live in
``visdetect.analysis.unit_selection`` to avoid a core -> analysis dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def _session_duration_from_spikes(session: Any) -> float:
    """Estimate session duration (seconds) from the span of spike times."""
    starts: list[float] = []
    ends: list[float] = []
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


def cluster_qc_stats(cluster: Any, session_duration: float) -> Dict[str, Any]:
    """Compute QC statistics for a single cluster."""
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


def trial_qc_stats(session: Any) -> Dict[str, Any]:
    """Compute QC statistics for all trials in a session."""
    outcomes: Dict[str, int] = {}
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


def run_qc(session: Any, outdir: str) -> Dict[str, Any]:
    """Run QC checks and save JSON summaries and diagnostic plots."""
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


# -------------------------------
# Unit filter helpers
# -------------------------------

def apply_unit_filters(
    df: pd.DataFrame,
    *,
    require_good_cluster: bool = True,
    min_total_spikes: int = 500,
    min_mean_rate_hz: float = 0.1,
    max_isi_viol_frac: float = 0.2,
    min_median_spikes_per_trial: float = 0.1,
    max_median_spikes_per_trial: Optional[float] = None,
) -> pd.DataFrame:
    """Add boolean pass flags per criterion and an overall 'keep' column.

    Defaults are conservative and intended as a starting point. Adjust as needed.
    """
    out = df.copy()
    # Handle possible None for is_good_cluster
    if "is_good_cluster" in out.columns and out["is_good_cluster"].notna().any():
        pass_good = out["is_good_cluster"].fillna(False)
    else:
        pass_good = pd.Series([True] * len(out), index=out.index)
        require_good_cluster = False  # no good list available

    out["pass_good"] = pass_good if require_good_cluster else True
    out["pass_total_spikes"] = out["n_spikes"] >= int(min_total_spikes)
    out["pass_mean_rate"] = out["mean_rate_hz"] >= float(min_mean_rate_hz)
    # isi_violations_frac may be NaN; treat NaN as pass (unknown) and leave for manual inspection
    out["pass_isi"] = out["isi_violations_frac"].fillna(0.0) <= float(max_isi_viol_frac)
    # median_spikes_per_trial may be NaN if no trials; require at least threshold
    out["pass_trial_content_min"] = out["median_spikes_per_trial"].fillna(0.0) >= float(min_median_spikes_per_trial)
    if max_median_spikes_per_trial is not None:
        out["pass_trial_content_max"] = out["median_spikes_per_trial"].fillna(0.0) <= float(max_median_spikes_per_trial)
    else:
        out["pass_trial_content_max"] = True

    cols = ["pass_good", "pass_total_spikes", "pass_mean_rate", "pass_isi", "pass_trial_content_min", "pass_trial_content_max"]
    out["keep"] = out[cols].all(axis=1)
    return out


def load_qc_profile(name: str, profiles_path: Optional[str] = None) -> Dict[str, Any]:
    """Load a QC profile dict by name from YAML (config/qc_profiles.yml by default).

    Returns an empty dict if file or profile is not found.
    """
    try:
        path = (
            Path(profiles_path)
            if profiles_path is not None
            else Path(__file__).resolve().parents[1] / "config" / "qc_profiles.yml"
        )
        if not path.exists():
            return {}
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}
        prof = data.get(name, {})
        if not isinstance(prof, dict):
            return {}
        return prof
    except Exception:
        return {}


def read_kept_cluster_ids(selection_csv_path: str) -> List[int]:
    """Return kept cluster IDs from a unit_selection.csv (expects a 'keep' column).

    If the file is missing or malformed, returns an empty list.
    """
    try:
        df = pd.read_csv(selection_csv_path)
        if "keep" not in df.columns or "cluster_id" not in df.columns:
            return []
        return (
            df.loc[df["keep"].astype(bool), "cluster_id"].dropna().astype(int).tolist()
        )
    except Exception:
        return []


# ── Stability filter (port of find_good_stable_units_PaperVersion.m) ──

def _movmean(x: np.ndarray, k: int) -> np.ndarray:
    """Moving mean matching MATLAB's movmean with 'shrink' endpoints.

    MATLAB's movmean uses a centered window that shrinks at the edges so
    that only available samples are averaged.  This replicates that behavior
    using a cumulative-sum approach (O(n), fully vectorized).
    """
    n = len(x)
    if k >= n:
        return np.full(n, x.mean())
    cs = np.concatenate(([0.0], np.cumsum(x)))
    half = k // 2
    # Build lo/hi index arrays for each position
    idx = np.arange(n)
    lo = np.maximum(0, idx - half)
    hi = np.minimum(n, idx - half + k)
    return (cs[hi] - cs[lo]) / (hi - lo)


def find_good_stable_units(clusters: list, good_cluster_ids: list) -> list[int]:
    """Keep only stable units from KS-good units.

    Python port of find_good_stable_units_PaperVersion.m
    (Khilkevich & Lohse 2024 criteria):
      - Average firing rate >= 0.5 Hz
      - Rate in 20 / 10 / 5-min sliding window never drops below
        30% / 20% / 10% of the session average
      - ISI distribution peak in first 5 ms is at >= 2 ms
      - ISI distribution is smooth (tallest bin < 4x second-tallest)

    Parameters
    ----------
    clusters : list of Cluster
        All clusters (not just good ones).
    good_cluster_ids : list of int
        Cluster IDs labeled "good" by Kilosort.

    Returns
    -------
    list of int
        Sorted list of cluster IDs that are both good and stable.
    """
    BIN_SEC = 0.01  # 10 ms bins, matching MATLAB
    # Window sizes in bins: 20 min, 10 min, 5 min
    WIN_20 = int(20 * 60 / BIN_SEC)  # 120 000
    WIN_10 = int(10 * 60 / BIN_SEC)  #  60 000
    WIN_5  = int( 5 * 60 / BIN_SEC)  #  30 000

    # Recording duration = latest spike across ALL clusters
    max_t = max(
        (c.spike_times[-1] if len(c.spike_times) > 0 else 0)
        for c in clusters
    )
    rec_dur = max_t if max_t > 0 else 1.0

    good_set = set(good_cluster_ids)
    stable_ids: list[int] = []

    for c in clusters:
        if c.cluster_id not in good_set:
            continue
        sp = c.spike_times
        if len(sp) < 2:
            continue

        avg_fr = len(sp) / rec_dur
        if avg_fr < 0.5:
            continue

        # ── Firing-rate stability ────────────────────────────
        n_bins = int(np.ceil(rec_dur / BIN_SEC))
        counts, _ = np.histogram(sp, bins=n_bins, range=(0, rec_dur))
        fr = counts.astype(np.float64) / BIN_SEC

        # movmean matching MATLAB's default 'shrink' endpoint behavior:
        # at the edges the window shrinks so only available samples are used.
        if n_bins >= WIN_5:  # need at least a 5-min recording
            fr_20 = _movmean(fr, min(WIN_20, n_bins))
            fr_10 = _movmean(fr, min(WIN_10, n_bins))
            fr_5  = _movmean(fr, min(WIN_5,  n_bins))

            if fr_20.min() < 0.3 * avg_fr:
                continue
            if fr_10.min() < 0.2 * avg_fr:
                continue
            if fr_5.min() < 0.1 * avg_fr:
                continue

        # ── ISI quality ──────────────────────────────────────
        isi = np.diff(sp)
        # 50 bins from 0-50 ms (edges: 0, 0.001, ..., 0.050)
        isi_counts, _ = np.histogram(isi, bins=np.arange(0, 0.051, 0.001))

        # Peak among first 5 bins must be at >= 2 ms
        # (MATLAB 1-indexed > 2 <-> Python 0-indexed >= 2)
        first5 = isi_counts[:5]
        if first5.max() == 0:
            continue
        peak_positions = np.where(first5 == first5.max())[0]
        if peak_positions.min() < 2:
            continue

        # Smoothness: tallest bin < 4x second-tallest (all 50 bins)
        top2 = np.sort(isi_counts)[-2:]  # [second, first]
        if top2[0] == 0:  # only one non-zero bin -> not smooth
            continue
        if top2[1] >= 4 * top2[0]:
            continue

        stable_ids.append(c.cluster_id)

    return sorted(stable_ids)


if __name__ == "__main__":
    print("qc module: import and call run_qc(session, outdir)")
