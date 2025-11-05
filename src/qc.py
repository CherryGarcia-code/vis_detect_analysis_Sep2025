"""QC utilities for sessions: trial-level and cluster-level checks, plus unit selection.

Functions
---------
- run_qc(session, outdir): runs QC and writes JSON summary and a couple of PNG plots.
- compute_unit_selection_table(session, ...): per-cluster metrics helpful for filtering.
- apply_unit_filters(df, ...): apply threshold rules; returns a DataFrame with pass flags and 'keep'.
- run_unit_selection(session, outdir, ...): convenience to compute, filter, and save CSVs.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

# Import alignment helpers for per-trial metrics
from src import align as align_mod


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


# -------------------------------
# Unit selection helpers
# -------------------------------

def compute_unit_selection_table(
    session,
    event_name: str = "Change_ON",
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.02,
) -> pd.DataFrame:
    """Compute per-cluster metrics to guide early filtering.

    Metrics:
      - cluster_id
      - n_spikes (total)
      - mean_rate_hz (approx total_spikes / session_duration)
      - isi_violations_frac (fraction of ISIs < 2 ms)
      - n_trials_used (for alignment to event)
      - median_spikes_per_trial (within the alignment window)
      - is_good_cluster (from session.good_cluster_ids)

    Notes:
      - Session duration is estimated from spike-time span if not inferable from NI events.
      - Trial-aligned metrics depend on the chosen event/window.
    """
    # Estimate session duration (seconds)
    sess_dur = None
    ni = getattr(session, "ni_events", {}) or {}
    # Prefer NI Baseline/Change timing to get an upper bound (+ small buffer)
    for k in ("Baseline_ON", "Change_ON"):
        if k in ni and np.asarray(ni[k]).size > 0:
            arr = np.asarray(ni[k]).flatten()
            try:
                sess_dur = float(np.nanmax(arr) + 10.0)
                break
            except Exception:
                pass
    if sess_dur is None:
        # Fallback to max spike time across clusters
        max_sp = 0.0
        for c in session.clusters:
            st = np.asarray(c.spike_times).flatten()
            if st.size:
                try:
                    max_sp = max(max_sp, float(np.nanmax(st)))
                except Exception:
                    pass
        sess_dur = max(max_sp, 1.0)

    good_ids = set(getattr(session, "good_cluster_ids", []) or [])
    event_times = align_mod.get_event_times(session, event_name)

    rows = []
    for c in session.clusters:
        cid = int(c.cluster_id)
        st = np.asarray(c.spike_times).flatten()
        n_spikes = int(st.size)
        mean_rate = float(n_spikes / sess_dur) if sess_dur > 0 else np.nan
        # ISI violations (< 2 ms)
        isi = np.diff(np.sort(st)) if st.size > 1 else np.array([])
        if isi.size > 0:
            isi_viol = float((isi < 0.002).sum()) / isi.size
        else:
            isi_viol = np.nan

        # Trial-aligned counts within window
        trials_mat, _ = align_mod.align_spikes_to_events(st, event_times, window=window, bin_size=bin_size)
        n_trials = int(trials_mat.shape[0]) if trials_mat is not None else 0
        spikes_per_trial = trials_mat.sum(axis=1) if n_trials > 0 else np.array([])
        med_spt = float(np.median(spikes_per_trial)) if spikes_per_trial.size > 0 else np.nan

        rows.append(
            {
                "cluster_id": cid,
                "n_spikes": n_spikes,
                "mean_rate_hz": mean_rate,
                "isi_violations_frac": isi_viol,
                "n_trials_used": n_trials,
                "median_spikes_per_trial": med_spt,
                "is_good_cluster": (cid in good_ids) if good_ids else None,
            }
        )

    df = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
    return df


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


def run_unit_selection(
    session,
    outdir: str,
    *,
    event_name: str = "Baseline_ON",
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.02,
    require_good_cluster: bool = True,
    min_total_spikes: int = 500,
    min_mean_rate_hz: float = 0.1,
    max_isi_viol_frac: float = 0.2,
    min_median_spikes_per_trial: float = 0.1,
    max_median_spikes_per_trial: Optional[float] = None,
    make_plots: bool = True,
    plot_metrics: Optional[List[str]] = None,
    profile: Optional[str] = None,
    profiles_path: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute unit metrics, apply filters, and save CSV tables.

    Returns dict with paths and the list of kept cluster IDs.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve parameters: defaults -> profile (if provided) -> explicit args/params
    used_params = {
        "require_good_cluster": require_good_cluster,
        "min_total_spikes": min_total_spikes,
        "min_mean_rate_hz": min_mean_rate_hz,
        "max_isi_viol_frac": max_isi_viol_frac,
        "min_median_spikes_per_trial": min_median_spikes_per_trial,
        "max_median_spikes_per_trial": max_median_spikes_per_trial,
        "event_name": event_name,
        "window": list(window),
        "bin_size": float(bin_size),
    }
    # Merge in profile
    if profile is not None:
        prof = load_qc_profile(profile, profiles_path=profiles_path)
        used_params.update({k: v for k, v in prof.items() if v is not None})
    # Merge in ad-hoc params
    if params:
        used_params.update({k: v for k, v in params.items() if v is not None})

    base_df = compute_unit_selection_table(
        session, event_name=used_params["event_name"], window=tuple(used_params["window"]), bin_size=used_params["bin_size"]
    )
    filt_df = apply_unit_filters(
        base_df,
        require_good_cluster=bool(used_params["require_good_cluster"]),
        min_total_spikes=int(used_params["min_total_spikes"]),
        min_mean_rate_hz=float(used_params["min_mean_rate_hz"]),
        max_isi_viol_frac=float(used_params["max_isi_viol_frac"]),
        min_median_spikes_per_trial=float(used_params["min_median_spikes_per_trial"]),
        max_median_spikes_per_trial=used_params.get("max_median_spikes_per_trial"),
    )

    # Save
    base_path = out / "unit_metrics.csv"
    filt_path = out / "unit_selection.csv"
    base_df.to_csv(base_path, index=False)
    filt_df.to_csv(filt_path, index=False)
    # Save params used for reproducibility
    with (out / "unit_selection_params.json").open("w") as f:
        json.dump(used_params, f, indent=2)

    keep_ids = filt_df.loc[filt_df["keep"], "cluster_id"].astype(int).tolist()

    plot_paths: List[str] = []
    if make_plots:
        plot_paths = plot_kept_vs_dropped_distributions(
            filt_df, outdir=str(out), metrics=plot_metrics
        )

    return {
        "unit_metrics_csv": str(base_path),
        "unit_selection_csv": str(filt_path),
        "kept_cluster_ids": keep_ids,
        "n_kept": len(keep_ids),
        "n_total": int(len(filt_df)),
        "plots": plot_paths,
    }


def _safe_hist(ax, data_keep, data_drop, label_keep: str, label_drop: str, xlabel: str, bins: int = 50):
    ax.hist(data_drop, bins=bins, alpha=0.5, color="C3", label=label_drop)
    ax.hist(data_keep, bins=bins, alpha=0.5, color="C0", label=label_keep)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Units")
    ax.legend(fontsize="small")


def plot_kept_vs_dropped_distributions(
    filt_df: pd.DataFrame,
    outdir: str,
    metrics: Optional[List[str]] = None,
) -> List[str]:
    """Generate quick comparison plots for kept vs dropped units.

    Creates:
      - Overlaid histograms for selected metrics (single multi-panel figure)
      - Scatter of mean_rate_hz vs isi_violations_frac colored by keep

    Returns list of saved plot file paths.
    """
    if metrics is None:
        metrics = [
            "mean_rate_hz",
            "isi_violations_frac",
            "median_spikes_per_trial",
            "n_spikes",
        ]

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    kept = filt_df[filt_df["keep"]]
    dropped = filt_df[~filt_df["keep"]]

    # Multi-panel hist figure
    n = len(metrics)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, m in enumerate(metrics):
        ax = axes[i]
        k = kept[m].dropna().to_numpy()
        d = dropped[m].dropna().to_numpy()
        _safe_hist(ax, k, d, "kept", "dropped", xlabel=m)
        ax.set_title(m)
    # hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Kept vs Dropped: distributions", y=1.02)
    fig.tight_layout()
    p1 = out / "unit_selection_dists.png"
    fig.savefig(p1, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Scatter: mean_rate vs ISI violations
    x = filt_df["mean_rate_hz"].to_numpy()
    y = filt_df["isi_violations_frac"].to_numpy()
    c = np.where(filt_df["keep"].to_numpy(), "C0", "C3")
    fig2, ax2 = plt.subplots(figsize=(5.5, 4))
    ax2.scatter(x, y, c=c, s=10, alpha=0.7)
    ax2.set_xlabel("mean_rate_hz")
    ax2.set_ylabel("isi_violations_frac")
    ax2.set_title("Scatter: mean_rate vs ISI (blue=kept, red=dropped)")
    fig2.tight_layout()
    p2 = out / "unit_selection_scatter.png"
    fig2.savefig(p2, dpi=140, bbox_inches="tight")
    plt.close(fig2)

    return [str(p1), str(p2)]


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


if __name__ == "__main__":
    print("qc module: import and call run_qc(session, outdir)")
