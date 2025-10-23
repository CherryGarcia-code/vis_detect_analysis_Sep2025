"""Single-unit analysis helpers: QC metrics, raster and PSTH plotting.

This module provides small utilities to compute per-cluster QC metrics and
to plot a raster + PSTH aligned to events (e.g., Change_ON). It is intentionally
lightweight and depends on the repository's `src` helpers (session_io, align).
"""
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from src.session_io import load_session
from src import align as align_mod


def compute_qc_table(session, event_name: str = "Change_ON", window: Tuple[float, float] = (-0.5, 1.0), bin_size: float = 0.02) -> pd.DataFrame:
    """Compute a per-cluster QC table.

    Columns:
      - cluster_id
      - total_spikes
      - mean_firing_rate (Hz) computed as total_spikes / session_duration
      - n_trials_used (for alignment to event)
      - median_spikes_per_trial
      - isi_violations_fraction (proportion of ISIs < 2 ms)

    Notes:
      - session_duration: estimated from max spike time across clusters if NI events
        do not provide an explicit duration.
    """
    rows = []

    # estimate session duration: prefer NI event range if available
    sess_dur = None
    ni = getattr(session, "ni_events", {}) or {}
    # try to infer from Baseline_ON / Change_ON arrays
    for k in ("Baseline_ON", "Change_ON"):
        if k in ni and np.asarray(ni[k]).size > 0:
            arr = np.asarray(ni[k]).flatten()
            sess_dur = float(np.nanmax(arr) + 10.0)  # add small buffer
            break

    # fallback: use max spike time across clusters
    if sess_dur is None:
        max_sp = 0.0
        for c in session.clusters:
            st = np.asarray(c.spike_times).flatten()
            if st.size:
                max_sp = max(max_sp, float(np.nanmax(st)))
        sess_dur = max_sp if max_sp > 0 else 1.0

    event_times = align_mod.get_event_times(session, event_name)

    for c in session.clusters:
        st = np.asarray(c.spike_times).flatten()
        total_spikes = int(st.size)
        mean_fr = float(total_spikes / sess_dur) if sess_dur > 0 else np.nan

        # per-trial alignment counts
        trials_mat, _ = align_mod.align_spikes_to_events(st, event_times, window=window, bin_size=bin_size)
        n_trials = int(trials_mat.shape[0]) if trials_mat is not None else 0
        spikes_per_trial = trials_mat.sum(axis=1) if n_trials > 0 else np.array([])
        median_spikes_per_trial = float(np.median(spikes_per_trial)) if spikes_per_trial.size > 0 else np.nan

        # ISI violations: proportion of ISIs < 2 ms (0.002 s)
        isi = np.diff(np.sort(st)) if st.size > 1 else np.array([])
        if isi.size > 0:
            viol = float((isi < 0.002).sum()) / isi.size
        else:
            viol = np.nan

        rows.append(
            {
                "cluster_id": int(c.cluster_id),
                "total_spikes": total_spikes,
                "mean_firing_rate": mean_fr,
                "n_trials_used": n_trials,
                "median_spikes_per_trial": median_spikes_per_trial,
                "isi_violations_fraction": viol,
            }
        )

    df = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
    return df


def _spikes_relative_to_events(spike_times: np.ndarray, event_times: List[float], window: Tuple[float, float]) -> List[np.ndarray]:
    """Return list of per-trial spike-time arrays relative to each event time (within window)."""
    out = []
    st = np.asarray(spike_times).flatten()
    for et in event_times:
        aligned = st - float(et)
        mask = (aligned >= window[0]) & (aligned <= window[1])
        out.append(aligned[mask])
    return out


def plot_raster_psth(
    session,
    cluster_id: int,
    event_name: str = "Change_ON",
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.02,
    smooth_sigma: Optional[float] = 1.0,
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None,
):
    """Plot raster (top) and PSTH (bottom) for a single cluster aligned to event_name.

    Returns matplotlib Figure.
    """
    # find cluster
    cluster = None
    for c in session.clusters:
        if int(c.cluster_id) == int(cluster_id):
            cluster = c
            break
    if cluster is None:
        raise ValueError(f"Cluster {cluster_id} not found in session")

    event_times = align_mod.get_event_times(session, event_name)
    trials_spikes = _spikes_relative_to_events(cluster.spike_times, event_times, window)

    # PSTH via counts
    trials_mat, bin_centers = align_mod.align_spikes_to_events(cluster.spike_times, event_times, window=window, bin_size=bin_size)
    # trials_mat already in Hz (counts / bin_size)
    if trials_mat.size == 0:
        mean_psth = np.zeros_like(bin_centers)
    else:
        mean_psth = np.nanmean(trials_mat, axis=0)

    if smooth_sigma is not None and mean_psth.size > 1:
        # gaussian_filter1d expects sigma in number of bins
        mean_psth_smooth = gaussian_filter1d(mean_psth, sigma=smooth_sigma)
    else:
        mean_psth_smooth = mean_psth

    fig, (ax_raster, ax_psth) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [2, 1]}, sharex=True)

    # Raster
    for i, sp in enumerate(trials_spikes):
        if sp.size == 0:
            continue
        ax_raster.vlines(sp, i + 0.1, i + 0.9, color="k", linewidth=0.6)
    ax_raster.set_ylabel("Trial")
    ax_raster.set_title(f"Raster: cluster {cluster_id} aligned to {event_name}")
    ax_raster.axvline(0, color="C1", linestyle="--", linewidth=0.8)

    # PSTH
    ax_psth.plot(bin_centers, mean_psth, color="C0", alpha=0.6, label="PSTH")
    ax_psth.plot(bin_centers, mean_psth_smooth, color="C0", label="Smoothed")
    ax_psth.set_xlabel("Time (s)")
    ax_psth.set_ylabel("Firing rate (Hz)")
    ax_psth.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax_psth.legend(fontsize="small")

    fig.tight_layout()
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_change_rasters_by_outcome(
    session,
    cluster_id: int,
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.02,
    smooth_sigma: Optional[float] = 1.0,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
):
    """Plot rasters/PSTHs aligned to Change_ON, split by outcome (Hit vs Miss).

    Creates a 2-column figure with Hit on the left and Miss on the right.
    """
    # find cluster
    cluster = None
    for c in session.clusters:
        if int(c.cluster_id) == int(cluster_id):
            cluster = c
            break
    if cluster is None:
        raise ValueError(f"Cluster {cluster_id} not found in session")

    change_by_trial = align_mod.get_event_times_by_trial(session, "Change_ON")
    trials = getattr(session, "trials", []) or []
    # Build groups
    idx_hit = [i for i, t in enumerate(trials) if getattr(t, "trialoutcome", None) == "Hit"]
    idx_miss = [i for i, t in enumerate(trials) if getattr(t, "trialoutcome", None) == "Miss"]

    def _times_for(indices):
        out = []
        for i in indices:
            try:
                val = float(change_by_trial[i])
            except Exception:
                continue
            if np.isnan(val):
                continue
            out.append(val)
        return out

    ets_hit = _times_for(idx_hit)
    ets_miss = _times_for(idx_miss)

    # Compute matrices and PSTHs
    m_hit, bc = align_mod.align_spikes_to_events(cluster.spike_times, ets_hit, window=window, bin_size=bin_size)
    m_miss, _ = align_mod.align_spikes_to_events(cluster.spike_times, ets_miss, window=window, bin_size=bin_size)
    psth_hit = np.nanmean(m_hit, axis=0) if m_hit.shape[0] > 0 else np.zeros_like(bc)
    psth_miss = np.nanmean(m_miss, axis=0) if m_miss.shape[0] > 0 else np.zeros_like(bc)
    if smooth_sigma is not None and psth_hit.size > 1:
        psth_hit_s = gaussian_filter1d(psth_hit, sigma=smooth_sigma)
        psth_miss_s = gaussian_filter1d(psth_miss, sigma=smooth_sigma)
    else:
        psth_hit_s, psth_miss_s = psth_hit, psth_miss

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax_r_hit, ax_r_miss = axes[0]
    ax_p_hit, ax_p_miss = axes[1]

    # Raster Hit
    trials_spikes_hit = _spikes_relative_to_events(cluster.spike_times, ets_hit, window)
    for i, sp in enumerate(trials_spikes_hit):
        if len(sp) == 0:
            continue
        ax_r_hit.vlines(sp, i + 0.1, i + 0.9, color="k", linewidth=0.6)
    ax_r_hit.set_title(f"Hit (n={len(ets_hit)})")
    ax_r_hit.set_ylabel("Trial")
    ax_r_hit.axvline(0, color="C1", linestyle="--", linewidth=0.8)

    # Raster Miss
    trials_spikes_miss = _spikes_relative_to_events(cluster.spike_times, ets_miss, window)
    for i, sp in enumerate(trials_spikes_miss):
        if len(sp) == 0:
            continue
        ax_r_miss.vlines(sp, i + 0.1, i + 0.9, color="k", linewidth=0.6)
    ax_r_miss.set_title(f"Miss (n={len(ets_miss)})")
    ax_r_miss.axvline(0, color="C1", linestyle="--", linewidth=0.8)

    # PSTHs
    ax_p_hit.plot(bc, psth_hit, color="C0", alpha=0.5, label="PSTH")
    ax_p_hit.plot(bc, psth_hit_s, color="C0", label="Smoothed")
    ax_p_hit.set_ylabel("FR (Hz)")
    ax_p_hit.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax_p_hit.legend(fontsize="small")

    ax_p_miss.plot(bc, psth_miss, color="C0", alpha=0.5, label="PSTH")
    ax_p_miss.plot(bc, psth_miss_s, color="C0", label="Smoothed")
    ax_p_miss.set_xlabel("Time (s)")
    ax_p_miss.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax_p_miss.legend(fontsize="small")

    fig.suptitle(f"Cluster {cluster_id} aligned to Change_ON by Outcome", y=1.02)
    fig.tight_layout()
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_change_rasters_hit_by_size(
    session,
    cluster_id: int,
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.02,
    smooth_sigma: Optional[float] = 1.0,
    max_groups: int = 6,
    figsize: Tuple[int, int] = (10, 3),
    save_dir: Optional[str] = None,
):
    """Plot rasters/PSTHs aligned to Change_ON for Hit trials, separated by change size.

    If there are many unique sizes, plot up to `max_groups` most frequent sizes.
    Produces one figure per size level to keep axes clear; saves to save_dir if provided.
    Returns list of (size_label, fig_path or None) for produced figures.
    """
    # find cluster
    cluster = None
    for c in session.clusters:
        if int(c.cluster_id) == int(cluster_id):
            cluster = c
            break
    if cluster is None:
        raise ValueError(f"Cluster {cluster_id} not found in session")

    change_by_trial = align_mod.get_event_times_by_trial(session, "Change_ON")
    trials = getattr(session, "trials", []) or []
    # collect hit trials with size
    rows = []
    for i, t in enumerate(trials):
        if getattr(t, "trialoutcome", None) != "Hit":
            continue
        size = getattr(t, "change_size", None)
        try:
            et = float(change_by_trial[i])
        except Exception:
            et = np.nan
        if size is None or np.isnan(et):
            continue
        rows.append((i, float(size), et))

    if len(rows) == 0:
        return []

    df = pd.DataFrame(rows, columns=["trial_idx", "change_size", "event_time"])  # type: ignore
    # Choose up to max_groups sizes by frequency, then by size
    size_counts = df["change_size"].value_counts().reset_index()
    size_counts.columns = ["change_size", "count"]
    size_levels = (
        size_counts.sort_values(["count", "change_size"], ascending=[False, True])["change_size"].tolist()[:max_groups]
    )

    outputs = []
    for size_val in size_levels:
        ets = df.loc[df["change_size"] == size_val, "event_time"].tolist()
        # Compute matrices and PSTH
        m, bc = align_mod.align_spikes_to_events(cluster.spike_times, ets, window=window, bin_size=bin_size)
        psth = np.nanmean(m, axis=0) if m.shape[0] > 0 else np.zeros_like(bc)
        psth_s = gaussian_filter1d(psth, sigma=smooth_sigma) if smooth_sigma is not None and psth.size > 1 else psth

        fig, (axr, axp) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        # Raster
        trials_spikes = _spikes_relative_to_events(cluster.spike_times, ets, window)
        for i, sp in enumerate(trials_spikes):
            if len(sp) == 0:
                continue
            axr.vlines(sp, i + 0.1, i + 0.9, color="k", linewidth=0.6)
        axr.set_title(f"Hit size={size_val:g} (n={len(ets)})")
        axr.set_ylabel("Trial")
        axr.axvline(0, color="C1", linestyle="--", linewidth=0.8)

        # PSTH
        axp.plot(bc, psth, color="C0", alpha=0.5, label="PSTH")
        axp.plot(bc, psth_s, color="C0", label="Smoothed")
        axp.set_xlabel("Time (s)")
        axp.set_ylabel("FR (Hz)")
        axp.axvline(0, color="k", linestyle="--", linewidth=0.8)
        axp.legend(fontsize="small")

        fig.tight_layout()
        fig_path = None
        if save_dir is not None:
            p = Path(save_dir) / f"cluster_{cluster_id}_change_hit_size_{str(size_val).replace('.', 'p')}_raster_psth.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(p), dpi=150, bbox_inches="tight")
            fig_path = str(p)
            plt.close(fig)
        outputs.append((size_val, fig_path))

    return outputs


def demo_for_session(session_path: str, out_dir: str = "png_output/demo_single_unit", n_examples: int = 1, event_name: str = "Change_ON") -> Dict[str, Any]:
    """Run a small demo: compute QC table and generate PNG for first n_examples clusters.

    Returns dict with keys: 'qc_csv', 'pngs' (list)
    """
    session = load_session(session_path)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    qc = compute_qc_table(session, event_name=event_name)
    qc_csv = out_dir_p / "qc_table.csv"
    qc.to_csv(str(qc_csv), index=False)

    pngs = []
    cids = qc["cluster_id"].tolist()[:n_examples]
    for cid in cids:
        png = out_dir_p / f"cluster_{cid}_raster_psth.png"
        plot_raster_psth(session, cid, event_name=event_name, save_path=str(png))
        pngs.append(str(png))

    return {"qc_csv": str(qc_csv), "pngs": pngs}
