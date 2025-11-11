"""Single-unit analysis helpers: QC metrics, raster and PSTH plotting.

This module provides small utilities to compute per-cluster QC metrics and
to plot a raster + PSTH aligned to events (e.g., Change_ON). It is intentionally
lightweight and depends on the repository's `src` helpers (session_io, align).
"""
from typing import List, Optional, Tuple, Dict, Any, Sequence
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from src.session_io import load_session
from src import align as align_mod
from src import qc as qc_mod


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


# ------------------------------
# Kept-units integration helpers
# ------------------------------

def selection_csv_default_path(session, root: str = "table_output/unit_qc") -> Path:
    subject = getattr(session, "subject", None) or "unknown"
    sname = getattr(session, "session_name", None) or "unknown"
    return Path(root) / f"{subject}_{sname}" / "unit_selection.csv"


def load_kept_ids(
    session,
    selection_csv: Optional[str] = None,
) -> List[int]:
    """Return kept cluster IDs from a selection CSV; falls back to session.good_cluster_ids.

    If selection_csv is not provided, tries the conventional path under table_output/unit_qc/.
    """
    if selection_csv is None:
        p = selection_csv_default_path(session)
    else:
        p = Path(selection_csv)
    if p.exists():
        ids = qc_mod.read_kept_cluster_ids(str(p))
        if ids:
            return ids
    # Fallback: use good_cluster_ids if present
    gids = getattr(session, "good_cluster_ids", None)
    if gids:
        return list(map(int, gids))
    return []


def plot_population_heatmap(
    session,
    event_name: str = "Baseline_ON",
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.02,
    selection_csv: Optional[str] = None,
    kept_ids: Optional[List[int]] = None,
    normalize: str = "zscore",
    vmax_percentile: float = 99.0,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Plot per-unit mean PSTH heatmaps for kept vs dropped units around event.

    Returns dict with figure and the arrays used.
    """
    if kept_ids is None:
        kept_ids = load_kept_ids(session, selection_csv)
    kept_set = set(kept_ids)
    event_times = align_mod.get_event_times(session, event_name)

    psths = []
    ids = []
    for c in session.clusters:
        trials_mat, bin_centers = align_mod.align_spikes_to_events(
            c.spike_times, event_times, window=window, bin_size=bin_size
        )
        m = np.nanmean(trials_mat, axis=0) if trials_mat.shape[0] > 0 else np.zeros(len(bin_centers))
        psths.append(m)
        ids.append(int(c.cluster_id))

    if len(psths) == 0:
        raise ValueError("No clusters available for heatmap")

    M = np.vstack(psths)  # units x time
    # normalization
    if normalize == "zscore":
        mu = M.mean(axis=1, keepdims=True)
        sd = M.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        Mnorm = (M - mu) / sd
    elif normalize == "minmax":
        mn = M.min(axis=1, keepdims=True)
        mx = M.max(axis=1, keepdims=True)
        rng = np.where((mx - mn) == 0, 1.0, (mx - mn))
        Mnorm = (M - mn) / rng
    else:
        Mnorm = M

    # split kept/dropped
    kept_idx = [i for i, cid in enumerate(ids) if cid in kept_set]
    drop_idx = [i for i, cid in enumerate(ids) if cid not in kept_set]

    def _sort_by_peak(mat):
        if mat.size == 0:
            return mat
        peaks = np.argmax(mat, axis=1)
        order = np.argsort(peaks)
        return mat[order]

    Mk = _sort_by_peak(Mnorm[kept_idx]) if kept_idx else np.empty((0, Mnorm.shape[1]))
    Md = _sort_by_peak(Mnorm[drop_idx]) if drop_idx else np.empty((0, Mnorm.shape[1]))

    vmax = np.nanpercentile(Mnorm, vmax_percentile)
    vmin = -vmax if normalize == "zscore" else 0.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    im1 = ax1.imshow(Mk, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax, extent=[bin_centers[0], bin_centers[-1], 0, Mk.shape[0]])
    ax1.set_title(f"Kept units (n={Mk.shape[0]})")
    ax1.axvline(0, color="w", linestyle="--", linewidth=0.7)

    im2 = ax2.imshow(Md, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax, extent=[bin_centers[0], bin_centers[-1], 0, Md.shape[0]])
    ax2.set_title(f"Dropped units (n={Md.shape[0]})")
    ax2.axvline(0, color="w", linestyle="--", linewidth=0.7)
    ax2.set_xlabel("Time (s)")

    fig.colorbar(im1, ax=[ax1, ax2], orientation="vertical", fraction=0.02, pad=0.02, label="Normalized FR")
    fig.suptitle(f"Population heatmap around {event_name}")
    fig.tight_layout()
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=140, bbox_inches="tight")
        plt.close(fig)

    return {"fig": fig, "bin_centers": bin_centers, "Mk": Mk, "Md": Md, "kept_ids": kept_ids}


def plot_session_population_psth_by_outcome(
    session,
    event_name: str = "Baseline_ON",
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.02,
    selection_csv: Optional[str] = None,
    kept_only: bool = True,
    outcome_order: Optional[Sequence[str]] = ("Hit", "FA", "Abort", "Miss", "Ref", "Other"),
    outcome_colors: Optional[Dict[str, str]] = None,
    smooth_sigma: Optional[float] = 1.0,
    show_sem: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
    separate_panels: bool = True,
    panel_height: float = 1.2,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Plot session-level average PSTH per outcome across clusters.

    If kept_only is True, restricts to clusters in the selection CSV; otherwise uses all clusters.
    Shading shows SEM across clusters.

    Parameters
    ----------
    session : Session-like object
        Must expose `.clusters` (each with `cluster_id` and `spike_times`), `.trials`, and NI events for alignment.
    event_name : str
        Event to align spikes to (e.g. Baseline_ON or Change_ON).
    window : (float, float)
        Time window around event.
    bin_size : float
        PSTH bin size (seconds).
    kept_only : bool
        Restrict to kept units (from selection_csv) if True.
    outcome_order : Sequence[str] | None
        Ordering of outcome panels / curves. Only outcomes present in data are plotted.
    smooth_sigma : float | None
        Optional Gaussian smoothing sigma (in bins) applied to mean (and SEM if present).
    show_sem : bool
        Whether to display SEM shading.
    figsize : (w, h) | None
        If separate_panels is False: figure size for single overlay panel. If separate_panels is True
        and figsize is None, height auto-scales as panel_height * n_present (min 3.0) and width defaults to 9.
    separate_panels : bool
        If True (default) plot one subplot per outcome for clarity instead of overlaying all outcomes.
    panel_height : float
        Height per panel (ignored if separate_panels is False or figsize provided).
    save_path : str | None
        Optional path to save figure.
    """
    colors = _get_outcome_colors(outcome_colors)

    # Build trial dataframe with outcomes and event times
    trials = getattr(session, "trials", []) or []
    by_trial = align_mod.get_event_times_by_trial(session, event_name)
    rows = []  # (trial_idx, et, outcome)
    for i, t in enumerate(trials):
        try:
            et = float(by_trial[i])
        except Exception:
            et = np.nan
        if np.isnan(et):
            continue
        rows.append((i, et, _normalize_outcome_label(getattr(t, "trialoutcome", None))))
    if not rows:
        raise ValueError("No valid events for session trials")
    df = pd.DataFrame(rows, columns=["trial_idx", "event_time", "outcome"])  # type: ignore

    # Determine cluster set
    cluster_ids = [int(c.cluster_id) for c in session.clusters]
    if kept_only:
        kept = set(load_kept_ids(session, selection_csv))
        if kept:
            cluster_ids = [cid for cid in cluster_ids if cid in kept]
    if not cluster_ids:
        raise ValueError("No clusters available for population PSTH")

    # Consistent binning
    _, bin_centers = align_mod.align_spikes_to_events(np.array([]), df["event_time"].tolist(), window=window, bin_size=bin_size)

    # Compute per-cluster PSTH per outcome
    present = [o for o in (list(outcome_order) if outcome_order is not None else list(colors.keys())) if o in set(df["outcome"].unique())]
    out_curves: Dict[str, Dict[str, np.ndarray]] = {}
    for o in present:
        out_curves[o] = {"psths": []}
        ets = df.loc[df["outcome"] == o, "event_time"].tolist()
        if len(ets) == 0:
            continue
        for c in session.clusters:
            cid = int(c.cluster_id)
            if cid not in cluster_ids:
                continue
            m, _ = align_mod.align_spikes_to_events(c.spike_times, ets, window=window, bin_size=bin_size)
            psth = np.nanmean(m, axis=0) if m.shape[0] > 0 else np.zeros_like(bin_centers)
            out_curves[o]["psths"].append(psth)

    # Aggregate across clusters
    axes: List[plt.Axes] = []
    if not present:
        raise ValueError("No outcomes present to plot")

    # Determine total number of units used (constant across outcomes when kept_only True)
    n_units_total = len(set([int(c.cluster_id) for c in session.clusters if (not kept_only) or (int(c.cluster_id) in set(cluster_ids))]))

    if separate_panels and len(present) > 1:
        # Determine figure size dynamically if not provided
        if figsize is None:
            height = max(3.0, panel_height * len(present))
            figsize = (9, height)
        # Build a 2-column layout: left = plots, right = legend area
        fig, axes_grid = plt.subplots(len(present), 2, figsize=figsize, sharex=True,
                                      constrained_layout=False,
                                      gridspec_kw={"width_ratios": [5, 2], "wspace": 0.25})
        # Normalize to 2D array
        if not isinstance(axes_grid, np.ndarray) or axes_grid.ndim == 1:
            axes_grid = np.array([axes_grid])
        left_axes = [axes_grid[i, 0] for i in range(len(present))]
        legend_axes = [axes_grid[i, 1] for i in range(len(present))]
        # We'll use the top-right axis for the legend and hide others
        for la in legend_axes[1:]:
            la.axis("off")
        legend_ax = legend_axes[0]
        legend_ax.axis("off")

        axes = left_axes
        handles_for_legend = []
        labels_for_legend = []
        for ax, o in zip(left_axes, present):
            psths = out_curves[o].get("psths", [])
            if not psths:
                ax.text(0.5, 0.5, f"No trials for {o}", ha="center", va="center")
                continue
            M = np.vstack(psths)
            mean = np.nanmean(M, axis=0)
            sem = np.nanstd(M, axis=0) / np.sqrt(max(1, M.shape[0])) if show_sem else None
            if smooth_sigma is not None and mean.size > 1:
                mean = gaussian_filter1d(mean, sigma=smooth_sigma)
                if show_sem and sem is not None:
                    sem = gaussian_filter1d(sem, sigma=smooth_sigma)
            ln = ax.plot(bin_centers, mean, color=colors.get(o, colors["Other"]))[0]
            if show_sem and sem is not None:
                ax.fill_between(bin_centers, mean - sem, mean + sem, color=colors.get(o, colors["Other"]), alpha=0.25, linewidth=0)
            ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
            ax.set_ylabel("FR (Hz)")
            # Collect legend handle and label once per outcome
            handles_for_legend.append(ln)
            labels_for_legend.append(o)
        axes[-1].set_xlabel("Time (s)")
        # Build a unified legend in the right column
        if handles_for_legend:
            lg_title = f"Outcomes (nUnits={n_units_total})\nAligned to: {event_name}"
            legend_ax.legend(handles_for_legend, labels_for_legend, loc="upper left", frameon=False, title=lg_title)
        fig.suptitle(f"Session population PSTH by outcome ({'kept' if kept_only else 'all'} units)")
    else:
        # Fallback to single overlay panel (either only one outcome or user requested overlay)
        if figsize is None:
            figsize = (9, 4)
        fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True)
        axes = [ax]
        handles = []
        labels = []
        for o in present:
            psths = out_curves[o].get("psths", [])
            if not psths:
                continue
            M = np.vstack(psths)
            mean = np.nanmean(M, axis=0)
            sem = np.nanstd(M, axis=0) / np.sqrt(max(1, M.shape[0])) if show_sem else None
            if smooth_sigma is not None and mean.size > 1:
                mean = gaussian_filter1d(mean, sigma=smooth_sigma)
                if show_sem and sem is not None:
                    sem = gaussian_filter1d(sem, sigma=smooth_sigma)
            line = ax.plot(bin_centers, mean, color=colors.get(o, colors["Other"]))[0]
            if show_sem and sem is not None:
                ax.fill_between(bin_centers, mean - sem, mean + sem, color=colors.get(o, colors["Other"]), alpha=0.2, linewidth=0)
            handles.append(line)
            labels.append(o)
        ax.set_title(f"Session population PSTH by outcome — aligned to {event_name} ({'kept' if kept_only else 'all'} units; n={n_units_total})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("FR (Hz)")
        ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
        if labels:
            ax.legend(handles, labels, fontsize="small", ncol=2, title=f"nUnits={n_units_total}")

    fig.tight_layout()
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        plt.close(fig)
    return {"fig": fig, "axes": axes, "bin_centers": bin_centers, "present_outcomes": present}


def plot_rasters_for_kept(
    session,
    selection_csv: Optional[str] = None,
    event_name: str = "Baseline_ON",
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.02,
    out_dir: Optional[str] = None,
    max_units: int = 10,
) -> List[str]:
    """Plot single-unit raster+PSTH for first N kept units.

    Returns list of saved PNG paths if out_dir provided.
    """
    kept = load_kept_ids(session, selection_csv)
    if not kept:
        return []
    paths = []
    for cid in kept[:max_units]:
        png = None
        if out_dir is not None:
            png = str(Path(out_dir) / f"cluster_{cid}_raster_psth.png")
        plot_raster_psth(session, cid, event_name=event_name, window=window, bin_size=bin_size, save_path=png)
        if png is not None:
            paths.append(png)
    return paths


# ---------------------------------------------
# Baseline-aligned, outcome-colored raster/PSTH
# ---------------------------------------------

def _get_outcome_colors(custom: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return a mapping from canonical outcome label to color (case-insensitive).

    Defaults per user spec:
      - Hit: green
      - FA: red
      - Abort: violet/purple
      - Miss: gray
      - Ref: black
      - Other: gray
    """
    # canonical, case-sensitive keys that we'll use internally
    default = {
        "Hit": "#2ca02c",    # green
        "FA": "#d62728",     # red
        "Abort": "#7b1fa2",  # purple/violet
        "Miss": "#7f7f7f",   # gray
        "Ref": "#000000",    # black
        "Other": "#7f7f7f",  # gray fallback
    }
    if not custom:
        return default
    out = default.copy()
    # allow case-insensitive override keys
    for k, v in custom.items():
        if not isinstance(k, str):
            continue
        kk = k.strip().lower()
        # map to canonical
        if kk in ("hit",):
            out["Hit"] = v
        elif kk in ("fa", "falsealarm", "false_alarm"):
            out["FA"] = v
        elif kk in ("abort",):
            out["Abort"] = v
        elif kk in ("miss",):
            out["Miss"] = v
        elif kk in ("ref", "reference", "ref_trial"):
            out["Ref"] = v
        elif kk in ("other",):
            out["Other"] = v
    return out


def _normalize_outcome_label(label: Optional[str]) -> str:
    """Map various outcome strings to canonical labels: Hit, FA, Abort, Miss, Ref, Other."""
    if label is None:
        return "Other"
    s = str(label).strip().lower()
    if s == "hit":
        return "Hit"
    if s in ("fa", "falsealarm", "false_alarm", "false alarm"):
        return "FA"
    if s == "abort":
        return "Abort"
    if s == "miss":
        return "Miss"
    if s in ("ref", "reference", "ref_trial", "reftrial"):
        return "Ref"
    return "Other"


def _trial_reaction_time(trial) -> float:
    """Best-effort extraction of a scalar reaction time (seconds) from a Trial.

    Looks for common keys in Trial.reactiontimes; returns NaN if unavailable.
    """
    import math

    rt = getattr(trial, "reactiontimes", None) or {}
    if not isinstance(rt, dict):
        return float("nan")
    for key in (
        "true",
        "firstlick",
        "change_to_lick",
        "ChangeOn_to_Response",
        "rt",
    ):
        val = rt.get(key, None)
        if val is None:
            continue
        try:
            f = float(val)
            if math.isfinite(f):
                return f
        except Exception:
            continue
    return float("nan")


def plot_baseline_raster_psth_by_future_outcome(
    session,
    cluster_id: int,
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.02,
    smooth_sigma: Optional[float] = 1.0,
    sort_trials: str = "outcome",  # one of: 'outcome' | 'future_rt' | 'none'
    outcome_order: Optional[Sequence[str]] = ("Hit", "FA", "Abort", "Miss", "Ref", "Other"),
    outcome_colors: Optional[Dict[str, str]] = None,
    peth_scale: str = "per_outcome",  # 'shared' or 'per_outcome'
    show_sem: bool = True,
    figsize: Tuple[int, int] = (9, 5),
    save_path: Optional[str] = None,
):
    """Baseline-aligned raster with trials colored by FUTURE outcome; PSTH split by outcome.

    - Raster: each trial's spikes are drawn in the color for that trial's future outcome.
    - PSTH: multiple lines, one per outcome, using the same color mapping.

    sort_trials controls trial ordering in the raster:
      - 'outcome': group by outcome (order given by outcome_order), preserve within-group original order
      - 'future_rt': sort all trials by reaction time ascending (NaNs last)
      - 'none': keep original chronological order
    """
    # find cluster
    cluster = None
    for c in session.clusters:
        if int(c.cluster_id) == int(cluster_id):
            cluster = c
            break
    if cluster is None:
        raise ValueError(f"Cluster {cluster_id} not found in session")

    # Collect per-trial metadata
    trials = getattr(session, "trials", []) or []
    by_trial = align_mod.get_event_times_by_trial(session, "Baseline_ON")
    rows = []  # (trial_idx, et, outcome, rt)
    for i, t in enumerate(trials):
        try:
            et = float(by_trial[i])
        except Exception:
            et = np.nan
        if np.isnan(et):
            continue
        o = _normalize_outcome_label(getattr(t, "trialoutcome", None))
        rt = _trial_reaction_time(t)
        rows.append((i, et, o, rt))
    if len(rows) == 0:
        raise ValueError("No valid Baseline_ON events found for trials")

    df = pd.DataFrame(rows, columns=["trial_idx", "event_time", "outcome", "rt"])  # type: ignore

    # Sorting logic
    if sort_trials == "future_rt":
        df = df.sort_values(["rt", "trial_idx"], na_position="last").reset_index(drop=True)
    elif sort_trials == "outcome":
        order = list(outcome_order) if outcome_order is not None else ["Hit", "FA", "Abort", "Miss", "Ref", "Other"]
        cat = pd.Categorical(df["outcome"], categories=order, ordered=True)
        df = df.assign(_o=cat).sort_values(["_o", "trial_idx"]).drop(columns=["_o"]).reset_index(drop=True)
    else:
        # keep original order (by trial index)
        df = df.sort_values("trial_idx").reset_index(drop=True)

    # Outcome groups and colors
    colors = _get_outcome_colors(outcome_colors)
    # Keep only outcomes present and preserve desired order
    present = [o for o in (list(outcome_order) if outcome_order is not None else list(colors.keys())) if o in set(df["outcome"].unique())]
    groups = {o: df.loc[df["outcome"] == o] for o in present}

    # Build PSTHs per outcome and raster data per trial
    # Precompute bin centers using all events for consistency
    mat_all, bin_centers = align_mod.align_spikes_to_events(cluster.spike_times, df["event_time"].tolist(), window=window, bin_size=bin_size)

    # Prepare figure: raster + PSTH(s) with optional right-side legend column
    total_trials = len(df)
    if peth_scale == "shared":
        # 2 rows x 2 cols: left column holds raster and overlay PSTH; right column is legend
        fig, axes_grid = plt.subplots(2, 2, figsize=figsize, sharex=True,
                                      gridspec_kw={"height_ratios": [2, 1], "width_ratios": [5, 2], "wspace": 0.25})
        ax_r, ax_p = axes_grid[0, 0], axes_grid[1, 0]
        legend_ax = axes_grid[0, 1]
        # Hide lower-right cell
        axes_grid[1, 1].axis("off")
        legend_ax.axis("off")
    else:
        # per-outcome PSTH panels stacked below raster; add a legend column on the right
        n_pan = max(1, len(groups))
        heights = [2] + [1] * n_pan
        # Build a 2-column grid: left = plots, right = legend
        fig_w = figsize[0]
        fig_h = max(figsize[1], 3 + 2 * n_pan)
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = gridspec.GridSpec(1 + n_pan, 2, figure=fig, height_ratios=heights, width_ratios=[5, 2], wspace=0.25)
        # Left column axes
        ax_r = fig.add_subplot(gs[0, 0])
        ax_ps = [fig.add_subplot(gs[i, 0], sharex=ax_r) for i in range(1, 1 + n_pan)]
        # Right legend axis (top-right cell)
        legend_ax = fig.add_subplot(gs[0, 1])
        legend_ax.axis("off")
        # Hide any remaining right-column cells
        for i in range(1, 1 + n_pan):
            fig.add_subplot(gs[i, 1]).axis("off")

    # Raster: iterate in displayed order
    trials_spikes = _spikes_relative_to_events(cluster.spike_times, df["event_time"].tolist(), window)
    for row_idx, (_, row) in enumerate(df.iterrows()):
        sp = trials_spikes[row_idx]
        if sp.size == 0:
            continue
        col = colors.get(row["outcome"], colors.get("Other", "#666666"))
        ax_r.vlines(sp, row_idx + 0.1, row_idx + 0.9, color=col, linewidth=0.6)
    ax_r.set_ylabel("Trial")
    # Keep raster clean; legend and suptitle will carry meta-information
    ax_r.axvline(0, color="k", linestyle="--", linewidth=0.8)

    # PSTH: overlay or per-outcome panels
    if peth_scale == "shared":
        handles = []
        labels = []
        for o, g in groups.items():
            if g.shape[0] == 0:
                continue
            idxs = g.index.values
            if np.ndim(mat_all) != 2 or mat_all.shape[0] == 0:
                psth = np.zeros_like(bin_centers)
                sem = np.zeros_like(bin_centers)
            else:
                sub = mat_all[idxs, :]
                psth = np.nanmean(sub, axis=0)
                # SEM across trials
                sem = np.nanstd(sub, axis=0) / np.sqrt(max(1, sub.shape[0])) if show_sem else None
            if smooth_sigma is not None and psth.size > 1:
                psth = gaussian_filter1d(psth, sigma=smooth_sigma)
                if show_sem and sem is not None:
                    sem = gaussian_filter1d(sem, sigma=smooth_sigma)
            ln = ax_p.plot(bin_centers, psth, color=colors.get(o, colors["Other"]))[0]
            handles.append(ln)
            labels.append(o)
            if show_sem and sem is not None:
                ax_p.fill_between(bin_centers, psth - sem, psth + sem, color=colors.get(o, colors["Other"]), alpha=0.2, linewidth=0)
        ax_p.set_xlabel("Time (s)")
        ax_p.set_ylabel("Firing rate (Hz)")
        ax_p.axvline(0, color="k", linestyle="--", linewidth=0.8)
        if handles:
            legend_ax.legend(handles, labels, loc="upper left", frameon=False,
                             title=f"Outcomes (nTrials={total_trials})\nAligned to: Baseline_ON")
    else:
        # separate y-scales
        handles = []
        labels = []
        for (o, g), axp in zip(groups.items(), ax_ps):
            if g.shape[0] == 0:
                continue
            idxs = g.index.values
            if np.ndim(mat_all) != 2 or mat_all.shape[0] == 0:
                psth = np.zeros_like(bin_centers)
                sem = np.zeros_like(bin_centers)
            else:
                sub = mat_all[idxs, :]
                psth = np.nanmean(sub, axis=0)
                sem = np.nanstd(sub, axis=0) / np.sqrt(max(1, sub.shape[0])) if show_sem else None
            if smooth_sigma is not None and psth.size > 1:
                psth = gaussian_filter1d(psth, sigma=smooth_sigma)
                if show_sem and sem is not None:
                    sem = gaussian_filter1d(sem, sigma=smooth_sigma)
            ln = axp.plot(bin_centers, psth, color=colors.get(o, colors["Other"]))[0]
            handles.append(ln)
            labels.append(o)
            if show_sem and sem is not None:
                axp.fill_between(bin_centers, psth - sem, psth + sem, color=colors.get(o, colors["Other"]), alpha=0.2, linewidth=0)
            axp.axvline(0, color="k", linestyle="--", linewidth=0.8)
            axp.set_ylabel("FR (Hz)")
        ax_ps[-1].set_xlabel("Time (s)")
        # Build unified legend at right
        if handles:
            legend_ax.legend(handles, labels, loc="upper left", frameon=False,
                             title=f"Outcomes (nTrials={total_trials})\nAligned to: Baseline_ON")

    fig.tight_layout()
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_baseline_rasters_for_kept_by_outcome(
    session,
    selection_csv: Optional[str] = None,
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.02,
    out_dir: Optional[str] = None,
    max_units: int = 10,
    sort_trials: str = "outcome",
    outcome_colors: Optional[Dict[str, str]] = None,
    smooth_sigma: Optional[float] = 1.0,
    peth_scale: str = "per_outcome",
) -> List[str]:
    """Generate baseline-aligned, outcome-colored rasters/PSTHs for first N kept units.

    Returns list of saved PNG paths if out_dir is provided.
    """
    kept = load_kept_ids(session, selection_csv)
    if not kept:
        return []
    paths = []
    for cid in kept[:max_units]:
        png = None
        if out_dir is not None:
            png = str(Path(out_dir) / f"cluster_{cid}_baseline_by_outcome.png")
        plot_baseline_raster_psth_by_future_outcome(
            session,
            cid,
            window=window,
            bin_size=bin_size,
            smooth_sigma=smooth_sigma,
            sort_trials=sort_trials,
            outcome_colors=outcome_colors,
            peth_scale=peth_scale,
            save_path=png,
        )
        if png is not None:
            paths.append(png)
    return paths
