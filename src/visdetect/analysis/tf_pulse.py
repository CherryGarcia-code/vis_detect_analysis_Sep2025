"""TF pulse responsiveness screening (baseline pulses from St1TrialVector).

Defines fast- and slow-pulse responsive units by aligning to baseline TF pulses
and testing for post-pulse z-score deviations relative to a pre-pulse baseline.

Derived from notebook scripts in database/, adapted to the standardized Session
dataclasses and kept-only selection.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visdetect.analysis.constants import (
    TF_PULSE_PRE_WINDOW,
    TF_PULSE_POST_WINDOW,
    TF_PULSE_TRACE_PRE,
    DEFAULT_Z_THRESH_TF,
)
from visdetect.analysis.su_analysis import load_kept_ids
from scipy.ndimage import gaussian_filter1d
from visdetect.utils.progress import Progress
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


@dataclass
class TFRespPulseConfig:
    # Pulse classification thresholds on log2(TF)
    fast_thresh_log2: float = 0.25
    slow_thresh_log2: float = -0.25
    # How to read baseline TF vector and convert to times
    baseline_stride: int = 3  # sample every 3rd element (legacy convention)
    sample_period: float = 0.05  # seconds per baseline sample (50 ms)
    # Time windows around pulses
    pre_window: Tuple[float, float] = TF_PULSE_PRE_WINDOW
    post_window: Tuple[float, float] = TF_PULSE_POST_WINDOW
    # Wider trace start for extraction (z-score baseline still uses pre_window)
    trace_pre: Optional[float] = None  # None → use pre_window[0]
    # Smoothing for spike KDE
    dt: float = 0.001  # seconds per bin for smoothing grid
    sigma_ms: float = 17.0  # Gaussian sigma in ms (approx 40ms FWHM)
    # Trial constraints (enabled by default to match stricter legacy v2 script)
    use_constraints: bool = True
    min_after_baseline: float = 1.0
    min_before_change: float = 1.0
    min_before_outcome_fa_abort: float = 2.0
    # Unit selection
    kept_only: bool = True
    # Z-score threshold
    z_thresh: float = DEFAULT_Z_THRESH_TF
    # Plotting aggregation filter
    min_mean_rate_for_plot: float = 0.01


@dataclass
class TFUnitTrace:
    cluster_id: int
    fast_z: np.ndarray
    fast_z_sem: np.ndarray
    slow_z: np.ndarray
    slow_z_sem: np.ndarray
    z_max_fast: float
    z_min_fast: float
    z_max_slow: float
    z_min_slow: float


def _safe_log2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log2(x)
    y[~np.isfinite(y)] = np.nan
    return y


def _per_trial_event_times(session, key: str) -> np.ndarray:
    # Import the project alignment helper. Older code referenced `src.align`;
    # the correct import path in this repository layout is `visdetect.analysis.align`.
    from visdetect.analysis.align import get_event_times_by_trial

    arr = np.array(get_event_times_by_trial(session, key), dtype=float)
    return arr


def _outcome_time_for_trial(trial, baseline_t: Optional[float]) -> Optional[float]:
    out = getattr(trial, "trialoutcome", None)
    rts = getattr(trial, "reactiontimes", {}) or {}
    if out in ("FA", "abort") and baseline_t is not None:
        val = rts.get(out, np.nan)
        try:
            fv = float(val)
        except Exception:
            return None
        if np.isfinite(fv):
            return float(baseline_t + fv)
    return None


def _collect_pulses(session, cfg: TFRespPulseConfig, show_progress: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fast_times, slow_times) arrays of absolute pulse times across trials.

    When show_progress is True, emits periodic trial progress updates (about 20 steps).
    """
    trials = getattr(session, "trials", []) or []
    base_by_trial = _per_trial_event_times(session, "Baseline_ON")
    change_by_trial = _per_trial_event_times(session, "Change_ON")

    fast_times: List[float] = []
    slow_times: List[float] = []

    total = len(trials)
    pr = Progress("Scanning baseline TF pulses", total) if (show_progress and total) else None
    if pr is not None:
        pr.start()

    for i, t in enumerate(trials, 1):
        bv = getattr(t, "baseline_values", None)
        if bv is None:
            continue
        try:
            arr = np.array(bv).flatten()
        except Exception:
            continue
        # Subsample baseline vector first (legacy), then trim to n_seen
        if cfg.baseline_stride > 1:
            arr = arr[:: cfg.baseline_stride]
        n_seen = getattr(t, "n_seen", None)
        if isinstance(n_seen, (int, np.integer)) and n_seen is not None and n_seen > 0:
            arr = arr[: int(n_seen)]
        # Compute log2 TF and identify fast/slow bins
        log2_tf = _safe_log2(arr)
        # Absolute reference time for trial
        t0 = float(base_by_trial[i]) if i < len(base_by_trial) and np.isfinite(base_by_trial[i]) else None
        t_change = float(change_by_trial[i]) if i < len(change_by_trial) and np.isfinite(change_by_trial[i]) else None
        t_outcome = _outcome_time_for_trial(t, t0)

        for bin_idx, l2 in enumerate(log2_tf):
            if not np.isfinite(l2):
                continue
            t_pulse = (t0 + bin_idx * cfg.sample_period) if t0 is not None else None
            if t_pulse is None:
                continue
            # Optional constraints (off by default to mirror legacy behavior)
            if cfg.use_constraints:
                if t_pulse < (t0 + cfg.min_after_baseline):
                    continue
                if (t_change is not None) and (t_pulse > (t_change - cfg.min_before_change)):
                    continue
                if t_outcome is not None and (t_pulse > (t_outcome - cfg.min_before_outcome_fa_abort)):
                    continue
            # Classify
            if l2 >= cfg.fast_thresh_log2:
                fast_times.append(float(t_pulse))
            elif l2 <= cfg.slow_thresh_log2:
                slow_times.append(float(t_pulse))

        if pr is not None:
            pr.update(i)

    return np.array(fast_times, dtype=float), np.array(slow_times, dtype=float)


def _smooth_binned_activity(spike_times_rel: np.ndarray, t_vec: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Bin spikes onto t_vec grid and smooth with Gaussian (legacy-compatible)."""
    if spike_times_rel.size == 0:
        return np.zeros_like(t_vec)
    # Build 0/1 train on the grid
    train = np.zeros_like(t_vec)
    idx = np.searchsorted(t_vec, spike_times_rel)
    idx = idx[(idx >= 0) & (idx < train.size)]
    train[idx] = 1.0
    return gaussian_filter1d(train, sigma=sigma_bins)


def _mean_activity_per_unit(
    spike_times: np.ndarray,
    pulses: np.ndarray,
    pre_window: Tuple[float, float],
    post_window: Tuple[float, float],
    dt: float,
    sigma_ms: float,
    trace_start: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pulses = np.asarray(pulses, dtype=float)
    pulses = pulses[np.isfinite(pulses)]
    full0 = trace_start if trace_start is not None else pre_window[0]
    full1 = post_window[1]
    if pulses.size == 0:
        t_vec = np.arange(full0, full1, dt, dtype=float)
        return np.array([]), np.array([]), t_vec
    t_vec = np.arange(full0, full1, dt, dtype=float)
    sigma_bins = (sigma_ms / 1000.0) / dt
    traces = []
    for tp in pulses:
        rel = spike_times - tp
        mask = (rel >= full0) & (rel < full1)
        srel = rel[mask]
        rate = _smooth_binned_activity(srel, t_vec, sigma_bins)
        traces.append(rate)
    if not traces:
        return np.array([]), np.array([]), t_vec
    arr = np.stack(traces, axis=0)
    mean_trace = np.nanmean(arr, axis=0)
    if arr.shape[0] > 1:
        sem_trace = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
    else:
        sem_trace = np.zeros_like(mean_trace)
    return mean_trace, sem_trace, t_vec


def _zscore_trace(
    mean_trace: np.ndarray,
    t_vec: np.ndarray,
    pre_window: Tuple[float, float],
    return_stats: bool = False,
) -> np.ndarray | Tuple[np.ndarray, float]:
    if mean_trace.size == 0:
        return (mean_trace, np.nan) if return_stats else mean_trace
    pre_mask = (t_vec >= pre_window[0]) & (t_vec < pre_window[1])
    mu = float(np.nanmean(mean_trace[pre_mask])) if np.any(pre_mask) else 0.0
    sd = float(np.nanstd(mean_trace[pre_mask])) if np.any(pre_mask) else 0.0
    if not np.isfinite(sd) or sd <= 0:
        # Avoid divide-by-zero: return zero-centered trace
        z = mean_trace * 0.0
        return (z, sd) if return_stats else z
    z = (mean_trace - mu) / sd
    if return_stats:
        return z, sd
    return z


# ── Post-hoc linear detrending ──────────────────────────────────────

def detrend_tf_traces(
    t_vec: np.ndarray,
    traces: np.ndarray,
    baseline_window: Tuple[float, float] = (-0.4, -0.01),
    post_window: Tuple[float, float] = (0.0, 0.3),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear-detrend z-scored TF pulse traces and measure post-pulse peaks.

    Fits a degree-1 polynomial to the baseline window of each trace,
    subtracts the extrapolated trend from the full trace, then measures
    peak and trough z-scores in the post-pulse window.

    The trend is subtracted from the **full** trace (for clean
    visualisation), but peaks are measured only in *post_window*
    (default 0–300 ms) to avoid unreliable extrapolation far beyond
    the fit region.

    Parameters
    ----------
    t_vec : ndarray, shape (n_time,)
        Time vector in **seconds** (matching NPZ ``t_vec``).
    traces : ndarray, shape (n_units, n_time)
        Z-scored mean traces (e.g. ``fast_z`` or ``slow_z`` from NPZ).
    baseline_window : tuple of float
        (start, end) in seconds for the linear-fit region.
    post_window : tuple of float
        (start, end) in seconds for peak/trough measurement.

    Returns
    -------
    detrended : ndarray, shape (n_units, n_time)
        Full detrended traces.
    z_max_post : ndarray, shape (n_units,)
        Maximum z-score in *post_window* per unit.
    z_min_post : ndarray, shape (n_units,)
        Minimum z-score in *post_window* per unit.
    """
    t_s = np.asarray(t_vec, dtype=float)
    pre_mask = (t_s >= baseline_window[0]) & (t_s < baseline_window[1])
    post_mask = (t_s >= post_window[0]) & (t_s < post_window[1])

    n_units = traces.shape[0]
    detrended = traces.copy()
    z_max_post = np.full(n_units, np.nan)
    z_min_post = np.full(n_units, np.nan)

    if pre_mask.sum() < 2:
        # Not enough baseline bins for a linear fit — return as-is
        if post_mask.any():
            z_max_post = np.nanmax(traces[:, post_mask], axis=1)
            z_min_post = np.nanmin(traces[:, post_mask], axis=1)
        return detrended, z_max_post, z_min_post

    t_pre = t_s[pre_mask]
    for u in range(n_units):
        tr = traces[u]
        if np.all(np.isnan(tr)):
            continue
        coeffs = np.polyfit(t_pre, tr[pre_mask], deg=1)
        trend = np.polyval(coeffs, t_s)
        detrended[u] = tr - trend
        if post_mask.any():
            z_max_post[u] = np.nanmax(detrended[u, post_mask])
            z_min_post[u] = np.nanmin(detrended[u, post_mask])

    return detrended, z_max_post, z_min_post


def _compute_trace_for_cluster(args):
    """Top-level worker for parallel trace computation.

    args: tuple of 9 or 10 elements:
      (cid, spike_times, fast_times, slow_times, t_vec,
       pre_window, post_window, dt, sigma_ms[, trace_start])
    Returns a tuple (cid, TFUnitTrace)
    """
    if len(args) == 10:
        (cid, spike_times, fast_times, slow_times, t_vec,
         pre_window, post_window, dt, sigma_ms, trace_start) = args
    else:
        (cid, spike_times, fast_times, slow_times, t_vec,
         pre_window, post_window, dt, sigma_ms) = args
        trace_start = None
    import numpy as _np

    post_mask = (t_vec >= post_window[0]) & (t_vec < post_window[1])

    st = _np.asarray(spike_times, dtype=float).flatten()

    mf, mf_sem, _ = _mean_activity_per_unit(
        st, fast_times, pre_window, post_window, dt, sigma_ms,
        trace_start=trace_start)
    if mf.size > 0:
        zf, fast_sd = _zscore_trace(mf, t_vec, pre_window, return_stats=True)
        zf_sem = mf_sem / fast_sd if _np.isfinite(fast_sd) and fast_sd > 0 else _np.zeros_like(mf_sem)
        zf_peak = float(_np.nanmax(zf[post_mask])) if _np.any(post_mask) else _np.nan
        zf_trough = float(_np.nanmin(zf[post_mask])) if _np.any(post_mask) else _np.nan
    else:
        zf = _np.full_like(t_vec, _np.nan)
        zf_sem = _np.zeros_like(t_vec)
        zf_peak = _np.nan
        zf_trough = _np.nan

    ms, ms_sem, _ = _mean_activity_per_unit(
        st, slow_times, pre_window, post_window, dt, sigma_ms,
        trace_start=trace_start)
    if ms.size > 0:
        zs, slow_sd = _zscore_trace(ms, t_vec, pre_window, return_stats=True)
        zs_sem = ms_sem / slow_sd if _np.isfinite(slow_sd) and slow_sd > 0 else _np.zeros_like(ms_sem)
        zs_peak = float(_np.nanmax(zs[post_mask])) if _np.any(post_mask) else _np.nan
        zs_trough = float(_np.nanmin(zs[post_mask])) if _np.any(post_mask) else _np.nan
    else:
        zs = _np.full_like(t_vec, _np.nan)
        zs_sem = _np.zeros_like(t_vec)
        zs_peak = _np.nan
        zs_trough = _np.nan

    entry = TFUnitTrace(
        cluster_id=cid,
        fast_z=zf,
        fast_z_sem=zf_sem,
        slow_z=zs,
        slow_z_sem=zs_sem,
        z_max_fast=zf_peak,
        z_min_fast=zf_trough,
        z_max_slow=zs_peak,
        z_min_slow=zs_trough,
    )
    return cid, entry


def collect_tf_pulse_traces(
    session,
    cfg: Optional[TFRespPulseConfig] = None,
    selection_csv: Optional[str] = None,
    show_progress: bool = True,
    fast_times: Optional[np.ndarray] = None,
    slow_times: Optional[np.ndarray] = None,
    cache_path: Optional[str] = None,
    parallel: bool = False,
    n_workers: Optional[int] = None,
) -> Tuple[np.ndarray, List[TFUnitTrace]]:
    """Return time vector and per-cluster z-scored traces plus metadata.

    If `cache_path` is provided, attempt to load traces from the compressed
    `.npz` file at that path. If not present, compute traces and save a cache
    there for future reuse.
    """
    if cfg is None:
        cfg = TFRespPulseConfig()
    kept_ids = load_kept_ids(session, selection_csv) if cfg.kept_only else None
    cluster_ids = [int(c.cluster_id) for c in session.clusters if (kept_ids is None or int(c.cluster_id) in kept_ids)]

    # Allow caller to provide precomputed pulse times to avoid redundant work
    if fast_times is None or slow_times is None:
        fast_times, slow_times = _collect_pulses(session, cfg, show_progress=show_progress)

    full0 = cfg.trace_pre if cfg.trace_pre is not None else cfg.pre_window[0]
    full1 = cfg.post_window[1]
    t_vec = np.arange(full0, full1, cfg.dt, dtype=float)
    post_mask = (t_vec >= cfg.post_window[0]) & (t_vec < cfg.post_window[1])

    # Try loading cache if requested
    if cache_path is not None:
        try:
            p = Path(cache_path)
            if p.exists():
                npz = np.load(str(p), allow_pickle=False)
                t_vec = npz["t_vec"]
                cluster_ids_loaded = npz["cluster_ids"].astype(int).tolist()
                fast_z = npz["fast_z"]
                fast_z_sem = npz["fast_z_sem"]
                slow_z = npz["slow_z"]
                slow_z_sem = npz["slow_z_sem"]
                z_max_fast = npz.get("z_max_fast")
                z_min_fast = npz.get("z_min_fast")
                z_max_slow = npz.get("z_max_slow")
                z_min_slow = npz.get("z_min_slow")

                entries = []
                for i, cid in enumerate(cluster_ids_loaded):
                    entries.append(
                        TFUnitTrace(
                            cluster_id=int(cid),
                            fast_z=fast_z[i],
                            fast_z_sem=fast_z_sem[i],
                            slow_z=slow_z[i],
                            slow_z_sem=slow_z_sem[i],
                            z_max_fast=float(z_max_fast[i]) if z_max_fast is not None else np.nan,
                            z_min_fast=float(z_min_fast[i]) if z_min_fast is not None else np.nan,
                            z_max_slow=float(z_max_slow[i]) if z_max_slow is not None else np.nan,
                            z_min_slow=float(z_min_slow[i]) if z_min_slow is not None else np.nan,
                        )
                    )
                return t_vec, entries
        except Exception:
            # Ignore cache load errors and fall back to recompute
            pass

    entries: List[TFUnitTrace] = []
    total = len(cluster_ids)
    # Force tqdm usage if available by passing use_tqdm=True explicitly if show_progress is requested
    pr2 = Progress("Computing TF traces", total, use_tqdm=True) if (show_progress and total) else None
    if pr2 is not None:
        pr2.start()

    # Parallel path
    if parallel and total > 0:
        if n_workers is None:
            n_workers = min(os.cpu_count() or 1, 8)
        # Build args for each cluster
        trace_start = cfg.trace_pre  # None when not using wider window
        tasks = []
        for cid in cluster_ids:
            c = next((x for x in session.clusters if int(x.cluster_id) == int(cid)), None)
            if c is None:
                continue
            st = np.asarray(c.spike_times, dtype=float).flatten()
            tasks.append((int(cid), st, fast_times, slow_times, t_vec,
                          cfg.pre_window, cfg.post_window, cfg.dt, cfg.sigma_ms,
                          trace_start))

        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_compute_trace_for_cluster, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                try:
                    cid_ret, entry = fut.result()
                    results.append((cid_ret, entry))
                except Exception:
                    # If a worker fails, continue and let the serial path handle missing entries
                    pass

        # Preserve order of cluster_ids
        id_to_entry = {cid: entry for cid, entry in results}
        for cid in cluster_ids:
            e = id_to_entry.get(int(cid))
            if e is not None:
                entries.append(e)
            else:
                # Fall back to serial compute for missing cluster
                c = next((x for x in session.clusters if int(x.cluster_id) == int(cid)), None)
                if c is None:
                    continue
                st = np.asarray(c.spike_times, dtype=float).flatten()
                cid_ret, entry = _compute_trace_for_cluster((
                    int(cid), st, fast_times, slow_times, t_vec,
                    cfg.pre_window, cfg.post_window, cfg.dt, cfg.sigma_ms,
                    trace_start))
                entries.append(entry)
        if pr2 is not None:
            pr2.close()
    else:
        # Serial path (original behavior)
        for idx, cid in enumerate(cluster_ids, 1):
            c = next((x for x in session.clusters if int(x.cluster_id) == int(cid)), None)
            if c is None:
                continue
            st = np.asarray(c.spike_times, dtype=float).flatten()

            mf, mf_sem, _ = _mean_activity_per_unit(
                st, fast_times, cfg.pre_window, cfg.post_window, cfg.dt, cfg.sigma_ms,
                trace_start=cfg.trace_pre)
            if mf.size > 0:
                zf, fast_sd = _zscore_trace(mf, t_vec, cfg.pre_window, return_stats=True)
                zf_sem = mf_sem / fast_sd if np.isfinite(fast_sd) and fast_sd > 0 else np.zeros_like(mf_sem)
                zf_peak = float(np.nanmax(zf[post_mask])) if np.any(post_mask) else np.nan
                zf_trough = float(np.nanmin(zf[post_mask])) if np.any(post_mask) else np.nan
            else:
                zf = np.full_like(t_vec, np.nan)
                zf_sem = np.zeros_like(t_vec)
                zf_peak = np.nan
                zf_trough = np.nan

            ms, ms_sem, _ = _mean_activity_per_unit(
                st, slow_times, cfg.pre_window, cfg.post_window, cfg.dt, cfg.sigma_ms,
                trace_start=cfg.trace_pre)
            if ms.size > 0:
                zs, slow_sd = _zscore_trace(ms, t_vec, cfg.pre_window, return_stats=True)
                zs_sem = ms_sem / slow_sd if np.isfinite(slow_sd) and slow_sd > 0 else np.zeros_like(ms_sem)
                zs_peak = float(np.nanmax(zs[post_mask])) if np.any(post_mask) else np.nan
                zs_trough = float(np.nanmin(zs[post_mask])) if np.any(post_mask) else np.nan
            else:
                zs = np.full_like(t_vec, np.nan)
                zs_sem = np.zeros_like(t_vec)
                zs_peak = np.nan
                zs_trough = np.nan

            entries.append(
                TFUnitTrace(
                    cluster_id=cid,
                    fast_z=zf,
                    fast_z_sem=zf_sem,
                    slow_z=zs,
                    slow_z_sem=zs_sem,
                    z_max_fast=zf_peak,
                    z_min_fast=zf_trough,
                    z_max_slow=zs_peak,
                    z_min_slow=zs_trough,
                )
            )

            if pr2 is not None:
                pr2.update(idx)

        if pr2 is not None:
            pr2.close()

    # Save cache if requested
    if cache_path is not None:
        try:
            p = Path(cache_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            cluster_ids_arr = np.array([int(e.cluster_id) for e in entries], dtype=int)
            if entries and entries[0].fast_z.size:
                fast_z = np.stack([e.fast_z for e in entries])
                fast_z_sem = np.stack([e.fast_z_sem for e in entries])
            else:
                fast_z = np.zeros((0, t_vec.size))
                fast_z_sem = np.zeros((0, t_vec.size))
            if entries and entries[0].slow_z.size:
                slow_z = np.stack([e.slow_z for e in entries])
                slow_z_sem = np.stack([e.slow_z_sem for e in entries])
            else:
                slow_z = np.zeros((0, t_vec.size))
                slow_z_sem = np.zeros((0, t_vec.size))
            z_max_fast = np.array([e.z_max_fast for e in entries], dtype=float)
            z_min_fast = np.array([e.z_min_fast for e in entries], dtype=float)
            z_max_slow = np.array([e.z_max_slow for e in entries], dtype=float)
            z_min_slow = np.array([e.z_min_slow for e in entries], dtype=float)
            np.savez_compressed(
                str(p),
                t_vec=t_vec,
                cluster_ids=cluster_ids_arr,
                fast_z=fast_z,
                fast_z_sem=fast_z_sem,
                slow_z=slow_z,
                slow_z_sem=slow_z_sem,
                z_max_fast=z_max_fast,
                z_min_fast=z_min_fast,
                z_max_slow=z_max_slow,
                z_min_slow=z_min_slow,
            )
        except Exception:
            pass

    return t_vec, entries


def run_tf_pulse_screening(session, out_root: str, png_root: Optional[str] = None, cfg: Optional[TFRespPulseConfig] = None, selection_csv: Optional[str] = None, generate_grid: bool = True, show_progress: bool = False) -> Dict[str, str]:
    if cfg is None:
        cfg = TFRespPulseConfig()
    out_dir = Path(out_root) / f"{getattr(session,'subject','unknown')}_{getattr(session,'session_name','unknown')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = Path(png_root) / out_dir.name if png_root is not None else None

    # Determine cluster set
    kept_ids = load_kept_ids(session, selection_csv) if cfg.kept_only else None
    cluster_ids = [int(c.cluster_id) for c in session.clusters if (kept_ids is None or int(c.cluster_id) in kept_ids)]

    # Collect pulse times (allow progress updates) — compute once and pass into traces
    fast_times, slow_times = _collect_pulses(session, cfg, show_progress=show_progress)

    rows = []
    mean_traces_fast = []
    mean_traces_slow = []
    t_axis = None

    t_vec, trace_entries = collect_tf_pulse_traces(
        session, cfg=cfg, selection_csv=selection_csv, show_progress=show_progress, fast_times=fast_times, slow_times=slow_times
    )

    traces_by_id = {entry.cluster_id: entry for entry in trace_entries}

    for cid in cluster_ids:
        entry = traces_by_id.get(cid)
        if entry is None:
            continue
        # Consider both polarities: responsiveness can be an increase or a decrease.
        # Fast pulses: responsive if either a significant increase after fast pulses
        # (z_max_fast >= z_thresh) or a significant decrease after fast pulses
        # (z_min_fast <= -z_thresh).
        fast_resp = bool(
            (np.isfinite(entry.z_max_fast) and (entry.z_max_fast >= cfg.z_thresh))
            or (np.isfinite(entry.z_min_fast) and (entry.z_min_fast <= -cfg.z_thresh))
        )
        # Slow pulses: responsive if either a significant decrease after slow pulses
        # (z_min_slow <= -z_thresh) or a significant increase after slow pulses
        # (z_max_slow >= z_thresh).
        slow_resp = bool(
            (np.isfinite(entry.z_min_slow) and (entry.z_min_slow <= -cfg.z_thresh))
            or (np.isfinite(entry.z_max_slow) and (entry.z_max_slow >= cfg.z_thresh))
        )

        # For plotting aggregates: use underlying mean traces if available
        # Since collect_tf_pulse_traces only stores z-traces, recompute mean activity when needed
        c = next((x for x in session.clusters if int(x.cluster_id) == int(cid)), None)
        if c is None:
            continue
        st = np.asarray(c.spike_times, dtype=float).flatten()
        if fast_resp:
            mf, _, _ = _mean_activity_per_unit(st, fast_times, cfg.pre_window, cfg.post_window, cfg.dt, cfg.sigma_ms)
            if mf.size > 0 and float(np.nanmean(mf)) >= cfg.min_mean_rate_for_plot:
                mean_traces_fast.append(mf)
                t_axis = t_vec
        if slow_resp:
            ms, _, _ = _mean_activity_per_unit(st, slow_times, cfg.pre_window, cfg.post_window, cfg.dt, cfg.sigma_ms)
            if ms.size > 0 and float(np.nanmean(ms)) >= cfg.min_mean_rate_for_plot:
                mean_traces_slow.append(ms)
                t_axis = t_vec

        rows.append({
            "cluster_id": cid,
            "fast_responsive": bool(fast_resp),
            "slow_responsive": bool(slow_resp),
            "z_max_fast": float(entry.z_max_fast) if np.isfinite(entry.z_max_fast) else np.nan,
            "z_min_fast": float(entry.z_min_fast) if np.isfinite(entry.z_min_fast) else np.nan,
            "z_max_slow": float(entry.z_max_slow) if np.isfinite(entry.z_max_slow) else np.nan,
            "z_min_slow": float(entry.z_min_slow) if np.isfinite(entry.z_min_slow) else np.nan,
            "n_fast_pulses_used": int(fast_times.size),
            "n_slow_pulses_used": int(slow_times.size),
        })

    df = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
    csv_path = out_dir / "tf_pulse_units.csv"
    df.to_csv(csv_path, index=False)

    # Save pulse times for provenance
    pt_df = pd.DataFrame({
        "fast_times": pd.Series(fast_times, dtype=float),
        "slow_times": pd.Series(slow_times, dtype=float),
    })
    pt_csv = out_dir / "tf_pulse_times.csv"
    pt_df.to_csv(pt_csv, index=False)

    paths: Dict[str, str] = {"csv": str(csv_path), "times": str(pt_csv)}

    # Save a single "both" grid for visual inspection (faster than multiple plots)
    try:
        if png_dir is not None and generate_grid:
            png_dir.mkdir(parents=True, exist_ok=True)
            grid_path = png_dir / "tf_pulse_grid_both.png"
            gp = plot_tf_pulse_grid(session, str(grid_path), cfg=cfg, selection_csv=selection_csv, n_cols=12, which="both", show_progress=show_progress)
            paths["grid_png"] = str(gp)
    except Exception:
        pass

    return paths


def plot_tf_pulse_grid(
    session,
    out_png: str,
    cfg: Optional[TFRespPulseConfig] = None,
    selection_csv: Optional[str] = None,
    n_cols: int = 8,
    which: str = "slow",
    z_line: Optional[float] = None,
    filter_ids: Optional[Sequence[int]] = None,
    min_abs_z: Optional[float] = None,
    sort_by_strength: bool = False,
    show_progress: bool = True,
    save_csv_path: Optional[str] = None,
) -> str:
    """Save a grid of per-unit z-scored mean traces for visual threshold inspection.

    which: "fast", "slow", or "both" (overlays). Defaults to "slow".
    """
    if cfg is None:
        cfg = TFRespPulseConfig()
    t_vec, entries = collect_tf_pulse_traces(session, cfg=cfg, selection_csv=selection_csv, show_progress=show_progress)
    if filter_ids is not None:
        allowed = {int(x) for x in filter_ids}
        entries = [e for e in entries if e.cluster_id in allowed]
    if min_abs_z is not None:
        entries = [
            e
            for e in entries
            if (np.isfinite(e.z_max_fast) and e.z_max_fast >= min_abs_z)
            or (np.isfinite(e.z_min_slow) and e.z_min_slow <= -min_abs_z)
        ]

    if sort_by_strength:
        def _get_max_z(e):
            vals = []
            if np.isfinite(e.z_max_fast): vals.append(abs(e.z_max_fast))
            if np.isfinite(e.z_min_fast): vals.append(abs(e.z_min_fast))
            if np.isfinite(e.z_max_slow): vals.append(abs(e.z_max_slow))
            if np.isfinite(e.z_min_slow): vals.append(abs(e.z_min_slow))
            return max(vals) if vals else 0.0
        entries.sort(key=_get_max_z, reverse=True)

    # Save CSV if requested
    if save_csv_path:
        import pandas as pd
        rows = []
        for e in entries:
            rows.append({
                "cluster_id": e.cluster_id,
                "z_max_fast": e.z_max_fast,
                "z_min_fast": e.z_min_fast,
                "z_max_slow": e.z_max_slow,
                "z_min_slow": e.z_min_slow,
            })
        try:
            print(f"[plot_tf_pulse_grid] Writing CSV to {save_csv_path} with {len(rows)} rows...")
            pd.DataFrame(rows).to_csv(save_csv_path, index=False)
            print(f"[plot_tf_pulse_grid] Successfully wrote CSV to {save_csv_path}")
        except Exception as exc:
            print(f"[plot_tf_pulse_grid] ERROR writing CSV to {save_csv_path}: {exc}")

    ids = [e.cluster_id for e in entries]
    n = len(ids)
    if n == 0:
        return out_png
    n_cols = max(1, n_cols)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 1.8 * n_rows), sharex=True, sharey=True)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    entry_lookup = {e.cluster_id: e for e in entries}

    for i, cid in enumerate(ids):
        ax = axes[i]
        entry = entry_lookup[cid]
        if which in ("slow", "both"):
            slow_mean = entry.slow_z
            slow_sem = entry.slow_z_sem
            ax.plot(t_vec, slow_mean, color="#d62728", label="slow")
            ax.fill_between(
                t_vec,
                slow_mean - slow_sem,
                slow_mean + slow_sem,
                color="#d62728",
                alpha=0.25,
                linewidth=0,
            )
        if which in ("fast", "both"):
            fast_mean = entry.fast_z
            fast_sem = entry.fast_z_sem
            ax.plot(t_vec, fast_mean, color="#1f77b4", label="fast")
            ax.fill_between(
                t_vec,
                fast_mean - fast_sem,
                fast_mean + fast_sem,
                color="#1f77b4",
                alpha=0.25,
                linewidth=0,
            )
        ax.axvline(0, color="k", linestyle="--", lw=0.8)
        if z_line is not None:
            ax.axhline(float(z_line), color="#888", linestyle=":", lw=0.8)
            ax.axhline(-float(z_line), color="#888", linestyle=":", lw=0.8)
        ax.set_title(f"clu {cid}", fontsize=8)
        if i % n_cols == 0:
            ax.set_ylabel("z-score")
        if i // n_cols == n_rows - 1:
            ax.set_xlabel("time (s)")
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")
    for j in range(i + 1, n_rows * n_cols):
        axes[j].axis("off")
    ttl = "TF pulse: per-unit z-scored mean traces"
    if z_line is not None:
        ttl += f"  (z±{z_line:g})"
    fig.suptitle(ttl, y=0.995, fontsize=12)
    fig.tight_layout(h_pad=0.2, w_pad=0.1)
    p = Path(out_png)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def plot_tf_pulse_high_z_summary(
    session,
    out_png: str,
    cfg: Optional[TFRespPulseConfig] = None,
    selection_csv: Optional[str] = None,
    n_cols: int = 8,
    which: str = "both",
    min_abs_z: float = DEFAULT_Z_THRESH_TF,
) -> Optional[str]:
    """Plot only the units exceeding |z| >= min_abs_z and add a summary subplot.

    The summary subplot flips negative-going slow responses so the population
    modulation is always positive, making fast/slow label irrelevant.
    """
    if cfg is None:
        cfg = TFRespPulseConfig()
    t_vec, entries = collect_tf_pulse_traces(session, cfg=cfg, selection_csv=selection_csv)
    entries = [
        e
        for e in entries
        if (np.isfinite(e.z_max_fast) and e.z_max_fast >= min_abs_z)
        or (np.isfinite(e.z_min_slow) and (-e.z_min_slow) >= min_abs_z)
    ]
    if not entries:
        return None

    n = len(entries)
    n_cols = max(1, min(n_cols, n))
    n_rows = int(np.ceil(n / n_cols))
    height = 1.8 * n_rows + 2.2
    fig = plt.figure(figsize=(2.2 * n_cols, height))
    gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[1.0] * n_rows + [0.9], hspace=0.35, wspace=0.15)

    norm_traces: List[np.ndarray] = []
    for idx in range(n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        if idx >= n:
            ax.axis("off")
            continue
        entry = entries[idx]
        if which in ("slow", "both"):
            slow_mean = entry.slow_z
            slow_sem = entry.slow_z_sem
            ax.plot(t_vec, slow_mean, color="#d62728", label="slow")
            ax.fill_between(
                t_vec,
                slow_mean - slow_sem,
                slow_mean + slow_sem,
                color="#d62728",
                alpha=0.25,
                linewidth=0,
            )
        if which in ("fast", "both"):
            fast_mean = entry.fast_z
            fast_sem = entry.fast_z_sem
            ax.plot(t_vec, fast_mean, color="#1f77b4", label="fast")
            ax.fill_between(
                t_vec,
                fast_mean - fast_sem,
                fast_mean + fast_sem,
                color="#1f77b4",
                alpha=0.25,
                linewidth=0,
            )
        ax.axvline(0, color="k", linestyle="--", lw=0.8)
        ax.axhline(min_abs_z, color="#888", linestyle=":", lw=0.7)
        ax.axhline(-min_abs_z, color="#888", linestyle=":", lw=0.7)
        ax.set_title(f"clu {entry.cluster_id}", fontsize=8)
        if col == 0:
            ax.set_ylabel("z-score")
        if row == n_rows - 1:
            ax.set_xlabel("time (s)")
        if idx == 0:
            ax.legend(fontsize=6, loc="upper right")

        fast_peak = entry.z_max_fast if np.isfinite(entry.z_max_fast) else -np.inf
        slow_mag = -entry.z_min_slow if np.isfinite(entry.z_min_slow) else -np.inf
        if fast_peak >= min_abs_z and fast_peak >= slow_mag:
            norm_traces.append(entry.fast_z)
        elif slow_mag >= min_abs_z:
            norm_traces.append(-entry.slow_z)

    summary_ax = fig.add_subplot(gs[-1, :])
    if norm_traces:
        stack = np.stack(norm_traces)
        mean_trace = np.nanmean(stack, axis=0)
        if stack.shape[0] > 1:
            sem_trace = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(stack.shape[0])
        else:
            sem_trace = np.zeros_like(mean_trace)
        summary_ax.plot(t_vec, mean_trace, color="#4e79a7", lw=2)
        summary_ax.fill_between(
            t_vec,
            mean_trace - sem_trace,
            mean_trace + sem_trace,
            color="#4e79a7",
            alpha=0.2,
            linewidth=0,
        )
        summary_ax.axvline(0, color="k", linestyle="--", lw=0.8)
        summary_ax.set_ylabel("aligned z-score")
        summary_ax.set_xlabel("time (s)")
        summary_ax.set_title(
            f"Mean |z| response (n={len(norm_traces)})", fontsize=10
        )
    else:
        summary_ax.text(0.5, 0.5, "No units above cutoff", transform=summary_ax.transAxes, ha="center", va="center")
        summary_ax.set_axis_off()

    fig.suptitle(f"TF pulses: |z| >= {min_abs_z:g}", y=0.995, fontsize=12)
    fig.tight_layout(h_pad=0.4)
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
