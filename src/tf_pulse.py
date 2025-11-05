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

from src.su_analysis import load_kept_ids
from scipy.ndimage import gaussian_filter1d


@dataclass
class TFRespPulseConfig:
    # Pulse classification thresholds on log2(TF)
    fast_thresh_log2: float = 0.25
    slow_thresh_log2: float = -0.25
    # How to read baseline TF vector and convert to times
    baseline_stride: int = 3  # sample every 3rd element (legacy convention)
    sample_period: float = 0.05  # seconds per baseline sample (50 ms)
    # Time windows around pulses
    pre_window: Tuple[float, float] = (-0.4, 0.0)
    post_window: Tuple[float, float] = (0.0, 0.5)
    # Smoothing for spike KDE
    dt: float = 0.001  # seconds per bin for smoothing grid
    sigma_ms: float = 13.3  # Gaussian sigma in ms
    # Trial constraints (enabled by default to match stricter legacy v2 script)
    use_constraints: bool = True
    min_after_baseline: float = 1.0
    min_before_change: float = 1.0
    min_before_outcome_fa_abort: float = 2.0
    # Unit selection
    kept_only: bool = True
    # Z-score threshold
    z_thresh: float = 3.0
    # Plotting aggregation filter
    min_mean_rate_for_plot: float = 0.01


def _safe_log2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log2(x)
    y[~np.isfinite(y)] = np.nan
    return y


def _per_trial_event_times(session, key: str) -> np.ndarray:
    from src.align import get_event_times_by_trial

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


def _collect_pulses(session, cfg: TFRespPulseConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fast_times, slow_times) arrays of absolute pulse times across trials
    after applying constraints relative to Baseline_ON, Change_ON, and FA/abort.
    """
    trials = getattr(session, "trials", []) or []
    base_by_trial = _per_trial_event_times(session, "Baseline_ON")
    change_by_trial = _per_trial_event_times(session, "Change_ON")

    fast_times: List[float] = []
    slow_times: List[float] = []

    for i, t in enumerate(trials):
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


def _mean_activity_per_unit(spike_times: np.ndarray, pulses: np.ndarray, pre_window: Tuple[float, float], post_window: Tuple[float, float], dt: float, sigma_ms: float) -> Tuple[np.ndarray, np.ndarray]:
    pulses = np.asarray(pulses, dtype=float)
    pulses = pulses[np.isfinite(pulses)]
    if pulses.size == 0:
        return np.array([]), np.array([])
    full0, full1 = pre_window[0], post_window[1]
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
        return np.array([]), t_vec
    arr = np.stack(traces, axis=0)
    return np.nanmean(arr, axis=0), t_vec


def _zscore_trace(mean_trace: np.ndarray, t_vec: np.ndarray, pre_window: Tuple[float, float]) -> np.ndarray:
    if mean_trace.size == 0:
        return mean_trace
    pre_mask = (t_vec >= pre_window[0]) & (t_vec < pre_window[1])
    mu = float(np.nanmean(mean_trace[pre_mask])) if np.any(pre_mask) else 0.0
    sd = float(np.nanstd(mean_trace[pre_mask])) if np.any(pre_mask) else 0.0
    if not np.isfinite(sd) or sd <= 0:
        # Avoid divide-by-zero: return zero-centered trace
        return mean_trace * 0.0
    return (mean_trace - mu) / sd


 


def run_tf_pulse_screening(session, out_root: str, png_root: Optional[str] = None, cfg: Optional[TFRespPulseConfig] = None, selection_csv: Optional[str] = None, generate_grid: bool = True) -> Dict[str, str]:
    if cfg is None:
        cfg = TFRespPulseConfig()
    out_dir = Path(out_root) / f"{getattr(session,'subject','unknown')}_{getattr(session,'session_name','unknown')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = Path(png_root) / out_dir.name if png_root is not None else None

    # Determine cluster set
    kept_ids = load_kept_ids(session, selection_csv) if cfg.kept_only else None
    cluster_ids = [int(c.cluster_id) for c in session.clusters if (kept_ids is None or int(c.cluster_id) in kept_ids)]

    # Collect pulse times
    fast_times, slow_times = _collect_pulses(session, cfg)

    rows = []
    mean_traces_fast = []
    mean_traces_slow = []
    t_axis = None

    for cid in cluster_ids:
        c = next((x for x in session.clusters if int(x.cluster_id) == int(cid)), None)
        if c is None:
            continue
        st = np.asarray(c.spike_times, dtype=float).flatten()
        # Compute mean trace across pulses first (legacy behavior), then z-score
        # Fast
        mf, t_vec = _mean_activity_per_unit(st, fast_times, cfg.pre_window, cfg.post_window, cfg.dt, cfg.sigma_ms)
        zf_trace = _zscore_trace(mf, t_vec, cfg.pre_window) if mf.size > 0 else np.array([])
        post_mask = (t_vec >= cfg.post_window[0]) & (t_vec < cfg.post_window[1]) if zf_trace.size > 0 else np.array([])
        zf = float(np.nanmax(zf_trace[post_mask])) if zf_trace.size > 0 and np.any(post_mask) else np.nan
        fast_resp = bool(np.isfinite(zf) and (zf >= cfg.z_thresh))

        # Slow
        ms, _ = _mean_activity_per_unit(st, slow_times, cfg.pre_window, cfg.post_window, cfg.dt, cfg.sigma_ms)
        zs_trace = _zscore_trace(ms, t_vec, cfg.pre_window) if ms.size > 0 else np.array([])
        zs = float(np.nanmin(zs_trace[post_mask])) if zs_trace.size > 0 and np.any(post_mask) else np.nan
        slow_resp = bool(np.isfinite(zs) and (zs <= -cfg.z_thresh))

        # For plotting aggregates: store per-unit mean smoothed trace when responsive and mean rate is sufficient
        if fast_resp and mf.size > 0 and float(np.nanmean(mf)) >= cfg.min_mean_rate_for_plot:
            mean_traces_fast.append(mf)
            t_axis = t_vec
        if slow_resp and ms.size > 0 and float(np.nanmean(ms)) >= cfg.min_mean_rate_for_plot:
            mean_traces_slow.append(ms)
            t_axis = t_vec

        rows.append({
            "cluster_id": cid,
            "fast_responsive": bool(fast_resp),
            "slow_responsive": bool(slow_resp),
            "z_max_fast": float(zf) if np.isfinite(zf) else np.nan,
            "z_min_slow": float(zs) if np.isfinite(zs) else np.nan,
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
            gp = plot_tf_pulse_grid(session, str(grid_path), cfg=cfg, selection_csv=selection_csv, n_cols=12, which="both")
            paths["grid_png"] = str(gp)
    except Exception:
        pass

    return paths


def plot_tf_pulse_grid(session, out_png: str, cfg: Optional[TFRespPulseConfig] = None, selection_csv: Optional[str] = None, n_cols: int = 8, which: str = "slow", z_line: Optional[float] = None) -> str:
    """Save a grid of per-unit z-scored mean traces for visual threshold inspection.

    which: "fast", "slow", or "both" (overlays). Defaults to "slow".
    """
    if cfg is None:
        cfg = TFRespPulseConfig()
    kept_ids = load_kept_ids(session, selection_csv) if cfg.kept_only else None
    cluster_ids = [int(c.cluster_id) for c in session.clusters if (kept_ids is None or int(c.cluster_id) in kept_ids)]
    fast_times, slow_times = _collect_pulses(session, cfg)

    full0, full1 = cfg.pre_window[0], cfg.post_window[1]
    t_vec = np.arange(full0, full1, cfg.dt, dtype=float)

    traces_fast = {}
    traces_slow = {}
    for cid in cluster_ids:
        c = next((x for x in session.clusters if int(x.cluster_id) == int(cid)), None)
        if c is None:
            continue
        st = np.asarray(c.spike_times, dtype=float).flatten()
        mf, _ = _mean_activity_per_unit(st, fast_times, cfg.pre_window, cfg.post_window, cfg.dt, cfg.sigma_ms)
        ms, _ = _mean_activity_per_unit(st, slow_times, cfg.pre_window, cfg.post_window, cfg.dt, cfg.sigma_ms)
        zf = _zscore_trace(mf, t_vec, cfg.pre_window) if mf.size > 0 else np.zeros_like(t_vec)
        zs = _zscore_trace(ms, t_vec, cfg.pre_window) if ms.size > 0 else np.zeros_like(t_vec)
        traces_fast[cid] = zf
        traces_slow[cid] = zs

    ids = sorted(traces_slow.keys())
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

    for i, cid in enumerate(ids):
        ax = axes[i]
        if which in ("slow", "both"):
            ax.plot(t_vec, traces_slow[cid], color="#d62728", label="slow")
        if which in ("fast", "both"):
            ax.plot(t_vec, traces_fast[cid], color="#1f77b4", label="fast")
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
