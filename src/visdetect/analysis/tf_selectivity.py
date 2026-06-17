"""TF-pulse fast-minus-slow selectivity (Lohse 2025) for responder ID.

Replaces the retired source-level drift-detrend approach (tf_drift.py): the
pre-pulse firing-rate ramp is a within-trial temporal-expectation signal at the
same timescale as the response, so it cannot be modelled out. The fast-minus-
slow difference cancels that common-mode ramp by symmetry (the ramp is trial-
locked, not pulse-identity-locked; fast and slow pulses sample it identically),
with no detrend and no model.

Pipeline (per unit; all-trials in Phase B, per-state later):
  corrected pulses (fixed _collect_pulses)
    -> per-pulse smoothed Hz matrices (fast, slow) over [trace_pre, +0.5] s
    -> shared per-unit baseline (mu_b, sigma_b) pooled over the pre-window of
       BOTH mean traces  (fixes the old per-condition separate-baseline bug)
    -> selectivity(t) = (fast_hz - slow_hz) / max(sigma_b, eps)
    -> signed post-window peak / latency / AUC / half-width
    -> label-shuffle null (permute fast/slow labels, counts fixed) -> shuffle p
    -> within-unit split-half reliability of the selectivity
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from visdetect.analysis.constants import TF_PULSE_TRACE_PRE
from visdetect.analysis.tf_pulse import (
    TFRespPulseConfig,
    _collect_pulses,
    _smooth_binned_activity,
)


@dataclass
class TFSelectivityConfig:
    """Selectivity config. Wraps a TFRespPulseConfig (trace extended to -1.0 s)
    and adds the null/sufficiency knobs."""
    pulse: TFRespPulseConfig = field(
        default_factory=lambda: TFRespPulseConfig(trace_pre=TF_PULSE_TRACE_PRE)
    )
    n_shuffles: int = 200
    seed: int = 42
    eps: float = 1e-6
    min_pulses_per_label: int = 20


@dataclass
class TFUnitSelectivity:
    cluster_id: int
    t_vec: np.ndarray
    fast_hz: np.ndarray
    slow_hz: np.ndarray
    selectivity: np.ndarray
    fast_z: np.ndarray
    slow_z: np.ndarray
    baseline_mu: float
    baseline_sd: float
    sel_peak: float            # signed; selectivity value at |peak| in post window
    sel_peak_latency: float    # s
    sel_auc: float             # signed area under selectivity in post window
    sel_half_width: float      # s; width at half-max around the peak
    fast_peak: float           # signed fast_z post-window peak (sub-typing)
    slow_peak: float           # signed slow_z post-window peak (sub-typing)
    n_fast: int
    n_slow: int
    null_peak_mean: float
    null_peak_sd: float
    sel_z_vs_null: float
    shuffle_p: float
    split_half_r: float
    sufficient: bool


def _time_vector(cfg: TFSelectivityConfig) -> np.ndarray:
    p = cfg.pulse
    full0 = p.trace_pre if p.trace_pre is not None else p.pre_window[0]
    return np.arange(full0, p.post_window[1], p.dt, dtype=float)


def _per_pulse_rate_matrix(
    spike_times: np.ndarray,
    pulse_times: np.ndarray,
    t_vec: np.ndarray,
    dt: float,
    sigma_ms: float,
) -> np.ndarray:
    """(n_pulses, n_time) matrix of per-pulse Gaussian-smoothed rate in Hz."""
    st = np.asarray(spike_times, dtype=float).ravel()
    pulse_times = np.asarray(pulse_times, dtype=float).ravel()
    pulse_times = pulse_times[np.isfinite(pulse_times)]
    if pulse_times.size == 0:
        return np.zeros((0, t_vec.size), dtype=float)
    sigma_bins = (sigma_ms / 1000.0) / dt
    lo, hi = float(t_vec[0]), float(t_vec[-1] + dt)
    rows = np.empty((pulse_times.size, t_vec.size), dtype=float)
    for k, tp in enumerate(pulse_times):
        rel = st - tp
        rel = rel[(rel >= lo) & (rel < hi)]
        rows[k] = _smooth_binned_activity(rel, t_vec, sigma_bins) / dt
    return rows


def _shared_baseline(
    fast_hz: np.ndarray,
    slow_hz: np.ndarray,
    t_vec: np.ndarray,
    pre_window: Tuple[float, float],
    eps: float,
) -> Tuple[float, float]:
    """One (mu, sd) pooled over the pre-window bins of BOTH mean traces.

    Using a single shared sigma for fast and slow is the fix for the old
    circular separate-baseline z-scoring (CLAUDE.md "circular baseline").
    """
    pre_mask = (t_vec >= pre_window[0]) & (t_vec < pre_window[1])
    if not np.any(pre_mask):
        return 0.0, 1.0
    pooled = np.concatenate([fast_hz[pre_mask], slow_hz[pre_mask]])
    mu = float(np.nanmean(pooled))
    sd = float(np.nanstd(pooled))
    if not np.isfinite(sd) or sd <= eps:
        sd = 1.0
    return mu, sd


def _post_metrics(
    trace: np.ndarray,
    t_vec: np.ndarray,
    post_window: Tuple[float, float],
) -> Tuple[float, float, float, float]:
    """Signed peak, peak latency (s), signed AUC, and half-width (s) in post."""
    post_mask = (t_vec >= post_window[0]) & (t_vec < post_window[1])
    if not np.any(post_mask):
        return np.nan, np.nan, np.nan, np.nan
    seg = trace[post_mask]
    tt = t_vec[post_mask]
    if not np.any(np.isfinite(seg)):
        return np.nan, np.nan, np.nan, np.nan
    i_peak = int(np.nanargmax(np.abs(seg)))
    peak = float(seg[i_peak])
    latency = float(tt[i_peak])
    auc = float(np.trapz(seg, tt))
    half = abs(peak) / 2.0
    lo = i_peak
    while lo > 0 and abs(seg[lo - 1]) >= half:
        lo -= 1
    hi = i_peak
    while hi < seg.size - 1 and abs(seg[hi + 1]) >= half:
        hi += 1
    half_width = float(tt[hi] - tt[lo])
    return peak, latency, auc, half_width


def compute_unit_selectivity(spike_times, fast_times, slow_times, cfg=None, rng=None) -> TFUnitSelectivity:
    if cfg is None:
        cfg = TFSelectivityConfig()
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    p = cfg.pulse
    t_vec = _time_vector(cfg)
    mat_fast = _per_pulse_rate_matrix(spike_times, fast_times, t_vec, p.dt, p.sigma_ms)
    mat_slow = _per_pulse_rate_matrix(spike_times, slow_times, t_vec, p.dt, p.sigma_ms)
    n_fast, n_slow = mat_fast.shape[0], mat_slow.shape[0]
    sufficient = (n_fast >= cfg.min_pulses_per_label) and (n_slow >= cfg.min_pulses_per_label)

    if n_fast == 0 or n_slow == 0:
        nan = np.full(t_vec.size, np.nan)
        return TFUnitSelectivity(
            cluster_id=-1, t_vec=t_vec, fast_hz=nan.copy(), slow_hz=nan.copy(),
            selectivity=nan.copy(), fast_z=nan.copy(), slow_z=nan.copy(),
            baseline_mu=np.nan, baseline_sd=np.nan, sel_peak=np.nan,
            sel_peak_latency=np.nan, sel_auc=np.nan, sel_half_width=np.nan,
            fast_peak=np.nan, slow_peak=np.nan, n_fast=n_fast, n_slow=n_slow,
            null_peak_mean=np.nan, null_peak_sd=np.nan, sel_z_vs_null=np.nan,
            shuffle_p=np.nan, split_half_r=np.nan, sufficient=False)

    fast_hz = np.nanmean(mat_fast, axis=0)
    slow_hz = np.nanmean(mat_slow, axis=0)
    mu_b, sd_b = _shared_baseline(fast_hz, slow_hz, t_vec, p.pre_window, cfg.eps)
    selectivity = (fast_hz - slow_hz) / sd_b
    fast_z = (fast_hz - mu_b) / sd_b
    slow_z = (slow_hz - mu_b) / sd_b
    sel_peak, sel_lat, sel_auc, sel_hw = _post_metrics(selectivity, t_vec, p.post_window)
    fast_peak, _, _, _ = _post_metrics(fast_z, t_vec, p.post_window)
    slow_peak, _, _, _ = _post_metrics(slow_z, t_vec, p.post_window)

    # Label-shuffle null: permute fast/slow labels (counts fixed), keeping the
    # ramp/drift intact; destroys only the TF assignment.
    combined = np.vstack([mat_fast, mat_slow])
    n_total = combined.shape[0]
    post_mask = (t_vec >= p.post_window[0]) & (t_vec < p.post_window[1])
    null_peaks = np.empty(cfg.n_shuffles, dtype=float)
    for s in range(cfg.n_shuffles):
        perm = rng.permutation(n_total)
        f = np.nanmean(combined[perm[:n_fast]], axis=0)
        sl = np.nanmean(combined[perm[n_fast:]], axis=0)
        sel_s = (f - sl) / sd_b
        null_peaks[s] = float(np.nanmax(np.abs(sel_s[post_mask]))) if np.any(post_mask) else np.nan
    null_peak_mean = float(np.nanmean(null_peaks))
    null_peak_sd = float(np.nanstd(null_peaks))
    obs = abs(sel_peak)
    sel_z_vs_null = (obs - null_peak_mean) / null_peak_sd if null_peak_sd > cfg.eps else np.nan
    shuffle_p = float((1 + np.sum(null_peaks >= obs)) / (1 + cfg.n_shuffles))

    # split-half filled in Task 7
    split_half_r = np.nan

    return TFUnitSelectivity(
        cluster_id=-1, t_vec=t_vec, fast_hz=fast_hz, slow_hz=slow_hz,
        selectivity=selectivity, fast_z=fast_z, slow_z=slow_z,
        baseline_mu=mu_b, baseline_sd=sd_b, sel_peak=sel_peak,
        sel_peak_latency=sel_lat, sel_auc=sel_auc, sel_half_width=sel_hw,
        fast_peak=fast_peak, slow_peak=slow_peak, n_fast=n_fast, n_slow=n_slow,
        null_peak_mean=null_peak_mean, null_peak_sd=null_peak_sd,
        sel_z_vs_null=sel_z_vs_null, shuffle_p=shuffle_p,
        split_half_r=split_half_r, sufficient=sufficient)
