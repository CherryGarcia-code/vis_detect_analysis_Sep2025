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
