"""Per-neuron Poisson encoding GLM (Khilkevich-Lohse 2024 replication).

50-ms-binned, temporally-unfolded (FIR) design matrix -> ridge-Poisson per
neuron with nested 10-fold CV -> TF-responsive identification by the paper's
two held-out criteria (C1 fast-minus-slow prediction r>0.2; C2 ablation t-test
P<0.01 across folds). See docs/superpowers/specs/2026-06-18-tf-glm-replication-design.md.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TFGLMConfig:
    bin_s: float = 0.05
    # FIR kernel windows (seconds, relative to event); (lo, hi) inclusive of lo,
    # exclusive of hi, stepped by bin_s.
    kern: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "tf":            (0.0, 1.5),
        "trial_start":   (0.0, 1.0),
        "time_in_base":  (0.0, 0.0),    # ramp handled as a single graded column
        "change":        (0.0, 2.0),    # per change-size (applied 6x)
        "lick_prep":     (-1.25, 0.0),
        "lick_exec":     (0.0, 0.5),
        "reward":        (0.0, 0.4),
        "abort":         (-1.25, 0.25),
        "wheel":         (-0.05, 0.8),
        "phase":         (0.0, 0.0),    # 12 bins x up/down, no temporal unfold
    })
    sd_pulse: float = 0.5               # fast/slow = +/-0.5 SD of baseline TF
    pulse_eval_win: Tuple[float, float] = (-0.15, 0.75)  # PETH window around pulses
    n_folds: int = 10
    lambdas: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    c1_r_thresh: float = 0.2
    c2_p_thresh: float = 0.01
    seed: int = 42
    include_phase: bool = False         # off for DMS-first; on for cortex


def trial_bin_edges(t_start: float, t_end: float, bin_s: float) -> np.ndarray:
    """Left edges of 50-ms bins spanning [t_start, t_end)."""
    n = int(np.floor((t_end - t_start) / bin_s + 1e-9))
    return t_start + np.arange(max(n, 0)) * bin_s


def bin_spike_counts(spike_times: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Spike count per 50-ms bin. Bin i = [edges[i], edges[i]+bin_s)."""
    st = np.asarray(spike_times, dtype=float).ravel()
    if bin_edges.size == 0:
        return np.zeros(0, dtype=float)
    bin_s = bin_edges[1] - bin_edges[0] if bin_edges.size > 1 else 0.05
    full = np.append(bin_edges, bin_edges[-1] + bin_s)
    counts, _ = np.histogram(st, bins=full)
    return counts.astype(float)
