"""Pure primitives for the Fig-5 e-h preparatory-activity reproduction
(Khilkevich & Lohse 2024, Methods pp.17-18). No I/O; unit-tested in isolation.

The paper's "significantly active" is |z of the trial-MEAN PETH| > 2.576, with z
computed against a 2 s pre-CHANGE baseline; the fraction of active units is
bootstrapped OVER NEURONS (not trials); the population activation onset uses a
100 ms / 80 ms / mean>0.1 rule. See the design spec
(docs/superpowers/specs/2026-07-20-fig5eh-preparatory-transient-sustained-nonTF-design.md)
§2 for verbatim quotes.
"""
from __future__ import annotations
import numpy as np

Z_ACTIVE = 2.576  # |z| threshold, P<0.01 two-sided (Fig 5e z-test)


def baseline_mean_sd(baseline_binned) -> tuple[float, float]:
    """mu, sd of the TRIAL-AVERAGED pre-change baseline PETH across time bins.

    Khilkevich & Lohse Fig 5 z-score the trial-MEAN PETH by "the mean and s.d.
    estimated from activity during 2 s before the change onset" — i.e. of the
    (trial-averaged) baseline firing-rate trace, NOT of pooled single-trial bins.
    Pooling single trials inflates sd by ~sqrt(n_trials) of Poisson noise and
    collapses the fraction-active far below the paper's scale (verified: pooled
    peaks at 0.08 vs the paper's 0.5-0.9; the trial-averaged trace peaks ~0.83).

    Accepts (n_trials, n_bins) or a 1-D trace. sd<1e-6 -> max(mu, 1).
    """
    B = np.asarray(baseline_binned, float)
    trace = B if B.ndim == 1 else np.nanmean(B, axis=0)  # trial-average first
    trace = trace[np.isfinite(trace)]
    if trace.size == 0:
        return 0.0, 1.0
    mu, sd = float(np.mean(trace)), float(np.std(trace))
    if not np.isfinite(sd) or sd < 1e-6:
        sd = max(mu, 1.0)
    return mu, sd


def zscore_trace(mean_peth, mu, sd) -> np.ndarray:
    return (np.asarray(mean_peth, float) - mu) / sd


def active_mask(z, thresh=Z_ACTIVE) -> np.ndarray:
    return np.abs(np.asarray(z, float)) > thresh


def fraction_active(active_matrix, baseline_bins=None) -> np.ndarray:
    """Mean over units (rows) per bin; if baseline_bins (slice/index), subtract its mean."""
    A = np.asarray(active_matrix, float)
    frac = np.nanmean(A, axis=0)
    if baseline_bins is not None:
        frac = frac - np.nanmean(frac[baseline_bins])
    return frac


def bootstrap_fraction_ci(active_matrix, baseline_bins=None, n=5000, seed=42):
    """Bootstrap OVER UNITS. Returns (mean_frac, lo95, hi95) per bin."""
    A = np.asarray(active_matrix, float)
    nU = A.shape[0]
    base = fraction_active(A, baseline_bins)
    if nU < 3:
        return base, base, base
    rng = np.random.default_rng(seed)
    boots = np.empty((n, A.shape[1]))
    for b in range(n):
        boots[b] = fraction_active(A[rng.integers(0, nU, nU)], baseline_bins)
    return base, np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0)


def _first_sustained(cond, window_s, sustain_s, bin_s):
    need = int(round(sustain_s / bin_s))
    win = int(round(window_s / bin_s))
    for i in range(len(cond)):
        if cond[i] and cond[i:min(len(cond), i + win)].sum() >= need:
            return i
    return -1


def population_onset(t, mean_frac, ci_lo, *, window_s=0.1, sustain_s=0.08,
                     min_frac=0.1, bin_s=0.025) -> float:
    t = np.asarray(t, float)
    cond = (np.asarray(ci_lo, float) > 0) & (np.asarray(mean_frac, float) > min_frac)
    i = _first_sustained(cond, window_s, sustain_s, bin_s)
    return float(t[i]) if i >= 0 else np.nan


def cell_onset(t, z, *, thresh=Z_ACTIVE, window_s=0.1, sustain_s=0.08, bin_s=0.025) -> float:
    t = np.asarray(t, float)
    cond = np.abs(np.asarray(z, float)) > thresh
    i = _first_sustained(cond, window_s, sustain_s, bin_s)
    return float(t[i]) if i >= 0 else np.nan


def width_deciles(width, n=10):
    w = np.asarray(width, float)
    fin = w[np.isfinite(w)]
    edges = np.quantile(fin, np.linspace(0, 1, n + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.searchsorted(edges, w, side="right") - 1, 0, n - 1)
    idx = np.where(np.isfinite(w), idx, -1)
    return idx.astype(int), edges


def pulse_half_peak_width(mean_response, t, max_window_s=1.0):
    """Baseline-subtracted mean pulse response: peak = largest |change| within
    [0, max_window_s]; half-peak width = span where |resp| >= 0.5*|peak|
    around the peak (Khilkevich Methods p.18)."""
    r = np.asarray(mean_response, float)
    t = np.asarray(t, float)
    win = (t >= 0) & (t <= max_window_s)
    if not win.any():
        return np.nan, np.nan
    idxs = np.where(win)[0]
    pk_local = idxs[np.argmax(np.abs(r[idxs]))]
    peak_t, peak_v = float(t[pk_local]), r[pk_local]
    if peak_v == 0:
        return np.nan, peak_t
    half = 0.5 * abs(peak_v)
    lo = pk_local
    while lo > 0 and abs(r[lo - 1]) >= half:
        lo -= 1
    hi = pk_local
    while hi < len(r) - 1 and abs(r[hi + 1]) >= half:
        hi += 1
    return float(t[hi] - t[lo]), peak_t
