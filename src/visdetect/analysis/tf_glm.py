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


def _lag_offsets(win: Tuple[float, float], bin_s: float) -> np.ndarray:
    """Integer bin offsets for a kernel window [lo, hi) in bin_s steps."""
    lo, hi = win
    n = int(round((hi - lo) / bin_s))
    start = int(round(lo / bin_s))  # Fix 3: compute once, reuse
    return np.arange(start, start + max(n, 0))


def fir_event(event_times, bin_edges, win, bin_s) -> np.ndarray:
    """(n_bins, n_lags) FIR design for point events.

    Column j (lag = offsets[j]*bin_s): a 1 in bin b means an event occurred
    `lag` seconds before the start of bin b (i.e. event fell in bin b-offset).
    """
    n_bins = bin_edges.size
    offs = _lag_offsets(win, bin_s)
    X = np.zeros((n_bins, offs.size), dtype=float)
    ev = np.asarray(event_times, dtype=float).ravel()
    ev = ev[np.isfinite(ev)]
    if n_bins == 0 or ev.size == 0 or offs.size == 0:
        return X
    # bin index containing each event (keep all finite events; inner b-clip bounds writes)
    idx = np.floor((ev - bin_edges[0]) / bin_s + 1e-9).astype(int)
    # Fix 2: do NOT pre-filter idx to in-window here — events outside the window
    # may still contribute at shifted lags; the inner b-clip handles all bounds.
    for j, off in enumerate(offs):
        b = idx + off
        b = b[(b >= 0) & (b < n_bins)]
        X[b, j] = 1.0
    return X


def fir_continuous(signal, win, bin_s) -> np.ndarray:
    """(n_bins, n_lags) lagged copies of a per-bin continuous signal.

    Column j is `signal` shifted so that row b holds signal[b - offset]
    (causal positive lags look back in time), zero-filled at the edges.
    """
    sig = np.asarray(signal, dtype=float).ravel()
    n_bins = sig.size
    offs = _lag_offsets(win, bin_s)
    X = np.zeros((n_bins, offs.size), dtype=float)
    for j, off in enumerate(offs):
        if off == 0:
            X[:, j] = sig
        elif off > 0:
            X[off:, j] = sig[: n_bins - off]
        else:
            # Fix 1: guard against off < -n_bins where n_bins+off <= 0 causes
            # a shape mismatch (LHS selects 0 or 1 rows while RHS is empty).
            if n_bins + off <= 0:
                continue
            X[:n_bins + off, j] = sig[-off:]
    return X


# ---------------------------------------------------------------------------
# Task 5: Trial regressor container + full FIR design-matrix assembly
# ---------------------------------------------------------------------------

@dataclass
class TrialRegressors:
    t_start: float
    t_end: float
    change_time: float          # neural-clock change onset; NaN if change not reached
    change_size: float          # 1.0 (catch), 1.25, 1.35, 1.5, 2, 4
    tf_bins: np.ndarray         # (n_bins,) baseline TF per bin (0 outside baseline)
    lick_times: np.ndarray      # neural-clock lick-bout onset times
    reward_time: float          # neural-clock; NaN if none
    abort_time: float           # neural-clock; NaN if none
    wheel_bins: np.ndarray      # (n_bins,) wheel speed per bin
    phase_bins: Optional[np.ndarray] = None  # (n_bins,) phase degrees [0,360) or None


@dataclass
class DesignMatrix:
    X: np.ndarray
    col_groups: Dict[str, slice]
    bin_edges: np.ndarray
    trial_index: np.ndarray
    tf_bins: np.ndarray


CHANGE_SIZES = (1.0, 1.25, 1.35, 1.5, 2.0, 4.0)


def _phase_indicator(phase_deg: np.ndarray, n_bins_circ: int = 12) -> np.ndarray:
    """(n_rows, n_bins_circ) one-hot of phase into n_bins_circ angular bins."""
    out = np.zeros((phase_deg.size, n_bins_circ), dtype=float)
    valid = np.isfinite(phase_deg)
    b = np.floor((phase_deg[valid] % 360) / (360.0 / n_bins_circ)).astype(int)
    out[np.where(valid)[0], np.clip(b, 0, n_bins_circ - 1)] = 1.0
    return out


def _resize(a, n, fill=0.0):
    a = np.asarray(a, dtype=float).ravel()
    if a.size == n:
        return a
    out = np.full(n, fill)
    m = min(a.size, n)
    out[:m] = a[:m]
    return out


def _ramp_col(tr, edges, bs):
    """Seconds since baseline start, zero before 1 s and after change onset."""
    t = edges - tr.t_start
    ramp = np.where(t >= 1.0, t, 0.0)
    if np.isfinite(tr.change_time):
        ramp[edges >= tr.change_time] = 0.0
    return ramp.reshape(-1, 1)


def _blockwise(trials, per_edges, fn):
    blocks = [fn(tr, e) for tr, e in zip(trials, per_edges)]
    ncol = max((b.shape[1] for b in blocks), default=0)
    blocks = [b if b.shape[1] == ncol else np.zeros((b.shape[0], ncol)) for b in blocks]
    return np.concatenate(blocks, axis=0) if blocks else np.zeros((0, ncol))


def assemble_design(trials: List["TrialRegressors"], cfg: TFGLMConfig) -> DesignMatrix:
    bs = cfg.bin_s
    # Per-trial bin edges and concatenation bookkeeping
    per_edges, per_n, tf_all, wheel_all, phase_all = [], [], [], [], []
    for ti, tr in enumerate(trials):
        edges = trial_bin_edges(tr.t_start, tr.t_end, bs)
        per_edges.append(edges); per_n.append(edges.size)
        tf_all.append(_resize(tr.tf_bins, edges.size))
        wheel_all.append(_resize(tr.wheel_bins, edges.size))
        if cfg.include_phase and tr.phase_bins is not None:
            phase_all.append(_resize(tr.phase_bins, edges.size, fill=np.nan))
        else:
            phase_all.append(np.full(edges.size, np.nan))
    bin_edges = np.concatenate(per_edges) if per_edges else np.zeros(0)
    trial_index = np.concatenate([np.full(n, i) for i, n in enumerate(per_n)]) \
        if per_n else np.zeros(0, dtype=int)
    tf_bins = np.concatenate(tf_all) if tf_all else np.zeros(0)
    wheel_bins = np.concatenate(wheel_all) if wheel_all else np.zeros(0)
    phase_bins = np.concatenate(phase_all) if phase_all else np.zeros(0)
    N = bin_edges.size

    cols: List[np.ndarray] = []
    groups: Dict[str, slice] = {}

    def _add(name, block):
        start = sum(c.shape[1] for c in cols)
        cols.append(block)
        groups[name] = slice(start, start + block.shape[1])

    # 1) TF (continuous, per-bin, lagged) — built per-trial then stacked so lags
    #    do not bleed across trial boundaries.
    _add("tf", _blockwise(trials, per_edges, lambda tr, e: fir_continuous(
        _resize(tr.tf_bins, e.size), cfg.kern["tf"], bs)))
    # 2) trial start event
    _add("trial_start", _blockwise(trials, per_edges, lambda tr, e: fir_event(
        np.array([tr.t_start]), e, cfg.kern["trial_start"], bs)))
    # 3) time-in-baseline ramp (single graded column: seconds since t_start, 0
    #    after change; >=1 s region per the paper)
    _add("time_in_base", _blockwise(trials, per_edges, lambda tr, e:
        _ramp_col(tr, e, bs)))
    # 4-9) six change onsets by change size
    for cs in CHANGE_SIZES:
        _add(f"change_{cs}", _blockwise(trials, per_edges, lambda tr, e, cs=cs:
            fir_event(np.array([tr.change_time]) if (np.isfinite(tr.change_time)
                      and tr.change_size == cs) else np.zeros(0),
                      e, cfg.kern["change"], bs)))
    # 10) lick prep, 11) lick exec
    _add("lick_prep", _blockwise(trials, per_edges, lambda tr, e: fir_event(
        tr.lick_times, e, cfg.kern["lick_prep"], bs)))
    _add("lick_exec", _blockwise(trials, per_edges, lambda tr, e: fir_event(
        tr.lick_times, e, cfg.kern["lick_exec"], bs)))
    # 13) reward, 14) abort
    _add("reward", _blockwise(trials, per_edges, lambda tr, e: fir_event(
        np.array([tr.reward_time]), e, cfg.kern["reward"], bs)))
    _add("abort", _blockwise(trials, per_edges, lambda tr, e: fir_event(
        np.array([tr.abort_time]), e, cfg.kern["abort"], bs)))
    # 18) wheel (continuous)
    _add("wheel", _blockwise(trials, per_edges, lambda tr, e: fir_continuous(
        _resize(tr.wheel_bins, e.size), cfg.kern["wheel"], bs)))
    # 15-16) phase (optional)
    if cfg.include_phase:
        _add("phase", _phase_indicator(phase_bins))

    X = np.concatenate(cols, axis=1) if cols else np.zeros((N, 0))
    return DesignMatrix(X=X, col_groups=groups, bin_edges=bin_edges,
                        trial_index=trial_index, tf_bins=tf_bins)


def count_vector(trials, spike_times, design: DesignMatrix) -> np.ndarray:
    y = np.zeros(design.bin_edges.size, dtype=float)
    bs = design.bin_edges[1] - design.bin_edges[0] if design.bin_edges.size > 1 else 0.05
    for i in range(len(trials)):
        mask = design.trial_index == i
        edges = design.bin_edges[mask]
        y[mask] = bin_spike_counts(spike_times, edges)
    return y


# ---------------------------------------------------------------------------
# Task 6: Ridge-Poisson fit with trial-blocked nested 10-fold CV
# ---------------------------------------------------------------------------

from sklearn.linear_model import PoissonRegressor


@dataclass
class FitResult:
    pred: np.ndarray
    fold_ids: np.ndarray
    coef_by_fold: List[np.ndarray]
    best_lambdas: List[float]


def make_trial_folds(trial_index: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    trials = np.unique(trial_index)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(trials.size)
    fold_of_trial = {int(trials[perm[k]]): k % n_folds for k in range(trials.size)}
    return np.array([fold_of_trial[int(t)] for t in trial_index])


def _fit_one(Xtr, ytr, lam):
    m = PoissonRegressor(alpha=lam, fit_intercept=True, max_iter=300, tol=1e-6)
    m.fit(Xtr, ytr)
    return m


def fit_poisson_cv(X, y, cfg: TFGLMConfig, fold_ids=None) -> FitResult:
    X = np.asarray(X, float); y = np.asarray(y, float)
    n = y.size
    if fold_ids is None:
        fold_ids = np.repeat(np.arange(cfg.n_folds), int(np.ceil(n / cfg.n_folds)))[:n]
    pred = np.full(n, np.nan)
    coefs, best_lams = [], []
    for f in range(cfg.n_folds):
        te = fold_ids == f
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        # inner CV over lambda on the training rows (split by inner folds)
        inner = fold_ids[tr]
        best_lam, best_score = cfg.lambdas[0], -np.inf
        for lam in cfg.lambdas:
            scores = []
            for g in np.unique(inner):
                itr = inner != g; ite = inner == g
                if ite.sum() == 0 or itr.sum() == 0:
                    continue
                m = _fit_one(X[tr][itr], y[tr][itr], lam)
                mu = m.predict(X[tr][ite])
                # Poisson held-out log-likelihood (up to const)
                scores.append(np.sum(y[tr][ite] * np.log(mu + 1e-9) - mu))
            s = np.mean(scores) if scores else -np.inf
            if s > best_score:
                best_score, best_lam = s, lam
        m = _fit_one(X[tr], y[tr], best_lam)
        pred[te] = m.predict(X[te])
        coefs.append(m.coef_.copy()); best_lams.append(best_lam)
    return FitResult(pred=pred, fold_ids=fold_ids, coef_by_fold=coefs, best_lambdas=best_lams)
