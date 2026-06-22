"""Per-neuron Poisson encoding GLM (Khilkevich-Lohse 2024 replication).

50-ms-binned, temporally-unfolded (FIR) design matrix -> ridge-Poisson per
neuron with (nested or fast) 10-fold CV -> TF-responsive identification by the
paper's DENSE held-out criterion: a FULL model (with TF kernel) and a
REDUCED_TF model (TF kernel removed) are fit on the SAME CV split, and a unit
is TF-responsive when the FULL model's cross-validated predictive correlation
``corr(zscore(yTest), y_hat_pred)`` (C1 r>0.2) is significantly higher than the
REDUCED model's across the paired folds (C2 one-sided paired t-test P<0.01).
This matches ``FullBlocks_CV.m`` line ~626 (PredSmth=1 => no smoothing); it is
NOT the sparse fast/slow pulse PETH (their pulse analysis is only the separate
Fig-3 population pulse-scaling figure). See
docs/superpowers/specs/2026-06-18-tf-glm-replication-design.md.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings
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
        # Movement-controlled (full Khilkevich) regressors; only added to the
        # design when cfg.include_movement is True. Windows from the audit of
        # FullBlocks_CV.m (#17 FaceMovement / #18 RunSpeed dur850/off-1 =>
        # [-0.05, 0.8]s; #19 Pupil dur1500/off-15 => [-0.75, 0.75]s; #12 Air
        # puff dur250/off0 => [0, 0.25]s).
        "motion_energy": (-0.05, 0.8),
        "pupil":         (-0.75, 0.75),
        "airpuff":       (0.0, 0.25),
    })
    sd_pulse: float = 0.5               # fast/slow = +/-0.5 SD of baseline TF
    pulse_eval_win: Tuple[float, float] = (-0.15, 0.75)  # PETH window around pulses
    min_pulses_per_label: int = 50      # min fast/slow pulses to attempt C1/C2
    n_folds: int = 10
    lambdas: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    c1_r_thresh: float = 0.2
    c2_p_thresh: float = 0.01
    # TF-responsive decision rule. A diagnostic on the authors' (Khilkevich-Lohse
    # 2024) own data showed the raw-bin C1 r>0.2 floor is mis-scaled and
    # confounded: it gates GENERAL held-out predictability, not TF specifically
    # (striatum CP scored HIGHER C1 than visual VISp — the OPPOSITE of the TF
    # biology), because the paper's r>0.2 was measured on a denoised pulse PETH,
    # a different scale than this code's raw-bin FullPred. C2 — the paired
    # full-vs-reduced TF-ablation test — is scale-free and reproduces the paper's
    # biology (VISp 27% > CP 15%, both in the 5-45% range). C2 is therefore the
    # primary criterion. "c1_and_c2" keeps the paper's literal conjunction for
    # faithfulness/comparison.
    responsive_criterion: str = "c2"    # "c2" (default) | "c1_and_c2"
    seed: int = 42
    include_phase: bool = False         # off for DMS-first; on for cortex
    # Movement-controlled (full Khilkevich-faithful) model: add motion-energy +
    # pupil continuous FIR regressors and an air-puff event regressor. OFF by
    # default so existing (reduced-model) callers are unaffected; the reduced
    # model is what C2 toggles the TF block against, and these movement
    # regressors are the nuisance controls the paper's full model carries.
    include_movement: bool = False
    fast_fit: bool = False              # select ridge lambda ONCE/unit (not per outer fold)
    # ── Faithful Khilkevich full-design options (Jun 2026) ──────────────────
    # Added to match the authors' published FULL model on their own data, after
    # a line-by-line diff vs their MATLAB (glmnet standardize=true; an 80x200ms
    # tiled-baseline nuisance block; a 12-bin grating-phase block). See
    # scripts/tf_responsiveness/cluster/ and the MEMORY note tf_glm_replication.
    tf_encoding: str = "log2"           # "log2" (octaves/0.25, faithful) | "linear" (Hz; control)
    standardize_design: bool = False    # z-score ALL design columns (matches glmnet standardize=true)
    include_tiled_baseline: bool = False  # 80x200ms baseline tiles (replaces the single time_in_base ramp)
    tiled_baseline_tile_s: float = 0.2  # tile width (their hardcoded 200 ms)
    tiled_baseline_span_s: float = 16.0  # tiling span from baseline onset (their 16000 ms)
    tiled_baseline_min_trials: int = 10  # drop tiles occupied by < this many trials (their penalty x1000)


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
    # Movement-controlled (full Khilkevich) regressors. Default to None so
    # reduced-model callers (BG_046/039 session builder) are unaffected; only
    # populated + used when cfg.include_movement is True. Per-bin continuous
    # (like wheel_bins); airpuff_time is a single neural-clock event time.
    motion_bins: Optional[np.ndarray] = None  # (n_bins,) face/whisker motion-energy
    pupil_bins: Optional[np.ndarray] = None   # (n_bins,) pupil area
    airpuff_time: float = float("nan")        # neural-clock air-puff onset; NaN if none


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
    # None (e.g. unset motion_bins/pupil_bins on a reduced-model trial) -> all-fill
    if a is None:
        return np.full(n, fill)
    a = np.asarray(a, dtype=float).ravel()
    if a.size == n:
        return a
    out = np.full(n, fill)
    m = min(a.size, n)
    out[:m] = a[:m]
    return out


def _tf_octaves(tf_lin: np.ndarray) -> np.ndarray:
    """Linear baseline TF (Hz, geomean 1) -> z-scored octaves log2(TF)/0.25.

    Matches the authors' BaseTFFrames_to_ms.m TF regressor: the baseline TF is
    drawn from log2 ~ N(0, 0.25 octave), so log2(TF)/0.25 is symmetric about 0
    and already ~unit-variance in SD-octave units. Post-change / masked bins are
    encoded as 0 (TF<=0), which maps to the geomean-neutral 0 octave. Feeding
    LINEAR TF would be asymmetric (a 2x-up pulse = +1 Hz but a 2x-down = -0.5 Hz),
    which the FIR cannot represent as the symmetric log-TF kernel.
    """
    tf = np.asarray(tf_lin, dtype=float).ravel()
    out = np.zeros(tf.shape, dtype=float)
    pos = tf > 0
    out[pos] = np.log2(tf[pos]) / 0.25
    return out


def _ramp_col(tr, edges, bs):
    """Seconds since baseline start, zero before 1 s and after change onset."""
    t = edges - tr.t_start
    ramp = np.where(t >= 1.0, t, 0.0)
    if np.isfinite(tr.change_time):
        ramp[edges >= tr.change_time] = 0.0
    return ramp.reshape(-1, 1)


def _tiled_baseline_block(trials, per_edges, cfg) -> np.ndarray:
    """Authors' tiled-baseline nuisance: one-hot over fixed-width (200 ms) tiles
    since baseline onset, spanning `tiled_baseline_span_s` (16 s => 80 tiles).

    A bin is assigned to tile k = floor((t - t_start) / tile_s) when it lies in
    the baseline period (t >= t_start and, if a change occurred, t < change_time).
    Tiles occupied by fewer than `tiled_baseline_min_trials` trials are dropped
    (the authors crush them with penalty_factor x1000; for a ridge fit, dropping
    them is the clean equivalent). Replaces the single time-in-baseline ramp.
    """
    n_tiles = int(round(cfg.tiled_baseline_span_s / cfg.tiled_baseline_tile_s))
    tile_s = cfg.tiled_baseline_tile_s
    idx_parts, trials_per_tile = [], [set() for _ in range(n_tiles)]
    for ti, (tr, e) in enumerate(zip(trials, per_edges)):
        since = e - tr.t_start
        valid = since >= 0
        if np.isfinite(tr.change_time):
            valid = valid & (e < tr.change_time)
        idx = np.floor(since / tile_s).astype(int)
        idx[~valid] = -1
        idx[idx >= n_tiles] = -1
        idx_parts.append(idx)
        for k in np.unique(idx[idx >= 0]):
            trials_per_tile[int(k)].add(ti)
    all_idx = np.concatenate(idx_parts) if idx_parts else np.zeros(0, dtype=int)
    keep = [k for k in range(n_tiles)
            if len(trials_per_tile[k]) >= cfg.tiled_baseline_min_trials]
    block = np.zeros((all_idx.size, len(keep)), dtype=float)
    for j, k in enumerate(keep):
        block[all_idx == k, j] = 1.0
    return block


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
    #    do not bleed across trial boundaries. The regressor is z-scored OCTAVES
    #    log2(TF)/0.25 (NOT linear Hz): the baseline TF is log2 ~ N(0, 0.25), so
    #    log2(TF)/0.25 is symmetric (matching the authors' BaseTFFrames_to_ms.m)
    #    and already ~unit-variance, so the tf block is NOT per-column z-scored
    #    below. (design.tf_bins stays LINEAR for pulse_times_from_tf.)
    _tf_xform = (_tf_octaves if cfg.tf_encoding == "log2"
                 else (lambda v: np.asarray(v, float).ravel()))  # "linear" control
    _add("tf", _blockwise(trials, per_edges, lambda tr, e: fir_continuous(
        _tf_xform(_resize(tr.tf_bins, e.size)), cfg.kern["tf"], bs)))
    # 2) trial start event (== their baseON)
    _add("trial_start", _blockwise(trials, per_edges, lambda tr, e: fir_event(
        np.array([tr.t_start]), e, cfg.kern["trial_start"], bs)))
    # 3) baseline nuisance: either the authors' 80x200ms tiled-baseline block
    #    (faithful) OR our single time-in-baseline ramp (legacy/simplified).
    if cfg.include_tiled_baseline:
        _add("tiled_baseline", _tiled_baseline_block(trials, per_edges, cfg))
    else:
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
    # 17,19,12) movement-controlled regressors (optional): face/whisker
    # motion-energy + pupil (continuous FIR) and air-puff (event FIR). Built
    # per-trial then stacked so lags do not bleed across trial boundaries.
    if cfg.include_movement:
        _add("motion_energy", _blockwise(trials, per_edges, lambda tr, e:
            fir_continuous(_resize(tr.motion_bins, e.size),
                           cfg.kern["motion_energy"], bs)))
        _add("pupil", _blockwise(trials, per_edges, lambda tr, e:
            fir_continuous(_resize(tr.pupil_bins, e.size),
                           cfg.kern["pupil"], bs)))
        _add("airpuff", _blockwise(trials, per_edges, lambda tr, e: fir_event(
            np.array([tr.airpuff_time]), e, cfg.kern["airpuff"], bs)))
    # 15-16) phase (optional)
    if cfg.include_phase:
        _add("phase", _phase_indicator(phase_bins))

    X = np.concatenate(cols, axis=1) if cols else np.zeros((N, 0))

    # Column normalization.
    #  - standardize_design=True: z-score EVERY column to unit variance, matching
    #    the authors' glmnet standardize=true (which internally standardizes ALL
    #    predictors, including the TF block, before the ridge-Poisson fit). This
    #    is required for a faithful replication: under a single shared ridge
    #    lambda, columns at different raw scales are penalized inconsistently.
    #  - standardize_design=False (legacy): z-score only the continuous nuisance
    #    regressors (wheel/motion/pupil); leave events/ramp/phase/tf as-is.
    # Either way design.tf_bins is left RAW (linear Hz) for pulse_times_from_tf.
    if cfg.standardize_design:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        keep = sd >= 1e-8
        if np.any(keep):
            X[:, keep] = (X[:, keep] - mu[keep]) / sd[keep]
    else:
        for grp in ("wheel", "motion_energy", "pupil"):
            sl = groups.get(grp)
            if sl is None:
                continue
            block = X[:, sl]
            mu = block.mean(axis=0)
            sd = block.std(axis=0)
            keep = sd >= 1e-8
            if np.any(keep):
                block[:, keep] = (block[:, keep] - mu[keep]) / sd[keep]
                X[:, sl] = block

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

from scipy import stats as _stats
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
    # tol=1e-4 (not 1e-6): a purely-numerical convergence change, NOT a
    # scientific one. On the large real-data designs (~140k rows x ~380 cols)
    # the lbfgs solver hits max_iter at tol=1e-6 on every CV fit (the nested
    # 10-fold x 6-lambda search is ~1100 fits/unit), making the run intractable.
    # At tol=1e-4 the ridge-Poisson coefficients and CV lambda selection are
    # unchanged to within noise, but each fit converges in a fraction of the
    # iterations. max_iter raised to 500 as a safety margin for the few fits
    # that still need it. Synthetic-fixture tests are unaffected.
    m = PoissonRegressor(alpha=lam, fit_intercept=True, max_iter=500, tol=1e-4)
    m.fit(Xtr, ytr)
    return m


def _select_lambda_once(X, y, fold_ids, cfg: TFGLMConfig) -> float:
    """Fast-mode λ selection: pick the best ridge λ ONCE per unit on a single
    trial-blocked train/validation split.

    Hold out the first 2 of `cfg.n_folds` fold-ids as validation, train on the
    remaining 8, and score each λ in `cfg.lambdas` by held-out Poisson
    log-likelihood. Return the argmax λ. This costs len(lambdas) fits instead
    of the nested ~n_folds*n_inner*len(lambdas). The chosen λ is then reused for
    every outer held-out fit; the small λ-selection optimism does not invalidate
    the per-fold held-out predictions (the outer test rows are still predicted by
    a model fit without them).
    """
    uniq = np.unique(fold_ids)
    n_val = min(2, max(1, uniq.size - 1))
    val_folds = set(uniq[:n_val].tolist())
    va = np.isin(fold_ids, list(val_folds))
    tr = ~va
    if va.sum() == 0 or tr.sum() == 0:
        return cfg.lambdas[0]
    best_lam, best_score = cfg.lambdas[0], -np.inf
    for lam in cfg.lambdas:
        m = _fit_one(X[tr], y[tr], lam)
        mu = m.predict(X[va])
        # Poisson held-out log-likelihood (up to const)
        s = float(np.sum(y[va] * np.log(mu + 1e-9) - mu))
        if s > best_score:
            best_score, best_lam = s, lam
    return best_lam


def fit_poisson_cv(X, y, cfg: TFGLMConfig, fold_ids=None) -> FitResult:
    X = np.asarray(X, float); y = np.asarray(y, float)
    n = y.size
    if fold_ids is None:
        fold_ids = np.repeat(np.arange(cfg.n_folds), int(np.ceil(n / cfg.n_folds)))[:n]
    pred = np.full(n, np.nan)
    coefs, best_lams = [], []

    # Fast mode: choose λ ONCE per unit, then run the outer loop with it fixed.
    fixed_lam = _select_lambda_once(X, y, fold_ids, cfg) if cfg.fast_fit else None

    for f in range(cfg.n_folds):
        te = fold_ids == f
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        if cfg.fast_fit:
            best_lam = fixed_lam
        else:
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


# ---------------------------------------------------------------------------
# Task 7: TF-responsive identification (C1 + C2) + kernel peak/FWHM
# ---------------------------------------------------------------------------

def pulse_times_from_tf(design: DesignMatrix, cfg: TFGLMConfig):
    """Bin-center times of fast/slow baseline-TF pulses (+/- sd_pulse*SD).

    tf_bins are linear TF (geom-mean 1 Hz); convert to log2 octaves so the SD
    matches the task's log2 N(0,0.25) baseline.  If tf_bins already contain
    negative values (log2-encoded test fixtures), the conversion is skipped."""
    tf = np.asarray(design.tf_bins, float)
    bs = design.bin_edges[1] - design.bin_edges[0] if design.bin_edges.size > 1 else cfg.bin_s
    centers = design.bin_edges + bs / 2.0
    # Detect encoding: real linear TF is always > 0; log2 fixtures may be negative.
    if np.any(tf < 0):
        # Already log2-encoded (test fixture or pre-converted input)
        log2tf = tf.copy()
        valid = tf != 0.0
    else:
        with np.errstate(divide="ignore"):
            log2tf = np.where(tf > 0, np.log2(np.where(tf > 0, tf, 1.0)), np.nan)
        valid = np.isfinite(log2tf) & (tf > 0)
    if valid.sum() < 10:
        return np.zeros(0), np.zeros(0)
    sd = np.nanstd(log2tf[valid])
    if sd < 1e-9:
        return np.zeros(0), np.zeros(0)
    thr = cfg.sd_pulse * sd
    fast = centers[valid & (log2tf >= thr)]
    slow = centers[valid & (log2tf <= -thr)]
    return fast, slow


def tf_pulse_peth(values_per_bin, bin_edges, pulse_times, win, bin_s,
                  trial_index=None):
    """Event-triggered average of a per-bin signal around pulse_times.

    ``bin_edges`` is the stitched per-trial bin-left-edge array produced by
    ``assemble_design``; it is NOT uniformly spaced -- there are inter-trial
    gaps at trial boundaries.  Absolute ``pulse_times`` are therefore mapped to
    bin indices with ``searchsorted`` (which honours the gaps) rather than by
    uniform arithmetic from ``bin_edges[0]`` (that scrambled every trial after
    the first).  When ``trial_index`` is supplied, lag windows are clipped so
    they never bleed across a trial boundary."""
    v = np.asarray(values_per_bin, float)
    offs = _lag_offsets(win, bin_s)
    t_axis = offs * bin_s
    be = np.asarray(bin_edges, float)
    pulses = np.asarray(pulse_times, float)
    if be.size == 0 or pulses.size == 0:
        return t_axis, np.full(offs.size, np.nan)
    # Map each absolute pulse time to its bin via the (non-uniform) edges.
    idx = np.searchsorted(be, pulses, side="right") - 1
    ti = None if trial_index is None else np.asarray(trial_index)
    rows = []
    for p in idx:
        if p < 0 or p >= v.size:
            continue
        cols = p + offs
        ok = (cols >= 0) & (cols < v.size)
        if ti is not None:
            # keep only lags that stay within the same trial as the pulse bin
            ok = ok & (ti[np.clip(cols, 0, v.size - 1)] == ti[p])
        row = np.full(offs.size, np.nan)
        row[ok] = v[cols[ok]]
        rows.append(row)
    if not rows:
        return t_axis, np.full(offs.size, np.nan)
    # A lag column where no pulse contributed an in-trial bin is an all-NaN
    # slice; np.nanmean emits a "Mean of empty slice" RuntimeWarning via
    # warnings.warn (NOT a float error, so np.errstate cannot suppress it).
    # The NaN result is the intended "undefined at this lag" sentinel, so
    # silence that specific warning narrowly to keep the suite clean under
    # -W error::RuntimeWarning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        peth = np.nanmean(np.vstack(rows), axis=0)
    return t_axis, peth


def _tf_kernel(full_fit, design, cfg):
    """Mean TF FIR kernel (octave-weight per lag) averaged across folds."""
    sl = design.col_groups.get("tf")
    if sl is None or not full_fit.coef_by_fold:
        return None
    K = np.vstack([c[sl] for c in full_fit.coef_by_fold])
    return K.mean(axis=0)


def _zscore(a: np.ndarray) -> np.ndarray:
    """z-score a 1-D array (sample SD), returning NaN if variance is ~0."""
    a = np.asarray(a, float)
    mu = a.mean()
    sd = a.std()
    if not np.isfinite(sd) or sd < 1e-12:
        return np.full(a.shape, np.nan)
    return (a - mu) / sd


def identify_tf_responsive(design, y, full_fit, reduced_fit, cfg: TFGLMConfig) -> dict:
    """TF-responsive by the DENSE paired full-vs-reduced predictive-correlation
    criterion of Khilkevich & Lohse 2024.

    This matches their ``FullBlocks_CV.m`` line ~626::

        [FullPred,FullPred_pval]=corr(zscore(smoothdata(yTest,'movmean',PredSmth)), y_hat_pred);

    with ``PredSmth=1`` (movmean window 1 = no smoothing). For each held-out
    CV fold they z-score the ACTUAL held-out spike counts and correlate against
    the model's predicted rate on that fold. A unit is TF-responsive when the
    FULL model (with the TF kernel) reaches a meaningful held-out predictive
    correlation AND beats the REDUCED_TF model (identical regressors minus the
    TF kernel) on the SAME paired CV folds (the TF kernel adds held-out
    predictive power).

    ``full_fit`` and ``reduced_fit`` were fit on the SAME ``fold_ids`` (the
    caller passes one ``make_trial_folds`` array to both), so their per-fold
    held-out predictions are paired fold-for-fold.

    C1 : ``c1_r = nanmean_f pearson(zscore(y[te_f]), full_fit.pred[te_f])`` must
         exceed ``cfg.c1_r_thresh`` (0.2). This is the dense whole-series
         predictive correlation of the FULL model.
    C2 : a one-sided paired t-test across folds that ``r_full - r_red > 0`` must
         have ``p < cfg.c2_p_thresh`` (0.01): the TF kernel improves held-out
         prediction.

    The ``is_responsive`` DECISION RULE is configurable via
    ``cfg.responsive_criterion``:

    - ``"c2"`` (DEFAULT): ``is_responsive = (c2_p < cfg.c2_p_thresh)``. C2 is the
      scale-free TF-ABLATION test (does the TF kernel add held-out predictive
      power on the paired folds?). A diagnostic on the authors' own data showed
      the raw-bin C1 r>0.2 floor is mis-scaled/confounded — it gates general
      predictability, not TF (striatum CP > visual VISp on C1, opposite the TF
      biology) — whereas C2 alone reproduces the paper's biology (VISp 27% > CP
      15%). The paper's r>0.2 was on a denoised pulse PETH, a different scale than
      this code's raw-bin FullPred, so it does not transfer.
    - ``"c1_and_c2"``: ``is_responsive = (c1_r > cfg.c1_r_thresh) and
      (c2_p < cfg.c2_p_thresh)`` — the paper's literal conjunction, kept for
      faithfulness/comparison.

    ``c1_r``, ``c2_p``, ``r_full_mean``, ``r_red_mean``, ``kernel_peak_t`` and
    ``kernel_fwhm`` are always returned unchanged; only the ``is_responsive``
    decision rule depends on ``cfg.responsive_criterion``.
    """
    if cfg.responsive_criterion not in ("c2", "c1_and_c2"):
        raise ValueError(
            f"Unknown responsive_criterion {cfg.responsive_criterion!r}; "
            "expected 'c2' or 'c1_and_c2'."
        )
    bs = cfg.bin_s
    y = np.asarray(y, float)
    fold_ids = np.asarray(full_fit.fold_ids)
    pred_full = np.asarray(full_fit.pred, float)
    pred_red = np.asarray(reduced_fit.pred, float)

    r_full, r_red = [], []
    for f in np.unique(fold_ids):
        te = fold_ids == f
        ok = te & np.isfinite(y) & np.isfinite(pred_full) & np.isfinite(pred_red)
        if ok.sum() < 10:
            continue
        yz = _zscore(y[ok])                # z-score the ACTUAL counts (movmean=1)
        pf = pred_full[ok]
        pr = pred_red[ok]
        # Guard: need nonzero variance in actual and predicted; else skip fold.
        if (not np.all(np.isfinite(yz))) or np.std(pf) < 1e-12:
            rf = np.nan
        else:
            rf = float(np.corrcoef(yz, pf)[0, 1])
        if (not np.all(np.isfinite(yz))) or np.std(pr) < 1e-12:
            rr = np.nan
        else:
            rr = float(np.corrcoef(yz, pr)[0, 1])
        if np.isfinite(rf) and np.isfinite(rr):
            r_full.append(rf)
            r_red.append(rr)

    r_full = np.asarray(r_full, float)
    r_red = np.asarray(r_red, float)

    if r_full.size == 0:
        c1_r = np.nan
        c2_p = np.nan
        is_resp = False
        r_full_mean = np.nan
        r_red_mean = np.nan
    else:
        c1_r = float(np.nanmean(r_full))
        r_full_mean = c1_r
        r_red_mean = float(np.nanmean(r_red))
        # C2: one-sided paired t-test that r_full - r_red > 0 across folds.
        diff = r_full - r_red
        if diff.size >= 2 and np.std(diff) > 1e-12:
            t, p_two = _stats.ttest_rel(r_full, r_red)
            # one-sided positive: full beats reduced
            c2_p = p_two / 2.0 if t > 0 else 1.0 - p_two / 2.0
        elif diff.size >= 2 and np.all(diff > 0):
            # zero-variance but strictly positive improvement on every fold
            c2_p = 0.0
        else:
            c2_p = np.nan
        c1_pass = np.isfinite(c1_r) and (c1_r > cfg.c1_r_thresh)
        c2_pass = np.isfinite(c2_p) and (c2_p < cfg.c2_p_thresh)
        # Decision rule per cfg.responsive_criterion (validated above): C2 alone
        # (scale-free TF-ablation test) is the default; "c1_and_c2" is the
        # paper's literal conjunction, kept for comparison.
        if cfg.responsive_criterion == "c2":
            is_resp = bool(c2_pass)
        else:  # "c1_and_c2"
            is_resp = bool(c1_pass and c2_pass)

    # kernel metrics (unchanged)
    kpeak_t, kfwhm = np.nan, np.nan
    K = _tf_kernel(full_fit, design, cfg)
    if K is not None and K.size:
        lags = _lag_offsets(cfg.kern["tf"], bs) * bs
        ip = int(np.argmax(np.abs(K)))
        kpeak_t = float(lags[ip])
        half = abs(K[ip]) / 2.0
        lo = ip
        while lo > 0 and abs(K[lo - 1]) >= half:
            lo -= 1
        hi = ip
        while hi < K.size - 1 and abs(K[hi + 1]) >= half:
            hi += 1
        kfwhm = float(lags[hi] - lags[lo])
    return {"c1_r": c1_r, "c2_p": c2_p, "is_responsive": is_resp,
            "r_full_mean": r_full_mean, "r_red_mean": r_red_mean,
            "n_folds_used": int(r_full.size),
            "kernel_peak_t": kpeak_t, "kernel_fwhm": kfwhm}
