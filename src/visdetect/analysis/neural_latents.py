"""N1 neural-latent correspondence: join the B8 per-trial timing latent to
per-trial striatal tensors and test the urgency-ramp hypothesis. Library only
(no plotting / no __main__). See spec 2026-06-29-N1-...-design.md."""
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold

from visdetect.analysis.config import ROOT, canonical_session_id, parse_session_date
from visdetect.analysis.utils import (
    build_population_tensor, compute_zscore_normalized, get_good_cluster_ids,
    compute_lda_cd, bootstrap_ci)
from visdetect.analysis.constants import EVENT_RESPONSIVENESS_WINDOWS

# lab-canonical preparatory-motor windows (rel. corrected lick), shared with lick.py:
_FA_BASE, _FA_PRE = EVENT_RESPONSIVENESS_WINDOWS["FA"]   # ((-1.75,-1.25), (-0.3,-0.15))

DEFAULT_LATENT_CSV = os.path.join(
    ROOT, "data", "cache", "decision_latents", "decision_latents_by_state.csv")

# columns copied verbatim from the latent table into the per-trial y frame
_Y_COLS = ["trial_idx", "outcome", "change_size", "decision_time",
           "change_time_planned", "change_reached", "state_label",
           "timing_urgency_at_decision", "itchiness_caution", "sharpness_drift",
           "evidence_integral_at_decision", "expected_change_time"]

@dataclass
class JoinResult:
    z: np.ndarray            # (n_kept, n_bins, n_units), per-unit shared-baseline z
    bin_centers: np.ndarray  # (n_bins,) seconds rel. Baseline_ON
    y: pd.DataFrame          # one row per kept trial, in z row order
    unit_ids: list           # cluster ids, positional with z's unit axis
    kept_trials: list        # original session trial indices, in z row order

def load_latent_table(path=None):
    df = pd.read_csv(path or DEFAULT_LATENT_CSV, dtype={"session_name": str})
    df["sess_canon"] = df["session_name"].map(canonical_session_id)
    return df

def fitted_expert_sessions(df):
    fitted = df.loc[df["sharpness_drift"].notna(), "sess_canon"].unique()
    return sorted(fitted, key=parse_session_date)

def join_session(session, latent_rows, *, window, bin_size=0.025,
                 baseline_window=(-1.3, -0.3), min_rate_hz=1.0, verify=True):
    good_ids = get_good_cluster_ids(session, min_rate_hz=min_rate_hz)
    tensor, bin_centers, valid_trials = build_population_tensor(
        session, cluster_ids=good_ids, event_name="Baseline_ON",
        window=window, bin_size=bin_size)
    lut = {int(getattr(r, "trial_idx")): r for r in latent_rows.itertuples(index=False)}
    keep = [r for r, ti in enumerate(valid_trials) if int(ti) in lut]
    if not keep:
        raise ValueError(f"join_session: no overlap between tensor trials and "
                         f"latent trial_idx (n_valid={len(valid_trials)}, "
                         f"n_latent={len(lut)})")
    kept_trials = [int(valid_trials[r]) for r in keep]
    z = compute_zscore_normalized(tensor[keep], bin_centers, baseline_window)
    y = pd.DataFrame([{c: getattr(lut[ti], c) for c in _Y_COLS} for ti in kept_trials])
    if verify:
        _verify_join(session, kept_trials, lut)
    return JoinResult(z=z, bin_centers=bin_centers, y=y,
                      unit_ids=list(good_ids), kept_trials=kept_trials)

def _verify_join(session, kept_trials, lut):
    """Triple-check (outcome / change_size / change_time) that trial_idx indexes
    the SAME trial the latent row describes. Fails loud on any mismatch."""
    base = np.asarray(session.ni_events["Baseline_ON"]).ravel()
    assert len(base) >= len(session.trials), (
        f"Baseline_ON ({len(base)}) shorter than trials ({len(session.trials)})")
    for ti in kept_trials:
        tr, lr = session.trials[ti], lut[ti]
        assert (getattr(tr, "trialoutcome", "") or "").lower() == str(lr.outcome).lower(), \
            f"outcome mismatch at trial_idx={ti}"
        assert np.isclose(float(tr.change_size), float(lr.change_size)), \
            f"change_size mismatch at trial_idx={ti}"
        if np.isfinite(float(lr.change_time_planned)):
            assert np.isclose(float(tr.change_time), float(lr.change_time_planned), atol=1e-6), \
                f"change_time mismatch at trial_idx={ti}"

WINDOWS = {"early": (0.5, 2.5), "mid": (2.0, 4.0), "late": (4.0, 6.0)}

def window_feature_matrix(z, bin_centers, win):
    lo, hi = win
    mask = (bin_centers >= lo) & (bin_centers < hi)
    if not mask.any():
        raise ValueError(f"window {win} contains no bin centers "
                         f"(range {bin_centers.min():.2f}..{bin_centers.max():.2f})")
    return z[:, mask, :].mean(axis=1)

def project_out_axis(X, axis):
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    return X - np.outer(X @ a, a)

def motor_axis_signal(X, axis):
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    return X @ a

def fit_lick_motor_cd(z_lick, bin_centers, *, base_window=_FA_BASE, premove_window=_FA_PRE):
    """Unit-norm PREPARATORY-motor axis: LDA between the pre-movement ramp window
    (default (-0.3,-0.15) s before the corrected lick) and a clean pre-trial baseline
    (default (-1.75,-1.25) s) on a LICK-aligned tensor (t=0 = 200 ms-corrected lick).
    Class 1 = pre-movement. Windows are the lab-canonical EVENT_RESPONSIVENESS_WINDOWS
    (ported MATLAB def; matches lick.py) — imported, not invented."""
    def _feat(win):
        m = (bin_centers >= win[0]) & (bin_centers < win[1])
        return z_lick[:, m, :].mean(axis=1)
    pre, premove = _feat(base_window), _feat(premove_window)
    X = np.vstack([pre, premove])
    y = np.r_[np.zeros(len(pre)), np.ones(len(premove))]
    return compute_lda_cd(X, y, method="sklearn", reg=1.0, reg_style="flat")

# ── Task 4: per-session response-time decoder + cohort aggregation + nulls ──
# Decode WITHIN each session (units are NOT cross-session tracked — column u in
# session A != column u in session B; good_and_stable_ids is within-session QC).
# Cohort statistic = mean/median of {r_s} with bootstrap CI OVER SESSIONS
# (session = unit of replication). REJECTED: GroupKFold-across-sessions, a
# concatenated block-diagonal feature space, and any single global Spearman on
# pooled OOF predictions (Simpson's-paradox inflation from between-session offsets).

_MIN_TRIALS_PER_TYPE = 3  # brief's interface text says 5, but its own verbatim
# test_within_type_graded_separates_types supplies 3/type and asserts a populated
# result; the executable test is authoritative. 3 is the min for a defined Spearman.

def within_type_graded(y_pred, y_true, trial_type):
    """Spearman of pred-vs-true WITHIN each trial type for ONE session
    (>= _MIN_TRIALS_PER_TYPE trials/type, non-constant prediction).
    Returns {type -> spearman}."""
    y_pred, y_true, tt = map(np.asarray, (y_pred, y_true, trial_type))
    res = {}
    for t in np.unique(tt):
        m = tt == t
        if m.sum() >= _MIN_TRIALS_PER_TYPE and np.std(y_pred[m]) > 1e-9:
            r = spearmanr(y_pred[m], y_true[m]).correlation
            res[str(t)] = float(r) if np.isfinite(r) else 0.0
    return res

def decode_session(X, y, *, n_splits=5, seed=42):
    """Within ONE session: quantile-binned StratifiedKFold over trials, RidgeCV
    per fold, out-of-fold Spearman r (0.0 if the prediction is constant/degenerate)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n = len(y); k = max(2, min(n_splits, n // 2))
    nb = max(2, min(k, n // 10))
    ybin = pd.qcut(y, nb, labels=False, duplicates="drop")
    if len(np.unique(ybin)) < 2:                 # y too degenerate to stratify
        ybin = (y > np.median(y)).astype(int)
    y_pred = np.full(n, np.nan)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, ybin):
        y_pred[te] = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X[tr], y[tr]).predict(X[te])
    if np.std(y_pred) < 1e-9:
        r = 0.0
    else:
        r = spearmanr(y_pred, y).correlation
        r = 0.0 if not np.isfinite(r) else float(r)
    ss_res = np.sum((y - y_pred) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    return {"r": r, "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0, "y_pred": y_pred}

def _per_session_rs(sessions, seed):
    return np.array([decode_session(X, y, seed=seed)["r"] for _, X, y, _ in sessions])

# ── parallel within-session-shuffle null (determinism-safe) ─────────────────
# The null is embarrassingly parallel (the dominant runtime on the real cohort),
# so it runs across a spawn-based ProcessPoolExecutor with `n_workers` workers.
# DETERMINISM CONTRACT: one RNG seed is pre-generated PARENT-SIDE per shuffle
# (`shuffle_seeds`), so shuffle i is fully determined by `shuffle_seeds[i]` and is
# INDEPENDENT of how the work is sharded. ProcessPoolExecutor.map preserves input
# order, so null[i] is byte-identical across any worker count (and vs the serial
# loop, which uses the SAME pre-generated seeds). Workers never draw from a shared
# /global RNG; BLAS is pinned to 1 thread per worker for both no-oversubscription
# and bit-stability. `sessions` + `decode seed` are shipped to workers ONCE via the
# initializer (module-level globals, spawn-safe), so only the integer seed is
# pickled per task. The worker callable + initializer are module-level (picklable).
_NULL_SESSIONS = None   # per-worker stash: the (sid, X, y, tt) cohort
_NULL_SEED = None       # per-worker stash: the decode_session seed

def _pin_blas_single_thread():
    """Pin BLAS/OpenMP to 1 thread in this worker (no oversubscription + bit
    stability). Prefer threadpoolctl; fall back to env vars if unavailable."""
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[_v] = "1"

def _null_init(sessions, seed):
    """ProcessPoolExecutor initializer: stash the cohort + decode seed in module
    globals (shipped ONCE, not re-pickled per task) and pin BLAS."""
    global _NULL_SESSIONS, _NULL_SEED
    _NULL_SESSIONS, _NULL_SEED = sessions, seed
    _pin_blas_single_thread()

def _null_one_shuffle(shuffle_seed):
    """ONE null replicate: build a fresh RNG from `shuffle_seed`, permute each
    session's y in the fixed `sessions` order, recompute _per_session_rs, return
    the mean-over-sessions aggregate. Reads the cohort from the worker globals."""
    rng = np.random.default_rng(int(shuffle_seed))
    shuff = [(sid, X, rng.permutation(y), tt) for sid, X, y, tt in _NULL_SESSIONS]
    return float(np.nanmean(_per_session_rs(shuff, _NULL_SEED)))

def _cohort_null(sessions, *, n_null, seed, n_workers):
    """Raw within-session-shuffle null array (length n_null). Byte-identical for
    any `n_workers` because each shuffle's RNG seed is pre-generated parent-side.
    `n_workers <= 1` runs the serial loop with the SAME seeds (so serial == parallel)."""
    shuffle_seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=n_null)
    if n_workers <= 1:
        # in-process: stash globals so _null_one_shuffle reads the SAME cohort
        _null_init(sessions, seed)
        return np.array([_null_one_shuffle(s) for s in shuffle_seeds])
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(n_workers), mp_context=ctx,
                             initializer=_null_init,
                             initargs=(sessions, seed)) as ex:
        # .map preserves input order -> null[i] determined solely by shuffle_seeds[i]
        return np.array(list(ex.map(_null_one_shuffle, shuffle_seeds)))

def decode_cohort(sessions, *, n_null=200, seed=42, n_workers=1):
    """`sessions` = list of (sess_id, X, y, trial_type). Decode within each
    session, then aggregate {r_s} OVER SESSIONS (mean/median + bootstrap CI).
    Null = within-session shuffle of y, aggregated identically (mean over sessions).
    The null is computed by `_cohort_null` and is BYTE-IDENTICAL across `n_workers`
    (default 1 = serial); `n_workers>1` parallelizes it over a spawn ProcessPool."""
    per = []
    for sid, X, y, tt in sessions:
        d = decode_session(X, y, seed=seed)
        per.append({"sess_id": sid, "r": d["r"], "n": int(len(y)),
                    "within": within_type_graded(d["y_pred"], y, tt)})
    rs = np.array([p["r"] for p in per])
    ci_lo, ci_hi = bootstrap_ci(rs, n_bootstrap=1000, seed=seed)
    wt = {}
    for t in ("hit", "fa"):
        vals = [p["within"][t] for p in per if t in p["within"]]
        if vals:
            wt[t] = float(np.mean(vals))
    null = _cohort_null(sessions, n_null=n_null, seed=seed, n_workers=n_workers)
    return {"per_session": per, "mean_r": float(np.nanmean(rs)),
            "median_r": float(np.nanmedian(rs)), "ci": (float(ci_lo), float(ci_hi)),
            "null_mean": float(null.mean()), "null_sd": float(null.std()), "within_type": wt}

# ── Task 5: φ-vs-ramp discriminability on the readout window (NOTE A) ────────
# NOTE A: over a PRE-μ window (μ≈6.7-7.5 s, readout ends before μ) a Gaussian
# urgency bump φ (rising flank only) and a linear ramp are highly collinear, so
# a φ-weighted vs ramp-weighted temporal collapse of the SAME readout window may
# be statistically indistinguishable (underpowered). phi_specificity_session
# returns the per-session ΔCV r so Task 7 can aggregate `delta` across sessions
# with a bootstrap CI; an honest "not separable on the readout window" outcome
# is acceptable and gates the urgency-vs-ramp interpretation in Task 7.

def phi_ramp_bases(t, mu, sigma=0.8):
    """Two temporal-weight bases over the readout window `t` (seconds):
    `phi` = Gaussian urgency bump centred at the change time `mu`; `ramp` =
    linear monotonic 0->1 ramp. For a pre-μ window only φ's RISING FLANK is
    seen, so the two are strongly collinear (NOTE A)."""
    t = np.asarray(t, float)
    phi = np.exp(-0.5 * ((t - mu) / sigma) ** 2)
    ramp = (t - t.min()) / (t.max() - t.min() + 1e-12)
    return {"phi": phi, "ramp": ramp}

def phi_specificity_session(Xt, y, t, mu, sigma=0.8, *, seed=42):
    """Xt: (n_trials, n_bins, n_units) for ONE session. Compare within-session
    decode (decode_session) using a phi-weighted vs ramp-weighted temporal collapse
    of the readout window. Returns the per-session delta CV r."""
    b = phi_ramp_bases(t, mu, sigma)
    def _r(weight):
        w = weight / (weight.sum() + 1e-12)
        Xw = np.tensordot(Xt, w, axes=([1], [0]))    # (n_trials, n_units)
        return decode_session(Xw, y, seed=seed)["r"]
    r_phi, r_ramp = _r(b["phi"]), _r(b["ramp"])
    return {"r_phi": r_phi, "r_ramp": r_ramp, "delta": r_phi - r_ramp}

# ── Task 6b: within-FA timing — leakage filter FIRST, then movement-MATCHING ──
# The decisive prong-2 test. The raw within-FA decode (r ~ 0.34-0.56) is PARTLY
# CIRCULAR: 15-28% of FA trials lick INSIDE the readout window, so the decoder
# reads decision_time off peri/post-lick activity. ORDER IS MANDATORY: leakage
# filter FIRST (drop trials whose lick falls in / just after the readout window),
# THEN movement-MATCHING (hold the motor-axis signal ~constant). We NEVER partial
# out / match on decision_time itself (it is the target); we control for the
# per-trial motor-axis projection magnitude only. Matching uses the continuous
# `partial_spearman` AND the stratified `within_strata_spearman`; the high-dim
# subspace projection is a SECONDARY check (it can remove genuine signal).


def leakage_free_mask(decision_time, window, guard=0.25):
    """Boolean mask of FA trials whose lick falls comfortably AFTER the readout
    window, so the decoder cannot read `decision_time` off peri/post-lick activity.

    True where ``decision_time >= window[1] + guard`` (the lick happens at least
    `guard` seconds after the readout window closes). `window` is the (lo, hi)
    readout window in seconds rel. Baseline_ON; `guard` (default 0.25 s) is a
    buffer past `hi` to exclude licks that begin just outside the window but whose
    motor ramp already contaminates it."""
    dt = np.asarray(decision_time, float)
    return dt >= (float(window[1]) + float(guard))


def _rankdata(x):
    """Average-rank transform (ties shared), the rank basis for Spearman."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # average ties so the rank transform matches scipy's Spearman convention
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0          # mean rank within each tie group
    return avg[inv]


def _residualize_on_rank(target_rank, control_rank):
    """Residuals of an OLS fit of `target_rank` on `control_rank` (with intercept).
    The continuous analog of holding `control` fixed (the matching operation)."""
    A = np.column_stack([np.ones_like(control_rank), control_rank])
    beta, *_ = np.linalg.lstsq(A, target_rank, rcond=None)
    return target_rank - A @ beta


def partial_spearman(a, b, control):
    """Rank-partial correlation: Spearman of `a` and `b` after rank-regressing
    BOTH on `control`. Operationalizes movement-matching as a continuous control —
    it removes the component of the a<->b relationship that is mediated by
    `control` (here the per-trial motor-axis signal). Returns 0.0 if either
    residual is constant (undefined correlation)."""
    ar, br, cr = map(_rankdata, (a, b, control))
    ra = _residualize_on_rank(ar, cr)
    rb = _residualize_on_rank(br, cr)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return 0.0
    r = np.corrcoef(ra, rb)[0, 1]            # Pearson on rank residuals = partial Spearman
    return float(r) if np.isfinite(r) else 0.0


def within_strata_spearman(y_pred, y, control, *, n_strata=4):
    """Movement-MATCHED association: quantile-bin trials into `n_strata` bins of
    `control` (the motor-axis signal), compute Spearman(`y_pred`, `y`) WITHIN each
    stratum (movement held ~constant there), and return the trial-count-weighted
    mean across strata. A genuine timing signal survives; a relationship that is
    only a control-driven artifact collapses (no within-stratum structure)."""
    y_pred, y, control = map(lambda v: np.asarray(v, float), (y_pred, y, control))
    n = len(y)
    if n < n_strata * 3:                     # too few trials to stratify meaningfully
        n_strata = max(1, n // 3)
    if n_strata <= 1:
        r = spearmanr(y_pred, y).correlation
        return float(r) if np.isfinite(r) else 0.0
    bins = pd.qcut(control, n_strata, labels=False, duplicates="drop")
    rs, ws = [], []
    for b in np.unique(bins[~pd.isna(bins)]):
        m = bins == b
        if m.sum() >= 3 and np.std(y_pred[m]) > 1e-12 and np.std(y[m]) > 1e-12:
            r = spearmanr(y_pred[m], y[m]).correlation
            if np.isfinite(r):
                rs.append(r)
                ws.append(int(m.sum()))
    if not rs:
        return 0.0
    return float(np.average(rs, weights=ws))


def motor_subspace(z_lick, bin_centers, *, k=5, premove_window=_FA_PRE,
                   base_window=_FA_BASE):
    """Orthonormal top-`k` motor directions: PCA of the per-trial peri-vs-pre-lick
    difference vectors on a LICK-aligned tensor (t=0 = 200 ms-corrected lick).

    For each trial we take (mean z over the pre-movement window) minus (mean z over
    the pre-trial baseline window) — a per-trial peri-lick difference vector in unit
    space — and return the top-`k` right singular vectors (unit-norm, mutually
    orthonormal) as a ``(n_units, k)`` basis spanning the dominant motor subspace.
    `project_out_subspace` removes all k dims. SECONDARY control for Task 6b: a
    high-dim projection can remove genuine timing signal, so it is reported as a
    secondary, non-decisive leg."""
    def _feat(win):
        m = (bin_centers >= win[0]) & (bin_centers < win[1])
        if not m.any():
            raise ValueError(f"motor_subspace: window {win} has no bins")
        return z_lick[:, m, :].mean(axis=1)
    diff = _feat(premove_window) - _feat(base_window)   # (n_trials, n_units)
    diff = diff - diff.mean(axis=0, keepdims=True)      # center before PCA
    n_units = diff.shape[1]
    k_eff = int(min(k, n_units, max(1, diff.shape[0] - 1)))
    # right singular vectors of the centered difference matrix = unit-space PCs
    _, _, vt = np.linalg.svd(diff, full_matrices=False)
    return vt[:k_eff].T                                  # (n_units, k_eff), orthonormal


def project_out_subspace(X, basis):
    """Remove ALL columns of an orthonormal `basis` ((n_units, k)) from each row of
    `X` ((n_trials, n_units)): ``X - (X @ basis) @ basis.T``. With an orthonormal
    basis the result has zero component along every basis direction."""
    B = np.asarray(basis, float)
    X = np.asarray(X, float)
    return X - (X @ B) @ B.T
