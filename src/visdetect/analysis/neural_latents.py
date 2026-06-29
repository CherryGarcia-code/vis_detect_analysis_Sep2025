"""N1 neural-latent correspondence: join the B8 per-trial timing latent to
per-trial striatal tensors and test the urgency-ramp hypothesis. Library only
(no plotting / no __main__). See spec 2026-06-29-N1-...-design.md."""
import os
from dataclasses import dataclass
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

def decode_cohort(sessions, *, n_null=200, seed=42):
    """`sessions` = list of (sess_id, X, y, trial_type). Decode within each
    session, then aggregate {r_s} OVER SESSIONS (mean/median + bootstrap CI).
    Null = within-session shuffle of y, aggregated identically (mean over sessions)."""
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
    rng = np.random.default_rng(seed)
    null = np.empty(n_null)
    for i in range(n_null):                       # within-session shuffle of y, aggregate as mean r_s
        shuff = [(sid, X, rng.permutation(y), tt) for sid, X, y, tt in sessions]
        null[i] = float(np.nanmean(_per_session_rs(shuff, seed)))
    return {"per_session": per, "mean_r": float(np.nanmean(rs)),
            "median_r": float(np.nanmedian(rs)), "ci": (float(ci_lo), float(ci_hi)),
            "null_mean": float(null.mean()), "null_sd": float(null.std()), "within_type": wt}
