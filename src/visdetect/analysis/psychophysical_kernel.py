"""B10 — psychophysical / neural impulsivity kernel (Orsolic-style reverse
correlation of baseline TF fluctuations preceding impulsive licks).

Plain English: work out what pattern of grating-speed wobble the mouse mistakes
for a real change (i.e. what triggers an impulsive early lick), and how that
pattern — and its neural echo in striatal cells — changes as the mouse learns.
All estimators are pure and deterministic; scripts in scripts/evidence_learning/
apply them to real sessions.

Design notes
------------
* Stimulus reconstruction reuses the verified stride-3 / 60 Hz recipe
  (``tf_glm_data._BASELINE_STRIDE``): ``baseline_values`` (St1TrialVector) is
  logged 3x per 50 ms TF update, so ``bv[::3]`` recovers the 50 ms grid.
* No baked-in lick delay: align to the RECORDED FA lick; ``lick_shift_ms`` is a
  sensitivity knob (default 0). A constant shift cancels in every learning/state
  contrast, so no scientific claim depends on it.
* Everything is dt = 0.05 s (the 50 ms TF update). Never 0.25.
"""
from __future__ import annotations
import numpy as np

from visdetect.analysis.constants import FA_RT_SPLIT

DT = 0.05
KERNEL_PRE_S = 1.5           # window starts this far before the lick
KERNEL_REFRACTORY_S = 0.15   # exclude the last 150 ms (sensorimotor)
# Late-FA gate: impulsivity analyses use only self-timed LATE FAs (latency >=
# FA_RT_SPLIT = 3.0 s). Early FAs (<3 s) are reflexive / carry-over, and their
# pre-lick baseline window would run before Baseline_ON. (Matches lick.py / N1.)
MIN_FA_LATENCY_S = FA_RT_SPLIT
CHANGE_GUARD_S = 0.5         # drop FAs within this of change_time
BOOT_SEED = 42
N_BOOT = 1000
_MONITOR_HZ = 60.0
_STRIDE = 3
_WITHHOLD_TOL_S = 0.25


# ── stimulus reconstruction ──────────────────────────────────────────────
def baseline_log2tf(trial, dt=DT, tf_base=None):
    """Full-trial baseline log2-TF on the dt grid anchored at Baseline_ON.

    y[k] = log2(bv[_STRIDE*k] / base); base = tf_base or per-trial nanmedian(bv).
    Returns (t, y). No timestamp arithmetic -> immune to non-finite-time crashes.
    """
    bv = np.asarray(getattr(trial, "baseline_values", []), float).ravel()
    if bv.size == 0:
        return np.zeros(0), np.zeros(0)
    vals = bv[::_STRIDE]
    base = float(tf_base) if tf_base is not None else (float(np.nanmedian(bv)) or 1.0)
    if not np.isfinite(base) or base == 0:
        base = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(vals > 0, np.log2(vals / base), 0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    t = np.arange(vals.size) * dt
    return t, y


# ── FA (impulsive-lick) epochs ───────────────────────────────────────────
def _fa_latency(trial):
    rts = getattr(trial, "reactiontimes", {}) or {}
    v = rts.get("FA", rts.get("fa"))
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def fa_kernel_epochs(session, lick_shift_ms=0.0, dt=DT, min_fa_latency=None):
    """Per usable FA trial: {trial_idx, lick_t, window}. window = log2-TF over
    [lick_t - KERNEL_PRE_S, lick_t - KERNEL_REFRACTORY_S] (len L).

    Guards: outcome=='fa', finite FA latency, lick_t >= min_fa_latency (default
    MIN_FA_LATENCY_S = FA_RT_SPLIT = 3.0 s -> LATE FAs only, the self-timed
    impulsive licks; pass a smaller value to include early FAs), enough pre-lick
    history (j0 >= 0), and |lick_t - change_time| >= CHANGE_GUARD_S when
    change_time is finite (drop FAs that hug a real change)."""
    min_fa = MIN_FA_LATENCY_S if min_fa_latency is None else min_fa_latency
    L = round((KERNEL_PRE_S - KERNEL_REFRACTORY_S) / dt)
    shift = lick_shift_ms / 1000.0
    out = []
    for idx, tr in enumerate(getattr(session, "trials", []) or []):
        if (getattr(tr, "trialoutcome", "") or "").lower() != "fa":
            continue
        lick_t = _fa_latency(tr)
        if not np.isfinite(lick_t) or lick_t < min_fa:
            continue
        ct = float(getattr(tr, "change_time", np.nan) or np.nan)
        if np.isfinite(ct) and abs(lick_t - ct) < CHANGE_GUARD_S:
            continue
        _, y = baseline_log2tf(tr, dt=dt)
        if y.size == 0:
            continue
        j1 = int(round((lick_t - shift - KERNEL_REFRACTORY_S) / dt))
        j0 = j1 - L
        if j0 < 0 or j1 > y.size:
            continue
        out.append({"trial_idx": idx, "lick_t": lick_t, "window": y[j0:j1].copy()})
    return out


# ── matched-withhold control ─────────────────────────────────────────────
def _withhold_trials(session):
    """hit/miss trials with a finite change_time (their pre-change baseline is a
    genuine no-lick epoch of known duration)."""
    out = []
    for tr in getattr(session, "trials", []) or []:
        oc = (getattr(tr, "trialoutcome", "") or "").lower()
        ct = float(getattr(tr, "change_time", np.nan) or np.nan)
        if oc in ("hit", "miss") and np.isfinite(ct):
            out.append((tr, ct))
    return out


def withhold_epochs(session, fa_epochs, dt=DT, rng=None):
    """One time-in-trial-matched no-lick window per FA epoch (None if unmatched).

    For an FA at lick_t, use withhold trials whose pre-change baseline extends
    past lick_t (change_time - REFRACTORY >= lick_t) and slice the SAME
    [lick_t - PRE, lick_t - REFRACTORY] window."""
    rng = rng if rng is not None else np.random.default_rng(BOOT_SEED)
    L = round((KERNEL_PRE_S - KERNEL_REFRACTORY_S) / dt)
    wtrials = _withhold_trials(session)
    ys = {id(tr): baseline_log2tf(tr, dt=dt)[1] for tr, _ in wtrials}
    out = []
    for ep in fa_epochs:
        lick_t = ep["lick_t"]
        picks = []
        for tr, ct in wtrials:
            if ct - KERNEL_REFRACTORY_S < lick_t:
                continue
            y = ys[id(tr)]
            j1 = int(round((lick_t - KERNEL_REFRACTORY_S) / dt))
            j0 = j1 - L
            if j0 >= 0 and j1 <= y.size:
                picks.append(y[j0:j1])
        out.append(picks[int(rng.integers(len(picks)))].copy() if picks else None)
    return out


# ── reverse-correlation kernel ───────────────────────────────────────────
def reverse_correlation_kernel(fa_windows, withhold_windows):
    """FA-triggered mean minus withhold-matched mean, per lag."""
    fa = np.asarray(fa_windows, float)
    wh = np.asarray(withhold_windows, float)
    if fa.ndim != 2 or fa.size == 0:
        raise ValueError("fa_windows must be a non-empty list of equal-length arrays")
    return fa.mean(axis=0) - wh.mean(axis=0)


def kernel_lags(dt=DT):
    """Lag axis (s, negative): bin left-edges over [-KERNEL_PRE_S, -REFRACTORY)."""
    L = round((KERNEL_PRE_S - KERNEL_REFRACTORY_S) / dt)
    return -KERNEL_PRE_S + np.arange(L) * dt


def bootstrap_kernel_ci(fa_windows, withhold_windows, n_boot=N_BOOT, seed=BOOT_SEED):
    """Point kernel + 95% percentile bands, resampling PAIRS with replacement."""
    fa = np.asarray(fa_windows, float)
    wh = np.asarray(withhold_windows, float)
    n, L = fa.shape
    kernel = fa.mean(0) - wh.mean(0)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, L))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = fa[idx].mean(0) - wh[idx].mean(0)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return kernel, lo, hi


def kernel_shape_metrics(kernel, dt=DT):
    """Peak amplitude, its lag, and the contiguous half-max width around the peak.
    Amplitude and shape are reported SEPARATELY (spec: the learning claim is a
    shape result; amplitude is confounded by FA-count/base-rate)."""
    k = np.asarray(kernel, float)
    lags = kernel_lags(dt)
    pk_i = int(np.argmax(k))
    peak = float(k[pk_i])
    half = peak / 2.0
    lo = pk_i
    while lo - 1 >= 0 and k[lo - 1] >= half:
        lo -= 1
    hi = pk_i
    while hi + 1 < k.size and k[hi + 1] >= half:
        hi += 1
    return {"peak_amp": peak, "peak_lag_s": float(lags[pk_i]),
            "half_width_s": float((hi - lo + 1) * dt)}


# ── neural: signed population TF signal ──────────────────────────────────
def _trial_windows(session):
    """[(trial_idx, t0_abs, dur_s)] per trial: baseline start -> change or +6 s."""
    bon = np.asarray(session.ni_events.get("Baseline_ON", []), float).ravel()
    out = []
    for idx, tr in enumerate(getattr(session, "trials", []) or []):
        if idx >= bon.size or not np.isfinite(bon[idx]):
            continue
        ct = float(getattr(tr, "change_time", np.nan) or np.nan)
        dur = ct if (np.isfinite(ct) and ct > 0) else 6.0
        out.append((idx, float(bon[idx]), float(dur)))
    return out


def signed_population_signal(session, unit_signs, dt=DT):
    """Per-trial signed z-scored population TF signal aligned to Baseline_ON.

    S(t) = mean_i sign_i * z_i(t); z_i = per-unit z-score of the dt-binned rate to
    that unit's mean/SD over ALL baseline-period bins (shared-baseline
    equalization, so high-FR units don't dominate). Returns {trial_idx: (t, S)}.
    """
    clusters = {c.cluster_id: np.asarray(c.spike_times, float)
                for c in getattr(session, "clusters", [])}
    windows = _trial_windows(session)
    per_unit = {}                       # cid -> {trial_idx: rate_array}
    for cid in unit_signs:
        st = clusters.get(cid)
        if st is None:
            continue
        per_trial = {}
        for idx, t0, dur in windows:
            nb = int(round(dur / dt))
            if nb < 1:
                continue
            edges = t0 + np.arange(nb + 1) * dt
            per_trial[idx] = np.histogram(st, bins=edges)[0] / dt
        per_unit[cid] = per_trial
    z_unit = {}
    for cid, per_trial in per_unit.items():
        allr = np.concatenate(list(per_trial.values())) if per_trial else np.zeros(1)
        mu, sd = float(allr.mean()), float(allr.std())
        sd = sd if sd > 1e-9 else 1.0
        z_unit[cid] = {i: (r - mu) / sd for i, r in per_trial.items()}
    out = {}
    for idx, t0, dur in windows:
        nb = int(round(dur / dt))
        if nb < 1:
            continue
        acc = np.zeros(nb)
        ncontrib = 0
        for cid, sign in unit_signs.items():
            zt = z_unit.get(cid, {}).get(idx)
            if zt is not None and zt.size == nb:
                acc += sign * zt
                ncontrib += 1
        S = acc / ncontrib if ncontrib else acc
        out[idx] = (np.arange(nb) * dt, S)
    return out


def stimulus_matched_control(fa_windows, withhold_windows, fa_pop, withhold_pop):
    """Decompose the neural FA-vs-withhold signal into sensory + excess-gain.

    withhold_pop shares its FA's stimulus trajectory (stimulus-matched), so its
    mean is the sensory expectation; the FA-minus-withhold residual is gain."""
    if len(fa_windows) != len(withhold_windows):
        raise ValueError("stimulus windows must be paired 1:1")
    fa_p = np.asarray(fa_pop, float)
    wh_p = np.asarray(withhold_pop, float)
    sensory = wh_p.mean(0)
    total = fa_p.mean(0)
    return {"sensory": sensory, "gain": total - sensory, "total": total}


# ── real-time stimulus tracking (does the population ride the stimulus?) ──
def stimulus_tracking_xcorr(S, y, max_lag_bins):
    """Pearson r between neural signal S(t) and stimulus y(t-lag) for
    lag = 0..max_lag_bins (the neural signal LAGS the stimulus by its response
    latency). r[lag] = corr(S[lag:], y[:-lag]); NaN where a segment is constant
    or too short. Peak lag ~ the neural integration latency (~200-250 ms)."""
    S = np.asarray(S, float).ravel()
    y = np.asarray(y, float).ravel()
    n = min(S.size, y.size)
    out = np.full(max_lag_bins + 1, np.nan)
    for lag in range(max_lag_bins + 1):
        if n - lag < 3:
            break
        a = S[lag:n]
        b = y[:n - lag]
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        out[lag] = np.corrcoef(a, b)[0, 1]
    return out
