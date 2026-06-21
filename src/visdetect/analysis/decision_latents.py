"""B8 — behavioral decision-latents decomposed by state (Phase 1: descriptive).

Plain English: for every trial we want three numbers — 'sharpness' (how clearly
the mouse can tell the grating changed), 'itchiness' (how trigger-happy it is
before any real change), and 'timing' (how strongly it expects the change now).
Phase 1 measures these directly from behaviour, split by the mouse's mood
(Impulsive vs StimSens), across learning. No model fitting here.
"""
from __future__ import annotations
import glob
import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm

from visdetect.analysis import ddm
from visdetect.analysis.behavior import compute_session_performance, calculate_dprime
from visdetect.analysis.config import parse_session_date  # DDMMYYYY parser
# CHANGE_SIZES is the canonical ordered go-trial list (sorted ALL_GO_CHANGE_SIZES);
# it lives in config, NOT constants (the task brief's import path is a typo).
from visdetect.analysis.config import CHANGE_SIZES

MAIN_MOODS = ("Impulsive", "StimSens")
SEPARATE_MOODS = ("Disengaged",)
EXCLUDED_MOODS = ("Abort",)
_DEFAULT_TAG_DIR = os.path.join("data", "cache", "state_tags")

# Provisional response-window end (s) used as the Miss decision time. Mirrors
# ddm.RESPONSE_WINDOW_S; there is no canonical response-window constant in
# visdetect.analysis.constants — confirm against task params during the
# real-data run (spec §10). Defined HERE (not imported from ddm) so Phase-2 code
# never touches the buggy ddm evidence sampler.
RESPONSE_WINDOW_S = 2.155


def _decision_time_dl(trial):
    """Return (decision_time_s, lick {0,1}, censored), aligned to Baseline_ON.

    Phase-1 local mirror of ``ddm._decision_time`` (deliberately NOT imported
    from ddm, to keep Phase-2 off the buggy ``ddm.build_trial_evidence`` sampler):

    * ``hit``  → ``(change_time + RT, 1, False)``   (RT from reactiontimes RT/Hit/hit)
    * ``fa``   → ``(FA_latency, 1, False)``          (from reactiontimes FA/fa/RT)
    * ``miss`` → ``(change_time + RESPONSE_WINDOW_S, 0, True)`` (response-window end, censored)
    * anything else (abort/ref) → ``(nan, 0, True)`` (handled by the caller)

    ``trialoutcome`` is lowercased (real data may capitalize it).
    """
    oc = (getattr(trial, "trialoutcome", "") or "").lower()
    rts = getattr(trial, "reactiontimes", {}) or {}
    ct = float(getattr(trial, "change_time", np.nan) or np.nan)
    if oc == "hit":
        rt = rts.get("RT", rts.get("Hit", rts.get("hit")))
        return (ct + float(rt), 1, False)
    if oc == "fa":
        rt = rts.get("FA", rts.get("fa", rts.get("RT")))
        return (float(rt), 1, False)            # anticipatory lick, aligned to Baseline_ON
    if oc == "miss":
        return (ct + RESPONSE_WINDOW_S, 0, True)   # response-window end; no crossing (censored)
    return (np.nan, 0, True)                     # abort/ref handled by caller


def build_trial_evidence_corrected(session, dt=0.05, tf_base=None):
    """Per-trial log2-TF evidence on the dt grid, truncated to [0, decision_time].
    Returns DataFrame: trial_idx, outcome, change_size, change_time, decision_time,
    lick, censored, evidence(np.ndarray, len==n_bins), n_bins.
    Evidence bin k reads baseline frame index 3*k (60 Hz storage, 50 ms holds)."""
    MONITOR_HZ = 60.0
    frames_per_bin = int(round(dt * MONITOR_HZ))   # == 3 for dt=0.05
    trials = getattr(session, "trials", []) or []
    rows = []
    for uid, t in enumerate(trials):
        oc = (getattr(t, "trialoutcome", "") or "").lower()
        if oc in ("abort", "ref"):
            continue
        bv = getattr(t, "baseline_values", None)
        if bv is None:
            continue
        bv = np.asarray(bv, float).ravel()
        if bv.size == 0:
            continue
        cs = float(getattr(t, "change_size", np.nan) or np.nan)
        ct = float(getattr(t, "change_time", np.nan) or np.nan)
        base = float(tf_base) if tf_base is not None else float(np.nanmedian(bv)) or 1.0
        dec_t, lick, censored = _decision_time_dl(t)        # Phase-1 helper (Task 0.1 Step 1)
        if not np.isfinite(dec_t) or dec_t <= 0:
            continue
        n_bins = int(round(dec_t / dt))
        if n_bins < 1:
            continue
        ev = np.empty(n_bins, float)
        for k in range(n_bins):
            j = min(bv.size - 1, k * frames_per_bin)        # 60 Hz frame for this 50 ms bin
            tau = k * dt
            tf = bv[j] * cs if (np.isfinite(ct) and tau >= ct and cs > 1.0) else bv[j]
            ev[k] = np.log2(tf / base) if tf > 0 else 0.0
        ev = np.nan_to_num(ev, nan=0.0)
        rows.append({"trial_idx": uid, "outcome": oc, "change_size": cs,
                     "change_time": ct, "decision_time": dec_t, "lick": int(lick),
                     "censored": bool(censored), "evidence": ev, "n_bins": n_bins})
    import pandas as pd
    return pd.DataFrame(rows)

def load_state_labels(session_name, subject="BG_046", tag_dir=None):
    base = os.path.join(tag_dir or _DEFAULT_TAG_DIR, subject)
    candidates = [str(session_name)]
    try:
        candidates.append(str(int(session_name)).zfill(8))  # leading-zero form
    except (TypeError, ValueError):
        pass
    for cand in candidates:
        path = os.path.join(base, f"{cand}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df[df["trial_idx"].notna()].copy()
            df["trial_idx"] = df["trial_idx"].astype(int)
            return df.set_index("trial_idx")[["state_label", "state_confidence"]]
    raise FileNotFoundError(f"No state-tag file for {session_name} under {base}")


def enumerate_valid_sessions(subject="BG_046", tag_dir=None, min_total_trials=50):
    """Tier-1 integrity floor: sessions with a digit-named tag CSV passing a
    minimum total-trial count, sorted chronologically (DDMMYYYY ids)."""
    base = os.path.join(tag_dir or _DEFAULT_TAG_DIR, subject)
    sessions = []
    for path in glob.glob(os.path.join(base, "*.csv")):
        sname = os.path.splitext(os.path.basename(path))[0]
        if not sname.isdigit():               # skip _tag_summary.csv etc.
            continue
        n = sum(1 for _ in open(path)) - 1     # rows minus header (Tier-1 floor)
        if n >= min_total_trials:
            sessions.append(sname)
    return sorted(sessions, key=parse_session_date)


def session_dprime(session):
    """Per-session d′ (Tier-2 continuous covariate)."""
    # NOTE: the key is "d_prime" (NOT "dprime") — confirmed in behavior.py; wrong key = silent NaN
    return float(compute_session_performance(session).get("d_prime", float("nan")))


def assign_comprehension_flags(dprime_by_session, threshold=0.5):
    """First chronological session with d′ ≥ threshold marks the pre→post
    boundary; every session from there on is "post". (threshold=0.5 is the
    low "knows-the-rule" bar, distinct from the QC 0.8 gate; spec §7.)"""
    ordered = sorted(dprime_by_session, key=parse_session_date)
    flags, comprehended = {}, False
    for s in ordered:
        if (dprime_by_session[s] or 0) >= threshold:
            comprehended = True
        flags[s] = "post" if comprehended else "pre"
    return flags


def build_trial_table(session, state_labels, session_name, dt=0.05):
    """One row per usable trial: behavioral geometry + the trial's mood.

    Reuses ``ddm.build_trial_evidence`` ONLY for trial geometry (it lowercases
    outcome and excludes outcome ``abort``/``ref``). Trials whose labeler mood
    is in ``EXCLUDED_MOODS`` (the labeler-state ``Abort``, distinct from the
    trial-outcome ``abort``) are dropped. The evidence array itself is discarded
    here (its TF indexing is a Phase-2 concern); only ``n_bins = len(evidence)``
    is kept. ``trial_in_session`` is assigned after sorting so it is monotonic.
    """
    ev = ddm.build_trial_evidence(session, dt=dt)   # trial_uid, outcome, change_size,
                                                    # change_time, decision_time, lick, censored, evidence
    rows = []
    for _, r in ev.iterrows():
        uid = int(r["trial_uid"])
        mood = state_labels["state_label"].get(uid)
        conf = state_labels["state_confidence"].get(uid)
        if mood in EXCLUDED_MOODS:                  # drop labeler 'Abort'
            continue
        outcome = str(r["outcome"]).lower()
        rows.append({
            "session_name": session_name, "trial_idx": uid, "outcome": outcome,
            "change_size": float(r["change_size"]),
            "change_time_planned": float(r["change_time"]),
            "change_reached": outcome in ("hit", "miss"),
            "decision_time": float(r["decision_time"]),
            "lick": int(r["lick"]), "censored": bool(r["censored"]),
            "state_label": mood, "state_confidence": conf,
            "n_bins": int(len(r["evidence"])),
        })
    tab = pd.DataFrame(rows).sort_values("trial_idx").reset_index(drop=True)
    tab["trial_in_session"] = np.arange(len(tab))   # within-session position (satiety covariate)
    return tab


def censored_hazard(durations, events, dt=0.05, t_max=None):
    """Discrete-time hazard + survival with right-censoring.

    Each trial ends at ``durations[i]``; ``events[i] = True`` means the event
    occurred there, ``False`` means the trial was right-censored (e.g. a change
    planned at 15 s but the mouse FA-licked at 3 s → censored at 3 s, never an
    event at 15 s; spec §4 / user 2026-06-18). At bin *k* the hazard is
    ``(#events in bin k) / (#trials still at risk at the start of bin k)`` and
    ``survival = cumprod(1 - hazard)``. A trial censored at the start of bin *k*
    is still at risk during the bin it falls in, then drops out.
    """
    durations = np.asarray(durations, float); events = np.asarray(events, bool)
    if t_max is None:
        t_max = float(np.nanmax(durations)) + dt
    edges = np.arange(0.0, t_max + dt, dt)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Half-open (left, right] bins: bin k = (k·dt, (k+1)·dt]. A trial exiting at
    # duration d (d > 0) falls in bin ceil(d/dt)-1 (e.g. d=0.07, dt=0.05 →
    # ceil(1.4)-1 = bin1 (0.05,0.10]). It is at risk at the START of bin k only
    # while it has not yet exited, i.e. d > edges[k] (a trial exiting exactly at a
    # bin's right edge — e.g. censored at 0.05 — is in bin0 (0,0.05] and gone by
    # the start of bin1).
    n_bins = len(centers)
    event_bin = np.clip(np.ceil(durations / dt).astype(int) - 1, 0, n_bins - 1)  # bin (d-dt, d] in which the trial ends
    hazard = np.zeros(len(centers))
    for k in range(len(centers)):
        at_risk = np.sum(durations > edges[k] + 1e-12)         # still running at bin start
        n_event = np.sum(events & (event_bin == k))
        hazard[k] = (n_event / at_risk) if at_risk > 0 else 0.0
    survival = np.cumprod(1.0 - hazard)
    return centers, hazard, survival


def _logistic(x, a, b):
    return 1.0 / (1.0 + np.exp(-(a + b * x)))


def _logistic_lapse(x, a, b, lapse):
    """Lapse-aware psychometric: a symmetric lapse rate floors AND ceilings the
    logistic. ``P = lapse + (1 - 2*lapse) * logistic(a + b*x)`` so the lower
    asymptote is ``lapse`` and the upper is ``1 - lapse`` (the mouse licks/withholds
    by mistake at rate ``lapse`` regardless of evidence). ``x = log2(change_size)``."""
    return lapse + (1.0 - 2.0 * lapse) / (1.0 + np.exp(-(a + b * x)))


def sharpness_scores(trial_df):
    """Sharpness = how clearly the mouse tells the change happened.

    Operates on one (session × mood) cell's trial rows. Returns a dict:

    * ``psy_slope`` — logistic-fit slope of P(lick) vs ``log2(change_size)`` on
      GO trials (``change_size > 1.0``). NaN if < 8 go trials, < 2 distinct
      change sizes, or the fit fails.
    * ``psy_threshold`` — the change size at 50% detection from the same logistic
      fit. The 50% point in log2(change_size) space is ``x50 = -a/b``, so
      ``psy_threshold = 2 ** x50``. NaN if the fit failed / ``psy_slope`` is NaN /
      ``abs(b) < 1e-3``; otherwise clamped to ``[1.0, 8.0]`` (change sizes span
      1.25–4.0, so this avoids wild extrapolation off a near-flat fit).
    * ``psy_lapse`` / ``psy_threshold_lapse`` — a 3-param LAPSE-AWARE fit of the
      SAME go-trial psychometric: ``P(lick|cs) = lapse + (1-2*lapse)*logistic(a +
      b*log2(cs))`` (``_logistic_lapse``) with ``lapse ∈ [0, 0.3]``. This is the
      psychometric the Phase-2 generative sharpness latent is validated against
      (construct-validity F8), so both must measure the same thing. ``psy_lapse``
      is the recovered lapse rate; ``psy_threshold_lapse = 2 ** (-a/b)`` clamped to
      ``[1.0, 8.0]`` only when ``b > 0`` (the midpoint of the logistic core, where
      P is the average of the two asymptotes). Both are NaN if the lapse fit fails
      to converge or returns ``b <= 0``. Same go-trial support gate as the 2-param
      fit (< 8 go trials or < 2 distinct change sizes → NaN).
    * ``dprime`` — ``calculate_dprime(hit_rate, fa_rate)`` with go-trial lick
      mean as the hit rate and catch-trial (``change_size ≈ 1.0``) lick mean as
      the FA rate. (``calculate_dprime`` log-linear-clips the rates.)
    * ``rt_mean_cs{cs}`` / ``rt_cv_cs{cs}`` per ``cs`` in canonical
      ``CHANGE_SIZES`` — Hit RT = ``decision_time − change_time_planned`` on hit
      trials; NaN if < 3 such trials.
    """
    go = trial_df[trial_df["change_size"] > 1.0]
    catch = trial_df[np.isclose(trial_df["change_size"], 1.0)]
    out = {}
    # psychometric slope: P(lick) vs log2(change_size) on go trials
    if len(go) >= 8 and go["change_size"].nunique() >= 2:
        x = np.log2(go["change_size"].values); y = go["lick"].values.astype(float)
        try:
            # Bound intercept a and slope b each to [-20, 20]: an unbounded fit on
            # near-separable cells returns absurd slopes (up to ~418), corrupting
            # both the cached deliverable and the figures. A slope of 20 in
            # log2(change_size) space is already near-step, so this loses no signal.
            (a, b), _ = curve_fit(_logistic, x, y, p0=[0.0, 1.0],
                                  bounds=([-20.0, -20.0], [20.0, 20.0]), maxfev=5000)
            out["psy_slope"] = float(b)
        except Exception:
            out["psy_slope"] = float("nan")
    else:
        out["psy_slope"] = float("nan")
    # psy_threshold: change size at 50% detection = 2 ** (-a/b). NaN unless the
    # slope is POSITIVE and non-trivial (b >= 1e-3): a detection threshold is
    # only defined for an INCREASING psychometric. Near-flat (b≈0 → x50 explodes)
    # AND negative-slope fits (small-sample noise: "detection falls with bigger
    # Δ" is nonsense and yields an absurd clamped threshold) are both rejected.
    # Otherwise clamp 2**x50 to the plausible change-size range [1.0, 8.0].
    b = out["psy_slope"]
    if np.isfinite(b) and b >= 1e-3:
        x50 = -a / b
        out["psy_threshold"] = float(np.clip(2.0 ** x50, 1.0, 8.0))
    else:
        out["psy_threshold"] = float("nan")
    # 3-param LAPSE-AWARE fit on the SAME go-trial data (added alongside the
    # 2-param keys for Phase-2 F8 construct validity). lapse bounded [0, 0.3];
    # a/b kept wide. On convergence failure or non-increasing core (b<=0) → NaN.
    out["psy_lapse"] = float("nan")
    out["psy_threshold_lapse"] = float("nan")
    if len(go) >= 8 and go["change_size"].nunique() >= 2:
        xL = np.log2(go["change_size"].values); yL = go["lick"].values.astype(float)
        try:
            (aL, bL, lapseL), _ = curve_fit(
                _logistic_lapse, xL, yL, p0=[0.0, 1.0, 0.05],
                bounds=([-20.0, -20.0, 0.0], [20.0, 20.0, 0.3]), maxfev=10000)
            if bL > 0:                       # threshold only defined for an increasing core
                out["psy_lapse"] = float(lapseL)
                x50L = -aL / bL
                out["psy_threshold_lapse"] = float(np.clip(2.0 ** x50L, 1.0, 8.0))
            # b<=0: a lapse rate off a flat/decreasing fit is not interpretable → leave NaN
        except Exception:
            pass                             # keep the NaN defaults set above
    hit_rate = float(go["lick"].mean()) if len(go) else float("nan")
    fa_rate = float(catch["lick"].mean()) if len(catch) else float("nan")
    out["dprime"] = float(calculate_dprime(hit_rate, fa_rate))
    # per-change-size Hit RT mean + CV
    hits = go[go["outcome"] == "hit"].copy()
    hits["rt"] = hits["decision_time"] - hits["change_time_planned"]
    for cs in CHANGE_SIZES:
        rt = hits.loc[np.isclose(hits["change_size"], cs), "rt"].values
        out[f"rt_mean_cs{cs}"] = float(np.mean(rt)) if rt.size >= 3 else float("nan")
        out[f"rt_cv_cs{cs}"] = float(np.std(rt) / np.mean(rt)) if rt.size >= 3 and np.mean(rt) > 0 else float("nan")
    return out


def _loglinear(rate, n):
    """Log-linear correction: maps a rate into the open interval (0, 1) so
    ``norm.ppf`` never returns ±inf for a 0/1 hit- or FA-rate."""
    return (rate * n + 0.5) / (n + 1.0)


def itchiness_scores(trial_df, dt=0.05):
    """Itchiness = how trigger-happy the mouse is before any real evidence.

    Operates on one (session × mood) cell's trial rows. Returns a dict:

    * ``criterion_c`` — SDT criterion ``-(z(H) + z(FA)) / 2`` where ``H`` is the
      go-trial (``change_size > 1.0``) lick rate and ``FA`` the catch-trial
      (``change_size ≈ 1.0``) lick rate, each log-linear corrected so a 0/1 rate
      gives a finite z.
    * ``fa_rate`` — fraction of trials whose ``outcome == "fa"`` (anticipatory
      lick; NOT the SDT false-alarm rate).
    * ``baseline_hazard`` — mean FA (anticipatory-lick) hazard over the
      PRE-CHANGE window only. An anticipatory lick can only occur before the
      change, so non-FA trials are right-censored at ``change_time_planned``
      (mirroring ``fa_lick_hazard``) — NaN change-times fall back to
      ``decision_time``. This makes the baseline comparable across cells with
      different max decision times: previously the hazard was averaged over the
      full decision timeline, so post-change bins diluted it unevenly between
      cells (fix c, Phase 2). Computed from ``censored_hazard`` with FA-latency
      events (``outcome == "fa"``).
    """
    go = trial_df[trial_df["change_size"] > 1.0]
    catch = trial_df[np.isclose(trial_df["change_size"], 1.0)]
    H = _loglinear(go["lick"].mean() if len(go) else 0.0, max(len(go), 1))
    FA = _loglinear(catch["lick"].mean() if len(catch) else 0.0, max(len(catch), 1))
    crit = -(norm.ppf(H) + norm.ppf(FA)) / 2.0
    fa_rate = float((trial_df["outcome"] == "fa").mean())
    # baseline lick hazard over the PRE-CHANGE window: FA-latency events, with
    # non-FA trials censored at the change (min(change_time_planned, decision_time);
    # NaN change_time falls back to decision_time) — the same censoring as
    # fa_lick_hazard, so post-change bins never contaminate the baseline.
    is_fa = (trial_df["outcome"] == "fa").values
    dtime = trial_df["decision_time"].values.astype(float)
    ctime = trial_df["change_time_planned"].values.astype(float)
    change_censor = np.where(np.isnan(ctime), dtime, np.minimum(ctime, dtime))
    censor_t = np.where(is_fa, dtime, change_censor)
    _, hz, _ = censored_hazard(censor_t, is_fa, dt=dt)
    return {"criterion_c": float(crit), "fa_rate": fa_rate,
            "baseline_hazard": float(np.nanmean(hz)) if hz.size else float("nan")}


def change_onset_hazard(trial_df, dt=0.05):
    """Hazard of the change actually occurring over trial time.

    Event = the change being reached (``change_reached == True``) at
    ``change_time_planned``. Trials whose change was planned but never reached
    (``change_reached == False``, e.g. an FA-lick before the change time) are
    RIGHT-CENSORED at ``decision_time`` — they never contribute an event, only
    risk-time up to when the trial ended. Delegates to ``censored_hazard``.
    """
    reached = trial_df["change_reached"].values.astype(bool)
    dur = np.where(reached, trial_df["change_time_planned"].values,
                   trial_df["decision_time"].values).astype(float)
    return censored_hazard(dur, reached, dt=dt)


def lick_hazard(trial_df, dt=0.05):
    """Hazard of the first lick over trial time.

    Event = a lick occurred (``lick == 1``) at ``decision_time``; non-lick
    trials are right-censored at their ``decision_time``. Delegates to
    ``censored_hazard``.
    """
    ev = (trial_df["lick"].values.astype(int) == 1)
    return censored_hazard(trial_df["decision_time"].values.astype(float), ev, dt=dt)


def fa_lick_hazard(trial_df, dt=0.05):
    """Hazard of an anticipatory/early (FA) lick over trial time.

    Event = the trial outcome is the labeler ``fa`` (an early/anticipatory lick
    during baseline, BEFORE any change could occur) at ``decision_time``.

    FIX (round 2, 2026-06-18): an FA lick can ONLY happen before the change, so a
    non-FA (hit/miss) trial must leave the FA at-risk set at the CHANGE, not at its
    (later) ``decision_time``. Censoring non-FA trials at ``decision_time`` (which
    for hits/misses is AFTER the change) left the at-risk denominator un-depleted,
    collapsing this "hazard" onto the raw FA-event density (verified corr 0.97 with
    the FA-time histogram). Per-trial censor time:

    * FA trials → ``decision_time`` (the FA event time).
    * non-FA trials → ``min(change_time_planned, decision_time)`` (drop out at the
      change), falling back to ``decision_time`` when ``change_time_planned`` is NaN.

    Because changes never occur before 6 s (real-data fact; FA-lick median ≈ 4.57 s),
    this hazard isolates the *early-lick* timing — the temporal expectation expressed
    before the change can physically appear. Delegates to ``censored_hazard``.
    """
    is_fa = (trial_df["outcome"] == "fa").values
    dtime = trial_df["decision_time"].values.astype(float)
    ctime = trial_df["change_time_planned"].values.astype(float)
    # non-FA trials censor at the change; NaN change_time falls back to decision_time
    change_censor = np.where(np.isnan(ctime), dtime, np.minimum(ctime, dtime))
    censor_t = np.where(is_fa, dtime, change_censor)
    return censored_hazard(censor_t, is_fa, dt=dt)


def _peak_and_spread(centers, hazard):
    """Peak time (argmax of the hazard) and the std of the hazard-weighted time
    distribution. Returns ``(nan, nan)`` when the hazard is all-zero so an empty
    cell never raises on the weighted average."""
    w = np.clip(hazard, 0, None)
    if w.sum() <= 0:
        return float("nan"), float("nan")
    peak = centers[int(np.argmax(hazard))]
    mean = np.average(centers, weights=w)
    spread = float(np.sqrt(np.average((centers - mean) ** 2, weights=w)))
    return float(peak), spread


def timing_scores(trial_df, dt=0.05):
    """Timing = how strongly (and how sharply) the mouse expects the change now.

    Returns a dict:

    * ``change_hazard_peak_time`` — when the change-onset hazard peaks.
    * ``lick_hazard_peak_time`` — when the lick hazard peaks.
    * ``lick_hazard_spread`` — std of the hazard-weighted lick-time distribution
      (how temporally dispersed the licking is).
    * ``peak_offset`` — ``lick_peak − change_peak`` (how far the mouse's licking
      sits from the true change timing); NaN if either peak is undefined.
    """
    cc, ch, _ = change_onset_hazard(trial_df, dt=dt)
    lc, lh, _ = lick_hazard(trial_df, dt=dt)
    ch_peak, _ = _peak_and_spread(cc, ch)
    l_peak, l_spread = _peak_and_spread(lc, lh)
    return {"change_hazard_peak_time": ch_peak, "lick_hazard_peak_time": l_peak,
            "lick_hazard_spread": l_spread,
            "peak_offset": (l_peak - ch_peak) if np.isfinite(l_peak) and np.isfinite(ch_peak) else float("nan")}


# ── Data-quality gate (Phase-0.5) ──────────────────────────────────────────
# Distribution-justified inclusion thresholds — derived from, and re-runnable
# via, scripts/analysis/decision_latents/behavioral_qc_profile.py. The analysis
# UNIT is the (session × mood) CELL, not the session: a healthy session sliced by
# mood can still leave a thin cell. Each metric group has its OWN support gate, so
# a cell can be usable for one score and not another (e.g. enough go+catch for d′
# but a non-monotonic psychometric → no threshold). Figures/analyses filter on the
# matching ``usable_*`` flag so underpowered junk never reaches a result.
QC_MIN_GO = 8             # psychometric slope / d′: floor on go-trials (fit support)
QC_MIN_DISTINCT_CS = 2    # >= 2 distinct go change-sizes to fit a slope at all
QC_MIN_CATCH = 5          # SDT (d′, criterion, fa): catch trials for a real FA rate
QC_MIN_RT_PER_CS = 3      # >= 3 hit-RTs at a change-size for a stable per-cs RT-CV
QC_MIN_RTCV_CS = 2        # >= 2 such change-sizes for the aggregate rt_cv_by_cs
QC_MIN_TIMING_TRIALS = 20  # hazards/peaks need a populated trial timeline


def compute_cell_qc(trial_df):
    """Per-(session × mood) cell QC metrics + per-metric usability flags.

    Returns trial counts plus boolean ``usable_*`` flags built from the
    distribution-justified ``QC_*`` thresholds. ``usable_psychometric`` is only a
    COUNT pre-gate here (``has_psychometric_support``); the final monotonicity
    requirement (slope > 0) is ANDed in by ``descriptive_cell_table`` after the
    logistic fit, to avoid double-fitting. Flags:

    * ``has_psychometric_support`` — >= QC_MIN_GO go-trials AND >= QC_MIN_DISTINCT_CS
      distinct go change-sizes (necessary to attempt a slope/threshold fit).
    * ``usable_sdt`` — >= QC_MIN_GO go AND >= QC_MIN_CATCH catch trials (d′/criterion/fa).
    * ``usable_rtcv`` — >= QC_MIN_RTCV_CS change-sizes each with >= QC_MIN_RT_PER_CS hit-RTs.
    * ``usable_timing`` — >= QC_MIN_TIMING_TRIALS trials (hazard/peak support).
    """
    go = trial_df[trial_df["change_size"] > 1.0]
    catch = trial_df[np.isclose(trial_df["change_size"], 1.0)]
    n_go, n_catch, n_trials = len(go), len(catch), len(trial_df)
    n_distinct = int(go["change_size"].nunique())
    # hit-RT support per change size (RT = decision_time − planned change time on hits)
    hit = trial_df[trial_df["outcome"] == "hit"]
    if len(hit):
        rt = hit["decision_time"].values - hit["change_time_planned"].values
        n_cs_rt = sum(int((np.isclose(hit["change_size"].values, cs) & np.isfinite(rt)).sum())
                      >= QC_MIN_RT_PER_CS for cs in CHANGE_SIZES)
    else:
        n_cs_rt = 0
    return {
        "n_trials": n_trials, "n_go": n_go, "n_catch": n_catch,
        "n_distinct_cs": n_distinct, "n_cs_rt_support": n_cs_rt,
        "has_psychometric_support": bool(n_go >= QC_MIN_GO and n_distinct >= QC_MIN_DISTINCT_CS),
        "usable_sdt": bool(n_go >= QC_MIN_GO and n_catch >= QC_MIN_CATCH),
        "usable_rtcv": bool(n_cs_rt >= QC_MIN_RTCV_CS),
        "usable_timing": bool(n_trials >= QC_MIN_TIMING_TRIALS),
    }


def descriptive_cell_table(all_trials_df, min_cell_trials=1, dt=0.05):
    """One row per ``(session_name, state_label)`` cell, scored descriptively
    **behind a per-metric data-quality gate** (``compute_cell_qc``).

    Cells are kept only for moods in ``MAIN_MOODS + SEPARATE_MOODS``
    (Impulsive/StimSens/Disengaged); labeler ``Abort``/excluded moods never
    appear. Disengaged is kept but flagged ``reported_separately=True``.

    Each score group is computed ONLY if its gate passes — replacing the old
    blanket ``n_trials >= 20`` gate (which was the wrong granularity: it scored
    or dropped a whole cell, letting e.g. a no-distinct-Δ cell through to a junk
    d′, or excluding a small-but-clean psychometric). The matching ``usable_*``
    columns travel with every row so downstream figures filter on them. Scores
    whose gate fails are simply absent (NaN after the DataFrame is assembled).
    ``min_cell_trials`` is a hard floor below which a cell is skipped entirely
    (default 1 = keep all; the per-metric gates do the real work).
    ``session_dprime``/``comprehension_flag`` are read from the cell's first row.
    """
    keep = list(MAIN_MOODS) + list(SEPARATE_MOODS)
    rows = []
    for (sname, mood), cell in all_trials_df.groupby(["session_name", "state_label"]):
        if mood not in keep or len(cell) < min_cell_trials:
            continue
        qc = compute_cell_qc(cell)
        rec = {"session_name": sname, "state_label": mood,
               "reported_separately": mood in SEPARATE_MOODS,
               "session_dprime": cell["session_dprime"].iloc[0],
               "comprehension_flag": cell["comprehension_flag"].iloc[0]}
        rec.update(qc)
        s = None
        # ── psychometric + RT block (go-only) ──────────────────────────────
        if qc["has_psychometric_support"]:
            s = sharpness_scores(cell)
            rec["psy_slope"] = s["psy_slope"]
            rec["psy_threshold"] = s["psy_threshold"]
            # 3-param lapse-aware fit (Phase-2 F8 construct validity), carried
            # alongside the 2-param keys on the same psychometric-support gate.
            rec["psy_lapse"] = s["psy_lapse"]
            rec["psy_threshold_lapse"] = s["psy_threshold_lapse"]
            for cs in CHANGE_SIZES:
                rec[f"rt_mean_cs{cs}"] = s.get(f"rt_mean_cs{cs}", float("nan"))
                rec[f"rt_cv_cs{cs}"] = s.get(f"rt_cv_cs{cs}", float("nan"))
            # final psychometric usability = enough support AND a monotonic fit
            rec["usable_psychometric"] = bool(np.isfinite(s["psy_slope"]) and s["psy_slope"] > 0)
            # spec §5 cross-check: aggregate per-change-size Hit-RT CVs — only when
            # the per-cs support gate (usable_rtcv) passes.
            if qc["usable_rtcv"]:
                cv = [rec[f"rt_cv_cs{cs}"] for cs in CHANGE_SIZES
                      if np.isfinite(rec.get(f"rt_cv_cs{cs}", float("nan")))]
                rec["rt_cv_by_cs"] = float(np.mean(cv)) if cv else float("nan")
            else:
                rec["rt_cv_by_cs"] = float("nan")
        else:
            rec["usable_psychometric"] = False
            rec["rt_cv_by_cs"] = float("nan")
        # ── SDT block (needs catch): d′, criterion, fa ─────────────────────
        if qc["usable_sdt"]:
            if s is None:
                s = sharpness_scores(cell)
            it = itchiness_scores(cell, dt=dt)
            rec["dprime"] = s["dprime"]
            rec["criterion_c"] = it["criterion_c"]
            rec["fa_rate"] = it["fa_rate"]
            rec["baseline_hazard"] = it["baseline_hazard"]
        # ── timing block (hazards) ─────────────────────────────────────────
        if qc["usable_timing"]:
            rec.update(timing_scores(cell, dt=dt))
        rows.append(rec)
    return pd.DataFrame(rows)


def descriptive_latent_table(all_trials_df, cell_table):
    """Per-trial deliverable: each trial row joined to its cell's scores.

    Left-merges on ``(session_name, state_label)`` so every input trial keeps
    exactly one output row (cell scores broadcast to all of the cell's trials;
    trials in a cell that was never scored get NaN). Renames the cell-level
    columns to their per-trial latent names (``psy_slope -> sharpness_psy_slope``,
    ``fa_rate -> fa_rate_cell``, ``lick_hazard_peak_time -> hazard_peak_cell``)
    and keeps ``criterion_c`` and ``rt_cv_by_cs`` as-is (the latter is the spec §5
    cross-check column, propagated verbatim). The per-metric ``usable_*`` QC flags
    travel with every trial row so downstream (neural) analyses can filter latents
    on the data-quality of the cell that produced them.
    """
    key = ["session_name", "state_label"]
    cols = ["psy_slope", "criterion_c", "fa_rate", "lick_hazard_peak_time", "rt_cv_by_cs",
            "usable_psychometric", "usable_sdt", "usable_rtcv", "usable_timing"]
    avail = [c for c in cols if c in cell_table.columns]
    joined = all_trials_df.merge(cell_table[key + avail], on=key, how="left")
    return joined.rename(columns={"psy_slope": "sharpness_psy_slope",
                                  "fa_rate": "fa_rate_cell",
                                  "lick_hazard_peak_time": "hazard_peak_cell"})
