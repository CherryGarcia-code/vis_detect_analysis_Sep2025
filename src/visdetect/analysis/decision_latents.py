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


def sharpness_scores(trial_df):
    """Sharpness = how clearly the mouse tells the change happened.

    Operates on one (session × mood) cell's trial rows. Returns a dict:

    * ``psy_slope`` — logistic-fit slope of P(lick) vs ``log2(change_size)`` on
      GO trials (``change_size > 1.0``). NaN if < 8 go trials, < 2 distinct
      change sizes, or the fit fails.
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
            (a, b), _ = curve_fit(_logistic, x, y, p0=[0.0, 1.0], maxfev=5000)
            out["psy_slope"] = float(b)
        except Exception:
            out["psy_slope"] = float("nan")
    else:
        out["psy_slope"] = float("nan")
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
    * ``baseline_hazard`` — mean lick hazard over the pre-change window, computed
      from ``censored_hazard`` with FA-latency events (``outcome == "fa"``) and
      ``decision_time`` durations (everything else right-censored).
    """
    go = trial_df[trial_df["change_size"] > 1.0]
    catch = trial_df[np.isclose(trial_df["change_size"], 1.0)]
    H = _loglinear(go["lick"].mean() if len(go) else 0.0, max(len(go), 1))
    FA = _loglinear(catch["lick"].mean() if len(catch) else 0.0, max(len(catch), 1))
    crit = -(norm.ppf(H) + norm.ppf(FA)) / 2.0
    fa_rate = float((trial_df["outcome"] == "fa").mean())
    # baseline lick hazard: FA-latency events vs everything else censored at decision_time
    is_fa = (trial_df["outcome"] == "fa").values
    dur = trial_df["decision_time"].values.copy()
    _, hz, _ = censored_hazard(dur, is_fa, dt=dt)
    return {"criterion_c": float(crit), "fa_rate": fa_rate,
            "baseline_hazard": float(np.nanmean(hz)) if hz.size else float("nan")}
