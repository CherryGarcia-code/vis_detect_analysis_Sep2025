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

from visdetect.analysis import ddm
from visdetect.analysis.behavior import compute_session_performance
from visdetect.analysis.config import parse_session_date  # DDMMYYYY parser

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
    # A trial that exits at duration d occupies bins [0, d): its last (event) bin
    # is [d-dt, d), index round(d/dt)-1. It is at risk at the START of bin k only
    # while it has not yet exited, i.e. d > edges[k] (a trial exiting exactly at a
    # bin's left edge — e.g. censored at 0.05 — is at risk during bin0 [0,0.05)
    # but gone by the start of bin1).
    event_bin = np.round(durations / dt).astype(int) - 1   # bin [d-dt, d) in which the trial ends
    hazard = np.zeros(len(centers))
    for k in range(len(centers)):
        at_risk = np.sum(durations > edges[k] + 1e-12)         # still running at bin start
        n_event = np.sum(events & (event_bin == k))
        hazard[k] = (n_event / at_risk) if at_risk > 0 else 0.0
    survival = np.cumprod(1.0 - hazard)
    return centers, hazard, survival
