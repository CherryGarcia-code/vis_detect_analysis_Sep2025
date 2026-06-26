"""Engine-C — pyddm spot-check for B8 Phase-2 generative decision-latents.

GLM-vs-DDM **construct validity**: Engine-A (``decision_latents_generative``) is a
closed-form cloglog hazard-accumulator that exists precisely *because* a full
drift-diffusion (Fokker-Planck) fit is intractable on this task's long expert
baselines (change_time >= ~6 s). Engine-C is the honest cross-check: take B0's
*actual* pyddm two-route model (``visdetect.analysis.ddm`` — REFERENCE-ONLY, never
mutated here) and try to fit it on >=1 expert session, reporting the recovered
DDM knobs {v (drift), u (urgency), a (bound), z (start-point)} and the fit
log-likelihood.

Why this module is separate from ``decision_latents_generative``
----------------------------------------------------------------
``decision_latents_generative`` is deliberately **pyddm-free** (numpy/scipy only)
so the cluster recovery harness can import it on a minimal env. Engine-C imports
pyddm (via ``ddm``), so it lives here, isolated from that import path.

Tractability: the decision window
---------------------------------
A faithful pyddm fit would need ``T_dur`` to span each trial's whole evidence
trace. On expert sessions that is 6+ s of baseline before the change — exactly
where Fokker-Planck propagation blows up (the motivation for Engine-A). So this
spot-check restricts every trial to the **decision-relevant epoch**: the last
``DECISION_WINDOW_S`` seconds before the decision (covers the change pulse and the
response, plus a short pre-change lead-in), re-zeroed to t=0. The accumulator grid
is then bounded by ``DECISION_WINDOW_S`` regardless of the raw baseline length, so
the fit stays tractable and the spot-check runs in minutes.

If pyddm still cannot fit a session (raises, or produces no usable trials), the
failure is CAUGHT, a concise reason is logged, and that session's row is returned
with ``failed=True`` and NaN params — an informative, honest outcome (it
demonstrates the long-baseline intractability that motivates Engine-A), never a
silent skip.

DDM param -> returned-column mapping
------------------------------------
``ddm.fit_model`` returns canonical B0 names already (it applies
``ddm._PARAM_NAME_MAP = {'v':'v','u':'u','B':'a','x0':'z'}`` internally), so:
    v <- v   (sensory drift gain)
    u <- u   (impulsivity / urgency amplitude)
    a <- a   (= pyddm bound 'B', BoundConstant)
    z <- z   (= pyddm start-point ratio 'x0', ICPointRatio, in [-1, 1] of the bound)
    ll       (fit log-likelihood = -LossRobustLikelihood.loss)
"""
from __future__ import annotations

import logging
import warnings
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

# ddm is REFERENCE-ONLY: we READ its API (build_trial_evidence / fit_model /
# build_model / _sample_from_sim / LossRobustLikelihood) and never mutate it.
from visdetect.analysis import ddm

logger = logging.getLogger(__name__)

# Decision-relevant epoch kept per trial (s). Covers the change pulse + response
# window (RESPONSE_WINDOW_S ~= 2.155 s) plus a short pre-change lead-in. Bounding
# T_dur by this — instead of the raw 6+ s baseline — is what keeps the
# Fokker-Planck fit tractable for the spot-check. ~0.5 s margin above the window
# guards Miss decision times that land just past the change onset.
DECISION_WINDOW_S = 3.0
_T_DUR_MARGIN_S = 0.5

# Spot-check budget: cap sessions actually fit so the cross-check runs in minutes,
# not hours (it is a CONSTRUCT-VALIDITY spot-check, not a population fit).
MAX_SPOTCHECK_SESSIONS = 3

_PARAM_COLS = ("v", "u", "a", "z")


def _failed_row(session_name, reason: str) -> dict:
    """One result row for a session pyddm could not fit (NaN params, logged why)."""
    logger.warning("Engine-C: session %s pyddm fit FAILED: %s", session_name, reason)
    row = {"session": str(session_name), "failed": True, "reason": str(reason),
           "ll": np.nan, "n_trials": 0}
    for c in _PARAM_COLS:
        row[c] = np.nan
    return row


def _clip_to_decision_window(ev_df: pd.DataFrame, dt: float):
    """Build (evmap, sample_df) restricted to the last DECISION_WINDOW_S of each
    trial before its decision.

    ev_df is the output of ``ddm.build_trial_evidence`` (one row per usable trial
    with ``trial_uid``, ``evidence`` array, ``decision_time``, ``lick``). Returns:
      evmap     : {trial_uid -> clipped evidence array (re-zeroed to t=0)}
      sample_df : tidy DataFrame (trial_uid, RT, lick) with RT in the window.
    Trials whose clipped RT is not finite/>0 are dropped.
    """
    win_bins = int(round(DECISION_WINDOW_S / dt))
    evmap, rows = {}, []
    for _, r in ev_df.iterrows():
        uid = int(r["trial_uid"])
        e = np.asarray(r["evidence"], dtype=float)
        dec_t = float(r["decision_time"])
        if not np.isfinite(dec_t) or dec_t <= 0 or len(e) == 0:
            continue
        # keep only the last win_bins samples before the decision; re-zero time
        e_clip = e[-win_bins:] if len(e) > win_bins else e
        rt = min(dec_t, DECISION_WINDOW_S)
        if not np.isfinite(rt) or rt <= 0:
            continue
        evmap[uid] = e_clip
        rows.append({"trial_uid": uid, "RT": float(rt), "lick": int(r["lick"])})
    sample_df = pd.DataFrame(rows)
    return evmap, sample_df


def _fit_one_session(session, dt: float, fitparams: Optional[dict] = None) -> dict:
    """Fit B0's pyddm model on one session's decision-window data; return a result
    row. Raises on any genuine pyddm/intractability failure (caller catches).

    ``fitparams`` (optional) is forwarded to ``ddm.fit_model`` (e.g. a bounded +
    seeded differential-evolution config for a fast, deterministic fit); None uses
    pyddm's default optimizer."""
    session_name = getattr(session, "session_name", None) or getattr(session, "name", "?")

    ev_df = ddm.build_trial_evidence(session, dt=dt)
    if ev_df is None or len(ev_df) == 0:
        raise ValueError("no usable trials (build_trial_evidence returned empty)")

    evmap, sample_df = _clip_to_decision_window(ev_df, dt=dt)
    sample_df = sample_df.dropna(subset=["RT"])
    sample_df = sample_df[sample_df["RT"] > 0]
    if len(sample_df) == 0:
        raise ValueError("no usable trials after decision-window clipping")
    # pyddm needs both choices represented to identify a two-route model
    if sample_df["lick"].nunique() < 2:
        raise ValueError("degenerate sample: only one choice present (cannot fit DDM)")

    T_dur = DECISION_WINDOW_S + _T_DUR_MARGIN_S

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")               # silence benign "dt is large"
        sample = ddm._sample_from_sim(sample_df)      # tidy df -> pyddm Sample
        fit = ddm.fit_model(sample, evmap, dt=dt, T_dur=T_dur, fitparams=fitparams)
        # in-sample fit log-likelihood = -LossRobustLikelihood.loss
        model = ddm.build_model(fit, evmap, dt=dt, T_dur=T_dur)
        loss = ddm.LossRobustLikelihood(
            sample, required_conditions=["trial_uid"], dt=dt, T_dur=T_dur
        ).loss(model)
    ll = -float(loss)

    row = {"session": str(session_name), "failed": False, "reason": "",
           "ll": ll, "n_trials": int(len(sample_df))}
    for c in _PARAM_COLS:
        row[c] = float(fit[c])
    if not all(np.isfinite(row[c]) for c in _PARAM_COLS) or not np.isfinite(ll):
        raise ValueError(f"non-finite fit result: {{'ll': {ll}, "
                         + ", ".join(f"{c}: {row[c]}" for c in _PARAM_COLS) + "}")
    return row


def engine_c_spotcheck(expert_sessions: Sequence, dt: float = 0.02,
                       fitparams: Optional[dict] = None) -> pd.DataFrame:
    """Fit B0's pyddm DDM on each expert session; one result row per session.

    Parameters
    ----------
    expert_sessions : sequence of loaded session objects
        Each must expose ``trials`` (and ideally ``session_name``) so that
        ``ddm.build_trial_evidence`` can build per-trial evidence. Only the first
        ``MAX_SPOTCHECK_SESSIONS`` are fit (this is a spot-check, not a population
        fit); pass a pre-trimmed list to control exactly which sessions run.
    dt : float, default 0.02
        pyddm integration grid (s). 0.02 matches the B0 contract.
    fitparams : dict, optional
        Forwarded to ``ddm.fit_model``. None (default) uses pyddm's default
        differential-evolution optimizer (faithful but slow); pass a bounded +
        seeded config (e.g. ``{"seed":0,"maxiter":8,"popsize":5,"polish":False}``)
        for a fast, deterministic fit (used by the unit tests).

    Returns
    -------
    pandas.DataFrame
        One row per input session with columns
        ``session, v, u, a, z, ll, failed, reason, n_trials`` where
        {v=drift, u=urgency, a=bound, z=start-point} are B0's fitted DDM knobs and
        ll is the fit log-likelihood. Sessions pyddm could not fit have
        ``failed=True``, a logged ``reason``, and NaN params (never dropped).
    """
    sessions = list(expert_sessions)
    if len(sessions) > MAX_SPOTCHECK_SESSIONS:
        logger.info("Engine-C: %d sessions provided; spot-checking first %d.",
                    len(sessions), MAX_SPOTCHECK_SESSIONS)
        sessions = sessions[:MAX_SPOTCHECK_SESSIONS]

    rows: List[dict] = []
    for sess in sessions:
        session_name = getattr(sess, "session_name", None) or getattr(sess, "name", "?")
        try:
            rows.append(_fit_one_session(sess, dt=dt, fitparams=fitparams))
        except Exception as exc:  # noqa: BLE001 — any pyddm/intractability failure is informative
            rows.append(_failed_row(session_name, f"{type(exc).__name__}: {exc}"))

    cols = ["session", "v", "u", "a", "z", "ll", "failed", "reason", "n_trials"]
    return pd.DataFrame(rows, columns=cols)
