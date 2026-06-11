"""B0 — which DDM knob does learning turn? Two-route change-detection accumulator.

PYDDM 0.9.0 RECONCILED API CONTRACT (authoritative — Tasks 3-7 must use exactly these):
  Imports (all top-level): from pyddm import (Model, Drift, Sample, Fittable,
      fit_adjust_model, NoiseConstant, BoundConstant, ICPoint, OverlayNonDecision,
      LossRobustLikelihood)
  1. Custom Drift: def get_drift(self, t, x, conditions, **kwargs)  # ORDER IS (t, x)!
     required_parameters may include non-numeric items (dict evmap, str R_kind, float dt);
     required_conditions = ["trial_uid"]; access via conditions["trial_uid"].
  2. Model(name=..., drift=..., noise=NoiseConstant(noise=1.0), bound=BoundConstant(B=a),
     IC=ICPoint(x0=z), overlay=OverlayNonDecision(nondectime=t0), dx=0.01, dt=DT,
     T_dur=..., choice_names=("lick","nolick"))   # choice_names MUST match the Sample.
  3. Simulate: sol = model.solve(conditions={"trial_uid": uid});
     rs = sol.resample(k, seed=...) -> a Sample; np.asarray(rs.choice_upper) = lick RTs,
     np.asarray(rs.choice_lower) = nolick RTs; sol.prob("lick") = P(upper).
     Prefer ONE resample(n_per_trial) per trial (not a loop of resample(1)).
  4. Sample.from_pandas_dataframe(df, rt_column_name="RT", choice_column_name="lick",
     choice_names=("lick","nolick"))  # value 1->upper(lick), 0->lower(nolick);
     every row needs a finite RT>0 (drop NaN RTs first).
  5. fit_adjust_model(sample=samp, model=fitm, lossfunction=LossRobustLikelihood,
     verbose=False)  # mutates fitm in place. Read back:
     names = fitm.get_model_parameter_names(); vals = [float(p) for p in
     fitm.get_model_parameters()]  (order matches). Free params: Fittable(minval=, maxval=).
  6. Held-out / CV log-lik: loss = LossRobustLikelihood(sample,
     required_conditions=["trial_uid"], dt=DT, T_dur=T_dur).loss(model); CV-LL = -loss.
     (sample is FIRST positional; model goes to .loss(). NOT LossRobustLikelihood(model, ...).)
  7. dt=0.02 triggers a benign "dt is large" warning (filter it). Non-numeric
     required_parameters are fine (paranoid checking off by default).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DT = 0.02   # integration grid (s); aligned to ~20 ms (sub-50 ms TF update period)

# Provisional response-window end (s) used as the Miss decision time. There is no
# canonical response-window constant in visdetect.analysis.constants; confirm against
# task params during the real-data run (spec sec 10).
RESPONSE_WINDOW_S = 2.155


def _safe_log2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    pos = x > 0
    out[pos] = np.log2(x[pos])
    return out


def _decision_time(trial) -> Tuple[float, int, bool]:
    """Return (decision_time_s, lick {0,1}, censored). Aligned to Baseline_ON."""
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


def build_trial_evidence(session, tf_base: float = None, dt: float = DT) -> pd.DataFrame:
    """Per-trial evidence trace e(t)=log2(TF(t)/tf_base) on a dt grid, truncated to
    [0, decision_time]. One row per usable trial with: outcome, change_size,
    change_time, decision_time, lick, censored, evidence (np.ndarray), trial_uid.

    The TF stream is the pre-planned design; only values up to the decision are used.
    """
    trials = getattr(session, "trials", []) or []
    rows = []
    for uid, t in enumerate(trials):
        oc = (getattr(t, "trialoutcome", "") or "").lower()
        if oc in ("abort", "ref"):
            continue                              # see spec sec 3 (censor/exclude decision)
        bv = getattr(t, "baseline_values", None)
        if bv is None:
            continue
        bv = np.asarray(bv, dtype=float).ravel()
        n_seen = getattr(t, "n_seen", None)
        if isinstance(n_seen, (int, np.integer)) and n_seen and n_seen > 0:
            bv = bv[: int(n_seen)]
        ct = float(getattr(t, "change_time", np.nan) or np.nan)
        cs = float(getattr(t, "change_size", np.nan) or np.nan)
        base = tf_base if tf_base is not None else float(np.nanmedian(bv)) or 1.0
        dec_t, lick, censored = _decision_time(t)
        if not np.isfinite(dec_t) or dec_t <= 0:
            continue
        n = int(round(dec_t / dt))
        bperiod = (ct / len(bv)) if (np.isfinite(ct) and len(bv) > 0) else dt
        tf = np.empty(n, dtype=float)
        for i in range(n):
            tau = i * dt
            if np.isfinite(ct) and tau >= ct:
                j = min(len(bv) - 1, int(tau / bperiod)) if len(bv) else 0
                tf[i] = bv[j] * cs if cs > 1.0 else (bv[j] if len(bv) else base)
            else:
                j = min(len(bv) - 1, int(tau / bperiod)) if len(bv) else 0
                tf[i] = bv[j] if len(bv) else base
        e = _safe_log2(tf / base)
        e = np.nan_to_num(e, nan=0.0)
        rows.append({"trial_uid": uid, "outcome": oc, "change_size": cs,
                     "change_time": ct, "decision_time": dec_t, "lick": int(lick),
                     "censored": bool(censored), "evidence": e})
    return pd.DataFrame(rows)
