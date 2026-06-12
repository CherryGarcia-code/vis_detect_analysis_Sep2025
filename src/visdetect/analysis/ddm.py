"""B0 — which DDM knob does learning turn? Two-route change-detection accumulator.

PYDDM 0.9.0 RECONCILED API CONTRACT (authoritative — Tasks 3-7 must use exactly these):
  Imports (all top-level): from pyddm import (Model, Drift, Sample, Fittable,
      fit_adjust_model, NoiseConstant, BoundConstant, ICPointRatio, OverlayNonDecision,
      LossRobustLikelihood)
  1. Custom Drift: def get_drift(self, t, x, conditions, **kwargs)  # ORDER IS (t, x)!
     required_parameters may include non-numeric items (dict evmap, str R_kind, float dt);
     required_conditions = ["trial_uid"]; access via conditions["trial_uid"].
  2. Model(name=..., drift=..., noise=NoiseConstant(noise=1.0), bound=BoundConstant(B=a),
     IC=ICPointRatio(x0=z), overlay=OverlayNonDecision(nondectime=t0), dx=0.01, dt=DT,  # z = starting-point ratio in [-1,1] of bound
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


import warnings

import pyddm
from pyddm import (Model, Drift, NoiseConstant, BoundConstant, ICPointRatio,
                   OverlayNonDecision, Sample)

CHOICE_NAMES = ("lick", "nolick")   # value 1 -> upper(lick), 0 -> lower(nolick)


def rectify(e, kind: str, g_up: float = 1.0, g_down: float = 1.0):
    """Rectification nonlinearity R(.) applied to instantaneous TF evidence e(t)."""
    e = np.asarray(e, dtype=float)
    if kind == "symmetric":
        return e
    if kind == "halfwave":
        return np.clip(e, 0.0, None)                       # slow pulses ignored
    if kind == "asym":
        return np.where(e >= 0, g_up * e, g_down * e)
    raise ValueError(kind)


class DriftTwoRoute(Drift):
    """drift(t) = v*R(e(t)) - lam*x + u*h(t); e(t) looked up per trial via trial_uid.

    Route 1 (sensory) = v * R(evidence); route 2 (impulsivity) = u * h(t), with
    h(t)=t for rising urgency or h(t)=1 for constant. lam is the leak on x.
    """
    name = "two_route"
    required_parameters = ["v", "u", "lam", "R_kind", "urgency_kind", "dt", "evmap"]
    required_conditions = ["trial_uid"]

    def get_drift(self, t, x, conditions, **kwargs):
        ev = self.evmap.get(conditions["trial_uid"])
        i = int(round(t / self.dt))
        e_t = ev[i] if (ev is not None and 0 <= i < len(ev)) else 0.0
        sensory = self.v * rectify(np.array([e_t]), self.R_kind)[0]
        urge = self.u * t if self.urgency_kind == "rising" else self.u
        return sensory - self.lam * x + urge


def build_model(params: dict, evmap: dict, R: str = "halfwave",
                urgency: str = "rising", dt: float = DT, T_dur: float = 3.5) -> Model:
    """Assemble the two-route pyddm Model. params values may be floats or pyddm
    Fittable objects (for the free parameters during fitting)."""
    return Model(
        name="B0_two_route",
        drift=DriftTwoRoute(v=params["v"], u=params["u"], lam=params.get("lam", 0.0),
                            R_kind=R, urgency_kind=urgency, dt=dt, evmap=evmap),
        noise=NoiseConstant(noise=1.0),
        bound=BoundConstant(B=params["a"]),
        IC=ICPointRatio(x0=params.get("z", 0.0)),
        overlay=OverlayNonDecision(nondectime=params.get("t0", 0.0)),
        dx=0.01, dt=dt, T_dur=T_dur, choice_names=CHOICE_NAMES,
    )


def simulate_sample(evmap, conds, params, R="halfwave", urgency="rising",
                    dt=DT, T_dur=3.5, n_per_trial=200, seed=0) -> pd.DataFrame:
    """Simulate n_per_trial draws per trial condition; return a tidy DataFrame with
    columns (trial_uid, RT, lick). lick=1 = upper-bound (lick) crossing, lick=0 =
    lower-bound crossing. Undecided draws (no crossing within T_dur) are omitted."""
    model = build_model(params, evmap, R=R, urgency=urgency, dt=dt, T_dur=T_dur)
    rows = []
    for offset, uid in enumerate(conds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # silence "dt is large"
            sol = model.solve(conditions={"trial_uid": uid})
            rs = sol.resample(n_per_trial, seed=seed + offset)
        for rt in np.asarray(rs.choice_upper):
            rows.append({"trial_uid": uid, "RT": float(rt), "lick": 1})
        for rt in np.asarray(rs.choice_lower):
            rows.append({"trial_uid": uid, "RT": float(rt), "lick": 0})
    return pd.DataFrame(rows)


from pyddm import Fittable, fit_adjust_model, LossRobustLikelihood

# pyddm reports fitted-parameter names as ['v','u','B','x0']; map to canonical B0 names.
_PARAM_NAME_MAP = {"v": "v", "u": "u", "B": "a", "x0": "z"}
_FIXED_DEFAULTS = {"t0": 0.05, "lam": 0.0}


def _sample_from_sim(sim_df) -> "Sample":
    """Tidy DataFrame (trial_uid, RT, lick) -> pyddm Sample with matching choice_names."""
    df = sim_df.dropna(subset=["RT"]).copy()
    df = df[df["RT"] > 0]
    return Sample.from_pandas_dataframe(df, rt_column_name="RT",
                                        choice_column_name="lick",
                                        choice_names=CHOICE_NAMES)


def fit_model(sample, evmap, R="halfwave", urgency="rising", dt=DT, T_dur=3.5,
              fixed: Optional[dict] = None, fitparams: Optional[dict] = None) -> dict:
    """Fit free params {v,a,z,u} by robust likelihood; z is the starting-point ratio.
    Any key present in `fixed` (e.g. {"t0":...,"lam":...,"u":0.0}) is held constant.
    Returns a complete dict keyed by canonical names v,a,z,u,t0,lam."""
    fixed = fixed or {}

    def free(lo, hi):
        return Fittable(minval=lo, maxval=hi)

    params = dict(
        v=fixed.get("v", free(0, 10)),
        a=fixed.get("a", free(0.3, 3.0)),
        z=fixed.get("z", free(-0.9, 0.9)),         # ratio of bound (ICPointRatio)
        u=fixed.get("u", free(0, 5)),
        t0=fixed.get("t0", _FIXED_DEFAULTS["t0"]),
        lam=fixed.get("lam", _FIXED_DEFAULTS["lam"]),
    )
    model = build_model(params, evmap, R=R, urgency=urgency, dt=dt, T_dur=T_dur)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _fp = {"fitparams": fitparams} if fitparams is not None else {}
        fit_adjust_model(sample=sample, model=model,
                         lossfunction=LossRobustLikelihood, verbose=False, **_fp)
    raw = {name: float(val) for name, val in
           zip(model.get_model_parameter_names(), model.get_model_parameters())}
    out = {_PARAM_NAME_MAP.get(k, k): v for k, v in raw.items()}
    # fold in fixed scalars so the returned dict is a complete param set
    for k in ("v", "a", "z", "u", "t0", "lam"):
        if k not in out:
            val = fixed.get(k, _FIXED_DEFAULTS.get(k))
            out[k] = float(val) if isinstance(val, (int, float)) else val
    return out


def recover_parameters(true_params, evmap, conds, R="halfwave", urgency="rising",
                       dt=DT, T_dur=3.5, n_per_trial=1, seed=0) -> dict:
    """Simulate from known params using the given evidence, refit, return recovered dict.
    The core identifiability check (spec sec 6): poor recovery is itself a finding."""
    sim = simulate_sample(evmap, conds, true_params, R=R, urgency=urgency,
                          dt=dt, T_dur=T_dur, n_per_trial=n_per_trial, seed=seed)
    samp = _sample_from_sim(sim)
    return fit_model(samp, evmap, R=R, urgency=urgency, dt=dt, T_dur=T_dur,
                     fixed={"t0": true_params["t0"], "lam": true_params["lam"]})


from sklearn.model_selection import KFold


def _cv_loglik(sample_df, evmap, R, urgency, fixed, dt=DT, T_dur=3.5, k=3, seed=0,
               fitparams=None) -> float:
    """K-fold held-out mean log-likelihood (higher = better) of a model spec.

    sample_df is a tidy DataFrame (trial_uid, RT, lick). Held-out loss is
    LossRobustLikelihood(sample, ...).loss(model); CV log-likelihood = -loss.
    """
    df = sample_df.dropna(subset=["RT"]).reset_index(drop=True)
    df = df[df["RT"] > 0].reset_index(drop=True)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    lls = []
    for tr, te in kf.split(df):
        m = fit_model(_sample_from_sim(df.iloc[tr]), evmap, R=R, urgency=urgency,
                      dt=dt, T_dur=T_dur, fixed=fixed, fitparams=fitparams)
        model = build_model({**m, **fixed}, evmap, R=R, urgency=urgency, dt=dt, T_dur=T_dur)
        te_samp = _sample_from_sim(df.iloc[te])
        loss = LossRobustLikelihood(te_samp, required_conditions=["trial_uid"],
                                    dt=dt, T_dur=T_dur).loss(model)
        lls.append(-float(loss))
    return float(np.mean(lls))


def select_structure(sample_df, evmap, fixed=None, dt=DT, T_dur=3.5, k=3, seed=0,
                     fitparams=None) -> dict:
    """Step 0: choose rectification R and urgency form by CV log-likelihood."""
    fixed = fixed or {"t0": 0.05, "lam": 0.0}
    grid = [(R, U) for R in ("symmetric", "halfwave", "asym") for U in ("rising", "const")]
    scores = {f"{R}|{U}": _cv_loglik(sample_df, evmap, R, U, fixed, dt, T_dur, k=k,
                                     seed=seed, fitparams=fitparams)
              for (R, U) in grid}
    best = max(scores, key=scores.get)
    R, U = best.split("|")
    return {"R": R, "urgency": U, "scores": scores}


def route_attribution(sample_df, evmap, R="halfwave", urgency="rising", fixed=None,
                      dt=DT, T_dur=3.5, k=3, seed=0, fitparams=None) -> dict:
    """Step 0b: two-route vs TF-only (u fixed to 0) by CV log-likelihood."""
    fixed = fixed or {"t0": 0.05, "lam": 0.0}
    two = _cv_loglik(sample_df, evmap, R, urgency, fixed, dt, T_dur, k=k, seed=seed,
                     fitparams=fitparams)
    tf_only = _cv_loglik(sample_df, evmap, R, urgency, {**fixed, "u": 0.0}, dt, T_dur,
                         k=k, seed=seed, fitparams=fitparams)
    return {"two_route_cvll": two, "tf_only_cvll": tf_only,
            "two_route_wins": two > tf_only}


def _aic(ll: float, k_params: int) -> float:
    return 2 * k_params - 2 * ll


def compare_stage_models(samples_by_stage: Dict[str, tuple], R="halfwave",
                         urgency="rising", fixed=None, dt=DT, T_dur=3.5,
                         fitparams=None) -> dict:
    """Nested comparison: which single parameter must vary across stages.

    samples_by_stage: {stage: (sample_df, evmap)} with sample_df a tidy DataFrame
    (trial_uid, RT, lick). Fits each stage independently, then scores M_shared /
    M_v / M_a / M_zu / M_full by AIC over the pooled in-sample log-likelihood
    (using each stage's fit for the "free" parameter(s), stage-0's fit otherwise).
    The minimal model that fits ~as well as M_full names the knob learning turns.
    """
    fixed = fixed or {"t0": 0.05, "lam": 0.0}
    stages = list(samples_by_stage)
    per_stage = {s: fit_model(_sample_from_sim(df), ev, R=R, urgency=urgency,
                              dt=dt, T_dur=T_dur, fixed=fixed, fitparams=fitparams)
                 for s, (df, ev) in samples_by_stage.items()}

    def stage_ll(free_keys):
        ll = 0.0
        for s, (df, ev) in samples_by_stage.items():
            p = {**per_stage[stages[0]]}                  # shared baseline (stage 0)
            for kk in free_keys:                          # listed keys take this stage's value
                p[kk] = per_stage[s][kk]
            model = build_model({**p, **fixed}, ev, R=R, urgency=urgency, dt=dt, T_dur=T_dur)
            ll += -float(LossRobustLikelihood(_sample_from_sim(df),
                         required_conditions=["trial_uid"], dt=dt, T_dur=T_dur).loss(model))
        return ll

    ladder = {"M_shared": [], "M_v": ["v"], "M_a": ["a"], "M_zu": ["z", "u"],
              "M_full": ["v", "a", "z", "u"]}
    n = len(stages)
    aics = {name: _aic(stage_ll(keys), 4 + len(keys) * (n - 1))
            for name, keys in ladder.items()}
    winner = min(aics, key=aics.get)
    return {"winner": winner, "aic": aics,
            "delta_v": per_stage[stages[-1]]["v"] - per_stage[stages[0]]["v"],
            "delta_u": per_stage[stages[-1]]["u"] - per_stage[stages[0]]["u"],
            "per_stage": per_stage}
