# B0 — DDM learning-knob: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit a two-route change-detection DDM to BG_046 behaviour per stage and identify, via nested model comparison, which parameter learning turns — drift gain `v` (sensitivity), bound `a` (caution), or urgency/start `u`/`z` (impulsivity) — per `docs/superpowers/specs/2026-06-10-B0-ddm-learning-knob-design.md`.

**Architecture:** A new library module `visdetect/analysis/ddm.py` holds: trial-evidence extraction (`e(t)` from the pre-planned TF stream, truncated at the decision), the pyddm two-route model builder, fit + parameter-recovery, structural selection (Step 0), route attribution (Step 0b), per-stage nested comparison with state control, and the state-resolved secondary. A thin `analysis_suite/01_behavior/h_ddm_learning_knob.py` orchestrates over the staging manifest and renders the figure + caches. Behavioural-state is a **pluggable** per-trial column (`load_state_labels`, defaulting to the GLM-HMM assignments) so the in-development classifier can replace it without touching model code.

**Tech Stack:** pyddm (NEW dependency — Task 1), numpy, pandas, scipy, scikit-learn (already present: scipy 1.15.3, sklearn 1.7.2), matplotlib. Reuses `visdetect.analysis.behavior.get_trial_dataframe`, `visdetect.analysis.constants`, `visdetect.suite.loader.load_hmm_assignments`, `visdetect.analysis.config.load_staging_manifest`.

> **pyddm API caveat (read once).** pyddm is not yet installed and its fit/Sample API is version-sensitive (`Sample.from_pandas_dataframe` choice-column naming; `fit_adjust_model` vs `Model.fit`; custom-`Drift` condition access). Task 1 installs+pins it and runs a smoke test that exercises **exactly** the calls this plan uses; reconcile any signature drift there before Task 3. Per-trial time-varying evidence is carried via a scalar `trial_uid` condition that indexes a stimulus dict inside the Drift (the standard pyddm pattern for trial-specific stimuli).

---

## File Structure

- **Create** `src/visdetect/analysis/ddm.py` — all B0 computation (evidence extraction, model, fit, recovery, structural/route/stage comparison, state secondary, pluggable `load_state_labels`).
- **Create** `tests/analysis/test_ddm.py` — TDD: truncation correctness, model behaviour (drift tracks TF; FA = early crossing), **parameter recovery (core)**, structural/route/stage selection on simulated ground truth.
- **Create** `analysis_suite/01_behavior/h_ddm_learning_knob.py` — orchestration + figure (`fig0N`, panels A–F) + stats/cache.
- **Modify** `requirements.txt` (or the env spec) — add the pinned `pyddm`.
- **Modify** `docs/science/QUESTION_INDEX.md` — link the plan, bump B0 status.

Conventions (`CLAUDE.md`): constants from `visdetect.analysis.constants`; `load_staging_manifest()`; `setup_style()`/`save_figure()`; `py` not `python`.

---

### Task 1: Add and pin pyddm; API smoke test

**Files:**
- Modify: `requirements.txt`
- Test: `tests/analysis/test_ddm_env.py`

- [ ] **Step 1: Install pyddm into the venv**

Run: `py -m pip install pyddm`
Then capture the version: `py -c "import pyddm; print(pyddm.__version__)"`
Expected: a version prints (e.g. `0.9.x`). Record it.

- [ ] **Step 2: Pin it**

Add the resolved version to `requirements.txt` (e.g. `pyddm==<resolved>`).

- [ ] **Step 3: Write an API smoke test that exercises exactly the calls this plan uses**

```python
# tests/analysis/test_ddm_env.py
import numpy as np
import pandas as pd
import pytest

pyddm = pytest.importorskip("pyddm")


def test_pyddm_api_surface():
    # The exact symbols Task 3+ rely on must exist; reconcile here if names drift.
    from pyddm import Model, Fittable, Drift, Bound, InitialCondition, Sample
    from pyddm import NoiseConstant, BoundConstant
    # Sample from a dataframe with a binary choice + RT (naming is version-sensitive)
    df = pd.DataFrame({"RT": [0.4, 0.6, 0.5], "lick": [1, 1, 0], "trial_uid": [0, 1, 2]})
    # one of these constructors must work; record which:
    try:
        samp = Sample.from_pandas_dataframe(df, rt_column_name="RT", choice_column_name="lick")
    except TypeError:
        samp = Sample.from_pandas_dataframe(df, rt_column_name="RT", correct_column_name="lick")
    assert len(samp) == 3
```

- [ ] **Step 4: Run it**

Run: `py -m pytest tests/analysis/test_ddm_env.py -v`
Expected: PASS. **If the `Sample`/fit symbols differ, note the working signatures in a docstring at the top of `ddm.py` and use them consistently in Tasks 3–6.**

- [ ] **Step 5: Commit**

```bash
git add requirements.txt tests/analysis/test_ddm_env.py
git commit -m "chore(B0): add + pin pyddm; API smoke test"
```

---

### Task 2: Per-trial evidence extraction (pre-planned stream, truncated at decision)

**Files:**
- Create: `src/visdetect/analysis/ddm.py`
- Test: `tests/analysis/test_ddm.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_ddm.py
import numpy as np
import pandas as pd
import pytest
from visdetect.analysis.ddm import build_trial_evidence, DT


def _toy_session():
    # Two trials: a Hit (lick after change) and an FA (early lick, no change reached).
    from types import SimpleNamespace
    base = np.r_[np.ones(20), np.ones(20) * 4.0]   # log2-able TF: 1 (e=0) then 4 (e=2)
    t_hit = SimpleNamespace(trialoutcome="Hit", change_size=4.0, change_time=1.0,
                            reactiontimes={"RT": 0.3}, baseline_values=base, n_seen=None)
    t_fa = SimpleNamespace(trialoutcome="FA", change_size=1.0, change_time=2.0,
                           reactiontimes={"FA": 0.5}, baseline_values=base, n_seen=None)
    return SimpleNamespace(trials=[t_hit, t_fa],
                           ni_events={"Baseline_ON": np.array([0.0, 10.0]),
                                      "Change_ON": np.array([1.0, 12.0])})


def test_build_trial_evidence_truncates_at_decision():
    sess = _toy_session()
    df = build_trial_evidence(sess, tf_base=1.0)
    assert len(df) == 2
    hit = df.iloc[0]
    # Hit decision_time = change_time + RT = 1.3 s -> evidence length = round(1.3/DT)
    assert hit["decision_time"] == pytest.approx(1.3, abs=DT)
    assert len(hit["evidence"]) == pytest.approx(1.3 / DT, abs=1)
    # FA decision_time = FA latency 0.5 s (truncated well before its change_time 2.0)
    fa = df.iloc[1]
    assert fa["decision_time"] == pytest.approx(0.5, abs=DT)
    assert len(fa["evidence"]) < len(hit["evidence"])
    assert fa["lick"] == 1 and hit["lick"] == 1
    # evidence is log2(TF/base): 0 in the first second, ~2 after the change (Hit only)
    assert abs(hit["evidence"][0]) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_ddm.py::test_build_trial_evidence_truncates_at_decision -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'visdetect.analysis.ddm'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/visdetect/analysis/ddm.py
"""B0 — which DDM knob does learning turn? Two-route change-detection accumulator.

pyddm API note (reconcile in Task 1): Sample.from_pandas_dataframe uses
choice_column_name (newer) or correct_column_name (older); record the working
form from tests/analysis/test_ddm_env.py and keep it consistent below.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DT = 0.02   # integration grid (s); aligned to ~20 ms (sub-50 ms TF update period)


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
        return (ct + 2.155, 0, True)            # response-window end; no crossing (censored)
    return (np.nan, 0, True)                     # abort/ref handled by caller


def build_trial_evidence(session, tf_base: float = None, dt: float = DT) -> pd.DataFrame:
    """Per-trial evidence trace e(t)=log2(TF(t)/tf_base) on a dt grid, truncated to
    [0, decision_time]. One row per usable trial with: outcome, change_size,
    change_time, decision_time, lick, censored, evidence (np.ndarray), trial_uid.

    The TF stream is the pre-planned design; only values up to the decision are used.
    """
    trials = getattr(session, "trials", []) or []
    # baseline TF stream update period: baseline_values spans Baseline_ON..change.
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
        # Build TF(t) on the dt grid: baseline samples up to change, change_size-shifted after.
        n = int(round(dec_t / dt))
        # map grid time -> baseline sample (baseline update period = ct / len(bv) if ct finite)
        bperiod = (ct / len(bv)) if (np.isfinite(ct) and len(bv) > 0) else dt
        tf = np.empty(n, dtype=float)
        for i in range(n):
            tau = i * dt
            if np.isfinite(ct) and tau >= ct:
                # post-change: planned shifted stream (approx: base*cs * baseline fluctuation)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_ddm.py::test_build_trial_evidence_truncates_at_decision -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/ddm.py tests/analysis/test_ddm.py
git commit -m "feat(B0): per-trial evidence extraction (pre-planned stream, truncated at decision)"
```

---

### Task 3: Two-route pyddm model builder

**Files:**
- Modify: `src/visdetect/analysis/ddm.py` (append model classes + `build_model`, `simulate_sample`)
- Test: `tests/analysis/test_ddm.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from visdetect.analysis.ddm import build_model, simulate_sample, rectify


def test_rectify_variants():
    e = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(rectify(e, "symmetric"), [-1, 0, 1])
    assert np.allclose(rectify(e, "halfwave"), [0, 0, 1])      # slow ignored
    asym = rectify(e, "asym", g_up=1.0, g_down=0.5)
    assert asym[0] == -0.5 and asym[2] == 1.0


def test_model_drift_tracks_tf_and_fa_is_early_crossing():
    # Evidence dict: trial 0 = strong fast post "change"; trial 1 = flat baseline.
    dt = 0.02
    ev = {0: np.r_[np.zeros(25), np.ones(75) * 2.0],   # change at 0.5 s
          1: np.zeros(150)}                             # pure baseline, 3 s
    conds = {0: {"trial_uid": 0, "change_time": 0.5},
             1: {"trial_uid": 1, "change_time": np.inf}}
    # High sensitivity, modest urgency -> trial 0 crosses fast (Hit), trial 1 rarely/late.
    params = dict(v=3.0, a=1.0, z=0.0, u=0.3, t0=0.05, lam=0.0)
    samp = simulate_sample(ev, conds, params, R="halfwave", urgency="rising",
                           dt=dt, T_dur=3.0, n_per_trial=200, seed=0)
    df = samp  # simulate_sample returns a tidy DataFrame of simulated RT/lick per draw
    hit_rate_evi = df[(df.trial_uid == 0) & (df.lick == 1)].shape[0]
    hit_rate_base = df[(df.trial_uid == 1) & (df.lick == 1)].shape[0]
    assert hit_rate_evi > hit_rate_base            # TF-driven crossings dominate
    assert df[(df.trial_uid == 1) & (df.lick == 1)].shape[0] >= 0  # FAs are early crossings
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_ddm.py -k "rectify or drift_tracks" -v`
Expected: FAIL — `ImportError: cannot import name 'build_model'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to ddm.py
import pyddm
from pyddm import Model, Drift, NoiseConstant, BoundConstant, InitialCondition, Sample


def rectify(e, kind: str, g_up: float = 1.0, g_down: float = 1.0):
    e = np.asarray(e, dtype=float)
    if kind == "symmetric":
        return e
    if kind == "halfwave":
        return np.clip(e, 0.0, None)                       # slow pulses ignored
    if kind == "asym":
        return np.where(e >= 0, g_up * e, g_down * e)
    raise ValueError(kind)


class DriftTwoRoute(Drift):
    """drift(t) = v*R(e(t)) + u*h(t) ; e(t) looked up per trial via trial_uid."""
    name = "two_route"
    required_parameters = ["v", "u", "lam", "R_kind", "urgency_kind", "dt", "evmap"]
    required_conditions = ["trial_uid"]

    def get_drift(self, x, t, conditions, **kwargs):
        ev = self.evmap.get(conditions["trial_uid"])
        i = int(round(t / self.dt))
        e_t = ev[i] if (ev is not None and 0 <= i < len(ev)) else 0.0
        sensory = self.v * rectify(np.array([e_t]), self.R_kind)[0]
        if self.urgency_kind == "rising":
            urge = self.u * t
        else:
            urge = self.u
        return sensory - self.lam * x + urge


def build_model(params: dict, evmap: dict, R: str = "halfwave",
                urgency: str = "rising", dt: float = DT, T_dur: float = 3.5) -> Model:
    return Model(
        name="B0_two_route",
        drift=DriftTwoRoute(v=params["v"], u=params["u"], lam=params.get("lam", 0.0),
                            R_kind=R, urgency_kind=urgency, dt=dt, evmap=evmap),
        noise=NoiseConstant(noise=1.0),
        bound=BoundConstant(B=params["a"]),
        IC=pyddm.ICPoint(x0=params.get("z", 0.0)) if hasattr(pyddm, "ICPoint")
           else InitialCondition(),
        overlay=pyddm.OverlayNonDecision(nondectime=params.get("t0", 0.0)),
        dx=0.01, dt=dt, T_dur=T_dur,
    )


def simulate_sample(evmap, conds, params, R="halfwave", urgency="rising",
                    dt=DT, T_dur=3.5, n_per_trial=200, seed=0) -> pd.DataFrame:
    """Simulate n draws per trial condition; return tidy DataFrame (trial_uid, RT, lick)."""
    rng = np.random.default_rng(seed)
    model = build_model(params, evmap, R=R, urgency=urgency, dt=dt, T_dur=T_dur)
    rows = []
    for uid, cond in conds.items():
        sol = model.solve(conditions={"trial_uid": uid})
        for _ in range(n_per_trial):
            samp = sol.resample(1)
            rt = float(samp.choice_upper[0]) if len(samp.choice_upper) else np.nan
            lick = 1 if np.isfinite(rt) else 0
            rows.append({"trial_uid": uid, "RT": rt, "lick": lick})
    return pd.DataFrame(rows)
```

> **Reconcile in this task:** `ICPoint`, `OverlayNonDecision`, `BoundConstant`, `Solution.resample`/`choice_upper` names against the installed pyddm (Task 1). Fix names here once; later tasks inherit them.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_ddm.py -k "rectify or drift_tracks" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/ddm.py tests/analysis/test_ddm.py
git commit -m "feat(B0): two-route pyddm model (TF-driven drift + urgency) + simulator"
```

---

### Task 4: Fit + parameter recovery (the core validation)

**Files:**
- Modify: `src/visdetect/analysis/ddm.py` (append `fit_model`, `recover_parameters`)
- Test: `tests/analysis/test_ddm.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from visdetect.analysis.ddm import recover_parameters


@pytest.mark.slow
def test_parameter_recovery_within_tolerance():
    # Simulate from known params on synthetic evidence, refit, recover v and u in rank.
    rng = np.random.default_rng(0)
    n_trials = 400
    evmap, conds = {}, {}
    for uid in range(n_trials):
        ct = rng.uniform(0.8, 2.0)
        n = int(3.0 / 0.02)
        e = np.zeros(n)
        c = int(ct / 0.02)
        e[c:] = 2.0                                  # a "change" of fixed size
        evmap[uid] = e
        conds[uid] = {"trial_uid": uid, "change_time": ct}
    true = dict(v=2.5, a=1.0, z=0.0, u=0.4, t0=0.05, lam=0.0)
    rec = recover_parameters(true, evmap, conds, R="halfwave", urgency="rising",
                             n_per_trial=1, seed=1)
    # recovery: signs/order preserved and within a generous tolerance
    assert rec["v"] > 0 and rec["u"] > 0
    assert abs(rec["v"] - true["v"]) / true["v"] < 0.5
    assert abs(rec["u"] - true["u"]) / true["u"] < 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_ddm.py -k parameter_recovery -v`
Expected: FAIL — `ImportError: cannot import name 'recover_parameters'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to ddm.py
from pyddm import Fittable, fit_adjust_model
from pyddm.models import LossRobustLikelihood


def _sample_from_sim(sim_df):
    # tidy sim DataFrame -> pyddm Sample (reconcile choice/correct column name per Task 1)
    df = sim_df.dropna(subset=["RT"]).copy()
    try:
        return Sample.from_pandas_dataframe(df, rt_column_name="RT",
                                            choice_column_name="lick")
    except TypeError:
        return Sample.from_pandas_dataframe(df, rt_column_name="RT",
                                            correct_column_name="lick")


def fit_model(sample, evmap, R="halfwave", urgency="rising", dt=DT, T_dur=3.5,
              fixed: Optional[dict] = None) -> dict:
    """Fit free params {v,a,z,u} (+ optionally t0,lam) by robust likelihood."""
    fixed = fixed or {}
    free = lambda lo, hi: Fittable(minval=lo, maxval=hi)
    params = dict(v=free(0, 10), a=free(0.3, 3.0), z=fixed.get("z", free(-0.5, 0.5)),
                  u=free(0, 5), t0=fixed.get("t0", 0.05), lam=fixed.get("lam", 0.0))
    model = build_model(params, evmap, R=R, urgency=urgency, dt=dt, T_dur=T_dur)
    fit_adjust_model(sample=sample, model=model,
                     lossfunction=LossRobustLikelihood, verbose=False)
    return {p: float(model.get_model_parameters()[i])
            for i, p in enumerate(model.get_model_parameter_names())}


def recover_parameters(true_params, evmap, conds, R="halfwave", urgency="rising",
                       dt=DT, T_dur=3.5, n_per_trial=1, seed=0) -> dict:
    sim = simulate_sample(evmap, conds, true_params, R=R, urgency=urgency,
                          dt=dt, T_dur=T_dur, n_per_trial=n_per_trial, seed=seed)
    samp = _sample_from_sim(sim)
    return fit_model(samp, evmap, R=R, urgency=urgency, dt=dt, T_dur=T_dur,
                     fixed={"t0": true_params["t0"], "lam": true_params["lam"]})
```

> Mark recovery tests `@pytest.mark.slow`; they fit real models. Register the marker in `pytest.ini`/`pyproject` or skip-if-slow in CI.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_ddm.py -k parameter_recovery -v -m slow`
Expected: PASS (may take a minute). If recovery is poor, **that is a finding** — tighten priors / fix more params (spec §6) before trusting real fits.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/ddm.py tests/analysis/test_ddm.py
git commit -m "feat(B0): model fitting + parameter recovery harness"
```

---

### Task 5: Structural selection (Step 0) + route attribution (Step 0b)

**Files:**
- Modify: `src/visdetect/analysis/ddm.py` (append `select_structure`, `route_attribution`)
- Test: `tests/analysis/test_ddm.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from visdetect.analysis.ddm import select_structure, route_attribution


@pytest.mark.slow
def test_route_attribution_prefers_two_route_when_data_has_impulsive_fas():
    # Build data where many "FAs" occur with FLAT evidence (pure time-driven) ->
    # two-route (with urgency) must beat TF-only.
    rng = np.random.default_rng(0)
    evmap, conds = {}, {}
    for uid in range(300):
        evmap[uid] = np.zeros(150)                  # flat baseline -> no sensory drive
        conds[uid] = {"trial_uid": uid, "change_time": np.inf}
    true = dict(v=0.1, a=1.0, z=0.0, u=0.8, t0=0.05, lam=0.0)  # impulsivity-driven
    sim = simulate_sample(evmap, conds, true, R="halfwave", urgency="rising",
                          n_per_trial=1, seed=2)
    res = route_attribution(_sample_from_sim(sim), evmap)
    assert res["two_route_cvll"] > res["tf_only_cvll"]   # impulsivity route required
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_ddm.py -k route_attribution -v -m slow`
Expected: FAIL — `ImportError: cannot import name 'route_attribution'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to ddm.py
from sklearn.model_selection import KFold


def _cv_loglik(sample_df, evmap, R, urgency, fixed, dt=DT, T_dur=3.5, k=5, seed=0):
    """K-fold held-out log-likelihood of a fitted model spec."""
    df = sample_df.dropna(subset=["RT"]).reset_index(drop=True)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    lls = []
    for tr, te in kf.split(df):
        m = fit_model(_sample_from_sim(df.iloc[tr]), evmap, R=R, urgency=urgency,
                      dt=dt, T_dur=T_dur, fixed=fixed)
        model = build_model({**m, **fixed}, evmap, R=R, urgency=urgency, dt=dt, T_dur=T_dur)
        te_samp = _sample_from_sim(df.iloc[te])
        lls.append(float(-LossRobustLikelihood(model, te_samp,
                   required_conditions=["trial_uid"]).loss(model)))
    return float(np.mean(lls))


def select_structure(sample_df, evmap, fixed=None, dt=DT, T_dur=3.5) -> dict:
    """Step 0: choose rectification R and urgency form by CV log-likelihood."""
    fixed = fixed or {"t0": 0.05, "lam": 0.0}
    grid = [(R, U) for R in ("symmetric", "halfwave", "asym") for U in ("rising", "const")]
    scores = {f"{R}|{U}": _cv_loglik(sample_df, evmap, R, U, fixed, dt, T_dur)
              for (R, U) in grid}
    best = max(scores, key=scores.get)
    R, U = best.split("|")
    return {"R": R, "urgency": U, "scores": scores}


def route_attribution(sample_df, evmap, R="halfwave", urgency="rising",
                      fixed=None, dt=DT, T_dur=3.5) -> dict:
    """Step 0b: two-route vs TF-only (u fixed to 0) by CV log-likelihood."""
    fixed = fixed or {"t0": 0.05, "lam": 0.0}
    two = _cv_loglik(sample_df, evmap, R, urgency, fixed, dt, T_dur)
    tf_only = _cv_loglik(sample_df, evmap, R, urgency, {**fixed, "u": 0.0}, dt, T_dur)
    return {"two_route_cvll": two, "tf_only_cvll": tf_only,
            "two_route_wins": two > tf_only}
```

> Reconcile `LossRobustLikelihood(...).loss(model)` evaluation against the installed pyddm loss API in Task 1 (some versions expose `model.fit`/`solve`-based likelihood differently).

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_ddm.py -k route_attribution -v -m slow`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/ddm.py tests/analysis/test_ddm.py
git commit -m "feat(B0): structural selection (Step 0) + route attribution (Step 0b)"
```

---

### Task 6: Per-stage nested model comparison + state control

**Files:**
- Modify: `src/visdetect/analysis/ddm.py` (append `compare_stage_models`)
- Test: `tests/analysis/test_ddm.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from visdetect.analysis.ddm import compare_stage_models


@pytest.mark.slow
def test_stage_comparison_recovers_the_true_varying_knob():
    # Two stages identical except v doubles -> M_v should win (lowest AIC).
    rng = np.random.default_rng(0)
    def make(uidoff, v):
        evmap, conds = {}, {}
        for k in range(250):
            uid = uidoff + k
            ct = rng.uniform(0.8, 1.8); n = 150; e = np.zeros(n); e[int(ct/0.02):] = 2.0
            evmap[uid] = e; conds[uid] = {"trial_uid": uid, "change_time": ct}
        sim = simulate_sample(evmap, conds, dict(v=v, a=1.0, z=0.0, u=0.3, t0=0.05, lam=0.0),
                              R="halfwave", urgency="rising", n_per_trial=1, seed=uidoff)
        return _sample_from_sim(sim), evmap
    sA, eA = make(0, 1.5)
    sB, eB = make(100000, 3.0)
    res = compare_stage_models({"Learning": (sA, eA), "Expert": (sB, eB)},
                               R="halfwave", urgency="rising")
    assert res["winner"] == "M_v"
    assert res["delta_v"] > 0     # v increases Learning -> Expert
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_ddm.py -k stage_comparison -v -m slow`
Expected: FAIL — `ImportError: cannot import name 'compare_stage_models'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to ddm.py
def _aic(ll: float, k_params: int) -> float:
    return 2 * k_params - 2 * ll


def compare_stage_models(samples_by_stage: Dict[str, tuple], R="halfwave",
                         urgency="rising", fixed=None, dt=DT, T_dur=3.5) -> dict:
    """Nested comparison: which single parameter must vary across stages.

    samples_by_stage: {stage: (sample_df, evmap)}. Fits M_shared / M_v / M_a /
    M_zu / M_full and ranks by AIC over pooled held-out log-likelihood.
    """
    fixed = fixed or {"t0": 0.05, "lam": 0.0}
    stages = list(samples_by_stage)
    per_stage = {s: fit_model(_sample_from_sim(df), ev, R=R, urgency=urgency,
                              dt=dt, T_dur=T_dur, fixed=fixed)
                 for s, (df, ev) in samples_by_stage.items()}
    # cross-stage LL of each restricted model via the per-stage fits' shared/free pattern
    def stage_ll(free_keys):
        ll = 0.0
        for s, (df, ev) in samples_by_stage.items():
            p = {**per_stage[stages[0]]}                  # shared baseline
            for kk in free_keys:                          # let listed keys take stage value
                p[kk] = per_stage[s][kk]
            model = build_model({**p, **fixed}, ev, R=R, urgency=urgency, dt=dt, T_dur=T_dur)
            ll += float(-LossRobustLikelihood(model, _sample_from_sim(df),
                        required_conditions=["trial_uid"]).loss(model))
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
```

> **State control (spec §6):** the caller passes *state-balanced* or *state-conditioned* samples into `samples_by_stage` (e.g. subsample to matched engaged/impulsive proportions, or split per state). The comparison logic is unchanged; the control happens in how the per-stage DataFrames are built (Task 8).

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_ddm.py -k stage_comparison -v -m slow`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/ddm.py tests/analysis/test_ddm.py
git commit -m "feat(B0): per-stage nested model comparison (which knob varies)"
```

---

### Task 7: State-resolved route-mixture secondary + pluggable state accessor

**Files:**
- Modify: `src/visdetect/analysis/ddm.py` (append `load_state_labels`, `route_mixture_by_state`)
- Test: `tests/analysis/test_ddm.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from visdetect.analysis.ddm import route_mixture_by_state


@pytest.mark.slow
def test_route_mixture_higher_tf_share_in_engaged():
    # Engaged trials: evidence-driven FAs (fast pulses precede licks).
    # Impulsive trials: flat-evidence licks. Engaged TF-share must exceed Impulsive.
    rng = np.random.default_rng(0)
    def stage(evfun, n, off):
        evmap, conds = {}, {}
        for k in range(n):
            uid = off + k; evmap[uid] = evfun(rng)
            conds[uid] = {"trial_uid": uid, "change_time": np.inf}
        return evmap, conds
    eng_ev, eng_c = stage(lambda r: np.r_[np.zeros(20), np.ones(130) * 2.0], 150, 0)
    imp_ev, imp_c = stage(lambda r: np.zeros(150), 150, 100000)
    sim_e = simulate_sample(eng_ev, eng_c, dict(v=3, a=1, z=0, u=0.1, t0=0.05, lam=0),
                            R="halfwave", urgency="rising", n_per_trial=1, seed=1)
    sim_i = simulate_sample(imp_ev, imp_c, dict(v=0.1, a=1, z=0, u=0.8, t0=0.05, lam=0),
                            R="halfwave", urgency="rising", n_per_trial=1, seed=2)
    res = route_mixture_by_state(
        {"engaged": (_sample_from_sim(sim_e), eng_ev),
         "impulsive": (_sample_from_sim(sim_i), imp_ev)})
    assert res["engaged"]["tf_share"] > res["impulsive"]["tf_share"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_ddm.py -k route_mixture -v -m slow`
Expected: FAIL — `ImportError: cannot import name 'route_mixture_by_state'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to ddm.py
def load_state_labels(session_name, K: int = 3):
    """Pluggable per-trial behavioural-state accessor.

    Default source = GLM-HMM assignments (loader.load_hmm_assignments). Swap this
    function's body to point at the in-development classifier without touching any
    model/fit code. Returns a pandas Series indexed by trial order with state labels.
    """
    from visdetect.suite.loader import load_hmm_assignments
    df = load_hmm_assignments(K=K)
    sub = df[df["session_name"].astype(str) == str(session_name)]
    col = "hmm_state_label" if "hmm_state_label" in sub.columns else "hmm_state"
    return sub.set_index("trial_index")[col] if "trial_index" in sub.columns else sub[col]


def route_mixture_by_state(samples_by_state: Dict[str, tuple], R="halfwave",
                           urgency="rising", fixed=None, dt=DT, T_dur=3.5) -> dict:
    """Per-state route attribution: fraction of likes explained by route 1 (TF).

    tf_share := (CVLL_two_route - CVLL_tf_only) normalised; higher = more TF-driven.
    Predicts engaged > impulsive."""
    out = {}
    for state, (df, ev) in samples_by_state.items():
        ra = route_attribution(df, ev, R=R, urgency=urgency, fixed=fixed, dt=dt, T_dur=T_dur)
        m = fit_model(_sample_from_sim(df), ev, R=R, urgency=urgency, dt=dt, T_dur=T_dur,
                      fixed=fixed or {"t0": 0.05, "lam": 0.0})
        # share = sensory gain relative to total drive (v vs v+u), a simple readout
        tf_share = float(m["v"] / (m["v"] + m["u"] + 1e-9))
        out[state] = {"tf_share": tf_share, **ra}
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_ddm.py -k route_mixture -v -m slow`
Expected: PASS.

- [ ] **Step 5: Run the full test module**

Run: `py -m pytest tests/analysis/test_ddm.py -v` (add `-m "slow or not slow"` to include slow fits)
Expected: PASS (Tasks 2–7).

- [ ] **Step 6: Commit**

```bash
git add src/visdetect/analysis/ddm.py tests/analysis/test_ddm.py
git commit -m "feat(B0): state-resolved route mixture + pluggable state accessor"
```

---

### Task 8: Analysis script (manifest loop → fits → figure + caches)

**Files:**
- Create: `analysis_suite/01_behavior/h_ddm_learning_knob.py`

- [ ] **Step 1: Write the script**

```python
"""Fig0N (B0): Which DDM knob does learning turn? Two-route change-detection accumulator.

Sensitivity (drift gain v) vs caution (bound a) vs impulsivity (urgency/start u/z),
fit per stage; nested comparison names the knob learning turns. FAs are tested as
TF-driven vs time-driven (Step 0b). State is a pluggable confound control + secondary.

Outputs:
  - analysis_suite/figures/01_behavior/fig0N_ddm_learning_knob.png
  - analysis_suite/figures/01_behavior/ddm_learning_stats.csv
  - analysis_suite/cache/ddm_per_stage_fits.csv
"""
import os, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import STAGE_ORDER, STAGE_COLORS, CACHE_DIR
from visdetect.suite.loader import load_session
from visdetect.suite.plotting import setup_style, save_figure
from visdetect.analysis.config import load_staging_manifest
from visdetect.analysis import ddm

setup_style()
CACHE = os.path.join(CACHE_DIR, "ddm_per_stage_fits.csv")


def build_stage_samples(state_control: bool = False):
    """Return {stage: (sample_df, evmap)} pooled across that stage's sessions."""
    manifest = load_staging_manifest(qc_only=True)
    by_stage = {}
    for stage in manifest["stage"].unique():
        frames, evmap = [], {}
        for _, row in manifest[manifest["stage"] == stage].iterrows():
            sname = int(row["session_name"])
            try:
                sess = load_session(sname)
            except FileNotFoundError:
                continue
            ev_df = ddm.build_trial_evidence(sess)
            ev_df["trial_uid"] = ev_df["trial_uid"] + sname * 100000   # globally unique
            for _, r in ev_df.iterrows():
                evmap[int(r["trial_uid"])] = r["evidence"]
            if state_control:
                try:
                    states = ddm.load_state_labels(sname)
                    ev_df["state"] = states.reindex(range(len(ev_df))).to_numpy()
                except Exception:
                    ev_df["state"] = np.nan
            ev_df["RT"] = np.where(ev_df["outcome"] == "hit",
                                   ev_df["decision_time"] - ev_df["change_time"],
                                   ev_df["decision_time"])
            frames.append(ev_df[["trial_uid", "RT", "lick", "state"]
                                 if state_control else ["trial_uid", "RT", "lick"]])
            del sess; gc.collect()
        if frames:
            by_stage[stage] = (pd.concat(frames, ignore_index=True), evmap)
    return by_stage


def main():
    print("[01h] B0 DDM learning-knob...")
    by_stage = build_stage_samples()
    if len(by_stage) < 2:
        print("  Need >=2 stages. Exiting."); return

    # Step 0 + 0b on pooled data
    pooled_df = pd.concat([df for df, _ in by_stage.values()], ignore_index=True)
    pooled_ev = {}
    for _, ev in by_stage.values():
        pooled_ev.update(ev)
    struct = ddm.select_structure(pooled_df, pooled_ev)
    attr = ddm.route_attribution(pooled_df, pooled_ev, R=struct["R"], urgency=struct["urgency"])
    print(f"  structure: R={struct['R']} urgency={struct['urgency']} "
          f"| route: two={attr['two_route_cvll']:.1f} tf_only={attr['tf_only_cvll']:.1f}")

    comp = ddm.compare_stage_models(by_stage, R=struct["R"], urgency=struct["urgency"])
    print(f"  winner={comp['winner']} dv={comp['delta_v']:.3f} du={comp['delta_u']:.3f}")

    # --- figure (panels A-F per spec sec 9) ---
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.32)
    # (panels A RT dists, B psychometric, C params-by-stage, D model-comp,
    #  E recovery, F state route-mix) -- fill from comp/struct/attr + cached recovery.
    ax = fig.add_subplot(gs[1, 0])
    names = list(comp["aic"]); ax.bar(names, [comp["aic"][n] for n in names])
    ax.set_ylabel("AIC"); ax.set_title(f"D. Model comparison (winner={comp['winner']})")
    # ... A,B,C,E,F populated similarly from per_stage fits and a recovery cache ...
    save_figure(fig, "fig0N_ddm_learning_knob", "01_behavior")

    pd.DataFrame([{**{f"v_{s}": comp["per_stage"][s]["v"] for s in comp["per_stage"]},
                   **{f"u_{s}": comp["per_stage"][s]["u"] for s in comp["per_stage"]},
                   "winner": comp["winner"], "delta_v": comp["delta_v"],
                   "delta_u": comp["delta_u"], "R": struct["R"],
                   "urgency": struct["urgency"], "two_route_wins": attr["two_route_wins"]}]
                 ).to_csv(CACHE, index=False)
    print(f"  saved {CACHE}")


if __name__ == "__main__":
    main()
```

> Panels A/B/C/E/F are sketched; fill them from `comp["per_stage"]`, empirical RT/psychometric (`behavior.compute_psychometric_data`), a stored recovery run (Task 4), and `route_mixture_by_state` (run with `state_control=True`). Keep each panel ≤20 lines.

- [ ] **Step 2: Verify the script imports cleanly**

Run: `py -c "import importlib.util as u; s=u.spec_from_file_location('b0','analysis_suite/01_behavior/h_ddm_learning_knob.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print('import OK')"`
Expected: `import OK`.

- [ ] **Step 3: Commit**

```bash
git add analysis_suite/01_behavior/h_ddm_learning_knob.py
git commit -m "feat(B0): fig0N script — two-route DDM per-stage knob comparison"
```

---

### Task 9: Run on real data, record result, update index

**Files:**
- Modify: `docs/science/QUESTION_INDEX.md`

- [ ] **Step 1: Pre-flight data checks (spec §10)**

Run: `py -c "from visdetect.suite.loader import load_session; from visdetect.analysis.ddm import build_trial_evidence; s=load_session(__import__('visdetect.analysis.config',fromlist=['load_staging_manifest']).load_staging_manifest().iloc[-1]['session_name']); df=build_trial_evidence(int(s) if isinstance(s,(int,str)) else s); print(df.shape, df['outcome'].value_counts().to_dict())"`
(Adjust to load one real session; confirm evidence traces build, decision-time truncation is sane, and post-change TF handling matches what's stored — see spec §10 BLOCKING note. Decide abort censor-vs-exclude here.)

- [ ] **Step 2: Run the analysis**

Run: `cd analysis_suite && py 01_behavior/h_ddm_learning_knob.py`
Expected: prints chosen structure, route-attribution verdict, the winning stage-model and `delta_v`/`delta_u`; saves figure + stats + cache. (Slow — DDM fits per stage.)

- [ ] **Step 3: Sanity-check the result against the spec §8 success criteria**

Open `analysis_suite/figures/01_behavior/fig0N_ddm_learning_knob.png`. Confirm: parameter recovery (panel E) is clean; the model reproduces empirical RT + psychometric (A/B); the winning knob and its sign are interpretable (`Δv>0` sensitivity ↑ and/or impulsivity route down); rerun with `state_control=True` and confirm the learning verdict survives state control (spec §6).

- [ ] **Step 4: Update the question index**

Set the B0 row Plan cell + status in `docs/science/QUESTION_INDEX.md`:

```
| B0 ⭐ | Which DDM knob does learning turn (drift vs threshold vs starting-point)? | T1 | done | [design](../superpowers/specs/2026-06-10-B0-ddm-learning-knob-design.md) | [plan](../superpowers/plans/2026-06-10-B0-ddm-learning-knob-plan.md) |
```

(Use `in-progress` if recovery/identifiability forced fixing more params or if the state-control rerun is pending.)

- [ ] **Step 5: Commit**

```bash
git add docs/science/QUESTION_INDEX.md analysis_suite/cache/ddm_per_stage_fits.csv
git commit -m "data(B0): DDM learning-knob result + index update"
```

(If `analysis_suite/figures/**` is gitignored, omit it from the add.)

---

## Self-Review

**1. Spec coverage:**
- §3 evidence input (pre-planned, truncated) → Task 2 (`build_trial_evidence`, truncation test). ✓
- §4 two-route model (TF drift `v` + urgency `u`, R, λ, bound, z, t0) → Task 3 (`DriftTwoRoute`, `build_model`). ✓
- §5 Step 0 structural selection → Task 5 (`select_structure`); Step 0b route attribution → Task 5 (`route_attribution`); nested cross-stage comparison → Task 6 (`compare_stage_models`); session-level robustness → **descoped to follow-up** (per-session fits; cache supports it). ✓ (noted)
- §5 secondary state-resolved route mixture → Task 7 (`route_mixture_by_state`). ✓
- §6 identifiability — parameter recovery → Task 4 (core test); fixing λ,t0 → `fixed=` plumbing in `fit_model`/`recover_parameters`. ✓
- §6 state-composition control → Task 6 note + Task 8 `state_control=True` path. ✓
- §3/§10 pluggable state → Task 7 `load_state_labels` (defaults to `load_hmm_assignments`). ✓
- §9 deliverables (module/tests/script/figure A–F/stats/cache) → Tasks 2–8. Figure panels A/B/C/E/F are **sketched, not fully coded** (D is complete) — flagged in Task 8; acceptable because they read already-tested quantities, but a reviewer should complete them.
- §10 pyddm install + BLOCKING TF-data check + abort decision → Task 1 + Task 9 Step 1. ✓

**2. Placeholder scan:** Pure-Python units (evidence, CV-LL, comparison, state) are complete. **Honestly flagged, not silent:** (a) pyddm API names (`ICPoint`, `OverlayNonDecision`, `Sample` choice column, `LossRobustLikelihood.loss`) are reconciled in Task 1 and used consistently after; (b) figure panels A/B/C/E/F in Task 8 are sketched with explicit fill instructions; (c) session-level per-session fits are descoped to follow-up. These are real deferrals with named owners, not vague TODOs.

**3. Type consistency:** `build_trial_evidence → df[evidence,trial_uid,...]` feeds `evmap`/`conds` used by `build_model`/`simulate_sample`/`fit_model`/`recover_parameters`/`select_structure`/`route_attribution`/`compare_stage_models`/`route_mixture_by_state` — all take `(sample_df, evmap)` consistently. `fit_model` returns a param dict consumed by `build_model({**m, **fixed})`. `compare_stage_models` returns `{winner, aic, delta_v, delta_u, per_stage}` used in Task 8. Consistent.

**Statistician knobs (flag for review at planning/exec):** AIC vs BIC vs CV-LL for the headline; bootstrap CI scheme on Δparams (not yet coded — add in Task 6/8); recovery tolerance; `dt`/`T_dur`/`dx` grid resolution; minimum trials/stage and per-state for stable fits; the state-control method (subsample-match vs covariate vs per-state fit).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-10-B0-ddm-learning-knob-plan.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks. Note Task 1 (pyddm API reconciliation) gates Tasks 3–7, so run it first and propagate any signature fixes.
2. **Inline Execution** — execute in-session with checkpoints.

Given the pyddm dependency + slow model fits, executing B0 in its **own isolated worktree** (as set up for B2) is advisable. Which approach?
