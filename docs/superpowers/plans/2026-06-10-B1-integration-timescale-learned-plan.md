# B1 — Integration timescale is learned: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estimate BG_046's behavioural evidence-integration timescale `τ` **per learning stage** by **two independent estimators** (a whitened lick-triggered TF kernel ≈ Khilkevich's method, and a distributed-lag logistic-hazard filter ≈ Orsolic's method), and test whether `τ` **grows** Naive→Expert — triangulated across estimators and robust to a stimulus-autocorrelation correction and a behavioural-state control — per `docs/superpowers/specs/2026-06-10-B1-integration-timescale-learned-design.md`.

**Architecture:** A new library module `visdetect/analysis/integration_timescale.py` holds: a pyddm-free per-trial evidence extractor (`e(t)=log2(TF(t)/TF_base)` truncated at the lick); lick-set selection with non-decision-time (`t0`) truncation and engaged-FA conditioning (state is a **pluggable** input); the stimulus-autocovariance + whitening that corrects the kernel (B1's load-bearing confound); Estimator 1 (`lick_triggered_kernel` → `kernel_tau`); Estimator 2 (`fit_lagged_hazard` → `glm_filter_tau`); and `triangulate_tau` (per-stage τ for both estimators + bootstrap `Δτ`). A thin `analysis_suite/01_behavior/i_integration_timescale.py` orchestrates over the staging manifest and renders the figure + caches.

**Tech Stack:** numpy, pandas, scipy (`scipy.linalg.toeplitz`, `scipy.optimize.curve_fit`), scikit-learn (`LogisticRegression`), matplotlib — **all already present; NO new dependency, NO pyddm**. Reuses `visdetect.analysis.behavior.get_trial_dataframe` (RT/outcome semantics), `visdetect.analysis.constants`, `visdetect.suite.loader.load_hmm_assignments`, `visdetect.analysis.config.load_staging_manifest`.

---

> ## Planning-time resolutions (read once — they change the spec)
>
> Four things were settled while writing this plan; downstream tasks assume them:
>
> 1. **TF update period = 50 ms, NOT `TF_SAMPLE_PERIOD`.** The user confirmed the grating's TF fluctuations/pulses are presented at **50 ms**. `constants.TF_SAMPLE_PERIOD = 0.25` ("4 Hz base") conflates the 4 Hz *base grating temporal frequency* with the *sample period* and is **wrong as a sampling interval** — do not import it for the grid. The kernel/evidence grid uses **`dt = 0.05`**. The per-trial raw `baseline_values` resolution is inferred per trial as `change_time / len(baseline_values)` (robust to whatever the stored resolution is); **Task 1 confirms** this lands near 0.05 s (or a clean divisor like ~0.0167 s with the GLM's stride-3 convention) and characterises the stimulus autocorrelation. (See memory `tf_fluctuation_50ms_vs_constant`.) **Do NOT edit `constants.py`** in this work — a parallel chat owns it; only flag the constant.
> 2. **The existing lick-hazard GLM has no slow-exp + fast-derivative filter.** `analysis_suite/07_advanced/k_lick_hazard_glm.py` carries only an **instantaneous** `log2_tf` term (plus `post_change`, `change_evidence`, spline time basis, `stage×time`). It does **not** expose a multi-lag stimulus filter to read `τ` from. So spec §3/§4's "reuse the fitted GLM filter / cache" is **superseded**: **Estimator 2 is implemented here as a dedicated, focused distributed-lag logistic-hazard regression** — the Orsolic-style multi-lag filter the spec *intends*. It is regression-based and therefore **inherently autocorrelation-corrected** (the spec's requirement), and it shares B1's evidence representation rather than the GLM's params. (It reuses the GLM's *conventions* — 50 ms bins, at-risk discrete-time hazard, abort/ref exclusion, `_get_observation_window` lick-time semantics — not its fitted model.)
> 3. **Evidence extractor is duplicated from B0, deliberately.** B0's `visdetect.analysis.ddm.build_trial_evidence` does the same `log2(TF/base)`-truncated-at-decision job, but importing `ddm` pulls in **pyddm** (a heavy, not-yet-installed dep). B1 must ship without pyddm, so it re-implements a minimal `build_evidence_traces`. **When B0's Estimator 3 (`τ = 1/λ`) is added post-execution, reconcile/share the two extractors** (factor the pyddm-free core out of `ddm.py`). Tracked as the E3 follow-up; not done here.
> 4. **Estimator 3 (DDM leak `λ`) is deferred.** Per spec §4/§9, B1 ships on Estimators 1+2 only. No pyddm, no B0 dependency.

---

## File Structure

- **Create** `src/visdetect/analysis/integration_timescale.py` — all B1 computation: `build_evidence_traces`, `load_state_labels`, `collect_lick_segments`, `stimulus_autocov`/`whiten_kernel`, `lick_triggered_kernel`, `kernel_tau` (E1), `build_lagged_hazard_design`/`fit_lagged_hazard`/`glm_filter_tau` (E2), `triangulate_tau`.
- **Create** `tests/analysis/test_integration_timescale.py` — TDD: evidence truncation, lick-set + `t0` truncation, **whitening removes a planted autocorrelation artifact** (the load-bearing test), `τ`-recovery from a simulated leaky integrator (both estimators), `Δτ` bootstrap detects a planted learning increase.
- **Create** `analysis_suite/01_behavior/i_integration_timescale.py` — orchestration + figure (`fig0N`, panels A–F) + stats/cache.
- **Create** `docs/science/B1_stimulus_characterization.md` — Task 1's recorded answer to the BLOCKING stimulus question (resolution + ACF + `t0`).
- **Modify** `docs/science/QUESTION_INDEX.md` — link the plan, bump B1 status.

Conventions (`CLAUDE.md`): constants from `visdetect.analysis.constants`; `load_staging_manifest()`; `setup_style()`/`save_figure()`; `del sess; gc.collect()`; `py` not `python`.

---

### Task 1: Stimulus characterization — resolve the BLOCKING dt / autocorrelation / t0 (spec §9)

Everything downstream (grid `dt`, the whitening matrix, the `t0` truncation) depends on the real stimulus statistics. This task **measures them on real BG_046 sessions and records the answer**; it is a diagnostic, not TDD.

**Files:**
- Create: `scripts/analysis/behavior/characterize_tf_stream.py`
- Create: `docs/science/B1_stimulus_characterization.md`

- [ ] **Step 1: Write the diagnostic script**

```python
# scripts/analysis/behavior/characterize_tf_stream.py
"""B1 Task 1 — characterise the baseline TF stream: sample period, autocorrelation, t0.

Answers the spec §9 BLOCKING items before any kernel is computed:
  - what is the real per-sample duration of trial.baseline_values?  (expect ~50 ms,
    NOT constants.TF_SAMPLE_PERIOD=0.25 — see memory tf_fluctuation_50ms_vs_constant)
  - is e(t)=log2(TF/base) iid-white at 50 ms, or autocorrelated? (sets whitening)
  - the non-decision time t0 floor (ref/reflex + fastest-FA latency).
"""
import os, sys, gc
import numpy as np
import pandas as pd

from visdetect.analysis.config import load_staging_manifest
from visdetect.suite.loader import load_session

DT = 0.05


def main(n_sessions=4):
    manifest = load_staging_manifest(qc_only=True)
    periods, acfs, ref_lat, fast_fa = [], [], [], []
    for _, row in manifest.head(n_sessions).iterrows():
        sname = int(row["session_name"])
        try:
            sess = load_session(sname)
        except FileNotFoundError:
            continue
        for t in sess.trials:
            oc = (t.trialoutcome or "").lower()
            bv = getattr(t, "baseline_values", None)
            ct = getattr(t, "change_time", None)
            if bv is not None and ct:
                bv = np.asarray(bv).ravel()
                if bv.size > 5 and ct > 0:
                    periods.append(float(ct) / bv.size)      # inferred per-sample dt
                    e = np.log2(np.clip(bv, 1e-6, None) / np.median(bv))
                    e = e - e.mean()
                    n = min(40, e.size // 2)
                    if n > 2 and e.std() > 0:
                        ac = np.array([np.corrcoef(e[:e.size-l], e[l:])[0, 1]
                                       for l in range(n)])
                        acfs.append(ac)
            rts = getattr(t, "reactiontimes", {}) or {}
            if oc == "ref":
                v = rts.get("RT", rts.get("FA"))
                if v: ref_lat.append(float(v))
            if oc == "fa":
                v = rts.get("FA")
                if v: fast_fa.append(float(v))
        del sess; gc.collect()

    periods = np.array(periods)
    print(f"baseline sample period: median={np.median(periods)*1000:.1f} ms "
          f"[{np.percentile(periods,5)*1000:.1f}, {np.percentile(periods,95)*1000:.1f}]")
    if acfs:
        L = min(len(a) for a in acfs)
        mean_acf = np.mean([a[:L] for a in acfs], axis=0)
        # lag-1 autocorr on the 50 ms grid is the key number: ~0 => white => no whitening
        print(f"mean ACF lag1={mean_acf[1]:.3f} lag2={mean_acf[2]:.3f} "
              f"lag5={mean_acf[5] if L>5 else float('nan'):.3f}")
    ref_lat = np.array(ref_lat); fast_fa = np.array(fast_fa)
    t0 = np.nanpercentile(np.r_[ref_lat, fast_fa[fast_fa < 0.4]], 5) if ref_lat.size or fast_fa.size else 0.1
    print(f"t0 estimate (5th pct of reflex/fast-FA latencies): {t0*1000:.0f} ms")
    return periods, (mean_acf if acfs else None), t0


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it on real data**

Run: `py scripts/analysis/behavior/characterize_tf_stream.py`
Expected: prints a median sample period (confirm it is ~50 ms or a clean sub-multiple, **not** 250 ms), the ACF lag-1/2/5 values, and a `t0` estimate (~50–150 ms).

- [ ] **Step 3: Record the answer**

Write `docs/science/B1_stimulus_characterization.md` capturing: the measured sample period; **whether `e(t)` is white** at the 50 ms grid (`|ACF(lag≥1)| < ~0.1` ⇒ white ⇒ the whitening in Task 4 is near-identity, but still applied for honesty) **or autocorrelated** (⇒ whitening is load-bearing); the chosen `dt` (0.05) and `t0`. These three numbers are the inputs the Task 8 script passes down.

- [ ] **Step 4: Commit**

```bash
git add scripts/analysis/behavior/characterize_tf_stream.py docs/science/B1_stimulus_characterization.md
git commit -m "diag(B1): characterise TF stream (sample period, autocorrelation, t0)"
```

---

### Task 2: Per-trial evidence extraction (pyddm-free; truncated at the lick)

**Files:**
- Create: `src/visdetect/analysis/integration_timescale.py`
- Test: `tests/analysis/test_integration_timescale.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_integration_timescale.py
import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from visdetect.analysis.integration_timescale import build_evidence_traces, DT


def _toy_session():
    # Trial 0: FA (early lick at 0.5 s), pure baseline TF=2 (e=log2(2/2)=0 vs base 2).
    # Trial 1: Hit (change at 1.0 s, RT 0.3 -> lick 1.3 s); base TF=2, change_size=4.
    base = np.full(40, 2.0)                                  # 40 samples * ~50 ms = 2 s
    t_fa = SimpleNamespace(trialoutcome="FA", change_size=1.0, change_time=2.0,
                           reactiontimes={"FA": 0.5}, baseline_values=base, n_seen=None)
    t_hit = SimpleNamespace(trialoutcome="Hit", change_size=4.0, change_time=1.0,
                            reactiontimes={"RT": 0.3}, baseline_values=base, n_seen=None)
    return SimpleNamespace(trials=[t_fa, t_hit])


def test_build_evidence_truncates_at_lick_and_logs_ratio():
    df = build_evidence_traces(_toy_session(), dt=DT, tf_base=2.0)
    assert len(df) == 2
    fa = df[df.outcome == "fa"].iloc[0]
    hit = df[df.outcome == "hit"].iloc[0]
    # FA lick at 0.5 s -> evidence length round(0.5/DT)
    assert fa["lick_time"] == pytest.approx(0.5, abs=DT)
    assert len(fa["evidence"]) == pytest.approx(0.5 / DT, abs=1)
    # baseline TF==base -> e==0 in the baseline
    assert np.allclose(fa["evidence"], 0.0, atol=1e-6)
    # Hit lick at change_time+RT = 1.3 s; post-change e = log2(4*2/2)=2 after 1.0 s
    assert hit["lick_time"] == pytest.approx(1.3, abs=DT)
    post = hit["evidence"][int(1.0 / DT) + 1:]
    assert np.allclose(post, 2.0, atol=1e-6)
    assert abs(hit["evidence"][0]) < 1e-6                    # pre-change baseline
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_integration_timescale.py::test_build_evidence_truncates_at_lick_and_logs_ratio -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'visdetect.analysis.integration_timescale'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/visdetect/analysis/integration_timescale.py
"""B1 — is the evidence-integration timescale a learned quantity?

Two behavioural estimators of tau, per learning stage:
  E1: whitened lick-triggered TF kernel (reverse correlation) -> kernel_tau   (~Khilkevich)
  E2: distributed-lag logistic-hazard filter                  -> glm_filter_tau (~Orsolic)

NOTE (planning-time resolutions): TF update period = 50 ms (NOT constants.TF_SAMPLE_PERIOD
=0.25); the existing lick-hazard GLM has no multi-lag filter so E2 is built here; the
evidence extractor is duplicated pyddm-free from B0's ddm.build_trial_evidence and should
be reconciled when B0's E3 (tau=1/lambda) lands. See the B1 plan's resolutions box.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

DT = 0.05                       # integration grid (s) = confirmed 50 ms TF update period
RESPONSE_WINDOW_S = 2.155       # miss response-window end (mirrors lick-hazard GLM)


def _lick_time(trial) -> float:
    """Lick time aligned to Baseline_ON (mirrors GLM _get_observation_window). NaN if none."""
    oc = (getattr(trial, "trialoutcome", "") or "").lower()
    rts = getattr(trial, "reactiontimes", {}) or {}
    ct = getattr(trial, "change_time", None)
    if oc == "fa":
        v = rts.get("FA", rts.get("fa"))
        return float(v) if v else np.nan
    if oc == "hit":
        v = rts.get("RT", rts.get("Hit", rts.get("hit")))
        return (float(ct) + float(v)) if (v and ct) else np.nan
    return np.nan               # miss/abort/ref -> no usable lick


def build_evidence_traces(session, dt: float = DT, tf_base: Optional[float] = None,
                          sample_period: float = DT) -> pd.DataFrame:
    """Per-trial evidence e(t)=log2(TF(t)/TF_base) on a dt grid, truncated at the lick.

    One row per usable (hit/fa) trial: trial_idx, outcome, change_size, change_time,
    lick_time, evidence (np.ndarray on [0, lick] grid). TF stream is the pre-planned
    design; post-change samples are the planned change_size-shifted baseline.
    """
    rows = []
    for tidx, t in enumerate(getattr(session, "trials", []) or []):
        oc = (getattr(t, "trialoutcome", "") or "").lower()
        if oc not in ("hit", "fa"):
            continue                                          # only lick trials carry a kernel
        bv = getattr(t, "baseline_values", None)
        if bv is None:
            continue
        bv = np.asarray(bv, dtype=float).ravel()
        n_seen = getattr(t, "n_seen", None)
        if isinstance(n_seen, (int, np.integer)) and n_seen and n_seen > 0:
            bv = bv[: int(n_seen)]
        if bv.size == 0:
            continue
        ct = float(getattr(t, "change_time", np.nan) or np.nan)
        cs = float(getattr(t, "change_size", np.nan) or 1.0)
        lick = _lick_time(t)
        if not np.isfinite(lick) or lick <= 0:
            continue
        base = float(tf_base) if tf_base is not None else (float(np.nanmedian(bv)) or 1.0)
        # raw baseline resolution inferred per trial (robust to stored stride); fallback 50 ms
        bperiod = (ct / bv.size) if (np.isfinite(ct) and ct > 0) else sample_period
        n = max(1, int(round(lick / dt)))
        e = np.zeros(n)
        for i in range(n):
            tau_t = i * dt
            j = min(bv.size - 1, int(tau_t / bperiod))
            tf = bv[j]
            if np.isfinite(ct) and tau_t >= ct and cs > 1.0:
                tf = tf * cs                                  # planned post-change shift
            e[i] = np.log2(max(tf, 1e-6) / base)
        rows.append({"trial_idx": tidx, "outcome": oc, "change_size": cs,
                     "change_time": ct, "lick_time": lick, "evidence": e})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_integration_timescale.py::test_build_evidence_truncates_at_lick_and_logs_ratio -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/integration_timescale.py tests/analysis/test_integration_timescale.py
git commit -m "feat(B1): pyddm-free per-trial evidence extractor (truncated at lick)"
```

---

### Task 3: State accessor (pluggable) + lick-set selection with t0 truncation

**Files:**
- Modify: `src/visdetect/analysis/integration_timescale.py` (append `load_state_labels`, `collect_lick_segments`)
- Test: `tests/analysis/test_integration_timescale.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from visdetect.analysis.integration_timescale import collect_lick_segments


def _ev_df_two_fas():
    # Two FAs, lick at 1.0 s; evidence is a ramp e[i]=i so we can read which lags land.
    e = np.arange(20, dtype=float)                            # dt=0.05 -> 0..0.95 s
    return pd.DataFrame([
        {"trial_idx": 0, "outcome": "fa", "change_size": 1.0,
         "change_time": np.inf, "lick_time": 1.0, "evidence": e.copy()},
        {"trial_idx": 1, "outcome": "fa", "change_size": 1.0,
         "change_time": np.inf, "lick_time": 1.0, "evidence": e.copy()},
    ])


def test_lick_segments_truncate_t0_and_window_back():
    df = _ev_df_two_fas()
    segs, lags, info = collect_lick_segments(df, "all_fa", t0=0.10, max_lag=0.30, dt=0.05)
    assert segs.shape == (2, 6)                               # 0.30/0.05 = 6 lags
    # causal end = lick - t0 = 1.0 - 0.10 = 0.90 s -> index 18 (e[18]=18) is lag 1
    assert segs[0, 0] == pytest.approx(18.0)                  # nearest lag to (end - dt)
    assert segs[0, -1] < segs[0, 0]                           # further lags = earlier (smaller)


def test_engaged_fa_filter_uses_state_labels():
    df = _ev_df_two_fas()
    state = pd.Series({0: "engaged", 1: "impulsive"})         # only trial 0 is engaged
    segs, lags, info = collect_lick_segments(df, "engaged_fa", t0=0.10, max_lag=0.30,
                                             dt=0.05, state_labels=state)
    assert segs.shape[0] == 1 and info.iloc[0]["trial_idx"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_integration_timescale.py -k "lick_segments or engaged_fa" -v`
Expected: FAIL — `ImportError: cannot import name 'collect_lick_segments'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to integration_timescale.py
def load_state_labels(session_name, K: int = 3) -> pd.Series:
    """Pluggable per-trial behavioural-state accessor, indexed by trial_idx.

    Default source = GLM-HMM assignments (loader.load_hmm_assignments). Values mapped to
    {'engaged','impulsive','other'} by the renamed HMM labels (mirrors the lick-hazard
    GLM's _find_hmm_state_columns). SWAP THIS FUNCTION'S BODY to point at the
    in-development self-tailored classifier without touching any kernel/fit code.
    """
    from visdetect.suite.loader import load_hmm_assignments
    df = load_hmm_assignments(K=K)
    sub = df[df["session_name"].astype(int) == int(session_name)]
    if sub.empty:
        return pd.Series(dtype=object)

    def _map(lbl):
        s = str(lbl)
        if s in ("Engaged", "Engaged_2") or s.startswith("Engaged"):
            return "engaged"
        if s in ("Impulsive", "Biased") or s.startswith("Impulsive"):
            return "impulsive"
        return "other"

    return sub.set_index("trial_idx")["hmm_state_label"].map(_map)


def collect_lick_segments(ev_df, lick_set: str, t0: float, max_lag: float, dt: float = DT,
                          state_labels: Optional[pd.Series] = None,
                          sensory_latency: float = 0.05):
    """Stack the evidence window preceding each qualifying lick.

    Returns (segments [n_licks, n_lags], lags [n_lags], info_df). Lags run BACK from the
    causal end (lick - t0): stimulus within t0 of the lick is too late to have driven it
    (non-decision time, spec §4). For 'hit', lags whose absolute time precedes
    (change_time + sensory_latency) are masked NaN (post-change integration only).

    lick_set: 'engaged_fa' (PRIMARY; needs state_labels) | 'hit' | 'all_fa'.
    """
    n_lags = int(round(max_lag / dt))
    lags = (np.arange(1, n_lags + 1) * dt)
    segs, info = [], []
    for _, r in ev_df.iterrows():
        oc, lick, e = r["outcome"], r["lick_time"], np.asarray(r["evidence"], float)
        if not np.isfinite(lick):
            continue
        if lick_set in ("engaged_fa", "all_fa"):
            if oc != "fa":
                continue
            if lick_set == "engaged_fa" and state_labels is not None:
                if str(state_labels.get(r["trial_idx"], "other")) != "engaged":
                    continue
        elif lick_set == "hit":
            if oc != "hit":
                continue
        else:
            raise ValueError(f"unknown lick_set {lick_set!r}")
        end_t = lick - t0
        seg = np.full(n_lags, np.nan)
        for k in range(n_lags):
            t = end_t - lags[k]
            if t < 0:
                continue
            if oc == "hit" and np.isfinite(r["change_time"]) \
                    and t < (r["change_time"] + sensory_latency):
                continue                                       # pre-change: out of window
            i = int(round(t / dt))
            if 0 <= i < e.size:
                seg[k] = e[i]
        segs.append(seg)
        info.append({"trial_idx": r["trial_idx"], "outcome": oc})
    arr = np.asarray(segs, float) if segs else np.empty((0, n_lags))
    return arr, lags, pd.DataFrame(info)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -m pytest tests/analysis/test_integration_timescale.py -k "lick_segments or engaged_fa" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/integration_timescale.py tests/analysis/test_integration_timescale.py
git commit -m "feat(B1): pluggable state accessor + lick-set selection with t0 truncation"
```

---

### Task 4: Stimulus-autocovariance whitening — the load-bearing correction (spec §6)

**Files:**
- Modify: `src/visdetect/analysis/integration_timescale.py` (append `stimulus_autocov`, `whiten_kernel`, `lick_triggered_kernel`)
- Test: `tests/analysis/test_integration_timescale.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from scipy.linalg import toeplitz
from visdetect.analysis.integration_timescale import (
    whiten_kernel, lick_triggered_kernel, stimulus_autocov,
)


def test_whitening_inverts_a_planted_autocorrelation_artifact():
    # A correlated stimulus has autocovariance C (AR(1)-like). The raw reverse-correlation
    # kernel of a linear system is biased: k_raw = C @ k_true. Whitening must recover k_true.
    n_lags = 8
    rho = 0.6
    acf = rho ** np.arange(n_lags)                            # AR(1) autocorrelation
    C = toeplitz(acf)
    lags = (np.arange(1, n_lags + 1) * 0.05)
    k_true = np.exp(-lags / 0.20)                             # true filter, tau=0.2 s
    k_raw = C @ k_true                                        # what reverse-correlation returns
    # Raw kernel is biased away from the truth; whitening recovers it.
    assert not np.allclose(k_raw, k_true, atol=0.05)
    k_corr = whiten_kernel(k_raw, C, reg=0.0)
    assert np.allclose(k_corr, k_true, atol=1e-6)


def test_lick_triggered_kernel_averages_then_whitens():
    segs = np.array([[1.0, 0.5, 0.25], [1.0, 0.5, 0.25]])    # two identical segments
    C = np.eye(3)                                             # white stimulus -> identity
    k = lick_triggered_kernel(segs, autocov=C, correct_autocorr=True, reg=0.0)
    assert np.allclose(k, [1.0, 0.5, 0.25])                  # white => whitening is identity
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_integration_timescale.py -k "whitening or averages_then" -v`
Expected: FAIL — `ImportError: cannot import name 'whiten_kernel'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to integration_timescale.py
def stimulus_autocov(ev_df, n_lags: int, dt: float = DT, max_trials: Optional[int] = None):
    """Toeplitz stimulus autocovariance of baseline e(t) (pre-change, stationary segment)."""
    from scipy.linalg import toeplitz
    acov = np.zeros(n_lags)
    counts = np.zeros(n_lags)
    rows = ev_df if max_trials is None else ev_df.head(max_trials)
    for _, r in rows.iterrows():
        e = np.asarray(r["evidence"], float)
        ct = r["change_time"]
        if np.isfinite(ct):
            e = e[: int(ct / dt)]                             # baseline only
        if e.size <= 1:
            continue
        e = e - e.mean()
        for lag in range(n_lags):
            if e.size > lag:
                acov[lag] += float(np.dot(e[: e.size - lag], e[lag:]))
                counts[lag] += (e.size - lag)
    acov = acov / np.maximum(counts, 1.0)
    return toeplitz(acov)


def whiten_kernel(raw_kernel, autocov, reg: float = 1e-3):
    """Recover the true filter from a reverse-correlation kernel: k_true = C^{-1} k_raw.

    Ridge-regularised (reg as a fraction of mean diagonal) for ill-conditioned C.
    """
    C = np.asarray(autocov, float)
    if reg > 0:
        C = C + reg * (np.trace(C) / C.shape[0]) * np.eye(C.shape[0])
    return np.linalg.solve(C, np.asarray(raw_kernel, float))


def lick_triggered_kernel(segments, autocov=None, correct_autocorr: bool = True,
                          reg: float = 1e-3):
    """Mean evidence over lick-triggered segments, optionally whitened by the stimulus ACF."""
    if segments.shape[0] == 0:
        raise ValueError("no lick segments to average")
    raw = np.nanmean(segments, axis=0)
    if correct_autocorr and autocov is not None:
        return whiten_kernel(raw, autocov, reg=reg)
    return raw
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -m pytest tests/analysis/test_integration_timescale.py -k "whitening or averages_then" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/integration_timescale.py tests/analysis/test_integration_timescale.py
git commit -m "feat(B1): stimulus-autocovariance whitening (kernel autocorrelation correction)"
```

---

### Task 5: Estimator 1 — `kernel_tau` + leaky-integrator recovery

**Files:**
- Modify: `src/visdetect/analysis/integration_timescale.py` (append `kernel_tau`)
- Test: `tests/analysis/test_integration_timescale.py` (append a fast unit test + a slow recovery test + a shared simulator helper)

- [ ] **Step 1: Write the failing tests (+ the simulator the later tasks reuse)**

```python
from visdetect.analysis.integration_timescale import kernel_tau


def test_kernel_tau_recovers_a_clean_exponential():
    lags = (np.arange(1, 31) * 0.05)
    k = np.exp(-lags / 0.30)                                  # tau = 0.30 s
    assert kernel_tau(k, lags, method="exp") == pytest.approx(0.30, rel=0.1)
    # half-area is a coarser but monotone readout
    assert 0.1 < kernel_tau(k, lags, method="half_area") < 0.6


def simulate_leaky_integrator_fas(tau_true, n_trials, dt=0.05, T=3.0, thresh=1.0,
                                   gain=0.25, white=True, rho=0.0, seed=0):
    """Ground-truth generator: leaky integrator -> early-lick (FA) crossings on a
    baseline-only stimulus. Returns an ev_df identical in schema to build_evidence_traces.
    Reused by Tasks 5 and 7."""
    rng = np.random.default_rng(seed)
    n = int(T / dt)
    leak = dt / tau_true
    rows = []
    for uid in range(n_trials):
        stim = rng.standard_normal(n)
        if not white:                                        # AR(1) coloured stimulus
            for i in range(1, n):
                stim[i] = rho * stim[i - 1] + np.sqrt(1 - rho ** 2) * stim[i]
        x = 0.0
        lick_i = None
        for i in range(n):
            x += -leak * x + gain * stim[i]
            if x >= thresh:
                lick_i = i
                break
        if lick_i is None or lick_i < 5:
            continue
        rows.append({"trial_idx": uid, "outcome": "fa", "change_size": 1.0,
                     "change_time": np.inf, "lick_time": lick_i * dt,
                     "evidence": stim[: lick_i + 1].astype(float)})
    return pd.DataFrame(rows)


@pytest.mark.slow
def test_kernel_recovers_leaky_integrator_tau():
    df = simulate_leaky_integrator_fas(tau_true=0.30, n_trials=4000, white=True, seed=1)
    segs, lags, _ = collect_lick_segments(df, "all_fa", t0=0.0, max_lag=1.0, dt=0.05)
    C = stimulus_autocov(df, n_lags=len(lags), dt=0.05)       # ~identity (white)
    k = lick_triggered_kernel(segs, autocov=C, correct_autocorr=True)
    tau = kernel_tau(k, lags, method="exp")
    assert 0.15 < tau < 0.60                                  # recovers ~0.30 within tolerance
```

- [ ] **Step 2: Run to verify failure**

Run: `py -m pytest tests/analysis/test_integration_timescale.py -k "kernel_tau_recovers or leaky_integrator_tau" -v`
Expected: FAIL — `ImportError: cannot import name 'kernel_tau'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to integration_timescale.py
def kernel_tau(kernel, lags, method: str = "exp") -> float:
    """Effective integration timescale from a lag profile.

    'exp': fit k(lag)=A*exp(-lag/tau), return tau. 'half_area': lag at which the
    cumulative |kernel| reaches half its total (robust, parameterisation-free).
    """
    k = np.asarray(kernel, float)
    L = np.asarray(lags, float)
    m = np.isfinite(k) & np.isfinite(L)
    k, L = k[m], L[m]
    if k.size < 3:
        return float("nan")
    if method == "half_area":
        step = float(np.median(np.diff(L))) if L.size > 1 else float(L[0])
        area = np.cumsum(np.abs(k)) * step
        if area[-1] <= 0:
            return float("nan")
        return float(np.interp(0.5 * area[-1], area, L))
    from scipy.optimize import curve_fit

    def _exp(x, A, tau):
        return A * np.exp(-x / np.maximum(tau, 1e-3))

    try:
        p, _ = curve_fit(_exp, L, k, p0=[float(np.nanmax(np.abs(k))), float(np.median(L))],
                         maxfev=10000, bounds=([-np.inf, 1e-3], [np.inf, 10 * float(L.max())]))
        return float(p[1])
    except Exception:
        return float("nan")
```

- [ ] **Step 4: Run to verify pass**

Run: `py -m pytest tests/analysis/test_integration_timescale.py -k "kernel_tau_recovers or leaky_integrator_tau" -v -m "slow or not slow"`
Expected: PASS. If the slow recovery test is biased, that is a **finding** (per spec: τ is method-sensitive) — widen tolerance only after confirming the bias is the known finite-data / threshold-nonlinearity bias, not a bug.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/integration_timescale.py tests/analysis/test_integration_timescale.py
git commit -m "feat(B1): Estimator 1 kernel_tau + leaky-integrator recovery"
```

---

### Task 6: Estimator 2 — distributed-lag logistic-hazard filter (Orsolic analog)

**Files:**
- Modify: `src/visdetect/analysis/integration_timescale.py` (append `build_lagged_hazard_design`, `fit_lagged_hazard`, `glm_filter_tau`)
- Test: `tests/analysis/test_integration_timescale.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from visdetect.analysis.integration_timescale import fit_lagged_hazard, glm_filter_tau


@pytest.mark.slow
def test_lagged_hazard_recovers_tau_and_is_autocorr_robust():
    # COLOURED stimulus (rho=0.6): a raw reverse-correlation kernel would be biased, but the
    # regression-based Estimator 2 should still recover tau (it is autocorrelation-corrected).
    df = simulate_leaky_integrator_fas(tau_true=0.30, n_trials=5000, white=False, rho=0.6,
                                       seed=3)
    w = fit_lagged_hazard(df, "all_fa", n_lags=20, dt=0.05, t0=0.0)
    assert w is not None and w.shape[0] == 20
    tau = glm_filter_tau(w, dt=0.05, method="half_area")
    assert 0.10 < tau < 0.70                                  # recovers ~0.30 despite colour
```

- [ ] **Step 2: Run to verify failure**

Run: `py -m pytest tests/analysis/test_integration_timescale.py -k lagged_hazard_recovers -v -m slow`
Expected: FAIL — `ImportError: cannot import name 'fit_lagged_hazard'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to integration_timescale.py
def build_lagged_hazard_design(ev_df, lick_set: str, n_lags: int, dt: float = DT,
                               state_labels: Optional[pd.Series] = None, t0: float = 0.10,
                               sensory_latency: float = 0.05, max_time: float = 25.0):
    """At-risk discrete-time hazard rows with a lagged-evidence design (Orsolic-style).

    Each row = one at-risk dt-bin from n_lags..(lick - t0). X = [e(t), e(t-dt), ...,
    e(t-(n_lags-1)dt)]; y = 1 at the lick bin else 0; plus the bin time (nuisance).
    Mirrors the lick-hazard GLM's discrete-time hazard construction.
    """
    Xs, ys, ts = [], [], []
    for _, r in ev_df.iterrows():
        oc, lick, e = r["outcome"], r["lick_time"], np.asarray(r["evidence"], float)
        if not np.isfinite(lick):
            continue
        if lick_set in ("engaged_fa", "all_fa"):
            if oc != "fa":
                continue
            if lick_set == "engaged_fa" and state_labels is not None:
                if str(state_labels.get(r["trial_idx"], "other")) != "engaged":
                    continue
        elif lick_set == "hit":
            if oc != "hit":
                continue
        else:
            raise ValueError(lick_set)
        end_t = min(lick - t0, max_time)
        nb = int(end_t / dt)
        for b in range(n_lags, nb):
            t = b * dt
            if oc == "hit" and np.isfinite(r["change_time"]) \
                    and t < (r["change_time"] + sensory_latency):
                continue
            lagvec = e[b - n_lags + 1: b + 1][::-1]            # lag0 (current) .. lag n_lags-1
            if lagvec.size < n_lags:
                continue
            Xs.append(lagvec)
            ys.append(1.0 if b == nb - 1 else 0.0)
            ts.append(t)
    return (np.asarray(Xs, float), np.asarray(ys, float), np.asarray(ts, float))


def fit_lagged_hazard(ev_df, lick_set: str, n_lags: int, dt: float = DT,
                      state_labels: Optional[pd.Series] = None, t0: float = 0.10,
                      C: float = 1.0):
    """Fit the distributed-lag logistic hazard; return the n_lags stimulus-filter weights.

    Time-in-trial (linear+quadratic) is included as a nuisance to absorb the temporal/
    urgency clock, so the lag weights reflect stimulus drive. Regression on the full lag
    set is inherently autocorrelation-corrected (spec §6)."""
    X, y, tcol = build_lagged_hazard_design(ev_df, lick_set, n_lags, dt, state_labels, t0)
    if X.shape[0] < n_lags + 5 or y.sum() < 3:
        return None
    from sklearn.linear_model import LogisticRegression
    tn = (tcol - tcol.mean()) / (tcol.std() + 1e-9)
    Xn = np.column_stack([X, tn, tn ** 2])
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(Xn, y)
    return clf.coef_.ravel()[:n_lags]                         # the stimulus filter over lags


def glm_filter_tau(lag_weights, dt: float = DT, method: str = "exp") -> float:
    """Read tau from the fitted lag-weight filter (same readout as kernel_tau)."""
    lags = (np.arange(len(lag_weights)) + 1) * dt
    return kernel_tau(np.asarray(lag_weights, float), lags, method=method)
```

- [ ] **Step 4: Run to verify pass**

Run: `py -m pytest tests/analysis/test_integration_timescale.py -k lagged_hazard_recovers -v -m slow`
Expected: PASS (a real logistic fit; may take ~30 s).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/integration_timescale.py tests/analysis/test_integration_timescale.py
git commit -m "feat(B1): Estimator 2 distributed-lag logistic-hazard filter + glm_filter_tau"
```

---

### Task 7: `triangulate_tau` — per-stage τ for both estimators + bootstrap Δτ

**Files:**
- Modify: `src/visdetect/analysis/integration_timescale.py` (append `_stage_taus`, `triangulate_tau`)
- Test: `tests/analysis/test_integration_timescale.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from visdetect.analysis.integration_timescale import triangulate_tau


@pytest.mark.slow
def test_triangulate_detects_planted_learning_increase():
    # Learning: short tau (0.20). Expert: long tau (0.45). Delta_tau (kernel) must be > 0
    # with a bootstrap CI excluding 0, and the GLM estimator must agree in sign.
    learn = simulate_leaky_integrator_fas(tau_true=0.20, n_trials=2500, white=True, seed=10)
    expert = simulate_leaky_integrator_fas(tau_true=0.45, n_trials=2500, white=True, seed=11)
    res = triangulate_tau({"Learning": learn, "Expert": expert},
                          t0=0.0, dt=0.05, max_lag=1.2, n_boot=200, seed=0,
                          run_glm=True)
    assert res["per_stage"]["Expert"]["tau_kernel"] > res["per_stage"]["Learning"]["tau_kernel"]
    assert res["delta"]["tau_kernel"]["estimate"] > 0
    assert res["delta"]["tau_kernel"]["ci_low"] > 0          # CI excludes 0
    # sign agreement across estimators (the triangulation headline)
    assert np.sign(res["delta"]["tau_glm"]["estimate"]) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `py -m pytest tests/analysis/test_integration_timescale.py -k triangulate_detects -v -m slow`
Expected: FAIL — `ImportError: cannot import name 'triangulate_tau'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to integration_timescale.py
def _stage_taus(ev_df, t0, dt, max_lag, state_labels=None, lick_set="engaged_fa",
                run_glm=True):
    """Point estimates (tau_kernel, tau_glm, n_licks) for one stage."""
    n_lags = int(round(max_lag / dt))
    # fall back to all FAs if no state labels (so the function is usable pre-classifier)
    use_set = lick_set if (state_labels is not None or lick_set != "engaged_fa") else "all_fa"
    segs, lags, _ = collect_lick_segments(ev_df, use_set, t0, max_lag, dt,
                                          state_labels=state_labels)
    tau_k = np.nan
    if segs.shape[0] >= 5:
        C = stimulus_autocov(ev_df, n_lags, dt)
        tau_k = kernel_tau(lick_triggered_kernel(segs, autocov=C), lags, method="exp")
    tau_g = np.nan
    if run_glm:
        w = fit_lagged_hazard(ev_df, use_set, n_lags, dt, state_labels=state_labels, t0=t0)
        if w is not None:
            tau_g = glm_filter_tau(w, dt, method="exp")
    return {"tau_kernel": float(tau_k), "tau_glm": float(tau_g), "n_licks": int(segs.shape[0])}


def _bootstrap_delta(ev_lo, ev_hi, t0, dt, max_lag, state_lo, state_hi, n_boot, seed,
                     key, run_glm):
    """Bootstrap Δτ = τ(hi) − τ(lo) by resampling trials within each stage."""
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        bl = ev_lo.sample(len(ev_lo), replace=True, random_state=int(rng.integers(1 << 31)))
        bh = ev_hi.sample(len(ev_hi), replace=True, random_state=int(rng.integers(1 << 31)))
        tl = _stage_taus(bl, t0, dt, max_lag, state_lo, run_glm=run_glm)[key]
        th = _stage_taus(bh, t0, dt, max_lag, state_hi, run_glm=run_glm)[key]
        if np.isfinite(tl) and np.isfinite(th):
            deltas.append(th - tl)
    deltas = np.asarray(deltas, float)
    if deltas.size == 0:
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_boot": 0}
    return {"estimate": float(np.median(deltas)),
            "ci_low": float(np.percentile(deltas, 2.5)),
            "ci_high": float(np.percentile(deltas, 97.5)),
            "n_boot": int(deltas.size)}


def triangulate_tau(stage_data: Dict[str, pd.DataFrame], t0: float = 0.10, dt: float = DT,
                    max_lag: float = 1.5, n_boot: int = 1000, seed: int = 42,
                    state_by_stage: Optional[Dict[str, pd.Series]] = None,
                    run_glm: bool = True) -> dict:
    """Per-stage τ for Estimators 1 (kernel) & 2 (GLM) + bootstrap Δτ between the first
    and last stage (chronological order of the dict). state_by_stage is the pluggable
    state input (engaged-FA conditioning + the §6 state control)."""
    state_by_stage = state_by_stage or {}
    stages = list(stage_data)
    per_stage = {s: _stage_taus(stage_data[s], t0, dt, max_lag,
                                state_by_stage.get(s), run_glm=run_glm) for s in stages}
    out = {"per_stage": per_stage, "delta": {}}
    if len(stages) >= 2:
        lo, hi = stages[0], stages[-1]
        for key in ("tau_kernel", "tau_glm"):
            if key == "tau_glm" and not run_glm:
                continue
            out["delta"][key] = _bootstrap_delta(
                stage_data[lo], stage_data[hi], t0, dt, max_lag,
                state_by_stage.get(lo), state_by_stage.get(hi), n_boot, seed, key, run_glm)
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `py -m pytest tests/analysis/test_integration_timescale.py -k triangulate_detects -v -m slow`
Expected: PASS.

- [ ] **Step 5: Run the full module test suite**

Run: `py -m pytest tests/analysis/test_integration_timescale.py -v -m "slow or not slow"`
Expected: PASS (Tasks 2–7). Fast subset (`-m "not slow"`) must pass in seconds.

- [ ] **Step 6: Commit**

```bash
git add src/visdetect/analysis/integration_timescale.py tests/analysis/test_integration_timescale.py
git commit -m "feat(B1): triangulate_tau — per-stage tau (E1+E2) + bootstrap delta-tau"
```

---

### Task 8: Analysis script (manifest loop → per-stage τ → figure + caches)

**Files:**
- Create: `analysis_suite/01_behavior/i_integration_timescale.py`

- [ ] **Step 1: Write the script**

```python
"""Fig0N (B1): Is the evidence-integration timescale a learned quantity?

Two behavioural estimators of tau per stage — a whitened lick-triggered TF kernel
(~Khilkevich) and a distributed-lag logistic-hazard filter (~Orsolic) — tested for a
learning increase, triangulated, autocorrelation-corrected, and state-controlled.
The Estimator-1-vs-2 gap on the SAME data also adjudicates the 0.27 s vs ~1 s field
discrepancy (method artifact vs biology).

Outputs:
  - analysis_suite/figures/01_behavior/fig0N_integration_timescale.png
  - analysis_suite/figures/01_behavior/integration_timescale_stats.csv
  - analysis_suite/cache/integration_timescale_taus.csv
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
from visdetect.analysis.behavior import filter_manifest_by_stage
from visdetect.analysis import integration_timescale as it

setup_style()
CACHE = os.path.join(CACHE_DIR, "integration_timescale_taus.csv")

# From Task 1 (docs/science/B1_stimulus_characterization.md); override here if refined.
DT = 0.05
T0 = 0.10
MAX_LAG = 1.5
LIT_BRACKET = (0.27, 1.0)          # Khilkevich vs Orsolic


def build_stage_data(state_control=True):
    """Return ({stage: ev_df}, {stage: state_series}) pooled across each stage's sessions."""
    manifest = filter_manifest_by_stage(
        load_staging_manifest(qc_only=True),
        include_stages=["Naive", "Learning", "Expert"], merge_naive_learning=True)
    stage_data, state_by_stage = {}, {}
    for stage in [s for s in STAGE_ORDER if s in manifest["stage"].unique()]:
        frames, states = [], []
        for _, row in manifest[manifest["stage"] == stage].iterrows():
            sname = int(row["session_name"])
            try:
                sess = load_session(sname)
            except FileNotFoundError:
                continue
            ev = it.build_evidence_traces(sess, dt=DT)
            ev["trial_uid"] = ev["trial_idx"] + sname * 100000
            if state_control:
                try:
                    st = it.load_state_labels(sname)
                    st.index = st.index + sname * 100000      # align to trial_uid
                    states.append(st)
                except Exception:
                    pass
            ev = ev.rename(columns={"trial_idx": "_orig_idx", "trial_uid": "trial_idx"})
            frames.append(ev)
            del sess; gc.collect()
        if frames:
            stage_data[stage] = pd.concat(frames, ignore_index=True)
            if states:
                state_by_stage[stage] = pd.concat(states)
    return stage_data, state_by_stage


def main():
    print("[01i] B1 integration timescale...")
    stage_data, state_by_stage = build_stage_data(state_control=True)
    if len(stage_data) < 2:
        print("  Need >=2 stages. Exiting."); return

    res = it.triangulate_tau(stage_data, t0=T0, dt=DT, max_lag=MAX_LAG, n_boot=1000,
                             state_by_stage=state_by_stage, run_glm=True)
    for s, d in res["per_stage"].items():
        print(f"  {s}: tau_kernel={d['tau_kernel']:.3f}s tau_glm={d['tau_glm']:.3f}s "
              f"(n_licks={d['n_licks']})")
    for key, d in res["delta"].items():
        print(f"  Δ{key} = {d['estimate']:.3f}s [{d['ci_low']:.3f},{d['ci_high']:.3f}]")

    # ── figure (panels A–F per spec §8) ──────────────────────────────────
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.32)

    # Panel C (fully coded): tau-by-stage for both estimators + literature bracket
    axc = fig.add_subplot(gs[0, 2])
    stages = list(res["per_stage"])
    x = np.arange(len(stages))
    axc.plot(x, [res["per_stage"][s]["tau_kernel"] for s in stages], "o-",
             label="E1 kernel (~Khilkevich)")
    axc.plot(x, [res["per_stage"][s]["tau_glm"] for s in stages], "s--",
             label="E2 GLM filter (~Orsolic)")
    axc.axhspan(*LIT_BRACKET, color="gray", alpha=0.15,
                label=f"lit bracket {LIT_BRACKET[0]}–{LIT_BRACKET[1]} s")
    axc.set_xticks(x); axc.set_xticklabels(stages)
    axc.set_ylabel("integration tau (s)"); axc.set_title("C. tau by stage (triangulated)")
    axc.legend(fontsize=7)
    # Panels A (kernels overlaid per stage), B (E2 filters per stage), D (Δτ bootstrap
    # hist w/ CI), E (method-artifact: E1 vs E2 on pooled data), F (state-controlled rerun)
    # are filled from `res`, the per-stage kernels, and a state_control=False rerun.
    save_figure(fig, "fig0N_integration_timescale", "01_behavior")

    rows = [{"stage": s, **d} for s, d in res["per_stage"].items()]
    for key, d in res["delta"].items():
        rows.append({"stage": f"DELTA_{key}", **d})
    pd.DataFrame(rows).to_csv(CACHE, index=False)
    print(f"  saved {CACHE}")


if __name__ == "__main__":
    main()
```

> Panels A/B/D/E/F are sketched (C is complete). Fill from `res` + the per-stage kernels (re-call `collect_lick_segments`/`lick_triggered_kernel` for plotting), a `Δτ` bootstrap histogram, the pooled E1-vs-E2 method-artifact comparison (spec §5), and a `state_control=False` rerun overlaid on C. Keep each panel ≤20 lines.

- [ ] **Step 2: Verify the script imports cleanly**

Run: `py -c "import importlib.util as u; s=u.spec_from_file_location('b1','analysis_suite/01_behavior/i_integration_timescale.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print('import OK')"`
Expected: `import OK`.

- [ ] **Step 3: Commit**

```bash
git add analysis_suite/01_behavior/i_integration_timescale.py
git commit -m "feat(B1): fig0N script — per-stage integration timescale, triangulated"
```

---

### Task 9: Run on real data, record result, update index

**Files:**
- Modify: `docs/science/QUESTION_INDEX.md`

- [ ] **Step 1: Pre-flight (uses Task 1's resolved numbers)**

Confirm `docs/science/B1_stimulus_characterization.md` exists and that `DT/T0/MAX_LAG` in the script match it. Spot-check one real session:

Run: `py -c "from visdetect.analysis.config import load_staging_manifest; from visdetect.suite.loader import load_session; from visdetect.analysis.integration_timescale import build_evidence_traces, collect_lick_segments; m=load_staging_manifest(qc_only=True); s=load_session(int(m.iloc[-1]['session_name'])); ev=build_evidence_traces(s); seg,lags,info=collect_lick_segments(ev,'all_fa',0.1,1.5,0.05); print('trials',len(ev),'fa_segments',seg.shape, 'n_fa_licks', (ev.outcome=='fa').sum())"`
Expected: prints a sane trial count and a non-trivial number of FA segments. If FA licks per stage are too few for a stable kernel (spec §6/§9), pool to two stages and widen CIs.

- [ ] **Step 2: Run the analysis**

Run: `cd analysis_suite && py 01_behavior/i_integration_timescale.py`
Expected: prints per-stage `tau_kernel`/`tau_glm` and `Δτ` with CIs; saves figure + stats + cache. (Slow — bootstrap × GLM fits.)

- [ ] **Step 3: Sanity-check against the spec §7 success criteria**

Open `analysis_suite/figures/01_behavior/fig0N_integration_timescale.png`. Confirm: (a) `Δτ_kernel > 0` with CI excluding 0 **and** `Δτ_glm` same sign (→ "τ is learned"), or a clean null/inconclusive honestly reported; (b) the whitening changed the kernel as expected from Task 1's ACF; (c) the E1-vs-E2 gap on pooled data — does it reproduce the ~3–4× literature factor (method artifact) or not?; (d) the learning trend survives the `state_control=False` vs `True` comparison (panel F).

- [ ] **Step 4: Update the question index**

Set the B1 row Plan cell + status in `docs/science/QUESTION_INDEX.md`:

```
| B1 ⭐ | Is the evidence-integration timescale a *learned* quantity? | T1 | done | [design](../superpowers/specs/2026-06-10-B1-integration-timescale-learned-design.md) | [plan](../superpowers/plans/2026-06-10-B1-integration-timescale-learned-plan.md) |
```

(Use `in-progress` if E3/state-control/per-stage-data forced caveats.)

- [ ] **Step 5: Commit**

```bash
git add docs/science/QUESTION_INDEX.md analysis_suite/cache/integration_timescale_taus.csv
git commit -m "data(B1): integration-timescale result + index update"
```

(If `analysis_suite/figures/**` is gitignored, omit it from the add.)

---

## Self-Review

**1. Spec coverage:**
- §3 data inputs (pre-planned TF stream → `e(t)`, truncated at lick; behaviour RT/outcome; pluggable state) → Task 2 (`build_evidence_traces`) + Task 3 (`load_state_labels`). ✓
- §4 Estimator 1 (lick-triggered kernel, engaged-FA primary / Hits complementary, `t0` truncation, autocorrelation-corrected) → Tasks 3–5 (`collect_lick_segments` t0 + Hit window, `lick_triggered_kernel` + whitening, `kernel_tau`). ✓
- §4 Estimator 2 (regression filter, inherently autocorrelation-corrected) → Task 6. **Deviation (resolutions box #2):** built as a dedicated distributed-lag hazard regression because the existing GLM lacks a multi-lag filter — same intent, regression-based. ✓ (noted)
- §4/§9 Estimator 3 (DDM λ) **deferred post-B0** → not in this plan, by design. ✓
- §5 per-stage τ + bootstrap Δτ + decision rule (E1 CI excludes 0, E2 sign-agrees) → Task 7 (`triangulate_tau`). ✓
- §5 method-artifact test (E1 vs E2 on same data ≈ field discrepancy) → Task 8 panel E + Task 9 Step 3c. ✓ (figure-side)
- §5/§6 state control (matched/per-state) → `state_by_stage` plumbing (Tasks 7–8) + `state_control` rerun. ✓
- §6 stimulus-autocorrelation confound (the load-bearing one) → Task 1 (characterise) + Task 4 (whiten, deterministic test). ✓
- §6 non-decision-time truncation → Task 3 (`t0` end-cut) + Task 1 (`t0` estimate). ✓
- §7 success/negative/inconclusive criteria → Task 9 Step 3. ✓
- §8 deliverables (module/tests/script/figure A–F/stats/cache) → Tasks 2–8. Figure panels A/B/D/E/F **sketched, not fully coded** (C complete) — flagged in Task 8; they read already-tested quantities.
- §9 BLOCKING stimulus characterization + GLM extraction entry point + state entry point + E3 deferral + min-licks → Task 1 + resolutions box + Task 9 Step 1. ✓

**2. Placeholder scan:** Pure-Python units (evidence, segments, whitening, `kernel_tau`, lagged-hazard design, triangulate) are complete with real code. **Honestly flagged, not silent:** (a) figure panels A/B/D/E/F sketched with explicit fill instructions; (b) `dt`/`t0`/`max_lag` are Task-1-resolved and threaded as parameters (not hardcoded magic — defaults documented as provisional); (c) Estimator 2 deviates from "reuse the GLM" per resolutions box #2; (d) evidence-extractor duplication vs B0 is a named reconcile-at-E3 deferral. No vague TODOs.

**3. Type consistency:** `build_evidence_traces → ev_df[evidence, lick_time, outcome, change_time, trial_idx]` feeds `collect_lick_segments`, `stimulus_autocov`, `build_lagged_hazard_design`, and `_stage_taus`/`triangulate_tau` — all consume the same `ev_df` schema and `(segments, lags)` shapes. `kernel_tau(kernel, lags)` and `glm_filter_tau(weights, dt)` share the exp/half-area readout. `triangulate_tau` returns `{per_stage:{stage:{tau_kernel,tau_glm,n_licks}}, delta:{key:{estimate,ci_low,ci_high,n_boot}}}` consumed identically in Task 8. `load_state_labels` returns a `trial_idx`-indexed Series; the script reindexes it to `trial_uid` before passing as `state_by_stage`. Consistent.

**Statistician knobs (flag for review at planning/exec):** exp-fit vs half-area for the headline τ; ridge `reg` in the whitening (set from Task 1's conditioning); bootstrap scheme (trial-level vs session-level — session-level is more honest for n=1 across sessions; current code is trial-level, consider a session-block bootstrap in Task 7/8); `LogisticRegression` C / L2 strength for E2; minimum engaged-FA licks per stage for a stable kernel; whether to merge Naive→Learning (script does) or keep three stages.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-10-B1-integration-timescale-learned-plan.md`. Like B0/B2 this is **parked for a fresh chat** (open the B1 spec + this plan). Two execution options when it is picked up:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks. **Run Task 1 first** — its measured `dt`/ACF/`t0` configure Tasks 4/8, and confirm the GLM-deviation (resolutions box #2) still holds.
2. **Inline Execution** — execute in-session with checkpoints.

No new dependency, so a dedicated worktree is optional (lighter than B0). Which approach?
