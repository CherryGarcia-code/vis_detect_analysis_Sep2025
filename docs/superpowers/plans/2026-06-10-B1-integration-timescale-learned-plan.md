# B1 — Integration timescale is learned: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estimate BG_046's behavioural evidence-integration timescale `τ` **per learning stage** by **three independent estimators that faithfully reproduce the two reference papers' actual methods** — a model-free lick-triggered TF kernel (both papers' descriptive kernel), a Khilkevich-style leaky-integrator-to-threshold fit (their headline ~0.27 s), and an Orsolic-style multi-lag lick-hazard filter (their headline ~1 s) — and test whether `τ` **grows** Naive→Expert, triangulated across estimators and robust to a behavioural-state control, per `docs/superpowers/specs/2026-06-10-B1-integration-timescale-learned-design.md`.

**Architecture:** A new library module `visdetect/analysis/integration_timescale.py` holds: a pyddm-free per-trial evidence extractor (`e(t)=log2(TF(t)/TF_base)` truncated at the lick, **excluding the no-noise trials**); lick-set selection with the non-decision-time (`t0`/refractory) truncation and the engaged-FA conditioning (state is a **pluggable** input, with Khilkevich's classifier-free "lick ≥ 2 s after baseline" criterion as the default); the three τ estimators (`kernel_tau`, `leaky_integrator_tau`, `hazard_filter_tau`) sharing a single τ-readout helper; the stimulus-autocovariance whitening (kept, but near-identity for this white-by-design stimulus — Task 1 verifies); and `triangulate_tau` (per-stage τ for all three + bootstrap `Δτ`). A thin `analysis_suite/01_behavior/i_integration_timescale.py` orchestrates over the staging manifest and renders the figure + caches.

**Tech Stack:** numpy, pandas, scipy (`scipy.linalg.toeplitz`, `scipy.optimize.curve_fit`), scikit-learn (`LogisticRegression`), matplotlib — **all already present; NO new dependency, NO pyddm**. Reuses `visdetect.analysis.behavior.get_trial_dataframe` (RT/outcome semantics), `visdetect.analysis.constants` (`TF_FAST_THRESH_LOG2`/`TF_SLOW_THRESH_LOG2`), `visdetect.suite.loader.load_hmm_assignments`, `visdetect.analysis.config.load_staging_manifest`.

---

> ## Planning-time resolutions (read once — grounded in the two papers' Methods)
>
> The Methods of Khilkevich & Lohse 2024 (Nature) and Orsolic 2021 (Neuron) were read in full; the design below mirrors them. Settled facts downstream tasks assume:
>
> 1. **The stimulus is white by design.** Both papers: baseline TF is drawn **iid every 50 ms** (independent log-normal, log₂ mean 0, **SD 0.25 octaves**, geomean 1 Hz). So the kernel's stimulus-autocorrelation confound is **near-identity** — the whitening is kept for honesty but is expected to do nothing. **Task 1 verifies** the stored `baseline_values` are actually white at 50 ms (an upsampled/smoothed display would re-introduce colour). The grid is **`dt = 0.05`**; `constants.TF_SAMPLE_PERIOD = 0.25` ("4 Hz base") is the *base grating TF*, **not** a sample period — do **not** import it for the grid, and do **not** edit `constants.py` (a parallel chat owns it). (Memory `tf_fluctuation_50ms_vs_constant`.)
> 2. **The three estimators ARE the two papers' methods** (this replaces the spec's vaguer "E1≈Khilkevich / E2≈Orsolic"):
>    - **E1a — model-free LTA kernel.** *Both* papers compute it (Khilkevich: average TF over **[−1.5, 0] s** before early licks, 50 ms bins, bootstrap CI). Descriptive; τ read three ways.
>    - **E1b — Khilkevich leaky-integrator-to-threshold (their headline ~0.27 s).** Integrate log-TF with a leak time `τ`; lick when it crosses a threshold; **scan `τ` (0.05 s = no-integration anchor → ~3 s) × threshold** and pick the pair that best **predicts early-lick times** (crossing within 1 s preceding the lick); compare against the `τ=0.05` no-integration null. **Pure numpy — no pyddm.**
>    - **E2 — Orsolic-style multi-lag lick-hazard filter (their headline ~1 s).** A discrete-time lick-hazard logistic regression with a **2.5 s (Q=50-lag)** stimulus filter + a time-in-trial nuisance + L2/ARD regularisation; τ = the filter's integration window. This is a **tractable analog** of Orsolic's GP-classification model (their released code is `github.com/znamlab/rt_model_orsolic`; reimplementing the full GP + ARD filter bank + tanh time-warp is out of scope) — but it keeps the load-bearing properties: regression-based (autocorrelation-robust), long history, time nuisance. **My earlier "slow-exp + fast-derivative" idea was a mischaracterisation of Orsolic and is dropped.**
>    - **E3 — DDM leak `λ` (`τ=1/λ`)** stays deferred to post-B0 (spec §4/§9). Note Khilkevich's E1b is itself a lightweight leaky accumulator, so E1b is the no-pyddm stand-in for the generative-model timescale until E3.
> 3. **Confirmed task constants baked in:** non-decision/refractory floor **`t0 = 0.15 s`** (both papers exclude the first 150 ms; Khilkevich's pulse analysis cuts the last 0.2 s — Task 1 refines `t0` from BG_046's reflex/fast-lick floor); **exclude no-noise trials** (15–30 % of trials have constant 1 Hz baseline → zero fluctuation evidence → must not enter any kernel/fit); engaged-FA conditioning default = **early lick ≥ 2 s after baseline onset** (Khilkevich's "decrease the influence of impulsive licks"), refined by the pluggable state label when available; TF-pulse fast/slow = **±1 SD = ±0.25 log₂** = the existing `TF_FAST_THRESH_LOG2`/`TF_SLOW_THRESH_LOG2`.
>
> Evidence extractor is duplicated pyddm-free from B0 (`ddm.build_trial_evidence`); reconcile/share when B0's E3 lands.

---

## File Structure

- **Create** `src/visdetect/analysis/integration_timescale.py` — all B1 computation (extractor, lick-sets, whitening, the three τ estimators, triangulation; pluggable `load_state_labels`).
- **Create** `tests/analysis/test_integration_timescale.py` — TDD: evidence truncation + no-noise exclusion, lick-set + `t0` + ≥2 s conditioning, whitening identity-on-white / inverts-planted-colour, τ-recovery for **each** estimator from a simulated leaky integrator, `Δτ` bootstrap detects a planted learning increase.
- **Create** `analysis_suite/01_behavior/i_integration_timescale.py` — orchestration + figure (`fig0N`, panels A–F) + stats/cache.
- **Create** `scripts/analysis/behavior/characterize_tf_stream.py` + `docs/science/B1_stimulus_characterization.md` — Task 1's recorded answer to the BLOCKING stimulus question.
- **Modify** `docs/science/QUESTION_INDEX.md` — bump B1 status (Task 9).

Conventions (`CLAUDE.md`): constants from `visdetect.analysis.constants`; `load_staging_manifest()`; `setup_style()`/`save_figure()`; `del sess; gc.collect()`; `py` not `python`.

---

### Task 1: Stimulus characterization — verify white-at-50 ms, find t0 and the no-noise fraction (spec §9 BLOCKING)

The whitening (Task 4), `dt`, `t0`, and the no-noise exclusion all depend on the real stimulus statistics. This task **measures them on real BG_046 sessions and records the answer**; it is a diagnostic, not TDD.

**Files:**
- Create: `scripts/analysis/behavior/characterize_tf_stream.py`
- Create: `docs/science/B1_stimulus_characterization.md`

- [ ] **Step 1: Write the diagnostic script**

```python
# scripts/analysis/behavior/characterize_tf_stream.py
"""B1 Task 1 — characterise the baseline TF stream before any kernel is computed.

Answers spec §9 BLOCKING items: (a) real per-sample duration of trial.baseline_values
(expect ~50 ms, NOT constants.TF_SAMPLE_PERIOD=0.25 — memory tf_fluctuation_50ms_vs_constant);
(b) is e(t)=log2(TF/base) white at 50 ms (=> whitening is near-identity) or correlated;
(c) the non-decision floor t0 (reflex/fast-lick latency); (d) the fraction of no-noise
trials (constant 1 Hz baseline) that must be excluded.
"""
import numpy as np
import gc
from visdetect.analysis.config import load_staging_manifest
from visdetect.suite.loader import load_session

DT = 0.05


def main(n_sessions=4):
    manifest = load_staging_manifest(qc_only=True)
    periods, acfs, ref_lat, fast_fa, stds = [], [], [], [], []
    for _, row in manifest.head(n_sessions).iterrows():
        try:
            sess = load_session(int(row["session_name"]))
        except FileNotFoundError:
            continue
        for t in sess.trials:
            oc = (t.trialoutcome or "").lower()
            bv = getattr(t, "baseline_values", None)
            ct = getattr(t, "change_time", None)
            if bv is not None and ct:
                bv = np.asarray(bv).ravel()
                if bv.size > 5 and ct > 0:
                    periods.append(float(ct) / bv.size)
                    e = np.log2(np.clip(bv, 1e-6, None) / np.median(bv))
                    stds.append(float(np.std(e)))                # ~0.25 noisy, ~0 no-noise
                    e = e - e.mean()
                    n = min(40, e.size // 2)
                    if n > 2 and e.std() > 1e-6:
                        acfs.append(np.array([np.corrcoef(e[:e.size-l], e[l:])[0, 1]
                                              for l in range(n)]))
            rts = getattr(t, "reactiontimes", {}) or {}
            if oc == "ref" and rts.get("RT"):
                ref_lat.append(float(rts["RT"]))
            if oc == "fa" and rts.get("FA"):
                fast_fa.append(float(rts["FA"]))
        del sess; gc.collect()

    periods, stds = np.array(periods), np.array(stds)
    print(f"baseline sample period: median={np.median(periods)*1000:.1f} ms")
    if acfs:
        L = min(len(a) for a in acfs)
        m = np.mean([a[:L] for a in acfs], axis=0)
        print(f"mean ACF lag1={m[1]:.3f} lag2={m[2]:.3f} -> "
              f"{'WHITE (whitening ~identity)' if abs(m[1])<0.1 else 'CORRELATED (whiten!)'}")
    print(f"baseline log2 SD: median={np.median(stds):.3f} (expect ~0.25 noisy)")
    print(f"no-noise fraction (SD<0.05): {np.mean(stds < 0.05):.2%}")
    lat = np.r_[np.array(ref_lat), np.array(fast_fa)[np.array(fast_fa) < 0.4] if fast_fa else []]
    t0 = float(np.nanpercentile(lat, 5)) if lat.size else 0.15
    print(f"t0 estimate (5th pct reflex/fast-FA latency): {t0*1000:.0f} ms")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it on real data**

Run: `py scripts/analysis/behavior/characterize_tf_stream.py`
Expected: sample period ~50 ms (NOT 250 ms); ACF verdict (expect WHITE); baseline SD ~0.25 with a ~15–30 % no-noise spike at SD≈0; `t0` ~50–150 ms.

- [ ] **Step 3: Record the answer in `docs/science/B1_stimulus_characterization.md`**

Capture the measured period, the **white/correlated verdict** (decides whether Task 4's whitening matters), the no-noise fraction + the SD cutoff to use, and the chosen `dt`/`t0`. These configure Tasks 2/4/8.

- [ ] **Step 4: Commit**

```bash
git add scripts/analysis/behavior/characterize_tf_stream.py docs/science/B1_stimulus_characterization.md
git commit -m "diag(B1): characterise TF stream (white-at-50ms, t0, no-noise fraction)"
```

---

### Task 2: Per-trial evidence extraction (pyddm-free; truncated at lick; no-noise excluded)

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
    rng = np.random.default_rng(0)
    base_noisy = 2.0 * 2.0 ** (rng.normal(0, 0.25, 40))          # noisy baseline around 2 Hz
    base_flat = np.full(40, 2.0)                                  # no-noise trial (constant)
    t_fa = SimpleNamespace(trialoutcome="FA", change_size=1.0, change_time=2.0,
                           reactiontimes={"FA": 0.5}, baseline_values=base_noisy, n_seen=None)
    t_hit = SimpleNamespace(trialoutcome="Hit", change_size=4.0, change_time=1.0,
                            reactiontimes={"RT": 0.3}, baseline_values=base_noisy, n_seen=None)
    t_nonoise = SimpleNamespace(trialoutcome="FA", change_size=1.0, change_time=2.0,
                                reactiontimes={"FA": 0.6}, baseline_values=base_flat, n_seen=None)
    return SimpleNamespace(trials=[t_fa, t_hit, t_nonoise])


def test_evidence_truncates_at_lick_flags_noise_and_excludes_nonoise():
    df = build_evidence_traces(_toy_session(), dt=DT, tf_base=2.0, min_baseline_std=0.05)
    # the no-noise (constant) trial is dropped
    assert len(df) == 2
    assert df["noisy"].all()
    fa = df[df.outcome == "fa"].iloc[0]
    hit = df[df.outcome == "hit"].iloc[0]
    assert fa["lick_time"] == pytest.approx(0.5, abs=DT)
    assert len(fa["evidence"]) == pytest.approx(0.5 / DT, abs=1)
    # Hit lick at change_time+RT = 1.3 s; post-change e = log2(4*TF/base) ~ 2 + fluctuation
    assert hit["lick_time"] == pytest.approx(1.3, abs=DT)
    post = hit["evidence"][int(1.0 / DT) + 2:]
    assert np.all(post > 1.0)                                     # change step dominates
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_integration_timescale.py::test_evidence_truncates_at_lick_flags_noise_and_excludes_nonoise -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'visdetect.analysis.integration_timescale'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/visdetect/analysis/integration_timescale.py
"""B1 — is the evidence-integration timescale a learned quantity?

Three behavioural estimators of tau, per learning stage, each mirroring a reference method:
  E1a model-free lick-triggered TF kernel (LTA)        -> kernel_tau          (both papers)
  E1b Khilkevich leaky-integrator-to-threshold fit     -> leaky_integrator_tau (~0.27 s)
  E2  Orsolic-style multi-lag lick-hazard filter        -> hazard_filter_tau    (~1 s)

NOTE (planning-time resolutions): stimulus is white at 50 ms by design (whitening ~identity,
Task 1 verifies); dt=0.05 (NOT constants.TF_SAMPLE_PERIOD=0.25); exclude no-noise trials;
t0=0.15 s refractory; engaged-FA default = lick >= 2 s after baseline. Evidence extractor is
duplicated pyddm-free from B0's ddm.build_trial_evidence (reconcile when B0's E3 lands).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

DT = 0.05                       # integration grid (s) = confirmed 50 ms TF update period
REFRACTORY_S = 0.15             # non-decision/refractory floor (t0); Task 1 refines
ENGAGED_MIN_LICK_S = 2.0        # Khilkevich classifier-free engaged-FA criterion
LTA_WINDOW_S = 1.5              # Khilkevich lick-triggered-average window [-1.5, 0]
HAZARD_WINDOW_S = 2.5           # Orsolic stimulus-history window (Q=50 lags)


def _lick_time(trial) -> float:
    """Lick time aligned to Baseline_ON (mirrors lick-hazard GLM). NaN if none."""
    oc = (getattr(trial, "trialoutcome", "") or "").lower()
    rts = getattr(trial, "reactiontimes", {}) or {}
    ct = getattr(trial, "change_time", None)
    if oc == "fa":
        v = rts.get("FA", rts.get("fa"));  return float(v) if v else np.nan
    if oc == "hit":
        v = rts.get("RT", rts.get("Hit", rts.get("hit")))
        return (float(ct) + float(v)) if (v and ct) else np.nan
    return np.nan


def build_evidence_traces(session, dt: float = DT, tf_base: Optional[float] = None,
                          min_baseline_std: float = 0.05) -> pd.DataFrame:
    """Per-trial e(t)=log2(TF(t)/TF_base) on a dt grid, truncated at the lick.

    Columns: trial_idx, outcome, change_size, change_time, lick_time, baseline_std,
    noisy (bool), evidence (np.ndarray). NO-NOISE trials (baseline_std < min_baseline_std,
    i.e. the constant-1-Hz trials) are DROPPED — they carry no fluctuation evidence.
    """
    rows = []
    for tidx, t in enumerate(getattr(session, "trials", []) or []):
        oc = (getattr(t, "trialoutcome", "") or "").lower()
        if oc not in ("hit", "fa"):
            continue
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
        bl_e = np.log2(np.clip(bv, 1e-6, None) / base)
        baseline_std = float(np.std(bl_e))
        if baseline_std < min_baseline_std:
            continue                                             # no-noise trial: exclude
        bperiod = (ct / bv.size) if (np.isfinite(ct) and ct > 0) else dt
        n = max(1, int(round(lick / dt)))
        e = np.zeros(n)
        for i in range(n):
            tau_t = i * dt
            j = min(bv.size - 1, int(tau_t / bperiod))
            tf = bv[j] * cs if (np.isfinite(ct) and tau_t >= ct and cs > 1.0) else bv[j]
            e[i] = np.log2(max(tf, 1e-6) / base)
        rows.append({"trial_idx": tidx, "outcome": oc, "change_size": cs,
                     "change_time": ct, "lick_time": lick,
                     "baseline_std": baseline_std, "noisy": True, "evidence": e})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_integration_timescale.py::test_evidence_truncates_at_lick_flags_noise_and_excludes_nonoise -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/integration_timescale.py tests/analysis/test_integration_timescale.py
git commit -m "feat(B1): pyddm-free evidence extractor (truncated at lick, no-noise excluded)"
```

---

### Task 3: State accessor (pluggable) + lick-set selection (t0 truncation + Khilkevich ≥2 s engaged criterion)

**Files:**
- Modify: `src/visdetect/analysis/integration_timescale.py` (append `load_state_labels`, `select_licks`, `collect_lick_segments`)
- Test: `tests/analysis/test_integration_timescale.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from visdetect.analysis.integration_timescale import select_licks, collect_lick_segments


def _ev_df():
    e = np.arange(40, dtype=float)                               # ramp: e[i]=i (dt=0.05 -> 2 s)
    return pd.DataFrame([
        {"trial_idx": 0, "outcome": "fa", "change_size": 1.0, "change_time": np.inf,
         "lick_time": 1.0, "noisy": True, "evidence": e.copy()},   # impulsive (lick < 2 s)
        {"trial_idx": 1, "outcome": "fa", "change_size": 1.0, "change_time": np.inf,
         "lick_time": 2.5, "noisy": True, "evidence": e.copy()},   # engaged (lick >= 2 s)
    ])


def test_select_licks_engaged_uses_2s_default_then_state():
    df = _ev_df()
    eng = select_licks(df, "engaged_fa")                          # default: lick >= 2 s
    assert list(eng["trial_idx"]) == [1]
    state = pd.Series({0: "engaged", 1: "impulsive"})            # state overrides: trial1 not engaged
    eng2 = select_licks(df, "engaged_fa", state_labels=state)
    assert len(eng2) == 0                                         # trial1 fails state despite >=2 s


def test_lick_segments_truncate_t0_and_window_back():
    df = _ev_df().iloc[[1]]                                       # the engaged lick at 2.5 s
    segs, lags, info = collect_lick_segments(df, "all_fa", t0=0.10, max_lag=0.30, dt=0.05)
    assert segs.shape == (1, 6)                                   # 0.30/0.05 = 6 lags
    # causal end = lick - t0 = 2.5 - 0.10 = 2.40 s -> evidence index 48 is out of range (len 40)
    # so nearest in-range lag samples are taken; furthest lag is strictly earlier (smaller)
    assert segs[0, -1] < segs[0, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_integration_timescale.py -k "select_licks or lick_segments" -v`
Expected: FAIL — `ImportError: cannot import name 'select_licks'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to integration_timescale.py
def load_state_labels(session_name, K: int = 3) -> pd.Series:
    """Pluggable per-trial behavioural-state accessor, indexed by trial_idx, values in
    {'engaged','impulsive','other'}. Default = GLM-HMM (loader.load_hmm_assignments),
    mapped via the renamed labels (mirrors the lick-hazard GLM). SWAP THIS BODY to point at
    the in-development self-tailored classifier without touching any estimator code.
    """
    from visdetect.suite.loader import load_hmm_assignments
    df = load_hmm_assignments(K=K)
    sub = df[df["session_name"].astype(int) == int(session_name)]
    if sub.empty:
        return pd.Series(dtype=object)

    def _map(lbl):
        s = str(lbl)
        if s.startswith("Engaged"):
            return "engaged"
        if s.startswith("Impulsive") or s == "Biased":
            return "impulsive"
        return "other"

    return sub.set_index("trial_idx")["hmm_state_label"].map(_map)


def select_licks(ev_df, lick_set: str, state_labels: Optional[pd.Series] = None,
                 min_lick_s: float = ENGAGED_MIN_LICK_S) -> pd.DataFrame:
    """Subset ev_df to the qualifying lick trials for an estimator.

    'engaged_fa' (PRIMARY): FAs with lick >= min_lick_s after baseline (Khilkevich's
       classifier-free de-impulsive criterion); AND state=='engaged' if state_labels given.
    'hit': Hit trials. 'all_fa': all FA trials (diluted; not recommended for the headline).
    """
    df = ev_df[ev_df["lick_time"].notna()]
    if lick_set in ("engaged_fa", "all_fa"):
        df = df[df["outcome"] == "fa"]
        if lick_set == "engaged_fa":
            df = df[df["lick_time"] >= min_lick_s]
            if state_labels is not None:
                keep = df["trial_idx"].map(lambda i: str(state_labels.get(i, "other")) == "engaged")
                df = df[keep.values]
    elif lick_set == "hit":
        df = df[df["outcome"] == "hit"]
    else:
        raise ValueError(f"unknown lick_set {lick_set!r}")
    return df


def collect_lick_segments(ev_df, lick_set: str, t0: float = REFRACTORY_S,
                          max_lag: float = LTA_WINDOW_S, dt: float = DT,
                          state_labels: Optional[pd.Series] = None,
                          sensory_latency: float = 0.05):
    """Stack the evidence window preceding each qualifying lick (for the LTA kernel).

    Returns (segments [n_licks, n_lags], lags, info_df). Lags run BACK from the causal end
    (lick - t0): stimulus within t0 of the lick is too late to have driven it. For 'hit',
    lags whose absolute time precedes (change_time + sensory_latency) are masked NaN.
    """
    sel = select_licks(ev_df, lick_set, state_labels=state_labels)
    n_lags = int(round(max_lag / dt))
    lags = (np.arange(1, n_lags + 1) * dt)
    segs, info = [], []
    for _, r in sel.iterrows():
        e = np.asarray(r["evidence"], float)
        end_t = r["lick_time"] - t0
        seg = np.full(n_lags, np.nan)
        for k in range(n_lags):
            t = end_t - lags[k]
            if t < 0:
                continue
            if r["outcome"] == "hit" and np.isfinite(r["change_time"]) \
                    and t < (r["change_time"] + sensory_latency):
                continue
            i = int(round(t / dt))
            if 0 <= i < e.size:
                seg[k] = e[i]
        segs.append(seg)
        info.append({"trial_idx": r["trial_idx"], "outcome": r["outcome"]})
    arr = np.asarray(segs, float) if segs else np.empty((0, n_lags))
    return arr, lags, pd.DataFrame(info)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -m pytest tests/analysis/test_integration_timescale.py -k "select_licks or lick_segments" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/integration_timescale.py tests/analysis/test_integration_timescale.py
git commit -m "feat(B1): pluggable state accessor + lick-set selection (t0 + >=2s engaged)"
```

---

### Task 4: E1a — whitening + LTA kernel + `kernel_tau` (three τ readouts)

**Files:**
- Modify: `src/visdetect/analysis/integration_timescale.py` (append `stimulus_autocov`, `whiten_kernel`, `lick_triggered_kernel`, `tau_from_profile`, `kernel_tau`)
- Test: `tests/analysis/test_integration_timescale.py` (append a fast unit test, a deterministic whitening test, a slow recovery test + the shared simulator)

- [ ] **Step 1: Write the failing tests (+ the simulator Tasks 4–7 reuse)**

```python
from scipy.linalg import toeplitz
from visdetect.analysis.integration_timescale import (
    whiten_kernel, lick_triggered_kernel, stimulus_autocov, kernel_tau,
)


def test_whitening_inverts_planted_colour_and_is_identity_on_white():
    n = 8
    acf = 0.6 ** np.arange(n)
    C = toeplitz(acf)
    lags = (np.arange(1, n + 1) * 0.05)
    k_true = np.exp(-lags / 0.20)
    k_raw = C @ k_true                                           # reverse-corr bias on colour
    assert not np.allclose(k_raw, k_true, atol=0.05)
    assert np.allclose(whiten_kernel(k_raw, C, reg=0.0), k_true, atol=1e-6)
    # white stimulus (C = I) -> whitening does nothing
    assert np.allclose(whiten_kernel(k_true, np.eye(n), reg=0.0), k_true)


def test_kernel_tau_three_readouts_on_exponential():
    lags = (np.arange(1, 31) * 0.05)
    k = np.exp(-lags / 0.30)
    assert kernel_tau(k, lags, method="exp") == pytest.approx(0.30, rel=0.1)
    assert 0.1 < kernel_tau(k, lags, method="half_area") < 0.6
    assert kernel_tau(k, lags, method="window") > 0.3            # 90% cumulative extent


def simulate_leaky_integrator_fas(tau_true, n_trials, dt=0.05, T=3.0, thresh=1.0,
                                   gain=0.5, white=True, rho=0.0, seed=0):
    """Ground-truth generator: leaky integrator -> early-lick crossings on baseline-only
    stimulus. Returns an ev_df matching build_evidence_traces. Reused by Tasks 4-7."""
    rng = np.random.default_rng(seed)
    n = int(T / dt)
    leak = dt / tau_true
    rows = []
    for uid in range(n_trials):
        stim = rng.standard_normal(n) * gain
        if not white:
            for i in range(1, n):
                stim[i] = rho * stim[i - 1] + np.sqrt(1 - rho ** 2) * stim[i]
        x, lick_i = 0.0, None
        for i in range(n):
            x += -leak * x + stim[i]
            if x >= thresh:
                lick_i = i; break
        if lick_i is None or lick_i < 5:
            continue
        rows.append({"trial_idx": uid, "outcome": "fa", "change_size": 1.0,
                     "change_time": np.inf, "lick_time": lick_i * dt, "noisy": True,
                     "baseline_std": float(np.std(stim[: lick_i + 1])),
                     "evidence": stim[: lick_i + 1].astype(float)})
    return pd.DataFrame(rows)


@pytest.mark.slow
def test_kernel_recovers_leaky_integrator_tau():
    df = simulate_leaky_integrator_fas(tau_true=0.30, n_trials=4000, white=True, seed=1)
    segs, lags, _ = collect_lick_segments(df, "all_fa", t0=0.0, max_lag=1.0, dt=0.05)
    C = stimulus_autocov(df, n_lags=len(lags), dt=0.05)          # ~identity (white)
    tau = kernel_tau(lick_triggered_kernel(segs, autocov=C), lags, method="exp")
    assert 0.15 < tau < 0.60
```

- [ ] **Step 2: Run to verify failure**

Run: `py -m pytest tests/analysis/test_integration_timescale.py -k "whitening or kernel_tau_three or kernel_recovers" -v`
Expected: FAIL — `ImportError: cannot import name 'whiten_kernel'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to integration_timescale.py
def stimulus_autocov(ev_df, n_lags: int, dt: float = DT, max_trials: Optional[int] = None):
    """Toeplitz stimulus autocovariance of baseline e(t) (pre-change, stationary segment)."""
    from scipy.linalg import toeplitz
    acov, counts = np.zeros(n_lags), np.zeros(n_lags)
    rows = ev_df if max_trials is None else ev_df.head(max_trials)
    for _, r in rows.iterrows():
        e = np.asarray(r["evidence"], float)
        ct = r["change_time"]
        if np.isfinite(ct):
            e = e[: int(ct / dt)]
        if e.size <= 1:
            continue
        e = e - e.mean()
        for lag in range(n_lags):
            if e.size > lag:
                acov[lag] += float(np.dot(e[: e.size - lag], e[lag:]))
                counts[lag] += (e.size - lag)
    return toeplitz(acov / np.maximum(counts, 1.0))


def whiten_kernel(raw_kernel, autocov, reg: float = 1e-3):
    """Recover the true filter from a reverse-correlation kernel: k_true = C^{-1} k_raw
    (ridge-regularised). For a white stimulus C=I and this is the identity."""
    C = np.asarray(autocov, float)
    if reg > 0:
        C = C + reg * (np.trace(C) / C.shape[0]) * np.eye(C.shape[0])
    return np.linalg.solve(C, np.asarray(raw_kernel, float))


def lick_triggered_kernel(segments, autocov=None, correct_autocorr: bool = True,
                          reg: float = 1e-3):
    """Mean evidence over lick-triggered segments, optionally whitened (near-identity if white)."""
    if segments.shape[0] == 0:
        raise ValueError("no lick segments")
    raw = np.nanmean(segments, axis=0)
    if correct_autocorr and autocov is not None:
        return whiten_kernel(raw, autocov, reg=reg)
    return raw


def tau_from_profile(profile, lags, method: str = "exp") -> float:
    """Shared tau readout from any lag profile (LTA kernel or hazard filter).

    'exp': fit A*exp(-lag/tau). 'half_area': lag at 50% cumulative |profile|.
    'window': lag at 90% cumulative |profile| (the integration extent)."""
    k = np.asarray(profile, float); L = np.asarray(lags, float)
    m = np.isfinite(k) & np.isfinite(L)
    k, L = k[m], L[m]
    if k.size < 3:
        return float("nan")
    if method in ("half_area", "window"):
        frac = 0.5 if method == "half_area" else 0.9
        step = float(np.median(np.diff(L))) if L.size > 1 else float(L[0])
        area = np.cumsum(np.abs(k)) * step
        if area[-1] <= 0:
            return float("nan")
        return float(np.interp(frac * area[-1], area, L))
    from scipy.optimize import curve_fit

    def _exp(x, A, tau):
        return A * np.exp(-x / np.maximum(tau, 1e-3))

    try:
        p, _ = curve_fit(_exp, L, k, p0=[float(np.nanmax(np.abs(k))), float(np.median(L))],
                         maxfev=10000, bounds=([-np.inf, 1e-3], [np.inf, 10 * float(L.max())]))
        return float(p[1])
    except Exception:
        return float("nan")


def kernel_tau(kernel, lags, method: str = "exp") -> float:
    """E1a tau readout from the LTA kernel (alias of tau_from_profile)."""
    return tau_from_profile(kernel, lags, method=method)
```

- [ ] **Step 4: Run to verify pass**

Run: `py -m pytest tests/analysis/test_integration_timescale.py -k "whitening or kernel_tau_three or kernel_recovers" -v -m "slow or not slow"`
Expected: PASS. A biased slow recovery is a *finding* (τ is method-sensitive) — confirm it's the threshold-nonlinearity bias before widening tolerance.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/integration_timescale.py tests/analysis/test_integration_timescale.py
git commit -m "feat(B1): E1a LTA kernel + whitening + kernel_tau (exp/half-area/window)"
```

---

### Task 5: E1b — Khilkevich leaky-integrator-to-threshold (`leaky_integrator_tau`)

**Files:**
- Modify: `src/visdetect/analysis/integration_timescale.py` (append `_integrate_to_threshold`, `leaky_integrator_tau`)
- Test: `tests/analysis/test_integration_timescale.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from visdetect.analysis.integration_timescale import leaky_integrator_tau


@pytest.mark.slow
def test_leaky_integrator_recovers_tau_and_beats_no_integration():
    df = simulate_leaky_integrator_fas(tau_true=0.30, n_trials=2500, white=True,
                                       thresh=1.0, gain=0.5, seed=5)
    res = leaky_integrator_tau(df, "all_fa", dt=0.05,
                               taus=np.r_[0.05, np.geomspace(0.1, 3.0, 18)],
                               thresholds=np.linspace(0.3, 3.0, 24), predict_window=1.0)
    # the best-fit decay sits near the planted 0.30 s and beats the no-integration anchor
    assert 0.12 < res["tau"] < 0.8
    assert res["score"] > res["score_no_integration"]
```

- [ ] **Step 2: Run to verify failure**

Run: `py -m pytest tests/analysis/test_integration_timescale.py -k leaky_integrator_recovers -v -m slow`
Expected: FAIL — `ImportError: cannot import name 'leaky_integrator_tau'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to integration_timescale.py
def _integrate_to_threshold(evidence, dt, tau, threshold):
    """Leaky integrate e(t); return the FIRST crossing index, or None. x <- x*(1-dt/tau)+e."""
    leak = dt / tau
    x = 0.0
    for i, e in enumerate(np.asarray(evidence, float)):
        x += -leak * x + e
        if x >= threshold:
            return i
    return None


def leaky_integrator_tau(ev_df, lick_set: str, dt: float = DT, taus=None, thresholds=None,
                         predict_window: float = 1.0, t0: float = REFRACTORY_S,
                         state_labels: Optional[pd.Series] = None) -> dict:
    """Khilkevich leaky-integrator-to-threshold fit (~0.27 s headline).

    Scan (tau, threshold); a lick is 'predicted' if the integrated-TF threshold crossing
    lands within `predict_window` s preceding the actual lick (>= the t0 floor). Pick the
    pair maximising the fraction of qualifying early licks predicted. Returns the best tau,
    its score, and the no-integration (tau=0.05 s) anchor score. The tau=0.05 s grid point
    is the no-integration null."""
    if taus is None:
        taus = np.r_[0.05, np.geomspace(0.1, 3.0, 18)]
    if thresholds is None:
        thresholds = np.linspace(0.1, 3.0, 30)
    sel = select_licks(ev_df, lick_set, state_labels=state_labels)
    traces = [(np.asarray(r["evidence"], float), float(r["lick_time"])) for _, r in sel.iterrows()
              if np.asarray(r["evidence"], float).size > 3]
    if len(traces) < 10:
        return {"tau": np.nan, "score": np.nan, "score_no_integration": np.nan, "n_licks": len(traces)}

    def _score(tau, thr):
        ok = 0
        for e, lick in traces:
            c = _integrate_to_threshold(e, dt, tau, thr)
            if c is None:
                continue
            ct = c * dt
            if (lick - predict_window - t0) <= ct <= (lick - t0 + dt):
                ok += 1
        return ok / len(traces)

    best = {"tau": np.nan, "thr": np.nan, "score": -1.0}
    score_grid = {}                                              # per-tau best score (for the null)
    for tau in taus:
        s_tau = max(_score(tau, thr) for thr in thresholds)
        score_grid[tau] = s_tau
        if s_tau > best["score"]:
            best = {"tau": float(tau), "thr": np.nan, "score": float(s_tau)}
    return {"tau": best["tau"], "score": best["score"],
            "score_no_integration": float(score_grid.get(0.05, np.nan)),
            "n_licks": len(traces)}
```

- [ ] **Step 4: Run to verify pass**

Run: `py -m pytest tests/analysis/test_integration_timescale.py -k leaky_integrator_recovers -v -m slow`
Expected: PASS (a grid scan; ~30 s). If the best τ pins to the 0.05 s anchor on real data, that is the **no-integration / outlier-detector** result (spec §7 negative) — report it as such.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/integration_timescale.py tests/analysis/test_integration_timescale.py
git commit -m "feat(B1): E1b Khilkevich leaky-integrator-to-threshold tau"
```

---

### Task 6: E2 — Orsolic-style multi-lag lick-hazard filter (`hazard_filter_tau`)

**Files:**
- Modify: `src/visdetect/analysis/integration_timescale.py` (append `build_lagged_hazard_design`, `fit_lagged_hazard`, `hazard_filter_tau`)
- Test: `tests/analysis/test_integration_timescale.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from visdetect.analysis.integration_timescale import fit_lagged_hazard, hazard_filter_tau


@pytest.mark.slow
def test_hazard_filter_recovers_tau_and_is_autocorr_robust():
    # COLOURED stimulus (rho=0.6): a raw kernel would be biased, but the regression-based
    # hazard filter recovers tau (it is autocorrelation-corrected by construction).
    df = simulate_leaky_integrator_fas(tau_true=0.30, n_trials=5000, white=False, rho=0.6,
                                       thresh=1.0, gain=0.5, seed=7)
    w = fit_lagged_hazard(df, "all_fa", n_lags=20, dt=0.05, t0=0.0)
    assert w is not None and w.shape[0] == 20
    tau = hazard_filter_tau(w, dt=0.05, method="window")
    assert 0.10 < tau < 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `py -m pytest tests/analysis/test_integration_timescale.py -k hazard_filter_recovers -v -m slow`
Expected: FAIL — `ImportError: cannot import name 'fit_lagged_hazard'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to integration_timescale.py
def build_lagged_hazard_design(ev_df, lick_set: str, n_lags: int, dt: float = DT,
                               state_labels: Optional[pd.Series] = None,
                               t0: float = REFRACTORY_S, sensory_latency: float = 0.05,
                               max_time: float = 25.0):
    """At-risk discrete-time hazard rows with a lagged-evidence design (Orsolic-style).

    Each row = one at-risk dt-bin from n_lags..(lick - t0). X = [e(t), e(t-dt), ...,
    e(t-(n_lags-1)dt)]; y = 1 at the lick bin else 0; plus the bin time (nuisance)."""
    sel = select_licks(ev_df, lick_set, state_labels=state_labels)
    Xs, ys, ts = [], [], []
    for _, r in sel.iterrows():
        e = np.asarray(r["evidence"], float)
        end_t = min(r["lick_time"] - t0, max_time)
        nb = int(end_t / dt)
        for b in range(n_lags, nb):
            t = b * dt
            if r["outcome"] == "hit" and np.isfinite(r["change_time"]) \
                    and t < (r["change_time"] + sensory_latency):
                continue
            lagvec = e[b - n_lags + 1: b + 1][::-1]               # lag0 (current) .. lag n_lags-1
            if lagvec.size < n_lags:
                continue
            Xs.append(lagvec); ys.append(1.0 if b == nb - 1 else 0.0); ts.append(t)
    return np.asarray(Xs, float), np.asarray(ys, float), np.asarray(ts, float)


def fit_lagged_hazard(ev_df, lick_set: str, n_lags: int = None, dt: float = DT,
                      state_labels: Optional[pd.Series] = None, t0: float = REFRACTORY_S,
                      reg_C: float = 1.0):
    """Orsolic-style lick-hazard logistic regression; return the n_lags stimulus filter.

    n_lags defaults to HAZARD_WINDOW_S/dt (= 50 lags = 2.5 s, Orsolic's history window).
    A time-in-trial nuisance (linear+quadratic, ~ Orsolic's monotonic time warp) absorbs the
    temporal/urgency clock. L2 regularisation ~ Orsolic's ARD shrinkage. Regression on the
    full lag set is inherently autocorrelation-corrected (spec §6)."""
    if n_lags is None:
        n_lags = int(round(HAZARD_WINDOW_S / dt))
    X, y, tcol = build_lagged_hazard_design(ev_df, lick_set, n_lags, dt, state_labels, t0)
    if X.shape[0] < n_lags + 5 or y.sum() < 3:
        return None
    from sklearn.linear_model import LogisticRegression
    tn = (tcol - tcol.mean()) / (tcol.std() + 1e-9)
    Xn = np.column_stack([X, tn, tn ** 2])
    clf = LogisticRegression(C=reg_C, max_iter=5000)
    clf.fit(Xn, y)
    return clf.coef_.ravel()[:n_lags]


def hazard_filter_tau(lag_weights, dt: float = DT, method: str = "window") -> float:
    """E2 tau from the fitted hazard filter. Default 'window' = the integration extent
    (90% cumulative weight) ~ Orsolic's long integration; 'exp' for the decay constant."""
    lags = (np.arange(len(lag_weights)) + 1) * dt
    return tau_from_profile(np.asarray(lag_weights, float), lags, method=method)
```

- [ ] **Step 4: Run to verify pass**

Run: `py -m pytest tests/analysis/test_integration_timescale.py -k hazard_filter_recovers -v -m slow`
Expected: PASS (a real logistic fit; ~30 s).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/integration_timescale.py tests/analysis/test_integration_timescale.py
git commit -m "feat(B1): E2 Orsolic-style multi-lag lick-hazard filter + hazard_filter_tau"
```

---

### Task 7: `triangulate_tau` — per-stage τ for all three estimators + bootstrap Δτ

**Files:**
- Modify: `src/visdetect/analysis/integration_timescale.py` (append `_stage_taus`, `_bootstrap_delta`, `triangulate_tau`)
- Test: `tests/analysis/test_integration_timescale.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from visdetect.analysis.integration_timescale import triangulate_tau


@pytest.mark.slow
def test_triangulate_detects_planted_learning_increase():
    learn = simulate_leaky_integrator_fas(tau_true=0.20, n_trials=2500, white=True, seed=10)
    expert = simulate_leaky_integrator_fas(tau_true=0.45, n_trials=2500, white=True, seed=11)
    res = triangulate_tau({"Learning": learn, "Expert": expert},
                          t0=0.0, dt=0.05, max_lag=1.2, n_boot=200, seed=0,
                          estimators=("kernel", "leaky"))         # skip slow hazard in the test
    assert res["per_stage"]["Expert"]["tau_kernel"] > res["per_stage"]["Learning"]["tau_kernel"]
    assert res["delta"]["tau_kernel"]["estimate"] > 0
    assert res["delta"]["tau_kernel"]["ci_low"] > 0              # CI excludes 0
```

- [ ] **Step 2: Run to verify failure**

Run: `py -m pytest tests/analysis/test_integration_timescale.py -k triangulate_detects -v -m slow`
Expected: FAIL — `ImportError: cannot import name 'triangulate_tau'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to integration_timescale.py
_ALL_ESTIMATORS = ("kernel", "leaky", "hazard")


def _stage_taus(ev_df, t0, dt, max_lag, state_labels=None, lick_set="engaged_fa",
                estimators=_ALL_ESTIMATORS, kernel_method="exp") -> dict:
    """Point estimates for one stage. Falls back to all-FA if engaged set is empty."""
    use = lick_set
    if select_licks(ev_df, lick_set, state_labels=state_labels).shape[0] < 5 and lick_set == "engaged_fa":
        use = "all_fa"
    out = {"n_licks": int(select_licks(ev_df, use, state_labels=state_labels).shape[0])}
    if "kernel" in estimators:
        n_lags = int(round(max_lag / dt))
        segs, lags, _ = collect_lick_segments(ev_df, use, t0, max_lag, dt, state_labels)
        if segs.shape[0] >= 5:
            C = stimulus_autocov(ev_df, n_lags, dt)
            out["tau_kernel"] = kernel_tau(lick_triggered_kernel(segs, autocov=C), lags,
                                           method=kernel_method)
        else:
            out["tau_kernel"] = np.nan
    if "leaky" in estimators:
        out["tau_leaky"] = leaky_integrator_tau(ev_df, use, dt, t0=t0,
                                                state_labels=state_labels)["tau"]
    if "hazard" in estimators:
        w = fit_lagged_hazard(ev_df, use, dt=dt, state_labels=state_labels, t0=t0)
        out["tau_hazard"] = hazard_filter_tau(w, dt) if w is not None else np.nan
    return out


def _bootstrap_delta(ev_lo, ev_hi, key, t0, dt, max_lag, st_lo, st_hi, estimators,
                     n_boot, seed) -> dict:
    rng = np.random.default_rng(seed)
    est = {"tau_kernel": "kernel", "tau_leaky": "leaky", "tau_hazard": "hazard"}[key]
    deltas = []
    for _ in range(n_boot):
        bl = ev_lo.sample(len(ev_lo), replace=True, random_state=int(rng.integers(1 << 31)))
        bh = ev_hi.sample(len(ev_hi), replace=True, random_state=int(rng.integers(1 << 31)))
        tl = _stage_taus(bl, t0, dt, max_lag, st_lo, estimators=(est,)).get(key, np.nan)
        th = _stage_taus(bh, t0, dt, max_lag, st_hi, estimators=(est,)).get(key, np.nan)
        if np.isfinite(tl) and np.isfinite(th):
            deltas.append(th - tl)
    d = np.asarray(deltas, float)
    if d.size == 0:
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_boot": 0}
    return {"estimate": float(np.median(d)), "ci_low": float(np.percentile(d, 2.5)),
            "ci_high": float(np.percentile(d, 97.5)), "n_boot": int(d.size)}


def triangulate_tau(stage_data: Dict[str, pd.DataFrame], t0: float = REFRACTORY_S,
                    dt: float = DT, max_lag: float = LTA_WINDOW_S, n_boot: int = 1000,
                    seed: int = 42, state_by_stage: Optional[Dict[str, pd.Series]] = None,
                    estimators=_ALL_ESTIMATORS) -> dict:
    """Per-stage τ for the requested estimators + bootstrap Δτ between the first and last
    stage (chronological dict order). state_by_stage = the pluggable state input."""
    state_by_stage = state_by_stage or {}
    stages = list(stage_data)
    per_stage = {s: _stage_taus(stage_data[s], t0, dt, max_lag, state_by_stage.get(s),
                                estimators=estimators) for s in stages}
    out = {"per_stage": per_stage, "delta": {}}
    keys = {"kernel": "tau_kernel", "leaky": "tau_leaky", "hazard": "tau_hazard"}
    if len(stages) >= 2:
        lo, hi = stages[0], stages[-1]
        for est in estimators:
            out["delta"][keys[est]] = _bootstrap_delta(
                stage_data[lo], stage_data[hi], keys[est], t0, dt, max_lag,
                state_by_stage.get(lo), state_by_stage.get(hi), estimators, n_boot, seed)
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `py -m pytest tests/analysis/test_integration_timescale.py -k triangulate_detects -v -m slow`
Expected: PASS.

- [ ] **Step 5: Run the full module test suite**

Run: `py -m pytest tests/analysis/test_integration_timescale.py -v -m "slow or not slow"`
Expected: PASS (Tasks 2–7). Fast subset (`-m "not slow"`) passes in seconds.

- [ ] **Step 6: Commit**

```bash
git add src/visdetect/analysis/integration_timescale.py tests/analysis/test_integration_timescale.py
git commit -m "feat(B1): triangulate_tau — per-stage 3-estimator tau + bootstrap delta-tau"
```

---

### Task 8: Analysis script (manifest loop → per-stage τ → figure + caches)

**Files:**
- Create: `analysis_suite/01_behavior/i_integration_timescale.py`

- [ ] **Step 1: Write the script**

```python
"""Fig0N (B1): Is the evidence-integration timescale a learned quantity?

Three faithful behavioural estimators of tau per stage — a model-free lick-triggered TF
kernel (both papers), Khilkevich's leaky-integrator-to-threshold (~0.27 s), and an
Orsolic-style multi-lag lick-hazard filter (~1 s) — tested for a learning increase,
triangulated, and state-controlled. The estimator spread on the SAME data also adjudicates
the 0.27 s vs ~1 s field discrepancy (estimator/readout artifact vs biology).

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

from visdetect.suite.config import STAGE_ORDER, CACHE_DIR
from visdetect.suite.loader import load_session
from visdetect.suite.plotting import setup_style, save_figure
from visdetect.analysis.config import load_staging_manifest
from visdetect.analysis.behavior import filter_manifest_by_stage
from visdetect.analysis import integration_timescale as it

setup_style()
CACHE = os.path.join(CACHE_DIR, "integration_timescale_taus.csv")

# From Task 1 (docs/science/B1_stimulus_characterization.md); override here if refined.
DT, T0, MAX_LAG = 0.05, 0.15, it.LTA_WINDOW_S
LIT_BRACKET = (0.27, 1.0)          # Khilkevich vs Orsolic


def build_stage_data(state_control=True):
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
            ev["trial_idx"] = ev["trial_idx"] + sname * 100000    # globally unique
            if state_control:
                try:
                    st = it.load_state_labels(sname); st.index = st.index + sname * 100000
                    states.append(st)
                except Exception:
                    pass
            frames.append(ev); del sess; gc.collect()
        if frames:
            stage_data[stage] = pd.concat(frames, ignore_index=True)
            if states:
                state_by_stage[stage] = pd.concat(states)
    return stage_data, state_by_stage


def main():
    print("[01i] B1 integration timescale (3 estimators)...")
    stage_data, state_by_stage = build_stage_data(state_control=True)
    if len(stage_data) < 2:
        print("  Need >=2 stages. Exiting."); return

    res = it.triangulate_tau(stage_data, t0=T0, dt=DT, max_lag=MAX_LAG, n_boot=1000,
                             state_by_stage=state_by_stage)       # all three estimators
    for s, d in res["per_stage"].items():
        print(f"  {s}: kernel={d.get('tau_kernel', np.nan):.3f} "
              f"leaky={d.get('tau_leaky', np.nan):.3f} hazard={d.get('tau_hazard', np.nan):.3f}"
              f" (n_licks={d['n_licks']})")
    for key, d in res["delta"].items():
        print(f"  Δ{key} = {d['estimate']:.3f} [{d['ci_low']:.3f},{d['ci_high']:.3f}]")

    # ── figure (panels A–F per spec §8) ──────────────────────────────────
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.32)
    # Panel C (fully coded): tau-by-stage for all three estimators + literature bracket
    axc = fig.add_subplot(gs[0, 2])
    stages = list(res["per_stage"])
    x = np.arange(len(stages))
    for key, lab, mk in [("tau_kernel", "E1a kernel", "o-"),
                         ("tau_leaky", "E1b leaky-int (~Khilkevich)", "s--"),
                         ("tau_hazard", "E2 hazard (~Orsolic)", "^:")]:
        axc.plot(x, [res["per_stage"][s].get(key, np.nan) for s in stages], mk, label=lab)
    axc.axhspan(*LIT_BRACKET, color="gray", alpha=0.15, label=f"lit {LIT_BRACKET[0]}–{LIT_BRACKET[1]} s")
    axc.set_xticks(x); axc.set_xticklabels(stages); axc.set_ylabel("integration tau (s)")
    axc.set_title("C. tau by stage (3 estimators, triangulated)"); axc.legend(fontsize=7)
    # Panels A (per-stage LTA kernels), B (per-stage hazard filters), D (Δτ bootstrap hist
    # per estimator w/ CI), E (estimator-spread = method-artifact panel vs LIT_BRACKET),
    # F (state_control=False rerun overlaid) — fill from `res` + re-called per-stage profiles.
    save_figure(fig, "fig0N_integration_timescale", "01_behavior")

    rows = [{"stage": s, **d} for s, d in res["per_stage"].items()]
    rows += [{"stage": f"DELTA_{k}", **d} for k, d in res["delta"].items()]
    pd.DataFrame(rows).to_csv(CACHE, index=False)
    print(f"  saved {CACHE}")


if __name__ == "__main__":
    main()
```

> Panels A/B/D/E/F are sketched (C complete). Fill from `res` + per-stage profiles (re-call `collect_lick_segments`/`lick_triggered_kernel` and `fit_lagged_hazard` for plotting), a Δτ bootstrap histogram, the estimator-spread method-artifact panel, and a `state_control=False` rerun. Keep each ≤20 lines.

- [ ] **Step 2: Verify the script imports cleanly**

Run: `py -c "import importlib.util as u; s=u.spec_from_file_location('b1','analysis_suite/01_behavior/i_integration_timescale.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print('import OK')"`
Expected: `import OK`.

- [ ] **Step 3: Commit**

```bash
git add analysis_suite/01_behavior/i_integration_timescale.py
git commit -m "feat(B1): fig0N script — per-stage integration timescale, 3 estimators"
```

---

### Task 9: Run on real data, record result, update index

**Files:**
- Modify: `docs/science/QUESTION_INDEX.md`

- [ ] **Step 1: Pre-flight (uses Task 1's resolved numbers)**

Confirm `docs/science/B1_stimulus_characterization.md` exists and `DT/T0` match it. Spot-check one session:

Run: `py -c "from visdetect.analysis.config import load_staging_manifest; from visdetect.suite.loader import load_session; from visdetect.analysis.integration_timescale import build_evidence_traces, select_licks; m=load_staging_manifest(qc_only=True); s=load_session(int(m.iloc[-1]['session_name'])); ev=build_evidence_traces(s); print('noisy_trials',len(ev),'engaged_fa',len(select_licks(ev,'engaged_fa')),'hits',len(select_licks(ev,'hit')))"`
Expected: a sane count of noisy trials, a non-trivial number of engaged FAs (≥2 s). If engaged FAs per stage are too few for a stable kernel/fit (spec §6/§9), pool to two stages and widen CIs.

- [ ] **Step 2: Run the analysis**

Run: `cd analysis_suite && py 01_behavior/i_integration_timescale.py`
Expected: prints per-stage kernel/leaky/hazard τ and Δτ with CIs; saves figure + stats + cache. (Slow — bootstrap × leaky-grid × hazard-fit.)

- [ ] **Step 3: Sanity-check against the spec §7 success criteria**

Open the figure. Confirm: (a) `Δτ > 0` with CI excluding 0 on the PRIMARY estimator, **corroborated in sign** by the others → "τ is learned"; or a clean null/inconclusive honestly reported. (b) Where do the three estimators land vs the 0.27–1.0 s bracket — does the **estimator spread itself reproduce the ~3–4× field gap** (estimator/readout artifact, not biology)? (c) Does the learning trend survive `state_control=True` vs `False`?

- [ ] **Step 4: Update the question index**

Set the B1 row status in `docs/science/QUESTION_INDEX.md` to `done` (or `in-progress` if state-control/E3/per-stage-data forced caveats):

```
| B1 ⭐ | Is the evidence-integration timescale a *learned* quantity? | T1 | done | [design](../superpowers/specs/2026-06-10-B1-integration-timescale-learned-design.md) | [plan](../superpowers/plans/2026-06-10-B1-integration-timescale-learned-plan.md) |
```

- [ ] **Step 5: Commit**

```bash
git add docs/science/QUESTION_INDEX.md analysis_suite/cache/integration_timescale_taus.csv
git commit -m "data(B1): integration-timescale result + index update"
```

(If `analysis_suite/figures/**` is gitignored, omit it from the add.)

---

## Self-Review

**1. Spec coverage:**
- §3 inputs (pre-planned TF stream → `e(t)`, truncated at lick; pluggable state) → Task 2 + Task 3. **Strengthened:** no-noise trials excluded; engaged-FA has a classifier-free ≥2 s default (Khilkevich). ✓
- §4 multiple estimators, triangulated → **three faithful estimators** (E1a kernel Task 4, E1b leaky-integrator Task 5, E2 hazard filter Task 6) — replacing the spec's vaguer E1/E2 mapping with the papers' actual methods (resolutions box #2). ✓
- §4/§9 E3 (DDM λ) deferred post-B0; E1b is the no-pyddm stand-in until then. ✓
- §5 per-stage τ + bootstrap Δτ + decision rule (primary CI excludes 0, others sign-agree) → Task 7. ✓
- §5 method-artifact / field-discrepancy test → now the **estimator spread vs the 0.27–1.0 s bracket** (Task 8 panel E + Task 9 Step 3b) — sharper than the original 2-estimator framing. ✓
- §5/§6 state control → `state_by_stage` plumbing + `state_control` rerun. ✓
- §6 stimulus-autocorrelation confound → **reframed**: white-by-design (both papers) ⇒ whitening near-identity; Task 1 verifies; kept for honesty (Task 4 deterministic test). The real adjudicator is estimator class + readout. ✓
- §6 non-decision-time truncation → `t0=0.15 s` everywhere (Task 3/5/6) + Task 1 estimate. ✓
- §7 criteria → Task 9 Step 3. §8 deliverables → Tasks 2–8 (figure panels A/B/D/E/F sketched, C complete — flagged). §9 BLOCKING items → Task 1 + resolutions box. ✓

**2. Placeholder scan:** All estimator/extractor/triangulation units are complete real code. **Honestly flagged:** figure panels A/B/D/E/F sketched with fill instructions; `dt/t0` Task-1-resolved + threaded; E2 is a *tractable analog* of Orsolic's GP model (not a reimplementation — their code is released); evidence extractor duplicated vs B0 (reconcile at E3); leaky-integrator threshold-scan is a coarse grid (refine range from Task 1's SD if needed). No vague TODOs.

**3. Type consistency:** `build_evidence_traces → ev_df[evidence, lick_time, outcome, change_time, trial_idx, noisy, baseline_std]` feeds `select_licks` → `collect_lick_segments` / `leaky_integrator_tau` / `build_lagged_hazard_design`, all consuming the same schema. `tau_from_profile(profile, lags, method)` is the single readout shared by `kernel_tau` and `hazard_filter_tau` (methods `exp`/`half_area`/`window` consistent). `_stage_taus` emits `{tau_kernel, tau_leaky, tau_hazard, n_licks}`; `triangulate_tau`/`_bootstrap_delta` key off `tau_*` consistently; the script reads the same keys. ✓

**Statistician knobs (flag for review at planning/exec):** which estimator is the *primary* for the headline Δτ (recommend E1a kernel — model-free); τ readout per estimator (`exp` vs `window` — the choice itself drives 0.27-vs-1 s, so report all); leaky-integrator grid range + `predict_window`; hazard `reg_C`/n_lags; bootstrap unit (trial vs session-block — session-block is more honest for n=1 across sessions; current is trial-level); min engaged-FA licks per stage; merge-Naive vs three stages.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-10-B1-integration-timescale-learned-plan.md`. Like B0/B2 this is **parked for a fresh chat** (open the B1 spec + this plan). Two options when picked up:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between. **Run Task 1 first** — its measured `dt`/white-verdict/`t0`/no-noise-fraction configure Tasks 2/4/8.
2. **Inline Execution** — execute in-session with checkpoints.

No new dependency, so a dedicated worktree is optional (lighter than B0). Which approach?
