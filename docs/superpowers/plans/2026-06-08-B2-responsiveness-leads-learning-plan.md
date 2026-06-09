# B2 — Striatal responsiveness leads learning: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether BG_046's striatal *baseline TF-pulse* sensory encoding rises earlier in training than the behavioral d′ curve (t50_neural < t50_behavior), per `docs/superpowers/specs/2026-06-08-B2-responsiveness-leads-learning-design.md`.

**Architecture:** All scientific computation goes in a new pure-ish library module `visdetect/analysis/learning_trajectory.py` (unit-tested on numpy arrays + planted-signal fixtures). A thin `analysis_suite` script composes it over the staging-manifest session loop, computes the behavioral curve from the manifest `d_prime` column, fits sigmoids, runs the block-bootstrap lead test, and renders the figure + caches. The neural measure is the **motor-free baseline TF-pulse** response (decode fast-vs-slow pulse identity) — never change-aligned activity (spec §4.3).

**Tech Stack:** numpy, scipy (`curve_fit`), scikit-learn (`LogisticRegression`, `StratifiedKFold`), pandas, matplotlib. Reuses `visdetect.analysis.tf_pulse` (`_collect_pulses`, `collect_tf_pulse_traces`, `detrend_tf_traces`, `TFRespPulseConfig`), `visdetect.analysis.constants`, `visdetect.suite.loader/config/plotting`.

---

## File Structure

- **Create** `src/visdetect/analysis/learning_trajectory.py` — all B2 computation: per-pulse population matrix, CV pulse-identity decoding, learning-curve (sigmoid/t50) fit, block-bootstrap lead test, first-difference cross-correlation, the `compute_lead_result` orchestrator, and the per-session `session_neural_measures` wrapper. One responsibility: turn sessions + a per-session table into a lead-lag result.
- **Create** `tests/analysis/test_learning_trajectory.py` — unit tests (planted-signal fixtures + synthetic-session smoke test).
- **Create** `analysis_suite/05_longitudinal/d_responsiveness_leads_learning.py` — orchestration: manifest loop → per-session neural measures + d′ + day axis → `compute_lead_result` → figure (`fig22b`) + stats CSV + per-session cache CSV. (Sibling of the existing `a_neural_learning_curves.py`; deliberately uses the TF-pulse measure, not Fig21's change-aligned `frac_responsive`.)
- **Modify** `docs/science/QUESTION_INDEX.md` — add the plan link, bump B2 status.

Conventions enforced (from `CLAUDE.md`): import constants from `visdetect.analysis.constants`; sessions via `load_staging_manifest()` / `load_session()`; `del sess; gc.collect()` per session; `setup_style()` / `save_figure()`; `py` not `python`.

---

### Task 1: Module scaffold + per-pulse population matrix

**Files:**
- Create: `src/visdetect/analysis/learning_trajectory.py`
- Test: `tests/analysis/test_learning_trajectory.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_learning_trajectory.py
import numpy as np
import pytest
from visdetect.analysis.learning_trajectory import pulse_response_matrix


def test_pulse_response_matrix_planted_signal():
    # unit 0 fires 0.1 s after every FAST pulse (in post window 0..0.5);
    # unit 1 is silent. Expect unit-0 response higher on fast than slow.
    fast_times = np.array([10.0, 20.0, 30.0, 40.0])
    slow_times = np.array([15.0, 25.0, 35.0, 45.0])
    u0 = np.sort(fast_times + 0.1)            # responds to fast only
    u1 = np.array([], dtype=float)            # silent
    X, y = pulse_response_matrix([u0, u1], fast_times, slow_times)

    assert X.shape == (8, 2)
    assert set(np.unique(y)) == {0, 1}
    fast_resp = X[y == 1, 0].mean()
    slow_resp = X[y == 0, 0].mean()
    assert fast_resp > slow_resp
    assert np.allclose(X[:, 1], 0.0)          # silent unit
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_learning_trajectory.py::test_pulse_response_matrix_planted_signal -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'visdetect.analysis.learning_trajectory'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/visdetect/analysis/learning_trajectory.py
"""B2 — does striatal baseline TF-pulse encoding lead the behavioral learning curve?

Pure-ish computation for the lead-lag analysis. The neural measure is the
MOTOR-FREE baseline TF-pulse response (decode fast-vs-slow pulse identity),
never change-aligned activity (see the B2 design spec, sec 4.3).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from visdetect.analysis.constants import (
    TF_PULSE_PRE_WINDOW,
    TF_PULSE_POST_WINDOW,
    DEFAULT_Z_THRESH_TF,
)


def _rate_in_window(spikes: np.ndarray, centers: np.ndarray,
                    w0: float, w1: float) -> np.ndarray:
    """Per-center firing rate (Hz) in [center+w0, center+w1). `spikes` sorted."""
    if spikes.size == 0:
        return np.zeros(centers.size, dtype=float)
    lo = np.searchsorted(spikes, centers + w0, side="left")
    hi = np.searchsorted(spikes, centers + w1, side="left")
    dur = max(w1 - w0, 1e-9)
    return (hi - lo).astype(float) / dur


def pulse_response_matrix(
    spike_times_by_unit: Sequence[np.ndarray],
    fast_times: np.ndarray,
    slow_times: np.ndarray,
    pre_window: Tuple[float, float] = TF_PULSE_PRE_WINDOW,
    post_window: Tuple[float, float] = TF_PULSE_POST_WINDOW,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pulse, per-unit baseline-subtracted response matrix.

    Returns
    -------
    X : (n_pulses, n_units) float — post-window rate minus pre-window rate.
    y : (n_pulses,) int — 1 = fast pulse, 0 = slow pulse.
    """
    fast_times = np.asarray(fast_times, dtype=float)
    fast_times = fast_times[np.isfinite(fast_times)]
    slow_times = np.asarray(slow_times, dtype=float)
    slow_times = slow_times[np.isfinite(slow_times)]

    pulse_times = np.concatenate([fast_times, slow_times])
    y = np.concatenate([
        np.ones(fast_times.size, dtype=int),
        np.zeros(slow_times.size, dtype=int),
    ])
    n_units = len(spike_times_by_unit)
    X = np.zeros((pulse_times.size, n_units), dtype=float)
    if pulse_times.size == 0:
        return X, y
    for u, st in enumerate(spike_times_by_unit):
        st = np.sort(np.asarray(st, dtype=float))
        post = _rate_in_window(st, pulse_times, post_window[0], post_window[1])
        pre = _rate_in_window(st, pulse_times, pre_window[0], pre_window[1])
        X[:, u] = post - pre
    return X, y
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_learning_trajectory.py::test_pulse_response_matrix_planted_signal -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/learning_trajectory.py tests/analysis/test_learning_trajectory.py
git commit -m "feat(B2): per-pulse population response matrix (motor-free TF-pulse)"
```

---

### Task 2: Cross-validated pulse-identity decoding (the PRIMARY neural measure)

**Files:**
- Modify: `src/visdetect/analysis/learning_trajectory.py` (append `decode_pulse_identity`)
- Test: `tests/analysis/test_learning_trajectory.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from visdetect.analysis.learning_trajectory import decode_pulse_identity


def test_decode_pulse_identity_separable_vs_random():
    rng = np.random.default_rng(0)
    n_per, n_units = 60, 8
    y = np.r_[np.ones(n_per, int), np.zeros(n_per, int)]

    # Separable: fast pulses have +1 mean shift across units.
    X_sep = rng.normal(0, 1, (2 * n_per, n_units))
    X_sep[y == 1] += 1.0
    res = decode_pulse_identity(X_sep, y, n_shuffle=50, n_repeats=5, seed=1)
    assert res["ok"] is True
    assert res["acc"] > 0.75
    assert res["p"] < 0.05

    # Random: no signal -> chance accuracy, non-significant.
    X_rand = rng.normal(0, 1, (2 * n_per, n_units))
    res2 = decode_pulse_identity(X_rand, y, n_shuffle=50, n_repeats=5, seed=2)
    assert 0.35 < res2["acc"] < 0.65
    assert res2["p"] > 0.05


def test_decode_pulse_identity_guards_too_few():
    X = np.random.default_rng(0).normal(0, 1, (4, 8))
    y = np.array([1, 1, 0, 0])           # 2 per class < min_per_class
    res = decode_pulse_identity(X, y)
    assert res["ok"] is False
    assert np.isnan(res["acc"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_learning_trajectory.py -k decode_pulse_identity -v`
Expected: FAIL — `ImportError: cannot import name 'decode_pulse_identity'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to learning_trajectory.py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _cv_accuracy(X: np.ndarray, y: np.ndarray, n_folds: int, random_state: int) -> float:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return float(np.mean(cross_val_score(clf, X, y, cv=skf, scoring="accuracy")))


def decode_pulse_identity(
    X: np.ndarray,
    y: np.ndarray,
    n_sub: Optional[int] = None,
    n_repeats: int = 20,
    n_folds: int = 5,
    n_shuffle: int = 200,
    seed: int = 42,
    min_per_class: int = 10,
    min_units: int = 2,
) -> Dict[str, float]:
    """Fixed-n-unit-subsampled, CV decoding of fast-vs-slow pulse identity,
    with a label-shuffle null. Returns acc, null summary, permutation p, and ok flag."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n_pulses, n_units = X.shape
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2 or counts.min() < min_per_class or n_units < min_units:
        return {"acc": np.nan, "null_mean": np.nan, "null_sd": np.nan, "p": np.nan,
                "n_units_used": int(n_units), "n_pulses": int(n_pulses), "ok": False}

    rng = np.random.default_rng(seed)
    k = n_units if (n_sub is None or n_sub >= n_units) else int(n_sub)

    def _one(Xv, yv):
        cols = np.arange(n_units) if k >= n_units else rng.choice(n_units, size=k, replace=False)
        return _cv_accuracy(Xv[:, cols], yv, n_folds, int(rng.integers(1_000_000_000)))

    acc = float(np.mean([_one(X, y) for _ in range(n_repeats)]))
    null = np.array([_one(X, rng.permutation(y)) for _ in range(n_shuffle)], dtype=float)
    p = float((np.sum(null >= acc) + 1) / (n_shuffle + 1))
    return {"acc": acc, "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "p": p, "n_units_used": int(k), "n_pulses": int(n_pulses), "ok": True}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_learning_trajectory.py -k decode_pulse_identity -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/learning_trajectory.py tests/analysis/test_learning_trajectory.py
git commit -m "feat(B2): CV fast/slow pulse-identity decoder with subsample + shuffle null"
```

---

### Task 3: Learning-curve (sigmoid) fit and t50

**Files:**
- Modify: `src/visdetect/analysis/learning_trajectory.py` (append `fit_learning_curve`, `_logistic4`)
- Test: `tests/analysis/test_learning_trajectory.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from visdetect.analysis.learning_trajectory import fit_learning_curve, _logistic4


def test_fit_learning_curve_recovers_t50():
    t = np.linspace(0, 40, 21)
    y = _logistic4(t, y0=0.5, L=0.4, k=0.3, t50=22.0)
    y = y + np.random.default_rng(0).normal(0, 0.005, t.size)
    res = fit_learning_curve(t, y)
    assert res["success"] is True
    assert abs(res["t50"] - 22.0) < 3.0


def test_fit_learning_curve_too_few_points():
    res = fit_learning_curve(np.array([0.0, 1.0, 2.0]), np.array([0.1, 0.2, 0.3]))
    assert res["success"] is False
    assert np.isnan(res["t50"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_learning_trajectory.py -k fit_learning_curve -v`
Expected: FAIL — `ImportError: cannot import name 'fit_learning_curve'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to learning_trajectory.py
from scipy.optimize import curve_fit


def _logistic4(t, y0, L, k, t50):
    """4-parameter logistic: y0 + L / (1 + exp(-k (t - t50)))."""
    return y0 + L / (1.0 + np.exp(-k * (np.asarray(t, dtype=float) - t50)))


def fit_learning_curve(t: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Fit a monotonic 4-param logistic; return t50 (inflection = steepest rise)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    fail = {"t50": np.nan, "y0": np.nan, "L": np.nan, "k": np.nan, "success": False}
    if t.size < 4 or np.ptp(t) == 0:
        return fail
    y0_0 = float(np.min(y))
    L0 = float(np.ptp(y)) or 1.0
    t50_0 = float(np.median(t))
    k0 = 4.0 / (np.ptp(t) or 1.0)
    try:
        popt, _ = curve_fit(
            _logistic4, t, y, p0=[y0_0, L0, k0, t50_0], maxfev=10000,
            bounds=([-np.inf, 0.0, 1e-6, float(t.min())],
                    [np.inf, np.inf, np.inf, float(t.max())]),
        )
        return {"t50": float(popt[3]), "y0": float(popt[0]), "L": float(popt[1]),
                "k": float(popt[2]), "success": True}
    except Exception:
        return fail
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_learning_trajectory.py -k fit_learning_curve -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/learning_trajectory.py tests/analysis/test_learning_trajectory.py
git commit -m "feat(B2): 4-param logistic learning-curve fit + t50 inflection"
```

---

### Task 4: Block-bootstrap lead test (t50_behav − t50_neural)

**Files:**
- Modify: `src/visdetect/analysis/learning_trajectory.py` (append `bootstrap_t50_lead`)
- Test: `tests/analysis/test_learning_trajectory.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from visdetect.analysis.learning_trajectory import bootstrap_t50_lead


def _curve(t, t50):
    return _logistic4(t, y0=0.5, L=0.4, k=0.3, t50=t50)


def test_bootstrap_lead_detects_neural_leads():
    t = np.linspace(0, 40, 25)
    neural = _curve(t, 15.0)   # neural rises early
    behav = _curve(t, 25.0)    # behavior rises late  -> delta = 25-15 > 0
    res = bootstrap_t50_lead(t, neural, behav, n_boot=300, block=3, seed=0)
    assert res["delta_median"] > 0
    assert res["ci_lo"] > 0     # CI excludes zero -> neural leads
    assert res["p_lead"] < 0.05


def test_bootstrap_lead_contemporaneous_includes_zero():
    t = np.linspace(0, 40, 25)
    neural = _curve(t, 20.0)
    behav = _curve(t, 20.0)
    res = bootstrap_t50_lead(t, neural, behav, n_boot=300, block=3, seed=0)
    assert res["ci_lo"] <= 0 <= res["ci_hi"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_learning_trajectory.py -k bootstrap_lead -v`
Expected: FAIL — `ImportError: cannot import name 'bootstrap_t50_lead'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to learning_trajectory.py
def bootstrap_t50_lead(
    t: np.ndarray,
    neural: np.ndarray,
    behav: np.ndarray,
    n_boot: int = 2000,
    block: int = 3,
    seed: int = 42,
) -> Dict[str, float]:
    """Block-bootstrap the lead delta = t50_behav - t50_neural.

    Resamples contiguous session blocks (wrap-around) to respect temporal
    autocorrelation, refits both curves on the SAME resampled time points,
    and accumulates the delta. Positive delta => neural leads behavior.
    """
    t = np.asarray(t, dtype=float)
    neural = np.asarray(neural, dtype=float)
    behav = np.asarray(behav, dtype=float)
    n = t.size
    fail = {"delta_median": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
            "p_lead": np.nan, "n_ok": 0}
    if n < 4:
        return fail
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    deltas: List[float] = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
        tb = t[idx]
        t50_n = fit_learning_curve(tb, neural[idx])["t50"]
        t50_b = fit_learning_curve(tb, behav[idx])["t50"]
        if np.isfinite(t50_n) and np.isfinite(t50_b):
            deltas.append(t50_b - t50_n)
    if not deltas:
        return fail
    d = np.asarray(deltas, dtype=float)
    return {"delta_median": float(np.median(d)),
            "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)),
            "p_lead": float(np.mean(d <= 0.0)),
            "n_ok": int(d.size)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_learning_trajectory.py -k bootstrap_lead -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/learning_trajectory.py tests/analysis/test_learning_trajectory.py
git commit -m "feat(B2): block-bootstrap t50 lead test (neural vs behavioral)"
```

---

### Task 5: First-difference cross-correlation + `compute_lead_result` orchestrator

**Files:**
- Modify: `src/visdetect/analysis/learning_trajectory.py` (append `first_difference_xcorr`, `compute_lead_result`)
- Test: `tests/analysis/test_learning_trajectory.py` (append)

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
from visdetect.analysis.learning_trajectory import (
    first_difference_xcorr, compute_lead_result,
)


def test_first_difference_xcorr_runs():
    t = np.linspace(0, 40, 25)
    neural = _curve(t, 15.0)
    behav = _curve(t, 25.0)
    res = first_difference_xcorr(neural, behav, max_lag=5)
    assert "lag" in res and "corr" in res
    assert -5 <= res["lag"] <= 5


def test_compute_lead_result_end_to_end():
    t = np.linspace(0, 40, 25)
    df = pd.DataFrame({
        "day": t,
        "decod_acc": _curve(t, 15.0),
        "dprime": _curve(t, 25.0),
    })
    res = compute_lead_result(df, day_col="day", neural_col="decod_acc",
                              behav_col="dprime", n_boot=300, seed=0)
    assert res["t50_neural"] < res["t50_behav"]
    assert res["lead"]["ci_lo"] > 0
    assert res["n_sessions"] == 25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_learning_trajectory.py -k "xcorr or compute_lead_result" -v`
Expected: FAIL — `ImportError: cannot import name 'first_difference_xcorr'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to learning_trajectory.py
def first_difference_xcorr(neural: np.ndarray, behav: np.ndarray,
                           max_lag: int = 5) -> Dict[str, object]:
    """Cross-correlate the first differences (trend-removed) of the two curves.

    Positive lag = neural deltas shifted forward to best match behavior =>
    neural changes precede behavioral changes (neural leads)."""
    a = np.diff(np.asarray(neural, dtype=float))
    b = np.diff(np.asarray(behav, dtype=float))
    a = (a - a.mean()) / (a.std() or 1.0)
    b = (b - b.mean()) / (b.std() or 1.0)
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = []
    for L in lags:
        if L < 0:
            x, yv = a[:L], b[-L:]
        elif L > 0:
            x, yv = a[L:], b[:-L]
        else:
            x, yv = a, b
        corrs.append(float(np.corrcoef(x, yv)[0, 1]) if x.size > 1 else np.nan)
    corrs = np.asarray(corrs, dtype=float)
    best = int(np.nanargmax(corrs)) if np.any(np.isfinite(corrs)) else max_lag
    return {"lag": int(lags[best]), "corr": float(corrs[best]),
            "lags": lags.tolist(), "corrs": corrs.tolist()}


def compute_lead_result(
    df,
    day_col: str = "day",
    neural_col: str = "decod_acc",
    behav_col: str = "dprime",
    n_boot: int = 2000,
    block: int = 3,
    seed: int = 42,
) -> Dict[str, object]:
    """Orchestrate the B2 lead analysis on a per-session table.

    Sorts by `day_col`, drops rows missing either curve, fits both learning
    curves, runs the block-bootstrap lead test and the first-difference xcorr."""
    sub = df[[day_col, neural_col, behav_col]].dropna().sort_values(day_col)
    t = sub[day_col].to_numpy(dtype=float)
    neural = sub[neural_col].to_numpy(dtype=float)
    behav = sub[behav_col].to_numpy(dtype=float)
    fit_n = fit_learning_curve(t, neural)
    fit_b = fit_learning_curve(t, behav)
    lead = bootstrap_t50_lead(t, neural, behav, n_boot=n_boot, block=block, seed=seed)
    xcorr = first_difference_xcorr(neural, behav)
    return {"t50_neural": fit_n["t50"], "t50_behav": fit_b["t50"],
            "fit_neural": fit_n, "fit_behav": fit_b,
            "lead": lead, "xcorr": xcorr, "n_sessions": int(t.size)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_learning_trajectory.py -k "xcorr or compute_lead_result" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/learning_trajectory.py tests/analysis/test_learning_trajectory.py
git commit -m "feat(B2): first-difference xcorr + compute_lead_result orchestrator"
```

---

### Task 6: Per-session neural measures wrapper

**Files:**
- Modify: `src/visdetect/analysis/learning_trajectory.py` (append `session_neural_measures`, `_detrended_z_summary`)
- Test: `tests/analysis/test_learning_trajectory.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from visdetect.utils.synthetic import make_synthetic_session
from visdetect.analysis.learning_trajectory import session_neural_measures


def test_session_neural_measures_plumbing():
    # Inject balanced fast/slow pulse times in each trial's baseline period and
    # plant a fast-responsive unit, so the wrapper composes end-to-end.
    sess = make_synthetic_session(n_trials=40, n_clusters=6, seed=3)
    base = np.asarray(sess.ni_events["Baseline_ON"], dtype=float)
    fast_times = base + 0.5
    slow_times = base + 1.0
    # make cluster 0 respond after fast pulses
    sess.clusters[0].spike_times = np.sort(np.r_[sess.clusters[0].spike_times,
                                                  fast_times + 0.1])
    cluster_ids = [c.cluster_id for c in sess.clusters]
    res = session_neural_measures(sess, cluster_ids,
                                  fast_times=fast_times, slow_times=slow_times,
                                  n_shuffle=20, n_repeats=3)
    for key in ("decod_acc", "decod_p", "decod_ok", "mean_abs_z",
                "frac_resp", "n_units", "n_fast", "n_slow"):
        assert key in res
    assert res["n_units"] == 6
    assert res["n_fast"] == base.size and res["n_slow"] == base.size
    assert isinstance(res["decod_acc"], float)  # may be nan but must be float
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_learning_trajectory.py -k session_neural_measures -v`
Expected: FAIL — `ImportError: cannot import name 'session_neural_measures'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to learning_trajectory.py
def _detrended_z_summary(session, cfg, fast_times, slow_times,
                         show_progress: bool = False,
                         z_thresh: float = DEFAULT_Z_THRESH_TF) -> Tuple[float, float]:
    """Mean |detrended z| and fraction responsive, from per-unit TF-pulse traces.

    Reuses the existing tf_pulse machinery (collect_tf_pulse_traces +
    detrend_tf_traces) so the (S)/(C) measures match the project's detrended
    responsiveness definition."""
    from visdetect.analysis.tf_pulse import collect_tf_pulse_traces, detrend_tf_traces
    t_vec, entries = collect_tf_pulse_traces(
        session, cfg, fast_times=fast_times, slow_times=slow_times,
        show_progress=show_progress,
    )
    good = [e for e in entries
            if np.size(getattr(e, "fast_z", [])) == t_vec.size
            and np.size(getattr(e, "slow_z", [])) == t_vec.size]
    if not good:
        return np.nan, np.nan
    fast = np.stack([e.fast_z for e in good], axis=0)
    slow = np.stack([e.slow_z for e in good], axis=0)
    _, zmax_f, zmin_f = detrend_tf_traces(t_vec, fast)
    _, zmax_s, zmin_s = detrend_tf_traces(t_vec, slow)
    zabs = np.nanmax(np.abs(np.vstack([zmax_f, zmin_f, zmax_s, zmin_s])), axis=0)
    return float(np.nanmean(zabs)), float(np.nanmean(zabs > z_thresh))


def session_neural_measures(
    session,
    cluster_ids: Sequence[int],
    cfg=None,
    fast_times: Optional[np.ndarray] = None,
    slow_times: Optional[np.ndarray] = None,
    n_sub: Optional[int] = None,
    n_shuffle: int = 200,
    n_repeats: int = 20,
    seed: int = 42,
    show_progress: bool = False,
) -> Dict[str, float]:
    """Compute the per-session neural sensory-encoding measures (motor-free).

    (P) decod_acc/decod_p : CV fast-vs-slow pulse-identity decoding.
    (S) mean_abs_z         : mean |detrended TF-pulse z| across units.
    (C) frac_resp          : fraction of units with |detrended z| > threshold.
    """
    from visdetect.analysis.tf_pulse import TFRespPulseConfig, _collect_pulses
    if cfg is None:
        cfg = TFRespPulseConfig()
    if fast_times is None or slow_times is None:
        fast_times, slow_times = _collect_pulses(session, cfg, show_progress=show_progress)

    cmap = {int(c.cluster_id): c for c in session.clusters}
    sel = [int(cid) for cid in cluster_ids if int(cid) in cmap]
    spikes = [np.asarray(cmap[cid].spike_times, dtype=float) for cid in sel]

    X, y = pulse_response_matrix(spikes, fast_times, slow_times,
                                 cfg.pre_window, cfg.post_window)
    dec = decode_pulse_identity(X, y, n_sub=n_sub, n_shuffle=n_shuffle,
                                n_repeats=n_repeats, seed=seed)
    mean_abs_z, frac_resp = _detrended_z_summary(
        session, cfg, fast_times, slow_times, show_progress=show_progress)

    return {"decod_acc": float(dec["acc"]), "decod_p": float(dec["p"]),
            "decod_ok": bool(dec["ok"]), "mean_abs_z": mean_abs_z,
            "frac_resp": frac_resp, "n_units": len(sel),
            "n_fast": int(np.sum(y == 1)), "n_slow": int(np.sum(y == 0))}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_learning_trajectory.py -k session_neural_measures -v`
Expected: PASS.

- [ ] **Step 5: Run the full new test module**

Run: `py -m pytest tests/analysis/test_learning_trajectory.py -v`
Expected: PASS (all tasks 1–6).

- [ ] **Step 6: Commit**

```bash
git add src/visdetect/analysis/learning_trajectory.py tests/analysis/test_learning_trajectory.py
git commit -m "feat(B2): per-session neural measures wrapper (decode + detrended z)"
```

---

### Task 7: Analysis script (manifest loop → figure + caches)

**Files:**
- Create: `analysis_suite/05_longitudinal/d_responsiveness_leads_learning.py`

- [ ] **Step 1: Write the script**

```python
"""Fig22b: Does striatal TF-pulse responsiveness LEAD the behavioral learning curve? (B2)

Neural measure = MOTOR-FREE baseline TF-pulse encoding (fast-vs-slow pulse
decodability), NOT change-aligned activity (the post-change ramp is lick-locked
motor prep — see the B2 design spec sec 4.3 and the contrast with 05a Fig21).

Behavioral measure = per-session d' (manifest `d_prime`).
Lead test = t50 inflection comparison with block-bootstrap.

Outputs:
  - analysis_suite/figures/05_longitudinal/fig22b_responsiveness_leads_learning.png
  - analysis_suite/figures/05_longitudinal/responsiveness_leads_stats.csv
  - analysis_suite/cache/responsiveness_leads_per_session.csv
"""
import os
import gc
from datetime import date

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import STAGE_ORDER, STAGE_COLORS, CACHE_DIR
from visdetect.suite.loader import load_session
from visdetect.suite.plotting import setup_style, save_figure
from visdetect.analysis.config import load_staging_manifest, parse_session_date
from visdetect.analysis.utils import get_good_cluster_ids
from visdetect.analysis.learning_trajectory import (
    session_neural_measures, compute_lead_result, _logistic4,
)

setup_style()

PER_SESSION_CACHE = os.path.join(CACHE_DIR, "responsiveness_leads_per_session.csv")
MIN_UNITS = 3


def _days_since_start(session_ints):
    ymd = [parse_session_date(int(s)) for s in session_ints]
    days = [date(y, m, d).toordinal() for (y, m, d) in ymd]
    d0 = min(days)
    return np.array([d - d0 for d in days], dtype=float)


def compute_or_load(force: bool = False) -> pd.DataFrame:
    if os.path.exists(PER_SESSION_CACHE) and not force:
        return pd.read_csv(PER_SESSION_CACHE)

    manifest = load_staging_manifest(qc_only=True)
    manifest["day"] = _days_since_start(manifest["session_name"].tolist())

    rows = []
    for _, mrow in manifest.iterrows():
        sname = int(mrow["session_name"])
        print(f"  Session {sname} ({mrow['stage']})...", end=" ")
        try:
            sess = load_session(sname)
        except FileNotFoundError:
            print("not found")
            continue
        ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
        if len(ids) < MIN_UNITS:
            print("too few units")
            del sess; gc.collect()
            continue
        m = session_neural_measures(sess, ids)
        rows.append({
            "session_name": sname, "stage": mrow["stage"],
            "session_idx": int(mrow["session_idx"]), "day": float(mrow["day"]),
            "dprime": float(mrow["d_prime"]) if "d_prime" in mrow else np.nan,
            **m,
        })
        print(f"acc={m['decod_acc']:.3f} z={m['mean_abs_z']:.2f} "
              f"n={m['n_units']} f/s={m['n_fast']}/{m['n_slow']}")
        del sess; gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(PER_SESSION_CACHE, index=False)
    return df


def main():
    print("[05d] B2: responsiveness-leads-learning...")
    df = compute_or_load()
    if len(df) < 4:
        print(f"  Only {len(df)} sessions — need >=4 for curve fits. Exiting.")
        return

    res = compute_lead_result(df, day_col="day", neural_col="decod_acc",
                              behav_col="dprime", n_boot=2000, block=3, seed=42)
    print(f"  t50_neural={res['t50_neural']:.1f}d  t50_behav={res['t50_behav']:.1f}d")
    print(f"  delta={res['lead']['delta_median']:.1f}d "
          f"CI[{res['lead']['ci_lo']:.1f},{res['lead']['ci_hi']:.1f}] "
          f"p_lead={res['lead']['p_lead']:.3f}")

    sub = df[["day", "decod_acc", "dprime"]].dropna().sort_values("day")
    t = sub["day"].to_numpy(float)
    tt = np.linspace(t.min(), t.max(), 200)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # A: both curves over training days + sigmoid fits + t50 markers
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(sub["day"], sub["decod_acc"], c="tab:blue", s=40, label="neural (decod acc)")
    axb = ax.twinx()
    axb.scatter(sub["day"], sub["dprime"], c="tab:red", s=40, marker="s", label="behavior (d')")
    if res["fit_neural"]["success"]:
        fn = res["fit_neural"]
        ax.plot(tt, _logistic4(tt, fn["y0"], fn["L"], fn["k"], fn["t50"]), c="tab:blue")
        ax.axvline(res["t50_neural"], color="tab:blue", ls="--", alpha=0.6)
    if res["fit_behav"]["success"]:
        fb = res["fit_behav"]
        axb.plot(tt, _logistic4(tt, fb["y0"], fb["L"], fb["k"], fb["t50"]), c="tab:red")
        axb.axvline(res["t50_behav"], color="tab:red", ls="--", alpha=0.6)
    ax.set_xlabel("Training day"); ax.set_ylabel("Fast/slow decode acc", color="tab:blue")
    axb.set_ylabel("Behavioral d'", color="tab:red")
    ax.set_title(f"A. Curves + t50 (neural {res['t50_neural']:.1f} vs behav {res['t50_behav']:.1f} d)")

    # B: bootstrap delta distribution
    ax = fig.add_subplot(gs[0, 1])
    lead = res["lead"]
    ax.axvline(0, color="gray", ls=":")
    ax.axvline(lead["delta_median"], color="k")
    ax.axvspan(lead["ci_lo"], lead["ci_hi"], alpha=0.2, color="tab:green")
    ax.set_xlabel("delta = t50_behav - t50_neural (days)")
    ax.set_title(f"B. Lead delta={lead['delta_median']:.1f} "
                 f"CI[{lead['ci_lo']:.1f},{lead['ci_hi']:.1f}] p_lead={lead['p_lead']:.3f}")

    # C: first-difference cross-correlation
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(res["xcorr"]["lags"], res["xcorr"]["corrs"])
    ax.axvline(0, color="gray", ls=":")
    ax.set_xlabel("lag (sessions)"); ax.set_ylabel("first-diff xcorr")
    ax.set_title(f"C. Trend-removed xcorr (peak lag={res['xcorr']['lag']})")

    # D: secondary measure (mean |detrended z|) over days, by stage
    ax = fig.add_subplot(gs[1, 1])
    for stg in STAGE_ORDER:
        s = df[df["stage"] == stg]
        if len(s):
            ax.scatter(s["day"], s["mean_abs_z"], c=STAGE_COLORS.get(stg, "gray"),
                       s=40, label=stg)
    ax.set_xlabel("Training day"); ax.set_ylabel("Mean |detrended z|")
    ax.set_title("D. Secondary measure (S)"); ax.legend(fontsize=8)

    save_figure(fig, "fig22b_responsiveness_leads_learning", "05_longitudinal")

    stats = pd.DataFrame([{
        "t50_neural": res["t50_neural"], "t50_behav": res["t50_behav"],
        "delta_median": lead["delta_median"], "ci_lo": lead["ci_lo"],
        "ci_hi": lead["ci_hi"], "p_lead": lead["p_lead"], "n_boot_ok": lead["n_ok"],
        "xcorr_lag": res["xcorr"]["lag"], "n_sessions": res["n_sessions"],
    }])
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "05_longitudinal", "responsiveness_leads_stats.csv")
    stats.to_csv(stats_path, index=False)
    print(f"  Saved figure + stats ({stats_path})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script imports cleanly (no execution)**

Run: `py -c "import importlib.util, sys; spec=importlib.util.spec_from_file_location('b2','analysis_suite/05_longitudinal/d_responsiveness_leads_learning.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('import OK')"`
Expected: `import OK` (matplotlib Agg; no figure shown). If an ImportError appears, fix the import path before proceeding.

- [ ] **Step 3: Commit**

```bash
git add analysis_suite/05_longitudinal/d_responsiveness_leads_learning.py
git commit -m "feat(B2): fig22b script — responsiveness-leads-learning (TF-pulse, t50 lead)"
```

---

### Task 8: Run on real data, record result, update the question index

**Files:**
- Modify: `docs/science/QUESTION_INDEX.md`

- [ ] **Step 1: Confirm pkls are present (data-in-hand, T1)**

Run: `py -c "from visdetect.analysis.config import load_staging_manifest; m=load_staging_manifest(qc_only=True); print(len(m), 'sessions; has d_prime:', 'd_prime' in m.columns)"`
Expected: a session count (>=4) and `has d_prime: True`. If `d_prime` is absent, the script's `dprime` column will be NaN — fall back to computing per-session d′ from `sess.trials` (mirror the SDT block in `analysis_suite/05_longitudinal/a_neural_learning_curves.py:104-122`) before re-running.

- [ ] **Step 2: Run the full analysis on real data**

Run: `cd analysis_suite && py 05_longitudinal/d_responsiveness_leads_learning.py`
Expected: per-session progress lines, then a printed `t50_neural / t50_behav / delta / CI / p_lead`, and "Saved figure + stats". First run is slow (per-session pulse extraction + decoding); subsequent runs read the per-session cache.

- [ ] **Step 3: Eyeball the figure and the result**

Open `analysis_suite/figures/05_longitudinal/fig22b_responsiveness_leads_learning.png`. Sanity checks (per spec §7–§8): are per-session `n_fast`/`n_slow` and `n_units` adequate (else the curve is yield-limited → note it); does `decod_acc` sit above 0.5; is the `delta` CI informative or too wide (the honest "consistent with neural-leads, unresolved" outcome)?

- [ ] **Step 4: Update the question index**

In `docs/science/QUESTION_INDEX.md`, set the B2 row's Plan cell to a link and bump Status:

```
| **B2** ⭐ | Does striatal sensory responsiveness *lead* the behavioral learning curve? | T1 | done | [design](../superpowers/specs/2026-06-08-B2-responsiveness-leads-learning-design.md) | [plan](../superpowers/plans/2026-06-08-B2-responsiveness-leads-learning-plan.md) |
```

(Use status `in-progress` if the real-data run still needs the d′ fallback or a yield decision; `done` once the figure + stats are produced and sane.)

- [ ] **Step 5: Commit**

```bash
git add docs/science/QUESTION_INDEX.md analysis_suite/cache/responsiveness_leads_per_session.csv analysis_suite/figures/05_longitudinal/responsiveness_leads_stats.csv
git commit -m "data(B2): responsiveness-leads-learning result + index update"
```

(If `analysis_suite/figures/**` is gitignored — per the recent "ignore figs/" commit — drop the figures path from the `git add` and commit only the cache + index.)

---

## Self-Review

**1. Spec coverage:**
- §4.1 behavioral curve (d′) → Task 7 (manifest `d_prime`), Task 8 Step 1 fallback. ✓
- §4.2 (P) decodability → Tasks 1–2, 6. ✓  (S) mean|z| / (C) frac_resp → Task 6 `_detrended_z_summary`, plotted Task 7 panel D. ✓
- §4.3 motor-free (TF-pulse not change-aligned) → enforced by construction (Task 1 uses `_collect_pulses` baseline pulses; docstrings state it); contrasted with Fig21 in the script docstring. ✓
- §5 lead inference: t50 (Task 3) + block-bootstrap (Task 4) + first-diff xcorr (Task 5). ✓  Granger tertiary = intentionally **descoped** from this plan (spec calls it "suggestive only"); add later if wanted.
- §5 trap avoidance (don't raw-cross-correlate rising curves) → no raw xcorr; only t50 + first-difference. ✓
- §7 yield handling: fixed-n subsample (`n_sub` in `decode_pulse_identity`) ✓; learning-phase binning = **not auto-run** but the per-session cache supports it ad hoc; surfaced in Task 8 Step 3. Depth-control robustness = **descoped** (note for follow-up).
- §6 bonus (B-track tracked cells, B-switch selectivity) → **descoped** from this plan (spec marks them "run only if §5 positive"); they become their own tasks/spec addendum later.
- §8 success criteria → Task 8 Step 3 reads them off `delta`, CI, `p_lead`, `decod_acc`.

**2. Placeholder scan:** No TBD/TODO; every code step has complete code; every command has expected output. Descoped items (Granger, depth control, binning automation, §6 bonuses) are named explicitly, not left as silent gaps.

**3. Type consistency:** `pulse_response_matrix → (X, y)` consumed by `decode_pulse_identity(X, y)` and `session_neural_measures`. `fit_learning_curve` returns `{t50,y0,L,k,success}` used by `bootstrap_t50_lead`, `compute_lead_result`, and the script's `_logistic4(tt, y0, L, k, t50)` call (arg order matches `_logistic4` signature). `compute_lead_result` returns `{t50_neural,t50_behav,fit_neural,fit_behav,lead,xcorr,n_sessions}` — all keys used in Task 7. `session_neural_measures` returns the 8 keys asserted in the Task 6 test and consumed in Task 7's row dict. Consistent. ✓

**Statistician knobs (flagged for review):** block size (`block=3`), n_boot, the 4-param logistic parameterization, `min_per_class=10`, and `n_sub` floor are the levers to confirm with the Research Statistician once real per-session yields and the session-count are known (Task 8 Step 1 prints them).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-08-B2-responsiveness-leads-learning-plan.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
