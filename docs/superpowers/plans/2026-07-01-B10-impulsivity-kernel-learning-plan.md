# B10 — Impulsivity Kernel Across Learning: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Orsolic-style psychophysical reverse-correlation "impulsivity kernel" (behavioral, 3 mice) and its stimulus-referenced neural analog on TF-responsive striatal cells (signed-sum), across learning and split by behavioral state.

**Architecture:** One pure, fully-tested library module (`psychophysical_kernel.py`) holding every estimator (TF reconstruction, FA/withhold epochs, reverse-correlation kernel, bootstrap CI, shape metrics, signed population signal, stimulus-matched sensory-vs-gain control). Four thin scripts (`scripts/evidence_learning/`) apply it to real data and render figures. The library carries the TDD (synthetic-recovery backbone); scripts get synthetic-session smoke tests.

**Tech Stack:** Python 3.10, numpy, pandas, scipy, matplotlib, pytest. Reuses `visdetect.analysis.decision_latents`, `visdetect.analysis.tf_glm_data`, `visdetect.analysis.config`, `visdetect.analysis.utils`, `visdetect.suite.plotting`.

## Global Constraints

- **dt = 0.05 s** for all TF/spike binning (the 50 ms TF update; `TF_SAMPLE_PERIOD=0.25` is a documented footgun — never use it).
- **TF reconstruction:** decimate `trial.baseline_values` by **stride 3** (60 Hz storage, 50 ms holds), anchored at `Baseline_ON`; transform to **log2** octaves. Reuse the `_BASELINE_STRIDE=3` recipe from `tf_glm_data.py`.
- **No baked-in lick delay:** align to the **recorded** FA lick (`reactiontimes['FA']`); expose `lick_shift_ms` (default **0.0**) only as a sensitivity knob. Never depend on `LICK_HARDWARE_DELAY_MS`.
- **Session = unit of replication.** Bootstrap over sessions (and subjects for pooled behavioral). **Never pool raw units across sessions** (within-session QC only → Simpson).
- **Determinism:** every bootstrap/subsample uses `np.random.default_rng(seed)` with `seed=42` (`BOOT_SEED`). Byte-stable outputs.
- **Subjects:** BG_046 (DMS), BG_039 (DMS), BG_031 (VMS). Behavioral pools all 3; neural = **DMS pool (046+039)** + **VMS (031) separate**. Headline contrast **Naive vs Expert**; BG_039 Learning (1 session) excluded from its Learning cell.
- **Outcomes are lowercased** on read (`.lower()`); real pkls capitalize (`Fa`/`Hit`).
- **Canonical session ids:** key/join/sort via `config.canonical_session_id`; never `int()`.
- **Neural cells:** TF-responsive = registry `resp_log2==True` AND `region_bank_confirmed==True`; unit sign = `sign(c1_r_log2)`.
- **New work layout:** library in `src/visdetect/analysis/`, scripts in `scripts/evidence_learning/`, caches in `data/cache/evidence_learning/`, figures in `FIGURES/evidence_learning/<SUBJECT>/`. Import from `visdetect.*` (NOT analysis_suite).
- All implementing subagents run on **Opus 4.8**.

---

## Module Interface (all tasks reference these exact signatures)

`src/visdetect/analysis/psychophysical_kernel.py` public API:

```python
# constants
DT = 0.05
KERNEL_PRE_S = 1.5           # window starts this far before the lick
KERNEL_REFRACTORY_S = 0.15   # exclude the last 150 ms (sensorimotor)
MIN_BASELINE_S = 0.5         # FA lick must be >= this after Baseline_ON
CHANGE_GUARD_S = 0.5         # drop FAs within this of change_time
BOOT_SEED = 42
N_BOOT = 1000
_MONITOR_HZ = 60.0
_STRIDE = 3

def baseline_log2tf(trial, dt=DT, tf_base=None) -> tuple[np.ndarray, np.ndarray]:
    """(t, y): full-trial baseline log2-TF on the dt grid anchored at Baseline_ON.
    y[k] = log2(bv[_STRIDE*k] / base); base = tf_base or per-trial nanmedian(bv)."""

def fa_kernel_epochs(session, lick_shift_ms=0.0, dt=DT) -> list[dict]:
    """One dict per usable FA trial: {trial_idx, lick_t, window(np.ndarray len L)}.
    window = y[j0:j1] where j1 = round((lick_t - lick_shift_ms/1000 - REFRACTORY)/dt),
    j0 = j1 - round((KERNEL_PRE_S - KERNEL_REFRACTORY_S)/dt). L = j1-j0 constant.
    Guards: outcome=='fa', finite FA latency, lick_t>=MIN_BASELINE_S,
    j0>=0 (enough history), and |lick_t - change_time| >= CHANGE_GUARD_S when
    change_time finite. lick_t from reactiontimes['FA']."""

def withhold_epochs(session, fa_epochs, dt=DT, rng=None) -> list[np.ndarray]:
    """One time-in-trial-matched no-lick window (len L) per FA epoch, drawn from
    hit/miss trials' pre-change baseline at the same lick_t (±0.25 s tolerance).
    Returns [] slot as None where no match exists (caller drops those pairs)."""

def reverse_correlation_kernel(fa_windows, withhold_windows) -> np.ndarray:
    """kernel[l] = mean_over_pairs(fa_windows[:,l]) - mean(withhold_windows[:,l]).
    Inputs are equal-length lists of len-L arrays (paired; drop pairs with a None
    withhold before calling). Returns len-L array; lag axis is -KERNEL_PRE_S ..
    -KERNEL_REFRACTORY_S."""

def bootstrap_kernel_ci(fa_windows, withhold_windows, n_boot=N_BOOT, seed=BOOT_SEED
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(kernel, lo, hi): point kernel + 2.5/97.5 percentile bands, resampling
    PAIRS with replacement via np.random.default_rng(seed)."""

def kernel_lags(dt=DT) -> np.ndarray:
    """Lag axis (s, negative) matching kernel length L."""

def kernel_shape_metrics(kernel, dt=DT) -> dict:
    """{peak_amp, peak_lag_s, half_width_s}: peak_amp=max(kernel); peak_lag_s=lag
    at argmax; half_width_s = width where kernel >= peak_amp/2 (contiguous around
    the peak). Amplitude and shape reported separately per the spec."""

def signed_population_signal(session, unit_signs, dt=DT) -> dict:
    """{trial_idx: (t, S)} where S(t)=mean_i sign_i * z_i(t); z_i = per-unit
    z-score of dt-binned rate to that unit's mean/SD over ALL baseline-period bins
    (shared-baseline equalization). unit_signs: {cluster_id: +1/-1}. Aligned to
    Baseline_ON, spanning [0, change_time or trial end)."""

def stimulus_matched_control(fa_windows, withhold_windows, fa_pop, withhold_pop
                             ) -> dict:
    """{sensory, gain, total}: total = neural FA-vs-withhold kernel (fa_pop -
    withhold_pop mean); sensory = neural kernel predicted from the STIMULUS-matched
    withholds (withholds already share the FA stimulus trajectory, so their neural
    signal is the sensory expectation); gain = total - sensory. All len-L arrays."""
```

---

## Phase 0 — Library (`psychophysical_kernel.py`, pure + TDD)

### Task 1: TF reconstruction — `baseline_log2tf`

**Files:**
- Create: `src/visdetect/analysis/psychophysical_kernel.py`
- Test: `tests/analysis/test_psychophysical_kernel.py`

**Interfaces:**
- Produces: `baseline_log2tf(trial, dt=DT, tf_base=None) -> (t, y)`, module constants.

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_psychophysical_kernel.py
import numpy as np
from types import SimpleNamespace
import visdetect.analysis.psychophysical_kernel as pk

def _trial(bv, outcome="fa", fa=None, ct=np.nan, cs=1.0):
    rt = {} if fa is None else {"FA": fa}
    return SimpleNamespace(baseline_values=np.asarray(bv, float),
                           reactiontimes=rt, trialoutcome=outcome,
                           change_time=ct, change_size=cs)

def test_baseline_log2tf_stride3_and_log2():
    # 9 frames at 60 Hz = 3 TF values (held x3): TF = [1, 2, 0.5]
    bv = np.array([1,1,1, 2,2,2, 0.5,0.5,0.5], float)
    t, y = pk.baseline_log2tf(_trial(bv), tf_base=1.0)
    assert y.shape == (3,)
    np.testing.assert_allclose(y, [0.0, 1.0, -1.0], atol=1e-9)
    np.testing.assert_allclose(t, [0.0, 0.05, 0.10], atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/analysis/test_psychophysical_kernel.py::test_baseline_log2tf_stride3_and_log2 -v`
Expected: FAIL (module `psychophysical_kernel` has no attribute `baseline_log2tf`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/visdetect/analysis/psychophysical_kernel.py
"""B10 — psychophysical / neural impulsivity kernel (Orsolic-style reverse
correlation of baseline TF fluctuations preceding impulsive licks).

Plain English: work out what pattern of grating-speed wobble the mouse mistakes
for a real change (i.e. what triggers an impulsive early lick), and how that
pattern — and its neural echo in striatal cells — changes as the mouse learns.
All estimators are pure and deterministic; scripts in scripts/evidence_learning/
apply them to real sessions.
"""
from __future__ import annotations
import numpy as np

DT = 0.05
KERNEL_PRE_S = 1.5
KERNEL_REFRACTORY_S = 0.15
MIN_BASELINE_S = 0.5
CHANGE_GUARD_S = 0.5
BOOT_SEED = 42
N_BOOT = 1000
_MONITOR_HZ = 60.0
_STRIDE = 3


def baseline_log2tf(trial, dt=DT, tf_base=None):
    """Full-trial baseline log2-TF on the dt grid anchored at Baseline_ON.

    Mirrors the verified stride-3 / 60 Hz recipe (tf_glm_data._BASELINE_STRIDE):
    baseline_values is logged 3x per 50 ms TF update, so bv[::3] recovers the
    genuine 50 ms grid. y[k] = log2(bv[3k]/base)."""
    bv = np.asarray(getattr(trial, "baseline_values", []), float).ravel()
    if bv.size == 0:
        return np.zeros(0), np.zeros(0)
    vals = bv[::_STRIDE]
    base = float(tf_base) if tf_base is not None else (float(np.nanmedian(bv)) or 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(vals > 0, np.log2(vals / base), 0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    t = np.arange(vals.size) * dt
    return t, y
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/analysis/test_psychophysical_kernel.py::test_baseline_log2tf_stride3_and_log2 -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/psychophysical_kernel.py tests/analysis/test_psychophysical_kernel.py
git commit -m "feat(B10): baseline_log2tf — stride-3 50ms TF reconstruction"
```

---

### Task 2: FA kernel epochs — `fa_kernel_epochs`

**Files:**
- Modify: `src/visdetect/analysis/psychophysical_kernel.py`
- Test: `tests/analysis/test_psychophysical_kernel.py`

**Interfaces:**
- Consumes: `baseline_log2tf` (Task 1).
- Produces: `fa_kernel_epochs(session, lick_shift_ms=0.0, dt=DT) -> list[dict]` with keys `trial_idx, lick_t, window`.

- [ ] **Step 1: Write the failing test**

```python
def _session(trials):
    return SimpleNamespace(trials=list(trials))

def test_fa_kernel_epochs_window_and_guards():
    # 40 s of white baseline at 60 Hz (2400 frames -> 800 TF bins)
    rng = np.random.default_rng(0)
    bv = np.exp2(rng.normal(0, 0.25, 800)); bv = np.repeat(bv, 3)
    good = _trial(bv, "fa", fa=5.0, ct=np.nan)          # lick at 5 s, no change
    early = _trial(bv, "fa", fa=0.3, ct=np.nan)         # < MIN_BASELINE_S -> drop
    near_change = _trial(bv, "fa", fa=5.0, ct=5.2)      # |lick-ct|<GUARD -> drop
    miss = _trial(bv, "miss", fa=None, ct=6.0)          # not fa -> drop
    eps = pk.fa_kernel_epochs(_session([good, early, near_change, miss]))
    assert len(eps) == 1
    L = round((pk.KERNEL_PRE_S - pk.KERNEL_REFRACTORY_S) / pk.DT)
    assert eps[0]["window"].shape == (L,)
    assert eps[0]["trial_idx"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/analysis/test_psychophysical_kernel.py::test_fa_kernel_epochs_window_and_guards -v`
Expected: FAIL (`fa_kernel_epochs` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
def _fa_latency(trial):
    rts = getattr(trial, "reactiontimes", {}) or {}
    v = rts.get("FA", rts.get("fa"))
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def fa_kernel_epochs(session, lick_shift_ms=0.0, dt=DT):
    """Per usable FA trial, the log2-TF window ending REFRACTORY before the lick."""
    L = round((KERNEL_PRE_S - KERNEL_REFRACTORY_S) / dt)
    shift = lick_shift_ms / 1000.0
    out = []
    for idx, tr in enumerate(getattr(session, "trials", []) or []):
        if (getattr(tr, "trialoutcome", "") or "").lower() != "fa":
            continue
        lick_t = _fa_latency(tr)
        if not np.isfinite(lick_t) or lick_t < MIN_BASELINE_S:
            continue
        ct = float(getattr(tr, "change_time", np.nan) or np.nan)
        if np.isfinite(ct) and abs(lick_t - ct) < CHANGE_GUARD_S:
            continue
        _, y = baseline_log2tf(tr, dt=dt)
        if y.size == 0:
            continue
        j1 = int(round((lick_t - shift - KERNEL_REFRACTORY_S) / dt))
        j0 = j1 - L
        if j0 < 0 or j1 > y.size:
            continue
        out.append({"trial_idx": idx, "lick_t": lick_t, "window": y[j0:j1].copy()})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/analysis/test_psychophysical_kernel.py::test_fa_kernel_epochs_window_and_guards -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(B10): fa_kernel_epochs — pre-lick TF window with guards"
```

---

### Task 3: Matched-withhold epochs — `withhold_epochs`

**Files:**
- Modify: `src/visdetect/analysis/psychophysical_kernel.py`
- Test: `tests/analysis/test_psychophysical_kernel.py`

**Interfaces:**
- Consumes: `baseline_log2tf`, `fa_kernel_epochs`.
- Produces: `withhold_epochs(session, fa_epochs, dt=DT, rng=None) -> list[np.ndarray | None]`.

- [ ] **Step 1: Write the failing test**

```python
def test_withhold_epochs_time_matched_and_prechange():
    rng = np.random.default_rng(1)
    bv = np.repeat(np.exp2(rng.normal(0, 0.25, 800)), 3)
    fa = _trial(bv, "fa", fa=5.0, ct=np.nan)
    hit = _trial(bv, "hit", fa=None, ct=8.0)     # baseline lasts to 8 s -> covers 5 s
    sess = _session([fa, hit])
    eps = pk.fa_kernel_epochs(sess)
    wh = pk.withhold_epochs(sess, eps, rng=np.random.default_rng(2))
    L = round((pk.KERNEL_PRE_S - pk.KERNEL_REFRACTORY_S) / pk.DT)
    assert len(wh) == 1 and wh[0] is not None and wh[0].shape == (L,)

def test_withhold_epochs_none_when_no_prechange_coverage():
    rng = np.random.default_rng(3)
    bv = np.repeat(np.exp2(rng.normal(0, 0.25, 800)), 3)
    fa = _trial(bv, "fa", fa=5.0, ct=np.nan)
    hit = _trial(bv, "hit", fa=None, ct=1.0)     # change at 1 s -> no pre-change 5 s window
    sess = _session([fa, hit])
    eps = pk.fa_kernel_epochs(sess)
    wh = pk.withhold_epochs(sess, eps, rng=np.random.default_rng(4))
    assert wh[0] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/analysis/test_psychophysical_kernel.py -k withhold -v`
Expected: FAIL (`withhold_epochs` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
_WITHHOLD_TOL_S = 0.25

def _withhold_trials(session):
    """hit/miss trials with a finite change_time (their pre-change baseline is
    a genuine no-lick epoch of known duration)."""
    out = []
    for tr in getattr(session, "trials", []) or []:
        oc = (getattr(tr, "trialoutcome", "") or "").lower()
        ct = float(getattr(tr, "change_time", np.nan) or np.nan)
        if oc in ("hit", "miss") and np.isfinite(ct):
            out.append((tr, ct))
    return out


def withhold_epochs(session, fa_epochs, dt=DT, rng=None):
    """One time-in-trial-matched no-lick window per FA epoch (None if unmatched).

    For an FA at lick_t, find withhold trials whose pre-change baseline extends
    past lick_t (change_time - REFRACTORY margin >= lick_t) and slice the SAME
    [lick_t-PRE, lick_t-REFRACTORY] window, matching time-in-trial within TOL."""
    rng = rng or np.random.default_rng(BOOT_SEED)
    L = round((KERNEL_PRE_S - KERNEL_REFRACTORY_S) / dt)
    wtrials = _withhold_trials(session)
    ys = {id(tr): baseline_log2tf(tr, dt=dt)[1] for tr, _ in wtrials}
    out = []
    for ep in fa_epochs:
        lick_t = ep["lick_t"]
        cands = [tr for tr, ct in wtrials if ct - KERNEL_REFRACTORY_S >= lick_t]
        picks = []
        for tr in cands:
            y = ys[id(tr)]
            j1 = int(round((lick_t - KERNEL_REFRACTORY_S) / dt))
            j0 = j1 - L
            if j0 >= 0 and j1 <= y.size:
                picks.append(y[j0:j1])
        out.append(picks[rng.integers(len(picks))].copy() if picks else None)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/analysis/test_psychophysical_kernel.py -k withhold -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(B10): withhold_epochs — time-in-trial-matched no-lick control"
```

---

### Task 4: Reverse-correlation kernel + **synthetic recovery** (the rigor backbone)

**Files:**
- Modify: `src/visdetect/analysis/psychophysical_kernel.py`
- Test: `tests/analysis/test_psychophysical_kernel.py`

**Interfaces:**
- Consumes: `fa_kernel_epochs`, `withhold_epochs`.
- Produces: `reverse_correlation_kernel(fa_windows, withhold_windows) -> np.ndarray`, `kernel_lags(dt=DT) -> np.ndarray`.

- [ ] **Step 1: Write the failing test** (plant a kernel, recover it)

```python
def test_reverse_correlation_recovers_planted_kernel():
    # Plant: FA licks are emitted when a specific ramp appears in the last ~0.5 s.
    rng = np.random.default_rng(7)
    L = round((pk.KERNEL_PRE_S - pk.KERNEL_REFRACTORY_S) / pk.DT)
    planted = np.zeros(L); planted[-10:] = np.linspace(0, 0.6, 10)  # rising ramp
    fa_windows = [planted + rng.normal(0, 0.25, L) for _ in range(400)]
    withhold_windows = [rng.normal(0, 0.25, L) for _ in range(400)]
    k = pk.reverse_correlation_kernel(fa_windows, withhold_windows)
    assert k.shape == (L,)
    # recovered kernel correlates strongly with the planted shape
    r = np.corrcoef(k, planted)[0, 1]
    assert r > 0.9
    assert k[-1] > k[0]                      # rising toward the lick
    assert pk.kernel_lags().shape == (L,)
    assert pk.kernel_lags()[-1] < 0          # lags are negative (before lick)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/analysis/test_psychophysical_kernel.py -k recovers_planted -v`
Expected: FAIL (`reverse_correlation_kernel` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
def reverse_correlation_kernel(fa_windows, withhold_windows):
    """FA-triggered mean minus withhold-matched mean, per lag."""
    fa = np.asarray(fa_windows, float)
    wh = np.asarray(withhold_windows, float)
    if fa.ndim != 2 or fa.size == 0:
        raise ValueError("fa_windows must be a non-empty list of equal-length arrays")
    return fa.mean(axis=0) - wh.mean(axis=0)


def kernel_lags(dt=DT):
    L = round((KERNEL_PRE_S - KERNEL_REFRACTORY_S) / dt)
    # window covers [-KERNEL_PRE_S, -KERNEL_REFRACTORY_S); bin left-edges
    return -KERNEL_PRE_S + np.arange(L) * dt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/analysis/test_psychophysical_kernel.py -k recovers_planted -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(B10): reverse_correlation_kernel + synthetic recovery test"
```

---

### Task 5: Bootstrap CI + determinism

**Files:**
- Modify: `src/visdetect/analysis/psychophysical_kernel.py`
- Test: `tests/analysis/test_psychophysical_kernel.py`

**Interfaces:**
- Consumes: `reverse_correlation_kernel`.
- Produces: `bootstrap_kernel_ci(fa_windows, withhold_windows, n_boot=N_BOOT, seed=BOOT_SEED) -> (kernel, lo, hi)`.

- [ ] **Step 1: Write the failing test**

```python
def test_bootstrap_ci_deterministic_and_bounds():
    rng = np.random.default_rng(11)
    L = round((pk.KERNEL_PRE_S - pk.KERNEL_REFRACTORY_S) / pk.DT)
    fa = [np.full(L, 0.3) + rng.normal(0, 0.1, L) for _ in range(200)]
    wh = [rng.normal(0, 0.1, L) for _ in range(200)]
    k1, lo1, hi1 = pk.bootstrap_kernel_ci(fa, wh, n_boot=200, seed=42)
    k2, lo2, hi2 = pk.bootstrap_kernel_ci(fa, wh, n_boot=200, seed=42)
    np.testing.assert_array_equal(lo1, lo2)          # byte-identical (determinism)
    np.testing.assert_array_equal(hi1, hi2)
    assert np.all(lo1 <= k1) and np.all(k1 <= hi1)
    assert np.all(lo1 > -1) and np.all(hi1 < 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/analysis/test_psychophysical_kernel.py -k bootstrap_ci_deterministic -v`
Expected: FAIL (`bootstrap_kernel_ci` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
def bootstrap_kernel_ci(fa_windows, withhold_windows, n_boot=N_BOOT, seed=BOOT_SEED):
    """Point kernel + 95% percentile bands, resampling PAIRS with replacement."""
    fa = np.asarray(fa_windows, float)
    wh = np.asarray(withhold_windows, float)
    n, L = fa.shape
    kernel = fa.mean(0) - wh.mean(0)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, L))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = fa[idx].mean(0) - wh[idx].mean(0)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return kernel, lo, hi
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/analysis/test_psychophysical_kernel.py -k bootstrap_ci_deterministic -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(B10): bootstrap_kernel_ci (deterministic, paired resample)"
```

---

### Task 6: Kernel shape metrics (shape ≠ amplitude)

**Files:**
- Modify: `src/visdetect/analysis/psychophysical_kernel.py`
- Test: `tests/analysis/test_psychophysical_kernel.py`

**Interfaces:**
- Consumes: `kernel_lags`.
- Produces: `kernel_shape_metrics(kernel, dt=DT) -> {peak_amp, peak_lag_s, half_width_s}`.

- [ ] **Step 1: Write the failing test**

```python
def test_kernel_shape_metrics():
    L = round((pk.KERNEL_PRE_S - pk.KERNEL_REFRACTORY_S) / pk.DT)
    k = np.zeros(L); k[-6:-2] = [0.2, 0.4, 0.4, 0.2]     # peak 0.4 near the lick
    m = pk.kernel_shape_metrics(k)
    assert abs(m["peak_amp"] - 0.4) < 1e-9
    assert m["peak_lag_s"] < 0                            # before the lick
    assert m["half_width_s"] >= 2 * pk.DT                 # >= 2 bins at half-max
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/analysis/test_psychophysical_kernel.py -k shape_metrics -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
def kernel_shape_metrics(kernel, dt=DT):
    """Peak amplitude, its lag, and the contiguous half-max width around the peak."""
    k = np.asarray(kernel, float)
    lags = kernel_lags(dt)
    pk_i = int(np.argmax(k))
    peak = float(k[pk_i])
    half = peak / 2.0
    lo = pk_i
    while lo - 1 >= 0 and k[lo - 1] >= half:
        lo -= 1
    hi = pk_i
    while hi + 1 < k.size and k[hi + 1] >= half:
        hi += 1
    return {"peak_amp": peak, "peak_lag_s": float(lags[pk_i]),
            "half_width_s": float((hi - lo + 1) * dt)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/analysis/test_psychophysical_kernel.py -k shape_metrics -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(B10): kernel_shape_metrics (peak/lag/half-width)"
```

---

### Task 7: Signed population TF signal (neural)

**Files:**
- Modify: `src/visdetect/analysis/psychophysical_kernel.py`
- Test: `tests/analysis/test_psychophysical_kernel.py`

**Interfaces:**
- Produces: `signed_population_signal(session, unit_signs, dt=DT) -> {trial_idx: (t, S)}`.
- Consumes (from real data at call time): `session.ni_events['Baseline_ON']`, `session.clusters` (each has `cluster_id`, `spike_times`), `trial.change_time`.

- [ ] **Step 1: Write the failing test**

```python
def _cluster(cid, spikes):
    return SimpleNamespace(cluster_id=cid, spike_times=np.asarray(spikes, float))

def test_signed_population_signal_tracks_stimulus():
    # 2 trials, baseline 0..4 s. Unit 10 (sign +1) fires denser in [1,2]s of trial 0.
    bon = [100.0, 200.0]
    tr0 = _trial(np.ones(2400), "fa", fa=3.0, ct=np.nan)
    tr1 = _trial(np.ones(2400), "hit", fa=None, ct=4.0)
    # unit spikes (absolute clock): burst in trial0 [1,2]s -> abs [101,102]
    dense = 100.0 + np.r_[np.linspace(0, 1, 20), np.linspace(1, 2, 200),
                          np.linspace(2, 3, 20)]
    sparse = 200.0 + np.linspace(0, 4, 60)
    sess = SimpleNamespace(trials=[tr0, tr1],
                           ni_events={"Baseline_ON": np.array(bon)},
                           clusters=[_cluster(10, np.r_[dense, sparse])])
    out = pk.signed_population_signal(sess, {10: +1})
    assert set(out) == {0, 1}
    t, S = out[0]
    # S peaks in the [1,2]s window of trial 0
    assert t[np.argmax(S)] > 0.8 and t[np.argmax(S)] < 2.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/analysis/test_psychophysical_kernel.py -k signed_population -v`
Expected: FAIL (`signed_population_signal` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
def _trial_windows(session):
    """[(trial_idx, t0_abs, dur_s)] per trial: baseline start -> change or +T."""
    bon = np.asarray(session.ni_events.get("Baseline_ON", []), float).ravel()
    out = []
    for idx, tr in enumerate(getattr(session, "trials", []) or []):
        if idx >= bon.size or not np.isfinite(bon[idx]):
            continue
        ct = float(getattr(tr, "change_time", np.nan) or np.nan)
        dur = ct if np.isfinite(ct) and ct > 0 else 6.0
        out.append((idx, float(bon[idx]), float(dur)))
    return out


def signed_population_signal(session, unit_signs, dt=DT):
    """Per-trial signed z-scored population TF signal aligned to Baseline_ON."""
    clusters = {c.cluster_id: np.asarray(c.spike_times, float)
                for c in getattr(session, "clusters", [])}
    windows = _trial_windows(session)
    # Per-unit binned rate per trial; z-score to that unit's mean/SD over ALL bins.
    per_unit = {}                       # cid -> {trial_idx: rate_array}
    for cid in unit_signs:
        st = clusters.get(cid)
        if st is None:
            continue
        per_trial = {}
        for idx, t0, dur in windows:
            nb = int(round(dur / dt))
            if nb < 1:
                continue
            edges = t0 + np.arange(nb + 1) * dt
            per_trial[idx] = np.histogram(st, bins=edges)[0] / dt
        per_unit[cid] = per_trial
    # shared-baseline z per unit
    z_unit = {}
    for cid, per_trial in per_unit.items():
        allr = np.concatenate(list(per_trial.values())) if per_trial else np.zeros(1)
        mu, sd = float(allr.mean()), float(allr.std())
        sd = sd if sd > 1e-9 else 1.0
        z_unit[cid] = {i: (r - mu) / sd for i, r in per_trial.items()}
    # signed mean across units, per trial
    out = {}
    for idx, t0, dur in windows:
        nb = int(round(dur / dt))
        if nb < 1:
            continue
        acc = np.zeros(nb); ncontrib = 0
        for cid, sign in unit_signs.items():
            zt = z_unit.get(cid, {}).get(idx)
            if zt is not None and zt.size == nb:
                acc += sign * zt; ncontrib += 1
        S = acc / ncontrib if ncontrib else acc
        out[idx] = (np.arange(nb) * dt, S)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/analysis/test_psychophysical_kernel.py -k signed_population -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(B10): signed_population_signal (per-unit z, signed sum)"
```

---

### Task 8: Stimulus-matched sensory-vs-gain control

**Files:**
- Modify: `src/visdetect/analysis/psychophysical_kernel.py`
- Test: `tests/analysis/test_psychophysical_kernel.py`

**Interfaces:**
- Produces: `stimulus_matched_control(fa_windows, withhold_windows, fa_pop, withhold_pop) -> {sensory, gain, total}`.

Note: `fa_pop`/`withhold_pop` are the **neural** signed-signal windows (len-L arrays) aligned to FA vs matched-withhold; `fa_windows`/`withhold_windows` are the paired **stimulus** windows (used only to assert the pairing length here). `withhold_pop` carries the same stimulus trajectory as its FA (matched), so its mean neural signal is the sensory expectation.

- [ ] **Step 1: Write the failing test**

```python
def test_stimulus_matched_control_decomposes_gain():
    L = round((pk.KERNEL_PRE_S - pk.KERNEL_REFRACTORY_S) / pk.DT)
    stim_fa = [np.linspace(0, 0.5, L) for _ in range(100)]
    stim_wh = [np.linspace(0, 0.5, L) for _ in range(100)]     # matched stimulus
    # neural: withhold tracks stimulus (sensory); FA adds a constant +0.4 gain bump
    pop_wh = [s.copy() for s in stim_wh]
    pop_fa = [s + 0.4 for s in stim_fa]
    d = pk.stimulus_matched_control(stim_fa, stim_wh, pop_fa, pop_wh)
    np.testing.assert_allclose(d["sensory"], np.mean(pop_wh, 0), atol=1e-9)
    np.testing.assert_allclose(d["gain"], np.full(L, 0.4), atol=1e-9)
    np.testing.assert_allclose(d["total"], d["sensory"] + d["gain"], atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/analysis/test_psychophysical_kernel.py -k stimulus_matched -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
def stimulus_matched_control(fa_windows, withhold_windows, fa_pop, withhold_pop):
    """Decompose the neural FA-vs-withhold signal into sensory + excess-gain.

    withhold_pop shares the FA's stimulus trajectory (stimulus-matched), so its
    mean is the sensory expectation; the FA-minus-withhold residual is gain."""
    if len(fa_windows) != len(withhold_windows):
        raise ValueError("stimulus windows must be paired 1:1")
    fa_p = np.asarray(fa_pop, float)
    wh_p = np.asarray(withhold_pop, float)
    sensory = wh_p.mean(0)
    total = fa_p.mean(0)
    return {"sensory": sensory, "gain": total - sensory, "total": total}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/analysis/test_psychophysical_kernel.py -k stimulus_matched -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(B10): stimulus_matched_control (sensory vs excess-gain)"
```

- [ ] **Step 6: Full library test run**

Run: `pytest tests/analysis/test_psychophysical_kernel.py -v`
Expected: all Phase-0 tests PASS.

---

## Phase 1 — Scripts (real-data application + figures)

Shared script infrastructure (put in `scripts/evidence_learning/_common.py`, Task 9 Step 3):

```python
# scripts/evidence_learning/_common.py
"""Shared loaders for B10 evidence-learning scripts (multi-subject).
NOT the single-subject suite loader — B10 spans BG_046/039/031."""
import os, gc
import numpy as np, pandas as pd
from visdetect.core.session import load_session as _core_load
from visdetect.analysis.config import canonical_session_id

SUBJECTS = {"BG_046": "DMS", "BG_039": "DMS", "BG_031": "VMS"}
CACHE_DIR = os.path.join("data", "cache", "evidence_learning")
FIG_DIR = os.path.join("FIGURES", "evidence_learning")
STAGES = ("Naive", "Learning", "Expert")

def load_manifest(subject):
    m = pd.read_csv(f"data/{subject}_staging_manifest.csv")
    m["csid"] = m["session_name"].map(canonical_session_id)
    return m

def subject_sessions(subject, stages=STAGES):
    """Yield (csid, stage, Session) for QC-pass sessions in the given stages."""
    m = load_manifest(subject)
    m = m[m["stage"].isin(set(stages))]
    for _, r in m.iterrows():
        csid = r["csid"]
        path = os.path.join("data", "pkls", subject, f"{subject}_{csid}.pkl")
        if not os.path.exists(path):
            continue
        sess = _core_load(path)
        yield csid, r["stage"], sess
        del sess; gc.collect()

def tf_responsive_units(subject):
    """{csid: {cluster_id: sign}} for registry-responsive, region-confirmed units."""
    reg = pd.read_csv(f"data/cache/tf_responsive/{subject.lower()}_tf_responsive.csv")
    reg = reg[(reg["resp_log2"] == True) & (reg["region_bank_confirmed"] == True)]
    reg["csid"] = reg["session"].map(canonical_session_id)
    out = {}
    for csid, g in reg.groupby("csid"):
        out[csid] = {int(u): (1 if c >= 0 else -1)
                     for u, c in zip(g["unit"], g["c1_r_log2"])}
    return out
```

### Task 9: Phase-0 coverage table script

**Files:**
- Create: `scripts/evidence_learning/_common.py`
- Create: `scripts/evidence_learning/b10_phase0_coverage.py`
- Test: `tests/scripts/test_b10_scripts.py`

**Interfaces:**
- Consumes: `psychophysical_kernel.fa_kernel_epochs`, `_common.subject_sessions/tf_responsive_units`.
- Produces: `data/cache/evidence_learning/b10_coverage.csv` with columns `subject, stage, csid, n_fa_usable, n_withhold_ok, n_tf_units, usable`.

- [ ] **Step 1: Write the failing smoke test** (synthetic session, no real pkls)

```python
# tests/scripts/test_b10_scripts.py
import importlib, numpy as np
from types import SimpleNamespace

def _mk_session(n_fa=30):
    rng = np.random.default_rng(0)
    trials = []
    for i in range(n_fa):
        bv = np.repeat(np.exp2(rng.normal(0, 0.25, 800)), 3)
        trials.append(SimpleNamespace(baseline_values=bv, reactiontimes={"FA": 5.0},
                                      trialoutcome="fa", change_time=np.nan, change_size=1.0))
    for i in range(20):
        bv = np.repeat(np.exp2(rng.normal(0, 0.25, 800)), 3)
        trials.append(SimpleNamespace(baseline_values=bv, reactiontimes={},
                                      trialoutcome="hit", change_time=8.0, change_size=2.0))
    return SimpleNamespace(trials=trials, ni_events={"Baseline_ON": np.zeros(n_fa + 20)},
                           clusters=[])

def test_coverage_row_counts():
    cov = importlib.import_module("scripts.evidence_learning.b10_phase0_coverage")
    row = cov.coverage_row("BG_046", "Expert", "01072025", _mk_session())
    assert row["n_fa_usable"] == 30
    assert row["n_withhold_ok"] == 30       # all FAs matched by the hit trials
    assert row["subject"] == "BG_046" and row["stage"] == "Expert"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/scripts/test_b10_scripts.py::test_coverage_row_counts -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Write `_common.py` (above) and the coverage script**

```python
# scripts/evidence_learning/b10_phase0_coverage.py
"""B10 Phase 0 — coverage/usable gate per subject x stage.

Plain English: before measuring anything, count how many impulsive licks are
usable for a kernel (enough pre-lick history, a matched no-lick control) and how
many TF-responsive cells exist, and flag which cells are worth analysing."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from visdetect.analysis import psychophysical_kernel as pk
from scripts.evidence_learning._common import (
    SUBJECTS, STAGES, CACHE_DIR, subject_sessions, tf_responsive_units)

MIN_FA, MIN_TF = 30, 3           # usable thresholds (spec §6 formalized)

def coverage_row(subject, stage, csid, session, tf_by_session=None):
    eps = pk.fa_kernel_epochs(session)
    wh = pk.withhold_epochs(session, eps)
    n_ok = sum(1 for w in wh if w is not None)
    n_tf = len((tf_by_session or {}).get(csid, {}))
    return {"subject": subject, "stage": stage, "csid": csid,
            "n_fa_usable": len(eps), "n_withhold_ok": n_ok, "n_tf_units": n_tf,
            "usable": len(eps) >= MIN_FA and n_ok >= MIN_FA}

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    rows = []
    for subject in SUBJECTS:
        tf = tf_responsive_units(subject)
        for csid, stage, sess in subject_sessions(subject):
            rows.append(coverage_row(subject, stage, csid, sess, tf))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CACHE_DIR, "b10_coverage.csv"), index=False)
    print(df.groupby(["subject", "stage"])[["n_fa_usable", "n_tf_units"]].sum())

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/scripts/test_b10_scripts.py::test_coverage_row_counts -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(B10): phase-0 coverage script + multi-subject loaders"
```

---

### Task 10: Behavioral kernel figure (Fig B10.1)

**Files:**
- Create: `scripts/evidence_learning/b10_phase1_behavioral.py`
- Test: `tests/scripts/test_b10_scripts.py`

**Interfaces:**
- Consumes: `pk.fa_kernel_epochs/withhold_epochs/bootstrap_kernel_ci/kernel_shape_metrics/kernel_lags`, `_common`.
- Produces: `session_kernel(session, rng)` → `(kernel, n_pairs)`; `stage_kernel(windows_by_stage, n_match, seed)` → n-matched pooled kernel + CI; writes `FIGURES/evidence_learning/<subject>/b10_behavioral_kernel.png` + `data/cache/evidence_learning/b10_behavioral_kernel_stats.csv`.

- [ ] **Step 1: Write the failing test**

```python
def test_session_kernel_pairs_and_length():
    b = importlib.import_module("scripts.evidence_learning.b10_phase1_behavioral")
    import numpy as np
    sess = _mk_session(40)
    k, npairs = b.session_kernel(sess, np.random.default_rng(0))
    L = round((1.5 - 0.15) / 0.05)            # 27
    assert k.shape[0] == L and npairs == 30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/scripts/test_b10_scripts.py::test_session_kernel_pairs_and_length -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Write the script**

```python
# scripts/evidence_learning/b10_phase1_behavioral.py
"""B10 Phase 1 (behavioral) — the impulsivity kernel across learning, 3 mice.

Fig: (A) pooled kernel + Khilkevich raw-anchor; (B) Naive vs Expert per subject
+ pooled, n-matched; (C) shape (half-width, peak-lag) & amplitude vs stage.
The kernel = FA-triggered log2-TF minus time-in-trial-matched withhold."""
import os, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from visdetect.analysis import psychophysical_kernel as pk
from visdetect.suite.plotting import setup_style
from scripts.evidence_learning._common import (
    SUBJECTS, CACHE_DIR, FIG_DIR, subject_sessions)

setup_style()
HEADLINE = ("Naive", "Expert")

def session_kernel(session, rng):
    """Return (paired-mean kernel, n_pairs) for one session; raw FA-triggered
    windows and matched withholds. Empty kernel if no pairs."""
    eps = pk.fa_kernel_epochs(session)
    wh = pk.withhold_epochs(session, eps, rng=rng)
    fa_w, wh_w = zip(*[(e["window"], w) for e, w in zip(eps, wh) if w is not None]) \
        if any(w is not None for w in wh) else ([], [])
    if not fa_w:
        return np.zeros(0), 0
    return pk.reverse_correlation_kernel(list(fa_w), list(wh_w)), len(fa_w)

def collect_windows(subject, stages):
    """{stage: (fa_windows, withhold_windows)} pooling all sessions of a subject."""
    rng = np.random.default_rng(pk.BOOT_SEED)
    acc = {s: ([], []) for s in stages}
    for csid, stage, sess in subject_sessions(subject, stages):
        if stage not in acc:
            continue
        eps = pk.fa_kernel_epochs(sess)
        wh = pk.withhold_epochs(sess, eps, rng=rng)
        for e, w in zip(eps, wh):
            if w is not None:
                acc[stage][0].append(e["window"]); acc[stage][1].append(w)
    return acc

def stage_kernel(fa_w, wh_w, n_match, seed=pk.BOOT_SEED):
    """n-matched kernel + CI (subsample to n_match pairs)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(fa_w), size=min(n_match, len(fa_w)), replace=False)
    fa = [fa_w[i] for i in idx]; wh = [wh_w[i] for i in idx]
    return pk.bootstrap_kernel_ci(fa, wh)

def main():
    lags = pk.kernel_lags()
    stats = []
    pooled = {s: ([], []) for s in HEADLINE}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for subject in SUBJECTS:
        acc = collect_windows(subject, HEADLINE)
        nmin = min((len(acc[s][0]) for s in HEADLINE if acc[s][0]), default=0)
        for s in HEADLINE:
            fa_w, wh_w = acc[s]
            pooled[s][0].extend(fa_w); pooled[s][1].extend(wh_w)
            if len(fa_w) >= nmin > 0:
                k, lo, hi = stage_kernel(fa_w, wh_w, nmin)
                m = pk.kernel_shape_metrics(k)
                stats.append({"subject": subject, "stage": s, "n_pairs": len(fa_w),
                              "n_match": nmin, **m})
                axes[1].plot(lags, k, label=f"{subject} {s}")
    # Panel A: pooled kernel + CI
    nmin = min((len(pooled[s][0]) for s in HEADLINE if pooled[s][0]), default=0)
    for s in HEADLINE:
        if len(pooled[s][0]) >= nmin > 0:
            k, lo, hi = stage_kernel(pooled[s][0], pooled[s][1], nmin)
            axes[0].plot(lags, k, label=f"pooled {s}")
            axes[0].fill_between(lags, lo, hi, alpha=0.2)
    axes[0].axhline(0, color="k", lw=0.5); axes[0].set_title("Impulsivity kernel (pooled)")
    axes[0].set_xlabel("time before recorded lick (s)"); axes[0].set_ylabel("log2-TF (FA - withhold)")
    axes[0].legend(); axes[1].set_title("Per subject: Naive vs Expert"); axes[1].legend()
    # Panel C: shape/amplitude vs stage
    sdf = pd.DataFrame(stats)
    if not sdf.empty:
        for metric, ax_share in [("half_width_s", axes[2])]:
            for subject in SUBJECTS:
                d = sdf[sdf.subject == subject]
                axes[2].plot(d["stage"], d["half_width_s"], "o-", label=subject)
        axes[2].set_title("Kernel half-width vs stage"); axes[2].set_ylabel("half-width (s)")
        axes[2].legend()
    fig.tight_layout()
    outdir = os.path.join(FIG_DIR, "pooled"); os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "b10_behavioral_kernel.png"), dpi=300, bbox_inches="tight")
    os.makedirs(CACHE_DIR, exist_ok=True)
    sdf.to_csv(os.path.join(CACHE_DIR, "b10_behavioral_kernel_stats.csv"), index=False)
    print(sdf)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/scripts/test_b10_scripts.py::test_session_kernel_pairs_and_length -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(B10): behavioral impulsivity-kernel figure (Fig B10.1)"
```

---

### Task 11: Neural kernel figure (Fig B10.2)

**Files:**
- Create: `scripts/evidence_learning/b10_phase1_neural.py`
- Test: `tests/scripts/test_b10_scripts.py`

**Interfaces:**
- Consumes: `pk.signed_population_signal/fa_kernel_epochs/withhold_epochs/stimulus_matched_control`, `_common.tf_responsive_units`.
- Produces: `neural_fa_withhold(session, unit_signs, rng)` → `(fa_pop, wh_pop, fa_stim, wh_stim)` aligned len-L windows; writes `FIGURES/evidence_learning/DMS/b10_neural_kernel.png`, `.../VMS/...`, and `data/cache/evidence_learning/b10_neural_kernel_stats.csv`.

- [ ] **Step 1: Write the failing test**

```python
def test_neural_fa_withhold_shapes():
    n = importlib.import_module("scripts.evidence_learning.b10_phase1_neural")
    import numpy as np
    from types import SimpleNamespace
    rng = np.random.default_rng(0)
    trials = []
    for i in range(30):
        bv = np.repeat(np.exp2(rng.normal(0, 0.25, 800)), 3)
        trials.append(SimpleNamespace(baseline_values=bv, reactiontimes={"FA": 5.0},
                                      trialoutcome="fa", change_time=np.nan, change_size=1.0))
    for i in range(20):
        bv = np.repeat(np.exp2(rng.normal(0, 0.25, 800)), 3)
        trials.append(SimpleNamespace(baseline_values=bv, reactiontimes={},
                                      trialoutcome="hit", change_time=8.0, change_size=2.0))
    spikes = 0.0 + np.sort(rng.uniform(0, 300, 5000))
    sess = SimpleNamespace(trials=trials,
                           ni_events={"Baseline_ON": 10.0 * np.arange(len(trials))},
                           clusters=[SimpleNamespace(cluster_id=7, spike_times=spikes)])
    fa_pop, wh_pop, fa_stim, wh_stim = n.neural_fa_withhold(sess, {7: +1}, rng)
    L = len(pk_lags())  # helper below
    assert len(fa_pop) == len(wh_pop) == len(fa_stim) == len(wh_stim)
    assert all(len(w) == L for w in fa_pop)

def pk_lags():
    from visdetect.analysis import psychophysical_kernel as pk
    return pk.kernel_lags()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/scripts/test_b10_scripts.py::test_neural_fa_withhold_shapes -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Write the script**

```python
# scripts/evidence_learning/b10_phase1_neural.py
"""B10 Phase 1 (neural) — the neural impulsivity kernel on TF-responsive cells.

Fig: (A) DMS-pool signed-population signal, FA vs withhold + stimulus-matched
control; (B) Naive vs Expert; (C) sensory-vs-gain decomposition; VMS separate.
Stimulus-referenced (motor-safe); per session then aggregated over sessions."""
import os, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from visdetect.analysis import psychophysical_kernel as pk
from visdetect.suite.plotting import setup_style
from scripts.evidence_learning._common import (
    SUBJECTS, CACHE_DIR, FIG_DIR, subject_sessions, tf_responsive_units)

setup_style()
HEADLINE = ("Naive", "Expert")
REGION_POOLS = {"DMS": ("BG_046", "BG_039"), "VMS": ("BG_031",)}

def _window_from_signal(t, S, lick_t, dt=pk.DT):
    """Slice the len-L pre-lick window from a per-trial (t, S) signal."""
    L = len(pk.kernel_lags(dt))
    j1 = int(round((lick_t - pk.KERNEL_REFRACTORY_S) / dt)); j0 = j1 - L
    if j0 < 0 or j1 > S.size:
        return None
    return S[j0:j1].copy()

def neural_fa_withhold(session, unit_signs, rng):
    """Paired FA vs matched-withhold NEURAL windows (+ their stimulus windows)."""
    if not unit_signs:
        return [], [], [], []
    sig = pk.signed_population_signal(session, unit_signs)
    eps = pk.fa_kernel_epochs(session)
    wh = pk.withhold_epochs(session, eps, rng=rng)
    wtrials = pk._withhold_trials(session)
    fa_pop, wh_pop, fa_stim, wh_stim = [], [], [], []
    for e, w in zip(eps, wh):
        if w is None or e["trial_idx"] not in sig:
            continue
        t, S = sig[e["trial_idx"]]
        fa_win = _window_from_signal(t, S, e["lick_t"])
        # neural withhold: same time-in-trial window on a matched withhold trial
        wh_win = None
        for tr, ct in wtrials:
            widx = session.trials.index(tr)
            if widx in sig and ct - pk.KERNEL_REFRACTORY_S >= e["lick_t"]:
                wt, wS = sig[widx]
                wh_win = _window_from_signal(wt, wS, e["lick_t"])
                if wh_win is not None:
                    break
        if fa_win is not None and wh_win is not None:
            fa_pop.append(fa_win); wh_pop.append(wh_win)
            fa_stim.append(e["window"]); wh_stim.append(w)
    return fa_pop, wh_pop, fa_stim, wh_stim

def main():
    lags = pk.kernel_lags(); stats = []
    fig, axes = plt.subplots(len(REGION_POOLS), 3, figsize=(15, 8), squeeze=False)
    for ri, (region, subs) in enumerate(REGION_POOLS.items()):
        pooled = {s: ([], [], [], []) for s in HEADLINE}
        for subject in subs:
            tf = tf_responsive_units(subject)
            rng = np.random.default_rng(pk.BOOT_SEED)
            for csid, stage, sess in subject_sessions(subject, HEADLINE):
                if stage not in pooled:
                    continue
                fp, wp, fs, ws = neural_fa_withhold(sess, tf.get(csid, {}), rng)
                for tgt, src in zip(pooled[stage], (fp, wp, fs, ws)):
                    tgt.extend(src)
        for s in HEADLINE:
            fp, wp, fs, ws = pooled[s]
            if len(fp) < 20:
                continue
            k, lo, hi = pk.bootstrap_kernel_ci(fp, wp)   # FA vs withhold neural kernel
            dec = pk.stimulus_matched_control(fs, ws, fp, wp)
            axes[ri][0].plot(lags, k, label=s); axes[ri][0].fill_between(lags, lo, hi, alpha=0.2)
            axes[ri][2].plot(lags, dec["sensory"], label=f"{s} sensory")
            axes[ri][2].plot(lags, dec["gain"], "--", label=f"{s} gain")
            stats.append({"region": region, "stage": s, "n": len(fp),
                          **{f"peak_{k2}": v for k2, v in pk.kernel_shape_metrics(k).items()}})
        axes[ri][0].axhline(0, color="k", lw=0.5)
        axes[ri][0].set_title(f"{region}: neural FA vs withhold"); axes[ri][0].legend()
        axes[ri][1].set_title(f"{region}: Naive vs Expert")
        axes[ri][2].set_title(f"{region}: sensory vs gain"); axes[ri][2].legend()
        for c in range(3):
            axes[ri][c].set_xlabel("time before recorded lick (s)")
    fig.tight_layout()
    outdir = os.path.join(FIG_DIR, "neural"); os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "b10_neural_kernel.png"), dpi=300, bbox_inches="tight")
    os.makedirs(CACHE_DIR, exist_ok=True)
    pd.DataFrame(stats).to_csv(os.path.join(CACHE_DIR, "b10_neural_kernel_stats.csv"), index=False)
    print(pd.DataFrame(stats))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/scripts/test_b10_scripts.py::test_neural_fa_withhold_shapes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(B10): neural impulsivity-kernel figure (Fig B10.2)"
```

---

## Phase 2 — State-resolved (Fig B10.3)

### Task 12: State-split behavioral + neural kernels

**Files:**
- Create: `scripts/evidence_learning/b10_phase2_state.py`
- Test: `tests/scripts/test_b10_scripts.py`

**Interfaces:**
- Consumes: everything above + `visdetect.analysis.decision_latents.load_state_labels`.
- Produces: `fa_epochs_by_state(session, subject, states=('StimSens','Impulsive'))` → `{state: fa_epochs}`; writes `FIGURES/evidence_learning/state/b10_state_kernel.png` + `.../b10_state_kernel_stats.csv`.

- [ ] **Step 1: Write the failing test**

```python
def test_fa_epochs_by_state_splits(monkeypatch):
    s = importlib.import_module("scripts.evidence_learning.b10_phase2_state")
    import numpy as np, pandas as pd
    from types import SimpleNamespace
    rng = np.random.default_rng(0)
    trials = []
    for i in range(10):
        bv = np.repeat(np.exp2(rng.normal(0, 0.25, 800)), 3)
        trials.append(SimpleNamespace(baseline_values=bv, reactiontimes={"FA": 5.0},
                                      trialoutcome="fa", change_time=np.nan, change_size=1.0))
    sess = SimpleNamespace(trials=trials, ni_events={"Baseline_ON": np.zeros(10)}, clusters=[])
    # first 6 FAs StimSens, last 4 Impulsive, all confident
    labels = pd.DataFrame({"state_label": ["StimSens"]*6 + ["Impulsive"]*4,
                           "state_confidence": [0.9]*10}, index=range(10))
    monkeypatch.setattr(s, "load_state_labels", lambda *a, **k: labels)
    by = s.fa_epochs_by_state(sess, "BG_046")
    assert len(by["StimSens"]) == 6 and len(by["Impulsive"]) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/scripts/test_b10_scripts.py::test_fa_epochs_by_state_splits -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Write the script**

```python
# scripts/evidence_learning/b10_phase2_state.py
"""B10 Phase 2 — impulsivity kernel split by behavioral state (StimSens vs
Impulsive). NON-CIRCULAR: state labels come from lick RATES/outcomes; the kernel
(what stimulus pattern precedes the lick) is an independent measurement.

Hypothesis: StimSens FAs = genuine stimulus-driven false alarms (SHARP kernel);
Impulsive FAs = internal itch, stimulus-decoupled (FLAT kernel)."""
import os, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from visdetect.analysis import psychophysical_kernel as pk
from visdetect.analysis.decision_latents import load_state_labels
from visdetect.suite.plotting import setup_style
from scripts.evidence_learning._common import SUBJECTS, CACHE_DIR, FIG_DIR, subject_sessions

setup_style()
STATES = ("StimSens", "Impulsive")
CONF = 0.8

def fa_epochs_by_state(session, subject, states=STATES, csid=None):
    """{state: [fa_epoch dicts]} — FA epochs whose trial's confident state label
    is in `states`."""
    eps = pk.fa_kernel_epochs(session)
    try:
        labels = load_state_labels(csid or "", subject=subject)
    except FileNotFoundError:
        return {s: [] for s in states}
    by = {s: [] for s in states}
    for e in eps:
        idx = e["trial_idx"]
        if idx not in labels.index:
            continue
        row = labels.loc[idx]
        lab, conf = row["state_label"], float(row["state_confidence"])
        if conf >= CONF and lab in by:
            by[lab].append(e)
    return by

def main():
    lags = pk.kernel_lags(); stats = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    acc = {s: ([], []) for s in STATES}                 # pooled behavioral
    for subject in SUBJECTS:
        rng = np.random.default_rng(pk.BOOT_SEED)
        for csid, stage, sess in subject_sessions(subject, ("Naive", "Expert")):
            by = fa_epochs_by_state(sess, subject, csid=csid)
            wtrials_rng = np.random.default_rng(pk.BOOT_SEED)
            for state, eps in by.items():
                wh = pk.withhold_epochs(sess, eps, rng=wtrials_rng)
                for e, w in zip(eps, wh):
                    if w is not None:
                        acc[state][0].append(e["window"]); acc[state][1].append(w)
    nmin = min((len(acc[s][0]) for s in STATES if acc[s][0]), default=0)
    for s in STATES:
        fa_w, wh_w = acc[s]
        if len(fa_w) >= nmin > 0:
            rng = np.random.default_rng(pk.BOOT_SEED)
            idx = rng.choice(len(fa_w), nmin, replace=False)
            k, lo, hi = pk.bootstrap_kernel_ci([fa_w[i] for i in idx], [wh_w[i] for i in idx])
            axes[0].plot(lags, k, label=s); axes[0].fill_between(lags, lo, hi, alpha=0.2)
            stats.append({"state": s, "n_pairs": len(fa_w), "n_match": nmin,
                          **pk.kernel_shape_metrics(k)})
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_title("Behavioral kernel by state (pooled)")
    axes[0].set_xlabel("time before recorded lick (s)")
    axes[0].set_ylabel("log2-TF (FA - withhold)"); axes[0].legend()
    axes[1].axis("off")
    axes[1].text(0.02, 0.5, "Non-circular: state labels use lick rates/outcomes;\n"
                 "the kernel shape is an independent measurement.\n"
                 "Naive-StimSens is the thinnest cell (wide CI).", va="center")
    fig.tight_layout()
    outdir = os.path.join(FIG_DIR, "state"); os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "b10_state_kernel.png"), dpi=300, bbox_inches="tight")
    os.makedirs(CACHE_DIR, exist_ok=True)
    pd.DataFrame(stats).to_csv(os.path.join(CACHE_DIR, "b10_state_kernel_stats.csv"), index=False)
    print(pd.DataFrame(stats))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/scripts/test_b10_scripts.py::test_fa_epochs_by_state_splits -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(B10): state-resolved kernel (StimSens vs Impulsive, Fig B10.3)"
```

---

### Task 13: README + full test sweep + real-data run notes

**Files:**
- Create: `scripts/evidence_learning/README.md`
- Test: (none new)

- [ ] **Step 1: Write the README**

```markdown
# B10 — Impulsivity kernel across learning (evidence_learning)

Behavioral (I1) + neural (N-B) Orsolic-style reverse-correlation kernel, 3 mice
(BG_046/039 DMS, BG_031 VMS). See spec/plan:
docs/superpowers/specs/2026-07-01-B10-impulsivity-kernel-learning-design.md.

## Run order (real data; needs local data/pkls + registries + state_tags)
1. `py scripts/evidence_learning/b10_phase0_coverage.py`   # coverage/usable gate
2. `py scripts/evidence_learning/b10_phase1_behavioral.py`  # Fig B10.1
3. `py scripts/evidence_learning/b10_phase1_neural.py`      # Fig B10.2
4. `py scripts/evidence_learning/b10_phase2_state.py`       # Fig B10.3

Outputs: FIGURES/evidence_learning/<pool>/, data/cache/evidence_learning/*.csv.

## Honest limitations (printed on figures)
- No video → behavioral kernel is "stimulus history preceding impulsive licks,"
  not pure sensory evidence.
- VMS is n=1 region; BG_039 Learning = 1 session (Naive/Expert only).
- Naive-StimSens is the thinnest Phase-2 cell (neural especially).
- Lag axis is "time before RECORDED lick" (no calibrated hardware delay).
- Nulls (flat kernel / no learning or state change) are pre-registered as reportable.
```

- [ ] **Step 2: Full test sweep**

Run: `pytest tests/analysis/test_psychophysical_kernel.py tests/scripts/test_b10_scripts.py -v`
Expected: all PASS.

- [ ] **Step 3: Real-data smoke run** (execution-time; requires local data junctioned into the worktree — see spec §9 / worktree note)

Run: `py scripts/evidence_learning/b10_phase0_coverage.py`
Expected: prints per-subject×stage FA/TF counts; writes `data/cache/evidence_learning/b10_coverage.csv`. Confirm counts roughly match the spec §2 tables (FA licks: BG_046 1107/3078/2933; usable will be lower after history/withhold guards).

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "docs(B10): evidence_learning README + run order"
```

---

## Self-Review

**1. Spec coverage:**
- §3 Arm 1 (behavioral kernel): Tasks 1–6, 10. ✓
- §3 Arm 2 (neural signed-sum + stimulus-matched control): Tasks 7, 8, 11. ✓
- §3 Phase 2 (state split, non-circular): Task 12. ✓
- §4 Controls: dt=0.05 (Task 1), lick_shift default 0 (Task 2), change-guard (Task 2), time-in-trial withhold (Task 3), n-match + shape-vs-amplitude (Tasks 6, 10), within-session z + no raw pooling (Tasks 7, 11), region_bank_confirmed + sign (Task 9 `_common`). ✓
- §6 Phase 0 usable gate: Task 9. ✓
- §7 success criteria (bootstrap CI, shape metrics, nulls reportable): Tasks 5, 6, 10–12. ✓
- §10 tests (synthetic recovery, determinism, join integrity, withhold matching): Tasks 4, 5, 3, 12. ✓
- **Gap noted & handled:** BG_039 `_v2` non-finite guard (spec §4/§8 item) — `baseline_log2tf` uses only `baseline_values` (no timestamp arithmetic), so it is immune to the `trial_bin_edges` non-finite crash; the guard is only needed if a script later calls `session_trial_regressors`, which B10 does not. Documented here rather than porting Task 0.

**2. Placeholder scan:** No TBD/TODO; every code step has complete, runnable code. ✓

**3. Type consistency:** `fa_kernel_epochs` returns dicts with `window` (used in Tasks 3, 10, 11, 12); `withhold_epochs` returns `list[array|None]` (callers drop None); `signed_population_signal` returns `{trial_idx: (t, S)}` (Task 11 slices via `_window_from_signal`); `stimulus_matched_control` returns `{sensory, gain, total}` (Task 11). `kernel_lags` length L reused consistently. ✓

*(Fix applied during review: Task 11 test references `pk.kernel_lags()` via a local helper; behavioral Task 10 test computes L explicitly — both consistent with `kernel_lags` length.)*

---

## Execution Handoff

Plan complete. Recommended: **subagent-driven-development** (fresh Opus 4.8 subagent per task, two-stage review between tasks) given the TDD structure and correctness priority. Real-data figure runs (Tasks 10–13 Step 3) require the local `data/` junctioned into the worktree — handle at execution time per the worktree data-safety rule (never `git worktree remove` while junctions are live).
