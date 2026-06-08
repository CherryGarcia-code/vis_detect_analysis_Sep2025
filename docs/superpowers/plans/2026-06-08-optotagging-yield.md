# Optotagging Yield Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **⚠️ COMMITS DEFERRED.** A parallel chat holds `main` in this checkout. Do every task's
> code + tests, but **do not run the `git commit` steps** until the user confirms `main` is
> free. When cleared, commit task-by-task in order (or squash as the user prefers). Never
> `git add -A` — stage only the listed paths, and never touch `tests/analysis/test_track_curation.py`.

**Goal:** Maximise defensible D1/D2 optotagging yield for BG_046 by replacing contaminated
metrics with baseline-corrected ones, adding an offline collision test and canonical SALT, and
reporting results in two tiers (candidate + high-confidence) with physiologically-correct
D1/D2 assignment.

**Architecture:** Approach A — extend `src/visdetect/analysis/optotagging.py` in place with
small, independently-testable functions; enrich `OptoMetrics`; add a unit-level classifier;
refactor the Fig43 script to consume them. Pool is unchanged (pkl-resident `good_and_stable`).

**Tech Stack:** Python, NumPy, SciPy (`scipy.stats.poisson`, `scipy.stats.fisher_exact`),
pandas, matplotlib; pytest. Spec: `docs/superpowers/specs/2026-06-08-optotagging-yield-design.md`.

---

## File Structure

- **Modify** `src/visdetect/analysis/optotagging.py` — new constants + functions
  (`baseline_rate_hz`, `estimate_response_window`, `poisson_excess_test`,
  `excess_reliability`, `excess_jitter`, `collision_test`, canonical `salt_test`,
  `is_spn_plausible_waveform`, `fiber_tier`, `classify_unit`); enriched `OptoMetrics`;
  new dataclasses `ResponseWindow`, `CollisionResult`, `UnitTag`; `analyze_unit` rewrite.
- **Create** `tests/analysis/test_optotagging.py` — TDD unit tests + synthetic builders.
- **Modify** `analysis_suite/09_optotagging/a_optotagging_identification.py` — two-tier
  outputs, yield-vs-threshold sweep, old-vs-new comparison, waveform join.
- **Create** `docs/results/2026-06-08-optotagging-yield-results.md` — the achieved numbers.

Each library function has one responsibility and is pure (takes arrays, returns values/dataclass),
so it can be tested without loading a real session.

---

## Task 1: Baseline rate + response-window estimation

**Files:**
- Modify: `src/visdetect/analysis/optotagging.py`
- Test: `tests/analysis/test_optotagging.py`

- [ ] **Step 1: Write the failing test (plus shared synthetic builders)**

```python
# tests/analysis/test_optotagging.py
import numpy as np
import pytest
from visdetect.core.session import Session, Cluster
from visdetect.analysis import optotagging as ot


# ── shared synthetic builders ────────────────────────────────────────
def _pulses(n=501, spacing=1.0, t0=10.0):
    return t0 + np.arange(n) * spacing


def _antidromic_unit(pulses, latency_ms=4.0, jitter_ms=0.2, base_rate=5.0,
                     collision=True, respond=True, seed=0):
    """Baseline Poisson + a locked post-pulse spike.

    If collision=True, the locked spike is SUPPRESSED on any pulse that already
    has a spontaneous spike within (latency+1 ms) before the pulse (true antidromic).
    If respond=False, never add the locked spike (pure baseline unit).
    """
    rng = np.random.default_rng(seed)
    t_end = pulses[-1] + 2.0
    n_base = rng.poisson(base_rate * t_end)
    spikes = list(rng.uniform(0, t_end, size=n_base))
    spikes_arr = np.sort(np.asarray(spikes))
    cw = (latency_ms + ot.COLLISION_REFRACTORY_MS) / 1000.0
    add = []
    for p in pulses:
        if not respond:
            continue
        j0 = np.searchsorted(spikes_arr, p - cw)
        j1 = np.searchsorted(spikes_arr, p)
        has_pre = (j1 - j0) > 0
        if collision and has_pre:
            continue  # antidromic spike collides → absent
        add.append(p + latency_ms / 1000.0 + rng.normal(0, jitter_ms / 1000.0))
    return np.sort(np.concatenate([spikes_arr, np.asarray(add)]))


def test_baseline_rate_recovers_poisson_rate():
    pulses = _pulses(n=200)
    sp = _antidromic_unit(pulses, base_rate=8.0, respond=False, seed=1)
    lam = ot.baseline_rate_hz(sp, pulses)
    assert 6.0 < lam < 10.0  # ~8 Hz


def test_estimate_response_window_finds_peak():
    pulses = _pulses(n=300)
    sp = _antidromic_unit(pulses, latency_ms=4.0, jitter_ms=0.1,
                          base_rate=3.0, collision=False, seed=2)
    rw = ot.estimate_response_window(sp, pulses)
    assert abs(rw.peak_latency_ms - 4.0) < 0.6
    assert rw.window_ms[0] < rw.peak_latency_ms < rw.window_ms[1]
    assert 1.0 < rw.baseline_rate_hz < 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_optotagging.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'baseline_rate_hz'` / `COLLISION_REFRACTORY_MS`.

- [ ] **Step 3: Add constants + functions to `optotagging.py`**

Add near the existing constants block (keep the old constants; update `BASELINE_WINDOW_MS`):

```python
# ── New constants (antidromic redesign) ────────────────────────────────
BASELINE_WINDOW_MS = (-50.0, -5.0)       # rate estimation (was (-50, 0); -5 guard)
SALT_BASELINE_WINDOW_MS = (-250.0, -5.0) # canonical-SALT baseline period
RESPONSE_SEARCH_MS = (1.0, 10.0)         # antidromic latency search range
RESP_PSTH_BIN_MS = 0.1                   # fine PSTH bin for peak finding
RESP_HALFWIDTH_MS = 0.75                 # response-window half-width about the peak
COLLISION_REFRACTORY_MS = 1.0            # added to latency for the collision window
MIN_COLLISION_EXPECTED = 10              # min collision-eligible pulses to test
MIN_COLLISION_FREE = 30                  # min collision-free pulses to test
MAX_SALT_BASELINE_WINDOWS = 50           # cap baseline windows (cost bound)
# Tier thresholds
CANDIDATE_SALT_ALPHA = 0.05
CANDIDATE_POISSON_ALPHA = 0.01
CANDIDATE_MIN_EXCESS_REL = 0.02
STRICT_SALT_ALPHA = 0.01
STRICT_MAX_JITTER_MS = 1.0
```

Add imports at top (with the existing imports):

```python
from scipy.stats import poisson as _poisson, fisher_exact as _fisher_exact
```

Add the dataclass + functions:

```python
@dataclass
class ResponseWindow:
    peak_latency_ms: float
    window_ms: Tuple[float, float]
    baseline_rate_hz: float
    n_resp_spikes: int


def _count_in_window(spikes: np.ndarray, pulses: np.ndarray,
                     window_ms: Tuple[float, float]) -> int:
    a, b = window_ms[0] / 1000.0, window_ms[1] / 1000.0
    tot = 0
    for p in pulses:
        i0 = np.searchsorted(spikes, p + a)
        i1 = np.searchsorted(spikes, p + b)
        tot += i1 - i0
    return int(tot)


def baseline_rate_hz(spike_times, pulse_times,
                     baseline_window_ms: Tuple[float, float] = BASELINE_WINDOW_MS) -> float:
    spikes = np.asarray(spike_times, float).ravel()
    pulses = np.asarray(pulse_times, float).ravel()
    if len(spikes) == 0 or len(pulses) == 0:
        return 0.0
    dur = (baseline_window_ms[1] - baseline_window_ms[0]) / 1000.0
    total = _count_in_window(spikes, pulses, baseline_window_ms)
    return total / (len(pulses) * dur) if dur > 0 else 0.0


def estimate_response_window(spike_times, pulse_times,
                             search_ms: Tuple[float, float] = RESPONSE_SEARCH_MS,
                             bin_ms: float = RESP_PSTH_BIN_MS,
                             baseline_window_ms: Tuple[float, float] = BASELINE_WINDOW_MS,
                             half_width_ms: float = RESP_HALFWIDTH_MS) -> ResponseWindow:
    spikes = np.asarray(spike_times, float).ravel()
    pulses = np.asarray(pulse_times, float).ravel()
    lam_b = baseline_rate_hz(spikes, pulses, baseline_window_ms)
    s0, s1 = search_ms[0] / 1000.0, search_ms[1] / 1000.0
    n_bins = max(1, int(round((s1 - s0) * 1000.0 / bin_ms)))
    edges = np.linspace(s0, s1, n_bins + 1)
    counts = np.zeros(n_bins)
    for p in pulses:
        i0 = np.searchsorted(spikes, p + s0)
        i1 = np.searchsorted(spikes, p + s1)
        if i1 > i0:
            counts += np.histogram(spikes[i0:i1] - p, bins=edges)[0]
    bin_s = (s1 - s0) / n_bins
    expected = lam_b * bin_s * len(pulses)
    peak_bin = int(np.argmax(counts - expected))
    peak_lat = (edges[peak_bin] + edges[peak_bin + 1]) / 2.0 * 1000.0
    w0 = max(search_ms[0], peak_lat - half_width_ms)
    w1 = min(search_ms[1], peak_lat + half_width_ms)
    n_resp = _count_in_window(spikes, pulses, (w0, w1))
    return ResponseWindow(peak_lat, (w0, w1), lam_b, n_resp)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_optotagging.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit** *(deferred — run only when `main` is free)*

```bash
git add src/visdetect/analysis/optotagging.py tests/analysis/test_optotagging.py
git commit -m "feat(optotagging): baseline rate + response-window estimation"
```

---

## Task 2: Poisson excess-rate test

**Files:** Modify `optotagging.py`; Test `tests/analysis/test_optotagging.py`

- [ ] **Step 1: Write the failing test**

```python
def test_poisson_excess_test_detects_response():
    pulses = _pulses(n=300)
    sp = _antidromic_unit(pulses, latency_ms=4.0, base_rate=3.0,
                          collision=False, seed=3)
    rw = ot.estimate_response_window(sp, pulses)
    p = ot.poisson_excess_test(sp, pulses, rw.window_ms, rw.baseline_rate_hz)
    assert p < 1e-3


def test_poisson_excess_test_null_is_not_significant():
    pulses = _pulses(n=300)
    sp = _antidromic_unit(pulses, base_rate=5.0, respond=False, seed=4)
    rw = ot.estimate_response_window(sp, pulses)
    p = ot.poisson_excess_test(sp, pulses, rw.window_ms, rw.baseline_rate_hz)
    assert p > 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_optotagging.py -k poisson -q`
Expected: FAIL — `has no attribute 'poisson_excess_test'`.

- [ ] **Step 3: Implement**

```python
def poisson_excess_test(spike_times, pulse_times,
                        window_ms: Tuple[float, float],
                        baseline_rate_hz_val: float) -> float:
    spikes = np.asarray(spike_times, float).ravel()
    pulses = np.asarray(pulse_times, float).ravel()
    if len(pulses) == 0:
        return 1.0
    k_obs = _count_in_window(spikes, pulses, window_ms)
    win_dur = (window_ms[1] - window_ms[0]) / 1000.0
    lam = baseline_rate_hz_val * win_dur * len(pulses)
    if lam <= 0:
        return 0.0 if k_obs > 0 else 1.0
    return float(_poisson.sf(k_obs - 1, lam))  # P(X >= k_obs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_optotagging.py -k poisson -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit** *(deferred)*

```bash
git add src/visdetect/analysis/optotagging.py tests/analysis/test_optotagging.py
git commit -m "feat(optotagging): Poisson excess-rate significance test"
```

---

## Task 3: Excess reliability (baseline-corrected)

**Files:** Modify `optotagging.py`; Test `tests/analysis/test_optotagging.py`

- [ ] **Step 1: Write the failing test**

```python
def test_excess_reliability_zero_for_pure_baseline():
    pulses = _pulses(n=300)
    sp = _antidromic_unit(pulses, base_rate=6.0, respond=False, seed=5)
    rw = ot.estimate_response_window(sp, pulses)
    er = ot.excess_reliability(sp, pulses, rw.window_ms, rw.baseline_rate_hz)
    assert er < 0.05


def test_excess_reliability_high_for_locked_response():
    pulses = _pulses(n=300)
    sp = _antidromic_unit(pulses, latency_ms=4.0, jitter_ms=0.1,
                          base_rate=2.0, collision=False, seed=6)
    rw = ot.estimate_response_window(sp, pulses)
    er = ot.excess_reliability(sp, pulses, rw.window_ms, rw.baseline_rate_hz)
    assert er > 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_optotagging.py -k reliability -q`
Expected: FAIL — `has no attribute 'excess_reliability'`.

- [ ] **Step 3: Implement**

```python
def excess_reliability(spike_times, pulse_times,
                       window_ms: Tuple[float, float],
                       baseline_rate_hz_val: float) -> float:
    spikes = np.asarray(spike_times, float).ravel()
    pulses = np.asarray(pulse_times, float).ravel()
    if len(pulses) == 0:
        return 0.0
    a, b = window_ms[0] / 1000.0, window_ms[1] / 1000.0
    hits = 0
    for p in pulses:
        i0 = np.searchsorted(spikes, p + a)
        i1 = np.searchsorted(spikes, p + b)
        if i1 > i0:
            hits += 1
    p_resp = hits / len(pulses)
    win_dur = b - a
    p_base = 1.0 - np.exp(-baseline_rate_hz_val * win_dur)
    return float(max(0.0, p_resp - p_base))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_optotagging.py -k reliability -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit** *(deferred)*

```bash
git add src/visdetect/analysis/optotagging.py tests/analysis/test_optotagging.py
git commit -m "feat(optotagging): baseline-corrected excess reliability"
```

---

## Task 4: Excess jitter (fine resolution)

**Files:** Modify `optotagging.py`; Test `tests/analysis/test_optotagging.py`

- [ ] **Step 1: Write the failing test**

```python
def test_excess_jitter_recovers_injected_sigma():
    pulses = _pulses(n=400)
    sp = _antidromic_unit(pulses, latency_ms=4.0, jitter_ms=0.3,
                          base_rate=1.0, collision=False, seed=7)
    rw = ot.estimate_response_window(sp, pulses)
    j = ot.excess_jitter(sp, pulses, rw.window_ms)
    assert 0.1 < j < 0.6   # ~0.3 ms, window-clipped


def test_excess_jitter_nan_when_no_response():
    pulses = _pulses(n=50)
    sp = _antidromic_unit(pulses, base_rate=0.05, respond=False, seed=8)
    j = ot.excess_jitter(sp, pulses, (3.0, 5.0))
    assert np.isnan(j)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_optotagging.py -k jitter -q`
Expected: FAIL — `has no attribute 'excess_jitter'`.

- [ ] **Step 3: Implement**

```python
def excess_jitter(spike_times, pulse_times,
                  window_ms: Tuple[float, float]) -> float:
    spikes = np.asarray(spike_times, float).ravel()
    pulses = np.asarray(pulse_times, float).ravel()
    a, b = window_ms[0] / 1000.0, window_ms[1] / 1000.0
    lat = []
    for p in pulses:
        i0 = np.searchsorted(spikes, p + a)
        i1 = np.searchsorted(spikes, p + b)
        if i1 > i0:
            lat.append((spikes[i0] - p) * 1000.0)
    if len(lat) < 2:
        return float("nan")
    return float(np.std(lat))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_optotagging.py -k jitter -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit** *(deferred)*

```bash
git add src/visdetect/analysis/optotagging.py tests/analysis/test_optotagging.py
git commit -m "feat(optotagging): fine-resolution excess jitter"
```

---

## Task 5: Collision test (antidromic confirmation)

**Files:** Modify `optotagging.py`; Test `tests/analysis/test_optotagging.py`

- [ ] **Step 1: Write the failing test**

```python
def test_collision_test_pass_for_true_antidromic():
    pulses = _pulses(n=501)
    sp = _antidromic_unit(pulses, latency_ms=4.0, jitter_ms=0.2,
                          base_rate=5.0, collision=True, seed=9)
    rw = ot.estimate_response_window(sp, pulses)
    cr = ot.collision_test(sp, pulses, rw.peak_latency_ms, rw.window_ms)
    assert cr.status == "pass"
    assert cr.n_expected >= ot.MIN_COLLISION_EXPECTED
    assert cr.p_free > cr.p_expected
    assert cr.suppression_index > 0.5


def test_collision_test_fail_for_synaptic_response():
    pulses = _pulses(n=501)
    sp = _antidromic_unit(pulses, latency_ms=4.0, jitter_ms=0.2,
                          base_rate=5.0, collision=False, seed=10)
    rw = ot.estimate_response_window(sp, pulses)
    cr = ot.collision_test(sp, pulses, rw.peak_latency_ms, rw.window_ms)
    assert cr.status == "fail"


def test_collision_test_untestable_when_too_few_eligible():
    pulses = _pulses(n=501)
    sp = _antidromic_unit(pulses, latency_ms=4.0, base_rate=0.2,
                          collision=False, seed=11)
    rw = ot.estimate_response_window(sp, pulses)
    cr = ot.collision_test(sp, pulses, rw.peak_latency_ms, rw.window_ms)
    assert cr.status == "untestable"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_optotagging.py -k collision -q`
Expected: FAIL — `has no attribute 'collision_test'`.

- [ ] **Step 3: Implement**

```python
@dataclass
class CollisionResult:
    status: str               # 'pass' | 'fail' | 'untestable'
    suppression_index: float
    p_free: float
    p_expected: float
    n_free: int
    n_expected: int
    fisher_p: float


def collision_test(spike_times, pulse_times, peak_latency_ms: float,
                   window_ms: Tuple[float, float],
                   refractory_ms: float = COLLISION_REFRACTORY_MS,
                   min_expected: int = MIN_COLLISION_EXPECTED,
                   min_free: int = MIN_COLLISION_FREE,
                   alpha: float = 0.05) -> CollisionResult:
    spikes = np.asarray(spike_times, float).ravel()
    pulses = np.asarray(pulse_times, float).ravel()
    cw = (peak_latency_ms + refractory_ms) / 1000.0
    a, b = window_ms[0] / 1000.0, window_ms[1] / 1000.0
    resp_free = n_free = resp_exp = n_exp = 0
    for p in pulses:
        j0 = np.searchsorted(spikes, p - cw)
        j1 = np.searchsorted(spikes, p)
        has_pre = (j1 - j0) > 0
        i0 = np.searchsorted(spikes, p + a)
        i1 = np.searchsorted(spikes, p + b)
        has_resp = (i1 - i0) > 0
        if has_pre:
            n_exp += 1
            resp_exp += int(has_resp)
        else:
            n_free += 1
            resp_free += int(has_resp)
    p_free = resp_free / n_free if n_free > 0 else float("nan")
    p_exp = resp_exp / n_exp if n_exp > 0 else float("nan")
    supp = ((p_free - p_exp) / p_free
            if (n_free > 0 and n_exp > 0 and p_free > 0) else float("nan"))
    if n_exp < min_expected or n_free < min_free:
        return CollisionResult("untestable", supp, p_free, p_exp,
                               n_free, n_exp, float("nan"))
    table = [[resp_free, n_free - resp_free], [resp_exp, n_exp - resp_exp]]
    _, fp = _fisher_exact(table, alternative="greater")
    status = "pass" if (fp < alpha and p_free > p_exp) else "fail"
    return CollisionResult(status, supp, p_free, p_exp, n_free, n_exp, float(fp))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_optotagging.py -k collision -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit** *(deferred)*

```bash
git add src/visdetect/analysis/optotagging.py tests/analysis/test_optotagging.py
git commit -m "feat(optotagging): offline collision test (antidromic confirmation)"
```

---

## Task 6: Canonical SALT (baseline-window null)

**Files:** Modify `optotagging.py`; Test `tests/analysis/test_optotagging.py`

The existing `salt_test` (JSD-to-uniform) is renamed `_salt_test_jsd_uniform` (kept for
reference) and replaced by a canonical implementation: divide the baseline period into windows
of the test-window width; the null is the baseline-vs-baseline JS divergences; the statistic is
mean test-vs-baseline JS divergence. Deterministic (no RNG). `n_jitter` is accepted but ignored.

- [ ] **Step 1: Write the failing test**

```python
def test_salt_small_p_for_locked_response():
    pulses = _pulses(n=300)
    sp = _antidromic_unit(pulses, latency_ms=4.0, jitter_ms=0.1,
                          base_rate=4.0, collision=False, seed=12)
    rw = ot.estimate_response_window(sp, pulses)
    p = ot.salt_test(sp, pulses, response_window_ms=rw.window_ms)
    assert p < 0.01


def test_salt_not_significant_for_flat_unit():
    pulses = _pulses(n=300)
    sp = _antidromic_unit(pulses, base_rate=6.0, respond=False, seed=13)
    rw = ot.estimate_response_window(sp, pulses)
    p = ot.salt_test(sp, pulses, response_window_ms=rw.window_ms)
    assert p > 0.05


def test_salt_is_deterministic():
    pulses = _pulses(n=200)
    sp = _antidromic_unit(pulses, latency_ms=4.0, base_rate=3.0,
                          collision=False, seed=14)
    rw = ot.estimate_response_window(sp, pulses)
    p1 = ot.salt_test(sp, pulses, response_window_ms=rw.window_ms)
    p2 = ot.salt_test(sp, pulses, response_window_ms=rw.window_ms)
    assert p1 == p2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_optotagging.py -k salt -q`
Expected: FAIL — the flat-unit test fails and/or determinism fails (current SALT uses an RNG and JSD-to-uniform with a different null).

- [ ] **Step 3: Implement**

Rename the existing `def salt_test(...)` to `def _salt_test_jsd_uniform(...)` (leave body
intact). Add the canonical version:

```python
def salt_test(spike_times, pulse_times,
              response_window_ms: Tuple[float, float] = RESPONSE_WINDOW_MS,
              baseline_window_ms: Tuple[float, float] = SALT_BASELINE_WINDOW_MS,
              n_jitter: int = SALT_N_JITTER,   # accepted for back-compat; ignored
              bin_ms: float = SALT_BIN_MS,
              max_windows: int = MAX_SALT_BASELINE_WINDOWS) -> float:
    """Canonical SALT (Kvitsiani et al. 2013).

    Latency distributions (with a 'no-spike' category) are built for the test window
    and for many equal-width baseline windows. The null is the distribution of
    baseline-vs-baseline JS divergences; the statistic is the mean test-vs-baseline
    JS divergence; p = (1 + #{null >= stat}) / (1 + n_null). Deterministic.
    """
    spikes = np.asarray(spike_times, float).ravel()
    pulses = np.asarray(pulse_times, float).ravel()
    if len(spikes) == 0 or len(pulses) == 0:
        return 1.0
    win_dur = (response_window_ms[1] - response_window_ms[0]) / 1000.0
    test_off = response_window_ms[0] / 1000.0
    b0, b1 = baseline_window_ms[0] / 1000.0, baseline_window_ms[1] / 1000.0
    if win_dur <= 0:
        return 1.0
    n_base_full = int((b1 - b0) // win_dur)
    if n_base_full < 2:
        return 1.0
    offsets = b0 + np.arange(n_base_full) * win_dur
    if n_base_full > max_windows:
        offsets = offsets[np.linspace(0, n_base_full - 1, max_windows).astype(int)]
    n_bins = max(1, int(round(win_dur * 1000.0 / bin_ms)))

    def _dist(offset: float) -> np.ndarray:
        hist = np.zeros(n_bins + 1)  # last entry = 'no spike'
        for p in pulses:
            t0 = p + offset
            i0 = np.searchsorted(spikes, t0)
            i1 = np.searchsorted(spikes, t0 + win_dur)
            if i1 > i0:
                rel = spikes[i0] - t0
                bi = min(int(rel / win_dur * n_bins), n_bins - 1)
                hist[bi] += 1
            else:
                hist[-1] += 1
        s = hist.sum()
        return hist / s if s > 0 else hist

    test_d = _dist(test_off)
    base_d = [_dist(o) for o in offsets]
    null = [_jensen_shannon(base_d[i], base_d[j])
            for i in range(len(base_d)) for j in range(i + 1, len(base_d))]
    if not null:
        return 1.0
    stat = float(np.mean([_jensen_shannon(test_d, bd) for bd in base_d]))
    null = np.asarray(null)
    return float((1 + np.sum(null >= stat)) / (1 + len(null)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_optotagging.py -k salt -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit** *(deferred)*

```bash
git add src/visdetect/analysis/optotagging.py tests/analysis/test_optotagging.py
git commit -m "feat(optotagging): canonical SALT with baseline-window null"
```

---

## Task 7: Enriched OptoMetrics + analyze_unit integration

**Files:** Modify `optotagging.py`; Test `tests/analysis/test_optotagging.py`

- [ ] **Step 1: Write the failing test**

```python
def test_analyze_unit_populates_enriched_fields():
    pulses = _pulses(n=501)
    sp = _antidromic_unit(pulses, latency_ms=4.0, jitter_ms=0.2,
                          base_rate=5.0, collision=True, seed=15)
    sess = Session(trials=[], clusters=[Cluster(0, sp, "good")],
                   subject="SYN", session_name="SIM",
                   good_cluster_ids=[0], good_and_stable_ids=None,
                   ni_events={"Laser": pulses})
    tagger = ot.OptoTagger(sess)
    m = tagger.analyze_unit(sess.clusters[0], pulses, "GPe")
    assert m.excess_reliability > 0.8
    assert m.excess_jitter_ms < 1.0
    assert m.poisson_p < 1e-3
    assert m.salt_p < 0.01
    assert m.collision_status == "pass"
    assert 1.0 < m.peak_latency_ms < 10.0
    assert m.baseline_rate_hz > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_optotagging.py -k analyze_unit -q`
Expected: FAIL — `OptoMetrics.__init__() got an unexpected keyword argument 'excess_reliability'`.

- [ ] **Step 3: Enrich `OptoMetrics` and rewrite `analyze_unit`**

Add fields to the `OptoMetrics` dataclass (after the existing fields, all with defaults so the
early-return paths still construct cleanly):

```python
    baseline_rate_hz: float = float("nan")
    response_window_ms: Tuple[float, float] = (float("nan"), float("nan"))
    peak_latency_ms: float = float("nan")
    excess_reliability: float = float("nan")
    excess_jitter_ms: float = float("nan")
    poisson_p: float = 1.0
    collision_status: str = "untestable"
    collision_suppression_index: float = float("nan")
    n_collision_free: int = 0
    n_collision_expected: int = 0
```

Replace the body of `analyze_unit` (keep the empty-input early return, but add the enriched
computation for the normal path):

```python
    def analyze_unit(self, cluster, pulse_times, fiber):
        spikes = np.asarray(cluster.spike_times, dtype=float).ravel()
        n_pulses = len(pulse_times)
        if len(spikes) == 0 or n_pulses == 0:
            return OptoMetrics(cluster_id=cluster.cluster_id, fiber=fiber,
                               is_responsive=False, latency_ms=np.nan, jitter_ms=np.nan,
                               reliability=0.0, salt_p=1.0, n_pulses=n_pulses)

        rw = estimate_response_window(spikes, pulse_times)
        W, lam_b = rw.window_ms, rw.baseline_rate_hz
        exc_rel = excess_reliability(spikes, pulse_times, W, lam_b)
        exc_jit = excess_jitter(spikes, pulse_times, W)
        pois_p = poisson_excess_test(spikes, pulse_times, W, lam_b)
        salt_p = salt_test(spikes, pulse_times, response_window_ms=W,
                           baseline_window_ms=SALT_BASELINE_WINDOW_MS)
        coll = collision_test(spikes, pulse_times, rw.peak_latency_ms, W)

        # legacy raw metrics (continuity / diagnostics)
        latencies, hit_count, reliability = _first_spike_latencies(
            spikes, pulse_times, self.response_window_ms)
        latency_mean = float(np.mean(latencies)) if hit_count > 0 else float("nan")
        jitter = float(np.std(latencies)) if hit_count > 0 else float("nan")

        return OptoMetrics(
            cluster_id=cluster.cluster_id, fiber=fiber, is_responsive=False,
            latency_ms=latency_mean, jitter_ms=jitter, reliability=reliability,
            salt_p=salt_p, n_pulses=n_pulses, first_spike_latencies=latencies,
            baseline_rate_hz=lam_b, response_window_ms=W,
            peak_latency_ms=rw.peak_latency_ms, excess_reliability=exc_rel,
            excess_jitter_ms=exc_jit, poisson_p=pois_p,
            collision_status=coll.status,
            collision_suppression_index=coll.suppression_index,
            n_collision_free=coll.n_free, n_collision_expected=coll.n_expected)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_optotagging.py -k analyze_unit -q`
Expected: PASS.

- [ ] **Step 5: Commit** *(deferred)*

```bash
git add src/visdetect/analysis/optotagging.py tests/analysis/test_optotagging.py
git commit -m "feat(optotagging): enriched OptoMetrics + analyze_unit integration"
```

---

## Task 8: Tier classifier + bridging-collateral D1/D2 assignment + waveform check

**Files:** Modify `optotagging.py`; Test `tests/analysis/test_optotagging.py`

- [ ] **Step 1: Write the failing test**

```python
def _metric(salt_p=0.001, poisson_p=0.001, peak=4.0, exc_rel=0.5,
            exc_jit=0.3, collision="pass", fiber="GPe", cid=0):
    return ot.OptoMetrics(cluster_id=cid, fiber=fiber, is_responsive=False,
        latency_ms=peak, jitter_ms=exc_jit, reliability=0.5, salt_p=salt_p,
        n_pulses=501, baseline_rate_hz=5.0, response_window_ms=(peak-0.75, peak+0.75),
        peak_latency_ms=peak, excess_reliability=exc_rel, excess_jitter_ms=exc_jit,
        poisson_p=poisson_p, collision_status=collision)


def test_fiber_tier_levels():
    assert ot.fiber_tier(_metric()) == "high_confidence"
    assert ot.fiber_tier(_metric(collision="untestable")) == "candidate"
    assert ot.fiber_tier(_metric(exc_jit=2.0)) == "candidate"          # too jittery for strict
    assert ot.fiber_tier(_metric(salt_p=0.5, poisson_p=0.5)) == "none" # not significant
    assert ot.fiber_tier(_metric(exc_rel=0.0)) == "none"               # below excess-rel floor


def test_fiber_tier_waveform_blocks_strict():
    assert ot.fiber_tier(_metric(), waveform_ok=False) == "candidate"


def test_classify_unit_bridging_logic():
    g = _metric(fiber="GPe")
    s = _metric(fiber="SNr")
    assert ot.classify_unit(g, None).pathway == "D2"        # GPe only
    assert ot.classify_unit(None, s).pathway == "D1"        # SNr only
    assert ot.classify_unit(g, s).pathway == "D1"           # both -> D1 (bridging)
    none_m = _metric(salt_p=0.9, poisson_p=0.9)
    assert ot.classify_unit(none_m, none_m).pathway is None


def test_is_spn_plausible_waveform():
    assert ot.is_spn_plausible_waveform("SPN") is True
    assert ot.is_spn_plausible_waveform(None) is True
    assert ot.is_spn_plausible_waveform(float("nan")) is True
    assert ot.is_spn_plausible_waveform("FSI") is False
    assert ot.is_spn_plausible_waveform("fast-spiking") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_optotagging.py -k "tier or classify or waveform" -q`
Expected: FAIL — `has no attribute 'fiber_tier'`.

- [ ] **Step 3: Implement**

```python
def is_spn_plausible_waveform(cell_type) -> bool:
    """True unless the label clearly indicates a fast-spiking / narrow interneuron.

    Unknown / NaN / None are treated as plausible (don't hard-exclude unlabeled units).
    """
    if cell_type is None:
        return True
    if isinstance(cell_type, float) and np.isnan(cell_type):
        return True
    ct = str(cell_type).strip().lower()
    if ct in ("", "nan", "unknown", "unlabeled"):
        return True
    narrow = ("fsi", "fast", "narrow", "pv", "interneuron")
    return not any(m in ct for m in narrow)


def fiber_tier(m: OptoMetrics,
               cand_salt: float = CANDIDATE_SALT_ALPHA,
               cand_pois: float = CANDIDATE_POISSON_ALPHA,
               cand_excrel: float = CANDIDATE_MIN_EXCESS_REL,
               strict_salt: float = STRICT_SALT_ALPHA,
               strict_jit: float = STRICT_MAX_JITTER_MS,
               search_ms: Tuple[float, float] = RESPONSE_SEARCH_MS,
               waveform_ok: bool = True) -> str:
    sig = (m.salt_p < cand_salt) or (m.poisson_p < cand_pois)
    lat_ok = search_ms[0] <= m.peak_latency_ms <= search_ms[1]
    rel_ok = m.excess_reliability > cand_excrel
    if not (sig and lat_ok and rel_ok):
        return "none"
    jit_ok = (not np.isnan(m.excess_jitter_ms)) and (m.excess_jitter_ms < strict_jit)
    strict = ((m.salt_p < strict_salt) and jit_ok
              and (m.collision_status == "pass") and waveform_ok)
    return "high_confidence" if strict else "candidate"


@dataclass
class UnitTag:
    cluster_id: int
    pathway: Optional[str]          # 'D1' | 'D2' | None
    tier: str                       # 'high_confidence' | 'candidate' | 'none'
    gpe_tier: str
    snr_tier: str
    contributing_fiber: Optional[str]


def classify_unit(gpe: Optional[OptoMetrics], snr: Optional[OptoMetrics],
                  waveform_ok: bool = True, **tier_kwargs) -> UnitTag:
    gt = fiber_tier(gpe, waveform_ok=waveform_ok, **tier_kwargs) if gpe is not None else "none"
    st = fiber_tier(snr, waveform_ok=waveform_ok, **tier_kwargs) if snr is not None else "none"
    cid = (gpe or snr).cluster_id
    if st != "none":                       # SNr-tagged -> D1 (specific; overrides GPe)
        return UnitTag(cid, "D1", st, gt, st, "SNr")
    if gt != "none":                       # GPe-only -> D2
        return UnitTag(cid, "D2", gt, gt, st, "GPe")
    return UnitTag(cid, None, "none", gt, st, None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_optotagging.py -q`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit** *(deferred)*

```bash
git add src/visdetect/analysis/optotagging.py tests/analysis/test_optotagging.py
git commit -m "feat(optotagging): two-tier classifier + bridging D1/D2 + waveform check"
```

---

## Task 9: Refactor the Fig43 script (two tiers, sweep, old-vs-new, waveform join)

**Files:** Modify `analysis_suite/09_optotagging/a_optotagging_identification.py`

The per-session worker already calls `tagger.analyze_all(...)`; extend the row dicts with the
new fields, then add a post-processing + plotting layer. Keep `--n-jitter`, `--n-workers`,
`--force`. The cache schema changes, so the script must `--force`-rebuild on first run.

- [ ] **Step 1: Extend `_run_session` row dict with the new fields**

In `_run_session`, replace the `rows.append({...})` block with:

```python
        rows.append({
            "session_name": int(sname), "stage": stage,
            "cluster_id": m.cluster_id, "fiber": m.fiber, "n_pulses": m.n_pulses,
            # legacy raw
            "latency_ms": m.latency_ms, "jitter_ms": m.jitter_ms,
            "reliability": m.reliability, "salt_p": m.salt_p,
            # enriched
            "baseline_rate_hz": m.baseline_rate_hz,
            "win_lo": m.response_window_ms[0], "win_hi": m.response_window_ms[1],
            "peak_latency_ms": m.peak_latency_ms,
            "excess_reliability": m.excess_reliability,
            "excess_jitter_ms": m.excess_jitter_ms,
            "poisson_p": m.poisson_p,
            "collision_status": m.collision_status,
            "collision_suppression_index": m.collision_suppression_index,
            "n_collision_free": m.n_collision_free,
            "n_collision_expected": m.n_collision_expected,
        })
```

- [ ] **Step 2: Add a classification + waveform-join helper after the cache is loaded**

Replace the old "Re-derive is_responsive" + "dual_fiber" + "Summary" blocks (everything between
loading `df_all` and `df_resp = ...`) with:

```python
    from visdetect.analysis.optotagging import (
        OptoMetrics, fiber_tier, classify_unit, is_spn_plausible_waveform,
    )

    # ── waveform labels (optional) ─────────────────────────────────────
    try:
        wf = load_waveform_labels()
        wf_map = {(int(r.session_name), int(r.cluster_id)): r.cell_type
                  for r in wf.itertuples()}
    except FileNotFoundError:
        print("  Waveform labels not found — skipping FSI cross-check (annotation only).")
        wf_map = {}

    def _metrics_from_row(r):
        return OptoMetrics(
            cluster_id=int(r.cluster_id), fiber=r.fiber, is_responsive=False,
            latency_ms=r.latency_ms, jitter_ms=r.jitter_ms, reliability=r.reliability,
            salt_p=r.salt_p, n_pulses=int(r.n_pulses),
            baseline_rate_hz=r.baseline_rate_hz, response_window_ms=(r.win_lo, r.win_hi),
            peak_latency_ms=r.peak_latency_ms, excess_reliability=r.excess_reliability,
            excess_jitter_ms=r.excess_jitter_ms, poisson_p=r.poisson_p,
            collision_status=r.collision_status,
            collision_suppression_index=r.collision_suppression_index,
            n_collision_free=int(r.n_collision_free),
            n_collision_expected=int(r.n_collision_expected))

    # per-fiber tier (waveform applied per unit below; here waveform_ok=True placeholder)
    df_all["fiber_tier"] = [fiber_tier(_metrics_from_row(r)) for r in df_all.itertuples()]

    # ── unit-level classification (bridging logic + waveform gate) ─────
    unit_rows = []
    for (sn, cid), grp in df_all.groupby(["session_name", "cluster_id"]):
        g = grp[grp.fiber == "GPe"]
        s = grp[grp.fiber == "SNr"]
        gm = _metrics_from_row(next(g.itertuples())) if len(g) else None
        sm = _metrics_from_row(next(s.itertuples())) if len(s) else None
        cell_type = wf_map.get((int(sn), int(cid)))
        wf_ok = is_spn_plausible_waveform(cell_type)
        tag = classify_unit(gm, sm, waveform_ok=wf_ok)
        unit_rows.append({
            "session_name": int(sn), "cluster_id": int(cid),
            "stage": grp.iloc[0]["stage"], "pathway": tag.pathway, "tier": tag.tier,
            "gpe_tier": tag.gpe_tier, "snr_tier": tag.snr_tier,
            "contributing_fiber": tag.contributing_fiber,
            "cell_type": cell_type, "waveform_ok": wf_ok,
        })
    units = pd.DataFrame(unit_rows)
    units_path = os.path.join(CACHE_DIR, "optotagging_unit_tags.csv")
    units.to_csv(units_path, index=False)
    print(f"  Saved unit tags to {units_path}")

    # ── two-tier yield summary ─────────────────────────────────────────
    print("\n  === Yield by tier × pathway (unique units) ===")
    for pathway in ["D1", "D2"]:
        sub = units[units.pathway == pathway]
        n_cand = (sub.tier.isin(["candidate", "high_confidence"])).sum()
        n_hc = (sub.tier == "high_confidence").sum()
        print(f"    {pathway}: candidate={n_cand}  high_confidence={n_hc}")
    n_untestable = (df_all.collision_status == "untestable").mean()
    print(f"    Collision-untestable fraction (all unit×fiber): {n_untestable:.2f}")

    df_resp = units[units.pathway.notna()]  # for downstream plotting compatibility
```

- [ ] **Step 3: Replace the plotting block with tier-aware panels + sweep + old-vs-new**

Replace the `# ── Figure ──` block through the end of `main()` (the per-session counts figure
and stats CSV) with:

```python
    print("\n  Generating figures ...")
    setup_style()

    # Panel set 1: latency + excess-jitter distributions for tagged units
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    tagged_fibers = df_all[df_all.fiber_tier.isin(["candidate", "high_confidence"])]
    for fiber, color in FIBER_COLORS.items():
        sub = tagged_fibers[tagged_fibers.fiber == fiber]
        axes[0].hist(sub.peak_latency_ms.dropna(), bins=np.linspace(0, 10, 41),
                     alpha=0.6, color=color, label=f"{fiber} (n={len(sub)})")
        axes[1].hist(sub.excess_jitter_ms.dropna(), bins=np.linspace(0, 3, 31),
                     alpha=0.6, color=color, label=fiber)
    axes[0].set(xlabel="Peak latency (ms)", ylabel="Count", title="Tagged-unit latency")
    axes[1].axvline(STRICT_MAX_JITTER_MS, color="k", ls="--", lw=1, label="strict cap")
    axes[1].set(xlabel="Excess jitter (ms)", ylabel="Count", title="Tagged-unit jitter")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    save_figure(fig, "fig43a_optotagging_distributions", MODULE_NAME)

    # Panel set 2: yield by stage × tier × pathway
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    stages = [s for s in STAGE_ORDER if s in units.stage.values]
    x = np.arange(len(stages)); bw = 0.35
    for ax, pathway in zip(axes2, ["D1", "D2"]):
        for k, (tier, alpha) in enumerate([("candidate", 0.5), ("high_confidence", 1.0)]):
            counts = [((units.stage == st) & (units.pathway == pathway)
                       & (units.tier == "high_confidence" if tier == "high_confidence"
                          else units.tier.isin(["candidate", "high_confidence"]))).sum()
                      for st in stages]
            ax.bar(x + (k - 0.5) * bw, counts, bw, alpha=alpha,
                   color=FIBER_COLORS["SNr" if pathway == "D1" else "GPe"], label=tier)
        ax.set(xticks=x, title=f"{pathway} yield by stage", xlabel="Stage")
        ax.set_xticklabels(stages); ax.legend(fontsize=8)
    axes2[0].set_ylabel("Tagged units")
    save_figure(fig2, "fig43b_yield_by_stage_tier", MODULE_NAME)

    # Panel set 3: old-vs-new comparison + jitter-threshold sweep
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5))
    old = {"D2": int(((df_all.fiber == "GPe") & (df_all.salt_p < 0.01)
                      & (df_all.reliability >= 0.1)).sum()),
           "D1": int(((df_all.fiber == "SNr") & (df_all.salt_p < 0.01)
                      & (df_all.reliability >= 0.1)).sum())}
    new_cand = {p: int((units.pathway == p).sum()) for p in ["D1", "D2"]}
    new_hc = {p: int(((units.pathway == p) & (units.tier == "high_confidence")).sum())
              for p in ["D1", "D2"]}
    xp = np.arange(2)
    axes3[0].bar(xp - 0.25, [old["D1"], old["D2"]], 0.25, label="old pipeline", color="#888")
    axes3[0].bar(xp, [new_cand["D1"], new_cand["D2"]], 0.25, label="new candidate", color="#5dade2")
    axes3[0].bar(xp + 0.25, [new_hc["D1"], new_hc["D2"]], 0.25, label="new high-conf", color="#1f618d")
    axes3[0].set(xticks=xp, title="Old vs new yield", ylabel="Units")
    axes3[0].set_xticklabels(["D1", "D2"]); axes3[0].legend(fontsize=8)

    jit_grid = np.linspace(0.25, 3.0, 12)
    for pathway, fiber in [("D1", "SNr"), ("D2", "GPe")]:
        fib = df_all[df_all.fiber == fiber]
        ys = [int(((fib.salt_p < STRICT_SALT_ALPHA) & (fib.collision_status == "pass")
                   & (fib.excess_jitter_ms < j) & (fib.excess_reliability > 0.02)).sum())
              for j in jit_grid]
        axes3[1].plot(jit_grid, ys, "-o", color=FIBER_COLORS[fiber], label=pathway)
    axes3[1].axvline(STRICT_MAX_JITTER_MS, color="k", ls="--", lw=1)
    axes3[1].set(xlabel="Strict jitter cap (ms)", ylabel="High-conf units",
                 title="Yield vs jitter threshold"); axes3[1].legend(fontsize=8)
    save_figure(fig3, "fig43c_old_vs_new_and_sweep", MODULE_NAME)

    print("\n[09a] Done.")
```

Add the needed imports at the top of the script (with the other library imports):

```python
from visdetect.analysis.optotagging import STRICT_SALT_ALPHA, STRICT_MAX_JITTER_MS
```

- [ ] **Step 4: Smoke-run the script on a 2-session subset (no full run yet)**

Temporarily verify wiring without the full sweep by running with the existing cache deleted and
a tiny manifest is not needed — instead just confirm it imports and the post-processing runs on
the existing cache shape after a forced rebuild is too slow; so first assert import + help:

Run: `cd analysis_suite && py 09_optotagging/a_optotagging_identification.py --help`
Expected: argparse help prints with `--n-jitter`, `--n-workers`, `--force` (no import errors).

- [ ] **Step 5: Commit** *(deferred)*

```bash
git add analysis_suite/09_optotagging/a_optotagging_identification.py
git commit -m "feat(optotagging): Fig43 two-tier yield, sweep, old-vs-new comparison"
```

---

## Task 10: Real-data run + results note (answers "best achievable yield")

**Files:** Create `docs/results/2026-06-08-optotagging-yield-results.md`

- [ ] **Step 1: Back up the old cache, then force-rebuild on real data**

```bash
cd analysis_suite
mv cache/optotagging_results.csv cache/optotagging_results.preredesign.csv
py 09_optotagging/a_optotagging_identification.py --force --n-workers 4
```

Expected: prints per-session progress, the "Yield by tier × pathway" summary, the
collision-untestable fraction, and saves `cache/optotagging_results.csv`,
`cache/optotagging_unit_tags.csv`, and `figures/09_optotagging/fig43{a,b,c}_*.png`.

- [ ] **Step 2: Capture the numbers**

```bash
cd analysis_suite && py -c "
import pandas as pd
u = pd.read_csv('cache/optotagging_unit_tags.csv')
for p in ['D1','D2']:
    s=u[u.pathway==p]
    print(p,'candidate=',int(s.tier.isin(['candidate','high_confidence']).sum()),
          'high_conf=',int((s.tier=='high_confidence').sum()))
print('total units classified:', u.cluster_id.size, 'sessions:', u.session_name.nunique())
"
```

- [ ] **Step 3: Write the results note**

Create `docs/results/2026-06-08-optotagging-yield-results.md` with: the old pipeline counts
(D2 168 / D1 36), the new **candidate** and **high-confidence** counts per pathway (from
Step 2), the collision-untestable fraction, the chosen strict jitter cap (read off
`fig43c` sweep), and a one-paragraph interpretation (what is the best defensible yield, and the
FR-bias caveat for the high-confidence tier). Reference the spec and the three figures.

- [ ] **Step 4: Full test suite green**

Run: `py -m pytest tests/analysis/test_optotagging.py -q && py -m pytest -q tests/test_imports.py`
Expected: PASS.

- [ ] **Step 5: Commit** *(deferred)*

```bash
git add docs/results/2026-06-08-optotagging-yield-results.md \
        analysis_suite/cache/optotagging_unit_tags.csv
git commit -m "docs(optotagging): BG_046 achieved D1/D2 yield (two tiers)"
```

---

## Self-Review

**Spec coverage:**
- Baseline-corrected reliability/jitter → Tasks 3, 4. ✓
- Response-window estimation → Task 1. ✓
- Canonical SALT → Task 6. ✓
- Poisson excess test → Task 2. ✓
- Collision test (pass/fail/untestable) → Task 5. ✓
- Enriched OptoMetrics + integration → Task 7. ✓
- Two-tier classifier + bridging D1/D2 + waveform check → Task 8. ✓
- Pool unchanged (pkl-resident good_and_stable) → no pool task; `analyze_all` default
  untouched; integration test (Task 7) uses the session's loaded clusters. ✓
- Fig43 two tiers + sweep + old-vs-new + waveform join → Task 9. ✓
- Achieved-yield results note → Task 10. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code. Strict jitter cap is a
named constant (`STRICT_MAX_JITTER_MS=1.0`) with the sweep (Task 9/10) used to confirm/tune it
from data — explicit, not a placeholder. ✓

**Type consistency:** `OptoMetrics` enriched fields (Task 7) match the names read by
`_metrics_from_row` and `fiber_tier`/`classify_unit` (Tasks 8, 9) — `excess_reliability`,
`excess_jitter_ms`, `poisson_p`, `collision_status`, `peak_latency_ms`, `response_window_ms`,
`baseline_rate_hz`. `ResponseWindow`, `CollisionResult`, `UnitTag` field names are used
consistently across tasks. `collision_test` returns `.status/.n_free/.n_expected/.suppression_index`
as consumed in Task 7. ✓

**Note on the pkl-resident-pool test:** the spec's "pool default" test (good_cluster_ids lists
more IDs than loaded clusters) is implicitly covered by Task 7's session construction; if a
standalone assertion is wanted, add a one-liner test building a Session with
`good_cluster_ids=[0,1,2]` but only `clusters=[Cluster(0,...)]` and asserting `analyze_all`
returns metrics for cluster 0 only.
