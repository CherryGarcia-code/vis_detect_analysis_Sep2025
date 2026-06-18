# TF-Responsiveness Drift-Correction Core — Implementation Plan (Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate the source-level drift-correction core for TF-responsive cell identification — a fixed pulse-selection guard, a new `tf_drift.py` library (drift estimate → detrended pulse-triggered average → null bank → QC), and a Phase-0 validation script that proves on real exemplar units that the detrend flattens the pre-pulse baseline without eating the response.

**Architecture:** Reuse `_collect_pulses` / `_mean_activity_per_unit` / `_zscore_trace` from `tf_pulse.py`; add pure, testable functions in `tf_drift.py` that estimate each unit's slow firing drift over the whole session and subtract it at the source before pulse-triggered averaging. A standalone Phase-0 script renders before/after traces with a circular-shift null envelope and reports the pre-pulse-slope QC. No GUI, no model, no full-session scaling yet (those are Plans 2–3).

**Tech Stack:** Python 3.10, numpy, scipy (`gaussian_filter1d`), matplotlib (Agg), pytest. Tests import `visdetect` via the repo `conftest.py` (prepends `src/`).

**Spec:** `docs/superpowers/specs/2026-06-15-tf-responsiveness-labeler-design.md` (§5 Drift correction & Pulse selection; Phase 0).

**Scope note:** This is Plan 1 of 3. Plan 2 = Phase 1 completion (state-conditioned all/engaged/disengaged averaging + full-session NPZ caches + population QC). Plan 3 = Phases 2–4 (tagger GUI, active-learning model, integration). Each builds on validated outputs of the prior.

---

## File Structure

- **Modify** `src/visdetect/analysis/tf_pulse.py` — fix `_outcome_time_for_trial` (case-robust; cover `fa`/`abort`/`ref`).
- **Create** `src/visdetect/analysis/tf_drift.py` — `estimate_drift`, `detrended_pulse_average`, `prepulse_slope`, `circular_shift_null`. One responsibility: turn a unit's spikes + pulse times into a drift-corrected, z-scored pulse-triggered trace plus its null envelope and QC.
- **Create** `scripts/tf_responsiveness/validate_drift_phase0.py` — Phase-0 eyeball gate: run the core on a handful of `session:cluster` exemplars, render before/after + null envelope, print the pre-pulse-slope table.
- **Create** `tests/analysis/test_tf_pulse_guards.py` — guard-fix tests.
- **Create** `tests/analysis/test_tf_drift.py` — drift-math tests.

---

### Task 1: Fix the early-lick guard in `_collect_pulses`

**Files:**
- Modify: `src/visdetect/analysis/tf_pulse.py:92-103` (`_outcome_time_for_trial`)
- Test: `tests/analysis/test_tf_pulse_guards.py`

The current guard matches `out in ("FA","abort")` — `"FA"` uppercase, `"abort"` lowercase — and looks up `rts.get(out, ...)` with the original-case key. Real/synthetic outcomes are capitalized (`"FA"`, `"Hit"`), so the `"abort"` branch is unreachable, and lowercase data would break the `"FA"` branch. Make it case-insensitive and cover the three baseline-lick outcomes (`fa`, `abort`, `ref`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/analysis/test_tf_pulse_guards.py
import numpy as np
from visdetect.core.session import Trial
from visdetect.analysis.tf_pulse import _outcome_time_for_trial, _collect_pulses, TFRespPulseConfig
from visdetect.core.session import Session


def _trial(outcome, rts):
    return Trial(trialoutcome=outcome, reactiontimes=rts)


def test_outcome_time_uppercase_fa():
    # Existing behaviour must still work: FA lick at baseline_t + rt
    t = _trial("FA", {"FA": 3.5})
    assert _outcome_time_for_trial(t, 10.0) == 10.0 + 3.5


def test_outcome_time_lowercase_fa_is_now_caught():
    # Was silently None before the fix (lowercase 'fa' != 'FA')
    t = _trial("fa", {"fa": 3.5})
    assert _outcome_time_for_trial(t, 10.0) == 13.5


def test_outcome_time_capitalised_abort_is_now_caught():
    # Was silently None before the fix ('Abort' != 'abort')
    t = _trial("Abort", {"Abort": 1.2})
    assert _outcome_time_for_trial(t, 10.0) == 11.2


def test_outcome_time_ref_is_covered():
    t = _trial("ref", {"ref": 0.2})
    assert _outcome_time_for_trial(t, 10.0) == 10.2


def test_outcome_time_hit_returns_none():
    # Hit is not a baseline lick -> no early-reaction time
    t = _trial("Hit", {"RT": 0.3})
    assert _outcome_time_for_trial(t, 10.0) is None


def test_outcome_time_none_baseline_returns_none():
    t = _trial("FA", {"FA": 3.5})
    assert _outcome_time_for_trial(t, None) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/analysis/test_tf_pulse_guards.py -v`
Expected: `test_outcome_time_lowercase_fa_is_now_caught`, `test_outcome_time_capitalised_abort_is_now_caught`, `test_outcome_time_ref_is_covered` FAIL (return `None`); the rest pass.

- [ ] **Step 3: Implement the case-robust guard**

Replace `_outcome_time_for_trial` (lines 92-103) with:

```python
def _outcome_time_for_trial(trial, baseline_t: Optional[float]) -> Optional[float]:
    """Absolute time of a baseline lick (fa/abort/ref), or None.

    Case-insensitive on both the outcome label and the reaction-time key,
    so lowercase ('fa') or capitalised ('Abort') data are handled uniformly.
    Hit response licks are post-change and not returned here (the change-onset
    guard already removes pulses near them).
    """
    out = getattr(trial, "trialoutcome", None)
    rts = getattr(trial, "reactiontimes", {}) or {}
    if baseline_t is None or out is None:
        return None
    out_l = str(out).lower()
    if out_l not in ("fa", "abort", "ref"):
        return None
    val = np.nan
    for k, v in rts.items():
        if str(k).lower() == out_l:
            val = v
            break
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return None
    if np.isfinite(fv):
        return float(baseline_t + fv)
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/analysis/test_tf_pulse_guards.py -v`
Expected: all 6 PASS.

- [ ] **Step 5: Add a `_collect_pulses` smoke test (constraints reduce pulse count)**

```python
# append to tests/analysis/test_tf_pulse_guards.py
from visdetect.utils.synthetic import make_synthetic_session


def test_collect_pulses_constraints_reduce_count():
    sess = make_synthetic_session(n_trials=30, n_clusters=2, seed=3)
    cfg_on = TFRespPulseConfig(use_constraints=True)
    cfg_off = TFRespPulseConfig(use_constraints=False)
    fast_on, slow_on = _collect_pulses(sess, cfg_on)
    fast_off, slow_off = _collect_pulses(sess, cfg_off)
    # Guards can only remove pulses, never add them.
    assert fast_on.size <= fast_off.size
    assert slow_on.size <= slow_off.size
```

- [ ] **Step 6: Run and commit**

Run: `python -m pytest tests/analysis/test_tf_pulse_guards.py -v`
Expected: all 7 PASS.

```bash
git add src/visdetect/analysis/tf_pulse.py tests/analysis/test_tf_pulse_guards.py
git commit -m "fix(tf): case-robust early-lick guard in _collect_pulses (fa/abort/ref)"
```

---

### Task 2: `estimate_drift` — slow per-unit firing-rate estimate

**Files:**
- Create: `src/visdetect/analysis/tf_drift.py`
- Test: `tests/analysis/test_tf_drift.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/analysis/test_tf_drift.py
import numpy as np
from visdetect.analysis.tf_drift import estimate_drift


def _ramp_spikes(t_end, r0, r1, seed):
    """Inhomogeneous Poisson via thinning: rate ramps linearly r0 -> r1."""
    rng = np.random.default_rng(seed)
    r_max = max(r0, r1)
    cand = np.sort(rng.uniform(0, t_end, size=int(r_max * t_end * 1.5)))
    lam = r0 + (r1 - r0) * (cand / t_end)
    keep = rng.random(cand.size) < (lam / r_max)
    return cand[keep]


def test_estimate_drift_recovers_ramp():
    t_end = 600.0
    spikes = _ramp_spikes(t_end, 2.0, 10.0, seed=0)
    grid_t, drift, mean_rate = estimate_drift(spikes, 0.0, t_end, bin_s=1.0, kernel_s=20.0)
    assert grid_t.shape == drift.shape
    early = drift[grid_t < 100].mean()
    late = drift[grid_t > 500].mean()
    assert late > early + 3.0            # rising drift recovered
    assert abs(mean_rate - spikes.size / t_end) < 1e-6


def test_estimate_drift_flat_is_flat():
    rng = np.random.default_rng(1)
    t_end = 600.0
    spikes = np.sort(rng.uniform(0, t_end, size=int(5.0 * t_end)))
    grid_t, drift, mean_rate = estimate_drift(spikes, 0.0, t_end, bin_s=1.0, kernel_s=20.0)
    assert abs(mean_rate - 5.0) < 0.5
    assert np.std(drift) < 1.0           # ~flat, no spurious trend
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/analysis/test_tf_drift.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'visdetect.analysis.tf_drift'`.

- [ ] **Step 3: Create `tf_drift.py` with `estimate_drift`**

```python
"""Source-level drift correction for TF pulse-triggered analysis.

See docs/superpowers/specs/2026-06-15-tf-responsiveness-labeler-design.md (§5).
Estimate each unit's slow firing drift over the whole session and subtract it
*before* pulse-triggered averaging, so the pre-pulse baseline (and its z-score
SD) is genuinely flat. Pure functions; reuses the KDE/z-score helpers in
tf_pulse.py rather than reimplementing them.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d


def estimate_drift(
    spike_times: np.ndarray,
    t_start: float,
    t_end: float,
    bin_s: float = 0.5,
    kernel_s: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Slow firing-rate estimate (Hz) over [t_start, t_end).

    Bins spikes at ``bin_s`` and smooths with a Gaussian of width
    ``kernel_s`` (seconds). Wide ``kernel_s`` captures session-scale drift;
    narrower also removes faster within-window structure — Phase 0 tunes it.

    Returns
    -------
    grid_t : (n_bins,) bin-centre times (s)
    drift  : (n_bins,) smoothed rate (Hz)
    mean_rate : scalar mean rate (Hz) over the window
    """
    spike_times = np.asarray(spike_times, dtype=float)
    spike_times = spike_times[(spike_times >= t_start) & (spike_times < t_end)]
    dur = max(t_end - t_start, 1e-9)
    n_bins = max(int(np.ceil(dur / bin_s)), 1)
    edges = t_start + np.arange(n_bins + 1) * bin_s
    counts, _ = np.histogram(spike_times, bins=edges)
    rate = counts.astype(float) / bin_s
    sigma_bins = max(kernel_s / bin_s, 1e-6)
    drift = gaussian_filter1d(rate, sigma=sigma_bins, mode="nearest")
    grid_t = 0.5 * (edges[:-1] + edges[1:])
    mean_rate = float(spike_times.size / dur)
    return grid_t, drift, mean_rate
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/analysis/test_tf_drift.py -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/tf_drift.py tests/analysis/test_tf_drift.py
git commit -m "feat(tf): tf_drift.estimate_drift — slow per-unit firing-rate estimate"
```

---

### Task 3: `detrended_pulse_average` — drift-corrected pulse-triggered average (Hz)

**Files:**
- Modify: `src/visdetect/analysis/tf_drift.py`
- Test: `tests/analysis/test_tf_drift.py`

Reuse `_mean_activity_per_unit` for the fine pulse-triggered KDE (note: it returns smoothed **spikes-per-`dt`-bin**, so divide by `dt` for Hz), then subtract the pulse-triggered average of the slow drift and restore the mean. `detrended = PTA(fine)/dt − PTA(drift) + mean_rate`, all in Hz.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/analysis/test_tf_drift.py
from visdetect.analysis.tf_drift import detrended_pulse_average

PRE = (-0.4, 0.0)
POST = (0.0, 0.5)


def _flat_spikes(t_end, rate, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0, t_end, size=rng.poisson(rate * t_end)))


def _flat_plus_bump(t_end, rate, pulses, bump_hz, bump_dur, seed):
    rng = np.random.default_rng(seed)
    base = np.sort(rng.uniform(0, t_end, size=rng.poisson(rate * t_end)))
    parts = [base]
    for p in pulses:
        n = rng.poisson(bump_hz * bump_dur)
        if n:
            parts.append(rng.uniform(p, p + bump_dur, size=n))
    return np.sort(np.concatenate(parts))


def test_detrended_baseline_is_in_hz_near_mean_rate():
    t_end = 400.0
    spikes = _flat_spikes(t_end, 5.0, seed=2)
    pulses = np.arange(10.0, t_end - 10.0, 2.0)
    gt, dr, mr = estimate_drift(spikes, 0.0, t_end, bin_s=1.0, kernel_s=20.0)
    det, sem, t_vec = detrended_pulse_average(
        spikes, pulses, PRE, POST, 0.005, 20.0, gt, dr, mr)
    pre = det[t_vec < 0.0]
    assert abs(pre.mean() - 5.0) < 1.5            # baseline ~ true rate in Hz
    assert abs(np.polyfit(t_vec[t_vec < 0.0], pre, 1)[0]) < 5.0  # ~flat


def test_detrended_preserves_pulse_response():
    t_end = 400.0
    pulses = np.arange(10.0, t_end - 10.0, 2.0)
    spikes = _flat_plus_bump(t_end, 5.0, pulses, bump_hz=30.0, bump_dur=0.1, seed=4)
    gt, dr, mr = estimate_drift(spikes, 0.0, t_end, bin_s=1.0, kernel_s=20.0)
    det, sem, t_vec = detrended_pulse_average(
        spikes, pulses, PRE, POST, 0.005, 20.0, gt, dr, mr)
    baseline = det[t_vec < 0.0].mean()
    peak = det[(t_vec >= 0.0) & (t_vec < 0.15)].max()
    assert peak > baseline + 8.0                  # injected bump survives detrend


def test_detrended_empty_pulses_returns_empty():
    spikes = _flat_spikes(100.0, 5.0, seed=5)
    det, sem, t_vec = detrended_pulse_average(
        spikes, np.array([]), PRE, POST, 0.005, 20.0,
        *estimate_drift(spikes, 0.0, 100.0))
    assert det.size == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/analysis/test_tf_drift.py -k detrended -v`
Expected: FAIL with `ImportError: cannot import name 'detrended_pulse_average'`.

- [ ] **Step 3: Implement `detrended_pulse_average`**

```python
# add to src/visdetect/analysis/tf_drift.py
def detrended_pulse_average(
    spike_times: np.ndarray,
    pulse_times: np.ndarray,
    pre_window: Tuple[float, float],
    post_window: Tuple[float, float],
    dt: float,
    sigma_ms: float,
    drift_grid_t: np.ndarray,
    drift_rate: np.ndarray,
    mean_rate: float,
    trace_start=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drift-corrected pulse-triggered average in Hz.

    detrended(t) = PTA(fine_rate)/dt - PTA(slow_drift) + mean_rate
    """
    from visdetect.analysis.tf_pulse import _mean_activity_per_unit

    mean_fine, sem, t_vec = _mean_activity_per_unit(
        spike_times, pulse_times, pre_window, post_window, dt, sigma_ms,
        trace_start=trace_start)
    if mean_fine.size == 0:
        return mean_fine, sem, t_vec

    mean_fine_hz = mean_fine / dt
    sem_hz = sem / dt

    pulses = np.asarray(pulse_times, dtype=float)
    pulses = pulses[np.isfinite(pulses)]
    drift_grid_t = np.asarray(drift_grid_t, dtype=float)
    drift_rate = np.asarray(drift_rate, dtype=float)

    drift_pta = np.zeros_like(t_vec)
    for tp in pulses:
        drift_pta += np.interp(
            tp + t_vec, drift_grid_t, drift_rate,
            left=drift_rate[0], right=drift_rate[-1])
    drift_pta /= max(pulses.size, 1)

    detrended = mean_fine_hz - drift_pta + float(mean_rate)
    return detrended, sem_hz, t_vec
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/analysis/test_tf_drift.py -k detrended -v`
Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/tf_drift.py tests/analysis/test_tf_drift.py
git commit -m "feat(tf): detrended_pulse_average — source-level drift-corrected PTA in Hz"
```

---

### Task 4: `prepulse_slope` — the flat-baseline QC metric

**Files:**
- Modify: `src/visdetect/analysis/tf_drift.py`
- Test: `tests/analysis/test_tf_drift.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/analysis/test_tf_drift.py
from visdetect.analysis.tf_drift import prepulse_slope


def test_prepulse_slope_of_flat_is_zero():
    t_vec = np.linspace(-0.4, 0.5, 180)
    trace = np.full_like(t_vec, 7.0)
    assert abs(prepulse_slope(trace, t_vec, PRE)) < 1e-6


def test_prepulse_slope_of_ramp_matches_slope():
    t_vec = np.linspace(-0.4, 0.5, 180)
    trace = 3.0 * t_vec + 2.0           # slope 3.0 over the pre-window
    assert abs(prepulse_slope(trace, t_vec, PRE) - 3.0) < 1e-6


def test_prepulse_slope_too_few_bins_is_nan():
    t_vec = np.array([0.1, 0.2])        # nothing in the pre-window
    assert np.isnan(prepulse_slope(np.array([1.0, 2.0]), t_vec, PRE))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/analysis/test_tf_drift.py -k prepulse -v`
Expected: FAIL with `ImportError: cannot import name 'prepulse_slope'`.

- [ ] **Step 3: Implement `prepulse_slope`**

```python
# add to src/visdetect/analysis/tf_drift.py
def prepulse_slope(trace, t_vec, pre_window: Tuple[float, float]) -> float:
    """Linear slope (units/s) of ``trace`` within the pre-pulse window.

    The Phase-0 success metric: after detrend, the population distribution of
    this should collapse toward 0. NaN if < 2 samples fall in the window.
    """
    t_vec = np.asarray(t_vec, dtype=float)
    trace = np.asarray(trace, dtype=float)
    mask = (t_vec >= pre_window[0]) & (t_vec < pre_window[1])
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(t_vec[mask], trace[mask], 1)[0])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/analysis/test_tf_drift.py -k prepulse -v`
Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/tf_drift.py tests/analysis/test_tf_drift.py
git commit -m "feat(tf): prepulse_slope — flat-baseline QC metric for the detrend"
```

---

### Task 5: `circular_shift_null` — independent shuffle null bank

**Files:**
- Modify: `src/visdetect/analysis/tf_drift.py`
- Test: `tests/analysis/test_tf_drift.py`

Recompute the detrended, z-scored pulse trace under random circular time-shifts of the spike train. Provides the GUI's null envelope and an independent significance check that does not depend on the drift model being correct.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/analysis/test_tf_drift.py
from visdetect.analysis.tf_drift import circular_shift_null


def test_null_envelope_flags_real_bump():
    t_end = 200.0
    pulses = np.arange(10.0, t_end - 10.0, 3.0)
    spikes = _flat_plus_bump(t_end, 5.0, pulses, bump_hz=30.0, bump_dur=0.1, seed=7)
    gt, dr, mr = estimate_drift(spikes, 0.0, t_end, bin_s=1.0, kernel_s=20.0)
    det, _, t_vec = detrended_pulse_average(
        spikes, pulses, PRE, POST, 0.005, 20.0, gt, dr, mr)
    from visdetect.analysis.tf_pulse import _zscore_trace
    obs_z = _zscore_trace(det, t_vec, PRE)

    null_z, t_null = circular_shift_null(
        spikes, pulses, PRE, POST, 0.005, 20.0, bin_s=1.0, kernel_s=20.0,
        session_dur=t_end, n_shuffles=30, seed=0)
    hi = np.percentile(null_z, 95, axis=0)
    post = (t_vec >= 0.0) & (t_vec < 0.15)
    assert obs_z[post].max() > hi[post].max()      # real response exits the null


def test_null_envelope_contains_flat_unit():
    t_end = 200.0
    pulses = np.arange(10.0, t_end - 10.0, 3.0)
    spikes = _flat_spikes(t_end, 5.0, seed=8)
    gt, dr, mr = estimate_drift(spikes, 0.0, t_end, bin_s=1.0, kernel_s=20.0)
    det, _, t_vec = detrended_pulse_average(
        spikes, pulses, PRE, POST, 0.005, 20.0, gt, dr, mr)
    from visdetect.analysis.tf_pulse import _zscore_trace
    obs_z = _zscore_trace(det, t_vec, PRE)
    null_z, _ = circular_shift_null(
        spikes, pulses, PRE, POST, 0.005, 20.0, bin_s=1.0, kernel_s=20.0,
        session_dur=t_end, n_shuffles=30, seed=0)
    lo = np.percentile(null_z, 2.5, axis=0)
    hi = np.percentile(null_z, 97.5, axis=0)
    post = (t_vec >= 0.0) & (t_vec < 0.5)
    frac_inside = np.mean((obs_z[post] >= lo[post]) & (obs_z[post] <= hi[post]))
    assert frac_inside > 0.8                       # flat unit mostly within null
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/analysis/test_tf_drift.py -k null -v`
Expected: FAIL with `ImportError: cannot import name 'circular_shift_null'`.

- [ ] **Step 3: Implement `circular_shift_null`**

```python
# add to src/visdetect/analysis/tf_drift.py
def circular_shift_null(
    spike_times: np.ndarray,
    pulse_times: np.ndarray,
    pre_window: Tuple[float, float],
    post_window: Tuple[float, float],
    dt: float,
    sigma_ms: float,
    bin_s: float,
    kernel_s: float,
    session_dur: float,
    n_shuffles: int = 200,
    seed: int = 0,
    trace_start=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Null bank of z-scored detrended pulse traces under circular shifts.

    Returns (null_z, t_vec): null_z is (n_shuffles, n_time).
    """
    from visdetect.analysis.tf_pulse import _zscore_trace

    rng = np.random.default_rng(seed)
    spike_times = np.sort(np.asarray(spike_times, dtype=float))
    min_shift = max(30.0, session_dur * 0.05)
    hi = session_dur - min_shift
    if hi <= min_shift:
        hi = session_dur * 0.95
    rows = []
    t_vec = None
    for _ in range(int(n_shuffles)):
        shift = rng.uniform(min_shift, hi)
        shifted = np.sort((spike_times + shift) % session_dur)
        gt, dr, mr = estimate_drift(shifted, 0.0, session_dur, bin_s, kernel_s)
        det, _, t_vec = detrended_pulse_average(
            shifted, pulse_times, pre_window, post_window, dt, sigma_ms,
            gt, dr, mr, trace_start=trace_start)
        rows.append(_zscore_trace(det, t_vec, pre_window))
    return np.asarray(rows), t_vec
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/analysis/test_tf_drift.py -k null -v`
Expected: both PASS (each ~1-3 s).

- [ ] **Step 5: Run the full drift test module and commit**

Run: `python -m pytest tests/analysis/test_tf_drift.py -v`
Expected: all tests PASS.

```bash
git add src/visdetect/analysis/tf_drift.py tests/analysis/test_tf_drift.py
git commit -m "feat(tf): circular_shift_null — independent shuffle null bank for detrended PTA"
```

---

### Task 6: Phase-0 validation script (the eyeball gate)

**Files:**
- Create: `scripts/tf_responsiveness/validate_drift_phase0.py`
- Test: `tests/analysis/test_tf_drift.py` (smoke test on a synthetic session)

Run the core on a handful of `session:cluster` exemplars (you pick ~5 obvious responders + ~5 obvious drift-only after a first look), render raw-vs-detrended fast/slow traces with the null envelope, and print/save a pre-pulse-slope-before/after table. This is the gate the experimenter signs off before any full-session scaling (Plan 2).

- [ ] **Step 1: Write the smoke test**

```python
# append to tests/analysis/test_tf_drift.py
def test_phase0_run_on_session_smoke(tmp_path):
    from visdetect.utils.synthetic import make_synthetic_session
    import sys, importlib.util, pathlib
    mod_path = pathlib.Path(__file__).resolve().parents[2] / \
        "scripts" / "tf_responsiveness" / "validate_drift_phase0.py"
    spec = importlib.util.spec_from_file_location("validate_drift_phase0", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    sess = make_synthetic_session(n_trials=60, n_clusters=4, seed=11)
    rows = mod.run_units(
        sess, cluster_ids=[0, 1], kernel_s=20.0, n_shuffles=10,
        out_png=str(tmp_path / "phase0.png"))
    assert (tmp_path / "phase0.png").exists()
    assert len(rows) == 2
    for r in rows:
        assert "slope_raw" in r and "slope_detrended" in r
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run: `python -m pytest tests/analysis/test_tf_drift.py -k phase0 -v`
Expected: FAIL (file/module not found).

- [ ] **Step 3: Implement the Phase-0 script**

```python
"""Phase 0: validate the drift correction on a few exemplar units.

Eyeball gate before full-session scaling (Plan 2). Renders raw vs detrended
fast/slow pulse traces with the circular-shift null envelope, and reports the
pre-pulse slope before/after per unit.

Usage:
  py scripts/tf_responsiveness/validate_drift_phase0.py \
     --session 7072025 --clusters 42 108 211 --kernel-s 20 --out phase0.png
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visdetect.analysis.constants import TF_PULSE_PRE_WINDOW, TF_PULSE_POST_WINDOW
from visdetect.analysis.tf_pulse import (
    _collect_pulses, _zscore_trace, TFRespPulseConfig,
)
from visdetect.analysis.tf_drift import (
    estimate_drift, detrended_pulse_average, prepulse_slope, circular_shift_null,
)

PRE = TF_PULSE_PRE_WINDOW
POST = TF_PULSE_POST_WINDOW
DT = 0.005
SIGMA_MS = 20.0
BIN_S = 1.0


def _raw_pulse_average_hz(spike_times, pulses):
    from visdetect.analysis.tf_pulse import _mean_activity_per_unit
    mean_fine, sem, t_vec = _mean_activity_per_unit(
        spike_times, pulses, PRE, POST, DT, SIGMA_MS)
    if mean_fine.size == 0:
        return mean_fine, t_vec
    return mean_fine / DT, t_vec


def run_units(session, cluster_ids, kernel_s=20.0, n_shuffles=100, out_png="phase0.png"):
    """Validate drift correction for the given clusters; returns a list of dict rows."""
    cfg = TFRespPulseConfig(use_constraints=True)
    fast_pulses, slow_pulses = _collect_pulses(session, cfg)
    spikes_by_cid = {int(c.cluster_id): np.asarray(c.spike_times, float).ravel()
                     for c in session.clusters}

    rows = []
    n = len(cluster_ids)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.6 * max(n, 1)), squeeze=False)
    for r, cid in enumerate(cluster_ids):
        st = np.sort(spikes_by_cid.get(int(cid), np.array([])))
        sess_dur = float(st.max()) if st.size else 1.0
        gt, dr, mr = estimate_drift(st, 0.0, sess_dur, bin_s=BIN_S, kernel_s=kernel_s)
        for col, (pulses, label) in enumerate(
                [(fast_pulses, "fast ▲"), (slow_pulses, "slow ▼")]):
            ax = axes[r][col]
            raw_hz, t_vec = _raw_pulse_average_hz(st, pulses)
            det_hz, _, t_det = detrended_pulse_average(
                st, pulses, PRE, POST, DT, SIGMA_MS, gt, dr, mr)
            if raw_hz.size == 0 or det_hz.size == 0:
                ax.text(0.5, 0.5, "no pulses", ha="center", transform=ax.transAxes)
                continue
            null_z, t_null = circular_shift_null(
                st, pulses, PRE, POST, DT, SIGMA_MS, BIN_S, kernel_s,
                session_dur=sess_dur, n_shuffles=n_shuffles, seed=0)
            det_z = _zscore_trace(det_hz, t_det, PRE)
            lo = np.percentile(null_z, 5, axis=0)
            hi = np.percentile(null_z, 95, axis=0)
            ax.fill_between(t_null, lo, hi, color="0.75", alpha=0.5, lw=0,
                            label="null 5-95%")
            ax.plot(t_det, _zscore_trace(raw_hz, t_vec, PRE), "k--", lw=1.0,
                    label="raw")
            ax.plot(t_det, det_z, "k-", lw=1.6, label="detrended")
            ax.axvline(0, color="0.5", lw=0.7, ls=":")
            ax.axhline(0, color="0.6", lw=0.4)
            ax.set_title(f"clu{cid} {label}", fontsize=9)
            if r == 0 and col == 0:
                ax.legend(fontsize=6, loc="upper left")
            rows.append({
                "cluster_id": int(cid), "direction": label.split()[0],
                "slope_raw": prepulse_slope(_zscore_trace(raw_hz, t_vec, PRE), t_vec, PRE),
                "slope_detrended": prepulse_slope(det_z, t_det, PRE),
            })
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return rows


def main():
    from visdetect.suite.loader import load_session
    ap = argparse.ArgumentParser(description="Phase 0 drift-correction validation")
    ap.add_argument("--session", required=True)
    ap.add_argument("--clusters", type=int, nargs="+", required=True)
    ap.add_argument("--kernel-s", type=float, default=20.0)
    ap.add_argument("--n-shuffles", type=int, default=100)
    ap.add_argument("--out", default="figures/tf_responsiveness/phase0_drift.png")
    args = ap.parse_args()

    sess = load_session(args.session)
    rows = run_units(sess, args.clusters, kernel_s=args.kernel_s,
                     n_shuffles=args.n_shuffles, out_png=args.out)
    print(f"\n  {'cluster':>8s} {'dir':>5s} {'slope_raw':>12s} {'slope_detr':>12s}")
    for r in rows:
        print(f"  {r['cluster_id']:8d} {r['direction']:>5s} "
              f"{r['slope_raw']:12.3f} {r['slope_detrended']:12.3f}")
    print(f"\n  Figure: {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `python -m pytest tests/analysis/test_tf_drift.py -k phase0 -v`
Expected: PASS; a `phase0.png` is written to the tmp path and 2 rows returned.

- [ ] **Step 5: Run the whole new test surface and commit**

Run: `python -m pytest tests/analysis/test_tf_drift.py tests/analysis/test_tf_pulse_guards.py -v`
Expected: all PASS.

```bash
git add scripts/tf_responsiveness/validate_drift_phase0.py tests/analysis/test_tf_drift.py
git commit -m "feat(tf): Phase-0 drift-correction validation script + smoke test"
```

---

## Phase-0 Gate (manual, after this plan)

Run on real exemplars and **eyeball the result before Plan 2**:

```bash
py scripts/tf_responsiveness/validate_drift_phase0.py \
   --session <SID> --clusters <responder ids> <drift-only ids> --kernel-s 20 \
   --out figures/tf_responsiveness/phase0_drift.png
```

Gate criteria: drift-only units' pre-pulse slope collapses toward 0 (`slope_detrended` ≪ `slope_raw`); responders keep a clear post-pulse deflection that exits the null envelope. If drift-only slopes don't flatten, narrow `--kernel-s` (toward a within-window detrend) and re-run; if responders get eaten, widen it. Record the chosen `kernel_s` — it becomes the default for Plan 2's full extraction.

---

## Self-Review

- **Spec coverage:** §5 step 1 (estimate_drift, kernel spectrum) → Task 2; §5 steps 2-3 (subtract-at-source + re-average in Hz) → Task 3; §5 step 6 (null bank) → Task 5; §5 "Pulse selection" guard fix → Task 1; §5 success test (pre-pulse slope) → Task 4 + Phase-0 gate; Phase 0 (exemplar eyeball) → Task 6. State conditioning, full-session NPZ caches, and population QC are intentionally **out of scope** (Plan 2).
- **Placeholder scan:** none — every step has runnable code/commands and expected output.
- **Type consistency:** `estimate_drift → (grid_t, drift, mean_rate)` consumed unchanged by `detrended_pulse_average` and `circular_shift_null`; `detrended_pulse_average → (detrended, sem, t_vec)` consumed by `prepulse_slope`/`_zscore_trace`; `_mean_activity_per_unit → (mean, sem, t_vec)` per `tf_pulse.py:215`; `_zscore_trace(mean_trace, t_vec, pre_window)` per `tf_pulse.py:218`. Units: PTA converted Hz via `/dt`, validated by `test_detrended_baseline_is_in_hz_near_mean_rate`.
