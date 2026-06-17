# TF-Responsiveness Selectivity — Phase B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the fast-minus-slow selectivity detection core (`tf_selectivity.py`, TDD), land the two minor `tf_pulse.py` correctness fixes, retire the dead drift module, and run the early-validation gate on a real session to confirm a real sparse responder population exists and re-pick clean exemplars.

**Architecture:** A new library module `tf_selectivity.py` reuses the *fixed* pulse collection from `tf_pulse.py`. Per unit it builds per-pulse smoothed Hz matrices for fast and slow pulses over `[-1.0, +0.5]` s, computes one **shared per-unit baseline** (pooled pre-window of both mean traces — fixes the old circular separate-baseline bug), forms `selectivity(t) = (fast_hz − slow_hz)/σ_baseline`, measures a signed post-window peak/latency/AUC/half-width, and tests significance with a **label-shuffle null** (permute fast/slow labels, counts fixed). A standalone gate script runs this on real units and reports whether a sparse population exits the null at short latency.

**Tech Stack:** Python 3, NumPy, SciPy (`gaussian_filter1d`), pandas, matplotlib (Agg); pytest (TDD). Worktree `E:/python_analysis/git_repos/vd_tf_phase0`, branch `feature/tf-responsiveness-labeler`. Spec: `docs/superpowers/specs/2026-06-17-tf-responsiveness-selectivity-design.md`.

---

## Execution environment (read once before starting)

- **All commands run from the worktree root:** `cd /e/python_analysis/git_repos/vd_tf_phase0`.
- **Python launcher is `py`** (Windows + Git Bash), never `python`.
- **The editable `visdetect` install is pinned to the PRIMARY repo's `src/`, not this worktree.** `conftest.py` already prepends this worktree's `src/` for pytest, so `py -m pytest` tests the worktree code. For scripts, always run with `PYTHONPATH=src` AND the script also self-inserts its `src/` (belt and suspenders).
- **Do not switch the primary repo's branch** — a parallel chat owns it. Work only in this worktree.
- Run the **full TF test set** after each task: `py -m pytest tests/analysis/test_tf_pulse_guards.py tests/analysis/test_tf_pulse_alignment.py tests/analysis/test_tf_selectivity.py -q`

## File structure

- **Modify** `src/visdetect/analysis/tf_pulse.py` — two correctness fixes (`_smooth_binned_activity` spike undercount; `_collect_pulses` unbounded-baseline leakage). No API changes.
- **Create** `src/visdetect/analysis/tf_selectivity.py` — the detection core (config, dataclass, per-pulse matrix, shared baseline, selectivity metrics, label-shuffle null, split-half, feature row, session driver).
- **Create** `tests/analysis/test_tf_selectivity.py` — TDD tests for the core.
- **Modify** `tests/analysis/test_tf_pulse_guards.py` — add the two fix regression tests.
- **Create** `scripts/tf_responsiveness/validate_selectivity_phase0.py` — the early-validation gate (real-session run + figure + exemplar candidates) with a unit-testable `build_feature_table` seam.
- **Create** `tests/scripts/test_validate_selectivity_phase0.py` — smoke test for the gate's `build_feature_table`.
- **Delete** `src/visdetect/analysis/tf_drift.py` and `tests/analysis/test_tf_drift.py` — the retired drift approach (the spec keeps `scripts/tf_responsiveness/validate_drift_phase0.py` as the record of the pivot, so leave that file).

---

## Task 1: Fix `_smooth_binned_activity` spike undercount

The smoother builds a binary 0/1 train, so ≥2 spikes landing in the same `dt` bin count as one. Switch to `np.add.at` so the bin holds the true count. `gaussian_filter1d` conserves the sum, so a correctly-counted train's smoothed output sums to the spike count.

**Files:**
- Modify: `src/visdetect/analysis/tf_pulse.py:185-194` (`_smooth_binned_activity`)
- Test: `tests/analysis/test_tf_pulse_guards.py`

- [ ] **Step 1: Write the failing test**

Add to the end of `tests/analysis/test_tf_pulse_guards.py`:

```python
import numpy as np
from visdetect.analysis.tf_pulse import _smooth_binned_activity


def test_smooth_counts_multiple_spikes_per_bin():
    # Two spikes inside the SAME 1 ms bin, far from the trace edges so the
    # Gaussian kernel sits fully inside (sum is conserved).
    t_vec = np.arange(-0.5, 0.5, 0.001)
    rel = np.array([0.0005, 0.00051])  # both fall in the bin at t=0
    out = _smooth_binned_activity(rel, t_vec, sigma_bins=17.0)
    # gaussian_filter1d preserves the integral -> two spikes must sum to ~2,
    # not ~1 as the old binary 0/1 train produced.
    assert np.isclose(out.sum(), 2.0, atol=1e-6), out.sum()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_tf_pulse_guards.py::test_smooth_counts_multiple_spikes_per_bin -q`
Expected: FAIL — `out.sum()` ≈ 1.0 (binary train collapses the two spikes).

- [ ] **Step 3: Apply the fix**

In `src/visdetect/analysis/tf_pulse.py`, replace the body of `_smooth_binned_activity`:

```python
def _smooth_binned_activity(spike_times_rel: np.ndarray, t_vec: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Bin spikes onto t_vec grid and smooth with Gaussian (legacy-compatible).

    Uses np.add.at so multiple spikes in the same dt-bin accumulate (a binary
    0/1 train would undercount >=2 spikes/bin).
    """
    if spike_times_rel.size == 0:
        return np.zeros_like(t_vec)
    train = np.zeros_like(t_vec)
    idx = np.searchsorted(t_vec, spike_times_rel)
    idx = idx[(idx >= 0) & (idx < train.size)]
    np.add.at(train, idx, 1.0)
    return gaussian_filter1d(train, sigma=sigma_bins)
```

- [ ] **Step 4: Run the TF guard + alignment tests to verify pass and no regression**

Run: `py -m pytest tests/analysis/test_tf_pulse_guards.py tests/analysis/test_tf_pulse_alignment.py -q`
Expected: PASS (all green).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/tf_pulse.py tests/analysis/test_tf_pulse_guards.py
git commit -m "$(cat <<'EOF'
fix(tf): count multiple spikes per bin in _smooth_binned_activity

Binary 0/1 train undercounted >=2 spikes landing in the same dt bin.
Use np.add.at so the smoothed rate reflects the true spike count.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Fix unbounded-baseline leakage in `_collect_pulses`

When a trial has `n_seen=None` (vector not truncated), no Change_ON (NaN → `t_change=None`, e.g. abort/fa), AND no recoverable baseline-lick time (`t_outcome=None`), there is **no bound** on the baseline window, so pulses are scanned across the entire `baseline_values` vector — which can run past the real end of the trial. Skip such unbounded trials. Trials with a known `n_seen` (already truncated), a known `t_change`, or a known `t_outcome` stay bounded and are kept.

**Files:**
- Modify: `src/visdetect/analysis/tf_pulse.py:138-182` (`_collect_pulses` loop)
- Test: `tests/analysis/test_tf_pulse_guards.py`

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/analysis/test_tf_pulse_guards.py`:

```python
from visdetect.core.session import Session


def test_unbounded_trial_skipped_when_no_bounds():
    # n_seen=None, abort with NaN Change_ON and no usable lick time -> cannot
    # bound the baseline window -> the whole trial must be skipped.
    bv = np.ones(3 * 400)
    bv[3 * 100] = 2.0  # a fast sample at post-stride idx 100 -> +5.0 s
    t = Trial(trialoutcome="abort", reactiontimes={}, change_size=1.0,
              change_time=None, baseline_values=bv, n_seen=None)
    ni = {"Baseline_ON": np.array([0.0]), "Change_ON": np.array([np.nan])}
    sess = Session(trials=[t], clusters=[], ni_events=ni, session_name="X")
    fast, slow = _collect_pulses(sess, TFRespPulseConfig(use_constraints=True))
    assert fast.size == 0 and slow.size == 0


def test_outcome_bounded_trial_keeps_early_pulses():
    # fa lick at 5.0 s bounds the window (guard 2 s -> cutoff 3.0 s):
    # the +2.0 s pulse is kept; the +10.0 s pulse is dropped; trial NOT skipped.
    bv = np.ones(3 * 400)
    bv[3 * 40] = 2.0   # fast at +2.0 s
    bv[3 * 200] = 2.0  # fast at +10.0 s
    t = Trial(trialoutcome="fa", reactiontimes={"fa": 5.0}, change_size=1.0,
              change_time=None, baseline_values=bv, n_seen=None)
    ni = {"Baseline_ON": np.array([0.0]), "Change_ON": np.array([np.nan])}
    sess = Session(trials=[t], clusters=[], ni_events=ni, session_name="X")
    fast, slow = _collect_pulses(sess, TFRespPulseConfig(use_constraints=True))
    assert np.any(np.isclose(fast, 2.0, atol=0.051)), np.sort(fast)
    assert not np.any(np.isclose(fast, 10.0, atol=0.051)), np.sort(fast)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -m pytest tests/analysis/test_tf_pulse_guards.py::test_unbounded_trial_skipped_when_no_bounds tests/analysis/test_tf_pulse_guards.py::test_outcome_bounded_trial_keeps_early_pulses -q`
Expected: `test_unbounded_trial_skipped_when_no_bounds` FAILS (the +5.0 s pulse is currently collected, so `fast.size == 1`). The second test should already pass (it documents the bound that must keep working).

- [ ] **Step 3: Apply the fix**

In `src/visdetect/analysis/tf_pulse.py`, inside `_collect_pulses`, replace the `n_seen` block and add the skip guard. The current block is:

```python
        n_seen = getattr(t, "n_seen", None)
        if isinstance(n_seen, (int, np.integer)) and n_seen is not None and n_seen > 0:
            arr = arr[: int(n_seen)]
        # Compute log2 TF and identify fast/slow bins
        log2_tf = _safe_log2(arr)
        # Absolute reference time for trial
        t0 = float(base_by_trial[i]) if i < len(base_by_trial) and np.isfinite(base_by_trial[i]) else None
        t_change = float(change_by_trial[i]) if i < len(change_by_trial) and np.isfinite(change_by_trial[i]) else None
        t_outcome = _outcome_time_for_trial(t, t0)
```

Replace it with:

```python
        n_seen = getattr(t, "n_seen", None)
        has_n_seen = isinstance(n_seen, (int, np.integer)) and n_seen is not None and n_seen > 0
        if has_n_seen:
            arr = arr[: int(n_seen)]
        # Compute log2 TF and identify fast/slow bins
        log2_tf = _safe_log2(arr)
        # Absolute reference time for trial
        t0 = float(base_by_trial[i]) if i < len(base_by_trial) and np.isfinite(base_by_trial[i]) else None
        t_change = float(change_by_trial[i]) if i < len(change_by_trial) and np.isfinite(change_by_trial[i]) else None
        t_outcome = _outcome_time_for_trial(t, t0)
        # Leakage guard: with constraints on, if nothing bounds the baseline
        # window (no n_seen truncation, no change time, no outcome lick time),
        # the full baseline_values vector can run past the real end of the
        # trial. Skip such unbounded trials rather than scan the whole vector.
        if cfg.use_constraints and not has_n_seen and t_change is None and t_outcome is None:
            continue
```

- [ ] **Step 4: Run the TF guard + alignment tests to verify pass and no regression**

Run: `py -m pytest tests/analysis/test_tf_pulse_guards.py tests/analysis/test_tf_pulse_alignment.py -q`
Expected: PASS (all green, including `test_collect_pulses_constraints_reduce_count`).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/tf_pulse.py tests/analysis/test_tf_pulse_guards.py
git commit -m "$(cat <<'EOF'
fix(tf): skip unbounded baseline trials in _collect_pulses

When n_seen is unknown and neither Change_ON nor a baseline-lick time
bounds the window, the full baseline_values vector can extend past the
real trial end. Skip these trials under use_constraints to stop leakage.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `tf_selectivity.py` — per-pulse Hz rate matrix

Create the new module with its config, and the per-pulse rate-matrix helper. Each row is one pulse's Gaussian-smoothed firing rate in **Hz** (`_smooth_binned_activity` returns counts-per-`dt`-bin; divide by `dt`). Reuse the now-fixed `_smooth_binned_activity` so there is one smoothing implementation.

**Files:**
- Create: `src/visdetect/analysis/tf_selectivity.py`
- Test: `tests/analysis/test_tf_selectivity.py`

- [ ] **Step 1: Write the failing test**

Create `tests/analysis/test_tf_selectivity.py`:

```python
"""Tests for the fast-minus-slow selectivity detection core."""
import numpy as np

from visdetect.core.session import Session, Trial, Cluster
from visdetect.analysis.tf_selectivity import (
    TFSelectivityConfig,
    _time_vector,
    _per_pulse_rate_matrix,
)


def test_per_pulse_rate_matrix_recovers_hz():
    # A unit firing at a regular 100 Hz over the whole window; the per-pulse
    # mean rate in a flat interior region should be ~100 Hz.
    cfg = TFSelectivityConfig()
    t_vec = _time_vector(cfg)
    rate = 100.0
    spikes = np.arange(0.0, 1000.0, 1.0 / rate)
    pulses = np.array([100.0, 200.0, 300.0, 400.0])
    mat = _per_pulse_rate_matrix(spikes, pulses, t_vec, cfg.pulse.dt, cfg.pulse.sigma_ms)
    assert mat.shape == (4, t_vec.size)
    interior = (t_vec >= -0.5) & (t_vec < -0.1)
    mean_hz = np.nanmean(mat[:, interior])
    assert np.isclose(mean_hz, rate, rtol=0.05), mean_hz


def test_per_pulse_rate_matrix_empty_pulses():
    cfg = TFSelectivityConfig()
    t_vec = _time_vector(cfg)
    mat = _per_pulse_rate_matrix(np.arange(0, 10, 0.01), np.array([]), t_vec,
                                 cfg.pulse.dt, cfg.pulse.sigma_ms)
    assert mat.shape == (0, t_vec.size)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_tf_selectivity.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'visdetect.analysis.tf_selectivity'`.

- [ ] **Step 3: Create the module with config + matrix helper**

Create `src/visdetect/analysis/tf_selectivity.py`:

```python
"""TF-pulse fast-minus-slow selectivity (Lohse 2025) for responder ID.

Replaces the retired source-level drift-detrend approach (tf_drift.py): the
pre-pulse firing-rate ramp is a within-trial temporal-expectation signal at the
same timescale as the response, so it cannot be modelled out. The fast-minus-
slow difference cancels that common-mode ramp by symmetry (the ramp is trial-
locked, not pulse-identity-locked; fast and slow pulses sample it identically),
with no detrend and no model.

Pipeline (per unit; all-trials in Phase B, per-state later):
  corrected pulses (fixed _collect_pulses)
    -> per-pulse smoothed Hz matrices (fast, slow) over [trace_pre, +0.5] s
    -> shared per-unit baseline (mu_b, sigma_b) pooled over the pre-window of
       BOTH mean traces  (fixes the old per-condition separate-baseline bug)
    -> selectivity(t) = (fast_hz - slow_hz) / max(sigma_b, eps)
    -> signed post-window peak / latency / AUC / half-width
    -> label-shuffle null (permute fast/slow labels, counts fixed) -> shuffle p
    -> within-unit split-half reliability of the selectivity
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from visdetect.analysis.constants import TF_PULSE_TRACE_PRE
from visdetect.analysis.tf_pulse import (
    TFRespPulseConfig,
    _collect_pulses,
    _smooth_binned_activity,
)


@dataclass
class TFSelectivityConfig:
    """Selectivity config. Wraps a TFRespPulseConfig (trace extended to -1.0 s)
    and adds the null/sufficiency knobs."""
    pulse: TFRespPulseConfig = field(
        default_factory=lambda: TFRespPulseConfig(trace_pre=TF_PULSE_TRACE_PRE)
    )
    n_shuffles: int = 200
    seed: int = 42
    eps: float = 1e-6
    min_pulses_per_label: int = 20


def _time_vector(cfg: TFSelectivityConfig) -> np.ndarray:
    p = cfg.pulse
    full0 = p.trace_pre if p.trace_pre is not None else p.pre_window[0]
    return np.arange(full0, p.post_window[1], p.dt, dtype=float)


def _per_pulse_rate_matrix(
    spike_times: np.ndarray,
    pulse_times: np.ndarray,
    t_vec: np.ndarray,
    dt: float,
    sigma_ms: float,
) -> np.ndarray:
    """(n_pulses, n_time) matrix of per-pulse Gaussian-smoothed rate in Hz."""
    st = np.asarray(spike_times, dtype=float).ravel()
    pulse_times = np.asarray(pulse_times, dtype=float).ravel()
    pulse_times = pulse_times[np.isfinite(pulse_times)]
    if pulse_times.size == 0:
        return np.zeros((0, t_vec.size), dtype=float)
    sigma_bins = (sigma_ms / 1000.0) / dt
    lo, hi = float(t_vec[0]), float(t_vec[-1] + dt)
    rows = np.empty((pulse_times.size, t_vec.size), dtype=float)
    for k, tp in enumerate(pulse_times):
        rel = st - tp
        rel = rel[(rel >= lo) & (rel < hi)]
        rows[k] = _smooth_binned_activity(rel, t_vec, sigma_bins) / dt
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_tf_selectivity.py -q`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/tf_selectivity.py tests/analysis/test_tf_selectivity.py
git commit -m "$(cat <<'EOF'
feat(tf): tf_selectivity per-pulse Hz rate matrix + config

New module for fast-minus-slow selectivity. Reuses the fixed pulse
smoother; each row is one pulse's smoothed rate in Hz.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Shared-baseline selectivity

Add the shared-baseline helper and a first version of `compute_unit_selectivity` that returns the traces, the shared `(mu_b, sigma_b)`, and `selectivity = (fast_hz − slow_hz)/sigma_b`. The shared baseline is the fix for the old circular separate-baseline z-scoring: **one** `sigma_b` pooled across the pre-window of both mean traces, used for both conditions.

**Files:**
- Modify: `src/visdetect/analysis/tf_selectivity.py`
- Test: `tests/analysis/test_tf_selectivity.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/analysis/test_tf_selectivity.py`:

```python
from visdetect.analysis.tf_selectivity import (
    _shared_baseline,
    compute_unit_selectivity,
)


def test_shared_baseline_is_single_value():
    cfg = TFSelectivityConfig()
    t_vec = _time_vector(cfg)
    # Two traces with deliberately different pre-window spread.
    fast_hz = np.zeros_like(t_vec); fast_hz[:] = 10.0
    slow_hz = np.zeros_like(t_vec); slow_hz[:] = 10.0
    pre = (t_vec >= cfg.pulse.pre_window[0]) & (t_vec < cfg.pulse.pre_window[1])
    rng = np.random.default_rng(0)
    fast_hz[pre] += rng.normal(0, 5.0, pre.sum())
    slow_hz[pre] += rng.normal(0, 1.0, pre.sum())
    mu, sd = _shared_baseline(fast_hz, slow_hz, t_vec, cfg.pulse.pre_window, cfg.eps)
    # The pooled sd must lie between the two per-condition sds, i.e. it is one
    # shared number, not computed separately per condition.
    assert sd > 1.0 and sd < 5.0


def test_selectivity_uses_shared_sigma():
    cfg = TFSelectivityConfig(n_shuffles=10)
    t_vec = _time_vector(cfg)
    # Hand-built fast/slow Hz traces: identical baseline, fast bump in post.
    sel = compute_unit_selectivity.__wrapped__ if hasattr(compute_unit_selectivity, "__wrapped__") else None
    # Use the real driver via a tiny session in the next tasks; here we check
    # the algebra directly through the public helper composition:
    fast_hz = np.full_like(t_vec, 8.0)
    slow_hz = np.full_like(t_vec, 8.0)
    post = (t_vec >= 0.0) & (t_vec < 0.2)
    fast_hz[post] = 18.0
    mu, sd = _shared_baseline(fast_hz, slow_hz, t_vec, cfg.pulse.pre_window, cfg.eps)
    selectivity = (fast_hz - slow_hz) / sd
    # baseline difference is zero -> selectivity flat there; post bump positive.
    pre = (t_vec >= cfg.pulse.pre_window[0]) & (t_vec < cfg.pulse.pre_window[1])
    assert np.allclose(selectivity[pre], 0.0)
    assert selectivity[post].max() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_tf_selectivity.py::test_shared_baseline_is_single_value -q`
Expected: FAIL with `ImportError: cannot import name '_shared_baseline'`.

- [ ] **Step 3: Add the shared-baseline helper and a partial driver**

Append to `src/visdetect/analysis/tf_selectivity.py`:

```python
def _shared_baseline(
    fast_hz: np.ndarray,
    slow_hz: np.ndarray,
    t_vec: np.ndarray,
    pre_window: Tuple[float, float],
    eps: float,
) -> Tuple[float, float]:
    """One (mu, sd) pooled over the pre-window bins of BOTH mean traces.

    Using a single shared sigma for fast and slow is the fix for the old
    circular separate-baseline z-scoring (CLAUDE.md "circular baseline").
    """
    pre_mask = (t_vec >= pre_window[0]) & (t_vec < pre_window[1])
    if not np.any(pre_mask):
        return 0.0, 1.0
    pooled = np.concatenate([fast_hz[pre_mask], slow_hz[pre_mask]])
    mu = float(np.nanmean(pooled))
    sd = float(np.nanstd(pooled))
    if not np.isfinite(sd) or sd <= eps:
        sd = 1.0
    return mu, sd
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_tf_selectivity.py::test_shared_baseline_is_single_value tests/analysis/test_tf_selectivity.py::test_selectivity_uses_shared_sigma -q`
Expected: PASS. (`test_selectivity_uses_shared_sigma` only exercises `_shared_baseline` + the algebra; the `compute_unit_selectivity` import resolves once Task 5 lands its full body. If the import line fails at collection, proceed to Task 5 which defines it; re-run there.)

> Note: `compute_unit_selectivity` is imported at the top of the test file but its full body is written in Task 5. To keep Task 4 self-contained, add a minimal stub now and replace it in Task 5:

```python
def compute_unit_selectivity(spike_times, fast_times, slow_times, cfg=None, rng=None):
    raise NotImplementedError  # full body lands in Task 5
```

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/tf_selectivity.py tests/analysis/test_tf_selectivity.py
git commit -m "$(cat <<'EOF'
feat(tf): shared per-unit baseline for selectivity

One sigma pooled over the pre-window of both fast and slow mean traces,
used for both conditions. Fixes the old circular separate-baseline
z-scoring that re-leaked the common-mode ramp.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Post-window metrics + full `compute_unit_selectivity` (no null yet)

Add `_post_metrics` (signed peak, peak latency, signed AUC, half-width) and the full `compute_unit_selectivity` driver that builds the fast/slow matrices, the shared baseline, `selectivity`, `fast_z`/`slow_z`, the post metrics, and the result dataclass. The label-shuffle null fields are filled with placeholders here and completed in Task 6.

**Files:**
- Modify: `src/visdetect/analysis/tf_selectivity.py`
- Test: `tests/analysis/test_tf_selectivity.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/analysis/test_tf_selectivity.py`:

```python
from visdetect.analysis.tf_selectivity import _post_metrics


def test_post_metrics_signed_peak_and_latency():
    cfg = TFSelectivityConfig()
    t_vec = _time_vector(cfg)
    trace = np.zeros_like(t_vec)
    post = (t_vec >= 0.0) & (t_vec < 0.5)
    # a positive bump centred at ~0.10 s
    idx = np.argmin(np.abs(t_vec - 0.10))
    trace[idx - 20: idx + 20] = 5.0
    peak, lat, auc, hw = _post_metrics(trace, t_vec, cfg.pulse.post_window)
    assert np.isclose(peak, 5.0)
    assert abs(lat - 0.10) < 0.03
    assert auc > 0
    assert 0.0 < hw < 0.1


def _make_selectivity_session(n_trials=40, base_rate=20.0, evoked_rate=140.0,
                              evoked_dur=0.15, seed=0, inject=True):
    """Synthetic session yielding BOTH fast and slow pulses.

    Each trial baseline (neutral TF=1) carries alternating fast (TF=2) and slow
    (TF=0.5) samples at post-stride indices 40..140 spaced 1.0 s apart (>=2.0 s,
    before the change at +250 s). The 1.0 s spacing keeps a fast pulse's 0.15 s
    burst tail out of the next slow pulse's [-0.4, 0] pre-window (no cross-pulse
    contamination). The injected cluster fires a regular base train everywhere
    plus a high-rate burst after each FAST pulse only -> positive selectivity bump.
    """
    base_on = (np.arange(n_trials) * 300.0).astype(float)
    change_on = base_on + 250.0
    trials, fast_t, slow_t = [], [], []
    for k in range(n_trials):
        bv = np.ones(3 * 200)
        for j, idx in enumerate(range(40, 160, 20)):
            val = 2.0 if (j % 2 == 0) else 0.5
            bv[3 * idx] = val
            t_abs = base_on[k] + idx * 0.05
            (fast_t if val == 2.0 else slow_t).append(t_abs)
        trials.append(Trial(trialoutcome="Hit", reactiontimes={"RT": 0.3},
                            change_size=2.0, change_time=250.0,
                            baseline_values=bv, n_seen=None))
    fast_t = np.array(fast_t); slow_t = np.array(slow_t)
    t_end = float(change_on[-1] + 10.0)
    spikes = [np.arange(0.0, t_end, 1.0 / base_rate)]
    if inject:
        for tp in fast_t:
            spikes.append(np.arange(tp + 0.005, tp + 0.005 + evoked_dur, 1.0 / evoked_rate))
    spikes = np.sort(np.concatenate(spikes))
    clusters = [Cluster(cluster_id=0, spike_times=spikes, quality="good")]
    ni = {"Baseline_ON": base_on, "Change_ON": change_on}
    sess = Session(trials=trials, clusters=clusters, subject="SYN",
                   session_name="SEL", good_cluster_ids=[0], ni_events=ni)
    return sess, fast_t, slow_t


def test_compute_unit_selectivity_detects_injected_unit():
    cfg = TFSelectivityConfig(n_shuffles=50)
    sess, fast_t, slow_t = _make_selectivity_session(inject=True)
    st = sess.clusters[0].spike_times
    sel = compute_unit_selectivity(st, fast_t, slow_t, cfg)
    assert sel.n_fast > 0 and sel.n_slow > 0
    assert sel.sufficient
    # injected fast-locked unit -> clearly positive selectivity peak
    assert sel.sel_peak > 3.0, sel.sel_peak
    assert 0.0 < sel.sel_peak_latency < 0.25
    # Common-mode (drift) must cancel in the baseline. Check away from t=0: the
    # smoothed response legitimately smears ~50 ms back past the pulse (17 ms
    # sigma), so exclude the last 50 ms of the pre-window -- that smear is a real
    # effect, not leakage. Cancellation is essentially perfect in the rest.
    clean = (sel.t_vec >= cfg.pulse.pre_window[0]) & (sel.t_vec < -0.05)
    assert np.nanmax(np.abs(sel.selectivity[clean])) < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -m pytest tests/analysis/test_tf_selectivity.py::test_post_metrics_signed_peak_and_latency tests/analysis/test_tf_selectivity.py::test_compute_unit_selectivity_detects_injected_unit -q`
Expected: FAIL — `_post_metrics` import error / `compute_unit_selectivity` raises `NotImplementedError`.

- [ ] **Step 3: Add `_post_metrics`, the result dataclass, and the full driver**

In `src/visdetect/analysis/tf_selectivity.py`, add the dataclass after `TFSelectivityConfig`:

```python
@dataclass
class TFUnitSelectivity:
    cluster_id: int
    t_vec: np.ndarray
    fast_hz: np.ndarray
    slow_hz: np.ndarray
    selectivity: np.ndarray
    fast_z: np.ndarray
    slow_z: np.ndarray
    baseline_mu: float
    baseline_sd: float
    sel_peak: float            # signed; selectivity value at |peak| in post window
    sel_peak_latency: float    # s
    sel_auc: float             # signed area under selectivity in post window
    sel_half_width: float      # s; width at half-max around the peak
    fast_peak: float           # signed fast_z post-window peak (sub-typing)
    slow_peak: float           # signed slow_z post-window peak (sub-typing)
    n_fast: int
    n_slow: int
    null_peak_mean: float
    null_peak_sd: float
    sel_z_vs_null: float
    shuffle_p: float
    split_half_r: float
    sufficient: bool
```

Add `_post_metrics`:

```python
def _post_metrics(
    trace: np.ndarray,
    t_vec: np.ndarray,
    post_window: Tuple[float, float],
) -> Tuple[float, float, float, float]:
    """Signed peak, peak latency (s), signed AUC, and half-width (s) in post."""
    post_mask = (t_vec >= post_window[0]) & (t_vec < post_window[1])
    if not np.any(post_mask):
        return np.nan, np.nan, np.nan, np.nan
    seg = trace[post_mask]
    tt = t_vec[post_mask]
    if not np.any(np.isfinite(seg)):
        return np.nan, np.nan, np.nan, np.nan
    i_peak = int(np.nanargmax(np.abs(seg)))
    peak = float(seg[i_peak])
    latency = float(tt[i_peak])
    auc = float(np.trapz(seg, tt))
    half = abs(peak) / 2.0
    lo = i_peak
    while lo > 0 and abs(seg[lo - 1]) >= half:
        lo -= 1
    hi = i_peak
    while hi < seg.size - 1 and abs(seg[hi + 1]) >= half:
        hi += 1
    half_width = float(tt[hi] - tt[lo])
    return peak, latency, auc, half_width
```

Replace the `compute_unit_selectivity` stub with the full body (the null section is a placeholder filled in Task 6):

```python
def compute_unit_selectivity(spike_times, fast_times, slow_times, cfg=None, rng=None) -> TFUnitSelectivity:
    if cfg is None:
        cfg = TFSelectivityConfig()
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    p = cfg.pulse
    t_vec = _time_vector(cfg)
    mat_fast = _per_pulse_rate_matrix(spike_times, fast_times, t_vec, p.dt, p.sigma_ms)
    mat_slow = _per_pulse_rate_matrix(spike_times, slow_times, t_vec, p.dt, p.sigma_ms)
    n_fast, n_slow = mat_fast.shape[0], mat_slow.shape[0]
    sufficient = (n_fast >= cfg.min_pulses_per_label) and (n_slow >= cfg.min_pulses_per_label)

    if n_fast == 0 or n_slow == 0:
        nan = np.full(t_vec.size, np.nan)
        return TFUnitSelectivity(
            cluster_id=-1, t_vec=t_vec, fast_hz=nan.copy(), slow_hz=nan.copy(),
            selectivity=nan.copy(), fast_z=nan.copy(), slow_z=nan.copy(),
            baseline_mu=np.nan, baseline_sd=np.nan, sel_peak=np.nan,
            sel_peak_latency=np.nan, sel_auc=np.nan, sel_half_width=np.nan,
            fast_peak=np.nan, slow_peak=np.nan, n_fast=n_fast, n_slow=n_slow,
            null_peak_mean=np.nan, null_peak_sd=np.nan, sel_z_vs_null=np.nan,
            shuffle_p=np.nan, split_half_r=np.nan, sufficient=False)

    fast_hz = np.nanmean(mat_fast, axis=0)
    slow_hz = np.nanmean(mat_slow, axis=0)
    mu_b, sd_b = _shared_baseline(fast_hz, slow_hz, t_vec, p.pre_window, cfg.eps)
    selectivity = (fast_hz - slow_hz) / sd_b
    fast_z = (fast_hz - mu_b) / sd_b
    slow_z = (slow_hz - mu_b) / sd_b
    sel_peak, sel_lat, sel_auc, sel_hw = _post_metrics(selectivity, t_vec, p.post_window)
    fast_peak, _, _, _ = _post_metrics(fast_z, t_vec, p.post_window)
    slow_peak, _, _, _ = _post_metrics(slow_z, t_vec, p.post_window)

    # Label-shuffle null + split-half are filled in Task 6/7.
    null_peak_mean = np.nan
    null_peak_sd = np.nan
    sel_z_vs_null = np.nan
    shuffle_p = np.nan
    split_half_r = np.nan

    return TFUnitSelectivity(
        cluster_id=-1, t_vec=t_vec, fast_hz=fast_hz, slow_hz=slow_hz,
        selectivity=selectivity, fast_z=fast_z, slow_z=slow_z,
        baseline_mu=mu_b, baseline_sd=sd_b, sel_peak=sel_peak,
        sel_peak_latency=sel_lat, sel_auc=sel_auc, sel_half_width=sel_hw,
        fast_peak=fast_peak, slow_peak=slow_peak, n_fast=n_fast, n_slow=n_slow,
        null_peak_mean=null_peak_mean, null_peak_sd=null_peak_sd,
        sel_z_vs_null=sel_z_vs_null, shuffle_p=shuffle_p,
        split_half_r=split_half_r, sufficient=sufficient)
```

Also remove the Task-4 stub of `compute_unit_selectivity` and the `__wrapped__` line in `test_selectivity_uses_shared_sigma` is harmless (it sets `sel=None` then ignores it); leave that test as-is.

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -m pytest tests/analysis/test_tf_selectivity.py -q`
Expected: PASS (all selectivity tests so far).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/tf_selectivity.py tests/analysis/test_tf_selectivity.py
git commit -m "$(cat <<'EOF'
feat(tf): compute_unit_selectivity core + post-window metrics

Builds fast/slow Hz traces, shared-baseline selectivity, signed
post-window peak/latency/AUC/half-width. Detects an injected fast-locked
synthetic unit; baseline difference cancels to ~0.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Label-shuffle null + significance

Fill the null. Stack fast+slow per-pulse matrices, repeatedly re-partition the rows into `n_fast`/`n_slow` random groups (keeping counts), recompute the selectivity peak with the **fixed** `sigma_b`, and build a null distribution of `|peak|`. Report `sel_z_vs_null` and a one-sided `shuffle_p`. The shuffle preserves the ramp/drift entirely and destroys only the fast/slow assignment.

**Files:**
- Modify: `src/visdetect/analysis/tf_selectivity.py` (`compute_unit_selectivity` null section)
- Test: `tests/analysis/test_tf_selectivity.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/analysis/test_tf_selectivity.py`:

```python
def test_null_separates_injected_from_random():
    cfg = TFSelectivityConfig(n_shuffles=200, seed=1)
    sess_pos, fast_t, slow_t = _make_selectivity_session(inject=True, seed=1)
    sel_pos = compute_unit_selectivity(sess_pos.clusters[0].spike_times, fast_t, slow_t, cfg)
    # injected unit clears the null
    assert sel_pos.shuffle_p < 0.05, sel_pos.shuffle_p
    assert sel_pos.sel_z_vs_null > 3.0, sel_pos.sel_z_vs_null

    cfg2 = TFSelectivityConfig(n_shuffles=200, seed=2)
    sess_neg, fast_t2, slow_t2 = _make_selectivity_session(inject=False, seed=2)
    sel_neg = compute_unit_selectivity(sess_neg.clusters[0].spike_times, fast_t2, slow_t2, cfg2)
    # no fast/slow difference (only common-mode base) -> stays in the null
    assert sel_neg.shuffle_p > 0.05, sel_neg.shuffle_p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_tf_selectivity.py::test_null_separates_injected_from_random -q`
Expected: FAIL — `shuffle_p` is currently NaN, so `NaN < 0.05` is False.

- [ ] **Step 3: Implement the null section**

In `compute_unit_selectivity`, replace the placeholder null block:

```python
    # Label-shuffle null + split-half are filled in Task 6/7.
    null_peak_mean = np.nan
    null_peak_sd = np.nan
    sel_z_vs_null = np.nan
    shuffle_p = np.nan
    split_half_r = np.nan
```

with:

```python
    # Label-shuffle null: permute fast/slow labels (counts fixed), keeping the
    # ramp/drift intact; destroys only the TF assignment.
    combined = np.vstack([mat_fast, mat_slow])
    n_total = combined.shape[0]
    post_mask = (t_vec >= p.post_window[0]) & (t_vec < p.post_window[1])
    null_peaks = np.empty(cfg.n_shuffles, dtype=float)
    for s in range(cfg.n_shuffles):
        perm = rng.permutation(n_total)
        f = np.nanmean(combined[perm[:n_fast]], axis=0)
        sl = np.nanmean(combined[perm[n_fast:]], axis=0)
        sel_s = (f - sl) / sd_b
        null_peaks[s] = float(np.nanmax(np.abs(sel_s[post_mask]))) if np.any(post_mask) else np.nan
    null_peak_mean = float(np.nanmean(null_peaks))
    null_peak_sd = float(np.nanstd(null_peaks))
    obs = abs(sel_peak)
    sel_z_vs_null = (obs - null_peak_mean) / null_peak_sd if null_peak_sd > cfg.eps else np.nan
    shuffle_p = float((1 + np.sum(null_peaks >= obs)) / (1 + cfg.n_shuffles))

    # split-half filled in Task 7
    split_half_r = np.nan
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -m pytest tests/analysis/test_tf_selectivity.py -q`
Expected: PASS (all, including the null separation test).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/tf_selectivity.py tests/analysis/test_tf_selectivity.py
git commit -m "$(cat <<'EOF'
feat(tf): label-shuffle null for selectivity significance

Permute fast/slow labels (counts fixed) keeping the ramp intact; build a
null of |selectivity peak|. Injected unit clears the null; a common-mode
unit stays inside it.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Split-half reliability + sufficiency/div-zero guards

Add the within-unit split-half reliability of the selectivity (random halves of the pulses, to avoid the early/late-session confound of a contiguous split), wire it into `compute_unit_selectivity`, and add explicit tests for the sufficiency guard and the silent-unit div-zero guard.

**Files:**
- Modify: `src/visdetect/analysis/tf_selectivity.py`
- Test: `tests/analysis/test_tf_selectivity.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/analysis/test_tf_selectivity.py`:

```python
def test_split_half_high_for_injected_unit():
    cfg = TFSelectivityConfig(n_shuffles=50, seed=3)
    sess, fast_t, slow_t = _make_selectivity_session(inject=True, seed=3)
    sel = compute_unit_selectivity(sess.clusters[0].spike_times, fast_t, slow_t, cfg)
    assert sel.split_half_r > 0.5, sel.split_half_r


def test_insufficient_pulses_flagged():
    cfg = TFSelectivityConfig(n_shuffles=10, min_pulses_per_label=1000)
    sess, fast_t, slow_t = _make_selectivity_session(inject=True, seed=4)
    sel = compute_unit_selectivity(sess.clusters[0].spike_times, fast_t, slow_t, cfg)
    assert sel.sufficient is False  # far fewer than 1000 pulses per label


def test_silent_unit_does_not_crash():
    cfg = TFSelectivityConfig(n_shuffles=10, seed=5)
    sess, fast_t, slow_t = _make_selectivity_session(inject=True, seed=5)
    sel = compute_unit_selectivity(np.array([]), fast_t, slow_t, cfg)
    # no spikes -> zero traces, finite (guarded) selectivity, no exception
    assert sel.n_fast > 0 and sel.n_slow > 0
    assert np.all(np.isfinite(sel.selectivity))
    assert np.allclose(sel.selectivity, 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -m pytest tests/analysis/test_tf_selectivity.py::test_split_half_high_for_injected_unit tests/analysis/test_tf_selectivity.py::test_insufficient_pulses_flagged tests/analysis/test_tf_selectivity.py::test_silent_unit_does_not_crash -q`
Expected: FAIL — `split_half_r` is NaN (`NaN > 0.5` is False); the other two should already pass (they document guards that must hold) but run them to confirm.

- [ ] **Step 3: Add `_split_half_r` and wire it in**

Append `_split_half_r` to `src/visdetect/analysis/tf_selectivity.py`:

```python
def _split_half_r(
    mat_fast: np.ndarray,
    mat_slow: np.ndarray,
    t_vec: np.ndarray,
    pre_window: Tuple[float, float],
    post_window: Tuple[float, float],
    eps: float,
    rng: np.random.Generator,
) -> float:
    """Correlate post-window selectivity computed from two random halves."""
    nf, ns = mat_fast.shape[0], mat_slow.shape[0]
    if nf < 4 or ns < 4:
        return np.nan
    fi = rng.permutation(nf)
    si = rng.permutation(ns)
    fh1, fh2 = fi[: nf // 2], fi[nf // 2:]
    sh1, sh2 = si[: ns // 2], si[ns // 2:]

    def _sel(mf, ms):
        fhz = np.nanmean(mf, axis=0)
        shz = np.nanmean(ms, axis=0)
        _, sd = _shared_baseline(fhz, shz, t_vec, pre_window, eps)
        return (fhz - shz) / sd

    s1 = _sel(mat_fast[fh1], mat_slow[sh1])
    s2 = _sel(mat_fast[fh2], mat_slow[sh2])
    post_mask = (t_vec >= post_window[0]) & (t_vec < post_window[1])
    a, b = s1[post_mask], s2[post_mask]
    if np.std(a) <= eps or np.std(b) <= eps:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])
```

In `compute_unit_selectivity`, replace the placeholder:

```python
    # split-half filled in Task 7
    split_half_r = np.nan
```

with:

```python
    split_half_r = _split_half_r(mat_fast, mat_slow, t_vec, p.pre_window,
                                 p.post_window, cfg.eps, rng)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -m pytest tests/analysis/test_tf_selectivity.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/tf_selectivity.py tests/analysis/test_tf_selectivity.py
git commit -m "$(cat <<'EOF'
feat(tf): split-half reliability + sufficiency/div-zero guards

Random-half split-half correlation of the post-window selectivity;
silent units return guarded zero selectivity; thin units flagged
insufficient.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Feature row + session driver

Add `unit_features` (flat dict per unit for the eventual model/cache) and `compute_session_selectivity` (loop over cluster IDs, set `cluster_id`, share one RNG). These are the seams the gate script and later phases consume.

**Files:**
- Modify: `src/visdetect/analysis/tf_selectivity.py`
- Test: `tests/analysis/test_tf_selectivity.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/analysis/test_tf_selectivity.py`:

```python
from visdetect.analysis.tf_selectivity import unit_features, compute_session_selectivity

EXPECTED_FEATURE_KEYS = {
    "cluster_id", "sel_peak", "sel_peak_latency", "sel_auc", "sel_half_width",
    "fast_peak", "slow_peak", "sel_z_vs_null", "shuffle_p", "split_half_r",
    "n_fast", "n_slow", "baseline_sd", "sufficient",
}


def test_unit_features_keys_and_session_driver():
    cfg = TFSelectivityConfig(n_shuffles=20, seed=6)
    sess, fast_t, slow_t = _make_selectivity_session(inject=True, seed=6)
    sels = compute_session_selectivity(sess, [0], fast_t, slow_t, cfg)
    assert len(sels) == 1
    assert sels[0].cluster_id == 0
    feats = unit_features(sels[0])
    assert set(feats.keys()) == EXPECTED_FEATURE_KEYS
    assert feats["cluster_id"] == 0
    assert feats["n_fast"] > 0 and feats["n_slow"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_tf_selectivity.py::test_unit_features_keys_and_session_driver -q`
Expected: FAIL with `ImportError: cannot import name 'unit_features'`.

- [ ] **Step 3: Add `unit_features` and `compute_session_selectivity`**

Append to `src/visdetect/analysis/tf_selectivity.py`:

```python
def unit_features(sel: TFUnitSelectivity) -> Dict[str, float]:
    """Flat per-unit feature row for the model/cache (all-trials, Phase B)."""
    return {
        "cluster_id": int(sel.cluster_id),
        "sel_peak": sel.sel_peak,
        "sel_peak_latency": sel.sel_peak_latency,
        "sel_auc": sel.sel_auc,
        "sel_half_width": sel.sel_half_width,
        "fast_peak": sel.fast_peak,
        "slow_peak": sel.slow_peak,
        "sel_z_vs_null": sel.sel_z_vs_null,
        "shuffle_p": sel.shuffle_p,
        "split_half_r": sel.split_half_r,
        "n_fast": int(sel.n_fast),
        "n_slow": int(sel.n_slow),
        "baseline_sd": sel.baseline_sd,
        "sufficient": bool(sel.sufficient),
    }


def compute_session_selectivity(
    session,
    cluster_ids: List[int],
    fast_times: np.ndarray,
    slow_times: np.ndarray,
    cfg: Optional[TFSelectivityConfig] = None,
) -> List[TFUnitSelectivity]:
    """Per-unit selectivity for the given clusters (one shared RNG)."""
    if cfg is None:
        cfg = TFSelectivityConfig()
    rng = np.random.default_rng(cfg.seed)
    by_id = {int(c.cluster_id): np.asarray(c.spike_times, dtype=float).ravel()
             for c in session.clusters}
    out: List[TFUnitSelectivity] = []
    for cid in cluster_ids:
        st = by_id.get(int(cid))
        if st is None:
            continue
        sel = compute_unit_selectivity(st, fast_times, slow_times, cfg, rng)
        sel.cluster_id = int(cid)
        out.append(sel)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -m pytest tests/analysis/test_tf_selectivity.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/tf_selectivity.py tests/analysis/test_tf_selectivity.py
git commit -m "$(cat <<'EOF'
feat(tf): unit_features row + compute_session_selectivity driver

Flat per-unit feature row and a session-level driver that sets
cluster_id and shares one RNG across units.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Early-validation gate script

Create the gate script with a unit-testable `build_feature_table` seam, plus `main()` that loads a real session, collects corrected pulses once, computes selectivity for all good-and-stable units, writes a feature CSV, renders a 3-panel diagnostic figure, and prints the top exemplar candidates. The script self-inserts its `src/` so it never silently runs the primary repo's code.

**Files:**
- Create: `scripts/tf_responsiveness/validate_selectivity_phase0.py`
- Test: `tests/scripts/test_validate_selectivity_phase0.py`

- [ ] **Step 1: Write the failing smoke test**

Create `tests/scripts/test_validate_selectivity_phase0.py`:

```python
"""Smoke test for the selectivity early-validation gate's pure seam."""
import importlib.util
from pathlib import Path

import numpy as np

from visdetect.core.session import Session, Trial, Cluster
from visdetect.analysis.tf_selectivity import TFSelectivityConfig

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "tf_responsiveness" / "validate_selectivity_phase0.py")


def _load_script_module():
    spec = importlib.util.spec_from_file_location("validate_selectivity_phase0", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_both_pulse_session():
    base_on = (np.arange(20) * 300.0).astype(float)
    change_on = base_on + 250.0
    trials, fast_t = [], []
    for k in range(20):
        bv = np.ones(3 * 200)
        for j, idx in enumerate(range(40, 160, 10)):
            val = 2.0 if (j % 2 == 0) else 0.5
            bv[3 * idx] = val
            if val == 2.0:
                fast_t.append(base_on[k] + idx * 0.05)
        trials.append(Trial(trialoutcome="Hit", reactiontimes={"RT": 0.3},
                            change_size=2.0, change_time=250.0,
                            baseline_values=bv, n_seen=None))
    spikes = [np.arange(0.0, float(change_on[-1] + 10), 0.05)]
    for tp in fast_t:
        spikes.append(np.arange(tp + 0.005, tp + 0.155, 1.0 / 140.0))
    spikes = np.sort(np.concatenate(spikes))
    ni = {"Baseline_ON": base_on, "Change_ON": change_on}
    return Session(trials=trials, clusters=[Cluster(cluster_id=0, spike_times=spikes,
                   quality="good")], subject="SYN", session_name="SEL",
                   good_cluster_ids=[0], ni_events=ni)


def test_build_feature_table_runs():
    mod = _load_script_module()
    sess = _tiny_both_pulse_session()
    cfg = TFSelectivityConfig(n_shuffles=20)
    df = mod.build_feature_table(sess, [0], cfg)
    assert len(df) == 1
    assert {"cluster_id", "sel_peak", "shuffle_p"}.issubset(df.columns)
```

Create the directory marker if your test runner needs it: `tests/scripts/` (no `__init__.py` required — pytest uses rootdir discovery).

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/scripts/test_validate_selectivity_phase0.py -q`
Expected: FAIL — the script file does not exist yet (`FileNotFoundError` / spec load error).

- [ ] **Step 3: Create the gate script**

Create `scripts/tf_responsiveness/validate_selectivity_phase0.py`:

```python
"""Phase-B early-validation gate for the fast-minus-slow selectivity core.

Runs the selectivity detector on a real session's good-and-stable units and
answers the gate question: does a *sparse* population of units have a
fast-minus-slow selectivity peak that exits the label-shuffle null at short
latency (~0.12-0.17 s)? Re-picks clean exemplars for the eventual HITL tagger.

Usage:
    cd /e/python_analysis/git_repos/vd_tf_phase0
    PYTHONPATH=src py scripts/tf_responsiveness/validate_selectivity_phase0.py \
        --session BG_046_16092025

Outputs (under the worktree root):
    data/cache/tf_selectivity/<session>_features.csv
    figures/tf_responsiveness/<session>_selectivity_gate.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Self-insert this worktree's src so we never run the primary repo's editable
# install by accident (the editable visdetect is pinned to the primary src).
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visdetect.analysis.tf_pulse import _collect_pulses
from visdetect.analysis.tf_selectivity import (
    TFSelectivityConfig,
    compute_session_selectivity,
    unit_features,
)
from visdetect.analysis.constants import LOHSE_SENSORY_CD_WINDOW

_ROOT = Path(__file__).resolve().parents[2]
_CACHE = _ROOT / "data" / "cache" / "tf_selectivity"
_FIGS = _ROOT / "figures" / "tf_responsiveness"


def build_feature_table(session, cluster_ids, cfg=None) -> pd.DataFrame:
    """Pure seam: corrected pulses -> per-unit selectivity -> feature table."""
    if cfg is None:
        cfg = TFSelectivityConfig()
    fast_times, slow_times = _collect_pulses(session, cfg.pulse)
    sels = compute_session_selectivity(session, cluster_ids, fast_times, slow_times, cfg)
    rows = [unit_features(s) for s in sels]
    # keep the full selectivity objects on the frame for plotting
    df = pd.DataFrame(rows)
    df.attrs["selectivities"] = sels
    df.attrs["n_fast_total"] = int(np.asarray(fast_times).size)
    df.attrs["n_slow_total"] = int(np.asarray(slow_times).size)
    return df


def _render_gate_figure(df, cfg, out_png, session_name):
    sels = df.attrs.get("selectivities", [])
    sig = df[(df["shuffle_p"] < 0.05) & (df["sufficient"])]
    sig_ids = set(sig["cluster_id"].tolist())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Panel 1: significant units' selectivity traces + mean
    ax = axes[0]
    traces = []
    for s in sels:
        if int(s.cluster_id) in sig_ids and np.all(np.isfinite(s.selectivity)):
            ax.plot(s.t_vec, s.selectivity, color="0.6", lw=0.6, alpha=0.7)
            traces.append(s.selectivity)
    if traces:
        m = np.nanmean(np.vstack(traces), axis=0)
        ax.plot(sels[0].t_vec, m, color="k", lw=2.0, label=f"mean (n={len(traces)})")
        ax.legend(fontsize=8)
    ax.axvline(0, color="r", ls="--", lw=0.8)
    ax.axvspan(*LOHSE_SENSORY_CD_WINDOW, color="orange", alpha=0.15)
    ax.set_xlabel("time from pulse (s)")
    ax.set_ylabel("selectivity (fast-slow)/sigma_b")
    ax.set_title("Significant-unit selectivity")

    # Panel 2: peak-latency histogram of significant units
    ax = axes[1]
    if len(sig):
        ax.hist(sig["sel_peak_latency"], bins=20, range=(0, 0.5), color="steelblue")
    ax.axvspan(*LOHSE_SENSORY_CD_WINDOW, color="orange", alpha=0.25,
               label="Lohse 0.122-0.167 s")
    ax.set_xlabel("peak latency (s)")
    ax.set_ylabel("# significant units")
    ax.set_title("Peak latency")
    ax.legend(fontsize=8)

    # Panel 3: |sel_peak| vs sel_z_vs_null, coloured by significance
    ax = axes[2]
    finite = df[np.isfinite(df["sel_z_vs_null"])]
    is_sig = (finite["shuffle_p"] < 0.05) & (finite["sufficient"])
    ax.scatter(finite.loc[~is_sig, "sel_peak"].abs(),
               finite.loc[~is_sig, "sel_z_vs_null"], s=10, color="0.7", label="ns")
    ax.scatter(finite.loc[is_sig, "sel_peak"].abs(),
               finite.loc[is_sig, "sel_z_vs_null"], s=14, color="crimson", label="p<0.05")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("|sel peak|")
    ax.set_ylabel("selectivity z vs null")
    ax.set_title("Detection scatter")
    ax.legend(fontsize=8)

    fig.suptitle(f"TF selectivity gate — {session_name} "
                 f"(fast={df.attrs.get('n_fast_total')}, slow={df.attrs.get('n_slow_total')})")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="e.g. BG_046_16092025 or 16092025")
    ap.add_argument("--n-shuffles", type=int, default=200)
    ap.add_argument("--top", type=int, default=15, help="# exemplar candidates to print")
    args = ap.parse_args()

    from visdetect.suite.loader import load_session
    from visdetect.analysis.utils import get_good_cluster_ids

    sess = load_session(args.session)
    cluster_ids = get_good_cluster_ids(sess)
    print(f"[gate] {args.session}: {len(cluster_ids)} good-and-stable units")

    cfg = TFSelectivityConfig(n_shuffles=args.n_shuffles)
    df = build_feature_table(sess, cluster_ids, cfg)

    sname = str(getattr(sess, "session_name", args.session))
    _CACHE.mkdir(parents=True, exist_ok=True)
    csv_path = _CACHE / f"{sname}_features.csv"
    df.drop(columns=[]).to_csv(csv_path, index=False)
    print(f"[gate] wrote {csv_path}")

    _render_gate_figure(df, cfg, _FIGS / f"{sname}_selectivity_gate.png", sname)
    print(f"[gate] wrote {_FIGS / f'{sname}_selectivity_gate.png'}")

    sig = df[(df["shuffle_p"] < 0.05) & (df["sufficient"])].copy()
    n_total = int((df["sufficient"]).sum())
    frac = (len(sig) / n_total) if n_total else float("nan")
    print(f"[gate] significant responders: {len(sig)} / {n_total} sufficient "
          f"units ({100*frac:.1f}%)")
    in_win = sig[(sig["sel_peak_latency"] >= LOHSE_SENSORY_CD_WINDOW[0]) &
                 (sig["sel_peak_latency"] <= LOHSE_SENSORY_CD_WINDOW[1])]
    print(f"[gate] of those, {len(in_win)} peak in Lohse window "
          f"{LOHSE_SENSORY_CD_WINDOW}")

    print(f"[gate] top {args.top} exemplar candidates (by sel_z_vs_null):")
    cols = ["cluster_id", "sel_peak", "sel_peak_latency", "sel_z_vs_null",
            "shuffle_p", "split_half_r", "n_fast", "n_slow"]
    top = sig.sort_values("sel_z_vs_null", ascending=False).head(args.top)
    print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `py -m pytest tests/scripts/test_validate_selectivity_phase0.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/tf_responsiveness/validate_selectivity_phase0.py tests/scripts/test_validate_selectivity_phase0.py
git commit -m "$(cat <<'EOF'
feat(tf): selectivity early-validation gate script

Runs the selectivity detector on a real session, writes a feature CSV
and a 3-panel diagnostic, prints yield and exemplar candidates. Pure
build_feature_table seam is smoke-tested.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Retire the dead drift module

The drift approach is dead (killed by its own Phase-0 gate). Remove `tf_drift.py` and its test. Keep `scripts/tf_responsiveness/validate_drift_phase0.py` (the spec designates it the record of the pivot). Confirm the whole suite is green afterward.

**Files:**
- Delete: `src/visdetect/analysis/tf_drift.py`
- Delete: `tests/analysis/test_tf_drift.py`

- [ ] **Step 1: Confirm nothing imports `tf_drift` except its own test**

Run: `grep -rn "tf_drift" src/ scripts/ tests/ analysis_suite/ 2>/dev/null`
Expected: matches only in `tests/analysis/test_tf_drift.py` (and possibly a comment in `validate_drift_phase0.py`). If any *library/script* code imports `tf_drift`, STOP and report — do not delete.

- [ ] **Step 2: Delete the retired files**

```bash
git rm src/visdetect/analysis/tf_drift.py tests/analysis/test_tf_drift.py
```

- [ ] **Step 3: Run the full test suite to verify no regressions**

Run: `py -m pytest -q`
Expected: PASS (no collection errors from the deleted module; all TF tests green).

- [ ] **Step 4: Commit**

```bash
git commit -m "$(cat <<'EOF'
chore(tf): retire dead tf_drift module

The source-level drift-detrend approach was killed by its Phase-0 gate
(within-trial temporal-expectation ramp can't be modelled out). Replaced
by tf_selectivity. validate_drift_phase0.py kept as the pivot record.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Run the gate on a real session and record the verdict

This is the actual early-validation gate — a run + human judgment, not a code change. Run the detector on a trusted real session and decide go/no-go for the GUI/model phases (C–F), recording the verdict so a fresh chat can act on it.

**Files:**
- Modify: `docs/superpowers/specs/2026-06-17-tf-responsiveness-selectivity-design.md` (append a "Phase-B gate verdict" subsection)

- [ ] **Step 1: Run the gate on session 16092025**

```bash
cd /e/python_analysis/git_repos/vd_tf_phase0
PYTHONPATH=src py scripts/tf_responsiveness/validate_selectivity_phase0.py --session BG_046_16092025 --n-shuffles 200
```

Expected: prints unit count, the significant-responder fraction, how many peak in the Lohse window, and a table of top exemplar candidates; writes `data/cache/tf_selectivity/16092025_features.csv` and `figures/tf_responsiveness/16092025_selectivity_gate.png`.

- [ ] **Step 2: Inspect the figure and decide go/no-go**

Open `figures/tf_responsiveness/16092025_selectivity_gate.png`. Apply the spec §8 criteria:
- A *sparse* set of units (expect low; Lohse ~3% and posterior-biased, BG_046 medial striatum possibly sparser — near-zero is a finding, not a failure).
- Significant units' selectivity peaks cluster at **short latency** (~0.12–0.17 s, the shaded Lohse window) and exit the null (Panel 3 crimson points well above 0).
- Mean significant trace (Panel 1) shows a clean post-pulse deflection, flat baseline.

GO if a coherent short-latency population exists (even if small). NO-GO/iterate if "significant" units have random latencies or baseline structure — that points to a residual confound to investigate before building the GUI.

- [ ] **Step 3 (optional sanity): run a second session**

```bash
PYTHONPATH=src py scripts/tf_responsiveness/validate_selectivity_phase0.py --session BG_046_14082025 --n-shuffles 200
```
Confirm the picture is consistent (sparse, short-latency). If `BG_046_14082025.pkl` is not present in this worktree, skip — one trusted session satisfies the gate.

- [ ] **Step 4: Record the verdict in the spec**

Append to `docs/superpowers/specs/2026-06-17-tf-responsiveness-selectivity-design.md`:

```markdown
## 13. Phase-B gate verdict (2026-06-17)

Ran `validate_selectivity_phase0.py` on <session(s)>. Result:
- Good-and-stable units: <N>. Significant responders (shuffle_p<0.05 & sufficient): <k> (<pct>%).
- Peak-latency clustering: <describe — in/around the 0.122–0.167 s Lohse window?>.
- Decision: <GO to Phase C–F | iterate>. 
- Re-picked exemplar candidates (corrected pulses, supersede pre-fix picks):
  <cluster_ids with sel_peak / latency / sel_z_vs_null>.
```

Fill the angle-bracket placeholders with the actual numbers from Step 1 before committing.

- [ ] **Step 5: Commit the verdict**

```bash
git add docs/superpowers/specs/2026-06-17-tf-responsiveness-selectivity-design.md
git commit -m "$(cat <<'EOF'
docs(tf): Phase-B early-validation gate verdict + re-picked exemplars

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## What Phase B deliberately does NOT include (deferred to C–F)

- Full per-state extraction + parallelized feature cache (Phase C).
- HITL GUI, seed tags (Phase D).
- Active-learning loop + shallow model + LOSO κ (Phase E).
- Tag-all + `tf_responsive_tags.csv` integration + retiring `g_tf_cell_classifier.py`/Fig41 (Phase F).

Phase B's job is to prove the detection core is trustworthy on corrected pulses and re-pick clean exemplars before any of that is built.

---

## Self-Review

**1. Spec coverage** (spec → task):
- §1 off-by-one fix → already landed (commit `e53ddd1`, pre-plan).
- §1 / §4 separate-baseline fix → Task 4 (`_shared_baseline`).
- §4 minor fix (a) binary-train undercount → Task 1.
- §4 minor fix (b) n_seen/NaN-change leakage → Task 2.
- §4 step 1 corrected pulses → reused fixed `_collect_pulses` (Task 9 `build_feature_table`).
- §4 step 2 fast/slow Hz traces over [-1,+0.5] → Task 3 (`_per_pulse_rate_matrix`) + Task 5 (`_time_vector`, driver).
- §4 step 3 shared-baseline selectivity → Task 4 + Task 5.
- §4 step 4 signed peak + AUC, transient & sustained → Task 5 (`_post_metrics`: peak, latency, AUC, half-width).
- §4 step 5 label-shuffle null → Task 6.
- §5 features (peak/AUC/latency/half-width, signed fast & slow peaks, z-vs-null + p, split-half, n_pulses sufficiency) → Task 5 (signed fast/slow peaks), Task 6 (z/p), Task 7 (split-half, sufficiency), Task 8 (`unit_features`). State-gating index is Phase C (per-state) — correctly out of Phase B.
- §8 validation/yield/latency sanity → Task 9 (gate figure + prints) + Task 11 (verdict).
- §9 retire `tf_drift.py`, keep `validate_drift_phase0.py` → Task 10.
- §11 Phase B "build core (TDD) + early-validation gate + re-pick exemplars" → Tasks 3–9 (build) + Task 11 (gate/exemplars).

No Phase-B spec requirement is left without a task.

**2. Placeholder scan:** Every code step contains complete code. The only intentional transient stub is `compute_unit_selectivity` in Task 4 (explicitly replaced in Task 5) and the null/split-half placeholders inside it (filled in Tasks 6/7) — each is named with the task that completes it. Task 11's spec-append has angle-bracket fields the engineer fills from real run output (a run artifact, not a code placeholder), with an explicit instruction to fill them before committing.

**3. Type/name consistency:** `TFSelectivityConfig` (fields `pulse`, `n_shuffles`, `seed`, `eps`, `min_pulses_per_label`), `TFUnitSelectivity` (all fields referenced by `unit_features` and the gate figure exist), `_time_vector`, `_per_pulse_rate_matrix(spike_times, pulse_times, t_vec, dt, sigma_ms)`, `_shared_baseline(fast_hz, slow_hz, t_vec, pre_window, eps)`, `_post_metrics(trace, t_vec, post_window)`, `_split_half_r(..., rng)`, `compute_unit_selectivity(spike_times, fast_times, slow_times, cfg, rng)`, `compute_session_selectivity(session, cluster_ids, fast_times, slow_times, cfg)`, `unit_features(sel)` — names are identical across all tasks, tests, and the gate script. `build_feature_table(session, cluster_ids, cfg)` signature matches its smoke test. `EXPECTED_FEATURE_KEYS` matches the dict `unit_features` returns. Reused `tf_pulse` names (`_collect_pulses`, `_smooth_binned_activity`, `TFRespPulseConfig`) match the audited source. `get_good_cluster_ids(session)` and `load_session(session_name)` match the verified library API.
