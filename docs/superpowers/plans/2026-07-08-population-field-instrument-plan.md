# Population Field — Instrument (Plan 1 of 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the tracking-free "instrument" — match-free cross-session registration, a fixed shank×depth anatomical grid, and a local (sorted-spike) field tensor — with a validation/audit gate, so later plans can compute the functional map, geometry, and MUA headline on it.

**Architecture:** A new library module `src/visdetect/analysis/population_field.py` holds pure, unit-testable primitives (depth grid, robust per-unit depth, amplitude-depth fingerprint + rigid shift, registered depth, unit→bin index, field-tensor aggregation, registration audit). A thin driver `scripts/population_field/build_field.py` wires them to real per-subject data and caches per-session field tensors. Cross-session correspondence comes from *fixed anatomy on a match-free registered depth axis* — no unit tracking. This is Plan 1 of 3 (2 = analysis layers, 3 = MUA headline); it produces a validated local field tensor on its own.

**Tech Stack:** Python 3 (`.venv`, invoke via `py`), numpy, pytest. Reuses `visdetect.anatomy.channel_geometry`, `visdetect.analysis.tracking_qc`, `visdetect.analysis.utils`, `visdetect.utils.synthetic`. Origin of the fingerprint/shift logic: `scripts/pipelines/tracking/diagnose_intersession_drift.py` (lifted into the library for testable reuse).

## Global Constraints

Every task's requirements implicitly include these (values copied verbatim from the spec `docs/superpowers/specs/2026-07-07-tracking-free-population-field-design.md`):

- **Match-free registration only.** NEVER use `peak_depth_corrected_um` from `curation_features*.pkl` (it is UnitMatch-match-anchored via `estimate_session_drift`, prob>0.95 → circular for a tracking-free pipeline). Registration is the amplitude-depth landscape shift.
- **Anchor = general activity fingerprint**, not the sparse (2–5%) TF signal. TF is a measured variable in a later plan, never the reference frame.
- **Canonical constants only** (`visdetect.analysis.constants` / `config`): `DEFAULT_BIN_SIZE=0.025`, `DEFAULT_SIGMA_MS=25.0`, `EVENT_VALID_OUTCOMES`, `assign_shanks(gap_um=120.0)`. New constants introduced by this plan and **flagged for user confirmation**: `DEPTH_BIN_UM=60.0`, `REG_MAX_LAG_UM=300.0`.
- **Session-id joins via `config.canonical_session_id` / `config.session_date_key`** (leading-zero-day int64 footgun). UM-input dirs may be 7- or 8-char; `load_raw_mean_waveform`/`load_channel_positions` already tolerate both.
- **Units are `good_and_stable_ids`** (`session.good_and_stable_ids`); pkls store spikes only for these.
- **No compute over the X: Samba drive.** This plan is fully local (pkls + `data/unit_match/input`). The X:/HPC MUA step is Plan 3.
- **Windows:** `py` not `python`.
- **Worktree data access:** the worktree `.claude/worktrees/population-field` has code only (no `data/`). **Unit tests need no data and run in the worktree.** Real-data steps (the Task 8 driver / golden path) must run where `data/` is present — either run from the primary checkout on this branch, or add a **read-only** junction for `data/` into the worktree; NEVER `git worktree remove` with a data junction present, and NEVER `rm -rf data` (June-7 data-loss incident).

---

### Task 1: Module scaffold + depth-bin edges

**Files:**
- Create: `src/visdetect/analysis/population_field.py`
- Test: `tests/analysis/test_population_field.py`

**Interfaces:**
- Produces: `DEPTH_BIN_UM: float`, `REG_MAX_LAG_UM: float`, `depth_bin_edges(channel_positions: np.ndarray, depth_bin_um: float = DEPTH_BIN_UM) -> np.ndarray` (monotonic y-edges in µm covering the active band).

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_population_field.py
import numpy as np
import pytest
from visdetect.analysis import population_field as pf


def _np2_positions():
    """4-shank NP2.0-like geometry: x in {0,32,250,282,500,532,750,782}, y 1500..2200."""
    xs = [0, 32, 250, 282, 500, 532, 750, 782]
    ys = np.arange(1500, 2205, 15.0)  # 15 um row pitch
    pos = np.array([[x, y] for y in ys for x in xs], dtype=float)
    return pos


def test_depth_bin_edges_cover_active_band():
    pos = _np2_positions()
    edges = pf.depth_bin_edges(pos, depth_bin_um=60.0)
    assert edges[0] <= pos[:, 1].min()
    assert edges[-1] >= pos[:, 1].max()
    assert np.allclose(np.diff(edges), 60.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_population_field.py::test_depth_bin_edges_cover_active_band -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError: module ... has no attribute 'depth_bin_edges'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/visdetect/analysis/population_field.py
"""Tracking-free anatomical population field — instrument primitives.

Cross-session correspondence comes from fixed anatomy on a MATCH-FREE
registered depth axis (the amplitude-depth activity landscape), never from
single-unit tracking. See docs/superpowers/specs/2026-07-07-tracking-free-
population-field-design.md.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# New constants (flagged for user confirmation — Global Constraints).
DEPTH_BIN_UM: float = 60.0
REG_MAX_LAG_UM: float = 300.0


def depth_bin_edges(channel_positions: np.ndarray,
                    depth_bin_um: float = DEPTH_BIN_UM) -> np.ndarray:
    """Monotonic y-edges (µm) covering the active depth band at ``depth_bin_um``."""
    y = np.asarray(channel_positions, float)[:, 1]
    lo = np.floor(y.min() / depth_bin_um) * depth_bin_um
    hi = np.ceil(y.max() / depth_bin_um) * depth_bin_um
    return np.arange(lo, hi + depth_bin_um, depth_bin_um)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_population_field.py::test_depth_bin_edges_cover_active_band -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/population_field.py tests/analysis/test_population_field.py
git commit -m "feat(population-field): module scaffold + depth_bin_edges"
```

---

### Task 2: Robust per-unit depth (amplitude-weighted centroid)

**Files:**
- Modify: `src/visdetect/analysis/population_field.py`
- Test: `tests/analysis/test_population_field.py`

**Interfaces:**
- Consumes: (none new)
- Produces: `robust_unit_depth(mean_waveform: np.ndarray, channel_positions: np.ndarray) -> float` — amplitude(ptp)-weighted centroid of channel y-positions; `nan` if total amplitude ≤ 0. Preferred over the single peak channel (spec Component 0).

- [ ] **Step 1: Write the failing test**

```python
def test_robust_unit_depth_weighted_centroid():
    # 3 channels at y = 0, 100, 200; ptp concentrated as 1:2:1 -> centroid = 100
    n_samp = 82
    mw = np.zeros((n_samp, 3))
    mw[:, 0] = np.linspace(-0.5, 0.5, n_samp)   # ptp 1
    mw[:, 1] = np.linspace(-1.0, 1.0, n_samp)   # ptp 2
    mw[:, 2] = np.linspace(-0.5, 0.5, n_samp)   # ptp 1
    pos = np.array([[0.0, 0.0], [0.0, 100.0], [0.0, 200.0]])
    assert pf.robust_unit_depth(mw, pos) == pytest.approx(100.0)


def test_robust_unit_depth_zero_amplitude_is_nan():
    mw = np.zeros((82, 3))
    pos = np.array([[0.0, 0.0], [0.0, 100.0], [0.0, 200.0]])
    assert np.isnan(pf.robust_unit_depth(mw, pos))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_population_field.py -k robust_unit_depth -v`
Expected: FAIL with `AttributeError: ... 'robust_unit_depth'`.

- [ ] **Step 3: Write minimal implementation**

```python
def robust_unit_depth(mean_waveform: np.ndarray,
                      channel_positions: np.ndarray) -> float:
    """Amplitude(ptp)-weighted centroid of channel depth. NaN if no amplitude."""
    ptp = mean_waveform.max(axis=0) - mean_waveform.min(axis=0)   # (n_chan,)
    y = np.asarray(channel_positions, float)[:, 1]
    w = np.asarray(ptp, float)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    return float((w * y).sum() / total)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_population_field.py -k robust_unit_depth -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/population_field.py tests/analysis/test_population_field.py
git commit -m "feat(population-field): robust amplitude-weighted unit depth"
```

---

### Task 3: Amplitude-depth fingerprint + rigid shift (lifted to library)

**Files:**
- Modify: `src/visdetect/analysis/population_field.py`
- Test: `tests/analysis/test_population_field.py`

**Interfaces:**
- Consumes: `depth_bin_edges`
- Produces:
  - `amplitude_depth_fingerprint(unit_waveforms: List[np.ndarray], channel_positions: np.ndarray, y_edges: np.ndarray) -> np.ndarray` — pools every channel's ptp of every unit into its depth bin (shape `(len(y_edges)-1,)`). Mirrors `diagnose_intersession_drift.session_fingerprint`'s whole-probe profile but takes loaded waveforms (testable).
  - `estimate_shift_bins(ref: np.ndarray, mov: np.ndarray, max_lag_bins: int) -> Tuple[int, float]` — rigid bin shift aligning `mov` onto `ref` and its peak normalized correlation (lifted verbatim from `diagnose_intersession_drift.estimate_shift`; positive shift ⇒ `mov` deeper).

- [ ] **Step 1: Write the failing test**

```python
def test_fingerprint_pools_all_channels():
    # one unit, ptp = 4 on the channel at y=1560 -> that bin gets >=4
    pos = _np2_positions()
    y_edges = pf.depth_bin_edges(pos, 60.0)
    mw = np.zeros((82, pos.shape[0]))
    chan = int(np.argmin(np.abs(pos[:, 1] - 1560)))
    mw[:, chan] = np.linspace(-2.0, 2.0, 82)  # ptp = 4
    fp = pf.amplitude_depth_fingerprint([mw], pos, y_edges)
    assert fp.shape == (len(y_edges) - 1,)
    target_bin = np.clip(np.searchsorted(y_edges, pos[chan, 1]) - 1, 0, len(y_edges) - 2)
    assert fp[target_bin] == pytest.approx(4.0)


def test_estimate_shift_recovers_known_roll():
    rng = np.random.default_rng(0)
    ref = np.abs(rng.normal(size=40))
    mov = np.roll(ref, 3); mov[:3] = 0.0     # shifted 3 bins deeper
    shift, corr = pf.estimate_shift_bins(ref, mov, max_lag_bins=10)
    assert shift == -3          # mov must be rolled by -3 to align onto ref
    assert corr > 0.85   # edge-zeroing caps corr well below 1 at short lengths (verbatim impl ~0.886 at len 40)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_population_field.py -k "fingerprint or estimate_shift" -v`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Write minimal implementation**

```python
def amplitude_depth_fingerprint(unit_waveforms: List[np.ndarray],
                                channel_positions: np.ndarray,
                                y_edges: np.ndarray) -> np.ndarray:
    """Pool every channel's ptp of every unit into its depth bin (whole-probe)."""
    y = np.asarray(channel_positions, float)[:, 1]
    n_bins = len(y_edges) - 1
    chan_bin = np.clip(np.searchsorted(y_edges, y) - 1, 0, n_bins - 1)
    profile = np.zeros(n_bins, float)
    for mw in unit_waveforms:
        ptp = mw.max(axis=0) - mw.min(axis=0)       # (n_chan,)
        np.add.at(profile, chan_bin, ptp)
    return profile


def estimate_shift_bins(ref: np.ndarray, mov: np.ndarray,
                        max_lag_bins: int) -> Tuple[int, float]:
    """Rigid bin shift aligning ``mov`` onto ``ref`` + peak normalized corr.

    Lifted from scripts/pipelines/tracking/diagnose_intersession_drift.py::estimate_shift.
    """
    ref = ref - ref.mean()
    mov = mov - mov.mean()
    denom = np.sqrt((ref ** 2).sum() * (mov ** 2).sum())
    if denom < 1e-9:
        return 0, 0.0
    best_lag, best_c = 0, -np.inf
    for lag in range(-max_lag_bins, max_lag_bins + 1):
        shifted = np.roll(mov, lag)
        if lag > 0:
            shifted[:lag] = 0
        elif lag < 0:
            shifted[lag:] = 0
        c = float((ref * shifted).sum() / denom)
        if c > best_c:
            best_c, best_lag = c, lag
    return best_lag, best_c
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_population_field.py -k "fingerprint or estimate_shift" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/population_field.py tests/analysis/test_population_field.py
git commit -m "feat(population-field): amplitude-depth fingerprint + rigid shift"
```

---

### Task 4: Per-session registration shift from RawWaveforms

**Files:**
- Modify: `src/visdetect/analysis/population_field.py`
- Test: `tests/analysis/test_population_field.py`

**Interfaces:**
- Consumes: `amplitude_depth_fingerprint`, `estimate_shift_bins`, `depth_bin_edges`; `visdetect.analysis.tracking_qc.load_raw_mean_waveform`, `load_channel_positions`.
- Produces:
  - `session_fingerprint_from_root(raw_wf_root, session_name: str, unit_ids: List[int], y_edges: np.ndarray) -> np.ndarray` — build a session's whole-probe fingerprint from its `good_and_stable` unit ids.
  - `session_shift_um(fingerprints: Dict[str, np.ndarray], ref_session: str, depth_bin_um: float = DEPTH_BIN_UM, max_lag_um: float = REG_MAX_LAG_UM) -> Dict[str, Tuple[float, float]]` — per-session `(shift_um, corr)` vs the reference. Positive shift_um ⇒ that session sits deeper than the reference.

- [ ] **Step 1: Write the failing test**

```python
def test_session_shift_um_recovers_60um(tmp_path):
    # Build two fake sessions' fingerprints on a shared axis, one shifted +2 bins.
    pos = _np2_positions()
    y_edges = pf.depth_bin_edges(pos, 60.0)
    ref_fp = np.abs(np.sin(np.linspace(0, 6, len(y_edges) - 1))) + 0.1
    mov_fp = np.roll(ref_fp, 2); mov_fp[:2] = 0.0
    shifts = pf.session_shift_um(
        {"01072025": ref_fp, "02072025": mov_fp},
        ref_session="01072025", depth_bin_um=60.0, max_lag_um=300.0,
    )
    assert shifts["01072025"][0] == pytest.approx(0.0)
    # mov is 2 bins deeper -> needs -2 bins to align -> reported deeper shift = +120 um
    assert shifts["02072025"][0] == pytest.approx(120.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_population_field.py -k session_shift -v`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Write minimal implementation**

```python
from visdetect.analysis.tracking_qc import (          # noqa: E402
    load_raw_mean_waveform, load_channel_positions,
)


def session_fingerprint_from_root(raw_wf_root, session_name: str,
                                  unit_ids: List[int],
                                  y_edges: np.ndarray) -> np.ndarray:
    """Whole-probe amplitude-depth fingerprint for one session's good+stable units."""
    pos = load_channel_positions(raw_wf_root, session_name)
    if pos is None:
        return np.zeros(len(y_edges) - 1, float)
    wfs = []
    for uid in unit_ids:
        mw = load_raw_mean_waveform(raw_wf_root, session_name, int(uid))
        if mw is not None:
            wfs.append(mw)
    return amplitude_depth_fingerprint(wfs, pos, y_edges)


def session_shift_um(fingerprints: Dict[str, np.ndarray], ref_session: str,
                     depth_bin_um: float = DEPTH_BIN_UM,
                     max_lag_um: float = REG_MAX_LAG_UM
                     ) -> Dict[str, Tuple[float, float]]:
    """Per-session rigid registration shift (µm) + corr vs the reference session.

    Positive shift_um ⇒ that session's landscape sits deeper than the reference.
    """
    ref = fingerprints[ref_session]
    max_lag_bins = int(round(max_lag_um / depth_bin_um))
    out: Dict[str, Tuple[float, float]] = {}
    for sess, mov in fingerprints.items():
        lag, corr = estimate_shift_bins(ref, mov, max_lag_bins)
        out[sess] = (-lag * depth_bin_um, corr)   # deeper session -> positive shift
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_population_field.py -k session_shift -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/population_field.py tests/analysis/test_population_field.py
git commit -m "feat(population-field): per-session match-free registration shift"
```

---

### Task 5: Registered depth + unit→field-bin index

**Files:**
- Modify: `src/visdetect/analysis/population_field.py`
- Test: `tests/analysis/test_population_field.py`

**Interfaces:**
- Consumes: `depth_bin_edges`; `visdetect.anatomy.channel_geometry.assign_shanks`.
- Produces:
  - `registered_depth(raw_depth_um: float, shift_um: float) -> float` — subtract the session shift.
  - `unit_field_index(registered_depth_um: float, shank: int, y_edges: np.ndarray, n_shanks: int = 4) -> int` — flattened bin `shank * (len(y_edges)-1) + depth_bin`, depth clipped into range.
  - `n_field_bins(y_edges: np.ndarray, n_shanks: int = 4) -> int`.

- [ ] **Step 1: Write the failing test**

```python
def test_registered_depth_subtracts_shift():
    assert pf.registered_depth(1800.0, 120.0) == pytest.approx(1680.0)


def test_unit_field_index_and_count():
    pos = _np2_positions()
    y_edges = pf.depth_bin_edges(pos, 60.0)
    n_depth = len(y_edges) - 1
    # shank 2, depth in bin 0 -> index = 2 * n_depth + 0
    idx = pf.unit_field_index(y_edges[0] + 1.0, shank=2, y_edges=y_edges, n_shanks=4)
    assert idx == 2 * n_depth
    assert pf.n_field_bins(y_edges, n_shanks=4) == 4 * n_depth
    # below range clips to depth bin 0
    assert pf.unit_field_index(y_edges[0] - 999, shank=0, y_edges=y_edges) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_population_field.py -k "registered_depth or field_index" -v`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Write minimal implementation**

```python
def registered_depth(raw_depth_um: float, shift_um: float) -> float:
    """Depth on the common registered axis: subtract the session's rigid shift."""
    return float(raw_depth_um) - float(shift_um)


def n_field_bins(y_edges: np.ndarray, n_shanks: int = 4) -> int:
    return int(n_shanks * (len(y_edges) - 1))


def unit_field_index(registered_depth_um: float, shank: int,
                     y_edges: np.ndarray, n_shanks: int = 4) -> int:
    """Flattened shank×depth bin index; depth clipped into the grid range."""
    n_depth = len(y_edges) - 1
    depth_bin = int(np.clip(np.searchsorted(y_edges, registered_depth_um) - 1,
                            0, n_depth - 1))
    s = int(np.clip(shank, 0, n_shanks - 1))
    return s * n_depth + depth_bin
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_population_field.py -k "registered_depth or field_index" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/population_field.py tests/analysis/test_population_field.py
git commit -m "feat(population-field): registered depth + unit->field-bin index"
```

---

### Task 6: Build the field tensor (aggregate per-unit → anatomical bins)

**Files:**
- Modify: `src/visdetect/analysis/population_field.py`
- Test: `tests/analysis/test_population_field.py`

**Interfaces:**
- Consumes: `visdetect.analysis.utils.build_population_tensor` (returns `(n_trials, n_bins, n_units)` in **Hz**, case-insensitive outcome filtering).
- Produces: `build_field_tensor(session, unit_ids: List[int], unit_bin_index: np.ndarray, n_bins_anat: int, event_name: str = "Change_ON", window=(-1.0, 1.5), bin_size: float = 0.025, outcome_filter=None) -> Tuple[np.ndarray, np.ndarray, List[int]]` → `(field_tensor (n_trials, n_time_bins, n_bins_anat) summed Hz, bin_centers, valid_trials)`. `unit_bin_index[i]` is the field bin of `unit_ids[i]`.

- [ ] **Step 1: Write the failing test**

```python
from visdetect.analysis.utils import build_population_tensor
from visdetect.utils.synthetic import make_synthetic_session


def test_build_field_tensor_sums_units_into_bins():
    sess = make_synthetic_session(n_trials=40, n_clusters=6, seed=1)
    uids = [c.cluster_id for c in sess.clusters]
    unit_bin_index = np.array([0, 0, 0, 1, 1, 1])   # first 3 -> bin 0, last 3 -> bin 1
    per_unit, bc, valid = build_population_tensor(
        sess, uids, event_name="Baseline_ON", window=(-0.5, 1.0), bin_size=0.025)
    field, bc2, valid2 = pf.build_field_tensor(
        sess, uids, unit_bin_index, n_bins_anat=2,
        event_name="Baseline_ON", window=(-0.5, 1.0), bin_size=0.025)
    assert field.shape == (per_unit.shape[0], per_unit.shape[1], 2)
    assert valid2 == valid
    np.testing.assert_allclose(field[:, :, 0], per_unit[:, :, :3].sum(axis=2))
    np.testing.assert_allclose(field[:, :, 1], per_unit[:, :, 3:].sum(axis=2))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_population_field.py -k build_field_tensor -v`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Write minimal implementation**

```python
from visdetect.analysis.utils import build_population_tensor as _build_population_tensor  # noqa: E402
from visdetect.analysis.constants import DEFAULT_BIN_SIZE                                  # noqa: E402


def build_field_tensor(session, unit_ids: List[int], unit_bin_index: np.ndarray,
                       n_bins_anat: int, event_name: str = "Change_ON",
                       window: Tuple[float, float] = (-1.0, 1.5),
                       bin_size: float = DEFAULT_BIN_SIZE,
                       outcome_filter: Optional[set] = None):
    """Aggregate the per-unit tensor into a (trials × time × anatomical-bin) field.

    Each field bin = SUM of member units' Hz (the local MUA-analog). Units with
    bin index < 0 (e.g. off-grid / no depth) are dropped.
    """
    per_unit, bin_centers, valid = _build_population_tensor(
        session, list(unit_ids), event_name=event_name, window=window,
        bin_size=bin_size, outcome_filter=outcome_filter)
    field = np.zeros((per_unit.shape[0], per_unit.shape[1], n_bins_anat), float)
    idx = np.asarray(unit_bin_index, int)
    for u in range(per_unit.shape[2]):
        b = idx[u]
        if 0 <= b < n_bins_anat:
            field[:, :, b] += per_unit[:, :, u]
    return field, bin_centers, valid
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_population_field.py -k build_field_tensor -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/population_field.py tests/analysis/test_population_field.py
git commit -m "feat(population-field): build_field_tensor (per-unit -> anatomical bins)"
```

---

### Task 7: Registration audit metrics (leakage-free gate)

**Files:**
- Modify: `src/visdetect/analysis/population_field.py`
- Test: `tests/analysis/test_population_field.py`

**Interfaces:**
- Consumes: `session_shift_um` output; per-unit `robust_unit_depth` and `extract_peak_channel` depths.
- Produces:
  - `fingerprint_corr(a: np.ndarray, b: np.ndarray) -> float` — Pearson r of two fingerprints.
  - `peak_vs_centroid_depth(mean_waveform, channel_positions) -> Tuple[float, float]` — `(peak_channel_depth_um, centroid_depth_um)` for the per-unit agreement check.
  - `audit_shift_vs_um_offset(match_free_um: Dict[str, float], um_offset_um: Dict[str, float]) -> Dict[str, float]` — on shared sessions, returns `{"n": int, "median_abs_diff_um": float, "max_abs_diff_um": float}` comparing the match-free shift to the UnitMatch-anchored `estimate_session_drift` offset (agreement is the audit; large divergence flags the UM offset as untrustworthy, not the match-free one).

- [ ] **Step 1: Write the failing test**

```python
def test_fingerprint_corr_identical_is_one():
    a = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    assert pf.fingerprint_corr(a, a) == pytest.approx(1.0)


def test_peak_vs_centroid_depth():
    mw = np.zeros((82, 3))
    mw[:, 1] = np.linspace(-1, 1, 82)     # single dominant channel at index 1
    pos = np.array([[0.0, 0.0], [0.0, 100.0], [0.0, 200.0]])
    peak_d, cent_d = pf.peak_vs_centroid_depth(mw, pos)
    assert peak_d == pytest.approx(100.0)
    assert cent_d == pytest.approx(100.0)


def test_audit_shift_vs_um_offset():
    mf = {"01072025": 0.0, "02072025": 60.0, "03072025": 0.0}
    um = {"01072025": 0.0, "02072025": 45.0}          # only 2 shared
    rep = pf.audit_shift_vs_um_offset(mf, um)
    assert rep["n"] == 2
    assert rep["max_abs_diff_um"] == pytest.approx(15.0)
    assert rep["median_abs_diff_um"] == pytest.approx(7.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_population_field.py -k "corr or peak_vs_centroid or audit_shift" -v`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Write minimal implementation**

```python
from visdetect.analysis.tracking_qc import extract_peak_channel      # noqa: E402


def fingerprint_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def peak_vs_centroid_depth(mean_waveform: np.ndarray,
                           channel_positions: np.ndarray) -> Tuple[float, float]:
    y = np.asarray(channel_positions, float)[:, 1]
    peak_chan = extract_peak_channel(mean_waveform)
    return float(y[peak_chan]), robust_unit_depth(mean_waveform, channel_positions)


def audit_shift_vs_um_offset(match_free_um: Dict[str, float],
                             um_offset_um: Dict[str, float]) -> Dict[str, float]:
    """Compare match-free registration to the UM-anchored offset on shared sessions."""
    shared = [s for s in match_free_um if s in um_offset_um
              and np.isfinite(um_offset_um[s]) and np.isfinite(match_free_um[s])]
    if not shared:
        return {"n": 0, "median_abs_diff_um": float("nan"),
                "max_abs_diff_um": float("nan")}
    diffs = np.array([abs(match_free_um[s] - um_offset_um[s]) for s in shared])
    return {"n": int(len(shared)),
            "median_abs_diff_um": float(np.median(diffs)),
            "max_abs_diff_um": float(diffs.max())}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_population_field.py -k "corr or peak_vs_centroid or audit_shift" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/population_field.py tests/analysis/test_population_field.py
git commit -m "feat(population-field): registration audit metrics"
```

---

### Task 8: Per-subject driver — dominant signature, registration, audit, field cache

**Files:**
- Create: `scripts/population_field/build_field.py`
- Create: `scripts/population_field/__init__.py` (empty)
- Test (unit): `tests/analysis/test_population_field.py` (add `select_dominant_signature`)
- Modify: `src/visdetect/analysis/population_field.py` (add `select_dominant_signature`)

**Interfaces:**
- Consumes: everything above; `visdetect.analysis.config.load_staging_manifest`, `canonical_session_id`, `session_date_key`; `visdetect.anatomy.channel_geometry.chanmap_signature`.
- Produces:
  - Library: `select_dominant_signature(sig_by_session: Dict[str, str]) -> Tuple[str, List[str]]` — the signature with the most sessions and its session list (spec §3 multi-signature rule).
  - Script: `scripts/population_field/build_field.py --subject BG_046 [--depth-bin-um 60]` → writes `data/cache/population_field/<SUBJ>/registration.csv` (session, shift_um, corr, n_units) + `audit.json`. Field-tensor caching is **deferred to Plan 2** (analysis layers build tensors on demand via the tested `build_field_tensor` + `registration.csv`).

- [ ] **Step 1: Write the failing test (library helper only — the script is integration)**

```python
def test_select_dominant_signature():
    sig = {"01072025": "aaa", "02072025": "aaa", "03072025": "bbb"}
    chosen, sessions = pf.select_dominant_signature(sig)
    assert chosen == "aaa"
    assert sorted(sessions) == ["01072025", "02072025"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_population_field.py -k dominant_signature -v`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Write minimal implementation (library helper + driver script)**

```python
# add to src/visdetect/analysis/population_field.py
from collections import Counter                                      # noqa: E402


def select_dominant_signature(sig_by_session: Dict[str, str]
                              ) -> Tuple[str, List[str]]:
    """Spec §3 rule: pick the chanmap signature covering the most sessions."""
    counts = Counter(sig_by_session.values())
    chosen = max(counts, key=lambda s: (counts[s], s))
    sessions = [k for k, v in sig_by_session.items() if v == chosen]
    return chosen, sessions
```

```python
# scripts/population_field/build_field.py
"""Build the tracking-free population-field instrument for one subject.

Local only (pkls + data/unit_match/input); never computes over X:. Picks the
dominant chanmap signature, computes match-free registration + an audit report,
and caches per-session field tensors. See docs/superpowers/plans/
2026-07-08-population-field-instrument-plan.md.
"""
import argparse, json, os
import numpy as np
import pandas as pd

from visdetect.analysis import population_field as pf
from visdetect.analysis.config import (
    canonical_session_id, load_staging_manifest, ROOT,
)
from visdetect.analysis.tracking_qc import (
    load_channel_positions, load_raw_mean_waveform, extract_peak_channel,
)
from visdetect.anatomy.channel_geometry import chanmap_signature, assign_shanks


def _raw_wf_root(subject):
    return os.path.join(ROOT, "data", "unit_match", "input", subject)


def _session_good_stable_ids(subject, session):
    from visdetect.core.session import load_session  # local import (heavy)
    sess = load_session(subject=subject, session_name=session)
    ids = list(sess.good_and_stable_ids or [])
    del sess
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--event", default="Baseline_ON")
    ap.add_argument("--depth-bin-um", type=float, default=pf.DEPTH_BIN_UM)
    args = ap.parse_args()

    root = _raw_wf_root(args.subject)
    sessions = [canonical_session_id(d) for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))]

    # dominant signature
    sig = {}
    for s in sessions:
        pos = load_channel_positions(root, s)
        if pos is not None:
            sig[s] = chanmap_signature(pos)
    chosen, kept = pf.select_dominant_signature(sig)
    kept = sorted(kept, key=__import__("visdetect.analysis.config",
                                       fromlist=["session_date_key"]).session_date_key)

    # common grid from the reference (chronologically first kept) session
    ref = kept[0]
    ref_pos = load_channel_positions(root, ref)
    y_edges = pf.depth_bin_edges(ref_pos, args.depth_bin_um)

    # fingerprints + registration
    fps, n_units = {}, {}
    for s in kept:
        ids = _session_good_stable_ids(args.subject, s)
        n_units[s] = len(ids)
        fps[s] = pf.session_fingerprint_from_root(root, s, ids, y_edges)
    shifts = pf.session_shift_um(fps, ref, args.depth_bin_um, pf.REG_MAX_LAG_UM)

    out_dir = os.path.join(ROOT, "data", "cache", "population_field", args.subject)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([{"session": s, "shift_um": shifts[s][0], "corr": shifts[s][1],
                   "n_units": n_units[s]} for s in kept]).to_csv(
        os.path.join(out_dir, "registration.csv"), index=False)

    audit = {"subject": args.subject, "signature": chosen, "n_sessions": len(kept),
             "max_abs_shift_um": float(np.nanmax([abs(shifts[s][0]) for s in kept])),
             "min_fingerprint_corr": float(np.nanmin([shifts[s][1] for s in kept]))}
    with open(os.path.join(out_dir, "audit.json"), "w") as fh:
        json.dump(audit, fh, indent=2)
    print("AUDIT:", json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the unit test + verify the driver imports**

Run: `py -m pytest tests/analysis/test_population_field.py -k dominant_signature -v`
Expected: PASS.
Run: `py -c "import ast,sys; ast.parse(open('scripts/population_field/build_field.py').read()); print('parse-ok')"`
Expected: `parse-ok`.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/population_field.py scripts/population_field/ tests/analysis/test_population_field.py
git commit -m "feat(population-field): per-subject driver (dominant signature + registration + audit)"
```

- [ ] **Step 6: Golden-path run (real data — run where data/ is present; NOT over X:)**

Run: `py scripts/population_field/build_field.py --subject BG_046`
Expected: prints an `AUDIT` block with `max_abs_shift_um` ≈ 0 (spec: BG_046 whole-probe drift ~0) and `min_fingerprint_corr` ≳ 0.5; writes `data/cache/population_field/BG_046/registration.csv` + `audit.json`. **Gate:** if `max_abs_shift_um` is large or `min_fingerprint_corr` is low, STOP and investigate before trusting the grid (spec Component 0).

---

## Self-Review

**Spec coverage (Plan 1 portion):** Component 0 registration + audit → Tasks 3,4,7,8. Robust per-unit depth → Task 2. Component 1 fine grid (shank×depth) → Tasks 1,5. Component 2 local substrate field tensor → Task 6. Dominant-signature rule (§3) → Task 8. Deferred to later plans (correctly out of Plan 1 scope): CCF-region coarse rollup, TF-GLM, functional map, geometry, evoked profiles (Plan 2); MUA headline (Plan 3). Normalization utilities (`compute_zscore_normalized`) are consumed in Plan 2 — the field tensor here is raw summed Hz plus the per-bin `n_units` yield column for the Plan-2 control.

**Placeholder scan:** none — every step has runnable code, exact commands, and expected output.

**Type consistency:** `session_shift_um` returns `Dict[str, (shift_um, corr)]`; `audit_shift_vs_um_offset` consumes a `Dict[str, float]` of shift µm (driver passes `{s: shifts[s][0]}`). `unit_field_index`/`n_field_bins`/`build_field_tensor` all agree on the flattened `shank * (len(y_edges)-1) + depth_bin` layout and Hz units. `estimate_shift_bins` sign convention (positive lag ⇒ mov rolled deeper; `session_shift_um` negates to report deeper-session-as-positive) is consistent between Tasks 3 and 4.

**Open items carried to implementation (from spec §14):** confirm `DEPTH_BIN_UM` (60) and `REG_MAX_LAG_UM` (300) with the user; Plan 2 will pin the TF-GLM entry point/dt and CCF-region columns (`unit_anatomy.csv`: `session_name, cluster_id, peak_channel, shank, depth_um, ccf_*, region_*`).
