# Population Field — Layer 0 + Layer 1 (Plan 2a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the field-tensor cache and the per-bin functional map (the PRIMARY deliverable) on BG_046, measuring how change / TF / choice encoding reshapes across learning on fixed anatomical bins — with a motor-confound contrast ladder so a motor reshaping can never masquerade as a sensory one.

**Architecture:** Plan 1 already delivers match-free registration (`registration.csv`, all `shift_um` = 0) and `build_field_tensor`. This plan adds three focused library modules — `field_bins.py` (unit→bin assignment + caching), `field_encoding.py` (per-bin contrasts + descriptors), `field_stats.py` (learning model + bespoke nulls) — plus drivers under `scripts/population_field/`. Everything operates on `(trials × time × shank-depth bin)` tensors; no single-unit tracking anywhere.

**Tech Stack:** Python 3 (`.venv`, invoke via `py`), numpy, pandas, scipy, statsmodels, matplotlib, pytest. Reuses `visdetect.analysis.{population_field,utils,config,constants,lick_channels}` and `visdetect.anatomy.channel_geometry`.

**Scope note:** This is Plan **2a**. The spec (`docs/superpowers/specs/2026-07-30-population-field-analysis-layers-design.md`) gates Layers 2–3 on Layer 1 passing its validation gates (§13). Layers 2 (geometry) and 3 (evoked profiles) get their own plan **2b**, written after Layer 1 validates — because their design may legitimately change based on what Layer 1 shows.

---

## Global Constraints

Every task's requirements implicitly include these. Values copied verbatim from the spec.

- **Subject scope: BG_046 ONLY.** Learning→Expert primary via `load_staging_manifest(qc_only=True)`; the 3 Naive sessions are a flagged exploratory arm only.
- **Match-free registration ONLY.** Never use `peak_depth_corrected_um` (UnitMatch-anchored ⇒ circular). Depth comes from `robust_unit_depth` + `registered_depth(raw, shift_um)` using `registration.csv`.
- **TF ONLY via the cached TF-GLM registry** `data/cache/tf_responsive/bg046_tf_responsive.csv` (`resp_log2`, `c1_r_log2`). NEVER the deprecated single-pulse `tf_pulse` screening. ⚠️ That registry carries a **STALE** banner (predates the lick-channel fix); Layer 1 consumes it as-is and every TF output must carry `tf_registry_stale=True`.
- **Canonical constants only** (`visdetect.analysis.constants`): `DEFAULT_BIN_SIZE=0.025`, `DEFAULT_SIGMA_MS=25.0`, `EVENT_VALID_OUTCOMES`, `EVENT_RESPONSIVENESS_WINDOWS`, `BIG_CHANGE_SIZES`, `SMALL_CHANGE_SIZES`. Grid: `DEPTH_BIN_UM=60.0`, `assign_shanks(gap_um=120.0)`. **Introduce NO new numeric constant without flagging it in the task and to the user.**
- **Dual binning, never cross-applied.** Slow evoked (C1–C4): 25 ms bins + 25 ms sigma. TF (C5): consumed at the registry's native 50 ms grid — never re-binned.
- **Session ids** via `config.canonical_session_id`; chronological order via `config.session_date_key` / `config.chronological_sort` — NEVER raw `sorted()`.
- **Outcome labels in pkls are CAPITALIZED** (`'Miss'`,`'Hit'`,`'FA'`). `build_population_tensor` lowercases internally; `get_event_times`' behavioural branch is CASE-SENSITIVE (pass `'FA'`/`'Hit'`).
- **Lick times** via `visdetect.analysis.lick_channels` ONLY (`resolve_lick_channel` / `get_lick_times`). Never read `Piezo_*`/`Lick_*` by name. Use licks for a **binary did-lick screen only**, never counts/rates — the two extraction conventions differ ~6–16× in detection density and alias with the learning timeline.
- **FR-normalize every cross-bin magnitude.** AUROC is rank-based (inherently normalized); `compute_zscore_normalized` for anything in Hz.
- **Yield control:** every per-bin statistic regresses out that bin's contributing-unit count (`bin_yield`).
- **Nulls:** `utils.permutation_test` tests a DIFFERENCE OF MEANS and **cannot** null an AUROC or a slope. Write bespoke shuffles that recompute the actual statistic.
- **pingouin is NOT installed.** Partial correlation = OLS residualization (`statsmodels.formula.api.ols`) then `scipy.stats.spearmanr`.
- **`fdr_correct` returns a BOOLEAN mask**, not q-values. Use `statsmodels.stats.multitest.multipletests` where q-values are reported.
- **Windows:** `py` not `python`. **Worktree:** `export PYTHONPATH=<worktree>/src` or you silently test main's code.
- **NEVER compute over X:** (Samba). This plan is fully local.
- Memory: `del sess; gc.collect()` after every session in a loop.

---

### Task 1: `trial_indices` passthrough on `build_field_tensor`

The lick-free sensory contrast needs "outcome == miss AND change_size ≈ 1.0". `build_population_tensor` supports that via `outcome_filter` + `trial_indices` (AND-combined), but `build_field_tensor` does not currently forward `trial_indices` — so the CR group cannot be built without this.

**Files:**
- Modify: `src/visdetect/analysis/population_field.py:221`
- Test: `tests/analysis/test_population_field.py`

**Interfaces:**
- Produces: `build_field_tensor(session, unit_ids, unit_bin_index, n_bins_anat, event_name="Change_ON", window=(-1.0,1.5), bin_size=DEFAULT_BIN_SIZE, outcome_filter=None, trial_indices=None)` — `trial_indices` is a `List[int]` of raw trial indices, AND-combined with `outcome_filter`.

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_population_field.py
def test_build_field_tensor_honours_trial_indices():
    """The CR group = outcome 'miss' AND catch trials; the caller supplies the
    catch trial indices because build_population_tensor has no change_size
    awareness."""
    from visdetect.utils.synthetic import make_synthetic_session
    sess = make_synthetic_session(n_trials=40, n_clusters=4, seed=3)
    uids = [c.cluster_id for c in sess.clusters]
    idx = np.array([0, 0, 1, 1])
    subset = [0, 1, 2, 3, 4]
    field, bc, valid = pf.build_field_tensor(
        sess, uids, idx, n_bins_anat=2, event_name="Baseline_ON",
        window=(-0.5, 1.0), bin_size=0.025, trial_indices=subset)
    assert set(valid).issubset(set(subset))
    assert field.shape[0] == len(valid)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_population_field.py -k trial_indices -v`
Expected: FAIL — `TypeError: build_field_tensor() got an unexpected keyword argument 'trial_indices'`.

- [ ] **Step 3: Write minimal implementation**

```python
def build_field_tensor(session, unit_ids: List[int], unit_bin_index: np.ndarray,
                       n_bins_anat: int, event_name: str = "Change_ON",
                       window: Tuple[float, float] = (-1.0, 1.5),
                       bin_size: float = DEFAULT_BIN_SIZE,
                       outcome_filter: Optional[set] = None,
                       trial_indices: Optional[List[int]] = None):
    per_unit, bin_centers, valid = _build_population_tensor(
        session, list(unit_ids), event_name=event_name, window=window,
        bin_size=bin_size, outcome_filter=outcome_filter,
        trial_indices=trial_indices)
    # ... rest unchanged ...
```

Also extend the docstring: `trial_indices` is AND-combined with `outcome_filter`; the caller computes it (this function has no `change_size` awareness).

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_population_field.py -q`
Expected: PASS (all pre-existing population_field tests still green).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/population_field.py tests/analysis/test_population_field.py
git commit -m "feat(population-field): trial_indices passthrough on build_field_tensor"
```

---

### Task 2: Unit→field-bin assignment (`field_bins.py`)

**Files:**
- Create: `src/visdetect/analysis/field_bins.py`
- Test: `tests/analysis/test_field_bins.py`

**Interfaces:**
- Consumes: `population_field.{depth_bin_edges, robust_unit_depth, registered_depth, unit_field_index, n_field_bins}`, `anatomy.channel_geometry.assign_shanks`, `analysis.tracking_qc.{load_channel_positions, load_raw_mean_waveform}`.
- Produces:
  - `FieldBinAssignment` dataclass: `unit_ids: List[int]`, `unit_bin_index: np.ndarray`, `n_bins_anat: int`, `y_edges: np.ndarray`, `bin_yield: np.ndarray`, `n_offgrid: int`.
  - `assign_units_to_bins(raw_wf_root, session_id, unit_ids, shift_um, y_edges, n_shanks=4) -> FieldBinAssignment`
  - `bin_depth_um(y_edges, n_shanks) -> np.ndarray` — the depth (µm, bin centre) of each flattened field bin, needed by the descriptors.

Off-grid / undetermined-depth units get index **−1** (never NaN — an int-cast NaN is garbage).

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_field_bins.py
import numpy as np
import pytest
from visdetect.analysis import field_bins as fb
from visdetect.analysis import population_field as pf


def _positions():
    xs = [0, 32, 250, 282, 500, 532, 750, 782]
    ys = np.arange(1500, 2205, 15.0)
    return np.array([[x, y] for y in ys for x in xs], dtype=float)


def test_bin_depth_um_is_bin_centre_per_shank():
    y_edges = pf.depth_bin_edges(_positions(), 60.0)
    depths = fb.bin_depth_um(y_edges, n_shanks=4)
    n_depth = len(y_edges) - 1
    assert depths.shape == (4 * n_depth,)
    expected_first = 0.5 * (y_edges[0] + y_edges[1])
    assert depths[0] == pytest.approx(expected_first)
    # same depth axis repeats for every shank
    np.testing.assert_allclose(depths[:n_depth], depths[n_depth:2 * n_depth])


def test_bin_yield_counts_units_per_bin():
    y_edges = pf.depth_bin_edges(_positions(), 60.0)
    n_bins = pf.n_field_bins(y_edges, 4)
    idx = np.array([0, 0, 5, -1])           # 2 units in bin 0, 1 in bin 5, 1 off-grid
    counts = fb.bin_yield_from_index(idx, n_bins)
    assert counts[0] == 2
    assert counts[5] == 1
    assert counts.sum() == 3                # the -1 unit is excluded
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_field_bins.py -v`
Expected: FAIL — `ImportError: cannot import name 'field_bins'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/visdetect/analysis/field_bins.py
"""Assign units to fixed shank x depth field bins on the registered axis.

Cross-session correspondence comes from fixed anatomy on a MATCH-FREE registered
depth axis, never from unit tracking. See docs/superpowers/specs/
2026-07-30-population-field-analysis-layers-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from visdetect.analysis.population_field import (
    depth_bin_edges, n_field_bins, registered_depth, robust_unit_depth,
    unit_field_index,
)
from visdetect.anatomy.channel_geometry import assign_shanks
from visdetect.analysis.tracking_qc import (
    load_channel_positions, load_raw_mean_waveform,
)

OFF_GRID = -1


@dataclass
class FieldBinAssignment:
    unit_ids: List[int]
    unit_bin_index: np.ndarray
    n_bins_anat: int
    y_edges: np.ndarray
    bin_yield: np.ndarray
    n_offgrid: int


def bin_depth_um(y_edges: np.ndarray, n_shanks: int = 4) -> np.ndarray:
    """Depth (bin centre, um) of each flattened shank x depth bin."""
    centres = 0.5 * (np.asarray(y_edges, float)[:-1] + np.asarray(y_edges, float)[1:])
    return np.tile(centres, n_shanks)


def bin_yield_from_index(unit_bin_index: np.ndarray, n_bins_anat: int) -> np.ndarray:
    """Contributing-unit count per field bin (the yield covariate)."""
    idx = np.asarray(unit_bin_index, int)
    counts = np.zeros(n_bins_anat, int)
    valid = idx[(idx >= 0) & (idx < n_bins_anat)]
    np.add.at(counts, valid, 1)
    return counts


def assign_units_to_bins(raw_wf_root, session_id: str, unit_ids: List[int],
                         shift_um: float, y_edges: np.ndarray,
                         n_shanks: int = 4) -> FieldBinAssignment:
    """Map each unit to a flattened shank x depth bin on the registered axis."""
    pos = load_channel_positions(raw_wf_root, session_id)
    n_bins = n_field_bins(y_edges, n_shanks)
    if pos is None:
        idx = np.full(len(unit_ids), OFF_GRID, int)
        return FieldBinAssignment(list(unit_ids), idx, n_bins, y_edges,
                                  bin_yield_from_index(idx, n_bins), len(unit_ids))
    shank_of_chan = assign_shanks(pos, n_shanks=n_shanks)
    idx = np.full(len(unit_ids), OFF_GRID, int)
    for i, uid in enumerate(unit_ids):
        mw = load_raw_mean_waveform(raw_wf_root, session_id, int(uid))
        if mw is None:
            continue
        depth = robust_unit_depth(mw, pos)
        if not np.isfinite(depth):
            continue
        ptp = mw.max(axis=0) - mw.min(axis=0)
        shank = int(shank_of_chan[int(np.argmax(ptp))])
        idx[i] = unit_field_index(registered_depth(depth, shift_um), shank,
                                  y_edges, n_shanks=n_shanks)
    return FieldBinAssignment(list(unit_ids), idx, n_bins, y_edges,
                              bin_yield_from_index(idx, n_bins),
                              int(np.sum(idx == OFF_GRID)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_field_bins.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/field_bins.py tests/analysis/test_field_bins.py
git commit -m "feat(population-field): unit->field-bin assignment + yield covariate"
```

---

### Task 3: Trial-group selection for the contrast ladder

**Files:**
- Modify: `src/visdetect/analysis/field_bins.py`
- Test: `tests/analysis/test_field_bins.py`

**Interfaces:**
- Produces:
  - `catch_trial_indices(session, tol=0.01) -> List[int]` — trials with `abs(change_size - 1.0) <= tol`.
  - `go_trial_indices(session) -> List[int]` — trials with `change_size > 1.0`.
  - `lick_free_trial_indices(session, indices, window, event_times) -> List[int]` — drops any trial with a lick inside `window` of its event, using the canonical resolver. **Binary screen only.**

`tol=0.01` is NOT a new scientific constant — it is float-equality slack on the documented `change_size ≈ 1.0` catch definition.

- [ ] **Step 1: Write the failing test**

```python
def test_catch_and_go_trial_indices():
    class _T:
        def __init__(self, cs, oc): self.change_size, self.trialoutcome = cs, oc
    class _S:
        pass
    s = _S()
    s.trials = [_T(1.0, "Miss"), _T(2.0, "Hit"), _T(1.0, "Hit"), _T(1.25, "Miss")]
    s.ni_events = {}
    assert fb.catch_trial_indices(s) == [0, 2]
    assert fb.go_trial_indices(s) == [1, 3]


def test_lick_free_screen_drops_trials_with_a_lick_in_window():
    class _S:
        pass
    s = _S()
    s.ni_events = {"Lick_L": np.array([10.2, 55.0])}   # lick at 10.2 s
    s.session_name = "TEST"
    kept = fb.lick_free_trial_indices(
        s, indices=[0, 1], window=(0.0, 0.5), event_times=[10.0, 30.0])
    assert kept == [1]          # trial 0 has a lick 0.2 s after its event
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_field_bins.py -k "trial_indices or lick_free" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'catch_trial_indices'`.

- [ ] **Step 3: Write minimal implementation**

```python
def _change_size(trial) -> float:
    return float(getattr(trial, "change_size", np.nan))


def catch_trial_indices(session, tol: float = 0.01) -> List[int]:
    """Catch trials: change_size ~= 1.0 (no real TF change)."""
    return [i for i, t in enumerate(getattr(session, "trials", []) or [])
            if np.isfinite(_change_size(t)) and abs(_change_size(t) - 1.0) <= tol]


def go_trial_indices(session, tol: float = 0.01) -> List[int]:
    """Go trials: change_size > 1.0 (the TF really changed)."""
    return [i for i, t in enumerate(getattr(session, "trials", []) or [])
            if np.isfinite(_change_size(t)) and _change_size(t) > 1.0 + tol]


def lick_free_trial_indices(session, indices, window, event_times) -> List[int]:
    """Drop trials with ANY lick inside ``window`` of their event.

    BINARY screen only -- never counts or rates. The two NI extraction
    conventions differ ~6-16x in detection density, but a LESS sensitive channel
    cannot manufacture licks, so 'no lick detected' is safe in both.
    """
    from visdetect.analysis.lick_channels import (
        NoLickChannelError, get_lick_times,
    )
    try:
        licks = get_lick_times(session)
    except NoLickChannelError:
        return list(indices)           # cannot screen; keep and let the audit flag it
    lo, hi = window
    kept = []
    for i, t0 in zip(indices, event_times):
        if not np.isfinite(t0):
            continue
        if not np.any((licks >= t0 + lo) & (licks <= t0 + hi)):
            kept.append(i)
    return kept
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_field_bins.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/field_bins.py tests/analysis/test_field_bins.py
git commit -m "feat(population-field): contrast-ladder trial selection + binary lick screen"
```

---

### Task 4: Per-bin encoding strength (`field_encoding.py`)

**Files:**
- Create: `src/visdetect/analysis/field_encoding.py`
- Test: `tests/analysis/test_field_encoding.py`

**Interfaces:**
- Consumes: `utils.compute_auroc`.
- Produces:
  - `bin_response(field, bin_centers, resp_window, base_window) -> np.ndarray` — `(n_trials, n_bins_anat)` baseline-subtracted mean response per trial per bin.
  - `per_bin_auroc(resp_a, resp_b, match_trials=True, seed=42) -> np.ndarray` — signed AUROC per bin.
  - `encoding_strength(auroc) -> np.ndarray` — `|AUROC − 0.5|`.

Trial-count matching subsamples the larger group (fixed seed) — unequal n biases variance estimates.

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_field_encoding.py
import numpy as np
import pytest
from visdetect.analysis import field_encoding as fe


def test_bin_response_is_baseline_subtracted():
    bin_centers = np.arange(-1.0, 1.5, 0.025) + 0.0125
    field = np.zeros((5, len(bin_centers), 2))
    field[:, :, 0] = 10.0                                  # flat 10 Hz -> 0 after subtraction
    post = (bin_centers >= 0.0) & (bin_centers <= 0.25)
    field[:, post, 1] = 4.0                                # bin 1 responds
    resp = fe.bin_response(field, bin_centers, (0.0, 0.25), (-0.4, -0.05))
    assert resp.shape == (5, 2)
    np.testing.assert_allclose(resp[:, 0], 0.0, atol=1e-9)
    np.testing.assert_allclose(resp[:, 1], 4.0, atol=1e-9)


def test_per_bin_auroc_separates_and_is_symmetric():
    rng = np.random.default_rng(0)
    a = np.column_stack([rng.normal(5, 1, 200), rng.normal(0, 1, 200)])
    b = np.column_stack([rng.normal(0, 1, 200), rng.normal(0, 1, 200)])
    auroc = fe.per_bin_auroc(a, b, match_trials=False)
    assert auroc[0] > 0.9            # bin 0 separates
    assert 0.4 < auroc[1] < 0.6      # bin 1 does not
    assert fe.encoding_strength(auroc)[1] < 0.1


def test_trial_matching_equalises_group_sizes():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, (300, 2))
    b = rng.normal(0, 1, (40, 2))
    auroc = fe.per_bin_auroc(a, b, match_trials=True, seed=42)
    assert auroc.shape == (2,)
    # deterministic under a fixed seed
    np.testing.assert_allclose(auroc, fe.per_bin_auroc(a, b, match_trials=True, seed=42))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_field_encoding.py -v`
Expected: FAIL — `ImportError: cannot import name 'field_encoding'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/visdetect/analysis/field_encoding.py
"""Per-bin encoding strength on the fixed anatomical field grid (Layer 1).

AUROC is rank-based and therefore inherently FR-normalized -- cross-bin
comparability comes for free, which is what makes it the right metric here
(cf. the retracted un-normalized-Hz result in tf_transient_sustained_state).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from visdetect.analysis.utils import compute_auroc


def bin_response(field: np.ndarray, bin_centers: np.ndarray,
                 resp_window: Tuple[float, float],
                 base_window: Tuple[float, float]) -> np.ndarray:
    """Per-trial, per-bin baseline-subtracted mean response (Hz)."""
    bc = np.asarray(bin_centers, float)
    resp_mask = (bc >= resp_window[0]) & (bc <= resp_window[1])
    base_mask = (bc >= base_window[0]) & (bc <= base_window[1])
    if not resp_mask.any() or not base_mask.any():
        raise ValueError(f"empty window: resp={resp_window} base={base_window} "
                         f"over bin_centers [{bc[0]:.3f}, {bc[-1]:.3f}]")
    return field[:, resp_mask, :].mean(axis=1) - field[:, base_mask, :].mean(axis=1)


def per_bin_auroc(resp_a: np.ndarray, resp_b: np.ndarray,
                  match_trials: bool = True, seed: int = 42) -> np.ndarray:
    """Signed AUROC per bin for group A vs group B (0.5 = no separation)."""
    a, b = np.asarray(resp_a, float), np.asarray(resp_b, float)
    if match_trials and a.shape[0] != b.shape[0]:
        n = min(a.shape[0], b.shape[0])
        rng = np.random.default_rng(seed)
        a = a[np.sort(rng.choice(a.shape[0], n, replace=False))]
        b = b[np.sort(rng.choice(b.shape[0], n, replace=False))]
    return np.array([compute_auroc(a[:, k], b[:, k]) for k in range(a.shape[1])])


def encoding_strength(auroc: np.ndarray) -> np.ndarray:
    """|AUROC - 0.5|. Sign is reported separately -- never derived from the data
    it is then averaged over (circularity hard rule)."""
    return np.abs(np.asarray(auroc, float) - 0.5)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_field_encoding.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/field_encoding.py tests/analysis/test_field_encoding.py
git commit -m "feat(population-field): per-bin AUROC encoding strength"
```

---

### Task 5: Map descriptors (strengthen / sharpen / relocate)

**Files:**
- Modify: `src/visdetect/analysis/field_encoding.py`
- Test: `tests/analysis/test_field_encoding.py`

**Interfaces:**
- Produces: `map_descriptors(strength, depths_um) -> Dict[str, float]` with keys `total_strength`, `centroid_um`, `spread_um`, `effective_bins`, `n_hotspots`.

`spread_um` (encoding-weighted SD of bin depth) is the sharpening measure in tissue units. `effective_bins` is the participation ratio `(Σx)²/Σx²` — arrangement-agnostic, so it stays valid when the profile is multi-modal and the weighted mean lands in a gap between hotspots. `n_hotspots` says which to trust.

- [ ] **Step 1: Write the failing test**

```python
def test_descriptors_on_a_single_tight_hotspot():
    depths = np.arange(0, 1000, 100.0)          # 10 bins
    strength = np.zeros(10); strength[5] = 1.0
    d = fe.map_descriptors(strength, depths)
    assert d["centroid_um"] == pytest.approx(500.0)
    assert d["spread_um"] == pytest.approx(0.0)
    assert d["effective_bins"] == pytest.approx(1.0)
    assert d["n_hotspots"] == 1


def test_effective_bins_equals_n_when_uniform():
    depths = np.arange(0, 1000, 100.0)
    d = fe.map_descriptors(np.ones(10), depths)
    assert d["effective_bins"] == pytest.approx(10.0)


def test_two_hotspots_are_detected_so_spread_is_not_trusted_blindly():
    """With two peaks the weighted centroid lands in the empty gap between them
    and spread inflates for the wrong reason -- n_hotspots flags it."""
    depths = np.arange(0, 1000, 100.0)
    strength = np.zeros(10); strength[1] = 1.0; strength[8] = 1.0
    d = fe.map_descriptors(strength, depths)
    assert d["n_hotspots"] == 2
    assert d["centroid_um"] == pytest.approx(450.0)     # a depth with NO signal
    assert d["effective_bins"] == pytest.approx(2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_field_encoding.py -k descriptors -v`
Expected: FAIL — `AttributeError: ... has no attribute 'map_descriptors'`.

- [ ] **Step 3: Write minimal implementation**

```python
def map_descriptors(strength: np.ndarray, depths_um: np.ndarray) -> dict:
    """Summarise an across-bin encoding profile.

    total_strength  - how much encoding there is overall
    centroid_um     - where along the probe it sits (encoding-weighted mean depth)
    spread_um       - over how many microns it is smeared (weighted SD) = SHARPENING
    effective_bins  - participation ratio (Sx)^2/Sx^2: how many bins genuinely
                      carry the signal (1 = one hotspot, n = spread evenly)
    n_hotspots      - local maxima above half the peak; if > 1, centroid/spread
                      describe a gap between hotspots and effective_bins is the
                      measure to trust.
    """
    w = np.clip(np.asarray(strength, float), 0.0, None)
    d = np.asarray(depths_um, float)
    total = float(w.sum())
    if total <= 0:
        return {"total_strength": 0.0, "centroid_um": float("nan"),
                "spread_um": float("nan"), "effective_bins": 0.0, "n_hotspots": 0}
    centroid = float((w * d).sum() / total)
    spread = float(np.sqrt((w * (d - centroid) ** 2).sum() / total))
    effective = float(total ** 2 / float((w ** 2).sum()))
    peak = w.max()
    above = w >= 0.5 * peak
    n_hotspots = int(np.sum(above & ~np.r_[False, above[:-1]]))
    return {"total_strength": total, "centroid_um": centroid,
            "spread_um": spread, "effective_bins": effective,
            "n_hotspots": n_hotspots}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_field_encoding.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/field_encoding.py tests/analysis/test_field_encoding.py
git commit -m "feat(population-field): map descriptors (centroid, spread, effective bins)"
```

---

### Task 6: Learning model + bespoke nulls (`field_stats.py`)

**Files:**
- Create: `src/visdetect/analysis/field_stats.py`
- Test: `tests/analysis/test_field_stats.py`

**Interfaces:**
- Produces:
  - `partial_spearman(x, y, covar) -> Tuple[float, float]` — OLS-residualized Spearman (pingouin is absent).
  - `per_bin_learning_slope(values, session_index, bin_yield) -> np.ndarray` — yield-controlled rho per bin. `values` is `(n_sessions, n_bins)`.
  - `session_shuffle_null(values, session_index, bin_yield, n_perm=1000, seed=42) -> np.ndarray` — permutes SESSION ORDER and recomputes the actual statistic; returns `(n_perm, n_bins)`.
  - `null_pvalues(observed, null) -> np.ndarray` — two-sided.

⚠️ `utils.permutation_test` must NOT be used here: it tests a difference of means and cannot null a correlation.

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_field_stats.py
import numpy as np
import pytest
from visdetect.analysis import field_stats as fs


def test_partial_spearman_removes_the_covariate_effect():
    rng = np.random.default_rng(0)
    covar = rng.normal(size=200)
    x = covar + 0.01 * rng.normal(size=200)      # x is essentially the covariate
    y = covar + 0.01 * rng.normal(size=200)      # so is y
    raw_rho, _ = fs.partial_spearman(x, y, covar=np.zeros(200))
    part_rho, _ = fs.partial_spearman(x, y, covar=covar)
    assert raw_rho > 0.9            # correlated before controlling
    assert abs(part_rho) < 0.4      # explained away after controlling


def test_per_bin_learning_slope_finds_the_bin_that_grows():
    n_sess = 20
    sess_idx = np.arange(n_sess)
    values = np.zeros((n_sess, 3))
    values[:, 0] = sess_idx * 0.05                       # grows with learning
    values[:, 1] = np.random.default_rng(2).normal(size=n_sess)
    values[:, 2] = -sess_idx * 0.05                      # declines
    rho = fs.per_bin_learning_slope(values, sess_idx, bin_yield=np.ones((n_sess, 3)))
    assert rho[0] > 0.9
    assert abs(rho[1]) < 0.6
    assert rho[2] < -0.9


def test_session_shuffle_null_is_flat_on_shuffled_data():
    """MANDATORY circularity control: a null must be centred on zero."""
    n_sess = 20
    sess_idx = np.arange(n_sess)
    values = np.random.default_rng(3).normal(size=(n_sess, 4))
    yields = np.ones((n_sess, 4))
    null = fs.session_shuffle_null(values, sess_idx, yields, n_perm=200, seed=42)
    assert null.shape == (200, 4)
    assert abs(float(np.mean(null))) < 0.1          # centred on no effect


def test_null_pvalues_flag_a_real_effect_and_spare_noise():
    n_sess = 20
    sess_idx = np.arange(n_sess)
    values = np.zeros((n_sess, 2))
    values[:, 0] = sess_idx * 0.05
    values[:, 1] = np.random.default_rng(4).normal(size=n_sess)
    yields = np.ones((n_sess, 2))
    obs = fs.per_bin_learning_slope(values, sess_idx, yields)
    null = fs.session_shuffle_null(values, sess_idx, yields, n_perm=500, seed=42)
    p = fs.null_pvalues(obs, null)
    assert p[0] < 0.01
    assert p[1] > 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_field_stats.py -v`
Expected: FAIL — `ImportError: cannot import name 'field_stats'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/visdetect/analysis/field_stats.py
"""Learning-axis statistics for the per-bin functional map.

The replication unit is the SESSION (one value per bin per session) -- never the
trial or the unit, which is where pseudoreplication would enter.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import spearmanr


def _residualize(v: np.ndarray, covar: np.ndarray) -> np.ndarray:
    """Residuals of v after removing a linear fit on covar (OLS via lstsq).

    pingouin is not installed and scipy has no partial correlation, so we
    residualize explicitly.
    """
    v = np.asarray(v, float)
    c = np.asarray(covar, float)
    if np.allclose(c, c[0]):
        return v - v.mean()
    X = np.column_stack([np.ones_like(c), c])
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    return v - X @ beta


def partial_spearman(x, y, covar) -> Tuple[float, float]:
    """Spearman rho of x vs y after residualizing BOTH on covar."""
    rx, ry = _residualize(x, covar), _residualize(y, covar)
    if np.allclose(rx, rx[0]) or np.allclose(ry, ry[0]):
        return 0.0, 1.0
    rho, p = spearmanr(rx, ry)
    return float(rho), float(p)


def per_bin_learning_slope(values: np.ndarray, session_index: np.ndarray,
                           bin_yield: np.ndarray) -> np.ndarray:
    """Yield-controlled partial Spearman of encoding vs session order, per bin.

    values / bin_yield: (n_sessions, n_bins). session_index: (n_sessions,).
    """
    values = np.asarray(values, float)
    out = np.zeros(values.shape[1])
    for k in range(values.shape[1]):
        out[k] = partial_spearman(np.asarray(session_index, float),
                                  values[:, k], np.asarray(bin_yield, float)[:, k])[0]
    return out


def session_shuffle_null(values: np.ndarray, session_index: np.ndarray,
                         bin_yield: np.ndarray, n_perm: int = 1000,
                         seed: int = 42) -> np.ndarray:
    """Null by permuting SESSION ORDER and recomputing the ACTUAL statistic.

    utils.permutation_test cannot be used: it nulls a difference of means, not a
    correlation.
    """
    rng = np.random.default_rng(seed)
    n_sess = np.asarray(values).shape[0]
    null = np.zeros((n_perm, np.asarray(values).shape[1]))
    for i in range(n_perm):
        perm = rng.permutation(n_sess)
        null[i] = per_bin_learning_slope(values, np.asarray(session_index)[perm],
                                         bin_yield)
    return null


def null_pvalues(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Two-sided p from a null distribution, with the +1 correction."""
    obs = np.abs(np.asarray(observed, float))
    nul = np.abs(np.asarray(null, float))
    return (np.sum(nul >= obs[None, :], axis=0) + 1.0) / (nul.shape[0] + 1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_field_stats.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/field_stats.py tests/analysis/test_field_stats.py
git commit -m "feat(population-field): yield-controlled learning slope + bespoke nulls"
```

---

### Task 7: Field-tensor cache driver

**Files:**
- Create: `scripts/population_field/cache_tensors.py`
- Modify: `tests/analysis/test_field_bins.py` (add the manifest-join helper test)
- Modify: `src/visdetect/analysis/field_bins.py` (add `load_registration`)

**Interfaces:**
- Produces:
  - `field_bins.load_registration(subject) -> pd.DataFrame` — reads `registration.csv`, canonicalizes `session`, returns columns `session, shift_um, corr, n_units`.
  - Script: `py scripts/population_field/cache_tensors.py --subject BG_046` → one `.npz` per `(session, contrast)` under `data/cache/population_field/BG_046/tensors/`.

Each npz stores: `field`, `bin_centers`, `valid_trials`, `unit_bin_index`, `bin_yield`, `n_bins_anat`, and provenance (`lick_time_source`, `movement_controlled`, `code_commit`, `contrast`, `session`).

- [ ] **Step 1: Write the failing test**

```python
def test_load_registration_canonicalizes_session_ids(tmp_path):
    import pandas as pd
    d = tmp_path / "BG_046"; d.mkdir()
    pd.DataFrame({"session": [1072025, 23062025], "shift_um": [0.0, 0.0],
                  "corr": [0.9, 0.9], "n_units": [10, 20]}).to_csv(
        d / "registration.csv", index=False)
    reg = fb.load_registration("BG_046", cache_root=str(tmp_path))
    # int64 dropped the leading-zero DAY; canonicalization must restore it
    assert set(reg["session"]) == {"01072025", "23062025"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_field_bins.py -k load_registration -v`
Expected: FAIL — `AttributeError: ... has no attribute 'load_registration'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/visdetect/analysis/field_bins.py
import os
import pandas as pd
from visdetect.analysis.config import ROOT, canonical_session_id


def load_registration(subject: str, cache_root: Optional[str] = None):
    """Per-session registration shifts, with canonical 8-digit session ids.

    registration.csv round-trips 'session' through pandas as int64, which DROPS
    the leading-zero day for days 1-9 -- always canonicalize on read.
    """
    root = cache_root or os.path.join(ROOT, "data", "cache", "population_field")
    df = pd.read_csv(os.path.join(root, subject, "registration.csv"))
    df["session"] = df["session"].map(canonical_session_id)
    return df
```

```python
# scripts/population_field/cache_tensors.py
"""Cache (trials x time x shank-depth bin) field tensors for the contrast ladder.

Local only. Reads audit.json as a GATE, registration.csv for the registered
depth axis, and writes one npz per (session, contrast). See docs/superpowers/
plans/2026-08-03-population-field-layer1-plan.md.
"""
import argparse, gc, json, os, subprocess
import numpy as np

from visdetect.analysis import field_bins as fb
from visdetect.analysis import population_field as pf
from visdetect.analysis.config import ROOT, canonical_session_id, session_date_key
from visdetect.analysis.constants import (
    DEFAULT_BIN_SIZE, EVENT_RESPONSIVENESS_WINDOWS,
)
from visdetect.analysis.tracking_qc import load_channel_positions
from visdetect.core.session import load_session

CONTRASTS = {
    # name        event         outcome_filter  window
    "c1_go_miss": ("Change_ON", {"miss"}, (-1.0, 1.5)),
    "c1_cr":      ("Change_ON", {"miss"}, (-1.0, 1.5)),
    "c4_hit":     ("Change_ON", {"hit"},  (-1.0, 1.5)),
    "c4_miss":    ("Change_ON", {"miss"}, (-1.0, 1.5)),
    "c2_fa":      ("FA",        {"fa"},   (-2.0, 1.0)),
    "c3_hit_lick": ("Hit",      {"hit"},  (-2.0, 1.0)),
}


def _commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="BG_046")
    args = ap.parse_args()

    cache = os.path.join(ROOT, "data", "cache", "population_field", args.subject)
    with open(os.path.join(cache, "audit.json")) as fh:
        audit = json.load(fh)
    if not np.isfinite(audit.get("max_abs_shift_um", np.nan)):
        raise SystemExit("registration audit gate failed; refusing to build tensors")
    print("AUDIT GATE OK:", json.dumps(audit))

    reg = fb.load_registration(args.subject)
    raw_root = os.path.join(ROOT, "data", "unit_match", "input", args.subject)
    out_dir = os.path.join(cache, "tensors")
    os.makedirs(out_dir, exist_ok=True)
    commit = _commit()

    ref_pos = load_channel_positions(raw_root, reg["session"].iloc[0])
    y_edges = pf.depth_bin_edges(ref_pos, pf.DEPTH_BIN_UM)

    for sess_id in sorted(reg["session"], key=session_date_key):
        shift = float(reg.loc[reg["session"] == sess_id, "shift_um"].iloc[0])
        pkl = os.path.join(ROOT, "data", "pkls", args.subject,
                           f"{args.subject}_{sess_id}.pkl")
        if not os.path.exists(pkl):
            print(f"  SKIP {sess_id}: no pkl"); continue
        sess = load_session(pkl)
        uids = list(sess.good_and_stable_ids or [])
        asg = fb.assign_units_to_bins(raw_root, sess_id, uids, shift, y_edges)

        catch = set(fb.catch_trial_indices(sess))
        go = set(fb.go_trial_indices(sess))
        for name, (event, outcomes, window) in CONTRASTS.items():
            ti = None
            if name == "c1_cr":
                ti = sorted(catch)
            elif name in ("c1_go_miss", "c4_miss"):
                ti = sorted(go)
            try:
                field, bc, valid = pf.build_field_tensor(
                    sess, asg.unit_ids, asg.unit_bin_index, asg.n_bins_anat,
                    event_name=event, window=window, bin_size=DEFAULT_BIN_SIZE,
                    outcome_filter=outcomes, trial_indices=ti)
            except ValueError as exc:
                print(f"  {sess_id}/{name}: {exc}"); continue
            np.savez_compressed(
                os.path.join(out_dir, f"{sess_id}__{name}.npz"),
                field=field, bin_centers=bc, valid_trials=np.asarray(valid),
                unit_bin_index=asg.unit_bin_index, bin_yield=asg.bin_yield,
                n_bins_anat=asg.n_bins_anat, session=sess_id, contrast=name,
                lick_time_source="spout_contact_minus200ms",
                movement_controlled=False, code_commit=commit)
            print(f"  cached {sess_id}/{name}: {field.shape}")
        del sess; gc.collect()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the unit test + verify the driver parses**

Run: `py -m pytest tests/analysis/test_field_bins.py -v`
Expected: PASS.
Run: `py -c "import ast; ast.parse(open('scripts/population_field/cache_tensors.py').read()); print('parse-ok')"`
Expected: `parse-ok`.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/field_bins.py scripts/population_field/cache_tensors.py tests/analysis/test_field_bins.py
git commit -m "feat(population-field): field-tensor cache driver with provenance"
```

- [ ] **Step 6: Golden-path run (real data — needs data/, NOT over X:)**

Run: `py scripts/population_field/cache_tensors.py --subject BG_046`
Expected: prints `AUDIT GATE OK`, then per-session/per-contrast tensor shapes; writes npz files under `data/cache/population_field/BG_046/tensors/`.
**Gate:** if `c1_cr` yields fewer than ~15 trials on most sessions, STOP and report — the lick-free sensory contrast is the headline and needs adequate n (spec §3.2 measured ~30–40/session).

---

### Task 8: Build the functional map (`build_map.py`)

**Files:**
- Create: `scripts/population_field/build_map.py`
- Test: `tests/analysis/test_field_encoding.py` (add the ladder-assembly test)
- Modify: `src/visdetect/analysis/field_encoding.py` (add `ladder_row`)

**Interfaces:**
- Produces:
  - `field_encoding.ladder_row(strength, depths_um, session_id, contrast, stage, session_idx) -> dict` — one flat record per (session, contrast) carrying descriptors + provenance.
  - Script: `py scripts/population_field/build_map.py --subject BG_046` → `map_per_bin.csv` (session × bin × contrast strength + yield) and `map_descriptors.csv`.

- [ ] **Step 1: Write the failing test**

```python
def test_ladder_row_is_flat_and_carries_provenance():
    depths = np.arange(0, 500, 100.0)
    strength = np.array([0.0, 0.2, 0.4, 0.1, 0.0])
    row = fe.ladder_row(strength, depths, session_id="01072025",
                        contrast="c1_sensory", stage="Learning", session_idx=3)
    assert row["session"] == "01072025"
    assert row["contrast"] == "c1_sensory"
    assert row["stage"] == "Learning"
    assert row["session_idx"] == 3
    assert row["centroid_um"] == pytest.approx(200.0)
    assert "effective_bins" in row and "n_hotspots" in row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_field_encoding.py -k ladder_row -v`
Expected: FAIL — `AttributeError: ... has no attribute 'ladder_row'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/visdetect/analysis/field_encoding.py
def ladder_row(strength: np.ndarray, depths_um: np.ndarray, session_id: str,
               contrast: str, stage: str, session_idx: int) -> dict:
    """One flat per-(session, contrast) record: descriptors + provenance."""
    row = {"session": str(session_id), "contrast": contrast, "stage": stage,
           "session_idx": int(session_idx)}
    row.update(map_descriptors(strength, depths_um))
    return row
```

```python
# scripts/population_field/build_map.py
"""Layer 1: the per-bin functional map + the motor-confound contrast ladder.

Reads cached field tensors, computes per-bin encoding strength for each rung of
the ladder, fits the yield-controlled learning slope with a session-shuffle
null, and FDR-corrects across bins WITHIN each rung.
"""
import argparse, glob, os
import numpy as np
import pandas as pd

from visdetect.analysis import field_bins as fb
from visdetect.analysis import field_encoding as fe
from visdetect.analysis import field_stats as fs
from visdetect.analysis import population_field as pf
from visdetect.analysis.config import (
    ROOT, canonical_session_id, load_staging_manifest, session_date_key,
)
from visdetect.analysis.constants import EVENT_RESPONSIVENESS_WINDOWS
from visdetect.analysis.utils import fdr_correct

# Ladder rung -> (group A contrast, group B contrast, event key for windows)
LADDER = {
    "c1_sensory_lickfree": ("c1_go_miss", "c1_cr", "Change_ON"),
    "c4_detection_motor":  ("c4_hit", "c4_miss", "Change_ON"),
    "c3_motor_matched":    ("c3_hit_lick", "c2_fa", "FA"),
}


def _load(tensor_dir, session, contrast):
    p = os.path.join(tensor_dir, f"{session}__{contrast}.npz")
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="BG_046")
    args = ap.parse_args()

    cache = os.path.join(ROOT, "data", "cache", "population_field", args.subject)
    tensor_dir = os.path.join(cache, "tensors")

    manifest = load_staging_manifest(qc_only=True)
    manifest["sess_key"] = manifest["session_name"].map(canonical_session_id)
    stage_of = dict(zip(manifest["sess_key"], manifest["stage"]))
    sessions = sorted(stage_of, key=session_date_key)

    reg = fb.load_registration(args.subject)
    from visdetect.analysis.tracking_qc import load_channel_positions
    raw_root = os.path.join(ROOT, "data", "unit_match", "input", args.subject)
    y_edges = pf.depth_bin_edges(load_channel_positions(raw_root, reg["session"].iloc[0]),
                                 pf.DEPTH_BIN_UM)
    depths = fb.bin_depth_um(y_edges, n_shanks=4)

    per_bin_rows, desc_rows = [], []
    for rung, (ca, cb, event) in LADDER.items():
        base_w, resp_w = EVENT_RESPONSIVENESS_WINDOWS[event]
        vals, yields, idxs, used = [], [], [], []
        for si, sess in enumerate(sessions):
            a, b = _load(tensor_dir, sess, ca), _load(tensor_dir, sess, cb)
            if a is None or b is None:
                continue
            ra = fe.bin_response(a["field"], a["bin_centers"], resp_w, base_w)
            rb = fe.bin_response(b["field"], b["bin_centers"], resp_w, base_w)
            if ra.shape[0] < 5 or rb.shape[0] < 5:
                print(f"  {sess}/{rung}: too few trials ({ra.shape[0]}, {rb.shape[0]})")
                continue
            auroc = fe.per_bin_auroc(ra, rb, match_trials=True, seed=42)
            strength = fe.encoding_strength(auroc)
            vals.append(strength); yields.append(a["bin_yield"]); idxs.append(si)
            used.append(sess)
            for k in range(strength.size):
                per_bin_rows.append({"session": sess, "contrast": rung, "bin": k,
                                     "depth_um": depths[k], "auroc": auroc[k],
                                     "strength": strength[k],
                                     "bin_yield": int(a["bin_yield"][k]),
                                     "stage": stage_of[sess]})
            desc_rows.append(fe.ladder_row(strength, depths, sess, rung,
                                           stage_of[sess], si))
        if len(vals) < 5:
            print(f"  {rung}: only {len(vals)} usable sessions; skipping stats")
            continue
        V, Y, I = np.array(vals), np.array(yields, float), np.array(idxs, float)
        rho = fs.per_bin_learning_slope(V, I, Y)
        null = fs.session_shuffle_null(V, I, Y, n_perm=1000, seed=42)
        p = fs.null_pvalues(rho, null)
        sig = fdr_correct(p, alpha=0.05)
        pd.DataFrame({"bin": np.arange(rho.size), "depth_um": depths,
                      "rho": rho, "p": p, "fdr_sig": sig,
                      "n_sessions": len(vals)}).to_csv(
            os.path.join(cache, f"learning_slope__{rung}.csv"), index=False)
        print(f"  {rung}: {len(vals)} sessions, {int(sig.sum())}/{rho.size} bins FDR-sig")

    pd.DataFrame(per_bin_rows).to_csv(os.path.join(cache, "map_per_bin.csv"), index=False)
    pd.DataFrame(desc_rows).to_csv(os.path.join(cache, "map_descriptors.csv"), index=False)
    print("wrote map_per_bin.csv + map_descriptors.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the unit test + verify the driver parses**

Run: `py -m pytest tests/analysis/test_field_encoding.py -v`
Expected: PASS (7 tests).
Run: `py -c "import ast; ast.parse(open('scripts/population_field/build_map.py').read()); print('parse-ok')"`
Expected: `parse-ok`.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/field_encoding.py scripts/population_field/build_map.py tests/analysis/test_field_encoding.py
git commit -m "feat(population-field): Layer 1 functional map + contrast ladder driver"
```

- [ ] **Step 6: Golden-path run (real data)**

Run: `py scripts/population_field/build_map.py --subject BG_046`
Expected: per-rung line reporting usable sessions and FDR-significant bin counts; writes `map_per_bin.csv`, `map_descriptors.csv`, `learning_slope__*.csv`.

---

### Task 9: TF map (rung C5) from the cached registry

**Files:**
- Modify: `scripts/population_field/build_map.py`
- Test: `tests/analysis/test_field_encoding.py`

**Interfaces:**
- Produces: `field_encoding.tf_map_from_registry(reg_df, session_id, unit_ids, unit_bin_index, n_bins_anat) -> Tuple[np.ndarray, np.ndarray]` — `(mean_c1_r_per_bin, frac_responsive_per_bin)`.

⚠️ Join **per session only** (`region_bank_confirmed = False` — never pool TF-responsive unit ids across sessions to label a fixed depth bin). ⚠️ The registry is STALE (predates the lick-channel fix) — stamp `tf_registry_stale=True` on every output row.

- [ ] **Step 1: Write the failing test**

```python
def test_tf_map_aggregates_per_bin_and_clips_negative_c1():
    import pandas as pd
    reg = pd.DataFrame({
        "subject": ["BG_046"] * 4,
        "session_date": ["01072025"] * 4,
        "unit": [10, 11, 12, 13],
        "resp_log2": [True, False, False, True],
        "c1_r_log2": [0.5, -0.2, 0.1, 0.3],
    })
    mean_c1, frac = fe.tf_map_from_registry(
        reg, session_id="01072025", unit_ids=[10, 11, 12, 13],
        unit_bin_index=np.array([0, 0, 1, 1]), n_bins_anat=2)
    # bin 0: c1 = 0.5 and clip(-0.2 -> 0) => mean 0.25 ; 1 of 2 responsive
    assert mean_c1[0] == pytest.approx(0.25)
    assert frac[0] == pytest.approx(0.5)
    assert mean_c1[1] == pytest.approx(0.2)
    assert frac[1] == pytest.approx(0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_field_encoding.py -k tf_map -v`
Expected: FAIL — `AttributeError: ... has no attribute 'tf_map_from_registry'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/visdetect/analysis/field_encoding.py
def tf_map_from_registry(reg_df, session_id: str, unit_ids, unit_bin_index,
                         n_bins_anat: int):
    """Aggregate cached per-unit TF-GLM calls into field bins for ONE session.

    Returns (mean c1_r_log2 clipped at 0, fraction resp_log2) per bin.

    Per-session join ONLY: region_bank_confirmed is False in the registry, so
    TF-responsive unit ids must never be pooled across sessions to label a fixed
    depth bin (chronic probes drift).
    """
    from visdetect.analysis.config import canonical_session_id
    key = canonical_session_id(session_id)
    sub = reg_df[reg_df["session_date"].map(canonical_session_id) == key]
    c1 = dict(zip(sub["unit"].astype(int), sub["c1_r_log2"].astype(float)))
    rp = dict(zip(sub["unit"].astype(int), sub["resp_log2"].astype(bool)))
    sums = np.zeros(n_bins_anat); counts = np.zeros(n_bins_anat)
    resp = np.zeros(n_bins_anat)
    idx = np.asarray(unit_bin_index, int)
    for i, uid in enumerate(unit_ids):
        b = idx[i]
        if not (0 <= b < n_bins_anat) or int(uid) not in c1:
            continue
        sums[b] += max(0.0, c1[int(uid)])
        resp[b] += 1.0 if rp[int(uid)] else 0.0
        counts[b] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_c1 = np.where(counts > 0, sums / counts, np.nan)
        frac = np.where(counts > 0, resp / counts, np.nan)
    return mean_c1, frac
```

Then in `build_map.py`, after the LADDER loop, add the C5 block: load
`data/cache/tf_responsive/bg046_tf_responsive.csv`, call `tf_map_from_registry`
per session using the `unit_bin_index` stored in any of that session's npz files,
run the same `per_bin_learning_slope` / `session_shuffle_null` / `fdr_correct`
pipeline on `mean_c1`, and write `learning_slope__c5_tf.csv` with an added
`tf_registry_stale=True` column.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_field_encoding.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/field_encoding.py scripts/population_field/build_map.py tests/analysis/test_field_encoding.py
git commit -m "feat(population-field): TF rung from cached GLM registry (per-session join)"
```

---

### Task 10: Validation gates + figures

**Files:**
- Create: `scripts/population_field/validate_map.py`
- Create: `scripts/population_field/plot_map.py`

**Interfaces:**
- Consumes: `map_per_bin.csv`, `map_descriptors.csv`, `learning_slope__*.csv`.
- Produces: `FIGURES/population_field/BG_046/{functional_map.png, ladder_comparison.png, descriptors_across_learning.png}` + `validation_report.json`.

- [ ] **Step 1: Write the failing test (the mandatory circularity control)**

```python
# tests/analysis/test_field_stats.py
def test_shuffled_encoding_yields_no_significant_bins():
    """MANDATORY control: shuffle -> flat. A non-flat null means a BUG, not a
    finding (project hard rule on circular analysis)."""
    rng = np.random.default_rng(7)
    n_sess, n_bins = 25, 40
    values = rng.normal(size=(n_sess, n_bins))
    yields = rng.integers(5, 50, size=(n_sess, n_bins)).astype(float)
    sess_idx = np.arange(n_sess)
    rho = fs.per_bin_learning_slope(values, sess_idx, yields)
    null = fs.session_shuffle_null(values, sess_idx, yields, n_perm=300, seed=1)
    p = fs.null_pvalues(rho, null)
    from visdetect.analysis.utils import fdr_correct
    assert int(fdr_correct(p, alpha=0.05).sum()) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_field_stats.py -k shuffled_encoding -v`
Expected: FAIL if any statistic is circular; PASS once the pipeline is honest. (If it fails, the pipeline has a bug — fix the pipeline, never the test.)

- [ ] **Step 3: Write the validation + plotting scripts**

`validate_map.py` writes `validation_report.json` with:
- `registration_gate` — echo of `audit.json`.
- `yield_control_delta` — per-rung count of FDR-significant bins **with** vs **without** the yield covariate (a rung significant only *without* it is a yield artifact).
- `odd_even_reproducibility` — Spearman rho between per-bin strength computed on odd vs even trials.
- `ladder_agreement` — Spearman rho between the per-bin learning slopes of `c1_sensory_lickfree` and `c4_detection_motor`. **This is the scientific payload**: high agreement ⇒ the reshaping is robust; divergence ⇒ what reshapes is the MOTOR code, which is itself a real finding.

`plot_map.py` draws: (a) depth × session heatmap of strength per rung; (b) the ladder comparison (slope per bin, one line per rung); (c) descriptor trajectories (centroid, spread, effective bins) across sessions with `STAGE_COLORS`. Use `visdetect.suite.plotting.save_figure(fig, name, "population_field/BG_046")`.

- [ ] **Step 4: Run everything**

Run: `py -m pytest tests/analysis/test_field_stats.py tests/analysis/test_field_encoding.py tests/analysis/test_field_bins.py -q`
Expected: all PASS.
Run: `py scripts/population_field/validate_map.py --subject BG_046 && py scripts/population_field/plot_map.py --subject BG_046`
Expected: `validation_report.json` + 3 PNGs.

- [ ] **Step 5: Commit**

```bash
git add scripts/population_field/validate_map.py scripts/population_field/plot_map.py tests/analysis/test_field_stats.py
git add -f FIGURES/population_field/BG_046 data/cache/population_field/BG_046/validation_report.json
git commit -m "feat(population-field): Layer 1 validation gates + figures"
```

- [ ] **Step 6: STOP — Layer 1 gate**

Report to the user before any Layer 2/3 work:
1. Did the registration gate pass?
2. How many bins are FDR-significant per rung?
3. **Do the lick-free (C1) and motor-inclusive (C4) maps agree or diverge?**
4. Did any rung survive only without yield control?

Layers 2–3 (Plan 2b) are written only after this gate is reviewed.

---

## Self-Review

**Spec coverage.** Layer 0 field-tensor cache → Tasks 2, 7. Contrast ladder C1/C3/C4 → Tasks 3, 8. C5 TF → Task 9. C2 motor map → cached in Task 7 (`c2_fa`), consumed by C3 in Task 8. Map descriptors → Task 5. Learning model + bespoke nulls + FDR + yield control → Tasks 6, 8. Validation gates §13 → Task 10. Binary lick screen + provenance stamps (§9) → Tasks 3, 7. **Deliberately deferred to Plan 2b:** Layer 2 geometry, Layer 3 evoked profiles (spec gates them on Layer 1). **Deliberately out of scope** (spec §14): C6 change-size scaling (underpowered per-session; needs the pooled design), the naive exploratory arm, and the TF-pulse-preceded FA control — all three are additive to this pipeline and belong with Plan 2b once Layer 1's gate is read.

**Placeholder scan.** None. Every step has runnable code, an exact command, and expected output. Task 10's two scripts are specified by their exact outputs and inputs rather than full listings — they are presentation/validation code whose content depends on Task 8's actual output columns, which the implementer will have in hand.

**Type consistency.** `unit_bin_index` is `np.ndarray[int]` with `-1` for off-grid everywhere (Tasks 2, 7, 9). `bin_yield` is `(n_bins_anat,)` per session from Task 2, stacked to `(n_sessions, n_bins)` in Task 8 and consumed in that shape by `per_bin_learning_slope` (Task 6). `strength` is always `|AUROC−0.5|`, `(n_bins_anat,)`. `depths_um` from `bin_depth_um` matches the flattened `shank*n_depth + depth_bin` layout used by `unit_field_index`. `session` is the canonical 8-digit string at every boundary.

**Known gap flagged for the implementer.** `EVENT_RESPONSIVENESS_WINDOWS["FA"]` is `((-1.75,-1.25), (-0.3,-0.15))`, so the `c2_fa`/`c3_hit_lick` tensors must span at least `-2.0` s — hence their `(-2.0, 1.0)` window in `CONTRASTS`, wider than the `Change_ON` default. Do not narrow it.
