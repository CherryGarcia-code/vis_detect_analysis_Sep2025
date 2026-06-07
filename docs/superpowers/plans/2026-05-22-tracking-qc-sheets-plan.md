# Tracking QC Sheets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build per-UID QC sheets (2-page PDFs) for the 61 long-track UnitMatch cohort, plus a verdicts.csv index, so Ben can trust/exclude tracked units before scientific use.

**Architecture:** A reusable metric/loader module in `src/visdetect/analysis/tracking_qc.py` (no I/O orchestration) and a CLI script in `scripts/pipelines/tracking/build_qc_sheets.py` that loops sessions, extracts per-UID intermediates into a single pickle cache, then renders the 2-page PDFs and verdicts CSV from the cache. All imports come from canonical `visdetect.*` paths.

**Tech Stack:** Python 3.10, NumPy, pandas, matplotlib (gridspec + PdfPages), the existing `visdetect.analysis.utils.build_population_tensor` for PSTHs, `visdetect.core.session.load_session` for pkls, raw waveforms from `data/unit_match/input/BG_046/{DDMMYYYY}/RawWaveforms/Unit{kid}_RawSpikes.npy`.

**Spec:** `docs/superpowers/specs/2026-05-21-tracking-qc-sheets-design.md` (commit c10f264).

---

## File structure

| Path | Created/Modified | Responsibility |
|---|---|---|
| `src/visdetect/analysis/tracking_qc.py` | **Create** | Thresholds, dataclasses, metric functions (ISI/depth/waveform/FR), verdict logic, ISI-score loader, waveform-extraction primitives, PSTH-extraction wrapper. No I/O orchestration. |
| `tests/analysis/test_tracking_qc.py` | **Create** | Unit tests for metric and verdict functions (synthetic inputs, known outputs). |
| `scripts/pipelines/tracking/build_qc_sheets.py` | **Create** | CLI driver. Loop over sessions, build per-UID cache, render PDFs, write verdicts.csv. |
| `scripts/pipelines/tracking/qc_sheet_figures.py` | **Create** | Figure-rendering helpers (badge header, page 1, page 2, PdfPages writer). Kept as a sibling module to the CLI rather than in `visdetect.viz` because it's QC-sheet-specific. |
| `FIGURES/tracking_qc/per_uid_sheets/uid_{NNNN}.pdf` | Output | 61 two-page PDFs. |
| `FIGURES/tracking_qc/verdicts.csv` | Output | Composite index. |
| `data/cache/tracking_qc_intermediates.pkl` | Output | Cached per-UID dicts so figure tweaks skip the slow extraction. |

---

## Conventions

- **All imports** come from canonical paths: `visdetect.analysis.constants`, `visdetect.analysis.config`, `visdetect.analysis.utils`, `visdetect.suite.config`, `visdetect.suite.loader`, `visdetect.suite.plotting`, `visdetect.core.session`. **Never** import from `analysis_suite/`.
- **Long tracks**: `span ≥ 10`. The "span" column in `track_validation_stats.csv` is authoritative; for UIDs not in that CSV, span = number of unique sessions in `unit_index.csv` group.
- **Change-size pools** (override the constants `SMALL_CHANGE_SIZES`/`BIG_CHANGE_SIZES` because we want 1.5 excluded):
  - `BIG_POOL = {2.0, 4.0}`
  - `SMALL_POOL = {1.25, 1.35}`
- **Stages**: use `STAGE_ORDER = ['Learning', 'Expert']` from `visdetect.suite.config` (Naive is already merged by `SESSION_FILTER`).
- **PSTH defaults**: `bin_size = DEFAULT_BIN_SIZE` (25 ms), `sigma_ms = DEFAULT_SIGMA_MS` (25 ms). Smoothing applied after binning via `smooth_psth`.
- **Frequent commits**: commit after each task passes its tests.

---

## Task 1: Module scaffold + threshold constants

**Files:**
- Create: `src/visdetect/analysis/tracking_qc.py`
- Create: `tests/analysis/test_tracking_qc.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_tracking_qc.py
import pytest
from visdetect.analysis import tracking_qc as qc


def test_thresholds_present():
    assert qc.ISI_PASS == 0.75
    assert qc.ISI_WARN == 0.65
    assert qc.DEPTH_PASS_UM == 15.0
    assert qc.DEPTH_WARN_UM == 30.0
    assert qc.WAVE_PASS_R == 0.95
    assert qc.WAVE_WARN_R == 0.90
    assert qc.FR_CV_PASS == 0.35
    assert qc.FR_CV_WARN == 0.60


def test_change_size_pools():
    assert qc.BIG_POOL == {2.0, 4.0}
    assert qc.SMALL_POOL == {1.25, 1.35}
```

- [ ] **Step 2: Run to confirm failure**

Run: `py -m pytest tests/analysis/test_tracking_qc.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'visdetect.analysis.tracking_qc'`

- [ ] **Step 3: Create the module**

```python
# src/visdetect/analysis/tracking_qc.py
"""Per-UID tracking QC: metrics, badge logic, and extraction primitives.

This module is library code (no I/O orchestration). The
`scripts/pipelines/tracking/build_qc_sheets.py` driver wires it up.

See docs/superpowers/specs/2026-05-21-tracking-qc-sheets-design.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# ─── Badge thresholds (tweakable; documented in spec §7) ──────────────
ISI_PASS: float = 0.75
ISI_WARN: float = 0.65

DEPTH_PASS_UM: float = 15.0
DEPTH_WARN_UM: float = 30.0

WAVE_PASS_R: float = 0.95
WAVE_WARN_R: float = 0.90

FR_CV_PASS: float = 0.35
FR_CV_WARN: float = 0.60

# ─── Change-size pools for Change_ON heatmaps ─────────────────────────
# Spec excludes 1.5× from heatmaps (ambiguous mid).
BIG_POOL: Set[float] = {2.0, 4.0}
SMALL_POOL: Set[float] = {1.25, 1.35}

# ─── Footprint extraction ─────────────────────────────────────────────
# How many channels above/below the peak to include in the footprint snippet.
FOOTPRINT_HALFWIDTH_CHANS: int = 8
```

- [ ] **Step 4: Run tests, expect pass**

Run: `py -m pytest tests/analysis/test_tracking_qc.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```
git add src/visdetect/analysis/tracking_qc.py tests/analysis/test_tracking_qc.py
git commit -m "Add tracking_qc module scaffold with badge thresholds"
```

---

## Task 2: Metric functions (depth std, waveform corr, FR CV)

**Files:**
- Modify: `src/visdetect/analysis/tracking_qc.py`
- Modify: `tests/analysis/test_tracking_qc.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/analysis/test_tracking_qc.py`:

```python
import numpy as np


def test_depth_std_um_basic():
    depths = np.array([100.0, 105.0, 95.0, 100.0])
    assert qc.depth_std_um(depths) == pytest.approx(3.5355, rel=1e-3)


def test_depth_std_um_handles_nans():
    depths = np.array([100.0, np.nan, 110.0, np.nan])
    assert qc.depth_std_um(depths) == pytest.approx(5.0, rel=1e-3)


def test_depth_std_um_empty_returns_nan():
    assert np.isnan(qc.depth_std_um(np.array([])))


def test_waveform_corr_identical_returns_one():
    waves = np.tile(np.array([0.0, 1.0, 0.0, -1.0, 0.0]), (4, 1)).astype(float)
    assert qc.waveform_corr(waves) == pytest.approx(1.0, rel=1e-6)


def test_waveform_corr_normalizes_then_correlates():
    w1 = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
    w2 = w1 * 10.0          # same shape, larger amplitude
    w3 = -w1                # flipped polarity
    waves = np.stack([w1, w2, w3])
    # pairs: (1,2)=+1, (1,3)=-1, (2,3)=-1 → mean = -1/3
    assert qc.waveform_corr(waves) == pytest.approx(-1.0 / 3.0, abs=1e-6)


def test_waveform_corr_too_few_returns_nan():
    waves = np.array([[1.0, 2.0, 3.0]])     # only one session
    assert np.isnan(qc.waveform_corr(waves))


def test_fr_cv_basic():
    rates = np.array([10.0, 12.0, 8.0, 11.0, 9.0])
    expected = np.std(rates, ddof=0) / np.mean(rates)
    assert qc.fr_cv(rates) == pytest.approx(expected, rel=1e-6)


def test_fr_cv_zero_mean_returns_nan():
    assert np.isnan(qc.fr_cv(np.array([0.0, 0.0, 0.0])))


def test_fr_cv_handles_nans():
    rates = np.array([10.0, np.nan, 12.0])
    assert qc.fr_cv(rates) == pytest.approx(np.std([10.0, 12.0]) / 11.0, rel=1e-3)
```

- [ ] **Step 2: Run to confirm failure**

Run: `py -m pytest tests/analysis/test_tracking_qc.py -v`
Expected: 8 new tests fail with `AttributeError: module 'visdetect.analysis.tracking_qc' has no attribute 'depth_std_um'`.

- [ ] **Step 3: Implement metric functions**

Append to `src/visdetect/analysis/tracking_qc.py`:

```python
# ─── Cross-session metric functions ───────────────────────────────────

def depth_std_um(depths_um: np.ndarray) -> float:
    """Std of peak-channel depth across sessions, in microns.

    NaN values are ignored. Returns NaN if fewer than 2 finite values.
    """
    arr = np.asarray(depths_um, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    return float(np.std(arr, ddof=0))


def waveform_corr(waveforms: np.ndarray) -> float:
    """Mean pairwise Pearson r of L2-normalized peak-channel waveforms.

    Parameters
    ----------
    waveforms : ndarray, shape (n_sessions, n_samples)
        Per-session mean waveform on the peak channel.

    Returns
    -------
    float
        Mean over the (n*(n-1)/2) cross-session pairwise correlations.
        NaN if fewer than 2 sessions or if normalization fails.
    """
    arr = np.asarray(waveforms, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return float("nan")

    # L2-normalize per row; drop rows that are all-zero
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    keep = norms.flatten() > 1e-12
    if keep.sum() < 2:
        return float("nan")
    normed = arr[keep] / norms[keep]

    # Pearson r of normalized vectors == cosine == dot product after mean removal
    # We want Pearson, not cosine — subtract row mean first
    normed = normed - normed.mean(axis=1, keepdims=True)
    # Renormalize after mean-subtraction
    norms2 = np.linalg.norm(normed, axis=1, keepdims=True)
    norms2[norms2 < 1e-12] = 1.0
    normed = normed / norms2

    n = normed.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(float(np.dot(normed[i], normed[j])))
    return float(np.mean(pairs))


def fr_cv(rates_hz: np.ndarray) -> float:
    """Coefficient of variation (std/mean) of baseline firing rate.

    NaNs are dropped. Returns NaN for empty / zero-mean / single-session inputs.
    """
    arr = np.asarray(rates_hz, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    mean = float(np.mean(arr))
    if abs(mean) < 1e-9:
        return float("nan")
    return float(np.std(arr, ddof=0) / mean)
```

- [ ] **Step 4: Run tests, expect pass**

Run: `py -m pytest tests/analysis/test_tracking_qc.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```
git add src/visdetect/analysis/tracking_qc.py tests/analysis/test_tracking_qc.py
git commit -m "Add depth/waveform/FR cross-session metric functions"
```

---

## Task 3: Badge assignment + composite verdict

**Files:**
- Modify: `src/visdetect/analysis/tracking_qc.py`
- Modify: `tests/analysis/test_tracking_qc.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/analysis/test_tracking_qc.py`:

```python
def test_badge_isi():
    assert qc.badge_isi(0.91) == "pass"
    assert qc.badge_isi(0.70) == "warn"
    assert qc.badge_isi(0.28) == "fail"
    assert qc.badge_isi(float("nan")) == "fail"


def test_badge_depth():
    assert qc.badge_depth(8.0) == "pass"
    assert qc.badge_depth(20.0) == "warn"
    assert qc.badge_depth(45.0) == "fail"
    assert qc.badge_depth(float("nan")) == "fail"


def test_badge_waveform():
    assert qc.badge_waveform(0.97) == "pass"
    assert qc.badge_waveform(0.92) == "warn"
    assert qc.badge_waveform(0.50) == "fail"


def test_badge_fr():
    assert qc.badge_fr(0.20) == "pass"
    assert qc.badge_fr(0.45) == "warn"
    assert qc.badge_fr(0.80) == "fail"


def test_composite_all_pass_is_trusted():
    assert qc.composite_verdict(["pass", "pass", "pass", "pass"]) == "trusted"


def test_composite_one_warn_is_review():
    assert qc.composite_verdict(["pass", "warn", "pass", "pass"]) == "review"


def test_composite_two_warns_is_suspect():
    assert qc.composite_verdict(["pass", "warn", "warn", "pass"]) == "suspect"


def test_composite_any_fail_is_suspect():
    assert qc.composite_verdict(["pass", "pass", "pass", "fail"]) == "suspect"
    assert qc.composite_verdict(["pass", "warn", "fail", "pass"]) == "suspect"
```

- [ ] **Step 2: Run to confirm failure**

Run: `py -m pytest tests/analysis/test_tracking_qc.py -v`
Expected: 8 new tests fail.

- [ ] **Step 3: Implement badge + verdict functions**

Append to `src/visdetect/analysis/tracking_qc.py`:

```python
# ─── Badge / verdict logic ────────────────────────────────────────────

def _badge_threshold(value: float, pass_thr: float, warn_thr: float,
                     direction: str) -> str:
    """Apply pass/warn/fail thresholds.

    direction='high' : pass if value >= pass_thr, warn between, fail below.
    direction='low'  : pass if value <= pass_thr, warn between, fail above.
    NaN always returns 'fail'.
    """
    if not np.isfinite(value):
        return "fail"
    if direction == "high":
        if value >= pass_thr:
            return "pass"
        if value >= warn_thr:
            return "warn"
        return "fail"
    elif direction == "low":
        if value <= pass_thr:
            return "pass"
        if value <= warn_thr:
            return "warn"
        return "fail"
    raise ValueError(f"direction must be 'high' or 'low', got {direction!r}")


def badge_isi(median_corr: float) -> str:
    return _badge_threshold(median_corr, ISI_PASS, ISI_WARN, direction="high")


def badge_depth(std_um: float) -> str:
    return _badge_threshold(std_um, DEPTH_PASS_UM, DEPTH_WARN_UM, direction="low")


def badge_waveform(mean_pairwise_r: float) -> str:
    return _badge_threshold(mean_pairwise_r, WAVE_PASS_R, WAVE_WARN_R, direction="high")


def badge_fr(cv: float) -> str:
    return _badge_threshold(cv, FR_CV_PASS, FR_CV_WARN, direction="low")


def composite_verdict(badges: Sequence[str]) -> str:
    """Spec §7 composite logic.

    trusted = all pass
    review  = ≤1 warn AND no fails
    suspect = any fail OR ≥2 warns
    """
    n_fail = sum(1 for b in badges if b == "fail")
    n_warn = sum(1 for b in badges if b == "warn")
    if n_fail >= 1 or n_warn >= 2:
        return "suspect"
    if n_warn == 1:
        return "review"
    return "trusted"
```

- [ ] **Step 4: Run tests, expect pass**

Run: `py -m pytest tests/analysis/test_tracking_qc.py -v`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```
git add src/visdetect/analysis/tracking_qc.py tests/analysis/test_tracking_qc.py
git commit -m "Add per-criterion badges and composite verdict logic"
```

---

## Task 4: ISI median loader (from validate_long_tracks.py output)

**Files:**
- Modify: `src/visdetect/analysis/tracking_qc.py`
- Modify: `tests/analysis/test_tracking_qc.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/analysis/test_tracking_qc.py`:

```python
import pandas as pd
import tempfile
from pathlib import Path


def test_load_isi_scores(tmp_path):
    csv = tmp_path / "track_validation_stats.csv"
    csv.write_text(
        "global_uid,mean,median,min,count,span,nonmatched_rank_pct\n"
        "334,0.73,0.91,-0.39,725,27,82.9\n"
        "779,0.30,0.28,-0.40,500,15,5.0\n"
    )
    scores = qc.load_isi_scores(csv)
    assert scores[334] == pytest.approx(0.91)
    assert scores[779] == pytest.approx(0.28)
    assert scores.get(9999, float("nan")) != scores.get(9999, float("nan"))  # NaN sentinel for missing
```

- [ ] **Step 2: Run to confirm failure**

Run: `py -m pytest tests/analysis/test_tracking_qc.py::test_load_isi_scores -v`
Expected: FAIL — `AttributeError: ... has no attribute 'load_isi_scores'`.

- [ ] **Step 3: Implement loader**

Append to `src/visdetect/analysis/tracking_qc.py`:

```python
import pandas as pd
from pathlib import Path
from collections import defaultdict


def load_isi_scores(csv_path) -> Dict[int, float]:
    """Read the median ISI corr per global_uid from validate_long_tracks output.

    Missing UIDs are returned as NaN via a defaultdict.
    """
    df = pd.read_csv(csv_path)
    scores = defaultdict(lambda: float("nan"))
    for _, row in df.iterrows():
        scores[int(row["global_uid"])] = float(row["median"])
    return scores
```

- [ ] **Step 4: Run test, expect pass**

Run: `py -m pytest tests/analysis/test_tracking_qc.py::test_load_isi_scores -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add src/visdetect/analysis/tracking_qc.py tests/analysis/test_tracking_qc.py
git commit -m "Add ISI median score loader for validate_long_tracks CSV"
```

---

## Task 5: ISI histogram primitive + waveform extraction primitives

**Files:**
- Modify: `src/visdetect/analysis/tracking_qc.py`
- Modify: `tests/analysis/test_tracking_qc.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/analysis/test_tracking_qc.py`:

```python
def test_isi_log_histogram():
    rng = np.random.default_rng(0)
    spike_times = np.sort(rng.exponential(0.1, size=1000).cumsum())
    h, centers = qc.isi_log_histogram(spike_times, n_bins=50)
    assert h.shape == (50,)
    assert centers.shape == (50,)
    assert h.sum() == pytest.approx(1.0, rel=1e-6)


def test_isi_log_histogram_too_few_spikes_returns_nans():
    h, centers = qc.isi_log_histogram(np.array([0.1, 0.2]), n_bins=50)
    assert np.all(np.isnan(h))
    assert centers.shape == (50,)


def test_extract_peak_channel_picks_max_amplitude():
    # raw waveform shape: (n_samples, n_channels, n_cv_halves)
    n_samp, n_ch, n_cv = 82, 384, 2
    waveforms = np.zeros((n_samp, n_ch, n_cv), dtype=np.float32)
    # channel 17 has a clean spike
    waveforms[30:40, 17, :] = -1.5
    waveforms[40, 17, :] = 0.5
    mean_wave = waveforms.mean(axis=-1)  # (n_samp, n_ch)
    peak_chan = qc.extract_peak_channel(mean_wave)
    assert peak_chan == 17


def test_extract_footprint_centered_on_peak():
    n_samp, n_ch = 82, 384
    mean_wave = np.zeros((n_samp, n_ch), dtype=np.float32)
    mean_wave[:, 100] = np.linspace(-1.0, 1.0, n_samp)
    fp, channels = qc.extract_footprint(mean_wave, peak_chan=100, halfwidth=8)
    assert fp.shape == (n_samp, 17)        # 2*8 + 1
    assert channels.tolist() == list(range(92, 109))


def test_extract_footprint_clips_at_probe_edge():
    n_samp, n_ch = 82, 384
    mean_wave = np.zeros((n_samp, n_ch), dtype=np.float32)
    fp, channels = qc.extract_footprint(mean_wave, peak_chan=2, halfwidth=8)
    assert fp.shape[1] == 11               # 0..10 inclusive
    assert channels.tolist() == list(range(0, 11))
```

- [ ] **Step 2: Run to confirm failure**

Run: `py -m pytest tests/analysis/test_tracking_qc.py -v`
Expected: 5 new tests fail.

- [ ] **Step 3: Implement primitives**

Append to `src/visdetect/analysis/tracking_qc.py`:

```python
# ─── ISI histogram ────────────────────────────────────────────────────
# Matches the binning used by validate_long_tracks.py (1 ms .. 10 s, log).
_ISI_BIN_EDGES = np.logspace(-3, 1, 51)
_ISI_CENTERS = 0.5 * (_ISI_BIN_EDGES[:-1] + _ISI_BIN_EDGES[1:])


def isi_log_histogram(spike_times: np.ndarray, n_bins: int = 50
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Normalised log-ISI histogram, 1 ms .. 10 s, 50 bins by default.

    Returns
    -------
    h : ndarray, shape (n_bins,)
        Probability mass per bin (sums to 1).  All-NaN if too few spikes.
    centers : ndarray, shape (n_bins,)
        Bin centres (s).
    """
    if n_bins != 50:
        edges = np.logspace(-3, 1, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
    else:
        edges = _ISI_BIN_EDGES
        centers = _ISI_CENTERS

    if spike_times is None or len(spike_times) < 20:
        return np.full(n_bins, np.nan), centers
    isis = np.diff(np.sort(spike_times))
    isis = isis[(isis > 0) & (isis < 10)]
    if len(isis) < 10:
        return np.full(n_bins, np.nan), centers
    h, _ = np.histogram(isis, bins=edges)
    if h.sum() == 0:
        return np.full(n_bins, np.nan), centers
    return h.astype(float) / h.sum(), centers


# ─── Waveform / footprint extraction ──────────────────────────────────

def extract_peak_channel(mean_waveform: np.ndarray) -> int:
    """Index of the channel with the largest peak-to-peak amplitude.

    Parameters
    ----------
    mean_waveform : ndarray, shape (n_samples, n_channels)
    """
    ptp = mean_waveform.max(axis=0) - mean_waveform.min(axis=0)
    return int(np.argmax(ptp))


def extract_footprint(mean_waveform: np.ndarray, peak_chan: int,
                      halfwidth: int = FOOTPRINT_HALFWIDTH_CHANS
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Footprint snippet: (n_samples, 2*halfwidth+1) clipped at probe edges.

    Returns
    -------
    snippet : ndarray, shape (n_samples, n_channels_kept)
    channel_indices : ndarray, shape (n_channels_kept,)
    """
    n_ch = mean_waveform.shape[1]
    lo = max(0, peak_chan - halfwidth)
    hi = min(n_ch, peak_chan + halfwidth + 1)
    channels = np.arange(lo, hi)
    snippet = mean_waveform[:, lo:hi]
    return snippet, channels
```

- [ ] **Step 4: Run tests, expect pass**

Run: `py -m pytest tests/analysis/test_tracking_qc.py -v`
Expected: 23 passed.

- [ ] **Step 5: Commit**

```
git add src/visdetect/analysis/tracking_qc.py tests/analysis/test_tracking_qc.py
git commit -m "Add ISI log-histogram and waveform/footprint extraction primitives"
```

---

## Task 6: Raw-waveform loader for UnitMatch input files

**Files:**
- Modify: `src/visdetect/analysis/tracking_qc.py`

This pulls `data/unit_match/input/BG_046/{DDMMYYYY}/RawWaveforms/Unit{kid}_RawSpikes.npy` and reduces it to a mean waveform `(n_samples, n_channels)`. The on-disk shape is `(n_samples, n_channels, n_cv_halves)` per the UnitMatch input spec. No unit test — this is an I/O wrapper and a real-data smoke test runs in Task 16.

- [ ] **Step 1: Implement loader**

Append to `src/visdetect/analysis/tracking_qc.py`:

```python
import os


def load_raw_mean_waveform(raw_wf_root, session_name: str, ks_unit_id: int
                            ) -> Optional[np.ndarray]:
    """Load Unit{kid}_RawSpikes.npy and return mean across CV halves.

    Parameters
    ----------
    raw_wf_root : str or Path
        e.g. ``data/unit_match/input/BG_046``
    session_name : str
        DDMMYYYY (8-digit) — matches the unit-match input layout.
    ks_unit_id : int

    Returns
    -------
    mean_waveform : ndarray, shape (n_samples, n_channels), or None if file missing.
    """
    candidates = [session_name, session_name.zfill(8)]
    for cand in candidates:
        path = os.path.join(str(raw_wf_root), cand, "RawWaveforms",
                            f"Unit{ks_unit_id}_RawSpikes.npy")
        if os.path.exists(path):
            raw = np.load(path)   # (n_samples, n_channels, n_cv)
            if raw.ndim == 3:
                return raw.mean(axis=-1).astype(np.float32)
            elif raw.ndim == 2:
                return raw.astype(np.float32)
            return None
    return None


def load_channel_positions(raw_wf_root, session_name: str) -> Optional[np.ndarray]:
    """Load channel_positions.npy for a session.  Shape (n_channels, 2) [x_um, y_um]."""
    for cand in (session_name, session_name.zfill(8)):
        path = os.path.join(str(raw_wf_root), cand, "channel_positions.npy")
        if os.path.exists(path):
            return np.load(path).astype(np.float32)
    return None
```

- [ ] **Step 2: Smoke-check the loader against real data**

Run an inline check to confirm at least one real waveform loads:

```
py -c "from visdetect.analysis.tracking_qc import load_raw_mean_waveform, load_channel_positions; import numpy as np; w = load_raw_mean_waveform('data/unit_match/input/BG_046', '23062025', 3); p = load_channel_positions('data/unit_match/input/BG_046', '23062025'); print('wf:', None if w is None else w.shape, 'pos:', None if p is None else p.shape)"
```

Expected: prints something like `wf: (82, 384) pos: (384, 2)` (exact sample/channel counts depend on probe config — accept any shape with `(n_samples, n_channels)` for `wf` and `(n_channels, 2)` for `pos`).

- [ ] **Step 3: Commit**

```
git add src/visdetect/analysis/tracking_qc.py
git commit -m "Add raw-waveform and channel-position loaders for UnitMatch inputs"
```

---

## Task 7: PSTH extraction wrapper

**Files:**
- Modify: `src/visdetect/analysis/tracking_qc.py`

Wraps `visdetect.analysis.utils.build_population_tensor` with our event/outcome/change-size configurations. No unit test — `build_population_tensor` is already tested; this is a thin orchestration wrapper exercised in the smoke test.

- [ ] **Step 1: Implement wrapper**

Append to `src/visdetect/analysis/tracking_qc.py`:

```python
from visdetect.analysis.utils import build_population_tensor, smooth_psth
from visdetect.analysis.constants import (
    DEFAULT_BIN_SIZE, DEFAULT_SIGMA_MS, EVENT_RESPONSIVENESS_WINDOWS,
)

# Spec §5 / §4: PSTH conditions per UID per session.
# Keys are stable IDs used as dict keys in the intermediate record.
PSTH_CONDITIONS: Dict[str, Dict] = {
    "baseline_on":        {"event": "Baseline_ON", "outcomes": None,           "sizes": None,       "window": (-0.5, 1.5)},
    "change_on_big_hit":  {"event": "Change_ON",   "outcomes": {"hit"},        "sizes": BIG_POOL,   "window": (-0.5, 0.5)},
    "change_on_big_miss": {"event": "Change_ON",   "outcomes": {"miss"},       "sizes": BIG_POOL,   "window": (-0.5, 0.5)},
    "change_on_sm_hit":   {"event": "Change_ON",   "outcomes": {"hit"},        "sizes": SMALL_POOL, "window": (-0.5, 0.5)},
    "change_on_sm_miss":  {"event": "Change_ON",   "outcomes": {"miss"},       "sizes": SMALL_POOL, "window": (-0.5, 0.5)},
    "hit_lick":           {"event": "Hit",         "outcomes": {"hit"},        "sizes": None,       "window": (-1.0, 1.0)},
}


def _trial_indices_for_sizes(session, sizes: Optional[Set[float]]) -> Optional[List[int]]:
    """Return trial indices whose change_size is in `sizes`, or None for no filter."""
    if sizes is None:
        return None
    out = []
    for i, t in enumerate(session.trials):
        cs = getattr(t, "change_size", None)
        if cs is None:
            continue
        # Match within tolerance because change sizes are floats
        for sz in sizes:
            if abs(float(cs) - sz) < 1e-3:
                out.append(i)
                break
    return out


def extract_unit_psths(session, ks_unit_id: int
                        ) -> Dict[str, Tuple[np.ndarray, np.ndarray, int]]:
    """Build PSTHs for all spec conditions for one (session, unit).

    Returns
    -------
    dict[condition_key] -> (psth_smoothed_hz, bin_centers, n_trials)
        psth shape: (n_bins,)
        bin_centers shape: (n_bins,)
        n_trials: int — number of trials averaged
        If no trials match, value is (None, None, 0).
    """
    out: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}
    for key, cfg in PSTH_CONDITIONS.items():
        trial_idx = _trial_indices_for_sizes(session, cfg["sizes"])
        if trial_idx is not None and len(trial_idx) == 0:
            out[key] = (None, None, 0)
            continue
        try:
            tensor, centers, valid = build_population_tensor(
                session,
                cluster_ids=[ks_unit_id],
                event_name=cfg["event"],
                window=cfg["window"],
                bin_size=DEFAULT_BIN_SIZE,
                outcome_filter=cfg["outcomes"],
                trial_indices=trial_idx,
            )
        except ValueError:
            out[key] = (None, None, 0)
            continue
        # tensor: (n_trials, n_bins, 1) — collapse units, mean over trials, smooth
        mean_rate = tensor[:, :, 0].mean(axis=0)
        smoothed = smooth_psth(mean_rate, bin_size=DEFAULT_BIN_SIZE,
                                sigma_ms=DEFAULT_SIGMA_MS)
        out[key] = (smoothed, centers, len(valid))
    return out
```

- [ ] **Step 2: Commit**

```
git add src/visdetect/analysis/tracking_qc.py
git commit -m "Add PSTH extraction wrapper for spec's 6 event/outcome conditions"
```

---

## Task 8: Per-UID intermediate record + extraction loop

**Files:**
- Modify: `src/visdetect/analysis/tracking_qc.py`

Defines the intermediate dataclass and the outer-loop-by-session extractor. No unit test — exercised by the smoke test in Task 16.

- [ ] **Step 1: Implement intermediate record + extractor**

Append to `src/visdetect/analysis/tracking_qc.py`:

```python
import gc


@dataclass
class SessionRecord:
    """Per-session extracted data for one UID."""
    session_name: str
    ks_unit_id: int
    stage: str
    peak_chan: int
    peak_depth_um: float
    amplitude: float
    baseline_fr_hz: float
    waveform_peak: np.ndarray             # (n_samples,)
    footprint: np.ndarray                 # (n_samples, n_channels_kept)
    footprint_channels: np.ndarray        # (n_channels_kept,)
    isi_hist: np.ndarray                  # (50,)
    isi_centers: np.ndarray               # (50,)
    psths: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = field(default_factory=dict)


@dataclass
class UIDIntermediate:
    """Everything needed to render one UID's QC sheet."""
    global_uid: int
    span: int
    has_naive_to_expert: bool
    suspect_known: bool
    sessions: List[SessionRecord] = field(default_factory=list)


def _compute_baseline_fr(cluster, session) -> float:
    """Spikes during the pre-Baseline_ON window / total ITI duration.

    Cheap robust proxy: total spikes / max-spike-time.  Same convention as
    visdetect.analysis.utils.get_good_cluster_ids.
    """
    if cluster.spike_times is None or len(cluster.spike_times) == 0:
        return float("nan")
    duration = float(cluster.spike_times[-1])
    if duration < 1.0:
        return float("nan")
    return len(cluster.spike_times) / duration


def extract_session_records(session, ks_unit_ids: Sequence[int], session_name: str,
                             stage: str, raw_wf_root, channel_positions: Optional[np.ndarray]
                             ) -> Dict[int, SessionRecord]:
    """Extract per-UID SessionRecord for every (uid, ks_id) in this session.

    Returns a dict keyed by ks_unit_id.  Caller maps ks_id -> global_uid.
    """
    out: Dict[int, SessionRecord] = {}
    cluster_map = {c.cluster_id: c for c in session.clusters}
    for kid in ks_unit_ids:
        cluster = cluster_map.get(int(kid))
        if cluster is None:
            continue

        # Waveform / footprint
        mean_wf = load_raw_mean_waveform(raw_wf_root, session_name, int(kid))
        if mean_wf is None:
            # Cluster exists but no raw waveform file — skip
            continue
        peak_chan = extract_peak_channel(mean_wf)
        peak_wave = mean_wf[:, peak_chan]
        footprint, fp_chans = extract_footprint(mean_wf, peak_chan)

        # Depth & amplitude
        if channel_positions is not None and peak_chan < channel_positions.shape[0]:
            depth_um = float(channel_positions[peak_chan, 1])
        else:
            depth_um = float("nan")
        amplitude = float(peak_wave.max() - peak_wave.min())

        # FR / ISI
        baseline_fr = _compute_baseline_fr(cluster, session)
        spike_times = np.asarray(cluster.spike_times)
        isi_h, isi_c = isi_log_histogram(spike_times)

        # PSTHs
        psths = extract_unit_psths(session, int(kid))

        out[int(kid)] = SessionRecord(
            session_name=session_name,
            ks_unit_id=int(kid),
            stage=stage,
            peak_chan=peak_chan,
            peak_depth_um=depth_um,
            amplitude=amplitude,
            baseline_fr_hz=baseline_fr,
            waveform_peak=peak_wave.astype(np.float32),
            footprint=footprint.astype(np.float32),
            footprint_channels=fp_chans,
            isi_hist=isi_h.astype(np.float32),
            isi_centers=isi_c.astype(np.float32),
            psths=psths,
        )
    return out
```

- [ ] **Step 2: Commit**

```
git add src/visdetect/analysis/tracking_qc.py
git commit -m "Add SessionRecord/UIDIntermediate dataclasses and session extractor"
```

---

## Task 9: Cache I/O + cohort selection

**Files:**
- Modify: `src/visdetect/analysis/tracking_qc.py`

- [ ] **Step 1: Implement cohort selection + cache helpers**

Append to `src/visdetect/analysis/tracking_qc.py`:

```python
import pickle


KNOWN_SUSPECTS: Set[int] = {779, 873, 872}


def select_long_tracks(unit_index_csv, isi_stats_csv,
                       min_span: int = 10) -> pd.DataFrame:
    """Long-track cohort: UIDs with span >= min_span.

    Span is taken from isi_stats_csv (authoritative). UIDs not present there
    fall back to counting unique sessions in unit_index.

    Returns
    -------
    DataFrame with columns: global_uid, span, has_naive_to_expert, suspect_known
    """
    ui = pd.read_csv(unit_index_csv)
    span_by_uid = ui.groupby("global_uid")["session"].nunique().to_dict()

    if Path(isi_stats_csv).exists():
        stats = pd.read_csv(isi_stats_csv)
        for _, r in stats.iterrows():
            span_by_uid[int(r["global_uid"])] = int(r["span"])

    rows = []
    for uid, span in span_by_uid.items():
        if span < min_span:
            continue
        sessions = ui.loc[ui["global_uid"] == uid, "session"].astype(str).tolist()
        rows.append({
            "global_uid": int(uid),
            "span": int(span),
            "sessions": sessions,
            "suspect_known": int(uid) in KNOWN_SUSPECTS,
        })
    return pd.DataFrame(rows).sort_values("global_uid").reset_index(drop=True)


def annotate_naive_to_expert(cohort: pd.DataFrame, manifest: pd.DataFrame
                              ) -> pd.DataFrame:
    """Add has_naive_to_expert column based on manifest stage assignments.

    A UID is N→E if it spans (any of first 8 sessions) and (any of last 8 sessions).
    Uses chronological order from manifest.session_name.
    """
    chrono = manifest.sort_values("session_name").reset_index(drop=True)
    first_eight = set(chrono["session_name"].astype(str).head(8))
    last_eight  = set(chrono["session_name"].astype(str).tail(8))

    flags = []
    for _, row in cohort.iterrows():
        sess = set(str(s) for s in row["sessions"])
        flags.append(bool(sess & first_eight) and bool(sess & last_eight))
    cohort = cohort.copy()
    cohort["has_naive_to_expert"] = flags
    return cohort


def save_cache(intermediates: Dict[int, UIDIntermediate], path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(intermediates, f)


def load_cache(path) -> Optional[Dict[int, UIDIntermediate]]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)
```

- [ ] **Step 2: Commit**

```
git add src/visdetect/analysis/tracking_qc.py
git commit -m "Add long-track cohort selection, N→E annotation, and cache I/O"
```

---

## Task 10: Figure helpers — badge header + stage stripe

**Files:**
- Create: `scripts/pipelines/tracking/qc_sheet_figures.py`

Figure helpers live alongside the CLI driver, not in `visdetect.viz`, because they are QC-sheet-specific.

- [ ] **Step 1: Create the figures module with badge header**

```python
# scripts/pipelines/tracking/qc_sheet_figures.py
"""Figure-rendering helpers for the per-UID QC sheets.

Two pages per UID.  All gridspec ratios are picked per
docs/superpowers/specs/2026-05-21-tracking-qc-sheets-design.md §6.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from visdetect.suite.config import STAGE_COLORS, STAGE_ORDER  # noqa: E402
from visdetect.suite.plotting import setup_style                # noqa: E402
from visdetect.analysis.tracking_qc import (                    # noqa: E402
    UIDIntermediate, SessionRecord,
    badge_isi, badge_depth, badge_waveform, badge_fr, composite_verdict,
)

setup_style()

# Per-criterion colors
BADGE_COLORS = {"pass": "#2d5a2d", "warn": "#5a5a2d", "fail": "#5a2d2d"}
BADGE_SYMBOLS = {"pass": "✅", "warn": "⚠", "fail": "❌"}


def draw_header(ax, uid: UIDIntermediate,
                isi_score: float, depth_std: float,
                wave_corr: float, fr_cv_val: float) -> str:
    """Draw the 4-badge header strip + stage stripe.  Returns composite verdict."""
    ax.set_axis_off()

    b_isi   = badge_isi(isi_score)
    b_dep   = badge_depth(depth_std)
    b_wave  = badge_waveform(wave_corr)
    b_fr    = badge_fr(fr_cv_val)
    verdict = composite_verdict([b_isi, b_dep, b_wave, b_fr])

    ne_flag = " · N→E" if uid.has_naive_to_expert else ""
    suspect = " · ⚠ KNOWN SUSPECT" if uid.suspect_known else ""
    title = (f"UID {uid.global_uid} · span {uid.span}{ne_flag}{suspect}"
             f"   composite: {verdict.upper()}")
    ax.text(0.0, 0.92, title, fontsize=13, fontweight="bold",
            transform=ax.transAxes, va="top")

    # Badge row
    badges = [
        (f"ISI {isi_score:.2f}", b_isi),
        (f"depth {depth_std:.1f}µm", b_dep),
        (f"wave r={wave_corr:.2f}", b_wave),
        (f"FR CV {fr_cv_val:.2f}", b_fr),
    ]
    x = 0.0
    for label, level in badges:
        text = f"{BADGE_SYMBOLS[level]} {label}"
        ax.text(x, 0.55, text, fontsize=10,
                transform=ax.transAxes, va="center",
                bbox=dict(facecolor=BADGE_COLORS[level], edgecolor="none",
                          pad=4, alpha=0.85),
                color="white")
        x += 0.20

    # Stage stripe at the bottom: one cell per session in chronological order
    if uid.sessions:
        n = len(uid.sessions)
        bar_y = 0.05
        bar_h = 0.18
        for i, rec in enumerate(uid.sessions):
            color = STAGE_COLORS.get(rec.stage, "#888888")
            ax.add_patch(Rectangle((i / n, bar_y), 1.0 / n, bar_h,
                                    transform=ax.transAxes,
                                    facecolor=color, edgecolor="none"))

    return verdict
```

- [ ] **Step 2: Smoke-render the header on a synthetic UID**

```
py -c "
import sys; from pathlib import Path
sys.path.insert(0, str(Path('src')))
from scripts.pipelines.tracking.qc_sheet_figures import draw_header
from visdetect.analysis.tracking_qc import UIDIntermediate, SessionRecord
import matplotlib.pyplot as plt
import numpy as np

recs = [SessionRecord(session_name=f's{i}', ks_unit_id=0, stage='Learning' if i<10 else 'Expert',
                       peak_chan=0, peak_depth_um=100.0, amplitude=1.0, baseline_fr_hz=5.0,
                       waveform_peak=np.zeros(82), footprint=np.zeros((82,17)), footprint_channels=np.arange(17),
                       isi_hist=np.zeros(50), isi_centers=np.zeros(50)) for i in range(20)]
uid = UIDIntermediate(global_uid=334, span=20, has_naive_to_expert=True, suspect_known=False, sessions=recs)
fig, ax = plt.subplots(figsize=(10, 1.6))
verdict = draw_header(ax, uid, isi_score=0.91, depth_std=8.0, wave_corr=0.97, fr_cv_val=0.45)
print('verdict:', verdict)
fig.savefig('FIGURES/tracking_qc/_smoke_header.png', dpi=120, bbox_inches='tight')
print('OK')
"
```

Expected: prints `verdict: review` (one warn from FR), saves a PNG. Open `FIGURES/tracking_qc/_smoke_header.png` and confirm the badges, title, and stage stripe render correctly.

- [ ] **Step 3: Commit**

```
git add scripts/pipelines/tracking/qc_sheet_figures.py
git commit -m "Add badge header + stage stripe drawer for QC sheets"
```

---

## Task 11: Figure helpers — page 1 (physical)

**Files:**
- Modify: `scripts/pipelines/tracking/qc_sheet_figures.py`

Page 1: badge header, 3 footprints (first/mid/last), peak-channel waveform overlay, depth-on-probe, amplitude, UM pairwise scores. W:H ratios per spec §6.

- [ ] **Step 1: Implement page 1 renderer**

Append to `scripts/pipelines/tracking/qc_sheet_figures.py`:

```python
def _waveform_color(stage: str) -> str:
    return STAGE_COLORS.get(stage, "#888888")


def render_page1(uid: UIDIntermediate, um_pair_scores: Optional[np.ndarray],
                 isi_score: float, depth_std: float, wave_corr: float,
                 fr_cv_val: float) -> plt.Figure:
    """Render page 1 (physical) — returns the Figure."""
    fig = plt.figure(figsize=(8.5, 11.0))

    # Master gridspec: header / footprints / waveform / depth-amp / um-scores
    gs = gridspec.GridSpec(
        nrows=5, ncols=1,
        height_ratios=[0.9, 2.5, 1.5, 1.8, 0.9],
        hspace=0.55, top=0.96, bottom=0.04, left=0.08, right=0.96,
    )

    # Header
    ax_hdr = fig.add_subplot(gs[0])
    draw_header(ax_hdr, uid, isi_score, depth_std, wave_corr, fr_cv_val)

    # Footprints @ first / mid / last
    fp_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], wspace=0.25)
    n = len(uid.sessions)
    if n >= 1:
        idxs = [0, n // 2, n - 1]
        labels = ["first", "mid", "last"]
        for col, (idx, lab) in enumerate(zip(idxs, labels)):
            ax = fig.add_subplot(fp_gs[col])
            rec = uid.sessions[idx]
            # Footprint: lines per channel stacked vertically by channel index
            fp = rec.footprint                                # (n_samples, n_chans)
            offsets = np.arange(fp.shape[1])[None, :] * (np.abs(fp).max() + 1e-6) * 1.2
            ax.plot(fp + offsets, color=_waveform_color(rec.stage), linewidth=0.6)
            ax.set_title(f"{lab}: {rec.session_name}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    # Peak-channel waveform overlay (~2:1)
    ax_wf = fig.add_subplot(gs[2])
    for rec in uid.sessions:
        ax_wf.plot(rec.waveform_peak, color=_waveform_color(rec.stage),
                   linewidth=0.6, alpha=0.6)
    ax_wf.set_title("Peak-channel waveform overlay", fontsize=10)
    ax_wf.set_xlabel("samples"); ax_wf.set_ylabel("µV (raw)")

    # Depth + amplitude side by side (~3:1 each)
    da_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3], wspace=0.30)
    ax_d = fig.add_subplot(da_gs[0])
    ax_a = fig.add_subplot(da_gs[1])
    xs = np.arange(len(uid.sessions))
    depths = [r.peak_depth_um for r in uid.sessions]
    amps   = [r.amplitude for r in uid.sessions]
    colors = [_waveform_color(r.stage) for r in uid.sessions]
    ax_d.scatter(xs, depths, c=colors, s=18); ax_d.plot(xs, depths, color="0.5", linewidth=0.7)
    ax_d.set_xlabel("session #"); ax_d.set_ylabel("peak depth (µm)")
    ax_d.set_title("Depth on probe", fontsize=10)
    ax_a.scatter(xs, amps, c=colors, s=18);   ax_a.plot(xs, amps,   color="0.5", linewidth=0.7)
    ax_a.set_xlabel("session #"); ax_a.set_ylabel("amplitude (µV)")
    ax_a.set_title("Amplitude", fontsize=10)

    # UM pairwise scores
    ax_um = fig.add_subplot(gs[4])
    if um_pair_scores is not None and len(um_pair_scores) > 0:
        ax_um.bar(np.arange(len(um_pair_scores)), um_pair_scores, color="0.4")
        ax_um.set_ylim(0, 1)
        ax_um.set_title("UM consecutive-session match probability", fontsize=10)
        ax_um.set_xlabel("pair # (session i, i+1)")
    else:
        ax_um.text(0.5, 0.5, "UM pair scores unavailable", ha="center", va="center",
                   transform=ax_um.transAxes, fontsize=10, color="0.5")
        ax_um.set_axis_off()

    return fig
```

- [ ] **Step 2: Commit**

```
git add scripts/pipelines/tracking/qc_sheet_figures.py
git commit -m "Add page-1 renderer (footprints, waveform overlay, depth/amp, UM scores)"
```

---

## Task 12: Figure helpers — page 2 (functional)

**Files:**
- Modify: `scripts/pipelines/tracking/qc_sheet_figures.py`

Page 2 layout per spec §5: header, ISI overlay + baseline FR (top row), Baseline_ON heatmap, BigHit | SmallHit heatmaps (side by side), Hit lick heatmap. Each heatmap gets a small L-vs-E stage-mean inset in its top-right corner.

- [ ] **Step 1: Implement helpers + page 2 renderer**

Append to `scripts/pipelines/tracking/qc_sheet_figures.py`:

```python
def _psth_matrix(uid: UIDIntermediate, key: str) -> Optional[tuple]:
    """Stack per-session PSTH rows into (n_sessions, n_bins) + bin_centers + stages.

    Returns (matrix, centers, stages, n_trials_per_session) or None if every session is empty.
    """
    rows, centers, stages, n_trials = [], None, [], []
    for rec in uid.sessions:
        psth, c, n = rec.psths.get(key, (None, None, 0))
        if psth is None:
            continue
        rows.append(psth)
        centers = c
        stages.append(rec.stage)
        n_trials.append(n)
    if not rows:
        return None
    return np.vstack(rows), centers, stages, n_trials


def _draw_heatmap_with_inset(parent_gs, uid: UIDIntermediate, key: str,
                              title: str, miss_keys: Optional[List[str]] = None) -> None:
    """Render a chronological PSTH heatmap with a stage-mean inset.

    miss_keys (optional): list of keys whose stage-mean traces to overlay in the
    inset for hit/miss comparison (Change_ON only).
    """
    fig = parent_gs.get_gridspec().figure
    ax_main = fig.add_subplot(parent_gs)
    data = _psth_matrix(uid, key)
    if data is None:
        ax_main.text(0.5, 0.5, f"no trials for {key}", ha="center", va="center",
                     transform=ax_main.transAxes, fontsize=9, color="0.5")
        ax_main.set_axis_off()
        return

    mat, centers, stages, _ = data
    vmax = np.percentile(mat, 99)
    ax_main.imshow(mat, aspect="auto", origin="lower", cmap="magma",
                   extent=[centers[0], centers[-1], 0, mat.shape[0]],
                   vmin=0, vmax=max(vmax, 1e-6))
    ax_main.axvline(0, color="white", linewidth=0.8, alpha=0.7)
    ax_main.set_title(title, fontsize=10)
    ax_main.set_xlabel("time (s)"); ax_main.set_ylabel("session #")

    # Inset: L vs E stage-mean
    bbox = ax_main.get_position()
    iw, ih = 0.18 * bbox.width / 0.4, 0.20 * bbox.height / 0.4   # roughly 1.2:1
    inset = fig.add_axes([bbox.x0 + bbox.width - iw - 0.005,
                          bbox.y0 + bbox.height - ih - 0.005,
                          iw, ih])
    inset.set_facecolor("#0d0d0d")
    for st in STAGE_ORDER:
        mask = np.array([s == st for s in stages])
        if mask.sum() == 0:
            continue
        inset.plot(centers, mat[mask].mean(axis=0), color=STAGE_COLORS[st],
                   linewidth=1.0, label=st)
    if miss_keys:
        for mk in miss_keys:
            mdata = _psth_matrix(uid, mk)
            if mdata is None:
                continue
            mmat, mcenters, mstages, _ = mdata
            for st in STAGE_ORDER:
                mask = np.array([s == st for s in mstages])
                if mask.sum() == 0:
                    continue
                inset.plot(mcenters, mmat[mask].mean(axis=0),
                           color=STAGE_COLORS[st], linewidth=1.0,
                           linestyle="--", alpha=0.7)
    inset.axvline(0, color="white", linewidth=0.6, alpha=0.5)
    inset.tick_params(labelsize=6)
    inset.set_xticks([centers[0], 0.0, centers[-1]])
    inset.set_yticks([])


def render_page2(uid: UIDIntermediate, isi_score: float, depth_std: float,
                 wave_corr: float, fr_cv_val: float) -> plt.Figure:
    fig = plt.figure(figsize=(8.5, 11.0))
    gs = gridspec.GridSpec(
        nrows=5, ncols=2,
        height_ratios=[0.9, 1.6, 1.8, 2.2, 1.8],
        width_ratios=[1, 1],
        hspace=0.55, wspace=0.25,
        top=0.96, bottom=0.04, left=0.09, right=0.96,
    )

    # Header (spans both columns)
    ax_hdr = fig.add_subplot(gs[0, :])
    draw_header(ax_hdr, uid, isi_score, depth_std, wave_corr, fr_cv_val)

    # Row 1: ISI overlay + baseline FR
    ax_isi = fig.add_subplot(gs[1, 0])
    for rec in uid.sessions:
        ax_isi.semilogx(rec.isi_centers, rec.isi_hist,
                        color=_waveform_color(rec.stage),
                        linewidth=0.7, alpha=0.6)
    ax_isi.set_xlabel("ISI (s, log)"); ax_isi.set_ylabel("prob")
    ax_isi.set_title("ISI distribution", fontsize=10)

    ax_fr = fig.add_subplot(gs[1, 1])
    xs = np.arange(len(uid.sessions))
    colors = [_waveform_color(r.stage) for r in uid.sessions]
    ax_fr.scatter(xs, [r.baseline_fr_hz for r in uid.sessions], c=colors, s=18)
    ax_fr.plot(xs, [r.baseline_fr_hz for r in uid.sessions], color="0.5", linewidth=0.7)
    ax_fr.set_xlabel("session #"); ax_fr.set_ylabel("FR (Hz)")
    ax_fr.set_title("Baseline FR", fontsize=10)

    # Row 2: Baseline_ON heatmap (full width)
    _draw_heatmap_with_inset(
        gs[2, :], uid, "baseline_on",
        title="PSTH · Baseline_ON · all outcomes pooled [TODO: split by outcome in v2]",
    )

    # Row 3: Change_ON Big-Hit | Small-Hit
    _draw_heatmap_with_inset(
        gs[3, 0], uid, "change_on_big_hit",
        title="Change_ON · Big-Hit (2.0× + 4.0×)",
        miss_keys=["change_on_big_miss"],
    )
    _draw_heatmap_with_inset(
        gs[3, 1], uid, "change_on_sm_hit",
        title="Change_ON · Small-Hit (1.25× + 1.35×)",
        miss_keys=["change_on_sm_miss"],
    )

    # Row 4: Hit lick (full width)
    _draw_heatmap_with_inset(
        gs[4, :], uid, "hit_lick",
        title="PSTH · Hit lick",
    )

    return fig
```

- [ ] **Step 2: Commit**

```
git add scripts/pipelines/tracking/qc_sheet_figures.py
git commit -m "Add page-2 renderer (ISI/FR row + 4 PSTH heatmaps with stage insets)"
```

---

## Task 13: Two-page PDF writer

**Files:**
- Modify: `scripts/pipelines/tracking/qc_sheet_figures.py`

- [ ] **Step 1: Implement PDF writer**

Append to `scripts/pipelines/tracking/qc_sheet_figures.py`:

```python
def write_uid_pdf(out_path: Path, uid: UIDIntermediate,
                  um_pair_scores: Optional[np.ndarray],
                  isi_score: float, depth_std: float,
                  wave_corr: float, fr_cv_val: float) -> str:
    """Write the 2-page PDF; return the composite verdict string."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        f1 = render_page1(uid, um_pair_scores, isi_score, depth_std, wave_corr, fr_cv_val)
        pdf.savefig(f1); plt.close(f1)
        f2 = render_page2(uid, isi_score, depth_std, wave_corr, fr_cv_val)
        pdf.savefig(f2); plt.close(f2)
    # Re-run the composite using the same inputs (cheap; keeps the API tidy)
    from visdetect.analysis.tracking_qc import (
        badge_isi, badge_depth, badge_waveform, badge_fr, composite_verdict,
    )
    return composite_verdict([
        badge_isi(isi_score), badge_depth(depth_std),
        badge_waveform(wave_corr), badge_fr(fr_cv_val),
    ])
```

- [ ] **Step 2: Commit**

```
git add scripts/pipelines/tracking/qc_sheet_figures.py
git commit -m "Add 2-page PDF writer for QC sheets"
```

---

## Task 14: UM pair-score loader

**Files:**
- Modify: `src/visdetect/analysis/tracking_qc.py`

The probability matrix at `batch0/output_prob_matrix.npy` is indexed by row-order in `batch0/unit_index.csv`. We need consecutive-session match probabilities for a given UID.

- [ ] **Step 1: Implement the loader**

Append to `src/visdetect/analysis/tracking_qc.py`:

```python
def load_um_pair_scores(um_output_root, uid_to_sessions: Dict[int, List[str]],
                         uid_to_ks: Dict[int, Dict[str, int]]
                         ) -> Dict[int, np.ndarray]:
    """Read batch0/output_prob_matrix.npy + batch0/unit_index.csv, then
    return per-UID arrays of consecutive-session match probabilities.

    Parameters
    ----------
    um_output_root : Path
        e.g. ``X:/.../unit_match/output/all42``
    uid_to_sessions : dict[uid -> chronological list of session names (strings)]
    uid_to_ks : dict[uid -> dict[session_name -> ks_unit_id]]

    Returns
    -------
    dict[uid] -> ndarray of shape (n_sessions_for_uid - 1,)
        Empty array if matrix or rows are missing.
    """
    root = Path(um_output_root)
    matrix_path = root / "batch0" / "output_prob_matrix.npy"
    index_path  = root / "batch0" / "unit_index.csv"
    if not matrix_path.exists() or not index_path.exists():
        return {uid: np.array([]) for uid in uid_to_sessions}

    mat = np.load(matrix_path)
    idx = pd.read_csv(index_path)
    idx["session"] = idx["session"].astype(str)
    lookup: Dict[Tuple[str, int], int] = {}
    for i, row in idx.iterrows():
        lookup[(str(row["session"]), int(row["ks_unit_id"]))] = i

    out = {}
    for uid, sess_list in uid_to_sessions.items():
        ks_map = uid_to_ks.get(uid, {})
        rows = []
        for s in sess_list:
            kid = ks_map.get(s)
            if kid is None:
                rows.append(None)
                continue
            rows.append(lookup.get((s, int(kid))))
        scores = []
        for a, b in zip(rows[:-1], rows[1:]):
            if a is None or b is None:
                scores.append(np.nan)
                continue
            scores.append(float(mat[a, b]))
        out[uid] = np.array(scores, dtype=float)
    return out
```

- [ ] **Step 2: Commit**

```
git add src/visdetect/analysis/tracking_qc.py
git commit -m "Add UM consecutive-pair match-probability loader"
```

---

## Task 15: CLI driver — build_qc_sheets.py

**Files:**
- Create: `scripts/pipelines/tracking/build_qc_sheets.py`

- [ ] **Step 1: Implement the driver**

```python
#!/usr/bin/env python3
"""Build per-UID QC sheets for the UnitMatch long-track cohort.

See docs/superpowers/specs/2026-05-21-tracking-qc-sheets-design.md and
docs/superpowers/plans/2026-05-22-tracking-qc-sheets-plan.md.

Usage:
    py scripts/pipelines/tracking/build_qc_sheets.py \\
        [--rebuild-cache] [--uids 334 1294 600] [--max-uids N]
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from visdetect.analysis.tracking_qc import (        # noqa: E402
    UIDIntermediate, SessionRecord,
    select_long_tracks, annotate_naive_to_expert,
    extract_session_records, load_channel_positions,
    load_isi_scores, load_um_pair_scores,
    depth_std_um, waveform_corr, fr_cv,
    badge_isi, badge_depth, badge_waveform, badge_fr, composite_verdict,
    save_cache, load_cache,
)
from visdetect.core.session import load_session                 # noqa: E402
from visdetect.suite.loader import load_staging_manifest        # noqa: E402

from qc_sheet_figures import write_uid_pdf                       # noqa: E402

UM_ROOT       = Path("X:/public/projects/BeJG_20230130_VisDetect/wEPhys/"
                     "BG_046/unit_match/output/all42")
UNIT_INDEX    = UM_ROOT / "unit_index.csv"
ISI_STATS     = REPO_ROOT / "FIGURES" / "tracking_qc" / "track_validation_stats.csv"
RAW_WF_ROOT   = REPO_ROOT / "data" / "unit_match" / "input" / "BG_046"
PKL_DIR       = REPO_ROOT / "data" / "pkls" / "BG_046"

OUT_DIR       = REPO_ROOT / "FIGURES" / "tracking_qc" / "per_uid_sheets"
VERDICTS_CSV  = REPO_ROOT / "FIGURES" / "tracking_qc" / "verdicts.csv"
CACHE_PATH    = REPO_ROOT / "data" / "cache" / "tracking_qc_intermediates.pkl"


def _session_pkl(session_name: str) -> Optional[Path]:
    for s in (session_name, session_name.zfill(8)):
        p = PKL_DIR / f"BG_046_{s}.pkl"
        if p.exists():
            return p
    return None


def build_cache(unit_index_df: pd.DataFrame, cohort: pd.DataFrame,
                manifest: pd.DataFrame) -> Dict[int, UIDIntermediate]:
    """Outer loop by session.  Returns dict[uid -> UIDIntermediate]."""
    # Build session -> stage map
    stage_by_session = {str(r["session_name"]): str(r["stage"])
                        for _, r in manifest.iterrows()}

    # Build uid -> {session -> ks_unit_id} map (only cohort UIDs)
    cohort_uids = set(cohort["global_uid"].astype(int).tolist())
    in_cohort = unit_index_df[unit_index_df["global_uid"].astype(int).isin(cohort_uids)].copy()
    in_cohort["session"] = in_cohort["session"].astype(str)
    uid_to_ks: Dict[int, Dict[str, int]] = {}
    for _, row in in_cohort.iterrows():
        uid = int(row["global_uid"])
        uid_to_ks.setdefault(uid, {})[str(row["session"])] = int(row["ks_unit_id"])

    # Initialise empty UIDIntermediate objects
    cohort = cohort.set_index("global_uid")
    intermediates: Dict[int, UIDIntermediate] = {}
    for uid in cohort_uids:
        row = cohort.loc[uid]
        intermediates[uid] = UIDIntermediate(
            global_uid=uid,
            span=int(row["span"]),
            has_naive_to_expert=bool(row["has_naive_to_expert"]),
            suspect_known=bool(row["suspect_known"]),
            sessions=[],
        )

    # Order the manifest chronologically (already done by load_staging_manifest)
    sessions_chrono = manifest["session_name"].astype(str).tolist()
    sess_set = sorted({s for ksmap in uid_to_ks.values() for s in ksmap.keys()},
                      key=lambda s: sessions_chrono.index(s) if s in sessions_chrono else 1e9)

    for sess in sess_set:
        pkl = _session_pkl(sess)
        if pkl is None:
            print(f"  skip {sess}: no pkl", flush=True); continue
        t0 = time.time()
        S = load_session(str(pkl))
        chan_pos = load_channel_positions(RAW_WF_ROOT, sess)
        # Which UIDs need extraction from this session?
        uids_here = [u for u, ks in uid_to_ks.items() if sess in ks]
        ks_ids_here = [uid_to_ks[u][sess] for u in uids_here]
        records = extract_session_records(
            S, ks_ids_here, session_name=sess,
            stage=stage_by_session.get(sess, "Learning"),
            raw_wf_root=RAW_WF_ROOT, channel_positions=chan_pos,
        )
        for u in uids_here:
            rec = records.get(int(uid_to_ks[u][sess]))
            if rec is not None:
                intermediates[u].sessions.append(rec)
        del S
        gc.collect()
        print(f"  {sess}: {len(records)}/{len(uids_here)} cached "
              f"in {time.time() - t0:.1f}s", flush=True)

    # Sort each UID's sessions chronologically by the manifest order
    order_idx = {s: i for i, s in enumerate(sessions_chrono)}
    for uid in intermediates:
        intermediates[uid].sessions.sort(
            key=lambda r: order_idx.get(r.session_name, 1e9)
        )
    return intermediates


def compute_uid_metrics(uid: UIDIntermediate) -> Dict[str, float]:
    """Depth std, waveform corr, FR CV for one UID across its sessions."""
    depths = np.array([r.peak_depth_um for r in uid.sessions], dtype=float)
    rates  = np.array([r.baseline_fr_hz for r in uid.sessions], dtype=float)
    # Stack peak waveforms; pad to common length if any vary (they shouldn't)
    waves = [r.waveform_peak for r in uid.sessions if r.waveform_peak is not None]
    if waves:
        min_len = min(w.size for w in waves)
        wf_stack = np.stack([w[:min_len] for w in waves])
    else:
        wf_stack = np.zeros((0, 0), dtype=np.float32)
    return {
        "depth_std_um": depth_std_um(depths),
        "wave_corr":    waveform_corr(wf_stack),
        "fr_cv":        fr_cv(rates),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--uids", type=int, nargs="*", default=None,
                        help="Only render these UIDs (cohort filter still applies)")
    parser.add_argument("--max-uids", type=int, default=None,
                        help="Render at most N UIDs (debug)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading manifest + cohort ...", flush=True)
    manifest = load_staging_manifest(qc_only=True, apply_filter=True)
    unit_index_df = pd.read_csv(UNIT_INDEX)
    cohort = select_long_tracks(UNIT_INDEX, ISI_STATS, min_span=10)
    cohort = annotate_naive_to_expert(cohort, manifest)
    print(f"  cohort size: {len(cohort)}", flush=True)

    if args.rebuild_cache or not CACHE_PATH.exists():
        print("Building cache (this is slow — outer loop by session) ...", flush=True)
        intermediates = build_cache(unit_index_df, cohort, manifest)
        save_cache(intermediates, CACHE_PATH)
        print(f"  saved cache to {CACHE_PATH}", flush=True)
    else:
        print(f"Loading cached intermediates from {CACHE_PATH}", flush=True)
        intermediates = load_cache(CACHE_PATH)

    # Pair-score loader
    uid_to_sessions = {u: [r.session_name for r in iv.sessions]
                       for u, iv in intermediates.items()}
    uid_to_ks = {}
    for _, row in unit_index_df.iterrows():
        uid = int(row["global_uid"])
        uid_to_ks.setdefault(uid, {})[str(row["session"])] = int(row["ks_unit_id"])
    pair_scores = load_um_pair_scores(UM_ROOT, uid_to_sessions, uid_to_ks)

    isi_scores = load_isi_scores(ISI_STATS)

    # Render each UID
    rows = []
    uids_to_render = sorted(intermediates)
    if args.uids:
        uids_to_render = [u for u in uids_to_render if u in set(args.uids)]
    if args.max_uids:
        uids_to_render = uids_to_render[: args.max_uids]
    print(f"Rendering {len(uids_to_render)} UIDs ...", flush=True)

    for uid in uids_to_render:
        iv = intermediates[uid]
        if not iv.sessions:
            print(f"  uid {uid}: no sessions extracted, skipping"); continue
        metrics = compute_uid_metrics(iv)
        isi = isi_scores[uid]
        out_path = OUT_DIR / f"uid_{uid:04d}.pdf"
        verdict = write_uid_pdf(
            out_path, iv, pair_scores.get(uid),
            isi_score=isi,
            depth_std=metrics["depth_std_um"],
            wave_corr=metrics["wave_corr"],
            fr_cv_val=metrics["fr_cv"],
        )
        rows.append({
            "global_uid": uid,
            "span": iv.span,
            "sessions": ";".join(r.session_name for r in iv.sessions),
            "has_naive_to_expert": iv.has_naive_to_expert,
            "suspect_known": iv.suspect_known,
            "isi_median": isi,
            "depth_std_um": metrics["depth_std_um"],
            "wave_corr": metrics["wave_corr"],
            "fr_cv": metrics["fr_cv"],
            "badge_isi":   badge_isi(isi),
            "badge_depth": badge_depth(metrics["depth_std_um"]),
            "badge_wave":  badge_waveform(metrics["wave_corr"]),
            "badge_fr":    badge_fr(metrics["fr_cv"]),
            "verdict": verdict,
        })
        print(f"  uid {uid}: {verdict}", flush=True)

    pd.DataFrame(rows).to_csv(VERDICTS_CSV, index=False)
    print(f"Wrote {VERDICTS_CSV}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Commit**

```
git add scripts/pipelines/tracking/build_qc_sheets.py
git commit -m "Add build_qc_sheets CLI driver"
```

---

## Task 16: End-to-end smoke test on top 3 UIDs

- [ ] **Step 1: Run on a small subset**

```
py scripts/pipelines/tracking/build_qc_sheets.py --rebuild-cache --uids 334 1294 600
```

Expected (concrete):
- Console: `cohort size: 61` (give or take if span boundaries shift), then per-session lines like `23062025: 1/1 cached in 4.3s`.
- Output: `FIGURES/tracking_qc/per_uid_sheets/uid_0334.pdf`, `uid_1294.pdf`, `uid_0600.pdf`.
- Output: `FIGURES/tracking_qc/verdicts.csv` with 3 rows.
- UID 334's verdict should be `trusted` or `review` (ISI 0.91 is well above pass).
- UID 779 should NOT be in the output (not in the `--uids` filter).

- [ ] **Step 2: Open the PDFs and verify**

Visual checklist for each PDF:
- Page 1: header has 4 badges + composite tag, stage stripe at bottom, footprints render as visible spikes, waveform overlay shows traces colored by stage, depth and amplitude scatter+line plots have one dot per session, UM pair-score bars present.
- Page 2: ISI overlay shows log-x axis with ~20–30 traces, baseline FR has one dot per session, Baseline_ON heatmap title shows the TODO marker, BigHit and SmallHit heatmaps are side-by-side, Hit lick heatmap at bottom, each heatmap has a stage-mean inset in the upper-right.
- No matplotlib errors in console.

- [ ] **Step 3: Run on the suspects to confirm they flag**

```
py scripts/pipelines/tracking/build_qc_sheets.py --uids 779 873 872
```

Expected: all three verdicts come back `suspect` (ISI badge fails on each).

- [ ] **Step 4: Run the full cohort**

```
py scripts/pipelines/tracking/build_qc_sheets.py
```

Expected: 61 PDFs in `FIGURES/tracking_qc/per_uid_sheets/`, one verdicts.csv row each, total runtime dominated by the first cache build (~5–20 min for 42 sessions × ~15 UIDs/session). Subsequent runs reuse the cache and finish in 2–3 min for figures.

- [ ] **Step 5: Spot-check verdicts.csv**

```
py -c "import pandas as pd; df = pd.read_csv('FIGURES/tracking_qc/verdicts.csv'); print(df['verdict'].value_counts()); print(df[df.global_uid.isin([334,1294,600,511,177,779,873,872])])"
```

Expected:
- N→E anchors 334/1294/600/511/177 land in `trusted` or `review`.
- Suspects 779/873/872 land in `suspect`.
- Distribution roughly matches the ISI baseline (~40 `trusted`, ~15 `review`, ~6 `suspect`).

- [ ] **Step 6: Commit any small fixes that came out of the smoke test**

```
git add -u
git commit -m "Smoke-test fixes for QC sheet builder"
```

(If the smoke test ran cleanly, skip this commit.)

---

## Self-review checklist

Spec sections covered:

| Spec § | Implemented in |
|---|---|
| §1 purpose | Task 16 produces the per-UID PDFs with composite verdicts |
| §2 cohort (61 UIDs, span ≥ 10, N→E flag, suspects) | Task 9 (cohort), Task 15 (annotation), Task 8 (loop) |
| §3 architecture / canonical imports | Tasks 1–15 (all imports from `visdetect.*`) |
| §4 pooling + event/outcome rules | Task 7 `PSTH_CONDITIONS` |
| §5 page-1 panel inventory (8) | Task 11 `render_page1` |
| §5 page-2 panel inventory (7) | Task 12 `render_page2` |
| §6 panel proportions | Tasks 11, 12 (gridspec `height_ratios`/`width_ratios`) |
| §7 metrics + verdict | Tasks 2, 3, 4 (metrics + badges + composite) |
| §8 outputs (PDFs + verdicts.csv + cache) | Tasks 13, 15 |
| §9 v2 TODOs | Encoded in titles (Task 12) + spec, no code yet — correct |
| §10 non-goals | Respected — no DeepUM logic, no within-cell learning code |

Open items I'm aware of:
- The "UM pairwise centroid_dist" panel name in the spec was relaxed to "match probability" because UM only persists the probability matrix at `batch0/output_prob_matrix.npy`. Task 11's title says "match probability", which is the truer label. If you want raw centroid distances later, that's a v2 addition.
- `_compute_baseline_fr` uses total-spikes / total-time as a robust per-session FR proxy rather than averaging per-trial pre-stimulus windows. This is faster and stable; the spec didn't pin a specific definition.

No placeholders, no "fill in later" steps, every code block in the plan is executable as written.
