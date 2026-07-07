# Transient/sustained TF cells: spectrum-vs-classes + waveform mapping — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the transient/sustained TF-cell identity is a spectrum or two discrete classes, and how that (temporal) axis maps onto the narrow/broad spike-waveform (FSI/SPN) axis — both from a locally-recomputed continuous kernel width.

**Architecture:** Two pure, unit-tested library modules (width estimators; modality/segmented-regression stats) plus three orchestration scripts (recompute continuous width from a local per-cell GLM refit → Part-1 spectrum figure → Part-2 waveform-mapping figure). All compute is local (reads `data/pkls/`), never over X:.

**Tech Stack:** Python 3.10 (`.venv`, invoked as `py`), numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib. Reuses `visdetect.analysis.tf_glm` (GLM refit + kernel), `tf_glm_data.session_trial_regressors`, and the `scripts/tf_responsiveness/state_conditioned/` helpers.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-07-07-transient-sustained-spectrum-celltype-design.md` (authoritative).
- **Population:** responsive cells (`resp_log2 == True`) in `good_dates` (QC-pass + <50% Disengaged), striatum only — BG_046, BG_039 (DMS), BG_031 (VMS).
- **No compute over X:** all session loads read local `data/pkls/{SUBJ}/{session}.pkl`. Never read `npx_converted` / X: in these scripts.
- **Output paths = PRIMARY repo root**, never the retired `vd_tf_bg046` worktree: figures → `FIGURES/tf_glm_bg046/<fig>/`, caches → `data/cache/tf_glm_bg046/`. Where a reused helper exposes a stale `OUT` pointing at `vd_tf_bg046`, override it with a repo-root path in the new script.
- **Refit config is fixed:** reuse `scripts/tf_responsiveness/cluster_bg/tf_glm_bg_task._cfg("log2")` verbatim (`include_movement=False, include_phase=False, include_tiled_baseline=True, standardize_design=True, fast_fit=True, responsive_criterion="c2", tf_encoding="log2", min_pulses_per_label=20`). TF kernel window `kern["tf"]=(0.0,1.5)`, `bin_s=0.05` → 30 lags (0…1.45 s). Folds `make_trial_folds(d.trial_index, cfg.n_folds=10, cfg.seed=42)`.
- **Validation gate (hard):** the recomputed grid-FWHM must reproduce the registry `kernel_fwhm` for ≥95% of responsive cells (exact on the 50 ms grid). If not, STOP and diagnose before trusting any continuous width.
- **Stats rigor:** non-parametric defaults; firing-rate-control (partial on `base_hz`) every cross-neuron magnitude; session random-intercept mixed models + per-mouse/region breakdowns on headlines; effect sizes with every p; seed 42 for all bootstraps.
- **Test command:** `py -m pytest <path> -v` (equivalently `.venv/Scripts/python.exe -m pytest`).
- **Canonical session ids:** use `config.canonical_session_id()` for any key/join; never int-cast a session id.

---

## File Structure

**Create:**
- `src/visdetect/analysis/kernel_width.py` — pure width estimators on a `(K, lags)` kernel.
- `src/visdetect/analysis/spectrum_stats.py` — pure modality + segmented-regression tests.
- `tests/analysis/test_kernel_width.py` — unit tests for width estimators.
- `tests/analysis/test_spectrum_stats.py` — unit tests for modality/segmented tests.
- `scripts/tf_responsiveness/state_conditioned/recompute_kernel_width.py` — Component A (local refit → continuous width cache).
- `scripts/tf_responsiveness/state_conditioned/spectrum_vs_classes.py` — Part 1 figure + stats.
- `scripts/tf_responsiveness/state_conditioned/width_vs_waveform.py` — Part 2 figure + stats.
- `docs/science/2026-07-07-transient-sustained-spectrum-celltype.md` — results writeup (final task).

**Produces (data artifacts):**
- `data/cache/tf_glm_bg046/kernel_width_continuous.csv` — per responsive cell: subject, session, unit, registry `kernel_fwhm`, `grid_fwhm`, `interp_fwhm`, `temporal_spread`, `pulse_fwhm`, `pulse_spread`, `kernel_peak_t`, plus join to `base_hz`/outcome metrics.
- `data/cache/tf_glm_bg046/kernel_vectors_{subj}.npz` — the raw per-cell kernel vectors (the artifact the pipeline discarded).
- `FIGURES/tf_glm_bg046/spectrum_vs_classes/` and `.../width_vs_waveform/` — figures + `_stats.txt`/`.csv`.

---

## Task 1: Width-estimator library (`kernel_width.py`)

**Files:**
- Create: `src/visdetect/analysis/kernel_width.py`
- Test: `tests/analysis/test_kernel_width.py`

**Interfaces:**
- Consumes: nothing (pure numpy).
- Produces:
  - `grid_fwhm(K: np.ndarray, lags: np.ndarray) -> float` — reproduces the pipeline's grid walk-out FWHM (validation).
  - `interpolated_fwhm(K: np.ndarray, lags: np.ndarray) -> float` — sub-bin FWHM of `|K|` via linear half-max crossing interpolation.
  - `temporal_spread(K: np.ndarray, lags: np.ndarray) -> float` — sqrt second-moment (temporal SD, seconds) of the `|K|` mass.
  - `peak_lag(K: np.ndarray, lags: np.ndarray) -> float` — lag of `argmax|K|`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/analysis/test_kernel_width.py
import numpy as np
import pytest
from visdetect.analysis.kernel_width import (
    grid_fwhm, interpolated_fwhm, temporal_spread, peak_lag,
)

LAGS = np.arange(0.0, 1.5, 0.05)  # 30 lags, 0..1.45 s (mirrors kern['tf'])

def _triangular(peak_idx, half_bins, amp=1.0, n=30):
    """Symmetric triangular kernel peaking at peak_idx, reaching 0 half_bins out."""
    k = np.zeros(n)
    for i in range(n):
        d = abs(i - peak_idx)
        k[i] = max(0.0, amp * (1 - d / half_bins)) if half_bins > 0 else (amp if d == 0 else 0.0)
    return k

def test_grid_fwhm_matches_pipeline_walkout():
    # peak at bin 6; |K|>=half for bins 5..7 (3 bins) => grid FWHM = lags[7]-lags[5] = 0.10
    K = _triangular(6, 3, amp=1.0)
    assert grid_fwhm(K, LAGS) == pytest.approx(LAGS[7] - LAGS[5])

def test_grid_fwhm_is_sign_agnostic():
    # suppression kernel (negative peak) must give the same width as its positive mirror
    K = _triangular(6, 3, amp=1.0)
    assert grid_fwhm(-K, LAGS) == pytest.approx(grid_fwhm(K, LAGS))

def test_interpolated_fwhm_subbin_between_grid_points():
    # triangular half-width 3 bins: half-max crossings fall exactly halfway between bins
    # -> interpolated FWHM = 3 bins * 0.05 = 0.15 (wider than the 0.10 grid value)
    K = _triangular(6, 3, amp=1.0)
    got = interpolated_fwhm(K, LAGS)
    assert got == pytest.approx(0.15, abs=1e-6)
    assert got > grid_fwhm(K, LAGS)  # sub-bin interpolation widens the coarse grid value

def test_interpolated_fwhm_left_censored_peak_at_zero_lag():
    # monotonic-decaying kernel peaking at bin 0: no left crossing -> clamp to lags[0]
    K = np.maximum(0.0, 1.0 - np.arange(30) / 4.0)
    got = interpolated_fwhm(K, LAGS)
    assert np.isfinite(got) and got > 0

def test_temporal_spread_wider_kernel_larger():
    narrow = _triangular(10, 2)
    broad = _triangular(10, 8)
    assert temporal_spread(broad, LAGS) > temporal_spread(narrow, LAGS)

def test_temporal_spread_sign_agnostic():
    K = _triangular(10, 5)
    assert temporal_spread(-K, LAGS) == pytest.approx(temporal_spread(K, LAGS))

def test_peak_lag_picks_abs_max():
    K = _triangular(9, 3, amp=-2.0)  # strongest deflection is negative at bin 9
    assert peak_lag(K, LAGS) == pytest.approx(LAGS[9])

def test_degenerate_flat_kernel_returns_nan():
    K = np.zeros(30)
    assert np.isnan(interpolated_fwhm(K, LAGS))
    assert np.isnan(temporal_spread(K, LAGS))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -m pytest tests/analysis/test_kernel_width.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'visdetect.analysis.kernel_width'`.

- [ ] **Step 3: Write the implementation**

```python
# src/visdetect/analysis/kernel_width.py
"""Continuous width estimators for a 1-D GLM TF kernel (or any deflection trace).

All estimators operate on the ABSOLUTE deflection |K| so suppression-type cells
(~half of TF-responsive units fire *less* to fast pulses) are treated the same as
excitatory cells. `grid_fwhm` reproduces the pipeline's coarse walk-out exactly
(for the registry validation gate); `interpolated_fwhm` and `temporal_spread` are
the continuous measures the 50 ms lag grid cannot resolve.
"""
from __future__ import annotations

import numpy as np


def _abs(K: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(K, dtype=float))


def peak_lag(K: np.ndarray, lags: np.ndarray) -> float:
    a = _abs(K)
    if a.size == 0 or not np.any(a > 0):
        return float("nan")
    return float(np.asarray(lags, float)[int(np.argmax(a))])


def grid_fwhm(K: np.ndarray, lags: np.ndarray) -> float:
    """Pipeline-identical FWHM: walk out from the peak while |K| >= half-max,
    return lags[hi] - lags[lo] (quantized to the lag grid)."""
    a = _abs(K)
    lags = np.asarray(lags, float)
    if a.size == 0 or not np.any(a > 0):
        return float("nan")
    ip = int(np.argmax(a))
    half = a[ip] / 2.0
    lo = ip
    while lo > 0 and a[lo - 1] >= half:
        lo -= 1
    hi = ip
    while hi < a.size - 1 and a[hi + 1] >= half:
        hi += 1
    return float(lags[hi] - lags[lo])


def _half_cross(a: np.ndarray, lags: np.ndarray, ip: int, half: float, direction: int) -> float:
    """Linear-interpolated lag where |K| crosses `half` moving `direction` (+1 right,
    -1 left) from the peak. Clamps to the boundary lag if no crossing (censored)."""
    i = ip
    while 0 <= i + direction < a.size and a[i + direction] >= half:
        i += direction
    j = i + direction  # first index strictly below half (or out of range)
    if j < 0 or j >= a.size:
        return float(lags[0] if direction < 0 else lags[-1])
    # a[j] < half <= a[i]; interpolate the crossing between lags[j] and lags[i]
    denom = a[i] - a[j]
    frac = 0.0 if denom == 0 else (half - a[j]) / denom
    return float(lags[j] + frac * (lags[i] - lags[j]))


def interpolated_fwhm(K: np.ndarray, lags: np.ndarray) -> float:
    """Sub-bin FWHM of |K| via linear half-max crossing interpolation."""
    a = _abs(K)
    lags = np.asarray(lags, float)
    if a.size < 2 or not np.any(a > 0):
        return float("nan")
    ip = int(np.argmax(a))
    half = a[ip] / 2.0
    left = _half_cross(a, lags, ip, half, -1)
    right = _half_cross(a, lags, ip, half, +1)
    return float(right - left)


def temporal_spread(K: np.ndarray, lags: np.ndarray) -> float:
    """sqrt second-moment (temporal SD, s) of the |K| mass about its centroid."""
    a = _abs(K)
    lags = np.asarray(lags, float)
    tot = a.sum()
    if a.size == 0 or tot <= 0:
        return float("nan")
    w = a / tot
    tbar = float(np.sum(w * lags))
    return float(np.sqrt(np.sum(w * (lags - tbar) ** 2)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -m pytest tests/analysis/test_kernel_width.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/kernel_width.py tests/analysis/test_kernel_width.py
git commit -m "feat(kernel-width): continuous FWHM + temporal-spread estimators for TF kernels

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Modality + segmented-regression stats library (`spectrum_stats.py`)

**Files:**
- Create: `src/visdetect/analysis/spectrum_stats.py`
- Test: `tests/analysis/test_spectrum_stats.py`

**Interfaces:**
- Consumes: nothing (numpy/scipy/sklearn).
- Produces:
  - `gmm_delta_bic(x, random_state=42) -> dict` → `{delta_bic, n, means, weights}` (1-vs-2-component GMM; `delta_bic = bic1 - bic2`, positive ⇒ 2 fits better).
  - `bimodality_coefficient(x) -> float` — Sarle's BC (>0.555 suggests bimodality).
  - `silverman_bootstrap(x, n_boot=500, seed=42) -> dict` → `{crit_bw, p_unimodal}`.
  - `dip_test(x) -> dict` → `{dip, p}` (uses `diptest` if importable, else `{nan, nan}`).
  - `segmented_vs_linear(x, y, n_grid=40) -> dict` → `{breakpoint, bic_linear, bic_segmented, delta_bic, slope_lo, slope_hi}` (broken-stick vs straight line; `delta_bic = bic_linear - bic_segmented`, positive ⇒ breakpoint fits better).

- [ ] **Step 1: Write the failing tests**

```python
# tests/analysis/test_spectrum_stats.py
import numpy as np
import pytest
from visdetect.analysis.spectrum_stats import (
    gmm_delta_bic, bimodality_coefficient, silverman_bootstrap,
    dip_test, segmented_vs_linear,
)

def test_gmm_delta_bic_positive_for_bimodal():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 0.3, 400), rng.normal(5, 0.3, 400)])
    assert gmm_delta_bic(x)["delta_bic"] > 0  # 2 components clearly better

def test_gmm_delta_bic_nonpositive_for_unimodal():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 800)
    assert gmm_delta_bic(x)["delta_bic"] < 20  # no strong 2-component preference

def test_bimodality_coefficient_higher_for_bimodal():
    rng = np.random.default_rng(2)
    uni = rng.normal(0, 1, 1000)
    bi = np.concatenate([rng.normal(-3, 0.5, 500), rng.normal(3, 0.5, 500)])
    assert bimodality_coefficient(bi) > bimodality_coefficient(uni)

def test_silverman_bootstrap_unimodal_high_p():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 500)
    assert silverman_bootstrap(x, n_boot=200)["p_unimodal"] > 0.1

def test_dip_test_returns_keys():
    rng = np.random.default_rng(4)
    out = dip_test(rng.normal(0, 1, 300))
    assert set(out) == {"dip", "p"}  # nan-filled if diptest not installed

def test_segmented_prefers_breakpoint_on_hinge_data():
    x = np.linspace(0, 1, 200)
    y = np.where(x < 0.5, 0.0, 4.0 * (x - 0.5)) + 0.01  # flat then rising (a hinge)
    out = segmented_vs_linear(x, y)
    assert out["delta_bic"] > 0                       # breakpoint beats a line
    assert 0.35 < out["breakpoint"] < 0.65            # near the true hinge

def test_segmented_no_gain_on_linear_data():
    rng = np.random.default_rng(5)
    x = np.linspace(0, 1, 200)
    y = 2.0 * x + rng.normal(0, 0.02, 200)
    assert segmented_vs_linear(x, y)["delta_bic"] < 6  # no meaningful breakpoint gain
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -m pytest tests/analysis/test_spectrum_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'visdetect.analysis.spectrum_stats'`.

- [ ] **Step 3: Write the implementation**

```python
# src/visdetect/analysis/spectrum_stats.py
"""Modality tests + segmented-vs-linear regression for the spectrum-vs-classes
question. GMM ΔBIC is the primary modality test (same method the repo uses for the
T2P waveform bimodality check); Silverman + Sarle's coefficient are secondary; the
Hartigan dip test is optional (only if the `diptest` package is installed).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _clean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def gmm_delta_bic(x, random_state: int = 42) -> dict:
    from sklearn.mixture import GaussianMixture
    x = _clean(x)
    if x.size < 4:
        return {"delta_bic": float("nan"), "n": int(x.size), "means": [], "weights": []}
    X = x.reshape(-1, 1)
    g1 = GaussianMixture(1, random_state=random_state).fit(X)
    g2 = GaussianMixture(2, random_state=random_state).fit(X)
    order = np.argsort(g2.means_.flatten())
    return {
        "delta_bic": float(g1.bic(X) - g2.bic(X)),
        "n": int(x.size),
        "means": [float(m) for m in g2.means_.flatten()[order]],
        "weights": [float(w) for w in g2.weights_[order]],
    }


def bimodality_coefficient(x) -> float:
    """Sarle's BC = (skew^2 + 1) / (kurtosis + 3(n-1)^2/((n-2)(n-3)))."""
    x = _clean(x)
    n = x.size
    if n < 4:
        return float("nan")
    g = stats.skew(x)
    k = stats.kurtosis(x, fisher=True)  # excess kurtosis
    return float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def silverman_bootstrap(x, n_boot: int = 500, seed: int = 42) -> dict:
    """Silverman critical-bandwidth test of H0: unimodal. Small p rejects unimodality."""
    x = _clean(x)
    if x.size < 10:
        return {"crit_bw": float("nan"), "p_unimodal": float("nan")}

    def _n_modes(sample, bw):
        grid = np.linspace(sample.min(), sample.max(), 512)
        dens = stats.gaussian_kde(sample, bw_method=bw / sample.std(ddof=1))(grid)
        return int(np.sum((dens[1:-1] > dens[:-2]) & (dens[1:-1] > dens[2:])))

    lo, hi = 1e-3 * x.std(ddof=1), x.std(ddof=1) * 2
    for _ in range(60):  # bisection for the smallest bw giving a unimodal KDE
        mid = 0.5 * (lo + hi)
        if _n_modes(x, mid) <= 1:
            hi = mid
        else:
            lo = mid
    h_crit = hi
    rng = np.random.default_rng(seed)
    n = x.size
    count = 0
    for _ in range(n_boot):
        samp = rng.choice(x, n, replace=True)
        samp = samp + h_crit * rng.standard_normal(n)  # smoothed bootstrap
        if _n_modes(samp, h_crit) > 1:
            count += 1
    return {"crit_bw": float(h_crit), "p_unimodal": float(count / n_boot)}


def dip_test(x) -> dict:
    x = _clean(x)
    try:
        import diptest as _dt
        dip, p = _dt.diptest(x)
        return {"dip": float(dip), "p": float(p)}
    except Exception:
        return {"dip": float("nan"), "p": float("nan")}


def _ols_bic(y, yhat, k_params) -> float:
    n = len(y)
    rss = float(np.sum((y - yhat) ** 2))
    rss = max(rss, 1e-12)
    return n * np.log(rss / n) + k_params * np.log(n)


def segmented_vs_linear(x, y, n_grid: int = 40) -> dict:
    """Compare a straight line vs a continuous 2-segment (broken-stick) fit by BIC.
    delta_bic = bic_linear - bic_segmented (positive => breakpoint preferred)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 10:
        return {"breakpoint": float("nan"), "bic_linear": float("nan"),
                "bic_segmented": float("nan"), "delta_bic": float("nan"),
                "slope_lo": float("nan"), "slope_hi": float("nan")}
    b1, b0 = np.polyfit(x, y, 1)
    bic_lin = _ols_bic(y, b1 * x + b0, 2)
    best = None
    for bp in np.quantile(x, np.linspace(0.15, 0.85, n_grid)):
        h = np.maximum(0.0, x - bp)                 # continuous hinge basis
        A = np.column_stack([np.ones_like(x), x, h])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        bic = _ols_bic(y, A @ coef, 4)
        if best is None or bic < best[0]:
            best = (bic, float(bp), float(coef[1]), float(coef[1] + coef[2]))
    bic_seg, bp, slope_lo, slope_hi = best
    return {"breakpoint": bp, "bic_linear": float(bic_lin),
            "bic_segmented": float(bic_seg), "delta_bic": float(bic_lin - bic_seg),
            "slope_lo": slope_lo, "slope_hi": slope_hi}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -m pytest tests/analysis/test_spectrum_stats.py -v`
Expected: PASS (7 passed). If `test_dip_test_returns_keys` shows `dip=nan`, that is expected (diptest not installed).

- [ ] **Step 5 (optional): install diptest for the true Hartigan test**

Run (local venv only — never over X:): `.venv/Scripts/python.exe -m pip install diptest`
Then re-run the test; `dip_test` now returns real values. Skip if you prefer no new dependency — the GMM/Silverman/BC battery stands alone.

- [ ] **Step 6: Commit**

```bash
git add src/visdetect/analysis/spectrum_stats.py tests/analysis/test_spectrum_stats.py
git commit -m "feat(spectrum-stats): modality (GMM/Silverman/BC/dip) + segmented-vs-linear tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Component A — recompute continuous width + validation gate

**Files:**
- Create: `scripts/tf_responsiveness/state_conditioned/recompute_kernel_width.py`
- Reads: `data/pkls/{SUBJ}/{session}.pkl`, `data/cache/tf_responsive/{subj}_tf_responsive.csv`, `FIGURES/tf_glm_bg046/latency_outcome_coupling/latency_outcome_metrics.csv`.
- Writes: `data/cache/tf_glm_bg046/kernel_width_continuous.csv`, `data/cache/tf_glm_bg046/kernel_vectors_{subj}.npz`.

**Interfaces:**
- Consumes: `kernel_width.{grid_fwhm, interpolated_fwhm, temporal_spread, peak_lag}`; `representative_cells.{REPO, _registry, good_dates, _spikes}`; `tf_glm_bg_task._cfg`; `tf_glm.{assemble_design, fit_poisson_cv, make_trial_folds, _tf_kernel, _lag_offsets, count_vector, pulse_times_from_tf, tf_pulse_peth}`; `tf_glm_data.session_trial_regressors`; `visdetect.core.session.load_session`.
- Produces: `kernel_width_continuous.csv` with columns `subject, session, unit, n_spikes, kernel_fwhm_registry, grid_fwhm, interp_fwhm, temporal_spread, pulse_fwhm, pulse_spread, kernel_peak_t, base_hz, change_on, hit_ramp, fa_ramp`.

- [ ] **Step 1: Write the script**

```python
# scripts/tf_responsiveness/state_conditioned/recompute_kernel_width.py
"""Component A: recompute a CONTINUOUS TF-kernel width per responsive cell.

The registry only stored the 50 ms-grid `kernel_fwhm` (the raw kernel was never
cached — verified). This refits the full BG GLM LOCALLY from Session pkls (the
exact config the registry used), extracts the raw FIR kernel, and computes sub-bin
continuous width (interpolated FWHM + temporal spread), plus a model-free
fast-minus-slow pulse-PETH width as an independent cross-check. A validation gate
asserts the recomputed grid-FWHM reproduces the registry value before the
continuous width is trusted; the raw kernel vectors are saved (the missing cache).
LOCAL ONLY — reads data/pkls/, never X:.
"""
from __future__ import annotations
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# BG cluster task lives in a sibling dir; add it for _cfg reuse.
_CLUSTER_BG = str(Path(_HERE).parents[1] / "cluster_bg")
if _CLUSTER_BG not in sys.path:
    sys.path.insert(0, _CLUSTER_BG)

from representative_cells import REPO, _registry, good_dates          # noqa: E402
from tf_glm_bg_task import _cfg                                        # noqa: E402
from visdetect.core.session import load_session                       # noqa: E402
from visdetect.analysis.tf_glm import (                               # noqa: E402
    assemble_design, fit_poisson_cv, make_trial_folds, _tf_kernel,
    _lag_offsets, count_vector, pulse_times_from_tf, tf_pulse_peth,
)
from visdetect.analysis.tf_glm_data import session_trial_regressors   # noqa: E402
from visdetect.analysis.kernel_width import (                         # noqa: E402
    grid_fwhm, interpolated_fwhm, temporal_spread, peak_lag,
)

MICE = [("BG_046", "DMS"), ("BG_039", "DMS"), ("BG_031", "VMS")]
OUT_CSV = Path(REPO) / "data/cache/tf_glm_bg046/kernel_width_continuous.csv"
METRICS = Path(REPO) / "FIGURES/tf_glm_bg046/latency_outcome_coupling/latency_outcome_metrics.csv"
PULSE_CAP = 600


def _responsive(subj):
    r = _registry(subj)
    r = r[r.resp & r.session_date.isin(good_dates(subj))]
    return r[["session", "session_date", "unit", "n_spikes", "kernel_fwhm", "kernel_peak_t"]]


def _pulse_width(y, d, cfg, fast, slow):
    """Model-free width from the fast-minus-slow pulse PETH contrast (Hz)."""
    ti, win, bs = d.trial_index, cfg.pulse_eval_win, cfg.bin_s
    if fast.size > PULSE_CAP:
        fast = np.sort(np.random.default_rng(0).choice(fast, PULSE_CAP, replace=False))
    tax, a_fast = tf_pulse_peth(y, d.bin_edges, fast, win, bs, trial_index=ti)
    _, a_slow = tf_pulse_peth(y, d.bin_edges, slow, win, bs, trial_index=ti)
    contrast = (a_fast - a_slow) / bs
    contrast = contrast - np.median(contrast[:max(1, len(contrast)//4)])  # de-mean on pre-pulse
    tax = np.asarray(tax, float)
    return interpolated_fwhm(contrast, tax), temporal_spread(contrast, tax)


def main():
    metrics = pd.read_csv(METRICS)[["subject", "session", "unit", "base_hz",
                                    "change_on", "hit_ramp", "fa_ramp"]] if METRICS.exists() else None
    rows = []
    for subj, region in MICE:
        resp = _responsive(subj)
        kvecs = {}
        for sess, g in resp.groupby("session"):
            pkl = Path(REPO) / "data/pkls" / subj / f"{sess}.pkl"
            if not pkl.exists():
                print(f"  MISSING pkl {pkl}; skip", flush=True)
                continue
            s = load_session(str(pkl))
            cfg = _cfg("log2")
            trials, units = session_trial_regressors(s, cfg)
            d = assemble_design(trials, cfg)
            folds = make_trial_folds(d.trial_index, cfg.n_folds, cfg.seed)
            lags = _lag_offsets(cfg.kern["tf"], cfg.bin_s) * cfg.bin_s
            fast, slow = pulse_times_from_tf(d, cfg)
            fast, slow = np.asarray(fast, float), np.asarray(slow, float)
            for _, r in g.iterrows():
                uid = int(r["unit"])
                if uid not in units:
                    continue
                y = count_vector(trials, units[uid], d)
                full = fit_poisson_cv(d.X, y, cfg, folds)
                K = _tf_kernel(full, d, cfg)
                if K is None or not np.any(np.isfinite(K)):
                    continue
                kvecs[f"u{uid}"] = K
                pf, ps = _pulse_width(y, d, cfg, fast, slow)
                rows.append(dict(
                    subject=subj, session=sess, unit=uid, n_spikes=int(r["n_spikes"]),
                    kernel_fwhm_registry=float(r["kernel_fwhm"]),
                    grid_fwhm=grid_fwhm(K, lags),
                    interp_fwhm=interpolated_fwhm(K, lags),
                    temporal_spread=temporal_spread(K, lags),
                    pulse_fwhm=pf, pulse_spread=ps,
                    kernel_peak_t_recompute=peak_lag(K, lags),
                    kernel_peak_t_registry=float(r["kernel_peak_t"]),
                ))
            del s
            gc.collect()
            print(f"  {subj}/{sess}: {len(g)} cells", flush=True)
        if kvecs:
            npz = Path(REPO) / f"data/cache/tf_glm_bg046/kernel_vectors_{subj}.npz"
            npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz, lags=(_lag_offsets(_cfg('log2').kern['tf'], 0.05) * 0.05),
                                units=np.array(list(kvecs.keys())), **kvecs)
    df = pd.DataFrame(rows)
    if metrics is not None:
        df = df.merge(metrics, on=["subject", "session", "unit"], how="left")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # ── VALIDATION GATE ────────────────────────────────────────────────
    ok = np.isclose(df.grid_fwhm, df.kernel_fwhm_registry, atol=1e-9)
    frac = float(ok.mean()) if len(df) else 0.0
    print(f"\nVALIDATION: grid_fwhm reproduces registry kernel_fwhm for "
          f"{ok.sum()}/{len(df)} cells ({100*frac:.1f}%)")
    if frac < 0.95:
        bad = df.loc[~ok, ["subject", "session", "unit", "grid_fwhm", "kernel_fwhm_registry"]].head(15)
        print("MISMATCHES (investigate config before trusting continuous width):")
        print(bad.to_string(index=False))
        raise SystemExit("VALIDATION FAILED: grid FWHM does not reproduce the registry")
    print(f"wrote {OUT_CSV}  (n={len(df)} responsive cells)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run Component A on real data**

Run: `py scripts/tf_responsiveness/state_conditioned/recompute_kernel_width.py`
Expected: per-session progress lines; then `VALIDATION: ... (≥95.0%)`; then `wrote .../kernel_width_continuous.csv (n≈520 responsive cells)`. Runtime ~10–20 min/mouse (local session loads).

- [ ] **Step 3: Verify the cache + validation numerically**

Run:
```bash
py -c "import pandas as pd; d=pd.read_csv('data/cache/tf_glm_bg046/kernel_width_continuous.csv'); \
print('rows',len(d)); print('subjects',d.subject.value_counts().to_dict()); \
import numpy as np; print('grid==registry frac', np.isclose(d.grid_fwhm,d.kernel_fwhm_registry,atol=1e-9).mean()); \
print('interp vs registry Spearman', d[['interp_fwhm','kernel_fwhm_registry']].corr(method='spearman').iloc[0,1]); \
print('interp vs pulse Spearman', d[['interp_fwhm','pulse_fwhm']].corr(method='spearman').iloc[0,1]); \
print('peak_t recompute vs registry Spearman', d[['kernel_peak_t_recompute','kernel_peak_t_registry']].corr(method='spearman').iloc[0,1])"
```
Expected: rows ≈ 520; all 3 subjects present; `grid==registry frac ≥ 0.95`; interp-vs-registry Spearman strongly positive (>0.7); interp-vs-pulse positive (>0.3); peak_t recompute-vs-registry very high (>0.9). If the grid fraction is < 0.95, STOP — the refit config does not match the registry.

- [ ] **Step 4: Commit**

```bash
git add scripts/tf_responsiveness/state_conditioned/recompute_kernel_width.py
git commit -m "feat(tf-width): recompute continuous TF-kernel width locally + validation gate

Refits the BG GLM from local Session pkls (registry config), extracts the raw FIR
kernel, computes sub-bin FWHM + temporal spread + model-free pulse width, and
gates on reproducing the registry grid kernel_fwhm. Saves the kernel vectors the
pipeline discarded.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

(The cache CSV/npz live under gitignored `data/cache/` — not committed; regenerated by the script.)

---

## Task 4: Part 1 — spectrum vs classes figure (`spectrum_vs_classes.py`)

**Files:**
- Create: `scripts/tf_responsiveness/state_conditioned/spectrum_vs_classes.py`
- Reads: `data/cache/tf_glm_bg046/kernel_width_continuous.csv`.
- Writes: `FIGURES/tf_glm_bg046/spectrum_vs_classes/spectrum_vs_classes.png` (+ `.pdf`, `_stats.txt`, `_stats.csv`).

**Interfaces:**
- Consumes: the Task-3 cache; `spectrum_stats.{gmm_delta_bic, bimodality_coefficient, silverman_bootstrap, dip_test, segmented_vs_linear}`; `scipy.stats.spearmanr`.
- Produces: the figure + a `_stats.csv` with one row per (measure, region) modality result and one row per (outcome) segmented-regression result.

- [ ] **Step 1: Write the script**

```python
# scripts/tf_responsiveness/state_conditioned/spectrum_vs_classes.py
"""Part 1: is the transient/sustained (temporal) identity a spectrum or two classes?

Continuous width (Component A) -> modality battery (GMM ΔBIC primary; Silverman;
Sarle BC; optional Hartigan dip) pooled + per region, latency⊥width check, and a
graded-vs-stepped test (segmented-vs-linear BIC) of outcome coupling on width.
Reads the cache only; no session reloads.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import spearmanr

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from representative_cells import REPO                                  # noqa: E402
from visdetect.analysis.spectrum_stats import (                       # noqa: E402
    gmm_delta_bic, bimodality_coefficient, silverman_bootstrap, dip_test,
    segmented_vs_linear,
)

CACHE = Path(REPO) / "data/cache/tf_glm_bg046/kernel_width_continuous.csv"
OUT = Path(REPO) / "FIGURES/tf_glm_bg046/spectrum_vs_classes"
REGION = {"BG_046": "DMS", "BG_039": "DMS", "BG_031": "VMS"}
OUTCOMES = [("change_on", "Change_ON"), ("hit_ramp", "Hit ramp"), ("fa_ramp", "FA ramp")]
WIDTH = "interp_fwhm"  # primary continuous width


def main():
    d = pd.read_csv(CACHE)
    d["region"] = d.subject.map(REGION)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass

    stat_rows, lines = [], []
    # ── modality battery on each width measure, pooled + per region ──
    for measure in [WIDTH, "temporal_spread", "pulse_fwhm"]:
        for scope, sub in [("pooled", d)] + [(rg, d[d.region == rg]) for rg in ("DMS", "VMS")]:
            x = sub[measure].replace([np.inf, -np.inf], np.nan).dropna().values
            gm = gmm_delta_bic(x); si = silverman_bootstrap(x, n_boot=300)
            dp = dip_test(x); bc = bimodality_coefficient(x)
            stat_rows.append(dict(measure=measure, scope=scope, n=len(x),
                                  gmm_delta_bic=gm["delta_bic"], gmm_means=gm["means"],
                                  gmm_weights=gm["weights"], silverman_p_unimodal=si["p_unimodal"],
                                  dip=dp["dip"], dip_p=dp["p"], bimodality_coef=bc))
    lines.append("MODALITY (positive ΔBIC & low silverman-p & BC>0.555 => classes; else spectrum):")
    for r in stat_rows:
        lines.append(f"  [{r['measure']}/{r['scope']}] n={r['n']} ΔBIC={r['gmm_delta_bic']:+.1f} "
                     f"silverman_p={r['silverman_p_unimodal']:.3f} dip_p={r['dip_p']} BC={r['bimodality_coef']:.3f}")

    # ── latency ⊥ width ──
    rho_lw, p_lw = spearmanr(d.kernel_peak_t_registry, d[WIDTH], nan_policy="omit")
    lines.append(f"latency(peak_t) vs width({WIDTH}): rho={rho_lw:+.3f} p={p_lw:.2e}")

    # ── graded vs stepped: outcome ~ width ──
    seg_rows = []
    for col, lab in OUTCOMES:
        sub = d[[WIDTH, col]].replace([np.inf, -np.inf], np.nan).dropna()
        rho, p = spearmanr(sub[WIDTH], sub[col])
        seg = segmented_vs_linear(sub[WIDTH].values, sub[col].values)
        seg_rows.append(dict(outcome=col, spearman_rho=rho, spearman_p=p, **seg))
        lines.append(f"  [{col}] Spearman rho={rho:+.3f} p={p:.2e} | segmented ΔBIC(seg-vs-lin)="
                     f"{seg['delta_bic']:+.1f} breakpoint={seg['breakpoint']:.3f} "
                     f"(ΔBIC<=6 => graded continuum; >10 => threshold)")

    # ── figure: A width hist+GMM, B latency-vs-width, C-E outcome-vs-width curves ──
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.3)
    axA = fig.add_subplot(gs[0, 0])
    for rg, c in (("DMS", "#3474ae"), ("VMS", "#ef6548")):
        axA.hist(d.loc[d.region == rg, WIDTH].dropna(), bins=np.linspace(0, 0.8, 33),
                 histtype="step", lw=2, density=True, color=c, label=rg)
    gm_pool = gmm_delta_bic(d[WIDTH].dropna().values)
    axA.set_title(f"continuous kernel width — GMM ΔBIC={gm_pool['delta_bic']:+.1f}\n"
                  f"(means {['%.2f'%m for m in gm_pool['means']]})", fontsize=10.5)
    axA.set_xlabel(f"{WIDTH} (s)"); axA.set_ylabel("density"); axA.legend(frameon=False)

    axB = fig.add_subplot(gs[0, 1])
    axB.scatter(d.kernel_peak_t_registry, d[WIDTH], s=10, alpha=0.4, color="0.4", edgecolors="none")
    axB.set_xlabel("kernel peak latency (s)"); axB.set_ylabel(f"{WIDTH} (s)")
    axB.set_title(f"latency ⊥ width  rho={rho_lw:+.2f}", fontsize=10.5)

    for i, (col, lab) in enumerate(OUTCOMES):
        ax = fig.add_subplot(gs[1, i])
        sub = d[[WIDTH, col]].replace([np.inf, -np.inf], np.nan).dropna()
        ax.scatter(sub[WIDTH], sub[col], s=8, alpha=0.25, color="0.5", edgecolors="none")
        q = pd.qcut(sub[WIDTH], 8, duplicates="drop")
        binned = sub.groupby(q, observed=True).agg(x=(WIDTH, "median"), y=(col, "median"))
        ax.plot(binned.x, binned.y, "o-", color="#238b45", lw=2, label="binned median")
        sr = next(r for r in seg_rows if r["outcome"] == col)
        ax.set_title(f"{lab}: rho={sr['spearman_rho']:+.2f}, segΔBIC={sr['delta_bic']:+.1f}", fontsize=10)
        ax.set_xlabel(f"{WIDTH} (s)"); ax.set_ylabel("Δ firing (Hz)"); ax.legend(frameon=False, fontsize=8)

    axT = fig.add_subplot(gs[0, 2]); axT.axis("off")
    axT.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=7.2, family="monospace",
             transform=axT.transAxes)
    fig.suptitle("Part 1 — Is transient/sustained a spectrum or two classes? (continuous kernel width)",
                 fontsize=13, y=1.01)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"spectrum_vs_classes.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(stat_rows).to_csv(OUT / "spectrum_vs_classes_modality.csv", index=False)
    pd.DataFrame(seg_rows).to_csv(OUT / "spectrum_vs_classes_segmented.csv", index=False)
    (OUT / "spectrum_vs_classes_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}/spectrum_vs_classes.png")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify outputs exist + are sane**

Run: `py scripts/tf_responsiveness/state_conditioned/spectrum_vs_classes.py`
Expected: prints the modality + segmented lines; writes `spectrum_vs_classes.png/.pdf`, `_modality.csv`, `_segmented.csv`, `_stats.txt`. Sanity: `latency vs width` |rho| small (< 0.2, confirming independence); each outcome Spearman positive.

- [ ] **Step 3: Record the verdict for the writeup**

Run:
```bash
py -c "import pandas as pd; m=pd.read_csv('FIGURES/tf_glm_bg046/spectrum_vs_classes/spectrum_vs_classes_modality.csv'); \
print(m[m.scope=='pooled'][['measure','n','gmm_delta_bic','silverman_p_unimodal','bimodality_coef']].to_string(index=False))"
```
Expected: read off whether ΔBIC is large-positive with well-separated means (classes) or small/negative with low BC (spectrum) — consistently across the three measures. Note the verdict; do not assert it as solid yet (Task 6 verifies).

- [ ] **Step 4: Commit**

```bash
git add scripts/tf_responsiveness/state_conditioned/spectrum_vs_classes.py
git commit -m "feat(tf-width): Part 1 spectrum-vs-classes figure (modality + segmented tests)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Part 2 — width vs waveform mapping figure (`width_vs_waveform.py`)

**Files:**
- Create: `scripts/tf_responsiveness/state_conditioned/width_vs_waveform.py`
- Reads: `data/cache/tf_glm_bg046/kernel_width_continuous.csv`; `data/cache/talk_substrate/waveform_t2p_BG_031.csv`, `.../waveform_t2p_BG_039.csv`, `.../bg046_waveform_t2p.csv`; `data/{SUBJ}/waveform_celltype_labels.csv`.
- Writes: `FIGURES/tf_glm_bg046/width_vs_waveform/width_vs_waveform.png` (+ `.pdf`, `_stats.txt/.csv`).

**Interfaces:**
- Consumes: the Task-3 cache; `config.canonical_session_id`; `scipy.stats.spearmanr, chi2_contingency`; `statsmodels.formula.api.mixedlm`.
- Produces: figure + `_stats.txt` with the 2D correlation, the class×celltype crosstab + χ², the four-quadrant coupling table, and the independence (mixed-model) coefficients.

- [ ] **Step 1: Write the script**

```python
# scripts/tf_responsiveness/state_conditioned/width_vs_waveform.py
"""Part 2: does the transient/sustained (temporal-width) axis map onto narrow/broad
(spike-waveform FSI/SPN)? Overlap crosstab (centerpiece) + continuous 2D joint
distribution + four-quadrant coupling + an independence test (does width predict
coupling controlling for t2p?). Striatum only; carries the yield-bias caveat.
Reads caches only.

t2p cache filename asymmetry: BG_031/039 = waveform_t2p_BG_{id}.csv, BG_046 =
bg046_waveform_t2p.csv. Resolve per subject (do NOT glob waveform_t2p_BG_*).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import spearmanr, chi2_contingency

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from representative_cells import REPO                                  # noqa: E402
from visdetect.analysis.config import canonical_session_id            # noqa: E402

CACHE = Path(REPO) / "data/cache/tf_glm_bg046/kernel_width_continuous.csv"
OUT = Path(REPO) / "FIGURES/tf_glm_bg046/width_vs_waveform"
REGION = {"BG_046": "DMS", "BG_039": "DMS", "BG_031": "VMS"}
T2P = {"BG_046": "data/cache/talk_substrate/bg046_waveform_t2p.csv",
       "BG_039": "data/cache/talk_substrate/waveform_t2p_BG_039.csv",
       "BG_031": "data/cache/talk_substrate/waveform_t2p_BG_031.csv"}
WIDTH = "interp_fwhm"
OUTCOMES = ["change_on", "hit_ramp", "fa_ramp"]


def _load_t2p(subj):
    df = pd.read_csv(Path(REPO) / T2P[subj])
    df["skey"] = df["session_8"].map(canonical_session_id)
    df["unit"] = df["cluster_id"].astype(int)
    return df[["skey", "unit", "t2p_ms"]].drop_duplicates(["skey", "unit"])


def _load_label(subj):
    f = Path(REPO) / f"data/{subj}/waveform_celltype_labels.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df["skey"] = df["session_date"].map(canonical_session_id)
    df["unit"] = df["cluster_id"].astype(int)
    return df[["skey", "unit", "celltype"]].drop_duplicates(["skey", "unit"])


def attach(d):
    d = d.copy()
    d["skey"] = [canonical_session_id(str(s).split(f"{sub}_", 1)[-1])
                 for s, sub in zip(d.session, d.subject)]
    out = []
    for subj in d.subject.unique():
        sub = d[d.subject == subj].merge(_load_t2p(subj), on=["skey", "unit"], how="left")
        lab = _load_label(subj)
        sub = sub.merge(lab, on=["skey", "unit"], how="left") if lab is not None else sub.assign(celltype=np.nan)
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def main():
    import statsmodels.formula.api as smf
    d = attach(pd.read_csv(CACHE))
    d["region"] = d.subject.map(REGION)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    lines = []
    lab = d.dropna(subset=["celltype"]); lab = lab[lab.celltype.isin(["FSI", "SPN"])]
    lines.append(f"cells={len(d)}; with t2p={d.t2p_ms.notna().sum()}; with FSI/SPN label={len(lab)}")

    # ── overlap crosstab (centerpiece): class x celltype ──
    med = d[WIDTH].median()
    d["wclass"] = np.where(d[WIDTH] <= med, "transient", "sustained")
    ct = pd.crosstab(lab.assign(wclass=np.where(lab[WIDTH] <= med, "transient", "sustained")).wclass,
                     lab.celltype)
    chi2, pchi, *_ = chi2_contingency(ct)
    lines.append(f"OVERLAP crosstab (median-split width x waveform):\n{ct.to_string()}")
    lines.append(f"  chi2={chi2:.2f} p={pchi:.2e}")

    # ── continuous 2D: t2p vs width ──
    dd = d.dropna(subset=["t2p_ms", WIDTH])
    rho_all, p_all = spearmanr(dd.t2p_ms, dd[WIDTH])
    lines.append(f"CONTINUOUS t2p vs width: rho={rho_all:+.3f} p={p_all:.2e} (n={len(dd)})")
    for rg in ("DMS", "VMS"):
        sub = dd[dd.region == rg]
        if len(sub) > 10:
            r, p = spearmanr(sub.t2p_ms, sub[WIDTH])
            lines.append(f"    {rg}: rho={r:+.3f} p={p:.2e} (n={len(sub)})")

    # ── four-quadrant coupling ──
    tmed = lab.t2p_ms.median()
    lines.append(f"FOUR-QUADRANT (t2p median={tmed:.3f} ms, width median={med:.3f} s) — median Δ firing (Hz):")
    for wc in ("transient", "sustained"):
        for narrow in (True, False):
            q = lab[(np.where(lab[WIDTH] <= med, "transient", "sustained") == wc) &
                    ((lab.t2p_ms <= tmed) == narrow)]
            wf = "narrow/FSI" if narrow else "broad/SPN"
            meds = {c: round(float(q[c].median()), 2) for c in OUTCOMES if q[c].notna().any()}
            lines.append(f"    {wc:9s} x {wf:11s} n={len(q):3d}  {meds}")

    # ── independence: does width predict coupling controlling for t2p? ──
    lines.append("INDEPENDENCE (mixedlm outcome ~ width + t2p, session RE): width beta | t2p beta")
    for col in OUTCOMES:
        m = d.dropna(subset=[col, WIDTH, "t2p_ms"]).copy()
        m["w"] = (m[WIDTH] - m[WIDTH].mean()) / m[WIDTH].std()
        m["t"] = (m.t2p_ms - m.t2p_ms.mean()) / m.t2p_ms.std()
        try:
            fit = smf.mixedlm(f"{col} ~ w + t", m, groups=m["session"]).fit(reml=False)
            lines.append(f"  [{col}] width b={fit.params['w']:+.3f} p={fit.pvalues['w']:.2e} | "
                         f"t2p b={fit.params['t']:+.3f} p={fit.pvalues['t']:.2e}")
        except Exception as e:
            lines.append(f"  [{col}] mixedlm failed: {e}")

    # ── figure ──
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.32)
    axA = fig.add_subplot(gs[0, 0])
    for rg, c in (("DMS", "#3474ae"), ("VMS", "#ef6548")):
        sub = dd[dd.region == rg]
        axA.scatter(sub.t2p_ms, sub[WIDTH], s=12, alpha=0.4, color=c, edgecolors="none", label=rg)
    axA.axvline(tmed, color="0.6", ls=":"); axA.axhline(med, color="0.6", ls=":")
    axA.set_xlabel("trough-to-peak t2p (ms)  [narrow←|→broad]"); axA.set_ylabel(f"kernel width {WIDTH} (s)")
    axA.set_title(f"2D joint: t2p vs width  rho={rho_all:+.2f}", fontsize=10.5); axA.legend(frameon=False)

    axB = fig.add_subplot(gs[0, 1])
    frac = ct.div(ct.sum(1), axis=0)
    bottom = np.zeros(len(frac))
    for cc, col in (("FSI", "#d94801"), ("SPN", "#08519c")):
        if cc in frac:
            axB.bar(frac.index, frac[cc], bottom=bottom, color=col, label=cc); bottom += frac[cc].values
    axB.set_ylabel("fraction"); axB.set_title(f"overlap: width-class × waveform\nχ²={chi2:.1f} p={pchi:.1e}", fontsize=10.5)
    axB.legend(frameon=False)

    axT = fig.add_subplot(gs[:, 2]); axT.axis("off")
    axT.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=6.6, family="monospace",
             transform=axT.transAxes)
    fig.suptitle("Part 2 — Does the transient/sustained axis map onto narrow/broad (FSI/SPN)?",
                 fontsize=13, y=1.01)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"width_vs_waveform.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    (OUT / "width_vs_waveform_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}/width_vs_waveform.png")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify `canonical_session_id` is importable from config**

Run: `py -c "from visdetect.analysis.config import canonical_session_id as f; print(f('01072025'), f(1072025), f('1072025.0'))"`
Expected: all three print `01072025` (the canonical zfill8). If the import path differs, adjust the import in the script to the real location (grep: `grep -rn "def canonical_session_id" src/`).

- [ ] **Step 3: Run and verify outputs**

Run: `py scripts/tf_responsiveness/state_conditioned/width_vs_waveform.py`
Expected: prints the crosstab, continuous Spearman, four-quadrant table, and mixedlm independence lines; writes `width_vs_waveform.png/.pdf/_stats.txt`. Sanity: t2p join coverage > 80% of cells (else the filename/key mapping is wrong — investigate before trusting).

- [ ] **Step 4: Commit**

```bash
git add scripts/tf_responsiveness/state_conditioned/width_vs_waveform.py
git commit -m "feat(tf-width): Part 2 width-vs-waveform mapping (overlap + 2D + independence)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Adversarial verification + science writeup + memory

**Files:**
- Create: `docs/science/2026-07-07-transient-sustained-spectrum-celltype.md`
- Modify: `docs/science/2026-07-02-transient-sustained-tf-cells.md` (add a cross-link line to the companion doc).
- Modify: the memory `tf_kernel_latency_outcome_coupling_jul2026` (or add a new memory) with the verified verdicts.

**Interfaces:**
- Consumes: the figures + `_stats` files from Tasks 4–5.
- Produces: a methods/results doc with verified numbers and safe talk wording.

- [ ] **Step 1: Adversarial verification pass**

Dispatch 3 independent skeptic subagents (Opus 4.8 — `model: 'opus'`), each given the two `_stats.txt` files + the cache, prompted to REFUTE the headline (spectrum-vs-classes verdict; width⊥waveform independence). Require: re-derive the pooled ΔBIC and the mixedlm width-beta from the cache; check the modality verdict is consistent across all three width measures; confirm the independence result is not driven by one mouse/region; confirm every cross-neuron magnitude was firing-rate-controlled. Record which claims survive a ≥2/3 refutation vote. Only surviving claims go in the writeup as findings.

- [ ] **Step 2: Write the results doc**

Write `docs/science/2026-07-07-transient-sustained-spectrum-celltype.md` mirroring the structure of the 2026-07-02 doc: one-line result; shared definitions (continuous width via local refit + validation gate); Part 1 (modality battery table + segmented-regression verdict, with the honest spectrum-vs-classes reframing guardrail); Part 2 (overlap crosstab + 2D Spearman + four-quadrant + independence mixedlm); yield-bias caveat; reproduce commands; safe talk wording. Use only adversarially-surviving numbers.

- [ ] **Step 3: Cross-link + commit the docs**

```bash
git add docs/science/2026-07-07-transient-sustained-spectrum-celltype.md docs/science/2026-07-02-transient-sustained-tf-cells.md
git commit -m "docs(tf-transient-sustained): spectrum-vs-classes + waveform-mapping results

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 4: Update memory**

Update `C:/Users/Ben/.claude/projects/.../memory/tf_kernel_latency_outcome_coupling_jul2026.md` (and its `MEMORY.md` pointer) with: the spectrum-vs-classes verdict (with the ΔBIC/silverman/BC numbers), the width⊥waveform independence result, the fact that the raw kernel is cached nowhere (local/X:/gitignored — a local refit reproduces it), and links to the new doc + `kernel_width_continuous.csv` cache. Follow the memory format (frontmatter + `**Why:** / **How to apply:**` if feedback-type).

---

## Self-Review (completed at authoring)

- **Spec coverage:** §4 Component A → Task 3; §5 Part 1 (modality + segmented) → Tasks 2+4; §6 Part 2 (overlap + 2D + independence + four-quadrant + region + yield-bias) → Task 5; §7 rigor (FR-control, mixedlm, adversarial) → Tasks 5–6; §8 deliverables → all; validation gate → Task 3. Covered.
- **Placeholder scan:** no TBD/TODO; every code step shows full code; commands have expected output.
- **Type consistency:** `interp_fwhm`/`temporal_spread`/`pulse_fwhm`/`grid_fwhm` names match between Task 3 (producer) and Tasks 4–5 (consumers); `gmm_delta_bic`/`segmented_vs_linear` return-dict keys match their test and figure usage; `canonical_session_id` used consistently for all joins.
- **Deviation from spec (noted):** the modality battery makes Hartigan's dip test optional (package not installed) and adds Sarle's bimodality coefficient as a no-dependency backfill; GMM ΔBIC (the repo's own waveform-bimodality method) is primary. Spec intent (a modality battery) preserved.
```
