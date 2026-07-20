# Fig 5 e–h (preparatory activity by cell class) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. All subagents run on **Opus 4.8** (`claude-opus-4-8`) — never a smaller model.

**Goal:** Faithfully reproduce Khilkevich & Lohse (Nature 2024) Figure 5 e–h within striatum, replacing the brain-area grouping with the transient→sustained kernel-width axis + non-TF reference, and prove or refute the result with a built-in adversarial-verification battery.

**Architecture:** Three stages. **Stage 1** (`build_prep_cache.py`, local ProcessPool) recomputes, per unit (TF-responsive + non-TF, all 3 mice), the **mean** hit-lick and FA-lick PETHs z-scored to the paper's **2 s pre-change** baseline → a small per-unit cache (no per-trial tensors — the paper's "active" test is `|z of mean PETH| > 2.576`). **Stage 2** (cache-only figure scripts) renders panels e/f/g/h with the blend representation. **Stage 3** (`nulls_and_hardening.py` + a `harden-result` Opus refutation workflow) runs null controls, the confound battery, independent re-derivation, and a 6-lens adversarial pass before any claim is believed.

**Tech Stack:** Python 3.10 (`.venv`), numpy/pandas/scipy/matplotlib, statsmodels (mixedlm), `visdetect.*` library (editable-installed in this repo's venv), pytest.

## Global Constraints

Copy these verbatim into every task's mental checklist.

- **Design spec:** `docs/superpowers/specs/2026-07-20-fig5eh-preparatory-transient-sustained-nonTF-design.md`. Read §2 (verbatim paper recipe) and §5/Stage-3 (adversarial battery) before coding.
- **Imports:** `import visdetect...` works directly in this venv (editable install verified). **Do NOT** hardcode the sibling repo `E:/python_analysis/git_repos/vd_tf_bg046/src` (the existing `representative_cells.py` does this — do not copy it). New scripts derive `REPO = Path(__file__).resolve().parents[3]` for **data/figure** paths only.
- **Compute:** LOCAL only, **never over X:** (Samba/ceph HARD RULE). `ProcessPoolExecutor`, BLAS pinned 1/worker via `os.environ.setdefault("OMP_NUM_THREADS"/"OPENBLAS_NUM_THREADS"/"MKL_NUM_THREADS","1")` **before** importing numpy. `del sess; gc.collect()` after each session.
- **Constants (from `visdetect.analysis.constants`, never hardcode):** `DEFAULT_BIN_SIZE=0.025`, `DEFAULT_SIGMA_MS=25.0`, `EVENT_VALID_OUTCOMES` (Hit→{hit}, FA→{fa}, Change_ON→{hit,miss}), `LICK_HARDWARE_DELAY_MS=200`. New Fig5 constant `Z_ACTIVE=2.576` lives in `visdetect/analysis/preparatory.py`.
- **Paper parameters (verbatim, spec §2):** z-baseline = 2 s pre-**change** per unit; active = `|z of mean PETH| > 2.576`; baseline-fraction window = `[−2, −1.8]` s pre-lick; onset = earliest t with, within a 100 ms window for ≥80 ms, `lower-95%-CI(fraction) > 0` AND `mean(fraction) > 0.1`; CI bootstrap = **over neurons**, 5,000×.
- **Session-id joins:** normalize any join/compare key through `visdetect.analysis.config.canonical_session_id`. Never write `session` to CSV as int.
- **Regions (HARD RULE — per-region always):** DMS = {BG_046, BG_039}, VMS = {BG_031}. Every panel + every stat reports pooled **and** DMS-vs-VMS.
- **No silent truncation:** every unit/session dropped (missing pkl, <10 licks, <0.4 s RT, NaN width) is counted and logged.
- **Outputs:** `data/cache/preparatory_fig5/…` and `FIGURES/preparatory_fig5/…` in THIS repo (canonical convention). Each figure writes `.png`, `.pdf`, and a `_stats.csv`.
- **Branch:** work on `feature/fig5eh-preparatory-cellclass` (already created; spec committed 2cb1c30).
- **Run tests:** `.venv/Scripts/python.exe -m pytest tests/analysis/test_preparatory.py -v`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/visdetect/analysis/preparatory.py` | **Pure** stat primitives: z-scoring, active mask, fraction-active, bootstrap-over-units CI, population & per-cell onset, width deciles, pulse half-peak width. No I/O. |
| `tests/analysis/test_preparatory.py` | Unit tests for every primitive (closed-form synthetics). |
| `scripts/tf_responsiveness/preparatory_fig5/prep_common.py` | I/O + config: `REPO`, registry/`good_dates`/spikes loaders (this-repo paths), width-cache join, region map, transient/sustained/non-TF palette, per-unit lick-z-trace builder. |
| `scripts/tf_responsiveness/preparatory_fig5/build_prep_cache.py` | Stage 1: ProcessPool recompute → `data/cache/preparatory_fig5/prep_<lick>.npz` + validation gates. |
| `scripts/tf_responsiveness/preparatory_fig5/fig5e_fraction_active.py` | Panel e (3-line fraction-active, DMS/VMS/pooled). |
| `scripts/tf_responsiveness/preparatory_fig5/fig5fg_onset_heatmaps.py` | Panels f/g (width-decile & non-TF onset heatmaps). |
| `scripts/tf_responsiveness/preparatory_fig5/fig5h_onset_vs_width.py` | Panel h (per-decile faithful + per-cell scatter). |
| `scripts/tf_responsiveness/preparatory_fig5/nulls_and_hardening.py` | Stage 3: null shuffles, mixedlm, lick-leakage & lick-responsiveness controls, independent re-derivation. |
| `tests/scripts/test_prep_fig5.py` | Synthetic-session test of the per-unit builder + null-shuffle-flattens test. |

---

## Task 1: Pure stat primitives (`preparatory.py`)

**Files:**
- Create: `src/visdetect/analysis/preparatory.py`
- Test: `tests/analysis/test_preparatory.py`

**Interfaces — Produces:**
- `Z_ACTIVE = 2.576`
- `baseline_mean_sd(baseline_binned) -> (mu, sd)`
- `zscore_trace(mean_peth, mu, sd) -> np.ndarray`
- `active_mask(z, thresh=Z_ACTIVE) -> np.ndarray[bool]`
- `fraction_active(active_matrix, baseline_bins=None) -> np.ndarray`
- `bootstrap_fraction_ci(active_matrix, baseline_bins=None, n=5000, seed=42) -> (mean, lo, hi)`
- `population_onset(t, mean_frac, ci_lo, window_s=0.1, sustain_s=0.08, min_frac=0.1, bin_s=0.025) -> float`
- `cell_onset(t, z, thresh=Z_ACTIVE, window_s=0.1, sustain_s=0.08, bin_s=0.025) -> float`
- `width_deciles(width, n=10) -> (idx, edges)`
- `pulse_half_peak_width(mean_response, t, max_window_s=1.0) -> (width, peak_t)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/analysis/test_preparatory.py
import numpy as np
import pytest
from visdetect.analysis import preparatory as P


def test_baseline_mean_sd_guards_tiny_sd():
    mu, sd = P.baseline_mean_sd(np.full((5, 8), 3.0))  # zero variance
    assert mu == pytest.approx(3.0)
    assert sd == pytest.approx(3.0)  # sd<1e-6 -> max(mu,1)=3.0


def test_zscore_and_active_mask():
    z = P.zscore_trace(np.array([0.0, 5.0, -5.0]), mu=0.0, sd=1.0)
    m = P.active_mask(z)  # |z|>2.576
    assert list(m) == [False, True, True]


def test_fraction_active_baseline_subtraction():
    A = np.array([[1, 1, 0, 0], [1, 0, 0, 0]], float)  # 2 units x 4 bins
    frac = P.fraction_active(A, baseline_bins=slice(2, 4))  # baseline frac = 0
    assert frac[0] == pytest.approx(1.0)
    assert frac[1] == pytest.approx(0.5)


def test_bootstrap_fraction_ci_brackets_mean():
    rng = np.random.default_rng(0)
    A = (rng.random((60, 10)) < 0.5).astype(float)
    mean, lo, hi = P.bootstrap_fraction_ci(A, n=500, seed=1)
    assert np.all(lo <= mean + 1e-9) and np.all(mean - 1e-9 <= hi)


def test_population_onset_detects_sustained_rise():
    t = np.arange(-2, 1, 0.025)
    frac = np.zeros_like(t); lo = np.zeros_like(t)
    on_idx = np.argmin(np.abs(t - (-0.5)))
    frac[on_idx:] = 0.4; lo[on_idx:] = 0.1  # sustained, >0.1, ci>0
    onset = P.population_onset(t, frac, lo)
    assert onset == pytest.approx(-0.5, abs=0.03)


def test_population_onset_returns_nan_when_flat():
    t = np.arange(-2, 1, 0.025)
    assert np.isnan(P.population_onset(t, np.zeros_like(t), np.zeros_like(t)))


def test_cell_onset_single_bin_blip_rejected():
    t = np.arange(-1, 1, 0.025)
    z = np.zeros_like(t); z[10] = 5.0  # one bin only -> not sustained 80ms
    assert np.isnan(P.cell_onset(t, z))


def test_width_deciles_equal_count():
    w = np.arange(100.0)
    idx, edges = P.width_deciles(w, n=10)
    counts = np.bincount(idx[idx >= 0])
    assert counts.min() >= 9 and len(edges) == 11


def test_pulse_half_peak_width_triangle():
    t = np.linspace(0, 1, 201)
    resp = np.maximum(0, 1 - np.abs(t - 0.2) / 0.1)  # peak at 0.2, half-width 0.1
    w, pk = P.pulse_half_peak_width(resp, t)
    assert pk == pytest.approx(0.2, abs=0.02)
    assert w == pytest.approx(0.1, abs=0.02)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/analysis/test_preparatory.py -v`
Expected: FAIL (module `preparatory` not found).

- [ ] **Step 3: Implement `preparatory.py`**

```python
# src/visdetect/analysis/preparatory.py
"""Pure primitives for the Fig-5 e-h preparatory-activity reproduction
(Khilkevich & Lohse 2024, Methods pp.17-18). No I/O; unit-tested in isolation.

The paper's "significantly active" is |z of the trial-MEAN PETH| > 2.576, with z
computed against a 2 s pre-CHANGE baseline; the fraction of active units is
bootstrapped OVER NEURONS (not trials); the population activation onset uses a
100 ms / 80 ms / mean>0.1 rule. See the design spec §2 for verbatim quotes.
"""
from __future__ import annotations
import numpy as np

Z_ACTIVE = 2.576  # |z| threshold, P<0.01 two-sided (Fig 5e z-test)


def baseline_mean_sd(baseline_binned) -> tuple[float, float]:
    """mu, sd of pooled pre-change baseline firing (trials x bins). sd<1e-6 -> max(mu,1)."""
    v = np.asarray(baseline_binned, float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    mu, sd = float(np.mean(v)), float(np.std(v))
    if not np.isfinite(sd) or sd < 1e-6:
        sd = max(mu, 1.0)
    return mu, sd


def zscore_trace(mean_peth, mu, sd) -> np.ndarray:
    return (np.asarray(mean_peth, float) - mu) / sd


def active_mask(z, thresh=Z_ACTIVE) -> np.ndarray:
    return np.abs(np.asarray(z, float)) > thresh


def fraction_active(active_matrix, baseline_bins=None) -> np.ndarray:
    """Mean over units (rows) per bin; if baseline_bins (slice/index), subtract its mean."""
    A = np.asarray(active_matrix, float)
    frac = np.nanmean(A, axis=0)
    if baseline_bins is not None:
        frac = frac - np.nanmean(frac[baseline_bins])
    return frac


def bootstrap_fraction_ci(active_matrix, baseline_bins=None, n=5000, seed=42):
    """Bootstrap OVER UNITS. Returns (mean_frac, lo95, hi95) per bin."""
    A = np.asarray(active_matrix, float)
    nU = A.shape[0]
    base = fraction_active(A, baseline_bins)
    if nU < 3:
        return base, base, base
    rng = np.random.default_rng(seed)
    boots = np.empty((n, A.shape[1]))
    for b in range(n):
        boots[b] = fraction_active(A[rng.integers(0, nU, nU)], baseline_bins)
    return base, np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0)


def _first_sustained(cond, window_s, sustain_s, bin_s):
    need = int(round(sustain_s / bin_s))
    win = int(round(window_s / bin_s))
    for i in range(len(cond)):
        if cond[i] and cond[i:min(len(cond), i + win)].sum() >= need:
            return i
    return -1


def population_onset(t, mean_frac, ci_lo, *, window_s=0.1, sustain_s=0.08,
                     min_frac=0.1, bin_s=0.025) -> float:
    t = np.asarray(t, float)
    cond = (np.asarray(ci_lo, float) > 0) & (np.asarray(mean_frac, float) > min_frac)
    i = _first_sustained(cond, window_s, sustain_s, bin_s)
    return float(t[i]) if i >= 0 else np.nan


def cell_onset(t, z, *, thresh=Z_ACTIVE, window_s=0.1, sustain_s=0.08, bin_s=0.025) -> float:
    t = np.asarray(t, float)
    cond = np.abs(np.asarray(z, float)) > thresh
    i = _first_sustained(cond, window_s, sustain_s, bin_s)
    return float(t[i]) if i >= 0 else np.nan


def width_deciles(width, n=10):
    w = np.asarray(width, float)
    fin = w[np.isfinite(w)]
    edges = np.quantile(fin, np.linspace(0, 1, n + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.searchsorted(edges, w, side="right") - 1, 0, n - 1)
    idx = np.where(np.isfinite(w), idx, -1)
    return idx.astype(int), edges


def pulse_half_peak_width(mean_response, t, max_window_s=1.0):
    """Baseline-subtracted mean pulse response: peak = largest |change| within
    [0, max_window_s]; half-peak width = span where |resp| >= 0.5*|peak|
    around the peak (Khilkevich Methods p.18)."""
    r = np.asarray(mean_response, float)
    t = np.asarray(t, float)
    win = (t >= 0) & (t <= max_window_s)
    if not win.any():
        return np.nan, np.nan
    idxs = np.where(win)[0]
    pk_local = idxs[np.argmax(np.abs(r[idxs]))]
    peak_t, peak_v = float(t[pk_local]), r[pk_local]
    if peak_v == 0:
        return np.nan, peak_t
    half = 0.5 * abs(peak_v)
    lo = pk_local
    while lo > 0 and abs(r[lo - 1]) >= half:
        lo -= 1
    hi = pk_local
    while hi < len(r) - 1 and abs(r[hi + 1]) >= half:
        hi += 1
    return float(t[hi] - t[lo]), peak_t
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/analysis/test_preparatory.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/preparatory.py tests/analysis/test_preparatory.py
git commit -m "feat(preparatory): pure Fig5 e-h stat primitives (z/active/fraction/onset/width)"
```

---

## Task 2: I/O + config (`prep_common.py`)

**Files:**
- Create: `scripts/tf_responsiveness/preparatory_fig5/prep_common.py`

**Interfaces — Consumes:** Task 1 primitives. **Produces:**
- `REPO` (Path), `MICE=[("BG_046","DMS"),("BG_039","DMS"),("BG_031","VMS")]`, `REGION` map
- `CLASS_COLORS = {"transient":..,"sustained":..,"non-TF":..}`, `WIDTH_CMAP="viridis"`
- `LICK_WIN=(-2.0,1.5)`, `BASE_WIN=(-2.0,0.0)`, `BASE_FRAC_WIN=(-2.0,-1.8)`, `BIN`, `SIG_BINS`, `MIN_LICKS=10`, `MIN_RT=0.4`
- `load_registry(subj) -> DataFrame` (with `resp` bool, `kernel_fwhm`, `session`, `session_date`)
- `good_dates(subj) -> set[str]`
- `spikes_for(session, uid) -> np.ndarray`
- `load_width() -> DataFrame` (kernel_width_continuous.csv: subject/session/unit/interp_fwhm)
- `class_from_fwhm(fwhm) -> str` ("transient"/"sustained"/"intermediate")

- [ ] **Step 1: Write the module** (mirrors the verified `representative_cells.py` loaders but with this-repo paths and no sibling-repo insert)

```python
# scripts/tf_responsiveness/preparatory_fig5/prep_common.py
"""I/O + config for the Fig-5 e-h preparatory-activity port. THIS-repo paths only
(no vd_tf_bg046 sibling hardcode). Pure math lives in visdetect.analysis.preparatory."""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
import numpy as np
import pandas as pd

from visdetect.analysis.constants import DEFAULT_BIN_SIZE, DEFAULT_SIGMA_MS

REPO = Path(__file__).resolve().parents[3]
MICE = [("BG_046", "DMS"), ("BG_039", "DMS"), ("BG_031", "VMS")]
REGION = {"BG_046": "DMS", "BG_039": "DMS", "BG_031": "VMS"}

CLASS_COLORS = {"transient": "#3182bd", "sustained": "#e6550d", "non-TF": "#969696"}
WIDTH_CMAP = "viridis"

BIN = DEFAULT_BIN_SIZE                       # 0.025 s
SIG_BINS = DEFAULT_SIGMA_MS / 1000.0 / BIN   # 25 ms sigma in bins
LICK_WIN = (-2.0, 1.5)                        # around lick onset
BASE_WIN = (-2.0, 0.0)                        # 2 s pre-CHANGE (paper z-baseline)
BASE_FRAC_WIN = (-2.0, -1.8)                  # pre-lick baseline-fraction window (paper)
MIN_LICKS = 10
MIN_RT = 0.4
NARROW, BROAD = 0.05, 0.15                    # grid-fwhm class cut (project convention)
DISENG_MAX = 50.0


def load_registry(subj: str) -> pd.DataFrame:
    r = pd.read_csv(
        REPO / f"data/cache/tf_responsive/{subj.lower().replace('_','')}_tf_responsive.csv",
        dtype={"session": str, "session_date": str})
    r["resp"] = r.resp_log2.astype(str).str.lower().isin(["true", "1", "1.0"])
    return r


def good_dates(subj: str, max_diseng: float = DISENG_MAX) -> set:
    man = pd.read_csv(REPO / f"data/{subj}_staging_manifest.csv", dtype={"session_name": str})
    qc = man.loc[~man.qc_fail.astype(bool), "session_name"]
    keep = set()
    for d in qc:
        sf = REPO / f"data/cache/state_tags/{subj}/{d}.csv"
        if sf.exists():
            if 100 * (pd.read_csv(sf).state_label == "Disengaged").mean() < max_diseng:
                keep.add(d)
        else:
            keep.add(d)
    return keep


def spikes_for(session, uid: int) -> np.ndarray:
    for c in session.clusters:
        if int(c.cluster_id) == int(uid):
            return np.sort(np.asarray(c.spike_times, float).ravel())
    return np.zeros(0)


def load_width() -> pd.DataFrame:
    return pd.read_csv(REPO / "data/cache/tf_glm_bg046/kernel_width_continuous.csv",
                       dtype={"session": str})[["subject", "session", "unit", "interp_fwhm"]]


def class_from_fwhm(fwhm: float) -> str:
    return "transient" if fwhm <= NARROW else ("sustained" if fwhm >= BROAD else "intermediate")
```

- [ ] **Step 2: Smoke-test the loaders on real data**

Run:
```bash
.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'scripts/tf_responsiveness/preparatory_fig5'); import prep_common as C; r=C.load_registry('BG_046'); print('reg', r.resp.sum(), 'resp of', len(r)); print('good_dates', len(C.good_dates('BG_046'))); print('width', len(C.load_width()))"
```
Expected: prints `reg 195 resp of 7047`, a good_dates count (~32), `width 520`.

- [ ] **Step 3: Commit**

```bash
git add scripts/tf_responsiveness/preparatory_fig5/prep_common.py
git commit -m "feat(prep-fig5): I/O + config loaders (this-repo paths, no sibling hardcode)"
```

---

## Task 3: Stage-1 recompute (`build_prep_cache.py`)

**Files:**
- Create: `scripts/tf_responsiveness/preparatory_fig5/build_prep_cache.py`
- Test: `tests/scripts/test_prep_fig5.py`

**Interfaces — Consumes:** `prep_common`, `visdetect.analysis.align`, `visdetect.analysis.preparatory`. **Produces:** `unit_lick_ztrace(spikes, lick_times, change_times, lick_win, base_win, bin_s, sigma_bins) -> (z, t, n_licks)`; cache `data/cache/preparatory_fig5/prep_<lick>.npz` with keys `meta_subject/session/unit/region/cls/resp`, `interp_fwhm`, `z` (n_units×n_bins), `t`, `n_licks`.

- [ ] **Step 1: Write the failing per-unit test** (synthetic: a unit that fires a ramp only before the lick must produce z>2.576 pre-lick and a valid onset)

```python
# tests/scripts/test_prep_fig5.py
import sys, numpy as np, pytest
sys.path.insert(0, "scripts/tf_responsiveness/preparatory_fig5")
import build_prep_cache as B
from visdetect.analysis import preparatory as P


def _poisson_spikes(rate_fn, t0, t1, seed):
    rng = np.random.default_rng(seed)
    # thin a homogeneous max-rate process
    rmax = 80.0
    n = rng.poisson(rmax * (t1 - t0))
    cand = np.sort(rng.uniform(t0, t1, n))
    keep = rng.random(cand.size) < (rate_fn(cand) / rmax)
    return cand[keep]


def test_unit_lick_ztrace_detects_prelick_ramp():
    # baseline (pre-change) = flat 5 Hz; around each lick, a ramp peaking at lick
    lick_times = np.arange(10) * 20.0 + 100.0
    change_times = np.arange(10) * 20.0 + 108.0  # change 8 s into each window (far from lick)

    def rate(ts):
        r = np.full(ts.shape, 5.0)
        for L in lick_times:
            d = ts - L
            r += 40.0 * np.exp(-((d + 0.2) ** 2) / (2 * 0.15 ** 2))  # bump ~0.2 s pre-lick
        return r
    spk = _poisson_spikes(rate, 0, 400, seed=7)
    z, t, n = B.unit_lick_ztrace(spk, list(lick_times), list(change_times),
                                 lick_win=(-2.0, 1.5), base_win=(-2.0, 0.0),
                                 bin_s=0.025, sigma_bins=1.0)
    assert n == 10
    onset = P.cell_onset(t, z)
    assert -0.6 < onset < 0.05          # ramp onset is just before the lick
    assert np.nanmax(z) > 2.576         # clearly active


def test_unit_lick_ztrace_flat_unit_no_onset():
    lick_times = np.arange(10) * 20.0 + 100.0
    change_times = np.arange(10) * 20.0 + 108.0
    spk = _poisson_spikes(lambda ts: np.full(ts.shape, 8.0), 0, 400, seed=3)
    z, t, n = B.unit_lick_ztrace(spk, list(lick_times), list(change_times),
                                 lick_win=(-2.0, 1.5), base_win=(-2.0, 0.0),
                                 bin_s=0.025, sigma_bins=1.0)
    assert np.isnan(P.cell_onset(t, z))  # no sustained supra-threshold activity
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_prep_fig5.py -v`
Expected: FAIL (`build_prep_cache` has no `unit_lick_ztrace`).

- [ ] **Step 3: Implement the builder**

```python
# scripts/tf_responsiveness/preparatory_fig5/build_prep_cache.py
"""Stage 1: per-unit MEAN lick-aligned PETH z-scored to the 2 s pre-CHANGE baseline
(Khilkevich & Lohse Fig 5). TF-responsive + non-TF, all 3 mice. LOCAL ProcessPool.
Usage: py build_prep_cache.py [--lick hit|fa] [--workers N]"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, gc, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep_common as C
from visdetect.core.session import load_session
from visdetect.analysis.align import get_event_times_by_trial, align_spikes_to_events


def unit_lick_ztrace(spikes, lick_times, change_times, *, lick_win, base_win, bin_s, sigma_bins):
    """Return (z, t, n_licks). z = (smoothed mean lick-PETH - mu_bl)/sd_bl, with
    (mu_bl, sd_bl) from pooled pre-change baseline bins (unsmoothed)."""
    lick_times = [x for x in lick_times if np.isfinite(x)]
    change_times = [x for x in change_times if np.isfinite(x)]
    if len(lick_times) < 1 or len(change_times) < 1 or len(spikes) == 0:
        return None, None, len(lick_times)
    b_binned, _bt = align_spikes_to_events(spikes, change_times, window=base_win, bin_size=bin_s)
    from visdetect.analysis.preparatory import baseline_mean_sd, zscore_trace
    mu, sd = baseline_mean_sd(b_binned)
    l_binned, t = align_spikes_to_events(spikes, lick_times, window=lick_win, bin_size=bin_s)
    m = np.nanmean(np.asarray(l_binned, float), axis=0)
    if sigma_bins > 0:
        m = gaussian_filter1d(m, sigma_bins)
    return zscore_trace(m, mu, sd), np.asarray(t, float), len(lick_times)


def _select(subj, resp):
    r = C.load_registry(subj)
    r = r[(r.resp == resp) & r.session_date.isin(C.good_dates(subj))]
    return r[["session", "unit", "kernel_fwhm"]]


def _process_session(task):
    subj, sess, recs, lick = task
    pkl = C.REPO / "data/pkls" / subj / f"{sess}.pkl"
    if not pkl.exists():
        return {"rows": [], "err": f"MISSING {pkl}", "dropped": 0}
    try:
        s = load_session(str(pkl))
        change = np.asarray(get_event_times_by_trial(s, "Change_ON"), float)  # hit+miss (valid)
        lick_ev = "Hit" if lick == "hit" else "FA"
        licks = np.asarray(get_event_times_by_trial(s, lick_ev), float)        # finite = matching trials
        if lick == "hit":  # >=MIN_RT s from change (paper Fig 6 rule)
            rt = licks - change
            licks = np.where(np.isfinite(rt) & (rt >= C.MIN_RT), licks, np.nan)
        change_t = change[np.isfinite(change)]
        lick_t = licks[np.isfinite(licks)]
        rows, dropped = [], 0
        for r in recs:
            uid = int(r["unit"])
            spk = C.spikes_for(s, uid)
            z, t, n = unit_lick_ztrace(spk, list(lick_t), list(change_t),
                                       lick_win=C.LICK_WIN, base_win=C.BASE_WIN,
                                       bin_s=C.BIN, sigma_bins=C.SIG_BINS)
            if z is None or n < C.MIN_LICKS:
                dropped += 1
                continue
            rows.append({"subject": subj, "session": sess, "unit": uid,
                         "kernel_fwhm": float(r["kernel_fwhm"]), "z": z, "t": t, "n": n})
        del s; gc.collect()
        return {"rows": rows, "err": None, "dropped": dropped}
    except Exception as e:
        import traceback
        return {"rows": [], "err": f"{type(e).__name__}: {e}\n{traceback.format_exc()}", "dropped": 0}


def main(lick="hit", n_workers=10):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    width = C.load_width()
    wmap = {(r.subject, r.session, int(r.unit)): float(r.interp_fwhm) for r in width.itertuples()}
    tasks = []
    for subj, _ in C.MICE:
        for resp in (True, False):
            sel = _select(subj, resp)
            for sess, g in sel.groupby("session"):
                tasks.append((subj, sess, g[["unit", "kernel_fwhm"]].to_dict("records"), lick))
    n_workers = max(1, min(n_workers, len(tasks)))
    print(f"START prep cache lick={lick} | {len(tasks)} session-jobs | {n_workers} workers", flush=True)
    rows, errs, dropped = [], [], 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_process_session, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs)):
            res = fut.result()
            rows += res["rows"]; dropped += res["dropped"]
            if res["err"]:
                errs.append(res["err"])
            print(f"  [{i+1}/{len(tasks)}] {'ERR '+res['err'].splitlines()[0] if res['err'] else str(len(res['rows']))+' cells'}", flush=True)
    t_axis = next((r["t"] for r in rows if r["t"] is not None), np.zeros(0))
    L = len(t_axis)
    Z = np.full((len(rows), L), np.nan)
    for i, r in enumerate(rows):
        if r["z"] is not None and len(r["z"]) == L:
            Z[i] = r["z"]
    subjects = np.array([r["subject"] for r in rows])
    sessions = np.array([r["session"] for r in rows])
    fwhm = np.array([r["kernel_fwhm"] for r in rows])
    resp = np.array([wmap.get((r["subject"], r["session"], int(r["unit"])) ) is not None for r in rows])
    out = {
        "meta_subject": subjects, "meta_session": sessions,
        "meta_unit": np.array([r["unit"] for r in rows]),
        "region": np.array([C.REGION[s] for s in subjects]),
        "resp": resp,  # True if in the width table (TF-responsive)
        "cls": np.array([C.class_from_fwhm(f) if rp else "non-TF" for f, rp in zip(fwhm, resp)]),
        "interp_fwhm": np.array([wmap.get((r["subject"], r["session"], int(r["unit"])), np.nan) for r in rows]),
        "n_licks": np.array([r["n"] for r in rows]),
        "z": Z, "t": t_axis,
    }
    outp = C.REPO / f"data/cache/preparatory_fig5/prep_{lick}.npz"
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outp, **out)
    n_resp = int(resp.sum()); n_non = int((~resp).sum())
    print(f"wrote {outp} | {len(rows)} cells (resp {n_resp} / non-TF {n_non}) | "
          f"dropped {dropped} (<{C.MIN_LICKS} licks) | {len(errs)} session errors", flush=True)
    print(f"per region: {pd.Series(out['region']).value_counts().to_dict()}", flush=True)
    print("END OK", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lick", choices=["hit", "fa"], default="hit")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    main(lick=a.lick, n_workers=a.workers)
```

- [ ] **Step 4: Run the unit tests, then build the real cache**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_prep_fig5.py -v` → PASS.
Then (LOCAL, background): `.venv/Scripts/python.exe scripts/tf_responsiveness/preparatory_fig5/build_prep_cache.py --lick hit --workers 10` and again `--lick fa`.

- [ ] **Step 5: Validation gate** (a separate check script or inline asserts — must pass before Stage 2)

Run a check that: (a) `resp.sum()` ≈ registry responsive count intersected with width table (≈520 minus <10-lick drops); (b) non-TF count is in the thousands per the registry; (c) both DMS and VMS present with n>0; (d) `dropped`/errors printed. Record the numbers in the commit message. **If responsive n collapses (e.g. join key mismatch), STOP** — likely a `canonical_session_id` session-key mismatch between the width cache and registry.

- [ ] **Step 6: Commit**

```bash
git add scripts/tf_responsiveness/preparatory_fig5/build_prep_cache.py tests/scripts/test_prep_fig5.py
git commit -m "feat(prep-fig5): Stage-1 recompute of lick-aligned z-traces vs pre-change baseline"
```

---

## Task 4: Panel e — fraction of significantly active units

**Files:** Create `scripts/tf_responsiveness/preparatory_fig5/fig5e_fraction_active.py`.

**Interfaces — Consumes:** `prep_common`, `visdetect.analysis.preparatory` (`active_mask`, `fraction_active`, `bootstrap_fraction_ci`). **Produces:** `FIGURES/preparatory_fig5/{pooled,DMS,VMS}/fig5e_<lick>.{png,pdf}` + `fig5e_<lick>_stats.csv`.

- [ ] **Step 1: Implement the data core + figure**

Load `prep_<lick>.npz`. Build the per-unit boolean active matrix `A = |z|>2.576` (via `active_mask`). For each region set (pooled, DMS, VMS) and each group (`cls in {transient, sustained}` and `resp==False → non-TF`), select rows, compute `mean, lo, hi = bootstrap_fraction_ci(A[group], baseline_bins=<mask of BASE_FRAC_WIN>, n=5000)`, plot line + shaded CI in `CLASS_COLORS`. x = `t` (time from lick), vline at 0, label axes ("time from {hit|FA} lick (s)", "fraction active above baseline"). Follow the layout idiom of `scripts/tf_responsiveness/state_conditioned/fa_lick_continuum.py` (set_style('talk'), despine, savefig png+pdf). Write `_stats.csv` with columns `region, group, n_units, onset_s, peak_frac, t_peak`.

- [ ] **Step 2: Smoke test**

Run: `.venv/Scripts/python.exe scripts/tf_responsiveness/preparatory_fig5/fig5e_fraction_active.py --lick hit`
Expected: writes 3 PNGs (+pdf) and a stats CSV; prints per-group n_units; no exception. Eyeball: sustained line leads transient leads non-TF in DMS/VMS integrating view.

- [ ] **Step 3: Commit** `feat(prep-fig5): panel e fraction-active (3-line, per-region)`.

---

## Task 5: Panels f/g — onset heatmaps

**Files:** Create `scripts/tf_responsiveness/preparatory_fig5/fig5fg_onset_heatmaps.py`.

**Interfaces — Consumes:** `preparatory.width_deciles`, `bootstrap_fraction_ci`, `population_onset`. **Produces:** `FIGURES/preparatory_fig5/{DMS,VMS,pooled}/fig5fg_<lick>.{png,pdf}` + `_stats.csv`.

- [ ] **Step 1: Implement.** Panel f: among TF-responsive cells, assign `width_deciles(interp_fwhm, n=10)`; for each decile compute `mean,lo,hi = bootstrap_fraction_ci(A[decile], baseline_bins)` and `onset = population_onset(t, mean, lo)`; build a (10 × n_bins) matrix of `mean` fraction; **sort rows by onset**; imshow with the onset points overlaid (black line, paper style); left strip = median decile width (viridis). Panel g: the non-TF population — rows = non-TF cells binned by their own onset (or per-session), same colour scale, no width gradient. Facet DMS/VMS. `_stats.csv`: `panel, region, row_id, decile_median_width, onset_s, n_units`.

- [ ] **Step 2: Smoke test.** Run for `--lick hit`; assert 10 decile rows, finite onsets for most deciles, PNG written. Eyeball: onset shortens from transient→sustained deciles in f; g shows no such gradient.

- [ ] **Step 3: Commit** `feat(prep-fig5): panels f/g onset heatmaps (width deciles + non-TF)`.

---

## Task 6: Panel h — onset vs width

**Files:** Create `scripts/tf_responsiveness/preparatory_fig5/fig5h_onset_vs_width.py`.

**Interfaces — Consumes:** `population_onset`, `cell_onset`, `width_deciles`, `pulse_half_peak_width` (optional pulse metric), scipy `pearsonr`/`spearmanr`. **Produces:** `FIGURES/preparatory_fig5/{pooled,DMS,VMS}/fig5h_<lick>.{png,pdf}` + `_stats.csv`.

- [ ] **Step 1: Implement.** Primary (faithful): per width-decile dot, x = decile `population_onset`, y = decile median `interp_fwhm`; Pearson + Spearman + bootstrap(10,000 over cells) CI on the correlation. Supplement: per-cell scatter x = `cell_onset(t, z)` (per unit), y = `interp_fwhm`, coloured by class, with a `binned_trend`-style decile overlay; non-TF drawn as an x-axis rug (onset only, no width). Repeat with `pulse_fwhm_1s` on y as the paper-faithful corroborator IF Task 7-opt built it, else reuse cached `pulse_fwhm_allpulses.csv` with a window caveat annotation. `_stats.csv`: `region, unit_of_obs, pearson_r, pearson_p, spearman_r, spearman_p, n, slope, ci_lo, ci_hi`.

- [ ] **Step 2: Smoke test.** Run `--lick hit`; assert a finite Pearson r and ≥8 decile points; PNG written. Report sign of slope.

- [ ] **Step 3: Commit** `feat(prep-fig5): panel h onset-vs-width (per-decile faithful + per-cell)`.

---

## Task 7: Stage-3 null controls + confound battery (`nulls_and_hardening.py`)

**Files:** Create `scripts/tf_responsiveness/preparatory_fig5/nulls_and_hardening.py`; extend `tests/scripts/test_prep_fig5.py`.

**Interfaces — Consumes:** all Stage-2 machinery. **Produces:** `FIGURES/preparatory_fig5/hardening/*.{png,csv}` and a `hardening_report.md`.

- [ ] **Step 1: Write the null-shuffle test** (the effect MUST die under the null)

```python
def test_label_shuffle_flattens_onset_gradient():
    import numpy as np
    sys.path.insert(0, "scripts/tf_responsiveness/preparatory_fig5")
    import nulls_and_hardening as H
    rng = np.random.default_rng(0)
    # synthetic: onset genuinely decreases with width
    width = np.linspace(0.03, 0.5, 200)
    onset = -0.2 * (width - 0.03) / 0.47 + rng.normal(0, 0.02, 200)  # wider -> earlier
    r_obs = H.width_onset_corr(width, onset)
    r_null = [H.width_onset_corr(rng.permutation(width), onset) for _ in range(200)]
    assert abs(r_obs) > np.percentile(np.abs(r_null), 95)  # real effect beats shuffled labels
```

- [ ] **Step 2: Run to fail, then implement** the battery: `width_onset_corr(width, onset)`; `label_shuffle_null(width, onset, n=1000)`; `lick_time_shuffle_null(...)` (circular-shift lick times in Stage 1 → fraction ramp collapses to baseline); `mixedlm_onset_width(df)` (statsmodels `mixedlm("onset ~ interp_fwhm", groups="session")`, also `+ C(region)`); `leakage_check` (recompute onset with post-lick bins censored → sustained still leads); `stratify_by_lickresp` (join `lick_acquisition_cells.csv`; width→onset holds within lick-responsive and within non-lick-responsive); `independent_rederivation` (recompute panel-h slope with a different onset implementation + seed and assert agreement within CI). Each writes to `hardening_report.md` with the null distribution beside every headline number, and per-region DMS/VMS.

- [ ] **Step 3: Run tests + the battery on real caches.** `pytest tests/scripts/test_prep_fig5.py -v` PASS; then run `nulls_and_hardening.py --lick hit`. Read `hardening_report.md`.

- [ ] **Step 4: Commit** `feat(prep-fig5): Stage-3 null controls + confound battery + independent re-derivation`.

---

## Task 8: Adversarial refutation pass (harden-result / Opus 4.8 workflow)

**Not a code task — a verification gate. Do this before writing any results doc or telling the user "we found X".**

- [ ] **Step 1:** Invoke the `harden-result` skill on the panel-h headline (and the panel-e ordering). It runs the lab's mandatory battery + a Workflow of ≥6 independent **Opus 4.8** skeptics, each on a distinct lens (FR/yield, lick leakage, circularity, pseudoreplication, region confound, null-shuffle adequacy), each prompted to REFUTE and re-derive from the caches. Bar = the `tf_spectrum_celltype_orthogonality` precedent ("0/6 refuted, high confidence").
- [ ] **Step 2:** Fold every surviving caveat into a results write-up (`research-notes-summarizer`) under `docs/science/2026-07-20-preparatory-activity-transient-sustained.md`. Report a **flat** panel h as-is (striatum need not mirror the brain-wide result) — never massage toward the paper's sign.
- [ ] **Step 3:** Preserve deliverables to the branch: `git add -f FIGURES/preparatory_fig5/**` (FIGURES is gitignored) + the stats/report, commit, and offer to merge to `main` per the project's artifact-preservation rule.

---

## Self-Review

**Spec coverage:** e→Task 4; f/g→Task 5; h→Task 6; recompute+baseline→Task 3; primitives→Task 1; null controls & confound battery & independent re-derivation→Task 7; adversarial refutation & caveats & flat-result honesty→Task 8; per-region everywhere→Global Constraints + each task; palette→Task 2. Panel-h faithful pulse-width (model-free 1 s) is available via `pulse_half_peak_width` (Task 1) and used in Task 6 (recompute is optional/deferred; cached `pulse_fwhm_allpulses.csv` is the fallback with a window caveat).

**Placeholder scan:** none — all math and the cache builder are complete code; figure/hardening tasks give the exact data core, output schema, and smoke tests, and reference the real template `fa_lick_continuum.py` for layout (a concrete file, not a placeholder).

**Type consistency:** `active_mask`/`fraction_active`/`bootstrap_fraction_ci`/`population_onset`/`cell_onset`/`width_deciles`/`pulse_half_peak_width` signatures are defined once in Task 1 and consumed unchanged in Tasks 4–7; `unit_lick_ztrace` signature defined in Task 3 and consumed in the tests; `prep_<lick>.npz` key set defined in Task 3 and read in Tasks 4–7.

**Open items deferred to review, not blocking:** panel-g row unit (cells-by-onset vs per-session); pulse-width recompute vs cached; exact per-cell-onset restriction to pre-lick. All have defaults; the spec §10 flags them.
