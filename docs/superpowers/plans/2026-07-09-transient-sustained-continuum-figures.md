# Continuum re-renders of the transient/sustained figures — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add continuum/binned versions of the transient/sustained TF-cell figures (organized by continuous kernel width) alongside the originals, reflecting the SPECTRUM finding.

**Architecture:** One shared helper (`continuum_common.py`: width+metrics loader, `binned_trend` panel, width-bin family) + a parallel per-cell trace rebuild for all 520 cells (Component 0) + five `*_continuum.py` figure scripts. All additive; originals untouched.

**Tech Stack:** Python (`.venv`, `py`), numpy/pandas/scipy/statsmodels/matplotlib. Reuses `visdetect.analysis.{spectrum_stats, align, tf_glm, tf_glm_data, constants}` and the `state_conditioned/` helpers.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-07-09-transient-sustained-continuum-figures-design.md`.
- **Population:** the 520 good_dates responsive cells (BG_046 162, BG_039 39 = DMS; BG_031 319 = VMS).
- **Width axis = `interp_fwhm`** from `data/cache/tf_glm_bg046/kernel_width_continuous.csv` (primary). `temporal_spread` optional robustness overlay only.
- **Representation:** every metric-vs-width panel = faint per-cell scatter + width-decile mean ± bootstrap CI (1000 resamples, seed 42) + monotonic trend + Spearman(ρ,p). PSTHs (where originals had them) = width-binned family of population means (~5 gradient bins), NOT two class lines.
- **Additive & primary paths:** new scripts → new `FIGURES/tf_glm_bg046/*_continuum/` dirs; resolve all paths under `REPO` (repo root), NEVER `E:/.../vd_tf_bg046/...`. Do not modify any existing class-based script/figure.
- **Compute:** all cache-based EXCEPT Component 0 (parallel session rebuild — LOCAL, reads `data/pkls/`, never X:). Run long compute as a background bash from the main session.
- **Every figure** saves `png` + `pdf` + `_stats.txt`.
- **Test command:** `py -m pytest <path> -v` (`.venv/Scripts/python.exe -m pytest`).

---

## File Structure

**Create:**
- `scripts/tf_responsiveness/state_conditioned/continuum_common.py` — shared helper.
- `tests/analysis/test_continuum_common.py` — unit tests for the pure helper functions.
- `scripts/tf_responsiveness/state_conditioned/rebuild_peth_traces_all.py` — Component 0 (parallel trace rebuild, all 520 cells).
- `scripts/tf_responsiveness/state_conditioned/core_metrics_continuum.py` — Fig 4a.
- `scripts/tf_responsiveness/state_conditioned/heatmap_continuum.py` — Fig 4b.
- `scripts/tf_responsiveness/state_conditioned/hardening_continuum.py` — Fig 4c.
- `scripts/tf_responsiveness/state_conditioned/learning_continuum.py` — Fig 4d.
- `scripts/tf_responsiveness/state_conditioned/fa_lick_continuum.py` — Fig 4e.

**Produces (gitignored data/figures):**
- `data/cache/tf_glm_bg046/peth_traces_all.npz` (520 cells).
- `FIGURES/tf_glm_bg046/{core_metrics,heatmap,hardening,learning,fa_lick}_continuum/`.

---

## Task 1: Shared helper `continuum_common.py`

**Files:**
- Create: `scripts/tf_responsiveness/state_conditioned/continuum_common.py`
- Test: `tests/analysis/test_continuum_common.py`

**Interfaces — Produces:**
- `REPO: str`, `REGION: dict`, `OUTCOMES: list[tuple]`, `WIDTH = "interp_fwhm"`, `WIDTH_CMAP` (matplotlib colormap).
- `decile_stats(x, y, n_bins=10, n_boot=1000, seed=42) -> dict` → `{centers, mean, ci_lo, ci_hi, rho, p, n_per_bin}` (pure; equal-count width bins via quantiles).
- `width_bin_assign(width, n=5) -> (idx: np.ndarray, edges: np.ndarray)` — assign each cell to one of `n` equal-count width bins (pure).
- `binned_trend(ax, x, y, *, n_bins=10, color="#238b45", scatter=True, label=None) -> dict` — plots the panel (scatter + decile mean±CI + trend + Spearman text), returns `decile_stats(...)`.
- `load_width_metrics() -> pd.DataFrame` — one row per responsive cell: joins `kernel_width_continuous.csv` to the registry `c1_r_log2` (TF selectivity), adds `region`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/analysis/test_continuum_common.py
import sys, os
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                       "scripts/tf_responsiveness/state_conditioned"))
from continuum_common import decile_stats, width_bin_assign  # noqa: E402

def test_decile_stats_monotone_positive_relationship():
    rng = np.random.default_rng(0)
    x = rng.random(500)
    y = 3.0 * x + rng.normal(0, 0.1, 500)          # strong positive
    d = decile_stats(x, y, n_bins=10)
    assert d["rho"] > 0.9 and d["p"] < 1e-20
    assert len(d["centers"]) == 10 and len(d["mean"]) == 10
    # bin means increase with width
    assert d["mean"][-1] > d["mean"][0]
    # CI brackets the mean
    assert np.all(d["ci_lo"] <= d["mean"] + 1e-9) and np.all(d["ci_hi"] >= d["mean"] - 1e-9)

def test_decile_stats_deterministic():
    rng = np.random.default_rng(1); x = rng.random(300); y = rng.random(300)
    a = decile_stats(x, y, seed=42); b = decile_stats(x, y, seed=42)
    assert np.allclose(a["ci_lo"], b["ci_lo"]) and np.allclose(a["ci_hi"], b["ci_hi"])

def test_decile_stats_handles_nan():
    x = np.array([1., 2., np.nan, 4., 5., 6., 7., 8., 9., 10., 11., 12.])
    y = np.array([1., np.nan, 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.])
    d = decile_stats(x, y, n_bins=3)
    assert np.isfinite(d["rho"])

def test_width_bin_assign_equal_count():
    w = np.arange(100.0)
    idx, edges = width_bin_assign(w, n=5)
    # 5 bins, each ~20 cells, monotone assignment
    counts = np.bincount(idx, minlength=5)
    assert counts.min() >= 18 and counts.max() <= 22
    assert idx[0] == 0 and idx[-1] == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -m pytest tests/analysis/test_continuum_common.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'continuum_common'`.

- [ ] **Step 3: Write the implementation**

```python
# scripts/tf_responsiveness/state_conditioned/continuum_common.py
"""Shared helpers for the continuum re-renders of the transient/sustained figures:
a width+metrics loader, the decile-binned-trend panel, and a width-bin family for
PSTHs. Pure functions are unit-tested; plotting wrappers are thin.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from representative_cells import REPO, _registry, good_dates   # noqa: E402

REGION = {"BG_046": "DMS", "BG_039": "DMS", "BG_031": "VMS"}
MICE = [("BG_046", "DMS"), ("BG_039", "DMS"), ("BG_031", "VMS")]
WIDTH = "interp_fwhm"
OUTCOMES = [("change_on", "Change_ON response"), ("hit_ramp", "Hit motor ramp"),
            ("fa_ramp", "FA motor ramp")]
CACHE = Path(REPO) / "data/cache/tf_glm_bg046/kernel_width_continuous.csv"


def _cmap():
    import matplotlib.cm as cm
    return cm.get_cmap("viridis")


WIDTH_CMAP = None  # lazily set in plotting code via _cmap() to avoid import at load


def load_width_metrics() -> pd.DataFrame:
    """One row per responsive cell: continuous width + coupling metrics (from
    kernel_width_continuous.csv) joined to registry TF selectivity c1_r_log2."""
    d = pd.read_csv(CACHE, dtype={"session": str})
    # registry c1_r_log2 keyed by (subject, session, unit)
    regs = []
    for subj, _ in MICE:
        r = _registry(subj)[["session", "unit", "c1_r_log2"]].copy()
        r["subject"] = subj
        regs.append(r)
    reg = pd.concat(regs, ignore_index=True)
    reg["unit"] = reg["unit"].astype(int)
    d["unit"] = d["unit"].astype(int)
    d = d.merge(reg, on=["subject", "session", "unit"], how="left")
    d["region"] = d["subject"].map(REGION)
    return d


def decile_stats(x, y, n_bins=10, n_boot=1000, seed=42) -> dict:
    """Equal-count width bins; per-bin mean of y + bootstrap CI; global Spearman."""
    from scipy.stats import spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    order = np.argsort(x)
    x, y = x[order], y[order]
    rng = np.random.default_rng(seed)
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    centers, mean, lo, hi, npb = [], [], [], [], []
    for b in range(n_bins):
        sel = (x >= edges[b]) & (x < edges[b + 1])
        yb = y[sel]
        if yb.size == 0:
            continue
        centers.append(float(np.median(x[sel])))
        mean.append(float(np.mean(yb)))
        boots = np.array([np.mean(rng.choice(yb, yb.size)) for _ in range(n_boot)])
        lo.append(float(np.percentile(boots, 2.5)))
        hi.append(float(np.percentile(boots, 97.5)))
        npb.append(int(yb.size))
    rho, p = spearmanr(x, y) if x.size > 2 else (np.nan, np.nan)
    return {"centers": np.array(centers), "mean": np.array(mean),
            "ci_lo": np.array(lo), "ci_hi": np.array(hi),
            "rho": float(rho), "p": float(p), "n_per_bin": np.array(npb)}


def width_bin_assign(width, n=5):
    """Assign each cell to one of n equal-count width bins (0..n-1) + return edges."""
    width = np.asarray(width, float)
    finite = width[np.isfinite(width)]
    edges = np.quantile(finite, np.linspace(0, 1, n + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.searchsorted(edges, width, side="right") - 1, 0, n - 1)
    idx = np.where(np.isfinite(width), idx, -1)
    return idx.astype(int), edges


def binned_trend(ax, x, y, *, n_bins=10, color="#238b45", scatter=True, label=None):
    """Scatter + decile mean±bootstrap-CI + monotonic trend + Spearman annotation."""
    d = decile_stats(x, y, n_bins=n_bins)
    x = np.asarray(x, float); y = np.asarray(y, float)
    if scatter:
        ax.scatter(x, y, s=6, alpha=0.18, color="0.5", edgecolors="none", zorder=1)
    ax.fill_between(d["centers"], d["ci_lo"], d["ci_hi"], color=color, alpha=0.25, zorder=2)
    ax.plot(d["centers"], d["mean"], "o-", color=color, lw=2, ms=5,
            label=label, zorder=3)
    ax.text(0.03, 0.95, f"ρ={d['rho']:+.2f}\np={d['p']:.1e}", transform=ax.transAxes,
            va="top", ha="left", fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return d
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -m pytest tests/analysis/test_continuum_common.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Sanity-check the loader on real data**

Run:
```bash
py -c "import sys; sys.path.insert(0,'scripts/tf_responsiveness/state_conditioned'); \
from continuum_common import load_width_metrics as f; d=f(); \
print('rows',len(d),'| c1_r_log2 coverage',round(d.c1_r_log2.notna().mean(),3), \
'| cols', [c for c in ['interp_fwhm','c1_r_log2','base_hz','change_on','region'] if c in d.columns])"
```
Expected: rows 520; c1_r_log2 coverage > 0.95; all listed cols present.

- [ ] **Step 6: Commit**

```bash
git add scripts/tf_responsiveness/state_conditioned/continuum_common.py tests/analysis/test_continuum_common.py
git commit -m "feat(continuum): shared helper — width+metrics loader, binned-trend panel, width-bin family

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Component 0 — parallel per-cell trace rebuild (all 520 cells)

**Files:**
- Create: `scripts/tf_responsiveness/state_conditioned/rebuild_peth_traces_all.py`
- Produces: `data/cache/tf_glm_bg046/peth_traces_all.npz`

**Interfaces:**
- Consumes: `continuum_common.{REPO, MICE}`; `representative_cells.{_registry, good_dates, _spikes, load_session, get_event_times_by_trial}`; `heatmap_transient_sustained.{ALIGN, BIN, SIG, PULSE_CAP, MIN_EV, _cfg or _ztrace/_outcome_times}`; `tf_glm.{assemble_design, pulse_times_from_tf}`; `tf_glm_data.session_trial_regressors`; `align.align_spikes_to_events`.
- Produces npz keys: `meta_subject, meta_session, meta_unit, meta_cls` (cls from width classes for reference) + `t_pulse/t_change/t_fa` + `mat_pulse/mat_change/mat_fa` (n_cells × n_bins, NaN rows for missing traces). Same schema as the existing `peth_traces.npz` but for ALL 520 cells.

- [ ] **Step 1: Write the script**

```python
# scripts/tf_responsiveness/state_conditioned/rebuild_peth_traces_all.py
"""Component 0: rebuild per-cell z-scored PETH traces (pulse / Change_ON / FA) for
ALL 520 responsive cells — including the ~106 intermediate-width cells the cached
peth_traces.npz drops (they are the MIDDLE of the width continuum). Reuses the
heatmap trace logic (session_trial_regressors -> design -> pulse_times -> _ztrace)
but with NO transient/sustained class filter. Parallelised across sessions
(ProcessPool, BLAS pinned 1/worker); deterministic per-session pulse-subsample seed.
LOCAL ONLY (reads data/pkls/, never X:). Usage: py rebuild_peth_traces_all.py [--workers N]
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from continuum_common import REPO, MICE                                       # noqa: E402
from representative_cells import (_registry, good_dates, _spikes, load_session,  # noqa: E402
                                  get_event_times_by_trial)
from heatmap_transient_sustained import ALIGN, BIN, SIG, PULSE_CAP, MIN_EV, _cfg  # noqa: E402
from visdetect.analysis.align import align_spikes_to_events                   # noqa: E402
from visdetect.analysis.tf_glm import assemble_design, pulse_times_from_tf    # noqa: E402
from visdetect.analysis.tf_glm_data import session_trial_regressors           # noqa: E402

OUT_NPZ = Path(REPO) / "data/cache/tf_glm_bg046/peth_traces_all.npz"
NARROW, BROAD = 0.05, 0.15  # for the reference cls label only (grid kernel_fwhm)


def _ztrace(spk, times, win, base):
    if len(times) < MIN_EV:
        return None, None
    binned, t = align_spikes_to_events(spk, list(times), window=win, bin_size=BIN)
    binned = np.asarray(binned, float)
    bmask = (t >= base[0]) & (t < base[1])
    bvals = binned[:, bmask].ravel()
    mu, sd = bvals.mean(), bvals.std()
    if not np.isfinite(sd) or sd < 1e-6:
        sd = max(bvals.mean(), 1.0)
    z = gaussian_filter1d(binned.mean(0), SIG) if SIG > 0 else binned.mean(0)
    return (z - mu) / sd, t


def _outcome_times(session, event, outcome):
    et = np.asarray(get_event_times_by_trial(session, event), float)
    return [et[i] for i, tr in enumerate(session.trials)
            if str(getattr(tr, "trialoutcome", "") or "").lower() == outcome
            and i < et.size and np.isfinite(et[i])]


def _responsive_all(subj):
    r = _registry(subj)
    r = r[r.resp & r.session_date.isin(good_dates(subj))]
    return r[["session", "unit", "kernel_fwhm"]]


def _process_session(task):
    subj, sess, recs = task
    pkl = Path(REPO) / "data/pkls" / subj / f"{sess}.pkl"
    if not pkl.exists():
        return {"rows": [], "err": f"MISSING {pkl}"}
    try:
        s = load_session(str(pkl))
        cfg = _cfg()
        trials, _ = session_trial_regressors(s, cfg)
        d = assemble_design(trials, cfg)
        fast, _slow = pulse_times_from_tf(d, cfg)
        fast = np.asarray(fast, float)
        rng = np.random.default_rng(abs(hash(sess)) % (2**32))  # deterministic per session
        if fast.size > PULSE_CAP:
            fast = np.sort(rng.choice(fast, PULSE_CAP, replace=False))
        ev = {"pulse": fast,
              "change": _outcome_times(s, "Change_ON", "hit"),
              "fa": _outcome_times(s, "FA", "fa")}
        rows = []
        for r in recs:
            uid = int(r["unit"])
            spk = np.sort(_spikes(s, uid))
            tr = {}
            for k, (win, base) in ALIGN.items():
                z, t = _ztrace(spk, ev[k], win, base)
                tr[k] = (z, t)
            fw = float(r["kernel_fwhm"])
            cls = "transient" if fw <= NARROW else ("sustained" if fw >= BROAD else "intermediate")
            rows.append({"subject": subj, "session": sess, "unit": uid, "cls": cls, "tr": tr})
        del s; gc.collect()
        return {"rows": rows, "err": None}
    except Exception as e:
        import traceback
        return {"rows": [], "err": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


def main(n_workers=10):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    tasks = []
    for subj, _ in MICE:
        r = _responsive_all(subj)
        for sess, g in r.groupby("session"):
            tasks.append((subj, sess, g[["unit", "kernel_fwhm"]].to_dict("records")))
    n_workers = max(1, min(n_workers, len(tasks)))
    print(f"START rebuild | {len(tasks)} sessions | {sum(len(t[2]) for t in tasks)} cells "
          f"| {n_workers} workers", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_process_session, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            res = fut.result(); results.append(res); done += 1
            print(f"  [{done}/{len(tasks)}] {'ERR '+res['err'].splitlines()[0] if res['err'] else str(len(res['rows']))+' cells'}", flush=True)

    # Determine each alignment's time axis (first non-None), then assemble padded mats.
    all_rows = [row for res in results for row in res["rows"]]
    tax = {k: None for k in ALIGN}
    for row in all_rows:
        for k in ALIGN:
            z, t = row["tr"][k]
            if t is not None and tax[k] is None:
                tax[k] = np.asarray(t, float)
    out = {"meta_subject": np.array([r["subject"] for r in all_rows]),
           "meta_session": np.array([r["session"] for r in all_rows]),
           "meta_unit": np.array([r["unit"] for r in all_rows]),
           "meta_cls": np.array([r["cls"] for r in all_rows])}
    for k in ALIGN:
        L = len(tax[k]) if tax[k] is not None else 0
        M = np.full((len(all_rows), L), np.nan)
        for i, row in enumerate(all_rows):
            z, t = row["tr"][k]
            if z is not None and len(z) == L:
                M[i] = z
        out[f"mat_{k}"] = M
        out[f"t_{k}"] = tax[k] if tax[k] is not None else np.zeros(0)
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, **out)
    errs = [r["err"] for r in results if r["err"]]
    print(f"wrote {OUT_NPZ} | {len(all_rows)} cells | {len(errs)} session errors", flush=True)
    print(f"cls counts: {pd.Series(out['meta_cls']).value_counts().to_dict()}", flush=True)
    print("END OK", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=10)
    main(n_workers=ap.parse_args().workers)
```

- [ ] **Step 2: Pre-flight (import + task build + picklability), then run in background**

Run pre-flight: `py -c "import sys; sys.path.insert(0,'scripts/tf_responsiveness/state_conditioned'); import rebuild_peth_traces_all as m, pickle; pickle.dumps(m._process_session); print('ok')"`
Expected: `ok`.
Then run the full rebuild as a BACKGROUND bash from the main session (exceeds the 10-min foreground timeout): `.venv/Scripts/python.exe scripts/tf_responsiveness/state_conditioned/rebuild_peth_traces_all.py --workers 10`. Poll its output for `END OK`.

- [ ] **Step 3: Verify the npz has all 520 cells incl. intermediates**

Run:
```bash
py -c "import numpy as np; z=np.load('data/cache/tf_glm_bg046/peth_traces_all.npz', allow_pickle=True); \
import pandas as pd; print('cells', len(z['meta_unit'])); \
print('cls', pd.Series(z['meta_cls']).value_counts().to_dict()); \
print('mat_fa', z['mat_fa'].shape, '| finite rows', int(np.isfinite(z['mat_fa']).any(1).sum()))"
```
Expected: cells ≈ 520; cls dict includes a non-zero `intermediate` count (~106); mat_fa has ~520 rows with a healthy finite-row count.

- [ ] **Step 4: Commit**

```bash
git add scripts/tf_responsiveness/state_conditioned/rebuild_peth_traces_all.py
git commit -m "feat(continuum): parallel per-cell PETH-trace rebuild for all 520 cells (incl. intermediates)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `core_metrics_continuum.py` (re-renders §2)

**Files:** Create `scripts/tf_responsiveness/state_conditioned/core_metrics_continuum.py` → `FIGURES/tf_glm_bg046/core_metrics_continuum/`.

**Interfaces:** Consumes `continuum_common.{load_width_metrics, binned_trend, WIDTH, OUTCOMES, REPO}` and `visdetect.analysis.spectrum_stats.segmented_vs_linear`.

**Analysis (cache-only, fast):** load_width_metrics(); for each of 5 metrics — `c1_r_log2` (TF selectivity), `base_hz` (baseline rate), `change_on`, `hit_ramp`, `fa_ramp` — one panel = `binned_trend(ax, d[WIDTH], d[metric])`. Plus a 6th panel = width (`interp_fwhm`) histogram (log-x optional) marking the median. Stats txt: per-metric Spearman(ρ,p) + `segmented_vs_linear(d[WIDTH], d[metric])` ΔBIC (echoes "graded not stepped"). Mirror the panel/gridspec style of `transient_vs_sustained.py`. Suptitle: "Core transient/sustained metrics on the continuous width axis (binned deciles + trend)".

- [ ] **Step 1:** Write `core_metrics_continuum.py` per the analysis above (follow `transient_vs_sustained.py` layout; use `binned_trend` for each metric panel; write png+pdf+_stats.txt to the primary `FIGURES/.../core_metrics_continuum/`).
- [ ] **Step 2:** Run `py scripts/tf_responsiveness/state_conditioned/core_metrics_continuum.py`. Expected: writes the figure + stats; each outcome Spearman positive; selectivity/base_hz mild-positive; prints per-metric ρ + segmented ΔBIC.
- [ ] **Step 3:** Commit (script only; figures gitignored).

```bash
git add scripts/tf_responsiveness/state_conditioned/core_metrics_continuum.py
git commit -m "feat(continuum): core-metrics continuum figure (selectivity/rate/coupling vs width)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `heatmap_continuum.py` (re-renders §3; needs Task 2)

**Files:** Create `heatmap_continuum.py` → `FIGURES/tf_glm_bg046/heatmap_continuum/`.

**Interfaces:** Consumes `peth_traces_all.npz` (Task 2), `continuum_common.{load_width_metrics, width_bin_assign, WIDTH, REPO}`.

**Analysis:** load `peth_traces_all.npz`; join continuous `interp_fwhm` per (subject,session,unit) from load_width_metrics(). For each alignment (pulse/change/fa): (a) **heatmap** of `mat_<align>` with rows **ordered by continuous width** (ascending), a continuous-width colorbar strip on the left (viridis), per-unit z already computed; (b) above each heatmap a **PSTH family** = mean trace per width bin (`width_bin_assign(width, n=5)`), lines colored by the viridis gradient, legend = bin width-range. Use `TwoSlopeNorm(-1.5,0,3)` for change/fa, peak-normalized or `TwoSlopeNorm` for pulse (match the class heatmap's scaling). Suptitle notes cells are ordered by continuous width (no class blocks) and all 520 incl. intermediates. Follow `heatmap_transient_sustained.py` gridspec.

- [ ] **Step 1:** Write `heatmap_continuum.py` per the analysis (reuse the class heatmap's plotting scaling; order by width; add the width-binned PSTH families + width colorbar).
- [ ] **Step 2:** Run it. Expected: 3 heatmaps (pulse/change/fa) width-ordered + 3 PSTH families; cell count ≈ (finite rows per alignment); figure written to primary path.
- [ ] **Step 3:** Commit.

```bash
git add scripts/tf_responsiveness/state_conditioned/heatmap_continuum.py
git commit -m "feat(continuum): width-ordered heatmap + width-binned PSTH families

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: `hardening_continuum.py` (re-renders §6)

**Files:** Create `hardening_continuum.py` → `FIGURES/tf_glm_bg046/hardening_continuum/`.

**Interfaces:** Consumes `continuum_common.{load_width_metrics, binned_trend, WIDTH, OUTCOMES, REPO}`; `statsmodels.formula.api`; `waveform_celltype_join._norm_date` (for the consensus-cohort join); `scipy.stats.{spearmanr, wilcoxon}`.

**Analysis (cache-only):** continuous-width robustness of width→coupling (NOT a class gap):
- **Session random-intercept regression** `outcome ~ z(interp_fwhm) + C(region)`, groups=session, per outcome → width β + p (`mixedlm`; on non-convergence report a session-cluster-robust OLS `smf.ols(...).fit(cov_type="cluster", cov_kwds={"groups": session})`, per the Task-5 lesson — emit both).
- **Per-session Spearman(width, outcome)** for sessions with ≥5 cells; Wilcoxon of the per-session ρ across sessions (session = replication unit).
- **Tracked-unit collapse** (BG_046 consensus `data/cache/tracking_consensus/BG_046/consensus_members.csv`): collapse to one mean `interp_fwhm` + one mean outcome per `um_uid`; Spearman(width, outcome) on the collapsed units.
- Panels: `binned_trend` per outcome (pooled) + a bar panel comparing raw vs mixed/cluster-robust/tracked width-effect −log10(p) or ρ. Stats txt with all coefficients + a convergence flag. Follow `hardening_pseudoreplication.py` layout.

- [ ] **Step 1:** Write `hardening_continuum.py` per the analysis.
- [ ] **Step 2:** Run it. Expected: width→coupling significant pooled + survives session-RE (mixed or cluster-robust) + per-session sign test; tracked-collapse same direction; prints all coefficients.
- [ ] **Step 3:** Commit.

```bash
git add scripts/tf_responsiveness/state_conditioned/hardening_continuum.py
git commit -m "feat(continuum): pseudoreplication hardening on continuous width

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: `learning_continuum.py` (re-renders learning figure)

**Files:** Create `learning_continuum.py` → `FIGURES/tf_glm_bg046/learning_continuum/`.

**Interfaces:** Consumes `continuum_common.{load_width_metrics, binned_trend, WIDTH, OUTCOMES, REPO}`; `representative_cells._pdate`; `scipy.stats.spearmanr`; staging manifests `data/{SUBJ}_staging_manifest.csv`.

**Analysis (cache-only):** attach stage (Naive→Learning merge) + d′ + chrono per cell (as `learning_transient_sustained.attach_stage`, keyed on the width-metrics `session`). Row 1: per outcome, **within-stage** `binned_trend(width, outcome)` overlaid for Learning vs Expert + within-stage Spearman(width, outcome) each stage (drift-robust). Row 2: per outcome, **per-session** Spearman(width, outcome) slope vs session d′ (+ session-order partial Spearman, drift proxy), colored by region. Carry the drift-confound caveat in the suptitle + stats. Follow `learning_transient_sustained.py` layout.

- [ ] **Step 1:** Write `learning_continuum.py` per the analysis.
- [ ] **Step 2:** Run it. Expected: within-stage width→coupling Spearman reported per stage (drift-robust); per-session slope-vs-d′ with partial|order. Prints stats.
- [ ] **Step 3:** Commit.

```bash
git add scripts/tf_responsiveness/state_conditioned/learning_continuum.py
git commit -m "feat(continuum): learning axis on continuous width (within-stage + per-session vs d')

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: `fa_lick_continuum.py` (re-renders FA-lick figure; needs Task 2)

**Files:** Create `fa_lick_continuum.py` → `FIGURES/tf_glm_bg046/fa_lick_continuum/`.

**Interfaces:** Consumes `peth_traces_all.npz` (Task 2), `FIGURES/tf_glm_bg046/lick_acquisition/lick_acquisition_cells.csv`, `continuum_common.{load_width_metrics, binned_trend, width_bin_assign, WIDTH, REPO}`.

**Analysis:** join `interp_fwhm` + `lick_sig` to the FA traces (`mat_fa`). PRE window (−0.3,−0.15) s pre-lick ramp = mean z per cell. Panels: (a) `binned_trend(width, pre_lick_ramp)` — ramp vs continuous width (replaces the class comparison); (b) **width-ordered FA heatmap** (`mat_fa` rows sorted by width) + a width colorbar strip + a lick-responsive overlay strip (grey where `lick_sig` is missing — intermediates lack a lick label); (c) **FA PSTH family** by width bin (`width_bin_assign(width, 5)`), each trace annotated with % lick-responsive among the labeled cells in that bin. Follow `fa_lick_activity.py` layout. Note in the figure that lick labels cover the 414 class cells only.

- [ ] **Step 1:** Write `fa_lick_continuum.py` per the analysis.
- [ ] **Step 2:** Run it. Expected: ramp-vs-width Spearman positive; width-ordered FA heatmap + FA PSTH family; % lick-resp rising with width bin. Handles missing lick labels (grey).
- [ ] **Step 3:** Commit.

```bash
git add scripts/tf_responsiveness/state_conditioned/fa_lick_continuum.py
git commit -m "feat(continuum): FA-lick continuum (pre-lick ramp vs width + width-ordered FA heatmap/PSTH)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Doc note + memory

**Files:** Modify `docs/science/2026-07-07-transient-sustained-spectrum-celltype.md` (add a "Continuum figure set" pointer listing the 5 `*_continuum` figures + Component 0). Update memory `tf_spectrum_celltype_orthogonality_jul2026` (note the continuum re-render set + `continuum_common` + `peth_traces_all.npz`).

- [ ] **Step 1:** Append the continuum-figures pointer to the 2026-07-07 doc (paths + one line each).
- [ ] **Step 2:** Update the memory file + its MEMORY.md pointer.
- [ ] **Step 3:** Commit the doc.

```bash
git add docs/science/2026-07-07-transient-sustained-spectrum-celltype.md
git commit -m "docs(continuum): point the 2026-07-07 doc at the continuum figure set

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review (completed at authoring)

- **Spec coverage:** helper §3 → Task 1; Component 0 §3.5 (parallel) → Task 2; figs 4a–4e → Tasks 3–7; doc §6 → Task 8. Covered.
- **Placeholder scan:** Tasks 1–2 have complete code; Tasks 3–7 give exact inputs/analysis/panels + the helper API + named reference layouts (no vague "add a plot"); commands have expected output.
- **Type consistency:** `decile_stats`/`width_bin_assign`/`binned_trend`/`load_width_metrics` names + return keys match between Task 1 (producer) and Tasks 3–7 (consumers); `peth_traces_all.npz` schema matches between Task 2 (producer) and Tasks 4/7 (consumers); `WIDTH="interp_fwhm"` used everywhere.
- **Parallelization:** Component 0 uses ProcessPool across sessions, BLAS pinned, deterministic per-session seed (per the user's directive), mirroring `recompute_kernel_width.py`.
- **Deviation note:** Tasks 3–7 are figure/integration tasks (not TDD) — verified by running + sanity-checking printed stats, since matplotlib output isn't unit-testable; the pure helper (Task 1) is fully TDD.
```
