# Per-neuron Poisson GLM Replication (Khilkevich-Lohse 2024) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replicate the Khilkevich-Lohse 2024 per-neuron Poisson encoding GLM and its TF-responsive identification (lick-controlled), validate it on the paper's own data, then apply it to BG_046 (DMS) and BG_039 (cortex) to test whether graded baseline-TF coding survives lick/motor control where the earlier single-pulse-triggered metric found ≈0%.

**Architecture:** A new `tf_glm.py` builds a 50-ms-binned, temporally-unfolded (FIR) design matrix per session and fits a per-neuron ridge-Poisson GLM with nested 10-fold CV; a `tf_glm_data.py` adapter layer turns either a visdetect `Session` (BG_046/039, reduced regressor set) or a Khilkevich `npx_converted` session (full 19 regressors) into the model's regressor dict. TF-responsiveness is decided by the paper's two held-out criteria (C1 fast−slow prediction r>0.2, C2 ablation t-test P<0.01). Grating phase is backfilled onto existing pkls from raw `trials.json` (no Kilosort re-read).

**Tech Stack:** Python 3.10 (`.venv`, invoke as `py`), numpy, pandas, scikit-learn `PoissonRegressor` (L2 = ridge), matplotlib; pytest. Data on `X:/public/` (= ceph).

**Spec:** `docs/superpowers/specs/2026-06-18-tf-glm-replication-design.md` (read it first).

## Global Constraints

- **Invoke Python as `py`** (Windows + Git Bash), never `python`.
- **Build in the worktree** `E:/python_analysis/git_repos/vd_tf_phase0` on branch `feature/tf-responsiveness-labeler`. Run all commands with `PYTHONPATH=src` (`PYTHONPATH=E:/python_analysis/git_repos/vd_tf_phase0/src`) so you test the worktree's code, not the editable install pinned to the primary repo.
- **Bin size = 0.05 s (50 ms)** exactly (one TF pulse); never hardcode elsewhere — read from `TFGLMConfig.bin_s`.
- **Ridge only:** `PoissonRegressor(alpha=λ)` is L2-penalized Poisson (the glmnet α=0 equivalent). Penalize coefficients, not the intercept (sklearn default).
- **Fast/slow pulse = baseline TF ≥/≤ ±0.5 SD** of per-session mean baseline TF (paper value; the existing `tf_selectivity` uses ±1 SD — do not reuse that threshold here).
- **pkl edits are additive only.** New `Trial` fields; never alter/remove existing fields or spike data. Re-save to a **staging dir**, validate with `validate_pkl.py` (all existing fields must PASS), then swap. Never `rm -rf`/junction-overwrite live pkls (cf. June-2026 data-loss incident).
- **Units:** select via `get_good_cluster_ids(session)` (prefers `good_and_stable_ids`).
- **Canonical constants** from `visdetect.analysis.constants`; never duplicate thresholds.
- **Data inputs in the worktree:** `data/pkls` should be a junction to the primary repo's pkls; the BG_046 staging manifest must be present at `data/BG_046_staging_manifest.csv`. Verify before runs.

---

## Task 1: Add per-frame stimulus fields (phase / displayed-TF / vbl) to `Trial` + ingest

**Files:**
- Modify: `src/visdetect/core/session.py` (Trial dataclass, ~line 21-30)
- Modify: `src/visdetect/core/ingest.py` (`load_behavioral_trials`, ~line 82-105; add `extract_stim_timeseries` helper)
- Test: `tests/core/test_stim_extraction.py` (create; `tests/core/` may need an `__init__.py` — check sibling test dirs)

**Interfaces:**
- Produces: `Trial.stim_phase: Optional[np.ndarray]` (per-frame, shape (n_frames, 2)), `Trial.stim_tf_disp: Optional[np.ndarray]` (per-frame, (n_frames,)), `Trial.stim_vbl: Optional[np.ndarray]` (per-frame flip times, (n_frames,)).
- Produces: `extract_stim_timeseries(raw_trial: dict) -> dict` with keys `stim_phase`, `stim_tf_disp`, `stim_vbl` (numpy arrays or None). Consumed by Task 2.

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_stim_extraction.py
import numpy as np
from visdetect.core.ingest import extract_stim_timeseries
from visdetect.core.session import Trial

def test_extract_stim_timeseries_parses_arrays():
    raw = {
        "vbl": [100.0, 100.0166, 100.0333],
        "TF": [0.0, 1.2, 0.8],
        "phase": [[0, 0], [10, 0], [25, 0]],
    }
    out = extract_stim_timeseries(raw)
    assert out["stim_vbl"].shape == (3,)
    assert out["stim_tf_disp"].shape == (3,)
    assert out["stim_phase"].shape == (3, 2)
    np.testing.assert_allclose(out["stim_vbl"][0], 100.0)

def test_extract_stim_timeseries_missing_keys_returns_none():
    out = extract_stim_timeseries({"trialoutcome": "Hit"})
    assert out["stim_phase"] is None and out["stim_vbl"] is None

def test_trial_has_new_fields_default_none():
    t = Trial()
    assert t.stim_phase is None and t.stim_tf_disp is None and t.stim_vbl is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py -m pytest tests/core/test_stim_extraction.py -v`
Expected: FAIL (`ImportError: cannot import name 'extract_stim_timeseries'` and `Trial() got unexpected ... stim_phase`).

- [ ] **Step 3: Add the three fields to `Trial`**

In `src/visdetect/core/session.py`, extend the `Trial` dataclass:

```python
@dataclass
class Trial:
    trialoutcome: Optional[str] = None
    reactiontimes: Dict[str, float] = field(default_factory=dict)
    change_size: Optional[float] = None
    orientation: Optional[float] = None
    ITI: Optional[float] = None
    change_time: Optional[float] = None
    baseline_values: Optional[Any] = None
    n_seen: Optional[int] = None
    # Per-frame stimulus log (added 2026-06; for the TF-encoding GLM).
    # All None on legacy pkls until backfilled from raw trials.json.
    stim_phase: Optional[Any] = None    # (n_frames, 2) grating phase per flip
    stim_tf_disp: Optional[Any] = None  # (n_frames,)  displayed TF per flip
    stim_vbl: Optional[Any] = None      # (n_frames,)  Psychtoolbox vbl flip times
```

- [ ] **Step 4: Add `extract_stim_timeseries` and wire into `load_behavioral_trials`**

In `src/visdetect/core/ingest.py`, add the helper (place it just above `load_behavioral_trials`):

```python
def extract_stim_timeseries(raw_trial: dict) -> dict:
    """Pull the per-frame stimulus log (phase, displayed TF, vbl flip times)
    from a raw trials.json trial dict. Returns None for any absent key."""
    def _arr(key, ncol=None):
        v = raw_trial.get(key)
        if v is None:
            return None
        a = np.asarray(v, dtype=np.float64)
        if ncol is not None and (a.ndim != 2 or a.shape[1] != ncol):
            a = a.reshape(-1, ncol)
        return a
    return {
        "stim_phase": _arr("phase", ncol=2),
        "stim_tf_disp": _arr("TF"),
        "stim_vbl": _arr("vbl"),
    }
```

Then in `load_behavioral_trials`, inside the `for t in all_trials_raw:` loop, after computing `baseline_values`, add:

```python
        stim = extract_stim_timeseries(t)
```

and pass the fields into the `Trial(...)` constructor:

```python
        trials.append(Trial(
            trialoutcome=outcome,
            reactiontimes=rt,
            change_size=t.get("Stim2TF"),
            orientation=t.get("Stim2Ori"),
            ITI=t.get("stimD"),
            change_time=t.get("stimT"),
            baseline_values=baseline_values,
            stim_phase=stim["stim_phase"],
            stim_tf_disp=stim["stim_tf_disp"],
            stim_vbl=stim["stim_vbl"],
        ))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py -m pytest tests/core/test_stim_extraction.py -v`
Expected: PASS (3 passed). If `tests/core/` import fails, add an empty `tests/core/__init__.py`.

- [ ] **Step 6: Confirm legacy pkls still load (additive change)**

Run: `cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py -c "from visdetect.core.session import load_session; s=load_session(r'E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/data/pkls/BG_046/BG_046_01072025.pkl'); print('loaded', len(s.trials), 'trials; stim_phase on t0 =', s.trials[0].stim_phase)"`
Expected: `loaded 634 trials; stim_phase on t0 = None` (old pkl, fields default None — proves backward compatibility).

- [ ] **Step 7: Commit**

```bash
cd /e/python_analysis/git_repos/vd_tf_phase0
git add src/visdetect/core/session.py src/visdetect/core/ingest.py tests/core/test_stim_extraction.py
git commit -m "feat(ingest): extract per-frame phase/TF/vbl stimulus log into Trial"
```

---

## Task 2: Backfill stimulus fields onto existing BG_046 + BG_039 pkls (staged + validated)

**Files:**
- Create: `scripts/conversion/backfill_stim_phase.py`
- Test: `tests/conversion/test_backfill_stim_phase.py` (create; add `tests/conversion/__init__.py` if needed)

**Interfaces:**
- Consumes: `extract_stim_timeseries` (Task 1).
- Produces: `backfill_session(pkl_path, raw_session_dir, out_path) -> dict` — loads a pkl, reads the matching `Session/*trials.json`, attaches `stim_phase/stim_tf_disp/stim_vbl` to each Trial by positional index, saves to `out_path`. Returns `{"n_trials", "n_with_phase", "matched"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/conversion/test_backfill_stim_phase.py
import json, numpy as np
from pathlib import Path
from visdetect.core.session import Session, Trial, save_session, load_session
from scripts.conversion.backfill_stim_phase import backfill_session

def _make_raw(dirpath, n=3, nframes=5):
    sess = dirpath / "Session"; sess.mkdir(parents=True)
    trials = [{"trialoutcome": "Hit", "Stim2TF": 1.5,
               "vbl": list(np.arange(nframes) * 0.0166 + 100.0 + i),
               "TF": [0.0] * nframes,
               "phase": [[k, 0] for k in range(nframes)]} for i in range(n)]
    (sess / "run1__trials.json").write_text(json.dumps(trials))

def test_backfill_attaches_phase(tmp_path):
    raw = tmp_path / "BG_999_01012025"; _make_raw(raw, n=3, nframes=5)
    s = Session(trials=[Trial(trialoutcome="Hit") for _ in range(3)],
                session_name="BG_999_01012025")
    pkl = tmp_path / "in.pkl"; save_session(s, str(pkl))
    out = tmp_path / "out.pkl"
    info = backfill_session(str(pkl), str(raw), str(out))
    assert info["n_trials"] == 3 and info["n_with_phase"] == 3 and info["matched"]
    s2 = load_session(str(out))
    assert s2.trials[0].stim_phase.shape == (5, 2)
    assert s2.trials[1].stim_vbl.shape == (5,)

def test_backfill_count_mismatch_flags_unmatched(tmp_path):
    raw = tmp_path / "BG_999_01012025"; _make_raw(raw, n=2)   # 2 raw trials
    s = Session(trials=[Trial() for _ in range(3)], session_name="BG_999_01012025")  # 3 pkl trials
    pkl = tmp_path / "in.pkl"; save_session(s, str(pkl))
    info = backfill_session(str(pkl), str(raw), str(tmp_path / "out.pkl"))
    assert info["matched"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py -m pytest tests/conversion/test_backfill_stim_phase.py -v`
Expected: FAIL (`ModuleNotFoundError: scripts.conversion.backfill_stim_phase`).

- [ ] **Step 3: Implement the backfill script**

```python
# scripts/conversion/backfill_stim_phase.py
"""Attach the per-frame stimulus log (phase/TF/vbl) to EXISTING pkls from raw
trials.json — without re-running Kilosort. Additive: only sets the new Trial
fields; spike data and all existing fields are untouched.

Usage (single):
    py scripts/conversion/backfill_stim_phase.py \
        --pkl   data/pkls/BG_046/BG_046_01072025.pkl \
        --raw   "X:/public/.../BG_046/Raw data/BG_046_01072025" \
        --out   data/pkls_stim_staging/BG_046/BG_046_01072025.pkl
Usage (batch over a subject):
    py scripts/conversion/backfill_stim_phase.py \
        --pkl-dir data/pkls/BG_046 \
        --raw-root "X:/public/.../BG_046/Raw data" \
        --out-dir  data/pkls_stim_staging/BG_046
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np

from visdetect.core.session import load_session, save_session
from visdetect.core.ingest import extract_stim_timeseries


def _load_raw_trials(raw_session_dir: str) -> list:
    sess = Path(raw_session_dir) / "Session"
    raw = []
    for tf in sorted(sess.glob("*trials.json")):
        with open(tf, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw.extend(data if isinstance(data, list) else [data])
    return raw


def backfill_session(pkl_path: str, raw_session_dir: str, out_path: str) -> dict:
    s = load_session(pkl_path)
    raw = _load_raw_trials(raw_session_dir)
    matched = len(raw) == len(s.trials)
    n_with = 0
    if matched:
        for trial, r in zip(s.trials, raw):
            stim = extract_stim_timeseries(r)
            trial.stim_phase = stim["stim_phase"]
            trial.stim_tf_disp = stim["stim_tf_disp"]
            trial.stim_vbl = stim["stim_vbl"]
            if stim["stim_phase"] is not None:
                n_with += 1
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_session(s, out_path)
    return {"n_trials": len(s.trials), "n_with_phase": n_with, "matched": matched}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pkl"); p.add_argument("--raw"); p.add_argument("--out")
    p.add_argument("--pkl-dir"); p.add_argument("--raw-root"); p.add_argument("--out-dir")
    a = p.parse_args(argv)
    if a.pkl:
        info = backfill_session(a.pkl, a.raw, a.out)
        print(Path(a.pkl).name, info)
        return 0 if info["matched"] else 1
    pkls = sorted(glob.glob(str(Path(a.pkl_dir) / "*.pkl")))
    bad = []
    for pkl in pkls:
        sname = Path(pkl).stem
        raw = Path(a.raw_root) / sname
        out = Path(a.out_dir) / Path(pkl).name
        if not (raw / "Session").exists():
            print("NO RAW:", sname); bad.append(sname); continue
        info = backfill_session(pkl, str(raw), str(out))
        print(sname, info)
        if not info["matched"] or info["n_with_phase"] == 0:
            bad.append(sname)
    print(f"\nDONE: {len(pkls)} pkls, {len(bad)} need attention: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py -m pytest tests/conversion/test_backfill_stim_phase.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit the script**

```bash
cd /e/python_analysis/git_repos/vd_tf_phase0
git add scripts/conversion/backfill_stim_phase.py tests/conversion/test_backfill_stim_phase.py
git commit -m "feat(conversion): backfill per-frame stim log onto existing pkls"
```

- [ ] **Step 6: Backfill BG_046 to staging (real data)**

Run (one line):
```bash
cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py scripts/conversion/backfill_stim_phase.py \
  --pkl-dir data/pkls/BG_046 \
  --raw-root "X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/Raw data" \
  --out-dir data/pkls_stim_staging/BG_046
```
Expected: per-session lines like `BG_046_01072025 {'n_trials': 634, 'n_with_phase': 634, 'matched': True}` and `DONE: 46 pkls, 0 need attention: []`. If any session shows `matched: False`, note it (trial-count mismatch between pkl and raw runs) and exclude it — do **not** swap it.

- [ ] **Step 7: Validate staging is additive-only vs live pkls**

Run:
```bash
cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py scripts/conversion/validate_pkl.py \
  --new-dir data/pkls_stim_staging/BG_046 --ref-dir data/pkls/BG_046
```
Expected: `BATCH SUMMARY: 46 passed, 0 failed` (validate_pkl checks only existing fields — they must be byte-identical; the new stim fields are ignored, which is correct for an additive change).

- [ ] **Step 8: Swap staging → live, then spot-check**

```bash
cd /e/python_analysis/git_repos/vis_detect_analysis_Sep2025
# back up nothing destructively: move live aside, move staging in
mv data/pkls/BG_046 data/pkls/BG_046_prestim && \
mv /e/python_analysis/git_repos/vd_tf_phase0/data/pkls_stim_staging/BG_046 data/pkls/BG_046
```
Spot-check: `PYTHONPATH=/e/python_analysis/git_repos/vd_tf_phase0/src py -c "from visdetect.core.session import load_session; s=load_session('data/pkls/BG_046/BG_046_01072025.pkl'); t=s.trials[0]; print('phase', None if t.stim_phase is None else t.stim_phase.shape, '| vbl', None if t.stim_vbl is None else t.stim_vbl.shape)"`
Expected: `phase (787, 2) | vbl (787,)`. Keep `data/pkls/BG_046_prestim` until the whole plan is validated, then delete.

> NOTE: `data/pkls/BG_046` is a junction to primary; the `mv` operates on the primary repo's real files. Because the change is additive and validated, this is safe; the `_prestim` copy is the rollback.

- [ ] **Step 9: Repeat backfill+validate+swap for BG_039**

Same three commands with `BG_039` substituted and raw-root `X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_039/Raw data`. Expected: all matched, validate passes, swap, spot-check shows phase populated. (If BG_039 raw layout differs, confirm `Raw data/<session>/Session/*trials.json` exists; adjust `--raw-root` accordingly.)

---

## Task 3: TFGLMConfig + per-trial 50 ms time base and spike-count binning

**Files:**
- Create: `src/visdetect/analysis/tf_glm.py`
- Test: `tests/analysis/test_tf_glm_binning.py`

**Interfaces:**
- Produces: `TFGLMConfig` dataclass (fields below). Consumed by all later tasks.
- Produces: `trial_bin_edges(t_start, t_end, bin_s) -> np.ndarray` (bin left edges).
- Produces: `bin_spike_counts(spike_times, bin_edges) -> np.ndarray` (counts per bin, length = len(edges)).

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_tf_glm_binning.py
import numpy as np
from visdetect.analysis.tf_glm import TFGLMConfig, trial_bin_edges, bin_spike_counts

def test_bin_edges_50ms():
    cfg = TFGLMConfig()
    assert cfg.bin_s == 0.05
    e = trial_bin_edges(10.0, 10.2, cfg.bin_s)
    np.testing.assert_allclose(e, [10.0, 10.05, 10.10, 10.15])

def test_bin_spike_counts():
    e = np.array([0.0, 0.05, 0.10, 0.15])
    st = np.array([0.01, 0.02, 0.12, 0.99])  # two in bin0, one in bin2, one past end
    c = bin_spike_counts(st, e)
    np.testing.assert_array_equal(c, [2, 0, 1, 0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py -m pytest tests/analysis/test_tf_glm_binning.py -v`
Expected: FAIL (`ModuleNotFoundError: visdetect.analysis.tf_glm`).

- [ ] **Step 3: Create `tf_glm.py` with config + binning**

```python
# src/visdetect/analysis/tf_glm.py
"""Per-neuron Poisson encoding GLM (Khilkevich-Lohse 2024 replication).

50-ms-binned, temporally-unfolded (FIR) design matrix -> ridge-Poisson per
neuron with nested 10-fold CV -> TF-responsive identification by the paper's
two held-out criteria (C1 fast-minus-slow prediction r>0.2; C2 ablation t-test
P<0.01 across folds). See docs/superpowers/specs/2026-06-18-tf-glm-replication-design.md.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TFGLMConfig:
    bin_s: float = 0.05
    # FIR kernel windows (seconds, relative to event); (lo, hi) inclusive of lo,
    # exclusive of hi, stepped by bin_s.
    kern: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "tf":            (0.0, 1.5),
        "trial_start":   (0.0, 1.0),
        "time_in_base":  (0.0, 0.0),    # ramp handled as a single graded column
        "change":        (0.0, 2.0),    # per change-size (applied 6x)
        "lick_prep":     (-1.25, 0.0),
        "lick_exec":     (0.0, 0.5),
        "reward":        (0.0, 0.4),
        "abort":         (-1.25, 0.25),
        "wheel":         (-0.05, 0.8),
        "phase":         (0.0, 0.0),    # 12 bins x up/down, no temporal unfold
    })
    sd_pulse: float = 0.5               # fast/slow = +/-0.5 SD of baseline TF
    pulse_eval_win: Tuple[float, float] = (-0.15, 0.75)  # PETH window around pulses
    n_folds: int = 10
    lambdas: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    c1_r_thresh: float = 0.2
    c2_p_thresh: float = 0.01
    seed: int = 42
    include_phase: bool = False         # off for DMS-first; on for cortex


def trial_bin_edges(t_start: float, t_end: float, bin_s: float) -> np.ndarray:
    """Left edges of 50-ms bins spanning [t_start, t_end)."""
    n = int(np.floor((t_end - t_start) / bin_s + 1e-9))
    return t_start + np.arange(max(n, 0)) * bin_s


def bin_spike_counts(spike_times: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Spike count per 50-ms bin. Bin i = [edges[i], edges[i]+bin_s)."""
    st = np.asarray(spike_times, dtype=float).ravel()
    if bin_edges.size == 0:
        return np.zeros(0, dtype=float)
    bin_s = bin_edges[1] - bin_edges[0] if bin_edges.size > 1 else 0.05
    full = np.append(bin_edges, bin_edges[-1] + bin_s)
    counts, _ = np.histogram(st, bins=full)
    return counts.astype(float)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py -m pytest tests/analysis/test_tf_glm_binning.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /e/python_analysis/git_repos/vd_tf_phase0
git add src/visdetect/analysis/tf_glm.py tests/analysis/test_tf_glm_binning.py
git commit -m "feat(tf_glm): config + 50ms time base and spike-count binning"
```

---

## Task 4: FIR lagged-column builders (events + continuous)

**Files:**
- Modify: `src/visdetect/analysis/tf_glm.py`
- Test: `tests/analysis/test_tf_glm_design.py`

**Interfaces:**
- Produces: `fir_event(event_times, bin_edges, win, bin_s) -> np.ndarray` — (n_bins, n_lags) FIR design for point events; lag columns span `win` in `bin_s` steps; a column is 1 where a bin sits at that lag from an event.
- Produces: `fir_continuous(signal, win, bin_s) -> np.ndarray` — (n_bins, n_lags) lagged copies of a per-bin continuous `signal` (length n_bins), shifted by each lag in `win`.

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_tf_glm_design.py
import numpy as np
from visdetect.analysis.tf_glm import fir_event, fir_continuous

def test_fir_event_places_unit_at_each_lag():
    edges = np.arange(0.0, 0.5, 0.05)            # 10 bins
    ev = np.array([0.20])                        # event at bin 4
    X = fir_event(ev, edges, (0.0, 0.15), 0.05)  # lags 0,0.05,0.10 -> 3 cols
    assert X.shape == (10, 3)
    assert X[4, 0] == 1 and X[5, 1] == 1 and X[6, 2] == 1
    assert X[:, 0].sum() == 1

def test_fir_event_negative_lags():
    edges = np.arange(0.0, 0.5, 0.05)
    ev = np.array([0.20])
    X = fir_event(ev, edges, (-0.10, 0.05), 0.05)  # lags -0.10,-0.05,0.0
    assert X[2, 0] == 1 and X[3, 1] == 1 and X[4, 2] == 1

def test_fir_continuous_shifts():
    sig = np.array([1.0, 2.0, 3.0, 4.0])
    X = fir_continuous(sig, (0.0, 0.10), 0.05)   # lags 0, 0.05 -> 2 cols
    assert X.shape == (4, 2)
    np.testing.assert_array_equal(X[:, 0], [1, 2, 3, 4])   # lag 0
    np.testing.assert_array_equal(X[:, 1], [0, 1, 2, 3])   # lag +1 bin (causal)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py -m pytest tests/analysis/test_tf_glm_design.py -v`
Expected: FAIL (`ImportError: cannot import name 'fir_event'`).

- [ ] **Step 3: Implement the builders**

Append to `src/visdetect/analysis/tf_glm.py`:

```python
def _lag_offsets(win: Tuple[float, float], bin_s: float) -> np.ndarray:
    """Integer bin offsets for a kernel window [lo, hi) in bin_s steps."""
    lo, hi = win
    n = int(round((hi - lo) / bin_s))
    return np.arange(int(round(lo / bin_s)), int(round(lo / bin_s)) + max(n, 0))


def fir_event(event_times, bin_edges, win, bin_s) -> np.ndarray:
    """(n_bins, n_lags) FIR design for point events.

    Column j (lag = offsets[j]*bin_s): a 1 in bin b means an event occurred
    `lag` seconds before the start of bin b (i.e. event fell in bin b-offset).
    """
    n_bins = bin_edges.size
    offs = _lag_offsets(win, bin_s)
    X = np.zeros((n_bins, offs.size), dtype=float)
    ev = np.asarray(event_times, dtype=float).ravel()
    ev = ev[np.isfinite(ev)]
    if n_bins == 0 or ev.size == 0 or offs.size == 0:
        return X
    # bin index containing each event
    idx = np.floor((ev - bin_edges[0]) / bin_s + 1e-9).astype(int)
    idx = idx[(idx >= 0) & (idx < n_bins)]
    for j, off in enumerate(offs):
        b = idx + off
        b = b[(b >= 0) & (b < n_bins)]
        X[b, j] = 1.0
    return X


def fir_continuous(signal, win, bin_s) -> np.ndarray:
    """(n_bins, n_lags) lagged copies of a per-bin continuous signal.

    Column j is `signal` shifted so that row b holds signal[b - offset]
    (causal positive lags look back in time), zero-filled at the edges.
    """
    sig = np.asarray(signal, dtype=float).ravel()
    n_bins = sig.size
    offs = _lag_offsets(win, bin_s)
    X = np.zeros((n_bins, offs.size), dtype=float)
    for j, off in enumerate(offs):
        if off == 0:
            X[:, j] = sig
        elif off > 0:
            X[off:, j] = sig[: n_bins - off]
        else:
            X[:n_bins + off, j] = sig[-off:]
    return X
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py -m pytest tests/analysis/test_tf_glm_design.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd /e/python_analysis/git_repos/vd_tf_phase0
git add src/visdetect/analysis/tf_glm.py tests/analysis/test_tf_glm_design.py
git commit -m "feat(tf_glm): FIR lagged-column builders for events and continuous regressors"
```

---

## Task 5: Session regressor container + full design-matrix assembly

**Files:**
- Modify: `src/visdetect/analysis/tf_glm.py`
- Test: `tests/analysis/test_tf_glm_assemble.py`

**Interfaces:**
- Consumes: `fir_event`, `fir_continuous`, `trial_bin_edges`, `bin_spike_counts`, `TFGLMConfig`.
- Produces: `TrialRegressors` dataclass — per-trial fields the model needs (all on the **neural clock**):
  `t_start, t_end, change_time` (float, NaN if N/A), `change_size` (float), `tf_bins` (per-bin baseline TF, length n_bins, zeros after baseline), `lick_times, reward_time, abort_time` (arrays/float/NaN), `wheel_bins` (per-bin speed, length n_bins), `phase_bins` (optional (n_bins,) phase in [0,360) or None).
- Produces: `assemble_design(trials: List[TrialRegressors], cfg) -> DesignMatrix` with `.X` (n_bins_total, n_cols), `.col_groups: Dict[str, slice]`, `.bin_edges` (n_bins_total,), `.trial_index` (n_bins_total,), and `.tf_bins` (n_bins_total,) for pulse evaluation.
- Produces: `count_vector(trials, spike_times, design) -> np.ndarray` (y aligned to design rows for one neuron).

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_tf_glm_assemble.py
import numpy as np
from visdetect.analysis.tf_glm import (TFGLMConfig, TrialRegressors,
                                        assemble_design, count_vector)

def _toy_trial(t0, dur=2.0, bin_s=0.05, change_size=2.0, seed=0):
    n = int(dur / bin_s)
    rng = np.random.default_rng(seed)
    tf = np.zeros(n); tf[: n // 2] = rng.normal(0, 0.25, n // 2)  # baseline only
    return TrialRegressors(
        t_start=t0, t_end=t0 + dur, change_time=t0 + dur / 2, change_size=change_size,
        tf_bins=tf, lick_times=np.array([t0 + 1.6]), reward_time=t0 + 1.7,
        abort_time=np.nan, wheel_bins=np.zeros(n), phase_bins=None)

def test_assemble_shapes_and_groups():
    cfg = TFGLMConfig()
    trials = [_toy_trial(10.0), _toy_trial(20.0, seed=1)]
    d = assemble_design(trials, cfg)
    assert d.X.shape[0] == d.bin_edges.size == d.trial_index.size
    # six change-size columns groups present
    assert "tf" in d.col_groups and "lick_prep" in d.col_groups
    # TF group width == number of tf lags (1.5/0.05 = 30)
    assert d.col_groups["tf"].stop - d.col_groups["tf"].start == 30
    assert d.tf_bins.size == d.X.shape[0]

def test_count_vector_matches_rows():
    cfg = TFGLMConfig()
    trials = [_toy_trial(10.0)]
    d = assemble_design(trials, cfg)
    y = count_vector(trials, np.array([10.3, 10.32, 11.0]), d)
    assert y.size == d.X.shape[0] and y.sum() == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py -m pytest tests/analysis/test_tf_glm_assemble.py -v`
Expected: FAIL (`ImportError: cannot import name 'TrialRegressors'`).

- [ ] **Step 3: Implement the container + assembly**

Append to `src/visdetect/analysis/tf_glm.py`:

```python
@dataclass
class TrialRegressors:
    t_start: float
    t_end: float
    change_time: float          # neural-clock change onset; NaN if change not reached
    change_size: float          # 1.0 (catch), 1.25, 1.35, 1.5, 2, 4
    tf_bins: np.ndarray         # (n_bins,) baseline TF per bin (0 outside baseline)
    lick_times: np.ndarray      # neural-clock lick-bout onset times
    reward_time: float          # neural-clock; NaN if none
    abort_time: float           # neural-clock; NaN if none
    wheel_bins: np.ndarray      # (n_bins,) wheel speed per bin
    phase_bins: Optional[np.ndarray] = None  # (n_bins,) phase degrees [0,360) or None


@dataclass
class DesignMatrix:
    X: np.ndarray
    col_groups: Dict[str, slice]
    bin_edges: np.ndarray
    trial_index: np.ndarray
    tf_bins: np.ndarray


CHANGE_SIZES = (1.0, 1.25, 1.35, 1.5, 2.0, 4.0)


def _phase_indicator(phase_deg: np.ndarray, n_bins_circ: int = 12) -> np.ndarray:
    """(n_rows, n_bins_circ) one-hot of phase into n_bins_circ angular bins."""
    out = np.zeros((phase_deg.size, n_bins_circ), dtype=float)
    valid = np.isfinite(phase_deg)
    b = np.floor((phase_deg[valid] % 360) / (360.0 / n_bins_circ)).astype(int)
    out[np.where(valid)[0], np.clip(b, 0, n_bins_circ - 1)] = 1.0
    return out


def assemble_design(trials: List["TrialRegressors"], cfg: TFGLMConfig) -> DesignMatrix:
    bs = cfg.bin_s
    # Per-trial bin edges and concatenation bookkeeping
    per_edges, per_n, tf_all, wheel_all, phase_all = [], [], [], [], []
    for ti, tr in enumerate(trials):
        edges = trial_bin_edges(tr.t_start, tr.t_end, bs)
        per_edges.append(edges); per_n.append(edges.size)
        tf_all.append(_resize(tr.tf_bins, edges.size))
        wheel_all.append(_resize(tr.wheel_bins, edges.size))
        if cfg.include_phase and tr.phase_bins is not None:
            phase_all.append(_resize(tr.phase_bins, edges.size, fill=np.nan))
        else:
            phase_all.append(np.full(edges.size, np.nan))
    bin_edges = np.concatenate(per_edges) if per_edges else np.zeros(0)
    trial_index = np.concatenate([np.full(n, i) for i, n in enumerate(per_n)]) \
        if per_n else np.zeros(0, dtype=int)
    tf_bins = np.concatenate(tf_all) if tf_all else np.zeros(0)
    wheel_bins = np.concatenate(wheel_all) if wheel_all else np.zeros(0)
    phase_bins = np.concatenate(phase_all) if phase_all else np.zeros(0)
    N = bin_edges.size

    cols: List[np.ndarray] = []
    groups: Dict[str, slice] = {}

    def _add(name, block):
        start = sum(c.shape[1] for c in cols)
        cols.append(block)
        groups[name] = slice(start, start + block.shape[1])

    # 1) TF (continuous, per-bin, lagged) — built per-trial then stacked so lags
    #    do not bleed across trial boundaries.
    _add("tf", _blockwise(trials, per_edges, lambda tr, e: fir_continuous(
        _resize(tr.tf_bins, e.size), cfg.kern["tf"], bs)))
    # 2) trial start event
    _add("trial_start", _blockwise(trials, per_edges, lambda tr, e: fir_event(
        np.array([tr.t_start]), e, cfg.kern["trial_start"], bs)))
    # 3) time-in-baseline ramp (single graded column: seconds since t_start, 0
    #    after change; >=1 s region per the paper)
    _add("time_in_base", _blockwise(trials, per_edges, lambda tr, e:
        _ramp_col(tr, e, bs)))
    # 4-9) six change onsets by change size
    for cs in CHANGE_SIZES:
        _add(f"change_{cs}", _blockwise(trials, per_edges, lambda tr, e, cs=cs:
            fir_event(np.array([tr.change_time]) if (np.isfinite(tr.change_time)
                      and tr.change_size == cs) else np.zeros(0),
                      e, cfg.kern["change"], bs)))
    # 10) lick prep, 11) lick exec
    _add("lick_prep", _blockwise(trials, per_edges, lambda tr, e: fir_event(
        tr.lick_times, e, cfg.kern["lick_prep"], bs)))
    _add("lick_exec", _blockwise(trials, per_edges, lambda tr, e: fir_event(
        tr.lick_times, e, cfg.kern["lick_exec"], bs)))
    # 13) reward, 14) abort
    _add("reward", _blockwise(trials, per_edges, lambda tr, e: fir_event(
        np.array([tr.reward_time]), e, cfg.kern["reward"], bs)))
    _add("abort", _blockwise(trials, per_edges, lambda tr, e: fir_event(
        np.array([tr.abort_time]), e, cfg.kern["abort"], bs)))
    # 18) wheel (continuous)
    _add("wheel", _blockwise(trials, per_edges, lambda tr, e: fir_continuous(
        _resize(tr.wheel_bins, e.size), cfg.kern["wheel"], bs)))
    # 15-16) phase (optional)
    if cfg.include_phase:
        _add("phase", _phase_indicator(phase_bins))

    X = np.concatenate(cols, axis=1) if cols else np.zeros((N, 0))
    return DesignMatrix(X=X, col_groups=groups, bin_edges=bin_edges,
                        trial_index=trial_index, tf_bins=tf_bins)


def _resize(a, n, fill=0.0):
    a = np.asarray(a, dtype=float).ravel()
    if a.size == n:
        return a
    out = np.full(n, fill)
    m = min(a.size, n)
    out[:m] = a[:m]
    return out


def _ramp_col(tr, edges, bs):
    """Seconds since baseline start, zero before 1 s and after change onset."""
    t = edges - tr.t_start
    ramp = np.where(t >= 1.0, t, 0.0)
    if np.isfinite(tr.change_time):
        ramp[edges >= tr.change_time] = 0.0
    return ramp.reshape(-1, 1)


def _blockwise(trials, per_edges, fn):
    blocks = [fn(tr, e) for tr, e in zip(trials, per_edges)]
    ncol = max((b.shape[1] for b in blocks), default=0)
    blocks = [b if b.shape[1] == ncol else np.zeros((b.shape[0], ncol)) for b in blocks]
    return np.concatenate(blocks, axis=0) if blocks else np.zeros((0, ncol))


def count_vector(trials, spike_times, design: DesignMatrix) -> np.ndarray:
    y = np.zeros(design.bin_edges.size, dtype=float)
    bs = design.bin_edges[1] - design.bin_edges[0] if design.bin_edges.size > 1 else 0.05
    for i in range(len(trials)):
        mask = design.trial_index == i
        edges = design.bin_edges[mask]
        y[mask] = bin_spike_counts(spike_times, edges)
    return y
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py -m pytest tests/analysis/test_tf_glm_assemble.py -v`
Expected: PASS (2 passed). Fix any column-width mismatches surfaced by `_blockwise` (all per-trial blocks for a regressor must share lag count — they do, since `_lag_offsets` depends only on `win`/`bin_s`).

- [ ] **Step 5: Commit**

```bash
cd /e/python_analysis/git_repos/vd_tf_phase0
git add src/visdetect/analysis/tf_glm.py tests/analysis/test_tf_glm_assemble.py
git commit -m "feat(tf_glm): trial regressor container + full FIR design-matrix assembly"
```

---

## Task 6: Ridge-Poisson fit with nested 10-fold CV (held-out predictions)

**Files:**
- Modify: `src/visdetect/analysis/tf_glm.py`
- Test: `tests/analysis/test_tf_glm_fit.py`

**Interfaces:**
- Consumes: `TFGLMConfig`, `DesignMatrix`.
- Produces: `fit_poisson_cv(X, y, cfg, fold_ids=None) -> FitResult` with `.pred` (held-out prediction per row, full length), `.fold_ids` (n_rows,), `.coef_by_fold` (list of (n_cols,) arrays), `.best_lambdas` (per fold). Standardizes continuous columns internally is NOT done here (done at assembly); fit centers via intercept.
- Produces: `make_trial_folds(trial_index, n_folds, seed) -> np.ndarray` — fold id per row, assigned by **trial** (no within-trial leakage).

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_tf_glm_fit.py
import numpy as np
from visdetect.analysis.tf_glm import (TFGLMConfig, fit_poisson_cv, make_trial_folds)

def test_make_trial_folds_keeps_trials_intact():
    trial_index = np.array([0,0,0, 1,1, 2,2,2,2, 3])
    folds = make_trial_folds(trial_index, n_folds=2, seed=0)
    # all rows of a trial share one fold
    for t in np.unique(trial_index):
        assert len(set(folds[trial_index == t])) == 1

def test_fit_recovers_known_rate():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(0, 1, n)
    X = x.reshape(-1, 1)
    rate = np.exp(-1.0 + 0.8 * x)         # true log-linear rate
    y = rng.poisson(rate).astype(float)
    cfg = TFGLMConfig(n_folds=5, lambdas=(1e-3, 1e-2, 1e-1))
    fold_ids = np.repeat(np.arange(5), n // 5)
    res = fit_poisson_cv(X, y, cfg, fold_ids=fold_ids)
    # held-out prediction correlates with true rate
    assert np.corrcoef(res.pred, rate)[0, 1] > 0.5
    # recovered slope (mean across folds) near 0.8
    slope = np.mean([c[0] for c in res.coef_by_fold])
    assert 0.5 < slope < 1.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py -m pytest tests/analysis/test_tf_glm_fit.py -v`
Expected: FAIL (`ImportError: cannot import name 'fit_poisson_cv'`).

- [ ] **Step 3: Implement the fitter**

Append to `src/visdetect/analysis/tf_glm.py`:

```python
from sklearn.linear_model import PoissonRegressor


@dataclass
class FitResult:
    pred: np.ndarray
    fold_ids: np.ndarray
    coef_by_fold: List[np.ndarray]
    best_lambdas: List[float]


def make_trial_folds(trial_index: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    trials = np.unique(trial_index)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(trials.size)
    fold_of_trial = {int(trials[perm[k]]): k % n_folds for k in range(trials.size)}
    return np.array([fold_of_trial[int(t)] for t in trial_index])


def _fit_one(Xtr, ytr, lam):
    m = PoissonRegressor(alpha=lam, fit_intercept=True, max_iter=300, tol=1e-6)
    m.fit(Xtr, ytr)
    return m


def fit_poisson_cv(X, y, cfg: TFGLMConfig, fold_ids=None) -> FitResult:
    X = np.asarray(X, float); y = np.asarray(y, float)
    n = y.size
    if fold_ids is None:
        fold_ids = np.repeat(np.arange(cfg.n_folds), int(np.ceil(n / cfg.n_folds)))[:n]
    pred = np.full(n, np.nan)
    coefs, best_lams = [], []
    for f in range(cfg.n_folds):
        te = fold_ids == f
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        # inner CV over lambda on the training rows (split by inner folds)
        inner = fold_ids[tr]
        best_lam, best_score = cfg.lambdas[0], -np.inf
        for lam in cfg.lambdas:
            scores = []
            for g in np.unique(inner):
                itr = inner != g; ite = inner == g
                if ite.sum() == 0 or itr.sum() == 0:
                    continue
                m = _fit_one(X[tr][itr], y[tr][itr], lam)
                mu = m.predict(X[tr][ite])
                # Poisson held-out log-likelihood (up to const)
                scores.append(np.sum(y[tr][ite] * np.log(mu + 1e-9) - mu))
            s = np.mean(scores) if scores else -np.inf
            if s > best_score:
                best_score, best_lam = s, lam
        m = _fit_one(X[tr], y[tr], best_lam)
        pred[te] = m.predict(X[te])
        coefs.append(m.coef_.copy()); best_lams.append(best_lam)
    return FitResult(pred=pred, fold_ids=fold_ids, coef_by_fold=coefs, best_lambdas=best_lams)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py -m pytest tests/analysis/test_tf_glm_fit.py -v`
Expected: PASS (2 passed). (If `PoissonRegressor` convergence warns, raise `max_iter`; do not change tolerances of the test.)

- [ ] **Step 5: Commit**

```bash
cd /e/python_analysis/git_repos/vd_tf_phase0
git add src/visdetect/analysis/tf_glm.py tests/analysis/test_tf_glm_fit.py
git commit -m "feat(tf_glm): ridge-Poisson fit with trial-blocked nested 10-fold CV"
```

---

## Task 7: TF-responsive identification (C1 + C2) + kernel peak/FWHM

**Files:**
- Modify: `src/visdetect/analysis/tf_glm.py`
- Test: `tests/analysis/test_tf_glm_identify.py`

**Interfaces:**
- Consumes: `DesignMatrix`, `FitResult`, `TFGLMConfig`, fast/slow pulse times.
- Produces: `pulse_times_from_tf(design, cfg) -> (fast_times, slow_times)` — neural-clock bin-center times where per-bin baseline TF (log2) crosses ±sd_pulse·SD.
- Produces: `tf_pulse_peth(values_per_bin, bin_edges, pulse_times, win, bin_s) -> (t_axis, peth)` — event-triggered average of a per-bin signal (actual counts or predicted rate) around pulses.
- Produces: `identify_tf_responsive(design, y, full_fit, reduced_fit, cfg) -> dict` with `c1_r`, `c2_p`, `is_responsive`, `kernel_peak_t`, `kernel_fwhm` (kernel metrics computed from `full_fit.coef_by_fold` averaged over the TF group).

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_tf_glm_identify.py
import numpy as np
from visdetect.analysis.tf_glm import (TFGLMConfig, tf_pulse_peth,
                                        pulse_times_from_tf, identify_tf_responsive,
                                        DesignMatrix)

def test_tf_pulse_peth_triggers():
    edges = np.arange(0.0, 1.0, 0.05)
    sig = np.zeros(edges.size); sig[10] = 5.0       # impulse at t=0.5
    pulses = np.array([0.5])
    t, peth = tf_pulse_peth(sig, edges, pulses, (-0.15, 0.20), 0.05)
    assert peth[np.argmin(np.abs(t - 0.0))] == 5.0

def test_pulse_times_split_by_sd():
    # fabricate a design with tf_bins having a clear +/- excursion
    edges = np.arange(0.0, 1.0, 0.05)
    tf = np.zeros(edges.size); tf[5] = 1.0; tf[15] = -1.0   # +4SD, -4SD if SD~0.25
    d = DesignMatrix(X=np.zeros((edges.size, 0)), col_groups={}, bin_edges=edges,
                     trial_index=np.zeros(edges.size, int), tf_bins=tf)
    cfg = TFGLMConfig()
    fast, slow = pulse_times_from_tf(d, cfg)
    assert fast.size == 1 and slow.size == 1
    assert fast[0] < slow[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py -m pytest tests/analysis/test_tf_glm_identify.py -v`
Expected: FAIL (`ImportError: cannot import name 'tf_pulse_peth'`).

- [ ] **Step 3: Implement identification**

Append to `src/visdetect/analysis/tf_glm.py`:

```python
from scipy import stats as _stats


def pulse_times_from_tf(design: DesignMatrix, cfg: TFGLMConfig):
    """Bin-center times of fast/slow baseline-TF pulses (+/- sd_pulse*SD).

    tf_bins are linear TF (geom-mean 1 Hz); convert to log2 octaves so the SD
    matches the task's log2 N(0,0.25) baseline."""
    tf = np.asarray(design.tf_bins, float)
    bs = design.bin_edges[1] - design.bin_edges[0] if design.bin_edges.size > 1 else cfg.bin_s
    centers = design.bin_edges + bs / 2.0
    with np.errstate(divide="ignore"):
        log2tf = np.where(tf > 0, np.log2(np.where(tf > 0, tf, 1.0)), np.nan)
    valid = np.isfinite(log2tf) & (np.abs(log2tf) > 1e-9)
    if valid.sum() < 10:
        return np.zeros(0), np.zeros(0)
    sd = np.nanstd(log2tf[valid])
    thr = cfg.sd_pulse * sd
    fast = centers[valid & (log2tf >= thr)]
    slow = centers[valid & (log2tf <= -thr)]
    return fast, slow


def tf_pulse_peth(values_per_bin, bin_edges, pulse_times, win, bin_s):
    """Event-triggered average of a per-bin signal around pulse_times."""
    v = np.asarray(values_per_bin, float)
    offs = _lag_offsets(win, bin_s)
    t_axis = offs * bin_s
    if bin_edges.size == 0 or np.asarray(pulse_times).size == 0:
        return t_axis, np.full(offs.size, np.nan)
    idx = np.floor((np.asarray(pulse_times) - bin_edges[0]) / bin_s + 1e-9).astype(int)
    rows = []
    for p in idx:
        cols = p + offs
        ok = (cols >= 0) & (cols < v.size)
        row = np.full(offs.size, np.nan)
        row[ok] = v[cols[ok]]
        rows.append(row)
    return t_axis, np.nanmean(np.vstack(rows), axis=0)


def _tf_kernel(full_fit, design, cfg):
    """Mean TF FIR kernel (Hz-weight per lag) averaged across folds."""
    sl = design.col_groups.get("tf")
    if sl is None or not full_fit.coef_by_fold:
        return None
    K = np.vstack([c[sl] for c in full_fit.coef_by_fold])
    return K.mean(axis=0)


def identify_tf_responsive(design, y, full_fit, reduced_fit, cfg: TFGLMConfig) -> dict:
    fast, slow = pulse_times_from_tf(design, cfg)
    bs = cfg.bin_s
    win = cfg.pulse_eval_win

    def diff_peth(values):
        _, pf = tf_pulse_peth(values, design.bin_edges, fast, win, bs)
        _, ps = tf_pulse_peth(values, design.bin_edges, slow, win, bs)
        return pf - ps

    # C1: full-model predicted fast-slow vs actual fast-slow, Pearson r per fold
    folds = np.unique(full_fit.fold_ids)
    rs, resid = [], []
    actual_diff = diff_peth(y)
    for f in folds:
        m = full_fit.fold_ids == f
        # evaluate PETHs on this fold's rows only
        ev = np.zeros_like(y); ev[:] = np.nan
        # build per-fold actual / predicted aligned to all bins but masked to fold
        pred_full = np.where(m, full_fit.pred, np.nan)
        pred_red = np.where(m, reduced_fit.pred, np.nan)
        act = np.where(m, y, np.nan)
        d_pred = diff_peth(pred_full)
        d_act = diff_peth(act)
        good = np.isfinite(d_pred) & np.isfinite(d_act)
        if good.sum() >= 3 and np.std(d_pred[good]) > 1e-9 and np.std(d_act[good]) > 1e-9:
            rs.append(np.corrcoef(d_pred[good], d_act[good])[0, 1])
        # C2: residual TF prediction = full - reduced predicted fast-slow shape,
        #     scored against actual fast-slow shape (full should beat reduced)
        d_red = diff_peth(pred_red)
        gg = np.isfinite(d_pred) & np.isfinite(d_red) & np.isfinite(d_act)
        if gg.sum() >= 3:
            err_full = np.nansum((d_act[gg] - d_pred[gg]) ** 2)
            err_red = np.nansum((d_act[gg] - d_red[gg]) ** 2)
            resid.append(err_red - err_full)   # >0 means full predicts TF better
    c1_r = float(np.nanmean(rs)) if rs else np.nan
    # C2 one-sided t-test that residual improvement > 0 across folds
    if len(resid) >= 3:
        t, p_two = _stats.ttest_1samp(resid, 0.0)
        c2_p = p_two / 2.0 if t > 0 else 1.0 - p_two / 2.0
    else:
        c2_p = np.nan
    is_resp = bool((c1_r > cfg.c1_r_thresh) and (c2_p < cfg.c2_p_thresh))

    # kernel metrics
    kpeak_t, kfwhm = np.nan, np.nan
    K = _tf_kernel(full_fit, design, cfg)
    if K is not None and K.size:
        lags = _lag_offsets(cfg.kern["tf"], bs) * bs
        ip = int(np.argmax(np.abs(K)))
        kpeak_t = float(lags[ip])
        half = abs(K[ip]) / 2.0
        lo = ip
        while lo > 0 and abs(K[lo - 1]) >= half:
            lo -= 1
        hi = ip
        while hi < K.size - 1 and abs(K[hi + 1]) >= half:
            hi += 1
        kfwhm = float(lags[hi] - lags[lo])
    return {"c1_r": c1_r, "c2_p": c2_p, "is_responsive": is_resp,
            "n_fast": int(fast.size), "n_slow": int(slow.size),
            "kernel_peak_t": kpeak_t, "kernel_fwhm": kfwhm}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py -m pytest tests/analysis/test_tf_glm_identify.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Add an end-to-end synthetic recovery test**

```python
# tests/analysis/test_tf_glm_e2e.py
import numpy as np
from visdetect.analysis.tf_glm import (TFGLMConfig, TrialRegressors, assemble_design,
    count_vector, fit_poisson_cv, make_trial_folds, identify_tf_responsive)

def _session(n_trials=40, dur=6.0, bin_s=0.05, tf_gain=0.0, lick_gain=0.0, seed=0):
    rng = np.random.default_rng(seed)
    trials, spikes = [], []
    for i in range(n_trials):
        t0 = i * (dur + 1.0)
        n = int(dur / bin_s)
        tf = np.zeros(n); nb = n // 2
        tf[:nb] = 2 ** rng.normal(0, 0.25, nb)       # linear TF, log2 N(0,0.25)
        licks = np.array([t0 + dur - 0.5])
        tr = TrialRegressors(t_start=t0, t_end=t0 + dur, change_time=t0 + dur/2,
            change_size=2.0, tf_bins=tf, lick_times=licks, reward_time=np.nan,
            abort_time=np.nan, wheel_bins=np.zeros(n), phase_bins=None)
        trials.append(tr)
    cfg = TFGLMConfig(n_folds=5, lambdas=(1e-2, 1e-1, 1.0))
    design = assemble_design(trials, cfg)
    # synth rate: baseline + tf_gain * (log2 tf at lag 0) + lick bump
    log2tf = np.where(design.tf_bins > 0, np.log2(np.clip(design.tf_bins, 1e-9, None)), 0.0)
    lograte = -1.5 + tf_gain * log2tf
    rate = np.exp(lograte)
    y = rng.poisson(rate).astype(float)
    return trials, design, y, cfg

def test_tf_neuron_is_responsive():
    trials, design, y, cfg = _session(tf_gain=1.5, seed=1)
    folds = make_trial_folds(design.trial_index, cfg.n_folds, cfg.seed)
    full = fit_poisson_cv(design.X, y, cfg, folds)
    Xr = design.X.copy(); Xr[:, design.col_groups["tf"]] = 0.0
    red = fit_poisson_cv(Xr, y, cfg, folds)
    out = identify_tf_responsive(design, y, full, red, cfg)
    assert out["c1_r"] > 0.2 and out["c2_p"] < 0.01 and out["is_responsive"]

def test_flat_neuron_not_responsive():
    trials, design, y, cfg = _session(tf_gain=0.0, seed=2)
    folds = make_trial_folds(design.trial_index, cfg.n_folds, cfg.seed)
    full = fit_poisson_cv(design.X, y, cfg, folds)
    Xr = design.X.copy(); Xr[:, design.col_groups["tf"]] = 0.0
    red = fit_poisson_cv(Xr, y, cfg, folds)
    out = identify_tf_responsive(design, y, full, red, cfg)
    assert not out["is_responsive"]
```

Run: `cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py -m pytest tests/analysis/test_tf_glm_e2e.py -v`
Expected: PASS (2 passed). This is the key correctness gate: an injected-TF neuron is flagged; a flat neuron is not. If the TF neuron is missed, inspect `n_fast/n_slow` (need ≥ tens) and the reduced-model construction (zeroing the TF columns is the ablation).

- [ ] **Step 6: Commit**

```bash
cd /e/python_analysis/git_repos/vd_tf_phase0
git add src/visdetect/analysis/tf_glm.py tests/analysis/test_tf_glm_identify.py tests/analysis/test_tf_glm_e2e.py
git commit -m "feat(tf_glm): C1/C2 TF-responsive identification + kernel metrics + e2e recovery test"
```

---

## Task 8: Khilkevich `npx_converted` loader adapter (positive control input)

**Files:**
- Create: `src/visdetect/analysis/tf_glm_data.py`
- Test: `tests/analysis/test_tf_glm_data_khilkevich.py`

**Interfaces:**
- Produces: `load_khilkevich_session(session_dir) -> KhilSession` with `.units: Dict[int, np.ndarray]` (spike times by unit id), `.regions: Dict[int, str]`, `.trials: pd.DataFrame`, `.licks: np.ndarray`, `.change_on/.baseline_on/.valve/.airpuff` (np arrays), `.stim: pd.DataFrame` (per-frame TF/phase/vbl), `.movement: dict` (motion energy, pupil with times), `.running: np.ndarray`.
- Produces: `khilkevich_trial_regressors(ks: KhilSession, cfg, region=None) -> (List[TrialRegressors], Dict[int,np.ndarray])` — TrialRegressors + per-unit spike times for the requested region (full 19-regressor set: includes phase + wheel; motion energy/pupil enter via extra continuous columns — see Step 3).

- [ ] **Step 1: Install parquet support, then write the failing smoke test**

```bash
cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py -m pip install pyarrow -q
```

```python
# tests/analysis/test_tf_glm_data_khilkevich.py
import os, pytest
from visdetect.analysis.tf_glm_data import load_khilkevich_session

BASE = r"X:/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx_converted"

@pytest.mark.skipif(not os.path.isdir(BASE), reason="ceph not mounted")
def test_load_one_khilkevich_session():
    animal = sorted(os.listdir(BASE))[0]
    sess = sorted(os.listdir(os.path.join(BASE, animal)))[0]
    ks = load_khilkevich_session(os.path.join(BASE, animal, sess))
    assert len(ks.units) > 0
    assert ks.trials.shape[0] > 0
    assert ks.change_on.ndim == 1
    # at least one region label present
    assert len(set(ks.regions.values())) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py -m pytest tests/analysis/test_tf_glm_data_khilkevich.py -v`
Expected: FAIL (`ModuleNotFoundError: visdetect.analysis.tf_glm_data`).

- [ ] **Step 3: Implement the loader**

First **inspect** the real schema (one-off, not committed) so the column names are exact:

```bash
cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py -c "
import pandas as pd, os
base=r'X:/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx_converted'
a=sorted(os.listdir(base))[0]; s=sorted(os.listdir(os.path.join(base,a)))[0]
d=os.path.join(base,a,s)
for f in ['trials.parquet','neural.parquet','daq.parquet']:
    df=pd.read_parquet(os.path.join(d,f)); print(f, list(df.columns)[:25]); print(df.head(2)); print()
print('clusters.csv', pd.read_csv(os.path.join(d,'clusters.csv')).columns.tolist())
"
```

Then write `src/visdetect/analysis/tf_glm_data.py`. Map the discovered columns into the structure below (adjust the column-name constants `COL_*` to match the printout — the schema is authoritative, the names here are the expected defaults):

```python
# src/visdetect/analysis/tf_glm_data.py
"""Adapters that turn a data source into tf_glm TrialRegressors + spike times.

Two sources:
  - load_khilkevich_session(): the paper's npx_converted parquet/csv sessions
    (full 19-regressor positive control).
  - session_trial_regressors(): a visdetect Session (BG_046/BG_039 reduced set).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

from visdetect.analysis.tf_glm import TFGLMConfig, TrialRegressors, trial_bin_edges

# Column-name constants — VERIFY against the Step-3 schema printout and edit.
COL_UNIT = "cluster_id"
COL_SPIKE_T = "spike_time"
COL_REGION = "region"
COL_CHANGE_SIZE = "change_size"
COL_BASELINE_ON = "baseline_on"
COL_CHANGE_ON = "change_on"
COL_OUTCOME = "outcome"


@dataclass
class KhilSession:
    units: Dict[int, np.ndarray]
    regions: Dict[int, str]
    trials: pd.DataFrame
    licks: np.ndarray
    baseline_on: np.ndarray
    change_on: np.ndarray
    valve: np.ndarray
    airpuff: np.ndarray
    stim: pd.DataFrame
    movement: dict
    running: np.ndarray


def _read(d: Path, name_parquet: str, name_csv: str) -> pd.DataFrame:
    p = d / name_parquet
    if p.exists():
        return pd.read_parquet(p)
    return pd.read_csv(d / name_csv)


def load_khilkevich_session(session_dir) -> KhilSession:
    d = Path(session_dir)
    neural = _read(d, "neural.parquet", "spikes.csv")
    trials = _read(d, "trials.parquet", "trials.csv")
    daq = _read(d, "daq.parquet", "daq.csv") if (d / "daq.parquet").exists() else None
    clusters = pd.read_csv(d / "clusters.csv")
    units, regions = {}, {}
    for uid, g in neural.groupby(COL_UNIT):
        units[int(uid)] = np.sort(g[COL_SPIKE_T].to_numpy(float))
    for _, r in clusters.iterrows():
        if COL_REGION in clusters.columns:
            regions[int(r[COL_UNIT])] = str(r[COL_REGION])

    def _daq(channel_csv):
        fp = d / channel_csv
        if fp.exists():
            col = pd.read_csv(fp)
            return col.iloc[:, 0].to_numpy(float)
        return np.zeros(0)

    licks = _daq("daq_Lick_L.csv")
    baseline_on = _daq("daq_Baseline_ON.csv")
    change_on = _daq("daq_Change_ON.csv")
    valve = _daq("daq_Valve_L.csv")
    airpuff = _daq("daq_Air_puff.csv")
    stim = _read(d, "stim.parquet", "stim.csv")
    running = pd.read_csv(d / "running.csv").to_numpy(float) if (d / "running.csv").exists() else np.zeros((0, 2))
    movement = {}
    mp = d / "movement.pkl"
    if mp.exists():
        import pickle
        with open(mp, "rb") as f:
            movement = pickle.load(f)
    return KhilSession(units=units, regions=regions, trials=trials, licks=licks,
                       baseline_on=baseline_on, change_on=change_on, valve=valve,
                       airpuff=airpuff, stim=stim, movement=movement, running=running)
```

(The `khilkevich_trial_regressors` builder is written in Task 9, where the per-trial mapping is exercised by the run.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py -m pytest tests/analysis/test_tf_glm_data_khilkevich.py -v`
Expected: PASS (1 passed) — or SKIP if ceph isn't mounted. If it fails on a missing column, fix the `COL_*` constants to match the Step-3 schema printout.

- [ ] **Step 5: Commit**

```bash
cd /e/python_analysis/git_repos/vd_tf_phase0
git add src/visdetect/analysis/tf_glm_data.py tests/analysis/test_tf_glm_data_khilkevich.py
git commit -m "feat(tf_glm): Khilkevich npx_converted loader adapter"
```

---

## Task 9: Positive-control run — reproduce 5–45% on Khilkevich data

**Files:**
- Create: `scripts/tf_responsiveness/run_tf_glm_khilkevich.py`
- Modify: `src/visdetect/analysis/tf_glm_data.py` (add `khilkevich_trial_regressors`)
- Test: manual run (no unit test — it's a data-driven validation)

**Interfaces:**
- Consumes: `load_khilkevich_session`, all of `tf_glm`.
- Produces: per-region TF-responsive fraction CSV + a printed comparison to the paper's 5–45%.

- [ ] **Step 1: Implement `khilkevich_trial_regressors`**

Append to `src/visdetect/analysis/tf_glm_data.py`. Build `TrialRegressors` per trial on the neural clock: `t_start = baseline_on[i]`, `t_end = next baseline_on / trial end`, `change_time = change_on[i]` (NaN if not present), `change_size` from `trials[COL_CHANGE_SIZE]`, `tf_bins` resampled from `stim` (per-frame TF on the trial's bins via the `vbl`→neural map anchored at `t_start`), `lick_times` = licks within the trial, `reward_time` = first `valve` in trial, `wheel_bins` from `running` resampled, `phase_bins` from `stim` per-frame phase. Return `(trials_regs, {uid: units[uid] for uid in region_units})`.

```python
def _resample_to_bins(times, values, bin_edges, bin_s, fill=0.0):
    """Mean of a (times,values) signal within each bin; fill empties."""
    if bin_edges.size == 0 or np.asarray(times).size == 0:
        return np.full(bin_edges.size, fill)
    idx = np.floor((np.asarray(times) - bin_edges[0]) / bin_s + 1e-9).astype(int)
    out = np.full(bin_edges.size, fill); cnt = np.zeros(bin_edges.size)
    ok = (idx >= 0) & (idx < bin_edges.size)
    np.add.at(out, idx[ok], np.asarray(values)[ok] - fill)  # accumulate
    np.add.at(cnt, idx[ok], 1.0)
    nz = cnt > 0
    out[nz] = fill + out[nz] / cnt[nz]
    return out


def khilkevich_trial_regressors(ks: KhilSession, cfg: TFGLMConfig, region=None):
    bs = cfg.bin_s
    bon = np.sort(ks.baseline_on)
    n = bon.size
    # trial ends: next baseline on, last trial gets +max baseline duration
    ends = np.append(bon[1:], bon[-1] + 20.0) if n else np.zeros(0)
    # per-frame stim arrays (neural clock already, if converted; else vbl-anchored)
    stim = ks.stim
    s_t = stim["vbl"].to_numpy(float) if "vbl" in stim.columns else stim.iloc[:, 0].to_numpy(float)
    s_tf = stim["TF"].to_numpy(float) if "TF" in stim.columns else stim.iloc[:, 1].to_numpy(float)
    s_phase = stim["phase"].to_numpy(float) if "phase" in stim.columns else np.full(s_t.size, np.nan)
    run_t = ks.running[:, 0] if ks.running.size else np.zeros(0)
    run_v = np.abs(np.diff(ks.running[:, 1], prepend=ks.running[0, 1])) if ks.running.size else np.zeros(0)

    trials_regs = []
    cs_col = ks.trials[COL_CHANGE_SIZE].to_numpy(float) if COL_CHANGE_SIZE in ks.trials.columns else np.ones(n)
    con = ks.change_on
    for i in range(n):
        t0, t1 = bon[i], ends[i]
        edges = trial_bin_edges(t0, t1, bs)
        tf_bins = _resample_to_bins(s_t, s_tf, edges, bs, fill=0.0)
        phase_bins = _resample_to_bins(s_t, s_phase, edges, bs, fill=np.nan)
        wheel_bins = _resample_to_bins(run_t, run_v, edges, bs, fill=0.0)
        licks = ks.licks[(ks.licks >= t0) & (ks.licks < t1)]
        rew = ks.valve[(ks.valve >= t0) & (ks.valve < t1)]
        ch = con[i] if i < con.size and np.isfinite(con[i]) else np.nan
        trials_regs.append(TrialRegressors(
            t_start=t0, t_end=t1, change_time=ch,
            change_size=float(cs_col[i]) if i < cs_col.size else 1.0,
            tf_bins=tf_bins, lick_times=licks,
            reward_time=float(rew[0]) if rew.size else np.nan,
            abort_time=np.nan, wheel_bins=wheel_bins,
            phase_bins=phase_bins if cfg.include_phase else None))
    if region is None:
        unit_ids = list(ks.units.keys())
    else:
        unit_ids = [u for u in ks.units if ks.regions.get(u, "") == region]
    return trials_regs, {u: ks.units[u] for u in unit_ids}
```

- [ ] **Step 2: Write the run script**

```python
# scripts/tf_responsiveness/run_tf_glm_khilkevich.py
"""Positive control: run the TF-encoding GLM on Khilkevich npx_converted data
and compare the per-region TF-responsive fraction to the paper's 5-45%."""
import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd

from visdetect.analysis.tf_glm import (TFGLMConfig, assemble_design, count_vector,
    fit_poisson_cv, make_trial_folds, identify_tf_responsive)
from visdetect.analysis.tf_glm_data import (load_khilkevich_session,
    khilkevich_trial_regressors)

BASE = r"X:/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx_converted"


def run_session(session_dir, region, cfg, max_units=None):
    ks = load_khilkevich_session(session_dir)
    trials, units = khilkevich_trial_regressors(ks, cfg, region=region)
    design = assemble_design(trials, cfg)
    folds = make_trial_folds(design.trial_index, cfg.n_folds, cfg.seed)
    Xr = design.X.copy(); Xr[:, design.col_groups["tf"]] = 0.0
    rows = []
    uids = list(units)[:max_units] if max_units else list(units)
    for uid in uids:
        y = count_vector(trials, units[uid], design)
        if y.sum() < 100:        # skip near-silent units
            continue
        full = fit_poisson_cv(design.X, y, cfg, folds)
        red = fit_poisson_cv(Xr, y, cfg, folds)
        out = identify_tf_responsive(design, y, full, red, cfg)
        out["unit"] = uid; out["region"] = region
        rows.append(out)
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--session-dir", required=True)
    p.add_argument("--region", required=True, help="region label to analyse (e.g. VISp, CP)")
    p.add_argument("--max-units", type=int, default=None)
    p.add_argument("--include-phase", action="store_true")
    p.add_argument("--out", default="data/cache/tf_glm/khilkevich_posctrl.csv")
    a = p.parse_args(argv)
    cfg = TFGLMConfig(include_phase=a.include_phase)
    df = run_session(a.session_dir, a.region, cfg, a.max_units)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    frac = 100.0 * df["is_responsive"].mean() if len(df) else float("nan")
    print(f"\n{a.region}: {len(df)} units, TF-responsive = {frac:.1f}%  (paper: 5-45%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run on a visual-cortex session (expect high fraction)**

First find a session/region with units (use the Step-3 schema knowledge of region labels). Run, e.g.:
```bash
cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py scripts/tf_responsiveness/run_tf_glm_khilkevich.py \
  --session-dir "X:/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx_converted/<animal>/<session>" \
  --region VISp --include-phase --max-units 40
```
Expected: a printed `VISp: N units, TF-responsive = XX.X% (paper: 5-45%)` with XX in a plausibly-high range for visual cortex. **This is the gold-standard gate**: if a visual area lands well below 5% with all regressors present, the implementation has a bug — debug before trusting any BG_046 result. Iterate on `khilkevich_trial_regressors` (esp. the `tf_bins` resampling and the change/lick mapping) until cortex reproduces the paper's ballpark.

- [ ] **Step 4: Run on a basal-ganglia region (the paper's BG positive)**

Same command with `--region CP` (caudoputamen) or the dataset's striatal label. Expect a non-zero fraction within 5–45%. Save both CSVs.

- [ ] **Step 5: Commit**

```bash
cd /e/python_analysis/git_repos/vd_tf_phase0
git add src/visdetect/analysis/tf_glm_data.py scripts/tf_responsiveness/run_tf_glm_khilkevich.py
git commit -m "feat(tf_glm): Khilkevich positive-control run (reproduce 5-45% TF-responsive)"
```

---

## Task 10: Apply to BG_046 (DMS) + BG_039 (cortex)

**Files:**
- Modify: `src/visdetect/analysis/tf_glm_data.py` (add `session_trial_regressors`)
- Create: `scripts/tf_responsiveness/run_tf_glm.py`
- Test: `tests/analysis/test_tf_glm_data_session.py`

**Interfaces:**
- Consumes: visdetect `Session` (now phase-backfilled), `get_good_cluster_ids`, all of `tf_glm`.
- Produces: `session_trial_regressors(session, cfg) -> (List[TrialRegressors], Dict[int,np.ndarray])` — reduced regressor set from a visdetect Session (lick from `Piezo_1`, wheel from `Rot_enc_A/B`, reward from finite `Valve_L`, change from finite `Change_ON`, TF from `baseline_values`; phase optional from `stim_phase`+`stim_vbl`).
- Produces: per-session/per-subject TF-responsive table + a summary fraction.

- [ ] **Step 1: Write the failing test (uses a real BG_046 session)**

```python
# tests/analysis/test_tf_glm_data_session.py
import os, numpy as np, pytest
from visdetect.core.session import load_session
from visdetect.analysis.tf_glm import TFGLMConfig
from visdetect.analysis.tf_glm_data import session_trial_regressors

PKL = r"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/data/pkls/BG_046/BG_046_01072025.pkl"

@pytest.mark.skipif(not os.path.exists(PKL), reason="pkl not present")
def test_session_regressors_shapes():
    s = load_session(PKL)
    cfg = TFGLMConfig()
    trials, units = session_trial_regressors(s, cfg)
    assert len(trials) == len(s.trials)
    # change_time finite only for change-reached trials (Hit/Miss/Ref)
    finite_change = sum(np.isfinite(t.change_time) for t in trials)
    assert finite_change > 0 and finite_change < len(trials)
    # tf_bins length matches each trial's bin count
    t0 = trials[0]
    assert t0.tf_bins.ndim == 1 and t0.wheel_bins.ndim == 1
    assert len(units) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py -m pytest tests/analysis/test_tf_glm_data_session.py -v`
Expected: FAIL (`ImportError: cannot import name 'session_trial_regressors'`).

- [ ] **Step 3: Implement `session_trial_regressors`**

Append to `src/visdetect/analysis/tf_glm_data.py`:

```python
def _baseline_tf_bins(trial, t0, t_change, edges, bin_s):
    """Per-bin baseline TF (linear) on the trial's bins from baseline_values
    (St1TrialVector, stride 3 -> 50 ms grid), zeros after change onset."""
    bv = np.asarray(trial.baseline_values, float).ravel() if trial.baseline_values is not None else np.zeros(0)
    tf50 = bv[::3] if bv.size else np.zeros(0)          # 50-ms grid from baseline start
    out = np.zeros(edges.size)
    rel_idx = np.floor((edges - t0) / bin_s + 1e-9).astype(int)
    ok = (rel_idx >= 0) & (rel_idx < tf50.size)
    out[ok] = tf50[rel_idx[ok]]
    if np.isfinite(t_change):
        out[edges >= t_change] = 0.0
    return out


def session_trial_regressors(session, cfg: TFGLMConfig):
    from visdetect.analysis.tf_glm import trial_bin_edges
    ne = session.ni_events or {}
    bon = np.asarray(ne.get("Baseline_ON", []), float)
    con = np.asarray(ne.get("Change_ON", []), float)
    valve = np.asarray(ne.get("Valve_L", []), float)
    licks = np.asarray(ne.get("Piezo_1", []), float)
    enc_a = np.asarray(ne.get("Rot_enc_A", []), float)
    bs = cfg.bin_s
    n = len(session.trials)
    ends = np.append(bon[1:], bon[-1] + 20.0) if bon.size else np.zeros(0)
    trials_regs = []
    for i, tr in enumerate(session.trials):
        t0 = bon[i] if i < bon.size else np.nan
        t1 = ends[i] if i < ends.size else (t0 + 20.0)
        if not np.isfinite(t0):
            continue
        edges = trial_bin_edges(t0, t1, bs)
        t_change = con[i] if (i < con.size and np.isfinite(con[i])) else np.nan
        tf_bins = _baseline_tf_bins(tr, t0, t_change, edges, bs)
        # wheel speed: encoder tick density per bin (proxy for speed)
        wheel = _resample_to_bins(enc_a, np.ones(enc_a.size), edges, bs, fill=0.0) if enc_a.size else np.zeros(edges.size)
        trl_licks = licks[(licks >= t0) & (licks < t1)]
        rew = valve[(valve >= t0) & (valve < t1) & np.isfinite(valve)] if valve.size else np.zeros(0)
        abort_t = t_change if str(tr.trialoutcome).lower() == "abort" else np.nan
        # phase (optional)
        phase_bins = None
        if cfg.include_phase and tr.stim_phase is not None and tr.stim_vbl is not None:
            vbl = np.asarray(tr.stim_vbl, float).ravel()
            ph = np.asarray(tr.stim_phase, float)
            ph1 = ph[:, 0] if ph.ndim == 2 else ph
            neural_t = t0 + (vbl - vbl[0])           # anchor first frame at Baseline_ON
            phase_bins = _resample_to_bins(neural_t, ph1 % 360, edges, bs, fill=np.nan)
        trials_regs.append(TrialRegressors(
            t_start=t0, t_end=t1, change_time=t_change,
            change_size=float(tr.change_size) if tr.change_size is not None else 1.0,
            tf_bins=tf_bins, lick_times=trl_licks,
            reward_time=float(rew[0]) if rew.size else np.nan,
            abort_time=abort_t, wheel_bins=wheel, phase_bins=phase_bins))
    # units
    from visdetect.analysis.tf_glm import TFGLMConfig as _C  # noqa
    try:
        from utils import get_good_cluster_ids  # analysis_suite flat import
        ids = get_good_cluster_ids(session)
    except Exception:
        ids = session.good_and_stable_ids or session.good_cluster_ids or [c.cluster_id for c in session.clusters]
    by_id = {int(c.cluster_id): np.asarray(c.spike_times, float).ravel() for c in session.clusters}
    units = {int(i): by_id[int(i)] for i in ids if int(i) in by_id}
    return trials_regs, units
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py -m pytest tests/analysis/test_tf_glm_data_session.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Write the BG run driver**

```python
# scripts/tf_responsiveness/run_tf_glm.py
"""Run the TF-encoding GLM on BG_046 (DMS) / BG_039 (cortex) reduced regressor
set; output per-unit TF-responsive table + per-subject fraction."""
import argparse, glob, gc, sys
from pathlib import Path
import numpy as np, pandas as pd

from visdetect.core.session import load_session
from visdetect.analysis.tf_glm import (TFGLMConfig, assemble_design, count_vector,
    fit_poisson_cv, make_trial_folds, identify_tf_responsive)
from visdetect.analysis.tf_glm_data import session_trial_regressors


def run_one(pkl, cfg):
    s = load_session(pkl)
    trials, units = session_trial_regressors(s, cfg)
    design = assemble_design(trials, cfg)
    folds = make_trial_folds(design.trial_index, cfg.n_folds, cfg.seed)
    Xr = design.X.copy(); Xr[:, design.col_groups["tf"]] = 0.0
    rows = []
    for uid, st in units.items():
        y = count_vector(trials, st, design)
        if y.sum() < 100:
            continue
        full = fit_poisson_cv(design.X, y, cfg, folds)
        red = fit_poisson_cv(Xr, y, cfg, folds)
        out = identify_tf_responsive(design, y, full, red, cfg)
        out["unit"] = uid; out["session"] = Path(pkl).stem
        rows.append(out)
    del s; gc.collect()
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pkl-dir", required=True)
    p.add_argument("--include-phase", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="first N sessions")
    p.add_argument("--out", default="data/cache/tf_glm/bg_tf_glm.csv")
    a = p.parse_args(argv)
    cfg = TFGLMConfig(include_phase=a.include_phase)
    pkls = sorted(glob.glob(str(Path(a.pkl_dir) / "*.pkl")))[: a.limit]
    all_rows = []
    for pkl in pkls:
        df = run_one(pkl, cfg)
        if len(df):
            frac = 100.0 * df["is_responsive"].mean()
            print(f"{Path(pkl).stem}: {len(df)} units, TF-resp {frac:.1f}%")
            all_rows.append(df)
    res = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    if len(res):
        print(f"\nOVERALL: {len(res)} units, TF-responsive = {100*res['is_responsive'].mean():.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Smoke-run on 2 BG_046 sessions, then the full set**

```bash
cd /e/python_analysis/git_repos/vd_tf_phase0 && PYTHONPATH=src py scripts/tf_responsiveness/run_tf_glm.py \
  --pkl-dir data/pkls/BG_046 --limit 2 --out data/cache/tf_glm/bg046_smoke.csv
```
Expected: two `BG_046_...: N units, TF-resp X.X%` lines without error. Then run the full BG_046 set (drop `--limit`) and BG_039 (`--pkl-dir data/pkls/BG_039`, add `--include-phase`). Compare the DMS/cortex fractions to the Khilkevich positive control.

- [ ] **Step 7: Commit**

```bash
cd /e/python_analysis/git_repos/vd_tf_phase0
git add src/visdetect/analysis/tf_glm_data.py scripts/tf_responsiveness/run_tf_glm.py tests/analysis/test_tf_glm_data_session.py
git commit -m "feat(tf_glm): BG_046/BG_039 reduced-set run driver + session regressor adapter"
```

---

## Task 11: Results note + memory update

**Files:**
- Create: `docs/science/2026-06-tf-glm-results.md`
- Update: memory `tf_responsiveness_null_finding_jun2026.md` + MEMORY.md index

- [ ] **Step 1: Write the results note** comparing (a) Khilkevich positive control vs their 5–45%, (b) BG_046 DMS fraction, (c) BG_039 cortex fraction, with the interpretation gate: positive-control-passes → BG fractions are trustworthy; DMS positive → TF direction revived; DMS null *with lick control* → strong null. Include the sanity checks actually run (lick-prep/exec recovery %, optional shuffle-null collapse).

- [ ] **Step 2: Update memory** with the outcome (one or two sentences; keep index line < 200 chars) and link `[[tf_responsiveness_null_finding_jun2026]]`, `[[paper-khilkevich-lohse-2024-brainwide]]`.

- [ ] **Step 3: Commit, then offer to merge the branch to main.**

---

## Self-Review

**Spec coverage:** §2 exact model → Tasks 3–6; §2.2 C1/C2 → Task 7; §3 reduced BG set (Change_ON, licks, wheel, reward, no airpuff, phase) → Tasks 5,10; phase from raw → Tasks 1–2,10; §6 positive control (ceph path, full 19-reg, movement.pkl) → Tasks 8–9; §6 internal checks (lick recovery, optional shuffle, cortex>DMS) → Tasks 9–10 + 11; §5 deviations (no ME/pupil on BG) → reduced set in Task 10. **Gap noted:** motion-energy/pupil regressors are only built for the Khilkevich source (full set); BG runs omit them by design (spec §5) — acceptable. Grating-phase indicator is implemented (Task 5 `_phase_indicator`) and wired for both sources behind `cfg.include_phase`.

**Placeholder scan:** no TBD/TODO; every code step has concrete code; the one explicit "verify against schema" step (Task 8 Step 3) is a real inspection command, not a placeholder — the loader has working defaults.

**Type consistency:** `TrialRegressors`, `DesignMatrix`, `FitResult`, `TFGLMConfig` field names are reused verbatim across Tasks 5–10; `col_groups["tf"]` (the ablation target) is referenced consistently; `make_trial_folds`/`fit_poisson_cv`/`identify_tf_responsive` signatures match between definition (Tasks 6–7) and callers (Tasks 9–10).

**Known risk flagged for execution:** the `tf_bins` neural-clock resampling (Khilkevich `vbl`-anchored vs BG `baseline_values`-indexed) and the exact dataset column names are the two places most likely to need iteration; Task 9 Step 3 makes cortex-reproduces-5–45% the explicit gate that catches resampling bugs before any BG interpretation.
