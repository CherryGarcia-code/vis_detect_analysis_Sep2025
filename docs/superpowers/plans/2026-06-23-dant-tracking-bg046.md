# DANT Cross-Session Tracking on BG_046 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run DANT (pyDANT) on BG_046's 42 already-extracted sessions and produce a figure-backed read-out of cross-session tracking quality, benchmarked against the existing UnitMatch registry.

**Architecture:** A thin repo-side adapter converts our extracted RawWaveforms + pkl spike trains into DANT's input layout; the unmodified `pyDANT` package (in a dedicated venv) does the tracking; a converter normalizes DANT's output to a long registry; an evaluation harness compares DANT vs UnitMatch and computes a held-out ISI AUC, saving presentation-ready figures.

**Tech Stack:** Python 3.10, numpy, pandas, scikit-learn, matplotlib (analysis venv); `pyDANT` + `hdbscan` (dedicated `.venv_dant`); `visdetect` (read pkls); spec at `docs/superpowers/specs/2026-06-23-dant-tracking-bg046-design.md`.

## Global Constraints

- **Worktree:** `E:/python_analysis/git_repos/vd_dant` on branch `feature/dant-tracking`. All paths below are relative to it unless absolute.
- **Primary repo (read-only data source):** `E:/python_analysis/git_repos/vis_detect_analysis_Sep2025`.
- **Analysis interpreter** (`ANALYSIS_PY`): `E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe` — has `visdetect` (editable, pinned to primary `src`/main), numpy, pandas, sklearn, scipy, matplotlib, pytest. Used for adapter, registry, eval, and ALL tests.
- **DANT interpreter** (`DANT_PY`): `E:/python_analysis/git_repos/vd_dant/.venv_dant/Scripts/python.exe` — used ONLY for the `runDANTMultiShank` call.
- **NO compute over `X:`** (Samba). All inputs are local (`data/unit_match/input/BG_046/`, `data/pkls/BG_046/`, `data/unit_match/output/BG_046_um329_CellRegistry.csv` in the primary repo). NO junctions (avoid the worktree-remove data-loss hazard) — pass primary paths as args.
- **Identity features: Waveform + ACG only. NO PETH.** (Avoids functional circularity; comparable to UnitMatch.)
- **Multi-shank** run (`runDANTMultiShank`), NP2.0 4-shank.
- **Spike times → milliseconds** (×1000) before writing for DANT.
- **Reproducibility:** `np.random.seed(42)` before the DANT run.
- **Opus 4.8 for every subagent/dispatch.** Never downgrade.
- **Visualize at every helpful step;** figures presentation-ready under `FIGURES/tracking_dant/BG_046/`.
- **Every commit message** ends with the trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` (shown via a second `-m` in commit commands).
- **Run tests from the worktree root** so `tests/tracking_dant/conftest.py` resolves the package path.

## File Structure

```
scripts/tracking_dant/
  __init__.py            # marks package
  adapter.py             # PURE helpers: collapse_cv, derive_channel_shanks, seconds_to_ms, is_positive_going
  registry.py            # PURE helpers: idxcluster_to_registry, tracked_lengths, survival_function, comembership_agreement, melt_cellregistry
  build_dant_inputs.py   # CLI: our data -> DANT input layout (uses adapter + visdetect)
  settings_bg046.json    # DANT config (hjson)
  run_dant_bg046.py      # CLI: seed + load settings + runDANTMultiShank (DANT_PY)
  dant_to_registry.py    # CLI: Output -> dant_registry.csv (uses registry)
  evaluate_dant.py       # CLI: comparison + ISI AUC + figures (uses registry)
  README.md              # how to run, env, paths
tests/tracking_dant/
  conftest.py            # adds scripts/tracking_dant to sys.path
  test_adapter.py        # unit tests for adapter.py
  test_registry.py       # unit tests for registry.py
data/cache/dant/BG_046/input/   # waveform_all.npy, session_index.npy, channel_locations.npy, channel_shanks.npy, spike_times/, unit_lookup.csv
FIGURES/tracking_dant/BG_046/    # diagnostics, comparison, example tracks (+ dant_output/)
```

---

### Task 1: Workspace, dedicated venv, and package skeleton

**Files:**
- Create: `scripts/tracking_dant/__init__.py`
- Create: `tests/tracking_dant/conftest.py`
- Create (dir): `.venv_dant/`, `data/cache/dant/BG_046/`, `FIGURES/tracking_dant/BG_046/`

**Interfaces:**
- Produces: a working `.venv_dant` with `pyDANT` importable; an importable `scripts.tracking_dant` package path for tests.

- [ ] **Step 1: Create output/package directories**

Run (Bash, from worktree root):
```bash
mkdir -p scripts/tracking_dant tests/tracking_dant data/cache/dant/BG_046 FIGURES/tracking_dant/BG_046
touch scripts/tracking_dant/__init__.py
```

- [ ] **Step 2: Create the dedicated DANT venv and install pyDANT**

Run (Bash, from worktree root):
```bash
"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe" -m venv .venv_dant
.venv_dant/Scripts/python.exe -m pip install --upgrade pip
.venv_dant/Scripts/python.exe -m pip install pyDANT
```
Expected: pip installs `pyDANT` and deps (`hdbscan`, `scikit-learn`, `scipy`, `joblib`, `tqdm`, `h5py`, `matplotlib`, `hjson`) without error.

- [ ] **Step 3: Verify pyDANT + hdbscan import**

Run:
```bash
.venv_dant/Scripts/python.exe -c "import hdbscan, pyDANT; from pyDANT import runDANTMultiShank, runDANT; print('pyDANT OK', hdbscan.__version__)"
```
Expected: prints `pyDANT OK <version>` with no ImportError. If `hdbscan` fails to build, install a prebuilt wheel: `.venv_dant/Scripts/python.exe -m pip install hdbscan --only-binary :all:` and re-run.

- [ ] **Step 4: Write the test conftest (path bootstrap)**

Create `tests/tracking_dant/conftest.py`:
```python
import pathlib
import sys

# Make the tracking_dant package importable by bare module name in tests.
_PKG = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "tracking_dant"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
```

- [ ] **Step 5: Commit**

```bash
git add scripts/tracking_dant/__init__.py tests/tracking_dant/conftest.py
git commit -m "chore(dant): scaffold tracking_dant package + dedicated venv" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
(`.venv_dant/`, `data/`, `FIGURES/` are gitignored — only the two source files are committed.)

---

### Task 2: Adapter pure helpers (TDD)

**Files:**
- Create: `scripts/tracking_dant/adapter.py`
- Test: `tests/tracking_dant/test_adapter.py`

**Interfaces:**
- Produces (consumed by `build_dant_inputs.py`):
  - `collapse_cv(raw_spikes: np.ndarray) -> np.ndarray` — `(n_samp, n_ch, 2)` → `(n_ch, n_samp)` (mean over CV, transpose)
  - `derive_channel_shanks(channel_positions: np.ndarray, gap_um: float = 150.0) -> np.ndarray` — `(n_ch, 2)` → `(n_ch,)` int64 shank ids (0-based)
  - `seconds_to_ms(spike_times: np.ndarray) -> np.ndarray`
  - `is_positive_going(waveform: np.ndarray) -> bool` — `waveform` is `(n_ch, n_samp)`

- [ ] **Step 1: Write the failing tests**

Create `tests/tracking_dant/test_adapter.py`:
```python
import numpy as np
import pytest

import adapter


def test_collapse_cv_shape_and_mean():
    # (n_samp=4, n_ch=3, n_cv=2)
    raw = np.zeros((4, 3, 2), dtype=np.float32)
    raw[..., 0] = 1.0
    raw[..., 1] = 3.0  # mean over cv -> 2.0 everywhere
    out = adapter.collapse_cv(raw)
    assert out.shape == (3, 4)            # (n_ch, n_samp)
    assert np.allclose(out, 2.0)


def test_collapse_cv_rejects_bad_shape():
    with pytest.raises(ValueError):
        adapter.collapse_cv(np.zeros((4, 3)))         # not 3D
    with pytest.raises(ValueError):
        adapter.collapse_cv(np.zeros((4, 3, 5)))      # cv axis != 2


def test_derive_channel_shanks_four_shanks():
    # BG_046 x-layout: 4 shanks x 2 columns, ~250 um apart
    xs = np.array([27, 59, 277, 309, 527, 559, 777, 809], dtype=float)
    pos = np.column_stack([xs, np.zeros_like(xs)])
    shanks = adapter.derive_channel_shanks(pos)
    assert shanks.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert shanks.dtype == np.int64


def test_seconds_to_ms():
    out = adapter.seconds_to_ms(np.array([0.0, 1.0, 2.5]))
    assert np.allclose(out, [0.0, 1000.0, 2500.0])


def test_is_positive_going():
    # negative-going: trough deeper than peak on the peak channel
    neg = np.zeros((2, 10)); neg[0] = -5.0; neg[0, 5] = -10.0; neg[0, 0] = 2.0
    assert adapter.is_positive_going(neg) is False
    # positive-going: peak taller than trough on the peak channel
    pos = np.zeros((2, 10)); pos[0, 5] = 10.0; pos[0, 0] = -2.0
    assert adapter.is_positive_going(pos) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run (from worktree root):
```bash
"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe" -m pytest tests/tracking_dant/test_adapter.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'adapter'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/tracking_dant/adapter.py`:
```python
"""Pure helpers converting visdetect-extracted unit data into DANT's input conventions.

No file I/O here — these operate on arrays so they are unit-testable. Orchestration
(reading RawWaveforms/pkls, writing the DANT input folder) lives in build_dant_inputs.py.
"""
import numpy as np


def collapse_cv(raw_spikes):
    """RawWaveforms (n_samp, n_ch, n_cv=2) -> DANT waveform (n_ch, n_samp).

    Averages the two cross-validation halves, then transposes so the channel axis
    is first (DANT's waveform_all is (n_unit, n_channel, n_sample)).
    """
    arr = np.asarray(raw_spikes, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(f"expected (n_samp, n_ch, 2), got shape {arr.shape}")
    mean_wave = arr.mean(axis=2)        # (n_samp, n_ch)
    return mean_wave.T                  # (n_ch, n_samp)


def derive_channel_shanks(channel_positions, gap_um=150.0):
    """(n_ch, 2) x/y positions -> (n_ch,) 0-based shank id, grouping x by gaps."""
    pos = np.asarray(channel_positions, dtype=float)
    x = pos[:, 0]
    ux = np.unique(x)
    shank_of_ux = np.zeros(len(ux), dtype=np.int64)
    cur = 0
    for i in range(1, len(ux)):
        if ux[i] - ux[i - 1] > gap_um:
            cur += 1
        shank_of_ux[i] = cur
    mapping = {val: s for val, s in zip(ux, shank_of_ux)}
    return np.array([mapping[v] for v in x], dtype=np.int64)


def seconds_to_ms(spike_times):
    """Spike times in seconds -> milliseconds (DANT ACG/ISI bins are in ms)."""
    return np.asarray(spike_times, dtype=np.float64) * 1000.0


def is_positive_going(waveform):
    """True if the peak channel's waveform is positive-going (|max| > |min|).

    waveform: (n_ch, n_samp). DANT's trough-centering assumes negative-going spikes,
    so positive-going units should be excluded before centering.
    """
    w = np.asarray(waveform, dtype=float)
    ptp = w.max(axis=1) - w.min(axis=1)
    peak = int(np.argmax(ptp))
    return abs(float(w[peak].max())) > abs(float(w[peak].min()))
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe" -m pytest tests/tracking_dant/test_adapter.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/tracking_dant/adapter.py tests/tracking_dant/test_adapter.py
git commit -m "feat(dant): adapter pure helpers (cv-collapse, shanks, ms, positivity)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Registry conversion + comparison metrics (TDD)

**Files:**
- Create: `scripts/tracking_dant/registry.py`
- Test: `tests/tracking_dant/test_registry.py`

**Interfaces:**
- Produces:
  - `idxcluster_to_registry(idx_cluster: np.ndarray, lookup: pd.DataFrame) -> pd.DataFrame` — long `[session, ks_unit_id, dant_uid]`
  - `tracked_lengths(registry: pd.DataFrame, uid_col: str = "dant_uid") -> pd.Series` — uid → #distinct sessions (tracked uids only, uid > 0)
  - `survival_function(lengths, n_sessions: int) -> tuple[np.ndarray, np.ndarray]` — (k, fraction tracked ≥ k)
  - `comembership_agreement(reg_a, reg_b, uid_a, uid_b) -> dict` — `{n_shared, ari, pairwise_precision, pairwise_recall}`
  - `melt_cellregistry(wide: pd.DataFrame, uid_col: str = "UID") -> pd.DataFrame` — UnitMatch wide CellRegistry → long `[session, ks_unit_id, um_uid]`
- Consumes: `lookup` has columns `[pooled_index, session, ks_unit_id]` in pooled-index order.

- [ ] **Step 1: Write the failing tests**

Create `tests/tracking_dant/test_registry.py`:
```python
import numpy as np
import pandas as pd
import pytest

import registry


def _lookup():
    return pd.DataFrame({
        "pooled_index": [0, 1, 2, 3],
        "session": ["01072025", "02072025", "01072025", "02072025"],
        "ks_unit_id": [10, 11, 12, 13],
    })


def test_idxcluster_to_registry_basic():
    idx = np.array([1, 1, -1, 2])   # units 0&1 = neuron 1; unit 2 untracked; unit 3 = neuron 2
    out = registry.idxcluster_to_registry(idx, _lookup())
    assert list(out.columns) == ["session", "ks_unit_id", "dant_uid"]
    assert out.loc[out.ks_unit_id == 10, "dant_uid"].item() == 1
    assert out.loc[out.ks_unit_id == 12, "dant_uid"].item() == -1


def test_idxcluster_to_registry_rejects_length_mismatch():
    with pytest.raises(ValueError):
        registry.idxcluster_to_registry(np.array([1, 2]), _lookup())


def test_tracked_lengths_counts_distinct_sessions():
    reg = pd.DataFrame({
        "session": ["a", "b", "a", "b", "a"],
        "ks_unit_id": [1, 2, 3, 4, 5],
        "dant_uid": [1, 1, 2, -1, 1],   # uid 1 spans sessions a,b (unit 5 also session a)
    })
    lengths = registry.tracked_lengths(reg)
    assert lengths[1] == 2     # sessions a, b
    assert lengths[2] == 1
    assert -1 not in lengths.index


def test_survival_function():
    lengths = pd.Series([1, 2, 2, 3])
    ks, frac = registry.survival_function(lengths, n_sessions=3)
    assert ks.tolist() == [1, 2, 3]
    assert np.allclose(frac, [1.0, 0.75, 0.25])


def test_comembership_agreement_identical_is_one():
    reg_a = pd.DataFrame({"session": ["a", "b", "a"], "ks_unit_id": [1, 2, 3], "dant_uid": [1, 1, 2]})
    reg_b = reg_a.rename(columns={"dant_uid": "um_uid"})
    res = registry.comembership_agreement(reg_a, reg_b, "dant_uid", "um_uid")
    assert res["n_shared"] == 3
    assert res["ari"] == pytest.approx(1.0)
    assert res["pairwise_precision"] == pytest.approx(1.0)
    assert res["pairwise_recall"] == pytest.approx(1.0)


def test_melt_cellregistry():
    wide = pd.DataFrame({"UID": [7, 8], "01072025": [10, 0], "02072025": [11, 99]})
    # 0/NaN/empty cells mean "absent in this session"
    long = registry.melt_cellregistry(wide)
    row = long[(long.um_uid == 7) & (long.session == "02072025")]
    assert row.ks_unit_id.item() == 11
    assert ((long.um_uid == 8) & (long.session == "01072025")).sum() == 0  # 0 dropped
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe" -m pytest tests/tracking_dant/test_registry.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'registry'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/tracking_dant/registry.py`:
```python
"""DANT output -> comparable long registry, plus UnitMatch-comparison metrics. Pure functions."""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def idxcluster_to_registry(idx_cluster, lookup):
    """Per-unit cluster ids (-1 = untracked) + lookup -> long [session, ks_unit_id, dant_uid]."""
    idx = np.asarray(idx_cluster).astype(int)
    if len(idx) != len(lookup):
        raise ValueError(f"idx_cluster len {len(idx)} != lookup rows {len(lookup)}")
    df = lookup.copy().reset_index(drop=True)
    df["dant_uid"] = idx
    out = df[["session", "ks_unit_id", "dant_uid"]].reset_index(drop=True)
    if out.duplicated(subset=["session", "ks_unit_id"]).any():
        raise ValueError("duplicate (session, ks_unit_id) in registry")
    return out


def tracked_lengths(registry, uid_col="dant_uid"):
    """uid -> number of distinct sessions, for tracked uids (uid > 0)."""
    tracked = registry[registry[uid_col] > 0]
    return tracked.groupby(uid_col)["session"].nunique()


def survival_function(lengths, n_sessions):
    """(k, fraction of tracked neurons appearing in >= k sessions) for k=1..n_sessions."""
    lengths = np.asarray(lengths, dtype=float)
    ks = np.arange(1, n_sessions + 1)
    n = len(lengths)
    if n == 0:
        return ks, np.zeros(n_sessions)
    frac = np.array([(lengths >= k).sum() / n for k in ks])
    return ks, frac


def _relabel_singletons(labels):
    """Replace untracked (<=0) entries with unique negative singleton labels."""
    out = np.asarray(labels).astype(np.int64).copy()
    nxt = -1
    for i in range(len(out)):
        if out[i] <= 0:
            out[i] = nxt
            nxt -= 1
    return out


def _pair_count(sizes):
    sizes = np.asarray(sizes, dtype=np.int64)
    return int((sizes * (sizes - 1) // 2).sum())


def comembership_agreement(reg_a, reg_b, uid_a="dant_uid", uid_b="um_uid"):
    """Agreement between two registries on shared (session, ks_unit_id) units.

    Returns ARI plus pairwise precision/recall treating reg_b (UnitMatch) as reference:
    precision = (pairs same in BOTH) / (pairs same in A); recall = same / (pairs same in B).
    """
    reg_a = reg_a.drop_duplicates(["session", "ks_unit_id"])
    reg_b = reg_b.drop_duplicates(["session", "ks_unit_id"])
    a = reg_a.set_index(["session", "ks_unit_id"])[uid_a]
    b = reg_b.set_index(["session", "ks_unit_id"])[uid_b]
    shared = a.index.intersection(b.index)
    a = a.loc[shared]
    b = b.loc[shared]
    la = _relabel_singletons(a.to_numpy())
    lb = _relabel_singletons(b.to_numpy())
    ari = float(adjusted_rand_score(la, lb)) if len(shared) > 1 else float("nan")

    cont = pd.crosstab(la, lb).to_numpy()
    tp = _pair_count(cont.ravel())
    pairs_a = _pair_count(cont.sum(axis=1))
    pairs_b = _pair_count(cont.sum(axis=0))
    precision = tp / pairs_a if pairs_a else float("nan")
    recall = tp / pairs_b if pairs_b else float("nan")
    return {
        "n_shared": int(len(shared)),
        "ari": ari,
        "pairwise_precision": float(precision),
        "pairwise_recall": float(recall),
    }


def melt_cellregistry(wide, uid_col="UID"):
    """UnitMatch wide CellRegistry (UID + per-session-date columns of ks ids) -> long.

    Cells may be empty/NaN/0 (absent) or ';'-joined (merged) ks ids. Output columns:
    [session, ks_unit_id, um_uid].
    """
    session_cols = [c for c in wide.columns if c != uid_col]
    rows = []
    for _, r in wide.iterrows():
        uid = int(r[uid_col])
        for sess in session_cols:
            cell = r[sess]
            if pd.isna(cell):
                continue
            text = str(cell).strip()
            if text in ("", "0", "0.0", "nan"):
                continue
            for part in text.split(";"):
                part = part.strip()
                if not part or part in ("0", "0.0"):
                    continue
                rows.append({"session": str(sess), "ks_unit_id": int(float(part)), "um_uid": uid})
    return pd.DataFrame(rows, columns=["session", "ks_unit_id", "um_uid"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe" -m pytest tests/tracking_dant/test_registry.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/tracking_dant/registry.py tests/tracking_dant/test_registry.py
git commit -m "feat(dant): registry conversion + UnitMatch comparison metrics" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `build_dant_inputs.py` — assemble the DANT input folder

**Files:**
- Create: `scripts/tracking_dant/build_dant_inputs.py`

**Interfaces:**
- Consumes: `adapter.collapse_cv/derive_channel_shanks/seconds_to_ms/is_positive_going`; `visdetect.core.load_session`; `visdetect.analysis.config.parse_session_date`.
- Produces (in `--out-dir`, default `data/cache/dant/BG_046/input/`): `waveform_all.npy (n_unit,383,82)`, `session_index.npy (n_unit,)`, `channel_locations.npy (383,2)`, `channel_shanks.npy (383,)`, `spike_times/Unit{k}.npy` (ms), `unit_lookup.csv [pooled_index, session, ks_unit_id, session_index]`, `build_log.txt`.

- [ ] **Step 1: Write the script**

Create `scripts/tracking_dant/build_dant_inputs.py`:
```python
"""Assemble DANT's multi-shank input folder from visdetect-extracted BG_046 data.

Reads per-session RawWaveforms + pkl spike trains, pools all good units across the 42
sessions, and writes DANT's expected .npy layout. Spike times are converted to ms;
positive-going units are excluded (DANT trough-centering assumes negative spikes).

Run with ANALYSIS_PY from the worktree root.
"""
import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapter  # noqa: E402

from visdetect.core.session import load_session  # noqa: E402
from visdetect.analysis.config import parse_session_date  # noqa: E402

PRIMARY = "E:/python_analysis/git_repos/vis_detect_analysis_Sep2025"
DEFAULT_UM_INPUT = os.path.join(PRIMARY, "data", "unit_match", "input", "BG_046")
DEFAULT_PKL_DIR = os.path.join(PRIMARY, "data", "pkls", "BG_046")
DEFAULT_OUT = "data/cache/dant/BG_046/input"


def _ks_ids_for_session(session_dir):
    """ks unit ids present as RawWaveforms in a session input dir, sorted."""
    rw = os.path.join(session_dir, "RawWaveforms")
    ids = []
    for fn in os.listdir(rw):
        if fn.startswith("Unit") and fn.endswith("_RawSpikes.npy"):
            ids.append(int(fn[len("Unit"):-len("_RawSpikes.npy")]))
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--um-input", default=DEFAULT_UM_INPUT)
    ap.add_argument("--pkl-dir", default=DEFAULT_PKL_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--subject", default="BG_046")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "spike_times"), exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    sessions = [d for d in os.listdir(args.um_input)
                if os.path.isdir(os.path.join(args.um_input, d, "RawWaveforms"))]
    sessions = sorted(sessions, key=parse_session_date)
    log(f"{len(sessions)} sessions found; chronological order established.")

    waveforms = []
    session_index = []
    lookup_rows = []
    ref_channel_pos = None
    pooled = 0
    n_excluded_positive = 0
    n_missing_spikes = 0

    for s_idx, sdir in enumerate(sessions, start=1):
        spath = os.path.join(args.um_input, sdir)
        chan_pos = np.load(os.path.join(spath, "channel_positions.npy"))
        if ref_channel_pos is None:
            ref_channel_pos = chan_pos
        elif not np.array_equal(chan_pos, ref_channel_pos):
            raise ValueError(f"channel_positions for session {sdir} differ from session 1 "
                             f"({chan_pos.shape} vs {ref_channel_pos.shape}); pooled geometry ambiguous.")

        pkl_path = os.path.join(args.pkl_dir, f"{args.subject}_{sdir}.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"missing pkl for session {sdir}: {pkl_path}")
        sess = load_session(pkl_path)
        spike_map = {int(c.cluster_id): np.asarray(c.spike_times) for c in sess.clusters}

        ks_ids = _ks_ids_for_session(spath)
        n_sess_units = 0
        for ks in ks_ids:
            raw = np.load(os.path.join(spath, "RawWaveforms", f"Unit{ks}_RawSpikes.npy"))
            wave = adapter.collapse_cv(raw)            # (383, 82)
            if adapter.is_positive_going(wave):
                n_excluded_positive += 1
                continue
            if ks not in spike_map:
                n_missing_spikes += 1
                log(f"  [skip] session {sdir} ks {ks}: no spike train in pkl")
                continue
            st_ms = adapter.seconds_to_ms(spike_map[ks])
            np.save(os.path.join(args.out_dir, "spike_times", f"Unit{pooled}.npy"), st_ms)
            waveforms.append(wave)
            session_index.append(s_idx)
            lookup_rows.append({"pooled_index": pooled, "session": sdir,
                                "ks_unit_id": ks, "session_index": s_idx})
            pooled += 1
            n_sess_units += 1

        log(f"  session {sdir} (idx {s_idx}): {n_sess_units} units")
        del sess
        gc.collect()

    waveform_all = np.stack(waveforms, axis=0)          # (n_unit, 383, 82)
    session_index = np.asarray(session_index, dtype=np.int64)
    channel_shanks = adapter.derive_channel_shanks(ref_channel_pos)

    # DANT requires contiguous 1..n_session
    uniq = np.unique(session_index)
    assert uniq.min() == 1 and len(uniq) == uniq.max(), f"session_index not contiguous: {uniq}"

    np.save(os.path.join(args.out_dir, "waveform_all.npy"), waveform_all)
    np.save(os.path.join(args.out_dir, "session_index.npy"), session_index)
    np.save(os.path.join(args.out_dir, "channel_locations.npy"), ref_channel_pos.astype(np.float64))
    np.save(os.path.join(args.out_dir, "channel_shanks.npy"), channel_shanks)
    pd.DataFrame(lookup_rows).to_csv(os.path.join(args.out_dir, "unit_lookup.csv"), index=False)

    log(f"DONE: {pooled} pooled units, {len(uniq)} sessions, waveform_all {waveform_all.shape}, "
        f"{int((channel_shanks.max()+1))} shanks.")
    log(f"Excluded positive-going: {n_excluded_positive}; missing spikes: {n_missing_spikes}.")
    with open(os.path.join(args.out_dir, "build_log.txt"), "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on all 42 sessions**

Run (from worktree root):
```bash
"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe" scripts/tracking_dant/build_dant_inputs.py
```
Expected: prints per-session unit counts and a final `DONE: <N> pooled units, 42 sessions, waveform_all (<N>, 383, 82), 4 shanks.`

- [ ] **Step 3: Verify the outputs**

Run:
```bash
"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe" - <<'PY'
import numpy as np, pandas as pd, os, glob
d = "data/cache/dant/BG_046/input"
w = np.load(f"{d}/waveform_all.npy"); si = np.load(f"{d}/session_index.npy")
cl = np.load(f"{d}/channel_locations.npy"); cs = np.load(f"{d}/channel_shanks.npy")
lk = pd.read_csv(f"{d}/unit_lookup.csv")
assert w.ndim == 3 and w.shape[1:] == (383, 82), w.shape
assert w.shape[0] == len(si) == len(lk), (w.shape[0], len(si), len(lk))
assert set(np.unique(si)) == set(range(1, 43)), np.unique(si)
assert cl.shape == (383, 2) and cs.shape == (383,)
assert sorted(np.unique(cs)) == [0, 1, 2, 3], np.unique(cs)
st = np.load(sorted(glob.glob(f"{d}/spike_times/Unit*.npy"))[0])
assert st.max() > 1000, f"spike times look like seconds not ms: max {st.max()}"
print("OK", w.shape, "units; ms-range max ISI source", round(float(st.max()),1))
PY
```
Expected: `OK (<N>, 383, 82) units; ...` with no AssertionError.

- [ ] **Step 4: Commit**

```bash
git add scripts/tracking_dant/build_dant_inputs.py
git commit -m "feat(dant): build DANT multi-shank input folder from extracted BG_046 data" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: DANT settings + run (pilot, then full)

**Files:**
- Create: `scripts/tracking_dant/settings_bg046.json`
- Create: `scripts/tracking_dant/run_dant_bg046.py`

**Interfaces:**
- Consumes: the input folder from Task 4.
- Produces: `FIGURES/tracking_dant/BG_046/dant_output/` containing `Output.npz`, `IdxCluster.npy`, similarity matrices, `motion_*.npy`, `Figures/`.

- [ ] **Step 1: Write the DANT settings**

Create `scripts/tracking_dant/settings_bg046.json`:
```json
{
    "path_to_data": "data/cache/dant/BG_046/input",
    "output_folder": "FIGURES/tracking_dant/BG_046/dant_output",
    "save_intermediate_results": false,
    "n_jobs": -1,
    "centering_waveforms": true,
    "spikeLocation": {
        "location_algorithm": "monopolar_triangulation",
        "n_nearest_channels": 20
    },
    "waveformCorrection": {
        "n_nearest_channels": 38,
        "linear_correction": false,
        "n_templates": 2
    },
    "autoCorr": { "window": 300, "binwidth": 1, "gaussian_sigma": 5 },
    "ISI": { "window": 100, "binwidth": 1, "gaussian_sigma": 1 },
    "motionEstimation": {
        "features": [ ["AutoCorr"], ["Waveform", "AutoCorr"] ],
        "max_iter": 15,
        "repeat_last_feature_set": true,
        "stop_early": true
    },
    "clustering": {
        "max_distance": 100,
        "features": ["Waveform", "AutoCorr"],
        "n_iter": 10,
        "weight_tol": 1e-8
    },
    "autoCuration": { "auto_split": true }
}
```

- [ ] **Step 2: Write the run wrapper**

Create `scripts/tracking_dant/run_dant_bg046.py`:
```python
"""Run DANT (multi-shank) on the BG_046 input folder. Use DANT_PY from the worktree root."""
import argparse
import os

import numpy as np
import hjson

np.random.seed(42)  # DANT does not seed its motion init / bootstrap

from pyDANT import runDANTMultiShank  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default=os.path.join(os.path.dirname(__file__), "settings_bg046.json"))
    ap.add_argument("--path-to-data", default=None, help="override settings path_to_data")
    ap.add_argument("--output-folder", default=None, help="override settings output_folder")
    args = ap.parse_args()

    with open(args.settings) as f:
        user_settings = hjson.load(f)
    if args.path_to_data:
        user_settings["path_to_data"] = args.path_to_data
    if args.output_folder:
        user_settings["output_folder"] = args.output_folder
    os.makedirs(user_settings["output_folder"], exist_ok=True)
    print(f"Running DANT multi-shank: {user_settings['path_to_data']} -> {user_settings['output_folder']}")
    runDANTMultiShank(user_settings)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Pilot run on a 3-session subset**

Build a 3-session pilot input and run DANT on it to verify the end-to-end wiring before the full run. Run (from worktree root):
```bash
# Build a 3-session pilot input folder by symlink-free copy of the first 3 sessions' worth of pooled units:
"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe" - <<'PY'
import numpy as np, pandas as pd, os, shutil
src="data/cache/dant/BG_046/input"; dst="data/cache/dant/BG_046/input_pilot"
os.makedirs(os.path.join(dst,"spike_times"),exist_ok=True)
si=np.load(f"{src}/session_index.npy"); lk=pd.read_csv(f"{src}/unit_lookup.csv")
keep=np.where(si<=3)[0]
w=np.load(f"{src}/waveform_all.npy")[keep]
np.save(f"{dst}/waveform_all.npy",w)
np.save(f"{dst}/session_index.npy",si[keep])
shutil.copy(f"{src}/channel_locations.npy",dst); shutil.copy(f"{src}/channel_shanks.npy",dst)
for new,old in enumerate(keep):
    shutil.copy(f"{src}/spike_times/Unit{old}.npy", f"{dst}/spike_times/Unit{new}.npy")
lk.iloc[keep].reset_index(drop=True).assign(pooled_index=range(len(keep))).to_csv(f"{dst}/unit_lookup.csv",index=False)
print("pilot units",len(keep))
PY
.venv_dant/Scripts/python.exe scripts/tracking_dant/run_dant_bg046.py \
  --path-to-data data/cache/dant/BG_046/input_pilot \
  --output-folder FIGURES/tracking_dant/BG_046/dant_output_pilot
```
Expected: DANT prints motion-estimation iterations and `DANT done!`; `FIGURES/tracking_dant/BG_046/dant_output_pilot/Output.npz` and `IdxCluster.npy` exist. If it errors, fix the input/settings before the full run.

- [ ] **Step 4: Full run on all 42 sessions**

Run (from worktree root):
```bash
.venv_dant/Scripts/python.exe scripts/tracking_dant/run_dant_bg046.py
```
Expected: completes (paper reports <1 h for ~10k units; ours is smaller); `FIGURES/tracking_dant/BG_046/dant_output/Output.npz`, `IdxCluster.npy`, and `Figures/MatchedProbability.png` exist.

- [ ] **Step 5: Verify the output loads**

Run:
```bash
.venv_dant/Scripts/python.exe - <<'PY'
import numpy as np
o="FIGURES/tracking_dant/BG_046/dant_output"
idx=np.load(f"{o}/IdxCluster.npy")
print("units",len(idx),"tracked clusters",int(idx.max()),"untracked",int((idx==-1).sum()))
PY
```
Expected: prints unit count, a positive number of tracked clusters, and an untracked count.

- [ ] **Step 6: Commit**

```bash
git add scripts/tracking_dant/settings_bg046.json scripts/tracking_dant/run_dant_bg046.py
git commit -m "feat(dant): DANT settings (Waveform+ACG, no PETH, multishank) + run wrapper" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: `dant_to_registry.py` — normalize output to a long registry

**Files:**
- Create: `scripts/tracking_dant/dant_to_registry.py`

**Interfaces:**
- Consumes: `IdxCluster.npy` + `unit_lookup.csv`; `registry.idxcluster_to_registry`.
- Produces: `data/cache/dant/BG_046/dant_registry.csv` `[session, ks_unit_id, dant_uid]`.

- [ ] **Step 1: Write the script**

Create `scripts/tracking_dant/dant_to_registry.py`:
```python
"""Convert DANT IdxCluster.npy + unit_lookup.csv into a long registry CSV."""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dant-output", default="FIGURES/tracking_dant/BG_046/dant_output")
    ap.add_argument("--input-dir", default="data/cache/dant/BG_046/input")
    ap.add_argument("--out", default="data/cache/dant/BG_046/dant_registry.csv")
    args = ap.parse_args()

    idx = np.load(os.path.join(args.dant_output, "IdxCluster.npy"))
    lookup = pd.read_csv(os.path.join(args.input_dir, "unit_lookup.csv"))
    lookup["session"] = lookup["session"].astype(str)
    reg = registry.idxcluster_to_registry(idx, lookup)
    reg.to_csv(args.out, index=False)
    n_tracked = (reg["dant_uid"] > 0).sum()
    print(f"wrote {args.out}: {len(reg)} units, {reg['dant_uid'].nunique()} clusters, {n_tracked} tracked units")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run (from worktree root):
```bash
"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe" scripts/tracking_dant/dant_to_registry.py
```
Expected: `wrote data/cache/dant/BG_046/dant_registry.csv: <N> units, <K> clusters, <M> tracked units`.

- [ ] **Step 3: Commit**

```bash
git add scripts/tracking_dant/dant_to_registry.py
git commit -m "feat(dant): convert DANT output to long dant_registry.csv" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: `evaluate_dant.py` — comparison, held-out ISI AUC, figures

**Files:**
- Create: `scripts/tracking_dant/evaluate_dant.py`

**Interfaces:**
- Consumes: `dant_registry.csv`; the local UnitMatch CellRegistry; the built `spike_times/` + `unit_lookup.csv`; `registry.*`.
- Produces (in `FIGURES/tracking_dant/BG_046/`): `survival_comparison.png`, `summary_stats.json`, `isi_auc.png`, and copies/refs to DANT's own diagnostic figures.

- [ ] **Step 1: Write the script**

Create `scripts/tracking_dant/evaluate_dant.py`:
```python
"""Evaluate DANT tracking on BG_046: yield + survival vs UnitMatch, co-membership agreement,
and a held-out ISI-fingerprint AUC. Saves presentation-ready figures + a summary JSON.

Run with ANALYSIS_PY from the worktree root.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry  # noqa: E402

PRIMARY = "E:/python_analysis/git_repos/vis_detect_analysis_Sep2025"
DEFAULT_UM = os.path.join(PRIMARY, "data", "unit_match", "output", "BG_046_um329_CellRegistry.csv")
FIGDIR = "FIGURES/tracking_dant/BG_046"


def _isi_hist(spike_ms, window=100, binwidth=1, sigma=1):
    isi = np.diff(np.sort(spike_ms))
    h = np.histogram(isi, bins=np.arange(0, window + binwidth, binwidth))[0].astype(float)
    s = h.sum()
    if s > 0:
        h /= s
    return gaussian_filter1d(h, sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dant-registry", default="data/cache/dant/BG_046/dant_registry.csv")
    ap.add_argument("--input-dir", default="data/cache/dant/BG_046/input")
    ap.add_argument("--um-registry", default=DEFAULT_UM)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(FIGDIR, exist_ok=True)

    dant = pd.read_csv(args.dant_registry)
    dant["session"] = dant["session"].astype(str)
    lookup = pd.read_csv(os.path.join(args.input_dir, "unit_lookup.csv"))
    lookup["session"] = lookup["session"].astype(str)
    n_sessions = int(lookup["session_index"].max())

    summary = {"n_units": int(len(dant)),
               "dant_n_clusters": int(dant.loc[dant["dant_uid"] > 0, "dant_uid"].nunique()),
               "dant_n_tracked_units": int((dant["dant_uid"] > 0).sum())}

    # --- DANT tracked-length survival ---
    dant_len = registry.tracked_lengths(dant)
    ks, dant_surv = registry.survival_function(dant_len, n_sessions)
    summary["dant_mean_tracked_len"] = float(dant_len.mean()) if len(dant_len) else 0.0

    # --- UnitMatch comparison (best-effort; skip cleanly if registry absent) ---
    um_surv = None
    if os.path.exists(args.um_registry):
        um_wide = pd.read_csv(args.um_registry)
        uid_col = "UID" if "UID" in um_wide.columns else um_wide.columns[0]
        um_long = registry.melt_cellregistry(um_wide, uid_col=uid_col)
        um_long["session"] = um_long["session"].astype(str)
        um_len = registry.tracked_lengths(um_long, uid_col="um_uid")
        _, um_surv = registry.survival_function(um_len, n_sessions)
        summary["um_mean_tracked_len"] = float(um_len.mean()) if len(um_len) else 0.0
        summary["um_n_tracked_units"] = int((um_long["um_uid"] > 0).sum())
        agree = registry.comembership_agreement(dant, um_long, "dant_uid", "um_uid")
        summary["comembership_vs_unitmatch"] = agree
    else:
        summary["um_note"] = f"UnitMatch registry not found at {args.um_registry}; comparison skipped."

    # --- Survival comparison figure ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(ks, dant_surv, "-o", ms=3, label=f"DANT (mean {summary['dant_mean_tracked_len']:.1f})")
    if um_surv is not None:
        ax.plot(ks, um_surv, "-s", ms=3, label=f"UnitMatch (mean {summary.get('um_mean_tracked_len', float('nan')):.1f})")
    ax.set_xlabel("Tracked length (# sessions)")
    ax.set_ylabel("Fraction of tracked neurons ≥ k sessions")
    ax.set_title("BG_046 cross-session tracking: survival")
    ax.legend(); ax.set_ylim(0, 1); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "survival_comparison.png"), dpi=200)
    plt.close(fig)

    # --- Held-out ISI-fingerprint AUC ---
    def isi_for(pooled_index):
        st = np.load(os.path.join(args.input_dir, "spike_times", f"Unit{int(pooled_index)}.npy"))
        return _isi_hist(st)

    key_to_pooled = {(r.session, int(r.ks_unit_id)): int(r.pooled_index) for r in lookup.itertuples()}
    tracked = dant[dant["dant_uid"] > 0].copy()
    matched_sims, nonmatched_sims = [], []
    # matched: cross-session pairs within the same dant_uid
    for uid, grp in tracked.groupby("dant_uid"):
        members = [(row.session, int(row.ks_unit_id)) for row in grp.itertuples()]
        if len(members) < 2:
            continue
        hists = {m: isi_for(key_to_pooled[m]) for m in members if m in key_to_pooled}
        ms = list(hists)
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                if ms[i][0] != ms[j][0]:  # different session
                    r = np.corrcoef(hists[ms[i]], hists[ms[j]])[0, 1]
                    if np.isfinite(r):
                        matched_sims.append(r)
    # non-matched: within-session pairs of different units (random sample, balanced)
    by_session = tracked.groupby("session")
    target = len(matched_sims)
    attempts = 0
    while len(nonmatched_sims) < target and attempts < target * 50:
        attempts += 1
        sess = rng.choice(tracked["session"].unique())
        g = by_session.get_group(sess)
        if len(g) < 2:
            continue
        rows = list(g.sample(2, random_state=int(rng.integers(1 << 30))).itertuples(index=False))
        ka = (rows[0].session, int(rows[0].ks_unit_id))
        kb = (rows[1].session, int(rows[1].ks_unit_id))
        if ka not in key_to_pooled or kb not in key_to_pooled:
            continue
        r = np.corrcoef(isi_for(key_to_pooled[ka]), isi_for(key_to_pooled[kb]))[0, 1]
        if np.isfinite(r):
            nonmatched_sims.append(r)

    if matched_sims and nonmatched_sims:
        y = np.r_[np.ones(len(matched_sims)), np.zeros(len(nonmatched_sims))]
        score = np.r_[matched_sims, nonmatched_sims]
        auc = float(roc_auc_score(y, score))
        summary["heldout_isi_auc"] = auc
        summary["n_matched_pairs"] = len(matched_sims)
        summary["n_nonmatched_pairs"] = len(nonmatched_sims)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        bins = np.linspace(-1, 1, 41)
        ax.hist(nonmatched_sims, bins=bins, density=True, alpha=0.6, label="within-session, different unit")
        ax.hist(matched_sims, bins=bins, density=True, alpha=0.6, label="cross-session, same DANT id")
        ax.set_xlabel("ISI-histogram correlation"); ax.set_ylabel("density")
        ax.set_title(f"Held-out ISI fingerprint (AUC = {auc:.3f})")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, "isi_auc.png"), dpi=200)
        plt.close(fig)
    else:
        summary["heldout_isi_auc"] = None

    with open(os.path.join(FIGDIR, "summary_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run (from worktree root):
```bash
"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe" scripts/tracking_dant/evaluate_dant.py
```
Expected: prints a JSON summary including `dant_mean_tracked_len`, `comembership_vs_unitmatch` (ari/precision/recall), and `heldout_isi_auc`; writes `survival_comparison.png`, `isi_auc.png`, `summary_stats.json` to `FIGURES/tracking_dant/BG_046/`.

- [ ] **Step 3: Sanity-check the AUC and agreement**

Inspect `FIGURES/tracking_dant/BG_046/summary_stats.json`. Expected sane ranges: `heldout_isi_auc` clearly > 0.5 (well-separated, ideally > 0.8); `comembership_vs_unitmatch.ari` between 0 and 1. If AUC ≈ 0.5, investigate (likely a pooled-index/lookup misalignment) before trusting results.

- [ ] **Step 4: Commit**

```bash
git add scripts/tracking_dant/evaluate_dant.py
git commit -m "feat(dant): evaluation - survival vs UnitMatch, co-membership ARI, held-out ISI AUC" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: README, example-track figure, and wrap-up

**Files:**
- Create: `scripts/tracking_dant/README.md`
- Modify: (none)

**Interfaces:**
- Consumes: all prior tasks.
- Produces: `scripts/tracking_dant/README.md`; `FIGURES/tracking_dant/BG_046/example_tracks.png`.

- [ ] **Step 1: Write an example-tracks figure helper and run it**

Create the figure inline (run from worktree root):
```bash
"E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe" - <<'PY'
import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
inp="data/cache/dant/BG_046/input"; figdir="FIGURES/tracking_dant/BG_046"
reg=pd.read_csv("data/cache/dant/BG_046/dant_registry.csv"); reg["session"]=reg["session"].astype(str)
lk=pd.read_csv(f"{inp}/unit_lookup.csv"); lk["session"]=lk["session"].astype(str)
wav=np.load(f"{inp}/waveform_all.npy")  # (n,383,82)
key={(r.session,int(r.ks_unit_id)):int(r.pooled_index) for r in lk.itertuples()}
lengths=reg[reg.dant_uid>0].groupby("dant_uid")["session"].nunique().sort_values(ascending=False)
top=lengths.head(6).index.tolist()
fig,axes=plt.subplots(1,len(top),figsize=(3*len(top),3),sharey=True)
for ax,uid in zip(np.atleast_1d(axes),top):
    g=reg[reg.dant_uid==uid]
    for row in g.itertuples():
        pi=key.get((row.session,int(row.ks_unit_id)))
        if pi is None: continue
        w=wav[pi]; pk=int(np.argmax(w.max(1)-w.min(1)))
        ax.plot(w[pk],lw=0.6,alpha=0.7)
    ax.set_title(f"id {uid} ({len(g)} sess)"); ax.set_xlabel("sample")
axes[0].set_ylabel("uV (peak ch)")
fig.suptitle("DANT example tracks: peak-channel waveform across sessions")
fig.tight_layout(); fig.savefig(f"{figdir}/example_tracks.png",dpi=200)
print("wrote",f"{figdir}/example_tracks.png","for uids",top)
PY
```
Expected: writes `example_tracks.png` overlaying peak-channel waveforms across sessions for the 6 longest tracks (they should look consistent within each panel).

- [ ] **Step 2: Write the README**

Create `scripts/tracking_dant/README.md`:
```markdown
# DANT cross-session tracking on BG_046

Runs DANT (pyDANT, density-based across-day neuron tracking) on BG_046's 42 extracted
sessions; Waveform + ACG identity (no PETH); multi-shank. See the design spec at
`docs/superpowers/specs/2026-06-23-dant-tracking-bg046-design.md`.

## Environments
- Adapter / registry / eval: the analysis venv (`...vis_detect_analysis_Sep2025/.venv`) — has `visdetect`.
- DANT run only: `./.venv_dant` (`pip install pyDANT`).

## Pipeline (run from this worktree root)
1. `<ANALYSIS_PY> scripts/tracking_dant/build_dant_inputs.py`  -> `data/cache/dant/BG_046/input/`
2. `.venv_dant/Scripts/python.exe scripts/tracking_dant/run_dant_bg046.py` -> `FIGURES/tracking_dant/BG_046/dant_output/`
3. `<ANALYSIS_PY> scripts/tracking_dant/dant_to_registry.py` -> `data/cache/dant/BG_046/dant_registry.csv`
4. `<ANALYSIS_PY> scripts/tracking_dant/evaluate_dant.py`    -> figures + `summary_stats.json`

`<ANALYSIS_PY>` = `E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe`.

## Notes
- Spike times are converted to ms; positive-going units excluded (DANT trough-centering).
- Inputs are read from the primary repo (no junctions). Nothing is written outside this worktree.
- Reproducibility: `np.random.seed(42)` in the run wrapper.
```

- [ ] **Step 3: Commit**

```bash
git add scripts/tracking_dant/README.md
git commit -m "docs(dant): README + example-tracks figure for BG_046 run" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 4: Final report**

Summarize for the user: pooled unit count, # DANT clusters, mean tracked length (DANT vs UnitMatch), co-membership ARI, held-out ISI AUC, and any units dropped (positive-going / missing spikes). Point to the figures in `FIGURES/tracking_dant/BG_046/`. Note follow-ups (PETH-for-motion, multi-subject, curation-tier mapping) remain out of scope.

---

## Notes for the implementer

- **TDD is real for Tasks 2–3** (pure helpers). Tasks 4–7 are integration scripts verified by running them with explicit expected output — treat the verification steps as the test.
- **If `build_dant_inputs` asserts on geometry**, a session's `channel_positions` differs (bank change). Report it; do not silently coerce.
- **If DANT errors on `n_nearest_channels`**, a shank has < 38 channels — check `channel_shanks` derivation.
- **If the ISI AUC ≈ 0.5**, suspect a pooled-index↔lookup misalignment in Task 4 (the order of `waveform_all`, `session_index`, `spike_times/Unit{k}`, and `unit_lookup.pooled_index` must all match).
- **Never run compute over `X:`**; all paths are local/primary-repo.
- **Before any `git worktree remove`** of this worktree, confirm no junctions exist (we created none by design).
