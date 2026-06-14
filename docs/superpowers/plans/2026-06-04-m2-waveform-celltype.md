# M2 — FSI/SPN Waveform Cell-Type Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce trustworthy FSI/SPN cell-type labels from the **current** RawWaveforms, and make the spine's `celltype` column actually populate (it is currently structurally always-NaN). This fills P0's `celltype` contract column for the Learning + TF headlines.

**Architecture:** A pure feature/classification library module (`waveform_celltype.py`: trough-to-peak + half-width features; 2-component GMM on T2P → FSI/SPN with a ΔBIC bimodality check), a producer CLI that loops QC sessions (reusing `tracking_qc.load_raw_mean_waveform` + `extract_peak_channel` for I/O — DRY), writing `waveform_celltype_labels.csv` to a non-stale path; plus a **bug fix** in `build_unit_table`'s waveform merge and a `WAVEFORM_LABELS_PATH` repoint. Independent of M1 (different files), so it can run in a parallel worktree.

**Tech Stack:** Python 3.10, numpy, pandas, scikit-learn (`sklearn.mixture.GaussianMixture`), pytest. `py` on Windows. Library under `src/visdetect/analysis/`; producer under `scripts/analysis/`; tests under `tests/analysis/` and `tests/suite/`.

**Branch:** a dedicated M2 worktree off `docs/presentation-prep-roadmap` via `superpowers:using-git-worktrees`; set `PYTHONPATH=<worktree>/src`. M2 touches a new module + the config + the waveform merge in `loader.py`; it does NOT touch M1's GLT/registry code.

**Context the engineer must know (read first):**
- **Source waveforms:** `data/unit_match/input/BG_046/{DDMMYYYY}/RawWaveforms/Unit{ks_id}_RawSpikes.npy`, shape ≈ (82 samples, 383 channels, 2 split-halves) @ 30 kHz. This is the **current** UnitMatch input — NOT `scripts/pipelines/concat_sort/` (retired) and NOT Kilosort templates. Mean across the 2 split-halves, take the peak channel by peak-to-peak amplitude, then the 1-D peak-channel waveform is what we featurize.
- **Reuse (DRY):** `visdetect.analysis.tracking_qc.load_raw_mean_waveform(raw_wf_root, session_name, ks_unit_id)` → mean waveform `(n_samples, n_channels)` or None; `tracking_qc.extract_peak_channel(mean_waveform)` → peak channel index. `RAW_WF_DIR` is in `visdetect.analysis.config` (= `ROOT/data/unit_match/input/BG_046`).
- **Reference implementation (port, don't reinvent):** `AI_exploration/analysis_3_waveform_celltype.py` — feature formulas (T2P, half-width) and the GMM(2)-on-T2P classification with threshold = mean of the two GMM means; ΔBIC (BIC₁ − BIC₂) as the bimodality statistic. SR = 30000 Hz; T2P fit window (0.02, 1.5) ms.
- **THE BUG to fix:** `build_unit_table` (`src/visdetect/suite/loader.py` ~line 379-395) reads `wf[["session_date","cluster_id","celltype"]]` guarded by `if "session_date" in wf.columns`, but `load_waveform_labels` **renames** `session_date→session_name` and `celltype→cell_type`. So the guard is always False → `glt["celltype"] = np.nan` **always**. The merge must use the loader's normalized columns and `fillna("unknown")`.
- **P0 contract:** `celltype` is a contract column (default "unknown"); not value-checked in `ALLOWED_VALUES`, so {FSI, SPN, Unclassified, unknown} are all fine. `build_unit_table(validate=True)` enforces key uniqueness + the contract.
- **Label CSV schema (this plan defines it):** columns `session_date` (int DDMMYYYY), `cluster_id` (int), `celltype` (str: FSI/SPN/Unclassified). After `load_waveform_labels` normalization these become `session_name`(int), `cluster_id`, `cell_type`.
- **Spec:** `docs/superpowers/specs/2026-06-03-presentation-prep-roadmap-design.md` (§3 contract, §9 groundwork).

---

## File structure

| Path | Action | Responsibility |
|---|---|---|
| `src/visdetect/analysis/waveform_celltype.py` | Create | `compute_waveform_features` (t2p_ms, half_width_ms, pt_ratio); `classify_celltype` (GMM(2) on T2P → FSI/SPN/Unclassified + ΔBIC info). Pure, no I/O. |
| `tests/analysis/test_waveform_celltype.py` | Create | Synthetic feature + classification tests. |
| `src/visdetect/analysis/config.py` | (already done) | `WAVEFORM_LABELS_PATH` is subject-scoped by `feature/subject-scope-outputs`; M2 does NOT edit it. |
| `src/visdetect/suite/loader.py` | Modify | Fix the waveform merge so `celltype` actually populates (normalized columns + `fillna("unknown")`). |
| `tests/suite/test_unit_table_build.py` | Modify (append) | Test `celltype` is populated from a synthetic label set. |
| `scripts/analysis/build_waveform_celltype_labels.py` | Create | Producer CLI: loop QC sessions, extract features, one global GMM, write labels + stats CSVs. |
| `docs/superpowers/specs/2026-06-03-presentation-prep-roadmap-design.md` | Modify | Record M2 outcome + the always-NaN merge bug fix. |

---

## Task 1: Waveform feature extraction

**Files:**
- Create: `src/visdetect/analysis/waveform_celltype.py`
- Test: `tests/analysis/test_waveform_celltype.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/analysis/test_waveform_celltype.py`:

```python
"""Tests for FSI/SPN waveform cell-type features + classification (M2)."""
import numpy as np
import pytest

from visdetect.analysis.waveform_celltype import (
    compute_waveform_features, classify_celltype, SR_HZ,
)


def _synthetic_spike(trough_lo=28, trough_hi=33, peak_idx=40, n=82):
    """Broad trough (depth -1) over [trough_lo, trough_hi), positive peak after."""
    w = np.zeros(n, dtype=float)
    w[trough_lo:trough_hi] = -1.0
    w[peak_idx] = 0.5
    return w


def test_features_t2p_and_halfwidth_known_values():
    w = _synthetic_spike(28, 33, 40)
    f = compute_waveform_features(w)
    # trough argmin = 28; peak after = 40 -> t2p = 12 samples
    assert f["t2p_ms"] == pytest.approx((40 - 28) / SR_HZ * 1000, rel=1e-6)
    # w < -0.5 at indices 28..32 -> half width = 4 samples
    assert f["half_width_ms"] == pytest.approx((32 - 28) / SR_HZ * 1000, rel=1e-6)
    assert f["pt_ratio"] == pytest.approx(0.5, rel=1e-6)


def test_features_short_input_returns_nans():
    f = compute_waveform_features(np.array([0.0, -1.0, 0.5]))
    assert np.isnan(f["t2p_ms"])


def test_features_flat_input_safe():
    f = compute_waveform_features(np.zeros(82))
    assert set(f) == {"t2p_ms", "half_width_ms", "pt_ratio"}
```

- [ ] **Step 2: Run to confirm failure**

Run: `py -m pytest tests/analysis/test_waveform_celltype.py -v`
Expected: `ModuleNotFoundError: No module named 'visdetect.analysis.waveform_celltype'`.

- [ ] **Step 3: Create the module (features only)**

Create `src/visdetect/analysis/waveform_celltype.py`:

```python
"""FSI/SPN cell-type from extracellular waveform shape (M2).

Features and the 2-component-GMM-on-T2P classification are ported from
AI_exploration/analysis_3_waveform_celltype.py. Pure (no I/O): the producer
script wires these to RawWaveforms via visdetect.analysis.tracking_qc.

See docs/superpowers/specs/2026-06-03-presentation-prep-roadmap-design.md (§9).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

SR_HZ: float = 30000.0          # Neuropixels sample rate
T2P_MIN_MS: float = 0.02        # GMM fit window lower bound
T2P_MAX_MS: float = 1.5         # GMM fit window upper bound

_NAN_FEATURES = {"t2p_ms": np.nan, "half_width_ms": np.nan, "pt_ratio": np.nan}


def compute_waveform_features(peak_waveform: np.ndarray) -> Dict[str, float]:
    """Trough-to-peak, half-width, and peak/trough ratio from a 1-D peak-channel waveform.

    Returns NaN features for degenerate inputs (too short / flat).
    """
    w = np.asarray(peak_waveform, dtype=float)
    if w.size < 10:
        return dict(_NAN_FEATURES)
    denom = np.abs(w).max()
    if denom < 1e-12:
        return dict(_NAN_FEATURES)
    w_norm = w / (denom + 1e-12)

    trough_idx = int(np.argmin(w_norm))
    after = w_norm[trough_idx:]
    if after.size < 2:
        return dict(_NAN_FEATURES)
    peak_after_idx = trough_idx + int(np.argmax(after))
    t2p_ms = (peak_after_idx - trough_idx) / SR_HZ * 1000.0

    half_min = w_norm[trough_idx] / 2.0
    below_half = np.where(w_norm < half_min)[0]
    hw_ms = ((below_half[-1] - below_half[0]) / SR_HZ * 1000.0
             if below_half.size >= 2 else np.nan)

    pt_ratio = float(w_norm[peak_after_idx] / (-w_norm[trough_idx] + 1e-12))
    return {"t2p_ms": float(t2p_ms), "half_width_ms": float(hw_ms), "pt_ratio": pt_ratio}
```

- [ ] **Step 4: Run tests, expect pass**

Run: `py -m pytest tests/analysis/test_waveform_celltype.py -v`
Expected: 3 passed. (Create empty `tests/analysis/__init__.py` if needed.)

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/waveform_celltype.py tests/analysis/test_waveform_celltype.py
git commit -m "M2: waveform feature extraction (t2p, half-width, pt-ratio)"
```

---

## Task 2: GMM cell-type classification

**Files:**
- Modify: `src/visdetect/analysis/waveform_celltype.py`
- Modify: `tests/analysis/test_waveform_celltype.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/analysis/test_waveform_celltype.py`:

```python
def test_classify_bimodal_splits_fsi_spn():
    rng = np.random.default_rng(0)
    narrow = rng.normal(0.20, 0.02, 60)      # FSI-like short T2P
    broad = rng.normal(0.65, 0.05, 60)       # SPN-like long T2P
    t2p = np.concatenate([narrow, broad])
    labels, info = classify_celltype(t2p)
    assert set(np.unique(labels)) <= {"FSI", "SPN", "Unclassified"}
    # threshold falls between the two modes; counts roughly balanced
    assert 0.20 < info["threshold_ms"] < 0.65
    assert (labels == "FSI").sum() == pytest.approx(60, abs=8)
    assert (labels == "SPN").sum() == pytest.approx(60, abs=8)
    assert info["delta_bic"] > 0            # 2 comps beat 1 on bimodal data


def test_classify_nan_is_unclassified():
    labels, _ = classify_celltype(np.array([0.2, np.nan, 0.7]))
    assert labels[1] == "Unclassified"


def test_classify_labels_align_with_input_length():
    t2p = np.array([0.2, 0.65, np.nan, 0.25, 0.6])
    labels, info = classify_celltype(t2p)
    assert labels.shape == t2p.shape
    assert info["n"] >= 1
```

- [ ] **Step 2: Run to confirm failure**

Run: `py -m pytest tests/analysis/test_waveform_celltype.py -v`
Expected: the 3 new tests fail (`classify_celltype` undefined).

- [ ] **Step 3: Implement `classify_celltype`**

Append to `src/visdetect/analysis/waveform_celltype.py`:

```python
def classify_celltype(
    t2p_ms: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Classify units FSI/SPN from trough-to-peak via a 2-component GMM.

    A single global GMM is fit on T2P values within (T2P_MIN_MS, T2P_MAX_MS).
    The decision threshold is the mean of the two component means; units with
    T2P below it are FSI (narrow), at/above it SPN (broad). NaN T2P → Unclassified.

    Returns
    -------
    labels : ndarray of str, same shape as t2p_ms (values in {FSI, SPN, Unclassified}).
    info : dict with threshold_ms, narrow_mean_ms, broad_mean_ms, delta_bic, n.
    """
    from sklearn.mixture import GaussianMixture

    arr = np.asarray(t2p_ms, dtype=float)
    finite = np.isfinite(arr)
    in_window = finite & (arr > T2P_MIN_MS) & (arr < T2P_MAX_MS)
    X = arr[in_window].reshape(-1, 1)
    if X.shape[0] < 2:
        labels = np.full(arr.shape, "Unclassified", dtype=object)
        return labels, {"threshold_ms": np.nan, "narrow_mean_ms": np.nan,
                        "broad_mean_ms": np.nan, "delta_bic": np.nan, "n": int(X.shape[0])}

    gmm2 = GaussianMixture(n_components=2, random_state=random_state).fit(X)
    gmm1 = GaussianMixture(n_components=1, random_state=random_state).fit(X)
    means = np.sort(gmm2.means_.flatten())
    threshold = float(means.mean())

    labels = np.full(arr.shape, "Unclassified", dtype=object)
    labels[finite & (arr < threshold)] = "FSI"
    labels[finite & (arr >= threshold)] = "SPN"

    info = {
        "threshold_ms": threshold,
        "narrow_mean_ms": float(means[0]),
        "broad_mean_ms": float(means[1]),
        "delta_bic": float(gmm1.bic(X) - gmm2.bic(X)),
        "n": int(X.shape[0]),
    }
    return labels, info
```

- [ ] **Step 4: Run tests, expect pass**

Run: `py -m pytest tests/analysis/test_waveform_celltype.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/waveform_celltype.py tests/analysis/test_waveform_celltype.py
git commit -m "M2: GMM(2) T2P cell-type classification (FSI/SPN) with delta-BIC"
```

---

## Task 3: Repoint label path + fix the always-NaN celltype merge

**Files:**
- Modify: `src/visdetect/analysis/config.py` (`WAVEFORM_LABELS_PATH`)
- Modify: `src/visdetect/suite/loader.py` (`build_unit_table` waveform merge)
- Modify: `tests/suite/test_unit_table_build.py` (append)

- [ ] **Step 1: Append the failing test**

Append to `tests/suite/test_unit_table_build.py`:

```python
def test_build_unit_table_populates_celltype(tmp_path, monkeypatch):
    from visdetect.suite import loader as L

    glt = pd.DataFrame({
        "Session_Date": [1072025, 1072025],
        "Cluster_ID": [3, 4],
        "Global_UID": [10, 11],
        "stage": ["Learning", "Learning"],
        "session_idx": [0, 0],
    })
    monkeypatch.setattr(L, "load_glt", lambda qc_only=True: glt.copy())
    monkeypatch.setattr(L, "load_all_lick_responsiveness", lambda: pd.DataFrame())
    # load_waveform_labels returns the NORMALIZED columns (session_name/cell_type):
    wf = pd.DataFrame({"session_name": [1072025], "cluster_id": [3], "cell_type": ["FSI"]})
    monkeypatch.setattr(L, "load_waveform_labels", lambda path=None: wf.copy())
    monkeypatch.setattr(L, "load_tf_responsiveness_detrended", lambda: pd.DataFrame())
    monkeypatch.setattr(L, "load_tf_classification_detrended", lambda: pd.DataFrame())

    df = L.build_unit_table(qc_only=True)
    by_key = df.set_index(["Session_Date", "Cluster_ID"])["celltype"]
    assert by_key[(1072025, 3)] == "FSI"          # matched label
    assert by_key[(1072025, 4)] == "unknown"      # no label -> filled, not NaN
```

- [ ] **Step 2: Run to confirm failure**

Run: `py -m pytest tests/suite/test_unit_table_build.py::test_build_unit_table_populates_celltype -v`
Expected: FAIL — `celltype` is NaN for cluster 3 (current guard checks `session_date`, which the loader renamed away).

- [ ] **Step 3: Fix the waveform merge in `build_unit_table`**

In `src/visdetect/suite/loader.py`, replace the waveform-merge block (the `# Merge waveform cell-type labels` `try/except`, currently ~lines 379-395) with:

```python
    # Merge waveform cell-type labels.
    # load_waveform_labels normalizes to session_name/cell_type, so accept either
    # naming and always populate `celltype` (fillna so unlabeled units are "unknown").
    try:
        wf = load_waveform_labels()
        sess_col = next((c for c in ("session_name", "session_date") if c in wf.columns), None)
        type_col = next((c for c in ("cell_type", "celltype") if c in wf.columns), None)
        if sess_col and "cluster_id" in wf.columns and type_col:
            wf_sub = wf[[sess_col, "cluster_id", type_col]].copy()
            wf_sub.columns = ["_wf_session", "_wf_cluster", "celltype"]
            wf_sub["_wf_session"] = wf_sub["_wf_session"].astype(int)
            wf_sub["_wf_cluster"] = wf_sub["_wf_cluster"].astype(int)
            glt = glt.merge(
                wf_sub,
                left_on=["Session_Date", "Cluster_ID"],
                right_on=["_wf_session", "_wf_cluster"],
                how="left",
            )
            glt.drop(columns=["_wf_session", "_wf_cluster"], errors="ignore", inplace=True)
            glt["celltype"] = glt["celltype"].fillna("unknown")
        else:
            glt["celltype"] = "unknown"
    except FileNotFoundError:
        glt["celltype"] = "unknown"
```

Note: this changes the no-label default from `np.nan` to the contract default `"unknown"`, which `add_label_defaults` would otherwise have to supply (but cannot, because the column already exists after the merge).

- [ ] **Step 4: Run tests, expect pass**

Run: `py -m pytest tests/suite/test_unit_table_build.py -v`
Expected: all passed (P0 tests + the new celltype test).

- [ ] **Step 5: Confirm `WAVEFORM_LABELS_PATH` is already subject-scoped (NO edit needed)**

The subject-scoping change (branch `feature/subject-scope-outputs`, merged to main) already
repointed this in `src/visdetect/analysis/config.py` to a per-subject, non-stale path:

```python
WAVEFORM_LABELS_PATH   = os.path.join(ROOT, "data", SUBJECT, "waveform_celltype_labels.csv")
```

Do **not** edit it. Just verify it resolves under `data/<SUBJECT>/` (defaults to `data/BG_046/`):

```bash
py -c "import sys; sys.path.insert(0,'src'); from visdetect.analysis.config import WAVEFORM_LABELS_PATH; print(WAVEFORM_LABELS_PATH)"
```
Expected: `…/data/BG_046/waveform_celltype_labels.csv`. The Task 4 producer already writes to
`WAVEFORM_LABELS_PATH` (imported from config), so its output lands in the subject folder automatically.

- [ ] **Step 6: Run the suite again (regression)**

Run: `py -m pytest tests/suite/ -v`
Expected: all passed; real-data celltype test (if any) skips cleanly when the CSV is absent (the producer hasn't run yet).

- [ ] **Step 7: Commit**

```bash
git add src/visdetect/suite/loader.py tests/suite/test_unit_table_build.py
git commit -m "M2: fix always-NaN celltype merge in build_unit_table"
```
(`WAVEFORM_LABELS_PATH` is already subject-scoped on main — M2 does not modify `config.py`.)

---

## Task 4: Producer CLI (RawWaveforms → labels)

**Files:**
- Create: `scripts/analysis/build_waveform_celltype_labels.py`

Loops QC sessions, extracts the peak-channel waveform per unit (reusing `tracking_qc`),
fits ONE global GMM, and writes the labels + stats CSVs.

- [ ] **Step 1: Create the producer**

Create `scripts/analysis/build_waveform_celltype_labels.py`:

```python
"""Build FSI/SPN waveform cell-type labels from current RawWaveforms.

One global GMM(2) over trough-to-peak across all QC-session units.

Usage:
    py scripts/analysis/build_waveform_celltype_labels.py
"""
import glob
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from visdetect.analysis.config import RAW_WF_DIR, WAVEFORM_LABELS_PATH  # noqa: E402
from visdetect.suite.loader import load_staging_manifest                # noqa: E402
from visdetect.analysis.tracking_qc import (                            # noqa: E402
    load_raw_mean_waveform, extract_peak_channel,
)
from visdetect.analysis.waveform_celltype import (                      # noqa: E402
    compute_waveform_features, classify_celltype,
)

STATS_PATH = os.path.join(REPO_ROOT, "FIGURES", "qc", "waveform_celltype_stats.csv")


def session_unit_ids(session_str: str):
    """Kilosort ids with a RawWaveforms file for this session."""
    rw_dir = os.path.join(RAW_WF_DIR, session_str, "RawWaveforms")
    ids = []
    for f in glob.glob(os.path.join(rw_dir, "Unit*_RawSpikes.npy")):
        name = os.path.basename(f)
        try:
            ids.append(int(name.replace("Unit", "").replace("_RawSpikes.npy", "")))
        except ValueError:
            continue
    return sorted(ids)


def main():
    manifest = load_staging_manifest(qc_only=True)
    rows = []
    for sess_int in sorted(manifest["session_name"].astype(int)):
        sess_str = str(sess_int).zfill(8)
        ids = session_unit_ids(sess_str)
        if not ids:
            print(f"  {sess_str}: no RawWaveforms"); continue
        n = 0
        for kid in ids:
            mean_wf = load_raw_mean_waveform(RAW_WF_DIR, sess_str, kid)
            if mean_wf is None:
                continue
            peak_chan = extract_peak_channel(mean_wf)
            feats = compute_waveform_features(mean_wf[:, peak_chan])
            rows.append({"session_date": sess_int, "cluster_id": int(kid),
                         "t2p_ms": feats["t2p_ms"], "half_width_ms": feats["half_width_ms"]})
            n += 1
        print(f"  {sess_str}: {n} units")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No waveforms extracted — check RAW_WF_DIR.")

    labels, info = classify_celltype(df["t2p_ms"].values)
    df["celltype"] = labels
    print(f"GMM: threshold={info['threshold_ms']:.3f} ms, delta_BIC={info['delta_bic']:.1f}, "
          f"n={info['n']}; counts={df['celltype'].value_counts().to_dict()}")

    os.makedirs(os.path.dirname(WAVEFORM_LABELS_PATH), exist_ok=True)
    df[["session_date", "cluster_id", "celltype"]].to_csv(WAVEFORM_LABELS_PATH, index=False)
    print(f"Wrote labels: {WAVEFORM_LABELS_PATH}  ({len(df)} units)")

    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    pd.DataFrame([info]).to_csv(STATS_PATH, index=False)
    print(f"Wrote stats: {STATS_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-check the I/O reuse against one real session**

```bash
py -c "import sys; sys.path.insert(0,'src'); from visdetect.analysis.config import RAW_WF_DIR; from visdetect.analysis.tracking_qc import load_raw_mean_waveform, extract_peak_channel; from visdetect.analysis.waveform_celltype import compute_waveform_features; w=load_raw_mean_waveform(RAW_WF_DIR,'23062025',3); print('mean wf shape', None if w is None else w.shape); pc=extract_peak_channel(w); print('peak chan', pc, 'features', compute_waveform_features(w[:,pc]))"
```

Expected: prints a `(n_samples, n_channels)` shape, a peak channel index, and finite `t2p_ms`/`half_width_ms`. (If `load_raw_mean_waveform` returns None, confirm the session/id exists under `RAW_WF_DIR`.)

- [ ] **Step 3: Commit**

```bash
git add scripts/analysis/build_waveform_celltype_labels.py
git commit -m "M2: producer CLI for FSI/SPN waveform labels (one global GMM over QC sessions)"
```

---

## Task 5: Generate labels + confirm celltype populates (real-data run)

**Files:** none (a run + verification step).

- [ ] **Step 1: Run the producer**

```bash
py scripts/analysis/build_waveform_celltype_labels.py
```

Expected: per-session unit counts, then a GMM summary (threshold ~0.3-0.5 ms, **delta_BIC > 0** confirming bimodality), and writes `data/waveform_celltype_labels.csv` + `FIGURES/qc/waveform_celltype_stats.csv`. If delta_BIC ≤ 0, the T2P distribution is not bimodal for this dataset — stop and report (do not ship a forced 2-class split); revisit whether half-width or a different feature separates better.

- [ ] **Step 2: Confirm the spine now carries celltype (only if the GLT exists)**

If M1 has regenerated the GLT (`table_output/Grand_Longitudinal_Table.csv` present):

```bash
py -c "import sys; sys.path.insert(0,'src'); from visdetect.suite.loader import build_unit_table; df=build_unit_table(qc_only=True); print(df['celltype'].value_counts(dropna=False))"
```

Expected: a real FSI/SPN/unknown breakdown (not all-NaN/all-unknown). If the GLT is absent (M1 not yet run), skip this step — the label CSV is still produced and will be picked up once the GLT exists.

- [ ] **Step 3: Commit the labels + stats**

```bash
git add data/waveform_celltype_labels.csv FIGURES/qc/waveform_celltype_stats.csv
git commit -m "M2: generate FSI/SPN waveform labels from current RawWaveforms"
```

(The labels CSV is small — committing it is fine and makes the spine reproducible without re-running the producer.)

---

## Task 6: Record outcome in the spec

**Files:**
- Modify: `docs/superpowers/specs/2026-06-03-presentation-prep-roadmap-design.md`

- [ ] **Step 1: Append the M2 outcome to §9**

Append under §9:

```markdown
> **M2 done (date):** FSI/SPN labels regenerated from current RawWaveforms
> (`scripts/analysis/build_waveform_celltype_labels.py`, one global GMM(2) on T2P,
> delta_BIC=<paste>). Fixed the always-NaN `celltype` merge in `build_unit_table`
> (loader normalizes to session_name/cell_type; the merge now accepts that and
> fills "unknown"). `WAVEFORM_LABELS_PATH` is subject-scoped at `data/<SUBJECT>/waveform_celltype_labels.csv`.
> celltype counts: <paste>.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-03-presentation-prep-roadmap-design.md
git commit -m "M2: record waveform cell-type outcome in roadmap spec"
```

---

## Definition of done

- `waveform_celltype.py` exists, fully unit-tested (features + GMM classification).
- `build_unit_table` populates `celltype` (the always-NaN merge bug is fixed); contract validates.
- `WAVEFORM_LABELS_PATH` points at the regenerated `data/<SUBJECT>/waveform_celltype_labels.csv`.
- The producer writes labels + a stats CSV with a **positive delta_BIC** (bimodality confirmed).
- With the GLT present, `build_unit_table()["celltype"]` shows a real FSI/SPN/unknown breakdown.

**Unblocks:** cell-type-resolved Learning (FSI vs SPN tracked across stages) and the TF/evidence
figure split by cell type — both join straight onto the same `(Session_Date, Cluster_ID)` rows.
