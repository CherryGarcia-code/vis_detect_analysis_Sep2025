# B9 — State-matched baseline TF-encoding across learning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether TF-responsive striatal units in BG_039 encode the baseline grating more faithfully late-vs-early in learning **at matched behavioral state (StimSens)**, by re-running the registry's own TF-GLM on state×stage trial subsets and comparing `c1_r`.

**Architecture:** Thin **glue + validation** around existing, unchanged library code. The registry estimator (`visdetect.analysis.tf_glm` + `tf_glm_data.session_trial_regressors`, byte-identical on `main` except one non-scientific guard fixed in Task 0) is reused verbatim; B9's only change is **which trials are fed in** (filtered by state label + stage). A new library module `state_tf_learning.py` holds the loaders/joins, the pinned config, and the state-conditioned encoding wrapper; three scripts run Phase 0 (feasibility + free preliminary) and Phase 1 (the state-conditioned re-score + stats/figure). A **faithfulness test** (whole-session re-run reproduces the registry's `c1_r_log2`) guarantees the reuse is exact before any state split is trusted.

**Tech Stack:** Python 3.10 (`.venv`, invoke via `py`), numpy, pandas, scikit-learn (`PoissonRegressor`), statsmodels (mixed-effects), matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-07-01-B9-state-matched-baseline-tf-encoding-learning-design.md`

## Global Constraints

- **Reuse, do NOT re-implement the estimator.** `visdetect.analysis.tf_glm` and `visdetect.analysis.tf_glm_data` are used as-is. The **only** estimator edit is Task 0: porting the worktree's non-finite crash-guard into `tf_glm.py::trial_bin_edges` (scientifically inert — affects only trials with a non-finite timestamp, e.g. `_v2` split sessions; convergent with `feature/tf-glm-bg046` where it already exists). No other edits to those files.
- **Pinned config (`b9_cfg()`), matching the registry's `_cfg("log2")` exactly:** `TFGLMConfig(include_movement=False, include_phase=False, include_tiled_baseline=True, standardize_design=True, fast_fit=True, responsive_criterion="c2", tf_encoding="log2", min_pulses_per_label=20)`; all other fields at dataclass defaults (`bin_s=0.05`, `n_folds=10`, `seed=42`, `c1_r_thresh=0.2`, `c2_p_thresh=0.01`, `lambdas=(1e-3..100)`). **Never change these.**
- **`MIN_SPIKES = 500`** per unit on the trial subset (`y.sum() < 500 → skip`), matching the registry.
- **Canonical session ids on BOTH sides of every join** via `visdetect.analysis.config.canonical_session_id` (handles leading-zero day, 6-digit DDMMYY `270325`, `_v2` suffixes).
- **Readout metric = `c1_r`** (the registry's `c1_r_log2`), returned by `identify_tf_responsive_pulse(...)["c1_r"]`.
- **Responsive/non-responsive split = registry `resp_log2`** (a label we *read*; B9 never re-classifies pass/fail).
- **No cross-session cell pooling** — compute per session, aggregate `c1_r` across sessions within a stage as independent per-unit rows.
- **Paths:** lib `src/visdetect/analysis/state_tf_learning.py`; scripts `scripts/state_tf_learning/`; cache `data/cache/state_tf_learning/`; figures `FIGURES/state_tf_learning/BG_039/`; tests `tests/analysis/test_state_tf_learning.py`.
- **Windows:** invoke Python as `py`; run tests `py -m pytest`. Memory: `del sess; gc.collect()` after each session. **No compute over `X:`** (BG_039 pkls are local under `data/pkls/BG_039/`).
- **Registry file:** `data/cache/tf_responsive/bg039_tf_responsive.csv` (cols `subject,session,session_date,unit,resp_log2,c1_r_log2,c2_p_log2,kernel_peak_t,kernel_fwhm,n_spikes,resp_lin,c1_r_lin`).
- **State tags:** `data/cache/state_tags/BG_039/<8digit>.csv` (cols incl. `trial_idx,state_label,state_confidence`; `state_label ∈ {Impulsive,StimSens,Disengaged,Abort}`; `state_confidence` gate = 0.8).
- **Staging manifest:** `data/BG_039_staging_manifest.csv` (has `session_name`, `stage`).
- **All subagents Opus 4.8** (`claude-opus-4-8`).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/visdetect/analysis/tf_glm.py` | (Task 0 only) port the non-finite `trial_bin_edges` guard. |
| `src/visdetect/analysis/state_tf_learning.py` | Core lib: loaders + canonical joins, `b9_cfg()`, `session_stage_map()`, `state_trial_indices()`, `state_conditioned_encoding()`, `preliminary_learning_trend()`, coverage helpers. |
| `scripts/state_tf_learning/b9_phase0_profile.py` | Runner: free registry preliminary + coverage landscape → `usable` table + figures. |
| `scripts/state_tf_learning/b9_phase1_run.py` | Runner: state-conditioned `c1_r` per usable session×state×stage → joined cache table. |
| `scripts/state_tf_learning/b9_phase1_figure.py` | Runner: mixed-effects stats + headline figure. |
| `scripts/state_tf_learning/README.md` | How to run Phase 0 → Phase 1. |
| `tests/analysis/test_state_tf_learning.py` | Unit + integration tests (incl. the faithfulness gate, marked `slow`). |

---

### Task 0: Port the `trial_bin_edges` non-finite guard into main

**Files:**
- Modify: `src/visdetect/analysis/tf_glm.py` (function `trial_bin_edges`, ~line 93)
- Test: `tests/analysis/test_state_tf_learning.py`

**Interfaces:** no signature change — `trial_bin_edges(t_start, t_end, bin_s) -> np.ndarray` now returns an empty array when either endpoint is non-finite (instead of raising).

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_state_tf_learning.py
import numpy as np, pandas as pd, pytest

def test_trial_bin_edges_guards_nonfinite():
    from visdetect.analysis.tf_glm import trial_bin_edges
    assert trial_bin_edges(float("nan"), 0.0, 0.05).size == 0   # would crash pre-guard
    assert trial_bin_edges(1.0, float("inf"), 0.05).size == 0
    e = trial_bin_edges(1.0, 1.2, 0.05)                          # finite: unchanged
    assert e.size == 4 and np.isclose(e[0], 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_state_tf_learning.py::test_trial_bin_edges_guards_nonfinite -v`
Expected: FAIL/ERROR (`ValueError: cannot convert float NaN to integer`).

- [ ] **Step 3: Add the guard (exact worktree version)**

In `src/visdetect/analysis/tf_glm.py`, replace the body of `trial_bin_edges`:

```python
def trial_bin_edges(t_start: float, t_end: float, bin_s: float) -> np.ndarray:
    """Left edges of 50-ms bins spanning [t_start, t_end).

    Returns an empty array if either endpoint is non-finite. A trial with no
    neural-clock timestamp (e.g. behavioural trials beyond the recorded
    Baseline_ON/ephys coverage, as in split ``_b`` re-recordings where the
    behaviour log outruns the NI events) then contributes zero bins to the
    design instead of raising ``cannot convert float NaN to integer`` and
    killing the whole session.
    """
    if not (np.isfinite(t_start) and np.isfinite(t_end)):
        return np.zeros(0, dtype=float)
    n = int(np.floor((t_end - t_start) / bin_s + 1e-9))
    return t_start + np.arange(max(n, 0)) * bin_s
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_state_tf_learning.py::test_trial_bin_edges_guards_nonfinite -v`
Expected: PASS. Also run the existing tf_glm suite to confirm no regression: `py -m pytest tests/analysis -k tf_glm -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/tf_glm.py tests/analysis/test_state_tf_learning.py
git commit -m "fix(tf-glm): guard trial_bin_edges against non-finite endpoints (port from feature/tf-glm-bg046; scientifically inert)"
```

---

### Task 1: Loaders + canonical joins

**Files:**
- Create: `src/visdetect/analysis/state_tf_learning.py`
- Test: `tests/analysis/test_state_tf_learning.py`

**Interfaces:**
- Produces: `load_registry(path=None) -> pd.DataFrame` (adds `sess_key`); `load_state_tags(subject, session_date, states_dir=None) -> pd.DataFrame` (adds `sess_key`).

- [ ] **Step 1: Write the failing test**

```python
from visdetect.analysis import state_tf_learning as stl
from visdetect.analysis.config import canonical_session_id

def test_registry_join_handles_id_footguns(tmp_path):
    reg = pd.DataFrame({
        "subject": ["BG_039"] * 3,
        "session": ["BG_039_270325", "BG_039_01042025", "BG_039_25042025_v2"],
        "session_date": ["270325", "01042025", "25042025_v2"],
        "unit": [10, 11, 12], "resp_log2": [True, False, True],
        "c1_r_log2": [0.5, 0.05, 0.3], "c2_p_log2": [0.001, 0.5, 0.002],
    })
    p = tmp_path / "reg.csv"; reg.to_csv(p, index=False)
    out = stl.load_registry(p)
    assert "sess_key" in out.columns
    assert out["sess_key"].tolist() == [canonical_session_id(s) for s in reg["session_date"]]
    assert out["sess_key"].nunique() == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_state_tf_learning.py::test_registry_join_handles_id_footguns -v`
Expected: FAIL (`AttributeError: load_registry`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/visdetect/analysis/state_tf_learning.py
"""B9 — state-matched baseline TF-encoding across learning.

Thin glue around the registry TF-GLM (visdetect.analysis.tf_glm +
tf_glm_data.session_trial_regressors, reused unchanged). B9's only change vs the
registry run is WHICH trials are fed in (filtered by state label + stage).
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

from visdetect.analysis.config import canonical_session_id
from visdetect.analysis.tf_glm import (
    TFGLMConfig, assemble_design, count_vector, make_trial_folds,
    fit_poisson_cv, identify_tf_responsive_pulse, pulse_times_from_tf,
)
from visdetect.analysis.tf_glm_data import session_trial_regressors

_REPO = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = _REPO / "data" / "cache" / "tf_responsive" / "bg039_tf_responsive.csv"
DEFAULT_STATES_DIR = _REPO / "data" / "cache" / "state_tags"
DEFAULT_MANIFEST = _REPO / "data" / "BG_039_staging_manifest.csv"
PKL_ROOT = _REPO / "data" / "pkls"
STATE_CONF_THRESH = 0.8


def load_registry(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or DEFAULT_REGISTRY)
    df["sess_key"] = df["session_date"].map(canonical_session_id)
    return df


def load_state_tags(subject: str, session_date: str, states_dir=None) -> pd.DataFrame:
    key = canonical_session_id(session_date)
    fp = Path(states_dir or DEFAULT_STATES_DIR) / subject / f"{key}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"No state tags for {subject}/{key}: {fp}")
    df = pd.read_csv(fp)
    df["sess_key"] = key
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_state_tf_learning.py::test_registry_join_handles_id_footguns -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/state_tf_learning.py tests/analysis/test_state_tf_learning.py
git commit -m "feat(B9): registry + state-tag loaders with canonical joins"
```

---

### Task 2: `b9_cfg()` + stage map + state trial indices

**Files:**
- Modify: `src/visdetect/analysis/state_tf_learning.py`
- Test: `tests/analysis/test_state_tf_learning.py`

**Interfaces:**
- Produces: `b9_cfg() -> TFGLMConfig`; `session_stage_map(manifest_path=None) -> dict[str,str]`; `state_trial_indices(tags, state, conf_thresh=STATE_CONF_THRESH) -> list[int]`.

- [ ] **Step 1: Write the failing test**

```python
def test_b9_cfg_matches_registry_overrides():
    cfg = stl.b9_cfg()
    assert (cfg.bin_s, cfg.n_folds, cfg.seed) == (0.05, 10, 42)
    assert cfg.include_tiled_baseline and cfg.standardize_design and cfg.fast_fit
    assert cfg.tf_encoding == "log2" and cfg.min_pulses_per_label == 20
    assert not cfg.include_movement and not cfg.include_phase

def test_state_trial_indices_gates_on_confidence():
    tags = pd.DataFrame({
        "trial_idx": [0, 1, 2, 3, 4],
        "state_label": ["StimSens", "StimSens", "Disengaged", "StimSens", "Impulsive"],
        "state_confidence": [0.95, 0.5, 0.99, 0.85, 0.99],
    })
    assert stl.state_trial_indices(tags, "StimSens", conf_thresh=0.8) == [0, 3]
```

- [ ] **Step 2: Run to verify it fails**

Run: `py -m pytest tests/analysis/test_state_tf_learning.py -k "b9_cfg or confidence" -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
def b9_cfg() -> TFGLMConfig:
    """The registry's exact _cfg('log2') — DO NOT change these values."""
    return TFGLMConfig(
        include_movement=False, include_phase=False,
        include_tiled_baseline=True, standardize_design=True,
        fast_fit=True, responsive_criterion="c2",
        tf_encoding="log2", min_pulses_per_label=20,
    )


def session_stage_map(manifest_path=None) -> dict:
    m = pd.read_csv(manifest_path or DEFAULT_MANIFEST)
    return {canonical_session_id(s): str(stg)
            for s, stg in zip(m["session_name"], m["stage"])}


def state_trial_indices(tags: pd.DataFrame, state: str,
                        conf_thresh: float = STATE_CONF_THRESH) -> List[int]:
    keep = (tags["state_label"] == state) & (tags["state_confidence"] >= conf_thresh)
    return sorted(int(i) for i in tags.loc[keep, "trial_idx"].tolist())
```

- [ ] **Step 4: Run to verify it passes**

Run: `py -m pytest tests/analysis/test_state_tf_learning.py -k "b9_cfg or confidence" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/state_tf_learning.py tests/analysis/test_state_tf_learning.py
git commit -m "feat(B9): pinned b9_cfg + stage map + confidence-gated state indices"
```

---

### Task 3: Core `state_conditioned_encoding()` (the reuse wrapper)

**Files:**
- Modify: `src/visdetect/analysis/state_tf_learning.py`
- Test: `tests/analysis/test_state_tf_learning.py`

**Interfaces:**
- Consumes: `session_trial_regressors`, `assemble_design`, `count_vector`, `make_trial_folds`, `fit_poisson_cv`, `identify_tf_responsive_pulse` (unchanged).
- Produces: `state_conditioned_encoding(session, subset_idx, cfg, min_spikes=500) -> pd.DataFrame` with columns `unit, c1_r, c2_p, is_responsive_rerun, r_red_mean, n_folds_used, kernel_peak_t, kernel_fwhm, n_spikes, n_trials`. Empty DataFrame (same columns) if `subset_idx` < `n_folds` or the design is degenerate.

- [ ] **Step 1: Write the failing test** (adds a synthetic TF-session helper)

```python
def _make_tf_session(n_trials=60, n_units=3, baseline_s=7.0, dt=0.05, seed=0):
    """Minimal Session-like object with the fields the estimator needs.
    Unit 100 fires proportional to baseline TF; units 101+ are flat.
    """
    import types
    rng = np.random.default_rng(seed)
    period = baseline_s + 3.0
    bon = np.arange(n_trials) * period + 5.0
    con = bon + baseline_s
    n_bv = int(baseline_s / dt) * 3
    trials, bv_all = [], []
    for i in range(n_trials):
        bv = 2.0 ** rng.normal(0.0, 0.25, size=n_bv)
        bv_all.append(bv)
        trials.append(types.SimpleNamespace(
            trialoutcome="Hit" if i % 2 == 0 else "Miss",
            change_size=2.0, change_time=float(baseline_s),
            baseline_values=bv, reactiontimes={"RT": 0.4, "Miss": 2.0},
            orientation=0.0, ITI=3.0, n_seen=5))
    clusters = []
    for u in range(n_units):
        uid = 100 + u; spikes = []
        for i in range(n_trials):
            bv = bv_all[i]; t = bon[i] + np.arange(bv.size) * (dt / 3.0)
            rate = 5.0 + (8.0 * bv if u == 0 else 0.0)
            k = rng.poisson(rate * (dt / 3.0))
            for tt, kk in zip(t, k):
                if kk:
                    spikes.extend(tt + rng.random(kk) * (dt / 3.0))
            spikes.extend(bon[i] - 1.0 + rng.random(30) * 0.5)
        clusters.append(types.SimpleNamespace(cluster_id=uid, spike_times=np.sort(np.array(spikes))))
    ni = {"Baseline_ON": bon, "Change_ON": con, "Lick_L": np.array([]),
          "Valve_L": np.full(n_trials, np.nan), "Rot_enc_A": np.sort(rng.random(500) * bon[-1])}
    return types.SimpleNamespace(trials=trials, clusters=clusters, ni_events=ni,
                                 good_and_stable_ids=[c.cluster_id for c in clusters],
                                 good_cluster_ids=[c.cluster_id for c in clusters])

def test_state_conditioned_encoding_returns_per_unit_rows():
    sess = _make_tf_session(seed=1); cfg = stl.b9_cfg()
    df = stl.state_conditioned_encoding(sess, subset_idx=list(range(60)), cfg=cfg)
    assert {"unit", "c1_r", "is_responsive_rerun", "n_trials"}.issubset(df.columns)
    assert df["n_trials"].iloc[0] == 60
    c = df.set_index("unit")["c1_r"]
    assert c.get(100, np.nan) > c.get(101, -1)   # TF-driven unit scores higher
```

- [ ] **Step 2: Run to verify it fails**

Run: `py -m pytest tests/analysis/test_state_tf_learning.py::test_state_conditioned_encoding_returns_per_unit_rows -v`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Write minimal implementation** (mirrors the registry `run_task` wiring; the ONLY B9 change is the `subset` filter)

```python
_ENC_COLS = ["unit", "c1_r", "c2_p", "is_responsive_rerun", "r_red_mean",
             "n_folds_used", "kernel_peak_t", "kernel_fwhm", "n_spikes", "n_trials"]


def state_conditioned_encoding(session, subset_idx: List[int], cfg: TFGLMConfig,
                               min_spikes: int = 500) -> pd.DataFrame:
    trials_regs, units = session_trial_regressors(session, cfg)
    subset_idx = [i for i in subset_idx if 0 <= i < len(trials_regs)]
    if len(subset_idx) < cfg.n_folds:
        return pd.DataFrame(columns=_ENC_COLS)
    sub = [trials_regs[i] for i in subset_idx]
    design = assemble_design(sub, cfg)
    if design.bin_edges.size == 0:
        return pd.DataFrame(columns=_ENC_COLS)
    fold_ids = make_trial_folds(design.trial_index, cfg.n_folds, cfg.seed)
    tf_cols = design.col_groups["tf"]
    rows = []
    for uid, spikes in units.items():
        y = count_vector(sub, spikes, design)
        ns = float(y.sum())
        if ns < min_spikes:
            continue
        full = fit_poisson_cv(design.X, y, cfg, fold_ids)
        Xr = design.X.copy(); Xr[:, tf_cols] = 0.0
        red = fit_poisson_cv(Xr, y, cfg, fold_ids)
        r = identify_tf_responsive_pulse(design, y, full, red, cfg)
        rows.append({"unit": int(uid), "c1_r": r["c1_r"], "c2_p": r["c2_p"],
                     "is_responsive_rerun": r["is_responsive"], "r_red_mean": r["r_red_mean"],
                     "n_folds_used": r["n_folds_used"], "kernel_peak_t": r["kernel_peak_t"],
                     "kernel_fwhm": r["kernel_fwhm"], "n_spikes": ns, "n_trials": len(subset_idx)})
    return pd.DataFrame(rows, columns=_ENC_COLS)
```

- [ ] **Step 4: Run to verify it passes**

Run: `py -m pytest tests/analysis/test_state_tf_learning.py::test_state_conditioned_encoding_returns_per_unit_rows -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/state_tf_learning.py tests/analysis/test_state_tf_learning.py
git commit -m "feat(B9): state_conditioned_encoding — reuse the registry GLM on a trial subset"
```

---

### Task 4: Faithfulness gate — whole-session re-run reproduces the registry `c1_r_log2`

**Files:**
- Modify: `tests/analysis/test_state_tf_learning.py`

The **make-or-break** validation: on the whole session (no state filter) B9's re-run reproduces the registry's `c1_r_log2` for the units present. If this fails, the reuse wiring is wrong. **Gate for Tasks 6–8.**

- [ ] **Step 1: Write the failing test** (marked `slow` — loads a real session + fits GLMs)

```python
@pytest.mark.slow
def test_reproduces_registry_c1r_on_whole_session():
    from visdetect.core.session import load_session
    reg = stl.load_registry()
    key = "16052025"
    sess = load_session(str(stl.PKL_ROOT / "BG_039" / f"BG_039_{key}.pkl"))
    df = stl.state_conditioned_encoding(sess, list(range(len(sess.trials))), stl.b9_cfg())
    got = df.set_index("unit")["c1_r"]
    want = reg[reg.sess_key == stl.canonical_session_id(key)].set_index("unit")["c1_r_log2"]
    common = [u for u in want.index if u in got.index]
    assert len(common) >= 20
    diff = np.abs(got.loc[common].to_numpy() - want.loc[common].to_numpy())
    assert np.nanmedian(diff) < 0.02, f"median |Δc1_r| = {np.nanmedian(diff):.4f}"
    assert np.nanpercentile(diff, 90) < 0.05
```

- [ ] **Step 2: Run to surface any real gap**

Run: `py -m pytest tests/analysis/test_state_tf_learning.py::test_reproduces_registry_c1r_on_whole_session -v`
Expected: PASS if the wiring matches the registry. If it FAILS, the diff is real — investigate trial inclusion (the registry uses ALL trials; `session_trial_regressors` NaN's `change_time` on FA/abort automatically), unit-id dtype, or fold seeding, and reconcile until the tolerance passes.

- [ ] **Step 3: (only if Step 2 failed) reconcile the wiring** — no new production code unless a real mismatch is found.

- [ ] **Step 4: Confirm pass**

Run: same command → PASS (median |Δ| < 0.02).

- [ ] **Step 5: Commit**

```bash
git add tests/analysis/test_state_tf_learning.py
git commit -m "test(B9): faithfulness gate — whole-session re-run reproduces registry c1_r_log2"
```

---

### Task 5: Free registry-only learning preliminary (Phase 0a)

**Files:**
- Modify: `src/visdetect/analysis/state_tf_learning.py`
- Create: `scripts/state_tf_learning/b9_phase0_profile.py`
- Test: `tests/analysis/test_state_tf_learning.py`

**Interfaces:**
- Produces: `preliminary_learning_trend(reg, stage_map) -> pd.DataFrame` (per `sess_key`: `stage, n_resp, mean_c1r_resp, mean_c1r_nonresp`).

- [ ] **Step 1: Write the failing test**

```python
def test_preliminary_learning_trend_groups_by_stage():
    reg = pd.DataFrame({
        "sess_key": ["01042025", "01042025", "16062025", "16062025"],
        "unit": [1, 2, 3, 4], "resp_log2": [True, False, True, False],
        "c1_r_log2": [0.30, 0.02, 0.55, 0.03],
    })
    out = stl.preliminary_learning_trend(reg, {"01042025": "Learning", "16062025": "Expert"})
    row = out.set_index("sess_key")
    assert row.loc["16062025", "mean_c1r_resp"] == pytest.approx(0.55)
    assert row.loc["01042025", "stage"] == "Learning"
```

- [ ] **Step 2: Run to verify it fails** → `py -m pytest ...::test_preliminary_learning_trend_groups_by_stage -v` → FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
def preliminary_learning_trend(reg: pd.DataFrame, stage_map: dict) -> pd.DataFrame:
    r = reg.copy(); r["stage"] = r["sess_key"].map(stage_map)
    rows = []
    for key, g in r.groupby("sess_key"):
        resp = g[g["resp_log2"] == True]; non = g[g["resp_log2"] == False]  # noqa: E712
        rows.append({"sess_key": key, "stage": g["stage"].iloc[0], "n_resp": int(len(resp)),
                     "mean_c1r_resp": float(resp["c1_r_log2"].mean()) if len(resp) else np.nan,
                     "mean_c1r_nonresp": float(non["c1_r_log2"].mean()) if len(non) else np.nan})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run to verify it passes** → PASS.

- [ ] **Step 5: Write the runner + figure, then commit**

```python
# scripts/state_tf_learning/b9_phase0_profile.py  (preliminary section)
"""B9 Phase 0 — free registry preliminary + coverage landscape (BG_039)."""
import os, sys, gc
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from visdetect.analysis import state_tf_learning as stl

SUBJECT = os.environ.get("VISDETECT_SUBJECT", "BG_039")
OUT_FIG = stl._REPO / "FIGURES" / "state_tf_learning" / SUBJECT
OUT_CACHE = stl._REPO / "data" / "cache" / "state_tf_learning"
OUT_FIG.mkdir(parents=True, exist_ok=True); OUT_CACHE.mkdir(parents=True, exist_ok=True)

def _preliminary():
    reg = stl.load_registry(); stage_map = stl.session_stage_map()
    trend = stl.preliminary_learning_trend(reg, stage_map).dropna(subset=["stage"])
    trend.to_csv(OUT_CACHE / "b9_preliminary_trend.csv", index=False)
    present = [s for s in ["Naive", "Learning", "Expert"] if s in set(trend["stage"])]
    trend["stage"] = pd.Categorical(trend["stage"], present, ordered=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for col, lbl in [("mean_c1r_resp", "TF-responsive"), ("mean_c1r_nonresp", "non-responsive")]:
        m = trend.groupby("stage")[col].mean()
        ax.plot(range(len(m)), m.values, "-o", label=lbl)
    ax.set_xticks(range(len(present))); ax.set_xticklabels(present)
    ax.set_ylabel("mean c1_r (registry, whole-session)")
    ax.set_title(f"{SUBJECT}: TF-encoding vs stage (registry-only preliminary)")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT_FIG / "b9_preliminary_trend.png", dpi=150); plt.close(fig)

if __name__ == "__main__":
    _preliminary()
```

```bash
py scripts/state_tf_learning/b9_phase0_profile.py
git add src/visdetect/analysis/state_tf_learning.py scripts/state_tf_learning/b9_phase0_profile.py tests/analysis/test_state_tf_learning.py
git commit -m "feat(B9): free registry-only learning preliminary (Phase 0a) + figure"
```

---

### Task 6: Phase 0 coverage profiler + `usable` gate

**Files:**
- Modify: `src/visdetect/analysis/state_tf_learning.py`, `scripts/state_tf_learning/b9_phase0_profile.py`
- Test: `tests/analysis/test_state_tf_learning.py`

**Interfaces:**
- Produces: `count_tf_pulses(session, subset_idx, cfg) -> tuple[int,int]`; `coverage_row(session, tags, reg_sess, state, stage, cfg) -> dict`. `usable = (n_conf_trials >= n_folds) AND (n_fast >= min_pulses_per_label*n_folds) AND (n_slow >= same) AND (n_resp_units >= 1)`.

- [ ] **Step 1: Write the failing test**

```python
def test_coverage_usable_flag():
    sess = _make_tf_session(n_trials=60, seed=2); cfg = stl.b9_cfg()
    tags = pd.DataFrame({"trial_idx": list(range(60)),
                         "state_label": ["StimSens"] * 60, "state_confidence": [0.99] * 60})
    reg_sess = pd.DataFrame({"unit": [100, 101, 102], "resp_log2": [True, False, False],
                             "c1_r_log2": [0.4, 0.02, 0.03]})
    row = stl.coverage_row(sess, tags, reg_sess, "StimSens", "Expert", cfg)
    assert row["n_conf_trials"] == 60 and row["n_resp_units"] == 1
    assert isinstance(row["usable"], (bool, np.bool_))
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
def count_tf_pulses(session, subset_idx, cfg):
    trials_regs, _ = session_trial_regressors(session, cfg)
    subset_idx = [i for i in subset_idx if 0 <= i < len(trials_regs)]
    if len(subset_idx) < cfg.n_folds:
        return 0, 0
    design = assemble_design([trials_regs[i] for i in subset_idx], cfg)
    if design.bin_edges.size == 0:
        return 0, 0
    fast, slow = pulse_times_from_tf(design, cfg)
    return int(np.asarray(fast).size), int(np.asarray(slow).size)


def coverage_row(session, tags, reg_sess, state, stage, cfg) -> dict:
    idx = state_trial_indices(tags, state)
    n_fast, n_slow = count_tf_pulses(session, idx, cfg)
    resp = set(reg_sess.loc[reg_sess["resp_log2"] == True, "unit"].astype(int))    # noqa: E712
    non = set(reg_sess.loc[reg_sess["resp_log2"] == False, "unit"].astype(int))    # noqa: E712
    need = cfg.min_pulses_per_label * cfg.n_folds
    usable = bool(len(idx) >= cfg.n_folds and n_fast >= need and n_slow >= need and len(resp) >= 1)
    return {"state": state, "stage": stage, "n_conf_trials": len(idx),
            "n_fast_pulses": n_fast, "n_slow_pulses": n_slow,
            "n_resp_units": len(resp), "n_nonresp_units": len(non), "usable": usable}
```

- [ ] **Step 4: Run to verify it passes** → PASS.

- [ ] **Step 5: Extend the runner (loop BG_039 sessions → coverage table + landscape figure), then commit**

Add a `_coverage()` function to `b9_phase0_profile.py`: for each `sess_key` in `load_registry()` that has a state-tag file, `load_session`, build `coverage_row` for `state ∈ {StimSens, Disengaged}` with `stage = session_stage_map()[key]`, append `sess_key`; write `data/cache/state_tf_learning/b9_coverage.csv`; plot a sessions×state grid coloured by `n_conf_trials` with a `usable` marker → `FIGURES/state_tf_learning/BG_039/b9_coverage_landscape.png`; `del sess; gc.collect()` per iteration. Call both `_preliminary()` and `_coverage()` in `__main__`.

```bash
py scripts/state_tf_learning/b9_phase0_profile.py
git add src/visdetect/analysis/state_tf_learning.py scripts/state_tf_learning/b9_phase0_profile.py tests/analysis/test_state_tf_learning.py
git commit -m "feat(B9): Phase-0 coverage profiler + usable gate + landscape figure"
```

**GATE:** inspect `b9_coverage.csv`. If no state is `usable` at ≥2 stages, STOP and report the ceiling (spec §10b: fewer folds / relax early boundary / Learning-vs-Expert / StimSens-only). Otherwise continue.

---

### Task 7: Phase 1 run — state-conditioned `c1_r` across usable sessions

**Files:**
- Create: `scripts/state_tf_learning/b9_phase1_run.py`
- Test: `tests/analysis/test_state_tf_learning.py`

**Interfaces:**
- Produces: `data/cache/state_tf_learning/b9_phase1_encoding.csv` — cols `sess_key, stage, state, unit, resp_class ('responsive'|'nonresponsive'), c1_r, c2_p, n_spikes, n_trials`.

- [ ] **Step 1: Write the failing test** (assembly contract on two synthetic sessions)

```python
def test_phase1_assembles_encoding_rows():
    cfg = stl.b9_cfg(); frames = []
    for key in ("01042025", "16062025"):
        sess = _make_tf_session(n_trials=60, seed=abs(hash(key)) % 100)
        df = stl.state_conditioned_encoding(sess, list(range(60)), cfg)
        df["sess_key"] = key; frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    assert {"unit", "c1_r", "sess_key"}.issubset(out.columns) and out["sess_key"].nunique() == 2
```

- [ ] **Step 2: Run to verify it passes-shaped** → `py -m pytest ...::test_phase1_assembles_encoding_rows -v` → PASS.

- [ ] **Step 3: Write the runner**

```python
# scripts/state_tf_learning/b9_phase1_run.py
"""B9 Phase 1 — state-conditioned c1_r per usable session×state (BG_039)."""
import os, sys, gc
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from visdetect.analysis import state_tf_learning as stl
from visdetect.core.session import load_session

SUBJECT = os.environ.get("VISDETECT_SUBJECT", "BG_039")
CACHE = stl._REPO / "data" / "cache" / "state_tf_learning"

def main(states=("StimSens", "Disengaged")):
    reg = stl.load_registry(); cfg = stl.b9_cfg(); stage_map = stl.session_stage_map()
    cov = pd.read_csv(CACHE / "b9_coverage.csv")
    usable = cov[cov["usable"] == True]                                   # noqa: E712
    out = []
    for key in sorted(usable["sess_key"].astype(str).unique()):
        pkl = stl.PKL_ROOT / SUBJECT / f"{SUBJECT}_{key}.pkl"
        if not pkl.exists():
            continue
        sess = load_session(str(pkl)); tags = stl.load_state_tags(SUBJECT, key)
        reg_s = reg[reg.sess_key == key]
        resp_map = dict(zip(reg_s["unit"].astype(int), reg_s["resp_log2"]))
        for state in states:
            if not bool(usable[(usable.sess_key.astype(str) == key) & (usable.state == state)]["usable"].any()):
                continue
            df = stl.state_conditioned_encoding(sess, stl.state_trial_indices(tags, state), cfg)
            if df.empty:
                continue
            df["sess_key"] = key; df["stage"] = stage_map.get(key); df["state"] = state
            df["resp_class"] = df["unit"].map(
                lambda u: "responsive" if resp_map.get(int(u), False) else "nonresponsive")
            out.append(df)
        del sess; gc.collect()
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    res.to_csv(CACHE / "b9_phase1_encoding.csv", index=False)
    print(f"[B9] wrote {len(res)} rows to b9_phase1_encoding.csv")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Smoke run** (after Phase 0 wrote `b9_coverage.csv`): `py scripts/state_tf_learning/b9_phase1_run.py` → writes `b9_phase1_encoding.csv`.

- [ ] **Step 5: Commit**

```bash
git add scripts/state_tf_learning/b9_phase1_run.py tests/analysis/test_state_tf_learning.py
git commit -m "feat(B9): Phase-1 runner — state-conditioned c1_r per usable session/state"
```

---

### Task 8: Phase 1 stats + headline figure

**Files:**
- Modify: `src/visdetect/analysis/state_tf_learning.py`
- Create: `scripts/state_tf_learning/b9_phase1_figure.py`
- Test: `tests/analysis/test_state_tf_learning.py`

**Interfaces:**
- Produces: `stage_class_interaction(df) -> dict` (mixed-effects `c1_r ~ C(stage)*C(resp_class)`, `groups=sess_key`; returns interaction p + per-class deltas). Headline PNG + `b9_phase1_stats.csv`.

- [ ] **Step 1: Write the failing test** (planted interaction: responsive rises, non-responsive flat)

```python
def test_stage_class_interaction_recovers_planted_effect():
    rng = np.random.default_rng(0); rows = []
    for stage, resp_mu, non_mu in [("Learning", 0.20, 0.02), ("Expert", 0.45, 0.02)]:
        for s in range(6):
            for _ in range(8):
                rows.append({"sess_key": f"{stage}{s}", "stage": stage,
                             "resp_class": "responsive", "c1_r": resp_mu + rng.normal(0, 0.05)})
                rows.append({"sess_key": f"{stage}{s}", "stage": stage,
                             "resp_class": "nonresponsive", "c1_r": non_mu + rng.normal(0, 0.05)})
    res = stl.stage_class_interaction(pd.DataFrame(rows))
    assert res["interaction_p"] < 0.05
    assert res["resp_delta"] > 0.15 and abs(res["nonresp_delta"]) < 0.05
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
import statsmodels.formula.api as smf

def stage_class_interaction(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["c1_r"]).copy()
    fit = smf.mixedlm("c1_r ~ C(stage) * C(resp_class)", d, groups=d["sess_key"]).fit(method="lbfgs")
    ix = [p for p in fit.params.index if ":" in p and "resp_class" in p and "stage" in p]
    def _delta(cls):
        cell = d[d.resp_class == cls].groupby("stage")["c1_r"].median()
        return float(cell.iloc[-1] - cell.iloc[0]) if len(cell) >= 2 else np.nan
    return {"interaction_terms": ix,
            "interaction_p": float(fit.pvalues[ix].min()) if ix else np.nan,
            "resp_delta": _delta("responsive"), "nonresp_delta": _delta("nonresponsive"),
            "aic": float(fit.aic)}
```

- [ ] **Step 4: Run to verify it passes** → PASS.

- [ ] **Step 5: Write the headline-figure runner + commit**

`b9_phase1_figure.py`: load `b9_phase1_encoding.csv`; headline = StimSens per-unit `c1_r` by stage split by `resp_class` (strip+box using `config.STATE_LABEL_COLORS`), annotate `stage_class_interaction` p + deltas; Disengaged as a control panel; bootstrap CIs (1000 resamples, seed=42) on per-class stage medians; save `FIGURES/state_tf_learning/BG_039/b9_headline.png` + `data/cache/state_tf_learning/b9_phase1_stats.csv`.

```bash
py scripts/state_tf_learning/b9_phase1_figure.py
git add src/visdetect/analysis/state_tf_learning.py scripts/state_tf_learning/b9_phase1_figure.py tests/analysis/test_state_tf_learning.py
git commit -m "feat(B9): Phase-1 mixed-effects interaction + headline figure"
```

---

### Task 9: Determinism + README wiring

**Files:**
- Modify: `tests/analysis/test_state_tf_learning.py`
- Create: `scripts/state_tf_learning/README.md`

- [ ] **Step 1: Write the test**

```python
def test_encoding_is_deterministic():
    sess = _make_tf_session(seed=7); cfg = stl.b9_cfg()
    a = stl.state_conditioned_encoding(sess, list(range(60)), cfg).set_index("unit")["c1_r"]
    b = stl.state_conditioned_encoding(sess, list(range(60)), cfg).set_index("unit")["c1_r"]
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)
```

- [ ] **Step 2: Run to verify it passes** (seed=42 fixed) → PASS.

- [ ] **Step 3: Write `scripts/state_tf_learning/README.md`** — run order: (1) `py scripts/state_tf_learning/b9_phase0_profile.py` (preliminary + coverage → inspect `b9_coverage.csv`, confirm `usable` at ≥2 stages); (2) `py scripts/state_tf_learning/b9_phase1_run.py` then `b9_phase1_figure.py`. Note the faithfulness gate (`py -m pytest -k reproduces_registry` — a `slow` test) must pass first; the deferred population decoder; the DMS-pool extension (BG_046) once `region_bank_confirmed`.

- [ ] **Step 4: Run the full unit suite (excluding slow)**

Run: `py -m pytest tests/analysis/test_state_tf_learning.py -m "not slow" -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/analysis/test_state_tf_learning.py scripts/state_tf_learning/README.md
git commit -m "test(B9): determinism guard + run README"
```

---

## Execution order & gates

1. **Task 0** (guard) → Tasks 1–3 (glue) → **Task 4 faithfulness gate MUST pass** before trusting any state split.
2. Task 5 (free preliminary) is independent — run any time for the first-look figure.
3. Task 6 (coverage) → **inspect `b9_coverage.csv`**: if no state is `usable` at ≥2 stages, STOP and report the ceiling.
4. Tasks 7–8 (Phase 1) only on `usable` sessions.
5. Task 9 wraps up.

## Self-review notes (author)

- **Spec coverage:** guard port (Task 0) ↔ spec estimator-pinned header; free preliminary (Task 5) ↔ §6; coverage gate with concrete CV/pulse criterion (Task 6) ↔ §2/§6; state-conditioned `c1_r` reuse (Tasks 3,7) ↔ §4; faithfulness reproduction (Task 4) ↔ §8; mixed-effects stage×class interaction (Task 8) ↔ §7; determinism (Task 9) ↔ §8. Population decoder (1b) is **deferred** per §6/§10a — intentionally not a task.
- **Types consistent:** `c1_r` = B9's re-run column; registry column is `c1_r_log2`; `resp_class ∈ {'responsive','nonresponsive'}`; `sess_key = canonical_session_id(session_date)` everywhere; `b9_cfg()` values are frozen.
- **No placeholders:** every code step has real code; the one "reconcile until tolerance passes" (Task 4 Step 3) is a genuine validation loop, not a stub.
