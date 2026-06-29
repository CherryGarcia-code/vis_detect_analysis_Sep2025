# N1 — Neural urgency-ramp & the B8 timing latent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether expert BG_046 striatal pre-decision activity carries an urgency signal that predicts *when* the animal responds (beyond generic motor prep), is temporal-expectation (φ) specific, and is cell-type-specific (SPN/FSI) — joining the B8 per-trial timing latent to per-trial neural tensors.

**Architecture:** A reusable library module `src/visdetect/analysis/neural_latents.py` (latent↔neural join + verification, windowed feature builder, lick/motor coding direction + projection, response-time decoder with shuffle nulls, φ-vs-ramp discriminability, single-unit ME, across-session u-graded test). Thin orchestration + figure scripts under `scripts/neural_latents/`. Outputs to `FIGURES/neural_latents/BG_046/` and `data/cache/neural_latents/`. TDD throughout; synthetic recovery validates the method before any real-data claim.

**Tech Stack:** Python 3.10 (`.venv`, invoke as `py`), numpy, pandas, scikit-learn (ridge/LDA), statsmodels (mixed-effects), matplotlib. Reuses `visdetect.analysis.{utils,align,config,constants}` and `visdetect.suite.loader`.

**Spec:** `docs/superpowers/specs/2026-06-29-N1-neural-urgency-ramp-timing-latent-design.md`

## Global Constraints

Every task implicitly includes these (verbatim from the spec / project rules):

- **Existence GATE before cell-type:** gate = (prong 1) predict response timing vs trial-shuffle null **AND** (prong 2) survive motor-CD projection. **φ-specificity is NOT a gate prong** — it is a layered claim that must not block C2.
- **NOTE A:** φ-vs-ramp discriminability must be demonstrated **on the actual pre-μ readout window** (φ on its rising flank ≈ a monotonic ramp there), NOT inferred from the decision-time distribution. φ-specificity is the expected exploratory/underpowered leg; prongs 1+2 carry the headline.
- **NOTE B:** the response-time decode must show **within-trial-type graded prediction (especially within hits)**, not just hit-vs-fa separation.
- **Never partial out / never match on `decision_time`** — it is the target. Movement-match on motor-CD magnitude / lick presence only.
- **Window is a swept parameter** (early `[0.5,2.5]`, mid `[2,4]`, late-pre-change), fixed-length, **never ending at the lick** (Latimer & Huk duration-confound guard); PRIMARY = the **latest window still motor-CD-free**.
- **Cell type = `celltype ∈ {SPN, FSI}`** (broad/narrow by waveform); `unknown`/`Unclassified` excluded from SPN/FSI contrasts. **D1/D2 deferred** (not waveform-separable; `opto_tag` default `none`).
- **Trusted dials only** load-bearing: timing (this question) + caution (sibling). **Sharpness/evidence-axis = descriptive**, caveated, never headline.
- **Session ids:** `from visdetect.analysis.config import canonical_session_id` on **both** sides of every join (the deliverable stores the leading-zero-stripped form, e.g. `1072025`). Chronological order via `parse_session_date`, never raw `sorted()`.
- **Sessions** via `load_staging_manifest(qc_only=True)` (stage col `stage`, d′ col `d_prime`, id col `session_name` STRING); **units** via `get_good_cluster_ids()` (prefers `good_and_stable_ids`).
- **Compute hygiene:** heavy independent loops → `concurrent.futures.ProcessPoolExecutor`, param **`n_workers`** (NOT `n_jobs`), BLAS pinned to 1/worker; session pkl loading stays **sequential** with `del sess; gc.collect()`. **NO compute over the X: Samba mount.**
- **Repo placement:** library logic in `src/visdetect/`, NOT `analysis_suite/`; scripts in `scripts/neural_latents/`; figures `FIGURES/neural_latents/BG_046/`; caches `data/cache/neural_latents/`. Plain-language titles/captions + a stats CSV beside every figure.
- **Reproducibility:** `seed=42` everywhere; CV is k=5 stratified by session; nulls ≥200 shuffles; bootstrap 1000.
- **Subagent execution:** every subagent (implementer/reviewer) is **Opus 4.8** (`claude-opus-4-8`). `ddm.py` and the B8 fitters are reference-only — do NOT mutate.

**Key API facts (from recon, treat as ground truth):**
- `build_population_tensor(session, cluster_ids, event_name="Baseline_ON", window, bin_size, outcome_filter=None, trial_indices=None) -> (tensor[n_trials,n_bins,n_units] Hz, bin_centers[n_bins], valid_trials)`. `valid_trials` = **original `session.trials` indices**, ascending; row `r` ↔ trial `valid_trials[r]`; unit col `u` ↔ `cluster_ids[u]`. `EVENT_VALID_OUTCOMES["Baseline_ON"] is None` → all trial types kept.
- `compute_zscore_normalized(tensor, bin_centers, baseline_window)` — shared per-unit mean/std over baseline-window bins × all trials; inclusive bounds in **seconds**; σ floored 1e-6.
- `get_good_cluster_ids(session, min_rate_hz=1.0)` (in `visdetect.analysis.utils`).
- Latent `trial_idx` = **raw `enumerate(session.trials)` index, with gaps**; `trial_in_session` is a DECOY (never use). Join by literal `trial_idx`. `decision_time`/`change_time_planned` are **relative to Baseline_ON (seconds)**.
- Cohort cell types: `build_unit_table(qc_only=True)` → int-keyed `(Session_Date, Cluster_ID)`, column `celltype`.
- Pre-trial baseline window constant: `LOHSE_TRIAL_NORM_BASELINE = (-1.3, -0.3)` (`constants.py`).

---

## File structure

| File | Responsibility |
|---|---|
| `src/visdetect/analysis/neural_latents.py` | Library: `load_latent_table`, `fitted_expert_sessions`, `join_session`, feature builder, `fit_lick_motor_cd`/`project_out_axis`, `decode_response_timing`, `within_type_graded`, `phi_ramp_bases`/`phi_specificity_delta`, `single_unit_timing_glm`, `u_graded_test`. No plotting, no `__main__`. |
| `tests/analysis/test_neural_latents.py` | Unit tests: join+verification, feature builder, motor-CD projection, decode null calibration, φ-discriminability on synthetic, within-type graded, u-graded. |
| `scripts/neural_latents/_synthetic_recovery.py` | Synthetic φ-urgency vs pure-motor recovery + φ-vs-ramp discriminability-on-readout-window check (NOTE A prerequisite). Writes a validation figure + JSON. |
| `scripts/neural_latents/n1_c1_gate.py` | C1: real-data response-time decode, window sweep, trial-shuffle null, motor-CD survival, NOTE-B within-hit graded; gate report + figure. |
| `scripts/neural_latents/n1_phi_specificity.py` | φ-specificity ΔCV on the PRIMARY window (exploratory); figure. |
| `scripts/neural_latents/n1_c2_singleunit_celltype.py` | C2: single-unit ME timing encoding + SPN/FSI breakdown; figure. |
| `scripts/neural_latents/n1_c3_u_graded.py` | C3: across-session u-graded test (mood-controlled); + C4 descriptive evidence-axis strand; figures. |
| `scripts/neural_latents/run_n1.py` | Orchestrator: gate → (if pass) cell-type + φ + u-graded; secondary d′ gradient; assembles `n1_results.json` + a one-page summary figure. |

---

### Task 1: Library scaffold + latent↔neural join with verification (the linchpin)

**Files:**
- Create: `src/visdetect/analysis/neural_latents.py`
- Test: `tests/analysis/test_neural_latents.py`

**Interfaces:**
- Produces: `load_latent_table(path=None) -> pd.DataFrame` (adds `sess_canon`); `fitted_expert_sessions(df) -> list[str]` (chronological canonical ids with trusted latents); `join_session(session, latent_rows, *, window, bin_size=0.025, baseline_window=(-1.3,-0.3), min_rate_hz=1.0, verify=True) -> JoinResult` where `JoinResult` is a dataclass `(z, bin_centers, y, unit_ids, kept_trials)`; `z` shape `(n_kept, n_bins, n_units)`, `y` a DataFrame (one row per kept trial, in `z` row order) with columns `trial_idx, outcome, change_size, decision_time, change_time_planned, change_reached, state_label, timing_urgency_at_decision, itchiness_caution, sharpness_drift, evidence_integral_at_decision, expected_change_time`.
- Consumes: `build_population_tensor`, `compute_zscore_normalized`, `get_good_cluster_ids` (`visdetect.analysis.utils`); `canonical_session_id`, `parse_session_date` (`visdetect.analysis.config`).

- [ ] **Step 1: Write the failing test** (`tests/analysis/test_neural_latents.py`)

```python
import types, numpy as np, pandas as pd, pytest
from visdetect.analysis import neural_latents as nl

def _fake_session(outcomes, change_times, change_sizes, baseline_on, n_clusters=4, seed=0):
    """Minimal Session stand-in for build_population_tensor: trials with
    trialoutcome/change_time/change_size, ni_events Baseline_ON/Change_ON
    (absolute s), clusters with cluster_id/spike_times, and good id lists."""
    rng = np.random.default_rng(seed)
    trials = [types.SimpleNamespace(trialoutcome=o, change_time=ct, change_size=cs)
              for o, ct, cs in zip(outcomes, change_times, change_sizes)]
    change_on = [b + ct if o in ("hit", "miss") else np.nan
                 for b, ct, o in zip(baseline_on, change_times, outcomes)]
    ni = {"Baseline_ON": np.array(baseline_on, float),
          "Change_ON": np.array(change_on, float)}
    clusters = [types.SimpleNamespace(cluster_id=cid,
                spike_times=np.sort(rng.uniform(0, max(baseline_on) + 12, 4000)))
                for cid in range(10, 10 + n_clusters)]
    return types.SimpleNamespace(trials=trials, ni_events=ni, clusters=clusters,
        good_and_stable_ids=[c.cluster_id for c in clusters], good_cluster_ids=[])

def test_join_keys_by_literal_trial_idx_with_gaps():
    # 6 session trials; latent table covers a NON-contiguous subset (gaps),
    # mimicking the real deliverable (abort/ref dropped, not renumbered).
    outcomes = ["abort", "hit", "fa", "miss", "ref", "hit"]
    cts      = [7.0,      6.9,   7.1,  7.2,    7.0,   6.8]
    css      = [1.0,      2.0,   1.0,  1.5,    1.0,   4.0]
    base     = [10.0, 30.0, 55.0, 80.0, 105.0, 130.0]
    sess = _fake_session(outcomes, cts, css, base)
    # latent rows reference trial_idx 1,2,3,5 (skip aborts/ref 0,4) — note the GAP at 4
    latent_rows = pd.DataFrame([
        dict(trial_idx=1, outcome="hit",  change_size=2.0, decision_time=7.4,
             change_time_planned=6.9, change_reached=True,  state_label="StimSens",
             timing_urgency_at_decision=0.3, itchiness_caution=-5.0, sharpness_drift=1.0,
             evidence_integral_at_decision=0.5, expected_change_time=7.0),
        dict(trial_idx=2, outcome="fa",   change_size=1.0, decision_time=3.0,
             change_time_planned=7.1, change_reached=False, state_label="Impulsive",
             timing_urgency_at_decision=0.1, itchiness_caution=-4.5, sharpness_drift=1.0,
             evidence_integral_at_decision=0.0, expected_change_time=7.0),
        dict(trial_idx=3, outcome="miss", change_size=1.5, decision_time=9.355,
             change_time_planned=7.2, change_reached=True,  state_label="StimSens",
             timing_urgency_at_decision=0.05, itchiness_caution=-5.0, sharpness_drift=1.0,
             evidence_integral_at_decision=0.2, expected_change_time=7.0),
        dict(trial_idx=5, outcome="hit",  change_size=4.0, decision_time=7.1,
             change_time_planned=6.8, change_reached=True,  state_label="StimSens",
             timing_urgency_at_decision=0.4, itchiness_caution=-5.0, sharpness_drift=1.0,
             evidence_integral_at_decision=0.8, expected_change_time=7.0),
    ])
    res = nl.join_session(sess, latent_rows, window=(-1.3, 6.0), bin_size=0.05,
                          baseline_window=(-1.3, -0.3), min_rate_hz=0.0, verify=True)
    # rows align to literal trial_idx, gap-trial 4 absent, decoy ordering not used
    assert list(res.y["trial_idx"]) == [1, 2, 3, 5]
    assert res.z.shape[0] == 4 and res.z.shape[2] == 4   # 4 trials, 4 units
    assert list(res.kept_trials) == [1, 2, 3, 5]
    # verification triple-check passes silently (outcome/size/time match the session)

def test_join_verification_catches_misalignment():
    outcomes = ["hit", "fa"]; cts = [6.9, 7.1]; css = [2.0, 1.0]; base = [10.0, 30.0]
    sess = _fake_session(outcomes, cts, css, base)
    bad = pd.DataFrame([dict(trial_idx=0, outcome="fa",  # WRONG outcome for trial 0 (really hit)
        change_size=2.0, decision_time=7.4, change_time_planned=6.9, change_reached=True,
        state_label="StimSens", timing_urgency_at_decision=0.3, itchiness_caution=-5.0,
        sharpness_drift=1.0, evidence_integral_at_decision=0.5, expected_change_time=7.0)])
    with pytest.raises(AssertionError):
        nl.join_session(sess, bad, window=(-1.3, 6.0), bin_size=0.05,
                        baseline_window=(-1.3, -0.3), min_rate_hz=0.0, verify=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `e:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe -m pytest tests/analysis/test_neural_latents.py -k join -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'visdetect.analysis.neural_latents'`.

- [ ] **Step 3: Write minimal implementation** (`src/visdetect/analysis/neural_latents.py`)

```python
"""N1 neural-latent correspondence: join the B8 per-trial timing latent to
per-trial striatal tensors and test the urgency-ramp hypothesis. Library only
(no plotting / no __main__). See spec 2026-06-29-N1-...-design.md."""
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from visdetect.analysis.config import ROOT, canonical_session_id, parse_session_date
from visdetect.analysis.utils import (
    build_population_tensor, compute_zscore_normalized, get_good_cluster_ids)

DEFAULT_LATENT_CSV = os.path.join(
    ROOT, "data", "cache", "decision_latents", "decision_latents_by_state.csv")

# columns copied verbatim from the latent table into the per-trial y frame
_Y_COLS = ["trial_idx", "outcome", "change_size", "decision_time",
           "change_time_planned", "change_reached", "state_label",
           "timing_urgency_at_decision", "itchiness_caution", "sharpness_drift",
           "evidence_integral_at_decision", "expected_change_time"]

@dataclass
class JoinResult:
    z: np.ndarray            # (n_kept, n_bins, n_units), per-unit shared-baseline z
    bin_centers: np.ndarray  # (n_bins,) seconds rel. Baseline_ON
    y: pd.DataFrame          # one row per kept trial, in z row order
    unit_ids: list           # cluster ids, positional with z's unit axis
    kept_trials: list        # original session trial indices, in z row order

def load_latent_table(path=None):
    df = pd.read_csv(path or DEFAULT_LATENT_CSV, dtype={"session_name": str})
    df["sess_canon"] = df["session_name"].map(canonical_session_id)
    return df

def fitted_expert_sessions(df):
    fitted = df.loc[df["sharpness_drift"].notna(), "sess_canon"].unique()
    return sorted(fitted, key=parse_session_date)

def join_session(session, latent_rows, *, window, bin_size=0.025,
                 baseline_window=(-1.3, -0.3), min_rate_hz=1.0, verify=True):
    good_ids = get_good_cluster_ids(session, min_rate_hz=min_rate_hz)
    tensor, bin_centers, valid_trials = build_population_tensor(
        session, cluster_ids=good_ids, event_name="Baseline_ON",
        window=window, bin_size=bin_size)
    lut = {int(getattr(r, "trial_idx")): r for r in latent_rows.itertuples(index=False)}
    keep = [r for r, ti in enumerate(valid_trials) if int(ti) in lut]
    if not keep:
        raise ValueError(f"join_session: no overlap between tensor trials and "
                         f"latent trial_idx (n_valid={len(valid_trials)}, "
                         f"n_latent={len(lut)})")
    kept_trials = [int(valid_trials[r]) for r in keep]
    z = compute_zscore_normalized(tensor[keep], bin_centers, baseline_window)
    y = pd.DataFrame([{c: getattr(lut[ti], c) for c in _Y_COLS} for ti in kept_trials])
    if verify:
        _verify_join(session, kept_trials, lut)
    return JoinResult(z=z, bin_centers=bin_centers, y=y,
                      unit_ids=list(good_ids), kept_trials=kept_trials)

def _verify_join(session, kept_trials, lut):
    """Triple-check (outcome / change_size / change_time) that trial_idx indexes
    the SAME trial the latent row describes. Fails loud on any mismatch."""
    base = np.asarray(session.ni_events["Baseline_ON"]).ravel()
    assert len(base) >= len(session.trials), (
        f"Baseline_ON ({len(base)}) shorter than trials ({len(session.trials)})")
    for ti in kept_trials:
        tr, lr = session.trials[ti], lut[ti]
        assert (getattr(tr, "trialoutcome", "") or "").lower() == str(lr.outcome).lower(), \
            f"outcome mismatch at trial_idx={ti}"
        assert np.isclose(float(tr.change_size), float(lr.change_size)), \
            f"change_size mismatch at trial_idx={ti}"
        if np.isfinite(float(lr.change_time_planned)):
            assert np.isclose(float(tr.change_time), float(lr.change_time_planned), atol=1e-6), \
                f"change_time mismatch at trial_idx={ti}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/analysis/test_neural_latents.py -k join -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/neural_latents.py tests/analysis/test_neural_latents.py
git commit -m "feat(N1): latent<->neural join keyed by literal trial_idx + triple-check verification"
```

---

### Task 2: Windowed per-trial feature builder

**Files:**
- Modify: `src/visdetect/analysis/neural_latents.py`
- Test: `tests/analysis/test_neural_latents.py`

**Interfaces:**
- Produces: `window_feature_matrix(z, bin_centers, win) -> np.ndarray` shape `(n_trials, n_units)` — mean z over the half-open `[win[0], win[1])` slice (the fixed-length readout window); raises if the window contains no bins. `WINDOWS = {"early": (0.5, 2.5), "mid": (2.0, 4.0), "late": (4.0, 6.0)}` (the swept positions; "late" is the latest pre-μ slice for μ≈6.7–7.5 s).

- [ ] **Step 1: Write the failing test**

```python
def test_window_feature_matrix_means_over_fixed_window():
    n_tr, n_units = 5, 3
    bin_centers = np.arange(0.0, 6.0, 0.05)
    z = np.zeros((n_tr, bin_centers.size, n_units))
    early = (bin_centers >= 0.5) & (bin_centers < 2.5)
    z[:, early, :] = 2.0                       # constant 2.0 inside the early window
    X = nl.window_feature_matrix(z, bin_centers, nl.WINDOWS["early"])
    assert X.shape == (n_tr, n_units)
    assert np.allclose(X, 2.0)
    with pytest.raises(ValueError):
        nl.window_feature_matrix(z, bin_centers, (100.0, 200.0))  # no bins
```

- [ ] **Step 2: Run** `... -k window_feature -v` → FAIL (`AttributeError: WINDOWS`).

- [ ] **Step 3: Implement** (append to `neural_latents.py`)

```python
WINDOWS = {"early": (0.5, 2.5), "mid": (2.0, 4.0), "late": (4.0, 6.0)}

def window_feature_matrix(z, bin_centers, win):
    lo, hi = win
    mask = (bin_centers >= lo) & (bin_centers < hi)
    if not mask.any():
        raise ValueError(f"window {win} contains no bin centers "
                         f"(range {bin_centers.min():.2f}..{bin_centers.max():.2f})")
    return z[:, mask, :].mean(axis=1)
```

- [ ] **Step 4: Run** `... -k window_feature -v` → PASS.
- [ ] **Step 5: Commit** `feat(N1): fixed-length windowed feature matrix + swept window positions`.

---

### Task 3: Lick/motor coding direction + projection (movement control)

**Files:** Modify `neural_latents.py`; Test `tests/analysis/test_neural_latents.py`.

**Interfaces:**
- Produces: `fit_lick_motor_cd(z_lick, bin_centers, *, lick_window=(-0.15,0.05), base_window=(-1.0,-0.5)) -> np.ndarray` (unit-norm motor axis over units, from peri-lick vs pre-lick mean z; uses `compute_lda_cd`-style sign so peri-lick > pre); `project_out_axis(X, axis) -> np.ndarray` (removes the component of each trial's feature vector along `axis`); `motor_axis_signal(X, axis) -> np.ndarray` (per-trial projection magnitude, for movement-matching + the "is this window movement-free?" check).
- Consumes: `compute_lda_cd` (`visdetect.analysis.utils`).
- *Note:* the motor CD is built **fresh** here from a **lick-aligned** tensor (Task 5 builds it per session via `build_population_tensor(event_name="FA"/"Hit")`); it does **not** reuse the Fig14c template. The plan audits Fig14c only as an optional cross-check (Task 8 risk note), never as a dependency.

- [ ] **Step 1: Write the failing test**

```python
def test_project_out_axis_removes_component():
    rng = np.random.default_rng(0)
    axis = np.array([1.0, 0.0, 0.0])
    X = rng.normal(size=(20, 3))
    Xp = nl.project_out_axis(X, axis)
    assert np.allclose(Xp @ axis, 0.0, atol=1e-10)          # component along axis removed
    # variance orthogonal to axis preserved
    assert np.allclose(Xp[:, 1:], X[:, 1:], atol=1e-10)

def test_motor_axis_signal_tracks_projection():
    axis = np.array([0.0, 1.0, 0.0])
    X = np.array([[0, 3.0, 0], [0, -1.0, 0]])
    assert np.allclose(nl.motor_axis_signal(X, axis), [3.0, -1.0])
```

- [ ] **Step 2: Run** `... -k "project_out or motor_axis" -v` → FAIL.

- [ ] **Step 3: Implement**

```python
from visdetect.analysis.utils import compute_lda_cd

def project_out_axis(X, axis):
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    return X - np.outer(X @ a, a)

def motor_axis_signal(X, axis):
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    return X @ a

def fit_lick_motor_cd(z_lick, bin_centers, *, lick_window=(-0.15, 0.05),
                      base_window=(-1.0, -0.5)):
    """Unit-norm motor axis: LDA between peri-lick and pre-lick population states
    on a LICK-aligned tensor (t=0 = lick). Class 1 = peri-lick."""
    def _feat(win):
        m = (bin_centers >= win[0]) & (bin_centers < win[1])
        return z_lick[:, m, :].mean(axis=1)
    peri, pre = _feat(lick_window), _feat(base_window)
    X = np.vstack([pre, peri])
    y = np.r_[np.zeros(len(pre)), np.ones(len(peri))]
    return compute_lda_cd(X, y, method="sklearn", reg=1.0, reg_style="flat")
```

- [ ] **Step 4: Run** `... -k "project_out or motor_axis" -v` → PASS.
- [ ] **Step 5: Commit** `feat(N1): fresh lick/motor coding direction + projection + movement-axis signal`.

---

### Task 4: Response-time decoder + within-type graded + trial-shuffle null

**Files:** Modify `neural_latents.py`; Test `tests/analysis/test_neural_latents.py`.

**Interfaces:**
- Produces:
  - `decode_response_timing(X, y, groups, *, n_splits=5, seed=42) -> dict(r=..., r2=..., y_pred=...)` — ridge regression (`sklearn.linear_model.RidgeCV`), `GroupKFold` by session `groups`, returns out-of-fold Spearman `r`, `r2`, and `y_pred`.
  - `shuffle_null(X, y, groups, *, n=200, seed=42) -> np.ndarray` — distribution of out-of-fold Spearman r under trial-shuffled `y` (shuffle within session group); chance = `mean ± 2 SD`.
  - `within_type_graded(y_pred, y_true, trial_type) -> dict[type]->spearman` — Spearman of pred-vs-true computed **within each trial type** (NOTE B; the headline requires graded prediction within `hit`).
- Consumes: `scipy.stats.spearmanr`, `sklearn.linear_model.RidgeCV`, `sklearn.model_selection.GroupKFold`.

- [ ] **Step 1: Write the failing test**

```python
def test_decode_recovers_planted_signal_and_null_is_chance():
    rng = np.random.default_rng(1)
    n, p = 300, 8
    groups = rng.integers(0, 5, n)                 # 5 "sessions"
    w = rng.normal(size=p)
    y = rng.normal(size=n)
    X = y[:, None] * w[None, :] + rng.normal(scale=0.5, size=(n, p))  # X encodes y
    out = nl.decode_response_timing(X, y, groups, n_splits=5, seed=42)
    assert out["r"] > 0.5                            # real signal decodes
    null = nl.shuffle_null(X, y, groups, n=100, seed=42)
    assert out["r"] > null.mean() + 2 * null.std()   # beats trial-shuffle null
    assert abs(null.mean()) < 0.15                    # null centered near chance

def test_within_type_graded_separates_types():
    y_true = np.array([1., 2., 3., 10., 11., 12.])
    y_pred = np.array([1.1, 2.0, 2.9, 10.2, 10.9, 12.1])
    tt = np.array(["fa", "fa", "fa", "hit", "hit", "hit"])
    g = nl.within_type_graded(y_pred, y_true, tt)
    assert g["hit"] > 0.8 and g["fa"] > 0.8          # graded WITHIN each type
```

- [ ] **Step 2: Run** `... -k "decode or within_type" -v` → FAIL.

- [ ] **Step 3: Implement**

```python
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

def _oof_predict(X, y, groups, n_splits, seed):
    y_pred = np.full_like(y, np.nan, dtype=float)
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    for tr, te in gkf.split(X, y, groups):
        model = RidgeCV(alphas=np.logspace(-3, 3, 13))
        model.fit(X[tr], y[tr])
        y_pred[te] = model.predict(X[te])
    return y_pred

def decode_response_timing(X, y, groups, *, n_splits=5, seed=42):
    X, y, groups = np.asarray(X, float), np.asarray(y, float), np.asarray(groups)
    y_pred = _oof_predict(X, y, groups, n_splits, seed)
    r = spearmanr(y_pred, y).correlation
    ss_res = np.sum((y - y_pred) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    return {"r": float(r), "r2": float(1 - ss_res / ss_tot), "y_pred": y_pred}

def shuffle_null(X, y, groups, *, n=200, seed=42):
    rng = np.random.default_rng(seed)
    X, y, groups = np.asarray(X, float), np.asarray(y, float), np.asarray(groups)
    out = np.empty(n)
    for i in range(n):
        ys = y.copy()
        for g in np.unique(groups):                 # shuffle WITHIN session
            idx = np.where(groups == g)[0]
            ys[idx] = rng.permutation(ys[idx])
        yp = _oof_predict(X, ys, groups, 5, seed)
        out[i] = spearmanr(yp, ys).correlation
    return out

def within_type_graded(y_pred, y_true, trial_type):
    y_pred, y_true, tt = map(np.asarray, (y_pred, y_true, trial_type))
    res = {}
    for t in np.unique(tt):
        m = tt == t
        if m.sum() >= 5:
            res[str(t)] = float(spearmanr(y_pred[m], y_true[m]).correlation)
    return res
```

- [ ] **Step 4: Run** `... -k "decode or within_type" -v` → PASS.
- [ ] **Step 5: Commit** `feat(N1): grouped-CV response-time decoder + within-session shuffle null + within-type graded`.

---

### Task 5: Synthetic recovery + φ-vs-ramp discriminability on the readout window (NOTE A prerequisite)

**Files:** Create `scripts/neural_latents/_synthetic_recovery.py`; Modify `neural_latents.py` (φ/ramp bases); Test `tests/analysis/test_neural_latents.py`.

**Interfaces:**
- Produces: `phi_ramp_bases(t, mu, sigma=0.8) -> dict("phi"=gaussian bump at mu, "ramp"=linear monotonic)`; `phi_specificity_delta(X, y, groups, t_window_center, mu_by_trial, sigma) -> dict` comparing CV decode using a φ-weighted vs ramp-weighted temporal readout (ΔCV r, bootstrap CI). `_synthetic_recovery.py` simulates (a) a φ-urgency ramp population and (b) a pure-motor ramp, over the REAL μ range and `decision_time` distribution restricted to the **pre-μ readout window**, and checks: decode recovers timing in both; motor-CD projection KILLS the pure-motor case but spares φ-urgency; φ-vs-ramp separable **on the readout window** (or reports "underpowered").

- [ ] **Step 1: Write the failing test** (discriminability behaves correctly on synthetic)

```python
def test_phi_ramp_bases_shapes():
    t = np.linspace(0, 6, 120)
    b = nl.phi_ramp_bases(t, mu=7.0, sigma=0.8)
    assert b["phi"].argmax() == len(t) - 1          # rising flank: phi peaks toward mu>6
    assert np.all(np.diff(b["ramp"]) > 0)           # ramp strictly monotonic
    # over a pre-mu window phi and ramp are highly collinear (NOTE A) -> documents the tension
    assert np.corrcoef(b["phi"], b["ramp"])[0, 1] > 0.95
```

- [ ] **Step 2: Run** `... -k phi_ramp_bases -v` → FAIL.

- [ ] **Step 3: Implement** `phi_ramp_bases` + `phi_specificity_delta` in `neural_latents.py`

```python
def phi_ramp_bases(t, mu, sigma=0.8):
    t = np.asarray(t, float)
    phi = np.exp(-0.5 * ((t - mu) / sigma) ** 2)
    ramp = (t - t.min()) / (t.max() - t.min() + 1e-12)
    return {"phi": phi, "ramp": ramp}

def phi_specificity_delta(Xt, y, groups, t, mu, sigma=0.8, *, seed=42):
    """Xt: (n_trials, n_bins, n_units). Compare decode using a phi-weighted vs
    ramp-weighted temporal collapse of the readout window. Returns delta CV r."""
    b = phi_ramp_bases(t, mu, sigma)
    def _decode(weight):
        w = weight / (weight.sum() + 1e-12)
        Xw = np.tensordot(Xt, w, axes=([1], [0]))    # (n_trials, n_units)
        return decode_response_timing(Xw, y, groups, seed=seed)["r"]
    r_phi, r_ramp = _decode(b["phi"]), _decode(b["ramp"])
    return {"r_phi": r_phi, "r_ramp": r_ramp, "delta": r_phi - r_ramp}
```

- [ ] **Step 4: Run** `... -k phi_ramp_bases -v` → PASS.

- [ ] **Step 5: Write `_synthetic_recovery.py`** (simulate φ-urgency vs pure-motor; the NOTE-A check)

```python
"""N1 synthetic recovery + NOTE-A discriminability prerequisite.
Validates the decoder BEFORE any real-data claim:
 (1) decode recovers planted response-timing in both a phi-urgency and a
     pure-motor population;
 (2) motor-CD projection KILLS the pure-motor case but spares phi-urgency;
 (3) phi-vs-ramp is separable ON THE PRE-mu READOUT WINDOW (else: underpowered).
Writes FIGURES/neural_latents/BG_046/n1_synthetic_recovery.png + a JSON verdict.
NO real data, NO X: compute. Run: PYTHONPATH=src py scripts/neural_latents/_synthetic_recovery.py
"""
import os, json, numpy as np
from visdetect.analysis.config import ROOT
from visdetect.analysis import neural_latents as nl
# ... simulate per spec; reuse nl.decode_response_timing / project_out_axis /
#     fit_lick_motor_cd / phi_specificity_delta; real mu range 6.7-7.5; readout
#     windows from nl.WINDOWS; write verdict JSON {recovers, motor_killed,
#     phi_separable_on_window} + figure. (Full body authored at execution.)
```

*(The synthetic body is mechanical given the library functions above; it asserts the three conditions and writes the verdict. Execution agent fills the simulation loop using `nl.*` — no new science.)*

- [ ] **Step 6: Run the harness** `PYTHONPATH=src .venv/Scripts/python.exe scripts/neural_latents/_synthetic_recovery.py`
Expected: prints `recovers=True motor_killed=True phi_separable_on_window=<bool>`; writes figure + `data/cache/neural_latents/n1_synthetic_verdict.json`.

- [ ] **Step 7: Commit** `feat(N1): synthetic recovery + NOTE-A phi-vs-ramp discriminability prerequisite`.

---

### Task 6: C1 real-data gate — window sweep, null, motor-CD survival, within-hit graded

**Files:** Create `scripts/neural_latents/n1_c1_gate.py`; Modify `neural_latents.py` if a shared per-session-build helper is factored.

**Interfaces:**
- Produces (script, no new public lib API required beyond Tasks 1–5): a per-session loop (sequential pkl load, `del sess; gc.collect()`) building, for each swept window, the pooled feature matrix `X` (trials × units, units stacked per session into a block-diagonal/within-session decode), target `y = decision_time` on **lick trials (hit+fa)**, `groups = session`. Runs `decode_response_timing`, `shuffle_null`, motor-CD `project_out_axis` survival, and `within_type_graded`. Selection rule: PRIMARY window = the **latest window whose `motor_axis_signal` in that window is not significantly above its pre-trial baseline** (movement-free check). Writes `n1_c1_gate.json` (per-window r, null mean±2SD, survives_projection, within-hit r) + `fig_n1_c1_gate.png` + `n1_c1_gate_stats.csv`.

- [ ] **Step 1: Write the failing test** (a thin integration smoke on synthetic sessions, in the test file)

```python
def test_c1_gate_pipeline_smoke(tmp_path, monkeypatch):
    # Two fake sessions whose early-window activity encodes decision_time;
    # assert the gate helper returns r>null and within-hit graded > 0.
    from scripts.neural_latents import n1_c1_gate as gate   # importable module
    res = gate.evaluate_window_on_features(  # pure function: features in, verdict out
        X=_synthetic_encoding_X(), y=_synthetic_y(), groups=_synthetic_groups(),
        trial_type=_synthetic_types(), motor_axis=_synthetic_motor_axis(),
        n_null=50, seed=42)
    assert res["r"] > res["null_mean"] + 2 * res["null_sd"]
    assert res["within"]["hit"] > 0.0
    assert "survives_projection" in res
```

*(Helpers `_synthetic_*` are added to the test file; they reuse the Task-4 planted-signal generator. `evaluate_window_on_features` is the pure, testable core of the gate script — the per-session pkl loop calls it.)*

- [ ] **Step 2: Run** `... -k c1_gate_pipeline -v` → FAIL (module/func missing).

- [ ] **Step 3: Implement `n1_c1_gate.py`** with a pure `evaluate_window_on_features(...)` core + a `main()` that does the sequential per-session build and writes outputs. Core:

```python
def evaluate_window_on_features(X, y, groups, trial_type, motor_axis, *,
                                n_null=200, seed=42):
    from visdetect.analysis import neural_latents as nl
    base = nl.decode_response_timing(X, y, groups, seed=seed)
    null = nl.shuffle_null(X, y, groups, n=n_null, seed=seed)
    Xp = nl.project_out_axis(X, motor_axis)
    proj = nl.decode_response_timing(Xp, y, groups, seed=seed)
    within = nl.within_type_graded(base["y_pred"], y, trial_type)
    survives = proj["r"] > null.mean() + 2 * null.std()
    return {"r": base["r"], "r2": base["r2"], "null_mean": float(null.mean()),
            "null_sd": float(null.std()), "r_after_projection": proj["r"],
            "survives_projection": bool(survives), "within": within}
```

`main()`: load `load_latent_table()`, `fitted_expert_sessions()`, then per session `load_session(int(sess))`, `join_session(..., window=(-1.3, 6.0))`, build a lick-aligned tensor for `fit_lick_motor_cd`, restrict `y` to `outcome ∈ {hit, fa}`, accumulate features per window, call `evaluate_window_on_features`, pick PRIMARY by the latest movement-free window, dump JSON + figure (per-window r vs null bar plot; within-hit scatter). Plain-language title.

- [ ] **Step 4: Run** `... -k c1_gate_pipeline -v` → PASS; then run `main()` on real data:
`PYTHONPATH=src .venv/Scripts/python.exe scripts/neural_latents/n1_c1_gate.py`
Expected: writes `data/cache/neural_latents/n1_c1_gate.json`, `FIGURES/neural_latents/BG_046/fig_n1_c1_gate.png`, stats CSV; prints the gate verdict per window + the PRIMARY selection.

- [ ] **Step 5: Commit** `feat(N1): C1 response-time gate — window sweep, null, motor-CD survival, within-hit graded`.

---

### Task 7: φ-specificity on the PRIMARY window (exploratory, non-gating)

**Files:** Create `scripts/neural_latents/n1_phi_specificity.py`.

**Interfaces:** Reuses `nl.phi_specificity_delta` over the PRIMARY window's `(trials × bins × units)` block with per-session `mu = expected_change_time`. **Runs only after the §5 synthetic prerequisite reports `phi_separable_on_window=True`; otherwise the script writes a "tested, underpowered" verdict and exits 0** (per NOTE A / FIX 1 — never blocks anything). Writes `n1_phi_specificity.json` (ΔCV r + bootstrap CI) + `fig_n1_phi_specificity.png`.

- [ ] **Step 1:** Test: `test_phi_specificity_reports_underpowered_when_not_separable` — feed a synthetic case where φ≈ramp; assert verdict `"underpowered"` and no exception.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement: read `n1_synthetic_verdict.json`; if not separable → `{"verdict": "underpowered"}`; else compute `phi_specificity_delta` per session, bootstrap the cross-session mean Δ, write JSON+figure.
- [ ] **Step 4:** Run test → PASS; run on real data.
- [ ] **Step 5:** Commit `feat(N1): phi-specificity (exploratory, gated on synthetic discriminability)`.

---

### Task 8: C2 single-unit ME timing encoding + SPN/FSI breakdown

**Files:** Create `scripts/neural_latents/n1_c2_singleunit_celltype.py`; Modify `neural_latents.py` (`single_unit_timing_glm`).

**Interfaces:**
- Produces: `single_unit_timing_glm(rate, decision_time, session_id) -> dict(beta, p)` — per-unit relationship of PRIMARY-window mean z-rate to `decision_time` via a permutation test (≥200 within-session shuffles of `decision_time`); cohort-level **mixed-effects** `decision_time ~ rate + (1|session)` via `statsmodels.formula.api.mixedlm` for the population readout; per-unit p-values FDR-corrected (`fdr_correct`); encoder fraction + effect-size split by `celltype ∈ {SPN, FSI}` (from `build_unit_table`). **Runs only if Task 6 gate passed** (guard at top of `main`).

- [ ] **Step 1: Write the failing test**

```python
def test_single_unit_glm_detects_encoding_and_fdr():
    rng = np.random.default_rng(2)
    dt = rng.normal(size=120); sess = rng.integers(0, 4, 120)
    rate = 0.8 * dt + rng.normal(scale=0.5, size=120)     # unit encodes timing
    out = nl.single_unit_timing_glm(rate, dt, sess)
    assert out["p"] < 0.05 and out["beta"] > 0
    mask = nl.fdr_correct(np.array([out["p"], 0.9, 0.8]))  # only the real one survives
    assert mask[0] and not mask[1]
```

(`from visdetect.analysis.utils import fdr_correct` re-exported via `nl.fdr_correct = fdr_correct`.)

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `single_unit_timing_glm` (permutation on within-session shuffled `decision_time`, Spearman beta sign) + the cohort `mixedlm` call in the script. Cell-type join: `build_unit_table(qc_only=True)`, filter `celltype.isin({"SPN","FSI"})`, map per `(Session_Date,Cluster_ID)` to the session's `unit_ids` (cast session id via `canonical_session_id`→int both sides). Report encoder fraction + median |beta| per celltype with bootstrap CIs; Mann-Whitney SPN-vs-FSI on |beta|.
- [ ] **Step 4:** Run test → PASS; run on real data → `fig_n1_c2_celltype.png` + stats CSV (report SPN/FSI counts + the chronic-probe under-yield caveat in the caption).
- [ ] **Step 5:** Commit `feat(N1): C2 single-unit timing encoding (ME + permutation/FDR) + SPN/FSI breakdown`.

---

### Task 9: C3 across-session u-graded + C4 descriptive evidence-axis + orchestrator

**Files:** Create `scripts/neural_latents/n1_c3_u_graded.py`, `scripts/neural_latents/run_n1.py`; Modify `neural_latents.py` (`u_graded_test`).

**Interfaces:**
- `u_graded_test(ramp_amp_by_cell, u_by_cell, mood_by_cell) -> dict` — across the ~56 session×mood cells, partial Spearman of neural urgency-ramp amplitude vs fitted `u` **controlling for the binary mood label** (regress both on mood, correlate residuals); bootstrap CI; reports it as correlational/confounded.
- `run_n1.py` — orchestrator: runs Task 6 gate; **iff** gate passes, runs Tasks 7/8/9 + the **secondary within-expert d′ gradient** (Spearman of per-session decode r vs `d_prime` from the manifest, labeled secondary); assembles `data/cache/neural_latents/n1_results.json` + a one-page `fig_n1_summary.png`.

- [ ] **Step 1:** Test `test_u_graded_controls_for_mood` — synthetic cells where the raw u↔ramp correlation is driven entirely by mood; assert the mood-controlled partial drops toward 0 (proving the control works).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `u_graded_test` (residualize on a mood dummy via `numpy.linalg.lstsq`, Spearman of residuals, bootstrap). Implement `n1_c3_u_graded.py` (build per-cell ramp amplitude = mean PRIMARY-window decode projection per session×mood) and the C4 descriptive evidence-axis strand (`evidence_integral_at_decision` single-trial decode, captioned descriptive). Implement `run_n1.py` orchestration with the gate guard + d′ gradient.
- [ ] **Step 4:** Run tests → PASS; run `run_n1.py` end-to-end → all figures + `n1_results.json`.
- [ ] **Step 5:** Commit `feat(N1): C3 u-graded (mood-controlled) + C4 descriptive evidence axis + run_n1 orchestrator`.

---

## Self-Review

**Spec coverage:** §2 gate → T6 (prongs 1+2); §6 φ-specificity + NOTE-A prerequisite → T5 (synthetic) + T7 (exploratory, gated); §4 window sweep + Latimer-Huk fixed window → T2 + T6 selection rule; §5 movement controls → T3 + T6 (projection survival, motor-free selection, never partial RT); NOTE B within-hit graded → T4 + T6; §7-C1 → T6; C2 SPN/FSI (D1/D2 deferred) → T8; C3 u-graded mood-controlled → T9; C4 descriptive evidence → T9; secondary d′ gradient → T9; §8 synthetic recovery → T5; join/canonicalization → T1; §9 join-integrity test → T1; determinism/no-X:/n_workers → Global Constraints + sequential pkl loop. caution-z is correctly ABSENT (moved to sibling). **No gaps.**

**Placeholder scan:** the `_synthetic_recovery.py` body and the figure-drawing inside scripts are described as "authored at execution" but every step names the exact `nl.*` functions to call and the exact verdict/outputs — no "TBD/add error handling." The pure cores (`evaluate_window_on_features`, `phi_specificity_delta`, `single_unit_timing_glm`, `u_graded_test`, the join, decoder, null) are fully coded and tested.

**Type consistency:** `decode_response_timing` returns `dict(r,r2,y_pred)` — consumed identically in T5/T6/T7/T9. `join_session` returns `JoinResult(z,bin_centers,y,unit_ids,kept_trials)` — `y` columns are the single `_Y_COLS` list, consumed by T6/T8/T9. `WINDOWS` keys `early/mid/late` consistent T2→T6→T7. `celltype` values `{SPN,FSI}` consistent T8.

---

## Open choices surfaced for review (do not block the plan)

1. **Pooling units across sessions for the decode (T6).** Units differ per session, so the population decode is naturally **within-session, results pooled** (GroupKFold by session). An alternative is a session-concatenated block-diagonal feature space. The plan uses within-session decode + cross-session pooling of the per-session r (cleaner, no cross-session unit-identity assumption). Flag if you prefer the concatenated variant.
2. **Lick-aligned tensor for the motor CD (T3/T6).** Built via `build_population_tensor(event_name="FA")` and `"Hit"` (the motor events), `window=(-0.5, 0.3)` around the lick. Confirm FA/Hit are the right motor-onset events given no video.
