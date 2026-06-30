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
| `src/visdetect/analysis/neural_latents.py` | Library: `load_latent_table`, `fitted_expert_sessions`, `join_session`, `window_feature_matrix`, `fit_lick_motor_cd`/`project_out_axis`/`motor_axis_signal`, `decode_session`/`decode_cohort`/`within_type_graded`, `phi_ramp_bases`/`phi_specificity_session`, `single_unit_timing_glm`, `u_graded_test`. No plotting, no `__main__`. |
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
- Produces: `fit_lick_motor_cd(z_lick, bin_centers, *, base_window=_FA_BASE, premove_window=_FA_PRE) -> np.ndarray` (unit-norm **preparatory-motor** axis from the pre-movement ramp window vs a clean pre-trial baseline, both **lab-canonical** `EVENT_RESPONSIVENESS_WINDOWS["FA"]/["Hit"]`; `compute_lda_cd` sign so pre-movement > baseline); `project_out_axis(X, axis) -> np.ndarray` (removes each trial's component along `axis`); `motor_axis_signal(X, axis) -> np.ndarray` (per-trial projection magnitude, for movement-matching + the "is this window movement-free?" check).
- Consumes: `compute_lda_cd`, `EVENT_RESPONSIVENESS_WINDOWS` (`visdetect.analysis.constants`).
- *Note:* the motor CD is built **fresh** here from a **lick-aligned** tensor (Task 6 builds it per session via `build_population_tensor(event_name="FA"/"Hit")`, lick-aligned to the **200 ms-corrected** lick time, span `(-2.0, 0.75)`); it does **NOT** reuse the suspect Fig14c template. The windows are **imported, not invented** (review FIX): baseline `(-1.75, -1.25)`, pre-movement `(-0.3, -0.15)` rel. corrected lick — the ported MATLAB preparatory-motor definition shared with `lick.py`. Because it is the *preparatory* ramp (not peri-lick), prong-2 projection asks the stringent, correct question: is "urgency" collinear with the generic motor-prep ramp?

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
from visdetect.analysis.constants import EVENT_RESPONSIVENESS_WINDOWS
# lab-canonical preparatory-motor windows (rel. corrected lick), shared with lick.py:
_FA_BASE, _FA_PRE = EVENT_RESPONSIVENESS_WINDOWS["FA"]   # ((-1.75,-1.25), (-0.3,-0.15))

def project_out_axis(X, axis):
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    return X - np.outer(X @ a, a)

def motor_axis_signal(X, axis):
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    return X @ a

def fit_lick_motor_cd(z_lick, bin_centers, *, base_window=_FA_BASE, premove_window=_FA_PRE):
    """Unit-norm PREPARATORY-motor axis: LDA between the pre-movement ramp window
    (default (-0.3,-0.15) s before the corrected lick) and a clean pre-trial baseline
    (default (-1.75,-1.25) s) on a LICK-aligned tensor (t=0 = 200 ms-corrected lick).
    Class 1 = pre-movement. Windows are the lab-canonical EVENT_RESPONSIVENESS_WINDOWS
    (ported MATLAB def; matches lick.py) — imported, not invented."""
    def _feat(win):
        m = (bin_centers >= win[0]) & (bin_centers < win[1])
        return z_lick[:, m, :].mean(axis=1)
    pre, premove = _feat(base_window), _feat(premove_window)
    X = np.vstack([pre, premove])
    y = np.r_[np.zeros(len(pre)), np.ones(len(premove))]
    return compute_lda_cd(X, y, method="sklearn", reg=1.0, reg_style="flat")
```

- [ ] **Step 4: Run** `... -k "project_out or motor_axis" -v` → PASS.
- [ ] **Step 5: Commit** `feat(N1): fresh lick/motor coding direction + projection + movement-axis signal`.

---

### Task 4: Per-session response-time decoder + cohort aggregation + nulls

**Files:** Modify `neural_latents.py`; Test `tests/analysis/test_neural_latents.py`.

> **Decode scheme (LOCKED by adversarial review — do not deviate):** units are **NOT cross-session tracked** (`good_and_stable_ids` are *within-session* QC — the `get_good_cluster_ids` "UnitMatch-tracked" docstring is documented-wrong; see `memory/good_and_stable_ids_definition.md`). So column *u* in session A ≠ column *u* in session B. Therefore: **decode WITHIN each session** (`StratifiedKFold` over that session's trials on quantile-binned `decision_time`), producing ONE out-of-fold Spearman `r_s` per session; the cohort statistic is the **mean/median of {r_s} with a bootstrap CI OVER SESSIONS** (session = unit of replication). **REJECTED: `GroupKFold`-across-sessions** (held-out session's columns are different neurons → meaningless) **and the concatenated block-diagonal variant** (held-out columns all-zero in training) **and any single global Spearman on pooled OOF** (Simpson's-paradox inflation from between-session offsets). Precedent to mirror: `analysis_suite/04_decoding/c_state_decoding.py::decode_state_session` (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` within session).

**Interfaces:**
- Produces:
  - `within_type_graded(y_pred, y_true, trial_type) -> dict[type]->spearman` — Spearman of pred-vs-true **within each trial type** for ONE session (NOTE B; the headline reads `["hit"]`). Min 5 trials/type.
  - `decode_session(X, y, *, n_splits=5, seed=42) -> dict(r, r2, y_pred)` — within ONE session: quantile-binned `StratifiedKFold`, `RidgeCV` per fold, OOF Spearman `r` (0.0 if the prediction is constant/degenerate).
  - `decode_cohort(sessions, *, n_null=200, seed=42) -> dict` — `sessions` = list of `(sess_id, X, y, trial_type)`. Returns `per_session` (list of `dict(sess_id, r, n, within)`), `mean_r`, `median_r`, `ci` (bootstrap over the `r_s`), `null_mean`, `null_sd` (within-session-shuffle null, aggregated identically over sessions), `within_type` (per-type mean `r_s` across sessions). **No `groups` arg, no pooled global Spearman anywhere.**
- Consumes: `scipy.stats.spearmanr`, `sklearn.linear_model.RidgeCV`, `sklearn.model_selection.StratifiedKFold`, `pandas.qcut`, `bootstrap_ci` (`visdetect.analysis.utils`).

- [ ] **Step 1: Write the failing test**

```python
def _sessions_with_within_signal(K=6, n=80, p=6, seed=1):
    rng = np.random.default_rng(seed); out = []
    for s in range(K):
        w = rng.normal(size=p); y = rng.normal(size=n)
        X = y[:, None] * w[None, :] + rng.normal(scale=0.5, size=(n, p))  # within-session signal
        tt = np.where(rng.random(n) < 0.5, "hit", "fa")
        out.append((s, X, y, tt))
    return out

def _sessions_simpson_only(K=6, n=80, p=6, seed=2):
    """NO within-session signal; only between-session y-offsets + disjoint feature
    blocks encoding the offset. A pooled global Spearman is spuriously high; the
    correct per-session aggregate is near chance."""
    rng = np.random.default_rng(seed); out = []
    for s in range(K):
        offset = 5.0 * s
        y = offset + rng.normal(scale=0.3, size=n)                 # ~constant within session
        X = np.zeros((n, p * K))
        X[:, s*p:(s+1)*p] = offset + rng.normal(scale=0.3, size=(n, p))  # disjoint block ~ offset
        tt = np.where(rng.random(n) < 0.5, "hit", "fa")
        out.append((s, X, y, tt))
    return out

def test_decode_cohort_recovers_within_session_signal():
    res = nl.decode_cohort(_sessions_with_within_signal(), n_null=100, seed=42)
    assert len(res["per_session"]) == 6                       # ONE r per session
    assert res["mean_r"] > 0.5                                # within-session signal recovered
    assert res["mean_r"] > res["null_mean"] + 2 * res["null_sd"]
    assert res["within_type"]["hit"] > 0.3                    # NOTE B: graded within hits

def test_per_session_scheme_defeats_simpson_inflation():
    sessions = _sessions_simpson_only()
    res = nl.decode_cohort(sessions, n_null=50, seed=42)
    assert len(res["per_session"]) == 6
    assert abs(res["mean_r"]) < 0.2                           # CORRECT aggregate ~ chance
    # REGRESSION GUARD: the REJECTED pooled-global-Spearman scheme is spuriously high
    from scipy.stats import spearmanr
    yp = [nl.decode_session(X, y, seed=42)["y_pred"] for _, X, y, _ in sessions]
    yt = [y for _, _, y, _ in sessions]
    r_global = spearmanr(np.concatenate(yp), np.concatenate(yt)).correlation
    assert r_global > 0.5                                     # between-session inflation, as feared

def test_within_type_graded_separates_types():
    y_true = np.array([1., 2., 3., 10., 11., 12.])
    y_pred = np.array([1.1, 2.0, 2.9, 10.2, 10.9, 12.1])
    tt = np.array(["fa", "fa", "fa", "hit", "hit", "hit"])
    g = nl.within_type_graded(y_pred, y_true, tt)
    assert g["hit"] > 0.8 and g["fa"] > 0.8                   # graded WITHIN each type
```

- [ ] **Step 2: Run** `... -k "decode_cohort or simpson or within_type" -v` → FAIL.

- [ ] **Step 3: Implement**

```python
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from visdetect.analysis.utils import bootstrap_ci

def within_type_graded(y_pred, y_true, trial_type):
    y_pred, y_true, tt = map(np.asarray, (y_pred, y_true, trial_type))
    res = {}
    for t in np.unique(tt):
        m = tt == t
        if m.sum() >= 5 and np.std(y_pred[m]) > 1e-9:
            r = spearmanr(y_pred[m], y_true[m]).correlation
            res[str(t)] = float(r) if np.isfinite(r) else 0.0
    return res

def decode_session(X, y, *, n_splits=5, seed=42):
    """Within ONE session: quantile-binned StratifiedKFold over trials, RidgeCV."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n = len(y); k = max(2, min(n_splits, n // 2))
    nb = max(2, min(k, n // 10))
    ybin = pd.qcut(y, nb, labels=False, duplicates="drop")
    if len(np.unique(ybin)) < 2:                 # y too degenerate to stratify
        ybin = (y > np.median(y)).astype(int)
    y_pred = np.full(n, np.nan)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, ybin):
        y_pred[te] = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X[tr], y[tr]).predict(X[te])
    if np.std(y_pred) < 1e-9:
        r = 0.0
    else:
        r = spearmanr(y_pred, y).correlation
        r = 0.0 if not np.isfinite(r) else float(r)
    ss_res = np.sum((y - y_pred) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    return {"r": r, "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0, "y_pred": y_pred}

def _per_session_rs(sessions, seed):
    return np.array([decode_session(X, y, seed=seed)["r"] for _, X, y, _ in sessions])

def decode_cohort(sessions, *, n_null=200, seed=42):
    per = []
    for sid, X, y, tt in sessions:
        d = decode_session(X, y, seed=seed)
        per.append({"sess_id": sid, "r": d["r"], "n": int(len(y)),
                    "within": within_type_graded(d["y_pred"], y, tt)})
    rs = np.array([p["r"] for p in per])
    ci_lo, ci_hi = bootstrap_ci(rs, n_bootstrap=1000, seed=seed)
    wt = {}
    for t in ("hit", "fa"):
        vals = [p["within"][t] for p in per if t in p["within"]]
        if vals:
            wt[t] = float(np.mean(vals))
    rng = np.random.default_rng(seed)
    null = np.empty(n_null)
    for i in range(n_null):                       # within-session shuffle of y, aggregate as mean r_s
        shuff = [(sid, X, rng.permutation(y), tt) for sid, X, y, tt in sessions]
        null[i] = float(np.nanmean(_per_session_rs(shuff, seed)))
    return {"per_session": per, "mean_r": float(np.nanmean(rs)),
            "median_r": float(np.nanmedian(rs)), "ci": (float(ci_lo), float(ci_hi)),
            "null_mean": float(null.mean()), "null_sd": float(null.std()), "within_type": wt}
```

- [ ] **Step 4: Run** `... -k "decode_cohort or simpson or within_type" -v` → PASS (incl. the Simpson regression guard).
- [ ] **Step 5: Commit** `feat(N1): per-session response-time decoder + cohort aggregation over sessions + within-session shuffle null (defeats Simpson inflation)`.

---

### Task 5: Synthetic recovery + φ-vs-ramp discriminability on the readout window (NOTE A prerequisite)

**Files:** Create `scripts/neural_latents/_synthetic_recovery.py`; Modify `neural_latents.py` (φ/ramp bases); Test `tests/analysis/test_neural_latents.py`.

**Interfaces:**
- Produces: `phi_ramp_bases(t, mu, sigma=0.8) -> dict("phi"=gaussian bump at mu, "ramp"=linear monotonic)`; `phi_specificity_session(Xt, y, t, mu, sigma=0.8, *, seed=42) -> dict(r_phi, r_ramp, delta)` — within ONE session, compares `decode_session` on a φ-weighted vs ramp-weighted temporal collapse of the readout window (per-session ΔCV r; Task 7 aggregates `delta` across sessions with a bootstrap CI over sessions). `_synthetic_recovery.py` simulates (a) a φ-urgency ramp population and (b) a pure-motor ramp **per session**, over the REAL μ range and `decision_time` distribution restricted to the **pre-μ readout window**, and checks: per-session decode recovers timing in both; the **per-session** motor-CD projection KILLS the pure-motor case but spares φ-urgency; φ-vs-ramp separable **on the readout window** (or reports "underpowered").

- [ ] **Step 1: Write the failing test** (discriminability behaves correctly on synthetic)

```python
def test_phi_ramp_bases_shapes():
    # Use the REAL late readout window (nl.WINDOWS["late"] = (4,6), the latest pre-μ
    # window / PRIMARY candidate). NOTE-A collinearity is window-dependent: negligible
    # far below μ (early/mid are near-flat φ tails) and strong only near μ. Over [0,6]
    # (a 7σ tail at σ=0.8) corr is only ~0.60; over [4,6] it is ~0.89.
    t = np.linspace(4.0, 6.0, 120)
    b = nl.phi_ramp_bases(t, mu=7.0, sigma=0.8)
    assert b["phi"].argmax() == len(t) - 1          # rising flank: phi peaks toward mu>6
    assert np.all(np.diff(b["ramp"]) > 0)           # ramp strictly monotonic
    # over the latest pre-μ readout window φ's rising flank is highly collinear with a
    # monotonic ramp (NOTE A) -> φ-specificity is expected underpowered there
    assert np.corrcoef(b["phi"], b["ramp"])[0, 1] > 0.85
```

- [ ] **Step 2: Run** `... -k phi_ramp_bases -v` → FAIL.

- [ ] **Step 3: Implement** `phi_ramp_bases` + `phi_specificity_session` in `neural_latents.py`

```python
def phi_ramp_bases(t, mu, sigma=0.8):
    t = np.asarray(t, float)
    phi = np.exp(-0.5 * ((t - mu) / sigma) ** 2)
    ramp = (t - t.min()) / (t.max() - t.min() + 1e-12)
    return {"phi": phi, "ramp": ramp}

def phi_specificity_session(Xt, y, t, mu, sigma=0.8, *, seed=42):
    """Xt: (n_trials, n_bins, n_units) for ONE session. Compare within-session
    decode (decode_session) using a phi-weighted vs ramp-weighted temporal collapse
    of the readout window. Returns the per-session delta CV r."""
    b = phi_ramp_bases(t, mu, sigma)
    def _r(weight):
        w = weight / (weight.sum() + 1e-12)
        Xw = np.tensordot(Xt, w, axes=([1], [0]))    # (n_trials, n_units)
        return decode_session(Xw, y, seed=seed)["r"]
    r_phi, r_ramp = _r(b["phi"]), _r(b["ramp"])
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
# ... simulate K synthetic SESSIONS per spec; reuse nl.decode_cohort /
#     decode_session / project_out_axis / fit_lick_motor_cd / phi_specificity_session;
#     real mu range 6.7-7.5; readout windows from nl.WINDOWS; per-session motor-CD +
#     projection; write verdict JSON {recovers, motor_killed, phi_separable_on_window}
#     + figure. (Full body authored at execution.)
```

*(The synthetic body is mechanical given the library functions above; it builds per-session simulations, asserts the three conditions via the per-session `nl.*` API, and writes the verdict. No new science.)*

- [ ] **Step 6: Run the harness** `PYTHONPATH=src .venv/Scripts/python.exe scripts/neural_latents/_synthetic_recovery.py`
Expected: prints `recovers=True motor_killed=True phi_separable_on_window=<bool>`; writes figure + `data/cache/neural_latents/n1_synthetic_verdict.json`.

- [ ] **Step 7: Commit** `feat(N1): synthetic recovery + NOTE-A phi-vs-ramp discriminability prerequisite`.

---

### Task 6: C1 real-data gate — per-session decode, window sweep, per-session motor-CD survival, within-hit graded

**Files:** Create `scripts/neural_latents/n1_c1_gate.py`.

**Interfaces:**
- Produces (script): for each swept window, builds per expert session (sequential pkl load, `del sess; gc.collect()`): the windowed feature matrix `X_s` (trials × that session's units, z-scored in `join_session`), target `y_s = decision_time` on **lick trials (hit+fa)** (relative to Baseline_ON), `trial_type_s`, and a **fresh per-session preparatory-motor axis** from a lick-aligned tensor (`build_population_tensor(event_name="FA")` and `"Hit"`, aligned to the **200 ms-corrected** lick via `compute_true_reaction_time`, span `(-2.0, 0.75)`, late-licks ≥3 s after Baseline_ON; `fit_lick_motor_cd`). Then the pure core `evaluate_window(sessions, motor_axes)` runs the **per-session** `decode_cohort` (prong 1 vs within-session-shuffle null), the **per-session** motor-CD projection (prong 2: aggregate per-session r on projected features beats the base null), and `within_type` (NOTE-B within-hit). Selection rule: PRIMARY = the **latest window in which the per-session `motor_axis_signal` is not significantly above its pre-trial baseline** (movement-free). Writes `n1_c1_gate.json` (per-window mean_r, CI, null mean±2SD, mean_r_after_projection, survives_projection, within_type["hit"]) + `fig_n1_c1_gate.png` + `n1_c1_gate_stats.csv`.

**Prong-2 pass criterion + caveats (do not over-read Task 5's synthetic):** prong 2 passes iff **`mean_r_after_projection` still beats the base null** (`> null_mean + 2·null_sd`) — NOT "`r` unchanged after projection." The Task-5 synthetic spared φ-urgency fully only because that signal was built **perfectly orthogonal** to the motor axis; **real** urgency will be partly aligned, so expect the motor-CD projection to drop the real timing decode **somewhat** — that is fine as long as enough survives to beat the null. The synthetic also used **tuned SNR** (it proves the control *can* dissociate orthogonal-vs-aligned signals; it does NOT predict real-data magnitude). Task 6 on real data is the actual test of how much survives.

- [ ] **Step 1: Write the failing test**

```python
def test_evaluate_window_per_session_gate():
    from scripts.neural_latents import n1_c1_gate as gate
    sessions = _sessions_with_within_signal()         # Task-4 generator: per-session signal
    motor_axes = {s: np.eye(X.shape[1])[0] for s, X, y, tt in sessions}  # remove 1 dim only
    res = gate.evaluate_window(sessions, motor_axes, n_null=50, seed=42)
    assert res["mean_r"] > res["null_mean"] + 2 * res["null_sd"]         # prong 1
    assert isinstance(res["survives_projection"], bool)                 # prong 2 computed
    assert res["mean_r_after_projection"] > 0.3                         # signal spread > 1 dim survives
    assert res["within_type"]["hit"] > 0.0                              # NOTE B
```

- [ ] **Step 2: Run** `... -k evaluate_window_per_session -v` → FAIL (module missing).

- [ ] **Step 3: Implement `n1_c1_gate.py`** — pure core + `main()`. Core:

```python
import numpy as np
from visdetect.analysis import neural_latents as nl

def evaluate_window(sessions, motor_axes, *, n_null=200, seed=42):
    """sessions: list of (sess_id, X, y, trial_type); motor_axes: {sess_id: axis}.
    Per-session decode + within-session-shuffle null (prong 1) + per-session motor-CD
    projection survival (prong 2). NO cross-session pooling, NO global Spearman."""
    base = nl.decode_cohort(sessions, n_null=n_null, seed=seed)
    proj = [(sid, nl.project_out_axis(X, motor_axes[sid]), y, tt) for sid, X, y, tt in sessions]
    proj_mean = float(np.nanmean(nl._per_session_rs(proj, seed)))   # reuse, no extra null
    survives = proj_mean > base["null_mean"] + 2 * base["null_sd"]
    return {"mean_r": base["mean_r"], "median_r": base["median_r"], "ci": base["ci"],
            "null_mean": base["null_mean"], "null_sd": base["null_sd"],
            "mean_r_after_projection": proj_mean, "survives_projection": bool(survives),
            "within_type": base["within_type"], "per_session": base["per_session"]}
```

`main()`: `df = nl.load_latent_table(); sess_ids = nl.fitted_expert_sessions(df)`. Per window in `nl.WINDOWS`, per session (sequential): `s = load_session(int(sid))`; `jr = nl.join_session(s, df[df.sess_canon == sid], window=(-1.3, 6.0))`; mask `jr.y.outcome.isin({"hit","fa"})`; `X_s = nl.window_feature_matrix(jr.z[mask], jr.bin_centers, win)`, `y_s = jr.y.decision_time[mask].values`, `tt_s = jr.y.outcome[mask].values`. Build the lick-aligned z tensor (`event_name` FA then Hit, span `(-2.0,0.75)`, corrected-lick) → `axis_s = nl.fit_lick_motor_cd(z_lick, lick_bin_centers)`; also record per-session `motor_axis_signal(X_s, axis_s)` vs the pre-trial-baseline projection for the movement-free check. `del s; gc.collect()`. Then `res = evaluate_window(sessions, motor_axes)`; pick PRIMARY = latest movement-free window. Dump JSON + figure (per-window mean_r ± CI with the null band; within-hit panel). Plain-language title + stats CSV.

- [ ] **Step 4: Run** `... -k evaluate_window_per_session -v` → PASS; then `PYTHONPATH=src .venv/Scripts/python.exe scripts/neural_latents/n1_c1_gate.py` → JSON + figure + stats CSV; prints per-window verdict + PRIMARY.

- [ ] **Step 5: Commit** `feat(N1): C1 gate — per-session decode, window sweep, per-session motor-CD survival, within-hit graded`.

---

### Task 6b: Within-FA timing — leakage filter FIRST, then movement-MATCHING (decisive prong-2)

**Files:** Create `scripts/neural_latents/n1_c1b_within_fa.py`; add pure cores to `neural_latents.py`; append tests. Gates Tasks 7–9 (do not proceed until this answer is in).

**Why (from the C1 real run + review):** the real urgency signal is **self-timed (FA) timing**, NOT hit timing — within-hit≈0.05 is a **clean, leakage-free NEGATIVE CONTROL** (hits always lick >6 s, after every readout window) that *validates* the decode, not a failure. But the raw within-FA r (0.34–0.56) is **partly circular**: 15–28% of FA trials lick INSIDE the readout window (early 26.8%, mid 15.0%, late 28.0%; up to ~74% leak-prone), so the decoder reads `decision_time` off peri/post-lick activity. Prong 2 is decided by whether the **leakage-filtered** within-FA decode **survives MOVEMENT-MATCHING**. **Headline reframe (locked): "pre-change striatal ramp predicts self-timed (anticipatory FA) lick timing."**

**Order is mandatory: leakage filter FIRST, then matching.**

**Pure cores (in `neural_latents.py`):**
- `leakage_free_mask(decision_time, window, guard=0.25) -> np.ndarray[bool]` — True where `decision_time >= window[1] + guard` (lick comfortably AFTER the readout window).
- `partial_spearman(a, b, control) -> float` — rank-partial: Spearman of the residuals of `rank(a)~rank(control)` and `rank(b)~rank(control)` (the continuous analog of movement-matching).
- `within_strata_spearman(y_pred, y, control, *, n_strata=4) -> float` — bin trials into `n_strata` quantiles of `control` (motor signal), Spearman(`y_pred`,`y`) within each stratum, return the trial-count-weighted mean (movement held ~constant within stratum). 
- `motor_subspace(z_lick, bin_centers, *, k=5, premove_window=_FA_PRE, base_window=_FA_BASE) -> np.ndarray[n_units,k]` — orthonormal top-k motor directions (PCA of the per-trial peri-vs-pre-lick difference vectors); `project_out_subspace(X, basis) -> X` removes all k dims.

**Script `n1_c1b_within_fa.py` — the cascade, per swept window, FA trials only, per session aggregated over sessions (decode_cohort-style), with within-session-shuffle nulls + bootstrap CIs over sessions, `n_workers>1`:**
1. **raw** within-FA r (all FA trials).
2. **leakage-filtered** within-FA r (`leakage_free_mask`, guard 0.25 s) — *expect a drop*; report how many FA trials/session survive.
3. **MATCHED (primary)** on the leakage-filtered set: decode once per session (OOF `y_pred` from `decode_session`), then `within_strata_spearman(y_pred, y, motor_signal)` (movement matched) AND `partial_spearman(y_pred, y, motor_signal)` (complement); aggregate over sessions vs null.
4. **subspace-projected (secondary)**: `project_out_subspace` (top-k motor PCs) then re-decode within-FA; report r (caveat: high-dim projection can remove genuine signal — secondary, not decisive).
- **Lead with the EARLY window** (best raw signal 0.56, least movement 68%, most clean trials ~64%; early activity predicting a lick seconds later is itself evidence against pure peri-movement motor-prep, which peaks near the lick). Report all three.
- Outputs: `data/cache/neural_latents/n1_c1b_within_fa.json` (the full cascade r raw→filtered→matched→subspace per window, n_clean per session, nulls, CIs) + `fig_n1_c1b_within_fa.png` (plain-language: the cascade as a descending bar chart per window, lead-early) + stats CSV.

**Honest framing of the outcome (state in the JSON `verdict` + figure caption):**
- survives leakage-filter + matching → **"self-timed urgency predicts FA timing beyond generic motor prep."**
- collapses → **"for self-timed licks the urgency/commitment ramp and motor preparation are not separable"** — a REAL, meaningful basal-ganglia action-commitment conclusion, NOT a control failure (this earns the Option-2 reframe).

- [ ] **Step 1:** Tests (pure cores, synthetic): `test_leakage_free_mask` (only trials with `decision_time >= hi+guard` kept); `test_partial_spearman_controls` (a planted a–b correlation that is entirely mediated by `control` → partial≈0; an independent one survives); `test_within_strata_spearman` (signal present within strata where `control` is held constant → high; a control-driven-only signal → ~0); `test_motor_subspace_projection` (`project_out_subspace(X, basis) @ basis ≈ 0`, orthogonal variance preserved).
- [ ] **Step 2:** Run `... -k "leakage or partial_spearman or within_strata or motor_subspace" -v` → FAIL.
- [ ] **Step 3:** Implement the pure cores + `n1_c1b_within_fa.py` main() (FA-only per-session build reusing Task-6 wiring; `n_workers=max(1,cpu-2)`). Whole test file still passes.
- [ ] **Step 4:** Run tests → PASS. **The controller runs the real-data script in the background** (avoids the long-run abandonment that hit Task 6); inspect the cascade JSON.
- [ ] **Step 5:** Commit `feat(N1): within-FA timing — leakage-filtered + movement-matched (decisive prong-2)`.

---

### Task 6c: Ramp-SLOPE readout — the ONE pre-specified ramp-appropriate check (LAST readout)

**Why (user, post-6b null):** Task-6b tested the MEAN-window feature — a weak proxy for the ramp-to-threshold hypothesis, whose decision-relevant content is the **SLOPE** (rate of rise → time-to-threshold). A null on the mean is NOT a null on the slope; concluding "no urgency ramp" without a ramp readout is a reviewer-obvious validity gap. **This is ONE pre-specified additional readout** (the last). **Discipline: if it is also null → finalize the controlled negative (NO further readouts — that is fishing); if it revives → treat with suspicion (scrutiny/replication, not a headline).**

**Pure core (`neural_latents.py`):** `ramp_slope_feature_matrix(z, bin_centers, win) -> (n_trials, n_units)` — per (trial, unit), the OLS **slope** of `z` vs time over the window's bin centers (vectorized: `cov(t, z)/var(t)` across the in-window bins); raises `ValueError` if < 2 bins in the window.

**Script `scripts/neural_latents/n1_c1c_ramp_slope.py`:** the SAME cascade as Task-6b (raw → leakage-filtered [`decision_time ≥ window_hi+0.25`] → MATCHED [`partial_spearman` primary + `within_strata_spearman`; multi-dim subspace secondary] → per-session aggregate → bootstrap-over-sessions; lead EARLY; FA-only), but the decode FEATURE is the per-unit **ramp slope** (`ramp_slope_feature_matrix`) instead of the mean. **Reuse Task-6b's cascade core** (`evaluate_window_within_fa` / the per-session build in `n1_c1b_within_fa.py`) — only the feature builder changes; the movement-matching control stays `motor_axis_signal` (movement magnitude). Optional SNR variant noted in the docstring (slope of the projection onto a TRAIN-FOLD timing CD) — do NOT implement (would add a circularity surface; the per-unit slope is the clean check). Outputs `n1_c1c_ramp_slope.json` + `fig_n1_c1c_ramp_slope.png` + stats CSV; `verdict` states null-or-revive per the same partial-Spearman-beats-its-null + bootstrap-CI rule.

- [ ] **Step 1:** Test `test_ramp_slope_feature_matrix`: a planted linear ramp `z = a + b·t` → recovered slope ≈ `b` per unit; a flat trial → ≈0; `ValueError` on < 2 in-window bins.
- [ ] **Step 2:** Run `... -k ramp_slope -v` → FAIL.
- [ ] **Step 3:** Implement the core + `n1_c1c_ramp_slope.py` (reuse 6b cascade). Whole test file still passes. **Do NOT run the heavy real-data job** (controller runs it).
- [ ] **Step 4:** Controller runs the real-data script in background; read the cascade.
- [ ] **Step 5:** Commit `feat(N1): ramp-slope readout — the one pre-specified ramp-appropriate within-FA check`.

---

### Task 7: φ-specificity on the near-μ window (MINIMAL, expect-null, reported as a test-limitation; decoupled, non-gating)

**Files:** Create `scripts/neural_latents/n1_phi_specificity.py`.

**Why minimal + expect-null (Task-5-synthetic-confirmed, 48796e4 — supersedes the earlier "modestly testable"):** within-window φ-weighting is **underpowered BY CONSTRUCTION**, not for lack of an urgency signal. On the late near-μ window [4,6] the **φ-weight profile and the ramp-weight profile are ~0.89 collinear**, so weighting the readout by φ vs by a ramp yields nearly the same feature → nearly the same decode → ΔCV≈0. The Task-5 synthetic proved this: with a genuine φ-urgency signal built in (PHI_AMP=2.0, orthogonal axis), ΔCV was still **+0.013, CI [−0.004,+0.033] (spans 0)**. The test's power comes from the *difference* between the two weighting shapes, which is small over any achievable pre-μ window (the early window is worse — φ is the ~1e-7 deep tail → ill-conditioned). So φ-specificity is run **once, minimally, on the late near-μ window** (decoupled from C1's PRIMARY), with a **null EXPECTED**, **non-gating** (FIX 1), and **reported as a test-limitation, NOT a science claim**:
- ✅ "the within-window φ-weighting cannot separate the anticipation shape from a generic ramp over the achievable readout window (≈0.89 collinear weighting profiles; synthetic-confirmed underpowered even with signal present)"
- ❌ NOT "we tested for temporal-expectation coding and found none."
The **headline is prongs 1+2** (the validated real result: urgency ramp predicts response timing beyond motor prep). φ-specificity is a reported negative, not a hoped-for positive. Do not over-invest.

**Interfaces:** Pure core `phi_specificity_verdict(sessions, bin_centers, mu_by_session, *, window_is_motor_free, phi_min_max=0.05, collinearity_thresh=0.85, sigma=0.8, seed=42) -> dict` (`sessions` = list of `(sess_id, Xt_window, y)`, the near-μ window block). Logic:
1. **φ conditioning:** `phi_max = max over sessions of phi_ramp_bases(bin_centers, mu_s, sigma)["phi"].max()`. If `phi_max < phi_min_max` → `{"verdict": "not_testable_ill_conditioned", "phi_max": ...}` (deep tail).
2. **Movement:** else if `not window_is_motor_free` → `{"verdict": "not_testable_on_movement_free_window", "phi_max": ...}`.
3. **Else** compute the weighting-profile collinearity `profile_corr = corr(phi_profile, ramp_profile)` over the window (representative μ = median), AND the per-session `delta_s = phi_specificity_session(...)["delta"]` with a bootstrap mean Δ + CI. If `profile_corr >= collinearity_thresh` (the realistic case) → `{"verdict": "underpowered_by_construction", "delta_mean": ..., "ci": [...], "profile_corr": ..., "phi_max": ..., "note": "<the ✅ test-limitation sentence>"}`. Only if `profile_corr < collinearity_thresh` (not expected on achievable windows) → `{"verdict": "testable", "delta_mean": ..., "ci": [...], ...}`.

`main()`: select the late `nl.WINDOWS["late"]` near-μ window, read `n1_c1_gate.json` for `window_is_motor_free`, build per-session near-μ blocks via `join_session` + slice, call `phi_specificity_verdict`, write `n1_phi_specificity.json` + `fig_n1_phi_specificity.png` (plain-language title stating the case + the test-limitation framing). **Only meaningfully-stronger alternative (note in docstring; DO NOT implement unless prongs 1+2 land and a reviewer asks):** μ-anchoring across sessions — does each session's neural ramp shift its timing to match that session's `expected_change_time`? Also power-limited (μ range ≈ 0.8 s over 29 experts).

- [ ] **Step 1:** Test `test_phi_specificity_window_conditioning` (pure-core, synthetic): (a) deep-tail window (`bin_centers` far below μ → `phi_max ~1e-7`) → `"not_testable_ill_conditioned"`, no exception; (b) near-μ window (`bin_centers` in [4,6], μ=7, profiles ~0.89 collinear) with `window_is_motor_free=True` → `"underpowered_by_construction"` with a finite `delta_mean` and `profile_corr >= 0.85`; (c) same near-μ window but `window_is_motor_free=False` → `"not_testable_on_movement_free_window"`.
- [ ] **Step 2:** Run `... -k phi_specificity_window_conditioning -v` → FAIL.
- [ ] **Step 3:** Implement `phi_specificity_verdict` (pure) + minimal `main()` wiring (reads C1 gate JSON; uses `phi_ramp_bases`/`phi_specificity_session` from Task 5). Whole test file still passes.
- [ ] **Step 4:** Run test → PASS; run on real data (`PYTHONPATH=src .venv/Scripts/python.exe scripts/neural_latents/n1_phi_specificity.py`).
- [ ] **Step 5:** Commit `feat(N1): phi-specificity (minimal, expect-null, reported as test-limitation; window-conditioned; decoupled from C1)`.

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
- [ ] **Step 3:** Implement `u_graded_test` (residualize on a mood dummy via `numpy.linalg.lstsq`, Spearman of residuals, bootstrap). Implement `n1_c3_u_graded.py` (build per-cell ramp amplitude = mean PRIMARY-window decode projection per session×mood) and the C4 descriptive evidence-axis strand (per-session `decode_cohort` of `evidence_integral_at_decision`, captioned descriptive — sharpness/evidence is non-load-bearing). Implement `run_n1.py` orchestration with the gate guard + the secondary d′ gradient (`spearmanr` of each session's `per_session` `r` vs its manifest `d_prime`).
- [ ] **Step 4:** Run tests → PASS; run `run_n1.py` end-to-end → all figures + `n1_results.json`.
- [ ] **Step 5:** Commit `feat(N1): C3 u-graded (mood-controlled) + C4 descriptive evidence axis + run_n1 orchestrator`.

---

## Self-Review

**Spec coverage:** §2 gate → T6 (prongs 1+2); §6 φ-specificity + NOTE-A prerequisite → T5 (synthetic) + T7 (exploratory, gated); §4 window sweep + Latimer-Huk fixed window → T2 + T6 selection rule; §5 movement controls → T3 + T6 (projection survival, motor-free selection, never partial RT); NOTE B within-hit graded → T4 + T6; §7-C1 → T6; C2 SPN/FSI (D1/D2 deferred) → T8; C3 u-graded mood-controlled → T9; C4 descriptive evidence → T9; secondary d′ gradient → T9; §8 synthetic recovery → T5; join/canonicalization → T1; §9 join-integrity test → T1; determinism/no-X:/n_workers → Global Constraints + sequential pkl loop. caution-z is correctly ABSENT (moved to sibling). **No gaps.**

**Placeholder scan:** the `_synthetic_recovery.py` body and the figure-drawing inside scripts are described as "authored at execution" but every step names the exact `nl.*` functions to call and the exact verdict/outputs — no "TBD/add error handling." The pure cores (`evaluate_window`, `decode_session`/`decode_cohort`, `phi_specificity_session`, `single_unit_timing_glm`, `u_graded_test`, the join, the null) are fully coded and tested.

**Type consistency:** `decode_session` → `dict(r,r2,y_pred)`; `decode_cohort(sessions)` → `dict(per_session, mean_r, median_r, ci, null_mean, null_sd, within_type)` — consumed identically in T5/T6/T7/T9; `phi_specificity_session` → `dict(r_phi,r_ramp,delta)`. No `decode_response_timing`/`shuffle_null`/`groups` survive (removed in the per-session rework). `join_session` returns `JoinResult(z,bin_centers,y,unit_ids,kept_trials)` — `y` columns are the single `_Y_COLS` list, consumed by T6/T8/T9. `WINDOWS` keys `early/mid/late` consistent T2→T6→T7. `fit_lick_motor_cd(base_window,premove_window)` uses imported `EVENT_RESPONSIVENESS_WINDOWS` (T3→T6). `celltype` values `{SPN,FSI}` consistent T8.

---

## Choices (RESOLVED in adversarial review)

1. **Decode scheme (was Choice 1):** **per-session decode, aggregate `r_s` over sessions, bootstrap CI over sessions** (Task 4). `GroupKFold`-across-sessions and the concatenated block-diagonal are both **REJECTED** — units are within-session QC, not cross-session tracked, so cross-session columns are different neurons. The within-session-shuffle null + the Simpson regression-guard test enforce this. Locked.
2. **Motor events / windows (was Choice 2):** motor CD from **FA + Hit** lick-aligned tensors with the **canonical** `EVENT_RESPONSIVENESS_WINDOWS` (baseline `(-1.75,-1.25)`, pre-movement `(-0.3,-0.15)`), span **`(-2.0, 0.75)`**, **200 ms-corrected** lick (`compute_true_reaction_time`), late-licks **≥3 s** after Baseline_ON. Locked. Optional refinement (note in code): drop very-early FAs whose `-1.75 s` baseline would precede Baseline_ON.

## Minor notes (non-blocking; carried from review)

- **Deliverable is a strict subset** of non-abort/ref trials (B8 applies an extra usability filter; ~16 more trials dropped on `15092025`). The join keeps the intersection of `valid_trials` and the latent `trial_idx` set — correct; just documented.
- **Triple-check uniqueness** leans on `change_time` (`atol=1e-6`). If `change_time_planned` is ever NaN it falls back to `outcome`+`change_size` (~25% porous); all finite in the current data. Add a `warnings.warn` if a NaN-`change_time` row coincides with a non-unique `(outcome,change_size)` so the guard never silently weakens.
- **`load_staging_manifest` missing-manifest fallback** advertises `dprime` not `d_prime`; irrelevant for BG_046 (has a real manifest with `d_prime`) — read `d_prime`, don't hardcode the fallback name.
- **`load_session(int(sid))` is safe** — `_session_pkl_candidates` zfill(8)s the id back, so the leading-zero-day bug does not bite the loader (verified in review). The canonicalization rule still applies to the *latent-table join* keys.
