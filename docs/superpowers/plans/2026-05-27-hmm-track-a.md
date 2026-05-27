# GLM-HMM Track A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Track A of the GLM-HMM audit (spec `docs/superpowers/specs/2026-05-27-hmm-glm-audit-design.md`) on the BG_046 test bed: six findings (F1, F3, F4, F14, F22, F25) that together bring the model to a defensible, Ashwood-aligned, externally-validated state for single-subject publication.

**Architecture:** Modify `src/visdetect/analysis/hmm.py` (additive helpers + selection-logic changes), add `src/visdetect/analysis/hmm_validation.py` for external state validation, add one figure script for F22. All changes preserve the existing API; behavior changes are opt-in via new parameters with backward-compatible defaults where reasonable.

**Tech Stack:** Python 3, numpy, scipy, pandas, matplotlib, pytest. Invoke Python with `py` (Windows convention for this project).

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `src/visdetect/analysis/hmm.py` | Modify | F4 docstring; F14 confidence helper; F25 explicit auto-label; F1 CV-based selection; F3 lapse-fit helper |
| `src/visdetect/analysis/hmm_downstream.py` | Modify (small) | Surface `cv_ll_bits_per_trial` in API |
| `src/visdetect/analysis/hmm_validation.py` | Create | F22 external observable computations |
| `tests/test_hmm_confidence.py` | Create | F14 tests |
| `tests/test_hmm_labels_explicit.py` | Create | F25 tests |
| `tests/test_hmm_cv_selection.py` | Create | F1 tests |
| `tests/test_hmm_lapse.py` | Create | F3 tests |
| `tests/test_hmm_validation.py` | Create | F22 tests |
| `scripts/analysis/behavior/hmm_external_validation.py` | Create | F22 figure for BG_046 |
| `scripts/analysis/behavior/hmm_state_signature_diagnostic.py` | Create | F4 diagnostic (state placement in (p_catch, p_high) plane) |

**Branch.** Suggest a single feature branch `feature/hmm-track-a` with one commit per task.

---

## Task ordering rationale

1. **Task 1 (F4)** — quickest, no behavior change; sets the documented commitment that justifies F25's labeling criteria.
2. **Task 2 (F14)** — pure addition, no API change; foundation for F22.
3. **Task 3 (F25)** — adds a new labeling function and switches `decode_session` default; affects `auto_label_states` callers but the rank-based function stays available.
4. **Task 4 (F1)** — modifies `fit_best_model` selection logic; depends on `loso_cross_validation` already in `hmm_downstream.py`.
5. **Task 5 (F3)** — extends Task 4's `selection_df` with the lapse "L" row.
6. **Task 6 (F22)** — uses all of the above (CV-selected K, explicit labels, posterior thresholding) to produce the external-validation figure.

---

## Task 1: F4 — Document the `y = is_hit | is_fa` commitment

**Files:**
- Modify: `src/visdetect/analysis/hmm.py:94-177` (docstring of `prepare_session_data`)
- Create: `scripts/analysis/behavior/hmm_state_signature_diagnostic.py`

This task adds (a) a docstring paragraph explaining the encoding commitment with a forward reference to the spec §1.1, and (b) a small diagnostic script that plots the fitted states in the (P(lick|catch), P(lick|large-go)) plane — visual confirmation that the three states fall into the predicted regions.

- [ ] **Step 1.1: Replace `prepare_session_data` docstring**

In `src/visdetect/analysis/hmm.py`, find the existing docstring of `prepare_session_data` (line ~94-117) and replace it with:

```python
def prepare_session_data(
    session: Session,
    *,
    exclude_outcomes: Sequence[str] = ("abort", "ref"),
) -> Dict[str, Any]:
    """Extract binary choice vector *y* and covariate matrix *X* from a Session.

    Choice encoding (commitment, see specs/2026-05-27-hmm-glm-audit-design.md §1.1):
    ----------------------------------------------------------------------------
    ``y = is_hit | is_fa`` — the mouse "licked" if it produced ANY lick on the
    trial, whether a response-window lick after a real change (``is_hit``) or
    an early/impulsive lick before the change was presented (``is_fa``).

    This encoding is a scientific commitment, not a hyperparameter. The project's
    a priori three-state hypothesis (Impulsive / Stimulus-sensitive / Disengaged)
    requires the Impulsive state to be identifiable as a distinct cognitive
    regime — one in which the mouse licks regardless of stimulus. Treating fa
    as a no-lick observation would fold impulsive licking into the Disengaged
    state and collapse the K=3 structure to K=2.

    The alternative — ``y = is_hit`` only — is documented and rejected in F4 of
    the audit spec.

    Parameters
    ----------
    session : Session
        Loaded session object.
    exclude_outcomes : sequence of str
        Trial outcomes to discard (default: abort, ref).

    Returns
    -------
    dict with keys:
        y               : ndarray (T,)  binary choice (1 = lick, 0 = no-lick)
        X               : ndarray (T, D) design matrix
                          [bias, log2(change_size), prev_choice, prev_reward,
                          prev_early_lick]
        df              : DataFrame      trial-level metadata (filtered)
        session_name    : str
        feature_names   : list[str]
    """
```

- [ ] **Step 1.2: Create diagnostic script**

Create `scripts/analysis/behavior/hmm_state_signature_diagnostic.py`:

```python
"""Plot fitted GLM-HMM states in the (P(lick|catch), P(lick|large-go)) plane.

This is the F4 diagnostic figure: it confirms that K=3 fitted states fall
into the three predicted corners corresponding to the a priori state structure
(spec §1.1):
    - Impulsive:        high p(catch) AND high p(go)
    - Stim-sensitive:   low p(catch)  AND high p(go)
    - Disengaged:       low p(catch)  AND low p(go)

Usage
-----
    py scripts/analysis/behavior/hmm_state_signature_diagnostic.py \
        --model data/hmm/BG_046/best_model.pkl \
        --out FIGURES/behavior/BG_046/hmm/state_signature.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from visdetect.analysis.hmm import GLMHMM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path,
                    help="Path to fitted GLMHMM pickle")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output PNG path")
    ap.add_argument("--stim-high", type=float, default=2.0,
                    help="log2 of largest change_size used as 'high stim' (default 2.0 = 4x)")
    args = ap.parse_args()

    model = GLMHMM.load(args.model)
    K, D = model.n_states, model.n_features

    x_catch = np.zeros(D); x_catch[0] = 1.0
    x_high  = np.zeros(D); x_high[0]  = 1.0; x_high[1] = args.stim_high

    p_catch = np.array([float(expit(model.weights[k] @ x_catch)) for k in range(K)])
    p_high  = np.array([float(expit(model.weights[k] @ x_high))  for k in range(K)])

    fig, ax = plt.subplots(figsize=(5, 5))

    # Reference regions (a priori state predictions).
    ax.axhspan(0.5, 1.0, xmin=0.5, xmax=1.0, alpha=0.08, color="#d95f02",
               label="Impulsive region")
    ax.axhspan(0.5, 1.0, xmin=0.0, xmax=0.2, alpha=0.08, color="#1b9e77",
               label="Stim-sensitive region")
    ax.axhspan(0.0, 0.2, xmin=0.0, xmax=0.2, alpha=0.08, color="#7570b3",
               label="Disengaged region")

    palette = plt.cm.tab10(np.arange(K))
    for k in range(K):
        ax.scatter(p_catch[k], p_high[k], s=120, color=palette[k],
                   edgecolor="k", zorder=10)
        ax.annotate(f"State {k}", (p_catch[k], p_high[k]),
                    xytext=(6, 6), textcoords="offset points", fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("P(lick | catch, baseline history)")
    ax.set_ylabel(f"P(lick | large go [log2={args.stim_high}])")
    ax.set_title("Fitted state signature in P(lick) plane (F4 diagnostic)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.3: Commit**

```bash
git add src/visdetect/analysis/hmm.py scripts/analysis/behavior/hmm_state_signature_diagnostic.py
git commit -m "F4: document y = is_hit|is_fa commitment + state-signature diagnostic"
```

---

## Task 2: F14 — Posterior-confidence gating helper

**Files:**
- Modify: `src/visdetect/analysis/hmm.py` (add `assign_states_with_confidence`; update `decode_session`)
- Create: `tests/test_hmm_confidence.py`

This task adds a helper that takes per-trial posterior probabilities (γ) and returns a state assignment that uses `-1` ("unassigned") for trials where no state's posterior exceeds a threshold. Then `decode_session` gains a `confidence_threshold` option that, when set, applies this helper to the column it returns.

- [ ] **Step 2.1: Write the failing test**

Create `tests/test_hmm_confidence.py`:

```python
"""Tests for F14: posterior-confidence gating helper."""

import numpy as np
import pytest

from visdetect.analysis.hmm import assign_states_with_confidence


def test_assign_high_confidence_keeps_argmax():
    """When every trial has γ_max > threshold, return argmax."""
    # T=3, K=3.  Each row has a clear winner above 0.8.
    posteriors = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.85, 0.05],
        [0.05, 0.1, 0.85],
    ])
    out = assign_states_with_confidence(posteriors, threshold=0.8)
    np.testing.assert_array_equal(out, np.array([0, 1, 2]))


def test_assign_low_confidence_returns_minus_one():
    """When γ_max <= threshold, the trial gets -1 (unassigned)."""
    posteriors = np.array([
        [0.5, 0.4, 0.1],   # max 0.5 < 0.8 → -1
        [0.1, 0.85, 0.05], # max 0.85 → state 1
        [0.45, 0.45, 0.10],# max 0.45 < 0.8 → -1
    ])
    out = assign_states_with_confidence(posteriors, threshold=0.8)
    np.testing.assert_array_equal(out, np.array([-1, 1, -1]))


def test_assign_threshold_zero_passes_everything():
    """Threshold = 0 always returns argmax (no -1)."""
    posteriors = np.array([
        [0.34, 0.33, 0.33],
        [0.4, 0.4, 0.2],
    ])
    out = assign_states_with_confidence(posteriors, threshold=0.0)
    # argmax breaks ties at the first equal index — that's fine for our test.
    assert (out >= 0).all()
    np.testing.assert_array_equal(out, np.argmax(posteriors, axis=1))


def test_assign_empty_input():
    """Empty input returns empty array (shape preserved)."""
    posteriors = np.empty((0, 3))
    out = assign_states_with_confidence(posteriors, threshold=0.5)
    assert out.shape == (0,)


def test_assign_dtype_is_int():
    """Output is always integer (for indexing downstream)."""
    posteriors = np.array([[0.9, 0.1], [0.5, 0.5]])
    out = assign_states_with_confidence(posteriors, threshold=0.6)
    assert np.issubdtype(out.dtype, np.integer)
```

- [ ] **Step 2.2: Run the test — verify it fails**

```bash
py -m pytest tests/test_hmm_confidence.py -v
```

Expected: ImportError or `AttributeError: module 'visdetect.analysis.hmm' has no attribute 'assign_states_with_confidence'`.

- [ ] **Step 2.3: Implement `assign_states_with_confidence` in `hmm.py`**

Add this function in `src/visdetect/analysis/hmm.py` after `auto_label_states` (around line ~904):

```python
# =====================================================================
# Gating safety (F14)
# =====================================================================

def assign_states_with_confidence(
    posteriors: np.ndarray,
    threshold: float = 0.8,
) -> np.ndarray:
    """Assign each trial to its argmax state, except return -1 when no state's
    posterior exceeds *threshold*.

    The purpose is gating safety for downstream neural analyses: trials with
    ambiguous posteriors (e.g., γ = [0.45, 0.55, 0.0]) should not contribute
    to any per-state PSTH or decoder, because they reflect a mixed regime.

    Parameters
    ----------
    posteriors : ndarray (T, K)
        Posterior state probabilities (each row sums to ~1).
    threshold : float, default 0.8
        Minimum γ_max to accept the argmax assignment.

    Returns
    -------
    states : ndarray (T,) int
        argmax-assigned state per trial, with -1 where γ_max <= threshold.

    Notes
    -----
    Use this for neural-conditioning calls (per-state PSTHs, decoders, …).
    For behavioral characterization (state fractions, dwell times), prefer
    the raw Viterbi sequence from ``GLMHMM.most_likely_states``.
    """
    if posteriors.size == 0:
        return np.empty(0, dtype=int)
    max_prob = posteriors.max(axis=1)
    assigned = posteriors.argmax(axis=1).astype(int)
    assigned[max_prob <= threshold] = -1
    return assigned
```

- [ ] **Step 2.4: Run the test — verify it passes**

```bash
py -m pytest tests/test_hmm_confidence.py -v
```

Expected: 5 passed.

- [ ] **Step 2.5: Update `decode_session` to accept `confidence_threshold`**

In `src/visdetect/analysis/hmm.py`, modify `decode_session` (around line ~911) signature and body:

```python
def decode_session(
    model: GLMHMM,
    session: Session,
    state_labels: Optional[List[str]] = None,
    confidence_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Decode a session: return a DataFrame with per-trial state assignments.

    Columns added to the trial DataFrame:
      hmm_state          : int    (Viterbi)
      hmm_state_label    : str    (if *state_labels* provided)
      p_state_0 … K-1    : float  (posterior probabilities)
      hmm_state_gated    : int    (only if confidence_threshold given;
                                   -1 where γ_max <= threshold)

    Parameters
    ----------
    confidence_threshold : float, optional
        If given, also add ``hmm_state_gated`` column using
        ``assign_states_with_confidence`` for gating-safe neural analyses.
        Typical value: 0.8.

    The returned DataFrame only contains valid (non-excluded) trials.
    """
    data = prepare_session_data(session)
    if len(data["y"]) == 0:
        return data["df"]

    states = model.most_likely_states(data)
    posteriors = model.state_posteriors(data)

    df = data["df"].copy()
    df["hmm_state"] = states
    if state_labels is not None:
        df["hmm_state_label"] = [state_labels[s] for s in states]
    for k in range(model.n_states):
        df[f"p_state_{k}"] = posteriors[:, k]

    if confidence_threshold is not None:
        df["hmm_state_gated"] = assign_states_with_confidence(
            posteriors, threshold=confidence_threshold
        )

    return df
```

- [ ] **Step 2.6: Add a smoke test for `decode_session` confidence behavior**

Append to `tests/test_hmm_confidence.py`:

```python
def test_decode_session_gated_column_only_when_threshold_set(monkeypatch):
    """`decode_session` adds hmm_state_gated only when confidence_threshold is given."""
    import pandas as pd
    from visdetect.analysis.hmm import GLMHMM, decode_session

    # Build a minimal fake session via monkey-patching prepare_session_data
    # — we just need the data dict shape; the model needs to fit it.
    fake_df = pd.DataFrame({
        "outcome": ["hit", "miss", "fa"],
        "is_hit": [True, False, False],
        "is_fa":  [False, False, True],
        "is_go":  [True, True, False],
        "is_catch": [False, False, True],
        "change_size": [2.0, 2.0, 1.0],
    })
    fake_data = {
        "y": np.array([1.0, 0.0, 1.0]),
        "X": np.array([
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ]),
        "df": fake_df,
        "session_name": "fake",
        "feature_names": ["bias","stim","pc","pr","pel"],
    }

    from visdetect.analysis import hmm as hmm_mod
    monkeypatch.setattr(hmm_mod, "prepare_session_data", lambda s: fake_data)

    model = GLMHMM(n_states=2, n_features=5)
    model._init_params(seed=0)

    # without threshold -> no gated column
    out = decode_session(model, session=None)
    assert "hmm_state_gated" not in out.columns

    # with threshold -> gated column present
    out = decode_session(model, session=None, confidence_threshold=0.8)
    assert "hmm_state_gated" in out.columns
    assert out["hmm_state_gated"].dtype.kind in ("i", "u")
```

- [ ] **Step 2.7: Run all confidence tests**

```bash
py -m pytest tests/test_hmm_confidence.py -v
```

Expected: 6 passed.

- [ ] **Step 2.8: Commit**

```bash
git add src/visdetect/analysis/hmm.py tests/test_hmm_confidence.py
git commit -m "F14: add posterior-confidence gating helper and decode_session option"
```

---

## Task 3: F25 — Explicit a priori auto-labeling

**Files:**
- Modify: `src/visdetect/analysis/hmm.py` (add `auto_label_states_explicit`; keep `auto_label_states`)
- Create: `tests/test_hmm_labels_explicit.py`

Replace the rank-based labeling heuristic with explicit (p_catch, p_high) joint criteria so that "Impulsive" / "Stimulus_sensitive" / "Disengaged" mean the same thing across CV folds and (in future) across mice. Existing `auto_label_states` stays as a fallback when explicit criteria produce all-Unlabeled.

- [ ] **Step 3.1: Write the failing test**

Create `tests/test_hmm_labels_explicit.py`:

```python
"""Tests for F25: explicit a priori auto-labeling."""

import numpy as np
import pytest

from visdetect.analysis.hmm import GLMHMM, auto_label_states_explicit


def _make_model_with_weights(weights: np.ndarray) -> GLMHMM:
    """Build a GLMHMM with the given weight matrix; no fitting."""
    K, D = weights.shape
    model = GLMHMM(n_states=K, n_features=D)
    model._init_params(seed=0)
    model._weights = weights.copy()
    return model


def test_three_states_canonical():
    """K=3 with weights placing states cleanly in Impulsive/Stim/Disengaged regions."""
    # D = 5: [bias, stim, prev_choice, prev_reward, prev_early_lick]
    # Impulsive: bias high (large positive) → P(lick | catch)≈1 AND P(lick | go)≈1
    # Stim-sensitive: bias very negative, stim weight large positive
    #                 → P(lick | catch)≈0, P(lick | log2=2)≈1
    # Disengaged: bias very negative, stim weight ~0
    #             → P(lick | catch)≈0, P(lick | go)≈0
    weights = np.array([
        [ 3.0, 0.0, 0.0, 0.0, 0.0],   # Impulsive
        [-3.0, 2.5, 0.0, 0.0, 0.0],   # Stim-sensitive
        [-3.0, 0.0, 0.0, 0.0, 0.0],   # Disengaged
    ])
    model = _make_model_with_weights(weights)
    labels = auto_label_states_explicit(model, tau_low=0.2, tau_high=0.5)
    assert labels == ["Impulsive", "Stimulus_sensitive", "Disengaged"]


def test_intermediate_states_marked():
    """A state that falls into none of the three regions is marked Intermediate."""
    weights = np.array([
        [ 3.0, 0.0, 0.0, 0.0, 0.0],   # Impulsive
        [ 0.0, 0.5, 0.0, 0.0, 0.0],   # ambiguous (mid-bias, low stim)
        [-3.0, 0.0, 0.0, 0.0, 0.0],   # Disengaged
    ])
    model = _make_model_with_weights(weights)
    labels = auto_label_states_explicit(model, tau_low=0.2, tau_high=0.5)
    assert labels[0] == "Impulsive"
    assert labels[2] == "Disengaged"
    assert labels[1].startswith("Intermediate")


def test_label_count_matches_state_count():
    """Output has one label per state, always."""
    for K in (2, 3, 4, 5):
        weights = np.random.RandomState(K).normal(0, 1, (K, 5))
        model = _make_model_with_weights(weights)
        labels = auto_label_states_explicit(model)
        assert len(labels) == K


def test_threshold_tuning_changes_labels():
    """Lowering tau_high lets more states qualify as Impulsive."""
    weights = np.array([
        [ 0.8, 0.0, 0.0, 0.0, 0.0],   # P(catch) ≈ 0.69
        [-3.0, 0.0, 0.0, 0.0, 0.0],
    ])
    model = _make_model_with_weights(weights)

    strict = auto_label_states_explicit(model, tau_low=0.2, tau_high=0.7)
    loose  = auto_label_states_explicit(model, tau_low=0.2, tau_high=0.5)
    # With strict τ_high=0.7, state 0's P(catch)=0.69 falls *just* below;
    # with loose τ_high=0.5 it's above.
    assert "Impulsive" not in strict[0]
    assert loose[0] == "Impulsive"
```

- [ ] **Step 3.2: Run the test — verify it fails**

```bash
py -m pytest tests/test_hmm_labels_explicit.py -v
```

Expected: ImportError on `auto_label_states_explicit`.

- [ ] **Step 3.3: Implement `auto_label_states_explicit`**

In `src/visdetect/analysis/hmm.py`, add this function right after `auto_label_states` (around line ~905):

```python
def auto_label_states_explicit(
    model: GLMHMM,
    *,
    tau_low: float = 0.2,
    tau_high: float = 0.5,
    stim_high: float = 2.0,
) -> List[str]:
    """Assign labels using explicit a priori criteria over (P(lick|catch), P(lick|large-go)).

    Foundation for cross-mouse state correspondence (see audit spec §1.1, F25,
    CC-2). Unlike ``auto_label_states`` (rank-based), this guarantees that two
    states labeled "Impulsive" in different fits/animals satisfy the same joint
    signature.

    Criteria:
        Impulsive          : p_catch >= tau_high AND p_high >= tau_high
        Stimulus_sensitive : p_catch <  tau_low  AND p_high >= tau_high
        Disengaged         : p_catch <  tau_low  AND p_high <  tau_high
        else               : "Intermediate_{k}"

    For K > 3, multiple states may match the same region; suffix with `_1, _2`
    by ascending sensitivity (p_high - p_catch).

    Parameters
    ----------
    model : GLMHMM
        Fitted model.
    tau_low : float
        Upper bound on P(lick|catch) for "low impulsivity" classification.
    tau_high : float
        Lower bound on P(lick) for "high responsiveness" classification.
    stim_high : float
        log2(change_size) value treated as "large go" stimulus. Default 2.0
        (= log2(4.0), the largest change_size in the BG_046 protocol).

    Returns
    -------
    list of str, length K.
    """
    K, D = model.n_states, model.n_features
    x_catch = np.zeros(D); x_catch[0] = 1.0
    x_high  = np.zeros(D); x_high[0]  = 1.0; x_high[1] = stim_high

    p_catch = np.array([float(expit(model.weights[k] @ x_catch)) for k in range(K)])
    p_high  = np.array([float(expit(model.weights[k] @ x_high))  for k in range(K)])

    raw_labels: List[str] = []
    for k in range(K):
        if p_catch[k] >= tau_high and p_high[k] >= tau_high:
            raw_labels.append("Impulsive")
        elif p_catch[k] < tau_low and p_high[k] >= tau_high:
            raw_labels.append("Stimulus_sensitive")
        elif p_catch[k] < tau_low and p_high[k] < tau_high:
            raw_labels.append("Disengaged")
        else:
            raw_labels.append(f"Intermediate_{k}")

    # Disambiguate duplicates by sensitivity ascending.
    sensitivity = p_high - p_catch
    counts: Dict[str, int] = {}
    for lbl in raw_labels:
        counts[lbl] = counts.get(lbl, 0) + 1

    final: List[str] = list(raw_labels)
    for canonical in ("Impulsive", "Stimulus_sensitive", "Disengaged"):
        if counts.get(canonical, 0) > 1:
            idxs = [i for i, lbl in enumerate(raw_labels) if lbl == canonical]
            order = sorted(idxs, key=lambda i: sensitivity[i])
            for rank, idx in enumerate(order, start=1):
                final[idx] = f"{canonical}_{rank}"
    return final
```

- [ ] **Step 3.4: Run the test — verify it passes**

```bash
py -m pytest tests/test_hmm_labels_explicit.py -v
```

Expected: 4 passed.

- [ ] **Step 3.5: Commit**

```bash
git add src/visdetect/analysis/hmm.py tests/test_hmm_labels_explicit.py
git commit -m "F25: add auto_label_states_explicit (a priori joint criteria)"
```

---

## Task 4: F1 — CV-based K selection in bits/trial

**Files:**
- Modify: `src/visdetect/analysis/hmm.py` (extend `fit_best_model`; add helper for bits/trial)
- Modify: `src/visdetect/analysis/hmm_downstream.py` (small return-shape change)
- Create: `tests/test_hmm_cv_selection.py`

Wire `loso_cross_validation` (already in `hmm_downstream.py`) into `fit_best_model` so that the K-comparison output includes a `cv_ll_bits_per_trial` column and the best K is chosen by maximum CV LL (not minimum BIC). BIC and AIC remain in the output for reference.

- [ ] **Step 4.1: Write the failing test**

Create `tests/test_hmm_cv_selection.py`:

```python
"""Tests for F1: CV-based K selection in bits/trial."""

import numpy as np
import pytest
from scipy.special import expit

from visdetect.analysis.hmm import GLMHMMConfig, fit_best_model


def _generate_synthetic_sessions(
    n_sessions: int = 4,
    T: int = 120,
    K_true: int = 2,
    seed: int = 0,
):
    """Synthetic data: K_true states, identifiable enough that K=2 beats K=1."""
    rng = np.random.default_rng(seed)
    true_w = np.array([[-2.0, 1.5, 0.0, 0.0, 0.0],
                       [ 1.5, 0.3, 0.0, 0.0, 0.0]])
    true_A = np.array([[0.95, 0.05],
                       [0.06, 0.94]])

    sessions = []
    for s in range(n_sessions):
        z = np.empty(T, dtype=int)
        z[0] = rng.choice(K_true)
        for t in range(1, T):
            z[t] = rng.choice(K_true, p=true_A[z[t-1]])

        X = np.column_stack([
            np.ones(T),
            rng.uniform(-1, 2, T),
            rng.binomial(1, 0.5, T).astype(float),
            rng.binomial(1, 0.3, T).astype(float),
            rng.binomial(1, 0.2, T).astype(float),
        ])
        y = np.array([rng.binomial(1, expit(true_w[z[t]] @ X[t])) for t in range(T)],
                     dtype=float)
        sessions.append({
            "y": y, "X": X, "df": None,
            "session_name": f"sess{s}",
            "feature_names": ["bias","stim","pc","pr","pel"],
        })
    return sessions


def test_fit_best_model_returns_cv_column():
    sessions = _generate_synthetic_sessions(n_sessions=4, T=120, seed=1)
    cfg = GLMHMMConfig(max_iter=80, n_restarts=3, verbose=False)
    _, selection_df, _ = fit_best_model(
        sessions, K_range=(1, 2, 3), config=cfg, verbose=False,
        use_cross_validation=True, n_workers=1,
    )
    assert "cv_ll_bits_per_trial" in selection_df.columns
    assert "cv_ll_std" in selection_df.columns


def test_fit_best_model_cv_selects_higher_ll():
    """Best K maximises cv_ll_bits_per_trial (not minimises BIC)."""
    sessions = _generate_synthetic_sessions(n_sessions=4, T=120, seed=2)
    cfg = GLMHMMConfig(max_iter=80, n_restarts=3, verbose=False)
    best_model, selection_df, _ = fit_best_model(
        sessions, K_range=(1, 2, 3), config=cfg, verbose=False,
        use_cross_validation=True, n_workers=1,
    )
    best_K = best_model.n_states
    # The best K's CV LL must be the maximum in the table.
    assert selection_df.loc[selection_df["K"] == best_K, "cv_ll_bits_per_trial"].iloc[0] \
        == selection_df["cv_ll_bits_per_trial"].max()


def test_fit_best_model_legacy_bic_path_still_works():
    """use_cross_validation=False keeps the old BIC-based selection."""
    sessions = _generate_synthetic_sessions(n_sessions=3, T=120, seed=3)
    cfg = GLMHMMConfig(max_iter=60, n_restarts=2, verbose=False)
    best_model, selection_df, _ = fit_best_model(
        sessions, K_range=(1, 2), config=cfg, verbose=False,
        use_cross_validation=False, n_workers=1,
    )
    # Legacy path: selected K minimises BIC.
    best_K = best_model.n_states
    assert selection_df.loc[selection_df["K"] == best_K, "bic"].iloc[0] \
        == selection_df["bic"].min()
    # No CV column in legacy path.
    assert "cv_ll_bits_per_trial" not in selection_df.columns
```

- [ ] **Step 4.2: Run the test — verify it fails**

```bash
py -m pytest tests/test_hmm_cv_selection.py -v
```

Expected: `TypeError` about `use_cross_validation` being an unexpected keyword argument (the parameter doesn't exist yet).

- [ ] **Step 4.3: Add a bits-per-trial helper to `hmm.py`**

In `src/visdetect/analysis/hmm.py`, add this function after the existing scoring methods (around line ~580, before the `interpretation` section):

```python
def _baseline_bernoulli_ll(y_all: np.ndarray) -> float:
    """Log-likelihood of a constant-rate Bernoulli null model.

    Used as the baseline against which bits-per-trial is computed
    (Ashwood Eq. 22). The null model predicts each y_t to be a Bernoulli
    draw with probability equal to the empirical mean of y.
    """
    if y_all.size == 0:
        return 0.0
    p = float(np.clip(y_all.mean(), _EPS, 1 - _EPS))
    return float((y_all * np.log(p) + (1 - y_all) * np.log(1 - p)).sum())


def ll_to_bits_per_trial(
    ll: float,
    sessions_data: List[Dict[str, Any]],
) -> float:
    """Convert raw log-likelihood to bits-per-trial vs Bernoulli null.

    bits_per_trial = (LL_model - LL_null) / (n_trials * log(2))

    where LL_null is the log-likelihood of a single-probability Bernoulli
    model. This matches Ashwood Methods Eq. 22 and makes log-likelihoods
    comparable across animals with different trial counts.
    """
    y_all = np.concatenate([s["y"] for s in sessions_data if len(s["y"]) > 0])
    n = len(y_all)
    if n == 0:
        return 0.0
    ll_null = _baseline_bernoulli_ll(y_all)
    return (ll - ll_null) / (n * np.log(2.0))
```

- [ ] **Step 4.4: Extend `fit_best_model` signature and CV path**

In `src/visdetect/analysis/hmm.py`, modify the `fit_best_model` function signature and add the CV branch. Find the function (around line ~718) and rewrite as follows. Replace the entire function:

```python
def fit_best_model(
    sessions_data: List[Dict[str, Any]],
    K_range: Sequence[int] = (2, 3, 4, 5),
    config: Optional[GLMHMMConfig] = None,
    verbose: bool = True,
    n_workers: int = 1,
    seed: int = 0,
    use_cross_validation: bool = True,
    cv_n_restarts: int = 5,
) -> Tuple["GLMHMM", pd.DataFrame, Dict[int, "GLMHMM"]]:
    """Fit GLM-HMMs for each K, selecting the best by CV LL (default) or BIC.

    Default: maximises mean leave-one-session-out CV LL in bits-per-trial
    (Ashwood Methods Eq. 22). To revert to BIC selection (legacy), pass
    ``use_cross_validation=False``.

    Parameters
    ----------
    sessions_data : list of session dicts.
    K_range : sequence of int.
    config : GLMHMMConfig, optional.
    verbose : bool.
    n_workers : int.  Parallel workers across K values for the training fit.
    seed : int.
    use_cross_validation : bool, default True.
        True  → select K on maximum cv_ll_bits_per_trial via LOSO.
        False → select K on minimum BIC (legacy path).
    cv_n_restarts : int, default 5.
        Random restarts within each LOSO fold (smaller than training to
        keep CV affordable; LOSO already enforces stability).

    Returns
    -------
    best_model, selection_df, all_models.
    selection_df columns when use_cross_validation=True:
        K, train_ll, bic, aic, n_params,
        cv_ll_bits_per_trial, cv_ll_std
    """
    cfg = config or GLMHMMConfig()
    cfg_copy = GLMHMMConfig(**{k: getattr(cfg, k) for k in cfg.__dataclass_fields__})
    cfg_copy.verbose = False

    n_features = sessions_data[0]["X"].shape[1] if len(sessions_data) > 0 else len(FEATURE_NAMES)

    tasks = [
        KFitTask(
            K=K,
            sessions_data=sessions_data,
            n_features=n_features,
            config=cfg_copy,
            n_restarts=cfg.n_restarts,
            base_seed=seed,
        )
        for K in K_range
    ]

    records: List[Dict[str, Any]] = []
    all_models: Dict[int, GLMHMM] = {}

    # ---------------- Stage 1: training fits ----------------
    fit_results: Dict[int, Tuple[Optional[GLMHMM], float, int]] = {}
    if n_workers > 1:
        if verbose:
            print(f"\nFitting {len(K_range)} K values in parallel with {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_fit_single_K, task) for task in tasks]
            for future in tqdm(futures, desc="K-fits", disable=not verbose):
                K, m, ll, nf = future.result()
                fit_results[K] = (m, ll, nf)
    else:
        for task in tasks:
            K, m, ll, nf = _fit_single_K(task)
            fit_results[K] = (m, ll, nf)
            if verbose:
                print(f"  K={K}  train LL={ll:.2f}")

    # ---------------- Stage 2 (optional): cross-validation ----------------
    cv_results: Dict[int, Tuple[float, float]] = {}
    if use_cross_validation:
        from visdetect.analysis.hmm_downstream import loso_cross_validation
        for K in K_range:
            if verbose:
                print(f"\n  LOSO CV at K={K}  ({len(sessions_data)} folds, "
                      f"{cv_n_restarts} restarts/fold)")
            cv_cfg = GLMHMMConfig(**{
                k: getattr(cfg_copy, k) for k in cfg_copy.__dataclass_fields__
            })
            cv_cfg.n_restarts = cv_n_restarts
            cv_df = loso_cross_validation(
                sessions_data, K=K, config=cv_cfg,
                n_restarts=cv_n_restarts, seed=seed, verbose=False,
            )
            # Compute bits-per-trial relative to per-session null
            if len(cv_df):
                bpt = []
                for _, row in cv_df.iterrows():
                    held_out_y = sessions_data[int(row["fold"])]["y"]
                    null_ll = _baseline_bernoulli_ll(held_out_y)
                    n = int(row["n_trials_test"])
                    bpt.append((row["test_ll"] - null_ll) / (n * np.log(2.0)))
                cv_results[K] = (float(np.mean(bpt)), float(np.std(bpt)))
            else:
                cv_results[K] = (np.nan, np.nan)

    # ---------------- Aggregate selection_df ----------------
    for K in K_range:
        best_model_K, best_ll_K, n_failures = fit_results[K]
        if best_model_K is None:
            if verbose:
                print(f"  K={K}: All restarts failed.")
            continue
        bic_val = best_model_K.bic(sessions_data)
        aic_val = best_model_K.aic(sessions_data)
        all_models[K] = best_model_K
        row: Dict[str, Any] = {
            "K": K,
            "train_ll": best_ll_K,
            "bic": bic_val,
            "aic": aic_val,
            "n_params": best_model_K.n_params(),
        }
        if use_cross_validation:
            mean, std = cv_results.get(K, (np.nan, np.nan))
            row["cv_ll_bits_per_trial"] = mean
            row["cv_ll_std"] = std
        records.append(row)

    selection_df = pd.DataFrame(records)
    if selection_df.empty:
        raise RuntimeError("All model fits failed.")

    if use_cross_validation:
        best_K = int(selection_df.loc[selection_df["cv_ll_bits_per_trial"].idxmax(), "K"])
    else:
        best_K = int(selection_df.loc[selection_df["bic"].idxmin(), "K"])

    best_model = all_models[best_K]
    if verbose:
        criterion = "CV LL (bits/trial)" if use_cross_validation else "BIC"
        print(f"\n*** Best model: K={best_K} (by {criterion}) ***\n")
        print(best_model.summary())

    return best_model, selection_df, all_models
```

Make sure the file's existing import of `_fit_single_K` and `KFitTask` is unchanged. The function above relies on `_baseline_bernoulli_ll` added in Step 4.3.

- [ ] **Step 4.5: Run the tests — verify they pass**

```bash
py -m pytest tests/test_hmm_cv_selection.py -v
```

Expected: 3 passed.

If the synthetic data is too small for CV to be stable, increase `T` in `_generate_synthetic_sessions` calls within the test to 200, or reduce `n_restarts` to 2.

- [ ] **Step 4.6: Regression check — existing smoke test still works**

```bash
py tests/test_hmm_smoke.py
```

Expected: `SMOKE TEST PASSED` (the smoke test uses the model directly, not `fit_best_model`, so it should be unaffected).

- [ ] **Step 4.7: Commit**

```bash
git add src/visdetect/analysis/hmm.py tests/test_hmm_cv_selection.py
git commit -m "F1: CV-based K selection in bits-per-trial (Ashwood Eq. 22)"
```

---

## Task 5: F3 — Lapse-model baseline

**Files:**
- Modify: `src/visdetect/analysis/hmm.py` (add `fit_lapse_model`; integrate "L" row in selection_df)
- Create: `tests/test_hmm_lapse.py`

The Ashwood "L" point: a restricted 2-state GLM-HMM whose state-2 weights are all zero except the bias, and whose transition rows are identical (so lapse probability is time- and stimulus-independent). For lick/no-lick (binary y), we use a single γ parameter for spontaneous-lick probability rather than Ashwood's (γ_l, γ_r) split.

- [ ] **Step 5.1: Write the failing test**

Create `tests/test_hmm_lapse.py`:

```python
"""Tests for F3: lapse-model baseline."""

import numpy as np
import pytest
from scipy.special import expit

from visdetect.analysis.hmm import GLMHMMConfig, fit_lapse_model


def _generate_lapse_data(T=400, lapse_rate=0.1, seed=0):
    """Synthetic data: clean engaged GLM with a lapse rate of `lapse_rate`."""
    rng = np.random.default_rng(seed)
    w_engaged = np.array([-0.5, 2.0, 0.0, 0.0, 0.0])
    X = np.column_stack([
        np.ones(T),
        rng.uniform(-0.5, 2.0, T),
        rng.binomial(1, 0.5, T).astype(float),
        rng.binomial(1, 0.3, T).astype(float),
        rng.binomial(1, 0.2, T).astype(float),
    ])
    is_lapse = rng.binomial(1, lapse_rate, T).astype(bool)
    p_engaged = expit(X @ w_engaged)
    p_lapse = 0.5
    y = np.empty(T, dtype=float)
    y[~is_lapse] = rng.binomial(1, p_engaged[~is_lapse])
    y[is_lapse]  = rng.binomial(1, p_lapse, is_lapse.sum())
    return [{
        "y": y, "X": X, "df": None,
        "session_name": "lapse_sess",
        "feature_names": ["bias","stim","pc","pr","pel"],
    }]


def test_lapse_model_returns_glmhmm():
    sessions = _generate_lapse_data(T=300, lapse_rate=0.1, seed=1)
    cfg = GLMHMMConfig(max_iter=100, n_restarts=3, verbose=False)
    model = fit_lapse_model(sessions, n_features=5, config=cfg)
    assert model.n_states == 2
    # Lapse state's stimulus weight must be ~0 (constrained).
    np.testing.assert_allclose(model.weights[1, 1:], 0.0, atol=1e-6)


def test_lapse_model_transition_rows_identical():
    """Lapse model has identical transition rows (stimulus-independent lapse)."""
    sessions = _generate_lapse_data(T=300, lapse_rate=0.1, seed=2)
    cfg = GLMHMMConfig(max_iter=100, n_restarts=3, verbose=False)
    model = fit_lapse_model(sessions, n_features=5, config=cfg)
    A = model.transition_matrix
    np.testing.assert_allclose(A[0], A[1], atol=1e-3)


def test_lapse_model_recovers_engaged_weights():
    """On data with low lapse rate, engaged GLM should recover stimulus sensitivity."""
    sessions = _generate_lapse_data(T=600, lapse_rate=0.05, seed=3)
    cfg = GLMHMMConfig(max_iter=150, n_restarts=5, verbose=False)
    model = fit_lapse_model(sessions, n_features=5, config=cfg)
    # Engaged state has positive stim weight (~2.0); lapse has 0.
    stim_weights = model.weights[:, 1]
    assert stim_weights.max() > 1.0
    assert abs(stim_weights.min()) < 0.1
```

- [ ] **Step 5.2: Run the test — verify it fails**

```bash
py -m pytest tests/test_hmm_lapse.py -v
```

Expected: ImportError on `fit_lapse_model`.

- [ ] **Step 5.3: Implement `fit_lapse_model`**

In `src/visdetect/analysis/hmm.py`, add the function after `fit_best_model` (around the end of the model-selection section, before `auto_label_states`):

```python
# =====================================================================
# Lapse model baseline (F3)
# =====================================================================

class _LapseGLMHMM(GLMHMM):
    """Restricted 2-state GLM-HMM for the Ashwood "L" baseline.

    Constraints (enforced after each M-step):
      - State 1 (lapse) has zero weights except for bias.
      - Transition matrix has identical rows (lapse probability is
        time-independent and stimulus-independent).
    """

    def _m_step(self, sessions_data, all_gamma, total_xi, total_init):
        super()._m_step(sessions_data, all_gamma, total_xi, total_init)
        # Enforce constraint: lapse state has only a bias term.
        self._weights[1, 1:] = 0.0
        # Enforce constraint: identical transition rows (stationary lapse).
        A = np.exp(self._log_A)
        col_means = A.mean(axis=0, keepdims=True)        # avg of two rows
        col_means = col_means / col_means.sum()           # renormalise
        A_constrained = np.repeat(col_means, 2, axis=0)
        self._log_A = np.log(A_constrained + _EPS)


def fit_lapse_model(
    sessions_data: List[Dict[str, Any]],
    n_features: int,
    config: Optional[GLMHMMConfig] = None,
    seed: int = 0,
) -> _LapseGLMHMM:
    """Fit the restricted 2-state lapse model used as Ashwood's "L" baseline.

    Returns the fitted _LapseGLMHMM with the highest log-likelihood across
    ``config.n_restarts`` random restarts.

    For lick/no-lick (binary y), the lapse state's single bias parameter
    captures the spontaneous-lick probability (analog of Ashwood's
    γ_lick / (γ_lick + γ_no_lick) ratio under binary choice).
    """
    cfg = config or GLMHMMConfig()
    cfg_copy = GLMHMMConfig(**{k: getattr(cfg, k) for k in cfg.__dataclass_fields__})
    cfg_copy.verbose = False

    best_ll = -np.inf
    best_model: Optional[_LapseGLMHMM] = None
    for r in range(cfg.n_restarts):
        m = _LapseGLMHMM(n_states=2, n_features=n_features, config=cfg_copy)
        try:
            ll = m.fit(sessions_data, seed=seed + r * 137, smart_init=(r == 0))
        except Exception:
            continue
        if ll > best_ll:
            best_ll = ll
            best_model = m
    if best_model is None:
        raise RuntimeError("Lapse model: all restarts failed.")
    return best_model
```

- [ ] **Step 5.4: Run the tests — verify they pass**

```bash
py -m pytest tests/test_hmm_lapse.py -v
```

Expected: 3 passed.

If tests are flaky on random seeds, increase `T` to 600 or `n_restarts` to 5.

- [ ] **Step 5.5: Integrate "L" row into `fit_best_model` output**

In `src/visdetect/analysis/hmm.py`, modify `fit_best_model` to also fit the lapse model and add it as a row in `selection_df`. After the K-loop and before the `selection_df = pd.DataFrame(records)` line, add:

```python
    # Lapse "L" baseline (F3)
    try:
        lapse_model = fit_lapse_model(
            sessions_data, n_features=n_features, config=cfg_copy, seed=seed,
        )
        lapse_ll = lapse_model.log_likelihood(sessions_data)
        row_L: Dict[str, Any] = {
            "K": "L",
            "train_ll": lapse_ll,
            "bic": lapse_model.bic(sessions_data),
            "aic": lapse_model.aic(sessions_data),
            "n_params": lapse_model.n_params(),
        }
        if use_cross_validation:
            # CV the lapse model the same way as the others.
            from visdetect.analysis.hmm_downstream import loso_cross_validation
            # We cross-validate with a custom inner loop because LOSO expects K=int.
            bpt = []
            for fold_idx in range(len(sessions_data)):
                train = [s for i, s in enumerate(sessions_data) if i != fold_idx]
                held = sessions_data[fold_idx]
                if len(held["y"]) == 0:
                    continue
                try:
                    m_fold = fit_lapse_model(
                        train, n_features=n_features, config=cfg_copy, seed=seed + fold_idx,
                    )
                    test_ll = m_fold.log_likelihood([held])
                    null_ll = _baseline_bernoulli_ll(held["y"])
                    n = len(held["y"])
                    bpt.append((test_ll - null_ll) / (n * np.log(2.0)))
                except Exception:
                    continue
            if bpt:
                row_L["cv_ll_bits_per_trial"] = float(np.mean(bpt))
                row_L["cv_ll_std"] = float(np.std(bpt))
            else:
                row_L["cv_ll_bits_per_trial"] = np.nan
                row_L["cv_ll_std"] = np.nan
        records.append(row_L)
        all_models["L"] = lapse_model
    except Exception as exc:
        if verbose:
            print(f"  Lapse model fit failed: {exc}  (continuing without L)")
```

The selection logic at the end of the function must skip the "L" row when picking the best K:

Find and replace:
```python
    if use_cross_validation:
        best_K = int(selection_df.loc[selection_df["cv_ll_bits_per_trial"].idxmax(), "K"])
    else:
        best_K = int(selection_df.loc[selection_df["bic"].idxmin(), "K"])
```

With:
```python
    K_only = selection_df[selection_df["K"] != "L"]
    if use_cross_validation:
        best_K = int(K_only.loc[K_only["cv_ll_bits_per_trial"].idxmax(), "K"])
    else:
        best_K = int(K_only.loc[K_only["bic"].idxmin(), "K"])
```

- [ ] **Step 5.6: Add integration test**

Append to `tests/test_hmm_lapse.py`:

```python
def test_fit_best_model_includes_lapse_row():
    """selection_df now has an 'L' row alongside the K-state rows."""
    from visdetect.analysis.hmm import fit_best_model
    sessions = _generate_lapse_data(T=300, lapse_rate=0.1, seed=10)
    cfg = GLMHMMConfig(max_iter=80, n_restarts=3, verbose=False)
    best_model, selection_df, all_models = fit_best_model(
        sessions, K_range=(1, 2), config=cfg, verbose=False,
        use_cross_validation=True, n_workers=1, cv_n_restarts=2,
    )
    # Lapse row present, identifiable by K == "L"
    assert (selection_df["K"] == "L").any()
    assert "L" in all_models
    # Selected best_K must be an integer (the lapse row is excluded from selection)
    assert isinstance(best_model.n_states, int)
```

- [ ] **Step 5.7: Run all lapse tests**

```bash
py -m pytest tests/test_hmm_lapse.py tests/test_hmm_cv_selection.py -v
```

Expected: all passed (4 + 3 = 7).

- [ ] **Step 5.8: Commit**

```bash
git add src/visdetect/analysis/hmm.py tests/test_hmm_lapse.py
git commit -m "F3: add lapse-model baseline as 'L' row in K-comparison output"
```

---

## Task 6: F22 — External behavioral validation

**Files:**
- Create: `src/visdetect/analysis/hmm_validation.py`
- Create: `tests/test_hmm_validation.py`
- Create: `scripts/analysis/behavior/hmm_external_validation.py`

Implement per-state observable computations (lick latency, RT distribution, TF-pulse responsiveness, psychometric slope) and a script that produces the F22 figure for BG_046 using the fitted Track-A model. TF-pulse responsiveness is the key discriminator per spec §4.7.

- [ ] **Step 6.1: Write failing tests for observable computations**

Create `tests/test_hmm_validation.py`:

```python
"""Tests for F22: external behavioral validation observables."""

import numpy as np
import pandas as pd
import pytest

from visdetect.analysis.hmm_validation import (
    per_state_lick_latency,
    per_state_response_time_quantiles,
    per_state_psychometric_slope,
)


def _make_assignments_df(n_per_state=20, seed=0):
    """Synthetic assignments DataFrame with 3 states, RTs that differ by state."""
    rng = np.random.default_rng(seed)
    rows = []
    # State 0 = Impulsive: short RT (early-anticipatory licks)
    # State 1 = Stim-sensitive: short tight RT
    # State 2 = Disengaged: long, variable RT
    for state, mean, scale in [(0, 0.25, 0.10), (1, 0.30, 0.06), (2, 0.70, 0.30)]:
        for _ in range(n_per_state):
            rt = max(0.0, rng.normal(mean, scale))
            rows.append({
                "hmm_state": state,
                "rt": rt,                # change-relative reaction time
                "change_time": 1.0,      # absolute (arbitrary)
                "response_time": rt + 1.0,
                "is_hit": True,
                "is_go": True,
                "is_catch": False,
                "is_fa": False,
                "change_size": rng.choice([1.25, 1.5, 2.0, 4.0]),
            })
    # A handful of catch / fa trials
    for state in (0, 1, 2):
        for _ in range(5):
            rows.append({
                "hmm_state": state,
                "rt": np.nan,
                "change_time": np.nan,
                "response_time": np.nan,
                "is_hit": False,
                "is_go": False,
                "is_catch": True,
                "is_fa": False,
                "change_size": 1.0,
            })
    return pd.DataFrame(rows)


def test_lick_latency_distinguishes_states():
    df = _make_assignments_df(seed=1)
    out = per_state_lick_latency(df, n_states=3)
    # Disengaged (state 2) must have higher median latency than Stim-sensitive (state 1).
    med = out.set_index("state")["median_latency_s"]
    assert med[2] > med[1]
    # All three states present.
    assert set(out["state"]) == {0, 1, 2}


def test_response_time_quantiles_shape():
    df = _make_assignments_df(seed=2)
    out = per_state_response_time_quantiles(df, n_states=3, quantiles=(0.25, 0.5, 0.75, 0.9))
    assert set(out.columns) >= {"state", "q25", "q50", "q75", "q90", "n"}
    assert len(out) == 3


def test_psychometric_slope_higher_in_stim_sensitive():
    """Stim-sensitive state should have a steeper P(lick) vs change_size slope."""
    # Build a dataset where state 1 (stim-sensitive) has hits scaling with change_size,
    # while state 2 has uniform low hit rate.
    rng = np.random.default_rng(3)
    rows = []
    sizes = [1.25, 1.5, 2.0, 4.0]
    for cs in sizes:
        # State 1: hit rate scales (0.3, 0.5, 0.75, 0.95)
        p1 = {1.25: 0.3, 1.5: 0.5, 2.0: 0.75, 4.0: 0.95}[cs]
        for _ in range(30):
            rows.append({"hmm_state": 1, "is_hit": rng.binomial(1, p1) == 1,
                          "is_go": True, "is_catch": False, "change_size": cs})
        # State 2: flat low ~0.2
        for _ in range(30):
            rows.append({"hmm_state": 2, "is_hit": rng.binomial(1, 0.2) == 1,
                          "is_go": True, "is_catch": False, "change_size": cs})
    df = pd.DataFrame(rows)
    out = per_state_psychometric_slope(df, n_states=3)
    slope = out.set_index("state")["slope"]
    assert slope.get(1, 0) > slope.get(2, 0)
```

- [ ] **Step 6.2: Run the test — verify it fails**

```bash
py -m pytest tests/test_hmm_validation.py -v
```

Expected: ImportError on `visdetect.analysis.hmm_validation`.

- [ ] **Step 6.3: Create `hmm_validation.py`**

Create `src/visdetect/analysis/hmm_validation.py`:

```python
"""External state validation observables for the GLM-HMM (F22).

These compute per-state distributions of *observables not used in fitting*,
to test whether the inferred behavioral states correspond to genuinely
different regimes (per audit spec §4.7).

Observables provided here:
  - per-state lick latency on hits (change-relative)
  - per-state response-time quantiles (Ashwood-style Q-Q analog)
  - per-state psychometric slope (logistic fit P(lick) vs log2(change_size))

A separate script (``scripts/analysis/behavior/hmm_external_validation.py``)
computes TF-pulse responsiveness per state by integrating with the existing
``visdetect.analysis.tf_pulse`` module.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize


# =====================================================================
# 1.  Lick latency
# =====================================================================

def per_state_lick_latency(
    assignments_df: pd.DataFrame,
    n_states: int,
    state_col: str = "hmm_state",
    latency_col: str = "rt",
) -> pd.DataFrame:
    """Median, IQR, and n_hits of lick latency per HMM state.

    Latency here = change-relative reaction time (``rt`` column in the
    trial dataframe), restricted to hits on go trials.

    Returns DataFrame: state, median_latency_s, iqr_s, n_hits.
    """
    hits = assignments_df[assignments_df["is_hit"] & assignments_df["is_go"]]
    rows = []
    for k in range(n_states):
        sub = hits[hits[state_col] == k][latency_col].dropna()
        if len(sub):
            rows.append({
                "state": k,
                "median_latency_s": float(np.median(sub)),
                "iqr_s": float(np.percentile(sub, 75) - np.percentile(sub, 25)),
                "n_hits": int(len(sub)),
            })
        else:
            rows.append({
                "state": k,
                "median_latency_s": np.nan,
                "iqr_s": np.nan,
                "n_hits": 0,
            })
    return pd.DataFrame(rows)


# =====================================================================
# 2.  Response-time quantiles (Ashwood Fig 6 Q-Q analog)
# =====================================================================

def per_state_response_time_quantiles(
    assignments_df: pd.DataFrame,
    n_states: int,
    quantiles: Iterable[float] = (0.25, 0.5, 0.75, 0.9),
    state_col: str = "hmm_state",
    latency_col: str = "rt",
) -> pd.DataFrame:
    """Per-state RT quantiles on hits."""
    hits = assignments_df[assignments_df["is_hit"] & assignments_df["is_go"]]
    rows = []
    quantiles = list(quantiles)
    for k in range(n_states):
        sub = hits[hits[state_col] == k][latency_col].dropna().values
        row = {"state": k, "n": int(len(sub))}
        for q in quantiles:
            label = f"q{int(round(q * 100)):02d}"
            row[label] = float(np.quantile(sub, q)) if len(sub) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# =====================================================================
# 3.  Per-state psychometric slope
# =====================================================================

def _logistic_slope(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """MLE fit of P(y=1) = sigmoid(beta0 + beta1 * x); return (intercept, slope).

    Uses scipy.optimize.minimize on negative log-likelihood. Returns
    (np.nan, np.nan) if fitting fails or the data is degenerate.
    """
    if len(x) == 0 or y.std() == 0 or x.std() == 0:
        return (np.nan, np.nan)

    def nll(params):
        b0, b1 = params
        z = b0 + b1 * x
        return -np.sum(y * z - np.logaddexp(0, z))

    try:
        res = minimize(nll, x0=[0.0, 0.0], method="L-BFGS-B")
        if res.success:
            return (float(res.x[0]), float(res.x[1]))
    except Exception:
        pass
    return (np.nan, np.nan)


def per_state_psychometric_slope(
    assignments_df: pd.DataFrame,
    n_states: int,
    state_col: str = "hmm_state",
) -> pd.DataFrame:
    """Per-state psychometric: logistic fit of P(lick) vs log2(change_size) on go trials.

    Returns DataFrame: state, intercept, slope, n_go.
    """
    go = assignments_df[assignments_df["is_go"]]
    rows = []
    for k in range(n_states):
        sub = go[go[state_col] == k]
        if len(sub) < 5:
            rows.append({"state": k, "intercept": np.nan, "slope": np.nan,
                         "n_go": int(len(sub))})
            continue
        x = np.log2(np.clip(sub["change_size"].values.astype(float), 1.0, None))
        y = sub["is_hit"].astype(float).values
        b0, b1 = _logistic_slope(x, y)
        rows.append({"state": k, "intercept": b0, "slope": b1, "n_go": int(len(sub))})
    return pd.DataFrame(rows)
```

- [ ] **Step 6.4: Run the tests — verify they pass**

```bash
py -m pytest tests/test_hmm_validation.py -v
```

Expected: 3 passed.

- [ ] **Step 6.5: Add TF-pulse-responsiveness helper to `hmm_validation.py`**

The TF-pulse module already exists (`src/visdetect/analysis/tf_pulse.py`) and provides utilities for detecting and aligning to baseline TF outliers. For F22 we need a behavioral measure: per state, what is the lick rate within a short window after a sub-threshold TF outlier? Append to `src/visdetect/analysis/hmm_validation.py`:

```python
# =====================================================================
# 4.  TF-pulse responsiveness per state (key F22 discriminator)
# =====================================================================

def per_state_tf_pulse_lick_rate(
    session,
    assignments_df: pd.DataFrame,
    n_states: int,
    *,
    pulse_log2_threshold: float = 0.10,
    response_window_s: float = 0.40,
    state_col: str = "hmm_state",
) -> pd.DataFrame:
    """Per-state probability of lick within ``response_window_s`` of a sub-threshold
    TF pulse during baseline.

    A pulse is any TF excursion with |log2(tf/baseline_tf)| > pulse_log2_threshold
    that is NOT itself a scheduled change event (i.e., happens during baseline).

    Strategy:
      1. For each trial in ``assignments_df``, identify TF pulses during the
         baseline period (before change_time).
      2. For each pulse, check whether any lick (any outcome) occurred within
         response_window_s.
      3. Aggregate per state.

    Returns DataFrame: state, n_pulses, n_pulse_locked_licks, p_lick_pulse_locked.

    Notes
    -----
    Requires session.trials to expose ``tf_trace``, ``tf_times``, and the
    trial's ``lick_times`` / ``firstlick`` fields. If the per-trial TF trace
    isn't available, returns NaN rows (per-state lengths preserved).
    """
    # Attempt to extract per-trial TF traces and lick times.
    rows = []
    pulses_per_state = {k: [] for k in range(n_states)}
    locked_per_state = {k: 0 for k in range(n_states)}

    for trial_idx, row in assignments_df.iterrows():
        k = int(row[state_col])
        if k < 0 or k >= n_states:
            continue
        # Try to fetch trial-level TF trace.
        try:
            trial = session.trials[trial_idx]
        except Exception:
            continue
        tf_trace = getattr(trial, "tf_trace", None)
        tf_times = getattr(trial, "tf_times", None)
        if tf_trace is None or tf_times is None or len(tf_trace) == 0:
            continue
        tf_trace = np.asarray(tf_trace, dtype=float)
        tf_times = np.asarray(tf_times, dtype=float)

        baseline_tf = float(np.median(tf_trace))
        if baseline_tf <= 0:
            continue
        log2_dev = np.log2(np.maximum(tf_trace, 1e-6) / baseline_tf)
        pulse_mask = np.abs(log2_dev) > pulse_log2_threshold

        # Limit to baseline period (before change_time if available).
        ct = getattr(trial, "change_time", None)
        if ct is not None and ct > 0:
            pulse_mask &= tf_times < ct
        pulse_times = tf_times[pulse_mask]
        if len(pulse_times) == 0:
            continue
        pulses_per_state[k].extend(pulse_times.tolist())

        # Lick times on this trial.
        lick_times = getattr(trial, "lick_times", None)
        if lick_times is None or len(lick_times) == 0:
            continue
        lick_times = np.asarray(lick_times, dtype=float)
        for pt in pulse_times:
            if np.any((lick_times > pt) & (lick_times <= pt + response_window_s)):
                locked_per_state[k] += 1

    for k in range(n_states):
        n_p = len(pulses_per_state[k])
        n_l = locked_per_state[k]
        rows.append({
            "state": k,
            "n_pulses": n_p,
            "n_pulse_locked_licks": n_l,
            "p_lick_pulse_locked": (n_l / n_p) if n_p > 0 else np.nan,
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 6.6: Create the F22 figure script**

Create `scripts/analysis/behavior/hmm_external_validation.py`:

```python
"""F22: External behavioral validation per HMM state for BG_046.

Produces a four-panel figure analogous to Ashwood Fig 6:
  Panel A  Lick latency distributions per state (boxplot or violin).
  Panel B  Response-time quantile bars (analog of Q-Q tail).
  Panel C  Per-state psychometric curves (P(lick) vs log2 change_size).
  Panel D  TF-pulse responsiveness per state.

The figure is the "are the states real?" evidence for manuscript purposes.

Usage
-----
    py scripts/analysis/behavior/hmm_external_validation.py \
        --model data/hmm/BG_046/best_model.pkl \
        --manifest data/BG_046_staging_manifest_v2.csv \
        --out FIGURES/behavior/BG_046/hmm/external_validation.png \
        --data-out data/hmm/BG_046/external_validation \
        --confidence-threshold 0.8
"""

import argparse
import gc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visdetect.analysis.config import load_staging_manifest
from visdetect.analysis.hmm import GLMHMM, decode_session, auto_label_states_explicit
from visdetect.analysis.hmm_validation import (
    per_state_lick_latency,
    per_state_response_time_quantiles,
    per_state_psychometric_slope,
    per_state_tf_pulse_lick_rate,
)
from visdetect.core.session import load_session
from visdetect.viz.plotting import set_style, despine


STATE_COLORS = {
    "Impulsive": "#d95f02",
    "Stimulus_sensitive": "#1b9e77",
    "Disengaged": "#7570b3",
}


def _color_for_label(label: str) -> str:
    base = label.split("_")[0] if not label.startswith("Stimulus") else "Stimulus_sensitive"
    if label.startswith("Stimulus"):
        return STATE_COLORS["Stimulus_sensitive"]
    return STATE_COLORS.get(label, "#666666")


def _resolve_pkl_path(manifest_row: pd.Series, pkl_dir: Path) -> Path | None:
    """Find a session .pkl file. Tries an explicit 'pkl_path' column first,
    then falls back to globbing pkl_dir by session_name (the idiom in
    scripts/analysis/behavior/hmm_cross_validation.py)."""
    if "pkl_path" in manifest_row and pd.notna(manifest_row["pkl_path"]):
        return Path(manifest_row["pkl_path"])
    sname = str(manifest_row.get("session_name", ""))
    if not sname:
        return None
    candidates = list(pkl_dir.glob(f"*{sname}*.pkl"))
    return candidates[0] if candidates else None


def gather_assignments(
    model: GLMHMM,
    manifest: pd.DataFrame,
    pkl_dir: Path,
    *,
    confidence_threshold: float = 0.8,
):
    """Decode every session and concatenate into a single trial DataFrame.

    Returns (assignments_df, state_labels, sessions_by_name) where the third
    item is a dict {session_name -> Session} for downstream per-session
    access (needed for TF-pulse responsiveness which uses trial-level traces).
    """
    state_labels = auto_label_states_explicit(model)
    rows = []
    sessions_by_name: dict = {}
    for _, mrow in manifest.iterrows():
        pkl = _resolve_pkl_path(mrow, pkl_dir)
        if pkl is None or not pkl.exists():
            print(f"  Skip {mrow.get('session_name')}: pkl not found")
            continue
        try:
            sess = load_session(str(pkl))
        except Exception as exc:
            print(f"  Skip {pkl}: {exc}")
            continue
        df = decode_session(model, sess, state_labels=state_labels,
                            confidence_threshold=confidence_threshold)
        sname = sess.session_name or str(mrow.get("session_name", ""))
        df["session_name"] = sname
        rows.append(df)
        sessions_by_name[sname] = sess
    if not rows:
        raise RuntimeError("No sessions decoded.")
    return pd.concat(rows, ignore_index=True), state_labels, sessions_by_name


def plot_validation(
    latency_df: pd.DataFrame,
    rt_q_df: pd.DataFrame,
    psy_df: pd.DataFrame,
    tf_df: pd.DataFrame,
    state_labels,
    out_path: Path,
):
    set_style()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Panel A — lick latency
    ax = axes[0]
    K = len(state_labels)
    for k in range(K):
        row = latency_df[latency_df["state"] == k].iloc[0]
        if not np.isnan(row["median_latency_s"]):
            ax.bar(k, row["median_latency_s"],
                   color=_color_for_label(state_labels[k]), edgecolor="k")
            ax.errorbar(k, row["median_latency_s"], yerr=row["iqr_s"] / 2,
                        fmt="none", color="k", capsize=3)
    ax.set_xticks(range(K))
    ax.set_xticklabels(state_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Lick latency (s, median ± IQR/2)")
    ax.set_title("A. Lick latency per state")
    despine(ax)

    # Panel B — RT quantiles
    ax = axes[1]
    for k in range(K):
        row = rt_q_df[rt_q_df["state"] == k].iloc[0]
        ax.plot([0.25, 0.5, 0.75, 0.9],
                [row["q25"], row["q50"], row["q75"], row["q90"]],
                "-o", color=_color_for_label(state_labels[k]),
                label=state_labels[k])
    ax.set_xlabel("Quantile")
    ax.set_ylabel("RT (s)")
    ax.set_title("B. RT distribution shape")
    ax.legend(fontsize=8)
    despine(ax)

    # Panel C — psychometric slope
    ax = axes[2]
    for k in range(K):
        row = psy_df[psy_df["state"] == k].iloc[0]
        if np.isnan(row["slope"]):
            continue
        xx = np.linspace(0, 2.2, 60)   # log2 change_size from 0 (catch) to ~4x
        yy = 1.0 / (1.0 + np.exp(-(row["intercept"] + row["slope"] * xx)))
        ax.plot(xx, yy, color=_color_for_label(state_labels[k]),
                label=f"{state_labels[k]} (slope={row['slope']:.2f})")
    ax.set_xlabel("log2(change_size)")
    ax.set_ylabel("P(lick)")
    ax.set_ylim(0, 1)
    ax.set_title("C. Per-state psychometric")
    ax.legend(fontsize=7)
    despine(ax)

    # Panel D — TF-pulse responsiveness
    ax = axes[3]
    for k in range(K):
        row = tf_df[tf_df["state"] == k].iloc[0]
        if not np.isnan(row["p_lick_pulse_locked"]):
            ax.bar(k, row["p_lick_pulse_locked"],
                   color=_color_for_label(state_labels[k]), edgecolor="k")
            ax.text(k, row["p_lick_pulse_locked"] + 0.005,
                    f"n={row['n_pulses']}", ha="center", fontsize=7)
    ax.set_xticks(range(K))
    ax.set_xticklabels(state_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("P(lick within 400 ms of sub-threshold TF pulse)")
    ax.set_title("D. TF-pulse responsiveness  (key discriminator)")
    despine(ax)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path,
                    help="(unused if --pkl-dir is given; manifest comes from load_staging_manifest)")
    ap.add_argument("--pkl-dir", required=True, type=Path,
                    help="Directory containing per-session .pkl files (e.g. data/pkls/BG_046)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data-out", required=True, type=Path)
    ap.add_argument("--confidence-threshold", type=float, default=0.8)
    args = ap.parse_args()

    model = GLMHMM.load(args.model)
    K = model.n_states
    manifest = load_staging_manifest(qc_only=True)

    # Build the concatenated assignments dataframe.
    assignments_df, state_labels, sessions_by_name = gather_assignments(
        model, manifest, args.pkl_dir,
        confidence_threshold=args.confidence_threshold,
    )

    # Use the raw Viterbi assignment for behavioral observables (don't drop
    # trials). The `hmm_state_gated` column (set by decode_session) is the
    # one to use for neural-conditioning calls.
    state_col = "hmm_state"

    args.data_out.mkdir(parents=True, exist_ok=True)

    latency_df = per_state_lick_latency(assignments_df, n_states=K, state_col=state_col)
    latency_df.to_csv(args.data_out / "lick_latency_per_state.csv", index=False)

    rt_q_df = per_state_response_time_quantiles(assignments_df, n_states=K, state_col=state_col)
    rt_q_df.to_csv(args.data_out / "rt_quantiles_per_state.csv", index=False)

    psy_df = per_state_psychometric_slope(assignments_df, n_states=K, state_col=state_col)
    psy_df.to_csv(args.data_out / "psychometric_slope_per_state.csv", index=False)

    # TF-pulse responsiveness must be computed per-session (needs trial-level TF traces).
    tf_rows = []
    for sess_name, sub in assignments_df.groupby("session_name"):
        sess = sessions_by_name.get(sess_name)
        if sess is None:
            continue
        tf_local = per_state_tf_pulse_lick_rate(sess, sub, n_states=K, state_col=state_col)
        tf_local["session_name"] = sess_name
        tf_rows.append(tf_local)
    # Free session memory after TF analysis.
    sessions_by_name.clear()
    gc.collect()
    if tf_rows:
        tf_concat = pd.concat(tf_rows, ignore_index=True)
        tf_concat.to_csv(args.data_out / "tf_pulse_per_state_per_session.csv", index=False)
        # Pool across sessions: weighted by n_pulses
        tf_pooled = (
            tf_concat.groupby("state")
                     .agg(n_pulses=("n_pulses", "sum"),
                          n_pulse_locked_licks=("n_pulse_locked_licks", "sum"))
                     .reset_index()
        )
        tf_pooled["p_lick_pulse_locked"] = np.where(
            tf_pooled["n_pulses"] > 0,
            tf_pooled["n_pulse_locked_licks"] / tf_pooled["n_pulses"],
            np.nan,
        )
        tf_pooled.to_csv(args.data_out / "tf_pulse_per_state_pooled.csv", index=False)
    else:
        # Build placeholder so plotting doesn't crash.
        tf_pooled = pd.DataFrame({
            "state": list(range(K)),
            "n_pulses": [0] * K,
            "n_pulse_locked_licks": [0] * K,
            "p_lick_pulse_locked": [np.nan] * K,
        })

    plot_validation(latency_df, rt_q_df, psy_df, tf_pooled, state_labels, args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.7: Run the validation tests**

```bash
py -m pytest tests/test_hmm_validation.py -v
```

Expected: 3 passed.

- [ ] **Step 6.8: Smoke-test the script against BG_046 (manual)**

Before committing, ensure the script runs end-to-end on real data. This is a manual smoke test — replace paths if needed:

```bash
py scripts/analysis/behavior/hmm_external_validation.py \
    --model data/hmm/BG_046/best_model.pkl \
    --manifest data/BG_046_staging_manifest_v2.csv \
    --pkl-dir data/pkls/BG_046 \
    --out FIGURES/behavior/BG_046/hmm/external_validation.png \
    --data-out data/hmm/BG_046/external_validation \
    --confidence-threshold 0.8
```

Expected: PNG saved + CSVs in `data/hmm/BG_046/external_validation/`. If `best_model.pkl` doesn't exist yet, first run the upstream fitting pipeline (`scripts/analysis/behavior/hmm_behavioral_states.py` or equivalent) with the new CV-based selection.

If TF traces aren't exposed in the current Session API, Panel D will show NaN bars — that's acceptable for the first iteration; a follow-up PR can wire TF traces. The other three panels (A, B, C) are sufficient evidence on their own.

- [ ] **Step 6.9: Commit**

```bash
git add src/visdetect/analysis/hmm_validation.py tests/test_hmm_validation.py scripts/analysis/behavior/hmm_external_validation.py
git commit -m "F22: external state validation (lick latency, RT, psychometric, TF-pulse)"
```

---

## Final integration check

After all six tasks land, verify the whole pipeline runs together:

- [ ] **Step 7.1: Run the full test suite**

```bash
py -m pytest tests/test_hmm_confidence.py tests/test_hmm_labels_explicit.py tests/test_hmm_cv_selection.py tests/test_hmm_lapse.py tests/test_hmm_validation.py -v
```

Expected: all green.

- [ ] **Step 7.2: Refit BG_046 with CV-based selection and lapse baseline**

```bash
py scripts/analysis/behavior/hmm_cross_validation.py    # if this script wraps fit_best_model
# OR a new top-level fitting call equivalent to:
#   fit_best_model(sessions_data, K_range=(1,2,3,4,5),
#                  use_cross_validation=True, cv_n_restarts=5)
```

Inspect the resulting `selection_df`:
- It should have rows for K=1, 2, 3, 4, 5 AND a row for `K=L`.
- The `cv_ll_bits_per_trial` column should be present and monotone-ish (small for K=1, peaking somewhere ≥ 2).
- The best K should be the maximum of `cv_ll_bits_per_trial`.

- [ ] **Step 7.3: Produce F4 diagnostic + F22 figure**

```bash
py scripts/analysis/behavior/hmm_state_signature_diagnostic.py \
    --model data/hmm/BG_046/best_model.pkl \
    --out FIGURES/behavior/BG_046/hmm/state_signature.png

py scripts/analysis/behavior/hmm_external_validation.py \
    --model data/hmm/BG_046/best_model.pkl \
    --manifest data/BG_046_staging_manifest_v2.csv \
    --pkl-dir data/pkls/BG_046 \
    --out FIGURES/behavior/BG_046/hmm/external_validation.png \
    --data-out data/hmm/BG_046/external_validation \
    --confidence-threshold 0.8
```

Acceptance criteria for Track A:
- `state_signature.png` shows three fitted states near the predicted three corners.
- `external_validation.png` shows per-state differences on at least 2 of the 4 panels (lick latency, RT quantiles, psychometric slope, TF-pulse responsiveness).
- Best K chosen by CV LL matches the rank-based intuition (likely K=3 for BG_046).

If these acceptance criteria pass, Track A is complete and you have the foundation for either (a) drafting the methods/results sections for a manuscript using BG_046, or (b) proceeding to Track B (priors, time-expectancy covariate, per-stage analysis).

---

## Self-review checklist (already applied while writing this plan)

- All steps include actual code, not pseudocode.
- All file paths are absolute relative to the repo root.
- `use_cross_validation` parameter naming is consistent across Task 4 and Task 5.
- Function names (`assign_states_with_confidence`, `auto_label_states_explicit`, `fit_lapse_model`, `per_state_lick_latency`, `per_state_tf_pulse_lick_rate`) are consistent across tasks and tests.
- Each task ends with a commit step.
- No TBDs or placeholders.
- Each test step includes the exact pytest invocation and expected output.
