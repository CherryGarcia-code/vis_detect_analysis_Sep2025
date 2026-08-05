# QC1 Trial/Event Alignment Repair — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every pkl's trial table provably index-aligned to its `ni_events` per-trial arrays, repair the 17 misaligned non-BG_012 sessions without damaging behaviour, and stop the converter reintroducing the defect.

**Architecture:** A pure, unit-testable primitive (`visdetect.core.run_alignment`) scores a candidate `(trial_slice, event_offset)` pairing with two checks — a categorical outcome↔change-presence agreement over 100% of trials, and a timing residual over the ~45% of trials whose change was realised. A brute-force solver picks the unique candidate passing both. The result is stored as a per-trial index map `Session.trial_event_index` (never truncation), which `align.py` and every direct `ni_events` consumer honours.

**Tech Stack:** Python 3.10, numpy, pandas, pytest. Repo venv invoked as `py`.

**Branch state:** `feature/early-lick-and-session-sorting`, **merged up to main on 2026-08-04** (`bf0c875`) so that `tf_glm_data.py` carries the lick-channel fix. Do not revert `_collect_lick_times`'s delegation to `lick_channels.get_lick_times`. Two pre-existing stale test modules (`tests/test_coding_direction.py`, `tests/test_population.py`) reference modules deleted in the `analysis_suite` archival and fail to import — that is not caused by this work; leave them alone.

**Spec:** `docs/superpowers/specs/2026-08-03-QC1-trial-event-alignment-repair-design.md` — read §1 (root cause), §2 (both checks), §3 (representation) before starting.

## Global Constraints

- Invoke Python as `py`, never `python` (Windows + Git Bash).
- Tests live in `tests/` at repo root. Run: `py -m pytest tests/<file> -v`.
- Library imports are absolute: `from visdetect.core...`, `from visdetect.analysis...`.
- **X: is READ-ONLY.** You may read from `/x/public/projects/BeJG_20230130_VisDetect/wEPhys/`. Never write, move, rename or delete there. Never run pipelines over the share.
- **Never overwrite a pkl without a backup first.** Backups go to `data/pkls/<SUBJ>/qc1_backup/<file>.bak_<UTC-stamp>`; abort the repair if the backup cannot be written.
- `data/` and `FIGURES/` are gitignored. Deliverable CSVs are force-added with `git add -f` when they must be preserved.
- Sessions are large: `del sess; gc.collect()` after each one in any loop.
- `CHANGE_PRESENTED_OUTCOMES = {"Hit", "Miss", "Ref"}` is **case-sensitive** and **must not** be refactored onto `EVENT_VALID_OUTCOMES` (which is lowercase `{'hit','miss'}` and omits `Ref`).
- Acceptance: Check 1 agreement **== 1.0 exactly**; Check 2 median |residual| **< 0.05 s** over **≥ 20** finite-change trials. Fewer than 20 → **reject**, never "not applicable → pass".
- Verify the git branch before any git operation. Multiple worktrees are live.

---

### Task 1: Alignment scoring primitive

**Files:**
- Create: `src/visdetect/core/run_alignment.py`
- Test: `tests/test_run_alignment.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces:
  - `CHANGE_PRESENTED_OUTCOMES: frozenset[str]`
  - `ACCEPT_RESID_S: float = 0.05`, `MIN_RESID_N: int = 20`
  - `per_trial_event_keys(ni_events: dict) -> list[str]`
  - `outcome_change_agreement(trials, ni_events, trial_slice: slice, event_offset: int) -> tuple[float, int]` → `(agreement, n_compared)`
  - `alignment_residual(trials, ni_events, trial_slice: slice, event_offset: int) -> tuple[float, int]` → `(median_abs_residual_seconds, n_finite)`; returns `(nan, n)` when `n < MIN_RESID_N`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_alignment.py
import numpy as np
import pytest

from visdetect.core.run_alignment import (
    CHANGE_PRESENTED_OUTCOMES,
    MIN_RESID_N,
    alignment_residual,
    outcome_change_agreement,
    per_trial_event_keys,
)


class FakeTrial:
    def __init__(self, outcome, change_time):
        self.trialoutcome = outcome
        self.change_time = change_time


def make_case(n=60, offset=0, n_pad=0):
    """Build trials + ni_events that are aligned at `offset`.

    Every 2nd trial is a Hit (change presented); the rest alternate FA/abort.
    `n_pad` prepends unrelated events so the true alignment is at index n_pad.
    """
    rng = np.random.default_rng(0)
    trials, bon, con = [], [], []
    for _ in range(n_pad):                      # orphan events from earlier runs
        t0 = len(bon) * 10.0
        bon.append(t0)
        con.append(t0 + 5.0)
    for i in range(n):
        t0 = (n_pad + i) * 10.0
        ct = round(float(rng.uniform(6.0, 11.0)), 3)
        if i % 2 == 0:
            outcome = "Hit"
            con.append(t0 + ct)                 # change WAS presented
        else:
            outcome = "FA" if i % 4 == 1 else "abort"
            con.append(np.nan)                  # change never presented
        bon.append(t0)
        trials.append(FakeTrial(outcome, ct))
    ni = {
        "Baseline_ON": np.array(bon, float),
        "Change_ON": np.array(con, float),
        "Valve_L": np.zeros(len(bon), float),
        "Rot_enc_A": np.zeros(9999, float),     # NOT per-trial
    }
    return trials, ni


def test_per_trial_event_keys_finds_equal_length_arrays():
    trials, ni = make_case(n=40)
    keys = per_trial_event_keys(ni)
    assert set(keys) == {"Baseline_ON", "Change_ON", "Valve_L"}
    assert "Rot_enc_A" not in keys


def test_agreement_is_one_at_correct_offset_and_chance_when_shifted():
    trials, ni = make_case(n=60, n_pad=25)
    good, n_cmp = outcome_change_agreement(trials, ni, slice(None), 25)
    assert good == pytest.approx(1.0)
    assert n_cmp == 60                      # 100% trial coverage
    bad, _ = outcome_change_agreement(trials, ni, slice(None), 24)
    assert bad < 0.95                       # a single-trial shift breaks it


def test_residual_is_zero_at_correct_offset_and_large_when_shifted():
    trials, ni = make_case(n=60, n_pad=25)
    med, n = alignment_residual(trials, ni, slice(None), 25)
    assert med == pytest.approx(0.0, abs=1e-9)
    assert n == 30                          # only the Hit trials
    med_bad, _ = alignment_residual(trials, ni, slice(None), 24)
    assert med_bad > 0.5


def test_residual_rejects_when_too_few_finite_trials():
    """n < MIN_RESID_N must yield nan (reject), never a vacuous pass."""
    trials, ni = make_case(n=10)            # only 5 Hit trials
    med, n = alignment_residual(trials, ni, slice(None), 0)
    assert n < MIN_RESID_N
    assert np.isnan(med)


def test_outcome_set_is_case_sensitive_and_includes_ref():
    assert CHANGE_PRESENTED_OUTCOMES == frozenset({"Hit", "Miss", "Ref"})
    assert "hit" not in CHANGE_PRESENTED_OUTCOMES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_run_alignment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'visdetect.core.run_alignment'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/visdetect/core/run_alignment.py
"""Trial <-> ni_events alignment scoring and solving (QC1).

A recording's per-trial NI arrays (Baseline_ON, Change_ON, Valve_L) must be
index-aligned to the trial table. They are not, on 17 sessions, because the
converter loads whatever *trials.json files sit in Session/ without checking
they belong to that recording. See
docs/superpowers/specs/2026-08-03-QC1-trial-event-alignment-repair-design.md

Two checks score a candidate pairing:
  Check 1 (primary, 100% trial coverage): isfinite(Change_ON) must agree with
          "was a change presented", i.e. outcome in {Hit, Miss, Ref}.
  Check 2 (secondary, precision): (Change_ON - Baseline_ON) must equal the
          trial's scheduled change_time -- only on trials where the change was
          actually presented (~45%).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

# CASE-SENSITIVE. Real pkl labels are capitalised: Hit/Miss/FA/abort/Ref.
# Do NOT refactor onto EVENT_VALID_OUTCOMES -- that is lowercase and omits Ref.
CHANGE_PRESENTED_OUTCOMES = frozenset({"Hit", "Miss", "Ref"})

ACCEPT_AGREEMENT = 1.0      # Check 1: exact, no tolerance
ACCEPT_RESID_S = 0.05       # Check 2: 10x above the observed 0.0051 s aligned value
MIN_RESID_N = 20            # below this Check 2 is not evaluable -> REJECT

_REQUIRED_KEYS = ("Baseline_ON", "Change_ON")


def _arr(x: Any) -> np.ndarray:
    if isinstance(x, dict) and "rise_t" in x:
        return np.asarray(x["rise_t"], dtype=float).ravel()
    if x is None:
        return np.zeros(0, dtype=float)
    return np.asarray(x, dtype=float).ravel()


def per_trial_event_keys(ni_events: Dict[str, Any]) -> List[str]:
    """Event keys whose arrays have one entry per recorded trial.

    Defined as: same length as Baseline_ON (which is per-trial by construction).
    """
    ni_events = ni_events or {}
    n = len(_arr(ni_events.get("Baseline_ON")))
    if n == 0:
        return []
    keys = []
    for k, v in ni_events.items():
        if k == "session_name":
            continue
        try:
            if len(_arr(v)) == n:
                keys.append(k)
        except Exception:
            continue
    return sorted(keys)


def _trial_fields(trials: Sequence[Any], trial_slice: slice):
    sub = list(trials)[trial_slice]
    outcomes = np.array([str(getattr(t, "trialoutcome", "") or "") for t in sub])
    ct = np.array(
        [
            float(getattr(t, "change_time", np.nan))
            if getattr(t, "change_time", None) is not None
            else np.nan
            for t in sub
        ],
        dtype=float,
    )
    return outcomes, ct


def outcome_change_agreement(
    trials: Sequence[Any], ni_events: Dict[str, Any], trial_slice: slice, event_offset: int
) -> Tuple[float, int]:
    """Check 1. Fraction of trials where change-presence agrees with the outcome label.

    Returns (agreement, n_compared). Returns (nan, 0) if the candidate does not fit.
    """
    outcomes, _ = _trial_fields(trials, trial_slice)
    n = len(outcomes)
    con = _arr((ni_events or {}).get("Change_ON"))
    if n == 0 or event_offset < 0 or event_offset + n > len(con):
        return float("nan"), 0
    observed = np.isfinite(con[event_offset : event_offset + n])
    expected = np.isin(outcomes, list(CHANGE_PRESENTED_OUTCOMES))
    return float(np.mean(observed == expected)), int(n)


def alignment_residual(
    trials: Sequence[Any], ni_events: Dict[str, Any], trial_slice: slice, event_offset: int
) -> Tuple[float, int]:
    """Check 2. Median |(Change_ON - Baseline_ON) - change_time| in seconds.

    Scored ONLY over trials whose scheduled change was actually presented.
    Returns (nan, n) when n < MIN_RESID_N -- an empty/thin residual set is a
    REJECT, never a pass.
    """
    _, ct = _trial_fields(trials, trial_slice)
    n = len(ct)
    ni = ni_events or {}
    bon = _arr(ni.get("Baseline_ON"))
    con = _arr(ni.get("Change_ON"))
    if n == 0 or event_offset < 0 or event_offset + n > min(len(bon), len(con)):
        return float("nan"), 0
    sl = slice(event_offset, event_offset + n)
    resid = (con[sl] - bon[sl]) - ct
    finite = np.isfinite(resid)
    n_fin = int(finite.sum())
    if n_fin < MIN_RESID_N:
        return float("nan"), n_fin
    return float(np.median(np.abs(resid[finite]))), n_fin
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_run_alignment.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/core/run_alignment.py tests/test_run_alignment.py
git commit -m "feat(QC1): alignment scoring primitive (Check 1 + Check 2)"
```

---

### Task 2: Brute-force solver with uniqueness reporting

**Files:**
- Modify: `src/visdetect/core/run_alignment.py`
- Test: `tests/test_run_alignment.py`

**Interfaces:**
- Consumes: Task 1's `outcome_change_agreement`, `alignment_residual`, `ACCEPT_*`, `MIN_RESID_N`.
- Produces:
  - `@dataclass Alignment` with fields `trial_start: int`, `n_trials_matched: int`, `event_offset: int`, `agreement: float`, `resid_s: float`, `resid_n: int`, `runner_up_agreement: float`, `runner_up_resid_s: float`
  - `solve_alignment(trials, ni_events) -> Optional[Alignment]`
  - `build_trial_event_index(n_trials: int, alignment: Optional[Alignment]) -> np.ndarray` (int array, `-1` where no event)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_run_alignment.py
from visdetect.core.run_alignment import Alignment, build_trial_event_index, solve_alignment


def test_solver_finds_sign_b_offset():
    """Sign B: trials are correct, events are untrimmed -> a pure offset."""
    trials, ni = make_case(n=60, n_pad=25)
    a = solve_alignment(trials, ni)
    assert a is not None
    assert a.event_offset == 25
    assert a.trial_start == 0
    assert a.agreement == pytest.approx(1.0)
    assert a.resid_s < 0.05


def test_solver_finds_sign_a_trial_slice():
    """Sign A: trial table spans runs; only a suffix belongs to this recording."""
    trials_run2, ni = make_case(n=60)
    trials_run1, _ = make_case(n=40)
    trials = trials_run1 + trials_run2          # concatenated, as the converter does
    a = solve_alignment(trials, ni)
    assert a is not None
    assert a.trial_start == 40
    assert a.n_trials_matched == 60
    assert a.event_offset == 0


def test_solver_returns_none_when_unsolvable():
    trials, ni = make_case(n=60)
    ni["Change_ON"] = np.full(len(ni["Change_ON"]), np.nan)   # no usable evidence
    assert solve_alignment(trials, ni) is None


def test_solver_returns_none_for_empty_trial_table():
    _, ni = make_case(n=60)
    assert solve_alignment([], ni) is None


def test_build_trial_event_index_maps_and_marks_missing():
    trials_run1, _ = make_case(n=40)
    trials_run2, ni = make_case(n=60)
    trials = trials_run1 + trials_run2
    a = solve_alignment(trials, ni)
    idx = build_trial_event_index(len(trials), a)
    assert idx.shape == (100,)
    assert (idx[:40] == -1).all()               # run-1 trials have no ephys here
    assert (idx[40:] == np.arange(60)).all()


def test_build_trial_event_index_all_minus_one_when_unsolved():
    idx = build_trial_event_index(7, None)
    assert (idx == -1).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_run_alignment.py -v`
Expected: FAIL — `ImportError: cannot import name 'Alignment'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/visdetect/core/run_alignment.py`:

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class Alignment:
    """A verified pairing of a contiguous trial block to a contiguous event block."""

    trial_start: int
    n_trials_matched: int
    event_offset: int
    agreement: float
    resid_s: float
    resid_n: int
    runner_up_agreement: float = float("nan")
    runner_up_resid_s: float = float("nan")


def _passes(agreement: float, resid_s: float) -> bool:
    return (
        np.isfinite(agreement)
        and agreement >= ACCEPT_AGREEMENT
        and np.isfinite(resid_s)
        and resid_s < ACCEPT_RESID_S
    )


def solve_alignment(trials: Sequence[Any], ni_events: Dict[str, Any]) -> Optional[Alignment]:
    """Brute-force search for the unique (trial_start, event_offset) pairing.

    NOTE: this operates on a built pkl, where the per-run JSON boundaries are no
    longer available -- so the search is exhaustive by construction. The
    converter has a different, JSON-informed path (see ingest.py).

    Search space:
      sign B  -> trial_start = 0, event_offset varies   (events outnumber trials)
      sign A  -> event_offset = 0, trial_start varies   (trials outnumber events)
    Both reduce to matching a contiguous trial block against a contiguous event
    block; we scan whichever dimension has slack.
    """
    trials = list(trials or [])
    n_tr = len(trials)
    ni = ni_events or {}
    n_ev = len(_arr(ni.get("Baseline_ON")))
    if n_tr == 0 or n_ev == 0:
        return None
    for key in _REQUIRED_KEYS:
        if len(_arr(ni.get(key))) != n_ev:
            return None

    candidates = []
    if n_ev >= n_tr:
        # sign B: whole trial table fits; slide it along the event arrays
        for off in range(0, n_ev - n_tr + 1):
            candidates.append((0, n_tr, off))
    else:
        # sign A: whole event array is covered; slide the trial window
        for start in range(0, n_tr - n_ev + 1):
            candidates.append((start, n_ev, 0))

    scored = []
    for start, n_match, off in candidates:
        sl = slice(start, start + n_match)
        agr, _ = outcome_change_agreement(trials, ni, sl, off)
        if not np.isfinite(agr):
            continue
        res, res_n = alignment_residual(trials, ni, sl, off)
        scored.append((agr, res, res_n, start, n_match, off))

    if not scored:
        return None

    # rank on Check 1 (full coverage) first, then Check 2 (precision)
    scored.sort(key=lambda r: (-r[0], r[1] if np.isfinite(r[1]) else np.inf))
    best = scored[0]
    runner = scored[1] if len(scored) > 1 else None
    if not _passes(best[0], best[1]):
        return None

    return Alignment(
        trial_start=best[3],
        n_trials_matched=best[4],
        event_offset=best[5],
        agreement=best[0],
        resid_s=best[1],
        resid_n=best[2],
        runner_up_agreement=runner[0] if runner else float("nan"),
        runner_up_resid_s=runner[1] if runner else float("nan"),
    )


def build_trial_event_index(n_trials: int, alignment: Optional[Alignment]) -> np.ndarray:
    """Per-trial map into the per-trial ni_events arrays. -1 = no ephys event."""
    idx = np.full(int(n_trials), -1, dtype=int)
    if alignment is None:
        return idx
    a = alignment
    idx[a.trial_start : a.trial_start + a.n_trials_matched] = np.arange(
        a.event_offset, a.event_offset + a.n_trials_matched, dtype=int
    )
    return idx
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_run_alignment.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/core/run_alignment.py tests/test_run_alignment.py
git commit -m "feat(QC1): brute-force alignment solver + trial_event_index builder"
```

---

### Task 3: Validate the solver against the three real BG_046 sessions

This task proves the primitive on real data before anything mutates a pkl. It is the Phase-1 gate.

**Files:**
- Create: `tests/test_run_alignment_realdata.py`

**Interfaces:**
- Consumes: Task 2's `solve_alignment`.
- Produces: nothing consumed later; a gate.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_alignment_realdata.py
"""Real-pkl validation of the QC1 alignment solver.

Skips when the pkls are absent (data/ is gitignored), so CI stays green.
Expected values come from the spec's measured table (§2).
"""
import gc
import os

import numpy as np
import pytest

from visdetect.core.run_alignment import solve_alignment
from visdetect.core.session import load_session

PKL_DIR = os.path.join("data", "pkls", "BG_046")

CASES = [
    # file,                    trial_start, event_offset, n_matched
    ("BG_046_19082025.pkl",    0,   0,   587),   # known good -> identity
    ("BG_046_20082025.pkl",    0,   228, 486),   # sign B: untrimmed ephys
    ("BG_046_05092025_b.pkl",  281, 0,   248),   # sign A: concatenated runs
]


@pytest.mark.parametrize("fname,exp_start,exp_off,exp_n", CASES)
def test_solver_recovers_known_alignment(fname, exp_start, exp_off, exp_n):
    path = os.path.join(PKL_DIR, fname)
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    s = load_session(path)
    try:
        a = solve_alignment(s.trials, s.ni_events)
        assert a is not None, f"{fname}: solver failed to find an alignment"
        assert a.trial_start == exp_start
        assert a.event_offset == exp_off
        assert a.n_trials_matched == exp_n
        assert a.agreement == pytest.approx(1.0)
        assert a.resid_s < 0.05
        # uniqueness: the runner-up must NOT also pass
        assert not (a.runner_up_agreement >= 1.0 and a.runner_up_resid_s < 0.05)
    finally:
        del s
        gc.collect()
```

- [ ] **Step 2: Run test to verify it fails or skips**

Run: `py -m pytest tests/test_run_alignment_realdata.py -v`
Expected: 3 PASS if the pkls are present. If any FAIL, **stop and report** — the primitive does not reproduce the spec's measured result and the plan cannot proceed.

- [ ] **Step 3: No implementation needed**

This task is a gate over Task 2's code. If it fails, fix `run_alignment.py`, not the test.

- [ ] **Step 4: Commit**

```bash
git add tests/test_run_alignment_realdata.py
git commit -m "test(QC1): solver reproduces the three measured BG_046 alignments"
```

---

### Task 4: `Session.trial_event_index` field with proven backwards compatibility

**Files:**
- Modify: `src/visdetect/core/session.py:44-51` (the `Session` dataclass)
- Test: `tests/test_session_trial_event_index.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Session.trial_event_index: Optional[np.ndarray] = None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_session_trial_event_index.py
"""The new field must not break the ~253 existing pkls, which lack it."""
import pickle

import numpy as np

from visdetect.core.session import Session


def test_field_defaults_to_none():
    s = Session()
    assert s.trial_event_index is None


def test_old_pickle_without_the_field_still_loads_and_reads_none():
    """Simulate an existing pkl: pickle a Session, strip the key, unpickle."""
    s = Session(subject="BG_046", session_name="01072025")
    raw = pickle.loads(pickle.dumps(s))
    del raw.__dict__["trial_event_index"]          # what an old pkl looks like
    revived = pickle.loads(pickle.dumps(raw))
    # A plain None default makes this safe; default_factory would AttributeError.
    assert getattr(revived, "trial_event_index", None) is None


def test_field_round_trips_an_array():
    s = Session()
    s.trial_event_index = np.array([-1, -1, 0, 1, 2], dtype=int)
    back = pickle.loads(pickle.dumps(s))
    assert np.array_equal(back.trial_event_index, np.array([-1, -1, 0, 1, 2]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_session_trial_event_index.py -v`
Expected: FAIL — `AttributeError: 'Session' object has no attribute 'trial_event_index'`

- [ ] **Step 3: Write minimal implementation**

In `src/visdetect/core/session.py`, add as the **last** field of `Session` (after `ni_events`, so it stays a defaulted field and the module still imports):

```python
    # QC1: per-trial map into the per-trial ni_events arrays (Baseline_ON,
    # Change_ON, Valve_L). -1 = this trial has no corresponding ephys event.
    # None = alignment not yet verified for this session.
    # MUST be a plain None default: field(default_factory=...) leaves the key
    # out of __dict__ on pkls written before this field existed, so attribute
    # access would raise AttributeError.
    trial_event_index: Optional[np.ndarray] = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_session_trial_event_index.py tests/test_session.py tests/test_session_io.py -v`
Expected: PASS — the two existing session test modules must stay green.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/core/session.py tests/test_session_trial_event_index.py
git commit -m "feat(QC1): add Session.trial_event_index with None default"
```

---

### Task 5: Repair script

**Files:**
- Create: `scripts/QC_technical/repair_trial_event_alignment.py`
- Test: `tests/test_repair_trial_event_alignment.py`

**Interfaces:**
- Consumes: Task 2 (`solve_alignment`, `build_trial_event_index`), Task 4 (the field).
- Produces:
  - `backup_pkl(path: str) -> str` (returns backup path; raises on failure)
  - `repair_session(path: str, dry_run: bool = False) -> dict` (one report row)
  - CLI: `py scripts/QC_technical/repair_trial_event_alignment.py --subjects BG_046 [--dry-run]`
  - Output: `data/cache/qc_alignment/alignment_repair_report.csv`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repair_trial_event_alignment.py
import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "QC_technical"))
from repair_trial_event_alignment import backup_pkl, repair_session  # noqa: E402

from visdetect.core.session import Session, Trial


def _make_pkl(tmp_path, n_pad=25, n=60):
    trials, bon, con = [], [], []
    rng = np.random.default_rng(0)
    for _ in range(n_pad):
        t0 = len(bon) * 10.0
        bon.append(t0); con.append(t0 + 5.0)
    for i in range(n):
        t0 = (n_pad + i) * 10.0
        ct = round(float(rng.uniform(6.0, 11.0)), 3)
        if i % 2 == 0:
            con.append(t0 + ct); outcome = "Hit"
        else:
            con.append(np.nan); outcome = "FA"
        bon.append(t0)
        trials.append(Trial(trialoutcome=outcome, change_time=ct))
    s = Session(trials=trials, subject="BG_TEST", session_name="01012025")
    s.ni_events = {
        "Baseline_ON": np.array(bon, float),
        "Change_ON": np.array(con, float),
        "Valve_L": np.zeros(len(bon), float),
    }
    p = tmp_path / "BG_TEST_01012025.pkl"
    with open(p, "wb") as f:
        pickle.dump(s, f)
    return str(p)


def test_backup_is_written_before_mutation(tmp_path):
    p = _make_pkl(tmp_path)
    b = backup_pkl(p)
    assert os.path.exists(b)
    assert "qc1_backup" in b


def test_repair_writes_index_map_and_preserves_behaviour(tmp_path):
    p = _make_pkl(tmp_path, n_pad=25, n=60)
    with open(p, "rb") as f:
        before = pickle.load(f)
    outcomes_before = [t.trialoutcome for t in before.trials]

    row = repair_session(p)

    assert row["solved"] is True
    assert row["event_offset"] == 25
    assert row["agreement"] == pytest.approx(1.0)

    with open(p, "rb") as f:
        after = pickle.load(f)
    # behaviour untouched
    assert [t.trialoutcome for t in after.trials] == outcomes_before
    assert len(after.trials) == len(before.trials)
    # map correct
    assert np.array_equal(after.trial_event_index, np.arange(25, 85))


def test_dry_run_does_not_mutate(tmp_path):
    p = _make_pkl(tmp_path)
    row = repair_session(p, dry_run=True)
    assert row["solved"] is True
    with open(p, "rb") as f:
        after = pickle.load(f)
    assert getattr(after, "trial_event_index", None) is None


def test_unsolvable_session_gets_all_minus_one(tmp_path):
    p = _make_pkl(tmp_path)
    with open(p, "rb") as f:
        s = pickle.load(f)
    s.ni_events["Change_ON"] = np.full(len(s.ni_events["Change_ON"]), np.nan)
    with open(p, "wb") as f:
        pickle.dump(s, f)

    row = repair_session(p)
    assert row["solved"] is False
    with open(p, "rb") as f:
        after = pickle.load(f)
    assert (after.trial_event_index == -1).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_repair_trial_event_alignment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'repair_trial_event_alignment'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/QC_technical/repair_trial_event_alignment.py
"""QC1: write a verified trial->event index map into each pkl.

Never truncates the trial table: trials with no ephys event get -1, so
behaviour-only analyses keep every trial while neural code hard-skips them.
Always backs up before mutating.

Run: py scripts/QC_technical/repair_trial_event_alignment.py --subjects BG_046
Out: data/cache/qc_alignment/alignment_repair_report.csv
"""
import argparse
import gc
import glob
import os
import pickle
import shutil
import sys
from datetime import datetime, timezone

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np
import pandas as pd

from visdetect.core.run_alignment import build_trial_event_index, solve_alignment
from visdetect.core.session import load_session

OUT_DIR = os.path.join(_ROOT, "data", "cache", "qc_alignment")
OUT_CSV = os.path.join(OUT_DIR, "alignment_repair_report.csv")


def backup_pkl(path: str) -> str:
    """Copy the pkl into <dir>/qc1_backup/ with a UTC stamp. Raises on failure."""
    d = os.path.join(os.path.dirname(path), "qc1_backup")
    os.makedirs(d, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = os.path.join(d, f"{os.path.basename(path)}.bak_{stamp}")
    shutil.copy2(path, dest)
    if not os.path.exists(dest):
        raise IOError(f"backup failed for {path}")
    return dest


def repair_session(path: str, dry_run: bool = False) -> dict:
    s = load_session(path)
    try:
        n_tr = len(s.trials or [])
        a = solve_alignment(s.trials, s.ni_events)
        row = {
            "file": os.path.basename(path),
            "n_trials": n_tr,
            "solved": a is not None,
            "trial_start": a.trial_start if a else -1,
            "n_matched": a.n_trials_matched if a else 0,
            "event_offset": a.event_offset if a else -1,
            "agreement": a.agreement if a else float("nan"),
            "resid_s": a.resid_s if a else float("nan"),
            "resid_n": a.resid_n if a else 0,
            "runner_up_agreement": a.runner_up_agreement if a else float("nan"),
            "runner_up_resid_s": a.runner_up_resid_s if a else float("nan"),
            "n_no_ephys": 0,
        }
        idx = build_trial_event_index(n_tr, a)
        row["n_no_ephys"] = int((idx == -1).sum())
        if dry_run:
            return row

        outcomes_before = [getattr(t, "trialoutcome", None) for t in (s.trials or [])]
        backup_pkl(path)
        s.trial_event_index = idx
        with open(path, "wb") as f:
            pickle.dump(s, f, protocol=pickle.HIGHEST_PROTOCOL)

        # behaviour must be byte-identical
        chk = load_session(path)
        try:
            assert [getattr(t, "trialoutcome", None) for t in (chk.trials or [])] == outcomes_before, (
                f"{path}: trial outcomes changed during repair"
            )
        finally:
            del chk
        return row
    finally:
        del s
        gc.collect()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=["BG_046"])
    ap.add_argument("--files", nargs="*", default=None, help="explicit pkl basenames")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = []
    for subj in args.subjects:
        for p in sorted(glob.glob(os.path.join(_ROOT, "data", "pkls", subj, f"{subj}_*.pkl"))):
            if args.files and os.path.basename(p) not in args.files:
                continue
            rec = {"subject": subj}
            rec.update(repair_session(p, dry_run=args.dry_run))
            rows.append(rec)
            print(f"  {subj} {rec['file']}: solved={rec['solved']} "
                  f"start={rec['trial_start']} off={rec['event_offset']} "
                  f"agr={rec['agreement']:.4f} resid={rec['resid_s']:.4f}")

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    if not args.dry_run:
        df.to_csv(OUT_CSV, index=False)
        print(f"\nSaved: {OUT_CSV}")
    print(f"solved {int(df['solved'].sum())}/{len(df)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_repair_trial_event_alignment.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Dry-run against the two real BG_046 sessions**

```bash
py scripts/QC_technical/repair_trial_event_alignment.py --subjects BG_046 \
   --files BG_046_20082025.pkl BG_046_05092025_b.pkl --dry-run
```
Expected: both `solved=True`; `20082025` → `off=228`, `05092025_b` → `start=281`; both `agr=1.0000`, `resid≈0.0051`.
**If either is False, stop and report.**

- [ ] **Step 6: Commit**

```bash
git add scripts/QC_technical/repair_trial_event_alignment.py tests/test_repair_trial_event_alignment.py
git commit -m "feat(QC1): repair script writing trial_event_index with backup + behaviour check"
```

---

### Task 6: Extend the audit to measured alignment

**Files:**
- Modify: `scripts/QC_technical/audit_trial_baselineon_alignment.py`
- Test: `tests/test_audit_alignment_columns.py`

**Interfaces:**
- Consumes: Task 2's `solve_alignment`.
- Produces: audit CSV gains `agreement`, `median_resid_s`, `resid_n`, `runner_up_resid_s`, `aligned`. `neural_safe` becomes `aligned`-based, with the old count check kept as `count_safe`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_audit_alignment_columns.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "QC_technical"))
import audit_trial_baselineon_alignment as audit  # noqa: E402


def test_audit_exposes_measured_columns():
    assert hasattr(audit, "audit_pkl")
    assert hasattr(audit, "TOL_BENIGN")
    # the measured columns the repair depends on
    for col in ("agreement", "median_resid_s", "resid_n", "aligned"):
        assert col in audit.MEASURED_COLUMNS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_audit_alignment_columns.py -v`
Expected: FAIL — `AttributeError: module has no attribute 'MEASURED_COLUMNS'`

- [ ] **Step 3: Write minimal implementation**

In `scripts/QC_technical/audit_trial_baselineon_alignment.py`:

Add near the top, after `TOL_BENIGN = 9`:

```python
from visdetect.core.run_alignment import solve_alignment

MEASURED_COLUMNS = ("agreement", "median_resid_s", "resid_n", "runner_up_resid_s", "aligned")
```

In `audit_pkl`, extend the returned dict (keep every existing key):

```python
        a = solve_alignment(s.trials, s.ni_events)
        measured = {
            "agreement": a.agreement if a else float("nan"),
            "median_resid_s": a.resid_s if a else float("nan"),
            "resid_n": a.resid_n if a else 0,
            "runner_up_resid_s": a.runner_up_resid_s if a else float("nan"),
            "aligned": a is not None,
            "trial_start": a.trial_start if a else -1,
            "event_offset": a.event_offset if a else -1,
        }
        return {"n_trials": n, "n_baseline_on": int(len(bon)), "diff": int(len(bon) - n),
                "ephys_s": round(max(spikes), 1) if spikes else np.nan,
                "bon_last": round(float(bon.max()), 1) if len(bon) else np.nan,
                **measured}
```

In `main()`, replace the two derived columns:

```python
    df["match"] = df["diff"] == 0
    df["count_safe"] = df["diff"].abs() <= TOL_BENIGN     # old proxy, retained
    df["neural_safe"] = df["aligned"].fillna(False).astype(bool)   # measured
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_audit_alignment_columns.py -v`
Expected: PASS

- [ ] **Step 5: Run the full audit and check the benign band**

```bash
py scripts/QC_technical/audit_trial_baselineon_alignment.py
```
Expected: 253 pkls. The 48 `|diff| ∈ [1,9]` benign sessions should now be **measured** rather than assumed. Record how many of them come back `aligned=True` — this is new information the spec predicted but did not have.

- [ ] **Step 6: Commit**

```bash
git add scripts/QC_technical/audit_trial_baselineon_alignment.py tests/test_audit_alignment_columns.py
git add -f data/cache/qc_alignment/trial_vs_baselineon_audit.csv
git commit -m "feat(QC1): audit reports measured alignment, not count proxy"
```

---

### Task 7: Enforce the map in `align.py`

**Files:**
- Modify: `src/visdetect/analysis/align.py:300-361` (`get_event_times_by_trial`), `:136` (`get_event_times`)
- Test: `tests/test_align_honours_trial_event_index.py`

**Interfaces:**
- Consumes: Task 4's field.
- Produces: `get_event_times_by_trial` returns NaN for `-1` trials and reads `arr[trial_event_index[i]]` otherwise. Behaviour is unchanged when the field is `None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_align_honours_trial_event_index.py
import numpy as np

from visdetect.analysis.align import get_event_times_by_trial
from visdetect.core.session import Session, Trial


def _session(map_=None):
    trials = [Trial(trialoutcome=o, change_time=5.0) for o in ("Hit", "Hit", "Hit")]
    s = Session(trials=trials)
    s.ni_events = {
        "Baseline_ON": np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
        "Change_ON": np.array([105.0, 205.0, 305.0, 405.0, 505.0]),
    }
    s.trial_event_index = map_
    return s


def test_without_map_behaviour_is_unchanged():
    out = get_event_times_by_trial(_session(None), "Baseline_ON")
    assert out == [100.0, 200.0, 300.0]          # legacy prefix pairing


def test_map_offsets_the_lookup():
    s = _session(np.array([2, 3, 4]))
    assert get_event_times_by_trial(s, "Baseline_ON") == [300.0, 400.0, 500.0]


def test_minus_one_trials_are_nan_not_backfilled():
    """A -1 trial has no ephys; Change_ON must NOT be reconstructed from change_time."""
    s = _session(np.array([-1, 3, 4]))
    out = get_event_times_by_trial(s, "Change_ON")
    assert np.isnan(out[0])
    assert out[1] == 405.0
    assert out[2] == 505.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_align_honours_trial_event_index.py -v`
Expected: FAIL — `test_map_offsets_the_lookup` returns `[100.0, 200.0, 300.0]`

- [ ] **Step 3: Write minimal implementation**

In `get_event_times_by_trial`, replace the prefix-pairing block at `align.py:302-308`:

```python
    if event_name in ["Baseline_ON", "Change_ON"]:
        ev = ni_events.get(event_name, None)
        arr = _to_array(ev)
        tei = getattr(session, "trial_event_index", None)
        if tei is not None:
            # QC1: honour the verified trial->event map.
            tei = _np.asarray(tei, dtype=int).ravel()
            for idx in range(min(n_trials, tei.size)):
                j = int(tei[idx])
                if 0 <= j < len(arr):
                    out[idx] = arr[j]
            # -1 trials have no ephys: leave NaN and do NOT backfill below.
            no_ephys = _np.zeros(n_trials, dtype=bool)
            no_ephys[: tei.size] = tei < 0
        else:
            m = min(len(arr), n_trials)
            if m > 0:
                out[:m] = arr[:m]
            no_ephys = _np.zeros(n_trials, dtype=bool)
```

Then guard the Change_ON backfill loop (`align.py:314` onward) so a no-ephys trial is skipped **before** the `change_time` fill:

```python
            for idx in range(n_trials):
                if no_ephys[idx]:
                    continue        # QC1: no ephys for this trial -> stay NaN
                if _np.isnan(out[idx]):
```

Apply the same `no_ephys` guard to the second `out[:m] = arr[:m]` at `align.py:359` (the behavioural-outcome branch), so `-1` trials return NaN there too.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_align_honours_trial_event_index.py tests/test_align.py -v`
Expected: PASS — `tests/test_align.py` must stay green (backwards compatibility).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/align.py tests/test_align_honours_trial_event_index.py
git commit -m "feat(QC1): align.py honours trial_event_index; -1 trials never backfilled"
```

---

### Task 8: Patch the direct `ni_events` consumer `tf_glm_data.py`

This is the highest-stakes consumer: it bypasses `align.py` entirely and is the validated TF-encoding GLM already run on all three mice. Contamination is measured in the spec (BG_031 VMS: 7/42 sessions, `resp_log2` 1.26% on affected vs 6.31% clean).

**Files:**
- Modify: `src/visdetect/analysis/tf_glm_data.py:524-546` (line numbers are **post-merge with main**, which brought in the lick-channel fix — `_collect_lick_times` now delegates to `lick_channels.get_lick_times`. Do not revert that.)
- Test: `tests/test_tf_glm_data_alignment.py`

> **Not in scope for this task:** the separate builder at `tf_glm_data.py:321-345` reads `ks.trials`
> — a pandas table whose `Baseline_ON`/`Change_ON` are already columns, aligned within that table by
> construction. It is a different ingestion path, not the `session.ni_events` positional bug. Its
> alignment is inherited from whatever built that table, so it is a **Task 9 audit item**, not an
> assumption. Do not "fix" it here.

**Interfaces:**
- Consumes: Task 4's field.
- Produces: `_event_index_for(session, i) -> int` helper; trial windows and change times read through the map.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tf_glm_data_alignment.py
"""tf_glm_data pairs trial i with event i. On misaligned sessions that convolves
one trial's stimulus against another trial's spikes. It must use the map."""
import numpy as np

from visdetect.analysis.tf_glm_data import _event_index_for
from visdetect.core.session import Session, Trial


def _session(map_=None):
    s = Session(trials=[Trial(trialoutcome="Hit", change_time=5.0) for _ in range(3)])
    s.ni_events = {
        "Baseline_ON": np.arange(5, dtype=float) * 100.0,
        "Change_ON": np.arange(5, dtype=float) * 100.0 + 5.0,
    }
    s.trial_event_index = map_
    return s


def test_identity_when_no_map():
    s = _session(None)
    assert [_event_index_for(s, i) for i in range(3)] == [0, 1, 2]


def test_offset_applied_when_map_present():
    s = _session(np.array([2, 3, 4]))
    assert [_event_index_for(s, i) for i in range(3)] == [2, 3, 4]


def test_minus_one_returns_negative():
    s = _session(np.array([-1, 3, 4]))
    assert _event_index_for(s, 0) < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_tf_glm_data_alignment.py -v`
Expected: FAIL — `ImportError: cannot import name '_event_index_for'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/visdetect/analysis/tf_glm_data.py` above the builder function:

```python
def _event_index_for(session, i: int) -> int:
    """QC1: map trial index -> per-trial ni_events index. <0 means no ephys."""
    tei = getattr(session, "trial_event_index", None)
    if tei is None:
        return i
    tei = np.asarray(tei, dtype=int).ravel()
    return int(tei[i]) if i < tei.size else -1
```

Then in the trial loop, replace the positional reads (`tf_glm_data.py:539` and `:546`):

```python
    for i, trial in enumerate(session.trials):
        j = _event_index_for(session, i)
        if j < 0:
            # QC1: this trial has no ephys event -- emit an empty window so it
            # contributes nothing, rather than silently pairing with event i.
            trials_regs.append(TrialRegressors(edges=np.zeros(0, float)))
            continue
        t0 = float(bon[j]) if j < bon.size else np.nan
        t1 = float(ends[j]) if j < ends.size else np.nan
```

and

```python
        raw_change = float(con[j]) if (j < con.size and np.isfinite(con[j])) else np.nan
```

Also fix the `ends` computation at `tf_glm_data.py:524-535`, which is indexed by trial index. Replace `n = len(session.trials)` with the event count and drop the `i >= n` guard so `ends` is an **event**-indexed array:

```python
    n_ev = int(bon.size)
    order = np.argsort(np.where(np.isfinite(bon), bon, np.inf)) if bon.size else np.zeros(0, int)
    ends = np.full(n_ev, np.nan)
    for k in range(order.size):
        i_ev = order[k]
        if k + 1 < order.size:
            ends[i_ev] = bon[order[k + 1]]
        else:
            ends[i_ev] = bon[i_ev] + 20.0 if np.isfinite(bon[i_ev]) else np.nan
```

> If `TrialRegressors` cannot be constructed with only `edges`, construct it with the same zero-length arrays the existing code uses for an empty trial — read the dataclass definition in the same file and mirror it. Do not invent fields.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_tf_glm_data_alignment.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/tf_glm_data.py tests/test_tf_glm_data_alignment.py
git commit -m "fix(QC1): tf_glm_data honours trial_event_index instead of positional pairing"
```

---

### Task 9: Systematic consumer audit

`tf_glm_data.py` was found by review, not by search. ~120 files touch `ni_events`; any other direct positional reader stays silently wrong.

**Files:**
- Create: `scripts/QC_technical/audit_ni_events_consumers.py`
- Create: `docs/science/QC1-ni-events-consumers.md`

**Interfaces:**
- Consumes: nothing.
- Produces: a report listing every file reading `Baseline_ON`/`Change_ON`/`Valve_L` without routing through `align`.

- [ ] **Step 1: Write the scanner**

```python
# scripts/QC_technical/audit_ni_events_consumers.py
"""QC1: find code that reads per-trial ni_events arrays WITHOUT going through align.py.

Such code pairs trial i with event i positionally and stays wrong on the 17
misaligned sessions even after the pkls are repaired.

Run: py scripts/QC_technical/audit_ni_events_consumers.py
"""
import os
import re
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PER_TRIAL = ("Baseline_ON", "Change_ON", "Valve_L")
ALIGN_MARKERS = ("get_event_times", "align_spikes_to_events", "trial_event_index")
SKIP_DIRS = {".git", ".venv", "__pycache__", "archive", "matlab_scripts", ".claude"}


def main():
    hits = []
    for root, dirs, files in os.walk(_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(root, fn)
            try:
                src = open(p, encoding="utf-8", errors="replace").read()
            except Exception:
                continue
            if not any(k in src for k in PER_TRIAL):
                continue
            routed = any(m in src for m in ALIGN_MARKERS)
            positional = re.search(r"\[\s*(i|idx|tr|trial_idx|k)\s*\]", src) is not None
            rel = os.path.relpath(p, _ROOT)
            hits.append((rel, routed, positional))

    unrouted = [h for h in hits if not h[1]]
    risky = [h for h in unrouted if h[2]]
    print(f"files touching per-trial ni_events: {len(hits)}")
    print(f"  routed through align/map      : {sum(1 for h in hits if h[1])}")
    print(f"  NOT routed                    : {len(unrouted)}")
    print(f"  NOT routed AND index by [i]   : {len(risky)}   <-- REVIEW THESE")
    for rel, _, _ in sorted(risky):
        print("   ", rel)
    return 0 if not risky else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it**

Run: `py scripts/QC_technical/audit_ni_events_consumers.py`
Expected: a list of files. `tf_glm_data.py` should now be **absent** from the risky list (Task 8 added `trial_event_index`).

- [ ] **Step 3: Triage every risky file**

For each file printed, open it and decide: (a) route through `align.get_event_times_by_trial`, (b) add `_event_index_for`-style mapping as in Task 8, or (c) document it as not per-trial-paired. Record the verdict per file in `docs/science/QC1-ni-events-consumers.md` as a table with columns `file | reads | verdict | action`.

> Do not batch-edit. Each file needs its own judgement and its own test if it is patched.

- [ ] **Step 4: Commit**

```bash
git add scripts/QC_technical/audit_ni_events_consumers.py docs/science/QC1-ni-events-consumers.md
git commit -m "chore(QC1): systematic audit of direct ni_events consumers"
```

---

### Task 10: Converter guard in `ingest.py`

**Files:**
- Modify: `src/visdetect/core/ingest.py:72-98` (`load_behavioral_trials`), and the pkl-emitting path in `build_session_from_raw`
- Test: `tests/test_ingest_alignment_guard.py`

**Interfaces:**
- Consumes: Task 2's `solve_alignment`, Task 2's `build_trial_event_index`.
- Produces: `load_behavioral_trials` orders runs by the filename-embedded timestamp; `build_session_from_raw` populates `trial_event_index` and refuses to emit an unverifiable pkl unless `allow_unaligned=True`.

**Critical constraint from the spec (§1, §4):** the glob **must stay non-recursive**. `Session/delete/` and `Session/partial/` hold runs deliberately curated out of the analysis set — recursing would re-inject 228 aborted/partial trials into `BG_046 20082025`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest_alignment_guard.py
import json
import os

from visdetect.core.ingest import _run_sort_key, load_behavioral_trials


def test_runs_are_ordered_by_filename_timestamp_not_mtime(tmp_path):
    """MATLAB ordered by mtime; reorganisation passes have since touched files."""
    d = tmp_path / "Session"
    d.mkdir()
    late = d / "BG_046_20250905_115246__trials.json"
    early = d / "BG_046_20250905_104819__trials.json"
    late.write_text(json.dumps([{"trialoutcome": "Hit"}]))
    early.write_text(json.dumps([{"trialoutcome": "Miss"}]))
    os.utime(early, (10**9, 10**9))          # make the EARLY run the NEWEST by mtime
    names = sorted([late.name, early.name], key=_run_sort_key)
    assert names[0] == early.name            # filename timestamp wins


def test_curated_subfolders_are_not_loaded(tmp_path):
    """delete/ and partial/ are curated out on purpose -- never recurse."""
    root = tmp_path / "BG_046_20082025"
    sess = root / "Session"
    (sess / "delete").mkdir(parents=True)
    (sess / "partial").mkdir(parents=True)
    (sess / "BG_046_20250820_121236__trials.json").write_text(
        json.dumps([{"trialoutcome": "Hit"}, {"trialoutcome": "Miss"}])
    )
    (sess / "delete" / "BG_046_20250820_111153__trials.json").write_text(
        json.dumps([{"trialoutcome": "abort"}])
    )
    (sess / "partial" / "BG_046_20250820_111747__trials.json").write_text(
        json.dumps([{"trialoutcome": "FA"}])
    )
    trials, _, _ = load_behavioral_trials(root)
    assert len(trials) == 2                  # NOT 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_ingest_alignment_guard.py -v`
Expected: FAIL — `ImportError: cannot import name '_run_sort_key'`

- [ ] **Step 3: Write minimal implementation**

In `src/visdetect/core/ingest.py`, add above `load_behavioral_trials`:

```python
import re

_RUN_STAMP = re.compile(r"_(\d{8})_(\d{6})__")


def _run_sort_key(name: str):
    """Order runs by the timestamp embedded in the FILENAME, never by mtime.

    The MATLAB pipeline sorted by [fname.datenum]; these directories have since
    been touched by reorganisation passes, so mtime no longer reflects run order.
    """
    m = _RUN_STAMP.search(str(name))
    return (m.group(1), m.group(2)) if m else ("", str(name))
```

Replace line 72:

```python
    # NON-RECURSIVE on purpose: Session/delete/ and Session/partial/ hold runs
    # curated out of the analysis set. Recursing would re-inject them.
    trial_files = sorted(session_dir.glob("*trials.json"), key=lambda p: _run_sort_key(p.name))
```

In `build_session_from_raw`, after `ni_events` and `trials` are both available and before the Session is returned:

```python
    from visdetect.core.run_alignment import build_trial_event_index, solve_alignment

    _align = solve_alignment(trials, ni_events)
    session.trial_event_index = build_trial_event_index(len(trials), _align)
    if _align is None and not allow_unaligned:
        raise ValueError(
            f"{raw_session_dir.name}: trial table could not be aligned to ni_events "
            f"({len(trials)} trials vs {len(np.asarray(ni_events.get('Baseline_ON', [])).ravel())} "
            f"Baseline_ON). Refusing to emit a misaligned pkl. "
            f"Pass allow_unaligned=True to override."
        )
```

Add `allow_unaligned: bool = False` to `build_session_from_raw`'s signature.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_ingest_alignment_guard.py tests/conversion -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/core/ingest.py tests/test_ingest_alignment_guard.py
git commit -m "fix(QC1): converter orders runs by filename stamp and refuses misaligned pkls"
```

---

### Task 11: Repair the remaining sessions and close the audit

**Files:**
- Modify: `data/cache/qc_alignment/alignment_repair_report.csv` (generated)
- Create: `docs/science/2026-08-04-QC1-alignment-repair-results.md`

**Interfaces:**
- Consumes: Tasks 5, 6.
- Produces: the results document and the closed audit.

- [ ] **Step 1: Repair BG_046 (the primary subject) for real**

```bash
py scripts/QC_technical/repair_trial_event_alignment.py --subjects BG_046 \
   --files BG_046_20082025.pkl BG_046_05092025_b.pkl
```
Expected: both `solved=True`, backups written under `data/pkls/BG_046/qc1_backup/`.

- [ ] **Step 2: Verify behaviour is unchanged**

```bash
py -c "
import sys,glob,os,pickle; sys.path.insert(0,'src')
from visdetect.core.session import load_session
for f in ['BG_046_20082025.pkl','BG_046_05092025_b.pkl']:
    cur = load_session(os.path.join('data','pkls','BG_046',f))
    bak = sorted(glob.glob(os.path.join('data','pkls','BG_046','qc1_backup',f+'.bak_*')))[-1]
    old = load_session(bak)
    a=[(t.trialoutcome,t.change_size,t.change_time) for t in old.trials]
    b=[(t.trialoutcome,t.change_size,t.change_time) for t in cur.trials]
    print(f, 'behaviour identical:', a==b, '| n_trials', len(a), '->', len(b))
"
```
Expected: `behaviour identical: True` and unchanged trial counts for both.

- [ ] **Step 3: Repair the remaining 15 non-BG_012 sessions**

```bash
py scripts/QC_technical/repair_trial_event_alignment.py --subjects BG_031 BG_038 BG_039 BG_041
```
Record which solve and which fall to all-`-1`. Per the spec, `BG_031 20052025` (0 trials) is expected to be unsolvable, and `BG_038 22082025` carries an independent truncation defect that alignment repair does not address.

- [ ] **Step 4: Re-run the audit**

```bash
py scripts/QC_technical/audit_trial_baselineon_alignment.py
```
Expected: every repaired session returns `aligned=True`.

- [ ] **Step 5: Write the results document**

Create `docs/science/2026-08-04-QC1-alignment-repair-results.md` recording: sessions repaired with their `(trial_start, event_offset)`, sessions left unsolvable with reasons, the before/after audit tallies, how many of the 48 benign sessions were confirmed aligned by measurement, and the standing caveats from the spec's §5a. Index it in `docs/science/QUESTION_INDEX.md`.

- [ ] **Step 6: Commit**

```bash
git add docs/science/2026-08-04-QC1-alignment-repair-results.md docs/science/QUESTION_INDEX.md
git add -f data/cache/qc_alignment/alignment_repair_report.csv data/cache/qc_alignment/trial_vs_baselineon_audit.csv
git commit -m "feat(QC1): repair all solvable sessions; audit closed with measured alignment"
```

---

## Follow-on work — NOT in this plan

### The TF registries must be rebuilt ONCE, for BOTH defects

An earlier task in this plan proposed regenerating `data/cache/tf_responsive/*.csv` here. **That was
removed on 2026-08-04**, for reasons established by the already-merged lick-channel work
(`fix/lick-channel-resolver`, merged to main as `c62448a`; memory note `lick_channel_defect_jul2026`):

- **The registries were already stale before QC1.** A MATLAB re-extraction batch (6 Mar 2026,
  33 BG_046 sessions) under-detects licks 10-40x, and the old `_collect_lick_times` pooled all four
  lick channels. Both are now fixed, but the shipped registries predate the fix and are **not
  reproducible from current code** (a banner already says so in that directory's README).
- **A local re-run cannot reproduce the registry schema.** `resp_lin` / `c1_r_lin` / `kernel_fwhm`
  come only from `cluster_bg/tf_glm_bg_task.py` (3 fits/unit); the local runner does 2. The original
  build was a **cluster** job; a faithful rebuild is ~**624 core-hr**.
- **The cheap trick does not transfer to QC1.** The lick fix leaves `trial_index` unchanged, so the
  seed-fixed `make_trial_folds` yields identical CV folds and a *paired within-unit re-fit*
  (~150-500 units, 1-4 wall-h locally) sizes the impact. QC1 **changes which trials have events**
  (offset 228; trials mapped to -1), so folds change on the 17 affected sessions — those need
  genuine refits, not paired comparisons.
- **The two contaminations are entangled.** All three BG_046 sessions checked are `piezo_2026`,
  including both QC1-affected ones — they are double-affected. The spec's "clean-only VMS 6.31 %"
  is therefore **not** a clean baseline; it isolates nothing, because those sessions still carry the
  lick defect. Do not quote it as a corrected figure.

**Therefore:** this plan ends at a closed audit and repaired pkls. The registry rebuild is its own
scoped piece of work covering both defects in a single pass.

Traps for whoever picks it up (all from `lick_channel_defect_jul2026`):
- **Clear `data/cache/tf_glm_*` and `results_bg_*` first.** `run_tf_glm_bg046.py` skips on file
  existence and the cluster task resumes per-unit, so old pooled rows silently interleave with new.
- `run_tf_glm_bg046.py:35` prepends the **deleted** sibling `E:/python_analysis/git_repos/vd_tf_bg046/src`;
  it needs `PYTHONPATH=<worktree>/src` and an `--out-dir` fix before any local run.
- Cluster is re-run-ready on ceph: `tf_glm_cluster/bg_mice/` (46 pkls, `targets_bg_046.csv`,
  sbatch `--array=1-368%80`).
- When it is done, supersede the recorded headline "VMS 5.3 % > DMS 2.8-3.1 %" in
  `tf_glm_replication_jun2026`, noting it was contaminated by **both** defects.

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §1 root cause (sign A + sign B), non-recursive glob, filename ordering | 10 |
| §2 Check 1 (case-sensitive set, 100% acceptance) | 1 |
| §2 Check 2 (n=0 rejects, MIN_RESID_N) | 1 |
| §2 solver, brute force, uniqueness/runner-up | 2 |
| §2 measured separation on real data | 3 |
| §3 index map not truncation; `None` default | 4 |
| §3 backup, idempotency, behaviour preserved | 5 |
| §4 audit becomes measured | 6 |
| §4 `align.py` + `-1` short-circuit before change_time fill | 7 |
| §4 `tf_glm_data.py` | 8 |
| §4 consumer audit work item | 9 |
| §4 converter refuses misaligned pkls | 10 |
| §5 verification (regression, null, behaviour diff, uniqueness, audit closes) | 3, 5, 11 |
| §6 phasing (BG_046 first, then generalise) | 5 step 5, 11 |
| §7 special cases (BG_031 20052025, BG_038 22082025) | 11 step 3 |
| §4 measured TF contamination → regenerate | **deliberately deferred** — see "Follow-on work"; the registries are stale from the lick fix independently of QC1 and must be rebuilt once, for both defects, as separate scoped work |

**Placeholder scan:** no TBD/TODO; every code step carries runnable code. Task 9 step 3 and Task 12 step 2 require judgement rather than fixed code — both say explicitly what to produce and forbid guessing entry points.

**Type consistency:** `solve_alignment` returns `Optional[Alignment]` in Tasks 2, 3, 5, 6, 10. `build_trial_event_index(n_trials, alignment)` is called identically in Tasks 5 and 10. `Alignment` field names (`trial_start`, `n_trials_matched`, `event_offset`, `agreement`, `resid_s`, `resid_n`, `runner_up_agreement`, `runner_up_resid_s`) are used consistently in Tasks 2, 3, 5, 6. `_event_index_for(session, i) -> int` is defined and used only in Task 8.

**Known risk carried into execution:** Task 8's `TrialRegressors(edges=...)` construction is written against a dataclass this plan has not read in full; the step says to mirror the file's existing empty-trial construction rather than invent fields.
