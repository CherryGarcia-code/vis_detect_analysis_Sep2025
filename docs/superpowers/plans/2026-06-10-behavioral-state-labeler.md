# Behavioral State Labeler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a human-in-the-loop system that learns interpretable behavioral-state thresholds from the experimenter's sparse outcome-raster labels and tags every session, in a form drop-in compatible with the existing GLM-HMM downstream interface.

**Architecture:** Library primitives (`state_labeling.py` for the data model / raster / queue; `state_calibration.py` for features / decision-tree calibration / tagger) wrapped by three CLIs (calibrate, tag, validate) and one matplotlib GUI. Calibration fits a shallow `DecisionTreeClassifier` on local outcome-composition features over a fitted window `W`; tagging mirrors `hmm.decode_session` columns and confidence gating.

**Tech Stack:** Python, numpy, pandas, scikit-learn (`DecisionTreeClassifier`, `cohen_kappa_score`, `export_text`), matplotlib (TkAgg GUI), pytest. Branch: `feature/behavioral-state-labeler` (already created).

**Spec:** `docs/superpowers/specs/2026-06-10-behavioral-state-labeler-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/visdetect/analysis/constants.py` (modify) | Add `STATE_*` constants |
| `src/visdetect/analysis/config.py` (modify) | Add `LICK_VALENCE_COLORS` |
| `src/visdetect/analysis/state_labeling.py` (create) | `StateEpisode`, save/load, `classify_lick_valence`, `build_outcome_raster`, `episodes_to_trial_labels`, `get_labeling_queue`, `render_raster` |
| `src/visdetect/analysis/state_calibration.py` (create) | `extract_state_features`, `attach_episode_labels`, `fit_state_tree`, `calibrate_states`, `CalibrationResult`, `tag_features`, `decode_session_states` |
| `scripts/state_labeling/run_state_labeler.py` (create) | matplotlib TkAgg GUI |
| `scripts/state_labeling/calibrate_states.py` (create) | CLI: fit rule → save model + `rules.md` |
| `scripts/state_labeling/tag_sessions.py` (create) | CLI: batch tag → per-session cache |
| `scripts/state_labeling/validate_states.py` (create) | CLI: κ / confusion vs labels & HMM; re-shade figure |
| `tests/test_state_labeling.py` (create) | Tests for `state_labeling.py` |
| `tests/test_state_calibration.py` (create) | Tests for `state_calibration.py` |

All test commands use `py -m pytest` (Windows). Each task ends with a commit.

---

### Task 1: Constants & colors

**Files:**
- Modify: `src/visdetect/analysis/constants.py`
- Modify: `src/visdetect/analysis/config.py`
- Test: `tests/test_state_labeling.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_state_labeling.py
from visdetect.analysis import constants as C
from visdetect.analysis import config as CFG


def test_state_constants_exist():
    assert C.STATE_LABELS == ["Impulsive", "StimSens", "Disengaged"]
    assert C.STATE_EASY_CHANGE_THRESH == 2.0
    assert C.STATE_CONFIDENCE_THRESHOLD == 0.8
    assert C.STATE_LABEL_W_DEFAULT in C.STATE_LABEL_W_GRID
    assert C.STATE_FEATURE_COLS == [
        "f_applick", "f_inapplick", "f_nolick", "f_abort", "f_miss_easy", "f_hit_hard",
    ]


def test_lick_valence_colors():
    for k in ["appropriate_lick", "inappropriate_lick", "nolick", "abort", "ref"]:
        assert k in CFG.LICK_VALENCE_COLORS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_state_labeling.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'STATE_LABELS'`

- [ ] **Step 3: Add constants and colors**

Append to `src/visdetect/analysis/constants.py`:

```python
# =====================================================================
# Behavioral state labeler (user-defined states)
# =====================================================================
STATE_LABELS = ["Impulsive", "StimSens", "Disengaged"]
STATE_EASY_CHANGE_THRESH = 2.0          # change_size >= this is an "easy"/obvious change
STATE_CONFIDENCE_THRESHOLD = 0.8        # gate trials below this tagger confidence
STATE_LABEL_W_GRID = [11, 15, 21, 31, 41, 51, 61]   # candidate window widths (trials)
STATE_LABEL_W_DEFAULT = 31
STATE_FEATURE_COLS = [
    "f_applick", "f_inapplick", "f_nolick", "f_abort", "f_miss_easy", "f_hit_hard",
]
```

Append to `src/visdetect/analysis/config.py`:

```python
# =====================================================================
# Lick-valence colors (behavioral state labeler raster)
# =====================================================================
LICK_VALENCE_COLORS: Dict[str, str] = {
    "appropriate_lick":   "#2e8b57",   # green  — hit on a real change
    "inappropriate_lick": "#d6453a",   # red    — early lick or catch SDT-FA
    "nolick":             "#7b5cb8",   # purple — miss or correct rejection
    "abort":              "#9aa0a6",   # grey
    "ref":                "#d9c7a0",   # muted  — reflex lick (excluded from fractions)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_state_labeling.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/constants.py src/visdetect/analysis/config.py tests/test_state_labeling.py
git commit -m "feat(state-labeler): add state constants and lick-valence colors"
```

---

### Task 2: `classify_lick_valence`

**Files:**
- Create: `src/visdetect/analysis/state_labeling.py`
- Test: `tests/test_state_labeling.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_state_labeling.py`:

```python
import pytest
from visdetect.analysis.state_labeling import classify_lick_valence


@pytest.mark.parametrize("outcome,is_go,is_catch,expected", [
    ("hit",  True,  False, "appropriate_lick"),    # go hit
    ("Hit",  True,  False, "appropriate_lick"),    # case-insensitive
    ("hit",  False, True,  "inappropriate_lick"),  # catch SDT false alarm
    ("miss", True,  False, "nolick"),              # go miss
    ("miss", False, True,  "nolick"),              # correct rejection
    ("fa",   True,  False, "inappropriate_lick"),  # early lick on go
    ("fa",   False, True,  "inappropriate_lick"),  # early lick on catch
    ("abort", True, False, "abort"),
    ("ref",  True,  False, "ref"),
])
def test_classify_lick_valence(outcome, is_go, is_catch, expected):
    assert classify_lick_valence(outcome, is_go, is_catch) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_state_labeling.py::test_classify_lick_valence -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'visdetect.analysis.state_labeling'`

- [ ] **Step 3: Create module with `classify_lick_valence`**

Create `src/visdetect/analysis/state_labeling.py`:

```python
"""User-defined behavioral state labeling — data model, raster, queue, rendering.

See docs/superpowers/specs/2026-06-10-behavioral-state-labeler-design.md.
States are anchored to the experimenter's sparse labels on the outcome raster,
not to a latent HMM. Color encodes the *lick decision's valence*.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from visdetect.analysis.behavior import get_trial_dataframe
from visdetect.analysis.config import LICK_VALENCE_COLORS, STAGE_ORDER, parse_session_date


def classify_lick_valence(outcome: str, is_go: bool, is_catch: bool) -> str:
    """Map a trial outcome to its lick-valence class.

    appropriate_lick   : go-trial hit (licked to a real change)
    inappropriate_lick : early lick ('fa', any trial) OR catch-trial 'hit' (SDT false alarm)
    nolick             : 'miss' (covers go-miss AND catch correct-rejection)
    abort / ref        : as-is ('ref' is excluded from fractions downstream)
    """
    o = (outcome or "").lower()
    if o == "abort":
        return "abort"
    if o == "ref":
        return "ref"
    if o == "fa":
        return "inappropriate_lick"
    if o == "hit":
        return "appropriate_lick" if is_go else "inappropriate_lick"
    if o == "miss":
        return "nolick"
    return "ref"  # unknown -> excluded from fractions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_state_labeling.py::test_classify_lick_valence -v`
Expected: PASS (9 parametrized cases)

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/state_labeling.py tests/test_state_labeling.py
git commit -m "feat(state-labeler): classify_lick_valence"
```

---

### Task 3: `StateEpisode`, save/load, `episodes_to_trial_labels`

**Files:**
- Modify: `src/visdetect/analysis/state_labeling.py`
- Test: `tests/test_state_labeling.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_state_labeling.py`:

```python
import numpy as np
from visdetect.analysis.state_labeling import (
    StateEpisode, save_episode, load_episodes, episodes_to_trial_labels,
)


def test_episode_save_load_roundtrip(tmp_path):
    path = tmp_path / "episodes.csv"
    e1 = StateEpisode("07072025", 10, 25, "Impulsive", "ben", "2026-06-10T00:00:00")
    e2 = StateEpisode("07072025", 40, 55, "Disengaged", "ben", "2026-06-10T00:01:00", notes="zoned out")
    save_episode(e1, path)
    save_episode(e2, path)
    loaded = load_episodes(path)
    assert len(loaded) == 2
    assert loaded[0].session_name == "07072025"
    assert loaded[0].start_trial == 10 and loaded[0].end_trial == 25
    assert loaded[1].state_label == "Disengaged"
    assert loaded[1].notes == "zoned out"


def test_episodes_to_trial_labels():
    eps = [
        StateEpisode("S1", 2, 4, "Impulsive", "ben", "t"),
        StateEpisode("S1", 7, 8, "Disengaged", "ben", "t"),
        StateEpisode("S2", 0, 1, "StimSens", "ben", "t"),  # different session ignored
    ]
    labels = episodes_to_trial_labels(eps, "S1", n_trials=10)
    assert labels[0] is None and labels[1] is None
    assert list(labels[2:5]) == ["Impulsive", "Impulsive", "Impulsive"]
    assert labels[5] is None and labels[6] is None
    assert list(labels[7:9]) == ["Disengaged", "Disengaged"]
    assert labels[9] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_state_labeling.py::test_episode_save_load_roundtrip -v`
Expected: FAIL with `ImportError: cannot import name 'StateEpisode'`

- [ ] **Step 3: Implement the dataclass and helpers**

Append to `src/visdetect/analysis/state_labeling.py`:

```python
@dataclass
class StateEpisode:
    """A contiguous span of trials the experimenter is confident about."""
    session_name: str
    start_trial: int          # inclusive index into the trial DataFrame
    end_trial: int            # inclusive
    state_label: str
    labeler: str
    timestamp: str
    notes: str = ""


_EPISODE_COLUMNS = [
    "session_name", "start_trial", "end_trial", "state_label", "labeler", "timestamp", "notes",
]


def save_episode(episode: StateEpisode, path) -> None:
    """Append one episode to the labels CSV (creates the file with a header)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([asdict(episode)])[_EPISODE_COLUMNS]
    header = not path.exists()
    row.to_csv(path, mode="a", header=header, index=False)


def load_episodes(path) -> List[StateEpisode]:
    """Load all episodes from the labels CSV."""
    path = Path(path)
    if not path.exists():
        return []
    df = pd.read_csv(path, dtype={"session_name": str, "notes": str})
    df["notes"] = df["notes"].fillna("")
    return [
        StateEpisode(
            session_name=str(r.session_name),
            start_trial=int(r.start_trial),
            end_trial=int(r.end_trial),
            state_label=str(r.state_label),
            labeler=str(r.labeler),
            timestamp=str(r.timestamp),
            notes=str(r.notes),
        )
        for r in df.itertuples(index=False)
    ]


def episodes_to_trial_labels(
    episodes: List[StateEpisode], session_name: str, n_trials: int
) -> np.ndarray:
    """Expand sparse episodes for one session to a per-trial label array.

    Unlabeled trials are ``None``.
    """
    labels = np.array([None] * n_trials, dtype=object)
    for ep in episodes:
        if str(ep.session_name) != str(session_name):
            continue
        lo = max(0, int(ep.start_trial))
        hi = min(n_trials - 1, int(ep.end_trial))
        labels[lo:hi + 1] = ep.state_label
    return labels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_state_labeling.py -v`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/state_labeling.py tests/test_state_labeling.py
git commit -m "feat(state-labeler): StateEpisode model + save/load + trial-label expansion"
```

---

### Task 4: `build_outcome_raster`

**Files:**
- Modify: `src/visdetect/analysis/state_labeling.py`
- Test: `tests/test_state_labeling.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_state_labeling.py`:

```python
from visdetect.core.session import Session, Trial
from visdetect.analysis.state_labeling import build_outcome_raster


def _trial(outcome, change_size):
    return Trial(
        trialoutcome=outcome, reactiontimes={}, change_size=change_size,
        orientation=None, ITI=1.0, change_time=2.0, baseline_values=np.zeros(5),
    )


def _session(trials):
    return Session(
        trials=trials, clusters=[], subject="T", session_name="T1",
        good_cluster_ids=[], ni_events={},
    )


def test_build_outcome_raster_lick_valence():
    trials = [
        _trial("Hit", 2.0),    # go hit          -> appropriate_lick
        _trial("Hit", 1.0),    # catch SDT FA    -> inappropriate_lick
        _trial("Miss", 4.0),   # go miss         -> nolick
        _trial("Miss", 1.0),   # correct reject  -> nolick
        _trial("FA", 1.5),     # early lick      -> inappropriate_lick
        _trial("abort", 1.5),  # abort           -> abort
    ]
    raster = build_outcome_raster(_session(trials))
    assert list(raster["lick_valence"]) == [
        "appropriate_lick", "inappropriate_lick", "nolick",
        "nolick", "inappropriate_lick", "abort",
    ]
    # color column is populated from LICK_VALENCE_COLORS
    assert raster.loc[0, "color"] == "#2e8b57"
    assert set(["trial_idx", "is_go", "is_catch", "change_size"]).issubset(raster.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_state_labeling.py::test_build_outcome_raster_lick_valence -v`
Expected: FAIL with `ImportError: cannot import name 'build_outcome_raster'`

- [ ] **Step 3: Implement `build_outcome_raster`**

Append to `src/visdetect/analysis/state_labeling.py`:

```python
def build_outcome_raster(session) -> pd.DataFrame:
    """Per-trial raster frame: outcome, trial type, change size, lick-valence + color."""
    df = get_trial_dataframe(session)
    if df.empty:
        return df
    out = pd.DataFrame({
        "trial_idx": df["trial_idx"].astype(int),
        "outcome": df["outcome"],
        "is_go": df["is_go"].astype(bool),
        "is_catch": df["is_catch"].astype(bool),
        "change_size": df["change_size"].astype(float),
    })
    out["lick_valence"] = [
        classify_lick_valence(o, g, c)
        for o, g, c in zip(out["outcome"], out["is_go"], out["is_catch"])
    ]
    out["color"] = out["lick_valence"].map(LICK_VALENCE_COLORS)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_state_labeling.py::test_build_outcome_raster_lick_valence -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/state_labeling.py tests/test_state_labeling.py
git commit -m "feat(state-labeler): build_outcome_raster"
```

---

### Task 5: `get_labeling_queue` (Expert→Naive)

**Files:**
- Modify: `src/visdetect/analysis/state_labeling.py`
- Test: `tests/test_state_labeling.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_state_labeling.py`:

```python
import pandas as pd
from visdetect.analysis.state_labeling import get_labeling_queue


def test_get_labeling_queue_expert_first_then_recent():
    manifest = pd.DataFrame({
        "session_name": ["01012025", "01032025", "01062025", "15062025"],
        "stage": ["Learning", "Learning", "Expert", "Expert"],
    })
    queue = get_labeling_queue(manifest=manifest)
    # Expert sessions first (most-recent first), then Learning (most-recent first)
    assert queue == ["15062025", "01062025", "01032025", "01012025"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_state_labeling.py::test_get_labeling_queue_expert_first_then_recent -v`
Expected: FAIL with `ImportError: cannot import name 'get_labeling_queue'`

- [ ] **Step 3: Implement `get_labeling_queue`**

Append to `src/visdetect/analysis/state_labeling.py`:

```python
def get_labeling_queue(manifest: Optional[pd.DataFrame] = None) -> List[str]:
    """Return session names ordered Expert -> Naive (stage priority, then most-recent first).

    If ``manifest`` is None, loads the QC-filtered staging manifest.
    """
    if manifest is None:
        from visdetect.analysis.config import load_staging_manifest
        manifest = load_staging_manifest(qc_only=True)

    stage_priority = {s: i for i, s in enumerate(reversed(STAGE_ORDER))}  # Expert -> 0
    fallback = len(STAGE_ORDER)
    rows = []
    for _, r in manifest.iterrows():
        sn = str(r["session_name"])
        rank = stage_priority.get(str(r.get("stage", "")), fallback)
        ymd = parse_session_date(int(sn))                 # (yyyy, mm, dd)
        rows.append((rank, tuple(-x for x in ymd), sn))   # negate for most-recent-first
    rows.sort(key=lambda t: (t[0], t[1]))
    return [sn for _, _, sn in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_state_labeling.py::test_get_labeling_queue_expert_first_then_recent -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/state_labeling.py tests/test_state_labeling.py
git commit -m "feat(state-labeler): get_labeling_queue (Expert->Naive)"
```

---

### Task 6: `render_raster` (shared by GUI + validation)

**Files:**
- Modify: `src/visdetect/analysis/state_labeling.py`
- Test: `tests/test_state_labeling.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_state_labeling.py`:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from visdetect.analysis.state_labeling import render_raster


def test_render_raster_draws_one_patch_per_trial():
    raster = build_outcome_raster(_session([
        _trial("Hit", 2.0), _trial("Miss", 1.0), _trial("FA", 1.5),
    ]))
    fig, ax = plt.subplots()
    render_raster(ax, raster)
    # one colored bar per trial
    assert len(ax.patches) >= 3
    plt.close(fig)


def test_render_raster_change_size_shading_runs():
    raster = build_outcome_raster(_session([_trial("Hit", 1.25), _trial("Hit", 4.0)]))
    fig, ax = plt.subplots()
    render_raster(ax, raster, change_size_shading=True)
    assert len(ax.patches) >= 2
    plt.close(fig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_state_labeling.py::test_render_raster_draws_one_patch_per_trial -v`
Expected: FAIL with `ImportError: cannot import name 'render_raster'`

- [ ] **Step 3: Implement `render_raster`**

Append to `src/visdetect/analysis/state_labeling.py`:

```python
# change_size -> opacity for the optional difficulty shading (bigger = more opaque)
_CS_OPACITY = {1.25: 0.30, 1.35: 0.45, 1.5: 0.60, 2.0: 0.80, 4.0: 1.0}


def _hex_to_rgb01(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def render_raster(ax, raster_df, change_size_shading: bool = False, episodes=None):
    """Draw the outcome raster on ``ax``: one colored bar per trial.

    Catch trials get a black outline. With ``change_size_shading``, go-trial hits
    and genuine (go-trial) misses are shaded by change size. ``episodes`` (list of
    StateEpisode) are drawn as translucent state spans behind the ticks.
    """
    import matplotlib.patches as mpatches
    from visdetect.analysis.config import LICK_VALENCE_COLORS

    n = len(raster_df)
    if episodes:
        state_tints = {"Impulsive": (0.84, 0.15, 0.16), "StimSens": (0.17, 0.63, 0.17),
                       "Disengaged": (0.48, 0.36, 0.72)}
        for ep in episodes:
            rgb = state_tints.get(ep.state_label, (0.5, 0.5, 0.5))
            ax.axvspan(ep.start_trial - 0.5, ep.end_trial + 0.5, color=rgb, alpha=0.18, lw=0)

    for i, row in enumerate(raster_df.itertuples(index=False)):
        lv = row.lick_valence
        base = LICK_VALENCE_COLORS.get(lv, "#999999")
        if change_size_shading and row.is_go and lv in ("appropriate_lick", "nolick"):
            rgb = _hex_to_rgb01(base)
            alpha = _CS_OPACITY.get(round(float(row.change_size), 2), 1.0)
            color = (rgb[0], rgb[1], rgb[2], alpha)
        else:
            color = base
        edge = "#111111" if row.is_catch else "none"
        ax.add_patch(mpatches.Rectangle((i - 0.5, 0), 1.0, 1.0, facecolor=color,
                                        edgecolor=edge, linewidth=0.6))
    ax.set_xlim(-0.5, max(n - 0.5, 0.5))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("trial index")
    return ax
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_state_labeling.py -v`
Expected: PASS (all `state_labeling` tests)

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/state_labeling.py tests/test_state_labeling.py
git commit -m "feat(state-labeler): render_raster (shared GUI + validation)"
```

---

### Task 7: `extract_state_features`

**Files:**
- Create: `src/visdetect/analysis/state_calibration.py`
- Test: `tests/test_state_calibration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_state_calibration.py`:

```python
import numpy as np
import pandas as pd
import pytest

from visdetect.analysis.state_calibration import extract_state_features


def _raster(lick_valences, is_go=None, change_size=None):
    n = len(lick_valences)
    return pd.DataFrame({
        "trial_idx": range(n),
        "lick_valence": lick_valences,
        "is_go": [True] * n if is_go is None else is_go,
        "change_size": [1.5] * n if change_size is None else change_size,
    })


def test_features_center_window_fractions():
    lv = ["appropriate_lick", "appropriate_lick", "inappropriate_lick",
          "inappropriate_lick", "inappropriate_lick", "nolick", "nolick"]
    feats = extract_state_features(_raster(lv), W=3)
    # center index 3 window {2,3,4} are all inappropriate_lick
    assert feats.loc[3, "f_inapplick"] == pytest.approx(1.0)
    # index 0 window {0,1} are all appropriate_lick
    assert feats.loc[0, "f_applick"] == pytest.approx(1.0)
    # four primary fractions sum to 1 everywhere (no ref/abort here)
    s = feats[["f_applick", "f_inapplick", "f_nolick", "f_abort"]].sum(axis=1)
    assert np.allclose(s.values, 1.0)


def test_features_difficulty_aware():
    lv = ["inappropriate_lick", "nolick", "nolick"]
    feats = extract_state_features(
        _raster(lv, is_go=[True, True, True], change_size=[1.5, 4.0, 4.0]), W=3,
    )
    # index 1 window {0,1,2}: miss_easy at idx1,2 (nolick & go & cs>=2) -> 2/3
    assert feats.loc[1, "f_miss_easy"] == pytest.approx(2.0 / 3.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_state_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'visdetect.analysis.state_calibration'`

- [ ] **Step 3: Create module with `extract_state_features`**

Create `src/visdetect/analysis/state_calibration.py`:

```python
"""Feature extraction, decision-tree calibration, and tagging for behavioral states.

See docs/superpowers/specs/2026-06-10-behavioral-state-labeler-design.md.
Features are local outcome-composition fractions (lick-valence + difficulty-aware)
over a symmetric window W. The rule is a shallow DecisionTreeClassifier fit on the
experimenter's labeled trials; tagging mirrors hmm.decode_session columns.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import pickle

import numpy as np
import pandas as pd

from visdetect.analysis.constants import (
    STATE_EASY_CHANGE_THRESH, STATE_FEATURE_COLS, STATE_LABEL_W_GRID,
    STATE_CONFIDENCE_THRESHOLD,
)


def extract_state_features(raster_df: pd.DataFrame, W: int) -> pd.DataFrame:
    """Add local-window composition features (STATE_FEATURE_COLS) per trial.

    Fractions use a symmetric, centered window of width ``W`` trials, with the
    denominator = window trials excluding 'ref'. Edges shrink (min_periods=1).
    """
    df = raster_df.reset_index(drop=True).copy()
    lv = df["lick_valence"]
    applick = (lv == "appropriate_lick").astype(int)
    inapplick = (lv == "inappropriate_lick").astype(int)
    nolick = (lv == "nolick").astype(int)
    abort = (lv == "abort").astype(int)
    ref = (lv == "ref").astype(int)
    non_ref = (1 - ref)
    is_go = df["is_go"].astype(bool)
    easy = df["change_size"].astype(float) >= STATE_EASY_CHANGE_THRESH
    miss_easy = (nolick.astype(bool) & is_go & easy).astype(int)
    hit_hard = (applick.astype(bool) & (~easy)).astype(int)

    def roll(s):
        return s.rolling(W, center=True, min_periods=1).sum()

    denom = roll(non_ref).replace(0, np.nan)
    df["f_applick"]   = (roll(applick) / denom).fillna(0.0)
    df["f_inapplick"] = (roll(inapplick) / denom).fillna(0.0)
    df["f_nolick"]    = (roll(nolick) / denom).fillna(0.0)
    df["f_abort"]     = (roll(abort) / denom).fillna(0.0)
    df["f_miss_easy"] = (roll(miss_easy) / denom).fillna(0.0)
    df["f_hit_hard"]  = (roll(hit_hard) / denom).fillna(0.0)
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_state_calibration.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/state_calibration.py tests/test_state_calibration.py
git commit -m "feat(state-labeler): extract_state_features (local-window composition)"
```

---

### Task 8: `attach_episode_labels` + `fit_state_tree`

**Files:**
- Modify: `src/visdetect/analysis/state_calibration.py`
- Test: `tests/test_state_calibration.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_state_calibration.py`:

```python
from visdetect.analysis.state_labeling import StateEpisode
from visdetect.analysis.state_calibration import attach_episode_labels, fit_state_tree


def test_attach_episode_labels_by_trial_idx():
    feats = extract_state_features(_raster(["nolick"] * 6), W=3)
    eps = [StateEpisode("S1", 1, 3, "Disengaged", "ben", "t")]
    labeled = attach_episode_labels(feats, eps, "S1")
    assert labeled.loc[0, "state"] is None
    assert list(labeled.loc[1:3, "state"]) == ["Disengaged"] * 3
    assert labeled.loc[4, "state"] is None


def _separable_training_frame():
    # clean, linearly separable 3-class table over the feature columns
    from visdetect.analysis.constants import STATE_FEATURE_COLS
    data = []
    for _ in range(8):
        data.append({**{c: 0.0 for c in STATE_FEATURE_COLS}, "f_inapplick": 0.9, "state": "Impulsive"})
        data.append({**{c: 0.0 for c in STATE_FEATURE_COLS}, "f_nolick": 0.9, "state": "Disengaged"})
        data.append({**{c: 0.0 for c in STATE_FEATURE_COLS}, "f_applick": 0.9, "state": "StimSens"})
    return pd.DataFrame(data)


def test_fit_state_tree_separates_classes_and_is_deterministic():
    df = _separable_training_frame()
    t1 = fit_state_tree(df, seed=42)
    t2 = fit_state_tree(df, seed=42)
    from visdetect.analysis.constants import STATE_FEATURE_COLS
    pred = t1.predict(df[STATE_FEATURE_COLS].values)
    assert (pred == df["state"].values).mean() == 1.0           # separable -> perfect train fit
    assert list(t1.feature_importances_) == list(t2.feature_importances_)  # deterministic
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_state_calibration.py::test_fit_state_tree_separates_classes_and_is_deterministic -v`
Expected: FAIL with `ImportError: cannot import name 'fit_state_tree'`

- [ ] **Step 3: Implement helpers**

Append to `src/visdetect/analysis/state_calibration.py`:

```python
def attach_episode_labels(features_df: pd.DataFrame, episodes, session_name: str) -> pd.DataFrame:
    """Add a 'state' column from episodes (None where unlabeled), keyed by trial_idx."""
    from visdetect.analysis.state_labeling import episodes_to_trial_labels
    df = features_df.copy()
    n = int(df["trial_idx"].max()) + 1 if len(df) else 0
    lab = episodes_to_trial_labels(episodes, session_name, n)
    df["state"] = [lab[int(i)] for i in df["trial_idx"]]
    return df


def fit_state_tree(features_df: pd.DataFrame, seed: int = 42):
    """Fit a shallow, readable decision tree on labeled rows (the 'state' column)."""
    from sklearn.tree import DecisionTreeClassifier
    train = features_df[features_df["state"].notna()]
    X = train[STATE_FEATURE_COLS].values
    y = train["state"].astype(str).values
    tree = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=seed,
    )
    tree.fit(X, y)
    return tree
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_state_calibration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/state_calibration.py tests/test_state_calibration.py
git commit -m "feat(state-labeler): attach_episode_labels + fit_state_tree"
```

---

### Task 9: `calibrate_states` + `CalibrationResult`

**Files:**
- Modify: `src/visdetect/analysis/state_calibration.py`
- Test: `tests/test_state_calibration.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_state_calibration.py`:

```python
from visdetect.analysis.state_calibration import calibrate_states, CalibrationResult


def _planted_raster(session_name):
    # trials 0-9 impulsive (inappropriate_lick), 10-19 stimsens (appropriate_lick, easy),
    # 20-29 disengaged (nolick, go, easy)
    lv = (["inappropriate_lick"] * 10 + ["appropriate_lick"] * 10 + ["nolick"] * 10)
    cs = ([1.5] * 10 + [4.0] * 10 + [4.0] * 10)
    return pd.DataFrame({
        "trial_idx": range(30), "lick_valence": lv,
        "is_go": [True] * 30, "change_size": cs,
    })


def test_calibrate_states_returns_result_and_fits():
    rasters = {"A": _planted_raster("A"), "B": _planted_raster("B")}
    eps = []
    for s in ("A", "B"):
        eps += [
            StateEpisode(s, 2, 7, "Impulsive", "ben", "t"),
            StateEpisode(s, 12, 17, "StimSens", "ben", "t"),
            StateEpisode(s, 22, 27, "Disengaged", "ben", "t"),
        ]
    result = calibrate_states(rasters, eps, w_grid=[3, 5], seed=42)
    assert isinstance(result, CalibrationResult)
    assert result.window in (3, 5)
    assert set(result.state_labels) == {"Impulsive", "StimSens", "Disengaged"}
    assert result.loso_kappa > 0.5
    assert "f_" in result.rules_text


def test_calibration_result_save_load(tmp_path):
    rasters = {"A": _planted_raster("A"), "B": _planted_raster("B")}
    eps = []
    for s in ("A", "B"):
        eps += [
            StateEpisode(s, 2, 7, "Impulsive", "ben", "t"),
            StateEpisode(s, 12, 17, "StimSens", "ben", "t"),
            StateEpisode(s, 22, 27, "Disengaged", "ben", "t"),
        ]
    result = calibrate_states(rasters, eps, w_grid=[3, 5], seed=42)
    p = tmp_path / "model.pkl"
    result.save(p)
    loaded = CalibrationResult.load(p)
    assert loaded.window == result.window
    assert loaded.state_labels == result.state_labels
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_state_calibration.py::test_calibrate_states_returns_result_and_fits -v`
Expected: FAIL with `ImportError: cannot import name 'calibrate_states'`

- [ ] **Step 3: Implement `CalibrationResult` and `calibrate_states`**

Append to `src/visdetect/analysis/state_calibration.py`:

```python
@dataclass
class CalibrationResult:
    tree: object                 # sklearn DecisionTreeClassifier
    window: int
    state_labels: List[str]
    feature_cols: List[str]
    loso_kappa: float
    rules_text: str

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> "CalibrationResult":
        with open(path, "rb") as f:
            return pickle.load(f)


def _pool_labeled(rasters: Dict[str, pd.DataFrame], episodes, W: int) -> pd.DataFrame:
    frames = []
    for sn, raster in rasters.items():
        feats = extract_state_features(raster, W)
        feats = attach_episode_labels(feats, episodes, sn)
        feats = feats[feats["state"].notna()].copy()
        feats["__session"] = sn
        frames.append(feats)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def calibrate_states(rasters, episodes, w_grid=None, seed: int = 42) -> CalibrationResult:
    """Fit the state rule: choose W by LOSO Cohen's kappa, then refit on all labels."""
    from sklearn.metrics import cohen_kappa_score
    from sklearn.tree import export_text
    if w_grid is None:
        w_grid = STATE_LABEL_W_GRID

    best = None  # (W, mean_kappa, pooled)
    for W in w_grid:
        pooled = _pool_labeled(rasters, episodes, W)
        if pooled.empty:
            continue
        sessions = pooled["__session"].unique()
        kappas = []
        for hold in sessions:
            tr = pooled[pooled["__session"] != hold]
            te = pooled[pooled["__session"] == hold]
            if te.empty or tr["state"].nunique() < 2:
                continue
            m = fit_state_tree(tr, seed=seed)
            pred = m.predict(te[STATE_FEATURE_COLS].values)
            kappas.append(cohen_kappa_score(te["state"].astype(str).values, pred))
        mean_k = float(np.mean(kappas)) if kappas else float("nan")
        if best is None or (not np.isnan(mean_k) and (np.isnan(best[1]) or mean_k > best[1])):
            best = (W, mean_k, pooled)

    if best is None:
        raise ValueError("No labeled trials found for any window in w_grid.")
    W, kappa, pooled = best
    tree = fit_state_tree(pooled, seed=seed)
    rules = export_text(tree, feature_names=list(STATE_FEATURE_COLS))
    return CalibrationResult(
        tree=tree, window=W, state_labels=list(tree.classes_),
        feature_cols=list(STATE_FEATURE_COLS), loso_kappa=kappa, rules_text=rules,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_state_calibration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/state_calibration.py tests/test_state_calibration.py
git commit -m "feat(state-labeler): calibrate_states + CalibrationResult (LOSO W selection)"
```

---

### Task 10: `tag_features` + `decode_session_states`

**Files:**
- Modify: `src/visdetect/analysis/state_calibration.py`
- Test: `tests/test_state_calibration.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_state_calibration.py`:

```python
from visdetect.analysis.state_calibration import tag_features, decode_session_states


def test_tag_features_columns_and_confidence_gating():
    df = _separable_training_frame()
    tree = fit_state_tree(df, seed=42)
    from visdetect.analysis.constants import STATE_FEATURE_COLS
    feats = df[STATE_FEATURE_COLS].copy()
    feats["trial_idx"] = range(len(feats))

    tagged = tag_features(tree, feats, confidence_threshold=0.8)
    K = len(tree.classes_)
    for k in range(K):
        assert f"p_state_{k}" in tagged.columns
    assert {"state", "state_label", "state_confidence", "state_gated"}.issubset(tagged.columns)
    # separable data -> pure leaves -> confidence 1.0 -> nothing gated at 0.8
    assert (tagged["state_gated"] == -1).sum() == 0
    # threshold above the max confidence gates everything
    tagged_hi = tag_features(tree, feats, confidence_threshold=1.0)
    assert (tagged_hi["state_gated"] == -1).all()


def test_decode_session_states_runs_on_synthetic_session():
    from visdetect.utils.synthetic import make_synthetic_session
    df = _separable_training_frame()
    tree = fit_state_tree(df, seed=42)
    result = CalibrationResult(tree, 5, list(tree.classes_), list(df.columns[:-1]), 1.0, "")
    sess = make_synthetic_session(n_trials=30, n_clusters=2, seed=1)
    tagged = decode_session_states(result, sess)
    assert len(tagged) == 30
    assert {"state", "state_label", "state_confidence", "state_gated"}.issubset(tagged.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_state_calibration.py::test_tag_features_columns_and_confidence_gating -v`
Expected: FAIL with `ImportError: cannot import name 'tag_features'`

- [ ] **Step 3: Implement `tag_features` and `decode_session_states`**

Append to `src/visdetect/analysis/state_calibration.py`:

```python
def tag_features(tree, features_df: pd.DataFrame,
                 confidence_threshold: float = STATE_CONFIDENCE_THRESHOLD) -> pd.DataFrame:
    """Tag each row with a state + confidence, mirroring hmm.decode_session columns."""
    from visdetect.analysis.hmm import assign_states_with_confidence
    probs = tree.predict_proba(features_df[STATE_FEATURE_COLS].values)
    classes = list(tree.classes_)
    out = features_df.copy()
    for k in range(len(classes)):
        out[f"p_state_{k}"] = probs[:, k]
    idx = probs.argmax(axis=1)
    out["state"] = idx.astype(int)
    out["state_label"] = [classes[i] for i in idx]
    out["state_confidence"] = probs.max(axis=1)
    out["state_gated"] = assign_states_with_confidence(probs, threshold=confidence_threshold)
    return out


def decode_session_states(result: CalibrationResult, session,
                          confidence_threshold: float = STATE_CONFIDENCE_THRESHOLD) -> pd.DataFrame:
    """Decode one session to per-trial states (mirrors hmm.decode_session)."""
    from visdetect.analysis.state_labeling import build_outcome_raster
    raster = build_outcome_raster(session)
    if raster.empty:
        return raster
    feats = extract_state_features(raster, result.window)
    return tag_features(result.tree, feats, confidence_threshold=confidence_threshold)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_state_calibration.py -v`
Expected: PASS (all calibration tests)

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/state_calibration.py tests/test_state_calibration.py
git commit -m "feat(state-labeler): tag_features + decode_session_states (HMM-compatible columns)"
```

---

### Task 11: CLI — `calibrate_states.py`

**Files:**
- Create: `scripts/state_labeling/calibrate_states.py`
- Test: `tests/test_state_calibration.py`

- [ ] **Step 1: Write the failing smoke test**

Append to `tests/test_state_calibration.py`:

```python
import subprocess, sys, os

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "state_labeling")


def test_calibrate_cli_help():
    r = subprocess.run([sys.executable, os.path.join(_SCRIPTS, "calibrate_states.py"), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "usage" in (r.stdout + r.stderr).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_state_calibration.py::test_calibrate_cli_help -v`
Expected: FAIL (file not found → non-zero returncode)

- [ ] **Step 3: Create the CLI**

Create `scripts/state_labeling/calibrate_states.py`:

```python
"""CLI: fit the behavioral-state rule from labeled episodes; save model + rules.md."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from visdetect.suite.loader import load_session
from visdetect.analysis.state_labeling import load_episodes, build_outcome_raster
from visdetect.analysis.state_calibration import calibrate_states


def main():
    ap = argparse.ArgumentParser(description="Calibrate behavioral-state rule from labeled episodes.")
    ap.add_argument("--labels", default="data/state_labels/state_episodes.csv")
    ap.add_argument("--out-model", default="data/state_labels/state_rule.pkl")
    ap.add_argument("--out-rules", default="data/state_labels/rules.md")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    episodes = load_episodes(args.labels)
    if not episodes:
        raise SystemExit(f"No episodes found in {args.labels}")
    label_sessions = sorted({e.session_name for e in episodes})

    rasters = {}
    for sn in label_sessions:
        sess = load_session(sn)
        rasters[sn] = build_outcome_raster(sess)
        del sess

    result = calibrate_states(rasters, episodes, seed=args.seed)
    result.save(args.out_model)
    os.makedirs(os.path.dirname(args.out_rules), exist_ok=True)
    with open(args.out_rules, "w", encoding="utf-8") as f:
        f.write(f"# Behavioral-state rule\n\nwindow W = {result.window}\n")
        f.write(f"LOSO Cohen's kappa = {result.loso_kappa:.3f}\n")
        f.write(f"states = {result.state_labels}\n\n```\n{result.rules_text}\n```\n")
    print(f"Saved model -> {args.out_model}  (W={result.window}, kappa={result.loso_kappa:.3f})")
    print(f"Saved rules -> {args.out_rules}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_state_calibration.py::test_calibrate_cli_help -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/state_labeling/calibrate_states.py tests/test_state_calibration.py
git commit -m "feat(state-labeler): calibrate_states CLI"
```

---

### Task 12: CLI — `tag_sessions.py`

**Files:**
- Create: `scripts/state_labeling/tag_sessions.py`
- Test: `tests/test_state_calibration.py`

- [ ] **Step 1: Write the failing smoke test**

Append to `tests/test_state_calibration.py`:

```python
def test_tag_cli_help():
    r = subprocess.run([sys.executable, os.path.join(_SCRIPTS, "tag_sessions.py"), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "usage" in (r.stdout + r.stderr).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_state_calibration.py::test_tag_cli_help -v`
Expected: FAIL (file not found)

- [ ] **Step 3: Create the CLI**

Create `scripts/state_labeling/tag_sessions.py`:

```python
"""CLI: tag all manifest sessions with behavioral states -> per-session CSV cache."""
import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from visdetect.analysis.config import load_staging_manifest
from visdetect.suite.loader import load_session
from visdetect.analysis.state_calibration import CalibrationResult, decode_session_states


def main():
    ap = argparse.ArgumentParser(description="Tag sessions with behavioral states.")
    ap.add_argument("--model", default="data/state_labels/state_rule.pkl")
    ap.add_argument("--out-dir", default="data/cache/state_tags")
    ap.add_argument("--confidence", type=float, default=0.8)
    args = ap.parse_args()

    result = CalibrationResult.load(args.model)
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = load_staging_manifest(qc_only=True)
    for _, row in manifest.iterrows():
        sn = str(row["session_name"])
        sess = load_session(sn)
        tagged = decode_session_states(result, sess, confidence_threshold=args.confidence)
        tagged.to_csv(os.path.join(args.out_dir, f"{sn}.csv"), index=False)
        print(f"tagged {sn}: {len(tagged)} trials")
        del sess, tagged
        gc.collect()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_state_calibration.py::test_tag_cli_help -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/state_labeling/tag_sessions.py tests/test_state_calibration.py
git commit -m "feat(state-labeler): tag_sessions CLI"
```

---

### Task 13: CLI — `validate_states.py`

**Files:**
- Create: `scripts/state_labeling/validate_states.py`
- Test: `tests/test_state_calibration.py`

- [ ] **Step 1: Write the failing smoke test**

Append to `tests/test_state_calibration.py`:

```python
def test_validate_cli_help():
    r = subprocess.run([sys.executable, os.path.join(_SCRIPTS, "validate_states.py"), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "usage" in (r.stdout + r.stderr).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_state_calibration.py::test_validate_cli_help -v`
Expected: FAIL (file not found)

- [ ] **Step 3: Create the CLI**

Create `scripts/state_labeling/validate_states.py`:

```python
"""CLI: validate the state rule vs the experimenter's labels (kappa, confusion) and
produce a re-shade figure per labeled session."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visdetect.suite.loader import load_session
from visdetect.analysis.state_labeling import (
    load_episodes, build_outcome_raster, episodes_to_trial_labels, render_raster,
)
from visdetect.analysis.state_calibration import (
    CalibrationResult, extract_state_features, tag_features,
)


def main():
    ap = argparse.ArgumentParser(description="Validate state rule vs labels; re-shade figures.")
    ap.add_argument("--model", default="data/state_labels/state_rule.pkl")
    ap.add_argument("--labels", default="data/state_labels/state_episodes.csv")
    ap.add_argument("--fig-dir", default="figures/state_labeler")
    args = ap.parse_args()

    result = CalibrationResult.load(args.model)
    episodes = load_episodes(args.labels)
    os.makedirs(args.fig_dir, exist_ok=True)

    y_true, y_pred = [], []
    for sn in sorted({e.session_name for e in episodes}):
        sess = load_session(sn)
        raster = build_outcome_raster(sess)
        feats = extract_state_features(raster, result.window)
        tagged = tag_features(result.tree, feats, confidence_threshold=0.0)  # no gating for agreement
        lab = episodes_to_trial_labels(episodes, sn, len(raster))
        for i in range(len(raster)):
            if lab[i] is not None:
                y_true.append(lab[i])
                y_pred.append(tagged.loc[i, "state_label"])

        fig, ax = plt.subplots(figsize=(12, 2))
        render_raster(ax, raster, episodes=[e for e in episodes if e.session_name == sn])
        ax.set_title(f"{sn} — tagger vs your labels")
        fig.savefig(os.path.join(args.fig_dir, f"reshade_{sn}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
        del sess

    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    if y_true:
        k = cohen_kappa_score(y_true, y_pred)
        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print(f"Cohen's kappa vs labels: {k:.3f}")
        print("Confusion (rows=true, cols=pred):", labels)
        print(pd.DataFrame(cm, index=labels, columns=labels))
    else:
        print("No labeled trials to validate.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_state_calibration.py::test_validate_cli_help -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/state_labeling/validate_states.py tests/test_state_calibration.py
git commit -m "feat(state-labeler): validate_states CLI (kappa, confusion, re-shade)"
```

---

### Task 14: GUI — `run_state_labeler.py`

**Files:**
- Create: `scripts/state_labeling/run_state_labeler.py`
- Test: `tests/test_state_calibration.py`

- [ ] **Step 1: Write the failing smoke test**

Append to `tests/test_state_calibration.py`:

```python
def test_gui_help():
    r = subprocess.run([sys.executable, os.path.join(_SCRIPTS, "run_state_labeler.py"), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "usage" in (r.stdout + r.stderr).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_state_calibration.py::test_gui_help -v`
Expected: FAIL (file not found)

- [ ] **Step 3: Create the GUI**

Create `scripts/state_labeling/run_state_labeler.py`. Argparse is parsed BEFORE any Tk/import-heavy work so `--help` is safe in headless CI:

```python
"""Interactive matplotlib GUI to sparsely label behavioral-state episodes on the
outcome raster. Mirrors scripts/tf_labeling/run_labeling_gui.py.

Keys: 1=Impulsive 2=StimSens 3=Disengaged  | drag=paint span  | backspace=erase span
      r=toggle rolling overlay (off)  c=toggle change-size shading (off)
      left/right=prev/next session (Expert->Naive)  s=save  q=quit
"""
import argparse
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def main():
    ap = argparse.ArgumentParser(description="Behavioral-state labeling GUI.")
    ap.add_argument("--labels", default="data/state_labels/state_episodes.csv")
    ap.add_argument("--labeler", default=os.environ.get("USERNAME", "user"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import SpanSelector

    from visdetect.suite.loader import load_session
    from visdetect.analysis.state_labeling import (
        get_labeling_queue, build_outcome_raster, render_raster, save_episode, StateEpisode,
    )

    queue = get_labeling_queue()
    state = {"i": 0, "label": "Impulsive", "cs_shade": False}
    keymap = {"1": "Impulsive", "2": "StimSens", "3": "Disengaged"}

    fig, ax = plt.subplots(figsize=(14, 3))

    def draw():
        ax.clear()
        sn = queue[state["i"]]
        sess = load_session(sn)
        raster = build_outcome_raster(sess)
        render_raster(ax, raster, change_size_shading=state["cs_shade"])
        ax.set_title(f"{sn}  [{state['i']+1}/{len(queue)}]  active label: {state['label']}")
        fig.canvas.draw_idle()
        state["raster_len"] = len(raster)
        state["session_name"] = sn

    def on_span(xmin, xmax):
        lo, hi = int(round(xmin)), int(round(xmax))
        ep = StateEpisode(state["session_name"], lo, hi, state["label"], args.labeler,
                          dt.datetime.now().isoformat())
        save_episode(ep, args.labels)
        ax.axvspan(lo - 0.5, hi + 0.5, alpha=0.18, color="orange", lw=0)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in keymap:
            state["label"] = keymap[event.key]
        elif event.key == "right":
            state["i"] = min(state["i"] + 1, len(queue) - 1); draw()
        elif event.key == "left":
            state["i"] = max(state["i"] - 1, 0); draw()
        elif event.key == "c":
            state["cs_shade"] = not state["cs_shade"]; draw()
        elif event.key == "q":
            plt.close(fig)
        ax.set_title(ax.get_title().rsplit("active label:", 1)[0] + f"active label: {state['label']}")
        fig.canvas.draw_idle()

    span = SpanSelector(ax, on_span, "horizontal", useblit=True,
                        props=dict(alpha=0.2, facecolor="orange"))
    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_state_calibration.py::test_gui_help -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and commit**

Run: `py -m pytest tests/test_state_labeling.py tests/test_state_calibration.py -v`
Expected: PASS (all tests)

```bash
git add scripts/state_labeling/run_state_labeler.py tests/test_state_calibration.py
git commit -m "feat(state-labeler): interactive labeling GUI"
```

---

## Post-implementation (manual, not automated)

These are run by the experimenter, not part of the TDD suite:

1. **Label Expert sessions** with `run_state_labeler.py` (Expert→Naive queue).
2. **Calibrate**: `py scripts/state_labeling/calibrate_states.py` → inspect `rules.md`.
3. **Tag**: `py scripts/state_labeling/tag_sessions.py`.
4. **Validate**: `py scripts/state_labeling/validate_states.py` → review κ, confusion, re-shade PNGs; relabel where the tagger disagrees (refinement loop).
5. **Per-stage W check** and **GLM-HMM / legacy `identify_session_state` comparison** (spec §9.3–§9.5) — extend `validate_states.py` once labels exist.
6. **Cross-subject transfer** (spec §9.4) — run `tag_sessions.py` against BG_031/038/039 manifests and eyeball.
7. **Logistic soft-classifier comparison** (spec §7, deferred Option 2) — fit `LogisticRegression(multi_class='multinomial')` on the same labeled features and report its LOSO κ alongside the tree's in `validate_states.py`. Not used for tagging; comparison only.
