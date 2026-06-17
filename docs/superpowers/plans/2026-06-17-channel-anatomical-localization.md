# Channel & Unit Anatomical Localization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estimate each Neuropixels-2.0 channel's Allen-CCF coordinate + region + uncertainty (per chanmap, per subject) and propagate to units via peak channel, persisted as sidecar artifacts + `build_unit_table` columns (Phase A — no PKL re-ingest).

**Architecture:** A human, off-repo, produces one tool-agnostic **track artifact** per subject (4 shank polylines in CCF + depth calibration + uncertainty), via brainreg + brainglobe-segmentation (+ Pinpoint). In-repo automated code consumes that artifact: assign each active channel to a shank and a depth, place it on the shank's CCF polyline (arc-length), look up the Allen region + a confidence, and join to units by peak channel. New `src/visdetect/anatomy/` subpackage holds pure logic; `scripts/anatomy/` holds CLIs; `build_unit_table` gains the columns.

**Tech Stack:** Python 3, NumPy, pandas, `brainglobe-atlasapi` (Allen Mouse CCF), matplotlib; reuses `visdetect.analysis.tracking_qc` waveform primitives. pytest for tests.

**Spec:** `docs/superpowers/specs/2026-06-17-channel-anatomical-localization-design.md`.

## Global Constraints

- **Invoke Python as `py`** (Windows + Git Bash), or `.venv/Scripts/python.exe` directly. Tests: `.venv/Scripts/python.exe -m pytest`.
- **New code imports from `visdetect.*`** (canonical package). The flat `analysis_suite/{config,loader,utils,plotting}.py` modules no longer exist.
- **Reuse existing primitives** — `extract_peak_channel`, `load_raw_mean_waveform`, `load_channel_positions` from `visdetect.analysis.tracking_qc`. Do not reimplement peak-channel/depth logic.
- **Unit-table keys are integers**: `Session_Date` (int, e.g. `7072025`) and `Cluster_ID` (int). The frozen contract (`visdetect.suite.unit_table_schema`) rejects non-integer keys and duplicate `(Session_Date, Cluster_ID)` rows.
- **RawWaveform layout**: `data/unit_match/input/<SUBJECT>/<session>/RawWaveforms/Unit{cluster_id}_RawSpikes.npy`, shape `(n_samples=82, n_channels=383, n_cv=2)`; `channel_positions.npy` shape `(n_channels, 2)` = `[x_um, y_um]`; `y_um` increases away from the shank tip (larger y = more dorsal/superficial).
- **Probe = NP2.0 four-shank**: 8 x-columns → 4 shanks ~250 µm apart, 2 columns/shank ~32 µm apart. Chronic/fixed → one track-set per subject; sessions differ only by the active y-window.
- **Session-name normalization (cross-subject)**: BG_046 session dirs are the bare date token (`01072025`); BG_031/038/039 dirs carry the subject prefix (`BG_031_01042025`), and `tracking_qc.load_raw_mean_waveform`/`load_channel_positions` only try `name` and `name.zfill(8)`. The CLIs therefore **key everything by the numeric date token** and resolve the actual session subdir with the `session_token()` / `resolve_session_dir()` helpers (Task 8). For BG_046 token == dir name == pkl token, so it is a no-op there.
- **Atlas frame**: all CCF coordinates are microns in the `allen_mouse_25um` BrainGlobe atlas space, axis order `(AP, ML, DV)` consistent everywhere. Store this string in every artifact and atlas CSV.
- **Phase A only**: do not modify the `Cluster` dataclass or re-ingest PKLs.
- **Commit after every task.** Branch: `feature/channel-anatomical-localization` (this worktree).

---

### Task 1: Track artifact contract (`tracks.py`)

The tool-agnostic contract every downstream step consumes. Dataclasses + JSON load/save + fail-loud validation.

**Files:**
- Create: `src/visdetect/anatomy/__init__.py`
- Create: `src/visdetect/anatomy/tracks.py`
- Test: `tests/anatomy/test_tracks.py`

**Interfaces:**
- Produces:
  - `@dataclass ShankTrack(probe_shank_index:int, ccf_polyline:np.ndarray, tip_y_um:float, method:str, sigma_along_um:float, sigma_across_um:float, sigma_growth_k:float, planned_entry:Optional[np.ndarray], planned_vector:Optional[np.ndarray])` — `ccf_polyline` is `(N,3)` float µm `(AP,ML,DV)`, ordered **deepest point first** (index 0 = closest to tip).
  - `@dataclass TrackArtifact(subject:str, atlas:str, hemisphere:str, barcode_orientation:str, source_tool:str, created:str, shanks:List[ShankTrack])`
  - `VALID_METHODS = {"brainreg_traced","extended_from_tip","pinpoint_planned"}`
  - `class TrackArtifactError(ValueError)`
  - `load_track_artifact(path)->TrackArtifact`, `save_track_artifact(art, path)->None`, `validate_track_artifact(art)->None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/anatomy/test_tracks.py
import numpy as np
import pytest
from visdetect.anatomy.tracks import (
    ShankTrack, TrackArtifact, TrackArtifactError,
    load_track_artifact, save_track_artifact, validate_track_artifact,
)

def _shank(idx=0):
    return ShankTrack(
        probe_shank_index=idx,
        ccf_polyline=np.array([[5000., 1600., 3500.], [5000., 1600., 2500.]]),
        tip_y_um=0.0, method="brainreg_traced",
        sigma_along_um=30.0, sigma_across_um=30.0, sigma_growth_k=0.0,
        planned_entry=None, planned_vector=None,
    )

def _artifact():
    return TrackArtifact(
        subject="BG_046", atlas="allen_mouse_25um", hemisphere="right",
        barcode_orientation="forward", source_tool="brainglobe-segmentation",
        created="2026-06-17", shanks=[_shank(i) for i in range(4)],
    )

def test_roundtrip(tmp_path):
    art = _artifact()
    p = tmp_path / "BG_046_shank_tracks.json"
    save_track_artifact(art, p)
    loaded = load_track_artifact(p)
    assert loaded.subject == "BG_046"
    assert len(loaded.shanks) == 4
    np.testing.assert_allclose(loaded.shanks[0].ccf_polyline, art.shanks[0].ccf_polyline)

def test_validate_rejects_bad_method():
    art = _artifact()
    art.shanks[0].method = "guesswork"
    with pytest.raises(TrackArtifactError, match="method"):
        validate_track_artifact(art)

def test_validate_rejects_bad_orientation():
    art = _artifact()
    art.barcode_orientation = "sideways"
    with pytest.raises(TrackArtifactError, match="orientation"):
        validate_track_artifact(art)

def test_validate_rejects_wrong_polyline_shape():
    art = _artifact()
    art.shanks[0].ccf_polyline = np.zeros((3, 2))  # not (N,3)
    with pytest.raises(TrackArtifactError, match="polyline"):
        validate_track_artifact(art)

def test_load_validates(tmp_path):
    art = _artifact(); art.hemisphere = "middle"
    p = tmp_path / "bad.json"
    save_track_artifact(art, p)  # save does not validate
    with pytest.raises(TrackArtifactError):
        load_track_artifact(p)   # load does
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/anatomy/test_tracks.py -v`
Expected: FAIL (ModuleNotFoundError: visdetect.anatomy.tracks)

- [ ] **Step 3: Implement `tracks.py`**

```python
# src/visdetect/anatomy/tracks.py
"""Tool-agnostic probe-track artifact: the contract between histology tracing
(brainreg / brainglobe-segmentation / Pinpoint) and the in-repo localizer.

See docs/superpowers/specs/2026-06-17-channel-anatomical-localization-design.md (§5).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

VALID_METHODS = {"brainreg_traced", "extended_from_tip", "pinpoint_planned"}
VALID_ORIENTATIONS = {"forward", "backward"}
VALID_HEMISPHERES = {"left", "right"}


class TrackArtifactError(ValueError):
    """Raised when a track artifact violates its schema."""


@dataclass
class ShankTrack:
    probe_shank_index: int
    ccf_polyline: np.ndarray            # (N, 3) float um, (AP, ML, DV), deepest-first
    tip_y_um: float                     # channel y_um at polyline[0]
    method: str                         # one of VALID_METHODS
    sigma_along_um: float
    sigma_across_um: float
    sigma_growth_k: float               # extra sigma per um of upward extension
    planned_entry: Optional[np.ndarray] = None   # (3,) or None
    planned_vector: Optional[np.ndarray] = None  # (3,) unit-ish, points tip->entry


@dataclass
class TrackArtifact:
    subject: str
    atlas: str
    hemisphere: str
    barcode_orientation: str
    source_tool: str
    created: str
    shanks: List[ShankTrack] = field(default_factory=list)


def validate_track_artifact(art: TrackArtifact) -> None:
    if art.barcode_orientation not in VALID_ORIENTATIONS:
        raise TrackArtifactError(
            f"barcode_orientation {art.barcode_orientation!r} not in {sorted(VALID_ORIENTATIONS)}"
        )
    if art.hemisphere not in VALID_HEMISPHERES:
        raise TrackArtifactError(
            f"hemisphere {art.hemisphere!r} not in {sorted(VALID_HEMISPHERES)}"
        )
    seen = set()
    for sh in art.shanks:
        if sh.method not in VALID_METHODS:
            raise TrackArtifactError(
                f"shank {sh.probe_shank_index}: method {sh.method!r} not in {sorted(VALID_METHODS)}"
            )
        poly = np.asarray(sh.ccf_polyline)
        if poly.ndim != 2 or poly.shape[1] != 3 or poly.shape[0] < 2:
            raise TrackArtifactError(
                f"shank {sh.probe_shank_index}: ccf_polyline must be (N>=2, 3), got {poly.shape}"
            )
        if sh.probe_shank_index in seen:
            raise TrackArtifactError(f"duplicate probe_shank_index {sh.probe_shank_index}")
        seen.add(sh.probe_shank_index)
        if sh.method != "brainreg_traced" and (sh.planned_vector is None):
            raise TrackArtifactError(
                f"shank {sh.probe_shank_index}: method {sh.method!r} requires planned_vector"
            )


def _shank_to_dict(sh: ShankTrack) -> dict:
    return {
        "probe_shank_index": int(sh.probe_shank_index),
        "ccf_polyline": np.asarray(sh.ccf_polyline, float).tolist(),
        "tip_y_um": float(sh.tip_y_um),
        "method": sh.method,
        "sigma_along_um": float(sh.sigma_along_um),
        "sigma_across_um": float(sh.sigma_across_um),
        "sigma_growth_k": float(sh.sigma_growth_k),
        "planned_entry": None if sh.planned_entry is None else np.asarray(sh.planned_entry, float).tolist(),
        "planned_vector": None if sh.planned_vector is None else np.asarray(sh.planned_vector, float).tolist(),
    }


def _shank_from_dict(d: dict) -> ShankTrack:
    def _arr(x):
        return None if x is None else np.asarray(x, float)
    return ShankTrack(
        probe_shank_index=int(d["probe_shank_index"]),
        ccf_polyline=np.asarray(d["ccf_polyline"], float),
        tip_y_um=float(d["tip_y_um"]),
        method=d["method"],
        sigma_along_um=float(d["sigma_along_um"]),
        sigma_across_um=float(d["sigma_across_um"]),
        sigma_growth_k=float(d["sigma_growth_k"]),
        planned_entry=_arr(d.get("planned_entry")),
        planned_vector=_arr(d.get("planned_vector")),
    )


def save_track_artifact(art: TrackArtifact, path) -> None:
    payload = {
        "subject": art.subject, "atlas": art.atlas, "hemisphere": art.hemisphere,
        "barcode_orientation": art.barcode_orientation, "source_tool": art.source_tool,
        "created": art.created, "shanks": [_shank_to_dict(s) for s in art.shanks],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_track_artifact(path) -> TrackArtifact:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    art = TrackArtifact(
        subject=d["subject"], atlas=d["atlas"], hemisphere=d["hemisphere"],
        barcode_orientation=d["barcode_orientation"], source_tool=d["source_tool"],
        created=d["created"], shanks=[_shank_from_dict(s) for s in d["shanks"]],
    )
    validate_track_artifact(art)
    return art
```

Also create empty `src/visdetect/anatomy/__init__.py` (single line docstring).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/anatomy/test_tracks.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/anatomy/__init__.py src/visdetect/anatomy/tracks.py tests/anatomy/test_tracks.py
git commit -m "feat(anatomy): track-artifact contract (dataclasses + JSON IO + validation)"
```

---

### Task 2: Channel geometry (`channel_geometry.py`)

Assign each active channel to a shank from its x-column, and compute a stable chanmap signature so sessions sharing a bank reuse one atlas.

**Files:**
- Create: `src/visdetect/anatomy/channel_geometry.py`
- Test: `tests/anatomy/test_channel_geometry.py`

**Interfaces:**
- Consumes: `channel_positions` `(n_channels, 2)` from `tracking_qc.load_channel_positions`.
- Produces:
  - `assign_shanks(channel_positions:np.ndarray, n_shanks:int=4, gap_um:float=120.0)->np.ndarray` — per-channel int shank index `0..n_shanks-1`, ordered by ascending x (probe shank index).
  - `chanmap_signature(channel_positions:np.ndarray)->str` — stable hex hash of the rounded, sorted `(x,y)` site set (order-independent; sensitive to y-offset).

- [ ] **Step 1: Write the failing tests**

```python
# tests/anatomy/test_channel_geometry.py
import numpy as np
from visdetect.anatomy.channel_geometry import assign_shanks, chanmap_signature

def _np2_positions(y0=1515.0):
    # 4 shanks at x-base {0,250,500,750}, 2 cols per shank (+0,+32), 48 rows @15um
    xs_base = [0, 250, 500, 750]
    rows = np.arange(48) * 15.0 + y0
    pos = []
    for xb in xs_base:
        for col in (27.0, 59.0):
            for y in rows:
                pos.append([xb + col, y])
    return np.array(pos)

def test_assign_shanks_four_groups():
    pos = _np2_positions()
    sh = assign_shanks(pos)
    assert set(np.unique(sh)) == {0, 1, 2, 3}
    # lowest-x channels are shank 0
    assert sh[np.argmin(pos[:, 0])] == 0
    assert sh[np.argmax(pos[:, 0])] == 3
    # ~equal counts per shank
    counts = np.bincount(sh)
    assert counts.min() == counts.max()

def test_signature_stable_under_reorder():
    pos = _np2_positions()
    perm = np.random.RandomState(0).permutation(len(pos))
    assert chanmap_signature(pos) == chanmap_signature(pos[perm])

def test_signature_changes_with_y_offset():
    a = _np2_positions(1515.0)
    b = _np2_positions(765.0)
    assert chanmap_signature(a) != chanmap_signature(b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/anatomy/test_channel_geometry.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `channel_geometry.py`**

```python
# src/visdetect/anatomy/channel_geometry.py
"""Shank assignment + chanmap signature for NP2.0 four-shank probes."""
from __future__ import annotations

import hashlib

import numpy as np


def assign_shanks(channel_positions: np.ndarray, n_shanks: int = 4,
                  gap_um: float = 120.0) -> np.ndarray:
    """Per-channel probe shank index (0..n_shanks-1), ordered by ascending x.

    Shanks are detected as clusters of x separated by gaps > gap_um (NP2.0 shank
    pitch ~250 um, within-shank column spacing ~32 um).
    """
    x = np.asarray(channel_positions, float)[:, 0]
    order = np.argsort(np.unique(x))
    ux = np.unique(x)
    # group unique x values into shanks by gaps
    group_of_ux = np.zeros(len(ux), dtype=int)
    g = 0
    for i in range(1, len(ux)):
        if ux[i] - ux[i - 1] > gap_um:
            g += 1
        group_of_ux[i] = g
    n_found = g + 1
    if n_found != n_shanks:
        raise ValueError(f"expected {n_shanks} shanks, found {n_found} (gap_um={gap_um})")
    ux_to_group = {v: int(group_of_ux[i]) for i, v in enumerate(ux)}
    return np.array([ux_to_group[v] for v in x], dtype=int)


def chanmap_signature(channel_positions: np.ndarray) -> str:
    """Order-independent hex hash of the (x,y) site set, rounded to 1 um."""
    pos = np.round(np.asarray(channel_positions, float), 1)
    rows = sorted(map(tuple, pos.tolist()))
    h = hashlib.sha1(repr(rows).encode("utf-8")).hexdigest()
    return h[:16]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/anatomy/test_channel_geometry.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/anatomy/channel_geometry.py tests/anatomy/test_channel_geometry.py
git commit -m "feat(anatomy): shank assignment + chanmap signature"
```

---

### Task 3: Probe orientation (`orientation.py`)

Associate each traced track with its probe shank index from the documented barcode orientation + hemisphere (import-time), and provide the fail-loud monotonicity guard.

**Files:**
- Create: `src/visdetect/anatomy/orientation.py`
- Test: `tests/anatomy/test_orientation.py`

**Interfaces:**
- Consumes: `TrackArtifact`, `ShankTrack` (Task 1).
- Produces:
  - `assign_probe_shank_indices(shanks:List[ShankTrack], barcode_orientation:str, hemisphere:str, n_shanks:int=4)->List[ShankTrack]` — sets each shank's `probe_shank_index` by sorting on tip ML (polyline[0][1]); orientation/hemisphere fix the medial→lateral direction. Returns shanks sorted by probe_shank_index.
  - `validate_shank_order(art:TrackArtifact, shank_pitch_um:float=250.0, tol_um:float=120.0)->None` — raises `TrackArtifactError` if tip ML is not monotonic in probe_shank_index or spacing deviates from pitch beyond tol.

**Convention (recorded as data, not assumed):** With `barcode_orientation="forward"` we take **probe shank index increasing = ML increasing** (lateral) in the **right** hemisphere; `"backward"` reverses it; the **left** hemisphere reverses it again. This is the single place the vendor/hemisphere convention is encoded; the monotonicity guard catches a wrong entry.

- [ ] **Step 1: Write the failing tests**

```python
# tests/anatomy/test_orientation.py
import numpy as np
import pytest
from visdetect.anatomy.tracks import ShankTrack, TrackArtifact, TrackArtifactError
from visdetect.anatomy.orientation import assign_probe_shank_indices, validate_shank_order

def _shank_at_ml(ml):
    return ShankTrack(
        probe_shank_index=-1,
        ccf_polyline=np.array([[5000., ml, 3500.], [5000., ml, 2500.]]),
        tip_y_um=0.0, method="brainreg_traced",
        sigma_along_um=30., sigma_across_um=30., sigma_growth_k=0.,
    )

def _art(shanks, orientation="forward", hemi="right"):
    return TrackArtifact("BG_046", "allen_mouse_25um", hemi, orientation,
                         "test", "2026-06-17", shanks)

def test_forward_right_assigns_increasing_ml_to_increasing_index():
    shanks = [_shank_at_ml(ml) for ml in (1850, 1600, 2100, 1350)]  # unsorted
    out = assign_probe_shank_indices(shanks, "forward", "right")
    mls = [s.ccf_polyline[0, 1] for s in out]
    assert [s.probe_shank_index for s in out] == [0, 1, 2, 3]
    assert mls == sorted(mls)  # forward+right -> index 0 is most-medial (smallest ML)

def test_backward_reverses():
    shanks = [_shank_at_ml(ml) for ml in (1350, 1600, 1850, 2100)]
    out = assign_probe_shank_indices(shanks, "backward", "right")
    mls = [s.ccf_polyline[0, 1] for s in out]
    assert mls == sorted(mls, reverse=True)  # backward -> index 0 is most-lateral

def test_validate_passes_on_good_order():
    shanks = [_shank_at_ml(ml) for ml in (1350, 1600, 1850, 2100)]
    out = assign_probe_shank_indices(shanks, "forward", "right")
    validate_shank_order(_art(out))  # no raise

def test_validate_raises_on_nonmonotonic():
    shanks = [_shank_at_ml(ml) for ml in (1350, 1600, 1850, 2100)]
    out = assign_probe_shank_indices(shanks, "forward", "right")
    out[2].ccf_polyline[0, 1] = 1000.0  # break monotonicity
    with pytest.raises(TrackArtifactError, match="monoton|spacing"):
        validate_shank_order(_art(out))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/anatomy/test_orientation.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `orientation.py`**

```python
# src/visdetect/anatomy/orientation.py
"""Probe barcode-orientation handling: associate traced shanks with probe shank
indices, and guard the medial/lateral ordering. See spec §7."""
from __future__ import annotations

from typing import List

import numpy as np

from visdetect.anatomy.tracks import ShankTrack, TrackArtifact, TrackArtifactError


def _index_increasing_with_ml(barcode_orientation: str, hemisphere: str) -> bool:
    """Does probe_shank_index increase with CCF ML (True) or decrease (False)?

    Convention (recorded here, guarded by validate_shank_order): forward+right
    -> index increases with ML (index 0 = most medial / smallest ML). Each of
    {backward, left} flips it.
    """
    increasing = True
    if barcode_orientation == "backward":
        increasing = not increasing
    if hemisphere == "left":
        increasing = not increasing
    return increasing


def assign_probe_shank_indices(shanks: List[ShankTrack], barcode_orientation: str,
                               hemisphere: str, n_shanks: int = 4) -> List[ShankTrack]:
    if len(shanks) != n_shanks:
        raise ValueError(f"expected {n_shanks} shanks, got {len(shanks)}")
    ml = np.array([s.ccf_polyline[0, 1] for s in shanks])  # tip ML
    order = np.argsort(ml)  # medial(small ML) -> lateral(large ML)
    if not _index_increasing_with_ml(barcode_orientation, hemisphere):
        order = order[::-1]
    out = []
    for new_idx, src in enumerate(order):
        s = shanks[src]
        s.probe_shank_index = int(new_idx)
        out.append(s)
    return sorted(out, key=lambda s: s.probe_shank_index)


def validate_shank_order(art: TrackArtifact, shank_pitch_um: float = 250.0,
                         tol_um: float = 120.0) -> None:
    shanks = sorted(art.shanks, key=lambda s: s.probe_shank_index)
    ml = np.array([s.ccf_polyline[0, 1] for s in shanks])
    diffs = np.diff(ml)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise TrackArtifactError(f"tip ML not monotonic in shank index: {ml.tolist()}")
    if np.any(np.abs(np.abs(diffs) - shank_pitch_um) > tol_um):
        raise TrackArtifactError(
            f"shank ML spacing {np.abs(diffs).tolist()} deviates from pitch "
            f"{shank_pitch_um}±{tol_um} um"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/anatomy/test_orientation.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/anatomy/orientation.py tests/anatomy/test_orientation.py
git commit -m "feat(anatomy): probe orientation assignment + monotonicity guard"
```

---

### Task 4: Allen atlas wrapper (`atlas.py`)

Region lookup + coarse mapping + border distance, over the Allen Mouse CCF. Annotation is **injectable** so unit tests use a tiny synthetic volume (no atlas download); the default loads the real atlas via `brainglobe-atlasapi`.

**Files:**
- Create: `src/visdetect/anatomy/atlas.py`
- Test: `tests/anatomy/test_atlas.py`
- Modify: `pyproject.toml` (add `brainglobe-atlasapi` dependency)

**Interfaces:**
- Produces:
  - `COARSE_MAP: dict` (acronym/structure → coarse class in `{"CP","GPe","CTX","WM","VS","out","other"}`)
  - `class AllenAtlas` with:
    - `__init__(self, annotation:Optional[np.ndarray]=None, resolution_um:float=25.0, id_to_acronym:Optional[dict]=None, id_to_name:Optional[dict]=None, atlas_name:str="allen_mouse_25um")`
    - `region_at(self, ccf_xyz)->dict` → `{"id":int,"acronym":str,"name":str,"coarse":str}`
    - `border_distance_um(self, ccf_xyz, max_search_um:float=300.0)->float`

- [ ] **Step 1: Write the failing tests**

```python
# tests/anatomy/test_atlas.py
import numpy as np
from visdetect.anatomy.atlas import AllenAtlas, COARSE_MAP

def _toy_atlas():
    # 10x10x10 voxels @ 25um: left half region id 1 (CP), right half id 2 (GPe)
    ann = np.zeros((10, 10, 10), dtype=int)
    ann[:, :, :5] = 1
    ann[:, :, 5:] = 2
    id_to_acr = {0: "root", 1: "CP", 2: "GPe"}
    id_to_name = {0: "root", 1: "Caudoputamen", 2: "Globus pallidus external"}
    return AllenAtlas(annotation=ann, resolution_um=25.0,
                      id_to_acronym=id_to_acr, id_to_name=id_to_name)

def test_region_at_returns_acronym():
    a = _toy_atlas()
    r = a.region_at((50., 50., 25.))   # dv index 1 -> id 1 -> CP
    assert r["acronym"] == "CP"
    assert r["coarse"] == "CP"

def test_region_at_other_half():
    a = _toy_atlas()
    r = a.region_at((50., 50., 200.))  # dv index 8 -> id 2 -> GPe
    assert r["acronym"] == "GPe"

def test_out_of_volume_is_out():
    a = _toy_atlas()
    r = a.region_at((-100., 50., 25.))
    assert r["coarse"] == "out"

def test_border_distance_small_near_boundary():
    a = _toy_atlas()
    near = a.border_distance_um((50., 50., 112.))   # dv ~ index 4.5, near 4/5 border
    far = a.border_distance_um((50., 50., 12.))      # dv index 0, deep in region 1
    assert near < far

def test_coarse_map_has_core_classes():
    for acr in ("CP", "GPe"):
        assert acr in COARSE_MAP
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/anatomy/test_atlas.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `atlas.py`** (and add dependency)

Add to `pyproject.toml` dependencies: `"brainglobe-atlasapi"`.

```python
# src/visdetect/anatomy/atlas.py
"""Allen Mouse CCF region lookup over an annotation volume.

Annotation/resolution are injectable for testing; the default loads the real
atlas via brainglobe-atlasapi (cached download). Coordinates are microns
(AP, ML, DV) in atlas space.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

# Coarse classes used by analyses. Keys are Allen acronyms or acronym prefixes;
# resolution is by exact acronym first, then by the prefixes in _COARSE_PREFIXES.
COARSE_MAP: Dict[str, str] = {
    "CP": "CP",            # caudoputamen (dorsal striatum, the target)
    "ACB": "VS",           # nucleus accumbens (ventral striatum)
    "GPe": "GPe", "GPi": "GPe",
    "VL": "VS", "V3": "VS", "VS": "VS",   # ventricles
    "root": "out", "": "out",
}
# prefix fallbacks (longest match wins)
_COARSE_PREFIXES = [
    ("VIS", "CTX"), ("SS", "CTX"), ("MO", "CTX"), ("RSP", "CTX"),
    ("PTLp", "CTX"), ("ACA", "CTX"), ("AI", "CTX"),
    ("cc", "WM"), ("ec", "WM"), ("int", "WM"), ("fi", "WM"), ("or", "WM"),
    ("ccg", "WM"), ("ccb", "WM"),
]


def coarse_region(acronym: str) -> str:
    if acronym in COARSE_MAP:
        return COARSE_MAP[acronym]
    best = ("", "other")
    for pre, cls in _COARSE_PREFIXES:
        if acronym.startswith(pre) and len(pre) > len(best[0]):
            best = (pre, cls)
    return best[1]


class AllenAtlas:
    def __init__(self, annotation: Optional[np.ndarray] = None, resolution_um: float = 25.0,
                 id_to_acronym: Optional[dict] = None, id_to_name: Optional[dict] = None,
                 atlas_name: str = "allen_mouse_25um"):
        if annotation is None:
            from brainglobe_atlasapi import BrainGlobeAtlas
            bg = BrainGlobeAtlas(atlas_name)
            # BrainGlobe annotation axis order is (AP, DV, ML) ("asr"); standardize to
            # our (AP, ML, DV) convention so region_at indexing matches track coords.
            # VERIFY at implementation: the (0,2,1) transpose and the lookup-table API
            # (bg.lookup_df columns id/acronym/name) cannot be unit-tested offline.
            annotation = np.transpose(np.asarray(bg.annotation), (0, 2, 1))
            resolution_um = float(bg.resolution[0])
            lut = bg.lookup_df  # DataFrame: columns id, acronym, name
            id_to_acronym = dict(zip(lut["id"].astype(int), lut["acronym"]))
            id_to_name = dict(zip(lut["id"].astype(int), lut["name"]))
        self.annotation = np.asarray(annotation)
        self.resolution_um = float(resolution_um)
        self.id_to_acronym = id_to_acronym or {}
        self.id_to_name = id_to_name or {}

    def _voxel(self, ccf_xyz):
        return tuple(int(np.floor(c / self.resolution_um)) for c in ccf_xyz)

    def _in_bounds(self, vox) -> bool:
        return all(0 <= v < n for v, n in zip(vox, self.annotation.shape))

    def region_at(self, ccf_xyz) -> dict:
        vox = self._voxel(ccf_xyz)
        if not self._in_bounds(vox):
            return {"id": 0, "acronym": "", "name": "out of atlas", "coarse": "out"}
        rid = int(self.annotation[vox])
        acr = self.id_to_acronym.get(rid, "")
        name = self.id_to_name.get(rid, "")
        return {"id": rid, "acronym": acr, "name": name, "coarse": coarse_region(acr)}

    def border_distance_um(self, ccf_xyz, max_search_um: float = 300.0) -> float:
        """Approx distance to the nearest voxel of a different region id, by
        expanding-radius search along +/- each axis. Returns max_search_um if none."""
        vox = self._voxel(ccf_xyz)
        if not self._in_bounds(vox):
            return 0.0
        rid = int(self.annotation[vox])
        r_vox = int(np.ceil(max_search_um / self.resolution_um))
        for r in range(1, r_vox + 1):
            for ax in range(3):
                for sgn in (-1, 1):
                    nb = list(vox); nb[ax] += sgn * r
                    if self._in_bounds(tuple(nb)) and int(self.annotation[tuple(nb)]) != rid:
                        return r * self.resolution_um
        return max_search_um
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/anatomy/test_atlas.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/anatomy/atlas.py tests/anatomy/test_atlas.py pyproject.toml
git commit -m "feat(anatomy): Allen CCF atlas wrapper (region/coarse/border, injectable annotation)"
```

---

### Task 5: Localization core (`localize.py`)

Place a channel on a shank polyline by arc length, compute its uncertainty + region confidence, and build the per-channel atlas DataFrame.

**Files:**
- Create: `src/visdetect/anatomy/localize.py`
- Test: `tests/anatomy/test_localize.py`

**Interfaces:**
- Consumes: `ShankTrack`, `TrackArtifact` (Task 1); `assign_shanks` (Task 2); `AllenAtlas` (Task 4).
- Produces:
  - `place_channel_on_track(track:ShankTrack, y_um:float)->Tuple[np.ndarray, float]` → `(ccf_xyz (3,), sigma_um)`. Arc-length `s = y_um - track.tip_y_um` from polyline[0]; interpolate within the polyline; beyond its top, extrapolate along `planned_vector` (or last segment direction) and grow sigma by `sigma_growth_k * overshoot`.
  - `region_confidence(sigma_um:float, border_distance_um:float)->float` → `Φ(border_distance/max(sigma,1e-3))`, clipped `[0,1]`.
  - `build_channel_atlas(subject:str, art:TrackArtifact, channel_positions:np.ndarray, signature:str, atlas:AllenAtlas)->pd.DataFrame` with columns: `subject, chanmap_signature, channel, shank, x_um, y_um, ccf_ap, ccf_ml, ccf_dv, sigma_um, region_acronym, region_name, region_coarse, region_confidence, loc_method`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/anatomy/test_localize.py
import numpy as np
import pytest
from visdetect.anatomy.tracks import ShankTrack
from visdetect.anatomy.localize import (
    place_channel_on_track, region_confidence, build_channel_atlas,
)

def _straight_shank(idx=0, ml=1600.0):
    # deepest at DV=3500 (tip, y=0), top at DV=2500 (y=1000); straight in DV
    return ShankTrack(
        probe_shank_index=idx,
        ccf_polyline=np.array([[5000., ml, 3500.], [5000., ml, 2500.]]),
        tip_y_um=0.0, method="extended_from_tip",
        sigma_along_um=20., sigma_across_um=20., sigma_growth_k=0.1,
        planned_entry=None, planned_vector=np.array([0., 0., -1.0]),
    )

def test_place_within_polyline():
    sh = _straight_shank()
    xyz, sig = place_channel_on_track(sh, 500.0)  # halfway
    np.testing.assert_allclose(xyz, [5000., 1600., 3000.], atol=1e-6)
    assert sig == pytest.approx(20.0)

def test_place_extrapolates_above_with_growing_sigma():
    sh = _straight_shank()
    xyz, sig = place_channel_on_track(sh, 1200.0)  # 200 um above the top (y=1000)
    np.testing.assert_allclose(xyz, [5000., 1600., 2300.], atol=1e-6)
    assert sig > 20.0  # grew by sigma_growth_k * 200

def test_region_confidence_monotonic():
    assert region_confidence(30., 5.) < region_confidence(30., 200.)
    assert 0.0 <= region_confidence(30., 5.) <= 1.0

def test_build_channel_atlas_columns_and_rows():
    from test_channel_geometry import _np2_positions  # bare: tests/anatomy on sys.path (prepend mode)
    from visdetect.anatomy.tracks import TrackArtifact
    from visdetect.anatomy.atlas import AllenAtlas
    pos = _np2_positions()
    art = TrackArtifact("BG_046", "allen_mouse_25um", "right", "forward",
                        "test", "2026-06-17",
                        [_straight_shank(i, ml=1350. + 250. * i) for i in range(4)])
    ann = np.ones((400, 200, 200), dtype=int)  # all region id 1
    atlas = AllenAtlas(annotation=ann, resolution_um=25.0,
                       id_to_acronym={1: "CP"}, id_to_name={1: "Caudoputamen"})
    df = build_channel_atlas("BG_046", art, pos, "sigABC", atlas)
    assert len(df) == len(pos)
    for c in ("ccf_ap", "ccf_ml", "ccf_dv", "region_acronym", "region_confidence",
              "shank", "loc_method", "chanmap_signature"):
        assert c in df.columns
    assert (df["chanmap_signature"] == "sigABC").all()
    assert set(df["shank"].unique()) == {0, 1, 2, 3}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/anatomy/test_localize.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `localize.py`**

```python
# src/visdetect/anatomy/localize.py
"""Place channels on shank polylines -> CCF + region + confidence; build atlas."""
from __future__ import annotations

from math import erf, sqrt
from typing import Tuple

import numpy as np
import pandas as pd

from visdetect.anatomy.atlas import AllenAtlas
from visdetect.anatomy.channel_geometry import assign_shanks
from visdetect.anatomy.tracks import ShankTrack, TrackArtifact

ATLAS_COLUMNS = [
    "subject", "chanmap_signature", "channel", "shank", "x_um", "y_um",
    "ccf_ap", "ccf_ml", "ccf_dv", "sigma_um",
    "region_acronym", "region_name", "region_coarse", "region_confidence", "loc_method",
]


def place_channel_on_track(track: ShankTrack, y_um: float) -> Tuple[np.ndarray, float]:
    poly = np.asarray(track.ccf_polyline, float)
    seg = np.diff(poly, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])  # arc length from polyline[0]
    s = float(y_um - track.tip_y_um)
    total = cum[-1]
    if s <= 0:
        return poly[0].copy(), track.sigma_along_um
    if s <= total:
        j = int(np.searchsorted(cum, s) - 1)
        j = max(0, min(j, len(seg) - 1))
        frac = (s - cum[j]) / seg_len[j] if seg_len[j] > 0 else 0.0
        xyz = poly[j] + frac * seg[j]
        return xyz, track.sigma_along_um
    # extrapolate above the top of the traced polyline
    overshoot = s - total
    if track.planned_vector is not None:
        direction = np.asarray(track.planned_vector, float)
    else:
        direction = seg[-1]
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    xyz = poly[-1] + overshoot * direction
    sigma = track.sigma_along_um + track.sigma_growth_k * overshoot
    return xyz, sigma


def region_confidence(sigma_um: float, border_distance_um: float) -> float:
    z = border_distance_um / max(sigma_um, 1e-3)
    cdf = 0.5 * (1.0 + erf(z / sqrt(2.0)))   # P(jittered location stays on this side)
    return float(min(1.0, max(0.0, cdf)))


def build_channel_atlas(subject: str, art: TrackArtifact, channel_positions: np.ndarray,
                        signature: str, atlas: AllenAtlas) -> pd.DataFrame:
    pos = np.asarray(channel_positions, float)
    shank_of = assign_shanks(pos)
    track_by_idx = {s.probe_shank_index: s for s in art.shanks}
    rows = []
    for ch in range(len(pos)):
        x_um, y_um = float(pos[ch, 0]), float(pos[ch, 1])
        sh = int(shank_of[ch])
        track = track_by_idx[sh]
        xyz, sigma = place_channel_on_track(track, y_um)
        reg = atlas.region_at(xyz)
        bd = atlas.border_distance_um(xyz)
        rows.append({
            "subject": subject, "chanmap_signature": signature, "channel": ch,
            "shank": sh, "x_um": x_um, "y_um": y_um,
            "ccf_ap": float(xyz[0]), "ccf_ml": float(xyz[1]), "ccf_dv": float(xyz[2]),
            "sigma_um": float(sigma),
            "region_acronym": reg["acronym"], "region_name": reg["name"],
            "region_coarse": reg["coarse"],
            "region_confidence": region_confidence(sigma, bd),
            "loc_method": track.method,
        })
    return pd.DataFrame(rows, columns=ATLAS_COLUMNS)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/anatomy/test_localize.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/anatomy/localize.py tests/anatomy/test_localize.py
git commit -m "feat(anatomy): localization core (place-on-track, confidence, channel atlas)"
```

---

### Task 6: Peak channel (`peak_channel.py`)

Per-unit peak channel from RawWaveforms (primary), KS templates (fallback). Mostly wraps `tracking_qc` primitives.

**Files:**
- Create: `src/visdetect/anatomy/peak_channel.py`
- Test: `tests/anatomy/test_peak_channel.py`

**Interfaces:**
- Consumes: `tracking_qc.load_raw_mean_waveform`, `tracking_qc.extract_peak_channel`.
- Produces:
  - `unit_peak_channel(raw_wf_root, session_name:str, cluster_id:int)->Optional[int]` — RawWaveform primary; returns `None` if neither RawWaveform nor a KS-template fallback resolves.
  - `peak_channel_from_mean(mean_waveform:np.ndarray)->int` — thin reuse of `extract_peak_channel` (kept here so callers don't depend on tracking_qc directly).

- [ ] **Step 1: Write the failing tests**

```python
# tests/anatomy/test_peak_channel.py
import numpy as np
from pathlib import Path
from visdetect.anatomy.peak_channel import peak_channel_from_mean, unit_peak_channel

def test_peak_channel_from_mean_known():
    mw = np.zeros((82, 10))             # (samples, channels)
    mw[40, 7] = -5.0; mw[50, 7] = 4.0   # biggest peak-to-peak on channel 7
    assert peak_channel_from_mean(mw) == 7

def test_unit_peak_channel_reads_rawwaveform(tmp_path):
    sess = tmp_path / "01072025" / "RawWaveforms"
    sess.mkdir(parents=True)
    raw = np.zeros((82, 10, 2))
    raw[40, 3, :] = -6.0; raw[50, 3, :] = 5.0   # channel 3 dominant
    np.save(sess / "Unit42_RawSpikes.npy", raw)
    pc = unit_peak_channel(tmp_path, "01072025", 42)
    assert pc == 3

def test_unit_peak_channel_missing_returns_none(tmp_path):
    assert unit_peak_channel(tmp_path, "01072025", 999) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/anatomy/test_peak_channel.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `peak_channel.py`**

```python
# src/visdetect/anatomy/peak_channel.py
"""Per-unit peak channel. RawWaveforms primary (reuses tracking_qc), KS templates
fallback. See spec §3/§6."""
from __future__ import annotations

from typing import Optional

import numpy as np

from visdetect.analysis.tracking_qc import extract_peak_channel, load_raw_mean_waveform


def peak_channel_from_mean(mean_waveform: np.ndarray) -> int:
    return int(extract_peak_channel(np.asarray(mean_waveform)))


def unit_peak_channel(raw_wf_root, session_name: str, cluster_id: int) -> Optional[int]:
    mw = load_raw_mean_waveform(raw_wf_root, session_name, cluster_id)
    if mw is None:
        return None
    return peak_channel_from_mean(mw)
```

Note: `load_raw_mean_waveform` returns `None` when the file is absent and tries the
session name plus its `zfill(8)` form. It does **not** itself resolve subject-prefixed
dirs (`BG_031_01042025`); the CLIs pass an already-resolved dir name (Task 9). The
KS-template fallback is intentionally deferred — RawWaveform coverage is expected for
all subjects (spec §3), so add it only if a subject is found to lack RawWaveforms.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/anatomy/test_peak_channel.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/anatomy/peak_channel.py tests/anatomy/test_peak_channel.py
git commit -m "feat(anatomy): per-unit peak channel from RawWaveforms"
```

---

### Task 7: Unit-table schema + loader integration

Register the anatomy columns in the frozen contract and merge per-unit anatomy into `build_unit_table` (mirrors the celltype merge precedent).

**Files:**
- Modify: `src/visdetect/suite/unit_table_schema.py`
- Modify: `src/visdetect/suite/loader.py`
- Test: `tests/suite/test_unit_table_anatomy.py`

**Interfaces:**
- Consumes: per-unit anatomy CSV at `data/anatomy/unit_anatomy.csv` (produced by Task 9) with columns `session_name, cluster_id, peak_channel, shank, depth_um, ccf_ap, ccf_ml, ccf_dv, region_acronym, region_name, region_coarse, region_confidence, loc_method`.
- Produces:
  - `unit_table_schema`: new `ANATOMY_DEFAULTS` dict folded into `LABEL_DEFAULTS`/`CONTRACT_COLUMNS`; `region_coarse` added to `ALLOWED_VALUES`.
  - `loader.load_unit_anatomy(path:Optional[str]=None)->pd.DataFrame` (empty DataFrame if file missing).
  - `build_unit_table` merges anatomy by `(Session_Date, Cluster_ID)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/suite/test_unit_table_anatomy.py
import pandas as pd
import pytest
from visdetect.suite.unit_table_schema import (
    LABEL_DEFAULTS, CONTRACT_COLUMNS, ALLOWED_VALUES,
    add_label_defaults, validate_unit_table, UnitTableContractError,
)

def _minimal_table():
    return pd.DataFrame({
        "Session_Date": [7072025], "Cluster_ID": [3],
        "Global_UID": [1], "stage": ["Expert"], "session_idx": [0],
    })

def test_anatomy_defaults_present_after_add():
    df = add_label_defaults(_minimal_table())
    for c in ("region_coarse", "ccf_ap", "ccf_ml", "ccf_dv", "region_confidence", "loc_method"):
        assert c in df.columns

def test_contract_includes_anatomy():
    assert "region_coarse" in CONTRACT_COLUMNS

def test_region_coarse_value_check():
    df = add_label_defaults(_minimal_table())
    df["region_coarse"] = "Mars"
    with pytest.raises(UnitTableContractError, match="region_coarse"):
        validate_unit_table(df)
```

```python
# (same file) loader integration — hermetic (no real data junctions in a worktree).
def test_build_unit_table_has_anatomy_columns(monkeypatch):
    import pandas as pd
    from visdetect.suite import loader as L
    minimal = pd.DataFrame({
        "Session_Date": [7072025], "Cluster_ID": [3],
        "Global_UID": [1], "stage": ["Expert"], "session_idx": [0],
    })
    monkeypatch.setattr(L, "load_glt", lambda qc_only=True: minimal.copy())
    df = L.build_unit_table(qc_only=True, validate=True)
    for c in ("region_coarse", "ccf_ap", "ccf_dv", "loc_method"):
        assert c in df.columns
    # no anatomy file present -> defaults
    assert df.loc[0, "region_coarse"] == "unknown"
    assert df.loc[0, "loc_method"] == "none"
```

The other loaders (`load_all_lick_responsiveness`, `load_tf_*`, `load_waveform_labels`,
`load_unit_anatomy`, the verdicts block) all degrade to empty/skip when their data
files are absent, so monkeypatching `load_glt` is enough to make this hermetic.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/suite/test_unit_table_anatomy.py -v`
Expected: FAIL (region_coarse not in CONTRACT_COLUMNS)

- [ ] **Step 3: Implement schema + loader changes**

In `src/visdetect/suite/unit_table_schema.py`, after `LABEL_DEFAULTS`:

```python
# Anatomy localization columns (one workstream, several columns: a region label
# plus its CCF coordinates / confidence / method). Defaults mark not-yet-localized rows.
ANATOMY_DEFAULTS: Dict[str, object] = {
    "peak_channel": -1,
    "shank": -1,
    "depth_um": float("nan"),
    "ccf_ap": float("nan"),
    "ccf_ml": float("nan"),
    "ccf_dv": float("nan"),
    "region_acronym": "unknown",
    "region_name": "unknown",
    "region_coarse": "unknown",
    "region_confidence": float("nan"),
    "loc_method": "none",
}
LABEL_DEFAULTS.update(ANATOMY_DEFAULTS)

ALLOWED_VALUES["region_coarse"] = {
    "CP", "GPe", "CTX", "WM", "VS", "out", "other", "unknown",
}
```

(`CONTRACT_COLUMNS` is computed from `LABEL_DEFAULTS`, so it picks these up automatically.)

In `src/visdetect/suite/loader.py`, add a loader near `load_waveform_labels`:

```python
def load_unit_anatomy(path: Optional[str] = None) -> pd.DataFrame:
    """Per-unit anatomical localization (produced by scripts/anatomy/localize_units.py).

    Returns an empty DataFrame if the file does not exist yet.
    """
    from .config import ROOT   # repo root; loader.py already imports ROOT from .config
    p = path or os.path.join(ROOT, "data", "anatomy", "unit_anatomy.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_csv(p)
```

(`ROOT` is the verified config symbol the loader already uses, e.g. line ~115
`os.path.join(ROOT, "data", "pkls", subj)`.)

Then, in `build_unit_table`, immediately before the `# ── Add not-yet-produced
contract columns` block, add the anatomy merge (mirrors the celltype merge):

```python
    # Merge anatomical localization (peak channel -> CCF + region).
    anat = load_unit_anatomy()
    anat_cols = ["peak_channel", "shank", "depth_um", "ccf_ap", "ccf_ml", "ccf_dv",
                 "region_acronym", "region_name", "region_coarse",
                 "region_confidence", "loc_method"]
    if not anat.empty and {"session_name", "cluster_id"}.issubset(anat.columns):
        anat_sub = anat[["session_name", "cluster_id"] + anat_cols].copy()
        anat_sub["session_name"] = anat_sub["session_name"].astype(int)
        anat_sub["cluster_id"] = anat_sub["cluster_id"].astype(int)
        glt = glt.drop(columns=anat_cols, errors="ignore")
        glt = glt.merge(
            anat_sub, left_on=["Session_Date", "Cluster_ID"],
            right_on=["session_name", "cluster_id"], how="left",
        )
        glt.drop(columns=["session_name", "cluster_id"], errors="ignore", inplace=True)
        # Unmatched rows (sessions without a track artifact yet) get clean defaults,
        # mirroring the celltype merge. CCF coords / confidence stay NaN.
        for c, dflt in (("region_acronym", "unknown"), ("region_name", "unknown"),
                        ("region_coarse", "unknown"), ("loc_method", "none"),
                        ("peak_channel", -1), ("shank", -1)):
            glt[c] = glt[c].fillna(dflt)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/suite/test_unit_table_anatomy.py -v`
Expected: PASS (4 tests). The `build_unit_table` test passes because `add_label_defaults` fills anatomy columns even when `data/anatomy/unit_anatomy.csv` is absent.

- [ ] **Step 5: Run the full unit-table test set to confirm no contract regressions**

Run: `.venv/Scripts/python.exe -m pytest tests/ -k "unit_table or schema" -v`
Expected: PASS (existing contract tests still green).

- [ ] **Step 6: Commit**

```bash
git add src/visdetect/suite/unit_table_schema.py src/visdetect/suite/loader.py tests/suite/test_unit_table_anatomy.py
git commit -m "feat(anatomy): register anatomy columns in unit-table contract + merge in build_unit_table"
```

---

### Task 8: CLI — build the channel atlas (`build_channel_atlas.py`)

Orchestrate: load a subject's track artifact + each session's `channel_positions.npy`, compute one channel atlas per distinct chanmap signature, write `data/anatomy/<subject>_channel_atlas.csv` + a session→signature map.

**Files:**
- Create: `scripts/anatomy/build_channel_atlas.py`
- Test: `tests/anatomy/test_build_channel_atlas_cli.py`

**Interfaces:**
- Consumes: `load_track_artifact`, `validate_shank_order`, `chanmap_signature`, `build_channel_atlas`, `AllenAtlas`, `tracking_qc.load_channel_positions`.
- Produces (function, so it's testable without `argv`):
  - `build_subject_atlas(subject:str, artifact_path, raw_wf_root, session_names:List[str], atlas:AllenAtlas, out_dir)->pd.DataFrame` — writes `<subject>_channel_atlas.csv` (one row-set per unique signature) and `<subject>_session_signatures.csv` (`session_name,chanmap_signature`); returns the atlas DataFrame.

- [ ] **Step 1: Write the failing test**

```python
# tests/anatomy/test_build_channel_atlas_cli.py
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "anatomy"))
from build_channel_atlas import build_subject_atlas
from visdetect.anatomy.tracks import TrackArtifact, ShankTrack, save_track_artifact
from visdetect.anatomy.atlas import AllenAtlas
from test_channel_geometry import _np2_positions  # bare: tests/anatomy on sys.path (prepend mode)

def _artifact():
    shanks = [ShankTrack(i, np.array([[5000., 1350.+250.*i, 3500.],
                                      [5000., 1350.+250.*i, 2500.]]),
                         0.0, "brainreg_traced", 20., 20., 0.0,
                         None, np.array([0., 0., -1.])) for i in range(4)]
    return TrackArtifact("BG_046", "allen_mouse_25um", "right", "forward",
                         "test", "2026-06-17", shanks)

def test_build_subject_atlas_writes_files(tmp_path):
    # two sessions, same geometry -> one signature
    raw = tmp_path / "raw"
    for s in ("01072025", "02072025"):
        d = raw / s; d.mkdir(parents=True)
        np.save(d / "channel_positions.npy", _np2_positions())
    art_p = tmp_path / "BG_046_shank_tracks.json"
    save_track_artifact(_artifact(), art_p)
    ann = np.ones((400, 200, 200), dtype=int)
    atlas = AllenAtlas(annotation=ann, resolution_um=25.0,
                       id_to_acronym={1: "CP"}, id_to_name={1: "Caudoputamen"})
    out = tmp_path / "anatomy"
    df = build_subject_atlas("BG_046", art_p, raw, ["01072025", "02072025"], atlas, out)
    assert (out / "BG_046_channel_atlas.csv").exists()
    sig_map = pd.read_csv(out / "BG_046_session_signatures.csv")
    assert sig_map["chanmap_signature"].nunique() == 1  # shared bank
    assert df["region_coarse"].eq("CP").all()

def test_session_token_and_resolve(tmp_path):
    from build_channel_atlas import session_token, resolve_session_dir
    assert session_token("BG_031_01042025") == "01042025"
    assert session_token("01072025") == "01072025"
    (tmp_path / "BG_031_01042025").mkdir()
    assert resolve_session_dir(tmp_path, "01042025") == "BG_031_01042025"
    assert resolve_session_dir(tmp_path, "99999999") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/anatomy/test_build_channel_atlas_cli.py -v`
Expected: FAIL (ImportError build_channel_atlas)

- [ ] **Step 3: Implement `build_channel_atlas.py`**

```python
# scripts/anatomy/build_channel_atlas.py
"""Build a subject's per-channel CCF/region atlas from its track artifact.

Usage:
    py scripts/anatomy/build_channel_atlas.py --subject BG_046
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from visdetect.anatomy.atlas import AllenAtlas
from visdetect.anatomy.channel_geometry import chanmap_signature
from visdetect.anatomy.localize import build_channel_atlas
from visdetect.anatomy.orientation import validate_shank_order
from visdetect.anatomy.tracks import load_track_artifact
from visdetect.analysis.tracking_qc import load_channel_positions


def session_token(name) -> str:
    """Numeric date token (DDMMYYYY) from a session dir / pkl name.

    'BG_031_01042025' -> '01042025'; '01072025' -> '01072025'.
    """
    m = re.search(r"(\d{6,8})$", str(name))
    return m.group(1) if m else str(name)


def resolve_session_dir(raw_wf_root, token) -> Optional[str]:
    """Actual session subdir under raw_wf_root matching a date token, handling
    bare ('01072025') and subject-prefixed ('BG_031_01042025') dir names."""
    root = str(raw_wf_root)
    cands = {str(token), str(token).zfill(8)}
    if not os.path.isdir(root):
        return None
    for d in sorted(os.listdir(root)):
        if not os.path.isdir(os.path.join(root, d)):
            continue
        if d in cands or session_token(d) in cands:
            return d
    return None


def build_subject_atlas(subject, artifact_path, raw_wf_root, session_names: List[str],
                        atlas: AllenAtlas, out_dir) -> pd.DataFrame:
    art = load_track_artifact(artifact_path)
    validate_shank_order(art)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    sig_rows, atlas_by_sig = [], {}
    for name in session_names:
        token = session_token(name)
        sess_dir = resolve_session_dir(raw_wf_root, token) or name
        pos = load_channel_positions(raw_wf_root, sess_dir)
        if pos is None:
            print(f"  {name}: no channel_positions, skipping")
            continue
        sig = chanmap_signature(pos)
        sig_rows.append({"session_name": token, "chanmap_signature": sig})
        if sig not in atlas_by_sig:
            atlas_by_sig[sig] = build_channel_atlas(subject, art, pos, sig, atlas)

    atlas_df = (pd.concat(atlas_by_sig.values(), ignore_index=True)
                if atlas_by_sig else pd.DataFrame())
    atlas_df.to_csv(out_dir / f"{subject}_channel_atlas.csv", index=False)
    pd.DataFrame(sig_rows).to_csv(out_dir / f"{subject}_session_signatures.csv", index=False)
    print(f"{subject}: {len(atlas_by_sig)} unique chanmap(s), "
          f"{len(sig_rows)} sessions -> {out_dir}")
    return atlas_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--raw-wf-root", default=None,
                    help="defaults to data/unit_match/input/<subject>")
    ap.add_argument("--artifact", default=None,
                    help="defaults to data/anatomy/<subject>_shank_tracks.json")
    ap.add_argument("--out-dir", default="data/anatomy")
    args = ap.parse_args()

    raw_root = args.raw_wf_root or os.path.join("data", "unit_match", "input", args.subject)
    artifact = args.artifact or os.path.join("data", "anatomy", f"{args.subject}_shank_tracks.json")
    sessions = sorted(d for d in os.listdir(raw_root)
                      if os.path.isdir(os.path.join(raw_root, d)))
    atlas = AllenAtlas()  # real Allen atlas (downloads/caches on first use)
    build_subject_atlas(args.subject, artifact, raw_root, sessions, atlas, args.out_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/anatomy/test_build_channel_atlas_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/anatomy/build_channel_atlas.py tests/anatomy/test_build_channel_atlas_cli.py
git commit -m "feat(anatomy): CLI to build per-subject channel atlas from track artifact"
```

---

### Task 9: CLI — localize units (`localize_units.py`)

Join units to the channel atlas via peak channel; emit `data/anatomy/unit_anatomy.csv` (what `load_unit_anatomy` reads).

**Files:**
- Create: `scripts/anatomy/localize_units.py`
- Test: `tests/anatomy/test_localize_units_cli.py`

**Interfaces:**
- Consumes: `<subject>_channel_atlas.csv` + `<subject>_session_signatures.csv` (Task 8), `unit_peak_channel` (Task 6), the per-unit cluster ids (from `load_glt`/manifest restricted to the subject, or `good_and_stable_ids` per session).
- Produces (testable function):
  - `localize_subject_units(subject, atlas_csv, sig_csv, raw_wf_root, units_by_session:Dict[str,List[int]])->pd.DataFrame` with columns `session_name, cluster_id, peak_channel, shank, depth_um, ccf_ap, ccf_ml, ccf_dv, region_acronym, region_name, region_coarse, region_confidence, loc_method`. `depth_um` = `channel_positions[peak_channel, 1]` via the atlas row's `y_um`.
  - `append_unit_anatomy(df:pd.DataFrame, out_csv)->None` — upsert into `data/anatomy/unit_anatomy.csv` keyed by `(session_name, cluster_id)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/anatomy/test_localize_units_cli.py
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "anatomy"))
from localize_units import localize_subject_units

def test_localize_units_joins_via_peak_channel(tmp_path):
    # channel atlas: 4 channels, channel 2 is CP at known coords
    atlas = pd.DataFrame({
        "subject": "BG_046", "chanmap_signature": "sigA",
        "channel": [0, 1, 2, 3], "shank": [0, 0, 0, 0],
        "x_um": [27., 27., 27., 27.], "y_um": [100., 200., 300., 400.],
        "ccf_ap": [5000.]*4, "ccf_ml": [1600.]*4, "ccf_dv": [3400., 3300., 3200., 3100.],
        "sigma_um": [20.]*4,
        "region_acronym": ["CP"]*4, "region_name": ["Caudoputamen"]*4,
        "region_coarse": ["CP"]*4, "region_confidence": [0.9]*4, "loc_method": ["brainreg_traced"]*4,
    })
    atlas_csv = tmp_path / "BG_046_channel_atlas.csv"; atlas.to_csv(atlas_csv, index=False)
    sig = pd.DataFrame({"session_name": ["01072025"], "chanmap_signature": ["sigA"]})
    sig_csv = tmp_path / "BG_046_session_signatures.csv"; sig.to_csv(sig_csv, index=False)
    # raw waveform for unit 42 peaking on channel 2
    rw = tmp_path / "01072025" / "RawWaveforms"; rw.mkdir(parents=True)
    raw = np.zeros((82, 4, 2)); raw[40, 2, :] = -5.; raw[50, 2, :] = 4.
    np.save(rw / "Unit42_RawSpikes.npy", raw)

    df = localize_subject_units("BG_046", atlas_csv, sig_csv, tmp_path,
                                {"01072025": [42]})
    assert len(df) == 1
    row = df.iloc[0]
    assert row["peak_channel"] == 2
    assert row["region_coarse"] == "CP"
    assert row["depth_um"] == 300.0
    assert row["ccf_dv"] == 3200.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/anatomy/test_localize_units_cli.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `localize_units.py`**

```python
# scripts/anatomy/localize_units.py
"""Localize units: peak channel -> channel atlas row -> per-unit CCF/region.

Usage:
    py scripts/anatomy/localize_units.py --subject BG_046
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

from visdetect.anatomy.peak_channel import unit_peak_channel

UNIT_COLS = ["session_name", "cluster_id", "peak_channel", "shank", "depth_um",
             "ccf_ap", "ccf_ml", "ccf_dv", "region_acronym", "region_name",
             "region_coarse", "region_confidence", "loc_method"]


def localize_subject_units(subject, atlas_csv, sig_csv, raw_wf_root,
                           units_by_session: Dict[str, List[int]]) -> pd.DataFrame:
    from build_channel_atlas import resolve_session_dir, session_token
    atlas = pd.read_csv(atlas_csv)
    # dtype=str + zfill(8): session tokens like "01072025" otherwise read back as
    # int 1072025 (leading zero dropped) and the join silently misses.
    sig_df = pd.read_csv(sig_csv, dtype={"session_name": str})
    sig = {str(k).zfill(8): v
           for k, v in zip(sig_df["session_name"], sig_df["chanmap_signature"])}
    rows = []
    for sess, cluster_ids in units_by_session.items():
        token = session_token(sess).zfill(8)
        signature = sig.get(token)
        if signature is None:
            continue
        sess_dir = resolve_session_dir(raw_wf_root, token) or str(sess)
        chans = atlas[atlas["chanmap_signature"] == signature].set_index("channel")
        for cid in cluster_ids:
            pc = unit_peak_channel(raw_wf_root, sess_dir, cid)
            if pc is None or pc not in chans.index:
                continue
            a = chans.loc[pc]
            rows.append({
                "session_name": int(token), "cluster_id": int(cid), "peak_channel": int(pc),
                "shank": int(a["shank"]), "depth_um": float(a["y_um"]),
                "ccf_ap": float(a["ccf_ap"]), "ccf_ml": float(a["ccf_ml"]),
                "ccf_dv": float(a["ccf_dv"]),
                "region_acronym": a["region_acronym"], "region_name": a["region_name"],
                "region_coarse": a["region_coarse"],
                "region_confidence": float(a["region_confidence"]),
                "loc_method": a["loc_method"],
            })
    return pd.DataFrame(rows, columns=UNIT_COLS)


def append_unit_anatomy(df: pd.DataFrame, out_csv) -> None:
    out_csv = Path(out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        combined = pd.concat([prev, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["session_name", "cluster_id"], keep="last")
    else:
        combined = df
    combined.to_csv(out_csv, index=False)


def _units_by_session_for_subject(subject) -> Dict[str, List[int]]:
    """Per-session good_and_stable cluster ids from the subject's PKLs, keyed by
    numeric date token. Loads pkls by explicit path (avoids the SUBJECT-env-scoped
    suite.loader.load_session, which resolves only the active subject)."""
    import glob
    from visdetect.core.session import load_session   # path-based loader
    from visdetect.suite.config import ROOT
    from build_channel_atlas import session_token
    out: Dict[str, List[int]] = {}
    pkl_dir = os.path.join(ROOT, "data", "pkls", subject)
    for path in sorted(glob.glob(os.path.join(pkl_dir, f"{subject}_*.pkl"))):
        token = session_token(os.path.basename(path)[:-4])
        if not token.isdigit():
            continue   # skip variants/backups (e.g. *_preconsolidate, *_b)
        sess = load_session(path)
        ids = sess.good_and_stable_ids or [c.cluster_id for c in sess.clusters]
        out[token] = [int(i) for i in ids]
        del sess
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--raw-wf-root", default=None)
    ap.add_argument("--anatomy-dir", default="data/anatomy")
    args = ap.parse_args()
    raw_root = args.raw_wf_root or os.path.join("data", "unit_match", "input", args.subject)
    atlas_csv = os.path.join(args.anatomy_dir, f"{args.subject}_channel_atlas.csv")
    sig_csv = os.path.join(args.anatomy_dir, f"{args.subject}_session_signatures.csv")
    units = _units_by_session_for_subject(args.subject)
    df = localize_subject_units(args.subject, atlas_csv, sig_csv, raw_root, units)
    append_unit_anatomy(df, os.path.join(args.anatomy_dir, "unit_anatomy.csv"))
    print(f"{args.subject}: localized {len(df)} units -> {args.anatomy_dir}/unit_anatomy.csv")


if __name__ == "__main__":
    main()
```

(Verified: `visdetect.core.session.load_session(path: str) -> Session` is path-based —
distinct from `visdetect.suite.loader.load_session(name)`, which is SUBJECT-scoped.
Use the **core** one here.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/anatomy/test_localize_units_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/anatomy/localize_units.py tests/anatomy/test_localize_units_cli.py
git commit -m "feat(anatomy): CLI to localize units (peak channel -> unit_anatomy.csv)"
```

---

### Task 10: CLI — QC figure (`plot_shank_anatomy.py`)

Per-subject validation figure: each shank's region bands along depth, channels coloured by region, units at their depths, and the cortex↔striatum transition with its ±σ band.

**Files:**
- Create: `scripts/anatomy/plot_shank_anatomy.py`
- Test: `tests/anatomy/test_plot_shank_anatomy.py`

**Interfaces:**
- Consumes: `<subject>_channel_atlas.csv` (Task 8), optionally `unit_anatomy.csv` (Task 9).
- Produces (testable function): `plot_subject_anatomy(subject, atlas_csv, out_png, unit_csv=None)->str` (returns the PNG path; uses a non-interactive matplotlib backend).

- [ ] **Step 1: Write the failing test**

```python
# tests/anatomy/test_plot_shank_anatomy.py
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "anatomy"))
from plot_shank_anatomy import plot_subject_anatomy

def test_plot_produces_png(tmp_path):
    atlas = pd.DataFrame({
        "subject": "BG_046", "chanmap_signature": "sigA",
        "channel": range(8), "shank": [0]*4 + [1]*4,
        "x_um": [27.]*8, "y_um": [100., 200., 300., 400.]*2,
        "ccf_ap": [5000.]*8, "ccf_ml": [1600.]*8, "ccf_dv": [3400., 3300., 3200., 3100.]*2,
        "sigma_um": [20.]*8,
        "region_acronym": ["CP"]*8, "region_name": ["Caudoputamen"]*8,
        "region_coarse": ["CTX", "CTX", "CP", "CP"]*2,
        "region_confidence": [0.9]*8, "loc_method": ["brainreg_traced"]*8,
    })
    csv = tmp_path / "BG_046_channel_atlas.csv"; atlas.to_csv(csv, index=False)
    out = tmp_path / "fig.png"
    path = plot_subject_anatomy("BG_046", csv, out)
    assert os.path.exists(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/anatomy/test_plot_shank_anatomy.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `plot_shank_anatomy.py`**

```python
# scripts/anatomy/plot_shank_anatomy.py
"""QC figure: region-by-depth per shank for a subject. See spec §9."""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COARSE_COLORS = {
    "CP": "#2c7fb8", "GPe": "#d95f0e", "CTX": "#7fbc41", "WM": "#999999",
    "VS": "#9e9ac8", "out": "#000000", "other": "#cccccc", "unknown": "#eeeeee",
}


def plot_subject_anatomy(subject, atlas_csv, out_png, unit_csv=None) -> str:
    atlas = pd.read_csv(atlas_csv)
    shanks = sorted(atlas["shank"].unique())
    fig, axes = plt.subplots(1, len(shanks), figsize=(2.2 * len(shanks), 7),
                             sharey=True, squeeze=False)
    for ax, sh in zip(axes[0], shanks):
        d = atlas[atlas["shank"] == sh].sort_values("y_um")
        for _, r in d.iterrows():
            ax.scatter(0, r["y_um"], s=18,
                       color=COARSE_COLORS.get(r["region_coarse"], "#cccccc"))
        # mark CTX->CP transition (most dorsal CP channel)
        cp = d[d["region_coarse"] == "CP"]
        if not cp.empty:
            yt = cp["y_um"].max()
            sig = float(d.loc[cp["y_um"].idxmax(), "sigma_um"])
            ax.axhspan(yt - sig, yt + sig, color="red", alpha=0.15)
            ax.axhline(yt, color="red", lw=0.8)
        ax.set_title(f"shank {sh}"); ax.set_xticks([])
    axes[0][0].set_ylabel("depth along shank (um)")
    fig.suptitle(f"{subject}: region by depth per shank")
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=c, label=k)
               for k, c in COARSE_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return str(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--anatomy-dir", default="data/anatomy")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    atlas_csv = os.path.join(args.anatomy_dir, f"{args.subject}_channel_atlas.csv")
    out = args.out or os.path.join("figures", "anatomy", f"{args.subject}_shank_anatomy.png")
    plot_subject_anatomy(args.subject, atlas_csv, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/anatomy/test_plot_shank_anatomy.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/anatomy/plot_shank_anatomy.py tests/anatomy/test_plot_shank_anatomy.py
git commit -m "feat(anatomy): per-subject shank region-by-depth QC figure"
```

---

### Task 11: Tracing adapter + brainreg wrapper + SOP doc

The thin bridge from the human BrainGlobe/Pinpoint output to the track artifact, the unattended brainreg wrapper, and the operator SOP.

**Files:**
- Create: `scripts/anatomy/import_track.py`
- Create: `scripts/anatomy/run_brainreg.py`
- Create: `docs/anatomy/registration_recipe.md`
- Test: `tests/anatomy/test_import_track.py`

**Tracing-export contract (under our control, documented in the SOP):** the operator exports, per subject, two files:
- `<subject>_track_points.csv` — columns `probe_shank_index,point_order,ap_um,ml_um,dv_um` (deepest point = smallest `point_order`). *Or* leave `probe_shank_index` blank and let `import_track.py` assign it from orientation.
- `<subject>_track_meta.json` — `{hemisphere, barcode_orientation, atlas, source_tool, created, shanks:{<idx or order>:{tip_y_um, method, sigma_along_um, sigma_across_um, sigma_growth_k, planned_entry, planned_vector}}}`.

**Interfaces:**
- Produces: `import_track(points_csv, meta_json, out_json)->TrackArtifact` — builds shanks, runs `assign_probe_shank_indices` when indices are missing, validates (`validate_track_artifact` + `validate_shank_order`), writes `<subject>_shank_tracks.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/anatomy/test_import_track.py
import json
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "anatomy"))
from import_track import import_track
from visdetect.anatomy.tracks import load_track_artifact

def test_import_track_builds_valid_artifact(tmp_path):
    pts = []
    for i in range(4):
        ml = 1350. + 250. * i
        pts += [{"probe_shank_index": i, "point_order": 0, "ap_um": 5000., "ml_um": ml, "dv_um": 3500.},
                {"probe_shank_index": i, "point_order": 1, "ap_um": 5000., "ml_um": ml, "dv_um": 2500.}]
    pcsv = tmp_path / "BG_046_track_points.csv"; pd.DataFrame(pts).to_csv(pcsv, index=False)
    meta = {"subject": "BG_046", "hemisphere": "right", "barcode_orientation": "forward",
            "atlas": "allen_mouse_25um", "source_tool": "brainglobe-segmentation",
            "created": "2026-06-17",
            "shanks": {str(i): {"tip_y_um": 0.0, "method": "brainreg_traced",
                                "sigma_along_um": 25., "sigma_across_um": 25.,
                                "sigma_growth_k": 0.0,
                                "planned_entry": None, "planned_vector": [0, 0, -1]}
                       for i in range(4)}}
    mjson = tmp_path / "BG_046_track_meta.json"; mjson.write_text(json.dumps(meta))
    out = tmp_path / "BG_046_shank_tracks.json"
    art = import_track(pcsv, mjson, out)
    assert out.exists()
    loaded = load_track_artifact(out)   # re-validates
    assert len(loaded.shanks) == 4
    assert loaded.subject == "BG_046"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/anatomy/test_import_track.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement `import_track.py`, `run_brainreg.py`, and the SOP**

```python
# scripts/anatomy/import_track.py
"""Adapt a tracing export (brainglobe-segmentation / Pinpoint, re-exported to our
CSV+JSON contract) into a validated track artifact. See docs/anatomy/registration_recipe.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from visdetect.anatomy.tracks import (
    ShankTrack, TrackArtifact, save_track_artifact, validate_track_artifact,
)
from visdetect.anatomy.orientation import assign_probe_shank_indices, validate_shank_order


def _vec(x):
    return None if x is None else np.asarray(x, float)


def import_track(points_csv, meta_json, out_json) -> TrackArtifact:
    pts = pd.read_csv(points_csv)
    meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    have_idx = pts["probe_shank_index"].notna().all() if "probe_shank_index" in pts else False

    # group points into shank polylines (by index if present, else by an order key)
    group_col = "probe_shank_index" if have_idx else "shank_group"
    if group_col not in pts:
        raise ValueError("points need either probe_shank_index or shank_group")

    shanks = []
    for g, d in pts.sort_values([group_col, "point_order"]).groupby(group_col):
        poly = d[["ap_um", "ml_um", "dv_um"]].to_numpy(float)
        m = meta["shanks"][str(int(g))]
        shanks.append(ShankTrack(
            probe_shank_index=int(g) if have_idx else -1,
            ccf_polyline=poly, tip_y_um=float(m["tip_y_um"]), method=m["method"],
            sigma_along_um=float(m["sigma_along_um"]),
            sigma_across_um=float(m["sigma_across_um"]),
            sigma_growth_k=float(m["sigma_growth_k"]),
            planned_entry=_vec(m.get("planned_entry")),
            planned_vector=_vec(m.get("planned_vector")),
        ))

    if not have_idx:
        shanks = assign_probe_shank_indices(
            shanks, meta["barcode_orientation"], meta["hemisphere"])

    art = TrackArtifact(
        subject=meta["subject"], atlas=meta["atlas"], hemisphere=meta["hemisphere"],
        barcode_orientation=meta["barcode_orientation"], source_tool=meta["source_tool"],
        created=meta["created"], shanks=sorted(shanks, key=lambda s: s.probe_shank_index),
    )
    validate_track_artifact(art)
    validate_shank_order(art)
    save_track_artifact(art, out_json)
    return art


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import_track(a.points, a.meta, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
```

```python
# scripts/anatomy/run_brainreg.py
"""Thin unattended wrapper around the brainreg CLI (headless; cluster-friendly).

Usage:
    py scripts/anatomy/run_brainreg.py --image <vol.tif> --out <dir> \
        --voxel 5 5 5 --orientation asr --atlas allen_mouse_25um
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def build_command(image, out, voxel, orientation, atlas):
    return ["brainreg", image, out,
            "-v", str(voxel[0]), str(voxel[1]), str(voxel[2]),
            "--orientation", orientation, "--atlas", atlas]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel", nargs=3, type=float, required=True)
    ap.add_argument("--orientation", required=True, help="e.g. 'asr' (BrainGlobe convention)")
    ap.add_argument("--atlas", default="allen_mouse_25um")
    a = ap.parse_args()
    cmd = build_command(a.image, a.out, a.voxel, a.orientation, a.atlas)
    print("running:", " ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
```

Create `docs/anatomy/registration_recipe.md` documenting the operator workflow:
1. Reconstruct the serial stack → 3D volume (TIFF), note voxel size + BrainGlobe orientation code.
2. `py scripts/anatomy/run_brainreg.py ...` (locally or on SLURM `cpu` partition via an sbatch wrapper modelled on `slurm/run_unitmatch_subject.sbatch`).
3. In napari + brainglobe-segmentation, trace each shank's dye track in the registered space; for partial tracks trace the visible segment and record `method="extended_from_tip"` + `planned_vector`; for no-dye shanks use Pinpoint and record `method="pinpoint_planned"`.
4. Export per-shank CCF points to `<subject>_track_points.csv` and fill `<subject>_track_meta.json` (incl. documented `barcode_orientation` + `hemisphere`).
5. `py scripts/anatomy/import_track.py --points ... --meta ... --out data/anatomy/<subject>_shank_tracks.json`.
6. `py scripts/anatomy/build_channel_atlas.py --subject <S>` → `py scripts/anatomy/localize_units.py --subject <S>` → `py scripts/anatomy/plot_shank_anatomy.py --subject <S>`; eyeball the QC figure (BG_038 must read GPe; BG_046 must show cortex→WM→striatum with larger σ on extended channels).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/anatomy/test_import_track.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/anatomy/import_track.py scripts/anatomy/run_brainreg.py docs/anatomy/registration_recipe.md tests/anatomy/test_import_track.py
git commit -m "feat(anatomy): tracing-export adapter + brainreg wrapper + registration SOP"
```

---

### Task 12: Full-suite smoke + anatomy test sweep

Confirm the new subpackage is coherent and nothing regressed.

**Files:** none (verification only).

- [ ] **Step 1: Run the anatomy test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/anatomy tests/suite/test_unit_table_anatomy.py -v`
Expected: PASS (all anatomy + integration tests).

- [ ] **Step 2: Run the existing unit-table / contract tests**

Run: `.venv/Scripts/python.exe -m pytest tests/ -k "unit_table or schema or tracking_qc" -v`
Expected: PASS (no contract regressions).

- [ ] **Step 3: Import smoke**

Run: `.venv/Scripts/python.exe -c "import visdetect.anatomy.tracks, visdetect.anatomy.atlas, visdetect.anatomy.localize, visdetect.anatomy.channel_geometry, visdetect.anatomy.orientation, visdetect.anatomy.peak_channel; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit (if any lint/import fixups were needed)**

```bash
git add -A
git commit -m "test(anatomy): full anatomy + contract sweep green" || echo "nothing to commit"
```

---

## Validation against real data (post-implementation, needs the human-supplied artifact)

Once a real `data/anatomy/BG_046_shank_tracks.json` exists (from the SOP):
- `py scripts/anatomy/build_channel_atlas.py --subject BG_046` then `localize_units.py` then `plot_shank_anatomy.py`.
- **BG_046:** QC figure shows cortex → white-matter → striatum descent; extended upper channels carry larger σ.
- **BG_038:** localizes to **GPe** (positive control).
- `build_unit_table()` now exposes `region_coarse` / `ccf_*` populated for localized sessions, `"unknown"`/NaN elsewhere.

## Phase B (deferred, not in this plan)

Promote `peak_channel` + `ccf_*` + `region_*` onto the `Cluster` dataclass and bake into PKLs on the next re-ingest. Additive: the sidecar `unit_anatomy.csv` remains the producer; ingest reads it.
