# M1 — Tracking: UM 3.2.9 Registry Unification + track_verdict Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the unit-label-table spine carry real, consistent tracking labels — `Global_UID` and `track_verdict` in the **same UM 3.2.9 ID space** — so the Learning headline can group by trusted, stably-tracked units. This flips P0's skipped real-data test to passing.

**Architecture:** The tracking *science* is already done (`verdicts.csv` / `verdicts_trimmed.csv` from the executed QC-sheets pipeline, off UM 3.2.9 `all42/unit_index.csv`). M1 wires it into the spine in three pieces: (1) a **registry adapter** that pivots the canonical long registry `(session, ks_unit_id, global_uid)` into the **wide CellRegistry** the GLT producer consumes (registry-agnostic; UM now, DeepUM later via a flag); (2) a **collision-resolution** step (one cluster claimed by two UIDs = the bimodal-ISI failure that P0's dedupe guard catches); (3) a **track_verdict resolver** that assigns each `(Global_UID, Session_Date)` row the *trimmed* verdict, "trusted" only within that UID's stable kept-sessions. Then the GLT is regenerated from UM and `build_unit_table` merges the verdict.

**Tech Stack:** Python 3.10, pandas, numpy, pytest. `py` on Windows. Library code under `src/visdetect/analysis/`; tests under `tests/analysis/` and `tests/suite/`; orchestration under `scripts/pipelines/tracking/`.

**Branch:** a dedicated M1 worktree off `docs/presentation-prep-roadmap` via `superpowers:using-git-worktrees`; set `PYTHONPATH=<worktree>/src`. M1 is independent of M2 (different files) and can run while other chats run.

**Context the engineer must know (read first):**
- **Canonical long registry (UM 3.2.9):** `X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/unit_match/output/all42/unit_index.csv`, columns `session, ks_unit_id, global_uid` (session is DDMMYYYY, sometimes 7-digit when day < 10, e.g. `1072025`).
- **Wide CellRegistry (what the GLT producer wants):** `pd.read_csv(path, index_col=0)` → index = `global_uid`, columns = session-date strings, cells = `ks_unit_id` (float; `"a;b"` allowed for oversplit). Producer: `scripts/analysis/build_longitudinal_table.py::build_grand_table`.
  - **LATENT BUG to design around:** it only accepts a column as a session if `c.isdigit() and len(c) == 8`. A 7-digit `1072025` is silently dropped. **The adapter MUST zero-pad session columns to 8 digits (`zfill(8)`).**
  - It splits `"123;456"` cells into separate `Cluster_ID` rows under one `Global_UID`; rows are `{Global_UID, Session_Date=column, Cluster_ID=int(float(cell))}`.
- **Verdict files (already produced, UM 3.2.9-based):** `FIGURES/tracking_qc/verdicts.csv` (per `global_uid`: `verdict` ∈ {trusted, review, suspect}) and `FIGURES/tracking_qc/verdicts_trimmed.csv` (per `global_uid`: `kept_sessions` semicolon-joined, `trimmed_verdict`, `dropped_sessions`).
- **Trust rule (decided 2026-06-03):** row-level `track_verdict` = the UID's `trimmed_verdict` **iff** the row's session ∈ that UID's `kept_sessions`; otherwise `suspect`; UIDs absent from the trimmed cohort → `unknown`.
- **Session normalization gotcha:** GLT `Session_Date` and verdicts `kept_sessions` use mixed 7/8-digit forms. **Always compare sessions as `int`** (so `1072025 == int("01072025")`).
- **P0 contract:** `src/visdetect/suite/unit_table_schema.py` — `track_verdict` ∈ {trusted, review, suspect, unknown}; `build_unit_table(validate=True)` enforces it; keys `(Session_Date, Cluster_ID)` must be unique integers.
- **Spec:** `docs/superpowers/specs/2026-06-03-presentation-prep-roadmap-design.md` (§3 contract, §9 groundwork).

---

## File structure

| Path | Action | Responsibility |
|---|---|---|
| `src/visdetect/analysis/tracking_registry.py` | Create | Load canonical long registry; detect & resolve cluster collisions; pivot long→wide CellRegistry (zfill-8 columns, `;`-joined oversplits). |
| `src/visdetect/analysis/track_verdict.py` | Create | Load trimmed verdicts/kept-sessions; `resolve_row_verdict()` (trimmed+kept rule, int-session compare). |
| `tests/analysis/test_tracking_registry.py` | Create | Synthetic tests: zfill, collision detect/resolve, oversplit join, pivot round-trip. |
| `tests/analysis/test_track_verdict.py` | Create | Synthetic tests: kept→trimmed verdict, dropped→suspect, non-cohort→unknown, int-session match. |
| `src/visdetect/suite/loader.py` | Modify | `build_unit_table`: replace the `track_verdict` default with a real merge via `track_verdict.resolve_row_verdict`; add `verdicts_path` override. |
| `tests/suite/test_unit_table_build.py` | Modify (append) | track_verdict merge test (synthetic GLT + trimmed verdicts). |
| `scripts/pipelines/tracking/regen_glt_from_um.py` | Create | CLI: canonical long → resolved wide CellRegistry → run `build_longitudinal_table.py` → GLT; then rebuild + validate unit table. |
| `docs/superpowers/specs/2026-06-03-presentation-prep-roadmap-design.md` | Modify | Record M1 outcome + the GLT-producer zfill bug. |

---

## Task 1: Canonical long registry loader + collision detection

**Files:**
- Create: `src/visdetect/analysis/tracking_registry.py`
- Test: `tests/analysis/test_tracking_registry.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/analysis/test_tracking_registry.py`:

```python
"""Tests for the canonical tracking registry adapter (M1)."""
import pandas as pd
import pytest

from visdetect.analysis.tracking_registry import (
    load_canonical_long, find_cluster_collisions,
)


def _write_long(tmp_path, rows, cols=("session", "ks_unit_id", "global_uid")):
    csv = tmp_path / "reg.csv"
    pd.DataFrame(rows, columns=list(cols)).to_csv(csv, index=False)
    return csv


def test_load_zero_pads_session_to_8(tmp_path):
    csv = _write_long(tmp_path, [["1072025", 3, 0], ["23062025", 4, 1]])
    df = load_canonical_long(csv)
    assert list(df["session"]) == ["01072025", "23062025"]
    assert df["ks_unit_id"].dtype.kind in ("i", "u")
    assert df["global_uid"].dtype.kind in ("i", "u")


def test_load_accepts_ks_id_alias(tmp_path):
    csv = _write_long(tmp_path, [["1072025", 3, 0]], cols=("session", "ks_id", "global_uid"))
    df = load_canonical_long(csv)
    assert "ks_unit_id" in df.columns
    assert int(df.loc[0, "ks_unit_id"]) == 3


def test_load_missing_columns_raises(tmp_path):
    csv = _write_long(tmp_path, [["1072025", 3]], cols=("session", "ks_unit_id"))
    with pytest.raises(ValueError, match="global_uid"):
        load_canonical_long(csv)


def test_find_collisions_flags_cluster_claimed_by_two_uids(tmp_path):
    # (01072025, ks 3) appears under uid 0 AND uid 9 -> collision (bimodal-ISI failure)
    csv = _write_long(tmp_path, [
        ["1072025", 3, 0], ["1072025", 3, 9], ["1072025", 4, 1],
    ])
    df = load_canonical_long(csv)
    coll = find_cluster_collisions(df)
    assert set(coll["global_uid"]) == {0, 9}
    assert (coll["session"] == "01072025").all()
    assert (coll["ks_unit_id"] == 3).all()


def test_find_collisions_empty_when_clean(tmp_path):
    csv = _write_long(tmp_path, [["1072025", 3, 0], ["1072025", 4, 1]])
    df = load_canonical_long(csv)
    assert find_cluster_collisions(df).empty
```

- [ ] **Step 2: Run to confirm failure**

Run: `py -m pytest tests/analysis/test_tracking_registry.py -v`
Expected: `ModuleNotFoundError: No module named 'visdetect.analysis.tracking_registry'`.

- [ ] **Step 3: Create the module**

Create `src/visdetect/analysis/tracking_registry.py`:

```python
"""Canonical tracking-registry adapters (M1).

Bridges the canonical LONG registry (session, ks_unit_id, global_uid) and the
WIDE CellRegistry (UID-indexed; session-date columns; ks_unit_id cells) that
``scripts/analysis/build_longitudinal_table.py`` consumes. Registry-agnostic:
any method that emits the canonical long form (UM 3.2.9 now, DeepUM later) can
drive the same pipeline, so Global_UID and track_verdict share one ID space.

See docs/superpowers/specs/2026-06-03-presentation-prep-roadmap-design.md (§9).
"""
from __future__ import annotations

import pandas as pd

CANONICAL_COLS = ["session", "ks_unit_id", "global_uid"]


def load_canonical_long(path) -> pd.DataFrame:
    """Load a canonical long registry.

    Columns (after normalization): session (str, 8-digit DDMMYYYY), ks_unit_id
    (int), global_uid (int). Accepts ``ks_id`` as an alias for ``ks_unit_id``.
    """
    df = pd.read_csv(path, dtype={"session": str})
    if "ks_unit_id" not in df.columns and "ks_id" in df.columns:
        df = df.rename(columns={"ks_id": "ks_unit_id"})
    missing = [c for c in CANONICAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"registry missing columns {missing}; has {list(df.columns)}"
        )
    df = df[CANONICAL_COLS].copy()
    # Zero-pad to 8-digit DDMMYYYY so wide columns survive build_grand_table's
    # `len(c) == 8` session filter (it silently drops 7-digit single-digit-day dates).
    df["session"] = df["session"].astype(str).str.strip().str.zfill(8)
    df["ks_unit_id"] = df["ks_unit_id"].astype(int)
    df["global_uid"] = df["global_uid"].astype(int)
    return df


def find_cluster_collisions(long_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where one (session, ks_unit_id) is claimed by >1 global_uid.

    This is the bimodal-ISI matching failure (two different units fused under
    distinct tracked IDs) that P0's dedupe guard catches downstream. Empty frame
    if the registry is clean.
    """
    n_uid = long_df.groupby(["session", "ks_unit_id"])["global_uid"].transform("nunique")
    return long_df[n_uid > 1].copy()
```

- [ ] **Step 4: Run tests, expect pass**

Run: `py -m pytest tests/analysis/test_tracking_registry.py -v`
Expected: 5 passed. (Create empty `tests/analysis/__init__.py` if a collection error appears.)

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/tracking_registry.py tests/analysis/test_tracking_registry.py
git commit -m "M1: canonical long-registry loader + cluster-collision detector"
```

---

## Task 2: Collision resolution + long→wide CellRegistry pivot

**Files:**
- Modify: `src/visdetect/analysis/tracking_registry.py`
- Modify: `tests/analysis/test_tracking_registry.py`

Resolve collisions using the stable kept-set (a contested cluster goes to the UID
whose stable subset keeps that session; drop if none or still ambiguous), then
pivot to the wide CellRegistry (oversplit = one UID with multiple clusters in a
session → `;`-joined cell).

- [ ] **Step 1: Append failing tests**

Append to `tests/analysis/test_tracking_registry.py`:

```python
from visdetect.analysis.tracking_registry import (
    resolve_collisions, long_to_cellregistry,
)


def test_resolve_collisions_keeps_supported_uid(tmp_path):
    csv = _write_long(tmp_path, [
        ["1072025", 3, 0],   # uid 0 keeps this session (supported)
        ["1072025", 3, 9],   # uid 9 does NOT keep this session
        ["2072025", 4, 1],   # uncontested
    ])
    df = load_canonical_long(csv)
    kept = {0: {1072025}, 9: {2072025}}           # int-session kept-sets
    out = resolve_collisions(df, kept)
    # Only uid 0 retains (01072025, 3); uid 9's contested row dropped.
    held = out[(out["session"] == "01072025") & (out["ks_unit_id"] == 3)]
    assert list(held["global_uid"]) == [0]
    assert len(out) == 2                           # uncontested row survives


def test_resolve_collisions_drops_when_ambiguous(tmp_path):
    csv = _write_long(tmp_path, [["1072025", 3, 0], ["1072025", 3, 9]])
    df = load_canonical_long(csv)
    kept = {0: {1072025}, 9: {1072025}}            # BOTH keep it -> ambiguous
    out = resolve_collisions(df, kept)
    assert out[(out["session"] == "01072025") & (out["ks_unit_id"] == 3)].empty


def test_long_to_cellregistry_pivots_and_zero_pads(tmp_path):
    csv = _write_long(tmp_path, [
        ["1072025", 3, 0], ["2072025", 5, 0], ["1072025", 4, 1],
    ])
    df = load_canonical_long(csv)
    reg = long_to_cellregistry(df)
    # index = global_uid; columns = 8-digit sessions; cells = ks_unit_id
    assert list(reg.index) == [0, 1]
    assert "01072025" in reg.columns and "02072025" in reg.columns
    assert str(reg.loc[0, "01072025"]) == "3"
    assert pd.isna(reg.loc[1, "02072025"])


def test_long_to_cellregistry_joins_oversplit_with_semicolon(tmp_path):
    # uid 0 has TWO clusters (3 and 7) in the same session -> "3;7"
    csv = _write_long(tmp_path, [["1072025", 3, 0], ["1072025", 7, 0]])
    df = load_canonical_long(csv)
    reg = long_to_cellregistry(df)
    assert set(str(reg.loc[0, "01072025"]).split(";")) == {"3", "7"}
```

- [ ] **Step 2: Run to confirm failure**

Run: `py -m pytest tests/analysis/test_tracking_registry.py -v`
Expected: the 4 new tests fail (`resolve_collisions` / `long_to_cellregistry` undefined).

- [ ] **Step 3: Implement both functions**

Append to `src/visdetect/analysis/tracking_registry.py`:

```python
from typing import Dict, Set


def resolve_collisions(
    long_df: pd.DataFrame,
    kept_sessions_by_uid: Dict[int, Set[int]],
) -> pd.DataFrame:
    """Resolve (session, ks_unit_id)-claimed-by->1-UID collisions.

    Policy (trust rule, 2026-06-03): for a contested cluster, keep the UID whose
    stable kept-subset includes that session; if exactly one UID qualifies, keep
    it and drop the others; if zero or more than one qualify, drop ALL claims on
    that cluster (ambiguous → excluded from the registry). Uncontested rows pass
    through untouched.

    Parameters
    ----------
    long_df : canonical long registry (from load_canonical_long).
    kept_sessions_by_uid : {global_uid -> set of kept sessions as int}.
    """
    collisions = find_cluster_collisions(long_df)
    if collisions.empty:
        return long_df.copy()

    contested_keys = set(map(tuple, collisions[["session", "ks_unit_id"]].values))
    keep_rows = []
    for idx, row in long_df.iterrows():
        key = (row["session"], row["ks_unit_id"])
        if key not in contested_keys:
            keep_rows.append(idx)
            continue
        # Among UIDs claiming this cluster, which keep this session in their subset?
        sess_int = int(row["session"])
        claimants = long_df[(long_df["session"] == row["session"]) &
                            (long_df["ks_unit_id"] == row["ks_unit_id"])]["global_uid"]
        supported = [u for u in claimants
                     if sess_int in kept_sessions_by_uid.get(int(u), set())]
        if len(supported) == 1 and int(row["global_uid"]) == supported[0]:
            keep_rows.append(idx)
        # else: drop (ambiguous or unsupported)
    return long_df.loc[keep_rows].copy()


def long_to_cellregistry(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot canonical long → wide CellRegistry consumed by build_grand_table.

    index = global_uid; columns = 8-digit session strings; cells = ks_unit_id
    (``;``-joined string when a UID has multiple clusters in one session).
    """
    def _join(s):
        vals = sorted(int(v) for v in s)
        return ";".join(str(v) for v in vals) if len(vals) > 1 else str(vals[0])

    wide = (long_df
            .groupby(["global_uid", "session"])["ks_unit_id"]
            .apply(_join)
            .unstack("session"))
    wide.index.name = "UID"
    return wide
```

- [ ] **Step 4: Run tests, expect pass**

Run: `py -m pytest tests/analysis/test_tracking_registry.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/tracking_registry.py tests/analysis/test_tracking_registry.py
git commit -m "M1: collision resolution + long->wide CellRegistry pivot"
```

---

## Task 3: track_verdict resolver

**Files:**
- Create: `src/visdetect/analysis/track_verdict.py`
- Test: `tests/analysis/test_track_verdict.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/analysis/test_track_verdict.py`:

```python
"""Tests for row-level track_verdict resolution (M1)."""
import pandas as pd
import pytest

from visdetect.analysis.track_verdict import (
    load_kept_map, load_trimmed_verdicts, resolve_row_verdict,
)


def _write_trimmed(tmp_path):
    csv = tmp_path / "verdicts_trimmed.csv"
    pd.DataFrame({
        "global_uid": [177, 262],
        "kept_sessions": ["2092025;16092025;17092025", "25072025"],
        "trimmed_verdict": ["trusted", "suspect"],
    }).to_csv(csv, index=False)
    return csv


def test_load_kept_map_int_sessions(tmp_path):
    kept = load_kept_map(_write_trimmed(tmp_path))
    assert kept[177] == {2092025, 16092025, 17092025}
    assert kept[262] == {25072025}


def test_kept_session_gets_trimmed_verdict(tmp_path):
    csv = _write_trimmed(tmp_path)
    kept, verds = load_kept_map(csv), load_trimmed_verdicts(csv)
    # session present in 8-digit form must still match (int compare)
    assert resolve_row_verdict(177, "02092025", kept, verds) == "trusted"
    assert resolve_row_verdict(177, 16092025, kept, verds) == "trusted"


def test_dropped_session_is_suspect(tmp_path):
    csv = _write_trimmed(tmp_path)
    kept, verds = load_kept_map(csv), load_trimmed_verdicts(csv)
    # 177 is trimmed-trusted but this session was dropped from its stable subset
    assert resolve_row_verdict(177, 1072025, kept, verds) == "suspect"


def test_non_cohort_uid_is_unknown(tmp_path):
    csv = _write_trimmed(tmp_path)
    kept, verds = load_kept_map(csv), load_trimmed_verdicts(csv)
    assert resolve_row_verdict(9999, 1072025, kept, verds) == "unknown"


def test_output_in_contract_vocabulary(tmp_path):
    from visdetect.suite.unit_table_schema import ALLOWED_VALUES
    csv = _write_trimmed(tmp_path)
    kept, verds = load_kept_map(csv), load_trimmed_verdicts(csv)
    allowed = ALLOWED_VALUES["track_verdict"]
    for uid, sess in [(177, 2092025), (177, 1072025), (262, 25072025), (1, 1072025)]:
        assert resolve_row_verdict(uid, sess, kept, verds) in allowed
```

- [ ] **Step 2: Run to confirm failure**

Run: `py -m pytest tests/analysis/test_track_verdict.py -v`
Expected: `ModuleNotFoundError: No module named 'visdetect.analysis.track_verdict'`.

- [ ] **Step 3: Create the module**

Create `src/visdetect/analysis/track_verdict.py`:

```python
"""Row-level track_verdict resolution from the trimmed-verdict cohort (M1).

Trust rule (2026-06-03): a (Global_UID, Session) row is the UID's trimmed
verdict iff the session is in that UID's stable kept-subset; otherwise suspect;
UIDs absent from the trimmed cohort are unknown. Sessions are compared as int
to bridge 7/8-digit DDMMYYYY forms.
"""
from __future__ import annotations

from typing import Dict, Set

import pandas as pd


def _to_int_session(s) -> int:
    return int(str(s).strip())


def load_kept_map(trimmed_path) -> Dict[int, Set[int]]:
    """{global_uid -> set of kept sessions as int} from verdicts_trimmed.csv."""
    df = pd.read_csv(trimmed_path)
    out: Dict[int, Set[int]] = {}
    for _, row in df.iterrows():
        raw = row.get("kept_sessions")
        sessions: Set[int] = set()
        if isinstance(raw, str) and raw.strip():
            sessions = {_to_int_session(s) for s in raw.split(";") if s.strip()}
        out[int(row["global_uid"])] = sessions
    return out


def load_trimmed_verdicts(trimmed_path) -> Dict[int, str]:
    """{global_uid -> trimmed_verdict} from verdicts_trimmed.csv."""
    df = pd.read_csv(trimmed_path)
    return {int(r["global_uid"]): str(r["trimmed_verdict"]) for _, r in df.iterrows()}


def resolve_row_verdict(
    global_uid,
    session,
    kept_map: Dict[int, Set[int]],
    trimmed_verdict_map: Dict[int, str],
) -> str:
    """Return track_verdict for one (global_uid, session) row."""
    uid = int(global_uid)
    if uid not in trimmed_verdict_map:
        return "unknown"
    if _to_int_session(session) in kept_map.get(uid, set()):
        return trimmed_verdict_map[uid]
    return "suspect"
```

- [ ] **Step 4: Run tests, expect pass**

Run: `py -m pytest tests/analysis/test_track_verdict.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/track_verdict.py tests/analysis/test_track_verdict.py
git commit -m "M1: row-level track_verdict resolver (trimmed + kept-sessions rule)"
```

---

## Task 4: Wire track_verdict into build_unit_table

**Files:**
- Modify: `src/visdetect/suite/loader.py` (`build_unit_table`)
- Modify: `tests/suite/test_unit_table_build.py` (append)

Replace P0's `track_verdict` default ("unknown") with a real per-row resolution
merged onto the GLT by `Global_UID` + `Session_Date`.

- [ ] **Step 1: Append failing test**

Append to `tests/suite/test_unit_table_build.py`:

```python
def test_build_unit_table_fills_track_verdict(tmp_path, monkeypatch):
    from visdetect.suite import loader as L

    glt = pd.DataFrame({
        "Session_Date": [2092025, 1072025, 1072025],
        "Cluster_ID": [3, 4, 5],
        "Global_UID": [177, 177, 9999],
        "stage": ["Expert", "Learning", "Learning"],
        "session_idx": [20, 0, 0],
    })
    trimmed = tmp_path / "verdicts_trimmed.csv"
    pd.DataFrame({
        "global_uid": [177],
        "kept_sessions": ["2092025"],
        "trimmed_verdict": ["trusted"],
    }).to_csv(trimmed, index=False)

    monkeypatch.setattr(L, "load_glt", lambda qc_only=True: glt.copy())
    monkeypatch.setattr(L, "load_all_lick_responsiveness", lambda: pd.DataFrame())
    monkeypatch.setattr(L, "load_waveform_labels",
                        lambda path=None: (_ for _ in ()).throw(FileNotFoundError("none")))
    monkeypatch.setattr(L, "load_tf_responsiveness_detrended", lambda: pd.DataFrame())
    monkeypatch.setattr(L, "load_tf_classification_detrended", lambda: pd.DataFrame())

    df = L.build_unit_table(qc_only=True, verdicts_path=str(trimmed))
    by_key = df.set_index(["Session_Date", "Cluster_ID"])["track_verdict"]
    assert by_key[(2092025, 3)] == "trusted"     # 177 kept this session
    assert by_key[(1072025, 4)] == "suspect"     # 177, session dropped from subset
    assert by_key[(1072025, 5)] == "unknown"     # 9999 not in trimmed cohort
```

- [ ] **Step 2: Run to confirm failure**

Run: `py -m pytest tests/suite/test_unit_table_build.py::test_build_unit_table_fills_track_verdict -v`
Expected: FAIL — `build_unit_table()` has no `verdicts_path` kwarg / `track_verdict` is all "unknown".

- [ ] **Step 3: Modify `build_unit_table`**

In `src/visdetect/suite/loader.py`, change the `def` line to add `verdicts_path`:

```python
def build_unit_table(qc_only: bool = True, validate: bool = True,
                     verdicts_path: Optional[str] = None) -> pd.DataFrame:
```

Then, in the final-assembly block added by P0, insert the track_verdict resolution
**before** the `add_label_defaults(glt)` call (so the resolved values are present and
`add_label_defaults` leaves them untouched):

```python
    # ── Resolve track_verdict per (Global_UID, Session_Date) (M1) ──
    import os as _os
    from visdetect.suite.config import FIGURE_DIR
    vpath = verdicts_path or _os.path.join(
        FIGURE_DIR, "tracking_qc", "verdicts_trimmed.csv")
    if "Global_UID" in glt.columns and _os.path.exists(vpath):
        from visdetect.analysis.track_verdict import (
            load_kept_map, load_trimmed_verdicts, resolve_row_verdict,
        )
        kept_map = load_kept_map(vpath)
        verd_map = load_trimmed_verdicts(vpath)
        glt["track_verdict"] = [
            resolve_row_verdict(u, s, kept_map, verd_map)
            for u, s in zip(glt["Global_UID"], glt["Session_Date"])
        ]
```

(`FIGURE_DIR` lives in `visdetect.suite.config` — confirmed; `src/visdetect/suite/config.py`.)

- [ ] **Step 4: Run tests, expect pass**

Run: `py -m pytest tests/suite/test_unit_table_build.py -v`
Expected: all passed (P0 tests + the new track_verdict test).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/suite/loader.py tests/suite/test_unit_table_build.py
git commit -m "M1: build_unit_table fills track_verdict from trimmed-verdict cohort"
```

---

## Task 5: GLT-regeneration orchestrator (UM 3.2.9 → wide → GLT → unit table)

**Files:**
- Create: `scripts/pipelines/tracking/regen_glt_from_um.py`

This wires Tasks 1-4 into a runnable pipeline. It does NOT recompute physiology
itself — it builds the resolved wide CellRegistry, then shells out to the existing
`build_longitudinal_table.py`, then rebuilds + validates the unit table.

- [ ] **Step 1: Create the orchestrator**

Create `scripts/pipelines/tracking/regen_glt_from_um.py`:

```python
"""Regenerate the GLT from a canonical long registry, then rebuild+validate the unit table.

Registry-agnostic: --registry-long defaults to UM 3.2.9 all42/unit_index.csv;
swap to a DeepUM-derived long registry later via the same flag.

Usage:
    py scripts/pipelines/tracking/regen_glt_from_um.py --workers 6
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from visdetect.analysis.tracking_registry import (  # noqa: E402
    load_canonical_long, find_cluster_collisions, resolve_collisions, long_to_cellregistry,
)
from visdetect.analysis.track_verdict import load_kept_map  # noqa: E402

DEFAULT_LONG = ("X:/public/projects/BeJG_20230130_VisDetect/wEPhys/"
                "BG_046/unit_match/output/all42/unit_index.csv")
TRIMMED = REPO_ROOT / "FIGURES" / "tracking_qc" / "verdicts_trimmed.csv"
WIDE_OUT = REPO_ROOT / "data" / "unit_match" / "output" / "BG_046_um329_CellRegistry.csv"
GLT_OUT = REPO_ROOT / "table_output" / "Grand_Longitudinal_Table.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry-long", default=DEFAULT_LONG)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    long_df = load_canonical_long(args.registry_long)
    print(f"loaded {len(long_df)} unit-session rows, {long_df['global_uid'].nunique()} UIDs")

    coll = find_cluster_collisions(long_df)
    print(f"cluster collisions (one cluster -> >1 UID): {len(coll)} rows")

    kept = load_kept_map(TRIMMED) if TRIMMED.exists() else {}
    resolved = resolve_collisions(long_df, kept)
    print(f"after collision resolution: {len(resolved)} rows")

    wide = long_to_cellregistry(resolved)
    WIDE_OUT.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(WIDE_OUT)
    print(f"wrote wide CellRegistry: {WIDE_OUT}  ({wide.shape[0]} UIDs x {wide.shape[1]} sessions)")

    # Regenerate the GLT via the existing producer.
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "analysis" / "build_longitudinal_table.py"),
           "--registry", str(WIDE_OUT), "--output", str(GLT_OUT), "--workers", str(args.workers)]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Rebuild + validate the unit table.
    from visdetect.suite.loader import build_unit_table
    df = build_unit_table(qc_only=True, validate=True)
    print(f"unit table rebuilt + validated: {len(df)} rows; "
          f"track_verdict counts:\n{df['track_verdict'].value_counts()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run the adapter portion only (fast, no GLT compute)**

Confirm the adapter reads the real UM registry and produces a sane wide CellRegistry
without the heavy GLT step:

```bash
py -c "import sys; sys.path.insert(0,'src'); from visdetect.analysis.tracking_registry import load_canonical_long, find_cluster_collisions, resolve_collisions, long_to_cellregistry; from visdetect.analysis.track_verdict import load_kept_map; L=load_canonical_long('X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/unit_match/output/all42/unit_index.csv'); print('rows',len(L),'uids',L['global_uid'].nunique()); print('collisions',len(find_cluster_collisions(L))); k=load_kept_map('FIGURES/tracking_qc/verdicts_trimmed.csv'); R=resolve_collisions(L,k); W=long_to_cellregistry(R); print('wide',W.shape); print('all 8-digit cols:', all(len(str(c))==8 for c in W.columns))"
```

Expected: prints row/UID counts, a collision count, the wide shape, and `all 8-digit cols: True`. If any column is not 8 digits, fix `load_canonical_long`'s `zfill` before proceeding.

- [ ] **Step 3: Commit**

```bash
git add scripts/pipelines/tracking/regen_glt_from_um.py
git commit -m "M1: GLT-regeneration orchestrator (long registry -> wide -> GLT -> unit table)"
```

---

## Task 6: Regenerate the GLT and flip P0's skip-if test (real-data run)

**Files:** none (a run + verification step).

This is the heavy step: `build_longitudinal_table.py` recomputes behavior + TF per
session from pkls. Run it once; it produces the real GLT and makes the unit table live.

- [ ] **Step 1: Run the full regeneration**

```bash
py scripts/pipelines/tracking/regen_glt_from_um.py --workers 6
```

Expected: writes `data/unit_match/output/BG_046_um329_CellRegistry.csv`, then
`table_output/Grand_Longitudinal_Table.csv`, then prints `track_verdict` value counts
(a mix of trusted / review / suspect / unknown). If `build_longitudinal_table.py`
needs UnitMatch-specific deps, run it under that env (`conda activate unitmatch_env`)
per its module docstring; the adapter + unit-table steps run in the project venv.

- [ ] **Step 2: Confirm P0's real-data contract test now passes (no longer skipped)**

```bash
py -m pytest tests/suite/test_unit_table_build.py::test_build_unit_table_real_data_contract -v
```

Expected: **PASSED** (was SKIPPED). If it fails on duplicate keys, the collision
resolution left a residual (one cluster still under two UIDs) — inspect with the
audit harness and tighten `resolve_collisions` (the failure is real tracking
contamination, not a code bug):

```bash
py scripts/QC_CHECKS/audit_unit_table.py --out FIGURES/qc/unit_table_audit_m1.txt
```

- [ ] **Step 3: Run the full relevant test set**

Run: `py -m pytest tests/suite/ tests/analysis/test_tracking_registry.py tests/analysis/test_track_verdict.py -v`
Expected: all pass, none skipped (the real-data test now runs).

- [ ] **Step 4: Commit the audit report (not the large GLT/registry data)**

```bash
git add FIGURES/qc/unit_table_audit_m1.txt
git commit -m "M1: regenerate GLT from UM 3.2.9; P0 real-data contract test now passes"
```

(Do NOT commit `table_output/Grand_Longitudinal_Table.csv` or the wide CellRegistry —
they are regenerable data artifacts. Confirm they are git-ignored or leave them untracked.)

---

## Task 7: Record outcome in the spec

**Files:**
- Modify: `docs/superpowers/specs/2026-06-03-presentation-prep-roadmap-design.md`

- [ ] **Step 1: Append the M1 outcome to §9**

Append under §9:

```markdown
> **M1 done (date):** GLT regenerated from UM 3.2.9 via `regen_glt_from_um.py`
> (`tracking_registry.long_to_cellregistry` + collision resolution). `Global_UID`
> and `track_verdict` now share the UM 3.2.9 ID space. `track_verdict` = trimmed
> verdict within kept-sessions, else suspect, else unknown. P0's real-data test
> passes. Known GLT-producer bug worked around: `build_grand_table` drops session
> columns whose name length != 8, so the adapter zero-pads to 8-digit DDMMYYYY.
> track_verdict counts: <paste from the run>.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-03-presentation-prep-roadmap-design.md
git commit -m "M1: record tracking-registry outcome in roadmap spec"
```

---

## Definition of done

- `tracking_registry.py` + `track_verdict.py` exist, fully unit-tested (synthetic).
- `build_unit_table` fills `track_verdict` per the trimmed+kept rule; contract validates.
- `regen_glt_from_um.py` rebuilds the GLT from UM 3.2.9 with collisions resolved and 8-digit session columns.
- P0's `test_build_unit_table_real_data_contract` is **PASSED** (no longer skipped).
- The spine now carries real, ID-space-consistent `Global_UID` + `track_verdict` for the Learning headline.

**Late-bound swap:** to evaluate fine-tuned DeepUM instead, derive a canonical long
registry (`session, ks_unit_id, global_uid`) from its output and pass it via
`--registry-long`; everything downstream is unchanged.

**Unblocks:** the Learning figure (group by `track_verdict == "trusted"` rows across `stage`),
and M2's `celltype` enrichment joins straight onto the same rows.
