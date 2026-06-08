# Track Curation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Curate the liberal UnitMatch cross-session registry into precision-trusted neuron tracks via an Expert→Naive backward sweep (biophysical gate + availability-gated in-zone functional corroborator), with a pluggable state interface and held-out-ISI validation.

**Architecture:** Two new library modules in `src/visdetect/analysis/` — `state_provider.py` (file-contract state labeler, pluggable) and `track_curation.py` (features, per-link scoring, backward sweep, tier logic, validation) — both reusing the existing `tracking_qc.py` primitives. Three thin CLI scripts in `scripts/pipelines/tracking/` wire I/O around the libraries. Pure functions are unit-tested with synthetic sessions / hand-written fixtures; scripts are glue with smoke coverage.

**Tech Stack:** Python, numpy, pandas, pytest. Reuses `visdetect.analysis.tracking_qc`, `visdetect.analysis.hmm`, `visdetect.core.session`, `visdetect.utils.synthetic`.

**Spec:** `docs/superpowers/specs/2026-06-07-track-curation-design.md`

**Conventions for every task:**
- Run tests with: `.venv\Scripts\python.exe -m pytest <path> -v` (the `py` launcher ignores the venv; use the explicit interpreter).
- New tests live under `tests/analysis/`.
- Commit messages end with a trailing `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` line.
- Work on the current `main` branch unless the user chose a feature branch at handoff.

---

## File Structure

| Path | Responsibility |
|------|----------------|
| `src/visdetect/analysis/state_provider.py` | Canonical state vocabulary; per-session state-table CSV I/O; `in_zone_trial_indices`; `UniformInZoneStateProvider` (bootstrap) + `HMMStateProvider`. |
| `src/visdetect/analysis/track_curation.py` | `CurationParams`, `CurationFeature`, `LinkResult`, `SweepResult`; `partitioned_isi_hists`; `extract_curation_feature`; `score_link`; `sweep_uid` + `compute_tier`; `curate_registry`; `held_out_isi_auc_by_tier`. |
| `src/visdetect/analysis/tracking_qc.py` | **Modify:** add optional `restrict_trials` to `extract_unit_psths`. |
| `scripts/pipelines/tracking/make_state_tables.py` | CLI: write per-session state tables (bootstrap uniform, or HMM provider). |
| `scripts/pipelines/tracking/curate_tracks.py` | CLI: liberal registry + state tables + waveforms/pkls → `curated_links.csv` + `curated_tracks.csv`. |
| `scripts/pipelines/tracking/validate_curation.py` | CLI: held-out-ISI AUC by confidence tier. |
| `tests/analysis/test_state_provider.py` | Tests for `state_provider.py`. |
| `tests/analysis/test_track_curation.py` | Tests for `track_curation.py` + the `extract_unit_psths` change. |

---

## Task 1: State vocabulary + state-table I/O + `in_zone_trial_indices`

**Files:**
- Create: `src/visdetect/analysis/state_provider.py`
- Test: `tests/analysis/test_state_provider.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_state_provider.py
import numpy as np
import pandas as pd
import pytest
from visdetect.analysis import state_provider as sp


def test_canonical_states_are_three():
    assert sp.CANONICAL_STATES == ("disengaged", "impulsive", "in_zone")


def test_write_then_load_state_table_roundtrip(tmp_path):
    rows = [(0, "in_zone", 0.91), (2, "impulsive", 0.7), (3, "disengaged", 0.99)]
    sp.write_state_table("07072025", rows, tmp_path)
    loaded = sp.load_state_table("07072025", tmp_path)
    assert loaded[0] == ("in_zone", pytest.approx(0.91))
    assert loaded[2] == ("impulsive", pytest.approx(0.7))
    assert set(loaded.keys()) == {0, 2, 3}


def test_write_state_table_rejects_unknown_label(tmp_path):
    with pytest.raises(ValueError):
        sp.write_state_table("07072025", [(0, "engaged", 1.0)], tmp_path)


def test_in_zone_trial_indices_filters_by_label(tmp_path):
    rows = [(0, "in_zone", 0.9), (1, "disengaged", 0.9),
            (2, "in_zone", 0.5), (5, "in_zone", 0.95)]
    sp.write_state_table("07072025", rows, tmp_path)
    idx = sp.in_zone_trial_indices("07072025", tmp_path)
    assert idx == [0, 2, 5]


def test_in_zone_trial_indices_confidence_floor(tmp_path):
    rows = [(0, "in_zone", 0.9), (2, "in_zone", 0.5), (5, "in_zone", 0.95)]
    sp.write_state_table("07072025", rows, tmp_path)
    idx = sp.in_zone_trial_indices("07072025", tmp_path, min_confidence=0.8)
    assert idx == [0, 5]


def test_in_zone_trial_indices_missing_table_returns_empty(tmp_path):
    assert sp.in_zone_trial_indices("99999999", tmp_path) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/analysis/test_state_provider.py -v`
Expected: FAIL (`ModuleNotFoundError: visdetect.analysis.state_provider`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/visdetect/analysis/state_provider.py
"""Pluggable behavioural-state interface for track curation.

The curation pipeline consumes a per-session trial->state table; it never
imports any state model. The HMM is one provider; a hand/ethogram labeler can
write the same CSV later. See
docs/superpowers/specs/2026-06-07-track-curation-design.md sec 4.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

# ── Canonical 3-state vocabulary ──────────────────────────────────────
DISENGAGED = "disengaged"
IMPULSIVE = "impulsive"
IN_ZONE = "in_zone"
CANONICAL_STATES: Tuple[str, str, str] = (DISENGAGED, IMPULSIVE, IN_ZONE)


def state_table_path(session_name: str, states_dir) -> Path:
    return Path(states_dir) / f"{str(session_name).zfill(8)}_states.csv"


def write_state_table(session_name: str,
                      rows: Sequence[Tuple[int, str, float]],
                      states_dir) -> Path:
    """Write a per-session state table. rows = (trial_idx, state_label, confidence).

    trial_idx MUST index into session.trials (raw trial order) — the same space
    build_population_tensor / extract_unit_psths use. NOT the HMM valid-trial
    ordering. See spec sec 4.2 (index-space contract).
    """
    for _, label, _ in rows:
        if label not in CANONICAL_STATES:
            raise ValueError(
                f"state_label {label!r} not in canonical {CANONICAL_STATES}")
    states_dir = Path(states_dir)
    states_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["trial_idx", "state_label", "confidence"])
    df["trial_idx"] = df["trial_idx"].astype(int)
    out = state_table_path(session_name, states_dir)
    df.to_csv(out, index=False)
    return out


def load_state_table(session_name: str, states_dir
                     ) -> Dict[int, Tuple[str, float]]:
    """Return {raw trial_idx -> (state_label, confidence)}; {} if no file."""
    path = state_table_path(session_name, states_dir)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {int(r["trial_idx"]): (str(r["state_label"]), float(r["confidence"]))
            for _, r in df.iterrows()}


def in_zone_trial_indices(session_name: str, states_dir,
                          min_confidence: float = 0.0) -> List[int]:
    """Sorted raw trial indices labeled in_zone with confidence >= floor."""
    table = load_state_table(session_name, states_dir)
    return sorted(t for t, (lab, conf) in table.items()
                  if lab == IN_ZONE and conf >= min_confidence)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/analysis/test_state_provider.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/state_provider.py tests/analysis/test_state_provider.py
git commit -m "feat(curation): state vocabulary + state-table IO + in_zone selector"
```

---

## Task 2: HMM→canonical mapping + provider classes

**Files:**
- Modify: `src/visdetect/analysis/state_provider.py`
- Test: `tests/analysis/test_state_provider.py`

Adds `canonical_from_hmm_label`, a `UniformInZoneStateProvider` (bootstrap — labels every valid trial `in_zone`, lets the pipeline run before the final state method exists), and a thin `HMMStateProvider` whose row-assembly is the testable pure function `rows_from_decoded_df`.

- [ ] **Step 1: Write the failing test (append)**

```python
# tests/analysis/test_state_provider.py  (append)
from visdetect.core.session import Session, Trial, Cluster


def test_canonical_from_hmm_label_maps_three():
    assert sp.canonical_from_hmm_label("Stimulus_sensitive") == "in_zone"
    assert sp.canonical_from_hmm_label("Impulsive") == "impulsive"
    assert sp.canonical_from_hmm_label("Disengaged") == "disengaged"


def test_canonical_from_hmm_label_strips_rank_suffix():
    assert sp.canonical_from_hmm_label("Impulsive_2") == "impulsive"


def test_canonical_from_hmm_label_unknown_returns_none():
    assert sp.canonical_from_hmm_label("Intermediate_1") is None


def test_rows_from_decoded_df_uses_trial_idx_column():
    df = pd.DataFrame({
        "trial_idx": [0, 3, 7],                         # raw session.trials index
        "hmm_state_label": ["Stimulus_sensitive", "Impulsive", "Intermediate_0"],
        "p_state_max": [0.9, 0.8, 0.55],
    })
    rows = sp.rows_from_decoded_df(df)
    # Intermediate_0 -> None canonical -> dropped
    assert rows == [(0, "in_zone", pytest.approx(0.9)),
                    (3, "impulsive", pytest.approx(0.8))]


def _toy_session():
    trials = [Trial(trialoutcome="Hit", reactiontimes={"Hit": 0.4},
                    change_size=2.0, orientation=None, ITI=1.0,
                    change_time=2.3, baseline_values=None) for _ in range(4)]
    clusters = [Cluster(cluster_id=0, spike_times=__import__("numpy").array([0.1, 0.2]),
                        quality=None)]
    return Session(trials=trials, clusters=clusters, subject="S",
                   session_name="07072025",
                   good_cluster_ids=[0],
                   ni_events={"Baseline_ON": __import__("numpy").arange(4) * 3.0,
                              "Change_ON": __import__("numpy").arange(4) * 3.0 + 2.3})


def test_uniform_inzone_provider_labels_all_valid(tmp_path):
    sess = _toy_session()
    prov = sp.UniformInZoneStateProvider()
    prov.write(sess, "07072025", tmp_path)
    idx = sp.in_zone_trial_indices("07072025", tmp_path)
    assert idx == [0, 1, 2, 3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/analysis/test_state_provider.py -v`
Expected: FAIL (`AttributeError: ... has no attribute 'canonical_from_hmm_label'`).

- [ ] **Step 3: Write minimal implementation (append to `state_provider.py`)**

```python
# src/visdetect/analysis/state_provider.py  (append)
import re
from typing import List, Optional

from visdetect.core.session import Session

_HMM_CANONICAL = {
    "Stimulus_sensitive": IN_ZONE,
    "Impulsive": IMPULSIVE,
    "Disengaged": DISENGAGED,
}


def canonical_from_hmm_label(label: str) -> Optional[str]:
    """Map an HMM label to the canonical vocabulary; None if not one of the three.

    Strips a trailing rank suffix ('_1', '_2') produced by
    hmm.auto_label_states_explicit for duplicate states. 'Intermediate_*' has no
    canonical equivalent -> None (trial gets no state -> excluded from in_zone).
    """
    base = re.sub(r"_\d+$", "", str(label))
    return _HMM_CANONICAL.get(base)


def rows_from_decoded_df(df) -> List[Tuple[int, str, float]]:
    """Convert a decode_session DataFrame to state-table rows.

    Requires columns 'trial_idx' (raw index), 'hmm_state_label', 'p_state_max'.
    Rows whose label has no canonical mapping are dropped.
    """
    rows: List[Tuple[int, str, float]] = []
    for _, r in df.iterrows():
        canon = canonical_from_hmm_label(r["hmm_state_label"])
        if canon is None:
            continue
        rows.append((int(r["trial_idx"]), canon, float(r["p_state_max"])))
    return rows


class UniformInZoneStateProvider:
    """Bootstrap provider: labels EVERY valid trial 'in_zone' (confidence 1.0).

    Temporary — lets the curation pipeline run end-to-end before the final
    state-identification method exists. Equivalent to all-trials fingerprinting.
    """

    def write(self, session: Session, session_name: str, states_dir) -> Path:
        from visdetect.analysis.behavior import get_trial_dataframe
        df = get_trial_dataframe(session)
        rows = [(int(i), IN_ZONE, 1.0) for i in df["trial_idx"].tolist()]
        return write_state_table(session_name, rows, states_dir)


class HMMStateProvider:
    """Provider wrapping a fitted GLM-HMM via hmm.decode_session."""

    def __init__(self, model, state_labels: List[str]):
        self.model = model
        self.state_labels = state_labels

    def write(self, session: Session, session_name: str, states_dir) -> Path:
        from visdetect.analysis.hmm import decode_session
        df = decode_session(self.model, session, state_labels=self.state_labels)
        if "p_state_max" not in df.columns:
            pcols = [c for c in df.columns if c.startswith("p_state_")]
            df = df.copy()
            df["p_state_max"] = df[pcols].max(axis=1) if pcols else 1.0
        rows = rows_from_decoded_df(df)
        return write_state_table(session_name, rows, states_dir)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/analysis/test_state_provider.py -v`
Expected: PASS (all tests, ~11).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/state_provider.py tests/analysis/test_state_provider.py
git commit -m "feat(curation): HMM->canonical mapping + uniform/HMM state providers"
```

---

## Task 3: Spike-partition ISI histograms (held-out independence)

**Files:**
- Create: `src/visdetect/analysis/track_curation.py`
- Test: `tests/analysis/test_track_curation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_track_curation.py
import numpy as np
import pytest
from visdetect.analysis import track_curation as tc


def test_partitioned_isi_hists_disjoint_and_valid():
    rng = np.random.default_rng(0)
    spikes = np.cumsum(rng.exponential(0.05, size=4000))   # stationary-ish train
    cur, hold = tc.partitioned_isi_hists(spikes)
    assert cur.shape == (50,) and hold.shape == (50,)
    assert np.isfinite(cur).all() and np.isfinite(hold).all()
    # Same underlying distribution -> the two partitions correlate strongly
    r = np.corrcoef(cur, hold)[0, 1]
    assert r > 0.8


def test_partitioned_isi_hists_too_few_spikes_returns_nan():
    cur, hold = tc.partitioned_isi_hists(np.array([0.1, 0.2, 0.3]))
    assert np.isnan(cur).all() and np.isnan(hold).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/analysis/test_track_curation.py -v`
Expected: FAIL (`ModuleNotFoundError: visdetect.analysis.track_curation`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/visdetect/analysis/track_curation.py
"""Precision curation of UnitMatch cross-session tracks.

Expert->Naive backward sweep over the liberal UM registry: biophysical gate +
availability-gated in-zone functional corroborator, rolling anchor with
gap-bridge tolerance. Never alters the original registry. See
docs/superpowers/specs/2026-06-07-track-curation-design.md.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from visdetect.analysis.tracking_qc import isi_log_histogram


def partitioned_isi_hists(spike_times: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Two log-ISI histograms from disjoint spike partitions (even/odd index).

    The curation ISI feature uses one partition; validation uses the other, so
    ISI validation is statistically independent of the ISI curation feature
    (spec sec 8.1). Both estimate the same stationary fingerprint.

    Returns (curation_hist, holdout_hist), each shape (50,); all-NaN if a
    partition has too few spikes.
    """
    st = np.asarray(spike_times, dtype=float)
    st = np.sort(st)
    cur = st[0::2]
    hold = st[1::2]
    cur_h, _ = isi_log_histogram(cur)
    hold_h, _ = isi_log_histogram(hold)
    return cur_h, hold_h
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/analysis/test_track_curation.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/track_curation.py tests/analysis/test_track_curation.py
git commit -m "feat(curation): spike-partition ISI histograms for held-out validation"
```

---

## Task 4: Extend `extract_unit_psths` with `restrict_trials`

**Files:**
- Modify: `src/visdetect/analysis/tracking_qc.py:752` (`extract_unit_psths`)
- Test: `tests/analysis/test_track_curation.py`

Adds an optional `restrict_trials` set so PSTHs can be conditioned on in-zone trials. Backward-compatible (default `None` = unchanged).

- [ ] **Step 1: Write the failing test (append)**

```python
# tests/analysis/test_track_curation.py  (append)
from visdetect.analysis import tracking_qc as qc
from visdetect.utils.synthetic import make_synthetic_session


def test_extract_unit_psths_restrict_trials_subsets():
    sess = make_synthetic_session(n_trials=40, n_clusters=3, seed=1)
    full = qc.extract_unit_psths(sess, ks_unit_id=0)
    restricted = qc.extract_unit_psths(sess, ks_unit_id=0, restrict_trials={0, 1, 2, 3, 4})
    # baseline_on uses all trials when unrestricted; restricting lowers n_trials
    assert restricted["baseline_on"][2] <= 5
    assert full["baseline_on"][2] >= restricted["baseline_on"][2]


def test_extract_unit_psths_empty_restrict_returns_none():
    sess = make_synthetic_session(n_trials=40, n_clusters=3, seed=1)
    out = qc.extract_unit_psths(sess, ks_unit_id=0, restrict_trials=set())
    assert out["baseline_on"] == (None, None, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/analysis/test_track_curation.py -k restrict -v`
Expected: FAIL (`TypeError: extract_unit_psths() got an unexpected keyword argument 'restrict_trials'`).

- [ ] **Step 3: Modify `extract_unit_psths`**

Change the signature and add the intersection. Replace the function header and the trial-index block:

```python
def extract_unit_psths(session, ks_unit_id: int,
                       restrict_trials: Optional[Set[int]] = None,
                        ) -> Dict[str, Tuple[np.ndarray, np.ndarray, int]]:
```

Inside the loop, replace the existing:

```python
        trial_idx = _trial_indices_for_sizes(session, cfg["sizes"])
        if trial_idx is not None and len(trial_idx) == 0:
            out[key] = (None, None, 0)
            continue
```

with:

```python
        trial_idx = _trial_indices_for_sizes(session, cfg["sizes"])
        if restrict_trials is not None:
            allowed = set(int(t) for t in restrict_trials)
            if trial_idx is None:
                trial_idx = sorted(allowed)
            else:
                trial_idx = [i for i in trial_idx if i in allowed]
        if trial_idx is not None and len(trial_idx) == 0:
            out[key] = (None, None, 0)
            continue
```

(`Optional` and `Set` are already imported at the top of `tracking_qc.py`.)

- [ ] **Step 4: Run tests to verify pass (new + regression)**

Run: `.venv\Scripts\python.exe -m pytest tests/analysis/test_track_curation.py tests/analysis/test_tracking_qc.py -v`
Expected: PASS (new restrict tests + all existing tracking_qc tests still green).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/tracking_qc.py tests/analysis/test_track_curation.py
git commit -m "feat(curation): extract_unit_psths gains optional restrict_trials (in-zone conditioning)"
```

---

## Task 5: `CurationFeature` + `extract_curation_feature`

**Files:**
- Modify: `src/visdetect/analysis/track_curation.py`
- Test: `tests/analysis/test_track_curation.py`

Builds the per-(session,uid) feature record: biophysics (waveform/footprint/drift-corrected depth/FR), curation+holdout ISI hists, and in-zone PSTHs.

- [ ] **Step 1: Write the failing test (append)**

```python
# tests/analysis/test_track_curation.py  (append)
import os


def _write_toy_waveforms(root, session, kid, n_samples=82, n_ch=20, seed=0):
    """Write a UM-style RawWaveforms npy + channel_positions for one unit."""
    rng = np.random.default_rng(seed)
    sess_dir = os.path.join(str(root), session, "RawWaveforms")
    os.makedirs(sess_dir, exist_ok=True)
    wf = rng.standard_normal((n_samples, n_ch, 2)).astype(np.float32)
    # give channel 5 a clear peak so peak-channel detection is deterministic
    wf[40, 5, :] += 30.0
    wf[20, 5, :] -= 30.0
    np.save(os.path.join(sess_dir, f"Unit{kid}_RawSpikes.npy"), wf)
    pos = np.zeros((n_ch, 2), dtype=np.float32)
    pos[:, 1] = np.arange(n_ch) * 20.0      # y-depth 20 um spacing
    np.save(os.path.join(str(root), session, "channel_positions.npy"), pos)


def test_extract_curation_feature_assembles_record(tmp_path):
    sess = make_synthetic_session(n_trials=40, n_clusters=3, seed=2)
    _write_toy_waveforms(tmp_path, "07072025", kid=0)
    cp = qc.load_channel_positions(tmp_path, "07072025")
    feat = tc.extract_curation_feature(
        sess, ks_unit_id=0, session_name="07072025", stage="Expert",
        raw_wf_root=tmp_path, channel_positions=cp,
        in_zone_idx=list(range(40)), drift_offset=0.0,
    )
    assert feat.session_name == "07072025"
    assert feat.peak_depth_um == pytest.approx(5 * 20.0)        # channel 5 * 20um
    assert feat.peak_depth_corrected_um == pytest.approx(5 * 20.0)
    assert feat.waveform_peak.shape[0] == 82
    assert feat.isi_hist_curation.shape == (50,)
    assert feat.isi_hist_holdout.shape == (50,)
    assert "baseline_on" in feat.inzone_psths
    assert feat.n_inzone_trials == 40


def test_extract_curation_feature_missing_waveform_returns_none(tmp_path):
    sess = make_synthetic_session(n_trials=20, n_clusters=1, seed=3)
    feat = tc.extract_curation_feature(
        sess, ks_unit_id=0, session_name="07072025", stage="Expert",
        raw_wf_root=tmp_path, channel_positions=None,
        in_zone_idx=list(range(20)), drift_offset=0.0,
    )
    assert feat is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/analysis/test_track_curation.py -k curation_feature -v`
Expected: FAIL (`AttributeError: ... 'extract_curation_feature'`).

- [ ] **Step 3: Add implementation (append to `track_curation.py`)**

```python
# src/visdetect/analysis/track_curation.py  (append)
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from visdetect.analysis.tracking_qc import (
    load_raw_mean_waveform, extract_peak_channel, extract_footprint,
    extract_unit_psths,
)


@dataclass
class CurationFeature:
    session_name: str
    ks_unit_id: int
    stage: str
    waveform_peak: np.ndarray
    footprint: np.ndarray
    footprint_channels: np.ndarray
    peak_chan: int
    peak_depth_um: float
    peak_depth_corrected_um: float
    baseline_fr_hz: float
    isi_hist_curation: np.ndarray
    isi_hist_holdout: np.ndarray
    inzone_psths: Dict[str, Optional[np.ndarray]]
    n_inzone_trials: int


def _baseline_fr(cluster, session) -> float:
    st = np.asarray(cluster.spike_times, dtype=float)
    if st.size == 0:
        return 0.0
    dur = float(st.max() - st.min())
    return float(st.size / dur) if dur > 0 else 0.0


def extract_curation_feature(session, ks_unit_id: int, session_name: str,
                             stage: str, raw_wf_root,
                             channel_positions: Optional[np.ndarray],
                             in_zone_idx: List[int],
                             drift_offset: float = 0.0,
                             ) -> Optional[CurationFeature]:
    """Assemble a CurationFeature for one (session, uid). None if no waveform."""
    cluster_map = {c.cluster_id: c for c in session.clusters}
    cluster = cluster_map.get(int(ks_unit_id))
    if cluster is None:
        return None

    mean_wf = load_raw_mean_waveform(raw_wf_root, session_name, int(ks_unit_id))
    if mean_wf is None:
        return None
    peak_chan = extract_peak_channel(mean_wf)
    peak_wave = mean_wf[:, peak_chan]
    footprint, fp_chans = extract_footprint(mean_wf, peak_chan)

    if channel_positions is not None and peak_chan < channel_positions.shape[0]:
        depth_um = float(channel_positions[peak_chan, 1])
    else:
        depth_um = float("nan")
    depth_corr = depth_um - float(drift_offset) if np.isfinite(depth_um) else float("nan")

    cur_h, hold_h = partitioned_isi_hists(np.asarray(cluster.spike_times))

    in_zone_set = set(int(i) for i in in_zone_idx)
    psth_dict = extract_unit_psths(session, int(ks_unit_id),
                                   restrict_trials=in_zone_set)
    inzone_psths = {k: v[0] for k, v in psth_dict.items()}

    return CurationFeature(
        session_name=session_name, ks_unit_id=int(ks_unit_id), stage=stage,
        waveform_peak=peak_wave.astype(np.float32),
        footprint=footprint.astype(np.float32), footprint_channels=fp_chans,
        peak_chan=peak_chan, peak_depth_um=depth_um,
        peak_depth_corrected_um=depth_corr,
        baseline_fr_hz=_baseline_fr(cluster, session),
        isi_hist_curation=cur_h, isi_hist_holdout=hold_h,
        inzone_psths=inzone_psths, n_inzone_trials=len(in_zone_set),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/analysis/test_track_curation.py -k curation_feature -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/track_curation.py tests/analysis/test_track_curation.py
git commit -m "feat(curation): CurationFeature + extract_curation_feature"
```

---

## Task 6: `CurationParams` + `LinkResult` + `score_link`

**Files:**
- Modify: `src/visdetect/analysis/track_curation.py`
- Test: `tests/analysis/test_track_curation.py`

The per-link rule: biophysical gate (waveform + drift-corrected depth via reused `badge_*`) decides KEEP/SKIP/STOP; ISI-shape and availability-gated functional corroborator only set the `review_flag`.

- [ ] **Step 1: Write the failing test (append)**

```python
# tests/analysis/test_track_curation.py  (append)
def _feat(session_name, *, wave, depth, isi, psth_val, n_inzone, n_bins=40):
    """Build a minimal CurationFeature for score_link tests."""
    wave = np.asarray(wave, dtype=float)
    isi = np.asarray(isi, dtype=float)
    psth = None if psth_val is None else np.full(n_bins, 0.0)
    psths = {}
    if psth_val is not None:
        # a modulated ramp scaled by psth_val so two features correlate or not
        ramp = np.linspace(0, 1, n_bins) * 10.0
        psths["baseline_on"] = ramp * psth_val
    return tc.CurationFeature(
        session_name=session_name, ks_unit_id=0, stage="Expert",
        waveform_peak=wave, footprint=np.zeros((1, 1)), footprint_channels=np.array([0]),
        peak_chan=0, peak_depth_um=depth, peak_depth_corrected_um=depth,
        baseline_fr_hz=5.0, isi_hist_curation=isi, isi_hist_holdout=isi,
        inzone_psths=psths, n_inzone_trials=n_inzone,
    )


_W = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.5, -0.5, 0.2])
_ISI = np.linspace(0, 1, 50)


def test_score_link_clean_pair_keeps_no_flag():
    p = tc.CurationParams()
    a = _feat("S2", wave=_W, depth=100.0, isi=_ISI, psth_val=1.0, n_inzone=50)
    b = _feat("S1", wave=_W, depth=102.0, isi=_ISI, psth_val=1.0, n_inzone=50)
    lr = tc.score_link(a, b, a, p, gap_sessions=1)
    assert lr.decision == "KEEP"
    assert lr.review_flag is False
    assert lr.func_evaluable is True


def test_score_link_hard_contradiction_stops():
    p = tc.CurationParams()
    a = _feat("S2", wave=_W, depth=100.0, isi=_ISI, psth_val=1.0, n_inzone=50)
    b = _feat("S1", wave=-_W, depth=200.0, isi=_ISI, psth_val=1.0, n_inzone=50)  # flipped wf + 100um jump
    lr = tc.score_link(a, b, a, p, gap_sessions=1)
    assert lr.decision == "STOP"
    assert lr.stop_reason == "hard_contradiction"


def test_score_link_soft_depth_warn_skips():
    p = tc.CurationParams()
    a = _feat("S2", wave=_W, depth=100.0, isi=_ISI, psth_val=1.0, n_inzone=50)
    b = _feat("S1", wave=_W, depth=130.0, isi=_ISI, psth_val=1.0, n_inzone=50)  # 30um = depth warn
    lr = tc.score_link(a, b, a, p, gap_sessions=1)
    assert lr.decision == "SKIP"


def test_score_link_func_conflict_flags_review_but_keeps():
    p = tc.CurationParams()
    a = _feat("S2", wave=_W, depth=100.0, isi=_ISI, psth_val=1.0, n_inzone=50)
    b = _feat("S1", wave=_W, depth=101.0, isi=_ISI, psth_val=-1.0, n_inzone=50)  # anti-correlated PSTH
    lr = tc.score_link(a, b, a, p, gap_sessions=1)
    assert lr.decision == "KEEP"
    assert lr.review_flag is True


def test_score_link_func_not_evaluable_when_few_inzone():
    p = tc.CurationParams()      # min_inzone_trials default 20
    a = _feat("S2", wave=_W, depth=100.0, isi=_ISI, psth_val=1.0, n_inzone=50)
    b = _feat("S1", wave=_W, depth=101.0, isi=_ISI, psth_val=-1.0, n_inzone=5)  # too few in-zone
    lr = tc.score_link(a, b, a, p, gap_sessions=1)
    assert lr.decision == "KEEP"
    assert lr.func_evaluable is False
    assert lr.review_flag is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/analysis/test_track_curation.py -k score_link -v`
Expected: FAIL (`AttributeError: ... 'CurationParams'`).

- [ ] **Step 3: Add implementation (append to `track_curation.py`)**

```python
# src/visdetect/analysis/track_curation.py  (append)
from visdetect.analysis.tracking_qc import (
    badge_waveform, badge_depth, badge_isi_hist_corr, badge_func_resp,
    FUNC_RESP_MIN_PSTH_STD,
)

MAX_BRIDGE_GAP = 2
MIN_INZONE_TRIALS = 20
MIN_TRUSTED_SPAN = 3


@dataclass
class CurationParams:
    max_bridge_gap: int = MAX_BRIDGE_GAP
    min_inzone_trials: int = MIN_INZONE_TRIALS
    min_trusted_span: int = MIN_TRUSTED_SPAN
    corroborator_ref: str = "rolling"     # "rolling" | "expert"


@dataclass
class LinkResult:
    anchor_session: str
    candidate_session: str
    gap_sessions: int
    wave_corr: float
    depth_jump_um: float
    isi_shape_corr: float
    func_corr: float
    func_evaluable: bool
    n_inzone_trials: int
    decision: str           # "KEEP" | "SKIP" | "STOP"
    review_flag: bool
    stop_reason: str = ""


def _pearson(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return float("nan")
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n = min(a.size, b.size)
    if n < 2:
        return float("nan")
    a, b = a[:n], b[:n]
    if np.isnan(a).any() or np.isnan(b).any():
        return float("nan")
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _func_corr(ref: CurationFeature, cand: CurationFeature) -> float:
    """Median pairwise Pearson r over conditions both have modulated PSTHs for."""
    rs: List[float] = []
    for key, ref_psth in ref.inzone_psths.items():
        cand_psth = cand.inzone_psths.get(key)
        if ref_psth is None or cand_psth is None:
            continue
        if (float(np.std(ref_psth)) < FUNC_RESP_MIN_PSTH_STD
                or float(np.std(cand_psth)) < FUNC_RESP_MIN_PSTH_STD):
            continue
        r = _pearson(ref_psth, cand_psth)
        if np.isfinite(r):
            rs.append(r)
    return float(np.median(rs)) if rs else float("nan")


def score_link(anchor: CurationFeature, candidate: CurationFeature,
               corroborator_ref: CurationFeature, params: CurationParams,
               gap_sessions: int = 1) -> LinkResult:
    """Decide one cross-session link: biophysical gate + functional corroborator."""
    wave_corr = _pearson(anchor.waveform_peak, candidate.waveform_peak)
    depth_jump = abs(anchor.peak_depth_corrected_um
                     - candidate.peak_depth_corrected_um)
    isi_corr = _pearson(anchor.isi_hist_curation, candidate.isi_hist_curation)

    w = badge_waveform(wave_corr)
    d = badge_depth(depth_jump)

    # Functional corroborator (availability-gated).
    func_evaluable = candidate.n_inzone_trials >= params.min_inzone_trials
    func_corr = _func_corr(corroborator_ref, candidate) if func_evaluable else float("nan")
    if func_evaluable and not np.isfinite(func_corr):
        func_evaluable = False          # no modulated condition -> not evaluable

    base = dict(
        anchor_session=anchor.session_name,
        candidate_session=candidate.session_name,
        gap_sessions=int(gap_sessions),
        wave_corr=wave_corr, depth_jump_um=depth_jump, isi_shape_corr=isi_corr,
        func_corr=func_corr, func_evaluable=func_evaluable,
        n_inzone_trials=candidate.n_inzone_trials,
    )

    if w == "fail" and d == "fail":
        return LinkResult(**base, decision="STOP", review_flag=False,
                          stop_reason="hard_contradiction")
    if w == "pass" and d == "pass":
        review = (badge_isi_hist_corr(isi_corr) != "pass")
        if func_evaluable and badge_func_resp(func_corr) != "pass":
            review = True
        return LinkResult(**base, decision="KEEP", review_flag=review)
    return LinkResult(**base, decision="SKIP", review_flag=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/analysis/test_track_curation.py -k score_link -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/track_curation.py tests/analysis/test_track_curation.py
git commit -m "feat(curation): per-link score_link (biophysical gate + functional corroborator)"
```

---

## Task 7: `sweep_uid` + `compute_tier`

**Files:**
- Modify: `src/visdetect/analysis/track_curation.py`
- Test: `tests/analysis/test_track_curation.py`

The Expert→Naive backward sweep with rolling anchor + gap-bridge, and the per-track confidence tier.

- [ ] **Step 1: Write the failing test (append)**

```python
# tests/analysis/test_track_curation.py  (append)
def _chain_feats(session_names, *, swap_at=None, dropout_at=None):
    """Build a per-session feature dict for a clean chain, with optional defects.

    swap_at: session name whose unit is a different neuron (flipped wf + depth jump).
    dropout_at: session name whose unit is garbled (flipped wf only -> soft skip).
    """
    feats = {}
    for s in session_names:
        wave, depth = _W.copy(), 100.0
        if s == swap_at:
            wave, depth = -_W.copy(), 220.0      # hard contradiction
        elif s == dropout_at:
            depth = 130.0                         # soft (depth warn) -> SKIP
        feats[s] = _feat(s, wave=wave, depth=depth, isi=_ISI, psth_val=1.0, n_inzone=50)
    return feats


def test_sweep_clean_chain_is_one_trusted_track():
    p = tc.CurationParams()
    order = ["S1", "S2", "S3", "S4"]            # chronological ascending
    feats = _chain_feats(order)
    res = tc.sweep_uid(feats, order, p)
    assert res.anchor_session == "S4"
    assert set(res.kept_sessions) == {"S1", "S2", "S3", "S4"}
    assert res.confidence_tier == "trusted"


def test_sweep_mid_chain_swap_stops_and_truncates():
    p = tc.CurationParams()
    order = ["S1", "S2", "S3", "S4"]
    feats = _chain_feats(order, swap_at="S2")    # walking back S4->S3->S2 hits swap
    res = tc.sweep_uid(feats, order, p)
    assert "S2" in res.dropped_sessions and "S1" in res.dropped_sessions
    assert set(res.kept_sessions) == {"S3", "S4"}


def test_sweep_single_dropout_is_bridged():
    p = tc.CurationParams()                       # max_bridge_gap default 2
    order = ["S1", "S2", "S3", "S4"]
    feats = _chain_feats(order, dropout_at="S3")  # S3 soft-fails, S2/S1 clean -> resurface
    res = tc.sweep_uid(feats, order, p)
    assert "S3" in res.skipped_sessions
    assert set(res.kept_sessions) == {"S1", "S2", "S4"}
    assert res.confidence_tier == "review"        # a bridge present


def test_sweep_skips_exhausted_drops_trailing():
    p = tc.CurationParams(max_bridge_gap=1)
    order = ["S1", "S2", "S3", "S4"]
    # S3 and S2 both soft-fail -> 2 consecutive skips > max_bridge_gap=1 -> STOP
    feats = _chain_feats(order)
    feats["S3"] = _feat("S3", wave=_W, depth=130.0, isi=_ISI, psth_val=1.0, n_inzone=50)
    feats["S2"] = _feat("S2", wave=_W, depth=130.0, isi=_ISI, psth_val=1.0, n_inzone=50)
    res = tc.sweep_uid(feats, order, p)
    assert res.kept_sessions == ["S4"]
    assert "S3" in res.dropped_sessions and "S2" in res.dropped_sessions
    assert res.confidence_tier == "suspect"       # span 1


def test_compute_tier_short_is_suspect():
    p = tc.CurationParams()
    assert tc.compute_tier(["S4"], [], [], p) == "suspect"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/analysis/test_track_curation.py -k "sweep or compute_tier" -v`
Expected: FAIL (`AttributeError: ... 'sweep_uid'`).

- [ ] **Step 3: Add implementation (append to `track_curation.py`)**

```python
# src/visdetect/analysis/track_curation.py  (append)
@dataclass
class SweepResult:
    liberal_uid: int
    anchor_session: str
    kept_sessions: List[str]
    skipped_sessions: List[str]
    dropped_sessions: List[str]
    links: List[LinkResult] = field(default_factory=list)
    confidence_tier: str = "suspect"


def compute_tier(kept_sessions: List[str], skipped_sessions: List[str],
                 kept_links: List[LinkResult], params: CurationParams) -> str:
    """trusted / review / suspect for a curated track (spec sec 6.2)."""
    span = len(kept_sessions)
    if span < 2:
        return "suspect"
    any_review = any(lr.review_flag for lr in kept_links)
    any_bridge = len(skipped_sessions) > 0
    if span >= params.min_trusted_span and not any_review and not any_bridge:
        return "trusted"
    return "review"


def sweep_uid(features_by_session: Dict[str, CurationFeature],
              session_order: List[str], params: CurationParams,
              liberal_uid: int = -1) -> SweepResult:
    """Expert->Naive backward sweep over one liberal-uid's sessions.

    session_order: chronological ascending; anchor = most-recent (last).
    """
    present = [s for s in session_order if s in features_by_session]
    if not present:
        return SweepResult(liberal_uid, "", [], [], list(session_order))
    anchor_sess = present[-1]
    expert_anchor = features_by_session[anchor_sess]
    anchor = expert_anchor
    anchor_pos = len(present) - 1

    kept = [anchor_sess]
    skipped: List[str] = []
    dropped: List[str] = []
    pending: List[str] = []
    links: List[LinkResult] = []
    n_bridge = 0

    i = len(present) - 2
    while i >= 0:
        cand_sess = present[i]
        cand = features_by_session[cand_sess]
        ref = anchor if params.corroborator_ref == "rolling" else expert_anchor
        lr = score_link(anchor, cand, ref, params, gap_sessions=anchor_pos - i)
        links.append(lr)
        if lr.decision == "KEEP":
            kept.append(cand_sess)
            skipped.extend(pending); pending = []
            anchor = cand; anchor_pos = i; n_bridge = 0
        elif lr.decision == "SKIP":
            pending.append(cand_sess); n_bridge += 1
            if n_bridge > params.max_bridge_gap:
                dropped.extend(pending); pending = []
                dropped.extend(present[:i])         # all earlier sessions
                break
        else:  # STOP
            dropped.extend(pending); pending = []
            dropped.append(cand_sess)
            dropped.extend(present[:i])
            break
        i -= 1
    dropped.extend(pending)                          # trailing unclosed skips

    kept_links = [lr for lr in links if lr.decision == "KEEP"]
    tier = compute_tier(kept, skipped, kept_links, params)
    return SweepResult(
        liberal_uid=liberal_uid, anchor_session=anchor_sess,
        kept_sessions=kept, skipped_sessions=skipped, dropped_sessions=dropped,
        links=links, confidence_tier=tier,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/analysis/test_track_curation.py -k "sweep or compute_tier" -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/track_curation.py tests/analysis/test_track_curation.py
git commit -m "feat(curation): backward sweep_uid + compute_tier (rolling anchor, gap-bridge)"
```

---

## Task 8: `curate_registry` orchestrator (features → links_df, tracks_df)

**Files:**
- Modify: `src/visdetect/analysis/track_curation.py`
- Test: `tests/analysis/test_track_curation.py`

Turns a per-(uid,session) feature dict + chronological order into the two output DataFrames (spec sec 7).

- [ ] **Step 1: Write the failing test (append)**

```python
# tests/analysis/test_track_curation.py  (append)
def test_curate_registry_builds_links_and_tracks():
    p = tc.CurationParams()
    order = ["S1", "S2", "S3", "S4"]
    uid_to_sessions = {10: order, 11: ["S1", "S2"]}
    feats = {}
    for s in order:
        feats[(10, s)] = _feat(s, wave=_W, depth=100.0, isi=_ISI, psth_val=1.0, n_inzone=50)
    # uid 11: a swap between its two sessions -> short/suspect
    feats[(11, "S2")] = _feat("S2", wave=_W, depth=100.0, isi=_ISI, psth_val=1.0, n_inzone=50)
    feats[(11, "S1")] = _feat("S1", wave=-_W, depth=220.0, isi=_ISI, psth_val=1.0, n_inzone=50)

    links_df, tracks_df = tc.curate_registry(uid_to_sessions, feats, p)

    t10 = tracks_df[tracks_df.liberal_uid == 10].iloc[0]
    assert t10.confidence_tier == "trusted"
    assert t10.trimmed_span == 4
    t11 = tracks_df[tracks_df.liberal_uid == 11].iloc[0]
    assert t11.confidence_tier == "suspect"
    assert set(links_df.columns) >= {
        "liberal_uid", "anchor_session", "candidate_session", "wave_corr",
        "depth_jump_um", "isi_shape_corr", "func_corr", "func_evaluable",
        "link_decision", "review_flag", "stop_reason"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/analysis/test_track_curation.py -k curate_registry -v`
Expected: FAIL (`AttributeError: ... 'curate_registry'`).

- [ ] **Step 3: Add implementation (append to `track_curation.py`)**

```python
# src/visdetect/analysis/track_curation.py  (append)
import pandas as pd


def curate_registry(uid_to_sessions: Dict[int, List[str]],
                    features: Dict, params: CurationParams):
    """Run the sweep for every uid; return (links_df, tracks_df).

    uid_to_sessions: {liberal_uid -> chronological-ascending session list}.
    features: {(liberal_uid, session_name) -> CurationFeature}.
    """
    link_rows: List[dict] = []
    track_rows: List[dict] = []
    for uid in sorted(uid_to_sessions):
        order = uid_to_sessions[uid]
        feats = {s: features[(uid, s)] for s in order if (uid, s) in features}
        res = sweep_uid(feats, [s for s in order if s in feats], params,
                        liberal_uid=uid)
        for lr in res.links:
            link_rows.append({
                "liberal_uid": uid,
                "anchor_session": lr.anchor_session,
                "candidate_session": lr.candidate_session,
                "gap_sessions": lr.gap_sessions,
                "wave_corr": lr.wave_corr,
                "depth_jump_um": lr.depth_jump_um,
                "isi_shape_corr": lr.isi_shape_corr,
                "func_corr": lr.func_corr,
                "func_evaluable": lr.func_evaluable,
                "n_inzone_trials": lr.n_inzone_trials,
                "link_decision": lr.decision,
                "review_flag": lr.review_flag,
                "stop_reason": lr.stop_reason,
            })
        track_rows.append({
            "curated_uid": uid,            # 1:1 with liberal_uid (Expert-anchored)
            "liberal_uid": uid,
            "anchor_session": res.anchor_session,
            "kept_sessions": ";".join(res.kept_sessions),
            "skipped_sessions": ";".join(res.skipped_sessions),
            "dropped_sessions": ";".join(res.dropped_sessions),
            "trimmed_span": len(res.kept_sessions),
            "n_bridged": len(res.skipped_sessions),
            "confidence_tier": res.confidence_tier,
        })
    return pd.DataFrame(link_rows), pd.DataFrame(track_rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/analysis/test_track_curation.py -k curate_registry -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/track_curation.py tests/analysis/test_track_curation.py
git commit -m "feat(curation): curate_registry orchestrator -> links_df + tracks_df"
```

---

## Task 9: `held_out_isi_auc_by_tier` (validation)

**Files:**
- Modify: `src/visdetect/analysis/track_curation.py`
- Test: `tests/analysis/test_track_curation.py`

Scores curation precision via the held-out ISI partition (spec sec 8.2): matched cross-session (same curated_uid, kept sessions) vs non-matched within-session, AUC per tier.

- [ ] **Step 1: Write the failing test (append)**

```python
# tests/analysis/test_track_curation.py  (append)
def test_held_out_isi_auc_separates_matched_from_nonmatched():
    rng = np.random.default_rng(0)
    # Two distinct unit "shapes": A peaks early, B peaks late.
    shapeA = np.exp(-((np.arange(50) - 10) ** 2) / 20.0)
    shapeB = np.exp(-((np.arange(50) - 40) ** 2) / 20.0)

    def noisy(shape):
        h = shape + rng.normal(0, 0.02, size=50)
        h = np.clip(h, 0, None)
        return h / h.sum()

    # uid 1 = unit A across S1,S2,S3 ; uid 2 = unit B across S1,S2,S3
    holdout = {}
    for s in ["S1", "S2", "S3"]:
        holdout[(1, s)] = noisy(shapeA)
        holdout[(2, s)] = noisy(shapeB)
    tracks = pd.DataFrame([
        {"curated_uid": 1, "kept_sessions": "S1;S2;S3", "confidence_tier": "trusted"},
        {"curated_uid": 2, "kept_sessions": "S1;S2;S3", "confidence_tier": "trusted"},
    ])
    out = tc.held_out_isi_auc_by_tier(tracks, holdout)
    assert out["trusted"]["auc"] > 0.9
    assert out["trusted"]["n_matched"] == 6     # 2 uids * C(3,2)=3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/analysis/test_track_curation.py -k held_out -v`
Expected: FAIL (`AttributeError: ... 'held_out_isi_auc_by_tier'`).

- [ ] **Step 3: Add implementation (append to `track_curation.py`)**

```python
# src/visdetect/analysis/track_curation.py  (append)
from itertools import combinations


def _auc(matched: np.ndarray, nonmatched: np.ndarray) -> float:
    """ROC AUC of matched (label 1) vs nonmatched (label 0) scores."""
    if len(matched) == 0 or len(nonmatched) == 0:
        return float("nan")
    scores = np.concatenate([matched, nonmatched])
    labels = np.concatenate([np.ones_like(matched), np.zeros_like(nonmatched)])
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels); fp = np.cumsum(1 - labels)
    tpr = tp / max(1, labels.sum()); fpr = fp / max(1, (1 - labels).sum())
    return float(np.trapz(tpr, fpr))


def held_out_isi_auc_by_tier(tracks_df, holdout_isi: Dict) -> Dict[str, dict]:
    """Per-tier held-out-ISI AUC (spec sec 8.2).

    tracks_df: must have curated_uid, kept_sessions (';'-joined), confidence_tier.
    holdout_isi: {(curated_uid, session) -> holdout ISI hist (50,)}.
    Matched = cross-session pairs within a curated_uid's kept sessions.
    Non-matched = within-session pairs across different curated_uids.
    """
    out: Dict[str, dict] = {}
    for tier, grp in tracks_df.groupby("confidence_tier"):
        matched: List[float] = []
        # matched: cross-session, same uid
        sess_by_uid: Dict[int, List[str]] = {}
        for _, row in grp.iterrows():
            uid = int(row["curated_uid"])
            sess = [s for s in str(row["kept_sessions"]).split(";") if s]
            sess_by_uid[uid] = sess
            for s1, s2 in combinations(sess, 2):
                r = _pearson(holdout_isi.get((uid, s1)), holdout_isi.get((uid, s2)))
                if np.isfinite(r):
                    matched.append(r)
        # non-matched: within-session, different uid
        nonmatched: List[float] = []
        uids = list(sess_by_uid)
        for u1, u2 in combinations(uids, 2):
            shared = set(sess_by_uid[u1]) & set(sess_by_uid[u2])
            for s in shared:
                r = _pearson(holdout_isi.get((u1, s)), holdout_isi.get((u2, s)))
                if np.isfinite(r):
                    nonmatched.append(r)
        out[str(tier)] = {
            "auc": _auc(np.array(matched), np.array(nonmatched)),
            "n_matched": len(matched),
            "n_nonmatched": len(nonmatched),
        }
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/analysis/test_track_curation.py -k held_out -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/track_curation.py tests/analysis/test_track_curation.py
git commit -m "feat(curation): held-out-ISI AUC by tier (validation)"
```

---

## Task 10: `make_state_tables.py` CLI (state-table generator)

**Files:**
- Create: `scripts/pipelines/tracking/make_state_tables.py`

Bootstrap (uniform in-zone) or HMM-backed state tables for all manifest sessions. Glue around Task 1–2 providers; no new unit logic.

- [ ] **Step 1: Write the script**

```python
# scripts/pipelines/tracking/make_state_tables.py
#!/usr/bin/env python3
"""Write per-session trial->state tables consumed by track curation.

Two modes:
  --provider uniform   bootstrap: every valid trial labeled in_zone (default)
  --provider hmm       use a fitted GLM-HMM (requires --model-path)

Usage:
    py scripts/pipelines/tracking/make_state_tables.py --provider uniform
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from visdetect.analysis import state_provider as sp                # noqa: E402
from visdetect.core.session import load_session                   # noqa: E402
from visdetect.suite.loader import load_filtered_manifest          # noqa: E402

DEFAULT_PKL_DIR = REPO_ROOT / "data" / "pkls" / "BG_046"
DEFAULT_STATES_DIR = REPO_ROOT / "data" / "cache" / "states" / "BG_046"


def _session_pkl(pkl_dir: Path, sess: str):
    for s in (sess, str(sess).zfill(8)):
        p = pkl_dir / f"BG_046_{s}.pkl"
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["uniform", "hmm"], default="uniform")
    ap.add_argument("--model-path", type=Path, default=None,
                    help="Fitted GLM-HMM pickle (required for --provider hmm)")
    ap.add_argument("--pkl-dir", type=Path, default=DEFAULT_PKL_DIR)
    ap.add_argument("--states-dir", type=Path, default=DEFAULT_STATES_DIR)
    args = ap.parse_args()

    if args.provider == "uniform":
        provider = sp.UniformInZoneStateProvider()
    else:
        if args.model_path is None:
            ap.error("--provider hmm requires --model-path")
        import pickle
        from visdetect.analysis.hmm import auto_label_states_explicit
        with open(args.model_path, "rb") as f:
            model = pickle.load(f)
        provider = sp.HMMStateProvider(model, auto_label_states_explicit(model))

    manifest = load_filtered_manifest(
        include_stages=["Naive", "Learning", "Expert"],
        merge_naive_learning=True, min_trials=150, min_dprime=None)
    n = 0
    for _, mrow in manifest.iterrows():
        sess = str(mrow["session_name"])
        pkl = _session_pkl(args.pkl_dir, sess)
        if pkl is None:
            print(f"  skip {sess}: no pkl", flush=True); continue
        S = load_session(str(pkl))
        out = provider.write(S, sess, args.states_dir)
        print(f"  wrote {out.name}", flush=True)
        n += 1
        del S
    print(f"Done: {n} state tables -> {args.states_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-run the bootstrap mode**

Run: `.venv\Scripts\python.exe scripts/pipelines/tracking/make_state_tables.py --provider uniform`
Expected: prints `wrote <session>_states.csv` per manifest session; ends `Done: N state tables`. Verify one CSV exists under `data\cache\states\BG_046\`.

- [ ] **Step 3: Commit**

```bash
git add scripts/pipelines/tracking/make_state_tables.py
git commit -m "feat(curation): make_state_tables CLI (uniform bootstrap + HMM provider)"
```

---

## Task 11: `curate_tracks.py` CLI (main runner)

**Files:**
- Create: `scripts/pipelines/tracking/curate_tracks.py`

Wires liberal registry + state tables + waveforms/pkls → feature cache → `curate_registry` → CSVs. Mirrors `build_qc_sheets.py`'s session-outer-loop caching and drift estimation.

- [ ] **Step 1: Write the script**

```python
# scripts/pipelines/tracking/curate_tracks.py
#!/usr/bin/env python3
"""Curate the liberal UnitMatch registry into precision tracks.

Outputs (FIGURES/tracking_qc/curation/):
    curated_links.csv    per-link audit trail
    curated_tracks.csv   per-track kept/skipped/dropped + confidence tier

Usage:
    py scripts/pipelines/tracking/curate_tracks.py [--min-span 2] [--rebuild-cache]
"""
from __future__ import annotations

import argparse
import gc
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from visdetect.analysis import state_provider as sp                 # noqa: E402
from visdetect.analysis import track_curation as tc                 # noqa: E402
from visdetect.analysis.tracking_qc import (                        # noqa: E402
    load_channel_positions, estimate_session_drift)
from visdetect.core.session import load_session                     # noqa: E402
from visdetect.suite.loader import load_filtered_manifest           # noqa: E402

UM_ROOT = Path("X:/public/projects/BeJG_20230130_VisDetect/wEPhys/"
               "BG_046/unit_match/output/all42")
DEFAULT_REGISTRY = UM_ROOT / "unit_index.csv"
DEFAULT_PROB_MATRIX = UM_ROOT / "batch0" / "output_prob_matrix.npy"
DEFAULT_RAW_WF_ROOT = REPO_ROOT / "data" / "unit_match" / "input" / "BG_046"
DEFAULT_PKL_DIR = REPO_ROOT / "data" / "pkls" / "BG_046"
DEFAULT_STATES_DIR = REPO_ROOT / "data" / "cache" / "states" / "BG_046"
DEFAULT_OUT_DIR = REPO_ROOT / "FIGURES" / "tracking_qc" / "curation"
DEFAULT_CACHE = REPO_ROOT / "data" / "cache" / "curation_features.pkl"


def _date_key(s: str) -> Tuple[int, int, int]:
    p = str(s).zfill(8)
    return (int(p[4:8]), int(p[2:4]), int(p[0:2]))


def _session_pkl(pkl_dir: Path, sess: str):
    for s in (sess, str(sess).zfill(8)):
        p = pkl_dir / f"BG_046_{s}.pkl"
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--liberal-col", default="batch_uid_liberal")
    ap.add_argument("--prob-matrix", type=Path, default=DEFAULT_PROB_MATRIX)
    ap.add_argument("--raw-wf-root", type=Path, default=DEFAULT_RAW_WF_ROOT)
    ap.add_argument("--pkl-dir", type=Path, default=DEFAULT_PKL_DIR)
    ap.add_argument("--states-dir", type=Path, default=DEFAULT_STATES_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--min-span", type=int, default=2)
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--max-bridge-gap", type=int, default=tc.MAX_BRIDGE_GAP)
    ap.add_argument("--min-inzone-trials", type=int, default=tc.MIN_INZONE_TRIALS)
    ap.add_argument("--min-trusted-span", type=int, default=tc.MIN_TRUSTED_SPAN)
    ap.add_argument("--corroborator-ref", choices=["rolling", "expert"], default="rolling")
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Registry re-keyed on the liberal column ──────────────────────────
    reg = pd.read_csv(args.registry)
    reg["session"] = reg["session"].astype(str)
    reg["uid"] = reg[args.liberal_col].astype(int)
    span = reg.groupby("uid")["session"].nunique()
    keep_uids = set(span[span >= args.min_span].index.tolist())
    reg = reg[reg["uid"].isin(keep_uids)].copy()
    uid_to_ks: Dict[int, Dict[str, int]] = {}
    for _, r in reg.iterrows():
        uid_to_ks.setdefault(int(r["uid"]), {})[str(r["session"])] = int(r["ks_unit_id"])
    print(f"liberal cohort: {len(keep_uids)} uids span>={args.min_span}", flush=True)

    manifest = load_filtered_manifest(
        include_stages=["Naive", "Learning", "Expert"],
        merge_naive_learning=True, min_trials=150, min_dprime=None)
    stage_by_sess = {str(r["session_name"]).zfill(8): str(r["stage"])
                     for _, r in manifest.iterrows()}

    # ── Drift offsets across all registry sessions ───────────────────────
    all_sess = sorted(reg["session"].unique().tolist(), key=_date_key)
    drift_offsets = {}
    if args.prob_matrix.exists():
        prob = np.load(args.prob_matrix)
        drift_offsets = estimate_session_drift(reg, prob, args.raw_wf_root, all_sess)
    else:
        print("prob matrix missing — depth uses raw (offset 0)", flush=True)

    # ── Build / load feature cache (outer loop by session) ───────────────
    if args.rebuild_cache or not args.cache_path.exists():
        features: Dict[Tuple[int, str], tc.CurationFeature] = {}
        for sess in all_sess:
            pkl = _session_pkl(args.pkl_dir, sess)
            if pkl is None:
                print(f"  skip {sess}: no pkl", flush=True); continue
            S = load_session(str(pkl))
            cp = load_channel_positions(args.raw_wf_root, sess)
            in_zone = sp.in_zone_trial_indices(sess, args.states_dir,
                                               min_confidence=args.min_confidence)
            off = float(drift_offsets.get(str(sess).zfill(8),
                        drift_offsets.get(sess, 0.0)))
            stage = stage_by_sess.get(str(sess).zfill(8), "Unknown")
            for uid, ksmap in uid_to_ks.items():
                if sess not in ksmap:
                    continue
                feat = tc.extract_curation_feature(
                    S, ksmap[sess], session_name=sess, stage=stage,
                    raw_wf_root=args.raw_wf_root, channel_positions=cp,
                    in_zone_idx=in_zone, drift_offset=off)
                if feat is not None:
                    features[(uid, sess)] = feat
            del S; gc.collect()
            print(f"  {sess}: features cached", flush=True)
        args.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.cache_path, "wb") as f:
            pickle.dump(features, f)
        print(f"  saved {len(features)} features -> {args.cache_path}", flush=True)
    else:
        with open(args.cache_path, "rb") as f:
            features = pickle.load(f)
        print(f"loaded {len(features)} cached features", flush=True)

    # ── Sweep + write ────────────────────────────────────────────────────
    uid_to_sessions = {uid: sorted(ks.keys(), key=_date_key)
                       for uid, ks in uid_to_ks.items()}
    params = tc.CurationParams(
        max_bridge_gap=args.max_bridge_gap, min_inzone_trials=args.min_inzone_trials,
        min_trusted_span=args.min_trusted_span, corroborator_ref=args.corroborator_ref)
    links_df, tracks_df = tc.curate_registry(uid_to_sessions, features, params)
    links_df.to_csv(args.out_dir / "curated_links.csv", index=False)
    tracks_df.to_csv(args.out_dir / "curated_tracks.csv", index=False)
    n_tier = tracks_df["confidence_tier"].value_counts().to_dict()
    print(f"Wrote curated_links.csv + curated_tracks.csv -> {args.out_dir}", flush=True)
    print(f"  tiers: {n_tier}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-run (requires state tables from Task 10 + restored waveforms/pkls)**

Run: `.venv\Scripts\python.exe scripts/pipelines/tracking/curate_tracks.py --min-span 2 --rebuild-cache`
Expected: prints cohort size, per-session caching lines, then `Wrote curated_links.csv + curated_tracks.csv` and a tier breakdown. Confirm both CSVs exist under `FIGURES\tracking_qc\curation\`.

- [ ] **Step 3: Commit**

```bash
git add scripts/pipelines/tracking/curate_tracks.py
git commit -m "feat(curation): curate_tracks CLI runner (liberal registry -> curated tracks)"
```

---

## Task 12: `validate_curation.py` CLI + end-to-end run + memory note

**Files:**
- Create: `scripts/pipelines/tracking/validate_curation.py`
- Modify: memory (`MEMORY.md` + `neuron_tracking_may2026.md`)

- [ ] **Step 1: Write the validation script**

```python
# scripts/pipelines/tracking/validate_curation.py
#!/usr/bin/env python3
"""Held-out-ISI AUC by confidence tier for a curated-track table (spec sec 8.2).

Usage:
    py scripts/pipelines/tracking/validate_curation.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from visdetect.analysis import track_curation as tc                 # noqa: E402
from visdetect.core.session import load_session                     # noqa: E402

UM_ROOT = Path("X:/public/projects/BeJG_20230130_VisDetect/wEPhys/"
               "BG_046/unit_match/output/all42")
DEFAULT_REGISTRY = UM_ROOT / "unit_index.csv"
DEFAULT_TRACKS = REPO_ROOT / "FIGURES" / "tracking_qc" / "curation" / "curated_tracks.csv"
DEFAULT_PKL_DIR = REPO_ROOT / "data" / "pkls" / "BG_046"
OUT_DIR = REPO_ROOT / "FIGURES" / "tracking_qc" / "curation"


def _session_pkl(pkl_dir: Path, sess: str):
    for s in (sess, str(sess).zfill(8)):
        p = pkl_dir / f"BG_046_{s}.pkl"
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--liberal-col", default="batch_uid_liberal")
    ap.add_argument("--pkl-dir", type=Path, default=DEFAULT_PKL_DIR)
    args = ap.parse_args()

    tracks = pd.read_csv(args.tracks)
    reg = pd.read_csv(args.registry)
    reg["session"] = reg["session"].astype(str)
    reg["uid"] = reg[args.liberal_col].astype(int)
    # (uid, session) -> ks_unit_id, restricted to kept sessions of each curated_uid
    kept_pairs: Dict[Tuple[int, str], int] = {}
    for _, row in tracks.iterrows():
        uid = int(row["curated_uid"])
        for s in [s for s in str(row["kept_sessions"]).split(";") if s]:
            m = reg[(reg["uid"] == uid) & (reg["session"] == s)]
            if len(m):
                kept_pairs[(uid, s)] = int(m.iloc[0]["ks_unit_id"])

    # Build held-out ISI hist per (uid, session) — load each session once.
    holdout: Dict[Tuple[int, str], np.ndarray] = {}
    for sess in sorted({s for (_, s) in kept_pairs}):
        pkl = _session_pkl(args.pkl_dir, sess)
        if pkl is None:
            continue
        S = load_session(str(pkl))
        cmap = {c.cluster_id: c for c in S.clusters}
        for (uid, s), kid in kept_pairs.items():
            if s != sess or kid not in cmap:
                continue
            _, hold = tc.partitioned_isi_hists(np.asarray(cmap[kid].spike_times))
            holdout[(uid, s)] = hold
        del S

    result = tc.held_out_isi_auc_by_tier(tracks, holdout)
    print(json.dumps(result, indent=2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "curation_validation.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {OUT_DIR / 'curation_validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the full pipeline end-to-end on real data**

```bash
.venv\Scripts\python.exe scripts/pipelines/tracking/make_state_tables.py --provider uniform
.venv\Scripts\python.exe scripts/pipelines/tracking/curate_tracks.py --min-span 2 --rebuild-cache
.venv\Scripts\python.exe scripts/pipelines/tracking/validate_curation.py
```

Expected: a per-tier AUC JSON. Sanity check: `trusted` AUC should exceed `suspect` AUC (held-out ISI corroborates the biophysically-clean tracks). Record the numbers.

- [ ] **Step 3: Run the full test suite (no regressions)**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: all prior tests still pass plus the new `test_state_provider.py` and `test_track_curation.py`.

- [ ] **Step 4: Update memory**

Update `C:\Users\Ben\.claude\projects\e--python-analysis-git-repos-vis-detect-analysis-Sep2025\memory\neuron_tracking_may2026.md` with: the curation pipeline shipped (modules + scripts), the bootstrap `uniform` state provider (swap for the real ethogram labeler later via the same CSV contract), and the end-to-end per-tier AUC numbers from Step 2. Add a one-line pointer in `MEMORY.md` if not already covered.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipelines/tracking/validate_curation.py
git commit -m "feat(curation): validate_curation CLI (held-out-ISI AUC by tier)"
```

---

## Self-Review

**1. Spec coverage:**
- Spec sec 1 (5 locked decisions): liberal re-key (Task 11 `--liberal-col`), never-reject/annotate (curate_registry emits derived tables, registry untouched), backward sweep (Task 7), state-conditioned (Tasks 1–2,5), held-out-ISI validation (Tasks 3,9,12). ✓
- Spec sec 3 inputs: registry/prob-matrix/raw-wf/pkls/manifest/state-tables all wired in Task 11; waveform restore noted in Step-2 prerequisite. ✓
- Spec sec 4 state interface (vocabulary, CSV contract, raw-index, HMM + ethogram providers): Tasks 1–2 (ethogram = future stub; `UniformInZoneStateProvider` bootstrap covers "swap later"). ✓
- Spec sec 5 features (biophysics, in-zone PSTH set, whole-trial waveforms): Task 5 + Task 4. ✓
- Spec sec 6 sweep + per-link rule + tier: Tasks 6,7. ✓
- Spec sec 7 outputs (curated_links.csv, curated_tracks.csv): Task 8 columns. ✓
- Spec sec 8.1/8.2 held-out ISI: Tasks 3,9,12. Sec 8.3 gold set: explicitly deferred (no task) — matches spec "future extension". ✓
- Spec sec 9 module layout, sec 10 params, sec 11 tests: Tasks define all four files, `CurationParams` carries every knob, synthetic tests cover the five spec scenarios. ✓

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to". Each code step is complete and runnable. The `UniformInZoneStateProvider` is an intentional, documented bootstrap (not a placeholder); `EthogramStateProvider` is correctly deferred to the spec's future section, not referenced by any task.

**3. Type consistency:** `CurationFeature` fields used in Tasks 5–9 match the dataclass in Task 5 (`waveform_peak`, `peak_depth_corrected_um`, `isi_hist_curation`, `inzone_psths`, `n_inzone_trials`). `score_link`/`sweep_uid`/`curate_registry`/`CurationParams`/`LinkResult`/`SweepResult` signatures consistent across tasks. `partitioned_isi_hists` (Task 3) reused in Tasks 5 & 12. `_pearson` (Task 6) reused in Task 9. `held_out_isi_auc_by_tier` keys (`auc`, `n_matched`, `n_nonmatched`) match the Task 9 test and Task 12 consumer.

---

## Execution notes / prerequisites

- **Restored inputs required for Tasks 10–12 real runs:** `data/pkls/BG_046/*.pkl` (done) and `data/unit_match/input/BG_046/{session}/…` raw waveforms (done — copied from X:). `FIGURES/` is recreated by the runners.
- **State tables are a prerequisite for curation** — run Task 10 before Task 11.
- **Branch:** all work on `main` per the single-workspace preference, unless a feature branch is chosen at handoff.
