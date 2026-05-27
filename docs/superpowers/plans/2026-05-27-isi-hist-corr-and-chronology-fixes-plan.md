# ISI Histogram Correlation Metric + Chronology / Stage Fixes Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `baseline_isi_hist_corr` metric + 5th composite badge (replacing `badge_isi_peak`), and fix three chronology/stage bugs (broken chronological sort, too-strict tracking-QC filter, inverted heatmap direction) plus add conditional panel legends.

**Architecture:** All changes target the existing tracking_qc pipeline. New library code + 1 new metric + extension of `session_outlier_flags`. CLI driver rewired to use a looser per-tracking filter (`apply_dynamic_filter(min_trials=150, min_dprime=None)`), normalize session-name strings via a single helper, swap the badge in composite verdict. Figures module flips heatmap origin, adds "Unknown" stage color, and gets five conditional panel legends.

**Tech Stack:** Python 3.10, numpy, pandas, matplotlib, pytest. Project's `py` launcher (Windows).

**Spec:** `docs/superpowers/specs/2026-05-27-isi-hist-corr-and-chronology-fixes-design.md` (commit `8d234d6`).

**Prerequisites:** Branch off `main` before starting:
```bash
git checkout -b feature/isi-hist-corr-and-chronology-fixes
```

---

## Task 1: ISI histogram correlation metric + badge

**Files:**
- Modify: `src/visdetect/analysis/tracking_qc.py` (add 2 constants, 1 metric, 1 badge)
- Modify: `tests/analysis/test_tracking_qc.py` (add 7 tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/analysis/test_tracking_qc.py`:

```python
def test_baseline_isi_hist_corr_identical_returns_one():
    h = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
    hists = [h.copy() for _ in range(5)]
    assert qc.baseline_isi_hist_corr(hists) == pytest.approx(1.0, abs=1e-6)


def test_baseline_isi_hist_corr_handles_magnitude_scaling():
    # Same shape, different magnitudes — Pearson r should still be 1.
    base = np.array([1.0, 3.0, 5.0, 3.0, 1.0])
    hists = [base, base * 2.0, base * 0.5]
    assert qc.baseline_isi_hist_corr(hists) == pytest.approx(1.0, abs=1e-6)


def test_baseline_isi_hist_corr_flipped_polarity_median():
    base = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    # pairs: (a, -a) = -1, (a, a) = +1, (-a, a) = -1 → median = -1
    hists = [base, -base, base]
    assert qc.baseline_isi_hist_corr(hists) == pytest.approx(-1.0, abs=1e-6)


def test_baseline_isi_hist_corr_drops_none_sessions():
    base = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    hists = [base, None, base, None, base]
    # 3 valid sessions, all identical → r = 1
    assert qc.baseline_isi_hist_corr(hists) == pytest.approx(1.0, abs=1e-6)


def test_baseline_isi_hist_corr_drops_flat_sessions():
    flat = np.zeros(5)
    base = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    # 2 valid (after dropping flat), both identical → r = 1
    assert qc.baseline_isi_hist_corr([base, flat, base]) == pytest.approx(1.0, abs=1e-6)


def test_baseline_isi_hist_corr_too_few_returns_nan():
    base = np.array([1.0, 2.0, 3.0])
    assert np.isnan(qc.baseline_isi_hist_corr([base]))
    assert np.isnan(qc.baseline_isi_hist_corr([base, None]))


def test_badge_isi_hist_corr_thresholds():
    assert qc.badge_isi_hist_corr(0.95) == "pass"
    assert qc.badge_isi_hist_corr(qc.ISI_HIST_CORR_PASS) == "pass"     # 0.85 boundary
    assert qc.badge_isi_hist_corr(0.75) == "warn"
    assert qc.badge_isi_hist_corr(qc.ISI_HIST_CORR_WARN) == "warn"     # 0.65 boundary
    assert qc.badge_isi_hist_corr(0.40) == "fail"
    assert qc.badge_isi_hist_corr(float("nan")) == "fail"
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
py -m pytest tests/analysis/test_tracking_qc.py -v 2>&1 | tail -20
```
Expected: 7 new tests fail with `AttributeError: module 'visdetect.analysis.tracking_qc' has no attribute 'baseline_isi_hist_corr'` (and similar).

- [ ] **Step 3: Add constants**

In `src/visdetect/analysis/tracking_qc.py`, near the other badge threshold constants (after the `FUNC_RESP_*` block, before `BIG_POOL`):

```python
# ISI histogram cross-session correlation (richer than badge_isi_peak which only
# looks at argmax bin). Captures full ISI distribution shape — handles bursting
# cells (with consistent bimodal ISIs) correctly. Calibrated to BG_046 cohort
# distribution (May 2026): gold-standard UIDs ~0.97-0.99, anti-drift suspect
# ~0.74, known matching-failures 0.58-0.61.
ISI_HIST_CORR_PASS: float = 0.85
ISI_HIST_CORR_WARN: float = 0.65
```

- [ ] **Step 4: Add metric function**

In `src/visdetect/analysis/tracking_qc.py`, near the other metric functions (e.g., after `baseline_psth_corr` and `isi_peak_agreement`):

```python
def baseline_isi_hist_corr(per_session_isi_hists: Sequence[np.ndarray]) -> float:
    """Median pairwise Pearson r of per-session log-ISI histograms.

    Captures full ISI distribution shape — handles bursting cells (with
    consistent bimodal ISIs) correctly, unlike isi_peak_agreement which looks
    only at the argmax bin. Architecturally mirrors waveform_corr.

    Parameters
    ----------
    per_session_isi_hists : sequence of (n_bins,) ndarrays, or None
        Per-session log-ISI histograms. None / NaN-only / flat (std < 1e-12)
        hists are dropped.

    Returns
    -------
    float
        Median over the n*(n-1)/2 pairwise Pearson r values. NaN if fewer than
        2 valid sessions remain after dropping.
    """
    arrs: List[np.ndarray] = []
    for h in per_session_isi_hists:
        if h is None:
            continue
        a = np.asarray(h, dtype=float)
        if a.size == 0 or np.all(np.isnan(a)) or float(np.std(a)) < 1e-12:
            continue
        arrs.append(a)
    if len(arrs) < 2:
        return float("nan")
    min_len = min(a.size for a in arrs)
    stack = np.stack([a[:min_len] for a in arrs])
    # Pearson r via mean-subtract + L2-normalize → pairwise dot products
    centered = stack - stack.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    unit = centered / norms
    n = unit.shape[0]
    pairs = [float(np.dot(unit[i], unit[j])) for i in range(n) for j in range(i + 1, n)]
    return float(np.median(pairs))
```

- [ ] **Step 5: Add badge function**

In `src/visdetect/analysis/tracking_qc.py`, near the other badge functions (e.g., after `badge_func_resp`):

```python
def badge_isi_hist_corr(r: float) -> str:
    """ISI histogram cross-session correlation badge.

    NaN → "fail" (standard pattern for ISI metrics; distinct from badge_func_resp
    which is lenient on NaN). NaN here means we couldn't compute the metric,
    which is itself a signal that something is wrong with the unit.
    """
    return _badge_threshold(r, ISI_HIST_CORR_PASS, ISI_HIST_CORR_WARN,
                            direction="high")
```

- [ ] **Step 6: Run tests, verify they pass**

```bash
py -m pytest tests/analysis/test_tracking_qc.py -v 2>&1 | tail -10
```
Expected: previous test count + 7 new = all passing.

- [ ] **Step 7: Commit**

```bash
git add src/visdetect/analysis/tracking_qc.py tests/analysis/test_tracking_qc.py
git commit -m "Add baseline_isi_hist_corr metric + badge for full ISI shape similarity"
```

---

## Task 2: Extend session_outlier_flags with unknown_stage dimension

**Files:**
- Modify: `src/visdetect/analysis/tracking_qc.py` (`session_outlier_flags` function)
- Modify: `tests/analysis/test_tracking_qc.py` (add 2 tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/analysis/test_tracking_qc.py`:

```python
def test_session_outlier_flags_unknown_stage_is_outlier():
    """A session with stage='Unknown' is unconditionally flagged outlier."""
    rec_good = qc.SessionRecord(
        session_name="s00", ks_unit_id=0, stage="Learning", peak_chan=10,
        peak_depth_um=100.0, amplitude=50.0, baseline_fr_hz=5.0,
        waveform_peak=np.array([0.0, -1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        footprint=np.zeros((5, 17), dtype=np.float32),
        footprint_channels=np.arange(17),
        isi_hist=np.array([0.1, 0.5, 0.3, 0.1] + [0.0] * 46, dtype=np.float32),
        isi_centers=np.zeros(50, dtype=np.float32),
    )
    rec_unknown = qc.SessionRecord(
        session_name="s01", ks_unit_id=0, stage="Unknown", peak_chan=10,
        peak_depth_um=100.0, amplitude=50.0, baseline_fr_hz=5.0,
        waveform_peak=np.array([0.0, -1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        footprint=np.zeros((5, 17), dtype=np.float32),
        footprint_channels=np.arange(17),
        isi_hist=np.array([0.1, 0.5, 0.3, 0.1] + [0.0] * 46, dtype=np.float32),
        isi_centers=np.zeros(50, dtype=np.float32),
    )
    uid = qc.UIDIntermediate(
        global_uid=1, span=2, has_naive_to_expert=False, suspect_known=False,
        sessions=[rec_good, rec_good, rec_unknown, rec_good],
    )
    flags = qc.session_outlier_flags(uid)
    assert "unknown_stage" in flags
    assert flags["unknown_stage"] == [False, False, True, False]
    assert flags["is_outlier"] == [False, False, True, False]


def test_find_stable_subset_drops_unknown_sessions():
    """Unknown-stage sessions break the kept run."""
    rec_good = qc.SessionRecord(
        session_name="s00", ks_unit_id=0, stage="Learning", peak_chan=10,
        peak_depth_um=100.0, amplitude=50.0, baseline_fr_hz=5.0,
        waveform_peak=np.array([0.0, -1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        footprint=np.zeros((5, 17), dtype=np.float32),
        footprint_channels=np.arange(17),
        isi_hist=np.array([0.1, 0.5, 0.3, 0.1] + [0.0] * 46, dtype=np.float32),
        isi_centers=np.zeros(50, dtype=np.float32),
    )
    rec_unknown = qc.SessionRecord(
        session_name="s01", ks_unit_id=0, stage="Unknown", peak_chan=10,
        peak_depth_um=100.0, amplitude=50.0, baseline_fr_hz=5.0,
        waveform_peak=np.array([0.0, -1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        footprint=np.zeros((5, 17), dtype=np.float32),
        footprint_channels=np.arange(17),
        isi_hist=np.array([0.1, 0.5, 0.3, 0.1] + [0.0] * 46, dtype=np.float32),
        isi_centers=np.zeros(50, dtype=np.float32),
    )
    # Sequence: good, good, unknown, good, good, good → kept = last 3.
    uid = qc.UIDIntermediate(
        global_uid=1, span=6, has_naive_to_expert=False, suspect_known=False,
        sessions=[rec_good, rec_good, rec_unknown, rec_good, rec_good, rec_good],
    )
    stable = qc.find_stable_subset(uid)
    assert stable["kept_indices"] == [3, 4, 5]
    assert 2 in stable["dropped_indices"]
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
py -m pytest tests/analysis/test_tracking_qc.py::test_session_outlier_flags_unknown_stage_is_outlier tests/analysis/test_tracking_qc.py::test_find_stable_subset_drops_unknown_sessions -v
```
Expected: 2 failures (key `unknown_stage` not in returned dict; sessions not flagged outlier).

- [ ] **Step 3: Extend `session_outlier_flags`**

In `src/visdetect/analysis/tracking_qc.py`, find the existing `session_outlier_flags` function. At the top of the function where the output dict is initialized, add the `unknown_stage` key:

```python
def session_outlier_flags(uid: "UIDIntermediate") -> Dict[str, List[bool]]:
    """..."""  # existing docstring
    n = len(uid.sessions)
    out = {
        "isi_peak":      [False] * n,
        "fr":            [False] * n,
        "wave":          [False] * n,
        "depth":         [False] * n,
        "unknown_stage": [False] * n,   # NEW
        "is_outlier":    [False] * n,
    }
    if n == 0:
        return out
    # ... existing isi_peak / fr / wave / depth computation ...
```

Then, near the existing per-session strikes computation, add the unknown_stage flag and update the composite rule:

```python
    # NEW: unknown_stage flag mirrors rec.stage == "Unknown"
    for i, rec in enumerate(uid.sessions):
        if rec.stage == "Unknown":
            out["unknown_stage"][i] = True

    # Composite outlier rule: any of (isi_peak alone) OR (>=2 other strikes) OR unknown_stage
    for i in range(n):
        strikes = sum([out["isi_peak"][i], out["fr"][i], out["wave"][i], out["depth"][i]])
        out["is_outlier"][i] = (
            out["isi_peak"][i]
            or strikes >= 2
            or out["unknown_stage"][i]   # NEW
        )

    return out
```

(If `session_outlier_flags` does not currently end with a composite-rule loop matching the structure above, follow the existing function's pattern but add `out["unknown_stage"][i]` as a third clause in the `is_outlier` expression.)

- [ ] **Step 4: Run tests, verify they pass**

```bash
py -m pytest tests/analysis/test_tracking_qc.py -v 2>&1 | tail -10
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/tracking_qc.py tests/analysis/test_tracking_qc.py
git commit -m "Extend session_outlier_flags with unknown_stage dimension"
```

---

## Task 3: Chronological sort fix + tracking-QC filter relaxation + Unknown stage default

**Files:**
- Modify: `scripts/pipelines/tracking/build_qc_sheets.py` (`build_cache` function and its top-level imports)

No new unit tests for this task — `build_cache` is integration-heavy; verified end-to-end in Task 7.

- [ ] **Step 1: Read current build_cache to understand structure**

```bash
py -c "
import inspect
import sys
sys.path.insert(0, 'scripts/pipelines/tracking')
import build_qc_sheets
src = inspect.getsource(build_qc_sheets.build_cache)
print(src[:2000])
"
```

Confirms the function defines `stage_by_session`, `sessions_chrono`, `sess_set`, and an end-of-function per-UID `sessions.sort(...)` call. Note line numbers (they may differ from this plan if prior edits shifted them).

- [ ] **Step 2: Update imports at top of build_qc_sheets.py**

Find the existing imports from `visdetect.suite.loader`:

```python
from visdetect.suite.loader import load_staging_manifest        # noqa: E402
```

Replace with:

```python
from visdetect.suite.loader import load_staging_manifest, apply_dynamic_filter  # noqa: E402
```

- [ ] **Step 3: Change the manifest loader call in main()**

Find the existing manifest load in `main()`:

```python
    manifest = load_staging_manifest(qc_only=True, apply_filter=True)
```

Replace with:

```python
    # Tracking-QC uses a looser filter than behavioral analyses: keeps engaged
    # sessions (>=150 trials) regardless of d', so early-Naive/Learning sessions
    # with poor performance but enough trials are still tracked across stages.
    # min_dprime=0.8 (the SDT default) wrongly excludes the very sessions needed
    # for cross-stage tracking studies. See spec §3.4.
    manifest = apply_dynamic_filter(min_trials=150, min_dprime=None)
```

- [ ] **Step 4: Add `_norm_session` helper inside build_cache**

At the top of `build_cache` (after the function signature and docstring, before any other logic), add:

```python
def _norm_session(name) -> str:
    """Normalize session name to 8-char zero-padded form.

    The manifest stores session_name as int (so '1072025' from astype(str), not
    '01072025'). The cache's SessionRecord.session_name is the raw 7/8-char
    filesystem string. Lookups need both sides in the same form.
    """
    return str(name).zfill(8)
```

- [ ] **Step 5: Use _norm_session in stage_by_session construction**

Find:

```python
    stage_by_session = {str(r["session_name"]): str(r["stage"])
                        for _, r in manifest.iterrows()}
```

Replace with:

```python
    stage_by_session = {_norm_session(r["session_name"]): str(r["stage"])
                        for _, r in manifest.iterrows()}
```

- [ ] **Step 6: Use _norm_session in sessions_chrono and sess_set sort**

Find:

```python
    sessions_chrono = manifest["session_name"].astype(str).tolist()
    sess_set = sorted({s for ksmap in uid_to_ks.values() for s in ksmap.keys()},
                      key=lambda s: sessions_chrono.index(s) if s in sessions_chrono else 1e9)
```

Replace with:

```python
    sessions_chrono = [_norm_session(n) for n in manifest["session_name"].tolist()]
    sess_set = sorted(
        {s for ksmap in uid_to_ks.values() for s in ksmap.keys()},
        key=lambda s: sessions_chrono.index(_norm_session(s)) if _norm_session(s) in sessions_chrono else 1e9,
    )
```

- [ ] **Step 7: Change stage default to "Unknown" + use _norm_session for lookup**

Find:

```python
        records = extract_session_records(
            S, ks_ids_here, session_name=sess,
            stage=stage_by_session.get(sess, "Learning"),
            raw_wf_root=RAW_WF_ROOT, channel_positions=chan_pos,
        )
```

Replace with:

```python
        records = extract_session_records(
            S, ks_ids_here, session_name=sess,
            stage=stage_by_session.get(_norm_session(sess), "Unknown"),
            raw_wf_root=RAW_WF_ROOT, channel_positions=chan_pos,
        )
```

- [ ] **Step 8: Use _norm_session in the final per-UID sort**

Find:

```python
    order_idx = {s: i for i, s in enumerate(sessions_chrono)}
    for uid in intermediates:
        intermediates[uid].sessions.sort(
            key=lambda r: order_idx.get(r.session_name, 1e9)
        )
```

Replace with:

```python
    order_idx = {s: i for i, s in enumerate(sessions_chrono)}
    for uid in intermediates:
        intermediates[uid].sessions.sort(
            key=lambda r: order_idx.get(_norm_session(r.session_name), 1e9)
        )
```

- [ ] **Step 9: Smoke check that the module still imports**

```bash
py -c "import sys; sys.path.insert(0, 'scripts/pipelines/tracking'); import build_qc_sheets; print('ok')"
```
Expected: `ok`.

- [ ] **Step 10: Commit**

```bash
git add scripts/pipelines/tracking/build_qc_sheets.py
git commit -m "Fix chronological sort + relax tracking-QC filter + stage='Unknown' default"
```

---

## Task 4: Wire isi_hist_corr into compute_uid_metrics + composite verdict + CSV

**Files:**
- Modify: `scripts/pipelines/tracking/build_qc_sheets.py` (top-level imports, `compute_uid_metrics`, both verdict computations, both CSV row dicts)

No new unit tests — wiring is verified by Task 7 end-to-end smoke.

- [ ] **Step 1: Add the new symbol to the tracking_qc import block**

Find the existing import block from `visdetect.analysis.tracking_qc` and add `baseline_isi_hist_corr`, `badge_isi_hist_corr`, `ISI_HIST_CORR_PASS`, `ISI_HIST_CORR_WARN` to the imported names. Concretely, where you see something like:

```python
from visdetect.analysis.tracking_qc import (        # noqa: E402
    UIDIntermediate, SessionRecord,
    select_long_tracks, annotate_naive_to_expert,
    extract_session_records, load_channel_positions,
    load_isi_scores, load_um_pair_scores,
    depth_std_um, waveform_corr, fr_cv,
    isi_peak_agreement, baseline_psth_corr,
    badge_isi, badge_depth, badge_waveform, badge_fr,
    badge_isi_peak, badge_func_resp, composite_verdict,
    estimate_session_drift, depth_std_um_corrected,
    save_cache, load_cache,
    find_stable_subset,
)
```

Add `baseline_isi_hist_corr` to the metric-function imports and `badge_isi_hist_corr` to the badge-function imports:

```python
from visdetect.analysis.tracking_qc import (        # noqa: E402
    UIDIntermediate, SessionRecord,
    select_long_tracks, annotate_naive_to_expert,
    extract_session_records, load_channel_positions,
    load_isi_scores, load_um_pair_scores,
    depth_std_um, waveform_corr, fr_cv,
    isi_peak_agreement, baseline_psth_corr, baseline_isi_hist_corr,
    badge_isi, badge_depth, badge_waveform, badge_fr,
    badge_isi_peak, badge_func_resp, badge_isi_hist_corr, composite_verdict,
    estimate_session_drift, depth_std_um_corrected,
    save_cache, load_cache,
    find_stable_subset,
)
```

- [ ] **Step 2: Extend `compute_uid_metrics` to return `isi_hist_corr`**

Find the `compute_uid_metrics` function. Add the new metric computation alongside the existing ones. After the line that builds the `isi_hists` list:

```python
    isi_hists = [r.isi_hist for r in uid.sessions]
```

Update the returned dict to include the new key:

```python
    out = {
        "depth_std_um":     depth_std_um(depths),
        "wave_corr":        waveform_corr(wf_stack),
        "fr_cv":            fr_cv(rates),
        "isi_peak_agree":   isi_peak_agreement(isi_hists),
        "func_resp_corr":   baseline_psth_corr(baseline_psths),
        "isi_hist_corr":    baseline_isi_hist_corr(isi_hists),   # NEW
    }
```

If the function continues to add `depth_std_corrected_um`, leave that logic intact.

- [ ] **Step 3: Swap badge in the trimmed-verdict computation**

Find the trimmed-verdict composite (in the `uid_trim_info` build loop):

```python
            tv = composite_verdict([
                badge_isi(isi_scores[uid]),
                badge_depth(_depth_for_badge(tm)),
                badge_waveform(tm["wave_corr"]),
                badge_fr(tm["fr_cv"]),
                badge_isi_peak(tm["isi_peak_agree"]),
                badge_func_resp(tm["func_resp_corr"]),
            ])
```

Replace `badge_isi_peak(tm["isi_peak_agree"])` with `badge_isi_hist_corr(tm["isi_hist_corr"])`:

```python
            tv = composite_verdict([
                badge_isi(isi_scores[uid]),
                badge_depth(_depth_for_badge(tm)),
                badge_waveform(tm["wave_corr"]),
                badge_fr(tm["fr_cv"]),
                badge_isi_hist_corr(tm["isi_hist_corr"]),   # was badge_isi_peak(tm["isi_peak_agree"])
                badge_func_resp(tm["func_resp_corr"]),
            ])
```

- [ ] **Step 4: Swap badge in the main render-loop verdict computation**

Find the main composite:

```python
        b_isi   = badge_isi(isi)
        b_depth = badge_depth(_depth_for_badge(metrics))
        b_wave  = badge_waveform(metrics["wave_corr"])
        b_fr    = badge_fr(metrics["fr_cv"])
        b_peak  = badge_isi_peak(metrics["isi_peak_agree"])
        b_func  = badge_func_resp(metrics["func_resp_corr"])
        verdict_csv = composite_verdict([b_isi, b_depth, b_wave, b_fr, b_peak, b_func])
```

Replace `b_peak` with `b_hist`:

```python
        b_isi   = badge_isi(isi)
        b_depth = badge_depth(_depth_for_badge(metrics))
        b_wave  = badge_waveform(metrics["wave_corr"])
        b_fr    = badge_fr(metrics["fr_cv"])
        b_peak  = badge_isi_peak(metrics["isi_peak_agree"])     # kept for CSV transparency only
        b_func  = badge_func_resp(metrics["func_resp_corr"])
        b_hist  = badge_isi_hist_corr(metrics["isi_hist_corr"])   # NEW
        verdict_csv = composite_verdict([b_isi, b_depth, b_wave, b_fr, b_hist, b_func])
```

- [ ] **Step 5: Add new columns to the verdicts.csv row dict**

Find the `rows.append({...})` block for verdicts.csv. Add the new columns:

```python
        rows.append({
            # ...existing fields...
            "isi_peak_agree":   metrics["isi_peak_agree"],
            "isi_hist_corr":    metrics["isi_hist_corr"],          # NEW
            "func_resp_corr":   metrics["func_resp_corr"],
            "badge_isi":        b_isi,
            "badge_depth":      b_depth,
            "badge_wave":       b_wave,
            "badge_fr":         b_fr,
            "badge_isi_peak":   b_peak,
            "badge_isi_hist_corr": b_hist,                         # NEW
            "badge_func_resp":  b_func,
            # ...existing verdict/verdict_pdf/pdf_csv_disagree fields...
        })
```

- [ ] **Step 6: Add new columns to verdicts_trimmed.csv row dict**

Find the `trimmed_rows.append({...})` block. Add corresponding columns from the trimmed metrics `tm`:

```python
        trimmed_rows.append({
            # ...existing fields...
            "trimmed_isi_peak_agree":   tm["isi_peak_agree"],
            "trimmed_isi_hist_corr":    tm["isi_hist_corr"],       # NEW
            "trimmed_func_resp_corr":   tm["func_resp_corr"],
            # ...existing trimmed_verdict / rescued fields...
        })
```

(If the existing `trimmed_rows.append` does not currently include `trimmed_isi_peak_agree` / `trimmed_func_resp_corr`, add `trimmed_isi_hist_corr` next to whichever columns DO exist, matching the established naming pattern.)

- [ ] **Step 7: Smoke check that the module still imports + runs `--help`**

```bash
py scripts/pipelines/tracking/build_qc_sheets.py --help 2>&1 | head -5
```
Expected: argparse help block, no exception.

- [ ] **Step 8: Commit**

```bash
git add scripts/pipelines/tracking/build_qc_sheets.py
git commit -m "Wire isi_hist_corr: into compute_uid_metrics, both composite verdicts, both CSVs"
```

---

## Task 5: Heatmap origin flip + STAGE_COLORS Unknown + trim marker adjustment

**Files:**
- Modify: `scripts/pipelines/tracking/qc_sheet_figures.py` (`_draw_heatmap`, the local stage-color map)

No new unit tests — visual change verified by Task 7 smoke render.

- [ ] **Step 1: Extend the local STAGE_COLORS map with "Unknown"**

In `scripts/pipelines/tracking/qc_sheet_figures.py`, find the local `STAGE_COLORS` extension (or the imported `STAGE_COLORS` use in `draw_header` and `_waveform_color`). If there is a local `STAGE_COLORS_EXTENDED` or similar dict, add `"Unknown": "#bbbbbb"`. If the file uses the imported `STAGE_COLORS` directly (which lives in `visdetect/suite/config.py` and may not have "Unknown"), add a small local extension dict at module level:

```python
# Local extension: tracking_qc adds "Unknown" stage (sessions not in the
# tracking-QC filter, see spec §3.4). Light grey distinguishes from the
# dimmed-trace grey (0.7) used for trimmed-but-not-Unknown sessions.
STAGE_COLORS_LOCAL = {**STAGE_COLORS, "Unknown": "#bbbbbb"}
```

Then update `_waveform_color` and every `STAGE_COLORS[...]` access in the file to use `STAGE_COLORS_LOCAL[...]` instead.

- [ ] **Step 2: Flip heatmap origin in `_draw_heatmap`**

Find the existing `imshow` call in `_draw_heatmap`:

```python
    ax_main.imshow(mat, aspect="auto", origin="lower", cmap="magma",
                   extent=[centers[0], centers[-1], 0, mat.shape[0]],
                   vmin=0, vmax=max(vmax, 1e-6))
```

Replace with:

```python
    ax_main.imshow(mat, aspect="auto", origin="upper", cmap="magma",
                   extent=[centers[0], centers[-1], mat.shape[0], 0],
                   vmin=0, vmax=max(vmax, 1e-6))
```

(`origin='upper'` plus the flipped y-extent ensures earliest session = row 0 = top of panel.)

- [ ] **Step 3: Update the trim-marker rectangle anchoring**

Find the existing red-rectangle drawing for dropped rows in `_draw_heatmap`. It currently anchors each marker at `(x0_pad, i)` with `height=1`:

```python
        for i in dropped_row_indices:
            ax_main.add_patch(Rectangle((centers[0] - small_pad, i),
                                        small_pad, 1,
                                        facecolor="red", edgecolor="none",
                                        clip_on=False))
```

With `origin='upper'` and the swapped extent, the data coords for row `i` still map to y=i — so the rectangle anchor of `(x, i)` with height `1` still lands on the right row (it just renders top-to-bottom rather than bottom-to-top). No change needed if the code matches the snippet above. Verify by inspection and adjust only if the existing code does `mat.shape[0] - 1 - i` (which it shouldn't with the original `origin='lower'` setup, but check).

- [ ] **Step 4: Smoke render UID 942**

(Cache is from prior runs; OK to reuse for this visual check — we'll rebuild in Task 7.)

```bash
py scripts/pipelines/tracking/build_qc_sheets.py --uids 942 2>&1 | tail -3
```
Expected: `uid 942: csv=trusted pdf=trusted` (or similar — verdict may shift slightly since `badge_isi_peak` was swapped to `badge_isi_hist_corr`).

Open `FIGURES/tracking_qc/per_uid_sheets/uid_0942.pdf` (or read it via the Read tool) and verify:
- Page-2 heatmaps: row 0 (earliest session) at TOP of y-axis; y-axis labels count from `mat.shape[0]` at the top down to `0` at the bottom.
- Red trim markers (if any) still align with the correct dropped row.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipelines/tracking/qc_sheet_figures.py
git commit -m "Flip heatmap origin to put earliest session at top; add Unknown stage color"
```

---

## Task 6: Conditional panel legends

**Files:**
- Modify: `scripts/pipelines/tracking/qc_sheet_figures.py` (`draw_header`, `render_page1`, `_draw_heatmap`, `_draw_psth_summary`)

No new unit tests — visual change verified by Task 7 smoke render.

- [ ] **Step 1: Add stage-stripe legend below trim annotation in `draw_header`**

Find `draw_header` and the existing trim annotation text (added in prior trim-visualization work). After the trim text — or after the stage stripe if no trim annotation is present — add a small legend annotation:

```python
    # Stage-stripe legend (always shown, regardless of trim state)
    ax.text(
        0.0, -0.04,
        "stripe: Learning · Expert · Unknown · /// = trimmed",
        transform=ax.transAxes, fontsize=8, color="0.4",
        ha="left", va="top",
    )
```

If the existing axes are clipped tightly to the header, you may need to extend `ax.set_ylim(...)` slightly downward or use a negative y in `transAxes` as above with `va="top"`.

- [ ] **Step 2: Add "grey traces = dropped" legend on the ISI distribution panel**

In `render_page1`, in the block that renders the ISI distribution overlay, after the loop that plots each session's hist, conditionally add the legend:

```python
    # ISI distribution overlay (existing block)
    for rec_idx, rec in enumerate(uid.sessions):
        color = "0.7" if rec_idx in dropped_set else _waveform_color(rec.stage)
        lw = 0.5 if rec_idx in dropped_set else 0.7
        ax_isi.semilogx(rec.isi_centers, rec.isi_hist,
                        color=color, linewidth=lw, alpha=0.6)
    ax_isi.set_xlabel("ISI (s, log)"); ax_isi.set_ylabel("prob")
    ax_isi.set_title("ISI distribution", fontsize=10)

    if dropped_set:
        ax_isi.text(0.98, 0.97, "grey traces = dropped",
                    transform=ax_isi.transAxes, fontsize=7, color="0.4",
                    ha="right", va="top")
```

(`dropped_set` should already exist in the function as `set(dropped_indices or [])`. If not, add `dropped_set = set(dropped_indices or [])` near the top of `render_page1`.)

- [ ] **Step 3: Add "○ = dropped" legend on the Depth panel**

In `render_page1`, after the existing Depth-on-probe scatter+line block (the FIRST of the three Depth/Amp/Baseline-FR scatter panels), add:

```python
    if dropped_set:
        ax_depth.text(0.98, 0.97, "○ = dropped",
                      transform=ax_depth.transAxes, fontsize=7, color="0.4",
                      ha="right", va="top")
```

(Only on the Depth panel — to avoid 3× repetition of the same legend on Depth + Amplitude + Baseline FR.)

- [ ] **Step 4: Add "red bar = dropped row" legend on heatmaps**

In `_draw_heatmap`, after the trim-marker rendering, conditionally add a legend in the top-left corner. Use white text since the heatmap is dark (`cmap='magma'` saturates dark at low values):

```python
    if dropped_row_indices:
        ax_main.text(0.02, 0.97, "red bar = dropped row",
                     transform=ax_main.transAxes, fontsize=7, color="white",
                     ha="left", va="top",
                     bbox=dict(facecolor="0.1", edgecolor="none", alpha=0.5, pad=2))
```

- [ ] **Step 5: Extend the PSTH-summary legend for hit/miss when miss_keys present**

In `_draw_psth_summary` (or whatever the page-2 right-column renderer is named — same module), find the line where the L/E lines are plotted. Currently the solid lines get `label=st` (i.e., "Learning", "Expert"). For panels with miss_keys, the dashed lines have no label.

Update so that when `miss_keys` is present, the solid lines label as `f"{st} hit"` and the dashed lines label as `f"{st} miss"`. When `miss_keys` is absent, leave the solid labels as `st`.

Concretely, in the existing structure:

```python
def _draw_psth_summary(ax, uid, key, miss_keys=None):
    # ... fetch primary data, set up ax ...
    for st in STAGE_ORDER:
        mask = np.array([s == st for s in stages])
        if mask.sum() == 0:
            continue
        label_solid = f"{st} hit" if miss_keys else st
        ax.plot(centers, mat[mask].mean(axis=0), color=STAGE_COLORS_LOCAL[st],
                linewidth=1.2, label=label_solid)
    if miss_keys:
        for mk in miss_keys:
            # ... existing fetch ...
            for st in STAGE_ORDER:
                # ... existing mask + check ...
                ax.plot(mcenters, mmat[mask].mean(axis=0),
                        color=STAGE_COLORS_LOCAL[st], linewidth=1.2,
                        linestyle="--", alpha=0.8,
                        label=f"{st} miss")
    if ax.has_data():
        # Reduced fontsize when miss_keys is on (4 entries vs 2)
        ax.legend(loc="upper right", fontsize=6 if miss_keys else 7,
                  frameon=False)
```

(If the existing `_draw_psth_summary` already has a `legend(...)` call, just update its fontsize logic and update the label arguments per the snippet above.)

- [ ] **Step 6: Smoke render**

```bash
py scripts/pipelines/tracking/build_qc_sheets.py --uids 942 334 1207 2>&1 | tail -5
```
Read `FIGURES/tracking_qc/per_uid_sheets/uid_0942.pdf`, `uid_0334.pdf`, `uid_1207.pdf`. Check:
- UID 942 page 1: stage stripe legend visible below header; depth panel has "○ = dropped" annotation only if there are any dropped sessions; ISI distribution has "grey traces = dropped" only if dropped present.
- UID 334 page 1: same plus visible dropped annotations (UID 334 has many dropped sessions).
- UID 1207 page 1: NO dropped annotations (UID 1207 has all sessions kept).
- All page-2 Change_ON panels: legend shows "L hit · L miss · E hit · E miss" (or equivalent 4-entry layout).

- [ ] **Step 7: Commit**

```bash
git add scripts/pipelines/tracking/qc_sheet_figures.py
git commit -m "Add conditional panel legends + extend PSTH summary legend with hit/miss"
```

---

## Task 7: End-to-end smoke test, cache rebuild, cohort shift report

**Files:** none modified — verification + commit message.

- [ ] **Step 1: Rebuild cache (required — looser filter changes session set)**

```bash
py scripts/pipelines/tracking/build_qc_sheets.py --rebuild-cache 2>&1 | tail -10
```

Expected: takes ~5–10 min. Output ends with `Wrote ... verdicts.csv` and `Wrote ... verdicts_trimmed.csv (N rescued)`. Cohort size printed at the top may differ from the prior 61 (looser filter may include or exclude UIDs based on whether their span criteria still satisfy `min_span=10` against the new manifest's session set).

- [ ] **Step 2: Summarize verdict distribution shift**

```bash
py -c "
import pandas as pd
d = pd.read_csv('FIGURES/tracking_qc/verdicts.csv')
t = pd.read_csv('FIGURES/tracking_qc/verdicts_trimmed.csv')
print('verdicts.csv:'); print(d['verdict'].value_counts())
print()
print('verdicts_trimmed.csv:'); print(t['trimmed_verdict'].value_counts())
print(f'Rescued: {t.rescued.sum()}')
print()
print('Cohort size:', len(d))
print()
print('New columns present:', 'isi_hist_corr' in d.columns, 'badge_isi_hist_corr' in d.columns)
print()
anchors = [177, 334, 511, 600, 942, 1207, 779, 872, 873]
print(d[d.global_uid.isin(anchors)][['global_uid','isi_hist_corr','badge_isi_hist_corr','verdict']].to_string(index=False))
"
```

Expected:
- `isi_hist_corr` and `badge_isi_hist_corr` columns present in `verdicts.csv`.
- UID 942 verdict = `trusted` (unchanged — gold standard not regressed).
- UID 779 / 872 `badge_isi_hist_corr` = `fail`.
- UID 334 `badge_isi_hist_corr` = `pass` (0.945 well above 0.85).

- [ ] **Step 3: Visual checks on UID 942 + UID 334 + a "missing-manifest" example**

Read `FIGURES/tracking_qc/per_uid_sheets/uid_0942.pdf`, `uid_0334.pdf`, and check:

- **UID 942 (gold standard, all sessions kept):**
  - Page 1: stage stripe shows light-green Learning + dark-green Expert in chronological order (no interleaving).
  - Page 1: no dropped-session annotations (no "○ = dropped", no "grey traces").
  - Page 2: heatmaps have earliest session (row 0) at TOP of y-axis, descending.
  - Page-2 Change_ON panels: 4-entry legend "L hit · L miss · E hit · E miss".

- **UID 334 (heavily trimmed):**
  - Page 1: stage stripe shows hatched cells for dropped sessions (and grey cells if any are Unknown stage).
  - Page 1: "grey traces = dropped" annotation visible on ISI distribution panel.
  - Page 1: "○ = dropped" annotation visible on Depth panel.
  - Page 1: dropped-session scatter dots are open circles; kept-session dots are filled.
  - Page 2: heatmaps have red bars on left edge of dropped rows; "red bar = dropped row" legend visible.

- **A UID with manifest-missing sessions:** find one via the CSV:

```bash
py -c "
import pickle
with open('data/cache/tracking_qc_intermediates.pkl','rb') as f:
    inter = pickle.load(f)
for uid, iv in inter.items():
    unk = [r.session_name for r in iv.sessions if r.stage == 'Unknown']
    if len(unk) >= 1:
        print(f'UID {uid}: {len(unk)} Unknown sessions: {unk[:5]}')
        break
"
```

Read that UID's PDF. Confirm the Unknown-stage cells in the stage stripe appear GREY (not light/dark green), and those sessions appear as outliers (dropped) in the trim visualization.

- [ ] **Step 4: Verify chronological sort is correct**

```bash
py -c "
import pickle
from datetime import datetime
with open('data/cache/tracking_qc_intermediates.pkl','rb') as f:
    inter = pickle.load(f)
uid = inter[942]
parsed = [datetime.strptime(str(r.session_name).zfill(8), '%d%m%Y') for r in uid.sessions]
is_sorted = all(parsed[i] <= parsed[i+1] for i in range(len(parsed)-1))
print('UID 942 sessions in chronological order:', is_sorted)
if not is_sorted:
    for i, (rec, dt) in enumerate(zip(uid.sessions, parsed)):
        print(f'  row {i}: {rec.session_name} ({dt.date()}) stage={rec.stage}')
"
```

Expected: `UID 942 sessions in chronological order: True`.

- [ ] **Step 5: Re-run pytest to confirm nothing regressed**

```bash
py -m pytest tests/analysis/test_tracking_qc.py -q 2>&1 | tail -3
```
Expected: all tests pass (prior count + 9 added = total).

- [ ] **Step 6: Final commit (cohort shift summary)**

```bash
git commit --allow-empty -m "$(cat <<'EOF'
End-to-end smoke for ISI hist-corr + chronology fixes

Cohort size: <FILL IN>  (was 61 prior to filter relaxation)
verdicts.csv: <trusted>/<review>/<suspect>
verdicts_trimmed.csv: <trusted>/<review>/<suspect>  (<N> rescued)
UID 942 (gold): trusted (unchanged)
UID 779/872 (known suspects): badge_isi_hist_corr=fail (correctly caught)
Chronological sort verified correct.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Fill in the `<...>` from Step 2 output. This is an empty-tree commit that anchors the cohort numbers in git history.

---

## Self-review (post-write)

Spec → plan coverage:

| Spec section | Plan task |
|---|---|
| §3.1 baseline_isi_hist_corr metric | Task 1 |
| §3.2 composite verdict integration + CSV columns | Task 4 |
| §3.3 chronological sort fix | Task 3 (steps 4, 6, 8) |
| §3.4 tracking-QC filter relaxation + Unknown stage | Task 3 (steps 2-3, 5, 7); Task 5 (step 1); Task 2 (outlier behavior) |
| §3.5 heatmap origin flip | Task 5 (steps 2-3) |
| §3.6 panel legends (5 sub-items) | Task 6 (steps 1-5) |
| §6 testing | Tasks 1, 2 unit tests; Task 7 smoke |

All spec sections covered. No placeholders. Names consistent (`baseline_isi_hist_corr`, `badge_isi_hist_corr`, `ISI_HIST_CORR_PASS`, `ISI_HIST_CORR_WARN`, `STAGE_COLORS_LOCAL`, `_norm_session`) across tasks.
