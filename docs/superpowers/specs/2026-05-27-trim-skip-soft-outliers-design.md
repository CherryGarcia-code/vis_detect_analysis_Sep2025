# Skip-able Trimming for find_stable_subset

**Date:** 2026-05-27
**Owner:** Ben (UCL)
**Status:** Design

## 1. Purpose

Let `find_stable_subset` keep non-contiguous good sessions across soft outlier gaps when ISI fingerprint consistency confirms cell identity holds across the gap. Recovers UIDs whose Learning sessions are separated from their Expert sessions by interleaved Unknown-stage (or other soft) sessions, restoring cross-stage coverage for learning-trajectory analyses.

## 2. Background

The previous spec (`2026-05-27-isi-hist-corr-and-chronology-fixes`) fixed three chronology bugs and added the `baseline_isi_hist_corr` metric. End-to-end smoke on the trimmed cohort revealed a striking coverage problem:

| coverage requirement | usable UIDs (trusted + review) |
|---|---|
| ≥1 Learning AND ≥1 Expert | **1** out of 20 |
| ≥2 Learning AND ≥2 Expert | 0 |
| UIDs with zero Learning kept | 16 |
| UIDs with zero Expert kept | 3 |

UID 942 is the canonical example: 14 sessions chronologically (2 Learning + 8 Expert + 3 Unknown + 1 trimmed-by-other-metric). The Unknown sessions (13082025, 18082025, 26082025) interleave between Learning (rows 0-1, 4) and Expert (rows 6-13). The current `longest_good_run` picks the all-Expert tail (4 rows) and discards the Learning rows. The cell IS tracked across both stages by UnitMatch — it's being lost in the post-hoc trim, not in the matching.

`baseline_isi_hist_corr` on UID 942's full cohort = 0.99 (well above the 0.85 PASS threshold), strong evidence the cell identity holds across the Unknown gaps. The current trim algorithm has no way to use this evidence.

## 3. Components

### 3.1 — Soft/hard outlier classification

**Location:** `src/visdetect/analysis/tracking_qc.py`, `session_outlier_flags`.

Extend the returned dict with two new boolean lists alongside the existing `is_outlier`:

```python
out["is_hard_outlier"][i] = out["wave"][i] or out["depth"][i]
out["is_soft_outlier"][i] = out["is_outlier"][i] and not out["is_hard_outlier"][i]
```

Classification rationale:
- **Hard** = {`wave`, `depth`}: a waveform-shape mismatch or large depth jump suggests a physically different unit at that probe position. Concatenating across such a gap risks mixing two cells under one UID.
- **Soft** = {`unknown_stage`, `fr`, `isi_peak`}: data-quality issues (`unknown_stage` = <150 trials; `fr` = transient quiet-period; `isi_peak` = argmax flip on bursting cells, which the full-shape `isi_hist_corr` handles correctly). Cell identity may be intact across these.

The existing `is_outlier` flag continues to combine all five types — downstream code that doesn't care about the soft/hard distinction sees no change.

### 3.2 — New `longest_good_run` algorithm

**Location:** `src/visdetect/analysis/tracking_qc.py`, `longest_good_run`.

Replace the current contiguous-slice algorithm with:

1. Identify all maximal contiguous spans containing NO hard outliers.
2. For each candidate span, form `kept_set` = sessions in the span that are NOT outliers of any kind (soft or hard).
3. Compute `set_isi_hist_corr` = `baseline_isi_hist_corr(isi_hists of kept_set)`.
4. If `set_isi_hist_corr >= ISI_HIST_CORR_PASS` (0.85), the span yields a valid (kept_set, skipped_set) where `skipped_set` = soft outliers inside the span.
5. Among spans that pass the gate, pick the one with the **largest kept_set**. Ties broken by total span length, then by earliest start.
6. If NO span's kept_set passes the gate, fall back to the current behavior: longest contiguous all-good run (no skipping). This means a UID never gets WORSE under the new algorithm than the old.

Algorithm complexity is O(n²) in the worst case (computing `isi_hist_corr` is O(k²) for k kept sessions; spans are at most n). For typical n ≤ 30 sessions per UID this is negligible.

### 3.3 — `find_stable_subset` return-shape changes

**Location:** `src/visdetect/analysis/tracking_qc.py`, `find_stable_subset`.

Existing return shape:

```python
{
    "kept_indices": [int, ...],
    "dropped_indices": [int, ...],
    "trimmed_verdict": str,
    # (plus existing fields used by qc_sheet_figures and verdicts_trimmed.csv)
}
```

New shape — one new key, one redefined meaning:

```python
{
    "kept_indices": [int, ...],          # GOOD sessions in the kept span
    "skipped_indices": [int, ...],       # NEW: soft outliers INSIDE the kept span
    "dropped_indices": [int, ...],       # REDEFINED: sessions OUTSIDE the kept span,
                                          # plus hard outliers anywhere (i.e., not in kept_set
                                          # AND not in skipped_set)
    "trimmed_verdict": str,
    # ...other fields unchanged
}
```

Invariants:
- `set(kept_indices) ∪ set(skipped_indices) ∪ set(dropped_indices) == set(range(len(uid.sessions)))`
- The three sets are pairwise disjoint
- When the consistency gate fails and we fall back: `skipped_indices == []` and the result equals the old behavior

The redefinition of `dropped_indices` means it no longer fully covers "everything not in the kept set" — it now excludes the in-span soft outliers. Renderers that hatch `dropped_indices` would therefore leave skipped sessions un-hatched, which is wrong: skipped sessions should look identical to dropped per Component 3.5. The fix is at the call site (Component 3.4) — `build_qc_sheets.py` computes `visually_dropped = dropped_indices ∪ skipped_indices` and passes that single union to `write_uid_pdf`. Renderer function signatures don't change.

### 3.4 — CSV column changes

**Location:** `scripts/pipelines/tracking/build_qc_sheets.py`, `verdicts_trimmed.csv` row build.

Add ONE new column: `skipped_sessions` — a semicolon list of session_names for sessions in `skipped_indices`. Symmetric with the existing `dropped_sessions` and `kept_sessions` columns.

All other columns unchanged. No `n_skipped` count column, no `rescued_by_skip` flag — by design, the goal is the minimum CSV surface needed for audit.

### 3.5 — PDF rendering (no visual change)

**Location:** `scripts/pipelines/tracking/qc_sheet_figures.py` (no changes).

Skipped sessions render IDENTICALLY to dropped sessions:
- Stage stripe: diagonal hatch overlay
- Scatter panels (Depth, Amplitude, Baseline FR): open circle
- ISI overlay: dimmed grey trace
- Heatmap: red bar on row left edge

The existing `dropped_indices` plumbing in `qc_sheet_figures` continues to work — it receives the redefined `dropped_indices` (sessions outside the kept span). To render skipped sessions with the same dim-style, the rendering code receives `dropped_indices ∪ skipped_indices` as the "visually dropped" set. This is the only rendering-side change: a single union operation at the call site.

The PDF header trim annotation continues to show `kept N/M` — the larger M-N gap now naturally reflects the skip rescue (e.g., UID 942 currently says "kept 4/14"; post-skip it would say "kept 11/14").

The audit trail for "which dropped sessions were specifically skipped vs trim-edge-dropped" lives in the CSV `skipped_sessions` column.

## 4. Data flow

```
session_outlier_flags(uid)
   │
   ├── is_outlier (existing)
   ├── is_hard_outlier (NEW: wave or depth)
   └── is_soft_outlier (NEW: is_outlier and not hard)
   │
   ▼
longest_good_run(flags, isi_hists)
   │
   ├── Try each contiguous no-hard span
   ├── Compute set-wide isi_hist_corr on candidate kept_set
   ├── If gate passes: return (kept_set, skipped_set)
   └── Else: fall back to old longest-contiguous-all-good
   │
   ▼
find_stable_subset(uid)
   │
   └── Returns {kept_indices, skipped_indices, dropped_indices, trimmed_verdict, ...}
   │
   ▼
build_qc_sheets:
   ├── verdicts_trimmed.csv gets skipped_sessions column
   └── qc_sheet_figures receives (dropped ∪ skipped) as "visually dropped"
```

## 5. File-level changes

| File | Change |
|---|---|
| `src/visdetect/analysis/tracking_qc.py` | (a) `session_outlier_flags`: add `is_hard_outlier` and `is_soft_outlier` keys. (b) `longest_good_run`: new algorithm (skip-able with set-wide ISI gate + fallback). (c) `find_stable_subset`: extend return dict with `skipped_indices`; redefine `dropped_indices` to exclude skipped. |
| `tests/analysis/test_tracking_qc.py` | Add ~5 unit tests covering soft/hard classification, gate-pass skipping, gate-fail fallback, hard-outlier always-breaks, and `find_stable_subset` return shape. |
| `scripts/pipelines/tracking/build_qc_sheets.py` | Add `skipped_sessions` column to `verdicts_trimmed.csv` row dict. Pass `dropped ∪ skipped` to `qc_sheet_figures.write_uid_pdf` as the "visually dropped" set. |
| `scripts/pipelines/tracking/qc_sheet_figures.py` | No structural changes. The render functions already accept a single "dropped" set; the union happens at the call site in `build_qc_sheets.py`. |
| `docs/superpowers/specs/2026-05-27-trim-skip-soft-outliers-design.md` | This spec, committed. |

## 6. Testing

New unit tests in `tests/analysis/test_tracking_qc.py`:

- `test_session_outlier_flags_classifies_hard_vs_soft` — verifies the two new keys for a UID with one wave-flagged and one fr-flagged session.
- `test_longest_good_run_skips_soft_with_high_consistency` — sequence `[G, G, S, G, G]` with identical ISI hists → kept = `[0,1,3,4]`, skipped = `[2]`, set-wide corr = 1.0 passes gate.
- `test_longest_good_run_falls_back_when_consistency_fails` — sequence `[G, G, S, G, G]` with deliberately divergent ISI hists on the two halves → gate fails → falls back to longest contiguous good = either `[0,1]` or `[3,4]`, skipped = `[]`.
- `test_longest_good_run_never_skips_hard_outliers` — sequence `[G, G, H, G, G]` where H is a hard outlier → result is one of `[0,1]` or `[3,4]`; H is never in kept or skipped (always dropped).
- `test_find_stable_subset_returns_skipped_indices` — full integration: 14-session fixture mimicking UID 942's shape → kept ≥ 10, skipped includes the soft-outlier rows, dropped includes any hard outliers and the trim edges.

End-to-end smoke: rebuild cache + verdicts; confirm:
- Cohort size unchanged (61).
- `skipped_sessions` column present in `verdicts_trimmed.csv`.
- UID 942 trimmed: kept count rises from 4 to ~11; verdict stays trusted; `skipped_sessions` lists the three Unknown dates.
- Cohort cross-stage coverage (≥1 Learning AND ≥1 Expert) rises from 1 to at least several UIDs.
- No UID's trimmed verdict regresses to worse than before (the fallback guarantees this).
- All 68 existing tests + ~5 new tests pass.

## 7. Non-goals

The following are explicitly out of scope for this spec and will be addressed separately:

- **Distinct PDF visual marker for skipped vs dropped sessions** — the CSV column is the audit trail. Adding a dotted-vs-solid hatch distinction (or similar) was considered and rejected as visual clutter for unclear benefit. Revisit if reviewers complain about not being able to tell the two apart at a glance.
- **`rescued_by_skip` boolean column in CSV** — useful for cohort-level "did skipping help?" stats. Deferred; can be added trivially later by diffing pre-skip and post-skip verdicts.
- **Skip budget** — a cap on skipped:kept ratio. The consistency gate is the only check. If empirical use shows pathological rescues, revisit.
- **Promoting `isi_hist_corr` to a tier-1 auto-pass in the composite verdict** — Option B from the prior spec's brainstorming. Independent from this work; tracked in the prior spec's §7.
- **Heatmap normalization** (Q3 from May 2026 review). Tracked in the prior spec's §7.
- **Tightening depth thresholds** — depends on whether pre-UM drift correction lands. Out of scope here.
- **Adjusting the consistency gate threshold from 0.85** — same value as the composite badge for consistency. Revisit if cohort analysis shows it's too loose or too strict.

## 8. Open questions

None at design time. All clarifications resolved in the brainstorming dialogue.
