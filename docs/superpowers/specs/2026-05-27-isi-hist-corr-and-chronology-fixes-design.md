# ISI Histogram Correlation Metric + Chronology / Stage Fixes

**Date:** 2026-05-27
**Owner:** Ben (UCL)
**Status:** Design

## 1. Purpose

Add a richer, biophysically motivated ISI-based cross-session similarity metric to the tracking_qc pipeline, and fix three chronology / stage-related issues uncovered while reviewing the QC sheets.

This spec covers a single coordinated change because all components share `src/visdetect/analysis/tracking_qc.py` and `scripts/pipelines/tracking/{build_qc_sheets,qc_sheet_figures}.py`, and because the new metric's interpretation depends on correctly-ordered sessions.

## 2. Background

ISIs are biophysically specific cell fingerprints (refractory period, bursting kinetics, intrinsic firing patterns). They are unlikely to change for a given neuron across sessions and should therefore make a strong cross-session matching signal. The current pipeline has two ISI-based badges:

- `badge_isi` — from `track_validation_stats.csv` (validate_long_tracks pipeline; adjacent-pair Pearson r).
- `badge_isi_peak` — peak-bin agreement only (in-house; can mishandle bursting cells whose ISI argmax flips between burst and inter-burst modes depending on which is taller in any given session).

Neither captures the full ISI distribution shape, so a bursting cell consistent across sessions can be spuriously flagged.

Empirical exploration (n=61 long-track UIDs, May 2026):
- Pearson r of per-session log-ISI histograms separates gold-standard UIDs (0.97–0.99) from known matching-failure UIDs (0.58–0.61) with a large dynamic range.
- Cohort percentiles (5/25/50/75/95): 0.51 / 0.79 / 0.86 / 0.94 / 0.98.
- UID 334 (currently flagged suspect due to per-session FR drops) scores 0.945 — consistent with "one bursting cell that occasionally goes quiet" rather than a matching failure. Demonstrates the metric handles bursting cells correctly.

Three bugs surfaced during the same review:

- **B1 — Chronological sort broken.** `build_cache` uses `manifest["session_name"].astype(str).tolist()` to derive the chronological-order map. The manifest stores session names as ints (so `1072025` not `"01072025"`), but the cache stores them as raw 7/8-char filesystem strings. Lookups miss → default position 1e9 → effectively random sort.
- **B2 — Tracking cohort filter too strict.** The cache is built against `load_staging_manifest(qc_only=True, apply_filter=True)`, which requires both `min_trials=150` AND `min_dprime=0.8` (per `SESSION_FILTER` in `visdetect.analysis.config`). For *behavioral* analyses this is correct, but for *tracking* analyses the d′ floor wrongly excludes early-Naive and early-Learning sessions where the mouse was engaged (≥150 trials) but performing poorly — exactly the sessions needed to study a cell's trajectory through learning. Additionally, sessions absent from the manifest currently default to `stage="Learning"`, silently mislabeling late-date sessions that failed QC for reasons other than stage.
- **B3 — Heatmap chronology direction.** `_draw_heatmap` uses `origin='lower'` so the earliest session lands at the bottom of the heatmap y-axis. User prefers earliest at top (matches how the eye reads top-to-bottom).

## 3. Components

### 3.1 — `baseline_isi_hist_corr` metric (new)

**Location:** `src/visdetect/analysis/tracking_qc.py`.

**Signature:**

```python
def baseline_isi_hist_corr(per_session_isi_hists: Sequence[np.ndarray]) -> float:
    """Median pairwise Pearson r of per-session log-ISI histograms.

    Captures full ISI distribution shape — handles bursting cells (with
    consistent bimodal ISIs) correctly, unlike isi_peak_agreement which
    looks only at the argmax bin. Architecturally mirrors waveform_corr.

    NaN-only hists and flat (std < 1e-12) hists are dropped.
    Returns NaN if fewer than 2 valid sessions.
    """
```

**Implementation pattern** mirrors `waveform_corr` and `baseline_psth_corr`:

1. Drop None / NaN-only / flat hists.
2. Mean-subtract + L2-normalize each remaining hist.
3. Compute all n×(n−1)/2 pairwise dot products.
4. Return median (not mean — robust to one outlier session).

Median is preferred over mean because a single dramatic outlier session (matching failure mid-track) would pull the mean down disproportionately even when most of the track is genuinely consistent. Median preserves the "is the typical pair consistent?" question.

**Thresholds (new constants):**

```python
ISI_HIST_CORR_PASS: float = 0.85
ISI_HIST_CORR_WARN: float = 0.65
```

Calibration justification from cohort data:
- Gold-standard UIDs (942, 1207, 776, 1712): 0.97–0.99 → pass.
- Borderline review UID 511: 0.86 → just passes.
- Anti-drift suspect UID 177: 0.745 → warn.
- Known matching-failures UID 779, 872: 0.58–0.61 → fail.
- Salvageable suspect UID 334: 0.945 → pass on this badge (other badges still flag it).

**Badge function:**

```python
def badge_isi_hist_corr(r: float) -> str:
    return _badge_threshold(r, ISI_HIST_CORR_PASS, ISI_HIST_CORR_WARN, direction="high")
```

NaN → fail (standard pattern for ISI metrics; distinct from `badge_func_resp` which is lenient on NaN).

**Tests** in `tests/analysis/test_tracking_qc.py`:
- Identical hists → r=1.
- Same shape, different magnitudes (Pearson invariance) → r=1.
- Flipped-polarity hists → median r=-1.
- NaN-only and flat hists dropped.
- <2 valid → NaN.
- Threshold behavior at PASS / WARN / FAIL boundaries, plus NaN → fail.

### 3.2 — Composite verdict integration

**Location:** `scripts/pipelines/tracking/build_qc_sheets.py`.

In BOTH the main render-loop verdict computation and the trimmed-verdict computation: replace `badge_isi_peak(metrics["isi_peak_agree"])` with `badge_isi_hist_corr(metrics["isi_hist_corr"])`. The composite verdict still uses 6 badges (`badge_isi`, `badge_depth`, `badge_wave`, `badge_fr`, `badge_isi_hist_corr`, `badge_func_resp`).

In `compute_uid_metrics`: add `"isi_hist_corr"` to the returned dict alongside the existing `"isi_peak_agree"` (which stays for diagnostic transparency).

In `verdicts.csv` and `verdicts_trimmed.csv`: add columns `isi_hist_corr` and `badge_isi_hist_corr`. Keep `isi_peak_agree` and `badge_isi_peak` columns (no code loss; now informational, not in composite).

PDF visual layout unchanged (still 4 visual badges via Option A from prior bimodality work).

### 3.3 — Chronological sort fix (B1)

**Location:** `scripts/pipelines/tracking/build_qc_sheets.py`, `build_cache` and the final per-UID `sessions.sort(...)` call.

Normalize all session names to `zfill(8)` before constructing the `order_idx` lookup AND before each `r.session_name` lookup against it:

```python
def _norm_session(name) -> str:
    return str(name).zfill(8)

sessions_chrono = [_norm_session(s) for s in manifest["session_name"].tolist()]
order_idx = {s: i for i, s in enumerate(sessions_chrono)}
for uid in intermediates:
    intermediates[uid].sessions.sort(
        key=lambda r: order_idx.get(_norm_session(r.session_name), 1e9)
    )
```

Same `_norm_session` used in the per-session iteration earlier in `build_cache` for consistency, and for the `stage_by_session` lookup keys (so e.g. a cache entry for session `"9072025"` finds the manifest entry stored as int `9072025` → string `"9072025"` → zfill `"09072025"`).

**Side effect:** the trimming algorithm (`find_stable_subset` → `longest_good_run`) operates on `uid.sessions` in list order. After the fix, "longest contiguous run" correctly means "longest *chronological* contiguous run" — which is what trimming is supposed to mean. Existing rescue counts may shift modestly; expected direction is for the better, since spurious "contiguous" runs from random ordering get replaced by genuinely contiguous runs.

### 3.4 — Tracking-QC filter relaxation + Unknown stage handling (B2)

**Location:** `scripts/pipelines/tracking/build_qc_sheets.py`, `build_cache`.

**Behavior change.** Replace the current strict QC filter with a looser tracking-QC filter that drops the d′ gate but keeps the trial-count gate:

```python
# Was: manifest = load_staging_manifest(qc_only=True, apply_filter=True)
manifest = apply_dynamic_filter(min_trials=150, min_dprime=None)
```

Rationale: early-Naive and early-Learning sessions can be behaviorally engaged (>150 trials) but score d′<0.8, and those sessions are exactly what we need for cross-stage tracking analyses. The d′ floor is appropriate for SDT/psychometric analyses but wrongly excludes engaged-but-naive sessions for cell-tracking purposes. `min_trials=150` is kept because PSTHs from fewer than ~150 trials are too noisy to support cross-session comparisons. The number can be revisited in a future spec.

**Sessions absent from this looser manifest** (i.e. <150 trials) get `stage="Unknown"` AND are unconditionally treated as outliers in trimming. Rationale: a session with <150 trials reflects a disengaged mouse; the neural data is too sparse to be useful for tracking and should not contribute to "kept" runs.

**Sessions present in the looser manifest** get their original stage from `staging.csv` (Learning or Expert post-merge-naive-into-Learning per `SESSION_FILTER.merge_naive_learning=True`).

**Rendering changes (qc_sheet_figures.py):**

- Extend the local stage-color map with `"Unknown": "#bbbbbb"` (light grey, distinct from the trace-dimming grey `"0.7"` used for trimmed-not-Unknown sessions).
- Stage stripe in `draw_header`: Unknown cells colored grey. NO hatch overlay on top of Unknown — the grey color alone communicates "skip this" per user feedback in brainstorming.
- PSTH summary panels (page 2): no code change required; Unknown sessions are naturally skipped by the `for st in STAGE_ORDER` iteration since `STAGE_ORDER = ["Learning", "Expert"]`.

**Trimming interaction.** Add a fifth flag dimension `unknown_stage` to `session_outlier_flags`, mirrored from `rec.stage == "Unknown"`. Extend the composite outlier rule:

```python
out["is_outlier"][i] = (
    out["isi_peak"][i]
    or strikes >= 2
    or out["unknown_stage"][i]   # NEW
)
```

Unknown-stage sessions are unconditionally dropped from any kept run.

### 3.5 — Heatmap chronology direction (B3)

**Location:** `scripts/pipelines/tracking/qc_sheet_figures.py`, `_draw_heatmap`.

Flip `imshow` origin and extent so row 0 (earliest session, per the now-correct sort) appears at the TOP of the heatmap and later sessions count down:

```python
ax_main.imshow(mat, aspect="auto", origin="upper", cmap="magma",
               extent=[centers[0], centers[-1], mat.shape[0], 0],
               vmin=0, vmax=max(vmax, 1e-6))
```

Update the red trim-marker code (added in prior trim-visualization work) to use the same flipped y-coord convention so dropped-row markers appear adjacent to the correct rows. Marker for dropped row `i` should render at y-coord `i` with `origin='upper'` extent semantics.

### 3.6 — Panel legends / keys (new, per user request)

**Location:** `scripts/pipelines/tracking/qc_sheet_figures.py`.

Add small text-box legends to the panels where visual encoding can be ambiguous. Legends are conditional on relevance — no clutter when the corresponding marking isn't present in a given UID's PDF.

- **Header (`draw_header`):** add a one-line legend below the stage stripe (and below the existing trim annotation if present):
  - `"stripe: Learning · Expert · Unknown · /// = trimmed"` (8pt grey).
- **Page-1 ISI distribution panel:** if any dropped sessions present, add a tiny annotation in top-right: `"grey traces = dropped"` (7pt grey).
- **Page-1 scatter panels (Depth, Amplitude, Baseline FR):** if any dropped sessions present, add a tiny annotation: `"○ = dropped"` (7pt grey) on the first such panel only (to avoid 3× repetition).
- **Page-2 heatmaps:** if dropped sessions present, add a tiny annotation in top-left: `"red bar = dropped row"` (7pt white text overlaid in top-left corner of the dark heatmap).
- **Page-2 PSTH summary panels with miss_keys** (Change_ON Big-Hit, Small-Hit): existing Learning/Expert legend stays. Update legend to two entries per stage: `"L hit · L miss · E hit · E miss"` shown as solid/dashed × light/dark. Achieved by passing `linestyle` labels into `inset.legend`.

## 4. Data flow

```
manifest (apply_dynamic_filter min_trials=150)    unit_index.csv (UM)
       │                                                  │
       └──────────────────┬───────────────────────────────┘
                          │
       build_cache  ──── (_norm_session zfill(8) for chronological sort)
                          │  (stage = looser-manifest stage OR "Unknown")
                          ▼
       cache (per-UID intermediates, chronologically sorted)
                          │
       compute_uid_metrics ── isi_hist_corr (new) alongside existing metrics
                          │
       composite_verdict ── badge_isi_hist_corr replaces badge_isi_peak
                          │
       find_stable_subset ── Unknown-stage sessions forced outlier (new flag dim)
                          │
       write_uid_pdf
                          │
                          ├── render_page1 (greyed/open-circle dropped + legends + stage stripe w/ Unknown)
                          └── render_page2 (heatmap origin='upper' + matched PSTH summary + legends)
                          │
       verdicts.csv (adds isi_hist_corr, badge_isi_hist_corr)
       verdicts_trimmed.csv (same additions)
       per-UID PDFs
```

## 5. File-level changes

| File | Change |
|---|---|
| `src/visdetect/analysis/tracking_qc.py` | Add `ISI_HIST_CORR_PASS=0.85`, `ISI_HIST_CORR_WARN=0.65`. Add `baseline_isi_hist_corr` and `badge_isi_hist_corr`. Extend `session_outlier_flags` with `unknown_stage` dimension; update composite-outlier rule. |
| `tests/analysis/test_tracking_qc.py` | Add ~7 tests for the new metric + badge; add 2 trimming tests for the unknown-stage outlier behavior. |
| `scripts/pipelines/tracking/build_qc_sheets.py` | (a) Replace strict manifest load with `apply_dynamic_filter(min_trials=150, min_dprime=None)`. (b) Add `_norm_session` helper and use it in chronological sort + stage-by-session lookup. (c) Change missing-manifest stage default from `"Learning"` to `"Unknown"`. (d) Add `isi_hist_corr` to `compute_uid_metrics` returned dict. (e) Swap `badge_isi_peak` → `badge_isi_hist_corr` in both verdict computations. (f) Add `isi_hist_corr`, `badge_isi_hist_corr` columns to both verdict CSVs. |
| `scripts/pipelines/tracking/qc_sheet_figures.py` | (a) Extend stage-color map with `"Unknown": "#bbbbbb"`. (b) Flip heatmap `origin='lower'` → `'upper'` + extent y-swap. (c) Update trim-marker red-rectangle anchoring for flipped origin. (d) Add 5 conditional panel legends per §3.6. (e) Update PSTH-summary legend to include solid/dashed = hit/miss when miss_keys present. |
| `docs/superpowers/specs/2026-05-27-isi-hist-corr-and-chronology-fixes-design.md` | This spec, committed. |

## 6. Testing

**New unit tests** (in `tests/analysis/test_tracking_qc.py`):
- `test_baseline_isi_hist_corr_identical_returns_one`
- `test_baseline_isi_hist_corr_handles_magnitude_scaling`
- `test_baseline_isi_hist_corr_flipped_polarity_median`
- `test_baseline_isi_hist_corr_drops_none_sessions`
- `test_baseline_isi_hist_corr_drops_flat_sessions`
- `test_baseline_isi_hist_corr_too_few_returns_nan`
- `test_badge_isi_hist_corr_thresholds` (PASS / WARN / FAIL plus boundary values plus NaN→fail)
- `test_session_outlier_flags_unknown_stage_is_outlier`
- `test_find_stable_subset_drops_unknown_sessions`

**No new unit tests for**: sort fix, stage default change, heatmap origin flip, legend additions — all verified end-to-end via smoke render.

**End-to-end smoke**: re-run `py scripts/pipelines/tracking/build_qc_sheets.py` (cache rebuild required because the looser manifest filter changes which sessions are included). Confirm:
- 61/61 UIDs render without exception (or N/61 — cohort size may change with looser filter).
- Verdict CSVs have the new columns.
- UID 942 verdict unchanged (trusted on both full and trimmed) — gold standard not regressed.
- Verdict distribution shift documented in commit message.
- UID 942 page 2 heatmaps: earliest session at top of y-axis, later sessions descending.
- A UID with manifest-missing sessions (e.g., UID 942's session 26082025, currently mis-labeled Learning) now shows grey in the stage stripe and appears as outlier in trim.
- Spot-check: an early-Naive low-d′ session that was previously excluded now appears in cache with `stage="Learning"` (per the merge-naive rule) and is NOT forced outlier.

## 7. Non-goals

The following are explicitly out of scope for this spec and will be addressed separately:

- **Heatmap normalization** (Q3 from May 2026 review) — per-row z-score, baseline subtraction, or vmax percentile tuning. Deferred to a future spec.
- **Option B from brainstorming (ISI gate)** — promoting `isi_hist_corr` to a tier-1 auto-pass for very-high values. Deferred until empirical evidence supports it.
- **Re-running UnitMatch with drift-corrected positions** — large project; documented in `tracking_improvement_options.md` memory.
- **Replacement of `badge_isi`** (the validate_long_tracks pipeline metric) — kept as-is; remains the externally-validated long-track signal.
- **Adjusting `min_trials=150`** — kept at current value; revisit in a future spec.

## 8. Open questions

None at design time. All clarifications resolved in the brainstorming dialogue.
