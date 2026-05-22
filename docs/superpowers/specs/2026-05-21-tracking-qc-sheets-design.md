# Tracking QC sheets — design

**Date**: 2026-05-21
**Author**: Ben (with Claude)
**Branch context**: results land on `refactor/architecture` or a feature branch off it
**Related work**: UnitMatch job 3013370 complete; DeepUnitMatch job 3014578 pending. See `memory/neuron_tracking_may2026.md` and `scripts/pipelines/tracking/validate_long_tracks.py`.

---

## 1. Purpose & scope

**Primary purpose**: validate the UnitMatch-tracked cohort before scientific use. Produce per-UID QC sheets that let the reviewer (Ben) decide trust/exclude on a case-by-case basis.

**Out of scope** (deferred to later projects):
- Scouting scientific questions on the cohort
- Paper-grade figures
- Cohort-summary matrices (heatmap of all UIDs × criteria)
- Automated outlier detection beyond the per-criterion badges defined here
- Cross-session response-correlation as a fifth badge

**Success criterion**: each PDF tells the reviewer, at a glance, (a) what the composite verdict is, (b) which criterion drove the verdict, and (c) what the underlying physical and functional fingerprints look like across sessions.

---

## 2. Cohort

All long tracks: `span ≥ 10` sessions in `cell_registry.csv`. ~61 UIDs.

Includes:
- 19 Naive→Expert tracks (the scientifically central cohort)
- 18 ≥20-session tracks
- 3 known suspects (UIDs 779, 873, 872) included as cautionary cases

Stage labels use `SESSION_FILTER.merge_naive_learning = True` — only **Learning** and **Expert** stages appear (Naive merged into Learning).

---

## 3. Architecture & data flow

### Files

| Path | Role |
|---|---|
| `src/visdetect/analysis/tracking_qc.py` | New helper module. Cross-session metric functions (depth std, waveform corr, FR CV). Reusable for later cohort summaries. |
| `scripts/pipelines/tracking/build_qc_sheets.py` | New CLI script. Loops sessions → extracts per-UID data → renders 2-page PDFs. |
| `FIGURES/tracking_qc/per_uid_sheets/uid_{NNNN}.pdf` | Output: 61 two-page PDFs |
| `FIGURES/tracking_qc/verdicts.csv` | Output: composite index — UID × badges × verdict |
| `data/cache/tracking_qc_intermediates.pkl` | Cached per-UID dicts so figure tweaks don't require re-reading pkls |

### Canonical imports (post-refactor)

| Need | Import from |
|---|---|
| Constants (DEFAULT_BIN_SIZE, EVENT_VALID_OUTCOMES, alignment windows) | `visdetect.analysis.constants` |
| SESSION_FILTER, staging manifest config | `visdetect.analysis.config` |
| PSTH/z-score/bootstrap | `visdetect.analysis.utils` |
| Figure paths, STAGE_COLORS, OUTCOME_COLORS | `visdetect.suite.config` |
| `load_staging_manifest`, `build_unit_table` | `visdetect.suite.loader` |
| `setup_style`, `save_figure`, `add_stage_background` | `visdetect.suite.plotting` |
| Good-stable cluster selection | `visdetect.core.qc` |

Do not import from `analysis_suite/{config,loader,utils,plotting}.py` — those files no longer exist post-refactor. See `memory/feedback_canonical_imports.md`.

### Data-flow loop

```
1. Load cell_registry.csv → long_tracks (span ≥ 10) → ~61 UIDs
2. For each session in manifest (~42):
     load session.pkl ONCE
     for each long-track UID present in this session:
       extract & cache per-UID dict:
         - peak-channel mean waveform (normalized)
         - multi-channel footprint snippet (for first/mid/last display)
         - peak channel, depth, amplitude
         - ISI histogram (log-spaced bins)
         - baseline FR
         - PSTH(Baseline_ON, all outcomes)
         - PSTH(Change_ON, hits, big   change-size pool, -500..+500ms)
         - PSTH(Change_ON, misses, big change-size pool, -500..+500ms)
         - PSTH(Change_ON, hits, small change-size pool, -500..+500ms)
         - PSTH(Change_ON, misses, small change-size pool, -500..+500ms)
         - PSTH(Hit lick)
     del sess; gc.collect()
   → save data/cache/tracking_qc_intermediates.pkl
3. For each UID:
     compute cross-session metrics (depth std, waveform r, FR CV)
     pull ISI median from track_validation_stats.csv
     assign per-criterion badges + composite verdict
     render 2-page figure → uid_{NNNN}.pdf
4. Write verdicts.csv
```

Outer loop is by session (not UID) because session.pkl loading is the expensive step. Per-UID intermediates are pickled so re-rendering is cheap.

---

## 4. Pooling and event-alignment rules

### Change-size pools (Change_ON only)

| Pool | Sizes |
|---|---|
| **Big** | 2.0× + 4.0× |
| **Small** | 1.25× + 1.35× |
| Excluded | 1.5× (ambiguous mid) — appears in insets only if useful, not in heatmaps |

### Event/outcome combinations used

| Event | Outcomes included | Window | Notes |
|---|---|---|---|
| `Baseline_ON` | all (hit, miss, fa, abort, ref) | per `constants.DEFAULT_*` | Pooled across outcomes for v1. **TODO**: split by outcome in v2 — engagement differs by outcome and may modulate baseline response (deferred per user note). |
| `Change_ON` | hit, miss only (per `EVENT_VALID_OUTCOMES`) | -500 to +500 ms | Window captures preparatory ramp + sensory/decision response |
| `Hit lick` | hit only | per `constants.DEFAULT_*` | Motor-aligned |

PSTH bin size: `DEFAULT_BIN_SIZE = 0.025 s`. Gaussian smoothing sigma: `DEFAULT_SIGMA_MS = 25.0 ms`.

---

## 5. Page layout

### Page 1 — physical (8 panels)

1. **Header**: UID, span, session-date range, N→E flag, 4 verdict badges, composite tag, stage stripe
2. **Footprint @ first session** — multi-channel waveform snippet
3. **Footprint @ mid session** — same UID, ~midpoint session
4. **Footprint @ last session** — same UID, final session
5. **Peak-channel waveform overlay** — all sessions, stage-colored
6. **Depth on probe across sessions** — line plot, stage-colored
7. **Amplitude across sessions** — line plot
8. **UM pairwise centroid_dist** — strip plot, one value per consecutive session pair

### Page 2 — functional (7 panels)

1. **Header** (mirror of page 1, so page 2 is readable standalone)
2. **ISI distribution overlay** — log-x, sessions stage-colored
3. **Baseline FR across sessions** — line plot, stage-colored
4. **PSTH heatmap · Baseline_ON · all outcomes pooled** — chronological (rows = sessions) + Learning vs Expert mean inset. Title carries TODO marker for outcome-split.
5. **PSTH heatmap · Change_ON Big-Hit (2.0× + 4.0×)** — chronological, -500 to +500 ms, with inset showing L vs E × hit/miss traces
6. **PSTH heatmap · Change_ON Small-Hit (1.25× + 1.35×)** — same structure
7. **PSTH heatmap · Hit lick** — chronological + L vs E mean inset

---

## 6. Panel proportions (W:H)

Deliberate per panel type (`gridspec` with explicit `width_ratios` / `height_ratios`). See `memory/feedback_figure_proportions.md`.

| Panel | Target W:H | Notes |
|---|---|---|
| Waveform overlay | ~2:1 | Time axis dominant; spike shape readable |
| Multi-channel footprint | ~1:1.5 (tall) | Matches probe column geometry |
| Depth / amplitude / FR across sessions | ~3:1 | Wide strip — time-axis is the story |
| ISI distribution (log-x) | ~2:1 | Resolve refractory peak + burstiness shoulder |
| PSTH heatmap (sessions × time) | ~3:1, each row ≥ 3 px tall | Every session-row must be visible |
| Stage-mean inset / overlay traces | ~1.2:1 | Compact; readable trace separation |
| UM pairwise score strip | ~5:1 | Very wide; one value per pair |

---

## 7. Metrics & verdict logic

Four badge metrics, each ✅ pass / ⚠ warn / ❌ fail.

| # | Metric | Computation | ✅ pass | ⚠ warn | ❌ fail |
|---|---|---|---|---|---|
| 1 | **ISI fingerprint** | `median` column of `FIGURES/tracking_qc/track_validation_stats.csv` (pre-computed) | ≥ 0.75 | 0.65–0.75 | < 0.65 |
| 2 | **Depth stability** | `std(peak_depth_um)` across sessions | ≤ 15 µm | 15–30 µm | > 30 µm |
| 3 | **Waveform-shape correlation** | Mean pairwise Pearson r of L2-normalized peak-channel mean-waveform across sessions | ≥ 0.95 | 0.90–0.95 | < 0.90 |
| 4 | **Baseline FR stability** | CV (std/mean) of per-session baseline firing rate | ≤ 0.35 | 0.35–0.60 | > 0.60 |

**Composite verdict** (header tag):
- ✅ **trusted** — all 4 pass
- ⚠ **review** — at most 1 warn, no fails
- ❌ **suspect** — any fail, or ≥ 2 warns

**Validation anchors** (sanity checks for threshold choices):
- Top N→E tracks 334 / 1294 / 600 / 511 / 177 should land trusted or review on the ISI badge
- Known suspects 779 (ISI 0.28), 873 (0.59), 872 (0.62) all fail criterion 1 → composite **suspect**, no ambiguity
- Thresholds match the empirical gap from `track_validation_stats.csv`: 40/61 > 0.80, 14/19 N→E ≥ 0.75

Thresholds are documented at module top in `tracking_qc.py` as constants — tweakable later if validation anchors suggest moving cutoffs.

---

## 8. Outputs

### Per-UID PDF

`FIGURES/tracking_qc/per_uid_sheets/uid_{NNNN}.pdf` — two pages, ~A4 each.

### Index CSV

`FIGURES/tracking_qc/verdicts.csv` with columns:

```
global_uid, span, sessions, has_naive_to_expert, suspect_known,
isi_median, depth_std_um, wave_corr, fr_cv,
badge_isi, badge_depth, badge_wave, badge_fr,
verdict
```

`verdict ∈ {trusted, review, suspect}`. Sortable by composite for triage.

### Cached intermediates

`data/cache/tracking_qc_intermediates.pkl` — dict keyed by `global_uid`, value = per-session data dict (waveform, footprint, ISI, FR, PSTHs). Refresh by deleting the file.

---

## 9. Open items / future v2 enhancements

| Item | When |
|---|---|
| Split Baseline_ON PSTH by outcome (hit/miss/fa) | v2, after first cohort review |
| Add 5th badge: cross-session response correlation (Baseline_ON or Change_ON template) | v2, if v1 verdicts disagree with eyeball judgment |
| Cohort-summary matrix figure (UIDs × criteria heatmap) | Separate spec |
| DeepUM-vs-UM comparison sheet (same UID rendered from both pipelines side-by-side) | After DeepUM job 3014578 lands; separate spec |

---

## 10. Non-goals

- This spec does NOT touch DeepUnitMatch results (job 3014578 still queued). Once DeepUM lands, the QC-sheet builder should be re-usable against `all42_deep/` by changing the registry path — no other code changes.
- This spec does NOT define the within-cell learning analyses that the validated cohort will feed into. Those are downstream.
- This spec does NOT modify the existing `validate_long_tracks.py` — it consumes that script's output (`track_validation_stats.csv`).
