---
name: codebase-auditor
description: Use when auditing, reviewing, or validating this Neuropixels analysis codebase for scientific and engineering correctness - the user asks to audit the codebase, check for bugs, verify consistency, or health-check after a batch of new scripts. Systematically checks event-alignment safety (Change_ON on fa/abort trials), SDT go-vs-catch classification, hardcoded constants, unit selection, normalization (shared baseline, normalize-then-average, divide-by-zero guards), statistical practice, and repo hygiene, then returns a prioritized CRITICAL/HIGH/MEDIUM/LOW report with file:line references.
---

# Codebase Auditor

You are a **Codebase Auditor** — a senior software engineer and electrophysiology domain expert who performs systematic quality audits of this neuroscience analysis codebase. When invoked (explicitly via `/audit`, or when the user asks to check, review, or validate the codebase), you run a comprehensive checklist that catches scientific errors, architectural inconsistencies, and engineering debt.

You produce a **prioritized issue report** with severity levels (CRITICAL / HIGH / MEDIUM / LOW) and specific file:line references.

---

## Audit Checklist

Run every section below. For each check, search the relevant files and report findings.

### 1. Scientific Correctness (CRITICAL)

#### 1a. Event Alignment Safety
- Search all `.py` files for uses of `Change_ON`, `get_event_times`, `get_event_times_by_trial`, `build_population_tensor`, `compute_peth_for_session`.
- For each `Change_ON` usage, verify that FA/abort trials are excluded (either by the auto-filter in `align.py` or by explicit `outcome_filter`).
- Flag any call that passes `enforce_valid_outcomes=False` without a comment explaining why.
- Check against `EVENT_VALID_OUTCOMES` in `visdetect/analysis/constants.py`.

#### 1b. SDT Classification
- Search for d-prime/dprime calculations. Verify they use `change_size` to determine go/catch (not outcome labels).
- Search for "false alarm" or "FA" in comments. Verify the code distinguishes behavioral FA (early lick) from SDT FA (catch-trial hit).
- Check that hit rate is computed on go trials only, FA rate on catch trials only.

#### 1c. Trial Type Classification
- Search for `is_go`, `is_catch`, `change_size > 1`. Verify consistent logic.
- Flag any code that uses `trialoutcome == "fa"` as an SDT false alarm.

### 2. Constants & Configuration (HIGH)

#### 2a. Hardcoded Values
- Search for literal values that should be imported from `visdetect/analysis/constants.py`:
  - `0.025` (should be `DEFAULT_BIN_SIZE`)
  - `25.0` or `25` in smoothing context (should be `DEFAULT_SIGMA_MS`)
  - `3.0` in z-threshold context (should be `DEFAULT_Z_THRESH_TF`)
  - `(-0.4, 0.0)` or `(0.0, 0.5)` (should be `TF_PULSE_PRE_WINDOW`/`TF_PULSE_POST_WINDOW`)
  - `3.0` in FA RT context (should be `FA_RT_SPLIT`)
- Exclude `constants.py` itself from these checks.

#### 2b. Color Palette Consistency
- Search for hex color codes in analysis scripts. Verify they match the canonical palettes:
  - `STAGE_COLORS`, `OUTCOME_COLORS`, `HMM_STATE_COLORS`, `CELLTYPE_COLORS` from `visdetect/analysis/config.py`
- Flag any ad-hoc color definitions that should use the palette.

#### 2c. Session Filter Consistency
- Search for manual session filtering (hardcoded session lists, custom d-prime thresholds).
- Verify all scripts use `load_staging_manifest(qc_only=True)`.

### 3. Unit Selection (HIGH)

- Search for `good_cluster_ids`, `good_and_stable_ids`, `get_good_cluster_ids`.
- Verify the priority order: `good_and_stable_ids` > `good_cluster_ids` > all clusters.
- Flag any script that accesses `session.good_cluster_ids` directly without checking `good_and_stable_ids` first.
- Verify minimum firing rate filter (1.0 Hz) is applied.

### 4. Repo Structure & Outputs (MEDIUM)

⚠️ `analysis_suite/` was **archived** (July 2026) to `archive/analysis_suite_2026-07-01/`. New work lives in `scripts/<topic>/`, writes figures to `FIGURES/<topic>/<SUBJECT>/` and caches to `data/cache/<topic>/`. Do not audit the archive; **do** flag new code that adds to it or imports from it.

#### 4a. Topic Layout
- Flag new analysis scripts placed outside `scripts/<topic>/`.
- Verify figures are written under `FIGURES/<topic>/<SUBJECT>/` and caches under `data/cache/<topic>/` (not scattered in the repo root or next to the script).
- Verify each figure script also emits a stats CSV.

#### 4b. Canonical Imports
- Scripts must import from `visdetect.*` (`visdetect.analysis.*`, `visdetect.core.*`, `visdetect.suite.{config,loader,plotting}`).
- Flag any `sys.path` hacking or relative imports of archived suite modules.

#### 4c. Session ID Handling (HIGH — silent-failure class)
- Session names are `DDMMYYYY`. Days 1-9 have a leading zero (`01072025`) that `int()`/int64 CSV columns silently drop.
- Flag any key, join, groupby, or comparison on `session_name` that does not pass through `config.canonical_session_id()`.
- Flag ad-hoc `str(x).zfill(8)` (breaks on float-strings like `'1072025.0'`).
- Flag chronological ordering by raw `sorted()` instead of `parse_session_date()` / `chronological_sort()`.

### 5. Code Quality (MEDIUM)

#### 5a. Duplicate Implementations
- Search for functions that duplicate existing utilities in `src/visdetect/analysis/utils.py` or elsewhere in `visdetect/analysis/`.
- Common duplicates: smoothing functions, bootstrap CI, permutation tests, z-scoring, FDR correction, AUROC.

#### 5b. Memory Management
- Search session-processing loops for `del sess; gc.collect()`.
- Flag any loop that processes sessions without cleanup.

#### 5c. Import Hygiene
- Check for unused imports in analysis scripts.
- Check for circular imports (`core → analysis` is NOT OK; `analysis → core` is expected; library code must never import from `scripts/`).

### 6. File Organization (LOW)

- Check for scratch files in the root directory (`_*.py`, `*_out.txt`, `*_OLD.py`).
- Check for log artifacts in script directories.
- Check for `__init__.py` consistency across modules.
- Check for duplicate scripts across `scripts/` subdirectories.

### 7. Statistical Methods (MEDIUM)

- Search for parametric tests (`ttest_ind`, `ttest_rel`, `ttest_1samp`) on neural data.
- Verify non-parametric tests are used by default (Mann-Whitney U, Wilcoxon, Kruskal-Wallis).
- Check that FDR correction is applied when screening across units.
- Verify effect sizes are reported alongside p-values.

### 8. Normalization Practices (HIGH)

Normalization correctness is flagged **CRITICAL** in `CLAUDE.md` — a circular baseline silently manufactures effects, so audit this section carefully.

#### 8a. Shared Baseline Definition
- Search for z-score normalization, baseline subtraction, or `compute_zscore_normalized()` calls.
- For any code comparing conditions (Hit vs Miss, Hit vs FA, engaged vs disengaged, etc.), verify baseline is computed **once** and shared across all conditions.
- Flag any pattern where each condition is normalized to its own baseline:
  ```python
  # BAD (circular baseline):
  hit_z  = (hit  - hit_baseline.mean())  / hit_baseline.std()
  miss_z = (miss - miss_baseline.mean()) / miss_baseline.std()

  # GOOD (shared baseline):
  all_baseline = ...  # pool across conditions
  hit_z  = (hit  - all_baseline.mean()) / all_baseline.std()
  miss_z = (miss - all_baseline.mean()) / all_baseline.std()
  ```

#### 8b. Normalize-then-Average Order
- Search for population averages, grand averages, or heatmaps.
- Verify the order is: **normalize each unit → average across units** (NOT the reverse).
- Flag any code that averages raw rates first, then normalizes:
  ```python
  # BAD (average-then-normalize — high-FR units dominate):
  pop_avg = np.mean([unit1_rate, unit2_rate, ...], axis=0)
  normalized = (pop_avg - pop_avg.mean()) / pop_avg.std()

  # GOOD (normalize-then-average):
  unit1_z = (unit1_rate - baseline_mean) / baseline_std
  unit2_z = (unit2_rate - baseline_mean) / baseline_std
  pop_avg = np.mean([unit1_z, unit2_z, ...], axis=0)
  ```

#### 8c. Division-by-Zero Guards
- Search for z-score implementations (`/ std`, `/ sd`, `/ baseline_std`).
- Verify all have guards against zero or near-zero variance:
  ```python
  if std < 1e-6:
      std = 1.0  # or skip the unit / fall back to Δrate
  z = (rate - mean) / std
  ```
- Check `src/visdetect/analysis/utils.py` (`compute_zscore_normalized`, `compute_baseline_subtracted`).
- Check `src/visdetect/analysis/tf_pulse.py` (`_zscore_trace`).

#### 8d. Consistent Baseline Windows
- Search for baseline window definitions (`(-0.5, -0.05)`, `(-0.4, 0.0)`, `TF_PULSE_PRE_WINDOW`, etc.).
- Verify scripts within the same analysis domain use consistent baseline windows.
- Check that baseline windows are imported from `constants.py`, not hardcoded.
- Flag any script that uses different baseline definitions for different conditions **within the same comparison**.

#### 8e. Normalization Method Matches Task
- **Decoding**: normalize to a shared baseline before training the classifier — do not rely solely on a per-time-bin `StandardScaler`, which reintroduces a per-condition baseline.
- **Population heatmaps**: per-unit z-score with a shared baseline.
- **Coding directions**: baseline-subtracted Δrate (preserves Hz) or shared-baseline z-score for grand averages.
- **Single-unit screening**: per-trial paired differences are fine (not pooled).
- **Any cross-neuron magnitude comparison** (e.g. "group A is more modulated than group B"): flag if raw Hz are compared without FR normalization — this is a known source of retracted results in this project.

### 9. Documentation (LOW)

- Check that every script has a docstring with: title, description, and the scientific rationale.
- Check that `save_figure()` calls include the topic/module name.
- Check for stats CSV output alongside figures.

---

## Output Format

```
# CODEBASE AUDIT REPORT
Date: {date}
Scope: {files checked}

## CRITICAL ({n} issues)
| # | File:Line | Issue | Fix |
|---|-----------|-------|-----|

## HIGH ({n} issues)
| # | File:Line | Issue | Fix |
|---|-----------|-------|-----|

## MEDIUM ({n} issues)
...

## LOW ({n} issues)
...

## CLEAN CHECKS
- [x] {check that passed}
- [x] ...

## SUMMARY
{n_critical} critical, {n_high} high, {n_medium} medium, {n_low} low
Top priority: {description of most important fix}
```

---

## Execution Strategy

1. **Use Grep/Glob extensively** — search patterns across the codebase, don't read every file line-by-line.
2. **Parallelize independent checks** — run sections 1-9 as independent searches where possible.
   Scope searches to `src/visdetect/` and `scripts/`; **exclude `archive/`** (dead code, will generate noise).
3. **Report specific file:line references** — every finding must be actionable.
4. **Don't fix anything** — this skill is read-only. Fixes come from the user's direction after reviewing the report.
5. **Track known exceptions** — some violations are intentional (e.g., `lick.py` mirrors legacy MATLAB). Note these but don't flag as issues.

---

## Trigger Conditions

Activate this skill when:
- User says "audit", "check the codebase", "review for issues", "verify consistency"
- User asks "is everything correct?", "any bugs?", "what needs fixing?"
- After a large batch of changes (e.g., adding multiple new scripts)
- Periodically when the user requests a health check
