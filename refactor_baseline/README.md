# Refactor Baseline — Parity Oracle

Captured at the start of the in-place architecture refactor
(see [../docs/REFACTOR_PLAN.md](../docs/REFACTOR_PLAN.md), Phase 0).

**Branch**: `refactor/architecture`
**Baseline tag**: `pre-refactor-baseline`
**Captured**: 2026-05-16

## Purpose

This directory is the **regression oracle** for the refactor. The refactor is
behavior-preserving: no algorithm or parameter value may change. After every
refactor step, re-running an affected analysis must reproduce these numbers.
A refactor step that changes a number is a bug, not progress.

## Contents

- `stats/` — snapshot of all 41 live `*_stats.csv` files from
  `analysis_suite/figures/{NN}_*/` plus `analysis_suite/cache/optotagging_stats.csv`
  (copied as `_cache_optotagging_stats.csv`). Pre-TPrime and archived copies were
  excluded — they are stale (CLAUDE.md).
- `MANIFEST.sha256` — sha256 of every snapshot CSV, for fast parity diffs.

## Test baseline

`pytest tests/` at `pre-refactor-baseline`:

- **27 passed**
- **2 collection errors** (pre-existing, not caused by the refactor):
  - `tests/test_coding_direction.py` — imports `visdetect.analysis.coding_direction` (module does not exist)
  - `tests/test_population.py` — imports `visdetect.analysis.population` (module does not exist)
- 5 warnings (numpy "Mean of empty slice" — benign)

Parity target for tests: 27 passed + the same 2 collection errors after every step.

## How to check parity after a refactor step

1. Re-run the analysis scripts the step touched
   (`cd analysis_suite && py {module}/{script}.py`).
2. Compare against the snapshot:
   ```
   cd refactor_baseline/stats && sha256sum -c ../MANIFEST.sha256
   ```
   Or diff an individual CSV against `refactor_baseline/stats/<module>/<name>.csv`.
3. `pytest tests/ --continue-on-collection-errors` must still report 27 passed.

## Reconciliation (2026-05-18)

The Phase 0 snapshot grabbed on-disk CSVs assuming they were current. A fresh
`run_all.py` (correct environment, current code) was run to validate them and
got through 19/50 scripts before being stopped. Reconciliation of those 19:

- **35/40 stats CSVs byte-identical** to the snapshot — validated.
- **5/40 differed — all pre-refactor staleness, 0 numerical regressions:**
  - `learning_curve` — snapshot had n=20 sessions, current manifest has 25
  - `post_error_controls`, `post_error_streak_controls` — new
    `hmm_Impulsive/Engaged/Disengaged` rows from the Phase-0 `auto_label_states`
    rewrite (commit `bbfcfec`, on `main`, pre-refactor)
  - `population_heatmap` — stat keys renamed + 2 rows added; values identical
  - `sequence_significance` — 16th-digit float noise only

The 5 stale files were refreshed from the fresh run; the snapshot + manifest
now match current code for all 19 completed scripts. Phase 0/1/2 changed no
analysis code — the verification holds.

## Phase 3+ parity protocol — per-group self-baselining

The global `run_all.py` is slow (30-min timeouts). From Phase 3 on, parity is
checked **per module group**, self-baselined:

1. Run the group's scripts on the *current* (pre-edit) code → capture outputs.
2. Apply the refactor edits to that group.
3. Re-run the group → assert byte-identical to step 1.
4. `pytest` green, then commit.

This compares before/after on identical code+data and does not depend on the
(partially stale) global snapshot. The snapshot here remains a coarse
cross-check.

## Notes

- Live stats CSVs are not git-tracked, so this snapshot is the only durable
  oracle. Do not delete it until the refactor is complete and verified.
