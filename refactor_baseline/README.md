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

## Notes

- Live stats CSVs are not git-tracked, so this snapshot is the only durable
  oracle. Do not delete it until the refactor is complete and verified.
- A full `run_all.py` run is being captured separately to validate/refresh this
  snapshot; any discrepancies will be reconciled and noted here.
