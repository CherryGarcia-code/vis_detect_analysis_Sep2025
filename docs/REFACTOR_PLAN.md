# Refactor Plan: In-Place Cleanup of vis_detect_analysis_Sep2025

**Created**: 2026-05-16
**Supersedes**: `docs/AI_interaction/CODEBASE_REORGANIZATION_PLAN.md`,
`REORGANIZATION_PROGRESS_REPORT.md`, `REORGANIZATION_COMPLETE.md` (the stalled March 2026 effort)

## Decision

After evaluating a fresh-repo rebuild (`vis_detect_analysis_May2026`) versus an
in-place repair, the in-place repair was chosen. Rationale:

- The technical debt is **architectural/organizational**, not algorithmic — the
  March 2026 normalization audit graded the analysis logic A-.
- Architecture can be refactored in place. A rebuild would force re-porting and
  re-validating ~43 working analyses — more total work and more risk, landing
  that risk directly on validated science.
- Sep2025 is under version control (the rebuild repo is not). Git is the ideal
  safety net for an incremental, reversible refactor.
- A clean slate is only free for code not yet written; for working validated
  science it means re-validating everything.

The governance design developed in the May2026 effort (decision log, parameter
register, evidence protocol) is carried over as the discipline layer for this
refactor.

## Strategy

A behavior-preserving, branch-based, incremental refactor. Every step is a small
git commit; the test suite and analysis outputs must stay green/identical after
each. **No algorithm or parameter value changes in this track** — parameter
review runs as a separate parallel workstream.

The rule the March 2026 reorganization lacked: **a defined end-state, plus
enforcement so it cannot re-rot.**

## Phase 0 — Safety net & baseline

Goal: a reversible starting point and a parity oracle.

- Investigate the 3 modified files (`analysis_suite/loader.py`,
  `scripts/analysis/behavior/fit_behavioral_hmm.py`,
  `src/visdetect/analysis/hmm.py`) and the untracked `scripts/utils/` directory —
  commit or set aside deliberately; do not refactor on top of ambiguous state.
- Create branch `refactor/architecture` off `main`; tag the pre-refactor commit.
- Capture the parity oracle: run `pytest tests/` and `analysis_suite/run_all.py`
  (whatever runs in the local data environment), and snapshot the produced
  `*_stats.csv` files. After every later step, re-running must reproduce these.
  Stats CSVs are the regression check, not figure pixels.

## Phase 1 — Define the target architecture

Goal: the "definition of done" the 2026 reorganization lacked.

- Write a single concise `docs/ARCHITECTURE.md`: installable package, exactly one
  utilities location, no `sys.path` hacks, no hardcoded subject/paths, where
  library vs. analyses vs. scripts live, the archive policy, the docs policy.
  Everything below migrates toward this.

## Phase 2 — Low-risk consolidation (archives + docs)

Goal: easy wins first; shrink the surface before the risky surgery.

- Consolidate the legacy locations (`archive/`, `scripts/scripts_archive/`,
  `src/visdetect/analysis/archive/`, `analysis_suite/**/_archive/`,
  `AI_exploration/`) into one `archive/`, or delete (git history preserves them).
  Grep first to confirm nothing live imports them.
- Consolidate docs: 4 `NORMALIZATION_*.md` files into 1; the 3 reorganization
  docs archived (this plan supersedes them); `DOCUMENTATION_INDEX.md` kept as the
  single map.

## Phase 3 — Import system (kill `sys.path`)

Goal: a real installable package; the biggest mechanical task.
Current state: 154 `sys.path.insert` calls across 130 files.

- `setup.cfg` already supports `pip install -e .` — make that the supported
  workflow.
- Make `analysis_suite/` shared modules (`config`, `loader`, `utils`, `plotting`)
  importable as proper package modules rather than flat `sys.path` modules.
- Remove the `sys.path.insert` calls module group by module group
  (`01_behavior/`, then `02_single_unit/`, ...), committing per group, running
  tests + parity after each.

## Phase 4 — Single-source utilities

Goal: finish what commit `9283841` ("Centralize utils") started.

- One canonical utils location in `visdetect`. `analysis_suite/utils.py`
  functions move into the library; the parallel copy is removed.

## Phase 5 — Configuration (kill hardcoding)

- `SUBJECT` and `X:/` paths into config + env vars (`VISDETECT_SUBJECT`,
  `VISDETECT_DATA_ROOT`).
- Per-session constants (`CORNEAL_EYE_ROI`, `CORNEAL_DETECT_PARAMS`, ...) into a
  per-session YAML config, out of `constants.py`. This stops the config-drift
  pattern at its source.

## Phase 6 — Guardrails (so it cannot re-rot)

Goal: enforcement — the part every previous cleanup lacked.

- A pre-commit hook / CI check: reject new `sys.path.insert`, reject hardcoded
  `BG_046`/`X:/`, require `pytest` green, require new top-level docs to be
  indexed. Lean on the existing `pre-commit-checker` and `codebase-auditor`
  skills.

## Parallel workstreams (not blocked by the above)

- **Parameter review** — uniform full evidence protocol (literature + data +
  sensitivity) applied to `constants.py`, logged in a decision log. Independent
  of the structural refactor.
- **2D decomposition science** — `analysis_suite/03_population/f_2d_decomposition.py`
  already exists. It is not blocked by the refactor; once Phase 3 stabilizes
  imports it runs on clean infrastructure, but it can be validated/resumed
  sooner.

## Verification gate (every step)

Tests green + snapshot stats CSVs unchanged. A refactor step that changes a
number is a bug, not progress.
