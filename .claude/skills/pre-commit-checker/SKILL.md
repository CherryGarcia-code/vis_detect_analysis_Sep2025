---
name: pre-commit-checker
description: Use when about to commit or when the user asks to review changed files before committing ("pre-commit", "check my changes", "is this safe to commit?", "ready to commit") - a fast quality gate over only the files in git diff, checking event-alignment safety, hardcoded constants that belong in constants.py, normalization correctness, unit selection, session-id canonicalization, and memory cleanup, then returning a BLOCK/WARN/INFO verdict.
---

# Pre-Commit Checker

You are a **Pre-Commit Checker** — a fast, focused quality gate that reviews only the files that have changed since the last commit. You catch issues before they're committed, preventing regressions without the overhead of a full codebase audit.

You are a lightweight version of the **Codebase Auditor**, scoped to changed files only.

---

## Execution Flow

### Step 1: Identify Changed Files

```bash
git diff --name-only HEAD
git diff --name-only --cached
git ls-files --others --exclude-standard
```

Combine all three outputs into a unique set of changed/new files. Filter to only `.py` files.

### Step 2: Run Targeted Checks

For each changed file, run only the checks relevant to that file type:

#### For any analysis `.py` file in `scripts/<topic>/`

| Check | What to Look For |
|-------|-----------------|
| **Alignment safety** | Any `Change_ON` usage without outcome filtering? Any `enforce_valid_outcomes=False`? (`fa`/`abort` trials never saw a change stimulus) |
| **SDT correctness** | Go vs catch from `change_size`, not from the `trialoutcome` label? `trialoutcome == "fa"` never treated as an SDT false alarm? |
| **Hardcoded constants** | `0.025`, `25.0`, `3.0`, `(-0.4, 0.0)`, `(0.0, 0.5)` that should be imported from `visdetect.analysis.constants` |
| **Normalization** | Shared baseline across compared conditions (not per-condition — that's circular)? Normalize-then-average, not average-then-normalize? Divide-by-zero guard on the baseline SD? Cross-neuron magnitude claims FR-normalized? |
| **Session IDs** | Every `session_name` key/join/sort passed through `config.canonical_session_id()`? (int64 silently drops the leading-zero day → day 1-9 sessions vanish from joins) |
| **Color palette** | Ad-hoc hex colors instead of `STAGE_COLORS`, `OUTCOME_COLORS`, `CELLTYPE_COLORS`, `STATE_LABEL_COLORS`? |
| **Session loading** | `load_staging_manifest()` used (not manual filtering)? |
| **Unit selection** | `get_good_cluster_ids()` used (not raw `session.good_cluster_ids`)? |
| **Memory cleanup** | `del sess; gc.collect()` in session loops? |
| **Outputs** | Figure → `FIGURES/<topic>/<SUBJ>/`, cache → `data/cache/<topic>/`, plus a stats CSV? |
| **Existing utilities** | Is the script reimplementing something in `visdetect/analysis/utils.py`? (smooth_psth, bootstrap_ci, permutation_test, fdr_correct, compute_auroc) |
| **Style** | `setup_style()` called before plotting? `save_figure()` used for output? |

⚠️ `analysis_suite/` is archived (`archive/analysis_suite_2026-07-01/`). If the diff **adds** files there or imports from it, that is a BLOCK.

#### For any `.py` file in `src/visdetect/`

| Check | What to Look For |
|-------|-----------------|
| **No circular imports** | Library code must not import from `scripts/` or the archived suite |
| **Constants from canonical source** | No hardcoded values that already exist in `constants.py` |
| **Docstrings** | Public functions have docstrings? |
| **Type hints** | Function signatures have type annotations? |
| **Tests** | New library behavior covered by a test under `tests/`? |

#### For `constants.py` or `config.py` specifically

| Check | What to Look For |
|-------|-----------------|
| **Backwards compatibility** | Were any existing constants renamed or removed? |
| **Downstream impact** | Which scripts import the changed constants? |

### Step 3: Report

```
# PRE-COMMIT CHECK
Files checked: {n}
Issues found: {n}

## Issues
| Severity | File:Line | Issue |
|----------|-----------|-------|

## Clean Files
- {file1} ✓
- {file2} ✓

## Verdict: {PASS / FAIL}
{If FAIL: "Fix the above issues before committing."}
{If PASS: "All checks passed. Ready to commit."}
```

---

## Severity Classification

- **BLOCK** (must fix before commit): Alignment-safety violations (`Change_ON` on `fa`/`abort`), SDT classification errors, circular-baseline normalization, non-canonical session-id joins, new code added to the archived `analysis_suite/`
- **WARN** (should fix, but won't break anything): Hardcoded constants, missing memory cleanup, ad-hoc palettes, outputs written outside the topic layout, style issues
- **INFO** (nice to fix): Missing docstrings, type hints, unused imports

Only BLOCK issues cause a FAIL verdict.

---

## Performance Notes

- This skill should run in under 30 seconds for typical changesets (1-5 files).
- Use Grep with specific file paths (not whole-codebase search).
- Read only the changed files, not the entire codebase.
- Skip checks that can't apply (e.g., don't check alignment in a pure-behavior script).

---

## Trigger Conditions

Activate this skill when:
- User says "check before commit", "pre-commit", "review my changes"
- User is about to commit (says "commit this" or "ready to commit")
- After making changes to multiple files, before committing
- User asks "is this safe to commit?"
