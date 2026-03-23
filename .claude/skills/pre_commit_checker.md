# Skill: Pre-Commit Checker

## Identity & Purpose

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

#### For any `.py` file in `analysis_suite/`

| Check | What to Look For |
|-------|-----------------|
| **Alignment safety** | Any `Change_ON` usage without outcome filtering? Any `enforce_valid_outcomes=False`? |
| **Hardcoded constants** | `0.025`, `25.0`, `3.0`, `(-0.4, 0.0)`, `(0.0, 0.5)` that should be imported from `constants.py` |
| **Color palette** | Ad-hoc hex colors instead of `STAGE_COLORS`, `OUTCOME_COLORS`, etc. |
| **Session loading** | `load_staging_manifest()` used (not manual filtering)? |
| **Unit selection** | `get_good_cluster_ids()` used (not raw `session.good_cluster_ids`)? |
| **Memory cleanup** | `del sess; gc.collect()` in session loops? |
| **Figure number** | Docstring fig number matches `save_figure()` filename? No collisions with other scripts? |
| **Existing utilities** | Is the script reimplementing something in `utils.py`? (smooth_psth, bootstrap_ci, etc.) |
| **Style** | `setup_style()` called before plotting? `save_figure()` used for output? |

#### For any `.py` file in `src/visdetect/`

| Check | What to Look For |
|-------|-----------------|
| **No circular imports** | Library code should not import from `analysis_suite/` |
| **Constants from canonical source** | No hardcoded values that exist in `constants.py` |
| **Docstrings** | Public functions have docstrings? |
| **Type hints** | Function signatures have type annotations? |

#### For `run_all.py` specifically

| Check | What to Look For |
|-------|-----------------|
| **All scripts registered** | Compare SCRIPTS list against files on disk |
| **Figure numbers sequential** | No gaps, no duplicates |
| **Labels match scripts** | Each label corresponds to the correct script path |

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

- **BLOCK** (must fix before commit): Alignment safety violations, SDT classification errors, figure number collisions
- **WARN** (should fix, but won't break anything): Hardcoded constants, missing memory cleanup, style issues
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
