# D5 — Tests and tooling

Empirical audit of the test suite and the (absence of) quality tooling: a
census of every test file partitioned by real-data dependency, a mapping of
which `visdetect` library modules any test imports, a timed execution of the
offline partition, and a verification of what actually gates a commit in this
repo. Inventory-and-run only: **no test file was modified** — including the
two that abort collection and the one that fails by design.

- Script: `scripts/audit/d5_test_inventory.py`
  (`py scripts/audit/d5_test_inventory.py`, exit 0)
- Supplement: `scripts/audit/d5_test_inventory_ast.py`
  (`py scripts/audit/d5_test_inventory_ast.py`, exit 0) — AST-corrected
  untested-module count (see the blind-spot section)
- Partition CSV (gitignored; committed with `git add -f`):
  `data/cache/audit/test_partition.csv`
  (`test_file,needs_real_data,covers_modules`)
- Measurement ids: `d5.tests.*`, `d5.guardrail.*` in
  `docs/audit/measurements.csv`

## Summary

| Measurement | Value | Notes |
|---|---|---|
| `d5.tests.total` | 98 | test files under `tests/`, excluding the audit's own (`tests/audit/test_audit_lib.py`, 1 file) |
| `d5.tests.need_real_data` | 14 | literal-string heuristic: `load_session\|.pkl\|staging_manifest\|PKL_DIR\|data/cache` |
| `d5.tests.offline_runtime_s` | 630 | wall time; `1 failed, 654 passed, 4 deselected, 6 warnings, 2 errors in 627.46s` — see below |
| `d5.tests.realdata_runtime_s` | not-measured | running the real-data tier is out of audit scope; file list in the partition CSV |
| `d5.tests.untested_modules` | 32 | the shipped regex count — **overcount, see the blind-spot section** |
| `d5.tests.untested_modules_ast` | 14 | AST-corrected count (`scripts/audit/d5_test_inventory_ast.py`) |
| `d5.guardrail.before` | 1,590 | Task-2: HARD guardrail violations before fix |
| `d5.guardrail.after` | 220 | Task-2: real HARD violations after `.claude/` excluded (recon predicted ~218) |

## Partition: 84 offline / 14 real-data (98 files)

By directory (offline + real-data = total):

| Directory | Offline | Real-data | Total |
|---|---|---|---|
| `tests/` (root) | 18 | 8 | 26 |
| `tests/analysis/` | 31 | 3 | 34 |
| `tests/anatomy/` | 15 | 0 | 15 |
| `tests/conversion/` | 0 | 1 | 1 |
| `tests/core/` | 1 | 0 | 1 |
| `tests/scripts/` | 4 | 0 | 4 |
| `tests/suite/` | 4 | 0 | 4 |
| `tests/tracking_dant/` | 4 | 1 | 5 |
| `tests/video/` | 7 | 1 | 8 |
| **Total** | **84** | **14** | **98** |

The 14 `needs_real_data=True` files: `test_align.py`, `test_qc.py`,
`test_repair_trial_event_alignment.py`, `test_run_alignment_realdata.py`,
`test_session_io.py`, `test_staging.py`, `test_state_calibration.py`,
`test_state_labeling.py`, `analysis/test_decision_latents_generative.py`,
`analysis/test_lick_channels.py`, `analysis/test_tf_glm_data_session.py`,
`conversion/test_backfill_stim_phase.py`, `tracking_dant/test_curate_dant.py`,
`video/test_subject_pkl_resolver.py`. Full 98-row table:
`data/cache/audit/test_partition.csv`.

**Partition caveat — the heuristic is a literal-string grep, and it leaks in
both directions.** The known-RED `test_session_id_csv_integrity.py` landed
**offline** (`needs_real_data=False`) because it builds its `data/` paths from
`Path` objects rather than naming any pattern literal — yet at runtime it scans
every session-id CSV under the live `data/` tree (it self-skips via
`skipif(not _DATA.exists())` on a data-less checkout). Conversely, a file that
merely mentions `.pkl` while writing synthetic pickles to `tmp_path` counts as
real-data. "Offline" here means "the file survives without the data tree",
not "hermetic".

## Offline run: the exact command aborts at collection

The briefed command —
`py -m pytest $OFFLINE -q -m "not slow"` over the 84 offline files — does
**not run a single test**:

```
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!
4 deselected, 2 errors in 10.18s
```

Two orphaned test files import modules that no longer exist:
`tests/test_coding_direction.py` → `visdetect.analysis.coding_direction` and
`tests/test_population.py` → `visdetect.analysis.population`. Both modules were
deleted on **2026-02-02** (`4f56700` "Refactor: Prune obsolete pipelines…");
their tests have been left behind for six months. Because pytest's default
collection-error behavior is to interrupt, and both files sit inside the
configured `testpaths = tests` (`setup.cfg`), **the repo's default `py -m
pytest` invocation cannot complete collection at all** — anyone "running the
tests" plainly gets zero test results. The audit did not modify or delete the
orphans (finding, not repair).

Disclosure: in both runs a `tee` was inserted before the brief's `tail -2` so
the full pytest log survives for honest failure reporting — flags, ordering,
and timing untouched.

Re-run with `--continue-on-collection-errors` (added solely to obtain the
runtime the spec asks for; all other flags identical):

```
FAILED tests/test_session_id_csv_integrity.py::test_no_stripped_session_ids_in_csv_deliverables
ERROR tests/test_coding_direction.py
ERROR tests/test_population.py
1 failed, 654 passed, 4 deselected, 6 warnings, 2 errors in 627.46s (0:10:27)
offline_runtime_s=630
```

- **Wall time 630 s (~10.5 min)** for 654 passing tests; the 4 deselected are
  the `slow`-marked real-DDM fits (`setup.cfg` marker, `-m "not slow"`).
- **The single failure is the known RED tripwire** — expected, by design. Its
  failure message names exactly the 6 offender caches with exactly 15,802
  stripped rows (6,496 + 4,958 + 4,322 + 12 + 9 + 5 under
  `data/cache/behavior/`), byte-for-byte the Task-7 finding
  (`d4.ids.integrity_test_red` in `docs/audit/04-artefacts.md`). The test was
  deliberately left red; fixing data is out of scope for the audit.
- `tests/analysis/test_decision_latents*.py` (restored to HEAD earlier in the
  audit session) collected and **passed** — no residual concern.
- 6 warnings, all pre-existing deprecations/empty-slice warnings
  (`np.trapz`, `nanmean` of empty slice), none introduced by the audit.

The real-data tier (14 files) was **not executed**
(`d5.tests.realdata_runtime_s = not-measured`): it requires the real pkl/cache
trees and is out of audit scope; the file list above and the trigger-pattern
column in the partition CSV say exactly what it would need.

## Untested modules: shipped 32, true count 14 (regex blind spot)

The shipped inventory reports **32** library modules with zero test coverage
(`d5.tests.untested_modules`). **This is a material overcount.** The brief's
import-mapping regex (already twice-refined — its ROUND-2 comments fix
`from PKG import name` crediting and the blanket-prefix direction) cannot
match the *parenthesised multi-line* import style:

```python
from visdetect.analysis.kernel_width import (
    grid_fwhm, interpolated_fwhm, temporal_spread, peak_lag,
)
```

After `import ` the regex requires `([\w,\s]+)` — but the next character is
`(`, so the whole statement matches nothing and the file gets **zero credit**.
An `ast.parse` probe (ground truth for import statements; committed as
`scripts/audit/d5_test_inventory_ast.py`, same test set and same
exact-or-descendant crediting rule as the shipped census) finds the regex
under-credits **67 of 98** test files and that **18 of the 32 "untested"
modules do have test imports** — several with dedicated same-named test files
(`test_kernel_width.py`, `test_spectrum_stats.py`, `test_waveform_celltype.py`,
`test_tracking_registry.py`, `test_track_verdict.py`, `test_hmm_validation.py`,
`test_state_calibration.py`, `test_qc.py`, `test_align.py`,
`test_behavior.py`, …).

Falsely "untested" under the shipped regex (18): `analysis.align`,
`analysis.behavior`, `analysis.hmm_downstream`, `analysis.hmm_validation`,
`analysis.kernel_width`, `analysis.optotagging`, `analysis.spectrum_stats`,
`analysis.state_calibration`, `analysis.tf_glm_data`, `analysis.track_verdict`,
`analysis.tracking_registry`, `analysis.utils`, `analysis.waveform_celltype`,
`anatomy.localize`, `anatomy.orientation`, `anatomy.stereotaxic`, `core.qc`,
`suite.config` (all `visdetect.`-prefixed).

**Truly untested — no test file imports them at all
(`d5.tests.untested_modules_ast` = 14):**

| Module | Note |
|---|---|
| `visdetect.analysis.evidence_learning_io` | |
| `visdetect.analysis.lick` | FA lick-responsive detection — CLAUDE.md-named module, zero tests |
| `visdetect.analysis.state_tf_learning` | |
| `visdetect.analysis.su_analysis` | single-unit QC tables / rasters — zero tests |
| `visdetect.analysis.tf_labeling` | |
| `visdetect.analysis.unit_selection` | unit-selection logic — zero tests |
| `visdetect.core.kilosort` | |
| `visdetect.core.spikeglx` | |
| `visdetect.integrations.bombcell_wrapper` | |
| `visdetect.io` | legacy shim |
| `visdetect.session` | legacy shim |
| `visdetect.suite.plotting` | |
| `visdetect.utils.progress` | |
| `visdetect.viz.plotting` | |

Both numbers are recorded; **cite the AST-corrected 14** for any coverage
claim. Caveat on both: "imports the module" ≠ "meaningfully tests it" — this
is an import census, not a coverage measurement (no line/branch coverage was
run). The measurement is also import-statement-relative: a module exercised
only through a script wrapper or `sys.path` manipulation would still show as
untested (none of the 14 was observed to be such a case, but the census does
not rule it out).

## Guardrail numbers (Task 2, restated for the tooling picture)

The one purpose-built quality tool the repo does have is
`scripts/qc/check_refactor_guardrails.py` (manual invocation only):

- **Before fix: 1,590 HARD violations** (`d5.guardrail.before`) — the checker
  was counting its own exclusion list and `.claude/` worktree copies.
- **After `.claude/` excluded: 220 real HARD violations**
  (`d5.guardrail.after`; recon predicted ~218).

## De-facto gate: what actually stands between an edit and `main`

Verified in this checkout (2026-08-12/13), not just asserted from recon:

| Gate class | Finding |
|---|---|
| CI | **None.** `.github/` holds only `copilot-instructions.md` — no `workflows/`, no other CI config found |
| Linters / formatters | **None configured.** No `.pre-commit-config.yaml`, no `pyproject.toml`, no ruff/flake8/tox config; `setup.cfg` holds only metadata + `[tool:pytest]`. A single historical `ruff --fix` commit (`0dde91d`) left no surviving config |
| Native git hooks | **Zero.** `.git/hooks/` contains only the 13 stock `*.sample` files — nothing runs on commit or push |
| Claude Code hooks | **Exactly one**, and it is not a quality gate: `.claude/settings.json` registers a single `PreToolUse` hook on `Bash\|PowerShell` → `.claude/hooks/guard_recursive_delete.ps1` — the junction-scanning **delete guard** (data-loss protection after the Jun-7 worktree incident). It never sees an edit or a commit |
| Tests | **Manual only** — and the default `py -m pytest` invocation aborts at collection on the 2 orphaned test files (above), so even the manual gate yields zero test results unless the invoker knows to add `--continue-on-collection-errors` or deselect the orphans |

The recon claim ("zero CI/linters/pre-commit; sole hook = delete guard") is
**confirmed exactly**. The de-facto gate on this repo is: the committer's own
discipline, plus one delete guard that protects data, not code.
