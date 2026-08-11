# D2 — Layering, imports, packaging

Static + executed census of import architecture across `src/`, `scripts/`, `tests/`
(excluding `.venv`, `archive`, `__pycache__`, `.claude`, `_DeepUnitMatch_repo`,
`refactor_baseline`, `_preserved_from_worktrees_20260628`).

- Script: `scripts/audit/d2_layering.py` (`py scripts/audit/d2_layering.py`, exit 0)
- Census CSVs (gitignored; committed with `git add -f`):
  `data/cache/audit/syspath_sites.csv` (`file,line,target,category` — Task 15 input),
  `data/cache/audit/parents_sites.csv`
- Wheel-check evidence: `data/cache/audit/wheel_build_error.txt` (`wheel_contents.txt`
  could not be produced — see the packaging verdict)
- Measurement ids: `d2.*` in `docs/audit/measurements.csv`

## Summary

| Measurement | Value | Baseline | Verdict |
|---|---|---|---|
| `d2.dualroot.src_importers` | 7 | 9 | deviation; explained below |
| `d2.dualroot.mixed` | 6 | 7 | deviation; explained below |
| `d2.syspath.total` | 233 | ~228 | +8 rows are the audit's own footprint; attributed below |
| `d2.syspath.foreign_missing` | 18 | 17 | +1 = foreign *scripts*-dir target; attributed below |
| `d2.layers.upward_module_level` | 3 | 3 | exact match, at the exact predicted lines |
| `d2.importtime.visdetect.analysis.constants` | 9.11 s | ≥ 2 s | confirmed (4.5× the floor) |
| `d2.packaging.wheel_build` | FAIL | (not predicted) | wheel is UNBUILDABLE — supersedes the expected finding |
| `d2.packaging.viz_missing` / `integrations_missing` | absent-a-fortiori | grep exits 0/1 | greps unrunnable; static basis stands |
| `d2.parents.sites` | 4 | — | includes the live `parents[1]` qc bug (qc.py:218) |
| `d2.sideeffects.import` | not-measured | — | recon-evidenced sites spot-verified; enumeration deferred |

## Dual import roots (`d2.dualroot.*`)

Measured 7 files with module-level `src.visdetect` imports — all of them
`scripts/video/*` — of which 6 also import the plain `visdetect` root in the same file
(mixed): `batch_sync_sessions.py`, `characterize_camera_signal.py`,
`corneal_spatial_diagnostic.py`, `poc_multianchor_sync.py`, `select_roi.py`,
`sync_validation_figure.py`. The 7th, `compare_mask_sync.py`, imports the `src.` root
only. A mixed file binds the same source file to two distinct module objects, so classes
compare unequal across the two roots — `isinstance`/pickle identity silently breaks.

**Deviation vs baseline (9/7 → 7/6).** Not drift: no commit has touched `scripts/video/`
since before recon (last was `d54df00`). Every other in-scope occurrence of the token is
non-import text the anchored regex correctly excludes: the compatibility-shim docstrings
`src/visdetect/io.py:1` and `src/visdetect/session.py:1` (which advertise the
`src.visdetect.*` path), the `RenamingUnpickler` comment `core/session.py:203`, the
`migrate_pkls.py:5` docstring, and this audit script's own docstring. Recon's 9/7 most
plausibly came from looser text-matching sweeping ~2 of those mention-files in as
importers (and `compare_mask_sync.py` in as mixed); the anchored per-line import regex
used here is the authoritative census.

## `sys.path` mutation census (`d2.syspath.*`)

233 sites in `data/cache/audit/syspath_sites.csv`: 125 `computed`, 55 `repo-src`,
35 `other-literal`, 18 `foreign-absolute-MISSING` (and zero foreign targets that still
exist — every foreign site is a silent fall-through to whatever ambient `visdetect`
happens to be importable; provenance unverifiable).

**Attribution of the +5 vs ~228** (the audit's own files must never be reported as
drift): 233 = 218 sites in tracked pre-audit files + 7 in unrelated untracked
work-stream scripts (`chronic_feasibility_figure.py:53,54`,
`render_opto_exemplar_figure.py:44`, `characterize_unsolvable_alignment.py:85`,
`validate_event_spike_clock_drift.py:39`, `exemplar_tracking_figure.py:66,67`) + **8
audit-own rows**: `d1_constants_census.py:8`, `d1_executed_checks.py:13`,
`d5_guardrail_count.py:5`, `tests/audit/test_audit_lib.py:5`, and 4 rows from
`d2_layering.py` itself. Caveat for Task 15: of those 4, only `d2_layering.py:7` is a
real `sys.path.insert`; lines 2/30/36 are the census matching its own docstring,
comment, and detector string (classified `computed` by the text fallback) — self-scan
artifacts, left in the CSV as script output. Net of the audit's 8, the tree has 225
sites against an approximate ~228 baseline — within the "~".

**`d2.syspath.foreign_missing` = 18 vs 17**: 17 sites target non-existent foreign `src`
trees exactly as baselined (`vd_tf_bg046/src` ×6, `vd_tf_phase0/src` ×11); the 18th,
`scripts/tf_responsiveness/state_conditioned/combined_figure.py:22`, targets the
non-existent foreign *scripts* dir `vd_tf_bg046/scripts/tf_responsiveness/
state_conditioned` — caught by the census's `vd_tf` substring rule, plausibly outside
recon's foreign-src-tree count. All 18 verified non-existent at scan time.

## Module-level upward layer edges (`d2.layers.upward_module_level`)

Exactly the 3 predicted edges, at the predicted lines:

- `src/visdetect/core/video_sync.py:69 -> visdetect.analysis.constants`
- `src/visdetect/core/video_sync.py:105 -> visdetect.analysis.config`
- `src/visdetect/anatomy/peak_channel.py:10 -> visdetect.analysis.tracking_qc`

Layer order `core(0) < anatomy(1) < analysis(2) < suite(3)`; module level only (lazy
imports excluded by design). `core` and `anatomy` cannot be extracted or tested without
dragging in the full `analysis` stack.

## Import wall-times (`d2.importtime.*`)

Fresh interpreter per module, run in listed order (`sys.executable` = the venv Python):

| Module | Wall-time | `len(sys.modules)` |
|---|---|---|
| `visdetect` | 10.65 s | 1170 |
| `visdetect.core` | 2.19 s | 1170 |
| `visdetect.analysis.constants` | 9.11 s | 1577 |
| `visdetect.suite.loader` | 2.93 s | 1580 |

Prediction `constants ≥ 2 s` confirmed at 9.11 s. Reading order matters: the first
subprocess pays the cold OS file cache (10.65 s); `visdetect.core` (identical
1170-module set) shows the warm floor at 2.19 s. `constants` — a leaf module of plain
numbers — still costs 9.11 s warm because `visdetect.analysis.__init__` chains in ~407
additional modules (sklearn/stats stack) beyond bare `visdetect`'s numpy/scipy/pandas/
matplotlib load. There is no cheap way to import a single threshold constant.

## Packaging (`d2.packaging.*`) — expected finding SUPERSEDED by a stronger one

Expected (plan): the wheel builds; the control grep finds `visdetect/core/` files; the
`visdetect/(viz|integrations)/` grep prints nothing and exits 1 (packages dropped by
`find_packages` for lack of `__init__.py`).

Measured: **`py -m pip wheel . --no-deps -w data/cache/audit/wheel` FAILS — no wheel
exists to grep** (`d2.packaging.wheel_build = FAIL`, evidence
`data/cache/audit/wheel_build_error.txt`; both planned grep exit codes n/a; the wheel
output dir is empty). Failure: `error: package directory 'src\scripts' does not exist`.
Root cause chain: `setup.cfg` declares `packages = find:` with `package_dir = =src` but
has **no `[options.packages.find] where=src` section**, so `find:` scans the repo ROOT,
discovers the tracked top-level `scripts/__init__.py` (present since `b8b0ee0`,
2026-06-18 — pre-audit, so this failure is a repo defect, not audit footprint), and
`package_dir` then maps package `scripts` to the non-existent `src/scripts`.

Consequences, in increasing severity:

1. `visdetect.viz` and `visdetect.integrations` are recorded `absent-a-fortiori`
   (`d2.packaging.viz_missing`, `d2.packaging.integrations_missing`): the static basis
   stands — neither directory has an `__init__.py`, so `find_packages` drops both even
   after a `where=src` fix — and "50 importers break on any non-editable install".
2. It is worse than that: `find:` never looks in `src/` at all, so **no built
   distribution has ever contained ANY `visdetect` package**. The pre-build egg-info
   (18 May 2026) proves it: `top_level.txt` was empty and `SOURCES.txt` lists zero
   `src/visdetect` modules. Before `b8b0ee0` wheels built EMPTY; since `b8b0ee0` the
   build fails outright.
3. The venv works only because the editable install degenerated to
   `__editable__.visdetect-0.0.0.pth` = a bare `src` path injection — path hacking, not
   packaging. The 55 `repo-src` `sys.path.insert` sites (`d2.syspath.total`) are the
   de-facto distribution mechanism. Every non-editable consumer breaks at build time,
   before any import.

## `parents[N]` census (`d2.parents.sites` = 4)

Full listing in `data/cache/audit/parents_sites.csv` — the idiom class that produced the
live `parents[1]` qc-profile no-op (D1): `core/qc.py:218` (`parents[1]` — resolves
`config/` under `src/visdetect/` instead of the repo root; the confirmed bug), plus
three `parents[3]` repo-root computations (`analysis/config.py:78`,
`analysis/state_tf_learning.py:24`, `analysis/tf_labeling.py:25`) that are correct today
and silently wrong after any file move.

## Import side effects (`d2.sideeffects.import` = not-measured)

Full enumeration deferred; recon-evidenced sites spot-verified by grep: module-level
`matplotlib.use("Agg")` ×4 (`analysis/tf_pulse.py:17`, `analysis/unit_selection.py:23`,
`core/qc.py:27`, `suite/plotting.py:9`) plus directory creation at import time
(`os.makedirs`, `suite/config.py:19`). Importing the library mutates global matplotlib
state and the filesystem.

## Step-3 residue accounting

- Pre-build check: `build/` did **not** exist; `src/visdetect.egg-info/` **pre-existed**
  (gitignored, mtime 18 May 2026) — per the task rule it was reported, not deleted.
- The in-tree `egg_info` step of the failed build regenerated 3 of its 5 files
  (`PKG-INFO`, `requires.txt`, `top_level.txt` — the latter now reads `scripts`);
  `SOURCES.txt` and `dependency_links.txt` are byte-identical. Pre-build md5s:
  PKG-INFO `2e4492f2458e3260e05dada208c61c34`, SOURCES.txt
  `74fa225788afb02c9ccf4a26d30460e9`, dependency_links.txt
  `68b329da9893e34099c7d8ad5cb9c940`, requires.txt
  `218d4d7af66d8633d656c8002dcfd28c`, top_level.txt
  `68b329da9893e34099c7d8ad5cb9c940`.
- `build/` was never created (the failure precedes it), so the brief's `rm -rf` had
  nothing to remove; `data/cache/audit/wheel/` exists empty and gitignored (never
  staged; no `.whl` was produced).
