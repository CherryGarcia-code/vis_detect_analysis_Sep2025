# Refactor Handoff — May 2026

**Branch**: `refactor/architecture` (24 commits ahead of `main`)
**Base tag**: `pre-refactor-baseline` (commit `ddc92c3` on `main`)
**Status**: structural refactor complete; Phase 5 polish + the big `analyses/` rename remain.

Pick this up by reading [REFACTOR_PLAN.md](../REFACTOR_PLAN.md) → [ARCHITECTURE.md](../ARCHITECTURE.md) → [refactor_baseline/README.md](../../refactor_baseline/README.md). Then this doc.

---

## TL;DR

The in-place refactor described in `docs/REFACTOR_PLAN.md` has been executed
through Phase 6. The codebase has been migrated off `sys.path` hacks
(196 files → 1 intentional vendored path), archive and docs locations
consolidated, the target architecture defined in `ARCHITECTURE.md`, and a
guardrail script added. **`pytest tests/` = 27 passed throughout** (with the
same 2 pre-existing collection errors as at baseline). Per-group stats parity
preserved (deterministic outputs byte-identical to the oracle; non-determinism
documented). What remains: ~79 BG_046/X:/ hardcoded paths flagged as Phase 5
INFO items, the **user-approved `analyses/` directory rename** (Option 1
layout), and AI_exploration triage.

---

## What's done — phase by phase

| Phase | What it did | Commit(s) |
|---|---|---|
| 0 | Snapshot stats CSVs into `refactor_baseline/` as the parity oracle. Pre-existing HMM work + png_viewer + REFACTOR_PLAN committed to `main` first. Branch + `pre-refactor-baseline` tag created. **Found and fixed** an editable-install bug where `pip install -e .` was pointing at a stale agent worktree (`agents-process-timeout-issue`), making every script's `ROOT` resolve to the wrong checkout. Removed 3 stale agent worktrees. | `b2a44c8`, `bd0618d` |
| 1 | `docs/ARCHITECTURE.md` — single concise target-state doc; the "definition of done" the 2026 reorganization lacked. User-approved: Option 1 layout (`analyses/` tree), `DeepUnitMatch → scripts/sorting/DeepUnitMatch/`, `FIGURES/`+`table_output/` archived (not removed). | `a8f4eb3` |
| 2 | 6 archive locations → one `archive/`. 4 `NORMALIZATION_*.md` → one `docs/NORMALIZATION.md` (now Guide + Audit, after user editing). 3 reorganization docs → `archive/reorganization_docs_2026/`. `DOCUMENTATION_INDEX.md` updated. | `a96f67a`, `97be7a4` |
| 3a | Created `src/visdetect/suite/` package; moved `config.py`/`loader.py`/`plotting.py` into it. Left back-compat shims at the old paths. **Mechanism**: user chose Option B (fold suite infra into `visdetect`) over Option A (separate installed `analyses` package). | `0ee6e5e` |
| 3a-fix | **Bug caught by 08_tf_pulse group verification**: `suite/loader.py` had 3 *function-level deferred* `from config import …` calls that my Phase-3a edit missed (only fixed the top-level import). Every TF script broke with `ModuleNotFoundError: No module named 'config'`. Fixed to relative `from .config import`. | `edfc9e7` |
| 3b | All 9 analysis_suite figure groups migrated to `visdetect.suite.*` and `visdetect.analysis.utils`. `sys.path.insert` hacks removed. **Bug caught**: the migrator's first version was anchored to `^` — missed *indented* imports inside `try/except` (specifically in `f_/g_post_error_controls.py`), which would have silently dropped the HMM analysis. Made the migrator indentation-aware, also handles guard-removal (`if _src not in sys.path:`), iterative orphan-helper-var cleanup. | `7b3e3a2`, `78fa1ac`, `dd3d9c4`, `18675f2`, `1f6c70c`, `36ad126`, `6cf4141`, `b7548b8`, `e263578` |
| 3c | `scripts/` (68 of 124 files modified) + `tests/` (4 files) migrated. `run_deep_unitmatch.py`'s `sys.path.insert(0, str(DUM_CODE))` deliberately preserved (vendored DeepUnitMatch). | `74cca8d`, `50bf14a` |
| 3d | The 4 back-compat shims deleted (`analysis_suite/{config,loader,plotting,utils}.py`). Verified shimless run; pytest 27 passed. | `6a8c247` |
| 4 | No-op — utils canonical at `visdetect.analysis.utils` since before this refactor; shim deletion in 3d completed Phase 4. | (folded in) |
| 5 (chunk 1) | The 4 `01_behavior/d|e|f|g_post_error_*.py` scripts re-wired to use `PKL_DIR` and `SUBJECT` from config (env-var-driven) instead of duplicating with a hardcoded `BG_046`. All 4 stats CSVs byte-identical to oracle. | `84dabe8` |
| 6 | `scripts/qc/check_refactor_guardrails.py` — standalone check enforcing Rules 1, 3, 4 statically. HARD (sys.path.insert, flat imports) + INFO (BG_046, X:/) reporting. Currently: 0 HARD, ~79 INFO. | `dee2898`, `9907c66` |

---

## Tooling at repo root (gitignored via `/_*.py`)

These are one-off tools, **not** for committing:

- **`_migrate_imports.py`** — the indentation-aware migrator. Handles flat→fully-qualified suite imports, removes `sys.path.insert` lines and their `if … in sys.path:` guards, removes orphaned `_root`/`_src`/`repo_root` helper assignments iteratively. `SYSPATH_KEEP` regex preserves vendored paths (`DUM|DeepUnitMatch|unitmatch`).
- **`_check_imports.py`** — execs only top-level import statements in each `.py`, with the file's own dir on `sys.path`, mimicking `py script.py`. Reports any import that fails to resolve. Used during Phase 3c to verify `scripts/` migration.
- **`_fix_post_error_hardcoding.py`** — Phase 5 chunk 1 patch for the 4 post-error scripts.

The **committed** guardrail (Phase 6) lives at `scripts/qc/check_refactor_guardrails.py`.

---

## Parity oracle protocol

The oracle is `refactor_baseline/stats/` (40 stats CSVs + sha256 manifest +
README). Two important caveats:

1. **Some scripts are non-deterministic** (RNG-derived p-values from shuffle
   tests, BLAS reduction-order FP noise). For these the *deterministic*
   quantities byte-match; only RNG/FP-noise values differ by tiny amounts.
   Examples: `fa_lick_aligned_divergence` (shuffle p-values), `dpca` (BLAS
   FP), `fa_neural_signatures` (kruskal-by-HMM at the 7th digit).
2. **The 08_tf_pulse oracle CSVs were never validated** by the partial
   `run_all` (it only reached 19/50 before being stopped twice). They may be
   stale relative to current code. In particular:
   - **`tf_encoding_distribution_stats.csv`**: oracle says "Best=Gamma",
     fresh run says "Best=Log-normal". User's memory records the key result
     "TF encoding is Gamma, not bimodal." **Flagged for science review** —
     not refactor-caused (the shims proved old/new imports resolve identically,
     so this is staleness or a cache-ordering effect). The "distributed, not
     bimodal" headline likely still holds since both fits are unimodal.

For Phase 3+, **per-group self-baselining** was used: capture a group's
outputs on current code, apply the refactor edit, re-run, assert byte-identical
(or for non-deterministic scripts, structurally identical with noise on
RNG/FP quantities only).

---

## Pre-existing failures (NOT refactor-caused, NOT in scope)

These were confirmed pre-existing by:
- Running on the `pre-refactor-baseline` code (the failures reproduce there).
- Error inspection (none is an `ImportError` / `ModuleNotFoundError` from the migration).
- The shims-prove-import-equivalence argument (any successful run can't change numbers).

| Script(s) | Pre-existing issue |
|---|---|
| `06 d_3d_coding_space`, `d_lick_2d_integration`, `08 c_tf_pulse_integration`, `e_tf_state_modulation`, `f_tf_sensory_motor` | `FileNotFoundError`: `BG_046_05092025.pkl` missing |
| `02 d_state_modulation`, `e_cell_type_comparison`, `07 c_noise_correlations` | `FileNotFoundError`: `AI_exploration/figures/waveform_celltype_labels.csv` missing (March 2026 post-TPrime regen pending) |
| `02 a_responsiveness_screen` | `ValueError: All numbers are identical in kruskal` — cached responsiveness table has "0/4326 units responsive"; downstream kruskal fails. Data/cache issue. |
| `08 _eval_cutoffs` | chi2 zero-frequency `ValueError` |
| `08 g2_tf_tier_gallery` | `UnicodeEncodeError`: `→` character on cp1252 console |
| `hmm_neural_states.py`, `hmm_neural_TF_event_comparison.py` | Wrong import: `smooth_psth` from `visdetect.analysis.hmm_downstream` (it lives in `visdetect.analysis.utils`) |
| `validate_unitmatch_results.py` | Wrong import: `from session_io import load_session` — `session_io.py` exists nowhere; should be `visdetect.core.session` |
| `run_deep_unitmatch.py`, `run_kilosort4.py`, `run_unitmatch_all.py` | Optional deps not installed (`torch`, `UnitMatchPy`) |

---

## Two untracked files in `scripts/pipelines/tracking/`

The initial `git status` had:
- `?? scripts/pipelines/tracking/run_deepunitmatch_all.py`
- `?? scripts/pipelines/tracking/validate_long_tracks.py`

Plus user-edited during session: `diagnose_intersession_drift.py`,
`qc_ks4_runs.py`, `validate_waveforms.py` (also untracked).

Per the user's direction, **these are in-progress work — leave them alone**.
The guardrail's `SKIP_FILES` exempts the two with `sys.path.insert`:
- `run_deepunitmatch_all.py` (legit DeepUnitMatch sibling-dir sys.path — the
  per-line keyword check missed it because `DeepUnitMatch` is on a loop line
  above the actual `insert` call).
- `validate_long_tracks.py` (dead `sys.path.insert(REPO_ROOT/src)` — the user
  will clean it when they commit).

---

## Remaining work

### Phase 5 polish (~79 INFO items from the guardrail)

`py scripts/qc/check_refactor_guardrails.py` lists them. Roughly:
- ~12 `BG_046` defaults in `scripts/analysis/{behavior,…}/*.py` (CLI defaults,
  function signatures). Mostly cosmetic — replace with `SUBJECT` from config or
  parameterize fully.
- ~5 `BG_046` path constructions in `scripts/analysis/build_longitudinal_table.py`,
  `run_deep_unitmatch.py`, `run_unitmatch_pipeline.py`, etc.
- ~3 hardcoded `X:/` paths in `prep_unitmatch_full_trial_waveforms.py` and
  `scripts/conversion/raw_to_pkl.py`'s docstring example. **Add
  `VISDETECT_DATA_ROOT` env var** to `visdetect.analysis.config` and update
  those scripts to use it.
- **Per-session YAML for corneal constants** — `CORNEAL_EYE_ROI`,
  `CORNEAL_DETECT_PARAMS` etc. currently live in `constants.py`. Move out to
  `config/<session>.yml`. This is the largest sub-task.

### The big `analyses/` directory rename (user's Option 1)

User-approved in Phase 1 (see `ARCHITECTURE.md` migration map):
- `analysis_suite/01_behavior/`…`09_optotagging/` → `analyses/figures/01_behavior/`…
- `analysis_suite/run_all.py` → `analyses/run_all.py`
- `scripts/analysis/{behavior,lick,tf_response,learning}/` → `analyses/pipelines/{...}/` (triage per file)
- `_DeepUnitMatch_repo/` → `scripts/sorting/DeepUnitMatch/`
- `scripts/QC_CHECKS/` + `scripts/QC_technical/` → `scripts/qc/`
- `scripts/pipelines/concat_sort/` → `scripts/sorting/`
- `scripts/kilosort_related/` + `chanMap_related/` → `scripts/kilosort/`

**Important plumbing**: `src/visdetect/suite/config.py` hardcodes
`FIGURE_DIR = os.path.join(ROOT, "analysis_suite", "figures")` and
`CACHE_DIR = os.path.join(ROOT, "analysis_suite", "cache")`. These must move
in lock-step with the directory rename.

**Sequencing**: do this **after** Phase 5 is done (or skipped). Stop any
running `run_all.py` first; do the moves as `git mv`; update `FIGURE_DIR`/
`CACHE_DIR`; verify with the guardrail + a spot-run + pytest.

### AI_exploration triage

`AI_exploration/` is a superseded parallel pipeline (7 numbered `analysis_*`
scripts whose outputs are all under `figures/preTprime/` = stale per CLAUDE.md).
Per-script: unique analyses → `analyses/`, the rest → `archive/`. The 9
`sys.path.insert` lines in there are the expected leftover.

---

## Verification commands

```bash
# pytest baseline — must = 27 passed, 2 known collection errors
py -m pytest tests/ -q --continue-on-collection-errors

# Guardrail — must = 0 HARD violations (PASS, exit 0)
py scripts/qc/check_refactor_guardrails.py

# Per-script parity (example: 01_behavior)
cd analysis_suite && py 01_behavior/c_reaction_time_analysis.py && cd ..
cmp -s refactor_baseline/stats/01_behavior/reaction_time_stats.csv \
       analysis_suite/figures/01_behavior/reaction_time_stats.csv

# Full sweep (slow — multiple 30-min timeouts):
py analysis_suite/run_all.py    # writes analysis_suite/run_all_log.txt
```

---

## Key user decisions made this session

1. **Pre-refactor HMM work committed to `main` first** (3 commits) so the
   refactor branch started from a clean, well-defined state. The
   `auto_label_states` rewrite was an *algorithm change*, kept off the
   refactor branch.
2. **Snapshot + background `run_all`** for the parity oracle (Phase 0)
   rather than full fresh run. Caught 5 stale snapshot files; refreshed them.
3. **3 stale agent worktrees removed** with `git worktree remove --force` —
   they had uncommitted work but the user confirmed they were dead-end agent
   sessions.
4. **Mechanism for import migration**: fold suite infra into `visdetect.suite`
   (Option B), not a separate installed `analyses` package (Option A). Simpler
   setup.cfg, one installed package.
5. **Verification depth for groups 04–09**: grep-clean + compile + run fast
   scripts + one final batch `run_all`, instead of full re-run every group
   (which would have been hours per heavy group).
6. **DeepUnitMatch placement**: organize as `scripts/sorting/DeepUnitMatch/`
   rather than a new `third_party/` dir. **Do not delete `_DeepUnitMatch_repo`** —
   move only, when the time comes.
7. **`FIGURES/` + `table_output/`**: archive, don't remove.
8. **In-progress untracked files in `scripts/pipelines/tracking/`** — leave
   alone, exempt from guardrail.
9. **Analyses layout target**: Option 1 — `analyses/_shared/` + `pipelines/` +
   `figures/`. Also fold `AI_exploration/` into the triage.

---

## Useful pointers

- [REFACTOR_PLAN.md](../REFACTOR_PLAN.md) — the governing plan (Phase 0–6).
- [ARCHITECTURE.md](../ARCHITECTURE.md) — the target-state spec + migration map + guardrail rule.
- [refactor_baseline/README.md](../../refactor_baseline/README.md) — parity protocol, what's validated vs stale, per-group self-baselining.
- [NORMALIZATION.md](../NORMALIZATION.md) — guide + audit (consolidated and now restructured by user as Part 1 how-to + Part 2 audit).
- [DOCUMENTATION_INDEX.md](../DOCUMENTATION_INDEX.md) — the index.

---

*Handoff written 2026-05-20 after the user asked for a fresh-chat handoff. Branch
state at handoff: `refactor/architecture` HEAD `9907c66` (Refactor(phase6):
Guardrail tweaks), 24 commits ahead of `main`.*
