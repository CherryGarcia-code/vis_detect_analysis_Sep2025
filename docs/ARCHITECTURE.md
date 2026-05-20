# ARCHITECTURE.md — Target Architecture

**Status**: target state for the in-place refactor (`docs/REFACTOR_PLAN.md`).
**Created**: 2026-05-16 (Phase 1).

This document is the **definition of done**. Every refactor step migrates the
codebase toward the layout and rules below. The 2026 reorganization failed
because it had no defined end-state; this is that end-state, plus the
enforcement (Phase 6) that keeps it from re-rotting.

Scope reminder: this is a **behavior-preserving** refactor. Nothing here
changes an algorithm or a parameter value. Only *where code lives* and *how it
is imported and configured* changes.

---

## Target directory layout

```
vis_detect_analysis/                  repo root
│
├── src/visdetect/          ← THE LIBRARY. Installable, importable, no scripts.
│   ├── core/                 session, io, qc, kilosort, ingest
│   ├── analysis/             reusable analysis logic (behavior, align, hmm,
│   │                         tf_pulse, lick, optotagging, constants, config …)
│   ├── viz/                  plotting primitives (set_style, despine)
│   ├── utils/                THE single utilities location
│   └── integrations/         third-party bridges (UnitMatch, …)
│
├── analyses/               ← ALL analysis/visualization scripts (consolidated).
│   ├── _shared/              suite infrastructure: config, loader, plotting
│   ├── pipelines/            scripts that PRODUCE intermediate data products
│   │   ├── behavior/           (HMM fitting, behavioral pipelines)
│   │   ├── lick/               (lick-responsive neuron detection)
│   │   ├── tf_response/        (TF pulse screening, splitter extraction)
│   │   └── learning/           (learning-dynamics analyses)
│   ├── figures/              scripts that PRODUCE publication figures
│   │   ├── 01_behavior/  02_single_unit/  03_population/  04_decoding/
│   │   ├── 05_longitudinal/  06_lick_motor/  07_advanced/  08_tf_pulse/
│   │   └── 09_optotagging/
│   └── run_all.py            sequential runner for the figure pipeline
│
├── scripts/                ← OPERATIONAL tooling ONLY. No analyses, no figures.
│   ├── conversion/           raw→pkl, mat→pkl (+ matlab_scripts/)
│   ├── batch_processing/     batch converters
│   ├── data_management/      data organization tools
│   ├── kilosort/             Kilosort/channel-map tooling
│   ├── sorting/              concatenated-sort pipeline (was pipelines/concat_sort)
│   │   └── DeepUnitMatch/      vendored UnitMatch repo (was _DeepUnitMatch_repo/)
│   ├── qc/                   session & unit QC diagnostics
│   ├── video/                video-sync tooling
│   ├── tf_labeling/          TF manual-labeling GUI + raster precache
│   └── utils/                operational utilities (e.g. png_viewer)
│
├── tests/                  pytest suite
├── config/                 QC profiles + per-session YAML config
├── data/                   pkls, caches, manifests, labels   (git-ignored bulk)
├── docs/                   documentation (see Docs policy)
├── notebooks/              exploratory notebooks
└── archive/                THE single archive. Nothing here is imported live.
```

---

## Rules

### 1. Installable package — no `sys.path` hacks
- `pip install -e .` (backed by `setup.cfg`) is the **only** supported setup.
- `import visdetect...` must work from anywhere with no path manipulation.
- **Zero `sys.path.insert` calls** in the codebase. (Baseline: 196 files.)
- `analyses/_shared/` is an importable package, not a flat `sys.path` module.

### 2. One library
- All reusable logic lives in `src/visdetect/`. The library imports nothing
  from `analyses/` or `scripts/` — dependency flow is one-directional:
  `scripts/` → `analyses/` → `src/visdetect/`.

### 3. One utilities location
- `src/visdetect/utils/` is the only utilities home.
- `analysis_suite/utils.py` (currently a deprecated shim) is deleted; its
  functions live in the library. No parallel utils copy anywhere.

### 4. One analyses tree
- `analyses/` is the single home for analysis and visualization scripts.
- `pipelines/` produce intermediate data; `figures/` produce publication
  figures; `_shared/` is the common infrastructure.
- `scripts/` holds **operational tooling only** — if it makes a figure or a
  scientific result, it belongs in `analyses/`.

### 5. No hardcoded subject or paths
- Subject (`BG_046`) and data roots (`X:/...`) come from config + environment
  variables: `VISDETECT_SUBJECT`, `VISDETECT_DATA_ROOT`.
- Per-session constants (`CORNEAL_EYE_ROI`, `CORNEAL_DETECT_PARAMS`, …) live in
  a per-session YAML under `config/`, not in `constants.py`.
- `constants.py` keeps only true scientific constants (bin sizes, thresholds,
  event windows).

### 6. One archive
- `archive/` is the only archive. All legacy locations
  (`scripts/scripts_archive/`, `scripts/pipelines/archive/`,
  `src/visdetect/analysis/archive/`, `analyses/**/_archive/`, `AI_exploration/`,
  `FIGURES/preTprime/`) are merged into it or deleted (git history preserves
  them either way).
- Nothing in `archive/` is imported by live code — enforced in Phase 6.

### 7. Docs policy
- `docs/DOCUMENTATION_INDEX.md` is the single map of all documentation.
- Every top-level doc must be listed in the index (enforced in Phase 6).
- The 4 `NORMALIZATION_*.md` files are consolidated into one
  `docs/NORMALIZATION.md`. The 3 reorganization docs are archived (superseded
  by `REFACTOR_PLAN.md`).

---

## Migration map (current → target)

| Current location | Target |
|---|---|
| `analysis_suite/01_*`…`09_*` | `analyses/figures/01_*`…`09_*` |
| `analysis_suite/{config,loader,plotting}.py` | `analyses/_shared/` |
| `analysis_suite/utils.py` (shim) | deleted → `src/visdetect/utils/` |
| `analysis_suite/run_all.py` | `analyses/run_all.py` |
| `scripts/analysis/{behavior,lick,tf_response,learning}/` | `analyses/pipelines/*` (triaged) |
| `scripts/tf_response/` (orphan dup) | deleted |
| `AI_exploration/` (7 analyses, all preTprime) | triaged: unique → `analyses/`, rest → `archive/` |
| `scripts/kilosort_related/`, `chanMap_related/` | `scripts/kilosort/` |
| `scripts/QC_CHECKS/` + `scripts/QC_technical/` | `scripts/qc/` |
| `scripts/pipelines/concat_sort/` | `scripts/sorting/` |
| `_DeepUnitMatch_repo/` | `scripts/sorting/DeepUnitMatch/` |
| `scripts/scripts_archive/`, `scripts/pipelines/archive/`, `src/visdetect/analysis/archive/` | `archive/` |
| `FIGURES/`, `table_output/` (stale outputs) | `archive/` |
| 4× `NORMALIZATION_*.md` | `docs/NORMALIZATION.md` |
| 3× reorganization docs | `archive/` |

Triage = per-file decision: a script that is live and unique migrates; a script
superseded by a current equivalent is archived. Triage decisions are logged.

---

## Out of scope (explicitly)

- No algorithm changes. No parameter-value changes. A refactor commit that
  changes a number in any `*_stats.csv` is a bug — see `refactor_baseline/`.
- Parameter review and the 2D-decomposition science run as **separate parallel
  workstreams**, not part of this structural track.

## Verification gate

After every refactor step: `pytest tests/` green (27 passed + the 2 known
pre-existing collection errors) and the `refactor_baseline/` stats CSVs
reproduced bit-for-bit. See `refactor_baseline/README.md`.

## Guardrail (Phase 6)

`scripts/qc/check_refactor_guardrails.py` enforces Rules 1, 3, 4 statically:
no new `sys.path.insert` in maintained code, no flat
`from config/loader/plotting/utils import …`. It also reports (non-blocking)
the remaining hardcoded `BG_046` / `X:/` paths — the Phase 5 worklist. Run
from repo root; exit 1 on any hard violation. Wire into pre-commit or CI as
desired; Claude's `pre-commit-checker` skill can invoke it.
