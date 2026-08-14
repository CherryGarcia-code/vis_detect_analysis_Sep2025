# Cold list — modules to port on first use (ADR-020)

ADR-020's port sunset is **lazy porting**: "every old module starts on `cold-list.md` and is
ported **on first use** through the ADR-009 gate. 'Done' = foundation + gates complete; the
cold-list may be nonempty forever; more than ~12 modules pulled across before the foundation
stabilises is an alarm, not a plan."

This file is the **seed**, derived rather than asserted. A module is **hot** if any script under a
currently-live analysis line's subtree imports it, or if a hot module imports it transitively;
everything else is **cold**.

- Script: `scripts/audit/d8_module_classifier.py` (`py scripts/audit/d8_module_classifier.py`,
  exit 0)
- Census CSV (gitignored; committed with `git add -f`): `data/cache/audit/cold_list_seed.csv`
  (`module, status, live_lines`)
- Measurement id: `d8.coldlist.modules` = **26** cold of **64** classified
  (`d8.modules.classified`)

**The five live lines, and the subtrees that define them:**

| Line | Subtrees |
|---|---|
| early-lick / QC1 | `scripts/analysis/behavior`, `scripts/QC_technical`, `scripts/qc` |
| camera | `scripts/video` |
| population-field | `scripts/population_field` |
| state labeling | `scripts/state_labeling`, `scripts/state_dynamics`, `scripts/session_sorting` |
| tf-glm | `scripts/tf_responsiveness` |

---

## ⚠️ Cold ≠ dead, and one row proves it

**`visdetect.core.ingest` is cold — and it is the module the owner's entire re-ingest plan
depends on.** It reads cold because no script in the five live lines imports it: its caller is
`scripts/conversion/raw_to_pkl.py`, which belongs to no analysis line. It is nevertheless the
module that carries `keep_all_good` (`ingest.py:415`, branch at `:492-495`) — the mechanism that
**dissolves register entry E1**, the irreversible ingest-time QC gate.

Read this file as *"not pulled by a live analysis line"*, never as *"droppable"*. Things that are
droppable are in `drop-list.md`, and only two modules appear in both (`visdetect.io`,
`visdetect.session` — the 7-line shims with **0** importers, `d3.shim_importers`).

Sub-project 1 ports `core.ingest`, `core.io`, `core.session`, `core.qc` and the identity/constants
layer **regardless of temperature** — the foundation is not scheduled by this list.

---

## The 26 cold modules

`register_entries` is from `data/cache/audit/module_register_map.csv`; `clean` there means
"matches no defect symbol pattern", **not** "defect-free" (see the register's *Module coverage*
caveats — the classifier has no pattern for any of the five ephys entries).

| Module | Register entries | Port note |
|---|---|---|
| `visdetect.analysis.decision_latents_enginec` | clean | B8 engine variant; the live line is `analysis.decision_latents`, which is hot |
| `visdetect.analysis.decision_latents_generative` | id-corruption | uses a canonicaliser (mitigation); `slow`-marked real-DDM tests are among the 4 deselected in the offline run |
| `visdetect.analysis.neural_latents` | session-order | N1 line — recorded as a controlled NEGATIVE; port only if that question reopens |
| `visdetect.analysis.psychophysical_kernel` | clean | holds the **documentary source of truth** for `dt = 0.05` (`:18`, "Never 0.25") — register entry 2's only written justification. Port the *statement* into the new constants layer even if the module stays cold |
| `visdetect.analysis.state_provider` | id-corruption; state-tags | reads `data/cache/state_tags` — one of the four irreplaceable hand-label sets (`d7.handlabels.exposure`, 202 files / 29.4 MB, **0 tracked**) |
| `visdetect.analysis.state_tf_learning` | stale-tf-registries | B9 line; **truly untested** (`d5.tests.untested_modules_ast`) and downstream of register entry 5 |
| `visdetect.analysis.tf_labeling` | id-corruption | hand-label GUI backend; its 4,725-unit CSV is hand labour no code regenerates (`d7.handlabels.exposure`, 1 file / 1.5 MB, **0 tracked**). **Truly untested** |
| `visdetect.analysis.track_curation` | clean | curation CLI backend; on `main` per the tracking memory record |
| `visdetect.analysis.tracking_registry` | id-corruption | ADR-018 replaces "tracked unit" with a registry **table**, so port as a *design input*, not as code |
| `visdetect.analysis.unit_selection` | alignment-QC1; **qc-profile-noop** | **Truly untested**, and it is the amplifier of register entry 1 (`:249-250`, `used_params.update({})`). Do not port the `update`-on-empty pattern |
| `visdetect.analysis.waveform_celltype` | clean | cell-type labels; register entry **E2** (composition drift) applies to everything it produces, and ADR-019 makes `celltype_label_source` a required ledger field |
| `visdetect.anatomy.atlas` | clean | CCF localization stack — 6 modules, all cold |
| `visdetect.anatomy.localize` | clean | ⚠️ hemisphere = LEFT for BG_046 (memory `anatomy_localization_jun2026`) |
| `visdetect.anatomy.orientation` | clean | |
| `visdetect.anatomy.peak_channel` | clean | one of the **3 upward layer edges** (`anatomy → analysis.tracking_qc`, `d2.layers.upward_module_level`); porting it drags in `analysis` unless the edge is cut. Also inherits register entry **E3** |
| `visdetect.anatomy.stereotaxic` | clean | |
| `visdetect.anatomy.tracks` | clean | |
| **`visdetect.core.ingest`** | clean | **NOT cold in importance** — see the warning above. Register entry **E1** lives here |
| `visdetect.core.io` | clean | `.mat` loading with the h5py fallback; superseded in spirit by ADR-015's `SessionStore` |
| `visdetect.core.kilosort` | clean | waveform attachment; reads `templates.npy` only (`:42-49`) — part of the no-`.ap.bin` chain that makes E1 dissolvable. **Truly untested** |
| `visdetect.core.spikeglx` | clean | **Truly untested**. If register entry 6 / quarantine **Q6** resolves toward re-extracting NI from raw, this is where that code belongs |
| `visdetect.integrations.bombcell_wrapper` | clean | **Truly untested**; `visdetect.integrations` has **no `__init__.py`**, so it is dropped by `find_packages` even after the packaging fix (register entry A12) |
| `visdetect.io` | clean | 7-line legacy shim, **0 importers** — also on `drop-list.md` §2.2 |
| `visdetect.session` | clean | 7-line legacy shim, **0 importers** — also on `drop-list.md` §2.2 |
| `visdetect.suite.unit_table_schema` | clean | the `suite` layer is archived in spirit (`analysis_suite/` went to `archive/` on 2026-07-01); ADR-015 replaces it |
| `visdetect.utils.synthetic` | alignment-QC1 | `make_synthetic_session()` — **port early despite being cold**: ADR-020's CI tier runs against a *synthetic golden mini-session* because the repo is public and no real unpublished data may be committed. This module is the seed of that fixture |

## The 38 hot modules, and by which line

Not a port order — an exposure map. **Five** modules are reached by **all five** lines and are
therefore the true foundation surface: `analysis.align`, `analysis.behavior`, `analysis.config`,
`analysis.constants`, `core.session`. Three more are reached by four lines:
`analysis.track_verdict`, `analysis.utils`, `suite.loader`.

| Live line | Modules it reaches (transitively) |
|---|---|
| early-lick / QC1 | 24 |
| tf-glm | 21 |
| camera | 14 |
| state labeling | 14 |
| population-field | 9 |

Full per-module attribution is in `data/cache/audit/cold_list_seed.csv`.

## Caveats on this seed

1. **Import-statement census only.** A module reached by runtime string dispatch, a `subprocess`
   invocation or a Slurm job body reads **cold**. The same blind spot bounds
   `d3.scripts.orphan_nonentry` = 46, whose in-degree is likewise import-only.
2. **Line membership is defined by subtree, not by intent.** `scripts/analysis/behavior` is broad
   and inflates the early-lick line's reach; a module credited only to that line may be reached by
   a script that is itself dormant.
3. **Temperature is a snapshot.** It is a fact about the *scripts tree on 2026-08-14*, and the
   scripts tree is where dead code accumulates. Re-run the classifier before treating any row as
   settled.
4. **Coverage is not temperature.** 14 modules are truly untested
   (`d5.tests.untested_modules_ast`); 8 of them are cold and 6 are hot. Cite the AST-corrected
   14, never the shipped regex's 32 — that count under-credits 67 of 98 test files because it
   cannot match parenthesised multi-line imports.
