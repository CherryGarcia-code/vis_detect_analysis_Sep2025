# D3 — Scripts tree: date parsers, zfill dtypes, partial_spearman spread, dead writers

Executed census of the `scripts/` tree (excluding `scripts/audit/` and `__pycache__`
by component test — no self-scan inflation): local date-parser behaviour, ad-hoc
`zfill(8)` sites against the dtype the staging manifest actually returns, the
three-family `partial_spearman` estimator spread on one shared real input, writers
into the deleted `vd_tf_bg046` tree, and a per-script output/entry-point/in-degree
classification census.

- Script: `scripts/audit/d3_scripts_census.py` (`py scripts/audit/d3_scripts_census.py`, exit 0)
- Census CSVs (gitignored; committed with `git add -f`):
  `data/cache/audit/date_parser_sites.csv` (`file,line,kind` — Task 15 input),
  `data/cache/audit/script_classification.csv`
  (`file,has_main,has_argparse,writes_figure,writes_data,in_degree_0` — Task 15
  drop-list triage input)
- Measurement ids: `d3.*` in `docs/audit/measurements.csv`

## Summary

| Measurement | Value | Baseline | Verdict |
|---|---|---|---|
| `d3.dateparser.trio` | `01072025->2025-07-01 \| 1072025->2025-07-10 \| 1072025.0->ValueError` | silent wrong date on 7-digit | confirmed: 7-digit token parses to the WRONG date with no exception |
| `d3.dateparser.sites` | 19 | 23 | deviation; counting-basis, explained below |
| `d3.zfill.sites` | 77 | 78 | deviation; occurrences-vs-lines, explained below |
| `d3.zfill.manifest_dtype` | `{'str': 28}` | — | all 28 rows `str`, all classify `8digit` — zfill sites redundant-but-harmless for manifest-derived ids |
| `d3.pspearman.spread` | 0.892 / 0.901 / 0.901 | spread > 0.02 ⇒ upgrade | max spread 0.009 ≤ 0.02 — upgrade does NOT fire |
| `d3.vdtf.writers` | 10 | 10 | exact match |
| `d3.scripts.no_output` | 108 | ~130 (spec estimate) | measured; triage deferred to Task 15 |
| `d3.scripts.orphan_nonentry` | 46 | — | new data for the drop-list |
| `d3.shim_importers` | 0 | 0 | exact match — the shims are droppable |
| `d3.lick.overlap` | not-measured | — | honest gap: requires forbidden X:-side NI-file audit |

## Date parsers (`d3.dateparser.*`)

**The trio behaviour every local `strptime('%d%m%Y')` parser inherits**
(`d3.dateparser.trio`): the canonical 8-digit `01072025` parses correctly to
2025-07-01; the int64-stripped 7-digit `1072025` parses **silently to 2025-07-10 —
the wrong date, day 10 month 7 instead of day 1 month 7, with no exception**; the
float-string `1072025.0` raises `ValueError`. So of the three corrupted-id forms the
repo produces (see the session-id gotcha), one crashes loudly and one corrupts dates
silently. Every one of the 19 raw `strptime` sites is exposed to the silent case.

**Site count deviation (19 vs 23).** `d3.dateparser.sites` = 19 with the shipped
regex `strptime\([^)]*%d%m%Y` (same-line, scripts/ only, audit dir excluded); a git
grep cross-check on tracked files gives the identical 19, so this is not
untracked-file or self-scan drift. The plan-time 23 is exactly reconstructed by a
broader "local date parser" definition: 19 strptime-`%d%m%Y` sites + 3
`pd.to_datetime(..., format='%d%m%Y')` sites
(`scripts/analysis/plot_learning_curve.py:18`,
`scripts/analysis/run_deep_unitmatch.py:75`,
`scripts/batch_processing/build_manifest_and_behavior_summary.py:151`) + 1
`strptime(..., '%d%m%y')` site (`scripts/pipelines/tracking/run_unitmatch_all.py:73`,
lowercase `%y`, 6-digit) = 23. The 4 extra sites are equally exposed to the same
class of defect (the `to_datetime` sites inherit the identical `%d%m%Y` silent-wrong-
date behaviour; the `%d%m%y` site is the 6-digit variant) — they are just outside
the shipped regex. Baseline counting-basis difference, not tree drift. *(Updated
2026-08-18, Task 16: the hand-reconstructed 23 was itself superseded by the Task-15
wave-4 AST census — `d8.dateparser.recount` = **27** sites (`scripts/` 27, `src/` 0;
3 `to_datetime`, 1 six-digit `%d%m%y`; `scripts/audit/d3_parser_recount.py`,
`data/cache/audit/date_parser_recount.csv`). The AST call census catches multi-line
and keyword-arg calls both the shipped regex and the hand count missed. Treat the
parser-site population as **27**, with 19 captured in `date_parser_sites.csv`;
register entry 3 carries the recount and says the same.)*

## zfill(8) vs manifest dtype (`d3.zfill.*`)

`d3.zfill.sites` = 77 matching **lines**. The plan baseline of 78 counted
**occurrences**: `scripts/pipelines/tracking/_subject_paths.py:94` contains two
`zfill(8)` calls on one line (`git grep -oE 'zfill\(8\)'` → 78; line count → 77).
Exact reconciliation, zero tree drift.

`d3.zfill.manifest_dtype` = `{'str': 28}`: `load_staging_manifest(qc_only=False)`
returns all 28 `session_name` values as Python `str`, and a `classify_token`
cross-check classifies all 28 as `8digit`. Consequence: the 77 downstream `zfill(8)`
sites are **redundant-but-harmless for manifest-derived ids** — the ~78-defect
reading is wrong; the real exposure is only the sites fed ids from *other* sources
(raw CSV round-trips, int casts), plus the standing footgun that ad-hoc
`str(x).zfill(8)` breaks on float-strings (`'1072025.0'.zfill(8)` ≠ `'01072025'`),
which is why `canonical_session_id()` remains the required path.

## partial_spearman three-family spread (`d3.pspearman.spread`)

All three estimator families replicated verbatim from their source files, run on one
shared real input (`data/cache/session_sorting/session_group_features.csv`, n=44
sessions; x=`occ_StimSens`, y=`hit_rate_go`, control z=`n_trials`):

| Family | Sites | Value |
|---|---|---|
| A: rank → residualize ranks on ranks → `spearmanr` | `theta_prototype.py:106-115`, `theta_count_matched.py:147`, `within_session_dynamics.py:65-71` | **0.892** |
| B: rank → residualize ranks on ranks → `np.corrcoef` | `learning_continuum.py:94-104`, `learning_transient_sustained.py:95`, `latency_outcome_coupling.py:254` | **0.901** |
| C: closed-form from pairwise Spearman rhos | `explore4_partial_rt.py:49-57` | **0.901** |

Maximum pairwise spread = **0.0090** (unrounded: A=0.891614, B=0.900577,
C=0.900577; corrcoef-vs-spearmanr = 0.008963; B and C are identical to 6 dp — the
closed form coincides with Pearson-on-rank-residuals in the absence of divergent tie
corrections). This is **below the 0.02 threshold**, so **no register entry was
warranted** — the finding is carried as a residual measurement gap in
`quarantine.md` Q11, not as a defect entry.

> **Correction (2026-08-17, Task 15 wave 4, finding X4).** The paragraph above
> previously said "the register entry stays at 'different estimator in principle'
> and is NOT upgraded" — phrasing that implies a `partial_spearman` register entry
> exists. None does, and none was ever created: the measured spread (0.0090) fell
> below the pre-declared upgrade threshold, so the finding's home is quarantine
> Q11's residual-gaps table. The upgrade *rule* was real; the entry it would have
> upgraded was never instantiated.

On this input, re-ranking the rank-residuals (A) vs Pearson on the
rank-residuals (B) vs the closed form (C) agree to ~0.01 — a consolidation to one
canonical `partial_spearman` is still warranted for hygiene, but no existing result
is invalidated by the estimator choice on the evidence measured here.

## vd_tf_bg046 dead writers (`d3.vdtf.writers`)

10 scripts (exact baseline match) still write into the deleted `vd_tf_bg046/`
(FIGURES|data) tree: `scripts/tf_responsiveness/plot_bg046_pulses.py` @2026-06-29 and
9 under `scripts/tf_responsiveness/state_conditioned/` (last-commit dates 2026-07-01
… 2026-07-16; full list in the `d3.vdtf.writers` evidence field). Reruns succeed and
write nowhere visible. Cross-check: the brief's `.stdout.split()` parse was verified
against a `.splitlines()` parse — identical 10 paths, none containing spaces, so no
path mangling occurred in this run.

## Script classification census (`d3.scripts.*`, `d3.shim_importers`)

378 scripts scanned (scripts/ minus audit + `__pycache__`; untracked in-tree scripts
included — they are measurement subjects, not staged artefacts).

- `d3.scripts.no_output` = **108** scripts write neither a figure
  (`savefig|save_figure`) nor a data artefact (`.to_csv(`/`np.save`/`json.dump`) —
  the shared-module / job-body / dead population (spec estimated ~130).
- `d3.scripts.orphan_nonentry` = **46** scripts are in-degree-0 in the intra-scripts
  import DAG AND have no `__main__` guard AND no argparse — the strongest dead-code
  candidates.
- **Import-DAG note:** in-degree here is computed only over *intra-scripts*
  stem-name import edges (siblings imported by bare module name after sys.path
  tricks). In-degree-0 is therefore an upper bound on orphanhood — runner scripts,
  Slurm job bodies, and subprocess invocations are invisible to this DAG. The
  in-degree-0 **classification** (shared-module vs job-body vs genuinely dead) is
  deferred to the drop-list task (Task 15), which takes
  `script_classification.csv` and `date_parser_sites.csv` as input.
  *(Discharged 2026-08-17, Task 15 wave 4: `scripts/audit/d3_script_triage.py` →
  `data/cache/audit/script_triage.csv`, ids `d8.scripts.orphan_triage` /
  `d8.scripts.nooutput_triage`, drop-list §2.8. Headline: 1 dead, 38 job-bodies,
  7 package markers of the 46.)*
- `d3.shim_importers` = **0** importers of the top-level shims
  `src/visdetect/{session,io}.py` across scripts/, src/, tests/ — the drop-list
  evidence: the shims can go.

**CSV portability note for Task 15:** the `file` column in both census CSVs uses
Windows-native backslash separators (`scripts\...`) because the shipped code
stringifies `Path.relative_to()` results; consumers should normalize separators on
read.

## Lick-channel overlap (`d3.lick.overlap`)

`not-measured`, recorded honestly: the 33-session MATLAB re-extraction batch list is
not materialized anywhere in the repo, and deriving it needs NI-file inspection on
the X: mount, which the audit forbids. Direction carried in the register: lick rates
in affected sessions are under-detected 10-40×, so cross-session lick-rate trends
from Piezo/Lick-channel scripts are suspect until the batch list is materialized.
