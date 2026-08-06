# Sub-project 0 — Deep Empirical Audit: Specification

**Date:** 2026-08-06
**Status:** Draft for review
**Governed by:** `2026-08-05-new-repo-master-design.md` (ADR-001 … ADR-006)
**Consumes:** scoping recon of 2026-08-05 (6 domains, 65 problems: 12 critical / 18 high / 28 medium / 7 low) + migration-risk brief

---

## 1. Purpose

Produce the **findings corpus** from which specs 1–6 are written.

Under ADR-006 the project specifies everything before building. That places all the risk on the
specification, and specifications written against imagined data drift — this repo has already
proved it (`docs/ARCHITECTURE.md` is a "definition of done" for a tree that was never built, and
every one of its rules is currently violated).

The mitigation, and the defining constraint of this sub-project: **the audit is empirical.** It
loads real sessions, executes real code paths, and reports measured numbers. A finding that says
"this looks wrong" is not a finding. A finding says *what was run, what came back, and how many
rows/units/sessions it affected*.

## 2. Scope

Eight domains. The first six come from the scoping recon; D7 and D8 were added from the
migration-risk brief and the master design's known-defect requirement.

| ID | Domain | Anchor |
|---|---|---|
| D1 | Constants, config and shared definitions | 82 canonical constants; 1,557 module-level constant assignments; 130 divergent names |
| D2 | Package layering, imports, packaging | 70 library files; 1,112 import statements; 228 `sys.path` mutations |
| D3 | `scripts/` tree | 378 scripts, 84,363 LOC, 30 topic dirs |
| D4 | Data, caches, figures, provenance | 16,779 cache files / 15.9 GB; 3,056 figures / 36.1 GB; 285 pkls / 30.5 GB |
| D5 | Tests, packaging, CI, tooling | 104 test files; **zero CI, zero linters, zero pre-commit** |
| D6 | AI layer and documentation | `CLAUDE.md` 458 lines; 135 docs / 58,856 lines; 7 skills; 1 hook |
| D7 | Work-at-risk and migration surface | 139 single-copy commits; 6 worktrees; 2 stash-tags; external sibling repos |
| D8 | Known-defect register | Cross-cutting; gates sub-project 3 |

## 3. Method

### 3.1 Rules that apply to every domain

1. **Read-only with respect to the current repo.** The audit does not fix anything. Findings that
   demand a fix are recorded, not applied. (Exception: the audit may write its own outputs under
   `docs/audit/` and `data/cache/audit/`.)
2. **No compute over the `X:` Samba mount.** Project hard rule. Heavy compute goes to HPC/Slurm.
3. **Measured, not read.** Where a claim can be settled by executing code, execute it. The recon
   already demonstrated the difference: `load_qc_profile('default')` *reads* as correct and
   *returns* `{}`.
4. **Every finding carries** `file:line` evidence, the command run, its output, and a blast-radius
   count (rows / units / sessions / figures affected).
5. **Curated structured findings, not raw dumps.** Master design §8 requires the corpus to fit in a
   single working session. Raw output is kept separately and referenced by pointer.
6. **Latest Opus for every subagent**, per user hard rule, passed explicitly on every call.

### 3.2 Per-domain measurement requirements

Abbreviated; each is a checkable deliverable, not a topic to explore.

**D1 — Constants**
- For each of the 82 names in `constants.py`: value, whether `config.py` re-exports it (42 of 82
  currently do not), importer count, count of files retyping its literal, and whether every retyped
  literal agrees. Report the dead set (22) and the shadowed set separately.
- Execute `load_qc_profile` for all four profile names; record returned dicts; then diff unit counts
  on one real session under `default` vs `striatal_strict` to quantify how many published unit
  counts came from the wrong gate.
- Resolve the firing-rate floor actually applied on every unit-selection path
  (`utils.get_good_cluster_ids`, `core.qc.apply_unit_filters`, `core.qc.find_good_stable_units`, the
  six hardcoded `min_fr=1.0` sites). **One measured number per path on one real session.**
- Settle the TF sampling period empirically, then enumerate every consumer of
  `TF_SAMPLE_PERIOD=0.25` and every bare `dt=0.05` / `DT_GEN=0.05` / `DT=0.02` site, and state which
  published figures and caches were produced under each.
- **Settle whether `ref` trials saw the change stimulus, from the raw trial table** — not from
  either constant. Then report per-session trial counts each definition includes/excludes, and which
  analyses the `EVENT_VALID_OUTCOMES` vs `CHANGE_PRESENTED_OUTCOMES` disagreement affects.
- Classify all 130 divergent multi-file constant names into (a) scientific parameter — must unify,
  (b) path/output alias — must route through config, (c) genuinely local. Counts per bucket, plus
  `file:line` pairs for every member of (a).
- For every palette: enumerate every distinct hex used per semantic label across the 717 script hex
  literals; report which labels render in more than one colour anywhere in `FIGURES/`, naming the
  affected published figures.

**D2 — Layering**
- Files importing `src.visdetect.*` vs `visdetect.*`, and which mix both in one process (25
  statements across 9 files; 7 mixed). For each, whether an object crosses the boundary into an
  `isinstance`/dataclass check.
- Classify all 218 real-workspace `sys.path.insert` calls: redundant, pointing at a non-existent
  path, or load-bearing.
- Module-level (not lazy) upward import edges, by layer pair. Baseline: `core→analysis` 2,
  `anatomy→analysis` 1, `analysis→suite` 0 module-level / 2 lazy.
- Import wall-time and `sys.modules` delta for each public entry point. Baseline:
  `import visdetect.analysis.constants` = **2.24 s, +1,541 modules**.
- **Build a wheel and `pip install .` (not `-e`) into a clean venv, then import every module.**
  Expected failures: `visdetect.viz` (50 importers) and `visdetect.integrations` — neither has
  `__init__.py`, so `find_packages` drops them.
- Enumerate module-level side effects reachable from `import visdetect`: backend mutations
  (4 known sites), import-time `makedirs` (`suite/config.py:19-20`), filesystem reads.
- Library modules that break when cwd ≠ repo root or when installed outside `src/`: relative
  literals plus every `parents[N]` derivation, each index verified against real file depth.

**D3 — Scripts**
- For each of the 22 local date parsers and 23 raw `strptime('%d%m%Y')` sites: whether the token
  reaching it can be 7-digit or float-formatted, tested with the trio
  `'01072025'` / `'1072025'` / `'1072025.0'`, and whether the resulting mis-ordering changes any
  published figure.
- For the 78 `zfill(8)` sites: the dtype at that point, and whether `load_staging_manifest` already
  canonicalized — this determines whether the real defect count is 78 or ~10.
- For all 10 files writing under `vd_tf_bg046/`: whether the figure currently on disk in this repo's
  `FIGURES/` was produced by current code or predates the path break (mtime vs last-commit).
- Run all three `partial_spearman` variants on one shared input and report the spread. The
  `np.corrcoef`-on-residuals variant is a different estimator from the `spearmanr` variants and is
  expected to differ systematically.
- Which of the 9 unguarded lick-channel scripts touch any of the 33 sessions in the 6-Mar-2026
  re-extraction batch, and whether their outputs show a step change at that boundary.
- Build the real import DAG for `scripts/`; mark every in-degree-0 file that is not an argparse
  entry point. Classify the 130 scripts writing neither figure nor CSV.

**D4 — Artefacts and provenance**
- Re-run `tests/test_session_id_csv_integrity.py` and **extend its scope to `FIGURES/` and
  `table_output/`** (it currently walks `data/` only). Report exact file:row counts of 7-digit and
  `'00'`-prefixed tokens, flagging which offenders are git-tracked deliverables.
- For all 327 CSVs carrying a session-id column: classify the key domain actually used, and build
  the cross-table of which file pairs can legally be joined. **Quantify rows lost in each join the
  codebase actually performs.**
- Per-cache staleness: mtime vs last-commit of the writing script *and* the library modules it
  imports; validated by a schema/value check (the `predicted_session_groups.csv` `chron` column is
  the worked example — it holds values current code cannot produce).
- **Regenerability test: for ~20 tracked `FIGURES` artefacts, locate the producing script, re-run
  with the current manifest, and diff.** Report the pass rate and the fraction whose producing
  script cannot be identified at all.
- Collision counts for every date-keyed index over pkls/caches, per subject, against the real
  285-file pkl tree; which twin currently wins, and whether it is deterministic.
- `SESSION_FILTER` divergence: for each of the 28 scripts reading a manifest directly, its session
  set vs `load_staging_manifest(qc_only=True)`, symmetric difference per script.
- Audit the 3 `tf_responsive` registries against post-lick-channel-fix code: how many `resp_log2`
  calls flip, and which downstream caches and figures inherited the stale calls.

**D5 — Tests and tooling**
- Coverage by library module; name every `src/visdetect` module with no test at all.
- Partition tests into offline vs real-data-dependent; measure runtime of each partition.
- Fix `SKIP_DIRS` in `scripts/qc/check_refactor_guardrails.py` to exclude `.claude/` and report the
  **true** violation count (recon measured 218 real vs 1,375 reported, 84% worktree noise) — this is
  the number that should gate CI.
- Establish the current de-facto gate explicitly: what mechanically stops bad code landing today.
  Recon answer: **nothing but one delete-guard hook.**

**D6 — AI layer and docs**
- For every numeric literal in `CLAUDE.md`, `docs/*.md` and every `SKILL.md`: resolve the named
  symbol in `src/` and report match / mismatch / symbol-not-found.
- Line-level duplication matrix between `CLAUDE.md` and each doc (measured: `NEURO_BEST_PRACTICES`
  71%, `GOTCHAS` 78%, `SCRIPT_TEMPLATE` 88%, `NORMALIZATION` 0%). **For each duplicated pair, whether
  the copies agree** — duplication is tolerable, divergence is the bug.
- Dead-path count: every path, directory and script named in docs or skills, checked for existence
  (known dead: `analysis_suite/*` — 181 occurrences across 42 files; `analyses/`; `AI_exploration/`).
- How many of the 76 `docs/superpowers` files describe work later retracted or refuted, and whether
  each carries a terminal status marker.
- Whether each `docs/science/*-results.md` still agrees with the memory note for the same question.
- Pairwise trigger-overlap between the 7 skills; which pairs can both fire and what decides.
- Every model id, tool name and agent type named in prose across `.claude/`.
- Count instruction files claiming canonical authority (found: 4) and which any tool actually loads.

**D7 — Work at risk**
- Refresh `origin` (requires the SSH key) and re-verify every "unpushed" conclusion; the recon's
  counts rest on a tracking ref last fetched 2026-07-10.
- Per worktree and the primary: size and file count under gitignored `data/` and `FIGURES/`, and
  which outputs are deliverables that convention says must be `git add -f`-ed before any freeze.
- Re-check NTFS junctions immediately before any worktree removal — never trusting a prior snapshot.
- Content and current relevance of the two preserved stash-tags, so an explicit keep/drop decision
  is recorded rather than made by omission.
- Sibling repos: which duplicate constants, session handling or QC logic, and therefore constitute
  external sources of truth the new repo must account for.

**D8 — Known-defect register** *(the deliverable that gates sub-project 3)*

For every analysis module, a verdict: **must reproduce** the old number, or **must differ**, in a
stated direction, for a stated reason. Modules whose status cannot be determined are **quarantined**
and specified explicitly rather than ported on assumption.

Seed entries, already evidenced:

| Defect | Status |
|---|---|
| `load_qc_profile` returns `{}`; strict/lenient runs identical | must differ |
| `TF_SAMPLE_PERIOD = 0.25` (5× too coarse) | must differ |
| `parse_session_date(int(x))` mis-sorts 6-digit and day-1–9 tokens | must differ |
| 15,802 corrupted session-id rows in 6 live caches | must differ |
| Stale `tf_responsive` registries behind the VMS>DMS headline | must differ |
| Lick-channel under-detection, 33 sessions | must differ |
| TF-pulse PETH circularity and pre-fix caches | must differ |
| Trial/event alignment defect (QC1, in repair) | must differ |
| Retracted transient/sustained state result | must differ |
| Refuted "sustained StimSens = expert signature" | must differ |
| `ref`-trial change-presented ambiguity | **quarantined** pending D1 |
| `CHANGE_SIZES` membership divergence (catch included/excluded) | **quarantined** pending D1 |

## 4. Deliverables

Committed under `docs/audit/`:

1. `00-executive-summary.md` — ranked cross-domain findings; what must be fixed before the new repo
   is built vs what the new repo's design makes impossible by construction.
2. `01-…-08-<domain>.md` — one per domain, structured findings with evidence and blast radius.
3. `known-defect-register.md` — D8, the machine-readable table gating sub-project 3.
4. `measurements.csv` — every measured quantity, one row per measurement, so specs 1–6 can cite
   numbers rather than impressions.
5. `quarantine.md` — results whose validity is in question pending a decision, with the specific
   check that would settle each.
6. `drop-list.md` — code, docs and artefacts that should be dropped rather than analysed, with the
   evidence for calling each dead.

## 5. Acceptance criteria

The audit is complete when:

- **A1** Every domain deliverable exists, and every finding in it carries `file:line` evidence, the
  command run, and a blast-radius count.
- **A2** The known-defect register covers every analysis module that sub-project 3 will port. No
  module is unclassified.
- **A3** Every "must differ" entry states the direction of the difference and the reason.
- **A4** The regenerability test has been run on ≥20 tracked figures with a reported pass rate.
- **A5** The corpus is curated to fit one working session; raw output is referenced, not inlined.
- **A6** Every number in the executive summary traces to a row in `measurements.csv`.

## 6. Explicit non-goals

- Fixing anything. The audit measures; sub-projects 1–6 fix. The one exception under consideration
  is `check_refactor_guardrails.py`'s `SKIP_DIRS` — a one-line change that converts an unusable gate
  into the measurement instrument D2 and D5 both need. **Decision deferred to review.**
- Re-deriving scientific conclusions.
- Auditing code already on the drop-list. Dead code is enumerated and dropped, not analysed.
- Auditing `archive/`, `_DeepUnitMatch_repo/`, `_preserved_from_worktrees_20260628/`,
  `refactor_baseline/`, or `.venv/`.

## 7. Sequencing note

Sub-project −1 (**secure the work at risk**) precedes this audit: 139 commits and ~31 MB of
hand-labelled, code-irreproducible data currently have no off-disk copy, and the SSH key is not
loaded so no push can be attempted. Auditing a repo whose only copy is local is defensible; freezing
or replacing one is not.

The hand-labelled artefacts were copied to `e:\python_analysis\_handlabel_backup_20260806` on
2026-08-06 — same disk, so this mitigates accidental deletion only, not disk failure.

## 8. Open questions for review

1. Should the `SKIP_DIRS` one-line fix be made during the audit (§6)?
2. Does the regenerability test target 20 figures, or a stratified sample across all 23 topics?
3. Is BG_012 in audit scope, given its protocol variants are parked?
4. Should D7 block on the SSH key, or proceed and re-verify later?
