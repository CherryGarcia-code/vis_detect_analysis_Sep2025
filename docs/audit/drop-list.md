# Drop list — code, docs and artefacts to drop rather than analyse

Companion to `known-defect-register.md` and `cold-list.md`. Everything here is **dead by
evidence**, not by impression: each row carries the measurement id or `file:line` that convicts
it. Dropping means *not carrying it into the new repo*; where the item is also safe to delete from
the old repo, that is stated separately, because the old repo stays live until the sub-project 6
freeze.

**Nothing in this file was deleted, staged, edited or reverted by the audit.** These are
recommendations with evidence; the owner decides.

> ⚠️ **Read `quarantine.md` Q9 before acting on section 1.** Every `origin/*` claim below rests on
> a remote-tracking ref last fetched 2026-08-06; `git ls-remote` fails in this checkout because
> the SSH agent is not running. Re-verify against the real remote before deleting any ref.

---

## 1. Branch refs

Evidence: `d7.branches.unmerged`, `d7.local_only.commits`, and the per-branch
`git branch -a --contains` / `git merge-base --is-ancestor` runs recorded in
`branch-disposition.md`.

| Ref | Ahead (raw/cherry) | Local-only | Why it is droppable |
|---|---|---|---|
| `feature/tf-transient-sustained-spectrum` | 1/0 | 0 | The **only** branch `git cherry` clears outright: its single commit (`2f82abe`, anatomy TF/kernel-width cell maps) is already applied upstream under a rewritten sha |
| `fix/lick-channel-resolver` | 0/0 | 0 | Nothing unique — fully contained in `main` |
| `worktree-camera-tagging` | 0/0 | 0 | Nothing unique — fully contained in `main` |
| `worktree-population-field` | 0/0 | 0 | Nothing unique — fully contained in `main` |
| `worktree-theta-prototype` | 0/0 | 0 | Nothing unique — fully contained in `main` |
| `feature/camera-tagger-2b` | 0/0 | 0 | Nothing unique — merged to `main` as `caa377d` |
| `feature/fig5eh-preparatory-cellclass` | 4/4 | 0 | **The brief's stated precondition FAILS** (`git merge-base --is-ancestor … main` → exit 1; `git cherry main` → 4 `+` patches), so do **not** drop it on the recommendation as worded. It is droppable by a different, verified route: all four commits are ancestors of `design/new-repo-foundation` *and* of the cached `origin/` refs for it, so the content reaches `main` when that branch merges |

**Explicitly NOT droppable, despite also showing `Local-only = 0`:**

- **`hardening/fa-psth-and-manifest-sort`** — `2f6fcdc` (centralised `fa_lick` PSTH condition +
  the DDMMYY manifest-sort fix) is contained in **no other branch**
  (`git branch -a --contains 2f6fcdc` lists only that ref and its origin twin; not an ancestor of
  `main` nor of `design/new-repo-foundation`). Dropping the ref **strands the fix**, and the sort
  fix is register entry 3's mitigation. `Local-only = 0` distinguishes "is on a remote" from "is
  held by a branch that survives the freeze"; those are different questions and this row is the
  proof.
- **`feature/early-lick-and-session-sorting`** — live QC1 work, still receiving commits on the day
  of the audit (register entry 8 is IN-REPAIR on it).
- **The two stash-tags** (`pre-tidy-20260628/stash-0`, `stash-1`) — both readable, both on **no
  remote ref**; their 6 commits are part of `d7.local_only.commits`. Securing them is a **push**,
  not a merge decision. Do not drop before pushing.

## 2. Code

### 2.1 The seven orphaned `tf_response` leaf scripts

`scripts/analysis/tf_response/{barplot_top_splitters, extract_top_clusters, heatmap_zscore_diff,
pairwise_lineplot_splitters, plot_tf_pulse_grid, scatter_split_score_vs_zmax,
scatter_zmax_fast_vs_slow}.py`

- **Evidence** — all seven are `in_degree_0 = True` in
  `data/cache/audit/script_classification.csv` (they are CLI entry points nothing imports), and
  the single subprocess reference to any of them,
  `scripts/batch_processing/batch_plot_tf_grids.py:34`, targets
  `scripts/analysis/plot_tf_pulse_grid.py` — **a path that does not exist** (the file lives one
  directory deeper). So the batch runner has never launched it.
- **Scientific basis** — they serve the **single-pulse TF-responsiveness** method, which the GLM
  replication superseded (memory `tf_responsiveness_null_finding_jun2026`; the GLM is recorded as
  the only validated TF method). Their outputs are not cited by any live results doc.
- **Caveat, so this is not over-claimed** — `plot_tf_pulse_grid.py:56` is one of the two live
  `--profile` call sites named in register entry 1. Dropping the script does not fix the
  `load_qc_profile` defect; it removes one of its two reachable surfaces. The other,
  `scripts/batch_processing/batch_plot_tf_pulse.py:35`, imports the **library** function
  `visdetect.analysis.tf_pulse.plot_tf_pulse_grid` and stays.

### 2.2 The two top-level compatibility shims

`src/visdetect/session.py`, `src/visdetect/io.py` — 7 lines each.

- **Evidence** — `d3.shim_importers` = **0** across `scripts/`, `src/` and `tests/`. Both are
  also in the AST-verified truly-untested set (`d5.tests.untested_modules_ast` = 14). Their only
  other footprint is a docstring advertising the `src.visdetect.*` import path — i.e. they
  advertise register entry A7's hazard.

### 2.3 The two orphaned test files that break collection

`tests/test_coding_direction.py`, `tests/test_population.py`

- **Evidence** — both import modules deleted on **2026-02-02** (commit `4f56700`):
  `visdetect.analysis.coding_direction` and `visdetect.analysis.population`. Because they sit
  inside the configured `testpaths = tests` and pytest interrupts on collection errors, **the
  repo's default `py -m pytest` cannot complete collection at all** — register entry A10. Six
  months dead; deleting them restores the manual gate.

### 2.4 `scripts/analysis/coding_direction_stub.py`

- **Evidence** — a stub for the same deleted `coding_direction` module; its body is exploratory
  prose ("For now, I'll search `reactiontimes['Lick_L']` or similar", `:34`) and it guesses lick
  keys by trying a list (`:56`) — the exact hard-coded-name pattern
  `visdetect.analysis.lick_channels` exists to replace (register entry 6).

### 2.5 The empty `scripts/tf_response/` directory

- **Evidence** — `git ls-files scripts/tf_response` returns **0** files; on disk it contains only
  `__pycache__/`. A stale bytecode directory shadowing the real
  `scripts/analysis/tf_response/`, which is itself section 2.1's drop.

### 2.6 Dead output paths (drop the **path**, not the script)

Ten scripts still write into the **deleted** `vd_tf_bg046/` tree: reruns succeed and write nowhere
visible (`d3.vdtf.writers` = 10 — `scripts/tf_responsiveness/plot_bg046_pulses.py` plus nine under
`scripts/tf_responsiveness/state_conditioned/`).

- **This is not a script drop.** Nine of the ten are live `state_conditioned` analysis scripts
  from the transient/sustained line; deleting them discards real work. The dead thing is the
  **path constant**. One of them,
  `scripts/tf_responsiveness/state_conditioned/heatmap_transient_sustained.py:57-58`, has already
  been repointed and carries the comment "was a stale `vd_tf_bg046` path from the old repo —
  re-running wrote nothing where anyone looked", which is the pattern for the other nine.
- **In the new repo the class disappears** by construction: `FIGURES/<topic>/<SUBJ>/` is derived
  from configuration, not spelled per script.

### 2.7 Eighteen `sys.path` targets that point at non-existent foreign trees

- **Evidence** — `d2.syspath.foreign_missing` = 18, all verified non-existent at scan time:
  `vd_tf_bg046/src` ×6, `vd_tf_phase0/src` ×11, and
  `scripts/tf_responsiveness/state_conditioned/combined_figure.py:22` targeting the foreign
  *scripts* dir. Each is a **silent fall-through** to whatever ambient `visdetect` happens to be
  importable, so the provenance of anything those scripts produced is unverifiable. Drop the
  inserts; the new repo's packaging (register entry A12) removes the need for them.

## 3. Docs

### 3.1 `AI_exploration/` references — the directory does not exist

| Site | What it says |
|---|---|
| `src/visdetect/analysis/config.py:9`, `:111`, `:485` | docstring, a "stale, must not be used" warning, and an alias comment |
| `src/visdetect/analysis/waveform_celltype.py:4` | provenance docstring |
| `src/visdetect/suite/loader.py:487` | runtime error message naming the legacy CSV |
| `scripts/pipelines/concat_sort/regen_waveform_labels.py:11`, `:22` | **live output target** `ROOT / "AI_exploration" / "figures" / …` |
| `scripts/qc/check_refactor_guardrails.py:14`, `:37` | an exclusion entry marked "pending triage" |
| `.gitignore`, `RUNNING.md`, `docs/ARCHITECTURE.md`, `docs/REFACTOR_PLAN.md`, `docs/STAGE_FILTERING_EXAMPLES.md`, `docs/AI_interaction/handoff_refactor_may2026.md`, 2 skills | prose references |

The `regen_waveform_labels.py` row is the one that matters: it **writes** into a directory that
does not exist, so the script cannot have run successfully in its current form. Triage it before
dropping the reference.

### 3.2 `analysis_suite/*` references — archived 2026-07-01

- **Evidence** — `d6.deadpaths` = 111 refs / 53 unique paths / 24 docs, of which the
  `analysis_suite` slice is **45 refs across 14 docs** (the in-scope subset of recon's 181/42).
  Top targets: `analysis_suite/utils.py` ×11, `loader.py` ×9, `run_all.py` ×7, `plotting.py` ×4.
- **The refs that matter are in live steering documents, not historical plans**: **CLAUDE.md
  itself carries 2** (lines 224–225, the "Where Normalization Lives" table) and **two active
  skills** cite dead paths (`.claude/skills/research-statistician/SKILL.md:272`,
  `.claude/skills/research-visualizer/SKILL.md:254`) — these load into working sessions and steer
  agents at a directory that no longer exists. `docs/ARCHITECTURE.md` (12 dead refs, including
  `src/visdetect/analysis/archive/`) and `docs/DOCUMENTATION_INDEX.md` are presented as current
  and are not.

### 3.3 The CLAUDE.md doc twins

`docs/SCRIPT_TEMPLATE.md` (88 % line-overlap with CLAUDE.md), `docs/GOTCHAS.md` (78 %),
`docs/NEURO_BEST_PRACTICES.md` (71 %).

- **Evidence** — recon overlap percentages restated at `d6.dup_pair_agreement` (= not-measured by
  design; ADR-005 deletes the copies). Duplication is tolerable; **divergence is the bug**, and
  `docs/GOTCHAS.md:10` is register entry **D3** — it teaches the integer session-id form and never
  mentions `canonical_session_id()`, i.e. it instructs an agent to create defect 4.
- **Keep** `docs/NORMALIZATION.md` — 0 % overlap; independent text, not a copy.
- **Also diverging, outside this scan's scope**: `docs/AI_interaction/copilot-instructions.md`
  and `.github/copilot-instructions.md` (the copy Copilot actually loads) are **not
  byte-identical**.

### 3.4 The three non-loaded canonical-authority claimants

`docs/AI_interaction/copilot-instructions.md`,
`docs/AI_interaction/DOCUMENTATION_CONSOLIDATION_SUMMARY.md`,
`docs/AI_interaction/handoff_refactor_may2026.md`

- **Evidence** — `d6.authority.claimants` = 4. **Only CLAUDE.md is actually loaded by the
  harness**; the other three instruct nobody except a reader who happens to open them, and one of
  them carries a `CHANGE_SIZES` and a `STAGE_ORDER` claim of its own to drift against.

### 3.5 `docs/BrainBulb/` — two opaque Notion export zips

- **Evidence** — 2 files, **2,117 B and 1,911 B**, dated 2026-05-14, **untracked and gitignored**
  (`.gitignore:18` `*.zip`), with GUID filenames that name nothing. Carried by no branch,
  referenced by no doc, and too small to hold anything a reader could not re-export.

## 4. Artefacts

| Item | Evidence | Note |
|---|---|---|
| `data/cache/audit/nwbvenv/` | 1,204 dirs / 12,581 files / 321.7 MiB; gitignored (`.gitignore:48`); verified to contain **0** reparse points | **Owner-assigned, not ours.** The repo's own `PreToolUse` hook denies the delete (register entry A11); the removal command is in `docs/audit/09-storage-spike.md`. Do **not** apply the guard's suggested remedy of deleting the worktree junction |
| `data/cache/audit/wheel/` | empty; the wheel build fails before producing anything (`d2.packaging.wheel_build` = FAIL) | Residue, gitignored, never staged |
| `src/visdetect.egg-info/` | gitignored, mtime 2026-05-18, pre-existing; `top_level.txt` now reads `scripts` after the failed build | Reported, not deleted, per the task rule |
| Old `FIGURES/` tree (~37.8 GB primary) | `d7.gitignored.volume`; `d4.trace.untraceable_frac` = 0.42 | **Not a drop-list call.** ADR-019: no old-repo figure enters the manuscript; ADR-020: the tree becomes a read-only archive root and untracked figures are deleted only **after submission** |

## 5. Consumer warnings — do not carry these into a join

Not repo content; **defects in the audit's own census outputs** that a downstream consumer would
otherwise inherit.

| Artefact | Rows to drop or correct |
|---|---|
| `data/cache/audit/syspath_sites.csv` | 3 **self-scan junk rows** — `scripts\audit\d2_layering.py:2`, `:30`, `:36` (category `computed`) are the census matching its own docstring, comment and detector string. Filter `file` starting `scripts/audit`; the tree has **225** sites net of all 8 audit-own rows |
| `data/cache/audit/date_parser_sites.csv` | **Incomplete**: 19 `strptime` rows (plus 77 `zfill8` rows in the same file). The true parser-site population is **23** — the 4 out-of-regex sites are enumerated in register entry 3 |
| `data/cache/audit/csv_key_domains.csv` | **`rows_lost_on_join` / `joinable_to_manifest` are unusable** for the 126 path-scoped other-subject files (~118,621 phantom "lost" rows; the heuristic reads only the filename). Genuine BG_046 loss is **667 rows across 8 files** |
| `data/cache/audit/constants_census.csv` | **Bucket labels are not ground truth** (the `"OUT" in name` rule). Use `data/cache/audit/constants_retriage.csv` instead: of 127 disagreeing names, **43** are scientific parameters, 84 are non-scientific, 0 ambiguous (`d8.constants.scientific_divergent`). `defined_in` is capped at 6 sites, so names with more are under-enumerated. **The re-triage CSV holds only `retypes_agree = False` rows**, so of the three names `01-constants.md:51-70` flags as suspect mislabels, `OUTCOMES` and `OUTCOME_COLORS` appear (both non-scientific: a figure-panel spec and a palette) while **`N_TRIALS_PER_OUTCOME` is absent by construction — its two sites agree, so it is not a divergence at all.** All three of that caveat's candidates therefore resolve as *not* scientific-parameter divergences |
| `data/cache/audit/traceability_sample.csv` | The `producer` column names a **mentioner, not a verified writer**; 7 rows attribute `scripts/audit/d3_scripts_census.py`, which embeds artefact stems as data and sorts ahead of the real writers. The `method` column is unaffected. Separately, **`d4.trace.tracked_covered` = 83 counts `git ls-files` output, not verified census inclusion** — a tracked figure outside a `FIGURES/<topic>/` directory would inflate it silently. None exists today (checked), and the implementer's manual set-difference confirmed 0 missing, but the check is not mechanically re-derivable from the commit; re-verify with a set-difference, not by trusting the number |
| All census CSVs | **Windows backslash separators.** Normalise on read. `module_register_map.csv`, `cold_list_seed.csv` and `constants_retriage.csv` (Task-15 outputs) use forward slashes |

---

## Not dropped, and why — the near-misses

- **`tests/test_session_id_csv_integrity.py`** — it is **RED on purpose**. It is the live tripwire
  for register entry 4 and names the 6 offender caches with exactly 15,802 stripped rows. Port it
  first, do not "fix" it by fixing the data.
- **`data/cache/tf_responsive/README.md`** — its STALE banner is the only in-place record of
  register entry 5, and it is what defeats the mtime staleness heuristic. Keep the banner; fix the
  heuristic.
- **`docs/raw_data/NIDAQ_AND_EVENT_SPEC.md`** — appeared during the Task-15 fix pass, **untracked
  and on no ref**: **481 lines**, a full re-extraction of one BG_046 session directly from
  `nidq.bin`, then **adversarially audited by six independent reviewers** and corrected. It is now
  cited by register entries 6, 8, 11, E1, E4, A13, A14 and A15 and by quarantine Q6/Q12 — the
  single most load-bearing new evidence in the corpus — and it exists as **one uncommitted copy on
  the same disk as the repo**. That makes `d7.untracked.at_risk` 6 of 7, not 5 of 6. **Not a drop
  candidate; a commit candidate**, and it should be committed before anything else in this list is
  acted on. ⚠ It is also a **living** document that has already revised itself once mid-audit
  (386 → 481 lines, with retractions); cite it by section and re-read before relying on a figure.
  The working code it refers to (`tmpclaude-BG_046_17092025/`, ~6.3 GB, git-ignored, including six
  reviewers' scratch work under `_refute1/`…`_refute6/`) is a separate and much larger exposure
  that D7 did not size because it did not exist at D7's snapshot.
- **The five single-copy untracked scripts** (`validate_event_spike_clock_drift.py`,
  `chronic_feasibility_figure.py`, `render_opto_exemplar_figure.py`,
  `exemplar_tracking_figure.py`, `scratchpad_state_bout_inventory.csv`) — 1,489 lines that exist
  on **no ref** (`d7.untracked.at_risk` = 5 of 6). These are owner decisions in
  `branch-disposition.md`, **not drop candidates**; five of the `chronic_feasibility` figures are
  untraceable in the D4 census purely because their producer is untracked, which is one
  `git clean` away from genuine orphanhood.
- **The 26 cold modules** — see `cold-list.md`. Cold ≠ dead. `visdetect.core.ingest` reads cold and
  is the module the entire re-ingest plan depends on.
