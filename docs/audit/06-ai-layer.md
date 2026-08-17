# D6 — AI layer and documentation

Empirical audit of the instruction/documentation layer: every backticked
`SYMBOL` = value claim in CLAUDE.md, `docs/`, and the seven skills resolved
mechanically against code ground truth; a dead-path census of the same corpus;
a model-id survey of primary `.claude` prose; a count of instruction files
claiming canonical authority; plus three Step-2 surveys (retraction markers in
`docs/superpowers/`, `docs/science`-vs-memory agreement, skill trigger-overlap).
Measurement only: **no scanned doc, skill, or CLAUDE.md was modified**, and the
out-of-repo memory layer was read read-only.

- Script: `scripts/audit/d6_ai_layer.py`
  (`py scripts/audit/d6_ai_layer.py`, exit 0)
- Supplement: `scripts/audit/d6_retraction_survey.py`
  (`py scripts/audit/d6_retraction_survey.py`, exit 0) — the Step-2
  retraction-marker survey's one `record` row
- No script for `d6.science.stale_docs`: it is a per-claim judgment across
  `docs/science/` and the out-of-repo memory notes, recorded via a `record()`
  one-liner; every verdict is cited both sides (`doc:line` + memory note) below
- CSVs (gitignored; committed with `git add -f`):
  `data/cache/audit/doc_literals.csv`
  (`doc,line,symbol,doc_value,code_value,verdict`) and
  `data/cache/audit/dead_paths.csv` (`doc,line,path`)
- Measurement ids: `d6.*` in `docs/audit/measurements.csv`

## Summary

| Measurement | Value | Notes |
|---|---|---|
| `d6.literals.checked` | 20 | backticked `SYMBOL` claims resolved against `constants.py` + `analysis/config.py` |
| `d6.literals.mismatch` | 4 | **all four adjudicated as mechanical false positives — substantive doc-vs-code divergences: 0** (see below) |
| `d6.literals.symbol_missing` | 1 | `STAGE_ORDER` — exists, but assigned inside `if/else`, invisible to the top-level AST scan |
| `d6.deadpaths` | 111 | 53 unique paths across **24** docs (corrected from "25" 2026-08-17, Task 15 wave 4 — `dead_paths.csv` counts 24 unique `doc` values); `analysis_suite/*` = 45 refs / 14 docs (recon baseline 181/42, wider scope — see below) |
| `d6.modelids` | 3 | all in one file (`harden-result/SKILL.md`), all Opus-4.8-era |
| `d6.authority.claimants` | 4 | CLAUDE.md + 3 `docs/AI_interaction` files; only CLAUDE.md is loaded by the harness |
| `d6.dup_pair_agreement` | not-measured | recon overlap %s restated below; full pairwise diff deferred (ADR-005 deletes the copies) |
| `d6.superpowers.retraction_markers` | 4 | of 80 files under `docs/superpowers/` (spec recon counted 76 before the audit added its own) |
| `d6.science.stale_docs` | 4 | of 12 `docs/science/` results docs — but **not the two cases the spec predicted**, and 3 of the 4 are stale in a single clause (see below) |

## Scan scope and the self-contamination guard

Corpus: `CLAUDE.md` + `docs/**/*.md` + `.claude/skills/**/SKILL.md`, minus any
file with an `audit` or `superpowers` path part — so the audit's own
`docs/audit/` output and the audit plan/spec under `docs/superpowers/` cannot
feed back into their own measurements. **Verified in this run: zero rows in
either CSV come from `docs/audit` or `docs/superpowers`.** Ground truth is the
set of top-level UPPERCASE assignments (AST-parsed, source expression kept
verbatim) in `src/visdetect/analysis/constants.py` and
`src/visdetect/analysis/config.py`. Claims are backticked symbols followed by
`|`, `=`, or `:`; paths are backticked strings rooted at
`scripts|src|docs|config|analysis_suite|data`.

## Literals: 20 claims checked, zero substantive divergences

The CLAUDE.md constants-table check — until now a hand-eye exercise — is
mechanical. 15/20 claims match outright. The five flagged rows, adjudicated by
importing the live values:

| doc:line | Symbol | Verdict | Adjudication |
|---|---|---|---|
| `CLAUDE.md:90` | `CHANGE_SIZES` | MISMATCH | **false positive** — code is the non-literal `sorted(ALL_GO_CHANGE_SIZES)`, which evaluates to exactly the documented `[1.25, 1.35, 1.5, 2.0, 4.0]` (verified by import) |
| `docs/AI_interaction/copilot-instructions.md:114` | `CHANGE_SIZES` | MISMATCH | same false positive |
| `docs/science/state_labeler/methods.md:167` | `STATE_LABELS` | MISMATCH | formatting-only: prose list vs Python list repr; same four labels (`constants.py:252`) |
| `docs/science/state_labeler/methods.md:168` | `STATE_FEATURE_COLS` | MISMATCH | formatting-only: same six feature names (`constants.py:257`) |
| `docs/AI_interaction/copilot-instructions.md:118` | `STAGE_ORDER` | symbol-not-found | symbol **exists** but is assigned inside the `merge_naive_learning` conditional (`config.py:187/190`); live value `['Learning', 'Expert']` matches the claim |

Bottom line: **the hand-copied CLAUDE.md key-constants table (9 rows) is
accurate today.** The master-design worry ("hand-copies the constants table —
can silently go stale") is a real hazard but is not currently realized.
Caveats on the mechanical pass: it only sees backtick-`SYMBOL` claims (prose
numbers like "σ=25 ms" are untested); the containment match is lenient (a doc
value that is a substring of the code expression passes); and conditional or
computed constants are invisible to the top-level AST — the one
`symbol-not-found` is exactly that blind spot, not a phantom symbol.

## Dead paths: 111 refs, 53 unique paths, 24 docs (count corrected 2026-08-17, wave 4)

Recon baseline: **181 `analysis_suite` refs across 42 files** — counted over
the whole doc corpus (including `docs/superpowers/` and non-backticked
mentions). The mechanical scan counts only backticked path-shaped refs inside
the DOCS scope above, so its `analysis_suite` slice — **45 refs across 14
docs** — is the in-scope subset of that baseline, not a contradiction of it.

By root: `analysis_suite/` 45, `scripts/` 39, `data/` 14, `src/` 11,
`docs/` 2. Top dead targets:

| Refs | Path |
|---|---|
| 11 | `analysis_suite/utils.py` |
| 9 | `analysis_suite/loader.py` |
| 7 | `analysis_suite/run_all.py` |
| 4 | `analysis_suite/plotting.py` |
| 4 | `src/unit_tracking.py` |
| 4 | `analysis_suite/03_population/f_2d_decomposition.py` |
| 3 | `analysis_suite/cache/2d_decomposition/` |
| 3 | `scripts/video/validate_roi.py` |

The refs that matter are the ones in **live steering documents**, not
historical plans:

- **`CLAUDE.md` itself carries 2 dead refs** (lines 224–225): the "Where
  Normalization Lives" table still points at `analysis_suite/utils.py` —
  archived 2026-07-01.
- **Two active skills cite dead paths**:
  `.claude/skills/research-statistician/SKILL.md:272`
  (`analysis_suite/utils.py`) and
  `.claude/skills/research-visualizer/SKILL.md:254`
  (`analysis_suite/plotting.py`, `analysis_suite/utils.py`). These load into
  working sessions and steer agents toward a directory that no longer exists.
- The remaining **21** docs (corrected from "22" 2026-08-17, Task 15 wave 4:
  24 total − CLAUDE.md − 2 skills; the earlier arithmetic started from the
  wrong total of 25) are mostly plans/handoffs (`docs/AI_interaction/`,
  `docs/UNITMATCH/`) — but `docs/ARCHITECTURE.md` (12 dead refs, incl.
  `src/visdetect/analysis/archive/`) and `docs/DOCUMENTATION_INDEX.md` are
  presented as current.

Caveat: `data/`-rooted rows (14) test existence in **this checkout**; those
targets are gitignored artifacts that may exist on other machines or be
regenerable — for them "dead" means "not materialized here", not "gone".

## Duplication agreement: not-measured, recon restated — and the one divergence

`d6.dup_pair_agreement = not-measured` by design: the full pairwise diff is
deferred because the new repo deletes the copies outright (ADR-005). Recon's
line-level overlap between CLAUDE.md and its doc twins:

| Pair (CLAUDE.md vs) | Recon line-overlap |
|---|---|
| `docs/SCRIPT_TEMPLATE.md` | 88% |
| `docs/GOTCHAS.md` | 78% |
| `docs/NEURO_BEST_PRACTICES.md` | 71% |
| `docs/NORMALIZATION.md` | 0% (independent text, not a copy) |

Duplication is tolerable; divergence is the bug — and the one known divergent
row is the dangerous kind. **`docs/GOTCHAS.md:10`** reads:

> Session name format | DDMMYYYY as integer (e.g., `7072025` = July 7, 2025).
> Use `parse_session_date()` and `chronological_sort()`.

It recommends the **integer** session-id form and never mentions
`canonical_session_id()` — i.e. the copy teaches exactly the leading-zero
footgun that CLAUDE.md's own gotchas row spends a paragraph banning (and whose
fallout D4 measured as 15.8k corrupted cache rows). An agent that opens
`docs/GOTCHAS.md` instead of CLAUDE.md is instructed to create the bug.
Related twin, found incidentally: `docs/AI_interaction/copilot-instructions.md`
and `.github/copilot-instructions.md` (the copy GitHub Copilot actually loads;
outside this scan's scope) are **not byte-identical** — same divergence class.

## Model ids in primary `.claude` prose: 3 mentions, one file

Worktrees excluded (1,150 of 1,159 `.md` files under `.claude/` are duplicate
checkouts). All three mentions sit in
`.claude/skills/harden-result/SKILL.md`: "Opus 4.8" (lines 3, 252) and
`claude-opus-4-8` (line 252). Both are stale against the fleet standard
(Fable 5 / `claude-fable-5[1m]` in both settings files since 2026-08-10). At
runtime the `CLAUDE_CODE_SUBAGENT_MODEL` env var overrides the prose, so
dispatches land on the right model anyway — but the skill text pins its
adversarial-refutation tier to a superseded model by name. Scope note: the
survey is `.md`-prose only; model ids in `settings.json` are intentionally
out of scope (they are configuration, not prose).

## Canonical-authority claimants: 4 files

`CLAUDE.md`, `docs/AI_interaction/copilot-instructions.md`,
`docs/AI_interaction/DOCUMENTATION_CONSOLIDATION_SUMMARY.md`,
`docs/AI_interaction/handoff_refactor_may2026.md` — matching the recon count
of 4. Semantics caveat: the probe is keyword presence
(`canonical|authoritative|single source of truth`), an upper-bound proxy for
"claims to be THE instruction file"; e.g. CLAUDE.md matches on calling
`config.py` the single source of truth. The operative fact stands as recorded:
**only CLAUDE.md is actually loaded by the harness** — the other three
instruct nobody except the reader who happens to open them (and the
Copilot-loaded `.github/` twin diverges from the copy kept here).

## Retraction-marker survey: 4 of 80 `docs/superpowers/` files

`grep -E 'RETRACTED|REFUTED|SUPERSEDED'` (case-sensitive) per file under
`docs/superpowers/` (80 files; the spec's recon counted 76 before the audit
added its own plan/spec). Recorded as
`d6.superpowers.retraction_markers = 4`:

| File | What the marker actually is |
|---|---|
| `plans/2026-08-09-subproject-0-audit.md` | the audit's own plan, citing the existing retraction record |
| `specs/2026-07-31-S1-session-grouping-learning-axis-design.md` | **genuine terminal-status header**: "What was tested and REFUTED (do not retry)" |
| `specs/2026-08-03-QC1-trial-event-alignment-repair-design.md` | adversarial-review outcome ("Circularity hypothesis REFUTED") — an internal result, not a file status |
| `specs/2026-08-05-new-repo-master-design.md` | cites retracted findings in its defect inventory |

The count is an **upper bound on terminal-status labelling, and the real
finding is the complement**: the corpus has no status-header convention. Work
the memory layer records as dead — the transient/sustained **state** result
(RETRACTED, raw-Hz artifact), the "sustained StimSens = expert signature"
claim (REFUTED), single-pulse TF responsiveness (SUPERSEDED by the GLM) — is
not labelled in the corresponding spec/plan files; the
`2026-07-07-transient-sustained-spectrum-celltype-design` spec mentions the
retraction only as a lowercase aside about *prior* work. A reader (human or
agent) with only `docs/superpowers/` cannot tell live designs from dead ones;
the terminal-status record lives entirely in the private memory layer.

## `docs/science` vs the memory layer: 4 of 12 stale — and the disease is milder here

The retraction-marker survey above tests `docs/superpowers/` (designs and plans).
This section tests the same disease in `docs/science/` (results write-ups): for
each results doc, does a claim in it get walked back in the project's **memory
layer** — which lives outside the repo at
`C:\Users\Ben\.claude\projects\e--python-analysis-…\memory\` — with no marker in
the doc itself? Judgment + grep, not a script; the memory notes were read
read-only. Scope: the 11 dated results docs plus `QUESTION_INDEX.md`
(= the spec's 12); the three `state_labeler/` files were surveyed too and are
clean. Recorded as `d6.science.stale_docs = 4`.

**Both cases the spec named as known resolve as ALREADY MARKED.** This is the
headline, and it inverts the expectation:

- *Transient/sustained state retraction* — fully marked in-doc.
  `2026-07-02-transient-sustained-tf-cells.md` carries it at line 8 ("A follow-up
  state analysis was **null**"), lines 20–21 ("the state claim (§7) was retracted
  as a firing-rate artifact"), and the §7 heading itself (line 185, "**NULL**
  (corrected)"), with the full raw-Hz-artifact post-mortem at line 190. Memory
  side: `tf_transient_sustained_state_jul2026`.
- *"Sustained StimSens = expert signature" refutation* — **never asserted in
  `docs/science` at all.** The claim lived in a design spec
  (`docs/superpowers/specs/2026-07-31-S1-…-design.md:30`) and was refuted in that
  same spec at line 36; `QUESTION_INDEX.md:67` records the refutation explicitly
  ("REFUTED en route: 'sustained StimSens bouts' is NOT an expert signature").
  Memory side: `session_grouping_learning_axis_jul2026`.

So the predicted count for these two is **zero**. The four stale docs found are
different ones:

| Doc:line | Claim as written | Walked back by | Marked? |
|---|---|---|---|
| `QUESTION_INDEX.md:49` | "**VMS engagement-gated** (StimSens≫Disengaged)" | memory `b10_impulsivity_kernel_jul2026` — "the earlier 'VMS strongly engagement-gated' was **per-trial PSEUDOREPLICATION** … engagement modulation **UNRESOLVED**" | no |
| `2026-06-17-post-tf-null-research-direction.md:4, 48` | "…across BG_046 DMS, BG_031 striatum, **BG_039 cortex**, and **BG_038 GPe**"; "Batch on **BG_039 (cortex/M2)** — tests whether the TF-null is *regional*" | memory `multisubject_event_psth_readiness_jun2026` — BG_039 is "dorsal CP striatum (DMS…) **Pool-compatible with BG_046**"; BG_038 is cortex (MOp/SSp), "planning-doc 'GPe' = shank target not recording site" (resolved 2026-06-30) | no |
| `2026-07-02-transient-sustained-tf-cells.md:108, 116` | "Every cell is TF-locked (diagonal = latency tiling)"; "~50 % of responsive cells are **suppression-type**"; "Fast pulses subsampled to 600/session (thousands occur; **irrelevant to the mean**)" | memory `tf_pulse_peth_circularity_bug_jul2026` — that heatmap is "**STILL SUSPECT (not yet audited)**", its tiling/TF-locked claims "probably sorting artifacts"; suppression fraction corrected 49 % → **36.9 %**; `PULSE_CAP=600` "threw away ~98.5 % of the ~41k fast pulses/session" | **§7 only** |
| `2026-07-07-transient-sustained-spectrum-celltype.md:52, 172` | "`pulse_fwhm` … Spearman 0.11 … — **inherent, not a bug**"; mandatory caveat 2: "Describe the spectrum as skewed/heavy-tailed, **not clean lognormal**" | memory `tf_pulse_peth_circularity_bug_jul2026` (the ρ=0.11 weakness "is probably just noise" — same 600-cap) and `tf_spectrum_celltype_orthogonality_jul2026` ("a direct lognorm-vs-gamma MLE/AIC/KS fit later **FAVORED lognormal in all 3 regions** … OK to call `interp_fwhm` ~lognormal now") | no |

Two details make the first two rows firmer than a prose comparison would:

- **The B10 index row was never touched by its own correction.** It is
  byte-identical since `39c19db` (2026-07-01); the two commits that landed the
  correction — `e16fcd5` "honest CI corrections" and `bfefa87` "paired
  within-session tests settle the two suggestive contrasts", both 2026-07-02 —
  edited only `2026-07-01-B10-results.md` and `2026-07-02-B10-RESULTS-explained.md`.
  `QUESTION_INDEX.md` was then edited three further times (2026-07-21, 2026-08-03
  ×2) and the stale clause survived every pass. **Both results docs it links to
  say the opposite** ("Do NOT say 'tracking switches off when disengaged'"), so
  the index contradicts its own children — the failure is the summary layer, not
  the science.
- **The region error is load-bearing, not cosmetic.** The 2026-06-17 doc's
  argument is "**Four regions including cortex all at ≈0% ⇒ the floor reflects the
  metric, not the biology**" (line 6) and its recommended cheap control is to
  batch BG_039 to "test whether the TF-null is *regional*". BG_039 is the same
  region as BG_046, so that control was void as designed. Every `docs/science`
  doc from 2026-07-01 onward states "BG_046, BG_039 = DMS" — the corpus
  contradicts itself and only the older doc is wrong.

### The honest read: `docs/science` is markedly healthier than `docs/superpowers`

Eight of twelve are clean, and three of the four hits are stale in **one clause**
rather than wholesale. The corpus has a real correction habit that
`docs/superpowers/` lacks: `2026-08-03-early-lick-learning-results.md` carries an
explicit "❌ Corrected overclaim" section (§4) plus the "'Naive' is not naive"
S1 caveat; `2026-07-02-B10-RESULTS-explained.md` ends with a "Corrections applied
during verification" section; `2026-07-20-preparatory-activity-transient-sustained.md`
reframes its own Claim 1 from latency to magnitude and demotes Claim 2 to WEAK;
`2026-07-21-sensory-motor-geometry-regulation-null.md` is a write-up *of* a
result collapsing under its own confound battery. Row 3 above is the
instructive case: the same file that contains the corpus's **best** in-doc
retraction (§7) leaves §3 unmarked — corrections here are attached to the claim
that was attacked, not swept across the document.

### The bigger exposure is a different disease, and it is not in this count

`d6.science.stale_docs` counts walked-back **claims**. It deliberately does not
count invalidated **inputs**, which are more widespread and marked nowhere in
`docs/science`:

- **The stale TF-responsive registry.** `data/cache/tf_responsive/README.md`
  carries a prominent "⚠️ STALE — these registries predate the lick-channel fix
  (2026-07-31)" banner: the lick nuisance regressor changed on *every session of
  all three subjects*, so "borderline `resp_log2` calls will flip" and the
  "VMS 5.3 % > DMS 2.8 % / 3.1 %" headline "must be re-derived" (see
  `d4.tfresp.flips`). **Six** `docs/science` docs rest on that registry —
  `2026-07-01-B9`, `2026-07-01-B10`, `2026-07-02-B10-RESULTS-explained`,
  `2026-07-02-transient-sustained-tf-cells`, `2026-07-07-…-spectrum-celltype`,
  `2026-07-20-preparatory-activity-…` — and **none mentions it**. The cache knows
  it is stale; the write-ups do not.
- **Corrupted caches under a live results doc.**
  `2026-08-03-early-lick-learning-results.md` §7 lists
  `data/cache/behavior/early_lick_*.csv` as its artefacts. Three of those files
  (`early_lick_repl_BG_046/039/031.csv`) plus the three `fa_hazard_trials_*.csv`
  are exactly the six offenders in the RED integrity test
  (`d4.ids.integrity_test_red`, 15,802 stripped-id rows). No note in the doc.
- **QC1.** `QUESTION_INDEX.md:66` states that 23 pkls make "`ni_events`-aligned
  NEURAL analyses invalid on them", naming BG_046 `20082025` and `05092025_b`
  (both Expert, both on the primary subject). No neural results doc cites it.

These are not counted because none is a claim someone retracted — they are
inputs someone later found broken. But for the new repo they are the more
dangerous class: a reader who checks each doc for a retraction banner finds
none and concludes the work is current.

## Skill trigger-overlap: judgment from the seven descriptions

The spec's D6 bullet ("pairwise trigger-overlap between the 7 skills; which
pairs can both fire and what decides") was left unmeasured by recon; this
table is the audit's judgment from the seven `SKILL.md` descriptions — no
measurement id, no instrumentation of actual skill firing.

| Pair | Both can fire on | What decides |
|---|---|---|
| `codebase-auditor` / `pre-commit-checker` | "check / review my code" | scope + timing: diff-only fast gate vs whole-repo audit. Their checklists substantially duplicate each other (event alignment, constants, normalization, unit selection) — the same drift hazard as the doc copies above |
| `harden-result` / `research-statistician` | "is this result solid", "sanity check the effect" | object: choosing/implementing a test (statistician) vs adversarially battering a finished claim (harden-result, which embeds its own stats battery) |
| `harden-result` / `research-notes-summarizer` | "write up the result" | order: harden-result claims the BEFORE-writing slot in its description; nothing but that prose enforces the ordering |
| `research-statistician` / `research-visualizer` | "add significance to the figure" | artifact: the annotation/layout (visualizer) vs the test behind the stars (statistician) |
| `codebase-auditor` / `harden-result` | "verify this analysis" | target: code correctness with file:line output vs a scientific claim with controls + refutation |
| `analysis-runner` / all of the above | "run the controls / regenerate the figure" | compositional, not competitive: runner is the execution utility the others delegate to |

Judgment: the overlaps are mostly resolved by object/scope wording in the
descriptions themselves. The two genuine hazards are (a) the
auditor/pre-commit **checklist duplication**, which can silently diverge
exactly like the CLAUDE.md doc twins, and (b) the harden-result/summarizer
**ordering** being convention-only — the one place where firing the wrong
skill first lets an unhardened claim get written up.
