# New Repo — Master Design and Architecture Decision Records

**Date:** 2026-08-05
**Status:** Approved (design); sub-project specs pending
**Authors:** Project owner + Claude (Opus 5)
**Supersedes:** `docs/REFACTOR_PLAN.md` (in-place refactor; never completed)

---

## 1. Problem statement

The current workspace works, and has produced real, adversarially-verified science. But it has
accumulated the failure mode it explicitly set out to avoid: **the same definition existing in more
than one place, with no mechanism preventing divergence.**

`CLAUDE.md` declares `src/visdetect/analysis/constants.py` the single source of truth. It is not.
A first-pass scan found stage palettes, stage orderings and change-size lists re-defined in seven
files outside the canonical modules. Nothing detected this, because nothing was ever built to
detect it — the rule lived in prose, and prose does not fail a build.

The same pattern repeats at every level:

| Level | Duplication |
|---|---|
| Constants | canonical `constants.py` + `config.py`, shadowed in ≥7 script/library files |
| Package | `src/visdetect/session.py` and `io.py` alongside `core/session.py` and `core/io.py` |
| Package | `src/visdetect/suite/` shimming an `analysis_suite/` archived in July 2026 |
| Scripts | 377 files across 30 topic dirs, several topics superseded but not removed |
| Docs | `CLAUDE.md` restates `docs/NORMALIZATION.md`, `NEURO_BEST_PRACTICES.md`, `GOTCHAS.md` |
| Docs | `CLAUDE.md` hand-copies the constants table from `constants.py` — can silently go stale |
| Identity | session ids derived from filenames at many call sites, normalized by opt-in helpers |
| Memory | ~60 notes, several carrying findings since retracted or refuted |

The identity problems deserve special mention because they are one bug wearing several hats.
Leading-zero day loss (`01072025` → `1072025`), 6-digit vs 8-digit date forms across subjects,
and `_b`/`_c`/`_v2` suffixed session twins are all consequences of a single root cause:
**identity is derived from filenames, repeatedly, at many call sites, by code that must remember
to call a normalizer.**

## 2. Goal

A new repository, built from the conclusions of a deep audit, in which:

- every definition exists exactly once, and divergence is mechanically prevented rather than merely
  discouraged;
- the tooling, documentation and AI layer are tailored to this specific project and cannot drift
  from the code they describe;
- the validated science already produced is preserved, not re-litigated.

## 3. Success criteria

The new repo is "done" for a given sub-project when the corresponding criteria hold and are
demonstrated by an executing check, not by argument.

| # | Criterion | Demonstrated by |
|---|---|---|
| S1 | Every constant, palette, window and threshold has exactly one definition | CI gate fails on any shadow definition or inline magic number in the enforced set |
| S2 | Subject/session identity is typed; filenames are parsed in exactly one place | CI gate bans filename parsing outside the registry builder and bans `str`/`int` session ids in public signatures |
| S3 | Layer boundaries hold (`ingest → qc/tracking → analysis → figures`, no upward imports) | Import-contract check in CI |
| S4 | Every numerical difference from the old repo is attributed; none is unexplained | Per-component difference report (ADR-009) with each delta tagged `defect-fixed` / `known-defect` / `design-change`; any `unexplained` blocks |
| S5 | Every figure and results artefact carries five-part provenance: **execution log** (authoritative), code, environment (language + every package version), inputs (registry snapshot + decision-log revision), and git sha | Provenance sidecar written by the artefact layer; check that fails on an artefact without one. **Where the recorded code and the execution log disagree, the log is authoritative** — the current repo's failure is not unrecorded scripts but scripts whose stated behaviour diverged from what ran |
| S6 | The AI layer's factual claims cannot go stale | Generator script + CI check that regeneration produces no diff |
| S7 | Carried-over memory is true of the new repo | Every migrated note passed the verification gate; failures archived with a RETRACTED marker |
| S8 | Analysis is installable without the heavy stack | `pip install -e .[analysis]` succeeds in a clean env with no UnitMatch/GPU/MATLAB dependency |

## 4. Non-goals

- Re-deriving scientific conclusions. Hardened results stand; this is an engineering exercise in
  service of them.
- Changing the scientific questions or the analysis plan.
- Rewriting third-party pipelines (Kilosort 4, UnitMatch, DANT, Bombcell). They are wrapped, not
  reimplemented.
- Multi-user, production, or public-release infrastructure. This is a single-scientist research repo.
- Migrating raw data. Raw data stays where it is; the new repo references it.

---

## 5. Architecture Decision Records

Each record states the decision, the alternatives considered, why this one won, and what it commits
us to. These are immutable: to change one, add a superseding record.

### ADR-001 — Clean-room foundation, analysis ported behind an equivalence gate

> ⚠️ **The gate clause of this record is superseded by ADR-009 (2026-08-06).** The clean-room
> foundation decision stands unchanged; the definition of the gate does not. Read ADR-009 for the
> operative rule.

**Decision.** Write the foundation layer (registry, identity, constants, config, session model, IO,
QC) from scratch against a written spec. Port each analysis module across a gate: it lands only if
it reproduces the old repo's numbers on real data, or differs in a way declared and justified in
advance.

**Alternatives.** (a) Curated migration — move modules largely as-is. (b) Full clean-room rebuild
including analysis. (c) Strangler-fig refactor in place, no new repo.

**Why.** The inconsistency is concentrated in the foundation, which is small and cheap to rewrite.
The analysis layer embodies months of adversarial verification — re-deriving it would be expensive
and would risk losing correctness that is not fully captured in tests. (b) re-litigates that work;
(a) imports the design decisions that caused the drift; (c) never delivers the clean slate.

**Commits us to.** Maintaining a pinned oracle of the old repo for the duration of the port, and to
the known-defect register (§6), without which the gate would enforce reproduction of known-wrong
numbers.

### ADR-002 — One repo, hard internal layers, optional dependency extras

**Decision.** All layers live in one repository — ingest, QC/tracking, analysis, figures — organised
as enforced layers with separate optional dependency groups, so the analysis layer installs without
the UnitMatch/GPU/MATLAB stack.

**Alternatives.** (a) Analysis-only new repo consuming a frozen producer repo. (b) Two repos plus a
published shared core package. (c) One flat repo, no layering.

**Why.** The entire objective is one definition in one place. (a) and (b) both put constants and
session identity on a repo boundary, which is exactly the failure being cured — and for a solo
project the cross-repo version pinning is pure overhead. (c) forfeits the ability to enforce that
analysis never reaches into ingest internals, and forces the heavy dependency set on every run.

**Commits us to.** Maintaining dependency extras and an import-contract check; accepting that the
repo is larger than an analysis-only repo would be.

### ADR-003 — Mechanical gates first, AI auditor second

**Decision.** Anything mechanically checkable becomes a build-breaking gate in CI and pre-commit:
constants defined outside the registry, layer-violating imports, raw session ids used as keys,
per-condition baselines, missing normalization, artefacts without provenance. The AI auditor skill
covers only what a gate cannot decide — whether a test is appropriate, whether an analysis is
circular, whether an effect size is meaningful.

**Auditor checklist** (adopted 2026-08-06 from Claude Science's reviewer, whose categories map
almost exactly onto defects this audit found). The auditor checks claims against the **execution
record**, and does not re-run analyses:

- a result reported as computed when nothing ran *(cf. the ten scripts writing into a deleted
  `vd_tf_bg046/` tree, which "succeed" and produce nothing)*;
- a value that contradicts the file it came from *(cf. `CLAUDE.md`'s constants table vs
  `constants.py`)*;
- a citation that does not support the claim attributed to it;
- a plan step recorded as done that was not completed;
- a conclusion not supported by the method used *(cf. the retracted transient/sustained result)*.

Project-specific checks may be **added** to this list but may never remove or weaken a member of it.

**Alternatives.** (a) Gates only, no AI review. (b) Runtime-enforced frozen registry that raises on
unregistered access. (c) Convention and review only.

**Why.** (c) is the status quo and demonstrably drifted. (a) cannot catch scientifically wrong but
syntactically legal code, which in this project is the dangerous class — a circular baseline is
valid Python. (b) is the strongest guarantee but is invasive and hostile to exploratory notebook
work, which is a real part of the workflow.

**Commits us to.** Writing and maintaining custom AST checks; accepting some friction on commits;
and to writing the gates *before* the code they govern (see build order, §7).

### ADR-004 — Generated registry is the sole authority for identity

**Decision.** A generated registry table — subject, session, date, variant, artefact paths,
provenance — is the only place identity is derived from filenames, and it is derived once, at
ingest. All downstream code takes typed `SubjectID` and `SessionKey` objects. A gate bans filename
parsing outside the registry builder and bans bare `str`/`int` session ids in public signatures.

**Alternatives.** (a) Typed ids re-derived at each load. (b) Registry table but plain string ids.
(c) Hardened canonical-helper functions on strings (evolved status quo).

**Why.** Every recurring identity bug in this project's history — leading-zero day loss, 6- vs
8-digit cross-subject misjoins, suffixed twins — comes from deriving identity from filenames at many
call sites. (c) and (a) leave that derivation distributed and opt-in. (b) fixes the source of truth
but still permits a raw `int` to be used as a join key, which is the specific mechanism by which
day-1–9 sessions silently vanish from results.

**Commits us to.** Keeping the registry fresh (it becomes a build artefact with its own staleness
check), and to a typed-id API that is slightly more verbose at call sites than bare strings.

### ADR-005 — The AI layer's factual sections are generated from code

**Decision.** `CLAUDE.md`'s factual content — constants table, module map, layer rules, gate list —
is generated from the code registry by a checked-in script, with CI failing when regeneration
produces a diff. Prose sections (scientific rules, project identity, judgment guidance) remain
hand-written and single-sourced: they live in one document and are referenced, never restated.
Skills are rebuilt from audit findings. Memory notes carry over only through a verification gate.

**Alternatives.** (a) Hand-written but curated and single-sourced. (b) Port and adapt the existing
AI layer wholesale. (c) Fresh AI layer with empty memory.

**Why.** `CLAUDE.md` currently hand-copies the constants table and restates three other documents;
it can go stale silently, and a stale instruction file actively misleads every future session.
(a) fixes duplication but leaves the copied table able to rot. (b) carries the duplication and the
retracted findings into the clean repo. (c) discards genuinely load-bearing context — the
no-compute-over-Samba rule, the worktree-junction data-loss lesson, the lick-channel defect — that
would be expensive and risky to relearn.

**Commits us to.** A generator script as a maintained build tool, and to a one-time memory
verification pass over ~60 notes.

### ADR-006 — Specify fully, then build (with an empirical audit as the grounding)

**Decision.** Complete the specification of every layer before implementing, per the user's
explicit choice. To prevent this becoming big-design-up-front against imagined data, the audit that
precedes the spec is **empirical**: it loads real sessions and measures real behaviour, so the spec
is written against measured reality rather than a filename inventory.

**Alternatives.** (a) Vertical slice first, widening — build the thinnest complete path to one
finished figure, then generalise. (b) Full foundation, then port.

**Why.** This was the user's decision, made with the trade-offs stated. The recommendation on
record was (a), on the grounds that consistency is only provable by execution and that
big-design-up-front has already failed once here (`docs/REFACTOR_PLAN.md`). That concern is
recorded, not overridden.

**Commits us to.** An audit substantially deeper and more expensive than a structural survey, and to
the two-stage spec process (§8). **Mitigation:** the audit must produce measured quantities, not
just inventories; and the cross-spec consistency pass (§7, sub-project 6) is mandatory rather than
optional, because under this strategy no earlier execution will have caught spec contradictions.

### ADR-007 — Nothing lands without a plain-language approval packet

**Decision.** Every component entering the new repo is presented for approval *before* it lands,
as a fixed-shape packet. The reviewable unit is **one module or one gate** — not a file, not a
layer. Roughly 20–40 approvals across the build.

The packet has six required sections:

| Section | Content |
|---|---|
| What it is | Plain language, one short paragraph, no unglossed jargon |
| Why it exists | What breaks without it |
| Provenance | New / copied verbatim / ported-with-changes — and exactly what changed and why |
| How we know it's right | The specific check, **run**, with its actual output |
| Blast radius | What depends on it; what it depends on |
| Decision | Approve / change / reject |

Two rules make this a gate rather than a ceremony:

1. **No exemption for "trivial".** The defect that silently nulls every QC profile in the current
   repo is a single character: `parents[1]` where `parents[3]` was meant. Triviality is not a
   predictor of harm.
2. **"How we know it's right" must contain executed output, never an argument.** This is the
   existing project rule that fixes are proven by execution, applied to construction.

**Alternatives.** (a) Layer-level approval — 6–8 packets, far less interruption, but each is large
enough that genuine review degrades into assent. (b) File-level — over a hundred packets; review
quality collapses from fatigue. (c) Post-hoc review after a layer is built — cheapest, but by then
the cost of reversing a decision is highest.

**Why.** In scientific code a subtly wrong constant is worse than a crash, because it produces a
plausible number instead of an error. The audit found several such: four disagreeing firing-rate
floors, a 5×-wrong sampling period, two `CHANGE_SIZES` with different membership. Each would pass
any test that did not know the right answer. A human who understands the science is the only
reliable check, and that check only works if the explanation is legible.

**Commits us to.** Slower construction, and to writing every explanation in plain language —
including for machinery whose correctness is obvious to the author.

### ADR-008 — New repo is a sibling directory and contains no junctions

**Decision.** The new repo lives at `E:\python_analysis\git_repos\visdetect` — a sibling of
`vis_detect_analysis_Sep2025`, never nested inside it. It contains **no NTFS junctions or symlinks**
to data. Large inputs (`data/pkls`, `data/unit_match`, `data/anatomy`, existing `FIGURES/`) are
reached through a single configured root, per ADR-004.

**Alternatives.** (a) Nested inside the current repo. (b) A dated name in the existing family
(`vis_detect_analysis_2026`). (c) A different physical drive.

**Why.** Three independent reasons:

- **Nesting reproduces a bug the audit just measured.** `check_refactor_guardrails.py` reports 1,375
  violations of which ~1,157 are phantom, solely because it walks into `.claude/worktrees/` and
  finds other copies of the codebase. Every tree-walking tool — linters, pytest collection, and the
  gates of sub-project 2 — would do the same to a nested repo.
- **No junctions closes the June-2026 data-loss class permanently.** That incident occurred because
  `git worktree remove` followed a junction into the primary data. If the repo contains no links,
  no recursive delete can traverse one, regardless of who runs it.
- **The date-stamp naming is itself a defect.** `Sep2025` records when the repo started, not what it
  holds — and it already misleads, since the repo contains 2023–2026 work. `visdetect` matches the
  package name and cannot go stale.

A different drive was considered and rejected: the 30.5 GB of pkls and 36 GB of figures are on `E:`,
so cross-drive reads would tax every analysis. Off-disk safety is the remote's job, not the
filesystem's.

**Commits us to.** Configuring a data root explicitly on every machine, rather than relying on paths
that happen to resolve.

### ADR-009 — Explained-difference gate (supersedes the gate clause of ADR-001)

**Decision.** A ported component does **not** have to reproduce the old repo's numbers. It must
**explain every difference**. Each component reports its delta against the old output and attributes
each one to exactly one of:

| Attribution | Meaning |
|---|---|
| `defect-fixed` | The old number was wrong; the register (§6) says so, or this port proves it |
| `known-defect` | A register entry already predicted this difference, in this direction |
| `design-change` | An intentional, specified change (different normalization, different window) |
| `unexplained` | **Blocks the component. The only failing verdict.** |

Reproduction is not the target and bit-identity is not a virtue. An *unexplained* difference is.

**Alternatives.** (a) ADR-001 as written — reproduction is the default, difference the declared
exception. (b) A pure fresh-build with no comparison at all: run everything new, treat old outputs
as history.

**Why.** (a) has the gate pointing the wrong way. The audit found that a substantial number of the
old repo's outputs are simply wrong — every named QC profile silently no-ops, the TF sampling period
is 5× too coarse, 15,802 session-id rows are corrupted across six live caches, and ten scripts have
been writing into a deleted directory. Making reproduction the default would spend the port's effort
proving we can regenerate defects, and would create pressure to preserve them.

(b) fixes that but discards the one thing comparison is genuinely good for: catching the regression
you did **not** intend. If every difference is expected, an accidental bug is indistinguishable from
progress — and in this codebase the characteristic failure is a *plausible wrong number*, not a
crash. The audit's own examples make the point: four disagreeing firing-rate floors and two
`CHANGE_SIZES` with different membership would each pass any test that did not already know the
right answer.

The explained-difference rule takes the useful half of each. The new repo is free to be correct
rather than compatible, while every movement still has to be accounted for by a human who
understands the science.

**Commits us to.** A per-component difference report as part of the ADR-007 packet — meaning the
"how we know it's right" section carries both the executed output *and* the attributed delta. It
also means the old repo must remain readable (not merely archived) for the duration of the port,
and that the known-defect register is consulted per component rather than once.

**Note on scope.** This changes what "landing" means for every ported component, but does not change
ADR-001's clean-room foundation decision, which stands.

### ADR-010 — A paved road for new analyses

**Decision.** New analyses are not written from a blank file. Each is created by a scaffold command
from a template, and must satisfy a written **analysis contract**, enforced by the gates of
sub-project 2:

| The contract requires | Enforced by |
|---|---|
| Session set obtained from the registry, never by globbing or parsing filenames | Gate: filename parsing banned outside the registry builder (ADR-004) |
| Constants imported from the canonical module, never retyped | Gate: shadow-definition check (ADR-003) |
| Outputs written through the artefact API, never a hardcoded path | Gate: artefact without provenance fails (S5) |
| Declares which layer it belongs to; imports respect layer direction | Import-contract check |
| At least one test, offline-runnable | Coverage gate |
| Docstring states the scientific question and the literature grounding it | Auditor (judgment, not mechanical) |
| Registered in the analysis index | Index-freshness check |

**Alternatives.** (a) Template plus documentation, no scaffold and no enforcement — the status quo.
(b) Code review only. (c) No convention; let each analysis find its own shape.

**Why.** (a) is precisely what the current repo has, and it produced: 378 scripts across 30 topic
directories, 20+ mutually inconsistent `sys.path` idioms, READMEs in 5 of 30 directories,
`partial_spearman` reimplemented **seven times in three mathematically different forms** (two
`spearmanr`-on-residuals variants and one `np.corrcoef`-on-residuals, which is a different
estimator), `save_fig` defined 8 times, `make_figure` 13 times, and 245 of 378 scripts importing
none of the canonical config modules.

None of that is carelessness. It is what happens when **the easy way and the right way are
different**. A repo with corrected constants but the same absent scaffolding re-sprawls within a
year, and the corrected constants go stale exactly as they did before.

This ADR exists because the rest of the design is about repairing the past and porting the present;
without it, nothing addresses the future — which is the stated reason for the rebuild.

**Commits us to.** Maintaining a scaffold and template, and to the risk that the template itself
drifts from the gates. Mitigation: **the template is generated from the same source as the gates**,
so a rule change updates both or fails CI — the same principle as ADR-005.

### ADR-011 — Data decisions are versioned, append-only, and attributed

**Decision.** The session roster and every inclusion/exclusion decision are version-controlled in an
append-only **decision log**: what changed, why, when, and who decided. Staging manifests become
**build artefacts** derived from the registry plus the decision log — generated, never hand-edited.

**Alternatives.** (a) Track the existing hand-edited manifests in git. (b) A database. (c) Status
quo: gitignored CSVs edited in place.

**Why.** ADR-007 gates *code* components, but which sessions are included, which units pass QC, and
why a session was dropped are **scientific** decisions with a larger effect on results than most
code. Today they have no home at all: the seven staging manifests are gitignored
(`.gitignore:46 data/*`), untracked, and mutated in place. The `.bak` files prove it — on 2026-08-03
BG_031 lost the row `19052025,Expert` and BG_039 lost a duplicate `23042025`, with **no record
anywhere of who decided that or why**. Consequently no figure or cache in the repo can be
reproduced, because the exact roster that produced it was never versioned.

(a) is a real improvement but preserves hand-editing, so a silent edit stays possible and the
*reason* still goes unrecorded. (b) adds infrastructure without adding accountability; the problem is
not storage, it is attribution.

Generated-not-edited also closes the divergence the audit measured: 28 scripts read a manifest CSV
directly rather than through `load_staging_manifest()`, bypassing `SESSION_FILTER` entirely, so two
figures in the same paper can silently disagree on n.

**Commits us to.** A decision-log format, and the discipline of writing a reason whenever a session
is excluded — including when the reason is "obviously bad recording".

---

## 6. The known-defect register

The equivalence gate of ADR-001 cannot be applied blindly, because **some of the old repo's numbers
are known to be wrong.** Reproducing them would encode the defect.

Examples already known at design time:

- the trial/event alignment defect currently under repair (`trial_event_index`, QC1 work);
- the lick-channel extraction defect — a March 2026 MATLAB re-extraction that under-detects licks
  by 10–40× in 33 sessions;
- the TF-pulse PETH circularity bug, and caches computed before it was fixed;
- results since retracted (the transient/sustained state result, a raw-Hz artifact null after
  FR-normalization) or refuted (the "sustained StimSens = expert signature" claim).

Therefore the audit must produce a **known-defect register**: for every analysis module, the defects
known to affect it and the direction each one moves its outputs. Under ADR-009 the register is not a
list of pass/fail verdicts but the **evidence base for attribution** — when a ported component's
output differs, the register is what lets that difference be tagged `known-defect` rather than
`unexplained`. A module whose defect status cannot be determined is **quarantined**: it is specified
explicitly rather than ported on assumption, because every one of its deltas would otherwise be
unexplained by definition.

This register is a first-class deliverable of sub-project 0, and a precondition for sub-project 3.

---

## 7. Decomposition and build order

Too large for one spec. Six sub-projects, each with its own spec → plan → build cycle.

| # | Sub-project | Produces | Depends on |
|---|---|---|---|
| **−1** | **Secure the work at risk** | Off-disk copy of everything that exists in only one place: 139 unpushed commits, unmerged branches, uncommitted working trees, two stash-tags, and the gitignored hand-labelled artefacts no code can regenerate | — |
| **0** | Deep empirical audit | Findings corpus: definition inventory, duplication map, layering graph, dead-code census, artefact provenance survey, memory-claim verification, known-defect register, real-data measurements | −1 |
| **1** | Foundation | Registry + typed identity, constants, config, session model, IO, QC | 0 |
| **2** | Enforcement | AST gates, import contracts, pre-commit, CI | 1 |
| **3** | Analysis layer | Alignment, normalization, statistics, behaviour, spikes — ported behind the equivalence gate | 1, 2 |
| **4** | Figures & artefacts | Plot system, palettes, cache and provenance layout | 3 |
| **5** | AI layer | Generated `CLAUDE.md`, skills, hooks, curated memory | 1–4 |
| **6** | Migration & decommission | Cutover, freeze of the old repo, artefact preservation, cross-spec consistency pass | all |

Build order is −1 → 6 in sequence, with one deliberate deviation: **enforcement (2) precedes the
analysis port (3).** Gates written after the code they govern get bent to fit it. Written first,
they define what "landing" means, and every ported module clears a bar that already exists.

**Sub-project −1 was added on 2026-08-06**, after the scoping recon measured the exposure: 139
commits exist only on this disk (`origin/main` last updated 2026-07-10), there are zero stashes, the
SSH agent is not running so no push can be attempted, and every hand-labelled artefact — the
4,725-unit TF labelling, 202 state-tag files, the blinded session-sorter output, the pupil labels —
is gitignored. Auditing a repo whose only copy is local is defensible. Freezing or replacing one is
not. This sub-project is a precondition, not a phase.

---

## 8. The two-stage spec process

Under ADR-006 the specification carries all the risk, and specs 1–6 have a hard data dependency on
the audit. They therefore cannot honestly be written before it exists. The process is:

1. **Stage 1 (this document + the sub-project 0 spec).** Master design, decision records, and the
   audit's own specification. No dependencies.
2. **Audit runs.** Produces the findings corpus, committed to the repo as *curated structured
   findings* — not raw dumps — so that it fits comfortably in a single working session.
3. **Stage 2.** Specs 1–6 written *together, in one dedicated session*, from the corpus. Writing
   them together is what buys mutual consistency; writing them from the corpus is what buys
   grounding. Each is committed as it is completed, so that context summarization cannot lose them.
4. **Cross-spec consistency pass.** A dedicated review reading all six specs plus this document,
   reporting contradictions, gaps and unstated assumptions.

Coherence across sessions is carried by **committed artefacts, not conversation context.**
Conversation context is auto-summarized as it grows, is not reviewable, is not diffable, and cannot
be cited by a later session. This document exists to be the durable carrier of the reasoning above.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| Spec-everything-first drifts from reality (has failed here before) | Audit is empirical, not structural; mandatory cross-spec consistency pass; ADR-006 records the concern |
| Equivalence gate enforces known-wrong numbers | Known-defect register (§6) is a precondition for sub-project 3 |
| Two repos live simultaneously; work fragments | Old repo frozen at a tagged commit at the start of sub-project 3; no new science in it after that point |
| Migration loses artefacts (`data/`, `FIGURES/` are gitignored; junction hazard on worktrees) | Artefact preservation is an explicit deliverable of sub-project 6, executed before any deletion; junction-aware deletion guard carried over from the current repo's hooks |
| Audit corpus too large to use | Curated structured findings mandated by the sub-project 0 spec; raw output kept separately and referenced |
| Gates too strict, exploratory work becomes painful | Gates apply to committed library and script code; notebook/scratch paths are exempt by policy, declared in the sub-project 2 spec |

---

## 10. Open questions

Deferred to the specs that can answer them from evidence.

- Which subjects the new repo must support at v1, and whether BG_012's protocol variants are in
  scope (currently parked).
- Whether the old repo's git history is imported, referenced, or dropped.
- Where the registry lives physically, and how it is refreshed when new sessions arrive.
- What tolerance the equivalence gate uses per analysis class (exact, floating-point, statistical).
- Whether HPC/cluster job submission is in scope for v1 or deferred.
