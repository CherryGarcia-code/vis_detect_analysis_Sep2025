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

> ⚠️ The "off-disk safety is the remote's job" clause is **superseded by ADR-022** (2026-08-07).
> The sibling-location and no-junctions decisions stand.

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

> ⚠️ **Extended 2026-08-07**: the contract is split into two tiers (explore/claim) and the
> near-duplicate AST gate is demoted to a periodic report — see **ADR-020**. The statistical rows
> (pre-specification, pseudoreplication, nulls) are fully specified in **ADR-021**.

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
| **No redefinition of a registered shared helper** | Gate: a must-import registry (`partial_spearman`, `save_fig`, session-id and date helpers, bootstrap / permutation / FDR primitives); a local `def` of a registered name fails the build |
| **No near-duplicate of existing library code** | Gate: AST-normalised structural similarity across the repo; above threshold it fails and names the existing function |
| **Same name, different behaviour** | Gate: **hard fail — a separate and more severe category than duplication** |
| **A reuse statement** — what was searched for, what was found, why it is insufficient | ADR-007 packet section (judgment) |
| **Pre-specification**: the prediction, what would falsify it, and the smallest effect size worth caring about | Spec template; committed to git *before* the analysis runs — timestamped pre-registration at no extra cost |
| **Pseudoreplication declaration**: the clustering variable for every cross-unit test | Gate: a mixed-effects or cluster-robust method is required wherever n_units ≫ n_sessions ≫ n_subjects |
| **A null control** and, for any new estimator, **synthetic recovery** | ADR-014 |

**On reuse specifically.** "Search before writing" is already the first workflow rule in the current
repo's `CLAUDE.md`, and the audit measured exactly what prose achieved: `make_figure` defined 13
times, `process_session` 10, `save_fig` 8, `_stage_map` and `_mwu` 6 each, 22 local date parsers
against 4 files importing the canonical one, 78 ad-hoc `zfill(8)` sites against 35 using
`canonical_session_id`, and 142 files deriving their own repo root against 17 importing `ROOT`.

The severity split matters. Plain duplication wastes effort and drifts. **Same-name-different-
behaviour silently corrupts comparisons**, because a reader reasonably assumes a shared name means
shared meaning. `partial_spearman` is the worked example: two copies compute `spearmanr` on
rank-residuals and one computes `np.corrcoef` on residuals — a *different estimator*, without
re-ranking and without the small-n guard. Two numbers both reported as "partial Spearman rho" are
therefore not comparable, and that statistic underpins the width-vs-coupling and theta-vs-regulation
claims. `CHANGE_SIZES` is the same pathology in constant form. These fail hardest.

Structural similarity catches copy-paste; it cannot catch "conceptually the same analysis, expressed
differently". That is what the reuse statement and a human reader are for.

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

### Preamble to ADR-012 … ADR-014 — the other half of correctness

ADR-001 … ADR-011 guard against **engineering** error: a wrong constant, a broken join, a stale
cache, a duplicated helper. They are necessary and they are not sufficient.

The two failures this project has actually suffered were not engineering failures. The
transient/sustained state result was **retracted** — a raw-Hz artifact that went null once firing
rates were normalised. "Sustained StimSens = expert signature" was **refuted** — present in Naive,
and it collapsed to state occupancy. Both were produced by correct code. Both were analysis-choice
artifacts, and no gate written so far would have caught either.

ADR-012 … ADR-014 address that half.

### ADR-012 — Confirmatory cohort: mechanism built, activation deferred

> ⚠️ **Amended 2026-08-07 (ADR-020)**: the dormant loader-refusal guard is replaced by a
> session-access log; the `cohort` field and the deferral stand. ADR-021's design-sensitivity
> module supplies the quantitative revisit trigger.

**Status: accepted, activation deferred (decided 2026-08-07).**

**Decision.** The registry (ADR-004) carries a `cohort` field per session — `discovery` or
`confirmatory` — and the loader is capable of refusing to serve confirmatory sessions unless the
calling analysis declares confirmatory intent. **The mechanism is built now; no data is reserved
now.** As of 2026-08-07 every session is `discovery`.

Within-analysis holdout (cross-validation, train/test splits, held-out folds) is used wherever an
analysis supports it — and is already mandatory for decoders.

**Revisit trigger.** When the additional experiments currently in progress land and subject count
and quality support it, reconsider promoting a subject or a session block to `confirmatory`. That
decision is recorded as a superseding ADR, not made silently.

**Why deferred rather than adopted.** Reserving a whole subject or an arbitrary block of sessions is
too expensive at this timepoint. There are realistically two to three clean DMS subjects; locking
one away costs more statistical power than the confirmatory guarantee currently buys, and the data
are about to improve. This is a power judgement about a specific dataset at a specific moment, not a
rejection of the principle.

**Why build the mechanism anyway.** It is cheap now and impossible to retrofit honestly later — once
data has been looked at, a "held-out" set is held out in name only. Building the field and the
loader guard now means activation later is a one-line change plus an ADR, rather than a redesign.

⚠️ **What this leaves open, stated plainly.** Within-analysis cross-validation protects against
*overfitting inside one analysis*. It does **not** protect against the garden of forking paths
*across* analyses, because an analysis can be re-run with different choices and re-cross-validated
each time. That risk is real and remains open. It is mitigated — not eliminated — by ADR-013's
ledger (which makes the number of analyses run visible and honest) and by ADR-010's pre-specification
requirement (which timestamps the prediction before the result). Anyone reading a result from this
project should know which of those protections applied to it, and the ledger is where they will find
out.

### ADR-013 — A results ledger

> ⚠️ **Extended 2026-08-07**: required fields added by **ADR-018** (QC profile, stability control,
> cell-type provenance), **ADR-019** (prespec_commit, experimental_unit, figure_panels) and
> **ADR-021** (per_subject, scope, `inconclusive` status, enumerated Verification, bounded
> backfill).

**Decision.** One versioned, in-repo ledger holds every claim the project makes. Each row records:

| Field | Purpose |
|---|---|
| Question ID | Links to the spec that pre-specified it |
| Claim | One sentence, in plain language |
| Analysis | The component and commit that produced it |
| **Effect size + CI** | Required. A p-value alone is not a result |
| Verification | Which battery was applied (FR-normalisation, circularity, pseudoreplication, per-region breakdown, leakage controls) |
| Cohort | `discovery` / `confirmatory` (see ADR-012) |
| **Status** | `exploratory` / `confirmed` / `retracted` / `refuted` |
| Superseded by | Where a later result overturned it |

**Alternatives.** (a) Status quo — results in `docs/science/`, retractions in memory notes.
(b) A lab notebook. (c) Nothing; rely on the papers.

**Why.** (a) is what exists, and it has already drifted: the audit must check whether each
`docs/science/*-results.md` still agrees with the memory note for the same question, precisely
because retractions were recorded in one place and results in another. A reader of the repo cannot
currently tell which claims still stand.

The status field is the point. A retraction is not an embarrassment to be buried in a note — it is a
result, and a project that records them visibly is more trustworthy than one that appears never to
have been wrong.

Requiring effect size and CI in the same row as the claim also does real work at this project's n:
with thousands of units, p < 0.001 is available for effects far too small to matter.

Finally, the ledger makes **analysis multiplicity visible**. Knowing that a headline came from the
third of forty tests on the same dataset is essential to interpreting it, and today that number is
unknowable.

**Commits us to.** Writing a ledger row for every claim, including the ones that did not work out.

### ADR-014 — Null controls and synthetic recovery are gates, not habits

> ⚠️ **Scope refined 2026-08-07 (ADR-021)**: nulls apply to ledger-entering inferential outputs
> (descriptive artefacts declare themselves); recovery is narrowed to estimators with free
> parameters/latent structure, once per family, and gains a model-recovery requirement.

**Decision.** Two requirements, enforced by the sub-project 2 gates:

1. **Every analysis ships a null control** — label shuffle, circular shift, or the appropriate
   surrogate for its design — and its recorded output must be flat. Where the null is cheap it runs
   in CI; where it is expensive, the gate requires that the null exists, that its stored result is
   current with respect to the analysis code, and that it is flat.
2. **Every new estimator demonstrates parameter recovery on synthetic data with known ground truth
   before it is applied to real data.** `visdetect/utils/synthetic.py` becomes load-bearing rather
   than incidental.

**Alternatives.** (a) The current arrangement: a strong written rule in project memory
(`feedback_circular_analysis_null_controls`) and reviewer diligence. (b) Null controls only.
(c) Post-hoc verification once a result looks interesting.

**Why.** (a) is a good rule that is already written down — and the TF-pulse PETH circularity bug
still shipped, producing a sign-alignment artifact that survived until someone thought to check. A
rule that depends on remembering to apply it fails exactly when the result is exciting, which is
when it matters. This is the same argument as ADR-003, applied to statistics rather than syntax.

(c) inverts the incentive: verification arrives after you are attached to the answer.

Synthetic recovery earns its place separately. A null control tells you an effect is not noise; it
does **not** tell you the estimator measures the quantity you think it measures. Recovery on
ground-truth data is the only cheap way to establish that, and it would have caught the circularity
bug at the moment the estimator was written rather than after it produced a finding.

**Commits us to.** Writing a null per analysis and maintaining synthetic generators — the largest
ongoing cost of any ADR here, and the one most directly aimed at the project's stated goal of
findings that can be trusted.

---

### Preamble to ADR-015 … ADR-022 — the panel round

Adopted 2026-08-07, after a six-lens expert review of ADR-001…014 (data standards, statistical
rigor, reproducibility engineering, ephys QC, publication readiness, adversarial critic) and the
project owner's answers to its eight questions. Full evidence and rationale:
`2026-08-07-master-design-panel-review.md` (+ raw panel output alongside). These records carry the
decisions; the review document carries the arguments. Where an earlier ADR is extended or partly
superseded, it bears a banner pointing here.

### ADR-015 — The data layer: a self-describing, schema-versioned canonical store

**Decision.** The design finally specifies its data layer (it silently inherited pickle/CSV/`.npy`):

- **Session store: NWB (HDF5) per session, written via NeuroConv, behind a `SessionStore`
  boundary** (`load_session(SessionKey) → Session`) so the backend is swappable and the analysis
  layer never touches the format. Final confirmation comes from a **one-day measured storage spike
  in the audit** (3 real sessions incl. a BG_012 colliding twin: size, read latency on the three
  real access patterns, round-trip equality → `measurements.csv`). Declared fallback if the
  per-frame stimulus log proves awkward: NWB for units/trials/events + a referenced Parquet
  sidecar, same boundary.
- **Derived tables are Parquet; CSV is export-only** (no code re-reads a CSV under the cache root —
  gated). This kills the leading-zero session-id bug at the format level. Derived arrays carry
  named dimensions, bin centres and unit ids (no bare `.npy`).
- **Every artefact carries `schema_version`**; the loader refuses unknown versions; schema changes
  ship with explicit, tested migrations and a decision-log entry. The "`None` = both 'absent' and
  'unverified'" pattern is banned.
- **Identity is stamped inside the artefact** (opaque `session_uid` + subject + schema version) at
  ingest; the registry indexes stamped ids; a path-vs-stamp disagreement is a hard error. Twins
  (`_b`/`_c`/`_v2`) become data facts, not naming conventions.
- **One canonical per-session Units table**: spike times plus every per-unit attribute (anatomy,
  waveform, tracking ids, optotag verdict, cell-type label) as columns with `*_version`/`*_source`
  attributes. A typed `UnitID` (and `ChronicUnitID` carrying tracker + version) joins ADR-004's
  API; bare `(session, cluster)` tuples are banned from public signatures. The
  `(session_id, cluster_id)` string-join architecture — the surface the 15,802-row defect ran on —
  is retired.
- **The MATLAB NI-extraction is dispositioned**: preferred, read the raw SpikeGLX nidq channels
  directly in Python (permanently retiring the lick-defect class); otherwise the `.mat` is declared
  an external upstream artefact with recorded producer, version and content hash.
- Old pkls remain readable at the ingest boundary until migrated; after digest-verified round-trip
  they are deleted (sub-project 6) — running both stores indefinitely on one disk is the expensive
  default, not the safe one.

**Recorded rejections** (so they are not relitigated): DataJoint / DataJoint Elements / Spyglass
(NWB+MySQL+computed-table rewrite of the very layer ADR-001 ports — pure overhead for one person on
Windows; the two good ideas, NWB store and dependency-driven invalidation, are adopted separately);
NWB-Zarr (immature for this shape); a bespoke parallel schema document (one schema, owned upstream).

**Commits us to.** NeuroConv/pynwb as load-bearing dependencies; the migration of 285 sessions;
DANDI-eligibility as a by-product (deposit scope per the owner: processed NWB, not raw).

### ADR-016 — Environments are pinned and randomness is provenance

**Decision.**
- **Lockfiles, not version lists**: `uv.lock` for the analysis stack (one universal file covering
  `win_amd64` + `linux_x86_64`), `pixi.lock` for the heavy KS4/CUDA/UnitMatch layer. Python floor
  ≥3.12 (3.10 EOLs Oct 2026), confirmed by a trial lock (the `pyddm==0.9.0` pin is the likely
  blocker and is checked first). S5's "environment" means *lockfile hash + platform tag*, plus an
  **external-tools table** (Kilosort4 commit, UnitMatch/DANT/Bombcell versions, TPrime build,
  SpikeGLX version, MATLAB release) — the tools that actually determine spike times.
- **RNG policy**: an AST gate bans `np.random.seed` and bare `np.random.*`; every stochastic
  function takes `rng: np.random.Generator`; entry points draw fresh 128-bit entropy per run,
  **recorded in the provenance sidecar**; `rng.spawn()` for workers. A seed *registry* is
  explicitly rejected — registering magic numbers recreates the shadow-constant pathology and
  correlates "independent" nulls; what is registered is entropy per run, in the log.
- **Thread policy**: workers run under `threadpool_limits(1)`; the sidecar records BLAS
  name/version/threads.
- **`numerical-noise` becomes the fifth ADR-009 attribution**, with a measured per-analysis-class
  tolerance floor (≥5 repeat runs across platforms/thread settings; the observed spread is the
  floor). Bit-identity is abandoned as a target — unattainable under DYNAMIC_ARCH BLAS and wrong to
  chase. This resolves §10's former Open Question 4.
- Cheap coherence gates: `uv lock --check` in CI; a ledger-integrity check that every row's
  recorded lockfile hash resolves to a blob reachable from its commit.
- Containers (Apptainer) are **deferred with recorded reasons**; the lockfile is their prerequisite
  anyway.

### ADR-017 — Content-addressed inputs and caches; the time base is typed

**Decision.**
- **Registry rows carry content digests** (sha256 + size + mtime) of every upstream input and every
  produced artefact. Registry refresh diffs digests and emits a `changed-inputs` report that must
  be acknowledged in the decision log before downstream artefacts are considered valid. (This is
  the mechanism that would have caught the lick-channel re-extraction: content changed at unchanged
  paths, invisible to any path-based snapshot.) A committed manifest-of-hashes (`registry/
  session-manifest.tsv`, with a `regenerable` column) is the data-versioning mechanism; DVC /
  DataLad / git-annex are rejected at this scale with recorded reasons (revisit at publication —
  then DataLad).
- **Caches are content-addressed**: artefact key = hash(analysis id ‖ code version ‖ resolved
  constants ‖ input digests ‖ params ‖ lockfile hash). The loader refuses a key mismatch;
  `allow_stale=` is explicit and ledger-recorded. Staleness becomes an identity check, not an mtime
  heuristic; the `tf_responsive`-registries defect class becomes unrepresentable. ADR-014's
  null-currency check collapses into the same key comparison — one mechanism, not two.
- **Time-base provenance is mandatory.** The session artefact carries a `time_base` block (which
  spike-times file, TPrime build, reference stream, sync residual statistics). Ingest **fails
  closed** instead of silently falling back to uncorrected times; `time_base="uncorrected"`
  requires an explicit flag, and the loader refuses such a session unless the calling analysis
  declares that intent — the same guard pattern as ADR-012's cohort field.

### ADR-018 — QC is non-destructive, named, and versioned; groups are strata, not verdicts

**Decision.**
- **Ingest stops applying QC destructively.** The store keeps **all Kilosort-good units** plus a
  per-unit metric panel; every QC criterion is applied at analysis time as a **view**. (Today's
  pkls store spikes only for `good_and_stable` units, making every future analysis a subset of one
  unnamed 2025 decision — which quietly falsifies ADR-011 for the most consequential decision in
  the pipeline.)
- **The metric panel is computed once at ingest** using field-standard definitions
  (SpikeInterface / Bombcell, stored as the tools' own output tables with versions — no wrapper
  API), with the Khilkevich sliding-window stability statistic retained as a named metric.
- **Named, hashed, versioned QC profiles** (~4: `sorting_quality`, `striatal_default`,
  `striatal_strict`, `tracking_eligible`), each split into a `quality` block and an `eligibility`
  block. The four disagreeing firing-rate floors were different *questions*, not four copies of one
  number — the defect was namelessness, and the enforceable invariant is *named and recorded*, not
  *unique*. Inline thresholds are gated; sidecar + ledger record profile id + hash; profile changes
  are decision-log entries; promotion to `confirmed` requires a second-profile rerun.
- **Metrics are tri-state** (pass / fail / unknown) and **unknown fails closed** — `fillna(0)`
  passing a contamination gate is the `load_qc_profile() → {}` failure class again.
- **Registered named variants** for constants (e.g. `CHANGE_SIZE_POOLS['tracking_qc_v1']`): the
  canonical module owns every value; a deliberate divergence carries a name, an owner and a reason.
  A gate that cannot express a correct exception gets routed around.
- **Session-level covariates live in the registry**: days-from-implant, yield, AP RMS, median
  amplitude, narrow-waveform fraction, drift estimate, channel-map hash — plus the behavioural
  state columns (occupancies, session group manual+predicted, coverage flags). Both time axes are
  carried: `days_from_implant` (technical drift) and the training-session index (the science
  variable) — collinear within subject, partially decorrelated across subjects.
- **Strata, not verdicts** (project owner, 2026-08-07): only data-integrity QC (sync, clock drift,
  minimum trials) may remove a session globally. Group labels, state tags and eligibility rules are
  *selectors*: every session keeps all its labels; each analysis declares its stratum; the
  declaration is recorded in the sidecar and the ledger's scope field. A Disengaged-dominated
  session is excluded from a Balanced-vs-Balanced contrast and is exactly the inclusion set for
  disengaged-across-learning. The same principle governs units (a stricter profile hides, never
  deletes) and time base (uncorrected is refused-unless-declared, not erased).
- **Chronic-stability control is a contract row** for every across-session claim: a declared
  control from a named menu (days-from-implant covariate / composition matching / tracked-subset
  replication / within-window comparison), with a ledger field where `none` is legal but visible.
  Measured motivation: broad/SPN 89→15 % at the KS4 detection level with amplitude halving, while
  the behavioural gate excluded 5 of 6 SPN-rich June sessions — learning stage and recording epoch
  are collinear by construction.
- **"Tracked unit" becomes a registry table** keyed (subject, track_id, tracker, tracker_version,
  params_hash) with per-link scores, ISI-fingerprint checks, consensus flag, and a named
  per-subject-calibrated track-QC profile (within-day split-half sorts; across-shank negatives if
  multi-shank). Hand verdicts move to the decision log.
- **Cell-type label provenance is a required ledger field**: `celltype_label_source`
  ({optotag_collision_confirmed, optotag_candidate, waveform_gmm, multimodal_classifier,
  unlabelled}) + confidence. A claim naming D1/D2 cites collision-confirmed units or says
  "putative" with the source named (3 collision-confirmed units exist, all D1, zero D2).
- Five ephys entries join the known-defect register with directions (per the panel review),
  including BG_031's Laser-event extraction gap (35/43 sessions) — a data-completeness defect that
  looks like a biological result.

### ADR-019 — The publication layer

**Decision.**
- **A realized `n_table` is a mandatory sidecar block** (n subjects/sessions/units/trials, per
  condition, per panel, plus the declared clustering variable) — gate-enforced like provenance
  itself. Journals demand exact n per group per panel; adding this later means re-running
  everything.
- **The registry gains `subjects` and `acquisition` tables**: species, strain, genotype, **sex**
  (the cohort is mixed-sex — recorded per subject, reported as n-per-sex; sex is not modellable as
  a covariate at k ≤ 5), DOB, project-licence (PPL) number, surgery date, implant coordinates,
  hemisphere; per-session probe serial, IMRO map, rig, sorter + params, training stage.
- **Manuscript panels are typed artefacts** (`F3b`, `ED2a`) bound by a manifest to component +
  commit + registry snapshot + environment lock + n_table + supporting ledger rows; one-command
  `repro <panel>`; a CI smoke job regenerates two panels so the path cannot rot. **No figure
  produced by the old repo may enter the manuscript** — old figures are archive.
- **Every figure emits `source_data.csv`** (the plotted values, tidy) — near-free at write time;
  satisfies source-data requests; lets ADR-009 diff numbers instead of pixels.
- **Sub-project 5 produces a human layer too**: README/docs generated from the same source as
  CLAUDE.md, plus three hand-written pages (30-minute walkthrough; task/vocabulary glossary incl.
  the `fa` ≠ SDT-false-alarm trap; data-provenance map). The ADR-007 packet corpus is kept
  browsable — it *is* the onboarding manual and the Methods first draft.
- **Release hygiene**: CITATION.cff (+ ORCID); an explicit **data** licence decision at deposit
  time (MIT covers code only; DANDI needs CC0/CC-BY); `paper-freeze/<name>` tags with Zenodo DOIs
  at submission and each revision; generated ARRIVE-E10 + Reporting Summary drafts from registry +
  decision log + ledger (blank fields become explicit N/A disclosures; the blinded session sorter —
  including its built-in repeat-based intra-rater reliability check — is reported as the genuine
  blinding procedure it is).
- **The NWB export contract is tested from day one**: CI writes one synthetic session through the
  artefact writer and runs `nwbinspector --config dandi`, asserting zero CRITICAL findings.
- **Ledger columns added**: `prespec_commit`, `experimental_unit` (mouse/session/unit),
  `figure_panels` (a retraction immediately names the panels it invalidates).

### ADR-020 — Process reality: tiers, skeleton, budget, sunset

**Decision.**
- **Two-tier analysis contract.** Tier 1 (*explore*) costs only what the scaffold gives free:
  registry session sets, canonical constants, artefact API, layer declaration, index entry.
  Tier 2 (*claim*) fires **at promotion** — the moment output is cited in a ledger row, figure or
  results doc — and adds the null, recovery, pseudoreplication method, reuse statement, test and
  packet. Enforced by `ledger add` refusing a row whose Tier-2 artefacts are missing or stale.
  Scratch/notebook paths are exempt from *gates*, never from *provenance*: their artefacts are
  stamped `provenance_tier: scratch` and are mechanically ineligible for ledger rows or panels.
  (Measured basis: full-contract overhead ≈ 8–25 h vs 3–6 h to write an exploratory analysis;
  ~80 % of exploratory analyses die without a claim; and the gate now fires exactly at ADR-014's
  "moment of excitement".)
- **Milestone 0.5 — walking skeleton** (time-boxed 3–5 days, before the Stage-2 spec session): one
  real day-1–9 session plus one BG_012 colliding twin, end-to-end through registry → typed key →
  constants → load → PSTH → figure with sidecar → ledger row → one gate. Specs 1–6 must cite its
  measured ergonomics. If typed-ID or sidecar ergonomics fail here, amending ADR-004/S5 early is
  the mechanism working, not failing.
- **Gate tiers, stated per gate**: (1) pre-commit — source-only, seconds; (2) CI — source-only,
  minutes, GitHub-hosted win+linux, running against a **synthetic golden mini-session** (the repo
  is public, so no real unpublished data is committed); (3) verification runs — data-dependent,
  local/Slurm, pre-milestone or pre-ledger, advisory-with-report. Full-data runs produce a
  **commit-pinned receipt** (`.ci/receipts/fulldata-<sha>.json`) without which the merge fails.
  **No self-hosted runner on the data box.**
- **Packets re-scoped**: bundled by module group / gate family (~15–25 total); four of six sections
  machine-generated (provenance, blast radius, executed output or CI-receipt reference, ADR-009
  delta table); the human writes "what it is" and the decision. A **seventh, adversarial section**
  — "strongest objection found, and its resolution", produced by an independent pass — so the
  reviewer arbitrates a disagreement rather than assenting to a proposal.
- **The near-duplicate AST check is demoted** to a monthly/pre-milestone triage report (top-N
  candidate pairs, each triaged merge/justify/ignore-with-reason). The must-import registry and the
  same-name-different-behaviour ban stay hard — they are the zero-false-positive checks, and the
  latter is the genuinely corrupting class.
- **One sanctioned, logged override**: `# gate-override: <rule-id> reason="…"`, honoured by the
  gate, appended to a git-tracked log, counted and reviewed per milestone; `--no-verify` denied at
  the harness level. Every gate failure names the rule, the ADR, the historical defect it prevents,
  and the override syntax.
- **Time-box and stop-loss**: a stated budget per sub-project, a named stop-loss date, and the
  pre-agreed fallback — collapse to walking-skeleton scope (registry + constants + provenance + one
  gate family) with everything else ported lazily. Live science continues in the old repo until
  freeze; the freeze is preceded by a **per-branch disposition table** (sub-project 0 deliverable:
  merge / port / abandon-with-reason for every branch and untracked file, the QC1 repair decided
  explicitly); no new old-repo branches after the freeze date.
- **Port sunset = lazy porting** (owner's answer, 2026-08-07 — no figure list exists yet and
  forcing one would narrow an exploratory project): every old module starts on `cold-list.md` and
  is ported **on first use** through the ADR-009 gate. "Done" = foundation + gates complete; the
  cold-list may be nonempty forever; >~12 modules pulled across before the foundation stabilises is
  an alarm, not a plan.
- **Dependency fix**: the minimal artefact/provenance API moves into sub-project 1 (sub-project 2's
  gates need it; only the plot system and palettes remain in 4).
- **ADR-012's dormant loader-refusal guard is replaced by a session-access log** (every run records
  which sessions it touched, alongside the sidecar). The `cohort` field stays. The access log is
  what makes a future confirmatory split defensible — a candidate held-out set provably appears in
  zero prior analyses.
- **Old FIGURES/cache disposition**: the old tree becomes a read-only archive root referenced by
  the configured data root; tracked deliverables migrate; untracked figures are deleted only after
  submission.
- **ADR-005 refinement**: generated content lives in `CLAUDE.generated.md` (separate from prose);
  the CI failure names the one-command fix; the *prose* half gets the dead-path check (the old
  CLAUDE.md's 18 %-dead problem was prose, which a generator does not solve).

### ADR-021 — Statistical inference, specified (extends ADR-010 / 013 / 014)

**Decision.**
- **The hierarchy is named**: trial < unit < session < subject. Session-level random intercept is
  the default random effect; **subject is never a random effect at k ≤ 5** (fixed effect or
  stratification — a 3-level variance component at k=3 converges to ~0 and returns pooled inference
  wearing a rigorous hat). The mixedlm and a cluster-robust OLS are fitted as a pair with the
  convergence flag recorded; the hierarchical bootstrap is the sanctioned nonparametric
  alternative. Bootstrap resampling units are declared; bootstrapping over subjects at k ≤ 5 is
  banned.
- **Ledger requirements**: `per_subject` estimate array (sex-annotated), `n_subjects_replicating`,
  and `scope_of_inference` from a controlled vocabulary; a claim's wording must be derivable from
  its scope. **A claim that reverses sign in any contributing subject is capped at `exploratory`
  with a required `needs_more_data` note** (owner's rule, 2026-08-07).
- **Status vocabulary gains `inconclusive`**; any claimed null carries a TOST/equivalence test
  against the pre-registered minimum interesting effect, with the bound and CI stored. A negative
  that excludes effects larger than the MIE is a null; one that does not is inconclusive and is
  worded as underpowered.
- **Recovery is split**: (a) parameter recovery for estimators with free parameters or latent
  structure — simulated-vs-recovered scatter per parameter, at realistic trial counts and stimulus
  statistics, once per estimator *family*; standard library statistics are excluded; the
  kernel-width estimator additionally sweeps ground-truth firing rate and must be FR-invariant;
  (b) **model recovery** (confusion matrix across candidate models) wherever a claim rests on model
  comparison — drift-vs-threshold, HMM state count, retained dimensionality.
- **The MIE defaults to the estimator's measured resolution floor** from the recovery run (smallest
  effect recovered with 80 % sign-consistency and CI excluding zero); a smaller MIE requires a
  written override.
- **Nulls are specified and scoped**: required only for analyses producing an inferential statistic
  that enters the ledger (descriptive artefacts declare `descriptive`, recorded and ineligible as
  claim evidence). The surrogate must respect the claim's dependency level; flatness is
  quantitative; no permutation p below 1/(n_perm+1). CI runs synthetic-null smoke tests only; real
  nulls live as keyed artefacts (ADR-017).
- **A bounded specification curve is the promotion price**: `confirmed` claims only. Each analysis
  pre-declares its 3–6 analytic degrees of freedom as keyword arguments defaulting to registry
  constants (the inline-literal gate already forces this shape); at promotion the declared factorial
  runs with joint inference under a permutation null; refit-heavy axes get a coarse 3-point sweep.
  The session-ordering axes are the project's worked example, with the canonical axis fixed by S1's
  three-role decomposition (§4a of the panel review).
- **`prespec_sha` with a mechanical ancestry check**: the pre-specification commit must precede the
  analysis commit; failing or absent → capped at `exploratory`.
- **A `stability` field**: the effect re-estimated on interleaved halves (never chronological —
  chronological splits are confounded with the learning axis).
- **`visdetect.stats.effects`** joins the must-import registry: one canonical effect size per test
  class, computed at the inference level; ΔAIC/ΔBIC banned as effect sizes.
- **`visdetect.verify` lands in sub-project 2, not 5**: the harden-result battery reimplemented as
  library functions returning structured records; the ledger's Verification field is generated from
  those records as enumerated booleans; the prose skill becomes a thin pointer.
- **A sanctioned constants override** (`with constants.override(min_fr=0.5): …`) importable only
  from the sweep harness and tests, with active overrides written into the sidecar — resolving the
  otherwise head-on collision between S1 (one definition) and sensitivity analysis (many values).
- **Ledger backfill is bounded**: forward-only, plus claims in the paper outline and every
  retraction/refutation (~4 rows — the most valuable in the ledger).
- **A design-sensitivity module** resamples the existing subjects to report detectable effect size
  as a function of subject count — the quantitative basis for ADR-012's revisit trigger and for
  "how many more mice".

### ADR-022 — Backup is a policy, not an event (supersedes ADR-008's off-disk clause)

**Decision.** ADR-008's "off-disk safety is the remote's job" is true for code and false for
everything else — the git remote holds none of the pkls, caches, registry or hand-made files.
Policy: (i) the irreplaceable hand-made files enter git via `git add -f` **now** (217 hand-drawn
state episodes; the 132 blinded session sorts; the sorting rules + fitted rule; pupil sidecars
committed on their own branch); (ii) registry + decision log + hash manifest sync daily to
institutional storage via `rclone --checksum` (an I/O-only, off-hours carve-out from the
no-compute-over-Samba rule, recorded as such); (iii) derived caches and figures sync weekly;
(iv) a **quarterly restore test** pulls a random artefact and verifies it against the manifest.
New success criterion S9: the restore test passes.

---

### Success-criteria addendum (2026-08-07)

- **S5 (amended)**: the sidecar additionally carries the RNG record (ADR-016), the external-tools
  table (ADR-016), and the realized `n_table` (ADR-019); scratch-tier artefacts are stamped and
  ledger-ineligible (ADR-020).
- **S9 (new)**: the quarterly restore test passes — every non-git artefact class has a verified
  off-disk copy (ADR-022).

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
| **0** | Deep empirical audit | Findings corpus: definition inventory, duplication map, layering graph, dead-code census, artefact provenance survey, memory-claim verification, known-defect register, real-data measurements, storage-format spike, **per-branch disposition table** | −1 |
| **0.5** | **Walking skeleton** (ADR-020) | One real day-1–9 session + one BG_012 twin end-to-end: registry → typed key → constants → load → PSTH → provenanced figure → ledger row → one gate. Time-boxed 3–5 days; specs 1–6 cite its measured ergonomics | 0 |
| **1** | Foundation | Registry + typed identity, constants, config, session model (`SessionStore`, ADR-015), IO, QC metric panel, **minimal artefact/provenance API** (moved from 4 per ADR-020) | 0.5 |
| **2** | Enforcement | AST gates, import contracts, pre-commit, CI tiers + synthetic mini-session, `visdetect.verify` (ADR-021) | 1 |
| **3** | Analysis layer | **A standing rule, not a phase** (ADR-020): modules cold-listed, ported on first use behind the explained-difference gate | 1, 2 |
| **4** | Figures & artefacts | Plot system, palettes, panel manifest + `repro` (ADR-019) | 1, 2 |
| **5** | AI layer **+ human layer** | Generated `CLAUDE.generated.md`, skills, hooks, curated memory; README/docs + walkthrough/glossary/provenance pages (ADR-019) | 1–4 |
| **6** | Migration & decommission | Cutover, freeze of the old repo (per disposition table), artefact preservation, pkl deletion after digest-verified round-trip, cross-spec consistency pass | all |

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
| Gates too strict, exploratory work becomes painful | Two-tier contract (ADR-020): full gates fire at promotion-to-claim, not file creation; scratch is exempt from gates but never from provenance, and is ledger-ineligible |
| Rebuild consumes months and the paper goes unwritten | Time-box + named stop-loss + pre-agreed fallback to walking-skeleton scope with lazy porting (ADR-020); science continues in the old repo until a dispositioned freeze |

---

## 10. Open questions

Deferred to the specs that can answer them from evidence.

- Which subjects the new repo must support at v1, and whether BG_012's protocol variants are in
  scope (currently parked).
- Whether the old repo's git history is imported, referenced, or dropped.
- Where the registry lives physically, and how it is refreshed when new sessions arrive.
- What tolerance the equivalence gate uses per analysis class (exact, floating-point, statistical).
- Whether HPC/cluster job submission is in scope for v1 or deferred.
