# Literature Reading Plan — PHASE 3 ADDENDUM (Adjacent Literature)

> **For agentic workers:** REQUIRED SUB-SKILL: Use **superpowers:executing-plans** to implement this addendum task-by-task (sequential, checkpointed, in the current session). Steps use checkbox (`- [ ]`) syntax.
>
> **This is an addendum** to `2026-05-28-literature-reading-plan.md` (Phase 1 + Phase 2, Tasks 0–8, COMPLETE as of 2026-06-04). It reuses that plan's **Recipe R1–R5 per paper**, **V1/V2 end-of-batch steps**, **per-paper memory schema**, and **methods-ref schema** (Task 8). Read the parent plan + the design spec `2026-05-28-literature-reading-plan-design.md` before executing. The compact recipe + schemas are restated below so this file is self-contained.
>
> **TDD mismatch is expected** (same as parent): there are no code tests. Verification = "the memory file exists with the expected schema and content." V1 confirms artifacts landed.

**Goal:** Extend the `memory/literature/` scaffold with *adjacent* literature — striatal cell types / microcircuit, direct/indirect pathway function, basal-ganglia architecture + dopamine, behavioral-state/ethology, and a curated Dayan theoretical core — so future Claude sessions have working knowledge of the circuit + theory context surrounding BG_046's decision-making focus.

**Architecture:** Phased reading, identical shape to Phases 1–2. Each task = one batch (5–7 papers). Each paper → one `paper-*.md`. Each batch ends with `synthesis-batch*.md` + one `MEMORY.md ## Literature` line. Textbook / anatomy *chapters* use the **methods-ref schema** (like the NDS batch) and are read **on demand**, not in bulk.

**Tech Stack:** Read (PDFs, page ranges for >20pp), Write (markdown memory files), Edit (MEMORY.md), Glob (path verification). Memory dir: `C:\Users\Ben\.claude\projects\e--python-analysis-git-repos-vis-detect-analysis-Sep2025\memory\`.

**Source PDFs:** `G:\Postdoc_research\Mendeley_articles\` (+ `striatum\`, `pDayan\`, and `The mouse nervous system  - Watson, Paxinos, Puelles\` subfolders). **Search by TITLE, not author.** If an exact path misses, do a broad title-substring Glob before declaring a PDF absent. Watch for odd filename characters (en-dashes `–`, the `et et al.` typo, double spaces in the Mouse-Nervous-System folder name).

---

## Scope decisions (locked 2026-06-04, user-approved)

- **Included:** the 5 curated batches below (~32 papers). User picked **P3-1 first**.
- **pDayan theory depth:** **curated 6** (P3-5) + Dayan & Abbott *Theoretical Neuroscience* chapters **on demand** only.
- **"The Mouse Nervous System" (Watson/Paxinos/Puelles, 2012):** PRESENT on disk (~40 chapter PDFs). Treat as **on-demand anatomy reference** (methods-ref schema), NOT a batch. Relevant chapters: Ch7 Subpallial Structures (striatum), Ch25 Visual System, Ch19 Motor Cortex, Ch30 Prefrontal Cortex, Ch2 Gene-Targeting / Ch3 Genetic Neuroanatomy (D1/D2 Cre lines). Pull one only when an analysis raises a specific anatomy question.
- **Bogacz 2006 "physics of optimal decision making":** user has added a PDF to the Mendeley folder, but as of 2026-06-04 it is **not yet visible to Glob** (searched `physic*`, `*bogacz*`, `*optimal*`, `*two-alternative*`, `*decision making*`, `*models of performance*`; G: may be a synced drive with propagation lag). It belongs to **P3-5 (last batch)**, so it does not block P3-1. **Reconfirm the exact filename with the user before executing P3-5.** When read, it fills the dangling `[[paper-bogacz-2006-decision-review]]` target (see parent memory carry-forward #3) — keep filename `paper-bogacz-2006-decision-review.md` if first-author/year verify.

### Explicitly OUT of scope (do NOT read)
- `NAc\` folder (2 PDFs — κ-opioid / NPAS4 addiction).
- `striatum\` addiction / cocaine / dyskinesia / OCD cluster (~10 PDFs).
- Tangential pDayan theory: Helmholtz Machine, Q-learning (Watkins/Dayan original), Bayesian RL overview, Efficient Bayes-Adaptive RL, Doubly-distributional population codes, "How to set the switches on this thing", "Space and time in visual context", Structure in the space of value functions. (Re-open later only if a specific need arises.)

---

## Compact Recipe (R1–R5) — apply to EVERY paper

- **R1. Verify path** — Glob the exact path. If missing, broad title-substring Glob; if truly absent, report to user, do not silently skip.
- **R2. Read** — ≤20pp: whole. >20pp: page ranges, priority abstract+intro → discussion+figures → skim methods → selective results. ~10–15pp effective coverage.
- **R3. Filename** — `paper-<firstauthor-lastname>-<year>-<short-topic>.md` (ASCII lowercase, hyphens). Verify author/year against the PDF; rename starter if wrong.
- **R4. Write memory file** — schema below, ~150-word body.
- **R5. Verify** — Glob `memory/literature/paper-*.md` to confirm the new file appears.

### Per-paper schema (R4)
```markdown
---
name: paper-<firstauthor-lastname>-<year>-<short-topic>
description: <one-line relevance hook for future-me>
metadata:
  type: reference
---

**Citation:** Author et al., Journal Year. doi/preprint id.
**Question:** <1 sentence>
**Paradigm/methods:** <2-4 sentences — species, task, modality, key analyses>
**Findings (3-5 bullets):**
- <concrete claim with numbers/effect sizes where possible>
**My synthesis:** <2-4 sentences — relevance to BG_046, what it connects to, weak points>
**Links:** [[other-paper-name]], [[scientific_context]], etc.
```

### Methods-ref schema (for textbook / anatomy CHAPTERS only — P3-5 on-demand)
```markdown
---
name: methods-<source>-<chapter>-<topic>
description: <one-line — what this chapter is useful for>
metadata:
  type: reference
---

**Source:** <Book> (Year), Chapter NN — <title>.
**Topic:** <1 sentence>
**Canonical content (3-5 bullets):** key facts/techniques + typical use
**When to use in BG_046 context:** <2-3 sentences — which analyses/aims this informs>
**Links:** [[...]]
```

### End-of-batch (once per task)
- **V1. Verify artifacts** — Glob `paper-*.md` (total ≥ running sum), Glob the batch `synthesis-batch*-*.md` (expect 1), Read MEMORY.md to confirm the new `## Literature` line.
- **V2. Report to user** — single message: papers read (count + one-liners), top 3 surprises/connections to BG_046, suggested adjustments to later batches, any missing/unreadable PDFs (exact path), memory entries by filename. **Then STOP** — user decides when to start the next batch.

---

## Naming for Phase 3 synthesis files
- `synthesis-phase3-celltypes.md`, `synthesis-phase3-pathways.md`, `synthesis-phase3-bg-architecture.md`, `synthesis-phase3-behavioral-state.md`, `synthesis-phase3-theory.md`.
- (Phase 3 uses descriptive theme slugs rather than `batchNN` to avoid colliding with the Phase-1 batch numbering.)

---

## Task P3-1: Striatal cell types & microcircuit (7 papers) — EXECUTE FIRST

**Why BG_046:** feeds the active M2 waveform / cell-type workstream (D1/D2 SPN + FSI labels; `waveform_celltype_labels.csv`). Establishes ground truth on what physiologically/transcriptomically separates SPN subtypes and interneuron classes, so waveform/firing-based labels can be sanity-checked against biology.

**Files:** 7 × `paper-*.md` + `synthesis-phase3-celltypes.md` in `memory/literature/`; modify `MEMORY.md`.

- [ ] **Steps 1–7: Process each paper using R1–R5**

| # | Path (under `G:\Postdoc_research\Mendeley_articles\`) | Starter filename |
|---|---|---|
| 1 | `striatum\Diversity of Interneurons in the Dorsal Striatum Revealed by Single-Cell RNA Sequencing and PatchSeq.pdf` | `paper-munoz-castaneda-2021-striatal-interneuron-diversity.md` (verify author/year) |
| 2 | `striatum\Principles of Synaptic Organization of GABAergic Interneurons in the Striatum.pdf` | `paper-gabaergic-interneuron-synaptic-organization.md` (verify) |
| 3 | `striatum\Cholinergic control of striatal GABAergic microcircuits.pdf` | `paper-cholinergic-control-striatal-gaba-microcircuits.md` (verify) |
| 4 | `striatum\The microcircuits of striatum in silico.pdf` | `paper-striatum-microcircuits-in-silico.md` (verify) |
| 5 | `striatum\Differential Innervation of Direct- and Indirect-Pathway Striatal Projection Neurons.pdf` | `paper-differential-innervation-d1-d2-spn.md` (verify) |
| 6 | `striatum\Striatal projection neurons coexpressing dopamine D1 and D2 receptors modulate the motor function of D1- and D2-SPNs.pdf` | `paper-d1d2-coexpressing-spn-motor.md` (verify) |
| 7 | `striatum\DIVERSITY OF STRIATAL interneurons _ thesis by Anna Tokarska.pdf` (thesis — long, **skim** intro + summary chapters only) | `paper-tokarska-thesis-striatal-interneuron-diversity.md` |

- [ ] **Step 8: Write synthesis** → `synthesis-phase3-celltypes.md`. Sections: `## Convergent claims about SPN/interneuron identity` · `## What physiologically distinguishes FSI / cholinergic / SPN (for waveform labels)` · `## Disagreements / open questions` · `## Direct relevance to BG_046 cell typing` (link `[[p0_spine_audit_done_june2026]]`-era celltype work + waveform_celltype_labels) · `## Paper links`.
- [ ] **Step 9: Update MEMORY.md** (Edit, not Write) — append under `## Literature`:
  `- [Phase 3 — Striatal cell types & microcircuit](literature/synthesis-phase3-celltypes.md) — SPN/interneuron diversity (scRNA+PatchSeq), GABAergic + cholinergic microcircuit, D1/D2 innervation, for BG_046 D1/D2/FSI waveform labels (7 papers)`
- [ ] **Step 10: V1** — verify artifacts.
- [ ] **Step 11: V2** — report, then STOP.

---

## Task P3-2: Direct/indirect pathway function & push-pull (7 papers)

**Why BG_046:** the proposal's core mechanism is PPC/aMOs/pMOs → D1/D2 push-pull (see `[[proposal_aims]]`) and the Lohse AND-gate framework (`[[scientific_context]]`). This batch is the circuit-function backbone for interpreting D1 vs D2 dissociations.

**Files:** 7 × `paper-*.md` + `synthesis-phase3-pathways.md`; modify `MEMORY.md`.

- [ ] **Steps 1–7: Process each paper using R1–R5**

| # | Path | Starter filename |
|---|---|---|
| 1 ⭐ | `Lohse et et al., Frontal cortex gates striatal dynamics to enable flexible control of behaviour 071025.pdf` (note literal `et et al.` + date in filename; Lohse = project reference author) | `paper-lohse-2025-frontal-gates-striatal-dynamics.md` |
| 2 | `striatum\Activation of Direct and Indirect Pathway Medium Spiny Neurons Drives Distinct Brain-wide Responses.pdf` | `paper-d1-d2-activation-brainwide-responses.md` (verify) |
| 3 | `striatum\The respective activation and silencing of striatal direct and indirect pathway neurons support behavior encoding.pdf` | `paper-d1-d2-activation-silencing-behavior-encoding.md` (verify) |
| 4 | `striatum\From avoidance to new action the multifaceted role of the striatal indirect pathway.pdf` | `paper-indirect-pathway-avoidance-new-action.md` (verify) |
| 5 | `striatum\Action suppression reveals opponent parallel control via striatal circuits.pdf` | `paper-action-suppression-opponent-parallel-striatal.md` (verify) |
| 6 | `striatum\Corticostriatal Flow of Action Selection Bias.pdf` | `paper-corticostriatal-flow-action-selection-bias.md` (verify) |
| 7 | `striatum\Task-specific subnetworks extend from prefrontal cortex to striatum.pdf` | `paper-task-specific-subnetworks-pfc-striatum.md` (verify) |

- [ ] **Step 8: Synthesis** → `synthesis-phase3-pathways.md`. Sections: `## Are D1/D2 antagonistic, complementary, or co-active?` · `## Frontal→striatal gating (Lohse AND-gate)` · `## Methods (optogenetics, brain-wide imaging, ephys)` · `## Direct relevance to BG_046 (D1/D2 push-pull, evidence-axis, Aims)` · `## Paper links`. Cross-link `[[synthesis-batch02-striatum]]` and `[[paper-khilkevich-lohse-2024-brainwide]]`.
- [ ] **Step 9: MEMORY.md** — append:
  `- [Phase 3 — Direct/indirect pathway function & push-pull](literature/synthesis-phase3-pathways.md) — Lohse 2025 frontal-gates-striatum, D1/D2 activation/silencing brain-wide, indirect-pathway roles, action-selection bias (7 papers)`
- [ ] **Step 10: V1.**  **Step 11: V2 + STOP.**

---

## Task P3-3: BG architecture, loops & dopamine teaching signals (7 papers)

**Why BG_046:** situates medial striatum within the cortico-BG-thalamic loop and gives the dopamine prediction-error context for Naive→Expert learning.

**Files:** 7 × `paper-*.md` + `synthesis-phase3-bg-architecture.md`; modify `MEMORY.md`.

- [ ] **Steps 1–7: Process each paper using R1–R5**

| # | Path (all under `striatum\` unless noted) | Starter filename |
|---|---|---|
| 1 | `Functional Neuroanatomy of the Basal Ganglia.pdf` | `paper-functional-neuroanatomy-basal-ganglia.md` (verify; likely DeLong/Wichmann or Lanciego review) |
| 2 | `The mouse cortico–basal ganglia–thalamic network.pdf` (en-dash in filename) | `paper-mouse-cortico-bg-thalamic-network.md` (verify; likely Hunnicutt/Foster) |
| 3 | `Macro-architecture of basal ganglia loops with the cerebral cortex- use of rabies virus to reveal multisynaptic circuits.pdf` | `paper-bg-cortex-loop-macroarchitecture-rabies.md` (verify) |
| 4 | `THE BASAL GANGLIA-FOCUSED SELECTION AND INHIBITION OF COMPETING MOTOR PROGRAMS.pdf` | `paper-redgrave-focused-selection-inhibition.md` (verify author/year) |
| 5 | `What are the computations of the cerebellum, the basal ganglia and the cerebral cortex.pdf` | `paper-doya-computations-cerebellum-bg-cortex.md` (verify) |
| 6 | `A Neural Substrate of Prediction and Reward.pdf` | `paper-schultz-1997-prediction-reward.md` (verify author/year) |
| 7 | `Action prediction error- a value-free dopaminergic teaching signal that drives stable learning.pdf` | `paper-action-prediction-error-value-free-da.md` (verify) |

- [ ] **Step 8: Synthesis** → `synthesis-phase3-bg-architecture.md`. Sections: `## Canonical BG loop architecture` · `## Selection / gating theories` · `## Dopamine teaching signals (value vs value-free)` · `## Direct relevance to BG_046 (where medial striatum sits; learning across stages)` · `## Paper links`.
- [ ] **Step 9: MEMORY.md** — append:
  `- [Phase 3 — BG architecture, loops & dopamine](literature/synthesis-phase3-bg-architecture.md) — cortico-BG-thalamic loops, focused selection/inhibition, Doya computations, Schultz + value-free DA teaching signals (7 papers)`
- [ ] **Step 10: V1.**  **Step 11: V2 + STOP.**

---

## Task P3-4: Behavioral-state & ethology (5 papers)

**Why BG_046:** supports the HMM behavioral-state workstream (`[[analysis_frontiers]]`, `visdetect/analysis/hmm.py`) and the naturalistic-behavior framing for interpreting engagement/internal-state structure.

**Files:** 5 × `paper-*.md` + `synthesis-phase3-behavioral-state.md`; modify `MEMORY.md`.

- [ ] **Steps 1–5: Process each paper using R1–R5**

| # | Path (top-level unless noted) | Starter filename |
|---|---|---|
| 1 | `Unsupervised identification of the internal states that shape natural behavior.pdf` | `paper-calhoun-2019-internal-states-glmhmm.md` (verify; Calhoun/Pillow/Murthy) |
| 2 | `Natural behavior is the language of the brain.pdf` | `paper-datta-2023-natural-behavior-language.md` (verify author/year) |
| 3 | `Big behavioral data psychology, ethology and the foundations of neuroscience.pdf` | `paper-gomezmarin-2014-big-behavioral-data.md` (verify author/year) |
| 4 | `Spontaneous behaviors drive multidimensional, brainwide activity.pdf` | `paper-stringer-2019-spontaneous-brainwide-activity.md` (verify) |
| 5 | `striatum\Spontaneous behaviour is structured by reinforcement without explicit reward.pdf` | `paper-markowitz-2023-spontaneous-behaviour-reinforcement.md` (verify; DLS-DA + MoSeq) |

- [ ] **Step 6: Synthesis** → `synthesis-phase3-behavioral-state.md`. Sections: `## Latent behavioral states (GLM-HMM lineage)` · `## Spontaneous/movement signals in neural data` · `## Ethology framing` · `## Direct relevance to BG_046 (HMM states, engagement, movement confounds)` · `## Paper links`. Cross-link `[[paper-ashwood-2022-discrete-strategies]]`.
- [ ] **Step 7: MEMORY.md** — append:
  `- [Phase 3 — Behavioral-state & ethology](literature/synthesis-phase3-behavioral-state.md) — Calhoun internal-state GLM-HMM, Datta/Gomez-Marin ethology, Stringer spontaneous brain-wide activity, Markowitz DLS-DA structures spontaneous behaviour (5 papers)`
- [ ] **Step 8: V1.**  **Step 9: V2 + STOP.**

---

## Task P3-5: Dayan theoretical core (curated 6) + Bogacz + textbook on-demand

**Why BG_046:** theory backbone — population codes (↔ coding directions / decoding), uncertainty & neuromodulation (↔ behavioral-state regulation / HMM), goals-vs-habits (↔ D1/D2 control modes). Plus the classic Bogacz DDM theory to fill the dangling link.

**Files:** 6 × `paper-*.md` + (Bogacz, pending) + `synthesis-phase3-theory.md`; modify `MEMORY.md`. Optional textbook chapters use the **methods-ref schema**.

- [ ] **Step 0 (PRE-FLIGHT): reconfirm Bogacz filename** — Glob `G:\Postdoc_research\Mendeley_articles\**\*.pdf` (newest-first) + title-substring searches; if still absent, ask the user for the exact filename before proceeding. Read it via R1–R5 into `paper-bogacz-2006-decision-review.md` (verify Bogacz, Brown, Moehlis, Holmes, Cohen 2006, Psychological Review). On success, **repoint** the existing forward-marker links (boelts, collins-shenhav, edwards, masis, shadlen-kiani) only if they now resolve correctly (see parent memory carry-forward #3).

- [ ] **Steps 1–6: Process each paper using R1–R5** (all under `pDayan\`)

| # | Path | Starter filename |
|---|---|---|
| 1 | `Information processing with population codes.pdf` | `paper-pouget-2000-population-codes.md` (verify; Pouget/Dayan/Zemel) |
| 2 | `Uncertainty, Neuromodulation, and Attention.pdf` | `paper-yu-dayan-2005-uncertainty-neuromodulation.md` (verify) |
| 3 | `Uncertainty and learning.pdf` | `paper-dayan-uncertainty-and-learning.md` (verify author/year) |
| 4 | `Goals and Habits in the Brain.pdf` | `paper-dolan-dayan-2013-goals-habits.md` (verify) |
| 5 | `Goal-directed control and its antipodes.pdf` | `paper-dayan-goal-directed-control-antipodes.md` (verify) |
| 6 | `Choice values.pdf` | `paper-dayan-choice-values.md` (verify author/year) |

- [ ] **Step 7 (OPTIONAL, on-demand): textbook / anatomy chapters** — only if a specific question arises while writing the synthesis. Use methods-ref schema. Candidates: `pDayan\Theoretical neuroscience.pdf` (Dayan & Abbott) ch on population codes / RL; `The mouse nervous system  - Watson, Paxinos, Puelles\Chapter-7---Subpallial-Structures_2012_The-Mouse-Nervous-System.pdf` (striatum anatomy). Filenames: `methods-dayan-abbott-chNN-<topic>.md`, `methods-mns-ch07-subpallial-striatum.md`. Do NOT bulk-read.
- [ ] **Step 8: Synthesis** → `synthesis-phase3-theory.md`. Sections: `## Population codes & uncertainty representation` · `## Neuromodulation / behavioral-state control` · `## Goals vs habits (model-based/model-free ↔ D1/D2)` · `## DDM theory (Bogacz)` · `## Direct relevance to BG_046` · `## Paper links`. Cross-link `[[paper-daw-2005-uncertainty-competition]]`, `[[synthesis-batch04-modeling]]`.
- [ ] **Step 9: MEMORY.md** — append:
  `- [Phase 3 — Dayan theoretical core + Bogacz DDM](literature/synthesis-phase3-theory.md) — population codes, uncertainty/neuromodulation, goals-vs-habits, choice values, Bogacz 2006 optimal-decision DDM theory (6–7 entries)`
- [ ] **Step 10: V1.**  **Step 11: V2 + STOP.**

---

## Final Verification (after all desired Phase-3 tasks)

- [ ] Glob `memory/literature/paper-*.md` (expect ~68 + Phase-3 papers read).
- [ ] Glob `memory/literature/synthesis-phase3-*.md` (expect 1 per executed batch).
- [ ] Read MEMORY.md — `## Literature` has one new line per executed Phase-3 batch.
- [ ] Smoke test: pick an un-pre-considered question (e.g. "what physiologically distinguishes a striatal FSI from an SPN for waveform labeling?") and confirm it is answerable from memory alone.

## Acceptance criteria
- Each executed task produced its per-paper files + one synthesis + a MEMORY.md line.
- No silently-skipped papers (anything missing/unreadable reported in V2 with exact path).
- Out-of-scope clusters (NAc, addiction/clinical, tangential theory) left unread.
- Bogacz 2006 either read (link resolved) or explicitly reported still-absent with the failing path.
