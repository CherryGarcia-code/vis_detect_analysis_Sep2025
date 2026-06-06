# Literature Reading Plan — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **superpowers:executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Why executing-plans and not subagent-driven-development:** This plan is a *literature reading* plan, not code implementation. Each task processes 6-13 papers sequentially within one chat, with checkpoints between papers. `executing-plans` (sequential, checkpointed, in the current session) fits that shape. `subagent-driven-development` would dispatch a fresh subagent per task to read 8-12 papers and report back as a single block — that loses the per-paper visibility you want, and the subagent gets no benefit from the per-paper context that builds up while reading.
>
> **TDD mismatch is expected:** The skill is designed for code (write test → run test → implement → run test). There are no tests here — verification is "does the memory file exist with the expected schema and content". Treat the V1 step at the end of each task as the analog to "run tests": it confirms the artifacts landed. Do not invent code or unit tests for this plan.

**Goal:** Build a durable, queryable literature scaffold in `memory/literature/` covering decision-making theory, mouse/rodent perceptual decisions, striatal/BG circuits, computational modeling, and neural data-science methods — so that any future Claude session has working knowledge of the field for the BG_046 project.

**Architecture:** Phased reading. Each task = one batch (8-12 papers). Within a batch, each paper produces one `paper-*.md` memory file. The batch ends with a `synthesis-batch*.md` tying papers together and a `MEMORY.md` index update. Each task is self-contained — a fresh chat can execute any one task without context from this conversation.

**Tech Stack:** Read (PDFs, with page ranges for >20pp), Write (markdown memory files), Edit (MEMORY.md), Glob (path verification). Memory directory: `C:\Users\Ben\.claude\projects\e--python-analysis-git-repos-vis-detect-analysis-Sep2025\memory\`.

**Source spec:** `docs/superpowers/specs/2026-05-28-literature-reading-plan-design.md` — the authoritative reference for memory schema, batch composition, deviations notes, and acceptance criteria. **Read it before executing any task.**

---

## File Structure

```
memory/
  MEMORY.md                                       # add ## Literature section
  literature/                                     # created in Task 0 if missing
    paper-<author>-<year>-<topic>.md              # one per paper
    synthesis-batch01-foundations.md
    synthesis-batch02-striatum.md
    synthesis-batch03-rodent-perception.md
    synthesis-batch04-modeling.md
    synthesis-batch05-confidence-lapses.md
    synthesis-batch06-brainwide-population.md
    synthesis-batch07-sweep.md                    # conditional
    methods-nds-ch<NN>-<topic>.md                 # 6 chapter entries
    synthesis-methods-nds.md
```

Memory dir absolute path: `C:\Users\Ben\.claude\projects\e--python-analysis-git-repos-vis-detect-analysis-Sep2025\memory\`

---

## Common Per-Paper Workflow

This recipe applies to every paper task in Tasks 1-7. It is the *what to do for each paper*. Tasks list the specific papers; this section defines the steps.

### Recipe per paper

- [ ] **R1. Verify the PDF path exists**

```
Use the Glob tool on the exact path. If missing, report to user immediately; do not silently skip.
```

- [ ] **R2. Read the PDF**

For PDFs ≤20 pages: read the whole thing.
For PDFs >20 pages: use page ranges. Recommended priority order:
  1. Abstract + intro (pages 1-3)
  2. Discussion + figures (typically last 4-6 pages)
  3. Key methods (skim — only what's needed to interpret results)
  4. Results (selective — read sections matching the title's claim)

Aim for ~10-15 pages effective coverage even on long papers. Don't try to read 60-page methods papers cover to cover.

- [ ] **R3. Determine the filename**

Format: `paper-<firstauthor-lastname>-<year>-<short-topic>.md` (ASCII lowercase, hyphens, no spaces).

If the plan provides a starter filename, verify firstauthor and year against the paper. Rename if wrong. The starter name is a best guess — the paper itself is authoritative.

- [ ] **R4. Write the memory file** using exactly this schema

```markdown
---
name: paper-<firstauthor-lastname>-<year>-<short-topic>
description: <one-line summary that helps future-me decide if this is relevant>
metadata:
  type: reference
---

**Citation:** Author et al., Journal Year. doi or preprint id.
**Question:** <1 sentence — what they were asking>
**Paradigm/methods:** <2-4 sentences — species, task, recording/manipulation modality, key analyses>
**Findings (3-5 bullets):**
- <concrete claim with effect sizes/numbers where possible>
- <...>
**My synthesis:** <2-4 sentences — relevance to BG_046, what it connects to, weak points or open questions>
**Links:** [[other-paper-name]], [[scientific_context]], etc.
```

Target length: ~150 words body. Exception: the Orsolic paper (Task 1, paper 9) gets ~250 words with an extra `**BG_046 vs. Orsolic deviations:**` section — see Task 1 special note.

- [ ] **R5. Verify the file**

```
Use Glob on memory/literature/paper-*.md to confirm the new file appears.
```

### End-of-batch steps

These apply once per task, after all papers in a batch are processed and the synthesis file is written.

- [ ] **V1. Verify all batch artifacts exist**

```
Use Glob: memory/literature/paper-*.md  (expect: total ≥ sum of papers processed so far across all batches)
Use Glob: memory/literature/synthesis-batch<NN>-*.md  (expect: 1 — the current batch's synthesis)
Use Read on MEMORY.md to confirm the new ## Literature line is present for this batch
```

If any expected file is missing, fix before proceeding to V2.

- [ ] **V2. Report back to user — single message containing:**

- **Papers read:** Count + one-line per paper ("title — surprised by X" or "as expected").
- **Top 3 surprises / connections to BG_046 priorities** — the most useful signal from the batch.
- **Suggested adjustments to subsequent batches** — papers to move/swap/add/drop based on what was learned. Be specific (cite paper names + which batch).
- **Any papers that were unreadable or missing** — with the exact path that failed.
- **Memory entries written, by filename.**

After V2, **stop**. Do not auto-start the next batch — the user decides when to spin up the next chat.

---

## Task 0: Pre-flight Setup

**Files:**
- Create: `memory/literature/` (directory) if missing

- [ ] **Step 1: Read the source spec**

```
Use Read on: docs/superpowers/specs/2026-05-28-literature-reading-plan-design.md
```

Confirms current schema, batch composition, deviations.

- [ ] **Step 2: Check/create the literature subdirectory**

```bash
ls "C:\Users\Ben\.claude\projects\e--python-analysis-git-repos-vis-detect-analysis-Sep2025\memory\literature" 2>/dev/null || mkdir -p "C:\Users\Ben\.claude\projects\e--python-analysis-git-repos-vis-detect-analysis-Sep2025\memory\literature"
```

Expected: directory exists or is created silently.

- [ ] **Step 3: Read MEMORY.md to know current state**

```
Use Read on: C:\Users\Ben\.claude\projects\e--python-analysis-git-repos-vis-detect-analysis-Sep2025\memory\MEMORY.md
```

Note whether a `## Literature` section already exists (later tasks add to it).

---

## Task 1: Execute Batch 1 — Foundations + Direct Prior Art (12 papers)

**Files:**
- Create in `memory/literature/`:
  - `paper-gold-shadlen-2007-decision-window.md`
  - `paper-bogacz-2006-decision-review.md`
  - `paper-the-neural-basis-of-decision-making.md` (rename after reading per R3)
  - `paper-pouget-bayesian-inference-attention-decision.md` (verify author/year per R3)
  - `paper-khilkevich-lohse-2024-brainwide.md`
  - `paper-visual-evidence-accumulation-unrestrained-mice.md` (rename per R3)
  - `paper-ashwood-2022-discrete-strategies.md` (verify year per R3)
  - `paper-ibl-prior-information.md` (rename to specific first author per R3)
  - `paper-orsolic-mesoscale-task-origin.md` — **TASK-ORIGIN PAPER**
  - `paper-direct-indirect-pathways-perceptual.md` (rename per R3)
  - `paper-fast-slow-corticostriatal.md` (rename per R3)
  - `paper-daw-2005-uncertainty-competition.md`
  - `synthesis-batch01-foundations.md`
- Modify: `MEMORY.md` (add Literature section)

**Special note for paper 9 (Orsolic et al.):** Target ~250 words body. Add a `**BG_046 vs. Orsolic deviations:**` section capturing:
- Minimum change time / baseline duration (longer in BG_046 — record Orsolic's exact value)
- Airpuff punishment (not used in most BG_046 cohorts — note if/how Orsolic used it)
- Any other paradigm divergence encountered while reading

- [ ] **Step 1: Process paper 1 — Gold & Shadlen** using Recipe R1-R5

Path: `G:\Postdoc_research\Mendeley_articles\Decision Making as a Window on Cognition.pdf`
Starter filename: `paper-gold-shadlen-2007-decision-window.md`

- [ ] **Step 2: Process paper 2 — Bogacz** using R1-R5

Path: `G:\Postdoc_research\Mendeley_articles\Decision Making a Theoretical Review.pdf`
Starter filename: `paper-bogacz-2006-decision-review.md`

- [ ] **Step 3: Process paper 3 — The Neural Basis of Decision Making** using R1-R5

Path: `G:\Postdoc_research\Mendeley_articles\The Neural Basis of Decision Making.pdf`
Starter filename: `paper-the-neural-basis-of-decision-making.md` (rename after R3)

- [ ] **Step 4: Process paper 4 — Pouget/Beck Bayesian inference** using R1-R5

Path: `G:\Postdoc_research\Mendeley_articles\pDayan\Inference, attention, and decision in a Bayesian neural architecture.pdf`
Starter filename: `paper-pouget-bayesian-inference-attention-decision.md` (verify author/year)

- [ ] **Step 5: Process paper 5 — Khilkevich & Lohse** using R1-R5

Path: `G:\Postdoc_research\Mendeley_articles\Brain-wide dynamics linking sensation to action during decision-making.pdf`
Starter filename: `paper-khilkevich-lohse-2024-brainwide.md`

Extra: link this entry to `[[scientific_context]]` since it's the project's reference paper.

- [ ] **Step 6: Process paper 6 — Visual Evidence Accumulation in Unrestrained Mice** using R1-R5

Path: `G:\Postdoc_research\Mendeley_articles\Visual Evidence Accumulation Guides Decision-Making in Unrestrained Mice.pdf`
Starter filename: `paper-visual-evidence-accumulation-unrestrained-mice.md` (rename per R3)

- [ ] **Step 7: Process paper 7 — Ashwood (HMM strategies)** using R1-R5

Path: `G:\Postdoc_research\Mendeley_articles\Mice alternate between discrete strategies during perceptual decision-making.pdf`
Starter filename: `paper-ashwood-2022-discrete-strategies.md`

Extra: link to BG_046's HMM work — `[[handoff_refactor_may2026]]`-related and the project's HMM analyses. This paper is methodologically close to the project's `visdetect/analysis/hmm.py`.

- [ ] **Step 8: Process paper 8 — IBL prior information** using R1-R5

Path: `G:\Postdoc_research\Mendeley_articles\pDayan\Brain-wide representations of prior information in mouse decision-making.pdf`
Starter filename: `paper-ibl-prior-information.md` (rename to specific first author per R3)

- [ ] **Step 9: Process paper 9 — Orsolic et al. (TASK-ORIGIN PAPER)** using R1-R5 with ~250 word target and `BG_046 vs. Orsolic deviations` section

Path: `G:\Postdoc_research\Mendeley_articles\Mesoscale cortical dynamics reflect the interaction of sensory evidence and temporal expectation during perceptual decision-making.pdf`
Starter filename: `paper-orsolic-mesoscale-task-origin.md` (verify year — likely Orsolic, Rio, Mrsic-Flogel, Znamenskiy)

Use the schema in R4 but extend with:

```markdown
**BG_046 vs. Orsolic deviations:**
- **Minimum change time / baseline duration:** Orsolic used X s; BG_046 uses longer (verify exact values from project's CLAUDE.md or constants.py).
- **Airpuff punishment:** Orsolic [used / did not use]; most BG_046 cohorts do NOT use airpuff.
- **<other deviations encountered while reading>**
```

Link to `[[scientific_context]]` and to the project's task description in `[[handoff_refactor_may2026]]`.

- [ ] **Step 10: Process paper 10 — Direct/indirect pathways perceptual decisions** using R1-R5

Path: `G:\Postdoc_research\Mendeley_articles\The direct and indirect pathways of the basal ganglia antagonistically influence cortical activity and perceptual decisions.pdf`
Starter filename: `paper-direct-indirect-pathways-perceptual.md` (rename per R3)

- [ ] **Step 11: Process paper 11 — Fast/slow corticostriatal** using R1-R5

Path: `G:\Postdoc_research\Mendeley_articles\Fast and slow contributions to decision-making in corticostriatal circuits.pdf`
Starter filename: `paper-fast-slow-corticostriatal.md` (rename per R3)

- [ ] **Step 12: Process paper 12 — Daw/Niv/Dayan uncertainty competition** using R1-R5

Path: `G:\Postdoc_research\Mendeley_articles\pDayan\Uncertainty-based competition between prefrontal and dorsolateral striatal systems for behavioral control.pdf`
Starter filename: `paper-daw-2005-uncertainty-competition.md` (likely Daw, Niv, Dayan 2005)

- [ ] **Step 13: Write batch synthesis**

File: `memory/literature/synthesis-batch01-foundations.md`

Use this skeleton:

```markdown
---
name: synthesis-batch01-foundations
description: Synthesis of 12 foundational and direct-prior-art decision-making papers (Gold/Shadlen, Bogacz, Khilkevich/Lohse, Orsolic, Ashwood, IBL, Daw/Niv/Dayan, etc.)
metadata:
  type: reference
---

## Convergent claims
<3-6 bullets — things multiple papers agree on>

## Disagreements / open questions
<2-4 bullets — where the field is split>

## Methodological themes
<2-4 bullets — what kinds of tasks, models, analyses recur>

## Direct relevance to BG_046
<3-5 bullets tied to project priorities: HMM, sequential dynamics, evidence-axis, striatal cell types, Orsolic task lineage>

## Paper links
- [[paper-gold-shadlen-2007-decision-window]]
- [[paper-bogacz-2006-decision-review]]
- [[paper-the-neural-basis-of-decision-making]] (or renamed)
- [[paper-pouget-bayesian-inference-attention-decision]] (or renamed)
- [[paper-khilkevich-lohse-2024-brainwide]]
- [[paper-visual-evidence-accumulation-unrestrained-mice]] (or renamed)
- [[paper-ashwood-2022-discrete-strategies]]
- [[paper-ibl-prior-information]] (or renamed)
- [[paper-orsolic-mesoscale-task-origin]]
- [[paper-direct-indirect-pathways-perceptual]] (or renamed)
- [[paper-fast-slow-corticostriatal]] (or renamed)
- [[paper-daw-2005-uncertainty-competition]]
```

- [ ] **Step 14: Update MEMORY.md**

Use Read first to see current state. Add a `## Literature` heading if not present, then append:

```markdown
## Literature
- [Batch 1: Decision-making foundations + direct prior art](literature/synthesis-batch01-foundations.md) — Gold/Shadlen, Bogacz, Khilkevich/Lohse, Orsolic task-origin, Ashwood HMM, Daw uncertainty competition (12 papers)
```

Use Edit to add this — do NOT use Write (would overwrite MEMORY.md).

- [ ] **Step 15: Verify all batch artifacts (apply V1)**

- [ ] **Step 16: Report back to user (apply V2)**

---

## Task 2: Execute Batch 2 — Striatum / BG decisions (9 papers)

**Files:**
- Create in `memory/literature/`:
  - 9 `paper-*.md` files (starter names below; rename per R3)
  - `synthesis-batch02-striatum.md`
- Modify: `MEMORY.md` (append Literature line)

- [ ] **Step 1-9: Process each paper using Recipe R1-R5**

| # | Path | Starter filename |
|---|------|------------------|
| 1 | `G:\Postdoc_research\Mendeley_articles\striatum\Distinct roles of striatal direct and indirect pathways in value-based decision making.pdf` | `paper-distinct-roles-direct-indirect-value.md` |
| 2 | `G:\Postdoc_research\Mendeley_articles\The caudate nucleus contributes causally to decisions that balance reward and uncertain visual information.pdf` | `paper-caudate-causal-reward-uncertainty.md` |
| 3 | `G:\Postdoc_research\Mendeley_articles\Reward-driven changes in striatal pathway competition shape evidence evaluation in decision-making.pdf` | `paper-reward-driven-striatal-competition.md` |
| 4 | `G:\Postdoc_research\Mendeley_articles\A Corticostriatal Path Targeting Striosomes Controls Decision-Making under Conflict.pdf` | `paper-corticostriatal-striosomes-conflict.md` |
| 5 | `G:\Postdoc_research\Mendeley_articles\Corticostriatal Interactions during Learning, Memory Processing, and Decision Making.pdf` | `paper-corticostriatal-interactions-learning-memory.md` |
| 6 | `G:\Postdoc_research\Mendeley_articles\Temporal regularities shape perceptual decisions and striatal dopamine signals.pdf` | `paper-temporal-regularities-striatal-da.md` |
| 7 | `G:\Postdoc_research\Mendeley_articles\Dynamic control of decision and movement speed in the human basal ganglia.pdf` | `paper-dynamic-control-decision-movement-bg.md` |
| 8 | `G:\Postdoc_research\Mendeley_articles\Reinforcement-Based Decision Making in Corticostriatal Circuits- Mutual Constraints by Neurocomputational and Diffusion Models.pdf` | `paper-frank-reinforcement-corticostriatal-ddm.md` |
| 9 | `G:\Postdoc_research\Mendeley_articles\striatum\Temporal integration is a robust feature of perceptual decisions.pdf` | `paper-temporal-integration-robust.md` |

For each: apply R1 (verify path) → R2 (read with page strategy) → R3 (determine final filename) → R4 (write file using schema) → R5 (verify file).

- [ ] **Step 10: Write batch synthesis**

File: `memory/literature/synthesis-batch02-striatum.md`

Use the same skeleton as Task 1 Step 13 but with `name: synthesis-batch02-striatum`, theme = "striatal/BG involvement in decisions", and the 9 paper links above. Section headings:
- ## Convergent claims about D1/D2 contributions
- ## Disagreements (e.g., are pathways antagonistic or cooperative?)
- ## Methodological themes (optogenetics, recording, modeling)
- ## Direct relevance to BG_046 (D1/D2 SPN classification, evidence-axis, etc.)
- ## Paper links

- [ ] **Step 11: Update MEMORY.md**

Edit `MEMORY.md` under `## Literature`. Append:

```markdown
- [Batch 2: Striatum/BG decisions](literature/synthesis-batch02-striatum.md) — Direct/indirect pathways, caudate, striosomes, Frank reinforcement model, temporal integration (9 papers)
```

- [ ] **Step 12: Verify all batch artifacts (apply V1)**

- [ ] **Step 13: Report back to user (apply V2)**

---

## Task 3: Execute Batch 3 — Mouse/rodent perceptual decisions (8 papers)

**Files:**
- Create in `memory/literature/`: 8 `paper-*.md` + `synthesis-batch03-rodent-perception.md`
- Modify: `MEMORY.md`

- [ ] **Step 1-8: Process each paper using Recipe R1-R5**

| # | Path | Starter filename |
|---|------|------------------|
| 1 | `G:\Postdoc_research\Mendeley_articles\Posterior Parietal Cortex Guides Visual Decisions in Rats.pdf` | `paper-ppc-visual-decisions-rats.md` |
| 2 | `G:\Postdoc_research\Mendeley_articles\Distinct relationships of parietal and prefrontal cortices to evidence accumulation.pdf` | `paper-parietal-pfc-evidence-accumulation.md` |
| 3 | `G:\Postdoc_research\Mendeley_articles\Distinct roles of visual, parietal, and frontal motor cortices in memory-guided sensorimotor decisions.pdf` | `paper-visual-parietal-frontal-memory-guided.md` |
| 4 | `G:\Postdoc_research\Mendeley_articles\Sensory coding and the causal impact of mouse cortex in a visual decision.pdf` | `paper-sensory-coding-causal-cortex-visual.md` |
| 5 | `G:\Postdoc_research\Mendeley_articles\Multisensory task demands temporally extend the causal requirement for visual cortex in perception.pdf` | `paper-multisensory-task-demands-v1.md` |
| 6 | `G:\Postdoc_research\Mendeley_articles\Mouse frontal cortex mediates additive multisensory decisions.pdf` | `paper-mouse-frontal-additive-multisensory.md` |
| 7 | `G:\Postdoc_research\Mendeley_articles\A rapid whisker-based decision underlying skilled locomotion in mice.pdf` | `paper-rapid-whisker-decision-locomotion.md` |
| 8 | `G:\Postdoc_research\Mendeley_articles\Excitatory and Inhibitory Subnetworks Are Equally Selective during Decision-Making and Emerge Simultaneously during Learning.pdf` | `paper-exc-inh-subnetworks-selective.md` |

- [ ] **Step 9: Write batch synthesis**

File: `memory/literature/synthesis-batch03-rodent-perception.md`. Skeleton like Task 1 Step 13. Theme: "rodent cortical contributions to perceptual decisions". Section headings:
- ## Convergent claims about cortical roles (PPC, frontal, V1)
- ## Disagreements / open questions
- ## Methodological themes (optogenetic silencing, calcium imaging, Neuropixels)
- ## Direct relevance to BG_046 (since BG_046 records striatum, what does cortex tell us about the upstream input?)
- ## Paper links

- [ ] **Step 10: Update MEMORY.md**

Append:

```markdown
- [Batch 3: Rodent perceptual decisions](literature/synthesis-batch03-rodent-perception.md) — PPC, frontal, V1 causal roles; multisensory; whisker (8 papers)
```

- [ ] **Step 11: Verify all batch artifacts (apply V1)**

- [ ] **Step 12: Report back to user (apply V2)**

---

## Task 4: Execute Batch 4 — Modeling & latent variables (8 papers)

**Files:** 8 `paper-*.md` + `synthesis-batch04-modeling.md` + MEMORY.md update.

- [ ] **Step 1-8: Process each paper using Recipe R1-R5**

| # | Path | Starter filename |
|---|------|------------------|
| 1 | `G:\Postdoc_research\Mendeley_articles\Advances in modeling learning and decision-making in neuroscience.pdf` | `paper-advances-modeling-learning-decision.md` |
| 2 | `G:\Postdoc_research\Mendeley_articles\Flexible and efficient simulation-based inference for models of decision-making.pdf` | `paper-flexible-sbi-decision.md` |
| 3 | `G:\Postdoc_research\Mendeley_articles\A new theoretical framework jointly explains behavioral and neural variability across subjects performing flexible decision-making.pdf` | `paper-joint-behavioral-neural-variability.md` |
| 4 | `G:\Postdoc_research\Mendeley_articles\A decision-space model explains context-specific decision-making.pdf` | `paper-decision-space-context.md` |
| 5 | `G:\Postdoc_research\Mendeley_articles\Quantifying decision-making in dynamic, continuously evolving environments.pdf` | `paper-quantifying-dynamic-environments.md` |
| 6 | `G:\Postdoc_research\Mendeley_articles\Choice history biases subsequent evidence accumulation.pdf` | `paper-choice-history-biases-accumulation.md` |
| 7 | `G:\Postdoc_research\Mendeley_articles\pDayan\Low dimensional latent structure underlying the choices of mice.pdf` | `paper-low-dim-latent-mouse-choices.md` |
| 8 | `G:\Postdoc_research\Mendeley_articles\Initial conditions combine with sensory evidence to induce decision-related dynamics in premotor cortex.pdf` | `paper-initial-conditions-premotor-dynamics.md` |

- [ ] **Step 9: Write batch synthesis**

File: `memory/literature/synthesis-batch04-modeling.md`. Theme: "computational frameworks and latent-variable models for decision behavior + neural data". Section headings:
- ## Convergent modeling approaches (DDM family, GLM-HMM, SBI, latent-state)
- ## Disagreements / open questions (e.g., what counts as "evidence accumulation"?)
- ## Tooling notes (which packages, which fitting strategies)
- ## Direct relevance to BG_046 (HMM, evidence-axis decoding, behavior modeling)
- ## Paper links

- [ ] **Step 10: Update MEMORY.md**

Append:

```markdown
- [Batch 4: Modeling & latent variables](literature/synthesis-batch04-modeling.md) — SBI, GLM-HMM, joint behavioral+neural models, choice history (8 papers)
```

- [ ] **Step 11: Verify all batch artifacts (apply V1)**

- [ ] **Step 12: Report back to user (apply V2)**

---

## Task 5: Execute Batch 5 — Confidence, lapses, exploration, biases (8 papers)

**Files:** 8 `paper-*.md` + `synthesis-batch05-confidence-lapses.md` + MEMORY.md update.

- [ ] **Step 1-8: Process each paper using Recipe R1-R5**

| # | Path | Starter filename |
|---|------|------------------|
| 1 | `G:\Postdoc_research\Mendeley_articles\Neural correlates, computation and behavioural impact of decision confidence.pdf` | `paper-neural-confidence-correlates.md` |
| 2 | `G:\Postdoc_research\Mendeley_articles\Lapses in perceptual decisions reflect exploration.pdf` | `paper-pisupati-lapses-exploration.md` |
| 3 | `G:\Postdoc_research\Mendeley_articles\Reinforcement biases subsequent perceptual decisions when confidence is low, a widespread behavioral phenomenon.pdf` | `paper-reinforcement-biases-low-confidence.md` |
| 4 | `G:\Postdoc_research\Mendeley_articles\Strategically managing learning during perceptual decision making.pdf` | `paper-strategically-managing-learning.md` |
| 5 | `G:\Postdoc_research\Mendeley_articles\The impact of learning on perceptual decisions and its implication for speed-accuracy tradeoffs.pdf` | `paper-learning-speed-accuracy.md` |
| 6 | `G:\Postdoc_research\Mendeley_articles\Perceptual decisions and oculomotor responses rely on temporally distinct streams of evidence.pdf` | `paper-perceptual-oculomotor-distinct-streams.md` |
| 7 | `G:\Postdoc_research\Mendeley_articles\Pupil diameter encodes the idiosyncratic, cognitive complexity of belief updating.pdf` | `paper-pupil-belief-updating.md` |
| 8 | `G:\Postdoc_research\Mendeley_articles\Conversion of sensory signals into perceptual decisions.pdf` | `paper-conversion-sensory-perceptual.md` |

- [ ] **Step 9: Write batch synthesis**

File: `memory/literature/synthesis-batch05-confidence-lapses.md`. Theme: "confidence, lapses, biases — the not-just-correct/incorrect dimensions". Section headings:
- ## Convergent claims about confidence computation
- ## Lapses as exploration vs. inattention vs. bias
- ## Methodological themes (psychometric fitting, RT analyses, pupillometry)
- ## Direct relevance to BG_046 (early licks/FAs, lapses, learning-stage differences)
- ## Paper links

- [ ] **Step 10: Update MEMORY.md**

Append:

```markdown
- [Batch 5: Confidence, lapses, biases](literature/synthesis-batch05-confidence-lapses.md) — Pisupati lapses, confidence computation, choice-history biases, pupil belief updates (8 papers)
```

- [ ] **Step 11: Verify all batch artifacts (apply V1)**

- [ ] **Step 12: Report back to user (apply V2)**

---

## Task 6: Execute Batch 6 — Brain-wide & population dynamics (9 papers)

**Files:** 9 `paper-*.md` + `synthesis-batch06-brainwide-population.md` + MEMORY.md update.

- [ ] **Step 1-9: Process each paper using Recipe R1-R5**

| # | Path | Starter filename |
|---|------|------------------|
| 1 | `G:\Postdoc_research\Mendeley_articles\Distributed coding of choice, action and engagement across the mouse brain.pdf` | `paper-distributed-coding-choice-engagement.md` |
| 2 | `G:\Postdoc_research\Mendeley_articles\pDayan\A brain-wide map of neural activity during complex behaviour.pdf` | `paper-ibl-brainwide-map.md` |
| 3 | `G:\Postdoc_research\Mendeley_articles\A reservoir of foraging decision variables in the mouse brain.pdf` | `paper-reservoir-foraging-decision-variables.md` |
| 4 | `G:\Postdoc_research\Mendeley_articles\Brain-wide interactions between neural circuits.pdf` | `paper-brainwide-circuit-interactions.md` |
| 5 | `G:\Postdoc_research\Mendeley_articles\Neural dynamics outside task-coding dimensions drive decision trajectories through transient amplification.pdf` | `paper-transient-amplification-decision.md` |
| 6 | `G:\Postdoc_research\Mendeley_articles\A frontal motor circuit for economic decisions and actions.pdf` | `paper-frontal-motor-economic-decisions.md` |
| 7 | `G:\Postdoc_research\Mendeley_articles\Flow of Cortical Activity Underlying a Tactile Decision in Mice.pdf` | `paper-flow-cortical-tactile-decision.md` |
| 8 | `G:\Postdoc_research\Mendeley_articles\Neural Activity in Macaque Parietal Cortex Reflects Temporal Integration of Visual Motion Signals during Perceptual Decision Making.pdf` | `paper-roitman-shadlen-2002-lip-integration.md` |
| 9 | `G:\Postdoc_research\Mendeley_articles\A neural mechanism for terminating decisions.pdf` | `paper-neural-mechanism-terminating-decisions.md` |

- [ ] **Step 10: Write batch synthesis**

File: `memory/literature/synthesis-batch06-brainwide-population.md`. Theme: "distributed neural dynamics across the brain during decisions". Section headings:
- ## Convergent claims about distributed coding
- ## Coding dimensions and orthogonal subspaces (links to BG_046 CD work)
- ## Termination/commitment mechanisms
- ## Methodological themes (large-scale recording, population geometry)
- ## Direct relevance to BG_046 (CD orthogonality, sequential dynamics, brain-wide framing)
- ## Paper links

- [ ] **Step 11: Update MEMORY.md**

Append:

```markdown
- [Batch 6: Brain-wide & population dynamics](literature/synthesis-batch06-brainwide-population.md) — IBL brain-wide map, Steinmetz distributed coding, Roitman/Shadlen LIP, transient amplification (9 papers)
```

- [ ] **Step 12: Verify all batch artifacts (apply V1)**

- [ ] **Step 13: Report back to user (apply V2)**

---

## Task 7 (CONDITIONAL): Execute Sweep Batch

**Run only if:** the user explicitly asks for Phase 1 closure or batches 1-6 revealed important gaps. Otherwise skip and proceed to Task 8.

**Files:** Up to 13 `paper-*.md` + `synthesis-batch07-sweep.md` + MEMORY.md update.

- [ ] **Step 1-13: Process each paper using Recipe R1-R5**

| # | Path | Starter filename |
|---|------|------------------|
| 1 | `G:\Postdoc_research\Mendeley_articles\The theory of decision making.pdf` | `paper-edwards-1954-theory-decision.md` |
| 2 | `G:\Postdoc_research\Mendeley_articles\Mice exhibit stochastic and efficient action switching duringprobabilistic decision making.pdf` | `paper-mice-stochastic-action-switching.md` |
| 3 | `G:\Postdoc_research\Mendeley_articles\Probing perceptual decisions in rodents.pdf` | `paper-probing-perceptual-rodents.md` |
| 4 | `G:\Postdoc_research\Mendeley_articles\Seeing at a glance, smelling in a whiff- rapid forms of perceptual decision making.pdf` | `paper-seeing-glance-smelling-whiff.md` |
| 5 | `G:\Postdoc_research\Mendeley_articles\The mechanistic foundation of Weber's law.pdf` | `paper-mechanistic-webers-law.md` |
| 6 | `G:\Postdoc_research\Mendeley_articles\Using temperature to analyze the neural basis of a time-based decision.pdf` | `paper-temperature-time-based-decision.md` |
| 7 | `G:\Postdoc_research\Mendeley_articles\Value dynamics affect choice preparation during decision-making.pdf` | `paper-value-dynamics-choice-preparation.md` |
| 8 | `G:\Postdoc_research\Mendeley_articles\Competitive integration of time and reward explains value-sensitive foraging decisions and frontal cortex ramping dynamics.pdf` | `paper-competitive-integration-time-reward.md` |
| 9 | `G:\Postdoc_research\Mendeley_articles\Decision-making in sensorimotor control.pdf` | `paper-decision-sensorimotor-control.md` |
| 10 | `G:\Postdoc_research\Mendeley_articles\Neuronal Correlates of a Perceptual Decision in Ventral Premotor Cortex.pdf` | `paper-neuronal-correlates-pmv.md` |
| 11 | `G:\Postdoc_research\Mendeley_articles\Rats and humans can optimally accumulate evidence for decision-making.pdf` | `paper-brunton-rats-humans-evidence.md` |
| 12 | `G:\Postdoc_research\Mendeley_articles\Pyramidal cell types drive functionally distinct cortical activity patterns during decision-making.pdf` | `paper-pyramidal-celltypes-decision.md` |
| 13 | `G:\Postdoc_research\Mendeley_articles\Stop Signals Provide Cross Inhibition in Collective Decision-Making by Honeybee Swarms.pdf` | `paper-honeybee-stop-signals.md` (skip unless time) |

- [ ] **Step 14: Write batch synthesis**

File: `memory/literature/synthesis-batch07-sweep.md`. Skeleton like Task 1 Step 13. Theme: "remaining decision-titled papers — historical, niche, or methodologically distinctive". Acknowledge in the file that this batch is intentionally heterogeneous.

- [ ] **Step 15: Update MEMORY.md**

Append:

```markdown
- [Batch 7: Sweep — remaining decision papers](literature/synthesis-batch07-sweep.md) — Brunton evidence integration, Weber's law, time-based decisions, Edwards historical, etc. (up to 13 papers)
```

- [ ] **Step 16: Verify all batch artifacts (apply V1)**

- [ ] **Step 17: Report back to user (apply V2)**

---

## Task 8: Execute Methods Batch — Neural Data Science chapters (6 entries)

**Files:** 6 `methods-nds-ch*.md` + `synthesis-methods-nds.md` + MEMORY.md update.

**Treatment difference:** These are textbook chapters, not papers. The memory entry should be a **methods reference**, not a paper summary. Use this modified schema:

```markdown
---
name: methods-nds-ch<NN>-<topic>
description: <one-line — what this chapter is useful for>
metadata:
  type: reference
---

**Source:** Neural Data Science (2017), Chapter NN — <chapter title>.
**Topic:** <1 sentence — what the chapter covers>
**Canonical methods (3-5 bullets):** Standard techniques with their typical use case
**Frequently made mistakes (2-3 bullets, if the chapter discusses them; otherwise omit this field):** What the chapter warns against (cross-reference Appendix B where relevant)
**When to use in BG_046 context:** <2-3 sentences — which project analyses this would inform>
**Code/library pointers:** <if the chapter recommends specific tools — e.g., scipy.stats.ks_2samp, scikit-learn, statsmodels>
**Links:** [[methods-nds-ch<other>]], [[other-paper]]
```

- [ ] **Step 1: Process Chapter 3 — Wrangling Spike Trains** using R1-R5 with the methods schema above

Path: `G:\Postdoc_research\Mendeley_articles\data_science\Chapter-3---Wrangling-Spike-Trains_2017_Neural-Data-Science.pdf`
Filename: `methods-nds-ch03-wrangling-spike-trains.md`

- [ ] **Step 2: Process Chapter 4 — Correlating Spike Trains**

Path: `G:\Postdoc_research\Mendeley_articles\data_science\Chapter-4---Correlating-Spike-Trains_2017_Neural-Data-Science.pdf`
Filename: `methods-nds-ch04-correlating-spike-trains.md`

- [ ] **Step 3: Process Chapter 7 — Regression**

Path: `G:\Postdoc_research\Mendeley_articles\data_science\Chapter-7---Regression_2017_Neural-Data-Science.pdf`
Filename: `methods-nds-ch07-regression.md`

- [ ] **Step 4: Process Chapter 8 — Dimensionality Reduction**

Path: `G:\Postdoc_research\Mendeley_articles\data_science\Chapter-8---Dimensionality-Reduction_2017_Neural-Data-Science.pdf`
Filename: `methods-nds-ch08-dimensionality-reduction.md`

- [ ] **Step 5: Process Chapter 9 — Classification and Clustering**

Path: `G:\Postdoc_research\Mendeley_articles\data_science\Chapter-9---Classification-and-Clustering_2017_Neural-Data-Science.pdf`
Filename: `methods-nds-ch09-classification-clustering.md`

- [ ] **Step 6: Process Appendix B — Frequently Made Mistakes**

Path: `G:\Postdoc_research\Mendeley_articles\data_science\Appendix-B---Frequently-Made-Mistakes_2017_Neural-Data-Science.pdf`
Filename: `methods-nds-appB-frequent-mistakes.md`

- [ ] **Step 7: Write methods synthesis**

File: `memory/literature/synthesis-methods-nds.md`. Skeleton:

```markdown
---
name: synthesis-methods-nds
description: Methods reference organized by question type — which NDS chapter to consult for which analysis
metadata:
  type: reference
---

## When to use which chapter

| If you want to... | Look at | Key idea |
|--------------------|---------|----------|
| Detect/clean spike trains | [[methods-nds-ch03-wrangling-spike-trains]] | ... |
| Compute cross-correlations / co-firing | [[methods-nds-ch04-correlating-spike-trains]] | ... |
| Fit a GLM / regression | [[methods-nds-ch07-regression]] | ... |
| Reduce dimensions (PCA, factor analysis) | [[methods-nds-ch08-dimensionality-reduction]] | ... |
| Classify cell types / states | [[methods-nds-ch09-classification-clustering]] | ... |
| Avoid common pitfalls | [[methods-nds-appB-frequent-mistakes]] | always read first |

## Cross-cutting warnings
<3-5 bullets — recurring "don't do this" themes>

## Project relevance map
<3-5 bullets tying chapters to specific BG_046 analyses (e.g., Ch 8 → CD/dPCA; Ch 7 → GLM-HMM, lick-hazard GLM; Appendix B → general QC discipline)>
```

- [ ] **Step 8: Update MEMORY.md**

Append:

```markdown
- [Methods: Neural Data Science chapters](literature/synthesis-methods-nds.md) — When to use which chapter for spike trains, regression, dim reduction, classification (6 entries)
```

- [ ] **Step 9: Verify all batch artifacts (apply V1)**

- [ ] **Step 10: Report back to user (apply V2)**

---

## Final Verification (run after all desired tasks complete)

- [ ] **Step 1: Count memory files**

```
Use Glob: memory/literature/paper-*.md  (expect ~50-55 if Phase 1 complete)
Use Glob: memory/literature/methods-nds-*.md  (expect 6)
Use Glob: memory/literature/synthesis-*.md  (expect 7-8)
```

- [ ] **Step 2: Confirm MEMORY.md index**

```
Use Read on MEMORY.md
```

Expected: `## Literature` section with one bullet per completed batch (linking to synthesis files, not individual papers).

- [ ] **Step 3: Smoke test the scaffold**

Pick a question you didn't think about while reading — e.g., "What does the Frank corticostriatal DDM say about response time prediction?" — and verify you can answer it using only the memory files (no re-reading PDFs). If not, the memory entries are too sparse and need a follow-up pass.

---

## Acceptance criteria

- Each executed task produced:
  - The expected per-paper memory files (named per schema, content per schema, ~150 words each except Orsolic at ~250)
  - One batch synthesis file
  - A MEMORY.md update
- Future cold sessions can answer questions about specific papers from memory alone
- The user can ask cross-paper questions and get coherent answers
- No silently-skipped papers (anything unreadable was reported to the user)
