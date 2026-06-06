# Literature Reading Plan — Design Spec

**Date:** 2026-05-28
**Owner:** b.gonzales@ucl.ac.uk (BG_046 project)
**Source folder:** `G:\Postdoc_research\Mendeley_articles` (645 PDFs, 463 at top level)
**Goal:** Build a durable literature scaffold in the project's memory system so that future Claude sessions have working knowledge of decision-making theory, experimental paradigms, and analysis methods relevant to the BG_046 visual change-detection project.

---

## Scope

- **Phase 1 (this spec, ~55 papers):** All PDFs with "decision" in the title, plus targeted inserts from `pDayan/` and the Orsolic task-origin paper. Six thematic batches.
- **Phase 2 (~6 entries):** *Neural Data Science* textbook chapters as a methods reference.
- **Phase 3 (TBD):** Adjacent literature (striatum/BG anatomy, RL, behavior ethology, anatomy/connectivity). Planned after Phase 1 reveals gaps.

Out of scope: comprehensive reading of every PDF in the folder; books beyond targeted chapters; non-English papers; reviews older than 2000 unless explicitly listed.

---

## Memory schema

One markdown file per paper under `memory/literature/`. Frontmatter and body structure:

```markdown
---
name: paper-<firstauthor-lastname>-<year>-<short-topic>
description: <one-line summary that helps future-me decide if this is relevant>
metadata:
  type: reference
---

**Citation:** Author et al., Journal Year. doi or preprint id.
**Question:** What they were asking (1 sentence).
**Paradigm/methods:** Species, task, recording/manipulation modality, key analyses (2-4 sentences).
**Findings (3-5 bullets):** Concrete claims with effect sizes/numbers where possible. Not vague summaries.
**My synthesis:** Why this matters for the BG_046 project. What it connects to. Weak points or open questions. (2-4 sentences.)
**Links:** [[other-paper-name]] for related entries; [[scientific_context]] etc. for project memories.
```

Target length: ~150 words per paper ("tight summary").

**Per-batch synthesis** lives at `memory/literature/synthesis-batch<NN>-<theme>.md` with the same frontmatter (`type: reference`) but a body that ties papers together:
- Convergent claims across the batch
- Disagreements / open questions
- Methodological themes
- Direct relevance to BG_046 priorities (sequential dynamics, evidence-axis, HMM, striatal cell types)
- Links to individual paper entries

---

## MEMORY.md integration

Only the **synthesis files** get listed in `MEMORY.md` (under a new `## Literature` section). Individual paper entries remain discoverable via glob/grep but don't clutter the index. Expected final state: ~8 synthesis entries indexed when Phases 1+2 are done.

---

## Phase 1 batches

Each batch entry below gives the full PDF path so an execution chat can read directly without searching. Paths use Windows backslashes as they appear on disk.

### Batch 1 — Foundations + Direct Prior Art (12 papers)

**Theme:** Theoretical foundations of perceptual decision-making, plus the papers most directly upstream of the BG_046 task.

**Special note:** Paper #9 (Orsolic et al.) is the **task-origin paper** for the BG_046 visual change-detection paradigm. Treat with extra care: capture paradigm parameters (TF base, change ratios, baseline duration, response window), exclusion criteria, and any methodological choices that BG_046 inherits or deviates from. Write a slightly longer entry (~250 words) for this one.

**Known BG_046 deviations from the Orsolic paradigm** (verify and capture in the memory entry as a "BG_046 vs. Orsolic" section so future analyses don't conflate the two):
- **Minimum change time / baseline duration** is longer in BG_046 than in the Orsolic original. Capture the exact Orsolic value(s) and flag the difference.
- **Airpuff punishment** is not used in most BG_046 cohorts. If Orsolic used it, note when and how.
- Anything else encountered while reading should also be added to that section.

1. `G:\Postdoc_research\Mendeley_articles\Decision Making as a Window on Cognition.pdf` — Gold & Shadlen review
2. `G:\Postdoc_research\Mendeley_articles\Decision Making a Theoretical Review.pdf` — Bogacz drift-diffusion/race models
3. `G:\Postdoc_research\Mendeley_articles\The Neural Basis of Decision Making.pdf`
4. `G:\Postdoc_research\Mendeley_articles\pDayan\Inference, attention, and decision in a Bayesian neural architecture.pdf`
5. `G:\Postdoc_research\Mendeley_articles\Brain-wide dynamics linking sensation to action during decision-making.pdf` — Khilkevich & Lohse (CLAUDE.md reference paper)
6. `G:\Postdoc_research\Mendeley_articles\Visual Evidence Accumulation Guides Decision-Making in Unrestrained Mice.pdf`
7. `G:\Postdoc_research\Mendeley_articles\Mice alternate between discrete strategies during perceptual decision-making.pdf` — Ashwood (HMM, directly relevant to HMM-Track-A work)
8. `G:\Postdoc_research\Mendeley_articles\pDayan\Brain-wide representations of prior information in mouse decision-making.pdf` — IBL
9. `G:\Postdoc_research\Mendeley_articles\Mesoscale cortical dynamics reflect the interaction of sensory evidence and temporal expectation during perceptual decision-making.pdf` — **Orsolic, Rio, Mrsic-Flogel, Znamenskiy — TASK-ORIGIN PAPER for BG_046**
10. `G:\Postdoc_research\Mendeley_articles\The direct and indirect pathways of the basal ganglia antagonistically influence cortical activity and perceptual decisions.pdf`
11. `G:\Postdoc_research\Mendeley_articles\Fast and slow contributions to decision-making in corticostriatal circuits.pdf`
12. `G:\Postdoc_research\Mendeley_articles\pDayan\Uncertainty-based competition between prefrontal and dorsolateral striatal systems for behavioral control.pdf` — Daw/Niv/Dayan

Synthesis file: `synthesis-batch01-foundations.md`

### Batch 2 — Striatum / Basal Ganglia decisions (~9)

1. `G:\Postdoc_research\Mendeley_articles\striatum\Distinct roles of striatal direct and indirect pathways in value-based decision making.pdf`
2. `G:\Postdoc_research\Mendeley_articles\The caudate nucleus contributes causally to decisions that balance reward and uncertain visual information.pdf`
3. `G:\Postdoc_research\Mendeley_articles\Reward-driven changes in striatal pathway competition shape evidence evaluation in decision-making.pdf`
4. `G:\Postdoc_research\Mendeley_articles\A Corticostriatal Path Targeting Striosomes Controls Decision-Making under Conflict.pdf`
5. `G:\Postdoc_research\Mendeley_articles\Corticostriatal Interactions during Learning, Memory Processing, and Decision Making.pdf`
6. `G:\Postdoc_research\Mendeley_articles\Temporal regularities shape perceptual decisions and striatal dopamine signals.pdf`
7. `G:\Postdoc_research\Mendeley_articles\Dynamic control of decision and movement speed in the human basal ganglia.pdf`
8. `G:\Postdoc_research\Mendeley_articles\Reinforcement-Based Decision Making in Corticostriatal Circuits- Mutual Constraints by Neurocomputational and Diffusion Models.pdf` — Frank
9. `G:\Postdoc_research\Mendeley_articles\striatum\Temporal integration is a robust feature of perceptual decisions.pdf`

Synthesis file: `synthesis-batch02-striatum.md`

### Batch 3 — Mouse/rodent perceptual decisions (~8)

1. `G:\Postdoc_research\Mendeley_articles\Posterior Parietal Cortex Guides Visual Decisions in Rats.pdf`
2. `G:\Postdoc_research\Mendeley_articles\Distinct relationships of parietal and prefrontal cortices to evidence accumulation.pdf`
3. `G:\Postdoc_research\Mendeley_articles\Distinct roles of visual, parietal, and frontal motor cortices in memory-guided sensorimotor decisions.pdf`
4. `G:\Postdoc_research\Mendeley_articles\Sensory coding and the causal impact of mouse cortex in a visual decision.pdf`
5. `G:\Postdoc_research\Mendeley_articles\Multisensory task demands temporally extend the causal requirement for visual cortex in perception.pdf`
6. `G:\Postdoc_research\Mendeley_articles\Mouse frontal cortex mediates additive multisensory decisions.pdf`
7. `G:\Postdoc_research\Mendeley_articles\A rapid whisker-based decision underlying skilled locomotion in mice.pdf`
8. `G:\Postdoc_research\Mendeley_articles\Excitatory and Inhibitory Subnetworks Are Equally Selective during Decision-Making and Emerge Simultaneously during Learning.pdf`

Synthesis file: `synthesis-batch03-rodent-perception.md`

### Batch 4 — Modeling & latent variables (~8)

1. `G:\Postdoc_research\Mendeley_articles\Advances in modeling learning and decision-making in neuroscience.pdf`
2. `G:\Postdoc_research\Mendeley_articles\Flexible and efficient simulation-based inference for models of decision-making.pdf`
3. `G:\Postdoc_research\Mendeley_articles\A new theoretical framework jointly explains behavioral and neural variability across subjects performing flexible decision-making.pdf`
4. `G:\Postdoc_research\Mendeley_articles\A decision-space model explains context-specific decision-making.pdf`
5. `G:\Postdoc_research\Mendeley_articles\Quantifying decision-making in dynamic, continuously evolving environments.pdf`
6. `G:\Postdoc_research\Mendeley_articles\Choice history biases subsequent evidence accumulation.pdf`
7. `G:\Postdoc_research\Mendeley_articles\pDayan\Low dimensional latent structure underlying the choices of mice.pdf`
8. `G:\Postdoc_research\Mendeley_articles\Initial conditions combine with sensory evidence to induce decision-related dynamics in premotor cortex.pdf`

Synthesis file: `synthesis-batch04-modeling.md`

### Batch 5 — Confidence, lapses, exploration, biases (~8)

1. `G:\Postdoc_research\Mendeley_articles\Neural correlates, computation and behavioural impact of decision confidence.pdf`
2. `G:\Postdoc_research\Mendeley_articles\Lapses in perceptual decisions reflect exploration.pdf`
3. `G:\Postdoc_research\Mendeley_articles\Reinforcement biases subsequent perceptual decisions when confidence is low, a widespread behavioral phenomenon.pdf`
4. `G:\Postdoc_research\Mendeley_articles\Strategically managing learning during perceptual decision making.pdf`
5. `G:\Postdoc_research\Mendeley_articles\The impact of learning on perceptual decisions and its implication for speed-accuracy tradeoffs.pdf`
6. `G:\Postdoc_research\Mendeley_articles\Perceptual decisions and oculomotor responses rely on temporally distinct streams of evidence.pdf`
7. `G:\Postdoc_research\Mendeley_articles\Pupil diameter encodes the idiosyncratic, cognitive complexity of belief updating.pdf`
8. `G:\Postdoc_research\Mendeley_articles\Conversion of sensory signals into perceptual decisions.pdf`

Synthesis file: `synthesis-batch05-confidence-lapses.md`

### Batch 6 — Brain-wide & population dynamics (~9)

1. `G:\Postdoc_research\Mendeley_articles\Distributed coding of choice, action and engagement across the mouse brain.pdf`
2. `G:\Postdoc_research\Mendeley_articles\pDayan\A brain-wide map of neural activity during complex behaviour.pdf` — IBL
3. `G:\Postdoc_research\Mendeley_articles\A reservoir of foraging decision variables in the mouse brain.pdf`
4. `G:\Postdoc_research\Mendeley_articles\Brain-wide interactions between neural circuits.pdf`
5. `G:\Postdoc_research\Mendeley_articles\Neural dynamics outside task-coding dimensions drive decision trajectories through transient amplification.pdf`
6. `G:\Postdoc_research\Mendeley_articles\A frontal motor circuit for economic decisions and actions.pdf`
7. `G:\Postdoc_research\Mendeley_articles\Flow of Cortical Activity Underlying a Tactile Decision in Mice.pdf`
8. `G:\Postdoc_research\Mendeley_articles\Neural Activity in Macaque Parietal Cortex Reflects Temporal Integration of Visual Motion Signals during Perceptual Decision Making.pdf` — Roitman & Shadlen (historical anchor)
9. `G:\Postdoc_research\Mendeley_articles\A neural mechanism for terminating decisions.pdf`

Synthesis file: `synthesis-batch06-brainwide-population.md`

### Sweep batch (optional, if needed)

Remaining decision-titled papers that didn't fit cleanly into batches 1-6. Read only if Phase 1 needs closure.

- `G:\Postdoc_research\Mendeley_articles\The theory of decision making.pdf` (likely Edwards 1954, historical)
- `G:\Postdoc_research\Mendeley_articles\Mice exhibit stochastic and efficient action switching duringprobabilistic decision making.pdf`
- `G:\Postdoc_research\Mendeley_articles\Probing perceptual decisions in rodents.pdf`
- `G:\Postdoc_research\Mendeley_articles\Seeing at a glance, smelling in a whiff- rapid forms of perceptual decision making.pdf`
- `G:\Postdoc_research\Mendeley_articles\The mechanistic foundation of Weber's law.pdf`
- `G:\Postdoc_research\Mendeley_articles\Using temperature to analyze the neural basis of a time-based decision.pdf`
- `G:\Postdoc_research\Mendeley_articles\Value dynamics affect choice preparation during decision-making.pdf`
- `G:\Postdoc_research\Mendeley_articles\Competitive integration of time and reward explains value-sensitive foraging decisions and frontal cortex ramping dynamics.pdf`
- `G:\Postdoc_research\Mendeley_articles\Decision-making in sensorimotor control.pdf`
- `G:\Postdoc_research\Mendeley_articles\Neuronal Correlates of a Perceptual Decision in Ventral Premotor Cortex.pdf`
- `G:\Postdoc_research\Mendeley_articles\Rats and humans can optimally accumulate evidence for decision-making.pdf`
- `G:\Postdoc_research\Mendeley_articles\Pyramidal cell types drive functionally distinct cortical activity patterns during decision-making.pdf`
- `G:\Postdoc_research\Mendeley_articles\Stop Signals Provide Cross Inhibition in Collective Decision-Making by Honeybee Swarms.pdf` (skip unless time)

Synthesis file: `synthesis-batch07-sweep.md` (only if executed)

---

## Phase 2 — Methods batch

One entry per chapter, written as a **methods reference** ("what is this useful for, what's the canonical approach, what would I cite this for"), not paper-style summary.

1. `G:\Postdoc_research\Mendeley_articles\data_science\Chapter-3---Wrangling-Spike-Trains_2017_Neural-Data-Science.pdf`
2. `G:\Postdoc_research\Mendeley_articles\data_science\Chapter-4---Correlating-Spike-Trains_2017_Neural-Data-Science.pdf`
3. `G:\Postdoc_research\Mendeley_articles\data_science\Chapter-7---Regression_2017_Neural-Data-Science.pdf`
4. `G:\Postdoc_research\Mendeley_articles\data_science\Chapter-8---Dimensionality-Reduction_2017_Neural-Data-Science.pdf`
5. `G:\Postdoc_research\Mendeley_articles\data_science\Chapter-9---Classification-and-Clustering_2017_Neural-Data-Science.pdf`
6. `G:\Postdoc_research\Mendeley_articles\data_science\Appendix-B---Frequently-Made-Mistakes_2017_Neural-Data-Science.pdf`

Naming convention for these entries: `methods-nds-ch<NN>-<topic>.md` (e.g., `methods-nds-ch07-regression.md`).

Synthesis file: `synthesis-methods-nds.md` — organized by question type: "if you want to test X, look at chapter Y."

---

## Per-execution-chat workflow

Each future chat that executes a batch does the following:

1. **Read this spec first** at `docs/superpowers/specs/2026-05-28-literature-reading-plan-design.md`.
2. **Identify the assigned batch** (user specifies which batch number).
3. **Verify all paper paths exist** before starting. Report missing files to user; do not silently skip.
4. **Read each paper** in turn, using page ranges for PDFs >20 pages. Suggested page priority: abstract → discussion → key figures → methods (skim) → intro for context.
5. **Write one memory file per paper** under `memory/literature/` following the schema above. ~150 words target (~250 for Orsolic in batch 1).
6. **Write the batch synthesis file** after all papers in the batch are done.
7. **Update `MEMORY.md`** under a `## Literature` heading. Add a single line per synthesis file. Do not add individual papers to the index.
8. **Report back** to the user with:
   - Papers read (count + brief list)
   - Top 3 surprises / connections to BG_046 priorities
   - Suggested adjustments to subsequent batches (papers to move/swap/add/drop based on what was learned)
   - Any papers that were unreadable / missing

A batch is "done" only when all three artifacts exist: per-paper files, batch synthesis, MEMORY.md updated.

---

## Naming conventions

- **Per-paper files:** `paper-<firstauthor-lastname>-<year>-<short-topic>.md`
  - Example: `paper-gold-shadlen-2007-decision-window.md`
  - Use only ASCII lowercase, hyphens, no spaces
  - If author unknown until read, use placeholder name and rename after reading
- **Synthesis files:** `synthesis-batch<NN>-<theme>.md`
- **Methods chapters:** `methods-nds-ch<NN>-<topic>.md`

---

## Stopping criteria & checkpoints

- After **Batch 1**: Quick user check-in. Are the entries useful? Adjust schema / depth / synthesis style based on feedback before continuing.
- After **Batch 3**: Mid-Phase-1 review. Decide whether to keep batch boundaries or merge/split.
- After **Phase 1 sweep batch**: Decide whether to do Phase 2 immediately or pause.
- **Anytime**: User can pause, redirect, or kill a batch.

---

## Open questions (not blocking)

1. Should papers also get tagged with a "relevance to BG_046" rating (high/medium/low) so future searches can prioritize? Decide after batch 1.
2. Do we want a top-level `memory/literature/INDEX.md` summarizing the full library by theme, or is MEMORY.md + glob enough? Decide after batch 3.
3. Phase 3 scope (adjacent literature): defer planning until Phase 1 complete.

## Explicitly deferred (Phase 3 candidates)

Noted here so they aren't forgotten when Phase 3 is planned. Not read in Phases 1-2:

- **pDayan theoretical core:** `Theoretical neuroscience.pdf` (Dayan & Abbott — textbook, plan as chapters not single entry), `Information processing with population codes.pdf`, `The Helmholtz Machine.pdf`, `Doubly distributional population codes.pdf`, `Choice values.pdf`, `Structure in the space of value functions.pdf`, `Q-learning.pdf`, `Bayesian reinforcement learning- A basic overview.pdf`, `Efficient Bayes-Adaptive Reinforcement Learning.pdf`, `Uncertainty and learning.pdf`, `Uncertainty, Neuromodulation, and Attention.pdf`, `Goals and Habits in the Brain.pdf`, `Goal-directed control and its antipodes.pdf`, `How to set the switches on this thing.pdf`, `Space and time in visual context.pdf`.
- **Striatum/BG anatomy & physiology** (top-level + `striatum/` + `NAc/` folders): connectomics, dopamine, cell-type specificity, opioid modulation.
- **Mouse anatomy reference:** *The Mouse Nervous System* chapters (treat like methods — one entry per relevant chapter on demand, not bulk reading).
- **Adjacent behavior/ethology:** `Natural behavior is the language of the brain.pdf`, `Big behavioral data...pdf`, `Spontaneous behaviors drive multidimensional, brainwide activity.pdf`.
- **Cell-type / circuit papers** not directly about decisions but relevant to BG_046 unit classification.

---

## Acceptance criteria

- All 7 phase-1 batches plus 1 methods batch produce the expected files.
- `MEMORY.md` has a `## Literature` section with ~8 synthesis links.
- A future Claude session, starting cold, can answer questions like "what does Orsolic 2019 say about temporal expectation?" by loading the relevant memory file — without reading the original PDF.
- The user can ask "what's the strongest paper on direct/indirect pathway involvement in perceptual decisions?" and get a coherent answer drawing on multiple memory files.
