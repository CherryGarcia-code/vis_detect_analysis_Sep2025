# Research direction after the TF-responsiveness null (June 17, 2026)

**Status:** brainstorm synthesis (BG / Claude discussion). For meetings, presentations, and as the anchor for the next phase of analysis.
**Context:** a clean-slate TF-responsiveness redesign returned a robust **null on *single-pulse-triggered (instantaneous)* TF responsiveness** — ≈0% of units cross a per-unit threshold across BG_046 DMS, BG_031 striatum, BG_039 cortex, and BG_038 GPe (the off-by-one fix was *proven* correct, so the old "responders" were artifacts; units are decision/action-dominated, 72–83% change-responsive). This document is the redirection that follows.

> **⚠️ Scope of the null (corrected after the 4-region survey):** the null is specifically on *instantaneous single-pulse* responsiveness, which is a **blunt** readout — it under-detects the *integrated* (~250 ms) and *graded* TF code that Khilkevich/Lohse actually measure (a single ±1 SD/50 ms pulse leaves the 250 ms-integrated evidence only ≈+0.2 SD, ~5× diluted). **Four regions including cortex all at ≈0% ⇒ the floor reflects the metric, not the biology.** So **graded/integrated TF coding is *untested*** (needs a per-neuron GLM with a temporal kernel and/or a population decoder). The pivot below stands on its own merits — the robust signal is decision/motor/state, which *is* the project's question — **not** on a claim that "there is no TF coding."

---

## TL;DR (the one-paragraph pitch)

The null frees the project from the borrowed "find TF-evidence cells" framing and returns it to its actual spine: **how impulsivity vs. deliberation is learned.** The thesis the data point to: **learning this task is substantially learning *when* to commit (temporal expectation), and that learned timing sculpts impulsivity** — a learned, time-varying *urgency* signal develops in striatum, early licks migrate toward the expected change time as it does, and the spontaneous **Impulsive state is a transient *de-sculpting*** of that timing (a reversion to an earlier, leakier criterion), **not** a loss of sensitivity. A separable **drift/sensitivity** axis matures in parallel (easy changes first). The novelty is the **integration**: one *criterion/urgency* axis seen at three timescales — slow learning, fast state-switching, within-trial ramp — read out by striatal commitment dynamics, with temporal expectation as the learned structure.

---

## 1. The pivot
TF-responsiveness was a sub-method (Khilkevich/Lohse/Orsolic), not the goal. It is null **in DMS**. The project's documented spine — *how impulsivity/sensitivity/deliberation is learned* (DDM drift/threshold/start-point) — is a decision/action question, which is exactly what these neurons encode (~74% change/motor responsiveness). So this is a redirection to the spine, not a loss.

## 2. Refined thesis (the falsifiable claim)
> **Learning ≈ learning *when* to commit (temporal expectation); learned timing sculpts impulsivity.** A time-varying urgency signal develops in striatum; early (FA) licks migrate toward the expected change time as it does. The **Impulsive state = transient de-sculpting** (reversion to an earlier, leakier urgency/criterion profile), not degraded sensitivity. A **separable drift/sensitivity axis** matures easy-before-hard.

Predicts: the commitment **ramp** sharpens & time-locks to the expected change with learning; reverts (earlier/leakier) in Impulsive bouts; predicts FA timing. Impulsive state shows *more* responses to weak/sub-threshold changes (apparent "hypersensitivity" = liberal criterion) with **d′ flat-or-down**, not a genuine sensitivity gain.

## 3. Structural model
- **Two axes.**
  - **Drift / sensitivity (≈ d′)** — primarily the *learning* axis. Behavioral signatures: easy-changes-mastered-first; RT-variability shrinks (more & earlier for easy).
  - **Criterion / boundary / urgency / timing** — both a *learning* axis (caution rises) **and** the within-session **state** axis.
- **States load on the criterion/urgency axis**, not drift. **Impulsive** = liberal/low-boundary (trigger-happy → looks hypersensitive, but it's criterion). **StimSens** = conservative/high-boundary, waits for real evidence. (Consistent with the existing neural-validation result: engaged StimSens = lower-gain change response + slower RT than trigger-happy Impulsive.)
- **Shared learning↔state axis = the criterion/urgency one.** Expert Impulsive bout ≈ a transient reversion to a *mid-learning* "knows-but-can't-hold" regime.
- **Two impulsivities (caveat):** pre-comprehension (earliest naïve — *exclude/treat separately*) vs. post-comprehension lapse (mid-learning→expert — the thesis target).
- **Temporal expectation is the convergent organizing variable** — appears in FA-migration, hit-clustering at the change-mode, *and* the neural ramp; it is the task's *designed* variable (`dmdmTemporalExpectation`). The ramp that confounded the TF analysis is the *central signal* here.

### Behavioral sculpting over learning (each maps to a knob)
| Observation | Axis / DDM mapping |
|---|---|
| Large changes mastered first, then small | drift ↑, graded by stimulus |
| RT-variability shrinks (more/earlier for easy) | accumulation precision ↑ (drift side) |
| FA density migrates: wide-near-earliest → hump near expected change time | learned **time-varying urgency** (criterion/timing) |
| Correct-hit licks cluster at change-time mode | temporal expectation |
| Persistent striatal ramp | commitment / urgency readout |

## 4. Direction map (priorities)
| Role | Item | Note |
|---|---|---|
| **Spine** | Impulsivity/sensitivity/deliberation learning | the real project |
| **Substrate (now)** | Lick increasers vs decreasers + change/commitment ramp | go/brake functional typing, no optotagging needed |
| **Enabler** | Video → true **motion onset** + **pupil** | fixes RT (the DDM observable); arousal/uncertainty for states |
| **Cheap decisive control — run early** | Batch on **BG_039 (cortex/M2)** | tests whether the TF-null is *regional*; validates pipeline (~10 min, tool exists) |
| **Door-close on TF** | Graded-TF **population decoder** | once, briefly; catches weak distributed coding; low expected yield → makes null airtight |
| **Validation resource** | Khilkevich data on ceph | positive control for the pipeline; *not* a discovery target (their published data; remote-access cost) |
| **Supporting / later** | Optotagging (D1/D2), CCG connectivity, anatomy | optotag yield low (synaptic≠antidromic; await stronger laser); CCG sparse; anatomy = region assignment |

**Cell-typing status:** broad/narrow (SPN/FSI) available now (M2 done) — enough for FSI-gating questions. D1/D2 deferred to stronger-laser experiments.

## 5. Novelty (where it is, and the risk)
- **Ingredients are not novel** (GLM-HMM states, DDM-of-learning, state-gain). Run as 3 parallel descriptive analyses → incremental.
- **Novelty = integration + locus + a falsifiable claim:** one criterion/urgency axis at three timescales, temporal-expectation as the learned structure, striatal commitment dynamics as readout, and the **learning↔state unification** (Impulsive bout ≈ mid-learning regime).
- **Risks:** single-subject generality (mitigate: multi-subject + BG_031 replication); "states obvious by eye" → behavioral states may be known, so the contribution must be the *neural-axis-shared-with-learning* part; stay mechanistic (DDM-parameter mapping + lick push-pull), not "we decoded X."

## 6. Method order
1. **Characterize behavior first — inside the DDM/temporal-expectation frame, decomposed by state** (not re-plotting learning curves). **Deliverable = per-trial latent estimates** (drift, urgency/expected-time, criterion/start-point) to later regress neurons against.
2. **Anchor on expert + good-behavior sessions; backtrack** through learning to see which knob moves most and when (expect a mixture; the point is which dominates).
3. **Then neural:** does the ramp sharpen/time-lock with learning, revert in Impulsive bouts, predict FA timing?

## 7. Open structural questions
- Confirm the two-axis separation neurally: Impulsive = criterion shift with drift/d′ intact?
- Which knob dominates learning (boundary vs. start-point vs. drift)? Behavior largely answers before spikes.
- Is the expert-Impulsive neural state quantitatively closer to *mid-learning* than to session-1 (the shared-axis test)?

---
*Provenance: BG (b.gonzales@ucl.ac.uk) ↔ Claude discussion, 2026-06-17, following the TF-responsiveness null finding. Companion memory: `tf_responsiveness_null_finding_jun2026`, `question_landscape_jun2026`.*
