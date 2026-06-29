# BG_046 Scientific Question Index

Operational index for the per-question spec/plan corpus. The **conceptual** master is the
brainbulb landscape in memory (`memory/question_landscape_jun2026.md`); this file is the
**repo-side** map from each question ID → its design spec → implementation plan → status,
so a fresh chat can boot a single question in isolation.

## Conventions
- **Spec:** `docs/superpowers/specs/YYYY-MM-DD-<ID>-<slug>-design.md`
- **Plan:** `docs/superpowers/plans/YYYY-MM-DD-<ID>-<slug>-plan.md` (from the writing-plans skill)
- **ID** matches the landscape (A1, B0, B2, …). One question = one spec + one plan.
- **To start a question in a new chat:** open its spec (and plan, if present); they are self-contained.
- **Status:** `not-started` → `spec-draft` → `spec-approved` → `plan-draft` → `in-progress` → `done`.
- Tiers: **T1** = data-in-hand now · **T2** = cross-modal / moderate new analysis · **T3** = aspirational (new cohorts / optotagging / manipulations). ⭐ = sharpest-novelty bets.

## Spine
*How do mice LEARN to suppress impulsivity and increase sensitivity to informative stimuli, to drive perceptual decisions?* → decomposable as DDM **drift** (sensitivity) vs **threshold/starting-point** (impulsivity). Every question below is a facet of this.

### Decision-spine program (the B8 through-line)
The behavior-first method order (post-TF-null direction §6) runs as one continuous program, even though its later stages are filed under sibling question IDs:

1. **B8 Phase 1 — descriptive dials (DONE).** Per-(session×mood) sharpness/itchiness/timing over learning; censored hazards; a cached **per-trial descriptive latent table**. Result: bias-not-gain (states load on criterion, not sensitivity), with the circularity caveat (itchiness partly definitional → lean on timing/RT/neural).
2. **B8 Phase 2 — generative latents (DONE, 2026-06-27).** Recovery-gated cloglog hazard-accumulator emitting per-trial **drift(sharpness) / start-point(itchiness) / urgency(timing)**; corrected 60 Hz evidence builder; lapse-aware psychometric metric. **Recovery verdict (full-power cluster, n_trials=800, bootstrap=500; the gate stress-tested an expert-like AND a naive-like regime synthetically — the real fitted deliverable is expert-regime only): caution + timing = generative; sharpness = descriptive (the v↔z ridge — drift is a recoverable trend, not per-trial).** LEARNING ladder: robust single-dial order **sharpness > timing > caution** (AIC→M_full, BIC→M_shared; the all-vs-parsimony call is criterion-sensitive, the ordering is not). STATE ladder: modal M_all (mood loads on all dials; caution partly circular → lean on timing). Engine-C construct check (n=3 expert anchors — an underpowered QUALITATIVE spot-check, NOT a statistical test): all three dials show |Spearman|≈0.5 (ns, p≈0.67); the signs are directionally consistent with the verdict (caution & timing positive vs the full-DDM, sharpness negative) but n=3 affords no significance. Deliverable = the per-trial latent table (39 cols, trust-labeled) the neural phase regresses against.
3. **Neural phase (downstream IDs).** Regress striatal activity against the B8 latents — *this is where the circularity breaks* (a neural correlate of itchiness is labeler-independent). Maps onto:
   - **B0** — which DDM knob learning turns (drift vs threshold vs start-point);
   - **A1 / A2 / E1** — is the lick-locked population ramp the bound; single-trial drift-to-bound; DDM tests on the evidence axis;
   - **D1** — lapses as gain vs bias (state-conditioned psychometrics; absorbs B8's behavioral half).

B8's per-trial latent table is the shared **input** to B0/A1/E1/D1; those rows are "what follows B8," not independent threads.

## Index

| ID | Question (short) | Tier | Status | Spec | Plan |
|----|------------------|------|--------|------|------|
| **B2** ⭐ | Does striatal sensory responsiveness *lead* the behavioral learning curve? | T1 | plan-draft | [design](../superpowers/specs/2026-06-08-B2-responsiveness-leads-learning-design.md) | [plan](../superpowers/plans/2026-06-08-B2-responsiveness-leads-learning-plan.md) |
| B0 ⭐ | Which DDM knob does learning turn (drift vs threshold vs starting-point)? | T1 | in-progress | [design](../superpowers/specs/2026-06-10-B0-ddm-learning-knob-design.md) · [results](2026-06-13-B0-ddm-learning-knob-results.md) | [plan](../superpowers/plans/2026-06-10-B0-ddm-learning-knob-plan.md) |
| B1 ⭐ | Is the evidence-integration timescale a *learned* quantity? | T1 | plan-draft | [design](../superpowers/specs/2026-06-10-B1-integration-timescale-learned-design.md) | [plan](../superpowers/plans/2026-06-10-B1-integration-timescale-learned-plan.md) |
| A1 ⭐ | Is the lick-locked population ramp the commitment/bound signal? | T1→T2 | not-started | — | — |
| A2 | Single-trial drift-to-fixed-bound on the population evidence axis | T1 | not-started | — | — |
| A3 | D1 vs D2 asymmetric commitment / pre-response balance shift | T2/T3 | not-started | — | — |
| B3 | Learning = single-cell plasticity vs population re-weighting (needs tracking) | T1→T2 | not-started | — | — |
| B4 | AND-gate emergence: gradual vs phase transition | T1 | not-started | — | — |
| B5 | Does learning re-set the dSPN/iSPN baseline operating point? (photometry) | T2 | not-started | — | — |
| B6 | Divergent DA teaching-signal trajectories (RPE↑ vs APE↓) | T2/T3 | not-started | — | — |
| B7 | Falling learning rate / uncertainty reduction across training | T1→T2 | not-started | — | — |
| **B8** ⭐ | Per-trial decision-latents (sharpness/itchiness/timing) decomposed by state — behavior-first deliverable for the post-TF-null spine; absorbs D1's behavioral half | T1 | **Phase 1 done; Phase 2 done (2026-06-27)** — recovery-gated: caution+timing generative, sharpness descriptive; per-trial latent table (39 cols, trust-labeled) shipped | [design P1](../superpowers/specs/2026-06-18-B8-behavioral-decision-latents-by-state-design.md) · [design P2](../superpowers/specs/2026-06-20-B8-phase2-generative-latents-design.md) | [plan P1](../superpowers/plans/2026-06-18-B8-behavioral-decision-latents-by-state-plan.md) · [plan P2](../superpowers/plans/2026-06-20-B8-phase2-generative-latents-plan.md) |
| **N1** ⭐ | Neural urgency-ramp: does expert striatal activity carry a pre-decision urgency signal predicting response timing (the B8 timing latent) *beyond motor prep*, and is it cell-type-specific (broad-SPN/narrow-FSI; D1/D2 deferred)? — first neural-phase question; neural arm of A1/E1 + B0 timing facet | T1 | spec-draft | [design](../superpowers/specs/2026-06-29-N1-neural-urgency-ramp-timing-latent-design.md) | — |
| C1 ⭐ | FA = suppression failure; waiting-period D2/indirect (MOs→D2) brake | T1→T2 | not-started | — | — |
| C2 | Same evidence axis vs opposite-sign push-pull (mode-aware; photometry) | T2 | not-started | — | — |
| C3 | Proposal cell-type role table via anatomical (dorsal/ventral) stratification | T3 | not-started | — | — |
| C4 | FSI feedforward inhibition as the impulsivity brake | T1 | not-started | — | — |
| D1 | Are lapses gain or bias? (state-conditioned psychometrics) | T1 | not-started | — | — |
| D2 ⭐wild | Is the Disengaged state offline task reactivation? | T1→T2 | not-started | — | — |
| D3 | Temporal expectation as a recent-action prior (striatal baseline) | T2 | not-started | — | — |
| D4 | Striatal ensemble that predicts/drives HMM state transitions | T1→T2 | not-started | — | — |
| D5 | Slow drift ↔ a slowly-varying neural/engagement axis | T1 | not-started | — | — |
| E1 | DDM tests on the evidence axis (Hit vs Miss diverge; amplitude→RT) | T1 | not-started | — | — |
| E2 ⭐ | Distributed vs specialist evidence coding (decode the non-responsive 90%) | T1 | not-started | — | — |
| E3 | Movement-null vs movement-potent; movement ⊥ sensory orthogonality | T2 | not-started | — | — |
| E4 ⭐ | Quantitative match to the Lohse sister study + emergence across learning | T2 | not-started | — | — |
| F1 | Triangulate the D1/D2 baseline shift: photometry vs ephys | T2 | not-started | — | — |
| F2 | Tracking quality as a scientific gate (plasticity vs matching error) | T1 | not-started | — | — |
| F3 | Which claims survive as n=1 vs need cohort pooling (BG_031/038/039)? | meta | not-started | — | — |

_Last updated: 2026-06-29 (N1 spec-draft added — first neural-phase question: striatal urgency-ramp ↔ B8 timing latent, neural arm of A1/E1. Re-cast finding: the trusted dials are session×mood-level not single-trial — `timing_urgency`≈`decision_time` in φ-coords (binned η²≈0.98), caution z is per-cell constant — so trusted-dial neural tests are across-session graded; the timing question is an urgency-ramp/response-timing-prediction question). Prior: 2026-06-27 B8 Phase 2 DONE. Add a row's spec/plan links and bump status as each question is worked._
