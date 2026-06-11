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

## Index

| ID | Question (short) | Tier | Status | Spec | Plan |
|----|------------------|------|--------|------|------|
| **B2** ⭐ | Does striatal sensory responsiveness *lead* the behavioral learning curve? | T1 | plan-draft | [design](../superpowers/specs/2026-06-08-B2-responsiveness-leads-learning-design.md) | [plan](../superpowers/plans/2026-06-08-B2-responsiveness-leads-learning-plan.md) |
| B0 ⭐ | Which DDM knob does learning turn (drift vs threshold vs starting-point)? | T1 | plan-draft | [design](../superpowers/specs/2026-06-10-B0-ddm-learning-knob-design.md) | [plan](../superpowers/plans/2026-06-10-B0-ddm-learning-knob-plan.md) |
| B1 ⭐ | Is the evidence-integration timescale a *learned* quantity? | T1 | plan-draft | [design](../superpowers/specs/2026-06-10-B1-integration-timescale-learned-design.md) | [plan](../superpowers/plans/2026-06-10-B1-integration-timescale-learned-plan.md) |
| A1 ⭐ | Is the lick-locked population ramp the commitment/bound signal? | T1→T2 | not-started | — | — |
| A2 | Single-trial drift-to-fixed-bound on the population evidence axis | T1 | not-started | — | — |
| A3 | D1 vs D2 asymmetric commitment / pre-response balance shift | T2/T3 | not-started | — | — |
| B3 | Learning = single-cell plasticity vs population re-weighting (needs tracking) | T1→T2 | not-started | — | — |
| B4 | AND-gate emergence: gradual vs phase transition | T1 | not-started | — | — |
| B5 | Does learning re-set the dSPN/iSPN baseline operating point? (photometry) | T2 | not-started | — | — |
| B6 | Divergent DA teaching-signal trajectories (RPE↑ vs APE↓) | T2/T3 | not-started | — | — |
| B7 | Falling learning rate / uncertainty reduction across training | T1→T2 | not-started | — | — |
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

_Last updated: 2026-06-08. Add a row's spec/plan links and bump status as each question is worked._
