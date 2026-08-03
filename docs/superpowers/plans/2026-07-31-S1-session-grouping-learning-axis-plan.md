# S1 — Session grouping and a non-circular learning axis (plan)

**Spec:** [2026-07-31-S1-session-grouping-learning-axis-design.md](../specs/2026-07-31-S1-session-grouping-learning-axis-design.md)
**Date:** 2026-07-31 · **Status:** Phase 1 tooling built + smoke-tested; awaiting manual labelling

Phases are independently valuable and gated — each ends in a decision that can stop the rest.

---

## Phase 1 — Manual session sorting (BG_046) ✅ tooling built

**Goal:** capture the expert eye's session grouping, blinded, with a reliability ceiling.

1. ✅ `scripts/session_sorting/run_session_sorter.py` — blinded (stage/date/id/d′ hidden), fixed
   random order, ~15% silent repeats, autosave (atomic upsert keyed on `presentation_idx`), resume.
   Reads only the per-session tag CSVs (no pkl load → instant navigation).
   Panels: outcome raster · state strip · rolling hit-rate + early-lick · blinded stats/keys.
2. ✅ Queue built: **45 sessions + 7 repeats = 52 presentations**
   (`_tag_summary.csv` correctly excluded — it is a roll-up, not a session).
3. ⬜ **USER ACTION: label all 52 presentations** (~15 min).
   `py scripts/session_sorting/run_session_sorter.py --subject BG_046`
   Keys `1-5` group · `u` unsure · `←/→` navigate · `n` note · `q` quit+save.

**Gate:** if any group ends with < 3 sessions, revise the taxonomy *before* fitting.

---

## Phase 2 — Learn the rule + validate 🔜

1. ⬜ `py scripts/session_sorting/fit_session_group_rule.py --subject BG_046` (✅ built, smoke-tested).
   Reports, in order: **test–retest κ** (the ceiling) → **LOSO κ** of a depth-3 tree → the rules
   and the features carrying them. Session-level features are computed from tag CSVs: occupancy and
   normalised max-run per state, switch rate, early-lick fraction, go-trial hit rate, abort
   fraction, mean state confidence, and **first-vs-second-half contrasts** (the only features that
   can express `Deteriorating`).

**Gates:**
- **Test–retest κ < ~0.5** → labels too noisy; fix the taxonomy and relabel. Do not fit.
- **LOSO κ good** (precedent: 0.731) → we have a rule that generalises → Phase 3.
- **LOSO κ poor but test–retest good** → the eye is using something the features miss. That is
  itself the finding; inspect the confusion matrix to see *which* groups fail and add the missing
  feature (likely a dynamics/temporal-structure one).

---

## Phase 3 — Apply and extend ⬜

1. ⬜ Apply the fitted rule to **all** BG_046 sessions incl. `Excluded`, and to BG_039 / BG_031
   (already tagged: 30/32 and 42/42).
2. ⬜ Sanity check, **not** a validation: how do the learned groups relate to stage and d′?
   Because labelling was blinded, a relationship here is informative rather than tautological.
3. ⬜ Report per-mouse. BG_031 (VMS impulsive non-learner) is expected to differ — that is the
   negative control.

---

## Phase 4 — Rebuild the learning axis ⬜

1. ⬜ **Drop the `d′ ≥ 0.8` gate** for learning-axis work; recover the ~12 `Excluded` early
   sessions per mouse. Keep only DV-independent gates (`n_trials`, data integrity).
2. ⬜ Axis = exogenous training time (session index / cumulative trials / days since start).
3. ⬜ Optional latent **monotone competence** curve (state-space binomial or isotonic envelope);
   per-session deviation below the envelope = the state-quality residual.
4. ⬜ Eligibility filter = the Phase-2 rule, applied uniformly at every stage.

**Gate:** if the recovered `Excluded` sessions do **not** extend the early end of the axis
(i.e. they are excluded for data-integrity reasons rather than poor performance), say so and drop
the "Naive isn't naive" motivation.

---

## Phase 5 — Re-run the DVs against the new axis ⬜

1. ⬜ Re-run early-lick rate and the FA-lick hazard against training time with
   `DV ~ training_time + occupancy + (1|session)`, on eligible sessions.
2. ⬜ Compare to the current d′-staged result (BG_046 4–6 s FA hazard 0.019→0.007, session-level
   MWU p=0.028): does the effect strengthen, hold, or vanish once "Naive" is genuinely naive?
3. ⬜ **Expansion curve, not a best cut:** plot effect size + session-clustered CI against how many
   sessions are included, widening from most- to least-eligible. Report the whole curve — stopping
   at the clearest point would be p-hacking.

**Gate:** any headline claim goes through the `harden-result` battery before it leaves the repo.

---

## Standing constraints

- **Circularity scope:** behaviour-derived groups are clean for NEURAL DVs, circular for
  behavioural ones. State it wherever the groups are used.
- **Session = replication unit.** Naive is only 3 sessions/mouse; use session-clustered bootstraps,
  never trial-level CIs.
- **Never re-introduce d′** as a gate or an axis in this workstream.
- Reuse, don't reinvent: `state_labeling.py` (raster/strip render, episode IO),
  `state_calibration.py` (tree + LOSO κ pattern), `behavior.compute_session_performance`.
- Canonicalise session ids via `config.canonical_session_id` / `session_date_key`
  (BG_031/BG_039 use 6-digit DDMMYY).
