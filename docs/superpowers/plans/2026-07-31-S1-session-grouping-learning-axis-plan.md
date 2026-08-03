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

## Phase 4 — Rebuild the learning axis ⬜ (SCALED BACK 2026-08-03)

> **The Phase-4 gate has fired.** Manual labelling (Phase 1–2) resolved what the `Excluded`
> sessions actually are, and it **partly refutes the original motivation**. Of BG_046's 12
> `Excluded` sessions: **6 Disengaged-dominated, 2 Low-yield, 2 Deteriorating, 1
> Impulsive-dominated, 0 Balanced.** The d′ gate is mostly removing **bad-state** sessions, not
> naive-but-engaged ones. The only genuinely earliest sessions (chronological positions 0–1) are
> the 2 Low-yield ones — too sparse to judge.
>
> **Revised reading:** "Naive isn't naive" is more a **data limitation** than a gating artifact.
> Ungating will not manufacture a rich naive sample for BG_046. Do not expect Phase 4 to fix the
> 3-Naive-session problem.

1. ⬜ **Still drop the `d′ ≥ 0.8` gate** for learning-axis work — not to recover naive data (it
   isn't there), but because gating on the outcome is wrong in principle and the 2 Low-yield +
   1 Impulsive `Excluded` sessions do sit at the early edge. Report the recovered n honestly.
2. ⬜ Axis = exogenous training time (session index / cumulative trials / days since start).
3. ⬜ **Downgraded to optional/deferred:** the latent monotone competence curve. With only ~3
   usable Naive sessions the early end of the curve is unconstrained, so a state-space fit would
   be extrapolating. Revisit only if other subjects supply a denser early sample.
4. ⬜ Eligibility filter = the Phase-2 rule, applied uniformly at every stage.
5. ⬜ **NEW (user proposal, 2026-08-03) — epoch-level salvage.** Rather than discarding
   `Deteriorating` sessions whole, use their **engaged leading epoch**. Justified: labelling
   confirmed these sessions genuinely decline within-session (Δ hit-rate 2nd−1st = −0.095, 5/5
   negative, MWU p=0.0018 vs Balanced; Δ StimSens occupancy −0.208, p=0.0099), so an early epoch
   is qualitatively different from the late one. This moves eligibility from **session-level** to
   **epoch-level**, which matches the original diagnosis that contamination is within-session.
   Mandatory controls:
   - **Trial-count matching** — a salvaged 150-trial epoch vs a full 550-trial Balanced session
     will bias any encoding-strength / fit-quality metric downward (the B9 attenuation lesson).
     Subsample to matched N and pair within session; report matched and unmatched.
   - **Non-circular epoch selection** — pick the epoch by a criterion independent of the DV
     (e.g. state occupancy or a change-point on participation), never by the DV itself.
   - **Report the salvaged n** per session; if a salvaged epoch is under the power floor, drop it
     rather than reporting a point estimate.

**Gate (revised):** if trial-count matching cannot be achieved between salvaged epochs and full
sessions, the comparison is **under-powered and not reportable** — say so, do not report the point
estimate as a trend.

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
