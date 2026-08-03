# Early-lick (anticipatory) behaviour across learning — results

**Date:** 2026-08-03 · **Subjects:** BG_046 & BG_039 (DMS), BG_031 (VMS)
**Status:** DONE — adversarially verified (5 independent Opus lenses, all reproduced the hazards
bit-for-bit; verdict **PARTIAL**: core result holds, one component of my initial reading was
**refuted and is corrected below**).

---

## 1. Question

Does the mouse learn to **suppress anticipatory (early / `fa`-label) licking** as it becomes expert,
and does the *timing* of those licks change with training?

**Metric provenance (critical):** "early-lick rate" = `fraction_fa` = `n_fa / n_trials`, the
anticipatory `fa` behavioural label. This is **NOT** `sdt_fa_rate` (the SDT false-alarm rate =
licking on catch trials), which is what the manifest's `fa_rate` column holds and what most
pre-existing "FA rate across learning" figures actually plot. The two are distinct constructs.

---

## 2. Result 1 — early-lick rate declines with learning (DMS), trend-level

| Subject | Region | ρ (rate vs session) | p (iid) | p (block-perm) | Naive → Expert | session-level MWU |
|---|---|---|---|---|---|---|
| BG_046 | DMS | −0.40 | 0.018 | **0.055** | 0.58 → 0.29 | — |
| BG_039 | DMS | −0.54 | 0.009 | 0.243 | 0.28 → 0.16 | — |
| BG_031 | VMS | −0.09 | 0.617 | 0.462 | 0.44 → 0.43 | — |

Sessions within a mouse are autocorrelated (lag-1 residual ≈ 0.30), so the **block-permutation p
(circular-shift, autocorrelation-preserving) is the honest one** — the BG_046 decline is a *trend*,
not a significant single-mouse effect. Direction is consistent in both DMS mice; **BG_031 (VMS
impulsive non-learner) is flat**, reaching Expert d′ without suppressing impulsivity — behaving as
the intended negative control. ⚠ It is also the only VMS mouse, so learner-status and region cannot
be separated.

**Not circular:** learning stage is assigned from d′ only (`stage_sessions.py`), and d′ uses the
*SDT* fa_rate, not the early-lick rate.

---

## 3. Result 2 — the RT "bimodality" is largely structural, not learned timing

Initial reading was that Expert early-lick RTs are bimodal with a "self-timed" slow mode. Corrected:

- **A change is never presented before ~6 s** (`change_time` 5th pct = 6.05 s, median ≈ 6.9 s, in
  *both* Naive and Expert). An FA only exists if it precedes its own trial's change, so **FA RTs are
  right-censored per trial** and pile up toward the change. The slow mode is task-imposed, not
  learned anticipation. Panels relabelled *premature* vs *late / pre-change*.
- **Bimodality is transform- and stage-dependent.** Silverman p(unimodal): pooled Expert 0.006
  (linear) / 0.028 (log10); **per stage on linear RT: Naive 0.23 (unimodal), Learning 0.06, Expert
  0.007** — the two-mode separation *sharpens* with learning; Naive is not robustly bimodal.
- **The antimode is transform-dependent** — ≈1.5 s (log KDE) vs ≈2.3 s (linear KDE). Report a
  **band (1.5–2.3 s)**, never a precise value. Do **not** cite GMM ΔBIC (inflated by right skew);
  use `spectrum_stats.silverman_bootstrap`.
- Median early-lick RT ≈ 4.6 s; the fast premature mode is a **minority** (~1/3 of mass). Only ~2%
  of early licks are < 0.2 s, so the fast mode is not a trial-boundary artifact.

---

## 4. Result 3 — pre-change anticipatory licking is suppressed (the hardened core)

FA-lick hazard = P(early lick | trial still in baseline at *t*), via the canonical
`decision_latents.fa_lick_hazard` (non-FA trials censored at `min(change_time, decision_time)`).

**Holds (BG_046, DMS):**
- Absolute pre-change (4–6 s) FA hazard **halves**: Naive 0.019 → Expert 0.007.
- Depletion-free 4–6 s FA-lick **fraction** 0.363 → 0.278 (bootstrap CI excludes 0).
- **Session-clustered** (session = replicate) MWU Naive vs Expert **p = 0.028**, 90% pairwise dominance.
- Not an at-risk artifact (883–1341 trials at risk through 4–6 s); robust to including/excluding aborts.
- **Not a Hit-reclassification artifact**: before 6 s the FA hazard and the all-first-lick hazard are
  identical byte-for-byte (max|diff| = 0) — any pre-change first lick *is* an FA by definition.

**Per-mouse:** BG_039 same direction but **not significant** at session level (2 usable Naive
sessions, MWU p = 0.87/0.06 depending on metric). BG_031 (VMS) **reverses** — ramp grows 1.27 → 2.12,
4–6 s fraction 0.29 → 0.34, session-level p = 0.047.

### ❌ Corrected overclaim
My initial reading — "a stimulus-locked detection peak *develops* after 6 s" — is **refuted**. In the
hazard, the post-6 s peak is **largest in Naive** (BG_046 hit-hazard peak Naive 0.058 vs Expert
0.029), because the hazard conditions on survival and few Naive trials survive to the change. The
genuine shift toward detection is a **count** effect (FA-fraction of first-licks 0.73 → 0.51), not a
hazard-shape change. Do not claim the post-6 s hazard peak grows with learning.

Likewise, "anticipatory timing does not develop" is too strong: Naive's ramp toward 6 s *is* evidence
of temporal expectation. The defensible claim is **learned suppression of overt premature licking**.

---

## 5. Method lessons (reusable)

1. Plot heavy-tailed RTs as **linear density on a linear axis** (area = mass) or an ECDF — never
   linear density on a log axis (that is what made the minority fast mode look dominant).
2. **Check the task's censoring structure before any "anticipation" claim** — apparent modes can be
   imposed by trial timing.
3. Sessions within a mouse are autocorrelated → report **block-permutation** p, not the iid Spearman p.
4. The late/early hazard **"ramp index" is inflated ~1.2–1.5× by survival censoring in every stage**
   (a uniform-hazard null gives a *rising* curve) → headline the **absolute** pre-change hazard or a
   denominator-free fraction.
5. Post-change hazard peaks are **survivorship-conditioned**; to claim stimulus-locking, realign to
   `Change_ON` rather than reading the Baseline_ON hazard.
6. The 0–0.7 s hazard bump **grows with learning** (likely ITI/consummatory carryover) → it
   contaminates any ratio using an early-window denominator.

---

## 6. Caveats that must travel

- Single mouse per direction: DMS n=2 (effect carried by BG_046), VMS n=1. **Region vs individual
  cannot be separated.**
- **"Naive" is not naive** — the d′ ≥ 0.8 QC gate means BG_046's Naive sessions have d′ 0.98–1.45 and
  number only 3; genuinely naive sessions sit in `Excluded`. See
  [S1 spec](../superpowers/specs/2026-07-31-S1-session-grouping-learning-axis-design.md) — Phase 5
  will re-run these DVs against a non-circular axis.
- BG_039 Learning is a single session (n_fa = 28) — uninterpretable, excluded.
- Aborts rise with learning (BG_046 9% → 25%) — a real co-occurring behavioural change not captured
  by the FA hazard.

---

## 7. Artefacts

| Item | Path |
|---|---|
| Trajectory figure (8 panels) | `FIGURES/behavior/BG_046/early_lick_learning_trajectory.png` |
| Cross-subject replication | `FIGURES/behavior/replication/early_lick_replication_3mice.png` |
| FA hazard ± aborts | `FIGURES/behavior/fa_hazard/fa_lick_hazard_learning.png` |
| FA vs all-lick hazard | `FIGURES/behavior/fa_hazard/fa_vs_alllick_hazard_learning.png` |
| Scripts | `scripts/analysis/behavior/early_lick_{learning_trajectory,replication}.py`, `fa_lick_hazard_learning.py` |
| Caches + caveats sidecar | `data/cache/behavior/early_lick_*.csv`, `early_lick_learning_CAVEATS.txt`, `fa_*_summary.csv`, `fa_hazard_session_level.csv` |

**Verification record:** adversarially verified 2026-07-30, 5 lenses (independent reproduction,
at-risk artifact, alternative explanation, per-mouse/pseudoreplication, completeness critic).
All 5 reproduced the hazards exactly (max|diff| = 0). Verdict **PARTIAL** — core suppression result
survives; the post-6 s "detection develops" component was refuted and is corrected in §4.
