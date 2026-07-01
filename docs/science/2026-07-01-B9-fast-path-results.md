# B9 fast-path results — does the striatal baseline TF code sharpen with learning at matched state?

**Date:** 2026-07-01 · **Subjects:** BG_039 (DMS), BG_031 (VMS) · **Status:** fast-path done (start-small; strength hypothesis NULL, recruitment side-finding SUGGESTIVE)
**Spec:** `docs/superpowers/specs/2026-07-01-B9-state-matched-baseline-tf-encoding-learning-design.md` · **Plan:** `.../plans/2026-07-01-B9-...-plan.md`
**Code:** `src/visdetect/analysis/state_tf_learning.py`, `scripts/state_tf_learning/{b9_phase0_profile,b9_deliverable2,b9_recruitment}.py`
**Figures:** `FIGURES/state_tf_learning/{BG_039,BG_031}/b9_preliminary_trend.png`, `.../{subj}/b9_deliverable2_state_conditioned*.png`, `FIGURES/state_tf_learning/b9_recruitment_fraction.png`

## The question
In a matched behavioral state (StimSens = engaged), does the **fidelity with which TF-responsive striatal units encode the baseline grating** (`c1_r`, the registry's fast-minus-slow pulse-PETH correlation) **increase from early to late in learning** — specifically in TF-responsive units, with non-responsive units as the specificity null? Method: **reuse the registry TF-GLM unchanged** (`session_trial_regressors → assemble_design → fit_poisson_cv → identify_tf_responsive_pulse`), changing only the trial subset (state × stage).

## 1. Result — the encoding-STRENGTH hypothesis is NULL
- **Faithfulness gate PASSES exactly** on both subjects: B9's whole-session re-run reproduces the registry `c1_r_log2` with **median |Δ| = 0.0000** (BG_039 n=9, BG_031 n=25). The reuse is byte-faithful, so the numbers are trustworthy.
- **BG_031 (well-powered, VMS), trial-matched (N=140 StimSens, 3 draws): early(Learning) `c1_r` median 0.080 vs late(Expert) 0.080 — identical, MWU p=0.67.** Un-matched cross-check: 0.108 vs 0.128, p=0.46 (also null). Non-responsive control ≈ 0.
- **BG_039 (DMS): no effect once trial count is controlled.** The un-matched run showed a *spurious* early 0.04 << late 0.18 (p=0.005), which is a **trial-count artifact**: `c1_r` correlates with StimSens-trial-count at Spearman ρ=0.67 (p=2e-4), and the "early low" was driven entirely by one 81-trial session (`09042025`, `c1_r`→−0.03); the other early session (`02042025`, 453 trials) sat at 0.166, indistinguishable from the late sessions. BG_039's naive end is too thin to test.

**Verdict:** no evidence that baseline-TF encoding *strength* among responsive units sharpens with learning at matched state, in the well-powered **Learning→Expert** range. The registry preliminary agrees — the whole-session `c1_r` is essentially flat Learning→Expert (BG_031 0.279→0.286; BG_039 flat).

## 2. Methods cautionary tale (reusable)
> **`c1_r` (a pulse-PETH correlation) attenuates with trial count**, so *any* across-condition comparison with unequal trial counts is confounded. A raw BG_039 contrast read "sharpens with learning, p=0.005" purely because the early group happened to include a low-trial session. **Fix = trial-count matching** (subsample each session to a common StimSens-N, average over draws) — the spec §5 matching battery. The faithfulness spot-check + a `c1_r`-vs-`n_trials` diagnostic caught the false positive; without them it would have been reported.

## 3. Side-finding — RECRUITMENT (suggestive, under-powered)
The interesting signal is not per-cell strength but **how many cells are TF-responsive**. Yield-controlled (median units/session ~166–181, flat across stages), the **responsive fraction roughly doubles Naive→Learning, then plateaus**, in *both* subjects:

| stage | BG_031 frac (n sess) | BG_039 frac (n sess) |
|---|---:|---:|
| Naive | 0.033 (3) | 0.023 (2) |
| Learning | 0.066 (11) | 0.061 (1) |
| Expert | 0.064 (16) | 0.035 (18) |

BG_031 Naive-vs-Learning+Expert MWU **p=0.086** (Naive n=3 → under-powered); BG_039 too thin (Naive n=2, Learning n=1). So **recruitment is a consistent hint, not a confirmed effect** — it needs more Naive-stage data (or the cohort) to test.

## 4. Scope & limits
- **Fast-path, start-small:** hand-/auto-picked responsive-rich sessions, not the full cohort; StimSens only; responsive + a non-responsive sample (not all units); Learning→Expert well-powered, **Naive→Learning under-powered** (Naive is responsive-poor everywhere — itself the recruitment observation).
- **BG_031 is VMS** (kept separate from the DMS pool {046,039}); BG_039 is DMS but naive-thin/compressed-learning. No cross-subject pooling (`region_bank_confirmed=False`).
- **June engagement breakdown** in BG_039 is behavioral (StimSens→Disengaged), not an encoding change.

## 5. Next
- **Reframe B9 toward recruitment** (fraction TF-responsive vs learning), which is where both subjects hint — and state-condition + properly power the **Naive→Learning** step (the plateau after is confirmed flat).
- **Cohort / BG_046** (its TF registry pending) to power the Naive end and give a hierarchical estimate.
- If chasing strength further: a **Naive-vs-Expert** state-conditioned contrast (thin: ~8 Naive responsive units) to formally test the step, with the trial-matched runner.
