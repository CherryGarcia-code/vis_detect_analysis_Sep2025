# B9 fast-path results — does the striatal baseline TF code sharpen with learning at matched state?

**Date:** 2026-07-01 · **Subjects:** BG_031 (VMS), BG_039 + BG_046 (DMS) · **Status:** done — learning-sharpening NULL (engaged & disengaged); recruitment NOT robust; engagement-gating essentially ABSENT (paired matched-N Δ=0.017, p=0.21). Net: the striatal baseline-TF code is a **stable, ~engagement-independent bottom-up signal, unchanged by learning**.
**Spec:** `docs/superpowers/specs/2026-07-01-B9-state-matched-baseline-tf-encoding-learning-design.md` · **Plan:** `.../plans/2026-07-01-B9-...-plan.md`
**Code:** `src/visdetect/analysis/state_tf_learning.py`, `scripts/state_tf_learning/{b9_phase0_profile,b9_deliverable2,b9_recruitment}.py`
**Figures:** `FIGURES/state_tf_learning/{BG_039,BG_031}/b9_preliminary_trend.png`, `.../{subj}/b9_deliverable2_state_conditioned*.png`, `FIGURES/state_tf_learning/b9_recruitment_fraction.png`

## The question
In a matched behavioral state (StimSens = engaged), does the **fidelity with which TF-responsive striatal units encode the baseline grating** (`c1_r`, the registry's fast-minus-slow pulse-PETH correlation) **increase from early to late in learning** — specifically in TF-responsive units, with non-responsive units as the specificity null? Method: **reuse the registry TF-GLM unchanged** (`session_trial_regressors → assemble_design → fit_poisson_cv → identify_tf_responsive_pulse`), changing only the trial subset (state × stage).

## 1. Result — the encoding-STRENGTH hypothesis is NULL
- **Faithfulness gate PASSES exactly** on both subjects: B9's whole-session re-run reproduces the registry `c1_r_log2` with **median |Δ| = 0.0000** (BG_039 n=9, BG_031 n=25). The reuse is byte-faithful, so the numbers are trustworthy.
- **BG_031 (well-powered, VMS), trial-matched (N=140 StimSens, 3 draws): early(Learning) `c1_r` median 0.080 vs late(Expert) 0.080 — identical, MWU p=0.67.** Un-matched cross-check: 0.108 vs 0.128, p=0.46 (also null). Non-responsive control ≈ 0.
- **Same null in the DISENGAGED state** (BG_031, trial-matched N=120, 3 draws): early 0.036 vs late 0.037, **MWU p=0.93**. So there is no learning sharpening in *either* engaged or disengaged trials.
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

BG_031 Naive-vs-Learning+Expert MWU **p=0.086** (Naive n=3 → under-powered); BG_039 too thin (Naive n=2, Learning n=1). So **recruitment is a consistent hint, not a confirmed effect**.

**Update — NOT robust.** Date-staging **all 42 sessions** (so the 11 QC-Excluded, Disengaged-heavy sessions are included; they date-land in Expert) **flattens it**: BG_031 Naive 0.034 / Learning 0.066 / Expert **0.037**, p=0.27. The Excluded sessions carry *diluted* whole-session responsiveness (`resp_log2` is a whole-session call, pulled down by their many disengaged trials), so the earlier "recruitment" partly reflected manifest cherry-picking of behaviourally-clean sessions. **Verdict: no robust recruitment effect either.**

## 3b. Engagement — TF cells encode when disengaged (partial, not gated)
Registry-responsive units DO encode baseline TF in **Disengaged** trials (BG_031, trial-matched N=120): `c1_r ≈ 0.037` vs the ~0.000 non-responsive control. So encoding is **not fully engagement-gated** — answering "are TF cells responsive when disengaged?": **yes, partially**. It looks *weaker* than the engaged StimSens level (0.080), i.e. present-but-attenuated, but the two runs used different N (120 vs 140) and different sessions, so `c1_r`'s trial-count sensitivity confounds the magnitude — a clean engagement test needs a **matched StimSens-vs-Disengaged run (same N, same sessions)**. Consequence: the low whole-session responsiveness in Disengaged-heavy sessions is **at least partly statistical dilution**, not pure gating.

**Resolved — engagement-INDEPENDENT (paired matched-N test; `scripts/state_tf_learning/b9_engagement.py`).** Running StimSens and Disengaged on the **same sessions at the same per-session N** (min of the two, ≤200), paired per unit (16 eligible sessions across BG_031/039/046 — BG_046 is highly engaged so contributes 1), the paired **Δ (StimSens − Disengaged) centres near zero: median 0.017, Wilcoxon p=0.21 (n=128 unit-sessions)**; per subject BG_031 Δ=0.024 p=0.075 (weak, non-sig engaged-gain trend), BG_039 Δ=−0.069, BG_046 n=5. At matched N the earlier apparent gap collapses → **baseline-TF encoding is largely engagement-independent** (cells encode the grating about as well disengaged as engaged). Confirms the whole-session dilution was statistical, and vindicates the "TF cells respond regardless of acting" intuition. Figure: `FIGURES/state_tf_learning/b9_engagement_paired.png`.

## 4. Scope & limits
- **Fast-path, start-small:** hand-/auto-picked responsive-rich sessions, not the full cohort; StimSens only; responsive + a non-responsive sample (not all units); Learning→Expert well-powered, **Naive→Learning under-powered** (Naive is responsive-poor everywhere — itself the recruitment observation).
- **BG_031 is VMS** (kept separate from the DMS pool {046,039}); BG_039 is DMS but naive-thin/compressed-learning. No cross-subject pooling (`region_bank_confirmed=False`).
- **June engagement breakdown** in BG_039 is behavioral (StimSens→Disengaged), not an encoding change.

## 5. Bottom line & next
**Bottom line (n=3 striatal subjects: VMS BG_031, DMS BG_039/046).** The striatal baseline-TF representation is a **stable, ~engagement-independent, bottom-up sensory signal**: its per-unit fidelity (`c1_r`) does **not** sharpen with learning (engaged *or* disengaged), the responsive-fraction "recruitment" is **not robust**, and behavioural engagement does **not** gate it. A well-controlled negative on "learning sharpens the baseline sensory code," plus a clean positive characterisation and a reusable methods lesson (trial-count matching is mandatory for `c1_r`; faithfulness Δ=0.0000 throughout).

**If continued:**
- **Naive-powered learning test** — the only learning-linked hint left is the *number* of TF-responsive cells (Naive-poor everywhere), and it isn't robust; a subject with rich naive-stage yield would be needed to test it properly.
- **Cohort / hierarchical** across DMS{046,039} + VMS{031} (region-pool once `region_bank_confirmed`).
- **Beyond baseline-TF fidelity** — the learning signal may live elsewhere (change-evoked coding, population geometry, D1/D2 push-pull). B9 specifically closes the "baseline sensory-fidelity" door.
