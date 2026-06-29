# Design spec — N1: Neural urgency-ramp & the B8 timing latent (decision-spine, neural phase)

| | |
|---|---|
| **Question ID** | **N1** — first question of the neural-correspondence phase. Realizes the neural side of **A1/E1** (commitment/urgency ramp; evidence-axis) + the **B0** timing facet; filed in `docs/science/QUESTION_INDEX.md` under "Neural phase (downstream IDs)". One question = one spec + one plan. |
| **Date** | 2026-06-29 |
| **Status** | SPEC-DRAFT (brainstormed 2026-06-29; gate/controls hardened across four user refinements; awaiting user spec review → writing-plans) |
| **Feasibility tier** | **T1** — BG_046 expert ephys + the B8 per-trial latents are in hand. **Single-subject (n=1, BG_046)**, architected for a later hierarchical **F3** cohort. |
| **Spine** | The neural arm of the post-TF-null decision-spine (`docs/science/2026-06-17-post-tf-null-research-direction.md`). **This is where the state-label circularity breaks** — a neural correlate is labeler-independent ([[state-labeler-circularity-caveat]]). |
| **Input (verified)** | `data/cache/decision_latents/decision_latents_by_state.csv` — the restored **39-col, 29-expert-anchor, 11,783-trusted-row** deliverable (verified on disk 2026-06-29: shape `(16692,39)`; `trust_caution`/`trust_timing` = generative, `trust_sharpness` = descriptive). [[b8-phase2-design-jun2026]] |

> **Plain-language contract (standing, [[feedback-plain-language-and-save-figures]]).** Every concept gets a one-line plain-English gloss; every analysis step writes a labelled, presentation-ready PNG (plain title + caption) to `FIGURES/neural_latents/BG_046/`. Glossary: **urgency/timing** = a build-up of "now is about the time the change should come" toward the *expected change time*; **caution (z)** = how trigger-happy the mouse is before real evidence; **drift/sharpness** = how strongly accumulated change-evidence pushes a lick; **commitment ramp** = a rise in population activity that reaches a threshold at the moment of the lick.

---

## 0. Inherited / verified state (do not rebuild; confirmed 2026-06-29)

- **B8 is merged to `main`** (`f61e588`). Neural work is on branch `feature/neural-decision-latents` (off `main`).
- **Fitter determinism is CI-locked**: `tests/analysis/test_decision_latents_determinism.py` committed (`aaf1c64`) and passing — `learning_ladder` byte-identical aic/bic/cvll/ll + winner across `n_workers ∈ {1,2,3}`; `state_ladder` covered by `test_state_ladder_is_seed_reproducible`. Safe to reuse the ladders at `n_workers>1` here.
- **The trusted latents are session×mood-level, not single-trial** (measured on the live deliverable, 2026-06-29):
  - `itchiness_caution` (z): **0.0% within-cell** variance — exactly **1 value per session×mood cell** (binary within a session; graded across the ~56 cells).
  - `timing_urgency_at_decision`: **78.8% within-cell** variance, and within a cell it is a **near-deterministic non-linear function of `decision_time`** (binned η² = 0.982; φ is a Gaussian peaked at the expected change time → `timing_urgency` ≈ `decision_time` in φ-coordinates). The earlier "ρ=0.17 weak RT-coupling" was a Spearman/non-monotonicity artifact, **not** independence.
  - `evidence_integral_at_decision`: 92.5% within-cell and only weakly RT-tied (η² = 0.33) — the **only genuinely per-trial, non-RT** quantity, but it rides the **descriptive** sharpness dial.
- **Consequence (shapes the whole phase):** the trusted dials' neural tests are **across-session graded** (~56 session×mood cells, mood-controlled), **not** single-trial decoding. The single-trial story lives on the evidence axis under the descriptive caveat. The timing question is therefore re-cast as an **urgency-ramp / response-timing-prediction** question, with `decision_time` as the **signal**, not a nuisance.
- **No usable video/movement data** (no facemap/pupil/DLC/motion-onset; the `video_sync` caches are sync-fitting artifacts only). All movement controls are neural/behavioral-internal.

**Conventions:** new code in `scripts/neural_latents/`; figures in `FIGURES/neural_latents/BG_046/`; caches in `data/cache/neural_latents/`; reusable logic in `src/visdetect/`, **not** `analysis_suite/`; canonical `visdetect.*` imports; `del sess; gc.collect()` in session loops; mood colours `config.STATE_LABEL_COLORS`. **HARD:** session ids via `config.canonical_session_id()` on **both sides** of any join (verified live: the deliverable stores the leading-zero-stripped form `1072025`); sessions via `load_staging_manifest()`; units via `get_good_cluster_ids()` (prefer `good_and_stable_ids`); **no compute over X:** (ProcessPool + `n_workers` locally; SLURM/ceph if heavy); all subagents **Opus 4.8** ([[feedback-subagent-model-opus]]); `ddm.py` reference-only.

---

## 1. Scope

**In scope:**
- The **urgency-ramp / response-timing** neural question on the **29 expert anchors** (headline; §2, §4, §7-C1).
- A layered **φ-specificity** test (temporal-expectation-specific, not just "neural predicts RT"), **gated on a power prerequisite** (§6).
- **Single-unit** mixed-effects encoding + **broad-SPN / narrow-FSI** cell-type breakdown (§7-C2).
- **Across-session graded confirmatory** test of the **timing amplitude u** (neural urgency-ramp amplitude vs the learned urgency across the ~56 cells), mood-controlled (§7-C3). *Caution z is **moved to a sibling question** — see Out of scope.*
- A **descriptive** per-trial evidence-axis strand (§7-C4) and a **secondary** within-expert d′ gradient.

**Out of scope (cross-referenced):** the full **naive→expert** trajectory (no generative latents for non-expert sessions → a prerequisite-gated sibling question); **D1 vs D2** identity (waveform cannot separate them; optotag yield = 3 — [[optotagging-yield-result-jun2026]]); cohort pooling (**F3**); photometry (**B5/C2/F1**); mutating `ddm.py` or the B8 fitters; **caution z (start-point/criterion operating-point)** — a baseline/criterion readout, *not* a ramp, and the weakest/most-confounded leg (100% between-cell, n≈56, chronology/d′-confounded) → it moves to a **sibling criterion/operating-point question** (criterion family — **B5/D1/F1**) with proper framing, not a diluting appendix to a ramp question.

---

## 2. The question, the claim, and the GATE

**Headline question.** In expert BG_046 medial striatum, does **pre-decision (waiting-window) population activity carry an urgency signal that predicts *when* the animal will respond, beyond generic motor preparation** — and (layered) is that signal **temporal-expectation-specific** (φ-shaped, peaked at the session's expected change time), not a generic monotonic ramp?

**Existence GATE (must pass both prongs before any cell-type decomposition):**
1. **Prediction.** Pre-decision population activity predicts response timing (`decision_time`) on **held-out** trials above a **trial-shuffle null** (≥200 shuffles; chance = mean ± 2 SD).
2. **Beyond motor prep.** The prediction **survives projecting out a freshly-built, validated lick/motor coding direction** (§5).

Passing prongs 1+2 is **already a real, cell-type-worthy result** — a clean A1/E1 finding ("a striatal urgency ramp predicts response timing beyond motor preparation"). **Fail → report the ceiling and stop** before cell-type.

> *Numbering note (maps to the brainstorm):* the brainstorm used prong 1 = predicts timing, prong 2 = φ beats a generic ramp, prong 3 = survives motor-CD. Per FIX 1 the **gate = old prong 1 + old prong 3** (renumbered here to gate-prong 1 + gate-prong 2); **old prong 2 (φ) is demoted to the layered specificity claim** below.

**φ-SPECIFICITY (layered claim, NOT a gate prong — FIX 1).** Whether a **φ-shaped temporal-expectation basis** (peaked at the session's `expected_change_time`) predicts response timing **better than a generic monotonic ramp** (ΔCV > 0, bootstrapped) is the part that ties the neural signal to the **B8 timing latent** specifically. It is reported as a layered specificity result and **must not block** the cell-type decomposition if it is underpowered or fails (§9-a). It is itself gated on a power prerequisite (§6).

---

## 3. Data, units, latents, joins

- **Sessions.** The **29 expert anchors** with trusted latents = `load_staging_manifest(qc_only=True)` ∩ {deliverable rows with `sharpness_drift` non-null}. (Naive/Learning excluded — deferred sibling.)
- **Units.** Per session, `get_good_cluster_ids()` preferring `good_and_stable_ids` ([[good-and-stable-ids-definition]]); QC per `config/qc_profiles.yml`. The plan verifies per-session unit counts up front (yield bounds the population decode — §9-c).
- **Cell type (correction to the brief).** `build_unit_table` (`src/visdetect/suite/loader.py`) + `waveform_celltype.py` give **broad-spiking (putative SPN) vs narrow-spiking (putative FSI)** (± ChIN) by waveform. **D1 vs D2 is NOT waveform-separable** and is **deferred**. Report fractions + uncertainty; FSI is a continuum, use ≥2 axes (width AND ISI/adaptation) where available ([[synthesis-phase3-celltypes]]). The plan confirms the exact label vocabulary from `unit_table_schema.py`.
- **Latents.** Join the deliverable, mapping `session_name` through `canonical_session_id()` on **both sides**. Targets/covariates: `decision_time` (the signal), `timing_urgency_at_decision`, `expected_change_time` (μ, session-level), `itchiness_caution` (z), timing amplitude **u** (the between-cell component of urgency), `evidence_integral_at_decision`, `change_size`, `outcome`, `state_label` (mood — for the labeler-independence control).

---

## 4. Neural readout & alignment (with the ramping-artifact guard — FIX 4)

- **Alignment:** `Baseline_ON` (valid for all outcomes; the waiting window is pre-change and largely pre-lick).
- **FIX 4 — fixed-length early window, NOT a window ending at the lick.** Predicting `decision_time` from a window that *ends at* `decision_time` builds in a **duration confound** (longer trials → more bins), and trial-averaged ramps can be artifacts of variable-length alignment (**Latimer & Huk 2015**, ramping critique — external lit, named explicitly). Therefore the timing prediction uses a **fixed-length** window (never one ending at the lick) and/or **single-trial ramp estimators** rather than condition-averaged ramps. **The window position is a documented SWEEP with a pre-registered selection rule (Q1), not a single guess:** evaluate ~3 fixed-length positions — **early** [0.5–2.5 s], **mid** [2–4 s], and **late-pre-change** (latest pre-μ slice, with per-trial pre-movement exclusion) — report the power/contamination tradeoff (earlier = movement-safe but the ramp may not have developed and φ≈ramp; later = more ramp signal but movement/duration-confound risk), and select the **PRIMARY = the LATEST window that is still motor-CD-free** (verified per §5).
- **Normalization (CLAUDE.md golden rule):** per-unit **z-score to a shared baseline** (shared across conditions), **normalize-then-average**, div-by-zero guard.
- **Decoders / coding directions:** fit on **train folds only**, project on held-out (no circularity); k=5 CV stratified by session.

---

## 5. Movement-control battery (make-or-break; template-skeptical) — FIX 2

Movement controls are **only** these three (the φ-vs-ramp comparison is a *specificity* test, moved out — §6):
1. **Conservative pre-movement window**, verified **neurally** movement-free (the fresh motor-CD carries ~no signal there — no video motion-onset exists to use).
2. **Fresh, validated lick/motor coding direction**, built on *these* expert sessions, **sanity-checked that it actually captures peri-lick movement**; the timing prediction must **survive** projecting it out. The **old Fig14c dPCA "readiness" template is audited before any reuse and replaced if weak** (user caveat — it was an earlier attempt; do not trust blindly).
3. **Match on movement, NOT on RT** (RT = the signal; matching on RT would delete it). Movement-matching uses the motor-CD magnitude / lick presence, not `decision_time`.

> **Never partial out `decision_time`** (FIX, §0): the timing target *is* `decision_time` in φ-coordinates, so partialling it self-nulls the gate.

---

## 6. φ-specificity layer + its power prerequisite (FIX 2 + FIX 3)

- **The claim (separate box from movement — FIX 2).** A generic monotonic ramp is *not* synonymous with "motor prep" (motor prep can ramp; urgency can be monotonic). φ beating a generic ramp establishes **temporal-expectation specificity** (μ-anchoring / curvature about the expected change time), which is the B8-latent-specific content — **not** "beyond movement."
- **The window/φ tension (NOTE A — read carefully).** The readout window is **pre-μ** (before the 6 s change floor), and over a pre-μ window φ sits on its **rising flank**, where it is **nearly identical to a generic monotonic ramp** — they differ only in (i) **curvature** (φ accelerates toward μ) and (ii) **cross-session μ-anchoring** (and the μ range is only ~0.8 s). The "52% of decisions are post-μ" statistic is about the **target** (`decision_time`); the **features** live in the pre-μ readout window, so post-μ decision mass does **not** buy φ-discriminability.
- **Power prerequisite + window-conditioning (HARD — FIX 3, corrected per NOTE A; refined from the real φ geometry μ≈7, σ=0.8).** φ-vs-ramp separability is **strongly window-dependent**, so it must be assessed **on the actual readout window**, never inferred from the decision-time distribution: over the **early** window [0.5,2.5] φ_max ≈ 1e-7 (deep tail; corr 0.59) → **ill-conditioned**, a φ-vs-ramp ΔCV there tests weighting geometry, not temporal expectation; over the **late** near-μ window [4,6] φ_max ≈ 0.46 (corr 0.89). **Task-5 synthetic finding (revises "modestly testable"):** the within-window φ-weighting is **underpowered BY CONSTRUCTION** even there — the φ-weight and ramp-weight *profiles* are ≈0.89 collinear, so a φ-vs-ramp ΔCV ≈ 0 even with a genuine φ-urgency signal present (synthetic ΔCV +0.013, CI spans 0). Therefore φ-specificity is run **once, minimally**, on the near-μ window (decoupled from C1's PRIMARY), with a **null EXPECTED**, **non-gating** (FIX 1), and **reported as a TEST-LIMITATION** ("within-window φ-weighting cannot separate the anticipation shape from a generic ramp over the achievable window"), **NOT** as "tested for temporal-expectation coding and found none." Case verdicts: `underpowered_by_construction` (the realistic late-window case) / `not_testable_ill_conditioned` (early/deep-tail) / `not_testable_on_movement_free_window`. The **headline rests on prongs 1+2**; the only meaningfully-stronger φ test is μ-anchoring across sessions (also power-limited, μ range ≈ 0.8 s).

---

## 7. Analysis components

- **C1 — Urgency ramp / response-timing prediction (headline; the gate).** Population readout in the fixed-length early window → predict `decision_time` **on lick trials (hit + fa), where `decision_time` is an observed response** (miss/right-censored trials contribute to the waiting-window readouts and the ramp characterization but not to the response-time target); nested CV, k=5 stratified by session; trial-shuffle null (≥200); motor-CD projection (§5). **NOTE B — within-trial-type graded prediction is required:** hits (post-change licks ~7 s) and FAs (anticipatory, earlier) differ grossly in `decision_time`, so pooling them risks the decoder learning **trial-type (fa-early vs hit-late)** rather than a graded urgency signal; the decode must therefore show **graded prediction WITHIN trial type (especially within hits)**, not merely across-type separation — else "predicts timing" collapses to "classifies hit vs fa." Characterize the decoded **urgency axis** as a ramp and its relation to the session μ. Then the layered φ-vs-monotonic-ramp ΔCV (§6) with bootstrap CI.
- **C2 — Single-unit encoding + cell type (only if the gate passes).** Per-unit: does early-window firing track the urgency signal / predict timing? **Mixed-effects** with trials nested in session at the population-readout level; per-unit permutation + **FDR (BH)** for the screen; **effect sizes**. Break encoder fraction/strength down by **broad-SPN vs narrow-FSI** (D1/D2 deferred). Report fractions + uncertainty.
- **C3 — Across-session graded confirmatory: timing amplitude u (mood-controlled).** Across the ~56 session×mood cells, does the **neural urgency-ramp amplitude** track the fitted **timing amplitude u** (the between-cell component of urgency), **adding beyond the binary mood label** (the labeler-independence test)? Correlational; chronology/d′-confound caveated (§9-b). *(The caution-z half is removed from N1 → sibling criterion/operating-point question; see §1 Out of scope.)*
- **C4 — Evidence-axis per-trial strand (descriptive, caveated).** `evidence_integral_at_decision` is the genuinely per-trial signal; a single-trial evidence-axis decode is reported **only** under the descriptive-trust caveat (E1/E2 flavour, not a trusted-dial claim).
- **Secondary sensitivity:** within-expert **d′ gradient** across the 29 anchors — labeled secondary, correlational, never a headline.

---

## 8. Statistics

Mixed-effects (trial nested in session) for latent↔activity; **non-parametric defaults** (Spearman/Mann-Whitney/Kruskal-Wallis); **report effect sizes** with every p; **trial-match Hit vs Miss** (and balance outcome where relevant); CV decoders with **trial/label-shuffle nulls** (chance = mean ± 2 SD); **bootstrap CIs** (1000 resamples, seed=42, percentile); **FDR (BH, α=0.05) only within a screen**, never across separate figures/questions. Consult the Research Statistician skill for test selection.

---

## 9. Testing / validation

- **Synthetic recovery (keep — extend).** Simulate (i) a **φ-urgency ramp** and (ii) a **pure-motor ramp**; confirm C1 separates them and that the **motor-CD projection kills the pure-motor case** while sparing the φ-urgency case.
- **φ-vs-ramp discriminability (FIX 3, corrected per NOTE A).** Simulate **features on the actual pre-μ readout window** (real μ range; φ rising-flank vs a monotonic ramp) and confirm separability **there** — do **not** infer it from the decision-time distribution. Expect this leg to be underpowered; demote φ-specificity to exploratory if so.
- **Null calibration:** trial/label-shuffles must return chance for the decode.
- **Join integrity:** unit test that every leading-zero-day anchor (`01072025`…`09092025`) joins after `canonical_session_id()` (regression on the verified footgun).
- **Determinism:** reuse the CI-locked fitter determinism guarantee where ladders are touched.
- **Compute hygiene:** heavy loops → `ProcessPoolExecutor`, param **`n_workers`** (NOT `n_jobs`), BLAS pinned per worker; session pkl loading stays sequential; **no compute over X:**.

---

## 10. Outputs & repo structure

- `scripts/neural_latents/` (import `visdetect.*`); `data/cache/neural_latents/` (incl. the neural×latent joined table, session_name canonicalized both sides); `FIGURES/neural_latents/BG_046/`.
- Per-step **presentation-ready** figures (plain-language titles/captions) + **stats CSVs** alongside each figure.

---

## 11. Risks / decision points

- **(a) φ-specificity is expected NULL and reported as a test-limitation, not a science claim.** Task-5 synthetic showed the within-window φ-weighting is underpowered *by construction* even on the near-μ late window (φ-weight vs ramp-weight profiles ≈0.89 collinear → ΔCV≈0 even with a real signal present). Report it as "the test cannot separate the anticipation shape from a generic ramp over the achievable window" — NOT "no temporal-expectation coding." Per FIX 1 it **does not block** C2; the headline rests on prongs 1+2 (urgency ramp predicts response timing beyond motor prep — the validated real result). μ-anchoring across sessions is the only stronger alternative (power-limited, μ range ≈ 0.8 s) — pursue only if prongs 1+2 land and a reviewer asks. **F4-grounded framing (write-up):** temporal expectation is ALREADY established *behaviourally* by B8's **F4** — a true survival-corrected anticipatory FA-hazard that RISES toward the expected change — so N1 need not prove temporal expectation neurally. F4 also explains the φ null: over the pre-change window the anticipatory signal is a rising RAMP, and the φ-bump's distinguishing peak sits at/after the change time (entangled with detection + movement), outside the clean pre-movement readout window. So frame N1 as: **F4 (behaviour) → temporal expectation EXISTS; N1 prongs 1+2 → its neural correlate (urgency ramp predicts WHEN the mouse responds, beyond movement); φ-specificity → the honest, structurally-expected null.** (Possible discussion: an increasing change-time hazard means the true anticipation may genuinely be ramp-like, so "ramp ≈ φ" may partly reflect signal shape, not only observability.)
- **(b) The across-session graded u test (n≈56 cells)** is correlational and **chronology/d′-confounded** — reported with that caveat; never causal. (Caution z, the weakest such leg, has been moved out of N1 → sibling.)
- **(f) Hit/FA trial-type confound (NOTE B)** — the response-time decode must show **within-type graded prediction (esp. within hits)**; across-type separation alone would mean "classifies hit vs fa," not "encodes urgency." Enforced in C1.
- **(c) Unit yield** per expert session bounds the population decode (chronic-probe SPN under-yield/drift — [[qc-celltype-yield-investigation-jun2026]]); the plan verifies counts up front and reports them.
- **(d) Ramping artifact (Latimer & Huk 2015)** — mitigated by the fixed-length early window + single-trial methods (§4); named here as a standing risk.
- **(e) Circularity re-entry** — caution's within-session contrast is the two moods and is **forbidden** as a neural claim; only the across-session graded, mood-controlled test (C3) is permitted.

---

## 12. Literature grounding

- **Commitment / urgency ramp = the field's oldest open question** (Gold/Shadlen): [[synthesis-batch01-foundations]], [[synthesis-batch06-brainwide-population]] (Roitman & Shadlen ramp; Huk/Shadlen pulse = TF-analog); striatal commitment ramp candidates (Balewski fast-slow corticostriatal; van Beest direct/indirect; Stine terminating decisions) via landscape **A1/A2** ([[question-landscape-jun2026]]).
- **Temporal expectation = the convergent organizing variable** of the spine ([[research-direction-post-tf-null-jun2026]]).
- **Movement confound** (regress/establish movement-null first): Stringer spontaneous activity ([[synthesis-phase3-behavioral-state]]); cross-cutting cautions ([[question-landscape-jun2026]]).
- **DDM / urgency theory:** Bogacz; [[synthesis-phase3-theory]].
- **Ramping critique (external, named):** Latimer & Huk 2015 — variable-duration alignment can fake ramps (§4, §9-d).
- **Cell-type honesty:** [[synthesis-phase3-celltypes]], [[optotagging-yield-result-jun2026]], [[qc-celltype-yield-investigation-jun2026]].
