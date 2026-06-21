# Design spec — B8 Phase 2: Generative decision-latents by state

| | |
|---|---|
| **Question ID** | B8 Phase 2 (extends `2026-06-18-B8-behavioral-decision-latents-by-state-design.md` §4 Step 2, §6, §9) |
| **Date** | 2026-06-20 |
| **Status** | SPEC-DRAFT (brainstormed 2026-06-20; hardened by a read-only verification + adversarial-review pass; awaiting user review → writing-plans) |
| **Feasibility tier** | T1 (behavior-only; BG_046 trial data + new-labeler state tags). **Single-subject (n=1, BG_046)**, architected for a later hierarchical **F3** cohort replication. |
| **Spine** | The **generative** half of the post-TF-null direction (`docs/science/2026-06-17-post-tf-null-research-direction.md`, §6 step 2): emit the per-trial latents the neural phase regresses against. End goal (multi-subject behaviour + neural + mechanistic, D1/D2 SPNs by aMOs/pMOs input) is the destination; Phase 2 is one component, n=1. |
| **Modeling (locked)** | **Engine A** = a minimal **discrete-time survival hazard "regression-accumulator"** with a **closed-form** likelihood (chosen over a pyddm full-DDM at fine `dt` because the long-baseline regime makes Fokker-Planck recovery intractable). **Engine C** = a pyddm full-DDM **spot-check** on a few expert sessions (construct validity + the only place the bound-vs-start-point split is recoverable). Descriptive-first, **recovery-gated**. **Build anew; `ddm.py` is reference-only — do NOT mutate it.** |

> **Plain-language contract (standing, [[feedback-plain-language-and-save-figures]]).** Every concept gets a one-line plain-English gloss; every analysis step writes a labelled, presentation-ready PNG (plain-language title + caption) to `FIGURES/decision_latents/BG_046/`. Glossary: **sharpness** = how clearly the mouse tells the grating changed; **itchiness/caution** = how trigger-happy it is before real evidence; **timing** = how strongly it expects the change *now*.

---

## 0. What Phase 1 already shipped (inherited, do not rebuild)

Phase 1 (descriptive dials + a distribution-justified QC gate) is **merged to `main`** (commits `fc2b4ba`, `334bb85`). Verified current state (2026-06-20, read-only audit):

- **Library** `src/visdetect/analysis/decision_latents.py` with the reusable surface: `load_state_labels`, `enumerate_valid_sessions(min_total_trials=50)`, `session_dprime` (reads `compute_session_performance(...)['d_prime']` — key is **`d_prime`**, not `dprime`), `assign_comprehension_flags(threshold=0.5)`, `build_trial_table(session, state_labels, session_name, dt=0.05)`, `censored_hazard(durations, events, dt=0.05, t_max=None)`, `sharpness_scores`, `itchiness_scores`, `timing_scores`, `change_onset_hazard`, `lick_hazard`, `fa_lick_hazard` (already censors non-FA trials at `change_time_planned`), `compute_cell_qc`, `descriptive_cell_table`, `descriptive_latent_table`. Mood sets: `MAIN_MOODS=('Impulsive','StimSens')`, `SEPARATE_MOODS=('Disengaged',)`, `EXCLUDED_MOODS=('Abort',)`.
- **Per-metric QC** `compute_cell_qc()` emits per-(session×mood) counts + flags `has_psychometric_support`/`usable_sdt`/`usable_rtcv`/`usable_timing` from distribution-justified thresholds (`QC_MIN_GO=8`, `QC_MIN_DISTINCT_CS=2`, `QC_MIN_CATCH=5`, `QC_MIN_RT_PER_CS=3`, `QC_MIN_RTCV_CS=2`, `QC_MIN_TIMING_TRIALS=20`). `descriptive_cell_table` gates **per-metric** (a cell can be usable for d′ but not for a threshold).
- **Deliverable** `data/cache/decision_latents/decision_latents_by_state.csv` — **16,692 rows × 25 columns** (per-trial table with `usable_*` flags). Phase 2 **appends** to this; never overwrites.
- **Profiler** `scripts/analysis/decision_latents/behavioral_qc_profile.py` (→ `fig_b8_QC_distributions.png`, `behavioral_qc_cell_table.csv`). Current data: **79 usable / 115 cells** (36 dropped; the binding gate is ≥3 go-trials per change-size). *(The "33/115" in `memory/feedback_data_quality_gate_first` is stale; the live number is 36/115 as of 2026-06-20.)*
- **Discrete psychometric explorers** `scripts/analysis/decision_latents/explore_psychometric_discrete.py` → **F1C** (discrete P(lick) vs change-size, per-session spaghetti + bootstrap CI, criterion lines at 50/60/70%) and **F1D** (threshold at the three criteria over learning, floor-fraction annotations).

**Conventions (unchanged):** scripts in `scripts/analysis/decision_latents/`; figures in `FIGURES/decision_latents/BG_046/`; caches in `data/cache/decision_latents/`; library logic in `src/visdetect/`; **not** `analysis_suite/`; styling via `suite.plotting.setup_style` + a local `save_fig` helper; mood colours from `config.STATE_LABEL_COLORS` = `{Impulsive:#ef6548, StimSens:#6baed6, Disengaged:#3474ae, Abort:#bdbdbd}`; `del sess; gc.collect()` in loops; canonical `visdetect.*` imports.

---

## 1. Scope

**In scope (Phase 2):**
- **Engine A**: the discrete-time hazard-accumulator (§2), fit **expert-first, backward-seeded** (§4), gated on **parameter recovery at the real long-baseline regime** (§5).
- The **per-trial generative latent columns** appended to the deliverable (§9), with a `latent_trust` provenance flag carrying the honest fallback in the data.
- **Engine C**: a pyddm full-DDM spot-check on a few expert sessions (§8).
- The five **carried-forward fixes** + identifiability pre-flight, as **blocking Phase-0 prerequisites** (§7).
- Figures **F6** (recovery + confusion matrix), **F7** (latent distributions, timing-led), **F8** (construct validity + Engine-C panel).

**Out of scope (cross-referenced):** neural/spike data (downstream); whether the integration timescale `λ` is *learned* (**B1** owns it); the literal bound-vs-start-point split everywhere (only spot-checked, §8); cohort pooling (**F3**); a formal confidence model; mutating `ddm.py`.

---

## 2. The model — Engine A (discrete-time survival hazard-accumulator)

**Plain version:** at each 50 ms slice of a trial, *given the mouse has not yet licked*, there is some probability it licks in this slice. That probability is raised by three things — a baseline urge (**itchiness/caution**), the change-evidence it has accumulated (**sharpness**), and a "now is about the time" expectation (**timing**). Those three are the dials.

For trial *i*, bin *k* at time `t_k = k·dt` (dt = **0.05 s**, the verified 50 ms TF update period, §7-a), conditional on survival to *k*:

```
cloglog(h_{i,k}) = β0_i                        # ITCHINESS/CAUTION  (baseline-period urge to lick)
                 + βv_i · A_{i,k}              # SHARPNESS          (drive from accumulated change-evidence)
                 + (u0 + u_state[mood_i]) · φ(t_k; μ_s, σ)   # TIMING  (temporal-expectation bump)
```

- **`cloglog(h) = log(−log(1−h))`** — the complementary-log-log link, the standard link for *grouped/discrete-time survival* (Prentice & Gloeckler 1978; Allison). Plain version: it makes "chance of licking in a 50 ms slice" behave like a clean slice of an underlying continuous-time hazard, so conclusions don't wobble with bin width. *(General statistical methodology — not claimed to be in the project literature corpus.)*
- **`A_{i,k}` = leaky running total of rectified log2-TF evidence:** `A_{i,k} = (1−λ·dt)·A_{i,k-1} + R(e_{i,k})·dt`, with `e = log2(TF(t)/tf_base)`, `R` = rectification (§3-1). The leak **`λ` is fixed and shared** (§3-2).
- **`φ(t; μ_s, σ)` = a Gaussian temporal-expectation bump** peaked at the **per-session empirical change-time anchor `μ_s`** (median of `change_time_planned` over `change_reached` trials — §7-d), width `σ`. Amplitude is the timing dial.
- **Three dials carry a per-trial mood term** (Impulsive vs StimSens): `βv_i = v0 + v_state[mood_i]`, `β0_i = z0 + z_state[mood_i]`, urgency amplitude `u0 + u_state[mood_i]`.

**Likelihood (closed-form, censoring native):** for a trial that licks at bin `K`: `h_K · Π_{k<K}(1−h_k)`; for a censored trial (Miss / no-lick, right-censored at `K`): `Π_{k≤K}(1−h_k)`. Sum the negative log-likelihood over trials; fit by penalized maximum likelihood. This is the *parametric* extension of Phase 1's `censored_hazard` bookkeeping — **no Fokker-Planck grid**, so it is fast enough to run the full backward sweep *and* the recovery replications at the real long-baseline regime (the regime B0 never tested).

**Declarative parameter spec (extensibility hook).** Which parameters carry a learning term, which carry a state term, and which are shared is a **config object**, not hardwired — so adding "caution varies with learning" (already a ladder rung, §4) or swapping in the full-DDM `a`/`z` (§8) later is a config change, not a rewrite. Same principle as parameterizing by `subject=` for F3.

**Engine A's structural limit (stated honestly).** A hazard intercept is **not** an absorbing bound: Engine A's `β0` **conflates** the DDM bound (`a`) and start-point (`z`) into one **caution/itchiness** dial. Engine A therefore yields **three** dials (sharpness slope, caution/itchiness intercept, timing amplitude) and **cannot** split bound from start-point. That split is recoverable only via the **Engine-C** full-DDM spot-check (§8) and the Step-1 RT-distribution shape. This is a deliberate trade (tractability + recovery over a literal-DDM interpretation we don't need for the headline claim).

---

## 3. The three knob decisions (literature-grounded)

The field is genuinely split on all three, so the gold standard is **methodological**: *test* which knob moves (model comparison) and *prove* you can tell them apart (recovery), rather than hardcode. Verified literature (read-only corpus audit, 2026-06-20):

**(1) Rectification — selected by CV log-likelihood, not hardcoded.** The canonical DDM decision variable is accumulated **log-likelihood-ratio**, which is **signed** (Bogacz 2006; Gold & Shadlen) — samples consistent with "no change" push the accumulator *down*. Half-wave (B0's choice) discards the downward half. **Decision:** select `R ∈ {signed/symmetric, half-wave, asymmetric(g_up,g_down)}` once on the expert anchor by held-out CV-LL, then **freeze** for the backward sweep; record `rectification_kind` in provenance. Signed/half-wave cost zero extra parameters; with the leak in place a signed accumulator does not run away during the baseline.

**(2) Leak `λ` — fixed at the task-specific literature value, sensitivity-checked, "is it learned" deferred to B1.** Verified: **Khilkevich & Lohse 2024**, on *this exact task*, fit a **leaky-integrator-to-threshold** with **τ ≈ 0.27 s** (best-fit decay predicting early-lick *times* + single-trial RTs). Contrast: Orsolic ~1 s (a *different* method — learned multi-lag filter bank); Brunton near-leak-free ~1 s; Uchida rapid ~200–300 ms. **Decision:** fix `λ` at **τ ≈ 0.27 s**, run a **sensitivity sweep (≈150/270/400 ms)** and show the qualitative learning/state conclusions are robust; defer *"is the timescale learned?"* to **B1**. Caveat (verification, critic): τ comes from a *different model class*; the sweep + a B1-drop-in placeholder for a learned-`λ` function de-risk the circular-ordering worry.

**(3) Which dial moves with learning/state — tested by a nested model-comparison ladder, not assumed.** Verified: the learning-DDM literature (Bogacz threshold=SAT, start=log-prior; Masís 2023 SAT-over-learning; Ratcliff/Frank collapsing threshold) has **drift rise, boundary separation drop, non-decision time shrink** across learning — *multiple* knobs move, and the **bound is the speed-accuracy knob**. Hardcoding the bound as shared would **assume away** the SAT/caution account. **Decision:** run two ladders (§4) — a **learning ladder** (`M_shared / M_sharpness / M_caution / M_timing / M_full`) and a **state ladder** — scored by AIC/BIC **and** CV-LL. The **recovery gate** (§5) gates which subset is jointly identifiable; free those, fix the weakly-identified nuisances (leak, non-decision time), and **report the ceiling**.

> **Verification correction folded in (Orsolic):** temporal expectation gates **MOs/motor-cortex recruitment** by sensory evidence — *not* the integrator itself (the TF integration runs in early- and late-change blocks alike). The spec frames timing as the mouse's learned *read-out* expectation, consistent with this.
> **Identifiability framing confirmed (Bogacz caveat):** BG_046 is an **asymmetric go/withhold detection** task — only the lick side is timed; withhold/miss are **censored non-responses**, so there is **no wrong-side error-RT distribution** to separate bound from start-point (the lever symmetric 2AFC DDMs use). This is *why* the hazard formulation (native censoring) + model comparison + recovery are the right tools, and why bound-vs-start-point is Engine-C-only.

---

## 4. Fitting & anchoring — expert-first, backward-seeded + the ladders

**Plain version:** lock the model where the mouse is an expert (cleanest behaviour), then step *backwards* through training session-by-session, each step starting from the previous (more-expert) answer and moving only as far as its own licks force. Then ask, formally, *which* dial had to change to explain the whole trajectory.

1. **Learning axis = chronological session order** (`parse_session_date` / `chronological_sort` from `visdetect.analysis.config`), with per-session d′ a **covariate** (colour/size), **never the x-axis**.
2. **Expert anchor fit freely** (identifiable regime → reference template). Then fit each earlier session in **reverse-chronological** order, optimiser **seeded at and L2-regularised toward** its more-expert neighbour. *Fitting order (newest→oldest) is the opposite of the reporting axis (oldest→newest); docstrings + variable names + a chronologically-sorted output table will make this unambiguous (critic).*
3. **State terms inside each fit** (Impulsive vs StimSens). Disengaged reported separately; Abort dropped; **only QC-passing cells contribute** (§7-e). Engagement is handled **per-trial via mood**, not by dropping sessions.
4. **Two ladders** (the *test*, not an assumption; nested-model comparison in the spirit of B0's `compare_stage_models`, but **re-implemented for the hazard-GLM** — the pyddm AIC/parameter-count formula does **not** transfer directly; use GLM degrees-of-freedom + CV-LL):
   - **Learning ladder** (across anchors): `M_shared / M_sharpness / M_caution / M_timing / M_full` → names **which dial learning turns**.
   - **State ladder** (within anchor): which dial must carry a mood term → tests *"states load on caution/timing, not sharpness."*
5. **"Which dial moves when" = distance-from-the-expert-template** (carried from §6 of the base spec).

**Guardrails against the backward-seeding manufacturing the trajectory (critic, high-confidence — load-bearing).** Expert-anchored L2 shrinkage could, at a rank-deficient naïve-end likelihood, *dominate* the data and make the "learning trajectory" a regularization artifact (compressing real learning effects toward the expert value). Mandatory mitigations, all reported:
   - **Unseeded baseline:** fit naïve sessions **without** expert seeding first; report the **Hessian/Fisher-information condition number / effective rank** per anchor (a rank-deficient naïve Hessian is itself a finding → those latents are flagged descriptive-only).
   - **L2-weight sensitivity:** sweep the regularization weight (e.g. 0, 0.01, 0.1, 1, 10) and show the conclusions are not a tuning artifact.
   - **Recover-a-true-difference test (in the recovery gate, §5):** simulate stages with a **genuinely different** true dial (e.g. `v_naïve=1.0`, `v_expert=2.5`); confirm the seeded backward fit **recovers the difference**, not shrinks it to the mean. *This is the decisive check that seeding informs without erasing.*

---

## 5. The recovery gate (make-or-break) — quantitative, at both regimes

**Plain version:** before trusting a single latent, invent fake mice with *known* dials, generate their licks **on the real trials** (real TF streams, real ≥6 s change-times, real long durations — the regime B0 skipped), refit with the exact pipeline, and check we get the known dials back. Do it for *both* an expert-like and a naïve-like fake mouse, because the backward sweep leans hardest on the naïve end.

**Simulation harness.** Generate licks through the **same** cloglog hazard model on **real per-trial evidence + change-times + durations at dt = 0.05**. Engine A can plausibly pass where B0 could not: the closed-form hazard likelihood is *exact* on this grid — there is no Fokker-Planck `dt`-imprecision (the "dt is large" warning class). **Two regimes:** *expert-like* (more change-driven licks) and *naïve-like* (more flat-evidence hair-trigger licks; low sharpness, high itchiness).

**The three recovery tests (all required):**
1. **Point recovery** — per-dial recovered-vs-true over **N ≥ 100 replicates per regime**; provisional pass: **Pearson r ≥ 0.8** and **bias ≤ 0.1·SD(true)** per dial; **bootstrap CI coverage** ≈ nominal 95%.
2. **"Which-dial-varies" confusion matrix (decisive)** — simulate three sweeps where **only one** dial truly changes across anchors (only-sharpness / only-caution / only-timing); the **learning ladder must pick the true dial** in each. Report the full **3×3 confusion matrix**; the off-diagonal directly measures whether the **sharpness↔caution** and **urgency↔itchiness** trade-offs fool the model (both produce early licks — critic). High off-diagonal ⇒ the relevant dial is **descriptive-only**.
3. **Recover-a-true-difference** — §4 guardrail: seeded backward fit must recover a real stage difference, not regularize it away.

**Pre-flight (before any real fit).** Compute the **Fisher information / Hessian condition number** at the expert regime on real data, and confirm an **adequate expert anchor exists** (§7-h). Ill-conditioned ⇒ the identifiability ceiling is real ⇒ ship Phase-1 proxies.

**Decision rule — per-dial, not just binary (critic).**
- **Pass both regimes, all three tests** → ship the generative latent for that dial.
- **A dial passes point-recovery but fails the confusion test, or passes expert-only** → emit it for expert/post-comprehension trials, **flag naïve-extrapolated trials `latent_trust='descriptive'`** (fall back to the Phase-1 proxy there). Dials are gated **independently** (sharpness may ship while timing is descriptive-only).
- **Broad failure** → ship the Phase-1 descriptive latents as the table and **report the identifiability ceiling** (still a usable neural-regressor set and a real result).

*(Tolerances/bootstrap scheme — session-level vs trial-level resampling — are finalized with the **Research Statistician** at planning, base spec §11. The numbers above are provisional defaults to be confirmed, not invented thresholds patched after the fact.)*

---

## 6. Identifiability & confounds (load-bearing)

- **Bound vs start-point conflation** — Engine A cannot split them (§2); **validated, not asserted**: the confusion matrix (§5-2) must show *only-caution-varies* is correctly attributed to caution; the split itself is recovered only by Engine C (§8). If unresolved, naïve caution latents are descriptive-only.
- **Urgency ↔ itchiness** — both produce early licks; separable only via the FA **time-course** (a late-rising bump vs a flat intercept). The empirical-median anchor `μ_s` (§7-d, not the late-biased hazard peak) and an explicit Gaussian `φ` are what make the bump identifiable; the confusion test proves it. Report their fitted correlation.
- **State-label circularity (load-bearing — affects how the headline reads).** The labeler's features include `f_inapplick`, `f_hit_hard`, `f_miss_easy`, so **itchiness×mood is substantially definitional** and even **bias-not-gain is partially entangled** ([[state_labeler_circularity_caveat]]). **Execution requirement (critic, blocker):** F7/F8 and the narrative **lead with the labeler-independent dials — timing (hazard shapes / lick-time clustering) and RT variability** — and **explicitly label the FA-rate/criterion×mood contrast "confirmatory"** (expected from the labeling procedure), reserving "discovery" weight for timing/RT and the downstream neural correlates. The abstract/captions must state the circularity up-front.
- **Two impulsivities** — pre- vs post-comprehension. Handled by the `comprehension_flag` (§7-f), main claims read on post-comprehension; pre-comprehension analyzed as a labeled de-sculpted reference.
- **Within-session position / satiety** — carried as a covariate (`trial_in_session`); report itchiness vs trial-in-session so a satiety gradient isn't misread.
- **State-label reliability on naïve sessions** — OOD for the labeler; a concrete reliability protocol gates their use (§7-g).
- **n = 1** — within-subject mechanism + the neural deliverable are legitimately n=1; *generality* is **F3** (hierarchical, mouse as random effect — never pooled). Architecture is subject-parameterized so F3 is an extension.

---

## 7. Phase-0 prerequisites (BLOCKING — run and pass BEFORE any generative fit)

The data-quality-gate-first directive ([[feedback_data_quality_gate_first]]) plus the critique pass make these **blocking**: each profiles/validates the foundation and writes a presentation-ready figure; fitting starts only after they pass.

- **(a) Corrected evidence builder — non-negotiable, first.** Verified bug: `ddm.build_trial_evidence` uses `bperiod = ct/len(bv)` (correct only by accident when `len(bv)` happens to span `change_time`; wrong by ~2× on long-`change_time` sessions). Verified truth (`scripts/analysis/decision_latents/_tf_sampling_check.py`): `baseline_values` is stored at **60 Hz**, each TF value **held 3 frames (50 ms)**, **`n_seen` is always `None`**. **Build a NEW corrected evidence builder in `decision_latents.py`** (do **not** mutate `ddm.py`) that indexes at 60 Hz / collapses runs-of-3, with a unit test validating reconstructed evidence against the raw TF stream. Engine A uses **only** this builder.
- **(b) Lapse-aware psychometric (fix b).** Replace the 2-param logistic in `sharpness_scores` with **threshold + slope + lapse**: `P(lick|cs) = λ_lapse + (1−2λ_lapse)·logistic(a + b·log2 cs)`, `λ_lapse ∈ [0, 0.3]`; add a `psy_lapse` column. Motivated by **F1C/F1D** (Impulsive's curve is *shifted up*, not steeper; 70 % is the discriminating criterion; the lower asymptote is the lapse term). **Re-run the Phase-1 cell table** so Phase 1 and Phase 2 measure the same thing (else F8 construct validity shows spurious disagreement — critic). Lead with F1C/F1D; footnote the 50 %-only F1B.
- **(c) Baseline-hazard pre-change window (fix c).** `itchiness_scores.baseline_hazard` currently averages over the **full** decision timeline (diluted by post-change bins). Restrict to the pre-change window (censor non-FA trials at `change_time_planned`, as `fa_lick_hazard` already does); re-run the cell table if materially different.
- **(d) Empirical change-time anchor (fix d).** Replace the late-biased hazard **peak** with the **empirical median** `μ_s = median(change_time_planned | change_reached)` per session; this seeds `φ`'s peak and the `expected_change_time` column.
- **(e) QC re-derivation + a generative-sufficiency gate.** Re-derive the still-inherited **session-level** gates (`min_total_trials=50`, comprehension `d′≥0.5`) from the profiler. **Add** a distribution-justified **`usable_generative`** flag for the generative fit's *own* inputs (enough lick events **and** censored trials to identify a hazard slope; trials spanning `μ_s` for the urgency bump; real evidence excursions for sharpness). One canonical flag per cell, applied consistently; **cells failing QC do not enter the fit**.
- **(f) `comprehension_flag` operationalization.** Compare candidate rules (d′≥0.5 vs first session with reliable easy-change hit-rate vs both), overlay boundaries on a figure, pick by inspection, and **report sensitivity to ±1 session**.
- **(g) Naïve-session label-reliability protocol.** Re-label the ~18 untagged (mostly earliest, lowest-d′) sessions; compare mood proportions naïve vs post-comprehension; inspect exemplars; define a confidence-gating rule (e.g. if a session has <80 % of trials with `state_confidence>0.7`, drop it to coarse no-mood level). Extends the base plan's Task 0.2. **Do not silently trust naïve moods.**
- **(h) Expert-anchor data inventory.** Compute per-session d′ × per-mood `n_trials`; identify the expert subset (d′>0.7 **and** per-mood n≥20). **If <3 adequate expert sessions, trigger the contingency** (pool late post-comprehension sessions as the anchor, or ship Phase-1 proxies only) — caught now, not at fitting time.

---

## 8. Engine C — pyddm full-DDM spot-check (construct validity + the bound/start-point split)

On a few **expert** sessions only (where the full DDM is affordable), fit B0's pyddm model (import `ddm.build_model`/`fit_model`/`select_structure` **without mutating `ddm.py`**) and:
- scatter **GLM sharpness `βv` vs DDM drift `v`**, **GLM urgency amplitude vs DDM `u`** — do the cheap analog latents track the literal DDM?
- where afforded, read the **bound `a` vs start-point `z`** split that Engine A structurally cannot provide.
This is a **cross-check, not part of the gate**. Note the `dt` difference (`ddm.py` `DT=0.02`; Engine A `dt=0.05`) when comparing.

---

## 9. Deliverable — appended latent columns + figures

**Append** to `data/cache/decision_latents/decision_latents_by_state.csv` (never overwrite the 25-col Phase-1 table). New per-trial columns:

| Column | Meaning (plain) | Kind |
|---|---|---|
| `sharpness_drift` | fitted evidence-drive `βv` for the trial's mood/anchor | regression-varying |
| `itchiness_caution` | fitted baseline-urge intercept `β0` (= bound+start-point, combined) | regression-varying |
| `timing_urgency_at_decision` | urgency value at the decision bin | **trial-specific** |
| `evidence_integral_at_decision` | change-evidence banked at decision (`A_{i,K}`) | **trial-specific** |
| `expected_change_time` | session anchor `μ_s` | per-session |
| `lick_minus_expected` | `decision_time − μ_s` | **trial-specific** |
| `anchor_id`, `rectification_kind`, `leak_tau`, `recovery_regime`, `latent_trust` | provenance; `latent_trust ∈ {generative, descriptive}` carries the §5 fallback **in the data** | provenance |

**Figures** (plain-language titles/captions; `STATE_LABEL_COLORS`; `FIGURES/decision_latents/BG_046/`):
- **F6 — recovery:** recovered-vs-true per dial at both regimes + the **which-dial-varies confusion matrix** (decisive panel) + the Hessian-conditioning / L2-sensitivity summary.
- **F7 — latent distributions:** the three dials by mood and across learning anchors — **led by the labeler-independent timing dial + RT variability** (§6).
- **F8 — construct validity:** generative dials vs Phase-1 descriptive scores (sharpness↔lapse-aware psychometric, caution↔criterion/FA, timing↔change-time anchor) + the **Engine-C panel**.

**Code:** extend `src/visdetect/analysis/decision_latents.py` (corrected evidence builder, cloglog hazard likelihood + censoring, leaky accumulator, the anchored backward-sweep fitter, the two ladders re-implemented for the GLM, the recovery harness, the latent appender, the declarative param-spec config); reuse `ddm.py`/`behavior.py` helpers **without mutation**; tests in `tests/analysis/test_decision_latents.py` (**recovery + confusion-matrix are the primary tests**; hazard-likelihood/censoring correctness; lapse-psychometric; evidence-builder vs `_tf_sampling_check`); a Phase-2 orchestration script under `scripts/analysis/decision_latents/`.

---

## 10. Success criteria & honest fallbacks

- **Must-have:** the Phase-0 prerequisites pass; the model-comparison ladders answer *which dial learning turns / which dial the moods load on* on QC-clean cells, **with the load-bearing claims leaning on labeler-independent timing/RT** (§6).
- **The deliverable:** per-trial generative latents **with validated recovery** (§5), per-dial `latent_trust` flags, construct validity against Phase-1 scores (F8).
- **Negative/inconclusive (a real, shippable result):** recovery fails (or the expert anchor is inadequate, §7-h) → ship the Phase-1 descriptive proxies as the latent table and **report the identifiability ceiling**. Still a usable neural-regressor set.

---

## 11. Cross-checks against existing machinery

- **B0 `ddm.py`** — reuse `build_trial_evidence` *geometry only is NOT safe* (the `bperiod` bug, §7-a); reuse `rectify`, the pyddm contract, `build_model`/`fit_model`/`select_structure` for **Engine C**. The nested-ladder *pattern* (`compare_stage_models`) is conceptually reusable but its **pyddm AIC/parameter-count formula does not transfer** to the GLM — re-implement. **Do not mutate `ddm.py`.**
- **Phase-1 `decision_latents.py`** — reuse `censored_hazard`, `build_trial_table`, `compute_cell_qc`, the score functions, `enumerate_valid_sessions`, `assign_comprehension_flags`.
- **`behavior.py`** — SDT/psychometrics for the lapse-aware refit and F8.
- **`_tf_sampling_check.py`** — the authority for the corrected evidence builder (§7-a).

---

## 12. Verification record (read-only audit + critique, 2026-06-20)

Folded-in corrections to the brief/base spec (so they persist):
- **B0 recovery-test status — RESOLVED: the test PASSES on `main` (do NOT assert it "fails").** The brief and the Phase-1 plan stub state `tests/analysis/test_ddm.py::test_stage_comparison_recovers_the_true_varying_knob` *currently fails*. **This is not reproducible on `main` as of 2026-06-20:** two independent runs today **PASS** (173 s and ~655 s; `@pytest.mark.slow`; "dt is large" warnings present but non-fatal). So Phase 2's recovery motivation **must not** rest on a "B0 test fails" claim. It rests on the verified facts that (i) pyddm fits at coarse `dt=0.02` are flagged imprecise by pyddm itself ("dt is large — estimated pdfs may be imprecise"), (ii) the passing test is **marginal by design** (its own docstring notes `M_full` can edge `M_v`) and uses **reduced trial counts + fast/seeded DE on synthetic short trials**, and (iii) B0 **never tested recovery at the real long-baseline regime** (real ≥6 s change-times, `T_dur≈p99`, real TF streams) — combined with the go/no-go task structurally lacking wrong-side error-RTs (§3). *(Flagging for the user: the brief's premise about this test appears outdated; please confirm whether you observed a failure in a different state/branch.)*
- **TF sampling:** confirmed **60 Hz storage, 50 ms (3-frame) hold, `n_seen=None`** — *not* the `0.25 s` `TF_SAMPLE_PERIOD` constant (a known-wrong binning, `memory/tf_fluctuation_50ms_vs_constant`).
- **QC count:** 36/115 cells dropped (79 usable), not the stale "33/115".
- **Orsolic:** temporal expectation gates **MOs recruitment**, not the integrator (§3).
- **Carried-forward fixes (a)–(d) are NOT yet implemented** in Phase 1 (verified) → they are Phase-0 blockers here (§7).
- **`cloglog` is general statistical methodology**, not corpus-cited; flagged as such (§2).

---

## 13. Links

- Base spec: `docs/superpowers/specs/2026-06-18-B8-behavioral-decision-latents-by-state-design.md` (§4 Step 2, §6, §9); plan: `docs/superpowers/plans/2026-06-18-B8-behavioral-decision-latents-by-state-plan.md` (Phase-2 stub + carried-forward refinements).
- Direction: `docs/science/2026-06-17-post-tf-null-research-direction.md`; `memory/research_direction_post_tf_null_jun2026`.
- Memory: `state_labeler_circularity_caveat`, `feedback_data_quality_gate_first`, `feedback_plain_language_and_save_figures`, `feedback_repo_structure_scripts_figures`, `reference_state_label_colors`, `tf_fluctuation_50ms_vs_constant`.
- Literature: `synthesis-phase3-theory` (Bogacz), `synthesis-batch01-foundations` (Khilkevich/Lohse, Orsolic, Gold/Shadlen), `synthesis-batch07-sweep` (Brunton/Uchida/Masís integration-timescale debate), `synthesis-batch05-confidence-lapses` (Masís SAT-over-learning).
- Reference-only: `docs/superpowers/specs/2026-06-10-B0-ddm-learning-knob-design.md`; B1 owns `λ`.
- Index: add the B8-Phase-2 row to `docs/science/QUESTION_INDEX.md`.
