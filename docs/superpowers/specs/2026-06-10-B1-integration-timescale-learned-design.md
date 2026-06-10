# Design spec — B1: Is the evidence-integration timescale a *learned* quantity?

| | |
|---|---|
| **Question ID** | B1 (see `memory/question_landscape_jun2026.md`) |
| **Date** | 2026-06-10 |
| **Status** | SPEC-APPROVED (2026-06-10; discrepancy reframed as estimator-driven not modality/task, t0 truncation + engaged-FA-primary conditioning + E3 deferred to post-B0, all added in review) — ready for writing-plans |
| **Feasibility tier** | T1 (existing BG_046 behaviour) |
| **Spine** | *How do mice learn to suppress impulsivity and increase sensitivity?* — integration timescale is part of **how evidence is read** (the sensitivity half): a longer, well-shaped integration window is a learned sensory-readout strategy |
| **Lit anchor** | The unresolved cross-study discrepancy: Orsolic 2021 (`paper-orsolic-2021-mesoscale-task-origin`) ~1 s (slow integrator + fast derivative) vs Khilkevich/Lohse 2024 (`paper-khilkevich-lohse-2024-brainwide`) ~0.27 s on near-identical tasks; the integration-timescale debate (`synthesis-batch07-sweep`: Brunton, Pardo-Vázquez, Uchida; Hyafil robust integration). The synthesis verdict: **report multiple estimates, do not commit to one number.** |

---

## 1. Scientific question & hypothesis

**Question.** Across BG_046's Naive→Expert trajectory, does the **evidence-integration timescale `τ`** — the temporal window over which the mouse integrates stochastic TF fluctuations into its lick decision — **change with learning**?

**Hypothesis (H1, directional).** `τ` **grows** with learning: Naive behaviour is derivative-dominated / impulsive (short effective `τ`, reacting to transient pulses), Expert lengthens the integration window (longer `τ`, smoother accumulation). I.e. `τ_Expert > τ_Learning`.

**Null (H0).** `τ` is fixed across learning (a stable property of the circuit / stimulus statistics), and only other parameters (gain, bound) change — which would be the B0 result with B1 null.

**Why it's interesting (field gap).** The field has an **open, unreconciled disagreement** about the integration timescale on this very task family (0.27 s vs ~1 s). The standard assumption is that `τ` is a fixed property. BG_046's longitudinal design can make a claim almost nobody else can: **`τ` is itself learned**. This is the dedicated home for the integration-timescale question that B0 deliberately carved out (B0 holds the leak `λ` shared across stages; **B1 frees it**).

**On the cross-study discrepancy (what B1 can adjudicate).** Both headline numbers are **behavioural, not neural**: Orsolic's ~1 s is the slow-filter constant of a behavioural GP classification model (slow integrator + fast derivative); Khilkevich's ~0.27 s is an early-lick-triggered stimulus kernel. The tasks were essentially identical (same lab; same air-puff + block design), and neither number is read off the neural signal — so the gap is **not** a task difference and **not** the calcium-vs-ephys recording modality. By elimination it is most parsimoniously the **τ-estimator / parameterization** (Orsolic's hybrid slow+fast model vs Khilkevich's single kernel). Conveniently, **B1's Estimator 1 ≈ Khilkevich's method and Estimator 2 ≈ Orsolic's method**, so running both on one BG_046 dataset is a *direct test* of whether the gap is a method artifact (§5). **Modality caveat (future neural-τ only):** if BG_046 ever computes a *neural* τ (E-theme), a calcium estimate would be inflated by indicator kinetics but BG_046's ephys would be directly comparable to Khilkevich — neural-τ comparisons must match recording modality. (The behavioural `τ` estimated here is modality-immune.)

## 2. Scope

**In scope (T1, behaviour):**
- Estimate `τ` **per learning stage** by **multiple independent estimators** (model-free + GLM + optional model-based) and test whether `τ` grows.
- Triangulation: the headline requires the learning trend to **replicate across ≥2 estimators** (per the synthesis warning that τ is method-sensitive).
- State control + the evidence-driven-lick conditioning the two-route picture (B0) requires.

**Explicitly OUT of scope (cross-referenced):**
- **Neural integration timescale** (e.g. autocorrelation of the population evidence axis, neural TF-kernel extent) — a natural T1/T2 follow-up under the E-theme, not here. B1 is behavioural.
- **The DDM machinery itself** — owned by **B0**. B1 only *frees `λ` per stage* in the already-built B0 model as one of its estimators; it does not re-implement the accumulator.
- **Cohort generalization** — single subject; pooling BG_031/038/039 = F3.
- **Committing to a single `τ` number** — explicitly avoided; B1 reports the full set and the *trend*.

## 3. Data inputs

- **Sessions / stage axis:** `load_staging_manifest(qc_only=True)`; chronological, indexed by stage (and real day where the trajectory matters), per the B2 time-axis discipline.
- **TF stream (the stimulus):** the pre-planned realized TF time series — `trial.baseline_values` (pre-change) + the change-period design — converted to `e(t) = log2(TF(t)/TF_base)`, **truncated at the decision time** (shared with B0; reuse `visdetect.analysis.ddm.build_trial_evidence` once B0 lands, or its evidence extractor).
- **Lick times / outcomes:** `visdetect.analysis.behavior.get_trial_dataframe` (RT, change_time, outcome) — do not re-parse `reactiontimes`.
- **Per-trial behavioural-state labels:** a **pluggable input** (default GLM-HMM `load_hmm_assignments`; a self-tailored classifier is in development and may replace it — see B0 §3/§10). Used for the evidence-driven-lick conditioning and the state-composition control.
- **Existing lick-hazard GLM:** `analysis_suite/07_advanced/k_lick_hazard_glm.py` — its stimulus filter (slow exponential + fast derivative, with stage×time interactions) is **estimator 2**; reuse its fitted filters / cache, do not rebuild the GLM.
- **Constants:** `TF_FAST_THRESH_LOG2`, `TF_SLOW_THRESH_LOG2`, baseline update period (~50 ms) from `visdetect.analysis.constants`.

## 4. The model — `τ` estimated two ways now (→ three post-B0)

The integration timescale is method-sensitive, so B1 estimates it by **two independent estimators now** (a model-based third is a post-B0 addendum) and asks whether they *agree on the learning direction*.

- **Estimator 1 (PRIMARY, model-free) — lick-triggered TF kernel (reverse correlation).** Average the TF evidence `e(t−lag)` as a function of `lag` before the lick; a causal integrator yields a kernel elevated over the integration window, whose **decay/extent = effective `τ`** (fit an exponential or read the half-area lag). **Lick set:** *primary* = **engaged-state early licks (FAs)** — pure baseline integration, no change confound, the cleanest `τ` (depends on the pluggable state input); *complementary* = **Hits** — more data, but interpreted relative to the change step, which can dominate. Report both. Impulsive-state FAs are excluded (time-driven, B0 route 2 — they dilute the kernel). **Non-decision-time truncation:** stimulus within the sensorimotor delay `t0` before the lick is causally too late to have driven it, so exclude the final `t0` window (the kernel's rise-from-zero lag also *estimates* `t0`); for Hits use the window `[change_onset + sensory latency, lick − t0]`. Estimate `t0` from BG_046 itself (fast-lick / `ref` reflex-latency floor, or B0's fitted DDM `t0`) — **not** a literature constant, and not the integration window itself (circular). **Must be corrected for stimulus autocorrelation** (§6).
- **Estimator 2 (reuse) — lick-hazard GLM stimulus filter.** The existing GLM already carries a slow-exponential + fast-derivative TF filter with stage×time interactions. Extract the **stage-specific effective `τ`** from the fitted slow-exponential time constant and the slow-vs-fast weight ratio. Cheapest (infra exists); a within-model estimate — and, being regression-based, it is **inherently autocorrelation-corrected** (see §6), so it doubles as the corrected cross-check for Estimator 1.
- **Estimator 3 (post-B0 ADDENDUM — not in the first cut) — DDM leak `λ` per stage.** Once B0 has executed and its model is recovery-validated, free `λ` per stage (B0 holds it shared); `τ = 1/λ`, a generative-model estimate consistent with the spine. **B1 ships on Estimators 1 + 2 alone** (no pyddm, no B0 dependency); E3 is added later as a confirmatory third estimator.

**Headline = the trend, triangulated:** does `τ` increase Naive→Expert, and does that increase replicate across Estimators 1 and 2 (3 once available)?

## 5. Analysis & inference

- **Per-stage `τ`:** compute each estimator's `τ` for Learning and Expert (and Naive if not merged). Bootstrap over sessions (and trials) for CIs on each stage's `τ` and on `Δτ = τ_Expert − τ_Learning`.
- **Decision rule:** H1 supported if `Δτ > 0` with CI excluding 0 on the PRIMARY (Estimator 1), **corroborated in sign** by Estimator 2 (and 3 if run). A single-estimator effect that does not replicate is reported as *suggestive, method-dependent* (honouring the synthesis warning).
- **Where BG_046 lands:** report the absolute `τ` per stage against the literature bracket (0.27 s vs ~1 s) — does BG_046 span it across learning (which would *reconcile* the discrepancy as a proficiency effect)?
- **Method-artifact test (adjudicates the field discrepancy).** Because Estimator 1 ≈ Khilkevich's early-lick kernel and Estimator 2 ≈ Orsolic's slow+fast model, compare their `τ` on the *same* BG_046 data: if they diverge by ~the cross-study factor (~3–4×), the 0.27-vs-1 s gap is largely a **method artifact**, not biology — a publishable result independent of the learning question.
- **State control:** recompute Estimator 1 within matched state composition (or per state); confirm the learning trend is not a state-occupancy artifact (shared with B0 §6).

## 6. Caveats, confounds, honest scope

- **Stimulus-autocorrelation confound (the load-bearing one — B1's analog of B2's motor confound).** The lick-triggered kernel inherits any temporal autocorrelation in the TF stream itself, *even with no neural integration*. The kernel MUST be corrected: if TF updates are iid (~50 ms, white), the raw kernel ≈ the true filter; if autocorrelated, deconvolve / regress against the stimulus autocorrelation (Orsolic/Nienborg approach). **Verify the TF stream's autocorrelation structure at planning time** and apply the correction before interpreting any `τ`. Estimator 2 (the GLM) is regression-based and therefore **inherently autocorrelation-corrected** — a corrected Estimator 1 agreeing with Estimator 2 is the strong result.
- **Non-decision time, not neural motor-prep.** For a *behavioural* kernel the near-lick limit is the causal sensorimotor delay `t0` (handled by the §4 truncation), **not** B2-style neural motor contamination — we correlate stimulus with choice, not neural activity. The neural-motor concern applies only if B1 is ever extended to a neural kernel (out of scope).
- **Method divergence is expected, not a bug.** The whole premise (per the synthesis) is that estimators differ; B1's claim is about the *direction of change*, triangulated — not a single canonical `τ`. Report all estimates side by side.
- **Evidence-driven-lick conditioning (from B0).** Most impulsive FAs are time-driven (B0 route 2) and carry little TF kernel signal; computing Estimator 1 on *all* early licks dilutes it. Condition on Hits + engaged-state licks. This also makes B1 dependent on the same state input as B0.
- **State-composition confound.** As in B0: a state-mix shift across learning could masquerade as a `τ` change — control it.
- **Weak per-stage data / short series.** Kernel and GLM-filter estimates need enough evidence-driven licks per stage; if thin, pool to two stages (Learning/Expert) and report wide CIs honestly.
- **n = 1** → within-subject; cohort = F3.

## 7. Success criteria

- **Positive / informative:** `Δτ > 0`, CI excludes 0 on Estimator 1, **replicated in sign by Estimator 2** (and 3 if run), robust to the autocorrelation correction and to state control. Headline: "the integration timescale lengthens with learning — `τ` is a learned quantity," with the literature-bracket placement.
- **Negative:** `τ` flat or CI includes 0 across estimators → `τ` is fixed; learning acts on other parameters (the B0 knobs). Still publishable as a single-subject calibration of the 0.27-vs-1 s debate.
- **Inconclusive:** estimators disagree in direction, or CIs too wide at achievable data → report the method-dependence itself (which *is* the field's open problem) and the absolute estimates.

## 8. Deliverables

- **Library:** an integration-timescale module (e.g. `visdetect/analysis/integration_timescale.py`): `lick_triggered_kernel` (with autocorrelation correction), `kernel_tau` (extract τ), `glm_filter_tau` (read τ from the lick-hazard GLM filter), `triangulate_tau` (assemble per-stage estimates + bootstrap Δτ). Reuses B0's evidence extractor and `behavior.get_trial_dataframe`.
- **Tests:** recover a known `τ` from a simulated leaky integrator (kernel method); confirm the autocorrelation correction removes a planted stimulus-autocorrelation artifact; `Δτ` bootstrap detects a planted learning increase.
- **Script** (`analysis_suite/01_behavior/`): per-stage lick-triggered kernels overlaid; `τ`-by-stage for each estimator with CIs; the triangulation panel (do estimators agree?); placement vs the 0.27/1 s literature bracket; state-controlled rerun.
- **Stats CSV + cache:** per-stage τ per estimator, Δτ + CI, autocorrelation-correction diagnostics.
- Conventions: `setup_style()`/`save_figure()`; constants from `constants.py`; `del sess; gc.collect()`.

## 9. To resolve at planning time

- The TF stream's **autocorrelation structure** (is it iid ~50 ms or correlated?) — determines the kernel correction. BLOCKING for interpretation.
- The lick-hazard GLM's fitted-filter **extraction entry point** (cache vs refit) and how to read an effective `τ` from its slow-exp + fast-derivative parameterization.
- The **state-label entry point** (pluggable; default `load_hmm_assignments`).
- **Estimator 3 is deferred** to a post-B0 addendum (decided in review) — no action in the first cut; B1 ships on Estimators 1+2. Revisit once B0 has executed + recovery-validated.
- Minimum evidence-driven licks per stage for stable kernels; binning fallback.

## 10. Links

- **B0** (`2026-06-10-B0-ddm-learning-knob-design.md`) — the DDM whose leak `λ` is Estimator 3; B1 is the dedicated home for the timescale B0 holds fixed.
- **B2** (`2026-06-08-B2-responsiveness-leads-learning-design.md`) — same longitudinal stage axis and time-axis discipline.
- Landscape: `memory/question_landscape_jun2026.md` (B1, the spine).
- Lit: `paper-orsolic-2021-mesoscale-task-origin`, `paper-khilkevich-lohse-2024-brainwide`, `synthesis-batch07-sweep` (integration-timescale debate), `synthesis-batch01-foundations` (the "report all three, don't commit" verdict).
- Code: `analysis_suite/07_advanced/k_lick_hazard_glm.py`, `visdetect.analysis.behavior.get_trial_dataframe`, `visdetect.analysis.ddm` (B0 evidence extractor + model), `visdetect.suite.loader.load_hmm_assignments`.
