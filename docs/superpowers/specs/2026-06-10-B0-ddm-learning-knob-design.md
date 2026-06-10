# Design spec — B0: Which DDM knob does learning turn?

| | |
|---|---|
| **Question ID** | B0 (see `memory/question_landscape_jun2026.md`) |
| **Date** | 2026-06-10 |
| **Status** | SPEC-APPROVED (2026-06-10; two-route model + TF-driven drift + state control added in review) — ready for writing-plans |
| **Feasibility tier** | T1 (behavior-only; existing BG_046 trial data) |
| **Spine** | *How do mice learn to suppress impulsivity and increase sensitivity?* — B0 is the **direct decomposition** of the spine into mechanism |
| **Lit anchor** | Bogacz 2006 (`paper-bogacz-2006-decision-review`): DDM = optimal accumulator; **drift = sensitivity, threshold = speed-accuracy, starting-point = prior/hazard bias**. Orsolic 2021 (`paper-orsolic-2021-mesoscale-task-origin`): the DV **integrates the moment-to-moment stochastic TF fluctuations** (slow integrator + fast derivative/outlier detector), gated by **temporal expectation** (hazard); the change is a sustained TF-distribution shift, not a discrete cue |
| **Approach (decided)** | Generative DDM fit with **pyddm**, change-detection (single-bound) variant; per-stage fits + nested model comparison |

---

## 1. Scientific question & hypothesis

**Question.** Across BG_046's Naive→Expert trajectory, **which drift-diffusion parameter does learning move** — the drift rate (sensitivity), the bound height (caution/speed-accuracy criterion), or the starting-point/urgency (impulsivity bias)?

**Why it's the spine's keystone.** The user's organizing question — *learn to suppress impulsivity AND increase sensitivity* — is, in DDM terms, a claim about **two separable knobs**: sensitivity ↔ **drift**, impulsivity ↔ **starting-point/urgency**. B0 tests whether learning actually decomposes that way, or whether a single parameter (e.g. bound) explains the behavioral improvement. Every neural question downstream (A1 bound, A2/E1 evidence axis, C1 MOs-D2 brake) inherits its interpretation from which knob B0 says is moving.

**Hypotheses (directional, from the lit):**
- **H1 (sensitivity).** Drift rate `v` increases Naive→Expert (corroborates Marica striatal-responsiveness-rises; `paper-marica-2025-striatal-visual-responses-prelearning`).
- **H2 (impulsivity).** Starting-point/urgency bias toward early licking **decreases** Naive→Expert (corroborates the MOs-D2 FA-brake, Liu 2023; `paper-liu-2023-m2-striatum-false-alarm-suppression`).
- **H3 (caution, alternative).** Bound height `a` increases (more evidence required) — a competing single-knob account to be adjudicated by model comparison.
- **Null.** A single shared parameter set fits all stages (no parametric learning signature in behavior), or only `a` moves (improvement is pure caution, not sensitivity/impulsivity).

The *interesting* outcome is H1+H2 together (two knobs, two substrates); the *decisive* analysis is the nested model comparison that says which of {v, a, z/urgency} **must** vary by stage.

## 2. What this spec does and does NOT cover

**In scope (T1, behavior-only):**
- A generative single-bound DDM for the change-detection task, fit per stage.
- Identification of which parameter(s) carry the learning effect (nested model comparison + CIs).
- Parameter-recovery validation (detection tasks are weakly identifiable — see §6).
- HMM-state **control** of the learning result, plus a **bounded state-resolved route-mixture secondary** (§5).

**Explicitly OUT of scope (cross-referenced, not tested here):**
- **Neural linkage.** Whether drift maps to the population evidence axis (A2/E1) or the bound to the commitment ramp (A1) — B0 generates those predictions but does not test them. Keep neural data out.
- **Per-cell / cell-type DDM** — out; this is whole-behavior.
- **Full state×stage factorial decision dynamics** — only the *bounded* state-resolved secondary (§5) is in scope. A complete state-resolved DDM (every parameter free per state) is a candidate own question, overlapping D1 (state-conditioned psychometrics).
- **The integration timescale (leak `λ`).** B0 holds `λ` **shared across stages** and fits the drift **gain `v`** and **rectification shape `R`**, not the timescale. *Whether `λ` (the integration τ) changes Naive→Expert is B1's question* (`question_landscape_jun2026.md`). This boundary keeps the two specs separable for independent execution and protects B0's identifiability.
- **Cohort generalization** — single subject BG_046; pooling BG_031/038/039 is F3.
- **Replacing the lick-hazard GLM.** The existing `analysis_suite/07_advanced/k_lick_hazard_glm.py` is a *descriptive* discrete-time hazard model. B0 is a *generative* accumulator with mechanistically-named parameters. They are complementary; B0 must not duplicate or modify the GLM. (The GLM's temporal-spline / change-evidence decomposition is a useful cross-check on B0's urgency / drift terms — see §7.)

## 3. Data inputs

- **Sessions:** `load_staging_manifest(qc_only=True)` → stages (merged Naive→Learning; so **Learning** and **Expert**). Fits are **per stage**, pooling trials across that stage's sessions (with session-level robustness check, §5).
- **Per-trial fields** (from `visdetect.analysis.behavior`, which already extracts these): `trialoutcome`, `change_size`, `change_time` (per-trial onset, **varies**), `reactiontimes` (Hit RT, FA latency, generic `RT`), and a derived `response_time = rt + change_time`. Baseline duration per trial (Baseline_ON → change_time) governs FA opportunity and must be carried as a per-trial condition.
- **Trial typing (per `CLAUDE.md`):** go = `change_size > 1.0`; catch = `change_size ≈ 1.0`. **Two distinct FA notions** must be tracked separately and both mapped to the model's early-crossing term:
  - `fa` behavioral label = anticipatory lick **before** the change (impulsivity / baseline crossing).
  - SDT-FA = catch-trial `hit` (lick when no change) — also a baseline/no-drift crossing.
- **Outcome → model event mapping:**
  - Hit (go) → bound crossing at RT after change onset (first-passage).
  - Miss (go) → no crossing within the response window after change onset.
  - FA (`fa`, anticipatory) and SDT-FA (catch lick) → an **early crossing**, predominantly via route 2 (impulsivity/timing), occasionally route 1 (a real baseline fluctuation). The likelihood integrates `[0, lick_time]`.
  - CR (catch miss) → no crossing within the trial.
  - `abort` → **right-censored** at the abort time (neither route had crossed yet). *Planning decision:* if aborts are experimenter-terminated for reasons unrelated to the accumulator (disengagement), they may instead be excluded — flagged in §10.
  - `ref` (reflex / too-fast lick) → the fast-lick contaminant term (pyddm overlay) or excluded; not a decision crossing.
- **Per-trial TF stream (the drift input) — pre-planned, used only up to the decision.** The full TF sequence (baseline fluctuations **and** the change-period values) is **drawn in advance per trial** from the stimulus design; only *whether the change is reached* is stochastic (it isn't, on early-lick / FA / abort trials). So route-1 evidence `e(t) = log2(TF(t)/TF_base)` is recoverable from the planned trace — no `change_size`-statistics reconstruction needed. **CRITICAL — truncation:** the accumulator integrates `e(t)` only over `[0, decision_time]` (first lick for Hit/FA; response-window end for Miss; abort time for abort). Planned TF values *after* the lick are causally irrelevant and MUST be discarded — feeding them in would use the future to predict the past.
- **Per-trial behavioural-state labels** (engaged / impulsive / disengaged) — a **pluggable input**, NOT hardwired to any one classifier. Default source is the current canonical labeller (GLM-HMM, `visdetect.analysis.hmm`); a self-tailored state classifier is in development (separate workstream) and is expected to replace it. The analysis takes state as an injected per-trial column behind a single accessor, so the source swaps without touching the model/fit code. Required for the §6 control and §5 secondary.
- **Constants:** `CHANGE_SIZES = [1.25, 1.35, 1.5, 2.0, 4.0]`, `FA_RT_SPLIT = 3.0`, `TF_FAST_THRESH_LOG2 = 0.25`, `TF_SLOW_THRESH_LOG2 = -0.25` from `visdetect.analysis.constants` (the fast/slow thresholds define the rectification breakpoints in `R`). Evidence is `log2`-scaled per the project convention.

## 4. The model (pyddm change-detection variant)

A **single-bound** (one-choice-in-time) accumulator, decision variable `x(t)` aligned to **Baseline_ON** (`t = 0`), driven by **two routes to the bound** that an early lick can arise from *either* of:
1. a **sensory/evidence route** — TF-fluctuation-driven drift (the change-detection mechanism), and
2. an **impulsivity/timing route** — a time-dependent urgency/hazard drive that accumulates toward the bound **largely independent of the TF stream**.

**Why two routes (empirically motivated).** The robust gist — from BG_046 work and lab consensus — is that **a substantial share of early licks are *not* explained by the preceding TF evidence**; they look like a time/impulsivity phenomenon, not a sensory crossing. (An earlier binary stimulus-driven-vs-impulsive FA classification put this at κ≈0.02, but that specific result is held *loosely*; B0's Step 0b is the principled re-test of the gist, not a reliance on the old number.) A model that forces FAs to be baseline-fluctuation crossings would absorb an impulsivity effect into inflated noise or sensory drift — biasing the exact `v`-vs-`u` decomposition B0 exists to make. The two routes map cleanly onto the spine: **drift gain `v` = sensitivity** (route 1), **urgency/start `u`/`z` = impulsivity** (route 2). The change event is *not* a special cue — it is where the TF distribution mean shifts, making route-1 evidence sustained-positive; a Hit is route 1 (± route 2) crossing after the change, a typical FA is route 2 crossing early, and an FA *can* be route 1 (a real baseline fluctuation crossing).

**The route mixture is state-dependent (important — both a confound and a prediction).** The relative contribution of the two routes is expected to vary with behavioural state: in an **engaged** (less impulsive) state, early licks should be *more* TF-evidence-driven (route 1); in **impulsive/disengaged** states, *more* time-driven (route 2). This unifies the two-route model with the project's behavioural-state labels (currently GLM-HMM, but a replacement classifier is in development — state is a pluggable input, §3/§10): the **Impulsive** state ≈ high route-2 gain (or low bound), the **Engaged** state ≈ evidence-driven/route-1. It is the continuous, state-resolved version of the failed binary FA classification, and it is the neural echo of the "where was the population in 2D state space when it licked" framing (`scientific_context.md`). Consequences: (i) **confound** — state composition differs across learning, so it must be controlled or a state-mix shift will masquerade as a parameter change (see §6); (ii) **prediction** — a secondary analysis tests it directly (see §5).

- **Drift (two routes, per trial):** `drift(t) = v · R(e(t)) − λ·x(t) + u·h(t)`, where `e(t) = log2(TF(t)/TF_base)` is the instantaneous TF evidence (route 1) and `u·h(t)` is the time-driven impulsivity/urgency drive (route 2; `h(t)` = rising hazard/urgency profile). Specifically:
  - **`v` = drift gain = the SENSITIVITY parameter** (B0's H1 knob). `change_size`'s effect is *emergent* (bigger change → bigger sustained `e(t)`), not a separate multiplier.
  - **`R(·)` = rectification nonlinearity**, a structural variant selected by comparison (§5): symmetric linear (fast +, slow −, equal) vs half-wave rectified (fast +, slow → 0 = "slow pulses ignored") vs asymmetric gain (fast +g↑, slow −g↓). Orsolic predicts the rectified/asymmetric form wins.
  - **`λ` = leak** = temporal summation of consecutive pulses. **`1/λ` is the integration timescale, which is B1's question — so B0 holds `λ` shared across stages (fixed, or fit-once-shared); B1 owns "does `λ` change Naive→Expert" (see §2).**
- **`u·h(t)` = impulsivity/timing route (route 2, FIRST-CLASS — not a nuisance):** `u` = urgency gain = the **IMPULSIVITY parameter** (B0's H2 knob), `h(t)` = a rising hazard/urgency profile (Orsolic temporal expectation). This route drives the bound on elapsed time *independent of the TF stream* and is what produces most early licks (per the κ≈0.02 finding). Equivalent parameterizations to compare (§5 Step 0): rising-drift `u·h(t)` vs a collapsing bound `a(t)`. It is the principal confound for the starting-point `z` — both must be in the model or `z`/`v` are biased.
- **Bound:** `a` (the **caution / speed-accuracy** parameter); optionally collapsing.
- **Starting point:** `x(0) = z` (the **impulsivity / prior bias** parameter; `z` closer to bound → earlier crossings).
- **Non-decision time:** `t0` (motor/sensory latency overlay).
- **Noise:** fixed `σ = 1` (scaling convention; `v` and `a` measured in those units).
- **FA mixture:** an early-crossing / contaminant term for very-fast licks (`ref`-like) handled by a pyddm overlay if needed.

**First-passage outputs:** Hit RT density (post-change crossings), Miss probability (no crossing in window), and early-crossing probability/timing (FA family, route-2-dominated). pyddm provides these via custom `Drift`, `Bound`, `InitialCondition`, and `Overlay` subclasses with the **per-trial TF evidence trace `e(t)`** (route-1 input), `change_time`, `change_size`, and `baseline_dur` as per-trial **conditions**.

## 5. Fitting & inference

- **Step 0 — structural selection (do once, before cross-stage fits).** Fix the model *form* by comparing rectification variants `R` (symmetric / half-wave / asymmetric-gain) and the urgency form (rising-drift vs collapsing-bound) on the pooled data, by CV log-likelihood. Carry the winning structure into the cross-stage step. (This is where the "fast pulses drive drift, slow pulses maybe ignored" hypothesis is tested.)
- **Step 0b — route attribution for early licks (a real scientific test, not just bookkeeping).** Compare a **TF-only** model (route 2 off → early licks must be sensory crossings) vs the **two-route** model, on the early-lick/FA data. BG_046's κ≈0.02 finding predicts the two-route (impulsivity-dominated) model wins decisively; the fit also quantifies *what fraction of early licks each route explains*. If TF-only were somehow to win, that would overturn the prior FA work — either way it is reportable. Pre-change `e(t)` aligned to each early lick is the regressor that adjudicates it.
- **Per-stage fits:** fit the selected model separately to Learning and Expert trial sets (pooled across that stage's sessions), with `λ` shared across stages (§2). pyddm differential-evolution or BADS optimizer on the trial-wise likelihood (RT distributions for Hits, plus Miss / FA / CR probabilities).
- **The decisive analysis — nested model comparison.** Fit a ladder of models that share all parameters across stages except one:
  - `M_shared` (all params shared across stages),
  - `M_v` (only `v` differs by stage),
  - `M_a` (only `a`),
  - `M_z/u` (only starting-point/urgency),
  - `M_full` (all differ).
  Rank by **AIC/BIC and cross-validated (held-out-trial) log-likelihood**. **The minimal model that fits as well as `M_full` names the knob(s) learning turns** — that is the headline result.
- **Uncertainty:** bootstrap over sessions (and trials) to get CIs on each stage's parameters and on the cross-stage differences `Δv, Δa, Δz`.
- **Psychometric/chronometric checks:** overlay model-predicted vs empirical P(lick) and RT-by-`change_size` per stage (goodness-of-fit, not just likelihood).
- **Session-level robustness:** refit `v, a, z` per session (where trial counts allow) and correlate with session index / d′ — does the per-session parameter trajectory agree with the per-stage story (and connect to B2's learning curve)?
- **Secondary — state-resolved route mixture (tests the engaged→TF prediction).** Using per-trial HMM state labels (`hmm.py` / cached state assignments), either (a) refit conditioned on state or (b) let route gains depend on state, and test whether the **route-1 (TF) share of early licks is higher in Engaged than in Impulsive** states. This is the continuous, principled redo of the binary FA classification, and links B0 to D1 (state-conditioned psychometrics). Bounded: full per-(stage×state) fits are deferred if trial counts are thin — fall back to a state covariate.

## 6. Identifiability & caveats (load-bearing)

- **Detection tasks are weakly identifiable, and the TF-driven drift adds parameters.** Unlike 2AFC, there are no error-RTs to anchor the drift-vs-bound trade-off; `v` and `a` are partially degenerate (their ratio sets accuracy; only the **RT distribution shape** separates them). The TF-fluctuation drift adds gain `v`, rectification shape `R`, and leak `λ` on top of `a, z, u, t0`. **Therefore parameter recovery is mandatory and is the core TDD target (§9):** simulate trials from known `(v, R, λ, a, z, u, t0)` using the real per-trial TF streams, refit, and confirm recovery within tolerance *before* trusting any fit to real data. Report a recovery figure.
- **Reduce the free-parameter load by fixing what we can.** To keep the fit identifiable: fix `λ` (shared, §2) — seed it from the lick-hazard GLM's stimulus-kernel timescale or Orsolic's published integration window; fix `t0` from the fast-lick (`ref`) latency floor; constrain `R` to the structurally-selected form (§5 Step 0). Only `v, a, z, u` then vary in the cross-stage comparison.
- **Urgency ↔ starting-point confound.** Temporal expectation (rising hazard) and a biased starting point both produce earlier licking; they are separable only via the *time course* of FAs (late-rising vs flat). Fit both; report their correlation/identifiability.
- **change_time varies per trial** → drift onset must be per-trial-conditioned; pooling without conditioning on change_time biases everything.
- **Two FA notions** (`fa` vs SDT-FA) — both inform the baseline-crossing term but must be counted correctly against their denominators (`fa` per all trials in the baseline window; SDT-FA per catch trials).
- **State-composition confound (MUST control).** Engaged/Impulsive trial proportions differ across learning, and the route mixture is state-dependent (§4) — so pooling per stage lets a *state-mix shift* masquerade as a change in the impulsivity parameter `u`/`z`. Control by conditioning fits on HMM state, including state as a covariate, or matching state composition across stages; report the learning result both raw and state-controlled. (If they diverge, that itself is the finding: "learning" was partly a state-occupancy change.)
- **n = 1** → within-subject; cohort = F3.
- **pyddm dependency** — confirm it installs on Windows/Python `.venv` (§10); pin version; it is the only new dependency.

## 7. Cross-checks against existing machinery

- **Lick-hazard GLM** (`07_advanced/k_lick_hazard_glm.py`): its time-spline ≈ B0's urgency/temporal-expectation; its change-evidence × log2(change_size) term ≈ B0's drift; its baseline hazard ≈ B0's starting-point. The GLM coefficients across stages provide an **independent, assumption-light corroboration** of the DDM model-comparison verdict. If the GLM says "change-evidence slope grows, baseline hazard falls" and the DDM says "Δv > 0, Δz < 0," they agree — strong. Divergence is diagnostic.
- **B2 link:** B0's per-session `v` trajectory should rise on roughly the same training axis as B2's neural decodability — a behavior/neural convergence worth reporting jointly.

## 8. Success criteria

- **Positive / informative:** the nested comparison selects a minimal model with `Δv > 0` (route-1 sensitivity ↑) and/or the impulsivity route (`Δu`/`Δz`) reduced Naive→Expert, CIs excluding 0, with adequate goodness-of-fit (model reproduces empirical RT + psychometric + FA-timing curves), **and** clean parameter recovery. The headline is a one-sentence statement of which knob(s) moved. The two-route structure makes the canonical H1+H2 result — sensitivity route up, impulsivity route down — directly readable.
- **Negative:** only `a` varies, or `M_shared` wins → learning leaves no separable drift/start signature in behavior (still publishable; reframes the spine as a threshold phenomenon or a purely neural one).
- **Inconclusive:** parameter recovery fails or `v`/`a` non-identifiable at available trial counts → report the identifiability limit; fall back to the hazard-GLM reparameterization (§7) as the assumption-light read.

## 9. Deliverables

- **Library:** `src/visdetect/analysis/ddm.py` — pyddm model builders (custom Drift/Bound/IC/Overlay for the change-detection variant), a `fit_stage()` routine, the nested `compare_models()` driver, and a `recover_parameters()` simulation helper.
- **Tests:** `tests/analysis/test_ddm.py` — **parameter recovery** (simulate from known params using real TF streams → refit → recover within tolerance) as the primary test; model-builder sanity (drift tracks instantaneous TF evidence — positive on fast pulses, ≤0 on slow per the selected `R`; baseline fluctuations alone can cross the bound = FA; bigger `change_size` → higher Hit rate / shorter RT); structural selection recovers the true `R`; cross-stage comparison selects the true generating knob on simulated stage differences.
- **Script:** `analysis_suite/01_behavior/<letter>_ddm_learning_knob.py` — per-stage fits + comparison + bootstrap, producing:
  - **Figure** (`fig0N`): (A) RT distributions data vs model per stage; (B) psychometric P(lick) vs change_size data vs model; (C) fitted `v, a, z, u, t0` across stages with bootstrap CIs; (D) model-comparison ΔAIC/CV-LL bar (incl. Step 0b TF-only vs two-route); (E) parameter-recovery scatter (recovered vs true); (F) route-1 (TF) share of early licks by HMM state (Engaged vs Impulsive) — the state-dependence prediction.
  - **Stats CSV:** per-stage params + CIs, Δparams, model-comparison table, recovery metrics.
  - **Cache:** fitted-parameter + bootstrap cache.
- Conventions: constants from `constants.py`; `load_staging_manifest()`; `setup_style()`/`save_figure()`; `del sess; gc.collect()` if sessions are loaded (likely only the trial tables are needed — prefer loading behavior, not spikes).

## 10. To resolve at planning time (writing-plans)

- **pyddm install check** on the project `.venv` (Windows); pin the version; confirm the import name (`pyddm` / `ddm`).
- **TF-stream recovery (data check).** The full per-trial TF trace (baseline + change-period) is **planned in advance** from the stimulus design, so route-1 `e(t)` should be recoverable up to the decision time. Confirm where the planned trace lives: `trial.baseline_values` (pre-change, used by `tf_pulse._collect_pulses`) plus the change-period design values — verify the latter are stored/derivable per trial (they exist as *plan* even on trials where the change was never reached, but are used only up to the lick). Confirm the update period (~50 ms) and align to the pyddm integration grid (`dt`). Implement the `[0, decision_time]` truncation (§3) so post-lick planned values never enter the likelihood.
- **Abort handling decision.** Treat `abort` as right-censored at abort time, **unless** aborts are experimenter-terminated for accumulator-unrelated reasons (disengagement) — inspect why aborts end and decide censor-vs-exclude before fitting.
- **State labels are a pluggable input.** Default to the current canonical classifier (GLM-HMM); a self-tailored classifier is in development (separate workstream) and may replace it. Implement state as an injected per-trial column behind one accessor (e.g. `load_state_labels(session) -> Series`), so swapping the source touches no model/fit code. Confirm the current entry point at planning time.
- Exact behavior-table entry point (is there a cached per-trial behavioral DataFrame, or build from `sess.trials`?) — reuse `behavior.py` extraction, do not re-parse `reactiontimes`.
- Urgency parameterization: rising-drift `u·t` vs collapsing-bound — pick one as default, test the other as a model-comparison variant.
- Response-window length for the Miss definition (from constants / task params).
- Statistician knobs: AIC vs BIC vs CV-LL for the headline; bootstrap scheme (session vs trial vs hierarchical); recovery tolerance; minimum trials/stage for stable fits. → hand to **Research Statistician** during planning.

## 11. Links

- Landscape: `memory/question_landscape_jun2026.md` (B0, and the spine)
- Lit: `paper-bogacz-2006-decision-review` (DDM knobs), `synthesis-phase3-theory` (drift=sensitivity / threshold=SAT / start=prior), `paper-orsolic-2021-mesoscale-task-origin` (temporal-expectation hazard), `paper-marica-2025-striatal-visual-responses-prelearning` (H1), `paper-liu-2023-m2-striatum-false-alarm-suppression` (H2)
- Project: `memory/question_landscape_jun2026.md`, sibling spec `docs/superpowers/specs/2026-06-08-B2-responsiveness-leads-learning-design.md` (per-session `v` ↔ B2 neural curve)
- Code: `visdetect.analysis.behavior` (RT/outcome extraction), `visdetect.analysis.constants` (CHANGE_SIZES, FA_RT_SPLIT, TF thresholds), `visdetect.analysis.hmm` / `hmm_downstream` (per-trial state labels for §5/§6), `analysis_suite/07_advanced/k_lick_hazard_glm.py` (independent cross-check)
- Related question: **D1** (lapses gain-vs-bias / state-conditioned psychometrics) — the state-resolved secondary (§5) is the behavioural-model bridge to it
- Convention: `memory/science_spec_corpus_convention.md`
