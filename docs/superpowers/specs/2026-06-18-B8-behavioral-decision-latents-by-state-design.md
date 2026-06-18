# Design spec — B8: Behavioral decision-latents decomposed by state

| | |
|---|---|
| **Question ID** | B8 (proposed; confirm/rename against `memory/question_landscape_jun2026.md`) — absorbs the *behavioral* half of D1 (lapses gain-vs-bias) |
| **Date** | 2026-06-18 |
| **Status** | SPEC-DRAFT (brainstormed 2026-06-18; awaiting user review → writing-plans) |
| **Feasibility tier** | T1 (behavior-only; existing BG_046 trial data + the new state-labeler tags) |
| **Spine** | *How do mice learn to suppress impulsivity and increase sensitivity?* — the **behavior-first** deliverable of the post-TF-null direction (`docs/science/2026-06-17-post-tf-null-research-direction.md`, §6 step 1). Produces the per-trial latents the neural phase regresses against. |
| **Lit anchor** | Bogacz 2006 (DDM knobs: drift=sensitivity, bound=SAT, start=prior); Orsolic 2021 (temporal-expectation **hazard** gates a TF integrator); Marica 2025 (striatal/sensory responsiveness rises with learning → sharpness↑); Liu 2023 (MOs→D2 FA-brake → itchiness); Kepecs / Lak / Masís (RT-variability ↔ confidence, speed-accuracy over learning); Urai 2019 (single-trial DDM latents via regression). See `memory` synthesis batches. |
| **Approach (decided)** | **Two-step, descriptive-first** (brainstorming decision, 2026-06-18): Step 1 = robust per-(session×mood) axis decomposition that carries the science; Step 2 = a *minimal, recovery-gated* regression-accumulator that emits the per-trial latents. Build anew; B0 is **reference-only**. |

---

## 0. Working style — plain language + save every figure

Two standing requirements for this work (and a general user preference, [[feedback-plain-language-and-save-figures]]):
- **Explain in plain language alongside the formal terms.** Keep the jargon for rigor, but every concept — in this spec, the plan, code docstrings, and figure captions — also gets a one-line plain-English gloss (e.g. *drift / "sharpness" = how clearly the mouse can tell the change happened; criterion / start-point / "itchiness" = how trigger-happy it is before real evidence; urgency / "timing" = how strongly it expects the change right now*). The author should be able to read any output and explain it in a talk without decoding notation.
- **Every analysis step saves an inspectable figure.** No step is internal-only: each computation (the three descriptive dials, the two hazards, RT-variability, the bias-not-gain test, parameter recovery, latent distributions) writes a **labelled, presentation-ready PNG via `save_figure()` with a plain-language title + caption**, so results can be eyeballed and dropped straight into presentations. Figures are enumerated per-step in §10.

## 1. Scientific question & hypothesis

**Question.** Decompose BG_046's learning and within-session behavioral states onto **three interpretable decision dials**, *per trial*:

- **Sharpness** (drift / sensitivity) — how well the mouse tells the stimulus changed.
- **Itchiness** (criterion / start-point) — how trigger-happy it is *before* real evidence.
- **Timing** (urgency / temporal expectation) — how strongly it expects the change around *now*.

Then answer: **which dial does *learning* turn, and which dial do *states* (Impulsive vs StimSens) load on?** And deliver a **per-trial latent table** of the three dials, for the later neural-regression phase.

**Falsifiable thesis (from the post-TF-null direction).**
- **Learning** primarily raises **Sharpness** (sensitivity), with **Timing precision** sharpening (licks migrate toward the expected change time) and a secondary caution increase.
- **States load on the Itchiness/Timing axis, NOT Sharpness.** Impulsive = liberal/low-criterion (trigger-happy → *looks* hypersensitive but is bias); StimSens = conservative/high-criterion. **Decisive bias-not-gain test (Step 1, mood-split psychometrics):** expert-Impulsive shows *more responses to weak/sub-threshold changes* (apparent hypersensitivity) but with **d′ flat-or-down and NO leftward psychometric-threshold shift** — extra hits *bought with* extra FAs (criterion), not genuine discrimination gain (drift). A true leftward threshold shift with d′↑ would falsify the thesis (gain account). This is the behavioral half of D1.
- **Temporal expectation is the organizing variable** — the lick-hazard peak migrates toward, and sharpens at, the censored **change-onset hazard** as learning proceeds.

**Hypotheses (directional):**
- **H1 (learning = sharpness).** Sharpness rises Naïve→Expert: **large changes mastered first, small changes later** (the psychometric matures high-change-first); RT-variability shrinks, *more and earlier for large changes* (drift-precision signature; Marica). 
- **H2 (state = itchiness).** Impulsive vs StimSens differ on criterion/start-point and urgency timing, **not** on sharpness (bias-not-gain; consistent with the existing neural-validation result — engaged StimSens = lower-gain change response + slower RT than trigger-happy Impulsive, `memory/state_labeler_neural_validation_jun2026`).
- **Null / alternative.** Sharpness also separates the moods (gain account), or learning is pure caution (only the bound moves), or the latents are non-identifiable at available trial counts (→ ship descriptive layer, §9).

## 2. What this spec does and does NOT cover

**In scope (T1, behavior-only):**
- Step 1 descriptive axis decomposition (sharpness/itchiness/timing) per **(session × mood)**, traced on the learning axis.
- The censored **change-onset hazard** + **lick hazard**, reimplemented cleanly in the library (not the old script — see §8).
- Step 2 minimal **regression-accumulator** emitting the per-trial latents, **gated on parameter recovery** at the real long-baseline regime.
- The cached **per-trial latent table** (§5) — the deliverable.
- **All BG_046 sessions that have behavior + a state-tag file** — *not* only the QC/d′-filtered manifest. Low-d′, near-entirely-Impulsive early-learning sessions are kept on purpose as a **de-sculpted reference** (§3).
- State decomposition via the **new state labeler**: main fits on **Impulsive vs StimSens**; **Disengaged** reported separately; **Abort excluded** for now.

**Explicitly OUT of scope (cross-referenced, not done here):**
- **Neural data / regression.** This spec *produces* the latents; wiring them to spikes is a downstream spec (the direction's §6 step 3). No spike data is loaded.
- **The integration timescale / leak `λ`.** Held fixed & shared; *whether `λ` is learned is B1's question*.
- **Cell-typing / D1-D2.** None here.
- **Cohort pooling** (BG_031/038/039) — F3.
- **Formal confidence modeling.** RT-variability is used as a sharpness cross-check and its confidence link is *noted* (Kepecs); a confidence model is its own question.
- **Re-running / mutating B0.** B0's `ddm.py` is imported for reusable helpers only; its per-stage findings are not assumed.

## 3. Data inputs

- **Sessions — run on ALL *valid* sessions, not the QC manifest, via a two-tier filter.** B8's question is *how the dials move with learning*, so gating on performance would **condition on the outcome** and discard the most informative early sessions. But "all sessions" must mean *all valid task recordings*, not literally everything — d′ in the old gate does double duty ("not discriminating" = signal for us; "bad recording" = still exclude). So replace the single d′ gate with:
  - **Tier 1 — data-integrity floor (hard exclude):** keep a session only if it is a valid task recording — task ran, stimulus delivered, a minimum total-trial count, and a **state-tag file exists with adequate labeler coverage/confidence**. Drops technically-broken sessions *regardless of performance*.
  - **Tier 2 — performance & comprehension as analysis *dimensions*, not gates:** keep low-d′ sessions; carry **per-session d′ as a *continuous* learning-axis covariate** (chronological via `parse_session_date` / `chronological_sort`); use the **`comprehension_flag`** to separate pre-comprehension naïve (analyzed separately, §7); enforce a **per-(session×mood) minimum-trial threshold** for stable cells.
  - Genuinely checked-out trials self-exclude at *trial* granularity (mood = Disengaged/Abort). Keep the `load_staging_manifest(qc_only=True)` subset as a **robustness comparison** (does the story hold on the clean subset?).
- **Prerequisite — labeling coverage (data dependency).** Only **27 of ~45 BG_046 sessions are state-tagged today**; the **~18 untagged are predominantly the earliest (Jun–Jul 2025) low-performance sessions** — *exactly the de-sculpted early-learning references we want to add*. So full-coverage B8 **requires first extending the state labeler to the untagged sessions** (apply the trained rule; `scripts/state_labeling/`). Step 1 can run on the 27 tagged sessions immediately while that completes. Treat the naive-session labels as provisional pending the reliability check in §7.
- **Per-trial fields** (via `visdetect.analysis.behavior`, reuse — do not re-parse): `trialoutcome`, `change_size`, `change_time` (**planned** onset, *reached only on Hit/Miss*), `reactiontimes` (Hit RT, FA latency), `baseline_values` (the TF stream), `n_seen`. Trial typing per `CLAUDE.md`: go = `change_size>1.0`, catch = `change_size≈1.0`; `fa` = anticipatory lick (≠ SDT-FA = catch-trial `hit`).
- **Per-trial state labels (new labeler):** `data/cache/state_tags/BG_046/{session}.csv` — columns `trial_idx`, `state_label` ∈ {Impulsive, StimSens, Disengaged, **Abort**}, `state_confidence`, soft `p_state_*`. Accessed behind **one pluggable accessor** (so the source can swap without touching model code). **Main fits use Impulsive vs StimSens.** **Disengaged** is summarized separately (reported, not in the main fits — decision-axis estimates would be contaminated by checked-out, near-lickless trials). The **`Abort` mood is excluded entirely for now** (note: the *labeler* state `Abort` is distinct from the trial-outcome `abort`; both are dropped here).
- **Outcome → event mapping** (aligned to Baseline_ON, `t=0`): Hit → bound crossing at `change_time+RT`; `fa` → early crossing at FA latency; Miss → no crossing within response window (censored at `change_time + RESPONSE_WINDOW`); catch `hit` (SDT-FA) → early/baseline crossing; CR (catch miss) → no crossing; `abort`/`ref` → excluded (or right-censored — resolve at planning, §11). Likelihood/hazard integrates `[0, decision_time]` only; **planned TF values after the decision are discarded** (no future-to-past leakage).
- **Constants** from `visdetect.analysis.constants`: `CHANGE_SIZES`, `FA_RT_SPLIT`, `TF_FAST_THRESH_LOG2`, `TF_SLOW_THRESH_LOG2`, `DEFAULT_BIN_SIZE`. Evidence is `log2`-scaled. Integration grid **`dt = 0.05 s`** (the verified TF update period — see §7 TF-sampling).

## 4. Method — two steps

### Step 1 — descriptive axis decomposition (assumption-light; carries the science)

For each **(session × mood)** cell (mood ∈ {Impulsive, StimSens}; Disengaged reported separately; Abort dropped), over **all tagged sessions** (§3), compute sturdy readouts straight from behavior, reusing `behavior.py` (SDT/psychometrics):

| Dial | Readout(s) |
|---|---|
| **Sharpness** | psychometric P(detect) vs `change_size` (slope + threshold); d′ per change-size; **Hit-RT mean *and variability* (SD/CV/IQR) per change-size** — H1 predicts variability shrinks *more & earlier for large changes* (drift-precision; confidence link noted, Kepecs) |
| **Itchiness** | SDT criterion *c*; FA-rate per opportunity; baseline lick-hazard; *how early* FAs fall |
| **Timing** | mode & spread of the lick-time distribution (FA + Hit licks) **relative to the censored change-onset hazard** (below); lick-hazard time-spline peak/precision |

**The two hazards (resolves the "which change times" question).** Both are **censored survival** estimates on `dt=0.05` bins from Baseline_ON: at elapsed time *t*, each trial is "at risk" only while still running (no lick, no change, not aborted), and drops out (right-censored) at its lick/abort time.
- **Change-onset hazard** — event = the change actually occurring (observed on Hit/Miss; **right-censored** on `fa`/`abort`, so a planned-but-unreached change, e.g. 15 s on a trial that FA'd at 3 s, contributes to the at-risk denominator only up to 3 s and is **never counted as an event**). This is the *learnable* "when does the change come" structure — faithful to what the mouse experienced. (Optionally also compute the naïve all-planned-times curve as a contrast, to show the selection effect.)
- **Lick hazard** — event = first lick; the mouse's actual behavior.
- **Temporal expectation** = how the lick-hazard peak migrates toward / sharpens at the change-onset hazard across learning.

Each readout is **split by mood** and **traced along the learning axis** (per-session with d′ overlay; graceful fallback to session-bins or the 2 coarse stages if per-session is too noisy). Output: the **headline figure + a "which dial moves with learning / which dial separates the moods" table**, on solid ground *before* any generative fit.

### Step 2 — minimal regression-accumulator (emits per-trial latents; recovery-gated)

A single-bound change-detection accumulator, decision variable `x(t)` aligned to Baseline_ON, reusing B0's per-trial evidence builder (`ddm.build_trial_evidence`, with the TF-sampling fix and `[0,decision_time]` truncation). Within a learning anchor, parameters are **functions of state** (so they vary per trial via the trial's mood), fit with state as a per-trial regressor, ranges seeded/bounded from Step 1. The **learning trajectory** is obtained via the **expert-anchored, backward-seeded fits of §6** (not a single pooled learning regressor) — so "which dial moves when" reads as the dial's drift away from its expert value as fits walk back through training:

- **Drift** `drift(t) = vᵢ·R(e(t)) − λ·x + urgencyᵢ(t)`
  - `vᵢ` (**sharpness**) `= v₀ + v_state[stateᵢ]` (re-fit per learning anchor, §6); `change_size` enters *emergently* via `e(t)`.
  - `λ` (leak) **fixed & shared** (B1 owns the timescale; protects identifiability).
  - `urgencyᵢ(t)` (**timing**) = a **temporal-expectation-shaped profile peaked near the expected change time** (peak seeded from Step 1's change-onset hazard), amplitude `u₀ + u_state[stateᵢ]`. *This is the upgrade over B0's plain `u·t`.*
- **Start-point** `zᵢ` (**itchiness**) `= z₀ + z_state[stateᵢ]`.
- **Bound** `a`, **non-decision** `t₀` — shared (or a single learning term); kept minimal.

**Per-trial latents read out two ways** — the fitted parameter (`vᵢ, zᵢ, urgencyᵢ`) *and* a genuinely trial-specific realized quantity (evidence-integral at decision, urgency-value at decision) — so the table is honest about regression-varying vs. trial-specific.

**Recovery gate.** Mandatory parameter recovery at the **real long-baseline regime** (`dt=0.05`, `T_dur ≈ trial-duration p99`, real per-trial TF streams) — the check B0 skipped. Pass → trust the generative latents. Fail → fall back to Step-1 descriptive proxies as the latent table and report the ceiling (§9).

## 5. The deliverable — per-trial latent table schema

```
# one row per trial, ALL tagged sessions; Disengaged rows flagged (reported separately), Abort dropped
session_name, trial_idx, session_dprime, comprehension_flag,  # learning axis (continuous d′) + pre/post-comprehension
state_label, state_confidence, trial_in_session,              # new-labeler mood + within-session position (satiety covariate)
outcome, change_size, change_time_planned, change_reached,    # trial geometry
decision_time, lick, censored,
# --- the three latents (Step 2 generative) ---
sharpness_drift,                 # v_i
itchiness_startpoint,            # z_i
timing_urgency_at_decision,      # urgency_i(decision_time)
evidence_integral_at_decision, expected_change_time, lick_minus_expected,
# --- Step 1 descriptive cell-scores joined on (session,mood) for construct-validity cross-check ---
sharpness_psy_slope, rt_cv_by_cs, criterion_c, fa_rate_cell, hazard_peak_cell
```

Cached to `analysis_suite/cache/decision_latents_by_state.csv` (the regressor set the neural phase consumes).

## 6. Anchoring strategy (expert-first, backtrack) — §6 step 2 of the direction

The per-session *descriptive* computation is order-independent, but the expert anchor matters in three ways:
1. **Identifiability (Step 2 fit order).** Fit the accumulator on **expert / high-d′ sessions first** (most identifiable regime), then fit earlier sessions **seeded/regularized from the expert anchor**, letting parameters relax backward — buys down the weak-identifiability risk that hit B0 in the naïve regime.
2. **Reference template.** "Which dial moves *when*" is read as **distance-from-the-expert-template**: define each dial's mature (expert) value, walk backward, see which dial departs first/most. The expert **change-onset hazard** is the "learned expectation" reference for the earlier lick-hazards.
3. **Shared-axis hook.** The downstream "expert-Impulsive bout ≈ mid-learning regime" test (direction §7) is defined relative to the expert template; flagged here as a hook, not executed (it needs the neural phase).

## 7. Confounds & identifiability (load-bearing)

- **State-composition confound** — mood proportions shift across learning; a mood-mix change could masquerade as a dial change. The new labeler gives per-trial mood, so we condition directly and report learning effects **both raw and state-resolved** (divergence is itself a finding).
- **State-label reliability on naive sessions (load-bearing).** The whole decomposition leans on the moods, but the labeler was calibrated (κ≈0.73) on QC sessions; the early/naive sessions we now add are **out-of-distribution** (pre-comprehension behavior differs), so their labels are the *least validated* exactly where we most use them. Mitigate: weight/gate by `state_confidence`; sanity-check the labeler's mood proportions & exemplar trials on the added sessions before trusting them; if labels look unreliable there, fall back to treating those sessions only at the coarse behavioral level (no mood split). Do **not** silently trust naive-session moods.
- **Detection-task weak identifiability** — no error-RTs to anchor drift-vs-bound; recovery is mandatory (§4 Step 2), Step 1 constrains the fit, expert-anchoring stabilizes it.
- **Urgency ↔ start-point trade-off** — both produce earlier licks; separable only via the FA *time-course* (late-rising vs flat). Fit both; report their correlation.
- **Two impulsivities** — pre-comprehension naïve (the mouse hasn't yet learned lick→reward→window; exploratory licking) vs post-comprehension lapse (knows-but-can't-hold; *the thesis target*). Because we now **include** low-d′ sessions (§3), this is handled by a **per-session `comprehension_flag`** — a learned-the-rule marker (e.g. first session with reliable easy-change hits / d′ crossing a low threshold; definition resolved at planning) — **not** a hard drop. Pre-comprehension sessions are analyzed as a **labeled reference** (the de-sculpted extreme), and the main thesis claims are read on post-comprehension sessions; report both.
- **Within-session position / satiety** — Impulsive bouts cluster at session start (hungrier, "not in the zone"). Conditioning on per-trial mood absorbs most of this, but carry **within-session trial position** as a covariate / report itchiness vs. trial-in-session, so a satiety gradient isn't misread as a learning or pure-state effect.
- **TF-sampling correctness** — **resolve the 50 ms update period against the data before any TF term enters** (the old script's `BASELINE_STRIDE=3` is suspect per `memory/tf_fluctuation_50ms_vs_constant`); use `dt=0.05`, keep legit 0.25-octave thresholds.
- **change_time varies per trial** → per-trial drift-onset conditioning (done); pooling without it biases everything.
- **Two FA notions** (`fa` vs SDT-FA) counted against correct denominators.
- **n = 1** — within-subject; cohort = F3.

## 8. Cross-checks against existing machinery

- **`analysis_suite/07_advanced/k_lick_hazard_glm.py`** — sound discrete-time survival *skeleton* (50 ms bins, censoring, spline "learned clock", survival `P(lick)=1−Π(1−hₜ)`), but **NOT reused as-is**: it is hardwired to the old GLM-HMM states, uses the suspect `BASELINE_STRIDE=3` TF subsampling, fakes the change-period TF, uses heuristic `abs(eta)` attribution, and couples to neural CD. We **reimplement the hazard cleanly in the library** (new labeler, verified TF sampling, behavior-only) and **regression-check** the temporal hazard against this script.
- **B0 `ddm.py`** — reuse `build_trial_evidence`, the pyddm API contract, and the truncation rule; do not mutate.
- **`behavior.py`** — reuse SDT / psychometric / criterion computations for Step 1.

## 9. Success criteria

- **Step 1 (must-have, ships regardless):** a clean *which-dial-does-learning-turn / which-dial-separates-the-moods* table + figure, with the falsifiable prediction tested — **moods load on itchiness/timing with Sharpness ≈ flat** (bias-not-gain); **learning raises Sharpness + Timing precision**; **RT-variability shrinks more/earlier for large changes**.
- **Step 2 (the deliverable):** per-trial latent table **with validated recovery**; latents show construct validity against the Step-1 cell scores.
- **Negative/inconclusive:** recovery fails at long-baseline → ship Step-1 descriptive latents + per-cell scores as the table and report the identifiability ceiling. Still a usable regressor set and a real result.

## 10. Deliverables

- **Library:** `src/visdetect/analysis/decision_latents.py` — Step-1 axis computations (reusing `behavior.py`), the censored change-onset + lick hazards, the constrained regression-accumulator + recovery, the latent-table assembler, and the pluggable new-labeler state accessor. Imports reusable helpers from `ddm.py`; does **not** mutate B0.
- **Tests:** `tests/analysis/test_decision_latents.py` — long-baseline parameter recovery (primary); lick-hazard survival correctness (censoring, `1−Π(1−hₜ)`); descriptive-readout sanity; regression cross-check vs the old GLM's temporal hazard.
- **Script:** `analysis_suite/01_behavior/i_decision_latents_by_state.py` — stats CSV + cached latent table (§5) + a **set of saved, presentation-ready figures** (every step writes one; plain-language title + caption, §0):
  - **Step 1:** *F1 Sharpness over learning* (psychometric curves by mood & learning, d′ trajectory, large-changes-first); *F2 RT & its variability* (Hit-RT mean + CV per change-size over learning); *F3 Itchiness over learning* (criterion *c*, FA-rate, baseline hazard, by mood); *F4 Temporal expectation* (lick-hazard vs censored change-onset hazard, migration over learning; FA- & hit-time distributions); *F5 Bias-not-gain test* (Impulsive vs StimSens psychometrics — apparent hypersensitivity, d′ flat, no leftward shift); *F-summary which-dial-moves / which-dial-separates-moods table*.
  - **Step 2:** *F6 parameter recovery* (recovered vs true, long-baseline regime); *F7 per-trial latent distributions* (the three dials by mood & learning); *F8 construct validity* (generative latents vs Step-1 descriptive scores).
- **Index:** add the **B8** row to `docs/science/QUESTION_INDEX.md` (status `spec-draft`).
- Conventions: constants from `constants.py`; **iterate all state-tagged sessions** (not `load_staging_manifest` — kept only for the robustness subset); new-labeler state accessor; `setup_style()`/`save_figure()` for every figure; `del sess; gc.collect()` in session loops; canonical `visdetect.*` imports.

## 11. To resolve at planning time (writing-plans)

- **TF update period / sampling** — verify the true 50 ms cadence in `baseline_values` vs the old `BASELINE_STRIDE=3`; fix before any TF term; set `dt`.
- **Response-window length** for the Miss censor time (`RESPONSE_WINDOW`; ddm.py uses a provisional 2.155 s — confirm against task params).
- **Abort handling** — right-censor vs exclude (inspect why aborts terminate).
- **State accessor entry point** — confirm the new-labeler tag schema (`state_label` incl. `Abort`, `state_confidence`, gating) and the session-id ↔ tag-filename mapping (leading-zero / `zfill(8)` gotcha from `memory/state_labeler_neural_validation_jun2026`).
- **Session set & enumeration** — enumerate **all** sessions with a state-tag file (confirm count; pkl availability); confirm we are *not* gating on the manifest. Define the **Tier-1 integrity floor** (min total trials, valid task/stimulus) and the **per-(session×mood) minimum-trial** threshold for a stable cell.
- **Labeling coverage prerequisite** — 27/~45 tagged; **run the state labeler on the ~18 untagged (mostly earliest) sessions** (`scripts/state_labeling/`) and **reliability-check the naive-session labels** (mood proportions, exemplars, confidence) before trusting their mood split (§7). Sequence: Step 1 on the 27 now → label the rest → full-coverage rerun.
- **`comprehension_flag` definition** — operationalize the pre/post-comprehension split (e.g. first session with reliable easy-change hit-rate, or d′ crossing a low threshold) — used as a label, not an exclusion.
- **Learning-axis granularity** — continuous per-session d′ (chronological) vs session-bins vs 2 coarse stages; how to render the de-sculpted low-d′ reference sessions.
- **Regression-accumulator parameterization** — which params get a state term vs a learning term; urgency profile family (peak-time/precision free vs seeded); seeding/regularization scheme for the expert→backward fit.
- **Statistician knobs** — recovery tolerance; bootstrap scheme (session vs trial); AIC/BIC/CV-LL for any model comparison; FDR only within a question, not across figures. → hand to **Research Statistician** during planning.
- **B8 ID confirmation** — confirm/rename against the landscape; note D1 overlap.

## 12. Links

- Direction: `docs/science/2026-06-17-post-tf-null-research-direction.md` (§6 method order; the two-axis thesis); `memory/research_direction_post_tf_null_jun2026.md`
- Landscape: `memory/question_landscape_jun2026.md` (spine; B0/B1/D1)
- Sibling specs: `docs/superpowers/specs/2026-06-10-B0-ddm-learning-knob-design.md` (reference-only), `...-B1-integration-timescale-learned-design.md` (owns `λ`)
- Code: `visdetect.analysis.ddm` (reusable helpers), `visdetect.analysis.behavior` (SDT/psychometrics), `analysis_suite/07_advanced/k_lick_hazard_glm.py` (skeleton + cross-check, not a dependency), `data/cache/state_tags/BG_046/` (new-labeler tags)
- Results to date: `docs/science/2026-06-13-B0-ddm-learning-knob-results.md`; `memory/state_labeler_neural_validation_jun2026` (states differ by gain/RT, supports H2)
- Convention: `memory/science_spec_corpus_convention.md`; index `docs/science/QUESTION_INDEX.md`
