# Design spec — B9: Does the striatal baseline TF representation *sharpen with learning* at matched behavioral state? (decision-spine, neural phase)

| | |
|---|---|
| **Question ID** | **B9** — new sharp-novelty ID in the decision-spine / learning family (sibling of B0/B1/B2/B8). The **matched-state, baseline-TF-encoding** instantiation of the "does striatal sensory responsiveness track learning" thread (**B2**), confound-hardened: behavior *and* sensory drive are matched, and motor is controlled by the encoder's own regressors. One question = one spec + one plan; filed in `docs/science/QUESTION_INDEX.md`. |
| **Date** | 2026-07-01 (rev. same day: guard/safety-distance apparatus removed after verifying the registry estimator — motor/onset are controlled by regression, not censoring; readout re-cast as a state-conditioned re-run of the *existing* estimator) |
| **Status** | SPEC-DRAFT (brainstormed 2026-07-01: readout moved off change-evoked → baseline-TF-encoding; subject → BG_039; population decoder down-scoped; **estimator pinned to the registry's own GLM after a read-only worktree audit**; awaiting user spec review → writing-plans) |
| **Feasibility tier** | **T1** — data in hand for **BG_039**: 29 state-tagged sessions (`data/cache/state_tags/BG_039/`), the TF-responsive registry (`data/cache/tf_responsive/bg039_tf_responsive.csv`), 32 pkls, staging manifest (`data/BG_039_staging_manifest.csv`). **Single-subject start (BG_039, DMS)**, architected to extend to the **DMS pool (+BG_046)** and **VMS (BG_031) kept separate** as their TF runs finish. |
| **Spine** | The learning arm of the post-TF-null decision-spine ([[research_direction_post_tf_null_jun2026]]): *how do mice learn to boost sensitivity to informative stimuli?* Here, sensitivity = the fidelity of the striatal baseline TF representation. **This escapes the state-label circularity** ([[state_labeler_circularity_caveat]]) — the readout is a labeler-independent neural encoding, and behavior is matched by construction. |
| **Estimator (pinned; verified read-only on `feature/tf-glm-bg046` @ `5252d54`, 2026-07-01)** | B9 **reuses the exact GLM that produced the registry** — `session_trial_regressors` → `assemble_design` → `fit_poisson_cv` → `identify_tf_responsive_pulse` ([[tf_glm_replication_jun2026]]). `src/visdetect/analysis/tf_glm_data.py` is **byte-identical between `main` and that worktree**, so B9 builds on **main** (no worktree dependency). Config: `bin_s=0.05`, TF FIR (0–1.5 s, 30 lags, log2/0.25 octaves, baseline_values stride-3, zeroed at/after `Change_ON`), reduced regressor set (§4), ridge-Poisson, 10-fold **trial-blocked** CV (seed 42); readout = `c1_r_log2` (fast-minus-slow pulse-PETH correlation). |
| **Inputs (verify in Phase 0)** | (1) Registry — 2442 units / **75 responsive (3.1%)**; join via `config.canonical_session_id` on BOTH sides; `resp_log2` labels, `c1_r_log2` strength. (2) State tags — `state_label` + `state_confidence`, `trial_idx` 1:1 onto session trials. (3) `Trial.baseline_values` (St1TrialVector) present in the pkls (the TF source; confirmed on `16052025`). |

> **Plain-language contract ([[feedback_plain_language_and_save_figures]]).** Every concept gets a one-line gloss; every step writes a presentation-ready PNG to `FIGURES/state_tf_learning/BG_039/`. **Glossary:** *behavioral state* = a labeler-tagged stretch of trials (Impulsive / StimSens[engaged] / Disengaged / Abort); *baseline period* = the long (~7 s) pre-change grating epoch whose temporal frequency (TF) fluctuates stochastically (~50 ms updates); *TF-responsive unit* = a neuron whose firing tracks those fluctuations (registry `resp_log2`); *TF-encoding fidelity* (`c1_r_log2`) = how well the unit's firing predicts fast-vs-slow TF pulses; *matched state* = the *same* state early vs late in learning, subsampled to equivalent behavior; *sharpening* = the same input encoded more faithfully.

---

## 0. Design logic — why this dissociation is clean

The claim is a **dissociation**: hold the *behavior* fixed and ask whether the *neural code underneath it changed with learning*. Two things are matched, which is what makes a positive result **representational learning** rather than a confound:

1. **Behavior matched** — the same behavioral state (StimSens primary), subsampled to equivalent within-state behavior early vs late (§5).
2. **Sensory drive matched** — the baseline TF fluctuation generator is **set by the task**, the same process every session (Phase 0 *verifies* the baseline-TF distribution matches across the chosen early/late sets — not assumed).

With behavior *and* input matched, a change in how **TF-responsive** units encode that fixed input is a near-pure change in the **representation**. **Non-responsive units are the built-in specificity control** (the null channel): if they don't move, it's the sensory channel sharpening, not global drift/arousal ([[state_labeler_neural_validation_jun2026]]).

**Motor and onset are controlled by *regression*, not censoring — and B9 inherits that control unchanged.** The registry GLM already carries nuisance kernels that absorb peri-event activity (§4): pre/post-lick (`lick_prep (−1.25,0)`, `lick_exec (0,0.5)`), running `wheel`, `reward`, per-size `change`, `abort`, and the **gray→grating onset** (`trial_start (0,1.0)` + tiled-baseline tiles). TF is **zeroed at/after `Change_ON`** (baseline-only). So there is **no need for hand-rolled lick-free windows or event "safety distances"** — those are what the kernels *are*. (Events I earlier worried about — laser, airpuff — **do not exist in BG_039 baseline**: no `Laser` channel and optotagging is a *post-session* block; `Air_puff` is all-NaN; reward `Valve_L` fires post-change. Verified on a real session.) Residual honest caveat: control is **lick + wheel** (no video/facemap for BG mice) — the "no-movement" limitation, flagged in results.

**Conventions ([[feedback_repo_structure_scripts_figures]], [[feedback_canonical_imports]]):** code in `scripts/state_tf_learning/`; reusable logic in `src/visdetect/` (NOT `analysis_suite/`); caches `data/cache/state_tf_learning/`; figures `FIGURES/state_tf_learning/BG_039/`. **HARD:** ids via `config.canonical_session_id()` both sides (leading-zero day, 6-digit DDMMYY `270325`, `_v2` suffixes — [[feedback_canonical_session_id]]); sessions via the BG_039 manifest; units via `get_good_cluster_ids()` preferring `good_and_stable_ids` ([[good_and_stable_ids_definition]]); **dt = 0.05 s** (inherited from the estimator — never 0.25, [[tf_fluctuation_50ms_vs_constant]]); `state_confidence > 0.8`; `trialoutcome` CAPITALIZED; **no compute over X:** ([[feedback_no_compute_over_samba_gateway]]); all subagents **Opus 4.8** ([[feedback_subagent_model_opus]]); state colours `config.STATE_LABEL_COLORS` ([[reference_state_label_colors]]).

---

## 1. Scope

**In scope (BG_039 first):**
- **Phase 0 — feasibility / profiling** (the data-quality gate — [[feedback_data_quality_gate_first]]): the coverage landscape + the **free registry-only preliminary** (§6).
- **Phase 1 — state-conditioned TF-encoding fidelity** (headline): the registry estimator re-run on StimSens trials, early vs late; responsive vs non-responsive.
- **Phase 2 — gain / firing structure** and **Phase 3 — sensory-subspace geometry**: follow-on readouts on the same scaffolding.
- **Primary state** = StimSens (engaged); **control state** = Disengaged; Phase-0 may down-scope to StimSens-only or re-pick if coverage is thin.

**Out of scope (cross-referenced):**
- **Cross-session / cross-subject cell pooling** — blocked until `region_bank_confirmed` (chronic-probe drift; registry caveat). DMS = {BG_046, BG_039} poolable *once confirmed*; VMS (BG_031) never with DMS.
- **BG_046 / BG_031** — TF runs still finishing; pipeline is subject-parameterized so they drop in later (F3 cohort).
- **Movement-controlled (video) TF-responsiveness** — the registry is lick/wheel-controlled (no video); a motion-controlled re-run is a separate dependency.
- **Cell-type (SPN/FSI) decomposition** — deferred (kernel width in the registry enables it later).
- **Change-evoked / lick-locked readouts** — rejected (motor confound; N1's territory).
- **Re-classifying responsiveness** — B9 *uses* the registry's `resp_log2` label as given; it does not recompute the pass/fail call.

---

## 2. The question, the claim, and the GATE

**Headline question.** In BG_039 DMS, does the **baseline TF-encoding fidelity (`c1_r_log2`) of TF-responsive units increase from early to late in learning, at matched behavioral state (StimSens)** — while **non-responsive units stay flat** (specificity)?

**Existence GATE (Phase 0 must pass before Phase 1 is worth running):**
1. **State coverage (concrete).** At **both** an early and a late stage, the confidence-gated (`state_confidence>0.8`) StimSens trials must support the estimator's own requirements: **10-fold trial-blocked CV** with, per fold, **≥20 fast *and* ≥20 slow TF pulses** (fast/slow = log2-TF beyond ±0.5 SD) and units clearing **`MIN_SPIKES=500`** on the state subset. Baselines are ~7 s at 50 ms updates, so *pulses* are dense — the binding constraint is **trial count per state per stage** (thresholds read from the Phase-0 distribution, not guessed).
2. **Unit coverage.** ≥ *M* responsive units + an adequate non-responsive comparison set at both stages (only 4 sessions have ≥5 responsive units → likely aggregate a *stage's* sessions as independent **per-unit** observations; never concatenate spike trains across sessions — the no-pooling rule).
3. **Input match.** The baseline-TF pulse distribution is statistically indistinguishable across the chosen early/late sets (else the matched-input premise fails and must be reported).

Passing the gate + a positive Phase-1 result is the headline. **A flat or non-specific result is a clean, reportable negative** ("the baseline TF representation is stable across learning at matched state, in this subject/readout").

---

## 3. Data, units, labels, joins

- **Sessions.** BG_039, ordered chronologically via `parse_session_date` / `chronological_sort`; stage from `data/BG_039_staging_manifest.csv`. **Inclusion overrides the d′ filter** — admit a session on the coverage gate (§2), so early low-d′ sessions are *not* dropped (they are the point). Handle 6-digit DDMMYY (`270325`,`280325`) and `_v2` via `canonical_session_id`.
- **Units.** Per session, `get_good_cluster_ids()` preferring `good_and_stable_ids` (the estimator's own unit pool).
- **TF-responsive label.** Registry `resp_log2` splits **responsive vs non-responsive**; `c1_r_log2` retained for a graded-threshold sensitivity check. **The registry LABELS units; B9 re-MEASURES `c1_r_log2` on state-conditioned trial subsets** using the identical estimator.
- **State label.** State-tag CSV: `state_label`, `state_confidence`; `trial_idx` maps 1:1 onto `session.trials` (the partition key).
- **Joins.** `canonical_session_id` both sides; unit join on `(subject, sess_key, unit=cluster_id)`. A regression test asserts every early / 6-digit / `_v2` session joins (§8).

---

## 4. Readout — reuse the registry estimator, partition by state × stage

**The whole method in one line:** *run the existing encoder on the StimSens-early trials, run it on the StimSens-late trials, compare `c1_r_log2` — responsive vs non-responsive.* Nothing is re-implemented; the only change vs the registry run is the **trial subset** fed to `session_trial_regressors`.

- **Inherited, unchanged (do NOT re-tune):** `bin_s=0.05`; TF regressor = `baseline_values` stride-3 → linear Hz → `log2(TF)/0.25` octaves, **FIR (0–1.5 s, 30 lags)**, **zeroed at/after `Change_ON`** (baseline-only); reduced regressor set below; ridge-Poisson (`PoissonRegressor`, λ∈{1e-3…100}, `fast_fit` λ-once, `max_iter=500`, `tol=1e-4`, `standardize_design=True`); **10-fold trial-blocked CV, seed 42**; pulse criterion (`sd_pulse=0.5`, `pulse_eval_win=(−0.15,0.75)`, `min_pulses_per_label=20`); `MIN_SPIKES=500`.

  | Regressor | Kernel window | Role (the "safety distance" it absorbs) |
  |---|---|---|
  | `tf` | (0, 1.5) s, 30 lags | **the signal** (log2 octaves) |
  | `trial_start` | (0, 1.0) s | **gray→grating onset transient** |
  | `tiled_baseline` | 80×200 ms / 16 s | slow time-in-baseline drift |
  | `change_{1.0…4.0}` | (0, 2.0) s ×6 | change-evoked response |
  | `lick_prep` / `lick_exec` | (−1.25, 0) / (0, 0.5) s | pre-/post-lick motor |
  | `reward` | (0, 0.4) s | consummation |
  | `abort` | (−1.25, 0.25) s | aborts |
  | `wheel` | (−0.05, 0.8) s | running |

- **Baseline-onset (gray→grating) rise:** handled by the `trial_start (0,1.0)` kernel (+ early tiled-baseline tiles) — regressed, not censored. **This GLM *is* the lab convention for the onset**, so no first-second censoring and no extra onset branch.
- **The readout metric:** per unit, per (state, stage), the estimator's **`c1_r_log2`** (fast-minus-slow pulse-PETH correlation; the same number the registry reports) — plus its components `r_full`/`r_red` for interpretation. Non-responsive units get the identical measurement (expected ≈ flat null).
- **CV / no-circularity:** trial-blocked folds, fit on train, scored on held-out (inherited). λ selected once per unit on the subset (as in the registry).

---

## 5. Matching battery

1. **Matched state.** Same `state_label` (StimSens) early vs late; Disengaged as the control state.
2. **Within-state behavior match.** Subsample early/late to equal StimSens-trial counts and match a within-state engagement proxy (baseline lick rate); repeat over subsampling draws, report the distribution (not one draw).
3. **Input match.** Verified in Phase 0 (§2.3); if pulse distributions differ, report / stratify.
4. **Movement control (inherited + honest).** Peri-lick and running are regressed out by the `lick_prep`/`lick_exec`/`wheel` kernels (the registry's own control). No new movement channel exists for BG mice (no video); stated as the "no-motion-video" limitation, revisited when a motion-controlled TF pass exists.

---

## 6. Phased build

- **Phase 0 — feasibility / landscape (gate) + the FREE preliminary.**
  - *Free registry-only trend (zero re-run):* group the registry's existing whole-session `c1_r_log2` (responsive units) by **early vs late sessions** → a first look at whether TF-encoding shifts with learning *at all* (no state control yet). **Deliverable: a trend figure.**
  - *Coverage landscape:* BG_039 sessions × stage × StimSens/Disengaged occupancy (confidence-gated) × per-state trial counts vs the §2 CV/pulse requirement × responsive/non-responsive unit counts × baseline-TF-pulse-distribution check → one canonical **`usable`** flag + the chosen state + early/late session sets. **Deliverable: landscape figure + coverage table.** No state-conditioned scoring before this passes.
- **Phase 1 — state-conditioned TF-encoding fidelity (headline).**
  - **1a (PRIMARY): state-conditioned `c1_r_log2`.** Re-run the estimator on StimSens-early vs StimSens-late trial subsets; per-unit `c1_r_log2`, responsive vs non-responsive; mixed-effects (unit nested in session), effect sizes, permutation nulls; the pre-specified **stage×class interaction** (§7). **Deliverable: headline figure + stats CSV.** Sensitivity check: graded `c1_r_log2`/`resp` threshold.
  - **1b (DEFERRED): population TF-decoder.** A responsive-only per-session decoder is barely feasible (≤9 units; 4 sessions ≥5) and cell-pooling is blocked → **defer** to when BG_046 responsive units land + `region_bank_confirmed` enables DMS pooling. Log what was deferred and why (no-silent-caps).
- **Phase 2 — gain / firing structure**; **Phase 3 — sensory-subspace geometry** (gated on units; likely awaits pooling).

---

## 7. Statistics

Mixed-effects (**unit nested in session**) for `c1_r_log2` ↔ (stage, responsive-class); **session = unit of replication**; **non-parametric defaults** (Spearman / Mann-Whitney / Kruskal-Wallis); **effect sizes** with every p; **bootstrap CIs** (1000 resamples, seed=42, percentile); permutation nulls (≥200; chance = mean ± 2 SD); **FDR (BH, α=0.05) only within a per-unit screen**, never across phases/figures. Matched-state subsampling repeated over draws; report the across-draw distribution. **Success criterion (pre-registered):** responsive-unit `c1_r_log2` rises early→late beyond null **AND** non-responsive stays flat — a **stage×class interaction**, not a bare main effect. Consult the Research Statistician skill.

---

## 8. Testing / validation

- **Faithfulness / reproduction test (the key one).** On **whole-session** trials (no state split), B9's re-run must **reproduce the registry's `c1_r_log2`** for a sample of units (within numerical tolerance) — proves the reuse is byte-faithful to what defined `resp_log2` before we ever subset by state.
- **State-partition correctness.** The StimSens/early/late subset fed to `session_trial_regressors` is exactly the confidence-gated `trial_idx` set; a unit test asserts the partition matches the state-tag CSV.
- **Synthetic recovery.** Simulate a *sharpening* TF-encoding signal in a "responsive" subset across stages + flat "non-responsive"; confirm 1a detects the interaction and the null channel stays flat; confirm a *global* gain change (both classes) does NOT fake specificity.
- **Join integrity.** Every BG_039 id — 6-digit (`270325`,`280325`), leading-zero, `_v2` — joins to registry + state tags after `canonical_session_id`.
- **Determinism / compute hygiene.** `ProcessPoolExecutor`, param `n_workers`, BLAS pinned per worker; pkl loading sequential; `del sess; gc.collect()`; **no compute over X:**; seed=42 fixed (matches the estimator).

---

## 9. Outputs & repo structure

- `scripts/state_tf_learning/` (import `visdetect.*`); `data/cache/state_tf_learning/` (the state×stage×unit `c1_r_log2` table with canonicalized ids; the Phase-0 coverage table with `usable`); `FIGURES/state_tf_learning/BG_039/`.
- Per-step **presentation-ready** figures (plain-language titles/captions) + **stats CSVs**.

---

## 10. Risks / decision points

- **(a) Small responsive-N (the big one).** 75 responsive, ≤9/session, 4 sessions ≥5, pooling blocked → **per-unit `c1_r_log2` primary; population decoder deferred.** If even the per-unit contrast is too thin at a stage, **report the ceiling** and aggregate a stage's sessions (per-unit observations) or wait for BG_046/pooling. Rescue: graded `c1_r_log2` threshold to widen the responsive set (pre-declared sensitivity analysis).
- **(b) State coverage / CV feasibility.** State subsets may not support 10-fold CV with ≥20 fast+slow pulses/fold at the early end (impulsive/disengaged mice) — the wrangling worry, now made concrete. Phase 0 decides; fallbacks: fewer folds, relax the early boundary, Learning-vs-Expert instead of Naive-vs-Expert, StimSens-only.
- **(c) Movement not video-controlled.** lick+wheel regressed, no facemap/pupil; stated as a limitation.
- **(d) Face-validity states for BG_039.** Labeler κ ground-truthed only on BG_046; BG_039 states are the transferred rule (face-valid, [[state_labeler_jun2026]]). Reported as such.
- **(e) Circularity re-entry.** States are behaviorally defined; the encoding readout is labeler-independent, so the core claim is safe — do NOT re-describe the *behavioral* state difference as a finding.
- **(f) Input-match failure.** If baseline-TF pulse statistics differ across stages, the matched-input premise weakens → stratify / report.

---

## 11. Literature grounding

- **TF-encoding GLM = the arbiter** (Khilkevich & Lohse 2024 replication — [[tf_glm_replication_jun2026]]); the registry is its per-unit product, and B9 reuses its estimator.
- **Sensitivity is the learned axis** of the spine ([[research_direction_post_tf_null_jun2026]], [[question_landscape_jun2026]]); B9 tests "boost sensitivity to informative stimuli" neurally.
- **State sets sensory gain** (prior neural validation — [[state_labeler_neural_validation_jun2026]]); B9 asks whether the *baseline* sensory channel *sharpens with learning* at matched state.
- **Perceptual learning sharpens sensory codes** ([[synthesis-batch05-confidence-lapses]] SAT/learning; [[synthesis-batch06-brainwide-population]] population coding; [[synthesis-methods-nds]] normalize-before-reduce, decodability≠meaning).
- **Circularity break** via labeler-independent neural readout ([[state_labeler_circularity_caveat]]).
