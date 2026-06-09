# Design spec — B2: Does striatal sensory responsiveness *lead* the behavioral learning curve?

| | |
|---|---|
| **Question ID** | B2 (see `memory/question_landscape_jun2026.md`) |
| **Date** | 2026-06-08 |
| **Status** | SPEC-APPROVED (2026-06-08; motor-confound rationale §4.3 added per review) — ready for writing-plans |
| **Feasibility tier** | T1 (existing BG_046 ephys + behavior) |
| **Spine** | *How do mice learn to suppress impulsivity and increase sensitivity?* — this is the **sensitivity** half (drift rate), longitudinal |
| **Lit anchor** | Marica 2025 (`paper-marica-2025-striatal-visual-responses-prelearning`): striatal sensory responses rise *before* behavior and *before* cortex; mPFC-recipient (≈DMS) domain develops stimulus coding *with* learning |

---

## 1. Scientific question & hypothesis

**Question.** Across BG_046's Naive→Expert trajectory, does the striatal population's encoding of the TF change **rise earlier in training than** the behavioral sensitivity curve (d′)?

**Hypothesis (H1, directional).** The neural sensory-encoding curve leads the behavioral curve: `t50_neural < t50_behavior`, where `t50` is the training time of steepest rise of each curve. This is the BG_046-testable half of Marica's "striatum leads behavior and cortex."

**Null (H0).** The two curves rise contemporaneously (`t50_neural ≈ t50_behavior`), or the neural curve lags. A spurious lead produced by changing unit samples / recording-yield drift is a *confound to exclude*, not a positive result.

**Why it's interesting (field gap).** Almost all prior change-detection ephys (Khilkevich/Lohse, Orsolic, the Lohse sister study) is expert-only or cross-sectional. A *temporal-precedence* claim between neural and behavioral learning requires a dense within-subject longitudinal series — exactly BG_046's structural edge. A confirmed lead is mechanistically loaded: it says the striatal sensory representation is **upstream of**, not a readout of, the behavioral competence.

## 2. What this spec does and does NOT cover

**In scope (T1):**
- The **striatum-vs-behavior** precedence test, within subject BG_046.
- A continuous, yield-robust neural encoding measure + a behavioral sensitivity measure, each as a per-session curve over real training time.
- A lead-lag inference that is valid for two monotonically-rising, autocorrelated curves.

**Explicitly OUT of scope (recorded here so the chat that runs this doesn't overreach):**
- **"Striatum leads *cortex*"** — BG_046 is striatum-only. The cortical arm needs the Aim-2 simultaneous PPC/aMOs/pMOs + striatum cohort → **T3**, separate question.
- **Causation** ("the striatal signal *drives* learning") — needs Aim-3 silencing → **T3**.
- **Single-cell plasticity vs population re-weighting** — that is question **B3**; it needs a healthy tracked-and-responsive pool, which BG_046's tracking yield does not currently support (see §7). B2 is deliberately the *session-level* question that survives low tracking yield.

## 3. Data inputs

- **Sessions:** all QC-passing BG_046 sessions via `load_staging_manifest(qc_only=True)` (applies `SESSION_FILTER`; merged Naive→Learning). Order chronologically by real date — `parse_session_date()` / `chronological_sort()` — and index the curves by **days since first session** (not session ordinal), using pkl dates per the manifest-gaps rule (gaps ≠ training breaks).
- **Per-session analyzable unit pool:** the `good_and_stable_ids` set stored in each pkl. **NB:** `good_and_stable_ids` is the *within-session* `find_good_stable_units` QC filter (see `memory/good_and_stable_ids_definition.md`), **not** UnitMatch tracking; pkls store spikes only for these units. This is the correct and only per-session pool — no cross-session matching is required for the primary analysis.
- **Behavioral metric:** per-session d′ (and, if cheaply available, the change-size psychometric slope) from `visdetect.analysis.behavior` (SDT: hit rate on go-trials, FA rate on catch-trials, log-linear correction). Prefer the manifest's d′ column if it is the same definition; otherwise recompute for consistency.
- **TF-pulse responsiveness machinery:** `visdetect.analysis.tf_pulse` (z-scored TF-pulse traces, screening) with **detrending** (`detrend_tf_traces`, confirmed present in the tf_pulse/behavior/loader modules — verify exact signature at planning time). Constants from `visdetect.analysis.constants` (`TF_PULSE_PRE_WINDOW`, `TF_PULSE_POST_WINDOW`, `DEFAULT_Z_THRESH_TF`, `DEFAULT_BIN_SIZE`, `DEFAULT_SIGMA_MS`).
- **Population tensor:** `build_population_tensor()` (suite/utils) for the decodability measure.
- **(Bonus only) cross-session tracks:** the liberal UnitMatch registry `batch0/unit_index.csv` / curated-tracks output — used only for the §6 corroboration, kept entirely separate from `good_and_stable_ids`.

## 4. Measures

### 4.1 Behavioral curve (the "when did the mouse learn" axis)
- **Primary:** d′ per session. Optionally the psychometric slope (steepness of P(lick) vs change-size) as a second sensitivity readout.
- Rationale: the spine's *sensitivity* half maps to d′ / slope (≈ DDM drift), **not** FA rate (that is the impulsivity half, a different question).

### 4.2 Neural sensory-encoding curve (the "when did the neurons learn" axis)
Computed identically on every session's `good_and_stable_ids` pool. In increasing order of rigor / decreasing yield-sensitivity:

- **(P) PRIMARY — cross-validated TF-*pulse* decodability (motor-free).** Decode the **identity of baseline-period TF pulses** (fast vs slow, per `TF_FAST_THRESH_LOG2` / `TF_SLOW_THRESH_LOG2`) from the pulse-evoked population response (`TF_PULSE_POST_WINDOW` relative to `TF_PULSE_PRE_WINDOW`). Stratified 5-fold CV accuracy (or AUROC), label-shuffle null (≥200 shuffles). **Fixed-n unit subsample** per session (n = floor across sessions, or a chosen floor with sessions below it dropped/flagged), averaged over repeated subsamples, so recording yield does not drive the curve. This uses **all** units including weak encoders — the correct move given most units are individually non-significant (distributed coding; flank-unit information, `paper-pouget-2000-population-codes`). **The measure is built on the baseline TF-pulse response, NOT on change-aligned activity — see §4.3.**
- **(S) SECONDARY — continuous population TF drive.** Mean |detrended TF-pulse z-score| across the pool per session. Cheaper, graded, no decoder; corroborates (P).
- **(C) SANITY — detrended responsive-fraction.** Fraction with detrended z > `DEFAULT_Z_THRESH_TF`. Most yield/threshold-sensitive → sanity check only. Requires the **detrended re-extraction** first (the flagged 8.5%→~12–15% fix); without it this measure is biased by baseline drift that itself varies across sessions.

**Prerequisite:** run/confirm the detrended TF re-extraction into its own cache before computing (S)/(C); (P) should also use detrended traces for consistency.

### 4.3 Why TF-pulse, not change-evoked (the motor confound) — load-bearing
The post-change ramp in BG_046 is established to be **lick-locked motor preparation, not sensory tiling** (Fig14; `memory/scientific_context.md`, `memory/analysis_frontiers.md`). Therefore a **change-aligned** responsiveness index — decoding change-size, or change-evoked magnitude — would conflate *sensory* encoding with *motor-preparation* encoding. This is doubly dangerous for B2: motor preparation itself sharpens across training, so a change-aligned measure could manufacture a spurious neural-leads (or lags) result that has nothing to do with sensory learning. The baseline-period **TF-pulse response occurs away from both the change and the lick**, so it is the **motor-free sensory index** this question requires. **All three neural measures (P/S/C) are computed on baseline TF-pulse responses for this reason; change-aligned decoding is explicitly rejected as the responsiveness index.** (If a change-aligned view is ever wanted, restrict it to a strict pre-RT window and regress out movement — but it is not part of the primary B2 result.)

## 5. Analysis design — the lead-lag inference

**The trap (must avoid):** both curves rise monotonically with learning, so a raw cross-correlation is trivially maximal near lag 0 and the lead-lag estimate is dominated by the shared trend + autocorrelation. **Do not** report raw cross-correlation of the two rising curves as evidence of a lead.

**Primary inference — inflection (t50) comparison:**
1. Fit each curve (behavioral; neural-P) with a monotonic learning function (logistic/sigmoid in training-day units; robust to the short series). Extract `t50` = day of steepest rise.
2. **Bootstrap** CIs on each `t50`: resample sessions (block/stationary bootstrap to respect temporal autocorrelation); for the neural curve also resample units within each session (propagates yield noise). Report the distribution of `Δt = t50_behavior − t50_neural`.
3. **Positive result:** `Δt > 0` with bootstrap CI excluding 0 (neural leads, in days).

**Secondary inference — first-difference cross-correlation:**
- Cross-correlate the session-to-session deltas of the two curves (removing the shared monotonic trend). A genuine lead survives detrending; report lag of peak with a block-bootstrap CI.

**Tertiary (suggestive only) — predictive direction:**
- A lightweight Granger-style check (does neural[t] add predictive power for behavior[t+1] beyond behavior's own past, more than the reverse?). Flag as underpowered given n=1 short series; report as supporting, not decisive.

**Null / specificity control:** session-label circular-shift of one curve relative to the other to build a null for `Δt` and for the cross-correlation lag.

## 6. Bonus sub-analyses (run only if §5 is positive or as enrichment)

- **(B-track) Tracked-cell corroboration.** From the trusted UnitMatch tracks (`batch0/unit_index.csv` / curated cohort — small, per the tracking-QC findings), follow the **continuous mean TF drive across ALL tracked cells** (NOT a pre-selected TF-responsive subset — that is circular and impossible at this yield). Qualitative support that the same neurons gain drive over training; power-limited, one panel, explicitly not load-bearing.
- **(B-switch) Selectivity migration (the richer test).** Marica's second result: the ≈DMS domain switches **target→cue** selectivity with learning. BG_046 analog: per session, compare the population's encoding of the **change-cue** event vs the **reward/lick** event (align to each — respecting `EVENT_VALID_OUTCOMES`: Change_ON only on hit/miss; lick/Hit motor-aligned). Test whether the **cue-locked share grows** across training. This is about *what* the signal encodes, not just *how much* — arguably the more interesting finding if present. **Caveat (per §4.3):** the change-cue limb is itself partly motor (post-change ramp = lick-prep), so anchor "cue" selectivity on the **TF-pulse sensory axis** (or a strict pre-RT window with movement regressed), not on raw change-evoked magnitude — otherwise this collapses into a motor-vs-motor contrast.

## 7. Confounds, limitations, honest scope

- **Recording-yield (not tracking-yield) is the real power constraint.** Low units/session → noisy per-session decodability → wide `t50` CIs → harder to resolve a small lead. Mitigations: fixed-n subsample to the floor; the continuous (S) measure (more power than a fraction); **learning-phase binning** (early/mid/late or d′ tertiles) if session-level is too noisy. **Tradeoff to state in results:** binning buys SNR per point at the cost of temporal resolution — a lead measured in coarse bins is a softer claim. There is a floor below which the honest conclusion is "consistent with neural-leads, lag unresolved."
- **Changing unit sample across sessions** (the session-level confound): excluded via fixed-n subsampling + the (B-track) tracked corroboration + a depth/probe-position robustness check (is the trajectory stable when controlling for recording depth?).
- **Movement confound:** the primary measure is **motor-free by construction** — it is the baseline TF-pulse response, away from change and lick (§4.3), which is the whole reason change-aligned decoding is rejected. The TF-pulse window is early/pre-lick; still verify the measure does not track a baseline movement covariate that itself ramps with learning (Stringer caution; ties to the video/FaceMap workstream).
- **n = 1.** This is a within-subject trajectory claim. Cohort pooling (BG_031/038/039 — question F3) is the path to a population claim and is the natural follow-up.
- **Detrending dependency:** results conditioned on the detrended TF extraction; report both detrended and (for the fraction sanity) non-detrended where feasible.

## 8. Success criteria

- **Positive:** `Δt = t50_behavior − t50_neural > 0`, bootstrap CI excludes 0, on the PRIMARY neural measure (P), corroborated in sign by (S); robust to fixed-n subsampling and to a depth control; survives the shuffle null.
- **Negative:** CI includes 0 or `Δt < 0` → striatal responsiveness does not lead (still publishable as a calibration of Marica in DMS / a single-subject null).
- **Inconclusive:** lead sign consistent but CI too wide at achievable yield → report as "consistent with neural-leads, unresolved," motivating cohort + tracking improvements.

## 9. Deliverables

- **Figure:** (A) behavioral d′ curve + neural-(P) curve over training days with fitted sigmoids and t50 markers; (B) bootstrap distribution of `Δt`; (C) first-difference cross-correlation; (D) the (S) continuous measure overlay; optional (E) (B-switch) cue-vs-reward share over training.
- **Stats CSV:** per-session d′, neural-(P)/(S)/(C) values, n_units, subsample params; t50 estimates + CIs; `Δt` + CI; shuffle-null summary.
- **Cache:** per-session neural measures (decodability, mean|z|, fraction) keyed by session + params, so the figure is cheap to re-render.
- Conventions: `setup_style()` / `save_figure()`; constants from `constants.py`; `del sess; gc.collect()` in the session loop.

## 10. To resolve at planning time (writing-plans)

- Exact signatures/locations of `detrend_tf_traces`, the TF screening entry point, and whether a detrended cache already exists.
- Whether the staging manifest's d′ column matches the `behavior.py` SDT definition (else recompute).
- Decoder target choice (change-size class vs fast/slow vs change/catch) and the fixed-n floor (inspect the units/session distribution first).
- Trusted UnitMatch track count available for (B-track) — confirm it is worth a panel.
- Statistical detail (block-bootstrap flavor, sigmoid parameterization, multiple-comparison posture) → hand to the **Research Statistician** skill during planning.

## 11. Links

- Landscape: `memory/question_landscape_jun2026.md` (B2, and the spine)
- Lit: `paper-marica-2025-striatal-visual-responses-prelearning`, `synthesis-phase3-pathways`, `synthesis-phase3-theory` (Bogacz drift = sensitivity), `synthesis-batch01-foundations` (distributed integration), `paper-pouget-2000-population-codes` (flank units)
- Project: `proposal_aims` (Aim-2 cortical arm = the T3 cortex comparison), `memory/good_and_stable_ids_definition.md`, `memory/neuron_tracking_may2026.md` (tracking yield reality), `memory/feedback_manifest_session_gaps.md` (time axis)
- Code: `visdetect.analysis.tf_pulse`, `visdetect.analysis.behavior`, `visdetect.suite.utils.build_population_tensor`, `visdetect.suite.loader.load_staging_manifest`
