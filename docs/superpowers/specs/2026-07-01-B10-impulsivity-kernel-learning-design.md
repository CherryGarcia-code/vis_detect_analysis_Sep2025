# B10 — Learning to reject distractors: the impulsivity kernel (behavioral + neural)

- **ID:** B10 (behavioral arm) / N2 (neural arm — second neural-phase question after N1)
- **Date:** 2026-07-01
- **Status:** spec-draft
- **Tier:** T1 (data in hand; verified below)
- **Subjects:** BG_046 (DMS), BG_039 (DMS), BG_031 (VMS)
- **Anchors:** Orsolic 2021 (Neuron), Khilkevich & Lohse 2024 (Nature), Lohse 2025 (preprint)

---

## §0 Motivation & framing

The project spine: *how do mice **learn to suppress impulsivity and increase sensitivity** to
informative stimuli, to drive perceptual decisions?* This question attacks **both halves in one
model-free readout**: the **psychophysical reverse-correlation kernel** — Orsolic 2021's signature
analysis, on the very task BG_046 runs (detect a temporal-frequency change in a stochastically
fluctuating drifting grating).

**The idea in one line.** The baseline grating TF fluctuates every 50 ms. Sometimes the mouse
licks *early* (a false alarm / FA — an impulsive lick before any real change). Reverse-correlate
the TF fluctuations preceding those impulsive licks → **the temporal pattern of stimulus the mouse
mistakes for a change** (the "impulsivity kernel"). Then ask how it changes Naive→Expert: **does
the mouse learn to stop treating baseline noise as change-evidence?** That single contrast speaks to
impulsivity control, sensitivity, and learning at once.

**Why it is fresh (not the analyses already in flight).** B8 (decision latents), B0 (which DDM
knob), N1 (urgency ramp — controlled negative), and B9 (state-matched baseline TF-*encoding
fidelity* — fast-path **null**, 2026-07-01) are all either model-based latents or steady encoding
fidelity. **None computes a stimulus kernel.** A grep of `src/`, `scripts/`, `analysis_suite/`
returns no reverse-correlation / STA / psychophysical-kernel / matched-withhold code. This is a
model-free, behavior-first, motor-safe angle.

**Why the neural arm is distinct from B9.** B9 asks whether *steady-state* baseline TF-encoding
fidelity (`c1_r_log2`) rises with learning at matched behavioral **state**. B10's neural arm (N-B)
is **action-conditioned**: does the TF-responsive population transiently *over-represent* an upward
fluctuation in the moments *before an impulsive lick* (vs matched withhold), and does that
over-representation shrink with learning? Different conditioning (action vs state), different object
(a temporal kernel vs a scalar fidelity), different message (the neural echo of the behavioral
impulsivity kernel).

**Paper grounding (see §11).** Orsolic 2021 = the paradigm parent and the kernel's origin.
Khilkevich & Lohse 2024 = the reference paper on the same task; its *lick-triggered TF kernel* is
the faithful descriptive anchor we overlay. Lohse 2025 = the reason the neural arm restricts to
**TF-responsive cells** and a stimulus-referenced readout: sensory encoding and task-state occupy
orthogonal dimensions, and reading raw hit−miss (a movement contrast) conflates them — the exact
leakage N1 proved fatal.

---

## §1 Question & hypotheses

**Q1 — behavioral (Arm 1 / I1).** What temporal pattern of baseline TF fluctuation precedes
impulsive (FA) licks, and how does it change with learning?
- **H1:** an FA is preceded, on average, by an **upward** TF excursion (the mouse mistakes a fast
  fluctuation for a change; all task changes are TF *increases*, ratios 1.25–4.0). With learning the
  kernel **sharpens / narrows** (shorter effective integration window) and/or **shrinks in
  amplitude** — the mouse learns to reject distractor fluctuations.
- **Null (reportable):** the kernel is flat (impulsive licks are stimulus-independent) and/or does
  not change with learning.

**Q2 — neural (Arm 2 / N-B).** Does the TF-responsive striatal population transiently
over-represent an upward fluctuation before FA licks (vs matched withhold), and does this shrink
with learning?
- **H2:** yes; and the **stimulus-matched** control dissociates two components — a *sensory*
  component (the population faithfully encodes the real, elevated fluctuation that triggered the FA)
  from an *excess-gain* component (encoding is transiently up **beyond** what the stimulus explains
  — an internal impulsivity signal). Either or both may change with learning.
- **Null (reportable):** no pre-FA over-representation beyond the stimulus-matched control, or no
  learning change.

**Q3 — state-resolved (Phase 2).** Do the behavioral and neural kernels differ between FAs emitted
in the **StimSens** state vs the **Impulsive** state?
- **H3:** StimSens-state FAs are genuine stimulus-driven false alarms → **sharp** kernel + faithful
  neural over-representation; Impulsive-state FAs are an internal "itch" largely decoupled from the
  stimulus → **flat/weak** kernel + little stimulus-locked neural signal.
- **Non-circularity:** state labels are defined from lick *rates/outcomes* (`f_inapplick`,
  `f_hit_hard`, `f_miss_easy`), which the labeler sees; the **stimulus kernel preceding those
  licks** is an independent measurement the labeler never uses. A kernel-*shape* difference across
  states is therefore a genuine finding, not a definitional artifact. (This is precisely the
  independent-readout move that breaks the documented state-labeler circularity caveat.)

---

## §2 Data & scope

**Subjects & regions.** All three have the full stack (staging manifest + local pkls with ingested
trials + TF-responsive registry + state tags), verified 2026-07-01:

| Subject | Region | FA licks Naive / Learning / Expert | TF-responsive units (`resp_log2`) |
|---|---|---|---|
| BG_046 | DMS | 1107 / 3078 / 2933 | 195 / 7047 (2.8%) |
| BG_039 | DMS | 609 / **28** / 1682 | 75 / 2442 (3.1%) |
| BG_031 | VMS | 1205 / 4314 / 5530 | 399 / 7537 (5.3%) |

Trial ingestion, `baseline_values`, and `reactiontimes['FA']` are **100% populated** on probed
BG_039 and BG_031 pkls (the older "no ingested trials for new subjects" memory note is stale).

**FA-by-state coverage** (confidence-gated `state_confidence ≥ 0.8`), for Phase 2:

| Subject | StimSens N/L/E | Impulsive N/L/E |
|---|---|---|
| BG_046 | 80 / 564 / 896 | 998 / 2063 / 1487 |
| BG_039 | 110 / 8 / 816 | 61 / 0 / 606 |
| BG_031 | 165 / 729 / 642 | 968 / 3384 / 4326 |

**Pooling rules.**
- **Behavioral (Arm 1):** pool all 3 subjects; also report per-subject (n=3 replication is the
  point). Region is irrelevant for behavior.
- **Neural (Arm 2):** **DMS pool = BG_046 + BG_039** (primary); **VMS = BG_031** reported
  **separately** (chronic-probe drift / no cross-region pooling rule). Aggregation pools by SUBJECT.
  NOTE (execution finding): `region_bank_confirmed` is `False` across the entire registry, so it is
  NOT used as a gate — per-unit region labels are provisional (the `region` column is the
  subject-level DMS/VMS target).
- **Unit of replication = session.** Bootstrap over sessions (and over subjects for the pooled
  behavioral estimate). **Never pool raw units across sessions** (within-session QC only → Simpson
  inflation; the N1 lesson).

**Headline contrast: Naive vs Expert.** Show Learning where populated. BG_039's Learning stage is a
single session (28 FA) → excluded from the Learning cell for BG_039.

**Known constraints (carried into §8).** sparse TF-responsive yield; VMS is n=1 region;
Naive-StimSens is the thinnest Phase-2 cell (neural especially); BG_039 has `_v2` split sessions
(needs the non-finite-timestamp guard); no video for BG mice.

---

## §3 Methods

### Shared: stimulus reconstruction (reuse, do not re-derive)
The per-trial baseline TF fluctuation is `Trial.baseline_values` (MATLAB `St1TrialVector`), stored at
60 Hz with each 50 ms TF value held for 3 frames. Recover the true 50 ms grid with **stride 3**,
anchored at `Baseline_ON`, zeroed at/after change onset, transformed to **log2 octaves** — exactly
as codified in `src/visdetect/analysis/tf_glm_data.py:466-560` (`session_trial_regressors`,
`_BASELINE_STRIDE=3`) and `decision_latents.build_trial_evidence_corrected(session, dt=0.05)`. **dt =
0.05 s** always (the `TF_SAMPLE_PERIOD=0.25` binning is a documented footgun — never use it here).

### Arm 1 — behavioral impulsivity kernel (I1)
1. **FA alignment.** For each `outcome=='fa'` trial, lick time = `reactiontimes['FA']` (seconds from
   `Baseline_ON`). Align to the **recorded** lick time — **no hardware-delay correction is baked in**.
   The rig's lick-sensor latency is currently uncalibrated (`LICK_HARDWARE_DELAY_MS=200` is a
   placeholder), so we do not assert it. A fixed delay is a pure constant time-shift applied
   identically to every stage/state → it **cancels** in the Naive-vs-Expert and
   StimSens-vs-Impulsive contrasts; **no scientific claim depends on it**. It only moves the
   *absolute* lag axis, which we therefore label "time before **recorded** lick." A `lick_shift_ms`
   parameter (default **0**) is exposed purely as a sensitivity check.
2. **Window.** Extract the log2-TF trace over `[t_fa − 1.5 s, t_fa − 0.15 s]` at dt=0.05 (~27 lags).
   Exclude the last 150 ms (sensorimotor refractory). Require ≥ the full window of pre-lick history
   and t_fa ≥ 0.5 s after `Baseline_ON`.
3. **Matched-withhold control.** For each FA, draw no-lick epochs from hit/miss trials of the same
   session **matched on time-in-trial** (same latency bin) and base TF; extract the same-lag window.
   **Kernel = FA-triggered mean − withhold-matched mean.** (Because the stimulus is white by design
   the raw STA is near-unbiased, but the withhold subtraction neutralizes the time-in-trial hazard
   structure and any residual autocorrelation — verify white-ness per session, don't assume.)
4. **Contamination guard.** Exclude FAs within 0.5 s of `change_time` and any FA on a trial where a
   real change had already occurred.
5. **CI & learning contrast.** Bootstrap over FA events (and matched withholds), 1000×, seed 42, for
   the 95% band. Compute the kernel **per stage**; **n-match** each stage to the smallest stage's
   usable-FA count before comparing. Report kernel **shape** (peak lag, half-width / effective
   integration window) and **amplitude** (peak) **separately** — the learning claim is primarily a
   *shape* result (adversarial gate: amplitude is confounded by FA-count and base-rate).
6. **Faithful anchor.** Overlay Khilkevich's descriptive lick-triggered TF kernel (raw FA-triggered
   mean TF over [−1.5, 0], 50 ms bins, bootstrap CI) so the reader sees that the withhold
   subtraction matters.

### Arm 2 — neural impulsivity kernel (N-B), signed-sum estimator
1. **Cells.** TF-responsive units only (registry `resp_log2==True`; sign = `sign(c1_r_log2)`).
   DMS-pool and VMS separately. (`region_bank_confirmed` is not gated on — False registry-wide.)
2. **Population TF signal (estimator (a), chosen).** Per unit, per-unit **shared-baseline z-score**
   (baseline window shared across FA/withhold; golden rule). Signed population sum
   **`S(t) = Σ_i sign_i · z_i(t)`**, where `sign_i ∈ {+1,−1}` is the unit's fast-/slow-TF preference
   from the registry. Robust to sparse yield (~4–9 responsive units/session), no fragile per-session
   decoder, and `S(t)` is directly comparable across sessions/subjects (z-units).
   *Sensitivity check:* a cross-validated ridge decoder (population → momentary log2-TF) reported as
   a secondary estimator where unit counts allow.
3. **Neural kernel.** `S(t)` time-locked to the **recorded** FA lick vs matched-withhold epochs.
   Leakage-safety does **not** rest on any assumed hardware delay: the readout window ends a
   conservative margin before the recorded lick, and the decisive control is the stimulus-matched
   decomposition (step 4), which is leakage-safe by construction regardless of the exact
   motor-onset offset. Report robustness to the window end-time. Per session then aggregated
   (bootstrap over sessions). *(When reusing `align.py`, pass `shift_fa_hit_ms=0` to override its
   default 200 ms shift.)*
4. **Sensory-vs-gain dissociation (the honesty move).** A **stimulus-matched** withhold control:
   compare FA epochs to withhold epochs carrying the *same* baseline log2-TF trajectory. The
   component of the FA neural kernel explained by the matched stimulus = **faithful sensory
   encoding**; any excess of FA over stimulus-matched withhold = **transient excess gain** (an
   internal impulsivity signal). Also compare the neural kernel to the *stimulus-predicted* kernel
   (behavioral kernel passed through the units' known tuning) as a second read of sensory vs gain.
5. **Learning contrast.** As Arm 1: per stage, n-matched, Naive vs Expert, bootstrap over sessions.

### Phase 2 — state-resolved (both arms)
Repeat Arms 1 & 2 within **StimSens** vs **Impulsive** FA trials (join state tags on `trial_idx`,
gate `state_confidence ≥ 0.8`; canonicalize session ids for the join; handle the 6-digit vs 8-digit
state-tag filename footgun). Compare kernel shape/amplitude across states, and across learning where
cells permit. Disengaged and Abort states excluded (few informative FAs; different mechanism).

---

## §4 Controls & truthfulness gate

| Confound | Control |
|---|---|
| FA-count imbalance across stages → amplitude artifact | n-match to smallest stage; report **shape** separately from amplitude; bootstrap |
| Time-in-trial / hazard structure in withholds | withhold epochs matched on time-in-trial + base TF |
| Regression-to-a-real-change contamination | exclude FAs within 0.5 s of `change_time`; drop post-change FAs |
| Lick-sensor latency (uncalibrated) | align to **recorded** lick; delay exposed as `lick_shift_ms` (default 0), **not baked in**; learning/state contrasts invariant (constant shift cancels); absolute peak-lag reported "relative to recorded lick" |
| Wrong TF binning | dt = 0.05 (stride-3), never 0.25; per-session white-noise autocorr check |
| No video → movement/whisking kernel | framed as "stimulus history preceding impulsive licks," not "pure sensory evidence"; stated as a hard limit |
| Neural cross-session non-comparability | within-session z, per-session-then-aggregate, never pool raw units (Simpson) |
| Neural sensory-vs-motor confusion | TF-responsive cells only; stimulus-referenced signed-sum; pre-movement window; **stimulus-matched control isolates gain** |
| Sparse TF-responsive units | robust signed-sum estimator; DMS pool (subject-level); per-session kernels averaged |
| State-label circularity | kernel *shape* is independent of the rate/outcome-based label definition (see §1 Q3) |
| BG_039 `_v2` non-finite timestamps | port the non-finite guard into `tf_glm.py::trial_bin_edges` before reconstruction |

---

## §5 Deliverables (figures)

- **Fig B10.1 — behavioral kernel.** (A) pooled kernel + Khilkevich anchor; (B) Naive-vs-Expert,
  per-subject (3) + pooled, n-matched, CI bands; (C) shape (half-width / peak-lag) and amplitude
  scalars vs stage, per subject.
- **Fig B10.2 — neural kernel.** (A) DMS-pool `S(t)` FA-vs-withhold + stimulus-matched control; (B)
  Naive-vs-Expert; (C) sensory-vs-gain decomposition; VMS (BG_031) as a separate panel/row.
- **Fig B10.3 — state-resolved (Phase 2).** StimSens-vs-Impulsive kernels, behavioral and neural,
  with the non-circularity note on-panel.

Each figure: `setup_style()` / `save_figure()`, a stats CSV sidecar, **plain-language caption**, and
the key limitation printed on the panel. Presentation-ready (per project figure convention).

---

## §6 Phases

- **Phase 0 — feasibility gate.** Largely discharged by the 2026-07-01 recon (numbers in §2);
  formalize as a coverage table (usable FA counts per stage/state/subject, pre-lick-history yield,
  TF-responsive-unit counts per session) with an explicit `usable` flag before scoring.
- **Phase 1 — state-blind.** Arm 1 (behavioral) then Arm 2 (neural). Behavioral first (fast, the
  hero figure); neural second.
- **Phase 2 — state-resolved.** Both arms, StimSens vs Impulsive.

---

## §7 Success criteria (pre-registered)

- **Behavioral.** Kernel significantly **upward** in the pre-FA window vs withhold (bootstrap CI
  excludes 0). **Learning effect:** kernel half-width and/or amplitude differ Naive vs Expert
  (bootstrap over sessions, CI excludes 0; replicated in ≥2/3 subjects for a confident claim).
- **Neural.** `S(t)` FA-triggered signal exceeds withhold (CI); the **stimulus-matched** control
  attributes it to sensory vs excess-gain; learning effect as above (per-session bootstrap).
- **State.** Kernel shape differs StimSens vs Impulsive (CI on the shape difference).
- **A flat kernel, no learning change, or no state difference is a clean, reportable negative** —
  pre-declared, so a null is a result (as with N1/B9).

---

## §8 Risks & limitations

1. **Sparse TF-responsive yield** (~3–5%/session) → neural arm leans on the robust signed-sum +
   pooling; single-session neural kernels are noisy (report aggregate).
2. **VMS is n=1 region** (BG_031); DMS is n=2 (046+039). No population claim beyond "consistent
   across the available striatal recordings."
3. **BG_039 Learning = 1 session** → Naive-vs-Expert only for that subject.
4. **Naive-StimSens is thin** (behavioral pooled ~355; neural much thinner) → Phase-2 Naive
   state-split may be underpowered; report with wide CIs or restrict to Expert + pooled.
5. **No video** → cannot exclude that the behavioral kernel is a movement/whisking kernel correlated
   with TF; framed accordingly.
6. **Single-lab task** (Orsolic-derived); block structure / air-puff differs from Orsolic, so FA
   rates are not directly comparable — the kernel *shape* and its *learning change* are the claim,
   not absolute FA rates or a temporal-expectation-gating result.
7. **State circularity** — defended via the shape-vs-definition argument (§1 Q3); do not describe the
   behavioral state difference itself as the finding.
8. **Provisional region labels** — `region_bank_confirmed` is False registry-wide, so neural pooling
   is by SUBJECT (known probe target) with region treated as provisional; there is no per-unit region
   gate. (The `_v2` non-finite guard is unneeded: `baseline_log2tf` does no timestamp arithmetic.)

---

## §9 Repo layout

New work under the topic-dir convention (import from `visdetect.*`, **not** analysis_suite):

- **Library:** `src/visdetect/analysis/psychophysical_kernel.py` — pure, tested functions:
  `reconstruct_tf_50ms()` (thin wrapper over the existing reconstruction), `fa_lick_times()`,
  `matched_withhold_epochs()`, `reverse_correlation_kernel()`, `signed_population_signal()`,
  `stimulus_matched_control()`, `bootstrap_kernel_ci()`. No plotting, no I/O.
- **Scripts:** `scripts/evidence_learning/` — `b10_phase0_coverage.py`, `b10_phase1_behavioral.py`,
  `b10_phase1_neural.py`, `b10_phase2_state.py`, `README.md`.
- **Caches:** `data/cache/evidence_learning/`. **Figures:** `FIGURES/evidence_learning/{SUBJECT}/`.
- **Reuse:** `tf_glm_data.session_trial_regressors` / `decision_latents.build_trial_evidence_corrected`
  (TF reconstruction), `decision_latents._decision_time_dl` (FA latency), `align.py` (event times),
  `config.canonical_session_id` / `load_staging_manifest`, `tf_responsive` registry loader,
  state-tag loader (filename footgun handling). *(Do not depend on `LICK_HARDWARE_DELAY_MS`; align
  to recorded lick with `lick_shift_ms=0`.)*

---

## §10 Validation (TDD)

`tests/analysis/test_psychophysical_kernel.py`:
- **Synthetic recovery** — generate synthetic FA licks driven by a known TF kernel; recover it
  within tolerance (the rigor backbone).
- **Determinism** — seed 42 → byte-stable kernel + CI.
- **Withhold-matching correctness** — matched withholds share the time-in-trial / base-TF
  distribution of their FAs.
- **White-stimulus autocorr** — per-session stimulus autocorrelation ≈ identity within tolerance.
- **Join integrity** — state-tag `trial_idx` join is 1:1 with pkl trials (gappy-index / decoy-column
  guard, per the N1 join gotcha); session-id canonicalization round-trips.
- **Signed-sum sanity** — on a synthetic population with known signs, `S(t)` tracks the planted
  stimulus.

All subagents that implement this run on **Opus 4.8** (project rule).

---

## §11 Literature grounding

- **Orsolic 2021** — paradigm parent; the psychophysical reverse-correlation / lick-triggered
  stimulus kernel is *the* signature. Faithful version = a **regularized multi-lag lick-hazard
  filter** (ARD/ridge over ~50 lags), **not** a 2-parameter "slow-exp + fast-derivative" fit. Block
  structure + air-puff differ from BG_046 → treat temporal-expectation-gating as out of scope; the
  kernel + its learning change is the claim.
- **Khilkevich & Lohse 2024** — same task; its lick-triggered TF kernel is our descriptive anchor.
  Its τ ≈ 0.27 s is a **model-comparison** output (integration vs outlier-detector), **not** a
  kernel width — do not conflate. Refractory first 150 ms excluded (we match this).
- **Lohse 2025** — sensory encoding ⊥ task-state; isolate the sensory channel via **TF-responsive
  cells** and a stimulus-referenced readout (why the neural arm avoids raw hit−miss). Preprint →
  provisional.
- Synthesis: `synthesis-batch06-brainwide-population`, `synthesis-phase3-pathways`,
  `synthesis-batch01-foundations` (memory `## Literature`).
