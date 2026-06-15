# TF-Responsive Cell Identification — Clean-Slate Redesign

**Created**: 2026-06-15
**Status**: Design approved (brainstorm), pending spec review → implementation plan
**Supersedes**: the rule-based tiered classifier (`analysis_suite/08_tf_pulse/g_tf_cell_classifier.py`, Fig 41) and the April-2026 manual-labeling scaffold (`src/visdetect/analysis/tf_labeling.py`, `scripts/tf_labeling/`), neither of which is trusted. Old code stays in place untouched until the new pipeline validates, then is retired.
**Template**: mirrors the successful behavioral-state labeler (`2026-06-10-behavioral-state-labeler-design.md`): sparse human labels → shallow interpretable model → LOSO κ validation → gated tagging of all sessions.

---

## 1. Motivation — why this is worth a clean slate

"TF-responsive" is the **gateway label** for the entire sensory-coding story in BG_046. Learning emergence, state-dependent gain, sensory-vs-motor dissociation, D1/D2/FSI breakdowns, and the DDM evidence axis all condition on it. If the label is wrong, every downstream figure inherits the error.

We have a concrete signal that it *is* wrong: in the current pipeline, toggling detrending on/off reclassifies three-quarters of the "Omni" tier (167 → 36). When one preprocessing switch flips that many units, the method is measuring an artifact, not biology.

### What the TF pulses are
During the baseline period the grating's temporal frequency fluctuates stochastically (~50 ms updates) around a **mean of ~1 Hz** (a slowly drifting grating). These fluctuations are the **moment-to-moment sensory evidence stream** — the rodent-striatal analog of motion-energy pulses in the Shadlen/Huk RDK paradigm. A "fast pulse" = an instant TF jumped up; a "slow pulse" = an instant it dropped.

### What a TF-responsive neuron is thought to be
A unit whose firing is time-locked to those fluctuations is **carrying the sensory evidence** the animal must monitor and integrate to detect a change — in DDM terms, a cell feeding the **drift**. The lab's working interpretation (Orsolic 2021; Khilkevich & Lohse 2024; Lohse bioRxiv) is evidence-integration; both transient-fast and extended response shapes are observed, and it is **not yet known** whether they are one mechanism or two, whether either is truly evidence-encoding, or whether the identity is stable across a session/regions. The pipeline must therefore stay agnostic to the transient-vs-sustained distinction (see §6, §7).

This is **medial/associative striatum (DMS)** — not a visual sensory area (not Cover's visual tail, not Tang's auditory pDS). Signed TF-evidence coding *here* is a statement about how associative striatum holds the sensory variable for a decision.

### Why the baseline window is special
Baseline TF pulses occur **before any motor preparation or execution**. This makes them doubly valuable: they isolate the **sensory** component from the action-selection/motor confounds that saturate striatum, *and* they are the one window where state-dependent sensory gain can be read cleanly (the Impulsive state appears higher-gain than StimSens, but post-decision that is entangled with motor prep — baseline is not).

### Why the drift artifact is load-bearing, not cosmetic
The target effect is a small (~1–2 Hz on a ~10 Hz baseline), ~50–250 ms pulse-locked modulation riding on slow firing-rate drift (arousal, satiety, electrode motion). Every scientific claim here is **comparative** — Naive vs Expert, D1 vs D2, engaged vs disengaged. If drift correlates with the comparison variable (Expert sessions drifting more; high-FR D1 cells showing larger absolute slopes), a sloppy baseline manufactures **false biology** that mimics the hoped-for result. Getting the baseline genuinely flat is what licenses any comparative interpretation.

---

## 2. Goals & non-goals

### Goals (v1)
- A trustworthy, **binary** `is_tf_responsive` label per `(session, cluster)`.
- A drift-correction step that **provably** flattens the pre-pulse baseline (objective QC, not eyeball faith).
- A human-in-the-loop **tagger** + **active-learning loop** that learns the label from sparse expert tags, validated with LOSO κ.
- Output drop-in compatible with existing `tf_cell_classification.csv` consumers.

### Non-goals (explicitly deferred)
- **No 4-tier sub-typing** (Splitter / Unilateral / Omni) and no signed-direction label in v1. Signed fast/slow information is preserved end-to-end so sub-typing is a later no-re-run add-on.
- **No deep/CNN trace model** — shallow interpretable features only.
- **No cross-session neuron-tracking integration** — each `(session, cluster)` is tagged independently. The "does responsive identity drift across the session / across regions" question is acknowledged but out of scope.
- **No reinventing pulse detection** — reuse `_collect_pulses()` from `tf_pulse.py` (with its trial-onset / change-onset / early-lick guards), but **verify** those guards and **add** a general lick-exclusion guard — see §5 "Pulse selection".
- **No 3-way state-resolved responsiveness** (StimSens / Impulsive / Disengaged separately) — v1 uses a *binary* engaged-vs-disengaged split only (see §3, §5); per-state resolution is the later characterization layer (thin pulse counts per state otherwise).
- **State is the only trial-heterogeneity handle in v1** — other possible gates (arousal, pupil, time-on-task) are not modeled; behavioral-state tags are the principled available proxy.

---

## 3. Target definition

One label per unit: `is_tf_responsive ∈ {responsive, non-responsive}`, plus a model probability `model_score` and a gated `confidence`. A `borderline` tag is available **to the human** during labeling (it is gold for defining the boundary) but collapses to the binary target for the model, or is held out as an explicit "unsure" class — decided at calibration (§8).

**Responsive = TF-responsive in at least one behavioral state** (engaged *or* disengaged). The identity is deliberately *permissive* on the trial axis: a neuron that responds only when engaged is still TF-responsive. This prevents pooling-across-all-trials from diluting (and silently discarding) state-gated responders — the more dangerous failure mode, since state-gated units are precisely the StimSens/Impulsive-gain population of interest. *Where* a unit responds (its state-dependence) is a **characterization**, not the identity (§5–§7), mirroring the binary-now / sub-type-later split applied to the trial axis.

Unit pool: `good_and_stable_ids` (the within-session QC pool; pkls store spikes only for these), on the QC-filtered staging manifest (`load_staging_manifest(qc_only=True)`). Same selection rules as the rest of the project.

---

## 4. Architecture

Mirrors the state-labeler file layout.

### Library — `src/visdetect/analysis/`
- **`tf_drift.py`** *(new)* — source-level drift correction + re-extraction. Reuses `_collect_pulses()` for pulse times. Produces drift-corrected, z-scored pulse-triggered traces — **both all-trials and engaged-only**, by masking pulses with the per-trial behavioral-state tags (`data/cache/state_tags/`) — plus a per-unit drift-QC record and the circular-shift null bank. Single owner of the "clean trace" definition. (Note the known state-tag session-id gotcha: tag/pkl ids are zero-padded `zfill(8)` vs the manifest's leading-zero form.)
- **`tf_responsiveness.py`** *(new)* — the state-labeler analogue: `TFLabel` dataclass + crash-safe append-only CSV I/O, feature extraction from the clean caches, active-learning queue/ranking, shallow model fit/predict/LOSO, and GUI render helpers (kept import-light; matplotlib deferred inside functions, as in `state_labeling.py`).

### Scripts — `scripts/tf_responsiveness/` (mirrors `scripts/state_labeling/`)
- `extract_detrended_traces.py` — one-time re-extraction pass → NPZ caches.
- `run_labeler_gui.py` — binary tagging GUI; queue ordered by model uncertainty.
- `calibrate_model.py` — fit shallow model from labels; report LOSO κ + the learned rule.
- `tag_units.py` — apply model to full population → `tf_responsive_tags.csv` (gated).
- `validate.py` — agreement figures + detrend QC + downstream sanity.

### Data
- `data/labels/tf_responsive_labels.csv` — human tags (grows incrementally).
- `data/cache/tf_detrended_traces/BG_046/*.npz` — clean traces (one per session).
- `data/cache/tf_responsive_tags/BG_046/` + `tf_responsive_tags.csv` — model output.

### Data flow
```
raw pkl spikes ──► per-unit slow-drift model (whole session)
                      │ subtract at source (+ restore mean)
                      ▼
            detrended rate ──► pulse-triggered avg (fast/slow, signed) ──► z-score ──► NPZ cache
                                                                                │
                        ┌───────────────────────────────────────────────────┘
                        ▼                                            ▼
              GUI renders traces ◄── human tags (binary)     feature extraction
                        │                                            │
                        └──────────► active-learning loop ◄──────────┘
                              (seed → train → rank uncertain → retag → retrain)
                                                  │
                                                  ▼
                                    tag all units → tf_responsive_tags.csv
```

---

## 5. Drift correction & re-extraction (the detrend, done right)

**Principle**: leave nothing in the pulse-triggered trace except genuinely pulse-locked signal. Justified by a **timescale separation** — signal at ≤ sub-second (50 ms pulses), drift at minutes. The strategy is standard (remove slow nonstationarity before event-triggered averaging); only the *parameters* are empirical, and they are tuned to an objective target, not guessed.

1. **Estimate slow drift per unit, whole session.** Compute the unit's continuous firing rate; extract the slow component with a wide kernel (σ on the order of several seconds) or a robust running median / spline on session time. Real TF modulation is fast and ≈ zero-mean over seconds, so a multi-second estimator captures drift while averaging out signal. Kernel width / spline df is the one knob.
2. **Subtract at the source.** `detrended_rate(t) = rate(t) − drift(t) + mean_rate` (mean restored so z-scoring is stable). A multiplicative/gain mode (`rate(t)/drift(t) × mean_drift`) is available for gain-like drift; choice is decided by the flat-baseline QC, not by hand.
3. **Re-average + z-score on the clean signal.** Align the *detrended* rate to fast- and slow-pulse times separately → signed fast/slow mean traces; z-score to the now-flat pre-pulse window so the SD is real noise, not drift inflation. This is the structural fix the old post-hoc detrend could not make (it operated on the already-z-scored mean, whose SD was already contaminated).
4. **State-conditioned variants (binary).** Repeat the pulse-triggered averaging over three pulse subsets, masked by the per-trial state tag: **all-trials**, **engaged** (StimSens + Impulsive), **disengaged** (Disengaged). The drift model is fit once on the full train (step 1); only the *averaging* is conditioned. A `n_pulses` count is stored **per condition**, and an engaged/disengaged trace is flagged unreliable when its pulse count falls below a minimum (guards against thin splits, common in early/late stages).
5. **Generous window, full profile cached.** ~ −0.5 → +0.75 s, **no fixed sub-window threshold**. NPZ per session, for each of {all, engaged, disengaged}: `cluster_ids, t_vec, fast_z, slow_z, fast_z_sem, slow_z_sem`, signed scalar summaries (peak / AUC / latency / half-width, both directions), `n_pulses`, and **drift-QC scalars** (pre-pulse residual slope before & after; drift magnitude; FR). Single source for GUI + features + the null envelope.
6. **Circular-shift null bank.** For each unit, store the spread (e.g., 5–95 % envelope) of pulse-triggered traces under circular time-shift of the spike train. This is (a) the GUI's null envelope (§6) and (b) an independent significance cross-check that does not depend on the drift model being correct.

### Pulse selection (reused, with guards verified + one added)
Pulses come from `_collect_pulses(session, cfg)` with `use_constraints=True` (the dataclass default; call it **explicitly** — an inline comment in the function wrongly says constraints are "off by default"). Existing guards (`TFRespPulseConfig`): pulse must be ≥ `min_after_baseline` (1.0 s) after Baseline_ON, ≥ `min_before_change` (1.0 s) before Change_ON, and ≥ `min_before_outcome_fa_abort` (2.0 s) before the FA/abort reaction time. Our extraction window (−0.5 → +0.75 s) sits inside these with margin (a passing pulse + 0.75 s stays ≥ 0.25 s clear of the change).

Two required changes before we trust this for a *motor-clean* baseline:
1. **Verify the early-lick guard fires.** `_outcome_time_for_trial` matches `trialoutcome in ("FA","abort")` — **uppercase** — and reads `reactiontimes`. The project's outcomes are lowercase (`'fa'`); if so, the guard is silently inert. Phase 0/1 asserts on real pkls that the guard actually drops pulses (and fixes the case match if needed).
2. **Add a general lick-exclusion guard.** Drop any pulse whose [pre, post] window overlaps *any* lick time (not only FA/abort outcomes), so reflex/stray baseline licks can't contaminate the sensory window. This is additive to the reused detection logic.

### Success test for the detrend itself
- **Flat-baseline (primary):** the population distribution of post-correction pre-pulse residual slope collapses toward 0 (wide spread before → tight at 0 after).
- **Negative control:** toggling detrend on/off no longer flips classifications wholesale (the Omni-collapse symptom disappears).
- **Positive control:** a known responder still shows its pulse-locked deflection after correction (we did not eat real signal).

### Caveats baked in
- The dense 50 ms pulse cadence means any "pre-pulse" baseline still contains *other* pulses; this is handled by averaging over many random-context pulses (the average local context is flat), not by hunting for pulse-free baseline.
- Low-FR units get a noisy drift estimate → guarded (skip/widen kernel) and flagged, not force-corrected.
- The generous window + an interior baseline removes the old KDE-edge re-baselining hack (−300/−50 ms).

---

## 6. Labeling GUI

**Layout — shared-axis vertical stack.** One column on a single time axis (t from pulse), vertically aligned so the eye can drop a line through a deflection and check the spikes underneath:
```
┌──────────────────────────────┬─────────┐
│ Z-TRACE all-trials  ▲f ▼s     │ s7      │
│  + faint null envelope        │ clu42   │
├──────────────────────────────┤ D1 hi   │
│ Z-TRACE engaged-only          │ score   │
│  + null envelope              │ .82     │
├──────────────────────────────┤ nE 140  │
│ FAST raster  ▲ (tall)         │ nD 62   │
├──────────────────────────────┤202/4700 │
│ SLOW raster  ▼ (tall)         │ [r][n]  │
├──────────────────────────────┤ [b]ord  │
│ raw PSTH (Hz)                 │ [d]toggl│
└──────────────────────────────┴─────────┘
   one shared x-axis            sidebar
```
Deliberate panel proportions via `gridspec` `height_ratios` (z-traces + rasters tall, PSTH thin), per the project's panel-proportion convention — not an equal grid.

**State views (catch state-gated responders).** The trace column shows **all-trials** and **engaged-only** z-traces stacked, each with its own null envelope, so "flat overall but clear when engaged" is visible at a glance rather than depending on the tagger to remember to look. `d` toggles the disengaged trace. The sidebar shows per-condition pulse counts (`nE`/`nD`); when an engaged/disengaged count is below the reliability guard, that panel reads "insufficient" instead of plotting a noisy trace. The active-learning queue (§8) additionally surfaces **engaged-strong / all-weak** units so these are not missed.

**Monochrome.** Black traces. Fast vs slow distinguished by **panel subtitle + glyph** (▲ fast / ▼ slow) and, in the overlaid trace, solid (fast) vs dashed (slow) lines. No color at this stage — we tag responsive/not, not direction; color is reserved for later sub-typing when direction becomes the decision.

**Null envelope (the detection aid).** The averaged trace is the detector for small modulations (the raster cannot reveal a small mean shift against Poisson noise — that is expected, and the raster's job is only to confirm the response is not driven by a few trials). Overlay a **faint gray band = the circular-shift null spread** for that unit, so a real deflection is one that visibly leaves the noise envelope. This calibrates the eye to each unit's own noise floor and directly guards against "seeing something that isn't there." (Grayscale density-heatmap raster and a cumulative-residual/CUSUM panel were considered and are **Phase-0-revisitable options**, not v1.)

**Post-pulse window marker** drawn at low opacity (or thin boundary lines) — orient without manufacturing the impression of a response.

**Readable rasters:** adequate row height, clean spike marks, hard pulse-onset line at t=0, trials optionally sorted by spike density so a real band pops.

**Tagging action — keyboard *and* mouse.** Keys `r` responsive · `n` non-responsive · `b` borderline · `confidence` high/med/low · free-text notes, **and** on-screen clickable buttons (matplotlib `Button` widgets on the TkAgg backend) for the primary actions — `[Responsive]` `[Non-responsive]` `[Borderline]` `[Skip]` plus nav `[◀ Prev]` `[Next ▶]` — so the queue can be worked entirely by mouse if preferred. Both paths call the same save handler. Auto-save after every action to the crash-safe `TFLabel` CSV (header-guard for zero-byte files, as in `save_episode`). Big title with cluster + live model score; keyboard hints always on screen.

**Visual sign-off:** legibility is tuned against **real exemplar traces in Phase 0**, not a mockup.

---

## 7. Feature set

Interpretable, from the clean NPZ — both directions, both temporal regimes (so transient-vs-sustained is described, never thresholded). The magnitude/timing/shape features below are computed for **both the all-trials and the engaged condition**, so a state-gated response surfaces even when the all-trials value is weak:
- **Magnitude:** signed peak z, signed AUC — fast & slow.
- **Timing:** peak latency — fast & slow.
- **Shape:** half-width; transient-vs-sustained ratio (early-peak vs late-window mean).
- **Cross-direction:** mirror score `corr(fast, −slow)` (structured-tuning signal; seeds later splitter work).
- **State-gating index:** engaged − disengaged response magnitude — a feature *and* a scientific readout (how state-dependent is this unit's TF response). Computed only where both conditions clear the per-condition pulse guard.
- **Reliability:** split-half reproducibility computed **within-state** (odd/even within engaged trials), so a state-gated neuron is not penalized for the two halves differing in state mix — strongest real-vs-noise discriminator.
- **Exemplar template:** max correlation of the unit's trace to the human-confirmed "perfect example" templates — folds the perfect examples directly into the math.
- **Independent cross-check:** circular-shift shuffle p-value.
- **QC:** post-detrend pre-pulse slope (should ≈ 0); `n_pulses` (all / engaged / disengaged).

---

## 8. Model & active-learning loop

**Model:** shallow decision tree or logistic regression — inspectable, small-data friendly, class-weighted for the rare positive. `calibrate_model.py` prints the learned rule so the experimenter can see exactly what it keyed on (and confirm it is not leaning on `n_pulses` or a session artifact).

**Active-learning loop:**
1. **Seed** — experimenter tags a starter set: the "perfect examples" as clear *responsive* + obvious *non-responsive* (a few dozen each) to define both poles.
2. **Train** — fit shallow model; report LOSO κ + the rule.
3. **Rank** — score all ~4,700 units; re-order the queue so the **most uncertain** units surface next (plus a sample of confident-but-unlabeled predictions for spot-checking, and **engaged-strong / all-weak** units so state-gated responders are actively brought to the tagger).
4. **Retag → retrain** — label where it is unsure; repeat until κ plateaus and the boundary stops moving.

This spends labels on the decision boundary, which is what makes a rare class (~5–8 % positive) tractable.

---

## 9. Validation & success criteria

**Model validation (mirrors state labeler):**
- **LOSO (leave-one-session-out) κ** as the headline; target ≈ the state-labeler bar (κ ≈ 0.7).
- Held-out precision/recall; feature-importance / tree readout.
- Agreement map vs the *old* tiers — disagreements are expected to be the drift-artifact units, itself a confirmation.

**External checks (do not assume labels are perfect):**
- Within-state split-half response reproducibility on tagged-responsive units (state-gated units checked in their engaged trials).
- Sensible spread across stages & cell-types (not all from one session); and a first look at the **state-gating-index distribution** — how much of the responsive population is state-gated vs stably responsive (this is a result, not just a QC).
- Detrend QC from §5.

**Project done-tests:**
1. **Detrend:** population pre-pulse slope collapses to ≈ 0; detrend toggle no longer flips classifications wholesale.
2. **Model:** LOSO κ ≥ ~0.7 with an inspectable, biologically sensible rule.
3. **Stability:** responsive set robust to reasonable threshold/seed perturbation.
4. **Output:** `tf_responsive_tags.csv` is a drop-in replacement for `tf_cell_classification.csv` consumers.

---

## 10. Integration / migration

`tf_responsive_tags.csv` columns: `session_name, cluster_id, stage, cell_type, is_tf_responsive (bool), model_score (prob), confidence`, plus signed scalar summaries carried through for later sub-typing. Low-confidence predictions **gated** (flagged) so downstream can choose strict (high-confidence only) or inclusive.

Old `g_tf_cell_classifier.py` / Fig 41 stays untouched until the new tags validate; **then** retire it and re-point the `08_tf_pulse` consumers (`a, b, d, e, f, h, i`) to read the new tags.

---

## 11. Phased plan

- **Phase 0 — drift-correction validation (eyeball gate).** Run the correction on ~5 obvious responders + ~5 obvious drift-only units; produce a before/after figure. Commit only if it flattens the drift-only units while preserving responders. Adjust kernel / additive-vs-gain / fall back to per-epoch *before* scaling. Also the moment to tune GUI legibility on real traces.
- **Phase 1 — full re-extraction + NPZ caches** (drift correction + null bank + all/engaged/disengaged state-conditioned averaging, all manifest sessions).
- **Phase 2 — GUI + seed labels** (`tf_responsiveness.py` I/O + render, `run_labeler_gui.py`, seed tagging).
- **Phase 3 — active-learning loop + model + LOSO** (`calibrate_model.py`, iterate to κ plateau).
- **Phase 4 — tag-all + validation figures + integration** (`tag_units.py`, `validate.py`, re-point consumers, retire old classifier).

---

## 12. Open questions / risks

- **Kernel width / drift model** is empirical → resolved by the Phase 0 gate + flat-baseline QC; per-epoch linear detrend is the documented fallback.
- **Borderline handling** (collapse to binary vs explicit "unsure" class) → decided at calibration from how separable the tags are.
- **Class imbalance** (~5–8 % positive) → class weights + active-learning boundary sampling; watch precision/recall, not just κ.
- **State-split data sufficiency** → engaged/disengaged pulse counts thin in some sessions (stage-dependent); per-condition guard + "insufficient" display prevent noisy-trace false calls, but may leave some units judged on all-trials only — acceptable for a permissive (any-state) target.
- **State-tag dependency** → relies on the behavioral-state labeler caches existing for the session (`data/cache/state_tags/`); units in untagged sessions fall back to all-trials only. Mind the `zfill(8)` session-id format gotcha.
- **Pulse-guard correctness** → the reused early-lick guard may be silently inert due to an uppercase `trialoutcome` match against lowercase data (§5); must be verified on real pkls before any results are trusted, and the general lick-exclusion guard added.
- **Drift estimate eating signal** → guarded by the positive control + the independent shuffle null.
- **Generalization across stages** → LOSO is by session; if Expert/Naive differ systematically, inspect per-stage κ.

---

## 13. References

- Khilkevich & Lohse, *Nature* 2024 — brain-wide change-detection dynamics, ~250 ms integration. Lohse bioRxiv (sister study, same task). Orsolic 2021 — task origin / evidence integration.
- Huk & Shadlen 2005; Roitman & Shadlen 2002 — motion-pulse LIP integration (TF-pulse analog). Marica 2025 (preprint) — striatal sensory responses rise before behavior & before cortex across learning.
- Project literature syntheses: `synthesis-batch01-foundations`, `synthesis-batch06-brainwide-population`, `synthesis-phase3-pathways`, `synthesis-phase3-behavioral-state`.
- Method template: `docs/superpowers/specs/2026-06-10-behavioral-state-labeler-design.md`; library `src/visdetect/analysis/state_labeling.py`.
