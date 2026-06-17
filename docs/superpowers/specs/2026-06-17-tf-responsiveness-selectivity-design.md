# TF-Responsive Cell Identification — Selectivity Redesign

**Created**: 2026-06-17
**Status**: Design approved (brainstorm), pending spec review → plan rewrite
**Supersedes**: `2026-06-15-tf-responsiveness-labeler-design.md` (source-level drift correction) — invalidated by its own Phase-0 gate (see §1). The HITL / active-learning / state-conditioning / integration scaffolding from that spec survives; only the **detection front-end** changes.
**Branch**: `feature/tf-responsiveness-labeler`. Work currently done in worktree `E:/python_analysis/git_repos/vd_tf_phase0` (primary repo is on another chat's branch).
**Template**: behavioral-state labeler (`2026-06-10-behavioral-state-labeler-design.md`): sparse human labels → shallow interpretable model → LOSO-κ → gated tagging.

---

## 1. Motivation & how we got here

"TF-responsive" is the **gateway label** for BG_046's sensory-coding story (learning emergence, D1/D2 push-pull, state gain, sensory-vs-motor, decoding). Downstream single-subject results are sensitive to a few mislabeled units, so the label must be trustworthy.

**The task signal.** During baseline, grating temporal frequency fluctuates iid every **50 ms** (log2 ~ N(0, 0.25²) octaves, **geomean ~1 Hz**). A "fast pulse" = a sample >+1 SD (= +0.25 log2); "slow" = <−1 SD. These pulses are the moment-to-moment sensory evidence (the rodent analog of motion-energy pulses).

**Phase-0 killed the drift approach.** The 2026-06-15 design tried to *detrend* a slow firing-rate ramp out of the pulse-triggered average, then threshold. The Phase-0 gate (script `scripts/tf_responsiveness/validate_drift_phase0.py`, run on session 16092025) showed:
- After fixing a KDE edge artifact (extend trace to −1.0 s), the "drift-only" units have genuine large pre-pulse slopes (~6–8) that run **monotonically across the whole window** — a **within-trial temporal-expectation ramp**, not slow session drift.
- This ramp lives at the **same timescale as the response**, so a continuous-train drift model can't separate them: a source-level kernel sweep (20 s → 0.25 s, fine bins) barely moved the slopes; a per-window linear detrend flattened the slope by construction but **injected huge fake post-pulse peaks** via extrapolation (post 2.8 → 14).
- Lohse 2025 (sister study) names this ramp: striatal **task-state / temporal-expectation** activity is a dimension **orthogonal** to the sensory-evidence axis.

**The pivot (Lohse's method).** Lohse identifies evidence-encoders with a **fast-minus-slow pulse selectivity** — which cancels the common-mode ramp **by symmetry**: the ramp is locked to trial time, not pulse identity, and fast/slow pulses sample it identically, so the difference removes it with **no detrend and no model**. A genuine signed encoder responds oppositely to fast vs slow (large difference); a pure ramping unit responds identically (difference ≈ 0). This also explains the old pipeline's unstable "Omni" tier: same-sign-to-both = common-mode = confound, which is why detrending collapsed it 167→36.

**Two foundational code bugs found in audit (the reused `tf_pulse.py`):**
1. **Off-by-one in `_collect_pulses` — FIXED (commit `e53ddd1`).** `enumerate(trials, 1)` indexed 0-indexed onset arrays, pairing each trial's TF values with the *next* trial's onset and dropping the last trial. Since per-trial TF sequences are independent (verified corr ≈ −0.03), this **scrambled the fast/slow→spike alignment**, corrupting all prior TF results (old classifier, NPZ caches, Phase-0 picks). Fix = 0-indexed enumerate + regression tests (`tests/analysis/test_tf_pulse_alignment.py`).
2. **Separate-baseline z-scoring — fix in `tf_selectivity.py`.** The old code z-scores `fast_z` and `slow_z` *each to its own pre-window SD*, then differences — CLAUDE.md's "circular baseline" error. It re-leaks the ramp: `ramp/σ_f − ramp/σ_s ≠ 0` when `σ_f ≠ σ_s`. Fix = **shared per-unit baseline** (Lohse z-scores both to a common baseline): `selectivity = (fast_hz − slow_hz)/σ_baseline`.

**Verified NOT bugs:** the `baseline_stride=3` + `sample_period=0.05` handling correctly recovers the true 50 ms sequence (raw vector is 3× upsampled); log2 stats match the papers. `TF_SAMPLE_PERIOD=0.25` is a *different* (unused-here) constant; its project-wide audit is tracked separately.

---

## 2. Goals & non-goals

**Goals (v1):** a trustworthy **binary** `is_tf_responsive` per `(session, cluster)`, built on the *corrected* pulse collection + Lohse fast-minus-slow selectivity (shared baseline) + a **label-shuffle null**, with human tags + a shallow model drawing the boundary (replacing Lohse's hard `>7.5`), LOSO-validated. Output drop-in for `tf_cell_classification.csv` consumers.

**Non-goals (deferred):** no GLM and no unsigned/**omni** detection in v1 (the GLM/|TF| model is the noted future path for omni); no Splitter-vs-Unilateral **sub-typing** (signed fast/slow info preserved so it's a no-re-run add-on); no cross-session tracking; no reinventing pulse detection (reuse fixed `_collect_pulses`).

---

## 3. Target definition

One label per unit: `is_tf_responsive ∈ {responsive, non-responsive}` + `model_score` + gated `confidence`. **Responsive = selective in at least one behavioral state** (permissive on the trial axis — a state-gated responder counts; the engaged/disengaged split prevents pooling from diluting it). A `borderline` tag is available to the human. *Where/under-what-state* a unit responds is characterization, not identity.

Unit pool: `good_and_stable_ids`, QC-filtered staging manifest.

---

## 4. Detection core — `tf_selectivity.py` (new)

Reuses `tf_pulse.py` (`_collect_pulses` *fixed*, `_mean_activity_per_unit`, `_zscore_trace`). Per unit (× state):

1. **Pulses:** `_collect_pulses(session, cfg)` with `use_constraints=True` — `±0.25` log2 thresholds (= ±1 SD), exclude <1 s after baseline, <1 s before change, <2 s before fa/abort/ref lick. (= Lohse's protocol.)
2. **Traces over `[−1.0, +0.5]` s** (extended KDE support so the −0.4 edge is clean): fast and slow pulse-triggered mean rates in **Hz** (`_mean_activity_per_unit` returns spikes-per-`dt`-bin → divide by `dt`).
3. **Shared-baseline selectivity (the fix):** compute one per-unit baseline SD `σ_b` from the pre-pulse window pooled across **all** baseline pulses (mean cancels in the difference). Then `selectivity(t) = (fast_hz(t) − slow_hz(t)) / max(σ_b, ε)`. Also keep `fast_z = (fast_hz − μ_b)/σ_b`, `slow_z` likewise (for display/sub-typing) — all to the **same** `μ_b, σ_b`.
4. **Selectivity metric:** signed peak + AUC of `selectivity` in the post window (per-unit peak across the post window, so transient ~0.12–0.17 s *and* sustained both register).
5. **Label-shuffle null:** randomly relabel pulses as fast/slow keeping counts, recompute `selectivity`, repeat (≥200). Preserves the ramp/drift entirely; destroys only the TF assignment → clean significance. Store the null envelope (for the GUI) and a shuffle p / selectivity-z-vs-null.

**Minor `tf_pulse` fixes to land here (Phase B):** (a) `_smooth_binned_activity` uses a binary 0/1 train → undercounts ≥2 spikes/bin; use `np.add.at`. (b) `n_seen=None` + NaN-change (abort/fa) → unbounded baseline leakage; bound pulse collection by the outcome time and skip trials where neither change nor outcome time is known.

---

## 5. Features (shallow model)

From the clean selectivity, per unit × state:
- Selectivity signed **peak** + **AUC** (post window); **peak latency**; **half-width** (transient vs sustained).
- **Signed fast & slow peaks** separately (preserved for later unilateral-vs-splitter sub-typing).
- **Selectivity z relative to its label-shuffle null** + shuffle p (the significance).
- **Within-state split-half** reproducibility of the selectivity (real-vs-noise).
- **State-gating index** (engaged − disengaged selectivity); `n_pulses` per state (sufficiency guard).

---

## 6. HITL GUI

Tag responsive/not on the **selectivity trace** (the confound-robust decision signal), with fast ▲ / slow ▼ shown for context and the **shuffle-null band** overlaid. Shared-axis vertical stack; **monochrome** (glyph + solid/dashed, no color in v1); low-opacity post-pulse window marker; readable rasters; **mouse buttons + keyboard** (`r`/`n`/`b`/confidence). All-trials and engaged-only selectivity stacked (state-gating); sidebar shows per-state `n_pulses` with an "insufficient" guard. Crash-safe append-only `TFLabel` CSV. Visuals tuned on real exemplars in Phase B.

---

## 7. Model & active-learning loop

Shallow decision tree / logistic (inspectable, class-weighted). Loop: **seed** tags (clear responsive from the gate's exemplars + clear non-responsive) → **train** (`calibrate_model.py`, report LOSO κ + the rule) → **rank** all units by uncertainty, surfacing **engaged-strong / all-weak** units → **retag → retrain** to κ plateau.

---

## 8. Validation & success criteria

- **LOSO κ** headline (target ≈ state-labeler bar, κ ≈ 0.7); held-out precision/recall; the learned rule readout.
- **Agreement vs Lohse's hard `>7.5`** selectivity threshold (sanity).
- Within-state split-half reproducibility on tagged-responsive units.
- **Yield sanity:** expect *sparse* (Lohse ~3%, posterior-biased) and **possibly sparser in BG_046's medial striatum** — a near-zero yield is a finding, not a method failure.
- Detection sanity: real responders' selectivity peaks at short latency (~0.12–0.17 s) and exits the null.

**Done-tests:** LOSO κ ≥ ~0.7 with an inspectable rule; responder set robust to seed/threshold perturbation; `tf_responsive_tags.csv` is a drop-in replacement.

---

## 9. Architecture & data flow

**Library** (`src/visdetect/analysis/`): **`tf_selectivity.py`** (new — traces, shared-baseline selectivity, label-shuffle null, features); **`tf_responsiveness.py`** (TFLabel I/O, GUI render, active-learning queue, shallow model + LOSO). **Retired:** `tf_drift.py` + its tests.

**Scripts** (`scripts/tf_responsiveness/`): `fit_tf_selectivity.py` (per-session/unit features → cache), `run_labeler_gui.py`, `calibrate_model.py`, `tag_units.py`, `validate.py`. Keep `validate_drift_phase0.py` as the record of the pivot.

**Data:** `data/labels/tf_responsive_labels.csv`; `data/cache/tf_selectivity/BG_046/*.npz`; `tf_responsive_tags.csv`.

```
corrected pulses (fixed _collect_pulses)  ──►  fast_hz, slow_hz over [-1,+0.5]
        │ shared per-unit baseline σ_b
        ▼
selectivity = (fast_hz - slow_hz)/σ_b  +  label-shuffle null   → features → cache
        │                                          │
   GUI: tag on selectivity + null band     shallow model (active-learning, LOSO)
        └───────────────────────────►  tag-all → tf_responsive_tags.csv
```

---

## 10. Migration

`tf_responsive_tags.csv` columns: `session_name, cluster_id, stage, cell_type, is_tf_responsive, model_score, confidence` + signed selectivity summaries. Low-confidence **gated**. Old `g_tf_cell_classifier.py`/Fig41 and the existing NPZ caches are **stale** (built on buggy pulses) — regenerate; retire the old classifier after the new tags validate; re-point `08_tf_pulse` consumers.

---

## 11. Phasing (drives the rewritten plan)

- **A — corrected foundation** *(DONE)*: off-by-one fix + case-robust lick guard + (to add) shared-baseline rule + the two minor `tf_pulse` fixes.
- **B — `tf_selectivity.py` core + EARLY-VALIDATION GATE** (the new Phase-0): build traces/selectivity/null/features (TDD); run on 1–2 trusted sessions; confirm a sensible *sparse* population whose selectivity exceeds the null at short latency; **re-pick clean exemplars** from corrected data (old picks suspect). Proceed only if a real population exists.
- **C** — full state-conditioned extraction → feature cache (parallelized, `ProcessPoolExecutor`).
- **D** — GUI + seed tags. **E** — active-learning loop + shallow model + LOSO. **F** — tag-all + validation + integration.

---

## 12. References

- Lohse et al. 2025 (sister study, identical task) — `paper_references/Lohse et et al., Frontal cortex gates striatal dynamics ... 071025.pdf`. Methods: ±1 SD fast/slow pulse detection; guards 1 s/1 s/2 s; 40 ms FWHM smoothing; z-score to common baseline; fast-minus-slow selectivity peak `>7.5` → 109 evidence-encoders (3.3%, posterior-biased).
- Khilkevich & Lohse 2024 (Nature, the CLAUDE.md reference; per-neuron Poisson GLM — the future omni path). Orsolic 2021 (task origin).
- Method template: `2026-06-10-behavioral-state-labeler-design.md`; `src/visdetect/analysis/state_labeling.py`.
