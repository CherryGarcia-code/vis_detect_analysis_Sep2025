# Track Curation — Precision Curation of Cross-Session Neuron Tracks

**Date:** 2026-06-07
**Status:** Design (approved sections; pre-implementation)
**Subject:** BG_046 chronic Neuropixels 2.0, medial striatum, visual change-detection task

---

## 1. Goal & Guiding Principle

UnitMatch (UM 3.2.9) produces a cross-session registry of putative same-neuron
tracks, but a large fraction of long "tracked" UIDs are not reliably the same
unit across days (bimodal cross-session ISI overlays = same template, different
neuron — see [[tracking_qc_bimodal_isi_finding]]). The DeepUnitMatch line was
closed because its higher track *counts* were over-merges that fail ISI
validation ([[neuron_tracking_may2026]]).

This pipeline **curates UM tracks for precision**, on five locked decisions:

| Tag | Decision | Meaning |
|-----|----------|---------|
| **a** | Precision-only curation, **never reject** | The original UM registry is never altered or deleted. We *annotate* links and tracks with confidence and emit a derived trusted subset. |
| **liberal** | Start from the **liberal** UM registry | Begin with the most candidate links (`batch_uid_liberal`), then curate *down* — rather than starting strict and missing real cells. |
| **ii** | **Per-link, Expert→Naive backward sweep** with rolling anchor + gap-bridge tolerance | Validate each cross-session link individually, walking reverse-chronological; the anchor rolls forward as we go; a failing session is *skipped* (not chain-breaking) so a resurfacing unit is re-acquired. |
| **state** | **State-conditioned** functional fingerprints | Functional features computed on engaged ("in-zone") trials only; the state labeler is pluggable. |
| **c** | Validate with **held-out ISI** (+ optional hand-labeled gold set later) | Curation is scored by a spike-partition-independent ISI fingerprint, with a hand-labeled gold set deferred as a future extension. |

The **product** is a curated track table whose `trusted` rows downstream
engaged-state neural analyses can rely on.

### Literature grounding
- **ISI-fingerprint validation** follows the UnitMatch paper method (van Beest et
  al. 2024, *Nat Methods*): a real neuron's ISI distribution is highly stable
  across days; matched cross-day vs non-matched within-day ISI-correlation
  discriminability (AUC) measures whether tracking is real. Already implemented
  in `validate_long_tracks.py`.
- **State-conditioning** is motivated by the behavioural-state synthesis
  (`synthesis-phase3-behavioral-state`): GLM-HMM latent states are different
  cue→action *mappings* (Calhoun 2019), and spontaneous movement/engagement
  drives a large brain-wide signal that includes striatum and is orthogonal to
  sensory coding (Stringer 2019). A unit's evoked response shape is therefore
  only a stable fingerprint *within a behavioural state* — hence we fingerprint
  on in-zone trials.

---

## 2. Scope

**In scope:** curating the existing UM liberal registry into per-link verdicts
and curated tracks with confidence tiers; a pluggable state interface; built-in
held-out-ISI validation.

**Out of scope (this spec):** re-running UnitMatch; re-extracting raw waveforms
(reuse UM's, recoverable from X:); building a new state-identification method
(only the *interface* is built now; the HMM provider wraps existing code); any
downstream engaged-state neural analysis (it merely *consumes* the output).

**Deferred (future extension, only if judged necessary):** the hand-labeled
gold-set validation arm (§8.3).

---

## 3. Data inputs

| Input | Path (canonical) | Notes |
|-------|------------------|-------|
| Liberal registry | UM `unit_index.csv` → key on `batch_uid_liberal` | Already written by `run_unitmatch_all.py:143` (`uid_lists[0]`). No UM re-run. |
| Match-prob matrix | UM `output/.../output_prob_matrix.npy` + its `unit_index.csv` | For drift estimation + optional entry context. |
| Raw waveforms | `data/unit_match/input/BG_046/{session}/…` | UM RawWaveforms (whole-recording STA). **Currently must be restored from X:** `…/BG_046/unit_match/input/{session}/…` (6,763 `.npy`); copy into a local `input/BG_046/` level. |
| Session pkls | `data/pkls/BG_046/BG_046_{session}.pkl` | Restored (46 sessions). For spikes/ISI/PSTHs. |
| Staging manifest | `data/BG_046_staging_manifest.csv` | Stage + chronology, via `load_filtered_manifest` (tracking-QC filter: `min_trials=150`, `min_dprime=None`). |
| State tables | `data/cache/states/BG_046/{session}_states.csv` | §4. Produced by a state provider; consumed by curation. |

---

## 4. Pluggable state interface

The HMM is **not** assumed final — it may be replaced by a hand/ethogram-based
labeler. The swap point is a **file contract**, so curation never imports any
state model.

### 4.1 Canonical state vocabulary
Exactly three labels: `disengaged` (zoned-out), `impulsive`, `in_zone`
(in-the-zone / engaged). The functional fingerprint uses `in_zone` only.

### 4.2 State table contract
Per session CSV `data/cache/states/BG_046/{session}_states.csv`:

| column | type | meaning |
|--------|------|---------|
| `trial_idx` | int | **index into `session.trials`** (the raw trial list) — the same space `build_population_tensor` / `extract_unit_psths` use for `trial_indices`. NOT the HMM's valid-trial ordering. |
| `state_label` | str | one of the canonical three |
| `confidence` | float in [0,1] | provider-defined; HMM uses γ_max |

**Index-space contract (critical):** `decode_session` operates on the
*valid-trial* subset (`prepare_session_data` drops excluded trials), so the
`HMMStateProvider` MUST map each decoded row back to its raw `session.trials`
index before writing the CSV. Curation's `in_zone_trial_indices(session)` then
intersects directly with each PSTH condition's `_trial_indices_for_sizes`
output (also raw-trial space) — no re-mapping at consume time. Any state
provider must honor this raw-index contract.

### 4.3 Providers
- **`HMMStateProvider`** — wraps `hmm.decode_session()` +
  `auto_label_states_explicit()`; maps `Stimulus_sensitive → in_zone`,
  `Impulsive → impulsive`, `Disengaged → disengaged`; `confidence = p_state_max`.
  Writes the CSV. (Gating via `assign_states_with_confidence` optional.)
- **`EthogramStateProvider`** — **stub for later**; a hand/ethogram labeler writes
  the *same* CSV. No curation code changes when swapped.

A thin `load_state_table(session)` reader returns
`{trial_idx → (state_label, confidence)}`; `in_zone_trial_indices(session)`
returns the engaged-trial index set used to mask PSTHs.

---

## 5. Per-(session, uid) feature extraction

For each liberal-uid in each session it appears in, extract once and cache
(mirroring the existing `tracking_qc` intermediates cache). Reuses
`extract_session_records` primitives.

### 5.1 Biophysical features (learning/state-invariant — the link backbone)
- Peak-channel waveform + footprint (`waveform_peak`, `footprint`) from UM raw
  waveforms (whole-recording STA — consistent with the "whole-trial incl. ITI"
  decision; no change needed).
- Drift-corrected peak depth (`depth_std_um_corrected` machinery / per-session
  `estimate_session_drift` offsets).
- Baseline firing rate.
- ISI log-histogram — computed on a **curation spike partition** (§8.1) so the
  held-out partition stays independent for validation.

### 5.2 Functional fingerprint (in-zone event PSTH set)
For each condition in `PSTH_CONDITIONS` (`baseline_on`, `change_on_big_hit`,
`change_on_big_miss`, `change_on_sm_hit`, `change_on_sm_miss`, `hit_lick`):
- `trial_indices = condition_trials ∩ in_zone_trial_indices(session)`
- PSTH via the existing `extract_unit_psths(..., trial_indices=…)` path (it
  already accepts `trial_indices`), smoothed (`DEFAULT_SIGMA_MS`).
- Fingerprint similarity across a link = **Pearson r of response shapes**
  (magnitude-invariant), per condition, then aggregated (median across the
  evaluable conditions). Reuses `baseline_psth_corr`-style shape correlation.

---

## 6. The backward sweep (approach ii)

Per liberal-uid, independently:

```
sessions ← chronological list of the uid's sessions
anchor   ← most-recent session            # best-learned instance available
track    ← [anchor]; n_bridge ← 0
for candidate in reverse-chronological order (excluding anchor):
    v ← score_link(anchor, candidate)     # §6.1
    if v.decision == KEEP:
        track.append(candidate); anchor ← candidate; n_bridge ← 0   # roll anchor
    elif v.decision == SKIP:
        mark candidate skipped; n_bridge += 1
        if n_bridge > MAX_BRIDGE_GAP: break  # STOP: skips exhausted
    elif v.decision == STOP:
        break
record kept / skipped / dropped for this uid
```

- **Rolling anchor**: always compare to the *nearest kept* instance (waveform/
  depth drift slowly → nearest is the fairest reference).
- **Gap-bridge tolerance** (`MAX_BRIDGE_GAP`, default 1–2): a unit that vanishes
  for one or two sessions and **resurfaces** is re-acquired; the anchor is **not**
  rolled onto a skipped session.
- **Corroborator reference** (param, default = rolling anchor): the functional
  corroborator is scored against the rolling anchor. A config flag
  `corroborator_ref = "expert"` switches it to the fixed original (Expert-end)
  template for the strict "backwards-from-expert" reading.

### 6.1 Per-link rule — biophysical gate + functional corroborator
1. **Hard gate (must pass):** waveform-footprint corr ≥ `WAVE_PASS_R` **and**
   drift-corrected depth jump ≤ `DEPTH_PASS_UM`.
   - **Hard contradiction** (waveform anti-correlated **and** large depth jump)
     → `STOP`.
2. **ISI-shape badge** (curation partition): pass / warn / fail
   (`badge_isi_hist_corr` thresholds).
3. **Functional corroborator (availability-gated):** evaluated only if the
   candidate has ≥ `MIN_INZONE_TRIALS` in-zone trials **and** PSTH std >
   `FUNC_RESP_MIN_PSTH_STD`; else `not-evaluable` (no effect — biophysics
   decides). When evaluable: agree → promote toward *trusted*; conflict →
   demote to *review-flag*. **Never forces STOP.**

| Condition | `decision` | tier effect |
|-----------|-----------|-------------|
| gate pass, ISI pass, func corroborates or not-evaluable | `KEEP` | trusted-eligible |
| gate pass, but ISI warn OR func conflict | `KEEP` | review-flag |
| gate fail (soft) | `SKIP` | — |
| hard contradiction, or `n_bridge > MAX_BRIDGE_GAP` | `STOP` | ends extension |

### 6.2 Confidence tier per curated track
Aggregate over kept links + per-track metrics (analogous to
`composite_verdict` + `apply_isi_autopass`):
- **trusted**: every kept link passed the gate with ISI-pass and (func
  corroborates or not-evaluable); no review-flags; `trimmed_span ≥` a minimum.
- **review**: any review-flag, ISI-warn, or bridged gap present.
- **suspect**: any hard contradiction recovered-around, span too short, or
  majority of links not biophysically clean.

---

## 7. Output artifacts

Written under `FIGURES/tracking_qc/curation/` (and/or a `cache/` path), mirroring
the existing `verdicts.csv` / `verdicts_trimmed.csv` pattern.

### 7.1 `curated_links.csv` (per evaluated link — full audit trail)
`liberal_uid, anchor_session, candidate_session, gap_sessions, wave_corr,
depth_jump_um, isi_shape_corr, func_corr, func_evaluable, n_inzone_trials,
link_decision (KEEP|SKIP|STOP), review_flag, stop_reason`

### 7.2 `curated_tracks.csv` (per curated track — the consumable)
`curated_uid, liberal_uid, anchor_session, kept_sessions (ordered ; -joined),
skipped_sessions, dropped_sessions, trimmed_span, n_bridged,
confidence_tier (trusted|review|suspect)`

Downstream engaged-state analyses load `curated_tracks.csv` and filter to
`trusted` (or `trusted+review`). Shape matches today's verdict CSVs, so the
`build_qc_sheets.py` PDF renderer can spot-check curated tracks unchanged.

---

## 8. Validation (approach c)

### 8.1 Spike-partition ISI holdout (built-in)
ISI-shape is also a curation feature (§6.1.2), so validating with ISI must avoid
circularity. **Resolution:** partition each unit's spikes into two disjoint sets
(e.g. even- vs odd-indexed spikes). The **curation** ISI feature uses one
partition; the **validation** ISI uses the other. ISI-distribution shape is
stationary within a unit, so both partitions estimate the same fingerprint while
being statistically independent.

### 8.2 Held-out-ISI AUC by tier (built-in)
Reuse `validate_long_tracks.py` machinery on the **held-out** partition: compute
matched cross-day vs non-matched within-day ISI-correlation AUC, reported **per
confidence tier**. Expectation: `trusted` approaches the paper benchmark
(~0.8+ at long lags), clearly separated from `suspect`. This is the primary,
quantitative acceptance criterion for the curation.

### 8.3 Hand-labeled gold set (deferred future extension)
*Not built now.* If the held-out-ISI validation proves insufficient, a later
extension hand-labels ~30–50 curated links as same/different from the QC PDFs and
reports curation precision/recall against that gold standard — fully independent
of every automatic feature. Tracked here so the output schema (§7.1, per-link
rows) already supports attaching a `gold_label` column without rework.

---

## 9. Module layout

| Path | Role |
|------|------|
| `src/visdetect/analysis/state_provider.py` | Canonical vocabulary, state-table contract, `HMMStateProvider`, `EthogramStateProvider` (stub), `load_state_table`, `in_zone_trial_indices`. |
| `src/visdetect/analysis/track_curation.py` | Per-link `score_link`, the backward `sweep`, tier logic, feature aggregation. Pure functions; reuses `tracking_qc.py` primitives + `extract_unit_psths`. |
| `scripts/pipelines/tracking/curate_tracks.py` | CLI runner: load liberal registry → state tables → feature cache → sweep → write `curated_links.csv` + `curated_tracks.csv`. |
| `scripts/pipelines/tracking/validate_curation.py` (or extend `validate_long_tracks.py`) | §8.1–8.2 held-out-ISI validation by tier. |

Reused unchanged: `tracking_qc.py` metric/badge functions, drift estimation,
`extract_unit_psths`, `build_qc_sheets.py` renderer.

---

## 10. Parameters (single source: new constants in `constants.py` or module-level)

| Param | Default | Meaning |
|-------|---------|---------|
| `MAX_BRIDGE_GAP` | 2 | consecutive SKIPs tolerated before STOP |
| `MIN_INZONE_TRIALS` | 20 | min in-zone trials to evaluate the corroborator |
| `WAVE_PASS_R` | (reuse `tracking_qc`) | hard-gate waveform corr threshold |
| `DEPTH_PASS_UM` | (reuse `tracking_qc`) | hard-gate depth-jump threshold |
| `FUNC_RESP_MIN_PSTH_STD` | (reuse `tracking_qc`) | modulation floor for corroborator |
| `corroborator_ref` | `"rolling"` | `"rolling"` or `"expert"` reference template |
| `MIN_TRUSTED_SPAN` | 3 | min kept sessions for `trusted` |

---

## 11. Testing strategy

Synthetic-session unit tests (`make_synthetic_session`):
1. **Clean chain** — high all-feature similarity across N sessions → one
   `trusted` track, span N.
2. **Mid-chain swap** — injected waveform/depth jump at session k →
   `STOP` (or review-flag) at k; track truncates cleanly.
3. **One-session dropout** — unit absent/garbled for one session, clean before &
   after → that session `SKIP`-bridged, chain continues (resurface caught).
4. **Availability gate** — a session with zero in-zone trials → corroborator
   `not-evaluable`; link decided on biophysics alone.
5. **Holdout independence** — even/odd ISI partitions on a synthetic stationary
   spike train correlate ≈ equally; curation vs validation partitions disjoint.

---

## 12. Open / future

- `EthogramStateProvider` implementation (when the new state definition exists).
- Gold-set validation arm (§8.3).
- Whether `trusted+review` or `trusted`-only is the right downstream default
  (decide empirically from §8.2 per-tier AUC).
