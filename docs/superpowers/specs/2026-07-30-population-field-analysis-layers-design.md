# Tracking-free population field — Plan 2 analysis layers — design spec

- **Date:** 2026-07-30
- **Status:** DESIGN (awaiting user review → implementation plan)
- **Topic slug:** `population_field`
- **Parent spec:** `docs/superpowers/specs/2026-07-07-tracking-free-population-field-design.md` (this spec expands Components 4–6)
- **Parent plan:** `docs/superpowers/plans/2026-07-08-population-field-instrument-plan.md` (Plan 1 + 1.5, MERGED to `main`)
- **Owner question:** *How does the striatal population code for the change-detection task reshape across learning in BG_046 — measured on fixed anatomical bins, without tracking a single neuron, and without mistaking a **motor** reshaping for a **sensory** one?*

---

## 1. What Plan 1 already delivered (the substrate)

Plan 1 + 1.5 are merged and validated. Plan 2 consumes them and adds nothing to the instrument itself.

| Artifact | What it gives Plan 2 | Verified state (BG_046) |
|---|---|---|
| `data/cache/population_field/BG_046/registration.csv` | Per-session rigid depth shift on the common axis | 42 sessions, **all `shift_um` = 0.0**, `corr` 0.511–1.0, `n_units` 37–316. Tracked in git (force-added). |
| `data/cache/population_field/BG_046/audit.json` | The registration gate to check before analysing | `max_abs_shift_um` 0.0, `min_fingerprint_corr` 0.511, `peak_vs_centroid_median_um` 5.03 (max 281.6), signature `ea45e15ac6c9d028`, `n_sessions` 42 |
| `population_field.build_field_tensor(...)` | `(trials × time × shank-depth bin)` of **summed member-unit Hz** | Default `Change_ON`, window (−1.0, 1.5), `bin_size=DEFAULT_BIN_SIZE`; off-grid units passed as **−1**, never NaN |
| `depth_bin_edges` / `unit_field_index` / `registered_depth` / `robust_unit_depth` | The fixed grid + per-unit assignment | `DEPTH_BIN_UM = 60.0`; shanks via `assign_shanks(gap_um=120.0)` |

**Consequence of `shift_um` ≡ 0:** for BG_046 registered depth = raw depth. Registration is still applied through the same code path (so the pipeline is correct for future subjects) but it is a no-op here, and **no Plan-2 result may be attributed to registration**.

---

## 2. Scope (locked with the user, 2026-07-30)

| Decision | Choice | Rationale |
|---|---|---|
| **Subjects** | **BG_046 only** | Only mouse with a strong learning arc in the target region (DMS): 14 Learning + 17 Expert QC-pass sessions. BG_039 (DMS) and BG_038 (cortex) are Expert-dominant (1 Learning session each) → no arc; BG_031 has an arc but is **VMS** (different region, never pooled with DMS). Multi-subject generalization = separate fast-follow spec. |
| **Learning axis** | **Learning → Expert primary** (canonical `load_staging_manifest(qc_only=True)`); the 3 Naive sessions only as a **flagged exploratory arm** | Consistent with every other project analysis; the d′ ≥ 0.8 gate ensures the animal is performing, so "task encoding" is well-defined. |
| **Headline** | **Per-bin encoding × learning map** (inferential backbone) **+ 3 map descriptors** (interpretation) | The map says *where*; the descriptors say *strengthen / sharpen / relocate*. |
| **Substrate for change & choice** | **Summed field-tensor bin** (population/MUA-analog), with a per-unit `\|AUROC−0.5\|` sensitivity cross-check | True to "the field, not the particles"; transfers directly to the Plan-3 per-channel MUA. Per-unit aggregation would reintroduce the yield-drift confound and would not transfer. |
| **TF variable** | Cached **TF-GLM registry only** (`resp_log2`, `c1_r_log2`) — never the old single-pulse `tf_pulse` screening | The GLM is the only validated TF method in this project. |
| **Layers** | All three in this plan, **Layer 1 gated**: Layers 2–3 proceed only after Layer 1 validates | |

---

## 3. The motor-confound problem — and the contrast ladder *(the core design decision)*

### 3.1 Why the obvious contrast is not a sensory contrast

The natural change-encoding readout is `AUROC(hit, miss)` on the `Change_ON` response. **It is not a sensory contrast — it is a "did the animal lick" contrast.** Every hit contains a lick; no miss does. Any lick-related signal therefore separates the classes perfectly, and a "change-encoding map" could be a lick-motor map in disguise.

This cannot be fixed by shrinking the analysis window, because **the safe boundary is not yet known**:
- lick times today are **spout-contact** (`Piezo_1/2`, `Lick_L/R`) minus a **fixed 200 ms** constant (`LICK_HARDWARE_DELAY_MS`), not true tongue-motion onset;
- motor *preparation* precedes execution by a further, unmeasured interval;
- the user's video-derived lick-initiation and motion-energy pipelines are **not yet finished**.

Censoring the window would mean guessing at an unmeasured number. The project has already been burned here once (N1: a raw 0.56 "neural correlate" that was lick leakage).

### 3.2 The escape: contrasts where **neither** side has a lick

`go-miss` vs `CR` (catch-miss). Both are `trialoutcome = 'Miss'` — the mouse withheld on both — and the **only** difference is whether the TF actually changed. It is a genuine stimulus-change contrast and is **immune to lick-timing uncertainty by construction**: if there is no lick, when the lick "really" began is irrelevant. It therefore does **not** block on the video pipeline.

**Verified on real BG_046 data (read-only, 4–7 sessions spanning Learning and Expert):**
- Catch trials **do** carry a valid `Change_ON` anchor (100 % finite `change_time` and `Change_ON` in both groups) — so CRs are alignable.
- `reactiontimes['RT'/'FA']` are NaN on **100 %** of Miss trials.
- Raw pooled lick channels: **zero** go-miss or CR trials had any lick in **(0, +0.5 s)** post-change, in every session checked. In the wide (−1.0, +1.5 s) window one session (27082025) had 3 stray licks (1 go-miss, 2 CR).
- Counts per session (go-miss / CR): 73/48, 53/23, 30/30, 36/29, 17/30, 41/43, 38/37 — typically ~30–40 each, adequate throughout the arc. Misses do **not** collapse with learning (manifest medians: Learning 51 → Expert 38; CR 34 → 41).

### 3.3 The ladder (ordered by motor contamination)

Every rung is computed per bin, per session, and run through the **same** learning model (§6). The comparison **between** rungs is the scientific payload.

| # | Rung | Contrast | Motor status | Role |
|---|---|---|---|---|
| **C1** | **Sensory, lick-free** | `AUROC(go-miss, CR)` on `Change_ON` | **No lick on either side** (verified) | **Headline sensory map** |
| **C2** | **Motor, clean** | FA lick-aligned pre-movement vs baseline | Lick with no stimulus change | Pure motor map |
| **C3** | **Motor-matched change** | `Hit` vs `FA`, **lick-aligned** | Lick on *both* sides → motor matched | Brackets the problem from the opposite side |
| **C4** | **Detection, motor-inclusive** | `AUROC(hit, miss)` on `Change_ON` | **Contaminated** | Descriptive only — relabelled *detection/choice incl. motor*, **never** carries the sensory claim. Cross-checked by projecting out the FA-derived motor axis and recomputing. |
| **C5** | **Independent anchor** | TF-GLM encoding per bin | *Partially* controlled (lick + wheel regressors; **no** motion-energy/pupil) | Independent readout with a different failure mode |
| **C6** | **Graded sensory** *(secondary)* | Response vs `change_size` **within misses** | No lick | Graded tuning without motor. **Underpowered per session** (see §3.5) |

### 3.4 The payoff

Run the learning model on every rung:
- **If C1, C3, C5 reshape like C4** → the conclusion is robust and earned.
- **If they diverge** → what reshapes across learning is the **motor** code, not the sensory one. That is itself a real finding, entirely consistent with this project's track record (pre-FA activity was motor/gain not sensory; N1 was lick leakage).

Either way the answer is honest. **This converts the confound from an unaddressed threat into a measured axis** — the only safe way to proceed while the video pipeline is unfinished.

### 3.5 Honest limits of the ladder

1. **Lick-free ≠ movement-free.** Whisking and postural micro-movements could still differ between go-miss and CR. Only motion energy settles this → §9 provider seam.
2. **Miss trials are, by definition, non-detected.** C1 measures sensory encoding **in the non-detected regime** — a lower bound, entangled with engagement. Mitigated by C6 and (later) by conditioning on the existing behavioural state labels.
3. **C6 is underpowered per session.** Misses concentrate at small ratios: per-session `change_size` counts among go-misses are e.g. {1.25:24, 1.35:19, 1.5:20, 2.0:9, 4.0:1}. The 4.0 bin is 1–3 trials. → C6 runs **pooled across sessions**, or restricted to the three small ratios; reported as secondary/underpowered, never per-session across all five ratios.
4. **C5's TF registry is not movement-controlled** (BG mice have no processed video; the GLM used lick+wheel only). It is *partially* controlled, not pristine.
5. **Trial-count matching.** Hit/miss and go-miss/CR counts are unequal; every AUROC contrast subsamples the larger group to match (fixed seed), and both matched and unmatched values are reported.

### 3.6 Control: are early licks partly TF-pulse driven? *(user-raised)*

C2/C3 assume the FA lick is a *motor* event with no stimulus trigger. If a fraction of early licks are actually triggered by large baseline TF fluctuations ("fast"/"slow" pulses), then the FA map carries a sensory component — and the motor axis projected out in C4 would remove real sensory signal.

**Bounded control in this plan (needs no video):**
1. Extract fast/slow pulse times per session from `trial.baseline_values` (see §10 for the exact, non-obvious time base).
2. For each FA lick, look back over the established impulsivity-kernel window `[lick − 1.5 s, lick − 0.15 s]` (reusing `psychophysical_kernel.fa_kernel_epochs`, late FAs ≥ `FA_RT_SPLIT` = 3.0 s, change-guarded) and label it **pulse-preceded** vs **spontaneous**.
3. Recompute the C2 motor map on each subset. **If the map is stable across the split, the motor interpretation holds**; if it differs, report it and treat the motor axis as partly sensory.

**Out of scope here** (→ its own spec/chat): quantifying *what fraction* of FAs are TF-driven and *how large* the contribution is. That is a behavioural question in its own right. Note the repo has **no** existing code aligning FA times to discrete pulse *events* — the continuous reverse-correlation kernel (B10) is the closest, and this control is net-new (small) code.

⚠️ **Pulse-threshold convention must be chosen, not assumed** — two incompatible definitions coexist in the repo:
- `tf_glm.pulse_times_from_tf` → **±0.5 × SD** of log2(TF) (`TFGLMConfig.sd_pulse=0.5`) — the paper/GLM criterion, and the one behind the TF registry;
- `tf_pulse._collect_pulses` → **fixed ±0.25** in log2 units (`TF_FAST/SLOW_THRESH_LOG2`) — the deprecated screening path.

**Recommendation: ±0.5 × SD** (consistent with the validated GLM that produced `resp_log2`). **Flagged for user confirmation — not silently chosen.**

---

## 4. Layer 0 — field-tensor cache *(deferred from Plan 1; belongs here)*

`scripts/population_field/cache_tensors.py --subject BG_046`

1. Read `audit.json` → **gate**: refuse to run if the registration audit is not within tolerance.
2. Read `registration.csv`; join on `canonical_session_id(session)`.
3. Per session: `good_and_stable_ids` → `robust_unit_depth` → `registered_depth(raw, shift_um)` → `unit_field_index(depth, shank, y_edges)`; off-grid → **−1**.
4. Build tensors via `build_field_tensor` for the events/outcome subsets the ladder needs.
5. Cache one `.npz` per `(canonical_session_id, event, contrast)` under `data/cache/population_field/BG_046/tensors/`.

**Each npz stores:** the `(trials × time × bin)` Hz tensor, `bin_centers`, `valid_trials`, `unit_bin_index`, `n_bins_anat`, **per-bin unit count (`bin_yield`)** — the yield covariate, which is *per bin*, not the per-session `n_units` already in `registration.csv` — and a **provenance block** (§9.3).

**Binning (never cross-applied):**
- Slow evoked layers (C1–C4, Layer 3): `DEFAULT_BIN_SIZE` = 25 ms, `DEFAULT_SIGMA_MS` = 25 ms.
- TF (C5): the registry is already at the GLM's native 50 ms grid — **consumed as-is, never re-binned**.

---

## 5. Layer 1 — functional map *(PRIMARY)*

### 5.1 Per-bin encoding values

Per shank×depth bin, per session, per ladder rung. All windows come from `EVENT_RESPONSIVENESS_WINDOWS` / `EVENT_VALID_OUTCOMES` — none invented:

- **C1 / C4 (`Change_ON`)**: response `(0, 0.25) s`, baseline `(−0.4, −0.05) s`. *(The wider `(0, 0.5)` was also verified lick-free and serves as a robustness window.)*
- **C2 / C3 (lick-aligned)**: pre-movement `(−0.3, −0.15) s`, baseline `(−1.75, −1.25) s`.
- **C5 (TF)**: **mean `c1_r_log2` (clipped ≥ 0) of the bin's member units** = TF-strength map (primary); **fraction `resp_log2`** = TF-density map (secondary). Joined **per session only** (see §10 gotcha on `region_bank_confirmed`).

**Why AUROC is the right metric:** it is rank-based, hence **inherently FR-normalized** and immune to the un-normalized-Hz artifact that produced the retracted `tf_transient_sustained_state` result. Cross-bin comparability comes for free. `c1_r_log2` is likewise already normalized.

**Signed vs unsigned:** store the signed AUROC; the map statistic is **strength = |AUROC − 0.5|**. Sign is reported separately (it is a direction, not a magnitude, and must never be derived from the data it is then averaged over — cf. the circularity hard rule).

### 5.2 Map descriptors (the interpretation layer)

Per variable, per session, computed on the across-bin encoding profile:

| Descriptor | Plain meaning | Formula |
|---|---|---|
| **Total strength** | how much encoding there is overall | Σ over bins of strength |
| **Depth centroid (µm)** | where along the probe the encoding sits | encoding-weighted mean of bin depth |
| **Depth spread (µm)** | over how many microns of tissue the signal is smeared — **the sharpening measure** | encoding-weighted SD of bin depth |
| **Effective bin count** | how many bins genuinely carry the signal (1 = one hotspot; 48 = spread evenly) | participation ratio `(Σx)² / Σx²` |

**Why both spread and effective-bin-count:** the µm-spread quietly assumes a *single* hotspot. With **two** hotspots at different depths the weighted mean lands in the gap between them — a depth where nothing happens — and the spread inflates for the wrong reason. Effective-bin-count is arrangement-agnostic. The pipeline therefore reports a **hotspot count** (peaks in the smoothed profile) alongside, and leans on the appropriate measure. *(Gini was considered and rejected: it measures inequality across bins, not physical extent in tissue.)*

⚠️ `compute_effective_dim` (participation ratio) exists **only in the archived suite** (`archive/analysis_suite_2026-07-01/03_population/c_dimensionality_reduction.py:50`) — it must be **ported into the live package**, not imported from the archive.

---

## 6. Statistics — the learning claim

**Replication unit = the session** (one encoding value per bin per session). Never the trial and never the unit — that is where pseudoreplication would enter.

**Primary (per bin):** partial Spearman of encoding vs **chronological session index**, controlling for that bin's yield (`bin_yield`).
- `chronological_sort` / `session_date_key` for ordering — **never** raw `sorted()`.
- ⚠️ **pingouin is NOT installed**, and scipy has no partial correlation. Implement as: residualize both variables on the covariate with `statsmodels.formula.api.ols`, then `scipy.stats.spearmanr` on the residuals. Small reusable helper, not inlined.

**Null (per bin):** ⚠️ `utils.permutation_test` tests a **difference of means** and **cannot** null an AUROC or a slope. A bespoke shuffle is required that recomputes the **actual statistic** on each permutation:
- learning slope → permute session order, recompute the partial Spearman;
- AUROC → permute trial condition labels, recompute `compute_auroc` per bin.

**Multiple comparisons:** `fdr_correct` (BH) across bins, **within a rung** — never pooled across separate scientific questions. ⚠️ `fdr_correct` returns a **boolean mask, not q-values**; use `statsmodels.stats.multitest.multipletests` where reportable q-values are needed.

**Robustness:** Learning-vs-Expert two-group contrast (yield-controlled permutation) alongside the continuous slope; maps reported **with and without** yield control; odd/even-trial split reproducibility; trial-count matching (§3.5).

**Pseudoreplication hardening** (for the `harden-result` battery before any claim leaves the repo): `statsmodels.formula.api.mixedlm` with bins nested in session. Verified available (statsmodels 0.14.6).

---

## 7. Layer 2 — population geometry *(secondary; gated on Layer 1)*

On the field tensor with **anatomical bins as features**:

⚠️ **The field tensor is raw summed Hz.** Before any CD/subspace fit it must be z-scored per anatomical bin via `compute_zscore_normalized(field, bin_centers, baseline_window)` (shared baseline, treating the bin axis as the "unit" axis) — the convention every upstream consumer assumes.

- **Sensory CD** — `compute_lda_cd` on the **lick-free C1** contrast (go-miss vs CR), *not* hit-vs-miss.
- **Motor CD** — `fit_lick_motor_cd(z_lick, bin_centers)`. ⚠️ Requires a **lick-aligned** tensor (`build_field_tensor(event_name='FA'|'Hit')` with the `EVENT_VALID_OUTCOMES` filter), not the Change_ON default.
- **Angle** between sensory and motor CDs, tracked across learning (same model as §6).
- **Dimensionality** — participation ratio of the PCA spectrum of the z-scored, window-collapsed field matrix.
- **Motor-orthogonalized sensory encoding** — `motor_subspace` → `project_out_subspace`, then recompute; `motor_axis_signal` gives the per-trial motor magnitude used as a matching covariate.
- **CCA / Procrustes cross-check** — align sessions on baseline-period covariance and confirm the fixed-grid result survives. ⚠️ **No CCA or Procrustes code exists in the repo**; build on verified-available `sklearn.cross_decomposition.CCA` and `scipy.linalg.orthogonal_procrustes`.

⚠️ **Window-convention mismatch to resolve explicitly:** `neural_latents` window collapsers use **half-open** `[lo, hi)`; `compute_zscore_normalized`'s baseline mask is **closed** `[lo, hi]`. Pick one in `field_geometry.py`, document it, and do not mix.

---

## 8. Layer 3 — evoked profiles *(tertiary, descriptive)*

Per-bin event-aligned PSTHs (`Change_ON`, `FA`, `Hit`, `Baseline_ON`), shared-baseline z-scored, 25 ms/25 ms → **depth × time heatmaps per stage**. The intuitive "what you saw in the viewer" layer. Purely descriptive; no inferential claim.

---

## 9. Forward compatibility — wiring in corrected data later *(user requirement)*

**Goal: when time-correction and motion-energy land, re-running is a config flip, not a redesign.**

### 9.1 The provider seam

A single abstraction, modelled on the repo's existing `state_provider.py` pattern, keyed by `canonical_session_id`:

```python
class LickMovementProvider(Protocol):
    def available(self, session_id: str) -> Capability:      # flags: lick / movement
    def lick_initiation_times(self, session_id: str) -> Optional[np.ndarray]
        # length n_trials, NaN where none, on the NEURAL/NI-DAQ clock
    def movement_covariate(self, session_id: str, bin_centers: np.ndarray) -> Optional[np.ndarray]
        # (n_trials, n_time_bins) resampled onto the field-tensor grid, or None
```

- **Today's impl** — lick times from the existing path (`compute_true_reaction_time`, i.e. spout contact − fixed 200 ms; or first pooled-channel contact per trial); `movement_covariate` returns **None**, giving every confound-control branch a defined no-op.
- **Future impl** — reads a **local** movement/lick-onset cache, maps camera → neural clock via `load_video_sync` + `camera_to_nidaq`, resamples onto the field grid with the existing `_resample_to_bins`. **Same return contract ⇒ zero analysis-code change.**

All plumbing already exists: clock models are cached for **~24 BG_046 sessions** under `data/cache/video_sync/`, and `config` already defines `MOTION_ENERGY_DIR` and `PUPIL_DIR` as (currently empty) local cache dirs. ⚠️ `CAMERA_ROOT` points at **X: (Samba)** — the provider reads **only local caches**, never X:.

### 9.2 Where the seam is used
- Per-trial **lick screen** for C1 (drop any go-miss/CR trial with a lick in the analysis window).
- **Event times** for C2/C3 (`lick_time_source` switches spout-contact → video-initiation).
- **Movement covariate** as an additional control regressor / matching variable in every rung.

### 9.3 Provenance (so old and new runs can never be silently mixed)

Every cached tensor and every results CSV carries: `lick_time_source` (`spout_contact_minus200ms` | `video_initiation`), `movement_controlled` (bool), `tf_movement_controlled` (bool, currently **False**), `pulse_threshold_convention`, `code_commit`. A **re-run checklist** ships with the plan: flip the provider, re-run `cache_tensors` → the three layers, and diff old vs new maps as an explicit deliverable.

---

## 10. Verified grounding + gotchas ledger

Everything below was verified against source or live data on 2026-07-30 (5-agent read-only sweep + independent spot-checks). These are the traps this plan must not fall into.

| # | Gotcha | Consequence |
|---|---|---|
| 1 | **Outcome labels are CAPITALIZED** in pkls (`'Miss'`, `'Hit'`, `'FA'`, `'abort'`, `'Ref'`) | `build_population_tensor` and `get_event_times_by_trial` lowercase internally, but **`get_event_times`' behavioural branch is case-sensitive** — passing `'fa'`/`'hit'` silently returns an empty list. Always pass `'FA'`/`'Hit'`; compare via `.lower()` everywhere else. |
| 2 | **FA counts are LARGE, not small** | Manifest `fas` (~8–14) = **SDT** false alarms (catch-trial hits). The FA **early-lick** event = manifest `early_licks` = **~100–310/session** (verified: 194, 205; median 164). C2/C3 are well-powered. *(An earlier draft of this design wrongly assumed ~8/session.)* |
| 3 | **Lick acquisition config ALTERNATES between two mutually-exclusive channel pairs** | BG_046 flips config **six times** (Lick 23–30 Jun · Piezo 01–22 Jul · Lick 24–28 Jul · Piezo 04–20 Aug · Lick 26–29 Aug · Piezo 01–17 Sep) = **33 Piezo / 13 Lick** sessions. It is **not** an early→late switch. No session populates both pairs. **Correct rule: `Piezo_1` when present, else `Lick_L` — never `Piezo_2` or `Lick_R`** (verified: on Lick-config sessions `Lick_L`'s first post-change event equals `change+RT` to the millisecond, i.e. it is the channel software derives RT from; `Piezo_2` is a sparse ~11 ms-shifted subset of `Piezo_1`; `Lick_R` is a lower-fidelity second detector on the same single spout — `Valve_R` is always 0). ⚠️ **Do NOT use `tf_glm_data._collect_lick_times`** — it pools all four and over-counts (see §10a). |
| 3b | **`Lick_L` is raw threshold crossings, not lick-bout onsets** | `Lick_L` counts are 5 000–36 000/session vs `Piezo_1` 200–2 300 — **10–100× denser**. Any per-era lick rate/regressor must **de-bounce** (merge crossings within a refractory window) before Piezo- and Lick-config sessions are comparable. |
| 3c | **`Piezo_1` is a genuine but INSENSITIVE lick sensor** | Verified lick-locked against a circular-shift null (FA `z=12.7`, Hit `z=20.4`), so it is **not** a reward channel — but it detects only **~20–45 %** of licks and lags ~300–640 ms. `Valve_L` is the reward channel (**exactly 1 event per hit**); `Piezo_2` is **not** lick-locked (`z=0.9, p=0.25`) and is unusable. |
| 3d | **NI lick trains are NOT comparable across configs** | Detection sensitivity ~20–45 % (Piezo) vs ~100 % (`Lick_L`), and the configs **interleave across the learning arc** → any cross-session lick-rate/hazard measure built on NI channels is confounded by acquisition config. **The per-trial lick screen in this plan uses a binary did-lick test only** (its verified result — 0 licks on miss trials — holds under the worst-case sensitivity, since a *less* sensitive channel cannot manufacture licks). For lick **timing** use `compute_true_reaction_time`, accurate in both configs. |
| 4 | **`build_population_tensor` has no `change_size` awareness** | `outcome_filter` (case-insensitive set) and `trial_indices` are **AND-combined**, so `{'miss'} + <catch indices>` yields exactly the CR group — **no new API needed** — but the caller must compute the catch index list itself (`abs(change_size − 1.0) ≤ tol`). |
| 5 | **`get_event_times` returns a flat `List[float]`**, NaNs dropped, **not** index-aligned to trials | For per-trial alignment use `get_event_times_by_trial` (length n_trials, NaNs retained). |
| 6 | **`permutation_test` = difference of means only** | Cannot null AUROC or a learning slope. Bespoke shuffle required (§6). |
| 7 | **`fdr_correct` returns a boolean mask**, not q-values | Use `statsmodels.stats.multitest.multipletests` if q-values are reported. |
| 8 | **pingouin is NOT installed** | Partial correlation via OLS residualization + spearmanr. |
| 9 | **`compute_effective_dim` lives only in the ARCHIVE** | Port into the live package; never import from `archive/`. |
| 10 | **No CCA / Procrustes anywhere in the repo** | Build on `sklearn.cross_decomposition.CCA` / `scipy.linalg.orthogonal_procrustes` (both verified importable). |
| 11 | **`STAGE_ORDER` == `['Learning','Expert']`** after the default filter (`merge_naive_learning=True`) | Any code expecting a `'Naive'` key **KeyErrors**. The exploratory naive arm must use the unfiltered manifest path explicitly. |
| 12 | **`trial.baseline_values` has an IMPLICIT time base** | Fixed length 1800, logged ~3× per 50 ms → decimate `bv[::3]`, `dt = 0.05 s`, anchored at `ni_events['Baseline_ON'][i]`. It **overshoots** the real baseline (~30 s) → must be bounded by `n_seen` / `Change_ON` / outcome-lick guards or it leaks. Reuse `psychophysical_kernel.baseline_log2tf`. |
| 13 | **`constants.TF_SAMPLE_PERIOD = 0.25` is STALE/WRONG** for baseline TF | Every live extractor uses 0.05. Never use it here. |
| 14 | **TF registry `region_bank_confirmed = False`** for all rows | Per-session/per-unit joins **only**; never pool TF-responsive unit ids across sessions to label a fixed depth bin (chronic drift). `unit` = within-session Kilosort id, not a tracked identity — consistent with the tracking-free premise. |
| 15 | **The field tensor is raw summed Hz** | Z-score per anatomical bin before geometry (§7). |
| 16 | **Session-id leading-zero-day footgun** | Every key/join/sort through `canonical_session_id` / `session_date_key`; never write `session_name` to CSV as an int. |
| 17 | **`data/cache/` and `FIGURES/` are gitignored** | Deliverable CSVs/figures must be `git add -f`'d (project "preserve artifacts on merge" convention). `save_figure(fig, name, 'population_field/BG_046')` lands under `FIGURES/population_field/BG_046/`. |
| 18 | **Worktree hazards** | Set `PYTHONPATH=<worktree>/src` or you silently test main's code. `data/`/`FIGURES/` may be **junctions** — never `git worktree remove` / `rm -rf data` with one present (Jun-7 data-loss). The primary checkout is being used by parallel chats and **switched branches mid-session** on 2026-07-30 — do all work in this worktree. |
| 19 | **Never compute over X:** | Fully local plan (pkls + local caches). MUA/`ap.bin` is Plan 3 (HPC/Slurm). |

### 10a. Lick-channel defect found while writing this spec *(2026-07-30/31 audit)*

The per-trial lick screen in §3.2/§9.2 must not use the existing helper. A 4-agent read-only audit (all 46 BG_046 pkls + every code site) found **two real defects**, and this spec's own earlier draft repeated one of them:

- **Over-pooling (root defect).** `tf_glm_data._collect_lick_times` takes the **union of all four channels** with no de-duplication (`_LICK_CHANNELS = ("Piezo_1","Lick_L","Lick_R","Piezo_2")`, `tf_glm_data.py:594-610`). On Lick-config sessions it adds `Lick_R` (coincident with `Lick_L` within 2–3 ms on 85–87 % of events); on Piezo-config it adds `Piezo_2` (80 % within ~11 ms of `Piezo_1`) → **most licks counted twice, milliseconds apart**. It feeds `session_trial_regressors` (`:514`), i.e. the TF-GLM lick regressor behind **`data/cache/tf_responsive/bg046_tf_responsive.csv`** (7 047 unit-rows) and ~20 downstream scripts. Its own docstring (`:491`) still claims `Piezo_1`-only — stale.
- **Single-channel hardcodes (fail silently to zero).** `plot_tf_exemplars.py:136` reads `Piezo_1` only → 0 licks on the 13 Lick-config sessions. `plot_fa_lick_peth.py:108`, `plot_fa_neural_stratified.py:134` read `Lick_L` only → 0 licks on the **33 Piezo-config sessions (the majority)**. `hmm_neural_TF_event_comparison.py:167` uses `Lick_L` only as a baseline-lick contamination filter → the filter is **silently disabled** on 33/46 sessions (fails open).

**Immune** (verified): everything deriving lick times from software reaction times via `compute_true_reaction_time` / `get_event_times` — the early-lick learning, FA-hazard, B10 impulsivity-kernel, `lick.py` responsiveness, state-labeler, `neural_latents` CD and decision-latents results. The NI-channel defect cannot touch them.

**Consequence for this plan:** the C1 per-trial lick screen uses a **new correct resolver** — `Piezo_1` if present else `Lick_L`, de-bounced, never `Piezo_2`/`Lick_R` — exposed through the §9 provider so it upgrades to video lick-initiation later. Remediation of the *existing* sites (and any TF-registry re-run) is **out of scope here** and tracked separately (§14).

⚠️ **Do not generalize the channel rule across subjects.** `Piezo_*` exists only in BG_046 (33 sessions) and BG_031 (8); BG_038/039/012/040/041/049 are 100 % `Lick_*`. And **`Lick_L` is not always the clean channel**: in BG_031 `Lick_L` is contaminated (751 793 events, ~63 Hz) and `Lick_R` is real; in BG_012 the reverse (`Lick_R` 1 195 778 events, ~83 Hz). A robust resolver must **reject any channel with a physiologically impossible sustained rate**, not trust a fixed name.

---

## 11. Deliverables & file layout

**Library** (new, TDD, focused modules — `population_field.py` stays the instrument):
- `src/visdetect/analysis/field_encoding.py` — per-bin encoding (the ladder), map descriptors, learning regression + bespoke nulls.
- `src/visdetect/analysis/field_geometry.py` — CDs, angle, dimensionality (ported participation ratio), CCA/Procrustes cross-check.
- `src/visdetect/analysis/lick_movement_provider.py` — the §9 provider seam + today's default impl.

**Scripts:** `scripts/population_field/{cache_tensors,build_map,build_geometry,build_profiles}.py`
**Caches:** `data/cache/population_field/BG_046/{tensors/,map/,geometry/}`
**Figures:** `FIGURES/population_field/BG_046/`

---

## 12. Testing

- Unit tests on **synthetic sessions** (`visdetect.utils.synthetic`): encoding recovery when a bin is given a known effect; descriptor correctness on hand-built profiles (single hotspot vs two hotspots — the case that distinguishes spread from effective-bin-count); null calibration (**shuffled data → flat / uniform p**, the mandatory circularity control); provider contract (today's impl returns `None` movement and the analysis still runs).
- Golden-path: one BG_046 session end-to-end per layer.
- Every cross-neuron magnitude test FR-normalized by construction (AUROC / z-score).

## 13. Validation gates (in order)

1. **Registration gate** — read `audit.json`; refuse to proceed if out of tolerance.
2. **Known-response sanity** — the maps must put change/lick responses at plausible depths vs existing single-unit results.
3. **Null control** — shuffle → flat. Non-flat means a bug, not a finding.
4. **Yield control** — with and without; a map that only exists without it is a yield artifact.
5. **Layer-1 gate** — Layers 2–3 proceed only if Layer 1 passes 1–4.

## 14. Out of scope (explicit follow-ups)

- **What fraction of early licks are TF-pulse-driven, and how large is the contribution** — the user's question; its own spec/chat. §3.6 only checks that the C2 motor map is *stable* across that split.
- Multi-subject generalization (BG_039 DMS / BG_031 VMS / BG_038 cortex).
- MUA headline (Plan 3, HPC `ap.bin` threshold crossings) and local-vs-MUA agreement.
- Movement-controlled **re-fit** of the TF-GLM (needs the video pipeline).
- State-conditioned versions of the maps (engagement × encoding).
- **Remediating the lick-channel defects found in §10a** — fixing `_collect_lick_times` (promote to a public, era-aware, de-bouncing, contamination-rejecting resolver), fixing the four hardcoded single-channel sites, and **re-running the TF-GLM registry to size the impact** on `resp_log2` / `kernel_fwhm` / `kernel_peak_t` (especially the 13 Lick-config sessions). This plan only *avoids* the defect; it does not fix the existing code or re-derive the registry it consumes for C5.
- Confirming whether `LICK_HARDWARE_DELAY_MS = 200` should apply in **both** acquisition configs — on Lick-config sessions the stored RT already coincides with the `Lick_L` crossing at 0 ms offset, so the 200 ms subtraction may place the expected lick 200 ms too early there.

## 15. Open questions for the user

1. **TF pulse threshold convention** for §3.6 — recommend **±0.5 × SD of log2(TF)** (GLM/paper criterion, matches the registry) over the deprecated fixed ±0.25 log2. Confirm?
2. **Sensory response window** — canonical `(0, 0.25) s` as primary with `(0, 0.5) s` as robustness (both verified lick-free). Confirm?
3. **C6** (change-size scaling within misses) — pooled across sessions, or restricted to the three small ratios? Recommend **pooled, reported as secondary**.
4. Anything in the §10 ledger that contradicts your understanding of the data.
