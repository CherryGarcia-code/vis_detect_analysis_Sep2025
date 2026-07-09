# Tracking-free anatomical population field across learning — design spec

- **Date:** 2026-07-07
- **Status:** DESIGN (awaiting user review → implementation plan)
- **Topic slug:** `population_field`
- **Owner question:** *How does the striatal (and, for BG_038, cortical) population code for the change-detection task reshape across learning — measured without tracking a single neuron across sessions?*

---

## 1. Motivation

The chronic Neuropixels 2.0 recordings sit on a **single insertion per mouse** with a fixed channel map, and the **spatial activity landscape along the probe is visibly conserved across sessions and weeks** (the SpikeGLX-viewer observation that motivated this work). This is not a vibe: the repo already quantifies it — `scripts/pipelines/tracking/diagnose_intersession_drift.py` builds a per-session amplitude-depth "fingerprint" from every unit's full 383-channel footprint and cross-correlates consecutive sessions, finding **~0 whole-probe drift across BG_046's ~42 sessions (23 Jun → 17 Sep 2025), single chanmap signature, fingerprint correlation 0.5–0.98.**

Single-unit **tracking** (UnitMatch, DANT, UM∩DANT consensus) yields a small, hard-to-fully-trust cohort. Rather than fight that, this project **exploits the conserved landscape to give cross-session correspondence for free** and does the science on **fixed anatomical bins ("the field, not the particles")** — sidestepping unit tracking entirely.

Grounding for the approach (prior art): Steinmetz et al. 2021 (NP2.0 chronic activity-fingerprint stabilization for >8 weeks — the founding demonstration), IBL electrophysiology-based alignment, Kilosort datashift / DREDge / MEDiCINe drift lineage; Trautmann et al. 2019 (unsorted threshold-crossings recover the same population dynamics as sorted units); cortex-lab `psthByDepth` ("analyze the field"); Gallego et al. 2020 (CCA latent alignment across days).

## 2. Scientific question & claim

**Primary claim:** the depth/region-resolved population code for the task's three key variables — **change detection, temporal-frequency (TF) encoding, and choice/lick** — reshapes across learning (primarily **Learning → Expert**), and we can show *how* (strengthening, sharpening, or relocating along the probe) **without any single-unit tracking**.

**Headline metric priority (all three computed; this is the reporting order):**
1. **Functional map** — per-bin selectivity/decoding for change / TF / choice → an anatomical functional map of the probe, watched across learning. *(primary)*
2. **Population geometry** — coding directions / dimensionality on the fixed-grid tensor; reorganization across learning. *(secondary)*
3. **Evoked-response profiles** — per-bin event-aligned PSTHs; the descriptive "what you saw in the viewer" layer. *(tertiary)*

## 3. Scope

**Subjects:** BG_046 (primary), BG_039, BG_031, BG_038.
- Region identities: **BG_046 = DMS**, **BG_039 = DMS** (poolable with 046), **BG_031 = VMS** (separate), **BG_038 = cortex, M1/S1** (separate; here the CCF **region/laminar** axis is genuinely informative, unlike the single-region striatal mice where **depth within one region** is the discriminating axis). The pipeline does not privilege striatum.
- **Multi-signature rule (user-specified):** for a mouse with several `chanmap_signature` blocks, use the **block with the most sessions** and restrict the whole pipeline to it (BG_031 → 42-session block; BG_038 → 35-session block; BG_046 and BG_039 are single-signature). Sessions on other signatures are excluded (different geometry ⇒ not directly comparable on a fixed grid).

**Stages:** the learning-arc claim is strongest where ≥2 stages exist. BG_046 and BG_039 span Learning→Expert. **BG_031 may be Learning-dominant and BG_038's stage span is unverified — verify each subject's stage span at implementation;** single-stage subjects still contribute to instrument validation and within-stage stability. (BG_046's QC manifest merges Naive→Learning at d′≥0.8, so the primary arc is **Learning→Expert**, not *truly* naive→expert unless we deliberately use the unfiltered manifest.)

**Primary vs replication:** BG_046 is the cleanest primary deliverable (single insertion, single signature, drift≈0, full anatomy). The others are generalization/replication.

## 4. Architecture — fixed anatomical grid backbone

Fixed **shank × depth** bins on a **match-free registered common axis** give cross-session correspondence with no alignment step and no tracking: bin *k* in session 1 ≡ bin *k* in session 40. Everything downstream operates on a `(trials × time-bins × anatomical-bins)` tensor — the same shape `build_population_tensor` already returns, but with **anatomical bins replacing sorted units**. CCA/latent alignment is kept only as an **optional robustness cross-check** for the geometry layer, never as the backbone.

## 5. Components

### Component 0 — Match-free registration + depth audit gate *(runs first; nothing bins until it passes)*

**Do NOT use `peak_depth_corrected_um` from `curation_features*.pkl`.** It is `peak_depth_um − drift_offset` where `drift_offset` comes from `estimate_session_drift()` ([tracking_qc.py:293](../../../src/visdetect/analysis/tracking_qc.py)), which is **anchored on UnitMatch consecutive-pair matches (prob>0.95)** — using it would smuggle the tracker we are escaping back into a "tracking-free" pipeline (circular).

- **Registration = match-free.** Per-session rigid Z-shift from `diagnose_intersession_drift.py`'s amplitude-depth fingerprint (`session_fingerprint` + `estimate_shift`, ±300 µm search). Alignment uses the landscape **shape**, so it is robust even as overall activity magnitude drifts.
- **Robust per-unit depth** (local substrate only) = amplitude-weighted centroid of the footprint, **not** the single max-ptp peak channel. The **MUA substrate is per-channel**, so it needs no per-unit depth at all — binning is pure geometry + the rigid shift.
- **Audit gate (report before proceeding):** (a) match-free shift vs UM `drift_offset` agreement where both exist; (b) BG_046 shift ≈ 0 (⇒ corrected ≈ raw); (c) per-session fingerprint overlays actually line up; (d) per-unit peak-channel vs amplitude-centroid agreement. If registration is not trusted against the raw activity, stop.

### Component 1 — The grid (two nested resolutions, per session)

- **Fine:** `shank × depth` bins — shanks via `assign_shanks(gap_um=120)` ([anatomy/channel_geometry.py:10](../../../src/visdetect/anatomy/channel_geometry.py)); depth bins of width `DEPTH_BIN_UM` (**new constant — suggested 60 µm, to confirm**) over the active band (BG_046 ≈ 48 bins).
- **Coarse:** CCF-region rollup from `data/anatomy/<SUBJ>/unit_anatomy.csv` / `channel_atlas.csv`. One bin for single-region striatal mice; laminar/region-resolved for BG_038.

### Component 2 — Substrate & field tensor (staged)

Two substrates fill each bin; **their agreement is a core validation** that yield drift isn't manufacturing results.
- **Stage 1 — local prototype (no HPC):** summed/averaged `good_and_stable` spike_times (from pkls) of units whose registered robust depth falls in the bin. Fully local; fastest; exposed to the good-unit **yield-drift confound (89%→15%)** — hence the controls below and the MUA cross-check.
- **Stage 2 — MUA headline (one HPC job):** per-channel threshold-crossing rate, binned by channel depth. Immune to the good-unit QC gate. Requires high-passing `*.imec0.ap.bin`, which is **X:-only ⇒ a single Slurm job over CephFS. NEVER compute over the X: Samba gateway.** This is the only non-local piece.

### Component 3 — Normalization & anchor *(general activity fingerprint — NOT TF)*

The cross-session reference frame rests on the **general spatial activity fingerprint**, not on the sparse (2–5%) TF signal:
- **Registration / correspondence anchor:** the general spatial activity fingerprint (Component 0).
- **Normalization reference:** raw **pre-event baseline firing** per bin (all cells, stimulus-matched across sessions), never a post-change or outcome-defined window. Per-unit/per-bin shared-baseline z-score (`compute_zscore_normalized`) or Δrate (`compute_baseline_subtracted`).
- **Anti-circularity:** cross-session comparisons are anchored on **general, learning-invariant baseline activity**; we never align or normalize on hit/change/choice (that would build the learning effect into the reference). **Falsifiable:** we *test* the anchor's stability across stages; if the general baseline landscape itself reshapes, that's a finding and a signal to fall back to an even more stimulus-bound reference.

### Component 4 — Layer 1: functional map *(primary)*

Per `shank × depth` bin, per session, encoding strength for three variables:
- **Change detection:** AUROC hit vs miss on Change_ON-aligned response (`compute_auroc`), big-change hits vs misses, plus change-size scaling.
- **TF:** measured **only** by the **updated TF-GLM** (Khilkevich–Lohse replication — the sole validated method; outputs behind `data/cache/tf_responsive/*.csv`, `c1_r_log2`/`resp_log2`). **Local substrate:** fit per `good_and_stable` unit and aggregate each unit's TF-kernel strength into its depth bin → a **TF-encoding-density map**. **MUA substrate:** fit the same GLM to per-bin MUA rate. The old single-pulse `tf_pulse` z-score screening is **not** used for the TF readout anywhere. TF being sparse and possibly *emerging with learning* is a **result** ("does TF-encoding density grow / concentrate at particular depths?"), not a load-bearing assumption.
- **Choice/lick:** AUROC FA-lick vs no-lick, Hit-aligned.

Output = value per bin per variable per session → the functional map. **Learning claim** = per-bin regression of encoding vs stage/session (how the map sharpens/strengthens/relocates). FR-normalized; **yield regressed out** (per-bin contributing-unit count); label-shuffle nulls (`permutation_test`); FDR across bins (`fdr_correct`).

### Component 5 — Layer 2: population geometry *(secondary)*

`(trials × time × bin)` tensor → coding directions (`neural_latents.fit_lick_motor_cd`, `motor_subspace`, `project_out_axis`) for sensory/decision/motor; dimensionality (participation ratio); CD angles across stages. **Learning claim** = geometry reorganization. **CCA/Procrustes cross-check:** align per-session activity on the **general baseline-period covariance** (anatomical correspondence already handled by the grid; CCA is the optional robustness layer) and confirm the fixed-grid geometry result survives.

### Component 6 — Layer 3: evoked profiles *(tertiary/descriptive)*

Per-bin event-aligned PSTHs (change / TF / lick), FR-normalized — depth × time heatmaps per stage; the intuitive viewer-style layer.

**Events** follow `EVENT_VALID_OUTCOMES`: Change_ON (hit/miss, big/small), FA & Hit licks, Baseline_ON; TF via the GLM on the baseline period.

## 6. Binning regimes (two, used deliberately — never cross-applied)

- **Slow evoked layers** (change/lick PSTHs, evoked profiles, functional-map change/choice variables): `DEFAULT_BIN_SIZE` 25 ms + `DEFAULT_SIGMA_MS` 25 ms (canonical; these responses evolve over 100s of ms).
- **TF-GLM readout:** the baseline TF is redrawn ~every **50 ms**, so the TF layer runs at the **TF-update resolution (dt = 0.05 s) with no PSTH over-smoothing** — we adopt the **validated TF-GLM's own binning verbatim** rather than imposing the PSTH defaults. This sidesteps the known project gotcha that `TF_SAMPLE_PERIOD = 0.25` is the wrong TF bin (`tf_fluctuation_50ms_vs_constant`). Confirm the GLM's exact dt at implementation.

## 7. Constants ledger (no invented constants)

**Reuse (canonical — `visdetect.analysis.constants` / `config`):** `DEFAULT_BIN_SIZE` (25 ms), `DEFAULT_SIGMA_MS` (25 ms), `EVENT_VALID_OUTCOMES`, `EVENT_RESPONSIVENESS_WINDOWS`, `CHANGE_SIZES` / `SMALL_`/`BIG_CHANGE_SIZES`, `FA_RT_SPLIT`, `STAGE_ORDER` / `STAGE_COLORS`, `assign_shanks(gap_um=120)`, `chanmap_signature`, the TF-GLM's own established parameters, `diagnose_intersession_drift` ±300 µm search range.

**New — flagged for confirmation, not silently chosen:**
- `DEPTH_BIN_UM` — depth-bin width (suggested **60 µm**).
- MUA threshold-crossing params — high-pass cutoff (e.g. 300 Hz) and threshold in ×RMS (e.g. −4 to −5×RMS).

If implementation surfaces any other new parameter, it is **flagged, not invented**, and confirmed with the user.

## 8. Validation strategy (the instrument earns trust before any learning claim)

1. **Registration audit** (Component 0).
2. **Recover known responses** — the depth map must show expected change/lick responses at plausible depths (sanity vs existing single-unit results).
3. **Local vs MUA agreement** — prototype and headline maps must agree; disagreement flags a yield-drift artifact.
4. **Yield control** — regress out per-bin contributing-unit count; report maps with and without.
5. **Nulls & reproducibility** — session-shuffle and label-shuffle nulls; odd/even-trial split reproducibility; FDR across bins.

Note the project's prior lesson: an earlier cross-neuron magnitude result was an **un-normalized firing-rate artifact** (`tf_transient_sustained_state`). Every cross-bin magnitude comparison is FR-normalized and yield-controlled by construction here.

## 9. Deliverables & file layout (repo convention: `scripts/<topic>/`, `FIGURES/<topic>/<SUBJ>/`, `data/cache/<topic>/`, import from `visdetect.*`)

- **Library:** `src/visdetect/analysis/population_field.py` — reusable registration + grid + field-tensor builder (clean interfaces, independently testable).
- **Pipeline:** `scripts/population_field/` — per-subject: audit → build tensor → functional map → geometry → profiles → figures.
- **Caches:** `data/cache/population_field/<SUBJECT>/`.
- **Figures:** `FIGURES/population_field/<SUBJECT>/`.
- **HPC:** one Slurm MUA-extraction job over CephFS (the only non-local piece; outputs mirrored back to a local cache).

## 10. Testing

- Unit tests on synthetic sessions (`visdetect.utils.synthetic`): grid binning, registration shift recovery (inject a known shift → recover it), tensor shape.
- Golden-path test: the local pipeline runs end-to-end on one BG_046 session.

## 11. Reuse map (existing functions — search before writing new)

- **Footprints/geometry:** `tracking_qc.py` — `extract_peak_channel:655`, `extract_footprint:666`, `load_raw_mean_waveform:684`, `load_channel_positions:714`.
- **Registration:** `diagnose_intersession_drift.py` — `session_fingerprint:62`, `estimate_shift:106`, `assign_shanks:131`.
- **Anatomy:** `anatomy/channel_geometry.py` — `assign_shanks:10`, `chanmap_signature:33`; `anatomy/atlas.py:region_at:87`; `anatomy/localize.py` — `build_channel_atlas:60`, `place_channel_on_track:22`. Data: `data/anatomy/<SUBJ>/{channel_atlas,unit_anatomy,session_signatures}.csv`.
- **Tensor/stats:** `analysis/utils.py` — `build_population_tensor:24`, `smooth_psth:131`, `compute_zscore_normalized:152`, `compute_baseline_subtracted:187`, `compute_auroc:399`, `bootstrap_ci:262`, `permutation_test:315`, `fdr_correct:361`.
- **Geometry:** `analysis/neural_latents.py` — `fit_lick_motor_cd:131`, `project_out_axis:121`, `motor_subspace:404`.
- **TF-GLM:** the validated recent pipeline (outputs `data/cache/tf_responsive/*.csv`); confirm exact module/entry point at implementation.
- **Sessions/joins:** `config.canonical_session_id`, `config.session_date_key`, `config.load_staging_manifest(manifest_path=<SUBJ>_staging_manifest.csv)`, `config.chronological_sort`.

## 12. Gotchas / risks (encode in code)

- **Session-id leading-zero-day footgun** — all keys/joins via `canonical_session_id` / `session_date_key` (int64 drops the day for days 1–9). `localize_units.py` writes `session_name` as int — canonicalize on read.
- **Per-subject manifest** — `load_staging_manifest()` returns empty for non-BG_046 subjects unless `manifest_path=` is passed.
- **good_and_stable pkls store spikes only for those units** — the local substrate is inherently limited to them (denser needs the MUA/X: step); no widening without re-ingest.
- **No compute over X:** the MUA extraction runs on HPC/Slurm over CephFS.
- **Memory** — `del sess; gc.collect()` after each session in loops.
- **BG_038 is cortical** (region axis meaningful); **BG_031 VMS**; verify BG_031/BG_038 stage span before making a learning-arc claim for them.
- **Registration robustness** — align on landscape *shape* (rigid), and keep the yield-drift confound handled separately (it is not the registration signal).

## 13. Out of scope (follow-ups)

- LFP / CSD arms of the landscape (NP2.0 has no `*.lf.bin`; requires deriving from `ap.bin` on HPC) — a later extension, not this spec.
- Denser landscape over *all* KS-good units (needs KS `amplitudes`/`spike_positions` from X:).
- Cross-mouse hyperalignment into one shared DMS space (BG_046 + BG_039) — a natural next spec once the per-mouse instrument is validated.
- NoMAD/LFADS-style latent stabilizers.

## 14. Open questions to confirm at implementation

1. `DEPTH_BIN_UM` (suggested 60 µm) and the MUA threshold-crossing params (high-pass cutoff, ×RMS threshold).
2. The TF-GLM's exact entry point and native dt.
3. Per-subject stage span (BG_031, BG_038) — does the learning-arc claim hold, or is it within-stage stability there?
4. Whether to include a truly-naive arm for BG_046 via the unfiltered manifest.
