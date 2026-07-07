# Transient/sustained TF cells: spectrum-vs-classes, and mapping onto waveform cell type — design

**Date:** 2026-07-07
**Author:** Claude (design), b.gonzales@ucl.ac.uk (approver)
**Builds on:** `docs/science/2026-07-02-transient-sustained-tf-cells.md`
(the merged transient/sustained finding) and memory
`tf_kernel_latency_outcome_coupling_jul2026`.

---

## 1. Motivation

The merged finding established that TF-responsive striatal neurons split into two
functional classes by the **temporal width of their TF-encoding kernel**:
**transient** (fast, near-pure sensory) vs **sustained** (integrator-like), where
the sustained cells carry the change-detection and lick signals. That split used a
**hard threshold** on the GLM TF-kernel FWHM (`transient` fwhm ≤ 0.05 s, `sustained`
fwhm ≥ 0.15 s), and the kernel-width vs waveform-type mapping was only lightly
touched (a weak categorical χ², p=0.03).

Two open questions remain:

1. **Is the transient/sustained identity a genuine spectrum, or two discrete
   classes?** The current `kernel_fwhm` is quantized to the 50 ms GLM lag grid, so
   ~60 % of cells pile at the resolution floor — that grid artifact can masquerade
   as either a mode or a continuum, so it cannot settle the question.
2. **How does the transient/sustained (temporal) axis map onto the narrow/broad
   (spike-waveform) axis** — putative FSI vs SPN in these striatal recordings? Are
   the two axes aligned (functional identity reducible to biophysical cell type) or
   orthogonal (a functional dimension independent of cell type)?

### The two "widths" — terminology (they are different things)

| Term | Measures | Axis it defines | Units | Source |
|---|---|---|---|---|
| **Kernel width** | how long in time the TF response lasts | **transient ↔ sustained** | seconds (~0–0.6) | GLM TF-kernel FWHM |
| **Waveform width** | trough-to-peak of the spike shape | **narrow ↔ broad** (FSI ↔ SPN) | milliseconds (~0.2–0.6) | spike waveform `t2p_ms` |

Question 1 concerns the **kernel (temporal) width** only. Question 2 relates the two.

---

## 2. Decisions made at brainstorming

- **Scope: striatum only** — BG_046 + BG_039 (DMS) and BG_031 (VMS). Narrow = putative
  FSI, broad = putative SPN. Cortex (BG_038, where broad = putative pyramidal) has
  **no TF-responsive registry** and is **out of scope** (documented as future work).
- **Width metric: recompute a continuous kernel width** (the registry's 50 ms-grid
  `kernel_fwhm` cannot answer Q1).
- **Primary width = canonical GLM-kernel FWHM** (recomputed at sub-bin resolution),
  with the **model-free TF-pulse-response width as a cross-check**.

### Feasibility findings that shape the plan

- **The raw GLM FIR kernel is cached nowhere** — verified exhaustively across the
  local repo (including gitignored dirs), all X: `tf_glm_cluster/results*` staging
  (100 % scalar-only CSVs), and by confirming no code path persists the kernel. The
  pipeline saved only the two scalars `kernel_peak_t`, `kernel_fwhm` and discarded
  the vector every run. **⇒ a refit is genuinely required.**
- **The BG registries were produced from local Session pkls**, not the MoHa
  `npx_converted` data on X:. The worker is
  `scripts/tf_responsiveness/cluster_bg/tf_glm_bg_task.py`: `load_session` →
  `session_trial_regressors` (movement/phase excluded; tiled-baseline + standardized
  + log2-TF) → `fit_poisson_cv` → `_tf_kernel`. **⇒ the refit runs fully locally on
  `data/pkls/`, with NO compute over X:** (respects the hard "no compute over X:"
  rule).
- The worker already computes the raw kernel `K = _tf_kernel(full_fit, design, cfg)`
  in memory — we simply re-run the full-model fit per responsive cell and keep `K`.

---

## 3. Population and shared conventions

- **Cells:** responsive (`resp_log2 == True`) in `good_dates` — QC-pass
  (staging-manifest `qc_fail == False`) **and** < 50 % Disengaged trials. This is the
  **same population** as the merged finding (n ≈ 520 responsive cell-sessions across
  ~24 sessions, 3 mice), so results are directly comparable.
- **Unit of observation:** cell-session (units are not cross-session tracked).
  Pseudoreplication is controlled with session random-intercept mixed models and
  per-session / per-mouse breakdowns, matching the existing doc's rigor.
- **Reuse:** `representative_cells.REPO / _registry / good_dates / _spikes`;
  `transient_vs_sustained.load_cells` (registry + cached outcome metrics);
  `tf_glm_bg_task._cfg("log2")` for the exact fit config; `tf_glm._tf_kernel`,
  `assemble_design`, `fit_poisson_cv`, `session_trial_regressors`,
  `pulse_times_from_tf`, `tf_pulse_peth`.
- **Paths — write to the PRIMARY repo, not the retired worktree.** The reused
  `state_conditioned` helpers hardcode `E:/.../vd_tf_bg046/...` (a deleted worktree).
  New scripts must resolve outputs under repo root:
  `FIGURES/tf_glm_bg046/<fig>/` and `data/cache/tf_glm_bg046/`. Where a reused helper
  exposes a stale `OUT`, override it locally (do not write to `vd_tf_bg046`).
- **Cached outcome-coupling metrics** (`change_on`, `hit_ramp`, `fa_ramp`, `base_hz`)
  are read from the existing `FIGURES/tf_glm_bg046/latency_outcome_coupling/
  latency_outcome_metrics.csv` — no recompute needed.

---

## 4. Component A — recompute continuous kernel width (the enabling step)

**Script:** `scripts/tf_responsiveness/state_conditioned/recompute_kernel_width.py`
**Cache out:** `data/cache/tf_glm_bg046/kernel_width_continuous.csv` (+ raw kernel
vectors `kernel_vectors_<subj>.npz`, the artifact the pipeline never saved).

For each `good_dates` session (all 3 mice), load the local pkl once, build the design
via `session_trial_regressors` with `tf_glm_bg_task._cfg("log2")`, and for each
**responsive** unit in the session:

1. Fit the full Poisson GLM (`fit_poisson_cv(d.X, y, cfg, folds)`) — only the full
   model is needed for the kernel (skip the reduced fit, pulse test, and linear
   control to save time).
2. Extract the raw FIR kernel `K = _tf_kernel(full_fit, d, cfg)` and the lag grid
   `lags = _lag_offsets(cfg.kern["tf"], bs) * bs`.
3. Compute continuous width measures off `K`:
   - **Interpolated FWHM** — linear interpolation of the half-max (`|K_peak|/2`)
     crossings on each side of the peak → sub-bin FWHM in seconds. Continuous
     analogue of the registry `kernel_fwhm`.
   - **Effective duration** — area/second-moment temporal spread
     `√( Σ|K(t)|·(t − t̄)² / Σ|K(t)| )`, with `t̄ = Σ|K|·t / Σ|K|`. Fully continuous,
     robust to a single noisy crossing, sign-agnostic (handles suppression cells).
   - **Grid FWHM (validation)** — recompute the pipeline's exact
     `lags[hi] − lags[lo]` walk-out for the byte-for-byte gate below.
4. **Model-free pulse-response width (cross-check)** — from the fast/slow pulse
   PETHs (`tf_pulse_peth` on `y` at `pulse_times_from_tf`, subsample fast pulses to
   ~600/session as the heatmap does): take the **fast−slow** contrast trace, and
   measure the same interpolated-FWHM + effective-duration on it. Independent of the
   regression.

**Validation gate (must pass before trusting the continuous width):** the recomputed
**grid FWHM** must reproduce the registry `kernel_fwhm` per cell (equal on the 50 ms
grid, allowing only fit-determinism jitter). If it does not, stop and diagnose the
config mismatch — do not proceed. Also report Spearman(continuous FWHM, registry
`kernel_fwhm`) and Spearman(continuous FWHM, model-free pulse width) as
consistency checks; and, as a secondary independent cross-check, Spearman against
`data/cache/tf_labeling/tf_cell_classification.csv` `half_width_fast_ms` (a different
pipeline/population — expect positive but weaker).

**Compute:** responsive cells only, one full fit each (~seconds/cell per the
pipeline's logged `fit_s`), one session load each. ~10–20 min/mouse, local, reads
`data/pkls/` (never X:). Parallelizable per session with a ProcessPool
(BLAS pinned to 1 thread/worker) if needed.

---

## 5. Component B — Part 1: spectrum vs classes (Q1)

**Script:** `scripts/tf_responsiveness/state_conditioned/spectrum_vs_classes.py`
**Fig out:** `FIGURES/tf_glm_bg046/spectrum_vs_classes/spectrum_vs_classes.png`
(+ `.pdf`, `_stats.txt/.csv`). Reads the Component-A cache; no session reloads.

1. **Width distribution + modality tests** on the continuous width (primary =
   interpolated kernel FWHM; repeat on effective-duration and on the model-free
   width to confirm the verdict is not measure-specific):
   - **Hartigan's dip test** (unimodality). p < 0.05 ⇒ significantly non-unimodal
     (evidence for classes); NS ⇒ consistent with a continuum.
   - **GMM 1-vs-2-component ΔBIC** (same method as the T2P waveform test), with a
     bootstrap CI on ΔBIC and the fitted component means/weights.
   - **Silverman critical-bandwidth bootstrap** as a secondary modality check.
   - Pooled + per region (DMS/VMS) + per mouse.
2. **Latency ⊥ width** — Spearman(`kernel_peak_t`, continuous width) to confirm
   "early" ≠ "transient" (registry hint ρ ≈ 0.07). Scatter panel.
3. **Graded vs stepped function** — is outcome coupling (`change_on`, `hit_ramp`,
   `fa_ramp`) a graded/monotonic function of continuous width, or is there a
   threshold?
   - Continuous Spearman + a binned-mean / LOESS curve (per outcome).
   - **Segmented (broken-stick) regression** with a free breakpoint vs a straight
     line, compared by BIC (and an F-test on the improvement). A non-improving
     breakpoint ⇒ graded continuum; a well-localized breakpoint ⇒ two regimes.
   - Cross-neuron magnitudes are firing-rate-controlled (partial on `base_hz`).
4. **Interpretation guardrail:** whatever the verdict, state it honestly. A spectrum
   result **reframes** the earlier finding (the hard threshold is an analytically
   convenient cut on a continuum, with the sustained end carrying the coupling) but
   does **not** overturn it. A bimodal result would upgrade the classes to natural
   kinds.

---

## 6. Component C — Part 2: mapping onto narrow/broad waveform type (Q2)

**Script:** `scripts/tf_responsiveness/state_conditioned/width_vs_waveform.py`
**Fig out:** `FIGURES/tf_glm_bg046/width_vs_waveform/width_vs_waveform.png`
(+ `.pdf`, `_stats.txt/.csv`). Joins continuous width (Component A) to continuous
trough-to-peak `t2p_ms` and the FSI/SPN GMM label. **Note the filename asymmetry:**
BG_031/BG_039 are `data/cache/talk_substrate/waveform_t2p_BG_{031,039}.csv` but
BG_046 is `data/cache/talk_substrate/bg046_waveform_t2p.csv` — resolve per-subject,
do not glob `waveform_t2p_BG_*` (it silently drops BG_046). All keyed by `session_8`
+ `cluster_id`.

1. **Joint continuous distribution** — 2D scatter/density of (`t2p_ms`, kernel width)
   per region; Spearman(`t2p_ms`, width) pooled + per region + per mouse. Aligned
   (one axis) vs orthogonal (two axes)?
2. **Independence / not-reducible-to-cell-type** — regress each outcome-coupling
   metric on **width and `t2p_ms` jointly** (+ session random intercept). Does width
   retain predictive power **controlling for** waveform type (partial correlation /
   mixed-model coefficient)? Test the reverse too (does `t2p_ms` add anything beyond
   width?). This is the crux: if width predicts coupling independent of `t2p`, the
   functional axis is not reducible to biophysical cell type.
3. **Four-quadrant view** — narrow-transient / narrow-sustained / broad-transient /
   broad-sustained (using continuous medians or the established thresholds): counts +
   outcome coupling per quadrant. Do **sustained FSIs** and **transient SPNs** exist?
   (the cells that break the naive FSI=transient / SPN=sustained prior).
4. **Region/location** — DMS (BG_046+039) vs VMS (BG_031) separately; note the
   VMS-leaning-sustained hint from the registry. State explicitly that in cortex
   broad would = pyramidal, but cortex is out of scope (no TF registry).
5. **Yield-bias caveat** — carry forward the confirmed narrow-cell over-sampling
   (FSI fraction BG_046 84 % / BG_031 71 % / BG_039 43 %; mechanism = FSIs fire
   faster). Interpret population fractions cautiously; the within-sample width↔`t2p`
   relationship and the independence test are not fraction-dependent. Rate-match
   (decile) as a robustness control.

---

## 7. Statistics & rigor (both parts)

- Non-parametric by default (Mann–Whitney U, Spearman, Wilcoxon).
- FR-normalize / partial-out `base_hz` on **every** cross-neuron magnitude comparison
  (the retracted state result was a raw-Hz artifact — do not repeat it).
- Session random-intercept mixed models + per-session and per-mouse/region
  breakdowns on every headline claim (not just the pooled number).
- Bootstrap CIs (1000 resamples, seed 42) on key estimates; report effect sizes with
  every p-value.
- **Adversarial verification** of any headline via a multi-agent skeptic pass
  (Opus 4.8 subagents), per the standing rule, before the result is called solid.

---

## 8. Deliverables

- `scripts/tf_responsiveness/state_conditioned/recompute_kernel_width.py` (+ cache
  `data/cache/tf_glm_bg046/kernel_width_continuous.csv`, kernel-vector npz).
- `scripts/tf_responsiveness/state_conditioned/spectrum_vs_classes.py` (+ figure,
  stats).
- `scripts/tf_responsiveness/state_conditioned/width_vs_waveform.py` (+ figure,
  stats).
- A companion science write-up
  `docs/science/2026-07-07-transient-sustained-spectrum-celltype.md` (methods +
  verified stats + safe talk wording), cross-linked from the 2026-07-02 doc.
- Memory update to `tf_kernel_latency_outcome_coupling_jul2026` (or a new memory) with
  the spectrum-vs-classes verdict and the width↔waveform mapping result.

---

## 9. Out of scope (explicit)

- Cortex / pyramidal mapping (BG_038) — needs a TF-GLM run on cortical data first.
- Cross-session unit tracking of the width axis (coverage-limited; the existing
  consensus-cohort collapse already showed this is underpowered for these cells).
- Learning/developmental trajectory of the spectrum (drift-confounded; separate
  question).
- Any re-derivation of the state axis (settled null).

---

## 10. Reproduce (once built)

```bash
cd <repo-root>
py scripts/tf_responsiveness/state_conditioned/recompute_kernel_width.py   # Component A (session reloads; ~10-20 min/mouse, LOCAL)
py scripts/tf_responsiveness/state_conditioned/spectrum_vs_classes.py       # Part 1 (from cache, instant)
py scripts/tf_responsiveness/state_conditioned/width_vs_waveform.py         # Part 2 (from cache, instant)
```
