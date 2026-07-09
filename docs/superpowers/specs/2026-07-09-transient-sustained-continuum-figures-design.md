# Continuum re-renders of the transient/sustained TF-cell figures — design

**Date:** 2026-07-09
**Builds on:** `docs/science/2026-07-07-transient-sustained-spectrum-celltype.md` (the
SPECTRUM finding) and the 2026-07-02 transient/sustained figures.

## 1. Motivation

The transient/sustained identity is a **spectrum**, not two classes (2026-07-07).
The earlier figures visualized it as a hard transient-vs-sustained split (boxplots,
two-class PSTHs, class-block heatmaps). This adds **continuum/binned** versions of
those analyses — cells organized by *continuous* kernel width — **alongside** the
originals (additive; nothing overwritten). Two of the new figures
(`spectrum_vs_classes`, `width_vs_waveform`) already give the coupling and
width↔waveform continuum; this covers the remaining transient/sustained figures.

## 2. Decisions (from brainstorming)

- **Scope = broad sweep:** 5 continuum figures (core metrics, heatmap, hardening,
  learning, FA-lick).
- **Representation = binned deciles + trend overlay:** per-cell scatter + width-decile
  mean ± bootstrap CI + a monotonic trend line + the continuous Spearman.
- **PSTHs are included** where the originals had them, rendered as a **width-binned
  family** of population-average traces (4–5 gradient-colored width bins) — the
  continuum analogue of the two class-mean PSTHs.
- **Width axis = `interp_fwhm`** (the primary continuous width; `temporal_spread`
  available as a robustness overlay).
- **Mostly cache-based, ONE trace rebuild.** The metrics-based figures (core, hardening,
  learning) are fully cache-based — verified `latency_outcome_metrics.csv`,
  `kernel_width_continuous.csv`, `consensus_members.csv`, the staging manifests, and
  the registry (`c1_r_log2`) all exist on primary. **BUT** the cached
  `peth_traces.npz` holds only **414 cells (transient+sustained) — it EXCLUDES the
  ~106 intermediate cells, which are the middle of the continuum.** Using it for a
  "continuum" figure would omit the continuum's center, so the heatmap + FA-lick
  figures require a **one-time trace rebuild for ALL 520 responsive cells** (Component
  0 below): session reloads (~15 min, LOCAL, no X:), reusing the heatmap `build()`
  logic without the class filter. Run as a background bash from the main session (per
  the long-compute lesson).
- **Additive & primary paths:** new scripts write to new `FIGURES/tf_glm_bg046/
  *_continuum/` dirs; the class-based scripts/figures are untouched. New scripts use
  repo-root (`REPO`) paths, never the retired `vd_tf_bg046` worktree.
- **Git:** a new branch `feature/tf-continuum-figures` off the updated `main`
  (996d58d). Safe: `main` doesn't touch the anatomy files, so the parallel chat's
  uncommitted anatomy WIP carries across the branch switch untouched; the work is
  additive so no conflict with the population-field work.

## 3. Shared helper — `continuum_common.py`

`scripts/tf_responsiveness/state_conditioned/continuum_common.py`:
- `REPO`, `REGION = {BG_046:DMS, BG_039:DMS, BG_031:VMS}`, `OUTCOMES`, a width→bin
  gradient colormap.
- `load_width_metrics()` → one dataframe per responsive cell (good_dates population,
  520 cells): join `kernel_width_continuous.csv` (`interp_fwhm`, `temporal_spread`,
  `base_hz`, `change_on`, `hit_ramp`, `fa_ramp`, `kernel_peak_t_registry`) to the
  registry `c1_r_log2` (TF selectivity). Keyed by (subject, session, unit).
- `binned_trend(ax, x, y, n_bins=10, seed=42, ...)` → the reusable panel: faint
  per-cell scatter + decile-binned mean ± bootstrap CI (1000 resamples) + a monotonic
  trend line + a Spearman(ρ, p) annotation. One implementation, used by every figure.
- `width_bin_family(width, n=5)` → assign each cell to one of `n` width bins (for the
  PSTH families), returning bin index + gradient colors + bin-edge labels.

## 3.5. Component 0 — rebuild per-cell traces for ALL 520 cells

`rebuild_peth_traces_all.py`: reuse the heatmap `build()` path (load each good_dates
session, compute per-cell z-scored pulse/Change_ON/FA PETH traces) but **without the
transient/sustained class filter**, so all 520 responsive cells (incl. intermediates)
are included. Write `data/cache/tf_glm_bg046/peth_traces_all.npz` (meta_subject/
session/unit + t_pulse/change/fa + mat_pulse/change/fa). LOCAL (reads `data/pkls/`,
never X:), run as a background bash from the main session.

**Parallelized across sessions** (per the standing rule; the session loop is
independent) — `ProcessPoolExecutor` over the ~24 good sessions, param `n_workers`
(default ~10), BLAS pinned to 1 thread/worker (env before numpy import), mirroring
`recompute_kernel_width.py`. Cuts ~15 min serial to a few minutes. **Deterministic:**
each worker uses a fixed per-session seed for the fast-pulse subsampling
(`PULSE_CAP=600`), so the output is identical run-to-run and to a serial run; traces
re-assembled + written in the parent. Prerequisite for 4b + 4e only; 4a/4c/4d don't
need it.

## 4. The 5 figures (each `*_continuum.py`, own `FIGURES/tf_glm_bg046/*_continuum/`)

### 4a. `core_metrics_continuum` (re-renders §2 `transient_vs_sustained`)
Decile tuning-curves (`binned_trend`) of each §2 metric vs continuous width: **TF
selectivity `c1_r_log2`, baseline rate `base_hz`, Change_ON, Hit-ramp, FA-ramp**. One
panel per metric + a width-distribution panel + a stats txt (per-metric Spearman +
the segmented-vs-linear ΔBIC from `spectrum_stats`, echoing the "graded not stepped"
result). Replaces the transient/sustained boxplots.

### 4b. `heatmap_continuum` (re-renders §3 `heatmap_transient_sustained`)
From `peth_traces_all.npz` (all 520 cells, Component 0) joined to continuous width:
- Per-unit **heatmaps** for pulse / Change_ON / FA, all cells **ordered by continuous
  width** (no class blocks) with a continuous width colorbar strip on the left.
- **PSTH families** (the continuum analogue of the class-mean PSTHs) above each
  heatmap: population-average trace per width bin (`width_bin_family`, ~5 gradient
  bins), for pulse / Change / FA. Shows the response morphing along the width axis.

### 4c. `hardening_continuum` (re-renders §6 `hardening_pseudoreplication`)
Continuous-width robustness of the width→coupling relationship (Spearman/regression,
not a class gap):
- Session **random-intercept regression** `outcome ~ z(interp_fwhm)` (+ C(region)),
  per outcome — the width slope + p (mixedlm; cluster-robust OLS fallback per the
  Task-5 lesson).
- **Per-session Spearman**(width, outcome) sign test (session = replication unit;
  Wilcoxon over sessions).
- **Tracked-unit collapse** (BG_046 consensus cohort): collapse to one continuous
  width + one coupling value per `um_uid`, re-test Spearman(width, outcome).
- Panel: `binned_trend` per outcome + a raw-vs-hardened effect comparison.

### 4d. `learning_continuum` (re-renders `learning_transient_sustained`)
- **Within-stage** Spearman(width, outcome) per stage (Learning, Expert) — drift-robust:
  does the graded width→coupling relationship hold within each stage? `binned_trend`
  per stage overlaid.
- **Per-session** width→coupling slope vs behavioural d′ (+ session-order partial, the
  drift proxy) — the continuous analogue of the per-session gap-vs-d′ panel. Carry the
  same drift-confound caveat.

### 4e. `fa_lick_continuum` (re-renders `fa_lick_activity`)
From `peth_traces_all.npz` (FA traces, all 520; Component 0) + `lick_acquisition_cells.csv`
(note: lick labels cover the 414 transient/sustained cells — intermediates without a
lick label are shown in the heatmap/ramp but greyed in the lick-responsive overlay):
- **Pre-lick ramp** (mean z in the (−0.3,−0.15) s window) vs continuous width
  (`binned_trend`) — replaces the transient-vs-sustained ramp comparison.
- **Width-ordered FA heatmap** with a width colorbar + a lick-responsive overlay strip.
- **FA PSTH family** by width bin (the continuum PSTH), annotated with % lick-responsive
  per bin.

## 5. Statistics & conventions
- Non-parametric (Spearman, Wilcoxon); bootstrap CIs (1000, seed 42).
- FR note: coupling metrics are raw-Hz Δfiring; `base_hz` is a real secondary
  predictor — where a magnitude claim is made, note it (carry the 2026-07-07 caveats).
- Population = the 520 good_dates responsive cells (same as the new work).
- Every figure saves png + pdf + a `_stats.txt`.

## 6. Deliverables
- `scripts/tf_responsiveness/state_conditioned/continuum_common.py` + the 5
  `*_continuum.py` scripts.
- Figures/stats under `FIGURES/tf_glm_bg046/*_continuum/` (gitignored; regenerated).
- A short note appended to the 2026-07-07 doc pointing at the continuum figure set.

## 7. Out of scope
- Re-deriving any statistic already adversarially verified (this is visualization of
  the established SPECTRUM result, not new inference).
- The state/RT-leakage figures (state axis is a settled null).
- Session-reload rebuilds (everything is cache-based).
