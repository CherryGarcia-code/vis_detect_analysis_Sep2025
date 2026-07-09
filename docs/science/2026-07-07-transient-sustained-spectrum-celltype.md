# Transient/sustained TF cells: a spectrum, and its relation to spike-waveform type

**Companion to** `docs/science/2026-07-02-transient-sustained-tf-cells.md` (the merged
transient/sustained finding). This answers two follow-on questions with a
locally-recomputed **continuous** kernel width:

1. **Is the transient/sustained identity a spectrum or two discrete classes?** →
   **A SPECTRUM** (a skewed, heavy-tailed unimodal continuum; the earlier hard
   threshold is a convenient cut on that continuum, not a natural kind).
2. **Does the transient/sustained (temporal-width) axis map onto the narrow/broad
   (FSI/SPN) spike-waveform axis?** → **NO — the two axes are ORTHOGONAL.** Kernel
   width predicts change/lick coupling *controlling for* trough-to-peak (and for
   firing rate); the functional identity is not reducible to biophysical cell type.

**Status.** Both headlines independently reproduced and **adversarially verified**
(6-lens skeptic pass, Opus 4.8, Jul 2026): **0/6 lenses refuted, all high
confidence**. Every number below was re-derived from the cache. The caveats in §5
are mandatory framing, not optional polish.

Mice: **BG_046, BG_039 = DMS**; **BG_031 = VMS**. Population: **520 TF-responsive
cell-sessions** in QC-pass, <50%-Disengaged sessions (BG_046 162, BG_039 39,
BG_031 319) — the same population as the 2026-07-02 finding.

---

## 1. The enabling step — a continuous kernel width (local refit)

The registry stored only the 50 ms-grid `kernel_fwhm` (the raw FIR kernel was cached
**nowhere** — verified across the local repo, all X:/ceph `tf_glm_cluster/results*`
staging, and gitignored dirs). Because ~60 % of cells pile at the 0.05 s grid floor,
that discretization can masquerade as either a mode or a continuum and **cannot**
settle the spectrum question. So we recomputed a genuine sub-bin width.

- **Method** (`scripts/tf_responsiveness/state_conditioned/recompute_kernel_width.py`):
  refit the full BG Poisson-GLM per responsive cell **locally from Session pkls**,
  using the exact registry config (`tf_glm_bg_task._cfg("log2")` — movement/phase
  excluded, tiled baseline, standardized, 10-fold, seed 42), extract the raw TF FIR
  kernel `K`, and compute continuous width off it: **`interp_fwhm`** (sub-bin
  half-max interpolation) and **`temporal_spread`** (√ second-moment of |K|). Also a
  model-free fast−slow pulse-PETH width as a cross-check.
- **Parallel & local** — ProcessPool across 69 sessions, 10 workers, BLAS pinned
  1/worker; deterministic-identical to serial. (Local because the BG registries were
  built from Session pkls, not the X:-resident `npx_converted`; no compute over X:.)
- **Validation gate (passed 100 %):** the recomputed grid-FWHM reproduces the
  registry `kernel_fwhm` for **520/520 cells**. `interp_fwhm` vs registry
  `kernel_fwhm` Spearman **0.94**; recomputed peak latency vs registry **1.00**. The
  previously-discarded raw kernel vectors are now cached
  (`data/cache/tf_glm_bg046/kernel_vectors_{subj}.npz`, 520 session-keyed vectors).
- ⚠️ **The two width metrics are not interchangeable:** `interp_fwhm` is strongly
  right-skewed (skew 2.12); `temporal_spread` is near-symmetric. Report separately.
  The model-free `pulse_fwhm` is a **weak** cross-check (Spearman 0.11 vs
  `interp_fwhm`) — inherent, not a bug: the fast−slow PETH of overlapping 50 ms
  pulses is a different quantity from the FIR impulse kernel and the 0.75 s window
  truncates the widest sustained kernels. It is a corroborator only.

---

## 2. Part 1 — the width axis is a SPECTRUM, not two classes

`FIGURES/tf_glm_bg046/spectrum_vs_classes/`.

**Modality battery on the pooled continuous width** (`interp_fwhm`, n=520):

| Test | Value | Reads as |
|---|---|---|
| GMM ΔBIC (1-vs-2) | **+242** | *not* bimodality — see below |
| Silverman crit-bandwidth p(unimodal) | **0.30** | does **not** reject unimodality |
| Sarle bimodality coefficient | **0.528** | below the 0.555 "bimodal" line |
| skew / excess-kurtosis | 2.12 / 7.42 | strongly right-skewed |

The only statistic that "votes classes" is the GMM ΔBIC, and it is a **right-skew
artifact**, proven three ways: (1) **log-transforming** the width removes the skew
(2.12→0.29) and collapses ΔBIC **+242 → +2.4**; (2) a **matched unimodal-lognormal
null** produces median ΔBIC ≈ **168** (100 % of draws positive), so a large positive
ΔBIC is *expected* under unimodality and non-diagnostic (the observed +242 sits at
the 96.5th percentile of that null); (3) the fitted 2-component means (0.085/0.197)
**overlap heavily** (separation index 0.94 ≪ 2 needed for separable classes).
`temporal_spread` is unanimously unimodal (ΔBIC −11, Silverman p=0.41, BC 0.32).

**Region-consistent, not region-dependent.** DMS is the closer-to-bimodal region
(ΔBIC +121.6, BC 0.611, Silverman p=0.183) and VMS the clearer spectrum (ΔBIC
+125.2, BC 0.537, Silverman p=0.410) — but DMS's crossings are the *same* skew
artifact: a matched gamma null reproduces ΔBIC>0 in 100 % and BC>0.555 in 43 % of
sims, and log-transforming DMS collapses ΔBIC to **−1.7**. The skew-robust Silverman
test never rejects unimodality in either region or any single mouse. **No region
shows two classes.**

**The width→function relationship is graded, not stepped.** Segmented (broken-stick)
vs linear regression of outcome coupling on width: ΔBIC = **−7.4 / −11.8 / −11.9**
for Change_ON / Hit-ramp / FA-ramp (negative ⇒ a straight line beats a threshold),
negative in **both regions and all three subjects**; a breakpoint cuts residual sum
of squares by <1 %; binned (octile) means rise smoothly with no step or plateau. The
continuous width→coupling Spearman is monotonic (+0.23 / +0.29 / +0.35). Width and
**latency are independent** (Spearman 0.13), confirming "early" ≠ "transient".

**Read it as:** TF-responsive cells occupy a continuum of TF-response *durations*,
heavy at the short (transient) end with a long tail toward sustained/integrator-like
cells. "Transient" and "sustained" are the ends of one skewed distribution — a useful
descriptive split, not two natural kinds. This **reframes** the 2026-07-02 hard-
threshold result; it does not overturn it (the coupling still scales with width).

---

## 3. Part 2 — the width axis is ORTHOGONAL to spike-waveform (FSI/SPN) type

`FIGURES/tf_glm_bg046/width_vs_waveform/`. Continuous kernel width (`interp_fwhm`)
joined to continuous trough-to-peak `t2p_ms` + the FSI/SPN GMM label (join coverage
**491/520 = 94 %**; BG_031/039 100 %, the 29 unmatched are 8 whole BG_046 sessions
absent from the t2p cache — a coverage gap, not selective dropout).

- **The two axes are uncorrelated.** Spearman(`t2p_ms`, `interp_fwhm`) = **+0.058
  (p=0.20)** pooled, and near-zero within each region (DMS −0.027 p=0.73; VMS +0.090
  p=0.11) — independence is present *within* region, not manufactured by pooling. The
  categorical width-class × FSI/SPN crosstab is likewise ns (χ²=3.54, p=0.06).
- **Width is a functional axis not reducible to cell type.** Regressing each coupling
  metric on standardized width **and** t2p (session-cluster-robust OLS + mixed model):
  **width predicts all three outcomes** — Change_ON b=+0.55 (p=4e-15), Hit-ramp
  b=+1.86 (p=3e-10), FA-ramp b=+1.70 (p=7e-11) — while **t2p does not** dominate (its
  betas are ~4× smaller; ns for Change/FA, marginal for Hit p=0.047).
- **Survives the firing-rate confound** (the exact attack that retracted a prior
  state result): adding baseline rate `base_hz` as a covariate leaves width at
  p<1e-8 for all outcomes; the effect holds within base_hz quartiles and on
  FR-normalized (Δ/base_hz) outcomes. `base_hz` **is** a genuine secondary predictor
  of raw-Hz coupling (report it as a covariate), but width dominates it (standardized
  β ~2–3× larger) and is not an FR proxy (width–base_hz Spearman only 0.11). Wide and
  narrow cells have near-identical firing rates (16.4 vs 14.4 Hz), so no yield-bias
  drive.
- **Holds in both regions** (width significant for all three outcomes in DMS p<0.01
  and VMS p<1e-6, including the smaller DMS subset).

**Read it as:** the functional transient/sustained (duration) axis is a *separate
dimension* from the biophysical FSI/SPN (waveform) axis. Both width-classes are
majority-FSI; there are fast-spiking (narrow) cells that are functionally sustained
integrators and broad cells that are transient. So "sustained integrator cells" are
**not** just SPNs — the coupling structure is organized by response duration, not by
putative cell type.

**Yield-bias caveat (unchanged from 2026-07-02):** these recordings over-sample
narrow/FSI cells (FSI:SPN ≈ 391:100 in the sample) — do **not** read population
fractions as biology. The within-sample width↔t2p independence and the width→coupling
effect are not fraction-dependent.

**Cortex is out of scope:** all three mice are striatal (broad = SPN). In cortex
broad would be putative pyramidal, but BG_038 (M1/S1) has no TF-responsive registry —
a future extension.

---

## 4. Adversarial verification (6 lenses, 0/6 refuted)

Before calling either result solid, six independent Opus-4.8 skeptics re-derived the
numbers from the cache and tried to *refute* each headline (per the standing rule).

- **SPECTRUM** — (i) modality: is +242 real bimodality? → no (skew artifact, above).
  (ii) region: is DMS genuinely bimodal? → no (collapses under log-transform;
  Silverman holds). (iii) graded-vs-stepped: is there a threshold? → no (<1 % RSS
  gain, smooth octiles). **3/3 survive, high confidence.**
- **ORTHOGONAL** — (i) independence given mixedlm non-convergence → survives
  (cluster-robust OLS confirms; non-convergence was a degenerate zero-variance
  session RE on FA-ramp only). (ii) firing-rate/yield confound → survives every FR
  control. (iii) per region → holds within DMS and VMS. **3/3 survive, high
  confidence.**

---

## 5. Mandatory caveats (from the adversarial pass — carry these in any talk/paper)

1. **Never cite GMM ΔBIC as evidence for two classes.** It is skew-inflated; always
   present it with the log-transform (→+2.4) and matched-null (median ≈168) controls.
   The skew-robust **Silverman** test (never rejects unimodality) is the decisive
   metric. Sarle BC is likewise skew-inflated.
2. **Describe the spectrum as skewed/heavy-tailed, not clean lognormal.** Honest
   wrinkle: observed ΔBIC at the 96.5th percentile of the matched null; DMS BC 0.611
   exceeds 96 % of gamma sims; BG_046 alone reaches Silverman p=0.14 — a *whiff* of a
   slightly heavier right tail (a small excess of wide-kernel units), but no second
   mode.
3. **Spectrum is region-consistent** (both DMS and VMS), not region-dependent. State
   that explicitly; DMS is "closer to bimodal" but does not cross over.
4. **Report `interp_fwhm` and `temporal_spread` separately** (skewed vs symmetric);
   don't treat them as interchangeable. `pulse_fwhm` is a weak corroborator only.
5. **Graded coupling is "roughly linear, mildly accelerating at large widths,"** not
   strictly linear (top FWHM octile pulls up; a log/power form fits slightly better).
   The anti-threshold claim rests on the <1 % RSS gain and smooth octiles.
6. **`diptest` was not installed** — Hartigan's dip could not serve as an independent
   fourth modality check; Silverman was the skew-robust primary. Methods limitation.
7. **FR-control the ORTHOGONAL coupling metrics.** They are raw-Hz Δfiring, and
   `base_hz` is a genuine independent secondary predictor — report width with a
   base_hz covariate (or Δ/base_hz), not raw Hz alone. Width stays dominant and
   separable, but reviewers must see the FR control.
8. **"t2p is null" is not universal in DMS:** DMS t2p is nominally significant for
   Hit-ramp (p=0.029) and FA-ramp (p=2e-4), and pooled Hit-ramp t2p is marginal
   (p=0.047). Width still wins by a large margin — say "width predicts coupling
   controlling for t2p," and note t2p is not perfectly null in DMS.
9. **Strict orthogonality is t2p-definition-dependent.** With waveform trough-to-peak
   the correlation is ~0; with *kernel peak latency* as "t2p" it is a small but
   nominally significant +0.13–0.15 (p~0.03). The width→coupling result is robust to
   either; specify the **waveform-t2p** definition when saying "orthogonal."
10. **t2p's only marginal signals split sign across regions** (marginal + in DMS, − in
    VMS) — a classic no-true-effect pattern; report the sign inconsistency, not a
    directional t2p claim.
11. **Pseudoreplication scope:** units are not tracked across sessions, so
    session-clustered/random-intercept SEs address session-level clustering only, not
    cross-session unit reuse. Region samples are imbalanced (VMS 319 ≫ DMS 172);
    BG_039 (n≈39) is individually noisy and only significant pooled into DMS.

---

## 6. Reproduce

```bash
cd <repo-root>
py scripts/tf_responsiveness/state_conditioned/recompute_kernel_width.py   # Component A (~20 min, 10 workers, LOCAL; gate must print 520/520)
py scripts/tf_responsiveness/state_conditioned/spectrum_vs_classes.py       # Part 1 (from cache)
py scripts/tf_responsiveness/state_conditioned/width_vs_waveform.py         # Part 2 (from cache)
```
Caches: `data/cache/tf_glm_bg046/kernel_width_continuous.csv` (+ `kernel_vectors_{subj}.npz`).
Figures + `_stats`: `FIGURES/tf_glm_bg046/{spectrum_vs_classes,width_vs_waveform}/`.

## 7. How to say it in a talk (safe wording)

> "The transient-vs-sustained distinction among TF-responsive striatal neurons is a
> *spectrum*, not two cell types: response duration is a skewed continuum — heavy at
> the fast/transient end with a long tail of sustained integrator-like cells — and
> the change-detection and lick coupling scale *gradedly* with duration, with no
> threshold. And that functional duration axis is *orthogonal* to spike-waveform
> type: it predicts coupling even after controlling for FSI-vs-SPN waveform width and
> for firing rate, so the sustained-integrator cells are not simply the SPNs. Both
> results survive an adversarial multi-lens verification; the one honest nuance is
> that in DMS the waveform axis carries a little coupling signal too, though duration
> dominates."
