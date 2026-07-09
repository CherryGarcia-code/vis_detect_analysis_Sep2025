# Transient vs sustained TF-responsive striatal cells — methods & statistics

**One-line result.** Among TF-responsive striatal neurons there are two functional
classes defined by the **width** of their TF-encoding kernel: **transient** (fast,
near-pure sensory) and **sustained** (integrator-like). The **sustained** cells
carry the change-detection and lick-related signals; the transient cells do not.
This is robust to firing-rate/cell-type yield bias, pseudoreplication, and region
pooling. A follow-up state analysis was **null** (see §7).

**Follow-up (2026-07-07).** With a locally-recomputed *continuous* kernel width, the
transient/sustained identity is shown to be a **spectrum** (a skewed unimodal
continuum, not two discrete classes; the hard threshold here is a convenient cut on
it), and the temporal-width axis is **orthogonal** to the narrow/broad FSI/SPN
spike-waveform axis (width predicts coupling controlling for trough-to-peak and
firing rate). Both adversarially verified (0/6 lenses refuted). See the companion
`docs/science/2026-07-07-transient-sustained-spectrum-celltype.md`.

**Status.** Independently reproduced and adversarially verified (6-agent review,
Jul 2 2026): every headline number below was re-derived from the cached data.
Verdict PASS for §§2–6; the state claim (§7) was retracted as a firing-rate
artifact and is reported here as a null.

Mice: **BG_046, BG_039 = DMS** (dorsomedial striatum); **BG_031 = VMS**
(ventromedial). Data: chronic Neuropixels 2.0, visual temporal-frequency (TF)
change-detection task.

---

## 1. Shared definitions

- **TF-responsive cells.** Identified per session by the Khilkevich–Lohse-style
  per-neuron ridge-**Poisson GLM** (`visdetect.analysis.tf_glm`): a cell is
  responsive if its fast-vs-slow pulse-kernel correlation (C1) and a residual
  t-test (C2, p<0.01) pass. Registries: `data/cache/tf_responsive/{bg046,bg039,
  bg031}_tf_responsive.csv`. Units are per-session cluster ids (not cross-session
  tracked), so each observation is a **cell-session**.
- **Session selection (`good_dates`).** QC-pass (staging-manifest `qc_fail=False`)
  **and** <50 % Disengaged trials (drops the engagement-breakdown sessions that
  still pass QC).
- **Kernel width class** (`transient_vs_sustained.py`). From the GLM TF-kernel
  full-width-at-half-max `kernel_fwhm` (registry): **transient** = fwhm ≤ 0.05 s
  (one 50 ms bin), **sustained** = fwhm ≥ 0.15 s (≥3 bins), intermediate between.
  Population (good_dates): **n = 520 responsive cell-sessions → 315 transient
  (61 %), 99 sustained (19 %), 106 intermediate (20 %)**.
  ⚠️ `kernel_fwhm` is a **coarse 50 ms-grid narrowness index** — ~60 % of cells
  sit at the resolution floor. It is not a precise width; but the effect below
  survives median-split, continuous Spearman, and excluding the floor, so the
  class contrast is not a threshold artifact.
- **Outcome-coupling metrics** (baseline-subtracted Δrate, Hz, canonical
  `EVENT_RESPONSIVENESS_WINDOWS`):
  - **Change_ON response** — hit trials, response (0, 0.25) s − baseline (−0.4, −0.05) s.
  - **Hit motor ramp** — hit trials, (−0.3, −0.15) s − (−1.75, −1.25) s (before the response lick).
  - **FA motor ramp** — fa trials, same windows (before the impulsive early lick).
  - Event alignment respects `EVENT_VALID_OUTCOMES` (Change_ON→hit/miss, FA→fa, Hit→hit).
- **Statistics.** Non-parametric throughout (Mann–Whitney U, Spearman ρ, Wilcoxon).
  Pseudoreplication controlled with a session random-intercept mixed model and a
  per-session sign test (§6). Cross-neuron magnitudes are firing-rate-controlled.

---

## 2. `transient_vs_sustained/transient_vs_sustained.png` — the core finding

**What it shows.** Kernel width is (a) largely independent of kernel *latency*,
and (b) strongly predicts outcome coupling.

**Methods.** Per cell, `kernel_fwhm`, `kernel_peak_t`, C1 selectivity, baseline
rate, and the three Δrate metrics above (cache
`latency_outcome_coupling/latency_outcome_metrics.csv`). Class contrast by MWU.

**Statistics (verified).**
| Metric | transient (med) | sustained (med) | MWU p |
|---|---|---|---|
| Change_ON response (Hz) | 0.49 | **1.44** | 6.0×10⁻⁷ |
| Hit motor ramp (Hz) | 1.18 | **4.26** | 2.1×10⁻⁷ |
| FA motor ramp (Hz) | 1.03 | **4.00** | 3.8×10⁻¹¹ |
| TF selectivity C1 | 0.26 | 0.32 | 7.6×10⁻¹⁰ |
| baseline rate (Hz) | 12.8 | 14.9 | 6.3×10⁻³ |

- **Latency is the wrong axis:** kernel_peak_t vs kernel_fwhm Spearman ρ = +0.07
  (p=0.10, ~independent); and kernel_peak_t vs the three outcomes ρ = −0.02/+0.02/
  +0.06 (all NS, p=0.19–0.72). So "early vs late" does *not* predict coupling —
  **width does**. (This is why an earlier latency-based analysis was null.)
- **Monotonic, not a threshold cherry-pick:** continuous Spearman(kernel_fwhm,
  outcome) ρ = +0.18 / +0.24 / +0.32 (p down to 4×10⁻¹⁴).

**Read it as:** sustained-kernel cells fire ~3–4× more strongly to the change and
to both licks; transient cells have a sharp TF pulse response but little
change/lick signal.

**Robustness (`robustness_width_coupling/robustness_width_coupling.png`).** Because
`kernel_fwhm` is a coarse 50 ms-grid index, we confirmed the effect is not a
threshold artifact: the **continuous** Spearman(kernel_fwhm, outcome) is monotonic
and significant, **in both regions** — Change ρ=+0.18 (p=3×10⁻⁵; DMS +0.15/VMS +0.19),
Hit ρ=+0.24 (2×10⁻⁸; +0.35/+0.18), FA ρ=+0.32 (4×10⁻¹⁴; +0.32/+0.34); and the
sustained−transient gap is significant across four split definitions (current,
median-split, exclude-floor, strict) — all p<10⁻³ for Hit/FA, three of four for
Change (median-split is the only weak one, expected because it lumps borderline
0.05 s cells into "sustained").

---

## 3. `heatmap_transient_sustained/heatmap_transient_sustained.png` — presentation figure

**What it shows.** 315 transient + 99 sustained cells, three alignments.
- **Top-left:** kernel-width distribution by class (what defines the split).
- **Left heatmap:** fast-TF-pulse response, per-unit **peak-normalized** (shows
  *shape*; the pulse response ~1 Hz is small vs ongoing-firing SD, so a z-scale
  washes it out). Every cell is TF-locked (diagonal = latency tiling). ~50 % of
  responsive cells are **suppression-type** (fire *less* to fast pulses) — TF-
  responsiveness from the GLM is sign-agnostic.
- **Middle / right heatmaps + PSTHs:** Change_ON (hit) and FA, per-unit
  **baseline z-score** (`TwoSlopeNorm −1.5..3`, mostly excitatory). The sustained
  block (bottom) lights up strongly; transient barely.

**Method notes.** Per-unit z to a local pre-event baseline; grand-average PSTHs
are normalize-then-average across the class. Fast pulses subsampled to 600/session
(thousands occur; irrelevant to the mean). Cells ordered identically across
panels (block, then pulse-peak latency).

**Why the pulse pop-mean is not shown as a PSTH:** cells tile latencies and half
are suppression-type, so the signed peak-normalized mean cancels to ~flat and is
uninformative — the width *distribution* (top-left) is the honest summary.

---

## 4. `waveform_celltype_join/waveform_celltype_join.png` — cell-type & yield bias

**What it shows.** How the width classes relate to FSI/SPN waveform type, and that
the narrow-cell over-sampling does **not** drive the finding.

**Methods.** FSI/SPN from trough-to-peak via a 2-component GMM
(`waveform_celltype.py`; BG_046 GMM ΔBIC=6982 = genuinely bimodal, threshold
0.41 ms). Labels joined to responsive cells (94 % coverage). Firing-rate control:
decile rate-matching of transient vs sustained before re-testing.

**Statistics (verified).**
- **Yield bias is real** (as expected in these recordings): FSI (narrow) fraction
  **BG_046 84 %, BG_031 71 %, BG_039 43 %** — vs the SPN-dominant true striatal
  composition. Mechanism: **FSIs fire 15.9 vs SPN 6.1 Hz (p=1.4×10⁻¹⁶)** → easier
  to detect/sort. This biases *population fractions*, not the within-sample contrast.
- **Width ≠ waveform type:** mapping χ²=4.7, p=0.03 (weak); both classes are
  majority-FSI, so kernel width is not redundant with FSI/SPN.
- **Coupling survives the FR/yield control:** after **rate-matching** the gap
  persists (matched-base_hz p≈0.96, yet coupling matched-p = 1.5×10⁻⁴ / 9×10⁻⁵ /
  1.2×10⁻⁶, Δ +0.8/+2.7/+2.8 Hz, 100 % of resamples significant), and it holds
  **within FSI** (p=7.7×10⁻⁴/2.9×10⁻⁴/8.7×10⁻⁸) and **within SPN**
  (p=1.1×10⁻⁶/6.7×10⁻⁶/2.0×10⁻⁵) separately. kernel_fwhm~rate ρ only +0.14.

**Read it as:** the yield bias distorts *composition* claims (don't quote FSI/SPN
percentages as biology), but the transient-vs-sustained functional dissociation is
not a firing-rate or cell-type artifact.

---

## 5. Region check (pooling) — not a pooling artifact

The core contrast holds **independently within each region and each mouse**:
- **DMS only** (BG_046+BG_039, n=201): Change/Hit/FA MWU p = 6×10⁻³ / 1.6×10⁻⁵ / 1.1×10⁻⁴.
- **VMS only** (BG_031, n=319): p = 9.4×10⁻⁵ / 6×10⁻⁴ / 1.0×10⁻⁸.

So although the figures pool the three mice, the effect does not depend on pooling
VMS with DMS.

---

## 6. `hardening_pseudoreplication/hardening_pseudoreplication.png` — pseudoreplication

Because observations are cell-sessions (units not cross-session tracked), we
hardened against non-independence three ways:
- **Session random-intercept mixed model** (statsmodels, all mice; removes
  within-session clustering): class effect p = **1.9×10⁻⁷ / 1.5×10⁻⁸ / 1.3×10⁻¹²**
  (β = +0.97 / +3.49 / +3.76 Hz) — if anything stronger than the raw MWU.
- **Per-session sign test** (session = replication unit, 24 sessions): Wilcoxon
  p = 3.9×10⁻³ / 5.7×10⁻⁴ / 3.0×10⁻⁶.
- **Tracked-unit collapse** (BG_046 UM∩DANT consensus cohort): coverage-limited
  (only 24 responsive cell-sessions overlap → 20 units, ≈1 session/unit, so little
  cross-session duplication to remove) → underpowered but same direction (FA
  p=0.015, Hit p=0.06, Change unresolved p=0.78).

**Conclusion:** within-session pseudoreplication is neutralized; present the
mixed-model p-values.

---

## 7. `state_x_class/state_x_class.png` — behavioural state: **NULL** (corrected)

**Question.** Do sustained cells carry the engagement/task-state population offset
(Lohse task-state CD) more than transient cells?

**Result: NO — null after firing-rate normalization.** An initial version reported
a positive result (raw |engaged−Disengaged| baseline 3.65 vs 2.24 Hz, p=4.9×10⁻³),
but that used a **raw-Hz** metric that scales with firing rate (sustained fire
faster) — an invalid cross-neuron comparison. On the firing-rate-normalized metric
(per-session z), the class difference is **null**: pooled MWU **p=0.37**, per
subject all NS (0.68/0.47/0.31), per region NS (DMS 0.94, VMS 0.31), magnitude
session-mixed-model **p=0.49**. The raw-Hz result was also pseudoreplicated
(pooled over 3 mice) and region-confounded (DMS baseline shifts < VMS). The
"sensory-drive class-invariant" control was vacuous (both classes ~0 on this GLM
metric).

**StimSens vs Impulsive (within-cell, `statesplit_rt_leakage/`).** Sustained cells'
change response *looked* larger in the Impulsive state (paired Wilcoxon p=4×10⁻³),
but this is **lick/RT leakage**: Impulsive hits are ~120 ms faster (0.50 vs 0.63 s,
p=3×10⁻¹¹), so the lick leaks into the (0,0.25) s window. The effect decays as
leakage is removed — uncensored p=4×10⁻³ → **lick-censored (0→lick) p=0.067** →
**RT>0.25 s clean subset p=0.33 (null)**. Transient cells null throughout. So it
is not a genuine gain effect. (Reusable lesson: any Change_ON-window metric split
by a condition that differs in RT must be lick-censored + RT-matched.)

**Bottom line — the transient/sustained axis is STATE-INVARIANT.** Every route
(task-state loading, StimSens-vs-Impulsive, FA-by-state, single-cell state-split)
is null under proper controls. The functional dissociation (§§2–6) does not depend
on behavioural engagement state — consistent with the Lohse picture (sensory and
coupling structure preserved across states; state is a separate/orthogonal axis
not attributable to these cell classes).

---

## 8. Reproduce

```bash
cd <vd_tf_bg046>
py scripts/tf_responsiveness/state_conditioned/transient_vs_sustained.py      # §2
py scripts/tf_responsiveness/state_conditioned/heatmap_transient_sustained.py # §3 (uses cache)
py scripts/tf_responsiveness/state_conditioned/waveform_celltype_join.py      # §4
py scripts/tf_responsiveness/state_conditioned/hardening_pseudoreplication.py # §6
py scripts/tf_responsiveness/state_conditioned/state_x_class.py               # §7 (null)
```
Caches (per-cell metrics) live next to each figure; delete to force recompute
(session-loading scripts take ~10–15 min; plotting from cache is instant).

## 9. How to say it in a talk (safe wording)
> "TF-responsive striatal neurons split into two functional classes by how *long*
> their TF response lasts — a transient, near-pure sensory type and a sustained,
> integrator type. The sustained cells are the ones that also carry the change-
> detection and the lick signals; the transient cells don't. That dissociation is
> robust — it survives firing-rate matching, holds within both putative FSIs and
> SPNs, within each region, and after controlling for repeated sampling of neurons.
> We did *not* find that these classes differ by behavioural engagement state."
