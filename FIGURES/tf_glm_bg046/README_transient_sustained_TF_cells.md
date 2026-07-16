# Transient ↔ sustained TF-responsive striatal cells — methods, statistics & figure guide

**The headline, in one paragraph.** Among striatal neurons that respond to the drifting
grating's **temporal frequency (TF)**, the key functional axis is **how long each cell's TF
response lasts** — its *kernel width*. Cells range from **transient** (a brief blip; a
near-pure sensory response) to **sustained** (a prolonged response; integrator-like), and they
form **one continuous, lognormal spectrum — not two classes**. **Where a cell sits on that axis
predicts, in a graded way, how strongly it also carries the change-detection and the lick
signals** — and those are **two separable things**: the sensory change response (ρ = +0.24) and
the *impulsive* early lick (ρ = +0.35), which happens with **no change stimulus on screen** and
survives partialling out the sensory response (partial ρ = +0.275). Benchmarked against the
**11,078 TF-unresponsive cells** (§13d), the claim sharpens: **only the sustained end is
behaviourally engaged — the transient end is indistinguishable from a cell with no TF response
at all**, i.e. a literal sensory relay. This survives firing-rate,
cell-type, region and pseudoreplication controls, goes **flat under a width-label permutation**,
is **not** an artifact of change/lick signal leaking into the kernel (proven by refitting the
GLM without those regressors), and does **not** depend on behavioural engagement state.

> ### ⚠️ The story was UPDATED (July 2026) — read this before quoting the old framing
> The first analysis treated transient vs sustained as **two classes**. A follow-up, using a
> much finer measurement of kernel width, showed the population is really a **continuous
> spectrum** — one single, skewed (in fact **lognormal**) distribution, *not* two kinds of
> cell. **The two-class split is a convenient coarse slice through a continuum, not a
> biological boundary.**
> All the class-based results below still hold — they are just the coarse view of a graded
> effect. **When you present this, lead with "a continuum of response durations", not "two
> cell types".**

**How to read this document**

| Part | Sections | What it covers |
|---|---|---|
| **Part I** | §§1–9 | The original **two-class** analysis + its figures. Still valid; now understood as a coarse slice. |
| **Part II** | §§10–15 | The **continuum reframing** and all the newer figures. **This is the current framing.** |
| ⭐ **Corrections** | §13a | **Read before quoting any pre-July number.** The bugs found in the raw-pulse panel, what survived, what didn't. |
| ⭐ **Verification** | §13b | The GLM leakage test + the permutation null that close the loop. |
| ⭐ **The figure to show** | §13c | `width_continuum_summary/` — the clean, denoised summary. |
| ⭐ **The reference** | §13d | The 11,078 **TF-unresponsive** cells as a baseline — and how they sharpen the claim. |
| Reference | §§16–19 | Plain-language glossary · follow-up questions · how to reproduce · talk wording. |

**Data.** Chronic Neuropixels 2.0, visual TF change-detection task.
Mice: **BG_046 + BG_039 = DMS** (dorsomedial striatum), **BG_031 = VMS** (ventromedial).
**n = 520 responsive cell-sessions** (DMS 201, VMS 319) across 69 sessions.

**Status.** Adversarially verified **three times** by independent multi-agent review (Jul 2026);
every headline number below was re-derived from the cached data. Claims that **failed**
verification are reported here as such, not quietly dropped: behavioural state (§7) is a
**null**, and the third review (§13a) forced four corrections — the model-free width checks are
weaker than first reported, the fast-pulse *population response* panel does not survive, the
suppression fraction is a range not a point, and the three coupling outcomes are **not**
independent. **The width axis and the width→coupling finding survived all three reviews**,
including a direct GLM leakage test and a permutation null (§13b).

---
---

# PART I — the original two-class analysis (§§1–9)

## 1. Shared definitions

- **TF-responsive cells.** Identified per session by a Khilkevich–Lohse-style per-neuron
  ridge-**Poisson GLM** (`visdetect.analysis.tf_glm`). *(A GLM — generalised linear model —
  is a regression that predicts each neuron's spike count from the stimulus. "Poisson"
  because spike counts are counts; "ridge" = a penalty that stops the fit over-reacting to
  noise.)* A cell counts as responsive if its fast-vs-slow pulse-kernel correlation (C1) and
  a residual t-test (C2, p < 0.01) both pass. Units are per-session cluster IDs (**not**
  tracked across days), so each observation is a **cell-session**.
- **Session selection (`good_dates`).** QC-pass (`qc_fail=False`) **and** < 50 % Disengaged
  trials (drops engagement-breakdown sessions that still technically pass QC).
- **Kernel width class** (original). From the GLM TF-kernel full-width-at-half-maximum
  (`kernel_fwhm`): **transient** = fwhm ≤ 0.05 s (one 50 ms bin), **sustained** = fwhm
  ≥ 0.15 s (≥ 3 bins), intermediate between. Population: **520 cell-sessions → 315
  transient (61 %), 99 sustained (19 %), 106 intermediate (20 %)**.
  > ⚠️ **This is the coarse metric.** `kernel_fwhm` is measured on a **50 ms grid**, so ~60 %
  > of cells pile up at the resolution floor. It is a *narrowness index*, not a precise
  > width. **This bluntness is exactly what made a continuum look like two classes** — see
  > §10–§11.
- **Outcome-coupling metrics** (baseline-subtracted Δrate, Hz):
  - **Change_ON response** — hit trials, (0, 0.25) s − baseline (−0.4, −0.05) s. **The clean
    *sensory* leg.**
  - **Hit "pre-lick" window** — hit trials, (−0.3, −0.15) s − (−1.75, −1.25) s.
  - **FA motor ramp** — `fa` trials, same windows (*before* the impulsive early lick). **The
    clean *motor* leg** — an early lick happens with **no change stimulus on screen**, which is
    what makes it an independent probe.
    *(Reminder: the `fa` label = an early/anticipatory lick, NOT an SDT false alarm.)*
  - Alignment respects `EVENT_VALID_OUTCOMES` (Change_ON → hit/miss only, FA → fa, Hit → hit).
  > ⚠️ **`Hit ramp` is NOT an independent third outcome (Jul 2026 correction).**
  > The window is defined **relative to the LICK** (150–300 ms before it), *not* relative to the
  > change — so relative to Change_ON it sits at `[RT − 0.3 s, RT − 0.15 s]`. On a hit the lick
  > *is* the response to the change, and the median reaction time is **0.64 s**, so on the typical
  > trial that "pre-lick" window actually sits **+0.34 to +0.49 s AFTER Change_ON** — i.e. inside
  > the change-evoked response, not before it. Measured over 814 real hits: it overlaps the
  > 0–0.25 s sensory window on **37 %** of trials and sits *entirely inside* it on **10 %**.
  > It is therefore largely the **change response measured again near the lick** — empirically
  > ρ = **+0.58** with Change_ON and **+0.76** with FA ramp.
  > *(It escapes the change on only ~5 % of hits — those with RT ≤ 0.15 s, where reaching back
  > 150 ms from the lick predates the change. A mouse cannot detect and react to a visual change
  > in 150 ms, so those are almost certainly anticipatory licks that happened to land in the
  > response window and got scored as hits; the task filters the very fastest as `ref`, but not
  > all.)*
  > **Report it as a consistency check, not a third test.** The story rests on **two legs:
  > sensory (Change_ON) + motor (FA ramp)** — and those two *are* separable (partial ρ = +0.275,
  > §13a). Note the FA lick has no such problem: it happens in the baseline with **no change
  > stimulus at all**, which is exactly what makes it the clean motor probe.
  > ⚠️ **Baseline clamp (Jul 2026 fix).** The (−1.75, −1.25) s ramp baseline is taken relative to
  > the **lick**, and FA licks are early (median 4.6 s after Baseline_ON, but p5 ≈ 0.36 s) — so on
  > ~**21 %** of FA trials it started *before that trial's own Baseline_ON*, sampling the
  > inter-trial interval / previous trial. Metrics are now **complete-case**: a trial counts only
  > if **both** its baseline and response windows start at/after its Baseline_ON. *(Reassuringly,
  > this changed the width→coupling correlations by <0.01 — the bug added noise, it did not
  > manufacture the effect. That is the honest way to close a flagged concern: test it.)*
- **Statistics.** Non-parametric throughout (Mann–Whitney U, Spearman ρ, Wilcoxon).
  Pseudoreplication controlled by a session random-intercept model + a per-session sign test
  (§6). Cross-neuron magnitudes are firing-rate-controlled.

---

## 2. `transient_vs_sustained/` — the core finding

**What it shows.** Kernel width is (a) largely independent of kernel *latency*, and
(b) strongly predicts outcome coupling.

*(Regenerated Jul 2026 on the baseline-clamped coupling metrics — §1. Only `fa_ramp` moved
materially: its medians were 1.03 / 4.00 before the clamp.)*

| Metric | transient (med) | sustained (med) | Mann–Whitney p |
|---|---|---|---|
| Change_ON response (Hz) | 0.49 | **1.44** | 3.3×10⁻⁷ |
| Hit pre-lick window (Hz) *(not independent — §1)* | 1.18 | **4.26** | 2.1×10⁻⁷ |
| FA motor ramp (Hz) | 0.81 | **3.85** | 4.4×10⁻¹¹ |
| TF selectivity (C1) | 0.26 | 0.32 | 7.6×10⁻¹⁰ |
| baseline rate (Hz) | 12.8 | 14.9 | 5.2×10⁻³ |

- **Latency is the wrong axis.** Kernel *peak time* vs width: ρ = +0.07 (p = 0.10); peak time
  vs the three outcomes: ρ = −0.02 / +0.02 / +0.06 (all ns). "Early vs late" does **not**
  predict coupling — **duration does**. (This is why an earlier latency-based analysis was null.)
- **Not a threshold cherry-pick.** The *continuous* Spearman(width, outcome) is monotonic:
  ρ = +0.18 / +0.24 / +0.32.

**Read it as:** longer-kernel cells fire ~3–4× more strongly to the change and to both licks;
short-kernel cells have a sharp TF response but little change/lick signal.

**Robustness** (`robustness_width_coupling/`): the continuous relationship holds **in both
regions** (Change ρ=+0.19, DMS +0.15 / VMS +0.20; Hit +0.24, +0.36/+0.18; FA +0.32,
+0.30/+0.36), and the gap survives **four different split definitions** — including the
strictest one (fwhm == 0 vs ≥ 0.20 s), where the gap is *largest* (+2.27 / +4.81 Hz,
p = 1.5×10⁻⁸ / 1.7×10⁻⁸). It is not a threshold artifact.

---

## 3. `heatmap_transient_sustained/` — presentation figure

315 transient + 99 sustained cells, three alignments.
- **Top-left:** the kernel-width distribution (what defines the split).
- **Left heatmap:** fast-TF-pulse response, per-unit **peak-normalised** (shows *shape*; the
  ~1 Hz pulse response is small vs ongoing-firing variability, so a z-scale washes it out).
  > ⚠️ **Two corrections here (Jul 2026) — see the boxed note in §13.**
  > (1) The old "**~50 % are suppression-type**" figure was derived from the raw pulse PETH,
  > which is **noise-dominated** (its response sits *below its own noise floor*). The honest
  > fraction, from the **GLM kernel**, is **~30–40 %** — quote it as a **range**, never a point:
  > it is window-dependent (25–51 % across defensible kernel lag windows; 36.9 % at the 0–0.4 s
  > window used here).
  > (2) Treat the "diagonal = cells tiling latencies" impression with caution: cells are
  > ordered **by their own peak latency**, and sorting noise by its own peak *always*
  > produces a diagonal.
- **Middle/right:** Change_ON (hit) and FA, per-unit **baseline z-score**. The sustained block
  lights up strongly; transient barely.

**Why no pulse population-PSTH:** cells tile latencies and half are suppression-type, so a
signed average cancels to ~flat. The width *distribution* is the honest summary. *(The
continuum version, §13, fixes this properly by sign-aligning.)*

---

## 4. `waveform_celltype_join/` — cell type & yield bias

**Methods.** Putative cell type from spike-waveform **trough-to-peak (t2p)** duration via a
2-component Gaussian mixture: narrow = **FSI** (fast-spiking interneuron), broad = **SPN**
(spiny projection neuron). Firing-rate control by decile rate-matching.

- **Yield bias is real:** FSI fraction BG_046 84 %, BG_031 71 %, BG_039 43 % — far above the
  true SPN-dominant striatal composition. Mechanism: **FSIs fire 15.8 vs SPN 6.1 Hz**
  (p = 5.5×10⁻¹⁷) → easier to detect and sort.
  **⚠️ Never quote FSI/SPN percentages as biology.** It biases *composition*, not the
  within-sample contrast.
- **Width ≠ waveform type:** mapping χ² = 4.7, p = 0.03 (weak); both classes are majority-FSI.
- **Coupling survives the firing-rate control:** after rate-matching, the gap persists
  (matched p = 1.5×10⁻⁴ / 9×10⁻⁵ / 1.2×10⁻⁶) and holds **within FSI** and **within SPN**
  separately.

---

## 5. Region check — not a pooling artifact

- **DMS only** (n=201): Change/Hit/FA p = 7.3×10⁻³ / 1.6×10⁻⁵ / 5.3×10⁻⁴.
- **VMS only** (n=319): p = 4.0×10⁻⁵ / 6.0×10⁻⁴ / 1.7×10⁻⁹.

---

## 6. `hardening_pseudoreplication/` — repeated sampling of neurons

Because observations are cell-**sessions** (the same neuron may be re-recorded on many days),
they are not independent. Three controls:
- **Session random-intercept mixed model:** p = 1.8×10⁻⁷ / 1.5×10⁻⁸ / 3.4×10⁻¹³
  (β = +0.98 / +3.49 / +3.88 Hz) — if anything *stronger* than the raw test.
- **Per-session sign test** (session = the unit of replication, 24 sessions): Wilcoxon
  p = 3.9×10⁻³ / 5.7×10⁻⁴ / 5.1×10⁻⁶ (median Δ sustained−transient = +0.98 / +1.92 / +2.73 Hz).
- **Tracked-unit collapse** (BG_046 consensus cohort): only **23 unique units** (14 transient /
  6 sustained) → same direction but **underpowered and ns** (p = 0.78 / 0.062 / 0.051). Do not
  quote it as support; quote it as "the tracked cohort is too small to test this yet".

**Conclusion:** pseudoreplication is neutralised; quote the mixed-model p-values.

---

## 7. `state_x_class/` — behavioural state: **NULL** (a corrected claim)

**Question.** Do sustained cells carry the engagement/task-state signal more than transient cells?

**Answer: NO.** An early version said yes (raw p = 4.9×10⁻³) — but it used a **raw-Hz** metric
that scales with firing rate, and sustained cells fire faster, so the comparison was invalid.
After firing-rate normalisation the class difference is **null**: pooled p = 0.37; ns in every
subject and region; mixed model p = 0.49.

A second route also collapsed: sustained cells' change response *looked* bigger in the
Impulsive state, but that was **lick leakage** — Impulsive hits are ~120 ms faster, so the lick
bleeds into the (0, 0.25) s window. Removing the leakage kills it (p = 4×10⁻³ → lick-censored
0.067 → clean RT>0.25 s subset **p = 0.33, null**).

**Bottom line: the width axis is STATE-INVARIANT.** *(Reusable lesson: any Change_ON-window
metric split by a condition that differs in reaction time must be lick-censored and RT-matched.)*

---
---

# PART II — the continuum reframing (the newer figures)

## 10. Why we revisited it, and how (`kernel_width_continuous.csv`)

**The problem.** The class split rested on `kernel_fwhm`, measured on a **50 ms grid** — so
~60 % of cells sat on the resolution floor. A blunt ruler can *manufacture* the appearance of
two groups out of one continuous population. We needed a **precise, continuous width**.

**The obstacle.** The raw GLM kernel — the actual curve you measure width from — had never
been saved anywhere (verified across the local repo and the cluster storage; only scalar
summaries survived).

**What we did.** We **refit the entire GLM locally** from the raw session files, using the
exact same configuration that produced the original registry, extracted each cell's raw
kernel, and measured its width with **sub-bin precision** (`interp_fwhm` = full-width at
half-maximum, interpolated *between* the 50 ms bins).

**Statistical safeguard (important).** Before trusting the new fine-grained width, we ran a
**validation gate**: recompute the *coarse* width from our refit and check it reproduces the
original registry value. It did, for **520/520 cells (100 %)**. Only then did we use the
continuous value. *(This is what makes the whole Part II trustworthy — the refit is provably
the same model, just measured more finely.)*

Result: `data/cache/tf_glm_bg046/kernel_width_continuous.csv`, 520 cells, median width
**0.106 s** (IQR 0.073–0.171 s), range 0.026–0.691 s.

---

## 11. `spectrum_vs_classes/` — is it two classes, or a spectrum? → **SPECTRUM**

**The question.** Does the continuous width fall into two bumps (classes) or one continuous
distribution (a spectrum)?

**How we tested it — three independent modality tests**, because no single one is reliable:
1. **GMM ΔBIC** — fit the data with 1 vs 2 Gaussian bumps and compare fit scores. Positive =
   two bumps preferred.
2. **Silverman bootstrap** — a test whose null hypothesis is "one bump". A *small* p rejects
   one bump. It is **robust to skew**.
3. **Sarle's bimodality coefficient (BC)** — a summary number; > 0.555 hints at two modes.

Plus a fourth, different question: is the width→coupling relationship **stepped** (a threshold,
implying classes) or **graded** (a straight line, implying a spectrum)? Tested with a
**segmented-vs-linear regression** compared by BIC.

**Results.**

| Test | Result | Verdict |
|---|---|---|
| GMM ΔBIC | **+242** (pooled) | *looks* like classes — **but see below** |
| Silverman | p = 0.31 pooled (DMS 0.18, VMS 0.41) | **never rejects one bump** |
| Sarle's BC | 0.528 (< 0.555) | one bump |
| `temporal_spread` (an alternative, symmetric width measure) | ΔBIC = **−11.2** | unanimously one bump |
| `pulse_fwhm_all` (the **model-free**, leakage-guarded width) | ΔBIC = +235, Silverman p = **0.19**, BC 0.61 | one bump (Silverman never rejects) |
| segmented vs linear (all 3 outcomes) | ΔBIC = **−7.6 / −11.8 / −12.1** (negative) | **graded, not stepped** |

> ⚠️ **This table used to include a `pulse_fwhm` row computed from the 600-pulse-capped PETH.**
> That column was **noise** (ρ = +0.045 to the real width axis — i.e. it correlated with nothing),
> so its modality verdict meant nothing. It is now replaced by the guarded all-pulse
> `pulse_fwhm_all`. The verdict (spectrum) is unchanged either way, but the old row should not be
> quoted.

**⭐ The key catch — the GMM was lying.** Its +242 is a **right-skew artifact**, not evidence of
two classes:
- **Log-transform** the width (which removes the skew) and ΔBIC collapses to **+2.4**.
- Simulate a **matched, definitely-single-bump** (lognormal) population and it reproduces
  ΔBIC ≈ 168 in **100 %** of simulations — our observed value sits at only the 96.5th percentile
  of that *unimodal* null.
- The two fitted "components" **overlap almost completely** (separation 0.94; you need ≫ 2 for
  genuinely distinct groups).

**What it means.** Transient ↔ sustained is **one continuous, skewed population**, and its
coupling to behaviour rises **smoothly** with width. The hard split is a slicing convenience.

> **⭐ Methodological lesson worth carrying:** *GMM ΔBIC is not a unimodality test on skewed
> data — it fires on skew.* Always pair it with a skew-robust test (Silverman/dip), a
> log-transform, and a matched unimodal null.

---

## 12. `width_vs_waveform/` — is width just "cell type" in disguise? → **NO (orthogonal)**

**The question.** Narrow spikes = putative FSIs, broad spikes = putative SPNs. This is *the*
standard cell-type proxy. Are we just re-discovering FSI vs SPN under a new name?

**Method.** Attach each responsive cell's spike-waveform **trough-to-peak (t2p)** duration
(491/520 cells have one). Then ask two things: (a) do t2p and kernel width correlate? (b) does
kernel width **still** predict behavioural coupling once t2p is *statistically held constant*
(and, separately, once baseline firing rate is held constant)?

**Results — the two axes are independent.**

| Test | Result |
|---|---|
| t2p ↔ kernel width (Spearman) | **ρ = +0.058, p = 0.20** (n=491) — essentially **zero**; ns in both regions |
| class × cell-type crosstab | χ² = 3.54, **p = 0.060** — ns |
| Regression with **both** predictors (standardised β) | **width** +0.55 / +1.86 / +1.70 (p = 4×10⁻¹⁵ / 3×10⁻¹⁰ / 7×10⁻¹¹) vs **t2p** −0.11 / −0.40 / −0.22 (ns or marginal, **β ~4× smaller**) |

The four-quadrant breakdown makes it concrete: **within narrow/FSI cells alone**, sustained
cells still couple far more than transient ones (Hit ramp 3.3 vs 1.6 Hz) — i.e. **there are
fast-spiking (narrow-waveform) SUSTAINED integrators.**

**What it means.** Response *duration* is a **functional** property that is **not reducible to
the biophysical cell type**. This is a genuinely new axis, not a relabelling.

> ⚠️ **Yield-bias caveat:** the labelled sample is FSI:SPN = **391:100** — narrow cells are
> massively over-sampled (they fire faster → easier to sort). Do **not** read population
> fractions as biology. The *within-sample* relationships above don't depend on that marginal,
> which is why they remain valid.

---

## 13. The `*_continuum/` figure set — every analysis, redone on the continuous axis

Since it's a spectrum, we re-rendered the whole analysis **against the continuous width**,
**added alongside** (not replacing) the class-based originals.

**Shared method.** Cells are sorted by continuous width and placed in **equal-count bins**
(deciles, or 5 bins of n=104) purely **for display**; each bin shows its mean ± a **bootstrap
95 % confidence interval** (resample cells with replacement 1000×). **The statistic itself is
always computed on the raw, unbinned data** (Spearman ρ) — so no result depends on the binning.

**Statistical considerations carried throughout:**
- **Spearman ρ** (rank-based) → assumes neither normality nor a straight line.
- **segmented-vs-linear BIC** → explicitly tests "graded" against "stepped".
- **Pseudoreplication** → cells are cell-*sessions*, so a session random-intercept model and a
  per-session sign test are run (below).
- **Firing-rate control** → because a raw-Hz metric would just track how fast a cell fires.

| Figure | What it shows | Headline numbers |
|---|---|---|
| **`kernel_families_continuum/`** | ⭐ **What the axis IS.** The average GLM kernel per width bin — sign-aligned, latency-aligned and peak-normalised. Narrow bins = a sharp spike-and-decay; broad bins = a prolonged elevation; **the morph between them is smooth.** | 520 cells, 5 bins (n=104 each), bin medians 0.061 → 0.078 → 0.106 → 0.155 → 0.239 s. *Honest caveat: illustrative-by-construction (the bins are defined on this kernel's width) — it shows what narrow/broad **mean** and that it's graded, it is not an independent test.* |
| **`exemplar_kernels_continuum/`** | ⭐ **Real individual neurons** at each extreme, each with a **95 % confidence band**. | 3 sustained (fwhm 0.37–0.40 s) and 3 transient (0.083–0.090 s). Band = a **trial bootstrap**: resample the cell's trials, refit its GLM, ×200. **All 6 peak CIs exclude zero** → these single-cell shapes are reliable, not fitting artifacts. Grey overlay = the **model-free fast−slow pulse contrast** (see §13a) with its own bootstrap CI: it tracks the kernel's shape (transients blip and return; sustained stay up) but its CI is far wider — which is *why* we read duration off the GLM kernel. |
| **`core_metrics_continuum/`** | Every metric trended against continuous width. | Spearman ρ: **TF selectivity +0.324** (p=3.7×10⁻¹⁴), baseline rate +0.123 (0.005), **Change_ON +0.236** (4.9×10⁻⁸), Hit pre-lick +0.286 (3.0×10⁻¹¹, *not independent — §1*), **FA ramp +0.348** (2.7×10⁻¹⁶). Every segmented ΔBIC **negative** → graded. Holds in **both regions** (Change: DMS +0.22 / VMS +0.24; Hit: DMS +0.42 / VMS +0.21). |
| **`heatmap_continuum/`** | All 520 cells ordered by continuous width (narrow top → broad bottom), three alignments + width-binned PSTH families. | The **width-binned PSTH families** (top row) carry the message: the broadest bin stays elevated post-pulse while the narrowest blips and decays. Must be **sign-aligned** (excitation/suppression otherwise cancel), and the sign is taken from the **GLM kernel**, not the PETH — see §13a. Suppression-type ≈ **30–40 %** (36.9 % at the 0–0.4 s kernel window), and is *less* common at the broad end (**23 % vs 42 %**). ⚠️ **The per-cell heatmap rows are inherently muddy** — sorting is exact (verified 520/520) but the *displayed* per-cell PETH is ~20× below spiking noise, so individual narrow-kernel rows can still look sustained. **Read the axis off `width_continuum_summary/` (§13c), not off individual rows.** |
| **`fa_lick_continuum/`** | Pre-lick (impulsive-lick) ramp vs width. | ρ = **+0.34**, p = 1.7×10⁻¹⁵. % of cells that are lick-responsive **rises monotonically** across width bins: **63 → 62 → 70 → 85 → 87 %**; mean pre-lick ramp +0.036 → +0.183 z. |
| **`hardening_continuum/`** | Pseudoreplication controls, on the continuous axis. | The session random-intercept model **does not converge** here, so the script falls back to (and prefers) a **session-cluster-robust OLS**: β_width = **+0.54** (p=3.3×10⁻¹²) for Change_ON, **+1.81** (6.5×10⁻¹⁰) for Hit. Per-session Wilcoxon median ρ = **+0.27** over 44 sessions (70 % positive, p=2.7×10⁻³). Tracked-unit collapse (n=23) same direction but **ns** (ρ=+0.21, p=0.33) — underpowered, not contradictory. → **not an artifact of re-sampling the same neurons.** |
| **`learning_continuum/`** | Does the width→coupling relation change with learning? | **Within-stage (drift-robust):** Expert ρ = **+0.32 / +0.36 / +0.39** (n=317; all p ≤ 7.6×10⁻⁹); Learning weaker but present (FA +0.28, p=5.3×10⁻⁵; Hit +0.17, p=0.019; Change +0.10, **ns**) (n=203). Per-session slope vs d′ positive but **not significant** (Hit ρ=+0.29, p=0.06; Change ρ=+0.26, p=0.09; FA ρ=+0.07, ns; 44 sessions). |

> ⚠️ **The learning caveat you must state.** Neurons are **not tracked across days**, and the
> chronic probe **drifts** — so *which* cells you record changes over weeks. Any cross-stage
> comparison therefore mixes learning with drift. That's why we lead with the **within-stage**
> result (a relationship *inside* each stage can't be produced by drift changing the sample).

---

## 13a. 🐛 Three bugs in the raw-pulse panel, and what survived (Jul 2026)

This section is the honest history of the **raw fast-pulse PETH** — a *display/validation* path
that sits **beside** the science, never inside it. Everything here is worth reading before you
quote any pre-July number.

**Bug 1 — a pre-rise that could not exist.** Someone spotted the fast-pulse PSTH **rising ~150 ms
BEFORE the pulse**. The stimulus rules that out: the baseline TF is **white noise** (measured
autocorrelation |r| ≤ ~0.003 across 50–300 ms — *at the 50 ms update grid; the raw 60 Hz frame
sequence is autocorrelated only because each TF value is held for 3 frames*), and the
pulse-triggered average of the TF *itself* is a **clean spike at t = 0** (+0.286 log₂, ≤ ~1 % of
that at every other lag, including all negative ones). Nothing precedes a pulse to respond to.
The cause was **circular sign-alignment ("double dipping")**: each cell was flipped by the sign of
**its own post-pulse window**, then that same trace averaged. A smoothed PETH's pre- and post-bins
are correlated (r ≈ +0.20), so choosing the sign on the post window **dragged the pre window
positive**. **Fix:** take the sign from the **GLM kernel** (an estimator not derived from the trace
being averaged).

**Bug 2 — a 600-pulse cap.** The code used **600 of ~41,000** fast pulses/session (~1.5 %), leaving
the PETH noise-dominated. **Fix:** use all pulses.

**Bug 3 (the root cause, found last) — the canonical leakage guard was never applied.** The project
already defines which TF pulses are *eligible* (`TFRespPulseConfig`): **≥1 s after Baseline_ON, ≥1 s
before the change, ≥2 s before an fa/abort/ref lick**. The GLM's pulse detector applies **none** of
these — which is fine *for the GLM*, because it regresses change/lick/reward/wheel out with its own
regressors, but **fatal for a raw PETH**, which has no such protection. Fast pulses live in the trial
baseline and the alignment does not clip at trial boundaries, so a long window around a late-baseline
pulse ran **into the change and the lick**. Worse, that contamination **scales with a cell's
change/lick coupling — which itself correlates with width — so it could FAKE a width→duration
relationship.** **Fix:** apply the canonical guard to every raw-PETH path (~36 % of pulses survive in
BG_046, 28 % in BG_031 — still ~15 k/session, ~25× the old cap).

> ⚠️ **Do NOT "fix" this with per-lag censoring** (NaN-ing lags that run past the change). That
> changes the **sample composition with lag** — pulses dropped at long lags come from ~45–50 %
> *lower-firing* epochs — so a single global baseline drifts upward and fakes a sustained tail.
> Use **pulse-level eligibility + complete-case** (keep only pulses whose *whole* window is clean),
> so the sample is identical at every lag.

### What the third adversarial review changed (report these, not the old numbers)

| claim (pre-July) | verdict | the honest version |
|---|---|---|
| Spearman(PETH's own width, GLM width) = **+0.43** | ⚠️ inflated | **+0.342** (p=1×10⁻¹⁵) once the guard is applied *(broken 600-cap gave +0.07, ns)* |
| model-free `pulse_fwhm` vs GLM width = **+0.48** | ⚠️ inflated | **+0.218** (p=5×10⁻⁷) guarded — the unguarded value was inflated by exactly the coupling-scaled leakage above |
| per-cell shape corr(PETH, kernel) = **+0.82**, 100 % positive | ⚠️ was leakage-driven | the long-lag agreement was largely shared contamination; the **width** summary is far more robust than the full-trace shape |
| fast-pulse population response **+0.0025 (t=14.5), "validated"** | ❌ **did not survive** | **flat** (−0.00004, p=0.82) once the sign is taken split-half **and** trial-start pulses are dropped (their z-baseline sits in the pre-stimulus ITI). **This panel is retired as a claim.** |
| "~49 % suppression-type" | ❌ noise | a **coin-flip on noise** (its response sat below its own noise floor). Honest: **~30–40 %**, window-dependent |
| "**No** scientific conclusion changed" | ⚠️ over-broad | true of the **five headline results**, but the model-free corroboration is **weaker** than claimed, and the suppression fraction *did* change |
| the GLM-kernel sign is "**non-circular**" | ⚠️ overstated | it is **~3× less circular** than the PETH's own sign, not zero. The null offered as proof was **vacuous** (it imported the sign from outside the shuffle — a coin-flip sign scored the same) |

**What this does NOT touch — and why.** The width axis and every width→coupling result come from
the **GLM kernel** and from **event-aligned** (Change_ON / FA) PETHs. `interp_fwhm` is a pure,
sign-invariant function of the cached kernel (independently reproduced for **520/520 cells**,
max diff 1×10⁻¹⁶). The buggy fast-pulse PETH is a **third, display-only path**. That is why the
science survived three reviews while the display panel did not.

> **The reusable lessons.** (1) *Never derive a selection/normalisation/sign from the same data you
> then average* — circular analysis / "double dipping" (Kriegeskorte 2009). (2) **Always run the null
> control**: shuffle the events; if the pipeline still produces structure, it is broken — and make
> sure the null actually re-derives *everything* inside the shuffle, or it proves nothing. (3) A
> guard that exists in the codebase is worthless if the plotting path never calls it.

---

## 13b. The two verifications that close the loop

**(i) GLM leakage test — is the width axis just change/lick variance in disguise?**
This was the one remaining route by which width→coupling could be circular: if the GLM's change/lick
regressors failed to absorb their variance, the residual could be captured by the TF kernel and
**broaden high-coupling cells' kernels**, manufacturing the correlation. We refit **66 cells twice**
on the same spikes and folds — the **full** design, and a **reduced** design with the
change/lick/reward/abort blocks removed:

| test | result |
|---|---|
| kernel correlation (full vs reduced) | median r = **+0.998** (min +0.95) — the TF kernel barely notices |
| width shift | Spearman +0.94; median |Δ| = **2 ms** *(a Wilcoxon says p=0.001, but the shift is 0.5 ms — a textbook large-n/trivial-effect; quote the effect size)* |
| ⭐ **does the width added by dropping change/lick scale with coupling?** | **No** — ρ=+0.14 (p=0.26) vs Change_ON, ρ=+0.22 (p=0.078) vs FA ramp: **ns** |
| width→coupling using full vs reduced width | **identical** (Change +0.22 / +0.25; FA +0.34 / +0.32) |

If leakage drove the result, the reduced-design width would correlate *far more strongly*. It does
not. **The width axis is not a change/lick artifact.** *(It was never likely — the TF regressor is
white noise, hence near-orthogonal to the event regressors — but it is now tested, not argued.)*

**(ii) `null_controls/` — the shuffle the project's own hard rule demands.**
Permute the **width label** across cells and recompute width→coupling. It must go flat, and it does:

| outcome | observed ρ | permuted null | z (global) | z (within-mouse) |
|---|---|---|---|---|
| Change_ON | +0.236 | mean ≈ 0.000, sd 0.043 | **+5.45** | **+5.31** |
| Hit pre-lick | +0.286 | ≈ 0.000, sd 0.043 | **+6.68** | **+6.78** |
| FA ramp | +0.348 | ≈ 0.000, sd 0.044 | **+8.00** | **+8.05** |

p_perm = 5×10⁻⁴ (the floor for 2000 permutations) in every case. The **within-mouse** column matters:
it permutes only *inside* each animal, so the null cannot borrow the between-mouse/region difference —
the effect is not a pooling artifact.

---

## 13c. `width_continuum_summary/` — ⭐ the figure to actually show

**Why it exists.** The per-cell raw-PETH heatmap is **inherently muddy**, and this is not fixable by
sorting: it *displays* a quantity ~20× below spiking noise while *sorting* by a clean one (the kernel
width). We verified the sorting is exact — assigned width matches each cell's independently
recomputed kernel width for **520/520 cells** (max diff 1×10⁻¹⁶), and the order is monotone — so the
sustained-looking rows inside the "transient" block are **correctly-narrow cells with noisy traces**,
not a sorting error. (Within the narrow block, width doesn't predict raw duration at all, ρ=−0.07 ns:
per-cell noise dominates at that scale.) The message lives in the **denoised kernel**, so the fix is
to **average** or **scatter**, not to re-sort.

| Panel | What it shows |
|---|---|
| **A — peak-aligned mean kernel per width bin (ridgeline)** | Averaging **within** width bins cancels the per-cell noise. Narrow (top) → broad (bottom), the response **visibly widens**: bin-mean FWHM **0.054 → 0.064 → 0.079 → 0.093 → 0.124 → 0.243 s**, monotone. *Honest caveat: kernels are aligned to each cell's own peak before averaging (otherwise peaks at different lags smear the mean and destroy the very width being shown), and the bins are defined by each cell's own FWHM — so this is **illustrative-by-construction**, a picture of what the axis means, not an independent test.* |
| **B — every cell as one point on the width axis** | Top marginal = the width distribution: **one mode, right-skewed, no gap** → a **spectrum, not two classes**. Main scatter + decile trend = **broad-kernel cells couple more to the impulsive lick** (ρ = **+0.348**). One panel carries both the continuum and the payoff. |

---

## 13d. ⭐ The TF-**unresponsive** reference — and how it sharpens the whole claim

**Why a reference.** The 520 analysed cells are only **4.5 %** of the 11,598 recorded — a heavily
selected subset. The obvious question is therefore: *how much of the width→coupling story is just
"TF-responsive cells are engaged cells"?* To answer it we computed the same coupling metrics for
the **11,078 TF-UNRESPONSIVE cells**, using the **identical windows, baseline clamp and code**
(`latency_outcome_coupling.py --unresponsive`; only the cell selection differs, `~resp` vs `resp`).
That identity is the point — it is only a fair baseline if it is measured the same way.

> ⚠️ **They are a baseline on the COUPLING axes only — never a point on the width axis.** An
> unresponsive cell's TF kernel is noise, so its "width" is a noise statistic, not a response
> duration. In the figures they appear as a **horizontal band** (median + IQR), which is exactly
> what they can honestly be.

**The result — the narrow end is at baseline, and it is consistent across every metric.**

| metric | narrowest decile | **TF-unresponsive** (n=11,078) | verdict | broadest decile | verdict |
|---|---|---|---|---|---|
| Change_ON response | +0.14 | **+0.11** | **ns** (p = 0.36) | **+2.14** | p = 2.4×10⁻¹⁵ |
| Hit pre-lick | +0.64 | **+0.38** | **ns** (p = 0.61) | **+4.50** | p = 4.0×10⁻¹⁵ |
| FA motor ramp | +0.25 | **+0.38** | **ns** (p = 0.29) | **+4.80** | p = 5.2×10⁻¹⁶ |
| baseline rate (Hz) | 11.4 | **9.0** | **ns** (p = 0.29) | 13.7 | p = 4.0×10⁻⁴ |

**⭐ This reframes the headline.** The story is *not* "TF-responsive cells carry the behavioural
signals, more so if sustained". It is:

> **Only the sustained end carries them. The transient end is behaviourally indistinguishable
> from a cell with no TF response at all** — it responds to the stimulus and to nothing else.

That is a much sharper claim, and a much better fit to the "pure sensory relay" description: a
narrow-kernel TF cell is *literally* as uninvolved in the change-detection and the lick as a
random non-TF neuron. What the width axis tracks is not "how TF-ish" a cell is but **how far it
has moved from sensory relay toward behavioural engagement**.

**How to state it honestly.** "Indistinguishable" is a **null** (n ≈ 52 narrow-decile cells vs
11,078): it means we **cannot detect** any coupling above baseline at the narrow end, not that it
is provably zero. The *positive* half — the broad end sits far above baseline (p ≈ 10⁻¹⁵) — is what
carries the weight. Note also the reference band is wide (the unresponsive IQR spans roughly
−0.3 to +1.7 Hz on FA ramp): plenty of individual non-TF cells are lick-coupled, which is expected
— **licking is represented all over striatum**. The claim is about the *median* cell, not that no
unresponsive cell couples.

**Where it appears:** the reference band is drawn on `width_continuum_summary/` (panel B) and on
every coupling panel of `core_metrics_continuum/`. It is deliberately **not** drawn on the TF
selectivity panel, where an unresponsive cell's value is near-zero *by definition* and a band
would be circular decoration rather than information.

---

## 14. `width_logscale_distribution/` + `width_logscale_fit_diagnostics/` — the width is **LOGNORMAL**

**Background.** The width distribution has a long right tail (a few cells with very long
responses, most short). Buzsáki & Mizuseki (2014, *"The log-dynamic brain"*) note that many
neural variables — firing rates, synaptic weights — are **lognormal**, and that **log is their
natural axis**. Is ours?

**What "lognormal" means in plain terms.** Take the log of every cell's width; if those values
form a **symmetric bell**, the original quantity is lognormal. It means widths are spread
**multiplicatively** (a cell is "2× broader"), not additively.

**Method.** Fit three candidate distributions by maximum likelihood — **lognormal**, **gamma**
(another skewed, positive-only shape) and **normal** — and compare them with:
- **AIC** — a fit score that penalises extra parameters; **lower = better**.
- **Kolmogorov–Smirnov (KS) test** — a goodness-of-fit test; **large p = the model is *not*
  rejected** (the data are consistent with it).
- **Q-Q plot** — plots observed vs predicted quantiles; points on the diagonal = a good fit,
  and it exposes tail misfit that a histogram hides.

**Results (headline figure) — lognormal wins everywhere.**

| Sample | skew, linear → log | AIC: lognormal / gamma / normal | lognormal KS p |
|---|---|---|---|
| Pooled (n=520) | **+2.12 → +0.29** | **−1397** / −1346 / −1088 | 0.032 |
| DMS (n=201) | +2.81 → +0.60 | **−554** / −521 / −387 | **0.18** (not rejected) |
| VMS (n=319) | **+1.48 → +0.09** | **−841** / −823 / −703 | 0.064 (not rejected) |

Taking the log **collapses the skew** — VMS becomes an almost perfect bell (+0.09). That is the
lognormal signature, and it beats gamma and normal decisively.

**The diagnostics figure (supplementary) — why the fit isn't *perfect*, and what fixes it.**
- Freeing the lognormal's **location** (a 3-parameter "shifted" lognormal — a mere **5–23 ms
  offset**) makes the KS test **stop rejecting**: pooled 0.032 → **0.112**; BG_046 0.198 →
  **0.489**. So the leftover deviation is a **small offset, not a failure of lognormality**.
- **Per mouse:** BG_031 alone is **near-exact** (log-skew +0.09); BG_046 alone keeps a genuine
  small residual skew (+0.64). **Pooling mice with different median widths blurs the fit** —
  part of why the pooled panel looks least tidy.
- A **KDE** (a smooth estimate of the data's density) is overlaid so the fit is judged against
  the real shape rather than noisy histogram bars.
- *A perceptual note worth knowing:* on a **linear** axis the lognormal always **looks** like a
  near-perfect fit — because the axis squashes the long tail and hides the misfit there. The
  **log** axis spreads the data out and shows the truth. Same fit, honest view.

> ⚠️ **This is DISPLAY only — no result depends on it.** Our width→coupling statistics use
> **Spearman** (rank-based) and **equal-count bins** — and *both are completely unchanged by a
> log transform* (taking logs doesn't reorder anything). What the log scale buys us is the
> correct natural scale and the lognormal claim, not a different answer.

**What it suggests.** Kernel width behaves like other log-distributed neural variables: **most
cells integrate briefly, a minority integrate for a long time**, spread multiplicatively. Two
practical consequences: (1) report the **median / geometric mean**, never the arithmetic mean;
(2) a log-spread of time constants is exactly the ingredient you would need to build a
**multi-timescale integrator**.

---

## 15. What it all suggests — the synthesis

Striatal TF-responsive neurons do **not** come in two kinds. They occupy a **continuous,
lognormally-distributed spectrum of response durations**, from brief sensory blips to long
integrators. **Where a cell sits on that spectrum predicts, in a graded way, how strongly it
engages with change detection and with licking** — the long-integrator end carries the
decision- and movement-related signal, the brief end is closer to a pure sensory relay.

**The coupling claim stands on TWO legs, not three** (§1): a **sensory** leg (Change_ON,
ρ = +0.236) and a **motor** leg (the impulsive FA lick, ρ = +0.348). These are genuinely
separable — width still predicts the FA ramp after **partialling out** the Change_ON response
(**partial ρ = +0.275**), i.e. broad cells couple to the impulsive lick *beyond* anything they
share with the sensory response. The "Hit ramp" is a **consistency check**, not a third test:
its window is defined 150–300 ms before the *lick*, and since a hit lick follows the change by a
median 0.64 s, that window actually sits **~0.34–0.49 s after Change_ON** — inside the
change-evoked response rather than before it (§1).

That axis is **independent of the classic FSI/SPN spike-shape classification** (so it is a
*functional* property, not a cell type), holds in **both striatal regions**, is **stronger in
the Expert stage**, survives firing-rate, yield-bias and pseudoreplication controls, is
**invariant to behavioural engagement state**, goes **flat under a width-label permutation**
(§13b-ii), and is **not** an artifact of change/lick variance leaking into the kernel — proven
by refitting the GLM with those regressors removed (§13b-i).

**What we do *not* claim.** The raw fast-pulse *population response* panel did not survive
scrutiny (§13a). The **model-free corroboration of the width axis is moderate** once leakage is
removed (ρ ≈ +0.22 to +0.34, not the +0.43–0.48 the contaminated versions showed) — and it is
model-**free**, not data-**independent**: the kernel and the raw PETH come from the *same spikes*.
Call it "the kernel is not a model artifact", not "an independent replication".

---
---

## 16. Plain-language glossary

| Term | What it actually means |
|---|---|
| **TF (temporal frequency)** | How fast the grating drifts. The mouse must detect a *change* in it. It also jitters continuously (~every 50 ms) around a baseline. |
| **A "unit change in TF"** | The unit the GLM measures response *per*. It is **1 SD of the baseline TF jitter = 0.25 octave ≈ a 19 % faster grating.** |
| **GLM kernel** | The neuron's response to a unit TF change, estimated by regressing its spike train against the **whole continuous TF signal** while simultaneously regressing out licks, movement, reward and time-in-trial. It is *not* a raw average around TF pulses. **Why the raw average is worse:** *not* because of stimulus autocorrelation (the TF is white noise — verified), but because (a) the per-pulse response sits **~20× below the spiking noise**, and (b) the raw average controls for none of the nuisance variables. The GLM's regression is simply a far more efficient, better-controlled estimator. |
| **Kernel width / FWHM** | **Full width at half maximum** — how *wide* that response curve is at half its peak height. Small = transient; large = sustained. Our precise version is `interp_fwhm` (measured *between* the 50 ms bins). |
| **Transient / sustained** | The two ends of the width axis. **Not two classes** — the ends of one continuum. |
| **Coupling** | How strongly a cell's firing changes around the stimulus change or around a lick — i.e. how "involved in the behaviour" it is. |
| **Spearman ρ** | A correlation on **ranks**. Doesn't assume a straight line or a bell-shaped distribution — the right default for neural data. |
| **Bootstrap** | Re-run the analysis on hundreds of resampled versions of your own data to see how much the answer wobbles → a confidence interval. |
| **Pseudoreplication** | Counting non-independent observations as if independent (here: the same neuron recorded on many days). Inflates significance if ignored. |
| **Mixed model (random intercept)** | A regression that lets each session have its own baseline, so within-session correlation can't fake an effect. |
| **FSI / SPN** | Fast-spiking interneuron (narrow spike) / spiny projection neuron (broad spike) — the standard waveform-based cell-type guess. |
| **Yield bias** | Fast-firing cells are easier to detect and sort, so they are **over-represented** in the recording. Distorts *proportions*, not within-sample comparisons. |
| **AIC** | A model fit score that penalises extra parameters. **Lower = better.** |
| **KS test** | Goodness-of-fit. **Large p = the model is not rejected** (i.e. it's a plausible description). |
| **Lognormal** | Take the log and you get a normal bell. Values are spread *multiplicatively*. Common in the brain (Buzsáki's "log-dynamic brain"). |
| **Drift** | The chronic probe slowly moves, so *which* neurons you record changes over weeks — a confound for any across-week comparison. |
| **Leakage guard** (`TFRespPulseConfig`) | The project's rule for which TF pulses are *eligible* to analyse: ≥1 s after Baseline_ON, ≥1 s before the change, ≥2 s before an early lick. Stops a pulse's analysis window running into the change or the lick. The **GLM doesn't need it** (it regresses those events out); **any raw average does**. |
| **Complete-case** | Keep only pulses whose **entire** window is clean, rather than trimming offending time-points. Trimming per-lag changes *which* pulses contribute at each lag (and the survivors fire differently), which fakes a drift; complete-case keeps the sample identical at every lag. |
| **Fast−slow contrast** | Response to TF-up minus response to TF-down. Both pulse types sit on the same within-trial firing background, so subtracting **cancels that background** — which a fast-only average cannot. It is the model-free analogue of the GLM kernel (which measures response *per unit* TF). |
| **Circular analysis / double dipping** | Deriving a sign, sort, or normalisation from the same data you then average — it manufactures the effect you were testing for. The fix: take it from an independent estimator, or from held-out data. |
| **Partial correlation** | The correlation between two things *after* removing what both share with a third. Used here to show FA-lick coupling isn't just the sensory response in disguise. |
| **Permutation null** | Shuffle the labels and redo the whole test, many times. The real effect must sit far outside that shuffled distribution — otherwise the pipeline is producing structure from nothing. |

---

## 17. Follow-up questions (in line with the project)

**Most directly enabled by these results**
1. **Are the long-integrator cells the decision integrators the DDM predicts?** The project's
   spine is *learning to suppress impulsivity and boost sensitivity* (drift rate / threshold /
   start point). Sustained cells look like evidence accumulators. **Does a session's
   width→coupling slope, or its population of broad cells, predict that session's fitted DDM
   drift rate or decision bound?** This connects the neural axis directly to the behavioural model.
2. **Does the width spectrum span the task's integration timescale?** Khilkevich & Lohse report
   a ~250 ms brain-wide integration constant. Our widths run ~0.03–0.7 s. **Does the population's
   log-spread of durations tile that timescale — i.e. is this a multi-timescale integrator?**

**Requires the tracked (UnitMatch/DANT) cohort — the enabling step**
3. **Is width a property of a NEURON or of a state?** Cells here are *cell-sessions*. Using
   tracked neurons: **does one cell keep its width across weeks, or does its kernel lengthen as
   the mouse learns?** Right now only **23 tracked cells** overlap — far too few. Expanding the
   tracked cohort is the single highest-leverage next step, and it would also convert the
   learning result (§13) from drift-caveated to clean.

**Cell type / pathway (needs optotagging yield)**
4. **Are the sustained integrators D1 or D2 SPNs?** The width axis is orthogonal to FSI/SPN, so
   the obvious next question is whether it maps onto the **direct/indirect pathway push-pull**
   in the proposal. Currently blocked: only **3 collision-confirmed D1** cells.

**Mechanism**
5. **Where does the long kernel come from?** Is a sustained response **intrinsic** to the
   striatal cell, or **inherited** from a sustained cortical input (PPC / aMOs)? Comparing to a
   cortical recording (BG_038 = M1/S1) or an input-specific manipulation would separate these.
6. **Why lognormal?** A log-distributed set of time constants may be functionally *useful*
   (multi-timescale integration). Is the width spectrum **conserved across mice/regions** (the
   per-mouse fits suggest yes) — and does its *shape* change with learning even if its coupling does?

---

## 18. Reproduce

```bash
cd <repo root>

# --- Part I (original two-class figures) ---
py scripts/tf_responsiveness/state_conditioned/transient_vs_sustained.py       # §2
py scripts/tf_responsiveness/state_conditioned/heatmap_transient_sustained.py  # §3
py scripts/tf_responsiveness/state_conditioned/waveform_celltype_join.py       # §4
py scripts/tf_responsiveness/state_conditioned/hardening_pseudoreplication.py  # §6
py scripts/tf_responsiveness/state_conditioned/state_x_class.py                # §7 (null)

# --- Part II: the continuous width + coupling metrics (SLOW — refit/reload; run once) ---
py scripts/tf_responsiveness/state_conditioned/recompute_kernel_width.py --workers 10   # §10, ~20-25 min
py scripts/tf_responsiveness/state_conditioned/latency_outcome_coupling.py --force      # §1 coupling metrics (baseline-clamped), ~2 min
py scripts/tf_responsiveness/state_conditioned/latency_outcome_coupling.py --unresponsive --workers 12  # §13d TF-unresponsive reference (11,078 cells, ~1 min)
py scripts/tf_responsiveness/state_conditioned/rebuild_peth_traces_all.py --workers 16  # guarded traces, all 520 cells (~25 min)
py scripts/tf_responsiveness/state_conditioned/recompute_pulse_fwhm_allpulses.py --workers 14  # §13a guarded model-free width (~4 min)

# --- Part II: spectrum + cell-type ---
py scripts/tf_responsiveness/state_conditioned/spectrum_vs_classes.py          # §11
py scripts/tf_responsiveness/state_conditioned/width_vs_waveform.py            # §12

# --- Part II: the continuum figure set (fast, cache-only) ---
py scripts/tf_responsiveness/state_conditioned/kernel_families_continuum.py    # §13
py scripts/tf_responsiveness/state_conditioned/exemplar_kernels_continuum.py   # §13
py scripts/tf_responsiveness/state_conditioned/core_metrics_continuum.py       # §13
py scripts/tf_responsiveness/state_conditioned/heatmap_continuum.py            # §13
py scripts/tf_responsiveness/state_conditioned/fa_lick_continuum.py            # §13
py scripts/tf_responsiveness/state_conditioned/hardening_continuum.py          # §13
py scripts/tf_responsiveness/state_conditioned/learning_continuum.py           # §13
py scripts/tf_responsiveness/state_conditioned/width_continuum_summary.py      # §13c ⭐ the figure to show
py scripts/tf_responsiveness/state_conditioned/null_controls_continuum.py      # §13b-ii permutation null

# --- Part II: lognormal / log-scale ---
py scripts/tf_responsiveness/state_conditioned/width_logscale_distribution.py     # §14
py scripts/tf_responsiveness/state_conditioned/width_logscale_fit_diagnostics.py  # §14

# --- optional: the 95% CI bands on the exemplars (SLOW: ~34 min, parallel) ---
py scripts/tf_responsiveness/state_conditioned/compute_exemplar_ci.py          # kernel CIs
py scripts/tf_responsiveness/state_conditioned/compute_exemplar_pulse_ci.py    # grey fast-slow contrast + CI (~2 min)
```
Everything except `recompute_kernel_width.py`, `rebuild_peth_traces_all.py` and
`compute_exemplar_ci.py` runs from cache in seconds/minutes. Each figure writes a `*_stats.txt`
next to it with the exact numbers quoted above.

> **Order matters.** `latency_outcome_coupling.py` writes the coupling metrics that
> `recompute_kernel_width.py` joins into `kernel_width_continuous.csv`; every width→coupling figure
> reads that CSV. If you re-run the coupling metrics, re-run the join (or re-merge the four columns)
> before re-rendering.

---

## 19. How to say it in a talk (safe wording)

> "Striatal neurons that respond to the grating's temporal frequency differ in **how long**
> that response lasts. Crucially, this is a **continuum, not two cell types** — the durations
> form a single, lognormally-distributed spectrum, exactly the kind of log-scaled distribution
> Buzsáki describes for neural variables generally. **Where a cell sits on that spectrum
> predicts how strongly it also carries the change-detection and the lick signal**: the
> long-integrator end is behaviourally engaged, the brief end looks like a pure sensory relay.
> Importantly that holds for **two separable things** — the sensory change response, *and*
> the impulsive early lick, which happens with **no stimulus change on screen at all**; the
> lick coupling survives partialling out the sensory response, so it isn't the same signal
> twice. That axis is **independent of the usual fast-spiking / projection-neuron
> distinction** — we find fast-spiking cells that are long integrators — so it's a *functional*
> property, not a cell class. It holds in both striatal regions, is stronger in expert animals,
> and survives firing-rate, sampling-bias and repeated-measures controls. It goes **flat when we
> shuffle the width labels**, and it's **not** an artifact of movement or change signals leaking
> into the measurement — we refit the model with those regressors removed and the axis doesn't
> move. We did **not** find that it depends on the animal's behavioural engagement state."

**The sharpest version (use the TF-unresponsive reference):**
> "We can benchmark this against the ~11,000 cells that show **no TF response at all**, measured
> exactly the same way. The broad, sustained cells sit far above that baseline on the change
> response and on the impulsive lick. But the narrow, transient cells are **statistically
> indistinguishable from a non-TF cell** — they carry the stimulus and nothing else. So it isn't
> that TF cells are engaged and sustained ones more so; it's that **only the sustained end is
> engaged at all**. The width axis is really tracking how far a cell has moved from being a pure
> sensory relay toward being part of the behaviour."

**If someone asks "how do you know the width is real and not a model artifact?"**
> "Two ways. First, a model-free check: the raw pulse-triggered response, with the stimulus-
> contamination properly excluded, still tracks the model's width (ρ ≈ +0.22 to +0.34) — moderate,
> because a single cell's per-pulse response sits ~20× below the spiking noise, but clearly
> positive. Second, and more decisively, we refit the model with the change- and lick-regressors
> removed: the kernel is essentially unchanged (r = +0.998) and the width→behaviour relationship
> is identical, so the width isn't those signals in disguise."

**Do not say** (these did not survive verification): that the fast-pulse *population response* is
a validated effect; that the model-free check is "independent" (it's model-free but uses the same
spikes); "36.9 % are suppression-type" (say **~30–40 %**); or that the three coupling outcomes are
three independent confirmations (they're **two** — sensory and motor).
