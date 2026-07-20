# Fig 5 e–h ported to transient / sustained / non-TF cell classes

**A faithful within-striatum reproduction of Khilkevich & Lohse (Nature 2024)
Figure 5 e–h, replacing the brain-area grouping with the project's
transient → sustained kernel-width axis plus the non-TF (TF-non-responsive)
population.**

Status: DESIGN (brainstorming output). Not yet planned or implemented.
Author session: 2026-07-20.

---

## 1. Motivation

Khilkevich & Lohse Fig 5 e–h shows that, **across brain areas**, the *duration*
of a region's TF response predicts how early that region is recruited into
pre-lick **preparatory activity** — regions with longer (more sustained) TF
responses ramp toward the lick earlier. Their panel h formalises it: median
TF-pulse response half-peak width (per region) correlates negatively with the
onset of preparatory activity (r = −0.55).

This project has independently established, within striatum, a **transient →
sustained continuum of TF-response duration** (GLM kernel width `interp_fwhm`;
a skewed unimodal spectrum, adversarially verified; orthogonal to FSI/SPN
waveform type) and shown that wider cells couple more strongly to change and to
licks (the pre-lick FA ramp rises with width, ρ = +0.34;
`docs/science/2026-07-07-transient-sustained-spectrum-celltype.md`).

**The port:** run Khilkevich & Lohse's exact Fig 5 e–h machinery but replace
their **brain-area** grouping with the **cell-class / width** axis
{transient, sustained} + **non-TF** as the separate reference population. This
is the single-region, single-cell analog of their across-region result: *do the
more-sustained TF cells lead striatal preparatory activity toward the lick, with
non-TF cells recruited last?*

Reference: `paper_references/Brain-wide dynamics linking sensation to action
during decision-making.pdf` — Fig 5 legend p.7; Methods "Preparatory activity
before the lick onset" and "Estimation of peak time and width of TF pulse evoked
activity" pp.17–18.

---

## 2. The Khilkevich & Lohse method, verbatim (what we copy)

All quotes/parameters below are transcribed from the paper's Methods (pp.17–18)
and are copied exactly except where §4 notes a project-specific substitution.

**z-scoring (shared baseline for all of Fig 5).** "To study … hit lick-aligned
(Fig. 5) activity, we computed z-score of mean PETH for each unit. z-Scoring was
done using the mean and s.d. estimated from activity during **2 s before the
change onset**." → per unit: baseline mean `μ_bl` and s.d. `σ_bl` from the
[−2, 0] s pre-**change**-onset window; the hit-lick-aligned **mean** PETH is
z-scored with these. The same `μ_bl, σ_bl` are reused across alignments (this is
the shared-baseline rule; avoids the circular per-condition baseline).

**Panel e — fraction of significantly active units.** "for each brain region the
fraction of significantly active units within a group (that is, TF-responsive)
was measured by calculating **at every time point a fraction of units with the
absolute value of z-score larger than the significance threshold of 2.576
(corresponding to P < 0.01)**. Additionally, we subtracted the 'baseline' level
of activity calculated within **[−2, −1.8] s before hit-lick onset** …". So:
- per unit, per time bin: `active(t) = |z(t)| > 2.576` (captures excitation AND
  suppression via the absolute value; this is the z of the **mean** PETH, not a
  per-trial test);
- group fraction-active(t) = mean of `active(t)` over units in the group;
- subtract the baseline fraction measured in [−2, −1.8] s before hit-lick →
  "fraction of active units **above baseline level**".

**CIs (panels e/f/g).** "The confidence intervals were estimated by
**bootstrapping with replacement (5,000 times) across TF-responsive (or TF
non-responsive) neurons** and repeating the estimation of fraction of
significantly active neurons for each sample of neurons." → bootstrap is over
**neurons**, 5,000×, percentile CI. (No trial-level resampling anywhere in
e–h.)

**Onset / "latency of activation" (panel f/g row-sort AND panel h x-axis).** "The
latency of activation … was defined as the **earliest time point following which
within a 100-ms window for at least 80 ms: (1) the lower 95% confidence interval
of fraction of active units was above zero; and (2) the mean fraction of active
units was above 0.1**." A single **population-level** latency per group.

**Panel h y-axis — TF-pulse response half-peak width.** "for the baseline
subtracted mean response to a fast TF pulse, we calculated its peak time, as the
**time of the largest absolute change in firing rate within 1 s from the pulse
onset**, and a corresponding **half-peak width**." Model-free (from the pulse
PETH). Fast pulse = TF fluctuation > 1 s.d. of baseline (log2) above mean
(TF > 1.19 Hz); slow = TF < 0.84 Hz. For population means, responses of
suppression-type cells are sign-flipped; pulse z-score baseline = 0.5 s
preceding pulse onset. Pulse outlier-event exclusion criteria (p.18): ≥1 s after
baseline onset; earlier than 2 s + window before motion onset on early-lick/abort
trials; change period excluded — to remove baseline-onset/movement/preparatory
contamination.

**Panel h correlation.** Pearson correlation between per-region onset and
per-region median half-peak width; the paper reports r = −0.55, P = 0.0015.

---

## 3. Data sources (verified against live caches 2026-07-20)

| Source | What it gives | Status |
|---|---|---|
| `data/cache/tf_responsive/{bg046,bg039,bg031}_tf_responsive.csv` | per-unit `resp_log2` (TF-responsive vs non-TF), `kernel_fwhm`, `session_date`, region | reuse |
| `data/cache/tf_glm_bg046/kernel_width_continuous.csv` | 520 responsive cells: `interp_fwhm` (canonical width), grid/temporal_spread, coupling | reuse |
| `data/cache/tf_glm_bg046/pulse_fwhm_allpulses.csv` | model-free fast-pulse half-width per responsive cell (0.75 s window) | reuse / recompute with 1 s window (§4) |
| `data/pkls/<subj>/<session>.pkl` (Session objects) | spike times per unit; NI events; trials | recompute source (local only, never over X:) |
| `visdetect.analysis.align.get_event_times(s,'Hit'\|'FA')` + `align_spikes_to_events` | hit-lick / FA-lick onset alignment | reuse (verified: Hit → `t_change + RT − 0.2 s`) |

Population (good_dates cohort = QC-pass AND <50% Disengaged sessions): **520
TF-responsive** cell-sessions (BG_031 VMS 319, BG_046 DMS 162, BG_039 DMS 39;
transient 315 / intermediate 106 / sustained 99 by the grid cut) and **11,078
non-TF** cell-sessions (BG_046 5,278 / BG_031 4,937 / BG_039 863).

**Why a recompute is still needed:** the cached PETHs (`peth_traces_all.npz`,
`peth_traces_unresponsive.npz`) (a) have **no hit-lick alignment** (only Change_ON
and FA-lick), and (b) are z-scored to a **per-event** window, not the paper's
**2 s pre-change** baseline. The recompute produces per-unit **mean** hit-lick
and FA-lick PETHs z-scored to the pre-change baseline. It does **not** need
per-trial tensors (the paper's significance is `|z of the mean PETH|`), so it is
a modest extension of `rebuild_peth_traces_all.py`.

---

## 4. Project-specific substitutions (region → cell class)

1. **Grouping axis.** Brain region → cell class. Panel e = 3 lines
   {transient, sustained, non-TF}. Panels f/h "population unit" = **width decile**
   of the TF-responsive cells (equal-count `interp_fwhm` quantiles via
   `continuum_common.width_bin_assign`), which honors the *spectrum* result and
   restores the statistical resolution that 2–3 discrete classes would lose.
   Class labels (transient/sustained) use the established grid cut
   (`kernel_fwhm ≤ 0.05` / `≥ 0.15 s`) for the panel-e lines.
2. **Regions available.** Only DMS (BG_046+BG_039) and VMS (BG_031). Panel e has
   2 region subpanels + pooled. Every panel carries the mandatory **DMS-vs-VMS**
   breakdown (region-confound HARD RULE; VMS dominates 319/520).
3. **Panel h unit of observation.** Faithful primary = **per-width-decile** dots
   (paper-exact onset machinery, x = decile onset, y = decile median width) — the
   direct region→decile analog. Plus a **per-cell** scatter (x = per-cell onset,
   y = per-cell width, coloured by class) as the higher-n within-region view the
   user endorsed. Per-cell onset is an **extension beyond the paper** (they define
   onset only at the population level): per-cell onset = earliest pre-lick t with
   `|z(t)| > 2.576` sustained ≥80 ms within a 100-ms window.
4. **Panel h width metric.** Show BOTH: (a) paper-exact **model-free pulse
   half-peak width** (recompute with the paper's 1 s window; the cached
   `pulse_fwhm_all` used 0.75 s which truncates wide kernels), and (b) the
   project-canonical GLM **`interp_fwhm`**. This project documented that the two
   diverge within striatum (Spearman 0.11) — showing both is itself the
   within-region analog of the paper's Extended-Data-Fig-5 pulse-vs-kernel
   cross-check. Headline uses `interp_fwhm` (consistent with the class axis);
   pulse width is the faithful corroborator.
5. **Non-TF cells** have no width by construction: they appear in panel e (a
   line), panel g (their own heatmap/reference), and as an onset-only marginal in
   panel h — never on the width axis.
6. **Bins/smoothing.** Paper Fig 5 does not specify; use project-canonical
   25 ms bins, 25 ms σ Gaussian (`DEFAULT_BIN_SIZE`, `DEFAULT_SIGMA_MS`) — note
   their Fig 6 used 10 ms / 30 ms.
7. **Correlation stat (h).** Report **Pearson** (to match the paper) and
   **Spearman** (project standard for neural data); bootstrap CI over the unit of
   observation.

---

## 5. Architecture

Two compute-separated stages + a verification stage, under a new topic dir
`scripts/tf_responsiveness/preparatory_fig5/`, importing `visdetect.*`, writing
to canonical `FIGURES/preparatory_fig5/` and `data/cache/preparatory_fig5/`.

### Stage 1 — recompute cache (`build_prep_cache.py`)
Local ProcessPool, BLAS pinned 1/worker, **never over X:** (compute rule). For
each good_dates session × subject, for **every** unit (TF-responsive and non-TF,
tagged from the registry):
- baseline `μ_bl, σ_bl` from [−2, 0] s pre-change-onset binned rate (guard
  `σ_bl < 1e-6 → 1.0`);
- mean PETH aligned to **hit-lick** (`get_event_times(s,'Hit')`, outcome=hit) and
  to **FA-lick** (`get_event_times(s,'FA')`, outcome=fa), window ≈ [−2, 1.5] s,
  25 ms bins, 25 ms σ; require ≥10 lick events (else unit dropped, logged);
  optionally require hit-lick ≥0.4 s from change onset (paper Fig 6 rule);
- z-trace `z(t) = (meanPETH(t) − μ_bl)/σ_bl`; store `active(t) = |z| > 2.576`.
- join `interp_fwhm`, grid `kernel_fwhm`, class label, region, subject.
Output: `data/cache/preparatory_fig5/<subj>_<lick>.npz` with per-unit
`z`, `active`, `onset_cell`, `class`, `interp_fwhm`, `pulse_fwhm_1s`, region.
Plus a model-free pulse half-peak-width recompute (1 s window, paper recipe) →
`pulse_fwhm_1s` per responsive cell.

**Stage-1 validation gates (must pass or `SystemExit`).** (a) recomputed grid
`kernel_fwhm` reproduces registry `kernel_fwhm` for ≥95% of responsive cells
(the `recompute_kernel_width` gate); (b) responsive/non-TF membership and
per-subject/region counts match the registry exactly (520 resp; 11,078 non-TF;
BG_031 319 / BG_046 162 / BG_039 39 resp); (c) where the new hit-/FA-lick mean
traces overlap the existing `peth_traces_*` windows, they agree up to the
baseline-window change; (d) every unit/session drop (the ≥10-event and
≥0.4 s-from-change rules) is **logged with counts, never silent** (no silent
truncation — HARD RULE).

### Stage 2 — figures (cache-only, fast; one script per panel or one module)
- `fig5e_fraction_active.py`, `fig5fg_onset_heatmaps.py`, `fig5h_onset_vs_width.py`
  (or a single `fig5eh.py` with helpers). All read the Stage-1 cache only.
- Shared helper `prep_common.py`: `fraction_active_trace(units, baseline_window)`,
  `bootstrap_fraction_ci(units, n=5000)`, `population_onset(frac, ci_lo,
  win=0.1, sustain=0.08, min_frac=0.1)`, `cell_onset(z, thresh=2.576, …)`.
- **Stage-2 validation.** Every `prep_common` primitive is unit-tested against a
  closed-form synthetic (§8) BEFORE it touches real data; the panel scripts assert
  join coverage on the `canonical_session_id` key and n per group, and refuse to
  plot a group below a documented minimum n (fail loud, not silently thin).

### Stage 3 — adversarial verification (trust nothing; before ANY claim leaves the repo)

Nothing about the headline (panel h: sustained → earlier preparatory onset) is
believed until it survives this battery. Runs via the `harden-result` skill.

**(A) Built-in null controls — the result must DIE under the null.** For every
headline (panel-e ordering, panel-f onset gradient, panel-h slope): (1) **shuffle
class/width labels** across cells → the onset-vs-width gradient must flatten to
chance; (2) **circularly shift / shuffle lick times** per session → the fraction-
active ramp must collapse to the [−2,−1.8] s baseline level. If a "result"
survives the null, it is a **BUG REPORT, not a finding** (circular-analysis HARD
RULE). Report the null distribution beside every headline number.

**(B) Confound battery (each, per-region DMS-vs-VMS + per-subject).** (1)
**Firing-rate** — the panel-e statistic is a binary `|z|>2.576` fraction so it is
already largely FR-robust, but verify the width→onset relationship holds within
`base_hz` quartiles and is not a yield/FR proxy. (2) **Lick leakage** — prove the
onset is genuinely **pre-lick**: recompute with the lick-execution window censored
and confirm sustained cells still lead; check onset is not driven by faster-RT
subpopulations. (3) **Lick-responsiveness overlap** — sustained ≈ lick cells
(OR 4.2); recompute the width→onset relationship **conditioning on / stratified by**
lick-responsiveness so the gradient is not merely "lick cells ramp first". (4)
**Pseudoreplication** — onset~width via `mixedlm` with session (and subject)
random intercepts (`hardening_pseudoreplication` pattern); per-session sign test
as a replication-unit cross-check. (5) **Circularity provenance** — width is
estimated from the **TF-pulse GLM**, fully independent of lick alignment, so
onset-vs-width is **not** circular; state this explicitly and show that binning by
an independent width metric (`pulse_fwhm_1s`) gives the same ordering. (6)
**Region confound** — the effect must hold in DMS and VMS **separately**, not just
pooled (VMS dominates 319/520).

**(C) Independent re-derivation.** A separate agent recomputes the panel-h slope
(and the panel-e onset ordering) by an independent code path from the Stage-1
cache — different bootstrap seed, different onset implementation — and must land on
the same numbers within CI. Divergence blocks the claim.

**(D) Adversarial refutation pass (Opus 4.8 skeptics, `harden-result`).** A
Workflow of ≥6 independent Opus-4.8 refuters, each assigned a distinct lens
(FR confound, lick leakage, circularity, pseudoreplication, region confound,
yield/selection bias, null-shuffle adequacy), each **prompted to refute** the
headline and defaulting to "refuted" under uncertainty, re-deriving from the
cache. Majority-refute ⇒ the claim is killed or downgraded. Record every lens's
verdict and the mandatory caveats it raises (the
`tf_spectrum_celltype_orthogonality` doc is the template: "0/6 refuted, all high
confidence" is the bar).

**(E) Yield / interpretation caveats carried into any figure/talk.** FSI-yield
bias (do not read fractions as biology); cell-sessions not tracked (session/subject
REs address session clustering only); BG_039 thin; width is a spectrum not two
kinds. A flat panel-h is reported **as-is** (striatum need not mirror the
brain-wide result) — never massaged toward the paper's sign.

---

## 6. Panel specifications

**e — Fraction of significantly active units vs time from lick.** x = time from
hit-lick onset (≈ −2 to +1 s). y = fraction-active-above-baseline. Three lines:
transient / sustained / non-TF (project palette, to be added). Bootstrap-over-
neurons 95% CI shading (5,000×). Subpanels: DMS, VMS, pooled. Also render the
FA-lick version. Expected: sustained rise earliest, transient later, non-TF last.

**f — Onset heatmap, TF-responsive.** rows = width deciles (transient→sustained),
cols = time from hit-lick, colour = fraction active above baseline; rows sorted by
population onset; onset line overlaid (paper's black line). DMS/VMS faceted.

**g — Onset heatmap, non-TF.** same machinery for the non-TF population (rows =
sessions or non-TF cells ordered by their own onset), the reference showing no
width-ordered recruitment wave.

**h — Onset vs TF-response width.** (primary, faithful) per-width-decile scatter:
x = decile onset, y = decile median width; (supplement) per-cell scatter coloured
by class + `continuum_common.binned_trend`. Both `interp_fwhm` and
`pulse_fwhm_1s` on the y-axis (interp headline). Pearson + Spearman + bootstrap CI.
Per-region. Expected: negative slope (wider/sustained → earlier onset), the
within-striatum analog of the paper's r = −0.55.

---

## 7. Outputs
- `FIGURES/preparatory_fig5/{pooled,DMS,VMS}/fig5{e,f,g,h}_{hit,fa}.{png,pdf}`
- `_stats.csv` per panel (fractions, onsets, correlations, CIs, n).
- `data/cache/preparatory_fig5/<subj>_<lick>.npz` (+ a small combined table).
- Results write-up under `docs/science/` (via `research-notes-summarizer`).

## 8. Testing
Unit tests for `prep_common`: synthetic unit with a known ramp → correct
`cell_onset`; flat unit → no onset; synthetic population with a known fraction →
`fraction_active_trace` recovers it; `population_onset` respects the 100 ms/80 ms/
mean>0.1 rule on a synthetic fraction trace. Reuse `utils.synthetic` where
possible. A validation gate that the recompute reproduces the cached mean traces
where windows overlap.

## 9. Caveats (carry into any figure/talk)
- Width is a **spectrum**, not two classes — deciles/continuum are primary; the
  transient/sustained lines are a descriptive cut (state it).
- FSI-yield bias (FSI:SPN ≈ 391:100) — do not read population fractions as biology.
- Pseudoreplication: cell-sessions are not tracked across sessions; session/subject
  random effects address session clustering only.
- VMS-dominated pool; BG_039 thin (39 responsive, 863 non-TF) — per-region and
  per-subject breakdowns mandatory.
- "Preparatory" here is pre-**hit-lick** (headline) and pre-**FA-lick** (secondary,
  a different, impulsive construct) — label them distinctly.

## 10. Open choices (defaults chosen; flag in review if you disagree)
- Panel-h headline width metric = `interp_fwhm` (vs paper's model-free pulse
  width, shown alongside).
- Panel-g row unit = non-TF cells ordered by own onset (vs per-session rows).
- Bins/σ = 25 ms / 25 ms (vs Fig 6's 10 ms / 30 ms).
- Hit-lick ≥0.4 s-from-change inclusion rule = ON (paper Fig 6 rule; light).

## 11. Reproduce (once built)
```bash
py scripts/tf_responsiveness/preparatory_fig5/build_prep_cache.py   # Stage 1, LOCAL ProcessPool
py scripts/tf_responsiveness/preparatory_fig5/fig5e_fraction_active.py
py scripts/tf_responsiveness/preparatory_fig5/fig5fg_onset_heatmaps.py
py scripts/tf_responsiveness/preparatory_fig5/fig5h_onset_vs_width.py
```
