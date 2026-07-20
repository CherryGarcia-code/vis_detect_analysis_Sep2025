# Preparatory activity by TF-cell class: a within-striatum port of Khilkevich & Lohse Fig 5 e–h

**One-line honest summary.** Recreating Khilkevich & Lohse (Nature 2024) Fig 5 e–h
*within striatum* — grouping by TF-kernel-width cell class {transient, sustained,
non-TF} instead of brain area — reproduces the **recruitment-order** structure
(sustained cells carry the earliest/largest pre-lick ramp, non-TF the latest/smallest;
robust, replicates in all 3 mice) but **only weakly reproduces the paper's
onset-scales-with-width result** (defensible as a single-mouse, DMS-only, per-cell
Spearman ρ=−0.41 — *weaker* than the paper's −0.55, not stronger). Two subjects
DMS (BG_046, BG_039), one VMS (BG_031); 520 TF-responsive + 11,078 non-TF
cell-sessions, QC-pass & <50 %-Disengaged.

**Status.** Built + adversarially verified 2026-07-20 (Stage-3 hardening battery
+ a 6-lens Opus-4.8 refutation). The caveats in §5 are mandatory framing, not
optional polish. Branch `feature/fig5eh-preparatory-cellclass`.

---

## 1. What was built

Faithful port of the paper's Fig 5 e–h machinery (Methods pp.17–18), replacing
the brain-area grouping with the project's transient→sustained width axis + non-TF
reference:

- **Panel e** — fraction of significantly active units (`|z of the mean lick-PETH|
  > 2.576`, P<0.01) vs time from lick, per class, per region; the z-baseline is the
  paper's **2 s pre-change** window (per unit), CIs bootstrap **over neurons** (5,000×).
- **Panels f/g** — width-decile onset heatmap (TF-responsive) and the non-TF
  reference, rows sorted by the paper's population activation onset (earliest t with,
  within 100 ms for ≥80 ms, lower-95 %-CI>0 AND mean-fraction>0.1).
- **Panel h** — onset vs kernel width: per-width-decile (faithful) + per-cell.
- Both the **hit** (decision) and **fa** (impulsive) lick, all 3 mice, per-region.

Code: `scripts/tf_responsiveness/preparatory_fig5/` (pure primitives in
`src/visdetect/analysis/preparatory.py`, unit-tested). Caches
`data/cache/preparatory_fig5/prep_{hit,fa}.npz`. Figures + stats + hardening under
`FIGURES/preparatory_fig5/`.

**A methods pitfall that mattered.** The z-baseline **σ must be the SD of the
trial-averaged baseline trace**, not pooled single-trial bins — pooling inflates σ
by ~√(n_trials) of Poisson noise and collapses the fraction-active to ~0.08
(post-lick), unlike the paper's 0.5–0.9. This was caught by the validation gate and
fixed (commit `ba241c5`); an adversarial lens independently confirmed the corrected
convention is not itself the driver (the ordering survives SEM/fixed-1Hz/Poisson σ,
and the divisor is if anything *largest* for sustained cells).

---

## 2. Result — the two claims and their verdicts

### CLAIM 1 — recruitment order: sustained < transient < non-TF. **ROBUST (reframed).**

Per-class population onset (s from hit-lick), pooled: **sustained −0.738 < transient
−0.613 < non-TF −0.338**; monotonic in DMS (−0.688/−0.563/−0.363) and VMS
(−0.888/−0.713/−0.338); replicates **independently in all 3 mice**. Peak
active-fraction sustained 0.88 > transient 0.64 > non-TF 0.50.

Survives: the **label-shuffle null** in all 3 regions (p=0.001/0.004/0.011, holds
under Bonferroni); the **lick-time-shuffle** (re-aligning to random times collapses
the ramp from 0.88→0.06 → genuinely lick-locked); **pre-lick-only** recompute
(ordering identical); **firing-rate** controls (base_hz→onset ρ=−0.05 ns;
rate-matching non-TF closes only ~0.05 s of the 0.35 s gap; subsampling non-TF to
n=99 never beats sustained); and **alternative σ conventions**.

**Reframing (mandatory):** this is fundamentally an ordering of preparatory-response
**magnitude/reliability**, not a proven *latency* ordering. Sustained cells win
because they carry the largest, most reliable pre-lick ramps (~23 vs ~6 Hz),
read through a noise-relative z-threshold and an absolute fraction threshold. Under
amplitude (peak-relative) normalization the sustained-vs-transient split disappears
in VMS; the robust core is **both TF classes earlier/stronger than non-TF**
everywhere, plus the full three-way ordering in DMS/pooled.

### CLAIM 2 — onset scales with kernel width (wider = earlier). **WEAK.**

The defensible statistic is the **per-cell DMS Spearman ρ=−0.41 (p=4×10⁻⁹)** — the
**only** one of 8 region×lick×metric combos that survives FDR correction (also
survives Bonferroni and the session-RE mixed model). It holds within *both*
lick-responsive (ρ=−0.33) and non-lick-responsive (ρ=−0.42) cells, so it is not a
lick-cell confound.

But it does **not** reproduce the paper as first appeared:
- The **n=10 decile r=−0.66** (which superficially "beats" the paper's −0.55)
  **fails FDR** (p_adj=0.18), sits at the onset-shuffle noise floor (95th |r|=0.63
  vs observed 0.64), and is **apples-to-oranges** (across width-deciles within one
  region vs the paper's across-area correlation). Do **not** headline it. The honest
  per-cell magnitude (−0.41) is *weaker* than the paper's −0.55.
- It is **carried by a single mouse**: leave-one-out dropping BG_046 makes the
  pooled per-cell correlation non-significant (ρ=−0.067, p=0.22). BG_039 corroborates
  the sign/rank but is underpowered (n=35); VMS is null (ρ=−0.03) and is a lone mouse.
- It **collapses when restricted to strictly pre-lick** (DMS onset<−0.2 s ρ=−0.16,
  p=0.09 ns; pre-lick-only pooled Pearson +0.009) and is **absent on the FA lick**.

### Hit vs FA (user-requested comparison).
Hit onsets are earlier than FA onsets for every class/region; the gap is **largest
for transient** (~0.15–0.23 s) and **smallest for sustained** (~0.03–0.13 s), and
all classes show a **post-lick HIT>FA divergence** (sustained engagement after a
rewarded detection vs quick disengagement after an impulsive lick). Sustained cells
recruit early regardless of outcome; transient timing is more outcome-dependent.
`FIGURES/preparatory_fig5/hit_vs_fa/`.

---

## 3. Hardening battery (Stage 3) — all controls

`FIGURES/preparatory_fig5/hardening/hardening_report.md`. Label-shuffle null,
width-shuffle null, mixedlm pseudoreplication (session/subject RE + per-session sign
test), pre-lick-only leakage control, lick-responsiveness stratification, independent
re-derivation (onset MAE 0.024 s between two implementations), and the lick-time
shuffle (ramp is lick-locked). Verdicts as in §2.

## 4. Adversarial refutation (6 Opus-4.8 lenses)

`FIGURES/preparatory_fig5/hardening/adversarial_refutation.md`. Lenses: firing-rate/
yield/unequal-n; movement/lick/RT leakage; baseline-σ/circularity; onset-metric/
smoothing; pseudoreplication/single-mouse; statistics/multiple-comparisons. Claim 1
= 5 SURVIVES + 1 PARTIAL; Claim 2 = 2 SURVIVES + 3 PARTIAL. Every mandatory caveat
below comes from this pass.

## 5. Mandatory caveats (carry into any talk/paper)

1. **"Recruited earliest" = "engaged most strongly."** The ordering is a
   magnitude/reliability ordering read through a threshold; an amplitude-normalized
   timing measure does not preserve it (VMS sustained≈transient). Report the
   peak-relative result alongside the absolute-threshold onset.
2. **Claim 2 is a single-mouse, DMS-only, per-cell ρ=−0.41 effect** — weaker than
   the paper's −0.55. Never headline the decile r=−0.66 (fails FDR, noise-floor,
   non-comparable to the paper). Disclose: absent on FA, null in VMS, dropping BG_046
   removes pooled significance.
3. **"Preparatory" must be qualified.** Peak activity sits AT/AFTER the lick
   (+0.01…+0.14 s); the per-cell width→onset gradient concentrates in the peri-lick
   window and collapses strictly pre-lick. Broad-kernel cells have long sensory
   responses *by definition*, which bleed into the pre-lick window via the ~0.4–0.7 s
   RT — so the gradient is partly guaranteed by construction. **Decisive control still
   open:** a change-aligned or video-movement-regressed re-derivation (the project has
   `video_sync`).
4. **Cell-type confound on the non-TF rung.** In BG_046 the TF-responsive preparatory
   cells are ~95 % fast-spiking (127/133 FSI); non-TF is mixed FSI/SPN. The width→onset
   gradient describes order *within fast-spiking interneurons*; responsive-vs-non-TF is
   partly FSI-vs-mixed. No waveform label for the 11k non-TF cells or BG_031/039.
5. **Pseudoreplication.** Cells are untracked across sessions; per-cell N overstates
   independent sampling; session-RE mixedlm does not remove within-subject repeated-neuron
   inflation. VMS = 1 mouse; BG_039 thin (n=39).

## 6. Reproduce
```bash
py scripts/tf_responsiveness/preparatory_fig5/build_prep_cache.py --lick hit   # + --lick fa (LOCAL ProcessPool)
py scripts/tf_responsiveness/preparatory_fig5/fig5e_fraction_active.py --lick hit
py scripts/tf_responsiveness/preparatory_fig5/fig5fg_onset_heatmaps.py --lick hit
py scripts/tf_responsiveness/preparatory_fig5/fig5h_onset_vs_width.py --lick hit
py scripts/tf_responsiveness/preparatory_fig5/fig_hit_vs_fa.py
py scripts/tf_responsiveness/preparatory_fig5/nulls_and_hardening.py --lick hit
py scripts/tf_responsiveness/preparatory_fig5/licktime_shuffle_control.py
```

## 7. Safe wording for a talk
> "Within striatum, TF-responsive neurons that carry longer (sustained) responses to
> temporal-frequency fluctuations also carry the largest and earliest-crossing ramp of
> pre-lick activity, with TF non-responsive cells recruited last — a within-region echo
> of Khilkevich & Lohse's brain-wide result, robust across three mice and lick-locked.
> But two honest limits: the 'earlier' is largely 'stronger' read through a threshold,
> and the continuous onset-scales-with-kernel-width relationship is a single-mouse DMS
> effect (per-cell ρ=−0.41, weaker than their −0.55), absent in our VMS animal and on
> the impulsive lick, and not cleanly separable from the sensory-response duration of
> broad-kernel cells without a movement-regressed control."
```
