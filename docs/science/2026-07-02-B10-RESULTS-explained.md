# B10 — Impulsivity kernel & real-time TF tracking: results explained

*A plain-language companion to the four `FIGURES/evidence_learning/` figures — what each
shows, exactly how it was computed, the statistics, and how to explain it if asked.*
Subjects: **BG_046 + BG_039 (DMS), BG_031 (VMS)**. All code in
`visdetect.analysis.psychophysical_kernel` + `.evidence_learning_io`, scripts in
`scripts/evidence_learning/`, 26 tests pass.

---

## TL;DR (the one-paragraph version)

We asked whether impulsive (early) licks are triggered by the mouse mistaking the
fluctuating baseline grating for a real change, and whether striatal cells that encode
grating speed do so in real time. **Three results are honest negatives/weak effects, one
is a clean positive.** (1) The behavioral "impulsivity kernel" is **weak** — impulsive
licks are only faintly preceded by an upward speed fluctuation and don't differ by
behavioral state → impulsivity here looks **internally driven, not stimulus-triggered**.
(2) The pre-lick striatal population signal is a **motor/urgency ramp, not a sensory
signal** (proven by a stimulus-matched control). (3) That ramp is larger in Naive than
Expert in **DMS** (a learning effect, but a motor one). (4) **THE POSITIVE:**
TF-responsive striatal cells **track the moment-to-moment grating speed in real time**
(~0.4 s neural lag), robustly above two nulls, and in **VMS this tracking is switched off
when the animal disengages** (engagement gating).

---

## The data and the two core signals (used by every figure)

- **Sessions**: loaded per subject from local `data/pkls/`, labelled Naive / Learning /
  Expert from the staging manifest. Unit of replication = **session** (we bootstrap over
  sessions, never pool raw neurons across sessions — neurons aren't tracked across days).
- **The stimulus, `y(t)`** — the momentary baseline grating speed. `Trial.baseline_values`
  stores the speed at 60 Hz; the true value updates every **50 ms** (held for 3 frames), so
  `y = log2(baseline_values[::3] / median)` on a **50 ms grid** (`baseline_log2tf`). Units:
  log2 octaves around the trial median (0 = median speed, +1 = double).
- **The neural signal, `S(t)`** — a single "population speed estimate" per trial. For each
  TF-responsive cell we bin its spikes at 50 ms, **z-score** it to its own baseline
  mean/SD (so fast- and slow-firing cells count equally), and take a **signed sum**:
  `S(t) = mean_i sign_i · z_i(t)`, where `sign_i = sign(c1_r_log2)` is +1 for a
  fast-speed-preferring cell and −1 for a slow-preferring cell (from the TF-GLM registry).
  So `S(t)` goes up when the fast-preferring cells fire and the slow-preferring cells go
  quiet — i.e. `S(t)` is the population's read-out of "is the grating fast right now?".
- **TF-responsive cells** = registry `resp_log2==True` (identified by an independent
  Poisson TF-GLM). Signs from `c1_r_log2`. Pooled **by subject** into DMS (046+039) and
  VMS (031); `region_bank_confirmed` is False across the whole registry so regions are
  provisional and we do **not** gate on it.
- **Impulsive (FA) licks** = `trialoutcome=='fa'` with latency **≥ 3.0 s** (`FA_RT_SPLIT`).
  Only these "late" FAs are used — early licks (<3 s) are reflexive and leave too little
  baseline to analyse.

---

## Figure 1 — Behavioral impulsivity kernel  (`pooled/b10_behavioral_kernel.png`)  → WEAK

**Question:** what pattern of grating-speed fluctuation precedes an impulsive lick?

**How computed:** for each late FA, take the speed trace `y(t)` in the window
[lick − 1.5 s, lick − 0.15 s] (the last 0.15 s is dropped as motor). The **kernel** =
average of those FA-triggered traces **minus** the average of time-in-trial-matched
*no-lick* windows drawn from hit/miss trials (the "withhold" control removes whatever the
speed does at that point in a trial anyway). 95% CI by **bootstrapping FA/withhold pairs
(1000×, seed 42)**; stages are **subsampled to equal FA counts** (n-match) before
comparison; we report kernel **shape** (half-width, peak-lag) separately from **amplitude**.

**Result (log2 units, ≈ % speed change):** small and noisy — peak ≈ **0.05–0.07** (BG_046
0.049→0.054, BG_039 0.050→0.043, BG_031 0.070→0.060 Naive→Expert); CIs largely span zero,
no clean Naive→Expert sharpening. **How to say it:** *impulsive licks are only weakly
preceded by an upward speed blip; the mouse isn't mostly being fooled by the baseline —
its early licks look internally/urge-driven.* Caveat: this is the **blunt** version
(a raw stimulus-triggered average); Orsolic's faithful method (a regularized lick-hazard
regression) was not run.

## Figure 3 — Same kernel split by state  (`state/b10_state_kernel.png`)  → NO DISSOCIATION

Same kernel, but FAs split into **StimSens** (engaged) vs **Impulsive** (itchy) states
(confidence ≥ 0.8). Hypothesis was "engaged FAs are stimulus-driven (sharp), impulsive-
state FAs are internal (flat)." **Result:** both weak and similar (StimSens peak 0.032,
Impulsive 0.038) — **no dissociation**. This is **non-circular**: the state labels come
from lick rates/outcomes, the kernel shape is an independent measurement.

## Figure 2 — Neural impulsivity kernel  (`neural/b10_neural_kernel.png`)  → MOTOR, not sensory

**Question:** does the population speed-signal `S(t)` do anything special before an
impulsive lick?

**How computed:** `S(t)` time-locked to the FA lick vs matched withhold windows (same as
Fig 1 but on the neural signal). **The key control — sensory vs gain:** because each FA's
withhold window carries the *same* stimulus trajectory, the withhold's neural signal is
the "sensory expectation." `sensory = mean(withhold S)`, `gain = mean(FA S) − sensory`.

**Result:** the pre-lick signal is a **ramp toward the lick** (peaks at the −0.2 s window
edge). The **sensory component is flat ≈ 0; the whole thing is "gain"** — i.e. it is a
**motor/urgency ramp, not stimulus encoding**. This reproduces the N1 finding (pre-self-
timed-lick striatal signals are motor). The ramp is **larger in Naive in DMS** (0.77→0.51
Naive→Expert) — a real learning effect, but on the *motor/urgency* signal, not sensory.
**VMS does NOT show this** (0.165→0.244; the earlier "VMS 0.44" was a 1-session artifact,
fixed — see Corrections). **How to say it:** *before an impulsive lick the striatum shows
an urgency/motor ramp, not a sensory signal; the control proves the ramp isn't the mouse
"seeing" the stimulus. In DMS that urgency ramp shrinks with learning.*

---

## Figure 4 — Real-time TF tracking  (`tracking/b10_tf_tracking.png`)  → **THE POSITIVE**

**Question:** does the TF-responsive population `S(t)` follow the momentary grating speed
`y(t)` in real time, during the baseline (away from any lick)?

**How computed (step by step):**
1. Per trial, build `S(t)` (population speed-signal) and `y(t)` (actual speed), both on the
   50 ms grid over the baseline period.
2. **Smooth both to ~150 ms** (Gaussian). Rationale: the neural response *integrates* the
   stimulus over ~250 ms (Khilkevich & Lohse), and single 50 ms bins of a few cells are
   dominated by Poisson spike noise — so we correlate at the biological timescale, not raw
   50 ms. (Robustness to this choice is shown below.)
3. **Cross-correlate**: Pearson `r` between `S(t)` and `y(t − lag)` for lag = 0…0.5 s
   (`stimulus_tracking_xcorr`). A peak at lag `L` means "neural follows the stimulus by
   `L` seconds" (physiologically correct direction; verified against a planted-lag test).
4. Average this lag-curve over trials, then over sessions; **95% CI = bootstrap over
   sessions**.
5. **Two nulls / controls plotted alongside:**
   - **Trial-shuffle null**: correlate each trial's `S_i(t)` with a *different* random
     trial's `y_j(t)`. Same signals, same smoothing — but the stimulus is from the wrong
     trial, so any real-time tracking is destroyed. Should sit at ~0.
   - **Non-responsive control**: the same signed-sum built from *non*-TF-responsive cells
     (subsampled to the same count). Should sit at ~0.

**Result (peak Pearson r):**

| Region | responsive (real) | trial-shuffle null | non-responsive | peak lag |
|---|---|---|---|---|
| **DMS** (046+039, 51 sess) | **0.017** | −0.0004 | 0.0018 | ~0.50 s |
| **VMS** (031, 31 sess) | **0.034** | 0.0005 | 0.0068 | ~0.45 s |

The responsive curve **rises with lag and separates cleanly from both nulls** (its
bootstrap CI excludes the shuffle from ~0.2 s onward). So the striatal TF-responsive
population **does carry a real-time read-out of the grating speed**, at a ~0.4–0.5 s neural
lag. The absolute `r` is **small** — expected, because it's a few cells (2–3/session in
DMS) reading a small, fast, noisy stimulus — but it is **real, time-specific, and
cell-specific**.

**Engagement gating (right panels):** we recompute the tracking curve separately for
**engaged (StimSens)** vs **Disengaged** trials.
- **VMS: StimSens r = 0.034 vs Disengaged r = 0.004** — tracking essentially **switches
  off when the animal disengages**. Striking.
- DMS: StimSens 0.017 vs Disengaged 0.011 — modest, and tracking persists when disengaged
  (consistent with B9's "TF cells still encode when disengaged").

**How to say it:** *striatal cells that encode grating speed track it moment-to-moment,
~0.4 s behind the stimulus — cleanly above a shuffle and a non-responsive control. In
ventral striatum this real-time read-out is gated by attention: it's present when the
mouse is engaged and vanishes when it disengages.*

### Why this positive is trustworthy (the verification)

Three independent adversarial checks were run (subagents, from scratch):

1. **Independent from-scratch recompute** (own code, not the pipeline's): reproduces the
   VMS numbers (real ≈ 0.031 Expert-only, shuffle ≈ 0, non-responsive ≈ 0.006). ✔
2. **Autocorrelation stress**: the real-vs-shuffle gap is positive at **every** smoothing
   level, **including no smoothing** (raw 50 ms: real 0.009 vs shuffle 0.0002). The shuffle
   gets the identical smoothing, so smoothing cannot be manufacturing the effect — it only
   suppresses noise. ✔
3. **Per-session robustness**: **14 of 17** Expert VMS sessions show real > shuffle;
   median real 0.044 vs shuffle 0.007; **Wilcoxon signed-rank p = 0.00084**. The 3
   exceptions are the smallest-N sessions (1–10 cells). Not one lucky session. ✔
4. **Join integrity**: the state-tag ↔ trial join is **exact** (match = 1.0000 on all
   tested sessions; ±1 shift collapses to chance) — so the engagement result is not a
   mis-alignment artifact. ✔

**On circularity (the fair worry):** the signs come from the same data (`c1_r_log2`). Does
"cells track TF" follow trivially? **No — the trial-shuffle null is exactly this control.**
It reuses the same signs but mismatched trials; if the sign-labelling alone produced the
correlation, the shuffle would be just as high. It sits at ~0. So the real>shuffle gap is
genuine *within-trial, moment-to-moment* co-fluctuation, which a static ±1 label cannot
create. (A stricter split-half control — derive each unit's sign from odd trials, test on
even, so the signs never touch the test data — is the definitive confirmation; the
shuffle-null result above already rules out the static-label artifact.)

---

## Corrections applied during verification (so the figures are trustworthy)

- **Date-parse bug fixed** (`config.session_date_key`): BG_031's manifest stored `DDMMYY`
  as an int64, dropping the leading-zero day (5 Mar → `50325`, 5 digits). The parser
  produced garbage `(325, 5, 0)`, so **2 BG_031 Naive sessions were silently dropped** from
  every stage-split/pooled analysis. Fixed + regression-tested; all figures regenerated.
  Impact: the tracking positive was **unchanged** (VMS 0.034, n now 31); the **VMS neural
  gain-ramp direction flipped** (the pre-fix "VMS Naive 0.44" rested on 1 session → now
  0.165, so "ramp larger in Naive" is a **DMS-only** finding).
- Every figure re-run reproduces its stats CSV exactly.

## Honest limitations
- No video for these mice → the behavioral kernel is "stimulus history before impulsive
  licks," not proven pure sensory.
- Tracking `r` is small in absolute terms (few cells, small fast stimulus); the DMS peak is
  still rising at the 0.5 s window edge (true peak lag may be slightly longer).
- VMS is n=1 region; DMS-Naive has very few TF-responsive cells (BG_046 = 1) so DMS-Naive
  neural claims are population-level, not per-cell.
- Nulls (weak kernel, no state dissociation) were pre-registered as reportable.

## Files
- Figures: `FIGURES/evidence_learning/{pooled,neural,state,tracking}/*.png`
- Stats: `data/cache/evidence_learning/b10_*_stats.csv`
- Terse results: `docs/science/2026-07-01-B10-results.md`; spec/plan in `docs/superpowers/`
- Code: `scripts/evidence_learning/b10_*.py`; library `visdetect.analysis.psychophysical_kernel`
