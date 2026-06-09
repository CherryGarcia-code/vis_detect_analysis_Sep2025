# Optotagging Striatal D1/D2 SPNs — Protocol & Yield-Improvement Guide

- **Created:** 2026-06-09
- **Context:** Written after the BG_046 antidromic-optotagging re-analysis, which yielded ~162
  *candidate* but only **3 collision-confirmed** D1/D2 units (~90% of short-latency responses
  were synaptic, not antidromic). See results: `docs/results/2026-06-08-optotagging-yield-results.md`.
- **Purpose:** Reference for designing future optotagging experiments to improve *confirmed* yield.

> Parameter values below are typical ranges from the literature; they are opsin/prep/rig
> dependent and should be piloted and titrated, not copied blindly.

---

## 1. Two strategies — and which is "usual"

### Somatic photo-tagging ("PINP") — the common, higher-yield approach
Express a fast opsin in D1 *or* D2 SPNs (Cre line — **Drd1a-Cre** or **Adora2a/Drd2-Cre** —
with a Cre-dependent ChR2 AAV, or the **Ai32** reporter) and place the optic fiber **directly
over the striatum, where you record.** Light hits the somata; opsin-positive cells fire
directly. A unit responding at short latency / low jitter to *local* light = that cell type.
This is the Lima et al. 2009 / Kvitsiani et al. 2013 lineage (SALT was built for it) and is
what most striatal D1/D2 ephys studies use. **Higher yield** — you fire somata directly rather
than relying on a backpropagating spike surviving a thin axon.

### Antidromic tagging — rarer, lower-yield, but anatomically specific
Fiber over the *projection target* (**SNr** for D1, **GPe** for D2); look for back-propagating
spikes in striatal somata. Unique strengths: the **collision test** and **bridging-collateral
disambiguation** (only D1 reaches SNr). Weaknesses that hurt yield:
- **Antidromic propagation failure** — thin unmyelinated SPN axons, branch points, and a
  high-threshold soma mean many terminal spikes never reach home.
- **Network contamination** — stimulating a whole nucleus drives large trans-synaptic activity
  that mimics responses.

**Bottom line:** for maximum number of labelled cells, somatic PINP usually wins. Antidromic is
the method when you specifically need pathway-anatomical certainty. **Strongest design = both
fibers:** somatic over striatum for yield + terminal over SNr/GPe for a collision-confirmed
gold-standard subset.

---

## 2. Standard stimulation protocols

| Parameter | Typical | Notes |
|---|---|---|
| Pulse width | ~1–5 ms (ChR2) | Shorter (≤2 ms) for timing precision; long enough to reliably fire terminals |
| Power | titrated | Build a **recruitment curve** across several intensities, not one fixed power |
| **Frequency trains** | **1, 5, 10, 20, 50 Hz** | **Key — enables the following test (see §3). Not just single pulses.** |
| N pulses | ≥500 / condition | For statistics |
| Block structure | **interleave** GPe/SNr | Rather than all-of-one-then-the-other → controls for drift |

> BG_046's protocol used **single pulses at ~0.65 Hz** with all-GPe-then-all-SNr blocks — the
> main protocol gaps are the missing frequency trains and the non-interleaved blocks.

---

## 3. Confirmation criteria the field uses

Short latency + low jitter + SALT significance are necessary **but not sufficient** — synaptic
(network) responses can also be short-latency and significant. The discriminating criteria:

- **High-frequency following** — does the unit track each pulse 1:1 at constant latency at high
  rate? Direct/antidromic spikes follow faithfully; synaptic responses fatigue/fail. **Crucially,
  this is independent of the cell's spontaneous firing rate.**
- **Waveform correlation** — the light-evoked spike waveform must match the unit's spontaneous
  waveform (Pearson r typically >0.9). Proves it is the *same* cell, not a synaptically-driven
  neighbour.
- **Collision test** (antidromic only) — a spontaneous somatic spike just before the pulse
  collides with and cancels a true antidromic spike (single-track-railway logic). Requires
  enough spontaneous spikes → underpowered for quiet SPNs (the "untestable" bucket).
- **SALT** — statistical significance of stimulus-locked latency.

---

## 4. Prioritized recommendations for future experiments

Assuming the laser-power and fiber-coordinate upgrades are already done, in rough order of impact:

1. **Add high-frequency pulse trains (e.g., 10, 20, 50 Hz).** *Highest-leverage change.* Enables
   the **frequency-following test**, which (a) is a second antidromic confirmation that **does
   NOT need spontaneous spikes — so it rescues the low-firing SPNs that are collision-untestable**,
   and (b) cleanly separates antidromic (follows) from synaptic (fails at high frequency). Directly
   attacks both reasons the confirmed yield was tiny.
2. **Record & save the light-evoked waveform per unit; add the waveform-correlation criterion.**
   Firing-rate-independent, powerful, standard. (Independent of cell-type label availability.)
3. **Recruitment curve (multiple powers).** More power helps overcome terminal threshold /
   propagation failure — but it **also increases network contamination**, so verify all-or-none
   antidromic threshold behaviour rather than just maximizing power. Watch for tissue heating and
   light artifacts on the probe.
4. **Consider a faster / more potent opsin at the terminals.** ChR2 has slow kinetics and follows
   poorly at high frequency. **Chronos** or **CheRiff** (fast, sensitive) or **ChrimsonR**
   (red-shifted → deeper penetration, less light scatter) improve reliable terminal firing and
   high-frequency following — important for antidromic terminal stimulation.
5. **Interleave GPe/SNr blocks** instead of two long separate blocks → drift control.
6. **(Advanced) Closed-loop "active" collision test.** Detect a spontaneous somatic spike online
   and fire the laser at a fixed delay → definitive collision that doesn't depend on chance
   timing. Harder to implement, but gold-standard.
7. **Strategic: add a somatic fiber over the striatum.** If yield is the priority, likely the
   single biggest win — somatic PINP for the bulk, antidromic+collision for the certainty subset.

> **Key caution:** the laser/coordinate upgrades increase *both* real antidromic activation *and*
> network contamination. The analysis-side discriminators (following, waveform-match, collision)
> therefore become **more** important, not less — without them, a more powerful laser can inflate
> the false-positive (synaptic) count.

---

## 5. What our analysis pipeline supports / what it needs

`src/visdetect/analysis/optotagging.py` (two-tier antidromic pipeline) currently provides:
baseline-corrected excess reliability/jitter, canonical SALT, Poisson excess test, the
**collision test**, and bridging-collateral D1/D2 classification.

To exploit the recommended protocol changes, the pipeline still needs:
- a **frequency-following metric** (requires train data — pulse-train structure must be recorded
  and parsed; will rescue collision-untestable units); and
- a **light-evoked vs spontaneous waveform-correlation** metric (requires per-unit evoked
  waveforms to be saved).

Also outstanding from the BG_046 run (see results note): the `excess_jitter` gate is vacuous as
implemented (measure on a wider window to make it discriminating); session 5092025 was skipped
(pkl `_b` suffix not matched by the loader); and `waveform_celltype_labels.csv` was absent.

---

## 6. References / starting points
- **Lima, Hromádka, Znamenskiy, Zador (2009), PLoS ONE** — PINP / optogenetic photo-tagging method.
- **Kvitsiani et al. (2013), Nature** — SALT (Stimulus-Associated spike Latency Test).
- Antidromic identification + collision logic adapts classic antidromic electrical-stimulation
  criteria (constant latency, high-frequency following, collision) to optogenetic terminal stimulation.
- Striatal D1/D2 work commonly uses Drd1a-Cre / Adora2a(Drd2)-Cre lines with Ai32 or
  Cre-dependent ChR2 AAV.
