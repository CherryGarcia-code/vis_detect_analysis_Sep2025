# Optotagging Yield — BG_046 Results (antidromic D1/D2 redesign)

- **Run date:** 2026-06-09
- **Spec:** `docs/superpowers/specs/2026-06-08-optotagging-yield-design.md`
- **Plan:** `docs/superpowers/plans/2026-06-08-optotagging-yield.md`
- **Pipeline:** `analysis_suite/09_optotagging/a_optotagging_identification.py --force --n-workers 4`
- **Caches:** `cache/optotagging_results.csv` (per unit×fiber), `cache/optotagging_unit_tags.csv` (per unit)
- **Figures:** `figures/09_optotagging/fig43a_optotagging_distributions.png`, `fig43b_yield_by_stage_tier.png`, `fig43c_old_vs_new_and_sweep.png`

## Session funnel (answers "why so few sessions")

- 46 BG_046 `.pkl` files on disk.
- 28 in the QC-passed staging manifest (`load_staging_manifest(qc_only=True)`) — the only sessions the script iterates.
- **22** of those 28 have the post-task laser protocol recorded (clean ~501-pulse GPe + ~501-pulse SNr blocks).
- 5 manifest sessions had **no laser data** (27062025, 25072025, 27082025, 28082025, 29082025); 1 was **skipped** (5092025 — see Data issues).

Candidate pool = pkl-resident `good_and_stable` units (the only units with spike data in the pkls), **8,138 unit×fiber rows**.

## Yield (best achievable), two tiers

| Tier | D1 (SNr→direct) | D2 (GPe→indirect) | Total units |
|---|---|---|---|
| **Candidate** (max sensitivity: SALT<0.05 OR Poisson<0.01, latency 1–10 ms, excess-reliability>0.02) | 66 | 96 | **162** |
| **High-confidence** (candidate + SALT<0.01 + excess-jitter<1 ms + collision PASS + SPN-plausible waveform) | 3 | 0 | **3** |
| *Old pipeline, re-derived on this cache* (SALT<0.01 & lat<8 & jit<3.5 & rel≥0.1, per unit×fiber) | 20 | 71 | 91 |

**Best defensible (collision-confirmed antidromic) yield: 3 units. Best exploratory yield: 162 putative units.**

## The decisive finding: the collision test

Collision status across all 8,138 unit×fiber: **pass 103, fail 5,607, untestable 2,428**.

Among the **186 candidate-or-better** fiber-responses:
- collision **PASS: 12**, **FAIL: 167**, **untestable: 7**
- adding strict SALT (<0.01): **5 fiber-rows** (3 SNr, 2 GPe) → **3 units** after bridging aggregation.

**~90% of short-latency, SALT/Poisson-significant responses FAIL the collision test** — i.e. they are **synaptic (orthodromic), not antidromic**. Stimulating D1/D2 terminal fields in GPe/SNr drives mostly *trans-synaptic* striatal activity rather than back-propagating somatic spikes. This is the scientifically correct, conservative result and explains the gap between the permissive count (162) and the gold-standard count (3). It is consistent with the proposal-progress note of "mixed optotagging yield."

## Plain-language primer: antidromic vs synaptic, and the collision test

**The setup.** D1/D2 spiny projection neurons (SPNs) have their cell bodies in the
striatum (where the probe records) but send their axons far away — D1 → SNr, D2 → GPe. The
light fiber sits over the *target* (GPe/SNr), and the opsin (ChR2) is present all along the
membrane, including the axon terminals.

**What we want — an antidromic spike.** Flashing the terminal makes it fire. A spike can
travel both ways along an axon; the useful direction is *backwards*, up the axon to the soma
in the striatum. If the probe sees that soma fire right after the flash, it proves the cell's
axon goes to the stimulated site (e.g. SNr → it's a D1). This is a one-cell event with a
rock-steady, very short latency.

**The trap — synaptic ("trans-synaptic") responses.** The lit terminals also do their normal
job: release neurotransmitter onto the local GPe/SNr circuit. The basal ganglia is a tightly
wired loop and the striatum is densely interconnected, so the flash stirs up the network and
that activity comes *back* to the striatum through other neurons, a few synapses later. So a
striatal cell can fire shortly after the flash either (A) because its *own* axon was lit
(antidromic, a real tag) or (B) because *other* cells poked it (synaptic, a false positive —
this cell's axon may not even go there). Both look like a short-latency response, so a naive
test — and even SALT — cannot tell them apart.

**The collision test — single-track-railway logic.** An axon can't carry two spikes through
each other; if two action potentials travel toward each other they collide and both vanish
(like two trains on a one-track line). We exploit the neuron's *own* spontaneous spikes
(which start at the soma and travel down the axon) as the second train:

- On pulses with **no** spontaneous spike just before → the antidromic spike has a clear track
  home → response **present** (this is the positive control).
- On pulses **with** a spontaneous spike just before → for a *true antidromic* cell the two
  collide and the response **disappears**; for a *synaptic* response it shows up anyway
  (synapses don't care what the cell just did).

We split the pulses into these two groups and test whether the response is *suppressed* when a
spontaneous spike preceded the flash. **Result: 167 of 186 candidates fired regardless →
synaptic, not antidromic.** Only a handful show genuine collision suppression.

**Why it needs spontaneous spikes, and the "untestable" bucket.** We don't *send* spontaneous
spikes — they occur on their own at the cell's baseline firing rate, and we just observe which
pulses happened to be preceded by one. The test is a *comparison*, so it needs enough pulses
in **both** groups. A high-firing cell has spontaneous spikes before many pulses (testable); a
quiet ~1 Hz SPN has them before only ~2–3 of 501 pulses (**too few → "untestable"**, 30% of
unit×fiber here). That is why the high-confidence tier is biased toward higher-firing cells.
Collision is still the criterion we rely on because it is the *only* test in this dataset that
separates antidromic from synaptic — short latency, low jitter, and SALT significance are
necessary but not sufficient (synaptic responses can have them too). The textbook alternative,
high-frequency following, needs high-frequency pulse trains, which this single-pulse protocol
does not have.

## Why D2 high-confidence = 0 (bridging interaction)

2 GPe fiber-responses reach the strict gate, but `classify_unit` assigns pathway by bridging logic (SNr-tag → D1, overriding GPe, since only D1 projects to SNr) and reports the **determining fiber's** tier. Those 2 units also have an SNr response at *candidate* quality, so they aggregate to **D1/candidate**, not D2/high-confidence. Hence 3 high-confidence units, all D1.

## Known limitations / refinements

1. **The excess-jitter gate is vacuous as implemented.** Jitter is measured *within* the estimated ±0.75 ms response window, so it is mechanically ≤0.47 ms for every candidate (observed max 0.469 ms); `STRICT_MAX_JITTER_MS=1.0` never binds and the jitter-cap sweep is flat from 0.5→3.0 ms. It does not change the current result (collision is the binding gate), but to add real discrimination, jitter should be computed on a wider window (e.g. the full 1–10 ms search range) or replaced by a latency-stability metric. **Recommended fix before any future re-run that leans on the jitter criterion.**
2. **Collision-untestable = 30%** of unit×fiber (low-FR units lack enough pre-pulse spontaneous spikes). The high-confidence tier therefore excludes low-FR units it cannot test — a documented bias toward higher-FR units. Low-FR SPNs that are genuinely antidromic but untestable remain in the candidate tier.
3. **Post-hoc window selection** makes the Poisson leg of the candidate gate anti-conservative; it is the *sensitive* leg (OR-gated with SALT), not a strict gate. Appropriate for the candidate tier.

## Data issues (actionable)

1. **Session 5092025 skipped:** pkl is `BG_046_05092025_b.pkl`; the loader's date→filename match doesn't handle the `_b` suffix. Fixing the loader (or symlinking) recovers a 23rd laser session.
2. **Waveform labels absent** (`waveform_celltype_labels.csv` not found) → the FSI/narrow cross-check was a pass-through (`waveform_ok=True` for all). It did not inflate the high-confidence count here (collision is binding), but should be regenerated for label purity.
3. Worth auditing the 18 non-manifest pkls for a `Laser` key — if tagging was performed on additional days, those sessions are currently invisible to the manifest-gated script.

## Bottom line

- For **strong, publishable D1/D2 identity claims**, the antidromic yield in BG_046 is **~3 collision-confirmed units** across 22 sessions — too few to anchor cell-type-resolved population analyses on optotagging alone.
- For **exploratory / putative** cell-typing, **162 candidate units** (66 D1, 96 D2) are available, but they should be labelled "short-latency responsive," not "confirmed antidromic," because ~90% fail the collision test.
- Practical path to more confident tags: (a) fix the jitter metric; (b) recover 5092025 + audit non-manifest pkls; (c) regenerate waveform labels to exclude FSIs; (d) recognise that the low antidromic yield may be intrinsic to the prep (expression / terminal-stimulation efficacy), which is a wet-lab rather than analysis question.
