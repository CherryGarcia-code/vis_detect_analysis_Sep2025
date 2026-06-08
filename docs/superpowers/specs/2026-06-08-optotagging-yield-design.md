# Optotagging Yield Redesign — Antidromic D1/D2 Identification (BG_046)

- **Date:** 2026-06-08
- **Status:** Approved design (pre-implementation)
- **Branch (implementation):** commits **deferred** — a parallel chat holds `main` in this
  checkout; work stays in the primary working tree with no branch switch until the user
  signals `main` is free, then commit (or isolate via worktree if the other chat is still
  active).
- **Scope:** `src/visdetect/analysis/optotagging.py` (extend in place) + `analysis_suite/09_optotagging/a_optotagging_identification.py` (refactor) + tests.

---

## 1. Motivation

The proposal depends on identifying D1 (direct) and D2 (indirect) SPNs in BG_046 via
**antidromic optotagging**: post-task, 501 laser pulses are delivered to SNr (D1 terminal
field) and 501 to GPe (D2 terminal field); a striatal unit that emits a short-latency,
time-locked spike after terminal stimulation is identified as projecting there.

Current yield (cached `optotagging_results.csv`, 20 sessions with laser data):

| Fiber → pathway | Tagged / candidate | Yield |
|---|---|---|
| GPe → D2 | 168 / 3649 | 4.6 % |
| SNr → D1 | 36 / 3649 | 1.0 % |

### Diagnostic: where yield is lost and why the labels are weak

1. **SALT is the only real gate.** Of 3649 GPe candidate rows, 252 pass SALT p<0.01;
   the latency (<8 ms) and jitter (<3.5 ms) criteria pass for 3641 / 3618 rows
   respectively — they constrain almost nothing. Reliability ≥ 0.1 then trims 252 → 168.
2. **Metrics are not baseline-corrected.** `reliability` is the raw fraction of pulses
   with *any* spike in 0–10 ms; for a high-firing unit this is dominated by spontaneous
   spikes. The current top-reliability GPe unit fires **35 Hz spontaneously** with
   **1.8 ms first-spike jitter** and reliability 0.94 — almost entirely spontaneous, and
   far too jittery to be a clean antidromic spike. The current "best" tags are suspect.
3. **No antidromic confirmation.** The gold-standard antidromic criterion — the
   **collision test** — is not implemented, and jitter is allowed up to 3.5 ms (≈5× too
   lenient; true antidromic jitter is sub-millisecond).
4. **Candidate pool — corrected understanding (NOT a bottleneck).** `good_and_stable_ids`
   is *not* a UnitMatch cross-session set (as CLAUDE.md/MEMORY.md wrongly state). It is the
   Khilkevich & Lohse **within-session** filter `find_good_stable_units`
   ([qc.py:269](../../../src/visdetect/core/qc.py)): avg FR ≥ 0.5 Hz, firing-rate stability
   across the recording (rate never drops below 30/20/10 % of the mean in 20/10/5-min
   windows), ISI quality. Crucially, the pkls store spike data **only** for these units
   ([ingest.py:471](../../../src/visdetect/core/ingest.py) filters clusters to the stable
   set), so the candidate pool is fixed at the pkl-resident good_and_stable set and cannot be
   grown without re-processing. This is acceptable for a *post-task* laser protocol: the
   stability criterion ~guarantees the unit is present during the laser block — exactly what
   antidromic tagging needs. **The pool is therefore not the yield bottleneck; diagnostics
   1–3 are.** (Decision: use the pkl-resident pool as-is; no re-ingest.)

### Feasibility findings (probe of session 2072025)

- Pulse train is **~0.65 Hz with jittered inter-pulse intervals (1.0–2.2 s)**. This is
  ideal for an **offline collision test**: ~1.5 s of spontaneous activity precedes each
  pulse. 83 % of pulses had a spontaneous spike in the pre-pulse window for a high-FR unit.
- Pulses are **single pulses, not high-frequency trains** → classic frequency-following
  **cannot** be tested. The strict tier leans on **collision + sub-ms jitter** instead.
- Spike times are at 30 kHz (0.033 ms resolution) → sub-ms jitter is measurable.

## 2. Goals / Non-goals

**Goals**
- Report the *best achievable* D1/D2 yield for BG_046 under defensible criteria, as **two
  tiers**: a permissive **candidate** tier (max sensitivity) and a **high-confidence** tier
  (antidromic gold standard).
- Replace contaminated metrics with **baseline-corrected** equivalents.
- Add an **offline collision test** and **physiologically-correct D1/D2 assignment**.
- Quantify the gain vs. the current pipeline and expose the yield-vs-threshold tradeoff.

**Non-goals**
- Frequency-following (protocol has no trains).
- Re-running the laser protocol or any data re-acquisition.
- Changing optotagging for other subjects (BG_046 only here; code stays general).
- Causal/behavioral analysis of tagged units (downstream, separate work).

## 3. Approach (chosen: A — extend in place)

Extend `src/visdetect/analysis/optotagging.py` with new, independently-testable functions;
enrich `OptoMetrics`; add a unit-level aggregation/classification step; refactor the Fig43
script to produce both tiers, an old-vs-new comparison, and a yield-vs-threshold sweep.
Rejected: a parallel `antidromic.py` module (duplicates loading/splitting, two code paths,
against the consolidated-`main` ethos) and a script-only patch (scientific logic would live
outside the testable library).

## 4. Metric definitions (precise)

All windows are relative to pulse onset. Defaults are starting points; tunable constants
live at module top.

- **Baseline window** `BASELINE_WINDOW_MS = (-50, -5)` (the −5 ms guard avoids pre-pulse
  artifact). Baseline rate `λ_b` = baseline spike count / (n_pulses · |baseline window|).

- **Response-window estimation.** Build a pooled baseline-subtracted PSTH over `(1, 10)` ms
  at 0.1 ms bins (start at 1 ms to skip stimulus artifact). Locate the peak bin `t_peak`;
  define the response window `W = [t_peak − Δ, t_peak + Δ]` with `Δ = 0.75 ms` (or FWHM of
  the peak, whichever is tighter). Require the peak to exceed baseline (Poisson test, below)
  before a unit is eligible.

- **Excess reliability** `= p_resp − p_base`, clamped to ≥ 0, where
  `p_resp` = fraction of pulses with ≥1 spike in `W`, and
  `p_base = 1 − exp(−λ_b · |W|)` (expected hit rate from baseline). Replaces raw reliability.

- **Excess jitter** = std of first-spike latencies within `W` (on pulses with a response).
  Measured at native 30 kHz resolution. Antidromic expectation: sub-ms.

- **Peak latency** = `t_peak` (mean first-spike latency within `W` reported alongside).

- **SALT p-value (canonical fix).** Build the null from multiple equal-width baseline
  windows (Kvitsiani 2013) rather than the current JSD-to-uniform shortcut: compute the JS
  divergence of the test-window latency distribution against each baseline window; the null
  is the distribution of baseline-vs-baseline divergences; `p` = rank of the test divergence
  in that null. Keep the seed fixed for reproducibility.

- **Poisson excess-rate test (complementary, candidate-tier sensitivity).** `k_obs` = total
  spikes in `W` across pulses; expected `λ = λ_b · |W| · n_pulses`; `poisson_p` = upper-tail
  Poisson survival. A simple, sensitive short-latency-excess detector.

- **Collision test (antidromic confirmation).** Collision window
  `C = [pulse − (peak_latency + τ_ref), pulse]` with `τ_ref = 1.0 ms`. Partition pulses:
  - *collision-expected*: a spontaneous spike falls in `C`.
  - *collision-free*: no spike in `C`.
  Compute response occurrence (spike in `W`) for each set: `p_free`, `p_expected`.
  **Result is three-state:**
  - `PASS` — `p_free` high and `p_expected` significantly suppressed (Fisher exact, one-sided
    `p_free > p_expected`, α=0.05) **and** ≥10 collision-expected and ≥30 collision-free pulses.
  - `FAIL` — both sets present (testable) but no significant suppression.
  - `UNTESTABLE` — too few collision-expected pulses (low-FR unit).
  Report **collision suppression index** `= (p_free − p_expected) / p_free`.

- **Waveform cross-check.** Join tagged units to `waveform_celltype_labels` via
  `load_waveform_labels`. Annotate predicted cell type. SPN-implausible waveforms
  (narrow / fast-spiking — FSIs do not project to GPe/SNr) are flagged.

## 5. D1/D2 assignment (bridging-collateral logic, unit-level)

Only D1 SPNs project to SNr; D2 SPNs project to GPe; D1 SPNs also send **bridging
collaterals** to GPe. Therefore, per unit (combining its GPe and SNr results at a given tier):

- SNr-tagged → **D1** (specific; overrides GPe).
- GPe-tagged **only** → **D2**.
- GPe-tagged **and** SNr-tagged → **D1** (the GPe response is a bridging collateral).
- Neither → untagged.

This reassigns the previously "dual/ambiguous" units to D1 and will shift counts between
pathways relative to the current pipeline.

## 6. Two-tier classification

Per unit×fiber, then aggregated per unit via §5.

- **Candidate tier (max sensitivity):** good_and_stable (pkl-resident) unit;
  `(salt_p < 0.05) OR (poisson_p < 0.01)`;
  peak latency in `(1, 10)` ms; `excess_reliability > 0.02`.
- **High-confidence tier (gold standard):** candidate **AND** `salt_p < 0.01`
  (`STRICT_SALT_ALPHA`, stronger than the candidate 0.05) **AND** `excess_jitter < J*`
  (data-set, prior ≈ 0.5 ms; default cap 1.0 ms) **AND** `collision == PASS` **AND** waveform
  not flagged as FSI/narrow (unlabeled is allowed).

**Known tradeoff (documented, not hidden):** the high-confidence tier requires a testable
collision result, which biases it toward higher-FR units. Low-FR SPNs that pass SALT +
sub-ms jitter but are collision-`UNTESTABLE` remain in *candidate*. The explicit
`collision_status` column lets the user carve an intermediate "probable" set if wanted.

## 7. Architecture & data flow

```
load_session (good_and_stable = pkl-resident pool)   [analysis_suite Fig43 / per session]
        │
        ▼
OptoTagger.split → GPe / SNr pulse blocks      [existing, reused]
        │
        ▼  per unit × fiber
estimate_response_window → baseline-corrected metrics
        ├── excess_reliability, excess_jitter, peak_latency
        ├── salt_test (canonical),  poisson_excess_test
        └── collision_test → status + suppression index
        │
        ▼  enriched OptoMetrics (per unit × fiber)
classify_unit(gpe_metrics, snr_metrics)        [NEW unit-level aggregator]
        ├── tier ∈ {high_confidence, candidate, none}
        └── pathway ∈ {D1, D2, none}  (§5 bridging logic)
        │
        ▼  join waveform cell-type labels (annotation + FSI flag)
        ▼
results CSV (tiered) + figures (examples, latency/jitter, yield-by-stage,
             yield-vs-threshold sweep, old-vs-new comparison) + results note
```

### New/changed library surface (`optotagging.py`)
- `estimate_response_window(spikes, pulses, ...) -> (t_peak, W, λ_b, peak_significant)`
- `excess_reliability(...)`, `excess_jitter(...)`
- `salt_test(...)` — fixed to canonical baseline-window null (keep signature).
- `poisson_excess_test(...) -> p`
- `collision_test(spikes, pulses, peak_latency, W, τ_ref, ...) -> CollisionResult`
  (`status`, `suppression_index`, `p_free`, `p_expected`, `n_free`, `n_expected`).
- Enriched `OptoMetrics` (new fields: `baseline_rate_hz`, `response_window_ms`,
  `peak_latency_ms`, `excess_reliability`, `excess_jitter_ms`, `poisson_p`,
  `collision_status`, `collision_suppression_index`, `n_collision_free`,
  `n_collision_expected`).
- `classify_unit(gpe: OptoMetrics, snr: OptoMetrics, thresholds) -> UnitTag`
  (`tier`, `pathway`, contributing fiber).
- `OptoTagger.analyze_all` default pool is **unchanged**: the pkl-resident
  `good_and_stable` set (the only units with spike data). `get_good_cluster_ids` already
  resolves to this. No pool change is possible without re-processing.

### Constants (module top, single source of truth)
`BASELINE_WINDOW_MS=(-50,-5)`, `RESPONSE_SEARCH_MS=(1,10)`, `RESP_HALFWIDTH_MS=0.75`,
`COLLISION_REFRACTORY_MS=1.0`, `MIN_COLLISION_EXPECTED=10`, `MIN_COLLISION_FREE=30`,
`CANDIDATE_SALT_ALPHA=0.05`, `CANDIDATE_POISSON_ALPHA=0.01`, `CANDIDATE_MIN_EXCESS_REL=0.02`,
`STRICT_SALT_ALPHA=0.01`, `STRICT_MAX_JITTER_MS=1.0` (target ≈0.5, set from data).

## 8. Testing strategy (TDD; synthetic sessions)

Use `visdetect.utils.synthetic.make_synthetic_session` + injected laser events and crafted
spike trains:
- **excess_reliability / λ_b**: pure-Poisson unit → excess_reliability ≈ 0; injected locked
  responses → excess_reliability ≈ injected hit rate.
- **excess_jitter**: spikes at fixed latency + small Gaussian → recovered jitter ≈ σ.
- **response-window estimation**: peak recovered at injected latency.
- **collision_test**: construct a unit whose locked response is *removed* whenever a
  spontaneous spike precedes the pulse in `C` → `PASS`; a synaptic unit with no such
  dependence → `FAIL`; a low-FR unit with <10 collision-expected pulses → `UNTESTABLE`.
- **salt_test (canonical)**: locked response → small p; flat unit → p≈1; reproducible seed.
- **poisson_excess_test**: excess present → small p; none → ~uniform.
- **classify_unit / bridging logic**: SNr-only→D1, GPe-only→D2, both→D1, neither→none;
  tier gating honored.
- **pool default**: `analyze_all` with no ids analyzes exactly the session's loaded clusters
  (the pkl-resident good_and_stable pool); IDs listed in `good_cluster_ids` but lacking a
  Cluster object are silently absent — assert via a synthetic session where `good_cluster_ids`
  lists more IDs than there are loaded clusters.

## 9. Deliverables

1. Enriched, unit-tested `optotagging.py` (Approach A).
2. Refactored `Fig43` producing: tiered results CSV, example rasters/PSTH for a
   collision-confirmed unit per pathway, latency/excess-jitter distributions, yield-by-stage
   (both tiers), **yield-vs-threshold sweep**, **old-vs-new yield comparison**.
3. A short results note: the actual D1/D2 counts achievable for BG_046 in each tier, with the
   collision-untestable caveat quantified.

## 10. Risks & caveats

- **High-confidence tier is FR-biased** (collision testability) — documented; candidate tier
  and explicit `collision_status` mitigate.
- **Low-FR SPNs** yield few collision-eligible pulses → genuine antidromic units may sit in
  candidate, not high-confidence. This is honest, not a bug.
- **Bridging-collateral relabeling** changes label semantics vs. the old pipeline; the
  results note will state both the old and new D1/D2 counts.
- **Waveform labels** depend on `waveform_celltype_labels.csv`; if absent, the FSI flag is
  skipped (annotation-only, never a hard failure) and the run still completes.
- Canonical SALT is more expensive than the JSD-to-uniform shortcut; keep `n_jitter`
  configurable and the per-session loop parallelizable (existing `--n-workers`).
```
