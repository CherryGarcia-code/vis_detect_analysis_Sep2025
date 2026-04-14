# Deep Audit: Concat-Sort Drift Correction and nblocks Issue

**Date**: 2026-03-26
**Context**: Investigation into extreme drift patterns observed in KS4 diagnostic plots, and their impact on sorting quality and unit yield.

---

## Executive Summary

The concat-sort pipeline code is correctly implemented (session ordering, binary splitting, spike assignment all verified). The extreme drift patterns are caused by a **parameter misconfiguration**: `nblocks=5` on 96-channel per-shank data is pathologically underconstrained, producing noisy drift estimates that corrupt template estimation and reduce sorting quality.

---

## Finding 1: Session Ordering is CORRECT

Verified across all 12 pipeline scripts:

- `select_learning_sessions.py`: Sorts chronologically via `_parse_date()` (line 80-81)
- `build_concat_windows.py`: Explicitly re-sorts by date (lines 95-98) before writing `params.py`
- `ks4_run_manifest.json`: All 34 temporal windows verified to contain chronological session lists
- `params.py` files: Spot-checked in 8 windows — binary file lists match manifest order exactly

**The drift jumps are NOT from session mis-ordering.**

---

## Finding 2: nblocks=5 is Pathologically Underconstrained (CRITICAL)

### The Geometry Problem

NP2.0 4-shank probe, per-shank:
- 96 channels, 2 columns x 48 rows
- 15 um vertical pitch, 705 um total depth span

With `nblocks=5` (non-rigid drift in 5 depth sections):

| Metric | Per-shank (96 ch) | Standard NP1.0 (384 ch) |
|--------|-------------------|--------------------------|
| Channels per drift block | **19** | 77 |
| Depth per block | 141 um | 384 um |
| Signal for drift estimation | Very low | Adequate |

KS4 estimates drift by correlating spike density histograms across depth. With only 19 channels per block, each histogram has very few entries per batch, making the correlation extremely noisy.

### Quantitative Evidence

**Drift estimate quality across all 136 KS4 runs (34 windows x 4 shanks):**

| Drift Quality Group | N Runs | Lag-1 Autocorrelation | Mean |dshift| | Mean Good Clusters |
|---------------------|--------|----------------------|-----------------|-------------------|
| Good (AC > 0.9) | 66 | > 0.9 | varies | 73 (41.6%) |
| Mid (0.7-0.9) | 33 | 0.7-0.9 | varies | 55 (36.4%) |
| Bad (AC < 0.7) | 37 | < 0.7 | varies | 56 (36.0%) |

Real slow drift should produce autocorrelation > 0.95. Having 37/136 runs (27%) with AC < 0.7 indicates the drift estimates are dominated by noise in those runs.

**Single-batch drift spikes in Window 0, Shank 0:**
- 10 largest jumps: 260-332 um (on a 705 um probe!)
- These are single-batch noise spikes, not real probe movement
- 2.1% of all batches have >100 um jumps
- 4.2% have >50 um jumps
- All large jumps are intra-session, NOT at session boundaries

**Comparison — good late window (Window 31, Shank 0):**
- AC = 0.983, drift range [-12, 50] um
- Zero jumps > 50 um
- Median batch-to-batch jump: 0.5 um (vs 2.5 um for Window 0)

### Temporal Pattern

| Phase | Windows | Mean AC1 | Interpretation |
|-------|---------|----------|----------------|
| Early (Jun) | 0-8 | 0.46-0.75 | Noisy: probe settling + underconstrained estimation |
| Transition (Jul) | 9-20 | 0.53-0.99 | Mixed: some windows fine, some noisy |
| Late (Aug-Sep) | 21-33 | 0.58-1.00 | Mostly good, but shank 2/3 still noisy |

The temporal improvement reflects both real biology (probe settling over weeks) and the fact that later sessions have higher firing rates (more spikes per batch = better drift estimation).

### drift_smoothing is Too Weak

Default `drift_smoothing = [0.5, 0.5, 0.5]` provides minimal temporal smoothing. This was designed for NP1.0 with 384 channels where per-batch estimates are already reliable. For 96 channels, much stronger smoothing (e.g., [3.0, 3.0, 3.0]) is needed to suppress batch-to-batch noise.

---

## Finding 3: No Session Boundary Handling

KS4 treats the multi-file dat_path as one continuous recording:
- No padding, silence, or marker between sessions
- Last sample of session N (recorded day X) immediately followed by first sample of session N+1 (recorded day X+1 to X+7)
- Drift correction attempts to estimate smooth drift across these hard transitions
- Whitening matrix computed once across all sessions (does not adapt)

Session boundary jumps at Window 0, Shank 0:
| Boundary | Between sessions | Instantaneous jump | 3-batch mean step |
|----------|------------------|--------------------|-------------------|
| 1 | Jun 24 → Jun 25 | -4.5 um | -4.7 um |
| 2 | Jun 25 → Jun 26 | -16.5 um | +11.7 um |
| 3 | Jun 26 → Jun 27 | +8.5 um | +2.5 um |
| 4 | Jun 27 → Jun 30 (weekend) | 0.0 um | **-64.5 um** |

The -64.5 um step at boundary 4 (3-day gap over a weekend) is real probe displacement — consistent across all 4 shanks and all 9 depth columns. This is biologically plausible for a chronic NP2.0 implant.

---

## Finding 4: spike_datasets.npy Missing

KS4 version 4.1.1 did NOT produce `spike_datasets.npy` (file membership tracking). The `split_ks4_to_sessions.py` script handles this correctly by using `np.searchsorted` on cumulative sample offsets to assign spikes to sessions. Verified that:
- Session-local spike times start at 0
- Duration matches expected session length
- No spikes lost at boundaries

---

## Finding 5: Impact on Sorting Quality

The noisy drift correction has a statistically significant but modest direct effect:
- Spearman(drift_AC1, %_good_clusters) = 0.198 (p = 0.02)

However, the indirect effects are larger:
1. **Template corruption**: Templates estimated from drift-corrected data inherit drift noise, causing template shape distortion
2. **Split/merge artifacts**: Units that should be one cluster get split because their drift-corrected waveforms vary too much, or merged because noisy drift brings unrelated units together
3. **Quality metric inflation**: Contamination and amplitude metrics become unreliable when the drift correction itself introduces noise

This contributes to the full yield cascade: 43% fewer good labels × 33% below rate floor × 54% fail stability = 0.17x final yield.

---

## Per-Shank Sorting Quality Summary

| Shank | Mean Good Clusters | Mean AC1 | Notes |
|-------|-------------------|----------|-------|
| 0 | 42 | 0.839 | Fewest clusters overall |
| 1 | 44 | 0.826 | Similar to shank 0 |
| 2 | 67 | 0.803 | More clusters but worst drift |
| 3 | 104 | 0.834 | Most clusters (highest signal) |

---

## Recommended Fix

### Minimum viable re-sort (per-shank, corrected params):

```python
nblocks = 1           # Rigid drift correction, uses ALL 96 channels
drift_smoothing = [3.0, 3.0, 3.0]  # Stronger temporal smoothing
```

**Why nblocks=1**: Striatum is anatomically homogeneous over 705 um. Non-rigid correction is designed for cortex where different layers drift differently. For striatum, rigid (whole-probe) correction is both more appropriate and far more robust with 96 channels.

**Expected improvement**:
- Eliminates the single-batch noise spikes entirely
- Should increase the good-label rate from ~38% to potentially ~50-60%
- Combined with the stability filter, could recover significant yield

### Ideal re-sort (full 384-ch on H100):

If H100 (80 GB) GPUs are available:
- Sort all 384 channels with `nblocks=1-2`, `window_size=2`, `drift_smoothing=[3.0, 3.0, 3.0]`
- Recovers the 43% KS-good label loss from per-shank sorting
- Provides much better drift estimates (384 ch → 77 ch/block even at nblocks=5)

---

## Files Referenced

| File | Key Finding |
|------|-------------|
| `scripts/pipelines/concat_sort/run_kilosort4.py` | `nblocks=5` set at line ~167 |
| `scripts/pipelines/concat_sort/build_concat_windows.py` | Chronological sorting at lines 95-98 |
| `scripts/pipelines/concat_sort/select_learning_sessions.py` | Correct date parsing at lines 36-44 |
| `scripts/pipelines/concat_sort/split_ks4_to_sessions.py` | Correct np.searchsorted session assignment |
| `X:/.../concat_sort/ks4_runs/window_000/shank_0/ops.npy` | dshift shape (28496, 9), drift range [-282, +230] um |
| `X:/.../concat_sort/ks4_runs/ks4_run_manifest.json` | 136 runs, all verified chronological |
