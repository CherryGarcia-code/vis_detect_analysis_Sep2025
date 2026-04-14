# Concat-Sort Pipeline: Audit Results and Options Analysis

**Date**: 2026-03-25
**Context**: Full audit of the 12-stage concat-sort pipeline and diagnostic comparison against the original per-session sorting pipeline.

---

## Audit Summary

The concat-sort pipeline code is **well-engineered with no data-integrity bugs**. The stability filter (`find_good_stable_units`) is a verified line-by-line match of the MATLAB `find_good_stable_units_PaperVersion.m` (Khilkevich & Lohse 2024). All time corrections, stitching logic, and cluster ID mappings are correct.

The problem is **yield**: the concat-sort produces dramatically fewer usable units.

---

## Yield Comparison (38 sessions)

| Metric | Old Pipeline | Concat-Sort | Ratio |
|--------|-------------|-------------|-------|
| Total clusters in pkl | 6,100 (trimmed to stable only) | 22,705 | 3.7x more raw clusters |
| KS4 "good" labeled | 9,760 | 5,564 | **0.57x** |
| Good & stable (final) | 6,100 | 1,663 | **0.27x** |
| Stable / Good pass rate | 62.5% | 29.9% | Half the pass rate |
| Mean stable per session | 160.5 | 43.8 | ~3.7x fewer |

### Root Cause: Three Compounding Losses

**Loss 1 — Per-shank sorting yields 43% fewer KS4 "good" labels**
- Old: 384-channel single sort → 9,760 KS-good
- New: 4 × 96-channel per-shank sorts → 5,564 KS-good
- With only 96 channels, KS4 has less spatial information for template estimation and quality assessment

**Loss 2 — 33% of concat-sort "good" clusters are below 0.5 Hz rate floor**
- Multi-session windows detect units that are clean but fire too rarely in individual sessions
- Median rate of these sub-threshold units: 0.074 Hz (~874 spikes in ~197 min)
- Uniformly distributed across all 4 shanks (~30-34% each)

**Loss 3 — Stability filter pass rate drops from 62.5% → 29.9%**
- 20-min sliding window catches more rate dips in per-shank sorted data
- Combined with the lower-quality cluster population from 96-ch sorting

**Multiplicative effect**: 0.57 × 0.67 × 0.45 ≈ 0.17 → predicts 1,660 stable units (actual: 1,663).

### Additional Finding: Old PKL Structure

The old `.mat` files are trimmed by MATLAB's `trim_probe_struc` function, which **deletes all spike data from non-stable clusters** before saving. So old pkls contain only stable clusters as actual `Cluster` objects. The `good_cluster_ids` list preserves the full KS-good set (untrimmed), but the spikes for those clusters are gone. This is why `old_total == old_stable` in the yield summary.

---

## Options

### Option A: Re-sort with full 384 channels (concat windows)

Re-run the concatenated sort using all 384 channels with KS4's built-in `kcoords` shank handling, matching how the old pipeline sorted. This preserves the temporal continuity benefit of the concat approach while recovering the spatial information lost by per-shank splitting.

**What changes**:
- Skip `split_by_shank.py` entirely
- Modify `build_concat_windows.py` to use the full AP binary (384 ch + sync) with proper channel map including shank coords (`kcoords`)
- KS4 sorts 384 ch × 5 sessions per window (same channel count as old pipeline, 5× more time)

**Pros**:
- Most scientifically defensible — same spatial information as old sort + temporal continuity
- Expected to recover the 43% KS-good label loss
- All downstream pipeline stages (split, stitch, build_pkls) work unchanged

**Cons**:
- Much higher GPU memory per job: 384 ch × 5 sessions vs 96 ch × 5 sessions
- May require smaller window size (3 sessions) or larger GPUs (A100 80GB)
- Full HPC re-run required (~1-2 weeks of GPU time)

**See**: GPU feasibility analysis below.

### Option B: Keep per-shank sorting, relax stability filter

Adopt the existing concat-sort results with adjusted thresholds to recover more units.

**What changes**:
- Lower `avg_fr` threshold from 0.5 Hz to 0.2 Hz
- Or create a "concat-sort QC profile" with relaxed criteria
- Re-run `build_concat_pkls.py` with modified filter

**Pros**:
- No re-sorting required (saves weeks of HPC time)
- Quick to implement

**Cons**:
- Departs from Khilkevich & Lohse 2024 published criteria
- Only recovers the rate-filtered units (~33% of good), not the KS-good labeling loss
- Recovered units are lower-rate and may not contribute to population analyses

### Option C: Hybrid — old pkls for analysis, concat for longitudinal tracking

Use the existing old pkls (which work well) for all within-session analyses. Use the concat-sort stitching + UnitMatch outputs only for cross-session unit identity tracking.

**What changes**:
- Complete the UnitMatch or DeepUnitMatch step on the existing concat-sort data
- Build a cross-session unit registry mapping old-pipeline cluster IDs across sessions
- Keep analysis_suite pointing at old pkls

**Pros**:
- Most pragmatic — uses existing data immediately
- Longitudinal tracking (the original goal) can proceed independently
- No re-sorting required

**Cons**:
- Two parallel data paths to maintain
- Cross-session tracking quality limited by per-shank sort quality

### Option D: Re-sort with 384 channels, window_size=3

A lighter variant of Option A: re-run with full 384 channels but smaller windows.

**What changes**:
- Same as Option A but with `--window-size 3 --stride 1`
- 36 windows × 1 sort each (vs 34 × 4 shanks = 136 in current pipeline)

**Pros**:
- Reduces per-job GPU memory vs Option A with window_size=5
- Still provides overlapping windows for stitching
- Fewer total jobs (36 vs 136)

**Cons**:
- Less temporal context for template estimation
- 2 overlapping sessions per consecutive window pair (vs 4) → less robust stitching
- Still requires significant HPC time
