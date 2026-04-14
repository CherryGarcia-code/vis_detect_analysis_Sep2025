# GPU Feasibility: Option A — 384-Channel Concatenated Sort

**Date**: 2026-03-25
**Question**: Can we re-run the concat-sort using all 384 channels instead of per-shank 96-channel splits, given the GPUs available on the institute HPC cluster?

---

## Available GPU Hardware

| GPU | VRAM | Node(s) | Role in Current Pipeline |
|-----|------|---------|--------------------------|
| **NVIDIA L40S** | 48 GB | `gpu-sr675-31` | Best available — used for all final retries |
| **NVIDIA A100-40GB** | 40 GB | `gpu-sr670-20` to `gpu-sr670-23` | 4 nodes; many OOM failures, explicitly avoided in later retries |
| **A100-80GB** | 80 GB | *Not available on cluster* | Would be needed for 384-ch sorts |

System RAM: 128–192 GB per node (escalated to 192 GB for worst-case jobs).

---

## Current Pipeline: 96-Channel Memory Profile

From the batch run logs and retry scripts:

| Metric | Value | Source |
|--------|-------|--------|
| Channels per job | 96 (1 shank) | Per-shank split |
| Sessions per window | 5 (stride 1) | `build_concat_windows.py` |
| Default batch_size | 60,000 samples (2 s at 30 kHz) | `run_kilosort4.py` |
| Typical VRAM usage | 30–40 GB | Comment in auto-scaling logic |
| Worst-case single allocation | **17.05 GiB** | `run_ks4_retry_v5_task136.bash` — CUDA OOM on L40S (44.39 GB usable) |
| OOM rate at defaults (Th 9/8) | ~18% of 136 jobs | 21 initial + 5 subsequent OOM failures |
| Fix for OOM | Raise Th 10/9 or 11/10 (reduces spikes 20–30% per step) | Retry scripts |

The auto-scaling logic in `run_kilosort4.py` (lines 142–161):
```
<18 GB VRAM → batch_size = 15,000
<24 GB      → batch_size = 30,000
<44 GB      → batch_size = 45,000
≥48 GB      → batch_size = 60,000 (default)
```

---

## Scaling to 384 Channels: Memory Estimate

KS4's clustering step (`clustering_qr.py`) is the memory bottleneck. The key tensors scale with:
- **Number of channels** (template width)
- **Number of detected spikes** (data matrix rows)
- **Number of clusters** (template bank)

### Conservative Estimate

The dominant allocation during clustering involves the spike × feature matrix. With 4× more channels:

| Factor | 96 ch | 384 ch | Ratio |
|--------|-------|--------|-------|
| Channels per template | 96 | 384 | 4× |
| Spikes detected | N | ~N (same data, more channels don't increase spikes) | 1× |
| Feature dimensions (PCA of templates) | ~96 × T | ~384 × T | 4× |
| Worst-case single tensor | 17.05 GiB | **~68 GiB** | ~4× |
| Total VRAM required | 30–48 GB | **~120–192 GB** | ~4× |

The 17.05 GiB worst-case allocation on 96 channels would scale to ~68 GiB for 384 channels — **already exceeding any single GPU on the cluster**.

### Template Matching Phase

Template matching also scales with channel count (more spatial data per template). Task 136 hit system-RAM OOM at 94.7% through template matching on A100-40GB with 128 GB RAM — this was with only 96 channels. At 384 channels, system RAM requirements would also increase substantially.

---

## Verdict: NOT Feasible on Current Hardware

**384 ch × 5 sessions** (Option A as stated) is **not feasible** on L40S (48 GB) or A100-40GB.

The worst-case single allocation alone (68+ GiB) exceeds the VRAM of every GPU on the cluster. Even with aggressive threshold raising and batch_size reduction, the clustering step loads entire spike × feature matrices that cannot be further chunked without modifying KS4 internals.

---

## Alternatives Within Reach

### Alternative 1: 384 ch × Window Size 2–3 (Option D Variant)

Reducing the window from 5 to 2–3 sessions reduces the number of spikes proportionally:

| Window Size | Sessions | ~Spike Count | Est. Peak VRAM (384 ch) |
|-------------|----------|-------------|------------------------|
| 5 | 5 × ~40 min = 200 min | 100% | ~120–192 GB (not feasible) |
| 3 | 3 × ~40 min = 120 min | ~60% | ~72–115 GB (not feasible on 48 GB) |
| 2 | 2 × ~40 min = 80 min | ~40% | ~48–77 GB (borderline on L40S) |

Window size 2 *might* fit on L40S with raised thresholds (Th 10/9 or 11/10), but:
- Only 1 overlapping session per window pair → weaker stitching
- Still many OOM failures expected (the 17.05 GiB worst case was 96-ch; at 384-ch even window=2 may produce tensors >44 GB)

**Risk**: High. Expect 30–50% OOM rate even with threshold escalation.

### Alternative 2: Wait for A100-80GB Access

If the institute acquires A100-80GB (or H100-80GB) nodes:

| Configuration | Est. Peak VRAM | Feasible? |
|---------------|---------------|-----------|
| 384 ch × 3 sessions on A100-80GB | ~72–115 GB | Borderline (with Th escalation) |
| 384 ch × 2 sessions on A100-80GB | ~48–77 GB | **Likely feasible** |
| 384 ch × 5 sessions on A100-80GB | ~120–192 GB | Still too large |

A100-80GB + window_size=2 is the most realistic path to full 384-channel sorting.

### Alternative 3: KS4 `kcoords`-Aware Multi-Shank Mode

KS4 accepts `kcoords` (shank coordinates) in the channel map. When provided, KS4 internally restricts template estimation to same-shank channels while still using all channels for spike detection. This gives some benefit of cross-shank information without the full 384-ch memory cost.

**Status**: Not tested. Requires investigation of KS4's actual memory behavior with `kcoords` set — it may still load all 384 channels into the feature matrix during clustering.

---

## Recommendation

Given current hardware, **Option A (384-ch full sort) should not be attempted**. The most productive paths are:

1. **Option C (Hybrid)**: Use old pkls for analysis, concat-sort for tracking — zero re-sorting cost
2. **Option D with window_size=2**: Only if A100-80GB becomes available
3. **Investigate KS4 kcoords**: Could potentially offer a middle ground if KS4 reduces memory with multi-shank awareness

See `concat_sort_audit_and_options.md` for the full options comparison.
