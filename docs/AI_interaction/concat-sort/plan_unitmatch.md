# Plan: UnitMatch Integration for Concat-Sort Data

**Date**: 2026-03-25
**Goal**: Run Bayesian UnitMatch (UnitMatchPy) on the concat-sort per-shank data to produce cross-session unit identity tracking.

---

## Current State

### What Exists

| Component | Status | Path |
|-----------|--------|------|
| **Waveform input** | Ready (21,875 files) | `data/unit_match_concat_sort/input/BG_046/shank_{0-3}/{session}/RawWaveforms/` |
| **Channel maps** | Ready | `data/unit_match_concat_sort/input/BG_046/shank_{0-3}/{session}/channel_positions.npy` |
| **Pipeline script** | Written, never executed | `scripts/pipelines/concat_sort/run_concat_unitmatch.py` |
| **Conda env spec** | Written | `environment_unitmatch.yml` (name: `unitmatch_env`) |
| **Conda env** | Needs creation/verification | Requires Python 3.10, numpy<2, UnitMatchPy |
| **Output directory** | Does not exist | `data/unit_match_concat_sort/output/BG_046/shank_{0-3}/` |

### Input Data Details

- **4 shanks** × **36 sessions** (38 selected minus 2 dropped)
- Each session-shank directory contains:
  - `RawWaveforms/Unit{ID}_RawSpikes.npy` — shape `(82, N_channels, 2)` (CV-split mean waveforms)
  - `channel_map.npy`, `channel_positions.npy` — probe geometry
  - `cluster_group.tsv` — KS4 quality labels
  - `params.py` — metadata

### Pipeline Architecture

`run_concat_unitmatch.py` runs **independently per shank** using sliding-window batches:
- Default batch size: 12 sessions (fits ~64 GB system RAM)
- Reduces histogram bins from 100 → 50 for batches ≥ 8 sessions
- UnitMatchPy Bayesian pipeline: feature extraction → N×N probability matrix → thresholding → UID assignment
- Output per batch: `CellRegistry.csv`, `Unit_Long_Table.csv`, `MatchTable.csv`, `SessionList.txt`

---

## Execution Plan

### Step 1: Environment Setup

```bash
# On HPC or local machine
conda env create -f environment_unitmatch.yml
conda activate unitmatch_env

# Verify
python -c "import UnitMatchPy; print('OK')"
python -c "import numpy; print(numpy.__version__)"  # must be <2.0
```

**Requirements**:
- Python 3.10
- numpy < 2.0 (UnitMatchPy incompatible with numpy 2.x)
- UnitMatchPy (pip install)
- scipy, pandas, scikit-learn, h5py, tqdm

### Step 2: Run Per-Shank UnitMatch

```bash
conda activate unitmatch_env

# Run each shank (can be parallelized across 4 jobs)
python scripts/pipelines/concat_sort/run_concat_unitmatch.py --shank 0
python scripts/pipelines/concat_sort/run_concat_unitmatch.py --shank 1
python scripts/pipelines/concat_sort/run_concat_unitmatch.py --shank 2
python scripts/pipelines/concat_sort/run_concat_unitmatch.py --shank 3
```

**Resource requirements per shank**:
- CPU only (no GPU needed)
- ~32–64 GB RAM (N×N probability matrix for up to ~500 units per shank)
- Runtime: ~1–4 hours per shank depending on unit count

**Output**: `data/unit_match_concat_sort/output/BG_046/shank_{0-3}/` containing sliding-window batch folders with `CellRegistry.csv` files.

### Step 3: Stitch Batch Outputs

The sliding-window approach produces overlapping batch registries that must be merged. `run_concat_unitmatch.py` handles this internally for within-shank stitching.

The **cross-shank** question: Units on different shanks are physically separated by 250 µm and sorted independently, so cross-shank matching is not applicable. Each shank's registry is self-contained.

### Step 4: Build Unified Registry

Combine the 4 per-shank registries into a single concat-sort CellRegistry:

```python
# Pseudocode — needs a new script or addition to run_concat_unitmatch.py
import pandas as pd

registries = []
for shank in range(4):
    reg = pd.read_csv(f"data/unit_match_concat_sort/output/BG_046/shank_{shank}/CellRegistry.csv")
    # Prefix UIDs with shank to prevent collisions
    reg["UID"] = reg["UID"].apply(lambda x: f"s{shank}_{x}")
    registries.append(reg)

combined = pd.concat(registries, ignore_index=True)
combined.to_csv("data/unit_match_concat_sort/output/BG_046/ConcatSort_CellRegistry.csv", index=False)
```

### Step 5: Build Grand Longitudinal Table

The existing `scripts/analysis/build_longitudinal_table.py` accepts a `--registry` argument:

```bash
python scripts/analysis/build_longitudinal_table.py \
    --registry data/unit_match_concat_sort/output/BG_046/ConcatSort_CellRegistry.csv
```

However, this script currently loads sessions from `data/pkls/BG_046/` (old pkls). To use it with concat-sort pkls:
- Either pass `--pkl-dir data/pkls/BG_046_concat_sort/`
- Or modify `config.py` to point `PKL_DIR` at the concat-sort directory

**Note**: The concat-sort pkls have ~3.7× fewer stable units, so the GLT will be much sparser.

### Step 6: Validate

1. **Basic sanity**: Check that tracked units have consistent waveform shapes across sessions
2. **Track length distribution**: Most UIDs should span 1–5 sessions; a few should span 10+ if tracking works
3. **Waveform correlation**: Cross-session waveform Pearson r should be >0.8 for matched units
4. **Comparison with stitching**: The `stitch_across_windows.py` stitching (spike-time-based) and UnitMatch (waveform-based) should produce largely overlapping identities

Existing validation scripts:
- `scripts/QC_CHECKS/validate_unitmatch_results.py` — ISI fingerprint stability + waveform correlation
- `scripts/analysis/visualize_unitmatch.py` — probability matrix + waveform overlay plots

---

## Risks and Considerations

| Risk | Mitigation |
|------|------------|
| **Low unit count** per shank (~10–40 stable units) may give UnitMatchPy too few units for reliable Bayesian matching | Check output match probabilities; may need to relax probability threshold |
| **numpy 2.x incompatibility** | Conda env forces numpy<2; do NOT update numpy |
| **Memory for large batches** | batch_size=12 is pre-tuned; reduce to 8 if RAM issues |
| **Sparse registry** | With only ~44 stable units per session (across all 4 shanks), many UIDs will span only 1–2 sessions |
| **Concat-sort KS4 labels are less reliable** (per audit) | UnitMatch uses waveform shape, not KS labels, so this is partially mitigated |

---

## Timeline Estimate

| Step | Duration |
|------|----------|
| Env setup | 15 min |
| Run 4 shanks (parallel) | 1–4 hours |
| Stitch + combine | 30 min |
| GLT build | 1–2 hours |
| Validation | 1 hour |
| **Total** | **~Half a day** |

---

## Decision Point

Before executing, consider: **Is this the right pipeline for concat-sort data?**

The concat-sort already has its own spike-time-based stitching (`stitch_across_windows.py`) that assigns global UIDs. UnitMatch adds waveform-based identity tracking on top. The question is whether the additional waveform matching provides value given the low unit yield (~44 stable units per session).

If the plan is **Option C (Hybrid)** — use old pkls for analysis and concat-sort only for tracking — then UnitMatch on concat-sort data may have limited utility. Running UnitMatch (or Deep UnitMatch) on the **old per-session sort data** would give better results because:
1. More units to match (160 stable per session vs 44)
2. Higher-quality KS4 labels (384-ch vs 96-ch)
3. The old per-session UnitMatch pipeline already has input data prepared

See `plan_deep_unitmatch.md` for the Deep UnitMatch alternative.
