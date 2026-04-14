# Plan: Deep UnitMatch Integration

**Date**: 2026-03-25
**Goal**: Use the pre-trained DeepUnitMatch CNN model for cross-session unit identity tracking. Two paths evaluated: (A) re-run on concat-sort data, or (B) use existing results from old per-session sorts.

---

## Current State

### What Already Exists (Old Per-Session Sort)

The Deep UnitMatch pipeline has **already been run** on the old per-session Kilosort data:

| Component | Status | Path |
|-----------|--------|------|
| **Input waveforms** | Complete (42 sessions) | `data/unit_match/input/BG_046/{DDMMYYYY}/RawWaveforms/` |
| **DeepUM CellRegistry** | Complete (5,971 UIDs) | `data/deep_unit_match/output/BG_046/DeepUM_CellRegistry.csv` |
| **Match table** | Complete (708 matches) | `data/deep_unit_match/output/BG_046/DeepUM_MatchTable.csv` |
| **Embeddings** | Cached (27.6 MB) | `data/deep_unit_match/output/BG_046/embeddings.npz` |
| **Similarity matrix** | Cached (178 MB) | `data/deep_unit_match/output/BG_046/similarity_matrix.npy` |
| **Quality metrics** | Complete | `data/deep_unit_match/output/BG_046/Global_Unit_Quality_Metrics.csv` |
| **Pair registries** | Complete (88 pairs) | `data/deep_unit_match/output/BG_046/pair_registries/` |
| **Pre-trained model** | Available | `_DeepUnitMatch_repo/UnitMatchPy/DeepUnitMatch/utils/model` (21.7 MB) |

### GLT Integration Status

- `build_longitudinal_table.py` defaults to `--registry` → `DeepUM_CellRegistry.csv`
- **Post-TPrime GLT has NOT been built** — only `table_output/preTPrime/` exists
- Analysis suite (`loader.py::load_glt()`) is wired to consume `table_output/Grand_Longitudinal_Table.csv`

### Concat-Sort Waveform Input

| Component | Status | Path |
|-----------|--------|------|
| **Waveforms** | Complete (4 shanks × 36 sessions) | `data/unit_match_concat_sort/input/BG_046/shank_{0-3}/` |
| **DeepUM on concat data** | Not attempted | — |

---

## Path A: Re-Run Deep UnitMatch on Concat-Sort Data

### What Would Change

`run_deep_unitmatch.py` currently reads from `data/unit_match/input/BG_046/` (all sessions in flat directory). The concat-sort input is structured differently: `data/unit_match_concat_sort/input/BG_046/shank_{N}/{session}/`.

**Required modifications**:

1. **Input path handling**: Either:
   - Add a `--shank` flag to `run_deep_unitmatch.py` that scans `shank_{N}/` subdirectories instead of the flat layout
   - Or create a wrapper script that symlinks/copies concat-sort sessions into the expected flat structure

2. **Channel position mapping**: Concat-sort waveforms are per-shank (96 channels × 1 column layout). The preprocessing step (`_extract_snippet`) uses `channel_positions.npy` to find channels within 110 µm radius — this should work correctly since the positions are inherently per-shank.

3. **Cross-shank identity**: Deep UnitMatch (like Bayesian UnitMatch) would run **independently per shank**. Units on different shanks cannot be matched (250 µm inter-shank pitch exceeds the 110 µm channel radius).

4. **Registry combination**: Same as UnitMatch plan — prefix shank ID to UIDs and concatenate.

### Execution Steps

```bash
# 1. Set up environment (same as UnitMatch, plus PyTorch)
conda activate unitmatch_env
pip install torch  # CPU is sufficient

# 2. Modify or wrap run_deep_unitmatch.py for per-shank input
#    (see modification notes below)

# 3. Run per shank
python scripts/analysis/run_deep_unitmatch.py --input data/unit_match_concat_sort/input/BG_046/shank_0 --output data/deep_unit_match/output/BG_046_concat_sort/shank_0
python scripts/analysis/run_deep_unitmatch.py --input data/unit_match_concat_sort/input/BG_046/shank_1 --output data/deep_unit_match/output/BG_046_concat_sort/shank_1
python scripts/analysis/run_deep_unitmatch.py --input data/unit_match_concat_sort/input/BG_046/shank_2 --output data/deep_unit_match/output/BG_046_concat_sort/shank_2
python scripts/analysis/run_deep_unitmatch.py --input data/unit_match_concat_sort/input/BG_046/shank_3 --output data/deep_unit_match/output/BG_046_concat_sort/shank_3

# 4. Combine per-shank registries
# 5. Build GLT with --registry pointing to combined registry
```

### Script Modifications Needed

`run_deep_unitmatch.py` would need these changes:

1. **Add `--input` and `--output` CLI arguments** (currently hardcoded to `data/unit_match/input/BG_046/` and `data/deep_unit_match/output/BG_046/`)
2. **Session discovery**: Currently enumerates flat subdirectories matching DDMMYYYY. Per-shank directories have the same structure inside each shank folder, so this should work with just the path change.
3. **No preprocessing changes needed**: `_extract_snippet()` uses the per-session `channel_positions.npy` — already per-shank in concat-sort data.

### Expected Results

With ~10–40 stable units per shank per session (vs ~160 in old sort):
- **Fewer matches per pair**: The CLIP matching quality depends on having distinctive waveforms; fewer units means less competition but also fewer potential matches
- **Sparser registry**: Most UIDs will span 1–3 sessions
- **Quality concern**: 96-ch sorted units may have noisier waveform templates, degrading embedding quality

---

## Path B: Use Existing Deep UnitMatch Results (Recommended)

### Rationale

The Deep UnitMatch pipeline has **already completed** on the old per-session sort data with strong results:
- 5,971 UIDs across 42 sessions
- 708 pairwise matches
- Input data quality is higher (384-ch sorted, ~160 stable units/session)

This aligns with **Option C (Hybrid)** from the audit: use old pkls for analysis, use tracking results for longitudinal identity.

### Execution Steps

The only remaining work is to **rebuild the Grand Longitudinal Table** using the existing `DeepUM_CellRegistry.csv` with the post-TPrime pkl files:

```bash
# Step 1: Rebuild the GLT (uses existing DeepUM registry by default)
python scripts/analysis/build_longitudinal_table.py

# Step 2: Verify output
# → table_output/Grand_Longitudinal_Table.csv
# → table_output/Grand_Waveforms.pkl

# Step 3: Run analysis_suite scripts that depend on GLT
cd analysis_suite
py 05_longitudinal/a_neural_learning_curves.py
```

### What This Gets You

- **Cross-session unit tracking** across all 42 sessions using high-quality old-sort data
- **Full GLT** with behavioral metrics, TF responsiveness, QC metrics per unit per session
- **Ready for analysis_suite** — `loader.py::load_glt()` and `build_unit_table()` are already wired

### Prerequisites

1. **Post-TPrime pkls must exist** at `data/pkls/BG_046/` — confirmed: 45/46 sessions converted
2. **DeepUM registry** exists — confirmed: `DeepUM_CellRegistry.csv` with 5,971 UIDs
3. **Staging manifest** must include sessions from the registry — confirmed via `load_staging_manifest()`

---

## Comparison: Path A vs Path B

| Aspect | Path A (Concat-Sort DeepUM) | Path B (Old-Sort DeepUM) |
|--------|----------------------------|--------------------------|
| **Input quality** | 96-ch per-shank (lower) | 384-ch full-probe (higher) |
| **Units per session** | ~44 stable | ~160 stable |
| **Registry already exists?** | No — needs new run | Yes — `DeepUM_CellRegistry.csv` |
| **Script modifications** | Need `--input`/`--output` args | None |
| **Runtime** | ~2–4 hours (4 shanks) | **Already done** |
| **GLT compatibility** | Needs concat-sort pkl dir | Default path works |
| **Expected match quality** | Lower (fewer units, noisier templates) | Higher |
| **Tracking span** | 38 sessions (concat-sort subset) | 42 sessions (all available) |
| **Remaining work** | Modify script + run + combine + GLT | **Just rebuild GLT** |

---

## Recommendation

**Use Path B** — the existing Deep UnitMatch results with old per-session sort data.

The immediate next step is a single command:
```bash
python scripts/analysis/build_longitudinal_table.py
```

This builds the Grand Longitudinal Table from the existing `DeepUM_CellRegistry.csv` and post-TPrime pkl files, unblocking all longitudinal analysis in the analysis suite.

Path A (re-running on concat-sort data) only makes sense if:
1. You decide to replace old pkls with concat-sort pkls for all analyses (not recommended given yield gap)
2. The concat-sort yield is improved via Option A (384-ch re-sort) or Option B (relaxed thresholds)

---

## Dependencies: DeepUnitMatch Model and Code

| Component | Path | Notes |
|-----------|------|-------|
| Pre-trained model | `_DeepUnitMatch_repo/UnitMatchPy/DeepUnitMatch/utils/model` | 21.7 MB PyTorch state_dict |
| `SpatioTemporalCNN_V2` | `_DeepUnitMatch_repo/UnitMatchPy/DeepUnitMatch/utils/mymodel.py` | Conv1D encoder → 256-d |
| `clip_prob`, `clip_sim` | `_DeepUnitMatch_repo/UnitMatchPy/DeepUnitMatch/utils/losses.py` | CLIP temperature-scaled softmax |
| Conda env | `environment_unitmatch.yml` + `pip install torch` | PyTorch not in env file — must add manually |

**Note**: The `environment_unitmatch.yml` does NOT include PyTorch. For Deep UnitMatch, you must additionally:
```bash
conda activate unitmatch_env
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only, ~200 MB
```

---

## Related Files

- Audit: `docs/AI_interaction/concat-sort/concat_sort_audit_and_options.md`
- GPU analysis: `docs/AI_interaction/concat-sort/gpu_feasibility_option_a.md`
- UnitMatch plan: `docs/AI_interaction/concat-sort/plan_unitmatch.md`
- GLT builder: `scripts/analysis/build_longitudinal_table.py`
- DeepUM script: `scripts/analysis/run_deep_unitmatch.py`
