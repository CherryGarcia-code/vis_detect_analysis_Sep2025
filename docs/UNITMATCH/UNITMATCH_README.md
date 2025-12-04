# UnitMatch Implementation Guide

Complete implementation of chronic unit tracking using UnitMatch (Windolf et al. 2024, Nature Methods).

## Overview

This implementation provides:

1. **Configurable waveform extraction** - Kilosort templates, Bombcell waveforms, or both for comparison
2. **ITI-based filtering** - Extract waveforms during inter-trial intervals to avoid stimulus artifacts
3. **Full tracking pipeline** - Pairwise testing → batch tracking → validation
4. **Striatal learning support** - Validated parameters from paper (dorsomedial striatum, visuomotor task)

## Quick Start

### 1. Extract Waveforms

Extract waveforms from both sources for comparison:

```bash
python scripts/unitmatch/prepare_waveforms_for_unitmatch.py \
    --config config/unitmatch_sessions.yml \
    --source both \
    --use-iti \
    --compare
```

**Options:**
- `--source`: `kilosort`, `bombcell`, or `both` (default: from config)
- `--use-iti`: Filter to ITI periods only
- `--iti-method`: `trial_field`, `trial_boundaries`, or `fallback`
- `--compare`: Generate Kilosort vs Bombcell comparison plots

**Output:**
- `png_output/unitmatch_waveforms/{session}_waveforms_{source}_{iti|full}.npy`
- `png_output/unitmatch_waveforms/comparisons/{session}_waveform_comparison.png` (if `--compare`)
- `png_output/unitmatch_waveforms/comparison_summary.yaml` (correlation statistics)

### 2. Test Pairwise Matching

**Option A – Precomputed waveforms**

```bash
python scripts/run_unitmatch_pair.py \
  --config config/unitmatch_sessions.yml \
  --waveform-source kilosort \
  --use-iti
```

**Option B – ITI-only raw extraction (new)**

```bash
python scripts/run_unitmatch_iti.py \
  --config config/unitmatch_sessions.yml \
  --use-iti \
  --iti-max-spikes 100 \
  --iti-window-mode uniform \
  --iti-max-windows 50 \
  --iti-cache-dir table_output/unitmatch/iti_cache
```

`run_unitmatch_iti.py` loads the session pickles, extracts ITI windows from `Trial.ITI`/`Baseline_ON`, pulls raw snippets directly from the SpikeGLX `*.ap.bin`, and feeds the resulting templates into UnitMatch. Use this when Bombcell waveforms are unavailable or when you want to follow the paper's ITI-only recommendation for striatum. Set `--iti-window-mode uniform` (with `--iti-max-windows`) to subsample windows evenly and keep runtimes reasonable; omit the flag to process all ITIs. Progress bars are shown by default and can be disabled via `--no-progress`. Heavy ITI extractions are cached per session (`--iti-cache-dir`, `--no-cache`), so once a session's parameters/inputs are unchanged the script reloads cached waveforms instead of re-reading the raw binary.

**Output (both runners):**
- `table_output/unitmatch/unitmatch_pair_matches.csv` (unit pairs with match probabilities)
- `table_output/unitmatch/unitmatch_pair_diagnostic.json` (debugging info)

### 3. Run Batch Tracking

Track units across all sessions:

```bash
python scripts/unitmatch/run_unitmatch_batch.py \
    --config config/unitmatch_sessions.yml \
    --waveform-dir png_output/unitmatch_waveforms \
    --prob-threshold 0.5
```

**Options:**
- `--prob-threshold`: Probability threshold for matches (default: 0.5 from paper)
- `--no-neighbor-check`: Disable neighboring recordings check
- `--sessions`: Process specific sessions only

**Output:**
- `table_output/unitmatch/tracking_chains.csv` with columns:
  - `track_id`: Unique ID for tracked neuron
  - `session_id`, `session_name`: Session information
  - `unit_id`: Original Kilosort cluster ID
  - `max_match_prob`: Best match probability within track

### 4. Validate Results

Compute ISI stability and waveform correlations:

```bash
python scripts/qc_checks/validate_unitmatch_results.py \
    --tracking table_output/unitmatch/tracking_chains.csv \
    --sessions data/*.pkl \
    --waveform-dir png_output/unitmatch_waveforms \
    --waveform-source kilosort \
    --use-iti
```

**Output:**
- `png_output/unitmatch_validation/tracking_stability.png` (ISI/waveform stability)
- `png_output/unitmatch_validation/track_length_distribution.png`
- `png_output/unitmatch_validation/validation_summary.yaml` (statistics)

## Configuration

Edit `config/unitmatch_sessions.yml`:

```yaml
sessions:
  - path: "path/to/kilosort/output"
    name: "BG_046_13082025"
    session_pkl: "data/BG_046_13082025.pkl"   # Auto-generated from MAT if missing
    raw_ap: "path/to/kilosort/output/BG_046_13082025_g0_tcat.imec0.ap.bin"
    bombcell_dir: "path/to/bombcell/output"   # Optional, only needed for Bombcell source

waveform_config:
  source: 'both'  # 'kilosort', 'bombcell', or 'both'
  use_iti: true
  iti_method: 'trial_boundaries'  # 'trial_field', 'trial_boundaries', 'fallback'
  fallback_window: [1.0, 3.0]  # Seconds after trial end
  iti_window_mode: 'uniform'  # 'all' or 'uniform'
  max_iti_windows: 50  # Number of ITI windows to sample when using 'uniform'
  compare_sources: true
  max_spikes_per_unit: 100
  min_spikes_per_unit: 80
  min_spikes_per_half: 25
  show_progress: true
  cache_waveforms: true
  iti_cache_dir: "table_output/unitmatch/iti_cache"

spatial_constraints:
  max_cross_shank_distance_um: 350  # For Neuropixels 2.0 (4-shank, 250μm spacing)

matching_params:
  prob_threshold: 0.5  # From paper
  check_neighboring_recordings: true
  allow_disappear_reappear: true
```

## ITI Extraction Methods

**`trial_boundaries`** (recommended):
- Compute trial end from reaction times
- ITI = trial_end → next Baseline_ON
- Most accurate for variable ITI durations

**`trial_field`**:
- Use `Trial.ITI` field directly
- Requires pre-computed ITI values

**`fallback`**:
- Fixed window after trial end: [1.0, 3.0]s
- Robust fallback if trial boundaries unavailable

## Waveform Sources

### Kilosort Templates (Direct Extraction)

**Pros:**
- No Bombcell dependency
- Uses cross-validation (first/second half splits)
- Paper validated: "extracted either through Bombcell or through Unitmatch's ExtractAndSaveAverageWaveforms.m"

**Method:**
1. Load `templates.npy` (n_templates × n_samples × n_channels)
2. Map spikes → templates via `spike_templates.npy`
3. Compute per-cluster mean waveforms with CV splits

### Bombcell Pre-computed

**Pros:**
- Pre-computed with Bombcell QC
- Includes quality metrics

**Cons:**
- Requires Bombcell run (many sessions lack this)
- ITI filtering not supported (pre-computed from all spikes)

### Comparison Mode (`source='both'`)

Extract both and generate correlation plots to empirically determine which performs better.

## Algorithm Details

From Windolf et al. 2024:

> "The default version of the algorithm iteratively inspects all pairs, and merges a unit with a target group if its probability of matching with all of the units in the target group that are within the recording and in neighboring recordings is higher than 0.5."

**Implementation:**
1. Compute pairwise match probabilities (Bayesian combination of waveform/spatial/temporal features)
2. Sort pairs by probability (descending)
3. Iteratively build tracking groups:
   - Check prob > threshold with all units in target group
   - Check neighboring sessions (±1)
   - Allow 1-to-1 matching per session
   - Allow disappear/reappear across sessions

## Validation Metrics

### ISI Fingerprint Stability

Compute Euclidean distance between ISI histograms across sessions for tracked units.

**Expected:** Low distances for true matches (paper Fig 5)

### Waveform Correlation

Compute correlation between waveforms across sessions.

**Expected:** High correlations (>0.8) for true matches (paper Fig 5)

### Striatal Learning Validation

From paper Fig 6:
- ISI histograms remain stable despite learning
- Stimulus-evoked responses change (plasticity)
- Validates tracking during striatal learning

## Parameter Recommendations

From paper validation (dorsomedial striatum, visuomotor task):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `prob_threshold` | 0.5 | Default works well |
| `prob_threshold` (strict) | 0.6-0.7 | Early learning with rapid plasticity |
| `use_iti` | true | Recommended for task areas |
| `iti_method` | `trial_boundaries` | Most accurate |
| `fallback_window` | [1.0, 3.0] | Avoid trial edges |
| `check_neighboring` | true | Always for chronic recordings |
| `max_cross_shank_distance_um` | 350 | Neuropixels 2.0 (250μm spacing) |

## Troubleshooting

### Few Tracked Units

- Lower `prob_threshold` (0.5 → 0.4)
- Compare Kilosort vs Bombcell sources
- Check ITI coverage (should be ~20-40% of spikes)

### Poor ISI Stability

- Use `--use-iti` to avoid stimulus artifacts
- Check electrode drift (spatial constraints)
- Verify good cluster QC

### Low Waveform Correlations

- Compare Kilosort vs Bombcell
- Check probe movement (chronic implant validation)
- Verify channel map

### No Bombcell Waveforms

- Use `waveform_source='kilosort'`
- Paper validated both methods work!

## Code Structure

```
src/unit_tracking.py
├── extract_iti_spikes()                # ITI period extraction
├── extract_waveforms_from_kilosort()   # Main waveform extraction
├── _extract_from_kilosort_templates()  # Kilosort method
└── _extract_from_bombcell()            # Bombcell method

scripts/
├── prepare_waveforms_for_unitmatch.py  # Waveform extraction + comparison
├── run_unitmatch_pair.py               # Pairwise testing
├── run_unitmatch_batch.py              # Batch tracking
└── validate_unitmatch_results.py       # Validation metrics

notebooks/
└── unitmatch_workflow.ipynb            # Interactive tutorial

config/
└── unitmatch_sessions.yml              # Configuration
```

## References

Windolf, J., Schneider, M., Schröder, S., Keller, A., Churchland, A. K., & Paninski, L. (2024). Robust single-neuron tracking across long timescales and spatial distances. *Nature Methods*.

Key findings for this implementation:
- Fig 5: ISI stability and waveform correlations validate tracking
- Fig 6: Striatal tracking during learning (stable ISI, changing responses)
- Default prob > 0.5 threshold works well
- Waveforms extracted from Kilosort or Bombcell both validated

## Example Workflow

```bash
# 1. Extract waveforms from both sources with ITI filtering
python scripts/unitmatch/prepare_waveforms_for_unitmatch.py \
    --source both --use-iti --compare

# 2. Test on first pair
python scripts/run_unitmatch_pair.py \
    --waveform-source kilosort --use-iti

# 3. Run full tracking
python scripts/unitmatch/run_unitmatch_batch.py \
    --prob-threshold 0.5

# 4. Validate results
python scripts/qc_checks/validate_unitmatch_results.py \
    --tracking table_output/unitmatch/tracking_chains.csv \
    --sessions data/*.pkl \
    --waveform-source kilosort --use-iti

# 5. Compare Kilosort vs Bombcell tracking performance
python scripts/qc_checks/validate_unitmatch_results.py \
    --waveform-source bombcell --use-iti \
    --output-dir png_output/unitmatch_validation_bombcell

# Compare validation_summary.yaml from both runs
```

## Next Steps

1. **Extract waveforms** for your sessions with comparison mode
2. **Review comparison plots** to choose best source
3. **Run batch tracking** with default parameters
4. **Validate results** with ISI/waveform metrics
5. **Analyze tracked units** during striatal learning!
