# UnitMatch Quick Reference

## One-Command Workflows

### Basic Tracking (Kilosort waveforms, ITI-filtered)
```bash
# Extract waveforms
python scripts/prepare_waveforms_for_unitmatch.py --source kilosort --use-iti

# Track units
python scripts/run_unitmatch_batch.py

# Validate
python scripts/validate_unitmatch_results.py \
    --tracking table_output/unitmatch/tracking_chains.csv \
    --sessions data/*.pkl --waveform-source kilosort --use-iti
```

### Compare Kilosort vs Bombcell
```bash
# Extract both sources with comparison
python scripts/prepare_waveforms_for_unitmatch.py --source both --use-iti --compare

# Track with Kilosort
python scripts/run_unitmatch_batch.py \
    --output table_output/unitmatch/tracking_chains_kilosort.csv

# Track with Bombcell (after extracting bombcell waveforms)
python scripts/run_unitmatch_batch.py \
    --waveform-dir png_output/unitmatch_waveforms_bombcell \
    --output table_output/unitmatch/tracking_chains_bombcell.csv

# Compare validation metrics
python scripts/validate_unitmatch_results.py \
    --tracking table_output/unitmatch/tracking_chains_kilosort.csv \
    --sessions data/*.pkl --waveform-source kilosort --use-iti

python scripts/validate_unitmatch_results.py \
    --tracking table_output/unitmatch/tracking_chains_bombcell.csv \
    --sessions data/*.pkl --waveform-source bombcell --use-iti \
    --output-dir png_output/unitmatch_validation_bombcell
```

## Configuration Quick Edits

Edit `config/unitmatch_sessions.yml`:

```yaml
# Change waveform source
waveform_config:
  source: 'kilosort'  # or 'bombcell' or 'both'

# Adjust ITI settings
waveform_config:
  use_iti: true
  iti_method: 'trial_boundaries'  # most accurate
  fallback_window: [1.0, 3.0]  # seconds after trial

# Change matching threshold
matching_params:
  prob_threshold: 0.5  # lower = more permissive (0.4), higher = stricter (0.6-0.7)
```

## Common Parameter Adjustments

### More Tracked Units
- Lower `prob_threshold`: 0.5 → 0.4
- Disable ITI filtering: `use_iti: false`
- Try different waveform source

### Better Quality Matches
- Raise `prob_threshold`: 0.5 → 0.6 or 0.7
- Enable ITI filtering: `use_iti: true`
- Use both sources and compare

### Striatal Learning (Early Sessions)
- Use `prob_threshold: 0.6` (stricter for plasticity)
- Enable ITI: `use_iti: true`
- Use `iti_method: 'trial_boundaries'`

## Output Files Reference

```
png_output/unitmatch_waveforms/
├── {session}_waveforms_kilosort_iti.npy      # Kilosort, ITI-filtered
├── {session}_waveforms_kilosort_full.npy     # Kilosort, all spikes
├── {session}_waveforms_bombcell_iti.npy      # Bombcell, ITI-filtered
├── comparisons/
│   └── {session}_waveform_comparison.png     # Kilosort vs Bombcell
└── comparison_summary.yaml                    # Correlation statistics

table_output/unitmatch/
├── tracking_chains.csv                        # Main output: tracked units
├── unitmatch_pair_matches.csv                 # Pairwise test results
└── unitmatch_pair_diagnostic.json            # Debugging info

png_output/unitmatch_validation/
├── tracking_stability.png                     # ISI + waveform stability
├── track_length_distribution.png              # Tracking statistics
└── validation_summary.yaml                    # Validation metrics
```

## Troubleshooting One-Liners

```bash
# Check waveform extraction worked
ls png_output/unitmatch_waveforms/*.npy

# Check ITI coverage
python -c "import numpy as np; w=np.load('png_output/unitmatch_waveforms/BG_031_260325_waveforms_kilosort_iti.npy'); print(f'Shape: {w.shape}')"

# Count tracked units
python -c "import pandas as pd; df=pd.read_csv('table_output/unitmatch/tracking_chains.csv'); print(f'Total tracks: {df.track_id.nunique()}, Multi-session: {(df.groupby(\"track_id\").session_id.nunique()>1).sum()}')"

# Check validation metrics
cat png_output/unitmatch_validation/validation_summary.yaml
```

## Key Function Arguments

### prepare_waveforms_for_unitmatch.py
- `--source`: kilosort | bombcell | both
- `--use-iti`: Filter to ITI periods
- `--iti-method`: trial_field | trial_boundaries | fallback
- `--compare`: Generate comparison plots (if source=both)
- `--sessions`: Specific sessions to process

### run_unitmatch_batch.py
- `--prob-threshold`: Match probability threshold (default 0.5)
- `--no-neighbor-check`: Disable neighboring session check
- `--waveform-dir`: Directory with prepared waveforms
- `--output`: Output CSV path

### validate_unitmatch_results.py
- `--tracking`: Path to tracking_chains.csv
- `--sessions`: Paths to session .pkl files
- `--waveform-source`: kilosort | bombcell
- `--use-iti`: Use ITI waveforms
- `--compare-sources`: Compare Kilosort vs Bombcell

## Paper Reference Quick Facts

- **Default threshold**: prob > 0.5 (works well for most cases)
- **Stricter threshold**: prob > 0.6-0.7 (early learning with plasticity)
- **ITI recommended**: For task-related brain areas
- **Validation**: Stable ISI + high waveform correlation = good tracking
- **Striatal learning**: ISI stable despite changing stimulus responses (Fig 6)
- **Waveform sources**: Both Kilosort and Bombcell validated in paper

## Interactive Tutorial

```bash
# Open Jupyter notebook
jupyter notebook notebooks/unitmatch_workflow.ipynb
```

The notebook walks through:
1. Loading configuration
2. Extracting waveforms
3. Comparing sources
4. Running tracking
5. Validating results
6. Striatal learning example
7. Troubleshooting

## Help Commands

```bash
python scripts/prepare_waveforms_for_unitmatch.py --help
python scripts/run_unitmatch_pair.py --help
python scripts/run_unitmatch_batch.py --help
python scripts/validate_unitmatch_results.py --help
```

## Full Documentation

- **Quick Start**: `UNITMATCH_README.md` (sections 1-4)
- **Configuration**: `UNITMATCH_README.md` (Configuration section)
- **Algorithm Details**: `UNITMATCH_README.md` (Algorithm Details section)
- **Troubleshooting**: `UNITMATCH_README.md` (Troubleshooting section)
- **Implementation**: `IMPLEMENTATION_SUMMARY.md`
- **Interactive Tutorial**: `notebooks/unitmatch_workflow.ipynb`
