# Analysis Suite

This directory contains the production analysis scripts that generate the figures and statistics for the visual change-detection project (mouse BG_046).

## Overview

The analysis suite consists of 43 figure-generating scripts organized into 9 modules:
- `01_behavior/` - Behavioral analysis and learning curves (Figs 1-7)
- `02_single_unit/` - Single-unit responses and tuning (Figs 8-12)
- `03_population/` - Population dynamics and coding directions (Figs 13-17)
- `04_decoding/` - Population decoding analyses (Figs 18-20)
- `05_longitudinal/` - Cross-session tracking (Figs 21-23)
- `06_lick_motor/` - Lick-related motor responses (Figs 24-26)
- `07_advanced/` - Advanced analyses (GLM, HMM) (Figs 27-34)
- `08_tf_pulse/` - Temporal frequency pulse responses (Figs 35-42)
- `09_optotagging/` - Optogenetic identification (Fig 43)

## Quick Start

### 1. Setup Environment

The analysis suite requires the `visdetect` package to be installed in development mode:

```bash
# From project root
pip install -e .
```

### 2. Run All Analyses

```bash
# From project root
cd analysis_suite
python run_all.py
```

### 3. Run Individual Scripts

```bash
# Example: Run behavioral learning curve analysis
cd analysis_suite
python 01_behavior/a_learning_curve.py
```

## Output Locations

- **Figures**: `analysis_suite/figures/` (organized by module)
- **Cache**: `analysis_suite/cache/` (intermediate results for reuse)
- **Stats**: `analysis_suite/table_output/` (statistical summaries)

## Key Infrastructure Files

| File | Purpose |
|------|---------|
| `run_all.py` | Master script to run all 43 analyses in sequence |
| `config.py` | Configuration wrapper (re-exports from `visdetect.analysis.config`) |
| `loader.py` | Session loading and data access utilities |
| `utils.py` | Shared analysis utilities (PSTH, population tensors, statistics) |
| `plotting.py` | Plotting utilities and style settings |

## Script Naming Convention

Scripts use alphabetical prefixes to control execution order:
- `a_*.py` - Primary analyses for each module
- `b_*.py` - Secondary analyses
- `c_*.py` - Additional analyses

Example: `01_behavior/a_learning_curve.py` generates Figure 1.

## Dependencies

**Core Requirements**:
- Python 3.8+
- `visdetect` package (this repository)
- Standard scientific stack: `numpy`, `pandas`, `matplotlib`, `scipy`
- `scikit-learn` for decoding analyses

**Data Requirements**:
- Session `.pkl` files in `data/pkls/BG_046/`
- Staging manifest: `data/staging_manifest.csv`
- Optional: Waveform labels, HMM fits, lick traces

## Troubleshooting

### Import Errors
If you see import errors like `ModuleNotFoundError: No module named 'visdetect'`:
```bash
# Ensure editable install
pip install -e .

# Or set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/project/src"
```

### Missing Data Files
Run the data conversion pipeline first:
```bash
# Convert .mat files to .pkl
python scripts/conversion/convert_mat_to_pkl.py

# Generate staging manifest
python scripts/analysis/stage_sessions.py
```

### Cache Issues
Clear the cache if results seem stale:
```bash
rm -rf analysis_suite/cache/*
```

## Performance Notes

- **Full suite runtime**: ~2-4 hours (depends on cache state)
- **Memory usage**: ~4-8 GB RAM for population analyses
- **Parallel execution**: Use `run_all.py --n_workers N` for compatible scripts

## Contributing

When adding new analysis scripts:
1. Follow the naming convention (`module_NN/x_descriptive_name.py`)
2. Include proper docstrings and figure descriptions
3. Use shared utilities from `utils.py` when possible
4. Cache expensive computations in `analysis_suite/cache/`
5. Add to `run_all.py` script mapping

---

*For detailed project documentation, see the main [README.md](../README.md) and [CLAUDE.md](../CLAUDE.md)*