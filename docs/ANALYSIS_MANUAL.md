# Analysis Manual and Workflow Guide

This document explains the organization of analysis scripts in the `vis_detect_analysis_Sep2025` project and defines the standard workflow for processing a subject's data from raw files to final figures.

## Environment Setup
Ensure your conda environment is active before running any scripts:
```bash
conda activate copilot_ephys
```

## Workflow Overview

The analysis follows a linear dependency chain:
1. **Data Staging**: Inventory sessions and categorize by performance (Naive/Learning/Expert).
2. **Lick Analysis (Stats)**: Compute lick responsiveness for all units in all sessions.
3. **Advanced Analyses**: Run specific questions (Group dynamics, Learning progression) using the staged data and computed stats.

---

## 1. Data Staging
**Script**: `scripts/analysis/stage_sessions.py`

**Purpose**: 
Scans a subject's pickle folder, computes basic psychometrics (Hit Rate, FA Rate, d'), applies Quality Control (QC), and assigns a learning stage (Naive, Learning, Expert).

**Usage**:
```bash
python scripts/analysis/stage_sessions.py --subject_dir data/pkls/BG_046 --output data/BG_046_staging_manifest.csv
```

**Output**:
*   `data/<subject>_staging_manifest.csv`: The master list of sessions used by downstream scripts.

---

## 2. Lick Analysis (Batch Processing)
**Script**: `scripts/analysis/lick/batch_run_lick_analysis.py`

**Purpose**: 
Runs the lick analysis pipeline on multiple sessions. Can run in two modes:
1.  `--stats-only`: Fast. Generates `lick_responsiveness.csv` (unit stats) and `.npz` files needed for group analysis.
2.  (Default): Full Report. Generates heatmaps, rastes, and a PDF summary for every session.

**Usage (Stats Generation - Recommended First Step)**:
```bash
python scripts/analysis/lick/batch_run_lick_analysis.py \
  --manifest data/BG_046_staging_manifest.csv \
  --pkl-dir data/pkls/BG_046 \
  --out FIGURES/lick/BG_046 \
  --stats-only
```

**Usage (Full Visualization)**:
```bash
# Omit --stats-only to generate all plots
python scripts/analysis/lick/batch_run_lick_analysis.py ... 
```

**Key Output**:
*   `FIGURES/lick/<details>/lick_responsiveness.csv`: The "database" of unit types (Excited/Inhibited) used by other scripts.

---

## 3. Group Analyses & Learning
Once steps 1 and 2 are complete, you can run these scripts in any order to generate paper figures.

### A. Learning Progression (Early vs Late FAs)
**Script**: `scripts/analysis/learning/compare_early_late_fa.py`

**Purpose**:
Compares neural activity during "Impulsive" (Early < 3s) vs "Derived" (Late > 3s) False Alarms, splitting sessions by the learning stage (Naive vs Expert).

**Usage**:
```bash
python scripts/analysis/learning/compare_early_late_fa.py \
  --manifest data/BG_046_staging_manifest.csv \
  --stats-root FIGURES/lick/BG_046 \
  --out-dir FIGURES/learning_fa_split
```

### B. FA Neural Stratification (Single vs Multi Lick)
**Script**: `scripts/analysis/lick/plot_fa_neural_stratified.py`

**Purpose**:
Investigates if neural activity differs when the animal licks once (Single) vs multiple times (Multi) during a False Alarm.

**Usage**:
```bash
python scripts/analysis/lick/plot_fa_neural_stratified.py \
  --manifest data/BG_046_staging_manifest.csv \
  --stats-root FIGURES/lick/BG_046 \
  --out FIGURES/lick_stratified
```

### C. Chronological Progression (Example)
**Script**: `scripts/analysis/lick/plot_chronological_progression.py`

**Purpose**:
Plots the evolution of lick responsiveness amplitude across days.

---

## Script Inventory & Reference

| Logic Layer | Path | Description |
| :--- | :--- | :--- |
| **Core** | `src/visdetect/core/session.py` | Defines `Session`, `Trial`, `Cluster` data structures. |
| **Lick Logic** | `src/visdetect/analysis/lick.py` | Scientific logic for defining lick responsive units (mirroring MATLAB). |
| **Pipeline** | `scripts/analysis/lick/run_lick_analysis_pipeline.py` | The driver script that runs everything for ONE session. |
| **Helper** | `scripts/analysis/lick/find_lick_responsive_neurons.py` | Extracts valid units and computes significance/stats. |

## Adding New Analyses
1.  **Prototype** in `notebooks/` to validate logic.
2.  **Move Logic** to a standardized script in `scripts/analysis/<topic>/`.
3.  **Use Manifest**: Always read `data/<subject>_staging_manifest.csv` to ensure consistent session inclusion criteria.
4.  **Use Stats**: Reuse `lick_responsiveness.csv` rather than re-calculating significance inside the plotting script.
