# Behavior Analysis Manual

Scripts for analyzing behavioral performance (Psychometrics, Hit Rates, Reaction Times) independent of neural data.

## 1. Session Behavior
**Script**: `scripts/analysis/behavior/run_behavior_pipeline.py`

**Purpose**:
Generates a standard behavior report for a single session, including:
*   Psychometric Curve (Prob. Hit vs Change Magnitude).
*   Reaction Time Histogram.
*   Trial Outcome Counts.

**Usage**:
```bash
python scripts/analysis/behavior/run_behavior_pipeline.py \
  --session BG_046_17092025 \
  --pkl data/pkls/BG_046/BG_046_17092025.pkl \
  --out FIGURES/behavior/BG_046_17092025
```

---

## 2. Cross-Session Analysis
**Script**: `scripts/analysis/plot_learning_curve.py` (or variants in `scripts/analysis/behavior/`)

**Purpose**:
Plots behavioral metrics (d', Hit Rate, FA Rate) as a function of time/session index to visualize learning.

**Usage**:
```bash
python scripts/analysis/plot_learning_curve.py --manifest data/BG_046_staging_manifest.csv
```

**Script**: `scripts/batch_processing/build_manifest_and_behavior_summary.py`
**Purpose**:
Legacy script often used to perform an initial scan of all pickle files to build a raw summary CSV (before the Staging/QC step).
