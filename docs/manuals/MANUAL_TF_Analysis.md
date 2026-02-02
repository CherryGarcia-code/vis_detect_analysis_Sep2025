# Visual Responsiveness (TF) Analysis Manual

This set of scripts analyzes the neural response to the "TF Pulse" (Temporal Frequency mapping) stimulus to determine which units are visually responsive.

## 1. TF Analysis Pipeline
**Script**: `scripts/analysis/tf_response/batch_run_tf_analysis.py`

**Purpose**:
Runs the visual responsiveness analysis for multiple sessions. It identifies "Splitters" (units that differentiate between specific TF/Ori conditions) and "Responsive" units.

**Usage**:
```bash
python scripts/analysis/tf_response/batch_run_tf_analysis.py \
  --manifest data/BG_046_staging_manifest.csv \
  --pkl-dir data/pkls/BG_046
```
**Output**: 
Creates `FIGURES/tf/<session>/` containing:
*   `tf_grid_zscore.csv`: Z-scores for every condition.
*   `top_splitters.json`: List of units that best distinguish conditions.

---

## 2. Visualization & Summary
**Script**: `scripts/batch_processing/batch_plot_tf_grids.py`
**Purpose**: Generates grid plots (Heatmaps) of visual responses across orientation/frequency space for all processed sessions.

**Script**: `scripts/analysis/tf_response/barplot_top_splitters.py`
**Purpose**: Aggregates the number of significant "splitter" units across sessions to show if visual discrimination improves with learning.

**Script**: `scripts/analysis/tf_response/extract_top_clusters.py`
**Purpose**: Helper to pull out the Cluster IDs of the best visual units for use in other analyses (like decoding).

---

## 3. Underlying Logic
*   `run_tf_analysis_pipeline.py`: The single-session driver.
*   `find_splitters_from_tf_grid_csv.py`: Statistical logic for identifying condition-selective units.
