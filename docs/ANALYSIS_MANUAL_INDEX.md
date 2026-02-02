# Analysis Manual Index

This directory contains detailed manuals for each component of the analysis pipeline. Use this index to navigate to the specific instructions for the analysis you want to run.

## 📚 Detailed Manuals
1.  **[Data Management & QC](manuals/MANUAL_Data_QC.md)**
    *   *Topics*: Importing `.mat` files to `.pkl`, session staging, creating manifests, integrity checks.
    *   *Key Scripts*: `convert_mat_to_pkl.py`, `stage_sessions.py`

2.  **[Lick & Learning Analysis](manuals/MANUAL_Lick_Learning.md)**
    *   *Topics*: Lick responsiveness (stats generation), Early vs Late FA comparisons, Chronological progression.
    *   *Key Scripts*: `batch_run_lick_analysis.py`, `compare_early_late_fa.py`

3.  **[Visual (TF) Analysis](manuals/MANUAL_TF_Analysis.md)**
    *   *Topics*: Visual responsiveness mapping, identifying "splitter" neurons (TF/Orientation selective).
    *   *Key Scripts*: `batch_run_tf_analysis.py`

4.  **[Behavior Analysis](manuals/MANUAL_Behavior.md)**
    *   *Topics*: Psychometric curves, Hit/FA rates, Learning curves.
    *   *Key Scripts*: `run_behavior_pipeline.py`, `plot_learning_curve.py`

---

## 🚀 Quick Start: The "Golden Path"
To fully process a subject (`BG_046`) from scratch, generating all artifacts:

### 1. Stage Data (Common Requisite)
Create the master manifest that classifies sessions and standardizes paths.
```bash
python scripts/analysis/stage_sessions.py \
  --subject_dir data/pkls/BG_046 \
  --output data/BG_046_staging_manifest.csv
```

### 2. Run Batch Pipelines (Can run in parallel)
Run the three main analysis pillars (Lick, Behavior, Visual) for all sessions.

**A. Lick Stats & Plots**
```bash
python scripts/analysis/lick/batch_run_lick_analysis.py \
  --manifest data/BG_046_staging_manifest.csv \
  --pkl-dir data/pkls/BG_046 \
  --out FIGURES/lick/BG_046 \
  --stats-only # Remove this flag to generate full heatmap/raster plots
```

**B. Behavior Reports**
```bash
python scripts/analysis/behavior/batch_run_behavior.py \
  --manifest data/BG_046_staging_manifest.csv \
  --pkl-dir data/pkls/BG_046 \
  --out FIGURES/behavior/BG_046
```

**C. Visual Responsiveness (TF)**
```bash
python scripts/analysis/tf_response/batch_run_tf_analysis.py \
  --manifest data/BG_046_staging_manifest.csv \
  --pkl-dir data/pkls/BG_046 
```

### 3. Run Group/Questions Analyses
After the batch steps above are done, run specific scientific question scripts.

**Learning Analysis (Early vs Late FAs)**
```bash
python scripts/analysis/learning/compare_early_late_fa.py \
  --manifest data/BG_046_staging_manifest.csv \
  --stats-root FIGURES/lick/BG_046 \
  --out-dir FIGURES/learning_fa_split
```

---

## 📂 Directory Structure & Key Paths

*   **`src/visdetect/`**: The Python package containing reusable logic.
    *   `core/`: Data definitions (`Session`, `Trial`, `Cluster`).
    *   `analysis/`: Scientific algorithms (e.g., `lick.py` for defining responsiveness).
    
*   **`scripts/`**: Executable scripts organized by topic.
    *   `conversion/`: Raw data import tools.
    *   `analysis/lick/`: Lick-related batch and plotting scripts.
    *   `analysis/learning/`: Cross-session learning scripts.
    *   `analysis/tf_response/`: Visual response scripts.
    
*   **`data/`**: Analysis inputs.
    *   `pkls/`: The standardized `.pkl` session files.
    *   `*_manifest.csv`: Session lists used to drive batch scripts.
    
*   **`FIGURES/`**: Analysis outputs.
    *    Organized by subject and analysis type (e.g., `FIGURES/lick/BG_046/`).

---

## ❓ Troubleshooting
*   **"Manifest not found"**: Ensure you ran `stage_sessions.py` first. Most scripts depend on this CSV.
*   **"Stats not found"**: If a plotting script fails for specific sessions, run `batch_run_lick_analysis.py --stats-only` again. It skips existing files unless you delete them or force update.
*   **"ImportError"**: Always run scripts from the repo root (e.g., `python scripts/...`) so python can find the `src/` directory.
