# Data Management & Quality Control Manual

This manual covers the foundational scripts for converting data, managing the session inventory, and verifying data integrity.

## 1. Data Conversion (MATLAB → Python)
**Script**: `scripts/conversion/convert_mat_to_pkl.py`
**Batch Script**: `scripts/batch_processing/batch_convert_MatToPkl.py`

**Purpose**:
Converts legacy `.mat` files (exported from the acquisition pipeline) into the standardized Python `Session` object (`.pkl` format) used by all analysis tools in this repo.

**Usage (Single File)**:
```bash
python scripts/conversion/convert_mat_to_pkl.py data/raw/BG_046_15082025.mat --out data/pkls/BG_046/BG_046_15082025.pkl
```

**Usage (Batch)**:
```bash
python scripts/batch_processing/batch_convert_MatToPkl.py --mat-dir E:/raw_data/BG_046 --out-dir pkls/BG_046 --workers 4
```

---

## 2. Session Manifest & Staging
**Script**: `scripts/analysis/stage_sessions.py`

**Purpose**:
The **most critical step** in the workflow. It scans the pickle folder, calculates performance metrics (d', Hit Rate), applies QC rules (e.g., minimum trial counts), and defines the "Stage" (Naive/Learning/Expert).

**Usage**:
```bash
python scripts/analysis/stage_sessions.py --subject_dir data/pkls/BG_046 --output data/BG_046_staging_manifest.csv
```
*   **Output**: A CSV file used as the input `--manifest` for almost all other batch scripts.
*   **Key Logic**:
    *   Excludes sessions with < 20 Go/Catch trials.
    *   Classifies based on d' quantiles (Naive < Q25, Expert > Q75).

---

## 3. Data Integrity & Management
**Script**: `scripts/data_management/check_backup_redundancy.py`
**Purpose**: Verifies that files in the active workspace have backups in the archive location.
**Usage**: `python scripts/data_management/check_backup_redundancy.py`

**Script**: `scripts/data_management/migrate_pkls.py`
**Purpose**: Helper to move `.pkl` files into the structured `BG_XXX` folder hierarchy.

---

## 4. Quality Control (QC) Checks
**Script**: `scripts/QC_CHECKS/inspect_session.py`
**Purpose**: Quick sanity check of a single `.pkl` file. Generates a JSON summary and a spike count histogram.
**Usage**:
```bash
python scripts/QC_CHECKS/inspect_session.py data/pkls/BG_046/BG_046_17092025.pkl contents_check_output/
```

**Script**: `scripts/QC_CHECKS/check_alignment_times.py`
**Purpose**: Verifies that NI-DAQ event times (trials, licks) align logically with the session start/end.

**Script**: `scripts/QC_technical/validate_photodiode_sync.py`
**Purpose**: Checks the timing of the photodiode flip relative to the trial start trigger to ensure visual stimulus latency is consistent.
