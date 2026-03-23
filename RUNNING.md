# RUNNING.md — How to Run the Analysis Pipeline

This manual documents how to set up and run the full analysis pipeline for the BG_046 visual change-detection project, from raw MATLAB files to publication figures.

---

## Prerequisites

### Environment Setup

```bash
# 1. Create and activate virtual environment (if not already done)
python -m venv .venv
.venv\Scripts\activate      # Windows
# OR: source .venv/bin/activate   # Linux/Mac

# 2. Install the visdetect package in editable mode
pip install -e .

# 3. Verify install
py -c "import visdetect; print('OK')"
```

**Dependencies** (installed automatically): numpy, scipy, pandas, matplotlib, seaborn, tqdm, pyyaml, scikit-learn.

**Additional** (install manually if needed): `h5py` (for legacy .mat loading), `pytest` (for tests).

### Platform Notes

- **Windows + Git Bash**: Use `py` not `python` to invoke Python.
- **RAM**: Sessions are ~100+ MB each. Recommend 16+ GB RAM for population analysis scripts.
- **Disk**: The full pipeline produces ~2 GB of figures, caches, and intermediate data.

---

## Pipeline Overview

```
Step 1: .mat → .pkl conversion
Step 2: Session staging (learning stages)
Step 3: Grand Longitudinal Table
Step 4: HMM behavioral states          ─┐
Step 5: Lick responsiveness analysis    ─┤ (independent, can run in parallel)
Step 6: TF pulse trace cache            ─┤
Step 7: Waveform cell-type labels       ─┘
Step 8: Analysis suite (43 figures)
```

Steps 4-7 are independent of each other (all depend on Steps 1-3). Step 8 depends on all prior steps.

---

## Step-by-Step Instructions

### Step 1: Convert MATLAB to Pickle

Converts raw `.mat` session files to Python Session dataclass pickles.

```bash
py scripts/batch_processing/batch_convert_MatToPkl.py \
    --data-dir data/mat/BG_046 \
    --out-dir data/pkls/BG_046
```

**Flags**: `--force` (overwrite existing), `--dry-run` (preview), `--verbose` (debug logging)

**Produces**: `data/pkls/BG_046/BG_046_DDMMYYYY.pkl` (one per session, ~45 files)

**Note**: BG_046_07072025.mat has no NPX_probes and will be skipped — this is expected.

### Step 2: Stage Sessions

Computes SDT metrics per session, applies QC gates, and assigns learning stages (Naive → Learning → Expert).

```bash
py scripts/analysis/stage_sessions.py \
    --subject_dir data/pkls/BG_046 \
    --subject_name BG_046 \
    --output data/BG_046_staging_manifest.csv
```

**Produces**: `data/BG_046_staging_manifest.csv`

**QC gates applied**:
1. Minimum go trials (≥20) and catch trials (≥10)
2. Total go+catch ≥ 100
3. Engagement: hit_rate ≥ 0.10 OR fa_rate ≥ 0.10
4. d' ≥ 0.8 (default; override with `--dprime-threshold`)

**Stage transitions**: Naive→Learning at d'>1.0 (3/4 window), Learning→Expert at d'>1.5.

### Step 3: Build Grand Longitudinal Table

Integrates UnitMatch tracking, physiology, behavior, and QC metrics into a per-(unit, session) table.

```bash
py scripts/analysis/build_longitudinal_table.py --n_workers 6
```

**Produces**: `table_output/Grand_Longitudinal_Table.csv`

### Step 4: Fit HMM Behavioral States

Fits Bernoulli GLM-HMM models (K=2..5) for trial-by-trial behavioral state decoding.

```bash
# 4a. Fit the HMM
py scripts/analysis/behavior/fit_behavioral_hmm.py \
    --manifest data/BG_046_staging_manifest.csv \
    --pkl-dir data/pkls/BG_046 \
    --out FIGURES/behavior/BG_046/hmm \
    --data-out data/hmm/BG_046 \
    --n-workers 12

# 4b. Compute per-session state metrics
py scripts/analysis/behavior/hmm_behavioral_states.py \
    --data-dir data/hmm/BG_046 \
    --manifest data/BG_046_staging_manifest.csv \
    --out FIGURES/behavior/BG_046/hmm/behavioral_states \
    --data-out data/hmm/BG_046
```

**Produces** (in `data/hmm/BG_046/`):
- `state_assignments_K3.csv` — per-trial HMM states (primary input for analysis suite)
- `per_session_state_metrics.csv` — per-session d'/criterion by state
- `learning_trajectory.csv` — state fraction evolution across sessions
- `model_selection.csv` — BIC/AIC for each K

**Used by**: Figs 02, 11, 16, 20, 30, 39

### Step 5: Lick Responsiveness Analysis

Detects neurons responsive to FA (anticipatory) lick events using SALT-style analysis.

```bash
py scripts/analysis/lick/batch_run_lick_analysis.py \
    --pkl-dir data/pkls/BG_046 \
    --out FIGURES/lick/BG_046 \
    --workers 4
```

**Produces** (per session in `FIGURES/lick/BG_046/<DDMMYYYY>/`):
- `lick_responsiveness.csv` — per-unit significance, p-values
- `lick_responsiveness.npz` — PETHs for plotting

**Used by**: Figs 24-26 (lick/motor module)

### Step 6: TF Pulse Trace Cache

Pre-computes z-scored TF pulse response traces for all units across all sessions.

```bash
py scripts/rebuild_tf_cache.py --workers 6 --qc-only
```

**Produces**: `data/cache/tf_traces/BG_046/BG_046_DDMMYYYY_traces.npz` (per session)

**Used by**: Figs 35-42 (TF pulse module)

### Step 7: Waveform Cell-Type Labels

Classifies units as Narrow (FSI) or Broad (MSN/Projecting) based on trough-to-peak duration.

```bash
py scripts/pipelines/concat_sort/regen_waveform_labels.py
```

**Produces**: `AI_exploration/figures/waveform_celltype_labels.csv`

**Used by**: Figs 12, 22 (cell-type analyses)

### Step 8: Run the Analysis Suite

After all prerequisites are complete:

```bash
cd analysis_suite

# Run all 43 scripts sequentially
py run_all.py

# Or with parallelism for heavy scripts (recommended)
py run_all.py --n_workers 4

# Or run a single figure
py 03_population/a_coding_direction.py

# Or run a single module
py 01_behavior/a_learning_curve.py
py 01_behavior/b_hmm_state_dynamics.py
py 01_behavior/c_reaction_time_analysis.py
# ... etc.
```

**Output**: Figures saved to `analysis_suite/figures/<module>/`, stats CSVs alongside figures.

**Log**: `analysis_suite/run_all_log.txt`

**Timeout**: 30 minutes per script (in `run_all.py`).

---

## Figure Inventory (43 figures across 9 modules)

| Fig | Script | Title | Key Dependencies |
|-----|--------|-------|-----------------|
| **01_behavior** | | | |
| 01 | `a_learning_curve.py` | Learning curve & psychometrics | Manifest |
| 02 | `b_hmm_state_dynamics.py` | HMM behavioral states | HMM assignments |
| 03 | `c_reaction_time_analysis.py` | Reaction time distributions | Manifest |
| 04 | `d_post_error_psychometric.py` | Post-error psychometric shifts | Manifest |
| 05 | `e_post_error_dynamics.py` | Post-error behavioral dynamics | HMM assignments |
| 06 | `f_post_error_controls.py` | Post-error control analyses | HMM assignments |
| 07 | `g_post_error_streak_controls.py` | Post-error streak effects | HMM assignments |
| **02_single_unit** | | | |
| 08 | `a_responsiveness_screen.py` | Unit responsiveness screening | PKL, GLT |
| 09 | `b_outcome_selectivity.py` | Hit/Miss selectivity per unit | PKL, GLT |
| 10 | `c_change_size_tuning.py` | Change-size tuning curves | PKL, GLT |
| 11 | `d_state_modulation.py` | HMM state × neural activity | HMM assignments |
| 12 | `e_cell_type_comparison.py` | FSI vs MSN comparison | Waveform labels |
| **03_population** | | | |
| 13 | `a_coding_direction.py` | Hit/Miss coding direction | PKL |
| 14 | `b_population_psth_heatmap.py` | Population PSTH heatmaps | PKL, GLT |
| 15 | `c_dimensionality_reduction.py` | PCA of population activity | PKL |
| 16 | `d_state_matched_cd.py` | State-matched coding direction | HMM assignments |
| 17 | `e_sensory_dose_response.py` | Dose-response population curves | PKL |
| **04_decoding** | | | |
| 18 | `a_hit_miss_decoding.py` | Hit vs Miss neural decoding | PKL |
| 19 | `b_change_size_decoding.py` | Change-size neural decoding | PKL |
| 20 | `c_state_decoding.py` | HMM state neural decoding | HMM assignments |
| **05_longitudinal** | | | |
| 21 | `a_neural_learning_curves.py` | Neural selectivity across learning | GLT |
| 22 | `b_celltype_learning.py` | Cell-type trajectories | GLT, Waveform labels |
| 23 | `c_population_geometry_shift.py` | Pop. geometry across learning | PKL |
| **06_lick_motor** | | | |
| 24 | `a_fa_neural_signatures.py` | FA-aligned neural signatures | Lick responsiveness |
| 25 | `b_pre_lick_ramping.py` | Pre-lick neural ramping | Lick responsiveness |
| 26 | `c_motor_vs_sensory.py` | Motor vs sensory dissociation | Lick responsiveness |
| **07_advanced** | | | |
| 27 | `a_glm_encoding.py` | GLM encoding models | PKL, HMM |
| 28 | `b_dpca.py` | Demixed PCA | PKL |
| 29 | `c_noise_correlations.py` | Noise correlation analysis | PKL |
| 30 | `d_impulsivity_regression.py` | Impulsivity regression models | HMM assignments |
| 31 | `e_trial_outcome_prediction.py` | Trial outcome prediction | HMM assignments |
| 32 | `f_fa_subtype_lick_triggered_tf.py` | FA subtype lick-triggered TF | TF traces, Lick |
| 33 | `g_fa_subtype_prediction.py` | FA subtype prediction | TF traces, Lick |
| 34 | `h_second_pulse_analysis.py` | Second TF pulse analysis | TF traces |
| **08_tf_pulse** | | | |
| 35 | `a_tf_responsiveness.py` | TF pulse responsiveness screen | TF traces |
| 36 | `b_tf_response_properties.py` | TF response properties | TF traces |
| 37 | `c_tf_pulse_integration.py` | Two-pulse temporal integration | TF traces |
| 38 | `d_tf_learning_emergence.py` | TF response learning emergence | TF traces, GLT |
| 39 | `e_tf_state_modulation.py` | TF × HMM state interaction | TF traces, HMM |
| 40 | `f_tf_sensory_motor.py` | TF sensory-motor responses | TF traces |
| 41 | `g_tf_cell_classifier.py` | TF cell-type classifier | TF traces |
| 41g | `g2_tf_tier_gallery.py` | TF tier example gallery | TF traces |
| 42 | `h_tf_post_error_modulation.py` | TF post-error modulation | TF traces, HMM |
| **09_optotagging** | | | |
| 43 | `a_optotagging_identification.py` | D1/D2 SALT identification | PKL (laser events) |

---

## Cache Management

Most analysis suite scripts cache intermediate results to avoid recomputing on every run.

| Cache Location | What | Rebuild |
|----------------|------|---------|
| `analysis_suite/cache/*.csv` | Per-script analysis results | Delete the CSV and rerun the script |
| `data/cache/tf_traces/` | TF pulse z-scored traces | `py scripts/rebuild_tf_cache.py` |
| `data/hmm/BG_046/` | HMM model fits and assignments | Rerun Step 4 |
| `FIGURES/lick/BG_046/` | Lick responsiveness per session | Rerun Step 5 |

To force a full cache rebuild for all analysis suite scripts, delete the entire `analysis_suite/cache/` directory:

```bash
rm -rf analysis_suite/cache/
cd analysis_suite && py run_all.py
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `0 sessions` / `No data` | Missing staging manifest | Run Step 2 |
| `FileNotFoundError: *.pkl` | PKL files not converted | Run Step 1 |
| `KeyError` in HMM scripts | HMM not fitted | Run Step 4 |
| `0 TF-responsive units` | TF cache missing | Run Step 6 |
| `No waveform labels found` | Waveform CSV missing | Run Step 7 |
| `insufficient data` for a session | Session has too few units or trials after QC | Expected — some sessions are excluded by quality filters |
| `MemoryError` | Session too large | Ensure `del sess; gc.collect()` in loops. Close other programs. |
| `ModuleNotFoundError: visdetect` | Package not installed | Run `pip install -e .` from project root |
| Script timeout (>30 min) | Heavy computation | Use `--n_workers 4` for parallelism-aware scripts |

---

## Quick Reference

```bash
# Full pipeline from scratch
pip install -e .
py scripts/batch_processing/batch_convert_MatToPkl.py --data-dir data/mat/BG_046 --out-dir data/pkls/BG_046
py scripts/analysis/stage_sessions.py --subject_dir data/pkls/BG_046 --subject_name BG_046 --output data/BG_046_staging_manifest.csv
py scripts/analysis/build_longitudinal_table.py --n_workers 6
py scripts/analysis/behavior/fit_behavioral_hmm.py --manifest data/BG_046_staging_manifest.csv --pkl-dir data/pkls/BG_046 --out FIGURES/behavior/BG_046/hmm --data-out data/hmm/BG_046 --n-workers 12
py scripts/analysis/behavior/hmm_behavioral_states.py --data-dir data/hmm/BG_046 --manifest data/BG_046_staging_manifest.csv --out FIGURES/behavior/BG_046/hmm/behavioral_states --data-out data/hmm/BG_046
py scripts/analysis/lick/batch_run_lick_analysis.py --pkl-dir data/pkls/BG_046 --out FIGURES/lick/BG_046 --workers 4
py scripts/rebuild_tf_cache.py --workers 6 --qc-only
py scripts/pipelines/concat_sort/regen_waveform_labels.py
cd analysis_suite && py run_all.py --n_workers 4
```
