# BG_046 Data Preparation & Validation Pipeline

This guide walks through the complete workflow for preparing BG_046 sessions for analysis, including data conversion, quality validation, and synchronization checks.

## Quick Start

### 1. Convert All .mat Files to .pkl Format

```bash
# Dry run first (see what would be converted)
python scripts/batch_convert_bg046.py --dry-run

# Convert all BG_046 sessions
python scripts/batch_convert_bg046.py

# Force overwrite existing .pkl files
python scripts/batch_convert_bg046.py --force

# Convert different subject
python scripts/batch_convert_bg046.py --subject BG_031
```

**Output**: `.pkl` files alongside each `.mat` file in `data/` directory

---

### 2. Run Full Validation Suite

```bash
# Validate all BG_046 sessions with HTML report
python scripts/run_validation_suite.py --subject BG_046 --html-report

# Validate single session
python scripts/run_validation_suite.py --session data/BG_046_15082025.pkl

# Verbose logging to file
python scripts/run_validation_suite.py --subject BG_046 --html-report --verbose
```

**Outputs**:
- `table_output/validation/validation_report_YYYYMMDD_HHMMSS.html` - Interactive HTML report
- `table_output/validation/validation_summary.csv` - CSV summary table
- `table_output/validation/validation_YYYYMMDD_HHMMSS.log` - Detailed log file

**Validation Checks**:
- ✓ Session completeness (subject, trials, clusters, ni_events)
- ✓ Trial integrity (outcomes, change_size, ITI distributions)
- ✓ Spike data quality (counts, good cluster IDs)
- ✓ NI-IMEC duration consistency (≤0.5s deviation)
- ✓ Photodiode-FSM synchronization (optional)

---

### 3. Individual Validation Components

#### a. Check NI-IMEC Duration Consistency

```bash
# Single session
python scripts/validate_metadata_duration.py --session data/BG_046_15082025.pkl

# Batch processing
python scripts/validate_metadata_duration.py --batch "data/BG_046_*.pkl"

# Save results to CSV
python scripts/validate_metadata_duration.py --batch "data/BG_046_*.pkl" \
    --output-csv table_output/duration_validation.csv
```

**Pass Criteria**: Deviation ≤ 0.5 seconds (configurable with `--threshold`)

---

#### b. Validate Photodiode-FSM Sync

```bash
# Single session with diagnostic plot
python scripts/validate_photodiode_sync.py \
    --session data/BG_046_15082025.pkl \
    --plot

# Batch with plots
python scripts/validate_photodiode_sync.py \
    --batch "data/BG_046_*.pkl" \
    --plot \
    --plot-dir png_output/sync_validation/

# Adjust matching tolerance
python scripts/validate_photodiode_sync.py \
    --batch "data/BG_046_*.pkl" \
    --max-lag 0.05  # 50ms tolerance
```

**Outputs**:
- Console report with sync quality ratings
- Diagnostic plots: `png_output/sync_validation/<session>_photodiode_sync.png`
- CSV summary (use `--output-csv`)

**Sync Quality Ratings**:
- `excellent`: <1ms max drift
- `good`: 1-10ms max drift
- `acceptable`: 10-50ms max drift
- `poor`: >50ms max drift

---

#### c. Analyze Video Frame Offsets

```bash
# Single video metadata CSV
python scripts/analyze_video_frame_offset.py \
    --video-csv path/to/session/video_metadata.csv

# Entire session directory
python scripts/analyze_video_frame_offset.py \
    --session-dir path/to/session_directory/

# Batch from manifest
python scripts/analyze_video_frame_offset.py \
    --batch-sessions data/BG_046_sessions_manifest.csv
```

**Outputs**:
- `table_output/video_frame_trim_recommendations.csv` - Trim values per video
- Console report with frame counts and durations

---

## Workflow Stages

### Stage 1: MATLAB Preprocessing (Manual)

**Before running Python scripts, complete in MATLAB**:

1. **Load session with your MATLAB script**
   ```matlab
   % Your existing session loading code
   session_data = load_session_BG046('BG_046_15082025');
   ```

2. **Extend NI events to include optotagging period**
   ```matlab
   % Your MATLAB script that modifies NI_events
   % to capture laser stimulation timestamps
   session_data = extend_nidaq_to_optotag_period(session_data);
   ```

3. **Save as .mat file**
   ```matlab
   save('data/BG_046_15082025.mat', 'session_data', '-v7.3');
   ```

**Repeat for all 33 BG_046 sessions**

---

### Stage 2: TPrime Clock Alignment (Optional, see `README_tprime_workflow.md`)

If using TPrime for multi-stream synchronization:

1. **Generate TPrime corrections** (manual step using TPrime executable)
   ```bash
   TPrime -syncperiod=1.0 \
          -imec=session_g0_t0.imec.ap.meta \
          -ni=session_ni.meta \
          -out=tprime_corrections/
   ```

2. **Apply corrections in MATLAB** (before saving .mat)
   ```matlab
   tprime_file = 'session_path/tprime_corrections/ni_to_imec.tprime.txt';
   session_data.NI_events = apply_tprime_to_nidaq(session_data.NI_events, tprime_file);
   ```

See `scripts/README_tprime_workflow.md` for detailed integration guide.

---

### Stage 3: Python Conversion & Validation

```bash
# 1. Convert all sessions
python scripts/batch_convert_bg046.py

# 2. Run comprehensive validation
python scripts/run_validation_suite.py --subject BG_046 --html-report

# 3. Review HTML report
# Open: table_output/validation/validation_report_<timestamp>.html
```

**Review validation report**:
- Check pass/fail status for each session
- Address any ERRORS before proceeding
- WARNINGS are informational but may not block analysis

---

### Stage 4: Sanity Checks (from validation report)

#### a. Duration Consistency ✓

**Expected**: NI and IMEC durations match within 0.5s

**If failed**:
- Check if session ended prematurely
- Verify MATLAB loading script captured full session
- Inspect raw data files for corruption

#### b. Photodiode Sync ✓

**Expected**: Sync quality = 'good' or 'excellent'

**If failed/missing**:
- Check if photodiode channel was recorded
- Verify channel names in ni_events dict
- May need to identify correct channel in MATLAB preprocessing

#### c. Video Frame Timing ✓

**Expected**: Trim recommendations generated for each video

**Action**:
- Use trim values to slice video files
- Update frame timestamps in analysis scripts
- Document trim values in session metadata

---

## Common Issues & Solutions

### Issue: "No .mat files found"

**Cause**: Wrong data directory or subject ID

**Solution**:
```bash
# Check data directory
ls data/BG_046_*.mat

# Specify custom directory
python scripts/batch_convert_bg046.py --data-dir /path/to/data/
```

---

### Issue: Conversion fails with "KeyError: 'NPX_probes'"

**Cause**: .mat file structure doesn't match expected schema

**Solution**:
- Verify MATLAB save format: `-v7.3` or `-v7`
- Check that session data is nested correctly: `data.BG_046.BG_046_DDMMYYYY`
- Inspect .mat structure with `scripts/inspect_session.py`

---

### Issue: "Missing Laser event" warning

**Cause**: Optotagging data not captured in NI_events

**Solution**:
- Re-run MATLAB script with extended recording period
- Ensure laser TTL pulses are on recorded NI channel
- Check that MATLAB script includes laser timing extraction

---

### Issue: Duration deviation >0.5s

**Possible causes**:
1. Session ended early (animal removed from rig)
2. Recording systems stopped at different times
3. Clock drift without TPrime correction

**Diagnostics**:
```bash
# Check raw durations
python scripts/validate_metadata_duration.py \
    --session data/BG_046_SESSION.pkl \
    --verbose
```

**Solutions**:
- If drift is consistent, consider TPrime integration
- If ending mismatch, trim to shortest duration
- Document limitation for that session

---

### Issue: No photodiode data found

**Cause**: Channel not recorded or named differently

**Solution**:
1. Check available channels:
   ```python
   import pickle
   with open('data/BG_046_15082025.pkl', 'rb') as f:
       session = pickle.load(f)
   print(session.ni_events.keys())
   ```

2. Update `validate_photodiode_sync.py` to include your channel name:
   ```python
   photodiode_keys = [
       'Photodiode', 'photodiode', 'PD', 'pd',
       'YOUR_CHANNEL_NAME'  # Add here
   ]
   ```

---

## Output File Organization

```
data/
├── BG_046_01072025.mat
├── BG_046_01072025.pkl  ← Converted
├── BG_046_02072025.mat
├── BG_046_02072025.pkl
└── ...

table_output/
├── validation/
│   ├── validation_report_20251114_143022.html  ← Main report
│   ├── validation_summary.csv
│   └── validation_20251114_143022.log
├── duration_validation.csv
└── video_frame_trim_recommendations.csv

png_output/
└── sync_validation/
    ├── 01072025_photodiode_sync.png
    ├── 02072025_photodiode_sync.png
    └── ...
```

---

## Next Steps After Validation

Once all sessions pass validation:

1. **Update manifest**:
   ```bash
   python scripts/build_manifest_and_behavior_summary.py --subject BG_046
   ```

2. **Run quality control**:
   ```bash
   python scripts/run_unit_selection_batch.py \
       --subject BG_046 \
       --profile striatal_strict
   ```

3. **Proceed with analyses**:
   - Optotagging: `scripts/run_optotag.py`
   - Responsiveness: `scripts/run_responsiveness_batch.py`
   - Decoding: `scripts/run_decoding_hit_miss.py`
   - Population: `scripts/run_demo_pipeline.py`

---

## Advanced Usage

### Custom Validation Thresholds

Edit validation scripts to adjust criteria:

**Duration threshold**:
```python
# In validate_metadata_duration.py
--threshold 1.0  # Allow up to 1 second deviation
```

**Sync matching window**:
```python
# In validate_photodiode_sync.py
--max-lag 0.05  # 50ms matching tolerance
```

### Parallel Processing

For large batch jobs, use GNU Parallel:

```bash
# Convert sessions in parallel
ls data/BG_046_*.mat | parallel -j 4 python scripts/convert_mat_to_pkl.py {}
```

### Integration with Existing Pipelines

To add validation to existing batch scripts:

```python
# In your analysis script
from scripts.run_validation_suite import run_full_validation

# Before analysis
result = run_full_validation(pkl_path)
if not result['overall_passed']:
    logging.warning(f"Session {pkl_path} failed validation")
    # Handle appropriately
```

---

## Troubleshooting & Support

**Check logs**: All validation scripts support `--verbose` flag for detailed diagnostics

**Inspect session manually**:
```bash
python scripts/inspect_session.py data/BG_046_15082025.pkl output_dir/
```

**Test on single session first**: Always validate workflow on 1-2 test sessions before batch processing

**Preserve originals**: Never delete .mat files until .pkl conversion is validated

---

## Script Reference

| Script | Purpose | Key Flags |
|--------|---------|-----------|
| `batch_convert_bg046.py` | .mat → .pkl conversion | `--force`, `--dry-run` |
| `run_validation_suite.py` | Full validation pipeline | `--html-report`, `--subject` |
| `validate_metadata_duration.py` | Check NI-IMEC sync | `--threshold`, `--output-csv` |
| `validate_photodiode_sync.py` | Photodiode alignment | `--plot`, `--max-lag` |
| `analyze_video_frame_offset.py` | Video trim analysis | `--batch-sessions` |

---

**Last Updated**: 2025-11-14  
**Version**: 1.0  
**Maintainer**: BG_046 Analysis Pipeline

For questions or issues, refer to repository documentation or contact lab coordinator.
