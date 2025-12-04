# BG_046 Data Preparation Checklist

Use this checklist to track progress through the data preparation pipeline.

## Pre-Python Steps (MATLAB)

### Session Loading & Modification
- [ ] **Step 1.1**: Run MATLAB session loading script for each session
  - Script: `your_matlab_session_loader.m`
  - Sessions: 33 total (BG_046_23062025 through BG_046_17092025)
  
- [ ] **Step 1.2**: Modify NI-DAQ objects to include optotagging period
  - Script: `your_matlab_nidaq_extender.m`
  - Verify: Laser timestamps appear in NI_events
  - Check: ~500-1000 laser pulses per session (10ms duration)

- [ ] **Step 1.3**: Save all sessions as .mat files
  - Location: `data/BG_046_*.mat`
  - Format: `-v7.3` for large files
  - Verify: 33 .mat files created

**Estimated time**: 2-3 hours for all 33 sessions

---

## Python Conversion & Validation

### Stage 1: Batch Conversion
- [ ] **Step 2.1**: Dry run batch conversion
  ```bash
  python scripts/batch_processing/batch_convert_bg046.py --dry-run
  ```
  - Review: Which files need conversion
  - Check: Expected 30-33 conversions

- [ ] **Step 2.2**: Execute batch conversion
  ```bash
  python scripts/batch_processing/batch_convert_bg046.py
  ```
  - Monitor: Conversion progress logs
  - Verify: All conversions successful (no errors)
  - Output: 33 .pkl files in `data/`

- [ ] **Step 2.3**: Verify .pkl files created
  ```bash
  ls -lh data/BG_046_*.pkl | wc -l  # Should be 33
  ```

**Estimated time**: 10-20 minutes

---

### Stage 2: Initial Validation
- [ ] **Step 3.1**: Run comprehensive validation suite
  ```bash
  python scripts/run_validation_suite.py --subject BG_046 --html-report
  ```
  - Check: HTML report generated
  - Review: Overall pass rate

- [ ] **Step 3.2**: Review validation report
  - Open: `table_output/validation/validation_report_*.html`
  - Check: Pass/fail status per session
  - Note: Any sessions with errors

- [ ] **Step 3.3**: Address critical failures
  - Review: Sessions marked as FAIL
  - Fix: Errors in source .mat files if needed
  - Re-convert: Failed sessions after fixes

**Estimated time**: 15-30 minutes

---

### Stage 3: Sanity Check 4a - Duration Consistency
- [ ] **Step 4a.1**: Validate NI-IMEC duration consistency
  ```bash
  python scripts/validate_metadata_duration.py \
      --batch "data/BG_046_*.pkl" \
      --output-csv table_output/duration_validation.csv
  ```

- [ ] **Step 4a.2**: Review duration results
  - Check: Pass rate (target: 100%)
  - Acceptable: Deviation ≤ 0.5 seconds
  - Inspect: Any sessions with >0.5s deviation
  
- [ ] **Step 4a.3**: Document deviations
  - Session: _______________  Deviation: _____ s
  - Session: _______________  Deviation: _____ s
  - Session: _______________  Deviation: _____ s
  
  **Action for >0.5s deviation**:
  - [ ] Consider TPrime integration (see README_tprime_workflow.md)
  - [ ] Document as known limitation
  - [ ] Exclude from timing-critical analyses

**Estimated time**: 10 minutes

**Expected outcome**: ≥90% sessions pass (≤0.5s deviation)

---

### Stage 4: Sanity Check 4b - Photodiode Sync
- [ ] **Step 4b.1**: Check photodiode-FSM synchronization
  ```bash
  python scripts/validate_photodiode_sync.py \
      --batch "data/BG_046_*.pkl" \
      --plot \
      --output-csv table_output/photodiode_sync.csv
  ```

- [ ] **Step 4b.2**: Review sync quality ratings
  - Excellent (<1ms): _____ sessions
  - Good (1-10ms): _____ sessions
  - Acceptable (10-50ms): _____ sessions
  - Poor (>50ms): _____ sessions
  - No data: _____ sessions

- [ ] **Step 4b.3**: Inspect diagnostic plots
  - Location: `png_output/sync_validation/`
  - Check: Event timing overlays
  - Verify: Inter-event interval distributions

- [ ] **Step 4b.4**: Handle missing photodiode data
  - If no photodiode data found:
    - [ ] Verify channel name in ni_events
    - [ ] Check MATLAB preprocessing captured photodiode
    - [ ] Document as limitation if unavailable

**Estimated time**: 15 minutes

**Expected outcome**: 
- If photodiode recorded: ≥80% sessions rated 'good' or better
- If not recorded: Document and skip this check

---

### Stage 5: Sanity Check 4c - Video Frame Offsets (Optional)

**Note**: Can be done later during video analysis

- [ ] **Step 4c.1**: Analyze video frame metadata
  ```bash
  python scripts/analyze_video_frame_offset.py \
      --batch-sessions data/BG_046_sessions_manifest.csv
  ```
  
  **Or** if no manifest yet:
  - [ ] Process individual session directories
  - [ ] Document video metadata file locations

- [ ] **Step 4c.2**: Review trim recommendations
  - File: `table_output/video_frame_trim_recommendations.csv`
  - Note: Sessions needing trim: _____ / 33

- [ ] **Step 4c.3**: Apply video trimming (future step)
  - [ ] Use trim values in video analysis scripts
  - [ ] Update video frame timestamps
  - [ ] Document trim parameters per session

**Estimated time**: 20 minutes (analysis only, trimming done later)

**Priority**: Low (can proceed with neural analyses without this)

---

## TPrime Integration (If Needed)

### Stage 6: Multi-Clock Alignment (Optional)

**Prerequisites**: 
- Sync signals recorded during acquisition
- TPrime software installed
- >0.5s duration deviations found in Step 4a

- [ ] **Step 5.1**: Generate TPrime corrections
  - For each session:
    ```bash
    TPrime -syncperiod=1.0 \
           -imec=session_g0_t0.imec.ap.meta \
           -ni=session_ni.meta \
           -out=tprime_corrections/
    ```
  - Sessions processed: _____ / 33

- [ ] **Step 5.2**: Verify TPrime outputs
  - Check: `tprime_corrections/ni_to_imec.tprime.txt` exists
  - Inspect: Correction magnitude (typical: <100ms over 30min)

- [ ] **Step 5.3**: Apply corrections in MATLAB
  - Use: `apply_tprime_to_nidaq.m` function
  - Re-save: Corrected .mat files
  - Flag: Set `NI_events_tprime_corrected = true`

- [ ] **Step 5.4**: Re-convert corrected sessions
  ```bash
  python scripts/batch_processing/batch_convert_bg046.py --force
  ```

- [ ] **Step 5.5**: Re-validate durations
  ```bash
  python scripts/validate_metadata_duration.py \
      --batch "data/BG_046_*.pkl"
  ```
  - Expected: Improved pass rate

**Estimated time**: 2-4 hours for full pipeline

**See**: `scripts/README_tprime_workflow.md` for detailed guide

---

## Post-Validation Steps

### Stage 7: Proceed with Analyses

- [ ] **Step 6.1**: Update session manifest
  ```bash
  python scripts/build_manifest_and_behavior_summary.py --subject BG_046
  ```

- [ ] **Step 6.2**: Run unit quality control
  ```bash
  python scripts/run_unit_selection_batch.py \
      --subject BG_046 \
      --profile striatal_strict
  ```

- [ ] **Step 6.3**: Optotagging analysis
  ```bash
  python scripts/run_optotag.py --subject BG_046
  ```

- [ ] **Step 6.4**: Responsiveness analysis
  ```bash
  python scripts/run_responsiveness_batch.py --subject BG_046
  ```

- [ ] **Step 6.5**: Population decoding
  ```bash
  python scripts/run_decoding_hit_miss.py --subject BG_046
  ```

---

## Quality Metrics Summary

Fill in after completing validation:

### Conversion Success Rate
- Total sessions: 33
- Successful conversions: _____ / 33 ( _____% )

### Validation Pass Rates
- Overall validation: _____ / 33 ( _____% )
- Completeness check: _____ / 33 ( _____% )
- Trial integrity: _____ / 33 ( _____% )
- Spike data quality: _____ / 33 ( _____% )
- Duration consistency: _____ / 33 ( _____% )
- Photodiode sync: _____ / 33 ( _____% )

### Critical Issues Identified
1. __________________________________________________
2. __________________________________________________
3. __________________________________________________

### Sessions Excluded from Analysis
- Session: _____________ Reason: _____________________
- Session: _____________ Reason: _____________________
- Session: _____________ Reason: _____________________

---

## Final Sign-Off

- [ ] All critical errors resolved
- [ ] Validation reports reviewed and approved
- [ ] Data quality documented
- [ ] Ready to proceed with scientific analyses

**Completed by**: ________________  
**Date**: ________________  
**Notes**: 
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________

---

## Quick Reference Commands

```bash
# Full pipeline in one go (after MATLAB steps)
python scripts/batch_processing/batch_convert_bg046.py && \
python scripts/run_validation_suite.py --subject BG_046 --html-report && \
python scripts/validate_metadata_duration.py --batch "data/BG_046_*.pkl" && \
python scripts/validate_photodiode_sync.py --batch "data/BG_046_*.pkl" --plot

# View results
open table_output/validation/validation_report_*.html  # or 'start' on Windows
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-14  
**Project**: BG_046 Visual Change Detection Analysis
