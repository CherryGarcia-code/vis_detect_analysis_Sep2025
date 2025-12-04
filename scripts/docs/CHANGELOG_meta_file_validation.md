# Changelog: .meta File-Based Duration Validation

**Date:** November 14, 2025  
**Changes by:** GitHub Copilot (requested by user)

## Summary

Modified the duration validation system to read recording durations directly from SpikeGLX `.meta` files instead of computing them from aggregated event/spike timestamps. This provides ground-truth durations from the acquisition system and dramatically improves validation accuracy.

## Motivation

**Previous Approach:**
- Computed NI duration from max-min of ALL event times (including ~226k rotary encoder events)
- Computed IMEC duration from max-min of ALL spike times (across 613 clusters)
- Result: 0.5-1.2s deviations over 2+ hour sessions
- Pass rate: 11/25 sessions (44%) with 0.5s threshold

**Problem:** While deviations were acceptably small (~0.009% error), the computed values didn't represent true acquisition duration, just the span of recorded events.

**New Approach:**
- Read `fileTimeSecs` directly from `.imec0.ap.meta` (IMEC probe recording)
- Read `fileTimeSecs` directly from `.nidq.meta` (NI-DAQ recording)
- Result: 0.042-0.062s deviations (ground truth comparison)
- Pass rate: 25/25 sessions (100%) with 0.5s threshold

## Files Modified

### 1. `scripts/validate_metadata_duration.py`

**Added Functions:**
- `parse_spikeglx_meta(meta_path)` - Parses SpikeGLX .meta files (key=value format)
- `find_meta_files(raw_data_root, subject, session_name)` - Locates .imec0.ap.meta and .nidq.meta files

**Modified Functions:**
- `extract_ni_duration(session, raw_data_root=None)`
  - Now accepts `raw_data_root` parameter
  - Reads `fileTimeSecs` from `.nidq.meta` if available
  - Falls back to legacy timestamp computation if not
  
- `extract_imec_duration(session, raw_data_root=None)`
  - Now accepts `raw_data_root` parameter
  - Reads `fileTimeSecs` from `.imec0.ap.meta` if available
  - Falls back to legacy timestamp computation if not

- `validate_session_duration(session, threshold=0.5, raw_data_root=None)`
  - Now accepts `raw_data_root` parameter
  - Passes it through to extraction functions

**Added CLI Arguments:**
```bash
--raw-data-root PATH    # Path to raw data directory for .meta files
                         # Example: "X:/public/.../BG_046/Raw data"
```

### 2. `scripts/run_validation_suite.py`

**Modified Functions:**
- `run_full_validation(pkl_path, raw_data_root=None)`
  - Now accepts `raw_data_root` parameter
  - Passes it to `duration_validator.validate_session_duration()`

**Added CLI Arguments:**
```bash
--raw-data-root PATH    # Path to raw data directory for .meta files
```

**Bug Fixes:**
- Changed `output_path.write_text(html)` → `output_path.write_text(html, encoding='utf-8')`
  - Fixes Windows cp1252 encoding errors with Unicode characters in HTML

## Expected Directory Structure

The validation scripts expect raw data in this structure:
```
raw_data_root/
├── BG_046_DDMMYYYY/
│   ├── EphysNidaq/
│   │   ├── BG_046_DDMMYYYY_g0_t0.nidq.meta     # NI-DAQ metadata
│   │   ├── BG_046_DDMMYYYY_g0_t0.nidq.bin      # NI-DAQ binary
│   │   └── BG_046_DDMMYYYY_g0_imec0/
│   │       ├── BG_046_DDMMYYYY_g0_t0.imec0.ap.meta  # IMEC metadata
│   │       └── BG_046_DDMMYYYY_g0_t0.imec0.ap.bin   # IMEC binary
```

## Usage Examples

### Validate Single Session (New Method)
```bash
python scripts/validate_metadata_duration.py \
  --session data/BG_046_24062025.pkl \
  --raw-data-root "X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/Raw data"
```

### Validate All Sessions (New Method)
```bash
python scripts/validate_metadata_duration.py \
  --batch "data/BG_046_*.pkl" \
  --raw-data-root "X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/Raw data" \
  --output-csv table_output/duration_validation_meta.csv
```

### Full Validation Suite with HTML Report
```bash
python scripts/run_validation_suite.py \
  --subject BG_046 \
  --raw-data-root "X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/Raw data" \
  --html-report
```

### Legacy Method (Without .meta Files)
```bash
# Falls back to computing from timestamps
python scripts/validate_metadata_duration.py --batch "data/BG_046_*.pkl"
```

## Validation Results

### Before (Computed Durations)
- Sessions processed: 25
- Passed (≤0.5s): 11
- Failed (>0.5s): 14
- Deviation range: 0.42s - 1.16s
- Pass rate: 44.0%

### After (.meta File Durations)
- Sessions processed: 25
- Passed (≤0.5s): 25
- Failed (>0.5s): 0
- Deviation range: 0.042s - 0.062s
- Pass rate: 100.0%

## Benefits

1. **Ground Truth:** Uses acquisition system's reported duration, not computed values
2. **Improved Accuracy:** Deviations reduced by ~20x (1.0s → 0.05s typical)
3. **Better Validation:** 100% pass rate indicates true clock synchronization
4. **Backward Compatible:** Falls back to legacy computation if .meta files unavailable
5. **Clear Intent:** Validates NI-IMEC clock sync, not event/spike span

## Notes

- The `fileTimeSecs` value in .meta files represents the total acquisition time reported by the SpikeGLX system
- Small deviations (<0.06s) are expected due to minor clock differences between NI-DAQ and IMEC systems
- If raw_data_root is not provided, scripts automatically fall back to computing durations from session data
- This change does not affect the .pkl session files - they remain unchanged

## Testing

All 25 BG_046 sessions validated successfully:
```bash
[INFO] Total sessions: 25
[INFO] Passed:         25
[INFO] Failed:         0
[INFO] Pass rate:      100.0%
```

HTML validation report generated:
`table_output/validation/validation_report_20251114_183446.html`
