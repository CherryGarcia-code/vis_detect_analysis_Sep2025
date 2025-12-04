# TPrime Integration Workflow for Multi-Stream Time Alignment

## Overview

TPrime is a tool for correcting clock drift between multiple data acquisition systems (e.g., Neuropixels IMEC, NI-DAQ, behavioral cameras). This document outlines the integration requirements for incorporating TPrime-corrected timestamps into the BG_046 analysis pipeline.

**Current Status**: ❌ TPrime is **NOT currently integrated** in this repository. All timing relies on NI-DAQ as the master clock.

---

## Background

### Why TPrime?

In multi-stream recordings, each acquisition system has its own clock:
- **IMEC (SpikeGLX)**: Records neural data at 30 kHz with its own timebase
- **NI-DAQ**: Records behavioral events (licks, valves, photodiode, laser)
- **Cameras**: Record video frames with USB timestamps

Clock drift can accumulate over long sessions (>30 minutes), causing:
- Misalignment between spike times and behavioral events (tens to hundreds of ms)
- Photodiode/frame timing desynchronization
- Optotagging analysis errors

TPrime corrects this by:
1. Using a shared sync signal (e.g., TTL pulses recorded by all systems)
2. Computing drift-corrected mapping functions
3. Providing aligned timestamps in a common timebase

---

## TPrime Workflow

### Step 1: Data Collection Requirements

**During recording**, ensure sync signals are captured:
- **SpikeGLX**: Record sync channel (usually digital line recording sync pulses)
- **NI-DAQ**: Record same sync signal on a dedicated channel
- **Video**: Frame timestamps should be recorded with sync markers

**Recommended sync signal**:
- 1 Hz square wave for long sessions
- Higher frequency (10-100 Hz) for better temporal resolution
- Must be recorded by ALL systems simultaneously

### Step 2: Run TPrime

TPrime requires:
- Path to SpikeGLX `.bin` metadata (`.meta` file with sync edges)
- Path to NI-DAQ sync data
- Configuration file specifying sync channel indices

**Typical TPrime command** (example):
```bash
TPrime -syncperiod=1.0 \
       -imec=path/to/session_g0_t0.imec.ap.meta \
       -ni=path/to/session_ni.meta \
       -out=path/to/tprime_output/
```

**TPrime outputs**:
- `*.tprime.txt`: Mapping files containing correction parameters
- One file per secondary stream (e.g., `ni_tprime.txt`)

### Step 3: Apply Time Corrections

Correction files contain polynomial coefficients or lookup tables:
```
# Example tprime.txt format
# time_original_sec time_corrected_sec
0.0000  0.0000
100.0000  100.0012
200.0000  200.0019
...
```

**Application**:
1. Load original timestamps (e.g., NI event times in samples or seconds)
2. Apply TPrime correction via interpolation
3. Replace original timestamps with corrected values

---

## Integration Plan for This Repository

### Option A: Pre-Processing (Recommended)

**Apply TPrime corrections BEFORE `.mat` to `.pkl` conversion**

1. **MATLAB Script Modification**:
   - After loading raw NI-DAQ data, apply TPrime correction to all event times
   - Update `ni_events` dict with corrected timestamps
   - Save corrected `.mat` file

2. **Python Conversion**:
   - Use existing `batch_convert_bg046.py` on corrected `.mat` files
   - No changes needed to Session dataclasses

**Pros**:
- Clean separation: correction done once at source
- Python analysis code remains unchanged
- Easy to verify corrections before analysis

**Cons**:
- Requires MATLAB TPrime integration or manual correction step
- Must reprocess if TPrime parameters change

### Option B: Post-Processing in Python

**Apply corrections during `.pkl` loading**

1. **Create TPrime Utility Module** (`src/tprime.py`):
   ```python
   def load_tprime_mapping(tprime_file: Path) -> Callable:
       """Load TPrime correction and return interpolation function."""
       pass
   
   def correct_ni_events(ni_events: dict, correction_fn: Callable) -> dict:
       """Apply TPrime correction to all NI event times."""
       pass
   ```

2. **Modify Session Loader**:
   - In `load_mat_file_to_session()`, check for companion `.tprime.txt` file
   - If found, automatically apply corrections
   - Log correction metadata in Session object

**Pros**:
- Flexible: can reapply different corrections without reprocessing `.mat`
- Transparent to downstream analysis code

**Cons**:
- Adds complexity to loading pipeline
- Correction applied on every load (could cache corrected `.pkl`)

### Option C: Separate Alignment Database

**Store corrected timestamps in parallel structure**

1. **Create Alignment Table**:
   ```csv
   session_name,event_type,trial_idx,original_time,corrected_time
   15082025,Baseline_ON,0,1.2345,1.2346
   15082025,Change_ON,0,8.5432,8.5434
   ...
   ```

2. **Analysis Code**:
   - Join alignment table with Session data during analysis
   - Use corrected times for spike-event alignment

**Pros**:
- Non-destructive: original data preserved
- Easy to audit corrections
- Can apply multiple correction schemes

**Cons**:
- More complex data management
- Requires joining logic in every analysis script

---

## Recommended Implementation: Option A (Pre-Processing)

### Detailed Steps

#### 1. Generate TPrime Corrections (Manual Step)

For each session directory:
```bash
cd /path/to/session/BG_046_15082025/
TPrime -syncperiod=1.0 \
       -imec=BG_046_15082025_g0_t0.imec.ap.meta \
       -ni=BG_046_15082025_ni.meta \
       -out=tprime_corrections/
```

Verify output files:
- `tprime_corrections/ni_to_imec.tprime.txt`

#### 2. Create MATLAB TPrime Correction Function

`apply_tprime_to_nidaq.m`:
```matlab
function corrected_events = apply_tprime_to_nidaq(ni_events, tprime_file)
    % Load TPrime correction mapping
    tprime_data = readtable(tprime_file);
    
    % Create interpolation function
    correction_fn = @(t) interp1(tprime_data.time_original, ...
                                  tprime_data.time_corrected, t, 'linear', 'extrap');
    
    % Apply to all event fields
    fields = fieldnames(ni_events);
    corrected_events = ni_events;
    
    for i = 1:length(fields)
        field = fields{i};
        if isstruct(ni_events.(field)) && isfield(ni_events.(field), 'rise_t')
            corrected_events.(field).rise_t = correction_fn(ni_events.(field).rise_t);
            if isfield(ni_events.(field), 'fall_t')
                corrected_events.(field).fall_t = correction_fn(ni_events.(field).fall_t);
            end
        end
    end
end
```

#### 3. Update Session Builder Script

Modify your existing MATLAB session builder to:
```matlab
% After loading NI data
ni_events = load_nidaq_events(session_path);

% Apply TPrime correction if file exists
tprime_file = fullfile(session_path, 'tprime_corrections', 'ni_to_imec.tprime.txt');
if exist(tprime_file, 'file')
    fprintf('Applying TPrime corrections from: %s\n', tprime_file);
    ni_events = apply_tprime_to_nidaq(ni_events, tprime_file);
    data.NI_events_tprime_corrected = true;
else
    warning('No TPrime file found; using uncorrected NI timestamps');
    data.NI_events_tprime_corrected = false;
end

% Save to .mat with correction flag
data.NI_events = ni_events;
save(output_mat_file, 'data');
```

#### 4. Validation in Python

Add TPrime validation to `run_validation_suite.py`:

```python
def validate_tprime_applied(session) -> Dict[str, any]:
    """Check if TPrime corrections were applied."""
    result = {
        'check': 'tprime_correction',
        'passed': False,
        'warnings': []
    }
    
    ni_events = getattr(session, 'ni_events', {})
    
    # Check for correction flag
    if ni_events.get('tprime_corrected', False):
        result['passed'] = True
        result['warnings'].append("TPrime corrections applied")
    else:
        result['warnings'].append("WARNING: No TPrime corrections found")
    
    return result
```

---

## Expected TPrime File Formats

### Input: SpikeGLX `.meta` File

Contains sync channel configuration:
```
syncSourceIdx=0
syncSourcePeriod=1.0
snsSaveChanSubset=0:383,SY:0
```

### Input: NI-DAQ Sync Recording

Binary file with sync pulses recorded alongside behavioral events.

### Output: TPrime Correction File

**Format**: Tab-separated values
```
# TPrime v1.0
# Original timebase: NI-DAQ (seconds)
# Target timebase: IMEC (samples at 30000 Hz, converted to seconds)
original_sec	corrected_sec
0.000000	0.000000
1.000000	1.000012
2.000000	2.000019
...
1800.000000	1800.082145
```

---

## Validation Criteria

After applying TPrime corrections, verify:

1. **Clock Drift Magnitude**:
   - Check max correction: should be <100ms for 30-minute sessions
   - Flag if >500ms (indicates sync failure)

2. **Photodiode Alignment**:
   - Run `validate_photodiode_sync.py` on corrected data
   - Expect sync_quality = 'excellent' (<1ms drift)

3. **Trial Event Consistency**:
   - Verify `Baseline_ON` → `Change_ON` intervals match expected distributions
   - Check that laser pulse timing matches expected 10ms duration

4. **Spike-Event Latencies**:
   - For optotagged units, verify latency distributions are tight (1-5ms)
   - Check that trial-aligned PETHs show consistent event locking

---

## Troubleshooting

### Problem: No sync signal recorded

**Solution**: Cannot apply TPrime retrospectively. Options:
- Proceed with uncorrected timestamps (document limitation)
- Use heuristic corrections based on expected drift rates
- For future sessions, ensure sync recording is enabled

### Problem: TPrime produces large corrections (>1s)

**Likely causes**:
- Sync channel mismatch (wrong channel index)
- Missing sync pulses (cable disconnection)
- Incorrect sync period parameter

**Solution**: Re-run TPrime with corrected parameters or mark session as unreliable.

### Problem: Corrected timestamps break trial structure

**Symptom**: `Change_ON` times precede `Baseline_ON`

**Solution**: Check that correction was applied consistently to all events. May need to regenerate `.mat` file.

---

## Future Enhancements

1. **Automated TPrime Execution**:
   - Create Python wrapper: `scripts/run_tprime_batch.py`
   - Batch process all BG_046 sessions

2. **Correction Quality Metrics**:
   - Visualize drift over session duration
   - Compute residual sync errors post-correction

3. **Integration with UnitMatch**:
   - Ensure TPrime corrections are consistent across days for chronic tracking

---

## References

- TPrime documentation: [Bill Karsh's TPrime GitHub](https://github.com/billkarsh/TPrime)
- SpikeGLX sync documentation: [SpikeGLX User Manual](https://github.com/billkarsh/SpikeGLX)

---

## Contact / Notes

**Last Updated**: 2025-11-14

If implementing TPrime integration:
1. Start with 2-3 test sessions
2. Validate corrections carefully before batch processing
3. Document correction parameters in session metadata
4. Keep uncorrected `.mat` files as backup

For questions or issues, consult repository maintainer or lab's TPrime expert.
