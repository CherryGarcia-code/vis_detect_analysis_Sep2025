# Normalization Bug Fixes - Population CD Scripts

## Summary

Fixed critical baseline normalization bugs in 3 population analysis scripts that were artificially equalizing outcome baselines and hiding real pre-stimulus differences.

## Problem

**Original (WRONG) approach**: Each outcome type was z-scored to its own baseline
```python
all_hit.append(_zscore_baseline(hit_trace, ref_bc, CHANGE_BL))  # Hit → Hit baseline
all_fa.append(_zscore_baseline(fa_trace, ref_bc, CHANGE_BL))    # FA → FA baseline
all_cr.append(_zscore_baseline(cr_trace, ref_bc, CHANGE_BL))    # CR → CR baseline
```

**Why wrong**: This artificially makes all baselines equal to zero, removing biologically meaningful pre-stimulus differences between trial types.

## Solution

**Fixed (CORRECT) approach**: All outcome types normalized to shared baseline
```python
# Compute Hit baseline stats ONCE
hit_bl = hit_trace[bl_mask]
mu_shared = hit_bl.mean()
sd_shared = hit_bl.std()

# Apply SAME normalization to all outcomes
all_hit.append((hit_trace - mu_shared) / sd_shared)
all_fa.append((fa_trace - mu_shared) / sd_shared)
all_cr.append((cr_trace - mu_shared) / sd_shared)
```

**Why correct**: Preserves relative baseline differences while enabling valid cross-session averaging.

## Files Fixed

### 1. `analysis_suite/03_population/a_coding_direction.py`
- **Panel D**: Hit/FA/CR change-aligned grand average (lines ~687-693)
- **Panel F**: Hit/FA lick-aligned grand average (lines ~794-797)

### 2. `analysis_suite/03_population/d_state_matched_cd.py`
- **Left panels**: FA/Hit_small/Hit_big grand averages (lines ~676)

### 3. `analysis_suite/03_population/e_sensory_dose_response.py`
- **Left panels**: FA/Go_small/Go_big grand averages (lines ~191)

## Implementation Details

- **Baseline reference**: Uses highest-signal condition (Hit/Hit_big/Go_big) as shared baseline
- **Fallback handling**: If shared baseline computation fails, falls back to per-outcome normalization with warning
- **Robustness**: Added checks for sufficient baseline samples (≥2) and non-zero standard deviation
- **Backward compatibility**: Maintains same figure structure and output format

## Scientific Impact

This fix will:
✅ **Preserve pre-stimulus differences** between trial types
✅ **Make cross-outcome comparisons interpretable**
✅ **Reveal true baseline activity patterns**
✅ **Enable valid 2D state-space analysis** (critical for your 2D decomposition plan)

## Testing

- All 3 scripts pass syntax validation
- Changes maintain existing function signatures and figure structure
- Ready for re-running to generate corrected results

## Next Steps

1. Re-run fixed scripts to regenerate figures with correct normalization
2. Compare old vs new results to assess impact magnitude
3. Apply same fixes to other affected scripts (06_lick_motor, 05_longitudinal)
4. Update 2D decomposition plan to use correct normalization approach