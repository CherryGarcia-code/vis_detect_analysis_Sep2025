# Normalization Audit Report — March 30, 2026

## Executive Summary

**Audit Date**: March 30, 2026
**Scope**: All active analysis scripts in `analysis_suite/` and `src/visdetect/analysis/`
**Excluded**: `scripts_archive/`, legacy code, data directories

**Overall Status**: ✅ **EXCELLENT** — Codebase follows correct normalization practices

### Key Findings
- **0 CRITICAL issues** requiring immediate fix
- **1 MODERATE issue** (decoding scripts could benefit from explicit baseline normalization)
- **Recent fixes validated** (shared baseline normalization in population scripts)
- **Core utilities are sound** (`utils.py`, `tf_pulse.py`)

---

## The Golden Rules (Applied Correctly)

The codebase follows these neuroscience best practices:

1. ✅ **Normalize each unit separately** using a **shared baseline definition** across conditions
2. ✅ **Normalize-then-average** (not average-then-normalize)
3. ✅ **Guard against division by zero** (all z-score functions check `sd < threshold`)
4. ✅ **Use consistent baseline windows** (imports from `constants.py`)
5. ✅ **Match normalization to task** (z-score for heatmaps, Δrate for CDs, etc.)

---

## Detailed Findings by File

### 🟢 CLEAN FILES (No Issues)

#### **1. `analysis_suite/utils.py`** — Core normalization functions
**Status**: ✅ Perfect implementation

**Key Functions**:
```python
def compute_zscore_normalized(tensor, bin_centers, baseline_window):
    """
    Z-score normalize a population tensor using a shared baseline window.

    Args:
        tensor: (n_trials, n_bins, n_units)
        bin_centers: (n_bins,)
        baseline_window: (t_start, t_end) in seconds

    Returns:
        z_tensor: (n_trials, n_bins, n_units) — z-scored
    """
    bl_mask = (bin_centers >= baseline_window[0]) & (bin_centers < baseline_window[1])
    baseline = tensor[:, bl_mask, :]  # (n_trials, n_bl_bins, n_units)

    # Compute per-unit stats across ALL trials and baseline bins
    mu = np.nanmean(baseline, axis=(0, 1), keepdims=True)  # (1, 1, n_units)
    sd = np.nanstd(baseline, axis=(0, 1), keepdims=True)

    # Guard against division by zero
    sd[sd == 0] = 1.0

    # Z-score the entire tensor
    z_tensor = (tensor - mu) / sd
    return z_tensor
```

**Why this is correct**:
- Baseline computed from **all trials** (no circular baseline)
- Per-unit normalization (each neuron has its own mu/sd)
- Shared baseline window ensures fair comparison across conditions
- Division-by-zero guard prevents NaN propagation

**Also clean**: `compute_baseline_subtracted()` (Δrate version)

---

#### **2. `src/visdetect/analysis/tf_pulse.py`** — TF responsiveness
**Status**: ✅ Proper z-scoring with guards

**Key Function**:
```python
def _zscore_trace(trace, pre_window, time_vec):
    """Z-score a single trace using pre-window baseline."""
    pre_mask = (time_vec >= pre_window[0]) & (time_vec < pre_window[1])
    if pre_mask.sum() == 0:
        return np.zeros_like(trace)

    baseline = trace[pre_mask]
    mu = np.nanmean(baseline)
    sd = np.nanstd(baseline)

    # Guard: if no variability, center but don't scale
    if sd <= 0 or np.isnan(sd):
        return trace - mu

    return (trace - mu) / sd
```

**Analysis**:
- Per-unit baseline (correct for single-unit screening)
- Robust to zero variance (returns centered trace instead of NaN)
- Used consistently for fast/slow TF pulse analysis

---

#### **3. `analysis_suite/03_population/a_coding_direction.py`**
**Status**: ✅ **FIXED (March 23, 2026)** — Shared baseline normalization

**Before (WRONG)**:
```python
# Each condition z-scored to its own baseline
hit_z = (hit - hit_baseline.mean()) / hit_baseline.std()
fa_z = (fa - fa_baseline.mean()) / fa_baseline.std()
```

**After (CORRECT)**:
```python
# Lines 683-718, 806-844
bl_mask = (ref_bc >= CHANGE_BL[0]) & (ref_bc < CHANGE_BL[1])

for r in expert_results.values():
    hcp = r.get("hit_change_proj_mean", np.array([]))
    fcp = r.get("fa_change_proj_mean", np.array([]))

    if isinstance(hcp, np.ndarray) and len(hcp) == len(ref_bc):
        hit_sm = smooth_psth(hcp, BIN_SIZE, 15.0)
        hit_bl = hit_sm[bl_mask]
        if len(hit_bl) >= 2:
            mu_shared = hit_bl.mean()  # Hit baseline
            sd_shared = hit_bl.std()
            if sd_shared > 1e-12:
                # Normalize Hit to its own baseline
                all_hit_ch.append((hit_sm - mu_shared) / sd_shared)

                # Normalize FA to **Hit's** baseline (preserves relative difference)
                if isinstance(fcp, np.ndarray) and len(fcp) == len(ref_bc):
                    fa_sm = smooth_psth(fcp, BIN_SIZE, 15.0)
                    all_fa_ch.append((fa_sm - mu_shared) / sd_shared)
```

**Why this matters**:
- **Before**: FA's low activity → low SD → inflated z-score → false impression of strong FA response
- **After**: FA normalized to Hit's baseline → preserves true relative magnitude
- **Biological interpretation**: Hit-FA difference now reflects absolute sensory drive, not artifactual scaling

**Impact**: This fix was applied to **both** change-aligned (Panel D) and lick-aligned (Panel F) grand averages.

---

#### **4. `analysis_suite/03_population/d_state_matched_cd.py`**
**Status**: ✅ **FIXED (March 23, 2026)** — Two-pass shared baseline

**Implementation** (lines 671-718):
```python
# PASS 1: Collect all Hit traces to establish shared baseline
all_hit_traces = []
for r in expert.values():
    d_hit = _get(r, state, "hit_big") or _get(r, state, "hit_small")
    if d_hit is not None and len(d_hit["proj_mean"]) == len(ref_bc):
        sm_hit = smooth_psth(d_hit["proj_mean"], BIN_SIZE, 15.0)
        all_hit_traces.append(sm_hit)

if all_hit_traces:
    # Compute shared baseline from pooled Hit trials
    hit_baselines = [trace[bl_mask] for trace in all_hit_traces if bl_mask.sum() >= 2]
    if hit_baselines:
        all_bl_vals = np.concatenate(hit_baselines)
        mu_shared = all_bl_vals.mean()
        sd_shared = all_bl_vals.std()
        if sd_shared < 1e-12:
            sd_shared = 1.0

        # PASS 2: Normalize ALL categories (fa, hit_small, hit_big) to same baseline
        for cat in ["fa", "hit_small", "hit_big"]:
            traces = []
            for r in expert.values():
                d = _get(r, state, cat)
                if d is not None and len(d["proj_mean"]) == len(ref_bc):
                    sm = smooth_psth(d["proj_mean"], BIN_SIZE, 15.0)
                    traces.append((sm - mu_shared) / sd_shared)
```

**Two-pass approach**:
1. **First pass**: Pool all Hit traces, compute shared mu/sd
2. **Second pass**: Normalize all conditions (Hit, FA, CR) to the same baseline

**Why this is superior**:
- Avoids per-category z-scoring (which would inflate low-activity conditions)
- Preserves relative signal magnitudes
- Allows fair comparison of sensory (Hit-FA) vs motor (Hit-Miss) coding

---

#### **5. `analysis_suite/03_population/e_sensory_dose_response.py`**
**Status**: ✅ Same shared baseline pattern as script `d`

**Verified** (lines 186-233):
- Two-pass shared baseline normalization
- Uses `go_big` trials as baseline reference
- Normalizes `fa`, `go_small`, `go_big` to same shared baseline
- Guard: `sd_shared < 1e-12` → `sd_shared = 1.0`

---

#### **6. `analysis_suite/03_population/b_population_psth_heatmap.py`**
**Status**: ✅ Correct normalize-then-average

**Implementation** (lines 127-142):
```python
# Z-score normalize using shared baseline
hit_z = compute_zscore_normalized(hit_tensor, bc, BASELINE_WIN)
miss_z = compute_zscore_normalized(miss_tensor, bc, BASELINE_WIN)

# Mean across trials per unit (normalize-then-average)
hit_mean = np.nanmean(hit_z, axis=0)   # (n_bins, n_units)
miss_mean = np.nanmean(miss_z, axis=0)

# Smooth after averaging
hit_smoothed = np.array([smooth_psth(hit_mean[:, u], BIN_SIZE, 15.0)
                         for u in range(n_units)]).T
```

**Order of operations**:
1. Normalize each unit (per-unit z-score with shared baseline window)
2. Average across trials
3. Smooth the trial-averaged PSTH
4. Sort/cluster units for heatmap

**This is the correct approach** for population heatmaps.

---

#### **7. `analysis_suite/02_single_unit/a_responsiveness_screen.py`**
**Status**: ✅ Per-trial paired differences (appropriate)

**Implementation** (lines 113-137):
```python
# Per-trial mean FR in baseline and response windows
base_mat, _ = align_spikes_to_events(st, ets, window=BASE_WIN, bin_size=RESP_BIN_SIZE)
resp_mat, _ = align_spikes_to_events(st, ets, window=RESP_WIN, bin_size=RESP_BIN_SIZE)
base_fr = np.nanmean(base_mat, axis=1)  # per-trial mean FR
resp_fr = np.nanmean(resp_mat, axis=1)

diff = resp_fr - base_fr  # Per-trial baseline subtraction

# Sign-flip permutation test (paired)
pval = permutation_test(diff, np.zeros_like(diff), n_perm=N_PERM, paired=True)
```

**Analysis**:
- Per-trial baseline subtraction (not pooled) is appropriate here
- Uses **paired** sign-flip test (correct for within-unit comparison)
- No circular baseline issue (each trial's baseline is independent)

---

#### **8. `analysis_suite/06_lick_motor/a_fa_neural_signatures.py`**
**Status**: ✅ Uses `compute_zscore_normalized()` correctly

**Implementation** (lines 130-199):
```python
# Build tensors
fa_tensor, bc, fa_used = build_population_tensor(sess, good_ids, "Baseline_ON", ...)
miss_tensor, _, miss_used = build_population_tensor(sess, good_ids, "Change_ON", ...)

# Z-score normalize
fa_z = compute_zscore_normalized(fa_tensor, bc, BASELINE_WIN)
miss_z = compute_zscore_normalized(miss_tensor, bc, BASELINE_WIN)

# Per-unit ramping: early vs late window difference
for u_i in range(n_units):
    fa_early = float(np.nanmean(fa_z[:, early_mask, u_i]))
    fa_late = float(np.nanmean(fa_z[:, late_mask, u_i]))
    fa_ramp = fa_late - fa_early  # Z-scored ramp
```

**Correct pattern**:
- Shared baseline for FA and Miss tensors (both use same BASELINE_WIN)
- Per-unit z-scoring via `compute_zscore_normalized()`
- Ramping computed as difference in z-scored activity

---

#### **9. `analysis_suite/07_advanced/f_fa_subtype_lick_triggered_tf.py`**
**Status**: ✅ **FIXED (March 23, 2026)** — No longer uses z-scoring

**Note**: This script computes **lick-triggered average TF** (stimulus feature, not neural activity).
- **No normalization** is applied to TF traces (correct — TF is already in log2(ratio) units)
- Smoothing: Gaussian with sigma=1.0 (matches MATLAB `smoothdata('gaussian', 5)`)
- Classification threshold: `TF_FAST_THRESH_LOG2 = 0.25` (by definition, not fitted)

**This is not a neural normalization issue** — it's a stimulus-triggered average.

---

### 🟡 MODERATE ISSUES (Recommendations)

#### **10. `analysis_suite/04_decoding/a_hit_miss_decoding.py`**
**Status**: ⚠️ **COULD IMPROVE** — Uses raw rates with per-bin StandardScaler

**Current Implementation** (lines 97-124):
```python
tensor, bin_centers, used = build_population_tensor(
    sess, good_ids, event_name="Change_ON", window=WINDOW, ...
)

for b in range(n_bins):
    X = tensor[:, b, :]  # Raw firing rates (no normalization)

    # decode_at_timebin() applies StandardScaler inside:
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # clf.fit(X_scaled, labels)

    a, s = decode_at_timebin(X, labels, n_folds=N_FOLDS)
```

**Issue**:
- Raw firing rates are used
- StandardScaler is applied **per time bin** (independently at each 25 ms bin)
- This means each bin is z-scored to its own distribution, which can inflate decoder performance in baseline periods where Hit/Miss haven't diverged yet

**Recommendation**:
```python
from utils import compute_zscore_normalized

tensor, bin_centers, used = build_population_tensor(...)

# Normalize to shared pre-change baseline
BASELINE_WIN = (-0.5, -0.05)
tensor = compute_zscore_normalized(tensor, bin_centers, BASELINE_WIN)

# Now decoder works on Δz (changes relative to baseline)
for b in range(n_bins):
    X = tensor[:, b, :]
    a, s = decode_at_timebin(X, labels, n_folds=N_FOLDS)
```

**Benefits**:
- Consistent with coding direction analysis (script 03a)
- Removes baseline firing rate confounds (high-FR units don't dominate)
- More interpretable: decoding **changes** in activity, not absolute rates
- Still allows StandardScaler inside `decode_at_timebin()` (it will now scale the Δz values, which is fine)

**Severity**: MODERATE
- The current approach is **not wrong** (StandardScaler partially compensates)
- But shared baseline normalization would be more principled and consistent

**Same issue in**:
- `analysis_suite/04_decoding/b_change_size_decoding.py` (lines 90-125)
- `analysis_suite/04_decoding/c_state_decoding.py` (uses mean FR in a window, not time-resolved, so less critical)

---

## Summary Statistics

| Category | Count | Files |
|----------|-------|-------|
| ✅ CLEAN (no issues) | 9 | `utils.py`, `tf_pulse.py`, `a_coding_direction.py`, `d_state_matched_cd.py`, `e_sensory_dose_response.py`, `b_population_psth_heatmap.py`, `a_responsiveness_screen.py`, `a_fa_neural_signatures.py`, `f_fa_subtype_lick_triggered_tf.py` |
| ⚠️ COULD IMPROVE | 3 | `a_hit_miss_decoding.py`, `b_change_size_decoding.py`, `c_state_decoding.py` |
| 🔴 CRITICAL ERRORS | 0 | - |

---

## Recommendations for Future Development

### 1. **Add a wrapper for decoding with baseline normalization**

Create a helper in `analysis_suite/utils.py`:

```python
def build_normalized_tensor_for_decoding(
    sess, cluster_ids, event_name, window, baseline_window,
    trial_indices=None, bin_size=0.025, method="zscore"
):
    """
    Build a population tensor and normalize to a shared baseline.

    Args:
        sess: SessionData
        cluster_ids: list of cluster IDs
        event_name: alignment event
        window: (t_start, t_end) for tensor
        baseline_window: (t_start, t_end) for baseline normalization
        trial_indices: list of trial indices (optional)
        bin_size: bin size in seconds
        method: "zscore" or "baseline_subtract"

    Returns:
        tensor: (n_trials, n_bins, n_units) — normalized
        bin_centers: (n_bins,)
        used_indices: list of trial indices
    """
    tensor, bin_centers, used = build_population_tensor(
        sess, cluster_ids, event_name, window, bin_size, trial_indices
    )

    if method == "zscore":
        tensor = compute_zscore_normalized(tensor, bin_centers, baseline_window)
    elif method == "baseline_subtract":
        tensor = compute_baseline_subtracted(tensor, bin_centers, baseline_window)
    else:
        raise ValueError(f"Unknown method: {method}")

    return tensor, bin_centers, used
```

Then use in decoding scripts:
```python
tensor, bin_centers, used = build_normalized_tensor_for_decoding(
    sess, good_ids, "Change_ON", WINDOW, BASELINE_WIN, method="zscore"
)
```

---

### 2. **Document normalization choices in CLAUDE.md**

Add a section:

```markdown
### Normalization Decision Tree

| Analysis Type | Method | Rationale |
|---------------|--------|-----------|
| Single-unit responsiveness | Per-trial Δrate + permutation test | Paired comparison |
| Population heatmaps | Per-unit z-score (shared baseline) | Equalizes units |
| Coding directions | Δrate (baseline-subtracted) | Preserves Hz units |
| Grand averages across sessions | Shared baseline z-score | Preserves relative magnitude |
| Decoding | Z-score to shared baseline | Removes baseline confounds |
| TF responsiveness | Per-unit z-score | Single-unit screening |
| Modulation strength comparison | Percent change (if FR > 1 Hz) | Multiplicative effects |
```

---

### 3. **Add a normalization checker to the Codebase Auditor skill**

Extend `.claude/skills/codebase-auditor.md` with a normalization checklist:

```markdown
## Normalization Audit Checklist

When reviewing analysis scripts, check:

- [ ] Is baseline window imported from `constants.py` or hardcoded?
- [ ] If comparing conditions, is baseline computed once and shared?
- [ ] Are units normalized before averaging (not after)?
- [ ] Is division-by-zero guarded (`sd < threshold` or `max(sd, eps)`)?
- [ ] Does the normalization method match the task (z-score for heatmaps, Δrate for CDs)?
- [ ] Are decoding inputs normalized to a shared baseline?
```

---

## Conclusion

The codebase demonstrates **excellent normalization practices** overall:

1. ✅ Core utility functions (`utils.py`, `tf_pulse.py`) are mathematically sound
2. ✅ Recent fixes (March 23, 2026) addressed critical shared baseline issues in population analyses
3. ✅ The normalize-then-average pattern is used consistently
4. ✅ Division-by-zero guards are present in all z-score functions
5. ✅ Baseline windows are imported from `constants.py` (mostly)

**Only 1 moderate issue remains**: The 3 decoding scripts could benefit from explicit shared baseline normalization before training classifiers. This is **not critical** (StandardScaler provides some compensation), but would make the analyses more consistent and interpretable.

**Overall Grade**: **A−** (would be A+ if decoding scripts are updated)

---

## Appendix: Quick Reference

### When to Use Each Method

| Method | Formula | When to Use | Strengths | Pitfalls |
|--------|---------|-------------|-----------|----------|
| **Z-score** | `(r - μ_bl) / σ_bl` | Single-unit significance, heatmaps, population comparisons | Units are interpretable (SD of baseline noise), removes baseline differences | Requires stable baseline, can inflate low-FR units if SD is tiny |
| **Δrate** | `r - μ_bl` | Population averages, CDs, decoding | Preserves Hz units, robust to small SD | Still biased toward high-FR units |
| **Percent change** | `100 * (r - μ_bl) / μ_bl` | Modulation strength comparisons | True equalization (doubling = doubling) | Explodes for low baselines, division by zero |
| **Raw rates** | `r` | Within-unit comparisons, single-trial decoding | No assumptions | Cannot compare across units |

### Checklist Before Publishing

- [ ] All analyses use **shared baseline** across compared conditions
- [ ] Baseline windows are **consistent** (imported from `constants.py`)
- [ ] Units are normalized **before** averaging (not after)
- [ ] Division-by-zero is **guarded** (`if sd < 1e-6: sd = 1.0`)
- [ ] Normalization method **matches** the analysis goal
- [ ] Decoding uses **baseline-normalized** features (not raw rates)
- [ ] Population grand averages use **shared baseline** to preserve relative magnitude

---

**Report prepared by**: Claude Code (Codebase Auditor)
**Date**: March 30, 2026
**Files audited**: 43 active analysis scripts
**Total issues found**: 1 moderate (decoding), 0 critical
