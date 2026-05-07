# Handoff — May 2026 (Corneal SV Sync Pipeline)

## What Was Done This Session

This session focused entirely on the **corneal spatial-variance (SV) video synchronization pipeline** for the eye camera footage. Goal: achieve sub-20ms RMSE sync between camera timestamps and NI-DAQ (electrophysiology) timestamps using the grating reflection in the mouse cornea as the timing anchor.

### Primary Technical Fixes

#### 1. Coarse offset aliasing for 03072025 (root cause found and fixed)
- **Bug**: `coarse_align()` brute-force scan was degenerate when ITI≈13.4s — offsets at `true_offset` and `true_offset + 13.4s` produced equal match counts. Cached value was 14.5s (= true ~1.1s + one ITI period).
- **Fix**: Added `scan_coarse_offset()` function to `corneal_spatial_diagnostic.py`. Scan confirmed 1.5s is correct. Updated `data/cache/video_sync/coarse_offsets.json`.
- **CLI**: `py scripts/video/corneal_spatial_diagnostic.py --session 03072025 --scan-coarse`

#### 2. Autocal angle constraint: [90°,170°] → [125°,170°]
- **Bug**: The 90-125° region (near-downward from pupil, code convention 0°=right 90°=down) contains tear duct / nasal tissue that shows strong std(diff) signal from eye-movement reflexes firing 200-700ms after Baseline_ON — NOT at grating onset. Autocal was placing masks on this artifact zone.
- **Fix**: `CORNEAL_CAL_ANGLE_MIN_DEG: 125.0` in `constants.py`. This restricts to the lower-left quadrant [125°,170°] where the actual grating reflection lives (~130-160° in these sessions).

#### 3. Min-inlier guard: 50% → 30%  
- **Bug**: `fit_clock_model()` in `video_sync.py` stopped iterative MAD outlier rejection at `max(20, 0.5×n_initial)` anchors. With relaxed detection thresholds (required for weak-signal sessions), this guard prevented the final convergence iterations from running.
- **Fix**: Changed to `int(0.3 * n_original)`. Allows rejection to continue until MAD stabilizes.
- **Impact**: 03072025 RMSE improved from 64.7ms → 10.9ms purely due to allowing 3 more iterations.

#### 4. Outlier rejection iterations: 3 → 10
- **Fix**: `VIDEO_SYNC_OUTLIER_N_ITER: 10` in `constants.py`. Needed for convergence in weak-signal sessions (29082025 required 7 iterations to reach MAD=14ms).

#### 5. Tight 20×20 manual ROIs for 3 sessions
- **Problem**: Large autocal masks (300-450px²) were mixing grating-responsive pixels with eye-movement-sensitive pixels, causing false positive detections at +400-700ms post-onset.
- **Solution**: Use small manually-specified ROIs (20×20px, radius=8) centered exactly on the autocal-identified reflection centroid. These exclude the surrounding contaminated pixels.
- `CORNEAL_EYE_ROI["03072025"] = (257, 277, 377, 397)` — center y:267, x:387 (from autocal bbox y:265-269, x:384-389)
- `CORNEAL_EYE_ROI["27062025"] = (247, 267, 397, 417)` — center y:257, x:407 (from autocal bbox y:238-276, x:392-421)
- `CORNEAL_EYE_ROI["29082025"] = (318, 338, 419, 439)` — center y:328, x:429 (from autocal bbox y:314-343, x:415-444)
- Run with: `--no-auto-calibrate` flag so the manual ROI is used

#### 6. Per-session detection sensitivity overrides
Added to `CORNEAL_DETECT_PARAMS` dict in `corneal_spatial_diagnostic.py`:
- `"03072025": (1.5, 1.05)` — very weak signal due to eye movements blurring the reflection
- `"27062025": (1.5, 1.05)` — same issue as 03072025
- `"29082025": (2.0, 1.10)` — moderate signal reduction
- `"14082025": (2.0, 1.10)` — carried over from before
- Default: `(3.0, 1.25)` for clean sessions (09092025)

---

## Session Sync Status — All 5 Local Sessions

| Session | Stage | Detection | RMSE | Quality | Method | Notes |
|---------|-------|-----------|------|---------|--------|-------|
| 09092025 | Expert | ~89% | **5.7ms** | good | autocal (305-335, 433-468, r=12) | Ground truth; unchanged this session |
| 03072025 | Learning | 55% | **10.9ms** | good | manual tight 20×20 + `--no-auto-calibrate` | Fixed: offset aliasing + tight ROI + (1.5,1.05) |
| 29082025 | ? | 95% | **20.1ms** | review | manual tight 20×20 + `--no-auto-calibrate` | 7 iterations to converge |
| 14082025 | ? | 43% | **21.4ms** | review | autocal 208px + (2.0,1.10) | First successful run |
| 27062025 | Naive? | 36% | ~300ms | **failed** | manual tight 20×20 — does not work | SV baseline too high (5.59 vs ~3.0 working sessions) |

### 27062025 Failure Hypothesis
The pre-baseline window `[-2s, -1s]` before expected Baseline_ON lands ~2-3s after the previous trial ends (ITI≈13.4s). In this early session, the stimulus screen may not return to a blank/uniform state fast enough — the grating from the previous trial may still be visible on the cornea at -2s relative to the next trial's Baseline_ON. This makes the "before-after" SV comparison ambiguous (both periods show grating), producing high baseline SV (ref=5.59) and near-random detection.

**Potential fix** (not yet implemented): Deepen the baseline window from `[-2s, -1s]` to `[-5s, -4s]` relative to expected onset. This would sample further into the ITI where the screen should be blank. The constants `BL_START_MS = -2000.0` and `BL_END_MS = -1000.0` in `detect_corneal_onset_in_trace()` (in `corneal_spatial_diagnostic.py`, ~line 760) would need to become configurable per session.

---

## Committed Changes (commit `1f6434d`)

### Key library changes:
- `src/visdetect/core/video_sync.py`: min-inlier guard 0.5→0.3
- `src/visdetect/analysis/constants.py`:
  - `VIDEO_SYNC_OUTLIER_N_ITER: 3→10`
  - `VIDEO_SYNC_DERIV_MAX_THRESH: 15.0` (new — caps adaptive threshold)
  - `CORNEAL_CAL_ANGLE_MIN_DEG: 125.0` (was lower, now correct)
  - `CORNEAL_CAL_MASK_THRESHOLD_PCT: 65` (top 35%, permissive)
  - `PUPIL_*` and `MOTION_ENERGY_*` constants (for future feature extraction)

### New scripts added:
- `scripts/video/corneal_spatial_diagnostic.py` — the main corneal SV sync script (~2000 lines)
- `scripts/video/characterize_camera_signal.py` — camera SNR/detection analysis
- `scripts/video/batch_sync_sessions.py` — batch sync stub
- `scripts/video/compare_mask_sync.py` — diagnostic for mask comparison
- `scripts/video/poc_multianchor_sync.py` — multi-anchor experimental approach
- `scripts/video/select_roi.py` — interactive ROI selection
- `scripts/pipelines/run_tprime.py` — TPrime spike-time correction
- `src/visdetect/core/spikeglx.py` — SpikeGLX metadata parser

---

## Cached Data Files (not git-tracked)

All in `data/cache/video_sync/`:

| File | Content |
|------|---------|
| `coarse_offsets.json` | `{"27062025":9.0, "03072025":1.5, "14082025":3.0, "29082025":5.5, "09092025":4.0}` |
| `{session}_corneal_sync.json` | Final clock model (RMSE, slope, offset, quality, n_anchors) per session |
| `corneal_cal/{session}_corneal_roi_cal.json` | Autocal result (pupil_center, bbox, mask_area, quality) |
| `corneal_cal/{session}_corneal_mask.npz` | Binary autocal mask (saved as `mask` key) |

---

## CLI Reference for corneal_spatial_diagnostic.py

```bash
# Full session sync (uses autocal by default):
py scripts/video/corneal_spatial_diagnostic.py --session 09092025 --full-session

# Full session with manual ROI (for 03072025/27062025/29082025):
py scripts/video/corneal_spatial_diagnostic.py --session 03072025 --full-session --no-auto-calibrate

# Re-run autocal (force):
py scripts/video/corneal_spatial_diagnostic.py --session 29082025 --auto-calibrate --force

# Scan for correct coarse offset (fixes ITI aliasing):
py scripts/video/corneal_spatial_diagnostic.py --session 03072025 --scan-coarse

# Show autocal diagnostic figure:
py scripts/video/corneal_spatial_diagnostic.py --session 14082025 --auto-calibrate
```

---

## Immediate Next Steps (Priority Order)

### 1. Fix 27062025 baseline SV issue (HIGHEST)
**Action**: Modify `detect_corneal_onset_in_trace()` to accept configurable baseline window constants (currently hardcoded `BL_START_MS = -2000.0`). Add a per-session dict `CORNEAL_BASELINE_WINDOW` and test with `(-5000.0, -4000.0)` for 27062025.

**Expected outcome**: SV baseline should drop to ~3.0 (matching other sessions), enabling clean detection.

### 2. Verify 29082025 and 14082025 figures
Both are quality="review" (RMSE ~20ms). Run:
```bash
py scripts/video/corneal_spatial_diagnostic.py --session 29082025 --full-session --no-auto-calibrate
py scripts/video/corneal_spatial_diagnostic.py --session 14082025 --full-session
```
Inspect figures: `figures/video_sync/corneal_spatial/{session}_full_session_corneal.png`

### 3. Build batch sync infrastructure
`batch_sync_sessions.py` exists as a stub. Needs to:
- Iterate over all QC-passing sessions that have local video files
- Run `corneal_spatial_diagnostic.py --full-session` for each
- Collect RMSE results into a summary CSV
- Currently only 5 sessions have local video copies (see below)

### 4. Acquire more local video copies
Local videos: `data/videos/BG_046_*_Eye_cam.mp4` — currently 5 sessions (~20-25GB each).
Source: `X:/public/projects/BeJG_20230130_VisDetect/wEPhys/Cameras_sortIntoSubjects/BG_046_DDMMYY/Eye_cam*.mp4`
~28 remaining QC-passing sessions need to be synced.

### 5. Pupil extraction pipeline
New constants `PUPIL_EYE_ROI`, `PUPIL_BLUR_KERNEL`, etc. are defined in `constants.py` but marked as PLACEHOLDER. Once ROIs are validated via `select_roi.py`, implement batch pupil extraction and link to trial timestamps via the corneal sync model.

### 6. Deeper baseline window for early sessions
For sessions like 27062025 where the ITI screen state is uncertain, implement:
```python
CORNEAL_BASELINE_WINDOW = {
    "27062025": (-5000.0, -4000.0),   # deeper into ITI
    # default: (-2000.0, -1000.0)
}
```

---

## Algorithm Reference

### How corneal SV sync works
1. **Screen-glow coarse model**: Detects luminance drop in background ROI (two strips above/below mouse head at x>600) at Baseline_ON. Gives ±190ms RMSE clock model — enough to place per-trial windows.
2. **Corneal SV extraction**: For each trial, extract ~3s SV trace from corneal ROI (within-frame spatial std). Window centered on screen-glow prediction.
3. **Onset detection (two-pass)**:
   - Pass 0: Search [-200,+500ms] around predicted onset, require 5-frame sustain + ratio≥threshold
   - Pass 1 (fallback): Full range scan, 15-frame sustain
4. **Clock model**: Theil-Sen regression → iterative MAD outlier rejection (10 iterations, min 30% inliers) → OLS refit on clean inliers.
5. **Output**: `{session}_corneal_sync.json` with final RMSE, slope, offset, anchor count.

### Why corneal SV beats screen glow
- Screen glow RMSE: ~190ms (jitter from luminance change timing uncertainty)
- Corneal SV RMSE: ~6-21ms (limited by frame quantization ~6ms + grating phase ~14ms)
- The grating SPATIAL pattern is instantaneous; temporal luminance changes have phase jitter.

### Angle convention in autocal
Code uses: 0°=right, 90°=down, measured CCW (atan2 convention in image coordinates where y increases downward). The grating reflection is at ~130-160° (lower-left of pupil). Tear duct artifacts are at 90-125° (near-downward).

User's description "180-270 degrees" uses compass convention (CW from top): 180°=down, 270°=left → equivalent to code [90°,180°]. The angular constraint [125°,170°] is a subset focusing on the lower-left quadrant excluding the near-downward tear duct zone.

---

## Files to Know

| File | Purpose |
|------|---------|
| `scripts/video/corneal_spatial_diagnostic.py` | Main pipeline (~2000 lines) |
| `src/visdetect/core/video_sync.py` | Library: `fit_clock_model()`, `auto_calibrate_corneal_roi()`, `build_corneal_mask()` |
| `src/visdetect/analysis/constants.py` | All VIDEO_SYNC_* and CORNEAL_CAL_* constants |
| `data/cache/video_sync/coarse_offsets.json` | Per-session camera-to-NIDAQ coarse offsets |
| `data/cache/video_sync/*_corneal_sync.json` | Final sync results per session |
| `figures/video_sync/corneal_spatial/` | All diagnostic figures |
