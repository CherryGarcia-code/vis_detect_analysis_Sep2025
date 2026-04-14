# Video-to-NIDAQ Temporal Synchronization Plan

## Problem Statement

Camera timestamps (USB clock, ms) are NOT aligned to the NI-DAQ master clock used for all neural and behavioral timing. Measured drift between clocks is **~16.5s over 2.6 hours (~1750 ppm)** — far too large for a single-offset correction. The eye cam and front cam also drift relative to each other (~1.4s). A robust, multi-anchor synchronization pipeline is needed.

**Hardware sync is unavailable**: Camera trigger channels (`Eye_cam`, `Front_cam`, `Top_cam`) were never mapped in SpikeGLX — all contain placeholder `[0, 1]` uint64 values in both raw `.mat` files and derived `.pkl` files. Verified via h5py inspection of `BG_046_01072025.mat`. Luminance-based sync is the only viable path.

## Data Inventory (per session)

| Stream | Clock | Rate | Resolution | Duration | Frames |
|--------|-------|------|-----------|----------|--------|
| Eye cam (976x1024 H.264 MP4) | USB camera | ~50 fps | 976x1024 | ~2.6 hr | ~470K |
| Front cam (640x272 H.264 MP4) | USB camera | ~100 fps | 640x272 | ~2.6 hr | ~940K |
| Eye cam metadata CSV | USB camera (ms) | per-frame | 3 cols: Timestamp, Acquired, Saved | matches video | matches |
| Front cam metadata CSV | USB camera (ms) | per-frame | same | matches video | matches |
| NI-DAQ events | NI-DAQ (s) | event-driven | sub-ms | ~2.6 hr | — |
| NI-DAQ Synch signal | NI-DAQ (s) | ~1 Hz | sub-ms | full session | ~9400 |

**Key NI-DAQ events**: `Baseline_ON` (~600-650 per session), `Change_ON`, `Valve_L` (reward), `Synch` (~1 Hz, full session span).

**Total data**: ~5.9 TB across 52 sessions (~35 GB/session average).

**Metadata details**: CSV has terminal zero-row (last row timestamp=0.0, drop it). Frame intervals: median=19.985ms, std=0.411ms for eye cam. All frames saved (Saved == Acquired for the tested session).

## Synchronization Strategy: Multi-Anchor Luminance-Based Alignment

### Approach Overview

1. **Extract luminance time series** from video frames (ROI around stimulus-visible area)
2. **Detect stimulus transitions** (gray ↔ grating) as step changes in luminance/variance
3. **Match detected transitions** to known `Baseline_ON` events from NI-DAQ
4. **Fit a linear clock model**: `t_nidaq = slope * t_camera + offset`
5. **Validate** on held-out anchors and cross-camera consistency
6. **Store** per-session sync parameters as sidecar files for downstream use

### Why Multi-Anchor (Not Single-Point)

- 16.5s drift over 2.6 hours means a single sync point at the start gives **>10s error** by session end
- With ~600 Baseline_ON events spanning the full session, we get ~600 anchor candidates
- Linear fit absorbs both offset AND drift rate
- Residuals quantify sync precision (expected: < 1 frame = 20ms)

---

## Implementation Plan

### Phase 1: Luminance Extraction (per session, per camera)

**Input**: MP4 video file + metadata CSV
**Output**: 1D luminance time series (one value per frame) + spatial variance time series + camera timestamps

1. Open video with OpenCV (`cv2.VideoCapture`)
2. Define ROI for stimulus area:
   - **Auto-detection strategy**: For ~100 frames spanning a known trial onset, compute per-pixel temporal variance. The region with the highest variance change between gray→grating frames is the stimulus-affected area. Extract a bounding box from this variance map.
   - **Fallback**: Hardcode a seed ROI from the first successfully processed session; allow per-session override via config dict.
   - Eye cam: the screen reflection/illumination region (side of head)
   - Front cam: ambient illumination region (lower priority, sync from eye cam preferred)
3. For each frame:
   - Convert to grayscale
   - Compute **mean pixel intensity** in ROI (primary signal)
   - Compute **spatial variance** in ROI (secondary signal — grating has high spatial structure, gray screen is uniform; more robust when mean luminance change is small)
4. Store as `(camera_timestamps_ms, mean_luminance, spatial_variance)` arrays in NPZ cache
5. **Optimization**: Downsample spatially (every 4th pixel in ROI) for speed. Full temporal resolution required (every frame).

**Performance target**: Process 470K frames in < 30 min per camera (achievable with OpenCV + spatial downsampling)

**Metadata handling**:
- Drop terminal zero-row from CSV
- Verify monotonicity of timestamps
- Check Saved vs Acquired for dropped frames — flag sessions where ratio < 0.99

### Phase 2: Transition Detection

**Input**: Luminance + spatial variance time series
**Output**: Array of detected transition times (camera clock, ms)

1. **Primary signal**: Use whichever of mean luminance or spatial variance shows stronger step changes (auto-select based on SNR of detected transitions)
2. **Smooth** trace (median filter, window = `VIDEO_SYNC_SMOOTH_FRAMES` frames, default 5 = 100ms)
3. **Compute derivative** (np.diff of smoothed trace)
4. **Threshold detection** with hysteresis:
   - High threshold to detect candidate transitions: `VIDEO_SYNC_DETECT_THRESH` (default: MAD-based, 5× median absolute deviation)
   - Low threshold to confirm: 0.5× high threshold
5. **Cluster** nearby detections (within `VIDEO_SYNC_CLUSTER_MS`, default 200ms) and take the first in each cluster
6. **Classify**: rising edge = trial onset (gray → grating at Baseline_ON), falling edge = trial offset / ITI
7. Return arrays of rising-edge times and falling-edge times

**Robustness**:
- Validate: number of detected rising edges should approximately match number of Baseline_ON events (within ±10%)
- If detection fails with primary signal, fallback to secondary signal

### Phase 3: Event Matching & Clock Model Fitting

**Input**: Detected camera transitions (ms), NI-DAQ `Baseline_ON` times (s)
**Output**: Clock model parameters (slope, offset) + quality metrics

1. **Initial coarse alignment**:
   - Camera transitions are in ms from camera start; NI-DAQ events in s from NI-DAQ start
   - Estimate rough offset by cross-correlating the inter-event intervals (robust to unknown absolute offset)
   - Method: compute pairwise inter-event-interval vectors for both streams, find offset that minimizes sum-of-squared-differences

2. **Greedy nearest-neighbor matching**:
   - After coarse alignment, match each camera transition to the nearest NI-DAQ Baseline_ON
   - Reject matches with residual > `VIDEO_SYNC_MATCH_REJECT_S` (default 1.0s)

3. **RANSAC linear regression**:
   - Model: `t_nidaq = slope * t_camera_ms / 1000 + offset`
   - RANSAC with inlier threshold = `VIDEO_SYNC_RANSAC_THRESH_S` (default 0.1s = 5 frames at 50fps)
   - Expected: slope ≈ 1.0018 (from 1750 ppm drift), offset = session-specific
   - Use `sklearn.linear_model.RANSACRegressor`

4. **Refit on inliers** with ordinary least squares for final parameters and confidence intervals

5. **Quality metrics** (all stored in output):
   - `n_anchors`: Number of matched anchor points (expect >90% of Baseline_ON events)
   - `rmse_ms`: RMSE of residuals in ms (expect < 20ms = 1 frame)
   - `max_residual_ms`: Max absolute residual (flag if > 40ms = 2 frames)
   - `slope_ppm`: Slope deviation from 1.0 in parts-per-million (flag if > 5000 ppm)
   - `cv_rmse_ms`: Cross-validated RMSE (Phase 4)

### Phase 4: Validation

1. **Cross-validation**: Hold out 20% of anchor points, fit on 80%, predict held-out Baseline_ON times. Report CV RMSE. Repeat 5 times (5-fold) and report mean ± SD.
2. **Cross-camera validation**: Fit eye cam and front cam independently. For Baseline_ON events detected in both cameras, their predicted NI-DAQ times should agree within ~20ms. Report cross-camera RMSE.
3. **Residual analysis**: Plot residuals vs time — should be flat (no systematic curvature). If residuals show curvature → upgrade to piecewise linear (breakpoints every 30 min) or quadratic model.
4. **Diagnostic figure** (per session, saved to `figures/video_sync/`):
   - Panel A: Raw luminance + spatial variance traces with detected transitions marked
   - Panel B: Matched anchor points (camera time vs NI-DAQ time) with regression line
   - Panel C: Residuals vs time (should be flat)
   - Panel D: Residual histogram with RMSE annotation

### Phase 5: Apply & Store

1. **Conversion function**: `camera_to_nidaq(t_camera_ms, slope, offset) → t_nidaq_s`
2. **Per-session output** saved as JSON sidecar file in `data/cache/video_sync/`:
   ```python
   {
       "session_name": "01072025",
       "eye_cam": {
           "slope": 1.00175,
           "offset": -15.234,
           "n_anchors": 612,
           "rmse_ms": 8.3,
           "max_residual_ms": 18.7,
           "cv_rmse_ms": 9.1,
           "slope_ppm": 1750.3,
           "roi": [y0, y1, x0, x1],
           "n_frames": 471805,
           "n_dropped": 0
       },
       "front_cam": { ... },
       "quality": "good"  // "good", "review", "failed"
   }
   ```
3. **Library helper**: `load_video_sync(session_name) → dict` loads the sidecar JSON. Returns `None` if not available. Follows the same pattern as HMM state assignments (separate file, loaded independently). **Do NOT modify the Session dataclass.**
4. **Downstream usage**: Any analysis needing camera timestamps calls `camera_to_nidaq()` to convert to NI-DAQ seconds.

### Phase 6: Batch Processing

1. Prioritize the ~28 sessions with QC-passing neural data (from `load_staging_manifest(qc_only=True)`). Process remaining sessions only if needed.
2. Run Phases 1-5 for each session. Cache luminance/variance traces (NPZ) and sync parameters (JSON).
3. Generate summary report:
   - Sessions by quality tier: good / review / failed
   - Aggregate statistics: median RMSE, drift range, detection rate
   - Flag sessions needing manual ROI adjustment or parameter tuning

---

## Code Organization

```
src/visdetect/core/video_sync.py       # Library: luminance extraction, transition detection, clock fitting, load/apply
scripts/video/                          # CLI scripts
  extract_luminance.py                  # Phase 1: batch luminance extraction
  fit_video_sync.py                     # Phases 2-4: fit and validate sync per session
  validate_video_sync.py                # Phase 4 standalone: cross-validation report
data/cache/video_sync/                  # Per-session sync params (JSON) + luminance caches (NPZ)
figures/video_sync/                     # Per-session diagnostic figures
```

**Note**: Library module lives in `src/visdetect/core/` (not `analysis/`) because clock synchronization is infrastructure (analogous to TPrime spike-time correction in `core/ingest.py`), not scientific analysis.

## Constants (to be added to `visdetect/analysis/constants.py`)

```python
# =====================================================================
# Video synchronization parameters
# =====================================================================
VIDEO_SYNC_SMOOTH_FRAMES: int = 5           # Median filter window for luminance trace
VIDEO_SYNC_DETECT_THRESH: float = 5.0       # MAD multiplier for transition detection
VIDEO_SYNC_CLUSTER_MS: float = 200.0        # Cluster nearby detections within this window
VIDEO_SYNC_MATCH_REJECT_S: float = 1.0      # Reject matches with residual > this
VIDEO_SYNC_RANSAC_THRESH_S: float = 0.1     # RANSAC inlier threshold (seconds)
VIDEO_SYNC_MAX_DRIFT_PPM: float = 5000.0    # Flag sessions with drift > this
VIDEO_SYNC_MAX_RESIDUAL_MS: float = 40.0    # Flag if max residual > 2 frames
```

## Config paths (to be added to `visdetect/analysis/config.py`)

```python
# Camera data root (may be on a different drive from main data)
CAMERA_ROOT = os.getenv("VISDETECT_CAMERA_ROOT",
    os.path.join("X:/public/projects/BeJG_20230130_VisDetect/wEPhys/Cameras_sortIntoSubjects"))
VIDEO_SYNC_DIR = os.path.join(ROOT, "data", "cache", "video_sync")
VIDEO_SYNC_FIG_DIR = os.path.join(ROOT, "figures", "video_sync")
```

## Dependencies

- `opencv-python-headless` (cv2) — video frame reading (headless avoids GUI deps)
- `numpy`, `scipy` (already installed) — signal processing
- `scikit-learn` (already installed) — RANSACRegressor

Install: `pip install opencv-python-headless`

## Session Name Mapping

Camera directories use `DDMMYY` format (e.g., `BG_046_010725`), while `.pkl` files use `DDMMYYYY` (e.g., `BG_046_01072025`). Add `camera_dir_to_session_name()` helper to resolve between formats, using existing `parse_session_date()` from `config.py`.

## Precision Budget

| Source | Contribution |
|--------|-------------|
| Camera frame interval | ±10ms (eye cam @ 50fps) — hard floor |
| Luminance detection jitter | ±5ms (with slope interpolation, sub-frame) |
| Clock model residual | ±10ms (expected from linear fit) |
| **Total expected precision** | **~15-20ms (1σ)** |

**Note on frame-level floor**: The fundamental limit is the frame exposure time (~20ms at 50fps). Sub-frame interpolation (fitting the luminance transition slope) can improve to ~5ms, but cannot go below the camera's internal exposure/readout time. The front cam at 100fps gives ±5ms resolution natively.

This precision is adequate for: pupil diameter (slow, ~100ms timescale), motion energy (50-100ms features), arousal state (seconds timescale). It is marginal for: precise motor onset timing (would need front cam at 100fps → ±5ms precision).

## Edge Cases & Failure Modes

1. **Luminance signal too weak**: Some sessions may have poor screen visibility in eye cam. Fallback: use spatial variance signal, front cam ambient light, or reward delivery (visible water drop) as secondary anchors.
2. **Non-linear drift**: If residuals show curvature, upgrade to piecewise linear (breakpoints every 30 min) or quadratic model.
3. **Missing frames / dropped frames**: Metadata CSV tracks Acquired vs Saved. If Saved < Acquired, the frame-number-to-video-frame mapping is offset. Flag and adjust.
4. **Session with very few trials**: Sessions with <50 trials may have insufficient anchors for robust RANSAC. Flag these and use wider inlier threshold.
5. **Session name format**: Camera dirs use `DDMMYY` vs pkl `DDMMYYYY`. Handle via `camera_dir_to_session_name()`.
6. **Mean luminance direction ambiguity**: Gray→grating may not always increase mean luminance (grating mean ≈ gray mean). Spatial variance is the robust fallback (always increases for gray→grating).
