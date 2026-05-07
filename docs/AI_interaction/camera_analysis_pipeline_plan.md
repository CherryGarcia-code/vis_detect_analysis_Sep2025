# Camera Analysis Pipeline — Implementation Plan

> **Companion files**: `FIGURES/video_sync/video_sync_notes.md` (methods & results for completed sync work), `docs/AI_interaction/camera_analysis_pipeline_plan.md` (copy of this plan for version control).

## Context

We have validated video-to-NI-DAQ temporal synchronization for one session (01072025, RMSE=17.4ms, quality="good") using derivative-based onset detection of screen glow in the eye camera. The sync infrastructure is mature (`src/visdetect/core/video_sync.py`, 1024 lines). This plan covers scaling that to all sessions, then extracting behavioral features (motion energy, pupil diameter) from the camera data, and integrating them with the existing neural and behavioral analyses.

**Why**: Pupil diameter is a well-established arousal proxy — correlating it with HMM engagement states validates both measures. Motion energy from the mouth region can provide more precise lick onset timing than spout contact (~200ms hardware delay), improving neural alignment.

**What exists**: Complete sync library, 1/25 sessions synced, no pupil or motion energy code yet. Eye cam at 50fps, front cam at ~100fps, 52 camera directories available.

---

## Phase 0: Batch Sync (~25 Sessions)

**Goal**: Sync all QC-passing sessions, produce summary report.

### New file: `scripts/video/batch_sync_sessions.py`
- Iterate `load_staging_manifest(qc_only=True)` (~25 sessions)
- For each: load session pkl → extract `Baseline_ON` from `sess.ni_events` → call `sync_session()`
- `sync_session()` already has skip-if-cached logic, handles file discovery, fitting, JSON + diagnostic PNG output
- After sync, generate a per-session mouth-zoom validation figure (analogous to `01072025_sync_validation_mouth_zoom.png`):
  - Reuse `sync_validation_figure.py` logic: find lick-responsive neuron, select 5 Hit + 5 FA trials, extract video frames around lick, show spike rasters (corrected + uncorrected) and PSTH with bootstrap CI
  - Save to `figures/video_sync/{session}_sync_validation_mouth_zoom.png`
  - Skip if neuron selection fails (e.g., no significant lick-responsive neurons in that session)
- Save `data/cache/video_sync/batch_sync_summary.csv` with columns: session_name, stage, quality, rmse_ms, coverage, n_anchors, slope_ppm, elapsed_s, validation_fig (bool)
- CLI: `py scripts/video/batch_sync_sessions.py [--force] [--skip-validation]`

### Outputs
- `data/cache/video_sync/{session}_video_sync.json` per session
- `figures/video_sync/{session}_eye_cam_sync.png` per session (4-panel diagnostic)
- `figures/video_sync/{session}_sync_validation_mouth_zoom.png` per session (mouth-zoom validation with rasters + PSTH)
- `data/cache/video_sync/batch_sync_summary.csv`

### Implementation note: validation figure generation
- Extract the core validation logic from `sync_validation_figure.py` into a reusable function in `camera_features.py` (or a helper in `video_sync.py`): `generate_sync_validation_figure(session_name, sess, sync_params, save_path)`
- The batch script calls this function after each successful sync
- `--skip-validation` flag to speed up batch runs when only sync JSONs are needed

### Verification
- Count JSON files ≈ 25; all "good" sessions RMSE < 20ms
- Spot-check 2-3 diagnostic PNGs: residuals should look random, not systematic
- Sessions with missing camera dirs logged as "no_camera"
- Per-session validation mouth-zoom PNGs visually confirm temporal coherence
- **Optional (statistician recs)**: (a) Fit quadratic model on first 3-5 sessions to confirm linear model is adequate (if quadratic term is negligible, linear is justified). (b) Report first/last CV fold RMSE separately to assess extrapolation quality. (c) Check ROI consistency across sessions (camera position may shift if headpost was adjusted).

---

## Phase 1: Motion Energy Extraction

**Goal**: Frame-to-frame absolute pixel differences in mouth ROI. Standard formulation per Stringer et al. (Science 2019), Musall et al. (Nat Neurosci 2019).

**Depends on**: Phase 0 (sync JSONs for time alignment).

### New file: `src/visdetect/core/camera_features.py`

Core library for all camera feature extraction. Key motion energy functions:

```python
def compute_motion_energy(
    video_path: str, metadata_path: str,
    roi: Tuple[int,int,int,int],  # (y0, y1, x0, x1) — mouth region
    downsample: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame mean |frame[t] - frame[t-1]| in ROI (float32 to avoid uint8 underflow).
    Returns (timestamps_ms, motion_energy). First frame = 0."""

def save_motion_energy(session_name, timestamps_ms, motion_energy, out_dir=None) -> str:
    """Save to {MOTION_ENERGY_DIR}/{session}_motion_energy.npz"""

def load_motion_energy(session_name, out_dir=None) -> Optional[Dict]:
    """Load cached NPZ. Returns None if missing."""

def extract_motion_energy_session(session_name, roi=None, force=False) -> Optional[Dict]:
    """End-to-end: check cache → find files → compute → save → return."""
```

Alignment utilities (also in `camera_features.py`):
```python
def align_feature_to_nidaq(timestamps_ms, feature, slope, offset):
    """Convert camera timestamps to NI-DAQ seconds using sync model."""

def get_feature_around_event(nidaq_times_s, feature, event_time_s, window):
    """Extract feature values in time window around an event."""
```

### New file: `scripts/video/batch_extract_motion_energy.py`
- Iterate synced sessions (quality "good" or "review")
- Call `extract_motion_energy_session()` for each
- CLI: `py scripts/video/batch_extract_motion_energy.py [--force]`

### Outputs
- `data/cache/motion_energy/{session}_motion_energy.npz` per session (~3.6 MB each)

### Verification
- NPZ frame count matches sync JSON `n_frames` (471805 for 01072025)
- Plot motion energy vs time: clear spikes around lick events
- `motion_energy[0] == 0`

---

## Phase 2: Pupil Extraction

**Goal**: Track pupil diameter using adaptive threshold + ellipse fitting. Standard approach for head-fixed mice with IR illumination (dark pupil against bright iris).

**Depends on**: Phase 0. Independent of Phase 1.

### Add to `src/visdetect/core/camera_features.py`:

```python
def detect_pupil_frame(
    gray_frame, roi, blur_kernel=5, thresh_block_size=51,
    min_area=200, max_area=15000, min_roundness=0.3,
) -> Optional[Dict[str, float]]:
    """Single-frame pupil detection.
    Algorithm: crop ROI → Gaussian blur → adaptive threshold (BINARY_INV) →
    morphological close → find contours → filter by area → fit ellipse →
    validate roundness. Returns {center_x, center_y, major_axis, minor_axis,
    angle, diameter, area} or None."""

def extract_pupil_trace(video_path, metadata_path, roi, **kwargs):
    """Full-video pupil trace. Returns (timestamps_ms, diameter, center_x, center_y).
    NaN where detection failed."""

def interpolate_pupil_blinks(diameter, timestamps_ms, max_gap_ms=500.0):
    """Linearly interpolate short NaN gaps (blinks). Leave long gaps as NaN."""

def save_pupil_data(session_name, timestamps_ms, diameter, diameter_interp, 
                    center_x, center_y, out_dir=None) -> str:
    """Save to {PUPIL_DIR}/{session}_pupil.npz"""

def load_pupil_data(session_name, out_dir=None) -> Optional[Dict]:

def extract_pupil_session(session_name, roi=None, force=False) -> Optional[Dict]:
    """End-to-end: cache check → find files → extract → interpolate → save.
    Also extracts mean eye-ROI luminance per frame (for confound control)."""
```

### Add luminance extraction to pupil pipeline (confound control)
- During `extract_pupil_trace()`, also compute mean luminance in the eye ROI per frame
- Save alongside pupil data in the NPZ: `luminance` key
- Downstream analyses can include luminance as a covariate to control for stimulus-driven pupil changes

### New file: `scripts/video/validate_roi.py`
- **MUST run before batch extraction** to calibrate both `PUPIL_EYE_ROI` and `MOTION_ENERGY_MOUTH_ROI`
- Display grid of ~20 sample frames with red ellipse overlay on detected pupil + green rectangle on mouth ROI
- Show pupil detection rate and full diameter trace strip
- CLI: `py scripts/video/validate_roi.py [--session 01072025]`
- Output: `figures/video_sync/{session}_roi_validation.png`

### New file: `scripts/video/batch_extract_pupil.py`
- Same pattern as motion energy batch script
- Reports per-session detection rate (target: >85%)

### Outputs
- `data/cache/pupil/{session}_pupil.npz` per session (~9 MB each)

### Verification
- Run `validate_pupil_roi.py` FIRST — confirm ellipses overlay correctly on actual eye
- Detection rate >85% for well-lit sessions
- Diameter values physically reasonable (20-80 pixels typical for head-fixed mouse)
- Blink interpolation fills short gaps, leaves long gaps as NaN

---

## Phase 3: Front Camera Sync

**Goal**: Derive front cam timing from eye cam clock model (same USB clock domain).

**Depends on**: Phase 0.

### Add to `src/visdetect/core/video_sync.py`:

```python
def derive_front_cam_sync(session_name, camera_root=None, sync_dir=None) -> Optional[dict]:
    """Both cameras share USB clock. Eye cam model applies directly.
    Adjust offset by first-frame timestamp difference between cameras:
      front_offset = eye_offset + eye_slope * (delta_first_frame_ms / 1000)
    Update sync JSON with front_cam entry. Returns updated dict."""
```

### New file: `scripts/video/derive_front_cam_sync.py`
- Quick operation — reads only metadata CSVs, no video processing
- CLI: `py scripts/video/derive_front_cam_sync.py`

### Verification
- Front cam metadata confirms ~100 fps (10ms inter-frame)
- Derived slope identical to eye cam slope
- Spot-check: convert known NI-DAQ event to front cam time, verify correct frame

---

## Phase 4: Analysis Suite Integration

**Goal**: New `analysis_suite/10_camera/` module with 5 publication figures.

**Depends on**: Phases 0-2 complete.

### Modify `analysis_suite/loader.py` — add loaders:
```python
def load_motion_energy(session_name) -> Optional[Dict]
def load_pupil_data(session_name) -> Optional[Dict]
def load_video_sync_params(session_name) -> Optional[dict]
```

### New scripts (all follow standard template):

| Script | Figure | Key Panels |
|--------|--------|------------|
| `a_sync_summary.py` | Fig 44 | A: RMSE by session, B: quality tier counts, C: clock drift, D: coverage |
| `b_motion_energy_overview.py` | Fig 45 | A: example ME trace, B: ME @ lick (Hit vs FA), C: ME @ Change_ON (Hit vs Miss), D: baseline ME by stage |
| `c_pupil_dynamics.py` | Fig 46 | A: example pupil trace, B: mean pupil by session, C: event-triggered pupil, D: detection quality |
| `d_pupil_hmm_states.py` | Fig 47 | A: pupil by HMM state (violin), B: per-trial pupil colored by state, C: pupil predicts Engaged (ROC/AUC), D: pupil around state transitions |
| `e_motion_energy_lick_timing.py` | Fig 48 | A: single-trial ME + spout overlay, B: ME-detected onset vs spout time, C: time difference histogram (expect ~200ms), D: neural PSTH at ME onset vs spout |

### Register in `analysis_suite/run_all.py`

---

## Phase 5: Cross-Modal Neural Analysis

**Goal**: Connect camera features to neural population data and the AND-gate framework.

**Depends on**: Phase 4 + existing CD cache from `03_population/a_coding_direction.py`.

| Script | Figure | Key Panels |
|--------|--------|------------|
| `f_pupil_neural_correlates.py` | Fig 49 | A: population PSTH by pupil quartile, B: single-unit modulation by pupil, C: CD projection by pupil state, D: d' conditioned on pupil |
| `g_arousal_state_neural.py` | Fig 50 | A: task-state CD by pupil, B: sensory CD by pupil, C: 2D decomposition colored by pupil, D: summary (arousal modulates readiness?) |

### Register in `run_all.py`

---

## Constants & Config Additions

### `src/visdetect/analysis/constants.py` (after VIDEO_SYNC block):
```python
# Motion energy
MOTION_ENERGY_MOUTH_ROI: tuple = (700, 1000, 100, 500)  # CALIBRATE from frames
MOTION_ENERGY_DOWNSAMPLE: int = 2

# Pupil extraction
PUPIL_EYE_ROI: tuple = (50, 450, 150, 600)  # CALIBRATE from frames
PUPIL_BLUR_KERNEL: int = 5
PUPIL_THRESH_BLOCK_SIZE: int = 51
PUPIL_MIN_AREA: int = 200
PUPIL_MAX_AREA: int = 15000
PUPIL_MIN_ROUNDNESS: float = 0.3
PUPIL_BLINK_MAX_GAP_MS: float = 500.0
```

### `src/visdetect/analysis/config.py`:
```python
MOTION_ENERGY_DIR = os.path.join(ROOT, "data", "cache", "motion_energy")
PUPIL_DIR = os.path.join(ROOT, "data", "cache", "pupil")
```

**ROI values are placeholders** — `validate_pupil_roi.py` MUST be run to calibrate before batch extraction.

---

## File Inventory

| Phase | New Files | Modified Files |
|-------|-----------|----------------|
| 0 | `scripts/video/batch_sync_sessions.py` | `constants.py`, `config.py` |
| 1 | `src/visdetect/core/camera_features.py`, `scripts/video/batch_extract_motion_energy.py` | — |
| 2 | `scripts/video/validate_roi.py`, `scripts/video/batch_extract_pupil.py` | `camera_features.py` |
| 3 | `scripts/video/derive_front_cam_sync.py` | `video_sync.py` |
| 4 | `analysis_suite/10_camera/a-e_*.py` (5 scripts) | `loader.py`, `run_all.py` |
| 5 | `analysis_suite/10_camera/f-g_*.py` (2 scripts) | `run_all.py` |

**Total**: 11 new files, 5 modified files

---

## Dependency Graph

```
Constants + Config (prerequisite)
         |
    Phase 0: Batch Sync
    /        |        \
Phase 1    Phase 2   Phase 3
(Motion)   (Pupil)   (Front cam)
    \        |        /
     Phase 4: Analysis Suite (Figs 44-48)
              |
     Phase 5: Cross-Modal (Figs 49-50)
```

Phases 1, 2, 3 can run in parallel after Phase 0.

---

## Statistical Methods (Cross-Cutting)

These standards apply to all Phase 4-5 scripts. Derived from project CLAUDE.md and statistician review.

### Session as unit of replication
- For group comparisons (e.g., pupil by HMM state): compute per-session medians first, then test across sessions to avoid pseudo-replication
- Kruskal-Wallis on session-level medians (not raw trials); post-hoc: Dunn's test with Bonferroni (3 pairwise)
- For logistic models predicting trial outcomes from pupil: use mixed-effects logistic regression with random session intercepts, OR two-stage approach (per-session coefficients → Wilcoxon across sessions)

### Continuous > categorical for primary inference
- Pupil splits (median, quartile) are for **visualization only**
- Primary inference uses continuous regression: pupil as predictor of FR, CD projection, d', etc.
- Report Spearman rho + bootstrap CI (1000 resamples, seed=42)

### Trial-count matching
- When comparing neural activity by pupil group, subsample larger group to match trial counts
- For CD projections by pupil state: analyze within Hit trials only to remove outcome confound

### Effect sizes (report alongside every p-value)
| Analysis | Effect size |
|----------|-------------|
| KW across HMM states | epsilon-squared |
| Pairwise comparisons | rank-biserial r |
| Logistic prediction | AUC + bootstrap CI |
| Pupil-neural correlation | Spearman rho |
| ME onset vs spout timing | paired Cohen's d_z |
| Single-unit modulation | median delta_FR (Hz) |

### FDR correction scope
- Apply BH FDR (alpha=0.05) within mass screening across units (Fig 49B: single-unit pupil modulation)
- Do NOT apply FDR across separate scientific questions (Fig 47 vs 48 vs 49)

### Normalization for neural analyses (Phase 5)
- All population averages use shared-baseline z-score normalization per unit (same baseline for all pupil/arousal groups)
- Use `compute_zscore_normalized()` from `visdetect.analysis.utils`
- Guard division-by-zero: `if baseline_std < 1e-6: baseline_std = 1.0`

### Cross-validation for predictors
- Fig 47C (pupil predicts state): 5-fold stratified CV + permutation null (200 label shuffles)
- Fig 49: any decoder uses held-out test data

### Confound controls
- **Luminance**: Extract mean eye-ROI luminance alongside pupil (Phase 2), include as covariate in regressions where pupil is the predictor. Drifting gratings maintain roughly constant mean luminance, but measure it.
- **Grooming/whisking in ME**: Flag high-ME epochs not associated with licks as potential grooming. Report what fraction of ME peaks correspond to lick events.

---

## Audit Fixes Incorporated

The following issues from the auditor review are addressed in the plan:

| # | Severity | Finding | Resolution |
|---|----------|---------|------------|
| H1 | HIGH | Fig 45C: ME @ Change_ON must filter by EVENT_VALID_OUTCOMES | Added to Phase 4 notes: Change_ON alignment uses only hit/miss outcomes |
| H2 | HIGH | Fig 49A: population PSTH needs shared-baseline z-score | Added to Statistical Methods: all neural analyses use `compute_zscore_normalized()` |
| H3 | HIGH | Phase 0 verification too optimistic | Updated: expect some "no_camera", "review", "failed" sessions |
| M1 | MEDIUM | `align_feature_to_nidaq()` duplicates `camera_to_nidaq()` | Implementation note: `align_feature_to_nidaq()` wraps `camera_to_nidaq()` internally |
| M2 | MEDIUM | Fig 45B: Hit/FA lick alignment needs distinct event filters | Added: Hit uses `Hit` event (outcome=hit), FA uses `FA` event (outcome=fa) |
| M3 | MEDIUM | Phase 5: missing `get_good_cluster_ids()` | Added: all neural scripts use standard unit selection |
| M4 | MEDIUM | No color palette specs | Added: use `STAGE_COLORS`, `HMM_STATE_COLORS`, `OUTCOME_COLORS` throughout |
| M5 | MEDIUM | No mouth ROI validation | Added: `validate_roi.py` covers both mouth and pupil ROIs |
| M6 | MEDIUM | Fig 47C: missing CV + null distribution | Added to Statistical Methods |
| M7 | MEDIUM | Downstream scripts need sync-failure filtering | Added: all Phase 4-5 scripts skip sessions without valid sync |
| M8 | MEDIUM | Standard template not demonstrated | Covered by CLAUDE.md template; scripts follow existing conventions |

---

## Implementation Notes

### Event alignment safety (CRITICAL)
- **Change_ON alignment**: Filter to `outcome in {"hit", "miss"}` per `EVENT_VALID_OUTCOMES`
- **Hit lick alignment**: Use `Hit` event, filter to `outcome == "hit"`
- **FA lick alignment**: Use `FA` event, filter to `outcome == "fa"`
- Import `EVENT_VALID_OUTCOMES` from `visdetect.analysis.constants`

### Sync-failure handling
- All Phase 4-5 scripts call `load_video_sync(session_name)` at loop top
- Skip if result is `None` or `quality == "failed"`
- Report count of skipped sessions in cache/stats output

### Unit selection (Phase 5)
- Use `get_good_cluster_ids(sess)` from `visdetect.analysis.utils`
- Prefers `good_and_stable_ids` when available

### Color palettes
- Stage comparisons: `STAGE_COLORS` from `config.py`
- HMM states: `HMM_STATE_COLORS` from `config.py`
- Outcomes (Hit/FA/Miss): `OUTCOME_COLORS` from `config.py`

### `align_feature_to_nidaq()` implementation
- Wraps existing `camera_to_nidaq()` from `video_sync.py` — does not reimplement
- Adds convenience of operating on arrays (timestamps + feature together)

### ROI validation
- `scripts/video/validate_roi.py` (renamed from `validate_pupil_roi.py`) validates BOTH mouth and pupil ROIs
- Shows overlaid rectangles on sample frames for both ROIs
- Must run and confirm before any batch extraction

### ME onset threshold calibration
- Before batch use of Fig 48's `median + 3*MAD` threshold: manually annotate ~50 lick onsets from 3-5 sessions
- Compute precision/recall against manual annotations
- Adjust threshold if needed; document sensitivity analysis

### Pupil × HMM circularity note
- The HMM uses only behavioral covariates (bias, stimulus, prev_choice, prev_reward, prev_early_lick) — pupil is genuinely independent
- Frame the analysis as **validation**: pupil provides physiological grounding for computational states
- Additional test: does pupil predict hit rate **within** the Engaged state? (Shows pupil captures variance beyond what HMM captures)

---

## Verification (End-to-End)

1. **Phase 0**: Count sync JSONs; `batch_sync_summary.csv` has quality distribution; expect most "good", some "review"/"no_camera". Per-session mouth-zoom validation PNGs exist for all synced sessions with lick-responsive neurons.
2. **Phase 1**: Load one NPZ → plot ME around known lick events → visible peaks; validate ROI first
3. **Phase 2**: `validate_roi.py` shows correct pupil ellipses AND mouth ROI → batch extraction → detection rate >85%
4. **Phase 3**: Convert known NI-DAQ event → front cam frame → visually verify
5. **Phase 4**: All 5 scripts produce figures in `figures/10_camera/`; stats CSVs have effect sizes
6. **Phase 5**: Figs 49-50 use shared-baseline z-scored neural data; continuous regression as primary; splits for viz only
7. **Full suite**: `py run_all.py` completes with 10_camera scripts included
