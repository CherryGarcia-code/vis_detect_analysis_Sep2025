# Video Sync — Anchor-and-Barcode Approach (Phase 1)

**Date:** 2026-05-27
**Branch:** `feature/isi-hist-corr-and-chronology-fixes` (working branch; this work will land on a new branch)
**Replaces previous approach to:** automatic per-trial corneal Baseline_ON detection
**Status:** Phase 1 spec — Phase 2 deliberately deferred

---

## Background

The eye-cam Baseline_ON sync problem has resisted automation:

- **Good sessions** (09092025): orientation-selective features (`horizontal_band_energy`, `horizontal_edge_energy`) detect the grating, but per-trial detection fires ~300–500ms late due to the 300ms-sustain requirement in `run_timeseries` and max-filter smear in `detect_onsets_variance`.
- **Bad sessions** (03072025, 27062025, 29082025): no clean step in current ROIs. The corneal reflection is small, the context ROI is too large, the signal is diluted.
- **Underlying tension:** the user can clearly see the grating reflection appear in the eye on most trials; automated detectors keep missing or biasing it.

Previous attempts (screen-glow luminance, spatial variance, orientation-selective features in fixed ROIs) treated this as N independent detection problems — one per trial — and tuned thresholds/sustains to minimize false positives across many trials. This produced systematic bias on good sessions and outright failure on bad sessions.

## Reframing

The task structure provides a stronger constraint than per-frame features:

- The NI-DAQ knows the absolute time of every Baseline_ON in NI-DAQ-clock seconds.
- ITIs are stochastic, so the trial-onset *pattern* across a session is unique.
- One reliable anchor (one (video-frame-time, NI-DAQ-time) pair) implies an offset that *predicts* every other trial's video time.
- Verifying that the predictions land on actual grating onsets is a *barcode-matching* problem, not a detection problem.

This converts a hard N-detection problem into a hard 1-detection problem + a cheap N-verification problem.

### Recording-order implication

During each session, recording was started in this order:

1. SpikeGLX / NI-DAQ recording
2. Eye-cam recording (some seconds later)
3. Behavioral task (some further seconds later)

So in the camera's clock, trial 1 Baseline_ON occurs `(t_task_start − t_camera_start)` seconds after frame 0. That gap is variable across sessions and is exactly what the cached "coarse offset" tries to capture. On 03072025 the cached value was off by 13 seconds before manual correction. Phase 1 therefore cannot trust the cache; the anchor-click UI must absorb several seconds of error in either direction. This motivates the two-stage flow in Step 2 below.

## Goal of Phase 1

Build the smallest possible end-to-end loop that:

1. Lets the user **manually anchor** trial 1 of a session by clicking the frame where the grating appears in the eye.
2. Renders a **static PNG montage** that visualizes the predicted barcode at a handful of sampled trials, so the user can confirm by eye whether the implied offset aligns subsequent trials.
3. Persists the anchor and montage to disk.

Phase 1 is intentionally **visual-only** and **single-session-at-a-time**. No feature extraction, no automated detection, no regression, no clock-drift correction.

## Why Phase 1 first

- It grounds the problem in user perception — the most reliable detector available.
- It produces concrete evidence (PNG montages on 3 anchor sessions: 09092025, 14082025, 03072025) that informs Phase 2 design.
- It commits to no algorithm yet — if the montages reveal drift, Phase 2 fits a slope; if they reveal trial-label mismatches, Phase 2 handles that; if everything aligns, Phase 2 is a thin automated verifier.

## Non-goals (deferred to Phase 2 or beyond)

- Feature extraction (SV, band energy, edge energy) for verification.
- Automated per-trial onset detection.
- Theil-Sen or any regression-based clock fitting.
- Multi-anchor / clock-drift correction.
- Production `{session}_video_sync.json` artifacts.
- Batch processing across all ~42 sessions.
- Handling sessions where the user cannot identify trial 1's grating by eye.
- Front-cam alignment.

## Anchor sessions

Phase 1 will be exercised on three sessions chosen to span the difficulty range:

| Session | Why included |
|--------|---|
| 09092025 | Known-good signal, clean baseline — sanity check |
| 14082025 | Good but historically required relaxed threshold |
| 03072025 | Currently "failed" — barcode match would reveal whether user click + schedule alone recovers it |

Sessions 27062025 and 29082025 are *not* in the Phase 1 anchor set. Once Phase 1 succeeds on the three above, the same tool can be run on them.

---

## Design

### Single script: `scripts/video/click_anchor.py`

One script, one session at a time. Does both the click step and the montage render in sequence.

Invocation:

```
py scripts/video/click_anchor.py --session 09092025 [--reuse-existing-anchor]
```

Behavior:

1. **Load session and locate trial 1 neighborhood.**
   - Load the session via `visdetect.suite.loader.load_session(session_name)`.
   - Read `baseline_on = np.asarray(sess.ni_events.get("Baseline_ON", []), dtype=float)`; drop non-positive entries and trim to `len(sess.trials)` (consistent with existing convention in `corneal_spatial_diagnostic.py`).
   - Find video + metadata files via `visdetect.core.video_sync.find_camera_files(session_name)`.
   - Load camera timestamps via `load_camera_metadata(metadata_path)` — returns `ts_ms` in milliseconds relative to video start.
   - Read coarse offset from `data/cache/video_sync/coarse_offsets.json` (already populated for the 3 anchor sessions: 09092025=4.0s, 14082025=3.0s, 03072025=1.5s). The cached offset is "seconds elapsed in NI-DAQ clock before video recording started." If missing, fall back to a 15s default so stage 1's window starts near frame 0 — the two-stage UI absorbs the resulting uncertainty (see Error handling).
   - Compute predicted video time for trial 1 Baseline_ON: `t_video_ms = (baseline_on[0] - coarse_offset_s) * 1000.0`.
   - Convert to nearest frame index: `frame_idx_predicted = int(np.argmin(np.abs(ts_ms - t_video_ms)))`.

2. **Two-stage click UI.**

   The cached coarse offset cannot be trusted in general (e.g., 03072025 was off by 13s before manual correction). The click flow is therefore two stages — both stages reuse the same grid-rendering primitive; only the sampling interval and centre frame differ.

   Common rendering primitive: load N specified frame indices via `cv2.VideoCapture`, crop each to the eye region (constant `EYE_REGION_CROP_BG046 = (200, 420, 320, 540)` — y0, y1, x0, x1 — chosen to comfortably enclose every per-session corneal ROI documented in the autocal memory; y ∈ [247, 338], x ∈ [377, 468] for BG_046). Lay out in a 5×10 grid. Each cell annotated with its absolute frame index and time-offset relative to the stage's centre.

   **Stage 1 — coarse window (1 click).**
   - 50 frames sampled at 1-second intervals (every 50 frames at 50fps), spanning 50 seconds total.
   - Window centered on `frame_idx_predicted` but shifted so it covers `[predicted − 15s, predicted + 35s]` — biased forward because the task can't start before the video, so most of the uncertainty is on the late side. Clamped to `[0, video_end]`.
   - Title bar: `"Stage 1 — Coarse scan. Click the cell where the grating first appears in the eye. ESC to cancel."`
   - You click the cell where you first see the grating in the reflection.

   **Stage 2 — fine strip (1 click).**
   - 50 frames at full frame rate (one frame per cell), centered on the stage-1-clicked frame. Spans ±25 frames ≈ ±500ms — chosen because stage 1's 1s sampling leaves ±500ms uncertainty around the clicked cell.
   - Same 5×10 grid layout.
   - Title bar: `"Stage 2 — Fine pick. Click the exact frame where the grating appears. ESC to cancel."`
   - You click the exact frame.

3. **Capture each click.**
   - Register a matplotlib `button_press_event` handler that maps click (x, y) → grid cell → selected frame index. Clicks outside the grid area are ignored.
   - Register an ESC key handler that exits non-zero without writing.
   - On a valid click: show a visual confirmation (red box around clicked cell), wait ~500ms, close the figure, advance to the next stage (or save anchor after stage 2).

4. **Save anchor JSON.**
   - Path: `data/cache/video_sync/{session}_anchor.json`
   - Schema:
     ```json
     {
       "session": "09092025",
       "anchor_trial_index": 0,
       "nidaq_baseline_on_s": 12.3456,
       "video_frame_idx": 1047,
       "video_time_s": 20.94,
       "implied_offset_s": 8.5944,
       "frame_rate_fps": 50.0,
       "n_trials": 350,
       "clicked_at": "2026-05-27T14:32:10"
     }
     ```
   - `implied_offset_s = video_time_s - nidaq_baseline_on_s` — this is the candidate clock offset; predicts video time of trial i as `baseline_on[i] + implied_offset_s`.
   - If `--reuse-existing-anchor` is passed and the JSON exists, skip steps 2–4 and go straight to montage rendering.

5. **Render barcode montage automatically.**
   - Sampled trials (default 5 rows): trial 0, ⌊N/4⌋, ⌊N/2⌋, ⌊3N/4⌋, N−1, where N = number of task trials.
   - Per row: 7 columns showing frames at predicted-onset−3, −2, −1, 0, +1, +2, +3 (50fps → ±60ms span).
   - Each cell: cropped eye region, ~200×200 px, grayscale.
   - Decoration:
     - Centre column (predicted onset) drawn with a red border.
     - Row labels on the left: trial index + NI-DAQ Baseline_ON time.
     - Column labels along the top: frame offset in ms relative to predicted onset.
   - Title: session, anchor info, implied offset, total trials.
   - Save to `figs/video_sync/{session}_barcode_montage.png`.

6. **Print a one-line summary to stdout** indicating both files written, with their paths.

### File artifacts

```
data/cache/video_sync/{session}_anchor.json      (machine-readable anchor)
figs/video_sync/{session}_barcode_montage.png    (visual barcode evidence)
```

These two files are the complete Phase 1 output for one session.

### Data flow

```
sess.ni_events["Baseline_ON"]   coarse_offsets.json    camera metadata + video
              \                       |                       /
               \                      |                      /
                +───────► predicted trial-1 frame ◄──────────+
                                      │
                                      ▼
              Stage 1: 50-frame coarse grid (1s sampling, 50s span)
                                      │
                            [user clicks rough cell]
                                      │
                                      ▼
              Stage 2: 50-frame fine grid (1 frame/cell, ±500ms span)
                                      │
                            [user clicks exact frame]
                                      │
                                      ▼
                  {session}_anchor.json  (implied_offset_s)
                                      │
                                      ▼
                     barcode montage (5 trials × 7 frames)
                                      │
                                      ▼
                      {session}_barcode_montage.png
                                      │
                                      ▼
                              user eyeballs PNG
```

### Module boundaries

A tiny amount of logic moves into the library — the rest stays in the script.

- New, in `src/visdetect/core/video_sync.py`:
  - `load_anchor(session_name) -> dict | None` — read `{session}_anchor.json`, return None if absent.
  - `save_anchor(session_name, anchor_dict) -> None` — write `{session}_anchor.json`.
  - These are 10–20 line helpers; they belong in the library so Phase 2 can read anchors without duplicating path logic.

- Stays in the script:
  - matplotlib UI (frame strip, click handler).
  - Montage rendering (uses matplotlib gridspec).
  - Frame I/O via `cv2.VideoCapture`.

### Eye-region crop

A single subject-level crop bounding box `EYE_REGION_CROP_BG046 = (200, 420, 320, 540)` is hard-coded in the script for now. It does *not* need to be tight — generous (~220×220 px) is fine; we want to *see* the eye, not isolate the cornea. The box was chosen to enclose every documented per-session corneal ROI for BG_046 (see autocal memory).

If/when other subjects are added, this becomes a per-subject dict — but that's a Phase 2 concern (along with auto-locating the eye region).

### What the montage tells us

Three possible outcomes per session, observable from the PNG:

| Pattern | Meaning | Phase 2 implication |
|---|---|---|
| All 5 centre-column cells show grating | Anchor good, barcode holds across session | Phase 2 = thin auto-verifier in tight window |
| Centre column misses on later rows; grating visible 1–2 columns earlier/later | Camera-NI-DAQ clock drift | Phase 2 fits slope (2-anchor or Theil-Sen) |
| Centre column hits early trials, no grating at all on later rows | Anchor wrong, trials missing, or recording problem | Phase 2 needs failure handling |
| No centre-column cell shows grating | Click was off / coarse offset wrong / signal absent | Investigate session-specifically |

### Validation (how we know Phase 1 works)

For each of the three anchor sessions:

1. Run `py scripts/video/click_anchor.py --session <s>`.
2. Click trial 1.
3. Inspect `figs/video_sync/{s}_barcode_montage.png`.
4. Record observation in a results table (informal note in conversation, not a separate spec):
   - Session, click frame, implied offset, qualitative barcode pattern from the table above.

Phase 1 is considered complete when all 3 sessions have an anchor JSON + a montage PNG, and we have a documented qualitative observation per session.

### Error handling / edge cases

- Missing coarse offset → not fatal. Treat predicted onset as `15s` (default) so stage 1's window becomes `[0s, 50s]` — biased to inspect the start of the video. Log a warning.
- Missing video file → abort with directive message (suggest running existing `find_camera_files` diagnostic).
- User presses ESC at either stage → exit non-zero, no file writes.
- Stage 1 click near grid boundary → stage 2 window is clamped to `[0, video_end]`; if the clamp eats more than half of stage 2's intended span, log a warning so the user knows the fine window was asymmetric.
- Existing `{session}_anchor.json` and no `--reuse-existing-anchor` flag → prompt user once at startup: overwrite? (default no, exit).
- Predicted onset within the first 25 frames or last 25 frames of the video → render the affected grid with fewer flanking frames; do not pad.
- User can't see grating in stage 1 → they ESC; we treat that as "Phase 1 cannot anchor this session" — that case is explicitly out of scope. Phase 2 will need to design a fallback.

Everything else (model failures, threshold failures, ROI failures) is out of scope because Phase 1 has no model, no threshold, no ROI.

### Testing

No automated tests in Phase 1. The deliverable is the PNG evidence on the 3 anchor sessions; visual inspection is the validation.

If Phase 2 turns Phase 1 helpers (`load_anchor`, `save_anchor`) into load-bearing API, those gain unit tests at that point.

---

## Phase 2 (not designed — listed as hooks only)

These items are explicitly out of scope for this spec but are flagged so reviewers can see the intended trajectory:

- Choose one feature among `horizontal_band_energy`, `horizontal_edge_energy`, fused — based on what the Phase 1 montages reveal.
- Implement a tight-window auto-verifier: for each predicted onset, scan ±N frames, find the feature step, refine the per-trial Baseline_ON time.
- Decide between assumed slope=1 vs fitting slope from anchor pair (start + end of session).
- Produce `{session}_video_sync.json` in the existing schema, replacing the current corneal pipeline outputs.
- Extend `find_camera_files` and the anchor convention to front-cam.
- Scale to all ~42 sessions and other subjects.

Each Phase 2 component will be designed in a follow-up spec once Phase 1 produces evidence.

---

## Files touched by Phase 1

| File | Change |
|---|---|
| `scripts/video/click_anchor.py` | NEW |
| `src/visdetect/core/video_sync.py` | + `load_anchor`, `save_anchor` helpers |

No other files change.
