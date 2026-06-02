# Video Sync — Phase 2: Multi-Anchor Slope Fit + Per-Trial Manual Tagging

**Date:** 2026-05-29
**Branch:** continues `feature/video-sync-anchor-barcode` (Phase 1 + 1.5)
**Replaces previous approach to:** automatic per-trial onset detection
**Status:** Phase 2 spec — design approved 2026-05-29

---

## Background

Phase 1 (the `--scrub` and 2-stage click flows on `scripts/video/click_anchor.py`) produces a **single anchor per session** — trial 0's video frame index plus the implied clock offset assuming slope=1. Phase 1 manual runs on BG_046 09092025, 14082025, and 03072025 revealed:

- **Clock drift is real and unidirectional.** 09092025 ≈ +11 ppm; 14082025 ≈ +24 ppm. Camera runs faster than NI-DAQ in both. The single-anchor slope=1 model is wrong by tens of ms across a session.
- **03072025's corneal signal contrast is too low.** Even trial 0 was barely visible; later trials show nothing across the ±60ms barcode window.
- **The 2-stage click UI can produce systematically wrong anchors.** 09092025's original anchor was off by ~9 seconds with the wrong sign on `implied_offset_s`. The `--scrub` mode reliably catches this; users should verify every anchor via scrub before trusting it.

Phase 2's job is to (a) correct the linear drift by fitting a slope from two manually-clicked anchors, and (b) provide a manual per-trial tagging tool as a fallback for hard sessions and as ground truth when the slope-fit alone isn't trusted.

Auto-detection (originally subsystem "B" in the Phase 2 sketch) is deliberately **out of scope**. Per the user's stated skepticism about feature-based detection working reliably on this subject's eye videos, Phase 2 ships zero auto-detection code. Phase 3 may build an auto-verifier *on top of* Phase 2's per-trial manual tags as ground truth — that's a far stronger evidence base than what could be gathered speculatively now.

## Goals

Phase 2 produces a canonical per-session sync artifact (`{session}_video_sync.json`) that downstream code uses to convert NI-DAQ trial times to video frame indices. Two production paths:

1. **Slope-fit only.** Two anchors (trial 0 + last task trial) → linear clock model → per-trial timestamps via `slope * nidaq_time + offset`. Fast: two clicks per session.
2. **Slope-fit + per-trial manual overrides.** Slope-fit as above, plus the user walks every (or some) trials with the per-trial tag tool. Overridden trials use the manual frame; unannotated trials fall back to slope-fit. Slower: tens of minutes per session, but deterministic.

Both paths produce the same output JSON schema; downstream consumers don't need to know which was used.

## Non-goals (deferred to Phase 3 or beyond)

- Feature-based per-trial auto-detection (`horizontal_band_energy`, `horizontal_edge_energy`, spatial variance verifiers).
- Front-cam alignment (eye-cam only for now; the existing `find_camera_files` returns both, but Phase 2 ignores `front_cam`).
- Batch processing across all ~42 sessions (each session is operated one at a time; batch is a Phase 3 wrapper).
- Multi-subject support (BG_046 only).
- Sub-frame interpolation (per-trial precision is to the nearest video frame ≈ 20ms at 50fps).

## Anchor sessions

Phase 2 will be exercised on the same three Phase 1 anchor sessions, in this order:

| Session | Phase 1 anchor (trial 0) | Expected Phase 2 outcome |
|---|---|---|
| 09092025 | frame 1167, implied_offset = −4.610s | Slope fit corrects ~11 ppm drift; final montage centre column hits all 5 sampled trials. |
| 14082025 | frame 1434, implied_offset = −3.393s | Slope fit corrects ~24 ppm drift; final montage holds. |
| 03072025 | frame 935, implied_offset = −9.136s | Slope fit alone unlikely to satisfy because the second anchor may also be unreliable. Almost certainly requires per-trial manual tagging. This is the test case for subsystem C. |

---

## Design

### Output schema: `{session}_video_sync.json`

Phase 2 produces a file at `data/cache/video_sync/{session}_video_sync.json` using the **existing** `save_video_sync` / `load_video_sync` helpers in `src/visdetect/core/video_sync.py`, with one **additive** extension to the `SyncResult` dataclass.

Existing `SyncResult` fields used by Phase 2:

| Field | Phase 2 meaning |
|---|---|
| `slope` | Dimensionless clock ratio `video_time_s = slope * nidaq_baseline_on_s + offset`; computed from the 2 anchors |
| `offset` | Intercept of the linear fit (seconds) |
| `n_anchors` | Number of clicked anchors (2 for the default path; could be ≥3 if multi-anchor extended) |
| `n_baseline_on` | Number of task trials in the session |
| `rmse_ms` | 0.0 for the exact 2-point fit; nonzero if ≥3 anchors |
| `slope_ppm` | `(slope − 1) * 1e6`; reported for human inspection |
| `detection_method` | New literal `"manual_slope_fit"` |
| `roi`, `n_frames`, `n_dropped`, `durbin_watson`, `cv_rmse_ms`, `max_residual_ms`, `inlier_mask` | Phase 2 does not populate these; leave at field defaults (None / 0 / etc.) |

New optional `SyncResult` field:

```python
per_trial_overrides: Optional[Dict[int, int]] = None
# trial_idx → video_frame_idx. Set only by tag_trials; None when no manual
# overrides exist. Trial indices are integers (cast from JSON string keys
# on load); video_frame_idx is an int.
```

The on-disk JSON gets the field via the existing `to_dict` serialization (which `save_video_sync` calls). `load_video_sync` returns the dict; downstream consumers read `eye_cam["per_trial_overrides"]` directly.

### Anchor JSON schema migration: `{session}_anchor.json` v1 → v2

Phase 1 anchor JSON has a single anchor at the top level:

```json
{
  "session": "09092025",
  "anchor_trial_index": 0,
  "nidaq_baseline_on_s": 27.829,
  "video_frame_idx": 1167,
  "video_time_s": 23.219,
  "implied_offset_s": -4.610,
  "frame_rate_fps": 50.04,
  "n_trials": 551,
  "clicked_at": "2026-05-29T11:51:46"
}
```

Phase 2 needs to store multiple anchors. New v2 schema:

```json
{
  "session": "09092025",
  "schema_version": 2,
  "frame_rate_fps": 50.04,
  "n_trials": 551,
  "anchors": [
    {
      "trial_index": 0,
      "nidaq_baseline_on_s": 27.829,
      "video_frame_idx": 1167,
      "video_time_s": 23.219,
      "clicked_at": "2026-05-29T11:51:46"
    },
    {
      "trial_index": 550,
      "nidaq_baseline_on_s": 7255.49,
      "video_frame_idx": 363270,
      "video_time_s": 7259.79,
      "clicked_at": "2026-06-01T14:00:00"
    }
  ]
}
```

`load_anchor` migrates v1 → v2 in memory on read; the on-disk file gets rewritten in v2 form on the next `save_anchor` call. The `implied_offset_s` field is dropped from the schema because it's redundant (computable from any single anchor) and ambiguous when there are multiple anchors.

A new helper `compute_implied_offset(anchor: dict) -> float` returns `video_time_s - nidaq_baseline_on_s` for a single-anchor dict — used in Phase 1.5 `--scrub` HUD and the new Phase 2 commands when they need to display an offset.

### Subsystem A — Multi-anchor + slope fit

**Command 1a: `click_anchor --session X --anchor-last`**

- New CLI flag on the existing script. When set:
  - Loads the anchor JSON (must exist; if not, errors with a directive message: "run --session X (no --anchor-last) first to anchor trial 0").
  - Requires the existing anchor list to contain `trial_index == 0`. Errors otherwise.
  - Computes the predicted last-trial frame using the first anchor's implied offset (slope=1 baseline): `predicted_last = round((baseline_on[-1] + (anchor0.video_time_s - anchor0.nidaq_baseline_on_s)) * fps)`.
  - Opens the existing `--scrub` UI starting at that predicted frame.
  - HUD now shows: "Anchoring trial N−1 (last task trial)" + the existing scrubber HUD.
  - Space/Enter saves a new anchor entry with `trial_index = n_trials - 1` and the current frame; appends to the anchor list. Quits on save.
  - If an anchor at `trial_index == n_trials - 1` already exists, the new save **overwrites** it (with a single y/N prompt on stdin).
- No 2-stage click variant. The 2-stage UI was prone to mis-clicks; the scrubber is more reliable.

**Command 2: `fit_sync --session X`**

- New script `scripts/video/fit_sync.py`.
- Loads the anchor JSON; requires ≥2 anchors.
- Fits a linear clock model from the anchor pairs. The model is:

  ```
  video_time_s = slope * nidaq_baseline_on_s + offset_s
  ```

  where `slope` is a dimensionless clock ratio (≈ 1 ± ppm-scale deviation) and `offset_s` has units of seconds. Per-trial video frame is then `round((slope * nidaq_time + offset_s) * fps)`.

  - Anchor i contributes `(x_i, y_i) = (nidaq_baseline_on_s, video_frame_idx / fps)`.
  - For exactly 2 anchors: closed-form linear fit. `rmse_ms = 0`.
  - For ≥3 anchors: least-squares fit; `rmse_ms` from residuals × 1000.
  - Report `slope_ppm = (slope − 1) * 1e6` for human inspection.
- Builds a `SyncResult` with `slope`, `offset`, `n_anchors`, `n_baseline_on = len(sess.trials)`, `rmse_ms`, `slope_ppm`, `detection_method = "manual_slope_fit"`. All other dataclass fields use their defaults (None / 0).
- Calls `save_video_sync(session_name, eye_cam=sync_result)` — produces `{session}_video_sync.json`.
- After saving, renders an updated barcode-montage PNG to `figs/video_sync/{session}_barcode_montage_slopefit.png`. The montage uses slope-fitted predictions (not the slope=1 prediction Phase 1's montage used). User inspects to verify the fit looks right across the session.
- Logs the slope, slope_ppm, anchor pair frames, and the montage path to stdout.

### Subsystem C — Per-trial manual tag

**Command 3: `tag_trials --session X`**

- New script `scripts/video/tag_trials.py`.
- Loads `{session}_video_sync.json` (must exist; if not, errors with directive message: "run fit_sync first").
- Loads the session and any existing `per_trial_overrides` from the JSON.
- Opens a TkAgg window using the same single-frame layout as Phase 1.5's `--scrub` (large frame view + HUD below).
- **State per trial:**
  - `trial_idx`: the trial currently being tagged (starts at 0)
  - `current_frame`: the frame currently displayed (initialized from per-trial override if present, else slope-fit prediction, else `0`)
- **HUD** (in addition to the standard scrubber HUD):
  - "Tagging trial X of N"
  - "Slope-fit predicted: frame F"
  - "Override status: ON (frame G) | OFF (using slope-fit)"
  - "Elapsed: X min   Estimated remaining: Y min" (based on average per-trial time so far)
- **Keys:**
  - `←` `→` `Shift+←` `Shift+→` `Ctrl+←` `Ctrl+→` — scrub within the current trial (frames only)
  - `Enter` — set `per_trial_overrides[trial_idx] = current_frame`, save the JSON to disk, advance to next trial
  - `S` — skip: advance to next trial **without changing the override state** for this trial. Non-destructive: existing override is preserved; absence of override is preserved.
  - `D` — delete: remove `per_trial_overrides[trial_idx]` if present (revert this trial to slope-fit), save the JSON, advance. Use this to undo a previous bad override.
  - `B` — back to previous trial (re-open it with its current override or slope-fit prediction)
  - `Q` / `Esc` — save the JSON one more time and quit
- **Per-trial autosave.** Every `Enter`, `D`, and `S` writes the JSON to disk (S writes even though it changes no state — a cheap heartbeat that the user reached this trial). Crashes / power loss do not lose work.
- **End of session.** When `trial_idx` reaches `n_trials`, the tool logs "all trials reviewed; quitting" and saves+exits.
- **Resume behavior.** If the JSON already has `per_trial_overrides`, the tool starts at the **lowest trial_idx that has no override** (resume-where-you-left-off). To re-review a previously-overridden trial, the user navigates to it explicitly via `B` from later trials. (A future Phase 3 enhancement could add an `--start-trial N` flag.)

**Subsystem C is self-contained.** It does not modify `click_anchor.py` or `fit_sync.py`; it only reads/writes `{session}_video_sync.json`.

### Downstream consumer contract

Code that reads `{session}_video_sync.json` and computes per-trial onsets follows this rule (the canonical Phase 2 consumer pattern):

```python
sync = load_video_sync(session_name)
slope = sync["eye_cam"]["slope"]        # dimensionless clock ratio (~1)
offset_s = sync["eye_cam"]["offset"]    # seconds
fps = sync["eye_cam"]["frame_rate_fps"] # from the sync result
overrides = sync["eye_cam"].get("per_trial_overrides") or {}

def video_frame_for_trial(trial_idx: int, nidaq_baseline_on_s: float) -> int:
    if str(trial_idx) in overrides:
        return int(overrides[str(trial_idx)])
    video_time_s = slope * nidaq_baseline_on_s + offset_s
    return int(round(video_time_s * fps))
```

(Note: JSON object keys are strings, so `per_trial_overrides`'s keys are strings on disk; the rule above accounts for this.)

This contract is documented at the top of `src/visdetect/core/video_sync.py` near the `save_video_sync` / `load_video_sync` helpers as a docstring or `# CONTRACT:` comment.

---

## Workflow per session

The intended user flow:

```
1. (Phase 1.5, already done) py scripts/video/click_anchor.py --session X
   → produces anchor JSON with trial 0 anchor

2. py scripts/video/click_anchor.py --session X --anchor-last
   → scrub UI starting at predicted last-trial frame; you click; saves trial N−1 anchor

3. py scripts/video/fit_sync.py --session X
   → reads anchor JSON, fits 2-anchor slope, writes {session}_video_sync.json,
     renders updated barcode montage. You inspect the montage.

4a. If montage looks good across all 5 sampled trials → DONE. The slope-fitted
    timestamps are the per-trial source of truth.

4b. If montage looks bad (some trials still missing the grating) OR you want
    higher-confidence ground truth →
    py scripts/video/tag_trials.py --session X
    → walks every trial with verify-and-advance. Persists per-trial overrides.
    Re-run is safe; resumes from where you left off.
```

Steps 2, 3, 4 are each atomic — each can be re-run independently. Step 4 can be partial (tag 50 trials, quit, come back later, resume at trial 51).

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `src/visdetect/core/video_sync.py` | Modify | Add `per_trial_overrides` field to `SyncResult`; add `compute_implied_offset`; add `_migrate_anchor_v1_to_v2` (used internally by `load_anchor`); update `load_anchor` to handle both v1 and v2 schemas; update `save_anchor` to write v2 |
| `src/visdetect/core/video_sync.py` | Modify | Add `fit_2anchor_clock(anchors, fps, n_baseline_on) -> SyncResult` library function (the math; `fit_sync.py` is a thin CLI wrapper) |
| `scripts/video/click_anchor.py` | Modify | Add `--anchor-last` flag and `_run_anchor_last` helper; reuses existing `_run_scrub` UI |
| `scripts/video/fit_sync.py` | Create | NEW CLI: load anchors → fit → save sync JSON → render slope-fit montage |
| `scripts/video/tag_trials.py` | Create | NEW CLI: per-trial verify-and-advance UI, persists overrides to sync JSON |
| `tests/test_video_sync_anchor.py` | Modify | Add tests for `fit_2anchor_clock`, schema migration, override read/write |
| `tests/test_video_sync_tag_trials.py` | Create | Tests for the per-trial state machine (pure-logic parts: next/prev/skip/save) |

No changes to existing Phase 1 tests; all 18 continue to pass.

### Testing scope

Unit-test the pure-logic / pure-math additions (per Phase 1's convention):
- `fit_2anchor_clock` (exact 2-point fit, ≥3-anchor lsq, slope_ppm, edge cases).
- Schema migration `_migrate_anchor_v1_to_v2` (round trip, missing fields, idempotent).
- `compute_implied_offset` (algebra).
- `tag_trials` state transitions (e.g., "advancing past the last trial returns done", "B at trial 0 stays at trial 0", "S preserves existing override", "D removes existing override and reverts to slope-fit", "Enter sets override to current_frame").

Interactive matplotlib UI is not unit-tested. Manual smoke test per session as in Phase 1.

---

## Error handling / edge cases

- `--anchor-last` without an existing trial-0 anchor → exit 2 with directive message.
- `fit_sync` without ≥2 anchors → exit 2 with directive message.
- `fit_sync` with anchors that produce a slope ≤ 0 (impossible-clock) → exit 2 with directive: anchors are likely swapped or one is wrong; re-verify via `--scrub`.
- `tag_trials` without an existing `{session}_video_sync.json` → exit 2 directing user to run `fit_sync`.
- Anchor JSON has unknown `schema_version` → exit 2, do not auto-migrate.
- User interrupts `tag_trials` mid-trial (`Q`/`Esc`/window close) → JSON saved through the last completed trial.
- Per-trial autosave write fails (disk full, permissions) → log error, continue UI; on quit, also log "autosave failures occurred, JSON may be out of date".

---

## Phase 3 hooks (not designed)

- Auto-verifier built on Phase 2 per-trial overrides as ground truth (feature comparison, per-trial detection-rate report).
- Front-cam alignment via inter-camera offset.
- Batch wrapper: `batch_phase2_sync --sessions all` that runs the 3-command pipeline non-interactively where possible.
- Other subjects.

Each Phase 3 component will be designed in a follow-up spec.

---

## Files explicitly NOT touched

- `src/visdetect/core/video_sync.py`'s existing detection helpers (`detect_onsets_variance`, `detect_onsets_derivative`, `fit_clock_model`, `extract_luminance`, etc.) — Phase 2 lives alongside them, not on top of them.
- The existing `corneal_spatial_diagnostic.py`, `batch_sync_sessions.py`, `poc_multianchor_sync.py`, `select_roi.py` scripts.
- All other repo modules.

---

## Open questions (none — all resolved during brainstorming)

(Section retained intentionally as a self-review checkpoint. If anything in the design is genuinely uncertain on review, it should be lifted here and resolved before writing the plan.)
