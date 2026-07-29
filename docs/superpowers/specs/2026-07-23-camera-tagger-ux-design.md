# Camera Pipeline — Unified Single-Pass Tagger + Label Capture (Plan 2 Design)

**Date:** 2026-07-23
**Status:** Design (brainstormed + approved; awaiting spec review → writing-plans)
**Part of:** sub-project **A** ([2026-07-21-camera-baseline-sync-multisubject-design.md](2026-07-21-camera-baseline-sync-multisubject-design.md)). Plan 1 (non-GUI backbone) is merged to `main` (`057c622`). This is **Plan 2** — the interactive tagging layer that drives the backbone.

---

## 1. Goal

Evolve `scripts/video/click_anchor.py`'s scrubber into **one unified per-session tagging window** that, in a single pass, lets the user: navigate/play the eye-cam video (full-frame or eye-zoom), place **baseline AND change** anchors (feeding the v3 schema + `fit_multianchor_clock`), and capture the per-session mouth+eye **ROIs + per-frame pupil/blink labels** — amortizing the one expensive manual pass across sub-projects A, B, and C. Plus three seams Plan 1 flagged: emit the v3 schema, un-block `tag_trials` for `manual_multianchor`, and make the neural validation figure subject-agnostic + orientation-aware.

**Non-goal:** motion-energy / pupil-trace *extraction* (sub-projects B/C) — Plan 2 only *captures the labels* those will consume.

---

## 2. The single-pass tagging flow (one session, one window)

1. **Open** `py scripts/video/tag_session.py --subject BG_031 --session 09042025` (new unified entry point; see §8).
2. **Migrate + seed** (§6): archive any prior anchors/sync to `_archive/`, pre-load legacy anchors as **editable seeds** (marked `source="legacy"`).
3. **ROIs (free, any time):** press `m` / `e` to rubber-band the mouth / eye ROI on the full-frame view; the eye box immediately overlays the `detect_pupil_in_frame` ellipse so a bad box is obvious. Not a forced step — draw or redraw whenever (e.g. the head shifts mid-session).
4. **Anchor:** navigate with arrows / `j`/`k`; `space` plays/pauses; `enter` saves an anchor at the current frame. `c` toggles the jump target between **baseline onsets** (all trials) and the **change queue** (auto-cycled size-4 then size-2, hit/miss only, §4). Live `cv_rmse` in the HUD shows the fit tightening as anchors accumulate.
5. **Label (optional, free):** on any frame you're viewing, `u` confirms the pupil ellipse, `x` flags a blink/occlusion → written to the label sidecar (§5).
6. **Quit** (`q`/`esc`): flush the label sidecar. Then run the already-built `fit_sync --subject … --session …` to fit + gate.

### Window layout (single TkAgg figure: frame panel + HUD text panel)

```
┌──────────────────────────────────────────────────────┐
│              FRAME VIEW  (full-frame  or  eye-zoom)     │
│      ┌ eye ROI + live pupil ellipse ┐                   │
│      └───────────────────────────────┘   ▾ predicted    │
│      ┌ mouth ROI ┐                          onset marker │
│      └───────────┘                                       │
├──────────────────────────────────────────────────────┤
│ BG_031  09042025   MODE: CHANGE (size4, hit)            │
│ trial 23/240   frame 402113 (8042.31s)   Δ+2 vs pred    │
│ anchors: 5 (3 base / 2 chg)     cv_rmse: 18.4 ms  good   │
│ ROI mouth ✓  eye ✓     labels: 4 pupil, 1 blink         │
│ [space]play [f]full [j/k]jump [c]base⇄chg [enter]save   │
└──────────────────────────────────────────────────────┘
```

### Keybindings (extend the existing arrows/`j`/`k` so muscle memory carries)

| Key | Action | Key | Action |
|---|---|---|---|
| ←/→ · ⇧ · ⌃ | step ±1 / ±10 / ±100 frames | `enter` | **save** anchor (baseline or change per mode) |
| `space` | **play/pause** forward (`[`/`]` slower/faster) | `d` | delete this trial's anchor (current mode's type) |
| `f` | toggle full-frame ⇄ eye-zoom | `m` / `e` | draw/redraw mouth / eye ROI (drag) |
| `j` / `k` | next / prev target onset | `u` / `x` | confirm pupil / flag blink on this frame |
| `c` | toggle baseline ⇄ change target mode | `r` | preview barcode montage |
| `home`/`end` | first / last trial | `q`/`esc` | flush labels + quit |

### Full-frame view

The display crop is the hardcoded `EYE_REGION_CROP_BG046 = (200,420,320,540)` in `click_anchor.py` (a 220×220 patch of the ~976×1024 frame, mouth cropped out). Plan 2 replaces the fixed slice with a **view mode**: `f` toggles between the full frame (for ROI drawing + judging the face/mouth) and an eye-zoom (for precise onset frame selection). The eye-zoom crop is **per-session** — derived from the eye ROI once drawn, else the legacy constant as fallback. `tag_trials.py` (which re-imports the constant) is updated in lockstep.

> **DEFERRED TO PLAN 2b (zoom/ROI) — decided in the A1 pilot (2026-07-27).** The `f` full/zoom toggle was **removed from `tag_session.py` in Plan 2a**. `tag.eye_roi` is always `None` in 2a, so the only crop `eye_zoom_crop` can return is the BG_046-specific *absolute-pixel* fallback `(200,420,320,540)`; on BG_031 (camera closer to the face) that patch lands on the snout, not the eye. Full-frame tagging worked well in the pilot, so 2a is full-frame only and all zoom/ROI is deferred to 2b. `eye_zoom_crop` stays as tested library API for 2b.
>
> **HARD LESSON for any future 2b zoom/ROI consumer — clamp the crop before you index a frame.** `eye_zoom_crop` returns an **unclamped** `(y0,y1,x0,x1)`: padding an ROI near a frame edge (or applying the BG_046 fallback to a smaller frame) can produce **negative or out-of-frame** coordinates. `numpy` slicing with a negative index does **not** error — it silently wraps from the far edge and yields the **WRONG crop** with no warning. Before indexing any frame with an `eye_zoom_crop` result, **clamp to real frame bounds** so `0 <= y0 < y1 <= H` and `0 <= x0 < x1 <= W` always hold (this is exactly what the now-removed `_clamp_crop` in `tag_session.py` did — resurrect that guard when zoom returns in 2b).

### Playback (`space`)

A matplotlib canvas timer advances the frame at ~fps while playing; `space` toggles play/pause; `[`/`]` set the speed (e.g. 0.5× / 1× / 2×). While paused, arrows step. Playback is bounded to the current view so "reverse a few frames and play from here" is one motion: step back, `space`, watch the grating appear, `space` to pause, fine-step, `enter`.

---

## 3. Anchoring — baseline (existing) + change (new)

- **Baseline anchors** (existing behavior, retained): `j`/`k` jump to each trial's predicted baseline onset (`jump_to_predicted_frame` / `compute_predicted_frame_idx`); `enter` saves. Now emitted in the **v3 schema** with `event_type="baseline_on"` (keeping `nidaq_baseline_on_s` for legacy readers).
- **Change anchors** (new): `c` enters change mode → the tool builds the **big-change queue** = go trials with `change_size ∈ BIG_CHANGE_SIZES = {4.0, 2.0}` and `outcome ∈ {hit, miss}` (`EVENT_VALID_OUTCOMES["Change_ON"]`), ordered **size-4 first, then size-2**. `j`/`k` advance/retreat through the queue, auto-jumping to each predicted Change_ON frame; `enter` saves via `_build_change_anchor_entry(change_on_s, ts_ms, trial_index, frame_idx, change_size, outcome)` (Plan 1, already in the library). `c` again returns to baseline mode.
- **Change-jump seeding:** the predicted Change_ON frame uses the current best offset — coarse (`coarse_offsets.json`) → refined by the first baseline anchor's implied offset → refined again by a **provisional `fit_multianchor_clock`** once ≥3 anchors exist. Accuracy improves as tagging proceeds.
- **Live QC:** once ≥3 anchors exist, the HUD runs `fit_multianchor_clock` on the current anchor set and shows `cv_rmse` + tier, so the user tags until it reads `good`.

---

## 4. ROI + label capture (the amortized layer)

- **ROIs:** `m` / `e` open a matplotlib `RectangleSelector` on the full-frame view; the drag defines the mouth / eye box (full-frame coords). On setting the eye box, `detect_pupil_in_frame` runs on the current frame and the ellipse is overlaid — a bad box shows a bad/absent ellipse, so the user re-drags. Free at any time; redrawing replaces.
- **Per-frame labels:** `u` records the proposed pupil ellipse on the current frame as `confirmed=true`; `x` records `blink=true` (occlusion). Only on frames the user is already viewing — zero extra navigation.
- **Sidecar** (`data/cache/video_labels/<subject>/<session>.json`, `schema_version`): `{ "me_roi": [y0,y1,x0,x1], "eye_roi": [...], "frames": [ {frame_idx, event_type?, pupil_ellipse:{cx,cy,major,minor,angle}, confirmed, blink} ] }`. Optional small PNG crops of confirmed-ellipse frames under `…/<session>/frames/` for a self-contained ML set. **Decoupled from the anchor JSON** so the sync stays clean and the label schema can evolve as B/C firm up. Written atomically (mirroring `tag_trials._persist_overrides`) on each label keystroke + a final flush on quit.
- Reuses: `detect_pupil_in_frame` (Plan 1 seed), the `select_roi.py` full-frame canvas + JSON-persistence patterns.

---

## 5. Migration / seed-from-archive (resolves Plan-1 blocker #2)

On opening a session, `tag_session`:
1. Calls `archive_sync_artifacts(session, subject, include_anchor=True)` (Plan 1) → moves any existing `_video_sync.json` **and** `_anchor.json` to `…/<subject>/_archive/<date>/`.
2. Loads the archived anchors (via `load_anchor` on the archived file) and pre-populates the live session as **editable seeds** tagged `source="legacy"`. The trial-0 / last-trial baseline onsets are ideal endpoint support; the user confirms/corrects them with the new playback and adds mid-session + change anchors.
3. New subjects (no prior anchors) start empty.

This is the §3.14 policy, now living in the tagger (not in `fit_sync`, which per Plan-1 fix archives only the sync JSON so re-fits stay repeatable).

---

## 6. Bundled fixes / generalizations

- **v3 emission:** update `click_anchor.py`'s `_build_or_merge_anchor_file` path (and the new `tag_session`) to write `event_type` + support change entries; bump the written `schema_version` to 3 (`load_anchor` already migrates older files).
- **`tag_trials` guard:** it currently refuses any sync whose `detection_method != "manual_slope_fit"` — so it would reject the new `manual_multianchor` syncs. Broaden the guard to accept `{manual_slope_fit, manual_multianchor}`, and make its `_slope_fit_frame` orientation-aware (§ below).
- **`sync_validation_figure.py` generalization (resolves Plan-1 blocker #1):** parametrize subject + session (drop the hardcoded `SESSION_NAME` and mouth-only `FRAME_CROP`); make the event→frame mapping **orientation-aware** — branch on `detection_method` before applying `nidaq_to_camera` (a `manual_slope_fit`/2-anchor result stores the inverse orientation; `manual_multianchor`/`derivative` match the converter). Add a **Change_ON sensory fallback** (hit/miss trials) when a session has too few licks / no lick-responsive unit. Emit a per-session pass/fail + PNG; non-blocking.
- **Front-cam clock derivation:** `derive_front_cam_sync(session, subject)` — metadata-only; both cameras share the USB clock, so `front_offset = eye_offset + eye_slope · (Δfirst_frame_ms / 1000)`; write a `front_cam` block into the sync JSON. (The Plan-1 A1 pilot confirmed eye/front share a clock domain — identical durations, ~2:1 frames.)

---

## 7. Components / files

**New:** `scripts/video/tag_session.py` (unified entry point orchestrating the modes; thin — delegates to library + reused render/scrub helpers); label-sidecar IO in `video_sync.py` (`save/load_video_labels`); `derive_front_cam_sync` in `video_sync.py`; tests.
**Modified:** `scripts/video/click_anchor.py` (full-frame view + per-session crop, playback timer, change-mode + auto-cycle queue, RectangleSelector ROI capture, live pupil overlay, per-frame label keys, v3 emission; refactor the reusable scrubber/render pieces so `tag_session` composes them); `scripts/video/tag_trials.py` (guard accepts `manual_multianchor`, per-session crop, orientation-aware slope-fit frame); `scripts/video/sync_validation_figure.py` (subject/session params, orientation-aware, Change_ON fallback); `src/visdetect/analysis/constants.py` (any new label/ROI keys if needed).

**Decomposition note:** `click_anchor.py` is already ~900 lines and this adds substantial interactive surface. The plan should **extract the reusable primitives** (frame reader, HUD, scrubber loop, montage) into a small module (e.g. `scripts/video/_tagger_ui.py` or `visdetect.core.tagger_ui`) that both `tag_session` and the legacy commands compose, rather than growing one file past what fits in context.

---

## 8. Testing

GUI interaction can't be unit-tested directly, so **factor the logic out of the event loop** and test that:
- **Pure logic (unit-tested):** the big-change queue construction (size-4-then-2, hit/miss filter, order); change-jump predicted-frame math at each seeding stage; the label-sidecar schema round-trip + atomic write; the v3 change-anchor emission; `derive_front_cam_sync` offset math; the orientation branch in the validation-figure event→frame mapping; the migration/seed (archive → pre-load as editable seeds); per-session crop derivation from the eye ROI.
- **Thin GUI shell (manual verify):** the RectangleSelector callbacks, playback timer, and key dispatch are kept minimal and exercised in the A1 pilot.
- Existing `tests/video/*` stay green; add `tests/video/test_tagger_logic.py`, `tests/video/test_video_labels.py`, `tests/video/test_front_cam_sync.py`, `tests/video/test_validation_orientation.py`.

**Acceptance = the A1 pilot itself:** 2 sessions each on BG_031/039/038 tagged end-to-end (mixed baseline+change anchors → `cv_rmse` `good`/`review` → labels captured → neural figure), proving the whole of sub-project A on real data.

---

## 9. Risks / open

- **Playback smoothness over staged-local video:** decode via `cv2.VideoCapture` on a locally-staged copy (Plan-1 `stage_session_video`) for responsive playback; scrubbing random frames over Samba is laggy. The tagger should offer to stage the session first.
- **RectangleSelector + imshow coordinate mapping** on the full frame vs. the eye-zoom must stay in full-frame coords for the sidecar.
- **`click_anchor.py` refactor risk:** extracting shared primitives must not regress the existing baseline-only flow (kept working throughout).
- **Change-onset fuzziness** (from the A spec): handled by the CV-RMSE tiers + more-anchors, not the tagger; the optional corneal motion/flow guide (A spec §3.3d) is out of scope unless the pilot needs it.

---

## 10. Decisions log (user)

1. **Unified single-pass tool** (not multi-tool, not wizard).
2. **Playback:** `space` = play/pause, `enter` = save anchor; `[`/`]` speed; step when paused.
3. **Change selection:** auto-cycle the big-change queue (size-4 then size-2, hit/miss), `c` toggles baseline⇄change.
4. **ROI capture:** rectangle drag (mouth + eye), live pupil overlay in the eye box; **free at any time, not a forced first step**.
5. Full label-capture layer (per-frame pupil-confirm `u` / blink-flag `x`) → decoupled `data/cache/video_labels/` sidecar (carried from the A spec).
