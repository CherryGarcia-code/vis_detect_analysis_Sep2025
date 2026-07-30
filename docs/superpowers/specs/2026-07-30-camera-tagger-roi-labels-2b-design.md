# Plan 2b — Amortized ROI + per-frame pupil label capture (design)

**Date:** 2026-07-30
**Status:** design approved (brainstormed with the user 2026-07-30)
**Builds on:** Plan 2a unified tagger (merged to `main` `5107003`), Plan 1 sync backbone (`057c622`)
**Parent spec:** `docs/superpowers/specs/2026-07-23-camera-tagger-ux-design.md` (§4–5 originally sketched this layer)

---

## 1. Purpose

The manual tagging pass is the expensive resource in this pipeline: a human watching video, one
session at a time, across ~40 sessions × 4+ subjects. Plan 2a made that pass produce a validated
video↔neural clock. **Plan 2b makes the same pass also produce the per-session ROIs and per-frame
pupil labels that sub-projects B and C consume** — so the videos are never re-watched for those.

- **Sub-project B** (motion energy → true movement onset) needs a **mouth/face ROI** per session.
- **Sub-project C** (pupillometry) needs an **eye ROI** plus **human-judged pupil frames** to
  validate the detector and quantify its error.

**Non-goal:** motion-energy and pupil-trace *extraction*. Plan 2b **captures** what B and C consume;
it does not compute traces. Keeping extraction out means the label schema can be settled now without
waiting for B/C to firm up.

---

## 2. Scope decisions (user, 2026-07-30)

| Question | Decision | Rationale |
|---|---|---|
| ROI across sessions | **Seed from the most recent prior session of that subject; confirm or adjust** | Camera geometry is usually fixed within a subject; drawing from scratch ~80+ times is wasted effort AND invites inconsistency. Cross-session motion energy is only comparable if the ROI means the same thing each session. |
| ROI within a session | **One ROI per session, drawn generously** | Simple schema, simple downstream extraction; ME tolerates a slightly loose box. Time-ranged ROIs would burden every consumer for a rare case. |
| Cameras | **Eye cam only** | Sub-project D (breathing/body) is not designed; an ROI drawn 40× against a guessed definition is worse than none. Also avoids staging ~8 GB of front cam per session. The ROI layer is built camera-agnostic so adding front cam later is cheap. |
| Per-frame labels | **confirm / correct / blink** | See §5. A *correction* replaces the originally-proposed "reject" tag. |
| PNG crops of labeled frames | **No — JSON only** | Frame index + video is sufficient; duplicated pixels drift from the JSON. An exporter can regenerate crops on demand if C ever trains a learned detector. |
| `f` zoom view | **Restored, derived from the eye ROI** | It was removed in 2a only because no ROI existed, leaving a BG_046-specific absolute crop that landed on BG_031's snout. With a real per-session eye ROI the zoom is correct for every subject, and helps on faint corneal reflections (BG_046). |

---

## 3. Architecture

Pure logic lives in a new library module; the GUI only wires it up. `tag_session.py` is already
~600 lines and its own design calls for it to stay thin — and sidecar schema / seeding / clamping is
exactly the logic that is cheap to unit-test and painful to debug through a GUI.

```
src/visdetect/analysis/video_labels.py     NEW — pure, no GUI, fully unit-tested
scripts/video/tag_session.py               MODIFIED — key handlers + RectangleSelector wiring + overlay
```

**Reused, not rebuilt** (per the project rule to search before writing):
- `video_sync.detect_pupil_in_frame(gray, search_roi=(y0,y1,x0,x1), ...) -> Optional[Dict]` — the live proposal
- `tagging.eye_zoom_crop(eye_roi, pad)` — kept in 2a expressly for this; **returns UNCLAMPED coords**
- `scripts/video/select_roi.py` — canvas + JSON-persistence patterns
- `_tagger_ui.run_scrubber` — the shared scrubber, extended with the mouse-drag hook

---

## 4. ROI capture

- `m` / `e` arm a matplotlib `RectangleSelector` on the **full-frame** view; the drag sets the mouth /
  eye box. **ROIs are always stored in full-frame pixel coordinates**, never view-relative, so a box
  drawn while zoomed still means the same thing.
- Setting the eye box immediately runs `detect_pupil_in_frame` on the current frame and overlays the
  proposed ellipse — a bad box shows an absent or obviously wrong ellipse, so the user re-drags. The
  feedback is the validation; there is no separate "check your ROI" step.
- ROI capture is **free at any time and never a forced step**. A session may be tagged for sync alone.

### 4.1 Seeding and provenance

On open, if this session has no sidecar, the most recent **prior** session of the same subject is
located and its ROIs are pre-loaded and drawn. The user accepts them implicitly (by labeling/saving)
or re-drags.

**Every ROI records how it came to exist:** `source: "drawn" | "inherited:<session>"`. Without this
there is no way, later, to distinguish an ROI a human actually looked at from one silently copied
forward across 30 sessions — a distinction that matters the first time a motion-energy result looks
odd. Re-dragging an inherited ROI flips it to `drawn`.

**Chronology must use the project's canonical id helpers** (`canonical_session_id` /
`parse_session_date`), never a raw `sorted()` on the id string: `'1072025'` sorts before `'23062025'`
though 1 Jul is after 23 Jun, and BG_031/BG_039 use 6-digit `DDMMYY` camera ids. "Most recent prior
session" is a date comparison, not a string comparison.

**`frame_size` (H, W) is recorded** with the ROIs. An inherited ROI whose recorded frame size differs
from the current session's is **not applied** (it is offered as a starting point only with a warning),
since an absolute-pixel box is meaningless at a different resolution.

---

## 5. Per-frame labels: confirm / correct / blink

While on any frame the user is already viewing (zero extra navigation):

| Key | Meaning | Stored |
|---|---|---|
| `u` | the proposed ellipse is **correct** | `verdict="confirmed"`, detector's ellipse |
| `p` | the proposed ellipse is **wrong — here is the right one** | `verdict="corrected"`, detector's proposal **and** the human ellipse |
| `x` | **blink / occluded** — no valid pupil in this frame | `verdict="blink"` |

**Correction, not rejection.** An earlier draft had a one-key "reject" tag. Correction supersedes it:
a correction registers that the detector was wrong (giving the same denominator) *and* supplies the
ground truth, which a bare tag cannot. This matters most for the failure mode that matters most —
a pupil partially covered by the eyelid yields an ellipse fitted to the visible crescent, reporting a
plausible but **too-small diameter**. A "wrong" tag cannot quantify that bias; a corrected ellipse
can. If eyelid occlusion covaries with arousal, that bias could otherwise masquerade as a real
pupil–arousal effect.

**Correction interaction:** `p` arms the same `RectangleSelector` used for ROIs; the user drags a box
around the true pupil and the **inscribed axis-aligned ellipse** is stored. Reusing the ROI drag keeps
the interaction and the code minimal.

**Known approximations, recorded honestly:**
- The inscribed ellipse is **axis-aligned** — rotation angle is lost. Negligible for a near-circular
  rodent pupil; a two-drag (major/minor axis) variant can follow if it ever matters.
- On a partially occluded pupil the human is **estimating the hidden extent**. That is an informed
  inference, not measured truth — but it is what the detector *should* report, so it is the right
  target. Corrected ellipses on frames also marked occluded should be read as estimates.

**Sampling caveat (must survive into any analysis that uses these labels):**
- Labeled frames are the ones the user visited — **baseline and change onsets** — not a random sample
  of the session. Any error rate computed from them is *"error rate at task events"*, not across the
  session. This is defensible because pupillometry here will be analyzed event-aligned, but it must
  never be reported as a session-wide detector error rate.
- **Unlabeled ≠ rejected.** A frame with no verdict was simply not judged. The denominator for any
  error rate is `confirmed + corrected + blink`, never total frames.

---

## 6. Sidecar schema

`data/cache/video_labels/<subject>/<session>.json` — **decoupled from the anchor JSON** so the sync
artifacts stay clean and the label schema can evolve with B/C independently of the sync contract.

```json
{
  "schema_version": 1,
  "subject": "BG_031",
  "session": "09042025",
  "camera": "eye_cam",
  "frame_size": [1024, 976],
  "rois": {
    "mouth": {"box": [y0, y1, x0, x1], "source": "drawn"},
    "eye":   {"box": [y0, y1, x0, x1], "source": "inherited:08042025"}
  },
  "frames": [
    {"frame_idx": 12345,
     "verdict": "confirmed",
     "proposed_ellipse": {"cx":.., "cy":.., "major":.., "minor":.., "angle":..},
     "corrected_ellipse": null,
     "labeled_at": "<iso8601>"}
  ]
}
```

- `frames` entries are **keyed on `frame_idx`** — re-labeling a frame replaces its entry (upsert), so
  changing your mind never produces duplicate or contradictory records.
- `rois` may be partially populated (`mouth` only, `eye` only, or neither).
- Written **atomically** (write temp + replace, mirroring `tag_trials._persist_overrides`) on every
  ROI/label change, plus a final flush on quit — a crash mid-session must not corrupt prior work.

---

## 7. Zoom view (restored)

`f` toggles full-frame ⇄ eye-zoom, the crop derived from the eye ROI via `eye_zoom_crop`. Full frame
remains the default and the only view ROIs are drawn on.

> **HARD REQUIREMENT — clamp before indexing.** `eye_zoom_crop` returns an **unclamped**
> `(y0,y1,x0,x1)`: padding an ROI near a frame edge can produce **negative or out-of-frame**
> coordinates. NumPy slicing with a negative index does **not** error — it silently wraps from the far
> edge and yields the **wrong crop**. Clamp to `0 <= y0 < y1 <= H`, `0 <= x0 < x1 <= W` before any
> frame indexing. This resurrects the `_clamp_crop` guard deleted in 2a; it belongs in
> `video_labels.py` as tested library code, not as a GUI-local helper.

---

## 7.1 Implementation risks to handle explicitly

**Matplotlib default keymaps collide with two of the new keys.** `p` is matplotlib's default *pan*
binding and `f` its *fullscreen* binding. Plan 2a already neutralizes
`keymap.{save,fullscreen,xscale,yscale,back,forward,home,pan,zoom,grid}` inside
`_tagger_ui.run_scrubber`, which covers both — but this is load-bearing, not incidental: if that
clearing is ever removed, `p` would pan the axes and `f` would go fullscreen *in addition to* the
tagger's own handler. The new keys `m`, `e`, `u`, `x` are free of matplotlib defaults. Any future key
must be checked against that list.

**`RectangleSelector` must not fight the scrubber.** The selector is armed only on demand (`m`/`e`/`p`)
and disarmed as soon as the drag completes or is cancelled, so arrow-key stepping, playback, and
save/quit keep working while no drag is in progress. A drag in progress must not advance the frame —
the frame under the box has to be the frame the ROI/ellipse is recorded against.

**The shared scrubber gains a mouse hook.** `run_scrubber` currently exposes key hooks only. Adding
ROI drag requires a mouse/selector seam. It must be optional, so `click_anchor` (which does not use it)
is unaffected — the same preservation constraint that governed the 2a refactor.

## 8. Testing

**Unit-tested pure logic** (`tests/video/test_video_labels.py`):
- sidecar schema round-trip; atomic write leaves no partial file on failure
- `upsert_frame_label` idempotence — re-labeling replaces, never duplicates
- ROI seeding picks the most recent **prior** session by **date**, including the 6-digit `DDMMYY`
  and leading-zero-day cases, and never a later session
- seeding refuses/warns on a `frame_size` mismatch
- `clamp_crop` on negative, oversized, and inverted boxes
- inscribed-ellipse geometry from a drag box

**GUI** (no automated test — as in 2a): import/parse + `--help` checks, then a **real acceptance pass**
by the user. 2a's pilot is the precedent: it caught a backend bug, a wrong crop, and a misleading
legend that every headless check had passed over.

> **Lesson carried from the 2a pilot:** headless verification runs under `MPLBACKEND=Agg` and therefore
> cannot validate anything about interactive backend selection or real GUI behavior. Any new GUI
> surface here is unproven until a human drives it. `tests/video/test_tagger_backend.py` pins the
> backend invariant; everything else needs the pilot.

---

## 9. Acceptance

1. Full suite green (97 at branch point, plus new tests).
2. **Interactive pass by the user** on a real session: draw mouth + eye ROI, see the live pupil
   overlay, confirm a few frames, correct at least one, flag a blink, toggle `f` zoom, quit, re-open
   the same session and confirm ROIs + labels persisted.
3. **Cross-session seeding proven:** open a *second* session of the same subject and confirm the ROIs
   are pre-loaded from the first and marked `inherited:<session>`.

---

## 10. Explicitly out of scope

- Motion-energy and pupil-trace extraction (sub-projects B and C)
- Front-camera ROIs (until sub-project D is designed) — the ROI layer stays camera-agnostic so this
  is cheap to add
- PNG crop export (an exporter can regenerate from JSON + video on demand)
- Time-ranged / mid-session ROIs
- Free-text frame notes, and a separate reject tag (superseded by correction)
- `sync_validation_figure` generalization + front-cam clock derivation — those are **Plan 2c**
