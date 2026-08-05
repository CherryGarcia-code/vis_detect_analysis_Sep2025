# Camera Pipeline — Cross-Subject Baseline_ON Sync + Amortized Label Capture (Design)

**Date:** 2026-07-21
**Status:** Design (brainstormed, awaiting user review → writing-plans)
**Scope:** Sub-project **A** specified in full; **B/C/D** captured as a roadmap appendix (each gets its own spec when reached).
**Subjects (priority):** BG_046 → BG_031 → BG_039 → BG_038 → rest.

---

## 1. Context & Motivation

The camera sync TTL was not recorded, so video frames have no hardware alignment to the NI-DAQ/neural clock. Baseline_ON (grating appearance) must be recovered **from the video itself**, then a linear clock maps video time ↔ NI-DAQ time. Once that clock exists, camera-derived behavioral features (motion energy, pupil, body/breathing) can be placed on the neural timeline.

**Current live state (verified 2026-07-21):**
- `src/visdetect/core/video_sync.py` (~2900 lines) is mature. The canonical human-in-the-loop workflow is `click_anchor.py` (tag Baseline_ON frame for trial 0 + `--anchor-last`) → `fit_sync.py` (linear clock) → `tag_trials.py` (per-trial fix).
- **All 24 committed syncs are BG_046, eye-cam only, and 2-anchor.** A 2-point line has zero residual by construction, so their `rmse_ms = cv_rmse_ms = 0` and `quality = "good"` are **not validation** — they carry no information about clock nonlinearity, dropped frames, or a mis-clicked anchor.
- **The validated multi-anchor machinery already exists but is bypassed:** `fit_clock_model` (video_sync.py:1497) does Theil-Sen + MAD outlier rejection + 5-fold temporal cross-validation → a real `cv_rmse_ms`. The Phase-2 manual path routes to `fit_2anchor_clock`, which hardcodes `cv_rmse_ms = 0.0`.
- **No feature extraction exists** (no motion energy / pupil / DLC / facemap ever computed for any BG_046 session). Reusable seeds: `detect_pupil_in_frame` (OpenCV ellipse fit), `load_camera_metadata`, `find_camera_files`, `camera_to_nidaq`, and `sync_validation_figure.py` (already finds a lick-responsive unit and plots corrected-vs-uncorrected peri-lick PSTHs). Installed vision deps: opencv-python-headless, scikit-image, imageio — **no DLC/facemap/torch**.
- **Camera data exists for all four priority subjects** (X: `Cameras_sortIntoSubjects/<subject>_<ddmmyy>/`): BG_046 (52 dirs), BG_038 (48), BG_031 (29), BG_039 (19), each with `Eye_cam.mp4` + `Front_cam.mp4` + `_metadata.csv`. Eye ≈ 50 fps, front ≈ 100 fps. All four have staging manifests (`data/<subject>_staging_manifest.csv`); BG_041 has only a UM-extract manifest (needs `stage_sessions.py`).
- **State framework is the behavioral state labeler**, not HMM: `data/cache/state_tags/<subject>/<session>.csv`, labels **Impulsive / StimSens / Disengaged / Abort** (`STATE_LABEL_COLORS`, config.py:240).
- **Reference:** Khilkevich & Lohse 2024, `paper_references/Brain-wide dynamics linking sensation to action during decision-making.pdf` — its `movement.pkl` schema (`mouth_me / whisker_me / pupil_area` on eye-cam frame times) is the template for B.

**Goal of this project:** a *validated* video↔neural clock for every neural-roster session of each subject, produced with the human-in-the-loop tag-then-expand workflow — and, because the manual pass is the expensive resource, **amortize it** by capturing per-session feature ROIs and per-frame labels in the same pass.

---

## 2. Decomposition

Four sub-projects. A is the foundation; B/C/D each depend on A's clock and get their own spec later.

| | Sub-project | Depends on | Camera |
|---|---|---|---|
| **A** | Cross-subject Baseline_ON sync + label capture | — | eye |
| **B** | Motion energy → true movement onset | A | eye (face/mouth) |
| **C** | Pupillometry → arousal / state validation | A | eye |
| **D** | Front-cam sync + body/breathing motion | A | front |

B and C are independent of each other. D's sync half (deriving the front-cam clock from the eye-cam clock) is cheap and folded into A.

---

## 3. Sub-project A — Design

### 3.1 Goal & success criteria

Produce a committed, **cross-validated** sync clock for every neural-roster session (per subject, priority order) that has usable video.

Acceptance per session:
- Clock fit via `fit_clock_model` on ≥3 anchors (≥5 recommended for a stable Theil-Sen + meaningful CV); **`cv_rmse_ms` reported and used for the quality tier** — `good` < 20 ms, `review` 20–40 ms, `failed` > 40 ms (≈ ≤1 / ≤2 eye-cam frames; consistent with the existing quality tiers and the ~250 ms integration timescale).
- A per-session **neural sharpening figure** (non-blocking) confirming the corrected clock; skipped-with-note when unavailable (§3.4).
- **Zero writes to X:** (§3.6).
- Sync cache **namespaced by subject** — no cross-subject session-date collisions (§3.5).
- Label sidecar populated for the session (§3.7).

### 3.2 Reuse vs. build

| Reuse as-is | Build / modify |
|---|---|
| `fit_clock_model` (Theil-Sen + MAD + 5-fold CV → `cv_rmse_ms`), `_temporal_cv` | `click_anchor` → tagger: playback, mid-session anchoring, change-anchor jump mode, full-frame view toggle, ROI draw + live pupil overlay, per-frame labels |
| `load_camera_metadata`, `find_camera_files`, `camera_to_nidaq`, `nidaq_to_camera` | `fit_sync`: route ≥3 anchors through `fit_clock_model`; sparse-manual quality rule; mixed-event-type anchors |
| `tag_trials` per-trial UI | Subject-aware, suffix-tolerant `camera_dir_to_session` / `find_camera_files` |
| `sync_validation_figure.py` (neural gate skeleton) | Generalize the validation figure (any subject, Change_ON fallback, pass/fail, non-blocking) |
| `detect_pupil_in_frame` (ellipse fit) | Subject-namespaced caches; local metadata reconstruction; local staging helper; label-sidecar IO |
| `sync_status.py` (progress tracker) | Generalize sync_status to subject-aware + `cv_rmse` + label coverage |

### 3.3 Tagging workflow (per session)

1. **Stage** the session's eye-cam mp4 from X: → local `VIDEO_STAGING_DIR` (one-time bulk read; §3.6).
2. **Tag** in the upgraded tagger:
   - **Views:** full-frame (draw the mouth/face ME ROI; locate the eye ROI) ↔ eye-zoom (pinpoint the exact onset frame; confirm the pupil ellipse). *The tagger currently shows only the eye crop; the full-frame view is a new requirement (user-confirmed: the full eye-cam frame contains the face, the current close-up does not).*
   - **Playback:** play a short clip from any frame; "rewind N frames and play forward" to judge the grating **appearing in motion** — decisive on faint sessions.
   - **Anchors (≥5, spread across the session; mix types):**
     - **Baseline_ON anchors** — grating appears from ITI (valid on all outcomes; lick-independent).
     - **Change_ON anchors (NEW)** — jump to the predicted video frame of a chosen Change_ON event, prioritizing `BIG_CHANGE_SIZES = {4.0, 2.0}`, restricted to `hit`/`miss` trials (`EVENT_VALID_OUTCOMES["Change_ON"]`); play around it, step toward the drift-speed-change frame. Abundant, and visible via *motion* even when the static grating contrast is faint — so they carry the weak sessions — but the exact onset frame is **lower-precision** than a baseline appearance (a speed change is gradual to perceive), especially in a weak reflection. Handled by the precision strategy below, not by pretending each anchor is frame-perfect.
     - Seed the jump with the existing coarse offset or one baseline anchor → provisional clock → predict each Change_ON frame → refine.
   - **Anchor-precision strategy (change-onset fuzziness).** (a) **Baseline-onset anchors are the precision backbone** wherever the grating is visible enough to see it appear (a sharp binary transition); change-anchors add spread and carry weak sessions. (b) **No anchor needs to be frame-perfect** — Theil-Sen averages roughly zero-mean anchor jitter into the slope, and the **CV-RMSE reports the resulting precision**, so a fuzzy session lands honestly in `review`, never a fake `good`. (c) **The cure for fuzzy anchors is more of them, spread wide** — slope uncertainty falls ~ σ/(√N · spread); on weak sessions tag 8–10 and let the CV say when it is tight enough. (d) *Optional aid:* overlay a corneal-patch motion/optical-flow trace that **steps up ~4× at a size-4 change** — a level-step is localizable to ~1–2 frames via a pre/post-window comparison, sharper than eyeballing (most useful on moderate reflections; the human still confirms). Deferred/optional — add only if the A1 pilot shows change-anchors are too fuzzy without it.
   - **Label capture (same pass; §3.7):** draw mouth + eye ROIs (full-frame coords) with a live `detect_pupil_in_frame` overlay confirming a good detection; optionally confirm/adjust the pupil ellipse and flag blink/occlusion on each frame you're already viewing.
3. **Fit** (`fit_sync`): ≥3 anchors → `fit_clock_model` (mixed baseline+change events; a clock is a clock — display latency is common to both types and absorbed into the offset). 2 anchors → `fit_2anchor_clock` fallback (flagged `review`, no CV).
4. **Gate:** `cv_rmse_ms` tier + non-blocking neural figure (§3.4).
5. **`tag_trials`** only for flagged/weak sessions.

### 3.4 Validation gate (decoupled)

- **Primary (mandatory, cheap, lick-free):** **CV-RMSE** from `fit_clock_model`. Pure timing arithmetic — needs neither spikes nor licks. This is what makes "good" mean something. *Note:* the dense-path default is 5-fold temporal-block CV, which leaves ~1 anchor/fold at the sparse manual counts here — so for manual anchors use **leave-one-out** (or `n_folds ≤ n_anchors − 1`); the fit routing must set this, not inherit the dense default.
- **Secondary (non-blocking confirmation):** generalize `sync_validation_figure.py` → per-session PNG showing that lick-responsive neurons sharpen at the *corrected* lick time vs. uncorrected. Fallbacks in order: (1) lick-aligned (Hit/FA) — the most informative, since the lick is an independent event not used as an anchor; (2) **Change_ON sensory-response** sharpening for lick-sparse sessions (note: partly redundant with change-anchors, which pin video↔nidaq at those events by construction, so it is a weaker check); (3) skip with a logged reason if no responsive unit / too few events. **A session never fails on the neural figure alone** — it fails only on CV-RMSE.

### 3.5 Multi-subject correctness

- **Namespace the caches.** `data/cache/video_sync/`, `motion_energy/`, `pupil/`, and the new `video_labels/` are currently flat and keyed by bare `DDMMYYYY` → BG_031 and BG_046 sessions on the same calendar day collide. Move to `…/<cache>/<SUBJECT>/…` (mirroring `PKL_DIR`).
- **Fix `camera_dir_to_session` / `find_camera_files`.** They hardcode `subject="BG_046"`, assume exactly 6-digit `DDMMYY`, and break on suffix dirs (`_laser`, `_b`). Make subject-aware and suffix-tolerant; pass the real subject (from `config.SUBJECT` / `VISDETECT_SUBJECT`).
- **Roster** = `load_staging_manifest(qc_only=True)` for the subject **∩** camera-dir availability; log-and-skip no-camera sessions (e.g. BG_046 19082025 has a pkl but no video).

### 3.6 Ceph-safe posture (X: is read-only)

- **All writes local.** Redirect `reconstruct_camera_metadata.py` — it currently backs up, **overwrites the metadata CSV in place**, and writes a `.reconstructed.json`, all on X: (`open(prov_path,"w")`, line 121) — to write rebuilt timestamps + provenance into `data/cache/video_sync/<SUBJECT>/`; have `find_camera_files`/`load_camera_metadata` prefer the local reconstructed copy. **Net: zero writes to ceph.**
- **Local staging helper** `stage_session_video(subject, session, cams)`: one-time bulk *read* of the mp4(s) from X: → `VIDEO_STAGING_DIR` (gitignored local scratch), tag/extract locally, delete after. Keeps scrubbing snappy and keeps random-access decode off Samba. This is the same helper B/C/D use for batch extraction (never decode whole videos over ceph).
- A test guards this: patch `CAMERA_ROOT` and assert the sync path performs **no writes** under it.

### 3.7 Label-capture layer (amortized)

Decoupled from the sync anchor JSON so A's clock stays clean and independently valid, and the schema can evolve as B/C firm up.

- **Sidecar:** `data/cache/video_labels/<subject>/<session>.json` (versioned `schema_version`), plus optional small frame crops under `…/<session>/frames/` for a self-contained ML set (local only).
- **Per session:** `me_roi` (mouth/face box, full-frame coords), `eye_roi` (box), each with the live-overlay detection outcome at capture time.
- **Per tagged frame (optional):** `{frame_idx, event_type, pupil_ellipse{cx,cy,major,minor,angle}, confirmed:bool, blink:bool, occluded:bool}`.
- **Consumers (later):** B seeds its mouth ROI from `me_roi`; C seeds its eye ROI + validates/​trains the pupil detector against the confirmed ellipses. Both still run a fast `validate_roi` confirmation seeded from the human ROI (computed-feature feedback: ME trace / detection rate) before any batch extraction.
- Capturing ROIs now does **not** lock B/C methods — an ROI is a rig-specific spatial box, independent of *how* motion energy is computed.

### 3.8 Anchor schema (generalized)

v2 anchor JSON per anchor generalizes from baseline-only to:
```
{ trial_index, event_type: "baseline_on"|"change_on",
  nidaq_event_s, change_size?, outcome?,
  video_frame_idx, video_time_s, clicked_at }
```
`fit_clock_model` consumes `(nidaq_event_s → video_time_s)` pairs regardless of `event_type`. Legacy v1/v2 baseline-only anchors read transparently.

### 3.9 Components

**New:** `stage_session_video()` helper (video_sync.py or `scripts/video/stage_video.py`); label-sidecar IO (`save/load_video_labels`); `data/cache/video_labels/` (data); tests.
**Modified:** `click_anchor.py` (→ tagger upgrades), `fit_sync.py` (fit routing + quality rule), `sync_status.py` (subject-aware + cv_rmse + labels), `reconstruct_camera_metadata.py` (write local), `video_sync.py` (subject/suffix-aware camera-dir fns, generalized anchor schema, local reconstruction, front-cam clock derivation), `sync_validation_figure.py` (generalize), `config.py` (namespaced dirs + `VIDEO_LABELS_DIR`, `VIDEO_STAGING_DIR`), `constants.py` (schema/threshold constants).

### 3.10 Data flow

```
staging manifest (subject, qc_only) ∩ camera dirs
  → stage_session_video (X: read-only → local)
  → tagger: full-frame view; playback; ≥5 baseline+change anchors;
            mouth+eye ROIs w/ live pupil overlay; per-frame ellipse/blink labels
      ├─ anchors JSON  (local, namespaced)
      └─ label sidecar (local, namespaced)
  → fit_sync (fit_clock_model, CV-RMSE, mixed events)
      → sync JSON (local, namespaced)  [+ front-cam clock derived]
  → validation figure (non-blocking)
  → sync_status (progress: cv_rmse + label coverage)
```

### 3.11 Rollout (tiered)

- **A0 — infra + BG_046 self-check.** Build §3.2 items + tests. Then retag BG_046 under the new workflow (§3.14), starting with 2–3 sessions as the self-check: legacy 2-anchor endpoints are archived and pre-loaded as editable seeds, the user adds mid-session + change anchors and the ROI/label layer, and the new CV-RMSE fit is sanity-checked against the old slope/offset.
- **A1 — cross-subject pilot.** 2 sessions each on BG_031 / BG_039 / BG_038, end-to-end (anchors → fit → gate → labels), proving namespacing + camera-dir fixes + the change-anchor path.
- **A2 — batch the neural rosters** in priority order: BG_046 (seeded retag, §3.14) → BG_031 → BG_039 → BG_038 → rest. BG_041 and any "rest" subject without a staging manifest need `stage_sessions.py` first.

### 3.12 Testing

- **Unit:** `camera_dir_to_session` (6-digit, `_laser`/`_b` suffixes, subject); namespaced path builders; anchor schema round-trip (baseline + change, mixed); fit routing (≥3 → `fit_clock_model`, 2 → fallback); sparse-manual quality rule; label-sidecar IO; local-metadata-reconstruction target; staging (mock copy); **read-only-X: guard**.
- **Integration:** existing ~52 sync tests still pass; one end-to-end validated session per new subject.

### 3.13 Risks & open questions

- **Tagger full-frame view** is net-new (currently eye-crop only) — modest, frames already decoded.
- **Change-anchor precision (user-flagged):** localizing a drift-speed change frame-by-frame in a weak reflection is genuinely fuzzy. Handled by graceful degradation, not denial (§3.3 precision strategy): jitter averages into the slope, CV-RMSE surfaces residual precision, more-anchors-spread-wide tightens it, and the honest floor is that genuinely weak sessions settle at `review` (~30–40 ms) — still well inside the 250 ms integration timescale and fine for feature alignment / ~200 ms movement-onset work. We do not manufacture precision we don't have. Optional corneal motion/flow step-guide (§3.3d) if the pilot needs it.
- **Front-cam clock derivation** assumes a shared USB clock domain + a known first-frame offset — validate on ≥2 sessions before trusting.
- **Staging disk:** ~23 GB/session → process one session at a time, delete after.
- **Label schema forward-compat:** versioned; B/C may add fields.

### 3.14 Migration & re-tag policy (existing anchors)

Retag **all** sessions, including the 24 existing BG_046 ones (they are 2-anchor with no label layer). The policy makes this lossless and correctable — **archive *and* seed, not either/or**:

- **Archive, don't discard.** On first retag of a session, move any existing `<session>_anchor.json` + `<session>_video_sync.json` to `data/cache/video_sync/<SUBJECT>/_archive/<YYYY-MM-DD>/` (local). Preserves the old fit, gives a rollback, and removes the misleading by-construction `RMSE = 0` "good" from the active set.
- **Seed from legacy, keep editable.** Pre-load the archived anchors into the new session as **editable** anchors tagged `source: "legacy_2anchor"` (migrated to `event_type: "baseline_on"`). The trial-0 and last-trial baseline onsets are well-spread endpoints — ideal linear-fit support — so the user only *adds* mid-session baseline + change anchors and the ROI/label layer, and confirms/corrects the two seeds with the new playback. No prior effort is wasted.
- **Idempotent re-fit.** `fit_sync` archives the prior sync JSON before writing a new one — **never a silent overwrite** (matches the no-data-loss / ceph-caution posture). Re-running is safe and repeatable.
- **New subjects** start with an empty anchor list (no seed).

---

## 4. Roadmap Appendix (B/C/D — own specs later)

- **B — Motion energy → true movement onset.** New `camera_features.py`: `compute_motion_energy` = mean |frame diff| in the mouth/face ROI (from the label sidecar), **following Khilkevich & Lohse 2024 Methods** (mouth_me/whisker_me; read the paper's Methods at spec time). Place on the neural clock via A; detect movement onset (e.g. median + 3·MAD) vs. spout contact; quantify the ~200 ms offset; show neural PSTH sharpening at ME-onset vs. spout. Batch decode **staged-local**. First fork: local-staging cadence vs. HPC.
- **C — Pupillometry.** Build on `detect_pupil_in_frame` + the confirmed-ellipse labels: full-trace extraction, blink interpolation, luminance-confound channel. Downstream = arousal proxy validated against the **behavioral state labels** (Impulsive/StimSens/Disengaged/Abort in `data/cache/state_tags/<subject>/`), framed as validation (pupil is a physiologically independent measure). Caveat: do not then loop pupil back into criterion claims (state-labeler circularity).
- **D — Front-cam sync + body/breathing.** Derive the front-cam clock from the eye-cam clock (shared USB domain; metadata-only — folded into A). Then novel body/breathing motion-energy ROIs on the front view.

---

## 5. Decisions Log (user)

1. **Sync rigor:** multi-anchor fit through `fit_clock_model` (real CV-RMSE) + optional non-blocking neural sharpening figure.
2. **Session scope:** tiered, neural-roster-first, per subject.
3. **Tagging GUI additions (user idea):** video playback + step-to-confirm; Change_ON anchors prioritizing size-4 then size-2 (hit/miss only), generalized anchor schema.
4. **Ceph:** X: strictly read-only; all writes local; metadata reconstruction redirected local; local staging helper.
5. **State framework:** behavioral state labels (not HMM).
6. **Label capture:** **full capture layer** — per-session mouth + eye ROIs with live pupil feedback + per-frame ellipse/blink labels → decoupled `data/cache/video_labels/` sidecar.
7. **Existing-anchor migration:** retag all sessions incl. BG_046; archive prior anchor/sync JSONs and pre-load the legacy 2 anchors as editable seeds (correct-in-place), never silent-overwrite.
