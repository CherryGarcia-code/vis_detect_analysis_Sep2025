"""tag_session.py — unified single-pass video<->neural sync tagger (Plan 2a).

One window that lets the user place BOTH baseline and change anchors for a
session's eye-cam video, with live sync-quality feedback, then run
``fit_sync`` to write the ``manual_multianchor`` sync JSON. This entry point is
*thin*: it COMPOSES already-built primitives and does not reimplement any of
them.

Composed primitives
--------------------
* ``visdetect.analysis.tagging`` — ``build_change_queue`` (Task 1),
  ``seed_from_archive`` (Task 3), ``nidaq_to_frame_oriented`` (Task 4),
  ``provisional_change_clock``, ``ChangeTarget``.
* ``scripts/video/_tagger_ui.py`` — ``run_scrubber(cfg)`` / ``ScrubberConfig``
  (Task 6): the shared keyboard scrubber + HUD core.
* ``visdetect.core.video_sync`` — ``stage_session_video``, ``find_camera_files``,
  ``load_camera_metadata``, ``compute_predicted_frame_idx`` (baseline jump math),
  ``fit_multianchor_clock`` (live cv_rmse), the v3 anchor writers
  (``_build_anchor_entry``, ``_build_change_anchor_entry``,
  ``_build_v3_anchor_file``, ``_merge_anchor_into_file``), ``save_anchor``.
* ``visdetect.analysis.config`` — ``canonical_camera_session``,
  ``subject_video_sync_dir``, ``ROOT``, ``SUBJECT``.
* ``visdetect.suite.loader.resolve_subject_pkl`` +
  ``visdetect.core.session.load_session`` — subject-aware behavioural PKL load
  (by ``--subject``, not the frozen ``config.SUBJECT`` env).

Plan 2b adds per-session ROI capture on this same pass: ``e``/``m`` drag the
eye/mouth ROI (stored full-frame in the ``video_labels`` sidecar), a live green
pupil ellipse (``detect_pupil_in_frame`` restricted to the eye ROI) is overlaid
via the ``on_refresh`` seam so it survives the scrubber's internal redraws, and
``f`` toggles a CLAMPED eye-zoom derived from the eye ROI (``eye_zoom_crop`` ->
``video_labels.clamp_crop``; a ``None`` clamp means the box misses the frame, so
the view stays full-frame rather than indexing an empty array). ROIs seed from
the subject's most recent prior session (honouring the ``applied`` frame-size
guard) and carry provenance (``inherited:<sess>`` until re-drawn -> ``drawn``).

Keybindings (see docs/superpowers/specs/2026-07-23-camera-tagger-ux-design.md)
------------------------------------------------------------------------------
  arrows / shift / ctrl   step +/-1 / +/-10 / +/-100 frames (built into scrubber)
  space                   play/pause forward
  [ / ]                   slower / faster playback
  j / k                   next / prev target onset (baseline trial OR change queue)
  c                       toggle baseline <-> change target mode
  e / m                   drag the eye / mouth ROI (full-frame view only)
  f                       toggle full-frame <-> clamped eye-zoom
  home / end              first / last target
  d                       delete this target's anchor (current mode's type)
  enter                   save anchor and KEEP the window open (design-primary;
                          the shared scrubber now routes enter through the hook
                          before its default save-and-close). ``s`` is an alias.
  q / esc                 quit without saving the current frame

Run:  py scripts/video/tag_session.py --subject BG_031 --session 09042025
"""
import argparse
import gc
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2

# NOTE: matplotlib backend selection happens AFTER the visdetect imports below,
# not here. Several library modules (visdetect.core.qc, suite.plotting, ...) call
# matplotlib.use("Agg") at import time, which silently clobbers an earlier TkAgg
# and leaves the window unable to show. click_anchor has always ordered it that
# way; tag_session must too. See the backend block after the imports.

# NOTE: we intentionally do NOT `from click_anchor import ...`; click_anchor
# forces TkAgg at import time, which would break the headless spec-import.
# Baseline predicted-frame math comes from the library
# (video_sync.compute_predicted_frame_idx), not a local reimplementation.

# Behavioural PKL is loaded by SUBJECT (not the frozen config.SUBJECT env): the
# subject-aware library resolver (suite.loader.resolve_subject_pkl) finds the
# subject's pkl path, loaded via the PATH-based core loader.
from visdetect.suite.loader import resolve_subject_pkl  # noqa: E402
from visdetect.core.session import load_session as _load_session_path  # noqa: E402
from visdetect.analysis import config  # noqa: E402
from visdetect.analysis.tagging import (  # noqa: E402
    build_change_queue,
    seed_from_archive,
    nidaq_to_frame_oriented,
    provisional_change_clock,
    eye_zoom_crop,
)
from visdetect.core.video_sync import (  # noqa: E402
    find_camera_files,
    load_camera_metadata,
    stage_session_video,
    compute_predicted_frame_idx,
    fit_multianchor_clock,
    save_anchor,
    detect_pupil_in_frame,
    _build_anchor_entry,
    _build_change_anchor_entry,
    _build_v3_anchor_file,
    _merge_anchor_into_file,
)
from visdetect.analysis import video_labels as vl  # noqa: E402

# --- Matplotlib backend selection (MUST stay after the visdetect imports) ----
# visdetect.core.qc / suite.plotting / tf_pulse call matplotlib.use("Agg") at
# import time, so selecting TkAgg any earlier gets silently overwritten and
# plt.show() then warns "FigureCanvasAgg is non-interactive". Selecting it here
# — last — is what click_anchor does and is why its scrubber works.
# The headless verifications (--help / spec-import) run under MPLBACKEND=Agg and
# must not require a display, so honour an explicit Agg request.
import matplotlib  # noqa: E402
if os.environ.get("MPLBACKEND", "").lower() != "agg":
    matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)
from matplotlib.patches import Ellipse  # noqa: E402  (patch artist for the pupil overlay)

# The shared scrubber sits in this same directory; make it importable whether
# run as a script or loaded via importlib.spec_from_file_location. Imported
# after the backend is set (it does `import matplotlib.pyplot`).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tagger_ui import ScrubberConfig, run_scrubber  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("tag_session")

# Script-level default (mirrors click_anchor.DEFAULT_COARSE_OFFSET_S). This is a
# UI seed for the very first change-jump when no anchor exists yet, NOT a
# scientific constant — replaced the instant the user places one anchor.
DEFAULT_COARSE_OFFSET_S = 15.0


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def _entry_implied_offset(a: dict) -> float:
    """``video_time_s - nidaq`` for an anchor entry (baseline or change).

    Robust to schema: prefers ``nidaq_baseline_on_s`` (baseline entries), falls
    back to ``nidaq_event_s`` (change entries).
    """
    nidaq = float(a.get("nidaq_baseline_on_s", a.get("nidaq_event_s")))
    return float(a["video_time_s"]) - nidaq


def _read_coarse_offset(session: str, sync_dir: str) -> Optional[float]:
    """Cached coarse offset for *session* (subject dir first, then global)."""
    for path in (os.path.join(sync_dir, "coarse_offsets.json"),
                 os.path.join(config.VIDEO_SYNC_DIR, "coarse_offsets.json")):
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            val = data.get(session)
            if val is not None:
                return float(val)
    return None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


@dataclass
class TagSessionState:
    """Live tagger state mutated by the scrubber hooks."""

    mode: str = "baseline"                 # {"baseline", "change"}
    baseline_pos: int = 0                  # current baseline trial index
    queue_pos: int = 0                     # current change-queue index
    queue: list = field(default_factory=list)  # List[ChangeTarget]
    anchors: Optional[dict] = None         # live v3 anchor file (seeded)
    eye_roi: Optional[tuple] = None        # full-frame (y0,y1,x0,x1) eye box or None
    speed: float = 1.0                     # playback speed multiplier
    playing: bool = False
    timer: object = None                   # matplotlib canvas timer
    fig: object = None                     # the scrubber's Figure
    scrub_state: Optional[dict] = None     # the scrubber's own state dict
    mouth_roi: Optional[tuple] = None      # full-frame (y0,y1,x0,x1) or None
    sidecar: Optional[dict] = None         # video_labels sidecar (schema v1)
    zoomed: bool = False                   # f-toggle: eye-zoom vs full frame
    arming: Optional[str] = None           # active drag intent: "eye"|"mouth"|"correct"
    last_proposed: Optional[dict] = None   # last detect_pupil ellipse on the shown frame
    overlay: object = None                 # matplotlib Ellipse artist on ax_frame


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Unified single-pass video<->neural sync tagger "
                    "(baseline + change anchors, live cv_rmse).",
    )
    parser.add_argument("--session", required=True,
                        help="Session id (DDMMYYYY / DDMMYY / subject-prefixed).")
    parser.add_argument("--subject", default=None,
                        help="Subject (default: config.SUBJECT).")
    parser.add_argument("--no-stage", action="store_true",
                        help="Read frames over X: instead of staging locally "
                             "(scrubbing is laggy over Samba).")
    args = parser.parse_args(argv)

    subject = args.subject
    subj_display = subject or config.SUBJECT
    session = config.canonical_camera_session(args.session)
    sync_dir = config.subject_video_sync_dir(subject)

    # --- Behavioural session -> baseline_on + change queue. Resolve the PKL by
    #     --subject (NOT the frozen config.SUBJECT env) via the shared library
    #     resolver. Load BEFORE seeding so a not-found session does not archive
    #     the user's prior anchors first.
    pkl_path = resolve_subject_pkl(session, subject)
    if pkl_path is None:
        raise SystemExit(
            f"No PKL for subject {subj_display} session {session} under "
            f"{os.path.join(config.ROOT, 'data', 'pkls', subj_display)} "
            f"(expected {subj_display}_<token>.pkl)."
        )
    sess = _load_session_path(pkl_path)

    # --- Migrate + seed (§5): archive prior anchors, pre-load as editable seeds.
    seed = seed_from_archive(session, subject)  # v3 dict with source='legacy', or None

    baseline_on = np.asarray(sess.ni_events.get("Baseline_ON", []), dtype=float)
    baseline_on = baseline_on[baseline_on > 0]
    n_task_trials = len(sess.trials)
    if n_task_trials > 0 and len(baseline_on) > n_task_trials:
        baseline_on = baseline_on[:n_task_trials]
    if len(baseline_on) == 0:
        logger.error("No Baseline_ON events for %s - aborting.", session)
        return 2
    queue = build_change_queue(sess)  # alignment-safe (hit/miss go-trials only)
    del sess
    gc.collect()

    # --- Camera video: stage locally (default) or read over X: (--no-stage).
    try:
        if args.no_stage:
            cam = find_camera_files(session, subject=subject)
        else:
            cam = stage_session_video(session, subject, cams=("eye_cam",))
    except Exception as exc:  # FileNotFoundError etc.
        logger.error("Could not locate camera files for %s: %s", session, exc)
        return 2
    if "eye_cam" not in cam:
        logger.error("No eye-cam video/metadata pair found for %s.", session)
        return 2
    video_path = cam["eye_cam"]["video"]
    meta_path = cam["eye_cam"]["metadata"]

    ts_ms, _, _ = load_camera_metadata(meta_path)
    fps = 1000.0 / float(np.median(np.diff(ts_ms)))
    n_frames = len(ts_ms)

    coarse_offset_s = _read_coarse_offset(session, sync_dir)
    if coarse_offset_s is None:
        coarse_offset_s = DEFAULT_COARSE_OFFSET_S

    # Real frame dimensions (for the full-frame playback placeholder). Prefer the
    # container props; fall back to decoding one frame if the props report 0.
    probe = cv2.VideoCapture(video_path)
    frame_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    if frame_h <= 0 or frame_w <= 0:
        probe.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, f0 = probe.read()
        if ok and f0 is not None:
            frame_h, frame_w = int(f0.shape[0]), int(f0.shape[1])
    probe.release()
    if frame_h <= 0 or frame_w <= 0:
        logger.error("Could not determine frame dimensions for %s.", video_path)
        return 2

    # --- Per-frame label + ROI sidecar (Plan 2b). Decoupled from the anchor JSON.
    sidecar = vl.load_sidecar(session, subject)
    if sidecar is None:
        sidecar = vl.new_sidecar(subj_display, session, (frame_h, frame_w))
        seeded = vl.seed_rois_from_previous(session, subject, (frame_h, frame_w))
        if seeded is not None and seeded["applied"]:
            for name, r in seeded["rois"].items():
                sidecar["rois"][name] = r
            logger.info("Seeded %d ROI(s) from prior session %s (inherited).",
                        len(seeded["rois"]), seeded["source_session"])
        elif seeded is not None:
            logger.warning(
                "Prior session %s frame_size %s != current %s; ROIs offered but "
                "NOT applied (draw fresh with e/m).",
                seeded["source_session"], seeded["frame_size"], [frame_h, frame_w])
        vl.save_sidecar(sidecar, session, subject)

    tag = TagSessionState(queue=queue, anchors=seed, sidecar=sidecar)

    # Adopt any seeded/loaded ROIs into live state (full-frame pixel boxes).
    _eye = sidecar["rois"].get("eye")
    tag.eye_roi = tuple(_eye["box"]) if _eye else None
    _mouth = sidecar["rois"].get("mouth")
    tag.mouth_roi = tuple(_mouth["box"]) if _mouth else None

    # ---------------------------------------------------------------------
    # Provisional clock models
    # ---------------------------------------------------------------------
    def _baseline_implied_offset() -> float:
        """implied offset (video = nidaq + offset) for baseline jumps."""
        if tag.anchors:
            for a in tag.anchors["anchors"]:
                if a.get("event_type", "baseline_on") == "baseline_on":
                    return _entry_implied_offset(a)
        return -float(coarse_offset_s)  # video ~= nidaq - coarse_offset

    def _provisional_change_clock():
        """(slope, offset) seeding the change jump. Thin GUI wrapper: gathers the
        live anchor list and delegates to the pure, unit-tested
        ``tagging.provisional_change_clock`` (design §8)."""
        anchors = tag.anchors["anchors"] if tag.anchors else []
        return provisional_change_clock(anchors, coarse_offset_s)

    def _change_frame(pos: int) -> int:
        slope, offset = _provisional_change_clock()
        fi = nidaq_to_frame_oriented(
            tag.queue[pos].change_on_s, slope, offset, fps, "manual_multianchor")
        return int(np.clip(fi, 0, n_frames - 1))

    # ---------------------------------------------------------------------
    # Frame reader for the playback timer (the scrubber owns its own cap and
    # exposes no per-tick redraw hook, so playback needs its own capture). The
    # tagger is full-frame in Plan 2a (cfg.crop is always None), so this mirrors
    # _tagger_ui._read_frame's full-frame path.
    # ---------------------------------------------------------------------
    play_cap = cv2.VideoCapture(video_path)

    def _read_play_frame(fi: int) -> np.ndarray:
        play_cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = play_cap.read()
        if not ok or frame is None:
            return np.zeros((frame_h, frame_w), dtype=np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------------------
    # Pupil detection + proposed-ellipse overlay (Plan 2b). Detection runs on
    # the full-frame grayscale reader restricted to the eye ROI, so the cached
    # ellipse is in full-frame pixel coords; _update_overlay subtracts the crop
    # origin when zoomed so it stays visible (and correct) in BOTH views.
    # Detection is skipped during playback streaming for responsiveness.
    # ---------------------------------------------------------------------
    def _run_detect(fi: int):
        """Detect the pupil in the eye ROI on frame *fi*; cache the proposal."""
        if tag.eye_roi is None:
            tag.last_proposed = None
            return
        gray = _read_play_frame(fi)                       # full-frame grayscale
        det = detect_pupil_in_frame(gray, search_roi=tag.eye_roi)
        tag.last_proposed = vl.ellipse_from_detection(det)  # {cx,cy,major,minor,angle}|None

    def _update_overlay():
        """Draw/refresh the proposed-ellipse patch on the scrubber's frame axis.

        The cached ellipse is in FULL-FRAME pixel coords, but whenever a crop is
        active the axes show CROP-LOCAL coords. Subtract the crop origin rather
        than hiding the overlay: the zoom exists precisely so the user can judge
        this ellipse closely, and Task 6 confirms/corrects it from the same view,
        so hiding it while zoomed would defeat the feature.
        """
        fig = tag.fig
        if fig is None or not fig.axes:
            return
        ax = fig.axes[0]
        ell = tag.last_proposed
        show = ell is not None
        if tag.overlay is None:
            tag.overlay = Ellipse((0.0, 0.0), 1.0, 1.0, angle=0.0, fill=False,
                                  edgecolor="#00ff00", linewidth=1.5)
            ax.add_patch(tag.overlay)
        if show:
            cx, cy = float(ell["cx"]), float(ell["cy"])
            crop = cfg.crop          # read live; `f` mutates it
            if crop is not None:
                cy -= int(crop[0])
                cx -= int(crop[2])
            tag.overlay.set_center((cx, cy))
            tag.overlay.width = ell["major"]
            tag.overlay.height = ell["minor"]
            tag.overlay.angle = ell["angle"]
        tag.overlay.set_visible(show)

    def _on_frame_shown(fi: int, fig) -> None:
        """cfg.on_refresh hook: re-detect + redraw the overlay on every manual
        frame change (arrow step / jump / mode toggle / ROI draw)."""
        tag.fig = fig
        _run_detect(fi)
        _update_overlay()

    def _draw_current(fi: int):
        fig = tag.fig
        if fig is None or not plt.fignum_exists(getattr(fig, "number", -1)):
            return
        try:
            im = fig.axes[0].images[0]
            hud = fig.axes[1].texts[0]
        except (IndexError, AttributeError):
            return
        im.set_data(_read_play_frame(fi))
        hud.set_text(_hud_fn(fi))
        # Playback streams frames without per-frame detection; hide any stale
        # overlay. It reappears on the next manual step via _on_frame_shown.
        if tag.overlay is not None:
            tag.overlay.set_visible(False)
        fig.canvas.draw_idle()

    # ---------------------------------------------------------------------
    # Playback timer
    # ---------------------------------------------------------------------
    def _play_interval_ms() -> int:
        return max(10, int(round(1000.0 / (fps * tag.speed))))

    def _stop_play():
        tag.playing = False
        if tag.timer is not None:
            tag.timer.stop()

    def _tick():
        if not tag.playing:
            return
        fig = tag.fig
        if fig is None or not plt.fignum_exists(getattr(fig, "number", -1)):
            _stop_play()
            return
        fi = min(n_frames - 1, tag.scrub_state["frame_idx"] + 1)
        tag.scrub_state["frame_idx"] = fi
        _draw_current(fi)
        if fi >= n_frames - 1:
            _stop_play()

    def _toggle_play(event) -> bool:
        if tag.timer is None:
            tag.timer = event.canvas.new_timer(interval=_play_interval_ms())
            tag.timer.add_callback(_tick)
        if tag.playing:
            _stop_play()
        else:
            tag.playing = True
            tag.timer.stop()
            tag.timer.interval = _play_interval_ms()
            tag.timer.start()
        return True

    def _apply_speed():
        if tag.playing and tag.timer is not None:
            tag.timer.stop()
            tag.timer.interval = _play_interval_ms()
            tag.timer.start()

    # ---------------------------------------------------------------------
    # Target navigation
    # ---------------------------------------------------------------------
    def _reseed_target(state):
        if tag.mode == "change":
            if tag.queue:
                state["frame_idx"] = _change_frame(tag.queue_pos)
        else:
            # compute_predicted_frame_idx maps NI time -> frame as
            # (nidaq - coarse_offset); our baseline implied offset is
            # (video - nidaq), so pass its negation as the coarse offset.
            state["frame_idx"] = compute_predicted_frame_idx(
                float(baseline_on[tag.baseline_pos]),
                -_baseline_implied_offset(), ts_ms)

    def _toggle_mode(state) -> bool:
        tag.mode = "change" if tag.mode == "baseline" else "baseline"
        _reseed_target(state)
        return True

    def _step_target(state, delta: int) -> bool:
        if tag.mode == "baseline":
            tag.baseline_pos = int(
                np.clip(tag.baseline_pos + delta, 0, len(baseline_on) - 1))
        elif tag.queue:
            tag.queue_pos = int(
                np.clip(tag.queue_pos + delta, 0, len(tag.queue) - 1))
        _reseed_target(state)
        return True

    def _goto_end(state, first: bool) -> bool:
        if tag.mode == "baseline":
            tag.baseline_pos = 0 if first else len(baseline_on) - 1
        elif tag.queue:
            tag.queue_pos = 0 if first else len(tag.queue) - 1
        _reseed_target(state)
        return True

    # ---------------------------------------------------------------------
    # Save / delete
    # ---------------------------------------------------------------------
    def _current_key():
        """(trial_index, event_type) of the current mode's target."""
        if tag.mode == "change":
            if not tag.queue:
                return None
            return (int(tag.queue[tag.queue_pos].trial_index), "change_on")
        return (int(tag.baseline_pos), "baseline_on")

    def _do_save(frame_idx: int) -> Optional[dict]:
        """Build + merge + persist the anchor for the current mode/target."""
        if tag.mode == "change":
            if not tag.queue:
                logger.warning("No change targets; nothing to save.")
                return tag.anchors
            tgt = tag.queue[tag.queue_pos]
            entry = _build_change_anchor_entry(
                tgt.change_on_s, ts_ms, tgt.trial_index, frame_idx,
                tgt.change_size, tgt.outcome)
        else:
            raw = _build_anchor_entry(baseline_on, ts_ms, tag.baseline_pos, frame_idx)
            # Normalise to v3 (adds event_type + nidaq_event_s) via the canonical
            # builder rather than hand-setting keys.
            entry = _build_v3_anchor_file(session, fps, len(baseline_on), [raw])["anchors"][0]

        if tag.anchors is None:
            merged = _build_v3_anchor_file(session, fps, len(baseline_on), [entry])
        else:
            merged = _merge_anchor_into_file(tag.anchors, entry)
        save_anchor(session, merged, sync_dir=sync_dir)
        tag.anchors = merged
        logger.info("Saved %s anchor (trial %s) at frame %d -> %d anchors total.",
                    tag.mode, entry["trial_index"], frame_idx, len(merged["anchors"]))
        return merged

    def _save_keepopen(state) -> bool:
        # Save WITHOUT closing so the user keeps tagging (multi-anchor flow).
        _do_save(state["frame_idx"])
        return True

    def _delete_current() -> bool:
        if tag.anchors is None:
            return True
        key = _current_key()
        if key is None:
            return True
        kept = [a for a in tag.anchors["anchors"]
                if (int(a["trial_index"]), a.get("event_type", "baseline_on")) != key]
        if len(kept) != len(tag.anchors["anchors"]):
            merged = dict(tag.anchors)
            merged["anchors"] = kept
            save_anchor(session, merged, sync_dir=sync_dir)
            tag.anchors = merged
            logger.info("Deleted %s anchor for trial %s (%d remain).",
                        key[1], key[0], len(kept))
        return True

    # ---------------------------------------------------------------------
    # HUD
    # ---------------------------------------------------------------------
    def _hud_fn(fi: int) -> str:
        video_s = float(ts_ms[fi] / 1000.0)
        if tag.mode == "change":
            if tag.queue:
                tgt = tag.queue[tag.queue_pos]
                mode_line = (f"{subj_display}  {session}   MODE: CHANGE "
                             f"(size{tgt.change_size:g}, {tgt.outcome})")
                pos, ntot, trial_no = tag.queue_pos + 1, len(tag.queue), tgt.trial_index
                pred = _change_frame(tag.queue_pos)
            else:
                mode_line = (f"{subj_display}  {session}   "
                             f"MODE: CHANGE (no big-change targets)")
                pos, ntot, trial_no, pred = 0, 0, -1, fi
        else:
            mode_line = f"{subj_display}  {session}   MODE: BASELINE"
            pos, ntot, trial_no = tag.baseline_pos + 1, len(baseline_on), tag.baseline_pos
            pred = compute_predicted_frame_idx(
                float(baseline_on[tag.baseline_pos]),
                -_baseline_implied_offset(), ts_ms)
        delta = fi - pred

        entries = tag.anchors["anchors"] if tag.anchors else []
        n_chg = sum(1 for a in entries
                    if a.get("event_type", "baseline_on") == "change_on")
        n_base = len(entries) - n_chg
        if len(entries) >= 3:
            try:
                sync = fit_multianchor_clock(entries, len(baseline_on))
                qc = f"cv_rmse: {sync.cv_rmse_ms:.1f} ms  {sync.quality}"
            except ValueError:
                qc = "cv_rmse: fit failed (degenerate anchors)"
        else:
            qc = "cv_rmse: need >=3 anchors"

        eye_state = "set" if tag.eye_roi is not None else "none"
        mouth_state = "set" if tag.mouth_roi is not None else "none"
        view = "ZOOM" if tag.zoomed else "full"
        roi_line = f"ROI: eye[{eye_state}] mouth[{mouth_state}]   view: {view}"

        legend = ("[space]play  [-/+]spd {:g}x  [j/k]jump  [c]base<->chg  "
                  "[e/m]roi  [f]zoom  [enter]save  [d]del  [q]quit"
                  ).format(tag.speed)
        return "\n".join([
            mode_line,
            (f"trial {pos}/{ntot} (idx {trial_no})   frame {fi} ({video_s:.2f}s)"
             f"   Delta {delta:+d} vs pred"),
            f"anchors: {len(entries)} ({n_base} base / {n_chg} chg)     {qc}",
            roi_line,
            legend,
        ])

    # ---------------------------------------------------------------------
    # ROI capture + eye-zoom (Plan 2b)
    # ---------------------------------------------------------------------
    def _toggle_zoom() -> bool:
        if tag.eye_roi is None:
            logger.warning("No eye ROI yet; draw one with 'e' before zooming.")
            return True
        if not tag.zoomed:
            raw = eye_zoom_crop(tag.eye_roi)              # UNCLAMPED (y0,y1,x0,x1)
            crop = vl.clamp_crop(raw, frame_h, frame_w)  # None if it misses the frame
            if crop is None:                             # no valid crop -> stay full-frame
                logger.warning("Eye ROI does not intersect the frame; staying on "
                               "the full view (no zoom).")
                return True                              # do NOT toggle into a broken zoom
            tag.zoomed = True
            cfg.crop = crop                              # guaranteed non-empty crop
        else:
            tag.zoomed = False
            cfg.crop = None                              # back to full frame
        return True

    def _on_roi_drawn(box, state) -> None:
        """cfg.on_selector: a completed drag sets the armed ROI (eye/mouth) or,
        in Task 6, a pupil correction. Boxes are full-frame (y0,y1,x0,x1)."""
        if tag.arming == "eye":
            tag.eye_roi = tuple(box)
            vl.set_roi(tag.sidecar, "eye", box, source="drawn")
            vl.save_sidecar(tag.sidecar, session, subject)
            _run_detect(state["frame_idx"])              # immediate live feedback
            _update_overlay()
        elif tag.arming == "mouth":
            tag.mouth_roi = tuple(box)
            vl.set_roi(tag.sidecar, "mouth", box, source="drawn")
            vl.save_sidecar(tag.sidecar, session, subject)
        # (Task 6 adds the tag.arming == "correct" branch here.)
        tag.arming = None

    # ---------------------------------------------------------------------
    # Key dispatch
    # ---------------------------------------------------------------------
    def _on_key_extra(event, state) -> bool:
        tag.scrub_state = state
        tag.fig = event.canvas.figure
        key = event.key
        if key == " ":
            return _toggle_play(event)
        # Speed keys. Primary is -/+ because the HUD marks keys as [key], so a
        # legend reading '['/']' was misread as the "/" key during the A1 pilot
        # (the brackets were never actually pressed). "+" needs shift on most
        # layouts, so "=" is accepted too; the original brackets stay as aliases,
        # incl. the Tk keysym spellings some backends deliver instead of "["/"]".
        if key in ("-", "[", "bracketleft"):
            tag.speed = max(0.25, tag.speed / 2.0)
            _apply_speed()
            return True
        if key in ("+", "=", "]", "bracketright"):
            tag.speed = min(8.0, tag.speed * 2.0)
            _apply_speed()
            return True
        if key == "c":
            return _toggle_mode(state)
        if key == "j":
            return _step_target(state, +1)
        if key == "k":
            return _step_target(state, -1)
        if key == "home":
            return _goto_end(state, first=True)
        if key == "end":
            return _goto_end(state, first=False)
        if key == "d":
            return _delete_current()
        if key in ("enter", "s"):
            # enter = design-primary save that KEEPS the window open (the shared
            # scrubber now routes enter through this hook before its default
            # save-and-close); 's' is a documented alias (keymap.save is cleared).
            return _save_keepopen(state)
        # ROI capture (Plan 2b). ROIs are only drawable in the full-frame view,
        # where imshow data coords equal full-frame pixels.
        if key == "e":
            if tag.zoomed:
                logger.warning("Return to full frame (press f) before drawing an ROI.")
                return True
            tag.arming = "eye"
            state["arm_selector"]()
            return True
        if key == "m":
            if tag.zoomed:
                logger.warning("Return to full frame (press f) before drawing an ROI.")
                return True
            tag.arming = "mouth"
            state["arm_selector"]()
            return True
        if key == "f":
            return _toggle_zoom()
        # Self-diagnosing: log (cheap, debug-level, not to the HUD) any key we do
        # not handle so a future dead-key report (like the bracket keys above) is
        # immediately diagnosable from the log instead of a silent no-op.
        logger.debug("unhandled key: %r", key)
        return False

    # ---------------------------------------------------------------------
    # Compose the scrubber. Starts FULL FRAME (crop=None); Plan 2b's `f` toggle
    # mutates cfg.crop live to a CLAMPED eye-zoom derived from the user's eye ROI
    # (never the BG_046-specific absolute-pixel fallback, which lands on the snout
    # on closer-camera subjects). ROI drag + pupil overlay ride the on_selector /
    # on_refresh seams (see the module docstring + the UX design doc).
    # ---------------------------------------------------------------------
    start_frame = compute_predicted_frame_idx(
        float(baseline_on[0]), -_baseline_implied_offset(), ts_ms)
    cfg = ScrubberConfig(
        video_path=video_path,
        ts_ms=ts_ms,
        fps=fps,
        n_frames=n_frames,
        start_frame=start_frame,
        crop=None,               # full-frame default
        hud_fn=_hud_fn,
        on_key_extra=_on_key_extra,
        on_save=_do_save,        # enter falls back here only if the hook declines
        on_selector=_on_roi_drawn,   # Task 5: eye/mouth ROI drag
        on_refresh=_on_frame_shown,  # Task 5: re-detect + redraw pupil overlay
    )

    logger.info("Tagging %s / %s: %d baseline trials, %d change targets, %d frames.",
                subj_display, session, len(baseline_on), len(queue), n_frames)
    try:
        run_scrubber(cfg)
    finally:
        play_cap.release()

    anchor_path = os.path.join(sync_dir, f"{session}_anchor.json")
    if os.path.exists(anchor_path):
        print(f"Anchor: {anchor_path}")
        print(f"Next:   py scripts/video/fit_sync.py --subject {subj_display} "
              f"--session {session}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
