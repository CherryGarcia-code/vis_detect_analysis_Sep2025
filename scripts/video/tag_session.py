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
  space                   play/pause forward (SEQUENTIAL decode; speed skips frames)
  [ / ]   (or -/+)        slower / faster playback (speed>1 advances >1 frame/tick)
  , / .                   lower / raise the pupil dark-pixel percentile (bigger blob)
  j / k                   next / prev target onset (baseline trial OR change queue)
  c                       toggle baseline <-> change target mode
  e / m                   drag the eye / mouth ROI (full-frame view only); each is
                          drawn as a persistent rectangle (eye=cyan, mouth=orange)
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
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402  (pupil + ROI overlays)

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
    eye_rect: object = None                # matplotlib Rectangle artist: eye ROI
    mouth_rect: object = None              # matplotlib Rectangle artist: mouth ROI
    dark_percentile: float = 8.0           # live detect_pupil dark-pixel percentile
    play_next_fi: Optional[int] = None     # frame index play_cap will read NEXT
                                           # (None = position unknown -> force seek)


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

    # Seed the live pupil dark-pixel percentile from the sidecar (a threshold the
    # human tuned on a prior pass) if present, else the detector default. Pilot
    # FIX 2: an under-inclusive threshold shrinks the proposed ellipse and biases
    # pupil diameter DOWNWARD, so this value is recorded per session.
    tag.dark_percentile = vl.get_pupil_dark_percentile(sidecar)

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
    # exposes no per-tick redraw hook, so playback needs its own capture).
    #
    # Pilot FIX 1: playback used to random-SEEK every frame
    # (set(CAP_PROP_POS_FRAMES, fi); read()). On this 22 GB H.264 file each seek
    # re-decodes from the nearest keyframe, so a tick cost FAR more than the timer
    # interval — playback crawled and the speed keys had no visible effect (the
    # interval never dominated). We now read SEQUENTIALLY: cv2 keeps decoder state
    # between successive read()s, so the common "advance by 1" case reuses it and
    # is many times cheaper. We re-seek ONLY on a discontinuity, tracked via
    # tag.play_next_fi (the index play_cap will return NEXT). For speed>1 we skip
    # the intermediate frames with the decode-light grab() (no BGR convert / copy)
    # and only decode the one we display. play_cap is the SOLE writer of
    # play_next_fi, so the tracked position never drifts from reality.
    # ---------------------------------------------------------------------
    play_cap = cv2.VideoCapture(video_path)
    _MAX_GRAB_SKIP = 30  # forward gaps up to this advance via grab(); larger -> seek

    def _read_seq_full(fi: int) -> Optional[np.ndarray]:
        """Full-frame grayscale for frame *fi*, read SEQUENTIALLY where possible.

        Re-seeks only when play_cap is not already positioned to return *fi*; a
        small forward gap is closed with cheap grab()s (frame-skip for speed>1).
        Updates tag.play_next_fi to the frame play_cap will read next. Returns
        ``None`` on a failed/absent read (callers substitute a placeholder).
        """
        fi = int(fi)
        if tag.play_next_fi != fi:
            gap = (fi - tag.play_next_fi) if tag.play_next_fi is not None else None
            if gap is not None and 0 < gap <= _MAX_GRAB_SKIP:
                for _ in range(gap):          # skip intermediate frames cheaply
                    play_cap.grab()
            else:                             # backward / large / unknown -> seek
                play_cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = play_cap.read()
        tag.play_next_fi = fi + 1
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _apply_crop(gray: Optional[np.ndarray]) -> np.ndarray:
        """Apply the LIVE cfg.crop to a full-frame array so the displayed image and
        the axes extent (also derived from cfg.crop) always agree.

        Pilot FIX 4: the playback path used to blit a FULL-frame array into a
        crop-sized extent when zoomed, stretching/misaligning it. Cropping here —
        exactly as _tagger_ui._read_frame does — keeps the two in lockstep in both
        views. None -> a correctly-shaped black placeholder for the current view.
        """
        if gray is None:
            if cfg.crop is not None:
                y0, y1, x0, x1 = cfg.crop
                return np.zeros((max(1, y1 - y0), max(1, x1 - x0)), dtype=np.uint8)
            return np.zeros((frame_h, frame_w), dtype=np.uint8)
        if cfg.crop is not None:
            y0, y1, x0, x1 = cfg.crop
            return gray[y0:y1, x0:x1]
        return gray

    # ---------------------------------------------------------------------
    # Pupil detection + proposed-ellipse overlay (Plan 2b). Detection runs on
    # the full-frame grayscale reader restricted to the eye ROI, so the cached
    # ellipse is in full-frame pixel coords; the frame axes live in full-frame
    # coords in BOTH views (the scrubber re-derives the imshow extent from
    # cfg.crop on every refresh), so _update_overlay places the ellipse at its
    # true cx/cy with no crop-origin fudge and keeps it visible while zoomed.
    # Detection is skipped during playback streaming for responsiveness.
    # ---------------------------------------------------------------------
    def _run_detect(fi: int):
        """Detect the pupil in the eye ROI on frame *fi*; cache the proposal.

        Uses the LIVE tag.dark_percentile (pilot FIX 2): a higher percentile
        admits more dark pixels -> a larger blob, so the human can widen an
        under-inclusive proposal with `,`/`.` and the label records the exact
        threshold it was judged against.
        """
        if tag.eye_roi is None:
            tag.last_proposed = None
            return
        gray = _read_seq_full(fi)                         # full-frame grayscale
        if gray is None:
            tag.last_proposed = None
            return
        det = detect_pupil_in_frame(gray, search_roi=tag.eye_roi,
                                    dark_percentile=tag.dark_percentile)
        tag.last_proposed = vl.ellipse_from_detection(det)  # {cx,cy,major,minor,angle}|None

    def _update_overlay():
        """Draw/refresh the proposed-ellipse patch on the scrubber's frame axis.

        The cached ellipse is in FULL-FRAME pixel coords, and the frame axes now
        ALWAYS live in full-frame data coords (the scrubber re-derives the imshow
        extent + limits from cfg.crop on every refresh), so the ellipse is placed
        at its true full-frame cx/cy in BOTH views — no crop-origin fudge. It
        stays VISIBLE while zoomed, which is the whole point of the zoom: the user
        judges (and, via `p`, corrects) the pupil close-up from this same overlay.
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
            tag.overlay.set_center((float(ell["cx"]), float(ell["cy"])))
            tag.overlay.width = ell["major"]
            tag.overlay.height = ell["minor"]
            tag.overlay.angle = ell["angle"]
        tag.overlay.set_visible(show)

    def _update_rois():
        """Draw/refresh a persistent rectangle for each ROI on the frame axis.

        Pilot FIX 3: the ROI drags WORK (the sidecar stores the boxes) but nothing
        was ever drawn, so the mouth ROI looked like a no-op and the eye ROI only
        *seemed* to register because the pupil ellipse appears inside it. The frame
        axes live in FULL-FRAME data coords in BOTH views (the scrubber re-derives
        the extent from cfg.crop every redraw), so each box draws at its stored
        coords directly — NO crop-origin math. Colours are distinct from the green
        pupil ellipse: eye=cyan, mouth=orange. An unset ROI is hidden; in the
        eye-zoom the mouth rectangle simply falls outside the axis limits.
        """
        fig = tag.fig
        if fig is None or not fig.axes:
            return
        ax = fig.axes[0]
        if tag.eye_rect is None:
            tag.eye_rect = Rectangle((0.0, 0.0), 0.0, 0.0, fill=False,
                                     edgecolor="#00bfff", linewidth=1.5)
            ax.add_patch(tag.eye_rect)
        if tag.mouth_rect is None:
            tag.mouth_rect = Rectangle((0.0, 0.0), 0.0, 0.0, fill=False,
                                       edgecolor="#ff8c00", linewidth=1.5)
            ax.add_patch(tag.mouth_rect)
        for box, rect in ((tag.eye_roi, tag.eye_rect),
                          (tag.mouth_roi, tag.mouth_rect)):
            if box is None:
                rect.set_visible(False)
                continue
            y0, y1, x0, x1 = box
            rect.set_xy((float(x0), float(y0)))         # full-frame data coords
            rect.set_width(float(x1 - x0))
            rect.set_height(float(y1 - y0))
            rect.set_visible(True)

    def _on_frame_shown(fi: int, fig) -> None:
        """cfg.on_refresh hook: re-detect + redraw the overlay AND ROI rectangles
        on every manual frame change (arrow step / jump / mode toggle / ROI draw /
        zoom toggle)."""
        tag.fig = fig
        _run_detect(fi)
        _update_overlay()
        _update_rois()

    def _draw_current(fi: int):
        fig = tag.fig
        if fig is None or not plt.fignum_exists(getattr(fig, "number", -1)):
            return
        try:
            im = fig.axes[0].images[0]
            hud = fig.axes[1].texts[0]
        except (IndexError, AttributeError):
            return
        # Pilot FIX 4: crop the played frame to match the LIVE cfg.crop (and thus
        # the axes extent set by the scrubber's _refresh), so a zoomed playback is
        # never stretched/misaligned. _apply_crop mirrors _tagger_ui._read_frame.
        im.set_data(_apply_crop(_read_seq_full(fi)))
        hud.set_text(_hud_fn(fi))
        # Playback streams frames without per-frame detection; hide any stale
        # pupil ellipse. It reappears on the next manual step via _on_frame_shown.
        # The ROI rectangles are static overlays and stay visible while playing.
        if tag.overlay is not None:
            tag.overlay.set_visible(False)
        fig.canvas.draw_idle()

    # ---------------------------------------------------------------------
    # Playback timer
    #
    # Pilot FIX 1: speed is now REAL. For speed>=1 we advance ``_play_step()``
    # frames per tick (skipping the intermediate ones via grab() in _read_seq_full)
    # and keep the tick interval near real-time, so 2x/4x advance 2/4 as many
    # frames per second — a clearly different rate. For speed<1 we keep 1
    # frame/tick and LENGTHEN the interval (the original behaviour). Because
    # decode can still dominate on this file, the interval is derived so that
    # frames-advanced-per-second targets fps*speed as closely as the decoder
    # allows; if decode is the bottleneck the rate is capped there, but the
    # frame-skip guarantees higher speeds visibly move faster.
    # ---------------------------------------------------------------------
    def _play_step() -> int:
        return max(1, int(round(tag.speed))) if tag.speed >= 1.0 else 1

    def _play_interval_ms() -> int:
        # Interval so that step frames / interval ~= fps*speed frames per second.
        step = _play_step()
        return max(10, int(round(1000.0 * step / (fps * tag.speed))))

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
        cur = tag.scrub_state["frame_idx"]
        if cur >= n_frames - 1:                       # already at the last frame
            _stop_play()
            return
        fi = min(n_frames - 1, cur + _play_step())    # advance by the speed step
        tag.scrub_state["frame_idx"] = fi
        _draw_current(fi)                             # skips fi-1 intermediates
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

    def _persist_dark_percentile():
        """Record the live pupil dark-pixel percentile in the sidecar (atomic).

        Pilot FIX 2: the threshold the human tuned the proposal against must be
        reproducible for sub-project C, so it is written on every nudge. The
        subsequent scrubber _refresh re-runs detection at the NEW value (via
        _on_frame_shown -> _run_detect), so the ellipse updates immediately.
        """
        vl.set_pupil_dark_percentile(tag.sidecar, tag.dark_percentile)
        vl.save_sidecar(tag.sidecar, session, subject)

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
        roi_line = (f"ROI: eye[{eye_state}] mouth[{mouth_state}]   view: {view}"
                    f"   pupil%: {tag.dark_percentile:g}")

        # Per-frame label tally for the session (Plan 2b, Task 6). Counts come
        # straight from the sidecar's frame-keyed upserts, so re-labelled frames
        # are counted once (never double-counted).
        _frames = tag.sidecar["frames"] if tag.sidecar else []
        n_conf = sum(1 for f in _frames if f["verdict"] == vl.VERDICT_CONFIRMED)
        n_corr = sum(1 for f in _frames if f["verdict"] == vl.VERDICT_CORRECTED)
        n_blink = sum(1 for f in _frames if f["verdict"] == vl.VERDICT_BLINK)
        label_line = f"labels: {n_conf} ok / {n_corr} fix / {n_blink} blink"

        legend = ("[space]play  [-/+]spd {:g}x  [,/.]pupil%  [j/k]jump  "
                  "[c]base<->chg  [e/m]roi  [f]zoom  [u]ok  [p]fix  [x]blink  "
                  "[enter]save  [d]del  [q]quit"
                  ).format(tag.speed)
        return "\n".join([
            mode_line,
            (f"trial {pos}/{ntot} (idx {trial_no})   frame {fi} ({video_s:.2f}s)"
             f"   Delta {delta:+d} vs pred"),
            f"anchors: {len(entries)} ({n_base} base / {n_chg} chg)     {qc}",
            roi_line,
            label_line,
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
        elif tag.arming == "correct":
            # The proposal was wrong: store BOTH the detector's proposal (may be
            # None -- the "miss" failure mode) AND the human's inscribed ellipse,
            # so proposed-vs-corrected is directly comparable (that is how the
            # eyelid-occlusion diameter bias is quantified downstream). Upsert on
            # frame_idx; persist atomically.
            corrected = vl.ellipse_from_box(box)
            vl.upsert_frame_label(tag.sidecar, state["frame_idx"],
                                  vl.VERDICT_CORRECTED,
                                  proposed_ellipse=tag.last_proposed,   # may be None
                                  corrected_ellipse=corrected)
            vl.save_sidecar(tag.sidecar, session, subject)
            logger.info("corrected frame %d", state["frame_idx"])
        tag.arming = None

    def _on_selector_cancel(state) -> None:
        """cfg.on_selector_cancel: a drag was armed (e/m/p) then CANCELLED (empty
        drag, or q/esc while armed) instead of completed. Drop the arming intent
        so a stale 'eye'/'mouth'/'correct' cannot mis-route the next completed
        drag. (A completed drag clears tag.arming itself in _on_roi_drawn.)"""
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
        # Pupil dark-pixel percentile tuning (pilot FIX 2). Chosen keys ','/'.'
        # (with '<'/'>' shift-aliases) collide with NO existing binding and no
        # live matplotlib default. A HIGHER percentile admits more dark pixels ->
        # a LARGER proposed blob (the pilot's ellipse was too small). Clamp
        # 1.0-40.0, step 1.0. Returning True makes the scrubber _refresh, which
        # re-detects at the new value and redraws the ellipse.
        if key in (",", "<"):
            tag.dark_percentile = max(1.0, round(tag.dark_percentile - 1.0, 3))
            _persist_dark_percentile()
            return True
        if key in (".", ">"):
            tag.dark_percentile = min(40.0, round(tag.dark_percentile + 1.0, 3))
            _persist_dark_percentile()
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
        # Per-frame pupil labels (Plan 2b, Task 6). All three upsert on frame_idx
        # (re-labelling a frame REPLACES its entry) and persist atomically after
        # every label via save_sidecar (temp + os.replace), so a crash never
        # costs more than the un-saved current keystroke.
        if key == "u":  # proposed ellipse is CORRECT
            # A confirmation must reference a real proposal: if the detector
            # returned nothing (no eye ROI, or no pupil found), do NOT invent a
            # null-proposal "confirmation" -- say so and no-op instead.
            if tag.last_proposed is None:
                logger.warning("No proposed ellipse to confirm "
                               "(set the eye ROI with 'e' and land on a frame "
                               "with a detected pupil).")
                return True
            vl.upsert_frame_label(tag.sidecar, state["frame_idx"],
                                  vl.VERDICT_CONFIRMED,
                                  proposed_ellipse=tag.last_proposed)
            vl.save_sidecar(tag.sidecar, session, subject)
            logger.info("confirmed frame %d", state["frame_idx"])
            return True
        if key == "p":  # proposal is WRONG -> drag the true pupil
            # Reuse the SAME correction-drag path Task 5 wired: arm the shared
            # selector and let _on_roi_drawn's "correct" branch record it. The
            # seam returns FULL-FRAME coords in EITHER view because the frame
            # axes THEMSELVES live in full-frame data coords (run_scrubber
            # re-derives the imshow extent from image_extent_for_crop on every
            # redraw) -- NO crop-origin add-back is applied anywhere. Do not
            # "restore" one: an origin offset cannot undo the coordinate stretch
            # a frozen extent imposes, which is what corrupted zoomed
            # corrections before 1d8866b. So correcting while zoomed is fully
            # supported and is the expected workflow (judge the pupil close-up).
            tag.arming = "correct"
            state["arm_selector"]()
            return True
        if key == "x":  # blink / occluded -> no valid pupil this frame
            # Store the proposal if one exists (it is then a detector false
            # positive); a blink never carries a corrected ellipse.
            vl.upsert_frame_label(tag.sidecar, state["frame_idx"],
                                  vl.VERDICT_BLINK,
                                  proposed_ellipse=tag.last_proposed)
            vl.save_sidecar(tag.sidecar, session, subject)
            logger.info("blink frame %d", state["frame_idx"])
            return True
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
        on_selector_cancel=_on_selector_cancel,  # drop arming intent on cancel
    )

    logger.info("Tagging %s / %s: %d baseline trials, %d change targets, %d frames.",
                subj_display, session, len(baseline_on), len(queue), n_frames)
    try:
        run_scrubber(cfg)
    finally:
        play_cap.release()
        # Belt-and-suspenders flush: every label already saved atomically on
        # keystroke, but re-persist the final sidecar state on quit so a session's
        # labels are durable even if some future path mutates without saving.
        if tag.sidecar is not None:
            vl.save_sidecar(tag.sidecar, session, subject)

    anchor_path = os.path.join(sync_dir, f"{session}_anchor.json")
    if os.path.exists(anchor_path):
        print(f"Anchor: {anchor_path}")
        print(f"Next:   py scripts/video/fit_sync.py --subject {subj_display} "
              f"--session {session}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
