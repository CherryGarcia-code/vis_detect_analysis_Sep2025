"""tag_session.py — unified single-pass video<->neural sync tagger (Plan 2a).

One window that lets the user place BOTH baseline and change anchors for a
session's eye-cam video, with live sync-quality feedback, then run
``fit_sync`` to write the ``manual_multianchor`` sync JSON. This entry point is
*thin*: it COMPOSES already-built primitives and does not reimplement any of
them.

Composed primitives
--------------------
* ``visdetect.analysis.tagging`` — ``build_change_queue`` (Task 1),
  ``seed_from_archive`` (Task 3), ``eye_zoom_crop`` (Task 5),
  ``nidaq_to_frame_oriented`` (Task 4), ``ChangeTarget``.
* ``scripts/video/_tagger_ui.py`` — ``run_scrubber(cfg)`` / ``ScrubberConfig``
  (Task 6): the shared keyboard scrubber + HUD core.
* ``visdetect.core.video_sync`` — ``stage_session_video``, ``find_camera_files``,
  ``load_camera_metadata``, ``compute_predicted_frame_idx`` (baseline jump math),
  ``fit_multianchor_clock`` (live cv_rmse), the v3 anchor writers
  (``_build_anchor_entry``, ``_build_change_anchor_entry``,
  ``_build_v3_anchor_file``, ``_merge_anchor_into_file``), ``save_anchor``.
* ``visdetect.analysis.config`` — ``canonical_camera_session``,
  ``subject_video_sync_dir``, ``ROOT``, ``SUBJECT``.
* ``visdetect.suite.loader.list_pkl_sessions`` +
  ``visdetect.core.session.load_session`` — subject-aware behavioural PKL load
  (by ``--subject``, not the frozen ``config.SUBJECT`` env).

Keybindings (see docs/superpowers/specs/2026-07-23-camera-tagger-ux-design.md)
------------------------------------------------------------------------------
  arrows / shift / ctrl   step +/-1 / +/-10 / +/-100 frames (built into scrubber)
  space                   play/pause forward
  [ / ]                   slower / faster playback
  f                       toggle full-frame <-> eye-zoom view
  j / k                   next / prev target onset (baseline trial OR change queue)
  c                       toggle baseline <-> change target mode
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

# --- Matplotlib backend selection ------------------------------------------
# The scrubber primitive (_tagger_ui) deliberately does NOT force a backend;
# the calling tool picks it. We want TkAgg for the interactive window, but the
# headless verifications (--help / spec-import) run under MPLBACKEND=Agg and
# must not require a display. So: force TkAgg only when NOT explicitly headless.
import matplotlib
if os.environ.get("MPLBACKEND", "").lower() != "agg":
    matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)

# The shared scrubber sits in this same directory; make it importable whether
# run as a script or loaded via importlib.spec_from_file_location.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tagger_ui import ScrubberConfig, run_scrubber  # noqa: E402

# NOTE: we intentionally do NOT `from click_anchor import ...`; click_anchor
# forces TkAgg at import time, which would break the headless spec-import.
# Baseline predicted-frame math comes from the library
# (video_sync.compute_predicted_frame_idx), not a local reimplementation.

# Behavioural PKL is loaded by SUBJECT (not the frozen config.SUBJECT env): we
# resolve the subject's pkl path ourselves (list_pkl_sessions convention) and
# load via the PATH-based core loader.
from visdetect.suite.loader import list_pkl_sessions  # noqa: E402
from visdetect.core.session import load_session as _load_session_path  # noqa: E402
from visdetect.analysis import config  # noqa: E402
from visdetect.analysis.tagging import (  # noqa: E402
    build_change_queue,
    seed_from_archive,
    eye_zoom_crop,
    nidaq_to_frame_oriented,
    provisional_change_clock,
)
from visdetect.core.video_sync import (  # noqa: E402
    find_camera_files,
    load_camera_metadata,
    stage_session_video,
    compute_predicted_frame_idx,
    fit_multianchor_clock,
    save_anchor,
    _build_anchor_entry,
    _build_change_anchor_entry,
    _build_v3_anchor_file,
    _merge_anchor_into_file,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("tag_session")

# Script-level default (mirrors click_anchor.DEFAULT_COARSE_OFFSET_S). This is a
# UI seed for the very first change-jump when no anchor exists yet, NOT a
# scientific constant — replaced the instant the user places one anchor.
DEFAULT_COARSE_OFFSET_S = 15.0


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def _resolve_subject_pkl(session: str, subject: Optional[str]) -> Optional[str]:
    """Path to *subject*'s behavioural PKL for canonical *session*, or None.

    Subject-aware (unlike ``suite.loader.resolve_session_pkl``, which is frozen to
    ``config.SUBJECT``). Reuses ``list_pkl_sessions`` (the on-disk pkl convention
    ``data/pkls/<subject>/<subject>_<token>.pkl``) and matches each token by
    ``canonical_camera_session`` so 6-vs-8-digit tokens and leading-zero days both
    resolve.
    """
    subj = subject or config.SUBJECT
    pkl_dir = os.path.join(config.ROOT, "data", "pkls", subj)
    for token in list_pkl_sessions(subj):
        if config.canonical_camera_session(token) == session:
            return os.path.join(pkl_dir, f"{subj}_{token}.pkl")
    return None


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
    eye_roi: Optional[tuple] = None        # None in Plan 2a (ROI capture = 2b)
    full_frame: bool = True                # start on the cross-subject-safe view
    speed: float = 1.0                     # playback speed multiplier
    playing: bool = False
    timer: object = None                   # matplotlib canvas timer
    fig: object = None                     # the scrubber's Figure
    scrub_state: Optional[dict] = None     # the scrubber's own state dict


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
    #     --subject (NOT the frozen config.SUBJECT env). Load BEFORE seeding so a
    #     not-found session does not archive the user's prior anchors first.
    pkl_path = _resolve_subject_pkl(session, subject)
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

    # Real frame dimensions (for crop clamping). Prefer the container props; fall
    # back to decoding one frame if the props report 0.
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

    tag = TagSessionState(queue=queue, anchors=seed)

    # ---------------------------------------------------------------------
    # Crop handling (CRITICAL: eye_zoom_crop can pad past the frame edge and
    # produce NEGATIVE indices; numpy negative slices wrap silently. Clamp to
    # real frame bounds so 0<=y0<y1<=H and 0<=x0<x1<=W ALWAYS hold.)
    # ---------------------------------------------------------------------
    def _clamp_crop(crop):
        y0, y1, x0, x1 = [int(v) for v in crop]
        y0 = max(0, min(y0, frame_h - 1))
        y1 = max(y0 + 1, min(y1, frame_h))
        x0 = max(0, min(x0, frame_w - 1))
        x1 = max(x0 + 1, min(x1, frame_w))
        return (y0, y1, x0, x1)

    def _current_zoom_crop():
        # eye_roi is None in Plan 2a -> eye_zoom_crop returns the BG_046 fallback
        # (200,420,320,540); we clamp it to the real frame so it is safe on
        # BG_031/039/038 too (ROI-derived crops arrive in Plan 2b).
        return _clamp_crop(eye_zoom_crop(tag.eye_roi))

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
    # exposes no per-tick redraw hook, so playback needs its own capture). This
    # mirrors _tagger_ui._read_frame and respects the CURRENT cfg.crop.
    # ---------------------------------------------------------------------
    play_cap = cv2.VideoCapture(video_path)

    def _read_play_frame(fi: int) -> np.ndarray:
        play_cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = play_cap.read()
        crop = cfg.crop
        if not ok or frame is None:
            if crop is not None:
                y0, y1, x0, x1 = crop
                return np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            return np.zeros((frame_h, frame_w), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if crop is not None:
            y0, y1, x0, x1 = crop
            return gray[y0:y1, x0:x1]
        return gray

    def _draw_current(fi: int):
        fig = tag.fig
        if fig is None or not plt.fignum_exists(getattr(fig, "number", -1)):
            return
        try:
            im = fig.axes[0].images[0]
            hud = fig.axes[1].texts[0]
        except (IndexError, AttributeError):
            return
        frame = _read_play_frame(fi)
        im.set_data(frame)
        # Update extent too: the scrubber's own _refresh only set_data()s, so a
        # crop change would otherwise keep the previous extent and mis-display.
        im.set_extent((-0.5, frame.shape[1] - 0.5, frame.shape[0] - 0.5, -0.5))
        hud.set_text(_hud_fn(fi))
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
    # View + target navigation
    # ---------------------------------------------------------------------
    def _toggle_view(state) -> bool:
        tag.full_frame = not tag.full_frame
        cfg.crop = None if tag.full_frame else _current_zoom_crop()
        # Immediately fix the image extent for the new crop shape; subsequent
        # same-crop redraws (arrow keys via the scrubber) keep it valid.
        fig = tag.fig
        if fig is not None and plt.fignum_exists(getattr(fig, "number", -1)):
            try:
                im = fig.axes[0].images[0]
                frame = _read_play_frame(state["frame_idx"])
                im.set_data(frame)
                im.set_extent(
                    (-0.5, frame.shape[1] - 0.5, frame.shape[0] - 0.5, -0.5))
            except (IndexError, AttributeError):
                pass
        return True

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

        legend = ("[space]play  '['/']'spd {:g}x  [f]full/zoom  [j/k]jump  "
                  "[c]base<->chg  [enter]save  [d]del  [q]quit"
                  ).format(tag.speed)
        return "\n".join([
            mode_line,
            (f"trial {pos}/{ntot} (idx {trial_no})   frame {fi} ({video_s:.2f}s)"
             f"   Delta {delta:+d} vs pred"),
            f"anchors: {len(entries)} ({n_base} base / {n_chg} chg)     {qc}",
            legend,
        ])

    # ---------------------------------------------------------------------
    # Key dispatch
    # ---------------------------------------------------------------------
    def _on_key_extra(event, state) -> bool:
        tag.scrub_state = state
        tag.fig = event.canvas.figure
        key = event.key
        if key == " ":
            return _toggle_play(event)
        if key == "[":
            tag.speed = max(0.25, tag.speed / 2.0)
            _apply_speed()
            return True
        if key == "]":
            tag.speed = min(8.0, tag.speed * 2.0)
            _apply_speed()
            return True
        if key == "f":
            return _toggle_view(state)
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
        return False

    # ---------------------------------------------------------------------
    # Compose the scrubber. Default view = FULL FRAME (crop=None): the eye-zoom
    # fallback is BG_046-specific and this pilot runs on BG_031/039/038, so
    # full-frame is the cross-subject-safe startup view (f toggles the zoom).
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
