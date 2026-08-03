"""_tagger_ui.py — shared keyboard-driven frame scrubber / HUD primitive.

Extracted from ``click_anchor._run_scrub`` so the video-tagger tools can share a
single, byte-identical scrubber core. ``run_scrubber(cfg)`` owns the matplotlib
2-row figure (frame imshow + monospace HUD text), the cv2 ``VideoCapture`` frame
I/O, the arrow-step navigation (+/-1 / +/-10 / +/-100), the save keys, and the
quit keys. Everything tool-specific is supplied via ``ScrubberConfig`` hooks:

  * ``crop``        — ``(y0, y1, x0, x1)`` eye-region slice, or ``None`` for the
                      full frame.
  * ``hud_fn``      — ``frame_idx -> str`` builds the HUD text for a frame.
  * ``on_key_extra``— ``(event, state) -> bool`` handles tool-specific keys and
                      returns ``True`` when it consumed the key (so the scrubber
                      redraws).
  * ``on_save``     — ``frame_idx -> Optional[dict]`` runs when the user presses
                      Space/Enter; its return value becomes the scrubber result.

This module must not force a matplotlib backend: the interactive backend is
selected by the calling tool (e.g. ``click_anchor`` forces TkAgg at import).

NOTE: this is a behavior-preserving extraction — the frame-I/O, figure layout,
key deltas, bounds clamping, and quit/close semantics are lifted verbatim from
``click_anchor._run_scrub``.

(This module intentionally avoids ``from __future__ import annotations``: the
``@dataclass`` below must resolve real type objects, not string annotations, so
it stays import-clean even when loaded via ``spec_from_file_location`` without
being registered in ``sys.modules``.)
"""
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import cv2
import numpy as np

import matplotlib.pyplot as plt

# Pure geometry helper (no cv2 / matplotlib): the imshow extent that keeps the
# displayed image in FULL-FRAME data coords in both the full and cropped views.
# Imported here (not reimplemented) so the unit test covers the real code path.
from visdetect.analysis.video_labels import image_extent_for_crop


@dataclass
class ScrubberConfig:
    """Configuration for :func:`run_scrubber`.

    Attributes:
        video_path: Path to the video file to open with cv2.
        ts_ms: Per-frame timestamps in milliseconds (used by hooks, not the core).
        fps: Frame rate (used by hooks, not the core loop).
        n_frames: Total number of frames (bounds for navigation clamping).
        start_frame: Initial frame index (clamped into ``[0, n_frames - 1]``).
        crop: ``(y0, y1, x0, x1)`` slice applied to each frame, or ``None`` for
            the full frame.
        hud_fn: ``frame_idx -> str``; text rendered in the HUD row.
        on_key_extra: ``(event, state) -> bool``; handles tool-specific keys.
            Return ``True`` if the key was consumed (triggers a redraw).
        on_save: ``frame_idx -> Optional[dict]``; runs on Space/Enter. Its
            return value is stored as ``state["result"]`` and returned.
        on_selector: Optional ``(box, state) -> None`` drag seam. When set, the
            scrubber wires a ``RectangleSelector`` on the frame axes that a tool
            arms on demand via ``state["arm_selector"]()``; on drag completion
            the callback receives a FULL-FRAME ``(y0, y1, x0, x1)`` pixel box.
            This holds in EITHER view because the frame axes ALWAYS live in
            full-frame data coords: ``_refresh`` re-derives the image extent +
            axis limits from ``image_extent_for_crop(cfg.crop, ...)`` on every
            redraw, so ``eclick.xdata/ydata`` are already full-frame pixels
            whether or not a crop is active — no crop-origin rebasing is applied.
            (An earlier version added the crop origin back, which corrects only
            an ORIGIN offset and cannot undo the coordinate STRETCH that a frozen
            full-frame extent imposes on a cropped array; fixing the extent
            removes the need for any offset math.) The selector auto-disarms
            after each drag (or cancel). ``None`` (default) leaves the seam inert.
        on_refresh: Optional ``(frame_idx, fig) -> None`` post-frame redraw hook,
            invoked at the end of every ``_refresh`` (after the frame image and
            HUD update, before the canvas redraw) so a tool can draw overlays
            that survive the scrubber's internal arrow-step/jump redraws.
            ``None`` (default) is inert.
        on_selector_cancel: Optional ``(state) -> None`` hook fired when an armed
            drag is CANCELLED (empty drag, or q/esc while armed) rather than
            completed, so a tool can drop its own arming intent. A completed drag
            resets its intent inside ``on_selector``; this covers the cancel
            paths, where ``on_selector`` never runs. ``None`` (default) is inert.
    """

    video_path: str
    ts_ms: np.ndarray
    fps: float
    n_frames: int
    start_frame: int
    crop: Optional[Tuple[int, int, int, int]]
    hud_fn: Callable[[int], str]
    on_key_extra: Callable[[Any, dict], bool]
    on_save: Callable[[int], Optional[dict]]
    on_selector: Optional[Callable[[Tuple[int, int, int, int], dict], None]] = None
    on_refresh: Optional[Callable[[int, Any], None]] = None
    on_selector_cancel: Optional[Callable[[dict], None]] = None


def run_scrubber(cfg: ScrubberConfig) -> Optional[dict]:
    """Keyboard-driven frame-by-frame scrubber over a video.

    Opens a matplotlib window showing one (optionally cropped) frame at a time.
    The user navigates with arrow keys (and Shift/Ctrl modifiers), saves with
    Space/Enter (routed to ``cfg.on_save``), and quits with Q/ESC without
    saving. Any other key is routed to ``cfg.on_key_extra``.

    Returns the value ``cfg.on_save`` produced (stored in ``state["result"]``),
    or ``None`` if the user quit without saving.
    """
    # Capture state via a mutable container so closures/hooks can mutate it.
    state = {
        "frame_idx": int(np.clip(cfg.start_frame, 0, cfg.n_frames - 1)),
        "result": None,  # Optional[dict]
    }

    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {cfg.video_path}")

    # Full-frame dimensions, needed for the imshow extent so the displayed image
    # lives in FULL-FRAME data coords in BOTH the full and cropped views (see
    # _refresh). Prefer the container props; fall back to a decoded frame's shape
    # if they report 0. This is the same source _read_frame decodes from, so the
    # dims are never hardcoded. (_read_frame re-seeks before every read, so the
    # probe read below does not disturb navigation.)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if frame_h <= 0 or frame_w <= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok0, f0 = cap.read()
        if ok0 and f0 is not None:
            frame_h, frame_w = int(f0.shape[0]), int(f0.shape[1])

    # Placeholder shape for unreadable frames (keeps the layout stable).
    if cfg.crop is not None:
        y0, y1, x0, x1 = cfg.crop
        placeholder_shape = (y1 - y0, x1 - x0)
    else:
        placeholder_shape = (frame_h, frame_w)

    def _read_frame(fi: int) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            return np.zeros(placeholder_shape, dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cfg.crop is not None:
            y0, y1, x0, x1 = cfg.crop
            return gray[y0:y1, x0:x1]
        return gray

    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1], hspace=0.1)
    ax_frame = fig.add_subplot(gs[0])
    ax_hud = fig.add_subplot(gs[1])
    ax_hud.axis("off")

    im = ax_frame.imshow(_read_frame(state["frame_idx"]), cmap="gray",
                         vmin=0, vmax=255, interpolation="nearest")
    ax_frame.set_xticks([]); ax_frame.set_yticks([])

    hud_text = ax_hud.text(
        0.02, 0.5, "", fontsize=9, family="monospace",
        verticalalignment="center", transform=ax_hud.transAxes,
    )

    def _refresh():
        fi = state["frame_idx"]
        im.set_data(_read_frame(fi))
        # WHY update the extent + limits on EVERY refresh: the image artist is
        # created once by imshow() above (while cfg.crop may be None), which
        # FREEZES its extent to the full-frame size. A tool can toggle cfg.crop
        # live (eye-zoom), and set_data() alone would then STRETCH the smaller
        # cropped array across that frozen full-frame extent — silently
        # rescaling every coordinate read off the axes (ROI drags, pupil
        # ellipses). Re-deriving the extent + matching limits from the live crop
        # keeps the displayed image in FULL-FRAME data coords in both views, so
        # coordinates need no crop-origin fudge (an origin offset cannot undo a
        # stretch anyway).
        left, right, bottom, top = image_extent_for_crop(cfg.crop, frame_h, frame_w)
        im.set_extent((left, right, bottom, top))
        ax_frame.set_xlim(left, right)
        ax_frame.set_ylim(bottom, top)
        hud_text.set_text(cfg.hud_fn(fi))
        if cfg.on_refresh is not None:
            cfg.on_refresh(fi, fig)
        fig.canvas.draw_idle()

    # Optional ROI/correction drag seam. Inert (and click_anchor-safe) unless the
    # tool supplies cfg.on_selector. The selector is armed only on demand (the tool
    # calls state["arm_selector"] from its on_key_extra) and auto-disarms as soon
    # as a drag completes or is cancelled, so arrow-stepping/playback/save/quit
    # keep working while no drag is in progress. Boxes are reported in FULL-FRAME
    # pixel coords: the frame axes always live in full-frame data coords (see
    # _refresh's extent update), so a drag reads true frame pixels in EITHER view.
    selector = {"obj": None}

    def _disarm_selector():
        if selector["obj"] is not None:
            selector["obj"].set_active(False)
        state["selector_armed"] = False

    def _cancel_selector():
        # Cancel an armed-but-uncompleted drag (empty drag, or q/esc while armed):
        # disarm the widget AND notify the tool so it can drop its own arming
        # INTENT. A completed drag resets its intent inside cfg.on_selector; a
        # cancel otherwise leaves a stale tool-side intent that would mis-route the
        # NEXT completed drag. Deliberately NOT called on the success path, which
        # must keep the intent until cfg.on_selector has consumed it.
        _disarm_selector()
        if cfg.on_selector_cancel is not None:
            cfg.on_selector_cancel(state)

    if cfg.on_selector is not None:
        from matplotlib.widgets import RectangleSelector

        def _on_select(eclick, erelease):
            if eclick.xdata is None or erelease.xdata is None:
                _cancel_selector()
                return
            x0, x1 = sorted((eclick.xdata, erelease.xdata))
            y0, y1 = sorted((eclick.ydata, erelease.ydata))
            # The frame axes ALWAYS live in FULL-FRAME data coords (the _refresh
            # extent update places both the full and cropped views there), so
            # eclick.xdata/ydata are already full-frame pixels in EITHER view.
            # NO crop-origin rebasing: an earlier version added crop[0]/crop[2]
            # back, but that corrects only an ORIGIN offset — it could not undo the
            # coordinate STRETCH a frozen full-frame extent imposed on a cropped
            # array, so a zoomed drag still recorded garbage. Fixing the extent
            # removes the need for (and the error in) any offset math.
            box = (int(round(y0)), int(round(y1)), int(round(x0)), int(round(x1)))
            _disarm_selector()
            cfg.on_selector(box, state)
            _refresh()

        selector["obj"] = RectangleSelector(
            ax_frame, _on_select, useblit=False, button=[1],
            minspanx=3, minspany=3, spancoords="pixels", interactive=False)
        selector["obj"].set_active(False)
        state["selector_armed"] = False

        def _arm_selector():
            if selector["obj"] is not None:
                selector["obj"].set_active(True)
                state["selector_armed"] = True

        state["arm_selector"] = _arm_selector

    def on_key(event):
        key = event.key
        if key in ("q", "escape"):
            if state.get("selector_armed"):
                _cancel_selector()
                _refresh()
                return
            plt.close(fig); return

        if state.get("selector_armed"):
            # A drag is being set up: do NOT navigate/advance the frame. Route the
            # key to the tool (e.g. to re-arm a different ROI) then redraw.
            if cfg.on_key_extra(event, state):
                _refresh()
            return

        step = 0
        if key == "left":
            step = -1
        elif key == "right":
            step = +1
        elif key in ("shift+left", "pageup"):
            step = -10
        elif key in ("shift+right", "pagedown"):
            step = +10
        elif key in ("ctrl+left",):
            step = -100
        elif key in ("ctrl+right",):
            step = +100

        if step != 0:
            state["frame_idx"] = int(np.clip(state["frame_idx"] + step, 0, cfg.n_frames - 1))
            _refresh()
            return

        if key == "enter":
            # Let a tool bind enter first (e.g. tag_session save-keep-open),
            # mirroring the space routing below. If the hook consumes enter
            # (returns True), do NOT close -- multi-anchor tagging keeps the
            # window open. click_anchor's hook has no enter case -> returns
            # False -> the default save-and-close below runs unchanged.
            if cfg.on_key_extra(event, state):
                _refresh()
                return
            # Default: enter saves and closes (click_anchor behavior).
            state["result"] = cfg.on_save(state["frame_idx"])
            plt.close(fig)
            return
        if key == " ":
            # Let a tool bind space first (e.g. tag_session play/pause).
            if cfg.on_key_extra(event, state):
                _refresh()
                return
            # Default: space still saves (click_anchor behavior).
            state["result"] = cfg.on_save(state["frame_idx"])
            plt.close(fig)
            return

        # Tool-specific keys: redraw if the hook consumed the event.
        if cfg.on_key_extra(event, state):
            _refresh()
        return

    # Neutralize matplotlib's default key bindings that collide with our hooks
    # (s=save-dialog, f=fullscreen, k/L=xscale->log garbles the frame,
    # c/left=back, v/right=forward, plus home/pan/zoom/grid/yscale), so on_key
    # is the sole handler for these keys.
    for _k in ("keymap.save", "keymap.fullscreen", "keymap.xscale", "keymap.yscale",
               "keymap.back", "keymap.forward", "keymap.home", "keymap.pan",
               "keymap.zoom", "keymap.grid"):
        plt.rcParams[_k] = []

    fig.canvas.mpl_connect("key_press_event", on_key)
    try:
        _refresh()
        plt.show()
    finally:
        cap.release()

    return state["result"]
