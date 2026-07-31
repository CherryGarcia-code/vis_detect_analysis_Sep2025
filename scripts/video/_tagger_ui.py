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
            The selector auto-disarms after each drag (or cancel). ``None``
            (default) leaves the seam completely inert.
        on_refresh: Optional ``(frame_idx, fig) -> None`` post-frame redraw hook,
            invoked at the end of every ``_refresh`` (after the frame image and
            HUD update, before the canvas redraw) so a tool can draw overlays
            that survive the scrubber's internal arrow-step/jump redraws.
            ``None`` (default) is inert.
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

    # Placeholder shape for unreadable frames (keeps the layout stable).
    if cfg.crop is not None:
        y0, y1, x0, x1 = cfg.crop
        placeholder_shape = (y1 - y0, x1 - x0)
    else:
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        placeholder_shape = (h, w)

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
        hud_text.set_text(cfg.hud_fn(fi))
        if cfg.on_refresh is not None:
            cfg.on_refresh(fi, fig)
        fig.canvas.draw_idle()

    # Optional ROI/correction drag seam. Inert (and click_anchor-safe) unless the
    # tool supplies cfg.on_selector. The selector is armed only on demand (the tool
    # calls state["arm_selector"] from its on_key_extra) and auto-disarms as soon
    # as a drag completes or is cancelled, so arrow-stepping/playback/save/quit
    # keep working while no drag is in progress. Boxes are reported in FULL-FRAME
    # pixel coords: the tool only arms the selector in the full-frame view
    # (cfg.crop is None), where the imshow data coords equal frame pixels.
    selector = {"obj": None}

    def _disarm_selector():
        if selector["obj"] is not None:
            selector["obj"].set_active(False)
        state["selector_armed"] = False

    if cfg.on_selector is not None:
        from matplotlib.widgets import RectangleSelector

        def _on_select(eclick, erelease):
            if eclick.xdata is None or erelease.xdata is None:
                _disarm_selector()
                return
            x0, x1 = sorted((eclick.xdata, erelease.xdata))
            y0, y1 = sorted((eclick.ydata, erelease.ydata))
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
                _disarm_selector()
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
