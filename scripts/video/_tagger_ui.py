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
        fig.canvas.draw_idle()

    def on_key(event):
        key = event.key
        if key in ("q", "escape"):
            plt.close(fig); return

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

        if key in (" ", "enter"):
            state["result"] = cfg.on_save(state["frame_idx"])
            plt.close(fig)
            return

        # Tool-specific keys: redraw if the hook consumed the event.
        if cfg.on_key_extra(event, state):
            _refresh()
        return

    fig.canvas.mpl_connect("key_press_event", on_key)
    try:
        _refresh()
        plt.show()
    finally:
        cap.release()

    return state["result"]
