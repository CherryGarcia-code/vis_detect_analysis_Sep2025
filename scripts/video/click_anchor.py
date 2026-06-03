"""click_anchor.py — Phase 1 of the video-sync anchor-barcode redesign.

Workflow:
  1. Load a session and pick a predicted trial-1 frame from cached coarse offset.
  2. Show a coarse 5x10 grid (1s sampling, 50s span); user clicks the cell where
     the grating first appears in the eye.
  3. Show a fine 5x10 grid (1 frame/cell, +/-500ms span) around the stage-1 click;
     user clicks the exact frame.
  4. Save the anchor JSON.
  5. Render a 5-row x 7-column barcode montage PNG so the user can verify the
     implied clock offset visually.

Run:  py scripts/video/click_anchor.py --session 09092025
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional, Sequence

import cv2
import numpy as np

# Visdetect imports
from visdetect.suite.loader import load_session
from visdetect.core.video_sync import (
    find_camera_files,
    load_camera_metadata,
    compute_predicted_frame_idx,
    load_anchor,
    save_anchor,
    compute_implied_offset,
    _build_anchor_entry,
    _build_v2_anchor_file,
    _merge_anchor_into_file,
)
from visdetect.analysis.config import VIDEO_SYNC_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("click_anchor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EYE_REGION_CROP_BG046 = (200, 420, 320, 540)  # y0, y1, x0, x1

# Both stages use the same 5x10 grid; only the frame indices differ.
GRID_ROWS = 5
GRID_COLS = 10
N_CELLS = GRID_ROWS * GRID_COLS  # 50

# Stage 1: coarse window, 1 s sampling, 50 s span, biased forward.
STAGE1_PRE_S = 15.0   # seconds before predicted onset
STAGE1_POST_S = 35.0  # seconds after predicted onset
STAGE1_SAMPLING_S = 1.0

# Stage 2: window biased entirely backward from stage-1 click.
# Stage 1's 1-second sampling means the actual onset is somewhere in the
# 1-second interval ending at the clicked cell. STAGE2_PRE_FRAMES + STAGE2_POST_FRAMES + 1
# must equal N_CELLS (=50) so the grid is full.
STAGE2_PRE_FRAMES = 49
STAGE2_POST_FRAMES = 0

# Default coarse offset if missing from cache.
DEFAULT_COARSE_OFFSET_S = 15.0

# Barcode montage layout.
MONTAGE_ROWS = 5
MONTAGE_COLS = 7  # frames around predicted onset: -3, -2, -1, 0, +1, +2, +3

# Output directories (created on demand).
FIGS_DIR = os.path.join("figs", "video_sync")


# ---------------------------------------------------------------------------
# Grid-math helpers (pure-logic, unit-tested)
# ---------------------------------------------------------------------------


def stage1_frame_indices(predicted: int, fps: float, n_frames: int) -> list[int]:
    """Return 50 frame indices for the coarse stage.

    Sampling interval is 1 s of video (= ``fps`` frames). Window covers
    ``[predicted - STAGE1_PRE_S, predicted + STAGE1_POST_S]`` in seconds,
    clamped to ``[0, n_frames - 1]``. If clamping shortens the window, the
    sampling step is preserved; the array may end earlier than predicted+post.
    """
    step = max(1, int(round(fps * STAGE1_SAMPLING_S)))
    start = predicted - int(round(fps * STAGE1_PRE_S))
    start = max(0, start)
    # End is start + (N_CELLS - 1) * step, but clamped to n_frames - 1.
    indices = [min(n_frames - 1, start + i * step) for i in range(N_CELLS)]
    return indices


def stage2_frame_indices(stage1_click: int, fps: float, n_frames: int) -> list[int]:
    """Return 50 frame indices for the fine stage (1 frame per cell).

    Window is biased entirely backward: [click - STAGE2_PRE_FRAMES, click + STAGE2_POST_FRAMES].
    The stage-1 click is the rightmost cell (gold-bordered in the UI), and the
    actual grating onset is in the 1-second interval ending there.
    """
    # Note: fps not used here (one frame per cell), but kept for signature symmetry.
    start = stage1_click - STAGE2_PRE_FRAMES
    start = max(0, min(n_frames - N_CELLS, start))
    return [min(n_frames - 1, start + i) for i in range(N_CELLS)]


def _build_or_merge_anchor_file(
    session_name: str,
    baseline_on: np.ndarray,
    ts_ms: np.ndarray,
    fps: float,
    trial_index: int,
    frame_idx: int,
) -> dict:
    """Build a v2 anchor JSON dict that merges this anchor into any existing file.

    All three Phase 1 anchor-creation paths (2-stage click, scrub Save, scrub
    preview-render) use this helper so the multi-anchor list stays consistent
    across re-saves.
    """
    new_entry = _build_anchor_entry(
        baseline_on=baseline_on,
        ts_ms=ts_ms,
        trial_index=trial_index,
        frame_idx=frame_idx,
    )
    existing = load_anchor(session_name)
    if existing is None:
        return _build_v2_anchor_file(
            session_name=session_name,
            fps=fps,
            n_trials=int(len(baseline_on)),
            anchor_entries=[new_entry],
        )
    return _merge_anchor_into_file(existing, new_entry)


def jump_to_predicted_frame(
    trial_idx: int,
    baseline_on: np.ndarray,
    implied_offset_s: float,
    ts_ms: np.ndarray,
) -> int:
    """Return the video frame index closest to trial *trial_idx*'s predicted onset.

    Predicted video time of trial i = baseline_on[i] + implied_offset_s.
    Returns the nearest frame index in ts_ms, clamped to [0, len(ts_ms)-1].
    Raises IndexError if trial_idx is out of range.
    """
    if trial_idx < 0 or trial_idx >= len(baseline_on):
        raise IndexError(
            f"trial_idx {trial_idx} out of range [0, {len(baseline_on) - 1}]"
        )
    target_ms = (float(baseline_on[trial_idx]) + implied_offset_s) * 1000.0
    if target_ms <= ts_ms[0]:
        return 0
    if target_ms >= ts_ms[-1]:
        return int(len(ts_ms) - 1)
    return int(np.argmin(np.abs(ts_ms - target_ms)))


def _predicted_last_trial_frame(
    anchor0: dict,
    baseline_on: np.ndarray,
    ts_ms: np.ndarray,
) -> int:
    """Predict the video frame for the last task trial using anchor 0's offset.

    Used to seed the --anchor-last scrubber so it opens close to the actual
    last-trial Baseline_ON. Uses slope=1 (the offset from anchor 0); the
    scrubber lets the user correct any drift.
    """
    implied_offset_s = (
        float(anchor0["video_time_s"]) - float(anchor0["nidaq_baseline_on_s"])
    )
    last_nidaq_s = float(baseline_on[-1])
    target_ms = (last_nidaq_s + implied_offset_s) * 1000.0
    if target_ms <= ts_ms[0]:
        return 0
    if target_ms >= ts_ms[-1]:
        return int(len(ts_ms) - 1)
    return int(np.argmin(np.abs(ts_ms - target_ms)))


# ---------------------------------------------------------------------------
# Frame I/O + eye-region crop
# ---------------------------------------------------------------------------


def load_cropped_frames(
    video_path: str,
    frame_indices: Sequence[int],
    crop: tuple[int, int, int, int] = EYE_REGION_CROP_BG046,
) -> list[np.ndarray]:
    """Load specified frames from *video_path*, crop to *crop*, return grayscale arrays.

    Reads via ``cv2.VideoCapture``. Frames are returned in the same order as
    ``frame_indices``. If a frame cannot be read, a zero-filled placeholder
    of the crop shape is inserted (so the grid layout never breaks).
    """
    y0, y1, x0, x1 = crop
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    out: list[np.ndarray] = []
    try:
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                out.append(np.zeros((y1 - y0, x1 - x0), dtype=np.uint8))
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out.append(gray[y0:y1, x0:x1])
    finally:
        cap.release()
    return out


# ---------------------------------------------------------------------------
# Interactive two-stage click UI
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("TkAgg", force=True)  # interactive backend; force=True for resilience vs other modules' Agg setup
import matplotlib.pyplot as plt


def _show_grid_and_get_click(
    frames: list[np.ndarray],
    frame_indices: Sequence[int],
    title: str,
    centre_frame: int,
    fps: float,
) -> Optional[int]:
    """Show a 5x10 grid of *frames* and wait for one click.

    Returns the absolute video frame index of the clicked cell, or ``None``
    if the user pressed ESC.

    Cell labels show "fr <abs_idx>\n<offset>ms" where offset is relative to
    *centre_frame*. Centre cell gets a yellow border.
    """
    assert len(frames) == N_CELLS == len(frame_indices)

    fig, axes = plt.subplots(
        GRID_ROWS, GRID_COLS, figsize=(15, 8.5),
        gridspec_kw=dict(wspace=0.05, hspace=0.25),
    )
    fig.suptitle(title, fontsize=10)

    centre_idx = int(np.argmin(np.abs(np.asarray(frame_indices) - centre_frame)))

    for i, (ax, frame, fidx) in enumerate(zip(axes.flat, frames, frame_indices)):
        ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
        offset_ms = (int(fidx) - int(centre_frame)) / fps * 1000.0
        ax.set_title(f"fr {fidx}\n{offset_ms:+.0f}ms", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if i == centre_idx:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("gold")
                spine.set_linewidth(2.0)

    # State captured by handlers via mutable list (closures in Python).
    result: list[Optional[int]] = [None]

    def on_click(event):
        if event.inaxes is None:
            return
        # Identify which axes was clicked.
        for i, ax in enumerate(axes.flat):
            if event.inaxes is ax:
                # Confirm visually: draw a red box around the clicked axes.
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor("red")
                    spine.set_linewidth(3.0)
                fig.canvas.draw_idle()
                result[0] = int(frame_indices[i])
                # Schedule close after a short pause so the user sees the confirmation.
                fig.canvas.start_event_loop(0.5)
                plt.close(fig)
                return

    def on_key(event):
        if event.key == "escape":
            result[0] = None
            plt.close(fig)

    cid_click = fig.canvas.mpl_connect("button_press_event", on_click)
    cid_key = fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()  # blocks until plt.close(fig)

    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)

    return result[0]


def run_stage1(
    video_path: str,
    predicted: int,
    fps: float,
    n_frames: int,
) -> Optional[int]:
    """Stage 1 - coarse 50-second window, 1 frame per second."""
    indices = stage1_frame_indices(predicted, fps, n_frames)
    frames = load_cropped_frames(video_path, indices)
    return _show_grid_and_get_click(
        frames=frames,
        frame_indices=indices,
        title=(
            "Stage 1 - Coarse scan. Click the cell where the grating first "
            "appears in the eye. ESC to cancel.\n"
            f"(predicted onset = frame {predicted}; 1 s between cells; gold = predicted)"
        ),
        centre_frame=predicted,
        fps=fps,
    )


def run_stage2(
    video_path: str,
    stage1_click: int,
    fps: float,
    n_frames: int,
) -> Optional[int]:
    """Stage 2 - fine +/-500ms window, 1 frame per cell."""
    indices = stage2_frame_indices(stage1_click, fps, n_frames)
    frames = load_cropped_frames(video_path, indices)
    return _show_grid_and_get_click(
        frames=frames,
        frame_indices=indices,
        title=(
            "Stage 2 - Fine pick. Click the exact frame where the grating "
            "appears. ESC to cancel.\n"
            f"(stage-1 click at right edge = gold; window is 1s biased backward; 1 frame between cells)"
        ),
        centre_frame=stage1_click,
        fps=fps,
    )


def _run_scrub(
    session_name: str,
    video_path: str,
    baseline_on: np.ndarray,
    ts_ms: np.ndarray,
    fps: float,
    n_frames: int,
    start_frame: int,
    existing_anchor: Optional[dict],
    anchor_trial_index: int = 0,
) -> Optional[dict]:
    """Keyboard-driven frame-by-frame scrubber for the eye-cam video.

    Opens a TkAgg window showing one cropped eye frame at a time. The user
    navigates with arrow keys (and modifiers), jumps between predicted trial
    onsets with J/K/Home/End, and saves the current frame as the anchor with
    Space/Enter. Quits with Q/ESC without saving.

    Returns the saved anchor dict on success, or None if the user quit without saving.

    The "implied offset" used for J/K/Home/End jumps is computed from the
    *existing* anchor when present, otherwise from the current scrub frame
    itself (so jumps remain meaningful as the user navigates).
    """
    # Capture state via mutable containers so closures can mutate it.
    state = {
        "frame_idx": int(np.clip(start_frame, 0, n_frames - 1)),
        "saved_anchor": None,  # Optional[dict]
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    y0, y1, x0, x1 = EYE_REGION_CROP_BG046

    def _read_frame(fi: int) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            return np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray[y0:y1, x0:x1]

    def _implied_offset_for_jumps() -> float:
        """Use existing anchor's offset if present, else compute from current frame."""
        if existing_anchor is not None:
            # Always use the trial-0 anchor for jump offset (slope=1 approximation).
            # --anchor-last intentionally passes the full v2 file; trial-0 remains the
            # best linear seed even when a last-trial anchor is also present.
            entry0 = existing_anchor["anchors"][0]
            return compute_implied_offset(entry0)
        # Fall back: pretend current frame anchors trial 0.
        return float(ts_ms[state["frame_idx"]] / 1000.0 - float(baseline_on[0]))

    def _nearest_trial_idx(frame_idx: int) -> int:
        """Find the trial whose predicted video frame is closest to *frame_idx*."""
        offs = _implied_offset_for_jumps()
        # Predicted video time of each trial in ms
        predicted_ms = (baseline_on + offs) * 1000.0
        actual_ms = ts_ms[frame_idx]
        return int(np.argmin(np.abs(predicted_ms - actual_ms)))

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
        # Build HUD
        video_time_s = float(ts_ms[fi] / 1000.0)
        trial_idx = _nearest_trial_idx(fi)
        offs_jumps = _implied_offset_for_jumps()
        predicted_frame = jump_to_predicted_frame(
            trial_idx, baseline_on, offs_jumps, ts_ms
        )
        delta = fi - predicted_frame
        if_saved_offset_s = float(
            ts_ms[fi] / 1000.0 - float(baseline_on[int(anchor_trial_index)])
        )
        if existing_anchor is not None:
            existing_entry0 = existing_anchor["anchors"][0]
            existing_frame = int(existing_entry0["video_frame_idx"])
            existing_offset_s = compute_implied_offset(existing_entry0)
            existing_line = (
                f"Existing anchor [trial {int(existing_entry0['trial_index'])}]: frame {existing_frame} "
                f"(implied_offset = {existing_offset_s:+.4f} s)"
            )
        else:
            existing_line = "Existing anchor: none"
        lines = [
            f"Session {session_name}  |  frame {fi}  |  video time {video_time_s:.4f} s",
            existing_line,
            f"Nearest trial: {trial_idx}  (NI {float(baseline_on[trial_idx]):.4f} s, "
            f"predicted frame {predicted_frame}, Delta = {delta:+d} frame{'s' if abs(delta) != 1 else ''})",
            f"If saved here (anchor for trial {anchor_trial_index}): implied_offset = {if_saved_offset_s:+.4f} s",
            "",
            "Arrow keys = +/-1 frame    Shift+Arrow = +/-10    Ctrl+Arrow = +/-100",
            "J / K = next/prev predicted trial    Home/End = first/last trial    R = re-render montage",
            "Space / Enter = save anchor    Q / ESC = quit",
        ]
        hud_text.set_text("\n".join(lines))
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
            state["frame_idx"] = int(np.clip(state["frame_idx"] + step, 0, n_frames - 1))
            _refresh()
            return

        if key == "j":
            trial_idx = _nearest_trial_idx(state["frame_idx"])
            new_trial = min(len(baseline_on) - 1, trial_idx + 1)
            state["frame_idx"] = jump_to_predicted_frame(
                new_trial, baseline_on, _implied_offset_for_jumps(), ts_ms
            )
            _refresh()
            return
        if key == "k":
            trial_idx = _nearest_trial_idx(state["frame_idx"])
            new_trial = max(0, trial_idx - 1)
            state["frame_idx"] = jump_to_predicted_frame(
                new_trial, baseline_on, _implied_offset_for_jumps(), ts_ms
            )
            _refresh()
            return
        if key == "home":
            state["frame_idx"] = jump_to_predicted_frame(
                0, baseline_on, _implied_offset_for_jumps(), ts_ms
            )
            _refresh()
            return
        if key == "end":
            state["frame_idx"] = jump_to_predicted_frame(
                len(baseline_on) - 1, baseline_on, _implied_offset_for_jumps(), ts_ms
            )
            _refresh()
            return

        if key in (" ", "enter"):
            fi = state["frame_idx"]
            anchor = _build_or_merge_anchor_file(
                session_name, baseline_on, ts_ms, fps,
                trial_index=int(anchor_trial_index),
                frame_idx=state["frame_idx"],
            )
            save_anchor(session_name, anchor)
            state["saved_anchor"] = anchor
            saved_entry = next(
                a for a in anchor["anchors"]
                if int(a["trial_index"]) == int(anchor_trial_index)
            )
            logger.info(
                "Anchor saved via scrub: frame %d (video time %.4fs); implied offset = %.4fs",
                fi, saved_entry["video_time_s"], compute_implied_offset(saved_entry),
            )
            plt.close(fig)
            return

        if key == "r":
            # Render montage with current frame as candidate anchor (no save)
            candidate_file = _build_or_merge_anchor_file(
                session_name, baseline_on, ts_ms, fps,
                trial_index=int(anchor_trial_index),
                frame_idx=state["frame_idx"],
            )
            # render_barcode_montage expects v1 single-anchor shape.
            entry = next(
                a for a in candidate_file["anchors"]
                if int(a["trial_index"]) == int(anchor_trial_index)
            )
            candidate_for_render = {
                "session": candidate_file["session"],
                "anchor_trial_index": entry["trial_index"],
                "nidaq_baseline_on_s": entry["nidaq_baseline_on_s"],
                "video_frame_idx": entry["video_frame_idx"],
                "video_time_s": entry["video_time_s"],
                "implied_offset_s": compute_implied_offset(entry),
                "frame_rate_fps": candidate_file["frame_rate_fps"],
                "n_trials": candidate_file["n_trials"],
                "clicked_at": entry["clicked_at"],
            }
            montage_path = os.path.join(
                FIGS_DIR, f"{session_name}_barcode_montage_PREVIEW.png"
            )
            render_barcode_montage(
                session_name=session_name,
                anchor=candidate_for_render,
                baseline_on=baseline_on,
                video_path=video_path,
                ts_ms=ts_ms,
                fps=fps,
                out_path=montage_path,
            )
            logger.info("Preview montage written: %s", montage_path)

    fig.canvas.mpl_connect("key_press_event", on_key)
    try:
        _refresh()
        plt.show()
    finally:
        cap.release()

    return state["saved_anchor"]


# ---------------------------------------------------------------------------
# Barcode montage renderer
# ---------------------------------------------------------------------------


def _pick_sampled_trials(n_trials: int, n_rows: int = MONTAGE_ROWS) -> list[int]:
    """Return ``n_rows`` evenly spaced trial indices including 0 and n_trials-1."""
    if n_trials <= n_rows:
        return list(range(n_trials))
    return [int(round(i * (n_trials - 1) / (n_rows - 1))) for i in range(n_rows)]


def render_barcode_montage(
    session_name: str,
    anchor: dict,
    baseline_on: np.ndarray,
    video_path: str,
    ts_ms: np.ndarray,
    fps: float,
    out_path: str,
) -> None:
    """Render a 5-row x 7-column montage of predicted-onset frames per sampled trial.

    Each row corresponds to a sampled trial; columns show frames at predicted
    onset +/- 3 frames. Centre column gets a red border (the predicted-onset
    frame); the user inspects whether the grating appears in the centre cells.
    """
    required = {"implied_offset_s", "video_frame_idx", "nidaq_baseline_on_s"}
    missing = required - anchor.keys()
    if missing:
        raise KeyError(
            f"anchor dict missing required keys: {sorted(missing)}"
        )
    if len(baseline_on) == 0:
        raise ValueError("baseline_on is empty; nothing to render")
    n_trials = len(baseline_on)
    trial_indices = _pick_sampled_trials(n_trials, MONTAGE_ROWS)
    implied_offset_s = float(anchor["implied_offset_s"])

    col_offsets = list(range(-(MONTAGE_COLS // 2), MONTAGE_COLS // 2 + 1))  # [-3..3]

    fig, axes = plt.subplots(
        MONTAGE_ROWS, MONTAGE_COLS,
        figsize=(MONTAGE_COLS * 1.8, MONTAGE_ROWS * 2.0),
        gridspec_kw=dict(wspace=0.05, hspace=0.25),
    )
    title = (
        f"Anchor-barcode montage - {session_name}\n"
        f"anchor trial 0 @ frame {anchor['video_frame_idx']} "
        f"(NI-DAQ {anchor['nidaq_baseline_on_s']:.3f}s, "
        f"implied offset {implied_offset_s:.3f}s) - {n_trials} trials"
    )
    fig.suptitle(title, fontsize=10)

    n_frames = len(ts_ms)

    for r, ti in enumerate(trial_indices):
        # Predicted video frame for this trial: where ts_ms is closest to
        # (baseline_on[ti] + implied_offset_s) * 1000.
        target_ms = (float(baseline_on[ti]) + implied_offset_s) * 1000.0
        target_ms_clamped = float(np.clip(target_ms, ts_ms[0], ts_ms[-1]))
        centre_frame = int(np.argmin(np.abs(ts_ms - target_ms_clamped)))
        indices = [
            int(np.clip(centre_frame + off, 0, n_frames - 1)) for off in col_offsets
        ]
        frames = load_cropped_frames(video_path, indices)

        for ax, frame, off in zip(axes[r], frames, col_offsets):
            ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if off == 0:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor("red")
                    spine.set_linewidth(2.0)
            if r == 0:
                offset_ms = off / fps * 1000.0
                ax.set_title(f"{offset_ms:+.0f}ms", fontsize=8)

        # Row label (left side).
        axes[r, 0].set_ylabel(
            f"trial {ti}\nNI {float(baseline_on[ti]):.2f}s",
            fontsize=8, rotation=0, ha="right", va="center", labelpad=30,
        )

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manually anchor trial 1 Baseline_ON for a session's eye-cam video.",
    )
    parser.add_argument("--session", required=True, help="Session name (e.g. 09092025).")
    parser.add_argument(
        "--reuse-existing-anchor", action="store_true",
        help="Skip the click UI and just render the montage from a saved anchor.",
    )
    parser.add_argument(
        "--scrub", action="store_true",
        help="Open frame-by-frame scrubber UI instead of the 2-stage click flow.",
    )
    parser.add_argument(
        "--start-from", choices=("anchor", "coarse", "zero"), default=None,
        help="Starting frame for --scrub. Default: 'anchor' if anchor JSON exists, else 'coarse'.",
    )
    parser.add_argument(
        "--anchor-last", action="store_true",
        help="Anchor the LAST task trial (uses scrubber UI, requires a trial-0 "
             "anchor to already exist).",
    )
    args = parser.parse_args()

    session_name = args.session
    # Normalize to 8-digit DDMMYYYY so cache lookups and file paths match
    # the convention used by save_anchor / load_anchor / save_video_sync.
    try:
        session_name = str(int(session_name)).zfill(8)
    except (TypeError, ValueError):
        logger.error("Session name '%s' is not numeric.", args.session)
        return 2

    # Load session + camera + coarse offset
    sess = load_session(session_name)
    baseline_on = np.asarray(
        sess.ni_events.get("Baseline_ON", []), dtype=float
    )
    baseline_on = baseline_on[baseline_on > 0]
    n_task_trials = len(sess.trials)
    if n_task_trials > 0 and len(baseline_on) > n_task_trials:
        baseline_on = baseline_on[:n_task_trials]
    if len(baseline_on) == 0:
        logger.error("No Baseline_ON events for session %s - aborting.", session_name)
        return 2

    try:
        cam_files = find_camera_files(session_name)
    except Exception as exc:
        logger.error("Could not locate camera files for %s: %s", session_name, exc)
        return 2
    if "eye_cam" not in cam_files:
        logger.error("No eye-cam video/metadata pair found for %s.", session_name)
        return 2
    video_path = cam_files["eye_cam"]["video"]
    meta_path = cam_files["eye_cam"]["metadata"]

    ts_ms, _, _ = load_camera_metadata(meta_path)
    fps = 1000.0 / float(np.median(np.diff(ts_ms)))
    n_frames = len(ts_ms)

    coarse_offset_s = _read_coarse_offset(session_name)
    if coarse_offset_s is None:
        logger.warning(
            "No cached coarse offset for %s; falling back to %.1fs default.",
            session_name, DEFAULT_COARSE_OFFSET_S,
        )
        coarse_offset_s = DEFAULT_COARSE_OFFSET_S

    predicted = compute_predicted_frame_idx(
        baseline_on_s=float(baseline_on[0]),
        coarse_offset_s=float(coarse_offset_s),
        ts_ms=ts_ms,
    )

    # Anchor: load existing or run two-stage click
    anchor: Optional[dict] = None
    if args.anchor_last:
        existing = load_anchor(session_name)
        if existing is None:
            logger.error(
                "--anchor-last requires an existing anchor file. "
                "Run --session %s first (no --anchor-last) to anchor trial 0.",
                session_name,
            )
            return 2
        # Find anchor for trial 0 (must exist).
        anchor0 = next(
            (a for a in existing["anchors"] if int(a["trial_index"]) == 0),
            None,
        )
        if anchor0 is None:
            logger.error(
                "Existing anchor file has no trial-0 anchor. "
                "Run --session %s first (no --anchor-last) to anchor trial 0.",
                session_name,
            )
            return 2
        last_trial_idx = int(len(baseline_on)) - 1
        if last_trial_idx <= 0:
            logger.error("Session has <=1 trial; nothing to anchor as 'last'.")
            return 2
        start_frame = _predicted_last_trial_frame(anchor0, baseline_on, ts_ms)
        logger.info(
            "Opening scrubber at predicted last-trial (idx %d) frame %d.",
            last_trial_idx, start_frame,
        )
        anchor_after = _run_scrub(
            session_name=session_name,
            video_path=video_path,
            baseline_on=baseline_on,
            ts_ms=ts_ms,
            fps=fps,
            n_frames=n_frames,
            start_frame=start_frame,
            existing_anchor=existing,  # full v2 file for HUD context + jump offsets
            anchor_trial_index=last_trial_idx,
        )
        if anchor_after is None:
            logger.info("Scrubber exited without saving.")
            return 1
        anchor = anchor_after
    elif args.scrub:
        # Resolve start-frame.
        start_mode = args.start_from
        existing = load_anchor(session_name)
        if start_mode is None:
            start_mode = "anchor" if existing is not None else "coarse"
        if start_mode == "anchor" and existing is not None:
            start_frame = int(existing["anchors"][0]["video_frame_idx"])
        elif start_mode == "coarse":
            start_frame = predicted
        else:  # "zero" or "anchor" without existing anchor
            start_frame = 0

        anchor_after = _run_scrub(
            session_name=session_name,
            video_path=video_path,
            baseline_on=baseline_on,
            ts_ms=ts_ms,
            fps=fps,
            n_frames=n_frames,
            start_frame=start_frame,
            existing_anchor=existing,
        )
        if anchor_after is None:
            # User quit without saving.
            logger.info("Scrubber exited without saving.")
            return 1
        anchor = anchor_after  # use for downstream montage rendering
    else:
        if args.reuse_existing_anchor:
            anchor = load_anchor(session_name)
            if anchor is None:
                logger.error(
                    "--reuse-existing-anchor passed but no anchor JSON found for %s.",
                    session_name,
                )
                return 2
        else:
            existing = load_anchor(session_name)
            if existing is not None:
                resp = input(
                    f"Anchor JSON for {session_name} already exists; overwrite? [y/N] "
                ).strip().lower()
                if resp not in ("y", "yes"):
                    logger.info("Using existing anchor; skipping click UI.")
                    anchor = existing

            if anchor is None:
                click1 = run_stage1(video_path, predicted, fps, n_frames)
                if click1 is None:
                    logger.info("Stage 1 cancelled by user.")
                    return 1
                click2 = run_stage2(video_path, click1, fps, n_frames)
                if click2 is None:
                    logger.info("Stage 2 cancelled by user.")
                    return 1

                anchor = _build_or_merge_anchor_file(
                    session_name, baseline_on, ts_ms, fps,
                    trial_index=0, frame_idx=click2,
                )
                save_anchor(session_name, anchor)
                entry0 = anchor["anchors"][0]
                logger.info(
                    "Anchor saved: trial 0 @ frame %d (video time %.3fs); implied offset = %.3fs",
                    entry0["video_frame_idx"],
                    entry0["video_time_s"],
                    compute_implied_offset(entry0),
                )

    # Render montage. render_barcode_montage takes v1 single-anchor shape.
    entry0 = anchor["anchors"][0]
    anchor_for_render = {
        "session": anchor["session"],
        "anchor_trial_index": entry0["trial_index"],
        "nidaq_baseline_on_s": entry0["nidaq_baseline_on_s"],
        "video_frame_idx": entry0["video_frame_idx"],
        "video_time_s": entry0["video_time_s"],
        "implied_offset_s": compute_implied_offset(entry0),
        "frame_rate_fps": anchor["frame_rate_fps"],
        "n_trials": anchor["n_trials"],
        "clicked_at": entry0["clicked_at"],
    }
    montage_path = os.path.join(FIGS_DIR, f"{session_name}_barcode_montage.png")
    render_barcode_montage(
        session_name=session_name,
        anchor=anchor_for_render,
        baseline_on=baseline_on,
        video_path=video_path,
        ts_ms=ts_ms,
        fps=fps,
        out_path=montage_path,
    )

    anchor_path = os.path.join(VIDEO_SYNC_DIR, f"{session_name}_anchor.json")
    print(f"Anchor:   {anchor_path}")
    print(f"Montage:  {montage_path}")
    return 0


def _read_coarse_offset(session_name: str) -> Optional[float]:
    """Return the cached coarse offset for *session_name*, or ``None`` if absent."""
    path = os.path.join(VIDEO_SYNC_DIR, "coarse_offsets.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    val = data.get(session_name)
    return float(val) if val is not None else None


if __name__ == "__main__":
    sys.exit(main())
