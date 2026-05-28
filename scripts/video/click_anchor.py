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
import datetime as _dt
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

# Stage 2: +/-25 frames around stage-1 click.
STAGE2_HALF_WIDTH_FRAMES = 25

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
    """Return 50 frame indices for the fine stage (1 frame per cell)."""
    # Note: fps not used here (one frame per cell), but kept for signature symmetry.
    start = stage1_click - STAGE2_HALF_WIDTH_FRAMES
    start = max(0, min(n_frames - N_CELLS, start))
    return [min(n_frames - 1, start + i) for i in range(N_CELLS)]


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
            f"(stage-1 click = frame {stage1_click}; 1 frame between cells; gold = stage-1 click)"
        ),
        centre_frame=stage1_click,
        fps=fps,
    )


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

            anchor = {
                "session": session_name,
                "anchor_trial_index": 0,
                "nidaq_baseline_on_s": float(baseline_on[0]),
                "video_frame_idx": int(click2),
                "video_time_s": float(ts_ms[int(click2)] / 1000.0),
                "implied_offset_s": float(ts_ms[int(click2)] / 1000.0 - float(baseline_on[0])),
                "frame_rate_fps": float(fps),
                "n_trials": int(len(baseline_on)),
                "clicked_at": _dt.datetime.now().isoformat(timespec="seconds"),
            }
            save_anchor(session_name, anchor)
            logger.info(
                "Anchor saved: trial 0 @ frame %d (video time %.3fs); implied offset = %.3fs",
                anchor["video_frame_idx"],
                anchor["video_time_s"],
                anchor["implied_offset_s"],
            )

    # Render montage
    montage_path = os.path.join(FIGS_DIR, f"{session_name}_barcode_montage.png")
    render_barcode_montage(
        session_name=session_name,
        anchor=anchor,
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
