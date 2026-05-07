"""Interactive polygon ROI selector for video sync.

Shows a frame from each requested session's eye camera video.
Draw one or more polygon ROIs (e.g. triangle at top + region at bottom).

Controls:
    Left-click   — add vertex to current polygon
    Right-click  — undo last vertex
    Backspace    — undo last vertex
    N            — finish current polygon, start a new one
    Enter        — confirm all polygons for this session
    R            — reset all polygons
    S / Q        — skip this session

The selected ROIs are saved to a JSON file that batch_sync_sessions.py
will read automatically.

Usage:
    py scripts/video/select_roi.py [session_name ...]

    # Select ROI for specific sessions:
    py scripts/video/select_roi.py 01072025 02072025 03072025

    # Select ROI for all QC-passing sessions:
    py scripts/video/select_roi.py --all

Output:
    data/cache/video_sync/session_rois.json
"""

import os
import sys
import json
import argparse

# Force interactive backend BEFORE any matplotlib import
os.environ["MPLBACKEND"] = "TkAgg"
import matplotlib
matplotlib.use("TkAgg", force=True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "analysis_suite"))
sys.path.insert(0, _PROJECT_ROOT)

from src.visdetect.core.video_sync import find_camera_files, load_camera_metadata
from src.visdetect.analysis.config import VIDEO_SYNC_DIR
from src.visdetect.analysis.constants import VIDEO_SYNC_DEFAULT_EYE_ROI

# Re-assert interactive backend. visdetect imports (qc.py, tf_pulse.py, etc.)
# call matplotlib.use("Agg") at module level, overriding our TkAgg setting.
matplotlib.use("TkAgg", force=True)
plt.switch_backend("TkAgg")

_backend = matplotlib.get_backend()
if "agg" in _backend.lower() and "tk" not in _backend.lower():
    raise RuntimeError(
        f"Failed to activate interactive backend (got '{_backend}'). "
        "Ensure tkinter is installed: py -c \"import tkinter\""
    )

ROI_FILE = os.path.join(VIDEO_SYNC_DIR, "session_rois.json")

# Polygon colors — cycle through these for multiple polygons
_POLY_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

# Frame to sample (1 minute into session at 50fps)
SAMPLE_FRAME = 3000


def load_roi_overrides() -> dict:
    """Load existing per-session ROI overrides."""
    if os.path.exists(ROI_FILE):
        with open(ROI_FILE) as f:
            return json.load(f)
    return {}


def save_roi_overrides(data: dict):
    """Save per-session ROI overrides."""
    os.makedirs(os.path.dirname(ROI_FILE), exist_ok=True)
    with open(ROI_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved ROI overrides to {ROI_FILE}")


def get_frame(session_name: str, frame_idx: int = SAMPLE_FRAME):
    """Extract a single frame from the eye camera video."""
    import cv2
    files = find_camera_files(session_name)
    vid_path = files["eye_cam"]["video"]
    cap = cv2.VideoCapture(vid_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {vid_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def draw_default_roi(ax, roi):
    """Draw the default ROI for reference (rectangle or multi-polygon)."""
    from src.visdetect.core.video_sync import _is_rectangle, _is_multi_polygon

    if _is_rectangle(roi):
        y0, y1, x0, x1 = roi
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                              linewidth=1, edgecolor="yellow",
                              facecolor="none", linestyle="--", alpha=0.5)
        ax.add_patch(rect)
        ax.text(x0, y0 - 5, "default ROI", color="yellow", fontsize=8, alpha=0.5)
    elif _is_multi_polygon(roi):
        for i, poly_verts in enumerate(roi):
            pts = np.array(poly_verts)
            patch = MplPolygon(pts, closed=True, linewidth=1,
                               edgecolor="yellow", facecolor="none",
                               linestyle="--", alpha=0.5)
            ax.add_patch(patch)
            ax.text(pts[0, 0], pts[0, 1] - 5,
                    f"default #{i+1}", color="yellow", fontsize=8, alpha=0.5)
    else:
        # Single polygon
        pts = np.array(roi)
        patch = MplPolygon(pts, closed=True, linewidth=1,
                           edgecolor="yellow", facecolor="none",
                           linestyle="--", alpha=0.5)
        ax.add_patch(patch)
        ax.text(pts[0, 0], pts[0, 1] - 5,
                "default ROI", color="yellow", fontsize=8, alpha=0.5)


def _draw_existing_rois(ax, existing):
    """Draw existing ROI polygons in cyan."""
    if existing is None:
        return
    # Normalise: could be single polygon [[x,y],...] or multi [[[x,y],...],...]
    polys = _normalise_roi_data(existing)
    for i, poly_pts in enumerate(polys):
        pts = np.array(poly_pts)
        patch = MplPolygon(pts, closed=True, linewidth=2,
                           edgecolor="cyan", facecolor="cyan", alpha=0.15)
        ax.add_patch(patch)
        ax.text(pts[0, 0], pts[0, 1] - 10,
                f"existing #{i+1}", color="cyan", fontsize=8)


def _normalise_roi_data(data):
    """Normalise ROI data from JSON to list-of-polygons format.

    Handles both legacy single-polygon [[x,y], ...] and new multi-polygon
    [[[x,y], ...], [[x,y], ...]] formats.
    """
    if not data:
        return []
    # Check if it's a single polygon (list of [x,y] pairs)
    # vs multi-polygon (list of lists of [x,y] pairs)
    first = data[0]
    if isinstance(first, (int, float)):
        # Old rectangular format [y0, y1, x0, x1] — shouldn't happen but handle
        return [data]
    if isinstance(first[0], (int, float)):
        # Single polygon: [[x,y], [x,y], ...]
        return [data]
    # Multi-polygon: [[[x,y], ...], [[x,y], ...]]
    return data


def select_roi_interactive(session_name: str, existing_roi=None,
                           frame_idx=SAMPLE_FRAME):
    """Interactive multi-polygon ROI selection for one session.

    Returns list of polygons [[[x,y], ...], ...], or None if skipped.
    """
    frame = get_frame(session_name, frame_idx=frame_idx)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
    ax.set_title(
        f"Session {session_name}\n"
        f"Click vertices | N = finish polygon & start next | "
        f"Enter = confirm all | R = reset | S = skip\n"
        f"Right-click/Backspace = undo last point",
        fontsize=10,
    )
    draw_default_roi(ax, VIDEO_SYNC_DEFAULT_EYE_ROI)
    _draw_existing_rois(ax, existing_roi)

    # State: list of completed polygons + current in-progress polygon
    completed_polys = []        # list of list-of-(x,y)
    completed_patches = []      # matplotlib patches for completed polygons
    current_verts = []          # vertices of polygon being drawn
    current_scatter = ax.scatter([], [], c="red", s=50, zorder=5)
    current_patch = [None]      # mutable ref for in-progress polygon patch
    result = [None]
    done = [False]

    def _color(idx):
        return _POLY_COLORS[idx % len(_POLY_COLORS)]

    def _update_status():
        n_done = len(completed_polys)
        n_verts = len(current_verts)
        status = f"Polygons: {n_done} done"
        if n_verts > 0:
            status += f" | drawing #{n_done+1} ({n_verts} pts)"
        for t in list(ax.texts):
            if getattr(t, '_is_status', False):
                t.remove()
        txt = ax.text(0.01, 0.01, status, transform=ax.transAxes,
                      fontsize=9, color="white",
                      bbox=dict(boxstyle="round,pad=0.3",
                                facecolor="black", alpha=0.7),
                      verticalalignment="bottom")
        txt._is_status = True
        fig.canvas.draw_idle()

    def _redraw_current():
        color = _color(len(completed_polys))
        if len(current_verts) > 0:
            xs = [v[0] for v in current_verts]
            ys = [v[1] for v in current_verts]
            current_scatter.set_offsets(np.column_stack([xs, ys]))
            current_scatter.set_facecolor(color)
        else:
            current_scatter.set_offsets(np.empty((0, 2)))

        if current_patch[0] is not None:
            current_patch[0].remove()
            current_patch[0] = None

        if len(current_verts) >= 3:
            poly = MplPolygon(current_verts, closed=True, linewidth=2,
                              edgecolor=color, facecolor=color, alpha=0.2)
            ax.add_patch(poly)
            current_patch[0] = poly

        _update_status()

    def _commit_current_polygon():
        """Finish the current polygon and add it to completed list."""
        if len(current_verts) < 3:
            return False
        color = _color(len(completed_polys))
        # Create a permanent patch for this polygon
        poly = MplPolygon(list(current_verts), closed=True, linewidth=2,
                          edgecolor=color, facecolor=color, alpha=0.25)
        ax.add_patch(poly)
        completed_patches.append(poly)
        completed_polys.append(list(current_verts))
        # Clear current state
        current_verts.clear()
        if current_patch[0] is not None:
            current_patch[0].remove()
            current_patch[0] = None
        current_scatter.set_offsets(np.empty((0, 2)))
        n = len(completed_polys)
        print(f"    Polygon #{n}: {completed_polys[-1]}")
        _update_status()
        return True

    def on_click(event):
        if event.inaxes != ax or done[0]:
            return
        if event.button == 1:  # Left click: add vertex
            current_verts.append(
                (int(round(event.xdata)), int(round(event.ydata))))
            _redraw_current()
        elif event.button == 3:  # Right click: undo
            if current_verts:
                current_verts.pop()
                _redraw_current()

    def on_key(event):
        if done[0]:
            return
        key = event.key.lower() if event.key else ""

        if key == "enter":
            # Commit current polygon if in progress, then finish
            if len(current_verts) >= 3:
                _commit_current_polygon()
            if completed_polys:
                result[0] = [list(p) for p in completed_polys]
            done[0] = True
            plt.close(fig)
        elif key == "n":
            # Finish current polygon, start a new one
            if _commit_current_polygon():
                print(f"    Starting polygon #{len(completed_polys)+1}...")
            else:
                print("    Need at least 3 vertices to finish a polygon")
        elif key == "r":
            # Reset everything
            current_verts.clear()
            completed_polys.clear()
            for p in completed_patches:
                p.remove()
            completed_patches.clear()
            _redraw_current()
        elif key in ("s", "q"):
            done[0] = True
            plt.close(fig)
        elif key == "backspace":
            if current_verts:
                current_verts.pop()
                _redraw_current()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    _update_status()
    plt.show(block=True)

    return result[0]


def main():
    parser = argparse.ArgumentParser(
        description="Interactive polygon ROI selector for video sync")
    parser.add_argument("sessions", nargs="*",
                        help="Session names (DDMMYYYY) to select ROIs for")
    parser.add_argument("--all", action="store_true",
                        help="Select ROIs for all QC-passing sessions")
    parser.add_argument("--frame", type=int, default=SAMPLE_FRAME,
                        help="Frame index to display (default: 3000)")
    args = parser.parse_args()

    sample_frame = args.frame

    if args.all:
        from loader import load_staging_manifest
        manifest = load_staging_manifest(qc_only=True)
        sessions = [
            str(int(row["session_name"])).zfill(8)
            for _, row in manifest.iterrows()
        ]
    elif args.sessions:
        sessions = [str(int(s)).zfill(8) for s in args.sessions]
    else:
        parser.print_help()
        return

    overrides = load_roi_overrides()
    print(f"Loaded {len(overrides)} existing ROI overrides from {ROI_FILE}")

    for sname in sessions:
        existing = overrides.get(sname)
        if existing:
            polys = _normalise_roi_data(existing)
            print(f"\n{sname}: {len(polys)} existing polygon(s)")

        print(f"\nOpening ROI selector for session {sname}...")
        try:
            roi = select_roi_interactive(sname, existing_roi=existing,
                                        frame_idx=sample_frame)
        except FileNotFoundError as e:
            print(f"  Skipping {sname}: {e}")
            continue

        if roi is not None:
            overrides[sname] = roi
            print(f"  Saved {len(roi)} polygon(s)")
            save_roi_overrides(overrides)
        else:
            print(f"  Skipped (no ROI set)")

    print(f"\nDone. {len(overrides)} session ROIs saved to {ROI_FILE}")


if __name__ == "__main__":
    main()
