"""tag_trials.py — Phase 2 of video sync: per-trial manual onset tagging.

Walks each trial in a session and lets the user verify-and-advance the
slope-fitted predicted onset, overriding it on a per-trial basis when the
prediction is wrong. Per-trial overrides are persisted to
{session}_video_sync.json on every keystroke that mutates state, so
crashes do not lose progress.

Requires {session}_video_sync.json to exist (run fit_sync.py first).

Keys:
    Left/Right     +-1 frame
    Shift+Left/Right (or PgUp/Down)  +-10 frames
    Ctrl+Left/Right  +-100 frames
    Enter   Save current frame as this trial's override, advance.
    S       Skip: advance without changing this trial's override state.
    D       Delete this trial's override (revert to slope-fit), advance.
    B       Back to previous trial.
    Q / Esc Save and quit.

Run:  py scripts/video/tag_trials.py --session 09092025
"""
import argparse
import gc
import json
import logging
import os
import sys
import tempfile
import time as _time
from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np

# matplotlib TkAgg setup (mirrors click_anchor.py for consistency).
os.environ["MPLBACKEND"] = "TkAgg"
import matplotlib
matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")

from visdetect.suite.loader import load_session
from visdetect.core.video_sync import (
    find_camera_files,
    load_camera_metadata,
    load_video_sync,
)
from visdetect.analysis.config import VIDEO_SYNC_DIR

# Reuse the eye-region crop from click_anchor.py (same subject, same camera).
import importlib.util as _ilu
_CA_SPEC = _ilu.spec_from_file_location(
    "click_anchor",
    os.path.join(os.path.dirname(__file__), "click_anchor.py"),
)
_CA = _ilu.module_from_spec(_CA_SPEC)
_CA_SPEC.loader.exec_module(_CA)
EYE_REGION_CROP_BG046 = _CA.EYE_REGION_CROP_BG046

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("tag_trials")


# ---------------------------------------------------------------------------
# Pure-logic state machine (unit-tested independently of the UI)
# ---------------------------------------------------------------------------


@dataclass
class TagState:
    """Per-trial tagging state. Immutable transitions: handlers return new instances."""
    trial_idx: int
    overrides: Dict[int, int]
    n_trials: int
    done: bool = False


def initial_resume_idx(overrides: Dict[int, int], n_trials: int) -> int:
    """Lowest trial index that has no override yet (resume-where-left-off)."""
    for i in range(n_trials):
        if i not in overrides:
            return i
    return n_trials


def handle_enter(state: TagState, current_frame: int) -> TagState:
    new_overrides = dict(state.overrides)
    new_overrides[state.trial_idx] = int(current_frame)
    new_idx = state.trial_idx + 1
    return TagState(
        trial_idx=new_idx,
        overrides=new_overrides,
        n_trials=state.n_trials,
        done=(new_idx >= state.n_trials),
    )


def handle_skip(state: TagState) -> TagState:
    new_idx = state.trial_idx + 1
    return TagState(
        trial_idx=new_idx,
        overrides=dict(state.overrides),
        n_trials=state.n_trials,
        done=(new_idx >= state.n_trials),
    )


def handle_delete(state: TagState) -> TagState:
    new_overrides = {
        k: v for k, v in state.overrides.items() if k != state.trial_idx
    }
    new_idx = state.trial_idx + 1
    return TagState(
        trial_idx=new_idx,
        overrides=new_overrides,
        n_trials=state.n_trials,
        done=(new_idx >= state.n_trials),
    )


def handle_back(state: TagState) -> TagState:
    new_idx = max(0, state.trial_idx - 1)
    return TagState(
        trial_idx=new_idx,
        overrides=dict(state.overrides),
        n_trials=state.n_trials,
        done=False,
    )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _sync_json_path(session_name: str) -> str:
    session_name = str(int(session_name)).zfill(8)
    return os.path.join(VIDEO_SYNC_DIR, f"{session_name}_video_sync.json")


def _persist_overrides(session_name: str, overrides: Dict[int, int]) -> None:
    """Read the sync JSON, write back with updated per_trial_overrides (atomic)."""
    path = _sync_json_path(session_name)
    with open(path, "r") as f:
        data = json.load(f)
    if "eye_cam" not in data:
        raise KeyError("sync JSON has no eye_cam entry; cannot persist overrides")
    # On-disk JSON object keys must be strings.
    data["eye_cam"]["per_trial_overrides"] = {
        str(k): int(v) for k, v in sorted(overrides.items())
    }
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_overrides(session_name: str) -> Dict[int, int]:
    """Read existing per_trial_overrides from the sync JSON (may be empty)."""
    data = load_video_sync(session_name) or {}
    raw = (data.get("eye_cam") or {}).get("per_trial_overrides") or {}
    return {int(k): int(v) for k, v in raw.items()}


def _slope_fit_frame(
    sync_json: dict, nidaq_baseline_on_s: float, fps: float,
) -> int:
    from visdetect.analysis.tagging import nidaq_to_frame_oriented
    eye = sync_json["eye_cam"]
    return nidaq_to_frame_oriented(
        nidaq_baseline_on_s, float(eye["slope"]), float(eye["offset"]), fps,
        eye.get("detection_method", "manual_slope_fit"))


# ---------------------------------------------------------------------------
# Interactive UI
# ---------------------------------------------------------------------------


def _run_tag_ui(
    session_name: str,
    video_path: str,
    baseline_on: np.ndarray,
    ts_ms: np.ndarray,
    sync_json: dict,
    n_trials: int,
    initial_state: TagState,
    fps: float,
) -> TagState:
    """Open the TkAgg per-trial UI and drive the state machine until quit/done.

    Returns the final TagState. Persists overrides on every state-changing
    keystroke (Enter, S, D).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    y0, y1, x0, x1 = EYE_REGION_CROP_BG046
    n_frames = int(len(ts_ms))

    state_ref = {"state": initial_state, "current_frame": 0}

    def _read_frame(fi: int) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            return np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray[y0:y1, x0:x1]

    def _start_frame_for(trial_idx: int) -> int:
        """Initial frame when entering trial_idx: override if present, else slope-fit."""
        if trial_idx in state_ref["state"].overrides:
            return int(state_ref["state"].overrides[trial_idx])
        return _slope_fit_frame(sync_json, float(baseline_on[trial_idx]), fps)

    state_ref["current_frame"] = _start_frame_for(initial_state.trial_idx)

    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1], hspace=0.1)
    ax_frame = fig.add_subplot(gs[0])
    ax_hud = fig.add_subplot(gs[1])
    ax_hud.axis("off")

    im = ax_frame.imshow(
        _read_frame(state_ref["current_frame"]),
        cmap="gray", vmin=0, vmax=255, interpolation="nearest",
    )
    ax_frame.set_xticks([]); ax_frame.set_yticks([])

    hud_text = ax_hud.text(
        0.02, 0.5, "", fontsize=9, family="monospace",
        verticalalignment="center", transform=ax_hud.transAxes,
    )

    start_time = _time.time()
    trials_processed = [0]  # closure-mutable counter

    def _refresh():
        st = state_ref["state"]
        if st.done:
            plt.close(fig)
            return
        fi = state_ref["current_frame"]
        im.set_data(_read_frame(fi))
        elapsed_min = (_time.time() - start_time) / 60.0
        rate_min_per_trial = (
            elapsed_min / trials_processed[0] if trials_processed[0] > 0 else 0.0
        )
        remaining_trials = st.n_trials - st.trial_idx
        eta_min = remaining_trials * rate_min_per_trial
        predicted_frame = _slope_fit_frame(
            sync_json, float(baseline_on[st.trial_idx]), fps
        )
        override_status = (
            f"ON  (frame {st.overrides[st.trial_idx]})"
            if st.trial_idx in st.overrides else "OFF (slope-fit)"
        )
        lines = [
            f"Tagging trial {st.trial_idx + 1} of {st.n_trials}  ({session_name})",
            f"NI baseline_on: {float(baseline_on[st.trial_idx]):.4f} s",
            f"Slope-fit predicted: frame {predicted_frame}",
            f"Current frame:        {fi}   (Delta vs predicted = {fi - predicted_frame:+d})",
            f"Override: {override_status}",
            f"Elapsed: {elapsed_min:.1f} min   ETA: {eta_min:.1f} min",
            "",
            "Arrows = +-1f  Shift+Arrows = +-10f  Ctrl+Arrows = +-100f",
            "Enter = save+advance  S = skip+advance  D = delete+advance  B = back  Q/Esc = save+quit",
        ]
        hud_text.set_text("\n".join(lines))
        fig.canvas.draw_idle()

    def on_key(event):
        key = event.key
        st = state_ref["state"]
        if key in ("q", "escape"):
            plt.close(fig); return
        # Frame stepping within the current trial.
        step = 0
        if key == "left":   step = -1
        elif key == "right": step = +1
        elif key in ("shift+left", "pageup"):   step = -10
        elif key in ("shift+right", "pagedown"): step = +10
        elif key in ("ctrl+left",):  step = -100
        elif key in ("ctrl+right",): step = +100
        if step != 0:
            state_ref["current_frame"] = int(np.clip(
                state_ref["current_frame"] + step, 0, n_frames - 1,
            ))
            _refresh()
            return

        if key == "enter":
            new_state = handle_enter(st, state_ref["current_frame"])
            trials_processed[0] += 1
            state_ref["state"] = new_state
            _persist_overrides(session_name, new_state.overrides)
            if new_state.done:
                logger.info("All %d trials reviewed; quitting.", st.n_trials)
                plt.close(fig); return
            state_ref["current_frame"] = _start_frame_for(new_state.trial_idx)
            _refresh(); return

        if key == "s":
            new_state = handle_skip(st)
            trials_processed[0] += 1
            state_ref["state"] = new_state
            _persist_overrides(session_name, new_state.overrides)
            if new_state.done:
                logger.info("All %d trials reviewed; quitting.", st.n_trials)
                plt.close(fig); return
            state_ref["current_frame"] = _start_frame_for(new_state.trial_idx)
            _refresh(); return

        if key == "d":
            new_state = handle_delete(st)
            trials_processed[0] += 1
            state_ref["state"] = new_state
            _persist_overrides(session_name, new_state.overrides)
            if new_state.done:
                logger.info("All %d trials reviewed; quitting.", st.n_trials)
                plt.close(fig); return
            state_ref["current_frame"] = _start_frame_for(new_state.trial_idx)
            _refresh(); return

        if key == "b":
            new_state = handle_back(st)
            state_ref["state"] = new_state
            state_ref["current_frame"] = _start_frame_for(new_state.trial_idx)
            _refresh(); return

    fig.canvas.mpl_connect("key_press_event", on_key)
    try:
        _refresh()
        plt.show()
    finally:
        cap.release()
        # Final save (covers Q/Esc case).
        _persist_overrides(session_name, state_ref["state"].overrides)

    return state_ref["state"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2: per-trial manual onset tagging.",
    )
    parser.add_argument(
        "--session", required=True, help="Session name (e.g. 09092025).",
    )
    args = parser.parse_args()
    session_name = str(int(args.session)).zfill(8)

    sync = load_video_sync(session_name)
    if sync is None:
        logger.error(
            "No %s_video_sync.json for %s. Run fit_sync.py first.",
            session_name, session_name,
        )
        return 2
    if "eye_cam" not in sync:
        logger.error("Sync JSON for %s has no eye_cam entry.", session_name)
        return 2
    method = (sync.get("eye_cam") or {}).get("detection_method", "")
    if method not in ("manual_slope_fit", "manual_multianchor"):
        logger.error(
            "Sync JSON for %s was not produced by fit_sync.py "
            "(detection_method=%r, expected 'manual_slope_fit' or "
            "'manual_multianchor'). Run fit_sync.py --session %s first.",
            session_name, method, session_name,
        )
        return 2

    sess = load_session(session_name)
    baseline_on = np.asarray(
        sess.ni_events.get("Baseline_ON", []), dtype=float
    )
    baseline_on = baseline_on[baseline_on > 0]
    n_task_trials = len(sess.trials)
    if n_task_trials > 0 and len(baseline_on) > n_task_trials:
        baseline_on = baseline_on[:n_task_trials]
    n_trials = int(len(baseline_on))
    if n_trials == 0:
        logger.error("No Baseline_ON events for %s.", session_name)
        return 2

    del sess; gc.collect()

    cam = find_camera_files(session_name)
    if "eye_cam" not in cam:
        logger.error("No eye_cam video for %s.", session_name)
        return 2
    video_path = cam["eye_cam"]["video"]
    meta_path = cam["eye_cam"]["metadata"]
    ts_ms, _, _ = load_camera_metadata(meta_path)

    if len(ts_ms) >= 2:
        fps = float(1000.0 / np.median(np.diff(ts_ms)))
    else:
        fps = 50.0

    overrides = _load_overrides(session_name)
    start_idx = initial_resume_idx(overrides, n_trials)
    if start_idx >= n_trials:
        logger.info(
            "All %d trials already have overrides. Nothing to do. "
            "Use B (back) inside the UI to re-review.",
            n_trials,
        )
        # Open anyway at trial 0 so user can navigate.
        start_idx = 0

    initial_state = TagState(
        trial_idx=start_idx, overrides=overrides, n_trials=n_trials,
    )
    logger.info(
        "Opening per-trial tag UI for %s: %d trials, resuming at trial %d, "
        "%d existing overrides.",
        session_name, n_trials, start_idx, len(overrides),
    )
    final_state = _run_tag_ui(
        session_name=session_name,
        video_path=video_path,
        baseline_on=baseline_on,
        ts_ms=ts_ms,
        sync_json=sync,
        n_trials=n_trials,
        initial_state=initial_state,
        fps=fps,
    )
    logger.info(
        "Tag UI exited. Final overrides: %d trials tagged.",
        len(final_state.overrides),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
