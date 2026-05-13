"""Corneal spatial diagnostic — does the grating appear as within-frame spatial structure?

Tests whether the corneal reflection ROI shows detectable spatial variance (grating stripes)
at Baseline_ON, independent of temporal drift.  Spatial variance is the correct signal to
use for onset detection because:
  - The baseline grating has stochastic 50ms TF pulses at mean ~1Hz
  - At 50fps, per-frame drift is ~0.6px — below temporal variance threshold on slow pulses
  - But the APPEARANCE of grating stripes at onset is instantaneous and phase-independent

Produces a grid figure: N trials × 2 columns (before / after Baseline_ON frame crop).
Reports spatial variance (within-frame std) before vs after for each trial.

Usage:
    py scripts/video/corneal_spatial_diagnostic.py
    py scripts/video/corneal_spatial_diagnostic.py --session 27062025 --n-trials 12
"""

import os
import sys
import argparse
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "analysis_suite"))

from loader import load_session
from src.visdetect.core.video_sync import (
    find_camera_files,
    load_camera_metadata,
    detect_onsets_variance,
    fit_clock_model,
    nidaq_to_camera,
    load_or_compute_coarse_offset,
    auto_calibrate_corneal_roi,
    load_corneal_cal,
    load_corneal_mask,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────
CORNEAL_EYE_ROI = {
    "09092025": (305, 335, 433, 468),   # validated
    "27062025": (240, 300, 310, 420),   # Wider ROI for spatial frequency
    "03072025": (257, 277, 377, 397),   # tight 20×20 box on autocal-found reflection (y:265-269,x:384-389)
    "14082025": (302, 332, 433, 468),   # validated
    "29082025": (318, 338, 419, 439),   # tight 20×20 box on autocal-found reflection (y:314-343,x:415-444)
}

# Radius (px) of the circular mask within each ROI bounding box.
# Excludes corners: bottom-left tear-film blob and upper-left pupil edge.
# None → use min(h, w) // 2  (inscribed circle, same for all sessions).
CORNEAL_CIRCLE_RADIUS = {
    "09092025": 12,   # 12px in 30×35 patch — excludes corner artefacts
    "27062025": 8,    # tight ROI — 8px radius in 20×20 box
    "03072025": 8,    # tight ROI — 8px radius in 20×20 box
    "14082025": 12,   # initial estimate; adjust after roi-overlay check
    "29082025": 8,    # tight ROI — 8px radius in 20×20 box
}

# Per-session detection sensitivity overrides.
# Default: (thresh_nsigma=3.0, min_step_ratio=1.25) — strict, works when SV step is large.
# Override when the true grating onset produces a sharp but small-amplitude SV step:
#   thresh_nsigma  : threshold = bl_mean + N×bl_std  (lower → accepts smaller steps)
#   min_step_ratio : post_mean / bl_mean must exceed this (lower → accepts weaker ratios)
_DEFAULT_DETECT_PARAMS = (3.0, 1.25)
CORNEAL_DETECT_PARAMS = {
    # 14082025: genuine onset step is sharp but low amplitude; strict threshold misses it
    # and a later spurious event (blink/eye movement) gets accepted instead.
    "14082025": (2.0, 1.10),
    # 03072025: very weak signal — eye movements blur the reflection across frames.
    # Use tight ROI (20×20) + relaxed threshold to extract available signal.
    "03072025": (1.5, 1.05),
    # 27062025: low-amplitude onset step; same relaxation needed as 14082025.
    "27062025": (1.5, 1.05),
    # 29082025: low-amplitude onset step.
    "29082025": (2.0, 1.10),
}

# Larger "context" ROI to show the wider eye region for reference
CONTEXT_EYE_ROI = {
    "09092025": (260, 380, 380, 530),
    "27062025": (185, 345, 307, 502),   # ±80y, ±97x around eye centre (y=265, x=404)
    "03072025": (160, 320, 317, 512),   # ±80y, ±97x around eye centre (y=240, x=414)
    "14082025": (257, 407, 358, 543),   # ±60y, ±75x around corneal center (y=317, x=450)
    "29082025": (266, 416, 353, 538),   # ±60y, ±75x around corneal center (y=326, x=445)
}

LOCAL_VIDEO_DIR = os.path.join(_PROJECT_ROOT, "data", "videos")
SESSION_VIDEO_MAP = {
    "09092025": "BG_046_090925_Eye_cam.mp4",
    "27062025": "BG_046_270625_Eye_cam.mp4",
    "03072025": "BG_046_030725_Eye_cam.mp4",
    "14082025": "BG_046_140825_Eye_cam.mp4",
    "29082025": "BG_046_290825_Eye_cam.mp4",
}
FIG_DIR = os.path.join(_PROJECT_ROOT, "figures", "video_sync", "corneal_spatial")

# Broad (loose) eye bounding boxes — one per subject, not per session.
# These are intentionally generous (~200-300px) to capture the full eye
# regardless of small session-to-session head position variation.
# The auto-calibration pipeline then finds the exact pupil + corneal
# reflection within this region automatically.
# Format: (y0, y1, x0, x1) in full-frame pixel coordinates.
BROAD_EYE_ROI = {
    "BG_046": (180, 420, 330, 590),  # derived from validated per-session ROIs
}

DELTA_MS = 100.0   # how many ms before/after Baseline_ON to sample


def build_sync_model(session_name, video_path, meta_path, baseline_on):
    """Build a screen-glow clock model, or fall back to coarse offset only."""
    coarse_offset = load_or_compute_coarse_offset(
        session_name, video_path, meta_path, baseline_on
    )
    logger.info(f"  Coarse offset: {coarse_offset:.1f}s")

    onset_result = detect_onsets_variance(
        video_path, meta_path, baseline_on, coarse_offset, progress=True
    )
    n_detected = len(onset_result.detected_cam_s)
    logger.info(f"  Screen-glow detections: {n_detected}/{len(baseline_on)}")

    if n_detected >= 10:
        sync = fit_clock_model(
            onset_result.detected_cam_s,
            onset_result.detected_nidaq_s,
            n_baseline_on=len(baseline_on),
        )
        logger.info(f"  Clock model: RMSE={sync.rmse_ms:.1f}ms, "
                    f"slope={sync.slope:.8f}, offset={sync.offset:.3f}s")
        return sync.slope, sync.offset, sync.rmse_ms
    else:
        logger.warning("  Too few screen-glow anchors — using coarse offset, slope=1.0")
        return 1.0, coarse_offset, 999.0


def seek_frame(cap, ts_ms, target_ms):
    """Seek video to the frame nearest target_ms and return it as float32 grayscale."""
    import cv2
    idx = int(np.argmin(np.abs(ts_ms - target_ms)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        return None, idx, ts_ms[idx]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return gray, idx, ts_ms[idx]


def _circle_mask(h, w, radius=None):
    """Boolean mask for a circle inscribed in an h×w patch."""
    cy, cx = h / 2.0, w / 2.0
    if radius is None:
        radius = min(h, w) / 2.0
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2


def spatial_variance(gray, roi, radius=None):
    """Within-frame spatial variance (std) of pixels in the specified region.

    Parameters
    ----------
    gray : np.ndarray (H, W), float32
        Grayscale frame.
    roi : (y0, y1, x0, x1) tuple  OR  2-D boolean np.ndarray
        - Tuple: bounding box. A circular mask of the given radius (or
          inscribed circle if None) is applied within the box.
        - Boolean array (shape H×W): pixels at True positions are used
          directly.  radius is ignored.  Used with auto-calibrated masks.
    radius : float or None
        Inscribed circle radius when roi is a tuple (None = inscribed).
    """
    if isinstance(roi, np.ndarray) and roi.ndim == 2:
        cols = np.any(roi, axis=0)
        rows = np.any(roi, axis=1)
        if not np.any(rows) or not np.any(cols):
            return 0.0
        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]
        patch = gray[y0:y1+1, x0:x1+1]
    else:
        y0, y1, x0, x1 = roi
        patch = gray[y0:y1, x0:x1]

    if patch.shape[0] < 4 or patch.shape[1] < 4:
        return 0.0
    
    # 2D FFT to isolate spatial frequencies
    patch = patch - np.mean(patch)
    fft_mag = np.abs(np.fft.rfft2(patch))
    
    # Remove DC and very low frequencies (large gradients like pupil edges)
    # We want higher frequencies corresponding to the grating stripes
    # Setting the top-left 3x3 region (low frequencies) to zero
    r, c = fft_mag.shape
    fft_mag[:min(r, 2), :min(c, 2)] = 0.0
    
    return float(np.sum(fft_mag))


def run_auto_calibrate(session_name, subject="BG_046", force=False):
    """Run automated corneal ROI calibration for a session.

    Detects the pupil automatically (using a loose per-subject bounding box),
    then builds a data-driven corneal reflection mask via std(post-pre diff)
    across Baseline_ON transitions.

    The result is cached in data/cache/video_sync/corneal_cal/ and will be
    used automatically by run_full_session(auto_calibrate=True).

    Produces a diagnostic figure:
      figures/video_sync/corneal_spatial/{session}_autocal_diagnostic.png
    showing (A) the pupil detection overlay, (B) the std(diff) heatmap,
    (C) the corneal mask overlaid on the eye frame, (D) a summary table.
    """
    import cv2

    os.makedirs(FIG_DIR, exist_ok=True)

    logger.info(f"Auto-calibrating corneal ROI for session {session_name}...")
    sess = load_session(session_name)
    baseline_on = np.asarray(
        sess.ni_events.get("Baseline_ON", []), dtype=float
    ).flatten()
    baseline_on = baseline_on[baseline_on > 0]
    n_task = len(getattr(sess, "trials", None) or [])
    if n_task > 0 and len(baseline_on) > n_task:
        baseline_on = baseline_on[:n_task]
    logger.info(f"  {len(baseline_on)} Baseline_ON events")

    fname = SESSION_VIDEO_MAP.get(session_name)
    if fname is None:
        raise ValueError(f"No video mapping for {session_name}")
    video_path = os.path.join(LOCAL_VIDEO_DIR, fname)
    meta_path = video_path.rsplit(".", 1)[0] + ".csv"
    if not os.path.exists(meta_path):
        cam_files = find_camera_files(session_name)
        meta_path = cam_files["eye_cam"]["metadata"]

    coarse_offset = load_or_compute_coarse_offset(
        session_name, video_path, meta_path, baseline_on
    )

    broad_roi = BROAD_EYE_ROI.get(subject)
    if broad_roi is None:
        logger.warning(
            f"No BROAD_EYE_ROI defined for subject '{subject}'. "
            "Searching full frame — may be slow and imprecise."
        )

    cal = auto_calibrate_corneal_roi(
        session_name, video_path, meta_path, baseline_on,
        rough_offset_s=coarse_offset,
        broad_eye_roi=broad_roi,
        force=force,
    )

    if cal is None:
        logger.error(f"Auto-calibration failed for {session_name}")
        return

    # Load the cached mask for visualization
    from src.visdetect.core.video_sync import load_corneal_mask as _load_mask
    mask = _load_mask(session_name)

    # ── Diagnostic figure ─────────────────────────────────────────────
    # Read the representative frame (same one used during calibration)
    ts_ms, _, _ = load_camera_metadata(meta_path)
    target_nidaq = float(np.percentile(baseline_on, 30))
    target_cam_ms = (target_nidaq - coarse_offset) * 1000.0
    fi = int(np.searchsorted(ts_ms, target_cam_ms))
    fi = max(0, min(fi, len(ts_ms) - 1))
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, bgr = cap.read()
    cap.release()

    if not ret:
        logger.warning("Could not read frame for diagnostic figure")
        return

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape

    pupil_cy, pupil_cx = cal["pupil_center"]
    pupil_r = cal["pupil_radius"]
    corneal_bbox = cal.get("corneal_bbox")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        f"Corneal auto-calibration — {session_name} ({subject})\n"
        f"Pupil: centre=({pupil_cy:.0f},{pupil_cx:.0f}) r={pupil_r:.0f}px  |  "
        f"Mask: {cal['mask_area_px']}px ({cal['mask_quality']})  "
        f"mean_std={cal['best_component_mean_std']:.2f}",
        fontsize=9,
    )

    # Panel A: frame with pupil circle + broad ROI + mask overlay
    scale = 0.35
    thumb_h = max(1, int(H * scale))
    thumb_w = max(1, int(W * scale))
    import cv2 as _cv2
    thumb_gray = _cv2.resize(gray, (thumb_w, thumb_h), interpolation=_cv2.INTER_AREA)
    thumb_rgb = np.stack([thumb_gray / thumb_gray.max()] * 3, axis=-1).copy()
    # Mask overlay in cyan
    if mask is not None:
        mask_ds = _cv2.resize(mask.astype(np.uint8), (thumb_w, thumb_h),
                              interpolation=_cv2.INTER_NEAREST).astype(bool)
        thumb_rgb[mask_ds, 0] = 0.0
        thumb_rgb[mask_ds, 1] = 1.0
        thumb_rgb[mask_ds, 2] = 1.0
    # Broad eye ROI in green
    if broad_roi:
        y0b, y1b, x0b, x1b = broad_roi
        _cv2.rectangle(
            (thumb_rgb * 255).astype(np.uint8),
            (int(x0b * scale), int(y0b * scale)),
            (int(x1b * scale), int(y1b * scale)),
            (0, 200, 0), 1,
        )
    axes[0].imshow(np.clip(thumb_rgb, 0, 1), interpolation="nearest")
    # Draw pupil circle manually on the plot
    theta = np.linspace(0, 2 * np.pi, 100)
    axes[0].plot(
        (pupil_cx + pupil_r * np.cos(theta)) * scale,
        (pupil_cy + pupil_r * np.sin(theta)) * scale,
        "r-", linewidth=1.5, label=f"pupil r={pupil_r:.0f}px",
    )
    axes[0].set_title(
        "A. Frame + pupil (red) + mask (cyan)\nGreen = broad search ROI",
        fontsize=8,
    )
    axes[0].legend(fontsize=7)
    axes[0].axis("off")

    # Panel B: zoomed eye region — std(diff) heatmap
    bb = cal.get("corneal_bbox") or (
        max(0, int(pupil_cy) - 80), min(H, int(pupil_cy) + 80),
        max(0, int(pupil_cx) - 80), min(W, int(pupil_cx) + 80),
    )
    by0, by1, bx0, bx1 = bb

    # Reload avg_diff from the mask cache npz
    from src.visdetect.core.video_sync import VIDEO_SYNC_DIR
    mask_cache = os.path.join(VIDEO_SYNC_DIR, "corneal_cal",
                              f"{session_name}_corneal_mask.npz")
    if os.path.exists(mask_cache):
        mdata = np.load(mask_cache, allow_pickle=True)
        info_dict = mdata["info"].item()
        avg_diff_crop = info_dict.get("avg_diff")
        bb_stored = info_dict.get("bb")
        if avg_diff_crop is not None and bb_stored is not None:
            sbb_y0, sbb_y1, sbb_x0, sbb_x1 = bb_stored
            axes[1].imshow(avg_diff_crop, cmap="hot", interpolation="nearest")
            axes[1].set_title(
                f"B. std(post-pre diff) in search box\n"
                f"[{sbb_y0}:{sbb_y1}, {sbb_x0}:{sbb_x1}]",
                fontsize=8,
            )
            # Overlay mask within bounding box
            if mask is not None:
                mask_crop = mask[sbb_y0:sbb_y1, sbb_x0:sbb_x1]
                ys_l, xs_l = np.where(mask_crop)
                if len(ys_l):
                    axes[1].scatter(xs_l, ys_l, s=1, c="cyan", alpha=0.7,
                                    label="mask")
                    axes[1].legend(fontsize=7)
            axes[1].axis("off")
        else:
            axes[1].text(0.5, 0.5, "no avg_diff cached",
                         transform=axes[1].transAxes, ha="center")
    else:
        axes[1].text(0.5, 0.5, "mask cache not found",
                     transform=axes[1].transAxes, ha="center")

    # Panel C: tight eye crop with mask overlay
    eye_gray = gray[max(0, int(pupil_cy) - 60):min(H, int(pupil_cy) + 60),
                    max(0, int(pupil_cx) - 70):min(W, int(pupil_cx) + 70)]
    ey0 = max(0, int(pupil_cy) - 60)
    ex0 = max(0, int(pupil_cx) - 70)
    eye_rgb = np.stack([eye_gray / max(eye_gray.max(), 1)] * 3, axis=-1).copy()
    if mask is not None:
        mask_eye = mask[ey0:ey0 + eye_gray.shape[0],
                        ex0:ex0 + eye_gray.shape[1]]
        eye_rgb[mask_eye, 0] = 0.0
        eye_rgb[mask_eye, 1] = 1.0
        eye_rgb[mask_eye, 2] = 1.0
    axes[2].imshow(np.clip(eye_rgb, 0, 1), interpolation="nearest")
    # Pupil circle relative to crop
    axes[2].plot(
        (pupil_cx - ex0) + pupil_r * np.cos(theta),
        (pupil_cy - ey0) + pupil_r * np.sin(theta),
        "r-", linewidth=1.5,
    )
    axes[2].set_title(
        "C. Zoomed eye: mask in cyan, pupil in red",
        fontsize=8,
    )
    axes[2].axis("off")

    # Panel D: summary table
    axes[3].axis("off")
    summary = [
        f"Session       : {session_name}",
        f"Subject       : {subject}",
        f"Quality       : {cal['mask_quality'].upper()}",
        f"Pupil centre  : ({pupil_cy:.0f}, {pupil_cx:.0f}) px",
        f"Pupil radius  : {pupil_r:.1f} px",
        f"Pupil area    : {cal['pupil_area_px']:.0f} px²",
        f"Pupil circ.   : {cal['pupil_circularity']:.2f}",
        f"Mask area     : {cal['mask_area_px']} px²",
        f"Mean std(diff): {cal['best_component_mean_std']:.2f}",
        f"Corneal bbox  : {corneal_bbox}",
        f"N transitions : {cal['n_transitions_used']}",
        f"Search mask   : {cal['search_mask_area_px']} px",
    ]
    color = {"good": "#2e7d32", "marginal": "#f57f17", "failed": "#c62828"}
    axes[3].text(
        0.05, 0.95, "\n".join(summary),
        transform=axes[3].transAxes,
        fontsize=9, fontfamily="monospace", verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=color.get(cal["mask_quality"], "#bbb"),
            alpha=0.15,
        ),
    )
    axes[3].set_title("D. Calibration summary", fontsize=8)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"{session_name}_autocal_diagnostic.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")

    print(f"\n{'='*55}")
    print(f"AUTO-CALIBRATION — {session_name}")
    print(f"{'='*55}")
    for line in summary:
        print(f"  {line}")
    print(f"\nDiagnostic figure: {out}")


def run_diagnostic(session_name, n_trials=12, trial_offset=20, delta_ms=DELTA_MS):
    """Extract before/after frames for N trials and report spatial variance."""
    import cv2

    os.makedirs(FIG_DIR, exist_ok=True)

    # Load session
    logger.info(f"Loading session {session_name}...")
    sess = load_session(session_name)
    baseline_on = np.asarray(
        sess.ni_events.get("Baseline_ON", []), dtype=float
    ).flatten()
    baseline_on = baseline_on[baseline_on > 0]
    logger.info(f"  {len(baseline_on)} Baseline_ON events")

    # Video
    fname = SESSION_VIDEO_MAP.get(session_name)
    if fname is None:
        raise ValueError(f"No video mapping for {session_name}")
    video_path = os.path.join(LOCAL_VIDEO_DIR, fname)
    meta_path = video_path.rsplit(".", 1)[0] + ".csv"
    if not os.path.exists(meta_path):
        logger.info("  Local metadata CSV not found — trying network camera root...")
        cam_files = find_camera_files(session_name)
        meta_path = cam_files["eye_cam"]["metadata"]
    logger.info(f"  Metadata: {meta_path}")

    ts_ms, _, _ = load_camera_metadata(meta_path)
    logger.info(f"  Video: {len(ts_ms)} frames, {ts_ms[-1]/1000:.0f}s")

    # Build sync model
    logger.info("Building sync model...")
    slope, offset, rmse_ms = build_sync_model(
        session_name, video_path, meta_path, baseline_on
    )

    # Pick evenly spaced trials from the middle of the session
    # (skip first/last 10% to avoid edge effects)
    n_avail = len(baseline_on)
    start = max(0, int(n_avail * 0.1))
    end = min(n_avail - 1, int(n_avail * 0.9))
    # Sample n_trials evenly across [start, end], offset from trial_offset
    indices = np.linspace(start + trial_offset, end, n_trials, dtype=int)
    indices = np.clip(indices, 0, n_avail - 1)

    roi = CORNEAL_EYE_ROI.get(session_name, list(CORNEAL_EYE_ROI.values())[0])
    radius = CORNEAL_CIRCLE_RADIUS.get(session_name)
    ctx_roi = CONTEXT_EYE_ROI.get(session_name, list(CONTEXT_EYE_ROI.values())[0])
    y0, y1, x0, x1 = roi
    cy0, cy1, cx0, cx1 = ctx_roi

    cap = cv2.VideoCapture(video_path)

    results = []
    for trial_i, idx in enumerate(indices):
        nidaq_t = baseline_on[idx]
        # Convert to camera ms using sync model
        cam_ms = nidaq_to_camera(np.array([nidaq_t]), slope, offset)[0]

        before_ms = cam_ms - delta_ms
        after_ms  = cam_ms + delta_ms

        frame_before, fidx_b, t_b = seek_frame(cap, ts_ms, before_ms)
        frame_after,  fidx_a, t_a = seek_frame(cap, ts_ms, after_ms)

        if frame_before is None or frame_after is None:
            logger.warning(f"  Trial {idx}: frame read failed, skipping")
            continue

        sv_before = spatial_variance(frame_before, roi, radius)
        sv_after  = spatial_variance(frame_after, roi, radius)
        ratio = sv_after / sv_before if sv_before > 0 else np.nan

        results.append({
            "trial_idx": int(idx),
            "nidaq_t": nidaq_t,
            "cam_ms": cam_ms,
            "sv_before": sv_before,
            "sv_after": sv_after,
            "ratio": ratio,
            "patch_before": frame_before[y0:y1, x0:x1].copy(),
            "patch_after":  frame_after[y0:y1, x0:x1].copy(),
            "ctx_before": frame_before[cy0:cy1, cx0:cx1].copy(),
            "ctx_after":  frame_after[cy0:cy1, cx0:cx1].copy(),
            "dt_before_ms": t_b - cam_ms,
            "dt_after_ms":  t_a - cam_ms,
        })
        logger.info(f"  Trial {idx}: SV before={sv_before:.2f}, after={sv_after:.2f}, "
                    f"ratio={ratio:.2f}  (Δt_b={t_b-cam_ms:+.0f}ms, Δt_a={t_a-cam_ms:+.0f}ms)")

    cap.release()

    if not results:
        logger.error("No valid frames extracted — check video path and sync model.")
        return

    # ── Figure 1: Before/After grid (tight corneal ROI patches) ─────────────
    n = len(results)
    ncols = 4   # before | after | before | after (two trial pairs per row)
    nrows = int(np.ceil(n / 2))

    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.2))
    fig1.suptitle(
        f"Corneal ROI: spatial structure before/after Baseline_ON\n"
        f"Session {session_name}  |  ROI {roi}  |  ±{delta_ms:.0f}ms  |  "
        f"Clock RMSE={rmse_ms:.0f}ms",
        fontsize=10
    )

    for i, r in enumerate(results):
        row = i // 2
        col_base = (i % 2) * 2

        ax_b = axes1[row, col_base]
        ax_a = axes1[row, col_base + 1]

        vmin = min(r["patch_before"].min(), r["patch_after"].min())
        vmax = max(r["patch_before"].max(), r["patch_after"].max())
        vmin = max(0, vmin - 5)
        vmax = min(255, vmax + 5)

        ax_b.imshow(r["patch_before"], cmap="gray", vmin=vmin, vmax=vmax,
                    interpolation="nearest")
        ax_b.set_title(
            f"T{r['trial_idx']} BEFORE\n"
            f"Δt={r['dt_before_ms']:+.0f}ms  SV={r['sv_before']:.1f}",
            fontsize=7
        )
        ax_b.axis("off")

        ax_a.imshow(r["patch_after"], cmap="gray", vmin=vmin, vmax=vmax,
                    interpolation="nearest")
        ax_a.set_title(
            f"T{r['trial_idx']} AFTER\n"
            f"Δt={r['dt_after_ms']:+.0f}ms  SV={r['sv_after']:.1f}  ×{r['ratio']:.1f}",
            fontsize=7,
            color="green" if r["ratio"] > 1.5 else ("orange" if r["ratio"] > 1.1 else "red"),
        )
        ax_a.axis("off")

    # Hide unused axes
    for i in range(n, nrows * 2):
        row = i // 2
        col_base = (i % 2) * 2
        axes1[row, col_base].axis("off")
        axes1[row, col_base + 1].axis("off")

    plt.tight_layout()
    out1 = os.path.join(FIG_DIR, f"{session_name}_corneal_before_after.png")
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    logger.info(f"Saved: {out1}")

    # ── Figure 2: Context (wider eye) patches for reference ─────────────────
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5))
    fig2.suptitle(
        f"Context ROI (wider eye): before/after Baseline_ON\n"
        f"Session {session_name}  |  Context ROI {ctx_roi}",
        fontsize=10
    )
    for i, r in enumerate(results):
        row = i // 2
        col_base = (i % 2) * 2
        ax_b = axes2[row, col_base]
        ax_a = axes2[row, col_base + 1]

        vmin = min(r["ctx_before"].min(), r["ctx_after"].min())
        vmax = max(r["ctx_before"].max(), r["ctx_after"].max())
        vmin = max(0, vmin - 5)
        vmax = min(255, vmax + 5)

        ax_b.imshow(r["ctx_before"], cmap="gray", vmin=vmin, vmax=vmax,
                    interpolation="nearest")
        ax_b.set_title(f"T{r['trial_idx']} BEFORE  SV={r['sv_before']:.1f}", fontsize=7)
        ax_b.axis("off")

        ax_a.imshow(r["ctx_after"], cmap="gray", vmin=vmin, vmax=vmax,
                    interpolation="nearest")
        ax_a.set_title(f"T{r['trial_idx']} AFTER  SV={r['sv_after']:.1f}  ×{r['ratio']:.1f}",
                       fontsize=7,
                       color="green" if r["ratio"] > 1.5 else ("orange" if r["ratio"] > 1.1 else "red"))
        ax_a.axis("off")

    for i in range(n, nrows * 2):
        row = i // 2
        col_base = (i % 2) * 2
        axes2[row, col_base].axis("off")
        axes2[row, col_base + 1].axis("off")

    plt.tight_layout()
    out2 = os.path.join(FIG_DIR, f"{session_name}_context_before_after.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"Saved: {out2}")

    # ── Figure 3: Summary — spatial variance ratio distribution ─────────────
    ratios = np.array([r["ratio"] for r in results if np.isfinite(r["ratio"])])
    sv_b = np.array([r["sv_before"] for r in results])
    sv_a = np.array([r["sv_after"]  for r in results])

    fig3, axes3 = plt.subplots(1, 3, figsize=(12, 4))
    fig3.suptitle(
        f"Spatial variance summary — {session_name}  (n={len(results)} trials)",
        fontsize=11
    )

    ax = axes3[0]
    ax.hist(ratios, bins=15, color="steelblue", edgecolor="white")
    ax.axvline(1.0, color="red", linestyle="--", label="ratio=1 (no change)")
    ax.axvline(np.median(ratios), color="orange", linestyle="-",
               label=f"median={np.median(ratios):.2f}")
    ax.set_xlabel("SV after / SV before")
    ax.set_ylabel("Count")
    ax.set_title("Spatial variance ratio (after/before)")
    ax.legend(fontsize=8)

    ax = axes3[1]
    ax.scatter(sv_b, sv_a, alpha=0.7, color="steelblue")
    lim = max(max(sv_b), max(sv_a)) * 1.05
    ax.plot([0, lim], [0, lim], "r--", label="no change")
    ax.set_xlabel("SV before Baseline_ON")
    ax.set_ylabel("SV after Baseline_ON")
    ax.set_title("Before vs After spatial variance")
    ax.legend(fontsize=8)

    ax = axes3[2]
    # Sorted by before SV to see if trials with lower baseline have larger ratio
    order = np.argsort(sv_b)
    ax.plot(sv_b[order], ratios[order], "o-", color="steelblue", markersize=4)
    ax.axhline(1.0, color="red", linestyle="--")
    ax.set_xlabel("SV before (sorted)")
    ax.set_ylabel("SV ratio")
    ax.set_title("Ratio vs baseline spatial variance")

    plt.tight_layout()
    out3 = os.path.join(FIG_DIR, f"{session_name}_sv_summary.png")
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    logger.info(f"Saved: {out3}")

    # ── Console summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SPATIAL VARIANCE DIAGNOSTIC — {session_name}")
    print(f"{'='*60}")
    print(f"Trials analysed : {len(results)}")
    print(f"Clock RMSE      : {rmse_ms:.0f}ms")
    print(f"Corneal ROI     : {roi}  ({y1-y0}×{x1-x0}px)")
    print(f"Delta            : ±{delta_ms:.0f}ms from Baseline_ON")
    print(f"\nSpatial variance (std of pixel values in ROI):")
    print(f"  Before  median={np.median(sv_b):.2f}  mean={np.mean(sv_b):.2f}")
    print(f"  After   median={np.median(sv_a):.2f}  mean={np.mean(sv_a):.2f}")
    print(f"  Ratio   median={np.median(ratios):.2f}  "
          f"fraction>1.5: {np.mean(ratios>1.5):.0%}  "
          f"fraction>1.1: {np.mean(ratios>1.1):.0%}")
    print(f"\nVerdict:")
    if np.median(ratios) > 1.5:
        print("  CLEAR: Grating stripes detectable as spatial structure. "
              "Spatial variance onset detection is viable.")
    elif np.median(ratios) > 1.1:
        print("  MARGINAL: Some spatial structure change but subtle. "
              "Spatial variance may work but needs enhancement.")
    else:
        print("  ABSENT: No detectable spatial structure change. "
              "Spatial variance approach unlikely to help — ROI may need adjustment.")
    print(f"\nFigures saved to: {FIG_DIR}")


def run_timeseries(session_name, n_trials=6, trial_offset=20, window_s=3.0):
    """Extract per-frame spatial variance time series across a wide window.

    For each sampled trial, extracts every frame in a ±window_s window around
    the predicted Baseline_ON camera time and computes spatial variance per frame.
    Plots the SV trace so the actual onset step is directly visible and measurable.
    """
    import cv2

    os.makedirs(FIG_DIR, exist_ok=True)

    logger.info(f"Loading session {session_name}...")
    sess = load_session(session_name)
    baseline_on = np.asarray(
        sess.ni_events.get("Baseline_ON", []), dtype=float
    ).flatten()
    baseline_on = baseline_on[baseline_on > 0]
    # Trim to task trials only — discard any post-task Baseline_ON events
    # (e.g., optotagging protocol runs after task stops; video continues).
    n_task_trials = len(getattr(sess, "trials", None) or [])
    if n_task_trials > 0 and len(baseline_on) > n_task_trials:
        logger.info(f"  Trimming Baseline_ON: {len(baseline_on)} → {n_task_trials} "
                    f"(dropping {len(baseline_on) - n_task_trials} post-task events)")
        baseline_on = baseline_on[:n_task_trials]
    logger.info(f"  {len(baseline_on)} Baseline_ON events")

    fname = SESSION_VIDEO_MAP.get(session_name)
    video_path = os.path.join(LOCAL_VIDEO_DIR, fname)
    meta_path = video_path.rsplit(".", 1)[0] + ".csv"
    if not os.path.exists(meta_path):
        logger.info("  Local metadata CSV not found — trying network camera root...")
        cam_files = find_camera_files(session_name)
        meta_path = cam_files["eye_cam"]["metadata"]

    ts_ms, _, _ = load_camera_metadata(meta_path)

    logger.info("Building sync model...")
    slope, offset, rmse_ms = build_sync_model(
        session_name, video_path, meta_path, baseline_on
    )

    roi = CORNEAL_EYE_ROI.get(session_name, list(CORNEAL_EYE_ROI.values())[0])
    radius = CORNEAL_CIRCLE_RADIUS.get(session_name)
    y0, y1, x0, x1 = roi

    n_avail = len(baseline_on)
    start = max(0, int(n_avail * 0.1))
    end = min(n_avail - 1, int(n_avail * 0.9))
    indices = np.linspace(start + trial_offset, end, n_trials, dtype=int)
    indices = np.clip(indices, 0, n_avail - 1)

    window_ms      = window_s * 1000.0
    BL_START_MS    = -2000.0  # baseline window start (ms relative to predicted onset)
    BL_END_MS      = -1000.0  # baseline window end
    SUSTAIN_MS     = 300.0    # minimum ms above threshold to confirm a real onset
    CEIL_FACTOR    = 2.0      # baseline_ceiling = session_ref × CEIL_FACTOR
    STEP_WINDOW_MS = 200.0    # window for post-onset mean (step ratio check)
    # Per-session sensitivity — see CORNEAL_DETECT_PARAMS
    THRESH_NSIGMA, MIN_STEP_RATIO = CORNEAL_DETECT_PARAMS.get(
        session_name, _DEFAULT_DETECT_PARAMS
    )

    # ── Phase 1: extract all SV traces (single video pass per trial) ─────────
    cap = cv2.VideoCapture(video_path)
    all_data = []   # list of (trial_i, nidaq_t, cam_ms, sv_array, t_array) | None

    for trial_i in indices:
        nidaq_t = baseline_on[trial_i]
        cam_ms  = nidaq_to_camera(np.array([nidaq_t]), slope, offset)[0]

        t_start = cam_ms - window_ms
        t_end   = cam_ms + window_ms
        mask = (ts_ms >= t_start) & (ts_ms <= t_end)
        frame_indices = np.where(mask)[0]

        if len(frame_indices) < 5:
            logger.warning(f"  Trial {trial_i}: too few frames in window, skipping")
            all_data.append(None)
            continue

        sv_trace, t_trace = [], []
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
        expected = frame_indices[0]
        for fidx in frame_indices:
            if fidx != expected:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            expected = fidx + 1
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            sv_trace.append(spatial_variance(gray, roi, radius))
            t_trace.append(ts_ms[fidx] - cam_ms)

        if not sv_trace:
            all_data.append(None)
            continue
        all_data.append((trial_i, nidaq_t, cam_ms,
                         np.array(sv_trace), np.array(t_trace)))

    cap.release()

    # ── Phase 2: session-level SV reference ──────────────────────────────────
    # 20th-percentile of all trial baseline means = "quiet eye" reference
    # (pre-grating, no blink).  Ceiling = ref × CEIL_FACTOR.  Any trial whose
    # baseline mean exceeds the ceiling is flagged as undetectable.
    baseline_means = []
    for item in all_data:
        if item is None:
            continue
        _, _, _, sv, t = item
        bl_mask = (t >= BL_START_MS) & (t <= BL_END_MS)
        if bl_mask.sum() >= 3:
            baseline_means.append(float(np.mean(sv[bl_mask])))
        else:
            baseline_means.append(float(np.mean(sv[:max(5, int(len(sv) * 0.20))])))

    session_sv_ref   = float(np.percentile(baseline_means, 20)) if baseline_means else 8.0
    baseline_ceiling = session_sv_ref * CEIL_FACTOR
    logger.info(f"  Session SV reference (20th pctile): {session_sv_ref:.2f}  "
                f"ceiling: {baseline_ceiling:.2f}")

    # ── Phase 3: detection + plotting ────────────────────────────────────────
    ncols = 2
    nrows = int(np.ceil(n_trials / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 7, nrows * 3.2), sharey=False)
    axes = np.array(axes).flatten()
    fig.suptitle(
        f"Corneal spatial variance time series — {session_name}\n"
        f"ROI {roi}  |  window ±{window_s:.1f}s  |  clock RMSE={rmse_ms:.0f}ms  |  "
        f"session SV ref={session_sv_ref:.1f}  ceil={baseline_ceiling:.1f}\n"
        f"Green shading = sustained ≥{SUSTAIN_MS:.0f}ms above threshold.  "
        f"Red bg = baseline elevated.  Yellow bg = no crossing found.",
        fontsize=8
    )

    n_detected  = 0
    n_elevated  = 0
    n_nocross   = 0
    detected_offsets = []

    for plot_i, item in enumerate(all_data):
        ax = axes[plot_i]

        if item is None:
            ax.text(0.5, 0.5, "no frames", transform=ax.transAxes, ha="center")
            ax.axis("off")
            continue

        trial_i, nidaq_t, cam_ms, sv_trace, t_trace = item

        # Frame interval → frames needed for SUSTAIN_MS
        frame_ms  = float(np.median(np.diff(t_trace))) if len(t_trace) > 2 else 20.0
        n_sustain = max(3, int(SUSTAIN_MS / frame_ms))

        bl_mask   = (t_trace >= BL_START_MS) & (t_trace <= BL_END_MS)
        if bl_mask.sum() >= 3:
            bl_mean = float(np.mean(sv_trace[bl_mask]))
            bl_std  = float(np.std(sv_trace[bl_mask]))
        else:
            n_fb    = max(5, int(len(sv_trace) * 0.20))
            bl_mean = float(np.mean(sv_trace[:n_fb]))
            bl_std  = float(np.std(sv_trace[:n_fb]))
        thresh    = bl_mean + THRESH_NSIGMA * bl_std
        # n_search: begin searching only after the baseline window ends
        bl_end_idx = int(np.searchsorted(t_trace, BL_END_MS))
        n_search   = max(bl_end_idx + 1, int(len(sv_trace) * 0.25))

        # Plot trace
        ax.plot(t_trace, sv_trace, color="steelblue", linewidth=0.8, alpha=0.85)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2,
                   label="predicted Baseline_ON")
        ax.axhline(thresh, color="orange", linestyle=":", linewidth=0.8,
                   label=f"bl+{THRESH_NSIGMA:.0f}SD={thresh:.1f}")
        ax.axhline(session_sv_ref, color="purple", linestyle=":", linewidth=0.7,
                   alpha=0.5, label=f"sess ref={session_sv_ref:.1f}")
        ax.axvspan(BL_START_MS, BL_END_MS,
                   alpha=0.08, color="gray", label="baseline region")

        # ── Flag: baseline too elevated ───────────────────────────────────
        if bl_mean > baseline_ceiling:
            n_elevated += 1
            ax.set_facecolor("#fff0f0")
            ax.text(0.02, 0.95,
                    f"BASELINE ELEVATED  bl={bl_mean:.1f} > ceil={baseline_ceiling:.1f}\n"
                    f"grating already on, or blink in baseline window",
                    transform=ax.transAxes, fontsize=7, color="red", va="top")
            logger.info(f"  Trial {trial_i}: ELEVATED  bl={bl_mean:.1f} > "
                        f"ceil={baseline_ceiling:.1f}")

        else:
            # ── Two-pass detection ──────────────────────────────────────────────
            # Pass 0: tight window [-200, +500ms] around predicted onset, short
            #   sustain (~100ms).  Catches genuine sharp onsets that are brief
            #   (e.g., eye movement right after grating appearance interrupts the
            #   signal, so 300ms sustain is never reached at the true onset time).
            # Pass 1: full-range scan, strict sustain (SUSTAIN_MS).  Fallback only
            #   when Pass 0 finds nothing near the predicted time.
            onset_i         = None
            step_ratio      = 0.0
            pass_used       = None
            n_step          = max(2, int(STEP_WINDOW_MS / frame_ms))
            n_tight_sustain = max(3, int(100.0 / frame_ms))
            tight_start_i   = int(np.searchsorted(t_trace, -200.0))
            tight_end_i     = int(np.searchsorted(t_trace, +500.0))

            # Pass 0 — tight window, short sustain
            for i in range(max(n_search, tight_start_i),
                           min(tight_end_i, len(sv_trace) - n_tight_sustain + 1)):
                if all(sv_trace[i + j] > thresh for j in range(n_tight_sustain)):
                    step_end   = min(i + n_step, len(sv_trace))
                    post_mean  = float(np.mean(sv_trace[i:step_end]))
                    ratio_here = post_mean / bl_mean if bl_mean > 0 else 0.0
                    if ratio_here >= MIN_STEP_RATIO:
                        onset_i    = i
                        step_ratio = ratio_here
                        pass_used  = 0
                        break

            # Pass 1 — full range, strict sustain (fallback)
            if onset_i is None:
                for i in range(n_search, len(sv_trace) - n_sustain + 1):
                    if all(sv_trace[i + j] > thresh for j in range(n_sustain)):
                        step_end   = min(i + n_step, len(sv_trace))
                        post_mean  = float(np.mean(sv_trace[i:step_end]))
                        ratio_here = post_mean / bl_mean if bl_mean > 0 else 0.0
                        if ratio_here >= MIN_STEP_RATIO:
                            onset_i    = i
                            step_ratio = ratio_here
                            pass_used  = 1
                            break

            if onset_i is not None:
                onset_t     = float(t_trace[onset_i])
                sustain_end = min(onset_i + n_sustain, len(t_trace) - 1)
                ax.axvspan(t_trace[onset_i], t_trace[sustain_end],
                           alpha=0.25, color="green")
                ax.axvline(onset_t, color="green", linestyle="-", linewidth=1.5,
                           label=f"onset Δ={onset_t:+.0f}ms  ratio={step_ratio:.2f}  p{pass_used}")
                n_detected += 1
                detected_offsets.append(onset_t)
                logger.info(f"  Trial {trial_i}: DETECTED  Δ={onset_t:+.0f}ms  "
                            f"ratio={step_ratio:.2f}  pass={pass_used}  bl={bl_mean:.1f}")
            else:
                n_nocross += 1
                ax.set_facecolor("#fffff0")
                ax.text(0.02, 0.95,
                        f"no qualifying crossing  "
                        f"(need {n_sustain}f sustained + ratio≥{MIN_STEP_RATIO})",
                        transform=ax.transAxes, fontsize=7, color="gray", va="top")
                logger.info(f"  Trial {trial_i}: NO CROSSING  bl={bl_mean:.1f}")

        ax.set_title(f"Trial {trial_i}  (NI-DAQ t={nidaq_t:.1f}s)", fontsize=8)
        ax.set_xlabel("Time from predicted Baseline_ON (ms)", fontsize=7)
        ax.set_ylabel("Spatial variance (px std)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")

    for i in range(n_trials, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"{session_name}_sv_timeseries.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")

    # ── Console summary ───────────────────────────────────────────────────────
    n_valid = len([d for d in all_data if d is not None])
    print(f"\n{'='*55}")
    print(f"TIMESERIES DETECTION SUMMARY — {session_name}")
    print(f"{'='*55}")
    print(f"Trials in window   : {n_valid}")
    print(f"Detected (clean)   : {n_detected}  ({100*n_detected/max(n_valid,1):.0f}%)")
    print(f"Baseline elevated  : {n_elevated}  (grating on / blink in baseline window)")
    print(f"No crossing        : {n_nocross}")
    if detected_offsets:
        arr = np.array(detected_offsets)
        print(f"\nOnset offset (Δ = cam_onset − predicted Baseline_ON):")
        print(f"  median  = {np.median(arr):+.0f} ms")
        print(f"  mean    = {np.mean(arr):+.0f} ms")
        print(f"  std     = {np.std(arr):.0f} ms")
        print(f"  IQR     = [{np.percentile(arr,25):+.0f}, {np.percentile(arr,75):+.0f}] ms")
    print(f"\nSession SV reference : {session_sv_ref:.2f}")
    print(f"Baseline ceiling     : {baseline_ceiling:.2f}  (ref × {CEIL_FACTOR})")


def run_roi_overlay(session_name, trial_idx=None):
    """Save an annotated frame showing the rectangular ROI + circular mask.

    Draws on the full-resolution frame:
      - Orange rectangle  : CORNEAL_EYE_ROI bounding box
      - Blue filled circle: pixels INCLUDED by the circular mask
      - Red  filled region: pixels EXCLUDED (inside rect, outside circle)
    Also saves a zoomed inset of just the patch with the mask overlay.

    If trial_idx is None, uses the first available Baseline_ON event.
    """
    import cv2

    os.makedirs(FIG_DIR, exist_ok=True)

    logger.info(f"Loading session {session_name} for ROI overlay...")
    sess = load_session(session_name)
    baseline_on = np.asarray(
        sess.ni_events.get("Baseline_ON", []), dtype=float
    ).flatten()
    baseline_on = baseline_on[baseline_on > 0]

    fname = SESSION_VIDEO_MAP.get(session_name)
    video_path = os.path.join(LOCAL_VIDEO_DIR, fname)
    meta_path = video_path.rsplit(".", 1)[0] + ".csv"
    if not os.path.exists(meta_path):
        cam_files = find_camera_files(session_name)
        meta_path = cam_files["eye_cam"]["metadata"]

    ts_ms, _, _ = load_camera_metadata(meta_path)

    logger.info("Building sync model for ROI overlay...")
    slope, offset, _ = build_sync_model(
        session_name, video_path, meta_path, baseline_on
    )

    # Pick trial: user-supplied, or trial #30 (middle of early session)
    if trial_idx is None:
        trial_idx = min(30, len(baseline_on) - 1)
    nidaq_t = baseline_on[trial_idx]
    cam_ms  = nidaq_to_camera(np.array([nidaq_t]), slope, offset)[0]
    # Use a frame ~600ms before predicted onset (expected grating onset time)
    target_ms = cam_ms - 600.0

    cap = cv2.VideoCapture(video_path)
    frame_idx = int(np.argmin(np.abs(ts_ms - target_ms)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, bgr = cap.read()
    cap.release()
    if not ret:
        logger.error("Could not read frame — check video path.")
        return

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape

    roi    = CORNEAL_EYE_ROI.get(session_name, list(CORNEAL_EYE_ROI.values())[0])
    radius = CORNEAL_CIRCLE_RADIUS.get(session_name)
    ctx    = CONTEXT_EYE_ROI.get(session_name, list(CONTEXT_EYE_ROI.values())[0])

    y0, y1, x0, x1 = roi
    cy0, cy1, cx0, cx1 = ctx
    patch_h, patch_w = y1 - y0, x1 - x0
    mask = _circle_mask(patch_h, patch_w, radius)

    # ── Figure: 4 panels ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        f"ROI overlay — {session_name}  trial {trial_idx}  "
        f"(frame at ~−600ms from predicted Baseline_ON)\n"
        f"Rect ROI {roi}   Circle radius={radius if radius else 'inscribed'}px   "
        f"Frame size: {H}×{W}",
        fontsize=9
    )

    # Panel 0: full frame thumbnail with both ROIs marked
    scale   = 0.25   # downsample factor for display
    thumb_h = max(1, int(H * scale))
    thumb_w = max(1, int(W * scale))
    import cv2 as _cv2
    thumb_gray = _cv2.resize(gray, (thumb_w, thumb_h), interpolation=_cv2.INTER_AREA)
    thumb_rgb  = np.clip(np.stack([thumb_gray] * 3, axis=-1), 0, 255).astype(np.uint8)
    # Draw context ROI in green
    _cv2.rectangle(thumb_rgb,
                   (int(cx0 * scale), int(cy0 * scale)),
                   (int(cx1 * scale), int(cy1 * scale)),
                   (0, 200, 0), 1)
    # Draw corneal ROI in orange
    _cv2.rectangle(thumb_rgb,
                   (int(x0 * scale), int(y0 * scale)),
                   (int(x1 * scale), int(y1 * scale)),
                   (255, 140, 0), 1)
    # Draw corneal circle in blue
    r_draw = int(radius) if radius else min(patch_h, patch_w) // 2
    _cv2.circle(thumb_rgb,
                (int((x0 + patch_w // 2) * scale), int((y0 + patch_h // 2) * scale)),
                max(1, int(r_draw * scale)), (0, 100, 255), 1)

    axes[0].imshow(thumb_rgb, interpolation="nearest")
    axes[0].set_title(
        f"Full frame ({H}×{W}) at 25%\n"
        f"Green=context  Orange=corneal  Blue=circle\n"
        f"Corneal centre: y={y0+patch_h//2}, x={x0+patch_w//2}",
        fontsize=7
    )
    axes[0].axis("off")

    # Panel 1: context crop (full eye) with ROI rectangle drawn on it
    ctx_gray = gray[cy0:cy1, cx0:cx1]
    ctx_rgb  = np.stack([ctx_gray] * 3, axis=-1).astype(np.uint8)
    # Draw orange rectangle for ROI bounds (relative to context crop)
    r_y0, r_y1 = y0 - cy0, y1 - cy0
    r_x0, r_x1 = x0 - cx0, x1 - cx0
    cv2.rectangle(ctx_rgb, (r_x0, r_y0), (r_x1, r_y1), (255, 140, 0), 1)
    # Draw circle in blue
    cy_rel = r_y0 + patch_h // 2
    cx_rel = r_x0 + patch_w // 2
    cv2.circle(ctx_rgb, (cx_rel, cy_rel), r_draw, (0, 100, 255), 1)

    axes[1].imshow(ctx_rgb, interpolation="nearest")
    axes[1].set_title(
        f"Context ROI {ctx}\n({cy1-cy0}×{cx1-cx0}px)\nOrange rect = bounding box, Blue = circle",
        fontsize=8
    )
    axes[1].axis("off")

    # Panel 2: zoomed patch — show included (blue tint) vs excluded (red tint)
    patch = gray[y0:y1, x0:x1].copy()
    patch_norm = (patch - patch.min()) / max(patch.max() - patch.min(), 1)
    rgb_patch = np.stack([patch_norm, patch_norm, patch_norm], axis=-1).copy()
    # Excluded corners → faint red tint
    excl = ~mask
    rgb_patch[excl, 0] = np.clip(rgb_patch[excl, 0] + 0.4, 0, 1)
    rgb_patch[excl, 1] = np.clip(rgb_patch[excl, 1] - 0.1, 0, 1)
    rgb_patch[excl, 2] = np.clip(rgb_patch[excl, 2] - 0.1, 0, 1)
    # Included pixels → faint blue tint
    rgb_patch[mask, 2] = np.clip(rgb_patch[mask, 2] + 0.15, 0, 1)

    axes[2].imshow(rgb_patch, interpolation="nearest")
    axes[2].set_title(
        f"Patch zoom ({patch_h}×{patch_w}px)\n"
        f"Blue = included ({mask.sum()} px)   Red = excluded ({(~mask).sum()} px)",
        fontsize=8
    )
    axes[2].axis("off")

    # Panel 3: histogram of pixel values — included vs excluded
    incl_vals = patch[mask].ravel()
    excl_vals = patch[~mask].ravel()
    axes[3].hist(incl_vals, bins=30, alpha=0.7, color="steelblue",
                 label=f"included n={len(incl_vals)}")
    axes[3].hist(excl_vals, bins=30, alpha=0.5, color="salmon",
                 label=f"excluded n={len(excl_vals)}")
    sv_incl = float(np.std(incl_vals)) if len(incl_vals) >= 4 else 0.0
    sv_excl = float(np.std(excl_vals)) if len(excl_vals) >= 4 else 0.0
    axes[3].set_title(
        f"Pixel intensity distribution\nSV included={sv_incl:.2f}  SV excluded={sv_excl:.2f}",
        fontsize=8
    )
    axes[3].set_xlabel("Pixel value")
    axes[3].set_ylabel("Count")
    axes[3].legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"{session_name}_roi_overlay.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")

    # ── Extra: full-resolution frame with labeled grid for coordinate lookup ──
    grid_spacing = 50   # label every 50px in full-frame coordinates
    fig_ref, ax_ref = plt.subplots(figsize=(W / 80, H / 80))
    ax_ref.imshow(gray, cmap="gray", interpolation="nearest")
    # Grid lines
    for gx in range(0, W, grid_spacing):
        ax_ref.axvline(gx, color="lime", linewidth=0.3, alpha=0.5)
        ax_ref.text(gx, 4, str(gx), color="lime", fontsize=4, va="top", ha="center")
    for gy in range(0, H, grid_spacing):
        ax_ref.axhline(gy, color="lime", linewidth=0.3, alpha=0.5)
        ax_ref.text(2, gy, str(gy), color="lime", fontsize=4, va="center", ha="left")
    # Current ROI in red (so you can see where it ended up)
    from matplotlib.patches import Rectangle
    ax_ref.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                edgecolor="red", facecolor="none", linewidth=1.5,
                                label=f"Current ROI ({y0},{y1},{x0},{x1})"))
    ax_ref.legend(fontsize=6, loc="upper right")
    ax_ref.set_title(
        f"{session_name} — full frame with 50px grid (x=horizontal, y=vertical)\n"
        f"Red = current corneal ROI  |  Read off eye-centre coordinates and report back",
        fontsize=7
    )
    ax_ref.axis("on")
    out_ref = os.path.join(FIG_DIR, f"{session_name}_frame_grid_ref.png")
    fig_ref.savefig(out_ref, dpi=150, bbox_inches="tight")
    plt.close(fig_ref)
    logger.info(f"Saved grid reference: {out_ref}")

    print(f"\nROI overlay saved: {out}")
    print(f"Grid reference   : {out_ref}  ← open this to read off eye coordinates")
    print(f"  Current ROI {roi}: {patch_h}×{patch_w}px")
    print(f"  Circle radius={r_draw}px: {mask.sum()} included, {(~mask).sum()} excluded")
    print(f"  SV (included pixels): {sv_incl:.2f}")
    print(f"  SV (excluded pixels): {sv_excl:.2f}")


def run_full_session(session_name, window_s=3.0, skip_short=False,
                     min_fa_rt_ms=500.0, auto_calibrate=True,
                     subject="BG_046"):
    """Process ALL Baseline_ON events, fit a corneal-SV clock model, and cache results.

    Sorts all trials chronologically before video reading to minimise H.264 seeking.
    Outputs:
      - Console summary  (detection rate, RMSE, offset distribution)
      - 3-panel figure   (offset scatter / histogram / residuals)
      - JSON cache       data/cache/video_sync/{session}_corneal_sync.json

    skip_short : if True, skip abort trials and fast-FA trials (FA RT < min_fa_rt_ms).
                 These trials have early eye movements that can corrupt the SV trace.
    min_fa_rt_ms : FA reaction-time threshold in ms (default 500 ms).
    """
    import cv2
    import json

    os.makedirs(FIG_DIR, exist_ok=True)
    cache_dir = os.path.join(_PROJECT_ROOT, "data", "cache", "video_sync")
    os.makedirs(cache_dir, exist_ok=True)

    logger.info(f"Loading session {session_name}...")
    sess = load_session(session_name)
    baseline_on = np.asarray(
        sess.ni_events.get("Baseline_ON", []), dtype=float
    ).flatten()
    baseline_on = baseline_on[baseline_on > 0]
    # Trim to task trials only — discard any post-task Baseline_ON events.
    n_task_trials = len(getattr(sess, "trials", None) or [])
    if n_task_trials > 0 and len(baseline_on) > n_task_trials:
        logger.info(f"  Trimming Baseline_ON: {len(baseline_on)} → {n_task_trials} "
                    f"(dropping {len(baseline_on) - n_task_trials} post-task events)")
        baseline_on = baseline_on[:n_task_trials]
    n_baseline_on = len(baseline_on)
    logger.info(f"  {n_baseline_on} Baseline_ON events")

    fname = SESSION_VIDEO_MAP.get(session_name)
    video_path = os.path.join(LOCAL_VIDEO_DIR, fname)
    meta_path = video_path.rsplit(".", 1)[0] + ".csv"
    if not os.path.exists(meta_path):
        logger.info("  Local metadata CSV not found — trying network camera root...")
        cam_files = find_camera_files(session_name)
        meta_path = cam_files["eye_cam"]["metadata"]

    ts_ms, _, _ = load_camera_metadata(meta_path)
    logger.info(f"  Video: {len(ts_ms)} frames, {ts_ms[-1]/1000:.0f}s")

    logger.info("Building initial sync model for window placement...")
    slope0, offset0, _ = build_sync_model(
        session_name, video_path, meta_path, baseline_on
    )

    roi    = CORNEAL_EYE_ROI.get(session_name, list(CORNEAL_EYE_ROI.values())[0])
    radius = CORNEAL_CIRCLE_RADIUS.get(session_name)

    # ── ROI resolution: auto-calibrate → manual fallback ──────────────
    _using_autocal = False
    if auto_calibrate:
        cal = load_corneal_cal(session_name)
        if cal is None:
            logger.info("  No cached calibration — running auto-calibration...")
            cal = auto_calibrate_corneal_roi(
                session_name, video_path, meta_path, baseline_on,
                rough_offset_s=offset0,
                broad_eye_roi=BROAD_EYE_ROI.get(subject),
            )
        if cal is not None and cal["mask_quality"] != "failed":
            corneal_mask = load_corneal_mask(session_name)
            if corneal_mask is not None:
                roi = corneal_mask     # spatial_variance() accepts bool mask
                radius = None
                _using_autocal = True
                logger.info(
                    f"  Using auto-calibrated mask: {cal['mask_area_px']}px "
                    f"({cal['mask_quality']}, mean_std={cal['best_component_mean_std']:.2f})"
                )
            else:
                logger.warning("  Auto-cal JSON exists but mask NPZ missing — "
                               "falling back to manual ROI")
        else:
            logger.warning("  Auto-calibration failed or quality='failed' — "
                           "falling back to manual ROI")

    if not _using_autocal:
        logger.info(f"  Using manual ROI: {roi}, radius={radius}")

    # Sort all trials by estimated camera time → near-sequential H.264 reading
    cam_times_est = nidaq_to_camera(baseline_on, slope0, offset0)
    order = np.argsort(cam_times_est)

    # ── Optional: build skip mask for short/disrupted trials ──────────
    skip_mask = np.zeros(n_baseline_on, dtype=bool)
    if skip_short:
        trials  = sess.trials
        n_match = min(len(trials), n_baseline_on)
        n_skip_abort = 0
        n_skip_fa    = 0
        for i in range(n_match):
            t       = trials[i]
            outcome = (t.trialoutcome or "").lower()
            if outcome == "abort":
                skip_mask[i] = True
                n_skip_abort += 1
            elif outcome == "fa":
                rts    = t.reactiontimes or {}
                fa_rt  = rts.get("FA", rts.get("fa", np.nan))
                if np.isnan(fa_rt) and rts:
                    try:
                        fa_rt = float(next(iter(rts.values())))
                    except Exception:
                        pass
                # RT stored in seconds; compare against threshold in ms
                if not np.isnan(fa_rt) and fa_rt * 1000.0 < min_fa_rt_ms:
                    skip_mask[i] = True
                    n_skip_fa += 1
        logger.info(f"  skip_short=True: {skip_mask.sum()} trials skipped "
                    f"({n_skip_abort} abort, {n_skip_fa} fast FA <{min_fa_rt_ms:.0f}ms)")
    else:
        logger.info("  skip_short=False: all trials processed")

    window_ms      = window_s * 1000.0
    BL_START_MS    = -2000.0  # baseline window start (ms relative to predicted onset)
    BL_END_MS      = -1000.0  # baseline window end
    SUSTAIN_MS     = 300.0
    CEIL_FACTOR    = 2.0
    STEP_WINDOW_MS = 200.0
    # Per-session sensitivity — see CORNEAL_DETECT_PARAMS
    THRESH_NSIGMA, MIN_STEP_RATIO = CORNEAL_DETECT_PARAMS.get(
        session_name, _DEFAULT_DETECT_PARAMS
    )

    # ── Phase 1: extract SV traces for all trials ─────────────────────
    logger.info(f"Phase 1: extracting SV traces ({n_baseline_on} trials)...")
    cap = cv2.VideoCapture(video_path)
    # all_data indexed by position in `order`, contains (orig_idx, nidaq_t, cam_ms_est, sv, t)
    all_data = []

    for count, orig_i in enumerate(order):
        nidaq_t = baseline_on[orig_i]
        cam_ms  = cam_times_est[orig_i]

        if skip_mask[orig_i]:
            all_data.append(None)
            continue

        mask_f = (ts_ms >= cam_ms - window_ms) & (ts_ms <= cam_ms + window_ms)
        fidxs  = np.where(mask_f)[0]

        if len(fidxs) < 5:
            all_data.append(None)
            continue

        sv_trace, t_trace = [], []
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidxs[0])
        expected = fidxs[0]
        for fidx in fidxs:
            if fidx != expected:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            expected = fidx + 1
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            sv_trace.append(spatial_variance(gray, roi, radius))
            t_trace.append(ts_ms[fidx] - cam_ms)

        if not sv_trace:
            all_data.append(None)
            continue
        all_data.append((orig_i, nidaq_t, cam_ms,
                         np.array(sv_trace), np.array(t_trace)))

        if (count + 1) % 100 == 0:
            logger.info(f"  {count+1}/{n_baseline_on} trials processed...")

    cap.release()
    n_valid = sum(d is not None for d in all_data)
    logger.info(f"  Phase 1 done — {n_valid} traces extracted.")

    # ── Phase 2: session-level SV reference ───────────────────────────
    baseline_means = []
    for item in all_data:
        if item is None:
            continue
        sv, t = item[3], item[4]
        bl_mask = (t >= BL_START_MS) & (t <= BL_END_MS)
        if bl_mask.sum() >= 3:
            baseline_means.append(float(np.mean(sv[bl_mask])))
        else:
            baseline_means.append(float(np.mean(sv[:max(5, int(len(sv) * 0.20))])))

    session_sv_ref   = float(np.percentile(baseline_means, 20)) if baseline_means else 8.0
    baseline_ceiling = session_sv_ref * CEIL_FACTOR
    logger.info(f"  Session SV ref={session_sv_ref:.2f}  ceiling={baseline_ceiling:.2f}")

    # ── Phase 3: detect onset for every trial ─────────────────────────
    logger.info("Phase 3: detecting onsets...")
    # Each entry: (orig_i, nidaq_t, cam_onset_ms, delta_ms, step_ratio, status)
    results = []

    for item in all_data:
        if item is None:
            results.append(None)
            continue

        orig_i, nidaq_t, cam_ms, sv_trace, t_trace = item
        frame_ms  = float(np.median(np.diff(t_trace))) if len(t_trace) > 2 else 20.0
        n_sustain = max(3, int(SUSTAIN_MS / frame_ms))
        n_step    = max(2, int(STEP_WINDOW_MS / frame_ms))

        bl_mask   = (t_trace >= BL_START_MS) & (t_trace <= BL_END_MS)
        if bl_mask.sum() >= 3:
            bl_mean = float(np.mean(sv_trace[bl_mask]))
            bl_std  = float(np.std(sv_trace[bl_mask]))
        else:
            n_fb    = max(5, int(len(sv_trace) * 0.20))
            bl_mean = float(np.mean(sv_trace[:n_fb]))
            bl_std  = float(np.std(sv_trace[:n_fb]))
        thresh    = bl_mean + THRESH_NSIGMA * bl_std
        bl_end_idx = int(np.searchsorted(t_trace, BL_END_MS))
        n_search   = max(bl_end_idx + 1, int(len(sv_trace) * 0.25))

        if bl_mean > baseline_ceiling:
            results.append((orig_i, nidaq_t, np.nan, np.nan, np.nan, "elevated"))
            continue

        onset_i         = None
        step_ratio      = 0.0
        n_tight_sustain = max(3, int(100.0 / frame_ms))
        tight_start_i   = int(np.searchsorted(t_trace, -200.0))
        tight_end_i     = int(np.searchsorted(t_trace, +500.0))

        # Pass 0 — tight window [-200, +500ms], short sustain (~100ms)
        for i in range(max(n_search, tight_start_i),
                       min(tight_end_i, len(sv_trace) - n_tight_sustain + 1)):
            if all(sv_trace[i + j] > thresh for j in range(n_tight_sustain)):
                step_end   = min(i + n_step, len(sv_trace))
                post_mean  = float(np.mean(sv_trace[i:step_end]))
                ratio_here = post_mean / bl_mean if bl_mean > 0 else 0.0
                if ratio_here >= MIN_STEP_RATIO:
                    onset_i    = i
                    step_ratio = ratio_here
                    break

        # Pass 1 — full range, strict sustain (fallback)
        if onset_i is None:
            for i in range(n_search, len(sv_trace) - n_sustain + 1):
                if all(sv_trace[i + j] > thresh for j in range(n_sustain)):
                    step_end   = min(i + n_step, len(sv_trace))
                    post_mean  = float(np.mean(sv_trace[i:step_end]))
                    ratio_here = post_mean / bl_mean if bl_mean > 0 else 0.0
                    if ratio_here >= MIN_STEP_RATIO:
                        onset_i    = i
                        step_ratio = ratio_here
                        break

        if onset_i is not None:
            cam_onset_ms = cam_ms + float(t_trace[onset_i])
            delta_ms     = float(t_trace[onset_i])
            results.append((orig_i, nidaq_t, cam_onset_ms, delta_ms, step_ratio, "detected"))
        else:
            results.append((orig_i, nidaq_t, np.nan, np.nan, np.nan, "nocross"))

    # ── Fit clock model from detected anchors ─────────────────────────
    det_rows   = [r for r in results if r is not None and r[5] == "detected"]
    n_detected = len(det_rows)
    logger.info(f"  Detected: {n_detected}/{n_baseline_on} ({100*n_detected/n_baseline_on:.0f}%)")

    if n_detected < 10:
        logger.error(f"Too few detections ({n_detected}) — cannot fit clock model.")
        return

    det_nidaq_s = np.array([r[1] for r in det_rows])
    det_cam_s   = np.array([r[2] for r in det_rows]) / 1000.0  # ms → s
    sync = fit_clock_model(det_cam_s, det_nidaq_s, n_baseline_on=n_baseline_on)
    logger.info(f"  Clock model: RMSE={sync.rmse_ms:.1f}ms  quality={sync.quality}  "
                f"slope={sync.slope:.8f}  offset={sync.offset:.3f}s")

    # ── Cache JSON ────────────────────────────────────────────────────
    deltas = np.array([r[3] for r in det_rows])
    # When auto-calibration is used, roi is a bool mask array — store bbox instead
    if _using_autocal and cal is not None:
        roi_serializable = cal.get("corneal_bbox", None)
    else:
        roi_serializable = [int(x) for x in roi] if roi is not None else None
    cache  = {
        "session_name": session_name,
        "method": "corneal_sv",
        "slope": sync.slope,
        "offset": sync.offset,
        "n_anchors": n_detected,
        "n_baseline_on": n_baseline_on,
        "coverage": round(n_detected / n_baseline_on, 4),
        "rmse_ms": round(sync.rmse_ms, 2),
        "quality": sync.quality,
        "median_onset_delta_ms": round(float(np.median(deltas)), 1),
        "roi": roi_serializable,
        "circle_radius": radius,
        "autocal": bool(_using_autocal),
    }
    cache_path = os.path.join(cache_dir, f"{session_name}_corneal_sync.json")
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    logger.info(f"  Saved: {cache_path}")

    # ── Summary figure (3 panels) ─────────────────────────────────────
    nidaq_det = np.array([r[1] for r in det_rows])
    cam_det   = np.array([r[2] for r in det_rows])
    cam_pred  = nidaq_to_camera(nidaq_det, sync.slope, sync.offset)
    residuals = cam_det - cam_pred

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"Corneal SV clock model — {session_name}\n"
        f"anchors={n_detected}/{n_baseline_on} ({100*n_detected/n_baseline_on:.0f}%)  "
        f"RMSE={sync.rmse_ms:.1f}ms  quality={sync.quality}  "
        f"slope={sync.slope:.7f}  offset={sync.offset:.2f}s",
        fontsize=9
    )

    ax = axes[0]
    ax.scatter(nidaq_det / 60, deltas, s=3, alpha=0.35, color="steelblue")
    ax.axhline(np.median(deltas), color="orange", linestyle="--",
               label=f"median={np.median(deltas):+.0f}ms")
    ax.axhline(np.percentile(deltas, 25), color="orange", linestyle=":", alpha=0.6)
    ax.axhline(np.percentile(deltas, 75), color="orange", linestyle=":", alpha=0.6,
               label=f"IQR=[{np.percentile(deltas,25):+.0f},{np.percentile(deltas,75):+.0f}]ms")
    ax.set_xlabel("NI-DAQ time (min)")
    ax.set_ylabel("Detected onset Δ (ms)")
    ax.set_title("Onset offset over session")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.hist(deltas, bins=50, color="steelblue", edgecolor="white")
    ax.axvline(np.median(deltas), color="orange", linestyle="--",
               label=f"median={np.median(deltas):+.0f}ms\n"
                     f"IQR=[{np.percentile(deltas,25):+.0f},{np.percentile(deltas,75):+.0f}]ms")
    ax.set_xlabel("Detected onset Δ (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of onset offsets")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.scatter(nidaq_det / 60, residuals, s=3, alpha=0.35, color="steelblue")
    ax.axhline(0, color="red", linestyle="--")
    ax.axhline(sync.rmse_ms,  color="orange", linestyle=":", alpha=0.7,
               label=f"±RMSE={sync.rmse_ms:.1f}ms")
    ax.axhline(-sync.rmse_ms, color="orange", linestyle=":", alpha=0.7)
    ax.set_xlabel("NI-DAQ time (min)")
    ax.set_ylabel("Residual (ms)")
    ax.set_title("Clock model residuals")
    ax.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"{session_name}_full_session_corneal.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved figure: {out}")

    # ── Console summary ───────────────────────────────────────────────
    n_elevated = sum(1 for r in results if r is not None and r[5] == "elevated")
    n_nocross  = sum(1 for r in results if r is not None and r[5] == "nocross")
    n_none     = sum(1 for r in results if r is None)
    n_skipped  = int(skip_mask.sum())
    print(f"\n{'='*60}")
    print(f"FULL SESSION CORNEAL SYNC — {session_name}")
    print(f"{'='*60}")
    print(f"Total Baseline_ON    : {n_baseline_on}")
    if skip_short:
        print(f"Skipped (abort/FA)   : {n_skipped}  "
              f"(abort + fast FA <{min_fa_rt_ms:.0f}ms)")
    print(f"Detected (clean)     : {n_detected}  ({100*n_detected/n_baseline_on:.0f}%)")
    print(f"Baseline elevated    : {n_elevated}")
    print(f"No qualifying cross  : {n_nocross}")
    print(f"Frame errors         : {n_none}")
    print(f"\nOnset Δ (detected):")
    print(f"  median = {np.median(deltas):+.0f} ms")
    print(f"  IQR    = [{np.percentile(deltas,25):+.0f}, {np.percentile(deltas,75):+.0f}] ms")
    print(f"  std    = {np.std(deltas):.0f} ms")
    print(f"\nCorneal SV clock model:")
    print(f"  slope  = {sync.slope:.8f}")
    print(f"  offset = {sync.offset:.3f} s")
    print(f"  RMSE   = {sync.rmse_ms:.1f} ms")
    print(f"  quality= {sync.quality}")
    print(f"\nCached : {cache_path}")
    print(f"Figure : {out}")


def run_corneal_validate(session_name, n_trials=12, trial_offset=20):
    """Visual validation: show frames before and after the corneal-predicted Baseline_ON.

    Loads the cached corneal sync model (data/cache/video_sync/{session}_corneal_sync.json)
    and for N sampled trials shows 4 frames per trial:
        −300 ms  |  −50 ms  |  +50 ms  |  +200 ms
    relative to the corrected camera onset time.

    Uses the wider CONTEXT_EYE_ROI so grating stripes are clearly visible.
    Red dashed line marks t=0 (predicted corneal onset).
    """
    import cv2
    import json

    os.makedirs(FIG_DIR, exist_ok=True)

    cache_path = os.path.join(_PROJECT_ROOT, "data", "cache", "video_sync",
                              f"{session_name}_corneal_sync.json")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Corneal sync cache not found: {cache_path}\n"
            f"Run --full-session first."
        )
    with open(cache_path) as f:
        cache = json.load(f)

    slope  = cache["slope"]
    offset = cache["offset"]
    logger.info(f"Loaded corneal model: slope={slope:.8f}  offset={offset:.3f}s  "
                f"RMSE={cache['rmse_ms']:.1f}ms")

    sess = load_session(session_name)
    baseline_on = np.asarray(
        sess.ni_events.get("Baseline_ON", []), dtype=float
    ).flatten()
    baseline_on = baseline_on[baseline_on > 0]

    fname      = SESSION_VIDEO_MAP.get(session_name)
    video_path = os.path.join(LOCAL_VIDEO_DIR, fname)
    meta_path  = video_path.rsplit(".", 1)[0] + ".csv"
    if not os.path.exists(meta_path):
        cam_files = find_camera_files(session_name)
        meta_path = cam_files["eye_cam"]["metadata"]

    ts_ms, _, _ = load_camera_metadata(meta_path)

    roi     = CORNEAL_EYE_ROI.get(session_name, list(CORNEAL_EYE_ROI.values())[0])
    radius  = CORNEAL_CIRCLE_RADIUS.get(session_name)
    ctx_roi = CONTEXT_EYE_ROI.get(session_name, list(CONTEXT_EYE_ROI.values())[0])
    cy0, cy1, cx0, cx1 = ctx_roi

    n_avail = len(baseline_on)
    start   = max(0, int(n_avail * 0.1))
    end     = min(n_avail - 1, int(n_avail * 0.9))
    indices = np.linspace(start + trial_offset, end, n_trials, dtype=int)
    indices = np.clip(indices, 0, n_avail - 1)

    # Four time offsets around the predicted corneal onset
    offsets_ms = [-300, -50, +50, +200]
    labels     = ["−300 ms\n(before)", "−50 ms\n(just before)",
                  "+50 ms\n(just after)", "+200 ms\n(after)"]

    cap     = cv2.VideoCapture(video_path)
    results = []

    for idx in indices:
        nidaq_t = baseline_on[idx]
        cam_ms  = nidaq_to_camera(np.array([nidaq_t]), slope, offset)[0]

        frames, svs, actual_dts = [], [], []
        for dms in offsets_ms:
            target_ms = cam_ms + dms
            fi        = int(np.argmin(np.abs(ts_ms - target_ms)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frm  = cap.read()
            if not ret:
                frames.append(None); svs.append(np.nan)
                actual_dts.append(np.nan); continue
            gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY).astype(np.float32)
            frames.append(gray[cy0:cy1, cx0:cx1].copy())
            svs.append(spatial_variance(gray, roi, radius))
            actual_dts.append(ts_ms[fi] - cam_ms)

        results.append({
            "trial_idx": int(idx),
            "nidaq_t":   nidaq_t,
            "cam_ms":    cam_ms,
            "frames":    frames,
            "svs":       svs,
            "dts":       actual_dts,
        })

    cap.release()

    # ── Figure ────────────────────────────────────────────────────────
    n_cols = len(offsets_ms)
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.2, n_rows * 2.8))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"Corneal SV validation — {session_name}\n"
        f"Corneal model: RMSE={cache['rmse_ms']:.1f}ms  "
        f"coverage={cache['coverage']*100:.0f}%  "
        f"median Δ={cache['median_onset_delta_ms']:+.0f}ms\n"
        f"Columns: time relative to corneal-predicted Baseline_ON   "
        f"(context ROI {ctx_roi}, {cy1-cy0}×{cx1-cx0}px)",
        fontsize=9
    )

    for row, r in enumerate(results):
        # shared contrast range across all 4 frames for this trial
        valid_frames = [f for f in r["frames"] if f is not None]
        vmin = min(f.min() for f in valid_frames) - 3 if valid_frames else 0
        vmax = max(f.max() for f in valid_frames) + 3 if valid_frames else 255
        vmin, vmax = max(0, vmin), min(255, vmax)

        for col, (dms, lab) in enumerate(zip(offsets_ms, labels)):
            ax  = axes[row, col]
            frm = r["frames"][col]
            sv  = r["svs"][col]
            dt  = r["dts"][col]

            if frm is None:
                ax.text(0.5, 0.5, "read\nerror",
                        transform=ax.transAxes, ha="center", va="center")
                ax.axis("off")
                continue

            ax.imshow(frm, cmap="gray", vmin=vmin, vmax=vmax,
                      interpolation="nearest")

            # Colour-code: blue = before onset, green = after onset
            border_col = "#2196F3" if dms < 0 else "#4CAF50"
            for spine in ax.spines.values():
                spine.set_edgecolor(border_col)
                spine.set_linewidth(2.5)

            title_top = (f"T{r['trial_idx']}  {lab}"
                         if row == 0 else f"{lab}")
            ax.set_title(title_top, fontsize=7.5, color=border_col)
            ax.set_xlabel(f"Δt={dt:+.0f}ms  SV={sv:.1f}", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"{session_name}_corneal_validation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")
    print(f"\nValidation figure saved: {out}")
    print(f"Blue border = before corneal onset, Green = after")


def scan_coarse_offset(session_name, n_trials=50, offset_min=1.0, offset_max=17.0,
                       offset_step=0.5, sv_ratio_thresh=1.20,
                       pre_ms_list=(-800, -600, -400), post_ms_list=(100, 200, 300)):
    """Scan a range of coarse offsets to find which one aligns video frames to Baseline_ON.

    For each candidate offset, reads ~n_trials frames at the predicted Baseline_ON time
    and checks whether the spatial variance of the corneal ROI increases from pre → post.
    Prints a ranked table and saves a bar-chart figure.

    Useful for diagnosing sessions where the automatic coarse offset detection is wrong
    (e.g., due to periodic trial structure causing aliased matches).

    Parameters
    ----------
    session_name : str
        Session identifier.
    n_trials : int
        How many (evenly spaced) trials to probe per offset candidate.
    offset_min, offset_max, offset_step : float
        Scan range and step size in seconds.
    sv_ratio_thresh : float
        Minimum sv_post / sv_pre ratio to count a trial as "detected".
    pre_ms_list, post_ms_list : lists of float
        ms offsets relative to predicted onset to sample for pre/post SV.
    """
    import cv2
    import json

    logger.info(f"Scanning coarse offsets [{offset_min}, {offset_max}]s "
                f"step={offset_step}s for session {session_name}...")

    sess = load_session(session_name)
    baseline_on = np.asarray(
        sess.ni_events.get("Baseline_ON", []), dtype=float
    ).flatten()
    baseline_on = baseline_on[baseline_on > 0]
    n_task = len(getattr(sess, "trials", []))
    if n_task > 0 and len(baseline_on) > n_task:
        baseline_on = baseline_on[:n_task]

    # Sample n_trials evenly across the session (skip first and last 5%)
    n_total = len(baseline_on)
    start = max(0, int(n_total * 0.05))
    end   = min(n_total - 1, int(n_total * 0.95))
    trial_idxs = np.linspace(start, end, min(n_trials, end - start), dtype=int)
    bl_sample  = baseline_on[trial_idxs]
    logger.info(f"  Sampling {len(bl_sample)} trials (total {n_total})")

    fname      = SESSION_VIDEO_MAP.get(session_name)
    video_path = os.path.join(LOCAL_VIDEO_DIR, fname)
    meta_path  = video_path.rsplit(".", 1)[0] + ".csv"
    if not os.path.exists(meta_path):
        cam_files = find_camera_files(session_name)
        meta_path = cam_files["eye_cam"]["metadata"]

    ts_ms, _, _ = load_camera_metadata(meta_path)
    video_dur_s = ts_ms[-1] / 1000.0
    logger.info(f"  Video: {len(ts_ms)} frames, {video_dur_s:.0f}s")

    roi    = CORNEAL_EYE_ROI.get(session_name, list(CORNEAL_EYE_ROI.values())[0])
    radius = CORNEAL_CIRCLE_RADIUS.get(session_name)

    cap = cv2.VideoCapture(video_path)

    candidates = np.arange(offset_min, offset_max + offset_step * 0.5, offset_step)
    results = []

    for cand_offset in candidates:
        cam_preds_ms = nidaq_to_camera(bl_sample, 1.0, cand_offset)

        # Skip candidates that push predictions out of video bounds
        if (cam_preds_ms < 0).any() or (cam_preds_ms > ts_ms[-1]).any():
            results.append((cand_offset, 0, 0, []))
            continue

        hits = 0
        ratios = []
        for cam_ms in cam_preds_ms:
            # Compute mean SV in pre-window
            pre_svs = []
            for dms in pre_ms_list:
                target = cam_ms + dms
                if target < 0 or target > ts_ms[-1]:
                    continue
                fi = int(np.argmin(np.abs(ts_ms - target)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frm = cap.read()
                if ret:
                    gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    pre_svs.append(spatial_variance(gray, roi, radius))

            # Compute mean SV in post-window
            post_svs = []
            for dms in post_ms_list:
                target = cam_ms + dms
                if target < 0 or target > ts_ms[-1]:
                    continue
                fi = int(np.argmin(np.abs(ts_ms - target)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frm = cap.read()
                if ret:
                    gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    post_svs.append(spatial_variance(gray, roi, radius))

            if pre_svs and post_svs:
                sv_pre  = float(np.mean(pre_svs))
                sv_post = float(np.mean(post_svs))
                ratio   = sv_post / max(sv_pre, 0.1)
                ratios.append(ratio)
                if ratio >= sv_ratio_thresh:
                    hits += 1

        hit_pct = 100.0 * hits / max(len(ratios), 1)
        med_ratio = float(np.median(ratios)) if ratios else 0.0
        results.append((cand_offset, hits, hit_pct, med_ratio))
        logger.info(f"  offset={cand_offset:5.1f}s  hits={hits:3d}/{len(ratios)}  "
                    f"({hit_pct:5.1f}%)  med_ratio={med_ratio:.3f}")

    cap.release()

    # Print ranked summary
    ranked = sorted(results, key=lambda r: (-r[1], -r[3]))
    print(f"\n{'='*58}")
    print(f"Coarse offset scan — {session_name}  (ROI: {roi})")
    print(f"{'offset_s':>10}  {'hits':>5}  {'hit%':>6}  {'med_ratio':>9}  {'rank':>5}")
    print(f"{'-'*58}")
    for rank, (off, hits, pct, med_r) in enumerate(ranked[:15], 1):
        marker = " <-- BEST" if rank == 1 else ""
        print(f"{off:10.1f}  {hits:5d}  {pct:6.1f}%  {med_r:9.3f}{marker}")
    print(f"{'='*58}\n")

    best_offset = ranked[0][0] if ranked else None

    # Save bar chart
    os.makedirs(FIG_DIR, exist_ok=True)
    offsets_arr  = np.array([r[0] for r in results])
    hitpcts_arr  = np.array([r[2] for r in results])
    ratios_arr   = np.array([r[3] for r in results])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        f"Coarse offset scan — {session_name}\n"
        f"Best: {best_offset:.1f}s  (ROI: {roi}, thresh={sv_ratio_thresh:.2f})",
        fontsize=10,
    )
    axes[0].bar(offsets_arr, hitpcts_arr, width=offset_step * 0.8,
                color=["#e53935" if abs(o - best_offset) < 0.01 else "#1976D2"
                       for o in offsets_arr])
    axes[0].set_ylabel("Hit rate (%)")
    axes[0].set_title("Detection rate per offset")
    # Mark cached value
    cached_off = 14.5  # known from coarse_offsets.json
    axes[0].axvline(cached_off, color="orange", lw=1.5, ls="--",
                    label=f"cached={cached_off:.1f}s")
    axes[0].legend(fontsize=8)

    axes[1].bar(offsets_arr, ratios_arr, width=offset_step * 0.8,
                color=["#e53935" if abs(o - best_offset) < 0.01 else "#1976D2"
                       for o in offsets_arr])
    axes[1].axhline(sv_ratio_thresh, color="gray", lw=1, ls=":")
    axes[1].axvline(cached_off, color="orange", lw=1.5, ls="--")
    axes[1].set_xlabel("Coarse offset (s)")
    axes[1].set_ylabel("Median SV ratio (post/pre)")
    axes[1].set_title("SV step magnitude per offset")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"{session_name}_coarse_scan.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")
    print(f"Bar chart saved: {out}")

    return best_offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="09092025",
                        choices=list(SESSION_VIDEO_MAP.keys()),
                        help="Session to diagnose (default: 09092025)")
    parser.add_argument("--n-trials", type=int, default=12,
                        help="Number of trials to sample (default: 12)")
    parser.add_argument("--trial-offset", type=int, default=20,
                        help="Skip first N trials (default: 20)")
    parser.add_argument("--delta-ms", type=float, default=100.0,
                        help="ms before/after Baseline_ON for snapshot mode (default: 100)")
    parser.add_argument("--timeseries", action="store_true",
                        help="Run time-series mode: plot SV trace over ±window-s for N trials")
    parser.add_argument("--window-s", type=float, default=3.0,
                        help="Half-window in seconds for time-series mode (default: 3.0)")
    parser.add_argument("--roi-overlay", action="store_true",
                        help="Save annotated frame showing ROI rect + circular mask")
    parser.add_argument("--trial-idx", type=int, default=None,
                        help="Specific trial index for --roi-overlay (default: trial 30)")
    parser.add_argument("--full-session", action="store_true",
                        help="Process ALL Baseline_ON events and fit a corneal-SV clock model")
    parser.add_argument("--no-auto-calibrate", action="store_true",
                        help="Disable auto-calibration in --full-session; use hardcoded ROI dict instead")
    parser.add_argument("--auto-calibrate", action="store_true",
                        help="Run automated corneal ROI calibration only (detect pupil + build mask)")
    parser.add_argument("--force", action="store_true",
                        help="Force recompute even if cache exists (for --auto-calibrate and --full-session)")
    parser.add_argument("--subject", default="BG_046",
                        help="Subject name for broad eye ROI lookup (default: BG_046)")
    parser.add_argument("--skip-short-trials", action="store_true",
                        help="Skip abort trials and fast FA trials (RT < --min-fa-rt-ms) "
                             "to avoid lick/wheel eye-movement contamination")
    parser.add_argument("--min-fa-rt-ms", type=float, default=500.0,
                        help="FA reaction-time threshold in ms for --skip-short-trials "
                             "(default: 500)")
    parser.add_argument("--validate-corneal", action="store_true",
                        help="Show before/after frames using the cached corneal sync model")
    parser.add_argument("--scan-coarse", action="store_true",
                        help="Scan coarse offsets [--offset-min to --offset-max] to find "
                             "the correct camera-to-NI-DAQ alignment for this session")
    parser.add_argument("--offset-min", type=float, default=1.0,
                        help="Minimum offset to scan in seconds (default: 1.0)")
    parser.add_argument("--offset-max", type=float, default=17.0,
                        help="Maximum offset to scan in seconds (default: 17.0)")
    parser.add_argument("--offset-step", type=float, default=0.5,
                        help="Step size for offset scan in seconds (default: 0.5)")
    parser.add_argument("--update-cache", action="store_true",
                        help="After --scan-coarse, update coarse_offsets.json with the best offset")
    args = parser.parse_args()

    if args.session == "27062025":
        logger.error(f"FATAL: Session {args.session} is explicitly blacklisted. "
                     "Recording has severe non-linear dropped frames. Bypassing.")
        return

    if args.scan_coarse:
        best = scan_coarse_offset(
            session_name=args.session,
            n_trials=args.n_trials,
            offset_min=args.offset_min,
            offset_max=args.offset_max,
            offset_step=args.offset_step,
        )
        if args.update_cache and best is not None:
            import json
            cache_file = os.path.join(_PROJECT_ROOT, "data", "cache", "video_sync",
                                      "coarse_offsets.json")
            cache = {}
            if os.path.exists(cache_file):
                with open(cache_file) as f:
                    cache = json.load(f)
            old = cache.get(str(args.session), None)
            cache[str(args.session)] = float(best)
            with open(cache_file, "w") as f:
                json.dump(cache, f, indent=2)
            logger.info(f"Updated coarse_offsets.json: {args.session} "
                        f"{old} → {best:.1f}s")
            print(f"\ncoarse_offsets.json updated: {args.session}  {old} → {best:.1f}s")
            print("Now re-run --auto-calibrate --force and then --full-session "
                  "to apply the corrected offset.")
    elif args.auto_calibrate:
        run_auto_calibrate(session_name=args.session, subject=args.subject,
                           force=args.force)
    elif args.full_session:
        run_full_session(session_name=args.session, window_s=args.window_s,
                         skip_short=args.skip_short_trials,
                         min_fa_rt_ms=args.min_fa_rt_ms,
                         auto_calibrate=not args.no_auto_calibrate,
                         subject=args.subject)
    elif args.validate_corneal:
        run_corneal_validate(
            session_name=args.session,
            n_trials=args.n_trials,
            trial_offset=args.trial_offset,
        )
    elif args.roi_overlay:
        run_roi_overlay(session_name=args.session, trial_idx=args.trial_idx)
    elif args.timeseries:
        run_timeseries(
            session_name=args.session,
            n_trials=args.n_trials,
            trial_offset=args.trial_offset,
            window_s=args.window_s,
        )
    else:
        run_diagnostic(
            session_name=args.session,
            n_trials=args.n_trials,
            trial_offset=args.trial_offset,
            delta_ms=args.delta_ms,
        )


if __name__ == "__main__":
    main()
