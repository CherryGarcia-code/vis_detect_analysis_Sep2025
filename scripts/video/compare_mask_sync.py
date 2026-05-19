"""Compare mask-based vs fixed-ROI video synchronization.

Runs both detection approaches (data-driven screen mask and fixed
rectangular ROI) on a set of characterized sessions and produces
paired comparison figures and a summary CSV.

Produces:
  - Per-session mask diagnostic: figures/video_sync/mask_sync/{session}_mask_diagnostic.png
  - Summary comparison: figures/video_sync/mask_sync/mask_vs_fixed_summary.png
  - Comparison CSV: data/cache/video_sync/mask_vs_fixed_comparison.csv

Usage:
    cd analysis_suite && py ../scripts/video/compare_mask_sync.py
    cd analysis_suite && py ../scripts/video/compare_mask_sync.py --sessions 27062025
    cd analysis_suite && py ../scripts/video/compare_mask_sync.py --force
"""

import argparse
import gc
import json
import logging
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd

# ── Project paths ────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

from src.visdetect.core.session import load_session
from src.visdetect.core.video_sync import (
    VIDEO_SYNC_DEFAULT_EYE_ROI,
    build_screen_mask,
    detect_onsets_variance,
    fast_coarse_offset,
    find_camera_files,
    fit_clock_model,
    load_camera_metadata,
    load_or_compute_coarse_offset,
)
from src.visdetect.analysis.config import (
    PKL_DIR,
    VIDEO_SYNC_DIR,
    VIDEO_SYNC_FIG_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Default sessions (same 5 from characterization) ──────────────
DEFAULT_SESSIONS = ["27062025", "03072025", "14082025", "29082025", "09092025"]

# Output paths
MASK_FIG_DIR = os.path.join(VIDEO_SYNC_FIG_DIR, "mask_sync")
COMPARISON_CSV = os.path.join(VIDEO_SYNC_DIR, "mask_vs_fixed_comparison.csv")

# Local video directory (checked before network)
LOCAL_VIDEO_DIR = os.path.join(_PROJECT_ROOT, "data", "videos")


# =====================================================================
# Helpers
# =====================================================================


def _find_video_and_metadata(session_name: str) -> tuple:
    """Find eye camera video and metadata, checking local copies first."""
    sn = str(session_name).zfill(8)
    dd, mm, yyyy = sn[:2], sn[2:4], sn[4:]
    yy = yyyy[2:]
    subject = "BG_046"

    # Try local video first
    local_pattern = f"{subject}_{dd}{mm}{yy}_Eye_cam.mp4"
    local_path = os.path.join(LOCAL_VIDEO_DIR, local_pattern)
    if os.path.isfile(local_path):
        video_path = local_path
        logger.info(f"  Using local video: {local_path}")
    else:
        cam_files = find_camera_files(session_name)
        if "eye_cam" not in cam_files:
            raise FileNotFoundError(
                f"No eye camera files for session {session_name}"
            )
        video_path = cam_files["eye_cam"]["video"]
        logger.info(f"  Using network video: {video_path}")

    # Metadata always from network (small file)
    cam_files = find_camera_files(session_name)
    if "eye_cam" not in cam_files:
        raise FileNotFoundError(
            f"No eye camera metadata for session {session_name}"
        )
    meta_path = cam_files["eye_cam"]["metadata"]

    return video_path, meta_path


def _run_detection(
    session_name: str,
    video_path: str,
    meta_path: str,
    baseline_on: np.ndarray,
    coarse_offset: float,
    roi,
    label: str,
) -> dict:
    """Run onset detection + clock fitting with a given ROI, return metrics."""
    t0 = time.time()

    detection = detect_onsets_variance(
        video_path, meta_path, baseline_on,
        rough_offset_s=coarse_offset,
        roi=roi,
        progress=False,
    )

    elapsed_detect = time.time() - t0

    if detection.n_detected < 10:
        logger.warning(
            f"  [{label}] Only {detection.n_detected} detections — "
            f"too few to fit clock model"
        )
        return {
            "session": session_name,
            "method": label,
            "n_trials": detection.n_trials,
            "n_detected": detection.n_detected,
            "detection_rate": detection.detection_rate,
            "rmse_ms": np.nan,
            "max_residual_ms": np.nan,
            "n_anchors": 0,
            "cv_rmse_ms": np.nan,
            "slope_ppm": np.nan,
            "quality": "failed",
            "median_confidence": np.nan,
            "time_s": elapsed_detect,
        }

    sync_result = fit_clock_model(
        detection.detected_cam_s, detection.detected_nidaq_s,
        n_baseline_on=len(baseline_on),
    )
    elapsed_total = time.time() - t0

    return {
        "session": session_name,
        "method": label,
        "n_trials": detection.n_trials,
        "n_detected": detection.n_detected,
        "detection_rate": detection.detection_rate,
        "rmse_ms": sync_result.rmse_ms,
        "max_residual_ms": sync_result.max_residual_ms,
        "n_anchors": sync_result.n_anchors,
        "cv_rmse_ms": sync_result.cv_rmse_ms,
        "slope_ppm": sync_result.slope_ppm,
        "quality": sync_result.quality,
        "median_confidence": float(np.median(detection.confidence)),
        "time_s": elapsed_total,
    }


def _plot_mask_diagnostic(
    session_name: str,
    video_path: str,
    mask: np.ndarray,
    mask_info: dict,
    save_path: str,
):
    """4-panel mask diagnostic figure."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)  # grab an early frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        logger.warning(f"  Cannot read sample frame for diagnostic")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_diff = mask_info.get("avg_diff", np.zeros_like(gray, dtype=np.float32))

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)

    # Panel A: Raw sample frame
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(gray, cmap="gray", aspect="auto")
    ax1.set_title("A. Sample frame (grayscale)", fontsize=10)
    ax1.set_xlabel("x (pixels)")
    ax1.set_ylabel("y (pixels)")

    # Panel B: Trial-to-trial std of (post-pre) difference
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(avg_diff, cmap="hot", aspect="auto")
    plt.colorbar(im, ax=ax2, label="Std of (post-pre) across trials")
    ax2.set_title("B. Trial-to-trial std (post-pre)", fontsize=10)
    ax2.set_xlabel("x (pixels)")
    ax2.set_ylabel("y (pixels)")

    # Panel C: Mask overlaid on frame
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(gray, cmap="gray", aspect="auto")
    mask_overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    mask_overlay[mask, 0] = 1.0   # red
    mask_overlay[mask, 3] = 0.4   # alpha
    ax3.imshow(mask_overlay, aspect="auto")
    # Show x_min cutoff line if present
    x_min_val = mask_info.get("x_min", 0)
    if x_min_val > 0:
        ax3.axvline(x_min_val, color="cyan", ls="--", lw=1.5, alpha=0.8)
        ax3.text(
            x_min_val + 5, mask.shape[0] - 20, f"x_min={x_min_val}",
            color="cyan", fontsize=8, va="bottom",
        )
    ax3.set_title("C. Screen mask overlay", fontsize=10)
    ax3.set_xlabel("x (pixels)")
    ax3.set_ylabel("y (pixels)")

    # Panel D: Text stats
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    stats_text = (
        f"Session: {session_name}\n"
        f"Screen pixels: {mask_info['n_screen_pixels']:,}\n"
        f"Screen fraction: {mask_info['screen_fraction']:.1%}\n"
        f"Otsu threshold: {mask_info['threshold']:.2f}\n"
        f"Otsu quality: {mask_info['otsu_quality']:.3f}\n"
        f"Transitions used: {mask_info['n_transitions_used']}\n"
        f"x_min: {mask_info.get('x_min', 'n/a')}\n"
        f"min_component: {mask_info.get('min_component_area', 'n/a')}\n"
        f"Frame shape: {mask.shape}"
    )
    ax4.text(
        0.1, 0.7, stats_text, transform=ax4.transAxes,
        fontsize=11, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )
    ax4.set_title("D. Mask statistics", fontsize=10)

    fig.suptitle(
        f"Screen Mask Diagnostic — {session_name}", fontsize=12, fontweight="bold"
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved mask diagnostic: {save_path}")


def _plot_summary(df: pd.DataFrame, save_path: str):
    """4-panel summary comparison figure."""
    sessions = df["session"].unique()
    n_sessions = len(sessions)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    x = np.arange(n_sessions)
    bar_w = 0.35

    def _get_values(metric):
        fixed = []
        mask = []
        for s in sessions:
            sdf = df[df["session"] == s]
            fv = sdf.loc[sdf["method"] == "fixed", metric]
            mv = sdf.loc[sdf["method"] == "mask", metric]
            fixed.append(fv.values[0] if len(fv) > 0 else np.nan)
            mask.append(mv.values[0] if len(mv) > 0 else np.nan)
        return np.array(fixed), np.array(mask)

    # Panel A: Detection rate
    ax1 = fig.add_subplot(gs[0, 0])
    fixed_dr, mask_dr = _get_values("detection_rate")
    ax1.bar(x - bar_w / 2, fixed_dr * 100, bar_w, label="Fixed ROI", color="#7986CB")
    ax1.bar(x + bar_w / 2, mask_dr * 100, bar_w, label="Screen mask", color="#4CAF50")
    ax1.set_ylabel("Detection rate (%)")
    ax1.set_title("A. Detection rate", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([s[-4:] for s in sessions], fontsize=8)
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 105)

    # Panel B: RMSE
    ax2 = fig.add_subplot(gs[0, 1])
    fixed_rmse, mask_rmse = _get_values("rmse_ms")
    ax2.bar(x - bar_w / 2, fixed_rmse, bar_w, label="Fixed ROI", color="#7986CB")
    ax2.bar(x + bar_w / 2, mask_rmse, bar_w, label="Screen mask", color="#4CAF50")
    ax2.set_ylabel("RMSE (ms)")
    ax2.set_title("B. Clock model RMSE", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s[-4:] for s in sessions], fontsize=8)
    ax2.legend(fontsize=8)
    ax2.axhline(20, color="gray", ls="--", lw=0.8, label="Good threshold")

    # Panel C: Number of anchors
    ax3 = fig.add_subplot(gs[1, 0])
    fixed_anch, mask_anch = _get_values("n_anchors")
    ax3.bar(x - bar_w / 2, fixed_anch, bar_w, label="Fixed ROI", color="#7986CB")
    ax3.bar(x + bar_w / 2, mask_anch, bar_w, label="Screen mask", color="#4CAF50")
    ax3.set_ylabel("Number of anchors")
    ax3.set_title("C. Anchors after outlier rejection", fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels([s[-4:] for s in sessions], fontsize=8)
    ax3.legend(fontsize=8)

    # Panel D: Quality tier table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    cell_text = []
    cell_colors = []
    tier_color = {"good": "#c8e6c9", "review": "#fff9c4", "failed": "#ffcdd2"}
    for s in sessions:
        sdf = df[df["session"] == s]
        fq = sdf.loc[sdf["method"] == "fixed", "quality"].values
        mq = sdf.loc[sdf["method"] == "mask", "quality"].values
        fq = fq[0] if len(fq) > 0 else "n/a"
        mq = mq[0] if len(mq) > 0 else "n/a"
        cell_text.append([s[-4:], fq, mq])
        cell_colors.append([
            "white",
            tier_color.get(fq, "white"),
            tier_color.get(mq, "white"),
        ])

    table = ax4.table(
        cellText=cell_text,
        colLabels=["Session", "Fixed ROI", "Screen mask"],
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax4.set_title("D. Quality tiers", fontsize=10)

    fig.suptitle(
        "Mask vs Fixed-ROI Video Sync Comparison",
        fontsize=13, fontweight="bold",
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved summary figure: {save_path}")


# =====================================================================
# Main
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions", nargs="+", default=DEFAULT_SESSIONS,
        help="Session names to compare (default: 5 characterized sessions)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if cached results exist",
    )
    args = parser.parse_args()

    os.makedirs(MASK_FIG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(COMPARISON_CSV), exist_ok=True)

    records = []

    for session_name in args.sessions:
        sname = str(session_name).zfill(8)
        logger.info(f"\n{'='*60}")
        logger.info(f"Session {sname}")
        logger.info(f"{'='*60}")

        # Load session for Baseline_ON events
        pkl_path = os.path.join(PKL_DIR, f"BG_046_{sname}.pkl")
        if not os.path.exists(pkl_path):
            logger.warning(f"  PKL not found: {pkl_path}, skipping")
            continue

        sess = load_session(pkl_path)
        baseline_on = np.asarray(
            sess.ni_events.get("Baseline_ON", []), dtype=float
        ).flatten()
        logger.info(f"  {len(baseline_on)} Baseline_ON events")
        del sess
        gc.collect()

        # Find camera files
        try:
            video_path, meta_path = _find_video_and_metadata(sname)
        except FileNotFoundError as e:
            logger.warning(f"  {e}, skipping")
            continue

        # Coarse offset (cached)
        coarse_offset = load_or_compute_coarse_offset(
            sname, video_path, meta_path, baseline_on
        )
        logger.info(f"  Coarse offset = {coarse_offset:.2f}s")

        # ── Fixed ROI detection ────────────────────────────────────
        logger.info(f"  Running fixed-ROI detection...")
        fixed_result = _run_detection(
            sname, video_path, meta_path, baseline_on,
            coarse_offset, VIDEO_SYNC_DEFAULT_EYE_ROI, "fixed",
        )
        records.append(fixed_result)
        logger.info(
            f"  Fixed: {fixed_result['n_detected']}/{fixed_result['n_trials']} "
            f"detected, RMSE={fixed_result['rmse_ms']:.1f}ms, "
            f"quality={fixed_result['quality']}"
        )

        # ── Mask-based detection ───────────────────────────────────
        logger.info(f"  Building screen mask...")
        try:
            mask, mask_info = build_screen_mask(
                video_path, meta_path,
                baseline_on, coarse_offset,
                session_name=sname,
                force=args.force,
            )

            # Diagnostic figure
            diag_path = os.path.join(
                MASK_FIG_DIR, f"{sname}_mask_diagnostic.png"
            )
            _plot_mask_diagnostic(sname, video_path, mask, mask_info, diag_path)

            logger.info(f"  Running mask-based detection...")
            mask_result = _run_detection(
                sname, video_path, meta_path, baseline_on,
                coarse_offset, mask, "mask",
            )
            mask_result["screen_fraction"] = mask_info["screen_fraction"]
            mask_result["otsu_quality"] = mask_info["otsu_quality"]

        except Exception as e:
            logger.error(f"  Mask construction failed: {e}")
            mask_result = {
                "session": sname,
                "method": "mask",
                "n_trials": len(baseline_on),
                "n_detected": 0,
                "detection_rate": 0.0,
                "rmse_ms": np.nan,
                "max_residual_ms": np.nan,
                "n_anchors": 0,
                "cv_rmse_ms": np.nan,
                "slope_ppm": np.nan,
                "quality": "failed",
                "median_confidence": np.nan,
                "time_s": 0.0,
            }

        records.append(mask_result)
        logger.info(
            f"  Mask:  {mask_result['n_detected']}/{mask_result['n_trials']} "
            f"detected, RMSE={mask_result.get('rmse_ms', np.nan):.1f}ms, "
            f"quality={mask_result.get('quality', 'n/a')}"
        )

    # ── Save comparison CSV ────────────────────────────────────────
    df = pd.DataFrame(records)
    df.to_csv(COMPARISON_CSV, index=False)
    logger.info(f"\nSaved comparison CSV: {COMPARISON_CSV}")

    # ── Summary figure ─────────────────────────────────────────────
    if len(df) > 0:
        summary_path = os.path.join(MASK_FIG_DIR, "mask_vs_fixed_summary.png")
        _plot_summary(df, summary_path)

    # ── Print summary table ────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 70)
    for session in df["session"].unique():
        sdf = df[df["session"] == session]
        for _, row in sdf.iterrows():
            logger.info(
                f"  {session} [{row['method']:>5s}]: "
                f"det={row['detection_rate']:.0%}, "
                f"RMSE={row['rmse_ms']:.1f}ms, "
                f"anchors={row['n_anchors']}, "
                f"quality={row['quality']}"
            )


if __name__ == "__main__":
    main()
