"""Batch sync all QC-passing sessions with eye camera video.

Iterates over the staging manifest, loads each session to extract
Baseline_ON events, and calls sync_session(). For sessions with
successful sync, generates a per-session validation figure (mouth-zoom
video frames + spike rasters + PSTH from a lick-responsive neuron).

Produces:
  - Per-session sync JSON in data/cache/video_sync/
  - Per-session diagnostic PNG in figures/video_sync/
  - Per-session validation PNG in figures/video_sync/ (optional)
  - Summary CSV at data/cache/video_sync/batch_sync_summary.csv

Usage:
    cd analysis_suite && py ../scripts/video/batch_sync_sessions.py [--force] [--skip-validation]
"""

import os
import sys
import gc
import json
import time
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ── Project paths ────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

from src.visdetect.core.video_sync import (
    find_camera_files,
    load_camera_metadata,
    load_video_sync,
    nidaq_to_camera,
    sync_session,
)
from src.visdetect.analysis.lick import compute_fa_lick_responsiveness
from src.visdetect.analysis.constants import (
    DEFAULT_SIGMA_MS, LICK_HARDWARE_DELAY_MS,
)
from src.visdetect.analysis.config import VIDEO_SYNC_DIR, VIDEO_SYNC_FIG_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Validation figure config (matches sync_validation_figure.py) ─
N_TRIALS_PER_OUTCOME = 5
FRAME_WINDOW_MS = 200
FRAME_STEP_MS = 40
PETH_WINDOW = (-0.5, 0.5)
PETH_BIN = 0.010
RASTER_WINDOW = (-0.3, 0.3)
SEED = 42
FRAME_CROP = (683, 1024)
HIT_COLOR = "#2166ac"
FA_COLOR = "#b2182b"


# ── Validation figure helpers (extracted from sync_validation_figure.py) ─

def _get_lick_times(sess, shift_ms):
    """Get absolute lick times for Hit and FA trials."""
    baseline_on = np.asarray(
        sess.ni_events.get("Baseline_ON", []), dtype=float
    ).flatten()
    change_on = np.asarray(
        sess.ni_events.get("Change_ON", []), dtype=float
    ).flatten()
    shift_s = shift_ms / 1000.0

    entries = []
    for i, trial in enumerate(sess.trials):
        outcome = (trial.trialoutcome or "").lower()
        if outcome not in ("hit", "fa"):
            continue
        rt_dict = trial.reactiontimes or {}

        if outcome == "hit":
            rt = rt_dict.get("RT", rt_dict.get("Hit", rt_dict.get("hit", np.nan)))
            if i >= len(change_on) or np.isnan(rt):
                continue
            t_change = change_on[i]
            if t_change == 0 or np.isnan(t_change):
                continue
            lick_time = t_change + rt - shift_s
        elif outcome == "fa":
            rt = rt_dict.get("FA", rt_dict.get("fa", np.nan))
            if i >= len(baseline_on) or np.isnan(rt):
                continue
            lick_time = baseline_on[i] + rt - shift_s

        entries.append((i, outcome, lick_time))
    return entries


def _find_lick_responsive_neuron(sess, cluster_ids):
    """Use lick.py definition; returns (best_cid, p, n_sig) or None."""
    try:
        result = compute_fa_lick_responsiveness(sess, good_ids=cluster_ids)
    except Exception as e:
        logger.warning(f"  lick responsiveness failed: {e}")
        return None

    table = result.table
    sig = table[table["is_significant"]].copy()
    n_sig = len(sig)

    if n_sig == 0:
        logger.warning("  No significant lick-responsive neurons")
        return None

    sig["abs_delta"] = sig["delta_mean"].abs()
    best = sig.sort_values("abs_delta", ascending=False).iloc[0]
    best_cid = int(best["cluster_id"])
    best_p = float(best["p_value"])
    logger.info(f"  Best neuron: cluster {best_cid}, p={best_p:.2e}, "
                f"delta={best['delta_mean']:.2f} Hz ({n_sig} sig)")
    return best_cid, best_p, n_sig


def _extract_frames_around_event(video_path, metadata_ts_ms, event_cam_ms,
                                 window_ms, step_ms, crop=None):
    """Extract video frames at regular intervals around an event."""
    import cv2

    rel_times = np.arange(-window_ms, window_ms + step_ms, step_ms)
    target_cam_ms = event_cam_ms + rel_times
    frame_indices = [int(np.argmin(np.abs(metadata_ts_ms - t)))
                     for t in target_cam_ms]

    cap = cv2.VideoCapture(video_path)
    try:
        frames = []
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if crop is not None:
                    gray = gray[crop[0]:crop[1], :]
                frames.append(gray)
            else:
                frames.append(None)
    finally:
        cap.release()

    return frames, rel_times


def _get_spikes_in_window(sess, cluster_id, event_time_s, window_s):
    """Get spike times relative to event for a given cluster."""
    for cl in sess.clusters:
        if cl.cluster_id == cluster_id:
            spk = np.array(cl.spike_times)
            mask = ((spk >= event_time_s + window_s[0]) &
                    (spk < event_time_s + window_s[1]))
            return spk[mask] - event_time_s
    return np.array([])


def _compute_peri_lick_psth(sess, cluster_id, lick_times, window, bin_size,
                            n_boot=1000, ci=95, seed=42):
    """Compute smoothed PSTH with bootstrap CI around lick events."""
    from scipy.ndimage import gaussian_filter1d

    edges = np.arange(window[0], window[1] + bin_size, bin_size)
    centers = edges[:-1] + bin_size / 2
    sigma_bins = (DEFAULT_SIGMA_MS / 1000.0) / bin_size
    n_trials = len(lick_times)

    trial_rates = np.zeros((n_trials, len(centers)))
    for cl in sess.clusters:
        if cl.cluster_id != cluster_id:
            continue
        spk = np.array(cl.spike_times)
        for j, lt in enumerate(lick_times):
            rel = spk - lt
            mask = (rel >= window[0]) & (rel < window[1])
            if mask.any():
                hist, _ = np.histogram(rel[mask], bins=edges)
                trial_rates[j] = gaussian_filter1d(hist / bin_size,
                                                   sigma=sigma_bins)
        break

    rate_mean = trial_rates.mean(axis=0)
    rng = np.random.default_rng(seed)
    boot_means = np.zeros((n_boot, len(centers)))
    for b in range(n_boot):
        idx = rng.choice(n_trials, n_trials, replace=True)
        boot_means[b] = trial_rates[idx].mean(axis=0)
    alpha = (100 - ci) / 2
    ci_low = np.percentile(boot_means, alpha, axis=0)
    ci_high = np.percentile(boot_means, 100 - alpha, axis=0)

    return centers, rate_mean, ci_low, ci_high


def generate_sync_validation_figure(session_name, sess, sync_params,
                                    save_path):
    """Generate mouth-zoom validation figure for one session.

    Parameters
    ----------
    session_name : str
    sess : SessionData
    sync_params : dict
        The eye_cam entry from load_video_sync().
    save_path : str
        Output PNG path.

    Returns
    -------
    bool
        True if figure was generated, False if skipped.
    """
    from visdetect.analysis.utils import get_good_cluster_ids

    slope = sync_params["slope"]
    offset = sync_params["offset"]

    # Find camera files
    try:
        cam_files = find_camera_files(session_name)
        eye_video = cam_files["eye_cam"]["video"]
        eye_meta = cam_files["eye_cam"]["metadata"]
    except (KeyError, FileNotFoundError):
        logger.warning(f"  Cannot find camera files for validation figure")
        return False

    ts_ms, _, _ = load_camera_metadata(eye_meta)

    # Lick times
    lick_corr = _get_lick_times(sess, shift_ms=LICK_HARDWARE_DELAY_MS)
    lick_raw = _get_lick_times(sess, shift_ms=0.0)

    hit_corr = [e for e in lick_corr if e[1] == "hit"]
    fa_corr = [e for e in lick_corr if e[1] == "fa"]
    hit_raw = [e for e in lick_raw if e[1] == "hit"]
    fa_raw = [e for e in lick_raw if e[1] == "fa"]

    if len(hit_corr) < 2 or len(fa_corr) < 2:
        logger.warning(f"  Too few Hit/FA trials for validation figure")
        return False

    raw_by_trial = {e[0]: e[2] for e in lick_raw}

    # Neuron selection
    cluster_ids = get_good_cluster_ids(sess)
    neuron_result = _find_lick_responsive_neuron(sess, cluster_ids)
    if neuron_result is None:
        return False
    best_cid, best_p, n_sig = neuron_result

    # Trial selection
    rng = np.random.default_rng(SEED)
    n_hit = min(N_TRIALS_PER_OUTCOME, len(hit_corr))
    n_fa = min(N_TRIALS_PER_OUTCOME, len(fa_corr))
    selected_hits = [hit_corr[i] for i in rng.choice(len(hit_corr), n_hit,
                                                      replace=False)]
    selected_fas = [fa_corr[i] for i in rng.choice(len(fa_corr), n_fa,
                                                    replace=False)]
    selected = selected_hits + selected_fas
    n_total = len(selected)

    # Build figure
    fig_height = n_total * 1.6 + 3.5
    fig = plt.figure(figsize=(18, fig_height))
    gs_main = gridspec.GridSpec(
        n_total + 1, 1,
        height_ratios=[1] * n_total + [2.0],
        hspace=0.15,
        top=0.94, bottom=0.04, left=0.02, right=0.98,
    )

    outcome_colors = {"hit": HIT_COLOR, "fa": FA_COLOR}

    for row_idx, (trial_idx, outcome, lick_time_corr) in enumerate(selected):
        lick_cam_ms = float(nidaq_to_camera(lick_time_corr, slope, offset))
        frames, rel_times = _extract_frames_around_event(
            eye_video, ts_ms, lick_cam_ms, FRAME_WINDOW_MS, FRAME_STEP_MS,
            crop=FRAME_CROP,
        )
        n_cols = len(frames)
        rel_spikes_corr = _get_spikes_in_window(
            sess, best_cid, lick_time_corr, RASTER_WINDOW)
        lick_time_raw = raw_by_trial.get(trial_idx)
        rel_spikes_raw = (_get_spikes_in_window(
            sess, best_cid, lick_time_raw, RASTER_WINDOW)
            if lick_time_raw is not None else np.array([]))

        gs_row = gridspec.GridSpecFromSubplotSpec(
            1, n_cols + 3,
            subplot_spec=gs_main[row_idx],
            width_ratios=[1] * n_cols + [0.1, 2.0, 2.0],
            wspace=0.03,
        )
        color = outcome_colors[outcome]

        first_ax = None
        for col_idx, (frame, rel_t) in enumerate(zip(frames, rel_times)):
            ax = fig.add_subplot(gs_row[0, col_idx])
            if col_idx == 0:
                first_ax = ax
            if frame is not None:
                ax.imshow(frame, cmap="gray", aspect="auto", vmin=0, vmax=255)
            else:
                ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)
            if rel_t == 0:
                for spine in ax.spines.values():
                    spine.set_linewidth(4)
            if row_idx == 0:
                ax.set_title(f"{rel_t:+.0f} ms", fontsize=7, pad=2)

        first_ax.set_ylabel(
            f"{'Hit' if outcome == 'hit' else 'FA'} #{trial_idx}",
            fontsize=8, rotation=0, labelpad=50, va="center",
            color=color, fontweight="bold",
        )

        # Raster: corrected
        ax_r1 = fig.add_subplot(gs_row[0, n_cols + 1])
        if len(rel_spikes_corr) > 0:
            ax_r1.eventplot([rel_spikes_corr], lineoffsets=0.5,
                            linelengths=0.8, colors=[color], linewidths=0.8)
        ax_r1.axvline(0, color="k", ls="--", lw=0.8, alpha=0.6)
        ax_r1.set_xlim(RASTER_WINDOW)
        ax_r1.set_ylim(0, 1)
        ax_r1.set_yticks([])
        if row_idx == 0:
            ax_r1.set_title("corrected\n(-200ms)", fontsize=6, pad=2)
        if row_idx == n_total - 1:
            ax_r1.set_xlabel("Time (s)", fontsize=6)
            ax_r1.tick_params(axis="x", labelsize=5)
        else:
            ax_r1.set_xticks([])
        for spine in ["top", "right", "left"]:
            ax_r1.spines[spine].set_visible(False)

        # Raster: uncorrected
        ax_r2 = fig.add_subplot(gs_row[0, n_cols + 2])
        if len(rel_spikes_raw) > 0:
            ax_r2.eventplot([rel_spikes_raw], lineoffsets=0.5,
                            linelengths=0.8, colors=[color], linewidths=0.8,
                            linestyles="dashed")
        ax_r2.axvline(0, color="k", ls="--", lw=0.8, alpha=0.6)
        ax_r2.set_xlim(RASTER_WINDOW)
        ax_r2.set_ylim(0, 1)
        ax_r2.set_yticks([])
        if row_idx == 0:
            ax_r2.set_title("uncorrected\n(raw software)", fontsize=6, pad=2)
        if row_idx == n_total - 1:
            ax_r2.set_xlabel("Time (s)", fontsize=6)
            ax_r2.tick_params(axis="x", labelsize=5)
        else:
            ax_r2.set_xticks([])
        for spine in ["top", "right", "left"]:
            ax_r2.spines[spine].set_visible(False)

    # Bottom PSTHs
    gs_psth = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_main[n_total], wspace=0.3,
    )
    for panel_idx, (label, h_entries, f_entries) in enumerate([
        ("t=0: software - 200ms (estimated contact)", hit_corr, fa_corr),
        ("t=0: raw software detection time", hit_raw, fa_raw),
    ]):
        ax = fig.add_subplot(gs_psth[0, panel_idx])
        h_times = [e[2] for e in h_entries]
        f_times = [e[2] for e in f_entries]
        t_h, r_h, ci_lo_h, ci_hi_h = _compute_peri_lick_psth(
            sess, best_cid, h_times, PETH_WINDOW, PETH_BIN)
        t_f, r_f, ci_lo_f, ci_hi_f = _compute_peri_lick_psth(
            sess, best_cid, f_times, PETH_WINDOW, PETH_BIN)
        ax.fill_between(t_h * 1000, ci_lo_h, ci_hi_h,
                        color=HIT_COLOR, alpha=0.15)
        ax.fill_between(t_f * 1000, ci_lo_f, ci_hi_f,
                        color=FA_COLOR, alpha=0.15)
        ax.plot(t_h * 1000, r_h, color=HIT_COLOR, lw=2,
                label=f"Hit (n={len(h_times)})")
        ax.plot(t_f * 1000, r_f, color=FA_COLOR, lw=2,
                label=f"FA (n={len(f_times)})")
        ax.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel("Time from lick (ms)")
        if panel_idx == 0:
            ax.set_ylabel("Firing rate (Hz)")
        ax.set_title(f"Cluster {best_cid} - {label}", fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Video-Neural Sync Validation - Session {session_name}\n"
        f"Cluster {best_cid} (lick.py: Wilcoxon p={best_p:.1e}, "
        f"{n_sig} significant units) | "
        f"Sync: RMSE={sync_params['rmse_ms']:.1f} ms, "
        f"quality={sync_params['quality']}",
        fontsize=11, y=0.99,
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


# ── Main batch processing ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch sync sessions")
    parser.add_argument("--force", action="store_true",
                        help="Force re-sync even if cached")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip validation figure generation")
    args = parser.parse_args()

    from visdetect.suite.loader import load_staging_manifest, load_session

    # Load per-session ROI overrides (created by select_roi.py)
    roi_override_path = os.path.join(VIDEO_SYNC_DIR, "session_rois.json")
    roi_overrides = {}
    if os.path.exists(roi_override_path):
        with open(roi_override_path) as f:
            roi_overrides = json.load(f)
        logger.info(f"Loaded {len(roi_overrides)} ROI overrides from {roi_override_path}")

    manifest = load_staging_manifest(qc_only=True)
    n_sessions = len(manifest)
    logger.info(f"Processing {n_sessions} QC-passing sessions")

    os.makedirs(VIDEO_SYNC_DIR, exist_ok=True)
    os.makedirs(VIDEO_SYNC_FIG_DIR, exist_ok=True)

    records = []

    for idx, (_, row) in enumerate(manifest.iterrows()):
        sname = str(int(row["session_name"])).zfill(8)
        stage = row["stage"]
        logger.info(f"\n[{idx+1}/{n_sessions}] Session {sname} ({stage})")

        t0 = time.time()
        record = {
            "session_name": sname,
            "stage": stage,
            "quality": "error",
            "rmse_ms": np.nan,
            "coverage": np.nan,
            "n_anchors": 0,
            "slope_ppm": np.nan,
            "elapsed_s": 0.0,
            "validation_fig": False,
        }

        # Load session
        try:
            sess = load_session(sname)
        except FileNotFoundError:
            logger.warning(f"  pkl not found, skipping")
            record["quality"] = "no_pkl"
            records.append(record)
            continue

        # Extract Baseline_ON
        baseline_on = np.asarray(
            sess.ni_events.get("Baseline_ON", []), dtype=float
        ).flatten()
        baseline_on = baseline_on[~np.isnan(baseline_on)]

        if len(baseline_on) < 10:
            logger.warning(f"  Too few Baseline_ON events ({len(baseline_on)})")
            record["quality"] = "no_events"
            del sess
            gc.collect()
            records.append(record)
            continue

        # Sync
        session_roi = roi_overrides.get(sname)
        if session_roi is not None:
            # ROI data comes as-is from JSON — _build_roi_mask handles
            # rectangles, single polygons, and multi-polygon formats.
            logger.info(f"  Using custom ROI override")
        try:
            result = sync_session(
                sname, baseline_on, force=args.force,
                roi=session_roi,
            )
        except FileNotFoundError:
            logger.warning(f"  No camera files found")
            record["quality"] = "no_camera"
            del sess
            gc.collect()
            records.append(record)
            continue
        except Exception as e:
            logger.error(f"  Sync failed: {e}")
            del sess
            gc.collect()
            records.append(record)
            continue

        # Extract results
        eye_sync = result.get("eye_cam", {})
        if isinstance(eye_sync, dict):
            record["quality"] = eye_sync.get("quality", "unknown")
            record["rmse_ms"] = eye_sync.get("rmse_ms", np.nan)
            record["coverage"] = eye_sync.get("coverage", np.nan)
            record["n_anchors"] = eye_sync.get("n_anchors", 0)
            record["slope_ppm"] = eye_sync.get("slope_ppm", np.nan)

        # Validation figure
        if (not args.skip_validation
                and record["quality"] in ("good", "review")):
            logger.info(f"  Generating validation figure...")
            val_path = os.path.join(
                VIDEO_SYNC_FIG_DIR,
                f"{sname}_sync_validation_mouth_zoom.png",
            )
            try:
                ok = generate_sync_validation_figure(
                    sname, sess, eye_sync, val_path)
                record["validation_fig"] = ok
                if ok:
                    logger.info(f"  Saved: {val_path}")
            except Exception as e:
                logger.warning(f"  Validation figure failed: {e}")

        record["elapsed_s"] = round(time.time() - t0, 1)
        records.append(record)
        logger.info(f"  quality={record['quality']}, "
                     f"RMSE={record['rmse_ms']:.1f} ms, "
                     f"elapsed={record['elapsed_s']:.0f}s")

        del sess
        gc.collect()

    # Save summary
    df = pd.DataFrame(records)
    summary_path = os.path.join(VIDEO_SYNC_DIR, "batch_sync_summary.csv")
    df.to_csv(summary_path, index=False)
    logger.info(f"\nSaved summary: {summary_path}")

    # Report
    logger.info("\n" + "=" * 60)
    logger.info("BATCH SYNC SUMMARY")
    logger.info("=" * 60)
    quality_counts = df["quality"].value_counts()
    for q, n in quality_counts.items():
        logger.info(f"  {q:>12s}: {n}")

    good = df[df["quality"] == "good"]
    if len(good) > 0:
        logger.info(f"\n  Good sessions: {len(good)}")
        logger.info(f"  Median RMSE: {good['rmse_ms'].median():.1f} ms")
        logger.info(f"  Max RMSE: {good['rmse_ms'].max():.1f} ms")
        logger.info(f"  Median coverage: {good['coverage'].median():.1%}")

    n_val = df["validation_fig"].sum()
    logger.info(f"\n  Validation figures generated: {n_val}/{len(df)}")
    logger.info(f"  Total elapsed: {df['elapsed_s'].sum():.0f}s")


if __name__ == "__main__":
    main()
