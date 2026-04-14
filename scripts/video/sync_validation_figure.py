"""Video-Neural Sync Validation Figure.

Proof-of-concept: shows aligned video frames + spike rasters (corrected and
uncorrected timing) for a lick-responsive neuron around lick events (Hit and
FA trials).  Lick-responsive neurons are identified using the formal definition
from ``visdetect.analysis.lick`` (Wilcoxon signed-rank test, FA-aligned,
late-baseline licks only).

This is a standalone validation script, not a registered analysis_suite figure.

Usage:
    cd analysis_suite && py ../scripts/video/sync_validation_figure.py
"""

import os
import sys
import gc

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ── Project paths ────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "analysis_suite"))
sys.path.insert(0, _PROJECT_ROOT)

from src.visdetect.core.video_sync import (
    find_camera_files,
    load_camera_metadata,
    load_video_sync,
    nidaq_to_camera,
)
from src.visdetect.analysis.lick import compute_fa_lick_responsiveness
from src.visdetect.analysis.constants import (
    DEFAULT_SIGMA_MS, FA_RT_SPLIT, LICK_HARDWARE_DELAY_MS,
)

# ── Configuration ────────────────────────────────────────────────
SESSION_NAME = "01072025"
N_TRIALS_PER_OUTCOME = 5  # 5 Hit + 5 FA
FRAME_WINDOW_MS = 200  # +/-200 ms around lick
FRAME_STEP_MS = 40  # sample every 40 ms (~2 eye-cam frames)
PETH_WINDOW = (-0.5, 0.5)  # s around lick for PSTH
PETH_BIN = 0.010  # 10 ms bins for smooth PSTH
RASTER_WINDOW = (-0.3, 0.3)  # s around lick for spike raster strip
SEED = 42

# Crop: (y_start, y_end) in pixels. None = full frame.
# Lower third of 1024px frame = mouth region
FRAME_CROP = (683, 1024)

# Colors — deliberate diverging palette (blue/red) for corrected vs uncorrected
# timing comparison. These differ from the canonical OUTCOME_COLORS (green/orange)
# which are designed for Hit/Miss/FA outcome classification.
HIT_COLOR = "#2166ac"
FA_COLOR = "#b2182b"
OUTCOME_COLORS = {"hit": HIT_COLOR, "fa": FA_COLOR}

# Figure output
FIG_DIR = os.path.join(_PROJECT_ROOT, "figures", "video_sync")
os.makedirs(FIG_DIR, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────

def load_session_data(session_name):
    """Load session .pkl via analysis_suite loader."""
    from loader import load_session as suite_load_session
    return suite_load_session(session_name)


def get_lick_times(sess, shift_ms=LICK_HARDWARE_DELAY_MS):
    """Get absolute lick times for Hit and FA trials.

    Parameters
    ----------
    shift_ms : float
        Hardware correction subtracted from software lick time.
        200.0 = standard correction (estimated actual contact, LICK_HARDWARE_DELAY_MS).
        0.0 = raw software detection time.

    Returns list of (trial_idx, outcome, lick_time_nidaq_s).
    """
    baseline_on = np.asarray(sess.ni_events.get("Baseline_ON", []), dtype=float).flatten()
    change_on = np.asarray(sess.ni_events.get("Change_ON", []), dtype=float).flatten()
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


def find_lick_responsive_neuron_formal(sess, cluster_ids):
    """Use the formal lick.py definition to find lick-responsive neurons.

    Method: compute_fa_lick_responsiveness() —
      - FA trials only, delay >= 3s (FA_RT_SPLIT)
      - Baseline window [-1.75, -1.25]s vs pre-movement [-0.3, -0.15]s
      - Wilcoxon signed-rank test, p < 0.05
      - Alignment: raw software time (no 200ms correction)

    Returns (best_cluster_id, p_value, n_significant, lick_table).
    """
    result = compute_fa_lick_responsiveness(sess, good_ids=cluster_ids)
    table = result.table

    sig = table[table["is_significant"]].copy()
    n_sig = len(sig)
    print(f"  lick.py: {n_sig}/{len(table)} neurons significant (Wilcoxon p<0.05)")

    if n_sig == 0:
        # Fall back to most modulated unit
        print("  WARNING: No formally significant units. Using largest |delta_mean|.")
        table["abs_delta"] = table["delta_mean"].abs()
        best = table.sort_values("abs_delta", ascending=False).iloc[0]
    else:
        # Pick the most modulated significant unit
        sig["abs_delta"] = sig["delta_mean"].abs()
        best = sig.sort_values("abs_delta", ascending=False).iloc[0]

    best_cid = int(best["cluster_id"])
    best_p = best["p_value"]
    print(f"  Best: cluster {best_cid}, p={best_p:.2e}, "
          f"delta={best['delta_mean']:.2f} Hz, n_events={int(best['n_events'])}")

    return best_cid, best_p, n_sig, table


def extract_frames_around_event(video_path, metadata_ts_ms, event_cam_ms,
                                window_ms, step_ms, crop=None):
    """Extract video frames at regular intervals around an event."""
    import cv2

    rel_times = np.arange(-window_ms, window_ms + step_ms, step_ms)
    target_cam_ms = event_cam_ms + rel_times

    frame_indices = [int(np.argmin(np.abs(metadata_ts_ms - t))) for t in target_cam_ms]

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


def get_spikes_in_window(sess, cluster_id, event_time_s, window_s):
    """Get spike times relative to event for a given cluster."""
    for cl in sess.clusters:
        if cl.cluster_id == cluster_id:
            spk = np.array(cl.spike_times)
            mask = (spk >= event_time_s + window_s[0]) & (spk < event_time_s + window_s[1])
            return spk[mask] - event_time_s
    return np.array([])


def compute_peri_lick_psth(sess, cluster_id, lick_times, window, bin_size,
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
                trial_rates[j] = gaussian_filter1d(hist / bin_size, sigma=sigma_bins)
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


# ── Main ─────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED)

    print("Loading session data...")
    sess = load_session_data(SESSION_NAME)

    print("Loading video sync...")
    sync = load_video_sync(SESSION_NAME)
    if sync is None or sync.get("quality") != "good":
        print("ERROR: No good sync for this session")
        return
    eye_sync = sync["eye_cam"]
    slope = eye_sync["slope"]
    offset = eye_sync["offset"]

    print("Finding camera files...")
    cam_files = find_camera_files(SESSION_NAME)
    eye_video = cam_files["eye_cam"]["video"]
    eye_meta = cam_files["eye_cam"]["metadata"]

    print("Loading eye cam metadata...")
    ts_ms, _, _ = load_camera_metadata(eye_meta)
    print(f"  {len(ts_ms)} frames, {ts_ms[-1]/1000:.1f} s duration")

    # ── Lick times (corrected and uncorrected) ───────────────────
    print("Computing lick times...")
    lick_corr = get_lick_times(sess, shift_ms=LICK_HARDWARE_DELAY_MS)  # corrected
    lick_raw = get_lick_times(sess, shift_ms=0.0)      # uncorrected

    hit_corr = [e for e in lick_corr if e[1] == "hit"]
    fa_corr = [e for e in lick_corr if e[1] == "fa"]
    hit_raw = [e for e in lick_raw if e[1] == "hit"]
    fa_raw = [e for e in lick_raw if e[1] == "fa"]
    print(f"  {len(hit_corr)} Hit licks, {len(fa_corr)} FA licks")

    # Build lookup: trial_idx -> uncorrected lick time
    raw_by_trial = {e[0]: e[2] for e in lick_raw}

    # ── Neuron selection (formal lick.py definition) ─────────────
    from visdetect.analysis.utils import get_good_cluster_ids
    cluster_ids = get_good_cluster_ids(sess)
    print(f"  {len(cluster_ids)} good clusters")

    print("Identifying lick-responsive neuron (lick.py formal test)...")
    best_cid, best_p, n_sig, lick_table = find_lick_responsive_neuron_formal(
        sess, cluster_ids)

    # ── Trial selection ──────────────────────────────────────────
    n_hit = min(N_TRIALS_PER_OUTCOME, len(hit_corr))
    n_fa = min(N_TRIALS_PER_OUTCOME, len(fa_corr))
    selected_hits = [hit_corr[i] for i in rng.choice(len(hit_corr), n_hit, replace=False)]
    selected_fas = [fa_corr[i] for i in rng.choice(len(fa_corr), n_fa, replace=False)]
    selected = selected_hits + selected_fas
    n_total = len(selected)

    # ── Build figure ─────────────────────────────────────────────
    print("Extracting video frames and building figure...")

    fig_height = n_total * 1.6 + 3.5
    fig = plt.figure(figsize=(18, fig_height))

    gs_main = gridspec.GridSpec(
        n_total + 1, 1,
        height_ratios=[1] * n_total + [2.0],
        hspace=0.15,
        top=0.94, bottom=0.04, left=0.02, right=0.98,
    )

    for row_idx, (trial_idx, outcome, lick_time_corr) in enumerate(selected):
        lick_cam_ms = float(nidaq_to_camera(lick_time_corr, slope, offset))

        frames, rel_times = extract_frames_around_event(
            eye_video, ts_ms, lick_cam_ms, FRAME_WINDOW_MS, FRAME_STEP_MS,
            crop=FRAME_CROP,
        )
        n_cols = len(frames)

        # Spikes relative to corrected lick time
        rel_spikes_corr = get_spikes_in_window(
            sess, best_cid, lick_time_corr, RASTER_WINDOW)
        # Spikes relative to uncorrected (raw) lick time
        lick_time_raw = raw_by_trial.get(trial_idx)
        rel_spikes_raw = (get_spikes_in_window(
            sess, best_cid, lick_time_raw, RASTER_WINDOW)
            if lick_time_raw is not None else np.array([]))

        # Layout: [frames | gap | raster_corr | raster_raw]
        gs_row = gridspec.GridSpecFromSubplotSpec(
            1, n_cols + 3,
            subplot_spec=gs_main[row_idx],
            width_ratios=[1] * n_cols + [0.1, 2.0, 2.0],
            wspace=0.03,
        )

        color = OUTCOME_COLORS[outcome]

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

        # Trial label
        first_ax.set_ylabel(
            f"{'Hit' if outcome == 'hit' else 'FA'} #{trial_idx}",
            fontsize=8, rotation=0, labelpad=50, va="center",
            color=color, fontweight="bold",
        )

        # ── Raster strip: corrected (-200ms) ─────────────────────
        ax_r1 = fig.add_subplot(gs_row[0, n_cols + 1])
        if len(rel_spikes_corr) > 0:
            ax_r1.eventplot(
                [rel_spikes_corr], lineoffsets=0.5, linelengths=0.8,
                colors=[color], linewidths=0.8,
            )
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

        # ── Raster strip: uncorrected (raw software) ─────────────
        ax_r2 = fig.add_subplot(gs_row[0, n_cols + 2])
        if len(rel_spikes_raw) > 0:
            ax_r2.eventplot(
                [rel_spikes_raw], lineoffsets=0.5, linelengths=0.8,
                colors=[color], linewidths=0.8, linestyles="dashed",
            )
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

    # ── Bottom: Two PSTH panels (corrected vs uncorrected) ───────
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

        t_h, r_h, ci_lo_h, ci_hi_h = compute_peri_lick_psth(
            sess, best_cid, h_times, PETH_WINDOW, PETH_BIN)
        t_f, r_f, ci_lo_f, ci_hi_f = compute_peri_lick_psth(
            sess, best_cid, f_times, PETH_WINDOW, PETH_BIN)

        ax.fill_between(t_h * 1000, ci_lo_h, ci_hi_h, color=HIT_COLOR, alpha=0.15)
        ax.fill_between(t_f * 1000, ci_lo_f, ci_hi_f, color=FA_COLOR, alpha=0.15)
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

    # ── Title ────────────────────────────────────────────────────
    fig.suptitle(
        f"Video-Neural Sync Validation - Session {SESSION_NAME}\n"
        f"Cluster {best_cid} (lick.py: Wilcoxon p={best_p:.1e}, "
        f"{n_sig} significant units) | "
        f"Sync: RMSE={eye_sync['rmse_ms']:.1f} ms, quality={eye_sync['quality']}",
        fontsize=11, y=0.99,
    )

    crop_tag = "_mouth_zoom" if FRAME_CROP else ""
    out_path = os.path.join(FIG_DIR, f"{SESSION_NAME}_sync_validation{crop_tag}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")

    del sess
    gc.collect()


if __name__ == "__main__":
    main()
