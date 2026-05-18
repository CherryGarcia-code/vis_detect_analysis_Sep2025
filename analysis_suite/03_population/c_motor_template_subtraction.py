"""Fig 14c: Evidence vs readiness decomposition (dPCA-style).

Decomposes Hit trial population activity into condition-independent (readiness/
motor) and condition-dependent (evidence) components using change_size variation.

Core idea (simplified dPCA / Kobak et al. 2016):
  For each unit, R(t, CS) = trial-averaged response at change_size CS.
  readiness(t)    = mean_CS[R(t, CS)]          — shared across conditions
  evidence(t, CS) = R(t, CS) - readiness(t)    — what varies with CS

The evidence component is guaranteed to contain ONLY what scales with change_size
(sensory evidence accumulation). The readiness component captures everything
shared across CS: motor preparation, temporal expectation, lick execution.

Also retains the bFA motor template subtraction (v1) for comparison.

Layout (5 rows x 2 cols = 10 panels):
  Row 1: A. Hit @ Change_ON (reference)        B. Readiness component (condition-indep)
  Row 2: C. Evidence component (condition-dep)  D. Evidence heatmap (sorted by evidence peak)
  Row 3: E. Population avg: original vs decomp  F. Evidence by change_size (dose-response)
  Row 4: G. Motor template subtraction residual H. dPCA vs template comparison
  Row 5: I. Variance explained by component     J. Validation: pre-change evidence

Hit-only baseline normalization. Expert sessions only.

Saves: figures/03_population/motor_template_subtraction_stats.csv
       cache/motor_template_subtraction.csv
"""

import os
import sys
import gc


import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import wilcoxon, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

from visdetect.suite.config import OUTCOME_COLORS, CACHE_DIR
from visdetect.suite.loader import load_staging_manifest, load_session, load_hmm_assignments
from visdetect.analysis.utils import (
    get_good_cluster_ids, build_population_tensor, smooth_psth,
)
from visdetect.suite.plotting import setup_style, save_figure

from visdetect.analysis.align import (
    get_event_times_by_trial, align_spikes_to_events,
)
from visdetect.analysis.constants import ALL_GO_CHANGE_SIZES

setup_style()

# ── Parameters ───────────────────────────────────────────────────────
CHANGE_WINDOW = (-0.5, 1.0)        # Standard Change_ON analysis window
LICK_WINDOW = (-2.5, 0.5)          # Wide lick-aligned window
BIN_SIZE = 0.01                     # 10ms bins (matches b_population_psth_heatmap)
BASELINE_WIN = (-0.5, -0.05)       # Pre-change baseline for z-scoring
RESP_WIN = (0.0, 0.25)             # Post-change response window for metrics
LATE_RESP_WIN = (0.2, 0.5)         # Late response window (where dose-response is clearest)
SIGMA_SMOOTH = 15.0                 # ms, Gaussian smoothing
MIN_UNITS = 5
MIN_HIT_TRIALS = 10
MIN_BFA_TRIALS = 10
MIN_TRIALS_PER_CS = 3              # Min trials per change_size for dPCA
FA_LICK_SHIFT = 0.2                # seconds, corrected RT shift
LARGE_CHANGE_SIZES = {2.0, 4.0}
GO_CHANGE_SIZES = sorted(ALL_GO_CHANGE_SIZES)  # [1.25, 1.35, 1.5, 2.0, 4.0]

CACHE_FILE = os.path.join(CACHE_DIR, "motor_template_subtraction.csv")


# ── Helper functions ─────────────────────────────────────────────────

def _compute_hit_baseline_stats(hit_tensor, bc, baseline_win):
    """Compute per-unit baseline mean and std from Hit trials only."""
    bl_mask = (bc >= baseline_win[0]) & (bc < baseline_win[1])
    bl_data = hit_tensor[:, bl_mask, :]
    n_units = hit_tensor.shape[2]
    mu = np.zeros(n_units)
    sigma = np.ones(n_units)
    for u in range(n_units):
        vals = bl_data[:, :, u].ravel()
        vals = vals[np.isfinite(vals)]
        if len(vals) > 1:
            mu[u] = np.mean(vals)
            s = np.std(vals)
            sigma[u] = s if s > 1e-6 else 1.0
    return mu, sigma


def _apply_zscore(tensor, mu, sigma):
    """Apply pre-computed per-unit z-score to a tensor."""
    return (tensor - mu) / sigma


def _build_lick_aligned_tensor(sess, good_ids, event_times, window, bin_size):
    """Build a population tensor aligned to arbitrary event times."""
    event_times = [t for t in event_times if np.isfinite(t)]
    if len(event_times) < 3:
        return None, None

    event_times_arr = np.array(event_times)
    n_events = len(event_times_arr)
    n_bins = int(round((window[1] - window[0]) / bin_size))
    bin_centers = np.linspace(window[0] + bin_size / 2,
                              window[1] - bin_size / 2, n_bins)

    n_units = len(good_ids)
    tensor = np.full((n_events, n_bins, n_units), np.nan)

    clusters_by_id = {c.cluster_id: c for c in sess.clusters}
    for u_idx, cid in enumerate(good_ids):
        cluster = clusters_by_id.get(cid)
        if cluster is None:
            continue
        st = np.asarray(cluster.spike_times, dtype=float).flatten()
        mat, _ = align_spikes_to_events(st, event_times_arr, window, bin_size)
        if mat.shape[0] == n_events and mat.shape[1] == n_bins:
            tensor[:, :, u_idx] = mat

    return tensor, bin_centers


def _sort_by_peak(mat, bc, post_onset=0.0):
    """Sort matrix rows by peak latency in post-onset window."""
    post_mask = bc >= post_onset
    post_bc_idx = np.where(post_mask)[0]
    if len(post_bc_idx) > 0:
        peak_idx = post_bc_idx[0] + np.argmax(mat[:, post_mask], axis=1)
    else:
        peak_idx = np.argmax(mat, axis=1)
    sort_order = np.argsort(peak_idx)
    return mat[sort_order], sort_order, peak_idx


def _plot_heatmap(ax, mat, bc, n_units_label, vmax, title, xlabel,
                  event_line=0, cmap="RdBu_r"):
    """Plot a single heatmap panel."""
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(
        mat, aspect="auto",
        extent=[bc[0], bc[-1], mat.shape[0], 0],
        cmap=cmap, norm=norm, interpolation="none",
    )
    if event_line is not None:
        ax.axvline(event_line, color="k", linewidth=0.8, linestyle="--",
                   alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Units (n={n_units_label})")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, label="z-score", shrink=0.7)
    return im


def _realign_lick_to_change(residual_lick, lick_bc, rts, change_bc, n_units):
    """Re-align per-trial residuals from lick to Change_ON coordinates."""
    n_trials = residual_lick.shape[0]
    n_change_bins = len(change_bc)
    residual_change = np.full((n_trials, n_change_bins, n_units), np.nan)

    for k in range(n_trials):
        rt = rts[k]
        if not np.isfinite(rt):
            continue
        trial_change_time = lick_bc + rt
        for u in range(n_units):
            trace = residual_lick[k, :, u]
            finite_mask = np.isfinite(trace)
            if finite_mask.sum() < 5:
                continue
            f = interp1d(trial_change_time[finite_mask],
                         trace[finite_mask],
                         kind="linear", bounds_error=False,
                         fill_value=np.nan)
            residual_change[k, :, u] = f(change_bc)

    return residual_change


def _dpca_decompose(hit_z_tensor, _bc, trial_change_sizes, _n_units):
    """Demixed PCA-style decomposition into condition-independent and
    condition-dependent components.

    Parameters
    ----------
    hit_z_tensor : (n_trials, n_bins, n_units)
        Z-scored Hit tensor aligned to Change_ON.
    bc : (n_bins,) float
    trial_change_sizes : (n_trials,) float
    n_units : int

    Returns
    -------
    readiness_mat : (n_units, n_bins)
        Condition-independent component (mean across CS).
    evidence_dict : {cs: (n_units, n_bins)}
        Condition-dependent component per change_size.
    cs_means_dict : {cs: (n_units, n_bins)}
        Full condition-averaged response per CS.
    cs_counts : {cs: int}
        Trial counts per CS.
    """
    cs_means_dict = {}
    cs_counts = {}

    for cs in GO_CHANGE_SIZES:
        cs_mask = np.abs(trial_change_sizes - cs) < 0.01
        n_cs = int(cs_mask.sum())
        if n_cs < MIN_TRIALS_PER_CS:
            continue
        # Trial-average per unit: (n_bins, n_units) -> transpose to (n_units, n_bins)
        cs_mean = np.nanmean(hit_z_tensor[cs_mask], axis=0).T
        cs_means_dict[cs] = cs_mean
        cs_counts[cs] = n_cs

    if len(cs_means_dict) < 2:
        return None, None, None, cs_counts

    # Condition-independent: mean across all available CS
    all_cs_stack = np.stack(list(cs_means_dict.values()), axis=0)  # (n_cs, n_units, n_bins)
    readiness_mat = np.mean(all_cs_stack, axis=0)  # (n_units, n_bins)

    # Condition-dependent: subtract condition-independent
    evidence_dict = {}
    for cs, cs_mean in cs_means_dict.items():
        evidence_dict[cs] = cs_mean - readiness_mat  # (n_units, n_bins)

    return readiness_mat, evidence_dict, cs_means_dict, cs_counts


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("[03c] Evidence vs readiness decomposition (dPCA-style)...",
          flush=True)
    manifest = load_staging_manifest(qc_only=True)
    load_hmm_assignments()  # ensures HMM cache is warm for loader internals

    expert_sessions = manifest[
        manifest["stage"] == "Expert"
    ]["session_name"].astype(int).tolist()

    n_change_bins = int(round((CHANGE_WINDOW[1] - CHANGE_WINDOW[0]) / BIN_SIZE))
    change_bc = np.linspace(CHANGE_WINDOW[0] + BIN_SIZE / 2,
                            CHANGE_WINDOW[1] - BIN_SIZE / 2, n_change_bins)

    # Per-unit storage
    all_hit_change_psths = []       # Hit @ Change_ON (full response)
    all_readiness_psths = []        # Condition-independent (readiness)
    all_evidence_by_cs = {cs: [] for cs in GO_CHANGE_SIZES}  # Evidence per CS
    all_template_residual_psths = []  # bFA template subtraction residual

    # dPCA variance tracking
    all_var_total = []
    all_var_readiness = []
    all_var_evidence = []

    unit_rows = []

    for sname in expert_sessions:
        print(f"  Session {sname}...", end=" ", flush=True)
        try:
            sess = load_session(sname)
        except FileNotFoundError:
            print("not found")
            continue

        good_ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
        if len(good_ids) < MIN_UNITS:
            print(f"{len(good_ids)} units (skip)")
            del sess; gc.collect()
            continue

        trials = sess.trials
        n_units_sess = len(good_ids)

        # ── Trial indices ────────────────────────────────────────────
        go_hit_idx = [
            i for i, t in enumerate(trials)
            if getattr(t, "trialoutcome", None) == "Hit"
            and (getattr(t, "change_size", None) or 1.0) > 1.01
        ]
        behav_fa_idx = [
            i for i, t in enumerate(trials)
            if getattr(t, "trialoutcome", None) == "FA"
        ]

        if len(go_hit_idx) < MIN_HIT_TRIALS:
            print(f"too few Hit trials ({len(go_hit_idx)})")
            del sess; gc.collect()
            continue

        # ── Change_ON Hit tensor + baseline stats ────────────────────
        hit_tensor, bc, _ = build_population_tensor(
            sess, good_ids, event_name="Change_ON",
            window=CHANGE_WINDOW, bin_size=BIN_SIZE,
            trial_indices=go_hit_idx,
        )
        if hit_tensor.shape[0] < MIN_HIT_TRIALS:
            print("too few valid hit trials in tensor")
            del sess; gc.collect()
            continue

        mu, sigma = _compute_hit_baseline_stats(hit_tensor, bc, BASELINE_WIN)
        hit_z = _apply_zscore(hit_tensor, mu, sigma)

        # Get per-trial change_sizes for the tensor trials
        trial_cs = np.array([
            getattr(trials[i], "change_size", 1.0) for i in go_hit_idx
        ])
        # Trim to actual tensor size (build_population_tensor may drop trials)
        trial_cs = trial_cs[:hit_z.shape[0]]

        # ── dPCA decomposition ───────────────────────────────────────
        readiness_mat, evidence_dict, cs_means_dict, cs_counts = \
            _dpca_decompose(hit_z, bc, trial_cs, n_units_sess)

        if readiness_mat is None:
            print("insufficient CS coverage for dPCA")
            del sess; gc.collect()
            continue

        # ── bFA template subtraction (v1 retained for comparison) ────
        change_times = get_event_times_by_trial(sess, "Change_ON")
        hit_lick_times = []
        hit_lick_rts = []
        hit_lick_cs = []
        for i in go_hit_idx:
            t = trials[i]
            if ("RT" in getattr(t, "reactiontimes", {})
                    and i < len(change_times) and np.isfinite(change_times[i])):
                rt = t.reactiontimes["RT"]
                if np.isfinite(rt) and 0.05 < rt < 2.0:
                    hit_lick_times.append(change_times[i] + rt)
                    hit_lick_rts.append(rt)
                    hit_lick_cs.append(
                        getattr(t, "change_size", None) or 1.0)

        baseline_times = get_event_times_by_trial(sess, "Baseline_ON")
        fa_lick_times = []
        for i in behav_fa_idx:
            t = trials[i]
            if ("FA" not in getattr(t, "reactiontimes", {})
                    or i >= len(baseline_times)):
                continue
            fa_rt = t.reactiontimes["FA"]
            if not np.isfinite(fa_rt):
                continue
            fa_lick_times.append(baseline_times[i] + fa_rt - FA_LICK_SHIFT)

        has_template = (len(hit_lick_times) >= MIN_HIT_TRIALS
                        and len(fa_lick_times) >= MIN_BFA_TRIALS)
        template_residual_mean = None

        if has_template:
            hit_lick_tensor, lick_bc = _build_lick_aligned_tensor(
                sess, good_ids, hit_lick_times, LICK_WINDOW, BIN_SIZE)
            bfa_lick_tensor, _ = _build_lick_aligned_tensor(
                sess, good_ids, fa_lick_times, LICK_WINDOW, BIN_SIZE)

            if hit_lick_tensor is not None and bfa_lick_tensor is not None:
                hit_lick_z = _apply_zscore(hit_lick_tensor, mu, sigma)
                bfa_lick_z = _apply_zscore(bfa_lick_tensor, mu, sigma)
                motor_template = np.nanmean(bfa_lick_z, axis=0)
                residual_lick = hit_lick_z - motor_template[np.newaxis, :, :]

                rts_arr = np.array(hit_lick_rts[:hit_lick_z.shape[0]])
                residual_change = _realign_lick_to_change(
                    residual_lick, lick_bc, rts_arr, change_bc, n_units_sess)
                template_residual_mean = np.nanmean(residual_change, axis=0)

                del hit_lick_tensor, bfa_lick_tensor, hit_lick_z, bfa_lick_z
                del residual_lick, residual_change
            else:
                has_template = False

        # ── Variance decomposition per unit ──────────────────────────
        # Total variance = var across all trials and time bins
        # Readiness variance = var of condition-independent component across time
        # Evidence variance = var of condition-dependent component across CS and time
        resp_mask = (bc >= 0.0) & (bc < 0.8)

        hit_grand_mean = np.nanmean(hit_z, axis=0).T  # (n_units, n_bins)

        for u in range(n_units_sess):
            # Smooth all PSTHs
            hit_psth = smooth_psth(hit_grand_mean[u], BIN_SIZE,
                                   sigma_ms=SIGMA_SMOOTH)
            read_psth = smooth_psth(readiness_mat[u], BIN_SIZE,
                                    sigma_ms=SIGMA_SMOOTH)

            all_hit_change_psths.append(hit_psth)
            all_readiness_psths.append(read_psth)

            # Per-CS evidence
            for cs in GO_CHANGE_SIZES:
                if cs in evidence_dict:
                    ev_psth = smooth_psth(evidence_dict[cs][u], BIN_SIZE,
                                          sigma_ms=SIGMA_SMOOTH)
                    all_evidence_by_cs[cs].append(ev_psth)
                else:
                    all_evidence_by_cs[cs].append(None)

            # Template residual (if available)
            if has_template and template_residual_mean is not None:
                tr_psth = smooth_psth(template_residual_mean[:, u], BIN_SIZE,
                                      sigma_ms=SIGMA_SMOOTH)
                all_template_residual_psths.append(tr_psth)
            else:
                all_template_residual_psths.append(None)

            # Variance decomposition in response window
            resp_total = hit_psth[resp_mask]
            resp_read = read_psth[resp_mask]

            var_total = float(np.var(resp_total)) if len(resp_total) > 0 else 0
            var_readiness = float(np.var(resp_read)) if len(resp_read) > 0 else 0

            # Evidence variance: average across available CS
            ev_vars = []
            for cs in GO_CHANGE_SIZES:
                if cs in evidence_dict:
                    ev = smooth_psth(evidence_dict[cs][u], BIN_SIZE,
                                     sigma_ms=SIGMA_SMOOTH)
                    ev_vars.append(float(np.var(ev[resp_mask])))
            var_evidence = float(np.mean(ev_vars)) if ev_vars else 0

            all_var_total.append(var_total)
            all_var_readiness.append(var_readiness)
            all_var_evidence.append(var_evidence)

            # ── Per-unit metrics for cache ────────────────────────────
            post_mask_ch = bc >= 0
            post_bc_ch = np.where(post_mask_ch)[0]
            if len(post_bc_ch) > 0:
                peak_ch_idx = post_bc_ch[0] + np.argmax(hit_psth[post_mask_ch])
                peak_ch_lat = bc[peak_ch_idx]
                peak_ch_z = hit_psth[peak_ch_idx]
            else:
                peak_ch_lat = peak_ch_z = np.nan

            # Evidence peak (use CS=4.0 as strongest signal)
            if 4.0 in evidence_dict:
                ev4_psth = smooth_psth(evidence_dict[4.0][u], BIN_SIZE,
                                       sigma_ms=SIGMA_SMOOTH)
                ev4_resp = ev4_psth[resp_mask]
                if len(ev4_resp) > 0:
                    resp_bc_idx = np.where(resp_mask)[0]
                    ev4_peak_local = np.argmax(np.abs(ev4_resp))
                    ev_peak_lat = bc[resp_bc_idx[ev4_peak_local]]
                    ev_peak_z = ev4_resp[ev4_peak_local]
                else:
                    ev_peak_lat = ev_peak_z = np.nan
            else:
                ev_peak_lat = ev_peak_z = np.nan

            # Readiness peak
            read_resp = read_psth[resp_mask]
            if len(read_resp) > 0:
                resp_bc_idx = np.where(resp_mask)[0]
                read_peak_local = np.argmax(np.abs(read_resp))
                read_peak_lat = bc[resp_bc_idx[read_peak_local]]
                read_peak_z = read_resp[read_peak_local]
            else:
                read_peak_lat = read_peak_z = np.nan

            # Pre-change evidence (validation — should be ~0)
            bl_mask_ch = (bc >= BASELINE_WIN[0]) & (bc < BASELINE_WIN[1])
            if 4.0 in evidence_dict:
                ev4_bl = ev4_psth[bl_mask_ch]
                pre_ev_mean = float(np.nanmean(ev4_bl))
            else:
                pre_ev_mean = np.nan

            # Evidence fraction: var_evidence / var_total
            ev_frac = (var_evidence / var_total
                       if var_total > 1e-10 else np.nan)

            unit_rows.append({
                "session_name": sname,
                "cluster_id": good_ids[u],
                "peak_latency_change_ON": peak_ch_lat,
                "peak_zscore_change_ON": peak_ch_z,
                "evidence_peak_latency": ev_peak_lat,
                "evidence_peak_zscore": ev_peak_z,
                "readiness_peak_latency": read_peak_lat,
                "readiness_peak_zscore": read_peak_z,
                "pre_change_evidence_mean": pre_ev_mean,
                "var_total": var_total,
                "var_readiness": var_readiness,
                "var_evidence": var_evidence,
                "evidence_frac": ev_frac,
                "n_hit_trials": hit_z.shape[0],
                "n_cs_available": len(cs_means_dict),
                "n_bfa_trials": len(fa_lick_times),
            })

        cs_str = ", ".join(f"{cs}:{n}" for cs, n in cs_counts.items())
        print(f"{n_units_sess} units (CS trials: {cs_str})")

        del sess, hit_tensor, hit_z
        gc.collect()

    # ── Stack cross-session results ───────────────────────────────────
    n_units = len(all_hit_change_psths)
    print(f"\n  Total: {n_units} units from {len(expert_sessions)} "
          f"Expert sessions", flush=True)

    if n_units == 0:
        print("  No data. Exiting.")
        return

    hit_mat = np.array(all_hit_change_psths)       # (n_units, n_bins)
    readiness_mat_all = np.array(all_readiness_psths)
    cbc = change_bc

    # Stack template residuals (optional, may have Nones)
    template_valid = np.array([p is not None for p in all_template_residual_psths])
    template_mat = (np.array([p for p in all_template_residual_psths if p is not None])
                    if template_valid.any() else None)

    # ── Sort by Hit peak latency ──────────────────────────────────────
    hit_sorted, sort_hit, peak_idx_hit = _sort_by_peak(hit_mat, cbc, 0.0)
    readiness_sorted = readiness_mat_all[sort_hit]

    # Compute grand evidence matrix: largest CS (4.0) for heatmap
    ev4_list = all_evidence_by_cs.get(4.0, [])
    ev4_valid = np.array([p is not None for p in ev4_list])
    if ev4_valid.any():
        ev4_mat = np.full((n_units, len(cbc)), np.nan)
        for i, p in enumerate(ev4_list):
            if p is not None:
                ev4_mat[i] = p
        ev4_sorted_hit = ev4_mat[sort_hit]
        # Also sort by evidence peak
        ev4_valid_mat = ev4_mat[ev4_valid]
        ev4_sorted_own, sort_ev4, _ = _sort_by_peak(ev4_valid_mat, cbc, 0.0)
    else:
        ev4_sorted_hit = None
        ev4_sorted_own = None

    vmax_hit = np.nanpercentile(np.abs(hit_sorted), 97)
    vmax_read = np.nanpercentile(np.abs(readiness_sorted), 97)
    vmax_ev = (np.nanpercentile(np.abs(ev4_sorted_hit[np.isfinite(ev4_sorted_hit)]), 97)
               if ev4_sorted_hit is not None else 0.1)

    print("  Building figure...", flush=True)

    # ── Figure (5 rows x 2 cols) ──────────────────────────────────────
    fig = plt.figure(figsize=(22, 40))
    gs = gridspec.GridSpec(5, 2, hspace=0.35, wspace=0.3)

    # Panel A: Hit @ Change_ON (reference)
    _plot_heatmap(fig.add_subplot(gs[0, 0]), hit_sorted, cbc, n_units,
                  vmax_hit, "A. Hit @ Change_ON (full response)",
                  "Time from Change_ON (s)")

    # Panel B: Readiness component (condition-independent)
    _plot_heatmap(fig.add_subplot(gs[0, 1]), readiness_sorted, cbc, n_units,
                  vmax_read,
                  "B. Readiness component (condition-independent)",
                  "Time from Change_ON (s)")

    # Panel C: Evidence component (CS=4.0, strongest signal, Hit-peak sort)
    ax_c = fig.add_subplot(gs[1, 0])
    if ev4_sorted_hit is not None:
        _plot_heatmap(ax_c, ev4_sorted_hit, cbc, n_units, vmax_ev,
                      "C. Evidence component CS=4.0 (Hit-peak sort)",
                      "Time from Change_ON (s)")
    else:
        ax_c.text(0.5, 0.5, "No CS=4.0 data", transform=ax_c.transAxes,
                  ha="center")
        ax_c.set_title("C. Evidence (no data)")

    # Panel D: Evidence heatmap sorted by evidence peak
    ax_d = fig.add_subplot(gs[1, 1])
    if ev4_sorted_own is not None:
        n_ev = ev4_sorted_own.shape[0]
        _plot_heatmap(ax_d, ev4_sorted_own, cbc, n_ev, vmax_ev,
                      f"D. Evidence CS=4.0 (evidence-peak sort, {n_ev}u)",
                      "Time from Change_ON (s)")
    else:
        ax_d.text(0.5, 0.5, "No CS=4.0 data", transform=ax_d.transAxes,
                  ha="center")
        ax_d.set_title("D. Evidence sorted (no data)")

    # Panel E: Population average - original, readiness, evidence
    ax_e = fig.add_subplot(gs[2, 0])
    hit_pop = np.nanmean(hit_mat, axis=0)
    hit_sem = np.nanstd(hit_mat, axis=0) / np.sqrt(n_units)
    read_pop = np.nanmean(readiness_mat_all, axis=0)
    read_sem = np.nanstd(readiness_mat_all, axis=0) / np.sqrt(n_units)

    ax_e.plot(cbc, hit_pop, color=OUTCOME_COLORS["Hit"], lw=2,
              label="Full response")
    ax_e.fill_between(cbc, hit_pop - hit_sem, hit_pop + hit_sem,
                       color=OUTCOME_COLORS["Hit"], alpha=0.2)
    ax_e.plot(cbc, read_pop, color="darkorange", lw=2,
              label="Readiness (CS-independent)")
    ax_e.fill_between(cbc, read_pop - read_sem, read_pop + read_sem,
                       color="darkorange", alpha=0.2)

    # Evidence for CS=4.0
    ev4_psths = [p for p in all_evidence_by_cs.get(4.0, []) if p is not None]
    if ev4_psths:
        ev4_pop = np.nanmean(np.array(ev4_psths), axis=0)
        ev4_sem = np.nanstd(np.array(ev4_psths), axis=0) / np.sqrt(len(ev4_psths))
        ax_e.plot(cbc, ev4_pop, color="purple", lw=2,
                  label="Evidence (CS=4.0)")
        ax_e.fill_between(cbc, ev4_pop - ev4_sem, ev4_pop + ev4_sem,
                           color="purple", alpha=0.2)

    # Template residual for comparison
    if template_mat is not None:
        tr_pop = np.nanmean(template_mat, axis=0)
        tr_sem = np.nanstd(template_mat, axis=0) / np.sqrt(template_mat.shape[0])
        ax_e.plot(cbc, tr_pop, color="gray", lw=1.5, ls="--",
                  label="Template residual (v1)")
        ax_e.fill_between(cbc, tr_pop - tr_sem, tr_pop + tr_sem,
                           color="gray", alpha=0.1)

    ax_e.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax_e.axhline(0, color="gray", lw=0.5, ls=":")
    ax_e.set_xlabel("Time from Change_ON (s)")
    ax_e.set_ylabel("Population z-score")
    ax_e.set_title("E. Population average: full vs decomposed")
    ax_e.legend(fontsize=8, loc="upper right")

    # Panel F: Evidence by change_size (THE KEY PANEL — dose-response)
    ax_f = fig.add_subplot(gs[2, 1])
    cs_colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(GO_CHANGE_SIZES)))
    cs_mean_amplitudes = {}

    for idx, cs in enumerate(GO_CHANGE_SIZES):
        cs_psths = [p for p in all_evidence_by_cs[cs] if p is not None]
        if len(cs_psths) >= 10:
            cs_mat_local = np.array(cs_psths)
            cs_pop = np.nanmean(cs_mat_local, axis=0)
            cs_sem_vals = (np.nanstd(cs_mat_local, axis=0)
                           / np.sqrt(len(cs_psths)))
            ax_f.plot(cbc, cs_pop, color=cs_colors[idx], lw=1.5,
                      label=f"CS={cs} (n={len(cs_psths)})")
            ax_f.fill_between(cbc, cs_pop - cs_sem_vals,
                               cs_pop + cs_sem_vals,
                               color=cs_colors[idx], alpha=0.15)
            # Mean amplitude in late response window
            late_mask = (cbc >= LATE_RESP_WIN[0]) & (cbc < LATE_RESP_WIN[1])
            cs_mean_amplitudes[cs] = float(np.nanmean(cs_pop[late_mask]))

    ax_f.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax_f.axhline(0, color="gray", lw=0.5, ls=":")
    ax_f.set_xlabel("Time from Change_ON (s)")
    ax_f.set_ylabel("Evidence z-score (condition-dependent)")
    ax_f.set_title("F. Evidence by change_size (dose-response)")
    ax_f.legend(fontsize=8, loc="upper right")

    # Panel G: Template subtraction residual heatmap (comparison)
    ax_g = fig.add_subplot(gs[3, 0])
    if template_mat is not None:
        # Remap: sort_hit is in full n_units space; template_mat is compressed
        # to only template_valid units.  Build a full→compressed index map.
        full_to_template = np.full(n_units, -1, dtype=int)
        full_to_template[template_valid] = np.arange(int(template_valid.sum()))
        valid_sorted = [idx for idx in sort_hit if template_valid[idx]]
        tr_sorted = template_mat[full_to_template[valid_sorted]]
        if tr_sorted.shape[0] > 0:
            vmax_tr = np.nanpercentile(np.abs(tr_sorted), 97)
            _plot_heatmap(ax_g, tr_sorted, cbc, tr_sorted.shape[0],
                          vmax_tr,
                          f"G. Template residual v1 ({tr_sorted.shape[0]}u)",
                          "Time from Change_ON (s)")
        else:
            ax_g.text(0.5, 0.5, "No template data",
                      transform=ax_g.transAxes, ha="center")
            ax_g.set_title("G. Template residual (no data)")
    else:
        ax_g.text(0.5, 0.5, "No template data",
                  transform=ax_g.transAxes, ha="center")
        ax_g.set_title("G. Template residual (no data)")

    # Panel H: dPCA evidence vs template residual comparison (scatter or overlay)
    ax_h = fig.add_subplot(gs[3, 1])
    if template_mat is not None and ev4_psths:
        ev4_pop_h = np.nanmean(np.array(ev4_psths), axis=0)
        tr_pop_h = np.nanmean(template_mat, axis=0)
        ax_h.plot(cbc, ev4_pop_h, color="purple", lw=2,
                  label="dPCA evidence (CS=4.0)")
        ax_h.plot(cbc, tr_pop_h, color="gray", lw=2, ls="--",
                  label="Template residual")
        ax_h.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax_h.axhline(0, color="gray", lw=0.5, ls=":")
        ax_h.set_xlabel("Time from Change_ON (s)")
        ax_h.set_ylabel("Population z-score")
        ax_h.set_title("H. dPCA evidence vs template residual")
        ax_h.legend(fontsize=9)
    else:
        ax_h.text(0.5, 0.5, "Insufficient data for comparison",
                  transform=ax_h.transAxes, ha="center")
        ax_h.set_title("H. Method comparison (no data)")

    # Panel I: Variance explained by component
    ax_i = fig.add_subplot(gs[4, 0])
    var_t = np.array(all_var_total)
    var_r = np.array(all_var_readiness)
    var_e = np.array(all_var_evidence)
    # Fraction of variance: readiness / total and evidence / total
    # NOTE: These are descriptive ratios, NOT orthogonal ANOVA components.
    # Readiness + evidence fractions need NOT sum to 1.0 because the
    # covariance term (2*cov) is not decomposed here.
    valid_var = var_t > 1e-10
    frac_readiness = np.full(n_units, np.nan)
    frac_evidence = np.full(n_units, np.nan)
    frac_readiness[valid_var] = var_r[valid_var] / var_t[valid_var]
    frac_evidence[valid_var] = var_e[valid_var] / var_t[valid_var]

    fr_valid = frac_readiness[np.isfinite(frac_readiness)]
    fe_valid = frac_evidence[np.isfinite(frac_evidence)]

    ax_i.hist(fr_valid, bins=50, alpha=0.6, color="darkorange",
              label=f"Readiness (med={np.median(fr_valid):.3f})", edgecolor="white")
    ax_i.hist(fe_valid, bins=50, alpha=0.6, color="purple",
              label=f"Evidence (med={np.median(fe_valid):.3f})", edgecolor="white")
    ax_i.set_xlabel("Fraction of total variance")
    ax_i.set_ylabel("Count (units)")
    ax_i.set_title("I. Variance decomposition")
    ax_i.legend(fontsize=9)

    # Panel J: Validation — pre-change evidence distribution
    ax_j = fig.add_subplot(gs[4, 1])
    pre_ev_means = np.array([r["pre_change_evidence_mean"] for r in unit_rows])
    pre_ev_valid = pre_ev_means[np.isfinite(pre_ev_means)]
    r_rb = np.nan  # default; overwritten if n >= 10

    if len(pre_ev_valid) > 0:
        ax_j.hist(pre_ev_valid, bins=50, color="steelblue", alpha=0.7,
                  edgecolor="white")
        ax_j.axvline(0, color="red", lw=2, ls="--", label="Expected = 0")
        ax_j.axvline(np.mean(pre_ev_valid), color="k", lw=2,
                      label=f"Actual = {np.mean(pre_ev_valid):.4f}")
        if len(pre_ev_valid) >= 10:
            stat, p_val_pre = wilcoxon(pre_ev_valid)
            # Rank-biserial effect size: r = 1 - (2T / n(n+1)/2)
            n_pre = len(pre_ev_valid)
            r_rb = 1.0 - (2.0 * stat) / (n_pre * (n_pre + 1) / 2.0)
            ax_j.set_title(
                f"J. Validation: pre-change evidence (CS=4.0)\n"
                f"Wilcoxon p={p_val_pre:.2e}, r={r_rb:.3f}, n={n_pre}")
        else:
            p_val_pre = np.nan
            ax_j.set_title("J. Pre-change evidence (n<10)")
        ax_j.set_xlabel("Mean evidence z-score in [-0.5, -0.05]s")
        ax_j.set_ylabel("Count (units)")
        ax_j.legend(fontsize=8)
    else:
        ax_j.text(0.5, 0.5, "No data", transform=ax_j.transAxes, ha="center")
        ax_j.set_title("J. Validation (no data)")
        p_val_pre = np.nan

    # ── Statistics ────────────────────────────────────────────────────
    stats = []
    stats.append({"test": "n_units_total", "value": n_units})
    stats.append({"test": "n_sessions", "value": len(expert_sessions)})
    stats.append({"test": "method", "value": "dPCA condition decomposition"})
    stats.append({"test": "normalization",
                   "value": "Hit-only baseline z-score"})

    # Variance fractions
    stats.append({"test": "median_readiness_var_frac",
                   "value": float(np.median(fr_valid))})
    stats.append({"test": "median_evidence_var_frac",
                   "value": float(np.median(fe_valid))})

    # Pre-change validation
    stats.append({"test": "pre_change_evidence_mean",
                   "value": float(np.mean(pre_ev_valid))
                   if len(pre_ev_valid) > 0 else np.nan})
    stats.append({"test": "pre_change_evidence_wilcoxon_p",
                   "value": p_val_pre})
    stats.append({"test": "pre_change_evidence_wilcoxon_r",
                   "value": r_rb if len(pre_ev_valid) >= 10 else np.nan})

    # Dose-response
    if len(cs_mean_amplitudes) >= 3:
        cs_vals = sorted(cs_mean_amplitudes.items())
        rho, p_rho = spearmanr([v[0] for v in cs_vals],
                                [v[1] for v in cs_vals])
        stats.append({"test": "dose_response_spearman_rho", "value": rho})
        stats.append({"test": "dose_response_spearman_p", "value": p_rho})
    else:
        stats.append({"test": "dose_response_spearman_rho", "value": np.nan})
        stats.append({"test": "dose_response_spearman_p", "value": np.nan})

    # Evidence peak latency
    ev_peaks = np.array([r["evidence_peak_latency"] for r in unit_rows])
    ev_peaks_valid = ev_peaks[np.isfinite(ev_peaks)]
    if len(ev_peaks_valid) > 0:
        stats.append({"test": "evidence_peak_latency_median",
                       "value": float(np.median(ev_peaks_valid)),
                       "iqr_low": float(np.percentile(ev_peaks_valid, 25)),
                       "iqr_high": float(np.percentile(ev_peaks_valid, 75))})

    stats.append({"test": "median_hit_trials_per_session",
                   "value": float(np.median([r["n_hit_trials"]
                                              for r in unit_rows]))})
    stats.append({"test": "median_bfa_trials_per_session",
                   "value": float(np.median([r["n_bfa_trials"]
                                              for r in unit_rows]))})

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig14c_motor_template_subtraction", "03_population")

    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "03_population", "motor_template_subtraction_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    cache_df = pd.DataFrame(unit_rows)
    cache_df.to_csv(CACHE_FILE, index=False)

    print(f"\n  Saved figure and stats:")
    for _, row in stats_df.iterrows():
        extra = ""
        if "iqr_low" in row and pd.notna(row.get("iqr_low")):
            extra = f" [IQR: {row['iqr_low']:.3f}-{row['iqr_high']:.3f}]"
        print(f"    {row['test']}: {row.get('value', 'N/A')}{extra}")
    print(f"  Cache: {CACHE_FILE} ({len(unit_rows)} rows)")


if __name__ == "__main__":
    main()
