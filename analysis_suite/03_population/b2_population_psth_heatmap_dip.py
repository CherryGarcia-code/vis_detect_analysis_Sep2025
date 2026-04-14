"""Fig 14b: Population heatmap — sorted by DIP (max suppression) latency.

Mirrors Fig 14 exactly but sorts all heatmaps by the time of the
deepest negative deflection (dip) instead of peak activation.
Reveals the structure of lick-inhibited / suppressed neurons.

Same 7x2 layout as b_population_psth_heatmap.py.
Hit-only baseline normalization throughout.

Saves: figures/03_population/population_heatmap_dip_stats.csv
"""

import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

from config import (
    STAGE_ORDER, STAGE_COLORS, OUTCOME_COLORS, CACHE_DIR,
    HMM_LABEL_RENAME,
)
from loader import load_staging_manifest, load_session, load_hmm_assignments
from utils import (
    get_good_cluster_ids, build_population_tensor, smooth_psth,
    compute_zscore_normalized,
)
from plotting import setup_style, save_figure

from visdetect.analysis.align import (
    get_event_times_by_trial, align_spikes_to_events,
)

setup_style()

# ── Parameters ───────────────────────────────────────────────────────
WINDOW = (-0.5, 1.0)
LICK_WINDOW = (-0.5, 0.5)
BIN_SIZE = 0.01
BASELINE_WIN = (-0.5, -0.05)
MIN_UNITS = 5
SIGMA_SMOOTH = 15.0
FA_LICK_SHIFT = 0.2
LARGE_CHANGE_SIZES = {2.0, 4.0}


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


def _get_state_matched_avg_rt(trials, hmm_df, sname):
    """Compute per-HMM-state average Hit RT for large change sizes."""
    sess_hmm = hmm_df[hmm_df["session_name"] == int(sname)]
    if sess_hmm.empty:
        return {}, 0.3

    trial_state = {}
    for _, row in sess_hmm.iterrows():
        trial_state[int(row["trial_idx"])] = row["hmm_state_label"]

    state_rts = {}
    all_rts = []
    for i, t in enumerate(trials):
        if (getattr(t, "trialoutcome", None) == "Hit"
                and (getattr(t, "change_size", None) or 1.0) in LARGE_CHANGE_SIZES
                and "RT" in getattr(t, "reactiontimes", {})):
            rt = t.reactiontimes["RT"]
            if np.isfinite(rt):
                state = trial_state.get(i, "Unknown")
                state_rts.setdefault(state, []).append(rt)
                all_rts.append(rt)

    overall = np.mean(all_rts) if all_rts else 0.3
    state_avg = {s: np.mean(rts) for s, rts in state_rts.items() if rts}
    return state_avg, overall


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


def _sort_by_dip(mat, bc, post_onset=0.0):
    """Sort matrix rows by DIP (minimum z-score) latency in post-onset window.

    Returns (sorted_mat, sort_order, dip_idx).
    """
    post_mask = bc >= post_onset
    post_bc_idx = np.where(post_mask)[0]
    if len(post_bc_idx) > 0:
        dip_idx = post_bc_idx[0] + np.argmin(mat[:, post_mask], axis=1)
    else:
        dip_idx = np.argmin(mat, axis=1)
    sort_order = np.argsort(dip_idx)
    return mat[sort_order], sort_order, dip_idx


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("[03b-dip] Population PSTH heatmaps sorted by DIP latency...",
          flush=True)
    manifest = load_staging_manifest(qc_only=True)
    hmm_df = load_hmm_assignments()

    expert_sessions = manifest[
        manifest["stage"] == "Expert"
    ]["session_name"].astype(int).tolist()

    # Per-unit storage
    all_hit_psths = []
    all_miss_psths = []
    all_sdt_fa_psths = []
    all_hit_lick_psths = []
    all_bfa_putchange_psths = []
    all_bfa_lick_psths = []

    bin_centers_ref = None
    lick_bc_ref = None
    unit_labels = []

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

        go_hit_idx = [
            i for i, t in enumerate(trials)
            if getattr(t, "trialoutcome", None) == "Hit"
            and (getattr(t, "change_size", None) or 1.0) > 1.01
        ]
        go_miss_idx = [
            i for i, t in enumerate(trials)
            if getattr(t, "trialoutcome", None) == "Miss"
            and (getattr(t, "change_size", None) or 1.0) > 1.01
        ]
        sdt_fa_idx = [
            i for i, t in enumerate(trials)
            if getattr(t, "trialoutcome", None) == "Hit"
            and (getattr(t, "change_size", None) or 1.0) <= 1.01
        ]
        behav_fa_idx = [
            i for i, t in enumerate(trials)
            if getattr(t, "trialoutcome", None) == "FA"
        ]

        hit_tensor, bc, _ = build_population_tensor(
            sess, good_ids, event_name="Change_ON",
            window=WINDOW, bin_size=BIN_SIZE,
            trial_indices=go_hit_idx,
        )
        miss_tensor, _, _ = build_population_tensor(
            sess, good_ids, event_name="Change_ON",
            window=WINDOW, bin_size=BIN_SIZE,
            trial_indices=go_miss_idx,
        )

        if hit_tensor.shape[0] < 5 or miss_tensor.shape[0] < 5:
            print("too few trials")
            del sess; gc.collect()
            continue

        has_sdt_fa = False
        sdt_fa_tensor = None
        if len(sdt_fa_idx) >= 3:
            sdt_fa_tensor, _, _ = build_population_tensor(
                sess, good_ids, event_name="Change_ON",
                window=WINDOW, bin_size=BIN_SIZE,
                trial_indices=sdt_fa_idx,
            )
            if sdt_fa_tensor.shape[0] >= 3:
                has_sdt_fa = True

        mu, sigma = _compute_hit_baseline_stats(hit_tensor, bc, BASELINE_WIN)

        hit_z = _apply_zscore(hit_tensor, mu, sigma)
        miss_z = _apply_zscore(miss_tensor, mu, sigma)
        sdt_fa_z = _apply_zscore(sdt_fa_tensor, mu, sigma) if has_sdt_fa else None

        hit_mean = np.nanmean(hit_z, axis=0)
        miss_mean = np.nanmean(miss_z, axis=0)
        sdt_fa_mean = np.nanmean(sdt_fa_z, axis=0) if has_sdt_fa else None

        change_times = get_event_times_by_trial(sess, "Change_ON")
        hit_lick_times = []
        for i in go_hit_idx:
            t = trials[i]
            if ("RT" in getattr(t, "reactiontimes", {})
                    and i < len(change_times) and np.isfinite(change_times[i])):
                hit_lick_times.append(change_times[i] + t.reactiontimes["RT"])

        hit_lick_tensor, lick_bc = _build_lick_aligned_tensor(
            sess, good_ids, hit_lick_times, LICK_WINDOW, BIN_SIZE)
        hit_lick_z = None
        if hit_lick_tensor is not None:
            hit_lick_z = _apply_zscore(hit_lick_tensor, mu, sigma)

        baseline_times = get_event_times_by_trial(sess, "Baseline_ON")
        state_avg_rt, overall_rt = _get_state_matched_avg_rt(
            trials, hmm_df, sname)

        sess_hmm = hmm_df[hmm_df["session_name"] == int(sname)]
        trial_state = {}
        for _, row in sess_hmm.iterrows():
            trial_state[int(row["trial_idx"])] = row["hmm_state_label"]

        putative_change_times = []
        fa_lick_times_raw = []
        for i in behav_fa_idx:
            t = trials[i]
            if ("FA" not in getattr(t, "reactiontimes", {})
                    or i >= len(baseline_times)):
                continue
            fa_rt = t.reactiontimes["FA"]
            if not np.isfinite(fa_rt):
                continue
            corrected_lick = baseline_times[i] + fa_rt - FA_LICK_SHIFT
            fa_state = trial_state.get(i, "Unknown")
            matched_rt = state_avg_rt.get(fa_state, overall_rt)
            putative_change_times.append(corrected_lick - matched_rt)
            fa_lick_times_raw.append(corrected_lick)

        bfa_putchange_tensor, _ = _build_lick_aligned_tensor(
            sess, good_ids, putative_change_times, WINDOW, BIN_SIZE)
        bfa_lick_tensor, _ = _build_lick_aligned_tensor(
            sess, good_ids, fa_lick_times_raw, LICK_WINDOW, BIN_SIZE)

        bfa_putchange_z = None
        if bfa_putchange_tensor is not None:
            bfa_putchange_z = _apply_zscore(bfa_putchange_tensor, mu, sigma)
        bfa_lick_z = None
        if bfa_lick_tensor is not None:
            bfa_lick_z = _apply_zscore(bfa_lick_tensor, mu, sigma)

        n_units_sess = hit_mean.shape[1]
        hit_lick_mean = (np.nanmean(hit_lick_z, axis=0)
                         if hit_lick_z is not None else None)
        bfa_put_mean = (np.nanmean(bfa_putchange_z, axis=0)
                        if bfa_putchange_z is not None else None)
        bfa_lick_mean = (np.nanmean(bfa_lick_z, axis=0)
                         if bfa_lick_z is not None else None)

        for u in range(n_units_sess):
            all_hit_psths.append(
                smooth_psth(hit_mean[:, u], BIN_SIZE, sigma_ms=SIGMA_SMOOTH))
            all_miss_psths.append(
                smooth_psth(miss_mean[:, u], BIN_SIZE, sigma_ms=SIGMA_SMOOTH))

            if has_sdt_fa:
                all_sdt_fa_psths.append(
                    smooth_psth(sdt_fa_mean[:, u], BIN_SIZE,
                                sigma_ms=SIGMA_SMOOTH))
            else:
                all_sdt_fa_psths.append(None)

            if hit_lick_mean is not None:
                all_hit_lick_psths.append(
                    smooth_psth(hit_lick_mean[:, u], BIN_SIZE,
                                sigma_ms=SIGMA_SMOOTH))
            else:
                all_hit_lick_psths.append(None)

            if bfa_put_mean is not None:
                all_bfa_putchange_psths.append(
                    smooth_psth(bfa_put_mean[:, u], BIN_SIZE,
                                sigma_ms=SIGMA_SMOOTH))
            else:
                all_bfa_putchange_psths.append(None)

            if bfa_lick_mean is not None:
                all_bfa_lick_psths.append(
                    smooth_psth(bfa_lick_mean[:, u], BIN_SIZE,
                                sigma_ms=SIGMA_SMOOTH))
            else:
                all_bfa_lick_psths.append(None)

            unit_labels.append((sname, good_ids[u]))

        bin_centers_ref = bc
        if lick_bc is not None:
            lick_bc_ref = lick_bc

        n_bfa = len(putative_change_times)
        print(f"{n_units_sess} units ({hit_tensor.shape[0]} hit, "
              f"{miss_tensor.shape[0]} miss, {n_bfa} bFA)")

        del sess; gc.collect()

    # ── Stack results ────────────────────────────────────────────────
    n_units = len(all_hit_psths)
    print(f"\n  Total: {n_units} units from {len(expert_sessions)} "
          f"Expert sessions", flush=True)

    if n_units == 0 or bin_centers_ref is None:
        print("  No data. Exiting.")
        return

    hit_mat = np.array(all_hit_psths)
    miss_mat = np.array(all_miss_psths)
    bc = bin_centers_ref

    def _stack_optional(lst):
        valid = np.array([p is not None for p in lst])
        n = int(valid.sum())
        mat = np.array([p for p in lst if p is not None]) if n > 0 else None
        return mat, valid, n

    sdt_fa_mat, sdt_fa_valid, n_sdt_fa = _stack_optional(all_sdt_fa_psths)
    hit_lick_mat, hit_lick_valid, n_hit_lick = _stack_optional(
        all_hit_lick_psths)
    bfa_put_mat, bfa_put_valid, n_bfa_put = _stack_optional(
        all_bfa_putchange_psths)
    bfa_lick_mat, bfa_lick_valid, n_bfa_lick = _stack_optional(
        all_bfa_lick_psths)

    if lick_bc_ref is None:
        n_lick_bins = int(round((LICK_WINDOW[1] - LICK_WINDOW[0]) / BIN_SIZE))
        lick_bc_ref = np.linspace(LICK_WINDOW[0] + BIN_SIZE / 2,
                                  LICK_WINDOW[1] - BIN_SIZE / 2, n_lick_bins)
    lbc = lick_bc_ref

    # ── Hit-DIP sort order (all units) ───────────────────────────────
    hit_sorted, sort_hit, dip_idx_hit = _sort_by_dip(hit_mat, bc, 0.0)
    miss_sorted = miss_mat[sort_hit]
    diff_sorted = hit_sorted - miss_sorted

    vmax_single = np.percentile(np.abs(hit_sorted), 97)
    vmax_diff = np.percentile(np.abs(diff_sorted), 97)

    # ── Helper: extract subset for valid units, dip-sorted ───────────
    def _extract_dip_sorted_subset(primary_mat, valid_mask, other_mat,
                                   bc_for_sort, post_onset=0.0):
        if primary_mat is None or primary_mat.shape[0] < 3:
            return None, None, None, 0
        prim_sorted, prim_sort, _ = _sort_by_dip(
            primary_mat, bc_for_sort, post_onset)
        global_indices = np.where(valid_mask)[0]
        sorted_global = global_indices[prim_sort]
        other_sorted = other_mat[sorted_global]
        return prim_sorted, other_sorted, prim_sort, prim_sorted.shape[0]

    # FA change-aligned dip sort
    bfa_put_fa_sorted, hit_fa_sorted, _, n_fa_change = \
        _extract_dip_sorted_subset(
            bfa_put_mat, bfa_put_valid, hit_mat, bc, 0.0)

    # bFA @ putative change under Hit-dip sort
    def _sort_optional_by_hit_dip(mat, valid_mask):
        if mat is None:
            return None, 0
        rows = []
        for idx in sort_hit:
            if valid_mask[idx]:
                pos = int(valid_mask[:idx].sum())
                rows.append(mat[pos])
        if rows:
            return np.array(rows), len(rows)
        return None, 0

    bfa_put_hit_sorted, n_bfa_put_hs = _sort_optional_by_hit_dip(
        bfa_put_mat, bfa_put_valid)
    sdt_fa_hit_sorted, n_sdt_hs = _sort_optional_by_hit_dip(
        sdt_fa_mat, sdt_fa_valid)

    # ── FA lick-aligned dip sort ─────────────────────────────────────
    both_lick_valid = hit_lick_valid & bfa_lick_valid
    n_both_lick = int(both_lick_valid.sum())

    bfa_lick_fa_sorted, _, _, n_fa_lick = \
        _extract_dip_sorted_subset(
            bfa_lick_mat, bfa_lick_valid, hit_mat, lbc, -0.1)

    if (bfa_lick_mat is not None and hit_lick_mat is not None
            and n_both_lick > 0):
        both_global = np.where(both_lick_valid)[0]
        hl_positions = np.array([int(hit_lick_valid[:g].sum())
                                 for g in both_global])
        fl_positions = np.array([int(bfa_lick_valid[:g].sum())
                                 for g in both_global])
        bfa_lick_both = bfa_lick_mat[fl_positions]
        hit_lick_both = hit_lick_mat[hl_positions]

        # Sort by FA-lick DIP
        bfa_lick_both_sorted, fa_lick_sort, _ = _sort_by_dip(
            bfa_lick_both, lbc, -0.1)
        hit_lick_fa_sorted_matched = hit_lick_both[fa_lick_sort]

        # Sort by Hit-lick DIP
        hit_lick_both_sorted, hit_lick_sort, _ = _sort_by_dip(
            hit_lick_both, lbc, -0.1)

        # Diff (matched units, FA-lick dip sort)
        lick_diff = hit_lick_fa_sorted_matched - bfa_lick_both_sorted
    else:
        bfa_lick_both_sorted = None
        hit_lick_fa_sorted_matched = None
        hit_lick_both_sorted = None
        lick_diff = None

    # Hit @ lick, sorted by Hit-lick DIP
    hit_lick_sorted_own = None
    n_hlick_own = 0
    if hit_lick_mat is not None and hit_lick_mat.shape[0] >= 3:
        hit_lick_sorted_own, _, _ = _sort_by_dip(hit_lick_mat, lbc, -0.1)
        n_hlick_own = hit_lick_sorted_own.shape[0]

    # bFA @ lick, sorted by FA-lick DIP
    bfa_lick_sorted_own = None
    n_bfa_lick_own = 0
    if bfa_lick_mat is not None and bfa_lick_mat.shape[0] >= 3:
        bfa_lick_sorted_own, _, _ = _sort_by_dip(bfa_lick_mat, lbc, -0.1)
        n_bfa_lick_own = bfa_lick_sorted_own.shape[0]

    print("  Building figure...", flush=True)

    # ── Create figure (7 rows x 2 cols) ──────────────────────────────
    fig = plt.figure(figsize=(24, 49))
    gs = gridspec.GridSpec(7, 2, hspace=0.35, wspace=0.3)

    fig.suptitle("Population heatmaps sorted by DIP (max suppression) latency",
                 fontsize=14, fontweight="bold", y=0.995)

    # Row 1: Hit + Miss (Hit-dip sort)
    _plot_heatmap(fig.add_subplot(gs[0, 0]), hit_sorted, bc, n_units,
                  vmax_single, "A. Hit @ Change_ON (Hit-dip sort)",
                  "Time from Change_ON (s)")
    _plot_heatmap(fig.add_subplot(gs[0, 1]), miss_sorted, bc, n_units,
                  vmax_single, "B. Miss (Hit-dip sort)",
                  "Time from Change_ON (s)")

    # Row 2: Hit-Miss diff + SDT FA (Hit-dip sort)
    _plot_heatmap(fig.add_subplot(gs[1, 0]), diff_sorted, bc, n_units,
                  vmax_diff, "C. Hit \u2212 Miss diff (Hit-dip sort)",
                  "Time from Change_ON (s)", cmap="PiYG")
    ax_d = fig.add_subplot(gs[1, 1])
    if sdt_fa_hit_sorted is not None:
        vmax_sdt = np.percentile(np.abs(sdt_fa_hit_sorted), 97)
        _plot_heatmap(ax_d, sdt_fa_hit_sorted, bc, n_sdt_hs,
                      vmax_sdt,
                      f"D. SDT FA ({n_sdt_hs}u, Hit-dip sort)",
                      "Time from Change_ON (s)")
    else:
        ax_d.text(0.5, 0.5, "No SDT FA data",
                  transform=ax_d.transAxes, ha="center")
        ax_d.set_title("D. SDT FA (no data)")

    # Row 3: bFA @ putative change — Hit-dip vs FA-dip sort
    ax_e = fig.add_subplot(gs[2, 0])
    if bfa_put_hit_sorted is not None:
        vmax_bfa = np.percentile(np.abs(bfa_put_hit_sorted), 97)
        _plot_heatmap(ax_e, bfa_put_hit_sorted, bc, n_bfa_put_hs,
                      vmax_bfa,
                      f"E. bFA @ put.Change (Hit-dip sort, {n_bfa_put_hs}u)",
                      "Time from putative Change_ON (s)")
    else:
        ax_e.text(0.5, 0.5, "No bFA data",
                  transform=ax_e.transAxes, ha="center")
        ax_e.set_title("E. bFA @ putative change (no data)")

    ax_f = fig.add_subplot(gs[2, 1])
    if bfa_put_fa_sorted is not None:
        vmax_bfa_fa = np.percentile(np.abs(bfa_put_fa_sorted), 97)
        _plot_heatmap(ax_f, bfa_put_fa_sorted, bc, n_fa_change,
                      vmax_bfa_fa,
                      f"F. bFA @ put.Change (FA-dip sort, {n_fa_change}u)",
                      "Time from putative Change_ON (s)")
    else:
        ax_f.text(0.5, 0.5, "No bFA data",
                  transform=ax_f.transAxes, ha="center")
        ax_f.set_title("F. bFA @ putative change (no data)")

    # Row 4: Hit under FA-dip sort + Hit-bFA diff (FA-dip sort)
    ax_g = fig.add_subplot(gs[3, 0])
    if hit_fa_sorted is not None:
        vmax_hfa = np.percentile(np.abs(hit_fa_sorted), 97)
        _plot_heatmap(ax_g, hit_fa_sorted, bc, n_fa_change,
                      vmax_hfa,
                      f"G. Hit @ Change_ON (FA-dip sort, {n_fa_change}u)",
                      "Time from Change_ON (s)")
    else:
        ax_g.text(0.5, 0.5, "No bFA data",
                  transform=ax_g.transAxes, ha="center")
        ax_g.set_title("G. Hit @ Change (FA-dip sort, no data)")

    ax_h = fig.add_subplot(gs[3, 1])
    if hit_fa_sorted is not None and bfa_put_fa_sorted is not None:
        change_diff = hit_fa_sorted - bfa_put_fa_sorted
        vmax_cd = np.percentile(np.abs(change_diff), 97)
        _plot_heatmap(ax_h, change_diff, bc, n_fa_change,
                      vmax_cd,
                      f"H. Hit \u2212 bFA diff (FA-dip sort, {n_fa_change}u)",
                      "Time (s)", cmap="PiYG")
    else:
        ax_h.text(0.5, 0.5, "Insufficient data",
                  transform=ax_h.transAxes, ha="center")
        ax_h.set_title("H. Hit \u2212 bFA diff (no data)")

    # Row 5: Hit @ lick (own dip sort) + bFA @ lick (own dip sort)
    ax_i = fig.add_subplot(gs[4, 0])
    if hit_lick_sorted_own is not None:
        vmax_hl = np.percentile(np.abs(hit_lick_sorted_own), 97)
        _plot_heatmap(ax_i, hit_lick_sorted_own, lbc, n_hlick_own,
                      vmax_hl,
                      f"I. Hit @ lick (Hit-lick dip sort, {n_hlick_own}u)",
                      "Time from Hit lick (s)")
    else:
        ax_i.text(0.5, 0.5, "No Hit-lick data",
                  transform=ax_i.transAxes, ha="center")
        ax_i.set_title("I. Hit @ lick (no data)")

    ax_j = fig.add_subplot(gs[4, 1])
    if bfa_lick_sorted_own is not None:
        vmax_fl = np.percentile(np.abs(bfa_lick_sorted_own), 97)
        _plot_heatmap(ax_j, bfa_lick_sorted_own, lbc, n_bfa_lick_own,
                      vmax_fl,
                      f"J. bFA @ lick (FA-lick dip sort, {n_bfa_lick_own}u)",
                      "Time from FA lick (s)")
    else:
        ax_j.text(0.5, 0.5, "No bFA-lick data",
                  transform=ax_j.transAxes, ha="center")
        ax_j.set_title("J. bFA @ lick (no data)")

    # Row 6: Hit @ lick (FA-lick dip sort) + Hit-bFA @ lick diff
    ax_k = fig.add_subplot(gs[5, 0])
    if hit_lick_fa_sorted_matched is not None:
        vmax_hfl = np.percentile(np.abs(hit_lick_fa_sorted_matched), 97)
        _plot_heatmap(ax_k, hit_lick_fa_sorted_matched, lbc, n_both_lick,
                      vmax_hfl,
                      f"K. Hit @ lick (FA-lick dip sort, {n_both_lick}u)",
                      "Time from lick (s)")
    else:
        ax_k.text(0.5, 0.5, "Insufficient matched data",
                  transform=ax_k.transAxes, ha="center")
        ax_k.set_title("K. Hit @ lick (FA dip sort, no data)")

    ax_l = fig.add_subplot(gs[5, 1])
    if lick_diff is not None:
        vmax_ld = np.percentile(np.abs(lick_diff), 97)
        _plot_heatmap(ax_l, lick_diff, lbc, n_both_lick,
                      vmax_ld,
                      f"L. Hit \u2212 bFA @ lick diff ({n_both_lick}u)",
                      "Time from lick (s)", cmap="PiYG")
    else:
        ax_l.text(0.5, 0.5, "Insufficient matched data",
                  transform=ax_l.transAxes, ha="center")
        ax_l.set_title("L. Hit \u2212 bFA @ lick (no data)")

    # Row 7: Population averages (same as peak version)
    ax_m = fig.add_subplot(gs[6, 0])
    hit_pop = np.nanmean(hit_mat, axis=0)
    hit_sem = np.nanstd(hit_mat, axis=0) / np.sqrt(n_units)
    miss_pop = np.nanmean(miss_mat, axis=0)
    miss_sem = np.nanstd(miss_mat, axis=0) / np.sqrt(n_units)

    ax_m.plot(bc, hit_pop, color=OUTCOME_COLORS["Hit"], lw=2,
              label=f"Hit (n={n_units})")
    ax_m.fill_between(bc, hit_pop - hit_sem, hit_pop + hit_sem,
                      color=OUTCOME_COLORS["Hit"], alpha=0.2)
    ax_m.plot(bc, miss_pop, color=OUTCOME_COLORS["Miss"], lw=2,
              label=f"Miss (n={n_units})")
    ax_m.fill_between(bc, miss_pop - miss_sem, miss_pop + miss_sem,
                      color=OUTCOME_COLORS["Miss"], alpha=0.2)
    if bfa_put_mat is not None:
        bfa_pop = np.nanmean(bfa_put_mat, axis=0)
        bfa_sem = np.nanstd(bfa_put_mat, axis=0) / np.sqrt(n_bfa_put)
        ax_m.plot(bc, bfa_pop, color=OUTCOME_COLORS["FA"], lw=2, ls="--",
                  label=f"bFA@putChange (n={n_bfa_put})")
        ax_m.fill_between(bc, bfa_pop - bfa_sem, bfa_pop + bfa_sem,
                          color=OUTCOME_COLORS["FA"], alpha=0.15)
    ax_m.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax_m.axhline(0, color="gray", lw=0.5, ls=":")
    ax_m.set_xlabel("Time from Change_ON / putative Change_ON (s)")
    ax_m.set_ylabel("Population z-score (Hit baseline)")
    ax_m.set_title("M. Population average: Change-aligned")
    ax_m.legend(fontsize=8)

    ax_n = fig.add_subplot(gs[6, 1])
    if hit_lick_mat is not None:
        hl_pop = np.nanmean(hit_lick_mat, axis=0)
        hl_sem = np.nanstd(hit_lick_mat, axis=0) / np.sqrt(n_hit_lick)
        ax_n.plot(lbc, hl_pop, color=OUTCOME_COLORS["Hit"], lw=2,
                  label=f"Hit@lick (n={n_hit_lick})")
        ax_n.fill_between(lbc, hl_pop - hl_sem, hl_pop + hl_sem,
                          color=OUTCOME_COLORS["Hit"], alpha=0.2)
    if bfa_lick_mat is not None:
        fl_pop = np.nanmean(bfa_lick_mat, axis=0)
        fl_sem = np.nanstd(bfa_lick_mat, axis=0) / np.sqrt(n_bfa_lick)
        ax_n.plot(lbc, fl_pop, color=OUTCOME_COLORS["FA"], lw=2,
                  label=f"bFA@lick (n={n_bfa_lick})")
        ax_n.fill_between(lbc, fl_pop - fl_sem, fl_pop + fl_sem,
                          color=OUTCOME_COLORS["FA"], alpha=0.2)
    ax_n.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax_n.axhline(0, color="gray", lw=0.5, ls=":")
    ax_n.set_xlabel("Time from lick (s)")
    ax_n.set_ylabel("Population z-score (Hit baseline)")
    ax_n.set_title("N. Population average: Lick-aligned")
    ax_n.legend(fontsize=8)

    # ── Statistics ────────────────────────────────────────────────────
    post_mask = bc >= 0
    stats = []

    dip_times_hit = bc[dip_idx_hit[sort_hit]]
    stats.append({"test": "hit_dip_latency_median",
                   "value": float(np.median(dip_times_hit)),
                   "iqr_low": float(np.percentile(dip_times_hit, 25)),
                   "iqr_high": float(np.percentile(dip_times_hit, 75))})

    # Fraction of units with dip in post-change window
    resp_mask_bc = (bc >= 0.1) & (bc < 0.5)
    dip_vals = np.nanmin(hit_sorted[:, resp_mask_bc], axis=1)
    stats.append({"test": "frac_units_suppressed_0.1-0.5s",
                   "value": float(np.mean(dip_vals < -1.0)),
                   "n_units": n_units})

    stats.append({"test": "n_units_total", "value": n_units})
    stats.append({"test": "n_bfa_putchange_units", "value": n_bfa_put})
    stats.append({"test": "n_bfa_lick_units", "value": n_bfa_lick})
    stats.append({"test": "n_hit_lick_units", "value": n_hit_lick})
    stats.append({"test": "n_both_lick_matched", "value": n_both_lick})
    stats.append({"test": "sort_criterion", "value": "dip (argmin)"})
    stats.append({"test": "normalization", "value": "Hit-only baseline"})

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig14b_population_heatmap_dip", "03_population")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "03_population", "population_heatmap_dip_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats:")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: {row.get('value', 'N/A')}")


if __name__ == "__main__":
    main()
