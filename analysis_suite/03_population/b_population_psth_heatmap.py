"""Fig 14: Population heatmap — PSTH heatmaps sorted by peak latency.

Produces sorted heatmaps showing all responsive units' activity aligned
to different events that cover key aspects of behavior.

Produces:
  - Fig 7A: Change_ON aligned (Hit trials), sorted by peak latency
  - Fig 7B: Change_ON aligned (Miss trials), same unit order as A
  - Fig 7C: Change_ON Hit minus Miss (delta signal across population)
  - Fig 7D: Population average PSTHs for Hit vs Miss (with SEM)

Saves: figures/03_population/population_heatmap_stats.csv
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
)
from loader import load_staging_manifest, load_session
from utils import (
    get_good_cluster_ids, build_population_tensor, smooth_psth,
    compute_zscore_normalized,
)
from plotting import setup_style, save_figure

setup_style()

# Parameters
WINDOW = (-0.5, 1.0)
BIN_SIZE = 0.01          # finer binning for heatmaps
BASELINE_WIN = (-0.5, -0.05)
MIN_UNITS = 5


def main():
    print("[03b] Population PSTH heatmaps...")
    manifest = load_staging_manifest(qc_only=True)

    # Collect unit PSTHs from Expert sessions
    expert_sessions = manifest[manifest["stage"] == "Expert"]["session_name"].astype(int).tolist()

    all_hit_psths = []
    all_miss_psths = []
    all_fa_psths = []   # None for units from sessions with too few FAs
    bin_centers_ref = None
    unit_labels = []  # (session, cluster_id) for each unit

    for sname in expert_sessions:
        print(f"  Session {sname}...", end=" ")
        try:
            sess = load_session(sname)
        except FileNotFoundError:
            print("not found")
            continue

        good_ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
        if len(good_ids) < MIN_UNITS:
            print(f"{len(good_ids)} units (skip)")
            del sess
            gc.collect()
            continue

        # Separate go-trial hits from catch-trial FAs
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
        fa_idx = [
            i for i, t in enumerate(trials)
            if getattr(t, "trialoutcome", None) == "Hit"
            and (getattr(t, "change_size", None) or 1.0) <= 1.01
        ]

        # Hit tensor (go trials only)
        hit_tensor, bc, hit_used = build_population_tensor(
            sess, good_ids, event_name="Change_ON",
            window=WINDOW, bin_size=BIN_SIZE,
            trial_indices=go_hit_idx,
        )

        # Miss tensor (go trials only)
        miss_tensor, _, miss_used = build_population_tensor(
            sess, good_ids, event_name="Change_ON",
            window=WINDOW, bin_size=BIN_SIZE,
            trial_indices=go_miss_idx,
        )

        if hit_tensor.shape[0] < 5 or miss_tensor.shape[0] < 5:
            print("too few trials")
            del sess
            gc.collect()
            continue

        # FA tensor (catch-trial licks)
        has_fa = False
        if len(fa_idx) >= 3:
            fa_tensor, _, fa_used = build_population_tensor(
                sess, good_ids, event_name="Change_ON",
                window=WINDOW, bin_size=BIN_SIZE,
                trial_indices=fa_idx,
            )
            if fa_tensor.shape[0] >= 3:
                has_fa = True

        # Z-score normalize using baseline
        hit_z = compute_zscore_normalized(hit_tensor, bc, BASELINE_WIN)
        miss_z = compute_zscore_normalized(miss_tensor, bc, BASELINE_WIN)

        # Mean across trials per unit
        hit_mean = np.nanmean(hit_z, axis=0)   # (n_bins, n_units)
        miss_mean = np.nanmean(miss_z, axis=0)

        fa_mean = None
        if has_fa:
            fa_z = compute_zscore_normalized(fa_tensor, bc, BASELINE_WIN)
            fa_mean = np.nanmean(fa_z, axis=0)

        n_fa_str = f", {fa_tensor.shape[0]} FA" if has_fa else ""
        for u in range(hit_mean.shape[1]):
            hit_psth = smooth_psth(hit_mean[:, u], BIN_SIZE, sigma_ms=15.0)
            miss_psth = smooth_psth(miss_mean[:, u], BIN_SIZE, sigma_ms=15.0)
            all_hit_psths.append(hit_psth)
            all_miss_psths.append(miss_psth)
            if has_fa:
                all_fa_psths.append(smooth_psth(fa_mean[:, u], BIN_SIZE, sigma_ms=15.0))
            else:
                all_fa_psths.append(None)
            unit_labels.append((sname, good_ids[u]))

        bin_centers_ref = bc
        print(f"{hit_mean.shape[1]} units ({hit_tensor.shape[0]} hit, "
              f"{miss_tensor.shape[0]} miss{n_fa_str})")

        del sess
        gc.collect()

    n_units = len(all_hit_psths)
    print(f"\n  Total: {n_units} units from {len(expert_sessions)} Expert sessions")

    if n_units == 0 or bin_centers_ref is None:
        print("  No data. Exiting.")
        return

    # Stack into matrices
    hit_mat = np.array(all_hit_psths)     # (n_units, n_bins)
    miss_mat = np.array(all_miss_psths)
    bc = bin_centers_ref

    # Build FA matrix for units with FA data
    fa_valid_mask = np.array([p is not None for p in all_fa_psths])
    n_fa_units = int(fa_valid_mask.sum())
    fa_mat = np.array([p for p in all_fa_psths if p is not None]) if n_fa_units > 0 else None

    # Sort by peak latency on Hit trials (post-change only)
    post_mask = bc >= 0
    post_bc_idx = np.where(post_mask)[0]
    if len(post_bc_idx) > 0:
        peak_idx = post_bc_idx[0] + np.argmax(hit_mat[:, post_mask], axis=1)
    else:
        peak_idx = np.argmax(hit_mat, axis=1)
    sort_order = np.argsort(peak_idx)

    hit_sorted = hit_mat[sort_order]
    miss_sorted = miss_mat[sort_order]
    diff_sorted = hit_sorted - miss_sorted

    # FA sorted: keep same relative order, only for units with FA data
    fa_sorted = None
    hit_fa_diff_sorted = None
    if fa_mat is not None and n_fa_units > 0:
        # Map from overall sort_order to FA-valid subset
        fa_display, hit_for_fa = [], []
        for idx in sort_order:
            if fa_valid_mask[idx]:
                # Find position in fa_mat
                fa_pos = int(fa_valid_mask[:idx].sum())
                fa_display.append(fa_mat[fa_pos])
                hit_for_fa.append(hit_mat[idx])
        if fa_display:
            fa_sorted = np.array(fa_display)
            hit_fa_diff_sorted = np.array(hit_for_fa) - fa_sorted

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 21))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

    # Shared colormap limits
    vmax_single = np.percentile(np.abs(hit_sorted), 97)
    vmax_diff = np.percentile(np.abs(diff_sorted), 97)

    # Panel A: Hit heatmap
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(
        hit_sorted, aspect="auto",
        extent=[bc[0], bc[-1], n_units, 0],
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-vmax_single, vcenter=0, vmax=vmax_single),
        interpolation="none",
    )
    ax_a.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_a.set_xlabel("Time from Change_ON (s)")
    ax_a.set_ylabel(f"Units (n={n_units})")
    ax_a.set_title("A. True Hit trials (z-scored, sorted by peak latency)")
    plt.colorbar(im_a, ax=ax_a, label="z-score", shrink=0.7)

    # Panel B: Miss heatmap (same unit order)
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(
        miss_sorted, aspect="auto",
        extent=[bc[0], bc[-1], n_units, 0],
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-vmax_single, vcenter=0, vmax=vmax_single),
        interpolation="none",
    )
    ax_b.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_b.set_xlabel("Time from Change_ON (s)")
    ax_b.set_ylabel(f"Units (n={n_units})")
    ax_b.set_title("B. Miss trials (same unit order)")
    plt.colorbar(im_b, ax=ax_b, label="z-score", shrink=0.7)

    # Panel C: True FA heatmap (same unit order, FA-valid subset)
    ax_c = fig.add_subplot(gs[1, 0])
    if fa_sorted is not None and len(fa_sorted) > 0:
        vmax_fa = np.percentile(np.abs(fa_sorted), 97)
        im_c = ax_c.imshow(
            fa_sorted, aspect="auto",
            extent=[bc[0], bc[-1], n_fa_units, 0],
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-vmax_fa, vcenter=0, vmax=vmax_fa),
            interpolation="none",
        )
        ax_c.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.7)
        ax_c.set_ylabel(f"Units (n={n_fa_units})")
        ax_c.set_title(f"C. True FA / catch-trial lick (same unit order, "
                       f"{n_fa_units}/{n_units} units with FA data)")
        plt.colorbar(im_c, ax=ax_c, label="z-score", shrink=0.7)
    else:
        ax_c.text(0.5, 0.5, "No FA data available", transform=ax_c.transAxes, ha="center")
        ax_c.set_title("C. True FA (no data)")
    ax_c.set_xlabel("Time from Change_ON (s)")

    # Panel D: Hit - Miss difference
    ax_d = fig.add_subplot(gs[1, 1])
    im_d = ax_d.imshow(
        diff_sorted, aspect="auto",
        extent=[bc[0], bc[-1], n_units, 0],
        cmap="PiYG",
        norm=TwoSlopeNorm(vmin=-vmax_diff, vcenter=0, vmax=vmax_diff),
        interpolation="none",
    )
    ax_d.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_d.set_xlabel("Time from Change_ON (s)")
    ax_d.set_ylabel(f"Units (n={n_units})")
    ax_d.set_title("D. Hit \u2212 Miss difference")
    plt.colorbar(im_d, ax=ax_d, label="\u0394 z-score", shrink=0.7)

    # Panel E: Hit - FA difference (FA-valid units only)
    ax_e = fig.add_subplot(gs[2, 0])
    if hit_fa_diff_sorted is not None and len(hit_fa_diff_sorted) > 0:
        vmax_hfa = np.percentile(np.abs(hit_fa_diff_sorted), 97)
        im_e = ax_e.imshow(
            hit_fa_diff_sorted, aspect="auto",
            extent=[bc[0], bc[-1], n_fa_units, 0],
            cmap="PiYG",
            norm=TwoSlopeNorm(vmin=-vmax_hfa, vcenter=0, vmax=vmax_hfa),
            interpolation="none",
        )
        ax_e.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.7)
        ax_e.set_ylabel(f"Units (n={n_fa_units})")
        ax_e.set_title(f"E. True Hit \u2212 True FA difference ({n_fa_units} units)")
        plt.colorbar(im_e, ax=ax_e, label="\u0394 z-score", shrink=0.7)
    else:
        ax_e.text(0.5, 0.5, "No FA data available", transform=ax_e.transAxes, ha="center")
        ax_e.set_title("E. Hit \u2212 FA difference (no data)")
    ax_e.set_xlabel("Time from Change_ON (s)")

    # Panel F: Population average Hit vs Miss vs FA
    ax_f = fig.add_subplot(gs[2, 1])
    hit_pop_mean = np.nanmean(hit_mat, axis=0)
    hit_pop_sem = np.nanstd(hit_mat, axis=0) / np.sqrt(n_units)
    miss_pop_mean = np.nanmean(miss_mat, axis=0)
    miss_pop_sem = np.nanstd(miss_mat, axis=0) / np.sqrt(n_units)

    ax_f.plot(bc, hit_pop_mean, color=OUTCOME_COLORS["Hit"], linewidth=2,
              label=f"True Hit (n={n_units})")
    ax_f.fill_between(bc, hit_pop_mean - hit_pop_sem, hit_pop_mean + hit_pop_sem,
                       color=OUTCOME_COLORS["Hit"], alpha=0.2)
    ax_f.plot(bc, miss_pop_mean, color=OUTCOME_COLORS["Miss"], linewidth=2,
              label=f"Miss (n={n_units})")
    ax_f.fill_between(bc, miss_pop_mean - miss_pop_sem, miss_pop_mean + miss_pop_sem,
                       color=OUTCOME_COLORS["Miss"], alpha=0.2)
    if fa_mat is not None and len(fa_mat) > 0:
        fa_pop_mean = np.nanmean(fa_mat, axis=0)
        fa_pop_sem = np.nanstd(fa_mat, axis=0) / np.sqrt(n_fa_units)
        ax_f.plot(bc, fa_pop_mean, color=OUTCOME_COLORS["FA"], linewidth=2,
                  label=f"True FA (n={n_fa_units})")
        ax_f.fill_between(bc, fa_pop_mean - fa_pop_sem, fa_pop_mean + fa_pop_sem,
                           color=OUTCOME_COLORS["FA"], alpha=0.2)
    ax_f.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_f.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_f.set_xlabel("Time from Change_ON (s)")
    ax_f.set_ylabel("Population z-score")
    ax_f.set_title("F. Population average: True Hit vs Miss vs True FA (Expert)")
    ax_f.legend(fontsize=9)

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # Peak population difference
    diff_pop = hit_pop_mean - miss_pop_mean
    post_diff = diff_pop[post_mask]
    if len(post_diff) > 0:
        peak_diff_time = bc[post_mask][np.argmax(np.abs(post_diff))]
        peak_diff_val = post_diff[np.argmax(np.abs(post_diff))]
        stats.append({
            "test": "peak_population_hit_miss_diff",
            "time": peak_diff_time,
            "value": peak_diff_val,
        })

    # Fraction of units with positive Hit-Miss difference in response window
    resp_mask_bc = (bc >= 0) & (bc < 0.25)
    resp_diff = np.nanmean(diff_sorted[:, resp_mask_bc], axis=1)
    frac_hit_pref = float(np.mean(resp_diff > 0))
    stats.append({
        "test": "frac_hit_preferring_response_window",
        "value": frac_hit_pref,
        "n_units": n_units,
    })

    # Peak latency statistics
    peak_times = bc[peak_idx[sort_order]]
    stats.append({
        "test": "peak_latency_median",
        "value": float(np.median(peak_times)),
        "iqr_low": float(np.percentile(peak_times, 25)),
        "iqr_high": float(np.percentile(peak_times, 75)),
    })

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig14_population_heatmap", "03_population")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "03_population", "population_heatmap_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: {row.get('value', 'N/A')}")


if __name__ == "__main__":
    main()
