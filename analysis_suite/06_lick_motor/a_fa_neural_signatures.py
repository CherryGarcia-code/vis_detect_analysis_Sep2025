"""Fig24: FA Neural Signatures — pre-FA vs pre-Miss population activity.

Compares the pre-event population activity on false-alarm (FA) trials to
pre-change activity on Miss trials.  A ramp in pre-FA firing that is absent
before Misses suggests motor-preparatory activity drives false alarms.

FA trials are aligned to the FA lick time (event_name="FA"), while Miss
trials are aligned to the change onset (event_name="Change_ON").  Both use
a wide window (-2.0, 0.5 s) and are z-scored against a distant baseline
(-2.0, -1.5 s).

Produces:
  - Fig 24A: Population average PSTH (FA vs Miss, Expert sessions)
  - Fig 24B: Pre-event activity difference heatmap (FA - Miss) per unit
  - Fig 24C: Pre-FA ramp magnitude across sessions
  - Fig 24D: Pre-FA ramp split by HMM behavioral state

Saves: figures/06_lick_motor/fa_neural_signatures_stats.csv
       cache/fa_neural_signatures.csv
"""

import os
import sys
import gc


import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr, kruskal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import (
    STAGE_ORDER, STAGE_COLORS, OUTCOME_COLORS,
    HMM_STATE_ORDER, HMM_STATE_COLORS, CACHE_DIR, DEFAULT_BIN_SIZE,
)
from visdetect.suite.loader import (
    load_staging_manifest, load_session, load_hmm_assignments,
)
from visdetect.analysis.utils import (
    get_good_cluster_ids, build_population_tensor, smooth_psth,
    compute_zscore_normalized,
)
from visdetect.suite.plotting import setup_style, save_figure, add_stage_background

setup_style()

# ── Parameters ────────────────────────────────────────────────────────
FA_WINDOW = (-2.0, 0.5)
MISS_WINDOW = (-2.0, 0.5)
BIN_SIZE = DEFAULT_BIN_SIZE
BASELINE_WIN = (-2.0, -1.5)
EARLY_WIN = (-1.5, -1.0)
LATE_WIN = (-0.5, 0.0)
MIN_TRIALS = 5
MIN_UNITS = 3


def main():
    print("[06a] FA neural signatures (pre-FA vs pre-Miss)...")
    manifest = load_staging_manifest(qc_only=True)

    # Load HMM trial-level assignments (graceful if unavailable)
    try:
        hmm = load_hmm_assignments()
    except Exception:
        hmm = pd.DataFrame()

    # ── Collect per-unit data across all sessions ─────────────────────
    all_units = []
    fa_pop_psths = []       # per-session Expert population PSTHs
    miss_pop_psths = []
    bin_centers_ref = None

    for _, mrow in manifest.iterrows():
        sname = int(mrow["session_name"])
        stage = mrow["stage"]
        sidx = mrow["session_idx"]

        print(f"  Session {sname} ({stage})...", end=" ")
        try:
            sess = load_session(sname)
        except FileNotFoundError:
            print("not found")
            continue

        good_ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
        if len(good_ids) < MIN_UNITS:
            print(f"too few units ({len(good_ids)})")
            del sess
            gc.collect()
            continue

        # FA tensor: aligned to FA lick time
        fa_tensor, fa_bc, fa_used = build_population_tensor(
            sess, good_ids, event_name="FA",
            window=FA_WINDOW, bin_size=BIN_SIZE,
            outcome_filter={"FA"},
        )

        # Miss tensor: aligned to Change_ON
        miss_tensor, miss_bc, miss_used = build_population_tensor(
            sess, good_ids, event_name="Change_ON",
            window=MISS_WINDOW, bin_size=BIN_SIZE,
            outcome_filter={"Miss"},
        )

        if fa_tensor.shape[0] < MIN_TRIALS or miss_tensor.shape[0] < MIN_TRIALS:
            print(
                f"too few trials (FA={fa_tensor.shape[0]}, "
                f"Miss={miss_tensor.shape[0]})"
            )
            del sess
            gc.collect()
            continue

        bin_centers_ref = fa_bc

        # Z-score both tensors against distant baseline
        fa_z = compute_zscore_normalized(fa_tensor, fa_bc, BASELINE_WIN)
        miss_z = compute_zscore_normalized(miss_tensor, miss_bc, BASELINE_WIN)

        # Pre-event window masks
        early_mask = (fa_bc >= EARLY_WIN[0]) & (fa_bc < EARLY_WIN[1])
        late_mask = (fa_bc >= LATE_WIN[0]) & (fa_bc < LATE_WIN[1])

        # HMM trial-state lookup for this session
        trial_states = {}
        if len(hmm) > 0:
            sess_hmm = hmm[hmm["session_name"] == sname]
            if ("trial_idx" in sess_hmm.columns
                    and "hmm_state_label" in sess_hmm.columns):
                for _, hr in sess_hmm.iterrows():
                    trial_states[int(hr["trial_idx"])] = hr["hmm_state_label"]

        n_added = 0
        for u_i, cid in enumerate(good_ids):
            if u_i >= fa_z.shape[2] or u_i >= miss_z.shape[2]:
                break

            # Mean z-scored activity in pre-event windows
            fa_early = float(np.nanmean(fa_z[:, early_mask, u_i]))
            fa_late = float(np.nanmean(fa_z[:, late_mask, u_i]))
            miss_early = float(np.nanmean(miss_z[:, early_mask, u_i]))
            miss_late = float(np.nanmean(miss_z[:, late_mask, u_i]))

            fa_ramp = fa_late - fa_early
            miss_ramp = miss_late - miss_early

            # Pre-event mean (late window) difference: FA - Miss
            pre_diff = fa_late - miss_late

            # HMM-state-conditioned FA ramp
            fa_ramps_by_state = {}
            for state in HMM_STATE_ORDER:
                state_trial_mask = np.array([
                    trial_states.get(fa_used[t_i], None) == state
                    for t_i in range(len(fa_used))
                ])
                if state_trial_mask.sum() >= 3:
                    s_late = float(
                        np.nanmean(fa_z[state_trial_mask][:, late_mask, u_i])
                    )
                    s_early = float(
                        np.nanmean(fa_z[state_trial_mask][:, early_mask, u_i])
                    )
                    fa_ramps_by_state[state] = s_late - s_early

            row_data = {
                "session_name": sname,
                "cluster_id": cid,
                "stage": stage,
                "session_idx": sidx,
                "fa_early": fa_early,
                "fa_late": fa_late,
                "miss_early": miss_early,
                "miss_late": miss_late,
                "fa_ramp": fa_ramp,
                "miss_ramp": miss_ramp,
                "pre_diff": pre_diff,
                "n_fa": fa_tensor.shape[0],
                "n_miss": miss_tensor.shape[0],
            }
            for state in HMM_STATE_ORDER:
                row_data[f"fa_ramp_{state}"] = fa_ramps_by_state.get(
                    state, np.nan
                )
            all_units.append(row_data)
            n_added += 1

        # Expert population PSTHs for Panel A
        if stage == "Expert":
            # Mean over trials, then over units -> (n_bins,)
            fa_pop = np.nanmean(np.nanmean(fa_z, axis=0), axis=1)
            miss_pop = np.nanmean(np.nanmean(miss_z, axis=0), axis=1)
            fa_pop_psths.append(smooth_psth(fa_pop, BIN_SIZE, sigma_ms=25.0))
            miss_pop_psths.append(smooth_psth(miss_pop, BIN_SIZE, sigma_ms=25.0))

        print(f"{n_added} units")
        del sess
        gc.collect()

    df = pd.DataFrame(all_units)
    print(f"\n  Total: {len(df)} units with FA/Miss data")

    if len(df) == 0 or bin_centers_ref is None:
        print("  No data. Exiting.")
        return

    # Cache results
    cache_path = os.path.join(CACHE_DIR, "fa_neural_signatures.csv")
    df.to_csv(cache_path, index=False)

    bc = bin_centers_ref

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Population average PSTH -- FA vs Miss (Expert)
    ax_a = fig.add_subplot(gs[0, 0])
    if len(fa_pop_psths) > 0 and len(miss_pop_psths) > 0:
        fa_mat = np.array(fa_pop_psths)
        miss_mat = np.array(miss_pop_psths)

        fa_mean = np.nanmean(fa_mat, axis=0)
        fa_sem = np.nanstd(fa_mat, axis=0) / np.sqrt(len(fa_pop_psths))
        miss_mean = np.nanmean(miss_mat, axis=0)
        miss_sem = np.nanstd(miss_mat, axis=0) / np.sqrt(len(miss_pop_psths))

        ax_a.plot(bc, fa_mean, color=OUTCOME_COLORS["FA"], linewidth=2,
                  label=f"FA (n={len(fa_pop_psths)} sess)")
        ax_a.fill_between(bc, fa_mean - fa_sem, fa_mean + fa_sem,
                          color=OUTCOME_COLORS["FA"], alpha=0.2)
        ax_a.plot(bc, miss_mean, color=OUTCOME_COLORS["Miss"], linewidth=2,
                  label=f"Miss (n={len(miss_pop_psths)} sess)")
        ax_a.fill_between(bc, miss_mean - miss_sem, miss_mean + miss_sem,
                          color=OUTCOME_COLORS["Miss"], alpha=0.2)
    else:
        ax_a.text(0.5, 0.5, "No Expert data",
                  transform=ax_a.transAxes, ha="center")

    ax_a.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_a.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_a.set_xlabel("Time from event (s)")
    ax_a.set_ylabel("Population z-score")
    ax_a.set_title("A. Pre-event population activity: FA vs Miss (Expert)")
    ax_a.legend(fontsize=9)

    # Panel B: Pre-event difference per unit (Expert, sorted by effect size)
    ax_b = fig.add_subplot(gs[0, 1])
    expert = df[df["stage"] == "Expert"].copy()

    if len(expert) > 0:
        expert_sorted = expert.sort_values("pre_diff", ascending=False)
        y_pos = np.arange(len(expert_sorted))
        diffs = expert_sorted["pre_diff"].values

        colors = [
            OUTCOME_COLORS["FA"] if d > 0 else OUTCOME_COLORS["Miss"]
            for d in diffs
        ]
        ax_b.barh(y_pos, diffs, color=colors, height=1.0, edgecolor="none")
        ax_b.axvline(0, color="k", linewidth=0.5)
        ax_b.set_xlabel("Pre-event z-score diff (FA - Miss)")
        ax_b.set_ylabel(f"Units (n={len(expert_sorted)})")
        ax_b.set_yticks([])
        n_fa_higher = int((diffs > 0).sum())
        ax_b.set_title(
            f"B. Pre-event difference (FA>Miss: {n_fa_higher}/{len(diffs)})"
        )
    else:
        ax_b.text(0.5, 0.5, "No Expert data",
                  transform=ax_b.transAxes, ha="center")
        ax_b.set_title("B. Pre-event difference heatmap")

    # Panel C: Pre-FA ramp across sessions
    ax_c = fig.add_subplot(gs[1, 0])
    sess_ramp = (
        df.groupby(["session_name", "session_idx", "stage"])
        .agg(
            mean_fa_ramp=("fa_ramp", "mean"),
            mean_miss_ramp=("miss_ramp", "mean"),
            n_units=("cluster_id", "count"),
        )
        .reset_index()
        .sort_values("session_idx")
    )

    add_stage_background(ax_c, manifest)
    for stage in STAGE_ORDER:
        sub = sess_ramp[sess_ramp["stage"] == stage]
        if len(sub) > 0:
            ax_c.scatter(
                sub["session_idx"], sub["mean_fa_ramp"],
                c=STAGE_COLORS[stage], s=60, edgecolors="white",
                linewidths=0.5, zorder=3, label=stage,
            )
    if len(sess_ramp) > 0:
        ax_c.plot(
            sess_ramp["session_idx"], sess_ramp["mean_fa_ramp"],
            color="gray", alpha=0.3, linewidth=1, zorder=2,
        )
    ax_c.axhline(0, color="gray", linestyle=":", linewidth=0.5)
    ax_c.set_xlabel("Session index")
    ax_c.set_ylabel("Mean pre-FA ramp (late - early z)")
    ax_c.set_title("C. Pre-FA ramp magnitude across learning")
    ax_c.legend(fontsize=8)

    # Panel D: Pre-FA ramp by HMM behavioral state
    ax_d = fig.add_subplot(gs[1, 1])
    state_data = []
    state_positions = []
    state_box_colors = []

    for i, state in enumerate(HMM_STATE_ORDER):
        col = f"fa_ramp_{state}"
        if col in df.columns:
            vals = df[col].dropna().values
            if len(vals) >= 2:
                state_positions.append(i)
                state_data.append(vals)
                state_box_colors.append(HMM_STATE_COLORS[state])

    if state_data:
        bp = ax_d.boxplot(
            state_data, positions=state_positions, widths=0.5,
            patch_artist=True, showfliers=False,
        )
        for patch, color in zip(bp["boxes"], state_box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for pos, vals, color in zip(
            state_positions, state_data, state_box_colors
        ):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax_d.scatter(
                pos + jitter, vals, c=color, s=20,
                edgecolors="white", linewidths=0.3, zorder=3, alpha=0.5,
            )

    ax_d.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_d.set_xticks(range(len(HMM_STATE_ORDER)))
    ax_d.set_xticklabels(HMM_STATE_ORDER)
    ax_d.set_ylabel("Pre-FA ramp (late - early z)")
    ax_d.set_title("D. Pre-FA ramp by HMM behavioral state")

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # FA vs Miss pre-event activity (Wilcoxon on per-unit late-window diff)
    pre_diff_vals = df["pre_diff"].dropna().values
    if len(pre_diff_vals) >= 10:
        try:
            w, p = wilcoxon(pre_diff_vals)
            stats.append({
                "test": "fa_vs_miss_pre_event_wilcoxon",
                "W": w, "p": p,
                "median_diff": float(np.median(pre_diff_vals)),
                "n": len(pre_diff_vals),
            })
        except ValueError:
            pass

    # Pre-FA ramp vs session index (Spearman)
    if len(sess_ramp) >= 3:
        rho, p = spearmanr(sess_ramp["session_idx"], sess_ramp["mean_fa_ramp"])
        stats.append({
            "test": "fa_ramp_vs_session_spearman",
            "rho": rho, "p": p, "n": len(sess_ramp),
        })

    # Pre-FA ramp by HMM state (Kruskal-Wallis)
    ramp_by_state = []
    for state in HMM_STATE_ORDER:
        col = f"fa_ramp_{state}"
        if col in df.columns:
            vals = df[col].dropna().values
            if len(vals) >= 2 and np.std(vals) > 0:
                ramp_by_state.append(vals)
    if len(ramp_by_state) >= 2:
        try:
            h, p = kruskal(*ramp_by_state)
            stats.append({
                "test": "fa_ramp_kruskal_by_hmm_state",
                "H": h, "p": p,
            })
        except ValueError:
            pass

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig24_fa_neural_signatures", "06_lick_motor")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "06_lick_motor", "fa_neural_signatures_stats.csv",
    )
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: p={row.get('p', 'N/A')}")


if __name__ == "__main__":
    main()
