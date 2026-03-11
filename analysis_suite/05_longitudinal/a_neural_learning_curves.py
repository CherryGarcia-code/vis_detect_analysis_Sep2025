"""05a - Neural learning curves: how neural responses change with learning.

Ties together neural metrics (response magnitude, selectivity, population
coding) with behavioral performance across the learning trajectory.

Produces:
  - Fig 11A: Per-session mean FR change (response - baseline) vs session index
  - Fig 11B: Per-session fraction responsive vs d' (neural-behavioral correlation)
  - Fig 11C: Population response magnitude by stage (boxplot)
  - Fig 11D: Neural metric summary table across stages

Saves: figures/05_longitudinal/neural_learning_stats.csv
"""

import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    STAGE_ORDER, STAGE_COLORS, CACHE_DIR,
)
from loader import load_staging_manifest, load_session, load_glt
from utils import (
    get_good_cluster_ids, build_population_tensor,
    compute_zscore_normalized,
)
from plotting import setup_style, save_figure, add_stage_background

setup_style()

WINDOW = (-0.5, 1.0)
BIN_SIZE = 0.025
BASELINE_WIN = (-0.4, -0.05)
RESP_WIN = (0.0, 0.25)


def main():
    print("[05a] Neural learning curves...")
    manifest = load_staging_manifest(qc_only=True)

    # Collect per-session neural metrics
    records = []
    for _, row in manifest.iterrows():
        sname = int(row["session_name"])
        stage = row["stage"]
        sidx = row["session_idx"]

        print(f"  Session {sname} ({stage})...", end=" ")
        try:
            sess = load_session(sname)
        except FileNotFoundError:
            print("not found")
            continue

        good_ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
        if len(good_ids) < 3:
            print("too few units")
            del sess
            gc.collect()
            continue

        # Build tensor for Hit+Miss trials
        tensor, bc, used = build_population_tensor(
            sess, good_ids, event_name="Change_ON",
            window=WINDOW, bin_size=BIN_SIZE,
            outcome_filter={"Hit", "Miss"},
        )

        if tensor.shape[0] < 5:
            print("too few trials")
            del sess
            gc.collect()
            continue

        resp_mask = (bc >= RESP_WIN[0]) & (bc < RESP_WIN[1])
        base_mask = (bc >= BASELINE_WIN[0]) & (bc < BASELINE_WIN[1])

        # Per-unit metrics
        n_units = tensor.shape[2]
        delta_frs = []
        z_resp = []

        z_tensor = compute_zscore_normalized(tensor, bc, BASELINE_WIN)

        for u in range(n_units):
            resp = np.nanmean(tensor[:, resp_mask, u])
            base = np.nanmean(tensor[:, base_mask, u])
            delta_frs.append(resp - base)

            # Z-scored response magnitude
            z_val = np.nanmean(z_tensor[:, resp_mask, u])
            z_resp.append(z_val)

        # Behavioral d' for this session
        trials = sess.trials
        outcomes = [getattr(t, "trialoutcome", None) for t in trials]
        change_sizes = [getattr(t, "change_size", None) for t in trials]

        go_trials = [(o, cs) for o, cs in zip(outcomes, change_sizes)
                     if cs is not None and cs > 1.0 and o in ("Hit", "Miss")]
        catch_trials = [(o, cs) for o, cs in zip(outcomes, change_sizes)
                        if cs is not None and cs <= 1.01]

        n_go = len(go_trials)
        hits = sum(1 for o, _ in go_trials if o == "Hit")
        catches = len(catch_trials)
        fas = sum(1 for o, _ in catch_trials if o == "Hit")

        hr = np.clip(hits / max(n_go, 1), 0.01, 0.99)
        far = np.clip(fas / max(catches, 1), 0.01, 0.99)
        from scipy.stats import norm
        dprime = float(norm.ppf(hr) - norm.ppf(far))

        # Fraction of units with significant response (z > 2)
        frac_responsive = np.mean([1 if z > 2.0 else 0 for z in z_resp])

        records.append({
            "session_name": sname,
            "session_idx": sidx,
            "stage": stage,
            "n_units": n_units,
            "mean_delta_fr": float(np.mean(delta_frs)),
            "mean_z_resp": float(np.mean(z_resp)),
            "frac_responsive": frac_responsive,
            "dprime": dprime,
        })

        print(f"{n_units} units, deltaFR={np.mean(delta_frs):.2f}, d'={dprime:.2f}")
        del sess
        gc.collect()

    df = pd.DataFrame(records)
    print(f"\n  Collected metrics for {len(df)} sessions")

    if len(df) == 0:
        print("  No data. Exiting.")
        return

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Mean delta FR across sessions
    ax_a = fig.add_subplot(gs[0, 0])
    add_stage_background(ax_a, manifest)
    for stage in STAGE_ORDER:
        sub = df[df["stage"] == stage]
        ax_a.scatter(sub["session_idx"], sub["mean_delta_fr"],
                     c=STAGE_COLORS[stage], s=60, edgecolors="white",
                     linewidths=0.5, zorder=3, label=stage)
    ax_a.plot(df["session_idx"], df["mean_delta_fr"],
              color="gray", alpha=0.3, linewidth=1, zorder=2)
    ax_a.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_a.set_xlabel("Session index")
    ax_a.set_ylabel("Mean ΔFR (Hz)")
    ax_a.set_title("A. Population response magnitude across learning")
    ax_a.legend(fontsize=8)

    # Panel B: Fraction responsive vs d' (neural-behavioral correlation)
    ax_b = fig.add_subplot(gs[0, 1])
    for stage in STAGE_ORDER:
        sub = df[df["stage"] == stage]
        ax_b.scatter(sub["dprime"], sub["frac_responsive"],
                     c=STAGE_COLORS[stage], s=60, edgecolors="white",
                     linewidths=0.5, zorder=3, label=stage)

    if len(df) >= 3:
        rho, p = spearmanr(df["dprime"], df["frac_responsive"])
        ax_b.set_title(f"B. Neural vs behavioral: ρ={rho:.2f}, p={p:.1e}")
    else:
        ax_b.set_title("B. Neural responsiveness vs behavioral d'")
    ax_b.set_xlabel("Behavioral d'")
    ax_b.set_ylabel("Fraction responsive (z > 2)")
    ax_b.legend(fontsize=8)

    # Panel C: Response magnitude by stage
    ax_c = fig.add_subplot(gs[1, 0])
    stage_data = []
    stage_positions = []
    stage_colors = []
    for i, stage in enumerate(STAGE_ORDER):
        vals = df[df["stage"] == stage]["mean_z_resp"].dropna().values
        if len(vals) >= 1:
            stage_positions.append(i)
            stage_data.append(vals)
            stage_colors.append(STAGE_COLORS[stage])

    if stage_data:
        bp = ax_c.boxplot(stage_data, positions=stage_positions, widths=0.5,
                          patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], stage_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for pos, vals, color in zip(stage_positions, stage_data, stage_colors):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax_c.scatter(pos + jitter, vals, c=color, s=40,
                         edgecolors="white", linewidths=0.5, zorder=3)
    ax_c.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_c.set_xticks(range(len(STAGE_ORDER)))
    ax_c.set_xticklabels(STAGE_ORDER)
    ax_c.set_ylabel("Mean z-scored response")
    ax_c.set_title("C. Population response by stage")

    # Panel D: Summary metrics table
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")

    # Create summary table
    table_data = []
    for stage in STAGE_ORDER:
        sub = df[df["stage"] == stage]
        if len(sub) > 0:
            table_data.append([
                stage,
                f"{len(sub)}",
                f"{sub['n_units'].mean():.0f}",
                f"{sub['mean_delta_fr'].mean():.2f}",
                f"{sub['mean_z_resp'].mean():.2f}",
                f"{sub['frac_responsive'].mean():.2f}",
                f"{sub['dprime'].mean():.2f}",
            ])

    if table_data:
        col_labels = ["Stage", "n_sess", "n_units", "ΔFR", "z-resp", "frac_resp", "d'"]
        table = ax_d.table(cellText=table_data, colLabels=col_labels,
                           loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
    ax_d.set_title("D. Summary by stage", y=0.85)

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # Delta FR trend
    if len(df) >= 3:
        rho, p = spearmanr(df["session_idx"], df["mean_delta_fr"])
        stats.append({"test": "delta_fr_vs_session_spearman", "rho": rho, "p": p})

    # z-response trend
    if len(df) >= 3:
        rho, p = spearmanr(df["session_idx"], df["mean_z_resp"])
        stats.append({"test": "z_resp_vs_session_spearman", "rho": rho, "p": p})

    # Neural-behavioral correlation
    if len(df) >= 3:
        rho, p = spearmanr(df["dprime"], df["frac_responsive"])
        stats.append({"test": "frac_resp_vs_dprime_spearman", "rho": rho, "p": p})

    # z-response by stage
    z_groups = [df[df["stage"] == s]["mean_z_resp"].dropna().values for s in STAGE_ORDER]
    z_groups = [g for g in z_groups if len(g) >= 2 and np.std(g) > 0]
    if len(z_groups) >= 2:
        try:
            h, p = kruskal(*z_groups)
            stats.append({"test": "z_resp_kruskal_by_stage", "H": h, "p": p})
        except ValueError:
            pass

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig15_neural_learning", "05_longitudinal")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "05_longitudinal", "neural_learning_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: p={row.get('p', 'N/A')}")


if __name__ == "__main__":
    main()
