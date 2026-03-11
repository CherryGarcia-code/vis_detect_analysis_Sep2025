"""06c - Motor vs sensory dissociation: Hit (Change_ON) vs FA contrast.

Computes a Sensory Index (SI) per unit by comparing the post-event
response on Hit trials (sensory stimulus change present) to FA trials
(no stimulus change, only lick).  Hit trials are aligned to Change_ON;
FA trials are aligned to the FA lick time.

    SI = (Hit_resp - FA_resp) / (|Hit_resp| + |FA_resp| + eps)

SI > 0 indicates a sensory-driven (change-responsive) unit.
SI < 0 indicates a motor/lick-driven unit.

Produces:
  - Fig 18A: Population PSTH -- Hit vs FA (Expert sessions)
  - Fig 18B: Scatter of Hit response vs FA response per unit
  - Fig 18C: Sensory Index histogram colored by cell type
  - Fig 18D: Sensory Index by learning stage (boxplot)

Saves: figures/06_lick_motor/motor_vs_sensory_stats.csv
       cache/motor_vs_sensory.csv
"""

import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu, kruskal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    STAGE_ORDER, STAGE_COLORS, OUTCOME_COLORS, CELLTYPE_COLORS, CACHE_DIR,
)
from loader import (
    load_staging_manifest, load_session, load_waveform_labels,
)
from utils import (
    get_good_cluster_ids, build_population_tensor, smooth_psth,
    compute_zscore_normalized,
)
from plotting import setup_style, save_figure

setup_style()

# ── Parameters ────────────────────────────────────────────────────────
HIT_WINDOW = (-0.5, 1.0)
FA_WINDOW = (-0.5, 1.0)
BIN_SIZE = 0.025
BASELINE_WIN = (-0.4, -0.05)
RESP_WIN = (0.0, 0.25)
MIN_TRIALS = 5
MIN_UNITS = 3
EPS = 1e-6


def main():
    print("[06c] Motor vs sensory dissociation (Hit vs FA)...")
    manifest = load_staging_manifest(qc_only=True)

    # Load cell-type labels (graceful if unavailable)
    ct_lookup = {}
    try:
        wf_labels = load_waveform_labels()
        if wf_labels is not None and len(wf_labels) > 0:
            for _, row in wf_labels.iterrows():
                ct_lookup[
                    (int(row["session_name"]), int(row["cluster_id"]))
                ] = row["cell_type"]
    except FileNotFoundError:
        print("  Warning: Waveform labels not found; "
              "cell-type panels will be empty.")

    # ── Collect per-unit metrics ──────────────────────────────────────
    all_units = []
    hit_pop_psths = []      # per-session Expert population PSTHs
    fa_pop_psths = []
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

        # Hit tensor: aligned to Change_ON, Hit trials only
        hit_tensor, hit_bc, hit_used = build_population_tensor(
            sess, good_ids, event_name="Change_ON",
            window=HIT_WINDOW, bin_size=BIN_SIZE,
            outcome_filter={"Hit"},
        )

        # FA tensor: aligned to FA lick time
        fa_tensor, fa_bc, fa_used = build_population_tensor(
            sess, good_ids, event_name="FA",
            window=FA_WINDOW, bin_size=BIN_SIZE,
            outcome_filter={"FA"},
        )

        if hit_tensor.shape[0] < MIN_TRIALS or fa_tensor.shape[0] < MIN_TRIALS:
            print(
                f"too few trials (Hit={hit_tensor.shape[0]}, "
                f"FA={fa_tensor.shape[0]})"
            )
            del sess
            gc.collect()
            continue

        bin_centers_ref = hit_bc
        bc = hit_bc

        # Z-score normalize both
        hit_z = compute_zscore_normalized(hit_tensor, bc, BASELINE_WIN)
        fa_z = compute_zscore_normalized(fa_tensor, fa_bc, BASELINE_WIN)

        # Response-window mask
        resp_mask = (bc >= RESP_WIN[0]) & (bc < RESP_WIN[1])

        n_added = 0
        for u_i, cid in enumerate(good_ids):
            if u_i >= hit_z.shape[2] or u_i >= fa_z.shape[2]:
                break

            # Mean z-scored response in response window
            hit_resp = float(np.nanmean(hit_z[:, resp_mask, u_i]))
            fa_resp = float(np.nanmean(fa_z[:, resp_mask, u_i]))

            # Sensory Index
            denom = abs(hit_resp) + abs(fa_resp) + EPS
            si = (hit_resp - fa_resp) / denom

            ct = ct_lookup.get((sname, cid), "Unknown")

            all_units.append({
                "session_name": sname,
                "cluster_id": cid,
                "stage": stage,
                "session_idx": sidx,
                "cell_type": ct,
                "hit_resp": hit_resp,
                "fa_resp": fa_resp,
                "sensory_index": si,
                "n_hit": hit_tensor.shape[0],
                "n_fa": fa_tensor.shape[0],
            })
            n_added += 1

        # Expert population PSTHs for Panel A
        if stage == "Expert":
            # Mean over trials, then over units -> (n_bins,)
            hit_pop = np.nanmean(np.nanmean(hit_z, axis=0), axis=1)
            fa_pop = np.nanmean(np.nanmean(fa_z, axis=0), axis=1)
            hit_pop_psths.append(smooth_psth(hit_pop, BIN_SIZE, sigma_ms=25.0))
            fa_pop_psths.append(smooth_psth(fa_pop, BIN_SIZE, sigma_ms=25.0))

        print(f"{n_added} units")
        del sess
        gc.collect()

    df = pd.DataFrame(all_units)
    print(f"\n  Total: {len(df)} units with Hit/FA data")

    if len(df) == 0 or bin_centers_ref is None:
        print("  No data. Exiting.")
        return

    # Cache
    cache_path = os.path.join(CACHE_DIR, "motor_vs_sensory.csv")
    df.to_csv(cache_path, index=False)

    bc = bin_centers_ref

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Population PSTH -- Hit vs FA (Expert)
    ax_a = fig.add_subplot(gs[0, 0])
    if len(hit_pop_psths) > 0 and len(fa_pop_psths) > 0:
        hit_mat = np.array(hit_pop_psths)
        fa_mat = np.array(fa_pop_psths)

        hit_mean = np.nanmean(hit_mat, axis=0)
        hit_sem = np.nanstd(hit_mat, axis=0) / np.sqrt(len(hit_pop_psths))
        fa_mean = np.nanmean(fa_mat, axis=0)
        fa_sem = np.nanstd(fa_mat, axis=0) / np.sqrt(len(fa_pop_psths))

        ax_a.plot(bc, hit_mean, color=OUTCOME_COLORS["Hit"], linewidth=2,
                  label=f"Hit (n={len(hit_pop_psths)} sess)")
        ax_a.fill_between(bc, hit_mean - hit_sem, hit_mean + hit_sem,
                          color=OUTCOME_COLORS["Hit"], alpha=0.2)
        ax_a.plot(bc, fa_mean, color=OUTCOME_COLORS["FA"], linewidth=2,
                  label=f"FA (n={len(fa_pop_psths)} sess)")
        ax_a.fill_between(bc, fa_mean - fa_sem, fa_mean + fa_sem,
                          color=OUTCOME_COLORS["FA"], alpha=0.2)
    else:
        ax_a.text(0.5, 0.5, "No Expert data",
                  transform=ax_a.transAxes, ha="center")

    ax_a.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_a.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_a.set_xlabel("Time from event (s)")
    ax_a.set_ylabel("Population z-score")
    ax_a.set_title("A. Hit vs FA population response (Expert)")
    ax_a.legend(fontsize=9)

    # Panel B: Scatter -- Hit response vs FA response per unit
    ax_b = fig.add_subplot(gs[0, 1])
    finite = df[df["hit_resp"].notna() & df["fa_resp"].notna()]

    if len(finite) > 0:
        # Plot typed units
        for ct, ct_label in [("Narrow (FSI)", "FSI"),
                              ("Broad (MSN/Proj)", "MSN")]:
            sub = finite[finite["cell_type"] == ct]
            if len(sub) > 0:
                ax_b.scatter(
                    sub["hit_resp"], sub["fa_resp"],
                    c=CELLTYPE_COLORS[ct], s=25, alpha=0.5,
                    edgecolors="white", linewidths=0.3,
                    label=f"{ct_label} (n={len(sub)})", zorder=3,
                )

        # Plot untyped units in gray
        unk = finite[
            ~finite["cell_type"].isin(["Narrow (FSI)", "Broad (MSN/Proj)"])
        ]
        if len(unk) > 0:
            ax_b.scatter(
                unk["hit_resp"], unk["fa_resp"],
                c="#bdbdbd", s=15, alpha=0.3, edgecolors="none",
                label=f"Other (n={len(unk)})", zorder=2,
            )

        # Unity line
        all_vals = np.concatenate(
            [finite["hit_resp"].values, finite["fa_resp"].values]
        )
        lims = [np.nanpercentile(all_vals, 2), np.nanpercentile(all_vals, 98)]
        ax_b.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="Unity")

    ax_b.set_xlabel("Hit response (z-score)")
    ax_b.set_ylabel("FA response (z-score)")
    ax_b.set_title("B. Hit vs FA response per unit")
    ax_b.legend(fontsize=8)

    # Panel C: Sensory Index histogram by cell type
    ax_c = fig.add_subplot(gs[1, 0])
    si_vals = df["sensory_index"].dropna().values

    if len(si_vals) > 0:
        # Background: all units
        ax_c.hist(
            si_vals, bins=40, color="#bdbdbd", edgecolor="white",
            linewidth=0.5, alpha=0.5, label=f"All (n={len(si_vals)})",
        )
        # Overlay per cell type
        for ct, ct_label in [("Narrow (FSI)", "FSI"),
                              ("Broad (MSN/Proj)", "MSN")]:
            ct_si = df[df["cell_type"] == ct]["sensory_index"].dropna().values
            if len(ct_si) > 0:
                ax_c.hist(
                    ct_si, bins=40, color=CELLTYPE_COLORS[ct],
                    edgecolor="white", linewidth=0.5, alpha=0.5,
                    label=f"{ct_label} (n={len(ct_si)})",
                )
        ax_c.axvline(0, color="k", linestyle="--", linewidth=0.8)

    ax_c.set_xlabel("Sensory Index (+ sensory, - motor)")
    ax_c.set_ylabel("Number of units")
    ax_c.set_title("C. Sensory Index distribution by cell type")
    ax_c.legend(fontsize=8)

    # Panel D: Sensory Index by stage (boxplot)
    ax_d = fig.add_subplot(gs[1, 1])
    stage_data = []
    stage_positions = []
    stage_box_colors = []

    for i, stage in enumerate(STAGE_ORDER):
        vals = df[df["stage"] == stage]["sensory_index"].dropna().values
        if len(vals) >= 2:
            stage_positions.append(i)
            stage_data.append(vals)
            stage_box_colors.append(STAGE_COLORS[stage])

    if stage_data:
        bp = ax_d.boxplot(
            stage_data, positions=stage_positions, widths=0.5,
            patch_artist=True, showfliers=False,
        )
        for patch, color in zip(bp["boxes"], stage_box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for pos, vals, color in zip(
            stage_positions, stage_data, stage_box_colors
        ):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax_d.scatter(
                pos + jitter, vals, c=color, s=20,
                edgecolors="white", linewidths=0.3, zorder=3, alpha=0.5,
            )

    ax_d.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_d.set_xticks(range(len(STAGE_ORDER)))
    ax_d.set_xticklabels(STAGE_ORDER)
    ax_d.set_ylabel("Sensory Index")
    ax_d.set_title("D. Sensory Index by learning stage")

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # SI vs 0 (Wilcoxon signed-rank)
    finite_si = df["sensory_index"].dropna().values
    if len(finite_si) >= 10:
        try:
            w, p = wilcoxon(finite_si)
            stats.append({
                "test": "si_vs_zero_wilcoxon",
                "W": w, "p": p,
                "median_si": float(np.median(finite_si)),
                "n": len(finite_si),
            })
        except ValueError:
            pass

    # SI by stage (Kruskal-Wallis)
    stage_groups = [
        df[df["stage"] == s]["sensory_index"].dropna().values
        for s in STAGE_ORDER
    ]
    stage_groups = [g for g in stage_groups if len(g) >= 2 and np.std(g) > 0]
    if len(stage_groups) >= 2:
        try:
            h, p = kruskal(*stage_groups)
            stats.append({
                "test": "si_kruskal_by_stage",
                "H": h, "p": p,
            })
        except ValueError:
            pass

    # SI: FSI vs MSN (Mann-Whitney U)
    fsi_si = df[df["cell_type"] == "Narrow (FSI)"]["sensory_index"].dropna().values
    msn_si = df[df["cell_type"] == "Broad (MSN/Proj)"]["sensory_index"].dropna().values
    if len(fsi_si) >= 2 and len(msn_si) >= 2:
        u, p = mannwhitneyu(fsi_si, msn_si, alternative="two-sided")
        stats.append({
            "test": "si_fsi_vs_msn_mwu",
            "U": u, "p": p,
            "median_fsi": float(np.median(fsi_si)),
            "median_msn": float(np.median(msn_si)),
        })

    # Fraction sensory vs motor
    n_sensory = int((finite_si > 0).sum())
    n_motor = int((finite_si < 0).sum())
    stats.append({
        "test": "fraction_sensory_vs_motor",
        "n_sensory": n_sensory,
        "n_motor": n_motor,
        "frac_sensory": float(n_sensory / max(len(finite_si), 1)),
    })

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig20_motor_vs_sensory", "06_lick_motor")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "06_lick_motor", "motor_vs_sensory_stats.csv",
    )
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: p={row.get('p', 'N/A')}")


if __name__ == "__main__":
    main()
