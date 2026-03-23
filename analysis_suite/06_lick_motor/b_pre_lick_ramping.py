"""Fig25: Pre-Lick Ramping — units with firing rate ramps before FA events.

Identifies units whose firing rate ramps up (or down) in the one-second
window preceding false-alarm licks.  For each unit, a Spearman correlation
between the mean FA-aligned PSTH and time in [-1.0, 0.0] s classifies it
as "ramping" when rho > 0 and p < 0.05.  A minimum of 5 FA trials per
session is required.

Produces:
  - Fig 25A: Example ramping unit PSTHs (top 3 by rho)
  - Fig 25B: Distribution of ramp rho values (all units)
  - Fig 25C: Fraction of ramping units by cell type (FSI vs MSN)
  - Fig 25D: Fraction ramping by learning stage

Saves: figures/06_lick_motor/pre_lick_ramping_stats.csv
       cache/pre_lick_ramping.csv
"""

import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, chi2_contingency, kruskal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    STAGE_ORDER, STAGE_COLORS, CELLTYPE_COLORS, CACHE_DIR, DEFAULT_BIN_SIZE,
)
from loader import (
    load_staging_manifest, load_session, load_waveform_labels,
)
from utils import (
    get_good_cluster_ids, build_population_tensor, smooth_psth,
)
from plotting import setup_style, save_figure

setup_style()

# ── Parameters ────────────────────────────────────────────────────────
FA_WINDOW = (-2.0, 0.5)
BIN_SIZE = DEFAULT_BIN_SIZE
RAMP_WIN = (-1.0, 0.0)     # window for ramp analysis
MIN_FA_TRIALS = 5
RAMP_P_THRESH = 0.05
MIN_UNITS = 3


def main():
    print("[06b] Pre-lick ramping analysis...")
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

    # ── Collect per-unit ramp metrics ─────────────────────────────────
    all_units = []
    # Keep track of top ramping PSTHs for Panel A
    top_ramp_psths = []  # (rho, psth_array, bin_centers, sname, cid)

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

        # Build FA tensor aligned to FA lick time
        fa_tensor, bc, fa_used = build_population_tensor(
            sess, good_ids, event_name="FA",
            window=FA_WINDOW, bin_size=BIN_SIZE,
            outcome_filter={"FA"},
        )

        if fa_tensor.shape[0] < MIN_FA_TRIALS:
            print(f"too few FA trials ({fa_tensor.shape[0]})")
            del sess
            gc.collect()
            continue

        # Ramp-window mask
        ramp_mask = (bc >= RAMP_WIN[0]) & (bc < RAMP_WIN[1])
        ramp_bc = bc[ramp_mask]

        n_added = 0
        for u_i, cid in enumerate(good_ids):
            if u_i >= fa_tensor.shape[2]:
                break

            # Mean PSTH across FA trials
            unit_psth = np.nanmean(fa_tensor[:, :, u_i], axis=0)  # (n_bins,)
            ramp_psth = unit_psth[ramp_mask]

            if len(ramp_psth) < 5 or np.all(np.isnan(ramp_psth)):
                continue

            # Lightly smooth for ramp correlation
            ramp_smooth = smooth_psth(ramp_psth, BIN_SIZE, sigma_ms=25.0)

            finite_mask = np.isfinite(ramp_smooth) & np.isfinite(ramp_bc)
            if finite_mask.sum() < 5:
                continue

            rho, p_val = spearmanr(
                ramp_bc[finite_mask], ramp_smooth[finite_mask]
            )
            is_ramping = (rho > 0) and (p_val < RAMP_P_THRESH)

            ct = ct_lookup.get((sname, cid), "Unknown")

            all_units.append({
                "session_name": sname,
                "cluster_id": cid,
                "stage": stage,
                "session_idx": sidx,
                "cell_type": ct,
                "ramp_rho": rho,
                "ramp_p": p_val,
                "is_ramping": is_ramping,
                "n_fa": fa_tensor.shape[0],
            })

            # Store full smoothed PSTH for top-ramp examples
            full_psth_smooth = smooth_psth(unit_psth, BIN_SIZE, sigma_ms=25.0)
            top_ramp_psths.append((rho, full_psth_smooth, bc, sname, cid))
            n_added += 1

        print(f"{n_added} units")
        del sess
        gc.collect()

    df = pd.DataFrame(all_units)
    print(f"\n  Total: {len(df)} units analyzed")

    if len(df) == 0:
        print("  No data. Exiting.")
        return

    n_ramping = int(df["is_ramping"].sum())
    print(f"  Ramping units: {n_ramping}/{len(df)} "
          f"({100 * n_ramping / len(df):.1f}%)")

    # Cache
    cache_path = os.path.join(CACHE_DIR, "pre_lick_ramping.csv")
    df.to_csv(cache_path, index=False)

    # Sort top-ramp examples by rho (descending)
    top_ramp_psths.sort(key=lambda x: x[0], reverse=True)

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Top 3 ramping unit PSTHs
    ax_a = fig.add_subplot(gs[0, 0])
    n_examples = min(3, len(top_ramp_psths))
    colors_ex = ["#E53935", "#FB8C00", "#43A047"]
    for i in range(n_examples):
        rho_val, psth_val, bc_val, sname_val, cid_val = top_ramp_psths[i]
        ax_a.plot(
            bc_val, psth_val, linewidth=1.5, color=colors_ex[i],
            label=f"Unit {cid_val} (rho={rho_val:.2f})",
        )
    ax_a.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_a.axvspan(
        RAMP_WIN[0], RAMP_WIN[1], alpha=0.08, color="red",
        label="Ramp window",
    )
    ax_a.set_xlabel("Time from FA (s)")
    ax_a.set_ylabel("Firing rate (Hz)")
    ax_a.set_title(f"A. Top ramping units (n={n_examples} examples)")
    ax_a.legend(fontsize=8)

    # Panel B: Distribution of ramp rho values
    ax_b = fig.add_subplot(gs[0, 1])
    rho_vals = df["ramp_rho"].dropna().values

    if len(rho_vals) > 0:
        ax_b.hist(
            rho_vals, bins=40, color="#78909C", edgecolor="white",
            linewidth=0.5, alpha=0.8,
        )
        # Overlay significant ramping subset
        sig_rho = df[df["is_ramping"]]["ramp_rho"].dropna().values
        if len(sig_rho) > 0:
            ax_b.hist(
                sig_rho, bins=40, color="#E53935", edgecolor="white",
                linewidth=0.5, alpha=0.6,
                label=f"Ramping (n={len(sig_rho)})",
            )
        ax_b.axvline(0, color="k", linestyle="--", linewidth=0.8)
        ax_b.axvline(
            np.median(rho_vals), color="#1976D2", linestyle="-",
            linewidth=1.5, label=f"Median={np.median(rho_vals):.3f}",
        )

    ax_b.set_xlabel("Spearman rho (PSTH vs time)")
    ax_b.set_ylabel("Number of units")
    ax_b.set_title(f"B. Pre-FA ramp distribution (n={len(rho_vals)} units)")
    ax_b.legend(fontsize=8)

    # Panel C: Fraction ramping by cell type
    ax_c = fig.add_subplot(gs[1, 0])
    ct_types = ["Narrow (FSI)", "Broad (MSN/Proj)"]
    ct_short = ["FSI", "MSN"]
    ct_fracs = []
    ct_ns = []
    ct_colors_list = []

    for ct in ct_types:
        sub = df[df["cell_type"] == ct]
        if len(sub) > 0:
            ct_fracs.append(sub["is_ramping"].mean())
            ct_ns.append((int(sub["is_ramping"].sum()), len(sub)))
        else:
            ct_fracs.append(0)
            ct_ns.append((0, 0))
        ct_colors_list.append(CELLTYPE_COLORS[ct])

    bar_x = range(len(ct_types))
    ax_c.bar(bar_x, ct_fracs, color=ct_colors_list,
             edgecolor="white", linewidth=1)
    for i, (frac, (n_sig, n_total)) in enumerate(zip(ct_fracs, ct_ns)):
        if n_total > 0:
            ax_c.text(i, frac + 0.01, f"{n_sig}/{n_total}",
                      ha="center", fontsize=9)

    ax_c.set_xticks(list(bar_x))
    ax_c.set_xticklabels(ct_short)
    ax_c.set_ylabel("Fraction ramping")
    ymax_c = max(ct_fracs) * 1.4 if ct_fracs and max(ct_fracs) > 0 else 0.5
    ax_c.set_ylim(0, ymax_c)
    ax_c.set_title("C. Fraction ramping by cell type")

    # Panel D: Fraction ramping by stage
    ax_d = fig.add_subplot(gs[1, 1])
    stage_fracs = []
    stage_ns = []
    stage_bar_colors = []

    for stage in STAGE_ORDER:
        sub = df[df["stage"] == stage]
        if len(sub) > 0:
            stage_fracs.append(sub["is_ramping"].mean())
            stage_ns.append((int(sub["is_ramping"].sum()), len(sub)))
        else:
            stage_fracs.append(0)
            stage_ns.append((0, 0))
        stage_bar_colors.append(STAGE_COLORS[stage])

    bar_x = range(len(STAGE_ORDER))
    ax_d.bar(bar_x, stage_fracs, color=stage_bar_colors,
             edgecolor="white", linewidth=1)
    for i, (frac, (n_sig, n_total)) in enumerate(zip(stage_fracs, stage_ns)):
        if n_total > 0:
            ax_d.text(i, frac + 0.01, f"{n_sig}/{n_total}",
                      ha="center", fontsize=9)

    ax_d.set_xticks(list(bar_x))
    ax_d.set_xticklabels(STAGE_ORDER)
    ax_d.set_ylabel("Fraction ramping")
    ymax_d = max(stage_fracs) * 1.4 if stage_fracs and max(stage_fracs) > 0 else 0.5
    ax_d.set_ylim(0, ymax_d)
    ax_d.set_title("D. Fraction ramping by learning stage")

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # Chi-square: ramping fraction by cell type
    fsi = df[df["cell_type"] == "Narrow (FSI)"]
    msn = df[df["cell_type"] == "Broad (MSN/Proj)"]
    if len(fsi) >= 5 and len(msn) >= 5:
        contingency = np.array([
            [fsi["is_ramping"].sum(), (~fsi["is_ramping"]).sum()],
            [msn["is_ramping"].sum(), (~msn["is_ramping"]).sum()],
        ])
        if contingency.min() >= 0:
            try:
                chi2, p, dof, _ = chi2_contingency(contingency)
                stats.append({
                    "test": "ramping_chi2_fsi_vs_msn",
                    "chi2": chi2, "p": p, "dof": dof,
                })
            except ValueError:
                pass

    # Kruskal-Wallis: rho by stage
    stage_groups = [
        df[df["stage"] == s]["ramp_rho"].dropna().values
        for s in STAGE_ORDER
    ]
    stage_groups = [g for g in stage_groups if len(g) >= 2 and np.std(g) > 0]
    if len(stage_groups) >= 2:
        try:
            h, p = kruskal(*stage_groups)
            stats.append({
                "test": "rho_kruskal_by_stage",
                "H": h, "p": p,
            })
        except ValueError:
            pass

    # Overall ramping fraction
    stats.append({
        "test": "overall_ramping_fraction",
        "frac_ramping": float(df["is_ramping"].mean()),
        "n_ramping": int(df["is_ramping"].sum()),
        "n_total": len(df),
    })

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig25_pre_lick_ramping", "06_lick_motor")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "06_lick_motor", "pre_lick_ramping_stats.csv",
    )
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: p={row.get('p', 'N/A')}")


if __name__ == "__main__":
    main()
