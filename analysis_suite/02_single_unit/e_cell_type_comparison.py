"""Fig 12: Cell-type comparison — FSI vs MSN response profiles.

Compares narrow-spiking (putative FSI) and broad-spiking (putative MSN/Proj)
neurons across multiple response dimensions using the pre-computed
waveform cell-type labels.

Produces:
  - Fig 10A: Mean PSTH by cell type (Expert sessions, Change_ON)
  - Fig 10B: Firing rate distributions by cell type
  - Fig 10C: Response magnitude (Change_ON delta FR) by cell type x stage
  - Fig 10D: Outcome selectivity (auROC) by cell type

Saves: figures/02_single_unit/celltype_comparison_stats.csv
"""

import os
import sys
import gc


import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import (
    STAGE_ORDER, STAGE_COLORS, CELLTYPE_COLORS, CACHE_DIR, DEFAULT_BIN_SIZE,
)
from visdetect.suite.loader import (
    load_staging_manifest, load_session, load_glt, load_waveform_labels,
)
from visdetect.analysis.utils import (
    get_good_cluster_ids, build_population_tensor, smooth_psth,
    compute_zscore_normalized,
)
from visdetect.suite.plotting import setup_style, save_figure

setup_style()

WINDOW = (-0.5, 1.0)
BIN_SIZE = DEFAULT_BIN_SIZE
BASELINE_WIN = (-0.4, -0.05)
RESP_WIN = (0.0, 0.25)


def main():
    print("[02e] Cell-type comparison: FSI vs MSN...")
    manifest = load_staging_manifest(qc_only=True)

    # Load cell-type labels
    wf_labels = load_waveform_labels()
    if wf_labels is None or len(wf_labels) == 0:
        print("  No waveform labels available. Exiting.")
        return

    # Ensure session_name is int for matching
    if "session_name" in wf_labels.columns:
        wf_labels["session_name"] = wf_labels["session_name"].astype(int)

    print(f"  Waveform labels: {len(wf_labels)} units")
    print(f"    Cell types: {wf_labels['cell_type'].value_counts().to_dict()}")

    # Build lookup: (session_name, cluster_id) -> cell_type
    ct_lookup = {}
    for _, row in wf_labels.iterrows():
        key = (int(row["session_name"]), int(row["cluster_id"]))
        ct_lookup[key] = row["cell_type"]

    # Collect per-unit metrics across all sessions
    all_units = []
    all_psths = {"Narrow (FSI)": [], "Broad (MSN/Proj)": []}
    bin_centers_ref = None

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

        # Build tensor
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

        bin_centers_ref = bc
        z_tensor = compute_zscore_normalized(tensor, bc, BASELINE_WIN)

        # Response and baseline masks
        resp_mask = (bc >= RESP_WIN[0]) & (bc < RESP_WIN[1])
        base_mask = (bc >= BASELINE_WIN[0]) & (bc < BASELINE_WIN[1])

        n_units_added = 0
        for u_i, cid in enumerate(good_ids):
            if u_i >= tensor.shape[2]:
                break

            ct = ct_lookup.get((sname, cid), None)
            if ct is None:
                continue

            # Mean firing rates
            unit_tensor = tensor[:, :, u_i]  # (n_trials, n_bins)
            mean_resp_fr = float(np.nanmean(unit_tensor[:, resp_mask]))
            mean_base_fr = float(np.nanmean(unit_tensor[:, base_mask]))
            delta_fr = mean_resp_fr - mean_base_fr

            # Mean z-scored PSTH
            z_mean = np.nanmean(z_tensor[:, :, u_i], axis=0)
            z_smoothed = smooth_psth(z_mean, BIN_SIZE, sigma_ms=15.0)

            # Overall firing rate
            overall_fr = float(np.nanmean(unit_tensor))

            all_units.append({
                "session_name": sname,
                "cluster_id": cid,
                "stage": stage,
                "session_idx": sidx,
                "cell_type": ct,
                "overall_fr": overall_fr,
                "mean_resp_fr": mean_resp_fr,
                "mean_base_fr": mean_base_fr,
                "delta_fr": delta_fr,
            })

            # Collect PSTHs for Expert sessions
            if stage == "Expert" and ct in all_psths:
                all_psths[ct].append(z_smoothed)

            n_units_added += 1

        print(f"{n_units_added} typed units")
        del sess
        gc.collect()

    df = pd.DataFrame(all_units)
    print(f"\n  Total: {len(df)} typed units")
    if len(df) == 0:
        print("  No data. Exiting.")
        return

    print(f"    {(df['cell_type']=='Narrow (FSI)').sum()} FSI, "
          f"{(df['cell_type']=='Broad (MSN/Proj)').sum()} MSN")

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Mean PSTH by cell type (Expert)
    ax_a = fig.add_subplot(gs[0, 0])
    bc = bin_centers_ref

    for ct, ct_label in [("Narrow (FSI)", "FSI"), ("Broad (MSN/Proj)", "MSN")]:
        psth_list = all_psths[ct]
        if len(psth_list) > 0:
            mat = np.array(psth_list)
            mean_psth = np.nanmean(mat, axis=0)
            sem_psth = np.nanstd(mat, axis=0) / np.sqrt(len(psth_list))
            ax_a.plot(bc, mean_psth, color=CELLTYPE_COLORS[ct], linewidth=2,
                      label=f"{ct_label} (n={len(psth_list)})")
            ax_a.fill_between(bc, mean_psth - sem_psth, mean_psth + sem_psth,
                              color=CELLTYPE_COLORS[ct], alpha=0.2)

    ax_a.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_a.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_a.set_xlabel("Time from Change_ON (s)")
    ax_a.set_ylabel("z-score")
    ax_a.set_title("A. Population PSTH by cell type (Expert)")
    ax_a.legend(fontsize=9)

    # Panel B: Firing rate distributions
    ax_b = fig.add_subplot(gs[0, 1])
    for ct in ["Narrow (FSI)", "Broad (MSN/Proj)"]:
        vals = df[df["cell_type"] == ct]["overall_fr"].dropna().values
        if len(vals) > 0:
            # Log-scale FR
            vals_log = np.log10(vals[vals > 0])
            ax_b.hist(vals_log, bins=30, color=CELLTYPE_COLORS[ct],
                      alpha=0.5, label=ct, density=True)
    ax_b.set_xlabel("log10(Firing rate, Hz)")
    ax_b.set_ylabel("Density")
    ax_b.set_title("B. Firing rate distributions by cell type")
    ax_b.legend(fontsize=8)

    # Panel C: Response magnitude by cell type x stage
    ax_c = fig.add_subplot(gs[1, 0])
    width = 0.35
    cell_types = ["Narrow (FSI)", "Broad (MSN/Proj)"]
    ct_short = ["FSI", "MSN"]

    for ct_i, ct in enumerate(cell_types):
        means, sems = [], []
        for stage in STAGE_ORDER:
            sub = df[(df["cell_type"] == ct) & (df["stage"] == stage)]
            vals = sub["delta_fr"].dropna().values
            if len(vals) >= 2:
                means.append(np.mean(vals))
                sems.append(np.std(vals) / np.sqrt(len(vals)))
            else:
                means.append(0)
                sems.append(0)

        x = np.arange(len(STAGE_ORDER))
        offset = (ct_i - 0.5) * width
        ax_c.bar(x + offset, means, width * 0.9, yerr=sems,
                 color=CELLTYPE_COLORS[ct], alpha=0.7,
                 label=ct_short[ct_i], capsize=3, edgecolor="white")

    ax_c.set_xticks(range(len(STAGE_ORDER)))
    ax_c.set_xticklabels(STAGE_ORDER)
    ax_c.set_ylabel("ΔFR (response - baseline, Hz)")
    ax_c.set_title("C. Response magnitude: FSI vs MSN by stage")
    ax_c.legend(fontsize=8)
    ax_c.axhline(0, color="gray", linewidth=0.5, linestyle=":")

    # Panel D: Cell type proportions by stage
    ax_d = fig.add_subplot(gs[1, 1])
    for ct_i, ct in enumerate(cell_types):
        fracs = []
        for stage in STAGE_ORDER:
            stage_total = len(df[df["stage"] == stage])
            ct_count = len(df[(df["stage"] == stage) & (df["cell_type"] == ct)])
            fracs.append(ct_count / max(stage_total, 1))

        x = np.arange(len(STAGE_ORDER))
        offset = (ct_i - 0.5) * width
        ax_d.bar(x + offset, fracs, width * 0.9,
                 color=CELLTYPE_COLORS[ct], alpha=0.7,
                 label=ct_short[ct_i], edgecolor="white")

    ax_d.set_xticks(range(len(STAGE_ORDER)))
    ax_d.set_xticklabels(STAGE_ORDER)
    ax_d.set_ylabel("Fraction of units")
    ax_d.set_title("D. Cell-type composition by stage")
    ax_d.legend(fontsize=8)

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # FR comparison: FSI vs MSN
    fsi_fr = df[df["cell_type"] == "Narrow (FSI)"]["overall_fr"].dropna().values
    msn_fr = df[df["cell_type"] == "Broad (MSN/Proj)"]["overall_fr"].dropna().values
    if len(fsi_fr) >= 2 and len(msn_fr) >= 2:
        u, p = mannwhitneyu(fsi_fr, msn_fr, alternative="two-sided")
        stats.append({
            "test": "fr_fsi_vs_msn_mwu", "U": u, "p": p,
            "median_fsi": float(np.median(fsi_fr)),
            "median_msn": float(np.median(msn_fr)),
        })

    # Delta FR comparison: FSI vs MSN
    fsi_delta = df[df["cell_type"] == "Narrow (FSI)"]["delta_fr"].dropna().values
    msn_delta = df[df["cell_type"] == "Broad (MSN/Proj)"]["delta_fr"].dropna().values
    if len(fsi_delta) >= 2 and len(msn_delta) >= 2:
        u, p = mannwhitneyu(fsi_delta, msn_delta, alternative="two-sided")
        stats.append({
            "test": "delta_fr_fsi_vs_msn_mwu", "U": u, "p": p,
            "median_fsi": float(np.median(fsi_delta)),
            "median_msn": float(np.median(msn_delta)),
        })

    # Expert only: celltype response comparison
    for ct in cell_types:
        expert_delta = df[(df["cell_type"] == ct) & (df["stage"] == "Expert")]["delta_fr"].dropna()
        naive_delta = df[(df["cell_type"] == ct) & (df["stage"] == "Naive")]["delta_fr"].dropna()
        if len(expert_delta) >= 2 and len(naive_delta) >= 2:
            u, p = mannwhitneyu(expert_delta, naive_delta, alternative="two-sided")
            ct_short_name = "fsi" if "FSI" in ct else "msn"
            stats.append({
                "test": f"delta_fr_{ct_short_name}_expert_vs_naive", "U": u, "p": p,
            })

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig12_celltype_comparison", "02_single_unit")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "02_single_unit", "celltype_comparison_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: p={row.get('p', 'N/A')}")


if __name__ == "__main__":
    main()
