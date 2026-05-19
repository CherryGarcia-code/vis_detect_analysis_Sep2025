"""Fig22: Cell-Type Learning — FSI vs MSN differential learning trajectories.

Compares how Narrow (FSI) and Broad (MSN/Proj) cell types differ
in their response properties across learning stages.

Produces:
  - Fig 22A: Mean response magnitude (delta FR) across sessions by cell type
  - Fig 22B: Fraction responsive by stage and cell type
  - Fig 22C: Response latency by cell type across stages
  - Fig 22D: Cell-type ratio (FSI/MSN) of responsive neurons across sessions

Saves: figures/05_longitudinal/celltype_learning_stats.csv
"""

import os
import sys
import gc


import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import (
    STAGE_ORDER, STAGE_COLORS, CELLTYPE_COLORS, CACHE_DIR, DEFAULT_BIN_SIZE,
)
from visdetect.suite.loader import (
    load_staging_manifest, load_session, load_waveform_labels,
)
from visdetect.analysis.utils import get_good_cluster_ids, build_population_tensor, compute_zscore_normalized
from visdetect.suite.plotting import setup_style, save_figure, add_stage_background

from visdetect.analysis.align import align_spikes_to_events, get_event_times_by_trial

setup_style()

WINDOW = (-0.5, 0.5)
BIN_SIZE = DEFAULT_BIN_SIZE
BASELINE_WIN = (-0.5, -0.05)
RESP_WIN = (0.0, 0.25)
Z_THRESH = 2.0


def main():
    print("[05b] Cell-type learning trajectories...")
    manifest = load_staging_manifest(qc_only=True)

    try:
        wf_labels = load_waveform_labels()
    except FileNotFoundError:
        print("  No waveform labels. Exiting.")
        return

    # Build cell-type lookup
    ct_lookup = {}
    for _, row in wf_labels.iterrows():
        key = (int(row["session_name"]) if "session_name" in row.index
               else int(row.get("session_date", 0)))
        ct_lookup[(key, int(row["cluster_id"]))] = row.get("cell_type", row.get("celltype", "Unknown"))

    all_records = []

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
        event_times = get_event_times_by_trial(sess, "Change_ON")
        cluster_map = {int(c.cluster_id): c for c in sess.clusters}

        # Filter Hit+Miss trials
        valid_trials = []
        for i, t in enumerate(sess.trials):
            outcome = getattr(t, "trialoutcome", None)
            if outcome in ("Hit", "Miss") and i < len(event_times) and np.isfinite(event_times[i]):
                valid_trials.append(i)

        if len(valid_trials) < 10:
            print("too few trials")
            del sess; gc.collect()
            continue

        trial_event_times = [float(event_times[i]) for i in valid_trials]
        n_units = 0

        for cid in good_ids:
            c = cluster_map.get(int(cid))
            if c is None:
                continue

            ct = ct_lookup.get((sname, int(cid)), "Unknown")
            if ct == "Unknown":
                continue

            mat, bc = align_spikes_to_events(
                c.spike_times, trial_event_times,
                window=WINDOW, bin_size=BIN_SIZE,
            )

            resp_mask = (bc >= RESP_WIN[0]) & (bc < RESP_WIN[1])
            base_mask = (bc >= BASELINE_WIN[0]) & (bc < BASELINE_WIN[1])

            resp_fr = np.nanmean(mat[:, resp_mask], axis=1)
            base_fr = np.nanmean(mat[:, base_mask], axis=1)

            delta_fr = float(np.nanmean(resp_fr) - np.nanmean(base_fr))

            # Z-score responsiveness
            base_mean = np.nanmean(base_fr)
            base_std = np.nanstd(base_fr)
            z_resp = (np.nanmean(resp_fr) - base_mean) / max(base_std, 1e-6)

            # Peak latency
            mean_psth = np.nanmean(mat, axis=0)
            resp_psth = mean_psth[resp_mask]
            if len(resp_psth) > 0:
                peak_bin = np.argmax(np.abs(resp_psth))
                peak_latency = float(bc[resp_mask][peak_bin])
            else:
                peak_latency = np.nan

            all_records.append({
                "session_name": sname,
                "stage": stage,
                "session_idx": sidx,
                "cluster_id": cid,
                "cell_type": ct,
                "delta_fr": delta_fr,
                "z_response": float(z_resp),
                "is_responsive": abs(z_resp) >= Z_THRESH,
                "peak_latency": peak_latency,
            })
            n_units += 1

        print(f"{n_units} typed units")
        del sess; gc.collect()

    df = pd.DataFrame(all_records)
    print(f"\n  Total: {len(df)} units from {df['session_name'].nunique()} sessions")

    if len(df) == 0:
        print("  No data. Exiting.")
        return

    # Standardize cell type names
    ct_short = {"Narrow (FSI)": "FSI", "Broad (MSN/Proj)": "MSN"}
    df["ct_short"] = df["cell_type"].map(ct_short).fillna("Other")
    df = df[df["ct_short"].isin(["FSI", "MSN"])]

    if len(df) == 0:
        print("  No FSI/MSN units found. Exiting.")
        return

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Mean delta FR across sessions by cell type
    ax_a = fig.add_subplot(gs[0, 0])
    add_stage_background(ax_a, manifest)

    for ct, color in [("FSI", CELLTYPE_COLORS["Narrow (FSI)"]),
                       ("MSN", CELLTYPE_COLORS["Broad (MSN/Proj)"])]:
        sub = df[df["ct_short"] == ct]
        sess_mean = sub.groupby(["session_name", "session_idx"]).agg(
            mean_delta=("delta_fr", "mean"),
        ).reset_index().sort_values("session_idx")
        if len(sess_mean) > 0:
            ax_a.plot(sess_mean["session_idx"], sess_mean["mean_delta"],
                      "o-", color=color, markersize=5, linewidth=1.5,
                      label=ct, alpha=0.8)

    ax_a.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_a.set_xlabel("Session index")
    ax_a.set_ylabel("Mean ΔFR (Hz)")
    ax_a.set_title("A. Response magnitude by cell type")
    ax_a.legend(fontsize=8)

    # Panel B: Fraction responsive by stage and cell type
    ax_b = fig.add_subplot(gs[0, 1])
    bar_width = 0.35
    for i, stage in enumerate(STAGE_ORDER):
        for j, (ct, color) in enumerate([("FSI", CELLTYPE_COLORS["Narrow (FSI)"]),
                                          ("MSN", CELLTYPE_COLORS["Broad (MSN/Proj)"])]):
            sub = df[(df["stage"] == stage) & (df["ct_short"] == ct)]
            if len(sub) > 0:
                frac = sub["is_responsive"].mean()
                x = i + (j - 0.5) * bar_width
                ax_b.bar(x, frac, bar_width * 0.9, color=color,
                         alpha=0.7, edgecolor="white",
                         label=ct if i == 0 else "")
                ax_b.text(x, frac + 0.01, f"{sub['is_responsive'].sum()}/{len(sub)}",
                          ha="center", fontsize=7)

    ax_b.set_xticks(range(len(STAGE_ORDER)))
    ax_b.set_xticklabels(STAGE_ORDER)
    ax_b.set_ylabel("Fraction responsive (|z| >= 2)")
    ax_b.set_title("B. Fraction responsive by cell type and stage")
    ax_b.legend(fontsize=8)

    # Panel C: Response latency by cell type and stage
    ax_c = fig.add_subplot(gs[1, 0])
    responsive = df[df["is_responsive"]]
    for ct, color in [("FSI", CELLTYPE_COLORS["Narrow (FSI)"]),
                       ("MSN", CELLTYPE_COLORS["Broad (MSN/Proj)"])]:
        for i, stage in enumerate(STAGE_ORDER):
            sub = responsive[(responsive["ct_short"] == ct) & (responsive["stage"] == stage)]
            vals = sub["peak_latency"].dropna().values
            if len(vals) >= 3:
                x = i + (0.15 if ct == "FSI" else -0.15)
                bp = ax_c.boxplot([vals], positions=[x], widths=0.25,
                                  patch_artist=True, showfliers=False)
                bp["boxes"][0].set_facecolor(color)
                bp["boxes"][0].set_alpha(0.6)

    ax_c.set_xticks(range(len(STAGE_ORDER)))
    ax_c.set_xticklabels(STAGE_ORDER)
    ax_c.set_ylabel("Peak latency (s)")
    ax_c.set_title("C. Response latency by cell type")

    # Panel D: FSI/MSN responsive ratio across sessions
    ax_d = fig.add_subplot(gs[1, 1])
    add_stage_background(ax_d, manifest)

    sess_ratio = []
    for (sname, sidx), grp in df.groupby(["session_name", "session_idx"]):
        fsi_resp = grp[(grp["ct_short"] == "FSI") & grp["is_responsive"]]
        msn_resp = grp[(grp["ct_short"] == "MSN") & grp["is_responsive"]]
        n_fsi = len(fsi_resp)
        n_msn = len(msn_resp)
        stage = grp["stage"].iloc[0]
        if n_fsi + n_msn > 0:
            ratio = n_fsi / (n_fsi + n_msn)
            sess_ratio.append({
                "session_idx": sidx, "stage": stage,
                "fsi_frac": ratio, "n_fsi": n_fsi, "n_msn": n_msn,
            })

    if sess_ratio:
        sr_df = pd.DataFrame(sess_ratio).sort_values("session_idx")
        for stage in STAGE_ORDER:
            sub = sr_df[sr_df["stage"] == stage]
            if len(sub) > 0:
                ax_d.scatter(sub["session_idx"], sub["fsi_frac"],
                             c=STAGE_COLORS[stage], s=60, edgecolors="white",
                             linewidths=0.5, zorder=3, label=stage)
        ax_d.plot(sr_df["session_idx"], sr_df["fsi_frac"],
                  color="gray", alpha=0.3, linewidth=1, zorder=2)
        ax_d.axhline(0.5, color="gray", linewidth=0.5, linestyle=":")

    ax_d.set_xlabel("Session index")
    ax_d.set_ylabel("FSI fraction of responsive units")
    ax_d.set_title("D. FSI vs MSN responsive ratio")
    ax_d.legend(fontsize=8)

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # FSI vs MSN delta FR by stage
    for stage in STAGE_ORDER:
        fsi = df[(df["ct_short"] == "FSI") & (df["stage"] == stage)]["delta_fr"].dropna().values
        msn = df[(df["ct_short"] == "MSN") & (df["stage"] == stage)]["delta_fr"].dropna().values
        if len(fsi) >= 5 and len(msn) >= 5:
            u, p = mannwhitneyu(fsi, msn, alternative="two-sided")
            stats.append({
                "test": f"fsi_vs_msn_delta_fr_{stage}",
                "U": u, "p": p,
                "fsi_median": float(np.median(fsi)),
                "msn_median": float(np.median(msn)),
            })

    # FSI vs MSN responsiveness (chi-square)
    from scipy.stats import chi2_contingency
    for stage in STAGE_ORDER:
        fsi_sub = df[(df["ct_short"] == "FSI") & (df["stage"] == stage)]
        msn_sub = df[(df["ct_short"] == "MSN") & (df["stage"] == stage)]
        if len(fsi_sub) >= 5 and len(msn_sub) >= 5:
            ct_table = np.array([
                [fsi_sub["is_responsive"].sum(), len(fsi_sub) - fsi_sub["is_responsive"].sum()],
                [msn_sub["is_responsive"].sum(), len(msn_sub) - msn_sub["is_responsive"].sum()],
            ])
            if ct_table.sum() > 0 and ct_table.min(axis=0).sum() > 0:
                try:
                    chi2, p, _, _ = chi2_contingency(ct_table)
                    stats.append({
                        "test": f"responsive_fsi_vs_msn_chi2_{stage}",
                        "chi2": chi2, "p": p,
                    })
                except ValueError:
                    pass

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig22_celltype_learning", "05_longitudinal")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "05_longitudinal", "celltype_learning_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: p={row.get('p', 'N/A')}")


if __name__ == "__main__":
    main()
