"""Fig 11: State modulation — HMM state modulation of neural responses.

Condition neural responses on the behavioral state (Disengaged, Engaged,
Impulsive) to ask: does the striatal population response to visual changes
depend on the animal's cognitive state?

Produces:
  - Fig 9A: Population PSTH by HMM state (Expert sessions)
  - Fig 9B: Per-unit modulation index (Engaged vs Disengaged delta FR)
  - Fig 9C: Modulation index by cell type (FSI vs MSN)
  - Fig 9D: State modulation across learning stages

Saves: figures/02_single_unit/state_modulation_stats.csv
       cache/state_modulation.csv
"""

import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    STAGE_ORDER, STAGE_COLORS, HMM_STATE_ORDER, HMM_STATE_COLORS,
    CELLTYPE_COLORS, CACHE_DIR, DEFAULT_BIN_SIZE,
)
from loader import (
    load_staging_manifest, load_session,
    load_hmm_assignments, load_waveform_labels,
)
from utils import (
    get_good_cluster_ids, build_population_tensor, smooth_psth,
    compute_zscore_normalized,
)
from plotting import setup_style, save_figure

setup_style()

WINDOW = (-0.5, 1.0)
BIN_SIZE = DEFAULT_BIN_SIZE
BASELINE_WIN = (-0.4, -0.05)
RESP_WIN = (0.0, 0.25)
MIN_TRIALS_PER_STATE = 5


def main():
    print("[02d] State modulation of neural responses...")
    manifest = load_staging_manifest(qc_only=True)

    # Load HMM trial-level assignments
    hmm = load_hmm_assignments()
    if hmm is None or len(hmm) == 0:
        print("  No HMM assignments. Exiting.")
        return

    # Load cell-type labels
    wf_labels = load_waveform_labels()
    ct_lookup = {}
    if wf_labels is not None and len(wf_labels) > 0:
        for _, row in wf_labels.iterrows():
            ct_lookup[(int(row["session_name"]), int(row["cluster_id"]))] = row["cell_type"]

    # Per-unit state modulation metrics
    all_units = []
    state_psths = {s: [] for s in HMM_STATE_ORDER}
    bin_centers_ref = None

    for _, mrow in manifest.iterrows():
        sname = int(mrow["session_name"])
        stage = mrow["stage"]
        sidx = mrow["session_idx"]

        # Get trial-level HMM assignments for this session
        sess_hmm = hmm[hmm["session_name"] == sname]
        if len(sess_hmm) == 0:
            continue

        # Build trial-to-state lookup (trial_idx -> state)
        trial_states = {}
        if "trial_idx" in sess_hmm.columns and "hmm_state_label" in sess_hmm.columns:
            for _, hr in sess_hmm.iterrows():
                trial_states[int(hr["trial_idx"])] = hr["hmm_state_label"]

        if not trial_states:
            continue

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

        # Build per-state tensors
        trials = sess.trials
        state_tensors = {}
        for state in HMM_STATE_ORDER:
            # Get trial indices for this state with Hit or Miss outcomes
            trial_idx_list = []
            for ti, t_state in trial_states.items():
                if t_state == state and ti < len(trials):
                    outcome = getattr(trials[ti], "trialoutcome", None)
                    if outcome in ("Hit", "Miss"):
                        trial_idx_list.append(ti)

            if len(trial_idx_list) >= MIN_TRIALS_PER_STATE:
                tensor, bc, used = build_population_tensor(
                    sess, good_ids, event_name="Change_ON",
                    window=WINDOW, bin_size=BIN_SIZE,
                    trial_indices=trial_idx_list,
                )
                if tensor.shape[0] >= MIN_TRIALS_PER_STATE:
                    state_tensors[state] = compute_zscore_normalized(tensor, bc, BASELINE_WIN)
                    bin_centers_ref = bc

        if len(state_tensors) < 2:
            print("insufficient state coverage")
            del sess
            gc.collect()
            continue

        bc = bin_centers_ref
        resp_mask = (bc >= RESP_WIN[0]) & (bc < RESP_WIN[1])

        n_added = 0
        for u_i, cid in enumerate(good_ids):
            if u_i >= min(t.shape[2] for t in state_tensors.values()):
                break

            state_resp = {}
            for state, tensor in state_tensors.items():
                unit_resp = np.nanmean(tensor[:, resp_mask, u_i])
                state_resp[state] = float(unit_resp)

            # Modulation index: (Engaged - Disengaged) / (|Engaged| + |Disengaged|)
            eng = state_resp.get("Engaged", np.nan)
            dis = state_resp.get("Disengaged", np.nan)
            if np.isfinite(eng) and np.isfinite(dis):
                denom = abs(eng) + abs(dis)
                mi = (eng - dis) / denom if denom > 0 else 0
            else:
                mi = np.nan

            ct = ct_lookup.get((sname, cid), "Unknown")

            all_units.append({
                "session_name": sname,
                "cluster_id": cid,
                "stage": stage,
                "session_idx": sidx,
                "cell_type": ct,
                "modulation_index": mi,
                **{f"resp_{s}": state_resp.get(s, np.nan) for s in HMM_STATE_ORDER},
            })
            n_added += 1

        # Collect population PSTHs for Expert sessions
        if stage == "Expert":
            for state, tensor in state_tensors.items():
                pop_mean = np.nanmean(np.nanmean(tensor, axis=0), axis=1)  # mean over units
                pop_smooth = smooth_psth(pop_mean, BIN_SIZE, sigma_ms=15.0)
                state_psths[state].append(pop_smooth)

        print(f"{n_added} units")
        del sess
        gc.collect()

    df = pd.DataFrame(all_units)
    print(f"\n  Total: {len(df)} units with state modulation data")

    if len(df) == 0:
        print("  No data. Exiting.")
        return

    # Cache
    cache_path = os.path.join(CACHE_DIR, "state_modulation.csv")
    df.to_csv(cache_path, index=False)

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
    bc = bin_centers_ref

    # Panel A: Population PSTH by HMM state (Expert)
    ax_a = fig.add_subplot(gs[0, 0])
    for state in HMM_STATE_ORDER:
        psth_list = state_psths[state]
        if len(psth_list) > 0:
            mat = np.array(psth_list)
            mean_p = np.nanmean(mat, axis=0)
            sem_p = np.nanstd(mat, axis=0) / np.sqrt(len(psth_list))
            ax_a.plot(bc, mean_p, color=HMM_STATE_COLORS[state], linewidth=2,
                      label=f"{state} (n={len(psth_list)} sess)")
            ax_a.fill_between(bc, mean_p - sem_p, mean_p + sem_p,
                              color=HMM_STATE_COLORS[state], alpha=0.2)

    ax_a.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_a.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_a.set_xlabel("Time from Change_ON (s)")
    ax_a.set_ylabel("Population z-score")
    ax_a.set_title("A. Population response by HMM state (Expert)")
    ax_a.legend(fontsize=8)

    # Panel B: Modulation index distribution
    ax_b = fig.add_subplot(gs[0, 1])
    mi_vals = df["modulation_index"].dropna().values
    if len(mi_vals) > 0:
        ax_b.hist(mi_vals, bins=40, color="#78909C", edgecolor="white",
                  linewidth=0.5, alpha=0.8)
        ax_b.axvline(0, color="k", linestyle="--", linewidth=0.8)
        ax_b.axvline(np.median(mi_vals), color="#E53935", linestyle="-",
                     linewidth=1.5, label=f"Median={np.median(mi_vals):.3f}")
    ax_b.set_xlabel("Modulation index (Eng - Dis)")
    ax_b.set_ylabel("Number of units")
    ax_b.set_title(f"B. State modulation index (n={len(mi_vals)} units)")
    ax_b.legend(fontsize=8)

    # Panel C: MI by cell type
    ax_c = fig.add_subplot(gs[1, 0])
    ct_types = ["Narrow (FSI)", "Broad (MSN/Proj)"]
    ct_short = ["FSI", "MSN"]
    ct_data = []
    ct_colors = []
    ct_positions = []

    for i, ct in enumerate(ct_types):
        vals = df[df["cell_type"] == ct]["modulation_index"].dropna().values
        if len(vals) >= 2:
            ct_positions.append(i)
            ct_data.append(vals)
            ct_colors.append(CELLTYPE_COLORS[ct])

    if ct_data:
        bp = ax_c.boxplot(ct_data, positions=ct_positions, widths=0.5,
                          patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], ct_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for pos, vals, color in zip(ct_positions, ct_data, ct_colors):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax_c.scatter(pos + jitter, vals, c=color, s=20,
                         edgecolors="white", linewidths=0.3, zorder=3, alpha=0.5)

    ax_c.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_c.set_xticks(range(len(ct_types)))
    ax_c.set_xticklabels(ct_short)
    ax_c.set_ylabel("Modulation index")
    ax_c.set_title("C. State modulation by cell type")

    # Panel D: Modulation index by stage
    ax_d = fig.add_subplot(gs[1, 1])
    stage_data = []
    stage_positions = []
    stage_box_colors = []

    for i, stage in enumerate(STAGE_ORDER):
        vals = df[df["stage"] == stage]["modulation_index"].dropna().values
        if len(vals) >= 2:
            stage_positions.append(i)
            stage_data.append(vals)
            stage_box_colors.append(STAGE_COLORS[stage])

    if stage_data:
        bp = ax_d.boxplot(stage_data, positions=stage_positions, widths=0.5,
                          patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], stage_box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for pos, vals, color in zip(stage_positions, stage_data, stage_box_colors):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax_d.scatter(pos + jitter, vals, c=color, s=20,
                         edgecolors="white", linewidths=0.3, zorder=3, alpha=0.5)

    ax_d.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_d.set_xticks(range(len(STAGE_ORDER)))
    ax_d.set_xticklabels(STAGE_ORDER)
    ax_d.set_ylabel("Modulation index")
    ax_d.set_title("D. State modulation by learning stage")

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # Is MI significantly != 0?
    if len(mi_vals) >= 10:
        try:
            w, p = wilcoxon(mi_vals)
            stats.append({"test": "mi_vs_zero_wilcoxon", "W": w, "p": p,
                          "median_mi": float(np.median(mi_vals))})
        except ValueError:
            pass

    # MI: FSI vs MSN
    fsi_mi = df[df["cell_type"] == "Narrow (FSI)"]["modulation_index"].dropna().values
    msn_mi = df[df["cell_type"] == "Broad (MSN/Proj)"]["modulation_index"].dropna().values
    if len(fsi_mi) >= 2 and len(msn_mi) >= 2:
        u, p = mannwhitneyu(fsi_mi, msn_mi, alternative="two-sided")
        stats.append({"test": "mi_fsi_vs_msn_mwu", "U": u, "p": p})

    # MI by stage
    from scipy.stats import kruskal as kruskal_test
    stage_groups = [df[df["stage"] == s]["modulation_index"].dropna().values
                    for s in STAGE_ORDER]
    stage_groups = [g for g in stage_groups if len(g) >= 2 and np.std(g) > 0]
    if len(stage_groups) >= 2:
        try:
            h, p = kruskal_test(*stage_groups)
            stats.append({"test": "mi_kruskal_by_stage", "H": h, "p": p})
        except ValueError:
            pass

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig11_state_modulation", "02_single_unit")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "02_single_unit", "state_modulation_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: p={row.get('p', 'N/A')}")


if __name__ == "__main__":
    main()
