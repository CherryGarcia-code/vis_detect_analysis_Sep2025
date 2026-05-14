"""Fig02: HMM behavioral state dynamics across learning.

Produces:
  - Fig 02A: Stacked area plot of HMM state fractions across sessions
  - Fig 02B: Per-state d' trajectories across sessions
  - Fig 02C: State transition matrices for each learning stage
  - Fig 02D: Per-state psychometric curves (Expert sessions)

Saves statistics to figures/01_behavior/hmm_state_stats.csv.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr, chi2_contingency

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    STAGE_ORDER, STAGE_COLORS,
    HMM_STATE_ORDER, HMM_STATE_COLORS,
    CHANGE_SIZES, CHANGE_SIZE_LABELS, CHANGE_SIZE_POSITIONS,
)
from loader import (
    load_staging_manifest,
    load_hmm_assignments,
    load_hmm_per_session,
    load_hmm_trajectory,
)
from plotting import setup_style, save_figure, add_stage_background

setup_style()


def main():
    print("[01b] Loading HMM data...")
    manifest = load_staging_manifest(qc_only=True)
    hmm_assign = load_hmm_assignments()
    hmm_per_sess = load_hmm_per_session()
    hmm_traj = load_hmm_trajectory()

    # Build session metadata lookup
    date_to_stage = dict(zip(manifest["session_name"].astype(int), manifest["stage"]))
    date_to_idx = dict(zip(manifest["session_name"].astype(int), manifest["session_idx"]))

    # Add stage and session_idx to trajectory
    hmm_traj["stage"] = hmm_traj["session_name"].map(date_to_stage)
    hmm_traj["session_idx"] = hmm_traj["session_name"].map(date_to_idx)
    hmm_traj = hmm_traj.dropna(subset=["stage"]).sort_values("session_idx")

    # Add to per-session metrics
    hmm_per_sess["stage"] = hmm_per_sess["session_name"].map(date_to_stage)
    hmm_per_sess["session_idx"] = hmm_per_sess["session_name"].map(date_to_idx)
    hmm_per_sess = hmm_per_sess.dropna(subset=["stage"])

    # Add to assignments
    hmm_assign["stage"] = hmm_assign["session_name"].map(date_to_stage)
    hmm_assign = hmm_assign.dropna(subset=["stage"])

    print(f"  {len(hmm_traj)} sessions in trajectory, "
          f"{len(hmm_assign)} trial assignments")

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Stacked area -- state fractions
    ax_a = fig.add_subplot(gs[0, 0])
    frac_cols = [f"frac_{s}" for s in HMM_STATE_ORDER]
    available_frac = [c for c in frac_cols if c in hmm_traj.columns]

    if available_frac:
        x = hmm_traj["session_idx"].values
        bottoms = np.zeros(len(x))
        for state in HMM_STATE_ORDER:
            col = f"frac_{state}"
            if col in hmm_traj.columns:
                vals = np.nan_to_num(hmm_traj[col].values, nan=0.0)
                ax_a.fill_between(x, bottoms, bottoms + vals,
                                  color=HMM_STATE_COLORS[state],
                                  alpha=0.7, label=state)
                bottoms += vals

    add_stage_background(ax_a, manifest, alpha=0.04)
    ax_a.set_xlabel("Session index")
    ax_a.set_ylabel("Fraction of trials")
    ax_a.set_ylim(0, 1.05)
    ax_a.set_title("A. HMM state fractions across learning")
    ax_a.legend(loc="upper right", fontsize=8)

    # Panel B: Per-state d' across sessions
    ax_b = fig.add_subplot(gs[0, 1])
    add_stage_background(ax_b, manifest, alpha=0.04)

    for state in HMM_STATE_ORDER:
        sub = hmm_per_sess[hmm_per_sess["label"] == state].sort_values("session_idx")
        if len(sub) > 0 and "dprime" in sub.columns:
            ax_b.plot(sub["session_idx"], sub["dprime"],
                      "o-", color=HMM_STATE_COLORS[state],
                      markersize=4, linewidth=1.5, label=state, alpha=0.8)
    ax_b.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax_b.set_xlabel("Session index")
    ax_b.set_ylabel("d'")
    ax_b.set_title("B. Per-state d' across learning")
    ax_b.legend(loc="upper left", fontsize=8)

    # Panel C: Transition matrices per stage
    ax_c_positions = [gs[1, 0]]
    # Create 3 sub-axes for transition matrices
    gs_inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 0], wspace=0.3)

    for si, stage in enumerate(STAGE_ORDER):
        ax_t = fig.add_subplot(gs_inner[si])
        stage_trials = hmm_assign[hmm_assign["stage"] == stage].sort_values(
            ["session_name", "trial_idx"] if "trial_idx" in hmm_assign.columns
            else ["session_name"]
        )

        if "hmm_state_label" in stage_trials.columns and len(stage_trials) > 1:
            states = stage_trials["hmm_state_label"].values
            sessions_seq = stage_trials["session_name"].values
            # Compute transition matrix — skip cross-session boundaries
            trans = np.zeros((3, 3))
            for i in range(len(states) - 1):
                if sessions_seq[i] != sessions_seq[i + 1]:
                    continue  # don't count last-of-session → first-of-next as a transition
                s_from = states[i]
                s_to = states[i + 1]
                if s_from in HMM_STATE_ORDER and s_to in HMM_STATE_ORDER:
                    fi = HMM_STATE_ORDER.index(s_from)
                    ti = HMM_STATE_ORDER.index(s_to)
                    trans[fi, ti] += 1
            # Normalize rows
            row_sums = trans.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            trans_norm = trans / row_sums

            im = ax_t.imshow(trans_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax_t.set_xticks(range(3))
            ax_t.set_yticks(range(3))
            short_labels = ["Dis", "Eng", "Imp"]
            ax_t.set_xticklabels(short_labels, fontsize=7)
            ax_t.set_yticklabels(short_labels, fontsize=7)
            # Annotate
            for r in range(3):
                for c in range(3):
                    val = trans_norm[r, c]
                    color = "white" if val > 0.5 else "black"
                    ax_t.text(c, r, f"{val:.2f}", ha="center", va="center",
                              fontsize=7, color=color)
        ax_t.set_title(stage, fontsize=10, color=STAGE_COLORS[stage], fontweight="bold")
        if si == 0:
            ax_t.set_ylabel("From state")
        ax_t.set_xlabel("To state")

    # Panel D: Psychometric by HMM state (Expert sessions)
    ax_d = fig.add_subplot(gs[1, 1])
    expert_trials = hmm_assign[hmm_assign["stage"] == "Expert"]

    for state in HMM_STATE_ORDER:
        state_trials = expert_trials[expert_trials["hmm_state_label"] == state]
        if len(state_trials) == 0:
            continue

        means, sems = [], []
        for cs in CHANGE_SIZES:
            # Find go trials at this change size
            if "change_size" in state_trials.columns:
                cs_trials = state_trials[
                    (state_trials["change_size"].between(cs - 0.01, cs + 0.01))
                ]
                # Determine hits: look for outcome column
                outcome_col = None
                for col in ["trialoutcome", "outcome"]:
                    if col in cs_trials.columns:
                        outcome_col = col
                        break
                if outcome_col and len(cs_trials) >= 3:
                    n_hit = (cs_trials[outcome_col] == "hit").sum()
                    hr = n_hit / len(cs_trials)
                    se = np.sqrt(hr * (1 - hr) / len(cs_trials))
                    means.append(hr)
                    sems.append(se)
                else:
                    means.append(np.nan)
                    sems.append(0)
            else:
                means.append(np.nan)
                sems.append(0)

        ax_d.errorbar(CHANGE_SIZE_POSITIONS, means, yerr=sems,
                      fmt="o-", color=HMM_STATE_COLORS[state], label=state,
                      linewidth=2, markersize=5, capsize=3)

    ax_d.set_xticks(CHANGE_SIZE_POSITIONS)
    ax_d.set_xticklabels(CHANGE_SIZE_LABELS)
    ax_d.set_xlabel("Change size")
    ax_d.set_ylabel("Hit rate")
    ax_d.set_ylim(-0.05, 1.05)
    ax_d.set_title("D. Psychometric by HMM state (Expert)")
    ax_d.legend(loc="lower right", fontsize=8)

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # State fraction trends (drop NaN before correlating)
    for state_name in HMM_STATE_ORDER:
        col = f"frac_{state_name}"
        if col in hmm_traj.columns:
            valid = hmm_traj[["session_idx", col]].dropna()
            if len(valid) >= 3:
                rho, p = spearmanr(valid["session_idx"], valid[col])
                stats.append({"test": f"{state_name.lower()}_frac_vs_session",
                              "rho": rho, "p": p, "n": len(valid)})

    # Chi-square: state x stage independence
    if "hmm_state_label" in hmm_assign.columns:
        ct = pd.crosstab(hmm_assign["stage"], hmm_assign["hmm_state_label"])
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            chi2, p, dof, _ = chi2_contingency(ct)
            stats.append({"test": "state_x_stage_chi2", "chi2": chi2, "p": p, "dof": dof})

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig02_hmm_state_dynamics", "01_behavior")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "01_behavior", "hmm_state_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: p={row.get('p', 'N/A')}")


if __name__ == "__main__":
    main()
