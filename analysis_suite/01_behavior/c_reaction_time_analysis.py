"""Fig03: Reaction time analysis across learning, change size, and HMM state.

Reaction times are computed differently per outcome:
  - Hit:   rt["RT"]    – time from Change_ON   to first lick
  - FA:    rt["FA"]    – time from Baseline_ON  to first lick (early response)
  - abort: rt["abort"] – time from Baseline_ON  to first lick (very early)
  - Miss:  rt["Miss"]  – always 2.155 s (response window limit, NOT a real RT)
  - Ref:   rt["Ref"]   – time from Baseline_ON  to lick on catch trial

Produces:
  - Fig 03A: Hit RT distributions by learning stage (violin)
  - Fig 03B: Hit RT vs change size (speed-accuracy, per stage)
  - Fig 03C: FA & abort RT distributions by learning stage
  - Fig 03D: Hit RT by HMM state (Expert sessions)
  - Fig 03E: Median Hit RT trajectory across sessions
  - Fig 03F: FA RT (from baseline) trajectory across sessions

Saves: figures/01_behavior/reaction_time_stats.csv
"""

import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr, mannwhitneyu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    STAGE_ORDER, STAGE_COLORS, HMM_STATE_ORDER, HMM_STATE_COLORS,
    CHANGE_SIZES, CHANGE_SIZE_LABELS, CHANGE_SIZE_POSITIONS, FA_RT_SPLIT,
)
from loader import load_staging_manifest, load_session, load_hmm_assignments
from plotting import setup_style, save_figure, add_stage_background

setup_style()


def _extract_rt(trial):
    """Extract RT value and reference event for a single trial.

    Returns
    -------
    rt_value : float or nan
        Reaction time in seconds.
    ref_event : str
        'Change_ON' for Hit/Miss, 'Baseline_ON' for FA/abort/Ref.
    """
    outcome = getattr(trial, "trialoutcome", None)
    rt_dict = getattr(trial, "reactiontimes", None)
    if not isinstance(rt_dict, dict):
        return float("nan"), "unknown"

    # Map outcome to the dict key and reference event
    RT_KEY_MAP = {
        "Hit": ("RT", "Change_ON"),
        "FA": ("FA", "Baseline_ON"),
        "abort": ("abort", "Baseline_ON"),
        "Miss": ("Miss", "Change_ON"),    # constant 2.155s, will exclude
        "Ref": ("Ref", "Baseline_ON"),
    }

    if outcome not in RT_KEY_MAP:
        return float("nan"), "unknown"

    key, ref = RT_KEY_MAP[outcome]
    try:
        val = float(rt_dict.get(key, float("nan")))
    except (TypeError, ValueError):
        val = float("nan")
    return val, ref


def main():
    print("[01c] Reaction time analysis...")
    manifest = load_staging_manifest(qc_only=True)
    hmm = load_hmm_assignments()

    # ── Collect RTs for ALL outcome types ─────────────────────────────
    all_rt = []
    for _, row in manifest.iterrows():
        sname = int(row["session_name"])
        stage = row["stage"]
        sidx = row["session_idx"]

        try:
            sess = load_session(sname)
        except FileNotFoundError:
            continue

        for i, t in enumerate(sess.trials):
            outcome = getattr(t, "trialoutcome", None)
            if outcome not in ("Hit", "FA", "abort"):
                continue  # skip Miss (constant 2.155s) and Ref (catch)
            rt_val, ref_event = _extract_rt(t)
            if not np.isfinite(rt_val) or rt_val <= 0 or rt_val > 20.0:
                continue
            cs = getattr(t, "change_size", None)
            all_rt.append({
                "session_name": sname,
                "stage": stage,
                "session_idx": sidx,
                "trial_idx": i,
                "outcome": outcome,
                "rt": float(rt_val),
                "ref_event": ref_event,
                "change_size": cs,
            })
        del sess
        gc.collect()

    df = pd.DataFrame(all_rt)

    # Subsets
    hits = df[df["outcome"] == "Hit"].copy()
    fas = df[df["outcome"] == "FA"].copy()
    aborts = df[df["outcome"] == "abort"].copy()
    early_responses = df[df["outcome"].isin(["FA", "abort"])].copy()

    print(f"  {len(hits)} Hit trials (RT from Change_ON)")
    print(f"  {len(fas)} FA trials  (RT from Baseline_ON)")
    print(f"  {len(aborts)} abort trials (RT from Baseline_ON)")

    if len(hits) == 0:
        print("  No Hit data. Exiting.")
        return

    # Merge HMM state
    if "trial_idx" in hmm.columns:
        hmm_sub = hmm[["session_name", "trial_idx", "hmm_state_label"]].copy()
        hits = hits.merge(hmm_sub, on=["session_name", "trial_idx"], how="left")
        fas = fas.merge(hmm_sub, on=["session_name", "trial_idx"], how="left")
        early_responses = early_responses.merge(
            hmm_sub, on=["session_name", "trial_idx"], how="left"
        )
    else:
        hits["hmm_state_label"] = np.nan
        fas["hmm_state_label"] = np.nan
        early_responses["hmm_state_label"] = np.nan

    # ── Create figure (3x2 layout) ───────────────────────────────────
    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(3, 2, hspace=0.40, wspace=0.30)

    # ── Panel A: Hit RT by stage (violin) ─────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    stage_data, stage_labels = [], []
    for stage in STAGE_ORDER:
        vals = hits[hits["stage"] == stage]["rt"].dropna().values
        if len(vals) > 0:
            stage_data.append(vals)
            stage_labels.append(f"{stage}\n(n={len(vals)})")
        else:
            stage_data.append(np.array([0.5]))  # placeholder
            stage_labels.append(f"{stage}\n(n=0)")

    parts = ax_a.violinplot(stage_data, positions=range(len(STAGE_ORDER)),
                            showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(list(STAGE_COLORS.values())[i])
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")
    ax_a.set_xticks(range(len(STAGE_ORDER)))
    ax_a.set_xticklabels(stage_labels)
    ax_a.set_ylabel("Reaction time from Change_ON (s)")
    ax_a.set_title("A. Hit RT by learning stage")
    ax_a.set_ylim(0, 2.5)

    # ── Panel B: Hit RT vs change size ────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    for stage in STAGE_ORDER:
        sub = hits[hits["stage"] == stage]
        medians, sems = [], []
        for cs in CHANGE_SIZES:
            cs_mask = sub["change_size"].between(cs - 0.01, cs + 0.01)
            cs_sub = sub[cs_mask]
            if len(cs_sub) >= 3:
                medians.append(cs_sub["rt"].median())
                sems.append(cs_sub["rt"].sem())
            else:
                medians.append(np.nan)
                sems.append(0)
        ax_b.errorbar(CHANGE_SIZE_POSITIONS, medians, yerr=sems,
                      fmt="o-", color=STAGE_COLORS[stage], label=stage,
                      linewidth=2, markersize=5, capsize=3)
    ax_b.set_xticks(CHANGE_SIZE_POSITIONS)
    ax_b.set_xticklabels(CHANGE_SIZE_LABELS)
    ax_b.set_xlabel("Change size (TF ratio)")
    ax_b.set_ylabel("Median Hit RT (s)")
    ax_b.set_title("B. Hit RT vs change magnitude")
    ax_b.legend(fontsize=8)

    # ── Panel C: FA & abort RT by stage ───────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    # Side-by-side violins: FA (left) and abort (right) per stage
    positions_fa = []
    positions_ab = []
    fa_data_list, ab_data_list = [], []
    x_labels, x_positions = [], []
    offset = 0
    for si, stage in enumerate(STAGE_ORDER):
        fa_vals = fas[fas["stage"] == stage]["rt"].dropna().values
        ab_vals = aborts[aborts["stage"] == stage]["rt"].dropna().values
        fa_data_list.append(fa_vals if len(fa_vals) > 0 else np.array([0.5]))
        ab_data_list.append(ab_vals if len(ab_vals) > 0 else np.array([0.5]))
        pos_fa = si * 3
        pos_ab = si * 3 + 1
        positions_fa.append(pos_fa)
        positions_ab.append(pos_ab)
        x_labels.append(f"{stage}\nFA={len(fa_vals)}\nAbort={len(ab_vals)}")
        x_positions.append(si * 3 + 0.5)

    if any(len(d) > 1 for d in fa_data_list):
        parts_fa = ax_c.violinplot(fa_data_list, positions=positions_fa,
                                   showmedians=True, widths=0.8)
        for pc in parts_fa["bodies"]:
            pc.set_facecolor("#FF9800")
            pc.set_alpha(0.6)
        parts_fa["cmedians"].set_color("black")

    if any(len(d) > 1 for d in ab_data_list):
        parts_ab = ax_c.violinplot(ab_data_list, positions=positions_ab,
                                   showmedians=True, widths=0.8)
        for pc in parts_ab["bodies"]:
            pc.set_facecolor("#9C27B0")
            pc.set_alpha(0.6)
        parts_ab["cmedians"].set_color("black")

    ax_c.set_xticks(x_positions)
    ax_c.set_xticklabels(x_labels, fontsize=8)
    ax_c.set_ylabel("RT from Baseline_ON (s)")
    ax_c.set_title("C. Early response RT by stage (FA & abort)")
    # Add legend manually
    from matplotlib.patches import Patch
    ax_c.legend(handles=[
        Patch(facecolor="#FF9800", alpha=0.6, label="FA"),
        Patch(facecolor="#9C27B0", alpha=0.6, label="Abort"),
    ], fontsize=8)
    ax_c.axhline(FA_RT_SPLIT, color="red", linestyle="--", alpha=0.5,
                 label=f"FA_RT_SPLIT={FA_RT_SPLIT}s")

    # ── Panel D: Hit RT by HMM state (Expert) ────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    expert_hits = hits[hits["stage"] == "Expert"]
    state_data, state_labels, state_colors = [], [], []
    for state in HMM_STATE_ORDER:
        vals = expert_hits[expert_hits["hmm_state_label"] == state]["rt"].dropna().values
        if len(vals) >= 5:
            state_data.append(vals)
            state_labels.append(f"{state}\n(n={len(vals)})")
            state_colors.append(HMM_STATE_COLORS[state])

    if state_data:
        parts = ax_d.violinplot(state_data, positions=range(len(state_data)),
                                showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(state_colors[i])
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("black")
        ax_d.set_xticks(range(len(state_data)))
        ax_d.set_xticklabels(state_labels)
    ax_d.set_ylabel("Hit RT from Change_ON (s)")
    ax_d.set_title("D. Hit RT by HMM state (Expert)")
    ax_d.set_ylim(0, 2.5)

    # ── Panel E: Hit median RT trajectory ─────────────────────────────
    ax_e = fig.add_subplot(gs[2, 0])
    add_stage_background(ax_e, manifest)

    sess_hit_rt = hits.groupby(["session_name", "session_idx", "stage"]).agg(
        median_rt=("rt", "median"),
        n_trials=("rt", "count"),
    ).reset_index().sort_values("session_idx")

    for stage in STAGE_ORDER:
        sub = sess_hit_rt[sess_hit_rt["stage"] == stage]
        if len(sub) > 0:
            ax_e.scatter(sub["session_idx"], sub["median_rt"],
                         c=STAGE_COLORS[stage], s=60, edgecolors="white",
                         linewidths=0.5, zorder=3, label=stage)
    if len(sess_hit_rt) > 0:
        ax_e.plot(sess_hit_rt["session_idx"], sess_hit_rt["median_rt"],
                  color="gray", alpha=0.3, linewidth=1, zorder=2)
    ax_e.set_xlabel("Session index")
    ax_e.set_ylabel("Median Hit RT (s)")
    ax_e.set_title("E. Hit RT trajectory across learning")
    ax_e.legend(fontsize=8)

    # ── Panel F: FA RT trajectory ─────────────────────────────────────
    ax_f = fig.add_subplot(gs[2, 1])
    add_stage_background(ax_f, manifest)

    if len(fas) > 0:
        sess_fa_rt = fas.groupby(["session_name", "session_idx", "stage"]).agg(
            median_rt=("rt", "median"),
            n_fa=("rt", "count"),
        ).reset_index().sort_values("session_idx")

        for stage in STAGE_ORDER:
            sub = sess_fa_rt[sess_fa_rt["stage"] == stage]
            if len(sub) > 0:
                ax_f.scatter(sub["session_idx"], sub["median_rt"],
                             c=STAGE_COLORS[stage], s=60, edgecolors="white",
                             linewidths=0.5, zorder=3, label=stage)
        if len(sess_fa_rt) > 0:
            ax_f.plot(sess_fa_rt["session_idx"], sess_fa_rt["median_rt"],
                      color="gray", alpha=0.3, linewidth=1, zorder=2)
        # Secondary axis: FA count
        ax_f2 = ax_f.twinx()
        ax_f2.bar(sess_fa_rt["session_idx"], sess_fa_rt["n_fa"],
                  alpha=0.15, color="#FF9800", width=0.8)
        ax_f2.set_ylabel("FA count per session", color="#FF9800", fontsize=9)
        ax_f2.tick_params(axis="y", labelcolor="#FF9800")
    ax_f.set_xlabel("Session index")
    ax_f.set_ylabel("Median FA RT from Baseline_ON (s)")
    ax_f.set_title("F. FA RT trajectory across learning")
    ax_f.legend(fontsize=8)

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # Hit RT by stage
    stage_groups = [hits[hits["stage"] == s]["rt"].dropna().values for s in STAGE_ORDER]
    stage_groups_valid = [g for g in stage_groups if len(g) >= 2]
    if len(stage_groups_valid) >= 2:
        try:
            h, p = kruskal(*stage_groups_valid)
            stats.append({"test": "hit_rt_kruskal_by_stage", "H": h, "p": p,
                          "note": "RT from Change_ON"})
        except ValueError:
            pass

    # Hit RT trend across sessions
    if len(sess_hit_rt) >= 3:
        rho, p = spearmanr(sess_hit_rt["session_idx"], sess_hit_rt["median_rt"])
        stats.append({"test": "hit_rt_vs_session_spearman", "rho": rho, "p": p})

    # Hit RT by HMM state (Expert)
    state_groups = [expert_hits[expert_hits["hmm_state_label"] == s]["rt"].dropna().values
                    for s in HMM_STATE_ORDER]
    state_groups_valid = [g for g in state_groups if len(g) >= 2]
    if len(state_groups_valid) >= 2:
        try:
            h, p = kruskal(*state_groups_valid)
            stats.append({"test": "hit_rt_kruskal_by_state_expert", "H": h, "p": p,
                          "note": "RT from Change_ON, Expert only"})
        except ValueError:
            pass

    # FA RT by stage
    fa_stage_groups = [fas[fas["stage"] == s]["rt"].dropna().values for s in STAGE_ORDER]
    fa_stage_valid = [g for g in fa_stage_groups if len(g) >= 2]
    if len(fa_stage_valid) >= 2:
        try:
            h, p = kruskal(*fa_stage_valid)
            stats.append({"test": "fa_rt_kruskal_by_stage", "H": h, "p": p,
                          "note": "RT from Baseline_ON"})
        except ValueError:
            pass

    # FA RT trend
    if len(fas) > 0:
        sess_fa_rt2 = fas.groupby(["session_idx"]).agg(median_rt=("rt", "median")).reset_index()
        if len(sess_fa_rt2) >= 3:
            rho, p = spearmanr(sess_fa_rt2["session_idx"], sess_fa_rt2["median_rt"])
            stats.append({"test": "fa_rt_vs_session_spearman", "rho": rho, "p": p})

    # Hit vs FA RT comparison (Expert)
    expert_fa = fas[fas["stage"] == "Expert"]["rt"].dropna().values
    expert_hit = expert_hits["rt"].dropna().values
    if len(expert_fa) >= 5 and len(expert_hit) >= 5:
        u, p = mannwhitneyu(expert_hit, expert_fa, alternative="two-sided")
        stats.append({"test": "expert_hit_vs_fa_rt_mannwhitney", "U": u, "p": p,
                      "hit_median": float(np.median(expert_hit)),
                      "fa_median": float(np.median(expert_fa)),
                      "note": "Hit from Change_ON vs FA from Baseline_ON"})

    # Overall summaries
    stats.append({
        "test": "summary_hit_rt", "median": float(hits["rt"].median()),
        "n_trials": len(hits), "note": "from Change_ON",
    })
    if len(fas) > 0:
        stats.append({
            "test": "summary_fa_rt", "median": float(fas["rt"].median()),
            "n_trials": len(fas), "note": "from Baseline_ON",
        })
    if len(aborts) > 0:
        stats.append({
            "test": "summary_abort_rt", "median": float(aborts["rt"].median()),
            "n_trials": len(aborts), "note": "from Baseline_ON",
        })

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig03_reaction_times", "01_behavior")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "01_behavior", "reaction_time_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row in stats_df.iterrows():
        test = row["test"]
        p = row.get("p", "")
        note = row.get("note", "")
        print(f"    {test}: p={p}  {note}")


if __name__ == "__main__":
    main()
