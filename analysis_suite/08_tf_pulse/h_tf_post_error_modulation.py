"""Fig42: TF post-error modulation — sensory gain after error vs correct trials.

Direct neural test of the post-error attentional reorienting hypothesis
discovered in 01d/01e.  Behavioral data showed mice are *faster* and
*more accurate* after errors, ruling out speed-accuracy tradeoff and
criterion shift.  This script asks: do sensory-evoked TF pulse responses
in striatal neurons increase on trials following errors?

Approach:
  For each session, split baseline TF pulses into those from
  POST-ERROR trials vs POST-CORRECT trials, recompute PSTHs for
  TF-responsive units, and compare response amplitudes.

REQUIRES session pickles for raw spike times + trial structure.
Uses NPZ cache to identify TF-responsive units.

Panels:
  A. Population TF PSTH: post-correct vs post-error (all responsive units)
  B. Per-unit amplitude scatter (post-correct vs post-error)
  C. Distribution of error modulation index (EMI)
  D. EMI by cell type (FSI vs MSN)
  E. EMI by learning stage (trajectory across sessions)
  F. EMI vs behavioral post-error HR boost (per session)

Saves:
  figures/08_tf_pulse/fig42_tf_post_error_modulation.png
  figures/08_tf_pulse/tf_post_error_modulation_stats.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    STAGE_ORDER, STAGE_COLORS, CELLTYPE_COLORS,
    CACHE_DIR,
)
from visdetect.analysis.constants import TF_PULSE_PRE_WINDOW, TF_PULSE_POST_WINDOW
from loader import (
    load_staging_manifest, load_waveform_labels,
    load_session,
)
from plotting import setup_style, save_figure, add_stage_background

setup_style()

# ── TF cell classification (permutation-based tiered system) ───────
TF_CLASS_CSV = os.path.join(CACHE_DIR, "tf_cell_classification.csv")
RESPONSIVE_TIERS = {
    "Tier 1 (Splitter)",
    "Tier 2 (Unilateral)",
    "Tier 3 (Omni)",
}
TIER_COLORS = {
    "Tier 1 (Splitter)": "#E91E63",
    "Tier 2 (Unilateral)": "#FF9800",
    "Tier 3 (Omni)": "#2196F3",
}
TIER_SHORT = {
    "Tier 1 (Splitter)": "Splitter",
    "Tier 2 (Unilateral)": "Unilateral",
    "Tier 3 (Omni)": "Omni",
}

# ── Parameters ─────────────────────────────────────────────────────
DT = 0.001
SIGMA_MS = 17.0
PRE_WIN = TF_PULSE_PRE_WINDOW
POST_WIN = TF_PULSE_POST_WINDOW
MIN_PULSES = 10  # Minimum fast pulses per condition to include


def _vectorized_psth(spike_times, pulses, t_vec, sigma_bins):
    """Fast vectorized pulse-triggered histogram."""
    if pulses.size == 0:
        return np.zeros_like(t_vec)
    dt = t_vec[1] - t_vec[0]
    full0, full1 = t_vec[0], t_vec[-1] + dt
    all_rel = []
    for tp in pulses:
        lo = np.searchsorted(spike_times, tp + full0)
        hi = np.searchsorted(spike_times, tp + full1)
        if hi > lo:
            all_rel.append(spike_times[lo:hi] - tp)
    if not all_rel:
        return np.zeros_like(t_vec)
    all_rel = np.concatenate(all_rel)
    counts, _ = np.histogram(all_rel, bins=np.append(t_vec, t_vec[-1] + dt))
    rate = counts.astype(float) / pulses.size
    return gaussian_filter1d(rate, sigma=sigma_bins)


def _zscore(trace, t_vec, pre_win):
    pre_mask = (t_vec >= pre_win[0]) & (t_vec < pre_win[1])
    mu = np.nanmean(trace[pre_mask]) if np.any(pre_mask) else 0.0
    sd = np.nanstd(trace[pre_mask]) if np.any(pre_mask) else 0.0
    if not np.isfinite(sd) or sd <= 0:
        return trace * 0.0
    return (trace - mu) / sd


def main():
    parser = argparse.ArgumentParser(description="TF post-error modulation")
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()

    print("=" * 70)
    print("[08h] TF Post-Error Modulation  [requires session pkls]")
    print("=" * 70)

    manifest = load_staging_manifest(qc_only=True)
    print(f"  Sessions: {len(manifest)}")

    # ── Build responsive-unit set from tiered classification ────────
    if not os.path.exists(TF_CLASS_CSV):
        print(f"  ERROR: TF classification CSV not found: {TF_CLASS_CSV}")
        print("  Run 08g (g_tf_cell_classifier.py) first.")
        return

    tf_class = pd.read_csv(TF_CLASS_CSV)
    tf_resp = tf_class[tf_class["tier"].isin(RESPONSIVE_TIERS)].copy()
    responsive_set = set(
        zip(tf_resp["session_name"].astype(int), tf_resp["cluster_id"].astype(int))
    )
    # Tier lookup per unit
    tier_lookup = {
        (int(r["session_name"]), int(r["cluster_id"])): r["tier"]
        for _, r in tf_resp.iterrows()
    }
    tier_counts = tf_resp["tier"].value_counts()
    print(f"  TF-responsive units (tiered classification): {len(responsive_set)}")
    for t in sorted(RESPONSIVE_TIERS):
        print(f"    {t}: {tier_counts.get(t, 0)}")

    # ── Cell-type lookup ────────────────────────────────────────────
    ct_lookup = {}
    try:
        wf = load_waveform_labels()
        for _, r in wf.iterrows():
            ct_lookup[(int(r["session_name"]), int(r["cluster_id"]))] = r["cell_type"]
    except Exception:
        print("  Warning: could not load waveform labels")

    from visdetect.analysis.tf_pulse import TFRespPulseConfig, _safe_log2
    from visdetect.analysis.align import get_event_times_by_trial

    cfg = TFRespPulseConfig()
    t_vec = np.arange(PRE_WIN[0], POST_WIN[1], DT, dtype=float)
    sigma_bins = (SIGMA_MS / 1000.0) / DT
    post_mask = (t_vec >= POST_WIN[0]) & (t_vec < POST_WIN[1])

    sidx_lookup = {int(r["session_name"]): r["session_idx"]
                   for _, r in manifest.iterrows()}
    stage_lookup = {int(r["session_name"]): r["stage"]
                    for _, r in manifest.iterrows()}

    # ── Session loop: split pulses by post-error vs post-correct ────
    records = []          # per-unit records
    pop_correct = []      # population z-traces after correct
    pop_error = []        # population z-traces after error
    sess_summary = []     # per-session summary

    import gc
    for _, mrow in manifest.iterrows():
        sname_int = int(mrow["session_name"])
        stage = mrow["stage"]
        sidx = sidx_lookup.get(sname_int, -1)

        try:
            session = load_session(sname_int)
        except FileNotFoundError:
            print(f"    {sname_int}: pkl not found – skip")
            continue

        trials = getattr(session, "trials", []) or []
        base_on = np.array(
            get_event_times_by_trial(session, "Baseline_ON"), dtype=float
        )

        # ── Classify each trial as post-error or post-correct ───────
        outcomes = [getattr(t, "trialoutcome", None) for t in trials]
        is_error = [o in ("FA", "abort") for o in outcomes]

        # For trial i (1-based), check if trial i-1 was error
        post_error_flag = [False]  # trial 1 has no prior
        for i in range(1, len(trials)):
            post_error_flag.append(is_error[i - 1])

        # ── Collect fast pulses split by condition ──────────────────
        fast_correct = []
        fast_error = []

        for ti, t in enumerate(trials):
            bv = getattr(t, "baseline_values", None)
            if bv is None:
                continue
            arr = np.array(bv).flatten()
            if cfg.baseline_stride > 1:
                arr = arr[::cfg.baseline_stride]
            n_seen = getattr(t, "n_seen", None)
            if isinstance(n_seen, (int, np.integer)) and n_seen and n_seen > 0:
                arr = arr[:int(n_seen)]
            log2_tf = _safe_log2(arr)
            # ti is 0-based; base_on indexing matches trial enumeration
            # In session_iterator, trials are 0-based; base_on from
            # get_event_times_by_trial is 1-based (index 0 = trial 1)
            bon_idx = ti + 1
            t0 = (float(base_on[bon_idx])
                   if bon_idx < len(base_on) and np.isfinite(base_on[bon_idx])
                   else None)
            if t0 is None:
                continue

            target = fast_error if post_error_flag[ti] else fast_correct

            for bi, l2 in enumerate(log2_tf):
                if not np.isfinite(l2):
                    continue
                if l2 >= cfg.fast_thresh_log2:
                    t_pulse = t0 + bi * cfg.sample_period
                    if t_pulse >= t0 + cfg.min_after_baseline:
                        target.append(float(t_pulse))

        fast_correct = np.sort(np.array(fast_correct, dtype=float))
        fast_error = np.sort(np.array(fast_error, dtype=float))

        n_c, n_e = len(fast_correct), len(fast_error)
        if n_c < MIN_PULSES or n_e < MIN_PULSES:
            print(f"    {sname_int}: too few pulses (correct={n_c}, error={n_e}) – skip")
            continue
        print(f"    {sname_int}: fast pulses correct={n_c}, error={n_e}")

        # ── Per-unit PSTHs by condition ─────────────────────────────
        sess_amps_c, sess_amps_e = [], []

        for c in session.clusters:
            cid = int(c.cluster_id)
            if (sname_int, cid) not in responsive_set:
                continue
            st_arr = np.sort(np.asarray(c.spike_times, dtype=float).flatten())
            if st_arr.size == 0:
                continue

            z_c = _zscore(_vectorized_psth(st_arr, fast_correct, t_vec, sigma_bins),
                          t_vec, PRE_WIN)
            z_e = _zscore(_vectorized_psth(st_arr, fast_error, t_vec, sigma_bins),
                          t_vec, PRE_WIN)

            amp_c = float(np.nanmax(np.abs(z_c[post_mask])))
            amp_e = float(np.nanmax(np.abs(z_e[post_mask])))

            # Error modulation index: positive = larger after error
            denom = amp_c + amp_e
            emi = (amp_e - amp_c) / denom if denom > 0 else np.nan

            records.append({
                "session_name": sname_int,
                "cluster_id": cid,
                "stage": stage,
                "session_idx": sidx,
                "cell_type": ct_lookup.get((sname_int, cid), "Unknown"),
                "tier": tier_lookup.get((sname_int, cid), "Unknown"),
                "amp_correct": amp_c,
                "amp_error": amp_e,
                "emi": emi,
                "n_pulses_correct": n_c,
                "n_pulses_error": n_e,
            })

            pop_correct.append(z_c.copy())
            pop_error.append(z_e.copy())
            sess_amps_c.append(amp_c)
            sess_amps_e.append(amp_e)

        # Session-level summary
        if sess_amps_c:
            sess_summary.append({
                "session_name": sname_int,
                "stage": stage,
                "session_idx": sidx,
                "mean_amp_correct": np.mean(sess_amps_c),
                "mean_amp_error": np.mean(sess_amps_e),
                "mean_emi": np.mean([(e - c) / (e + c)
                                     for e, c in zip(sess_amps_e, sess_amps_c)
                                     if (e + c) > 0]),
                "n_units": len(sess_amps_c),
                "n_pulses_correct": n_c,
                "n_pulses_error": n_e,
            })

        del session
        gc.collect()

    df = pd.DataFrame(records)
    sess_df = pd.DataFrame(sess_summary)
    stat_results = []

    print(f"\n  Total units: {len(df)}, Sessions: {len(sess_df)}")
    if df.empty:
        print("  No data to plot. Exiting.")
        return

    # ══════════════════════════════════════════════════════════════════
    # Figure
    # ══════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.35)

    # ── Panel A: Population PSTH ──────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])

    if pop_correct:
        mean_c = np.nanmean(np.stack(pop_correct), axis=0)
        sem_c = np.nanstd(np.stack(pop_correct), axis=0) / np.sqrt(len(pop_correct))
        mean_e = np.nanmean(np.stack(pop_error), axis=0)
        sem_e = np.nanstd(np.stack(pop_error), axis=0) / np.sqrt(len(pop_error))

        ax_a.plot(t_vec * 1000, mean_c, color="#4CAF50", lw=2, label="Post-correct")
        ax_a.fill_between(t_vec * 1000, mean_c - sem_c, mean_c + sem_c,
                          color="#4CAF50", alpha=0.2)
        ax_a.plot(t_vec * 1000, mean_e, color="#F44336", lw=2, label="Post-error")
        ax_a.fill_between(t_vec * 1000, mean_e - sem_e, mean_e + sem_e,
                          color="#F44336", alpha=0.2)
        ax_a.axvline(0, color="grey", ls="--", lw=0.8)
        ax_a.set_xlabel("Time from TF pulse (ms)")
        ax_a.set_ylabel("z-score (population mean)")
        ax_a.legend(fontsize=8)
    ax_a.set_title("A. Population TF PSTH: post-error vs post-correct",
                    fontweight="bold")

    # ── Panel B: Per-unit amplitude scatter ───────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    valid = df.dropna(subset=["amp_correct", "amp_error"])

    if not valid.empty:
        for tier in sorted(RESPONSIVE_TIERS):
            sub = valid[valid["tier"] == tier]
            if sub.empty:
                continue
            color = TIER_COLORS.get(tier, "grey")
            ax_b.scatter(sub["amp_correct"], sub["amp_error"],
                         c=color, s=25, alpha=0.6,
                         label=TIER_SHORT.get(tier, tier),
                         edgecolors="white", linewidths=0.3)

        lim = max(valid["amp_correct"].max(), valid["amp_error"].max()) * 1.05
        ax_b.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
        ax_b.set_xlabel("|z| amplitude (post-correct)")
        ax_b.set_ylabel("|z| amplitude (post-error)")
        ax_b.legend(fontsize=7)

        # Paired Wilcoxon on amplitudes
        w, p_b = wilcoxon(valid["amp_correct"], valid["amp_error"])
        stat_results.append({
            "test": "wilcoxon_amp_correct_vs_error",
            "W": w, "p": p_b,
            "median_amp_correct": valid["amp_correct"].median(),
            "median_amp_error": valid["amp_error"].median(),
            "n": len(valid),
        })
        frac_above = (valid["amp_error"] > valid["amp_correct"]).mean()
        sig = "***" if p_b < 0.001 else "**" if p_b < 0.01 else "*" if p_b < 0.05 else "ns"
        ax_b.set_title(f"B. Per-unit amplitudes (p={p_b:.2e} {sig})",
                       fontweight="bold")
        ax_b.text(0.05, 0.95,
                  f"{frac_above:.0%} units larger\nafter error",
                  transform=ax_b.transAxes, fontsize=8, va="top")
    else:
        ax_b.set_title("B. Per-unit amplitudes", fontweight="bold")

    # ── Panel C: Distribution of EMI ──────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    emi_valid = df["emi"].dropna()

    if len(emi_valid) > 5:
        ax_c.hist(emi_valid, bins=30, color="#9C27B0", edgecolor="white",
                  alpha=0.7, density=True)
        ax_c.axvline(0, color="grey", ls="--", lw=1)
        ax_c.axvline(emi_valid.median(), color="red", ls="-", lw=2,
                     label=f"Median = {emi_valid.median():.3f}")

        # One-sample Wilcoxon: is EMI != 0?
        w_c, p_c = wilcoxon(emi_valid)
        stat_results.append({
            "test": "wilcoxon_emi_vs_zero",
            "W": w_c, "p": p_c,
            "median_emi": float(emi_valid.median()),
            "mean_emi": float(emi_valid.mean()),
            "n": len(emi_valid),
        })
        sig = "***" if p_c < 0.001 else "**" if p_c < 0.01 else "*" if p_c < 0.05 else "ns"
        ax_c.set_title(f"C. Error Modulation Index (p={p_c:.2e} {sig})",
                       fontweight="bold")
        ax_c.legend(fontsize=8)
    else:
        ax_c.set_title("C. Error Modulation Index", fontweight="bold")

    ax_c.set_xlabel("EMI  (+ = larger after error)")
    ax_c.set_ylabel("Density")

    # ── Panel D: EMI by response tier ─────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    tier_order = sorted(RESPONSIVE_TIERS)
    tier_data = {t: df[df["tier"] == t]["emi"].dropna().values
                 for t in tier_order}
    tier_data = {k: v for k, v in tier_data.items() if len(v) >= 3}

    if tier_data:
        tier_names = list(tier_data.keys())
        bp = ax_d.boxplot(
            [tier_data[t] for t in tier_names],
            labels=[f"{TIER_SHORT[t]}\n(n={len(tier_data[t])})" for t in tier_names],
            patch_artist=True, widths=0.5, showfliers=False,
            medianprops=dict(color="black", linewidth=2),
        )
        for i, t in enumerate(tier_names):
            bp["boxes"][i].set_facecolor(TIER_COLORS.get(t, "grey"))
            bp["boxes"][i].set_alpha(0.6)
            # Overlay points
            rng = np.random.default_rng(42)
            jitter = rng.uniform(-0.12, 0.12, len(tier_data[t]))
            ax_d.scatter(np.full(len(tier_data[t]), i + 1) + jitter,
                         tier_data[t],
                         c=TIER_COLORS.get(t, "grey"),
                         s=15, alpha=0.4, zorder=3)

        ax_d.axhline(0, color="grey", ls="--", lw=0.8)

        # Kruskal-Wallis across tiers (if 2+)
        kw_groups = [tier_data[t] for t in tier_names if len(tier_data[t]) >= 3]
        if len(kw_groups) >= 2:
            from scipy.stats import kruskal
            H_d, p_d = kruskal(*kw_groups)
            stat_results.append({
                "test": "kruskal_emi_by_tier",
                "H": H_d, "p": p_d,
                **{f"median_{TIER_SHORT[t]}": float(np.median(tier_data[t]))
                   for t in tier_names},
                **{f"n_{TIER_SHORT[t]}": len(tier_data[t])
                   for t in tier_names},
            })
            sig = "***" if p_d < 0.001 else "**" if p_d < 0.01 else "*" if p_d < 0.05 else "ns"
            ax_d.set_title(f"D. EMI by response tier (KW p={p_d:.2e} {sig})",
                           fontweight="bold")
        else:
            ax_d.set_title("D. EMI by response tier", fontweight="bold")
    else:
        ax_d.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                  transform=ax_d.transAxes)
        ax_d.set_title("D. EMI by response tier", fontweight="bold")

    ax_d.set_ylabel("EMI  (+ = larger after error)")

    # ── Panel E: EMI trajectory across learning ───────────────────────
    ax_e = fig.add_subplot(gs[1, 1])

    if not sess_df.empty:
        add_stage_background(ax_e, manifest)
        for stage in STAGE_ORDER:
            sub = sess_df[sess_df["stage"] == stage]
            if sub.empty:
                continue
            ax_e.scatter(sub["session_idx"], sub["mean_emi"],
                         c=STAGE_COLORS[stage], s=60, edgecolors="white",
                         linewidths=0.5, label=stage, zorder=3)
        ax_e.plot(sess_df["session_idx"], sess_df["mean_emi"],
                  c="grey", alpha=0.3, lw=1, zorder=2)
        ax_e.axhline(0, color="grey", ls="--", lw=0.8)
        ax_e.set_xlabel("Session index")
        ax_e.set_ylabel("Mean EMI (per session)")
        ax_e.legend(fontsize=8)

        rho_e, p_e = spearmanr(sess_df["session_idx"], sess_df["mean_emi"])
        stat_results.append({
            "test": "spearman_session_vs_mean_emi",
            "rho": rho_e, "p": p_e, "n": len(sess_df),
        })
        ax_e.text(0.05, 0.05, f"ρ = {rho_e:.3f}, p = {p_e:.3f}",
                  transform=ax_e.transAxes, fontsize=9, va="bottom")

    ax_e.set_title("E. EMI trajectory across learning", fontweight="bold")

    # ── Panel F: EMI vs behavioral post-error HR boost ────────────────
    ax_f = fig.add_subplot(gs[1, 2])

    # Load behavioral cache to compute per-session HR boost
    trial_cache = os.path.join(CACHE_DIR, "all_trials_behavior.csv")
    if os.path.exists(trial_cache) and not sess_df.empty:
        beh = pd.read_csv(trial_cache)
        for c in ["is_hit", "is_go"]:
            if c in beh.columns:
                beh[c] = beh[c].astype(bool)

        # Add post-error flag
        beh["prev_outcome"] = beh.groupby("session_name")["outcome"].shift(1)
        beh["post_error"] = beh["prev_outcome"].isin(["fa", "abort"])

        go_hm = beh[(beh["is_go"]) & (beh["outcome"].isin(["hit", "miss"]))]

        beh_boost = []
        for sn, grp in go_hm.groupby("session_name"):
            ac = grp[~grp["post_error"]]
            ae = grp[grp["post_error"]]
            if len(ac) >= 5 and len(ae) >= 5:
                beh_boost.append({
                    "session_name": int(sn),
                    "delta_hr": ae["is_hit"].mean() - ac["is_hit"].mean(),
                })
        beh_boost_df = pd.DataFrame(beh_boost)

        merged = sess_df.merge(beh_boost_df, on="session_name", how="inner")
        if len(merged) >= 5:
            for stage in STAGE_ORDER:
                sub = merged[merged["stage"] == stage]
                if sub.empty:
                    continue
                ax_f.scatter(sub["mean_emi"], sub["delta_hr"],
                             c=STAGE_COLORS[stage], s=60, edgecolors="white",
                             linewidths=0.5, label=stage, zorder=3)

            ax_f.axhline(0, color="grey", ls="--", lw=0.8, alpha=0.5)
            ax_f.axvline(0, color="grey", ls="--", lw=0.8, alpha=0.5)
            ax_f.set_xlabel("Mean EMI (neural, + = larger after error)")
            ax_f.set_ylabel("ΔHit rate (behavioral, + = better after error)")
            ax_f.legend(fontsize=8)

            rho_f, p_f = spearmanr(merged["mean_emi"], merged["delta_hr"])
            stat_results.append({
                "test": "spearman_emi_vs_delta_hr",
                "rho": rho_f, "p": p_f, "n": len(merged),
            })
            sig = "***" if p_f < 0.001 else "**" if p_f < 0.01 else "*" if p_f < 0.05 else "ns"
            ax_f.text(0.05, 0.95, f"ρ = {rho_f:.3f}, p = {p_f:.3f} {sig}",
                      transform=ax_f.transAxes, fontsize=9, va="top")

            # Annotate quadrants
            ax_f.text(0.02, 0.02, "Neural ↓\nBehavior ↓", fontsize=7,
                      color="grey", transform=ax_f.transAxes, va="bottom",
                      ha="left", alpha=0.6)
            ax_f.text(0.98, 0.02, "Neural ↑\nBehavior ↓", fontsize=7,
                      color="grey", transform=ax_f.transAxes, va="bottom",
                      ha="right", alpha=0.6)
            ax_f.text(0.02, 0.85, "Neural ↓\nBehavior ↑", fontsize=7,
                      color="grey", transform=ax_f.transAxes, va="top",
                      ha="left", alpha=0.6)
            ax_f.text(0.98, 0.85, "Neural ↑\nBehavior ↑", fontsize=7,
                      color="grey", transform=ax_f.transAxes, va="top",
                      ha="right", alpha=0.6)
        else:
            ax_f.text(0.5, 0.5, "Insufficient overlap", ha="center",
                      va="center", transform=ax_f.transAxes)
    else:
        ax_f.text(0.5, 0.5, "No behavioral cache", ha="center",
                  va="center", transform=ax_f.transAxes)

    ax_f.set_title("F. Neural EMI vs behavioral HR boost", fontweight="bold")

    # ══════════════════════════════════════════════════════════════════
    # Save
    # ══════════════════════════════════════════════════════════════════
    fig.suptitle("Post-Error Sensory Modulation in TF-Responsive Neurons (BG_046)",
                 fontsize=14, fontweight="bold", y=0.98)

    paths = save_figure(fig, "fig42_tf_post_error_modulation", "08_tf_pulse")
    print(f"\n  Saved figure: {paths}")

    if stat_results:
        stats_df = pd.DataFrame(stat_results)
        stats_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "figures", "08_tf_pulse", "tf_post_error_modulation_stats.csv",
        )
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        stats_df.to_csv(stats_path, index=False)
        print(f"  Saved statistics: {stats_path}")
        print("\n  Statistics summary:")
        for _, r in stats_df.iterrows():
            cols = {k: v for k, v in r.items() if pd.notna(v)}
            print(f"    {cols.get('test', '?')}: {cols}")

    # Print unit-level summary
    if not df.empty:
        print(f"\n  Unit-level EMI summary:")
        print(f"    Median EMI: {df['emi'].median():.4f}")
        print(f"    Mean EMI:   {df['emi'].mean():.4f}")
        print(f"    Units with EMI > 0: {(df['emi'] > 0).sum()}/{len(df)} "
              f"({(df['emi'] > 0).mean():.1%})")

    print("\n[08h] Done.")


if __name__ == "__main__":
    main()
