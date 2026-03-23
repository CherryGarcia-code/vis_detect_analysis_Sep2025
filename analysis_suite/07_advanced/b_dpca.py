"""Fig28: Demixed PCA (dPCA) — variance decomposition by marginalization.

Applies demixed PCA to decompose population activity into components
driven by stimulus (change size), decision (hit/miss), time, and
their interactions.

Uses a manual implementation of dPCA since the dPCA package may not
be installed. The approach follows Kobak et al., 2016 eLife.

Produces:
  - Fig 28A: Variance explained by each marginalization
  - Fig 28B: Top dPC for stimulus marginalization (time course)
  - Fig 28C: Top dPC for decision marginalization (time course)
  - Fig 28D: Variance partition pie charts by stage

Saves: figures/07_advanced/dpca_stats.csv
"""

import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA

from config import (
    STAGE_ORDER, STAGE_COLORS, OUTCOME_COLORS,
    SMALL_CHANGE_SIZES, BIG_CHANGE_SIZES,
    CACHE_DIR, DEFAULT_BIN_SIZE,
)
from loader import load_staging_manifest, load_session
from utils import (
    get_good_cluster_ids, build_population_tensor,
    compute_zscore_normalized,
)
from plotting import setup_style, save_figure

setup_style()

WINDOW = (-0.5, 1.0)
BIN_SIZE = DEFAULT_BIN_SIZE
BASELINE_WIN = (-0.5, -0.05)
MIN_UNITS = 10
MIN_TRIALS_PER_COND = 5


def compute_marginalizations(X_full, labels_outcome, labels_size, n_bins):
    """Compute marginalized data matrices for dPCA.

    X_full: (n_conditions, n_bins, n_units) -- condition-averaged data
    labels_outcome: condition -> 0/1 (miss/hit)
    labels_size: condition -> 0/1 (small/big)

    Returns dict of marginalization name -> covariance matrix
    """
    n_cond, T, N = X_full.shape

    # Grand mean
    grand_mean = np.mean(X_full, axis=0)  # (T, N)

    # Marginalizations
    # Time: grand mean pattern over time
    X_time = np.tile(grand_mean, (n_cond, 1, 1))

    # Decision: mean within each outcome level
    X_decision = np.zeros_like(X_full)
    for oc in [0, 1]:
        mask = [i for i, o in enumerate(labels_outcome) if o == oc]
        if mask:
            oc_mean = np.mean(X_full[mask], axis=0)
            for i in mask:
                X_decision[i] = oc_mean - grand_mean

    # Stimulus: mean within each size level
    X_stimulus = np.zeros_like(X_full)
    for sz in [0, 1]:
        mask = [i for i, s in enumerate(labels_size) if s == sz]
        if mask:
            sz_mean = np.mean(X_full[mask], axis=0)
            for i in mask:
                X_stimulus[i] = sz_mean - grand_mean

    # Interaction: residual
    X_interaction = X_full - X_time - X_decision - X_stimulus

    marginals = {
        "time": X_time,
        "decision": X_decision,
        "stimulus": X_stimulus,
        "interaction": X_interaction,
    }

    # Compute variance for each
    total_var = np.sum(np.var(X_full.reshape(-1, N), axis=0))
    var_dict = {}
    for name, Xm in marginals.items():
        var_dict[name] = np.sum(np.var(Xm.reshape(-1, N), axis=0))

    return marginals, var_dict, total_var


def run_dpca_session(sess, sname, stage, sidx):
    """Run simplified dPCA for a session."""
    good_ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
    if len(good_ids) < MIN_UNITS:
        return None

    # Build separate tensors for each condition
    trials = sess.trials
    from visdetect.analysis.align import get_event_times_by_trial
    event_times = get_event_times_by_trial(sess, "Change_ON")

    # Group trials by outcome x size
    conditions = {}  # (outcome, size) -> list of trial indices
    for i, t in enumerate(trials):
        outcome = getattr(t, "trialoutcome", None)
        if outcome not in ("Hit", "Miss"):
            continue
        if i >= len(event_times) or not np.isfinite(event_times[i]):
            continue
        cs = getattr(t, "change_size", None)
        if cs is None:
            continue

        oc = 1 if outcome == "Hit" else 0
        sz = 1 if any(abs(cs - s) < 0.01 for s in BIG_CHANGE_SIZES) else 0

        key = (oc, sz)
        if key not in conditions:
            conditions[key] = []
        conditions[key].append(i)

    # Need all 4 conditions with enough trials
    required = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for key in required:
        if key not in conditions or len(conditions[key]) < MIN_TRIALS_PER_COND:
            return None

    # Build population tensor and compute condition averages
    all_trials = []
    for key in required:
        all_trials.extend(conditions[key])

    tensor, bc, used = build_population_tensor(
        sess, good_ids, event_name="Change_ON",
        window=WINDOW, bin_size=BIN_SIZE,
        trial_indices=all_trials,
    )

    if tensor.shape[2] < MIN_UNITS:
        return None

    # Z-score
    z_tensor = compute_zscore_normalized(tensor, bc, BASELINE_WIN)

    # Map used indices back to conditions
    used_set = set(used)
    cond_means = []
    labels_outcome = []
    labels_size = []

    for oc, sz in required:
        cond_trials = [idx for idx in conditions[(oc, sz)] if idx in used_set]
        if len(cond_trials) < MIN_TRIALS_PER_COND:
            return None

        # Find positions in used
        used_list = list(used)
        positions = [used_list.index(idx) for idx in cond_trials if idx in used_list]
        if len(positions) < MIN_TRIALS_PER_COND:
            return None

        cond_mean = np.nanmean(z_tensor[positions], axis=0)  # (n_bins, n_units)
        cond_means.append(cond_mean)
        labels_outcome.append(oc)
        labels_size.append(sz)

    X_full = np.array(cond_means)  # (4, n_bins, n_units)
    n_bins = X_full.shape[1]

    # Compute marginalizations
    marginals, var_dict, total_var = compute_marginalizations(
        X_full, labels_outcome, labels_size, n_bins
    )

    # Extract top PC for decision and stimulus marginalizations
    decision_pcs = {}
    stimulus_pcs = {}

    for name, Xm in [("decision", marginals["decision"]),
                       ("stimulus", marginals["stimulus"])]:
        flat = Xm.reshape(-1, Xm.shape[-1])
        valid = ~np.isnan(flat).any(axis=1)
        if valid.sum() > 2:
            pca = PCA(n_components=1)
            pca.fit(flat[valid])
            # Project condition means
            projections = {}
            for ci, (oc, sz) in enumerate(required):
                proj = pca.transform(Xm[ci].reshape(1, -1) if Xm[ci].ndim == 1
                                     else Xm[ci])
                label = f"{'Hit' if oc else 'Miss'}-{'Big' if sz else 'Small'}"
                projections[label] = proj[:, 0] if proj.ndim == 2 else proj

            if name == "decision":
                decision_pcs = projections
            else:
                stimulus_pcs = projections

    return {
        "var_dict": var_dict,
        "total_var": total_var,
        "decision_pcs": decision_pcs,
        "stimulus_pcs": stimulus_pcs,
        "bin_centers": bc,
        "stage": stage,
        "session_idx": sidx,
        "n_units": len(good_ids),
    }


def main():
    print("[07b] Demixed PCA analysis...")
    manifest = load_staging_manifest(qc_only=True)

    results = {}
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

        r = run_dpca_session(sess, sname, stage, sidx)
        if r is not None:
            results[sname] = r
            vd = r["var_dict"]
            print(f"time={vd.get('time', 0):.1f}, dec={vd.get('decision', 0):.1f}, "
                  f"stim={vd.get('stimulus', 0):.1f}")
        else:
            print("insufficient data")

        del sess; gc.collect()

    print(f"\n  dPCA computed for {len(results)} sessions")

    if not results:
        print("  No data. Exiting.")
        return

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    marg_names = ["time", "decision", "stimulus", "interaction"]
    marg_colors = ["#7f8c8d", "#2ecc71", "#e74c3c", "#9b59b6"]

    # Panel A: Variance explained by marginalization
    ax_a = fig.add_subplot(gs[0, 0])
    all_fracs = {m: [] for m in marg_names}
    for r in results.values():
        tv = r["total_var"]
        if tv > 0:
            for m in marg_names:
                all_fracs[m].append(r["var_dict"].get(m, 0) / tv)

    if all_fracs["time"]:
        means = [np.mean(all_fracs[m]) for m in marg_names]
        sems = [np.std(all_fracs[m]) / np.sqrt(len(all_fracs[m])) for m in marg_names]
        ax_a.bar(range(len(marg_names)), means, yerr=sems, color=marg_colors,
                 edgecolor="white", capsize=3)
        ax_a.set_xticks(range(len(marg_names)))
        ax_a.set_xticklabels(["Time", "Decision", "Stimulus", "Interaction"])
    ax_a.set_ylabel("Fraction of variance")
    ax_a.set_title(f"A. Variance decomposition (n={len(results)} sessions)")

    # Panel B: Top decision dPC (Expert sessions)
    ax_b = fig.add_subplot(gs[0, 1])
    expert = {k: v for k, v in results.items() if v["stage"] == "Expert"}
    if expert:
        best = max(expert.keys(), key=lambda k: expert[k]["n_units"])
        r = expert[best]
        bc = r["bin_centers"]
        for label, proj in r["decision_pcs"].items():
            color = OUTCOME_COLORS["Hit"] if "Hit" in label else OUTCOME_COLORS["Miss"]
            ls = "-" if "Big" in label else "--"
            if len(proj) == len(bc):
                ax_b.plot(bc, proj, color=color, linestyle=ls, linewidth=2, label=label)
        ax_b.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_b.set_title(f"B. Decision component (Expert {best})")
    else:
        ax_b.set_title("B. Decision component")
    ax_b.set_xlabel("Time from Change_ON (s)")
    ax_b.set_ylabel("dPC projection")
    ax_b.legend(fontsize=7)

    # Panel C: Top stimulus dPC (Expert sessions)
    ax_c = fig.add_subplot(gs[1, 0])
    if expert:
        r = expert[best]
        bc = r["bin_centers"]
        for label, proj in r["stimulus_pcs"].items():
            color = OUTCOME_COLORS["Hit"] if "Hit" in label else OUTCOME_COLORS["Miss"]
            ls = "-" if "Big" in label else "--"
            if len(proj) == len(bc):
                ax_c.plot(bc, proj, color=color, linestyle=ls, linewidth=2, label=label)
        ax_c.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_c.set_title(f"C. Stimulus component (Expert {best})")
    else:
        ax_c.set_title("C. Stimulus component")
    ax_c.set_xlabel("Time from Change_ON (s)")
    ax_c.set_ylabel("dPC projection")
    ax_c.legend(fontsize=7)

    # Panel D: Variance partition pie charts by stage
    gs_inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 1], wspace=0.3)
    for si, stage in enumerate(STAGE_ORDER):
        ax_pie = fig.add_subplot(gs_inner[si])
        stage_results = [v for v in results.values() if v["stage"] == stage]
        if stage_results:
            mean_fracs = []
            for m in marg_names:
                fracs = []
                for r in stage_results:
                    tv = r["total_var"]
                    if tv > 0:
                        fracs.append(r["var_dict"].get(m, 0) / tv)
                mean_fracs.append(np.mean(fracs) if fracs else 0)

            # Normalize to sum to 1
            total = sum(mean_fracs)
            if total > 0:
                mean_fracs = [f / total for f in mean_fracs]
            ax_pie.pie(mean_fracs, colors=marg_colors, autopct="%1.0f%%",
                       textprops={"fontsize": 7})
        ax_pie.set_title(stage, fontsize=10, color=STAGE_COLORS[stage], fontweight="bold")

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # Decision variance fraction by stage
    for m in ["decision", "stimulus"]:
        from scipy.stats import kruskal as kruskal_test
        groups = []
        for stage in STAGE_ORDER:
            vals = [r["var_dict"].get(m, 0) / r["total_var"]
                    for r in results.values()
                    if r["stage"] == stage and r["total_var"] > 0]
            if len(vals) >= 2:
                groups.append(vals)
        if len(groups) >= 2:
            try:
                h, p = kruskal_test(*groups)
                stats.append({"test": f"{m}_var_frac_kruskal_by_stage", "H": h, "p": p})
            except ValueError:
                pass

    # Overall mean fractions
    for m in marg_names:
        vals = all_fracs.get(m, [])
        if vals:
            stats.append({
                "test": f"mean_var_frac_{m}",
                "value": float(np.mean(vals)),
                "n_sessions": len(vals),
            })

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig28_dpca", "07_advanced")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "07_advanced", "dpca_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: {row.get('p', row.get('value', 'N/A'))}")


if __name__ == "__main__":
    main()
