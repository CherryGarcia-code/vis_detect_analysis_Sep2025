"""Fig 08: Responsiveness screen — population-wide responsiveness to Change_ON.

Produces:
  - Fig 08A: Volcano plot (d' vs -log10 p) colored by responsive/non-responsive
  - Fig 08B: Fraction responsive by stage (bar chart)
  - Fig 08C: Population PSTH heatmap sorted by peak latency (Expert sessions)
  - Fig 08D: Distribution of response magnitudes (delta_FR) by cell type

Caches: cache/responsiveness_all_sessions.csv
Saves statistics to figures/02_single_unit/responsiveness_stats.csv
"""

import os
import sys
import gc


import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import (
    STAGE_ORDER, STAGE_COLORS, CELLTYPE_COLORS,
    CACHE_DIR, DEFAULT_BIN_SIZE,
)
from visdetect.suite.loader import load_staging_manifest, load_session, load_waveform_labels
from visdetect.analysis.utils import (
    get_good_cluster_ids, build_population_tensor,
    smooth_psth, compute_zscore_normalized, compute_auroc,
    fdr_correct,
)
from visdetect.suite.plotting import setup_style, save_figure

# Non-archived alignment & canonical constants

from visdetect.analysis.align import align_spikes_to_events, get_event_times_by_trial
from visdetect.analysis.constants import EVENT_RESPONSIVENESS_WINDOWS

setup_style()

# Canonical responsiveness windows from visdetect constants
_CHANGE_ON_WINDOWS = EVENT_RESPONSIVENESS_WINDOWS["Change_ON"]
BASE_WIN = _CHANGE_ON_WINDOWS[0]   # (-0.4, -0.05)
RESP_WIN = _CHANGE_ON_WINDOWS[1]   # (0.0, 0.25)
RESP_BIN_SIZE = 0.01
MIN_TRIALS = 5
N_PERM = 500

CACHE_FILE = os.path.join(CACHE_DIR, "responsiveness_all_sessions.csv")


def _sign_flip_perm_p(diff, n_perm=500, rng=None):
    """Two-sided sign-flip permutation test on paired differences."""
    x = np.asarray(diff, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    if rng is None:
        rng = np.random.default_rng()
    obs = float(np.mean(x))
    signs = rng.choice([-1.0, 1.0], size=(n_perm, x.size))
    null = (signs * x).mean(axis=1)
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))


def _compute_responsiveness_for_session(sess):
    """Compute per-unit responsiveness to Change_ON for one session.

    Uses non-archived align functions and canonical EVENT_RESPONSIVENESS_WINDOWS.
    Returns a DataFrame with columns: cluster_id, outcome, n_trials, delta_fr,
    dprime, auc, p_value, is_responsive.
    """
    good_ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
    event_times = get_event_times_by_trial(sess, "Change_ON")
    trials = getattr(sess, "trials", []) or []

    # Pooled hit+miss trial indices with valid event times
    valid_indices = [
        i for i in range(len(trials))
        if getattr(trials[i], "trialoutcome", None) in ("Hit", "Miss")
        and i < len(event_times) and np.isfinite(event_times[i])
    ]

    cluster_map = {int(c.cluster_id): c for c in sess.clusters}
    rng = np.random.default_rng(12345)
    rows = []

    for cid in good_ids:
        c = cluster_map.get(cid)
        if c is None:
            continue
        if len(valid_indices) < MIN_TRIALS:
            rows.append({
                "cluster_id": cid, "outcome": "All", "n_trials": len(valid_indices),
                "delta_fr": np.nan, "dprime": np.nan, "auc": np.nan,
                "p_value": np.nan, "is_responsive": False,
            })
            continue

        ets = [float(event_times[i]) for i in valid_indices]
        st = np.asarray(c.spike_times).flatten()

        # Per-trial mean FR in baseline and response windows
        base_mat, _ = align_spikes_to_events(st, ets, window=BASE_WIN, bin_size=RESP_BIN_SIZE)
        resp_mat, _ = align_spikes_to_events(st, ets, window=RESP_WIN, bin_size=RESP_BIN_SIZE)
        base_fr = np.nanmean(base_mat, axis=1)  # per-trial mean FR
        resp_fr = np.nanmean(resp_mat, axis=1)

        diff = resp_fr - base_fr
        mask = np.isfinite(diff)
        diff_clean = diff[mask]
        base_clean = base_fr[mask]
        resp_clean = resp_fr[mask]

        if len(diff_clean) < MIN_TRIALS:
            rows.append({
                "cluster_id": cid, "outcome": "All", "n_trials": len(diff_clean),
                "delta_fr": np.nan, "dprime": np.nan, "auc": np.nan,
                "p_value": np.nan, "is_responsive": False,
            })
            continue

        delta_fr = float(np.mean(diff_clean))
        sd = float(np.std(diff_clean, ddof=1))
        dprime = delta_fr / sd if sd > 0 else np.nan
        auc = compute_auroc(resp_clean, base_clean)
        p_value = _sign_flip_perm_p(diff_clean, n_perm=N_PERM, rng=rng)
        is_resp = bool(p_value < 0.05 and np.isfinite(dprime))

        rows.append({
            "cluster_id": cid, "outcome": "All",
            "n_trials": len(diff_clean), "delta_fr": delta_fr,
            "dprime": dprime, "auc": auc,
            "p_value": p_value, "is_responsive": is_resp,
        })

    return pd.DataFrame(rows)


def compute_or_load_responsiveness(force_recompute=False):
    """Compute responsiveness for all sessions, or load from cache."""
    if os.path.exists(CACHE_FILE) and not force_recompute:
        print("  Loading cached responsiveness table...")
        return pd.read_csv(CACHE_FILE)

    print("  Computing responsiveness for all sessions...")
    manifest = load_staging_manifest(qc_only=True)

    all_frames = []
    for _, row in manifest.iterrows():
        sname = int(row["session_name"])
        stage = row["stage"]
        sidx = row["session_idx"]
        try:
            sess = load_session(sname)
        except FileNotFoundError:
            print(f"    Skipping {sname}: not found")
            continue

        print(f"    Processing {sname} ({stage})...", end=" ")
        try:
            df = _compute_responsiveness_for_session(sess)
            df["session_name"] = sname
            df["stage"] = stage
            df["session_idx"] = sidx
            all_frames.append(df)
            print(f"{len(df)} units, {df['is_responsive'].sum()} responsive")
        except Exception as e:
            print(f"Error: {e}")

        del sess
        gc.collect()

    if not all_frames:
        print("  No responsiveness data computed!")
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)
    result.to_csv(CACHE_FILE, index=False)
    print(f"  Cached {len(result)} rows to {CACHE_FILE}")
    return result


def main():
    print("[02a] Responsiveness screen...")
    manifest = load_staging_manifest(qc_only=True)

    # Step 1: Compute/load responsiveness
    resp_df = compute_or_load_responsiveness()
    if resp_df.empty:
        print("  No data. Exiting.")
        return

    # Step 1b: Apply FDR correction across all units
    valid_mask = resp_df["p_value"].notna()
    if valid_mask.sum() > 0:
        fdr_sig = fdr_correct(resp_df.loc[valid_mask, "p_value"].values)
        resp_df.loc[valid_mask, "is_responsive"] = fdr_sig
        n_before = resp_df["p_value"].notna().sum()
        n_after = resp_df["is_responsive"].sum()
        print(f"  FDR correction: {n_after}/{n_before} units responsive (alpha=0.05)")

    # Merge cell-type labels
    try:
        wf = load_waveform_labels()
        if "session_date" in wf.columns and "cluster_id" in wf.columns:
            wf_sub = wf[["session_date", "cluster_id", "celltype"]].copy()
            wf_sub["session_date"] = wf_sub["session_date"].astype(int)
            resp_df = resp_df.merge(
                wf_sub,
                left_on=["session_name", "cluster_id"],
                right_on=["session_date", "cluster_id"],
                how="left",
            )
    except Exception:
        resp_df["celltype"] = np.nan

    print(f"  Total units: {len(resp_df)}, "
          f"Responsive: {resp_df['is_responsive'].sum()} "
          f"({100*resp_df['is_responsive'].mean():.1f}%)")

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Volcano plot
    ax_a = fig.add_subplot(gs[0, 0])
    dp = resp_df["dprime"].astype(float).values
    pvals = resp_df["p_value"].astype(float).values
    neg_log_p = -np.log10(np.clip(pvals, 1e-12, 1.0))
    is_resp = resp_df["is_responsive"].astype(bool).values

    ax_a.scatter(dp[~is_resp], neg_log_p[~is_resp], s=8, c="#bdbdbd",
                 alpha=0.5, edgecolors="none", label="Non-responsive")
    ax_a.scatter(dp[is_resp], neg_log_p[is_resp], s=12, c="#d62728",
                 alpha=0.7, edgecolors="none", label="Responsive")
    ax_a.axhline(-np.log10(0.05), color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_a.set_xlabel("d' (effect size)")
    ax_a.set_ylabel("-log10(p)")
    ax_a.set_title("A. Responsiveness to Change_ON")
    ax_a.legend(fontsize=8)

    # Panel B: Fraction responsive by stage
    ax_b = fig.add_subplot(gs[0, 1])
    stage_fracs = []
    for stage in STAGE_ORDER:
        sub = resp_df[resp_df["stage"] == stage]
        if len(sub) > 0:
            frac = sub["is_responsive"].mean()
            se = np.sqrt(frac * (1 - frac) / len(sub))
        else:
            frac, se = 0, 0
        stage_fracs.append({"stage": stage, "frac": frac, "se": se, "n": len(sub)})

    sf_df = pd.DataFrame(stage_fracs)
    bars = ax_b.bar(range(len(STAGE_ORDER)), sf_df["frac"],
                    yerr=sf_df["se"], capsize=5,
                    color=[STAGE_COLORS[s] for s in STAGE_ORDER],
                    edgecolor="white", linewidth=1)
    ax_b.set_xticks(range(len(STAGE_ORDER)))
    ax_b.set_xticklabels(STAGE_ORDER)
    ax_b.set_ylabel("Fraction responsive")
    ax_b.set_ylim(0, 1)
    ax_b.set_title("B. Responsive fraction by stage")
    # Annotate n
    for i, row in sf_df.iterrows():
        ax_b.text(i, row["frac"] + row["se"] + 0.02, f"n={row['n']}",
                  ha="center", fontsize=8, color="gray")

    # Panel C: Population PSTH heatmap (Expert sessions, sorted by peak latency)
    ax_c = fig.add_subplot(gs[1, 0])
    expert_resp = resp_df[(resp_df["stage"] == "Expert") & resp_df["is_responsive"]]

    if len(expert_resp) > 0:
        # Pick up to 3 Expert sessions for heatmap
        expert_sessions = sorted(expert_resp["session_name"].unique())[:3]
        all_psths = []
        bin_centers_ref = None

        for sname in expert_sessions:
            try:
                sess = load_session(sname)
                cids = expert_resp[expert_resp["session_name"] == sname]["cluster_id"].tolist()
                if not cids:
                    del sess
                    gc.collect()
                    continue
                tensor, bc, _ = build_population_tensor(
                    sess, cids, event_name="Change_ON",
                    window=(-0.5, 1.0), bin_size=0.01,
                    outcome_filter={"Hit", "Miss"},
                )
                if tensor.shape[0] > 0:
                    # Compute mean PSTH per unit and z-score
                    z_tensor = compute_zscore_normalized(tensor, bc, (-0.5, -0.05))
                    mean_psth = np.nanmean(z_tensor, axis=0)  # (n_bins, n_units)
                    for u in range(mean_psth.shape[1]):
                        all_psths.append(smooth_psth(mean_psth[:, u], 0.01, sigma_ms=15.0))
                    bin_centers_ref = bc
                del sess
                gc.collect()
            except Exception:
                continue

        if all_psths and bin_centers_ref is not None:
            psth_matrix = np.array(all_psths)  # (n_units, n_bins)
            # Sort by peak latency in response window
            post_mask = bin_centers_ref >= 0
            peak_idx = np.argmax(psth_matrix[:, post_mask], axis=1)
            sort_order = np.argsort(peak_idx)
            psth_sorted = psth_matrix[sort_order]

            vmax = np.percentile(np.abs(psth_sorted), 95)
            ax_c.imshow(psth_sorted, aspect="auto", interpolation="nearest",
                        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        extent=[bin_centers_ref[0], bin_centers_ref[-1],
                                psth_sorted.shape[0], 0])
            ax_c.axvline(0, color="white", linewidth=1, linestyle="--")
            ax_c.set_xlabel("Time from Change_ON (s)")
            ax_c.set_ylabel("Unit (sorted by peak)")
            ax_c.set_title(f"C. Population heatmap (Expert, n={psth_sorted.shape[0]} units)")
        else:
            ax_c.text(0.5, 0.5, "No data", transform=ax_c.transAxes, ha="center")
            ax_c.set_title("C. Population heatmap")
    else:
        ax_c.text(0.5, 0.5, "No responsive Expert units", transform=ax_c.transAxes, ha="center")
        ax_c.set_title("C. Population heatmap")

    # Panel D: Delta FR distribution by cell type
    ax_d = fig.add_subplot(gs[1, 1])
    if "celltype" in resp_df.columns:
        for ct, color in CELLTYPE_COLORS.items():
            sub = resp_df[resp_df["celltype"] == ct]["delta_fr"].dropna()
            if len(sub) > 0:
                ax_d.hist(sub.values, bins=40, alpha=0.5, color=color, label=f"{ct} (n={len(sub)})")
    else:
        sub = resp_df["delta_fr"].dropna()
        ax_d.hist(sub.values, bins=40, alpha=0.7, color="#4C78A8")

    ax_d.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax_d.set_xlabel("Delta FR (Hz)")
    ax_d.set_ylabel("Units")
    ax_d.set_title("D. Response magnitude by cell type")
    ax_d.legend(fontsize=8)

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # Responsive fraction across stages
    stage_resp_counts = []
    for stage in STAGE_ORDER:
        sub = resp_df[resp_df["stage"] == stage]
        stage_resp_counts.append(sub["is_responsive"].values)
    valid_groups = [g for g in stage_resp_counts if len(g) >= 2]
    if len(valid_groups) >= 2:
        h, p = kruskal(*valid_groups)
        stats.append({"test": "responsive_frac_kruskal_by_stage", "H": h, "p": p})

    # Trend: responsive fraction vs session index
    per_sess = resp_df.groupby("session_idx").agg(
        frac_resp=("is_responsive", "mean"),
    ).reset_index()
    if len(per_sess) >= 3:
        rho, p = spearmanr(per_sess["session_idx"], per_sess["frac_resp"])
        stats.append({"test": "responsive_frac_vs_session_spearman", "rho": rho, "p": p})

    # Cell-type comparison
    if "celltype" in resp_df.columns:
        narrow = resp_df[resp_df["celltype"] == "Narrow (FSI)"]["delta_fr"].dropna()
        broad = resp_df[resp_df["celltype"] == "Broad (MSN/Proj)"]["delta_fr"].dropna()
        if len(narrow) >= 5 and len(broad) >= 5:
            u, p = mannwhitneyu(narrow, broad, alternative="two-sided")
            stats.append({"test": "delta_fr_narrow_vs_broad_mwu", "U": u, "p": p,
                          "n_narrow": len(narrow), "n_broad": len(broad)})

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig08_responsiveness_screen", "02_single_unit")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "02_single_unit", "responsiveness_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)
    print(f"  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: p={row.get('p', 'N/A')}")


if __name__ == "__main__":
    main()
