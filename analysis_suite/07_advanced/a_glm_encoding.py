"""Fig27: Poisson GLM encoding model — partial deviance explained by predictor.

Fits a Poisson GLM per unit with predictors: stimulus (change size),
choice (hit/miss), HMM behavioral state, and pre-event baseline FR.

Quantifies partial deviance explained (PDE) by each predictor to
understand what information striatal neurons encode.

Produces:
  - Fig 27A: Distribution of total deviance explained across units
  - Fig 27B: PDE by predictor (stacked bars)
  - Fig 27C: PDE by predictor across stages
  - Fig 27D: PDE by predictor for FSI vs MSN

Saves: figures/07_advanced/glm_encoding_stats.csv
       cache/glm_results.csv
"""

import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    STAGE_ORDER, STAGE_COLORS, CELLTYPE_COLORS, CACHE_DIR,
    CHANGE_SIZES,
)
from loader import (
    load_staging_manifest, load_session, load_hmm_assignments,
    load_waveform_labels,
)
from visdetect.analysis.utils import get_good_cluster_ids
from plotting import setup_style, save_figure

from visdetect.analysis.align import align_spikes_to_events, get_event_times_by_trial

setup_style()

RESP_WIN = (0.0, 0.25)
BASE_WIN = (-0.4, -0.05)
MIN_TRIALS = 20


def fit_poisson_glm(X, y):
    """Fit Poisson GLM using iteratively reweighted least squares.

    X: (n_trials, n_predictors) design matrix
    y: (n_trials,) spike counts (non-negative integers)

    Returns: coefficients, deviance_full, deviance_null
    """
    from scipy.optimize import minimize

    n, p = X.shape
    y = np.asarray(y, dtype=float)

    # Add intercept
    X_full = np.column_stack([np.ones(n), X])

    def neg_log_lik(beta):
        eta = X_full @ beta
        eta = np.clip(eta, -20, 20)
        mu = np.exp(eta)
        # Poisson log-likelihood (up to constant)
        ll = np.sum(y * eta - mu)
        return -ll

    # Initialize
    beta0 = np.zeros(p + 1)
    beta0[0] = np.log(max(np.mean(y), 0.1))

    try:
        res = minimize(neg_log_lik, beta0, method="L-BFGS-B",
                       options={"maxiter": 200, "disp": False})
        if not res.success:
            return None, np.nan, np.nan

        beta = res.x
        nll_full = res.fun

        # Null model (intercept only)
        def neg_log_lik_null(b0):
            eta = b0[0] * np.ones(n)
            mu = np.exp(np.clip(eta, -20, 20))
            return -np.sum(y * eta - mu)

        res_null = minimize(neg_log_lik_null, [np.log(max(np.mean(y), 0.1))],
                            method="L-BFGS-B")
        nll_null = res_null.fun

        dev_null = 2 * nll_null
        dev_full = 2 * nll_full

        return beta[1:], dev_full, dev_null

    except Exception:
        return None, np.nan, np.nan


def compute_partial_deviance(X, y, predictor_groups):
    """Compute partial deviance explained for each predictor group.

    predictor_groups: dict mapping name -> list of column indices
    """
    n, p = X.shape
    _, dev_full, dev_null = fit_poisson_glm(X, y)

    if not np.isfinite(dev_null) or not np.isfinite(dev_full):
        return {}

    total_de = 1 - dev_full / dev_null if dev_null > 0 else 0

    pde = {"total_DE": total_de}
    for name, cols in predictor_groups.items():
        # Fit reduced model without this predictor
        keep_cols = [c for c in range(p) if c not in cols]
        if len(keep_cols) == 0:
            pde[name] = total_de
            continue
        X_reduced = X[:, keep_cols]
        _, dev_reduced, _ = fit_poisson_glm(X_reduced, y)
        if np.isfinite(dev_reduced) and dev_null > 0:
            de_reduced = 1 - dev_reduced / dev_null
            pde[name] = total_de - de_reduced
        else:
            pde[name] = 0.0

    return pde


def _process_session_worker(args):
    """Module-level worker for ProcessPoolExecutor: fit Poisson GLM for one session.

    Reloads HMM and waveform data from disk so the worker is self-contained
    and avoids serialising large DataFrames across processes.
    Returns (sname, list_of_unit_result_dicts).
    """
    sname, stage, sidx = args
    try:
        sess = load_session(sname)
    except FileNotFoundError:
        return sname, []

    # Reload small supporting CSVs in the worker process
    try:
        hmm = load_hmm_assignments()
    except Exception:
        hmm = pd.DataFrame()

    try:
        wf = load_waveform_labels()
        ct_lookup = {}
        for _, row in wf.iterrows():
            sn = int(row.get("session_name", row.get("session_date", 0)))
            ct_lookup[(sn, int(row["cluster_id"]))] = row.get("cell_type", row.get("celltype", "Unknown"))
    except Exception:
        ct_lookup = {}

    good_ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
    event_times = get_event_times_by_trial(sess, "Change_ON")
    cluster_map = {int(c.cluster_id): c for c in sess.clusters}

    sess_hmm = hmm[hmm["session_name"] == sname] if not hmm.empty else pd.DataFrame()
    hmm_states = {}
    if "trial_idx" in sess_hmm.columns and "hmm_state_label" in sess_hmm.columns:
        for _, hr in sess_hmm.iterrows():
            hmm_states[int(hr["trial_idx"])] = hr["hmm_state_label"]

    trial_data = []
    for i, t in enumerate(sess.trials):
        outcome = getattr(t, "trialoutcome", None)
        if outcome not in ("Hit", "Miss"):
            continue
        if i >= len(event_times) or not np.isfinite(event_times[i]):
            continue
        cs = getattr(t, "change_size", None)
        if cs is None:
            continue
        cs_norm = (cs - min(CHANGE_SIZES)) / (max(CHANGE_SIZES) - min(CHANGE_SIZES) + 1e-6)
        state = hmm_states.get(i, "Unknown")
        state_val = {"Engaged": 1.0, "Impulsive": 0.5, "Disengaged": 0.0}.get(state, 0.5)
        trial_data.append({
            "trial_idx": i,
            "outcome": 1 if outcome == "Hit" else 0,
            "change_size": cs_norm,
            "state": state_val,
            "event_time": float(event_times[i]),
        })

    if len(trial_data) < MIN_TRIALS:
        del sess; gc.collect()
        return sname, []

    td = pd.DataFrame(trial_data)
    trial_event_times = td["event_time"].values.tolist()
    unit_results = []

    for cid in good_ids:
        c = cluster_map.get(int(cid))
        if c is None:
            continue
        mat, bc = align_spikes_to_events(
            c.spike_times, trial_event_times,
            window=(-0.5, 0.5), bin_size=0.025,
        )
        resp_mask = (bc >= RESP_WIN[0]) & (bc < RESP_WIN[1])
        base_mask = (bc >= BASE_WIN[0]) & (bc < BASE_WIN[1])
        resp_counts = np.nansum(mat[:, resp_mask], axis=1) * 0.025
        resp_counts = np.round(resp_counts).astype(int)
        resp_counts = np.clip(resp_counts, 0, None)
        base_fr = np.nanmean(mat[:, base_mask], axis=1)
        X = np.column_stack([
            td["change_size"].values,
            td["outcome"].values,
            td["state"].values,
            base_fr,
        ])
        valid = np.isfinite(X).all(axis=1) & np.isfinite(resp_counts)
        if valid.sum() < MIN_TRIALS:
            continue
        X_clean = X[valid]
        y_clean = resp_counts[valid]
        if y_clean.sum() < 5:
            continue
        predictor_groups = {"stimulus": [0], "choice": [1], "state": [2], "baseline": [3]}
        pde = compute_partial_deviance(X_clean, y_clean, predictor_groups)
        if not pde:
            continue
        ct = ct_lookup.get((sname, int(cid)), "Unknown")
        unit_results.append({
            "session_name": sname,
            "stage": stage,
            "session_idx": sidx,
            "cluster_id": cid,
            "cell_type": ct,
            **pde,
        })

    del sess; gc.collect()
    return sname, unit_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Parallel worker processes for session-level GLM fitting "
                             "(default: 1 = sequential). Each worker independently loads "
                             "a session and fits the Poisson GLM for all its units.")
    args = parser.parse_args()

    print("[07a] Poisson GLM encoding model...")
    manifest = load_staging_manifest(qc_only=True)
    hmm = load_hmm_assignments()

    try:
        wf = load_waveform_labels()
        ct_lookup = {}
        for _, row in wf.iterrows():
            sn = int(row.get("session_name", row.get("session_date", 0)))
            ct_lookup[(sn, int(row["cluster_id"]))] = row.get("cell_type", row.get("celltype", "Unknown"))
    except FileNotFoundError:
        ct_lookup = {}

    cache_path = os.path.join(CACHE_DIR, "glm_results.csv")

    if os.path.exists(cache_path):
        print(f"  Loading cached GLM results from {cache_path}")
        glm_df = pd.read_csv(cache_path)
    else:
        tasks = [
            (int(row["session_name"]), row["stage"], row["session_idx"])
            for _, row in manifest.iterrows()
        ]
        all_results = []

        if args.n_workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            print(f"  Using {args.n_workers} parallel workers")
            with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
                for sname, unit_results in ex.map(_process_session_worker, tasks):
                    stage = next((t[1] for t in tasks if t[0] == sname), "?")
                    print(f"  Session {sname} ({stage})... {len(unit_results)} units")
                    all_results.extend(unit_results)
        else:
            for sname, stage, sidx in tasks:
                print(f"  Session {sname} ({stage})...", end=" ")
                try:
                    sess = load_session(sname)
                except FileNotFoundError:
                    print("not found")
                    continue

                good_ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
                event_times = get_event_times_by_trial(sess, "Change_ON")
                cluster_map = {int(c.cluster_id): c for c in sess.clusters}

                sess_hmm = hmm[hmm["session_name"] == sname]
                hmm_states = {}
                if "trial_idx" in sess_hmm.columns and "hmm_state_label" in sess_hmm.columns:
                    for _, hr in sess_hmm.iterrows():
                        hmm_states[int(hr["trial_idx"])] = hr["hmm_state_label"]

                trial_data = []
                for i, t in enumerate(sess.trials):
                    outcome = getattr(t, "trialoutcome", None)
                    if outcome not in ("Hit", "Miss"):
                        continue
                    if i >= len(event_times) or not np.isfinite(event_times[i]):
                        continue
                    cs = getattr(t, "change_size", None)
                    if cs is None:
                        continue
                    cs_norm = (cs - min(CHANGE_SIZES)) / (max(CHANGE_SIZES) - min(CHANGE_SIZES) + 1e-6)
                    state = hmm_states.get(i, "Unknown")
                    state_val = {"Engaged": 1.0, "Impulsive": 0.5, "Disengaged": 0.0}.get(state, 0.5)
                    trial_data.append({
                        "trial_idx": i,
                        "outcome": 1 if outcome == "Hit" else 0,
                        "change_size": cs_norm,
                        "state": state_val,
                        "event_time": float(event_times[i]),
                    })

                if len(trial_data) < MIN_TRIALS:
                    print(f"too few trials ({len(trial_data)})")
                    del sess; gc.collect()
                    continue

                td = pd.DataFrame(trial_data)
                trial_event_times = td["event_time"].values.tolist()
                n_units = 0

                for cid in good_ids:
                    c = cluster_map.get(int(cid))
                    if c is None:
                        continue
                    mat, bc = align_spikes_to_events(
                        c.spike_times, trial_event_times,
                        window=(-0.5, 0.5), bin_size=0.025,
                    )
                    resp_mask = (bc >= RESP_WIN[0]) & (bc < RESP_WIN[1])
                    base_mask = (bc >= BASE_WIN[0]) & (bc < BASE_WIN[1])
                    resp_counts = np.nansum(mat[:, resp_mask], axis=1) * 0.025
                    resp_counts = np.round(resp_counts).astype(int)
                    resp_counts = np.clip(resp_counts, 0, None)
                    base_fr = np.nanmean(mat[:, base_mask], axis=1)
                    X = np.column_stack([
                        td["change_size"].values,
                        td["outcome"].values,
                        td["state"].values,
                        base_fr,
                    ])
                    valid = np.isfinite(X).all(axis=1) & np.isfinite(resp_counts)
                    if valid.sum() < MIN_TRIALS:
                        continue
                    X_clean = X[valid]
                    y_clean = resp_counts[valid]
                    if y_clean.sum() < 5:
                        continue
                    predictor_groups = {
                        "stimulus": [0],
                        "choice": [1],
                        "state": [2],
                        "baseline": [3],
                    }
                    pde = compute_partial_deviance(X_clean, y_clean, predictor_groups)
                    if not pde:
                        continue
                    ct = ct_lookup.get((sname, int(cid)), "Unknown")
                    all_results.append({
                        "session_name": sname,
                        "stage": stage,
                        "session_idx": sidx,
                        "cluster_id": cid,
                        "cell_type": ct,
                        **pde,
                    })
                    n_units += 1

                print(f"{n_units} units")
                del sess; gc.collect()

        glm_df = pd.DataFrame(all_results)
        if len(glm_df) > 0:
            glm_df.to_csv(cache_path, index=False)

    if len(glm_df) == 0:
        print("  No GLM results. Exiting.")
        return

    print(f"\n  GLM fitted for {len(glm_df)} units")

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    predictors = ["stimulus", "choice", "state", "baseline"]
    pred_colors = ["#e74c3c", "#2ecc71", "#3498db", "#95a5a6"]

    # Panel A: Total DE distribution
    ax_a = fig.add_subplot(gs[0, 0])
    de_vals = glm_df["total_DE"].dropna().clip(0, 1).values
    ax_a.hist(de_vals, bins=40, color="#7986CB", edgecolor="white", alpha=0.8)
    ax_a.axvline(np.median(de_vals), color="#E53935", linewidth=1.5,
                 label=f"Median={np.median(de_vals):.3f}")
    ax_a.set_xlabel("Total deviance explained")
    ax_a.set_ylabel("Number of units")
    ax_a.set_title(f"A. GLM goodness of fit (n={len(de_vals)})")
    ax_a.legend(fontsize=8)

    # Panel B: Mean PDE by predictor (stacked bar)
    ax_b = fig.add_subplot(gs[0, 1])
    mean_pde = []
    for pred in predictors:
        if pred in glm_df.columns:
            mean_pde.append(glm_df[pred].clip(0, None).mean())
        else:
            mean_pde.append(0)

    ax_b.bar(range(len(predictors)), mean_pde, color=pred_colors,
             edgecolor="white")
    ax_b.set_xticks(range(len(predictors)))
    ax_b.set_xticklabels(["Stimulus", "Choice", "State", "Baseline"])
    ax_b.set_ylabel("Mean partial deviance explained")
    ax_b.set_title("B. Information encoded by predictor")

    # Panel C: PDE by stage
    ax_c = fig.add_subplot(gs[1, 0])
    bar_width = 0.18
    for pi, pred in enumerate(predictors):
        if pred not in glm_df.columns:
            continue
        for si, stage in enumerate(STAGE_ORDER):
            sub = glm_df[glm_df["stage"] == stage]
            if len(sub) > 0:
                mean_val = sub[pred].clip(0, None).mean()
                x = si + (pi - 1.5) * bar_width
                ax_c.bar(x, mean_val, bar_width * 0.9, color=pred_colors[pi],
                         alpha=0.8, label=pred.capitalize() if si == 0 else "")

    ax_c.set_xticks(range(len(STAGE_ORDER)))
    ax_c.set_xticklabels(STAGE_ORDER)
    ax_c.set_ylabel("Mean PDE")
    ax_c.set_title("C. Encoding by predictor across stages")
    ax_c.legend(fontsize=7, ncol=2)

    # Panel D: PDE by cell type
    ax_d = fig.add_subplot(gs[1, 1])
    ct_types = ["Narrow (FSI)", "Broad (MSN/Proj)"]
    ct_labels = ["FSI", "MSN"]
    bar_width = 0.18

    for pi, pred in enumerate(predictors):
        if pred not in glm_df.columns:
            continue
        for ci, ct in enumerate(ct_types):
            sub = glm_df[glm_df["cell_type"] == ct]
            if len(sub) > 0:
                mean_val = sub[pred].clip(0, None).mean()
                x = ci + (pi - 1.5) * bar_width
                ax_d.bar(x, mean_val, bar_width * 0.9, color=pred_colors[pi],
                         alpha=0.8, label=pred.capitalize() if ci == 0 else "")

    ax_d.set_xticks(range(len(ct_types)))
    ax_d.set_xticklabels(ct_labels)
    ax_d.set_ylabel("Mean PDE")
    ax_d.set_title("D. Encoding by cell type")
    ax_d.legend(fontsize=7, ncol=2)

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # Overall DE
    stats.append({
        "test": "overall_total_DE",
        "median": float(np.median(de_vals)),
        "mean": float(np.mean(de_vals)),
        "n_units": len(de_vals),
    })

    # Which predictor dominates
    for pred in predictors:
        if pred in glm_df.columns:
            vals = glm_df[pred].dropna().values
            stats.append({
                "test": f"mean_PDE_{pred}",
                "value": float(np.mean(vals)),
                "median": float(np.median(vals)),
            })

    # Choice PDE: Expert vs Naive
    for pred in ["choice", "stimulus"]:
        if pred in glm_df.columns:
            expert = glm_df[glm_df["stage"] == "Expert"][pred].dropna().values
            naive = glm_df[glm_df["stage"] == "Naive"][pred].dropna().values
            if len(expert) >= 5 and len(naive) >= 5:
                u, p = mannwhitneyu(expert, naive, alternative="two-sided")
                stats.append({
                    "test": f"PDE_{pred}_expert_vs_naive",
                    "U": u, "p": p,
                })

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig27_glm_encoding", "07_advanced")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "07_advanced", "glm_encoding_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: {row.get('p', row.get('value', row.get('median', 'N/A')))}")


if __name__ == "__main__":
    main()
