"""Fig18: Hit vs Miss Decoding — Cross-validated decoding over time.

Complements the coding direction analysis with a standard classifier
framework: at each time bin, train/test logistic regression with
stratified 5-fold CV to decode trial outcome.

Produces:
  - Fig 18A: Decoding accuracy over time for a single Expert session
  - Fig 18B: Grand-average decoding accuracy across Expert sessions
  - Fig 18C: Decoding onset latency (first above-chance bin) vs session index
  - Fig 18D: Peak decoding accuracy by stage

Saves statistics to figures/04_decoding/hit_miss_decoding_stats.csv
"""

import os
import sys
import gc


import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from visdetect.suite.config import (
    STAGE_ORDER, STAGE_COLORS, OUTCOME_COLORS, CACHE_DIR,
)
from visdetect.suite.loader import load_staging_manifest, load_session
from visdetect.analysis.utils import (
    get_good_cluster_ids, build_population_tensor, smooth_psth,
    compute_zscore_normalized
)
from visdetect.suite.plotting import setup_style, save_figure, add_stage_background

setup_style()

# Parameters
WINDOW = (-0.5, 1.0)
BASELINE_WINDOW = (-0.5, -0.05)  # Shared baseline for normalization
BIN_SIZE = 0.050
MIN_UNITS = 10
MIN_TRIALS_PER_CLASS = 8
N_FOLDS = 5
N_PERM = 20  # for chance-level estimation


def decode_at_timebin(X, y, n_folds=5, random_state=42):
    """Cross-validated logistic regression accuracy at a single time bin.

    X: (n_trials, n_units), y: binary labels.
    Returns (accuracy, accuracy_sem).
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    accs = []
    for train_idx, test_idx in cv.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=500)
        clf.fit(X_train, y[train_idx])
        accs.append(clf.score(X_test, y[test_idx]))
    return np.mean(accs), np.std(accs) / np.sqrt(len(accs))


def decode_session(sess, session_name):
    """Run time-resolved decoding for one session."""
    good_ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
    if len(good_ids) < MIN_UNITS:
        return None

    trials = sess.trials
    go_hit_idx = [
        i for i, t in enumerate(trials)
        if getattr(t, "trialoutcome", None) == "Hit"
        and (getattr(t, "change_size", None) or 1.0) > 1.01
    ]
    go_miss_idx = [
        i for i, t in enumerate(trials)
        if getattr(t, "trialoutcome", None) == "Miss"
        and (getattr(t, "change_size", None) or 1.0) > 1.01
    ]
    fa_idx = [
        i for i, t in enumerate(trials)
        if getattr(t, "trialoutcome", None) == "Hit"
        and (getattr(t, "change_size", None) or 1.0) <= 1.01
    ]

    if len(go_hit_idx) < MIN_TRIALS_PER_CLASS or len(go_miss_idx) < MIN_TRIALS_PER_CLASS:
        return None

    tensor, bin_centers, used = build_population_tensor(
        sess, good_ids, event_name="Change_ON",
        window=WINDOW, bin_size=BIN_SIZE,
        trial_indices=go_hit_idx + go_miss_idx,
    )

    if tensor.shape[0] < 2 * MIN_TRIALS_PER_CLASS or tensor.shape[2] < MIN_UNITS:
        return None

    # Normalize to shared baseline (removes baseline rate confounds)
    tensor = compute_zscore_normalized(tensor, bin_centers, BASELINE_WINDOW)

    # Labels
    labels = np.array([
        1 if getattr(trials[i], "trialoutcome", None) == "Hit" else 0
        for i in used
    ])

    n_bins = tensor.shape[1]
    accs = np.zeros(n_bins)
    sems = np.zeros(n_bins)

    for b in range(n_bins):
        X = tensor[:, b, :]
        # Skip if no variance
        if np.std(X) < 1e-10:
            accs[b] = 0.5
            sems[b] = 0
            continue
        a, s = decode_at_timebin(X, labels, n_folds=N_FOLDS)
        accs[b] = a
        sems[b] = s

    # Chance level via label permutation
    rng = np.random.default_rng(42)
    chance_accs = np.zeros((N_PERM, n_bins))
    for p in range(N_PERM):
        y_perm = labels.copy()
        rng.shuffle(y_perm)
        for b in range(n_bins):
            X = tensor[:, b, :]
            if np.std(X) < 1e-10:
                chance_accs[p, b] = 0.5
                continue
            a, _ = decode_at_timebin(X, y_perm, n_folds=N_FOLDS, random_state=p)
            chance_accs[p, b] = a

    chance_95 = np.percentile(chance_accs, 95, axis=0)

    # Onset latency: first bin where accuracy > chance_95 in post-change window
    post_mask = bin_centers >= 0
    onset = np.nan
    for b in np.where(post_mask)[0]:
        if accs[b] > chance_95[b]:
            onset = bin_centers[b]
            break

    # Transfer decoding: apply Hit-Miss decoder to FA trials
    fa_prob_hit = np.full(n_bins, np.nan)
    n_fa = 0
    if len(fa_idx) >= 3:
        fa_tensor, _, _ = build_population_tensor(
            sess, good_ids, event_name="Change_ON",
            window=WINDOW, bin_size=BIN_SIZE,
            trial_indices=fa_idx,
        )
        if fa_tensor.shape[0] >= 3 and fa_tensor.shape[2] >= MIN_UNITS:
            # Normalize FA tensor to same baseline as training data
            fa_tensor = compute_zscore_normalized(fa_tensor, bin_centers, BASELINE_WINDOW)
            n_fa = fa_tensor.shape[0]
            for b in range(n_bins):
                X_train = tensor[:, b, :]
                X_fa = fa_tensor[:, b, :]
                if np.std(X_train) < 1e-10:
                    continue
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_fa_s = scaler.transform(X_fa)
                clf = LogisticRegression(
                    C=1.0, penalty="l2", solver="lbfgs", max_iter=500,
                )
                clf.fit(X_train_s, labels)
                probs = clf.predict_proba(X_fa_s)
                hit_class_idx = list(clf.classes_).index(1)
                fa_prob_hit[b] = float(probs[:, hit_class_idx].mean())

    return {
        "bin_centers": bin_centers,
        "accuracy": accs,
        "accuracy_sem": sems,
        "chance_95": chance_95,
        "onset_latency": onset,
        "peak_accuracy": float(np.max(accs[post_mask])) if post_mask.any() else 0.5,
        "n_hit": int(labels.sum()),
        "n_miss": int((~labels.astype(bool)).sum()),
        "n_units": tensor.shape[2],
        "fa_prob_hit": fa_prob_hit,
        "n_fa": n_fa,
    }


def _process_session_worker(args):
    """Module-level worker for ProcessPoolExecutor: load session, run decoding."""
    sname, stage, sidx = args
    try:
        sess = load_session(sname)
    except FileNotFoundError:
        return sname, stage, sidx, None, "not found"
    r = decode_session(sess, sname)
    del sess
    gc.collect()
    if r is not None:
        r["stage"] = stage
        r["session_idx"] = sidx
        msg = (f"peak={r['peak_accuracy']:.2f}, onset={r['onset_latency']:.3f}s"
               if np.isfinite(r["onset_latency"])
               else f"peak={r['peak_accuracy']:.2f}, no onset")
    else:
        msg = "insufficient data"
    return sname, stage, sidx, r, msg


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Parallel worker processes for session-level decoding "
                             "(default: 1 = sequential). Each worker loads and processes "
                             "one session independently.")
    args = parser.parse_args()

    print("[04a] Hit vs Miss decoding...")
    manifest = load_staging_manifest(qc_only=True)

    tasks = [
        (int(row["session_name"]), row["stage"], row["session_idx"])
        for _, row in manifest.iterrows()
    ]

    # ── Decode each session ───────────────────────────────────────────
    results = {}
    if args.n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        print(f"  Using {args.n_workers} parallel workers")
        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            for sname, stage, sidx, r, msg in ex.map(_process_session_worker, tasks):
                print(f"  Session {sname} ({stage})... {msg}")
                if r is not None:
                    results[sname] = r
    else:
        for sname, stage, sidx in tasks:
            print(f"  Session {sname} ({stage})...", end=" ", flush=True)
            try:
                sess = load_session(sname)
            except FileNotFoundError:
                print("not found")
                continue
            r = decode_session(sess, sname)
            if r is not None:
                r["stage"] = stage
                r["session_idx"] = sidx
                results[sname] = r
                print(f"peak={r['peak_accuracy']:.2f}, onset={r['onset_latency']:.3f}s"
                      if np.isfinite(r['onset_latency'])
                      else f"peak={r['peak_accuracy']:.2f}, no onset")
            else:
                print("insufficient data")
            del sess
            gc.collect()

    print(f"\n  Decoded {len(results)} sessions")

    if not results:
        print("  No results. Exiting.")
        return

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

    expert = {k: v for k, v in results.items() if v["stage"] == "Expert"}

    # Panel A: Single Expert session
    ax_a = fig.add_subplot(gs[0, 0])
    if expert:
        best = max(expert.keys(), key=lambda k: expert[k]["peak_accuracy"])
        r = expert[best]
        bc = r["bin_centers"]
        ax_a.plot(bc, smooth_psth(r["accuracy"], BIN_SIZE, 15.0),
                  color="#2196F3", linewidth=2, label="Accuracy")
        ax_a.fill_between(bc,
                          smooth_psth(r["accuracy"] - r["accuracy_sem"], BIN_SIZE, 15.0),
                          smooth_psth(r["accuracy"] + r["accuracy_sem"], BIN_SIZE, 15.0),
                          color="#2196F3", alpha=0.2)
        ax_a.plot(bc, smooth_psth(r["chance_95"], BIN_SIZE, 15.0),
                  color="gray", linewidth=1, linestyle="--", label="95% chance")
        ax_a.axhline(0.5, color="gray", linewidth=0.5, linestyle=":")
        ax_a.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_a.set_ylim(0.35, 1.0)
        ax_a.set_xlabel("Time from Change_ON (s)")
        ax_a.set_ylabel("Accuracy")
        ax_a.set_title(f"A. Expert session {best} (n={r['n_units']} units)")
        ax_a.legend(fontsize=8)

    # Panel B: Grand-average Expert
    ax_b = fig.add_subplot(gs[0, 1])
    if expert:
        ref_bc = list(expert.values())[0]["bin_centers"]
        all_acc = [r["accuracy"] for r in expert.values() if len(r["accuracy"]) == len(ref_bc)]
        all_chance = [r["chance_95"] for r in expert.values() if len(r["chance_95"]) == len(ref_bc)]

        if all_acc:
            mean_acc = np.mean(all_acc, axis=0)
            sem_acc = np.std(all_acc, axis=0) / np.sqrt(len(all_acc))
            mean_chance = np.mean(all_chance, axis=0)

            bc_s = ref_bc
            ax_b.plot(bc_s, smooth_psth(mean_acc, BIN_SIZE, 15.0),
                      color="#2196F3", linewidth=2, label="Mean accuracy")
            ax_b.fill_between(bc_s,
                              smooth_psth(mean_acc - sem_acc, BIN_SIZE, 15.0),
                              smooth_psth(mean_acc + sem_acc, BIN_SIZE, 15.0),
                              color="#2196F3", alpha=0.2)
            ax_b.plot(bc_s, smooth_psth(mean_chance, BIN_SIZE, 15.0),
                      color="gray", linewidth=1, linestyle="--", label="95% chance")
            ax_b.axhline(0.5, color="gray", linewidth=0.5, linestyle=":")
            ax_b.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
            ax_b.set_ylim(0.35, 1.0)
            ax_b.set_title(f"B. Grand-average (n={len(all_acc)} Expert sessions)")
        ax_b.set_xlabel("Time from Change_ON (s)")
        ax_b.set_ylabel("Accuracy")
        ax_b.legend(fontsize=8)

    # Panel C: FA transfer decoding – single Expert session
    ax_c = fig.add_subplot(gs[1, 0])
    if expert:
        best = max(expert.keys(), key=lambda k: expert[k]["peak_accuracy"])
        r = expert[best]
        bc = r["bin_centers"]
        fa_ph = r.get("fa_prob_hit")
        if fa_ph is not None and np.any(np.isfinite(fa_ph)):
            ax_c.plot(bc, smooth_psth(fa_ph, BIN_SIZE, 15.0),
                      color=OUTCOME_COLORS["FA"], linewidth=2,
                      label=f"P(Hit) for FA (n={r.get('n_fa', 0)})")
            ax_c.axhline(0.5, color="gray", linewidth=0.5, linestyle=":")
            ax_c.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
            ax_c.set_ylim(0.0, 1.0)
            ax_c.set_xlabel("Time from Change_ON (s)")
            ax_c.set_ylabel("P(Hit)")
            ax_c.set_title(f"C. FA decoded by Hit-Miss model \u2013 session {best}")
            ax_c.legend(fontsize=8)
        else:
            ax_c.text(0.5, 0.5, "No FA data", transform=ax_c.transAxes, ha="center")
            ax_c.set_title("C. FA decoded by Hit-Miss model")

    # Panel D: FA transfer decoding – grand average Expert
    ax_d = fig.add_subplot(gs[1, 1])
    if expert:
        ref_bc = list(expert.values())[0]["bin_centers"]
        all_fa_ph = []
        for r in expert.values():
            fa_ph = r.get("fa_prob_hit")
            if (fa_ph is not None and len(fa_ph) == len(ref_bc)
                    and np.any(np.isfinite(fa_ph))):
                all_fa_ph.append(fa_ph)
        if all_fa_ph:
            mean_fa = np.nanmean(all_fa_ph, axis=0)
            sem_fa = np.nanstd(all_fa_ph, axis=0) / np.sqrt(len(all_fa_ph))
            ax_d.plot(ref_bc, smooth_psth(mean_fa, BIN_SIZE, 15.0),
                      color=OUTCOME_COLORS["FA"], linewidth=2,
                      label="Mean P(Hit) for FA")
            ax_d.fill_between(
                ref_bc,
                smooth_psth(mean_fa - sem_fa, BIN_SIZE, 15.0),
                smooth_psth(mean_fa + sem_fa, BIN_SIZE, 15.0),
                color=OUTCOME_COLORS["FA"], alpha=0.2,
            )
            ax_d.axhline(0.5, color="gray", linewidth=0.5, linestyle=":")
            ax_d.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
            ax_d.set_ylim(0.0, 1.0)
            ax_d.set_title(f"D. FA transfer decoding (n={len(all_fa_ph)} Expert sessions)")
        else:
            ax_d.text(0.5, 0.5, "No FA data", transform=ax_d.transAxes, ha="center")
            ax_d.set_title("D. FA transfer decoding")
        ax_d.set_xlabel("Time from Change_ON (s)")
        ax_d.set_ylabel("P(Hit)")
        ax_d.legend(fontsize=8)

    # Panel E: Onset latency vs session
    ax_e = fig.add_subplot(gs[2, 0])
    add_stage_background(ax_e, manifest)

    sess_list = sorted(results.keys(), key=lambda k: results[k]["session_idx"])
    idxs = [results[k]["session_idx"] for k in sess_list]
    onsets = [results[k]["onset_latency"] for k in sess_list]
    stages = [results[k]["stage"] for k in sess_list]
    colors = [STAGE_COLORS[s] for s in stages]

    finite_mask = [np.isfinite(o) for o in onsets]
    ax_e.scatter(
        [i for i, m in zip(idxs, finite_mask) if m],
        [o for o, m in zip(onsets, finite_mask) if m],
        c=[c for c, m in zip(colors, finite_mask) if m],
        s=60, edgecolors="white", linewidths=0.5, zorder=3,
    )
    # Mark sessions without onset
    ax_e.scatter(
        [i for i, m in zip(idxs, finite_mask) if not m],
        [0.9] * sum(not m for m in finite_mask),
        c=[c for c, m in zip(colors, finite_mask) if not m],
        s=40, marker="x", zorder=3, alpha=0.5,
    )
    ax_e.set_xlabel("Session index")
    ax_e.set_ylabel("Onset latency (s)")
    ax_e.set_title("E. Decoding onset latency across learning")

    # Panel F: Peak accuracy by stage
    ax_f = fig.add_subplot(gs[2, 1])
    stage_peaks = {s: [] for s in STAGE_ORDER}
    for k in sess_list:
        stage_peaks[results[k]["stage"]].append(results[k]["peak_accuracy"])

    box_data = []
    box_pos = []
    box_colors = []
    for i, stage in enumerate(STAGE_ORDER):
        if stage_peaks[stage]:
            box_pos.append(i)
            box_data.append(stage_peaks[stage])
            box_colors.append(STAGE_COLORS[stage])

    if box_data:
        bp = ax_f.boxplot(box_data, positions=box_pos, widths=0.5,
                          patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for pos, vals, color in zip(box_pos, box_data, box_colors):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax_f.scatter(pos + jitter, vals, c=color, s=40,
                         edgecolors="white", linewidths=0.5, zorder=3)
    ax_f.axhline(0.5, color="gray", linewidth=0.5, linestyle=":")
    ax_f.set_xticks(range(len(STAGE_ORDER)))
    ax_f.set_xticklabels(STAGE_ORDER)
    ax_f.set_ylabel("Peak accuracy")
    ax_f.set_title("F. Peak decoding accuracy by stage")

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # Peak accuracy trend
    valid_idx = [(i, p) for i, p in zip(idxs, [results[k]["peak_accuracy"] for k in sess_list])
                 if np.isfinite(p)]
    if len(valid_idx) >= 3:
        x, y = zip(*valid_idx)
        rho, p = spearmanr(x, y)
        stats.append({"test": "peak_acc_vs_session_spearman", "rho": rho, "p": p})

    # Peak accuracy by stage
    valid_groups = [np.array(stage_peaks[s]) for s in STAGE_ORDER if stage_peaks[s]]
    valid_groups = [g for g in valid_groups if len(g) >= 2]
    if len(valid_groups) >= 2:
        h, p = kruskal(*valid_groups)
        stats.append({"test": "peak_acc_kruskal_by_stage", "H": h, "p": p})

    # Onset latency trend
    valid_onsets = [(i, o) for i, o in zip(idxs, onsets) if np.isfinite(o)]
    if len(valid_onsets) >= 3:
        x, y = zip(*valid_onsets)
        rho, p = spearmanr(x, y)
        stats.append({"test": "onset_latency_vs_session_spearman", "rho": rho, "p": p})

    # FA transfer decoding: mean P(Hit) for FA trials in Expert sessions
    if expert:
        from scipy.stats import wilcoxon as _wilcoxon
        ref_bc = list(expert.values())[0]["bin_centers"]
        post_mask_fa = ref_bc >= 0
        expert_fa_mean_ph = []
        for r in expert.values():
            fa_ph = r.get("fa_prob_hit")
            if fa_ph is not None and np.any(np.isfinite(fa_ph)):
                expert_fa_mean_ph.append(float(np.nanmean(fa_ph[post_mask_fa])))
        if len(expert_fa_mean_ph) >= 3:
            try:
                stat_w, p_w = _wilcoxon(np.array(expert_fa_mean_ph) - 0.5)
                stats.append({
                    "test": "fa_transfer_P_hit_vs_0.5_wilcoxon",
                    "W": stat_w, "p": p_w,
                    "mean_P_hit": float(np.mean(expert_fa_mean_ph)),
                    "n": len(expert_fa_mean_ph),
                })
            except ValueError:
                pass

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig18_hit_miss_decoding", "04_decoding")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "04_decoding", "hit_miss_decoding_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: p={row.get('p', 'N/A')}")


if __name__ == "__main__":
    main()
