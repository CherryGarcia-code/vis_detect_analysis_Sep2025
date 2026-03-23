"""Fig 13: Coding direction — time-resolved Hit vs Miss.

The most publishable population analysis: demonstrates that striatal
populations encode a decision-related signal that emerges with learning
and depends on behavioral state.

Produces:
  - Fig 4A: CD projection for a single Expert session (Hit vs Miss lines)
  - Fig 4B: Grand-average CD across Expert sessions
  - Fig 4C: CD effect size vs session index (emergence across learning)
  - Fig 4D: CD in Engaged vs Impulsive HMM states (Expert sessions)

Caches: cache/cd_results/{session_name}_hit_miss_cd.npz
Saves statistics to figures/03_population/coding_direction_stats.csv
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
    STAGE_ORDER, STAGE_COLORS, OUTCOME_COLORS,
    HMM_STATE_ORDER, HMM_STATE_COLORS,
    CACHE_DIR, DEFAULT_BIN_SIZE,
)
from loader import (
    load_staging_manifest, load_session,
    load_hmm_assignments,
)
from utils import get_good_cluster_ids, build_population_tensor, smooth_psth
from plotting import setup_style, save_figure, add_stage_background
from visdetect.analysis.align import align_spikes_to_events, get_event_times_by_trial

setup_style()


# ── Coding direction utilities (replaces archive.coding_direction) ────

def _stratified_kfold_indices(y, n_splits, random_state=0):
    """Stratified K-fold index generator for binary labels."""
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    classes = np.unique(y)
    per_class = {c: np.where(y == c)[0] for c in classes}
    min_count = min(len(v) for v in per_class.values())
    n_splits = max(2, min(n_splits, min_count))
    per_class_folds = {}
    for c, idxs in per_class.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        per_class_folds[c] = np.array_split(idxs, n_splits)
    folds = []
    for k in range(n_splits):
        test_idx = np.concatenate(
            [per_class_folds[c][k] for c in classes if len(per_class_folds[c][k]) > 0]
        )
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx, assume_unique=True)
        folds.append((train_idx, test_idx))
    return folds


def _compute_cd_shrinkage(X, y, reg=1.0):
    """Shrinkage coding direction: inv(Cov + reg*I) * (mu1 - mu0), normalized."""
    mu1 = X[y == 1].mean(axis=0)
    mu0 = X[y == 0].mean(axis=0)
    diff = mu1 - mu0
    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc, rowvar=False)
    cov_reg = cov + reg * np.eye(cov.shape[0])
    try:
        w = np.linalg.solve(cov_reg, diff)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(cov_reg) @ diff
    norm = np.linalg.norm(w)
    return w / norm if norm > 0 else w


def time_resolved_cd(pop, cond_mask, reg=1.0, n_splits=5, n_permutations=200,
                     random_state=0):
    """Compute cross-validated time-resolved coding direction (Hit vs Miss).

    Parameters
    ----------
    pop : ndarray, shape (trials, bins, units)
    cond_mask : boolean array (True = class 1 / Hit)
    reg : float — shrinkage regularization strength
    n_splits : int — stratified K-fold CV splits
    n_permutations : int — label-shuffle permutations for p-values

    Returns
    -------
    dict with keys: cds, proj, pvals, effect, mean0, mean1
    """
    rng = np.random.RandomState(random_state)
    trials, bins, units = pop.shape
    y = cond_mask.astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("Need both classes present in cond_mask")

    folds = _stratified_kfold_indices(y, n_splits=n_splits, random_state=random_state)
    proj = np.zeros((trials, bins), dtype=float)
    cds_accum = np.zeros((bins, units), dtype=float)
    fold_counts = np.zeros(bins, dtype=int)

    for train_idx, test_idx in folds:
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        for b in range(bins):
            X_train = pop[train_idx, b, :]
            X_test = pop[test_idx, b, :]
            y_train = y[train_idx]
            if len(np.unique(y_train)) < 2:
                continue
            cd = _compute_cd_shrinkage(X_train, y_train, reg=reg)
            cds_accum[b] += cd
            fold_counts[b] += 1
            proj[test_idx, b] = X_test @ cd

    cds = cds_accum / np.maximum(fold_counts[:, None], 1)
    mean0 = proj[y == 0].mean(axis=0)
    mean1 = proj[y == 1].mean(axis=0)
    effect = mean1 - mean0

    # Permutation test
    perm_effects = np.zeros((n_permutations, bins), dtype=float)
    for i in range(n_permutations):
        y_perm = y.copy()
        rng.shuffle(y_perm)
        proj_perm = np.zeros((trials, bins), dtype=float)
        for train_idx, test_idx in folds:
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue
            for b in range(bins):
                X_train = pop[train_idx, b, :]
                X_test = pop[test_idx, b, :]
                y_train = y_perm[train_idx]
                if len(np.unique(y_train)) < 2:
                    continue
                cd_perm = _compute_cd_shrinkage(X_train, y_train, reg=reg)
                proj_perm[test_idx, b] = X_test @ cd_perm
        perm_effects[i] = (proj_perm[y_perm == 1].mean(axis=0)
                           - proj_perm[y_perm == 0].mean(axis=0))

    pvals = np.array([
        ((np.abs(perm_effects[:, b]) >= abs(effect[b])).sum() + 1) / (n_permutations + 1)
        for b in range(bins)
    ])

    return {"cds": cds, "proj": proj, "pvals": pvals, "effect": effect,
            "mean0": mean0, "mean1": mean1}

CD_CACHE_DIR = os.path.join(CACHE_DIR, "cd_results")
os.makedirs(CD_CACHE_DIR, exist_ok=True)

# Analysis parameters
WINDOW = (-0.5, 1.0)      # relative to Change_ON
BIN_SIZE = DEFAULT_BIN_SIZE
MIN_UNITS = 10             # minimum units per session
MIN_TRIALS_PER_CLASS = 8   # minimum hit or miss trials
N_PERM = 200
N_SPLITS = 5
LICK_WINDOW = (-1.0, 0.5)  # relative to lick onset


def run_cd_for_session(sess, session_name, hmm_assign=None, force=False):
    """Compute coding direction for a single session. Returns dict or None."""
    cache_file = os.path.join(CD_CACHE_DIR, f"{session_name}_hit_miss_cd.npz")
    if os.path.exists(cache_file) and not force:
        data = dict(np.load(cache_file, allow_pickle=True))
        if "cds" in data:  # current cache format (v5 with CD weights)
            return data
        # stale cache format; recompute

    # Get good clusters
    good_ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
    if len(good_ids) < MIN_UNITS:
        return None

    # Build population tensor: Change_ON aligned, hit+miss on GO trials only.
    # change_size > 1.01 excludes catch trials where a lick is an SDT false alarm,
    # not a true detection Hit.
    trials = sess.trials
    go_hit_indices = [
        i for i, t in enumerate(trials)
        if getattr(t, "trialoutcome", None) == "Hit"
        and (getattr(t, "change_size", None) or 1.0) > 1.01
    ]
    go_miss_indices = [
        i for i, t in enumerate(trials)
        if getattr(t, "trialoutcome", None) == "Miss"
        and (getattr(t, "change_size", None) or 1.0) > 1.01
    ]

    if len(go_hit_indices) < MIN_TRIALS_PER_CLASS or len(go_miss_indices) < MIN_TRIALS_PER_CLASS:
        return None

    allowed_indices = set(go_hit_indices + go_miss_indices)
    tensor, bin_centers, used_indices = build_population_tensor(
        sess, good_ids,
        event_name="Change_ON",
        window=WINDOW,
        bin_size=BIN_SIZE,
        trial_indices=list(allowed_indices),
    )

    if tensor.shape[0] < 2 * MIN_TRIALS_PER_CLASS or tensor.shape[2] < MIN_UNITS:
        return None

    # Build condition mask (True = hit, False = miss)
    cond_mask = np.array([
        getattr(trials[i], "trialoutcome", None) == "Hit"
        for i in used_indices
    ])

    n_hit = int(cond_mask.sum())
    n_miss = int((~cond_mask).sum())
    if n_hit < MIN_TRIALS_PER_CLASS or n_miss < MIN_TRIALS_PER_CLASS:
        return None

    # Run time-resolved CD
    print(f"      CD: {tensor.shape[0]} trials ({n_hit} hit, {n_miss} miss), "
          f"{tensor.shape[2]} units, {tensor.shape[1]} bins")

    result = time_resolved_cd(
        tensor, cond_mask,
        reg=1.0,
        n_splits=N_SPLITS,
        n_permutations=N_PERM,
        random_state=42,
    )

    # ── Average CD axis from post-change bins [0, 0.5]s ───────────────
    post_mask = (bin_centers >= 0) & (bin_centers <= 0.5)
    avg_cd = result["cds"][post_mask].mean(axis=0)
    cd_norm = np.linalg.norm(avg_cd)
    if cd_norm > 0:
        avg_cd = avg_cd / cd_norm

    # ── Change-aligned projection: True Hit vs True FA ────────────────
    # Project hit trials onto avg CD axis (non-CV, for apples-to-apples with FA)
    hit_change_proj = tensor[cond_mask] @ avg_cd        # (n_hit, n_bins)
    hit_change_proj_mean = hit_change_proj.mean(axis=0)
    hit_change_proj_sem = hit_change_proj.std(axis=0) / np.sqrt(hit_change_proj.shape[0])

    # Build change-aligned tensor for catch-trial FAs and project
    fa_change_indices = [
        i for i, t in enumerate(trials)
        if getattr(t, "trialoutcome", None) == "Hit"
        and (getattr(t, "change_size", None) or 1.0) <= 1.01
    ]
    cr_change_indices = [
        i for i, t in enumerate(trials)
        if getattr(t, "trialoutcome", None) == "Miss"
        and (getattr(t, "change_size", None) or 1.0) <= 1.01
    ]
    fa_change_proj_mean = np.array([])
    fa_change_proj_sem = np.array([])
    n_fa_change = 0
    cr_change_proj_mean = np.array([])
    cr_change_proj_sem = np.array([])
    n_cr_change = 0

    if len(fa_change_indices) >= 3:
        fa_tensor, _, fa_used = build_population_tensor(
            sess, good_ids, event_name="Change_ON",
            window=WINDOW, bin_size=BIN_SIZE,
            trial_indices=fa_change_indices,
        )
        if fa_tensor.shape[0] >= 3 and fa_tensor.shape[2] == len(avg_cd):
            fa_proj = fa_tensor @ avg_cd              # (n_fa, n_bins)
            fa_change_proj_mean = fa_proj.mean(axis=0)
            fa_change_proj_sem = fa_proj.std(axis=0) / np.sqrt(fa_proj.shape[0])
            n_fa_change = fa_tensor.shape[0]

    if len(cr_change_indices) >= 3:
        cr_tensor, _, cr_used = build_population_tensor(
            sess, good_ids, event_name="Change_ON",
            window=WINDOW, bin_size=BIN_SIZE,
            trial_indices=cr_change_indices,
        )
        if cr_tensor.shape[0] >= 3 and cr_tensor.shape[2] == len(avg_cd):
            cr_proj = cr_tensor @ avg_cd
            cr_change_proj_mean = cr_proj.mean(axis=0)
            cr_change_proj_sem = cr_proj.std(axis=0) / np.sqrt(cr_proj.shape[0])
            n_cr_change = cr_tensor.shape[0]

    # ── Lick-aligned projection: True Hit vs True FA ──────────────────
    change_on_times = get_event_times_by_trial(sess, "Change_ON")
    cluster_map = {int(c.cluster_id): c for c in sess.clusters}
    n_lick_bins = int(np.round((LICK_WINDOW[1] - LICK_WINDOW[0]) / BIN_SIZE))

    def _extract_rt(trial):
        """Get RT from reactiontimes dict; key is 'RT' in this dataset."""
        rt_dict = getattr(trial, "reactiontimes", {}) or {}
        rt = rt_dict.get("RT", rt_dict.get("Hit", rt_dict.get("hit", np.nan)))
        try:
            return float(rt)
        except (TypeError, ValueError):
            return np.nan

    # ── Collect lick times for True Hits (go-trial hits used in CD) ───
    hit_lick_times, hit_rts = [], []
    for i in used_indices:
        t = trials[i]
        if getattr(t, "trialoutcome", None) != "Hit":
            continue
        rt = _extract_rt(t)
        if np.isfinite(rt) and i < len(change_on_times) and np.isfinite(change_on_times[i]):
            hit_lick_times.append(float(change_on_times[i]) + rt)
            hit_rts.append(rt)

    # ── Collect lick times for True FAs (catch-trial licks) ───────────
    fa_lick_times, fa_rts = [], []
    for i, t in enumerate(trials):
        if getattr(t, "trialoutcome", None) != "Hit":
            continue
        cs = getattr(t, "change_size", None) or 1.0
        if cs > 1.01:  # go trial → not an FA
            continue
        rt = _extract_rt(t)
        if np.isfinite(rt) and i < len(change_on_times) and np.isfinite(change_on_times[i]):
            fa_lick_times.append(float(change_on_times[i]) + rt)
            fa_rts.append(rt)

    def _lick_project(lick_event_times):
        """Build lick-aligned tensor and project onto avg CD axis."""
        mats = []
        bc_out = np.array([])
        for cid in good_ids:
            c = cluster_map.get(int(cid))
            if c is None:
                mats.append(np.zeros((len(lick_event_times), n_lick_bins)))
                continue
            mat, bc_out = align_spikes_to_events(
                c.spike_times, lick_event_times, window=LICK_WINDOW, bin_size=BIN_SIZE
            )
            mats.append(mat)
        tensor_l = np.stack(mats, axis=2)   # (n_trials, n_bins, n_units)
        proj_l = tensor_l @ avg_cd          # (n_trials, n_bins)
        return proj_l.mean(axis=0), proj_l.std(axis=0) / np.sqrt(proj_l.shape[0]), bc_out

    # Defaults
    lick_proj_mean = np.array([])
    lick_proj_sem = np.array([])
    lick_bc = np.array([])
    median_rt = np.nan
    fa_lick_proj_mean = np.array([])
    fa_lick_proj_sem = np.array([])
    n_fa = 0
    median_rt_fa = np.nan

    if len(hit_lick_times) >= MIN_TRIALS_PER_CLASS:
        median_rt = float(np.median(hit_rts))
        lick_proj_mean, lick_proj_sem, lick_bc = _lick_project(hit_lick_times)

    if len(fa_lick_times) >= 3:  # looser threshold for FAs (fewer trials)
        median_rt_fa = float(np.median(fa_rts))
        n_fa = len(fa_lick_times)
        fa_lick_proj_mean, fa_lick_proj_sem, lick_bc_fa = _lick_project(fa_lick_times)
        # Ensure lick_bc is set even if no Hit licks
        if len(lick_bc) == 0:
            lick_bc = lick_bc_fa

    # Cache
    np.savez(cache_file,
             bin_centers=bin_centers,
             cds=result["cds"],
             avg_cd=avg_cd,
             cluster_ids=np.array(good_ids),
             effect=result["effect"],
             pvals=result["pvals"],
             mean_hit=result["mean1"],
             mean_miss=result["mean0"],
             n_hit=n_hit,
             n_miss=n_miss,
             n_units=tensor.shape[2],
             hit_change_proj_mean=hit_change_proj_mean,
             hit_change_proj_sem=hit_change_proj_sem,
             fa_change_proj_mean=fa_change_proj_mean,
             fa_change_proj_sem=fa_change_proj_sem,
             n_fa_change=n_fa_change,
             cr_change_proj_mean=cr_change_proj_mean,
             cr_change_proj_sem=cr_change_proj_sem,
             n_cr_change=n_cr_change,
             lick_proj_mean=lick_proj_mean,
             lick_proj_sem=lick_proj_sem,
             lick_bin_centers=lick_bc,
             median_rt=median_rt,
             fa_lick_proj_mean=fa_lick_proj_mean,
             fa_lick_proj_sem=fa_lick_proj_sem,
             n_fa=n_fa,
             median_rt_fa=median_rt_fa,
             )

    return {
        "bin_centers": bin_centers,
        "effect": result["effect"],
        "pvals": result["pvals"],
        "mean_hit": result["mean1"],
        "mean_miss": result["mean0"],
        "n_hit": n_hit,
        "n_miss": n_miss,
        "n_units": tensor.shape[2],
        "hit_change_proj_mean": hit_change_proj_mean,
        "hit_change_proj_sem": hit_change_proj_sem,
        "fa_change_proj_mean": fa_change_proj_mean,
        "fa_change_proj_sem": fa_change_proj_sem,
        "n_fa_change": n_fa_change,
        "cr_change_proj_mean": cr_change_proj_mean,
        "cr_change_proj_sem": cr_change_proj_sem,
        "n_cr_change": n_cr_change,
        "lick_proj_mean": lick_proj_mean,
        "lick_proj_sem": lick_proj_sem,
        "lick_bin_centers": lick_bc,
        "median_rt": median_rt,
        "fa_lick_proj_mean": fa_lick_proj_mean,
        "fa_lick_proj_sem": fa_lick_proj_sem,
        "n_fa": n_fa,
        "median_rt_fa": median_rt_fa,
    }


def _process_session_worker(args):
    """Module-level worker for ProcessPoolExecutor: load session, compute CD."""
    sname, stage, sidx, force = args
    try:
        sess = load_session(sname)
    except FileNotFoundError:
        return sname, stage, sidx, None, "not found"
    result = run_cd_for_session(sess, sname, force=force)
    del sess
    gc.collect()
    if result is not None:
        result["stage"] = stage
        result["session_idx"] = sidx
        peak_effect = float(np.max(np.abs(result["effect"])))
        min_p = float(np.min(result["pvals"]))
        msg = f"peak |effect|={peak_effect:.3f}, min_p={min_p:.3e}"
    else:
        msg = "insufficient data"
    return sname, stage, sidx, result, msg


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Parallel worker processes for session-level CD computation "
                             "(default: 1 = sequential). Each worker loads and processes "
                             "one session independently.")
    parser.add_argument("--force", action="store_true",
                        help="Force recomputation, ignoring cached results.")
    args = parser.parse_args()

    print("[03a] Coding direction analysis (Hit vs Miss)...")
    manifest = load_staging_manifest(qc_only=True)
    hmm_assign = load_hmm_assignments()

    tasks = [
        (int(row["session_name"]), row["stage"], row["session_idx"], args.force)
        for _, row in manifest.iterrows()
    ]

    # ── Compute CD for each session ───────────────────────────────────
    cd_results = {}
    if args.n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        print(f"  Using {args.n_workers} parallel workers")
        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            for sname, stage, sidx, result, msg in ex.map(_process_session_worker, tasks):
                print(f"  Session {sname} ({stage}, idx={sidx})... {msg}")
                if result is not None:
                    cd_results[sname] = result
    else:
        for sname, stage, sidx in tasks:
            print(f"  Session {sname} ({stage}, idx={sidx})...", end=" ")
            try:
                sess = load_session(sname)
            except FileNotFoundError:
                print("not found, skipping")
                continue
            result = run_cd_for_session(sess, sname, force=args.force)
            if result is not None:
                result["stage"] = stage
                result["session_idx"] = sidx
                cd_results[sname] = result
                peak_effect = np.max(np.abs(result["effect"]))
                min_p = np.min(result["pvals"])
                print(f"peak |effect|={peak_effect:.3f}, min_p={min_p:.3e}")
            else:
                print("insufficient data")
            del sess
            gc.collect()

    print(f"\n  CD computed for {len(cd_results)} sessions")

    if len(cd_results) == 0:
        print("  No CD results. Exiting.")
        return

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(4, 2, hspace=0.4, wspace=0.3)

    # Helper to safely extract scalar from cache (npz loads scalars as 0-d arrays)
    def _scalar(val, default=np.nan):
        if isinstance(val, np.ndarray):
            return float(val) if val.ndim == 0 else default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    # ── Per-session baseline z-scoring for grand averages ─────────────
    # Puts all sessions on the same scale before cross-session averaging.
    # Baseline windows: pre-change for change-aligned, early window for lick.
    CHANGE_BL = (-0.5, -0.1)   # baseline for change-aligned panels
    LICK_BL   = (-1.0, -0.7)   # baseline for lick-aligned panels

    def _zscore_baseline(trace, bin_centers, bl_window):
        """Z-score a trace relative to a baseline window."""
        bl_mask = (bin_centers >= bl_window[0]) & (bin_centers < bl_window[1])
        if bl_mask.sum() < 2:
            return trace  # can't z-score with < 2 baseline bins
        bl = trace[bl_mask]
        mu, sd = bl.mean(), bl.std()
        if sd < 1e-12:
            return trace - mu
        return (trace - mu) / sd

    # ── Row 1: Change-aligned CD ──────────────────────────────────────
    # Panel A: Single Expert session CD (change-aligned)
    ax_a = fig.add_subplot(gs[0, 0])
    expert_results = {k: v for k, v in cd_results.items() if v["stage"] == "Expert"}

    best_session = None
    if expert_results:
        # Pick the session with largest peak effect
        best_session = max(expert_results.keys(),
                           key=lambda k: np.max(np.abs(expert_results[k]["effect"])))
        r = expert_results[best_session]
        bc = r["bin_centers"]

        ax_a.plot(bc, smooth_psth(r["mean_hit"], BIN_SIZE, 15.0),
                  color=OUTCOME_COLORS["Hit"], linewidth=2, label=f"Hit (n={r['n_hit']})")
        ax_a.plot(bc, smooth_psth(r["mean_miss"], BIN_SIZE, 15.0),
                  color=OUTCOME_COLORS["Miss"], linewidth=2, label=f"Miss (n={r['n_miss']})")
        ax_a.axvline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
        # Annotate median RT
        mrt = _scalar(r.get("median_rt"))
        if np.isfinite(mrt):
            ax_a.axvline(mrt, color="gray", linestyle=":", linewidth=1.2,
                         label=f"median RT ({mrt:.2f} s)")
        ax_a.fill_between(bc, ax_a.get_ylim()[0], ax_a.get_ylim()[1],
                          where=r["pvals"] < 0.05, alpha=0.1, color="gold")
        ax_a.set_xlabel("Time from Change_ON (s)")
        ax_a.set_ylabel("CD projection (a.u.)")
        ax_a.set_title(f"A. Expert session {best_session} — change-aligned "
                        f"(n={r['n_units']} units)")
        ax_a.legend(fontsize=8)
    else:
        ax_a.text(0.5, 0.5, "No Expert sessions", transform=ax_a.transAxes, ha="center")
        ax_a.set_title("A. Single Expert session — change-aligned")

    # Panel B: Grand-average across Expert sessions (change-aligned, z-scored)
    ax_b = fig.add_subplot(gs[0, 1])
    if expert_results:
        ref_bc = list(expert_results.values())[0]["bin_centers"]
        all_hit = []
        all_miss = []
        for r in expert_results.values():
            if len(r["mean_hit"]) == len(ref_bc):
                hit_sm = smooth_psth(r["mean_hit"], BIN_SIZE, 15.0)
                miss_sm = smooth_psth(r["mean_miss"], BIN_SIZE, 15.0)
                all_hit.append(_zscore_baseline(hit_sm, ref_bc, CHANGE_BL))
                all_miss.append(_zscore_baseline(miss_sm, ref_bc, CHANGE_BL))

        if all_hit:
            hit_mean = np.mean(all_hit, axis=0)
            hit_sem = np.std(all_hit, axis=0) / np.sqrt(len(all_hit))
            miss_mean = np.mean(all_miss, axis=0)
            miss_sem = np.std(all_miss, axis=0) / np.sqrt(len(all_miss))

            ax_b.plot(ref_bc, hit_mean, color=OUTCOME_COLORS["Hit"], linewidth=2, label="Hit")
            ax_b.fill_between(ref_bc, hit_mean - hit_sem, hit_mean + hit_sem,
                              color=OUTCOME_COLORS["Hit"], alpha=0.2)
            ax_b.plot(ref_bc, miss_mean, color=OUTCOME_COLORS["Miss"], linewidth=2, label="Miss")
            ax_b.fill_between(ref_bc, miss_mean - miss_sem, miss_mean + miss_sem,
                              color=OUTCOME_COLORS["Miss"], alpha=0.2)
            ax_b.axvline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
            # Grand median RT
            all_mrt = [_scalar(r.get("median_rt")) for r in expert_results.values()]
            all_mrt = [x for x in all_mrt if np.isfinite(x)]
            if all_mrt:
                grand_mrt = float(np.median(all_mrt))
                ax_b.axvline(grand_mrt, color="gray", linestyle=":", linewidth=1.2,
                             label=f"median RT ({grand_mrt:.2f} s)")
            ax_b.set_title(f"B. Grand-average CD — change-aligned "
                           f"(n={len(all_hit)} Expert sessions, z-scored)")
        else:
            ax_b.set_title("B. Grand-average CD — change-aligned")

        ax_b.set_xlabel("Time from Change_ON (s)")
        ax_b.set_ylabel("CD projection (z-score vs baseline)")
        ax_b.legend(fontsize=8)

    # ── Row 2: Change-aligned True Hit vs True FA ─────────────────────
    # Panel C: Single Expert session, change-aligned: True Hit vs True FA
    ax_c = fig.add_subplot(gs[1, 0])
    if expert_results and best_session is not None:
        r = expert_results[best_session]
        bc = r["bin_centers"]
        hcp = r.get("hit_change_proj_mean", np.array([]))
        fcp = r.get("fa_change_proj_mean", np.array([]))
        has_hit_c = isinstance(hcp, np.ndarray) and len(hcp) == len(bc)
        has_fa_c = isinstance(fcp, np.ndarray) and len(fcp) > 0 and len(fcp) == len(bc)

        if has_hit_c or has_fa_c:
            if has_hit_c:
                sm_h = smooth_psth(hcp, BIN_SIZE, 15.0)
                sem_h = smooth_psth(r.get("hit_change_proj_sem", np.zeros_like(hcp)),
                                    BIN_SIZE, 15.0)
                ax_c.plot(bc, sm_h, color=OUTCOME_COLORS["Hit"], linewidth=2,
                          label=f"True Hit (n={r['n_hit']})")
                ax_c.fill_between(bc, sm_h - sem_h, sm_h + sem_h,
                                  color=OUTCOME_COLORS["Hit"], alpha=0.2)
            if has_fa_c:
                n_fa_c = int(_scalar(r.get("n_fa_change", 0), 0))
                sm_f = smooth_psth(fcp, BIN_SIZE, 15.0)
                sem_f = smooth_psth(r.get("fa_change_proj_sem", np.zeros_like(fcp)),
                                    BIN_SIZE, 15.0)
                ax_c.plot(bc, sm_f, color=OUTCOME_COLORS["FA"], linewidth=2,
                          label=f"True FA (n={n_fa_c})")
                ax_c.fill_between(bc, sm_f - sem_f, sm_f + sem_f,
                                  color=OUTCOME_COLORS["FA"], alpha=0.2)
            # Correct Rejection (catch-trial no-lick)
            crp = r.get("cr_change_proj_mean", np.array([]))
            has_cr_c = isinstance(crp, np.ndarray) and len(crp) > 0 and len(crp) == len(bc)
            if has_cr_c:
                n_cr_c = int(_scalar(r.get("n_cr_change", 0), 0))
                sm_cr = smooth_psth(crp, BIN_SIZE, 15.0)
                sem_cr = smooth_psth(r.get("cr_change_proj_sem", np.zeros_like(crp)),
                                     BIN_SIZE, 15.0)
                ax_c.plot(bc, sm_cr, color=OUTCOME_COLORS["CR"], linewidth=2,
                          label=f"True CR (n={n_cr_c})")
                ax_c.fill_between(bc, sm_cr - sem_cr, sm_cr + sem_cr,
                                  color=OUTCOME_COLORS["CR"], alpha=0.2)
            ax_c.axvline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
            mrt = _scalar(r.get("median_rt"))
            if np.isfinite(mrt):
                ax_c.axvline(mrt, color="gray", linestyle=":", linewidth=1.2,
                             label=f"median RT ({mrt:.2f} s)")
            ax_c.set_title(f"C. Expert session {best_session} — change-aligned "
                           f"SDT outcomes (n={r['n_units']} units)")
        else:
            ax_c.text(0.5, 0.5, "Data unavailable", transform=ax_c.transAxes, ha="center")
            ax_c.set_title("C. Single Expert — change-aligned Hit vs FA")
    else:
        ax_c.text(0.5, 0.5, "No Expert sessions", transform=ax_c.transAxes, ha="center")
        ax_c.set_title("C. Single Expert — change-aligned Hit vs FA")
    ax_c.set_xlabel("Time from Change_ON (s)")
    ax_c.set_ylabel("Projection onto avg CD axis (a.u.)")
    ax_c.legend(fontsize=8)

    # Panel D: Grand-average change-aligned True Hit vs True FA (z-scored)
    ax_d = fig.add_subplot(gs[1, 1])
    if expert_results:
        ref_bc = list(expert_results.values())[0]["bin_centers"]
        all_hit_ch, all_fa_ch, all_cr_ch = [], [], []
        for r in expert_results.values():
            hcp = r.get("hit_change_proj_mean", np.array([]))
            fcp = r.get("fa_change_proj_mean", np.array([]))
            crp = r.get("cr_change_proj_mean", np.array([]))
            if isinstance(hcp, np.ndarray) and len(hcp) == len(ref_bc):
                all_hit_ch.append(
                    _zscore_baseline(smooth_psth(hcp, BIN_SIZE, 15.0), ref_bc, CHANGE_BL))
            if isinstance(fcp, np.ndarray) and len(fcp) > 0 and len(fcp) == len(ref_bc):
                all_fa_ch.append(
                    _zscore_baseline(smooth_psth(fcp, BIN_SIZE, 15.0), ref_bc, CHANGE_BL))
            if isinstance(crp, np.ndarray) and len(crp) > 0 and len(crp) == len(ref_bc):
                all_cr_ch.append(
                    _zscore_baseline(smooth_psth(crp, BIN_SIZE, 15.0), ref_bc, CHANGE_BL))

        plotted_d = False
        if all_hit_ch:
            h_mean = np.mean(all_hit_ch, axis=0)
            h_sem = np.std(all_hit_ch, axis=0) / np.sqrt(len(all_hit_ch))
            ax_d.plot(ref_bc, h_mean, color=OUTCOME_COLORS["Hit"], linewidth=2,
                      label=f"True Hit (n={len(all_hit_ch)} sess)")
            ax_d.fill_between(ref_bc, h_mean - h_sem, h_mean + h_sem,
                              color=OUTCOME_COLORS["Hit"], alpha=0.2)
            plotted_d = True
        if all_fa_ch:
            f_mean = np.mean(all_fa_ch, axis=0)
            f_sem = np.std(all_fa_ch, axis=0) / np.sqrt(len(all_fa_ch))
            ax_d.plot(ref_bc, f_mean, color=OUTCOME_COLORS["FA"], linewidth=2,
                      label=f"True FA (n={len(all_fa_ch)} sess)")
            ax_d.fill_between(ref_bc, f_mean - f_sem, f_mean + f_sem,
                              color=OUTCOME_COLORS["FA"], alpha=0.2)
            plotted_d = True
        if all_cr_ch:
            cr_mean = np.mean(all_cr_ch, axis=0)
            cr_sem = np.std(all_cr_ch, axis=0) / np.sqrt(len(all_cr_ch))
            ax_d.plot(ref_bc, cr_mean, color=OUTCOME_COLORS["CR"], linewidth=2,
                      label=f"True CR (n={len(all_cr_ch)} sess)")
            ax_d.fill_between(ref_bc, cr_mean - cr_sem, cr_mean + cr_sem,
                              color=OUTCOME_COLORS["CR"], alpha=0.2)
            plotted_d = True
        if plotted_d:
            ax_d.axvline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
            all_mrt = [_scalar(r.get("median_rt")) for r in expert_results.values()]
            all_mrt = [x for x in all_mrt if np.isfinite(x)]
            if all_mrt:
                grand_mrt = float(np.median(all_mrt))
                ax_d.axvline(grand_mrt, color="gray", linestyle=":", linewidth=1.2,
                             label=f"median RT ({grand_mrt:.2f} s)")
            ax_d.set_title(f"D. Grand-average — change-aligned "
                           f"SDT outcomes (z-scored)")
        else:
            ax_d.set_title("D. Grand-average — change-aligned SDT outcomes")
    else:
        ax_d.set_title("D. Grand-average — change-aligned SDT outcomes")
    ax_d.set_xlabel("Time from Change_ON (s)")
    ax_d.set_ylabel("Projection onto avg CD axis (z-score vs baseline)")
    ax_d.legend(fontsize=8)

    # ── Row 3: Lick-aligned CD projection ─────────────────────────────
    # Panel E: Single Expert session, lick-aligned: True Hit vs True FA
    ax_e = fig.add_subplot(gs[2, 0])
    if expert_results and best_session is not None:
        r = expert_results[best_session]
        lpm = r.get("lick_proj_mean", np.array([]))
        lbc = r.get("lick_bin_centers", np.array([]))
        has_hit = isinstance(lpm, np.ndarray) and len(lpm) > 0 and len(lbc) == len(lpm)
        fa_lpm = r.get("fa_lick_proj_mean", np.array([]))
        has_fa = isinstance(fa_lpm, np.ndarray) and len(fa_lpm) > 0 and len(fa_lpm) == len(lbc)

        if has_hit or has_fa:
            if has_hit:
                sm = smooth_psth(lpm, BIN_SIZE, 15.0)
                sem = smooth_psth(r.get("lick_proj_sem", np.zeros_like(lpm)), BIN_SIZE, 15.0)
                ax_e.plot(lbc, sm, color=OUTCOME_COLORS["Hit"], linewidth=2,
                          label=f"True Hit (n={r['n_hit']})")
                ax_e.fill_between(lbc, sm - sem, sm + sem,
                                  color=OUTCOME_COLORS["Hit"], alpha=0.2)
            if has_fa:
                sm_fa = smooth_psth(fa_lpm, BIN_SIZE, 15.0)
                sem_fa = smooth_psth(r.get("fa_lick_proj_sem", np.zeros_like(fa_lpm)), BIN_SIZE, 15.0)
                n_fa_val = int(_scalar(r.get("n_fa", 0), 0))
                ax_e.plot(lbc, sm_fa, color=OUTCOME_COLORS["FA"], linewidth=2,
                          label=f"True FA (n={n_fa_val})")
                ax_e.fill_between(lbc, sm_fa - sem_fa, sm_fa + sem_fa,
                                  color=OUTCOME_COLORS["FA"], alpha=0.2)
            ax_e.axvline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5,
                         label="Lick onset")
            ax_e.set_title(f"E. Expert session {best_session} — lick-aligned "
                           f"(n={r['n_units']} units)")
        else:
            ax_e.text(0.5, 0.5, "Lick data unavailable", transform=ax_e.transAxes,
                      ha="center")
            ax_e.set_title("E. Single Expert — lick-aligned")
    else:
        ax_e.text(0.5, 0.5, "No Expert sessions", transform=ax_e.transAxes, ha="center")
        ax_e.set_title("E. Single Expert — lick-aligned")
    ax_e.set_xlabel("Time from lick onset (s)")
    ax_e.set_ylabel("Projection onto avg CD axis (a.u.)")
    ax_e.legend(fontsize=8)

    # Panel F: Grand-average lick-aligned: True Hit vs True FA (z-scored)
    ax_f = fig.add_subplot(gs[2, 1])
    if expert_results:
        all_hit_lick, all_fa_lick = [], []
        ref_lbc = None
        for r in expert_results.values():
            lpm = r.get("lick_proj_mean", np.array([]))
            lbc = r.get("lick_bin_centers", np.array([]))
            fa_lpm = r.get("fa_lick_proj_mean", np.array([]))
            if isinstance(lbc, np.ndarray) and len(lbc) > 0:
                if ref_lbc is None:
                    ref_lbc = lbc
                if isinstance(lpm, np.ndarray) and len(lpm) == len(ref_lbc):
                    all_hit_lick.append(
                        _zscore_baseline(smooth_psth(lpm, BIN_SIZE, 15.0), ref_lbc, LICK_BL))
                if isinstance(fa_lpm, np.ndarray) and len(fa_lpm) == len(ref_lbc):
                    all_fa_lick.append(
                        _zscore_baseline(smooth_psth(fa_lpm, BIN_SIZE, 15.0), ref_lbc, LICK_BL))

        plotted = False
        if all_hit_lick and ref_lbc is not None:
            hit_grand = np.mean(all_hit_lick, axis=0)
            hit_sem = np.std(all_hit_lick, axis=0) / np.sqrt(len(all_hit_lick))
            ax_f.plot(ref_lbc, hit_grand, color=OUTCOME_COLORS["Hit"], linewidth=2,
                      label=f"True Hit (n={len(all_hit_lick)} sess)")
            ax_f.fill_between(ref_lbc, hit_grand - hit_sem, hit_grand + hit_sem,
                              color=OUTCOME_COLORS["Hit"], alpha=0.2)
            plotted = True
        if all_fa_lick and ref_lbc is not None:
            fa_grand = np.mean(all_fa_lick, axis=0)
            fa_sem = np.std(all_fa_lick, axis=0) / np.sqrt(len(all_fa_lick))
            ax_f.plot(ref_lbc, fa_grand, color=OUTCOME_COLORS["FA"], linewidth=2,
                      label=f"True FA (n={len(all_fa_lick)} sess)")
            ax_f.fill_between(ref_lbc, fa_grand - fa_sem, fa_grand + fa_sem,
                              color=OUTCOME_COLORS["FA"], alpha=0.2)
            plotted = True
        if plotted:
            ax_f.axvline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5,
                         label="Lick onset")
            ax_f.set_title(f"F. Grand-average CD — lick-aligned "
                           f"(True Hit vs True FA, z-scored)")
        else:
            ax_f.set_title("F. Grand-average CD — lick-aligned")
    else:
        ax_f.set_title("F. Grand-average CD — lick-aligned")
    ax_f.set_xlabel("Time from lick onset (s)")
    ax_f.set_ylabel("Projection onto avg CD axis (z-score vs baseline)")
    ax_f.legend(fontsize=8)

    # ── Row 4: Learning dynamics ──────────────────────────────────────
    # Panel G: CD effect size vs session index
    ax_g = fig.add_subplot(gs[3, 0])
    add_stage_background(ax_g, manifest)

    session_names = sorted(cd_results.keys(),
                           key=lambda k: cd_results[k]["session_idx"])
    idxs = [cd_results[k]["session_idx"] for k in session_names]
    # Peak effect in post-change window [0, 0.5]s
    peak_effects = []
    for k in session_names:
        r = cd_results[k]
        bc = r["bin_centers"]
        post_mask = (bc >= 0) & (bc <= 0.5)
        if post_mask.any():
            peak_effects.append(np.max(np.abs(r["effect"][post_mask])))
        else:
            peak_effects.append(0)

    stages = [cd_results[k]["stage"] for k in session_names]
    colors = [STAGE_COLORS[s] for s in stages]

    ax_g.scatter(idxs, peak_effects, c=colors, s=60, edgecolors="white",
                 linewidths=0.5, zorder=3)
    ax_g.plot(idxs, peak_effects, c="gray", alpha=0.3, linewidth=1, zorder=2)
    ax_g.set_xlabel("Session index")
    ax_g.set_ylabel("Peak |CD effect| (0\u2013500 ms)")
    ax_g.set_title("G. CD emergence across learning")

    # Panel H: CD effect by learning stage
    ax_h = fig.add_subplot(gs[3, 1])

    stage_effects = {}
    for stage in STAGE_ORDER:
        vals = [pe for k, pe in zip(session_names, peak_effects) if cd_results[k]["stage"] == stage]
        stage_effects[stage] = vals

    positions_list = []
    data_boxes = []
    colors_box = []
    for i, stage in enumerate(STAGE_ORDER):
        vals = stage_effects[stage]
        if vals:
            positions_list.append(i)
            data_boxes.append(vals)
            colors_box.append(STAGE_COLORS[stage])

    if data_boxes:
        bp = ax_h.boxplot(data_boxes, positions=positions_list, widths=0.5,
                          patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        # Overlay individual points
        for pos, vals, color in zip(positions_list, data_boxes, colors_box):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax_h.scatter(pos + jitter, vals, c=color, s=40, edgecolors="white",
                         linewidths=0.5, zorder=3)

    ax_h.set_xticks(range(len(STAGE_ORDER)))
    ax_h.set_xticklabels(STAGE_ORDER)
    ax_h.set_ylabel("Peak |CD effect|")
    ax_h.set_title("H. CD effect by learning stage")

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # CD effect trend across sessions
    if len(idxs) >= 3:
        rho, p = spearmanr(idxs, peak_effects)
        stats.append({"test": "cd_effect_vs_session_spearman", "rho": rho, "p": p,
                      "n": len(idxs)})

    # CD effect by stage
    valid_stage_data = [stage_effects[s] for s in STAGE_ORDER if stage_effects[s]]
    if len(valid_stage_data) >= 2:
        from scipy.stats import kruskal
        flat = [np.array(d) for d in valid_stage_data]
        flat = [d for d in flat if len(d) >= 2]
        if len(flat) >= 2:
            h, p = kruskal(*flat)
            stats.append({"test": "cd_effect_kruskal_by_stage", "H": h, "p": p})

    # Expert vs Naive comparison
    if stage_effects.get("Expert") and stage_effects.get("Naive"):
        e = np.array(stage_effects["Expert"])
        n = np.array(stage_effects["Naive"])
        if len(e) >= 2 and len(n) >= 2:
            u, p = mannwhitneyu(e, n, alternative="greater")
            stats.append({"test": "cd_expert_vs_naive_mwu", "U": u, "p": p})

    stats_df = pd.DataFrame(stats)

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig13_coding_direction", "03_population")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "03_population", "coding_direction_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: p={row.get('p', 'N/A')}")


if __name__ == "__main__":
    main()
