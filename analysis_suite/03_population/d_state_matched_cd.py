"""Fig 16: State-matched CD — state-matched sensory evidence on the coding direction.

Projects all SDT trial types onto the coding direction axis, matched
by HMM behavioral state and stratified by change size.  This dissociates
sensory evidence integration from motor / decision signals:

  * Hit and FA share the motor output (lick)
  * FA has no sensory change → any Hit−FA difference is sensory
  * Matching by HMM state controls for arousal / engagement
  * Stratifying Hits by change size reveals a sensory dose-response
  * FA trials are realigned to a pseudo-change time (lick − median Hit RT)
    and each FA is matched to the Hit (same HMM state) that is closest in
    both absolute session time (controlling for drift / satiety) and
    within-trial change latency from baseline onset (controlling for
    patience / foreperiod expectation)

If CD projection increases monotonically from FA (0 change) → small-
change Hit → big-change Hit, even controlling for motor output and
behavioral state, the coding direction encodes sensory evidence.

Produces (5 × 2 figure):
  Rows 1–4: One row per HMM state (Disengaged, Engaged, Impulsive) +
            All trials (pooled across states).
    Left column:  Grand-average time-resolved CD (Expert sessions, z-scored)
    Right column: Dose-response curve (FA → 1.25 → … → 4.0)
  Row 5 (summary):
    I. Dose-response slope (ρ) by learning stage (all-trials pooled)
    J. Sensory fraction by stage

Saves: figures/03_population/state_matched_cd_stats.csv
"""

import os
import sys
import gc
import argparse


import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import (
    STAGE_ORDER, STAGE_COLORS, OUTCOME_COLORS,
    HMM_STATE_ORDER, HMM_STATE_COLORS,
    CACHE_DIR, SMALL_CHANGE_SIZES, BIG_CHANGE_SIZES, DEFAULT_BIN_SIZE,
)
from visdetect.suite.loader import load_staging_manifest, load_session, load_hmm_assignments
from visdetect.analysis.utils import get_good_cluster_ids, build_population_tensor, smooth_psth, compute_lda_cd
from visdetect.analysis.align import (
    align_spikes_to_events, get_event_times_by_trial,
)
from visdetect.suite.plotting import setup_style, save_figure, add_stage_background

setup_style()

# ── Parameters ────────────────────────────────────────────────────────
WINDOW = (-0.5, 1.0)
BIN_SIZE = DEFAULT_BIN_SIZE
RESP_WIN = (0.0, 0.25)          # response window for scalar projection
MIN_UNITS = 10
MIN_TRIALS = 3                   # minimum trials per group for projection
CD_CACHE_DIR = os.path.join(CACHE_DIR, "cd_results")

# Dose-response x-axis: 0 = catch, then go change sizes
DOSE_LEVELS = [0.0, 1.25, 1.35, 1.5, 2.0, 4.0]
DOSE_LABELS = ["catch", "1.25", "1.35", "1.5", "2.0", "4.0"]


# ── Utilities ─────────────────────────────────────────────────────────

def _load_cd_axis(session_name, good_ids=None):
    """Load the average CD axis from the pre-computed cache (script 03a).

    Parameters
    ----------
    session_name : str
    good_ids : list[int], optional
        If provided, verify that the cached CD used the same clusters.

    Returns
    -------
    avg_cd : ndarray (n_units,) or None
    bc : ndarray (n_bins,) or None
    """
    path = os.path.join(CD_CACHE_DIR, f"{session_name}_hit_miss_cd.npz")
    if not os.path.exists(path):
        return None, None
    data = dict(np.load(path, allow_pickle=True))

    # Prefer the pre-computed avg_cd (post-change-averaged, normalised)
    avg_cd = data.get("avg_cd", None)
    bc = data.get("bin_centers", None)

    if avg_cd is None or bc is None:
        # Fallback: reconstruct from time-resolved CD matrix
        cds = data.get("cds", None)
        if cds is not None and bc is not None and cds.ndim == 2:
            post_mask = (bc >= 0) & (bc <= 0.5)
            avg_cd = cds[post_mask].mean(axis=0)
            norm = np.linalg.norm(avg_cd)
            if norm > 0:
                avg_cd = avg_cd / norm
            else:
                return None, None
        else:
            return None, None

    # Verify cluster alignment if possible
    if good_ids is not None:
        cached_ids = data.get("cluster_ids", None)
        if cached_ids is not None:
            if not np.array_equal(np.array(sorted(cached_ids)),
                                  np.array(sorted(good_ids))):
                print(f"    [!] CD cache cluster mismatch for {session_name}, "
                      f"recomputing")
                return None, None
        elif len(avg_cd) != len(good_ids):
            print(f"    [!] CD cache dimension mismatch for {session_name} "
                  f"({len(avg_cd)} vs {len(good_ids)}), recomputing")
            return None, None

    return avg_cd, bc



def _resp_window_proj(tensor, bc, cd_unit):
    """Mean scalar projection in the response window per trial."""
    resp_mask = (bc >= RESP_WIN[0]) & (bc < RESP_WIN[1])
    resp = np.nanmean(tensor[:, resp_mask, :], axis=1)   # (n_trials, n_units)
    valid = ~np.isnan(resp).any(axis=1)
    if valid.sum() == 0:
        return np.array([])
    return resp[valid] @ cd_unit                          # (n_valid,)


def _time_resolved_proj(tensor, cd_unit):
    """Per-trial time-resolved projection → (n_trials, n_bins)."""
    n_tr, n_bins, n_u = tensor.shape
    proj = np.full((n_tr, n_bins), np.nan)
    for b in range(n_bins):
        X = tensor[:, b, :]
        valid = ~np.isnan(X).any(axis=1)
        proj[valid, b] = X[valid] @ cd_unit
    return proj


def _build_tensor_custom_events(sess, cluster_ids, custom_event_times, window, bin_size):
    """Build a population tensor aligned to arbitrary absolute event times.

    Parameters
    ----------
    sess : Session
    cluster_ids : list of int
    custom_event_times : list of float
        One absolute event time per trial.
    window, bin_size : as usual

    Returns
    -------
    tensor : (n_trials, n_bins, n_units)
    bin_centers : (n_bins,)
    """
    cluster_map = {int(c.cluster_id): c for c in sess.clusters}
    unit_matrices = []
    for cid in cluster_ids:
        c = cluster_map.get(int(cid))
        if c is None:
            n_bins = int(np.round((window[1] - window[0]) / bin_size))
            unit_matrices.append(np.zeros((len(custom_event_times), n_bins)))
            continue
        mat, bin_centers = align_spikes_to_events(
            c.spike_times, custom_event_times,
            window=window, bin_size=bin_size,
        )
        unit_matrices.append(mat)
    tensor = np.stack(unit_matrices, axis=2)
    return tensor, bin_centers


# ── Per-session analysis ──────────────────────────────────────────────

def analyse_session(sess, sname, hmm_df, stage, sidx):
    """State-matched CD analysis for one session.

    Returns a dict keyed by (hmm_state, trial_category) with CD
    projections, or None if the session cannot be analysed.
    """
    good_ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
    if len(good_ids) < MIN_UNITS:
        return None

    trials = sess.trials

    # ── Obtain CD axis ────────────────────────────────────────────────
    # Prefer the cross-validated CD from script 03a (cached).
    # Fall back to a single response-window CD if no cache exists.
    cd_unit, bc = _load_cd_axis(sname, good_ids=good_ids)

    if cd_unit is not None:
        print(f"    Loaded cross-validated CD from cache ({len(cd_unit)} units)")
    else:
        # Fallback: compute CD from response-window activity (no CV)
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
        if len(go_hit_idx) < 5 or len(go_miss_idx) < 5:
            return None
        tensor_hm, bc, used_hm = build_population_tensor(
            sess, good_ids, event_name="Change_ON",
            window=WINDOW, bin_size=BIN_SIZE,
            trial_indices=go_hit_idx + go_miss_idx,
        )
        if tensor_hm.shape[0] < 10 or tensor_hm.shape[2] < MIN_UNITS:
            return None
        labels = np.array([
            1 if getattr(trials[i], "trialoutcome", None) == "Hit" else 0
            for i in used_hm
        ])
        if labels.sum() < 5 or (~labels.astype(bool)).sum() < 5:
            return None
        resp_mask = (bc >= RESP_WIN[0]) & (bc < RESP_WIN[1])
        resp_hm = np.nanmean(tensor_hm[:, resp_mask, :], axis=1)
        valid_hm = ~np.isnan(resp_hm).any(axis=1)
        if valid_hm.sum() < 10:
            return None
        cd_unit = compute_lda_cd(resp_hm[valid_hm], labels[valid_hm], method="manual", reg=1.0)
        print(f"    Computed fallback response-window CD ({len(cd_unit)} units)")

    # Need bin_centers for later projections — recompute if loaded from cache
    # (cache bc comes from script 1 which uses the same WINDOW/BIN_SIZE)
    # We still need go-trial indices for the median Hit RT calculation
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

    # ── HMM state lookup for this session ─────────────────────────────
    sess_hmm = hmm_df[hmm_df["session_name"] == sname].copy()
    if "trial_idx" not in sess_hmm.columns or len(sess_hmm) == 0:
        return None
    hmm_map = dict(zip(sess_hmm["trial_idx"].values,
                        sess_hmm["hmm_state_label"].values))

    # ── Absolute event times for temporal matching ────────────────────
    change_on_abs = get_event_times_by_trial(sess, "Change_ON")
    baseline_on_abs = get_event_times_by_trial(sess, "Baseline_ON")

    # Compute median Hit RT from go-trial Hits (for pseudo-change offset)
    hit_rts = []
    for i in go_hit_idx:
        rt = (trials[i].reactiontimes or {}).get("RT", np.nan)
        if np.isfinite(rt):
            hit_rts.append(rt)
    median_hit_rt = float(np.median(hit_rts)) if len(hit_rts) >= 3 else 0.2

    # ── Categorise every trial ────────────────────────────────────────
    # Categories: "fa" (catch lick), "cr" (catch no-lick),
    #             "hit_small", "hit_big", "hit_all", "miss"
    cat_indices = {}   # (hmm_state, category) → [trial_idx, ...]

    for i, t in enumerate(trials):
        outcome = getattr(t, "trialoutcome", None)
        cs = getattr(t, "change_size", None) or 1.0
        is_go = (cs - 1.0) > 0.01
        is_catch = not is_go
        state = hmm_map.get(i)
        if state is None:
            continue

        if is_catch and outcome == "Hit":
            cat = "fa"
        elif is_catch and outcome == "Miss":
            cat = "cr"
        elif is_go and outcome == "Hit":
            cat = "hit_all"
        elif is_go and outcome == "Miss":
            cat = "miss"
        else:
            continue   # abort / ref / early-FA

        key = (state, cat)
        cat_indices.setdefault(key, []).append(i)

        # Sub-split hits by change size
        if cat == "hit_all":
            if cs in SMALL_CHANGE_SIZES:
                cat_indices.setdefault((state, "hit_small"), []).append(i)
            elif cs in BIG_CHANGE_SIZES:
                cat_indices.setdefault((state, "hit_big"), []).append(i)

            # Individual dose level
            # Round to nearest known level for robustness
            for dose in DOSE_LEVELS[1:]:
                if abs(cs - dose) < 0.05:
                    cat_indices.setdefault((state, f"hit_{dose}"), []).append(i)
                    break

        # All go trials (hit + miss combined) by change size
        if is_go:
            cat_indices.setdefault((state, "go_all"), []).append(i)
            if cs in SMALL_CHANGE_SIZES:
                cat_indices.setdefault((state, "go_small"), []).append(i)
            elif cs in BIG_CHANGE_SIZES:
                cat_indices.setdefault((state, "go_big"), []).append(i)
            for dose in DOSE_LEVELS[1:]:
                if abs(cs - dose) < 0.05:
                    cat_indices.setdefault((state, f"go_{dose}"), []).append(i)
                    break

    # ── Compute pseudo-change times for FA trials ─────────────────────
    # For each FA: pseudo_change = lick_abs − median_hit_RT.
    # Then match to the Hit (same HMM state) that minimises a combined
    # distance in two dimensions:
    #   1. absolute session time  — controls for slow drift / satiety
    #   2. within-trial change latency (time from baseline onset to
    #      change event) — controls for patience / foreperiod expectation
    # The two distances are normalised by their respective IQRs before
    # combining as Euclidean distance so neither axis dominates.
    fa_pseudo_change = {}   # trial_idx → absolute pseudo-change time
    fa_match_info = {}      # trial_idx → matched hit trial_idx

    # Pre-compute within-trial change latency for all Hit trials
    hit_within_trial = {}   # trial_idx → change_on - baseline_on (seconds)
    for idx_list in cat_indices.values():
        for j in idx_list:
            if j in hit_within_trial:
                continue
            if (j < len(change_on_abs) and np.isfinite(change_on_abs[j])
                    and j < len(baseline_on_abs) and np.isfinite(baseline_on_abs[j])):
                hit_within_trial[j] = change_on_abs[j] - baseline_on_abs[j]

    # IQR-based normalisation constants (computed from all go-trial Hits)
    all_hit_abs = [change_on_abs[j] for j in go_hit_idx
                   if j < len(change_on_abs) and np.isfinite(change_on_abs[j])]
    all_hit_wt = [hit_within_trial[j] for j in go_hit_idx if j in hit_within_trial]
    iqr_abs = (float(np.subtract(*np.percentile(all_hit_abs, [75, 25])))
               if len(all_hit_abs) >= 4 else 1.0) or 1.0
    iqr_wt = (float(np.subtract(*np.percentile(all_hit_wt, [75, 25])))
              if len(all_hit_wt) >= 4 else 1.0) or 1.0

    for i, t in enumerate(trials):
        outcome = getattr(t, "trialoutcome", None)
        cs = getattr(t, "change_size", None) or 1.0
        is_catch = (cs - 1.0) <= 0.01
        if not (is_catch and outcome == "Hit"):
            continue
        state = hmm_map.get(i)
        if state is None:
            continue
        # Absolute lick time.  Catch-trial FAs have trialoutcome=="Hit" so
        # the RT is stored under "RT" (relative to Change_ON), not "FA".
        rt_dict = t.reactiontimes or {}
        rt = rt_dict.get("RT", np.nan)
        if not np.isfinite(rt):
            rt = rt_dict.get("FA", np.nan)       # fallback for rare formats
        if not np.isfinite(rt):
            continue
        if i < len(change_on_abs) and np.isfinite(change_on_abs[i]):
            lick_abs = change_on_abs[i] + rt      # RT is from Change_ON
        elif i < len(baseline_on_abs) and np.isfinite(baseline_on_abs[i]):
            ct = getattr(t, "change_time", None)
            if ct is not None and np.isfinite(float(ct)):
                lick_abs = baseline_on_abs[i] + float(ct) + rt
            else:
                continue
        else:
            continue
        pseudo_change_abs = lick_abs - median_hit_rt
        fa_pseudo_change[i] = pseudo_change_abs

        # Within-trial latency for this FA (pseudo-change rel. to baseline)
        if i < len(baseline_on_abs) and np.isfinite(baseline_on_abs[i]):
            fa_wt = pseudo_change_abs - baseline_on_abs[i]
        else:
            fa_wt = None

        # Find best-matching Hit: same HMM state, nearest in combined
        # (absolute time, within-trial latency) space
        same_state_hits = cat_indices.get((state, "hit_all"), [])
        if same_state_hits:
            best_hit, best_d = None, np.inf
            for j in same_state_hits:
                if j not in hit_within_trial:
                    continue
                d_abs = (change_on_abs[j] - pseudo_change_abs) / iqr_abs
                if fa_wt is not None:
                    d_wt = (hit_within_trial[j] - fa_wt) / iqr_wt
                    d = np.sqrt(d_abs ** 2 + d_wt ** 2)
                else:
                    d = abs(d_abs)
                if d < best_d:
                    best_d = d
                    best_hit = j
            if best_hit is not None:
                fa_match_info[i] = best_hit

    # ── Build tensors & project ───────────────────────────────────────
    result = {
        "bin_centers": bc,
        "cd_unit": cd_unit,
        "n_units": len(good_ids),
        "stage": stage,
        "session_idx": sidx,
        "median_hit_rt": median_hit_rt,
        "n_fa_matched": len(fa_match_info),
    }

    for key, idx_list in cat_indices.items():
        if len(idx_list) < MIN_TRIALS:
            continue
        state, cat = key

        # FA trials: align to pseudo-change (lick − median_hit_RT)
        if cat == "fa":
            fa_valid = [i for i in idx_list if i in fa_pseudo_change]
            if len(fa_valid) < MIN_TRIALS:
                continue
            custom_events = [fa_pseudo_change[i] for i in fa_valid]
            t, bc_t = _build_tensor_custom_events(
                sess, good_ids, custom_events,
                window=WINDOW, bin_size=BIN_SIZE,
            )
            n_fa_for_result = len(fa_valid)
        else:
            t, bc_t, _ = build_population_tensor(
                sess, good_ids, event_name="Change_ON",
                window=WINDOW, bin_size=BIN_SIZE,
                trial_indices=idx_list,
            )
            n_fa_for_result = 0  # not an FA category
        if t.shape[0] < MIN_TRIALS:
            continue

        # Time-resolved mean projection
        proj = _time_resolved_proj(t, cd_unit)
        proj_mean = np.nanmean(proj, axis=0)
        proj_sem = np.nanstd(proj, axis=0) / np.sqrt(proj.shape[0])

        # Scalar response-window projection
        scalar_proj = _resp_window_proj(t, bc, cd_unit)

        result[(state, cat)] = {
            "proj_mean": proj_mean,
            "proj_sem": proj_sem,
            "scalar_proj_mean": float(np.mean(scalar_proj)) if len(scalar_proj) > 0 else np.nan,
            "scalar_proj_sem": float(np.std(scalar_proj) / np.sqrt(len(scalar_proj)))
                               if len(scalar_proj) > 1 else 0.0,
            "n_trials": n_fa_for_result if cat == "fa" else len(idx_list),
        }

    # ── Pooled-state FA: combine FA trials across all HMM states ──────
    # FAs overwhelmingly occur in the Impulsive state, so per-state FA
    # in other states is often too sparse.  Pool all FA trials
    # (pseudo-change-aligned) into ("_pooled", "fa") for use in panels.
    all_fa_valid = [i for i in fa_pseudo_change]   # all with valid pseudo-change
    if len(all_fa_valid) >= MIN_TRIALS:
        custom_events = [fa_pseudo_change[i] for i in all_fa_valid]
        t_fa_all, _ = _build_tensor_custom_events(
            sess, good_ids, custom_events,
            window=WINDOW, bin_size=BIN_SIZE,
        )
        if t_fa_all.shape[0] >= MIN_TRIALS:
            proj = _time_resolved_proj(t_fa_all, cd_unit)
            scalar = _resp_window_proj(t_fa_all, bc, cd_unit)
            result[("_pooled", "fa")] = {
                "proj_mean": np.nanmean(proj, axis=0),
                "proj_sem": np.nanstd(proj, axis=0) / np.sqrt(proj.shape[0]),
                "scalar_proj_mean": float(np.mean(scalar)) if len(scalar) > 0 else np.nan,
                "scalar_proj_sem": float(np.std(scalar) / np.sqrt(len(scalar)))
                                   if len(scalar) > 1 else 0.0,
                "n_trials": len(all_fa_valid),
            }

    # ── All-trials pooled: combine across HMM states per category ────
    # This gives an "un-state-filtered" view for comparison.
    pooled_by_cat = {}   # cat → [trial_idx, ...]
    for (state, cat), idx_list in cat_indices.items():
        pooled_by_cat.setdefault(cat, []).extend(idx_list)

    for cat, idx_list in pooled_by_cat.items():
        # Deduplicate (sub-categories like hit_small are a subset of hit_all)
        idx_list = sorted(set(idx_list))
        if len(idx_list) < MIN_TRIALS:
            continue
        if cat == "fa":
            continue   # already handled above as ("_pooled", "fa")
        t_pool, _, _ = build_population_tensor(
            sess, good_ids, event_name="Change_ON",
            window=WINDOW, bin_size=BIN_SIZE,
            trial_indices=idx_list,
        )
        if t_pool.shape[0] < MIN_TRIALS:
            continue
        proj = _time_resolved_proj(t_pool, cd_unit)
        scalar = _resp_window_proj(t_pool, bc, cd_unit)
        result[("_pooled", cat)] = {
            "proj_mean": np.nanmean(proj, axis=0),
            "proj_sem": np.nanstd(proj, axis=0) / np.sqrt(proj.shape[0]),
            "scalar_proj_mean": float(np.mean(scalar)) if len(scalar) > 0 else np.nan,
            "scalar_proj_sem": float(np.std(scalar) / np.sqrt(len(scalar)))
                               if len(scalar) > 1 else 0.0,
            "n_trials": len(idx_list),
        }

    return result


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, default=1)
    args = parser.parse_args()

    print("[03d] State-matched sensory evidence analysis...")
    manifest = load_staging_manifest(qc_only=True)
    hmm_df = load_hmm_assignments()

    tasks = [
        (int(row["session_name"]), row["stage"], row["session_idx"])
        for _, row in manifest.iterrows()
    ]

    results = {}
    for sname, stage, sidx in tasks:
        print(f"  Session {sname} ({stage})...", end=" ", flush=True)
        try:
            sess = load_session(sname)
        except FileNotFoundError:
            print("not found")
            continue
        r = analyse_session(sess, sname, hmm_df, stage, sidx)
        if r is not None:
            results[sname] = r
            # Count how many state×category groups we got
            n_groups = sum(1 for k in r if isinstance(k, tuple))
            print(f"{n_groups} state x category groups")
        else:
            print("insufficient data")
        del sess
        gc.collect()

    print(f"\n  Analysed {len(results)} sessions")
    if not results:
        print("  No results. Exiting.")
        return

    # ── Create figure ─────────────────────────────────────────────────
    # Layout: 4 state rows (Disengaged, Engaged, Impulsive, All) × 2 cols
    #         (grand-avg time-trace | dose-response) + 1 summary row
    STATE_ROWS = list(HMM_STATE_ORDER) + ["_pooled"]
    STATE_LABELS = {s: s for s in HMM_STATE_ORDER}
    STATE_LABELS["_pooled"] = "All trials"
    STATE_ROW_COLORS = dict(HMM_STATE_COLORS)
    STATE_ROW_COLORS["_pooled"] = "#555555"

    n_rows = len(STATE_ROWS) + 1   # +1 for summary row
    fig = plt.figure(figsize=(20, 5 * n_rows + 2))
    gs = gridspec.GridSpec(n_rows, 2, hspace=0.45, wspace=0.3,
                           top=0.95, bottom=0.10)

    fig.suptitle("State-matched CD dose\u2013response (Hit-only at each change size)",
                 fontsize=14, fontweight="bold", y=0.98)

    expert = {k: v for k, v in results.items() if v["stage"] == "Expert"}

    # Colour scheme for change-size categories
    CAT_COLORS = {
        "fa":        OUTCOME_COLORS["FA"],
        "hit_small": "#81C784",
        "hit_big":   "#2E7D32",
        "hit_all":   OUTCOME_COLORS["Hit"],
        "miss":      OUTCOME_COLORS["Miss"],
        "cr":        OUTCOME_COLORS["CR"],
    }
    CAT_LABELS = {
        "fa":        "True FA",
        "hit_small": "Small-\u0394 Hit",
        "hit_big":   "Big-\u0394 Hit",
        "hit_all":   "All Hit",
        "miss":      "Miss",
        "cr":        "True CR (catch)",
    }

    # ── Per-session baseline z-scoring for grand averages ─────────────
    CHANGE_BL = (-0.5, -0.1)

    def _zscore_baseline(trace, bin_centers, bl_window):
        bl_mask = (bin_centers >= bl_window[0]) & (bin_centers < bl_window[1])
        if bl_mask.sum() < 2:
            return trace
        bl = trace[bl_mask]
        mu, sd = bl.mean(), bl.std()
        if sd < 1e-12:
            return trace - mu
        return (trace - mu) / sd

    def _zscore_resp_scalars(cat_dicts, bc):
        """Z-score multiple categories using a shared (pooled) baseline, return resp-window means.

        Parameters
        ----------
        cat_dicts : list of (dict or None)
            Each dict has key "proj_mean" (1-D trace). None entries yield NaN.
        bc : ndarray
            Bin centers.

        Returns
        -------
        list of float — one resp-window mean per category (shared baseline z-scored).
        """
        bl_mask = (bc >= CHANGE_BL[0]) & (bc < CHANGE_BL[1])
        resp_mask = (bc >= RESP_WIN[0]) & (bc < RESP_WIN[1])

        # Smooth all valid traces
        smoothed = []
        valid_idx = []
        for i, d in enumerate(cat_dicts):
            if d is None:
                smoothed.append(None)
                continue
            pm = d.get("proj_mean")
            if pm is None or len(pm) != len(bc):
                smoothed.append(None)
                continue
            smoothed.append(smooth_psth(pm, BIN_SIZE, 15.0))
            valid_idx.append(i)

        if not valid_idx or bl_mask.sum() < 2 or resp_mask.sum() < 1:
            return [np.nan] * len(cat_dicts)

        # Pool baseline from ALL valid categories
        pooled_bl = np.concatenate([smoothed[i][bl_mask] for i in valid_idx])
        mu, sd = pooled_bl.mean(), pooled_bl.std()
        if sd < 1e-12:
            sd = 1.0

        # Z-score each and extract resp-window mean
        out = []
        for sm in smoothed:
            if sm is None:
                out.append(np.nan)
            else:
                z = (sm - mu) / sd
                out.append(float(np.mean(z[resp_mask])))
        return out

    def _get(r, state, cat):
        """Lookup (state, cat) with fallback to pooled."""
        d = r.get((state, cat))
        if d is not None:
            return d
        if state == "_pooled":
            return None
        return r.get(("_pooled", cat))

    # Dose-response setup
    dose_cats = ["fa"] + [f"hit_{d}" for d in DOSE_LEVELS[1:]]
    dose_x = list(range(len(DOSE_LEVELS)))

    panel_letter = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    # ── State rows: grand-avg time trace (left) + dose-response (right) ──
    for row_i, state in enumerate(STATE_ROWS):
        state_label = STATE_LABELS[state]
        state_color = STATE_ROW_COLORS[state]
        ltr_left = next(panel_letter)
        ltr_right = next(panel_letter)

        # ── Left panel: Grand-average time-resolved (Expert sessions) ─
        ax_left = fig.add_subplot(gs[row_i, 0])
        if expert:
            ref_bc = list(expert.values())[0]["bin_centers"]
            grand = {}

            # FIXED: Use shared baseline normalization to preserve relative differences
            bl_mask = (ref_bc >= CHANGE_BL[0]) & (ref_bc < CHANGE_BL[1])

            # First pass: collect all Hit traces to establish shared baseline
            all_hit_traces = []
            for r in expert.values():
                # Use hit_big as reference (highest signal), fallback to hit_small
                d_hit = _get(r, state, "hit_big") or _get(r, state, "hit_small")
                if d_hit is not None and len(d_hit["proj_mean"]) == len(ref_bc):
                    sm_hit = smooth_psth(d_hit["proj_mean"], BIN_SIZE, 15.0)
                    all_hit_traces.append(sm_hit)

            if all_hit_traces:
                # Compute shared baseline from Hit trials
                hit_baselines = [trace[bl_mask] for trace in all_hit_traces if bl_mask.sum() >= 2]
                if hit_baselines:
                    all_bl_vals = np.concatenate(hit_baselines)
                    mu_shared = all_bl_vals.mean()
                    sd_shared = all_bl_vals.std()
                    if sd_shared < 1e-12:
                        sd_shared = 1.0  # Avoid division by zero

                    # Second pass: normalize all categories to shared baseline
                    for cat in ["fa", "hit_small", "hit_big"]:
                        traces = []
                        for r in expert.values():
                            d = _get(r, state, cat)
                            if d is not None and len(d["proj_mean"]) == len(ref_bc):
                                sm = smooth_psth(d["proj_mean"], BIN_SIZE, 15.0)
                                traces.append((sm - mu_shared) / sd_shared)
                        if traces:
                            grand[cat] = (np.mean(traces, axis=0),
                                          np.std(traces, axis=0) / np.sqrt(len(traces)),
                                          len(traces))
                else:
                    # Fallback to per-category if baseline computation fails
                    for cat in ["fa", "hit_small", "hit_big"]:
                        traces = []
                        for r in expert.values():
                            d = _get(r, state, cat)
                            if d is not None and len(d["proj_mean"]) == len(ref_bc):
                                sm = smooth_psth(d["proj_mean"], BIN_SIZE, 15.0)
                                traces.append(_zscore_baseline(sm, ref_bc, CHANGE_BL))
                        if traces:
                            grand[cat] = (np.mean(traces, axis=0),
                                          np.std(traces, axis=0) / np.sqrt(len(traces)),
                                          len(traces))

            plotted = False
            for cat in ["fa", "hit_small", "hit_big"]:
                if cat in grand:
                    m, s, n = grand[cat]
                    ax_left.plot(ref_bc, m, color=CAT_COLORS[cat], linewidth=2,
                                 label=f"{CAT_LABELS[cat]} (n={n} sess)")
                    ax_left.fill_between(ref_bc, m - s, m + s,
                                         color=CAT_COLORS[cat], alpha=0.2)
                    plotted = True
            if plotted:
                ax_left.axvline(0, color="k", linestyle="--",
                                linewidth=0.8, alpha=0.5)
            ax_left.set_title(f"{ltr_left}. Grand-average \u2014 {state_label} "
                              f"Expert sessions", color=state_color,
                              fontweight="bold")
        else:
            ax_left.text(0.5, 0.5, "No Expert sessions",
                         transform=ax_left.transAxes, ha="center")
            ax_left.set_title(f"{ltr_left}. Grand-average \u2014 {state_label}",
                              color=state_color, fontweight="bold")
        ax_left.set_xlabel("Time from Change_ON (s)")
        ax_left.set_ylabel("CD projection (z-score vs baseline)")
        ax_left.legend(fontsize=7, loc="upper left")

        # ── Right panel: Dose-response (Expert sessions) ─────────────
        ax_right = fig.add_subplot(gs[row_i, 1])
        if expert:
            dose_per_session = []
            for r in expert.values():
                cat_dicts = [_get(r, state, cat) for cat in dose_cats]
                dose_per_session.append(_zscore_resp_scalars(cat_dicts, ref_bc))

            dose_arr = np.array(dose_per_session)
            for sess_row in dose_arr:
                valid = np.isfinite(sess_row)
                if valid.sum() >= 2:
                    ax_right.plot(np.array(dose_x)[valid], sess_row[valid],
                                  color="gray", alpha=0.15, linewidth=0.8,
                                  zorder=1)

            dose_mean = np.nanmean(dose_arr, axis=0)
            dose_sem = np.nanstd(dose_arr, axis=0) / np.sqrt(
                np.sum(np.isfinite(dose_arr), axis=0).clip(1))
            finite = np.isfinite(dose_mean)
            if finite.any():
                x_f = np.array(dose_x)[finite]
                ax_right.errorbar(x_f, dose_mean[finite],
                                  yerr=dose_sem[finite],
                                  color=state_color, linewidth=2,
                                  marker="o", markersize=6, capsize=4,
                                  zorder=3, label="Grand mean")
                # Spearman per session
                rho_list = []
                for sess_row in dose_arr:
                    v = np.isfinite(sess_row)
                    if v.sum() >= 3:
                        r_s, _ = spearmanr(np.array(DOSE_LEVELS)[v],
                                           sess_row[v])
                        if np.isfinite(r_s):
                            rho_list.append(r_s)
                if rho_list:
                    med_rho = np.median(rho_list)
                    ax_right.set_title(
                        f"{ltr_right}. Hit-only dose-response ({state_label}) "
                        f"\u2014 median \u03c1={med_rho:.2f}",
                        color=state_color, fontweight="bold")
                else:
                    ax_right.set_title(
                        f"{ltr_right}. Hit-only dose-response ({state_label})",
                        color=state_color, fontweight="bold")
                ax_right.legend(fontsize=7)
            else:
                ax_right.set_title(
                    f"{ltr_right}. Hit-only dose-response ({state_label})",
                    color=state_color, fontweight="bold")
        else:
            ax_right.set_title(
                f"{ltr_right}. Hit-only dose-response ({state_label})",
                color=state_color, fontweight="bold")

        ax_right.set_xticks(dose_x)
        ax_right.set_xticklabels(DOSE_LABELS, rotation=30)
        ax_right.set_xlabel("Change size (0 = catch)")
        ax_right.set_ylabel("CD projection (z-score vs baseline)")

    # ── Summary row: dose-response slope + sensory fraction by stage ──
    ltr_e = next(panel_letter)
    ltr_f = next(panel_letter)

    # Compute per-state slopes for all sessions
    # For summary, use _pooled (all trials) — fairest comparison
    SUMMARY_STATE = "_pooled"
    ax_e = fig.add_subplot(gs[len(STATE_ROWS), 0])

    stage_slopes = {s: [] for s in STAGE_ORDER}
    for r in results.values():
        _bc = r["bin_centers"]
        cat_dicts = [_get(r, SUMMARY_STATE, cat) for cat in dose_cats]
        row = _zscore_resp_scalars(cat_dicts, _bc)
        finite_vals = [(DOSE_LEVELS[j], row[j])
                       for j in range(len(row)) if np.isfinite(row[j])]
        if len(finite_vals) >= 3:
            xs, ys = zip(*finite_vals)
            rho, _ = spearmanr(xs, ys)
            if np.isfinite(rho):
                stage_slopes[r["stage"]].append(rho)

    box_data, box_pos, box_colors = [], [], []
    for i, stage in enumerate(STAGE_ORDER):
        if stage_slopes[stage]:
            box_pos.append(i)
            box_data.append(stage_slopes[stage])
            box_colors.append(STAGE_COLORS[stage])

    if box_data:
        bp = ax_e.boxplot(box_data, positions=box_pos, widths=0.5,
                          patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for pos, vals, color in zip(box_pos, box_data, box_colors):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax_e.scatter(pos + jitter, vals, c=color, s=40,
                         edgecolors="white", linewidths=0.5, zorder=3)

    ax_e.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_e.set_xticks(range(len(STAGE_ORDER)))
    ax_e.set_xticklabels(STAGE_ORDER)
    ax_e.set_ylabel("Dose-response slope (\u03c1)")
    ax_e.set_title(f"{ltr_e}. Hit-only dose slope across learning")

    # ── Sensory fraction by stage ─────────────────────────────────────
    ax_f = fig.add_subplot(gs[len(STATE_ROWS), 1])

    stage_fracs = {s: [] for s in STAGE_ORDER}
    for r in results.values():
        d_big = _get(r, SUMMARY_STATE, "hit_big")
        d_fa = _get(r, SUMMARY_STATE, "fa")
        d_miss = _get(r, SUMMARY_STATE, "miss")
        if d_big is None or d_fa is None or d_miss is None:
            continue
        _bc = r["bin_centers"]
        h, f, m = _zscore_resp_scalars([d_big, d_fa, d_miss], _bc)
        if not (np.isfinite(h) and np.isfinite(f) and np.isfinite(m)):
            continue
        denom = h - m
        if abs(denom) < 1e-6:
            continue
        sensory_frac = (h - f) / denom
        stage_fracs[r["stage"]].append(sensory_frac)

    box2_data, box2_pos, box2_colors = [], [], []
    for i, stage in enumerate(STAGE_ORDER):
        if stage_fracs[stage]:
            box2_pos.append(i)
            box2_data.append(stage_fracs[stage])
            box2_colors.append(STAGE_COLORS[stage])

    if box2_data:
        bp2 = ax_f.boxplot(box2_data, positions=box2_pos, widths=0.5,
                           patch_artist=True, showfliers=False)
        for patch, color in zip(bp2["boxes"], box2_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for pos, vals, color in zip(box2_pos, box2_data, box2_colors):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax_f.scatter(pos + jitter, vals, c=color, s=40,
                         edgecolors="white", linewidths=0.5, zorder=3)

    ax_f.axhline(0, color="gray", linewidth=0.5, linestyle=":",
                 label="0 = purely motor")
    ax_f.axhline(1, color="gray", linewidth=0.5, linestyle="--",
                 label="1 = purely sensory")
    ax_f.set_xticks(range(len(STAGE_ORDER)))
    ax_f.set_xticklabels(STAGE_ORDER)
    ax_f.set_ylabel("Sensory fraction\n(bigHit\u2212FA) / (bigHit\u2212Miss)")
    ax_f.set_title(f"{ltr_f}. Sensory vs motor contribution to CD by stage")
    ax_f.legend(fontsize=8, loc="lower right")

    # ── Statistics ────────────────────────────────────────────────────
    stats = []

    # Dose-response slope across learning (all trials)
    all_slopes = []
    all_sidxs = []
    for r in results.values():
        _bc = r["bin_centers"]
        cat_dicts = [_get(r, SUMMARY_STATE, cat) for cat in dose_cats]
        row = _zscore_resp_scalars(cat_dicts, _bc)
        finite_vals = [(DOSE_LEVELS[j], row[j])
                       for j in range(len(row)) if np.isfinite(row[j])]
        if len(finite_vals) >= 3:
            xs, ys = zip(*finite_vals)
            rho, _ = spearmanr(xs, ys)
            if np.isfinite(rho):
                all_slopes.append(rho)
                all_sidxs.append(r["session_idx"])
    if len(all_slopes) >= 3:
        rho_trend, p_trend = spearmanr(all_sidxs, all_slopes)
        stats.append({"test": "dose_slope_vs_session_spearman",
                      "rho": rho_trend, "p": p_trend, "n": len(all_slopes)})

    from scipy.stats import kruskal as _kruskal
    valid_groups = [np.array(stage_slopes[s]) for s in STAGE_ORDER
                    if len(stage_slopes[s]) >= 2]
    if len(valid_groups) >= 2:
        try:
            h_val, p_val = _kruskal(*valid_groups)
            stats.append({"test": "dose_slope_kruskal_by_stage",
                          "H": h_val, "p": p_val})
        except ValueError:
            pass

    valid_frac = [np.array(stage_fracs[s]) for s in STAGE_ORDER
                  if len(stage_fracs[s]) >= 2]
    if len(valid_frac) >= 2:
        try:
            h_val, p_val = _kruskal(*valid_frac)
            stats.append({"test": "sensory_frac_kruskal_by_stage",
                          "H": h_val, "p": p_val})
        except ValueError:
            pass

    expert_slopes = stage_slopes.get("Expert", [])
    if len(expert_slopes) >= 3:
        from scipy.stats import wilcoxon as _wilcoxon
        try:
            w, p = _wilcoxon(expert_slopes)
            stats.append({"test": "expert_dose_slope_vs_0_wilcoxon",
                          "W": w, "p": p,
                          "median_rho": float(np.median(expert_slopes)),
                          "n": len(expert_slopes)})
        except ValueError:
            pass

    stats_df = pd.DataFrame(stats)

    # ── Explanation text box ──────────────────────────────────────────
    explanation = (
        "ANALYSIS  For each session the coding direction (CD) is computed from "
        "go-trial Hit vs Miss responses (change_size > 1).  Every trial is then "
        "assigned its HMM behavioural state (Engaged / Impulsive / Disengaged) and "
        "SDT category (Hit, Miss, FA, CR).  Catch-trial FAs are aligned to a "
        "pseudo-change time = lick \u2212 median Hit RT.  Each row shows results "
        "for one HMM state (or all trials pooled), with grand-average time-resolved "
        "CD projection on the left and dose\u2013response on the right.  "
        "Bottom row: summary statistics across learning stages (using all-trials "
        "pooled data)."
    )
    fig.text(
        0.5, 0.005, explanation,
        ha="center", va="bottom", fontsize=8,
        fontstyle="italic", color="#444444",
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5",
                  ec="#cccccc", alpha=0.9),
        transform=fig.transFigure,
    )

    # ── Save ──────────────────────────────────────────────────────────
    save_figure(fig, "fig16_state_matched_cd", "03_population")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "03_population", "state_matched_cd_stats.csv",
    )
    stats_df.to_csv(stats_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row in stats_df.iterrows():
        print(f"    {row['test']}: p={row.get('p', 'N/A')}")


if __name__ == "__main__":
    main()
