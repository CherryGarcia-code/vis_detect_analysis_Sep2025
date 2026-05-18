"""Fig 17b: 2D Decomposition — Task-State x Sensory Coding Directions (CORRECTED).

Tests the Lohse et al. framework: striatal population activity should have two
orthogonal dimensions: task-state CD and sensory CD.

CORRECTED METHOD (April 2026):
Task-state CD now computed using TF pulse alignment during baseline periods
(not Change_ON alignment), following Lohse et al. methodology exactly.
This compares Hit vs Miss trial contexts around individual TF pulses.

Phase 1 produces:
  - Panel A: Cosine similarity per session (colored by stage)
  - Panel B: Cosine similarity vs session index + trend
  - Panel C: Cosine similarity by stage (boxplot + stats)

Saves: figures/03_population/fig17b_2d_decomposition_corrected.png
Stats: figures/03_population/2d_decomposition_corrected_stats.csv
"""

import os
import sys
import gc
from concurrent.futures import ProcessPoolExecutor


import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, kruskal, mannwhitneyu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import (
    STAGE_ORDER, STAGE_COLORS, CACHE_DIR, DEFAULT_BIN_SIZE,
)
from visdetect.suite.loader import load_staging_manifest, load_session, load_tf_traces_npz
from visdetect.analysis.utils import (
    get_good_cluster_ids, build_population_tensor,
    permutation_test, bootstrap_ci,
)
from visdetect.analysis.utils import compute_lda_cd
from visdetect.suite.plotting import setup_style, save_figure, add_stage_background
from visdetect.analysis.align import align_spikes_to_events
from visdetect.analysis.constants import LOHSE_SENSORY_CD_WINDOW

setup_style()

# Parameters
TASK_CD_WINDOW = (-1.5, 1.0)       # Full tensor window for trajectory analysis
BASELINE_AVG_WINDOW = (-1.0, 0.0)  # Pre-change averaging for trajectory baseline
TF_PEAK_WINDOW = LOHSE_SENSORY_CD_WINDOW  # (0.122, 0.167) s — Lohse et al. 2025
MIN_UNITS = 10
MIN_TRIALS_PER_CLASS = 8
N_PERM_ORTHO = 1000
BIN_SIZE = DEFAULT_BIN_SIZE
CD_REG = 1.0                        # Shrinkage regularization

# Cache files (corrected version)
CACHE_FILE = os.path.join(CACHE_DIR, "2d_decomposition_corrected.csv")



def compute_task_cd(sess, common_ids):
    """Compute task-state coding direction using FAST TF pulse alignment (Lohse method).

    Uses activity around FAST TF pulses during baseline periods, comparing Hit vs Miss
    trials to identify the task-state dimension that controls sensorimotor gating.

    CORRECTED: Uses only fast pulses (following Lohse et al.), with existing constraints
    (min_before_change=1.0s) to avoid contamination from change-related activity.

    Parameters
    ----------
    sess : SessionData
    common_ids : list of int
        Cluster IDs that have both good QC and TF trace availability

    Returns
    -------
    cd_task : ndarray, shape (n_units,)
        Normalized task-state coding direction (Hit vs Miss)
    baseline_proj : dict
        {'hit': array, 'miss': array} - baseline projections for validation
    n_pulses : int
        Number of fast pulses used in computation
    """
    from visdetect.analysis.tf_pulse import _collect_pulses, TFRespPulseConfig
    from visdetect.analysis.align import get_event_times_by_trial

    trials = sess.trials

    # Get TF pulse times during baseline periods (with constraints enabled)
    cfg = TFRespPulseConfig(use_constraints=True)  # Ensures min_before_change=1.0s
    fast_times, slow_times = _collect_pulses(sess, cfg, show_progress=False)

    # Use ONLY fast pulses for task-state CD (Lohse method)
    pulse_times = fast_times

    if len(pulse_times) < 20:  # Need sufficient fast pulses
        return None, None, 0

    # Build tensor aligned to fast TF pulses (custom event times, not trial-aligned)
    pulse_window = (-0.5, 0.5)  # Around each pulse
    pulse_times_list = [float(t) for t in pulse_times]
    cluster_map = {c.cluster_id: c for c in sess.clusters}
    bc = None
    tensor = None
    for uid, cid in enumerate(common_ids):
        cluster = cluster_map.get(cid)
        if cluster is None:
            continue
        mat, bc_ = align_spikes_to_events(
            cluster.spike_times, pulse_times_list,
            window=pulse_window, bin_size=BIN_SIZE,
        )
        if tensor is None:
            bc = bc_
            tensor = np.zeros((len(pulse_times_list), len(bc_), len(common_ids)))
        tensor[:, :, uid] = mat

    if tensor is None or tensor.shape[0] < 20:
        return None, None, 0

    # Get trial outcomes for each pulse time
    baseline_times = get_event_times_by_trial(sess, "Baseline_ON")
    change_times = get_event_times_by_trial(sess, "Change_ON")

    pulse_outcomes = []
    for pulse_time in pulse_times:
        # Find which trial this pulse belongs to
        trial_outcome = None
        for i, trial in enumerate(trials):
            if (i < len(baseline_times) and i < len(change_times) and
                np.isfinite(baseline_times[i]) and np.isfinite(change_times[i])):
                # Only consider pulses during baseline period (before change)
                if baseline_times[i] <= pulse_time <= change_times[i]:
                    trial_outcome = getattr(trial, 'trialoutcome', None)
                    change_size = getattr(trial, 'change_size', 1.0)
                    # Only use go-trial outcomes (no catch trials)
                    if trial_outcome in ['Hit', 'Miss'] and change_size > 1.01:
                        pulse_outcomes.append(trial_outcome)
                    else:
                        pulse_outcomes.append(None)
                    break
        else:
            pulse_outcomes.append(None)

    # Filter to valid pulses from Hit and Miss trials
    valid_mask = np.array([outcome in ['Hit', 'Miss'] for outcome in pulse_outcomes])
    if np.sum(valid_mask) < 20:
        return None, None, 0

    tensor_filt = tensor[valid_mask]
    outcomes_filt = np.array(pulse_outcomes)[valid_mask]

    # Average activity in pre-pulse window for task-state (Lohse method)
    from visdetect.analysis.constants import TF_PULSE_PRE_WINDOW
    pre_mask = (bc >= TF_PULSE_PRE_WINDOW[0]) & (bc < TF_PULSE_PRE_WINDOW[1])
    if not np.any(pre_mask):
        return None, None, 0

    # Average each pulse in pre-window -> (n_pulses, n_units)
    pre_activity = np.mean(tensor_filt[:, pre_mask, :], axis=1)

    # Separate by outcome
    hit_mask = outcomes_filt == 'Hit'
    miss_mask = outcomes_filt == 'Miss'

    if np.sum(hit_mask) < MIN_TRIALS_PER_CLASS or np.sum(miss_mask) < MIN_TRIALS_PER_CLASS:
        return None, None, 0

    hit_activity = pre_activity[hit_mask]
    miss_activity = pre_activity[miss_mask]

    # Create labels for shrinkage LDA
    labels = np.concatenate([np.ones(len(hit_activity)), np.zeros(len(miss_activity))])
    all_activity = np.vstack([hit_activity, miss_activity])

    # Compute task-state coding direction with shrinkage LDA
    cd_task = compute_lda_cd(all_activity, labels, method="manual", reg=CD_REG)

    # Project baseline activity for validation
    hit_proj = hit_activity @ cd_task
    miss_proj = miss_activity @ cd_task

    baseline_proj = {
        'hit': hit_proj,
        'miss': miss_proj,
    }

    return cd_task, baseline_proj, len(pulse_times)


def compute_sensory_cd(tf_data, common_ids):
    """Compute sensory coding direction from TF pulse responsiveness.

    Parameters
    ----------
    tf_data : dict
        TF traces loaded from NPZ cache
    common_ids : list of int
        Cluster IDs that have both good QC and TF trace availability

    Returns
    -------
    cd_sensory : ndarray, shape (n_units,)
        Normalized sensory coding direction
    peak_amplitudes : ndarray, shape (n_units,)
        Peak z-score amplitudes for each unit (for validation)
    """
    t_vec = tf_data['t_vec']
    cluster_ids = tf_data['cluster_ids']
    fast_z = tf_data['fast_z']  # (n_units, n_time)

    # Find time window for peak extraction
    peak_mask = (t_vec >= TF_PEAK_WINDOW[0]) & (t_vec < TF_PEAK_WINDOW[1])
    if peak_mask.sum() == 0:
        return None, None

    # Extract peak amplitude for each common unit
    peak_amplitudes = np.zeros(len(common_ids))

    for i, cid in enumerate(common_ids):
        # Find index of this cluster in TF data
        tf_idx = np.where(cluster_ids == cid)[0]
        if len(tf_idx) == 0:
            peak_amplitudes[i] = 0.0  # Unit not in TF data
            continue

        tf_idx = tf_idx[0]

        # Extract signed peak (max absolute value preserving sign)
        z_trace = fast_z[tf_idx, peak_mask]
        max_idx = np.argmax(np.abs(z_trace))
        peak_amplitudes[i] = z_trace[max_idx]

    # Normalize to unit vector
    norm = np.linalg.norm(peak_amplitudes)
    cd_sensory = peak_amplitudes / norm if norm > 0 else peak_amplitudes

    return cd_sensory, peak_amplitudes


def test_orthogonality(cd_task, cd_sensory, common_ids, n_perm=N_PERM_ORTHO):
    """Test orthogonality between task and sensory coding directions with permutation null.

    Parameters
    ----------
    cd_task : ndarray
        Task-state coding direction
    cd_sensory : ndarray
        Sensory coding direction
    common_ids : list
        Unit IDs (for permutation)
    n_perm : int
        Number of permutations for null distribution

    Returns
    -------
    cos_sim : float
        Cosine similarity between the two directions
    p_value : float
        Two-tailed permutation p-value
    null_dist : ndarray
        Null distribution of cosine similarities
    """
    # Actual cosine similarity
    cos_sim = np.dot(cd_task, cd_sensory)

    # Generate null distribution by shuffling unit identities
    np.random.seed(42)  # Reproducible
    null_dist = np.zeros(n_perm)

    for i in range(n_perm):
        # Shuffle the sensory coding direction (permute unit assignments)
        perm_indices = np.random.permutation(len(cd_sensory))
        cd_sensory_perm = cd_sensory[perm_indices]
        null_dist[i] = np.dot(cd_task, cd_sensory_perm)

    # Two-tailed p-value
    pval = np.mean(np.abs(null_dist) >= np.abs(cos_sim))

    return cos_sim, pval, null_dist


def compute_2d_decodability_leave_two_out(sess, common_ids, cd_task, cd_sensory, min_trials=10):
    """Compute Hit vs Miss decodability in 2D coding space using leave-two-out CV.

    Parameters
    ----------
    sess : SessionData
    common_ids : list of int
    cd_task : ndarray
    cd_sensory : ndarray
    min_trials : int
        Minimum trials per class for analysis

    Returns
    -------
    dict
        Results with AUC, accuracy, and permutation p-value
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.preprocessing import StandardScaler
    import itertools

    trials = sess.trials

    # Get Hit and Miss go-trial indices
    hit_indices = [i for i, t in enumerate(trials)
                   if getattr(t, "trialoutcome", None) == "Hit"
                   and (getattr(t, "change_size", None) or 1.0) > 1.01]
    miss_indices = [i for i, t in enumerate(trials)
                    if getattr(t, "trialoutcome", None) == "Miss"
                    and (getattr(t, "change_size", None) or 1.0) > 1.01]

    if len(hit_indices) < min_trials or len(miss_indices) < min_trials:
        return None

    # Build population tensor for all trials
    all_trial_indices = hit_indices + miss_indices
    try:
        tensor, bin_centers, used_indices = build_population_tensor(
            sess, common_ids, event_name="Change_ON",
            window=TASK_CD_WINDOW, bin_size=BIN_SIZE,
            trial_indices=all_trial_indices,
        )
    except Exception:
        return None

    if tensor.shape[0] < 2 * min_trials:
        return None

    # Project onto 2D coding space and average in response window (0.0 to 0.5s)
    resp_window_mask = (bin_centers >= 0.0) & (bin_centers <= 0.5)
    if resp_window_mask.sum() == 0:
        return None

    # Average activity in response window, then project
    resp_activity = tensor[:, resp_window_mask, :].mean(axis=1)  # (n_trials, n_units)
    proj_task = resp_activity @ cd_task
    proj_sensory = resp_activity @ cd_sensory

    # Create feature matrix and labels
    X = np.column_stack([proj_task, proj_sensory])  # (n_trials, 2)
    y = np.array([1 if i < len(hit_indices) else 0 for i in range(len(all_trial_indices))])  # Hit=1, Miss=0

    # Leave-two-out cross-validation (1 Hit + 1 Miss per fold)
    n_hits = len(hit_indices)
    n_miss = len(miss_indices)
    n_folds = min(n_hits, n_miss)  # Maximum possible folds

    if n_folds < 3:  # Need minimum folds for reliable estimate
        return None

    # Generate leave-two-out folds (1 from each class)
    hit_idx_in_X = np.arange(n_hits)
    miss_idx_in_X = np.arange(n_hits, n_hits + n_miss)

    fold_scores = []
    fold_aucs = []

    # Use first n_folds trials from each class to create balanced folds
    for i in range(n_folds):
        # Test set: 1 hit + 1 miss
        test_hit = hit_idx_in_X[i]
        test_miss = miss_idx_in_X[i]
        test_idx = [test_hit, test_miss]

        # Train set: all others
        train_idx = [j for j in range(len(y)) if j not in test_idx]

        if len(train_idx) < 4:  # Need minimum training samples
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train classifier
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train_scaled, y_train)

        # Predict
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        fold_scores.append(acc)
        fold_aucs.append(auc)

    if len(fold_aucs) == 0:
        return None

    # Aggregate results
    mean_accuracy = np.mean(fold_scores)
    mean_auc = np.mean(fold_aucs)

    # Permutation test for chance-level estimation
    n_perms = 200
    perm_aucs = []

    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        y_perm = rng.permutation(y)

        perm_fold_aucs = []
        for i in range(n_folds):
            test_hit = hit_idx_in_X[i]
            test_miss = miss_idx_in_X[i]
            test_idx = [test_hit, test_miss]
            train_idx = [j for j in range(len(y_perm)) if j not in test_idx]

            if len(train_idx) < 4:
                continue

            X_train, X_test = X[train_idx], X[test_idx]
            y_train_perm, y_test_perm = y_perm[train_idx], y_perm[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(X_train_scaled, y_train_perm)
            y_prob_perm = clf.predict_proba(X_test_scaled)[:, 1]

            try:
                auc_perm = roc_auc_score(y_test_perm, y_prob_perm)
                perm_fold_aucs.append(auc_perm)
            except ValueError:
                continue  # Skip if AUC can't be computed

        if perm_fold_aucs:
            perm_aucs.append(np.mean(perm_fold_aucs))

    # Compute p-value
    if perm_aucs:
        p_value = np.mean(np.array(perm_aucs) >= mean_auc)
    else:
        p_value = np.nan

    return {
        'accuracy': mean_accuracy,
        'auc': mean_auc,
        'p_value': p_value,
        'n_folds': len(fold_aucs),
        'n_hit_trials': n_hits,
        'n_miss_trials': n_miss,
        'chance_auc_mean': np.mean(perm_aucs) if perm_aucs else np.nan,
        'chance_auc_std': np.std(perm_aucs) if perm_aucs else np.nan,
    }


def compute_2d_trajectories(sess, common_ids, cd_task, cd_sensory):
    """Compute 2D trajectories through task-state × sensory space.

    Parameters
    ----------
    sess : SessionData
    common_ids : list of int
    cd_task : ndarray, shape (n_units,)
    cd_sensory : ndarray, shape (n_units,)

    Returns
    -------
    dict
        Contains time-resolved projections for different trial types
    """
    trials = sess.trials

    # Define all trial types for comprehensive SDT analysis
    trial_types = {
        'hit': [i for i, t in enumerate(trials)
                if getattr(t, "trialoutcome", None) == "Hit"
                and (getattr(t, "change_size", None) or 1.0) > 1.01],
        'miss': [i for i, t in enumerate(trials)
                 if getattr(t, "trialoutcome", None) == "Miss"
                 and (getattr(t, "change_size", None) or 1.0) > 1.01],
        'sdt_fa': [i for i, t in enumerate(trials)
                   if getattr(t, "trialoutcome", None) == "Hit"
                   and abs((getattr(t, "change_size", None) or 1.0) - 1.0) < 0.05],
        'cr': [i for i, t in enumerate(trials)
               if getattr(t, "trialoutcome", None) == "Miss"
               and abs((getattr(t, "change_size", None) or 1.0) - 1.0) < 0.05],
    }

    trajectories = {}

    for trial_type, trial_indices in trial_types.items():
        if len(trial_indices) < 3:  # Need minimum trials
            continue

        # Build population tensor for this trial type
        try:
            tensor, bin_centers, used_indices = build_population_tensor(
                sess, common_ids, event_name="Change_ON",
                window=TASK_CD_WINDOW, bin_size=BIN_SIZE,
                trial_indices=trial_indices,
            )
        except Exception as e:
            print(f"    Warning: tensor build failed for {trial_type}: {e}")
            continue

        if tensor.shape[0] < 3 or tensor.shape[2] != len(common_ids):
            continue

        # Project onto both coding directions: (n_trials, n_time, n_units) @ (n_units,) = (n_trials, n_time)
        try:
            proj_task = tensor @ cd_task  # broadcasting: (n_trials, n_time, n_units) @ (n_units,)
            proj_sensory = tensor @ cd_sensory
        except Exception as e:
            print(f"    Warning: projection failed for {trial_type}: {e}")
            continue

        # Compute mean trajectory
        mean_proj_task = proj_task.mean(axis=0)
        mean_proj_sensory = proj_sensory.mean(axis=0)

        # Baseline position (average in pre-change window)
        baseline_mask = (bin_centers >= BASELINE_AVG_WINDOW[0]) & (bin_centers < BASELINE_AVG_WINDOW[1])
        if baseline_mask.sum() > 0:
            baseline_task = mean_proj_task[baseline_mask].mean()
            baseline_sensory = mean_proj_sensory[baseline_mask].mean()
        else:
            baseline_task = 0.0
            baseline_sensory = 0.0

        trajectories[trial_type] = {
            'time': bin_centers,
            'proj_task': mean_proj_task,
            'proj_sensory': mean_proj_sensory,
            'baseline_task': baseline_task,
            'baseline_sensory': baseline_sensory,
            'n_trials': tensor.shape[0],
        }

    return trajectories


def process_session(sname, stage, sidx):
    """Process one session for 2D decomposition analysis.

    Returns
    -------
    dict or None
        Results dictionary with keys: session_name, stage, session_idx,
        cos_sim, pval, n_units, cd_task, cd_sensory, trajectories, etc.
    """
    print(f"  Session {sname} ({stage})...", end=" ", flush=True)

    try:
        # Load session
        sess = load_session(sname)
    except FileNotFoundError:
        print("not found")
        return None

    # Load TF traces
    tf_data = load_tf_traces_npz(sname)
    if tf_data is None:
        print("no TF traces")
        del sess
        gc.collect()
        return None

    # Unit selection: intersection of good QC and TF availability
    good_ids = get_good_cluster_ids(sess, min_rate_hz=1.0)
    tf_cluster_ids = set(tf_data['cluster_ids'].astype(int))
    common_ids = sorted(set(good_ids) & tf_cluster_ids)

    if len(common_ids) < MIN_UNITS:
        print(f"{len(common_ids)} common units (need {MIN_UNITS})")
        del sess
        gc.collect()
        return None

    # Compute task-state coding direction
    cd_task, baseline_proj, n_fast_pulses = compute_task_cd(sess, common_ids)
    if cd_task is None:
        print("insufficient go trials")
        del sess
        gc.collect()
        return None

    # Compute sensory coding direction
    cd_sensory, peak_amplitudes = compute_sensory_cd(tf_data, common_ids)
    if cd_sensory is None:
        print("TF peak extraction failed")
        del sess
        gc.collect()
        return None

    # Test orthogonality (on raw CDs, before Gram-Schmidt)
    cos_sim, pval, null_dist = test_orthogonality(cd_task, cd_sensory, common_ids)

    # Gram-Schmidt orthogonalization (Lohse et al. lines 832-840):
    # Order: (1) task-state CD preserved, (2) sensory CD orthogonalized
    cd_sensory_orth = cd_sensory - np.dot(cd_sensory, cd_task) * cd_task
    norm_orth = np.linalg.norm(cd_sensory_orth)
    if norm_orth > 1e-10:
        cd_sensory_orth = cd_sensory_orth / norm_orth
    else:
        cd_sensory_orth = cd_sensory  # fallback if nearly parallel

    # Compute 2D trajectories (use orthogonalized sensory CD for projections)
    trajectories = compute_2d_trajectories(sess, common_ids, cd_task, cd_sensory_orth)

    # Compute 2D decodability (use orthogonalized sensory CD)
    decodability_results = compute_2d_decodability_leave_two_out(sess, common_ids, cd_task, cd_sensory_orth)

    # Validation metrics
    hit_mean = baseline_proj['hit'].mean()
    miss_mean = baseline_proj['miss'].mean()
    task_separation = hit_mean - miss_mean  # Should be > 0 if CD is correct

    if decodability_results is not None:
        print(f"cos_sim={cos_sim:.3f}, p={pval:.3f}, sep={task_separation:.3f}, auc={decodability_results['auc']:.3f}, {len(common_ids)} units")
    else:
        print(f"cos_sim={cos_sim:.3f}, p={pval:.3f}, sep={task_separation:.3f}, {len(common_ids)} units")

    result = {
        'session_name': sname,
        'stage': stage,
        'session_idx': sidx,
        'cos_sim': cos_sim,
        'pval': pval,
        'n_units': len(common_ids),
        'n_fast_pulses': n_fast_pulses,  # Added: number of fast pulses used
        'task_separation': task_separation,
        'hit_baseline_mean': hit_mean,
        'miss_baseline_mean': miss_mean,
        'peak_amplitudes_mean': np.mean(peak_amplitudes),
        'peak_amplitudes_std': np.std(peak_amplitudes),
        'null_mean': null_dist.mean(),
        'null_std': null_dist.std(),
        'cd_task': cd_task,
        'cd_sensory': cd_sensory,
        'common_ids': common_ids,
        'trajectories': trajectories,
        'decodability': decodability_results,
    }

    del sess
    gc.collect()

    return result


def _process_session_worker(args):
    """Module-level worker for ProcessPoolExecutor: load session, run 2D analysis."""
    sname, stage, sidx = args
    try:
        result = process_session(sname, stage, sidx)
        if result is not None:
            dec_info = ""
            if result.get('decodability') is not None:
                dec_info = f", auc={result['decodability']['auc']:.3f}"
            return sname, stage, sidx, result, f"cos_sim={result['cos_sim']:.3f}, p={result['pval']:.3f}, sep={result['task_separation']:.3f}{dec_info}, {result['n_units']} units"
        else:
            return sname, stage, sidx, None, "processing failed"
    except FileNotFoundError:
        return sname, stage, sidx, None, "not found"
    except Exception as e:
        return sname, stage, sidx, None, f"error: {e}"


def main():
    print("[03f] 2D Decomposition: Orthogonality + Trajectory Analysis...")
    manifest = load_staging_manifest(qc_only=True)

    # Prepare tasks for parallel processing
    tasks = []
    for _, row in manifest.iterrows():
        sname = int(row["session_name"])
        stage = row["stage"]
        sidx = row["session_idx"]
        tasks.append((sname, stage, sidx))

    print(f"  Processing {len(tasks)} sessions...")

    # Process sessions with parallel workers
    results = []
    n_workers = min(12, len(tasks))

    if n_workers > 1:
        print(f"  Using {n_workers} parallel workers")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for sname, stage, sidx, result, msg in executor.map(_process_session_worker, tasks):
                print(f"  Session {sname} ({stage})... {msg}")
                if result is not None:
                    results.append(result)
    else:
        # Fallback to sequential processing
        for sname, stage, sidx in tasks:
            print(f"  Session {sname} ({stage})...", end=" ", flush=True)
            result = process_session(sname, stage, sidx)
            if result is not None:
                results.append(result)
                print(f"cos_sim={result['cos_sim']:.3f}, p={result['pval']:.3f}, sep={result['task_separation']:.3f}, {result['n_units']} units")
            else:
                print("failed")

    print(f"\n  Processed {len(results)} sessions successfully")

    if len(results) == 0:
        print("  No valid sessions. Exiting.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Extract decodability metrics for analysis
    decodability_data = []
    for _, row in df.iterrows():
        if row.get('decodability') is not None:
            dec = row['decodability']
            decodability_data.append({
                'session_name': row['session_name'],
                'stage': row['stage'],
                'session_idx': row['session_idx'],
                'auc': dec['auc'],
                'accuracy': dec['accuracy'],
                'p_value': dec['p_value'],
                'n_folds': dec['n_folds'],
                'n_hit_trials': dec['n_hit_trials'],
                'n_miss_trials': dec['n_miss_trials'],
                'chance_auc_mean': dec.get('chance_auc_mean', np.nan),
                'chance_auc_std': dec.get('chance_auc_std', np.nan),
            })

    decodability_df = pd.DataFrame(decodability_data) if decodability_data else pd.DataFrame()

    print(f"\n  Decodability analysis: {len(decodability_df)} sessions with sufficient trials")

    # ══════════════════════════════════════════════════════════════════
    # COMPREHENSIVE FIGURE: Phases 1 + 2 + 3 (Orthogonality + Trajectories + Decodability)
    # ══════════════════════════════════════════════════════════════════

    fig = plt.figure(figsize=(24, 20))
    gs = gridspec.GridSpec(5, 3, hspace=0.35, wspace=0.25, height_ratios=[1, 1.2, 1, 1, 0.8])

    # ── Panel A: Cosine similarity per session ──────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])

    for stage in STAGE_ORDER:
        stage_data = df[df['stage'] == stage]
        if len(stage_data) > 0:
            ax_a.scatter(
                stage_data['session_idx'],
                stage_data['cos_sim'],
                c=STAGE_COLORS[stage],
                s=60,
                alpha=0.7,
                label=f"{stage} (n={len(stage_data)})",
                edgecolors='white',
                linewidths=0.5,
            )

    ax_a.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_a.set_xlabel('Session index')
    ax_a.set_ylabel('Cosine similarity\n(task-state × sensory)')
    ax_a.set_title('A. Orthogonality across sessions')
    ax_a.legend(fontsize=8)
    ax_a.grid(True, alpha=0.3)

    # Add stage background
    add_stage_background(ax_a, manifest)

    # ── Panel B: Trend analysis ──────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])

    # Scatter all points
    ax_b.scatter(df['session_idx'], df['cos_sim'],
                c=[STAGE_COLORS[s] for s in df['stage']],
                s=40, alpha=0.6, edgecolors='white', linewidths=0.3)

    # Trend line
    if len(df) >= 3:
        rho, p_trend = spearmanr(df['session_idx'], df['cos_sim'])

        # Fit line for visualization
        z = np.polyfit(df['session_idx'], df['cos_sim'], 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(df['session_idx'].min(), df['session_idx'].max(), 100)
        ax_b.plot(x_line, p_line(x_line), 'k-', alpha=0.7, linewidth=2)

        ax_b.text(0.05, 0.95, f'ρ = {rho:.3f}\np = {p_trend:.3f}',
                 transform=ax_b.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax_b.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_b.set_xlabel('Session index')
    ax_b.set_ylabel('Cosine similarity')
    ax_b.set_title('B. Trend across learning')
    ax_b.grid(True, alpha=0.3)

    # ── Panel C: By stage comparison ─────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])

    # Box plot
    stage_data = []
    stage_labels = []
    stage_colors = []

    for stage in STAGE_ORDER:
        stage_vals = df[df['stage'] == stage]['cos_sim'].values
        if len(stage_vals) > 0:
            stage_data.append(stage_vals)
            stage_labels.append(f"{stage}\n(n={len(stage_vals)})")
            stage_colors.append(STAGE_COLORS[stage])

    if len(stage_data) >= 2:
        bp = ax_c.boxplot(stage_data, tick_labels=stage_labels, patch_artist=True,
                         showfliers=False, widths=0.6)

        # Color boxes
        for patch, color in zip(bp['boxes'], stage_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Overlay points
        for i, (vals, color) in enumerate(zip(stage_data, stage_colors)):
            x = np.full(len(vals), i + 1)
            jitter = np.random.normal(0, 0.05, len(vals))
            ax_c.scatter(x + jitter, vals, c=color, s=30, alpha=0.8,
                        edgecolors='white', linewidths=0.3, zorder=10)

        # Statistical test
        if len(stage_data) == 2:
            stat, p_stage = mannwhitneyu(stage_data[0], stage_data[1],
                                        alternative='two-sided')
            test_name = "Mann-Whitney U"
        else:
            stat, p_stage = kruskal(*stage_data)
            test_name = "Kruskal-Wallis"

        ax_c.text(0.95, 0.95, f'{test_name}\np = {p_stage:.3f}',
                 transform=ax_c.transAxes, fontsize=9,
                 horizontalalignment='right', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax_c.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_c.set_ylabel('Cosine similarity')
    ax_c.set_title('C. Comparison by stage')
    ax_c.grid(True, alpha=0.3)

    # ── Panel D: Combined Trajectory Comparison ──────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])

    # Collect all Learning and Expert trajectories for overlay (all trial types)
    all_trajectories = {'Learning': {'hit': [], 'miss': [], 'sdt_fa': [], 'cr': []},
                        'Expert': {'hit': [], 'miss': [], 'sdt_fa': [], 'cr': []}}
    for _, row in df.iterrows():
        if row.get('trajectories') is not None:
            trajectories = row['trajectories']
            stage = row['stage']
            for trial_type in ['hit', 'miss', 'sdt_fa', 'cr']:
                if trial_type in trajectories:
                    all_trajectories[stage][trial_type].append(trajectories[trial_type])

    # Plot trajectories with clear stage distinction
    stage_styles = {
        'Learning': {'linestyle': '--', 'alpha': 0.7, 'linewidth': 2, 'marker_size': 60},
        'Expert': {'linestyle': '-', 'alpha': 0.9, 'linewidth': 3, 'marker_size': 100}
    }

    # Updated colors for all four SDT trial types
    trial_colors = {'hit': 'green', 'miss': 'red', 'sdt_fa': 'orange', 'cr': 'blue'}
    trial_labels = {'hit': 'Hit', 'miss': 'Miss', 'sdt_fa': 'SDT FA', 'cr': 'CR'}

    for stage in ['Learning', 'Expert']:
        for trial_type, base_color in trial_colors.items():
            if all_trajectories[stage][trial_type]:
                # Average trajectories across sessions
                all_task_proj = np.array([t['proj_task'] for t in all_trajectories[stage][trial_type]])
                all_sensory_proj = np.array([t['proj_sensory'] for t in all_trajectories[stage][trial_type]])

                mean_task_proj = all_task_proj.mean(axis=0)
                mean_sensory_proj = all_sensory_proj.mean(axis=0)

                # Plot trajectory with stage-specific styling
                style = stage_styles[stage]

                # Use different shades for stage distinction within each trial type
                if stage == 'Learning':
                    if base_color == 'green':
                        color = 'darkgreen'
                    elif base_color == 'red':
                        color = 'darkred'
                    elif base_color == 'orange':
                        color = 'darkorange'
                    else:  # blue
                        color = 'darkblue'
                else:  # Expert
                    if base_color == 'green':
                        color = 'lime'
                    elif base_color == 'red':
                        color = 'crimson'
                    elif base_color == 'orange':
                        color = 'gold'
                    else:  # blue
                        color = 'cyan'

                ax_d.plot(mean_task_proj, mean_sensory_proj,
                         color=color, linestyle=style['linestyle'],
                         alpha=style['alpha'], linewidth=style['linewidth'],
                         label=f'{trial_labels[trial_type]} {stage}')

                # Mark baseline and endpoint
                baseline_task = np.array([t['baseline_task'] for t in all_trajectories[stage][trial_type]]).mean()
                baseline_sensory = np.array([t['baseline_sensory'] for t in all_trajectories[stage][trial_type]]).mean()

                # Baseline marker
                marker = 'o' if stage == 'Learning' else 's'
                ax_d.scatter(baseline_task, baseline_sensory,
                           c=color, s=style['marker_size'], marker=marker,
                           alpha=0.8, edgecolors='white', linewidths=2)

                # Endpoint marker with different shape
                final_task = mean_task_proj[-1]
                final_sensory = mean_sensory_proj[-1]
                endpoint_marker = '^' if stage == 'Learning' else 'v'
                ax_d.scatter(final_task, final_sensory,
                           c=color, s=style['marker_size'], marker=endpoint_marker,
                           alpha=0.8, edgecolors='black', linewidths=2)

    ax_d.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_d.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax_d.set_xlabel('Task-state projection')
    ax_d.set_ylabel('Sensory projection')
    ax_d.set_title('D. All SDT Trial Types: Learning vs Expert')
    ax_d.legend(fontsize=7, ncol=2, loc='best')  # Smaller font and 2 columns for 8 entries
    ax_d.grid(True, alpha=0.3)

    # ── Panel E: Trajectory Magnitude Analysis ──────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])

    # Compute trajectory metrics for statistical comparison
    trajectory_metrics = {'Learning': [], 'Expert': []}

    for _, row in df.iterrows():
        if row.get('trajectories') is not None:
            trajectories = row['trajectories']
            stage = row['stage']

            # Compute session-level metrics (average across trial types)
            session_lengths = []
            session_peak_deflections = []
            session_path_lengths = []
            session_n_trials = []

            for trial_type in ['hit', 'miss', 'sdt_fa', 'cr']:
                if trial_type in trajectories:
                    traj = trajectories[trial_type]

                    # 1. Trajectory length (Euclidean distance from start to end)
                    start_task = traj['baseline_task']
                    start_sensory = traj['baseline_sensory']
                    end_task = traj['proj_task'][-1]
                    end_sensory = traj['proj_sensory'][-1]

                    trajectory_length = np.sqrt((end_task - start_task)**2 + (end_sensory - start_sensory)**2)

                    # 2. Peak deflection (maximum distance from baseline)
                    task_deflections = np.abs(traj['proj_task'] - start_task)
                    sensory_deflections = np.abs(traj['proj_sensory'] - start_sensory)
                    peak_deflection = np.max(np.sqrt(task_deflections**2 + sensory_deflections**2))

                    # 3. Total path length (sum of segment lengths)
                    task_diffs = np.diff(traj['proj_task'])
                    sensory_diffs = np.diff(traj['proj_sensory'])
                    path_length = np.sum(np.sqrt(task_diffs**2 + sensory_diffs**2))

                    session_lengths.append(trajectory_length)
                    session_peak_deflections.append(peak_deflection)
                    session_path_lengths.append(path_length)
                    session_n_trials.append(traj['n_trials'])

            # Average metrics across trial types for this session
            if session_lengths:  # Only if we have valid trial types
                trajectory_metrics[stage].append({
                    'session': row['session_name'],
                    'trajectory_length': np.mean(session_lengths),
                    'peak_deflection': np.mean(session_peak_deflections),
                    'path_length': np.mean(session_path_lengths),
                    'total_trials': sum(session_n_trials),
                    'n_trial_types': len(session_lengths)
                })

    # Statistical comparison of trajectory lengths
    learning_lengths = [m['trajectory_length'] for m in trajectory_metrics['Learning']]
    expert_lengths = [m['trajectory_length'] for m in trajectory_metrics['Expert']]

    if learning_lengths and expert_lengths:
        # Box plot comparison
        box_data = [learning_lengths, expert_lengths]
        box_labels = [f'Learning\n(n={len(learning_lengths)})', f'Expert\n(n={len(expert_lengths)})']

        bp = ax_e.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=False)

        # Color boxes
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Overlay individual points
        for i, (data, color) in enumerate(zip(box_data, ['blue', 'red'])):
            x = np.full(len(data), i + 1)
            jitter = np.random.normal(0, 0.05, len(data))
            ax_e.scatter(x + jitter, data, c=color, s=30, alpha=0.6,
                        edgecolors='white', linewidths=0.5)

        # Statistical test
        stat, p_length = mannwhitneyu(learning_lengths, expert_lengths, alternative='two-sided')

        # Effect size (Cohen's d equivalent for non-parametric)
        mean_learning = np.mean(learning_lengths)
        mean_expert = np.mean(expert_lengths)
        pooled_std = np.sqrt((np.var(learning_lengths) + np.var(expert_lengths)) / 2)
        cohens_d = (mean_expert - mean_learning) / pooled_std if pooled_std > 0 else 0

        ax_e.text(0.95, 0.95, f'Mann-Whitney U\np = {p_length:.4f}\nCohen\'s d = {cohens_d:.3f}',
                 transform=ax_e.transAxes, fontsize=9,
                 horizontalalignment='right', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        print(f"  Trajectory Length Analysis:")
        print(f"    Learning: {mean_learning:.4f} ± {np.std(learning_lengths):.4f}")
        print(f"    Expert: {mean_expert:.4f} ± {np.std(expert_lengths):.4f}")
        print(f"    p = {p_length:.6f}, Cohen's d = {cohens_d:.3f}")

    ax_e.set_ylabel('Trajectory length (Euclidean)')
    ax_e.set_title('E. Trajectory Magnitude Comparison')
    ax_e.grid(True, alpha=0.3)

    # ── Panel F: Pre-change 2D position scatter ─────────────────────────
    ax_f = fig.add_subplot(gs[2, 0])

    # Collect baseline positions across all sessions
    baseline_positions = {'hit': {'task': [], 'sensory': [], 'stage': []},
                         'miss': {'task': [], 'sensory': [], 'stage': []}}

    for _, row in df.iterrows():
        if row.get('trajectories') is not None:
            trajectories = row['trajectories']
            for trial_type in baseline_positions:
                if trial_type in trajectories:
                    baseline_positions[trial_type]['task'].append(trajectories[trial_type]['baseline_task'])
                    baseline_positions[trial_type]['sensory'].append(trajectories[trial_type]['baseline_sensory'])
                    baseline_positions[trial_type]['stage'].append(row['stage'])

    for trial_type, color in [('hit', 'green'), ('miss', 'red')]:
        if baseline_positions[trial_type]['task']:
            for stage in STAGE_ORDER:
                stage_mask = [s == stage for s in baseline_positions[trial_type]['stage']]
                if any(stage_mask):
                    task_vals = np.array(baseline_positions[trial_type]['task'])[stage_mask]
                    sensory_vals = np.array(baseline_positions[trial_type]['sensory'])[stage_mask]

                    marker = 'o' if stage == 'Learning' else 's'
                    ax_f.scatter(task_vals, sensory_vals, c=color, s=50,
                               marker=marker, alpha=0.7, edgecolors='white', linewidths=1,
                               label=f'{trial_type.title()} {stage}')

    ax_f.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_f.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax_f.set_xlabel('Task-state projection')
    ax_f.set_ylabel('Sensory projection')
    ax_f.set_title('F. Pre-change 2D position')
    ax_f.legend(fontsize=7)
    ax_f.grid(True, alpha=0.3)

    # ── Panel G: Pre-change task-state projection ───────────────────────
    ax_g = fig.add_subplot(gs[2, 1])

    hit_task_learning = []
    miss_task_learning = []
    hit_task_expert = []
    miss_task_expert = []

    for _, row in df.iterrows():
        if row.get('trajectories') is not None:
            trajectories = row['trajectories']
            if 'hit' in trajectories:
                if row['stage'] == 'Learning':
                    hit_task_learning.append(trajectories['hit']['baseline_task'])
                else:
                    hit_task_expert.append(trajectories['hit']['baseline_task'])
            if 'miss' in trajectories:
                if row['stage'] == 'Learning':
                    miss_task_learning.append(trajectories['miss']['baseline_task'])
                else:
                    miss_task_expert.append(trajectories['miss']['baseline_task'])

    # Box plot
    box_data = []
    box_labels = []
    box_colors = []

    for stage, hit_vals, miss_vals in [('Learning', hit_task_learning, miss_task_learning),
                                       ('Expert', hit_task_expert, miss_task_expert)]:
        if hit_vals:
            box_data.append(hit_vals)
            box_labels.append(f'Hit\n{stage}')
            box_colors.append('green')
        if miss_vals:
            box_data.append(miss_vals)
            box_labels.append(f'Miss\n{stage}')
            box_colors.append('red')

    if box_data:
        bp = ax_g.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    ax_g.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_g.set_ylabel('Task-state projection')
    ax_g.set_title('G. Pre-change task-state')
    ax_g.grid(True, alpha=0.3)

    # ── Panel H: Peak Deflection Analysis ───────────────────────────────
    ax_h2 = fig.add_subplot(gs[1, 2])  # Move to row 1, column 2 for better layout

    # Extract peak deflections for statistical comparison
    learning_peaks = [m['peak_deflection'] for m in trajectory_metrics['Learning']]
    expert_peaks = [m['peak_deflection'] for m in trajectory_metrics['Expert']]

    if learning_peaks and expert_peaks:
        # Box plot comparison
        box_data = [learning_peaks, expert_peaks]
        box_labels = [f'Learning\n(n={len(learning_peaks)})', f'Expert\n(n={len(expert_peaks)})']

        bp = ax_h2.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=False)

        # Color boxes
        colors = ['lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Overlay individual points
        for i, (data, color) in enumerate(zip(box_data, ['green', 'red'])):
            x = np.full(len(data), i + 1)
            jitter = np.random.normal(0, 0.05, len(data))
            ax_h2.scatter(x + jitter, data, c=color, s=30, alpha=0.6,
                         edgecolors='white', linewidths=0.5)

        # Statistical test
        stat, p_peak = mannwhitneyu(learning_peaks, expert_peaks, alternative='two-sided')

        # Effect size
        mean_learning_peak = np.mean(learning_peaks)
        mean_expert_peak = np.mean(expert_peaks)
        pooled_std_peak = np.sqrt((np.var(learning_peaks) + np.var(expert_peaks)) / 2)
        cohens_d_peak = (mean_expert_peak - mean_learning_peak) / pooled_std_peak if pooled_std_peak > 0 else 0

        ax_h2.text(0.95, 0.95, f'Mann-Whitney U\np = {p_peak:.4f}\nCohen\'s d = {cohens_d_peak:.3f}',
                  transform=ax_h2.transAxes, fontsize=9,
                  horizontalalignment='right', verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        print(f"  Peak Deflection Analysis:")
        print(f"    Learning: {mean_learning_peak:.4f} ± {np.std(learning_peaks):.4f}")
        print(f"    Expert: {mean_expert_peak:.4f} ± {np.std(expert_peaks):.4f}")
        print(f"    p = {p_peak:.6f}, Cohen's d = {cohens_d_peak:.3f}")

    ax_h2.set_ylabel('Peak deflection from baseline')
    ax_h2.set_title('H. Peak Deflection Comparison')
    ax_h2.grid(True, alpha=0.3)

    # ── Panel I: 2D Decodability Analysis ──────────────────────────────────
    ax_i = fig.add_subplot(gs[3, 0])

    if len(decodability_df) > 0:
        # Scatter plot of AUC values by stage
        for stage in STAGE_ORDER:
            stage_dec_data = decodability_df[decodability_df['stage'] == stage]
            if len(stage_dec_data) > 0:
                ax_i.scatter(
                    stage_dec_data['session_idx'],
                    stage_dec_data['auc'],
                    c=STAGE_COLORS[stage],
                    s=60,
                    alpha=0.7,
                    label=f"{stage} (n={len(stage_dec_data)})",
                    edgecolors='white',
                    linewidths=0.5,
                )

        # Add chance line (0.5 for binary classification)
        ax_i.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Chance (0.5)')

        ax_i.set_xlabel('Session index')
        ax_i.set_ylabel('AUC (Hit vs Miss)')
        ax_i.set_title('I. 2D Decodability across sessions')
        ax_i.legend(fontsize=8)
        ax_i.grid(True, alpha=0.3)
        ax_i.set_ylim(0.4, 1.0)

        # Add stage background
        add_stage_background(ax_i, manifest)
    else:
        ax_i.text(0.5, 0.5, 'No decodability data\n(insufficient trials)',
                 ha='center', va='center', transform=ax_i.transAxes)
        ax_i.set_title('I. 2D Decodability')

    # ── Panel J: Decodability by Stage ──────────────────────────────────────
    ax_j = fig.add_subplot(gs[3, 1])

    if len(decodability_df) > 0:
        # Box plot comparison
        dec_stage_data = []
        dec_stage_labels = []
        dec_stage_colors = []

        for stage in STAGE_ORDER:
            stage_aucs = decodability_df[decodability_df['stage'] == stage]['auc'].values
            if len(stage_aucs) > 0:
                dec_stage_data.append(stage_aucs)
                dec_stage_labels.append(f"{stage}\n(n={len(stage_aucs)})")
                dec_stage_colors.append(STAGE_COLORS[stage])

        if len(dec_stage_data) >= 2:
            bp = ax_j.boxplot(dec_stage_data, tick_labels=dec_stage_labels, patch_artist=True,
                             showfliers=False, widths=0.6)

            # Color boxes
            for patch, color in zip(bp['boxes'], dec_stage_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Overlay points
            for i, (vals, color) in enumerate(zip(dec_stage_data, dec_stage_colors)):
                x = np.full(len(vals), i + 1)
                jitter = np.random.normal(0, 0.05, len(vals))
                ax_j.scatter(x + jitter, vals, c=color, s=30, alpha=0.8,
                            edgecolors='white', linewidths=0.3, zorder=10)

            # Statistical test
            if len(dec_stage_data) == 2:
                stat, p_dec_stage = mannwhitneyu(dec_stage_data[0], dec_stage_data[1],
                                                alternative='two-sided')
                test_name = "Mann-Whitney U"
            else:
                stat, p_dec_stage = kruskal(*dec_stage_data)
                test_name = "Kruskal-Wallis"

            ax_j.text(0.95, 0.95, f'{test_name}\np = {p_dec_stage:.3f}',
                     transform=ax_j.transAxes, fontsize=9,
                     horizontalalignment='right', verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax_j.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
        ax_j.set_ylabel('AUC (Hit vs Miss)')
        ax_j.set_title('J. Decodability by stage')
        ax_j.grid(True, alpha=0.3)
        ax_j.set_ylim(0.4, 1.0)
    else:
        ax_j.text(0.5, 0.5, 'No decodability data', ha='center', va='center',
                 transform=ax_j.transAxes)
        ax_j.set_title('J. Decodability by stage')

    # ── Panel K: Decodability vs Orthogonality ────────────────────────────
    ax_k = fig.add_subplot(gs[3, 2])

    if len(decodability_df) > 0:
        # Merge decodability with orthogonality data
        merged = pd.merge(df[['session_name', 'cos_sim', 'stage']],
                         decodability_df[['session_name', 'auc']],
                         on='session_name', how='inner')

        if len(merged) > 0:
            # Scatter plot
            for stage in STAGE_ORDER:
                stage_merged = merged[merged['stage'] == stage]
                if len(stage_merged) > 0:
                    ax_k.scatter(
                        stage_merged['cos_sim'],
                        stage_merged['auc'],
                        c=STAGE_COLORS[stage],
                        s=50,
                        alpha=0.7,
                        label=stage,
                        edgecolors='white',
                        linewidths=0.5,
                    )

            # Correlation analysis
            if len(merged) >= 3:
                rho_dec, p_corr_dec = spearmanr(merged['cos_sim'], merged['auc'])

                ax_k.text(0.05, 0.95, f'ρ = {rho_dec:.3f}\np = {p_corr_dec:.3f}',
                         transform=ax_k.transAxes, fontsize=9,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax_k.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
            ax_k.axvline(0, color='gray', linestyle='--', alpha=0.7)
            ax_k.set_xlabel('Cosine similarity')
            ax_k.set_ylabel('AUC (Hit vs Miss)')
            ax_k.set_title('K. Decodability vs Orthogonality')
            ax_k.legend(fontsize=8)
            ax_k.grid(True, alpha=0.3)
            ax_k.set_ylim(0.4, 1.0)
        else:
            ax_k.text(0.5, 0.5, 'No merged data', ha='center', va='center',
                     transform=ax_k.transAxes)
    else:
        ax_k.text(0.5, 0.5, 'No decodability data', ha='center', va='center',
                 transform=ax_k.transAxes)
        ax_k.set_title('K. Decodability vs Orthogonality')

    # ── Compute statistics before figure summary ─────────────────────────
    stats = []

    # Overall orthogonality test (all sessions vs 0)
    all_cos_sim = df['cos_sim'].values
    if len(all_cos_sim) >= 3:
        stat_w, p_w = wilcoxon(all_cos_sim)
        stats.append({
            'test': 'orthogonality_all_sessions_wilcoxon',
            'statistic': stat_w,
            'p': p_w,
            'mean_cos_sim': all_cos_sim.mean(),
            'std_cos_sim': all_cos_sim.std(),
            'n': len(all_cos_sim),
        })

    # ── Panel L: Summary stats table ──────────────────────────────────────
    ax_l = fig.add_subplot(gs[4, :])
    ax_l.axis('off')

    # Create summary table with trajectory and decodability metrics
    n_learning_traj = len([m for m in trajectory_metrics.get('Learning', [])])
    n_expert_traj = len([m for m in trajectory_metrics.get('Expert', [])])

    n_dec_learning = len(decodability_df[decodability_df['stage'] == 'Learning']) if len(decodability_df) > 0 else 0
    n_dec_expert = len(decodability_df[decodability_df['stage'] == 'Expert']) if len(decodability_df) > 0 else 0

    mean_auc_learning = decodability_df[decodability_df['stage'] == 'Learning']['auc'].mean() if n_dec_learning > 0 else np.nan
    mean_auc_expert = decodability_df[decodability_df['stage'] == 'Expert']['auc'].mean() if n_dec_expert > 0 else np.nan

    summary_text = f"""
    SUMMARY STATISTICS:
    • Sessions analyzed: {len(df)} ({len(df[df['stage']=='Learning'])} Learning, {len(df[df['stage']=='Expert'])} Expert)
    • Mean cosine similarity: {df['cos_sim'].mean():.4f} ± {df['cos_sim'].std():.4f}
    • Orthogonality test (all): p = {stats[0]['p'] if stats else 'N/A':.4f}
    • Task separation: {df['task_separation'].mean():.3f} ± {df['task_separation'].std():.3f}
    • Units per session: {df['n_units'].mean():.0f} ± {df['n_units'].std():.0f}
    • Trajectory sessions: {n_learning_traj} Learning, {n_expert_traj} Expert
    • 2D Decodability: {n_dec_learning} Learning (AUC: {mean_auc_learning:.3f}), {n_dec_expert} Expert (AUC: {mean_auc_expert:.3f})
    """

    ax_l.text(0.05, 0.5, summary_text, transform=ax_l.transAxes, fontsize=11,
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    # Overall title
    fig.suptitle('2D Decomposition: Orthogonality, Trajectory Magnitude & Decodability Analysis\n(Learning: dashed lines, Expert: solid lines)',
                fontsize=16, fontweight='bold')

    # Save comprehensive figure
    save_figure(fig, "fig17b_2d_decomposition_corrected", "03_population")

    # ── Complete statistics computation ───────────────────────────────────

    # Per-stage orthogonality
    for stage in STAGE_ORDER:
        stage_vals = df[df['stage'] == stage]['cos_sim'].values
        if len(stage_vals) >= 3:
            stat_w, p_w = wilcoxon(stage_vals)
            stats.append({
                'test': f'orthogonality_{stage}_wilcoxon',
                'statistic': stat_w,
                'p': p_w,
                'mean_cos_sim': stage_vals.mean(),
                'std_cos_sim': stage_vals.std(),
                'n': len(stage_vals),
            })

    # Trend test
    if len(df) >= 3:
        rho, p_trend = spearmanr(df['session_idx'], df['cos_sim'])
        stats.append({
            'test': 'cos_sim_trend_spearman',
            'rho': rho,
            'p': p_trend,
            'n': len(df),
        })

    # Stage comparison
    if len(stage_data) >= 2:
        if len(stage_data) == 2:
            stat_s, p_s = mannwhitneyu(stage_data[0], stage_data[1])
            stats.append({
                'test': 'cos_sim_stage_comparison_mannwhitney',
                'statistic': stat_s,
                'p': p_s,
            })
        else:
            stat_s, p_s = kruskal(*stage_data)
            stats.append({
                'test': 'cos_sim_stage_comparison_kruskal',
                'statistic': stat_s,
                'p': p_s,
            })

    # Task separation validation
    if len(df) > 0:
        task_seps = df['task_separation'].values
        stat_w, p_w = wilcoxon(task_seps)
        stats.append({
            'test': 'task_separation_validation_wilcoxon',
            'statistic': stat_w,
            'p': p_w,
            'mean_separation': task_seps.mean(),
            'std_separation': task_seps.std(),
            'n': len(task_seps),
            'description': 'Hit > Miss baseline projection (should be positive)',
        })

    # 2D Decodability statistics
    if len(decodability_df) > 0:
        # Overall decodability above chance (0.5)
        all_aucs = decodability_df['auc'].values
        if len(all_aucs) >= 3:
            # Test if AUCs are significantly above chance (0.5)
            aucs_minus_chance = all_aucs - 0.5
            stat_w, p_w = wilcoxon(aucs_minus_chance, alternative='greater')
            stats.append({
                'test': 'decodability_above_chance_wilcoxon',
                'statistic': stat_w,
                'p': p_w,
                'mean_auc': all_aucs.mean(),
                'std_auc': all_aucs.std(),
                'n': len(all_aucs),
                'description': 'AUC > 0.5 (above chance)',
            })

        # Decodability by stage comparison
        learning_aucs = decodability_df[decodability_df['stage'] == 'Learning']['auc'].values
        expert_aucs = decodability_df[decodability_df['stage'] == 'Expert']['auc'].values

        if len(learning_aucs) >= 3 and len(expert_aucs) >= 3:
            stat_mw, p_mw = mannwhitneyu(learning_aucs, expert_aucs, alternative='two-sided')
            stats.append({
                'test': 'decodability_stage_comparison_mannwhitney',
                'statistic': stat_mw,
                'p': p_mw,
                'learning_mean_auc': learning_aucs.mean(),
                'expert_mean_auc': expert_aucs.mean(),
                'learning_n': len(learning_aucs),
                'expert_n': len(expert_aucs),
            })

        # Correlation between decodability and orthogonality
        merged_dec_ortho = pd.merge(df[['session_name', 'cos_sim']],
                                   decodability_df[['session_name', 'auc']],
                                   on='session_name', how='inner')
        if len(merged_dec_ortho) >= 3:
            rho_dec, p_dec = spearmanr(merged_dec_ortho['cos_sim'], merged_dec_ortho['auc'])
            stats.append({
                'test': 'decodability_orthogonality_correlation_spearman',
                'rho': rho_dec,
                'p': p_dec,
                'n': len(merged_dec_ortho),
            })

    # Convert stats to DataFrame and save
    stats_df = pd.DataFrame(stats)
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "03_population", "2d_decomposition_corrected_stats.csv"
    )
    stats_df.to_csv(stats_path, index=False)

    # Save session-level results
    results_path = os.path.join(CACHE_DIR, "2d_decomposition_phase1_sessions.csv")
    df.to_csv(results_path, index=False)

    print(f"\n  Saved figure and stats")
    print(f"  Session results: {results_path}")
    print(f"  Stats: {stats_path}")

    # Summary
    print(f"\n  SUMMARY:")
    print(f"  Sessions: {len(df)}")
    print(f"  Mean cosine similarity: {all_cos_sim.mean():.4f} ± {all_cos_sim.std():.4f}")
    print(f"  Orthogonality (all): p = {stats[0]['p']:.4f}" if stats else "")
    print(f"  Task separation (validation): {df['task_separation'].mean():.4f} ± {df['task_separation'].std():.4f}")

    if len(decodability_df) > 0:
        print(f"  2D Decodability: {len(decodability_df)} sessions, AUC = {decodability_df['auc'].mean():.3f} ± {decodability_df['auc'].std():.3f}")

    # Per-stage summary
    for stage in STAGE_ORDER:
        stage_data = df[df['stage'] == stage]
        if len(stage_data) > 0:
            stage_dec = decodability_df[decodability_df['stage'] == stage] if len(decodability_df) > 0 else pd.DataFrame()
            dec_info = f", AUC = {stage_dec['auc'].mean():.3f}" if len(stage_dec) > 0 else ""
            print(f"  {stage}: {len(stage_data)} sessions, cos_sim = {stage_data['cos_sim'].mean():.4f}{dec_info}")


if __name__ == "__main__":
    main()