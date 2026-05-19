"""Fig26: 3D Coding Space Analysis — Task-state × Sensory × Lick dimensions (METHOD 3C).

Maps lick-responsive neurons onto a 3D decomposition framework:
1. Task-state CD: Hit vs Miss (fast TF pulse pre-activity, Lohse method)
2. Sensory CD: Fast TF pulse responsiveness (post-pulse peaks)
3. Lick CD: Pre-lick vs Post-lick activity (Method 3C - lick-aligned)

OPTIONS for future exploration:
- Method 3A: Early-FA vs Late-FA coding direction
- Method 3B: Hit vs Catch-CR coding direction
- Method 3D: Lick-aligned ramping activity

CORRECTED METHOD (April 2026):
- Task-state CD uses FAST TF pulses only with constraints
- Capitalized trial outcomes ('Hit', 'Miss', 'FA')
- 3D orthogonality testing with permutation nulls
- Method 3C as primary lick dimension

Saves: figures/06_lick_motor/fig26_3d_coding_space.png
Stats: figures/06_lick_motor/3d_coding_space_stats.csv
"""
import os
import sys
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Suite infrastructure
from visdetect.suite.config import STAGE_COLORS, OUTCOME_COLORS, CACHE_DIR, SESSION_FILTER, DEFAULT_BIN_SIZE
from visdetect.suite.loader import (
    load_staging_manifest,
    load_session,
    load_lick_responsiveness,
    load_all_lick_responsiveness,
    load_tf_traces_npz
)
from visdetect.analysis.utils import (
    get_good_cluster_ids,
    build_population_tensor,
    compute_zscore_normalized,
    compute_lda_cd,
    permutation_test,
    bootstrap_ci
)
from visdetect.suite.plotting import setup_style, save_figure, add_stage_background

# Library imports
from visdetect.analysis.constants import (
    TF_PULSE_PRE_WINDOW,
    TF_PULSE_POST_WINDOW,
    FA_RT_SPLIT
)
from visdetect.analysis.align import get_event_times_by_trial, align_spikes_to_events

setup_style()

# Parameters
MIN_UNITS = 10
MIN_TRIALS_PER_CLASS = 8
MIN_LICKS = 10  # Minimum licks needed for lick CD
N_PERM_ORTHO = 1000
CD_REG = 0.01

# Lick alignment windows (Method 3C)
LICK_PRE_WINDOW = (-0.5, -0.1)   # Pre-lick ramping/preparation
LICK_POST_WINDOW = (0.1, 0.5)    # Post-lick execution/aftermath
LICK_ALIGN_WINDOW = (-0.6, 0.6)  # Full window for alignment

# Future method options (not implemented yet)
LICK_METHOD = "3C"  # Options: "3A" (Early vs Late FA), "3B" (Hit vs Catch-CR), "3C" (Pre vs Post), "3D" (Ramping)

# ── Cache management ────────────────────────────────────────────
CACHE_FILE = os.path.join(CACHE_DIR, "3d_coding_space_method3c.csv")


def _compute_task_state_cd(session, good_ids, tf_data):
    """Compute task-state coding direction using FAST TF pulse alignment (Lohse method)."""
    from visdetect.analysis.tf_pulse import _collect_pulses, TFRespPulseConfig

    trials = session.trials

    # Get FAST pulses with constraints (matching f_2d_decomposition.py)
    cfg = TFRespPulseConfig(use_constraints=True)
    fast_times, slow_times = _collect_pulses(session, cfg, show_progress=False)

    if len(fast_times) < 20:
        return None, 0, 0

    # Get trial outcomes for each pulse
    baseline_times = get_event_times_by_trial(session, "Baseline_ON")
    change_times = get_event_times_by_trial(session, "Change_ON")

    # Collect valid pulse-outcome pairs
    valid_pulses = []
    valid_outcomes = []

    for pulse_time in fast_times:
        for i, trial in enumerate(trials):
            if (i < len(baseline_times) and i < len(change_times) and
                np.isfinite(baseline_times[i]) and np.isfinite(change_times[i])):
                if baseline_times[i] <= pulse_time <= change_times[i]:
                    outcome = getattr(trial, 'trialoutcome', None)
                    change_size = getattr(trial, 'change_size', 1.0)
                    if outcome in ['Hit', 'Miss'] and change_size > 1.01:
                        valid_pulses.append(pulse_time)
                        valid_outcomes.append(outcome)
                    break

    if len(valid_pulses) < 20:
        return None, 0, 0

    # Build activity matrix aligned to valid pulses
    pulse_activities = []
    for i, cluster in enumerate(session.clusters):
        if int(cluster.cluster_id) not in good_ids:
            continue

        spike_times = np.array(cluster.spike_times, dtype=float)

        # Align spikes to pulse times, get pre-pulse activity
        aligned_matrix, bin_centers = align_spikes_to_events(
            spike_times, valid_pulses, LICK_ALIGN_WINDOW, DEFAULT_BIN_SIZE
        )

        # Average in pre-pulse window
        pre_mask = (bin_centers >= TF_PULSE_PRE_WINDOW[0]) & (bin_centers < TF_PULSE_PRE_WINDOW[1])
        if np.any(pre_mask):
            pre_activity = np.mean(aligned_matrix[:, pre_mask], axis=1)  # (n_pulses,)
            pulse_activities.append(pre_activity)

    if len(pulse_activities) == 0:
        return None, 0, 0

    # Stack into matrix: (n_pulses, n_units)
    pulse_matrix = np.column_stack(pulse_activities)

    # Create labels: Hit=1, Miss=0
    labels = np.array([1 if outcome == 'Hit' else 0 for outcome in valid_outcomes])

    # Check we have both classes
    if len(np.unique(labels)) < 2 or np.sum(labels) < MIN_TRIALS_PER_CLASS or np.sum(1-labels) < MIN_TRIALS_PER_CLASS:
        return None, 0, 0

    # Compute task-state coding direction
    cd_task = compute_lda_cd(pulse_matrix, labels, method="manual", reg=CD_REG, reg_style="trace_scaled")

    return cd_task, len(fast_times), len(valid_pulses)


def _compute_sensory_cd(tf_data, good_ids):
    """Extract sensory coding direction from FAST TF pulse responses."""
    if tf_data is None or 'cluster_ids' not in tf_data:
        return None

    tf_ids = tf_data['cluster_ids']
    common_ids = sorted(set(good_ids) & set(tf_ids))

    if len(common_ids) < 10:
        return None

    # Map to indices and extract fast pulse responses
    tf_id_to_idx = {cid: i for i, cid in enumerate(tf_ids)}
    common_indices = [tf_id_to_idx[cid] for cid in common_ids]

    t_vec = tf_data['t_vec']
    fast_z = tf_data['fast_z'][common_indices]

    # Peak in post-pulse window
    post_mask = (t_vec >= 0.0) & (t_vec < 0.2)
    if not np.any(post_mask):
        return None

    # Signed peak amplitudes
    peak_responses = []
    for unit_trace in fast_z:
        post_trace = unit_trace[post_mask]
        pos_peak = np.max(post_trace)
        neg_peak = np.min(post_trace)
        signed_peak = pos_peak if abs(pos_peak) > abs(neg_peak) else neg_peak
        peak_responses.append(signed_peak)

    cd_sensory = np.array(peak_responses)
    if np.linalg.norm(cd_sensory) > 0:
        cd_sensory = cd_sensory / np.linalg.norm(cd_sensory)

    return cd_sensory


def _compute_lick_cd_method3c(session, good_ids):
    """Compute lick coding direction using Method 3C: Pre-lick vs Post-lick activity."""

    # Collect all lick times (FA + Hit licks)
    lick_times = []

    for trial in session.trials:
        outcome = getattr(trial, 'trialoutcome', None)

        if outcome == 'FA':
            # FA lick time
            rt_dict = getattr(trial, 'reactiontimes', {}) or {}
            fa_rt = rt_dict.get('fa', np.nan)
            if np.isfinite(fa_rt):
                # FA RT is relative to baseline onset
                baseline_times = get_event_times_by_trial(session, "Baseline_ON")
                trial_idx = session.trials.index(trial)
                if trial_idx < len(baseline_times) and np.isfinite(baseline_times[trial_idx]):
                    lick_time = baseline_times[trial_idx] + fa_rt
                    lick_times.append(lick_time)

        elif outcome == 'Hit':
            # Hit lick time
            rt_dict = getattr(trial, 'reactiontimes', {}) or {}
            hit_rt = rt_dict.get('hit', np.nan)
            if np.isfinite(hit_rt):
                # Hit RT is relative to change onset
                change_times = get_event_times_by_trial(session, "Change_ON")
                trial_idx = session.trials.index(trial)
                if trial_idx < len(change_times) and np.isfinite(change_times[trial_idx]):
                    lick_time = change_times[trial_idx] + hit_rt
                    lick_times.append(lick_time)

    if len(lick_times) < MIN_LICKS:
        return None, 0

    # Build activity matrix aligned to lick times
    lick_activities = []
    for cluster in session.clusters:
        if int(cluster.cluster_id) not in good_ids:
            continue

        spike_times = np.array(cluster.spike_times, dtype=float)

        # Align spikes to lick times
        aligned_matrix, bin_centers = align_spikes_to_events(
            spike_times, lick_times, LICK_ALIGN_WINDOW, DEFAULT_BIN_SIZE
        )

        # Average in pre-lick and post-lick windows
        pre_mask = (bin_centers >= LICK_PRE_WINDOW[0]) & (bin_centers < LICK_PRE_WINDOW[1])
        post_mask = (bin_centers >= LICK_POST_WINDOW[0]) & (bin_centers < LICK_POST_WINDOW[1])

        if np.any(pre_mask) and np.any(post_mask):
            pre_activity = np.mean(aligned_matrix[:, pre_mask], axis=1)   # (n_licks,)
            post_activity = np.mean(aligned_matrix[:, post_mask], axis=1) # (n_licks,)

            # Stack pre and post for each lick
            unit_activity = np.column_stack([pre_activity, post_activity])  # (n_licks, 2)
            lick_activities.append(unit_activity)

    if len(lick_activities) == 0:
        return None, 0

    # Combine across units: (n_licks, 2*n_units)
    combined_activity = np.hstack(lick_activities)

    # Create labels: Pre-lick=1, Post-lick=0, repeated for each lick
    n_licks = len(lick_times)
    labels = np.tile([1, 0], n_licks)  # [1,0,1,0,1,0...]
    activity_matrix = combined_activity.reshape(-1, combined_activity.shape[-1])  # (2*n_licks, 2*n_units)

    # Compute lick coding direction
    cd_lick = compute_lda_cd(activity_matrix, labels, method="manual", reg=CD_REG, reg_style="trace_scaled")

    # Extract the unit-level projections (first n_units elements)
    n_units = len([c for c in session.clusters if int(c.cluster_id) in good_ids])
    cd_lick_units = cd_lick[:n_units]  # Take first n_units elements

    # Normalize
    if np.linalg.norm(cd_lick_units) > 0:
        cd_lick_units = cd_lick_units / np.linalg.norm(cd_lick_units)

    return cd_lick_units, len(lick_times)


def _test_3d_orthogonality(cd_task, cd_sensory, cd_lick, n_perm=N_PERM_ORTHO):
    """Test pairwise orthogonality between all three coding directions."""

    # Compute all pairwise cosine similarities
    cos_task_sensory = np.dot(cd_task, cd_sensory)
    cos_task_lick = np.dot(cd_task, cd_lick)
    cos_sensory_lick = np.dot(cd_sensory, cd_lick)

    # Permutation tests for each pair
    np.random.seed(42)

    # Task vs Sensory
    null_ts = np.zeros(n_perm)
    for i in range(n_perm):
        perm_sensory = cd_sensory[np.random.permutation(len(cd_sensory))]
        null_ts[i] = np.dot(cd_task, perm_sensory)
    p_task_sensory = np.mean(np.abs(null_ts) >= np.abs(cos_task_sensory))

    # Task vs Lick
    null_tl = np.zeros(n_perm)
    for i in range(n_perm):
        perm_lick = cd_lick[np.random.permutation(len(cd_lick))]
        null_tl[i] = np.dot(cd_task, perm_lick)
    p_task_lick = np.mean(np.abs(null_tl) >= np.abs(cos_task_lick))

    # Sensory vs Lick
    null_sl = np.zeros(n_perm)
    for i in range(n_perm):
        perm_lick = cd_lick[np.random.permutation(len(cd_lick))]
        null_sl[i] = np.dot(cd_sensory, perm_lick)
    p_sensory_lick = np.mean(np.abs(null_sl) >= np.abs(cos_sensory_lick))

    return {
        'cos_task_sensory': cos_task_sensory,
        'cos_task_lick': cos_task_lick,
        'cos_sensory_lick': cos_sensory_lick,
        'p_task_sensory': p_task_sensory,
        'p_task_lick': p_task_lick,
        'p_sensory_lick': p_sensory_lick,
        'null_task_sensory': null_ts,
        'null_task_lick': null_tl,
        'null_sensory_lick': null_sl
    }


def _classify_early_late_fa(session):
    """Classify FA trials as early vs late using FA_RT_SPLIT."""
    trials = getattr(session, "trials", [])
    early_fa_trials = []
    late_fa_trials = []

    for i, trial in enumerate(trials):
        if getattr(trial, 'trialoutcome', None) == 'FA':
            rt_dict = getattr(trial, 'reactiontimes', {}) or {}
            fa_rt = rt_dict.get('fa', np.nan)

            if np.isfinite(fa_rt):
                if fa_rt <= FA_RT_SPLIT:
                    early_fa_trials.append(i)
                else:
                    late_fa_trials.append(i)

    return early_fa_trials, late_fa_trials


def load_lick_cache_data():
    """Load lick responsiveness from pre_lick_ramping.csv cache."""
    cache_path = os.path.join(CACHE_DIR, "pre_lick_ramping.csv")
    if not os.path.exists(cache_path):
        print(f"  WARNING: Lick cache not found at {cache_path}")
        return {}

    df = pd.read_csv(cache_path)
    lick_data = {}

    for _, row in df.iterrows():
        session_name = int(row['session_name'])
        cluster_id = int(row['cluster_id'])
        is_lick_responsive = bool(row.get('is_ramping', False))

        if session_name not in lick_data:
            lick_data[session_name] = {}

        lick_data[session_name][cluster_id] = {
            'lick_responsive': is_lick_responsive,
            'early_fa_responsive': is_lick_responsive,  # Simplification for now
            'late_fa_responsive': is_lick_responsive,
            'hit_responsive': is_lick_responsive,
            'ramp_rho': row.get('ramp_rho', np.nan),
            'ramp_p': row.get('ramp_p', np.nan)
        }

    print(f"  Loaded lick data for {len(lick_data)} sessions")
    return lick_data


def compute_or_load(force=False, use_parallel=False):
    """Main computation function for 3D coding space analysis."""
    if os.path.exists(CACHE_FILE) and not force:
        return pd.read_csv(CACHE_FILE)

    manifest = load_staging_manifest(qc_only=True)

    # Load lick responsiveness from cache
    all_lick_data = load_lick_cache_data()

    # Process sessions (start with sequential, add parallel later)
    rows = []

    for _, mrow in manifest.iterrows():
        sname = str(mrow["session_name"])
        stage = str(mrow["stage"])
        sname_int = int(sname)
        lick_session_data = all_lick_data.get(sname_int, {})

        print(f"Processing {sname} ({stage})...")

        sess = load_session(sname)
        good_ids = get_good_cluster_ids(sess)

        if len(good_ids) < MIN_UNITS:
            print(f"  Insufficient units ({len(good_ids)} < {MIN_UNITS})")
            del sess; gc.collect()
            continue

        # Load TF pulse data
        try:
            tf_data = load_tf_traces_npz(sname)
        except Exception as e:
            print(f"  No TF data for {sname}, skipping... ({e})")
            del sess; gc.collect()
            continue

        # Find common units
        tf_ids = tf_data['cluster_ids'] if tf_data else []
        common_ids = sorted(set(good_ids) & set(tf_ids))

        if len(common_ids) < MIN_UNITS:
            print(f"  Insufficient common units ({len(common_ids)} < {MIN_UNITS})")
            del sess; gc.collect()
            continue

        # Compute all 3 coding directions
        print(f"  Computing 3D coding directions...")

        # 1. Task-state CD (Hit vs Miss, fast TF pulses)
        cd_task, n_fast_pulses, n_valid_pulses = _compute_task_state_cd(sess, common_ids, tf_data)

        # 2. Sensory CD (TF responsiveness)
        cd_sensory = _compute_sensory_cd(tf_data, common_ids)

        # 3. Lick CD (Method 3C: Pre-lick vs Post-lick)
        cd_lick, n_licks = _compute_lick_cd_method3c(sess, common_ids)

        if cd_task is None or cd_sensory is None or cd_lick is None:
            print(f"  Failed to compute 3D coding directions")
            del sess; gc.collect()
            continue

        # Test 3D orthogonality
        ortho_results = _test_3d_orthogonality(cd_task, cd_sensory, cd_lick)

        # FA classification
        early_fa_trials, late_fa_trials = _classify_early_late_fa(sess)

        print(f"  3D: {len(common_ids)} units, {n_fast_pulses} pulses, {n_licks} licks")
        print(f"  Orthogonality: T-S={ortho_results['cos_task_sensory']:.3f}, T-L={ortho_results['cos_task_lick']:.3f}, S-L={ortho_results['cos_sensory_lick']:.3f}")

        # Store results for each unit
        for i, unit_id in enumerate(common_ids):
            if i >= len(cd_task) or i >= len(cd_sensory) or i >= len(cd_lick):
                continue

            task_projection = cd_task[i]
            sensory_projection = cd_sensory[i]
            lick_projection = cd_lick[i]

            # Lick responsiveness
            lick_data = lick_session_data.get(unit_id, {})
            lick_responsive = lick_data.get('lick_responsive', False)
            early_fa_responsive = lick_data.get('early_fa_responsive', False)
            late_fa_responsive = lick_data.get('late_fa_responsive', False)
            hit_responsive = lick_data.get('hit_responsive', False)

            rows.append({
                'session_name': sname,
                'stage': stage,
                'cluster_id': unit_id,
                # 3D projections
                'task_projection': task_projection,
                'sensory_projection': sensory_projection,
                'lick_projection': lick_projection,
                # Orthogonality (session-level)
                'cos_task_sensory': ortho_results['cos_task_sensory'],
                'cos_task_lick': ortho_results['cos_task_lick'],
                'cos_sensory_lick': ortho_results['cos_sensory_lick'],
                'p_task_sensory': ortho_results['p_task_sensory'],
                'p_task_lick': ortho_results['p_task_lick'],
                'p_sensory_lick': ortho_results['p_sensory_lick'],
                # Counts
                'n_fast_pulses': n_fast_pulses,
                'n_valid_pulses': n_valid_pulses,
                'n_licks': n_licks,
                'n_common_units': len(common_ids),
                # Lick responsiveness
                'lick_responsive': lick_responsive,
                'early_fa_responsive': early_fa_responsive,
                'late_fa_responsive': late_fa_responsive,
                'hit_responsive': hit_responsive,
                # FA trials
                'n_early_fa_trials': len(early_fa_trials),
                'n_late_fa_trials': len(late_fa_trials),
                # Method identifier
                'lick_method': LICK_METHOD
            })

        print(f"  Added {len(common_ids)} units to 3D analysis")
        del sess; gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(CACHE_FILE, index=False)
    return df


def plot_3d_coding_space(df):
    """Create comprehensive 3D coding space visualization."""

    df_valid = df.dropna(subset=['task_projection', 'sensory_projection', 'lick_projection'])

    if len(df_valid) == 0:
        print("No valid sessions for 3D analysis")
        return

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 5, hspace=0.4, wspace=0.35)

    # Panel A: 3D scatter plot
    ax_3d = fig.add_subplot(gs[0:2, 0:2], projection='3d')

    # Separate by lick responsiveness
    lick_pos = df_valid[df_valid['lick_responsive']]
    lick_neg = df_valid[~df_valid['lick_responsive']]

    ax_3d.scatter(lick_neg['task_projection'], lick_neg['sensory_projection'], lick_neg['lick_projection'],
                 c='lightgray', s=15, alpha=0.3, label=f'Non-lick (n={len(lick_neg)})')
    ax_3d.scatter(lick_pos['task_projection'], lick_pos['sensory_projection'], lick_pos['lick_projection'],
                 c='red', s=25, alpha=0.8, edgecolors='white', linewidths=0.2,
                 label=f'Lick-responsive (n={len(lick_pos)})')

    ax_3d.set_xlabel('Task-state')
    ax_3d.set_ylabel('Sensory')
    ax_3d.set_zlabel('Lick (Method 3C)')
    ax_3d.set_title('A. 3D Coding Space', fontweight='bold')
    ax_3d.legend(frameon=False, fontsize=9)

    # Panel B: Pairwise orthogonality
    ax_ortho = fig.add_subplot(gs[0, 2])

    session_stats = df_valid.groupby('session_name').agg({
        'cos_task_sensory': 'first',
        'cos_task_lick': 'first',
        'cos_sensory_lick': 'first',
        'stage': 'first'
    }).reset_index()

    x_pos = np.arange(3)
    cos_means = [
        np.mean(session_stats['cos_task_sensory']),
        np.mean(session_stats['cos_task_lick']),
        np.mean(session_stats['cos_sensory_lick'])
    ]
    cos_stds = [
        np.std(session_stats['cos_task_sensory']),
        np.std(session_stats['cos_task_lick']),
        np.std(session_stats['cos_sensory_lick'])
    ]

    bars = ax_ortho.bar(x_pos, cos_means, yerr=cos_stds, capsize=5,
                       color=['blue', 'green', 'orange'], alpha=0.7)
    ax_ortho.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax_ortho.set_xticks(x_pos)
    ax_ortho.set_xticklabels(['Task×Sensory', 'Task×Lick', 'Sensory×Lick'], rotation=45)
    ax_ortho.set_ylabel('Cosine similarity')
    ax_ortho.set_title('B. Pairwise Orthogonality', fontweight='bold')

    # Panel C: Lick subtypes in 3D
    ax_subtypes = fig.add_subplot(gs[0, 3], projection='3d')

    fa_data = df_valid[df_valid['lick_responsive']]
    early_fa = fa_data[fa_data['early_fa_responsive']]
    late_fa = fa_data[fa_data['late_fa_responsive']]
    hit_only = fa_data[fa_data['hit_responsive'] & ~fa_data['early_fa_responsive'] & ~fa_data['late_fa_responsive']]

    ax_subtypes.scatter(early_fa['task_projection'], early_fa['sensory_projection'], early_fa['lick_projection'],
                       c='darkred', s=30, alpha=0.8, marker='s', label=f'Early FA (n={len(early_fa)})')
    ax_subtypes.scatter(late_fa['task_projection'], late_fa['sensory_projection'], late_fa['lick_projection'],
                       c='blue', s=30, alpha=0.8, marker='^', label=f'Late FA (n={len(late_fa)})')
    ax_subtypes.scatter(hit_only['task_projection'], hit_only['sensory_projection'], hit_only['lick_projection'],
                       c='green', s=25, alpha=0.7, marker='o', label=f'Hit only (n={len(hit_only)})')

    ax_subtypes.set_xlabel('Task-state')
    ax_subtypes.set_ylabel('Sensory')
    ax_subtypes.set_zlabel('Lick')
    ax_subtypes.set_title('C. Lick Subtypes', fontweight='bold')
    ax_subtypes.legend(frameon=False, fontsize=8)

    # Panel D: Stage comparison
    ax_stage = fig.add_subplot(gs[0, 4])

    stage_lick_fractions = []
    stage_names = []
    for stage in ['Learning', 'Expert']:
        stage_data = df_valid[df_valid['stage'] == stage]
        if len(stage_data) > 0:
            session_fractions = stage_data.groupby('session_name')['lick_responsive'].mean()
            stage_lick_fractions.append(session_fractions.values)
            stage_names.append(stage)

    if stage_lick_fractions:
        ax_stage.boxplot(stage_lick_fractions, labels=stage_names,
                        patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax_stage.set_ylabel('Fraction lick-responsive')
        ax_stage.set_title('D. Stage Comparison', fontweight='bold')

    # Panel E-G: 2D projections of 3D space
    projection_pairs = [
        ('task_projection', 'sensory_projection', 'E. Task × Sensory'),
        ('task_projection', 'lick_projection', 'F. Task × Lick'),
        ('sensory_projection', 'lick_projection', 'G. Sensory × Lick')
    ]

    for i, (x_col, y_col, title) in enumerate(projection_pairs):
        ax = fig.add_subplot(gs[1, i])

        ax.scatter(lick_neg[x_col], lick_neg[y_col],
                  c='lightgray', s=15, alpha=0.3)
        ax.scatter(lick_pos[x_col], lick_pos[y_col],
                  c='red', s=25, alpha=0.7, edgecolors='white', linewidths=0.2)

        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel(x_col.replace('_', ' ').title())
        ax.set_ylabel(y_col.replace('_', ' ').title())
        ax.set_title(title, fontweight='bold')

    # Panel H: Method 3C validation
    ax_method = fig.add_subplot(gs[1, 3:])

    lick_stats = df_valid.groupby('session_name').agg({
        'n_licks': 'first',
        'lick_responsive': 'sum',
        'n_common_units': 'first',
        'stage': 'first'
    }).reset_index()

    colors = [STAGE_COLORS.get(stage, 'gray') for stage in lick_stats['stage']]
    scatter = ax_method.scatter(lick_stats['n_licks'], lick_stats['lick_responsive'] / lick_stats['n_common_units'],
                              c=colors, s=60, alpha=0.7, edgecolors='white', linewidths=0.5)
    ax_method.set_xlabel('Number of licks used')
    ax_method.set_ylabel('Fraction lick-responsive')
    ax_method.set_title('H. Method 3C Validation', fontweight='bold')

    # Add stage legend
    for stage, color in STAGE_COLORS.items():
        if stage in lick_stats['stage'].values:
            ax_method.scatter([], [], c=color, s=60, alpha=0.7, label=stage)
    ax_method.legend(frameon=False, fontsize=9)

    # Panel I: Summary statistics table
    ax_table = fig.add_subplot(gs[2:, :])
    ax_table.axis('off')

    # Calculate comprehensive summary
    n_sessions = len(session_stats)
    n_total = len(df_valid)
    n_lick = np.sum(df_valid['lick_responsive']) if n_total > 0 else 0

    ortho_means = {
        'task_sensory': np.mean(session_stats['cos_task_sensory']),
        'task_lick': np.mean(session_stats['cos_task_lick']),
        'sensory_lick': np.mean(session_stats['cos_sensory_lick'])
    }

    ortho_stds = {
        'task_sensory': np.std(session_stats['cos_task_sensory']),
        'task_lick': np.std(session_stats['cos_task_lick']),
        'sensory_lick': np.std(session_stats['cos_sensory_lick'])
    }

    # Count significant orthogonality per pair
    p_cols = ['p_task_sensory', 'p_task_lick', 'p_sensory_lick']
    sig_ortho = {col: np.sum(session_stats.get(col, [1.0]) < 0.05) for col in p_cols}

    mean_licks = np.mean(df_valid.groupby('session_name')['n_licks'].first())

    table_text = f"""
    3D CODING SPACE ANALYSIS — Method {LICK_METHOD} (Pre-lick vs Post-lick)

    Sessions: {n_sessions} sessions, {n_total} total units
    Lick responsiveness: {n_lick}/{n_total} ({100*n_lick/n_total:.1f}%) units

    Coding Dimensions:
    1. Task-state: Hit vs Miss (fast TF pulses, pre-pulse baseline activity)
    2. Sensory: Fast TF pulse responsiveness (post-pulse peaks)
    3. Lick: Pre-lick vs Post-lick activity (Method 3C, {mean_licks:.1f} licks/session)

    Pairwise Orthogonality (cosine similarity, mean ± std):
    Task × Sensory:  {ortho_means['task_sensory']:6.3f} ± {ortho_stds['task_sensory']:.3f}  ({sig_ortho.get('p_task_sensory', 0)}/{n_sessions} sessions p<0.05)
    Task × Lick:     {ortho_means['task_lick']:6.3f} ± {ortho_stds['task_lick']:.3f}  ({sig_ortho.get('p_task_lick', 0)}/{n_sessions} sessions p<0.05)
    Sensory × Lick:  {ortho_means['sensory_lick']:6.3f} ± {ortho_stds['sensory_lick']:.3f}  ({sig_ortho.get('p_sensory_lick', 0)}/{n_sessions} sessions p<0.05)

    Lick Alignment Windows:
    Pre-lick: {LICK_PRE_WINDOW[0]:.1f} to {LICK_PRE_WINDOW[1]:.1f}s (ramping/preparation)
    Post-lick: {LICK_POST_WINDOW[0]:.1f} to {LICK_POST_WINDOW[1]:.1f}s (execution/aftermath)

    Future Methods Available:
    3A: Early-FA vs Late-FA coding direction
    3B: Hit vs Catch-CR coding direction
    3D: Lick-aligned ramping activity
    """

    ax_table.text(0.05, 0.5, table_text, transform=ax_table.transAxes,
                 fontsize=10, verticalalignment='center', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.3))

    plt.suptitle(f'3D Coding Space: Task-state × Sensory × Lick (Method {LICK_METHOD})',
                 fontsize=18, fontweight='bold', y=0.95)

    return fig


# ── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Computing 3D coding space analysis (Method {LICK_METHOD})...")
    print(f"Dimensions: Task-state (Hit vs Miss), Sensory (TF), Lick (Pre vs Post)")
    print(f"Lick windows: Pre {LICK_PRE_WINDOW}, Post {LICK_POST_WINDOW}")

    force_recompute = '--force' in sys.argv

    results = compute_or_load(force=force_recompute)

    n_sessions = results['session_name'].nunique() if len(results) > 0 else 0
    print(f"Loaded {len(results)} unit records from {n_sessions} sessions")

    if len(results) > 0:
        fig = plot_3d_coding_space(results)
        save_figure(fig, "fig26_3d_coding_space", "06_lick_motor")

        # Save detailed stats
        stats_path = os.path.join(CACHE_DIR.replace('cache', 'figures/06_lick_motor'),
                                 "3d_coding_space_stats.csv")
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)

        # Session-level summary
        session_stats = results.groupby('session_name').agg({
            'cos_task_sensory': 'first',
            'cos_task_lick': 'first',
            'cos_sensory_lick': 'first',
            'p_task_sensory': 'first',
            'p_task_lick': 'first',
            'p_sensory_lick': 'first',
            'stage': 'first',
            'lick_responsive': ['sum', 'count'],
            'n_fast_pulses': 'first',
            'n_licks': 'first',
            'n_common_units': 'first',
            'lick_method': 'first'
        }).reset_index()

        session_stats.columns = ['session_name', 'cos_task_sensory', 'cos_task_lick', 'cos_sensory_lick',
                                'p_task_sensory', 'p_task_lick', 'p_sensory_lick', 'stage',
                                'n_lick', 'n_total', 'n_fast_pulses', 'n_licks', 'n_common_units', 'lick_method']
        session_stats['lick_fraction'] = session_stats['n_lick'] / session_stats['n_total']

        # Mark significant orthogonality
        session_stats['sig_task_sensory'] = session_stats['p_task_sensory'] < 0.05
        session_stats['sig_task_lick'] = session_stats['p_task_lick'] < 0.05
        session_stats['sig_sensory_lick'] = session_stats['p_sensory_lick'] < 0.05

        session_stats.to_csv(stats_path, index=False)
        print(f"Saved statistics to {stats_path}")

        # Print summary
        n_sig_ts = np.sum(session_stats['sig_task_sensory'])
        n_sig_tl = np.sum(session_stats['sig_task_lick'])
        n_sig_sl = np.sum(session_stats['sig_sensory_lick'])
        print(f"Significant orthogonality: T-S={n_sig_ts}/{len(session_stats)}, T-L={n_sig_tl}/{len(session_stats)}, S-L={n_sig_sl}/{len(session_stats)}")
    else:
        print("No valid data found for 3D analysis")