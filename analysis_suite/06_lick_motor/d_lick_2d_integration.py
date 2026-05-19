"""Fig26: Lick neurons in 2D coding space — Task-state × Sensory integration (CORRECTED).

Maps lick-responsive neurons onto the corrected 2D decomposition framework.
Uses FAST TF pulse-aligned task-state coding direction (Lohse et al. method) and
examines where lick neurons sit in the orthogonal coding geometry.

CORRECTED METHOD (April 2026):
- Task-state CD computed using FAST TF pulse alignment during baseline periods
- Uses constraints (min_before_change=1.0s) and go-trials only
- Includes permutation testing for orthogonality significance
- Option to incorporate lick timing/magnitude (future enhancement)

Saves: figures/06_lick_motor/fig26_lick_2d_integration_corrected.png
Stats: figures/06_lick_motor/lick_2d_integration_corrected_stats.csv
"""
import os
import sys
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy.stats as stats
from pathlib import Path

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
from visdetect.analysis.align import get_event_times_by_trial

setup_style()

# Parameters (matching corrected f_2d_decomposition.py)
MIN_UNITS = 10
MIN_TRIALS_PER_CLASS = 8
N_PERM_ORTHO = 1000
CD_REG = 0.01  # Shrinkage LDA regularization: 1% of trace(Cov)

# Options for future enhancement
INCORPORATE_LICK_TIMING = False  # Future: use lick latency as weight
INCORPORATE_LICK_MAGNITUDE = False  # Future: use response strength as weight

# ── Cache management ────────────────────────────────────────────
CACHE_FILE = os.path.join(CACHE_DIR, "lick_2d_integration_corrected.csv")


def _compute_tf_pulse_task_state_cd(session, good_ids, tf_data):
    """Compute task-state coding direction using FAST TF pulse alignment (Lohse method).

    This exactly matches the corrected f_2d_decomposition.py implementation:
    - Uses ONLY fast pulses (not all pulses)
    - Enables constraints (min_before_change=1.0s)
    - Go-trials only (change_size > 1.01)
    - Pre-pulse window (-0.4, 0.0s) for task-state activity
    """
    from visdetect.analysis.tf_pulse import _collect_pulses, TFRespPulseConfig

    trials = session.trials

    # Get TF pulse times during baseline periods (with constraints enabled)
    cfg = TFRespPulseConfig(use_constraints=True)  # Ensures min_before_change=1.0s
    fast_times, slow_times = _collect_pulses(session, cfg, show_progress=False)

    # Use ONLY fast pulses for task-state CD (Lohse method)
    pulse_times = fast_times

    if len(pulse_times) < 20:  # Need sufficient fast pulses
        return None, None, 0

    # Build tensor aligned to fast TF pulses
    pulse_window = (-0.5, 0.5)  # Around each pulse
    tensor, bc = build_population_tensor(
        session, good_ids, pulse_times, pulse_window, DEFAULT_BIN_SIZE
    )

    if tensor is None or tensor.shape[0] < 20:
        return None, None, 0

    # Get trial outcomes for each pulse time
    baseline_times = get_event_times_by_trial(session, "Baseline_ON")
    change_times = get_event_times_by_trial(session, "Change_ON")

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
    cd_task = compute_lda_cd(all_activity, labels, method="manual", reg=CD_REG, reg_style="trace_scaled")

    return cd_task, len(pulse_times), len(hit_activity) + len(miss_activity)


def _compute_sensory_cd(tf_data, good_ids):
    """Extract sensory coding direction from FAST TF pulse responses."""
    if tf_data is None or 'cluster_ids' not in tf_data:
        return None

    tf_ids = tf_data['cluster_ids']
    common_ids = sorted(set(good_ids) & set(tf_ids))

    if len(common_ids) < 10:
        return None

    # Map common IDs to TF data indices
    tf_id_to_idx = {cid: i for i, cid in enumerate(tf_ids)}
    common_indices = [tf_id_to_idx[cid] for cid in common_ids]

    # Extract fast TF pulse responses (post-pulse peak)
    t_vec = tf_data['t_vec']
    fast_z = tf_data['fast_z'][common_indices]

    # Find peak in post-pulse window (0.0, 0.2)s
    post_mask = (t_vec >= 0.0) & (t_vec < 0.2)
    if not np.any(post_mask):
        return None

    # Signed peak (can be positive or negative response)
    peak_responses = []
    for unit_trace in fast_z:
        post_trace = unit_trace[post_mask]
        pos_peak = np.max(post_trace)
        neg_peak = np.min(post_trace)
        # Take peak with larger absolute value
        signed_peak = pos_peak if abs(pos_peak) > abs(neg_peak) else neg_peak
        peak_responses.append(signed_peak)

    cd_sensory = np.array(peak_responses)
    if np.linalg.norm(cd_sensory) > 0:
        cd_sensory = cd_sensory / np.linalg.norm(cd_sensory)  # Normalize

    return cd_sensory


def _test_orthogonality(cd_task, cd_sensory, n_perm=N_PERM_ORTHO):
    """Test orthogonality with permutation null distribution."""
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


def _classify_early_late_fa(session):
    """Classify FA trials as early vs late using FA_RT_SPLIT."""
    trials = getattr(session, "trials", [])
    early_fa_trials = []
    late_fa_trials = []

    for i, trial in enumerate(trials):
        if getattr(trial, 'trialoutcome', None) == 'fa':
            rt_dict = getattr(trial, 'reactiontimes', {}) or {}
            fa_rt = rt_dict.get('fa', np.nan)

            if np.isfinite(fa_rt):
                if fa_rt <= FA_RT_SPLIT:
                    early_fa_trials.append(i)
                else:
                    late_fa_trials.append(i)

    return early_fa_trials, late_fa_trials


def _get_lick_weights(session_lick_data, unit_id):
    """Get lick timing/magnitude weights for future enhancement."""
    if not (INCORPORATE_LICK_TIMING or INCORPORATE_LICK_MAGNITUDE):
        return 1.0  # Default weight

    lick_data = session_lick_data.get(unit_id, {})

    weight = 1.0
    if INCORPORATE_LICK_TIMING and 'lick_latency' in lick_data:
        # Weight by inverse latency (faster = stronger weight)
        latency = lick_data.get('lick_latency', 1.0)
        weight *= 1.0 / max(latency, 0.1)

    if INCORPORATE_LICK_MAGNITUDE and 'lick_magnitude' in lick_data:
        # Weight by response magnitude
        magnitude = lick_data.get('lick_magnitude', 1.0)
        weight *= max(magnitude, 0.1)

    return weight


def compute_or_load(force=False):
    """Main computation function."""
    if os.path.exists(CACHE_FILE) and not force:
        return pd.read_csv(CACHE_FILE)

    manifest = load_staging_manifest(qc_only=True)
    # Load all lick responsiveness data once
    all_lick_data = load_all_lick_responsiveness()

    rows = []

    for _, mrow in manifest.iterrows():
        sname = str(mrow["session_name"])
        stage = str(mrow["stage"])

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
        except:
            print(f"  No TF data for {sname}, skipping...")
            del sess; gc.collect()
            continue

        # Find common units (both good QC and TF data)
        tf_ids = tf_data['cluster_ids'] if tf_data else []
        common_ids = sorted(set(good_ids) & set(tf_ids))

        if len(common_ids) < MIN_UNITS:
            print(f"  Insufficient common units ({len(common_ids)} < {MIN_UNITS})")
            del sess; gc.collect()
            continue

        # Compute 2D coding directions (CORRECTED METHOD)
        cd_task, n_fast_pulses, n_valid_pulses = _compute_tf_pulse_task_state_cd(sess, common_ids, tf_data)
        cd_sensory = _compute_sensory_cd(tf_data, common_ids)

        if cd_task is None or cd_sensory is None:
            print(f"  Failed to compute coding directions for {sname}")
            del sess; gc.collect()
            continue

        # Test orthogonality with permutation
        cos_sim, pval, null_dist = _test_orthogonality(cd_task, cd_sensory)

        # Get lick responsiveness for this session
        session_lick_data = all_lick_data.get(sname, {})

        # Classify FA trials
        early_fa_trials, late_fa_trials = _classify_early_late_fa(sess)

        print(f"  {len(common_ids)} units, {n_fast_pulses} fast pulses, cos_sim={cos_sim:.3f}, p={pval:.3f}")

        # For each unit, get its position in 2D space and lick responsiveness
        for i, unit_id in enumerate(common_ids):
            if i >= len(cd_task) or i >= len(cd_sensory):
                continue

            task_projection = cd_task[i]
            sensory_projection = cd_sensory[i]

            # Lick responsiveness
            lick_data = session_lick_data.get(unit_id, {})
            lick_responsive = lick_data.get('lick_responsive', False)
            early_fa_responsive = lick_data.get('early_fa_responsive', False)
            late_fa_responsive = lick_data.get('late_fa_responsive', False)
            hit_responsive = lick_data.get('hit_responsive', False)

            # Get weights for future enhancement
            lick_weight = _get_lick_weights(session_lick_data, unit_id)

            rows.append({
                'session_name': sname,
                'stage': stage,
                'cluster_id': unit_id,
                'task_projection': task_projection,
                'sensory_projection': sensory_projection,
                'cos_sim_2d': cos_sim,
                'orthogonality_pval': pval,
                'n_fast_pulses': n_fast_pulses,
                'n_valid_pulses': n_valid_pulses,
                'n_common_units': len(common_ids),
                'lick_responsive': lick_responsive,
                'early_fa_responsive': early_fa_responsive,
                'late_fa_responsive': late_fa_responsive,
                'hit_responsive': hit_responsive,
                'lick_weight': lick_weight,  # Future enhancement
                'n_early_fa_trials': len(early_fa_trials),
                'n_late_fa_trials': len(late_fa_trials)
            })

        print(f"  Added {len(common_ids)} units to results")
        del sess; gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(CACHE_FILE, index=False)
    return df


def plot_lick_2d_integration(df):
    """Create the main figure showing lick neurons in 2D coding space."""

    # Filter to sessions with valid 2D decomposition
    df_valid = df.dropna(subset=['task_projection', 'sensory_projection'])

    if len(df_valid) == 0:
        print("No valid sessions for 2D analysis")
        return

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(4, 4, hspace=0.4, wspace=0.3)

    # Panel A: Orthogonality validation across sessions
    ax_ortho = fig.add_subplot(gs[0, 0])
    session_stats = df_valid.groupby('session_name').agg({
        'cos_sim_2d': 'first',
        'orthogonality_pval': 'first',
        'stage': 'first'
    }).reset_index()

    colors = [STAGE_COLORS.get(stage, 'gray') for stage in session_stats['stage']]
    ax_ortho.scatter(range(len(session_stats)), session_stats['cos_sim_2d'].values,
                    c=colors, s=60, alpha=0.7, edgecolors='white', linewidths=0.5)
    ax_ortho.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax_ortho.set_xlabel('Session index')
    ax_ortho.set_ylabel('Cosine similarity\n(Task × Sensory)')
    ax_ortho.set_title('A. 2D Orthogonality (CORRECTED)', fontweight='bold')

    # Test overall orthogonality
    if len(session_stats) > 5:
        _, p_ortho = stats.wilcoxon(session_stats['cos_sim_2d'].values)
        n_sig = np.sum(session_stats['orthogonality_pval'] < 0.05)
        ax_ortho.text(0.05, 0.95, f'Wilcoxon p={p_ortho:.3f}\n{n_sig}/{len(session_stats)} sessions p<0.05',
                     transform=ax_ortho.transAxes, fontsize=9, verticalalignment='top')

    # Panel B: 2D scatter - all neurons
    ax_2d = fig.add_subplot(gs[0, 1:3])

    # Separate by lick responsiveness
    lick_pos = df_valid[df_valid['lick_responsive']]
    lick_neg = df_valid[~df_valid['lick_responsive']]

    ax_2d.scatter(lick_neg['task_projection'], lick_neg['sensory_projection'],
                 c='lightgray', s=20, alpha=0.3, label=f'Non-lick (n={len(lick_neg)})')
    ax_2d.scatter(lick_pos['task_projection'], lick_pos['sensory_projection'],
                 c='red', s=30, alpha=0.7, edgecolors='white', linewidths=0.3,
                 label=f'Lick-responsive (n={len(lick_pos)})')

    ax_2d.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax_2d.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax_2d.set_xlabel('Task-state projection')
    ax_2d.set_ylabel('Sensory projection')
    ax_2d.set_title('B. Lick neurons in 2D coding space', fontweight='bold')
    ax_2d.legend(frameon=False, fontsize=9)

    # Panel C: FA subtype analysis
    ax_fa = fig.add_subplot(gs[0, 3])

    fa_data = df_valid[df_valid['lick_responsive']]
    early_fa = fa_data[fa_data['early_fa_responsive']]
    late_fa = fa_data[fa_data['late_fa_responsive']]
    other_lick = fa_data[~fa_data['early_fa_responsive'] & ~fa_data['late_fa_responsive']]

    ax_fa.scatter(other_lick['task_projection'], other_lick['sensory_projection'],
                 c='orange', s=25, alpha=0.5, label=f'Other lick (n={len(other_lick)})')
    ax_fa.scatter(early_fa['task_projection'], early_fa['sensory_projection'],
                 c='darkred', s=30, alpha=0.8, marker='s',
                 label=f'Early FA (n={len(early_fa)})')
    ax_fa.scatter(late_fa['task_projection'], late_fa['sensory_projection'],
                 c='blue', s=30, alpha=0.8, marker='^',
                 label=f'Late FA (n={len(late_fa)})')

    ax_fa.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax_fa.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax_fa.set_xlabel('Task-state projection')
    ax_fa.set_ylabel('Sensory projection')
    ax_fa.set_title('C. FA subtypes', fontweight='bold')
    ax_fa.legend(frameon=False, fontsize=7)

    # Panel D: Task-state distribution comparison
    ax_task = fig.add_subplot(gs[1, 0])

    task_lick = df_valid[df_valid['lick_responsive']]['task_projection']
    task_nonlick = df_valid[~df_valid['lick_responsive']]['task_projection']

    if len(task_lick) > 0 and len(task_nonlick) > 0:
        ax_task.hist(task_nonlick, bins=30, alpha=0.5, color='lightgray',
                    label='Non-lick', density=True)
        ax_task.hist(task_lick, bins=30, alpha=0.7, color='red',
                    label='Lick-responsive', density=True)
        ax_task.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax_task.set_xlabel('Task-state projection')
        ax_task.set_ylabel('Density')
        ax_task.set_title('D. Task-state distribution', fontweight='bold')
        ax_task.legend(frameon=False)

        # Statistical test
        stat, p_task = stats.mannwhitneyu(task_lick, task_nonlick, alternative='two-sided')
        ax_task.text(0.05, 0.95, f'MW U: p = {p_task:.3e}',
                    transform=ax_task.transAxes, fontsize=9, verticalalignment='top')

    # Panel E: Sensory distribution comparison
    ax_sens = fig.add_subplot(gs[1, 1])

    sens_lick = df_valid[df_valid['lick_responsive']]['sensory_projection']
    sens_nonlick = df_valid[~df_valid['lick_responsive']]['sensory_projection']

    if len(sens_lick) > 0 and len(sens_nonlick) > 0:
        ax_sens.hist(sens_nonlick, bins=30, alpha=0.5, color='lightgray',
                    label='Non-lick', density=True)
        ax_sens.hist(sens_lick, bins=30, alpha=0.7, color='red',
                    label='Lick-responsive', density=True)
        ax_sens.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax_sens.set_xlabel('Sensory projection')
        ax_sens.set_ylabel('Density')
        ax_sens.set_title('E. Sensory distribution', fontweight='bold')
        ax_sens.legend(frameon=False)

        # Statistical test
        stat, p_sens = stats.mannwhitneyu(sens_lick, sens_nonlick, alternative='two-sided')
        ax_sens.text(0.05, 0.95, f'MW U: p = {p_sens:.3e}',
                    transform=ax_sens.transAxes, fontsize=9, verticalalignment='top')

    # Panel F: Stage progression (lick responsiveness)
    ax_stage = fig.add_subplot(gs[1, 2:])

    stage_means = []
    stage_sems = []
    stage_names = []
    for stage in ['Learning', 'Expert']:
        stage_data = df_valid[df_valid['stage'] == stage]
        if len(stage_data) > 0:
            # Session-level lick fractions
            session_fractions = stage_data.groupby('session_name')['lick_responsive'].mean()
            if len(session_fractions) > 0:
                stage_means.append(np.mean(session_fractions))
                stage_sems.append(np.std(session_fractions) / np.sqrt(len(session_fractions)))
                stage_names.append(stage)

    if stage_means:
        x_pos = np.arange(len(stage_names))
        bars = ax_stage.bar(x_pos, stage_means, yerr=stage_sems,
                           color=[STAGE_COLORS[s] for s in stage_names],
                           alpha=0.7, edgecolor='white', linewidth=1,
                           capsize=5)
        ax_stage.set_xticks(x_pos)
        ax_stage.set_xticklabels(stage_names)
        ax_stage.set_ylabel('Fraction lick-responsive')
        ax_stage.set_title('F. Lick responsiveness by stage', fontweight='bold')
        ax_stage.set_ylim(0, max(stage_means) * 1.2)

        # Add values on bars
        for bar, val, sem in zip(bars, stage_means, stage_sems):
            ax_stage.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sem + 0.01,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Panel G: Quadrant analysis
    ax_quad = fig.add_subplot(gs[2, 0:2])

    # Define quadrants based on task-state and sensory projections
    lick_data = df_valid[df_valid['lick_responsive']]

    if len(lick_data) > 0:
        q1 = lick_data[(lick_data['task_projection'] > 0) & (lick_data['sensory_projection'] > 0)]  # High task, high sensory
        q2 = lick_data[(lick_data['task_projection'] < 0) & (lick_data['sensory_projection'] > 0)]  # Low task, high sensory
        q3 = lick_data[(lick_data['task_projection'] < 0) & (lick_data['sensory_projection'] < 0)]  # Low task, low sensory
        q4 = lick_data[(lick_data['task_projection'] > 0) & (lick_data['sensory_projection'] < 0)]  # High task, low sensory

        quadrant_counts = [len(q1), len(q2), len(q3), len(q4)]
        quadrant_labels = ['Q1: High task\nHigh sensory', 'Q2: Low task\nHigh sensory',
                          'Q3: Low task\nLow sensory', 'Q4: High task\nLow sensory']

        bars = ax_quad.bar(range(4), quadrant_counts,
                          color=['red', 'orange', 'blue', 'purple'], alpha=0.7)
        ax_quad.set_xticks(range(4))
        ax_quad.set_xticklabels(quadrant_labels, fontsize=9)
        ax_quad.set_ylabel('Number of lick neurons')
        ax_quad.set_title('G. Quadrant distribution of lick neurons', fontweight='bold')

        # Add counts on bars
        for bar, count in zip(bars, quadrant_counts):
            ax_quad.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontsize=10)

    # Panel H: Pulse count validation
    ax_pulses = fig.add_subplot(gs[2, 2:])

    pulse_stats = df_valid.groupby('session_name').agg({
        'n_fast_pulses': 'first',
        'n_valid_pulses': 'first',
        'stage': 'first'
    }).reset_index()

    colors = [STAGE_COLORS.get(stage, 'gray') for stage in pulse_stats['stage']]
    ax_pulses.scatter(pulse_stats['n_fast_pulses'], pulse_stats['n_valid_pulses'],
                     c=colors, s=60, alpha=0.7, edgecolors='white', linewidths=0.5)
    ax_pulses.set_xlabel('Total fast pulses')
    ax_pulses.set_ylabel('Valid pulses (Hit/Miss)')
    ax_pulses.set_title('H. Pulse count validation', fontweight='bold')

    # Add diagonal line
    max_pulses = max(pulse_stats['n_fast_pulses'].max(), pulse_stats['n_valid_pulses'].max())
    ax_pulses.plot([0, max_pulses], [0, max_pulses], 'k--', alpha=0.5, label='Unity')
    ax_pulses.legend(frameon=False)

    # Panel I: Summary statistics table
    ax_table = fig.add_subplot(gs[3, :])
    ax_table.axis('off')

    # Calculate comprehensive summary stats
    n_sessions = len(session_stats)
    n_total = len(df_valid)
    n_lick = np.sum(df_valid['lick_responsive']) if n_total > 0 else 0
    n_early_fa = np.sum(df_valid['early_fa_responsive']) if n_total > 0 else 0
    n_late_fa = np.sum(df_valid['late_fa_responsive']) if n_total > 0 else 0
    n_hit = np.sum(df_valid['hit_responsive']) if n_total > 0 else 0

    mean_cos_sim = np.mean(session_stats['cos_sim_2d']) if len(session_stats) > 0 else np.nan
    std_cos_sim = np.std(session_stats['cos_sim_2d']) if len(session_stats) > 0 else np.nan
    mean_pulses = np.mean(pulse_stats['n_fast_pulses']) if len(pulse_stats) > 0 else np.nan

    mean_task_lick = np.mean(task_lick) if len(task_lick) > 0 else np.nan
    mean_task_nonlick = np.mean(task_nonlick) if len(task_nonlick) > 0 else np.nan
    mean_sens_lick = np.mean(sens_lick) if len(sens_lick) > 0 else np.nan
    mean_sens_nonlick = np.mean(sens_nonlick) if len(sens_nonlick) > 0 else np.nan

    n_ortho_sig = np.sum(session_stats['orthogonality_pval'] < 0.05) if len(session_stats) > 0 else 0

    table_text = f"""
    CORRECTED 2D DECOMPOSITION SUMMARY (Fast TF pulses + constraints)

    Sessions: {n_sessions} sessions, {n_total} units total
    Method:   Task-state CD from fast TF pulses (Hit vs Miss, pre-pulse activity)
             Sensory CD from fast TF pulse responses (post-pulse peaks)
             Constraints: min_before_change=1.0s, go-trials only

    Orthogonality: cos(Task, Sensory) = {mean_cos_sim:.3f} ± {std_cos_sim:.3f}
                  {n_ortho_sig}/{n_sessions} sessions significantly orthogonal (p<0.05)
                  Fast pulses per session: {mean_pulses:.1f} ± {np.std(pulse_stats['n_fast_pulses']) if len(pulse_stats) > 0 else np.nan:.1f}

    Lick Responsiveness: {n_lick}/{n_total} ({100*n_lick/n_total:.1f}%) units lick-responsive
                        Early FA: {n_early_fa}, Late FA: {n_late_fa}, Hit: {n_hit}

    2D Projections: Task-state → Lick: {mean_task_lick:.3f}, Non-lick: {mean_task_nonlick:.3f}
                   Sensory → Lick: {mean_sens_lick:.3f}, Non-lick: {mean_sens_nonlick:.3f}

    Shrinkage LDA: reg = {CD_REG} (1% trace regularization for numerical stability)
    """

    ax_table.text(0.05, 0.5, table_text, transform=ax_table.transAxes,
                 fontsize=10, verticalalignment='center', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('Lick-responsive neurons in 2D task-state × sensory coding space (CORRECTED METHOD)',
                 fontsize=16, fontweight='bold', y=0.95)

    return fig


# ── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Computing lick neuron 2D integration analysis (CORRECTED METHOD)...")
    print(f"Method: Fast TF pulses only, constraints enabled, {N_PERM_ORTHO} permutations")
    print(f"Shrinkage LDA regularization: {CD_REG}")
    print(f"Enhancement options: Timing={INCORPORATE_LICK_TIMING}, Magnitude={INCORPORATE_LICK_MAGNITUDE}")

    # Force recomputation if needed
    force_recompute = '--force' in sys.argv

    results = compute_or_load(force=force_recompute)

    n_sessions = results['session_name'].nunique() if len(results) > 0 else 0
    print(f"Loaded {len(results)} unit records from {n_sessions} sessions")

    if len(results) > 0:
        fig = plot_lick_2d_integration(results)
        save_figure(fig, "fig26_lick_2d_integration_corrected", "06_lick_motor")

        # Save detailed stats
        stats_path = os.path.join(CACHE_DIR.replace('cache', 'figures/06_lick_motor'),
                                 "lick_2d_integration_corrected_stats.csv")
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)

        # Session-level stats
        session_stats = results.groupby('session_name').agg({
            'cos_sim_2d': 'first',
            'orthogonality_pval': 'first',
            'stage': 'first',
            'lick_responsive': ['sum', 'count'],
            'n_fast_pulses': 'first',
            'n_valid_pulses': 'first',
            'n_common_units': 'first'
        }).reset_index()
        session_stats.columns = ['session_name', 'cos_sim_2d', 'orthogonality_pval', 'stage',
                                'n_lick', 'n_total', 'n_fast_pulses', 'n_valid_pulses', 'n_common_units']
        session_stats['lick_fraction'] = session_stats['n_lick'] / session_stats['n_total']
        session_stats['orthogonal_sig'] = session_stats['orthogonality_pval'] < 0.05

        session_stats.to_csv(stats_path, index=False)
        print(f"Saved statistics to {stats_path}")
        print(f"Orthogonal sessions: {np.sum(session_stats['orthogonal_sig'])}/{len(session_stats)}")
    else:
        print("No valid data found for analysis")