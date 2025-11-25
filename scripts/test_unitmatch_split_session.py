#!/usr/bin/env python3
"""Test UnitMatch pipeline by splitting a single session in half.

This provides ground truth validation: all units should match between halves.

Usage:
    python scripts/test_unitmatch_split_session.py --session data/BG_046_02072025.pkl
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pickle

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from visdetect.core.legacy_io import load_session
from visdetect.analysis.tracking import extract_waveforms_from_kilosort, extract_iti_spikes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def split_session(session, split_time=None):
    """Split session into two halves at specified time (or midpoint).
    
    Args:
        session: Session to split
        split_time: Time to split at (default: midpoint)
    
    Returns:
        session_first, session_second, split_time: Two Session objects with split data and split time
    """
    from session_io import Session, Cluster, Trial
    
    # Determine split time
    if split_time is None:
        # Find midpoint based on ni_events
        ni_events = getattr(session, 'ni_events', {}) or {}
        if 'Baseline_ON' in ni_events:
            baseline_on = ni_events['Baseline_ON']
            if isinstance(baseline_on, dict) and 'rise_t' in baseline_on:
                times = np.array(baseline_on['rise_t']).flatten()
            else:
                times = np.array(baseline_on).flatten()
            times = times[~np.isnan(times)]
            if len(times) > 0:
                split_time = (times[0] + times[-1]) / 2
        
        # Fallback: use spike times
        if split_time is None:
            all_spikes = []
            for cluster in session.clusters:
                st = getattr(cluster, 'spike_times', np.array([])) if not isinstance(cluster, dict) else cluster.get('spike_times', np.array([]))
                all_spikes.extend(st)
            if all_spikes:
                split_time = np.median(all_spikes)
    
    logger.info(f"Splitting session at time {split_time:.2f}s")
    
    # Split clusters (filter to good quality only)
    clusters_first = []
    clusters_second = []
    
    logger.info("Filtering to 'good' quality clusters only")
    
    # Get good cluster IDs from session
    good_ids = set(session.good_cluster_ids) if hasattr(session, 'good_cluster_ids') else set()
    logger.info(f"Session has {len(good_ids)} good clusters")
    
    for cluster in session.clusters:
        cluster_id = getattr(cluster, 'cluster_id', -1) if not isinstance(cluster, dict) else cluster.get('cluster_id', -1)
        spike_times = getattr(cluster, 'spike_times', np.array([])) if not isinstance(cluster, dict) else cluster.get('spike_times', np.array([]))
        
        # Filter to only good clusters
        if cluster_id not in good_ids:
            continue
        
        # First half
        spikes_first = spike_times[spike_times < split_time]
        if len(spikes_first) >= 10:  # Minimum spike count
            clusters_first.append(Cluster(
                cluster_id=cluster_id,
                spike_times=spikes_first,
                quality='good'
            ))
        
        # Second half
        spikes_second = spike_times[spike_times >= split_time]
        if len(spikes_second) >= 10:
            clusters_second.append(Cluster(
                cluster_id=cluster_id,
                spike_times=spikes_second,
                quality='good'
            ))
    
    # Split trials (based on trial start times)
    trials_first = []
    trials_second = []
    
    for trial_idx, trial in enumerate(session.trials):
        # Try to get trial start time from ni_events
        if 'Baseline_ON' in ni_events:
            baseline_on = ni_events['Baseline_ON']
            if isinstance(baseline_on, dict) and 'rise_t' in baseline_on:
                times = np.array(baseline_on['rise_t']).flatten()
            else:
                times = np.array(baseline_on).flatten()
            
            if trial_idx < len(times) and not np.isnan(times[trial_idx]):
                trial_time = times[trial_idx]
                if trial_time < split_time:
                    trials_first.append(trial)
                else:
                    trials_second.append(trial)
    
    # Log split summary
    total_clusters = len(session.clusters)
    logger.info(f"First half: {len(clusters_first)} clusters, {len(trials_first)} trials")
    logger.info(f"Second half: {len(clusters_second)} clusters, {len(trials_second)} trials")
    logger.info(f"Quality filtering: {total_clusters} total → {len(clusters_first)} good quality ({100*len(clusters_first)/total_clusters:.1f}%)")
    
    # Create split sessions
    session_first = Session(
        trials=trials_first,
        clusters=clusters_first,
        subject=session.subject,
        session_name=f"{session.session_name}_first_half",
        good_cluster_ids=[c.cluster_id for c in clusters_first],
        ni_events=ni_events  # Share events (could filter, but not critical)
    )
    
    session_second = Session(
        trials=trials_second,
        clusters=clusters_second,
        subject=session.subject,
        session_name=f"{session.session_name}_second_half",
        good_cluster_ids=[c.cluster_id for c in clusters_second],
        ni_events=ni_events
    )
    
    # Remove duplicate logging that was moved earlier
    # logger.info lines already added above after quality filtering
    
    return session_first, session_second, split_time


def compute_waveform_similarity_matrix(wf_first, wf_second):
    """Compute all pairwise waveform correlations.
    
    Returns:
        corr_matrix: (n_first, n_second) correlation matrix
    """
    n_first = wf_first.shape[0]
    n_second = wf_second.shape[0]
    
    # Flatten waveforms (use first CV split)
    wf_first_flat = wf_first[:, :, :, 0].reshape(n_first, -1)
    wf_second_flat = wf_second[:, :, :, 0].reshape(n_second, -1)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(wf_first_flat, wf_second_flat)[:n_first, n_first:]
    
    return corr_matrix


def main():
    parser = argparse.ArgumentParser(description='Test UnitMatch by splitting a session')
    parser.add_argument('--session', type=str, required=True,
                        help='Path to session .pkl file')
    parser.add_argument('--ks-dir', type=str, default=None,
                        help='Path to Kilosort directory (if not in .pkl)')
    parser.add_argument('--use-iti', action='store_true',
                        help='Use ITI-filtered waveforms')
    parser.add_argument('--output-dir', type=str, default='png_output/unitmatch_test',
                        help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load session
    logger.info(f"Loading session: {args.session}")
    session = load_session(args.session)
    session_name = session.session_name or Path(args.session).stem
    
    logger.info(f"Session: {session_name}")
    logger.info(f"  Clusters: {len(session.clusters)}")
    logger.info(f"  Trials: {len(session.trials)}")
    
    # Split session
    session_first, session_second, split_time = split_session(session)
    
    # Get Kilosort directory
    if args.ks_dir:
        ks_dir = Path(args.ks_dir)
    else:
        # Try to infer from session name
        logger.error("Please provide --ks-dir with path to Kilosort output")
        return 1
    
    if not ks_dir.exists():
        logger.error(f"Kilosort directory not found: {ks_dir}")
        return 1
    
    # Extract waveforms for both halves
    logger.info("Extracting waveforms for first half...")
    waveforms_dict_first = extract_waveforms_from_kilosort(
        session_first,
        ks_dir=ks_dir,
        source='kilosort',
        use_iti_only=args.use_iti,
        iti_method='trial_boundaries',
        fallback_window=(1.0, 3.0)
    )
    wf_first = waveforms_dict_first['kilosort']
    
    logger.info("Extracting waveforms for second half...")
    waveforms_dict_second = extract_waveforms_from_kilosort(
        session_second,
        ks_dir=ks_dir,
        source='kilosort',
        use_iti_only=args.use_iti,
        iti_method='trial_boundaries',
        fallback_window=(1.0, 3.0)
    )
    wf_second = waveforms_dict_second['kilosort']
    
    logger.info(f"Waveforms extracted:")
    logger.info(f"  First half: {wf_first.shape}")
    logger.info(f"  Second half: {wf_second.shape}")
    
    # Compute waveform similarity matrix
    logger.info("Computing waveform similarity matrix...")
    corr_matrix = compute_waveform_similarity_matrix(wf_first, wf_second)
    
    # Find best matches (diagonal should be highest for ground truth)
    cluster_ids_first = [c.cluster_id for c in session_first.clusters]
    cluster_ids_second = [c.cluster_id for c in session_second.clusters]
    
    # Ground truth matches (same cluster ID)
    ground_truth_matches = []
    for i, cid_first in enumerate(cluster_ids_first):
        if cid_first in cluster_ids_second:
            j = cluster_ids_second.index(cid_first)
            ground_truth_matches.append((i, j, corr_matrix[i, j]))
    
    logger.info(f"Ground truth matches: {len(ground_truth_matches)}")
    
    # Compute metrics
    if ground_truth_matches:
        gt_corrs = [corr for _, _, corr in ground_truth_matches]
        mean_gt_corr = np.nanmean(gt_corrs)
        
        # Non-matches (off-diagonal)
        non_match_corrs = []
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                if (i, j, corr_matrix[i, j]) not in [(x, y, c) for x, y, c in ground_truth_matches]:
                    non_match_corrs.append(corr_matrix[i, j])
        
        mean_non_match_corr = np.nanmean(non_match_corrs)
        
        logger.info(f"\nResults:")
        logger.info(f"  Ground truth (same unit) correlation: {mean_gt_corr:.3f} ± {np.nanstd(gt_corrs):.3f}")
        logger.info(f"  Non-match correlation: {mean_non_match_corr:.3f} ± {np.nanstd(non_match_corrs):.3f}")
        logger.info(f"  Separation: {mean_gt_corr - mean_non_match_corr:.3f}")
        
        # Plot results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Correlation matrix heatmap
        im = axes[0].imshow(corr_matrix, aspect='auto', cmap='seismic', vmin=-1, vmax=1)
        axes[0].set_xlabel('Second half cluster index')
        axes[0].set_ylabel('First half cluster index')
        axes[0].set_title('Waveform Correlation Matrix')
        plt.colorbar(im, ax=axes[0])
        
        # Mark ground truth matches with small white dots (don't obscure the red diagonal)
        for i, j, _ in ground_truth_matches:
            axes[0].plot(j, i, 'wo', markersize=2, alpha=0.5)
        
        # 2. Distribution comparison
        axes[1].hist(gt_corrs, bins=20, alpha=0.7, label='Ground truth', edgecolor='black')
        axes[1].hist(non_match_corrs, bins=20, alpha=0.7, label='Non-matches', edgecolor='black')
        axes[1].axvline(mean_gt_corr, color='green', linestyle='--', label=f'GT mean: {mean_gt_corr:.2f}')
        axes[1].axvline(mean_non_match_corr, color='red', linestyle='--', label=f'Non-match mean: {mean_non_match_corr:.2f}')
        axes[1].set_xlabel('Waveform correlation')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Ground Truth vs Non-Matches')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Example waveform comparison
        if ground_truth_matches:
            best_match = max(ground_truth_matches, key=lambda x: x[2])
            i, j, corr = best_match
            
            # Get waveforms
            wf_i = wf_first[i, :, :, 0]  # (samples, channels)
            wf_j = wf_second[j, :, :, 0]
            
            # Find peak channel
            peak_ch = np.argmax(np.abs(wf_i).max(axis=0))
            
            # Use semi-transparent blue and red so perfect overlay shows as purple
            axes[2].plot(wf_i[:, peak_ch], color='blue', alpha=0.6, 
                        label=f'First half (cluster {cluster_ids_first[i]})', linewidth=2.5)
            axes[2].plot(wf_j[:, peak_ch], color='red', alpha=0.6,
                        label=f'Second half (cluster {cluster_ids_second[j]})', linewidth=2.5)
            axes[2].set_xlabel('Sample')
            axes[2].set_ylabel('Amplitude (μV)')
            axes[2].set_title(f'Best Match Example (r={corr:.3f}) - Purple = Perfect Overlay')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = output_dir / f'{session_name}_split_validation.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot: {plot_file}")
        plt.close()
        
        # Save results
        results = {
            'session_name': session_name,
            'split_time': float(split_time),
            'n_clusters_first': len(cluster_ids_first),
            'n_clusters_second': len(cluster_ids_second),
            'n_ground_truth_matches': len(ground_truth_matches),
            'mean_gt_correlation': float(mean_gt_corr),
            'std_gt_correlation': float(np.nanstd(gt_corrs)),
            'mean_non_match_correlation': float(mean_non_match_corr),
            'std_non_match_correlation': float(np.nanstd(non_match_corrs)),
            'separation': float(mean_gt_corr - mean_non_match_corr),
            'use_iti': args.use_iti
        }
        
        import yaml
        results_file = output_dir / f'{session_name}_split_results.yaml'
        with open(results_file, 'w') as f:
            yaml.dump(results, f)
        logger.info(f"Saved results: {results_file}")
        
        # Success criteria
        logger.info(f"\n{'='*60}")
        logger.info("VALIDATION RESULTS:")
        logger.info(f"{'='*60}")
        
        if mean_gt_corr > 0.8:
            logger.info("✅ PASS: Ground truth correlations > 0.8 (excellent)")
        elif mean_gt_corr > 0.6:
            logger.info("⚠️  WARN: Ground truth correlations 0.6-0.8 (acceptable)")
        else:
            logger.info("❌ FAIL: Ground truth correlations < 0.6 (poor)")
        
        if mean_gt_corr - mean_non_match_corr > 0.3:
            logger.info("✅ PASS: Good separation between matches and non-matches")
        else:
            logger.info("⚠️  WARN: Poor separation between matches and non-matches")
        
        logger.info(f"{'='*60}")
        
    else:
        logger.error("No ground truth matches found!")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
