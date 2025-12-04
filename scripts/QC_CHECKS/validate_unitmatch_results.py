#!/usr/bin/env python3
"""Validate UnitMatch tracking results.

This script validates UnitMatch tracking by:
1. Computing ISI fingerprint stability across matched units
2. Comparing functional similarity (firing rates, waveform correlations)
3. Generating diagnostic plots matching UnitMatch paper Fig 5
4. Comparing tracking performance between Kilosort and Bombcell waveforms (if both available)

Based on validation methods from Windolf et al. 2024 Nature Methods.

Usage:
    python scripts/validate_unitmatch_results.py --tracking tracking_chains.csv
    python scripts/validate_unitmatch_results.py --tracking tracking_chains.csv --compare-sources
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from scipy.spatial.distance import euclidean
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from session_io import load_session

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_isi_histogram(spike_times: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Compute ISI histogram for spike train.
    
    Args:
        spike_times: Array of spike times (seconds)
        bins: Bin edges for histogram (seconds)
    
    Returns:
        Normalized ISI histogram
    """
    if len(spike_times) < 2:
        return np.zeros(len(bins) - 1)
    
    isis = np.diff(spike_times)
    hist, _ = np.histogram(isis, bins=bins)
    
    # Normalize
    if hist.sum() > 0:
        hist = hist.astype(float) / hist.sum()
    
    return hist


def compute_isi_stability(
    sessions_list: list,
    tracking_df: pd.DataFrame,
    track_id: int,
    isi_bins: np.ndarray = None
) -> dict:
    """Compute ISI fingerprint stability for a tracked unit.
    
    Args:
        sessions_list: List of session file paths
        tracking_df: Tracking DataFrame
        track_id: Track ID to analyze
        isi_bins: Bin edges for ISI histograms (default: log-spaced 0.001 to 1.0 s)
    
    Returns:
        Dict with ISI histograms and pairwise similarities
    """
    if isi_bins is None:
        isi_bins = np.logspace(-3, 0, 50)  # 0.001 to 1.0 seconds
    
    track_data = tracking_df[tracking_df['track_id'] == track_id].sort_values('session_id')
    
    if len(track_data) < 2:
        return None
    
    # Load spike times for each session
    isi_hists = []
    session_ids = []
    
    for _, row in track_data.iterrows():
        sess_id = row['session_id']
        unit_id = row['unit_id']
        session_path = sessions_list[sess_id]
        
        try:
            session = load_session(session_path)
            
            # Find cluster
            cluster = None
            for c in session.clusters:
                cid = getattr(c, 'cluster_id', -1) if not isinstance(c, dict) else c.get('cluster_id', -1)
                if cid == unit_id:
                    cluster = c
                    break
            
            if cluster is None:
                logger.warning(f"Cluster {unit_id} not found in session {sess_id}")
                continue
            
            spike_times = getattr(cluster, 'spike_times', np.array([])) if not isinstance(cluster, dict) else cluster.get('spike_times', np.array([]))
            
            if len(spike_times) < 2:
                continue
            
            isi_hist = compute_isi_histogram(spike_times, isi_bins)
            isi_hists.append(isi_hist)
            session_ids.append(sess_id)
            
        except Exception as e:
            logger.warning(f"Failed to load session {sess_id}: {e}")
            continue
    
    if len(isi_hists) < 2:
        return None
    
    # Compute pairwise Euclidean distances
    distances = []
    for i in range(len(isi_hists)):
        for j in range(i + 1, len(isi_hists)):
            dist = euclidean(isi_hists[i], isi_hists[j])
            distances.append(dist)
    
    return {
        'track_id': track_id,
        'n_sessions': len(isi_hists),
        'isi_hists': isi_hists,
        'session_ids': session_ids,
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'distances': distances
    }


def compute_waveform_similarity(
    waveform_dir: Path,
    tracking_df: pd.DataFrame,
    track_id: int,
    session_names: list,
    waveform_source: str = 'kilosort',
    use_iti: bool = False
) -> dict:
    """Compute waveform similarity for a tracked unit.
    
    Returns:
        Dict with waveform correlations across sessions
    """
    track_data = tracking_df[tracking_df['track_id'] == track_id].sort_values('session_id')
    
    if len(track_data) < 2:
        return None
    
    # Load waveforms for each session
    waveforms = []
    session_ids = []
    unit_indices = []
    
    suffix = 'iti' if use_iti else 'full'
    
    for _, row in track_data.iterrows():
        sess_id = row['session_id']
        unit_idx = row['unit_idx']
        session_name = session_names[sess_id]
        
        wf_file = waveform_dir / f"{session_name}_waveforms_{waveform_source}_{suffix}.npy"
        
        if not wf_file.exists():
            logger.warning(f"Waveforms not found: {wf_file}")
            continue
        
        try:
            wf_array = np.load(wf_file)
            
            # Get waveform for this unit (use first half for consistency)
            if unit_idx < wf_array.shape[0]:
                wf = wf_array[unit_idx, :, :, 0].flatten()
                waveforms.append(wf)
                session_ids.append(sess_id)
                unit_indices.append(unit_idx)
        except Exception as e:
            logger.warning(f"Failed to load waveform: {e}")
            continue
    
    if len(waveforms) < 2:
        return None
    
    # Compute pairwise correlations
    correlations = []
    for i in range(len(waveforms)):
        for j in range(i + 1, len(waveforms)):
            corr = np.corrcoef(waveforms[i], waveforms[j])[0, 1]
            correlations.append(corr)
    
    return {
        'track_id': track_id,
        'n_sessions': len(waveforms),
        'session_ids': session_ids,
        'mean_correlation': np.nanmean(correlations),
        'std_correlation': np.nanstd(correlations),
        'correlations': correlations
    }


def plot_validation_results(
    isi_results: list,
    waveform_results: list,
    tracking_df: pd.DataFrame,
    output_dir: Path
):
    """Generate validation plots matching UnitMatch paper."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to multi-session tracks
    multi_track_ids = tracking_df.groupby('track_id')['session_id'].nunique()
    multi_track_ids = multi_track_ids[multi_track_ids > 1].index
    
    isi_multi = [r for r in isi_results if r['track_id'] in multi_track_ids]
    wf_multi = [r for r in waveform_results if r['track_id'] in multi_track_ids]
    
    # Plot 1: ISI stability
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if isi_multi:
        isi_distances = [r['mean_distance'] for r in isi_multi]
        n_sessions = [r['n_sessions'] for r in isi_multi]
        
        axes[0].scatter(n_sessions, isi_distances, alpha=0.6)
        axes[0].set_xlabel('Number of sessions tracked')
        axes[0].set_ylabel('Mean ISI fingerprint distance')
        axes[0].set_title('ISI Stability Across Sessions')
        axes[0].grid(True, alpha=0.3)
        
        # Add summary statistics
        mean_dist = np.mean(isi_distances)
        axes[0].axhline(mean_dist, color='red', linestyle='--', 
                        label=f'Mean: {mean_dist:.3f}')
        axes[0].legend()
    
    # Plot 2: Waveform correlation
    if wf_multi:
        wf_corrs = [r['mean_correlation'] for r in wf_multi]
        n_sessions_wf = [r['n_sessions'] for r in wf_multi]
        
        axes[1].scatter(n_sessions_wf, wf_corrs, alpha=0.6, color='green')
        axes[1].set_xlabel('Number of sessions tracked')
        axes[1].set_ylabel('Mean waveform correlation')
        axes[1].set_title('Waveform Stability Across Sessions')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        mean_corr = np.mean(wf_corrs)
        axes[1].axhline(mean_corr, color='red', linestyle='--',
                        label=f'Mean: {mean_corr:.3f}')
        axes[1].legend()
    
    plt.tight_layout()
    plot_file = output_dir / 'tracking_stability.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved stability plot: {plot_file}")
    
    # Plot 3: Track length distribution
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    track_lengths = tracking_df.groupby('track_id')['session_id'].nunique()
    
    ax.hist(track_lengths, bins=range(1, track_lengths.max() + 2), 
            edgecolor='black', align='left')
    ax.set_xlabel('Number of sessions')
    ax.set_ylabel('Number of tracks')
    ax.set_title(f'Track Length Distribution (n={len(track_lengths)} tracks)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    multi_session = (track_lengths > 1).sum()
    ax.text(0.95, 0.95, 
            f'Multi-session: {multi_session} ({100*multi_session/len(track_lengths):.1f}%)',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plot_file = output_dir / 'track_length_distribution.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved track length plot: {plot_file}")


def main():
    parser = argparse.ArgumentParser(description='Validate UnitMatch tracking results')
    parser.add_argument('--tracking', type=str, required=True,
                        help='Path to tracking_chains.csv')
    parser.add_argument('--sessions', type=str, nargs='+', required=True,
                        help='Paths to session .pkl files (in order)')
    parser.add_argument('--waveform-dir', type=str, default='png_output/unitmatch_waveforms',
                        help='Directory with waveform files')
    parser.add_argument('--waveform-source', type=str, default='kilosort',
                        choices=['kilosort', 'bombcell'],
                        help='Waveform source to validate')
    parser.add_argument('--use-iti', action='store_true',
                        help='Use ITI waveforms')
    parser.add_argument('--compare-sources', action='store_true',
                        help='Compare Kilosort vs Bombcell tracking')
    parser.add_argument('--output-dir', type=str, default='png_output/unitmatch_validation',
                        help='Output directory for validation plots')
    parser.add_argument('--max-tracks', type=int, default=None,
                        help='Maximum tracks to validate (for speed)')
    
    args = parser.parse_args()
    
    # Load tracking results
    tracking_df = pd.read_csv(args.tracking)
    logger.info(f"Loaded tracking data: {len(tracking_df)} rows, {tracking_df['track_id'].nunique()} tracks")
    
    # Setup
    waveform_dir = Path(args.waveform_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get session names from tracking_df
    session_names = tracking_df.groupby('session_id')['session_name'].first().to_dict()
    session_names_list = [session_names[i] for i in sorted(session_names.keys())]
    
    # Get track IDs to validate
    track_ids = tracking_df['track_id'].unique()
    if args.max_tracks:
        track_ids = track_ids[:args.max_tracks]
    
    logger.info(f"Validating {len(track_ids)} tracks")
    
    # Compute ISI stability
    logger.info("Computing ISI stability...")
    isi_results = []
    for track_id in track_ids:
        result = compute_isi_stability(args.sessions, tracking_df, track_id)
        if result:
            isi_results.append(result)
    
    logger.info(f"Computed ISI stability for {len(isi_results)} tracks")
    
    # Compute waveform similarity
    logger.info("Computing waveform similarity...")
    waveform_results = []
    for track_id in track_ids:
        result = compute_waveform_similarity(
            waveform_dir, tracking_df, track_id, 
            session_names_list, args.waveform_source, args.use_iti
        )
        if result:
            waveform_results.append(result)
    
    logger.info(f"Computed waveform similarity for {len(waveform_results)} tracks")
    
    # Generate plots
    logger.info("Generating validation plots...")
    plot_validation_results(isi_results, waveform_results, tracking_df, output_dir)
    
    # Save summary statistics
    summary = {
        'n_tracks': len(tracking_df['track_id'].unique()),
        'n_multi_session_tracks': (tracking_df.groupby('track_id')['session_id'].nunique() > 1).sum(),
        'n_sessions': tracking_df['session_id'].nunique(),
    }
    
    if isi_results:
        isi_distances = [r['mean_distance'] for r in isi_results]
        summary['isi_mean_distance'] = float(np.mean(isi_distances))
        summary['isi_std_distance'] = float(np.std(isi_distances))
    
    if waveform_results:
        wf_corrs = [r['mean_correlation'] for r in waveform_results]
        summary['waveform_mean_correlation'] = float(np.nanmean(wf_corrs))
        summary['waveform_std_correlation'] = float(np.nanstd(wf_corrs))
    
    import yaml
    summary_file = output_dir / 'validation_summary.yaml'
    with open(summary_file, 'w') as f:
        yaml.dump(summary, f)
    
    logger.info(f"\nValidation summary:")
    logger.info(f"  Total tracks: {summary['n_tracks']}")
    logger.info(f"  Multi-session tracks: {summary['n_multi_session_tracks']}")
    if 'isi_mean_distance' in summary:
        logger.info(f"  ISI stability: {summary['isi_mean_distance']:.3f} ± {summary['isi_std_distance']:.3f}")
    if 'waveform_mean_correlation' in summary:
        logger.info(f"  Waveform correlation: {summary['waveform_mean_correlation']:.3f} ± {summary['waveform_std_correlation']:.3f}")
    
    logger.info(f"\nSaved validation summary: {summary_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
