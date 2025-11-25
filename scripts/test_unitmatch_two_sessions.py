"""
Test UnitMatch tracking between two consecutive sessions.

This script:
1. Loads two sessions
2. Extracts waveforms for good clusters
3. Runs UnitMatch matching algorithm
4. Visualizes match probabilities and generates diagnostic plots
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging
import argparse

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import modules
from visdetect.core.legacy_io import load_session
from visdetect.analysis.tracking import extract_waveforms_from_kilosort, extract_iti_spikes

# Setup logging
# Base configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Default to stdout
)
logger = logging.getLogger(__name__)


def setup_file_logging(log_file):
    """Set up a file handler for logging."""
    # Remove all existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create new file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add file handler and a stream handler
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)
    logger.info(f"Logging to file: {log_file}")


def extract_session_waveforms(session_path, ks_dir, use_iti=True):
    """Extract waveforms for a single session."""
    logger.info(f"Loading session: {session_path.name}")
    session = load_session(session_path)
    
    session_name = session_path.stem
    logger.info(f"Session: {session_name}")
    logger.info(f"  Clusters: {len(session.clusters)}")
    logger.info(f"  Trials: {len(session.trials)}")
    
    # Filter to good clusters
    logger.info("Filtering to 'good' quality clusters only")
    if not hasattr(session, 'good_cluster_ids') or session.good_cluster_ids is None:
        raise ValueError("Session does not have good_cluster_ids attribute")
    
    good_ids = list(session.good_cluster_ids)
    logger.info(f"Session has {len(good_ids)} good clusters")
    
    # Create Cluster objects with quality='good' for good clusters
    # This is needed for extract_waveforms_from_kilosort to recognize them
    from dataclasses import dataclass
    @dataclass
    class Cluster:
        cluster_id: int
        spike_times: np.ndarray
        quality: str = 'good'
    
    good_clusters = []
    for c in session.clusters:
        cluster_id = getattr(c, 'cluster_id', -1)
        if cluster_id in good_ids:
            spike_times = getattr(c, 'spike_times', np.array([]))
            good_clusters.append(Cluster(
                cluster_id=cluster_id,
                spike_times=spike_times,
                quality='good'
            ))
    
    logger.info(f"Created {len(good_clusters)} good cluster objects")
    
    # Create filtered session copy with only good clusters
    session_filtered = type('Session', (), {})()
    session_filtered.clusters = good_clusters
    session_filtered.trials = session.trials
    session_filtered.ni_events = session.ni_events  # CRITICAL: Needed for ITI extraction!
    session_filtered.good_cluster_ids = good_ids
    
    # Extract waveforms (with optional ITI filtering)
    logger.info(f"Extracting waveforms from Kilosort: {ks_dir}")
    if use_iti:
        logger.info("Using ITI filtering to avoid stimulus artifacts")
    else:
        logger.info("Using all spikes (no ITI filtering)")
    waveforms_dict = extract_waveforms_from_kilosort(
        session_filtered,
        ks_dir=Path(ks_dir),
        source='kilosort',
        use_iti_only=use_iti,
        iti_method='trial_field',
        fallback_window=(1.0, 3.0)
    )
    
    waveforms = waveforms_dict['kilosort']
    logger.info(f"Extracted waveforms: {waveforms.shape}")
    logger.info(f"  Shape: (n_units={waveforms.shape[0]}, spike_w={waveforms.shape[1]}, "
                f"n_channels={waveforms.shape[2]}, n_splits={waveforms.shape[3]})")
    
    return session, waveforms, good_ids


def run_unitmatch_pair(wf1, wf2, good_ids1, good_ids2):
    """
    Run UnitMatch between two sessions.
    
    For now, we'll compute a simple waveform correlation matrix as a proof of concept.
    The full UnitMatch algorithm will be integrated later.
    """
    logger.info("Computing waveform similarity matrix...")
    
    n1, spike_w, n_ch, _ = wf1.shape
    n2 = wf2.shape[0]
    
    # Average across splits (cross-validation folds)
    wf1_mean = wf1.mean(axis=3)  # (n1, spike_w, n_ch)
    wf2_mean = wf2.mean(axis=3)  # (n2, spike_w, n_ch)
    
    # Flatten waveforms for correlation
    wf1_flat = wf1_mean.reshape(n1, -1)  # (n1, spike_w * n_ch)
    wf2_flat = wf2_mean.reshape(n2, -1)  # (n2, spike_w * n_ch)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(wf1_flat, wf2_flat)[:n1, n1:]  # (n1, n2)
    
    logger.info(f"Correlation matrix shape: {corr_matrix.shape}")
    logger.info(f"Correlation range: [{corr_matrix.min():.3f}, {corr_matrix.max():.3f}]")
    
    # Find best matches
    best_matches = []
    for i in range(n1):
        j = np.argmax(corr_matrix[i, :])
        corr = corr_matrix[i, j]
        best_matches.append({
            'session1_unit': good_ids1[i],
            'session2_unit': good_ids2[j],
            'correlation': corr,
            'session1_idx': i,
            'session2_idx': j
        })
    
    # Sort by correlation (highest first)
    best_matches = sorted(best_matches, key=lambda x: x['correlation'], reverse=True)
    
    return corr_matrix, best_matches


def plot_results(corr_matrix, best_matches, s1_name, s2_name, output_dir, use_iti, wf1, wf2, good_ids1, good_ids2):
    """Generate diagnostic plots."""
    logger.info("Generating diagnostic plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: Correlation matrix
    im = axes[0].imshow(corr_matrix, aspect='auto', cmap='seismic', vmin=-1, vmax=1)
    axes[0].set_xlabel(f'{s2_name} cluster index')
    axes[0].set_ylabel(f'{s1_name} cluster index')
    axes[0].set_title('Waveform Correlation Matrix')
    plt.colorbar(im, ax=axes[0])
    
    # Mark best matches
    for match in best_matches[:20]:  # Show top 20
        i, j = match['session1_idx'], match['session2_idx']
        axes[0].plot(j, i, 'wo', markersize=3, alpha=0.7)
    
    # Panel 2: Distribution of correlations
    all_corrs = corr_matrix.flatten()
    best_corrs = [m['correlation'] for m in best_matches]
    
    axes[1].hist(all_corrs, bins=50, alpha=0.6, label='All pairs', color='gray', edgecolor='black')
    axes[1].hist(best_corrs, bins=30, alpha=0.7, label='Best matches', color='red', edgecolor='black')
    axes[1].axvline(np.mean(best_corrs), color='red', linestyle='--', 
                    label=f'Best match mean: {np.mean(best_corrs):.2f}')
    axes[1].set_xlabel('Waveform correlation')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Correlation Distributions')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Example best match waveforms
    best = best_matches[0]
    i, j = best['session1_idx'], best['session2_idx']
    corr = best['correlation']
    
    # Get waveforms (average across splits)
    wf_i = wf1[i].mean(axis=1)  # (spike_w, n_ch) -> average over splits
    wf_j = wf2[j].mean(axis=1)
    
    # Find peak channel
    peak_ch = np.argmax(np.abs(wf_i).max(axis=0))
    
    # Plot waveforms
    axes[2].plot(wf_i[:, peak_ch], color='blue', alpha=0.6, 
                 label=f'{s1_name} (unit {good_ids1[i]})', linewidth=2.5)
    axes[2].plot(wf_j[:, peak_ch], color='red', alpha=0.6, 
                 label=f'{s2_name} (unit {good_ids2[j]})', linewidth=2.5)
    axes[2].set_xlabel('Sample')
    axes[2].set_ylabel('Amplitude (μV)')
    axes[2].set_title(f'Best Match Example (r={corr:.3f}) - Purple = Perfect Overlay')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_file = output_dir / f'{s1_name}_vs_{s2_name}_matches.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot: {plot_file}")
    
    plt.close(fig)


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Run UnitMatch between two sessions.")
    parser.add_argument('--session1', type=str, required=True, help='Path to session 1 .pkl file')
    parser.add_argument('--session2', type=str, required=True, help='Path to session 2 .pkl file')
    parser.add_argument('--ks-dir1', type=str, required=True, help='Path to Kilosort directory for session 1')
    parser.add_argument('--ks-dir2', type=str, required=True, help='Path to Kilosort directory for session 2')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save results and plots')
    parser.add_argument('--use-iti', action='store_true', help='Use ITI filtering for waveform extraction')
    parser.add_argument('--log-file', type=str, default=None, help='Optional file to write logs to')
    
    args = parser.parse_args()

    # Setup logging to file if specified
    if args.log_file:
        setup_file_logging(args.log_file)

    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    s1_path = project_root / args.session1
    s2_path = project_root / args.session2
    
    # --- Session 1 ---
    s1, wf1, gids1 = extract_session_waveforms(s1_path, args.ks_dir1, args.use_iti)
    
    # --- Session 2 ---
    s2, wf2, gids2 = extract_session_waveforms(s2_path, args.ks_dir2, args.use_iti)
    
    # --- Run Matching ---
    corr_matrix, best_matches = run_unitmatch_pair(wf1, wf2, gids1, gids2)
    
    # --- Save and Plot Results ---
    s1_name = s1_path.stem
    s2_name = s2_path.stem
    
    plot_results(corr_matrix, best_matches, s1_name, s2_name, output_path, args.use_iti, wf1, wf2, gids1, gids2)
    
    # Save results to YAML
    results_data = {
        'session1_name': s1_name,
        'session2_name': s2_name,
        'n_units_session1': len(gids1),
        'n_units_session2': len(gids2),
        'use_iti': args.use_iti,
        'correlations': [m['correlation'] for m in best_matches],
        'mean_match_correlation': np.mean([m['correlation'] for m in best_matches]),
        'std_match_correlation': np.std([m['correlation'] for m in best_matches]),
        'max_correlation': best_matches[0]['correlation'] if best_matches else 0,
        'min_correlation': best_matches[-1]['correlation'] if best_matches else 0,
        'high_confidence_matches': sum(1 for m in best_matches if m['correlation'] > 0.8),
        'medium_confidence_matches': sum(1 for m in best_matches if 0.6 <= m['correlation'] <= 0.8),
        'low_confidence_matches': sum(1 for m in best_matches if m['correlation'] < 0.6),
    }
    
    results_file = output_path / f"{s1_name}_vs_{s2_name}_results.yaml"
    with open(results_file, 'w') as f:
        yaml.dump(results_data, f, default_flow_style=False)
        
    logger.info(f"Results saved to {results_file}")
    logger.info(f"Best match correlations: {results_data['mean_match_correlation']:.3f}  {results_data['std_match_correlation']:.3f}")
    logger.info(f"  High (r>0.8): {results_data['high_confidence_matches']} ({100*results_data['high_confidence_matches']/len(gids1):.1f}%)")
    logger.info("Test completed successfully.")


if __name__ == "__main__":
    main()
