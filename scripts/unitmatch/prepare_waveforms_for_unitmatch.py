#!/usr/bin/env python3
"""Prepare waveforms for UnitMatch from Kilosort and/or Bombcell.

This script:
1. Loads session configuration from config/unitmatch_sessions.yml
2. Extracts waveforms using configurable sources (kilosort/bombcell/both)
3. Optionally filters spikes to ITI periods only
4. Saves waveforms in UnitMatch-compatible format
5. Generates comparison reports if both sources used

Usage:
    python scripts/prepare_waveforms_for_unitmatch.py --config config/unitmatch_sessions.yml
    python scripts/prepare_waveforms_for_unitmatch.py --config config/unitmatch_sessions.yml --source both --compare
"""

import argparse
import yaml
import logging
from pathlib import Path
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unit_tracking import extract_waveforms_from_kilosort, extract_iti_spikes
from session_io import load_session

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load UnitMatch configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compare_waveform_sources(
    waveforms_ks: np.ndarray,
    waveforms_bc: np.ndarray,
    session_name: str,
    output_dir: Path
):
    """Generate comparison plots between Kilosort and Bombcell waveforms.
    
    Args:
        waveforms_ks: (n_units, spike_w, n_ch, 2) from Kilosort
        waveforms_bc: (n_units, spike_w, n_ch, 2) from Bombcell
        session_name: Name for plot titles
        output_dir: Directory to save comparison plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check shapes match
    if waveforms_ks.shape[0] != waveforms_bc.shape[0]:
        logger.warning(f"Different number of units: KS={waveforms_ks.shape[0]}, BC={waveforms_bc.shape[0]}")
        n_units = min(waveforms_ks.shape[0], waveforms_bc.shape[0])
        waveforms_ks = waveforms_ks[:n_units]
        waveforms_bc = waveforms_bc[:n_units]
    
    n_units = waveforms_ks.shape[0]
    
    # Compute correlations between corresponding units
    correlations = []
    for i in range(n_units):
        # Use first half of each for comparison
        wf_ks = waveforms_ks[i, :, :, 0].flatten()
        wf_bc = waveforms_bc[i, :, :, 0].flatten()
        corr = np.corrcoef(wf_ks, wf_bc)[0, 1]
        correlations.append(corr)
    
    correlations = np.array(correlations)
    
    # Plot correlation distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(correlations, bins=30, edgecolor='black')
    axes[0].axvline(np.nanmean(correlations), color='red', linestyle='--', 
                    label=f'Mean: {np.nanmean(correlations):.3f}')
    axes[0].set_xlabel('Waveform Correlation')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'{session_name}\nKilosort vs Bombcell Waveform Correlation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot example waveforms
    example_unit = np.argmax(correlations)  # Best matching unit
    wf_ks_ex = waveforms_ks[example_unit, :, :, 0]  # (spike_w, n_ch)
    wf_bc_ex = waveforms_bc[example_unit, :, :, 0]
    
    # Show peak channel
    peak_ch_ks = np.argmax(np.abs(wf_ks_ex).max(axis=0))
    
    axes[1].plot(wf_ks_ex[:, peak_ch_ks], label='Kilosort', linewidth=2)
    axes[1].plot(wf_bc_ex[:, peak_ch_ks], label='Bombcell', linewidth=2, alpha=0.7)
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Amplitude (μV)')
    axes[1].set_title(f'Example Unit {example_unit} (Peak Channel)\nCorr={correlations[example_unit]:.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / f"{session_name}_waveform_comparison.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparison plot: {plot_file}")
    
    # Generate summary statistics
    summary = {
        'n_units': n_units,
        'mean_correlation': float(np.nanmean(correlations)),
        'median_correlation': float(np.nanmedian(correlations)),
        'min_correlation': float(np.nanmin(correlations)),
        'max_correlation': float(np.nanmax(correlations)),
        'std_correlation': float(np.nanstd(correlations))
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Prepare waveforms for UnitMatch')
    parser.add_argument('--config', type=str, default='config/unitmatch_sessions.yml',
                        help='Path to UnitMatch config file')
    parser.add_argument('--source', type=str, default=None,
                        choices=['kilosort', 'bombcell', 'both'],
                        help='Waveform source (overrides config)')
    parser.add_argument('--use-iti', action='store_true',
                        help='Extract waveforms from ITI periods only')
    parser.add_argument('--iti-method', type=str, default='trial_boundaries',
                        choices=['trial_field', 'trial_boundaries', 'fallback'],
                        help='Method for ITI extraction')
    parser.add_argument('--compare', action='store_true',
                        help='Generate comparison plots if both sources used')
    parser.add_argument('--sessions', type=str, nargs='+', default=None,
                        help='Process specific sessions (default: all in config)')
    parser.add_argument('--output-dir', type=str, default='png_output/unitmatch_waveforms',
                        help='Output directory for waveforms and plots')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1
    
    config = load_config(config_path)
    
    # Get waveform config
    wf_config = config.get('waveform_config', {})
    source = args.source or wf_config.get('source', 'kilosort')
    use_iti = args.use_iti or wf_config.get('use_iti', False)
    iti_method = args.iti_method or wf_config.get('iti_method', 'trial_boundaries')
    fallback_window = wf_config.get('fallback_window', [1.0, 3.0])
    compare = args.compare or (wf_config.get('compare_sources', False) and source == 'both')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Waveform extraction configuration:")
    logger.info(f"  Source: {source}")
    logger.info(f"  Use ITI only: {use_iti}")
    logger.info(f"  ITI method: {iti_method}")
    logger.info(f"  Fallback window: {fallback_window}")
    logger.info(f"  Compare sources: {compare}")
    
    # Get sessions to process
    sessions_list = config.get('sessions', [])
    if args.sessions:
        # Filter to requested sessions
        sessions_list = [s for s in sessions_list if s.get('name') in args.sessions]
    
    if not sessions_list:
        logger.error("No sessions to process")
        return 1
    
    logger.info(f"Processing {len(sessions_list)} sessions")
    
    # Process each session
    comparison_summaries = []
    
    for sess_config in sessions_list:
        session_name = sess_config.get('name', 'unknown')
        session_path = sess_config.get('path')
        ks_dir = sess_config.get('kilosort_dir')
        bc_dir = sess_config.get('bombcell_dir')
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing session: {session_name}")
        logger.info(f"{'='*60}")
        
        if not session_path or not Path(session_path).exists():
            logger.error(f"Session file not found: {session_path}")
            continue
        
        if not ks_dir or not Path(ks_dir).exists():
            logger.error(f"Kilosort directory not found: {ks_dir}")
            if source in ['kilosort', 'both']:
                continue
        
        if source in ['bombcell', 'both']:
            if not bc_dir or not Path(bc_dir).exists():
                logger.error(f"Bombcell directory not found: {bc_dir}")
                continue
        
        # Load session
        try:
            logger.info(f"Loading session: {session_path}")
            session = load_session(session_path)
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            continue
        
        # Extract waveforms
        try:
            waveforms_dict = extract_waveforms_from_kilosort(
                session=session,
                ks_dir=Path(ks_dir),
                bc_dir=Path(bc_dir) if bc_dir else None,
                source=source,
                use_iti_only=use_iti,
                iti_method=iti_method,
                fallback_window=tuple(fallback_window)
            )
        except Exception as e:
            logger.error(f"Failed to extract waveforms: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Save waveforms
        for src_name, waveforms in waveforms_dict.items():
            suffix = 'iti' if use_iti else 'full'
            output_file = output_dir / f"{session_name}_waveforms_{src_name}_{suffix}.npy"
            np.save(output_file, waveforms)
            logger.info(f"Saved {src_name} waveforms: {output_file} {waveforms.shape}")
        
        # Generate comparison if requested
        if compare and 'kilosort' in waveforms_dict and 'bombcell' in waveforms_dict:
            try:
                summary = compare_waveform_sources(
                    waveforms_dict['kilosort'],
                    waveforms_dict['bombcell'],
                    session_name,
                    output_dir / 'comparisons'
                )
                summary['session'] = session_name
                comparison_summaries.append(summary)
                logger.info(f"Comparison summary: mean_corr={summary['mean_correlation']:.3f}")
            except Exception as e:
                logger.error(f"Failed to generate comparison: {e}")
                import traceback
                traceback.print_exc()
    
    # Save comparison summaries
    if comparison_summaries:
        summary_file = output_dir / 'comparison_summary.yaml'
        with open(summary_file, 'w') as f:
            yaml.dump(comparison_summaries, f)
        logger.info(f"\nSaved comparison summary: {summary_file}")
        
        # Print overall statistics
        mean_corrs = [s['mean_correlation'] for s in comparison_summaries]
        logger.info(f"\nOverall comparison statistics:")
        logger.info(f"  Sessions compared: {len(comparison_summaries)}")
        logger.info(f"  Mean correlation: {np.mean(mean_corrs):.3f} ± {np.std(mean_corrs):.3f}")
        logger.info(f"  Range: [{np.min(mean_corrs):.3f}, {np.max(mean_corrs):.3f}]")
    
    logger.info(f"\n{'='*60}")
    logger.info("Waveform preparation complete!")
    logger.info(f"{'='*60}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
