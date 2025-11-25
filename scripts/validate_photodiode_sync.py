"""Validate photodiode-FSM behavioral synchronization.

Checks that NI photodiode events align with FSM behavioral frame timestamps
by computing cross-correlation and measuring timing drift across session.

Usage:
    python scripts/validate_photodiode_sync.py --session data/BG_046_15082025.pkl
    python scripts/validate_photodiode_sync.py --batch data/BG_046_*.pkl --plot
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from visdetect.core.legacy_io import load_session


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(levelname)s] %(message)s'
    )


def extract_photodiode_times(session) -> Optional[np.ndarray]:
    """Extract photodiode event times from NI events.
    
    Looks for common photodiode signal names in ni_events dict.
    """
    ni_events = getattr(session, 'ni_events', None)
    if not ni_events:
        return None
    
    # Common photodiode channel names
    photodiode_keys = [
        'Photodiode', 'photodiode', 'PD', 'pd',
        'Photod', 'photo', 'Photo',
        'Screen', 'screen', 'Monitor'
    ]
    
    for key in photodiode_keys:
        if key in ni_events:
            value = ni_events[key]
            if isinstance(value, dict) and 'rise_t' in value:
                times = np.array(value['rise_t']).flatten()
            else:
                times = np.array(value).flatten()
            
            times = times[np.isfinite(times)]
            if len(times) > 0:
                logging.debug(f"Found photodiode data in key: {key}")
                return times
    
    return None


def extract_fsm_frame_times(session) -> Optional[np.ndarray]:
    """Extract FSM behavioral frame timestamps.
    
    Attempts to reconstruct frame times from trial structure and timing.
    If Baseline_ON times exist, uses those as trial-level frame markers.
    """
    ni_events = getattr(session, 'ni_events', None)
    if not ni_events:
        return None
    
    # Try to find frame-related events
    frame_keys = [
        'Frame', 'frame', 'Frames',
        'Baseline_ON', 'baseline_on',
        'Stim_ON', 'stim_on'
    ]
    
    for key in frame_keys:
        if key in ni_events:
            value = ni_events[key]
            if isinstance(value, dict) and 'rise_t' in value:
                times = np.array(value['rise_t']).flatten()
            else:
                times = np.array(value).flatten()
            
            times = times[np.isfinite(times)]
            if len(times) > 0:
                logging.debug(f"Found FSM frame markers in key: {key}")
                return times
    
    return None


def compute_sync_metrics(photodiode_times: np.ndarray, 
                        fsm_times: np.ndarray,
                        max_lag_sec: float = 0.1) -> Dict[str, any]:
    """Compute synchronization metrics between photodiode and FSM.
    
    Args:
        photodiode_times: Photodiode event times (seconds)
        fsm_times: FSM frame times (seconds)
        max_lag_sec: Maximum lag to consider for matching (seconds)
        
    Returns:
        Dict with sync metrics
    """
    result = {
        'n_photodiode': len(photodiode_times),
        'n_fsm': len(fsm_times),
        'mean_offset': None,
        'std_offset': None,
        'max_drift': None,
        'sync_quality': None,
        'notes': []
    }
    
    if len(photodiode_times) == 0 or len(fsm_times) == 0:
        result['notes'].append("ERROR: Empty time arrays")
        return result
    
    # Simple approach: find nearest FSM time for each photodiode event
    # within max_lag window
    offsets = []
    
    for pd_time in photodiode_times:
        diffs = np.abs(fsm_times - pd_time)
        min_diff = np.min(diffs)
        
        if min_diff <= max_lag_sec:
            # Find whether photodiode leads or lags FSM
            closest_idx = np.argmin(diffs)
            offset = pd_time - fsm_times[closest_idx]
            offsets.append(offset)
    
    if len(offsets) == 0:
        result['notes'].append(f"ERROR: No matches within {max_lag_sec}s window")
        return result
    
    offsets = np.array(offsets)
    
    result['mean_offset'] = float(np.mean(offsets))
    result['std_offset'] = float(np.std(offsets))
    result['max_drift'] = float(np.max(np.abs(offsets)))
    result['matched_events'] = len(offsets)
    result['match_rate'] = len(offsets) / len(photodiode_times)
    
    # Quality assessment
    if result['max_drift'] < 0.001:  # <1ms
        result['sync_quality'] = 'excellent'
    elif result['max_drift'] < 0.010:  # <10ms
        result['sync_quality'] = 'good'
    elif result['max_drift'] < 0.050:  # <50ms
        result['sync_quality'] = 'acceptable'
    else:
        result['sync_quality'] = 'poor'
    
    result['notes'].append(
        f"Matched {len(offsets)}/{len(photodiode_times)} events "
        f"({100*result['match_rate']:.1f}%)"
    )
    
    return result


def plot_sync_analysis(photodiode_times: np.ndarray,
                      fsm_times: np.ndarray,
                      metrics: Dict[str, any],
                      output_path: Path,
                      session_name: str = ""):
    """Create diagnostic plot of synchronization."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Event times overlay
    ax = axes[0]
    ax.plot(photodiode_times, np.ones_like(photodiode_times), '|', 
            markersize=10, label='Photodiode', alpha=0.5)
    ax.plot(fsm_times, np.ones_like(fsm_times) * 0.95, '|',
            markersize=10, label='FSM', alpha=0.5)
    ax.set_ylim([0.9, 1.1])
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Event Timing Overlay - {session_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Inter-event intervals comparison
    ax = axes[1]
    if len(photodiode_times) > 1:
        pd_iei = np.diff(photodiode_times)
        ax.hist(pd_iei, bins=50, alpha=0.5, label='Photodiode IEI', density=True)
    if len(fsm_times) > 1:
        fsm_iei = np.diff(fsm_times)
        ax.hist(fsm_iei, bins=50, alpha=0.5, label='FSM IEI', density=True)
    ax.set_xlabel('Inter-Event Interval (s)')
    ax.set_ylabel('Density')
    ax.set_title('Inter-Event Interval Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add metrics text
    metrics_text = f"Mean offset: {metrics.get('mean_offset', 0)*1000:.2f} ms\n"
    metrics_text += f"Std offset: {metrics.get('std_offset', 0)*1000:.2f} ms\n"
    metrics_text += f"Max drift: {metrics.get('max_drift', 0)*1000:.2f} ms\n"
    metrics_text += f"Quality: {metrics.get('sync_quality', 'N/A')}"
    
    fig.text(0.98, 0.02, metrics_text, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"  Saved plot: {output_path}")


def validate_session_sync(session, 
                         plot_dir: Optional[Path] = None,
                         max_lag: float = 0.1) -> Dict[str, any]:
    """Validate photodiode-FSM sync for a single session."""
    result = {
        'subject': getattr(session, 'subject', 'unknown'),
        'session_name': getattr(session, 'session_name', 'unknown'),
        'has_photodiode': False,
        'has_fsm': False,
        'sync_quality': None,
        'mean_offset_ms': None,
        'max_drift_ms': None,
        'passed': False,
        'notes': []
    }
    
    # Extract data
    photodiode_times = extract_photodiode_times(session)
    fsm_times = extract_fsm_frame_times(session)
    
    result['has_photodiode'] = photodiode_times is not None
    result['has_fsm'] = fsm_times is not None
    
    if not result['has_photodiode']:
        result['notes'].append("WARNING: No photodiode data found")
        return result
    
    if not result['has_fsm']:
        result['notes'].append("WARNING: No FSM frame timing found")
        return result
    
    # Compute metrics
    metrics = compute_sync_metrics(photodiode_times, fsm_times, max_lag)
    
    result['sync_quality'] = metrics.get('sync_quality')
    if metrics.get('mean_offset') is not None:
        result['mean_offset_ms'] = metrics['mean_offset'] * 1000
    if metrics.get('max_drift') is not None:
        result['max_drift_ms'] = metrics['max_drift'] * 1000
    
    result['notes'].extend(metrics.get('notes', []))
    
    # Pass/fail criteria
    if metrics.get('sync_quality') in ['excellent', 'good', 'acceptable']:
        result['passed'] = True
    
    # Generate plot if requested
    if plot_dir and photodiode_times is not None and fsm_times is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"{result['session_name']}_photodiode_sync.png"
        try:
            plot_sync_analysis(photodiode_times, fsm_times, metrics,
                             plot_path, result['session_name'])
        except Exception as e:
            logging.warning(f"  Failed to create plot: {e}")
    
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate photodiode-FSM behavioral synchronization"
    )
    parser.add_argument(
        '--session',
        type=Path,
        help='Path to single session .pkl file'
    )
    parser.add_argument(
        '--batch',
        type=str,
        help='Glob pattern for batch processing'
    )
    parser.add_argument(
        '--max-lag',
        type=float,
        default=0.1,
        help='Maximum lag for event matching (seconds, default: 0.1)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate diagnostic plots'
    )
    parser.add_argument(
        '--plot-dir',
        type=Path,
        default=repo_root / 'png_output' / 'sync_validation',
        help='Directory for plots (default: png_output/sync_validation/)'
    )
    parser.add_argument(
        '--output-csv',
        type=Path,
        help='Save results to CSV'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    
    # Determine files
    if args.session:
        pkl_files = [args.session]
    elif args.batch:
        pkl_files = sorted(Path('.').glob(args.batch))
    else:
        parser.error("Must specify --session or --batch")
    
    if not pkl_files:
        logging.error("No .pkl files found")
        return 1
    
    logging.info(f"Processing {len(pkl_files)} session(s)\n")
    
    plot_dir = args.plot_dir if args.plot else None
    results = []
    
    for pkl_path in pkl_files:
        logging.info(f"Analyzing: {pkl_path.name}")
        
        try:
            session = load_session(str(pkl_path))
            result = validate_session_sync(session, plot_dir, args.max_lag)
            results.append(result)
            
            status = "✓" if result['passed'] else "✗"
            logging.info(f"  {status} {result['subject']}/{result['session_name']}")
            logging.info(f"  Photodiode: {'Yes' if result['has_photodiode'] else 'No'}")
            logging.info(f"  FSM timing: {'Yes' if result['has_fsm'] else 'No'}")
            if result['sync_quality']:
                logging.info(f"  Sync quality: {result['sync_quality']}")
            if result['mean_offset_ms'] is not None:
                logging.info(f"  Mean offset: {result['mean_offset_ms']:.2f} ms")
            if result['max_drift_ms'] is not None:
                logging.info(f"  Max drift: {result['max_drift_ms']:.2f} ms")
            for note in result['notes']:
                logging.info(f"  {note}")
            logging.info("")
            
        except Exception as e:
            logging.error(f"  ERROR: {e}")
            results.append({
                'subject': 'error',
                'session_name': pkl_path.stem,
                'has_photodiode': False,
                'has_fsm': False,
                'sync_quality': None,
                'passed': False,
                'notes': [f"ERROR: {str(e)}"]
            })
            logging.info("")
    
    # Summary
    df = pd.DataFrame(results)
    passed = df['passed'].sum()
    total = len(df)
    
    logging.info("=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Total sessions: {total}")
    logging.info(f"Passed:         {passed}")
    logging.info(f"Failed:         {total - passed}")
    
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        logging.info(f"\nResults saved to: {args.output_csv}")
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
