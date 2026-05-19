"""Analyze video frame offsets and recommend trim values.

Parses video CSV metadata files to identify frames recorded before trial
start that should be trimmed for proper synchronization with behavioral data.

Usage:
    python scripts/analyze_video_frame_offset.py --video-csv path/to/video_metadata.csv
    python scripts/analyze_video_frame_offset.py --session-dir path/to/session/
    python scripts/analyze_video_frame_offset.py --batch-sessions data/sessions_manifest.csv
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(levelname)s] %(message)s'
    )


def find_video_metadata_files(session_dir: Path) -> List[Path]:
    """Find all video metadata CSV files in session directory.
    
    Looks for common patterns:
    - *_metadata.csv
    - *_frames.csv
    - video_*.csv
    - *_timestamps.csv
    """
    patterns = [
        '*_metadata.csv',
        '*_frames.csv', 
        'video_*.csv',
        '*_timestamps.csv',
        '*_frame_times.csv'
    ]
    
    csv_files = []
    for pattern in patterns:
        csv_files.extend(session_dir.glob(pattern))
    
    return sorted(set(csv_files))


def parse_video_metadata(csv_path: Path) -> Optional[Dict[str, any]]:
    """Parse video metadata CSV to extract frame timing and offset info.
    
    Expected columns (flexible):
    - frame_number or frame_idx or index
    - timestamp or time or frame_time
    - trial_start or is_trial or in_trial (boolean/indicator)
    
    Returns:
        Dict with analysis results or None if parsing fails
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Failed to read {csv_path}: {e}")
        return None
    
    if df.empty:
        return {'error': 'Empty CSV file'}
    
    result = {
        'csv_path': str(csv_path),
        'total_frames': len(df),
        'trim_frames': 0,
        'trial_start_frame': None,
        'pre_trial_duration': None,
        'has_timing': False,
        'has_trial_marker': False,
        'notes': []
    }
    
    # Identify relevant columns
    frame_col = None
    time_col = None
    trial_col = None
    
    # Find frame column
    for col in ['frame_number', 'frame_idx', 'frame', 'index', 'Frame']:
        if col in df.columns:
            frame_col = col
            break
    
    # Find timestamp column
    for col in ['timestamp', 'time', 'frame_time', 'Time', 'Timestamp']:
        if col in df.columns:
            time_col = col
            result['has_timing'] = True
            break
    
    # Find trial marker column
    for col in ['trial_start', 'is_trial', 'in_trial', 'trial_active', 'Trial']:
        if col in df.columns:
            trial_col = col
            result['has_trial_marker'] = True
            break
    
    # Analyze trial start point
    if trial_col:
        # Find first frame where trial is active
        trial_active = df[trial_col].astype(bool)
        if trial_active.any():
            first_trial_idx = trial_active.idxmax()
            result['trial_start_frame'] = int(first_trial_idx)
            result['trim_frames'] = int(first_trial_idx)
            
            if time_col:
                pre_trial_time = float(df.loc[first_trial_idx, time_col])
                result['pre_trial_duration'] = pre_trial_time
                result['notes'].append(
                    f"Found {result['trim_frames']} pre-trial frames "
                    f"({pre_trial_time:.2f}s)"
                )
        else:
            result['notes'].append("No trial-active frames found")
    else:
        # Heuristic: assume first N% of frames are pre-trial
        # Look for timestamp jumps or frame rate changes
        if time_col and len(df) > 10:
            times = df[time_col].values
            diffs = np.diff(times)
            
            # Find large gaps (potential trial start)
            median_diff = np.median(diffs)
            large_gaps = np.where(diffs > 2 * median_diff)[0]
            
            if len(large_gaps) > 0:
                first_gap = large_gaps[0] + 1
                result['trim_frames'] = int(first_gap)
                result['trial_start_frame'] = int(first_gap)
                result['pre_trial_duration'] = float(times[first_gap])
                result['notes'].append(
                    f"Heuristic: detected gap at frame {first_gap} "
                    f"({times[first_gap]:.2f}s)"
                )
            else:
                result['notes'].append(
                    "No clear trial start marker; manual inspection needed"
                )
    
    return result


def analyze_session_videos(session_dir: Path) -> List[Dict[str, any]]:
    """Analyze all video metadata files in a session directory."""
    csv_files = find_video_metadata_files(session_dir)
    
    if not csv_files:
        logging.warning(f"No video metadata CSV files found in {session_dir}")
        return []
    
    results = []
    for csv_file in csv_files:
        logging.debug(f"Parsing: {csv_file.name}")
        result = parse_video_metadata(csv_file)
        if result:
            results.append(result)
    
    return results


def generate_trim_recommendations(results: List[Dict[str, any]]) -> pd.DataFrame:
    """Generate summary table of trim recommendations."""
    rows = []
    
    for res in results:
        row = {
            'csv_file': Path(res['csv_path']).name,
            'total_frames': res['total_frames'],
            'trim_frames': res['trim_frames'],
            'trial_start_frame': res['trial_start_frame'],
            'pre_trial_duration_s': res['pre_trial_duration'],
            'has_timing': res['has_timing'],
            'has_trial_marker': res['has_trial_marker'],
            'notes': '; '.join(res['notes'])
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze video frame offsets and recommend trim values"
    )
    parser.add_argument(
        '--video-csv',
        type=Path,
        help='Path to single video metadata CSV file'
    )
    parser.add_argument(
        '--session-dir',
        type=Path,
        help='Session directory containing video CSV files'
    )
    parser.add_argument(
        '--batch-sessions',
        type=Path,
        help='Path to sessions manifest CSV for batch processing'
    )
    parser.add_argument(
        '--output-csv',
        type=Path,
        default=repo_root / 'table_output' / 'video_frame_trim_recommendations.csv',
        help='Output CSV path (default: table_output/video_frame_trim_recommendations.csv)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    
    all_results = []
    
    # Single CSV file
    if args.video_csv:
        logging.info(f"Analyzing single CSV: {args.video_csv}")
        result = parse_video_metadata(args.video_csv)
        if result:
            all_results.append(result)
    
    # Single session directory
    elif args.session_dir:
        logging.info(f"Analyzing session directory: {args.session_dir}")
        results = analyze_session_videos(args.session_dir)
        all_results.extend(results)
    
    # Batch processing from manifest
    elif args.batch_sessions:
        logging.info(f"Batch processing from manifest: {args.batch_sessions}")
        try:
            manifest = pd.read_csv(args.batch_sessions)
            
            # Try to find session directory column
            dir_col = None
            for col in ['session_dir', 'directory', 'path', 'folder']:
                if col in manifest.columns:
                    dir_col = col
                    break
            
            if not dir_col:
                logging.error("Manifest must contain session directory column")
                return 1
            
            for idx, row in manifest.iterrows():
                session_dir = Path(row[dir_col])
                if not session_dir.exists():
                    logging.warning(f"Session dir not found: {session_dir}")
                    continue
                
                logging.info(f"[{idx+1}/{len(manifest)}] {session_dir.name}")
                results = analyze_session_videos(session_dir)
                all_results.extend(results)
                
        except Exception as e:
            logging.error(f"Failed to process manifest: {e}")
            return 1
    
    else:
        parser.error("Must specify --video-csv, --session-dir, or --batch-sessions")
    
    if not all_results:
        logging.warning("No results to report")
        return 1
    
    # Generate summary
    logging.info("\n" + "=" * 60)
    logging.info("TRIM RECOMMENDATIONS SUMMARY")
    logging.info("=" * 60)
    
    df = generate_trim_recommendations(all_results)
    
    # Display summary statistics
    total_videos = len(df)
    videos_needing_trim = (df['trim_frames'] > 0).sum()
    
    logging.info(f"\nTotal videos analyzed: {total_videos}")
    logging.info(f"Videos needing trim: {videos_needing_trim}")
    
    if videos_needing_trim > 0:
        avg_trim = df[df['trim_frames'] > 0]['trim_frames'].mean()
        max_trim = df['trim_frames'].max()
        logging.info(f"Average trim frames: {avg_trim:.1f}")
        logging.info(f"Maximum trim frames: {max_trim}")
    
    # Display per-file details
    logging.info("\nPer-file details:")
    for idx, row in df.iterrows():
        if row['trim_frames'] > 0:
            logging.info(f"  {row['csv_file']}: trim {row['trim_frames']} frames")
            if row['notes']:
                logging.info(f"    {row['notes']}")
    
    # Save output
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    logging.info(f"\n✓ Results saved to: {args.output_csv}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
