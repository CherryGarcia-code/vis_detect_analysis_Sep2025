"""Validate NI-DAQ and IMEC metadata duration consistency.

Checks that fileTimeSecs from .meta files (NI and IMEC) match within
acceptable tolerance (≤0.5 seconds) for proper multi-stream synchronization.

Usage:
    python scripts/validate_metadata_duration.py --session data/BG_046_15082025.pkl --raw-data-root "X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/Raw data"
    python scripts/validate_metadata_duration.py --batch data/BG_046_*.pkl --raw-data-root "X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/Raw data"
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from visdetect.core.session import load_session


def parse_spikeglx_meta(meta_path: Path) -> Dict[str, str]:
    """Parse SpikeGLX .meta file and return key-value pairs."""
    meta_dict = {}
    if not meta_path.exists():
        return meta_dict
    
    try:
        with open(meta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    meta_dict[key.strip()] = value.strip()
    except Exception as e:
        logging.debug(f"Failed to parse {meta_path}: {e}")
    
    return meta_dict


def find_meta_files(raw_data_root: Path, subject: str, session_name: str) -> Tuple[Optional[Path], Optional[Path]]:
    """Find the IMEC and NI .meta files for a session.
    
    Expected structure:
    raw_data_root/BG_046_DDMMYYYY/EphysNidaq/BG_046_DDMMYYYY_g0_imec0/BG_046_DDMMYYYY_g0_t0.imec0.ap.meta
    raw_data_root/BG_046_DDMMYYYY/EphysNidaq/BG_046_DDMMYYYY_g0_t0.nidq.meta
    
    Returns:
        Dict with 'imec' and 'nidq' keys containing paths or None
    """
    raw_data_root = Path(raw_data_root)
    session_dir_name = f"{subject}_{session_name}"
    session_path = raw_data_root / session_dir_name / "EphysNidaq"
    
    if not session_path.exists():
        logging.debug(f"Session path not found: {session_path}")
        return {'imec': None, 'nidq': None}
    
    # Look for IMEC meta file
    imec_pattern = f"{session_dir_name}_g0_t0.imec0.ap.meta"
    imec_meta = None
    for subdir in session_path.iterdir():
        if subdir.is_dir() and "imec0" in subdir.name:
            candidate = subdir / imec_pattern
            if candidate.exists():
                imec_meta = candidate
                break
    
    # Look for NI meta file
    ni_pattern = f"{session_dir_name}_g0_t0.nidq.meta"
    ni_meta = session_path / ni_pattern
    if not ni_meta.exists():
        ni_meta = None
    
    return {'imec': imec_meta, 'nidq': ni_meta}


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(levelname)s] %(message)s'
    )


def extract_ni_duration(session, raw_data_root: Optional[str] = None) -> Optional[float]:
    """Extract total duration from NI-DAQ .meta file or compute from events.
    
    If raw_data_root is provided, reads fileTimeSecs from .nidq.meta file.
    Otherwise falls back to computing duration from all event times.
    
    Args:
        session: Session object with subject and session_name attributes
        raw_data_root: Path to raw data directory (e.g., 'X:/public/.../BG_046/Raw data')
    """
    # Try reading from .meta file if raw_data_root provided
    if raw_data_root:
        subject = getattr(session, 'subject', None)
        session_name = getattr(session, 'session_name', None)
        
        if subject and session_name:
            meta_files = find_meta_files(raw_data_root, subject, session_name)
            if meta_files['nidq']:
                meta_data = parse_spikeglx_meta(meta_files['nidq'])
                if 'fileTimeSecs' in meta_data:
                    return float(meta_data['fileTimeSecs'])
    
    # Fallback: compute from event times
    ni_events = getattr(session, 'ni_events', None)
    if not ni_events:
        return None
    
    all_times = []
    
    for key, value in ni_events.items():
        if key == 'session_name' or isinstance(key, str) and key.startswith('_'):
            continue
        
        if value is None:
            continue
        try:
            if isinstance(value, np.ndarray) and value.size == 0:
                continue
            if not isinstance(value, (dict, np.ndarray)) and not value:
                continue
        except (ValueError, AttributeError):
            pass
            
        if isinstance(value, dict):
            if 'rise_t' in value:
                try:
                    times = np.array(value['rise_t'], dtype=float).flatten()
                    times = times[np.isfinite(times)]
                    all_times.extend(times)
                except (ValueError, TypeError):
                    pass
            if 'fall_t' in value:
                try:
                    times = np.array(value['fall_t'], dtype=float).flatten()
                    times = times[np.isfinite(times)]
                    all_times.extend(times)
                except (ValueError, TypeError):
                    pass
        else:
            try:
                times = np.array(value).flatten()
                if times.dtype.kind in ('i', 'u', 'f'):
                    all_times.extend(times)
            except Exception:
                continue
    
    if not all_times:
        return None
    
    all_times = np.array(all_times, dtype=float)
    all_times = all_times[np.isfinite(all_times)]
    
    if len(all_times) == 0:
        return None
    
    return float(np.max(all_times) - np.min(all_times))


def extract_imec_duration(session, raw_data_root: Optional[str] = None) -> Optional[float]:
    """Extract total duration from IMEC .meta file or compute from spike data.
    
    If raw_data_root is provided, reads fileTimeSecs from .imec0.ap.meta file.
    Otherwise falls back to computing duration from all spike times.
    
    Args:
        session: Session object with subject and session_name attributes
        raw_data_root: Path to raw data directory (e.g., 'X:/public/.../BG_046/Raw data')
    """
    # Try reading from .meta file if raw_data_root provided
    if raw_data_root:
        subject = getattr(session, 'subject', None)
        session_name = getattr(session, 'session_name', None)
        
        if subject and session_name:
            meta_files = find_meta_files(raw_data_root, subject, session_name)
            if meta_files['imec']:
                meta_data = parse_spikeglx_meta(meta_files['imec'])
                if 'fileTimeSecs' in meta_data:
                    return float(meta_data['fileTimeSecs'])
    
    # Fallback: compute from spike times
    clusters = getattr(session, 'clusters', None)
    if not clusters:
        return None
    
    all_spikes = []
    
    for cluster in clusters:
        spike_times = getattr(cluster, 'spike_times', None)
        if spike_times is not None and len(spike_times) > 0:
            all_spikes.extend(spike_times.flatten())
    
    if not all_spikes:
        return None
    
    all_spikes = np.array(all_spikes)
    all_spikes = all_spikes[np.isfinite(all_spikes)]
    
    if len(all_spikes) == 0:
        return None
    
    return float(np.max(all_spikes) - np.min(all_spikes))


def validate_session_duration(session, threshold: float = 0.5, raw_data_root: Optional[str] = None) -> Dict[str, any]:
    """Validate duration consistency for a single session.
    
    Args:
        session: Loaded Session object
        threshold: Maximum allowed deviation in seconds
        raw_data_root: Optional path to raw data directory for reading .meta files
        
    Returns:
        Dict with validation results
    """
    result = {
        'subject': getattr(session, 'subject', 'unknown'),
        'session_name': getattr(session, 'session_name', 'unknown'),
        'ni_duration': None,
        'imec_duration': None,
        'deviation': None,
        'passed': False,
        'notes': []
    }
    
    # Extract durations (with optional .meta file reading)
    ni_dur = extract_ni_duration(session, raw_data_root=raw_data_root)
    imec_dur = extract_imec_duration(session, raw_data_root=raw_data_root)
    
    result['ni_duration'] = ni_dur
    result['imec_duration'] = imec_dur
    
    # Validation logic
    if ni_dur is None:
        result['notes'].append("WARNING: No NI duration found")
        return result
    
    if imec_dur is None:
        result['notes'].append("WARNING: No IMEC duration found")
        return result
    
    deviation = abs(ni_dur - imec_dur)
    result['deviation'] = deviation
    
    if deviation <= threshold:
        result['passed'] = True
        result['notes'].append(f"PASS: Deviation {deviation:.3f}s ≤ {threshold}s")
    else:
        result['passed'] = False
        result['notes'].append(f"FAIL: Deviation {deviation:.3f}s > {threshold}s")
    
    return result


def format_duration(dur: Optional[float]) -> str:
    """Format duration in minutes:seconds."""
    if dur is None:
        return "N/A"
    minutes = int(dur // 60)
    seconds = dur % 60
    return f"{minutes}:{seconds:05.2f}"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate NI-DAQ and IMEC duration metadata consistency"
    )
    parser.add_argument(
        '--session',
        type=Path,
        help='Path to single session .pkl file'
    )
    parser.add_argument(
        '--batch',
        type=str,
        help='Glob pattern for batch processing (e.g., "data/BG_046_*.pkl")'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Maximum allowed deviation in seconds (default: 0.5)'
    )
    parser.add_argument(
        '--raw-data-root',
        type=Path,
        help='Path to raw data directory for reading .meta files (e.g., "X:/public/.../BG_046/Raw data")'
    )
    parser.add_argument(
        '--output-csv',
        type=Path,
        help='Save results to CSV file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    
    # Determine files to process
    if args.session:
        pkl_files = [args.session]
    elif args.batch:
        pkl_files = sorted(Path('.').glob(args.batch))
    else:
        parser.error("Must specify either --session or --batch")
    
    if not pkl_files:
        logging.error("No .pkl files found")
        return 1
    
    logging.info(f"Processing {len(pkl_files)} session(s) with threshold {args.threshold}s")
    if args.raw_data_root:
        logging.info(f"Reading .meta files from: {args.raw_data_root}")
    else:
        logging.info("Computing durations from session data (legacy method)")
    logging.info("")
    
    # Process each session
    results = []
    
    for pkl_path in pkl_files:
        logging.info(f"Loading: {pkl_path.name}")
        
        try:
            session = load_session(str(pkl_path))
            result = validate_session_duration(
                session, 
                threshold=args.threshold,
                raw_data_root=str(args.raw_data_root) if args.raw_data_root else None
            )
            results.append(result)
            
            # Display result
            status = "[PASS]" if result['passed'] else "[FAIL]"
            logging.info(f"  {status}: {result['subject']}/{result['session_name']}")
            logging.info(f"  NI duration:   {format_duration(result['ni_duration'])}")
            logging.info(f"  IMEC duration: {format_duration(result['imec_duration'])}")
            if result['deviation'] is not None:
                logging.info(f"  Deviation:     {result['deviation']:.3f}s")
            for note in result['notes']:
                logging.info(f"  {note}")
            logging.info("")
            
        except Exception as e:
            logging.error(f"  ERROR loading {pkl_path.name}: {e}")
            results.append({
                'subject': 'error',
                'session_name': pkl_path.stem,
                'ni_duration': None,
                'imec_duration': None,
                'deviation': None,
                'passed': False,
                'notes': [f"ERROR: {str(e)}"]
            })
            logging.info("")
    
    # Summary
    df = pd.DataFrame(results)
    passed_count = df['passed'].sum()
    total_count = len(df)
    
    logging.info("=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Total sessions: {total_count}")
    logging.info(f"Passed:         {passed_count}")
    logging.info(f"Failed:         {total_count - passed_count}")
    
    if total_count > 0:
        pass_rate = 100 * passed_count / total_count
        logging.info(f"Pass rate:      {pass_rate:.1f}%")
    
    # Save CSV if requested
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        logging.info(f"\nResults saved to: {args.output_csv}")
    
    # Return non-zero exit code if any failures
    return 0 if passed_count == total_count else 1


if __name__ == '__main__':
    sys.exit(main())
